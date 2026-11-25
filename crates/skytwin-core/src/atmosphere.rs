//! The atmosphere above a ground site: refraction and extinction.
//!
//! Refraction lifts every source towards the zenith by an angle that
//! depends on its altitude, the pressure and the temperature. Across a
//! narrow field the change of that angle compresses the image along the
//! vertical by up to a few arcseconds, which a star-fitted WCS absorbs as
//! scale and shear. Extinction dims every source by a fixed number of
//! magnitudes per airmass.
//!
//! Limits: the atmosphere (the local vertical, so the refraction of each
//! direction) is taken at mid-exposure, and extinction at the boresight. A
//! star is drawn for the whole exposure or not at all, by whether it is
//! above the horizon at mid-exposure: one rising or setting during the
//! exposure is not cut off part way.
//! Near the horizon in long exposures refraction changes enough to move
//! stars (about 15 px in 60 s at 5° altitude and 1.5″/px); above 30° and
//! for exposures of minutes it is under 0.1″.
//!
//! Sources:
//! - Refraction from the true altitude: Sæmundsson, *Sky and Telescope*
//!   72, 70 (1986), as given by Meeus, *Astronomical Algorithms*, 2nd ed.,
//!   eq. 16.4, with the pressure and temperature factor below eq. 16.4.
//! - Airmass: Kasten & Young, *Applied Optics* 28, 4735 (1989), eq. 3.
//! - Default pressure and temperature by height: the International
//!   Standard Atmosphere troposphere (ISO 2533:1975; *U.S. Standard
//!   Atmosphere*, 1976): P = 1013.25 (1 − 2.25577e-5 h)^5.25588 hPa and
//!   T = 15 − 0.0065 h °C, h in metres.

use serde::{Deserialize, Serialize};

use crate::orbit::Site;
use crate::time::Utc;
use crate::vec3::{add, dot, scale, sub, unit, V3};

/// The atmosphere along every line of sight from one site at one time.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Atmosphere {
    /// Local vertical (ellipsoid normal) at mid-exposure, GCRS unit vector.
    pub up: V3,
    /// Pressure at the site, hPa.
    pub pressure_hpa: f64,
    /// Air temperature at the site, °C.
    pub temperature_c: f64,
    /// Whether refraction is applied.
    pub refraction: bool,
    /// Extinction, magnitudes per airmass, in the camera band.
    pub extinction_mag_per_airmass: f64,
    /// Altitude of the sea-level horizon seen from the site's height,
    /// degrees (negative: the dip, about 1.5° at 2,200 m; terrain ignored).
    #[serde(default)]
    pub horizon_deg: f64,
}

/// Dip of the geometric horizon seen from height `h_m` above a sphere of
/// the Earth's equatorial radius, degrees (negative).
pub fn horizon_dip_deg(h_m: f64) -> f64 {
    let r = crate::orbit::R_EARTH;
    -libm::acos(r / (r + h_m.max(0.0) / 1000.0)).to_degrees()
}

/// Pressure of the International Standard Atmosphere at height `h_m`, hPa.
pub fn standard_pressure_hpa(h_m: f64) -> f64 {
    1013.25 * libm::pow((1.0 - 2.255_77e-5 * h_m).max(0.0), 5.255_88)
}

/// Temperature of the International Standard Atmosphere at height `h_m`, °C.
pub fn standard_temperature_c(h_m: f64) -> f64 {
    15.0 - 0.0065 * h_m
}

/// Sæmundsson's refraction for a true altitude `h` (degrees), arcminutes,
/// at 1010 hPa and 10 °C, before the zenith offset is removed.
///
/// Altitudes below −1° count as −1°: the formula diverges near −1.9°, and
/// nothing below the horizon is imaged.
fn saemundsson_arcmin(h: f64) -> f64 {
    let h = h.max(-1.0);
    1.02 / libm::tan((h + 10.3 / (h + 5.11)).to_radians())
}

/// Airmass at an apparent altitude (degrees), Kasten & Young (1989).
/// Altitudes below the horizon count as the horizon.
pub fn airmass(apparent_alt_deg: f64) -> f64 {
    let h = apparent_alt_deg.max(0.0);
    1.0 / (libm::sin(h.to_radians()) + 0.505_72 * libm::pow(h + 6.079_95, -1.636_4))
}

impl Atmosphere {
    /// The atmosphere at `site` at time `t` (the vertical turns with the
    /// Earth; frames take it at mid-exposure).
    pub fn for_site(site: &Site, t: Utc) -> Self {
        let up_ecef = crate::frames::geodetic_up(site.latitude_deg, site.longitude_deg);
        let (up, _) = crate::frames::ecef_to_gcrs_ut1(up_ecef, [0.0; 3], t, site.dut1_s);
        Self {
            up: unit(up),
            pressure_hpa: site
                .pressure_hpa
                .unwrap_or_else(|| standard_pressure_hpa(site.height_m)),
            temperature_c: site
                .temperature_c
                .unwrap_or_else(|| standard_temperature_c(site.height_m)),
            refraction: site.refraction,
            extinction_mag_per_airmass: site.extinction_mag_per_airmass,
            horizon_deg: horizon_dip_deg(site.height_m),
        }
    }

    /// Whether an observed direction is hidden by the Earth: below the
    /// horizon seen from the site's height.
    pub fn hidden(&self, v: V3) -> bool {
        self.altitude_deg(v) < self.horizon_deg
    }

    /// Altitude of a direction above the horizon, degrees.
    pub fn altitude_deg(&self, u: V3) -> f64 {
        libm::asin(dot(u, self.up).clamp(-1.0, 1.0)).to_degrees()
    }

    /// Refraction for a true altitude (degrees), degrees; zero at the
    /// zenith and without refraction.
    pub fn refraction_deg(&self, true_alt_deg: f64) -> f64 {
        if !self.refraction {
            return 0.0;
        }
        let r = saemundsson_arcmin(true_alt_deg) - saemundsson_arcmin(90.0);
        let factor = (self.pressure_hpa / 1010.0) * (283.0 / (273.0 + self.temperature_c));
        (r * factor / 60.0).max(0.0)
    }

    /// The observed direction of a source whose direction above the
    /// atmosphere is `u`: lifted towards the zenith by the refraction.
    pub fn refract(&self, u: V3) -> V3 {
        let r = self.refraction_deg(self.altitude_deg(u));
        tilt(u, self.up, r.to_radians())
    }

    /// The direction above the atmosphere of a source observed at `v`
    /// (inverse of [`Atmosphere::refract`], by fixed-point iteration on the
    /// altitude).
    pub fn unrefract(&self, v: V3) -> V3 {
        if !self.refraction {
            return v;
        }
        let h_obs = self.altitude_deg(v);
        let mut h = h_obs - self.refraction_deg(h_obs);
        for _ in 0..12 {
            h = h_obs - self.refraction_deg(h);
        }
        tilt(v, self.up, (h - h_obs).to_radians())
    }

    /// Airmass along an observed direction.
    pub fn airmass(&self, v: V3) -> f64 {
        airmass(self.altitude_deg(v))
    }

    /// Fraction of a source's light that reaches the telescope along an
    /// observed direction.
    pub fn transmission(&self, v: V3) -> f64 {
        libm::pow(
            10.0,
            -0.4 * self.extinction_mag_per_airmass * self.airmass(v),
        )
    }
}

/// Turn `u` by `angle` (radians) in the vertical plane through `up`:
/// positive towards the zenith. A direction at the zenith or nadir has no
/// such plane and is returned unchanged.
fn tilt(u: V3, up: V3, angle: f64) -> V3 {
    if angle == 0.0 {
        return u;
    }
    let towards = sub(up, scale(u, dot(u, up)));
    let n = crate::vec3::norm(towards);
    if n < 1e-12 {
        return u;
    }
    let (s, c) = libm::sincos(angle);
    unit(add(scale(u, c), scale(towards, s / n)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::{angle, radec_to_unit};

    fn atm(p: f64, t: f64) -> Atmosphere {
        Atmosphere {
            up: [0.0, 0.0, 1.0],
            pressure_hpa: p,
            temperature_c: t,
            refraction: true,
            extinction_mag_per_airmass: 0.2,
            horizon_deg: 0.0,
        }
    }

    /// Azimuth along +x, altitude `h`, with up = +z.
    fn at_alt(h: f64) -> V3 {
        radec_to_unit(0.0, h)
    }

    /// About one arcminute at 45° and none at the zenith; about 29′ for a
    /// source whose true altitude is 0° (Meeus ch. 16). Proportional to
    /// pressure, inversely to absolute temperature.
    #[test]
    fn refraction_values() {
        let a = atm(1010.0, 10.0);
        let r45 = a.refraction_deg(45.0) * 3600.0;
        assert!((57.0..62.0).contains(&r45), "{r45}");
        assert_eq!(a.refraction_deg(90.0), 0.0);
        let r0 = a.refraction_deg(0.0) * 60.0;
        assert!((28.0..30.0).contains(&r0), "{r0}");
        let r2 = atm(2020.0, 10.0).refraction_deg(30.0) / a.refraction_deg(30.0);
        assert!((r2 - 2.0).abs() < 1e-12);
        let rt = atm(1010.0, 0.0).refraction_deg(30.0) / a.refraction_deg(30.0);
        assert!((rt - 283.0 / 273.0).abs() < 1e-12);
    }

    /// Refraction lifts a source by the refraction at its altitude, keeps its
    /// azimuth (still in the x–z plane) and inverts.
    #[test]
    fn refract_lifts_and_round_trips() {
        let a = atm(780.0, 5.0);
        for h in [3.0, 10.0, 30.0, 60.0, 89.5] {
            let u = at_alt(h);
            let v = a.refract(u);
            let lift = a.altitude_deg(v) - h;
            assert!((lift - a.refraction_deg(h)).abs() < 1e-12, "{h} {lift}");
            assert!(v[1].abs() < 1e-15);
            let back = a.unrefract(v);
            assert!(angle(back, u).to_degrees() * 3600.0 < 1e-6, "{h}");
        }
    }

    /// Plane-parallel 1/sin h holds well at 30° (2.0). At the horizon the
    /// formula gives 37.92 (Kasten & Young tabulate 38.09 for their exact
    /// integration).
    #[test]
    fn airmass_values() {
        assert!((airmass(90.0) - 1.0).abs() < 1e-3);
        assert!((airmass(30.0) - 2.0).abs() < 0.01, "{}", airmass(30.0));
        assert!((airmass(0.0) - 37.92).abs() < 0.01, "{}", airmass(0.0));
        let a = atm(1010.0, 10.0);
        let t = a.transmission(at_alt(90.0));
        assert!((t - libm::pow(10.0, -0.08)).abs() < 1e-3, "{t}");
    }

    /// About 1.5° from 2,200 m; none at sea level.
    #[test]
    fn horizon_dip() {
        let d = horizon_dip_deg(2200.0);
        assert!((d + 1.505).abs() < 0.01, "{d}");
        assert_eq!(horizon_dip_deg(0.0), 0.0);
    }

    /// About 774 hPa and 0.5 °C at 2,225 m.
    #[test]
    fn standard_atmosphere() {
        assert!((standard_pressure_hpa(0.0) - 1013.25).abs() < 1e-9);
        assert!(
            (standard_pressure_hpa(2225.0) - 774.0).abs() < 5.0,
            "{}",
            standard_pressure_hpa(2225.0)
        );
        assert!((standard_temperature_c(2225.0) - 0.54).abs() < 0.01);
    }
}
