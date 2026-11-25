//! Target satellites: where they appear, whether they are lit, how bright.
//!
//! For each instant the target's direction is found with light-time (the
//! target is seen where it was when the light left it) and then aberrated by
//! the observer's barycentric velocity. Light-time is taken in the
//! barycentric frame, which cancels the annual part of that aberration: a
//! satellite moves with the Earth, so unlike the stars it shows only the
//! aberration of the observer's motion about the geocentre.
//!
//! Sources:
//! - Light-time and aberration for near objects: Kaplan et al., USNO
//!   Circular 179 (2005), section 7.2; Klioner, AJ 125, 1580 (2003).
//! - Earth shadow (conical umbra and penumbra, fraction of the Sun's disc
//!   visible): Montenbruck & Gill, *Satellite Orbits* (Springer, 2000),
//!   section 3.4.2.
//! - Lambertian sphere brightness and phase function: Hejduk, "Specular and
//!   diffuse components in spherical satellite photometric modeling", AMOS
//!   Conference, 2011; geometric albedo p = 2A/3 for a Lambertian sphere.
//!
//! Brightness is a flux ratio to the solar irradiance at 1 au, summed over
//! light sources: direct sunlight (scaled by the lit fraction of the Sun's
//! disc), Earthshine and Moonshine (each a sunlit Lambertian sphere lighting
//! the target). Positions are geocentric GCRS, km.

use serde::{Deserialize, Serialize};

use crate::atmosphere::Atmosphere;
use crate::camera::Camera;
use crate::ephem::{sun_geocentric, AU_KM};
use crate::error::SimError;
use crate::geometry::{aberrate, C_KM_S};
use crate::orbit::{Orbit, State, R_EARTH};
use crate::time::Utc;
use crate::vec3::{add, angle, dot, norm, scale, sub, unit, V3};

/// Sun's radius, km (IAU 2015 nominal).
const R_SUN: f64 = 695_700.0;

/// How bright a target is.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Brightness {
    /// A fixed magnitude when fully sunlit (scaled by the lit fraction of
    /// the Sun's disc, zero in umbra).
    Mag(f64),
    /// A diffusely reflecting sphere.
    LambertianSphere {
        /// Diameter, metres.
        diameter_m: f64,
        /// Lambertian (Bond) albedo, 0–1.
        albedo: f64,
    },
    /// Flat panels with diffuse and specular reflection, in an attitude
    /// that may tumble.
    Facets(FacetModel),
}

/// One flat panel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Facet {
    /// Outward normal in the body frame (normalised on use).
    pub normal: [f64; 3],
    /// Area, m².
    pub area_m2: f64,
    /// Diffuse (Lambertian) albedo.
    #[serde(default)]
    pub diffuse_albedo: f64,
    /// Specular albedo.
    #[serde(default)]
    pub specular_albedo: f64,
    /// Specular sharpness (Blinn–Phong exponent): higher, sharper glints.
    #[serde(default = "default_exponent")]
    pub specular_exponent: f64,
}

/// Default [`Facet::specular_exponent`].
fn default_exponent() -> f64 {
    100.0
}

/// How the body is oriented.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum BodyAttitude {
    /// Fixed in inertial space: body → GCRS quaternion `[w, x, y, z]`.
    Inertial([f64; 4]),
    /// Spinning about an inertial axis.
    Tumbling {
        /// Spin axis, GCRS (normalised on use).
        axis: [f64; 3],
        /// Spin rate, degrees per second.
        rate_deg_s: f64,
        /// Body → GCRS quaternion at `epoch_utc`.
        #[serde(default = "identity")]
        start: [f64; 4],
        /// UTC when the body has the `start` orientation.
        epoch_utc: String,
    },
}

/// Identity quaternion `[w, x, y, z]`, the default tumbling start.
fn identity() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

/// A faceted body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FacetModel {
    /// The panels.
    pub facets: Vec<Facet>,
    /// Orientation over time.
    pub attitude: BodyAttitude,
}

/// Hamilton product `a · b` of quaternions `[w, x, y, z]`.
fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Rotate `v` by `q` (normalised first): `q · v · q*`.
fn qrot(q: [f64; 4], v: V3) -> V3 {
    let n = libm::sqrt(q.iter().map(|x| x * x).sum::<f64>());
    let q = q.map(|x| x / n);
    let p = qmul(
        qmul(q, [0.0, v[0], v[1], v[2]]),
        [q[0], -q[1], -q[2], -q[3]],
    );
    [p[1], p[2], p[3]]
}

impl FacetModel {
    /// Shortest time a glint lasts, s: the specular lobe (about 2/√k rad
    /// wide) swept at the spin rate. `None` when nothing spins.
    pub fn glint_time_s(&self) -> Option<f64> {
        let BodyAttitude::Tumbling { rate_deg_s, .. } = &self.attitude else {
            return None;
        };
        let k = self
            .facets
            .iter()
            .filter(|f| f.specular_albedo > 0.0)
            .map(|f| f.specular_exponent)
            .fold(0.0, f64::max);
        (k > 0.0 && *rate_deg_s != 0.0)
            .then(|| (2.0 / libm::sqrt(k)) / rate_deg_s.abs().to_radians())
    }

    /// Body → GCRS quaternion `[w, x, y, z]` at `t`.
    ///
    /// A tumbling body turns about its unit spin axis at `rate_deg_s` from
    /// `start` at `epoch_utc`.
    ///
    /// # Errors
    ///
    /// [`SimError::BadTime`] when a tumbling `epoch_utc` does not parse.
    pub fn orientation(&self, t: Utc) -> Result<[f64; 4], SimError> {
        match &self.attitude {
            BodyAttitude::Inertial(q) => Ok(*q),
            BodyAttitude::Tumbling {
                axis,
                rate_deg_s,
                start,
                epoch_utc,
            } => {
                let t0 =
                    Utc::parse(epoch_utc).ok_or(SimError::BadTime("target facets epoch_utc"))?;
                let ang = (rate_deg_s * t.seconds_since(t0)).to_radians();
                let a = unit(*axis);
                let (s, c) = libm::sincos(0.5 * ang);
                Ok(qmul([c, s * a[0], s * a[1], s * a[2]], *start))
            }
        }
    }

    /// Reflected flux as a fraction of the solar irradiance at the target,
    /// times area, m² (divide by range² for the flux ratio): Lambertian plus
    /// normalised Blinn–Phong per panel. `q` is the body → GCRS quaternion;
    /// `to_sun` and `to_obs` are unit vectors from the target. Panels facing
    /// away from either get nothing.
    pub fn reflectance_m2(&self, q: [f64; 4], to_sun: V3, to_obs: V3) -> f64 {
        let h = unit(add(to_sun, to_obs));
        let pi = core::f64::consts::PI;
        self.facets
            .iter()
            .map(|f| {
                let n = unit(qrot(q, f.normal));
                let (ci, cr) = (dot(n, to_sun), dot(n, to_obs));
                if ci <= 0.0 || cr <= 0.0 {
                    return 0.0;
                }
                let k = f.specular_exponent.max(0.0);
                let spec =
                    f.specular_albedo * (k + 8.0) / (8.0 * pi) * libm::pow(dot(n, h).max(0.0), k);
                f.area_m2 * (f.diffuse_albedo / pi + spec) * ci * cr
            })
            .sum()
    }
}

/// A target satellite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    /// Name, used in the truth file and by `pointing.track`.
    pub name: String,
    /// The target's orbit.
    pub orbit: Orbit,
    /// Brightness model.
    pub brightness: Brightness,
    /// Add sunlight reflected by the Earth (default false).
    #[serde(default)]
    pub earthshine: bool,
    /// Add sunlight reflected by the Moon (default false).
    #[serde(default)]
    pub moonshine: bool,
}

/// Earth's Bond albedo (NASA Earth fact sheet, 0.306).
const EARTH_ALBEDO: f64 = 0.306;
/// The Moon's Bond albedo (NASA Moon fact sheet, 0.11).
const MOON_ALBEDO: f64 = 0.11;
/// The Moon's mean radius, km (NASA Moon fact sheet).
const R_MOON: f64 = 1_737.4;

/// What the observer sees of a target at one instant.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetView {
    /// Astrometric direction: light-time corrected, not aberrated. With the
    /// header WCS this is where the target appears among the stars.
    pub astrometric: V3,
    /// Apparent direction (aberrated), as the camera sees it.
    pub apparent: V3,
    /// Distance at the time the light left, km.
    pub range_km: f64,
    /// Sun–target–observer angle, degrees.
    pub phase_deg: f64,
    /// Fraction of the Sun's disc visible from the target (0 in umbra).
    pub illumination: f64,
    /// Whether the Earth (plus a limb margin) blocks the line of sight.
    pub occulted: bool,
    /// Magnitude as seen, if lit and not occulted.
    pub mag: Option<f64>,
    /// Signal rate, electrons per second.
    pub e_per_s: f64,
    /// Share of the light that is direct sunlight (the rest is Earthshine
    /// and Moonshine).
    pub sunlight_fraction: f64,
}

/// Fraction of the Sun's disc visible from `r` (geocentric km), with the
/// Earth as the only occulting body and `r_sun` the Sun's geocentric
/// position, km.
///
/// The discs' angular radii `a` (Sun) and `b` (Earth) and their separation
/// `c` give full light, umbra, an annular case (the Earth's disc inside the
/// Sun's, only far from the Earth) or a partial overlap of two circles.
pub fn sunlit_fraction(r: V3, r_sun: V3) -> f64 {
    let to_sun = sub(r_sun, r);
    let a = libm::asin((R_SUN / norm(to_sun)).min(1.0));
    let b = libm::asin((R_EARTH / norm(r)).min(1.0));
    let c = angle(scale(r, -1.0), to_sun);
    if c >= a + b {
        return 1.0;
    }
    if c <= b - a {
        return 0.0;
    }
    if c <= a - b {
        return 1.0 - (b * b) / (a * a);
    }
    let x = (c * c + a * a - b * b) / (2.0 * c);
    let y = libm::sqrt((a * a - x * x).max(0.0));
    let area = a * a * libm::acos((x / a).clamp(-1.0, 1.0))
        + b * b * libm::acos(((c - x) / b).clamp(-1.0, 1.0))
        - c * y;
    1.0 - area / (core::f64::consts::PI * a * a)
}

/// Whether the segment from `a` to `b` passes within `radius` of the
/// Earth's centre.
pub fn earth_blocks(a: V3, b: V3, radius: f64) -> bool {
    let d = sub(b, a);
    let t = (-dot(a, d) / dot(d, d)).clamp(0.0, 1.0);
    norm([a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]]) < radius
}

/// Lambertian sphere phase function, normalised to 1 at zero phase.
pub fn lambert_phase(phase_rad: f64) -> f64 {
    let p = phase_rad;
    ((core::f64::consts::PI - p) * libm::cos(p) + libm::sin(p)) / core::f64::consts::PI
}

impl Target {
    /// View of the target from an observer with GCRS state `obs` and
    /// barycentric velocity `beta` (v/c) at `t`.
    ///
    /// Light-time is iterated twice, which reaches sub-millimetre for Earth
    /// orbits; the Sun, Moon and body attitude are taken at the emission
    /// time.
    ///
    /// Light-time is taken in the barycentric frame (USNO Circular 179,
    /// section 7.2): while the light travelled, the Earth, carrying target
    /// and observer, moved by `v_E·τ`. That shift cancels the annual part of
    /// the aberration, so a satellite shows only the aberration of the
    /// observer's motion about the geocentre, unlike the stars behind it (up
    /// to 20.6″ apart). Phase angle and reflectance use the geocentric ray;
    /// only the direction uses the barycentric light-time.
    ///
    /// `horizon`, for an observer at a ground site, is its atmosphere at
    /// `t`: the Earth then hides what is below the observed (refracted)
    /// horizon, the same test as for stars. Without it the line of sight is
    /// tested against the Earth's limb plus the camera's margin.
    ///
    /// # Errors
    ///
    /// Any error from evaluating the target's orbit, or
    /// [`SimError::BadTime`] for a bad tumbling epoch.
    pub fn view(
        &self,
        t: Utc,
        obs: &State,
        beta: V3,
        camera: &Camera,
        horizon: Option<&Atmosphere>,
    ) -> Result<TargetView, SimError> {
        let mut tau = 0.0;
        let mut r_t = self.orbit.state(t)?.position_km;
        for _ in 0..2 {
            tau = norm(sub(r_t, obs.position_km)) / C_KM_S;
            r_t = self.orbit.state(t.add_seconds(-tau))?.position_km;
        }
        let v_earth = if dot(beta, beta) > 0.0 {
            sub(scale(beta, C_KM_S), obs.velocity_km_s)
        } else {
            [0.0; 3]
        };
        let geo = sub(r_t, obs.position_km);
        let los = sub(geo, scale(v_earth, tau));
        let range = norm(geo);
        let astrometric = unit(los);
        let apparent = aberrate(astrometric, beta);

        let t_emit = t.add_seconds(-tau);
        let r_sun = sun_geocentric(t_emit.tt_centuries());
        let to_sun = sub(r_sun, r_t);
        let phase = angle(to_sun, scale(geo, -1.0));
        let illumination = sunlit_fraction(r_t, r_sun);
        let occulted = match horizon {
            Some(atm) => atm.hidden(atm.refract(unit(geo))),
            None => earth_blocks(obs.position_km, r_t, R_EARTH + camera.earth_limb_margin_km),
        };

        let to_obs = unit(scale(geo, -1.0));
        let sun_au2 = (AU_KM / norm(to_sun)).powi(2);
        let fixed_mag = matches!(self.brightness, Brightness::Mag(_));
        let sun_ratio = self.reflect_ratio(t_emit, unit(to_sun), to_obs, range)? * sun_au2;
        let mut shine = 0.0;
        if self.earthshine && !fixed_mag {
            let d = norm(r_t);
            let alpha = angle(r_sun, r_t);
            let e =
                (2.0 / 3.0) * EARTH_ALBEDO * (R_EARTH / d).powi(2) * lambert_phase(alpha) * sun_au2;
            shine += e * self.reflect_ratio(t_emit, unit(scale(r_t, -1.0)), to_obs, range)?;
        }
        if self.moonshine && !fixed_mag {
            let r_m = crate::ephem::moon_geocentric(t_emit.tt_centuries());
            let tm = sub(r_m, r_t);
            let alpha = angle(sub(r_sun, r_m), scale(tm, -1.0));
            let e = (2.0 / 3.0)
                * MOON_ALBEDO
                * (R_MOON / norm(tm)).powi(2)
                * lambert_phase(alpha)
                * sun_au2;
            shine += e * self.reflect_ratio(t_emit, unit(tm), to_obs, range)?;
        }
        let direct = if occulted {
            0.0
        } else {
            sun_ratio * illumination
        };
        let total = if occulted { 0.0 } else { direct + shine };
        let mag = match self.brightness {
            Brightness::Mag(m) => {
                (!occulted && illumination > 0.0).then(|| m - 2.5 * libm::log10(illumination))
            }
            _ => (total > 0.0).then(|| camera.sun_mag - 2.5 * libm::log10(total)),
        };
        let e_per_s = mag.map_or(0.0, |m| camera.mag_to_e_per_s(m));
        let sunlight_fraction = if total > 0.0 {
            direct / total
        } else if fixed_mag {
            1.0
        } else {
            0.0
        };
        Ok(TargetView {
            astrometric,
            apparent,
            range_km: range,
            phase_deg: phase.to_degrees(),
            illumination,
            occulted,
            mag,
            e_per_s,
            sunlight_fraction,
        })
    }

    /// Reflected flux over incident irradiance from direction `to_light`,
    /// seen from `to_obs` at `range` km (0 for a fixed magnitude).
    fn reflect_ratio(&self, t: Utc, to_light: V3, to_obs: V3, range: f64) -> Result<f64, SimError> {
        Ok(match &self.brightness {
            Brightness::Mag(_) => 0.0,
            Brightness::Facets(fm) => {
                fm.reflectance_m2(fm.orientation(t)?, to_light, to_obs) * 1e-6 / (range * range)
            }
            &Brightness::LambertianSphere { diameter_m, albedo } => {
                let radius_km = diameter_m / 2000.0;
                (2.0 / 3.0)
                    * albedo
                    * (radius_km / range).powi(2)
                    * lambert_phase(angle(to_light, to_obs))
            }
        })
    }
}

/// Parse a two- or three-line element file. Names come from the title
/// lines of a three-line file (without a leading `0 `), or the catalogue
/// number otherwise. Every target gets `brightness` and no Earthshine or
/// Moonshine.
///
/// # Errors
///
/// [`SimError::Orbit`] when the file ends mid-record or a record's lines do
/// not start with `1 ` and `2 `.
pub fn parse_tle_file(text: &str, brightness: &Brightness) -> Result<Vec<Target>, SimError> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.trim().is_empty())
        .collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let (name, l1, l2) = if lines[i].starts_with("1 ") {
            let l2 = lines
                .get(i + 1)
                .ok_or(SimError::Orbit("TLE file ends after line 1".into()))?;
            i += 2;
            (
                lines[i - 2].get(2..7).unwrap_or("").trim().to_string(),
                lines[i - 2],
                *l2,
            )
        } else {
            let l1 = lines
                .get(i + 1)
                .ok_or(SimError::Orbit("TLE file ends after a name".into()))?;
            let l2 = lines
                .get(i + 2)
                .ok_or(SimError::Orbit("TLE file ends after line 1".into()))?;
            i += 3;
            (
                lines[i - 3].trim_start_matches("0 ").trim().to_string(),
                *l1,
                *l2,
            )
        };
        if !l1.starts_with("1 ") || !l2.starts_with("2 ") {
            return Err(SimError::Orbit(format!("bad TLE pair near {name:?}")));
        }
        out.push(Target {
            name,
            orbit: Orbit::Tle([l1.to_string(), l2.to_string()]),
            brightness: brightness.clone(),
            earthshine: false,
            moonshine: false,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUN: V3 = [AU_KM, 0.0, 0.0];

    /// Sunward is lit, directly behind at LEO is umbra, behind but far
    /// off-axis is lit, and the terminator crossing passes through (0, 1).
    #[test]
    fn shadow_cases() {
        assert_eq!(sunlit_fraction([7000.0, 0.0, 0.0], SUN), 1.0);
        assert_eq!(sunlit_fraction([-7000.0, 0.0, 0.0], SUN), 0.0);
        assert_eq!(sunlit_fraction([-7000.0, 6600.0, 0.0], SUN), 1.0);
        let fr: Vec<f64> = (0..200)
            .map(|k| sunlit_fraction([-7000.0, 6300.0 + f64::from(k), 0.0], SUN))
            .collect();
        assert!(fr.windows(2).all(|w| w[1] >= w[0]));
        assert!(fr.iter().any(|&f| f > 0.05 && f < 0.95));
    }

    #[test]
    fn line_of_sight_blocking() {
        assert!(earth_blocks(
            [7000.0, 0.0, 0.0],
            [-7000.0, 0.0, 0.0],
            R_EARTH
        ));
        assert!(!earth_blocks(
            [7000.0, 0.0, 0.0],
            [7000.0, 7000.0, 0.0],
            R_EARTH
        ));
    }

    #[test]
    fn phase_function() {
        assert!((lambert_phase(0.0) - 1.0).abs() < 1e-15);
        assert!(
            (lambert_phase(core::f64::consts::FRAC_PI_2) - 1.0 / core::f64::consts::PI).abs()
                < 1e-15
        );
        assert!(lambert_phase(core::f64::consts::PI).abs() < 1e-15);
    }

    /// p = 0.1333, (R/ρ)² = 2.5e-13, full phase: m = −26.74 + 33.69 = 6.95.
    #[test]
    fn one_metre_sphere_at_1000_km() {
        let cam = Camera::default();
        let p = 2.0 / 3.0 * 0.2;
        let m = cam.sun_mag - 2.5 * libm::log10(p * (0.0005f64 / 1000.0).powi(2));
        assert!((m - 6.95).abs() < 0.01, "{m}");
    }

    /// Sun and observer along the normal give A ρ/π; a mirror-like plate
    /// glints only near the specular direction.
    #[test]
    fn flat_plate_matches_lambert_and_glints() {
        let plate = |spec: f64| FacetModel {
            facets: vec![Facet {
                normal: [1.0, 0.0, 0.0],
                area_m2: 2.0,
                diffuse_albedo: 0.3,
                specular_albedo: spec,
                specular_exponent: 400.0,
            }],
            attitude: BodyAttitude::Inertial([1.0, 0.0, 0.0, 0.0]),
        };
        let r = plate(0.0).reflectance_m2([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!((r - 2.0 * 0.3 / core::f64::consts::PI).abs() < 1e-12);
        let s = (30f64).to_radians();
        let sun = [libm::cos(s), libm::sin(s), 0.0];
        let on = plate(0.5).reflectance_m2(
            [1.0, 0.0, 0.0, 0.0],
            sun,
            [libm::cos(s), -libm::sin(s), 0.0],
        );
        let off = plate(0.5).reflectance_m2(
            [1.0, 0.0, 0.0, 0.0],
            sun,
            [libm::cos(s), libm::sin(-s + 0.6), 0.0],
        );
        assert!(on > 20.0 * off, "{on} {off}");
    }

    #[test]
    fn tumbling_repeats_with_the_spin_period() {
        let fm = FacetModel {
            facets: vec![Facet {
                normal: [1.0, 0.0, 0.0],
                area_m2: 1.0,
                diffuse_albedo: 0.2,
                specular_albedo: 0.0,
                specular_exponent: 1.0,
            }],
            attitude: BodyAttitude::Tumbling {
                axis: [0.0, 0.0, 1.0],
                rate_deg_s: 36.0,
                start: identity(),
                epoch_utc: "2026-11-17T00:00:00".into(),
            },
        };
        let t0 = Utc::parse("2026-11-17T00:00:00.5").unwrap();
        let r = |t: Utc| {
            fm.reflectance_m2(fm.orientation(t).unwrap(), [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        };
        assert!((r(t0) - r(t0.add_seconds(10.0))).abs() < 1e-12);
        assert!((r(t0) - r(t0.add_seconds(2.5))).abs() > 1e-3);
    }

    fn fixed(name: &str, r: V3) -> Target {
        let sample = |t: &str| crate::orbit::EphemerisSample {
            time_utc: t.into(),
            position_km: r,
            velocity_km_s: [0.0; 3],
        };
        Target {
            name: name.into(),
            orbit: Orbit::Ephemeris(vec![
                sample("2026-11-17T00:00:00"),
                sample("2026-11-18T00:00:00"),
            ]),
            brightness: Brightness::Mag(10.0),
            earthshine: false,
            moonshine: false,
        }
    }

    /// Planetary aberration: a target at rest with respect to the
    /// geocentre, seen from the geocentre while the Earth moves at 30 km/s
    /// across the line of sight, appears where it is; a star in the same
    /// direction moves by 20.6″.
    #[test]
    fn a_satellite_shows_no_annual_aberration() {
        let cam = Camera::default();
        let t = Utc::parse("2026-11-17T03:00:00").unwrap();
        let r = [42_164.0, 0.0, 0.0];
        let obs = State {
            position_km: [0.0; 3],
            velocity_km_s: [0.0; 3],
        };
        let beta = [0.0, 30.0 / C_KM_S, 0.0];
        let v = fixed("geo", r).view(t, &obs, beta, &cam, None).unwrap();
        let sat = angle(v.apparent, unit(r)).to_degrees() * 3600.0;
        assert!(sat < 0.01, "{sat}");
        let star = angle(aberrate(unit(r), beta), unit(r)).to_degrees() * 3600.0;
        assert!((star - 20.64).abs() < 0.01, "{star}");
    }

    /// The default observer sits at the Earth's centre: every line of sight
    /// leaves through the Earth, as before ground sites existed.
    #[test]
    fn the_geocentre_still_sees_the_earth_in_the_way() {
        let cam = Camera::default();
        let t = Utc::parse("2026-11-17T03:00:00").unwrap();
        let obs = State {
            position_km: [0.0; 3],
            velocity_km_s: [0.0; 3],
        };
        let v = fixed("geo", [42_164.0, 0.0, 0.0])
            .view(t, &obs, [0.0; 3], &cam, None)
            .unwrap();
        assert!(v.occulted);
    }

    /// From a ground site a target overhead or 10° above the horizon is
    /// seen; one below the horizon is hidden and gives no light.
    #[test]
    fn a_ground_observer_sees_what_is_above_the_horizon() {
        let cam = Camera::default();
        let t = Utc::parse("2026-11-17T03:00:00").unwrap();
        let site = Orbit::Site(crate::orbit::Site {
            latitude_deg: 32.9,
            longitude_deg: -105.5,
            height_m: 2225.0,
            dut1_s: 0.0,
            refraction: true,
            pressure_hpa: None,
            temperature_c: None,
            extinction_mag_per_airmass: 0.0,
        });
        let obs = site.state(t).unwrap();
        let Orbit::Site(ref st) = site else {
            unreachable!()
        };
        let atm = crate::atmosphere::Atmosphere::for_site(st, t);
        let up = atm.up;
        let east = unit(crate::vec3::cross([0.0, 0.0, 1.0], up));
        for (dir, hidden) in [
            (up, false),
            (
                unit(add(
                    scale(east, libm::cos(0.1745)),
                    scale(up, libm::sin(0.1745)),
                )),
                false,
            ),
            (unit(add(scale(east, 1.0), scale(up, -0.2))), true),
        ] {
            let v = fixed("t", add(obs.position_km, scale(dir, 20_000.0)))
                .view(t, &obs, [0.0; 3], &cam, Some(&atm))
                .unwrap();
            assert_eq!(v.occulted, hidden, "{dir:?}");
            if hidden {
                assert_eq!(v.e_per_s, 0.0);
            }
        }
    }

    /// Target 700 km above the sub-solar point, observer off to the side
    /// seeing both the target and the lit Earth.
    #[test]
    fn earthshine_adds_light_from_the_lit_earth() {
        let cam = Camera::default();
        let t = Utc::parse("2026-11-17T03:00:00").unwrap();
        let sun = unit(sun_geocentric(t.tt_centuries()));
        let side = unit(crate::vec3::cross(sun, [0.0, 0.0, 1.0]));
        let r_t = scale(sun, R_EARTH + 700.0);
        let obs = State {
            position_km: crate::vec3::add(r_t, scale(side, 3_000.0)),
            velocity_km_s: [0.0; 3],
        };
        let mk = |es: bool| Target {
            name: "s".into(),
            orbit: Orbit::StateVector(crate::orbit::StateVector {
                epoch_utc: "2026-11-17T03:00:00".into(),
                position_km: r_t,
                velocity_km_s: [0.0; 3],
                j2: false,
                frame: crate::orbit::StateFrame::default(),
            }),
            brightness: Brightness::LambertianSphere {
                diameter_m: 2.0,
                albedo: 0.3,
            },
            earthshine: es,
            moonshine: false,
        };
        let off = mk(false).view(t, &obs, [0.0; 3], &cam, None).unwrap();
        let on = mk(true).view(t, &obs, [0.0; 3], &cam, None).unwrap();
        assert_eq!(off.sunlight_fraction, 1.0);
        assert!(
            on.e_per_s > off.e_per_s && on.sunlight_fraction < 0.95,
            "{} {}",
            on.sunlight_fraction,
            on.e_per_s / off.e_per_s
        );
    }

    #[test]
    fn tle_file_formats() {
        let b = Brightness::Mag(10.0);
        let l1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753";
        let l2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667";
        let three = format!("VANGUARD 1\n{l1}\n{l2}\n");
        assert_eq!(parse_tle_file(&three, &b).unwrap()[0].name, "VANGUARD 1");
        let two = format!("{l1}\n{l2}\n{l1}\n{l2}\n");
        let t = parse_tle_file(&two, &b).unwrap();
        assert_eq!((t.len(), t[0].name.as_str()), (2, "00005"));
    }
}
