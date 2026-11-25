//! Stray light from the Earth, Sun and Moon, and zodiacal light.
//!
//! Each stray-light source adds a background whose level is a table against
//! the source's angle from the boresight (for the Earth: the angle to the
//! nearest limb), with a gradient across the frame. The gradient's
//! direction is set relative to the source's direction in the image: 0°
//! means brighter towards it; 180° means brighter away from it, as when
//! scattered or ghost light from an off-axis source lands on the far side
//! of the focal plane. Tables come from measurements of the instrument.
//!
//! Zodiacal light is a table of surface brightness against elongation from
//! the Sun (helioecliptic longitude difference) and ecliptic latitude, in V
//! mag/arcsec²; the profile supplies it (e.g. from Leinert et al., A&AS
//! 127, 1, 1998, table 17).
//!
//! Moon phase: illuminated fraction k = (1 + cos i)/2 for phase angle i
//! (Meeus, *Astronomical Algorithms*, 2nd ed., eq. 48.1).

use serde::{Deserialize, Serialize};

use crate::camera::Camera;
use crate::ephem::{moon_geocentric, sun_geocentric, EPS_J2000_DEG};
use crate::geometry::Geometry;
use crate::orbit::R_EARTH;
use crate::vec3::{add, angle, dot, norm, radec_to_unit, scale, sub, unit, unit_to_radec, V3};

/// A stray-light source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Body {
    /// The Earth (angle measured to the limb).
    Earth,
    /// The Sun.
    Sun,
    /// The Moon.
    Moon,
}

/// Stray light from one body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StraySource {
    /// Which body.
    pub body: Body,
    /// Angles, degrees, increasing (Earth: from the limb; others: from the
    /// body's centre).
    pub angle_deg: Vec<f64>,
    /// Background at the image centre at those angles, e⁻/pixel/s. Log-
    /// interpolated; held at the first value below the table and zero
    /// beyond it.
    pub e_per_s: Vec<f64>,
    /// Gradient: ratio of the background at the bright edge to the image
    /// centre, minus one, across half the frame width (default 0).
    #[serde(default)]
    pub gradient: f64,
    /// Direction of increasing background relative to the body's direction
    /// in the image, degrees (0: towards it; 180: away from it).
    #[serde(default)]
    pub toward_deg: f64,
    /// Earth only: factor when the limb point nearest the boresight is in
    /// darkness (default 1).
    #[serde(default)]
    pub dark_limb_factor: Option<f64>,
    /// Moon only: scale by the illuminated fraction (default true).
    #[serde(default)]
    pub phase_scaled: Option<bool>,
}

/// Stray light added to one frame from one source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrayTruth {
    /// Which body.
    pub body: Body,
    /// Angle used, degrees.
    pub angle_deg: f64,
    /// Level at the image centre, e⁻/pixel/s.
    pub e_per_s: f64,
    /// Direction of increasing background in the image (unit `[dx, dy]`).
    pub direction: Option<[f64; 2]>,
    /// Earth: whether the nearest limb point is sunlit. Moon: lit fraction.
    pub lit: f64,
}

impl StraySource {
    /// Checks the table.
    ///
    /// # Errors
    ///
    /// Fails if the table is empty, `angle_deg` does not strictly increase,
    /// its length differs from `e_per_s`, or a level is negative or NaN.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.angle_deg.len();
        if n == 0 || n != self.e_per_s.len() || self.angle_deg.windows(2).any(|w| w[1] <= w[0]) {
            return Err("stray_light: angle_deg must increase and match e_per_s".into());
        }
        if self.e_per_s.iter().any(|v| v.is_nan() || *v < 0.0) {
            return Err("stray_light: e_per_s must be ≥ 0".into());
        }
        Ok(())
    }

    /// Background at angle `a` (degrees), e⁻/pixel/s: log-linear between
    /// table points, or linear where either neighbour is zero (the log is
    /// undefined there).
    fn level(&self, a: f64) -> f64 {
        let (x, y) = (&self.angle_deg, &self.e_per_s);
        if a <= x[0] {
            return y[0];
        }
        if a > x[x.len() - 1] {
            return 0.0;
        }
        let i = x.partition_point(|&v| v < a).clamp(1, x.len() - 1);
        let u = (a - x[i - 1]) / (x[i] - x[i - 1]);
        let (p, q) = (y[i - 1].max(1e-30), y[i].max(1e-30));
        if y[i - 1] == 0.0 || y[i] == 0.0 {
            y[i - 1] + u * (y[i] - y[i - 1])
        } else {
            libm::exp(libm::log(p) + u * (libm::log(q) - libm::log(p)))
        }
    }
}

/// Direction of `v` (a direction on the sky) in the image at its centre, as
/// a unit pixel vector; `None` when `v` is along or opposite the boresight.
fn image_direction(g: &Geometry, v: V3) -> Option<[f64; 2]> {
    let b = radec_to_unit(g.pointing.ra_deg, g.pointing.dec_deg);
    let t = sub(v, scale(b, dot(v, b)));
    if norm(t) < 1e-9 {
        return None;
    }
    let q = unit(add(b, scale(unit(t), 1e-3)));
    let (ra, dec) = unit_to_radec(q);
    let (cx, cy) = (g.camera_wcs.crpix0[0], g.camera_wcs.crpix0[1]);
    let (x, y) = g.camera_wcs.world_to_pixel(ra, dec)?;
    let (dx, dy) = (x - cx, y - cy);
    let n = libm::hypot(dx, dy);
    (n > 0.0).then(|| [dx / n, dy / n])
}

/// Adds stray light to a light image (electrons over the exposure) and
/// returns what each source added.
///
/// Needs a geometry with timing and an observer; without them nothing is
/// added. The Earth's limb is lit or dark according to the limb point
/// nearest the boresight: from the Earth's centre direction, turned towards
/// the boresight by the Earth's angular radius. Bodies are evaluated at
/// mid-exposure.
pub fn add_stray_light(
    img: &mut [f64],
    camera: &Camera,
    g: &Geometry,
    sources: &[StraySource],
) -> Vec<StrayTruth> {
    let (Some(tm), Some(obs)) = (g.timing, g.observer) else {
        return Vec::new();
    };
    let t = tm.mid.tt_centuries();
    let r = obs.position_km;
    let b = radec_to_unit(g.pointing.ra_deg, g.pointing.dec_deg);
    let sun = sun_geocentric(t);
    let mut out = Vec::new();
    for s in sources {
        let (dir_body, ang, factor, lit) = match s.body {
            Body::Earth => {
                let rn = norm(r);
                if rn <= R_EARTH {
                    continue;
                }
                let e = scale(r, -1.0 / rn);
                let half = libm::asin(R_EARTH / rn);
                let ang = (angle(e, b) - half).to_degrees();
                let ax = sub(b, scale(e, dot(e, b)));
                let lit = if norm(ax) > 1e-12 {
                    let ax = unit(ax);
                    let ld = add(scale(e, libm::cos(half)), scale(ax, libm::sin(half)));
                    let p = add(r, scale(ld, libm::sqrt(rn * rn - R_EARTH * R_EARTH)));
                    if dot(p, sun) > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    1.0
                };
                let f = if lit > 0.0 {
                    1.0
                } else {
                    s.dark_limb_factor.unwrap_or(1.0)
                };
                (e, ang, f, lit)
            }
            Body::Sun => {
                let d = unit(sub(sun, r));
                (d, angle(d, b).to_degrees(), 1.0, 1.0)
            }
            Body::Moon => {
                let m = moon_geocentric(t);
                let d = unit(sub(m, r));
                let phase = angle(sub(sun, m), sub(r, m));
                let k = 0.5 * (1.0 + libm::cos(phase));
                let f = if s.phase_scaled.unwrap_or(true) {
                    k
                } else {
                    1.0
                };
                (d, angle(d, b).to_degrees(), f, k)
            }
        };
        let level = s.level(ang) * factor;
        let dir = image_direction(g, dir_body).map(|[x, y]| {
            let (sn, cs) = libm::sincos(s.toward_deg.to_radians());
            [cs * x - sn * y, sn * x + cs * y]
        });
        if level > 0.0 {
            let (w, h) = (camera.width, camera.height);
            let (cx, cy) = (g.camera_wcs.crpix0[0], g.camera_wcs.crpix0[1]);
            let k = libm::log(1.0 + s.gradient.max(-0.99)) / (0.5 * w as f64);
            for y in 0..h {
                for x in 0..w {
                    let along =
                        dir.map_or(0.0, |[ux, uy]| (x as f64 - cx) * ux + (y as f64 - cy) * uy);
                    img[y * w + x] += level * libm::exp(k * along) * camera.exposure_s;
                }
            }
        }
        out.push(StrayTruth {
            body: s.body,
            angle_deg: ang,
            e_per_s: level,
            direction: dir,
            lit,
        });
    }
    out
}

/// Zodiacal light table: V mag/arcsec² against elongation from the Sun
/// (|λ − λ☉|, 0–180°) and ecliptic latitude (|β|, 0–90°).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZodiTable {
    /// |λ − λ☉| grid, degrees, increasing.
    pub elongation_deg: Vec<f64>,
    /// |β| grid, degrees, increasing.
    pub latitude_deg: Vec<f64>,
    /// Surface brightness, V mag/arcsec²: one row per elongation, one value
    /// per latitude.
    pub v_mag_arcsec2: Vec<Vec<f64>>,
}

impl ZodiTable {
    /// Checks the shape.
    ///
    /// # Errors
    ///
    /// Fails if either grid has fewer than two values or does not strictly
    /// increase, or if `v_mag_arcsec2` is not one row per elongation of one
    /// value per latitude.
    pub fn validate(&self) -> Result<(), String> {
        let incr = |v: &[f64]| v.len() >= 2 && v.windows(2).all(|w| w[1] > w[0]);
        if !incr(&self.elongation_deg) || !incr(&self.latitude_deg) {
            return Err("zodiacal table: grids must increase, at least two each".into());
        }
        if self.v_mag_arcsec2.len() != self.elongation_deg.len()
            || self
                .v_mag_arcsec2
                .iter()
                .any(|r| r.len() != self.latitude_deg.len())
        {
            return Err("zodiacal table: values must be (elongations) × (latitudes)".into());
        }
        Ok(())
    }

    /// Surface brightness at elongation `elong` and latitude `lat`
    /// (degrees), V mag/arcsec²: bilinear, clamped to the grid.
    ///
    /// # Panics
    ///
    /// Panics if the table fails [`ZodiTable::validate`].
    pub fn at(&self, elong: f64, lat: f64) -> f64 {
        let idx = |g: &[f64], v: f64| {
            let v = v.clamp(g[0], g[g.len() - 1]);
            let i = g.partition_point(|&x| x < v).clamp(1, g.len() - 1);
            (i - 1, (v - g[i - 1]) / (g[i] - g[i - 1]))
        };
        let (i, u) = idx(&self.elongation_deg, elong);
        let (j, w) = idx(&self.latitude_deg, lat);
        let m = &self.v_mag_arcsec2;
        (1.0 - u) * ((1.0 - w) * m[i][j] + w * m[i][j + 1])
            + u * ((1.0 - w) * m[i + 1][j] + w * m[i + 1][j + 1])
    }
}

/// Helioecliptic longitude difference |λ − λ☉| and ecliptic latitude |β|
/// of the boresight at mid-exposure, degrees (J2000 ecliptic); `None`
/// without timing or an observer.
pub fn ecliptic_geometry(g: &Geometry) -> Option<(f64, f64)> {
    let tm = g.timing?;
    let obs = g.observer?;
    let eps = EPS_J2000_DEG.to_radians();
    let to_ecl = |v: V3| {
        let (s, c) = libm::sincos(eps);
        [v[0], c * v[1] + s * v[2], -s * v[1] + c * v[2]]
    };
    let b = to_ecl(radec_to_unit(g.pointing.ra_deg, g.pointing.dec_deg));
    let sd = to_ecl(unit(sub(
        sun_geocentric(tm.mid.tt_centuries()),
        obs.position_km,
    )));
    let lam = libm::atan2(b[1], b[0]);
    let beta = libm::asin(b[2].clamp(-1.0, 1.0));
    let lam_s = libm::atan2(sd[1], sd[0]);
    let mut d = (lam - lam_s).to_degrees().rem_euclid(360.0);
    if d > 180.0 {
        d = 360.0 - d;
    }
    Some((d, beta.to_degrees().abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_table_interpolates_in_log() {
        let s = StraySource {
            body: Body::Earth,
            angle_deg: vec![10.0, 12.0, 14.0],
            e_per_s: vec![10_000.0, 100.0, 0.0],
            gradient: 0.0,
            toward_deg: 0.0,
            dark_limb_factor: None,
            phase_scaled: None,
        };
        s.validate().unwrap();
        assert_eq!(s.level(5.0), 10_000.0);
        assert!((s.level(11.0) - 1_000.0).abs() < 1e-6);
        assert!((s.level(13.0) - 50.0).abs() < 1e-9);
        assert_eq!(s.level(20.0), 0.0);
    }

    #[test]
    fn zodiacal_table_bilinear() {
        let z = ZodiTable {
            elongation_deg: vec![90.0, 180.0],
            latitude_deg: vec![0.0, 90.0],
            v_mag_arcsec2: vec![vec![22.0, 23.0], vec![22.4, 23.0]],
        };
        z.validate().unwrap();
        assert!((z.at(135.0, 45.0) - 22.6).abs() < 1e-12);
        assert_eq!(z.at(0.0, 0.0), 22.0);
    }
}
