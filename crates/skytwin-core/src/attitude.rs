//! Camera attitude: RA/Dec/roll or a quaternion.
//!
//! Camera frame: +z along the boresight, +x along increasing column, +y
//! along *decreasing* row. With FITS rows increasing upwards, image +y is
//! camera −y, and x × y = z as a right-handed frame needs.
//!
//! The quaternion `[w, x, y, z]` (scalar first) rotates camera-frame vectors
//! into GCRS: `v_gcrs = q · v_cam · q*`.
//!
//! Source: quaternion ↔ matrix and Shepperd's method, Markley & Crassidis,
//! *Fundamentals of Spacecraft Attitude Determination and Control*
//! (Springer, 2014), sections 2.7 and 2.9.3.

use serde::{Deserialize, Serialize};

use crate::error::SimError;
use crate::vec3::{dot, norm, radec_to_unit, scale, unit_to_radec, M3, V3};
use crate::wcs::Pointing;

/// Pointing as written in a scene file: `ra_deg`, `dec_deg` and optional
/// `roll_deg`, or a `quaternion`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PointingSpec {
    /// Boresight right ascension, degrees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ra_deg: Option<f64>,
    /// Boresight declination, degrees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dec_deg: Option<f64>,
    /// Roll: position angle of image +y, east of north, degrees (default 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roll_deg: Option<f64>,
    /// Camera → GCRS quaternion `[w, x, y, z]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quaternion: Option<[f64; 4]>,
    /// Name of a target to follow (with `roll_deg`): stars trail, the
    /// target stays at the image centre.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track: Option<String>,
}

impl From<Pointing> for PointingSpec {
    fn from(p: Pointing) -> Self {
        Self {
            ra_deg: Some(p.ra_deg),
            dec_deg: Some(p.dec_deg),
            roll_deg: Some(p.roll_deg),
            quaternion: None,
            track: None,
        }
    }
}

impl PointingSpec {
    /// The pointing this describes, with RA wrapped into `[0, 360)` and roll
    /// defaulting to 0.
    ///
    /// # Errors
    ///
    /// [`SimError::BadPointing`] if `dec_deg` is outside ±90°, if neither or
    /// both of the RA/Dec and quaternion forms are given, if `roll_deg` is
    /// combined with a quaternion, or if the quaternion has zero length.
    pub fn resolve(&self) -> Result<Pointing, SimError> {
        match (self.ra_deg, self.dec_deg, self.quaternion) {
            (Some(ra), Some(dec), None) => {
                if dec.abs() > 90.0 {
                    return Err(SimError::BadPointing("dec_deg outside ±90"));
                }
                Ok(Pointing {
                    ra_deg: ra.rem_euclid(360.0),
                    dec_deg: dec,
                    roll_deg: self.roll_deg.unwrap_or(0.0),
                })
            }
            (None, None, Some(q)) if self.roll_deg.is_none() => quaternion_to_pointing(q),
            _ => Err(SimError::BadPointing(
                "give ra_deg and dec_deg (and roll_deg), or quaternion",
            )),
        }
    }
}

/// East and north unit vectors at a sky direction.
pub fn east_north(ra_deg: f64, dec_deg: f64) -> (V3, V3) {
    let (sa, ca) = libm::sincos(ra_deg.to_radians());
    let (sd, cd) = libm::sincos(dec_deg.to_radians());
    ([-sa, ca, 0.0], [-sd * ca, -sd * sa, cd])
}

/// Camera → GCRS rotation matrix (columns are camera axes in GCRS).
///
/// Image +y lies at position angle `roll_deg` east of north and image +x is
/// `(−cos r, sin r)` in the local (east, north) basis. The columns are then
/// camera x = image x, camera y = −image y, camera z = boresight.
pub fn pointing_matrix(p: &Pointing) -> M3 {
    let b = radec_to_unit(p.ra_deg, p.dec_deg);
    let (e, n) = east_north(p.ra_deg, p.dec_deg);
    let (sr, cr) = libm::sincos(p.roll_deg.to_radians());
    let yi: V3 = std::array::from_fn(|k| n[k] * cr + e[k] * sr);
    let xi: V3 = std::array::from_fn(|k| -e[k] * cr + n[k] * sr);
    std::array::from_fn(|r| [xi[r], -yi[r], b[r]])
}

/// Rotation matrix of the quaternion `q` `[w, x, y, z]` after normalising
/// it; zero or NaN length is a [`SimError::BadPointing`].
fn quat_matrix(q: [f64; 4]) -> Result<M3, SimError> {
    let n = libm::sqrt(q.iter().map(|v| v * v).sum::<f64>());
    if n.is_nan() || n <= 1e-9 {
        return Err(SimError::BadPointing("quaternion has zero length"));
    }
    let [w, x, y, z] = q.map(|v| v / n);
    Ok([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])
}

/// RA, Dec and roll (degrees) of a camera → GCRS quaternion `[w, x, y, z]`.
///
/// The quaternion need not be normalised.
///
/// # Errors
///
/// [`SimError::BadPointing`] if the quaternion has zero (or NaN) length.
pub fn quaternion_to_pointing(q: [f64; 4]) -> Result<Pointing, SimError> {
    let m = quat_matrix(q)?;
    let col = |c: usize| -> V3 { [m[0][c], m[1][c], m[2][c]] };
    let b = col(2);
    let yi = scale(col(1), -1.0);
    let (ra, dec) = unit_to_radec(b);
    let (e, n) = east_north(ra, dec);
    let roll = libm::atan2(dot(yi, e), dot(yi, n)).to_degrees();
    Ok(Pointing {
        ra_deg: ra,
        dec_deg: dec,
        roll_deg: roll,
    })
}

/// Camera → GCRS quaternion `[w, x, y, z]` (`w ≥ 0`, unit length) for a
/// pointing.
///
/// Uses Shepperd's method: the component with the largest of 4w², 4x², 4y²,
/// 4z² is extracted first, which keeps the divisions well conditioned.
pub fn pointing_to_quaternion(p: &Pointing) -> [f64; 4] {
    let m = pointing_matrix(p);
    let tr = m[0][0] + m[1][1] + m[2][2];
    let cands = [tr, m[0][0], m[1][1], m[2][2]];
    let i = (0..4)
        .max_by(|&a, &b| cands[a].total_cmp(&cands[b]))
        .unwrap_or(0);
    let q = match i {
        0 => {
            let w = 0.5 * libm::sqrt(1.0 + tr);
            [
                w,
                (m[2][1] - m[1][2]) / (4.0 * w),
                (m[0][2] - m[2][0]) / (4.0 * w),
                (m[1][0] - m[0][1]) / (4.0 * w),
            ]
        }
        1 => {
            let x = 0.5 * libm::sqrt(1.0 + 2.0 * m[0][0] - tr);
            [
                (m[2][1] - m[1][2]) / (4.0 * x),
                x,
                (m[0][1] + m[1][0]) / (4.0 * x),
                (m[0][2] + m[2][0]) / (4.0 * x),
            ]
        }
        2 => {
            let y = 0.5 * libm::sqrt(1.0 + 2.0 * m[1][1] - tr);
            [
                (m[0][2] - m[2][0]) / (4.0 * y),
                (m[0][1] + m[1][0]) / (4.0 * y),
                y,
                (m[1][2] + m[2][1]) / (4.0 * y),
            ]
        }
        _ => {
            let z = 0.5 * libm::sqrt(1.0 + 2.0 * m[2][2] - tr);
            [
                (m[1][0] - m[0][1]) / (4.0 * z),
                (m[0][2] + m[2][0]) / (4.0 * z),
                (m[1][2] + m[2][1]) / (4.0 * z),
                z,
            ]
        }
    };
    let sign = if q[0] < 0.0 { -1.0 } else { 1.0 };
    let len = libm::hypot(norm([q[1], q[2], q[3]]), q[0]);
    q.map(|v| v * sign / len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcs::TanWcs;

    #[test]
    fn quaternion_round_trip() {
        for &(ra, dec, roll) in &[
            (172.5, 2.0, 30.0),
            (10.0, -80.0, -120.0),
            (300.0, 45.0, 179.0),
            (0.0, 0.0, 0.0),
        ] {
            let p = Pointing {
                ra_deg: ra,
                dec_deg: dec,
                roll_deg: roll,
            };
            let back = quaternion_to_pointing(pointing_to_quaternion(&p)).unwrap();
            assert!((back.ra_deg - ra).abs() < 1e-9 && (back.dec_deg - dec).abs() < 1e-9);
            assert!(((back.roll_deg - roll + 180.0).rem_euclid(360.0) - 180.0).abs() < 1e-9);
        }
    }

    /// A point off-axis along camera +x lands at a larger column, and along
    /// camera −y at a larger row.
    #[test]
    fn camera_axes_match_the_wcs() {
        let p = Pointing {
            ra_deg: 50.0,
            dec_deg: 20.0,
            roll_deg: 40.0,
        };
        let m = pointing_matrix(&p);
        let w = TanWcs::from_pointing(&p, 1001, 1001, 3.6);
        let toward = |c: usize, sgn: f64| {
            let v: V3 = std::array::from_fn(|k| m[k][2] + sgn * 0.01 * m[k][c]);
            let (ra, dec) = unit_to_radec(v);
            w.world_to_pixel(ra, dec).unwrap()
        };
        let (x, y) = toward(0, 1.0);
        assert!(x > 510.0 && (y - 500.0).abs() < 1e-6, "{x} {y}");
        let (x, y) = toward(1, -1.0);
        assert!(y > 510.0 && (x - 500.0).abs() < 1e-6, "{x} {y}");
    }

    /// RA alone is rejected; the identity quaternion points the boresight at
    /// GCRS +z (Dec 90°).
    #[test]
    fn spec_needs_one_form() {
        assert!(PointingSpec {
            ra_deg: Some(1.0),
            ..Default::default()
        }
        .resolve()
        .is_err());
        let q = PointingSpec {
            quaternion: Some([1.0, 0.0, 0.0, 0.0]),
            ..Default::default()
        };
        let p = q.resolve().unwrap();
        assert!((p.dec_deg - 90.0).abs() < 1e-9);
    }
}
