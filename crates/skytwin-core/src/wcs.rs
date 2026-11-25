//! Gnomonic (TAN) projection between the sky and the pixel grid.
//!
//! Source: Calabretta & Greisen, "Representations of celestial coordinates
//! in FITS", A&A 395, 1077 (2002), section 5.1.3 (TAN), with the CD-matrix
//! form of Greisen & Calabretta, A&A 395, 1061 (2002).
//!
//! Pixel coordinates here are 0-based with pixel centres on whole numbers;
//! the FITS header gets `CRPIX = centre + 1`.
//!
//! Lens distortion is written as SIP (Shupe et al., "The SIP Convention for
//! Representing Distortion in FITS Image Headers", ASP Conf. Ser. 347, 491,
//! 2005).

use serde::{Deserialize, Serialize};

use crate::fits::Value;

/// Where the camera points.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pointing {
    /// Right ascension of the image centre, degrees (ICRS).
    pub ra_deg: f64,
    /// Declination of the image centre, degrees (ICRS).
    pub dec_deg: f64,
    /// Roll, degrees: the position angle of the image +y axis, east of
    /// north (Fayde's convention). With roll 0, north is +y and east is −x;
    /// with roll 90, +y points east and north is +x.
    #[serde(default)]
    pub roll_deg: f64,
}

/// Image handedness on the sky.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Parity {
    /// With north up (+y), east is left (−x): the sky as seen from inside
    /// the celestial sphere (det CD < 0). The usual case.
    #[default]
    Negative,
    /// Mirror image: with north up, east is right (det CD > 0).
    Positive,
}

/// Radial lens distortion about the reference pixel (the optical axis):
/// a point the ideal TAN projection puts `r` pixels from the axis lands at
/// `r (1 + k1 ρ² + k2 ρ⁴)`, ρ = r / `radius_px`. Negative `k1` is barrel
/// distortion (the edges pulled in), positive is pincushion.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Radial {
    /// Coefficient of ρ².
    pub k1: f64,
    /// Coefficient of ρ⁴.
    #[serde(default)]
    pub k2: f64,
    /// Radius the coefficients refer to, pixels (say the half-diagonal).
    pub radius_px: f64,
}

impl Radial {
    /// Radial scale factor `1 + k1 ρ² + k2 ρ⁴` for a squared ideal radius
    /// `r2`, pixels².
    fn factor(&self, r2: f64) -> f64 {
        let p = r2 / (self.radius_px * self.radius_px);
        1.0 + self.k1 * p + self.k2 * p * p
    }

    /// Ideal offset from the axis to the actual one, pixels.
    pub fn distort(&self, dx: f64, dy: f64) -> (f64, f64) {
        let f = self.factor(dx * dx + dy * dy);
        (dx * f, dy * f)
    }

    /// Largest ideal radius, pixels, out to which the actual radius still
    /// grows with it (`d r_actual / d r = 1 + 3 k1 ρ² + 5 k2 ρ⁴` stays
    /// above 0.1), found in steps of 0.01 in ρ up to ρ = 10. Beyond it the
    /// polynomial folds back: a fit to one frame says nothing there.
    pub fn max_radius_px(&self) -> f64 {
        let mut rho: f64 = 0.0;
        while rho < 10.0 {
            let p = (rho + 0.01) * (rho + 0.01);
            if 1.0 + 3.0 * self.k1 * p + 5.0 * self.k2 * p * p <= 0.1 {
                break;
            }
            rho += 0.01;
        }
        rho * self.radius_px
    }

    /// As [`Radial::distort`], `None` beyond [`Radial::max_radius_px`].
    pub fn distort_checked(&self, dx: f64, dy: f64) -> Option<(f64, f64)> {
        (libm::hypot(dx, dy) <= self.max_radius_px()).then(|| self.distort(dx, dy))
    }

    /// Actual offset to the ideal one, pixels (Newton on the radius, at most
    /// 30 iterations).
    pub fn undistort(&self, dx: f64, dy: f64) -> (f64, f64) {
        let rd = libm::hypot(dx, dy);
        if rd == 0.0 {
            return (0.0, 0.0);
        }
        let r0 = self.radius_px;
        let mut r = rd;
        for _ in 0..30 {
            let p = r * r / (r0 * r0);
            let g = r * (1.0 + self.k1 * p + self.k2 * p * p) - rd;
            let dg = 1.0 + 3.0 * self.k1 * p + 5.0 * self.k2 * p * p;
            if dg.abs() < 1e-12 {
                break;
            }
            let step = g / dg;
            r -= step;
            if step.abs() < 1e-12 * rd.max(1.0) {
                break;
            }
        }
        (dx * r / rd, dy * r / rd)
    }
}

/// A TAN WCS: tangent point, reference pixel and CD matrix, with optional
/// radial distortion about the reference pixel.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TanWcs {
    /// Tangent point, degrees `[ra, dec]`.
    pub crval_deg: [f64; 2],
    /// Reference pixel, 0-based `[x, y]`.
    pub crpix0: [f64; 2],
    /// CD matrix, degrees per pixel, row-major `[[cd11, cd12], [cd21, cd22]]`.
    pub cd: [[f64; 2]; 2],
    /// Lens distortion (none: a pure TAN projection).
    #[serde(default)]
    pub distortion: Option<Radial>,
}

impl TanWcs {
    /// WCS for a `width x height` image with square pixels of `scale_arcsec`,
    /// centred on `p`.
    pub fn from_pointing(p: &Pointing, width: usize, height: usize, scale_arcsec: f64) -> Self {
        Self::from_pointing_parity(p, width, height, scale_arcsec, Parity::Negative)
    }

    /// As [`TanWcs::from_pointing`], with the image's parity. Roll is the
    /// position angle of +y either way; positive parity flips the x column.
    pub fn from_pointing_parity(
        p: &Pointing,
        width: usize,
        height: usize,
        scale_arcsec: f64,
        parity: Parity,
    ) -> Self {
        let mut w = Self::negative(p, width, height, scale_arcsec);
        if parity == Parity::Positive {
            w.cd[0][0] = -w.cd[0][0];
            w.cd[1][0] = -w.cd[1][0];
        }
        w
    }

    /// Negative-parity WCS centred on the image.
    ///
    /// The CD columns are the (east, north) steps of +x and +y. +y sits at
    /// position angle `roll`: `s·(sin r, cos r)`; +x completes a negative
    /// parity (east left) frame: `s·(−cos r, sin r)`.
    fn negative(p: &Pointing, width: usize, height: usize, scale_arcsec: f64) -> Self {
        let s = scale_arcsec / 3600.0;
        let (sr, cr) = libm::sincos(p.roll_deg.to_radians());
        Self {
            crval_deg: [p.ra_deg, p.dec_deg],
            crpix0: [(width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0],
            cd: [[-s * cr, s * sr], [s * sr, s * cr]],
            distortion: None,
        }
    }

    /// This WCS with the given lens distortion.
    #[must_use]
    pub fn with_distortion(mut self, d: Option<Radial>) -> Self {
        self.distortion = d;
        self
    }

    /// Ideal (undistorted) offset from the reference pixel of pixel `(x, y)`.
    pub fn ideal_offset(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.crpix0[0], y - self.crpix0[1]);
        match &self.distortion {
            Some(d) => d.undistort(dx, dy),
            None => (dx, dy),
        }
    }

    /// Largest ideal (undistorted) distance, pixels, from the reference
    /// pixel to the outer corners of a `width × height` image: the radius
    /// of the sky cone that covers the whole frame, wherever the optical
    /// axis sits and however the lens distorts. Equals half the diagonal
    /// for a centred axis without distortion.
    pub fn field_radius_px(&self, width: usize, height: usize) -> f64 {
        let (w, h) = (width as f64, height as f64);
        [
            (-0.5, -0.5),
            (w - 0.5, -0.5),
            (-0.5, h - 0.5),
            (w - 0.5, h - 0.5),
        ]
        .into_iter()
        .map(|(x, y)| {
            let (dx, dy) = self.ideal_offset(x, y);
            libm::hypot(dx, dy)
        })
        .fold(0.0, f64::max)
    }

    /// This WCS with its reference pixel moved to `crpix0` (0-based), if
    /// given: the pointing then sits on that pixel.
    #[must_use]
    pub fn with_crpix(mut self, crpix0: Option<[f64; 2]>) -> Self {
        if let Some(c) = crpix0 {
            self.crpix0 = c;
        }
        self
    }

    /// Pixel scale, arcsec per pixel (geometric mean of the axes).
    pub fn scale_arcsec(&self) -> f64 {
        let [[a, b], [c, d]] = self.cd;
        libm::sqrt((a * d - b * c).abs()) * 3600.0
    }

    /// Standard (tangent-plane) coordinates, degrees, of a sky position
    /// about the tangent point. `None` 90° or more away.
    pub fn standard_coords(&self, ra_deg: f64, dec_deg: f64) -> Option<(f64, f64)> {
        let (a0, d0) = (
            self.crval_deg[0].to_radians(),
            self.crval_deg[1].to_radians(),
        );
        let (a, d) = (ra_deg.to_radians(), dec_deg.to_radians());
        let (sd, cd) = libm::sincos(d);
        let (sd0, cd0) = libm::sincos(d0);
        let (sda, cda) = libm::sincos(a - a0);
        let cosc = sd0 * sd + cd0 * cd * cda;
        if cosc <= 1e-9 {
            return None;
        }
        Some((
            (cd * sda / cosc).to_degrees(),
            ((cd0 * sd - sd0 * cd * cda) / cosc).to_degrees(),
        ))
    }

    /// Sky to pixel. `None` for points 90° or more from the tangent point,
    /// or beyond the distortion's [`Radial::max_radius_px`].
    pub fn world_to_pixel(&self, ra_deg: f64, dec_deg: f64) -> Option<(f64, f64)> {
        let (xi, eta) = self.standard_coords(ra_deg, dec_deg)?;
        let [[a11, a12], [a21, a22]] = self.cd;
        let det = a11 * a22 - a12 * a21;
        let dx = (a22 * xi - a12 * eta) / det;
        let dy = (-a21 * xi + a11 * eta) / det;
        let (dx, dy) = match &self.distortion {
            Some(d) => d.distort_checked(dx, dy)?,
            None => (dx, dy),
        };
        Some((self.crpix0[0] + dx, self.crpix0[1] + dy))
    }

    /// Pixel to sky, degrees `(ra in [0, 360), dec)`.
    pub fn pixel_to_world(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = self.ideal_offset(x, y);
        let [[a11, a12], [a21, a22]] = self.cd;
        let xi = (a11 * dx + a12 * dy).to_radians();
        let eta = (a21 * dx + a22 * dy).to_radians();
        let (a0, d0) = (
            self.crval_deg[0].to_radians(),
            self.crval_deg[1].to_radians(),
        );
        let (sd0, cd0) = libm::sincos(d0);
        let den = cd0 - eta * sd0;
        let a = a0 + libm::atan2(xi, den);
        let d = libm::atan2(sd0 + eta * cd0, libm::hypot(xi, den));
        (a.to_degrees().rem_euclid(360.0), d.to_degrees())
    }

    /// FITS header cards for this WCS: TAN, or TAN-SIP (order 5, fitted to
    /// the radial model within `radius_px` of the axis) with distortion.
    pub fn fits_cards(&self) -> Vec<(&'static str, Value, &'static str)> {
        let sip = self.distortion.map(|d| sip_fit(&d));
        let (t1, t2, what) = if sip.is_some() {
            (
                "RA---TAN-SIP",
                "DEC--TAN-SIP",
                "gnomonic projection, SIP distortion",
            )
        } else {
            ("RA---TAN", "DEC--TAN", "gnomonic projection")
        };
        let mut cards = vec![
            ("WCSAXES", Value::Int(2), "number of WCS axes"),
            ("CTYPE1", Value::Str(t1.into()), what),
            ("CTYPE2", Value::Str(t2.into()), what),
            ("CUNIT1", Value::Str("deg".into()), ""),
            ("CUNIT2", Value::Str("deg".into()), ""),
            ("RADESYS", Value::Str("ICRS".into()), ""),
            (
                "LONPOLE",
                Value::Float(180.0),
                "native longitude of the celestial pole, deg",
            ),
            (
                "CRVAL1",
                Value::Float(self.crval_deg[0]),
                "RA at reference pixel, deg",
            ),
            (
                "CRVAL2",
                Value::Float(self.crval_deg[1]),
                "Dec at reference pixel, deg",
            ),
            (
                "CRPIX1",
                Value::Float(self.crpix0[0] + 1.0),
                "reference pixel (1-based)",
            ),
            (
                "CRPIX2",
                Value::Float(self.crpix0[1] + 1.0),
                "reference pixel (1-based)",
            ),
            ("CD1_1", Value::Float(self.cd[0][0]), "deg/pixel"),
            ("CD1_2", Value::Float(self.cd[0][1]), "deg/pixel"),
            ("CD2_1", Value::Float(self.cd[1][0]), "deg/pixel"),
            ("CD2_2", Value::Float(self.cd[1][1]), "deg/pixel"),
        ];
        if let Some((fwd, inv)) = sip {
            let sets = [
                (["A_ORDER", "B_ORDER"], fwd, 0, "SIP order, pixel to sky"),
                (["AP_ORDER", "BP_ORDER"], inv, 2, "SIP order, sky to pixel"),
            ];
            for (orders, coeffs, col, comment) in sets {
                for order in orders {
                    cards.push((order, Value::Int(SIP_ORDER as i64), comment));
                }
                for (k, (_, _, names)) in SIP_TERMS.iter().enumerate() {
                    cards.push((names[col], Value::Float(coeffs[0][k]), ""));
                    cards.push((names[col + 1], Value::Float(coeffs[1][k]), ""));
                }
            }
        }
        cards
    }
}

/// Polynomial order of the SIP fit.
const SIP_ORDER: usize = 5;

/// Number of SIP terms of order 2 to [`SIP_ORDER`].
const SIP_N: usize = 18;

/// One SIP term `x^p y^q`: `(p, q, [A, B, AP, BP card names])`.
type SipTerm = (u32, u32, [&'static str; 4]);

/// SIP coefficients for the two axes, in [`SIP_TERMS`] order.
type SipCoeffs = [[f64; SIP_N]; 2];

/// SIP terms `x^p y^q`, 2 ≤ p + q ≤ 5, with their A, B, AP, BP card names.
const SIP_TERMS: [SipTerm; SIP_N] = [
    (2, 0, ["A_2_0", "B_2_0", "AP_2_0", "BP_2_0"]),
    (1, 1, ["A_1_1", "B_1_1", "AP_1_1", "BP_1_1"]),
    (0, 2, ["A_0_2", "B_0_2", "AP_0_2", "BP_0_2"]),
    (3, 0, ["A_3_0", "B_3_0", "AP_3_0", "BP_3_0"]),
    (2, 1, ["A_2_1", "B_2_1", "AP_2_1", "BP_2_1"]),
    (1, 2, ["A_1_2", "B_1_2", "AP_1_2", "BP_1_2"]),
    (0, 3, ["A_0_3", "B_0_3", "AP_0_3", "BP_0_3"]),
    (4, 0, ["A_4_0", "B_4_0", "AP_4_0", "BP_4_0"]),
    (3, 1, ["A_3_1", "B_3_1", "AP_3_1", "BP_3_1"]),
    (2, 2, ["A_2_2", "B_2_2", "AP_2_2", "BP_2_2"]),
    (1, 3, ["A_1_3", "B_1_3", "AP_1_3", "BP_1_3"]),
    (0, 4, ["A_0_4", "B_0_4", "AP_0_4", "BP_0_4"]),
    (5, 0, ["A_5_0", "B_5_0", "AP_5_0", "BP_5_0"]),
    (4, 1, ["A_4_1", "B_4_1", "AP_4_1", "BP_4_1"]),
    (3, 2, ["A_3_2", "B_3_2", "AP_3_2", "BP_3_2"]),
    (2, 3, ["A_2_3", "B_2_3", "AP_2_3", "BP_2_3"]),
    (1, 4, ["A_1_4", "B_1_4", "AP_1_4", "BP_1_4"]),
    (0, 5, ["A_0_5", "B_0_5", "AP_0_5", "BP_0_5"]),
];

/// SIP coefficients for a radial model, by least squares on a grid over
/// the disc of `radius_px` (the whole frame when that is the
/// half-diagonal): forward (A, B: actual offset u to ideal u + A(u)) and
/// inverse (AP, BP: ideal U to actual U + AP(U)).
fn sip_fit(d: &Radial) -> (SipCoeffs, SipCoeffs) {
    let n = 41;
    let span = d.radius_px;
    let mut fwd_pts = Vec::with_capacity(n * n);
    let mut inv_pts = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let u = span * (2.0 * i as f64 / (n - 1) as f64 - 1.0);
            let v = span * (2.0 * j as f64 / (n - 1) as f64 - 1.0);
            if u * u + v * v > span * span {
                continue;
            }
            let (iu, iv) = d.undistort(u, v);
            fwd_pts.push((u, v, iu - u, iv - v));
            let (au, av) = d.distort(u, v);
            inv_pts.push((u, v, au - u, av - v));
        }
    }
    (lsq_terms(&fwd_pts, span), lsq_terms(&inv_pts, span))
}

/// Least squares of `(du, dv)` on the SIP terms, in coordinates scaled by
/// `span` for conditioning; coefficients returned for unscaled pixels.
fn lsq_terms(pts: &[(f64, f64, f64, f64)], span: f64) -> SipCoeffs {
    let mut ata = [[0.0; SIP_N]; SIP_N];
    let mut atb = [[0.0; SIP_N]; 2];
    for &(u, v, du, dv) in pts {
        let (s, t) = (u / span, v / span);
        let row =
            SIP_TERMS.map(|(p, q, _)| libm::pow(s, f64::from(p)) * libm::pow(t, f64::from(q)));
        for (a, &ra) in row.iter().enumerate() {
            for (b, &rb) in row.iter().enumerate() {
                ata[a][b] += ra * rb;
            }
            atb[0][a] += ra * du;
            atb[1][a] += ra * dv;
        }
    }
    let mut out = [[0.0; SIP_N]; 2];
    for (o, rhs) in out.iter_mut().zip(atb) {
        let x = solve(ata, rhs);
        for (k, &(p, q, _)) in SIP_TERMS.iter().enumerate() {
            o[k] = x[k] / libm::pow(span, f64::from(p + q));
        }
    }
    out
}

/// Solve `a · x = b` by Gaussian elimination with partial pivoting; a
/// (near-)zero pivot gives a zero component instead of a division by zero.
fn solve<const K: usize>(mut a: [[f64; K]; K], mut b: [f64; K]) -> [f64; K] {
    for c in 0..K {
        let p = (c..K)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap_or(c);
        a.swap(c, p);
        b.swap(c, p);
        if a[c][c].abs() < 1e-300 {
            continue;
        }
        for r in c + 1..K {
            let f = a[r][c] / a[c][c];
            let (upper, lower) = a.split_at_mut(r);
            for (v, &p) in lower[0][c..].iter_mut().zip(&upper[c][c..]) {
                *v -= f * p;
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; K];
    for c in (0..K).rev() {
        let s: f64 = (c + 1..K).map(|k| a[c][k] * x[k]).sum();
        x[c] = if a[c][c].abs() < 1e-300 {
            0.0
        } else {
            (b[c] - s) / a[c][c]
        };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wcs(roll: f64, dec: f64) -> TanWcs {
        TanWcs::from_pointing(
            &Pointing {
                ra_deg: 172.5,
                dec_deg: dec,
                roll_deg: roll,
            },
            1024,
            768,
            3.0,
        )
    }

    #[test]
    fn centre_maps_to_pointing() {
        let w = wcs(30.0, 2.0);
        let (x, y) = w.world_to_pixel(172.5, 2.0).unwrap();
        assert!((x - 511.5).abs() < 1e-9 && (y - 383.5).abs() < 1e-9);
    }

    #[test]
    fn round_trip() {
        for &(roll, dec) in &[(0.0, 2.0), (37.0, -45.0), (200.0, 89.5), (-90.0, 0.0)] {
            let w = wcs(roll, dec);
            for &(x, y) in &[(0.0, 0.0), (1023.0, 767.0), (100.25, 600.75)] {
                let (ra, de) = w.pixel_to_world(x, y);
                let (x2, y2) = w.world_to_pixel(ra, de).unwrap();
                assert!(
                    (x - x2).abs() < 1e-8 && (y - y2).abs() < 1e-8,
                    "{roll} {dec}: {x2} {y2}"
                );
            }
        }
    }

    #[test]
    fn north_up_east_left_at_roll_zero() {
        let w = wcs(0.0, 2.0);
        let (_, y) = w.world_to_pixel(172.5, 2.01).unwrap();
        assert!(y > 383.5, "north should be +y");
        let (x, _) = w.world_to_pixel(172.51, 2.0).unwrap();
        assert!(x < 511.5, "east should be -x");
    }

    /// Roll 90: +y points east, north is +x.
    #[test]
    fn roll_is_position_angle_of_plus_y() {
        let w = wcs(90.0, 2.0);
        let (x, y) = w.world_to_pixel(172.5, 2.01).unwrap();
        assert!(x > 511.5 + 5.0 && (y - 383.5).abs() < 0.1, "{x} {y}");
        let (x, y) = w.world_to_pixel(172.51, 2.0).unwrap();
        assert!(y > 383.5 + 5.0 && (x - 511.5).abs() < 0.1, "{x} {y}");
    }

    #[test]
    fn positive_parity_mirrors_east() {
        let p = Pointing {
            ra_deg: 172.5,
            dec_deg: 2.0,
            roll_deg: 0.0,
        };
        let w = TanWcs::from_pointing_parity(&p, 1024, 768, 3.0, Parity::Positive);
        let (x, _) = w.world_to_pixel(172.51, 2.0).unwrap();
        assert!(x > 511.5, "east should be +x");
        let (_, y) = w.world_to_pixel(172.5, 2.01).unwrap();
        assert!(y > 383.5, "north stays +y");
        let [[a, b], [c, d]] = w.cd;
        assert!(a * d - b * c > 0.0);
    }

    #[test]
    fn scale() {
        assert!((wcs(17.0, 2.0).scale_arcsec() - 3.0).abs() < 1e-12);
    }

    /// Barrel distortion pulls a corner star in and leaves the centre; the
    /// SIP cards reproduce the model to a small fraction of a pixel.
    #[test]
    fn radial_distortion_round_trips_and_its_sip_matches() {
        let d = Radial {
            k1: -0.03,
            k2: 0.004,
            radius_px: 180.0,
        };
        let w = wcs(30.0, 40.0).with_distortion(Some(d));
        let plain = wcs(30.0, 40.0);
        let (cx, cy) = (w.crpix0[0], w.crpix0[1]);
        let (ra, dec) = plain.pixel_to_world(cx + 150.0, cy + 100.0);
        let (x, y) = w.world_to_pixel(ra, dec).unwrap();
        assert!(libm::hypot(x - cx, y - cy) < libm::hypot(150.0, 100.0) - 1.0);
        let (ra0, dec0) = plain.pixel_to_world(cx, cy);
        let (x0, y0) = w.world_to_pixel(ra0, dec0).unwrap();
        assert!((x0 - cx).abs() < 1e-9 && (y0 - cy).abs() < 1e-9);
        for &(px, py) in &[
            (0.0, 0.0),
            (700.0, 20.0),
            (1023.0, 767.0),
            (cx + 3.0, cy - 250.0),
        ] {
            let (ra, dec) = w.pixel_to_world(px, py);
            let (x, y) = w.world_to_pixel(ra, dec).unwrap();
            assert!(
                (x - px).abs() < 1e-6 && (y - py).abs() < 1e-6,
                "{px} {py} -> {x} {y}"
            );
        }
        let (fwd, inv) = sip_fit(&d);
        let poly = |c: &[f64; SIP_N], u: f64, v: f64| -> f64 {
            SIP_TERMS
                .iter()
                .enumerate()
                .map(|(k, &(p, q, _))| {
                    c[k] * libm::pow(u, f64::from(p)) * libm::pow(v, f64::from(q))
                })
                .sum()
        };
        for &(u, v) in &[(100.0, 0.0), (-90.0, 120.0), (60.0, -170.0)] {
            let (iu, iv) = d.undistort(u, v);
            assert!(
                (u + poly(&fwd[0], u, v) - iu).abs() < 0.02,
                "A at {u} {v}: {} vs {iu}, AP {} vs {}",
                u + poly(&fwd[0], u, v),
                u + poly(&inv[0], u, v),
                d.distort(u, v).0
            );
            assert!((v + poly(&fwd[1], u, v) - iv).abs() < 0.02, "B at {u} {v}");
            let (au, av) = d.distort(u, v);
            assert!((u + poly(&inv[0], u, v) - au).abs() < 0.02, "AP at {u} {v}");
            assert!((v + poly(&inv[1], u, v) - av).abs() < 0.02, "BP at {u} {v}");
        }
        let cards = w.fits_cards();
        assert!(cards
            .iter()
            .any(|c| c.0 == "CTYPE1" && c.1 == Value::Str("RA---TAN-SIP".into())));
        assert!(cards.iter().any(|c| c.0 == "AP_5_0"));
        assert!(
            cards
                .iter()
                .any(|c| c.0 == "LONPOLE" && c.1 == Value::Float(180.0)),
            "LONPOLE 180 keeps the header's sky orientation at a pole"
        );
    }

    /// A fit fine out to ρ ≈ 1 that folds back beyond ρ ≈ 1.5: a point 3
    /// radii out is off the image, not mapped near the centre.
    #[test]
    fn distortion_does_not_fold_far_points_back_into_the_frame() {
        let d = Radial {
            k1: 0.077,
            k2: -0.109,
            radius_px: 181.0,
        };
        let rmax = d.max_radius_px();
        assert!(rmax > 181.0 && rmax < 2.0 * 181.0, "{rmax}");
        let w = wcs(0.0, 30.0).with_distortion(Some(d));
        let plain = wcs(0.0, 30.0);
        let (cx, cy) = (w.crpix0[0], w.crpix0[1]);
        let (ra, dec) = plain.pixel_to_world(cx + 3.0 * 181.0, cy);
        assert!(w.world_to_pixel(ra, dec).is_none());
        let (ra, dec) = plain.pixel_to_world(cx + 120.0, cy);
        assert!(w.world_to_pixel(ra, dec).is_some());
    }

    /// The sky cone covers the frame's corners: half the diagonal when
    /// centred, more with an off-centre axis or barrel distortion.
    #[test]
    fn field_radius_reaches_every_corner() {
        let w = wcs(0.0, 10.0);
        assert_eq!(
            w.field_radius_px(1024, 768),
            0.5 * libm::hypot(1024.0, 768.0)
        );
        let off = w.with_crpix(Some([100.0, 100.0]));
        assert!(off.field_radius_px(1024, 768) > 0.5 * libm::hypot(1024.0, 768.0) + 100.0);
        let barrel = w.with_distortion(Some(Radial {
            k1: -0.05,
            k2: 0.0,
            radius_px: 640.0,
        }));
        assert!(barrel.field_radius_px(1024, 768) > w.field_radius_px(1024, 768));
    }
}
