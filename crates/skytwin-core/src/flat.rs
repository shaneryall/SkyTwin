//! Flat field: how the response to light varies over the image.
//!
//! - Vignetting: a radial polynomial about the optical axis,
//!   `1 + a2 ρ² + a4 ρ⁴`, with ρ the distance in half-diagonals.
//! - Pixel response non-uniformity (PRNU): a fractional Gaussian scatter
//!   fixed to each chip pixel (a hash of the detector seed and the chip
//!   position, so any window or binning of one chip sees the same pattern;
//!   binned pixels average their chip pixels). Janesick, *Scientific
//!   Charge-Coupled Devices* (2001), section 2.6.
//! - Dust: out-of-focus dust on a window or filter casts an annular shadow
//!   (a "donut": the pupil's image, with the central obstruction's hole),
//!   fixed to the chip.
//! - Field stop: a circular aperture in the optics (or a fibre taper's
//!   end) imaged onto the chip: full response inside, a soft edge, and a
//!   small leak beyond it.
//!
//! The flat multiplies light only; dark current and hot pixels are not
//! flat-fielded.

use serde::{Deserialize, Serialize};

use crate::detector::Layout;
use crate::param::{DrawCtx, Param};
use crate::rng::{splitmix64, Rng};

/// Mixes `detector_seed` into the stream for dust positions.
const DUST_STREAM: u64 = 0xD057_0000_0000_0007;

/// Out-of-focus dust shadows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dust {
    /// Dust shadows per million chip pixels.
    pub density_per_mpix: f64,
    /// Outer radius of each shadow, chip pixels (drawn per shadow).
    pub radius_px: Param,
    /// Fraction of light removed in the ring (drawn per shadow).
    pub depth: Param,
    /// Inner radius as a fraction of the outer (central obstruction).
    #[serde(default = "default_hole")]
    pub hole_fraction: f64,
}

/// Serde default for [`Dust::hole_fraction`].
fn default_hole() -> f64 {
    0.35
}

/// Flat-field settings.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Flat {
    /// Vignetting coefficients `[a2, a4]` (response 1 + a2 ρ² + a4 ρ⁴).
    #[serde(default)]
    pub vignetting: Option<[f64; 2]>,
    /// Pixel response non-uniformity, fractional RMS.
    #[serde(default)]
    pub prnu_rms: Option<f64>,
    /// Dust shadows.
    #[serde(default)]
    pub dust: Option<Dust>,
    /// Circular field stop.
    #[serde(default)]
    pub field_stop: Option<FieldStop>,
}

/// A circular field stop: response `outside + (1 − outside) · ½ erfc((r −
/// radius) / (√2 edge))` at `r` pixels from its centre.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldStop {
    /// Centre, image pixels `[x, y]` (default: the optical axis).
    #[serde(default)]
    pub centre_px: Option<[f64; 2]>,
    /// Radius where the response is half, pixels.
    pub radius_px: f64,
    /// Edge width (Gaussian σ of the blur), pixels.
    #[serde(default = "default_edge")]
    pub edge_px: f64,
    /// Response beyond the stop, as a fraction of the inside's.
    #[serde(default)]
    pub outside: f64,
}

/// Serde default for [`FieldStop::edge_px`].
fn default_edge() -> f64 {
    1.0
}

impl FieldStop {
    /// Response at `r` pixels from the centre (`edge_px` floored at 1e-6).
    pub fn response(&self, r: f64) -> f64 {
        let edge = self.edge_px.max(1e-6);
        let t = 0.5 * libm::erfc((r - self.radius_px) / (core::f64::consts::SQRT_2 * edge));
        self.outside + (1.0 - self.outside) * t
    }
}

/// A dust shadow on the chip.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DustShadow {
    /// Centre, chip pixels `[x, y]`.
    pub centre: [f64; 2],
    /// Outer radius, chip pixels.
    pub radius: f64,
    /// Fraction of light removed.
    pub depth: f64,
}

/// A standard normal fixed to (seed, chip x, chip y): a hash, not a
/// stream, so any window or binning sees the same value (Box–Muller).
fn chip_normal(seed: u64, xc: usize, yc: usize) -> f64 {
    let h = splitmix64(seed ^ splitmix64(((xc as u64) << 32) | yc as u64));
    let u1 = ((splitmix64(h) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = ((splitmix64(h ^ 0xABCD) >> 11) as f64) / (1u64 << 53) as f64;
    libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * core::f64::consts::PI * u2)
}

/// The dust shadows on the chip: `density_per_mpix` scaled to the chip
/// area, centres uniform over the chip, radius and depth drawn per shadow.
pub fn dust_map(d: &Dust, layout: &Layout, rng: &mut Rng) -> Vec<DustShadow> {
    let [cw, ch] = layout.chip_px;
    let n = layout.count_on_chip(d.density_per_mpix);
    let ctx = &mut DrawCtx::default();
    (0..n)
        .map(|_| DustShadow {
            centre: [rng.range(0.0, cw as f64), rng.range(0.0, ch as f64)],
            radius: d.radius_px.draw(rng, ctx).max(0.5),
            depth: d.depth.draw(rng, ctx).clamp(0.0, 1.0),
        })
        .collect()
}

/// The flat for an image of `w × h`, optical axis `axis` (image pixels),
/// and the dust shadows on the chip.
///
/// Vignetting and the field stop are evaluated per image pixel; PRNU and
/// dust per chip pixel, averaged over each binned pixel. Each dust shadow is
/// a soft-edged annulus with 1-pixel edges between `hole_fraction × radius`
/// and `radius`. Dust positions come from their own stream keyed by
/// `detector_seed`, so they stay put from frame to frame.
pub fn flat_map(
    f: &Flat,
    layout: &Layout,
    w: usize,
    h: usize,
    axis: (f64, f64),
    detector_seed: u64,
) -> (Vec<f64>, Vec<DustShadow>) {
    let mut map = vec![1.0; w * h];
    let half_diag = 0.5 * libm::hypot(w as f64, h as f64);
    if let Some([a2, a4]) = f.vignetting {
        for y in 0..h {
            for x in 0..w {
                let r2 = ((x as f64 - axis.0).powi(2) + (y as f64 - axis.1).powi(2))
                    / (half_diag * half_diag);
                map[y * w + x] *= (1.0 + a2 * r2 + a4 * r2 * r2).max(0.0);
            }
        }
    }
    if let Some(fs) = &f.field_stop {
        let c = fs.centre_px.unwrap_or([axis.0, axis.1]);
        for y in 0..h {
            for x in 0..w {
                map[y * w + x] *= fs.response(libm::hypot(x as f64 - c[0], y as f64 - c[1]));
            }
        }
    }
    let [bx, by] = layout.binning;
    let mut shadows = Vec::new();
    if let Some(d) = &f.dust {
        let mut rng = Rng::new(detector_seed ^ DUST_STREAM);
        shadows = dust_map(d, layout, &mut rng);
    }
    if f.prnu_rms.is_some() || !shadows.is_empty() {
        let prnu = f.prnu_rms.unwrap_or(0.0);
        let hole = f.dust.as_ref().map_or(0.0, |d| d.hole_fraction);
        for y in 0..h {
            for x in 0..w {
                let (x0, y0) = layout.chip_origin_of(x, y);
                let mut sum = 0.0;
                for j in 0..by {
                    for i in 0..bx {
                        let (xc, yc) = (x0 + i, y0 + j);
                        let mut v = 1.0 + prnu * chip_normal(detector_seed, xc, yc);
                        for s in &shadows {
                            let r = libm::hypot(
                                xc as f64 + 0.5 - s.centre[0],
                                yc as f64 + 0.5 - s.centre[1],
                            );
                            let outer = (s.radius - r + 0.5).clamp(0.0, 1.0);
                            let inner = (r - hole * s.radius + 0.5).clamp(0.0, 1.0);
                            v *= 1.0 - s.depth * outer * inner;
                        }
                        sum += v;
                    }
                }
                map[y * w + x] *= sum / (bx * by) as f64;
            }
        }
    }
    (map, shadows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vignetting darkens the corners; PRNU is fixed to the chip, so a
    /// window shifted by (10, 0) sees the same values.
    #[test]
    fn vignetting_and_prnu() {
        let f = Flat {
            vignetting: Some([-0.2, 0.0]),
            prnu_rms: Some(0.01),
            dust: None,
            field_stop: None,
        };
        let l = Layout::identity(200, 200);
        let (m, _) = flat_map(&f, &l, 200, 200, (99.5, 99.5), 3);
        let corner = m[0];
        let centre = m[100 * 200 + 100];
        assert!(
            (centre - 1.0).abs() < 0.05 && (corner - 0.8).abs() < 0.05,
            "{centre} {corner}"
        );
        let l2 = Layout {
            chip_px: [400, 200],
            window_origin_px: [10, 0],
            binning: [1, 1],
        };
        let g = Flat {
            vignetting: None,
            ..f.clone()
        };
        let (a, _) = flat_map(
            &g,
            &Layout {
                chip_px: [400, 200],
                window_origin_px: [0, 0],
                binning: [1, 1],
            },
            200,
            200,
            (0.0, 0.0),
            3,
        );
        let (b, _) = flat_map(&g, &l2, 200, 200, (0.0, 0.0), 3);
        assert_eq!(a[50 * 200 + 60], b[50 * 200 + 50]);
        let mean = a.iter().sum::<f64>() / a.len() as f64;
        let rms = libm::sqrt(a.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / a.len() as f64);
        assert!((rms - 0.01).abs() < 0.001, "{rms}");
    }

    #[test]
    fn dust_makes_a_donut() {
        let f = Flat {
            vignetting: None,
            prnu_rms: None,
            dust: Some(Dust {
                density_per_mpix: 25.0,
                radius_px: 20.0.into(),
                depth: 0.1.into(),
                hole_fraction: 0.4,
            }),
            field_stop: None,
        };
        let l = Layout::identity(200, 200);
        let (m, sh) = flat_map(&f, &l, 200, 200, (0.0, 0.0), 1);
        assert_eq!(sh.len(), 1);
        let [cx, cy] = sh[0].centre;
        let at = |dx: f64| {
            let (x, y) = ((cx + dx) as i64, cy as i64);
            if x < 0 || y < 0 || x >= 200 || y >= 200 {
                1.0
            } else {
                m[y as usize * 200 + x as usize]
            }
        };
        assert!((at(0.0) - 1.0).abs() < 1e-9, "hole");
        assert!(
            (at(14.0) - 0.9).abs() < 1e-9 || (at(-14.0) - 0.9).abs() < 1e-9,
            "ring"
        );
    }

    #[test]
    fn field_stop_passes_inside_and_blocks_outside() {
        let f = Flat {
            field_stop: Some(FieldStop {
                centre_px: Some([40.0, 50.0]),
                radius_px: 30.0,
                edge_px: 2.0,
                outside: 0.05,
            }),
            ..Flat::default()
        };
        let (m, _) = flat_map(&f, &Layout::identity(100, 100), 100, 100, (49.5, 49.5), 1);
        let at = |x: usize, y: usize| m[y * 100 + x];
        assert!((at(40, 50) - 1.0).abs() < 1e-9);
        assert!((at(70, 50) - 0.525).abs() < 1e-9, "{}", at(70, 50));
        assert!((at(95, 95) - 0.05).abs() < 1e-6);
    }
}
