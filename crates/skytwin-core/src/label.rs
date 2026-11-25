//! What a detector could see of each source: the facts a training label
//! needs.
//!
//! While a source is drawn, the electrons it adds near its path are kept:
//! its footprint, on the rows and columns within a reach of the path. Once
//! the frame's expected image is complete (sky, every source, flat field,
//! smear, dark current, hot pixels), each footprint is weighed against the
//! noise of the pixels it falls on, with the same per-pixel variance as the
//! Cramér–Rao bounds: expected signal plus read and extra white noise.
//!
//! The reach takes in every PSF component carrying at least 5% of the
//! light, to 5σ plus 2 px, grown by the field PSF at the farthest corner.
//! Fainter, wider components (a Moffat tail, a weak halo) fall partly
//! outside; they add almost nothing to the signal-to-noise.

use serde::{Deserialize, Serialize};

use crate::path::PathPoint;
use crate::psf::Psf;

/// Most points kept in a label path.
pub const PATH_POINTS: usize = 65;

/// Largest distance between a label path and the drawn path, px. Jitter
/// below it is left to `extent_px`, which is measured from the simplified
/// path; with the example profile's sub-pixel wobble a star keeps 2 points
/// instead of 32 at 0.1 px.
pub const PATH_TOLERANCE_PX: f64 = 0.5;

/// PSF components carrying less of the light than this do not set the
/// reach.
const MIN_COMPONENT: f64 = 0.05;

/// What a detector could see of one source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceLabel {
    /// Where it was drawn over the exposure, `[x, y]` pixels (pointing
    /// motion included): one point for a still source, else the path from
    /// the first to the last lit point, simplified to within 0.5 px and to
    /// at most 65 points with both ends kept. Empty when nothing was drawn
    /// (dark, or never near the image).
    pub path: Vec<[f64; 2]>,
    /// Signal-to-noise of an ideal matched filter, `sqrt(Σ sᵢ² / vᵢ)` over
    /// its pixels on the image: `sᵢ` its expected electrons (after the flat),
    /// `vᵢ` the pixel's variance. No detector can do better.
    pub snr: f64,
    /// Highest single-pixel signal-to-noise, `max sᵢ / sqrt(vᵢ)`.
    pub peak_snr: f64,
    /// Farthest distance from `path` of a pixel centre where the source is
    /// at least the pixel's noise σ, px (0 when it is nowhere).
    pub extent_px: f64,
    /// Whether a pixel where the source is at least the noise σ saturates:
    /// reaches full well, or the level where the delivered pixel clips at
    /// the output's maximum (everything on the pixel counted).
    pub saturated: bool,
}

/// The electrons one source added to the image near its path, before the
/// flat field.
#[derive(Clone, Debug, Default)]
pub(crate) struct Footprint {
    /// Rows `(y, x0, x1)`: columns `x0..x1` of row `y`.
    rows: Vec<(usize, usize, usize)>,
    /// Electrons per pixel, row after row.
    e: Vec<f64>,
}

impl Footprint {
    /// The mean of `map` (row-major, width `w`) over the footprint, weighted
    /// by the electrons the source put in each pixel, or `None` when it put
    /// none on the image.
    pub(crate) fn weighted_mean(&self, map: &[f64], w: usize) -> Option<f64> {
        let (mut sum, mut total) = (0.0, 0.0);
        let mut k = 0;
        for &(y, x0, x1) in &self.rows {
            for m in &map[y * w + x0..y * w + x1] {
                sum += self.e[k] * m;
                total += self.e[k];
                k += 1;
            }
        }
        (total > 0.0).then(|| sum / total)
    }
}

/// How far from its path a source's footprint is kept, px.
pub(crate) fn reach(psf: &Psf, w: usize, h: usize) -> f64 {
    let sigma = psf
        .comps
        .iter()
        .filter(|c| c.0 >= MIN_COMPONENT)
        .map(|c| c.1)
        .fold(psf.core_sigma(), f64::max);
    let growth = psf.field.map_or(1.0, |f| {
        let corners = [
            (0.0, 0.0),
            (w as f64, 0.0),
            (0.0, h as f64),
            (w as f64, h as f64),
        ];
        let rho2 = corners
            .iter()
            .map(|&(x, y)| ((x - f.axis.0).powi(2) + (y - f.axis.1).powi(2)) / (f.r_ref * f.r_ref))
            .fold(0.0, f64::max);
        1.0 + f.radial.max(f.tangential).max(0.0) * rho2
    });
    (5.0 * sigma + 2.0) * growth
}

/// The image rows and columns within `r` px of the polyline `path`, as
/// `(y, x0, x1)` runs on a `w × h` image.
///
/// Each piece contributes, on every row it comes within `r` of, the
/// columns of its part in that row's band of ±`r`, widened by `r`; a row
/// takes the span of all of them.
fn rows_near(path: &[(f64, f64)], r: f64, w: usize, h: usize) -> Vec<(usize, usize, usize)> {
    if path.is_empty() || w == 0 || h == 0 {
        return Vec::new();
    }
    let pieces: Vec<((f64, f64), (f64, f64))> = if path.len() == 1 {
        vec![(path[0], path[0])]
    } else {
        path.windows(2).map(|p| (p[0], p[1])).collect()
    };
    let ylo = path.iter().map(|p| p.1).fold(f64::INFINITY, f64::min) - r;
    let yhi = path.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max) + r;
    let j0 = libm::ceil(ylo).max(0.0);
    let j1 = libm::floor(yhi).min(h as f64 - 1.0);
    if j0 > j1 {
        return Vec::new();
    }
    let mut rows = Vec::new();
    for j in j0 as usize..=j1 as usize {
        let (b0, b1) = (j as f64 - r, j as f64 + r);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &(a, b) in &pieces {
            let (ya, yb) = (a.1.min(b.1), a.1.max(b.1));
            if yb < b0 || ya > b1 {
                continue;
            }
            let x_at = |y: f64| {
                if b.1 == a.1 {
                    (a.0, b.0)
                } else {
                    let u = ((y - a.1) / (b.1 - a.1)).clamp(0.0, 1.0);
                    let x = a.0 + u * (b.0 - a.0);
                    (x, x)
                }
            };
            let (p, q) = (x_at(b0.max(ya)), x_at(b1.min(yb)));
            lo = lo.min(p.0).min(p.1).min(q.0).min(q.1);
            hi = hi.max(p.0).max(p.1).max(q.0).max(q.1);
        }
        let i0 = libm::ceil(lo - r).max(0.0);
        let i1 = libm::floor(hi + r).min(w as f64 - 1.0);
        if i0 <= i1 {
            rows.push((j, i0 as usize, i1 as usize + 1));
        }
    }
    rows
}

/// Runs `draw` on `img` (`w` columns) and keeps what it adds within `r` px
/// of `path`.
pub(crate) fn capture(
    img: &mut [f64],
    w: usize,
    h: usize,
    path: &[(f64, f64)],
    r: f64,
    draw: impl FnOnce(&mut [f64]),
) -> Footprint {
    let rows = rows_near(path, r, w, h);
    let mut e: Vec<f64> = rows
        .iter()
        .flat_map(|&(y, x0, x1)| img[y * w + x0..y * w + x1].iter().copied())
        .collect();
    draw(img);
    let mut k = 0;
    for &(y, x0, x1) in &rows {
        for v in &img[y * w + x0..y * w + x1] {
            e[k] = v - e[k];
            k += 1;
        }
    }
    Footprint { rows, e }
}

/// The drawn part of a traced path: its points from the start of the first
/// lit piece that projects to the end of the last one.
pub(crate) fn lit_path(pts: &[PathPoint]) -> Vec<(f64, f64)> {
    let lit = |k: &usize| {
        let (a, b) = (&pts[*k], &pts[*k + 1]);
        a.xy.is_some() && b.xy.is_some() && 0.5 * (a.rate + b.rate) * (b.s - a.s) > 0.0
    };
    let n = pts.len().saturating_sub(1);
    match ((0..n).find(lit), (0..n).rfind(lit)) {
        (Some(i), Some(j)) => pts[i..=j + 1].iter().filter_map(|p| p.xy).collect(),
        _ => Vec::new(),
    }
}

/// `path` simplified to within [`PATH_TOLERANCE_PX`] (Douglas & Peucker,
/// Cartographica 10, 112, 1973), then, if still longer than
/// [`PATH_POINTS`], thinned evenly by index; both ends kept, coordinates
/// rounded to 0.001 px.
pub(crate) fn thin(path: &[(f64, f64)]) -> Vec<[f64; 2]> {
    let n = path.len();
    let mut keep = vec![false; n];
    if n > 0 {
        keep[0] = true;
        keep[n - 1] = true;
    }
    let mut stack = vec![(0, n.saturating_sub(1))];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let ends = [[path[a].0, path[a].1], [path[b].0, path[b].1]];
        let (k, d) = (a + 1..b)
            .map(|k| (k, distance(path[k], &ends)))
            .fold((a, -1.0), |m, x| if x.1 > m.1 { x } else { m });
        if d > PATH_TOLERANCE_PX {
            keep[k] = true;
            stack.push((a, k));
            stack.push((k, b));
        }
    }
    let kept: Vec<(f64, f64)> = (0..n).filter(|&k| keep[k]).map(|k| path[k]).collect();
    let m = kept.len();
    let last = PATH_POINTS - 1;
    let pick = |k: usize| {
        if m <= PATH_POINTS {
            k
        } else {
            k * (m - 1) / last
        }
    };
    let r = |v: f64| libm::round(v * 1e3) / 1e3;
    (0..m.min(PATH_POINTS))
        .map(|k| kept[pick(k)])
        .map(|p| [r(p.0), r(p.1)])
        .collect()
}

/// Distance from `p` to the polyline `path` (a single point when it has
/// one).
fn distance(p: (f64, f64), path: &[[f64; 2]]) -> f64 {
    if path.len() == 1 {
        return libm::hypot(p.0 - path[0][0], p.1 - path[0][1]);
    }
    path.windows(2)
        .map(|s| {
            let (a, b) = (s[0], s[1]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let l2 = dx * dx + dy * dy;
            let u = if l2 > 0.0 {
                (((p.0 - a[0]) * dx + (p.1 - a[1]) * dy) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            libm::hypot(p.0 - a[0] - u * dx, p.1 - a[1] - u * dy)
        })
        .fold(f64::INFINITY, f64::min)
}

/// The finished image a source is weighed against.
pub(crate) struct Noise<'a> {
    /// Expected electrons per pixel, everything included.
    pub img: &'a [f64],
    /// Image width.
    pub w: usize,
    /// The flat field the light went through, if any.
    pub flat: Option<&'a [f64]>,
    /// Variance added to every pixel's signal: read and extra white noise,
    /// electrons².
    pub floor: f64,
    /// Saturation level, electrons: the full well, or the level where the
    /// delivered pixel clips at the output's maximum when that is lower.
    pub full_well: f64,
}

/// The label of a source drawn along `path` (already thinned) that left
/// `fp` on the image.
pub(crate) fn label(path: Vec<[f64; 2]>, fp: &Footprint, n: &Noise<'_>) -> SourceLabel {
    let (mut sum, mut peak, mut extent, mut saturated) = (0.0, 0.0_f64, 0.0_f64, false);
    let mut k = 0;
    for &(y, x0, x1) in &fp.rows {
        for x in x0..x1 {
            let i = y * n.w + x;
            let s = fp.e[k] * n.flat.map_or(1.0, |f| f[i]);
            k += 1;
            let v = n.img[i] + n.floor;
            if s <= 0.0 || v <= 0.0 {
                continue;
            }
            let sigma = libm::sqrt(v);
            sum += s * s / v;
            peak = peak.max(s / sigma);
            if s >= sigma && !path.is_empty() {
                extent = extent.max(distance((x as f64, y as f64), &path));
                saturated |= n.img[i] >= n.full_well;
            }
        }
    }
    SourceLabel {
        path,
        snr: libm::sqrt(sum),
        peak_snr: peak,
        extent_px: extent,
        saturated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A diagonal streak and a point near a corner: the rows near their
    /// paths hold all the light drawn on the image.
    #[test]
    fn footprints_hold_what_was_drawn() {
        let (w, h) = (120, 90);
        let psf = Psf::with_wings(2.0, &[(0.2, 5.0)]);
        let r = reach(&psf, w, h);
        let mut img = vec![7.0; w * h];
        let path = [(10.0, 80.0), (100.0, 15.0)];
        let fp = capture(&mut img, w, h, &path, r, |im| {
            psf.streak(im, w, h, path[0], path[1], 1e5);
        });
        let kept: f64 = fp.e.iter().sum();
        let drawn: f64 = img.iter().map(|v| v - 7.0).sum();
        assert!((kept - drawn).abs() < 1e-6 * drawn, "{kept} vs {drawn}");
        assert!(fp.e.len() < w * h / 2, "{}", fp.e.len());
        let fp = capture(&mut img, w, h, &[(1.0, 2.0)], r, |im| {
            psf.point(im, w, h, 1.0, 2.0, 1e4);
        });
        let kept: f64 = fp.e.iter().sum();
        let drawn: f64 = img.iter().map(|v| v - 7.0).sum::<f64>() - drawn;
        assert!((kept - drawn).abs() < 1e-6 * drawn, "{kept} vs {drawn}");
    }

    /// On a flat background a Gaussian star's matched-filter SNR is
    /// `sqrt(Σ sᵢ² / (b + sᵢ))`; one pixel of the star at the noise σ sets
    /// the extent; a full well below the peak flags it.
    #[test]
    fn label_weighs_the_footprint_against_the_noise() {
        let (w, h) = (40, 40);
        let psf = Psf::gaussian(1.0);
        let mut img = vec![100.0; w * h];
        let fp = capture(&mut img, w, h, &[(20.0, 20.0)], reach(&psf, w, h), |im| {
            psf.point(im, w, h, 20.0, 20.0, 2000.0);
        });
        let want: f64 = img
            .iter()
            .map(|&v| (v - 100.0) * (v - 100.0) / (v + 25.0))
            .sum::<f64>()
            .sqrt();
        let noise = |full_well| Noise {
            img: &img,
            w,
            flat: None,
            floor: 25.0,
            full_well,
        };
        let l = label(vec![[20.0, 20.0]], &fp, &noise(1e9));
        assert!((l.snr - want).abs() < 1e-9 * want, "{} vs {want}", l.snr);
        let edge = (0..w * h)
            .filter(|&i| img[i] - 100.0 >= (img[i] + 25.0).sqrt())
            .map(|i| libm::hypot((i % w) as f64 - 20.0, (i / w) as f64 - 20.0))
            .fold(0.0, f64::max);
        assert_eq!(l.extent_px, edge);
        assert!(l.peak_snr > 10.0 && !l.saturated);
        assert!(label(vec![[20.0, 20.0]], &fp, &noise(300.0)).saturated);
    }

    /// A straight line keeps its ends; an arc keeps enough points to stay
    /// within tolerance; a fast zigzag is capped at the most points.
    #[test]
    fn thinning_follows_the_path_within_tolerance() {
        let line: Vec<(f64, f64)> = (0..1000)
            .map(|k| (f64::from(k), 0.5 * f64::from(k)))
            .collect();
        assert_eq!(thin(&line), vec![[0.0, 0.0], [999.0, 499.5]]);
        let arc: Vec<(f64, f64)> = (0..=1000)
            .map(|k| f64::from(k) * core::f64::consts::PI / 1000.0)
            .map(|a| (50.0 * libm::cos(a), 50.0 * libm::sin(a)))
            .collect();
        let t = thin(&arc);
        assert!(t.len() > 2 && t.len() < PATH_POINTS, "{}", t.len());
        assert_eq!((t[0], t[t.len() - 1]), ([50.0, 0.0], [-50.0, 0.0]));
        for p in &arc {
            assert!(distance(*p, &t) <= PATH_TOLERANCE_PX + 1e-3);
        }
        let zigzag: Vec<(f64, f64)> = (0..1000)
            .map(|k| (f64::from(k), f64::from(k % 2)))
            .collect();
        assert_eq!(thin(&zigzag).len(), PATH_POINTS);
        assert_eq!(thin(&[(1.23456, 2.0)]), vec![[1.235, 2.0]]);
    }
}
