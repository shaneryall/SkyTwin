//! Point-spread functions, integrated over pixels.
//!
//! Pixel `(i, j)` covers `[i - 0.5, i + 0.5) x [j - 0.5, j + 0.5)`, so pixel
//! centres sit on integer coordinates (0-based, like FITS minus one).

use core::f64::consts::SQRT_2;
use std::sync::OnceLock;

/// Grid step of the erf table.
const ERF_STEP: f64 = 1.0 / 64.0;
/// Beyond this |x|, erf is ±1 to double precision.
const ERF_LIMIT: f64 = 6.0;

/// `erf` and its derivative on a fixed grid, built once with `libm` (so the
/// same bits on every machine).
fn erf_table() -> &'static [(f64, f64)] {
    static T: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    T.get_or_init(|| {
        let n = (2.0 * ERF_LIMIT / ERF_STEP) as usize;
        let k = 2.0 / libm::sqrt(core::f64::consts::PI);
        (0..=n)
            .map(|i| {
                let x = -ERF_LIMIT + i as f64 * ERF_STEP;
                (libm::erf(x), k * libm::exp(-x * x))
            })
            .collect()
    })
}

/// `erf` by cubic Hermite interpolation of a table (values and slopes on a
/// 1/64 grid): absolute error under 1e-9, several times faster than
/// `libm::erf`. Used for streaks, where it is evaluated per pixel.
#[inline]
pub fn erf_fast(x: f64) -> f64 {
    if x >= ERF_LIMIT {
        return 1.0;
    }
    if x <= -ERF_LIMIT || x.is_nan() {
        return if x.is_nan() { x } else { -1.0 };
    }
    let t = erf_table();
    let u = (x + ERF_LIMIT) / ERF_STEP;
    let i = (u as usize).min(t.len() - 2);
    let f = u - i as f64;
    let ((y0, d0), (y1, d1)) = (t[i], t[i + 1]);
    let (d0, d1) = (d0 * ERF_STEP, d1 * ERF_STEP);
    let f2 = f * f;
    let f3 = f2 * f;
    (2.0 * f3 - 3.0 * f2 + 1.0) * y0
        + (f3 - 2.0 * f2 + f) * d0
        + (-2.0 * f3 + 3.0 * f2) * y1
        + (f3 - f2) * d1
}

/// Smallest variance, px², of a Gaussian drawn by sampling at pixel
/// centres, as moving sources are: σ = 0.7 px, where sampling stops being
/// accurate to well under 1% (see [`add_streak`]). Narrower ones are
/// integrated over each pixel instead, so their flux is kept and a source
/// that barely moves gives the pixels of a still one.
const SAMPLED_MIN_VAR: f64 = 0.49;

/// Add `flux` × a Gaussian of mean `mu` and covariance `c` (`[sxx, sxy,
/// syy]`, px²), sampled at pixel centres (as streaks are) and normalised
/// analytically.
///
/// Narrower than σ = 0.7 px on either axis, it is integrated over each
/// pixel instead: exactly (erf per axis) when its axes are uncorrelated,
/// by 3×3-point Gauss–Legendre quadrature and renormalisation otherwise.
///
/// Only pixels within 6σ (Mahalanobis) are visited: on row `dy` that is
/// `a x² + 2 b dy x + cc dy² ≤ 36` for the precision matrix `[[a, b], [b,
/// cc]]` and `x = i − μx`. The exponential is stepped along each row by a
/// recurrence: `e(x+1)/e(x) = exp(−(a(2x+1) + 2 b dy)/2)`, a ratio that
/// itself changes by `e^(−a)` per pixel (two multiplications a pixel).
pub fn add_gauss_cov(img: &mut [f64], w: usize, h: usize, mu: (f64, f64), c: &[f64; 3], flux: f64) {
    const Q_MAX: f64 = 36.0;
    let [sxx, sxy, syy] = *c;
    let det = sxx * syy - sxy * sxy;
    if det.is_nan() || det <= 0.0 || flux == 0.0 {
        return;
    }
    let min_var = 0.5 * (sxx + syy) - libm::hypot(0.5 * (sxx - syy), sxy);
    if min_var < SAMPLED_MIN_VAR {
        if sxy.abs() <= 1e-9 * libm::sqrt(sxx * syy) {
            add_gaussian_xy(
                img,
                w,
                h,
                mu.0,
                mu.1,
                flux,
                libm::sqrt(sxx),
                libm::sqrt(syy),
            );
        } else {
            add_elliptical(img, w, h, mu.0, mu.1, flux, c);
        }
        return;
    }
    let (a, b, cc) = (syy / det, -sxy / det, sxx / det);
    let norm = flux / (2.0 * core::f64::consts::PI * libm::sqrt(det));
    let ry = libm::sqrt(Q_MAX * syy);
    let j0 = libm::ceil(mu.1 - ry).max(0.0) as i64;
    let j1 = (libm::floor(mu.1 + ry) as i64).min(h as i64 - 1);
    let step = libm::exp(-a);
    for j in j0..=j1 {
        let dy = j as f64 - mu.1;
        let half_b = b * dy;
        let disc = half_b * half_b - a * (cc * dy * dy - Q_MAX);
        if disc < 0.0 {
            continue;
        }
        let r = libm::sqrt(disc);
        let x_lo = (-half_b - r) / a;
        let x_hi = (-half_b + r) / a;
        let i0 = libm::ceil(mu.0 + x_lo).max(0.0) as i64;
        let i1 = (libm::floor(mu.0 + x_hi) as i64).min(w as i64 - 1);
        if i0 > i1 {
            continue;
        }
        let x = i0 as f64 - mu.0;
        let q = a * x * x + 2.0 * b * x * dy + cc * dy * dy;
        let mut e = libm::exp(-0.5 * q);
        let mut ratio = libm::exp(-0.5 * (a * (2.0 * x + 1.0) + 2.0 * b * dy));
        let row = &mut img[j as usize * w..(j as usize + 1) * w];
        for v in &mut row[i0 as usize..=i1 as usize] {
            *v += norm * e;
            e *= ratio;
            ratio *= step;
        }
    }
}

/// Fraction of a 1-D Gaussian (centre `mu`, width `sigma`) falling in `[a, b)`.
pub fn gauss_1d_fraction(a: f64, b: f64, mu: f64, sigma: f64) -> f64 {
    let s = SQRT_2 * sigma;
    0.5 * (libm::erf((b - mu) / s) - libm::erf((a - mu) / s))
}

/// Convert a FWHM to a Gaussian sigma.
pub fn fwhm_to_sigma(fwhm: f64) -> f64 {
    fwhm / (2.0 * libm::sqrt(2.0 * core::f64::consts::LN_2))
}

/// Add a pixel-integrated circular Gaussian of total `flux` at `(x, y)`.
///
/// The PSF is separable, so each pixel is the product of two erf
/// differences: exact integration, no sub-sampling needed.
pub fn add_gaussian(img: &mut [f64], w: usize, h: usize, x: f64, y: f64, flux: f64, sigma: f64) {
    add_gaussian_xy(img, w, h, x, y, flux, sigma, sigma);
}

/// Add a pixel-integrated axis-aligned Gaussian of total `flux` at
/// `(x, y)`, widths `sx` and `sy` px, by erf differences per axis.
#[allow(clippy::too_many_arguments)]
fn add_gaussian_xy(
    img: &mut [f64],
    w: usize,
    h: usize,
    x: f64,
    y: f64,
    flux: f64,
    sx: f64,
    sy: f64,
) {
    let (rx, ry) = ((5.0 * sx).ceil() as i64 + 1, (5.0 * sy).ceil() as i64 + 1);
    let (cx, cy) = (x.round() as i64, y.round() as i64);
    let i0 = (cx - rx).max(0);
    let i1 = (cx + rx).min(w as i64 - 1);
    let j0 = (cy - ry).max(0);
    let j1 = (cy + ry).min(h as i64 - 1);
    if i0 > i1 || j0 > j1 {
        return;
    }
    let fx: Vec<f64> = (i0..=i1)
        .map(|i| gauss_1d_fraction(i as f64 - 0.5, i as f64 + 0.5, x, sx))
        .collect();
    for j in j0..=j1 {
        let fy = gauss_1d_fraction(j as f64 - 0.5, j as f64 + 0.5, y, sy);
        let row = &mut img[j as usize * w..];
        for (k, i) in (i0..=i1).enumerate() {
            row[i as usize] += flux * fx[k] * fy;
        }
    }
}

/// Add a streak: a Gaussian PSF moving at constant speed from `p0` to `p1`
/// during the exposure, total `flux`.
///
/// Evaluated at pixel centres in the streak's own frame: across-track a
/// Gaussian, along-track the convolution of a box of length `L` with a
/// Gaussian (an `erf` difference). Pixel-centre sampling is accurate to well
/// under 1% for `sigma >= 0.7` px; a narrower streak is integrated over
/// each pixel instead. Cut by the image edge, a sampled streak can put up
/// to about 1% more light on the image than the pixel integral (σ = 1 px). A streak shorter than 0.02 px is drawn as a
/// pixel-integrated point at its middle.
///
/// Along row `j`, with `ry = j − p0.y`, the across-track distance is
/// `d(i) = da − i uy` and the along-track position `t(i) = ta + i ux`; only
/// pixels with `|d| ≤ pad` are visited. The across-track factor
/// `exp(−d²/2σ²)` is stepped by a ratio that changes by `exp(−uy²/σ²)` per
/// pixel.
pub fn add_streak(
    img: &mut [f64],
    w: usize,
    h: usize,
    p0: (f64, f64),
    p1: (f64, f64),
    flux: f64,
    sigma: f64,
) {
    let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
    let len = libm::hypot(dx, dy);
    if len < 0.02 {
        add_gaussian(img, w, h, p0.0 + 0.5 * dx, p0.1 + 0.5 * dy, flux, sigma);
        return;
    }
    if sigma * sigma < SAMPLED_MIN_VAR {
        add_streak_integrated(img, w, h, p0, p1, flux, sigma);
        return;
    }
    let (ux, uy) = (dx / len, dy / len);
    let pad = 5.0 * sigma + 1.0;
    let i0 = (p0.0.min(p1.0) - pad).floor().max(0.0) as usize;
    let i1 = ((p0.0.max(p1.0) + pad).ceil() as i64).min(w as i64 - 1);
    let j0 = (p0.1.min(p1.1) - pad).floor().max(0.0) as usize;
    let j1 = ((p0.1.max(p1.1) + pad).ceil() as i64).min(h as i64 - 1);
    if i1 < 0 || j1 < 0 {
        return;
    }
    let s = SQRT_2 * sigma;
    let norm = flux / (len * sigma * libm::sqrt(2.0 * core::f64::consts::PI));
    let inv2s2 = 0.5 / (sigma * sigma);
    let step = libm::exp(-2.0 * inv2s2 * uy * uy);
    for j in j0..=j1 as usize {
        let ry = j as f64 - p0.1;
        let da = p0.0 * uy + ry * ux;
        let ta = -p0.0 * ux + ry * uy;
        let (mut lo, mut hi) = (i0 as f64, i1 as f64);
        if uy.abs() > 1e-12 {
            let (p, q) = ((da - pad) / uy, (da + pad) / uy);
            lo = lo.max(libm::ceil(p.min(q)));
            hi = hi.min(libm::floor(p.max(q)));
        } else if da.abs() > pad {
            continue;
        }
        if lo > hi {
            continue;
        }
        let (ia, ib) = (lo as usize, hi as usize);
        let d0 = da - ia as f64 * uy;
        let mut e = libm::exp(-inv2s2 * d0 * d0);
        let mut ratio = libm::exp(-inv2s2 * (uy * uy - 2.0 * d0 * uy));
        let row = &mut img[j * w..(j + 1) * w];
        for (k, v) in row[ia..=ib].iter_mut().enumerate() {
            let t = ta + (ia + k) as f64 * ux;
            let along = 0.5 * (erf_fast((len - t) / s) - erf_fast(-t / s));
            *v += norm * e * along;
            e *= ratio;
            ratio *= step;
        }
    }
}

/// [`add_streak`] for `sigma` under 0.7 px: the same profile integrated
/// over each pixel by 3×3-point Gauss–Legendre quadrature, then scaled so
/// the whole streak, on the image or not, carries `flux`.
fn add_streak_integrated(
    img: &mut [f64],
    w: usize,
    h: usize,
    p0: (f64, f64),
    p1: (f64, f64),
    flux: f64,
    sigma: f64,
) {
    let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
    let len = libm::hypot(dx, dy);
    let (ux, uy) = (dx / len, dy / len);
    let s = SQRT_2 * sigma;
    let pad = (5.0 * sigma).ceil() + 1.0;
    let (i0, i1) = (
        libm::floor(p0.0.min(p1.0) - pad) as i64,
        libm::ceil(p0.0.max(p1.0) + pad) as i64,
    );
    let (j0, j1) = (
        libm::floor(p0.1.min(p1.1) - pad) as i64,
        libm::ceil(p0.1.max(p1.1) + pad) as i64,
    );
    let mut vals = Vec::new();
    let mut sum = 0.0;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let mut v = 0.0;
            for &(oy, wy) in &GL3 {
                for &(ox, wx) in &GL3 {
                    let (rx, ry) = (i as f64 + ox - p0.0, j as f64 + oy - p0.1);
                    let t = rx * ux + ry * uy;
                    let d = ry * ux - rx * uy;
                    let along = 0.5 * (erf_fast((len - t) / s) - erf_fast(-t / s));
                    v += wx * wy * libm::exp(-0.5 * d * d / (sigma * sigma)) * along;
                }
            }
            sum += v;
            vals.push((i, j, v));
        }
    }
    deposit_normalised(img, w, h, vals, sum, flux);
}

/// A PSF made of concentric Gaussians: a core plus optional wider wings,
/// each with a share of the flux (a multi-Gaussian expansion; Bendinelli,
/// Astrophys. J. 366, 599, 1991; Cappellari, MNRAS 333, 400, 2002). Every component is
/// separable, so points and streaks stay exactly pixel-integrated.
#[derive(Clone, Debug, PartialEq)]
pub struct Psf {
    /// `(fraction, sigma_px)` per component; fractions sum to 1.
    pub comps: Vec<(f64, f64)>,
    /// How the PSF changes across the field, if it does.
    pub field: Option<Field>,
}

/// PSF change across the field: every component's σ grows with distance
/// from the optical axis, separately along the radius and across it
/// (quadratic in radius, as coma and astigmatism grow; Born & Wolf,
/// *Principles of Optics*, 7th ed., section 9.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Field {
    /// Optical axis, pixels.
    pub axis: (f64, f64),
    /// Radius where the growth is reached (usually the half-diagonal), px.
    pub r_ref: f64,
    /// Fractional growth of σ along the radius at `r_ref`.
    pub radial: f64,
    /// Fractional growth of σ across the radius at `r_ref`.
    pub tangential: f64,
}

/// 2×2 covariance `[sxx, sxy, syy]`.
type Cov = [f64; 3];

impl Field {
    /// Covariance of a component of core `sigma` at pixel `(x, y)`: `σ`
    /// grows by `1 + growth · ρ²`, with `ρ` the distance from the axis in
    /// units of `r_ref`, separately along and across the radius.
    pub(crate) fn cov(&self, sigma: f64, x: f64, y: f64) -> Cov {
        let (dx, dy) = (x - self.axis.0, y - self.axis.1);
        let rho2 = (dx * dx + dy * dy) / (self.r_ref * self.r_ref);
        let sr = sigma * (1.0 + self.radial * rho2);
        let st = sigma * (1.0 + self.tangential * rho2);
        let r = libm::hypot(dx, dy);
        let (c, s) = if r > 0.0 {
            (dx / r, dy / r)
        } else {
            (1.0, 0.0)
        };
        let (a, b) = (sr * sr, st * st);
        [
            a * c * c + b * s * s,
            (a - b) * c * s,
            a * s * s + b * c * c,
        ]
    }
}

/// Multi-Gaussian expansion of a normalised 2-D Moffat profile
/// (Moffat, A&A 3, 455, 1969): weights (summing to 1) and sigmas, fitted by
/// non-negative least squares to the radial profile out to 12 FWHM.
///
/// Ten sigmas spaced geometrically from 0.2 to 12 FWHM; the rows (240 radii)
/// are weighted by `r` (area) over the target, so the fit minimises the
/// relative error.
pub fn moffat_mge(fwhm: f64, beta: f64) -> Vec<(f64, f64)> {
    let beta = beta.max(1.05);
    let alpha = fwhm / (2.0 * libm::sqrt(libm::pow(2.0, 1.0 / beta) - 1.0));
    let norm = (beta - 1.0) / (core::f64::consts::PI * alpha * alpha);
    let moffat = |r: f64| norm * libm::pow(1.0 + (r / alpha) * (r / alpha), -beta);
    let n = 10;
    let sig: Vec<f64> = (0..n)
        .map(|k| 0.2 * fwhm * libm::pow(60.0, f64::from(k) / f64::from(n - 1)))
        .collect();
    let rs: Vec<f64> = (0..240)
        .map(|k| 12.0 * fwhm * f64::from(k) / 239.0)
        .collect();
    let a: Vec<Vec<f64>> = rs
        .iter()
        .map(|&r| {
            let wt = (r + 0.1 * fwhm) / moffat(r);
            sig.iter()
                .map(|&s| {
                    wt * libm::exp(-r * r / (2.0 * s * s)) / (2.0 * core::f64::consts::PI * s * s)
                })
                .collect()
        })
        .collect();
    let bvec: Vec<f64> = rs.iter().map(|&r| r + 0.1 * fwhm).collect();
    let wts = nnls(&a, &bvec);
    let total: f64 = wts.iter().sum();
    sig.into_iter()
        .zip(wts)
        .filter(|(_, w)| *w > 0.0)
        .map(|(s, w)| (w / total, s))
        .collect()
}

/// Non-negative least squares by projected coordinate descent (small
/// problems only).
fn nnls(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a[0].len();
    let ata: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| a.iter().map(|row| row[i] * row[j]).sum())
                .collect()
        })
        .collect();
    let atb: Vec<f64> = (0..n)
        .map(|i| a.iter().zip(b).map(|(row, y)| row[i] * y).sum())
        .collect();
    let mut x = vec![0.0; n];
    for _ in 0..5000 {
        let mut change = 0.0f64;
        for i in 0..n {
            if ata[i][i] <= 0.0 {
                continue;
            }
            let g: f64 = atb[i] - (0..n).map(|j| ata[i][j] * x[j]).sum::<f64>();
            let new = (x[i] + g / ata[i][i]).max(0.0);
            change = change.max((new - x[i]).abs() / (x[i].abs() + 1e-12));
            x[i] = new;
        }
        if change < 1e-10 {
            break;
        }
    }
    x
}

/// Gauss–Legendre nodes and weights on [−½, ½], 3 points.
const GL3: [(f64, f64); 3] = [
    (-0.387_298_334_620_741_7, 5.0 / 18.0),
    (0.0, 8.0 / 18.0),
    (0.387_298_334_620_741_7, 5.0 / 18.0),
];

/// The whitening matrix `W = Σ^(−1/2)` of covariance `c`, through the eigen
/// decomposition of the 2×2 matrix, and its determinant (the smaller
/// eigenvalue floored at 1e-12).
fn whiten(c: &Cov) -> ([[f64; 2]; 2], f64) {
    let [a, b, d] = *c;
    let tr = a + d;
    let disc = libm::sqrt(((a - d) * 0.5).powi(2) + b * b);
    let (l1, l2) = (0.5 * tr + disc, 0.5 * tr - disc);
    let th = 0.5 * libm::atan2(2.0 * b, a - d);
    let (s, co) = libm::sincos(th);
    let (i1, i2) = (1.0 / libm::sqrt(l1), 1.0 / libm::sqrt(l2.max(1e-12)));
    let w = [
        [co * co * i1 + s * s * i2, co * s * (i1 - i2)],
        [co * s * (i1 - i2), s * s * i1 + co * co * i2],
    ];
    (w, i1 * i2)
}

/// `W · (dx, dy)`.
fn apply(wm: &[[f64; 2]; 2], dx: f64, dy: f64) -> (f64, f64) {
    (wm[0][0] * dx + wm[0][1] * dy, wm[1][0] * dx + wm[1][1] * dy)
}

/// Add `flux × v / sum` at each `(i, j, v)` that falls on the `w × h` image,
/// so the values (computed over the whole footprint, inside or not) carry
/// `flux` in all. Adds nothing when `sum` is not positive.
fn deposit_normalised(
    img: &mut [f64],
    w: usize,
    h: usize,
    vals: Vec<(i64, i64, f64)>,
    sum: f64,
    flux: f64,
) {
    if sum <= 0.0 {
        return;
    }
    for (i, j, v) in vals {
        if i >= 0 && j >= 0 && (i as usize) < w && (j as usize) < h {
            img[j as usize * w + i as usize] += flux * v / sum;
        }
    }
}

/// Add an elliptical Gaussian point source, pixel-integrated by 3×3
/// Gauss–Legendre quadrature and normalised to conserve flux.
fn add_elliptical(img: &mut [f64], w: usize, h: usize, x: f64, y: f64, flux: f64, c: &Cov) {
    let (wm, det) = whiten(c);
    let r = (5.0 * libm::sqrt(c[0].max(c[2]))).ceil() as i64 + 1;
    let (cx, cy) = (x.round() as i64, y.round() as i64);
    let mut vals = Vec::new();
    let mut sum = 0.0;
    for j in cy - r..=cy + r {
        for i in cx - r..=cx + r {
            let mut v = 0.0;
            for &(oy, wy) in &GL3 {
                for &(ox, wx) in &GL3 {
                    let (dx, dy) = (i as f64 + ox - x, j as f64 + oy - y);
                    let (u, q) = apply(&wm, dx, dy);
                    v += wx * wy * libm::exp(-0.5 * (u * u + q * q));
                }
            }
            v *= det / (2.0 * core::f64::consts::PI);
            sum += v;
            vals.push((i, j, v));
        }
    }
    deposit_normalised(img, w, h, vals, sum, flux);
}

/// Add a streak of an elliptical Gaussian: in coordinates whitened by its
/// covariance it is a round unit Gaussian moving in a straight line, whose
/// line integral is closed-form. Pixel-centre sampled out to 6σ across the
/// track, normalised to conserve flux; shorter than 0.02 whitened units, a
/// point at the middle.
pub(crate) fn add_elliptical_streak(
    img: &mut [f64],
    w: usize,
    h: usize,
    p0: (f64, f64),
    p1: (f64, f64),
    flux: f64,
    c: &Cov,
) {
    let (wm, det) = whiten(c);
    let (sx, sy) = apply(&wm, p1.0 - p0.0, p1.1 - p0.1);
    let len = libm::hypot(sx, sy);
    if len < 0.02 {
        add_elliptical(img, w, h, 0.5 * (p0.0 + p1.0), 0.5 * (p0.1 + p1.1), flux, c);
        return;
    }
    let (ux, uy) = (sx / len, sy / len);
    let pad = 5.0 * libm::sqrt(c[0].max(c[2])) + 1.0;
    let x0 = (p0.0.min(p1.0) - pad).floor() as i64;
    let x1 = (p0.0.max(p1.0) + pad).ceil() as i64;
    let y0 = (p0.1.min(p1.1) - pad).floor() as i64;
    let y1 = (p0.1.max(p1.1) + pad).ceil() as i64;
    let s2 = core::f64::consts::SQRT_2;
    let mut vals = Vec::new();
    let mut sum = 0.0;
    for j in y0..=y1 {
        for i in x0..=x1 {
            let (a, b) = apply(&wm, i as f64 - p0.0, j as f64 - p0.1);
            let t = a * ux + b * uy;
            let d = -a * uy + b * ux;
            if d.abs() > 6.0 {
                continue;
            }
            let along = 0.5 * (libm::erf(t / s2) - libm::erf((t - len) / s2));
            let v = det * libm::exp(-0.5 * d * d) * along
                / (libm::sqrt(2.0 * core::f64::consts::PI) * len);
            sum += v;
            vals.push((i, j, v));
        }
    }
    deposit_normalised(img, w, h, vals, sum, flux);
}

impl Psf {
    /// A single Gaussian.
    pub fn gaussian(sigma: f64) -> Self {
        Psf {
            comps: vec![(1.0, sigma)],
            field: None,
        }
    }

    /// A core of FWHM `core_fwhm` plus wings `(fraction, fwhm)`; the core
    /// keeps what the wings do not take.
    pub fn with_wings(core_fwhm: f64, wings: &[(f64, f64)]) -> Self {
        let taken: f64 = wings.iter().map(|w| w.0).sum();
        let mut comps = vec![((1.0 - taken).max(0.0), fwhm_to_sigma(core_fwhm))];
        comps.extend(wings.iter().map(|&(f, fw)| (f, fwhm_to_sigma(fw))));
        Psf { comps, field: None }
    }

    /// Add a Moffat tail carrying `fraction` of the flux, taken from the core.
    #[must_use]
    pub fn with_tail(mut self, fraction: f64, fwhm: f64, beta: f64) -> Self {
        if fraction > 0.0 {
            self.comps[0].0 = (self.comps[0].0 - fraction).max(0.0);
            self.comps.extend(
                moffat_mge(fwhm, beta)
                    .into_iter()
                    .map(|(wt, s)| (fraction * wt, s)),
            );
        }
        self
    }

    /// Make the PSF change across the field (dropped when it has no growth).
    #[must_use]
    pub fn with_field(mut self, field: Option<Field>) -> Self {
        self.field = field.filter(|f| f.radial != 0.0 || f.tangential != 0.0);
        self
    }

    /// Convolve every component with a Gaussian blur of `sigma` pixels.
    #[must_use]
    pub fn blurred(mut self, sigma: f64) -> Self {
        if sigma > 0.0 {
            for c in &mut self.comps {
                c.1 = libm::hypot(c.1, sigma);
            }
        }
        self
    }

    /// Cramér–Rao lower bounds on the centroid (x, y, pixels) and flux
    /// (electrons) of a point source of `flux` at (x, y), with per-pixel
    /// variance `var(i, j)` in electrons². Background is taken as known.
    ///
    /// The Fisher matrix for (x, y, F) is Σ ∂μ/∂θᵢ ∂μ/∂θⱼ / var over the
    /// pixels (e.g. Mendez, Silva & Lobos, PASP 126, 798, 2014), with μ the
    /// pixel-integrated PSF; derivatives by central differences. The bounds
    /// are the square roots of the diagonal of its inverse; `None` when the
    /// matrix is singular or a variance comes out non-positive. A field
    /// PSF takes its shape at `(x, y)` on the frame, held fixed for the
    /// derivatives.
    pub fn crlb(
        &self,
        x: f64,
        y: f64,
        flux: f64,
        var: impl Fn(i64, i64) -> f64,
    ) -> Option<(f64, f64, f64)> {
        let r = (6.0 * self.widest_sigma(x, y)).ceil() as i64 + 2;
        let n = (2 * r + 1) as usize;
        let (ox, oy) = (x.round() as i64 - r, y.round() as i64 - r);
        let render = |dx: f64, dy: f64| {
            let mut b = vec![0.0; n * n];
            self.point_shaped_at(
                &mut b,
                n,
                n,
                x - ox as f64 + dx,
                y - oy as f64 + dy,
                1.0,
                (x, y),
            );
            b
        };
        let h = 1e-3;
        let (px, mx, py, my, p0) = (
            render(h, 0.0),
            render(-h, 0.0),
            render(0.0, h),
            render(0.0, -h),
            render(0.0, 0.0),
        );
        let mut f = [[0.0f64; 3]; 3];
        for k in 0..n * n {
            let (i, j) = ((k % n) as i64 + ox, (k / n) as i64 + oy);
            let v = var(i, j);
            if v.is_nan() || v <= 0.0 {
                continue;
            }
            let g = [
                flux * (px[k] - mx[k]) / (2.0 * h),
                flux * (py[k] - my[k]) / (2.0 * h),
                p0[k],
            ];
            for a in 0..3 {
                for b in 0..3 {
                    f[a][b] += g[a] * g[b] / v;
                }
            }
        }
        let det = f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
            - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
            + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
        if det.is_nan() || det <= 0.0 {
            return None;
        }
        let c00 = (f[1][1] * f[2][2] - f[1][2] * f[2][1]) / det;
        let c11 = (f[0][0] * f[2][2] - f[0][2] * f[2][0]) / det;
        let c22 = (f[0][0] * f[1][1] - f[0][1] * f[1][0]) / det;
        (c00 > 0.0 && c11 > 0.0 && c22 > 0.0)
            .then(|| (libm::sqrt(c00), libm::sqrt(c11), libm::sqrt(c22)))
    }

    /// The core's sigma, pixels.
    pub fn core_sigma(&self) -> f64 {
        self.comps[0].1
    }

    /// Add a point source of `flux` at `(x, y)`.
    pub fn point(&self, img: &mut [f64], w: usize, h: usize, x: f64, y: f64, flux: f64) {
        self.point_shaped_at(img, w, h, x, y, flux, (x, y));
    }

    /// Add a point source of `flux` at `(x, y)` in `img`, with the field
    /// PSF's shape taken at frame pixel `at` (which differs from `(x, y)`
    /// when `img` is a cut-out).
    #[allow(clippy::too_many_arguments)]
    fn point_shaped_at(
        &self,
        img: &mut [f64],
        w: usize,
        h: usize,
        x: f64,
        y: f64,
        flux: f64,
        at: (f64, f64),
    ) {
        for &(f, s) in &self.comps {
            if f <= 0.0 {
                continue;
            }
            match &self.field {
                None => add_gaussian(img, w, h, x, y, flux * f, s),
                Some(fd) => add_elliptical(img, w, h, x, y, flux * f, &fd.cov(s, at.0, at.1)),
            }
        }
    }

    /// The largest σ, pixels, of any component carrying light, along its
    /// widest axis at frame pixel `(x, y)`.
    fn widest_sigma(&self, x: f64, y: f64) -> f64 {
        self.comps
            .iter()
            .filter(|c| c.0 > 1e-3)
            .map(|&(_, s)| match &self.field {
                None => s,
                Some(fd) => {
                    let [a, b, c] = fd.cov(s, x, y);
                    let half = 0.5 * (a - c);
                    libm::sqrt(0.5 * (a + c) + libm::hypot(half, b))
                }
            })
            .fold(0.0, f64::max)
    }

    /// Add a straight streak (the field PSF is taken at its middle).
    pub fn streak(
        &self,
        img: &mut [f64],
        w: usize,
        h: usize,
        p0: (f64, f64),
        p1: (f64, f64),
        flux: f64,
    ) {
        for &(f, s) in &self.comps {
            if f <= 0.0 {
                continue;
            }
            match &self.field {
                None => add_streak(img, w, h, p0, p1, flux * f, s),
                Some(fd) => {
                    let c = fd.cov(s, 0.5 * (p0.0 + p1.0), 0.5 * (p0.1 + p1.1));
                    add_elliptical_streak(img, w, h, p0, p1, flux * f, &c);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Far off axis, the bound for a field PSF equals the bound for a plain
    /// PSF of the σ the field gives there.
    #[test]
    fn crlb_uses_the_field_psf_at_the_frame_position() {
        let field = Field {
            axis: (512.0, 512.0),
            r_ref: 724.0,
            radial: 1.0,
            tangential: 1.0,
        };
        let (x, y) = (900.3, 880.6);
        let psf = Psf::gaussian(1.0).with_field(Some(field));
        let grown = field.cov(1.0, x, y)[0].sqrt();
        assert!(grown > 1.4, "{grown}");
        let plain = Psf::gaussian(grown);
        let var = |_: i64, _: i64| 100.0;
        let a = psf.crlb(x, y, 5_000.0, var).unwrap();
        let b = plain.crlb(x, y, 5_000.0, var).unwrap();
        for (u, v) in [(a.0, b.0), (a.1, b.1), (a.2, b.2)] {
            assert!((u - v).abs() < 1e-6 * v, "field {a:?}, plain {b:?}");
        }
    }

    #[test]
    fn gaussian_conserves_flux() {
        let (w, h) = (64, 64);
        let mut img = vec![0.0; w * h];
        add_gaussian(&mut img, w, h, 31.3, 30.7, 1000.0, 1.5);
        let sum: f64 = img.iter().sum();
        assert!((sum - 1000.0).abs() < 1e-6, "sum {sum}");
    }

    #[test]
    fn wings_conserve_flux() {
        let (w, h) = (200, 200);
        let mut img = vec![0.0; w * h];
        let psf = Psf::with_wings(2.46, &[(0.146, 6.5), (0.02, 15.8)]);
        psf.point(&mut img, w, h, 99.5, 100.2, 1e5);
        let sum: f64 = img.iter().sum();
        assert!((sum - 1e5).abs() < 1e-3 * 1e5, "{sum}");
        assert!((psf.comps[0].0 - 0.834).abs() < 1e-12);
    }

    #[test]
    fn moffat_expansion_matches_the_profile() {
        let (fwhm, beta) = (3.0, 2.5);
        let mge = moffat_mge(fwhm, beta);
        let alpha = fwhm / (2.0 * libm::sqrt(libm::pow(2.0, 1.0 / beta) - 1.0));
        let norm = (beta - 1.0) / (core::f64::consts::PI * alpha * alpha);
        for r in [0.0, 1.5, 4.0, 10.0, 20.0] {
            let m = norm * libm::pow(1.0 + (r / alpha) * (r / alpha), -beta);
            let g: f64 = mge
                .iter()
                .map(|&(w, s)| {
                    w * libm::exp(-r * r / (2.0 * s * s)) / (2.0 * core::f64::consts::PI * s * s)
                })
                .sum();
            assert!((g / m - 1.0).abs() < 0.1, "r {r}: {g} vs {m}");
        }
    }

    /// A source on the +x radius at ρ = 0.75: σ along x (radial) is
    /// 1 + 0.75² = 1.5625, σ along y is 1; pixel width adds 1/12 in variance.
    #[test]
    fn field_psf_elongates_radially_and_conserves_flux() {
        let (w, h) = (200, 200);
        let psf = Psf::gaussian(1.0).with_field(Some(Field {
            axis: (100.0, 100.0),
            r_ref: 100.0,
            radial: 1.0,
            tangential: 0.0,
        }));
        let mut img = vec![0.0; w * h];
        psf.point(&mut img, w, h, 175.0, 100.0, 1000.0);
        let sum: f64 = img.iter().sum();
        assert!((sum - 1000.0).abs() < 1e-6, "{sum}");
        let (mut sxx, mut syy) = (0.0, 0.0);
        for j in 0..h {
            for i in 0..w {
                let v = img[j * w + i];
                sxx += v * (i as f64 - 175.0).powi(2);
                syy += v * (j as f64 - 100.0).powi(2);
            }
        }
        assert!((libm::sqrt(sxx / sum) - libm::sqrt(1.5625f64.powi(2) + 1.0 / 12.0)).abs() < 0.03);
        assert!((libm::sqrt(syy / sum) - libm::sqrt(1.0 + 1.0 / 12.0)).abs() < 0.03);
        let mut img = vec![0.0; w * h];
        psf.streak(&mut img, w, h, (150.0, 60.0), (160.0, 140.0), 500.0);
        assert!((img.iter().sum::<f64>() - 500.0).abs() < 1e-6);
    }

    /// Faint star, flat background b ≫ signal: `σ_x → σ² √(8π b) / F` and
    /// `σ_F → √(4π σ² b)` (King, PASP 95, 163, 1983; Mendez et al. 2014), with
    /// σ² + 1/12 for pixel integration.
    #[test]
    fn crlb_matches_the_background_limited_formula() {
        let s = 2.0;
        let psf = Psf::gaussian(s);
        let (flux, b) = (500.0, 400.0);
        let (sx, sy, sf) = psf.crlb(50.3, 49.8, flux, |_, _| b).unwrap();
        let se2 = s * s + 1.0 / 12.0;
        let want = se2 * libm::sqrt(8.0 * core::f64::consts::PI * b) / flux;
        assert!(
            (sx / want - 1.0).abs() < 0.03 && (sy / want - 1.0).abs() < 0.03,
            "{sx} {want}"
        );
        assert!(
            (sf / libm::sqrt(4.0 * core::f64::consts::PI * se2 * b) - 1.0).abs() < 0.03,
            "{sf}"
        );
    }

    #[test]
    fn streak_conserves_flux() {
        let (w, h) = (128, 128);
        let mut img = vec![0.0; w * h];
        add_streak(&mut img, w, h, (20.2, 30.9), (100.6, 90.1), 5000.0, 1.2);
        let sum: f64 = img.iter().sum();
        assert!((sum - 5000.0).abs() / 5000.0 < 0.005, "sum {sum}");
    }
}
