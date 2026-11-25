//! Moving sources: trace a path in pixels over the exposure and draw it as
//! a chain of exact straight streaks.
//!
//! The exposure is split adaptively: a piece is accepted when the true
//! position at its middle time lies within `tol_px` of the straight chord,
//! and its rate (electrons per second) at the middle is within 1% of the
//! mean of its ends. A straight, steady path costs nine evaluations
//! (four pieces, each checked at its middle); curved paths or a shadow
//! crossing get more pieces only where needed.

use crate::psf::Psf;

/// Position and brightness at one time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathPoint {
    /// Seconds from shutter open.
    pub s: f64,
    /// Pixel position `(x, y)`, `None` if the source cannot be projected
    /// (behind the camera).
    pub xy: Option<(f64, f64)>,
    /// Signal rate, electrons per second.
    pub rate: f64,
}

/// How finely moving sources are sampled.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sampling {
    /// Largest distance between the true path and a straight piece, px.
    pub pixel_tolerance: f64,
    /// Largest relative brightness error over a piece.
    pub flux_relative_tolerance: f64,
    /// Longest piece, s (default: none).
    pub max_step_s: Option<f64>,
    /// Deepest halving of a piece.
    pub max_depth: u32,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling {
            pixel_tolerance: 0.01,
            flux_relative_tolerance: 0.01,
            max_step_s: None,
            max_depth: 14,
        }
    }
}

/// Whether the piece from `a` to `b` is straight and steady enough: its
/// midpoint `m` lies within the pixel tolerance of the chord's middle, and
/// its rate within the relative tolerance of the ends' mean. Both or none
/// of the three points must project.
fn close(a: &PathPoint, b: &PathPoint, m: &PathPoint, smp: &Sampling) -> bool {
    let rate_ok = (m.rate - 0.5 * (a.rate + b.rate)).abs()
        <= smp.flux_relative_tolerance * a.rate.max(b.rate).max(m.rate) + 1e-12;
    let pos_ok = match (a.xy, b.xy, m.xy) {
        (Some(pa), Some(pb), Some(pm)) => {
            libm::hypot(pm.0 - 0.5 * (pa.0 + pb.0), pm.1 - 0.5 * (pa.1 + pb.1))
                <= smp.pixel_tolerance
        }
        (None, None, None) => true,
        _ => false,
    };
    rate_ok && pos_ok
}

/// Appends the samples after `a` up to and including `b`, halving the
/// piece depth-first (earlier halves first, so samples stay in time order)
/// until [`close`] accepts it or `max_depth` is reached.
fn refine<F: FnMut(f64) -> PathPoint>(
    f: &mut F,
    a: PathPoint,
    b: PathPoint,
    smp: &Sampling,
    depth: u32,
    out: &mut Vec<PathPoint>,
) {
    let m = f(0.5 * (a.s + b.s));
    if depth >= smp.max_depth || close(&a, &b, &m, smp) {
        out.push(b);
    } else {
        refine(f, a, m, smp, depth + 1, out);
        refine(f, m, b, smp, depth + 1, out);
    }
}

/// Pieces the exposure is first cut into; each is then refined.
const START_PIECES: usize = 4;

/// Samples `f` over `[0, duration]` seconds, refined until each piece is
/// straight to `tol_px` pixels (default [`Sampling`] otherwise).
pub fn trace<F: FnMut(f64) -> PathPoint>(f: F, duration: f64, tol_px: f64) -> Vec<PathPoint> {
    let smp = Sampling {
        pixel_tolerance: tol_px,
        ..Sampling::default()
    };
    trace_with(f, duration, &smp, START_PIECES)
}

/// Samples `f` over `[0, duration]` seconds, starting from at least
/// `pieces` equal pieces (more if `max_step_s` asks, at most 100 000), each
/// refined to the tolerances of `smp`.
pub fn trace_with<F: FnMut(f64) -> PathPoint>(
    mut f: F,
    duration: f64,
    smp: &Sampling,
    pieces: usize,
) -> Vec<PathPoint> {
    let by_step = smp
        .max_step_s
        .filter(|s| *s > 0.0)
        .map_or(1, |s| libm::ceil(duration / s) as usize);
    let pieces = pieces.max(by_step).clamp(1, 100_000);
    let mut prev = f(0.0);
    let mut out = vec![prev];
    for k in 1..=pieces {
        let next = f(duration * k as f64 / pieces as f64);
        refine(&mut f, prev, next, smp, 1, &mut out);
        prev = next;
    }
    out
}

/// Largest extent of a chunk of short pieces, in units of the PSF
/// component's σ. A chunk of extent ρσ drawn as one Gaussian with matched
/// first and second moments differs from the exact chain of streaks only
/// at fourth order, about (ρ/2)⁴/8 of the peak: 6e-6 at ρ = 0.3.
const CHUNK_SIGMA: f64 = 0.3;

/// A straight piece of a path: start, end, electrons.
type Piece = ((f64, f64), (f64, f64), f64);

/// Draws a traced path into the `w` × `h` image `img` and returns the total
/// electrons drawn.
///
/// Per PSF component, runs of pieces shorter than `CHUNK_SIGMA` (0.3) σ are
/// merged into one Gaussian whose mean and covariance are the PSF's plus
/// the run's own (each piece a uniform line: mean at its middle, variance
/// L²/12 along it); longer pieces are drawn as exact streaks. A jittering
/// star of 256 tiny pieces costs a few Gaussians instead of 256 streaks.
pub fn draw(img: &mut [f64], w: usize, h: usize, pts: &[PathPoint], psf: &Psf) -> f64 {
    let segs: Vec<Piece> = pts
        .windows(2)
        .filter_map(|p| {
            let flux = 0.5 * (p[0].rate + p[1].rate) * (p[1].s - p[0].s);
            match (p[0].xy, p[1].xy) {
                (Some(a), Some(b)) if flux > 0.0 => Some((a, b, flux)),
                _ => None,
            }
        })
        .collect();
    let total: f64 = segs.iter().map(|s| s.2).sum();
    for &(frac, sigma) in &psf.comps {
        if frac <= 0.0 {
            continue;
        }
        draw_component(img, w, h, &segs, frac, sigma, psf);
    }
    total
}

/// Electron-weighted moments of a run of pieces.
///
/// Moments are taken about the run's first point `o`, not the image
/// origin, so they do not lose precision to cancellation far from 0.
#[derive(Default)]
struct Chunk {
    /// Origin of the moments: the first piece's start.
    o: (f64, f64),
    /// Total electrons.
    w: f64,
    mx: f64,
    my: f64,
    sxx: f64,
    sxy: f64,
    syy: f64,
    /// Bounding box of the run.
    lo: (f64, f64),
    hi: (f64, f64),
}

impl Chunk {
    /// Bounding box of the run grown to take in the piece `a`–`b`.
    fn grown(&self, a: (f64, f64), b: (f64, f64)) -> ((f64, f64), (f64, f64)) {
        let lo = (self.lo.0.min(a.0).min(b.0), self.lo.1.min(a.1).min(b.1));
        let hi = (self.hi.0.max(a.0).max(b.0), self.hi.1.max(a.1).max(b.1));
        (lo, hi)
    }

    /// Whether the piece `a`–`b` keeps the run's extent within `limit` px.
    fn fits(&self, a: (f64, f64), b: (f64, f64), limit: f64) -> bool {
        if self.w == 0.0 {
            return true;
        }
        let (lo, hi) = self.grown(a, b);
        libm::hypot(hi.0 - lo.0, hi.1 - lo.1) <= limit
    }

    /// Adds the piece `a`–`b` carrying `f` electrons, as a uniform line.
    fn add(&mut self, a: (f64, f64), b: (f64, f64), f: f64) {
        if self.w == 0.0 {
            self.o = a;
            self.lo = (a.0.min(b.0), a.1.min(b.1));
            self.hi = (a.0.max(b.0), a.1.max(b.1));
        } else {
            (self.lo, self.hi) = self.grown(a, b);
        }
        let (mx, my) = (0.5 * (a.0 + b.0) - self.o.0, 0.5 * (a.1 + b.1) - self.o.1);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        self.w += f;
        self.mx += f * mx;
        self.my += f * my;
        self.sxx += f * (mx * mx + dx * dx / 12.0);
        self.sxy += f * (mx * my + dx * dy / 12.0);
        self.syy += f * (my * my + dy * dy / 12.0);
    }

    /// Draws the run as one Gaussian (PSF covariance plus the run's own),
    /// carrying `frac` of its electrons, then empties it.
    fn flush(&mut self, img: &mut [f64], w: usize, h: usize, frac: f64, sigma: f64, psf: &Psf) {
        if self.w > 0.0 {
            let (mx, my) = (self.mx / self.w, self.my / self.w);
            let cxx = (self.sxx / self.w - mx * mx).max(0.0);
            let cyy = (self.syy / self.w - my * my).max(0.0);
            let cxy = self.sxy / self.w - mx * my;
            let (mx, my) = (mx + self.o.0, my + self.o.1);
            let base = match &psf.field {
                None => [sigma * sigma, 0.0, sigma * sigma],
                Some(fd) => fd.cov(sigma, mx, my),
            };
            let c = [base[0] + cxx, base[1] + cxy, base[2] + cyy];
            crate::psf::add_gauss_cov(img, w, h, (mx, my), &c, frac * self.w);
        }
        *self = Chunk::default();
    }
}

/// Draws one PSF component (`frac` of the flux, width `sigma` px) of the
/// pieces `segs`, in order: long pieces as exact streaks, runs of short
/// ones as [`Chunk`]s.
fn draw_component(
    img: &mut [f64],
    w: usize,
    h: usize,
    segs: &[Piece],
    frac: f64,
    sigma: f64,
    psf: &Psf,
) {
    let limit = CHUNK_SIGMA * sigma;
    let mut ch = Chunk::default();
    for &(a, b, f) in segs {
        let len = libm::hypot(b.0 - a.0, b.1 - a.1);
        if len > limit {
            ch.flush(img, w, h, frac, sigma, psf);
            match &psf.field {
                None => crate::psf::add_streak(img, w, h, a, b, frac * f, sigma),
                Some(fd) => {
                    let c = fd.cov(sigma, 0.5 * (a.0 + b.0), 0.5 * (a.1 + b.1));
                    crate::psf::add_elliptical_streak(img, w, h, a, b, frac * f, &c);
                }
            }
            continue;
        }
        if !ch.fits(a, b, limit) {
            ch.flush(img, w, h, frac, sigma, psf);
        }
        ch.add(a, b, f);
    }
    ch.flush(img, w, h, frac, sigma, psf);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact reference: every piece an exact streak, libm maths.
    fn reference(img: &mut [f64], w: usize, h: usize, pts: &[PathPoint], psf: &Psf) {
        for p in pts.windows(2) {
            let flux = 0.5 * (p[0].rate + p[1].rate) * (p[1].s - p[0].s);
            let (a, b) = (p[0].xy.unwrap(), p[1].xy.unwrap());
            for &(f, sg) in &psf.comps {
                let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                let len = libm::hypot(dx, dy);
                let s = core::f64::consts::SQRT_2 * sg;
                let norm = f * flux / (len * sg * libm::sqrt(2.0 * core::f64::consts::PI));
                let (ux, uy) = (dx / len, dy / len);
                for j in 0..h {
                    for i in 0..w {
                        let (rx, ry) = (i as f64 - a.0, j as f64 - a.1);
                        let t = rx * ux + ry * uy;
                        let d = -rx * uy + ry * ux;
                        let along = 0.5 * (libm::erf((len - t) / s) - libm::erf(-t / s));
                        img[j * w + i] += norm * libm::exp(-0.5 * d * d / (sg * sg)) * along;
                    }
                }
            }
        }
    }

    /// A jittering star (pieces of ~0.05 px wandering ~0.5 px) then a fast
    /// 6 px slide, whose long pieces stay exact streaks, matches the exact
    /// reference to 2e-5 of the peak.
    #[test]
    fn chunked_jitter_matches_exact_streaks() {
        let (w, h) = (80, 80);
        let psf = Psf {
            comps: vec![(0.76, 0.85), (0.2, 2.1), (0.04, 5.1)],
            field: None,
        };
        let mut rng = crate::rng::Rng::new(3);
        let (mut x, mut y) = (40.3, 39.6);
        let mut pts = vec![];
        for k in 0..=300 {
            if k > 0 {
                let big = (200..210).contains(&k);
                x += if big { 0.6 } else { 0.04 * rng.normal() };
                y += if big { 0.1 } else { 0.04 * rng.normal() };
            }
            pts.push(PathPoint {
                s: f64::from(k) / 100.0,
                xy: Some((x, y)),
                rate: 1e6,
            });
        }
        let mut exact = vec![0.0; w * h];
        reference(&mut exact, w, h, &pts, &psf);
        let mut fast = vec![0.0; w * h];
        let total = draw(&mut fast, w, h, &pts, &psf);
        let peak = exact.iter().copied().fold(0.0, f64::max);
        let worst = exact
            .iter()
            .zip(&fast)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!("chunked: worst {:.2e} of peak", worst / peak);
        assert!(worst < 2e-5 * peak, "worst {worst} peak {peak}");
        let (se, sf): (f64, f64) = (exact.iter().sum(), fast.iter().sum());
        assert!((sf - se).abs() < 1e-6 * se, "{se} {sf}");
        assert!((total - 3e6).abs() < 1e-3);
    }

    /// A star that drifts 1e-6 px keeps the flux of the same star held
    /// still at every FWHM and sub-pixel phase. Below σ = 0.7 px it also
    /// gives the same pixels, and the same flux when cut by the image edge;
    /// wider, sampling at pixel centres puts up to about 1% more light on
    /// the image there (a documented limit of [`crate::psf::add_streak`]).
    #[test]
    fn near_still_star_keeps_the_flux_and_pixels_of_a_still_one() {
        let (w, h) = (24, 24);
        for fwhm in [0.6, 1.0, 1.5, 2.5] {
            let sigma = crate::psf::fwhm_to_sigma(fwhm);
            let psf = Psf::gaussian(sigma);
            for k in 0..8 {
                let phase = f64::from(k) / 8.0;
                for (x, y) in [(12.0 + phase, 11.0 + 0.7 * phase), (0.1 + phase, 12.3)] {
                    let mut still = vec![0.0; w * h];
                    crate::psf::add_gaussian(&mut still, w, h, x, y, 1e4, sigma);
                    let pts = [
                        PathPoint {
                            s: 0.0,
                            xy: Some((x, y)),
                            rate: 1e4,
                        },
                        PathPoint {
                            s: 1.0,
                            xy: Some((x + 1e-6, y + 1e-6)),
                            rate: 1e4,
                        },
                    ];
                    let mut moved = vec![0.0; w * h];
                    draw(&mut moved, w, h, &pts, &psf);
                    let (a, b): (f64, f64) = (still.iter().sum(), moved.iter().sum());
                    let tol = if sigma < 0.7 || x > 1.5 { 1e-4 } else { 1.5e-2 };
                    assert!(
                        (a - b).abs() < tol * a,
                        "FWHM {fwhm} at ({x}, {y}): {a} vs {b}"
                    );
                    if sigma < 0.7 {
                        let worst = still
                            .iter()
                            .zip(&moved)
                            .map(|(u, v)| (u - v).abs())
                            .fold(0.0, f64::max);
                        assert!(worst < 1e-3 * 1e4, "FWHM {fwhm} at ({x}, {y}): {worst}");
                    }
                }
            }
        }
    }

    /// A streak narrower than 0.7 px carries its flux whatever its phase.
    #[test]
    fn narrow_streak_keeps_its_flux() {
        let (w, h) = (40, 40);
        for k in 0..6 {
            let o = f64::from(k) / 6.0;
            let mut img = vec![0.0; w * h];
            crate::psf::add_streak(&mut img, w, h, (8.0 + o, 10.0), (30.0, 25.0 + o), 5e4, 0.3);
            let sum: f64 = img.iter().sum();
            assert!((sum - 5e4).abs() < 1e-6 * 5e4, "{sum}");
        }
    }

    #[test]
    fn fast_erf_is_accurate() {
        let mut worst: f64 = 0.0;
        for k in -800_000..=800_000 {
            let x = f64::from(k) * 1e-5;
            worst = worst.max((crate::psf::erf_fast(x) - libm::erf(x)).abs());
        }
        eprintln!("erf_fast: worst {worst:.2e}");
        assert!(worst < 2e-9, "{worst}");
        assert_eq!(crate::psf::erf_fast(7.0), 1.0);
        assert_eq!(crate::psf::erf_fast(-7.0), -1.0);
    }

    #[test]
    fn straight_steady_path_is_not_refined() {
        let pts = trace(
            |s| PathPoint {
                s,
                xy: Some((10.0 + s, 20.0 + 2.0 * s)),
                rate: 100.0,
            },
            10.0,
            0.01,
        );
        assert_eq!(pts.len(), START_PIECES + 1);
    }

    #[test]
    fn curved_path_is_refined_within_tolerance() {
        let f = |s: f64| PathPoint {
            s,
            xy: Some((50.0 * libm::cos(s / 10.0), 50.0 * libm::sin(s / 10.0))),
            rate: 1.0,
        };
        let pts = trace(f, 10.0, 0.01);
        assert!(pts.len() > 5);
        for p in pts.windows(2) {
            let m = f(0.5 * (p[0].s + p[1].s)).xy.unwrap();
            let (a, b) = (p[0].xy.unwrap(), p[1].xy.unwrap());
            assert!(libm::hypot(m.0 - 0.5 * (a.0 + b.0), m.1 - 0.5 * (a.1 + b.1)) <= 0.01);
        }
    }

    /// A source entering shadow at s = 3.7 gives 3.7 s × 100 e/s.
    #[test]
    fn brightness_step_is_located() {
        let pts = trace(
            |s| PathPoint {
                s,
                xy: Some((s, 0.0)),
                rate: if s < 3.7 { 100.0 } else { 0.0 },
            },
            10.0,
            0.01,
        );
        let total: f64 = pts
            .windows(2)
            .map(|p| 0.5 * (p[0].rate + p[1].rate) * (p[1].s - p[0].s))
            .sum();
        assert!((total - 370.0).abs() < 1.0, "{total}");
    }
}
