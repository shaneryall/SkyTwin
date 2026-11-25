//! Pointing motion during the exposure, as an offset of the whole image.
//!
//! - Wobble: an Ornstein–Uhlenbeck process in x and y, the standard model
//!   of a stationary, smoothly correlated random motion (Uhlenbeck &
//!   Ornstein, Phys. Rev. 36, 823, 1930), sampled exactly on a time grid
//!   (Gillespie, Phys. Rev. E 54, 2084, 1996).
//! - Drift: a constant rate in a random direction, zero at mid-exposure.
//!   With a turn rate the direction turns steadily, so the image moves
//!   along a circular arc of radius rate / turn rate and every star trails
//!   in the same arc, as when a tracking camera follows a curving path.
//! - Roll: the image turns about a centre at a constant rate, zero at
//!   mid-exposure, so stars trail in arcs centred there, longer the
//!   farther they are from it.
//! - Jumps: sudden offsets at Poisson-distributed times, random directions.
//! - Oscillations: sinusoids (a reaction wheel or a structural mode). The
//!   image dwells at the turning points, so a trail along the oscillation
//!   turns into dashes and one across it into two parallel trails.
//!
//! The whole path is measured from the pointing at mid-exposure: the image
//! is unmoved then, whatever wobble, oscillation or jumps came before. The
//! mid-exposure star positions in the truth and the header WCS therefore
//! describe the image at that moment.
//!
//! Fast jitter that the exposure cannot resolve is a blur (`blur_arcsec`),
//! applied to the PSF instead.

use serde::{Deserialize, Serialize};

use crate::param::{DrawCtx, Param};
use crate::rng::Rng;

/// Pointing motion settings.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PointingMotion {
    /// Smooth random wobble, arcsec RMS per axis.
    #[serde(default)]
    pub wobble_rms_arcsec: Option<Param>,
    /// Wobble correlation time, s (default 1).
    #[serde(default)]
    pub wobble_time_s: Option<Param>,
    /// Steady drift, arcsec/s, in a random direction.
    #[serde(default)]
    pub drift_arcsec_per_s: Option<Param>,
    /// Rate at which the drift direction turns, deg/s, in a random sense:
    /// the image moves along a circular arc and stars trail in arcs.
    #[serde(default)]
    pub drift_turn_deg_per_s: Option<Param>,
    /// Rotation of the image during the exposure, deg/s, in a random
    /// sense, zero at mid-exposure.
    #[serde(default)]
    pub roll_rate_deg_per_s: Option<Param>,
    /// Centre of that rotation, 0-based pixels `[x, y]` (default: the image
    /// centre).
    #[serde(default)]
    pub roll_centre_px: Option<[f64; 2]>,
    /// Sudden jumps per second.
    #[serde(default)]
    pub jump_rate_per_s: Option<Param>,
    /// Size of each jump, arcsec (drawn per jump).
    #[serde(default)]
    pub jump_arcsec: Option<Param>,
    /// Unresolved fast jitter, arcsec (Gaussian σ added to the PSF).
    #[serde(default)]
    pub blur_arcsec: Option<Param>,
    /// Sinusoidal oscillations, resolved in the exposure.
    #[serde(default)]
    pub oscillations: Vec<Oscillation>,
    /// How wrong the pointing written to the header is.
    #[serde(default)]
    pub header: Option<HeaderPointing>,
}

/// One sinusoidal oscillation of the pointing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Oscillation {
    /// Amplitude (half the peak-to-peak), arcsec.
    pub amplitude_arcsec: Param,
    /// Period, s.
    pub period_s: Param,
    /// Direction on the image, degrees from +x toward +y (default: random
    /// per frame).
    #[serde(default)]
    pub angle_deg: Option<Param>,
}

/// Error of the pointing the spacecraft reports in the header.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderPointing {
    /// Pixel (0-based `[x, y]`) whose sky position the reported RA/Dec
    /// describe (default: the image centre).
    #[serde(default)]
    pub reference_px: Option<[f64; 2]>,
    /// Random error of the reported RA/Dec, arcsec σ per axis.
    #[serde(default)]
    pub error_arcsec: Option<Param>,
    /// Random error of the reported roll, degrees σ.
    #[serde(default)]
    pub roll_error_deg: Option<Param>,
    /// When during the exposure the reported pointing applies (default
    /// mid-exposure). Matters when tracking or when the pointing moves.
    #[serde(default)]
    pub epoch: PointingEpoch,
}

/// A moment in the exposure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointingEpoch {
    /// Shutter open.
    Start,
    /// Mid-exposure.
    #[default]
    Mid,
    /// Shutter close.
    End,
}

/// One jump.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Jump {
    /// When, s after shutter open.
    pub t_s: f64,
    /// Offset added from then on, pixels `[dx, dy]`.
    pub offset_px: [f64; 2],
}

/// The offset path for one frame, in pixels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionPath {
    /// Sample times, s after shutter open.
    pub times_s: Vec<f64>,
    /// Offsets at those times, pixels `[dx, dy]` (wobble, drift and
    /// oscillations), from the image at mid-exposure.
    pub offset_px: Vec<[f64; 2]>,
    /// Jumps, applied as steps on top.
    pub jumps: Vec<Jump>,
    /// Drift rate, pixels/s `[dx, dy]` (at mid-exposure when it turns).
    pub drift_px_per_s: [f64; 2],
    /// Turn rate of the drift direction, rad/s; positive turns +x toward
    /// +y.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub drift_turn_rad_per_s: f64,
    /// Rotation rate of the image, rad/s, positive turning +x toward +y;
    /// zero at mid-exposure.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub roll_rad_per_s: f64,
    /// Centre of the rotation, 0-based pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roll_centre_px: Option<[f64; 2]>,
    /// Pieces a source's path must start from to follow the oscillations
    /// (1 without any: the path is refined adaptively instead).
    #[serde(skip, default = "one")]
    pub min_pieces: usize,
}

/// Whether a rate is exactly zero (left out of the truth file).
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's skip_serializing_if passes a reference"
)]
fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// Default [`MotionPath::min_pieces`] when deserialising.
fn one() -> usize {
    1
}

/// One drawn oscillation, ready to evaluate.
struct Osc {
    /// Amplitude, pixels.
    amp: f64,
    /// Angular frequency, rad/s.
    omega: f64,
    /// Cosine of the direction on the image (from +x toward +y).
    cos: f64,
    /// Sine of the direction on the image.
    sin: f64,
    /// Phase at shutter open, radians.
    phase: f64,
}

/// Fewest wobble samples per exposure.
const STEPS: usize = 256;
/// Most samples per exposure, however fast the oscillations.
const MAX_STEPS: usize = 65_536;

impl MotionPath {
    /// Draw a path for an exposure of `exposure` seconds; `scale` is arcsec
    /// per pixel.
    ///
    /// The wobble is sampled exactly on a uniform grid of at least 256 steps,
    /// and of at least 16 steps per period of the fastest oscillation (capped
    /// at 65 536). The finished path is shifted so that the offset, jumps
    /// included, is zero at mid-exposure. Random draws happen in a
    /// fixed order (settings, drift direction, oscillations, wobble, jumps,
    /// then the turn and roll senses), so a seed always gives the same path.
    /// The turn and roll are drawn last and only when set, so profiles
    /// without them draw the same numbers as before they existed.
    ///
    /// A turning drift has velocity `drift · (cos(az + turn·m), sin(az +
    /// turn·m))` at `m` seconds from mid-exposure; its offset is that
    /// integrated from mid-exposure, replacing the straight drift term.
    pub fn draw(spec: &PointingMotion, exposure: f64, scale: f64, rng: &mut Rng) -> Self {
        let ctx = &mut DrawCtx::default();
        let mut d = |p: &Option<Param>, default: f64| {
            p.as_ref().map_or(default, |p| p.draw(rng, ctx)).max(0.0)
        };
        let sigma = d(&spec.wobble_rms_arcsec, 0.0) / scale;
        let tau = d(&spec.wobble_time_s, 1.0).max(1e-6);
        let drift = d(&spec.drift_arcsec_per_s, 0.0) / scale;
        let jump_rate = d(&spec.jump_rate_per_s, 0.0);
        let az = rng.range(0.0, 2.0 * core::f64::consts::PI);
        let drift_v = [drift * libm::cos(az), drift * libm::sin(az)];
        let osc: Vec<Osc> = spec
            .oscillations
            .iter()
            .map(|o| {
                let a = o.amplitude_arcsec.draw(rng, ctx).max(0.0) / scale;
                let per = o.period_s.draw(rng, ctx).max(1e-6);
                let ang = match &o.angle_deg {
                    Some(p) => p.draw(rng, ctx).to_radians(),
                    None => rng.range(0.0, core::f64::consts::PI),
                };
                let ph = rng.range(0.0, 2.0 * core::f64::consts::PI);
                Osc {
                    amp: a,
                    omega: 2.0 * core::f64::consts::PI / per,
                    cos: libm::cos(ang),
                    sin: libm::sin(ang),
                    phase: ph,
                }
            })
            .collect();
        let steps = osc
            .iter()
            .filter(|o| o.amp > 0.0)
            .map(|o| libm::ceil(16.0 * exposure * o.omega / (2.0 * core::f64::consts::PI)) as usize)
            .max()
            .unwrap_or(0)
            .clamp(STEPS, MAX_STEPS);
        let dt = exposure / steps as f64;
        let a = libm::exp(-dt / tau);
        let b = sigma * libm::sqrt(1.0 - a * a);
        let (mut x, mut y) = (sigma * rng.normal(), sigma * rng.normal());
        let mut times = Vec::with_capacity(steps + 1);
        let mut off = Vec::with_capacity(steps + 1);
        for k in 0..=steps {
            let t = k as f64 * dt;
            if k > 0 {
                x = a * x + b * rng.normal();
                y = a * y + b * rng.normal();
            }
            times.push(t);
            let m = t - 0.5 * exposure;
            let (mut ox, mut oy) = (x + drift_v[0] * m, y + drift_v[1] * m);
            for o in &osc {
                let v = o.amp * libm::sin(o.omega * t + o.phase);
                ox += v * o.cos;
                oy += v * o.sin;
            }
            off.push([ox, oy]);
        }
        let n_jumps = rng.poisson(jump_rate * exposure) as usize;
        let mut jumps: Vec<Jump> = (0..n_jumps)
            .map(|_| {
                let t_s = rng.range(0.0, exposure);
                let size = spec
                    .jump_arcsec
                    .as_ref()
                    .map_or(0.0, |p| p.draw(rng, ctx))
                    .max(0.0)
                    / scale;
                let dir = rng.range(0.0, 2.0 * core::f64::consts::PI);
                Jump {
                    t_s,
                    offset_px: [size * libm::cos(dir), size * libm::sin(dir)],
                }
            })
            .collect();
        jumps.sort_by(|p, q| p.t_s.total_cmp(&q.t_s));
        let sense = |rate: f64, rng: &mut Rng| {
            if rate == 0.0 {
                0.0
            } else if rng.uniform() < 0.5 {
                -rate
            } else {
                rate
            }
        };
        let turn = spec
            .drift_turn_deg_per_s
            .as_ref()
            .map_or(0.0, |p| p.draw(rng, ctx))
            .max(0.0)
            .to_radians();
        let turn = if drift > 0.0 { sense(turn, rng) } else { 0.0 };
        let roll = spec
            .roll_rate_deg_per_s
            .as_ref()
            .map_or(0.0, |p| p.draw(rng, ctx))
            .max(0.0)
            .to_radians();
        let roll = sense(roll, rng);
        if turn != 0.0 {
            let r = drift / turn;
            for (o, &t) in off.iter_mut().zip(&times) {
                let m = t - 0.5 * exposure;
                let a = az + turn * m;
                o[0] += r * (libm::sin(a) - libm::sin(az)) - drift_v[0] * m;
                o[1] += r * (libm::cos(az) - libm::cos(a)) - drift_v[1] * m;
            }
        }
        let min_pieces = if osc.iter().any(|o| o.amp > 0.0) {
            steps
        } else {
            1
        };
        let mut path = MotionPath {
            times_s: times,
            offset_px: off,
            jumps,
            drift_px_per_s: drift_v,
            drift_turn_rad_per_s: turn,
            roll_rad_per_s: roll,
            roll_centre_px: None,
            min_pieces,
        };
        let (mx, my) = path.offset(0.5 * exposure);
        for o in &mut path.offset_px {
            o[0] -= mx;
            o[1] -= my;
        }
        path
    }

    /// Image rotation at `s` seconds after shutter open, radians (zero at
    /// mid-exposure).
    pub fn roll_at(&self, s: f64) -> f64 {
        let exposure = self.times_s.last().copied().unwrap_or(0.0);
        self.roll_rad_per_s * (s - 0.5 * exposure)
    }

    /// Where pixel `p` (its position with no motion) is `s` seconds after
    /// shutter open: turned about the roll centre, then offset.
    pub fn apply(&self, p: (f64, f64), s: f64) -> (f64, f64) {
        let (mut x, mut y) = p;
        if self.roll_rad_per_s != 0.0 {
            let [cx, cy] = self.roll_centre_px.unwrap_or([0.0, 0.0]);
            let a = self.roll_at(s);
            let (sn, cs) = (libm::sin(a), libm::cos(a));
            let (dx, dy) = (x - cx, y - cy);
            x = cx + cs * dx - sn * dy;
            y = cy + sn * dx + cs * dy;
        }
        let o = self.offset(s);
        (x + o.0, y + o.1)
    }

    /// Whether the image does not move at all.
    pub fn is_still(&self) -> bool {
        self.jumps.is_empty()
            && self.roll_rad_per_s == 0.0
            && self.offset_px.iter().all(|o| o[0] == 0.0 && o[1] == 0.0)
    }

    /// Pieces a source's path must start from to follow this motion: the
    /// samples when it oscillates, else 1 (adaptive refinement suffices).
    pub fn pieces(&self) -> usize {
        self.min_pieces.max(1)
    }

    /// Offset at `s` seconds after shutter open, pixels `(dx, dy)`.
    ///
    /// Linear between samples, clamped to the sampled interval, plus every
    /// jump at or before `s`.
    ///
    /// # Panics
    ///
    /// If the path has fewer than two samples (never the case for a path
    /// from [`MotionPath::draw`]).
    pub fn offset(&self, s: f64) -> (f64, f64) {
        let n = self.times_s.len();
        let dt = self.times_s[1] - self.times_s[0];
        let f = (s / dt).clamp(0.0, (n - 1) as f64);
        let i = (f as usize).min(n - 2);
        let u = f - i as f64;
        let (p, q) = (self.offset_px[i], self.offset_px[i + 1]);
        let mut o = (p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1]));
        for j in &self.jumps {
            if s >= j.t_s {
                o.0 += j.offset_px[0];
                o.1 += j.offset_px[1];
            }
        }
        o
    }

    /// The path thinned to about 65 samples, for the truth file. The last
    /// sample (shutter close) is always kept, so the thinned path spans the
    /// whole exposure.
    #[must_use]
    pub fn thinned(&self) -> MotionPath {
        fn keep<T: Clone>(v: &[T], step: usize) -> Vec<T> {
            let mut out: Vec<T> = v.iter().step_by(step).cloned().collect();
            if v.len() > 1 && (v.len() - 1) % step != 0 {
                out.extend(v.last().cloned());
            }
            out
        }
        let step = (self.times_s.len() / 64).max(1);
        MotionPath {
            times_s: keep(&self.times_s, step),
            offset_px: keep(&self.offset_px, step),
            jumps: self.jumps.clone(),
            drift_px_per_s: self.drift_px_per_s,
            drift_turn_rad_per_s: self.drift_turn_rad_per_s,
            roll_rad_per_s: self.roll_rad_per_s,
            roll_centre_px: self.roll_centre_px,
            min_pieces: self.min_pieces,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Thinning keeps shutter close even when the sample count does not
    /// divide evenly, so the truth's path and its roll span the exposure.
    #[test]
    fn thinned_path_keeps_the_last_sample() {
        for n in [2usize, 64, 65, 129, 130, 1000] {
            let path = MotionPath {
                times_s: (0..n).map(|k| k as f64).collect(),
                offset_px: (0..n).map(|k| [k as f64, 0.0]).collect(),
                jumps: Vec::new(),
                drift_px_per_s: [0.0, 0.0],
                drift_turn_rad_per_s: 0.0,
                roll_rad_per_s: 0.1,
                roll_centre_px: None,
                min_pieces: 1,
            };
            let t = path.thinned();
            assert_eq!(t.times_s.last(), path.times_s.last(), "n = {n}");
            assert_eq!(t.offset_px.len(), t.times_s.len(), "n = {n}");
            assert!(t.times_s.len() <= 2 * 65, "n = {n}");
            assert_eq!(t.roll_at(3.0), path.roll_at(3.0), "n = {n}");
        }
    }

    /// A 3″ RMS wobble at 3″/px is 1 px RMS about the mean pointing (over
    /// 100 s, 200 correlation times, the sample mean takes about 1% of the
    /// variance); a 6″/s drift over 4 s at 3″/px moves 8 px and passes
    /// through zero at mid-exposure.
    #[test]
    fn wobble_has_the_set_rms_and_drift_the_set_rate() {
        let spec = PointingMotion {
            wobble_rms_arcsec: Some(3.0.into()),
            wobble_time_s: Some(0.5.into()),
            ..Default::default()
        };
        let mut rng = Rng::new(2);
        let mut sq = 0.0;
        let mut n = 0.0;
        for _ in 0..200 {
            let p = MotionPath::draw(&spec, 100.0, 3.0, &mut rng);
            let k = p.offset_px.len() as f64;
            let mx = p.offset_px.iter().map(|o| o[0]).sum::<f64>() / k;
            let my = p.offset_px.iter().map(|o| o[1]).sum::<f64>() / k;
            for o in &p.offset_px {
                sq += (o[0] - mx) * (o[0] - mx) + (o[1] - my) * (o[1] - my);
                n += 2.0;
            }
        }
        let rms = libm::sqrt(sq / n);
        assert!((rms - 1.0).abs() < 0.05, "{rms}");
        let spec = PointingMotion {
            drift_arcsec_per_s: Some(6.0.into()),
            ..Default::default()
        };
        let p = MotionPath::draw(&spec, 4.0, 3.0, &mut rng);
        let (a, b) = (p.offset(0.0), p.offset(4.0));
        assert!((libm::hypot(b.0 - a.0, b.1 - a.1) - 8.0).abs() < 1e-9);
        assert!(libm::hypot(p.offset(2.0).0, p.offset(2.0).1) < 1e-9);
    }

    /// A 6″, 0.25 s oscillation at 30° over 3 s at 3″/px: at least 16
    /// samples per period (12 periods), every offset along 30°, 2 px half
    /// peak-to-peak, and the same offset one period later.
    #[test]
    fn oscillation_has_its_amplitude_period_and_direction() {
        let spec: PointingMotion = serde_json::from_value(serde_json::json!({
            "oscillations": [{ "amplitude_arcsec": 6.0, "period_s": 0.25, "angle_deg": 30.0 }]
        }))
        .unwrap();
        let p = MotionPath::draw(&spec, 3.0, 3.0, &mut Rng::new(8));
        assert!(p.pieces() >= 192);
        let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for o in &p.offset_px {
            assert!((o[0] * s - o[1] * c).abs() < 1e-9);
            let along = o[0] * c + o[1] * s;
            lo = lo.min(along);
            hi = hi.max(along);
        }
        let amp = 0.5 * (hi - lo);
        assert!((amp - 2.0).abs() < 0.01, "{amp}");
        let (a, b) = (p.offset(1.0), p.offset(1.25));
        assert!((a.0 - b.0).abs() < 0.02 && (a.1 - b.1).abs() < 0.02);
    }

    /// A jump is a step of its full size between two adjacent instants.
    #[test]
    fn jumps_are_steps() {
        let spec = PointingMotion {
            jump_rate_per_s: Some(1.0.into()),
            jump_arcsec: Some(9.0.into()),
            ..Default::default()
        };
        let mut rng = Rng::new(5);
        let p = (0..50)
            .map(|_| MotionPath::draw(&spec, 3.0, 3.0, &mut rng))
            .find(|p| p.jumps.len() == 1)
            .unwrap();
        let t = p.jumps[0].t_s;
        let (a, b) = (p.offset(t - 1e-9), p.offset(t));
        assert!((libm::hypot(b.0 - a.0, b.1 - a.1) - 3.0).abs() < 1e-9);
    }

    /// 2 px/s for 4 s turning 45°/s is half a circle 8 px long: the ends
    /// are a diameter apart, and a second from mid-exposure the direction
    /// has turned 45°, a chord of 2r·sin(22.5°).
    #[test]
    fn a_turning_drift_follows_an_arc() {
        let spec = PointingMotion {
            drift_arcsec_per_s: Some(6.0.into()),
            drift_turn_deg_per_s: Some(45.0.into()),
            ..Default::default()
        };
        let p = MotionPath::draw(&spec, 4.0, 3.0, &mut Rng::new(3));
        assert!((p.drift_turn_rad_per_s.abs() - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
        let r = 8.0 / core::f64::consts::PI;
        let at = |s: f64| p.offset(s);
        assert!(libm::hypot(at(2.0).0, at(2.0).1) < 1e-9);
        let (a, b) = (at(0.0), at(4.0));
        assert!((libm::hypot(b.0 - a.0, b.1 - a.1) - 2.0 * r).abs() < 1e-6);
        let q = at(1.0);
        let chord = 2.0 * r * libm::sin(core::f64::consts::FRAC_PI_8);
        assert!((libm::hypot(q.0, q.1) - chord).abs() < 1e-3);
        let mut length = 0.0;
        for k in 0..400 {
            let (u, v) = (at(f64::from(k) * 0.01), at(f64::from(k + 1) * 0.01));
            length += libm::hypot(v.0 - u.0, v.1 - u.1);
        }
        assert!((length - 8.0).abs() < 1e-3, "{length}");
    }

    /// 2°/s for 10 s turns a point 200 px from the centre through 20° on a
    /// circle about it, still at mid-exposure.
    #[test]
    fn roll_turns_the_image_about_its_centre() {
        let spec = PointingMotion {
            roll_rate_deg_per_s: Some(2.0.into()),
            ..Default::default()
        };
        let mut p = MotionPath::draw(&spec, 10.0, 3.0, &mut Rng::new(4));
        p.roll_centre_px = Some([100.0, 50.0]);
        assert!(!p.is_still());
        let q = (300.0, 50.0);
        let mid = p.apply(q, 5.0);
        assert!(libm::hypot(mid.0 - q.0, mid.1 - q.1) < 1e-9);
        assert_eq!(p.apply((100.0, 50.0), 0.0), (100.0, 50.0));
        let (a, b) = (p.apply(q, 0.0), p.apply(q, 10.0));
        for e in [a, b] {
            assert!((libm::hypot(e.0 - 100.0, e.1 - 50.0) - 200.0).abs() < 1e-9);
        }
        let chord = 2.0 * 200.0 * libm::sin(10f64.to_radians());
        assert!((libm::hypot(b.0 - a.0, b.1 - a.1) - chord).abs() < 1e-9);
    }

    /// Every kind of motion together still leaves the image where it would
    /// be without motion at mid-exposure, so the mid-exposure positions in
    /// the truth and the header WCS describe the image at that moment.
    #[test]
    fn the_image_is_unmoved_at_mid_exposure() {
        let spec: PointingMotion = serde_json::from_value(serde_json::json!({
            "wobble_rms_arcsec": 3.0, "wobble_time_s": 0.3,
            "drift_arcsec_per_s": 6.0, "drift_turn_deg_per_s": 20.0,
            "roll_rate_deg_per_s": 1.0, "jump_rate_per_s": 1.0, "jump_arcsec": 9.0,
            "oscillations": [{ "amplitude_arcsec": 4.0, "period_s": 0.7 }]
        }))
        .unwrap();
        let mut rng = Rng::new(12);
        for _ in 0..50 {
            let mut p = MotionPath::draw(&spec, 4.0, 3.0, &mut rng);
            p.roll_centre_px = Some([60.0, 40.0]);
            let o = p.offset(2.0);
            assert!(libm::hypot(o.0, o.1) < 1e-9, "{o:?}");
            let q = p.apply((130.0, 75.0), 2.0);
            assert!(libm::hypot(q.0 - 130.0, q.1 - 75.0) < 1e-9, "{q:?}");
        }
    }

    /// Zero turn and roll rates draw no extra numbers.
    #[test]
    fn frames_without_turns_draw_the_same_path() {
        let plain = PointingMotion {
            wobble_rms_arcsec: Some(1.0.into()),
            drift_arcsec_per_s: Some(6.0.into()),
            jump_rate_per_s: Some(0.5.into()),
            jump_arcsec: Some(4.0.into()),
            ..Default::default()
        };
        let zero = PointingMotion {
            drift_turn_deg_per_s: Some(0.0.into()),
            roll_rate_deg_per_s: Some(0.0.into()),
            ..plain.clone()
        };
        let a = MotionPath::draw(&plain, 5.0, 3.0, &mut Rng::new(9));
        let b = MotionPath::draw(&zero, 5.0, 3.0, &mut Rng::new(9));
        assert_eq!(a, b);
    }
}
