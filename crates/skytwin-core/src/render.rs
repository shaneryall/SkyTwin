//! Turn a scene into pixels.
//!
//! [`render_profile`] builds a frame in the order the charge meets each
//! effect on a real detector:
//!
//! - light: sky, stars, targets, streaks and stray light, in electrons over
//!   the exposure;
//! - the flat field (vignetting, PRNU, dust), which acts on light only;
//! - frame-transfer smear, which also sees light only, then dark current,
//!   hot and flickering pixels and radiation charge;
//! - shot noise on the expected signal;
//! - full well, with serial overflow and column bleeding;
//! - charge-transfer inefficiency (CTI);
//! - read noise and serial read-out lag on the video signal;
//! - offsets, patterns and the ADC (gain, rounding, clipping);
//! - read-out faults on the delivered values.
//!
//! Every random component draws from its own stream, `seed` XOR a fixed
//! constant, so turning one effect on does not move the draws of another.
//! Shot and read noise go further: each pixel has its own stream keyed by
//! its index, so a pixel's noise depends only on its own signal and a clean
//! and a dirty frame with the same seed differ exactly by their sources.

use serde::{Deserialize, Serialize};

use crate::atmosphere::Atmosphere;
use crate::camera::Camera;
use crate::detector::{
    apply_cti, dark_factor, draw_hit_charge, draw_hits, hot_pixel_map, Hit, HotPixel,
};
use crate::error::SimError;
use crate::fits::Value;
use crate::geometry::{timing, Geometry, Timing};
use crate::label::{capture, label, lit_path, thin, Footprint, Noise, SourceLabel};
use crate::motion::MotionPath;
use crate::orbit::{Orbit, Site, State};
use crate::path::{draw, trace_with, PathPoint};
use crate::profile::{FrameSettings, Profile};
use crate::psf::Psf;
use crate::rng::Rng;
use crate::scene::{Scene, Star, Streak};
use crate::stray::{add_stray_light, ecliptic_geometry, StrayTruth};
use crate::target::{parse_tle_file, Target};
use crate::vec3::{angle, radec_to_unit, scale, unit_to_radec, V3};
use crate::wcs::{Pointing, TanWcs};

/// A star as drawn: every coordinate and brightness form filled in.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrueStar {
    /// Catalogue identifier, if the star came from one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    /// Column, 0-based, pixel centres on whole numbers.
    pub x: f64,
    /// Row, 0-based.
    pub y: f64,
    /// Right ascension, degrees (when the scene has a pointing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ra_deg: Option<f64>,
    /// Declination, degrees (when the scene has a pointing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dec_deg: Option<f64>,
    /// Magnitude in the camera band.
    pub mag: f64,
    /// Total signal over the exposure, electrons.
    pub flux_e: f64,
    /// Parallax used, mas.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallax_mas: Option<f64>,
    /// Colour index, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_index: Option<f64>,
    /// Cramér–Rao bound on the centroid `[x, y]`, pixels (static frames).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crlb_px: Option<[f64; 2]>,
    /// Cramér–Rao bound on the flux, electrons (static frames).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crlb_flux_e: Option<f64>,
    /// Photons reaching the detector, before quantum efficiency and the flat.
    pub photons: f64,
    /// Electrons expected after the flat: `flux_e` times the flat averaged
    /// over the pixels the star's light falls on, weighted by that light.
    pub detected_e: f64,
    /// Where a trailed star is at shutter open, `[x, y]` (tracking or
    /// pointing motion; none for a still star).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trail_start: Option<[f64; 2]>,
    /// Where it is at shutter close.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trail_end: Option<[f64; 2]>,
    /// What a detector could see of it, for training labels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<SourceLabel>,
}

/// A streak as drawn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrueStreak {
    /// Position on the image at shutter open, `[x, y]` pixels: the scene's
    /// start moved by any pointing motion at that moment.
    pub start: [f64; 2],
    /// Position on the image at shutter close, `[x, y]` pixels.
    pub end: [f64; 2],
    /// Sky position at shutter open, `[ra, dec]` degrees (with a pointing):
    /// the scene's start through the mid-exposure WCS, which pointing
    /// motion does not move.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_radec: Option<[f64; 2]>,
    /// Sky position at shutter close, `[ra, dec]` degrees (with a pointing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_radec: Option<[f64; 2]>,
    /// Magnitude in the camera band.
    pub mag: f64,
    /// Total signal over the exposure, electrons.
    pub flux_e: f64,
    /// What a detector could see of it, for training labels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<SourceLabel>,
}

/// Exposure times as written to the header.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimeTruth {
    /// Shutter open, UTC ISO 8601.
    pub start_utc: String,
    /// Mid-exposure, UTC ISO 8601.
    pub mid_utc: String,
    /// Shutter close, UTC ISO 8601.
    pub end_utc: String,
    /// MJD (UTC) at shutter open.
    pub mjd_start: f64,
    /// Julian year (TT) at mid-exposure: the epoch star positions are for.
    pub epoch_jyear: f64,
}

impl From<Timing> for TimeTruth {
    fn from(t: Timing) -> Self {
        Self {
            start_utc: t.start.to_string(),
            mid_utc: t.mid.to_string(),
            end_utc: t.end.to_string(),
            mjd_start: t.start.mjd(),
            epoch_jyear: t.mid.julian_year_tt(),
        }
    }
}

/// Where the camera was and where it looked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointingTruth {
    /// Apparent boresight pointing (what the spacecraft attitude says).
    pub apparent: Pointing,
    /// Pointing written to the header (with the profile's reporting error;
    /// equal to `apparent` without one).
    pub reported: Pointing,
    /// Camera → GCRS quaternion `[w, x, y, z]` of the apparent pointing.
    pub quaternion: [f64; 4],
    /// Pixels ↔ apparent directions.
    pub camera_wcs: TanWcs,
    /// Boresight shift from stellar aberration, arcseconds.
    pub aberration_arcsec: f64,
    /// Largest error of the header (sky) WCS against the exact mapping, px.
    pub sky_wcs_fit_px: f64,
}

/// The observer at mid-exposure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObserverTruth {
    /// GCRS state (km, km/s).
    pub gcrs: State,
    /// Barycentric velocity used for aberration, km/s.
    pub velocity_bary_km_s: [f64; 3],
    /// The ground site, for an observer on the Earth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<Site>,
    /// The atmosphere used (local vertical, pressure, temperature,
    /// extinction), for a ground site.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atmosphere: Option<Atmosphere>,
    /// Airmass at the boresight, for a ground site.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub airmass: Option<f64>,
    /// Altitude of the boresight, degrees (observed), for a ground site.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub altitude_deg: Option<f64>,
    /// Refraction at the boresight, arcseconds, for a ground site.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refraction_arcsec: Option<f64>,
}

/// What was drawn, exactly: the answer key for a frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Truth {
    /// RNG seed used.
    pub seed: u64,
    /// Camera settings used.
    pub camera: Camera,
    /// Exposure times, when the scene has a time.
    pub time: Option<TimeTruth>,
    /// Observer state, when the scene has a time.
    pub observer: Option<ObserverTruth>,
    /// The sky WCS (pixels ↔ astrometric RA/Dec), written to the header.
    /// This is what a plate solver should find.
    pub wcs: Option<TanWcs>,
    /// The WCS the header gives instead, when the profile's
    /// `header.wrong_wcs` fault fired for this frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header_wcs: Option<TanWcs>,
    /// Pointing details, when the scene has a pointing.
    pub pointing: Option<PointingTruth>,
    /// Sky background, electrons per pixel per second.
    pub sky_e_per_s: f64,
    /// Every star drawn, including random and catalogue ones.
    pub stars: Vec<TrueStar>,
    /// Every streak given in pixels.
    pub streaks: Vec<TrueStreak>,
    /// Every target satellite.
    pub targets: Vec<TrueTarget>,
    /// The instrument profile and the settings drawn for this frame.
    pub instrument: InstrumentTruth,
}

/// Which instrument made the frame, and its per-frame settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstrumentTruth {
    /// Profile name.
    pub name: String,
    /// Settings drawn for this frame.
    pub settings: FrameSettings,
    /// Hot pixels and their rates this frame.
    pub hot_pixels: Vec<HotPixel>,
    /// Flickering pixels that were on this frame, with their mean rate over
    /// the exposure.
    pub flickering_pixels: Vec<HotPixel>,
    /// Radiation hits, when the profile has a radiation block.
    pub radiation: Option<RadiationTruth>,
    /// Pointing motion during the exposure (thinned), when there is any.
    pub pointing_motion: Option<MotionPath>,
    /// Stray light added, per source.
    pub stray_light: Vec<StrayTruth>,
    /// Dust shadows on the chip.
    pub dust: Vec<crate::flat::DustShadow>,
    /// Zodiacal light at the boresight, V mag/arcsec², when used.
    pub zodiacal_v_mag_arcsec2: Option<f64>,
    /// Rows corrupted in read-out.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub readout_bands: Vec<crate::detector::Band>,
}

/// The pointing the header reports: the sky position of the profile's
/// reference pixel, plus random errors and the scene's `bias_arcsec`
/// (east, north).
///
/// The reference pixel is read at the header's epoch (start, mid or end of
/// the exposure); both the camera's own motion (tracking) and the pointing
/// motion move the sky under it. An image roll is undone about its centre
/// at that moment (none at mid-exposure), and the reported roll follows
/// it.
fn reported_pointing(
    g: &Geometry,
    profile: &Profile,
    seed: u64,
    motion: Option<&MotionPath>,
    bias_arcsec: Option<[f64; 2]>,
) -> Pointing {
    let biased = |p: Pointing| match bias_arcsec {
        Some([e, n]) => Pointing {
            ra_deg: (p.ra_deg + e / 3600.0 / libm::cos(p.dec_deg.to_radians())).rem_euclid(360.0),
            dec_deg: (p.dec_deg + n / 3600.0).clamp(-90.0, 90.0),
            roll_deg: p.roll_deg,
        },
        None => p,
    };
    let Some(hp) = profile.pointing.as_ref().and_then(|p| p.header.as_ref()) else {
        return biased(g.pointing);
    };
    let mut rng = Rng::new(seed ^ REPORT_STREAM);
    let ctx = &mut crate::param::DrawCtx::default();
    let [rx, ry] = hp.reference_px.unwrap_or(g.camera_wcs.crpix0);
    let exp = profile.camera.exposure_s;
    let s = match hp.epoch {
        crate::motion::PointingEpoch::Start => 0.0,
        crate::motion::PointingEpoch::Mid => 0.5 * exp,
        crate::motion::PointingEpoch::End => exp,
    };
    let (ox, oy) = motion.map_or((0.0, 0.0), |m| {
        let (a, b) = (m.offset(s), m.offset(0.5 * exp));
        (a.0 - b.0, a.1 - b.1)
    });
    let turn = motion.map_or(0.0, |m| m.roll_at(s));
    let (px, py) = match motion.and_then(|m| m.roll_centre_px) {
        Some([cx, cy]) if turn != 0.0 => {
            let (dx, dy) = (rx - ox - cx, ry - oy - cy);
            let (sn, cs) = (libm::sin(-turn), libm::cos(-turn));
            (cx + cs * dx - sn * dy, cy + sn * dx + cs * dy)
        }
        _ => (rx - ox, ry - oy),
    };
    let wcs_s = g.camera_wcs_at(s).unwrap_or(g.camera_wcs);
    let (ara, adec) = wcs_s.pixel_to_world(px, py);
    let (ra, dec) = unit_to_radec(crate::geometry::aberrate(
        g.unobserved(radec_to_unit(ara, adec)),
        scale(g.beta, -1.0),
    ));
    let err = hp
        .error_arcsec
        .as_ref()
        .map_or(0.0, |p| p.draw(&mut rng, ctx))
        .max(0.0)
        / 3600.0;
    let (ex, ey) = (err * rng.normal(), err * rng.normal());
    let roll_err = hp
        .roll_error_deg
        .as_ref()
        .map_or(0.0, |p| p.draw(&mut rng, ctx))
        .max(0.0);
    biased(Pointing {
        ra_deg: (ra + ex / libm::cos(dec.to_radians())).rem_euclid(360.0),
        dec_deg: (dec + ey).clamp(-90.0, 90.0),
        roll_deg: g.pointing.roll_deg - parity_sign(&g.camera_wcs) * turn.to_degrees()
            + roll_err * rng.normal(),
    })
}

/// The WCS a faulty header gives instead of the true one (another
/// solver's wrong solution), if the profile's `header.wrong_wcs` fires.
///
/// The tangent point moves by the drawn offset toward a random azimuth
/// (east of north), the sky axes turn by the drawn roll with a random sign,
/// and the scale changes by the drawn scale error.
fn wrong_header_wcs(profile: &Profile, wcs: Option<&TanWcs>, seed: u64) -> Option<TanWcs> {
    let f = profile.header.as_ref()?.wrong_wcs.as_ref()?;
    let wcs = wcs?;
    let mut rng = Rng::new(seed ^ HEADER_FAULT_STREAM);
    if rng.uniform() >= f.probability {
        return None;
    }
    let ctx = &mut crate::param::DrawCtx::default();
    let off = f.offset_arcmin.draw(&mut rng, ctx).max(0.0) / 60.0;
    let az = rng.range(0.0, 2.0 * core::f64::consts::PI);
    let rot = f.roll_deg.draw(&mut rng, ctx) * if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
    let scale = 1.0
        + f.scale_error
            .as_ref()
            .map_or(0.0, |p| p.draw(&mut rng, ctx));
    let (ra, dec) = (wcs.crval_deg[0], wcs.crval_deg[1]);
    let dec2 = (dec + off * libm::cos(az)).clamp(-89.999, 89.999);
    let ra2 = (ra + off * libm::sin(az) / libm::cos(dec.to_radians())).rem_euclid(360.0);
    let (s, c) = libm::sincos(rot.to_radians());
    let m = wcs.cd;
    let cd = [
        [
            scale * (c * m[0][0] - s * m[1][0]),
            scale * (c * m[0][1] - s * m[1][1]),
        ],
        [
            scale * (s * m[0][0] + c * m[1][0]),
            scale * (s * m[0][1] + c * m[1][1]),
        ],
    ];
    Some(TanWcs {
        crval_deg: [ra2, dec2],
        crpix0: wcs.crpix0,
        cd,
        distortion: wcs.distortion,
    })
}

/// The target called `name` in the scene's TLE file, unscreened: `None` when
/// the scene names no file or the file has no such target.
fn tle_file_target(scene: &Scene, name: &str) -> Result<Option<Target>, SimError> {
    let Some(file) = &scene.target_file else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(&file.path)
        .map_err(|e| SimError::Orbit(format!("{}: {e}", file.path.display())))?;
    Ok(parse_tle_file(&text, &file.brightness)?
        .into_iter()
        .find(|t| t.name == name))
}

/// Whether the named target is sunlit (any part of the Sun's disc) and not
/// blocked by the Earth at mid-exposure.
///
/// # Errors
///
/// [`SimError::BadTime`] if the scene has no time or an unreadable one,
/// [`SimError::UnknownTarget`] if no target in `scene.targets` or in
/// `scene.target_file` is called `name`, [`SimError::Orbit`] if the TLE
/// file is needed and cannot be read, and any error from propagating the
/// observer or the target.
pub fn target_visible(scene: &Scene, camera: &Camera, name: &str) -> Result<bool, SimError> {
    let tm = timing(scene.time.as_ref(), camera)?
        .ok_or(SimError::BadTime("only_visible needs scene.time"))?;
    let obs = scene.observer.clone().unwrap_or_default().state(tm.mid)?;
    let from_file;
    let tgt = if let Some(t) = scene.targets.iter().find(|t| t.name == name) {
        t
    } else {
        from_file = tle_file_target(scene, name)?;
        from_file
            .as_ref()
            .ok_or_else(|| SimError::UnknownTarget(name.to_string()))?
    };
    let atm = match &scene.observer {
        Some(Orbit::Site(site)) => Some(Atmosphere::for_site(site, tm.mid)),
        _ => None,
    };
    let v = tgt.view(tm.mid, &obs, [0.0; 3], camera, atm.as_ref())?;
    Ok(v.illumination > 0.0 && !v.occulted)
}

/// +1 for a negative-parity WCS (det CD < 0, the usual sky), −1 for a
/// mirrored one. Image content turning +x toward +y moves north toward
/// east on a negative-parity image, so the reported roll of +y falls by
/// the turn angle there and rises on a mirrored image.
fn parity_sign(w: &crate::wcs::TanWcs) -> f64 {
    let det = w.cd[0][0] * w.cd[1][1] - w.cd[0][1] * w.cd[1][0];
    if det < 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// A pixel position moved by the pointing motion at `t` seconds: turned
/// about the roll centre, then offset.
fn shifted(p: Option<(f64, f64)>, t: f64, motion: Option<&MotionPath>) -> Option<(f64, f64)> {
    match (p, motion) {
        (Some(p), Some(m)) => Some(m.apply(p, t)),
        _ => p,
    }
}

/// Radiation for one frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RadiationTruth {
    /// Sub-satellite latitude and longitude at mid-exposure, degrees.
    pub subpoint_deg: Option<[f64; 2]>,
    /// Observer altitude, km.
    pub altitude_km: Option<f64>,
    /// Rate used, hits/cm²/s.
    pub rate_per_cm2_s: f64,
    /// Collection time, s.
    pub collect_s: f64,
    /// Every hit.
    pub hits: Vec<Hit>,
}

const REPORT_STREAM: u64 = 0x2E70_2700_0000_0005;
const HEADER_FAULT_STREAM: u64 = 0x4EAD_FA17_0000_0007;
const READOUT_FAULT_STREAM: u64 = 0x2EAD_0B5D_0000_0009;
const STREAK_STREAM: u64 = 0x57EA_C000_0000_000A;
const SHOT_STREAM: u64 = 0x5407_0000_0000_000C;
const READ_STREAM: u64 = 0x2EAD_0000_0000_000D;
const ROW_NOISE_STREAM: u64 = 0x0F0F_5EED_7077_0001;
const SMOOTH_FIELD_STREAM: u64 = 0x5300_7F1E_1D00_0003;
const RTS_STREAM: u64 = 0x2754_F11C_0000_0006;
const RTS_FRAME_STREAM: u64 = 0x2754_F11C_0000_0016;
const MOTION_STREAM: u64 = 0x307_1077_0000_0004;
const COLUMN_STREAM: u64 = 0xC01_0FF5_E700_000E;
const HOT_STREAM: u64 = 0x407_9135_0000_0002;
const RADIATION_STREAM: u64 = 0x2AD1_A710_0000_0003;

/// Smear from charge moving while light still falls: every pixel gains, per
/// row it lies above (below) a source, `dwell.0` (`dwell.1`) seconds of
/// that source's rate. `img` holds light only, electrons over `exposure`.
///
/// Runs row by row (memory order) with per-column running sums of the rate
/// below each row: the same additions in the same order as a column-by-column
/// pass, so the result is bit-for-bit the same.
fn apply_smear(img: &mut [f64], w: usize, h: usize, exposure: f64, dwell: (f64, f64)) {
    if exposure <= 0.0 {
        return;
    }
    let mut total = vec![0.0; w];
    for row in img.chunks(w).take(h) {
        for (t, v) in total.iter_mut().zip(row) {
            *t += v / exposure;
        }
    }
    let mut below = vec![0.0; w];
    for row in img.chunks_mut(w).take(h) {
        for ((v, b), t) in row.iter_mut().zip(below.iter_mut()).zip(&total) {
            let r = *v / exposure;
            let above = t - *b - r;
            *v += dwell.0 * *b + dwell.1 * above;
            *b += r;
        }
    }
}

/// Move `o.fraction` of each pixel's charge above `full_well` along its row
/// toward `o.toward`, shared over the following pixels as (1 − q) qᵈ⁻¹ at
/// distance `d`, with q = exp(−1 / `o.length_px`). Charge that would run off
/// the chip is lost.
fn serial_overflow(
    img: &mut [f64],
    w: usize,
    h: usize,
    full_well: f64,
    o: &crate::profile::SerialOverflow,
) {
    let q = libm::exp(-1.0 / o.length_px);
    let reach = (libm::ceil(o.length_px * 20.0) as usize).max(1);
    for y in 0..h {
        let row = &mut img[y * w..(y + 1) * w];
        let spill: Vec<(usize, f64)> = row
            .iter()
            .enumerate()
            .filter(|(_, &e)| e > full_well)
            .map(|(x, &e)| (x, o.fraction * (e - full_well)))
            .collect();
        for (x, s) in spill {
            row[x] -= s;
            let mut wgt = 1.0 - q;
            for d in 1..=reach {
                let xx = match o.toward {
                    crate::profile::Toward::Left => x.checked_sub(d),
                    crate::profile::Toward::Right => (x + d < w).then_some(x + d),
                };
                let Some(xx) = xx else { break };
                row[xx] += s * wgt;
                wgt *= q;
            }
        }
    }
}

/// Single-pole low-pass along each row, in read order (the trail runs
/// toward `l.toward`, so reading starts at the other end):
/// `out = (1 − a) · in + a · previous out`, starting from the first pixel
/// read. Flux is kept up to what trails off the row's end.
fn serial_lag(img: &mut [f64], w: usize, l: &crate::profile::SerialLag) {
    let a = l.coefficient;
    if a <= 0.0 || w == 0 {
        return;
    }
    for row in img.chunks_mut(w) {
        let mut prev = match l.toward {
            crate::profile::Toward::Left => row[w - 1],
            crate::profile::Toward::Right => row[0],
        };
        let mut step = |v: &mut f64| {
            prev = (1.0 - a) * *v + a * prev;
            *v = prev;
        };
        match l.toward {
            crate::profile::Toward::Left => row.iter_mut().rev().for_each(&mut step),
            crate::profile::Toward::Right => row.iter_mut().for_each(&mut step),
        }
    }
}

/// Spill charge above `full_well` along each column, half up and half down,
/// filling neighbouring pixels in turn; charge reaching the column's end is
/// lost (it leaves the window).
fn bleed_columns(img: &mut [f64], w: usize, h: usize, full_well: f64) {
    for x in 0..w {
        for y in 0..h {
            let i = y * w + x;
            let excess = img[i] - full_well;
            if excess <= 0.0 {
                continue;
            }
            img[i] = full_well;
            for dir in [1i64, -1] {
                let mut left = excess / 2.0;
                let mut yy = y as i64 + dir;
                while left > 0.0 && (0..h as i64).contains(&yy) {
                    let j = yy as usize * w + x;
                    let room = (full_well - img[j]).max(0.0);
                    let put = room.min(left);
                    img[j] += put;
                    left -= put;
                    yy += dir;
                }
            }
        }
    }
}

/// A pattern's amplitude and its per-column and per-row (sin, cos) terms.
type Wave = (f64, Vec<(f64, f64)>, Vec<(f64, f64)>);

/// The delivered value range `(lo, hi)`, ADU, and whether values are
/// rounded to integers: from the profile's processing block, or 16-bit
/// unsigned integers without one.
/// The level, ADU, that zero electrons are delivered at: the bias unless
/// the ground processing removed it, plus the frame's pedestal.
fn adu_offset(profile: &Profile, camera: &Camera, fs: &FrameSettings) -> f64 {
    match profile.processing.as_ref() {
        Some(p) if p.bias_subtracted => fs.pedestal_adu,
        Some(_) => camera.bias_adu + fs.pedestal_adu,
        None => camera.bias_adu,
    }
}

/// The level, electrons, at which a pixel saturates: the full well, or
/// where the delivered value reaches the output's maximum, whichever comes
/// first (a camera of about 1 e⁻/ADU with a deep well clips in its ADC).
fn saturation_e(profile: &Profile, camera: &Camera, fs: &FrameSettings) -> f64 {
    let ((_, hi), _) = output_range(profile);
    let clip_e = (hi - adu_offset(profile, camera, fs)) * camera.gain_e_per_adu;
    camera.full_well_e.min(clip_e.max(0.0))
}

fn output_range(profile: &Profile) -> ((f64, f64), bool) {
    let proc = profile.processing.as_ref();
    (
        proc.map_or((0.0, 65_535.0), crate::profile::Processing::clip_range),
        proc.is_none_or(|p| p.output.is_integer()),
    )
}

/// Electrons to delivered ADU: shot noise, full well, CTI, read noise, gain
/// and bias, then the profile's read-out noise and ground processing.
///
/// `img` is the expected signal and `charge` the deposited (radiation)
/// charge, both electrons. Shot and read noise draw from one stream per
/// pixel (see the [module docs](self)). Charge above full well may first
/// spill along the row ([`serial_overflow`]) and bleed along the column
/// ([`bleed_columns`]); the rest is lost. Row offsets are drawn anew each
/// frame; column offsets come from the detector's own stream, one per chip
/// column (the first unbinned column of an image column), so they stay put
/// and two windows on the same chip agree where they overlap.
/// Read noise and extra white noise make the video signal, which serial lag
/// then smooths before offsets, the smooth and fixed fields, the ramp and
/// the patterns are added.
///
/// Each pattern is `amplitude · cos(k (x cos θ + y sin θ) + φ)`, evaluated
/// separably from per-column and per-row terms through
/// cos(a + b) = cos a cos b − sin a sin b, so the cost is O(w + h) per
/// pattern plus one multiply-add per pixel.
#[expect(
    clippy::too_many_arguments,
    reason = "one call site; the inputs are distinct stages of the frame"
)]
fn digitise(
    img: &[f64],
    w: usize,
    h: usize,
    camera: &Camera,
    profile: &Profile,
    fs: &FrameSettings,
    charge: &[f64],
    seed: u64,
) -> Vec<f64> {
    let shot_key = seed ^ SHOT_STREAM;
    let mut elec: Vec<f64> = img
        .iter()
        .zip(charge)
        .enumerate()
        .map(|(i, (&e, &q))| Rng::for_item(shot_key, i as u64).poisson(e) + q)
        .collect();
    if let Some(o) = profile
        .saturation
        .as_ref()
        .and_then(|s| s.serial_overflow.as_ref())
    {
        serial_overflow(&mut elec, w, h, camera.full_well_e, o);
    }
    if profile.saturation.as_ref().is_some_and(|s| s.bleed) {
        bleed_columns(&mut elec, w, h, camera.full_well_e);
    }
    for v in &mut elec {
        *v = v.min(camera.full_well_e);
    }
    if let Some(cti) = &profile.cti {
        apply_cti(
            &mut elec,
            w,
            h,
            cti,
            fs.cti_serial,
            fs.cti_parallel,
            &profile.layout(),
        );
    }
    let mut srng = Rng::new(seed ^ ROW_NOISE_STREAM);
    let rows: Vec<f64> = (0..h)
        .map(|_| {
            if fs.row_noise_adu > 0.0 {
                fs.row_noise_adu * srng.normal()
            } else {
                0.0
            }
        })
        .collect();
    let col_sigma = profile
        .noise
        .as_ref()
        .and_then(|n| n.column_noise_adu)
        .unwrap_or(0.0);
    let layout = profile.layout();
    let (x0, bx) = (layout.window_origin_px[0], layout.binning[0]);
    let cols: Vec<f64> = (0..w)
        .map(|x| {
            if col_sigma > 0.0 {
                let chip_col = (x0 + x * bx) as u64;
                col_sigma * Rng::for_item(profile.detector_seed ^ COLUMN_STREAM, chip_col).normal()
            } else {
                0.0
            }
        })
        .collect();
    let waves: Vec<Wave> = fs
        .patterns
        .iter()
        .map(|p| {
            let (s, c) = libm::sincos(p.angle_deg.to_radians());
            let k = 2.0 * core::f64::consts::PI / p.period_px;
            let xs = (0..w)
                .map(|x| libm::sincos(k * c * x as f64 + p.phase_rad))
                .collect();
            let ys = (0..h).map(|y| libm::sincos(k * s * y as f64)).collect();
            (p.amplitude_adu, xs, ys)
        })
        .collect();
    let field = smooth_field(w, h, fs.smooth_field, seed);
    let fixed = profile
        .noise
        .as_ref()
        .and_then(|n| n.fixed_field.as_ref())
        .map(|f| f.map(w, h));
    let ((lo, hi), integer) = output_range(profile);
    let offset = adu_offset(profile, camera, fs);
    let read_key = seed ^ READ_STREAM;
    let mut video: Vec<f64> = elec
        .iter()
        .enumerate()
        .map(|(i, &photo)| {
            let mut prng = Rng::for_item(read_key, i as u64);
            let read = camera.read_noise_e * prng.normal();
            let mut v = (photo + read) / camera.gain_e_per_adu;
            if fs.extra_white_adu > 0.0 {
                v += fs.extra_white_adu * prng.normal();
            }
            v
        })
        .collect();
    if let Some(l) = profile.noise.as_ref().and_then(|n| n.serial_lag.as_ref()) {
        serial_lag(&mut video, w, l);
    }
    video
        .iter()
        .enumerate()
        .map(|(i, &sig)| {
            let (x, y) = (i % w, i / w);
            let mut v = offset + sig;
            v += rows[y] + cols[x];
            if let Some(f) = &field {
                v += f[i];
            }
            if let Some(f) = &fixed {
                v += f[i];
            }
            v += fs.ramp_adu[0] * (x as f64 / (w.max(2) - 1) as f64 - 0.5)
                + fs.ramp_adu[1] * (y as f64 / (h.max(2) - 1) as f64 - 0.5);
            for (amp, xs, ys) in &waves {
                let ((sa, ca), (sb, cb)) = (xs[x], ys[y]);
                v += amp * (ca * cb - sa * sb);
            }
            if integer { libm::round(v) } else { v }.clamp(lo, hi)
        })
        .collect()
}

/// Cubic B-spline weights at fractional position `t`.
fn bspline(t: f64) -> [f64; 4] {
    let u = 1.0 - t;
    [
        u * u * u / 6.0,
        (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0,
        (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
        t * t * t / 6.0,
    ]
}

/// The smooth background field for one frame (`None` if its RMS is zero):
/// Gaussian values on a grid `scale` pixels apart, cubic B-spline
/// interpolated (smooth, no grid artefacts), then zero mean and exactly
/// the given RMS over the frame. Separable (along x for every grid row,
/// then along y), so O(w h).
fn smooth_field(w: usize, h: usize, (rms, scale): (f64, f64), seed: u64) -> Option<Vec<f64>> {
    if rms <= 0.0 || w == 0 || h == 0 {
        return None;
    }
    let mut rng = Rng::new(seed ^ SMOOTH_FIELD_STREAM);
    let (gw, gh) = (
        (w as f64 / scale) as usize + 4,
        (h as f64 / scale) as usize + 4,
    );
    let grid: Vec<f64> = (0..gw * gh).map(|_| rng.normal()).collect();
    let axis = |n: usize| -> Vec<(usize, [f64; 4])> {
        (0..n)
            .map(|p| {
                let g = p as f64 / scale;
                let i = g.floor();
                (i as usize, bspline(g - i))
            })
            .collect()
    };
    let (ax, ay) = (axis(w), axis(h));
    let mut rowsx = vec![0.0; gh * w];
    for gy in 0..gh {
        for (x, (i, wx)) in ax.iter().enumerate() {
            rowsx[gy * w + x] = (0..4).map(|k| wx[k] * grid[gy * gw + i + k]).sum();
        }
    }
    let mut f = vec![0.0; w * h];
    for (y, (j, wy)) in ay.iter().enumerate() {
        for x in 0..w {
            f[y * w + x] = (0..4).map(|k| wy[k] * rowsx[(j + k) * w + x]).sum();
        }
    }
    let n = f.len() as f64;
    let mean = f.iter().sum::<f64>() / n;
    let sd = (f.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    if sd <= 0.0 {
        return None;
    }
    for v in &mut f {
        *v = (*v - mean) * rms / sd;
    }
    Some(f)
}

/// A target at one instant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetSample {
    /// UTC, ISO 8601.
    pub time_utc: String,
    /// Pixel position where it was drawn at that instant, pointing motion
    /// included (`None` if behind the camera).
    pub xy: Option<[f64; 2]>,
    /// Astrometric RA, degrees: light-time corrected, as a reduction against
    /// the stars (and the header WCS) would report it.
    pub ra_deg: f64,
    /// Astrometric Dec, degrees.
    pub dec_deg: f64,
    /// Range when the light left, km.
    pub range_km: f64,
    /// Sun–target–observer angle, degrees.
    pub phase_deg: f64,
    /// Fraction of the Sun's disc visible from the target.
    pub illumination: f64,
    /// Whether the Earth blocks the line of sight.
    pub occulted: bool,
    /// Magnitude as seen (`None` when dark or blocked).
    pub mag: Option<f64>,
    /// Share of the light that is direct sunlight.
    pub sunlight_fraction: f64,
}

/// A target as drawn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrueTarget {
    /// Name.
    pub name: String,
    /// At shutter open.
    pub start: TargetSample,
    /// At mid-exposure.
    pub mid: TargetSample,
    /// At shutter close.
    pub end: TargetSample,
    /// Electrons drawn (all of it, including any part off the image).
    pub flux_e: f64,
    /// Whether any lit part of the path falls on the image.
    pub on_image: bool,
    /// Straight pieces used to draw the path.
    pub segments: usize,
    /// Light curve: `[seconds after shutter open, e⁻/s]` at the traced
    /// points (at most 200).
    pub light_curve: Vec<[f64; 2]>,
    /// Photons reaching the detector over the exposure (all of the path),
    /// before quantum efficiency and the flat.
    pub photons: f64,
    /// What a detector could see of it, for training labels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<SourceLabel>,
}

/// `tgt` at `s` seconds after shutter open, its pixel position moved by the
/// pointing motion as when drawn.
fn sample(
    g: &Geometry,
    tgt: &Target,
    s: f64,
    motion: Option<&MotionPath>,
) -> Result<TargetSample, SimError> {
    let (v, xy) = g.target_at(tgt, s)?;
    let xy = shifted(xy, s, motion);
    let (ra, dec) = unit_to_radec(v.astrometric);
    let t = g
        .timing
        .ok_or(SimError::BadTime("scene.targets need scene.time"))?;
    Ok(TargetSample {
        time_utc: t.start.add_seconds(s).to_string(),
        xy: xy.map(|p| [p.0, p.1]),
        ra_deg: ra,
        dec_deg: dec,
        range_km: v.range_km,
        phase_deg: v.phase_deg,
        illumination: v.illumination,
        occulted: v.occulted,
        mag: v.mag,
        sunlight_fraction: v.sunlight_fraction,
    })
}

/// Draw `tgt`'s path over the exposure into `img`, electrons, and return its
/// truth, with its drawn path (thinned) and footprint for its label.
///
/// Points more than one image size off the image count as not projectable,
/// so off-field paths are not refined. A faceted target starts from pieces
/// short enough (a third of its glint time) that no glint falls between
/// samples.
fn draw_target(
    g: &Geometry,
    tgt: &Target,
    camera: &Camera,
    img: &mut [f64],
    psf: &Psf,
    motion: Option<&MotionPath>,
    profile_sampling: &crate::path::Sampling,
) -> Result<(TrueTarget, Mark), SimError> {
    let (w, h) = (camera.width, camera.height);
    let far = w.max(h) as f64;
    let inside = |p: (f64, f64), m: f64| {
        p.0 > -m && p.1 > -m && p.0 < w as f64 - 1.0 + m && p.1 < h as f64 - 1.0 + m
    };
    let mut err = None;
    let dur = camera.exposure_s;
    let pieces = match &tgt.brightness {
        crate::target::Brightness::Facets(fm) => fm
            .glint_time_s()
            .map_or(4, |g| (libm::ceil(3.0 * dur / g) as usize).clamp(4, 4000)),
        _ => 4,
    }
    .max(motion.map_or(1, MotionPath::pieces));
    let pts = trace_with(
        |s| match g.target_at(tgt, s) {
            Ok((v, xy)) => PathPoint {
                s,
                xy: shifted(xy, s, motion).filter(|&p| inside(p, far)),
                rate: v.e_per_s,
            },
            Err(e) => {
                err.get_or_insert(e);
                PathPoint {
                    s,
                    xy: None,
                    rate: 0.0,
                }
            }
        },
        dur,
        profile_sampling,
        pieces,
    );
    if let Some(e) = err {
        return Err(e);
    }
    let mut flux_e = 0.0;
    let reach = crate::label::reach(psf, w, h);
    let mark = drawn(img, w, h, &lit_path(&pts), reach, |im| {
        flux_e = draw(im, w, h, &pts, psf);
    });
    let margin = 3.0 * psf.core_sigma();
    let on_image = pts
        .iter()
        .any(|p| p.rate > 0.0 && p.xy.is_some_and(|q| inside(q, margin)));
    let truth = TrueTarget {
        name: tgt.name.clone(),
        start: sample(g, tgt, 0.0, motion)?,
        mid: sample(g, tgt, dur / 2.0, motion)?,
        end: sample(g, tgt, dur, motion)?,
        flux_e,
        on_image,
        segments: pts.len() - 1,
        photons: flux_e / camera.quantum_efficiency.max(1e-12),
        light_curve: {
            let step = pts.len().div_ceil(200).max(1);
            pts.iter().step_by(step).map(|p| [p.s, p.rate]).collect()
        },
        label: None,
    };
    Ok((truth, mark))
}

/// A source's drawn path (thinned) and footprint, kept until the frame is
/// complete to make its label.
type Mark = (Vec<[f64; 2]>, Footprint);

/// Draws a source with `draw` while keeping its footprint near `path`.
fn drawn(
    img: &mut [f64],
    w: usize,
    h: usize,
    path: &[(f64, f64)],
    reach: f64,
    draw: impl FnOnce(&mut [f64]),
) -> Mark {
    (thin(path), capture(img, w, h, path, reach, draw))
}

/// If the scene names a TLE file, read it and add to `scene.targets` every
/// object that comes within the field (plus its motion) during the
/// exposure. Returns the number added.
///
/// Each object's direction is sampled at 9 evenly spaced times; it is kept
/// if any sample lies within the field's half-diagonal of the centre plus
/// the largest angle between successive samples. Objects whose orbits fail
/// to propagate (decayed, bad elements) are skipped. `scene.target_file` is
/// cleared.
///
/// Observed directions are compared with the camera's own boresight, so
/// that aberration and refraction near the horizon count as drawn.
///
/// # Errors
///
/// [`SimError::Orbit`] if the file cannot be read, an error if it cannot be
/// parsed, [`SimError::NoPointing`] if the scene has no pointing, and any
/// error from building the scene geometry.
pub fn add_tle_file_targets(scene: &mut Scene, camera: &Camera) -> Result<usize, SimError> {
    let Some(file) = scene.target_file.as_ref() else {
        return Ok(0);
    };
    let text = std::fs::read_to_string(&file.path)
        .map_err(|e| SimError::Orbit(format!("{}: {e}", file.path.display())))?;
    let all = parse_tle_file(&text, &file.brightness)?;
    scene.target_file = None;
    let g = Geometry::new(
        scene.pointing.as_ref(),
        scene.time.as_ref(),
        scene.observer.as_ref(),
        &scene.targets,
        camera,
    )?
    .ok_or(SimError::NoPointing("scene.target_file"))?;
    let (cra, cdec) = g
        .camera_wcs
        .pixel_to_world(g.camera_wcs.crpix0[0], g.camera_wcs.crpix0[1]);
    let centre = radec_to_unit(cra, cdec);
    let reach = g.camera_wcs.field_radius_px(camera.width, camera.height);
    let radius = (reach * camera.pixel_scale_arcsec / 3600.0).to_radians();
    let n = 8;
    let mut added = 0;
    for tgt in all {
        let dirs: Option<Vec<V3>> = (0..=n)
            .map(|k| {
                g.target_at(&tgt, camera.exposure_s * f64::from(k) / f64::from(n))
                    .ok()
                    .map(|(v, _)| v.apparent)
            })
            .collect();
        let Some(dirs) = dirs else { continue };
        let step = dirs
            .windows(2)
            .map(|d| angle(d[0], d[1]))
            .fold(0.0, f64::max);
        if dirs.iter().any(|&d| angle(d, centre) < radius + step) {
            scene.targets.push(tgt);
            added += 1;
        }
    }
    Ok(added)
}

/// A rendered frame.
#[derive(Clone, Debug)]
pub struct Frame {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Noiseless expected signal, electrons (sky + dark + sources).
    pub expected_e: Vec<f64>,
    /// Delivered image, ADU, row-major from the bottom row (FITS order):
    /// rounded and clipped as the output type requires.
    pub adu: Vec<f64>,
    /// Pixel type to write.
    pub output: crate::fits::OutputType,
    /// The answer key.
    pub truth: Truth,
}

/// A [`SimError::BadSource`] for source `index` of `kind`.
fn bad(kind: &'static str, index: usize, reason: &'static str) -> SimError {
    SimError::BadSource {
        kind,
        index,
        reason,
    }
}

/// One brightness from `flux_e` or `mag`, whichever was given.
fn brightness(
    camera: &Camera,
    flux_e: Option<f64>,
    mag: Option<f64>,
    kind: &'static str,
    i: usize,
) -> Result<(f64, f64), SimError> {
    match (flux_e, mag) {
        (Some(f), None) if f > 0.0 => Ok((f, camera.electrons_to_mag(f))),
        (Some(_), None) => Err(bad(kind, i, "flux_e must be positive")),
        (None, Some(m)) => Ok((camera.mag_to_electrons(m), m)),
        (None, None) => Err(bad(kind, i, "give flux_e or mag")),
        (Some(_), Some(_)) => Err(bad(kind, i, "give flux_e or mag, not both")),
    }
}

/// Whether a star given by sky position is below a ground site's horizon:
/// the Earth hides it, so it is neither drawn nor listed in the truth.
/// Stars without exactly one brightness are left for `resolve_star` to refuse.
fn below_horizon(s: &Star, geom: Option<&Geometry>) -> bool {
    match (geom, s.ra_deg, s.dec_deg) {
        (Some(g), Some(ra), Some(dec))
            if g.atmosphere.is_some()
                && s.x.is_none()
                && s.y.is_none()
                && (s.mag.is_some() != s.flux_e.is_some()) =>
        {
            g.below_horizon(g.apparent(radec_to_unit(ra, dec), s.parallax_mas.unwrap_or(0.0)))
        }
        _ => false,
    }
}

/// Star `i` with both pixel and sky positions and both brightness forms.
/// A catalogue magnitude moves to the camera band through the colour term
/// ([`Camera::star_mag`]).
///
/// A magnitude is above the atmosphere and is dimmed by the extinction at
/// the boresight; a given flux is what was detected. The truth reports the
/// magnitude as seen, like a target's.
fn resolve_star(
    s: &Star,
    i: usize,
    camera: &Camera,
    geom: Option<&Geometry>,
) -> Result<TrueStar, SimError> {
    let cat_mag = s.mag.map(|m| camera.star_mag(m, s.color_index, s.teff_k));
    let (mut flux_e, mut mag) = brightness(camera, s.flux_e, cat_mag, "star", i)?;
    if s.flux_e.is_none() {
        let tr = geom.map_or(1.0, Geometry::transmission);
        flux_e *= tr;
        mag -= 2.5 * libm::log10(tr);
    }
    let (x, y, ra, dec) = match (s.x, s.y, s.ra_deg, s.dec_deg) {
        (Some(x), Some(y), None, None) => {
            let sky = geom.map(|g| g.pixel_to_sky(x, y));
            (x, y, sky.map(|s| s.0), sky.map(|s| s.1))
        }
        (None, None, Some(ra), Some(dec)) => {
            let g = geom.ok_or(SimError::NoPointing("a star given by ra_deg/dec_deg"))?;
            let (x, y) = g
                .sky_to_pixel_par(ra, dec, s.parallax_mas.unwrap_or(0.0))
                .ok_or(bad("star", i, "90° or more from the pointing"))?;
            (x, y, Some(ra), Some(dec))
        }
        _ => return Err(bad("star", i, "give x and y, or ra_deg and dec_deg")),
    };
    Ok(TrueStar {
        id: s.id,
        x,
        y,
        ra_deg: ra,
        dec_deg: dec,
        mag,
        flux_e,
        parallax_mas: s.parallax_mas,
        color_index: s.color_index,
        crlb_px: None,
        crlb_flux_e: None,
        photons: flux_e / camera.quantum_efficiency.max(1e-12),
        detected_e: flux_e,
        trail_start: None,
        trail_end: None,
        label: None,
    })
}

/// Streak `i` with its brightness resolved and, with a pointing, its ends on
/// the sky.
fn resolve_streak(
    k: &Streak,
    i: usize,
    camera: &Camera,
    geom: Option<&Geometry>,
) -> Result<TrueStreak, SimError> {
    let (mut flux_e, mut mag) = brightness(camera, k.flux_e, k.mag, "streak", i)?;
    if k.flux_e.is_none() {
        let tr = geom.map_or(1.0, Geometry::transmission);
        flux_e *= tr;
        mag -= 2.5 * libm::log10(tr);
    }
    let radec = |p: [f64; 2]| {
        geom.map(|g| {
            let (a, d) = g.pixel_to_sky(p[0], p[1]);
            [a, d]
        })
    };
    Ok(TrueStreak {
        start: k.start,
        end: k.end,
        start_radec: radec(k.start),
        end_radec: radec(k.end),
        mag,
        flux_e,
        label: None,
    })
}

/// Render `scene` through a bare camera (no defects). Same inputs and seed,
/// same pixels.
///
/// # Errors
///
/// As [`render_profile`].
pub fn render(scene: &Scene, camera: &Camera, seed: u64) -> Result<Frame, SimError> {
    render_profile(scene, &Profile::from(camera.clone()), seed)
}

/// The optical axis, pixels: the camera's `optical_axis_px`, or the image
/// centre.
fn optical_axis(camera: &Camera) -> (f64, f64) {
    let (w, h) = (camera.width, camera.height);
    camera
        .optical_axis_px
        .map_or(((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0), |a| {
            (a[0], a[1])
        })
}

/// Render `scene` through an instrument profile. Same inputs and seed, same
/// pixels.
///
/// The stages run in the order given in the [module docs](self). A passband
/// on the camera first sets its zeropoint and the Sun's magnitude. The sky
/// comes from the profile's zodiacal table when there is one and the scene
/// geometry allows, converted from V to the camera band through the Sun's
/// colour; otherwise from the drawn settings or the camera. Hot pixels are a
/// detector-fixed map scaled to the frame temperature; flickering pixels
/// are on for a random part of the exposure; the radiation rate comes from
/// the profile's map at the sub-satellite point when there is one (the map
/// is for orbiting observers: a ground site gets the base rate). With
/// tracking or pointing motion every star is traced along its path and its
/// trail ends go into the truth; in a static frame stars are drawn in place
/// and get Cramér–Rao bounds from the expected image's own per-pixel
/// variance (signal, sky, dark, read and extra noise). A star's detected
/// flux takes the flat averaged over every pixel its light fell on, so a
/// trailed or wide star is not judged by its centre alone. Read-out faults garble rows
/// after digitisation; with an open clip (float output) garbled rows span
/// the frame's own values rather than the whole float range.
///
/// A scene that names a catalogue must have it loaded first with
/// [`crate::catalog::add_catalog_stars`], and one that names a TLE file with
/// [`add_tle_file_targets`].
///
/// # Errors
///
/// - [`SimError::BadProfile`] if the profile or its passband is invalid or
///   its settings cannot be drawn;
/// - [`SimError::CatalogNotLoaded`] or [`SimError::TargetFileNotLoaded`] if
///   the scene still names a catalogue or a TLE file;
/// - [`SimError::BadSource`] for a star or streak with a missing, doubled
///   or non-positive brightness, or a position not given as pixels or sky;
/// - [`SimError::NoPointing`] for sky-positioned stars or targets without a
///   pointing;
/// - [`SimError::BadTime`] and orbit errors from the scene's time, observer
///   and targets.
pub fn render_profile(scene: &Scene, profile: &Profile, seed: u64) -> Result<Frame, SimError> {
    let resolved;
    let profile = match profile
        .camera
        .with_passband()
        .map_err(SimError::BadProfile)?
    {
        Some(camera) => {
            resolved = Profile {
                camera,
                ..profile.clone()
            };
            &resolved
        }
        None => profile,
    };
    let camera = &profile.camera;
    profile.validate().map_err(SimError::BadProfile)?;
    let settings = profile
        .draw(&mut Profile::settings_rng(seed))
        .map_err(SimError::BadProfile)?;
    if scene.catalog.is_some() {
        return Err(SimError::CatalogNotLoaded);
    }
    let has_sources = !scene.stars.is_empty()
        || scene.random_stars.is_some()
        || !scene.streaks.is_empty()
        || scene.random_streaks.is_some()
        || !scene.targets.is_empty();
    if camera.exposure_s == 0.0 && has_sources {
        return Err(bad(
            "scene",
            0,
            "camera.exposure_s 0 makes a bias frame, which holds no stars, streaks or targets",
        ));
    }
    let (w, h) = (camera.width, camera.height);
    let timing = timing(scene.time.as_ref(), camera)?;
    let geom = Geometry::new(
        scene.pointing.as_ref(),
        scene.time.as_ref(),
        scene.observer.as_ref(),
        &scene.targets,
        camera,
    )?;
    if scene.target_file.is_some() {
        return Err(SimError::TargetFileNotLoaded);
    }
    let mut rng = Rng::new(seed);

    let mut star_specs = scene.stars.clone();
    if let Some(r) = &scene.random_stars {
        if !(r.min_flux_e > 0.0 && r.min_flux_e <= r.max_flux_e && r.max_flux_e.is_finite()) {
            return Err(bad(
                "random_stars",
                0,
                "needs 0 < min_flux_e ≤ max_flux_e, both finite",
            ));
        }
        star_specs.extend(r.draw(w, h, &mut rng));
    }
    let mut stars = star_specs
        .iter()
        .enumerate()
        .filter(|(_, s)| !below_horizon(s, geom.as_ref()))
        .map(|(i, s)| resolve_star(s, i, camera, geom.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut streak_specs = scene.streaks.clone();
    if let Some(r) = &scene.random_streaks {
        streak_specs.extend(r.draw(w, h, &mut Rng::new(seed ^ STREAK_STREAM)));
    }
    let mut streaks = streak_specs
        .iter()
        .enumerate()
        .map(|(i, k)| resolve_streak(k, i, camera, geom.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;

    let axis = optical_axis(camera);
    let field = crate::psf::Field {
        axis,
        r_ref: 0.5 * libm::hypot(w as f64, h as f64),
        radial: settings.psf_field.0,
        tangential: settings.psf_field.1,
    };
    let mut psf = Psf::with_wings(camera.fwhm_px, &settings.psf_wings);
    if let Some((f, fw, b)) = settings.psf_tail {
        psf = psf.with_tail(f, fw, b);
    }
    let psf = psf
        .blurred(settings.blur_arcsec / camera.pixel_scale_arcsec)
        .with_field(Some(field));
    let motion = profile.pointing.as_ref().and_then(|spec| {
        let mut mrng = Rng::new(seed ^ MOTION_STREAM);
        let mut m = MotionPath::draw(
            spec,
            camera.exposure_s,
            camera.pixel_scale_arcsec,
            &mut mrng,
        );
        if m.roll_rad_per_s != 0.0 {
            m.roll_centre_px = Some(spec.roll_centre_px.unwrap_or([
                0.5 * (camera.width as f64 - 1.0),
                0.5 * (camera.height as f64 - 1.0),
            ]));
        }
        (!m.is_still()).then_some(m)
    });
    let shift = |p: Option<(f64, f64)>, t: f64| shifted(p, t, motion.as_ref());
    let pieces = motion.as_ref().map_or(4, |m| m.pieces().max(4));
    let samp = profile.sampling.clone().unwrap_or_default();
    let zodi_mag = match (&profile.zodiacal, geom.as_ref().and_then(ecliptic_geometry)) {
        (Some(z), Some((elong, lat))) => Some(z.at(elong, lat)),
        _ => None,
    };
    let sky_e_per_s = match zodi_mag {
        Some(v) => {
            let m = v + (camera.sun_mag - (-26.74));
            libm::pow(10.0, -0.4 * (m - camera.zeropoint_mag))
                * camera.pixel_scale_arcsec
                * camera.pixel_scale_arcsec
        }
        None => settings.sky_e_per_s.unwrap_or_else(|| camera.sky_e_per_s()),
    };
    let dark_e_per_s = settings.dark_e_per_s.unwrap_or(camera.dark_e_per_s);
    let base = sky_e_per_s * camera.exposure_s;
    let mut img = vec![base; w * h];
    let tracking = geom.as_ref().filter(|g| g.is_tracking());
    let static_frame = motion.is_none() && tracking.is_none();
    let reach = crate::label::reach(&psf, w, h);
    let mut star_marks = Vec::with_capacity(stars.len());
    if static_frame {
        for s in &stars {
            star_marks.push(drawn(&mut img, w, h, &[(s.x, s.y)], reach, |im| {
                psf.point(im, w, h, s.x, s.y, s.flux_e);
            }));
        }
    } else {
        let dur = camera.exposure_s;
        for s in &mut stars {
            let (ra, dec) = (s.ra_deg.unwrap_or(0.0), s.dec_deg.unwrap_or(0.0));
            let par = s.parallax_mas.unwrap_or(0.0);
            let rate = s.flux_e / dur;
            let (sx, sy) = (s.x, s.y);
            let mut err = None;
            let pts = trace_with(
                |t| {
                    let base = match tracking {
                        Some(g) => g.sky_to_pixel_at_par(ra, dec, par, t).unwrap_or_else(|e| {
                            err.get_or_insert(e);
                            None
                        }),
                        None => Some((sx, sy)),
                    };
                    PathPoint {
                        s: t,
                        xy: shift(base, t),
                        rate,
                    }
                },
                dur,
                &samp,
                pieces,
            );
            if let Some(e) = err {
                return Err(e);
            }
            star_marks.push(drawn(&mut img, w, h, &lit_path(&pts), reach, |im| {
                draw(im, w, h, &pts, &psf);
            }));
            let end = |p: Option<&PathPoint>| p.and_then(|p| p.xy).map(|(x, y)| [x, y]);
            s.trail_start = end(pts.first());
            s.trail_end = end(pts.last());
        }
    }
    let mut targets = Vec::with_capacity(scene.targets.len());
    let mut target_marks = Vec::with_capacity(scene.targets.len());
    if !scene.targets.is_empty() {
        let g = geom.as_ref().ok_or(SimError::NoPointing("scene.targets"))?;
        for tgt in &scene.targets {
            let (t, mark) = draw_target(g, tgt, camera, &mut img, &psf, motion.as_ref(), &samp)?;
            targets.push(t);
            target_marks.push(mark);
        }
    }
    let mut streak_marks = Vec::with_capacity(streaks.len());
    for k in &mut streaks {
        let (p0, p1) = ((k.start[0], k.start[1]), (k.end[0], k.end[1]));
        if motion.is_none() {
            streak_marks.push(drawn(&mut img, w, h, &[p0, p1], reach, |im| {
                psf.streak(im, w, h, p0, p1, k.flux_e);
            }));
            continue;
        }
        let dur = camera.exposure_s;
        let moved = |p: (f64, f64), t: f64| shift(Some(p), t).map_or([p.0, p.1], |q| [q.0, q.1]);
        k.start = moved(p0, 0.0);
        k.end = moved(p1, dur);
        let pts = trace_with(
            |t| {
                let u = t / dur;
                let p = (p0.0 + u * (p1.0 - p0.0), p0.1 + u * (p1.1 - p0.1));
                PathPoint {
                    s: t,
                    xy: shift(Some(p), t),
                    rate: k.flux_e / dur,
                }
            },
            dur,
            &samp,
            pieces,
        );
        streak_marks.push(drawn(&mut img, w, h, &lit_path(&pts), reach, |im| {
            draw(im, w, h, &pts, &psf);
        }));
    }

    let stray = match &geom {
        Some(g) if !profile.stray_light.is_empty() => {
            add_stray_light(&mut img, camera, g, &profile.stray_light)
        }
        _ => Vec::new(),
    };
    let mut dust = Vec::new();
    let flat = profile.flat.as_ref().map(|fl| {
        let layout = profile.layout();
        let key = format!(
            "{fl:?}|{layout:?}|{w}x{h}|{:?}|{:?}|{}",
            axis.0.to_bits(),
            axis.1.to_bits(),
            profile.detector_seed
        );
        profile.cache.flat(key, || {
            crate::flat::flat_map(fl, &layout, w, h, axis, profile.detector_seed)
        })
    });
    if let Some(cached) = &flat {
        let (map, sh) = (&cached.0, &cached.1);
        img.iter_mut().zip(map).for_each(|(v, f)| *v *= f);
        for (st, (_, fp)) in stars.iter_mut().zip(&star_marks) {
            if let Some(f) = fp.weighted_mean(map, w) {
                st.detected_e = st.flux_e * f;
            }
        }
        dust.clone_from(sh);
    }
    if settings.smear != (0.0, 0.0) {
        apply_smear(&mut img, w, h, camera.exposure_s, settings.smear);
    }
    let dark_total = dark_e_per_s * camera.exposure_s;
    if dark_total != 0.0 {
        for v in &mut img {
            *v += dark_total;
        }
    }
    let mut hot = Vec::new();
    if let Some(hp) = &profile.hot_pixels {
        let factor = match (&profile.dark, settings.temperature_c) {
            (Some(d), Some(t)) => dark_factor(t, d.ref_temp_c, d.doubling_c),
            _ => 1.0,
        };
        let mut hrng = Rng::new(profile.detector_seed ^ HOT_STREAM);
        for (x, y, r) in hot_pixel_map(hp, &profile.layout(), w, h, &mut hrng) {
            let e_per_s = r * factor;
            img[y * w + x] += e_per_s * camera.exposure_s;
            hot.push(HotPixel { x, y, e_per_s });
        }
    }
    let mut flicker = Vec::new();
    if let Some(rp) = &profile.rts_pixels {
        let layout = profile.layout();
        let temp = settings.temperature_c.unwrap_or(rp.ref_temp_c);
        let amp = match &profile.dark {
            Some(d) => dark_factor(temp, rp.ref_temp_c, d.doubling_c),
            None => dark_factor(temp, rp.ref_temp_c, None),
        };
        let (fh, fl) = (
            crate::detector::arrhenius(temp, rp.ref_temp_c, rp.activation_high_ev),
            crate::detector::arrhenius(temp, rp.ref_temp_c, rp.activation_low_ev),
        );
        let mut mrng = Rng::new(profile.detector_seed ^ RTS_STREAM);
        let mut frng = Rng::new(seed ^ RTS_FRAME_STREAM);
        for px in crate::detector::rts_map(rp, &layout, &mut mrng) {
            let Some((x, y)) = layout.chip_to_image(px.xc, px.yc, w, h) else {
                continue;
            };
            let f = crate::detector::high_fraction(
                px.t_high * fh,
                px.t_low * fl,
                camera.exposure_s,
                &mut frng,
            );
            let e = px.high * amp * f * camera.exposure_s;
            img[y * w + x] += e;
            if f > 0.0 {
                flicker.push(HotPixel {
                    x,
                    y,
                    e_per_s: px.high * amp * f,
                });
            }
        }
    }
    let mut charge = vec![0.0; w * h];
    let mut radiation = None;
    if let Some(rad) = &profile.radiation {
        let sub = match (timing, scene.observer.as_ref()) {
            (_, Some(Orbit::Site(_))) => None,
            (Some(tm), Some(orbit)) => {
                let r = orbit.state(tm.mid)?.position_km;
                (crate::vec3::norm(r) > crate::orbit::R_EARTH)
                    .then(|| crate::frames::subpoint(r, tm.mid))
            }
            _ => None,
        };
        let rate = match (&rad.rate_map, sub) {
            (Some(m), Some((lat, lon, _))) => m.rate(lat, lon) * settings.radiation_map_scale,
            _ => settings.radiation_rate_per_cm2_s,
        };
        let mut rrng = Rng::new(seed ^ RADIATION_STREAM);
        let hits = draw_hits(
            rad,
            rate,
            settings.radiation_collect_s,
            w,
            h,
            camera.pixel_pitch_um,
            profile.layout().binning,
            &mut rrng,
        );
        draw_hit_charge(&mut charge, w, h, &hits, rad.diffusion_px);
        radiation = Some(RadiationTruth {
            subpoint_deg: sub.map(|(la, lo, _)| [la, lo]),
            altitude_km: sub.map(|s| s.2),
            rate_per_cm2_s: rate,
            collect_s: settings.radiation_collect_s,
            hits,
        });
    }
    let extra = settings.extra_white_adu * camera.gain_e_per_adu;
    let floor = camera.read_noise_e * camera.read_noise_e + extra * extra;
    let noise = Noise {
        img: &img,
        w,
        flat: flat.as_ref().map(|c| c.0.as_slice()),
        floor,
        full_well: saturation_e(profile, camera, &settings),
    };
    let labelled = |marks: Vec<Mark>| -> Vec<SourceLabel> {
        marks
            .into_iter()
            .map(|(p, fp)| label(p, &fp, &noise))
            .collect()
    };
    for (st, l) in stars.iter_mut().zip(labelled(star_marks)) {
        st.label = Some(l);
    }
    for (t, l) in targets.iter_mut().zip(labelled(target_marks)) {
        t.label = Some(l);
    }
    for (k, l) in streaks.iter_mut().zip(labelled(streak_marks)) {
        k.label = Some(l);
    }
    if static_frame {
        for st in &mut stars {
            let var = |i: i64, j: i64| {
                if i < 0 || j < 0 || i >= w as i64 || j >= h as i64 {
                    return 0.0;
                }
                img[j as usize * w + i as usize] + floor
            };
            if let Some((sx, sy, sf)) = psf.crlb(st.x, st.y, st.flux_e, var) {
                st.crlb_px = Some([sx, sy]);
                st.crlb_flux_e = Some(sf);
            }
        }
    }
    let mut adu = digitise(&img, w, h, camera, profile, &settings, &charge, seed);
    let readout_bands = match &profile.readout_faults {
        Some(f) => {
            let mut frng = Rng::new(seed ^ READOUT_FAULT_STREAM);
            let bands = f.draw(h, &mut frng);
            let ((mut lo, mut hi), integer) = output_range(profile);
            let finite = |v: f64| v.is_finite() && v.abs() < 1e12;
            if !finite(lo) {
                lo = adu.iter().copied().fold(f64::INFINITY, f64::min);
            }
            if !finite(hi) {
                hi = adu.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            }
            let range = (lo, hi.max(lo));
            crate::detector::apply_bands(&mut adu, w, &bands, range, integer, f, &mut frng);
            bands
        }
        None => vec![],
    };

    Ok(Frame {
        width: w,
        height: h,
        expected_e: img,
        adu,
        output: profile
            .processing
            .as_ref()
            .map(|p| p.output)
            .unwrap_or_default(),
        truth: Truth {
            seed,
            camera: camera.clone(),
            time: timing.map(TimeTruth::from),
            observer: geom.as_ref().and_then(|g| {
                g.observer.map(|st| {
                    let b = radec_to_unit(g.pointing.ra_deg, g.pointing.dec_deg);
                    let site = match &scene.observer {
                        Some(Orbit::Site(s)) => Some(s.clone()),
                        _ => None,
                    };
                    let atm = g.atmosphere;
                    ObserverTruth {
                        gcrs: st,
                        velocity_bary_km_s: scale(g.beta, crate::geometry::C_KM_S),
                        site,
                        atmosphere: atm,
                        airmass: g.airmass(),
                        altitude_deg: atm.map(|a| a.altitude_deg(b)),
                        refraction_arcsec: atm
                            .map(|a| angle(b, a.unrefract(b)).to_degrees() * 3600.0),
                    }
                })
            }),
            wcs: geom.as_ref().map(|g| g.sky_wcs),
            header_wcs: wrong_header_wcs(profile, geom.as_ref().map(|g| &g.sky_wcs), seed),
            pointing: geom.as_ref().map(|g| PointingTruth {
                apparent: g.pointing,
                reported: reported_pointing(
                    g,
                    profile,
                    seed,
                    motion.as_ref(),
                    scene.header_bias_arcsec,
                ),
                quaternion: g.quaternion(),
                camera_wcs: g.camera_wcs,
                aberration_arcsec: g.aberration_arcsec(),
                sky_wcs_fit_px: g.sky_wcs_fit_px,
            }),
            sky_e_per_s,
            stars,
            streaks,
            targets,
            instrument: InstrumentTruth {
                name: profile.name.clone(),
                settings,
                hot_pixels: hot,
                flickering_pixels: flicker,
                radiation,
                pointing_motion: motion.as_ref().map(MotionPath::thinned),
                stray_light: stray,
                dust,
                zodiacal_v_mag_arcsec2: zodi_mag,
                readout_bands,
            },
        },
    })
}

impl Truth {
    /// FITS header cards describing this frame.
    pub fn fits_cards(&self) -> Vec<(&'static str, Value, &'static str)> {
        self.skytwin_cards(true)
    }

    /// Header cards in an instrument's style (see [`crate::header`]).
    ///
    /// Without a style, the same cards as [`Truth::fits_cards`]. With one,
    /// the style's cards, preceded by the simulator's own when the style
    /// keeps them (the style's replace those of the same name), then the
    /// header WCS
    /// when the style asks for it: the faulty one if
    /// [`header_wcs`](Truth::header_wcs) is set.
    pub fn styled_cards(
        &self,
        style: Option<&crate::header::HeaderStyle>,
        binning: [usize; 2],
    ) -> Vec<(String, Value, String)> {
        let Some(st) = style else {
            return self
                .skytwin_cards(true)
                .into_iter()
                .map(|(k, v, c)| (k.to_string(), v, c.to_string()))
                .collect();
        };
        let mut out: Vec<(String, Value, String)> = Vec::new();
        let styled = crate::header::style_cards(st, self, binning);
        if st.skytwin_keywords {
            out.extend(
                self.skytwin_cards(false)
                    .into_iter()
                    .filter(|(k, _, _)| !styled.iter().any(|(sk, _, _)| sk == k))
                    .map(|(k, v, c)| (k.to_string(), v, c.to_string())),
            );
        }
        out.extend(styled);
        if st.wcs {
            if let Some(wcs) = self.header_wcs.as_ref().or(self.wcs.as_ref()) {
                out.extend(
                    wcs.fits_cards()
                        .into_iter()
                        .map(|(k, v, c)| (k.to_string(), v, c.to_string())),
                );
            }
        }
        out
    }

    /// The simulator's own cards: exposure, camera, time, observer and pointing,
    /// and the sky WCS if `with_wcs`.
    fn skytwin_cards(&self, with_wcs: bool) -> Vec<(&'static str, Value, &'static str)> {
        let cam = &self.camera;
        let mut cards = vec![
            ("EXPTIME", Value::Float(cam.exposure_s), "exposure time, s"),
            ("GAIN", Value::Float(cam.gain_e_per_adu), "e-/ADU"),
            ("RDNOISE", Value::Float(cam.read_noise_e), "read noise, e-"),
            (
                "MAGZERO",
                Value::Float(cam.zeropoint_mag),
                "mag giving 1 e-/s",
            ),
            (
                "PIXSCALE",
                Value::Float(cam.pixel_scale_arcsec),
                "arcsec/pixel",
            ),
            ("SIMSEED", Value::Int(self.seed as i64), "Skytwin RNG seed"),
            (
                "CREATOR",
                Value::Str(format!("Skytwin {}", env!("CARGO_PKG_VERSION"))),
                "",
            ),
        ];
        if let Some(p) = &cam.passband {
            cards.push((
                "MAGSYS",
                Value::Str("AB".into()),
                "magnitude system of MAGZERO",
            ));
            if !p.name.is_empty() {
                cards.push(("FILTER", Value::Str(p.name.clone()), "passband"));
            }
        }
        if let Some(t) = self.instrument.settings.temperature_c {
            cards.push(("CCDTEMP", Value::Float(t), "CCD temperature, C"));
        }
        if let Some([la, lo]) = self
            .instrument
            .radiation
            .as_ref()
            .and_then(|r| r.subpoint_deg)
        {
            cards.push(("SUB_LAT", Value::Float(la), "sub-satellite latitude, deg"));
            cards.push(("SUB_LON", Value::Float(lo), "sub-satellite longitude, deg"));
        }
        if let Some(t) = &self.time {
            cards.extend([
                (
                    "TIMESYS",
                    Value::Str("UTC".into()),
                    "time scale of DATE and MJD keywords",
                ),
                ("DATE-OBS", Value::Str(t.start_utc.clone()), "shutter open"),
                ("DATE-BEG", Value::Str(t.start_utc.clone()), "shutter open"),
                ("DATE-AVG", Value::Str(t.mid_utc.clone()), "mid-exposure"),
                ("DATE-END", Value::Str(t.end_utc.clone()), "shutter close"),
                ("MJD-OBS", Value::Float(t.mjd_start), "MJD at shutter open"),
                (
                    "MJD-AVG",
                    Value::Float(t.mjd_start + cam.exposure_s / 2.0 / 86_400.0),
                    "MJD at mid-exposure",
                ),
            ]);
        }
        if let Some(o) = &self.observer {
            let [x, y, z] = o.gcrs.position_km;
            let [vx, vy, vz] = o.gcrs.velocity_km_s;
            cards.extend([
                ("OBS_X", Value::Float(x), "observer GCRS x at DATE-AVG, km"),
                ("OBS_Y", Value::Float(y), "observer GCRS y, km"),
                ("OBS_Z", Value::Float(z), "observer GCRS z, km"),
                ("OBS_VX", Value::Float(vx), "observer GCRS vx, km/s"),
                ("OBS_VY", Value::Float(vy), "observer GCRS vy, km/s"),
                ("OBS_VZ", Value::Float(vz), "observer GCRS vz, km/s"),
            ]);
            if let Some(site) = &o.site {
                cards.extend([
                    (
                        "SITELAT",
                        Value::Float(site.latitude_deg),
                        "site geodetic latitude, deg (WGS84)",
                    ),
                    (
                        "SITELONG",
                        Value::Float(site.longitude_deg),
                        "site longitude, deg east (WGS84)",
                    ),
                    (
                        "SITEELEV",
                        Value::Float(site.height_m),
                        "site height above the ellipsoid, m",
                    ),
                ]);
            }
            if let Some(x) = o.airmass {
                cards.push(("AIRMASS", Value::Float(x), "airmass at the boresight"));
            }
        }
        if let Some(p) = &self.pointing {
            let [w, x, y, z] = p.quaternion;
            cards.extend([
                (
                    "ATT_RA",
                    Value::Float(p.reported.ra_deg),
                    "reported pointing RA, deg",
                ),
                (
                    "ATT_DEC",
                    Value::Float(p.reported.dec_deg),
                    "reported pointing Dec, deg",
                ),
                (
                    "ATT_ROLL",
                    Value::Float(p.reported.roll_deg),
                    "PA of image +y, E of N, deg",
                ),
                ("ATT_QW", Value::Float(w), "camera->GCRS quaternion, scalar"),
                ("ATT_QX", Value::Float(x), "camera->GCRS quaternion, x"),
                ("ATT_QY", Value::Float(y), "camera->GCRS quaternion, y"),
                ("ATT_QZ", Value::Float(z), "camera->GCRS quaternion, z"),
            ]);
        }
        if with_wcs {
            if let Some(wcs) = &self.wcs {
                cards.extend(wcs.fits_cards());
            }
        }
        cards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attitude::PointingSpec;
    use crate::camera::Sky;
    use crate::scene::RandomStars;
    use crate::target::Brightness;
    use crate::wcs::Pointing;

    fn px_star(x: f64, y: f64, flux: f64) -> Star {
        Star {
            x: Some(x),
            y: Some(y),
            flux_e: Some(flux),
            ..Star::default()
        }
    }

    fn scene() -> Scene {
        Scene {
            stars: vec![px_star(50.0, 50.0, 20_000.0)],
            random_stars: Some(RandomStars {
                count: 30,
                max_flux_e: 50_000.0,
                min_flux_e: 200.0,
            }),
            streaks: vec![Streak {
                start: [10.0, 80.0],
                end: [110.0, 20.0],
                flux_e: Some(30_000.0),
                mag: None,
            }],
            ..Scene::default()
        }
    }

    fn camera() -> Camera {
        Camera {
            width: 128,
            height: 96,
            ..Camera::default()
        }
    }

    #[test]
    fn deterministic() {
        let a = render(&scene(), &camera(), 1).unwrap();
        let b = render(&scene(), &camera(), 1).unwrap();
        assert_eq!(a.adu, b.adu);
        let c = render(&scene(), &camera(), 2).unwrap();
        assert_ne!(a.adu, c.adu);
    }

    /// A fractional clip limit on an integer output gives whole-number
    /// pixels inside the limit, the same in every reader.
    #[test]
    fn fractional_clip_limits_give_whole_number_pixels() {
        let p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 32, "height": 32, "bias_adu": 0.0, "sky": { "e_per_s": 0.0 } },
            "processing": { "clip_min_adu": 3.75, "clip_max_adu": 9.5 }
        }))
        .unwrap();
        let f = render_profile(&Scene::default(), &p, 1).unwrap();
        assert!(f
            .adu
            .iter()
            .all(|&v| v == v.round() && (4.0..=9.0).contains(&v)));
        assert!(f.adu.contains(&4.0));
    }

    /// Two windows on one chip see the same column offsets where they
    /// overlap.
    #[test]
    fn column_offsets_follow_the_chip_column() {
        let frame = |x0: usize| {
            let p: Profile = serde_json::from_value(serde_json::json!({
                "camera": { "width": 8, "height": 4, "read_noise_e": 0.0, "dark_e_per_s": 0.0,
                            "sky": { "e_per_s": 0.0 }, "bias_adu": 1000.0 },
                "processing": { "output": "f32" },
                "noise": { "column_noise_adu": 5.0 },
                "detector": { "chip_px": [12, 4], "window_origin_px": [x0, 0] }
            }))
            .unwrap();
            render_profile(&Scene::default(), &p, 1).unwrap().adu
        };
        let (a, b) = (frame(0), frame(4));
        for x in 4..8 {
            assert_eq!(a[x], b[x - 4], "chip column {x}");
        }
        assert!(a[..8].iter().any(|&v| (v - 1000.0).abs() > 0.5));
    }

    /// A camera no detector can have gives an error, not a frame.
    #[test]
    fn impossible_camera_is_refused() {
        for cam in [
            Camera {
                gain_e_per_adu: 0.0,
                ..camera()
            },
            Camera {
                exposure_s: -1.0,
                ..camera()
            },
            Camera {
                pixel_scale_arcsec: f64::NAN,
                ..camera()
            },
            Camera {
                width: 0,
                ..camera()
            },
            Camera {
                sky: Sky::EPerS(-5.0),
                ..camera()
            },
        ] {
            match render(&scene(), &cam, 1) {
                Err(SimError::BadProfile(msg)) => assert!(msg.starts_with("camera"), "{msg}"),
                other => panic!("expected BadProfile, got {:?}", other.map(|f| f.adu.len())),
            }
        }
        let bias = Camera {
            exposure_s: 0.0,
            ..camera()
        };
        assert!(render(&Scene::default(), &bias, 1).is_ok());
    }

    #[test]
    fn sky_level_matches_camera() {
        let cam = Camera {
            width: 200,
            height: 200,
            sky: Sky::EPerS(20.0),
            ..Camera::default()
        };
        let f = render(&Scene::default(), &cam, 3).unwrap();
        let mean = f.adu.iter().sum::<f64>() / f.adu.len() as f64;
        let expect = cam.bias_adu + (20.0 + cam.dark_e_per_s) * cam.exposure_s / cam.gain_e_per_adu;
        assert!((mean - expect).abs() < 0.5, "mean {mean} expect {expect}");
    }

    #[test]
    fn truth_lists_random_stars() {
        let f = render(&scene(), &camera(), 1).unwrap();
        assert_eq!(f.truth.stars.len(), 31);
        assert_eq!(f.truth.streaks.len(), 1);
        assert!(f.truth.wcs.is_none());
    }

    #[test]
    fn sky_star_lands_where_the_wcs_says() {
        let mut s = Scene {
            pointing: Some(PointingSpec::from(Pointing {
                ra_deg: 83.8,
                dec_deg: -5.4,
                roll_deg: 25.0,
            })),
            ..Scene::default()
        };
        s.stars.push(Star {
            ra_deg: Some(83.81),
            dec_deg: Some(-5.39),
            mag: Some(9.0),
            ..Star::default()
        });
        let f = render(&s, &camera(), 1).unwrap();
        let t = &f.truth.stars[0];
        let (ra, dec) = f.truth.wcs.unwrap().pixel_to_world(t.x, t.y);
        assert!((ra - 83.81).abs() < 1e-9 && (dec + 5.39).abs() < 1e-9);
        assert!((t.flux_e - camera().mag_to_electrons(9.0)).abs() < 1e-6);
    }

    /// Observer far out on the night side; target 7,000 km from the Earth's
    /// centre, in the umbra or 9,000 km clear of it.
    #[test]
    fn target_in_umbra_draws_nothing_and_sunlit_one_does() {
        use crate::orbit::{Orbit, StateVector};
        use crate::target::{Brightness, Target};
        use crate::time::Utc;
        use crate::vec3::{scale, unit};
        let start = "2026-11-17T03:00:00";
        let anti_sun = scale(
            unit(crate::ephem::sun_geocentric(
                Utc::parse(start).unwrap().tt_centuries(),
            )),
            -1.0,
        );
        let sv = |r: f64, along: f64| {
            let p = scale(anti_sun, r);
            Orbit::StateVector(StateVector {
                epoch_utc: start.into(),
                position_km: [p[0], p[1], p[2] + along],
                velocity_km_s: [0.0, 0.0, 0.0],
                j2: false,
                frame: crate::orbit::StateFrame::default(),
            })
        };
        let cam = Camera {
            width: 64,
            height: 64,
            exposure_s: 0.1,
            ..Camera::default()
        };
        let mut flux = vec![];
        for along in [0.0, 9_000.0] {
            let obs = sv(42_000.0, 0.0);
            let tgt = Target {
                name: "t".into(),
                orbit: sv(7_000.0, along),
                brightness: Brightness::Mag(8.0),
                earthshine: false,
                moonshine: false,
            };
            let st = obs.state(Utc::parse(start).unwrap()).unwrap();
            let tr = tgt.orbit.state(Utc::parse(start).unwrap()).unwrap();
            let d = unit(crate::vec3::sub(tr.position_km, st.position_km));
            let (ra, dec) = crate::vec3::unit_to_radec(d);
            let s = Scene {
                time: Some(crate::geometry::TimeSpec {
                    start_utc: start.into(),
                }),
                observer: Some(obs),
                pointing: Some(PointingSpec::from(Pointing {
                    ra_deg: ra,
                    dec_deg: dec,
                    roll_deg: 0.0,
                })),
                targets: vec![tgt],
                ..Scene::default()
            };
            let f = render(&s, &cam, 1).unwrap();
            let t = &f.truth.targets[0];
            flux.push((t.flux_e, t.mid.illumination, t.mid.mag));
        }
        assert_eq!(flux[0], (0.0, 0.0, None));
        assert_eq!(flux[1].1, 1.0);
        assert!((flux[1].0 - cam.mag_to_electrons(8.0)).abs() < 1e-6 * flux[1].0);
    }

    fn noisy_profile() -> Profile {
        serde_json::from_value(serde_json::json!({
            "camera": { "width": 256, "height": 256, "sky": { "e_per_s": 0.0 }, "dark_e_per_s": 0.0,
                        "read_noise_e": 0.0, "gain_e_per_adu": 1.0, "bias_adu": 1000.0 },
            "processing": { "pedestal_adu": 24 },
            "noise": { "extra_white_adu": 28,
                       "patterns": [{ "period_px": 4.3, "angle_deg": 43, "amplitude_adu": 10 }],
                       "column_noise_adu": 3 }
        }))
        .unwrap()
    }

    /// Pedestal 24 ADU under about 29.7 ADU of noise: Φ(−24/29.7) ≈ 21% of
    /// pixels clip to 0.
    #[test]
    fn processing_clips_like_real_data() {
        let p = noisy_profile();
        let f = render_profile(&Scene::default(), &p, 5).unwrap();
        let zeros = f.adu.iter().filter(|&&v| v == 0.0).count() as f64 / f.adu.len() as f64;
        assert!((0.17..0.25).contains(&zeros), "{zeros}");
        assert_eq!(f.truth.instrument.settings.pedestal_adu, 24.0);
    }

    #[test]
    fn column_offsets_belong_to_the_detector() {
        let mut p = noisy_profile();
        p.processing = None;
        p.noise.as_mut().unwrap().extra_white_adu = None;
        p.noise.as_mut().unwrap().patterns.clear();
        let col_means = |seed| {
            let f = render_profile(&Scene::default(), &p, seed).unwrap();
            (0..256)
                .map(|x| (0..256).map(|y| f.adu[y * 256 + x]).sum::<f64>() / 256.0)
                .collect::<Vec<_>>()
        };
        let (a, b) = (col_means(1), col_means(2));
        let d: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(d < 0.6, "column pattern changed between frames: {d}");
        let spread =
            a.iter().copied().fold(f64::MIN, f64::max) - a.iter().copied().fold(f64::MAX, f64::min);
        assert!(spread > 5.0, "{spread}");
    }

    /// Projecting the image onto the drawn wave recovers its amplitude.
    #[test]
    fn pattern_has_the_set_period_and_amplitude() {
        let mut p = noisy_profile();
        p.processing = None;
        p.noise.as_mut().unwrap().extra_white_adu = Some(2.0.into());
        p.noise.as_mut().unwrap().column_noise_adu = None;
        let f = render_profile(&Scene::default(), &p, 9).unwrap();
        let d = &f.truth.instrument.settings.patterns[0];
        let (s, c) = libm::sincos(d.angle_deg.to_radians());
        let k = 2.0 * core::f64::consts::PI / d.period_px;
        let (mut cc, mut n) = (0.0, 0.0);
        for (i, &v) in f.adu.iter().enumerate() {
            let (x, y) = ((i % 256) as f64, (i / 256) as f64);
            cc += (v - 1000.0) * libm::cos(k * (c * x + s * y) + d.phase_rad);
            n += 1.0;
        }
        let amp = 2.0 * cc / n;
        assert!((amp - 10.0).abs() < 0.3, "{amp}");
    }

    #[test]
    fn hot_pixels_stay_put_and_follow_temperature() {
        let mut p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 128, "height": 128 },
            "temperature_c": -30.0,
            "dark": { "e_per_s_at_ref": 0.1, "ref_temp_c": -30.0 },
            "hot_pixels": { "density_per_mpix": 2000, "e_per_s_at_ref": { "log_uniform": [10, 1000] } }
        }))
        .unwrap();
        let a = render_profile(&Scene::default(), &p, 1)
            .unwrap()
            .truth
            .instrument
            .hot_pixels;
        let b = render_profile(&Scene::default(), &p, 2)
            .unwrap()
            .truth
            .instrument
            .hot_pixels;
        assert_eq!(
            a, b,
            "the hot-pixel map belongs to the detector, not the frame"
        );
        assert_eq!(a.len(), 33);
        p.temperature_c = Some((-20.0).into());
        let c = render_profile(&Scene::default(), &p, 1).unwrap();
        let f = crate::detector::dark_factor(-20.0, -30.0, None);
        assert!((c.truth.instrument.hot_pixels[0].e_per_s / a[0].e_per_s - f).abs() < 1e-9);
        assert!((c.truth.instrument.settings.dark_e_per_s.unwrap() - 0.1 * f).abs() < 1e-12);
    }

    /// An Earth-fixed observer over (−30°, −45°) at 800 km is in the map's
    /// high cell (50 /cm²/s: about 500 hits on 1 cm² in 10 s); a geocentric
    /// scene falls back to the base rate.
    #[test]
    fn radiation_rate_follows_the_map_at_the_subpoint() {
        use crate::orbit::{Orbit, StateFrame, StateVector};
        let (lat, lon) = (-30f64.to_radians(), -45f64.to_radians());
        let r = 6_378.137 + 800.0;
        let pos = [
            r * lat.cos() * lon.cos(),
            r * lat.cos() * lon.sin(),
            r * lat.sin(),
        ];
        let p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 256, "height": 256, "exposure_s": 10.0, "pixel_pitch_um": 39.0625 },
            "radiation": { "rate_per_cm2_s": 1.0,
                "rate_map": { "lat_edges_deg": [-90, -50, 0, 90], "lon_edges_deg": [-180, -90, 20, 180],
                              "rate_per_cm2_s": [[1, 1, 1], [1, 50, 1], [1, 1, 1]] } }
        }))
        .unwrap();
        let start = "2026-11-17T03:00:00";
        let s = Scene {
            time: Some(crate::geometry::TimeSpec {
                start_utc: start.into(),
            }),
            observer: Some(Orbit::StateVector(StateVector {
                epoch_utc: start.into(),
                position_km: pos,
                velocity_km_s: [0.0; 3],
                j2: false,
                frame: StateFrame::Itrs,
            })),
            ..Scene::default()
        };
        let f = render_profile(&s, &p, 3).unwrap();
        let rad = f.truth.instrument.radiation.unwrap();
        let [la, lo] = rad.subpoint_deg.unwrap();
        assert!(
            (la + 30.0).abs() < 0.3 && (lo + 45.0).abs() < 0.3,
            "{la} {lo}"
        );
        assert_eq!(rad.rate_per_cm2_s, 50.0);
        assert!((420..580).contains(&rad.hits.len()), "{}", rad.hits.len());
        let g = render_profile(&Scene::default(), &p, 3).unwrap();
        assert_eq!(g.truth.instrument.radiation.unwrap().rate_per_cm2_s, 1.0);
    }

    /// A 1e5 e⁻/s source (1e6 e⁻ over 10 s) adds dwell × rate to every pixel
    /// of its column.
    #[test]
    fn smear_follows_the_dwell_times() {
        let (w, h) = (8, 100);
        let mut img = vec![0.0; w * h];
        img[50 * w + 3] = 1.0e6;
        apply_smear(&mut img, w, h, 10.0, (6.6e-5, 1.8e-5));
        assert!((img[80 * w + 3] - 6.6).abs() < 1e-9, "{}", img[80 * w + 3]);
        assert!((img[10 * w + 3] - 1.8).abs() < 1e-9);
        assert_eq!(img[80 * w + 2], 0.0);
    }

    /// Four wells of excess go two up and two down, all in the source's column.
    #[test]
    fn bleeding_conserves_charge_along_the_column() {
        let (w, h) = (5, 40);
        let mut img = vec![0.0; w * h];
        img[20 * w + 2] = 500_000.0;
        bleed_columns(&mut img, w, h, 100_000.0);
        let total: f64 = img.iter().sum();
        assert!((total - 500_000.0).abs() < 1e-6);
        assert!(img.iter().all(|&v| v <= 100_000.0));
        assert_eq!((18..=22).map(|y| img[y * w + 2]).sum::<f64>(), 500_000.0);
        assert_eq!(img[20 * w + 1], 0.0);
    }

    #[test]
    fn header_pointing_reports_the_reference_pixel() {
        let mut p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 200, "height": 100 },
            "pointing": { "header": { "reference_px": [150.0, 20.0] } }
        }))
        .unwrap();
        let s = Scene {
            pointing: Some(PointingSpec::from(Pointing {
                ra_deg: 40.0,
                dec_deg: 10.0,
                roll_deg: 15.0,
            })),
            ..Scene::default()
        };
        let f = render_profile(&s, &p, 1).unwrap();
        let t = f.truth.pointing.unwrap();
        let (x, y) = f
            .truth
            .wcs
            .unwrap()
            .world_to_pixel(t.reported.ra_deg, t.reported.dec_deg)
            .unwrap();
        assert!(
            (x - 150.0).abs() < 1e-6 && (y - 20.0).abs() < 1e-6,
            "{x} {y}"
        );
        p.pointing
            .as_mut()
            .unwrap()
            .header
            .as_mut()
            .unwrap()
            .error_arcsec = Some(36.0.into());
        let t = render_profile(&s, &p, 1).unwrap().truth;
        let (x, y) = t
            .wcs
            .unwrap()
            .world_to_pixel(
                t.pointing.as_ref().unwrap().reported.ra_deg,
                t.pointing.unwrap().reported.dec_deg,
            )
            .unwrap();
        let off = libm::hypot(x - 150.0, y - 20.0) * 3.0;
        assert!(off > 1.0 && off < 150.0, "{off}");
    }

    /// Observer over the sunlit side at 800 km, boresight 12° above the limb:
    /// the glare is log-midway between its 1000 and 10 e⁻/s ends and brightest
    /// on the set side.
    #[test]
    fn earth_glare_follows_the_limb_and_the_set_direction() {
        use crate::orbit::{Orbit, StateFrame, StateVector};
        use crate::vec3::unit_to_radec;
        let start = "2026-11-17T03:00:00";
        let t = crate::time::Utc::parse(start).unwrap();
        let sun = crate::vec3::unit(crate::ephem::sun_geocentric(t.tt_centuries()));
        let r = scale(sun, 6_378.137 + 800.0);
        let half = libm::asin(6_378.137 / (6_378.137 + 800.0));
        let down = scale(sun, -1.0);
        let side = crate::vec3::unit(crate::vec3::cross(down, [0.0, 0.0, 1.0]));
        let a = half + 12f64.to_radians();
        let b = crate::vec3::add(scale(down, libm::cos(a)), scale(side, libm::sin(a)));
        let (ra, dec) = unit_to_radec(b);
        let mk = |toward: f64| -> Profile {
            serde_json::from_value(serde_json::json!({
                "camera": { "width": 128, "height": 128, "sky": { "e_per_s": 0.0 }, "dark_e_per_s": 0.0, "read_noise_e": 0.0 },
                "stray_light": [{ "body": "earth", "angle_deg": [10, 14], "e_per_s": [1000, 10], "gradient": 0.5, "toward_deg": toward }]
            }))
            .unwrap()
        };
        let s = Scene {
            time: Some(crate::geometry::TimeSpec {
                start_utc: start.into(),
            }),
            observer: Some(Orbit::StateVector(StateVector {
                epoch_utc: start.into(),
                position_km: r,
                velocity_km_s: [0.0; 3],
                j2: false,
                frame: StateFrame::Gcrs,
            })),
            pointing: Some(PointingSpec::from(Pointing {
                ra_deg: ra,
                dec_deg: dec,
                roll_deg: 0.0,
            })),
            ..Scene::default()
        };
        let towards = render_profile(&s, &mk(0.0), 1).unwrap();
        let away = render_profile(&s, &mk(180.0), 1).unwrap();
        let st = &towards.truth.instrument.stray_light[0];
        assert!((st.angle_deg - 12.0).abs() < 0.01, "{}", st.angle_deg);
        assert_eq!(st.lit, 1.0);
        assert!((st.e_per_s - 100.0).abs() < 1.0, "{}", st.e_per_s);
        let [ex, ey] = st.direction.unwrap();
        let mean_along = |f: &Frame, sign: f64| -> f64 {
            let mut acc = 0.0;
            for (i, v) in f.adu.iter().enumerate() {
                let (x, y) = ((i % 128) as f64 - 63.5, (i / 128) as f64 - 63.5);
                if sign * (x * ex + y * ey) > 30.0 {
                    acc += v;
                }
            }
            acc
        };
        assert!(mean_along(&towards, 1.0) > mean_along(&towards, -1.0));
        assert!(mean_along(&away, -1.0) > mean_along(&away, 1.0));
    }

    /// Hot pixels in two windows' overlap appear in both, shifted by the
    /// window offset; binned 2×2, four chip pixels feed each image pixel
    /// (20000/Mpix × 0.12 Mpix in all).
    #[test]
    fn hot_pixels_belong_to_the_chip_not_the_window() {
        let mk = |x0: usize, y0: usize| -> Profile {
            serde_json::from_value(serde_json::json!({
                "camera": { "width": 100, "height": 80 },
                "detector": { "chip_px": [400, 300], "window_origin_px": [x0, y0] },
                "hot_pixels": { "density_per_mpix": 20000, "e_per_s_at_ref": 50.0 }
            }))
            .unwrap()
        };
        let a = render_profile(&Scene::default(), &mk(100, 100), 1)
            .unwrap()
            .truth
            .instrument
            .hot_pixels;
        let b = render_profile(&Scene::default(), &mk(110, 90), 1)
            .unwrap()
            .truth
            .instrument
            .hot_pixels;
        let in_a: Vec<(usize, usize)> = a
            .iter()
            .filter(|p| p.x >= 10 && p.y < 70)
            .map(|p| (p.x - 10, p.y + 10))
            .collect();
        assert!(!in_a.is_empty());
        for q in in_a {
            assert!(b.iter().any(|p| (p.x, p.y) == q), "{q:?} missing");
        }
        let mut p = mk(0, 0);
        p.camera.width = 200;
        p.camera.height = 150;
        p.detector.as_mut().unwrap().binning = [2, 2];
        let binned = render_profile(&Scene::default(), &p, 1)
            .unwrap()
            .truth
            .instrument
            .hot_pixels;
        let total: f64 = binned.iter().map(|h| h.e_per_s).sum();
        assert!((total - 50.0 * 2400.0).abs() < 1e-6, "{total}");
    }

    #[test]
    fn optical_axis_carries_the_pointing() {
        let p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 200, "height": 100, "optical_axis_px": [150.0, 30.0] }
        }))
        .unwrap();
        let s = Scene {
            pointing: Some(PointingSpec::from(Pointing {
                ra_deg: 40.0,
                dec_deg: 10.0,
                roll_deg: 15.0,
            })),
            ..Scene::default()
        };
        let w = render_profile(&s, &p, 1).unwrap().truth.wcs.unwrap();
        let (x, y) = w.world_to_pixel(40.0, 10.0).unwrap();
        assert!((x - 150.0).abs() < 1e-9 && (y - 30.0).abs() < 1e-9);
    }

    #[test]
    fn style_cards_replace_skytwin_cards_of_the_same_name() {
        let p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 16, "height": 16, "gain_e_per_adu": 1.2 },
            "header": { "cards": [{ "key": "GAIN", "value": "1.20" }] }
        }))
        .unwrap();
        let f = render_profile(&Scene::default(), &p, 1).unwrap();
        let cards = f.truth.styled_cards(p.header.as_ref(), [1, 1]);
        let gains: Vec<_> = cards.iter().filter(|c| c.0 == "GAIN").collect();
        assert_eq!(gains.len(), 1);
        assert_eq!(gains[0].1, Value::Str("1.20".into()));
    }

    /// The reported pointing is 30″ east and 40″ south of the true one; the
    /// header WCS is 20′ off and turned 5° while the truth keeps both; at
    /// probability 0 the header keeps the true WCS.
    #[test]
    fn header_bias_and_wrong_wcs() {
        let s = Scene {
            pointing: Some(PointingSpec::from(Pointing {
                ra_deg: 40.0,
                dec_deg: 60.0,
                roll_deg: 10.0,
            })),
            header_bias_arcsec: Some([30.0, -40.0]),
            ..Scene::default()
        };
        let prof: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 100, "height": 100, "exposure_s": 1.0, "pixel_scale_arcsec": 3.0 },
            "header": { "wrong_wcs": { "probability": 1.0, "offset_arcmin": 20.0, "roll_deg": 5.0 } }
        }))
        .unwrap();
        let t = render_profile(&s, &prof, 3).unwrap().truth;
        let p = t.pointing.as_ref().unwrap();
        let de = (p.reported.ra_deg - p.apparent.ra_deg) * 3600.0 * 60f64.to_radians().cos();
        let dn = (p.reported.dec_deg - p.apparent.dec_deg) * 3600.0;
        assert!(
            (de - 30.0).abs() < 0.01 && (dn + 40.0).abs() < 0.01,
            "{de} {dn}"
        );
        let (w, h) = (t.wcs.unwrap(), t.header_wcs.unwrap());
        let sep = crate::vec3::angle(
            crate::vec3::radec_to_unit(w.crval_deg[0], w.crval_deg[1]),
            crate::vec3::radec_to_unit(h.crval_deg[0], h.crval_deg[1]),
        )
        .to_degrees()
            * 60.0;
        assert!((sep - 20.0).abs() < 0.05, "{sep}");
        let ang = |m: [[f64; 2]; 2]| m[1][1].atan2(m[0][1]).to_degrees();
        let turn = (ang(h.cd) - ang(w.cd) + 540.0) % 360.0 - 180.0;
        assert!((turn.abs() - 5.0).abs() < 1e-6, "{turn}");
        assert!((h.scale_arcsec() - w.scale_arcsec()).abs() < 1e-9);
        let cards = t.styled_cards(prof.header.as_ref(), [1, 1]);
        let crval1 = cards.iter().find(|c| c.0 == "CRVAL1").unwrap();
        assert_eq!(crval1.1, Value::Float(h.crval_deg[0]));
        let mut p0 = prof.clone();
        p0.header
            .as_mut()
            .unwrap()
            .wrong_wcs
            .as_mut()
            .unwrap()
            .probability = 0.0;
        assert!(render_profile(&s, &p0, 3)
            .unwrap()
            .truth
            .header_wcs
            .is_none());
    }

    /// 30″/s for 4 s at 3″/px trails 40 px, centred on the star; a still frame
    /// has no trail.
    #[test]
    fn trailed_stars_have_their_ends_in_the_truth() {
        let prof: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 200, "height": 200, "exposure_s": 4.0, "pixel_scale_arcsec": 3.0 },
            "pointing": { "drift_arcsec_per_s": 30.0 }
        }))
        .unwrap();
        let s = Scene {
            stars: vec![Star {
                x: Some(100.0),
                y: Some(90.0),
                flux_e: Some(1e5),
                ..Star::default()
            }],
            ..Scene::default()
        };
        let t = render_profile(&s, &prof, 4).unwrap().truth;
        let st = &t.stars[0];
        let (a, b) = (st.trail_start.unwrap(), st.trail_end.unwrap());
        let len = (b[0] - a[0]).hypot(b[1] - a[1]);
        assert!((len - 40.0).abs() < 1e-6, "{len}");
        assert!((f64::midpoint(a[0], b[0]) - 100.0).abs() < 1e-6);
        assert!((f64::midpoint(a[1], b[1]) - 90.0).abs() < 1e-6);
        let still: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 200, "height": 200, "exposure_s": 4.0, "pixel_scale_arcsec": 3.0 }
        }))
        .unwrap();
        assert!(render_profile(&s, &still, 4).unwrap().truth.stars[0]
            .trail_start
            .is_none());
    }

    /// Under pixel-to-pixel flat variation, a star's `detected_e` equals the
    /// electrons it adds to the noiseless image.
    #[test]
    fn detected_electrons_match_the_image_under_prnu() {
        let p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 64, "height": 64, "fwhm_px": 2.0 },
            "flat": { "prnu_rms": 0.25 }
        }))
        .unwrap();
        let star = Star {
            x: Some(32.3),
            y: Some(31.6),
            flux_e: Some(10_000.0),
            ..Star::default()
        };
        let s = Scene {
            stars: vec![star],
            ..Scene::default()
        };
        let added: f64 = footprint(&s, &Scene::default(), &p, 1).iter().sum();
        let t = render_profile(&s, &p, 1).unwrap().truth;
        let d = t.stars[0].detected_e;
        assert!(
            (d - added).abs() < 1e-6 * added,
            "detected {d}, image {added}"
        );
        assert!((d - 10_000.0).abs() > 10.0, "the flat must change it: {d}");
    }

    /// Expected electrons that `with`'s extra sources add over `without`
    /// (same profile and seed, so everything else cancels).
    fn footprint(with: &Scene, without: &Scene, p: &Profile, seed: u64) -> Vec<f64> {
        let a = render_profile(with, p, seed).unwrap().expected_e;
        let b = render_profile(without, p, seed).unwrap().expected_e;
        a.iter().zip(&b).map(|(u, v)| u - v).collect()
    }

    /// Centroid `[x, y]`, direction (degrees in [0, 180)) and length of a
    /// line drawn at a steady rate, from the second moments of its
    /// footprint: a uniform line of length L blurred by an isotropic PSF has
    /// L²/12 more variance along it than across.
    fn line_moments(d: &[f64], w: usize) -> ([f64; 2], f64, f64) {
        let (mut f, mut sx, mut sy) = (0.0, 0.0, 0.0);
        for (i, &v) in d.iter().enumerate() {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            f += v;
            sx += v * x;
            sy += v * y;
        }
        let (cx, cy) = (sx / f, sy / f);
        let (mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0);
        for (i, &v) in d.iter().enumerate() {
            let (x, y) = ((i % w) as f64 - cx, (i / w) as f64 - cy);
            xx += v * x * x;
            yy += v * y * y;
            xy += v * x * y;
        }
        let (xx, yy, xy) = (xx / f, yy / f, xy / f);
        let split = libm::hypot(0.5 * (xx - yy), xy);
        let angle = (0.5 * libm::atan2(2.0 * xy, xx - yy))
            .to_degrees()
            .rem_euclid(180.0);
        ([cx, cy], angle, libm::sqrt(12.0 * 2.0 * split))
    }

    /// Asserts that a labelled segment matches the line actually drawn.
    fn assert_label_is_drawn(a: [f64; 2], b: [f64; 2], d: &[f64], w: usize, tol_px: f64) {
        let (c, angle, len) = line_moments(d, w);
        let mid = [f64::midpoint(a[0], b[0]), f64::midpoint(a[1], b[1])];
        let label_len = (b[0] - a[0]).hypot(b[1] - a[1]);
        let label_angle = (b[1] - a[1])
            .atan2(b[0] - a[0])
            .to_degrees()
            .rem_euclid(180.0);
        let turn = (angle - label_angle).abs();
        assert!(
            (c[0] - mid[0]).hypot(c[1] - mid[1]) < tol_px,
            "centre {c:?} vs {mid:?}"
        );
        assert!(
            (len - label_len).abs() < tol_px,
            "length {len} vs {label_len}"
        );
        assert!(
            turn.min(180.0 - turn).to_radians() * len < tol_px,
            "angle {angle} vs {label_angle}"
        );
    }

    /// A camera that drifts 9″/s (3 px/s at 3″/px) through a 10 s exposure,
    /// with nothing but the sources on the image.
    fn drifting_profile() -> Profile {
        serde_json::from_value(serde_json::json!({
            "camera": { "width": 256, "height": 256, "exposure_s": 10.0, "pixel_scale_arcsec": 3.0,
                        "fwhm_px": 2.0, "full_well_e": 1e12, "sky": { "e_per_s": 0.0 },
                        "dark_e_per_s": 0.0 },
            "pointing": { "drift_arcsec_per_s": 9.0 }
        }))
        .unwrap()
    }

    /// The drift lengthens and turns a pixel streak; its truth gives the
    /// ends that were drawn.
    #[test]
    fn streak_labels_follow_the_pointing_motion() {
        let p = drifting_profile();
        let with = Scene {
            streaks: vec![Streak {
                start: [40.0, 200.0],
                end: [190.0, 140.0],
                flux_e: Some(1e6),
                mag: None,
            }],
            ..Scene::default()
        };
        let d = footprint(&with, &Scene::default(), &p, 3);
        let k = &render_profile(&with, &p, 3).unwrap().truth.streaks[0];
        assert_label_is_drawn(k.start, k.end, &d, 256, 0.05);
        let path = &k.label.as_ref().unwrap().path;
        for (a, b) in [(path[0], k.start), (path[path.len() - 1], k.end)] {
            assert!((a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3);
        }
    }

    /// Through a flat field, with sky, dark current and read noise, each
    /// source's label SNR is `sqrt(Σ sᵢ² / vᵢ)` of its own expected
    /// electrons against the finished image's variance; label paths are
    /// the star's place and the streak's ends.
    #[test]
    fn labels_weigh_each_source_against_the_finished_image() {
        let p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 96, "height": 96, "exposure_s": 2.0, "pixel_scale_arcsec": 3.0,
                        "fwhm_px": 2.0, "read_noise_e": 6.0, "sky": { "e_per_s": 30.0 },
                        "dark_e_per_s": 1.0, "full_well_e": 1e12 },
            "flat": { "prnu_rms": 0.1 }
        }))
        .unwrap();
        let with = Scene {
            stars: vec![px_star(40.3, 50.6, 3e3), px_star(70.0, 20.0, 2e5)],
            streaks: vec![Streak {
                start: [5.0, 90.0],
                end: [90.0, 60.0],
                flux_e: Some(2e4),
                mag: None,
            }],
            ..Scene::default()
        };
        let f = render_profile(&with, &p, 7).unwrap();
        let snr = |without: &Scene| {
            let d = footprint(&with, without, &p, 7);
            d.iter()
                .zip(&f.expected_e)
                .filter(|(s, _)| **s > 0.0)
                .map(|(s, v)| s * s / (v + 36.0))
                .sum::<f64>()
                .sqrt()
        };
        for i in 0..2 {
            let mut without = with.clone();
            without.stars.remove(i);
            let (st, want) = (&f.truth.stars[i], snr(&without));
            let l = st.label.as_ref().unwrap();
            assert!(
                (l.snr - want).abs() < 1e-3 * want,
                "star {i}: {} vs {want}",
                l.snr
            );
            assert_eq!(l.path, vec![[st.x, st.y]]);
        }
        let without = Scene {
            streaks: vec![],
            ..with.clone()
        };
        let l = f.truth.streaks[0].label.as_ref().unwrap();
        let want = snr(&without);
        assert!(
            (l.snr - want).abs() < 1e-3 * want,
            "streak: {} vs {want}",
            l.snr
        );
        assert_eq!(l.path, vec![[5.0, 90.0], [90.0, 60.0]]);
        let faint = f.truth.stars[0].label.as_ref().unwrap();
        let bright = f.truth.stars[1].label.as_ref().unwrap();
        assert!(faint.extent_px < bright.extent_px && faint.peak_snr < faint.snr);
    }

    /// A target that hangs still in front of the stars trails 30 px with
    /// the drift; its start, mid and end samples are where it was drawn.
    #[test]
    fn target_labels_follow_the_pointing_motion() {
        use crate::orbit::{Orbit, StateVector};
        use crate::target::{Brightness, Target};
        use crate::time::Utc;
        use crate::vec3::{scale, unit};
        let start = "2026-11-17T03:00:00";
        let t0 = Utc::parse(start).unwrap();
        let anti_sun = scale(unit(crate::ephem::sun_geocentric(t0.tt_centuries())), -1.0);
        let sv = |r: f64, along: f64| {
            let q = scale(anti_sun, r);
            Orbit::StateVector(StateVector {
                epoch_utc: start.into(),
                position_km: [q[0], q[1], q[2] + along],
                velocity_km_s: [0.0, 0.0, 0.0],
                j2: false,
                frame: crate::orbit::StateFrame::default(),
            })
        };
        let obs = sv(42_000.0, 0.0);
        let tgt = Target {
            name: "t".into(),
            orbit: sv(7_000.0, 9_000.0),
            brightness: Brightness::Mag(6.0),
            earthshine: false,
            moonshine: false,
        };
        let (so, st) = (obs.state(t0).unwrap(), tgt.orbit.state(t0).unwrap());
        let (ra, dec) =
            crate::vec3::unit_to_radec(unit(crate::vec3::sub(st.position_km, so.position_km)));
        let without = Scene {
            time: Some(crate::geometry::TimeSpec {
                start_utc: start.into(),
            }),
            observer: Some(obs),
            pointing: Some(PointingSpec::from(Pointing {
                ra_deg: ra,
                dec_deg: dec,
                roll_deg: 0.0,
            })),
            ..Scene::default()
        };
        let with = Scene {
            targets: vec![tgt],
            ..without.clone()
        };
        let p = drifting_profile();
        let d = footprint(&with, &without, &p, 5);
        let t = &render_profile(&with, &p, 5).unwrap().truth.targets[0];
        let xy = |s: &TargetSample| s.xy.unwrap();
        let (a, b) = (xy(&t.start), xy(&t.end));
        assert!((b[0] - a[0]).hypot(b[1] - a[1]) > 25.0);
        assert_label_is_drawn(a, b, &d, 256, 0.3);
        let m = xy(&t.mid);
        assert!((m[0] - f64::midpoint(a[0], b[0])).hypot(m[1] - f64::midpoint(a[1], b[1])) < 0.3);
    }

    /// One 20 px jump during the exposure splits a star into two images. A
    /// jump before mid-exposure leaves the star after it for the longer
    /// part of the exposure, and that brighter image is where the truth
    /// puts the star.
    #[test]
    fn star_labels_sit_where_the_star_is_at_mid_exposure() {
        let p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 256, "height": 256, "exposure_s": 4.0, "pixel_scale_arcsec": 3.0,
                        "fwhm_px": 2.0, "full_well_e": 1e12, "sky": { "e_per_s": 0.0 },
                        "dark_e_per_s": 0.0 },
            "pointing": { "jump_rate_per_s": 0.25, "jump_arcsec": 60.0 }
        }))
        .unwrap();
        let with = Scene {
            stars: vec![px_star(128.3, 120.6, 1e6)],
            ..Scene::default()
        };
        let mut checked = 0;
        for seed in 0..60 {
            let t = render_profile(&with, &p, seed).unwrap().truth;
            let jumps = t
                .instrument
                .pointing_motion
                .map(|m| m.jumps)
                .unwrap_or_default();
            if jumps.len() != 1 || jumps[0].t_s >= 2.0 {
                continue;
            }
            let d = footprint(&with, &Scene::default(), &p, seed);
            let top = (0..d.len()).max_by(|&i, &j| d[i].total_cmp(&d[j])).unwrap();
            let (x, y) = ((top % 256) as f64, (top / 256) as f64);
            let st = &t.stars[0];
            assert!(
                (x - st.x).abs() <= 0.5 && (y - st.y).abs() <= 0.5,
                "seed {seed}: ({x}, {y}) vs ({}, {})",
                st.x,
                st.y
            );
            checked += 1;
        }
        assert!(checked >= 3, "{checked}");
    }

    /// The final truncation wins over any band it covers; at probability 0 the
    /// pixels are those of a profile without the block.
    #[test]
    fn readout_faults_corrupt_the_rows_in_the_truth() {
        let mk = |p: f64| -> Profile {
            serde_json::from_value(serde_json::json!({
                "camera": { "width": 64, "height": 128, "exposure_s": 1.0, "pixel_scale_arcsec": 3.0 },
                "processing": { "pedestal_adu": 100 },
                "readout_faults": { "probability": p, "bands": 6, "band_rows": 3,
                                    "truncate_probability": 1.0 }
            }))
            .unwrap()
        };
        let s = Scene::default();
        let f = render_profile(&s, &mk(1.0), 9).unwrap();
        let bands = &f.truth.instrument.readout_bands;
        assert_eq!(bands.len(), 7);
        assert_eq!(bands[6].last, 127);
        assert_eq!(bands[6].kind, crate::detector::BandKind::Dropped);
        let row = |y: usize| &f.adu[y * 64..(y + 1) * 64];
        for b in bands {
            assert!(b.last >= b.first);
            for y in b.first..=b.last {
                if y >= bands[6].first {
                    assert!(row(y).iter().all(|&v| v == 0.0));
                    continue;
                }
                match b.kind {
                    crate::detector::BandKind::Dropped => assert!(row(y).iter().all(|&v| v == 0.0)),
                    crate::detector::BandKind::Saturated => {
                        assert!(row(y).iter().all(|&v| v == 65_535.0));
                    }
                    crate::detector::BandKind::Noise => {
                        let z = row(y).iter().filter(|&&v| v == 0.0).count();
                        assert!((10..54).contains(&z), "{z}");
                    }
                }
            }
        }
        let mut none = mk(0.0);
        let g = render_profile(&s, &none, 9).unwrap();
        none.readout_faults = None;
        assert_eq!(g.adu, render_profile(&s, &none, 9).unwrap().adu);
        assert!(g.truth.instrument.readout_bands.is_empty());
    }

    #[test]
    fn profile_sky_is_light_through_the_field_stop() {
        let prof: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 100, "height": 100, "exposure_s": 1.0, "read_noise_e": 0.0,
                        "gain_e_per_adu": 1.0, "bias_adu": 0.0, "dark_e_per_s": 0.0, "sky": { "e_per_s": 5.0 } },
            "processing": { "output": "f32", "bias_subtracted": true },
            "flat": { "field_stop": { "radius_px": 30.0, "edge_px": 1.0, "outside": 0.0 } },
            "sky_e_per_s": 400.0
        }))
        .unwrap();
        let f = render_profile(&Scene::default(), &prof, 2).unwrap();
        let inside: Vec<f64> = (40..60)
            .flat_map(|y| (40..60).map(move |x| (x, y)))
            .map(|(x, y)| f.adu[y * 100 + x])
            .collect();
        let m = inside.iter().sum::<f64>() / inside.len() as f64;
        let sd =
            (inside.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / inside.len() as f64).sqrt();
        assert!((m - 400.0).abs() < 5.0, "{m}");
        assert!((sd - 20.0).abs() < 3.0, "shot noise {sd}");
        assert!(f.adu[2 * 100 + 2].abs() < 1e-9, "corner {}", f.adu[202]);
        assert_eq!(f.truth.sky_e_per_s, 400.0);
    }

    /// A source trails to the left with its flux kept; pure noise gets lag-1
    /// correlation `coefficient` along rows and none down columns.
    #[test]
    fn serial_lag_trails_sources_and_correlates_noise_along_rows() {
        let mk = |lag: Option<serde_json::Value>, read: f64| -> Profile {
            let mut v = serde_json::json!({
                "camera": { "width": 128, "height": 128, "exposure_s": 1.0, "pixel_scale_arcsec": 3.0,
                            "fwhm_px": 1.0, "full_well_e": 1e9, "read_noise_e": read, "gain_e_per_adu": 1.0,
                            "sky": { "e_per_s": 0.0 }, "dark_e_per_s": 0.0, "bias_adu": 0.0 },
                "processing": { "output": "f32", "bias_subtracted": true }
            });
            if let Some(l) = lag {
                v["noise"] = serde_json::json!({ "serial_lag": l });
            }
            serde_json::from_value(v).unwrap()
        };
        let star = Scene {
            stars: vec![Star {
                x: Some(64.0),
                y: Some(64.0),
                flux_e: Some(1e6),
                ..Star::default()
            }],
            ..Scene::default()
        };
        let lag = serde_json::json!({ "coefficient": 0.5 });
        let f = render_profile(&star, &mk(Some(lag.clone()), 0.0), 1).unwrap();
        let g = render_profile(&star, &mk(None, 0.0), 1).unwrap();
        let at = |fr: &Frame, x: usize| fr.adu[64 * 128 + x];
        assert!(at(&f, 60) > 0.03 * at(&f, 64) && at(&g, 60) < 1e-3 * at(&g, 64));
        assert!(at(&f, 68) <= at(&g, 68) + 1e-9);
        let (sf, sg): (f64, f64) = (f.adu.iter().sum(), g.adu.iter().sum());
        assert!((sf / sg - 1.0).abs() < 1e-3, "{sf} {sg}");
        let n = render_profile(&Scene::default(), &mk(Some(lag), 10.0), 3).unwrap();
        let corr = |dx: usize, dy: usize| {
            let (mut s, mut v) = (0.0, 0.0);
            for y in 0..128 - dy {
                for x in 0..128 - dx {
                    let a = n.adu[y * 128 + x];
                    s += a * n.adu[(y + dy) * 128 + x + dx];
                    v += a * a;
                }
            }
            s / v
        };
        assert!((corr(1, 0) - 0.5).abs() < 0.03, "{}", corr(1, 0));
        assert!((corr(2, 0) - 0.25).abs() < 0.03, "{}", corr(2, 0));
        assert!(corr(0, 1).abs() < 0.03, "{}", corr(0, 1));
    }

    /// Overflow draws a line along the star's row toward the read-out (left),
    /// nothing to the right beyond the bleed or on other rows, and nothing
    /// without the overflow block.
    #[test]
    fn serial_overflow_draws_a_line_toward_the_readout() {
        let prof: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 200, "height": 60, "exposure_s": 1.0, "pixel_scale_arcsec": 3.0,
                        "fwhm_px": 2.0, "full_well_e": 50000, "read_noise_e": 0.0, "gain_e_per_adu": 1.0 },
            "processing": { "output": "f32" },
            "saturation": { "bleed": true,
                            "serial_overflow": { "fraction": 0.5, "length_px": 15.0 } }
        }))
        .unwrap();
        let s = Scene {
            stars: vec![Star {
                x: Some(120.0),
                y: Some(30.0),
                flux_e: Some(2e7),
                ..Star::default()
            }],
            ..Scene::default()
        };
        let f = render_profile(&s, &prof, 1).unwrap();
        let at = |x: usize, y: usize| f.adu[y * 200 + x];
        let left: f64 = (60..100).map(|x| at(x, 30)).sum::<f64>() / 40.0;
        let right: f64 = (150..190).map(|x| at(x, 30)).sum::<f64>() / 40.0;
        let off_row: f64 = (60..100).map(|x| at(x, 10)).sum::<f64>() / 40.0;
        assert!(left > 1000.0, "{left}");
        assert!(right < 50.0 && off_row < 50.0, "{right} {off_row}");
        let mut p2 = prof.clone();
        p2.saturation.as_mut().unwrap().serial_overflow = None;
        let g = render_profile(&s, &p2, 1).unwrap();
        let left2: f64 = (60..100).map(|x| g.adu[30 * 200 + x]).sum::<f64>() / 40.0;
        assert!(left2 < 50.0, "{left2}");
    }

    /// Random streaks draw from their own stream: away from them, pixels match
    /// a frame without them.
    #[test]
    fn random_streaks_are_drawn_and_listed() {
        let prof: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 300, "height": 200, "exposure_s": 1.0, "pixel_scale_arcsec": 3.0 }
        }))
        .unwrap();
        let s: Scene = serde_json::from_value(serde_json::json!({
            "random_streaks": { "count": 5, "length_px": { "uniform": [20, 60] }, "mag": 10 }
        }))
        .unwrap();
        let t = render_profile(&s, &prof, 2).unwrap().truth;
        assert_eq!(t.streaks.len(), 5);
        for k in &t.streaks {
            let len = (k.end[0] - k.start[0]).hypot(k.end[1] - k.start[1]);
            assert!((20.0..=60.0).contains(&len), "{len}");
            assert!((k.mag - 10.0).abs() < 1e-9);
        }
        let none = render_profile(&Scene::default(), &prof, 2).unwrap();
        let with = render_profile(&s, &prof, 2).unwrap();
        let far: Vec<usize> = (0..300 * 200)
            .filter(|&i| {
                let (x, y) = ((i % 300) as f64, (i / 300) as f64);
                with.truth.streaks.iter().all(|k| {
                    let (dx, dy) = (k.end[0] - k.start[0], k.end[1] - k.start[1]);
                    let l2 = dx * dx + dy * dy;
                    let u = (((x - k.start[0]) * dx + (y - k.start[1]) * dy) / l2).clamp(0.0, 1.0);
                    (x - k.start[0] - u * dx).hypot(y - k.start[1] - u * dy) > 15.0
                })
            })
            .collect();
        assert!(far.len() > 30_000);
        assert!(far.iter().all(|&i| none.adu[i] == with.adu[i]));
    }

    /// A 30″/s drift moves the reported pointing 60″ between mid-exposure and
    /// either end.
    #[test]
    fn reported_pointing_follows_the_epoch() {
        let mk = |epoch: &str| -> Profile {
            serde_json::from_value(serde_json::json!({
                "camera": { "width": 100, "height": 100, "exposure_s": 4.0, "pixel_scale_arcsec": 3.0 },
                "pointing": { "drift_arcsec_per_s": 30.0, "header": { "epoch": epoch } }
            }))
            .unwrap()
        };
        let s = Scene {
            pointing: Some(PointingSpec::from(Pointing {
                ra_deg: 40.0,
                dec_deg: 0.0,
                roll_deg: 0.0,
            })),
            ..Scene::default()
        };
        let at = |e: &str| {
            render_profile(&s, &mk(e), 2)
                .unwrap()
                .truth
                .pointing
                .unwrap()
                .reported
        };
        let (a, m, z) = (at("start"), at("mid"), at("end"));
        let sep = |p: Pointing, q: Pointing| {
            crate::vec3::angle(
                crate::vec3::radec_to_unit(p.ra_deg, p.dec_deg),
                crate::vec3::radec_to_unit(q.ra_deg, q.dec_deg),
            )
            .to_degrees()
                * 3600.0
        };
        assert!((sep(a, m) - 60.0).abs() < 0.5, "{}", sep(a, m));
        assert!((sep(z, m) - 60.0).abs() < 0.5);
        assert!((sep(a, z) - 120.0).abs() < 0.5);
    }

    #[test]
    fn color_term_moves_catalogue_magnitudes() {
        let mut cam = camera();
        cam.color_term = Some(crate::camera::ColorTerm {
            bands: Some([
                crate::catalog::BandName::ApassB,
                crate::catalog::BandName::ApassV,
            ]),
            coefficient: -0.3,
            default_index: 0.6,
        });
        let star = |ci: Option<f64>| Star {
            x: Some(10.0),
            y: Some(10.0),
            mag: Some(12.0),
            color_index: ci,
            ..Star::default()
        };
        let s = Scene {
            stars: vec![star(Some(1.5)), star(None)],
            ..Scene::default()
        };
        let f = render(&s, &cam, 1).unwrap();
        assert!((f.truth.stars[0].mag - (12.0 - 0.45)).abs() < 1e-12);
        assert!((f.truth.stars[1].mag - (12.0 - 0.18)).abs() < 1e-12);
    }

    /// A red band makes the cool star brighter than the default one.
    #[test]
    fn passband_sets_ab_zeropoint_and_star_colours() {
        let band = |a: f64, b: f64| {
            serde_json::json!({
                "wavelength_nm": [a, b], "throughput": [1.0, 1.0]
            })
        };
        let p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 50, "height": 50, "passband": {
                "name": "R", "response": band(600.0, 700.0), "aperture_area_m2": 0.01,
                "reference": band(400.0, 500.0), "color_index": "b_v"
            } }
        }))
        .unwrap();
        let pb = p.camera.passband.clone().unwrap();
        let star = |ci: Option<f64>, teff: Option<f64>| Star {
            x: Some(25.0),
            y: Some(25.0),
            mag: Some(10.0),
            color_index: ci,
            teff_k: teff,
            ..Star::default()
        };
        let s = Scene {
            stars: vec![
                star(None, Some(3500.0)),
                star(Some(0.65), None),
                star(None, None),
            ],
            ..Scene::default()
        };
        let f = render_profile(&s, &p, 1).unwrap();
        assert!((f.truth.camera.zeropoint_mag - pb.zeropoint_ab()).abs() < 1e-12);
        assert!((f.truth.camera.sun_mag - pb.sun_ab_mag()).abs() < 1e-12);
        let m = |i: usize| f.truth.stars[i].mag;
        assert!((m(0) - 10.0 - pb.color_offset(3500.0)).abs() < 2e-3);
        let sun_like = pb.color_offset(crate::photometry::teff_from_bv(0.65));
        assert!((m(1) - 10.0 - sun_like).abs() < 2e-3);
        assert!((m(2) - 10.0 - pb.color_offset(5800.0)).abs() < 2e-3);
        assert!(m(0) < m(2) - 0.5);
    }

    /// Neighbouring pixels are nearly equal and pixels four scales apart
    /// uncorrelated (averaged over 8 fields: one holds only ~7 cells across).
    #[test]
    fn smooth_field_has_its_rms_and_scale() {
        let f = smooth_field(300, 200, (4.0, 40.0), 9).unwrap();
        let n = f.len() as f64;
        let mean = f.iter().sum::<f64>() / n;
        let rms = (f.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
        assert!(mean.abs() < 1e-9 && (rms - 4.0).abs() < 1e-9);
        let corr = |f: &[f64], d: usize| {
            let mut s = 0.0;
            let mut k = 0.0;
            for y in 0..200 {
                for x in 0..300 - d {
                    s += f[y * 300 + x] * f[y * 300 + x + d];
                    k += 1.0;
                }
            }
            s / k / 16.0
        };
        assert!(corr(&f, 1) > 0.99, "{}", corr(&f, 1));
        let far: f64 = (0..8)
            .map(|sd| corr(&smooth_field(300, 200, (4.0, 40.0), 100 + sd).unwrap(), 160))
            .sum::<f64>()
            / 8.0;
        assert!(far.abs() < 0.2, "{far}");
        assert!(smooth_field(10, 10, (0.0, 5.0), 1).is_none());
    }

    #[test]
    fn truth_reports_photons_and_flat_fielded_electrons() {
        let p: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 100, "height": 100, "quantum_efficiency": 0.5 },
            "flat": { "vignetting": [-0.5, 0.0] }
        }))
        .unwrap();
        let star = |x: f64, y: f64| Star {
            x: Some(x),
            y: Some(y),
            flux_e: Some(1000.0),
            ..Star::default()
        };
        let s = Scene {
            stars: vec![star(49.5, 49.5), star(5.0, 5.0)],
            ..Scene::default()
        };
        let t = render_profile(&s, &p, 1).unwrap().truth;
        assert_eq!(t.stars[0].photons, 2000.0);
        assert!((t.stars[0].detected_e - 1000.0).abs() < 1.0);
        assert!(t.stars[1].detected_e < 700.0, "{}", t.stars[1].detected_e);
    }

    #[test]
    fn rejects_bad_sources() {
        let mut s = Scene::default();
        s.stars.push(Star {
            x: Some(1.0),
            y: Some(1.0),
            ..Star::default()
        });
        assert!(matches!(
            render(&s, &camera(), 1),
            Err(SimError::BadSource { .. })
        ));
        let mut s = Scene::default();
        s.stars.push(Star {
            ra_deg: Some(1.0),
            dec_deg: Some(1.0),
            mag: Some(5.0),
            ..Star::default()
        });
        assert!(matches!(
            render(&s, &camera(), 1),
            Err(SimError::NoPointing(_))
        ));
    }

    /// A TLE file that cannot be read leaves `scene.target_file` in place.
    #[test]
    fn unreadable_tle_file_keeps_the_setting() {
        let mut scene: Scene = serde_json::from_value(serde_json::json!({
            "pointing": { "ra_deg": 10.0, "dec_deg": 20.0 },
            "time": { "start_utc": "2024-01-01T00:00:00Z" },
            "target_file": { "path": "no/such/file.tle", "brightness": { "mag": 8.0 } }
        }))
        .unwrap();
        assert!(add_tle_file_targets(&mut scene, &Camera::default()).is_err());
        assert!(scene.target_file.is_some());
    }

    /// `only_visible` finds a target named only in `scene.target_file`,
    /// and gives the same answer as for the same target listed inline.
    /// The TLE file is written by the test (the two GEO records of
    /// `examples/targets.tle`), so the test also runs from the published
    /// crate, which holds no `examples/`.
    #[test]
    fn only_visible_finds_a_tle_file_target() {
        const TLE: &str = "GEO-1
1 99101U          26321.00000000  .00000000  00000-0  00000+0 0    06
2 99101   0.0500  80.0000 0002000  90.0000 100.0000  1.00273791    07
GEO-2
1 99102U          26321.00000000  .00000000  00000-0  00000+0 0    07
2 99102   0.0800  80.0000 0004000  90.0000 100.1200  1.00271000    07
";
        let tle = std::env::temp_dir().join(format!("skytwin-visible-{}.tle", std::process::id()));
        std::fs::write(&tle, TLE).unwrap();
        let scene: Scene = serde_json::from_value(serde_json::json!({
            "time": { "start_utc": "2026-11-17T12:00:00Z" },
            "observer": { "site": { "latitude_deg": 0.0, "longitude_deg": 0.0, "height_m": 0.0 } },
            "target_file": { "path": tle, "brightness": { "mag": 12.0 } }
        }))
        .unwrap();
        let mut inline = scene.clone();
        inline.target_file = None;
        inline.targets = parse_tle_file(TLE, &Brightness::Mag(12.0)).unwrap();
        for name in ["GEO-1", "GEO-2"] {
            assert_eq!(
                target_visible(&scene, &Camera::default(), name).unwrap(),
                target_visible(&inline, &Camera::default(), name).unwrap(),
                "{name}"
            );
        }
        let nobody = target_visible(&scene, &Camera::default(), "nobody");
        std::fs::remove_file(&tle).unwrap();
        assert!(matches!(nobody, Err(SimError::UnknownTarget(_))));
    }

    /// A camera of 1 e⁻/ADU with a 200,000 e⁻ well clips in its 16-bit
    /// output long before its well fills: a star whose pixels reach 65,535
    /// ADU is labelled saturated.
    #[test]
    fn a_star_clipped_by_the_output_is_saturated() {
        let profile: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 64, "height": 64, "gain_e_per_adu": 1.0, "full_well_e": 200_000.0 }
        }))
        .unwrap();
        let scene: Scene = serde_json::from_value(serde_json::json!({
            "stars": [{ "x": 32.0, "y": 32.0, "flux_e": 1.0e6 }]
        }))
        .unwrap();
        let f = render_profile(&scene, &profile, 1).unwrap();
        assert!(f.adu.iter().any(|&v| v >= 65_535.0));
        assert!(f.truth.stars[0].label.as_ref().unwrap().saturated);
    }

    /// A zero exposure makes a bias frame; one that also holds a star is
    /// refused, since nothing could be drawn for it.
    #[test]
    fn zero_exposure_frames_hold_no_sources() {
        let profile: Profile = serde_json::from_value(serde_json::json!({
            "camera": { "width": 32, "height": 32, "exposure_s": 0.0 }
        }))
        .unwrap();
        assert!(render_profile(&Scene::default(), &profile, 1).is_ok());
        let scene: Scene = serde_json::from_value(serde_json::json!({
            "stars": [{ "x": 16.0, "y": 16.0, "flux_e": 1000.0 }]
        }))
        .unwrap();
        assert!(matches!(
            render_profile(&scene, &profile, 1),
            Err(SimError::BadSource { kind: "scene", .. })
        ));
    }

    /// Random stars with the faintest flux above the brightest are refused.
    #[test]
    fn inverted_random_star_fluxes_are_refused() {
        let scene: Scene = serde_json::from_value(serde_json::json!({
            "random_stars": { "count": 5, "min_flux_e": 1000.0, "max_flux_e": 10.0 }
        }))
        .unwrap();
        assert!(matches!(
            render_profile(&scene, &Profile::default(), 1),
            Err(SimError::BadSource {
                kind: "random_stars",
                ..
            })
        ));
    }

    /// `only_visible` naming no target is an unknown-target error.
    #[test]
    fn only_visible_needs_a_known_target() {
        let scene: Scene = serde_json::from_value(serde_json::json!({
            "time": { "start_utc": "2024-01-01T00:00:00Z" }
        }))
        .unwrap();
        let e = target_visible(&scene, &Camera::default(), "nobody").unwrap_err();
        assert!(
            matches!(e, SimError::UnknownTarget(ref n) if n == "nobody"),
            "{e}"
        );
    }
}
