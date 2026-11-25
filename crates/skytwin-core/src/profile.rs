//! Instrument profiles: everything about one satellite's camera, including
//! its defects, as settings.
//!
//! Every defect block is optional; leaving it out switches the defect off.
//! Settings written as [`Param`] may be fixed or drawn per frame. The draws
//! for a frame are recorded in its truth file.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::camera::Camera;
use crate::detector::{Cti, Dark, HotPixels, Radiation};
use crate::param::{Dist, DrawCtx, Param};
use crate::rng::Rng;
use crate::scene::Scene;

/// What the ground processing did to the image before delivery.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Processing {
    /// The bias level was subtracted (default true).
    #[serde(default = "yes")]
    pub bias_subtracted: bool,
    /// Offset added after processing, ADU.
    #[serde(default)]
    pub pedestal_adu: Param,
    /// Values below this are set to it, ADU (default: the output type's
    /// minimum, e.g. 0 for `u16`).
    #[serde(default)]
    pub clip_min_adu: Option<f64>,
    /// Values above this are set to it, ADU (default: the output type's
    /// maximum).
    #[serde(default)]
    pub clip_max_adu: Option<f64>,
    /// Pixel type delivered: `u16` (default), `i16`, `i32` or `f32`.
    #[serde(default)]
    pub output: crate::fits::OutputType,
}

impl Processing {
    /// Clip limits `(min, max)`, ADU: the given ones, inside the output
    /// type's range. For an integer type they are moved inward to whole
    /// numbers (3.75 → 4), so a clipped pixel is a value the type holds and
    /// every reader of the frame (FITS, Python) sees the same integer.
    pub fn clip_range(&self) -> (f64, f64) {
        let (lo, hi) = self.output.range();
        let (a, b) = (
            self.clip_min_adu.unwrap_or(lo).max(lo),
            self.clip_max_adu.unwrap_or(hi).min(hi),
        );
        if self.output.is_integer() {
            (libm::ceil(a), libm::floor(b))
        } else {
            (a, b)
        }
    }
}

/// Serde default for flags that are on unless switched off.
fn yes() -> bool {
    true
}

/// A coherent periodic pattern (electronic pickup), a plane wave across the
/// image: `A cos(2π (x cos θ + y sin θ) / P + φ)`, with φ random per frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pattern {
    /// Period across the stripes, pixels.
    pub period_px: Param,
    /// Direction of the wave vector, degrees from +x towards +y.
    pub angle_deg: Param,
    /// Amplitude, ADU.
    pub amplitude_adu: Param,
}

/// Read-out noise beyond the camera's read noise.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Noise {
    /// Extra white Gaussian noise, ADU RMS.
    #[serde(default)]
    pub extra_white_adu: Option<Param>,
    /// Coherent patterns.
    #[serde(default)]
    pub patterns: Vec<Pattern>,
    /// Offset per row, ADU RMS, new every frame.
    #[serde(default)]
    pub row_noise_adu: Option<Param>,
    /// Offset per column, ADU RMS, fixed for the detector (`detector_seed`).
    #[serde(default)]
    pub column_noise_adu: Option<f64>,
    /// Linear ramp along y, bottom to top, ADU (a fixed read-out part).
    #[serde(default)]
    pub ramp_y_adu: Option<Param>,
    /// Further ramp along y per second of exposure, ADU/s.
    #[serde(default)]
    pub ramp_y_adu_per_s: Option<Param>,
    /// Linear ramp along x, left to right, ADU.
    #[serde(default)]
    pub ramp_x_adu: Option<Param>,
    /// Smooth random background structure, new every frame.
    #[serde(default)]
    pub smooth_field: Option<SmoothField>,
    /// Slow settling of the read-out chain: each pixel keeps part of the
    /// pixels read before it, signal and read noise alike.
    #[serde(default)]
    pub serial_lag: Option<SerialLag>,
    /// Fixed additive structure, the same every frame (read-out ramps that
    /// are not straight, glow at the edges, a bias pattern).
    #[serde(default)]
    pub fixed_field: Option<FixedField>,
}

/// An additive map, ADU, given on a coarse grid and interpolated
/// bilinearly (held at the edge value beyond the grid). Node `[i][j]` (row
/// `i`, column `j`) sits at pixel `origin_px + (j, i) · spacing_px`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedField {
    /// Pixel `[x, y]` of node `[0][0]`.
    #[serde(default)]
    pub origin_px: [f64; 2],
    /// Node spacing, pixels.
    pub spacing_px: f64,
    /// Values, ADU, rows of nodes from y = origin upward.
    pub values: Vec<Vec<f64>>,
}

impl FixedField {
    /// Check the grid.
    ///
    /// # Errors
    ///
    /// When `spacing_px` is not positive or `values` is not a non-empty
    /// rectangular grid.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.values.first().map_or(0, Vec::len);
        if self.spacing_px.is_nan()
            || self.spacing_px <= 0.0
            || n == 0
            || self.values.iter().any(|r| r.len() != n)
        {
            return Err("noise.fixed_field: spacing_px must be positive and values a non-empty rectangular grid".into());
        }
        Ok(())
    }

    /// The map for a `w × h` image, ADU, row-major from row 0.
    ///
    /// # Panics
    ///
    /// When `values` is empty; [`FixedField::validate`] rejects that.
    pub fn map(&self, w: usize, h: usize) -> Vec<f64> {
        let (ny, nx) = (self.values.len(), self.values[0].len());
        let axis = |p: f64, o: f64, n: usize| -> (usize, usize, f64) {
            let g = ((p - o) / self.spacing_px).clamp(0.0, (n - 1) as f64);
            let i = (g.floor() as usize).min(n.saturating_sub(2));
            let i1 = (i + 1).min(n - 1);
            (i, i1, g - i as f64)
        };
        let xs: Vec<_> = (0..w)
            .map(|x| axis(x as f64, self.origin_px[0], nx))
            .collect();
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            let (j0, j1, ty) = axis(y as f64, self.origin_px[1], ny);
            let (r0, r1) = (&self.values[j0], &self.values[j1]);
            for (x, &(i0, i1, tx)) in xs.iter().enumerate() {
                let a = r0[i0] * (1.0 - tx) + r0[i1] * tx;
                let b = r1[i0] * (1.0 - tx) + r1[i1] * tx;
                out[y * w + x] = a * (1.0 - ty) + b * ty;
            }
        }
        out
    }
}

/// A single-pole low-pass along each row of the read-out signal:
/// `out = (1 − a) · in + a · previous out`, in read order. It conserves
/// flux, trails every source by `a`ᵈ at `d` pixels, and correlates the
/// noise along rows (lag-k correlation `a`ᵏ), which CTI (a charge effect)
/// does not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SerialLag {
    /// The pole `a`, 0 to below 1.
    pub coefficient: f64,
    /// Side the trail extends to: `left` (default; read right to left) or
    /// `right`.
    #[serde(default)]
    pub toward: Toward,
}

/// A smooth random field added to the background each frame: Gaussian
/// values on a grid `scale_px` apart, interpolated with cubic B-splines,
/// then set to zero mean and exactly `rms_adu` RMS over the frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothField {
    /// RMS over the frame, ADU.
    pub rms_adu: Param,
    /// Grid spacing (the size of the features), pixels.
    pub scale_px: Param,
}

/// One instrument.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    /// Name, recorded in the truth file.
    #[serde(default)]
    pub name: String,
    /// Free text.
    #[serde(default)]
    pub description: String,
    /// Seed for things fixed to the detector (column offsets, hot and
    /// flickering pixels, pixel response, dust), so they are the same in
    /// every frame.
    #[serde(default)]
    pub detector_seed: u64,
    /// Detector, optics and exposure basics.
    #[serde(default)]
    pub camera: Camera,
    /// Ground processing.
    #[serde(default)]
    pub processing: Option<Processing>,
    /// Extra read-out noise.
    #[serde(default)]
    pub noise: Option<Noise>,
    /// CCD temperature per frame, °C.
    #[serde(default)]
    pub temperature_c: Option<Param>,
    /// Dark current against temperature (replaces `camera.dark_e_per_s`).
    #[serde(default)]
    pub dark: Option<Dark>,
    /// Hot pixels fixed to the detector.
    #[serde(default)]
    pub hot_pixels: Option<HotPixels>,
    /// Flickering (random telegraph) pixels fixed to the detector.
    #[serde(default)]
    pub rts_pixels: Option<crate::detector::RtsPixels>,
    /// Radiation hits.
    #[serde(default)]
    pub radiation: Option<Radiation>,
    /// Charge-transfer inefficiency.
    #[serde(default)]
    pub cti: Option<Cti>,
    /// Rows lost or garbled in read-out.
    #[serde(default)]
    pub readout_faults: Option<crate::detector::ReadoutFaults>,
    /// PSF wings beyond the camera's core FWHM.
    #[serde(default)]
    pub psf: Option<PsfSpec>,
    /// Smear from charge moving while the shutter is open.
    #[serde(default)]
    pub smear: Option<Smear>,
    /// What happens above full well.
    #[serde(default)]
    pub saturation: Option<Saturation>,
    /// Pointing motion during the exposure.
    #[serde(default)]
    pub pointing: Option<crate::motion::PointingMotion>,
    /// Stray light from the Earth, Sun and Moon.
    #[serde(default)]
    pub stray_light: Vec<crate::stray::StraySource>,
    /// Zodiacal light table; with a time and an observer it replaces
    /// `camera.sky`.
    #[serde(default)]
    pub zodiacal: Option<crate::stray::ZodiTable>,
    /// Where the image sits on the chip (window, binning).
    #[serde(default)]
    pub detector: Option<crate::detector::Layout>,
    /// How the FITS header is written.
    #[serde(default)]
    pub header: Option<crate::header::HeaderStyle>,
    /// How finely moving sources are sampled.
    #[serde(default)]
    pub sampling: Option<crate::path::Sampling>,
    /// Flat field: vignetting, pixel response non-uniformity, dust,
    /// field stop.
    #[serde(default)]
    pub flat: Option<crate::flat::Flat>,
    /// Version of the profile format (1 now; absent in hand-written
    /// profiles from before it).
    #[serde(default)]
    pub schema_version: Option<u32>,
    /// The sensor this profile describes.
    #[serde(default)]
    pub sensor: Option<Sensor>,
    /// The star catalogue it was calibrated against: scenes using the
    /// profile take its settings (see [`crate::catalog::CatalogRequirement`]).
    #[serde(default)]
    pub catalog: Option<crate::catalog::CatalogRequirement>,
    /// How the profile was made (free form: tool, version, data, date).
    #[serde(default)]
    pub learned: Option<serde_json::Value>,
    /// Diffuse light per pixel, e⁻/s, drawn per frame (replaces
    /// `camera.sky`; a zodiacal table still wins). Unlike
    /// `processing.pedestal_adu` it is light: it passes through the flat
    /// (vignetting, field stop) and carries shot noise.
    #[serde(default)]
    pub sky_e_per_s: Option<Param>,
    /// Maps fixed to the detector, built on first use and shared by clones
    /// (frames of a batch reuse them).
    #[serde(skip)]
    pub cache: DetectorCache,
}

/// What a profile describes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sensor {
    /// Instrument, e.g. `CASSIOPE FAI NIR`.
    pub name: String,
    /// Spacecraft or mission.
    #[serde(default)]
    pub platform: Option<String>,
    /// Passband, nm `[from, to]`.
    #[serde(default)]
    pub band_nm: Option<[f64; 2]>,
    /// Anything else worth knowing (mode, binning, references).
    #[serde(default)]
    pub notes: Option<String>,
}

/// The flat map with the inputs it was built from.
type FlatEntry = (
    String,
    std::sync::Arc<(Vec<f64>, Vec<crate::flat::DustShadow>)>,
);

/// Per-detector maps that are the same in every frame. Keyed by their
/// inputs, so a changed profile never gets a stale map.
///
/// Every cache compares equal, so profiles compare by their settings
/// alone.
#[derive(Clone, Default)]
pub struct DetectorCache(std::sync::Arc<std::sync::Mutex<Option<FlatEntry>>>);

impl PartialEq for DetectorCache {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for DetectorCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DetectorCache")
    }
}

impl DetectorCache {
    /// The flat map for these inputs, built once.
    ///
    /// # Panics
    ///
    /// When another thread panicked while holding the cache lock.
    pub fn flat(
        &self,
        key: String,
        build: impl FnOnce() -> (Vec<f64>, Vec<crate::flat::DustShadow>),
    ) -> std::sync::Arc<(Vec<f64>, Vec<crate::flat::DustShadow>)> {
        if let Some((k, v)) = self.0.lock().expect("lock").as_ref() {
            if *k == key {
                return v.clone();
            }
        }
        let v = std::sync::Arc::new(build());
        *self.0.lock().expect("lock") = Some((key, v.clone()));
        v
    }
}

impl Profile {
    /// The chip layout (the whole chip, unbinned, without a detector block).
    pub fn layout(&self) -> crate::detector::Layout {
        self.detector.clone().unwrap_or_else(|| {
            crate::detector::Layout::identity(self.camera.width, self.camera.height)
        })
    }
}

/// One wing of the PSF: a wider Gaussian with a share of the flux.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Wing {
    /// Share of the flux.
    pub fraction: Param,
    /// FWHM, pixels.
    pub fwhm_px: Param,
}

/// PSF beyond the core.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PsfSpec {
    /// Wider Gaussian components; the core keeps the remaining flux.
    #[serde(default)]
    pub wings: Vec<Wing>,
    /// A power-law (Moffat) tail.
    #[serde(default)]
    pub tail: Option<Tail>,
    /// Change across the field.
    #[serde(default)]
    pub field: Option<FieldSpec>,
}

/// A Moffat tail, expanded into Gaussians for rendering.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tail {
    /// Share of the flux.
    pub fraction: Param,
    /// FWHM, pixels.
    pub fwhm_px: Param,
    /// Power-law index β (wings fall as r^−2β; must exceed 1).
    pub beta: Param,
}

/// PSF growth with distance from the optical axis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldSpec {
    /// Fractional growth of σ along the radius at the image's half-diagonal.
    #[serde(default)]
    pub radial_growth: Option<Param>,
    /// Fractional growth of σ across the radius at the half-diagonal.
    #[serde(default)]
    pub tangential_growth: Option<Param>,
}

/// Smear: seconds each row's charge spends under a source while moving.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Smear {
    /// For pixels above a source (higher row numbers), s per row.
    pub above_s_per_row: Param,
    /// For pixels below a source, s per row.
    pub below_s_per_row: Param,
}

/// Saturation behaviour.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Saturation {
    /// Charge above full well spills along the column (default false: lost).
    #[serde(default)]
    pub bleed: bool,
    /// Part of the excess spills along the row instead, as when the serial
    /// register overflows: a horizontal line from a saturated star.
    #[serde(default)]
    pub serial_overflow: Option<SerialOverflow>,
}

/// Charge above full well spilling along the row toward the read-out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SerialOverflow {
    /// Share of each saturated pixel's excess that spills along the row.
    pub fraction: f64,
    /// Decay length of the spill, pixels (exponential).
    pub length_px: f64,
    /// Which way it runs: `left` (default, toward an amplifier at x = 0)
    /// or `right`.
    #[serde(default)]
    pub toward: Toward,
}

/// A direction along the row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Toward {
    /// Toward x = 0.
    #[default]
    Left,
    /// Toward the last column.
    Right,
}

impl SerialOverflow {
    /// Check the settings.
    ///
    /// # Errors
    ///
    /// When `fraction` is outside 0–1 or `length_px` is not positive.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.fraction) || self.length_px.is_nan() || self.length_px <= 0.0
        {
            return Err("saturation.serial_overflow: fraction 0 to 1, length_px > 0".into());
        }
        Ok(())
    }
}

impl From<Camera> for Profile {
    fn from(camera: Camera) -> Self {
        Profile {
            camera,
            ..Profile::default()
        }
    }
}

/// One pattern as drawn for a frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternDraw {
    /// Period, pixels.
    pub period_px: f64,
    /// Wave-vector direction, degrees.
    pub angle_deg: f64,
    /// Amplitude, ADU.
    pub amplitude_adu: f64,
    /// Phase, radians.
    pub phase_rad: f64,
}

/// The settings drawn for one frame.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameSettings {
    /// Pedestal, ADU.
    pub pedestal_adu: f64,
    /// Extra white noise, ADU RMS.
    pub extra_white_adu: f64,
    /// Row-offset noise, ADU RMS.
    pub row_noise_adu: f64,
    /// Patterns.
    pub patterns: Vec<PatternDraw>,
    /// CCD temperature, °C.
    pub temperature_c: Option<f64>,
    /// Dark current at that temperature, e⁻/pixel/s.
    pub dark_e_per_s: Option<f64>,
    /// Radiation base rate, hits/cm²/s.
    pub radiation_rate_per_cm2_s: f64,
    /// Factor on the radiation map.
    pub radiation_map_scale: f64,
    /// Time radiation hits are collected, s.
    pub radiation_collect_s: f64,
    /// Serial CTI: loss per transfer and trail, px.
    pub cti_serial: (f64, f64),
    /// Parallel CTI: loss per transfer and trail, px.
    pub cti_parallel: (f64, f64),
    /// PSF wings: (fraction, FWHM px).
    pub psf_wings: Vec<(f64, f64)>,
    /// PSF tail: (fraction, FWHM px, β).
    pub psf_tail: Option<(f64, f64, f64)>,
    /// PSF growth across the field: (radial, tangential).
    pub psf_field: (f64, f64),
    /// Smear dwell above and below, s per row.
    pub smear: (f64, f64),
    /// Unresolved pointing jitter, arcsec σ.
    pub blur_arcsec: f64,
    /// Read-out ramps across the frame, ADU `[x, y]`.
    pub ramp_adu: [f64; 2],
    /// Smooth background field: RMS (ADU) and scale (pixels).
    #[serde(default)]
    pub smooth_field: (f64, f64),
    /// Diffuse light, e⁻/pixel/s (replaces `camera.sky`).
    #[serde(default)]
    pub sky_e_per_s: Option<f64>,
}

/// Mixes the frame seed into a separate stream for settings, so changing a
/// profile does not shift the pixel noise.
const SETTINGS_STREAM: u64 = 0x5E77_1A65_0F0F_1E57;
/// Mixes `detector_seed` into the stream for detector-fixed maps.
const DETECTOR_STREAM: u64 = 0xDE7E_C70A_0000_0001;
/// Mixes the frame seed into the stream for random pointings.
const POINTING_STREAM: u64 = 0x8A7C_4900_0000_000B;

/// Draw an optional setting, `default` when absent (drawing nothing).
fn draw_or(p: Option<&Param>, default: f64, rng: &mut Rng, ctx: &mut DrawCtx) -> f64 {
    p.map_or(default, |p| p.draw(rng, ctx))
}

/// Read and parse a JSON file; errors name the path.
fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

impl Profile {
    /// Check every setting, including that linked lists in one group have
    /// the same length.
    ///
    /// # Errors
    ///
    /// On the first invalid setting, with a message naming it.
    pub fn validate(&self) -> Result<(), String> {
        self.camera.validate()?;
        for (i, p) in self.params().iter().enumerate() {
            p.validate(&format!("setting {i}"))?;
        }
        if let Some(m) = self.radiation.as_ref().and_then(|r| r.rate_map.as_ref()) {
            m.validate()?;
        }
        if let Some(f) = &self.readout_faults {
            f.validate()?;
        }
        if let Some(o) = self
            .saturation
            .as_ref()
            .and_then(|s| s.serial_overflow.as_ref())
        {
            o.validate()?;
        }
        for s in &self.stray_light {
            s.validate()?;
        }
        if let Some(l) = &self.detector {
            l.validate(self.camera.width, self.camera.height)?;
        }
        if let Some(hs) = &self.header {
            hs.validate()?;
        }
        if let Some(z) = &self.zodiacal {
            z.validate()?;
        }
        if self
            .hot_pixels
            .as_ref()
            .is_some_and(|h| h.density_per_mpix.is_nan() || h.density_per_mpix < 0.0)
        {
            return Err("hot_pixels.density_per_mpix must be ≥ 0".into());
        }
        let mut groups = std::collections::HashMap::<&str, usize>::new();
        for p in self.params() {
            if let Param::Dist(Dist::EmpiricalLinked(l)) = p {
                let n = *groups.entry(l.group.as_str()).or_insert(l.values.len());
                if n != l.values.len() {
                    return Err(format!(
                        "linked group {:?}: lists differ in length",
                        l.group
                    ));
                }
            }
        }
        if let Some(p) = &self.processing {
            let (lo, hi) = p.clip_range();
            if lo > hi {
                return Err("processing: clip_min_adu above clip_max_adu".into());
            }
        }
        if let Some(fs) = self.flat.as_ref().and_then(|f| f.field_stop.as_ref()) {
            if fs.radius_px.is_nan()
                || fs.radius_px <= 0.0
                || fs.edge_px < 0.0
                || !(0.0..=1.0).contains(&fs.outside)
            {
                return Err("flat.field_stop: radius_px must be positive, edge_px not negative, outside 0 to 1".into());
            }
        }
        if let Some(n) = &self.noise {
            if let Some(f) = &n.fixed_field {
                f.validate()?;
            }
            if let Some(l) = &n.serial_lag {
                if !(0.0..1.0).contains(&l.coefficient) {
                    return Err("noise.serial_lag.coefficient must be 0 to below 1".into());
                }
            }
        }
        Ok(())
    }

    /// Every [`Param`] setting, for checks.
    fn params(&self) -> Vec<&Param> {
        let mut v = vec![];
        v.extend(self.sky_e_per_s.iter());
        if let Some(p) = &self.processing {
            v.push(&p.pedestal_adu);
        }
        v.extend(self.temperature_c.iter());
        if let Some(d) = &self.dark {
            v.push(&d.e_per_s_at_ref);
        }
        if let Some(h) = &self.hot_pixels {
            v.push(&h.e_per_s_at_ref);
        }
        if let Some(r) = &self.rts_pixels {
            v.extend([&r.high_e_per_s_at_ref, &r.mean_high_s, &r.mean_low_s]);
        }
        if let Some(r) = &self.radiation {
            v.push(&r.rate_per_cm2_s);
            v.extend(r.map_scale.iter());
            v.extend(r.collect_time_s.iter());
        }
        if let Some(c) = &self.cti {
            for a in c.serial.iter().chain(c.parallel.iter()) {
                v.extend([&a.loss_per_transfer, &a.trail_px]);
            }
        }
        if let Some(p) = &self.psf {
            for wg in &p.wings {
                v.extend([&wg.fraction, &wg.fwhm_px]);
            }
            if let Some(t) = &p.tail {
                v.extend([&t.fraction, &t.fwhm_px, &t.beta]);
            }
            if let Some(f) = &p.field {
                v.extend(f.radial_growth.iter());
                v.extend(f.tangential_growth.iter());
            }
        }
        if let Some(s) = &self.smear {
            v.extend([&s.above_s_per_row, &s.below_s_per_row]);
        }
        if let Some(p) = &self.pointing {
            for q in [
                &p.wobble_rms_arcsec,
                &p.wobble_time_s,
                &p.drift_arcsec_per_s,
                &p.jump_rate_per_s,
                &p.jump_arcsec,
                &p.blur_arcsec,
            ] {
                v.extend(q.iter());
            }
            for o in &p.oscillations {
                v.extend([&o.amplitude_arcsec, &o.period_s]);
                v.extend(o.angle_deg.iter());
            }
        }
        if let Some(n) = &self.noise {
            v.extend(n.ramp_y_adu.iter());
            v.extend(n.ramp_y_adu_per_s.iter());
            v.extend(n.ramp_x_adu.iter());
            v.extend(n.extra_white_adu.iter());
            v.extend(n.row_noise_adu.iter());
            if let Some(f) = &n.smooth_field {
                v.extend([&f.rms_adu, &f.scale_px]);
            }
            for p in &n.patterns {
                v.extend([&p.period_px, &p.angle_deg, &p.amplitude_adu]);
            }
        }
        v
    }

    /// Generator for per-frame settings, from the frame seed.
    pub fn settings_rng(seed: u64) -> Rng {
        Rng::new(seed ^ SETTINGS_STREAM)
    }

    /// Generator for detector-fixed maps, from `detector_seed`.
    pub fn detector_rng(&self) -> Rng {
        Rng::new(self.detector_seed ^ DETECTOR_STREAM)
    }

    /// Draw this frame's settings.
    ///
    /// The order of the draws is part of the reproducibility contract:
    /// settings added later (such as `sky_e_per_s`) are drawn last, so they
    /// leave every earlier setting's values as they were.
    ///
    /// # Errors
    ///
    /// When a drawn value is out of range: a smooth-field scale below 1 px,
    /// a pattern period or PSF FWHM not positive, a tail `beta` not above
    /// 1, or PSF wing and tail fractions adding up to more than 1.
    pub fn draw(&self, rng: &mut Rng) -> Result<FrameSettings, String> {
        let mut fs = FrameSettings::default();
        let ctx = &mut DrawCtx::default();
        if let Some(p) = &self.processing {
            fs.pedestal_adu = p.pedestal_adu.draw(rng, ctx);
        }
        if let Some(n) = &self.noise {
            fs.extra_white_adu = draw_or(n.extra_white_adu.as_ref(), 0.0, rng, ctx).max(0.0);
            fs.row_noise_adu = draw_or(n.row_noise_adu.as_ref(), 0.0, rng, ctx).max(0.0);
            let ry = draw_or(n.ramp_y_adu.as_ref(), 0.0, rng, ctx)
                + draw_or(n.ramp_y_adu_per_s.as_ref(), 0.0, rng, ctx) * self.camera.exposure_s;
            fs.ramp_adu = [draw_or(n.ramp_x_adu.as_ref(), 0.0, rng, ctx), ry];
            if let Some(f) = &n.smooth_field {
                let rms = f.rms_adu.draw(rng, ctx).max(0.0);
                let scale = f.scale_px.draw(rng, ctx);
                if scale < 1.0 {
                    return Err("noise.smooth_field.scale_px must be at least 1".into());
                }
                fs.smooth_field = (rms, scale);
            }
            for pat in &n.patterns {
                let period_px = pat.period_px.draw(rng, ctx);
                if period_px <= 0.0 {
                    return Err("noise pattern period must be positive".into());
                }
                fs.patterns.push(PatternDraw {
                    period_px,
                    angle_deg: pat.angle_deg.draw(rng, ctx),
                    amplitude_adu: pat.amplitude_adu.draw(rng, ctx),
                    phase_rad: rng.range(0.0, 2.0 * core::f64::consts::PI),
                });
            }
        }
        fs.temperature_c = self.temperature_c.as_ref().map(|p| p.draw(rng, ctx));
        if let Some(d) = &self.dark {
            let base = d.e_per_s_at_ref.draw(rng, ctx).max(0.0);
            let f = fs.temperature_c.map_or(1.0, |t| {
                crate::detector::dark_factor(t, d.ref_temp_c, d.doubling_c)
            });
            fs.dark_e_per_s = Some(base * f);
        }
        if let Some(r) = &self.radiation {
            fs.radiation_rate_per_cm2_s = r.rate_per_cm2_s.draw(rng, ctx).max(0.0);
            fs.radiation_map_scale = draw_or(r.map_scale.as_ref(), 1.0, rng, ctx).max(0.0);
            fs.radiation_collect_s =
                draw_or(r.collect_time_s.as_ref(), self.camera.exposure_s, rng, ctx).max(0.0);
        }
        if let Some(c) = &self.cti {
            let mut axis = |a: &Option<crate::detector::CtiAxis>| {
                a.as_ref().map_or((0.0, 0.0), |a| {
                    (
                        a.loss_per_transfer.draw(rng, ctx).clamp(0.0, 1.0),
                        a.trail_px.draw(rng, ctx).max(0.0),
                    )
                })
            };
            fs.cti_serial = axis(&c.serial);
            fs.cti_parallel = axis(&c.parallel);
        }
        if let Some(p) = &self.psf {
            for wg in &p.wings {
                let f = wg.fraction.draw(rng, ctx).clamp(0.0, 1.0);
                let fw = wg.fwhm_px.draw(rng, ctx);
                if fw <= 0.0 {
                    return Err("psf wing FWHM must be positive".into());
                }
                fs.psf_wings.push((f, fw));
            }
            if let Some(t) = &p.tail {
                let f = t.fraction.draw(rng, ctx).clamp(0.0, 1.0);
                let fw = t.fwhm_px.draw(rng, ctx);
                let b = t.beta.draw(rng, ctx);
                if fw <= 0.0 || b <= 1.0 {
                    return Err("psf tail needs fwhm_px > 0 and beta > 1".into());
                }
                fs.psf_tail = Some((f, fw, b));
            }
            let taken =
                fs.psf_wings.iter().map(|w| w.0).sum::<f64>() + fs.psf_tail.map_or(0.0, |t| t.0);
            if taken > 1.0 {
                return Err("psf wing and tail fractions add up to more than 1".into());
            }
            if let Some(f) = &p.field {
                fs.psf_field = (
                    draw_or(f.radial_growth.as_ref(), 0.0, rng, ctx),
                    draw_or(f.tangential_growth.as_ref(), 0.0, rng, ctx),
                );
            }
        }
        if let Some(b) = self.pointing.as_ref().and_then(|p| p.blur_arcsec.as_ref()) {
            fs.blur_arcsec = b.draw(rng, ctx).max(0.0);
        }
        if let Some(s) = &self.smear {
            fs.smear = (
                s.above_s_per_row.draw(rng, ctx).max(0.0),
                s.below_s_per_row.draw(rng, ctx).max(0.0),
            );
        }
        fs.sky_e_per_s = self.sky_e_per_s.as_ref().map(|p| p.draw(rng, ctx).max(0.0));
        Ok(fs)
    }
}

/// Tags of one-of settings (the sky, and every distribution): an object
/// holding one of these is a choice, which a patch replaces, not merges.
const CHOICE_TAGS: [&str; 7] = [
    "e_per_s",
    "mag_per_arcsec2",
    "normal",
    "uniform",
    "log_uniform",
    "empirical",
    "empirical_linked",
];

/// Whether `v` is a one-of setting: a non-empty object keyed only by
/// [`CHOICE_TAGS`].
fn is_choice(v: &Value) -> bool {
    v.as_object()
        .is_some_and(|o| !o.is_empty() && o.keys().all(|k| CHOICE_TAGS.contains(&k.as_str())))
}

/// RFC 7386 JSON merge patch: objects merge key by key, `null` deletes,
/// anything else replaces. One exception: a one-of setting (the sky, a
/// distribution) is replaced whole by another, so `{"e_per_s": 20}` over
/// `{"mag_per_arcsec2": 22.5}` gives the former, not both.
pub fn merge_patch(target: &mut Value, patch: &Value) {
    if is_choice(target) && is_choice(patch) {
        *target = patch.clone();
        return;
    }
    match patch {
        Value::Object(p) => {
            if !target.is_object() {
                *target = Value::Object(serde_json::Map::new());
            }
            if let Value::Object(t) = target {
                for (k, v) in p {
                    if v.is_null() {
                        t.remove(k);
                    } else {
                        merge_patch(t.entry(k.clone()).or_insert(Value::Null), v);
                    }
                }
            }
        }
        other => *target = other.clone(),
    }
}

/// Several frames in a row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sequence {
    /// Number of frames.
    pub frames: usize,
    /// Start-to-start interval, s.
    pub cadence_s: f64,
    /// Keep only frames when this target is sunlit and not blocked by the
    /// Earth, stepping on at the cadence until `frames` are found.
    #[serde(default)]
    pub only_visible: Option<String>,
    /// Most steps to try when `only_visible` is set (default 100 × frames).
    #[serde(default)]
    pub max_steps: Option<usize>,
    /// The boresight wanders from frame to frame (needs a pointing given
    /// as RA/Dec); the header follows it.
    #[serde(default)]
    pub pointing_walk: Option<PointingWalk>,
    /// The spacecraft is still settling after a slew: the boresight starts
    /// off the commanded pointing and converges on it, while the header
    /// reports the commanded pointing throughout.
    #[serde(default)]
    pub settling: Option<Settling>,
}

/// A random walk of the boresight, one step per frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PointingWalk {
    /// Step per frame, arcsec σ per axis (east, north).
    #[serde(default)]
    pub step_arcsec: f64,
    /// Roll step per frame, degrees σ.
    #[serde(default)]
    pub roll_step_deg: f64,
}

/// Settling after a slew: offset `offset_arcsec × exp(−t / time_s)` in a
/// direction drawn once per sequence, `t` from the first frame's start.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settling {
    /// Offset at the first frame, arcsec.
    pub offset_arcsec: Param,
    /// Decay time, s.
    pub time_s: Param,
}

impl Sequence {
    /// Check the pointing walk and settling settings.
    ///
    /// # Errors
    ///
    /// When a walk step is negative or NaN, or a settling setting is
    /// invalid.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(w) = &self.pointing_walk {
            if !(w.step_arcsec >= 0.0 && w.roll_step_deg >= 0.0) {
                return Err("sequence.pointing_walk: steps must be ≥ 0".into());
            }
        }
        if let Some(s) = &self.settling {
            s.offset_arcsec
                .validate("sequence.settling.offset_arcsec")?;
            s.time_s.validate("sequence.settling.time_s")?;
        }
        Ok(())
    }
}

/// Many independent frames for a training set: frame k uses seed + k and
/// draws everything random anew (stars, streaks, noise, every profile
/// distribution).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    /// Number of frames.
    pub frames: usize,
    /// A new random pointing per frame: RA uniform, Dec uniform on the
    /// sphere within `dec_deg`, roll uniform.
    #[serde(default)]
    pub random_pointing: Option<RandomPointing>,
}

/// Pointings spread evenly over a band of the sky.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomPointing {
    /// Declination band, degrees (default the whole sky).
    #[serde(default = "whole_sky")]
    pub dec_deg: [f64; 2],
}

/// Serde default for [`RandomPointing::dec_deg`].
fn whole_sky() -> [f64; 2] {
    [-90.0, 90.0]
}

impl RandomPointing {
    /// A frame's pointing from its `seed`: `(ra, dec, roll)`, degrees. Dec
    /// is uniform in sin(dec), so pointings are even on the sphere.
    pub fn draw(&self, seed: u64) -> (f64, f64, f64) {
        let mut r = Rng::new(seed ^ POINTING_STREAM);
        let (a, b) = (
            libm::sin(self.dec_deg[0].to_radians()),
            libm::sin(self.dec_deg[1].to_radians()),
        );
        let dec = libm::asin(r.range(a.min(b), a.max(b))).to_degrees();
        (r.range(0.0, 360.0), dec, r.range(0.0, 360.0))
    }
}

/// A run: seed, instrument and scene.
#[derive(Clone, Debug)]
pub struct Config {
    /// Frame seed (frame k of a sequence uses seed + k).
    pub seed: u64,
    /// A sequence of frames, if any.
    pub sequence: Option<Sequence>,
    /// Many independent frames, if any.
    pub batch: Option<Batch>,
    /// The instrument, with overrides applied.
    pub profile: Profile,
    /// What is in the field.
    pub scene: Scene,
}

/// Read a scene file.
///
/// Keys: `seed`; `instrument` (a profile file path, or a profile object);
/// `camera` (merged into the profile's camera); `overrides` (merged into the
/// whole profile); `scene`; `sequence` or `batch`. Paths are relative to
/// the working directory.
///
/// # Errors
///
/// When the file cannot be read or parsed, or as [`config_from_value`].
pub fn load_config(path: &Path) -> Result<Config, String> {
    config_from_value(&read_json(path)?, &path.display().to_string())
}

/// A scene file's contents, already parsed (`what` names it in errors).
/// Profile paths are read relative to the working directory, as in
/// [`load_config`]. The scene's `camera` patches a copy of the profile's
/// camera, which then replaces it rather than merging back, so keys the
/// scene deletes with `null` stay deleted.
///
/// # Errors
///
/// On an unknown key, an unreadable or invalid profile, a scene that does
/// not parse or does not meet the profile's catalogue requirement, a bad
/// seed, or an invalid `sequence` or `batch` (or both given).
pub fn config_from_value(v: &Value, what: &str) -> Result<Config, String> {
    let obj = v
        .as_object()
        .ok_or(format!("{what}: expected a JSON object"))?;
    for k in obj.keys() {
        if ![
            "seed",
            "instrument",
            "camera",
            "overrides",
            "scene",
            "sequence",
            "batch",
        ]
        .contains(&k.as_str())
        {
            return Err(format!("{what}: unknown key {k:?}"));
        }
    }
    let mut prof = match obj.get("instrument") {
        None => serde_json::to_value(Profile::default()).map_err(|e| e.to_string())?,
        Some(Value::String(p)) => read_json(Path::new(p))?,
        Some(o @ Value::Object(_)) => o.clone(),
        Some(_) => return Err("instrument: expected a path or an object".into()),
    };
    if let Some(c) = obj.get("camera") {
        let mut cam = match prof.get("camera") {
            Some(c) => c.clone(),
            None => serde_json::to_value(Camera::default()).map_err(|e| e.to_string())?,
        };
        merge_patch(&mut cam, c);
        prof["camera"] = cam;
    }
    if let Some(o) = obj.get("overrides") {
        merge_patch(&mut prof, o);
    }
    let profile: Profile = serde_json::from_value(prof).map_err(|e| format!("instrument: {e}"))?;
    profile.validate().map_err(|e| format!("instrument: {e}"))?;
    let mut scene_v = obj.get("scene").cloned();
    if let (Some(req), Some(cat)) = (
        &profile.catalog,
        scene_v.as_mut().and_then(|s| s.get_mut("catalog")),
    ) {
        req.fill(cat)?;
    }
    let scene: Scene = match scene_v {
        Some(s) => serde_json::from_value(s).map_err(|e| format!("scene: {e}"))?,
        None => Scene::default(),
    };
    let seed = match obj.get("seed") {
        None => 0,
        Some(s) => s.as_u64().ok_or("seed: expected a non-negative integer")?,
    };
    let sequence = match obj.get("sequence") {
        Some(s) => {
            let q: Sequence =
                serde_json::from_value(s.clone()).map_err(|e| format!("sequence: {e}"))?;
            if q.frames == 0 || q.cadence_s.is_nan() || q.cadence_s <= 0.0 {
                return Err("sequence: frames must be at least 1 and cadence_s positive".into());
            }
            if scene.time.is_none() {
                return Err("sequence: needs scene.time".into());
            }
            q.validate()?;
            let radec = scene.pointing.as_ref().is_some_and(|p| {
                p.ra_deg.is_some()
                    && p.dec_deg.is_some()
                    && p.quaternion.is_none()
                    && p.track.is_none()
            });
            if (q.pointing_walk.is_some() || q.settling.is_some()) && !radec {
                return Err(
                    "sequence: pointing_walk and settling need scene.pointing as ra_deg/dec_deg"
                        .into(),
                );
            }
            Some(q)
        }
        None => None,
    };
    let batch = match obj.get("batch") {
        Some(b) => {
            let b: Batch = serde_json::from_value(b.clone()).map_err(|e| format!("batch: {e}"))?;
            if b.frames == 0 {
                return Err("batch: frames must be at least 1".into());
            }
            if sequence.is_some() {
                return Err("batch and sequence: give one or the other".into());
            }
            if let Some(p) = &b.random_pointing {
                let ok = |d: f64| (-90.0..=90.0).contains(&d);
                if !(ok(p.dec_deg[0]) && ok(p.dec_deg[1])) {
                    return Err("batch.random_pointing.dec_deg: within ±90".into());
                }
            }
            Some(b)
        }
        None => None,
    };
    Ok(Config {
        seed,
        sequence,
        batch,
        profile,
        scene,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn patches_replace_one_of_settings() {
        let mut t = serde_json::json!({ "camera": { "sky": { "mag_per_arcsec2": 22.5 }, "width": 10 },
                                        "pedestal": { "normal": [1, 2] } });
        merge_patch(
            &mut t,
            &serde_json::json!({ "camera": { "sky": { "e_per_s": 20 } },
                                                 "pedestal": { "uniform": [0, 1] } }),
        );
        assert_eq!(t["camera"]["sky"], serde_json::json!({ "e_per_s": 20 }));
        assert_eq!(t["camera"]["width"], 10);
        assert_eq!(t["pedestal"], serde_json::json!({ "uniform": [0, 1] }));
    }

    /// Even on the sphere: the share above 0° is sin 60° / (sin 60° + sin 30°).
    #[test]
    fn random_pointings_cover_the_band_evenly() {
        let rp = RandomPointing {
            dec_deg: [-30.0, 60.0],
        };
        let n = 20_000;
        let mut north = 0;
        for k in 0..n {
            let (ra, dec, roll) = rp.draw(k);
            assert!((0.0..360.0).contains(&ra) && (0.0..360.0).contains(&roll));
            assert!((-30.0..=60.0).contains(&dec));
            north += usize::from(dec > 0.0);
        }
        let want = 0.866_025 / (0.866_025 + 0.5);
        assert!((north as f64 / n as f64 - want).abs() < 0.01);
        assert_eq!(rp.draw(5), rp.draw(5));
    }

    #[test]
    fn merge_patch_rfc7386() {
        let mut t = json!({"a": 1, "b": {"c": 2, "d": 3}});
        merge_patch(&mut t, &json!({"b": {"c": null, "e": 4}, "f": 5}));
        assert_eq!(t, json!({"a": 1, "b": {"d": 3, "e": 4}, "f": 5}));
    }

    #[test]
    fn draws_are_reproducible_and_recorded() {
        let p: Profile = serde_json::from_value(json!({
            "processing": {"pedestal_adu": {"normal": [24, 8]}},
            "noise": {"patterns": [{"period_px": 4.3, "angle_deg": 43, "amplitude_adu": {"uniform": [4, 16]}}]}
        }))
        .unwrap();
        p.validate().unwrap();
        let a = p.draw(&mut Profile::settings_rng(7)).unwrap();
        let b = p.draw(&mut Profile::settings_rng(7)).unwrap();
        assert_eq!(a, b);
        assert!((4.0..=16.0).contains(&a.patterns[0].amplitude_adu));
        assert_ne!(a, p.draw(&mut Profile::settings_rng(8)).unwrap());
    }

    #[test]
    fn unknown_profile_keys_are_errors() {
        assert!(serde_json::from_value::<Profile>(json!({"nosie": {}})).is_err());
    }
}

#[cfg(test)]
mod fixed_field_tests {
    use super::FixedField;

    #[test]
    fn interpolates_between_nodes_and_holds_beyond() {
        let f = FixedField {
            origin_px: [10.0, 0.0],
            spacing_px: 10.0,
            values: vec![vec![0.0, 10.0], vec![20.0, 30.0]],
        };
        let m = f.map(40, 20);
        let at = |x: usize, y: usize| m[y * 40 + x];
        assert!((at(15, 0) - 5.0).abs() < 1e-12);
        assert!((at(15, 5) - 15.0).abs() < 1e-12);
        assert!((at(0, 0) - 0.0).abs() < 1e-12, "held left of the grid");
        assert!(
            (at(39, 19) - 30.0).abs() < 1e-12,
            "held beyond the last node"
        );
        assert!(f.validate().is_ok());
        assert!(FixedField {
            origin_px: [0.0, 0.0],
            spacing_px: 1.0,
            values: vec![vec![1.0], vec![]]
        }
        .validate()
        .is_err());
    }
}

#[cfg(test)]
mod catalog_requirement_tests {
    use super::config_from_value;

    fn cfg(scene_cat: &serde_json::Value) -> Result<super::Config, String> {
        config_from_value(
            &serde_json::json!({
                "instrument": {
                    "schema_version": 1,
                    "sensor": { "name": "Test camera", "band_nm": [650.0, 1100.0] },
                    "catalog": { "name": "Gaia test extract", "kind": "csv", "mag_limit": 10.5,
                                 "epoch_jyear": 2016.0, "columns": { "mag": "phot_g_mean_mag", "color": "bp_rp" } }
                },
                "scene": { "pointing": { "ra_deg": 10.0, "dec_deg": 20.0 }, "catalog": scene_cat }
            }),
            "test",
        )
    }

    /// The profile fills what the scene leaves out; what the scene gives wins.
    #[test]
    fn a_path_is_enough_and_the_profile_fills_the_rest() {
        let c = cfg(&serde_json::json!({ "path": "stars.csv" })).unwrap();
        let cat = c.scene.catalog.unwrap();
        assert_eq!(cat.mag_limit, 10.5);
        let csv = cat.csv.unwrap();
        assert_eq!(csv.path.to_str(), Some("stars.csv"));
        assert_eq!(csv.columns.color.as_deref(), Some("bp_rp"));
        assert_eq!(c.profile.sensor.unwrap().name, "Test camera");
        let c = cfg(&serde_json::json!({ "path": "stars.csv", "mag_limit": 8.0 })).unwrap();
        assert_eq!(c.scene.catalog.unwrap().mag_limit, 8.0);
    }

    #[test]
    fn the_wrong_kind_of_catalogue_is_refused_unless_asked() {
        let e = cfg(&serde_json::json!({ "ucac4": "data/ucac4", "mag_limit": 12.0 })).unwrap_err();
        assert!(
            e.contains("Gaia test extract") && e.contains("ignore_profile"),
            "{e}"
        );
        let c = cfg(
            &serde_json::json!({ "ucac4": "data/ucac4", "mag_limit": 12.0, "ignore_profile": true }),
        )
        .unwrap();
        assert!(c.scene.catalog.unwrap().ucac4.is_some());
    }
}
