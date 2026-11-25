//! Detector defects: dark current against temperature, hot pixels,
//! radiation hits and charge-transfer inefficiency.
//!
//! Sources:
//! - Dark current against temperature: Janesick, *Scientific Charge-Coupled
//!   Devices* (SPIE, 2001), section 7.2: `D ∝ T^1.5 exp(−E_g / 2kT)`, with
//!   the silicon band gap `E_g(T) = 1.1557 − 7.021e-4 T² / (1108 + T)` eV.
//! - Charge from a minimum-ionising particle in silicon: about 80 e⁻/µm
//!   (most probable), Particle Data Group, *Review of Particle Physics*,
//!   "Passage of particles through matter" (2024); straggling shaped by the
//!   Moyal approximation to the Landau distribution, Moyal, Phil. Mag. 46,
//!   263 (1955).
//! - Isotropic flux through a plane: the number crossing at zenith angle θ
//!   goes as cos θ sin θ, so cos²θ is uniform.
//! - Charge-transfer inefficiency as capture and exponential release:
//!   Janesick (2001), section 5.3; trap release model after Massey et al.,
//!   MNRAS 401, 371 (2010), simplified to one trap species.
//! - Flickering pixels: see [`RtsPixels`].

use serde::{Deserialize, Serialize};

use crate::param::{pick_index, DrawCtx, Param};
use crate::psf::add_streak;
use crate::rng::Rng;

/// Boltzmann constant, eV/K.
const K_EV: f64 = 8.617_333_262e-5;

/// Dark current.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dark {
    /// Dark current at the reference temperature, e⁻/pixel/s.
    pub e_per_s_at_ref: Param,
    /// Reference temperature, °C.
    pub ref_temp_c: f64,
    /// If set, dark current doubles every this many °C instead of following
    /// the silicon band-gap law.
    #[serde(default)]
    pub doubling_c: Option<f64>,
}

/// Ratio of dark current at `t_c` to that at `ref_c`, both °C: doubling
/// every `doubling_c` °C when given, else the band-gap law in the module
/// docs.
pub fn dark_factor(t_c: f64, ref_c: f64, doubling_c: Option<f64>) -> f64 {
    if let Some(d) = doubling_c {
        return libm::pow(2.0, (t_c - ref_c) / d);
    }
    let law = |c: f64| {
        let t = c + 273.15;
        let eg = 1.1557 - 7.021e-4 * t * t / (1108.0 + t);
        libm::pow(t, 1.5) * libm::exp(-eg / (2.0 * K_EV * t))
    };
    law(t_c) / law(ref_c)
}

/// Hot pixels: a map fixed to the detector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HotPixels {
    /// Hot pixels per million pixels.
    pub density_per_mpix: f64,
    /// Each hot pixel's dark rate at the dark block's reference temperature
    /// (or at the frame temperature without one), e⁻/s. Drawn once per pixel
    /// from `detector_seed`.
    pub e_per_s_at_ref: Param,
}

/// A hot pixel at the frame's temperature.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HotPixel {
    /// Column.
    pub x: usize,
    /// Row.
    pub y: usize,
    /// Rate this frame, e⁻/s.
    pub e_per_s: f64,
}

/// Place the detector's hot pixels and draw their reference rates.
///
/// Returns `(x, y, e⁻/s at reference)` per image pixel of a `w × h` image;
/// hot chip pixels that share a binned pixel are summed into one entry.
pub fn hot_pixel_map(
    hp: &HotPixels,
    layout: &Layout,
    w: usize,
    h: usize,
    rng: &mut Rng,
) -> Vec<(usize, usize, f64)> {
    let [cw, ch] = layout.chip_px;
    let n = layout.count_on_chip(hp.density_per_mpix);
    let ctx = &mut DrawCtx::default();
    let mut out: Vec<(usize, usize, f64)> = Vec::new();
    for _ in 0..n {
        let xc = pick_index(rng, cw);
        let yc = pick_index(rng, ch);
        let rate = hp.e_per_s_at_ref.draw(rng, ctx).max(0.0);
        if let Some((x, y)) = layout.chip_to_image(xc, yc, w, h) {
            match out.iter_mut().find(|p| p.0 == x && p.1 == y) {
                Some(p) => p.2 += rate,
                None => out.push((x, y, rate)),
            }
        }
    }
    out
}

/// Where the delivered image sits on the chip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    /// Chip size, unbinned pixels `[width, height]`.
    pub chip_px: [usize; 2],
    /// Chip pixel (0-based, unbinned) of the image's first pixel.
    #[serde(default)]
    pub window_origin_px: [usize; 2],
    /// Binning `[x, y]` (default `[1, 1]`).
    #[serde(default = "one_one")]
    pub binning: [usize; 2],
}

/// Serde default for [`Layout::binning`].
fn one_one() -> [usize; 2] {
    [1, 1]
}

impl Layout {
    /// The image is the whole chip, unbinned.
    pub fn identity(w: usize, h: usize) -> Self {
        Layout {
            chip_px: [w, h],
            window_origin_px: [0, 0],
            binning: [1, 1],
        }
    }

    /// Check that a binned `w × h` image fits on the chip at the window
    /// origin.
    ///
    /// # Errors
    ///
    /// When a binning factor is 0 or the image does not fit.
    pub fn validate(&self, w: usize, h: usize) -> Result<(), String> {
        let [bx, by] = self.binning;
        if bx == 0 || by == 0 {
            return Err("detector.binning must be at least 1".into());
        }
        if self.window_origin_px[0] + bx * w > self.chip_px[0]
            || self.window_origin_px[1] + by * h > self.chip_px[1]
        {
            return Err(
                "detector: the binned image does not fit on the chip at that window origin".into(),
            );
        }
        Ok(())
    }

    /// Image pixel holding a chip pixel, if the window covers it.
    pub fn chip_to_image(
        &self,
        xc: usize,
        yc: usize,
        w: usize,
        h: usize,
    ) -> Option<(usize, usize)> {
        let [x0, y0] = self.window_origin_px;
        let [bx, by] = self.binning;
        if xc < x0 || yc < y0 {
            return None;
        }
        let (x, y) = ((xc - x0) / bx, (yc - y0) / by);
        (x < w && y < h).then_some((x, y))
    }

    /// First chip column of image column `x`, and first chip row of row `y`.
    pub fn chip_origin_of(&self, x: usize, y: usize) -> (usize, usize) {
        (
            self.window_origin_px[0] + self.binning[0] * x,
            self.window_origin_px[1] + self.binning[1] * y,
        )
    }

    /// How many features at `density_per_mpix` (per million chip pixels)
    /// fall on the whole chip, rounded.
    pub(crate) fn count_on_chip(&self, density_per_mpix: f64) -> usize {
        let [cw, ch] = self.chip_px;
        libm::round(density_per_mpix * (cw * ch) as f64 / 1e6) as usize
    }
}

/// Flickering pixels: dark current that switches at random between a low
/// and a high level (random telegraph signal).
///
/// Source: Hopkinson, Goiffon & Mohammadzadeh, "Random telegraph signals in
/// proton irradiated CCDs and APS", IEEE Trans. Nucl. Sci. 55, 2197 (2008):
/// two-level switching with exponentially distributed dwell times whose
/// means follow an Arrhenius law in temperature.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RtsPixels {
    /// Flickering pixels per million chip pixels.
    pub density_per_mpix: f64,
    /// Extra dark rate in the high state at `ref_temp_c`, e⁻/s (drawn once
    /// per pixel from `detector_seed`; scaled with temperature like dark
    /// current).
    pub high_e_per_s_at_ref: Param,
    /// Mean time in the high state at `ref_temp_c`, s.
    pub mean_high_s: Param,
    /// Mean time in the low state at `ref_temp_c`, s.
    pub mean_low_s: Param,
    /// Reference temperature, °C.
    pub ref_temp_c: f64,
    /// Activation energy of leaving the high state, eV (0: no temperature
    /// dependence).
    #[serde(default)]
    pub activation_high_ev: f64,
    /// Activation energy of leaving the low state, eV.
    #[serde(default)]
    pub activation_low_ev: f64,
}

/// One flickering pixel: chip position, high-state rate at reference, mean
/// dwell times at reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtsPixel {
    /// Chip column.
    pub xc: usize,
    /// Chip row.
    pub yc: usize,
    /// High-state rate at the reference temperature, e⁻/s.
    pub high: f64,
    /// Mean high dwell at reference, s.
    pub t_high: f64,
    /// Mean low dwell at reference, s.
    pub t_low: f64,
}

/// Place the detector's flickering pixels (fixed by `detector_seed`); mean
/// dwell times are floored at 1 µs.
pub fn rts_map(r: &RtsPixels, layout: &Layout, rng: &mut Rng) -> Vec<RtsPixel> {
    let [cw, ch] = layout.chip_px;
    let n = layout.count_on_chip(r.density_per_mpix);
    let ctx = &mut DrawCtx::default();
    (0..n)
        .map(|_| RtsPixel {
            xc: pick_index(rng, cw),
            yc: pick_index(rng, ch),
            high: r.high_e_per_s_at_ref.draw(rng, ctx).max(0.0),
            t_high: r.mean_high_s.draw(rng, ctx).max(1e-6),
            t_low: r.mean_low_s.draw(rng, ctx).max(1e-6),
        })
        .collect()
}

/// Arrhenius factor on a time constant, `τ(T) / τ_ref = exp(E/k (1/T − 1/T_ref))`,
/// for temperatures `t_c` and `ref_c` in °C and activation energy `ev` in eV.
pub fn arrhenius(t_c: f64, ref_c: f64, ev: f64) -> f64 {
    if ev == 0.0 {
        return 1.0;
    }
    libm::exp(ev / K_EV * (1.0 / (t_c + 273.15) - 1.0 / (ref_c + 273.15)))
}

/// Fraction of `exposure` spent in the high state, from a two-state
/// process started in equilibrium, with exponential dwell times of means
/// `t_high` and `t_low` (all in seconds).
pub fn high_fraction(t_high: f64, t_low: f64, exposure: f64, rng: &mut Rng) -> f64 {
    let mut high = rng.uniform() < t_high / (t_high + t_low);
    let (mut t, mut in_high) = (0.0, 0.0);
    while t < exposure {
        let dwell = -libm::log(1.0 - rng.uniform()) * if high { t_high } else { t_low };
        let step = dwell.min(exposure - t);
        if high {
            in_high += step;
        }
        t += step;
        high = !high;
    }
    in_high / exposure.max(1e-12)
}

/// Rate table by sub-satellite latitude and longitude.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateMap {
    /// Cell edges in latitude, increasing, degrees.
    pub lat_edges_deg: Vec<f64>,
    /// Cell edges in longitude, increasing, degrees.
    pub lon_edges_deg: Vec<f64>,
    /// Rate per cell, hits/cm²/s: one row per latitude cell, one value per
    /// longitude cell.
    pub rate_per_cm2_s: Vec<Vec<f64>>,
}

impl RateMap {
    /// Check the shape.
    ///
    /// # Errors
    ///
    /// When an edge list has fewer than two entries or does not strictly
    /// increase, or the table is not (lat cells) × (lon cells).
    pub fn validate(&self) -> Result<(), String> {
        let (nl, nn) = (self.lat_edges_deg.len(), self.lon_edges_deg.len());
        let incr = |v: &[f64]| v.windows(2).all(|w| w[1] > w[0]);
        if nl < 2 || nn < 2 || !incr(&self.lat_edges_deg) || !incr(&self.lon_edges_deg) {
            return Err("radiation.rate_map: edges must increase, at least two each".into());
        }
        if self.rate_per_cm2_s.len() != nl - 1
            || self.rate_per_cm2_s.iter().any(|r| r.len() != nn - 1)
        {
            return Err("radiation.rate_map: table must be (lat cells) × (lon cells)".into());
        }
        Ok(())
    }

    /// Rate at a position (clamped to the table's edges; longitude wrapped
    /// into the table's range when it spans 360°).
    pub fn rate(&self, lat: f64, lon: f64) -> f64 {
        let cell = |edges: &[f64], v: f64| {
            let i = edges.partition_point(|&e| e <= v);
            i.clamp(1, edges.len() - 1) - 1
        };
        let lo = self.lon_edges_deg[0];
        let lon = if self.lon_edges_deg[self.lon_edges_deg.len() - 1] - lo >= 359.999 {
            lo + (lon - lo).rem_euclid(360.0)
        } else {
            lon
        };
        self.rate_per_cm2_s[cell(&self.lat_edges_deg, lat)][cell(&self.lon_edges_deg, lon)]
    }
}

/// Serde default for [`Radiation::depth_um`].
fn default_depth() -> f64 {
    15.0
}

/// Serde default for [`Radiation::e_per_um`].
fn default_e_per_um() -> f64 {
    80.0
}

/// Serde default for [`Radiation::landau_scale`].
fn default_landau() -> f64 {
    0.15
}

/// Serde default for [`Radiation::diffusion_px`].
fn default_diffusion() -> f64 {
    0.4
}

/// Radiation hits (cosmic rays, trapped protons).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Radiation {
    /// Hit rate, hits/cm²/s, used without a position or a map.
    pub rate_per_cm2_s: Param,
    /// Rate by sub-satellite point (needs `scene.time` and an orbiting
    /// observer).
    #[serde(default)]
    pub rate_map: Option<RateMap>,
    /// Factor applied to the map's rates.
    #[serde(default)]
    pub map_scale: Option<Param>,
    /// Time hits are collected, s (default: the exposure; add the read time
    /// for a frame-transfer or shutterless camera).
    #[serde(default)]
    pub collect_time_s: Option<Param>,
    /// Read-out time, s: each row keeps collecting hits until it is read,
    /// so the last row read gets this much more time than the first. Hit
    /// density then rises linearly across the frame.
    #[serde(default)]
    pub readout_s: f64,
    /// Which row is read first: `bottom` (default, row 0) or `top`.
    #[serde(default)]
    pub first_row: FirstRow,
    /// Depth of the sensitive silicon, µm.
    #[serde(default = "default_depth")]
    pub depth_um: f64,
    /// Most probable charge per µm of track, e⁻.
    #[serde(default = "default_e_per_um")]
    pub e_per_um: f64,
    /// Relative width of the charge straggling.
    #[serde(default = "default_landau")]
    pub landau_scale: f64,
    /// Charge spread by diffusion, pixels (Gaussian σ).
    #[serde(default = "default_diffusion")]
    pub diffusion_px: f64,
}

/// The row read first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FirstRow {
    /// Row 0.
    #[default]
    Bottom,
    /// The last row.
    Top,
}

/// One hit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    /// Track start `[x, y]`, pixels.
    pub start: [f64; 2],
    /// Track end `[x, y]`, pixels.
    pub end: [f64; 2],
    /// Charge deposited, e⁻.
    pub charge_e: f64,
}

/// Draw the hits for a frame: `rate` hits/cm²/s over `time_s` on a
/// `w × h` image of `binning` chip pixels of `pitch_um`.
///
/// With a read-out time, row `u` (0 read first, 1 last) collects for
/// `time_s + readout_s · u`, so `u` is drawn by inverting the CDF
/// `(t u + ro u²/2) / (t + ro/2)`. Track directions have `cos²θ` uniform,
/// with `cos θ` floored at 0.05 so tracks are at most 20 depths long.
/// Charge follows the Moyal approximation, `−ln(z²)` for a standard normal
/// `z` (mode 0), scaled by `landau_scale`. Track lengths are computed on
/// the chip, then converted to binned image pixels per axis.
#[expect(
    clippy::too_many_arguments,
    reason = "public API: the frame geometry is passed as separate values"
)]
pub fn draw_hits(
    r: &Radiation,
    rate: f64,
    time_s: f64,
    w: usize,
    h: usize,
    pitch_um: f64,
    binning: [usize; 2],
    rng: &mut Rng,
) -> Vec<Hit> {
    let (bx, by) = (binning[0] as f64, binning[1] as f64);
    let area_cm2 = (w as f64 * bx * pitch_um * 1e-4) * (h as f64 * by * pitch_um * 1e-4);
    let (t, ro) = (time_s.max(0.0), r.readout_s.max(0.0));
    let n = rng.poisson(rate * area_cm2 * (t + 0.5 * ro)) as usize;
    (0..n)
        .map(|_| {
            let x = rng.range(-0.5, w as f64 - 0.5);
            let f = rng.uniform();
            let u = if ro > 1e-12 * t.max(1e-300) {
                (-t + libm::sqrt(t * t + 2.0 * ro * f * (t + 0.5 * ro))) / ro
            } else {
                f
            };
            let u = match r.first_row {
                FirstRow::Bottom => u,
                FirstRow::Top => 1.0 - u,
            };
            let y = -0.5 + u * h as f64;
            let c = libm::sqrt(rng.uniform()).max(0.05);
            let tan = libm::sqrt(1.0 - c * c) / c;
            let len_chip = r.depth_um * tan / pitch_um;
            let az = rng.range(0.0, 2.0 * core::f64::consts::PI);
            let (s, co) = libm::sincos(az);
            let path_um = r.depth_um / c;
            let z = rng.normal();
            let moyal = -libm::log((z * z).max(1e-300));
            let charge = r.e_per_um * path_um * (1.0 + r.landau_scale * moyal).max(0.2);
            let (lx, ly) = (len_chip * co / bx, len_chip * s / by);
            Hit {
                start: [x - 0.5 * lx, y - 0.5 * ly],
                end: [x + 0.5 * lx, y + 0.5 * ly],
                charge_e: charge,
            }
        })
        .collect()
}

/// Add hits to an electron image as streaks of Gaussian width
/// `diffusion_px` (at least 0.05 px).
pub fn draw_hit_charge(img: &mut [f64], w: usize, h: usize, hits: &[Hit], diffusion_px: f64) {
    for k in hits {
        add_streak(
            img,
            w,
            h,
            (k.start[0], k.start[1]),
            (k.end[0], k.end[1]),
            k.charge_e,
            diffusion_px.max(0.05),
        );
    }
}

/// One direction of charge transfer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CtiAxis {
    /// Fraction of a pixel's charge captured per transfer.
    pub loss_per_transfer: Param,
    /// Mean release distance, pixels (the trail length).
    pub trail_px: Param,
}

/// Charge-transfer inefficiency.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cti {
    /// Along rows, towards the output amplifier.
    #[serde(default)]
    pub serial: Option<CtiAxis>,
    /// Along columns, towards the serial register.
    #[serde(default)]
    pub parallel: Option<CtiAxis>,
    /// Output amplifier at the right end of the rows (default) or the left.
    #[serde(default = "default_true")]
    pub amp_right: bool,
    /// Serial register at the bottom row (default) or the top.
    #[serde(default = "default_true")]
    pub register_bottom: bool,
}

/// Serde default for [`Cti`]'s flags.
fn default_true() -> bool {
    true
}

/// Fraction captured at each position of a line in read order,
/// `1 − (1 − loss)^transfers(k)`, where `transfers(k)` is how many transfers
/// the k-th pixel read made. The same for every line, so computed once.
fn cti_fracs(n: usize, loss: f64, transfers: impl Fn(usize) -> f64) -> Vec<f64> {
    (0..n)
        .map(|k| 1.0 - libm::pow(1.0 - loss, transfers(k)))
        .collect()
}

/// Capture and exponential release along one line in read order: each
/// pixel loses `fracs[k]` of its charge to the traps, which release
/// `1 − e^(−1/trail)` of what they hold into every later pixel. Charge still
/// trapped at the end of the line is lost.
fn cti_line(line: &mut [f64], trail: f64, fracs: &[f64]) {
    let keep = libm::exp(-1.0 / trail.max(1e-3));
    let mut trapped = 0.0;
    for (v, &frac) in line.iter_mut().zip(fracs) {
        let captured = *v * frac;
        let released = trapped * (1.0 - keep);
        trapped += captured - released;
        *v += released - captured;
    }
}

/// Apply CTI to an electron image (rows of `w`, row 0 at the bottom).
///
/// `serial` and `parallel` are this frame's `(loss per transfer, trail px)`.
/// Transfers are counted on the chip, from the image's place on it.
/// Parallel CTI runs first, reading each column from the row nearest the
/// serial register.
pub fn apply_cti(
    img: &mut [f64],
    w: usize,
    h: usize,
    cti: &Cti,
    serial: (f64, f64),
    parallel: (f64, f64),
    layout: &Layout,
) {
    let [cw, ch] = layout.chip_px;
    if cti.parallel.is_some() && parallel.0 > 0.0 {
        let transfers = |k: usize| {
            let y = if cti.register_bottom { k } else { h - 1 - k };
            let yc = layout.chip_origin_of(0, y).1;
            (if cti.register_bottom { yc + 1 } else { ch - yc }) as f64
        };
        let fracs = cti_fracs(h, parallel.0, transfers);
        for x in 0..w {
            let mut col: Vec<f64> = (0..h).map(|y| img[y * w + x]).collect();
            if !cti.register_bottom {
                col.reverse();
            }
            cti_line(&mut col, parallel.1, &fracs);
            if !cti.register_bottom {
                col.reverse();
            }
            for (y, v) in col.into_iter().enumerate() {
                img[y * w + x] = v;
            }
        }
    }
    if cti.serial.is_some() && serial.0 > 0.0 {
        let transfers = |k: usize| {
            let x = if cti.amp_right { w - 1 - k } else { k };
            let xc = layout.chip_origin_of(x, 0).0;
            (if cti.amp_right { cw - xc } else { xc + 1 }) as f64
        };
        let fracs = cti_fracs(w, serial.0, transfers);
        for row in img.chunks_mut(w) {
            if cti.amp_right {
                row.reverse();
            }
            cti_line(row, serial.1, &fracs);
            if cti.amp_right {
                row.reverse();
            }
        }
    }
}

/// What one corrupted band holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BandKind {
    /// Salt and pepper: each pixel at the low clip or anywhere up to the
    /// high one.
    Noise,
    /// Dropped rows: at the low clip.
    Dropped,
    /// Rows at the high clip.
    Saturated,
}

/// Read-out corruption: whole rows lost or garbled after digitisation, as
/// a link or memory fault would leave them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadoutFaults {
    /// Fraction of frames with any corruption.
    pub probability: f64,
    /// Bands in a corrupted frame (default 1).
    #[serde(default = "one_param")]
    pub bands: Param,
    /// Rows per band (default 1).
    #[serde(default = "one_param")]
    pub band_rows: Param,
    /// Relative weights of the kinds `[noise, dropped, saturated]`
    /// (default `[1, 1, 1]`).
    #[serde(default = "even_kinds")]
    pub kind_weights: [f64; 3],
    /// Chance that a corrupted frame is also cut short: every row from a
    /// random row to the last reads at the low clip.
    #[serde(default)]
    pub truncate_probability: f64,
    /// In a `noise` band, the share of pixels read at the low clip
    /// (default 0.5).
    #[serde(default = "half")]
    pub noise_low_fraction: f64,
    /// In a `noise` band, the other pixels: without this, uniform over the
    /// whole output range (salt and pepper); with it, the row's own value
    /// plus Gaussian noise of this RMS, ADU, drawn per band.
    #[serde(default)]
    pub noise_rms_adu: Option<Param>,
}

/// Serde default for [`ReadoutFaults::noise_low_fraction`].
fn half() -> f64 {
    0.5
}

/// Serde default for [`ReadoutFaults`]' band counts.
fn one_param() -> Param {
    Param::Fixed(1.0)
}

/// Serde default for [`ReadoutFaults::kind_weights`].
fn even_kinds() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}

/// One corrupted band, rows `first..=last`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Band {
    /// First row, 0-based.
    pub first: usize,
    /// Last row.
    pub last: usize,
    /// What it holds.
    pub kind: BandKind,
}

impl ReadoutFaults {
    /// Check the settings.
    ///
    /// # Errors
    ///
    /// When a probability or `noise_low_fraction` is outside 0–1, a kind
    /// weight is negative or all are zero, or a [`Param`] is invalid.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.probability)
            || !(0.0..=1.0).contains(&self.truncate_probability)
        {
            return Err("readout_faults: probabilities must be 0 to 1".into());
        }
        if self.kind_weights.iter().any(|w| w.is_nan() || *w < 0.0)
            || self.kind_weights.iter().sum::<f64>() <= 0.0
        {
            return Err("readout_faults.kind_weights: non-negative, not all zero".into());
        }
        if !(0.0..=1.0).contains(&self.noise_low_fraction) {
            return Err("readout_faults.noise_low_fraction must be 0 to 1".into());
        }
        if let Some(p) = &self.noise_rms_adu {
            p.validate("readout_faults.noise_rms_adu")?;
        }
        self.bands.validate("readout_faults.bands")?;
        self.band_rows.validate("readout_faults.band_rows")
    }

    /// Draw this frame's bands for an image `h` rows high: none with
    /// probability `1 − probability`, else `bands` bands of `band_rows`
    /// rows each, with a kind chosen by `kind_weights`, plus possibly a
    /// truncation band from a random row to the last.
    pub fn draw(&self, h: usize, rng: &mut Rng) -> Vec<Band> {
        if h == 0 || rng.uniform() >= self.probability {
            return vec![];
        }
        let ctx = &mut DrawCtx::default();
        let n = libm::round(self.bands.draw(rng, ctx)).max(1.0) as usize;
        let total: f64 = self.kind_weights.iter().sum();
        let mut out = Vec::with_capacity(n + 1);
        for _ in 0..n {
            let rows = (libm::round(self.band_rows.draw(rng, ctx)).max(1.0) as usize).min(h);
            let first = (rng.uniform() * (h - rows + 1) as f64) as usize;
            let mut u = rng.uniform() * total;
            let mut kind = BandKind::Saturated;
            for (k, wgt) in [BandKind::Noise, BandKind::Dropped, BandKind::Saturated]
                .into_iter()
                .zip(self.kind_weights)
            {
                if u < wgt {
                    kind = k;
                    break;
                }
                u -= wgt;
            }
            out.push(Band {
                first: first.min(h - 1),
                last: (first + rows - 1).min(h - 1),
                kind,
            });
        }
        if rng.uniform() < self.truncate_probability {
            let first = (rng.uniform() * h as f64) as usize;
            out.push(Band {
                first: first.min(h - 1),
                last: h - 1,
                kind: BandKind::Dropped,
            });
        }
        out
    }
}

/// Write the bands into a digitised image whose values lie in `lo..=hi`;
/// `integer` rounds noise values for integer output types.
pub fn apply_bands(
    adu: &mut [f64],
    w: usize,
    bands: &[Band],
    (lo, hi): (f64, f64),
    integer: bool,
    faults: &ReadoutFaults,
    rng: &mut Rng,
) {
    let ctx = &mut DrawCtx::default();
    for b in bands {
        let rms = faults
            .noise_rms_adu
            .as_ref()
            .map(|p| p.draw(rng, ctx).max(0.0));
        for y in b.first..=b.last {
            for v in &mut adu[y * w..(y + 1) * w] {
                *v = match b.kind {
                    BandKind::Dropped => lo,
                    BandKind::Saturated => hi,
                    BandKind::Noise => {
                        if rng.uniform() < faults.noise_low_fraction {
                            lo
                        } else {
                            let x = match rms {
                                Some(s) => (*v + s * rng.normal()).clamp(lo, hi),
                                None => rng.range(lo, hi),
                            };
                            if integer {
                                libm::round(x)
                            } else {
                                x
                            }
                        }
                    }
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_doubles_about_every_six_degrees_near_minus_thirty() {
        let f = dark_factor(-24.0, -30.0, None);
        assert!((1.8..2.4).contains(&f), "{f}");
        assert!((dark_factor(-20.0, -30.0, Some(5.0)) - 4.0).abs() < 1e-12);
        assert_eq!(dark_factor(-30.0, -30.0, None), 1.0);
    }

    /// A 1:3 duty cycle; warmer, dwell times shrink.
    #[test]
    fn telegraph_duty_cycle() {
        let mut rng = Rng::new(9);
        let n = 4000;
        let f: f64 = (0..n)
            .map(|_| high_fraction(1.0, 3.0, 10.0, &mut rng))
            .sum::<f64>()
            / f64::from(n);
        assert!((f - 0.25).abs() < 0.01, "{f}");
        assert!(arrhenius(-20.0, -30.0, 0.6) < 0.5);
        assert_eq!(arrhenius(-20.0, -30.0, 0.0), 1.0);
    }

    #[test]
    fn rate_map_lookup() {
        let m = RateMap {
            lat_edges_deg: vec![-90.0, 0.0, 90.0],
            lon_edges_deg: vec![-180.0, 0.0, 180.0],
            rate_per_cm2_s: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };
        m.validate().unwrap();
        assert_eq!(m.rate(-30.0, -50.0), 1.0);
        assert_eq!(m.rate(10.0, 10.0), 4.0);
        assert_eq!(m.rate(10.0, 190.0), 3.0, "wraps to −170");
        assert_eq!(m.rate(95.0, 10.0), 4.0, "clamps");
    }

    /// 1 cm² × 5 /cm²/s × (10 + 20/2) s = 100 hits a frame; the halves in
    /// the ratio (10 + 15) / (10 + 5) = 1.67.
    #[test]
    fn hits_during_readout_rise_across_the_frame() {
        let mut r: Radiation = serde_json::from_value(serde_json::json!({
            "rate_per_cm2_s": 5.0, "readout_s": 20.0
        }))
        .unwrap();
        let mut rng = Rng::new(7);
        let (mut lo, mut hi, mut n) = (0usize, 0usize, 0usize);
        for _ in 0..40 {
            for k in draw_hits(&r, 5.0, 10.0, 1000, 1000, 10.0, [1, 1], &mut rng) {
                let y = 0.5 * (k.start[1] + k.end[1]);
                if y < 499.5 {
                    lo += 1;
                } else {
                    hi += 1;
                }
                n += 1;
            }
        }
        assert!((3700..4300).contains(&n), "{n}");
        let ratio = hi as f64 / lo as f64;
        assert!((ratio - 5.0 / 3.0).abs() < 0.12, "{ratio}");
        r.first_row = FirstRow::Top;
        let (mut lo, mut hi) = (0usize, 0usize);
        for _ in 0..40 {
            for k in draw_hits(&r, 5.0, 10.0, 1000, 1000, 10.0, [1, 1], &mut rng) {
                if 0.5 * (k.start[1] + k.end[1]) < 499.5 {
                    lo += 1;
                } else {
                    hi += 1;
                }
            }
        }
        assert!((lo as f64 / hi as f64 - 5.0 / 3.0).abs() < 0.12);
    }

    /// 1000 × 1000 pixels of 10 µm = 1 cm²; 5 /cm²/s × 20 s = 100 hits.
    /// Most hits are near-vertical: median charge a little above one
    /// depth's worth.
    #[test]
    fn hits_have_the_expected_count_and_charge() {
        let r = Radiation {
            rate_per_cm2_s: 5.0.into(),
            rate_map: None,
            map_scale: None,
            collect_time_s: None,
            readout_s: 0.0,
            first_row: FirstRow::Bottom,
            depth_um: 15.0,
            e_per_um: 80.0,
            landau_scale: 0.15,
            diffusion_px: 0.4,
        };
        let mut rng = Rng::new(4);
        let n: usize = (0..50)
            .map(|_| draw_hits(&r, 5.0, 20.0, 1000, 1000, 10.0, [1, 1], &mut rng).len())
            .sum();
        assert!((4600..5400).contains(&n), "{n}");
        let hits = draw_hits(&r, 5.0, 200.0, 1000, 1000, 10.0, [1, 1], &mut rng);
        let mut q: Vec<f64> = hits
            .iter()
            .map(|h| h.charge_e / (r.e_per_um * r.depth_um))
            .collect();
        q.sort_by(f64::total_cmp);
        let med = q[q.len() / 2];
        assert!((1.0..2.0).contains(&med), "{med}");
    }

    /// Charge still trapped at the end of the row is lost; 15 transfers to
    /// the right edge at 1e-3 capture about 1.5%.
    #[test]
    fn cti_conserves_charge_and_trails_away_from_the_amplifier() {
        let (w, h) = (20, 1);
        let mut img = vec![0.0; w * h];
        img[5] = 1000.0;
        let cti = Cti {
            serial: Some(CtiAxis {
                loss_per_transfer: 1e-3.into(),
                trail_px: 2.0.into(),
            }),
            parallel: None,
            amp_right: true,
            register_bottom: true,
        };
        apply_cti(
            &mut img,
            w,
            h,
            &cti,
            (1e-3, 2.0),
            (0.0, 0.0),
            &Layout::identity(w, h),
        );
        let total: f64 = img.iter().sum();
        assert!(total <= 1000.0 + 1e-9 && total > 990.0, "{total}");
        assert!(img[4] > 0.0 && img[6] == 0.0, "{:?}", &img[3..8]);
        assert!((1000.0 - img[5] - 14.9).abs() < 0.5, "{}", img[5]);
    }
}
