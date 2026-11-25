//! What is in the field: stars and satellite streaks.

use serde::{Deserialize, Serialize};

use crate::attitude::PointingSpec;
use crate::catalog::CatalogConfig;
use crate::geometry::TimeSpec;
use crate::orbit::Orbit;
use crate::rng::Rng;
use crate::target::{Brightness, Target};

/// A point source. Give its position as pixels (`x`, `y`) or as sky
/// coordinates (`ra_deg`, `dec_deg`, needs a pointing), and its brightness
/// as `flux_e` or `mag`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Star {
    /// Catalogue identifier, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    /// Column, 0-based, pixel centres on whole numbers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    /// Row, 0-based.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
    /// Right ascension, degrees (ICRS).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ra_deg: Option<f64>,
    /// Declination, degrees (ICRS).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dec_deg: Option<f64>,
    /// Total signal over the exposure, electrons.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flux_e: Option<f64>,
    /// Magnitude in the camera's zeropoint band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mag: Option<f64>,
    /// Colour index for `camera.color_term` (catalogue stars get theirs from
    /// the catalogue).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_index: Option<f64>,
    /// Effective temperature, K, for `camera.passband` (else from the
    /// colour index).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teff_k: Option<f64>,
    /// Parallax, milliarcseconds (sky positions only; needs a time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallax_mas: Option<f64>,
}

/// A satellite that moved in a straight line from `start` to `end` (pixels)
/// during the exposure. Brightness as `flux_e` or `mag`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Streak {
    /// Position at shutter open, `[x, y]` pixels.
    pub start: [f64; 2],
    /// Position at shutter close, `[x, y]` pixels.
    pub end: [f64; 2],
    /// Total signal over the exposure, electrons.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flux_e: Option<f64>,
    /// Magnitude (of the satellite, not per pixel).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mag: Option<f64>,
}

/// Randomly placed straight streaks (satellites, debris), for training
/// sets: each drawn per frame, start anywhere on the image, direction
/// uniform.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomStreaks {
    /// Number of streaks (drawn per frame, rounded).
    pub count: crate::param::Param,
    /// Length, pixels (drawn per streak).
    pub length_px: crate::param::Param,
    /// Magnitude (drawn per streak).
    pub mag: crate::param::Param,
}

impl RandomStreaks {
    /// Draws this frame's streaks for a `width` × `height` image.
    ///
    /// Each streak draws, in order, its start `x` and `y`, length,
    /// direction and magnitude from `rng`; keep that order, since it fixes
    /// the frames a seed gives.
    pub fn draw(&self, width: usize, height: usize, rng: &mut Rng) -> Vec<Streak> {
        let ctx = &mut crate::param::DrawCtx::default();
        let n = libm::round(self.count.draw(rng, ctx)).max(0.0) as usize;
        (0..n)
            .map(|_| {
                let x = rng.range(-0.5, width as f64 - 0.5);
                let y = rng.range(-0.5, height as f64 - 0.5);
                let len = self.length_px.draw(rng, ctx).max(0.0);
                let (s, c) = libm::sincos(rng.range(0.0, 2.0 * core::f64::consts::PI));
                Streak {
                    start: [x, y],
                    end: [x + len * c, y + len * s],
                    flux_e: None,
                    mag: Some(self.mag.draw(rng, ctx)),
                }
            })
            .collect()
    }
}

/// Randomly placed stars, for tests without a catalogue.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomStars {
    /// Number of stars.
    pub count: usize,
    /// Brightest star flux, electrons.
    pub max_flux_e: f64,
    /// Faintest star flux, electrons.
    pub min_flux_e: f64,
}

/// Everything drawn into one frame.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    /// When the exposure starts. Sets the catalogue epoch, the header times
    /// and, with the observer, the aberration.
    #[serde(default)]
    pub time: Option<TimeSpec>,
    /// Where the observer is (default: the geocentre). Needs `time`.
    #[serde(default)]
    pub observer: Option<Orbit>,
    /// Where the camera points: the apparent boresight. Needed for sky
    /// positions and catalogues; adds a WCS to the output.
    #[serde(default)]
    pub pointing: Option<PointingSpec>,
    /// Stars from a catalogue. Resolve with
    /// [`crate::catalog::add_catalog_stars`] before rendering.
    #[serde(default)]
    pub catalog: Option<CatalogConfig>,
    /// Explicit stars.
    #[serde(default)]
    pub stars: Vec<Star>,
    /// Extra random stars, added after the explicit ones.
    #[serde(default)]
    pub random_stars: Option<RandomStars>,
    /// Satellite streaks given in pixels.
    #[serde(default)]
    pub streaks: Vec<Streak>,
    /// Extra random streaks, added after the explicit ones.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random_streaks: Option<RandomStreaks>,
    /// Target satellites with orbits. Need `time` and `pointing`.
    #[serde(default)]
    pub targets: Vec<Target>,
    /// More targets from a TLE file, kept if they come near the field.
    /// Load with [`crate::render::add_tle_file_targets`] before rendering.
    #[serde(default)]
    pub target_file: Option<TleFile>,
    /// Reported minus true pointing, arcsec `[east, north]`: a header that
    /// lags the boresight, e.g. while it settles (added to the profile's
    /// random header error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header_bias_arcsec: Option<[f64; 2]>,
}

/// A two- or three-line element file of targets.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TleFile {
    /// Path to the file.
    pub path: std::path::PathBuf,
    /// Brightness model for every object in it.
    pub brightness: Brightness,
}

impl RandomStars {
    /// Draws the stars from `rng`, uniformly over a `width` × `height` image.
    ///
    /// Fluxes follow a power law with N(>F) ∝ F^-0.6, roughly the slope of
    /// star counts per magnitude at mid galactic latitudes (log N rises
    /// ~0.4 dex per magnitude).
    pub fn draw(&self, width: usize, height: usize, rng: &mut Rng) -> Vec<Star> {
        let alpha = 0.6;
        let (a, b) = (
            libm::pow(self.min_flux_e, -alpha),
            libm::pow(self.max_flux_e, -alpha),
        );
        (0..self.count)
            .map(|_| {
                let u = rng.uniform();
                Star {
                    flux_e: Some(libm::pow(a + u * (b - a), -1.0 / alpha)),
                    x: Some(rng.range(-0.5, width as f64 - 0.5)),
                    y: Some(rng.range(-0.5, height as f64 - 0.5)),
                    ..Star::default()
                }
            })
            .collect()
    }
}
