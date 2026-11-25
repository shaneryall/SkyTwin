//! Many independent frames from one scene file: training sets.
//!
//! Frame `k` uses seed + k and draws everything random anew (stars,
//! streaks, noise, every distribution in the profile), and, with
//! `batch.random_pointing`, its own pointing. With a fixed pointing the
//! catalogue is read once. Frames can be rendered in any order and on any
//! number of threads; each depends only on its index.

use crate::profile::{Config, RandomPointing};
use crate::render::{add_tle_file_targets, render_profile, Frame};
use crate::scene::Scene;

/// Renders the frames of a batch.
#[derive(Clone, Debug)]
pub struct Generator {
    cfg: Config,
    base: Scene,
    random_pointing: Option<RandomPointing>,
    seed: u64,
    frames: usize,
}

impl Generator {
    /// Prepares `cfg` for rendering, seeded with `seed` if given, else the
    /// file's seed. The frame count is the batch block's, else 1.
    ///
    /// With a fixed pointing the catalogue is read here, once for every
    /// frame.
    ///
    /// # Errors
    ///
    /// Fails if `cfg` has a sequence (a sequence is not a batch), if
    /// `batch.random_pointing` is combined with a catalogue but no
    /// `scene.time`, or if the catalogue cannot be read.
    pub fn new(cfg: Config, seed: Option<u64>) -> Result<Self, String> {
        if cfg.sequence.is_some() {
            return Err("a sequence is not a batch: give one or the other".into());
        }
        let random_pointing = cfg.batch.as_ref().and_then(|b| b.random_pointing.clone());
        let mut base = cfg.scene.clone();
        if random_pointing.is_none() {
            crate::catalog::add_catalog_stars(&mut base, &cfg.profile.camera)
                .map_err(|e| e.to_string())?;
        } else if base.catalog.is_some() && base.time.is_none() {
            return Err("batch.random_pointing with a catalogue needs scene.time".into());
        }
        Ok(Generator {
            seed: seed.unwrap_or(cfg.seed),
            frames: cfg.batch.as_ref().map_or(1, |b| b.frames),
            random_pointing,
            base,
            cfg,
        })
    }

    /// The configuration the generator was built from.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Frames in the batch (1 without a batch block).
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Frame `k`'s seed: the base seed plus `k`, wrapping.
    pub fn seed_of(&self, k: u64) -> u64 {
        self.seed.wrapping_add(k)
    }

    /// Frame `k`'s random pointing `(ra, dec, roll)`, degrees, if the
    /// batch asks for one.
    pub fn pointing(&self, k: u64) -> Option<(f64, f64, f64)> {
        self.random_pointing
            .as_ref()
            .map(|p| p.draw(self.seed_of(k)))
    }

    /// Frame `k`'s scene, with its pointing, catalogue stars and TLE
    /// targets in place.
    ///
    /// # Errors
    ///
    /// Fails if the catalogue or the TLE target file cannot be loaded.
    pub fn scene(&self, k: u64) -> Result<Scene, String> {
        let mut scene = self.base.clone();
        if let Some((ra, dec, roll)) = self.pointing(k) {
            let mut ps = scene.pointing.clone().unwrap_or_default();
            ps.ra_deg = Some(ra);
            ps.dec_deg = Some(dec);
            ps.roll_deg = Some(roll);
            ps.quaternion = None;
            scene.pointing = Some(ps);
        }
        let cam = &self.cfg.profile.camera;
        crate::catalog::add_catalog_stars(&mut scene, cam).map_err(|e| e.to_string())?;
        add_tle_file_targets(&mut scene, cam).map_err(|e| e.to_string())?;
        Ok(scene)
    }

    /// Renders frame `k`.
    ///
    /// # Errors
    ///
    /// Fails as [`Generator::scene`] does, or if the scene cannot be
    /// rendered (see [`crate::SimError`]).
    pub fn render(&self, k: u64) -> Result<Frame, String> {
        let scene = self.scene(k)?;
        render_profile(&scene, &self.cfg.profile, self.seed_of(k)).map_err(|e| e.to_string())
    }
}
