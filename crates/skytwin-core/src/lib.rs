//! Core of the `skytwin` simulator: render synthetic space-camera frames of
//! stars and satellite streaks with an exact record of what was drawn.
//!
//! A [`Scene`] lists the sources (point stars, streaks, orbiting targets),
//! optionally a pointing and a catalogue, and a [`Camera`] or instrument
//! [`Profile`] describes the detector. [`render()`] and [`render_profile`]
//! turn them into a [`Frame`]: an image in ADU plus a [`Truth`] record of
//! exactly what was drawn, so detectors and plate solvers can be scored
//! against known answers. [`Generator`] renders independent frames of a
//! batch for training sets.
//!
//! The `skytwin` command-line tool (crate `skytwin-cli`) and the Python
//! bindings (crate `skytwin-py`) are thin layers over this crate.
//!
//! Rendering is deterministic: the same scene and seed give the same pixels
//! on every machine, because the crate uses its own RNG and `libm` maths.
//!
//! # Example
//!
//! ```
//! use skytwin_core::{render, Camera, Scene, Star};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let camera = Camera {
//!         width: 64,
//!         height: 48,
//!         ..Camera::default()
//!     };
//!     let scene = Scene {
//!         stars: vec![Star {
//!             x: Some(31.5),
//!             y: Some(20.0),
//!             flux_e: Some(50_000.0),
//!             ..Star::default()
//!         }],
//!         ..Scene::default()
//!     };
//!     let frame = render(&scene, &camera, 42)?;
//!     assert_eq!((frame.width, frame.height), (64, 48));
//!     assert_eq!(frame.adu.len(), 64 * 48);
//!     assert_eq!(frame.truth.stars.len(), 1);
//!     Ok(())
//! }
//! ```

pub mod atmosphere;
pub mod attitude;
pub mod batch;
pub mod camera;
pub mod catalog;
pub mod detector;
pub mod ephem;
pub mod error;
pub mod fits;
pub mod flat;
pub mod frames;
pub mod geometry;
pub mod header;
pub mod label;
pub mod motion;
pub mod orbit;
pub mod param;
pub mod path;
pub mod photometry;
pub mod profile;
pub mod psf;
pub mod render;
pub mod rng;
pub mod scene;
pub mod stray;
pub mod target;
pub mod time;
pub mod vec3;
pub mod wcs;

pub use attitude::PointingSpec;
pub use batch::Generator;
pub use camera::{Camera, Sky};
pub use catalog::{add_catalog_stars, CatalogConfig, CatalogKind, CatalogRequirement};
pub use error::SimError;
pub use geometry::{Geometry, TimeSpec};
pub use orbit::{Orbit, Site, State};
pub use param::{Dist, Param};
pub use profile::{config_from_value, load_config, Config, Profile};
pub use render::{add_tle_file_targets, render_profile, target_visible};
pub use render::{render, Frame, TrueStar, TrueStreak, Truth};
pub use scene::{Scene, Star, Streak};
pub use target::{Brightness, Target};
pub use time::Utc;
pub use wcs::{Pointing, TanWcs};
