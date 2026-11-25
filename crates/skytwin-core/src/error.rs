//! Errors from preparing or rendering a scene.

use std::fmt;

/// Why a scene could not be rendered.
#[derive(Debug)]
pub enum SimError {
    /// Sky positions or a catalogue were given without a pointing.
    NoPointing(&'static str),
    /// A star or streak is missing a value or has conflicting ones.
    BadSource {
        /// `"star"` or `"streak"`.
        kind: &'static str,
        /// Index in its list.
        index: usize,
        /// What is wrong.
        reason: &'static str,
    },
    /// The scene still names a catalogue that was not loaded.
    CatalogNotLoaded,
    /// The scene still names a TLE file that was not loaded.
    TargetFileNotLoaded,
    /// Reading the catalogue failed.
    Catalog(String),
    /// A time string could not be parsed.
    BadTime(&'static str),
    /// An orbit could not be evaluated.
    Orbit(String),
    /// The attitude is missing or inconsistent.
    BadPointing(&'static str),
    /// A setting names a target that `scene.targets` does not hold.
    UnknownTarget(String),
    /// The instrument profile has an invalid setting.
    BadProfile(String),
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimError::NoPointing(what) => write!(f, "{what} needs scene.pointing"),
            SimError::BadSource {
                kind,
                index,
                reason,
            } => write!(f, "{kind} {index}: {reason}"),
            SimError::CatalogNotLoaded => {
                write!(
                    f,
                    "scene.catalog is set but was not loaded (call add_catalog_stars first)"
                )
            }
            SimError::Catalog(e) => write!(f, "catalogue: {e}"),
            SimError::TargetFileNotLoaded => write!(
                f,
                "scene.target_file is set but was not loaded (call add_tle_file_targets first)"
            ),
            SimError::BadTime(what) => {
                write!(f, "{what}: expected UTC as YYYY-MM-DDTHH:MM:SS[.fff]")
            }
            SimError::Orbit(e) => write!(f, "orbit: {e}"),
            SimError::BadPointing(e) => write!(f, "pointing: {e}"),
            SimError::UnknownTarget(n) => write!(f, "no target named {n:?} in scene.targets"),
            SimError::BadProfile(e) => write!(f, "instrument: {e}"),
        }
    }
}

impl std::error::Error for SimError {}
