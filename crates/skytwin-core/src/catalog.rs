//! Stars from a catalogue, projected through the scene's WCS.
//!
//! Reading uses `fayde-catalog`, so any catalogue Fayde can read works here
//! and simulated frames use the same star positions Fayde solves against.
//!
//! Stars are kept when they land on the image or within a margin of it:
//! five PSF sigmas plus two pixels, plus any extra margin (the trail length
//! when tracking, so trailed stars entering the frame are drawn).

use std::collections::HashMap;
use std::path::PathBuf;

use fayde_catalog::{ucac4::Ucac4Catalog, Band, Catalog, StarQuery};
use fayde_core::{Epoch, SkyPos};
use serde::{Deserialize, Serialize};

use crate::camera::Camera;
use crate::error::SimError;
use crate::geometry::Geometry;
use crate::psf::fwhm_to_sigma;
use crate::scene::{Scene, Star};
use crate::wcs::TanWcs;

/// Catalogue magnitude to use as the camera band.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BandName {
    /// UCAC4 model magnitude (579–642 nm), falling back to aperture.
    #[default]
    Default,
    /// UCAC4 model magnitude only.
    UcacModel,
    /// UCAC4 aperture magnitude only.
    UcacAperture,
    /// APASS B.
    ApassB,
    /// APASS V.
    ApassV,
    /// APASS g′.
    ApassG,
    /// APASS r′.
    ApassR,
    /// APASS i′.
    ApassI,
}

impl From<BandName> for Band {
    fn from(b: BandName) -> Self {
        match b {
            BandName::Default => Band::Default,
            BandName::UcacModel => Band::UcacModel,
            BandName::UcacAperture => Band::UcacAperture,
            BandName::ApassB => Band::ApassB,
            BandName::ApassV => Band::ApassV,
            BandName::ApassG => Band::ApassG,
            BandName::ApassR => Band::ApassR,
            BandName::ApassI => Band::ApassI,
        }
    }
}

/// Which catalogue to read and how.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogConfig {
    /// UCAC4 directory (holding `u4b/`, optionally `u4i/`).
    #[serde(default)]
    pub ucac4: Option<PathBuf>,
    /// A CSV catalogue file instead (e.g. a Gaia extract).
    #[serde(default)]
    pub csv: Option<CsvCatalog>,
    /// Faintest magnitude to draw.
    pub mag_limit: f64,
    /// Band whose magnitudes are used as the camera band.
    #[serde(default)]
    pub band: BandName,
    /// Epoch (Julian year) to propagate proper motions to. Defaults to
    /// mid-exposure when the scene has a time; otherwise positions stay at
    /// the catalogue epoch.
    #[serde(default)]
    pub epoch_jyear: Option<f64>,
}

/// Pixel bounds a star must fall within to be drawn.
struct Footprint {
    /// Margin around the image, pixels.
    margin: f64,
    /// Image width, pixels.
    w: f64,
    /// Image height, pixels.
    h: f64,
}

impl Footprint {
    /// The image plus a PSF margin and `extra_margin_px`.
    fn new(camera: &Camera, extra_margin_px: f64) -> Self {
        Self {
            margin: 5.0 * fwhm_to_sigma(camera.fwhm_px) + 2.0 + extra_margin_px,
            w: camera.width as f64,
            h: camera.height as f64,
        }
    }

    /// Whether 0-based pixel `(x, y)` lies within the footprint.
    fn contains(&self, x: f64, y: f64) -> bool {
        let m = self.margin;
        !(x < -m || y < -m || x > self.w - 1.0 + m || y > self.h - 1.0 + m)
    }
}

/// Map a catalogue failure to [`SimError::Catalog`].
fn cat_err(e: impl std::fmt::Display) -> SimError {
    SimError::Catalog(e.to_string())
}

/// Largest pixel scale of `wcs` along any direction, arcsec/px: the larger
/// singular value of its CD matrix. The catalogue cone uses it so that a
/// stretched footprint (a ground frame refracted near the horizon) stays
/// inside the cone.
fn max_scale_arcsec(wcs: &TanWcs) -> f64 {
    let [[a, b], [c, d]] = wcs.cd;
    let s1 = a * a + b * b + c * c + d * d;
    let det = a * d - b * c;
    let big = libm::sqrt(0.5 * (s1 + libm::sqrt((s1 * s1 - 4.0 * det * det).max(0.0))));
    big * 3600.0
}

/// Catalogue stars that land on the image or within a PSF footprint of it.
/// `wcs` maps pixels to astrometric positions (the sky WCS).
///
/// When the camera has colour bands (from its colour term or passband), two
/// more queries in those bands give each star's colour index.
///
/// # Errors
///
/// [`SimError::Catalog`] when the WCS centre is not a valid sky position or
/// a catalogue query fails.
pub fn catalog_stars<C: Catalog>(
    cat: &C,
    wcs: &TanWcs,
    camera: &Camera,
    cfg: &CatalogConfig,
    epoch_jyear: Option<f64>,
    extra_margin_px: f64,
) -> Result<Vec<Star>, SimError> {
    let fp = Footprint::new(camera, extra_margin_px);
    let reach_px = wcs.field_radius_px(camera.width, camera.height) + fp.margin;
    let radius = (reach_px * max_scale_arcsec(wcs) / 3600.0).to_radians();
    let centre = SkyPos::from_degrees(wcs.crval_deg[0], wcs.crval_deg[1]).ok_or(
        SimError::Catalog("pointing is not a valid sky position".into()),
    )?;
    let query = StarQuery {
        epoch: epoch_jyear.and_then(Epoch::julian_year),
        mag_max: Some(cfg.mag_limit as f32),
        band: cfg.band.into(),
        ..StarQuery::default()
    };
    let hits = cat.cone(centre, radius, &query).map_err(cat_err)?;
    let color_bands = camera
        .color_term
        .as_ref()
        .and_then(|c| c.bands)
        .or_else(|| camera.passband.as_ref()?.color_bands);
    let colors: Option<HashMap<u64, f64>> = match color_bands {
        None => None,
        Some(bands) => {
            let band_mags = |b: BandName| -> Result<HashMap<u64, f64>, SimError> {
                let q = StarQuery {
                    band: b.into(),
                    mag_max: None,
                    ..query
                };
                Ok(cat
                    .cone(centre, radius, &q)
                    .map_err(cat_err)?
                    .into_iter()
                    .filter_map(|s| {
                        s.mag
                            .filter(|m| m.is_finite())
                            .map(|m| (s.id, f64::from(m)))
                    })
                    .collect())
            };
            let (a, b) = (band_mags(bands[0])?, band_mags(bands[1])?);
            Some(
                a.iter()
                    .filter_map(|(id, ma)| b.get(id).map(|mb| (*id, ma - mb)))
                    .collect(),
            )
        }
    };
    let mut out = Vec::new();
    for s in hits {
        let Some(mag) = s.mag else { continue };
        let (ra, dec) = (s.pos.ra.to_degrees(), s.pos.dec.to_degrees());
        if !wcs
            .world_to_pixel(ra, dec)
            .is_some_and(|(x, y)| fp.contains(x, y))
        {
            continue;
        }
        out.push(Star {
            id: Some(s.id),
            ra_deg: Some(ra),
            dec_deg: Some(dec),
            mag: Some(f64::from(mag)),
            color_index: colors.as_ref().and_then(|c| c.get(&s.id).copied()),
            ..Star::default()
        });
    }
    Ok(out)
}

/// If the scene names a catalogue, read it, append its stars to
/// `scene.stars` and clear `scene.catalog`. Returns the number added.
///
/// Proper motions go to `epoch_jyear`, else to mid-exposure when the scene
/// has a time. When tracking, the selection widens by the trail length.
///
/// # Errors
///
/// [`SimError::NoPointing`] when the scene has a catalogue but no pointing,
/// [`SimError::Catalog`] when not exactly one of `ucac4` and `csv` is given
/// or the catalogue cannot be read, or any error from building the
/// geometry.
pub fn add_catalog_stars(scene: &mut Scene, camera: &Camera) -> Result<usize, SimError> {
    let Some(cfg) = scene.catalog.take() else {
        return Ok(0);
    };
    let geom = Geometry::new(
        scene.pointing.as_ref(),
        scene.time.as_ref(),
        scene.observer.as_ref(),
        &scene.targets,
        camera,
    )?
    .ok_or(SimError::NoPointing("scene.catalog"))?;
    let epoch = cfg
        .epoch_jyear
        .or(geom.timing.map(|t| t.mid.julian_year_tt()));
    let trail = if geom.is_tracking() {
        let (ra, dec) = geom.pixel_to_sky(geom.camera_wcs.crpix0[0], geom.camera_wcs.crpix0[1]);
        let a = geom.sky_to_pixel_at(ra, dec, 0.0)?;
        let b = geom.sky_to_pixel_at(ra, dec, camera.exposure_s)?;
        a.zip(b)
            .map_or(0.0, |(a, b)| libm::hypot(a.0 - b.0, a.1 - b.1))
    } else {
        0.0
    } + horizon_fit_margin_px(&geom);
    let stars = match (&cfg.ucac4, &cfg.csv) {
        (Some(dir), None) => {
            let cat = Ucac4Catalog::open(dir)
                .map_err(|e| SimError::Catalog(format!("{}: {e}", dir.display())))?;
            catalog_stars(&cat, &geom.sky_wcs, camera, &cfg, epoch, trail)?
        }
        (None, Some(c)) => csv_stars(c, &geom.sky_wcs, camera, cfg.mag_limit, epoch, trail)?,
        _ => return Err(SimError::Catalog("give exactly one of ucac4 or csv".into())),
    };
    let n = stars.len();
    scene.stars.extend(stars);
    Ok(n)
}

/// Extra selection margin, pixels, for an observer at a ground site: twice
/// the fitted sky WCS's own error, which refraction near the horizon makes
/// large. Zero for every other observer, whose selection is unchanged.
fn horizon_fit_margin_px(geom: &Geometry) -> f64 {
    if geom.atmosphere.is_some() {
        2.0 * geom.sky_wcs_fit_px
    } else {
        0.0
    }
}

/// Which kind of catalogue a profile needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogKind {
    /// A UCAC4 directory.
    Ucac4,
    /// A CSV file (a Gaia extract, say).
    Csv,
}

/// The star catalogue a profile was calibrated against. Its zeropoint,
/// colour term and the stars the camera sees hold for that catalogue
/// only, so scenes using the profile take these settings and give only
/// where the file is: `"catalog": {"path": "..."}`. Anything the scene
/// does give wins; a catalogue of the other kind is an error unless the
/// scene sets `"ignore_profile": true`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogRequirement {
    /// What it is, e.g. "Gaia DR3 G < 10.5 + BSC5 bright stars".
    pub name: String,
    /// UCAC4 directory or CSV file.
    pub kind: CatalogKind,
    /// Faintest magnitude worth drawing for this camera.
    pub mag_limit: f64,
    /// UCAC4 band used as the catalogue magnitude.
    #[serde(default)]
    pub band: Option<BandName>,
    /// CSV column names.
    #[serde(default)]
    pub columns: Option<CsvColumns>,
    /// CSV epoch of the positions, Julian year.
    #[serde(default)]
    pub epoch_jyear: Option<f64>,
    /// SHA-256 of the file calibrated against (CSV), to check a copy.
    #[serde(default)]
    pub sha256: Option<String>,
    /// Rows in that file (CSV, header excluded).
    #[serde(default)]
    pub rows: Option<u64>,
    /// How to get or rebuild it.
    #[serde(default)]
    pub source: Option<String>,
}

impl CatalogRequirement {
    /// Fill a scene's `catalog` JSON from this requirement (see the type).
    ///
    /// A `path` shorthand becomes `ucac4` or `csv.path`; `mag_limit`, `band`
    /// (UCAC4) and `columns`/`epoch_jyear` (CSV) are filled where missing.
    ///
    /// # Errors
    ///
    /// A message when `cat` is not an object, gives both `path` and
    /// `ucac4`/`csv`, names a catalogue of the other kind (unless
    /// `ignore_profile` is true), or gives no catalogue path.
    pub fn fill(&self, cat: &mut serde_json::Value) -> Result<(), String> {
        let obj = cat
            .as_object_mut()
            .ok_or("scene.catalog: expected an object")?;
        if obj.remove("ignore_profile").and_then(|v| v.as_bool()) == Some(true) {
            return Ok(());
        }
        let key = match self.kind {
            CatalogKind::Ucac4 => "ucac4",
            CatalogKind::Csv => "csv",
        };
        if let Some(p) = obj.remove("path") {
            if obj.contains_key("ucac4") || obj.contains_key("csv") {
                return Err("scene.catalog: give path, or ucac4/csv, not both".into());
            }
            let v = match self.kind {
                CatalogKind::Ucac4 => p,
                CatalogKind::Csv => serde_json::json!({ "path": p }),
            };
            obj.insert(key.into(), v);
        }
        let other = if key == "ucac4" { "csv" } else { "ucac4" };
        if obj.contains_key(other) {
            return Err(format!(
                "scene.catalog: the instrument profile was calibrated against {} (a {key} catalogue); \
                 its zeropoint and colour term do not hold for a {other} catalogue. Give \
                 scene.catalog.path to that catalogue, or set \"ignore_profile\": true",
                self.name
            ));
        }
        if !obj.contains_key(key) {
            return Err(format!(
                "scene.catalog: needs path (the profile wants {})",
                self.name
            ));
        }
        obj.entry("mag_limit").or_insert(self.mag_limit.into());
        if let (Some(b), CatalogKind::Ucac4) = (self.band, self.kind) {
            if !obj.contains_key("band") {
                obj.insert(
                    "band".into(),
                    serde_json::to_value(b).map_err(|e| e.to_string())?,
                );
            }
        }
        if self.kind == CatalogKind::Csv {
            let csv = obj
                .get_mut("csv")
                .and_then(|c| c.as_object_mut())
                .ok_or("scene.catalog.csv: expected an object with path")?;
            if let Some(c) = &self.columns {
                if !csv.contains_key("columns") {
                    csv.insert(
                        "columns".into(),
                        serde_json::to_value(c).map_err(|e| e.to_string())?,
                    );
                }
            }
            if let Some(e) = self.epoch_jyear {
                csv.entry("epoch_jyear").or_insert(e.into());
            }
        }
        Ok(())
    }
}

/// Column names in a CSV catalogue (defaults: Gaia DR3). A column set to
/// `null` is not read.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CsvColumns {
    /// Identifier.
    pub id: Option<String>,
    /// Right ascension, degrees.
    pub ra: String,
    /// Declination, degrees.
    pub dec: String,
    /// Magnitude used as the catalogue magnitude.
    pub mag: String,
    /// Proper motion in RA·cos(Dec), mas/yr.
    pub pmra: Option<String>,
    /// Proper motion in Dec, mas/yr.
    pub pmdec: Option<String>,
    /// Parallax, mas.
    pub parallax: Option<String>,
    /// Colour index for `camera.color_term`.
    pub color: Option<String>,
}

impl Default for CsvColumns {
    fn default() -> Self {
        Self {
            id: Some("source_id".into()),
            ra: "ra".into(),
            dec: "dec".into(),
            mag: "phot_g_mean_mag".into(),
            pmra: Some("pmra".into()),
            pmdec: Some("pmdec".into()),
            parallax: Some("parallax".into()),
            color: Some("bp_rp".into()),
        }
    }
}

/// Gaia DR3 reference epoch, Julian year.
fn gaia_epoch() -> f64 {
    2016.0
}

/// A CSV star catalogue (e.g. an extract from the Gaia archive).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CsvCatalog {
    /// The file: a header row, then one star per row, comma-separated.
    pub path: PathBuf,
    /// Column names (default: Gaia DR3).
    #[serde(default)]
    pub columns: CsvColumns,
    /// Epoch of the positions, Julian year (default 2016.0, Gaia DR3).
    #[serde(default = "gaia_epoch")]
    pub epoch_jyear: f64,
}

/// Stars from a CSV catalogue that land on the image, moved by proper
/// motion to `epoch_jyear` (when given).
///
/// Blank lines and lines starting with `#` are skipped, fields are split on
/// commas (no quoted commas) and stripped of quotes, and column names match
/// case-insensitively. Rows without a finite RA, Dec and magnitude, or
/// fainter than `mag_limit`, are skipped. Proper motion needs both columns;
/// only positive parallaxes are kept.
///
/// # Errors
///
/// [`SimError::Catalog`] when the file cannot be read or a named column is
/// missing from the header.
pub fn csv_stars(
    c: &CsvCatalog,
    wcs: &TanWcs,
    camera: &Camera,
    mag_limit: f64,
    epoch_jyear: Option<f64>,
    extra_margin_px: f64,
) -> Result<Vec<Star>, SimError> {
    let text = std::fs::read_to_string(&c.path)
        .map_err(|e| SimError::Catalog(format!("{}: {e}", c.path.display())))?;
    let mut lines = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let header: Vec<&str> = lines
        .next()
        .unwrap_or("")
        .split(',')
        .map(|h| h.trim().trim_matches('"'))
        .collect();
    let col = |name: &str| -> Result<usize, SimError> {
        header
            .iter()
            .position(|h| h.eq_ignore_ascii_case(name))
            .ok_or_else(|| SimError::Catalog(format!("{}: no column {name:?}", c.path.display())))
    };
    let opt = |n: &Option<String>| -> Result<Option<usize>, SimError> {
        n.as_deref().map(col).transpose()
    };
    let (ira, idec, imag) = (
        col(&c.columns.ra)?,
        col(&c.columns.dec)?,
        col(&c.columns.mag)?,
    );
    let (iid, ipa, ipd, ipx, icol) = (
        opt(&c.columns.id)?,
        opt(&c.columns.pmra)?,
        opt(&c.columns.pmdec)?,
        opt(&c.columns.parallax)?,
        opt(&c.columns.color)?,
    );
    let fp = Footprint::new(camera, extra_margin_px);
    let from = Epoch::julian_year(c.epoch_jyear);
    let to = epoch_jyear.and_then(Epoch::julian_year);
    let mut out = Vec::new();
    for line in lines {
        let f: Vec<&str> = line
            .split(',')
            .map(|v| v.trim().trim_matches('"'))
            .collect();
        let num = |i: Option<usize>| {
            i.and_then(|i| f.get(i))
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite())
        };
        let (Some(ra), Some(dec), Some(mag)) = (num(Some(ira)), num(Some(idec)), num(Some(imag)))
        else {
            continue;
        };
        if mag > mag_limit {
            continue;
        }
        let Some(mut pos) = SkyPos::from_degrees(ra, dec) else {
            continue;
        };
        if let (Some(a), Some(b), Some(from), Some(to)) = (num(ipa), num(ipd), from, to) {
            pos = fayde_catalog::propagate::propagate(pos, [a, b], from, to);
        }
        let (ra, dec) = (pos.ra.to_degrees(), pos.dec.to_degrees());
        if !wcs
            .world_to_pixel(ra, dec)
            .is_some_and(|(x, y)| fp.contains(x, y))
        {
            continue;
        }
        out.push(Star {
            id: iid
                .and_then(|i| f.get(i))
                .and_then(|v| v.parse::<u64>().ok()),
            ra_deg: Some(ra),
            dec_deg: Some(dec),
            mag: Some(mag),
            color_index: num(icol),
            parallax_mas: num(ipx).filter(|p| *p > 0.0),
            ..Star::default()
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcs::Pointing;
    use fayde_catalog::memory::MemoryCatalog;

    fn star(id: u64, ra: f64, dec: f64, mag: f32) -> fayde_catalog::Star {
        fayde_catalog::Star {
            id,
            pos: SkyPos::from_degrees(ra, dec).unwrap(),
            mag: Some(mag),
            pos_err_mas: [10.0, 10.0],
            pm_mas_yr: [None, None],
        }
    }

    /// One star with a 1″/yr proper motion north, one too faint.
    #[test]
    fn gaia_csv_with_proper_motion() {
        let dir = std::env::temp_dir().join(format!("skytwin-gaia-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("g.csv");
        std::fs::write(&p, "source_id,ra,dec,phot_g_mean_mag,pmra,pmdec,parallax,bp_rp\n7,10.0,20.0,12.0,0.0,1000.0,5.0,0.8\n8,10.0,20.0,19.0,,,,\n").unwrap();
        let cam = Camera {
            width: 200,
            height: 100,
            pixel_scale_arcsec: 3.6,
            ..Camera::default()
        };
        let pt = Pointing {
            ra_deg: 10.0,
            dec_deg: 20.0,
            roll_deg: 0.0,
        };
        let wcs = TanWcs::from_pointing(&pt, cam.width, cam.height, cam.pixel_scale_arcsec);
        let c = CsvCatalog {
            path: p,
            columns: CsvColumns::default(),
            epoch_jyear: 2016.0,
        };
        let s = csv_stars(&c, &wcs, &cam, 16.0, Some(2026.0), 0.0).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(
            (s[0].id, s[0].color_index, s[0].parallax_mas),
            (Some(7), Some(0.8), Some(5.0))
        );
        let moved = (s[0].dec_deg.unwrap() - 20.0) * 3600.0;
        assert!((moved - 10.0).abs() < 1e-6, "{moved}");
    }

    /// Of a centre star, one 300 px north (off), one too faint and one
    /// 40 px north (on), the first and last are kept.
    #[test]
    fn keeps_stars_on_the_image_and_bright_enough() {
        let cam = Camera {
            width: 200,
            height: 100,
            pixel_scale_arcsec: 3.6,
            ..Camera::default()
        };
        let p = Pointing {
            ra_deg: 10.0,
            dec_deg: 20.0,
            roll_deg: 0.0,
        };
        let wcs = TanWcs::from_pointing(&p, cam.width, cam.height, cam.pixel_scale_arcsec);
        let cat = MemoryCatalog::new(vec![
            star(1, 10.0, 20.0, 12.0),
            star(2, 10.0, 20.3, 12.0),
            star(3, 10.0, 20.0, 18.0),
            star(4, 10.0, 20.04, 11.0),
        ]);
        let cfg = CatalogConfig {
            ucac4: None,
            csv: None,
            mag_limit: 16.0,
            band: BandName::Default,
            epoch_jyear: None,
        };
        let ids: Vec<u64> = catalog_stars(&cat, &wcs, &cam, &cfg, None, 0.0)
            .unwrap()
            .iter()
            .filter_map(|s| s.id)
            .collect();
        assert_eq!(ids, vec![1, 4]);
    }
}
