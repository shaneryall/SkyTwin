//! Detector and optics model: size, scale, PSF, photometry, noise, gain.

use serde::{Deserialize, Serialize};

/// Sky background, given either directly or as a surface brightness.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Sky {
    /// Electrons per pixel per second.
    EPerS(f64),
    /// Surface brightness, magnitudes per square arcsecond, in the
    /// camera's zeropoint band. Zodiacal light seen from space is about
    /// 22–23.3 V mag/arcsec² away from the Sun and the ecliptic.
    MagPerArcsec2(f64),
}

/// Camera and exposure settings.
///
/// Missing fields take the [`Default`] values; unknown fields are errors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Camera {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Pixel scale, arcseconds per pixel.
    pub pixel_scale_arcsec: f64,
    /// PSF full width at half maximum, pixels.
    pub fwhm_px: f64,
    /// Exposure time, seconds.
    pub exposure_s: f64,
    /// Zeropoint: the magnitude that gives 1 electron per second in total
    /// (all optics, filter and quantum-efficiency losses folded in).
    pub zeropoint_mag: f64,
    /// Sky background.
    pub sky: Sky,
    /// Dark current, electrons per pixel per second (default 0.1).
    pub dark_e_per_s: f64,
    /// Read noise, electrons RMS.
    pub read_noise_e: f64,
    /// Gain, electrons per ADU.
    pub gain_e_per_adu: f64,
    /// Bias level, ADU.
    pub bias_adu: f64,
    /// Full well, electrons (pixels clip here before digitising).
    pub full_well_e: f64,
    /// Pixel pitch, µm (sets the area radiation hits fall on).
    pub pixel_pitch_um: f64,
    /// Image handedness on the sky (`negative`, the default, or `positive`).
    pub parity: crate::wcs::Parity,
    /// Pixel (0-based `[x, y]`) on the optical axis: where the pointing
    /// is (default: the image centre).
    pub optical_axis_px: Option<[f64; 2]>,
    /// Radial lens distortion about the optical axis (none: pure TAN).
    pub distortion: Option<crate::wcs::Radial>,
    /// Colour term from catalogue magnitudes to the camera band.
    pub color_term: Option<ColorTerm>,
    /// Quantum efficiency (electrons per incident photon), used to report
    /// photon counts in the truth; the zeropoint stays in electrons.
    pub quantum_efficiency: f64,
    /// The Sun's magnitude in the camera band (V: −26.74), for reflected
    /// light from targets.
    #[serde(default = "default_sun_mag")]
    pub sun_mag: f64,
    /// Height above the Earth's surface below which a line of sight counts
    /// as blocked (atmosphere and limb), km.
    #[serde(default = "default_limb_margin")]
    pub earth_limb_margin_km: f64,
    /// Passband and collecting area. When given, the zeropoint and the
    /// Sun's magnitude are computed from it (AB) and replace
    /// `zeropoint_mag` and `sun_mag`.
    pub passband: Option<crate::photometry::Passband>,
    /// Colour table from the passband (computed, not read).
    #[serde(skip)]
    pub color_table: Option<std::sync::Arc<crate::photometry::ColorTable>>,
}

/// Serde default for [`Camera::sun_mag`].
fn default_sun_mag() -> f64 {
    -26.74
}

/// Serde default for [`Camera::earth_limb_margin_km`].
fn default_limb_margin() -> f64 {
    100.0
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            pixel_scale_arcsec: 3.0,
            fwhm_px: 2.5,
            exposure_s: 10.0,
            zeropoint_mag: 20.0,
            sky: Sky::MagPerArcsec2(22.5),
            dark_e_per_s: 0.1,
            read_noise_e: 8.0,
            gain_e_per_adu: 1.5,
            bias_adu: 1000.0,
            full_well_e: 60_000.0,
            pixel_pitch_um: 13.0,
            parity: crate::wcs::Parity::Negative,
            optical_axis_px: None,
            distortion: None,
            color_term: None,
            quantum_efficiency: 1.0,
            sun_mag: default_sun_mag(),
            earth_limb_margin_km: default_limb_margin(),
            passband: None,
            color_table: None,
        }
    }
}

/// Colour term: `m_camera = m_catalogue + coefficient × colour index`,
/// with the index from two catalogue bands (first minus second).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColorTerm {
    /// The two bands of the colour index, e.g. `["apass_b", "apass_v"]`
    /// (UCAC4). Without them, the catalogue's own colour column (a CSV
    /// catalogue's `bp_rp`, say).
    #[serde(default)]
    pub bands: Option<[crate::catalog::BandName; 2]>,
    /// Coefficient, magnitudes per magnitude of colour.
    pub coefficient: f64,
    /// Index used for stars without both bands.
    #[serde(default)]
    pub default_index: f64,
}

impl Camera {
    /// Check that every number is one a detector can have.
    ///
    /// Zero exposure is allowed (bias frames); zero gain, scale, FWHM,
    /// full well or pixel pitch is not, since each divides or sizes the
    /// image.
    ///
    /// # Errors
    ///
    /// On the first invalid field, with a message naming it.
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err("camera: width and height must be at least 1".into());
        }
        let positive = [
            ("pixel_scale_arcsec", self.pixel_scale_arcsec),
            ("fwhm_px", self.fwhm_px),
            ("gain_e_per_adu", self.gain_e_per_adu),
            ("full_well_e", self.full_well_e),
            ("pixel_pitch_um", self.pixel_pitch_um),
            ("quantum_efficiency", self.quantum_efficiency),
        ];
        for (name, v) in positive {
            if !(v.is_finite() && v > 0.0) {
                return Err(format!(
                    "camera.{name} must be a finite number above 0, got {v}"
                ));
            }
        }
        let non_negative = [
            ("exposure_s", self.exposure_s),
            ("dark_e_per_s", self.dark_e_per_s),
            ("read_noise_e", self.read_noise_e),
            ("earth_limb_margin_km", self.earth_limb_margin_km),
        ];
        for (name, v) in non_negative {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!(
                    "camera.{name} must be a finite number ≥ 0, got {v}"
                ));
            }
        }
        for (name, v) in [
            ("zeropoint_mag", self.zeropoint_mag),
            ("bias_adu", self.bias_adu),
            ("sun_mag", self.sun_mag),
        ] {
            if !v.is_finite() {
                return Err(format!("camera.{name} must be finite, got {v}"));
            }
        }
        match self.sky {
            Sky::EPerS(v) if !(v.is_finite() && v >= 0.0) => Err(format!(
                "camera.sky.e_per_s must be a finite number ≥ 0, got {v}"
            )),
            Sky::MagPerArcsec2(v) if !v.is_finite() => Err(format!(
                "camera.sky.mag_per_arcsec2 must be finite, got {v}"
            )),
            _ => Ok(()),
        }
    }

    /// The camera with its zeropoint, Sun magnitude and colour table
    /// computed from the passband, or `None` without one.
    ///
    /// # Errors
    ///
    /// When the passband is invalid, or when both `color_term` and
    /// `passband.reference` are given.
    pub fn with_passband(&self) -> Result<Option<Camera>, String> {
        let Some(p) = &self.passband else {
            return Ok(None);
        };
        p.validate()?;
        if self.color_term.is_some() && p.reference.is_some() {
            return Err("camera: give color_term or passband.reference, not both".into());
        }
        let mut c = self.clone();
        c.zeropoint_mag = p.zeropoint_ab();
        c.sun_mag = p.sun_ab_mag();
        c.color_table = p.color_table().map(std::sync::Arc::new);
        Ok(Some(c))
    }

    /// Magnitude in the camera band of a star given in the reference band.
    pub fn star_mag(&self, mag: f64, color_index: Option<f64>, teff_k: Option<f64>) -> f64 {
        if let (Some(t), Some(p)) = (&self.color_table, &self.passband) {
            let teff = teff_k
                .or_else(|| Some(p.color_index?.teff(color_index?)))
                .unwrap_or(p.default_teff_k);
            return mag + t.at(teff);
        }
        match &self.color_term {
            Some(ct) => mag + ct.coefficient * color_index.unwrap_or(ct.default_index),
            None => mag,
        }
    }

    /// Total electrons over the exposure from a source of magnitude `mag`.
    pub fn mag_to_electrons(&self, mag: f64) -> f64 {
        self.mag_to_e_per_s(mag) * self.exposure_s
    }

    /// Electrons per second from a source of magnitude `mag`.
    pub fn mag_to_e_per_s(&self, mag: f64) -> f64 {
        libm::pow(10.0, -0.4 * (mag - self.zeropoint_mag))
    }

    /// Magnitude of a source giving `electrons` in total over the exposure.
    pub fn electrons_to_mag(&self, electrons: f64) -> f64 {
        self.zeropoint_mag - 2.5 * libm::log10(electrons / self.exposure_s)
    }

    /// Sky background, electrons per pixel per second.
    pub fn sky_e_per_s(&self) -> f64 {
        match self.sky {
            Sky::EPerS(e) => e,
            Sky::MagPerArcsec2(m) => {
                self.mag_to_e_per_s(m) * self.pixel_scale_arcsec * self.pixel_scale_arcsec
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round trip, and 5 magnitudes are a factor 100 in flux.
    #[test]
    fn magnitude_round_trip() {
        let c = Camera::default();
        assert!((c.mag_to_electrons(c.zeropoint_mag) - c.exposure_s).abs() < 1e-12);
        let e = c.mag_to_electrons(12.3);
        assert!((c.electrons_to_mag(e) - 12.3).abs() < 1e-12);
        assert!((c.mag_to_electrons(10.0) / c.mag_to_electrons(15.0) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn sky_surface_brightness() {
        let c = Camera {
            sky: Sky::MagPerArcsec2(20.0),
            zeropoint_mag: 20.0,
            pixel_scale_arcsec: 3.0,
            ..Camera::default()
        };
        assert!((c.sky_e_per_s() - 9.0).abs() < 1e-12);
    }

    /// A camera block that leaves out the dark current gets the
    /// documented default, not zero.
    #[test]
    fn omitted_dark_current_takes_the_default() {
        let c: Camera = serde_json::from_value(serde_json::json!({ "width": 64 })).unwrap();
        assert_eq!(c.dark_e_per_s, Camera::default().dark_e_per_s);
        assert_eq!(c.dark_e_per_s, 0.1);
    }
}
