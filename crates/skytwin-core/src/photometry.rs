//! Passbands and source spectra: AB magnitudes by synthetic photometry.
//!
//! A passband is the system's response against wavelength: optics, filter
//! and quantum efficiency together (electrons per photon), plus the
//! collecting area. Sources are blackbodies; reflected sunlight is the Sun's
//! spectrum. AB magnitudes are photon-weighted:
//!
//! ```text
//! m_AB = −2.5 log10( ∫ f_ν T dλ/λ / ∫ 3631 Jy · T dλ/λ ),
//! ```
//!
//! and the electron rate from a source is `A/h ∫ f_ν T dλ/λ`.
//!
//! Sources: Oke & Gunn, Astrophys. J. 266, 713 (1983) for AB magnitudes; Bessell &
//! Murphy, PASP 124, 140 (2012), eq. A13, for photon-weighted synthetic
//! photometry; Planck's law; nominal solar values from IAU 2015 Resolution
//! B3 (`T_eff` = 5772 K, R = 695,700 km); `T_eff` from B−V after
//! Ballesteros, EPL 97, 34008 (2012).

use serde::{Deserialize, Serialize};

/// Planck constant, J s.
const H: f64 = 6.626_070_15e-34;
/// Speed of light, m/s.
const C: f64 = 299_792_458.0;
/// Boltzmann constant, J/K.
const K: f64 = 1.380_649e-23;
/// AB zero point, W m⁻² Hz⁻¹.
const AB_ZERO: f64 = 3631e-26;
/// Nominal solar effective temperature, K.
const T_SUN: f64 = 5772.0;
/// Nominal solar radius, m.
const R_SUN_M: f64 = 695_700e3;
/// Astronomical unit, m.
const AU_M: f64 = 149_597_870_700.0;

/// A response curve against wavelength.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Curve {
    /// Wavelengths, nm, increasing.
    pub wavelength_nm: Vec<f64>,
    /// Response (0–1) at those wavelengths.
    pub throughput: Vec<f64>,
}

/// The camera's passband.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Passband {
    /// Name, written to the header (`FILTER`).
    #[serde(default)]
    pub name: String,
    /// System response: optics × filter × quantum efficiency.
    pub response: Curve,
    /// Collecting area, m².
    pub aperture_area_m2: f64,
    /// Band in which given magnitudes are measured (e.g. the catalogue's);
    /// default: this passband.
    #[serde(default)]
    pub reference: Option<Curve>,
    /// Temperature for sources without one, K (default 5800).
    #[serde(default = "default_teff")]
    pub default_teff_k: f64,
    /// What a star's `color_index` is, to get its temperature: `b_v` or
    /// `bp_rp`. Without it, stars without `teff_k` take `default_teff_k`.
    #[serde(default)]
    pub color_index: Option<ColorIndex>,
    /// Catalogue bands of the colour index (UCAC4: first minus second).
    #[serde(default)]
    pub color_bands: Option<[crate::catalog::BandName; 2]>,
}

/// A colour index that gives a temperature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorIndex {
    /// Johnson B−V (Ballesteros 2012).
    BV,
    /// Gaia BP−RP (Mucciarelli, Bellazzini & Massari 2021, dwarfs).
    BpRp,
}

impl ColorIndex {
    /// Effective temperature, K, clamped to 2000–50,000 K.
    pub fn teff(self, c: f64) -> f64 {
        let t = match self {
            ColorIndex::BV => teff_from_bv(c),
            ColorIndex::BpRp => teff_from_bp_rp(c),
        };
        if t.is_finite() && t > 0.0 {
            t.clamp(T_MIN, T_MAX)
        } else {
            T_MAX
        }
    }
}

/// Lowest temperature handled, K.
const T_MIN: f64 = 2000.0;
/// Highest temperature handled, K.
const T_MAX: f64 = 50_000.0;
/// Entries in a [`ColorTable`].
const TABLE_N: usize = 256;

/// Camera-band minus reference-band magnitude against temperature,
/// tabulated on a log grid so per-star lookups are cheap.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorTable {
    offsets: Vec<f64>,
}

impl ColorTable {
    /// Offset at `teff_k`, interpolated linearly in log T.
    pub fn at(&self, teff_k: f64) -> f64 {
        let u = libm::log(teff_k.clamp(T_MIN, T_MAX) / T_MIN) / libm::log(T_MAX / T_MIN);
        let x = u * (TABLE_N - 1) as f64;
        let i = (x.floor() as usize).min(TABLE_N - 2);
        let f = x - i as f64;
        self.offsets[i] * (1.0 - f) + self.offsets[i + 1] * f
    }
}

/// Serde default for [`Passband::default_teff_k`].
fn default_teff() -> f64 {
    5800.0
}

impl Curve {
    /// Check the grid. `what` names the curve in messages.
    ///
    /// # Errors
    ///
    /// When there are fewer than two samples, the wavelengths do not
    /// strictly increase or do not match the throughputs in number, a
    /// throughput is negative or NaN, or the first wavelength is not
    /// positive.
    pub fn validate(&self, what: &str) -> Result<(), String> {
        let n = self.wavelength_nm.len();
        if n < 2
            || n != self.throughput.len()
            || self.wavelength_nm.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(format!(
                "{what}: wavelength_nm must increase and match throughput"
            ));
        }
        if self.throughput.iter().any(|t| t.is_nan() || *t < 0.0) || self.wavelength_nm[0] <= 0.0 {
            return Err(format!(
                "{what}: throughput must be ≥ 0 and wavelengths positive"
            ));
        }
        Ok(())
    }

    /// ∫ g(λ) T(λ) dλ/λ by the trapezoid rule on the curve's grid,
    /// refined 8× between samples.
    fn integrate(&self, g: impl Fn(f64) -> f64) -> f64 {
        let (l, t) = (&self.wavelength_nm, &self.throughput);
        let mut sum = 0.0;
        for i in 0..l.len() - 1 {
            let sub = 8;
            for k in 0..sub {
                let (a, b) = (
                    f64::from(k) / f64::from(sub),
                    f64::from(k + 1) / f64::from(sub),
                );
                let la = l[i] + a * (l[i + 1] - l[i]);
                let lb = l[i] + b * (l[i + 1] - l[i]);
                let ta = t[i] + a * (t[i + 1] - t[i]);
                let tb = t[i] + b * (t[i + 1] - t[i]);
                let fa = g(la) * ta / la;
                let fb = g(lb) * tb / lb;
                sum += 0.5 * (fa + fb) * (lb - la);
            }
        }
        sum
    }

    /// AB magnitude of a source with spectral flux density `f_nu(λ nm)`.
    fn ab_mag(&self, f_nu: impl Fn(f64) -> f64) -> f64 {
        -2.5 * libm::log10(self.integrate(f_nu) / self.integrate(|_| AB_ZERO))
    }
}

/// Planck's law per unit frequency, W m⁻² Hz⁻¹ sr⁻¹, at wavelength `nm`.
pub fn planck_nu(t_k: f64, nm: f64) -> f64 {
    let lam = nm * 1e-9;
    let nu = C / lam;
    2.0 * H * nu * nu * nu / (C * C) / libm::expm1(H * nu / (K * t_k))
}

/// Effective temperature from Gaia BP−RP for dwarfs: θ = 5040 K / T =
/// 0.4929 + 0.5092 C − 0.0353 C² (Mucciarelli, Bellazzini & Massari 2021,
/// table 1, at solar metallicity; calibrated for 0.39 ≤ C ≤ 1.50, σ ≈ 61 K).
pub fn teff_from_bp_rp(c: f64) -> f64 {
    5040.0 / (0.4929 + 0.5092 * c - 0.0353 * c * c)
}

/// Effective temperature from B−V (Ballesteros 2012).
pub fn teff_from_bv(bv: f64) -> f64 {
    4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
}

impl Passband {
    /// Check the curves and area.
    ///
    /// # Errors
    ///
    /// When a curve is invalid (see [`Curve::validate`]), the aperture area
    /// is not positive, or `default_teff_k` is outside 2000–50,000 K.
    pub fn validate(&self) -> Result<(), String> {
        self.response.validate("passband.response")?;
        if let Some(r) = &self.reference {
            r.validate("passband.reference")?;
        }
        if self.aperture_area_m2.is_nan() || self.aperture_area_m2 <= 0.0 {
            return Err("passband.aperture_area_m2 must be positive".into());
        }
        if !(self.default_teff_k >= T_MIN && self.default_teff_k <= T_MAX) {
            return Err("passband.default_teff_k must be 2000–50000 K".into());
        }
        Ok(())
    }

    /// AB magnitude giving 1 electron per second.
    pub fn zeropoint_ab(&self) -> f64 {
        let n0 = self.aperture_area_m2 / H * self.response.integrate(|_| AB_ZERO);
        2.5 * libm::log10(n0)
    }

    /// Camera-band AB magnitude minus reference-band AB magnitude for a
    /// blackbody of `teff_k` (0 without a separate reference band).
    pub fn color_offset(&self, teff_k: f64) -> f64 {
        let Some(r) = &self.reference else { return 0.0 };
        let b = |nm: f64| planck_nu(teff_k, nm);
        self.response.ab_mag(b) - r.ab_mag(b)
    }

    /// The colour table, if there is a separate reference band.
    pub fn color_table(&self) -> Option<ColorTable> {
        self.reference.as_ref()?;
        let offsets = (0..TABLE_N)
            .map(|i| {
                let t = T_MIN * libm::pow(T_MAX / T_MIN, i as f64 / (TABLE_N - 1) as f64);
                self.color_offset(t)
            })
            .collect();
        Some(ColorTable { offsets })
    }

    /// The Sun's AB magnitude in this passband at 1 au (blackbody).
    pub fn sun_ab_mag(&self) -> f64 {
        let dilution = core::f64::consts::PI * (R_SUN_M / AU_M).powi(2);
        self.response.ab_mag(|nm| dilution * planck_nu(T_SUN, nm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tophat(a: f64, b: f64) -> Curve {
        let n = 201;
        Curve {
            wavelength_nm: (0..n)
                .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
                .collect(),
            throughput: vec![1.0; n],
        }
    }

    /// 1 m², T = 1 over 500–600 nm: N0 = 3631 Jy / h × ln(1.2) ≈ 9.99e9 e/s.
    #[test]
    fn zeropoint_of_a_flat_spectrum() {
        let p = Passband {
            name: "t".into(),
            response: tophat(500.0, 600.0),
            aperture_area_m2: 1.0,
            reference: None,
            default_teff_k: 5800.0,
            color_index: None,
            color_bands: None,
        };
        let n0 = AB_ZERO / H * (1.2f64).ln();
        assert!(
            (p.zeropoint_ab() - 2.5 * n0.log10()).abs() < 1e-6,
            "{}",
            p.zeropoint_ab()
        );
    }

    #[test]
    fn sun_is_about_minus_26_8_ab_in_v() {
        let p = Passband {
            name: "v".into(),
            response: tophat(500.0, 600.0),
            aperture_area_m2: 1.0,
            reference: None,
            default_teff_k: 5800.0,
            color_index: None,
            color_bands: None,
        };
        let m = p.sun_ab_mag();
        assert!((m + 26.8).abs() < 0.25, "{m}");
    }

    /// Red band minus blue band: cool stars brighter (negative), hot fainter.
    #[test]
    fn colour_offset_follows_temperature() {
        let p = Passband {
            name: "r".into(),
            response: tophat(600.0, 700.0),
            aperture_area_m2: 1.0,
            reference: Some(tophat(400.0, 500.0)),
            default_teff_k: 5800.0,
            color_index: None,
            color_bands: None,
        };
        assert!(p.color_offset(3500.0) < -0.5);
        assert!(p.color_offset(15000.0) > 0.0);
        assert!((teff_from_bv(0.65) - 5778.0).abs() < 10.0);
        assert!((teff_from_bp_rp(0.82) - 5700.0).abs() < 100.0);
        let t = p.color_table().unwrap();
        for teff in [3000.0, 5772.0, 12000.0] {
            assert!((t.at(teff) - p.color_offset(teff)).abs() < 2e-3);
        }
    }
}
