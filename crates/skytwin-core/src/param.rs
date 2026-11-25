//! Settings that are either fixed or drawn per frame from a distribution.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

/// Shared state while drawing one frame's settings: the row picked for each
/// linked group.
#[derive(Debug, Default)]
pub struct DrawCtx {
    picks: HashMap<String, usize>,
}

/// Values drawn together: every setting in the same `group` takes the entry
/// at the same index, so correlations between settings survive (e.g. one
/// row per measured real frame).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Linked {
    /// Group name; all lists in a group must have the same length.
    pub group: String,
    /// Values, one per row.
    pub values: Vec<f64>,
}

/// A distribution to draw a setting from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Dist {
    /// `[mean, sigma]`.
    Normal([f64; 2]),
    /// `[low, high]`.
    Uniform([f64; 2]),
    /// `[low, high]`, both positive: uniform in the logarithm.
    LogUniform([f64; 2]),
    /// Pick one of these values with equal probability (e.g. measured
    /// values from a real archive).
    Empirical(Vec<f64>),
    /// Pick a row shared with every other setting in the same group.
    EmpiricalLinked(Linked),
}

/// A number, or a distribution drawn once per frame.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Param {
    /// Always this value.
    Fixed(f64),
    /// Drawn per frame.
    Dist(Dist),
}

/// Parsed by hand so a bad distribution gets a precise message instead of
/// serde's "did not match any variant".
impl<'de> Deserialize<'de> for Param {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let v = serde_json::Value::deserialize(d)?;
        match v {
            serde_json::Value::Number(n) => n
                .as_f64()
                .map(Param::Fixed)
                .ok_or_else(|| D::Error::custom("number out of range")),
            obj @ serde_json::Value::Object(_) => serde_json::from_value::<Dist>(obj)
                .map(Param::Dist)
                .map_err(|e| {
                    D::Error::custom(format!(
                        "expected a number or one of normal, uniform, log_uniform, empirical, \
                         empirical_linked: {e}"
                    ))
                }),
            other => Err(D::Error::custom(format!(
                "expected a number or a distribution, got {other}"
            ))),
        }
    }
}

impl Default for Param {
    fn default() -> Self {
        Param::Fixed(0.0)
    }
}

impl From<f64> for Param {
    fn from(v: f64) -> Self {
        Param::Fixed(v)
    }
}

impl Param {
    /// Check the distribution's parameters. `what` names the setting.
    ///
    /// # Errors
    ///
    /// When a value is not finite, a normal's sigma is negative, a range has
    /// `low > high`, a log-uniform low bound is not positive, or an empirical
    /// list is empty.
    pub fn validate(&self, what: &str) -> Result<(), String> {
        let ok = match self {
            Param::Fixed(v) => v.is_finite(),
            Param::Dist(Dist::Normal([m, s])) => m.is_finite() && s.is_finite() && *s >= 0.0,
            Param::Dist(Dist::Uniform([a, b])) => a.is_finite() && b.is_finite() && a <= b,
            Param::Dist(Dist::LogUniform([a, b])) => *a > 0.0 && b.is_finite() && a <= b,
            Param::Dist(Dist::Empirical(v)) => !v.is_empty() && v.iter().all(|x| x.is_finite()),
            Param::Dist(Dist::EmpiricalLinked(l)) => {
                !l.values.is_empty() && l.values.iter().all(|x| x.is_finite())
            }
        };
        if ok {
            Ok(())
        } else {
            Err(format!("{what}: invalid value or distribution {self:?}"))
        }
    }

    /// Draw a value. A fixed value draws nothing from `rng`, so adding a
    /// fixed setting never changes other draws. A linked group draws its row
    /// once per `ctx`; later settings in the group reuse it.
    ///
    /// # Panics
    ///
    /// On an empty empirical list; [`Param::validate`] rejects those.
    pub fn draw(&self, rng: &mut Rng, ctx: &mut DrawCtx) -> f64 {
        match self {
            Param::Fixed(v) => *v,
            Param::Dist(Dist::Normal([m, s])) => m + s * rng.normal(),
            Param::Dist(Dist::Uniform([a, b])) => rng.range(*a, *b),
            Param::Dist(Dist::LogUniform([a, b])) => {
                libm::exp(rng.range(libm::log(*a), libm::log(*b)))
            }
            Param::Dist(Dist::Empirical(v)) => v[pick_index(rng, v.len())],
            Param::Dist(Dist::EmpiricalLinked(l)) => {
                let n = l.values.len();
                let i = *ctx
                    .picks
                    .entry(l.group.clone())
                    .or_insert_with(|| pick_index(rng, n));
                l.values[i.min(n - 1)]
            }
        }
    }
}

/// A uniform index in `0..n` from one draw, clamped so a draw of exactly 1
/// stays in range. Panics when `n` is 0.
pub(crate) fn pick_index(rng: &mut Rng, n: usize) -> usize {
    ((rng.uniform() * n as f64) as usize).min(n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numbers_and_distributions() {
        let p: Param = serde_json::from_str("24").unwrap();
        assert_eq!(p, Param::Fixed(24.0));
        let p: Param = serde_json::from_str(r#"{"normal": [24, 8]}"#).unwrap();
        assert_eq!(p, Param::Dist(Dist::Normal([24.0, 8.0])));
        let e = serde_json::from_str::<Param>(r#"{"gamma": [1, 2]}"#)
            .unwrap_err()
            .to_string();
        assert!(e.contains("log_uniform") && e.contains("gamma"), "{e}");
    }

    #[test]
    fn draws_stay_in_range() {
        let mut r = Rng::new(3);
        for _ in 0..1000 {
            let c = &mut DrawCtx::default();
            let v = Param::Dist(Dist::LogUniform([4.0, 16.0])).draw(&mut r, c);
            assert!((4.0..=16.0).contains(&v));
            let e = Param::Dist(Dist::Empirical(vec![1.0, 5.0])).draw(&mut r, c);
            assert!(e == 1.0 || e == 5.0);
        }
        assert!(Param::Dist(Dist::LogUniform([0.0, 1.0]))
            .validate("x")
            .is_err());
    }

    #[test]
    fn linked_values_come_from_the_same_row() {
        let a: Param =
            serde_json::from_str(r#"{"empirical_linked": {"group": "f", "values": [1, 2, 3, 4]}}"#)
                .unwrap();
        let b: Param = serde_json::from_str(
            r#"{"empirical_linked": {"group": "f", "values": [10, 20, 30, 40]}}"#,
        )
        .unwrap();
        let mut r = Rng::new(1);
        for _ in 0..50 {
            let c = &mut DrawCtx::default();
            let (x, y) = (a.draw(&mut r, c), b.draw(&mut r, c));
            assert_eq!(y, 10.0 * x);
        }
    }
}
