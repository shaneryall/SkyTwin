//! Header style: write FITS headers the way a given instrument does.
//!
//! A profile's `header` block can switch off the WCS (raw spacecraft frames
//! usually have none) and the simulator's own keywords, give the roll in the
//! instrument's convention, and add cards with fixed or computed values.
//! A string value of the form `{name}` or `{name:option}` is computed from
//! the frame:
//!
//! | Name | Value |
//! | --- | --- |
//! | `date_obs` | Exposure start, ISO 8601 UTC |
//! | `jd_obs`, `mjd_obs` | Exposure start, Julian / modified Julian date (UTC) |
//! | `exposure_s` | Exposure time |
//! | `ccd_temp_c`, `ccd_temp_k` | CCD temperature |
//! | `ra_deg`, `dec_deg`, `roll_deg` | Reported pointing (roll: position angle of image +y, east of north) |
//! | `ra_hms`, `dec_dms` | Reported pointing, sexagesimal; option `colon` (default) or `space` |
//! | `sub_lat_deg`, `sub_lon_deg` | Sub-satellite point at mid-exposure |
//! | `earth_limb_deg` | Angle from the boresight to the Earth's limb |
//! | `binning_x`, `binning_y` | Binning |
//!
//! Option `str` writes a number as a quoted string, as some instruments do
//! (e.g. `GAIN = '1.10'`).

use serde::{Deserialize, Serialize};

use crate::fits::Value;
use crate::render::Truth;

/// The roll keyword in an instrument's convention: `offset + sign × roll`,
/// where `roll` is the simulator's (position angle of image +y, east of
/// north).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RollKeyword {
    /// Keyword name.
    pub keyword: String,
    /// Offset, degrees.
    #[serde(default)]
    pub offset_deg: f64,
    /// +1 or −1.
    #[serde(default = "plus_one")]
    pub sign: f64,
    /// Comment.
    #[serde(default)]
    pub comment: String,
}

/// Serde default for [`RollKeyword::sign`].
fn plus_one() -> f64 {
    1.0
}

/// Serde default for the [`HeaderStyle`] switches that are on unless set.
fn yes() -> bool {
    true
}

/// One extra card.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CardSpec {
    /// Keyword (up to 8 characters: A–Z, 0–9, `-`, `_`).
    pub key: String,
    /// Number, string, boolean, or a `{template}` string.
    pub value: serde_json::Value,
    /// Comment.
    #[serde(default)]
    pub comment: String,
}

/// How the header is written.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderStyle {
    /// Write the WCS (default true).
    #[serde(default = "yes")]
    pub wcs: bool,
    /// Write the simulator's own keywords: times, observer, attitude, seed
    /// (default true).
    #[serde(default = "yes")]
    pub skytwin_keywords: bool,
    /// The roll keyword, if the instrument writes one.
    #[serde(default)]
    pub roll: Option<RollKeyword>,
    /// Extra cards, in order.
    #[serde(default)]
    pub cards: Vec<CardSpec>,
    /// Carry rounded sexagesimal seconds into the minutes (default true).
    /// `false` copies a common formatter bug: 59.96 s is written `60.0`.
    #[serde(default = "yes")]
    pub sexagesimal_carry: bool,
    /// Sometimes write a wrong WCS, as another solver's failed solution.
    #[serde(default)]
    pub wrong_wcs: Option<WrongWcs>,
}

/// A wrong WCS in the header: the tangent point moved and the axes turned.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WrongWcs {
    /// Fraction of frames with a wrong WCS.
    pub probability: f64,
    /// How far the tangent point is moved, arcmin, in a random direction.
    pub offset_arcmin: crate::param::Param,
    /// How far the axes are turned, degrees, either way (default 0).
    #[serde(default = "zero_param")]
    pub roll_deg: crate::param::Param,
    /// Relative scale error (default none).
    #[serde(default)]
    pub scale_error: Option<crate::param::Param>,
}

/// Serde default for [`WrongWcs::roll_deg`].
fn zero_param() -> crate::param::Param {
    crate::param::Param::Fixed(0.0)
}

/// Names a `{template}` value may use (see the module table).
const NAMES: [&str; 16] = [
    "date_obs",
    "jd_obs",
    "mjd_obs",
    "exposure_s",
    "ccd_temp_c",
    "ccd_temp_k",
    "ra_deg",
    "dec_deg",
    "roll_deg",
    "ra_hms",
    "dec_dms",
    "sub_lat_deg",
    "sub_lon_deg",
    "earth_limb_deg",
    "binning_x",
    "binning_y",
];

/// Whether `k` is a valid FITS keyword: 1 to 8 of `A`–`Z`, `0`–`9`, `-`, `_`
/// (FITS 4.0 section 4.1.2.1).
fn valid_key(k: &str) -> bool {
    !k.is_empty()
        && k.len() <= 8
        && k.chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Splits `{name}` or `{name:option}` into name and option (empty if none);
/// `None` if `s` is not a template.
fn template(s: &str) -> Option<(&str, &str)> {
    let inner = s.strip_prefix('{')?.strip_suffix('}')?;
    Some(inner.split_once(':').unwrap_or((inner, "")))
}

impl HeaderStyle {
    /// Checks keywords, template names and options, and the wrong-WCS
    /// settings.
    ///
    /// # Errors
    ///
    /// Fails on an invalid keyword, an unknown template name or option, a
    /// card value that is not a number, string or boolean, or a
    /// `wrong_wcs` probability outside 0 to 1 or invalid parameter.
    pub fn validate(&self) -> Result<(), String> {
        let mut keys: Vec<&str> = self.cards.iter().map(|c| c.key.as_str()).collect();
        keys.extend(self.roll.as_ref().map(|r| r.keyword.as_str()));
        for k in keys {
            if !valid_key(k) {
                return Err(format!("header: invalid keyword {k:?}"));
            }
        }
        if let Some(f) = &self.wrong_wcs {
            if !(0.0..=1.0).contains(&f.probability) {
                return Err("header.wrong_wcs.probability: must be 0 to 1".into());
            }
            f.offset_arcmin.validate("header.wrong_wcs.offset_arcmin")?;
            f.roll_deg.validate("header.wrong_wcs.roll_deg")?;
            if let Some(p) = &f.scale_error {
                p.validate("header.wrong_wcs.scale_error")?;
            }
        }
        for c in &self.cards {
            if !crate::fits::is_header_text(&c.comment) {
                return Err(format!(
                    "header: the comment of {} must be printable ASCII (FITS allows no other characters)",
                    c.key
                ));
            }
            if let Some(s) = c.value.as_str() {
                if template(s).is_none() {
                    if !crate::fits::is_header_text(s) {
                        return Err(format!(
                            "header: {} = {s:?} must be printable ASCII (FITS allows no other characters)",
                            c.key
                        ));
                    }
                    let len = s.len() + s.matches('\'').count();
                    if len > crate::fits::MAX_STRING_CHARS {
                        return Err(format!(
                            "header: {} is {len} characters; a FITS card holds at most {}",
                            c.key,
                            crate::fits::MAX_STRING_CHARS
                        ));
                    }
                }
                if let Some((name, opt)) = template(s) {
                    if !NAMES.contains(&name) {
                        return Err(format!("header: unknown template {{{name}}} in {}", c.key));
                    }
                    if !["", "str", "colon", "space"].contains(&opt) {
                        return Err(format!("header: unknown option {opt:?} in {}", c.key));
                    }
                }
            } else if !(c.value.is_number() || c.value.is_boolean()) {
                return Err(format!(
                    "header: {} must be a number, string or boolean",
                    c.key
                ));
            }
        }
        Ok(())
    }
}

/// Formats `v` degrees as `dd:mm:ss.s`, or as `hh:mm:ss.s` hours if
/// `hours`, with `sep` between fields.
///
/// With `carry`, the value is rounded to 0.1″ (or 0.1 s) before it is split,
/// so 59.95 carries into the minutes. Without it, the value is split first
/// and the seconds rounded alone, so 59.96 is written `60.0`, as some
/// instrument formatters do.
fn sexagesimal(v: f64, hours: bool, sep: char, carry: bool) -> String {
    let (sign, v) = if v < 0.0 { ("-", -v) } else { ("", v) };
    let v = if hours { v / 15.0 } else { v };
    if !carry {
        let a = libm::floor(v);
        let m = libm::floor((v - a) * 60.0);
        let sec = (v - a - m / 60.0) * 3600.0;
        return format!("{sign}{a:02}{sep}{m:02}{sep}{sec:04.1}");
    }
    let tenths = libm::round(v * 36_000.0) as i64;
    let (a, rem) = (tenths / 36_000, tenths % 36_000);
    let (b, c) = (rem / 600, rem % 600);
    format!("{sign}{a:02}{sep}{b:02}{sep}{:04.1}", c as f64 / 10.0)
}

/// The value of template `name` with option `opt` for frame `t`, if the
/// frame has it.
fn compute(t: &Truth, binning: [usize; 2], name: &str, opt: &str, carry: bool) -> Option<Value> {
    let rep = t.pointing.as_ref().map(|p| p.reported);
    let num = |v: f64| Some(Value::Float(v));
    let sep = if opt == "space" { ' ' } else { ':' };
    let v = match name {
        "date_obs" => return t.time.as_ref().map(|x| Value::Str(x.start_utc.clone())),
        "jd_obs" => num(t.time.as_ref()?.mjd_start + 2_400_000.5),
        "mjd_obs" => num(t.time.as_ref()?.mjd_start),
        "exposure_s" => num(t.camera.exposure_s),
        "ccd_temp_c" => num(t.instrument.settings.temperature_c?),
        "ccd_temp_k" => num(t.instrument.settings.temperature_c? + 273.15),
        "ra_deg" => num(rep?.ra_deg),
        "dec_deg" => num(rep?.dec_deg),
        "roll_deg" => num(rep?.roll_deg),
        "ra_hms" => return Some(Value::Str(sexagesimal(rep?.ra_deg, true, sep, carry))),
        "dec_dms" => return Some(Value::Str(sexagesimal(rep?.dec_deg, false, sep, carry))),
        "sub_lat_deg" => num(t.instrument.radiation.as_ref()?.subpoint_deg?[0]),
        "sub_lon_deg" => num(t.instrument.radiation.as_ref()?.subpoint_deg?[1]),
        "earth_limb_deg" => num(t
            .instrument
            .stray_light
            .iter()
            .find(|s| s.body == crate::stray::Body::Earth)?
            .angle_deg),
        "binning_x" => Some(Value::Int(binning[0] as i64)),
        "binning_y" => Some(Value::Int(binning[1] as i64)),
        _ => None,
    };
    match (v, opt) {
        (Some(Value::Float(f)), "str") => Some(Value::Str(format!("{f}"))),
        (Some(Value::Int(i)), "str") => Some(Value::Str(i.to_string())),
        (v, _) => v,
    }
}

/// Cards for an instrument-style header: the roll keyword, then the extra
/// cards in order; computed cards the frame has no value for are left out.
/// They go after the simulator's own cards, if any.
pub fn style_cards(
    style: &HeaderStyle,
    t: &Truth,
    binning: [usize; 2],
) -> Vec<(String, Value, String)> {
    let mut out = Vec::new();
    if let (Some(r), Some(p)) = (&style.roll, t.pointing.as_ref()) {
        let v = (r.offset_deg + r.sign * p.reported.roll_deg).rem_euclid(360.0);
        out.push((r.keyword.clone(), Value::Float(v), r.comment.clone()));
    }
    for c in &style.cards {
        let v = match &c.value {
            serde_json::Value::Bool(b) => Some(Value::Bool(*b)),
            serde_json::Value::Number(n) => match n.as_i64() {
                Some(i) => Some(Value::Int(i)),
                None => n.as_f64().map(Value::Float),
            },
            serde_json::Value::String(s) => match template(s) {
                Some((name, opt)) => compute(t, binning, name, opt, style.sexagesimal_carry),
                None => Some(Value::Str(s.clone())),
            },
            _ => None,
        };
        if let Some(v) = v {
            out.push((c.key.clone(), v, c.comment.clone()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sexagesimal_formats() {
        assert_eq!(sexagesimal(33.570_833, true, ':', true), "02:14:17.0");
        assert_eq!(sexagesimal(-3.491_361, false, ' ', true), "-03 29 28.9");
        assert_eq!(sexagesimal(14.999_99, true, ':', true), "01:00:00.0");
        assert_eq!(sexagesimal(14.999_99, true, ':', false), "00:59:60.0");
        assert_eq!(sexagesimal(33.570_833, true, ':', false), "02:14:17.0");
    }

    #[test]
    fn validation() {
        let ok: HeaderStyle = serde_json::from_value(serde_json::json!({
            "wcs": false,
            "roll": { "keyword": "ROLLANG", "offset_deg": 90.0, "sign": -1 },
            "cards": [{ "key": "RA", "value": "{ra_hms}" }, { "key": "GAIN", "value": "1.50" }, { "key": "JD-OBS", "value": "{jd_obs:str}" }]
        }))
        .unwrap();
        ok.validate().unwrap();
        let bad: HeaderStyle = serde_json::from_value(
            serde_json::json!({ "cards": [{ "key": "RA", "value": "{ra_hours}" }] }),
        )
        .unwrap();
        assert!(bad.validate().is_err());
        let bad: HeaderStyle = serde_json::from_value(
            serde_json::json!({ "cards": [{ "key": "TOOLONGKEY", "value": 1 }] }),
        )
        .unwrap();
        assert!(bad.validate().is_err());
    }

    /// Text FITS can't hold is refused before a frame is written.
    #[test]
    fn non_ascii_and_overlong_strings_are_refused() {
        let style = |card: serde_json::Value| -> HeaderStyle {
            serde_json::from_value(serde_json::json!({ "cards": [card] })).unwrap()
        };
        let accented = style(serde_json::json!({ "key": "OBSERVAT", "value": "Montréal" }));
        assert!(accented.validate().unwrap_err().contains("printable ASCII"));
        let long = style(serde_json::json!({ "key": "OBJECT", "value": "x".repeat(75) }));
        assert!(long.validate().unwrap_err().contains("at most 68"));
        let quotes = style(serde_json::json!({ "key": "OBJECT", "value": "'".repeat(35) }));
        assert!(quotes.validate().is_err());
        let comment =
            style(serde_json::json!({ "key": "OBJECT", "value": "M31", "comment": "Montréal" }));
        assert!(comment.validate().is_err());
        let fits = style(serde_json::json!({ "key": "OBJECT", "value": "x".repeat(68) }));
        fits.validate().unwrap();
    }
}
