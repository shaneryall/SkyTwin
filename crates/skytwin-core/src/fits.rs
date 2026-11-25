//! Minimal FITS writer: one primary HDU holding a 2-D image.
//!
//! Source: FITS Standard 4.0 (IAU FITS Working Group, 2018). Unsigned 16-bit
//! data is stored as BITPIX = 16 with BZERO = 32768 (section 5.3). Header
//! and data are each padded to whole 2880-byte blocks.

use std::io::{self, Write};

/// FITS logical record length, bytes.
const BLOCK: usize = 2880;

/// Longest string value a card holds: 80 columns less the keyword, the
/// value indicator and the two quotes (FITS 4.0 section 4.2.1.1).
pub const MAX_STRING_CHARS: usize = 68;

/// Whether `s` is FITS header text: printable ASCII, 0x20 to 0x7E (FITS 4.0
/// section 4.1.2.3).
pub fn is_header_text(s: &str) -> bool {
    s.bytes().all(|b| (0x20..=0x7E).contains(&b))
}

/// `s` with every character outside printable ASCII replaced by `?`.
fn header_text(s: &str) -> String {
    s.chars()
        .map(|c| if (' '..='~').contains(&c) { c } else { '?' })
        .collect()
}

/// One 80-character header card, with the comment cut to fit.
///
/// Values use the fixed format (FITS 4.0 section 4.2): strings start in
/// column 11, numbers and logicals end in column 30. An empty `value` gives
/// a card with no value indicator, such as `END`. The card is always 80
/// printable ASCII bytes: other characters become `?`, so a card can never
/// push `END` out of its column.
fn card(key: &str, value: &str, comment: &str) -> String {
    let mut c = if value.is_empty() {
        format!("{key:<8}")
    } else if value.starts_with('\'') {
        format!("{key:<8}= {value:<20}")
    } else {
        format!("{key:<8}= {value:>20}")
    };
    if !comment.is_empty() {
        c.push_str(" / ");
        c.push_str(comment);
    }
    let mut c = header_text(&c);
    c.truncate(80);
    format!("{c:<80}")
}

/// A header value: number or quoted string.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Integer.
    Int(i64),
    /// Real number, written with 17 significant digits so every `f64`
    /// reads back exactly.
    Float(f64),
    /// String (single quotes are escaped).
    Str(String),
    /// Logical.
    Bool(bool),
}

impl Value {
    /// The value as it appears in a card. A string is cut before its quotes
    /// are escaped, so the closing quote always fits and a doubled quote is
    /// never split.
    fn render(&self) -> String {
        match self {
            Value::Int(i) => i.to_string(),
            Value::Float(f) => format!("{f:.16E}"),
            Value::Str(s) => {
                let mut out = String::new();
                for c in header_text(s).chars() {
                    let add = if c == '\'' { 2 } else { 1 };
                    if out.len() + add > MAX_STRING_CHARS {
                        break;
                    }
                    out.push(c);
                    if c == '\'' {
                        out.push('\'');
                    }
                }
                format!("'{out:<8}'")
            }
            Value::Bool(b) => (if *b { "T" } else { "F" }).to_string(),
        }
    }
}

/// Pixel type of the written image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputType {
    /// Unsigned 16-bit (BITPIX 16 with BZERO 32768), 0 to 65535.
    #[default]
    U16,
    /// Signed 16-bit (BITPIX 16), −32768 to 32767.
    I16,
    /// Signed 32-bit (BITPIX 32).
    I32,
    /// 32-bit IEEE float (BITPIX −32): keeps fractions and negative values.
    F32,
}

impl OutputType {
    /// Smallest and largest storable value.
    pub fn range(self) -> (f64, f64) {
        match self {
            OutputType::U16 => (0.0, 65_535.0),
            OutputType::I16 => (-32_768.0, 32_767.0),
            OutputType::I32 => (-2_147_483_648.0, 2_147_483_647.0),
            OutputType::F32 => (-f64::from(f32::MAX), f64::from(f32::MAX)),
        }
    }

    /// Whether values are whole numbers.
    pub fn is_integer(self) -> bool {
        !matches!(self, OutputType::F32)
    }
}

/// Write `data` (row-major, first row = FITS row 1) as a FITS image of the
/// given type. Values are rounded (integer types) and clamped to the type's
/// range. `extra` cards (keyword, value, comment) go after the mandatory
/// ones.
///
/// # Errors
///
/// Returns any error from writing to `out`.
///
/// # Panics
///
/// Panics if `data.len()` is not `width * height`.
pub fn write_image<W: Write, K: AsRef<str>, C: AsRef<str>>(
    out: &mut W,
    width: usize,
    height: usize,
    data: &[f64],
    ty: OutputType,
    extra: &[(K, Value, C)],
) -> io::Result<()> {
    assert_eq!(
        data.len(),
        width * height,
        "data size does not match width x height"
    );
    let (bitpix, what) = match ty {
        OutputType::U16 | OutputType::I16 => ("16", "16-bit integers"),
        OutputType::I32 => ("32", "32-bit integers"),
        OutputType::F32 => ("-32", "32-bit IEEE floats"),
    };
    let mut hdr = String::new();
    hdr += &card("SIMPLE", "T", "conforms to FITS standard");
    hdr += &card("BITPIX", bitpix, what);
    hdr += &card("NAXIS", "2", "");
    hdr += &card("NAXIS1", &width.to_string(), "columns");
    hdr += &card("NAXIS2", &height.to_string(), "rows");
    if ty == OutputType::U16 {
        hdr += &card("BZERO", "32768", "unsigned 16-bit data");
        hdr += &card("BSCALE", "1", "");
    }
    for (k, v, c) in extra {
        hdr += &card(k.as_ref(), &v.render(), c.as_ref());
    }
    hdr += &card("END", "", "");
    while hdr.len() % BLOCK != 0 {
        hdr.push(' ');
    }
    out.write_all(hdr.as_bytes())?;
    let (lo, hi) = ty.range();
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        let v = if ty.is_integer() { libm::round(v) } else { v }.clamp(lo, hi);
        match ty {
            OutputType::U16 => bytes.extend_from_slice(&((v as i32 - 32768) as i16).to_be_bytes()),
            OutputType::I16 => bytes.extend_from_slice(&(v as i16).to_be_bytes()),
            OutputType::I32 => bytes.extend_from_slice(&(v as i32).to_be_bytes()),
            OutputType::F32 => bytes.extend_from_slice(&(v as f32).to_be_bytes()),
        }
    }
    while bytes.len() % BLOCK != 0 {
        bytes.push(0);
    }
    out.write_all(&bytes)
}

/// Writes unsigned 16-bit data as a [`OutputType::U16`] image; see
/// [`write_image`].
///
/// # Errors
///
/// Returns any error from writing to `out`.
///
/// # Panics
///
/// Panics if `data.len()` is not `width * height`.
pub fn write_u16<W: Write>(
    out: &mut W,
    width: usize,
    height: usize,
    data: &[u16],
    extra: &[(&str, Value, &str)],
) -> io::Result<()> {
    assert_eq!(
        data.len(),
        width * height,
        "data size does not match width x height"
    );
    let values: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
    write_image(out, width, height, &values, OutputType::U16, extra)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// I16 rounds and clamps (−5.4 → −5, 70 000 → 32 767); F32 and I32
    /// keep the value.
    #[test]
    fn signed_and_float_types() {
        let mut buf = Vec::new();
        write_image::<_, &str, &str>(&mut buf, 2, 1, &[-5.4, 70_000.0], OutputType::I16, &[])
            .unwrap();
        assert_eq!(&buf[BLOCK..BLOCK + 4], &[0xFF, 0xFB, 0x7F, 0xFF]);
        let mut buf = Vec::new();
        write_image::<_, &str, &str>(&mut buf, 1, 1, &[-1.5], OutputType::F32, &[]).unwrap();
        assert_eq!(&buf[BLOCK..BLOCK + 4], &(-1.5f32).to_be_bytes());
        assert!(String::from_utf8_lossy(&buf[..BLOCK]).contains("BITPIX  =                  -32"));
        let mut buf = Vec::new();
        write_image::<_, &str, &str>(&mut buf, 1, 1, &[-70_000.0], OutputType::I32, &[]).unwrap();
        assert_eq!(&buf[BLOCK..BLOCK + 4], &(-70_000i32).to_be_bytes());
    }

    /// Non-ASCII text and overlong strings still give 80-byte ASCII cards,
    /// with the closing quote in column 80 and `END` where readers look.
    #[test]
    fn header_text_stays_in_its_card() {
        let mut buf = Vec::new();
        write_u16(
            &mut buf,
            1,
            1,
            &[0],
            &[
                ("OBSERVAT", Value::Str("Montréal".into()), "café"),
                ("OBJECT", Value::Str("x".repeat(75)), "a long comment"),
                ("QUOTES", Value::Str("'".repeat(40)), ""),
            ],
        )
        .unwrap();
        let header = &buf[..BLOCK];
        assert!(header.iter().all(|b| (0x20..=0x7E).contains(b)));
        let cards: Vec<&[u8]> = header.chunks(80).collect();
        let end = cards
            .iter()
            .position(|c| c.starts_with(b"END     "))
            .unwrap();
        assert!(cards[end + 1..]
            .iter()
            .all(|c| c.iter().all(|&b| b == b' ')));
        let text = |k: &str| {
            let c = cards.iter().find(|c| c.starts_with(k.as_bytes())).unwrap();
            String::from_utf8(c.to_vec()).unwrap()
        };
        assert!(text("OBSERVAT").starts_with("OBSERVAT= 'Montr?al'"));
        let object = text("OBJECT");
        assert_eq!(&object[10..80], &format!("'{}'", "x".repeat(68)));
        let quotes = text("QUOTES");
        assert_eq!(&quotes[10..80], &format!("'{}'", "'".repeat(68)));
    }

    /// Header and data fill whole blocks; U16 pixels are offset by 32768
    /// (0 → −32768, 65535 → 32767, big-endian).
    #[test]
    fn layout_is_block_aligned() {
        let mut buf = Vec::new();
        write_u16(
            &mut buf,
            3,
            2,
            &[0, 1, 2, 65535, 32768, 7],
            &[("EXPTIME", Value::Float(1.0), "s")],
        )
        .unwrap();
        assert_eq!(buf.len(), 2 * BLOCK);
        assert!(buf.starts_with(b"SIMPLE  =                    T"));
        assert_eq!(
            &card("CREATOR", &Value::Str("x".into()).render(), "")[..20],
            "CREATOR = 'x       '"
        );
        assert_eq!(&buf[BLOCK..BLOCK + 2], &[0x80, 0x00]);
        assert_eq!(&buf[BLOCK + 6..BLOCK + 8], &[0x7F, 0xFF]);
    }
}
