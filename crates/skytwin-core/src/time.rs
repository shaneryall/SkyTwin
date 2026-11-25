//! Time scales: UTC as given, TAI through the leap-second table, TT.
//!
//! Leap seconds come from IERS Bulletin C (TAI − UTC = 37 s since
//! 2017-01-01, the last one added; extend `LEAP` when a bulletin announces a
//! new one). TT = TAI + 32.184 s (IAU 1991, Resolution A4). Civil-date
//! arithmetic follows Hinnant, "chrono-Compatible Low-Level Date
//! Algorithms" (2013).

use std::fmt;

/// Seconds in a day.
pub const DAY_S: f64 = 86_400.0;
/// MJD of J2000.0 (2000-01-01T12:00 TT).
pub const MJD_J2000: f64 = 51_544.5;

/// TAI − UTC from 1972 on: (MJD the value starts, seconds).
const LEAP: [(i64, f64); 28] = [
    (41317, 10.0),
    (41499, 11.0),
    (41683, 12.0),
    (42048, 13.0),
    (42413, 14.0),
    (42778, 15.0),
    (43144, 16.0),
    (43509, 17.0),
    (43874, 18.0),
    (44239, 19.0),
    (44786, 20.0),
    (45151, 21.0),
    (45516, 22.0),
    (46247, 23.0),
    (47161, 24.0),
    (47892, 25.0),
    (48257, 26.0),
    (48804, 27.0),
    (49169, 28.0),
    (49534, 29.0),
    (50083, 30.0),
    (50630, 31.0),
    (51179, 32.0),
    (53736, 33.0),
    (54832, 34.0),
    (56109, 35.0),
    (57204, 36.0),
    (57754, 37.0),
];

/// An instant, stored as a UTC day number and seconds into that day.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Utc {
    /// Modified Julian Date of the UTC day.
    pub mjd_day: i64,
    /// Seconds since UTC midnight.
    pub sec: f64,
}

/// Days from 1970-01-01 to the proleptic Gregorian date `y-m-d` (Hinnant's
/// `days_from_civil`).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Proleptic Gregorian `(year, month, day)` of day `z` counted from
/// 1970-01-01 (Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// MJD of 1970-01-01.
const MJD_UNIX: i64 = 40_587;

/// Days in month `mo` (1–12) of year `y`, Gregorian calendar.
fn days_in_month(y: i64, mo: i64) -> i64 {
    match mo {
        4 | 6 | 9 | 11 => 30,
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        _ => 31,
    }
}

impl Utc {
    /// The instant at UTC calendar date `y-mo-d` and time `h:mi:s`; out-of-range
    /// times of day carry into the date.
    pub fn from_civil(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: f64) -> Self {
        Self {
            mjd_day: days_from_civil(y, mo, d) + MJD_UNIX,
            sec: 0.0,
        }
        .add_seconds((h * 3600 + mi * 60) as f64 + s)
    }

    /// Parses `YYYY-MM-DDTHH:MM:SS[.fff][Z]` (a space may replace `T`);
    /// seconds may be omitted. Returns `None` if the text does not parse,
    /// has extra fields, or a field is out of range: a day past the end of
    /// its month, or seconds of 61 or more (60 is allowed, for a leap second).
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().trim_end_matches('Z');
        let (date, time) = s.split_once(['T', ' '])?;
        let mut dp = date.split('-');
        let (y, mo, d) = (
            dp.next()?.parse().ok()?,
            dp.next()?.parse().ok()?,
            dp.next()?.parse().ok()?,
        );
        let mut tp = time.split(':');
        let (h, mi) = (tp.next()?.parse().ok()?, tp.next()?.parse().ok()?);
        let sec: f64 = tp.next().unwrap_or("0").parse().ok()?;
        if dp.next().is_some() || tp.next().is_some() {
            return None;
        }
        let valid = (1..=12).contains(&mo)
            && (1..=days_in_month(y, mo)).contains(&d)
            && (0..24).contains(&h)
            && (0..60).contains(&mi)
            && (0.0..61.0).contains(&sec);
        valid.then(|| Self::from_civil(y, mo, d, h, mi, sec))
    }

    /// This instant plus `s` seconds (leap seconds inside the interval are
    /// not counted).
    #[must_use]
    pub fn add_seconds(self, s: f64) -> Self {
        let total = self.sec + s;
        let days = libm::floor(total / DAY_S);
        Self {
            mjd_day: self.mjd_day + days as i64,
            sec: total - days * DAY_S,
        }
    }

    /// Seconds from `other` to `self`.
    pub fn seconds_since(self, other: Utc) -> f64 {
        (self.mjd_day - other.mjd_day) as f64 * DAY_S + (self.sec - other.sec)
    }

    /// MJD (UTC) as one number.
    pub fn mjd(self) -> f64 {
        self.mjd_day as f64 + self.sec / DAY_S
    }

    /// TAI − UTC, seconds. Dates before 1972 get the 1972 value.
    pub fn tai_minus_utc(self) -> f64 {
        LEAP.iter()
            .rev()
            .find(|(m, _)| self.mjd_day >= *m)
            .map_or(LEAP[0].1, |l| l.1)
    }

    /// Days of TT since J2000.0.
    pub fn tt_days_since_j2000(self) -> f64 {
        (self.mjd_day as f64 - MJD_J2000) + (self.sec + self.tai_minus_utc() + 32.184) / DAY_S
    }

    /// Julian centuries of TT since J2000.0.
    pub fn tt_centuries(self) -> f64 {
        self.tt_days_since_j2000() / 36_525.0
    }

    /// Julian year (TT), e.g. 2026.88.
    pub fn julian_year_tt(self) -> f64 {
        2000.0 + self.tt_days_since_j2000() / 365.25
    }

    /// Calendar fields: year, month, day, hour, minute, second.
    pub fn civil(self) -> (i64, i64, i64, i64, i64, f64) {
        let (y, m, d) = civil_from_days(self.mjd_day - MJD_UNIX);
        let h = (self.sec / 3600.0) as i64;
        let mi = ((self.sec - h as f64 * 3600.0) / 60.0) as i64;
        (y, m, d, h, mi, self.sec - (h * 3600 + mi * 60) as f64)
    }
}

impl fmt::Display for Utc {
    /// ISO 8601 with milliseconds and no zone letter, as FITS `DATE-OBS` wants.
    ///
    /// The time is rounded to the millisecond before it is split into
    /// fields, so 59.9996 s carries into the next minute.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = libm::round(self.sec * 1000.0) / 1000.0;
        let t = Utc {
            mjd_day: self.mjd_day,
            sec: 0.0,
        }
        .add_seconds(ms);
        let (y, mo, d, h, mi, s) = t.civil();
        write!(f, "{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:06.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A date past the end of its month, or with extra fields, is refused
    /// instead of rolling into the next month.
    #[test]
    fn impossible_dates_are_refused() {
        for bad in [
            "2025-02-31T12:00:00",
            "2023-02-29T00:00:00",
            "1900-02-29T00:00:00",
            "2025-04-31T00:00:00",
            "2025-01-01-05T00:00:00",
            "2025-01-01T12:00:00:00",
        ] {
            assert!(Utc::parse(bad).is_none(), "{bad}");
        }
        for good in [
            "2024-02-29T00:00:00",
            "2000-02-29T00:00:00",
            "2025-12-31T23:59:60",
        ] {
            assert!(Utc::parse(good).is_some(), "{good}");
        }
    }

    #[test]
    fn known_dates() {
        assert_eq!(Utc::parse("2000-01-01T12:00:00").unwrap().mjd(), 51_544.5);
        assert_eq!(Utc::parse("1970-01-01T00:00:00Z").unwrap().mjd_day, 40_587);
        assert_eq!(Utc::parse("2026-11-17 00:00:00").unwrap().mjd_day, 61_361);
        assert!(Utc::parse("2026-13-01T00:00:00").is_none());
    }

    #[test]
    fn leap_seconds() {
        assert_eq!(
            Utc::parse("2016-12-31T23:59:59").unwrap().tai_minus_utc(),
            36.0
        );
        assert_eq!(
            Utc::parse("2017-01-01T00:00:00").unwrap().tai_minus_utc(),
            37.0
        );
        assert_eq!(
            Utc::parse("2026-11-17T00:00:00").unwrap().tai_minus_utc(),
            37.0
        );
    }

    /// J2000.0 is 2000-01-01T12:00 TT = 11:58:55.816 UTC.
    #[test]
    fn tt_offset() {
        let t = Utc::parse("2000-01-01T11:58:55.816").unwrap();
        assert!(t.tt_days_since_j2000().abs() * DAY_S < 1e-6);
    }

    #[test]
    fn display_round_trip_and_carry() {
        let t = Utc::parse("2026-11-17T03:04:05.123").unwrap();
        assert_eq!(t.to_string(), "2026-11-17T03:04:05.123");
        let t = Utc::parse("2026-12-31T23:59:59.9996").unwrap();
        assert_eq!(t.to_string(), "2027-01-01T00:00:00.000");
        assert!((t.add_seconds(10.5).seconds_since(t) - 10.5).abs() < 1e-9);
    }
}
