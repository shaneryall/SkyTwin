//! Orbits of the observer and of targets: geocentre, TLE, state vector or
//! ephemeris table.
//!
//! All states are GCRS: position in km from the Earth's centre, velocity in
//! km/s.
//!
//! Sources:
//! - SGP4: the `sgp4` crate, a Rust port of Vallado et al., "Revisiting
//!   Spacetrack Report #3", AIAA 2006-6753; output converted from TEME with
//!   [`crate::frames`].
//! - J2 acceleration and constants: Vallado, *Fundamentals of Astrodynamics
//!   and Applications*, 4th ed., eq. 8-30; EGM-96 J2 = 1.08262668e-3,
//!   GM = 398600.4418 km³/s², R = 6378.137 km.
//! - Interpolation: cubic Hermite on position and velocity (e.g. Burden &
//!   Faires, *Numerical Analysis*, section 3.4).

use serde::{Deserialize, Serialize};

use crate::error::SimError;
use crate::frames::teme_to_gcrs;
use crate::time::Utc;
use crate::vec3::{add, dot, norm, scale, V3};

/// Earth's GM, km³/s².
pub const GM_EARTH: f64 = 398_600.441_8;
/// Earth's equatorial radius, km.
pub const R_EARTH: f64 = 6_378.137;
/// Earth's second zonal harmonic (EGM-96).
const J2: f64 = 1.082_626_68e-3;
/// Largest RK4 step, seconds.
const MAX_STEP_S: f64 = 10.0;

/// A position and velocity, GCRS.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    /// Position, km.
    pub position_km: V3,
    /// Velocity, km/s.
    pub velocity_km_s: V3,
}

/// A state vector at a given time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateVector {
    /// UTC time of the state, ISO 8601.
    pub epoch_utc: String,
    /// GCRS position, km.
    pub position_km: V3,
    /// GCRS velocity, km/s.
    pub velocity_km_s: V3,
    /// Include the J2 term (default true); false gives pure two-body motion.
    #[serde(default = "yes")]
    pub j2: bool,
    /// Frame of `position_km` and `velocity_km_s`: `"gcrs"` (default) or
    /// `"itrs"` (Earth-fixed, as many spacecraft headers give it).
    #[serde(default)]
    pub frame: StateFrame,
}

/// Reference frame of a given state vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateFrame {
    /// Geocentric celestial (inertial).
    #[default]
    Gcrs,
    /// Earth-fixed (polar motion ignored).
    Itrs,
}

/// Serde default for [`StateVector::j2`].
fn yes() -> bool {
    true
}

/// One sample of an ephemeris table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EphemerisSample {
    /// UTC time, ISO 8601.
    pub time_utc: String,
    /// GCRS position, km.
    pub position_km: V3,
    /// GCRS velocity, km/s.
    pub velocity_km_s: V3,
}

/// A ground site: a telescope on the Earth's surface, turning with it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Site {
    /// Geodetic latitude, degrees (WGS84).
    pub latitude_deg: f64,
    /// Longitude, degrees east (WGS84).
    pub longitude_deg: f64,
    /// Height above the WGS84 ellipsoid, metres.
    #[serde(default)]
    pub height_m: f64,
    /// UT1 − UTC, seconds (IERS Bulletin A); 0 when not given. Each second
    /// moves the site by up to 0.46 km.
    #[serde(default)]
    pub dut1_s: f64,
    /// Apply atmospheric refraction (default true).
    #[serde(default = "yes")]
    pub refraction: bool,
    /// Pressure at the site, hPa (default: the standard atmosphere at
    /// `height_m`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressure_hpa: Option<f64>,
    /// Air temperature, °C (default: the standard atmosphere at `height_m`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_c: Option<f64>,
    /// Extinction in the camera band, magnitudes per airmass (default 0).
    #[serde(default)]
    pub extinction_mag_per_airmass: f64,
}

impl Site {
    /// Earth-fixed position, km.
    pub fn ecef_km(&self) -> V3 {
        crate::frames::geodetic_to_ecef(
            self.latitude_deg,
            self.longitude_deg,
            self.height_m / 1000.0,
        )
    }
}

/// How a position over time is given (observer or target).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Orbit {
    /// At the Earth's centre (annual aberration only).
    #[default]
    Geocenter,
    /// Two-line elements, propagated with SGP4.
    Tle([String; 2]),
    /// A state vector, propagated numerically.
    StateVector(StateVector),
    /// A table of states, interpolated. Times must increase.
    Ephemeris(Vec<EphemerisSample>),
    /// A ground site, turning with the Earth (observers only).
    Site(Site),
}

/// Parses a UTC time string, naming the field `what` in the error.
fn parse_time(s: &str, what: &'static str) -> Result<Utc, SimError> {
    Utc::parse(s).ok_or(SimError::BadTime(what))
}

/// Epoch of a TLE as a UTC instant.
fn tle_epoch(e: &sgp4::Elements) -> Utc {
    use sgp4::chrono::{Datelike, Timelike};
    let d = e.datetime;
    Utc::from_civil(
        i64::from(d.year()),
        i64::from(d.month()),
        i64::from(d.day()),
        i64::from(d.hour()),
        i64::from(d.minute()),
        f64::from(d.second()) + f64::from(d.nanosecond()) * 1e-9,
    )
}

/// SGP4 constants with WGS72, the model TLEs are fitted with (Vallado et
/// al. 2006 recommend it; the `sgp4` crate's default is WGS84).
fn wgs72_constants(el: &sgp4::Elements) -> Result<sgp4::Constants, SimError> {
    let d = core::f64::consts::PI / 180.0;
    let orbit = sgp4::Orbit::from_kozai_elements(
        &sgp4::WGS72,
        el.inclination * d,
        el.right_ascension * d,
        el.eccentricity,
        el.argument_of_perigee * d,
        el.mean_anomaly * d,
        el.mean_motion * (core::f64::consts::PI / 720.0),
    )
    .map_err(|e| SimError::Orbit(format!("TLE: {e}")))?;
    sgp4::Constants::new(
        sgp4::WGS72,
        sgp4::iau_epoch_to_sidereal_time,
        el.epoch(),
        el.drag_term,
        orbit,
    )
    .map_err(|e| SimError::Orbit(format!("TLE: {e}")))
}

/// Two-body plus J2 acceleration at GCRS position `r` (km), km/s²
/// (Vallado eq. 8-30).
fn accel(r: V3, j2: bool) -> V3 {
    let rn = norm(r);
    let mut a = scale(r, -GM_EARTH / (rn * rn * rn));
    if j2 {
        let k = -1.5 * J2 * GM_EARTH * R_EARTH * R_EARTH / libm::pow(rn, 5.0);
        let z2 = 5.0 * r[2] * r[2] / (rn * rn);
        a = add(
            a,
            [
                k * r[0] * (1.0 - z2),
                k * r[1] * (1.0 - z2),
                k * r[2] * (3.0 - z2),
            ],
        );
    }
    a
}

/// Propagates `s` by `dt` seconds with fixed-step RK4, in equal steps of
/// at most 10 s; `j2` adds the J2 term to two-body gravity.
pub fn propagate(s: State, dt: f64, j2: bool) -> State {
    let n = libm::ceil(dt.abs() / MAX_STEP_S).max(1.0) as usize;
    let h = dt / n as f64;
    let (mut r, mut v) = (s.position_km, s.velocity_km_s);
    for _ in 0..n {
        let k1r = v;
        let k1v = accel(r, j2);
        let k2r = add(v, scale(k1v, h / 2.0));
        let k2v = accel(add(r, scale(k1r, h / 2.0)), j2);
        let k3r = add(v, scale(k2v, h / 2.0));
        let k3v = accel(add(r, scale(k2r, h / 2.0)), j2);
        let k4r = add(v, scale(k3v, h));
        let k4v = accel(add(r, scale(k3r, h)), j2);
        let sum = |a: V3, b: V3, c: V3, d: V3| add(add(a, scale(b, 2.0)), add(scale(c, 2.0), d));
        r = add(r, scale(sum(k1r, k2r, k3r, k4r), h / 6.0));
        v = add(v, scale(sum(k1v, k2v, k3v, k4v), h / 6.0));
    }
    State {
        position_km: r,
        velocity_km_s: v,
    }
}

/// Cubic Hermite interpolation between samples `a` at `ta` and `b` at `tb`,
/// evaluated at `t`; velocity is the derivative of the same cubic.
fn hermite(a: &EphemerisSample, b: &EphemerisSample, ta: Utc, tb: Utc, t: Utc) -> State {
    let h = tb.seconds_since(ta);
    let s = t.seconds_since(ta) / h;
    let (s2, s3) = (s * s, s * s * s);
    let (h00, h10, h01, h11) = (
        2.0 * s3 - 3.0 * s2 + 1.0,
        s3 - 2.0 * s2 + s,
        -2.0 * s3 + 3.0 * s2,
        s3 - s2,
    );
    let (d00, d10, d01, d11) = (
        6.0 * s2 - 6.0 * s,
        3.0 * s2 - 4.0 * s + 1.0,
        -6.0 * s2 + 6.0 * s,
        3.0 * s2 - 2.0 * s,
    );
    let mut st = State {
        position_km: [0.0; 3],
        velocity_km_s: [0.0; 3],
    };
    for i in 0..3 {
        let (p0, p1, m0, m1) = (
            a.position_km[i],
            b.position_km[i],
            a.velocity_km_s[i] * h,
            b.velocity_km_s[i] * h,
        );
        st.position_km[i] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
        st.velocity_km_s[i] = (d00 * p0 + d10 * m0 + d01 * p1 + d11 * m1) / h;
    }
    st
}

impl Orbit {
    /// The GCRS state at `t`.
    ///
    /// # Errors
    ///
    /// Returns [`SimError::Orbit`] if a TLE does not parse or SGP4 fails,
    /// if ephemeris times do not increase or `t` falls outside the table,
    /// or if a site's latitude is outside ±90° or its longitude is not
    /// finite, and [`SimError::BadTime`] if an epoch or sample time does not parse.
    pub fn state(&self, t: Utc) -> Result<State, SimError> {
        match self {
            Orbit::Geocenter => Ok(State {
                position_km: [0.0; 3],
                velocity_km_s: [0.0; 3],
            }),
            Orbit::Tle([l1, l2]) => {
                let el = sgp4::Elements::from_tle(None, l1.as_bytes(), l2.as_bytes())
                    .map_err(|e| SimError::Orbit(format!("TLE: {e}")))?;
                let c = wgs72_constants(&el)?;
                let minutes = t.seconds_since(tle_epoch(&el)) / 60.0;
                let p = c
                    .propagate(sgp4::MinutesSinceEpoch(minutes))
                    .map_err(|e| SimError::Orbit(format!("SGP4: {e}")))?;
                let tc = t.tt_centuries();
                Ok(State {
                    position_km: teme_to_gcrs(p.position, tc),
                    velocity_km_s: teme_to_gcrs(p.velocity, tc),
                })
            }
            Orbit::StateVector(sv) => {
                let t0 = parse_time(&sv.epoch_utc, "state_vector epoch_utc")?;
                let (r0, v0) = match sv.frame {
                    StateFrame::Gcrs => (sv.position_km, sv.velocity_km_s),
                    StateFrame::Itrs => {
                        crate::frames::ecef_to_gcrs(sv.position_km, sv.velocity_km_s, t0)
                    }
                };
                let s0 = State {
                    position_km: r0,
                    velocity_km_s: v0,
                };
                Ok(propagate(s0, t.seconds_since(t0), sv.j2))
            }
            Orbit::Site(site) => {
                if !(-90.0..=90.0).contains(&site.latitude_deg) || !site.longitude_deg.is_finite() {
                    return Err(SimError::Orbit(format!(
                        "site latitude {} outside ±90°",
                        site.latitude_deg
                    )));
                }
                let (r, v) =
                    crate::frames::ecef_to_gcrs_ut1(site.ecef_km(), [0.0; 3], t, site.dut1_s);
                Ok(State {
                    position_km: r,
                    velocity_km_s: v,
                })
            }
            Orbit::Ephemeris(samples) => {
                let times = samples
                    .iter()
                    .map(|s| parse_time(&s.time_utc, "ephemeris time_utc"))
                    .collect::<Result<Vec<_>, _>>()?;
                if times.windows(2).any(|w| w[1].seconds_since(w[0]) <= 0.0) {
                    return Err(SimError::Orbit("ephemeris times must increase".into()));
                }
                let i = times
                    .windows(2)
                    .position(|w| t.seconds_since(w[0]) >= 0.0 && w[1].seconds_since(t) >= 0.0)
                    .ok_or(SimError::Orbit("time outside the ephemeris table".into()))?;
                Ok(hermite(
                    &samples[i],
                    &samples[i + 1],
                    times[i],
                    times[i + 1],
                    t,
                ))
            }
        }
    }
}

/// Specific orbital energy of `s` under two-body gravity, km²/s² (for
/// tests and sanity checks).
pub fn energy(s: &State) -> f64 {
    0.5 * dot(s.velocity_km_s, s.velocity_km_s) - GM_EARTH / norm(s.position_km)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::sub;

    fn leo() -> State {
        let r = 7_000.0;
        let v = libm::sqrt(GM_EARTH / r);
        State {
            position_km: [r, 0.0, 0.0],
            velocity_km_s: [0.0, v * 0.6, v * 0.8],
        }
    }

    #[test]
    fn two_body_circular_orbit_closes() {
        let s = leo();
        let period = 2.0 * core::f64::consts::PI * libm::sqrt(libm::pow(7_000.0, 3.0) / GM_EARTH);
        let e = propagate(s, period, false);
        assert!(
            norm(sub(e.position_km, s.position_km)) < 1e-4,
            "{:?}",
            e.position_km
        );
        assert!((energy(&e) - energy(&s)).abs() < 1e-9);
    }

    /// A prograde inclined orbit's ascending node moves west at the rate of
    /// Vallado eq. 9-41, −3/2 n J2 (R/p)² cos i: about −4.3°/day here.
    #[test]
    fn j2_regresses_the_node_westward() {
        let s = leo();
        let e = propagate(s, 86_400.0, true);
        let h0 = crate::vec3::cross(s.position_km, s.velocity_km_s);
        let h1 = crate::vec3::cross(e.position_km, e.velocity_km_s);
        let node0 = libm::atan2(h0[0], -h0[1]);
        let node1 = libm::atan2(h1[0], -h1[1]);
        let drift = (node1 - node0).to_degrees();
        assert!((-4.6..-4.0).contains(&drift), "{drift}");
    }

    /// Vallado et al. (2006) verification case 00005 at t = 0:
    /// TEME r = (7022.46529266, −1400.08296755, 0.03995155) km.
    #[test]
    fn sgp4_matches_vallado_test_vector() {
        let obs = Orbit::Tle([
            "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753".into(),
            "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667".into(),
        ]);
        let t = Utc::parse("2000-06-27T18:50:19.733568").unwrap();
        let s = obs.state(t).unwrap();
        let teme = crate::frames::gcrs_to_teme(s.position_km, t.tt_centuries());
        let want = [7_022.465_292_66, -1_400.082_967_55, 0.039_951_55];
        assert!(norm(sub(teme, want)) < 1e-6, "{teme:?}");
    }

    #[test]
    fn hermite_matches_propagation() {
        let s = leo();
        let t0 = Utc::parse("2026-11-17T00:00:00").unwrap();
        let mk = |dt: f64| {
            let p = propagate(s, dt, true);
            EphemerisSample {
                time_utc: t0.add_seconds(dt).to_string(),
                position_km: p.position_km,
                velocity_km_s: p.velocity_km_s,
            }
        };
        let obs = Orbit::Ephemeris(vec![mk(0.0), mk(60.0), mk(120.0)]);
        let got = obs.state(t0.add_seconds(90.0)).unwrap();
        let want = propagate(s, 90.0, true);
        assert!(norm(sub(got.position_km, want.position_km)) < 0.01);
        assert!(obs.state(t0.add_seconds(121.0)).is_err());
    }
}
