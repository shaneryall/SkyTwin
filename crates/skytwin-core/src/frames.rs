//! Celestial reference frames: SGP4's TEME to GCRS.
//!
//! Chain: TEME → true of date (equation of the equinoxes) → mean of date
//! (IAU 1980 nutation) → mean J2000 (IAU 1976 precession) → GCRS (frame
//! bias). This is the FK5-based reduction TEME is defined against.
//!
//! Sources:
//! - Vallado, Crawford, Hujsak & Kelso, "Revisiting Spacetrack Report #3",
//!   AIAA 2006-6753, appendix C (TEME definition and conversion).
//! - Lieske et al., "Expressions for the precession quantities based upon
//!   the IAU (1976) system of astronomical constants", A&A 58, 1 (1977).
//! - Seidelmann, "1980 IAU theory of nutation", Celestial Mechanics 27, 79
//!   (1982). The 63 largest terms are used, as tabulated by Meeus,
//!   *Astronomical Algorithms*, 2nd ed., table 22.A; the 43 dropped terms
//!   are each under 0.1 mas.
//! - Frame bias: IERS Conventions (2003), IERS Technical Note 32,
//!   section 5.4.
//!
//! Accuracy: about 0.1″ against the IAU 2006/2000A reduction, far below the
//! kilometre-level error of a TLE. Star positions never pass through here.
//!
//! Every `t: f64` argument is TT in Julian centuries since J2000.0.

use crate::time::{Utc, MJD_J2000};
use crate::vec3::{add, cross, mul_mm, mul_mtv, mul_mv, rot_x, rot_y, rot_z, M3, V3};

/// Arcseconds to radians.
const AS2R: f64 = core::f64::consts::PI / (180.0 * 3600.0);

/// Nutation terms: multipliers of D, M, M′, F, Ω, then Δψ and Δε
/// coefficients (constant, per century) in units of 0.0001″.
#[rustfmt::skip]
const NUT: [[f64; 9]; 63] = [
    [0.,0.,0.,0.,1., -171_996.,-174.2, 92025.,8.9],
    [-2.,0.,0.,2.,2., -13187.,-1.6, 5736.,-3.1],
    [0.,0.,0.,2.,2., -2274.,-0.2, 977.,-0.5],
    [0.,0.,0.,0.,2., 2062.,0.2, -895.,0.5],
    [0.,1.,0.,0.,0., 1426.,-3.4, 54.,-0.1],
    [0.,0.,1.,0.,0., 712.,0.1, -7.,0.],
    [-2.,1.,0.,2.,2., -517.,1.2, 224.,-0.6],
    [0.,0.,0.,2.,1., -386.,-0.4, 200.,0.],
    [0.,0.,1.,2.,2., -301.,0., 129.,-0.1],
    [-2.,-1.,0.,2.,2., 217.,-0.5, -95.,0.3],
    [-2.,0.,1.,0.,0., -158.,0., 0.,0.],
    [-2.,0.,0.,2.,1., 129.,0.1, -70.,0.],
    [0.,0.,-1.,2.,2., 123.,0., -53.,0.],
    [2.,0.,0.,0.,0., 63.,0., 0.,0.],
    [0.,0.,1.,0.,1., 63.,0.1, -33.,0.],
    [2.,0.,-1.,2.,2., -59.,0., 26.,0.],
    [0.,0.,-1.,0.,1., -58.,-0.1, 32.,0.],
    [0.,0.,1.,2.,1., -51.,0., 27.,0.],
    [-2.,0.,2.,0.,0., 48.,0., 0.,0.],
    [0.,0.,-2.,2.,1., 46.,0., -24.,0.],
    [2.,0.,0.,2.,2., -38.,0., 16.,0.],
    [0.,0.,2.,2.,2., -31.,0., 13.,0.],
    [0.,0.,2.,0.,0., 29.,0., 0.,0.],
    [-2.,0.,1.,2.,2., 29.,0., -12.,0.],
    [0.,0.,0.,2.,0., 26.,0., 0.,0.],
    [-2.,0.,0.,2.,0., -22.,0., 0.,0.],
    [0.,0.,-1.,2.,1., 21.,0., -10.,0.],
    [0.,2.,0.,0.,0., 17.,-0.1, 0.,0.],
    [2.,0.,-1.,0.,1., 16.,0., -8.,0.],
    [-2.,2.,0.,2.,2., -16.,0.1, 7.,0.],
    [0.,1.,0.,0.,1., -15.,0., 9.,0.],
    [-2.,0.,1.,0.,1., -13.,0., 7.,0.],
    [0.,-1.,0.,0.,1., -12.,0., 6.,0.],
    [0.,0.,2.,-2.,0., 11.,0., 0.,0.],
    [2.,0.,-1.,2.,1., -10.,0., 5.,0.],
    [2.,0.,1.,2.,2., -8.,0., 3.,0.],
    [0.,1.,0.,2.,2., 7.,0., -3.,0.],
    [-2.,1.,1.,0.,0., -7.,0., 0.,0.],
    [0.,-1.,0.,2.,2., -7.,0., 3.,0.],
    [2.,0.,0.,2.,1., -7.,0., 3.,0.],
    [2.,0.,1.,0.,0., 6.,0., 0.,0.],
    [-2.,0.,2.,2.,2., 6.,0., -3.,0.],
    [-2.,0.,1.,2.,1., 6.,0., -3.,0.],
    [2.,0.,-2.,0.,1., -6.,0., 3.,0.],
    [2.,0.,0.,0.,1., -6.,0., 3.,0.],
    [0.,-1.,1.,0.,0., 5.,0., 0.,0.],
    [-2.,-1.,0.,2.,1., -5.,0., 3.,0.],
    [-2.,0.,0.,0.,1., -5.,0., 3.,0.],
    [0.,0.,2.,2.,1., -5.,0., 3.,0.],
    [-2.,0.,2.,0.,1., 4.,0., 0.,0.],
    [-2.,1.,0.,2.,1., 4.,0., 0.,0.],
    [0.,0.,1.,-2.,0., 4.,0., 0.,0.],
    [-1.,0.,1.,0.,0., -4.,0., 0.,0.],
    [-2.,1.,0.,0.,0., -4.,0., 0.,0.],
    [1.,0.,0.,0.,0., -4.,0., 0.,0.],
    [0.,0.,1.,2.,0., 3.,0., 0.,0.],
    [0.,0.,-2.,2.,2., -3.,0., 0.,0.],
    [-1.,-1.,1.,0.,0., -3.,0., 0.,0.],
    [0.,1.,1.,0.,0., -3.,0., 0.,0.],
    [0.,-1.,1.,2.,2., -3.,0., 0.,0.],
    [2.,-1.,-1.,2.,2., -3.,0., 0.,0.],
    [0.,0.,3.,2.,2., -3.,0., 0.,0.],
    [2.,-1.,0.,2.,2., -3.,0., 0.,0.],
];

/// Mean obliquity of the ecliptic (IAU 1976), radians, `t` in TT centuries.
pub fn mean_obliquity(t: f64) -> f64 {
    (84_381.448 + t * (-46.8150 + t * (-0.00059 + t * 0.001_813))) * AS2R
}

/// Nutation in longitude Δψ and in obliquity Δε (IAU 1980), radians, `t`
/// in TT centuries.
pub fn nutation(t: f64) -> (f64, f64) {
    let deg = |a: f64| a.to_radians();
    let d = deg(297.850_36 + t * (445_267.111_480 + t * (-0.001_914_2 + t / 189_474.0)));
    let m = deg(357.527_72 + t * (35_999.050_340 + t * (-0.000_160_3 - t / 300_000.0)));
    let mp = deg(134.962_98 + t * (477_198.867_398 + t * (0.008_697_2 + t / 56_250.0)));
    let f = deg(93.271_91 + t * (483_202.017_538 + t * (-0.003_682_5 + t / 327_270.0)));
    let om = deg(125.044_52 + t * (-1_934.136_261 + t * (0.002_070_8 + t / 450_000.0)));
    let (mut dpsi, mut deps) = (0.0, 0.0);
    for r in &NUT {
        let arg = r[0] * d + r[1] * m + r[2] * mp + r[3] * f + r[4] * om;
        let (s, c) = libm::sincos(arg);
        dpsi += (r[5] + r[6] * t) * s;
        deps += (r[7] + r[8] * t) * c;
    }
    (dpsi * 1e-4 * AS2R, deps * 1e-4 * AS2R)
}

/// IAU 1976 precession matrix: mean J2000 → mean of date, `t` in TT
/// centuries.
pub fn precession(t: f64) -> M3 {
    let zeta = (2306.2181 + (0.30188 + 0.017_998 * t) * t) * t * AS2R;
    let z = (2306.2181 + (1.09468 + 0.018_203 * t) * t) * t * AS2R;
    let theta = (2004.3109 + (-0.42665 - 0.041_833 * t) * t) * t * AS2R;
    mul_mm(&rot_z(-z), &mul_mm(&rot_y(theta), &rot_z(-zeta)))
}

/// Frame bias: GCRS → mean J2000 (dynamical), first order.
fn frame_bias() -> M3 {
    let (da0, xi0, eta0) = (-0.0146 * AS2R, -0.016_617 * AS2R, -0.006_819 * AS2R);
    [[1.0, da0, -xi0], [-da0, 1.0, -eta0], [xi0, eta0, 1.0]]
}

/// Matrix taking TEME vectors to GCRS at TT centuries `t`.
///
/// Built as the transpose of GCRS → TEME = R3(eqe) · N · P · B, where B is
/// the frame bias, P precession, N nutation (mean → true of date) and eqe
/// the equation of the equinoxes, Δψ cos ε₀. Only the geometric part of the
/// equation of the equinoxes is applied, as TEME is defined.
pub fn teme_to_gcrs_matrix(t: f64) -> M3 {
    let eps0 = mean_obliquity(t);
    let (dpsi, deps) = nutation(t);
    let n = mul_mm(&rot_x(-(eps0 + deps)), &mul_mm(&rot_z(-dpsi), &rot_x(eps0)));
    let eqe = dpsi * libm::cos(eps0);
    let gcrs_to_teme = mul_mm(
        &rot_z(eqe),
        &mul_mm(&n, &mul_mm(&precession(t), &frame_bias())),
    );
    transpose(&gcrs_to_teme)
}

/// Transpose of a 3×3 matrix (the inverse of a rotation).
fn transpose(m: &M3) -> M3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Position or velocity from TEME to GCRS at TT centuries `t`.
///
/// The frame's own rotation rate, about 1e-11 rad/s, is ignored for
/// velocities.
pub fn teme_to_gcrs(v: V3, t: f64) -> V3 {
    mul_mv(&teme_to_gcrs_matrix(t), v)
}

/// Position or velocity from GCRS to TEME at TT centuries `t`; the inverse
/// of [`teme_to_gcrs`].
pub fn gcrs_to_teme(v: V3, t: f64) -> V3 {
    mul_mtv(&teme_to_gcrs_matrix(t), v)
}

/// Greenwich mean sidereal time (IAU 1982), radians, from UT1 ≈ UTC.
///
/// Source: Vallado, *Fundamentals of Astrodynamics and Applications*, 4th
/// ed., eq. 3-47. Taking UT1 = UTC (|DUT1| < 0.9 s) errs by under 0.004°.
pub fn gmst(t: Utc) -> f64 {
    let tu = (t.mjd() - MJD_J2000) / 36_525.0;
    let s = 67_310.548_41 + (876_600.0 * 3600.0 + 8_640_184.812_866) * tu + 0.093_104 * tu * tu
        - 6.2e-6 * tu * tu * tu;
    (s.rem_euclid(86_400.0) / 86_400.0) * 2.0 * core::f64::consts::PI
}

/// Earth's rotation rate, rad/s (IERS conventions; GMST rate is close to
/// this but slightly different, which is negligible here).
pub const OMEGA_EARTH: f64 = 7.292_115_146_706_979e-5;

/// Earth-fixed (pseudo-Earth-fixed: ITRS without polar motion) position (km)
/// and velocity (km/s) to GCRS.
///
/// The Earth-fixed velocity gains `ω⊕ × r` before rotating by [`gmst`].
pub fn ecef_to_gcrs(r: V3, v: V3, t: Utc) -> (V3, V3) {
    ecef_to_gcrs_ut1(r, v, t, 0.0)
}

/// As [`ecef_to_gcrs`], with the Earth's rotation angle taken at UT1 =
/// UTC + `dut1_s` (IERS Bulletin A gives DUT1; |DUT1| < 0.9 s). One second
/// of DUT1 moves a site by up to 0.46 km, about 3′ for a satellite 500 km
/// away, so frames of low satellites from the ground need it.
pub fn ecef_to_gcrs_ut1(r: V3, v: V3, t: Utc, dut1_s: f64) -> (V3, V3) {
    let rot = rot_z(gmst(t.add_seconds(dut1_s)));
    let w = [0.0, 0.0, OMEGA_EARTH];
    let r_teme = mul_mtv(&rot, r);
    let v_teme = mul_mtv(&rot, add(v, cross(w, r)));
    let c = t.tt_centuries();
    (teme_to_gcrs(r_teme, c), teme_to_gcrs(v_teme, c))
}

/// WGS84 ellipsoid: equatorial radius (km) and flattening.
pub const WGS84_A_KM: f64 = 6_378.137;
/// WGS84 flattening.
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;

/// Earth-fixed position (km) of a geodetic latitude and longitude (degrees,
/// WGS84, longitude east) and height above the ellipsoid (km).
///
/// Source: NIMA TR8350.2, *Department of Defense World Geodetic System
/// 1984*, 3rd ed. (2000), eq. 4-14.
pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, h_km: f64) -> V3 {
    let e2 = WGS84_F * (2.0 - WGS84_F);
    let (sp, cp) = libm::sincos(lat_deg.to_radians());
    let (sl, cl) = libm::sincos(lon_deg.to_radians());
    let n = WGS84_A_KM / libm::sqrt(1.0 - e2 * sp * sp);
    [
        (n + h_km) * cp * cl,
        (n + h_km) * cp * sl,
        (n * (1.0 - e2) + h_km) * sp,
    ]
}

/// Unit vector of the local vertical (the ellipsoid normal) at a geodetic
/// latitude and longitude, Earth-fixed.
pub fn geodetic_up(lat_deg: f64, lon_deg: f64) -> V3 {
    let (sp, cp) = libm::sincos(lat_deg.to_radians());
    let (sl, cl) = libm::sincos(lon_deg.to_radians());
    [cp * cl, cp * sl, sp]
}

/// Geodetic latitude and longitude (degrees, WGS84) and height (km) below
/// a GCRS position. Polar motion is ignored (under 20 m).
///
/// Source: Bowring's method, Bowring, Survey Review 23, 323 (1976). Height
/// uses the form `p cos φ + z sin φ − a √(1 − e² sin² φ)`, which stays well
/// conditioned at the poles.
pub fn subpoint(r_gcrs: V3, t: Utc) -> (f64, f64, f64) {
    let teme = gcrs_to_teme(r_gcrs, t.tt_centuries());
    let ecef = mul_mv(&rot_z(gmst(t)), teme);
    let (a, f) = (6_378.137, 1.0 / 298.257_223_563);
    let b = a * (1.0 - f);
    let e2 = f * (2.0 - f);
    let ep2 = e2 / (1.0 - e2);
    let p = libm::hypot(ecef[0], ecef[1]);
    let th = libm::atan2(ecef[2] * a, p * b);
    let (st, ct) = libm::sincos(th);
    let lat = libm::atan2(ecef[2] + ep2 * b * st * st * st, p - e2 * a * ct * ct * ct);
    let lon = libm::atan2(ecef[1], ecef[0]);
    let (sl, cl) = libm::sincos(lat);
    let h = p * cl + ecef[2] * sl - a * libm::sqrt(1.0 - e2 * sl * sl);
    (lat.to_degrees(), lon.to_degrees(), h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Meeus example 22.a: 1987-04-10 0h TD gives Δψ = −3.788″,
    /// Δε = +9.443″ and mean obliquity 23°26′27.407″.
    #[test]
    fn nutation_example() {
        let t = (2_446_895.5 - 2_451_545.0) / 36_525.0;
        let (dpsi, deps) = nutation(t);
        assert!((dpsi / AS2R + 3.788).abs() < 0.002, "{}", dpsi / AS2R);
        assert!((deps / AS2R - 9.443).abs() < 0.002, "{}", deps / AS2R);
        let e0 = mean_obliquity(t).to_degrees();
        assert!((e0 - (23.0 + 26.0 / 60.0 + 27.407 / 3600.0)).abs() * 3600.0 < 0.002);
    }

    /// Vallado example 3-5: 1992-08-20 12:14 UT1 gives GMST = 152.578788°.
    #[test]
    fn gmst_example() {
        let t = Utc::parse("1992-08-20T12:14:00").unwrap();
        let g = gmst(t).to_degrees();
        assert!((g - 152.578_788).abs() < 1e-4, "{g}");
    }

    /// Over the pole the height is above the polar radius; along the TEME x
    /// axis the longitude is −GMST.
    #[test]
    fn subpoint_of_a_point_over_the_pole_and_equator() {
        let t = Utc::parse("2026-11-17T00:00:00").unwrap();
        let (lat, _, h) = subpoint(teme_to_gcrs([0.0, 0.0, 7_000.0], t.tt_centuries()), t);
        assert!(
            (lat - 90.0).abs() < 1e-6 && (h - (7_000.0 - 6_356.752)).abs() < 0.01,
            "{lat} {h}"
        );
        let (lat, lon, _) = subpoint(teme_to_gcrs([7_000.0, 0.0, 0.0], t.tt_centuries()), t);
        let want = (-gmst(t).to_degrees() + 540.0).rem_euclid(360.0) - 180.0;
        assert!(
            lat.abs() < 1e-6 && (lon - want).abs() < 1e-6,
            "{lat} {lon} {want}"
        );
    }

    /// A real image header gives (−16.385°, −175.597°) for this Earth-fixed
    /// position.
    #[test]
    fn ecef_round_trip_through_gcrs() {
        let t = Utc::parse("2022-07-12T06:07:41.5").unwrap();
        let r = [-6_842.559_3, -526.882_6, -2_010.465_3];
        let (g, _) = ecef_to_gcrs(r, [0.0; 3], t);
        let (lat, lon, h) = subpoint(g, t);
        assert!(
            (lat + 16.385).abs() < 0.05 && (lon + 175.597).abs() < 0.05,
            "{lat} {lon}"
        );
        assert!((700.0..800.0).contains(&h), "{h}");
    }

    /// Geodetic → Earth-fixed → GCRS → subpoint returns the site, which
    /// moves east with the Earth at ω times its distance from the axis; the
    /// up vector at the equator is the position direction.
    #[test]
    fn geodetic_round_trip() {
        let t = crate::time::Utc::parse("2020-12-24T02:46:18").unwrap();
        for &(lat, lon, h) in &[
            (32.903, -105.53, 2.225),
            (-30.17, -70.81, 2.2),
            (0.0, 0.0, 0.0),
            (89.0, 45.0, 1.0),
        ] {
            let r = geodetic_to_ecef(lat, lon, h);
            let (g, v) = ecef_to_gcrs(r, [0.0; 3], t);
            let (lat2, lon2, h2) = subpoint(g, t);
            assert!(
                (lat2 - lat).abs() < 1e-7 && (h2 - h).abs() < 1e-6,
                "{lat} {lat2} {h} {h2}"
            );
            assert!(
                ((lon2 - lon + 540.0).rem_euclid(360.0) - 180.0).abs() < 1e-7,
                "{lon} {lon2}"
            );
            let want = OMEGA_EARTH * libm::hypot(r[0], r[1]);
            assert!(
                (crate::vec3::norm(v) - want).abs() < 1e-9,
                "{}",
                crate::vec3::norm(v)
            );
        }
        let u = geodetic_up(0.0, 30.0);
        let r = geodetic_to_ecef(0.0, 30.0, 0.0);
        let rn = crate::vec3::norm(r);
        assert!((0..3).all(|k| (u[k] - r[k] / rn).abs() < 1e-12));
    }

    /// One second of UT1 turns a site by 15.04″ of longitude.
    #[test]
    fn dut1_turns_the_earth() {
        let t = crate::time::Utc::parse("2026-11-19T00:00:00").unwrap();
        let r = geodetic_to_ecef(0.0, 0.0, 0.0);
        let (a, _) = ecef_to_gcrs_ut1(r, [0.0; 3], t, 0.0);
        let (b, _) = ecef_to_gcrs_ut1(r, [0.0; 3], t, 1.0);
        let ang = crate::vec3::angle(a, b).to_degrees() * 3600.0;
        assert!((ang - 15.041).abs() < 0.01, "{ang}");
    }

    /// The TEME → GCRS matrix is orthonormal.
    #[test]
    fn matrix_is_a_rotation() {
        let m = teme_to_gcrs_matrix(0.27);
        let mt = transpose(&m);
        let i = mul_mm(&m, &mt);
        for (r, row) in i.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                assert!((v - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }
}
