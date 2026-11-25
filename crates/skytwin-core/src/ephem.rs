//! Low-precision solar-system ephemerides in GCRS/ICRS axes.
//!
//! Enough for aberration (Earth's barycentric velocity to about 1e-4 of its
//! 30 km/s, i.e. about 0.01″ of aberration), illumination and stray-light
//! geometry. Not for astrometry of solar-system bodies.
//!
//! Sources:
//! - Standish, "Keplerian Elements for Approximate Positions of the Major
//!   Planets", JPL (table 1, valid 1800–2050), for the Earth–Moon barycentre
//!   and the four giant planets.
//! - Moon: the low-precision series of the *Astronomical Almanac*, section
//!   D (about 0.3° in position).
//! - Mass ratios: IAU 2009 system of astronomical constants (Luzum et al.,
//!   Celest. Mech. Dyn. Astr. 110, 293, 2011).

use crate::vec3::{add, scale, sub, V3};

/// Astronomical unit, km.
pub const AU_KM: f64 = 149_597_870.7;
/// Earth–Moon mass ratio.
const EARTH_MOON: f64 = 81.300_57;
/// Sun–Jupiter mass ratio (planet plus satellites).
const SUN_JUPITER: f64 = 1_047.348_6;
/// Sun–Saturn mass ratio (planet plus satellites).
const SUN_SATURN: f64 = 3_497.898;
/// Sun–Uranus mass ratio (planet plus satellites).
const SUN_URANUS: f64 = 22_902.98;
/// Sun–Neptune mass ratio (planet plus satellites).
const SUN_NEPTUNE: f64 = 19_412.26;
/// Obliquity of the ecliptic at J2000 (IAU 1976), degrees.
pub(crate) const EPS_J2000_DEG: f64 = 84_381.448 / 3600.0;

/// Elements and rates per century: a (au), e, I, L, ϖ, Ω (degrees).
type Elements = [[f64; 2]; 6];

#[rustfmt::skip]
const EMB: Elements = [
    [1.000_002_61, 0.000_005_62], [0.016_711_23, -0.000_043_92], [-0.000_015_31, -0.012_946_68],
    [100.464_571_66, 35_999.372_449_81], [102.937_681_93, 0.323_273_64], [0.0, 0.0],
];
#[rustfmt::skip]
const JUPITER: Elements = [
    [5.202_887_00, -0.000_116_07], [0.048_386_24, -0.000_132_53], [1.304_396_95, -0.001_837_14],
    [34.396_440_51, 3_034.746_127_75], [14.728_479_83, 0.212_526_68], [100.473_909_09, 0.204_691_06],
];
#[rustfmt::skip]
const URANUS: Elements = [
    [19.189_164_64, -0.001_961_76], [0.047_257_44, -0.000_043_97], [0.772_637_83, -0.002_429_39],
    [313.238_104_51, 428.482_027_85], [170.954_276_30, 0.408_052_81], [74.016_925_03, 0.042_405_89],
];
#[rustfmt::skip]
const NEPTUNE: Elements = [
    [30.069_922_76, 0.000_262_91], [0.008_590_48, 0.000_051_05], [1.770_043_47, 0.000_353_72],
    [-55.120_029_69, 218.459_453_25], [44.964_762_27, -0.322_414_64], [131.784_225_74, -0.005_086_64],
];
#[rustfmt::skip]
const SATURN: Elements = [
    [9.536_675_94, -0.001_250_60], [0.053_861_79, -0.000_509_91], [2.485_991_87, 0.001_936_09],
    [49.954_244_23, 1_222.493_622_01], [92.598_878_31, -0.418_972_16], [113.662_424_48, -0.288_677_94],
];

/// Rotates a vector from J2000 ecliptic to J2000 equatorial axes.
fn ecliptic_to_equatorial(v: V3) -> V3 {
    let (s, c) = libm::sincos(EPS_J2000_DEG.to_radians());
    [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]
}

/// Heliocentric position (km) and velocity (km/s) from Standish elements at
/// TT centuries `t` since J2000.0.
///
/// Kepler's equation is solved by ten Newton iterations; the mean motion
/// is the rate of the mean longitude L, in rad/s.
fn kepler(el: &Elements, t: f64) -> (V3, V3) {
    let at = |i: usize| el[i][0] + el[i][1] * t;
    let (a, e) = (at(0), at(1));
    let (inc, l, varpi, node) = (at(2).to_radians(), at(3), at(4), at(5));
    let w = (varpi - node).to_radians();
    let node = node.to_radians();
    let m = (l - varpi)
        .to_radians()
        .rem_euclid(2.0 * core::f64::consts::PI);
    let mut ea = m + e * libm::sin(m);
    for _ in 0..10 {
        ea -= (ea - e * libm::sin(ea) - m) / (1.0 - e * libm::cos(ea));
    }
    let (se, ce) = libm::sincos(ea);
    let b = libm::sqrt(1.0 - e * e);
    let n = el[3][1].to_radians() / (36_525.0 * 86_400.0);
    let edot = n / (1.0 - e * ce);
    let (xp, yp) = (a * (ce - e), a * b * se);
    let (vxp, vyp) = (-a * se * edot, a * b * ce * edot);
    let (sw, cw) = libm::sincos(w);
    let (so, co) = libm::sincos(node);
    let (si, ci) = libm::sincos(inc);
    let r1 = [cw * co - sw * so * ci, -sw * co - cw * so * ci];
    let r2 = [cw * so + sw * co * ci, -sw * so + cw * co * ci];
    let r3 = [sw * si, cw * si];
    let rot = |x: f64, y: f64| -> V3 {
        ecliptic_to_equatorial([
            r1[0] * x + r1[1] * y,
            r2[0] * x + r2[1] * y,
            r3[0] * x + r3[1] * y,
        ])
    };
    (scale(rot(xp, yp), AU_KM), scale(rot(vxp, vyp), AU_KM))
}

/// Geocentric Moon position, km, in J2000 equatorial axes, at TT centuries
/// `t` since J2000.0.
///
/// The series gives ecliptic-of-date coordinates; general precession in
/// longitude (1.3969713° per century) is removed to reach J2000 ecliptic
/// axes.
pub fn moon_geocentric(t: f64) -> V3 {
    let s = |a: f64, b: f64| libm::sin((a + b * t).to_radians());
    let c = |a: f64, b: f64| libm::cos((a + b * t).to_radians());
    let lon = 218.32 + 481_267.881 * t + 6.29 * s(135.0, 477_198.87) - 1.27 * s(259.3, -413_335.36)
        + 0.66 * s(235.7, 890_534.22)
        + 0.21 * s(269.9, 954_397.74)
        - 0.19 * s(357.5, 35_999.05)
        - 0.11 * s(186.5, 966_404.03);
    let lat = 5.13 * s(93.3, 483_202.02) + 0.28 * s(228.2, 960_400.89)
        - 0.28 * s(318.3, 6_003.15)
        - 0.17 * s(217.6, -407_332.21);
    let par = 0.9508
        + 0.0518 * c(135.0, 477_198.87)
        + 0.0095 * c(259.3, -413_335.36)
        + 0.0078 * c(235.7, 890_534.22)
        + 0.0028 * c(269.9, 954_397.74);
    let r = 6_378.14 / libm::sin(par.to_radians());
    let lon = (lon - 1.396_971_3 * t).to_radians();
    let lat = lat.to_radians();
    let (sl, cl) = libm::sincos(lon);
    let (sb, cb) = libm::sincos(lat);
    ecliptic_to_equatorial([r * cb * cl, r * cb * sl, r * sb])
}

/// Solar-system barycentric state of the Sun (km, km/s) at TT centuries
/// `t`, from the four giant planets (the terrestrial planets move it by
/// under 1,000 km).
pub fn sun_barycentric(t: f64) -> (V3, V3) {
    let planets = [
        (&JUPITER, SUN_JUPITER),
        (&SATURN, SUN_SATURN),
        (&URANUS, SUN_URANUS),
        (&NEPTUNE, SUN_NEPTUNE),
    ];
    let (mut r, mut v, mut mass) = ([0.0; 3], [0.0; 3], 1.0);
    for (el, ratio) in planets {
        let (rp, vp) = kepler(el, t);
        r = add(r, scale(rp, 1.0 / ratio));
        v = add(v, scale(vp, 1.0 / ratio));
        mass += 1.0 / ratio;
    }
    (scale(r, -1.0 / mass), scale(v, -1.0 / mass))
}

/// Barycentric position (km) and velocity (km/s) of the Earth's centre at
/// TT centuries `t` since J2000.0.
///
/// The Moon's velocity is a central difference over ±60 s.
pub fn earth_barycentric(t: f64) -> (V3, V3) {
    let (r_emb, v_emb) = kepler(&EMB, t);
    let (r_sun, v_sun) = sun_barycentric(t);
    let mu = 1.0 / (1.0 + EARTH_MOON);
    let dt = 60.0 / (36_525.0 * 86_400.0);
    let m0 = moon_geocentric(t);
    let v_moon = scale(
        sub(moon_geocentric(t + dt), moon_geocentric(t - dt)),
        1.0 / 120.0,
    );
    (
        sub(add(r_emb, r_sun), scale(m0, mu)),
        sub(add(v_emb, v_sun), scale(v_moon, mu)),
    )
}

/// Geocentric Sun position, km, at TT centuries `t` since J2000.0.
pub fn sun_geocentric(t: f64) -> V3 {
    let (r_earth, _) = earth_barycentric(t);
    let (r_sun, _) = sun_barycentric(t);
    sub(r_sun, r_earth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::norm;

    /// The Earth–barycentre distance stays within 0.03 au of 1 au: the
    /// orbit's eccentricity (±0.017) plus the Sun's wobble (±0.01).
    #[test]
    fn earth_speed_and_distance_are_plausible() {
        for i in 0..12 {
            let t = 0.26 + f64::from(i) / 1200.0;
            let (r, v) = earth_barycentric(t);
            assert!((norm(r) / AU_KM - 1.0).abs() < 0.03);
            assert!((norm(v) - 29.8).abs() < 0.6, "{}", norm(v));
        }
    }

    #[test]
    fn moon_distance_is_plausible() {
        let d = norm(moon_geocentric(0.27));
        assert!((356_000.0..407_000.0).contains(&d), "{d}");
    }
}
