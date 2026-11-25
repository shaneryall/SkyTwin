# References

Every algorithm in Skytwin cites its source here.

| Topic | Source | Used in |
| --- | --- | --- |
| xoshiro256++ RNG, SplitMix64 seeding | Blackman & Vigna, "Scrambled linear pseudorandom number generators", ACM TOMS 47(4), 2021 | `rng.rs` |
| Poisson deviates (product method) | Knuth, *The Art of Computer Programming* vol. 2, 3rd ed., section 3.4.1 | `rng.rs` |
| Normal deviates (polar method) | Marsaglia & Bray, "A convenient method for generating normal variables", SIAM Review 6, 260, 1964 | `rng.rs` |
| A short jittering path as one Gaussian of matched first and second moments (error fourth order in extent / σ) | Kendall & Stuart, *The Advanced Theory of Statistics* vol. 1, 4th ed., 1977, chapter 6 (Gram–Charlier expansion) | `path.rs` |
| Moving sources narrower than σ = 0.7 px integrated over each pixel (3×3-point Gauss–Legendre, renormalised); wider ones sampled at pixel centres | Project choice: 0.7 px is where centre sampling stops being accurate to well under 1% | `psf.rs` |
| FITS format, unsigned 16-bit via BZERO | FITS Standard 4.0, IAU FITS Working Group, 2018, section 5.3 | `fits.rs` |
| TAN projection | Calabretta & Greisen, "Representations of celestial coordinates in FITS", A&A 395, 1077, 2002, section 5.1.3 | `wcs.rs` |
| CD matrix | Greisen & Calabretta, "Representations of world coordinates in FITS", A&A 395, 1061, 2002 | `wcs.rs` |
| Catalogue and proper motion | UCAC4: Zacharias et al., AJ 145, 44, 2013; read and propagated by `fayde-catalog` | `catalog.rs` |
| Leap seconds | IERS Bulletin C (TAI − UTC = 37 s since 2017-01-01) | `time.rs` |
| Civil date ↔ day number | Hinnant, "chrono-Compatible Low-Level Date Algorithms", 2013 | `time.rs` |
| TEME definition and reduction | Vallado, Crawford, Hujsak & Kelso, "Revisiting Spacetrack Report #3", AIAA 2006-6753 | `frames.rs` |
| IAU 1976 precession | Lieske et al., A&A 58, 1, 1977 | `frames.rs` |
| IAU 1980 nutation (63 largest terms) | Seidelmann, Celest. Mech. 27, 79, 1982; table from Meeus, *Astronomical Algorithms*, 2nd ed., table 22.A | `frames.rs` |
| Frame bias | IERS Conventions 2003 (McCarthy & Petit), section 5.4 | `frames.rs` |
| SGP4 (via the `sgp4` crate, WGS72) | Vallado et al., AIAA 2006-6753 | `orbit.rs` |
| J2 acceleration | Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., eq. 8-30 | `orbit.rs` |
| Hermite interpolation | Burden & Faires, *Numerical Analysis*, section 3.4 | `orbit.rs` |
| Planet positions | Standish, "Keplerian Elements for Approximate Positions of the Major Planets", JPL, table 1 | `ephem.rs` |
| Moon position | *Astronomical Almanac*, section D, low-precision formulae | `ephem.rs` |
| Mass ratios | Luzum et al., "The IAU 2009 system of astronomical constants", Celest. Mech. Dyn. Astr. 110, 293, 2011 | `ephem.rs` |
| Relativistic aberration | Klioner, AJ 125, 1580, 2003, eq. 49; Kaplan et al., USNO Circular 179, 2005 | `geometry.rs` |
| Quaternions, Shepperd's method | Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*, 2014, sections 2.7 and 2.9.3 | `attitude.rs` |
| Light-time and aberration for near objects (barycentric light-time, so satellites show no annual aberration) | Kaplan et al., USNO Circular 179, 2005, section 7.2 | `target.rs` |
| Geodetic to Earth-fixed coordinates (WGS84) | NIMA TR8350.2, *Department of Defense World Geodetic System 1984*, 3rd ed., 2000, eq. 4-14 | `frames.rs` |
| Atmospheric refraction from the true altitude, pressure and temperature | Sæmundsson, *Sky and Telescope* 72, 70, 1986, as given in Meeus, *Astronomical Algorithms*, 2nd ed., 1998, eq. 16.4 | `atmosphere.rs` |
| Airmass | Kasten & Young, "Revised optical air mass tables and approximation formula", *Applied Optics* 28, 4735, 1989 | `atmosphere.rs` |
| Standard pressure and temperature by height | ISO 2533:1975, *Standard Atmosphere*; *U.S. Standard Atmosphere*, 1976 | `atmosphere.rs` |
| Earth shadow (conical, partial disc) | Montenbruck & Gill, *Satellite Orbits*, 2000, section 3.4.2 | `target.rs` |
| Lambertian sphere brightness | Hejduk, "Specular and diffuse components in spherical satellite photometric modeling", AMOS Conference, 2011 | `target.rs` |
| Sun radius | IAU 2015 Resolution B3 (nominal solar radius 695,700 km) | `target.rs` |
| JSON merge patch for profile overrides | IETF RFC 7386, "JSON Merge Patch", 2014 | `profile.rs` |
| Dark current against temperature | Janesick, *Scientific Charge-Coupled Devices*, SPIE, 2001, section 7.2 | `detector.rs` |
| Charge per µm from a minimum-ionising particle in silicon | Particle Data Group, *Review of Particle Physics*, "Passage of particles through matter", 2024 | `detector.rs` |
| Landau straggling (Moyal approximation) | Moyal, Phil. Mag. 46, 263, 1955 | `detector.rs` |
| Charge-transfer inefficiency (capture and exponential release) | Janesick, 2001, section 5.3; Massey et al., MNRAS 401, 371, 2010 | `detector.rs` |
| GMST (IAU 1982) | Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., eq. 3-47 | `frames.rs` |
| Geodetic latitude from Earth-fixed position | Bowring, Survey Review 23, 323, 1976 | `frames.rs` |
| Multi-Gaussian PSF | Bendinelli, ApJ 366, 599, 1991; Cappellari, MNRAS 333, 400, 2002 | `psf.rs` |
| Ornstein–Uhlenbeck wobble and its exact sampling | Uhlenbeck & Ornstein, Phys. Rev. 36, 823, 1930; Gillespie, Phys. Rev. E 54, 2084, 1996 | `motion.rs` |
| Moon illuminated fraction | Meeus, *Astronomical Algorithms*, 2nd ed., eq. 48.1 | `stray.rs` |
| Zodiacal light (the profile gives the table) | Leinert et al., "The 1997 reference of diffuse night sky brightness", A&AS 127, 1, 1998, table 17 | `stray.rs` |
| Random telegraph signal pixels | Hopkinson, Goiffon & Mohammadzadeh, IEEE Trans. Nucl. Sci. 55, 2197, 2008 | `detector.rs` |
| Moffat profile | Moffat, A&A 3, 455, 1969 | `psf.rs` |
| Field aberrations growing with radius | Born & Wolf, *Principles of Optics*, 7th ed., section 9.2 | `psf.rs` |
| Gauss–Legendre quadrature | Abramowitz & Stegun, *Handbook of Mathematical Functions*, 25.4.29 | `psf.rs` |
| Light deflection by the Sun | Kaplan et al., USNO Circular 179, 2005, eq. 7.69 | `geometry.rs` |
| Specular reflection (Blinn–Phong) | Blinn, "Models of light reflection for computer synthesized pictures", SIGGRAPH 1977; normalisation (k + 8)/8π from Akenine-Möller, Haines & Hoffman, *Real-Time Rendering*, 3rd ed., 2008, section 7.6 | `target.rs` |
| Cramér–Rao bounds for centroid and flux | King, PASP 95, 163, 1983; Mendez, Silva & Lobos, PASP 126, 798, 2014 | `psf.rs` |
| Pixel response non-uniformity | Janesick, *Scientific Charge-Coupled Devices*, SPIE, 2001, section 2.6 | `flat.rs` |
| Earth and Moon Bond albedos | NASA Earth and Moon fact sheets (NSSDCA): 0.306 and 0.11 | `target.rs` |
| Gaia DR3 column names and reference epoch | Gaia Collaboration, Vallenari et al., A&A 674, A1, 2023 | `catalog.rs` |
| AB magnitudes | Oke & Gunn, "Secondary standard stars for absolute spectrophotometry", ApJ 266, 713, 1983 | `photometry.rs` |
| Photon-counting synthetic photometry | Bessell & Murphy, "Spectrophotometric libraries, revised photonic passbands, and zero points for UBVRI, Hipparcos, and Tycho photometry", PASP 124, 140, 2012, appendix A | `photometry.rs` |
| Nominal solar temperature and radius | IAU 2015 Resolution B3 (Prša et al., AJ 152, 41, 2016) | `photometry.rs` |
| Temperature from B−V | Ballesteros, "New insights into black bodies", EPL 97, 34008, 2012 | `photometry.rs` |
| Temperature from Gaia BP−RP (dwarfs) | Mucciarelli, Bellazzini & Massari, "Exploiting the Gaia EDR3 photometry to derive stellar temperatures", A&A 653, A90, 2021 | `photometry.rs` |
