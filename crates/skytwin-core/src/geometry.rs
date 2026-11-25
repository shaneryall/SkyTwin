//! Exposure timing, observer motion, aberration and the two WCSs.
//!
//! The camera points along an *apparent* direction: what an observer moving
//! with the spacecraft sees. Catalogue positions are *astrometric*
//! (barycentric, at the epoch). Stellar aberration from the observer's
//! barycentric velocity (Earth's ~30 km/s plus the orbit's ~7.5 km/s)
//! shifts every star by up to ~25″ and changes the plate scale by up to
//! ~1e-4.
//!
//! So a frame has two WCSs:
//! - the **camera WCS** maps pixels to apparent directions (the physical
//!   pointing);
//! - the **sky WCS** maps pixels to astrometric directions. It is what a
//!   plate solver working from a catalogue should find, and it goes into the
//!   FITS header.
//!
//! Source: relativistic aberration, Klioner, "A practical relativistic model
//! for microarcsecond astrometry in space", AJ 125, 1580 (2003), eq. 49;
//! also Kaplan et al., USNO Circular 179 (2005), eq. 7.40.
//!
//! Also modelled when the scene has a time: gravitational light deflection
//! by the Sun (4 mas at 90° from the Sun, 1.75″ at the limb) and, for stars
//! given a parallax, the shift from the observer's barycentric place.
//! Astrometric → apparent applies parallax, then deflection, then aberration;
//! the inverse undoes them in reverse order.
//!
//! For an observer at a ground site the apparent direction is then
//! refracted by the atmosphere ([`crate::atmosphere`]): the camera WCS maps
//! pixels to *observed* directions, and the sky WCS undoes refraction too.

use serde::{Deserialize, Serialize};

use std::cell::RefCell;
use std::collections::HashMap;

use crate::atmosphere::Atmosphere;
use crate::attitude::{pointing_to_quaternion, PointingSpec};
use crate::camera::Camera;
use crate::ephem::earth_barycentric;
use crate::error::SimError;
use crate::orbit::{Orbit, State};
use crate::target::{Target, TargetView};
use crate::time::Utc;
use crate::vec3::{add, angle, dot, norm, radec_to_unit, scale, sub, unit, unit_to_radec, V3};
use crate::wcs::{Pointing, TanWcs};

/// Speed of light, km/s.
pub const C_KM_S: f64 = 299_792.458;

/// When the exposure happens.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeSpec {
    /// UTC at shutter open, ISO 8601 (`2026-11-17T03:04:05.123`).
    pub start_utc: String,
}

/// Exposure start, middle and end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    /// Shutter open.
    pub start: Utc,
    /// Mid-exposure (where positions and velocities are evaluated).
    pub mid: Utc,
    /// Shutter close.
    pub end: Utc,
}

impl Timing {
    /// Timing of an exposure of `exposure_s` seconds starting at `spec`.
    ///
    /// # Errors
    ///
    /// [`SimError::BadTime`] when `start_utc` is not a valid ISO 8601 UTC.
    pub fn new(spec: &TimeSpec, exposure_s: f64) -> Result<Self, SimError> {
        let start = Utc::parse(&spec.start_utc).ok_or(SimError::BadTime("scene.time.start_utc"))?;
        Ok(Self {
            start,
            mid: start.add_seconds(exposure_s / 2.0),
            end: start.add_seconds(exposure_s),
        })
    }
}

/// Apparent direction of a source at unit vector `u` for an observer moving
/// at `beta` = v/c (relativistic, Klioner 2003 eq. 49). Use `-beta` to go
/// back.
pub fn aberrate(u: V3, beta: V3) -> V3 {
    let inv_gamma = libm::sqrt(1.0 - dot(beta, beta));
    let ub = dot(u, beta);
    let num = add(
        scale(u, inv_gamma),
        scale(beta, 1.0 + ub / (1.0 + inv_gamma)),
    );
    unit(scale(num, 1.0 / (1.0 + ub)))
}

/// How the attitude moves during the exposure.
#[derive(Clone, Debug)]
pub enum Attitude {
    /// Inertially fixed (stare): stars stay put, targets move.
    Fixed,
    /// The boresight follows a target's apparent direction with the roll
    /// held: the target stays put, stars trail.
    Track(Box<Target>),
}

/// Everything about where the camera is and where it looks.
#[derive(Clone, Debug)]
pub struct Geometry {
    /// Exposure timing, if the scene has a time.
    pub timing: Option<Timing>,
    /// Observer GCRS state at mid-exposure, if the scene has a time.
    pub observer: Option<State>,
    /// Observer barycentric velocity / c at mid-exposure (zero without a time).
    pub beta: V3,
    /// Apparent boresight pointing at mid-exposure.
    pub pointing: Pointing,
    /// How the attitude moves.
    pub attitude: Attitude,
    /// Pixels ↔ apparent directions at mid-exposure.
    pub camera_wcs: TanWcs,
    /// Pixels ↔ astrometric directions at mid-exposure (the header WCS).
    pub sky_wcs: TanWcs,
    /// Largest difference between `sky_wcs` and the exact mapping on a grid
    /// over the image, pixels.
    pub sky_wcs_fit_px: f64,
    /// The atmosphere, for an observer at a ground site with a time.
    pub atmosphere: Option<Atmosphere>,
    orbit: Orbit,
    v_earth: V3,
    light: Option<Light>,
    camera: Camera,
    wcs_cache: RefCell<HashMap<u64, TanWcs>>,
}

/// Timing from the scene, if it has one.
///
/// # Errors
///
/// [`SimError::BadTime`] when the scene time does not parse.
pub fn timing(time: Option<&TimeSpec>, camera: &Camera) -> Result<Option<Timing>, SimError> {
    time.map(|t| Timing::new(t, camera.exposure_s)).transpose()
}

impl Geometry {
    /// Build the geometry. `None` when the scene has no pointing.
    ///
    /// Light deflection and parallax are modelled only when the scene has a
    /// time; without one `beta` is zero and the sky WCS equals the camera WCS.
    ///
    /// # Errors
    ///
    /// - [`SimError::BadTime`] when the time does not parse, an observer is
    ///   given without a time, or `track` is used without a time.
    /// - [`SimError::BadPointing`] when the pointing spec is invalid, `track`
    ///   is combined with an explicit direction, or names no target, or the
    ///   observer is a ground site without a pointing (airmass, extinction
    ///   and the horizon all need a boresight).
    /// - [`SimError::Orbit`] when a ground site's temperature, pressure or
    ///   extinction is out of range.
    /// - Any error from evaluating the observer orbit or the tracked target.
    pub fn new(
        pointing: Option<&PointingSpec>,
        time: Option<&TimeSpec>,
        observer: Option<&Orbit>,
        targets: &[Target],
        camera: &Camera,
    ) -> Result<Option<Self>, SimError> {
        let timing = timing(time, camera)?;
        if observer.is_some() && timing.is_none() {
            return Err(SimError::BadTime("scene.observer needs scene.time"));
        }
        if matches!(observer, Some(Orbit::Site(_))) && pointing.is_none() {
            return Err(SimError::BadPointing("a ground site needs scene.pointing"));
        }
        let Some(spec) = pointing else {
            return Ok(None);
        };
        let orbit = observer.cloned().unwrap_or_default();
        let (obs_mid, v_earth, r_earth) = match timing {
            Some(tm) => {
                let (r_earth, v_earth) = earth_barycentric(tm.mid.tt_centuries());
                (Some(orbit.state(tm.mid)?), v_earth, r_earth)
            }
            None => (None, [0.0; 3], [0.0; 3]),
        };
        let light = match (timing, obs_mid) {
            (Some(tm), Some(st)) => {
                let sun = crate::ephem::sun_geocentric(tm.mid.tt_centuries());
                let so = sub(st.position_km, sun);
                Some(Light {
                    sun_to_obs: unit(so),
                    sun_dist_km: norm(so),
                    obs_bary_au: scale(add(r_earth, st.position_km), 1.0 / crate::ephem::AU_KM),
                })
            }
            _ => None,
        };
        let beta = obs_mid.map_or([0.0; 3], |st| observer_beta(v_earth, &st));
        let atmosphere = site_atmosphere(&orbit, timing)?;
        let (pointing, attitude) = match &spec.track {
            None => (spec.resolve()?, Attitude::Fixed),
            Some(name) => {
                if spec.ra_deg.is_some() || spec.dec_deg.is_some() || spec.quaternion.is_some() {
                    return Err(SimError::BadPointing("track takes roll_deg only"));
                }
                let (tm, obs) = timing
                    .zip(obs_mid)
                    .ok_or(SimError::BadTime("pointing.track needs scene.time"))?;
                let target = targets
                    .iter()
                    .find(|t| &t.name == name)
                    .ok_or(SimError::BadPointing("pointing.track names no target"))?;
                let view = target.view(tm.mid, &obs, beta, camera, atmosphere.as_ref())?;
                let p = pointing_along(
                    observed(atmosphere.as_ref(), view.apparent),
                    spec.roll_deg.unwrap_or(0.0),
                );
                (p, Attitude::Track(Box::new(target.clone())))
            }
        };
        let camera_wcs = camera_wcs_for(&pointing, camera);
        let (sky_wcs, sky_wcs_fit_px) = fit_sky_wcs(
            &camera_wcs,
            beta,
            light,
            atmosphere,
            camera.width,
            camera.height,
        );
        Ok(Some(Self {
            timing,
            observer: obs_mid,
            beta,
            pointing,
            attitude,
            camera_wcs,
            sky_wcs,
            sky_wcs_fit_px,
            atmosphere,
            orbit,
            v_earth,
            light,
            camera: camera.clone(),
            wcs_cache: RefCell::new(HashMap::new()),
        }))
    }

    /// Whether the attitude moves during the exposure.
    pub fn is_tracking(&self) -> bool {
        matches!(self.attitude, Attitude::Track(_))
    }

    /// Time `s` seconds after shutter open (needs a timing).
    fn at(&self, s: f64) -> Result<Utc, SimError> {
        Ok(self
            .timing
            .ok_or(SimError::BadTime("moving sources need scene.time"))?
            .start
            .add_seconds(s))
    }

    /// Observer GCRS state and barycentric v/c, `s` seconds after shutter
    /// open.
    ///
    /// The Earth's barycentric velocity is held at its mid-exposure value.
    ///
    /// # Errors
    ///
    /// [`SimError::BadTime`] when the scene has no time, or any error from
    /// evaluating the observer orbit.
    pub fn observer_at(&self, s: f64) -> Result<(State, V3), SimError> {
        let st = self.orbit.state(self.at(s)?)?;
        Ok((st, observer_beta(self.v_earth, &st)))
    }

    /// Camera WCS (pixels ↔ apparent directions) `s` seconds after shutter
    /// open. Cached per time, since every star asks for the same times.
    ///
    /// # Errors
    ///
    /// When tracking: [`SimError::BadTime`] without a scene time, or any
    /// error from evaluating the observer or the tracked target.
    pub fn camera_wcs_at(&self, s: f64) -> Result<TanWcs, SimError> {
        let Attitude::Track(target) = &self.attitude else {
            return Ok(self.camera_wcs);
        };
        if let Some(w) = self.wcs_cache.borrow().get(&s.to_bits()) {
            return Ok(*w);
        }
        let (obs, beta) = self.observer_at(s)?;
        let t = self.at(s)?;
        let view = target.view(t, &obs, beta, &self.camera, self.atmosphere_at(t).as_ref())?;
        let w = camera_wcs_for(
            &pointing_along(self.observed(view.apparent), self.pointing.roll_deg),
            &self.camera,
        );
        self.wcs_cache.borrow_mut().insert(s.to_bits(), w);
        Ok(w)
    }

    /// Apparent direction of a star at an astrometric direction `u`, with
    /// parallax `parallax_mas`: parallax, then light deflection by the Sun,
    /// then aberration.
    pub fn apparent(&self, u: V3, parallax_mas: f64) -> V3 {
        self.observed(apparent_dir(
            u,
            parallax_mas,
            self.light.as_ref(),
            self.beta,
        ))
    }

    /// An apparent direction as observed through the atmosphere (unchanged
    /// without one).
    pub fn observed(&self, v: V3) -> V3 {
        observed(self.atmosphere.as_ref(), v)
    }

    /// The atmosphere at time `t` (the site's vertical turns with the
    /// Earth), for a ground site. Refraction of drawn positions uses the
    /// mid-exposure one; the horizon uses this.
    pub fn atmosphere_at(&self, t: Utc) -> Option<Atmosphere> {
        match &self.orbit {
            Orbit::Site(site) => Some(Atmosphere::for_site(site, t)),
            _ => None,
        }
    }

    /// Whether an observed direction is below a ground site's horizon.
    pub fn below_horizon(&self, v: V3) -> bool {
        self.atmosphere.as_ref().is_some_and(|a| a.hidden(v))
    }

    /// An observed direction back above the atmosphere.
    pub fn unobserved(&self, v: V3) -> V3 {
        self.atmosphere.as_ref().map_or(v, |a| a.unrefract(v))
    }

    /// Airmass at the boresight (ground sites only).
    pub fn airmass(&self) -> Option<f64> {
        let b = radec_to_unit(self.pointing.ra_deg, self.pointing.dec_deg);
        self.atmosphere.as_ref().map(|a| a.airmass(b))
    }

    /// Fraction of light that reaches the camera through the atmosphere at
    /// the boresight (1 without one). Every source in a frame gets the same
    /// factor: across a 2° field at 30° altitude the airmass changes by
    /// about 0.1, 0.02 mag at 0.2 mag per airmass.
    pub fn transmission(&self) -> f64 {
        let b = radec_to_unit(self.pointing.ra_deg, self.pointing.dec_deg);
        self.atmosphere.as_ref().map_or(1.0, |a| a.transmission(b))
    }

    /// Pixel position of an astrometric sky position at mid-exposure; `None`
    /// when it lies behind the tangent plane.
    pub fn sky_to_pixel(&self, ra_deg: f64, dec_deg: f64) -> Option<(f64, f64)> {
        self.sky_to_pixel_par(ra_deg, dec_deg, 0.0)
    }

    /// As [`Geometry::sky_to_pixel`], for a star with a parallax in mas.
    pub fn sky_to_pixel_par(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        parallax_mas: f64,
    ) -> Option<(f64, f64)> {
        let (ra, dec) = self.apparent_radec(ra_deg, dec_deg, parallax_mas);
        self.camera_wcs.world_to_pixel(ra, dec)
    }

    /// Pixel position of an astrometric sky position `s` seconds after
    /// shutter open (stars use the mid-exposure aberration; its change over
    /// an exposure is under 0.1″ in low Earth orbit).
    ///
    /// # Errors
    ///
    /// As [`Geometry::camera_wcs_at`].
    pub fn sky_to_pixel_at(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        s: f64,
    ) -> Result<Option<(f64, f64)>, SimError> {
        self.sky_to_pixel_at_par(ra_deg, dec_deg, 0.0, s)
    }

    /// As [`Geometry::sky_to_pixel_at`], for a star with a parallax in mas.
    ///
    /// # Errors
    ///
    /// As [`Geometry::camera_wcs_at`].
    pub fn sky_to_pixel_at_par(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        parallax_mas: f64,
        s: f64,
    ) -> Result<Option<(f64, f64)>, SimError> {
        let (ra, dec) = self.apparent_radec(ra_deg, dec_deg, parallax_mas);
        Ok(self.camera_wcs_at(s)?.world_to_pixel(ra, dec))
    }

    /// Apparent (RA, Dec) in degrees of an astrometric (RA, Dec) in degrees.
    fn apparent_radec(&self, ra_deg: f64, dec_deg: f64, parallax_mas: f64) -> (f64, f64) {
        unit_to_radec(self.apparent(radec_to_unit(ra_deg, dec_deg), parallax_mas))
    }

    /// Astrometric sky position of a pixel at mid-exposure (exact, not
    /// through `sky_wcs`).
    pub fn pixel_to_sky(&self, x: f64, y: f64) -> (f64, f64) {
        let (ra, dec) = self.camera_wcs.pixel_to_world(x, y);
        unit_to_radec(astrometric_dir(
            self.unobserved(radec_to_unit(ra, dec)),
            self.light.as_ref(),
            self.beta,
        ))
    }

    /// A target `s` seconds after shutter open: its view and pixel position.
    ///
    /// # Errors
    ///
    /// [`SimError::BadTime`] without a scene time, or any error from
    /// evaluating the observer or the target.
    pub fn target_at(
        &self,
        target: &Target,
        s: f64,
    ) -> Result<(TargetView, Option<(f64, f64)>), SimError> {
        let (obs, beta) = self.observer_at(s)?;
        let t = self.at(s)?;
        let mut view = target.view(t, &obs, beta, &self.camera, self.atmosphere_at(t).as_ref())?;
        if self.atmosphere.is_some() {
            view.apparent = self.observed(view.apparent);
            let tr = self.transmission();
            view.e_per_s *= tr;
            view.mag = view.mag.map(|m| m - 2.5 * libm::log10(tr));
        }
        let (ra, dec) = unit_to_radec(view.apparent);
        Ok((view, self.camera_wcs_at(s)?.world_to_pixel(ra, dec)))
    }

    /// Shift of the boresight by aberration, arcseconds.
    pub fn aberration_arcsec(&self) -> f64 {
        let b = radec_to_unit(self.pointing.ra_deg, self.pointing.dec_deg);
        angle(b, aberrate(b, scale(self.beta, -1.0))).to_degrees() * 3600.0
    }

    /// Camera → GCRS quaternion of the apparent pointing at mid-exposure.
    pub fn quaternion(&self) -> [f64; 4] {
        pointing_to_quaternion(&self.pointing)
    }
}

/// The atmosphere at mid-exposure for an observer at a ground site; `None`
/// for any other observer or without a time.
///
/// # Errors
///
/// [`SimError::Orbit`] when the site's temperature, pressure or extinction
/// is out of range (refraction would be NaN or meaningless).
fn site_atmosphere(orbit: &Orbit, timing: Option<Timing>) -> Result<Option<Atmosphere>, SimError> {
    let (Orbit::Site(site), Some(tm)) = (orbit, timing) else {
        return Ok(None);
    };
    let a = Atmosphere::for_site(site, tm.mid);
    if !(-100.0..80.0).contains(&a.temperature_c)
        || !(0.0..1200.0).contains(&a.pressure_hpa)
        || !(0.0..5.0).contains(&a.extinction_mag_per_airmass)
    {
        return Err(SimError::Orbit(format!(
            "site temperature {} °C, pressure {} hPa or extinction {} mag/airmass out of range",
            a.temperature_c, a.pressure_hpa, a.extinction_mag_per_airmass
        )));
    }
    Ok(Some(a))
}

/// Observer barycentric velocity / c: the Earth's barycentric velocity plus
/// the observer's GCRS velocity.
fn observer_beta(v_earth: V3, st: &State) -> V3 {
    scale(add(v_earth, st.velocity_km_s), 1.0 / C_KM_S)
}

/// Boresight pointing along the unit vector `u` with roll `roll_deg`.
fn pointing_along(u: V3, roll_deg: f64) -> Pointing {
    let (ra_deg, dec_deg) = unit_to_radec(u);
    Pointing {
        ra_deg,
        dec_deg,
        roll_deg,
    }
}

/// Camera WCS for a pointing: the camera's size, plate scale, parity,
/// optical axis and distortion.
fn camera_wcs_for(pointing: &Pointing, camera: &Camera) -> TanWcs {
    TanWcs::from_pointing_parity(
        pointing,
        camera.width,
        camera.height,
        camera.pixel_scale_arcsec,
        camera.parity,
    )
    .with_crpix(camera.optical_axis_px)
    .with_distortion(camera.distortion)
}

/// Sun's GM / c², km.
const SUN_GRAV_RADIUS_KM: f64 = 1.476_625;

/// Observer–Sun geometry for light deflection and parallax.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// Unit vector from the Sun to the observer.
    pub sun_to_obs: V3,
    /// Sun–observer distance, km.
    pub sun_dist_km: f64,
    /// Observer's barycentric position, au.
    pub obs_bary_au: V3,
}

/// Deflection of light from a distant source at direction `p` by the Sun
/// (USNO Circular 179, Kaplan et al. 2005, eq. 7.69, with the source at
/// infinity): pushes `p` away from the Sun. `sign` −1 undoes it (first
/// order; the error is below 1e-5″). A source directly behind the Sun's
/// centre is returned unchanged.
fn deflect(p: V3, l: &Light, sign: f64) -> V3 {
    let e = l.sun_to_obs;
    let pe = dot(p, e);
    if pe <= -1.0 + 1e-9 {
        return p;
    }
    let k = sign * 2.0 * SUN_GRAV_RADIUS_KM / l.sun_dist_km / (1.0 + pe);
    unit(add(p, scale(sub(e, scale(p, pe)), k)))
}

/// Astrometric → apparent: parallax, deflection, aberration.
///
/// With a parallax ϖ (mas) the star is placed at 1/ϖ au and seen from the
/// observer's barycentric place. Parallax and deflection need `light`.
pub fn apparent_dir(u: V3, parallax_mas: f64, light: Option<&Light>, beta: V3) -> V3 {
    let mut p = u;
    if let Some(l) = light {
        if parallax_mas > 0.0 {
            let d_au = 1.0 / (parallax_mas / 1000.0 / 3600.0).to_radians();
            p = unit(sub(scale(u, d_au), l.obs_bary_au));
        }
        p = deflect(p, l, 1.0);
    }
    aberrate(p, beta)
}

/// Refracted through the atmosphere, if there is one.
fn observed(atm: Option<&Atmosphere>, v: V3) -> V3 {
    atm.map_or(v, |a| a.refract(v))
}

/// Apparent → astrometric for a distant source (no parallax).
pub fn astrometric_dir(v: V3, light: Option<&Light>, beta: V3) -> V3 {
    let p = aberrate(v, scale(beta, -1.0));
    match light {
        Some(l) => deflect(p, l, -1.0),
        None => p,
    }
}

/// Fit a TAN WCS to the astrometric directions of a 7×7 pixel grid.
/// Returns the WCS and its largest residual in pixels.
///
/// The tangent point is the astrometric direction of `crpix0`, so the fit
/// has no offset term: a linear least-squares solve for
/// `[xi, eta] = CD · [dx, dy]`. The CD maps ideal (undistorted) offsets and
/// the camera's distortion is kept as is. With an atmosphere the observed
/// directions are first taken back above it.
fn fit_sky_wcs(
    cam: &TanWcs,
    beta: V3,
    light: Option<Light>,
    atm: Option<Atmosphere>,
    w: usize,
    h: usize,
) -> (TanWcs, f64) {
    if dot(beta, beta) == 0.0 && light.is_none() && atm.is_none() {
        return (*cam, 0.0);
    }
    let astro = |x: f64, y: f64| {
        let (ra, dec) = cam.pixel_to_world(x, y);
        let v = radec_to_unit(ra, dec);
        unit_to_radec(astrometric_dir(
            atm.as_ref().map_or(v, |a| a.unrefract(v)),
            light.as_ref(),
            beta,
        ))
    };
    let (cx, cy) = (cam.crpix0[0], cam.crpix0[1]);
    let (ra0, dec0) = astro(cx, cy);
    let mut fit = TanWcs {
        crval_deg: [ra0, dec0],
        crpix0: cam.crpix0,
        cd: [[0.0; 2]; 2],
        distortion: cam.distortion,
    };
    let n = 7;
    let grid: Vec<(f64, f64, f64, f64)> = (0..n * n)
        .map(|k| {
            let x = f64::from(k % n) / f64::from(n - 1) * (w as f64 - 1.0);
            let y = f64::from(k / n) / f64::from(n - 1) * (h as f64 - 1.0);
            let (ra, dec) = astro(x, y);
            let (xi, eta) = fit.standard_coords(ra, dec).unwrap_or((0.0, 0.0));
            let (ux, uy) = cam.ideal_offset(x, y);
            (ux, uy, xi, eta)
        })
        .collect();
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    let (mut bx, mut by) = ([0.0; 2], [0.0; 2]);
    for &(dx, dy, xi, eta) in &grid {
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
        bx[0] += dx * xi;
        bx[1] += dy * xi;
        by[0] += dx * eta;
        by[1] += dy * eta;
    }
    let det = sxx * syy - sxy * sxy;
    let solve = |b: [f64; 2]| {
        [
            (syy * b[0] - sxy * b[1]) / det,
            (sxx * b[1] - sxy * b[0]) / det,
        ]
    };
    fit.cd = [solve(bx), solve(by)];
    let worst = grid
        .iter()
        .map(|&(ux, uy, _, _)| {
            let (dx, dy) = cam.distortion.map_or((ux, uy), |d| d.distort(ux, uy));
            let (ra, dec) = astro(dx + cx, dy + cy);
            let (px, py) = fit.world_to_pixel(ra, dec).unwrap_or((f64::NAN, f64::NAN));
            libm::hypot(px - (dx + cx), py - (dy + cy))
        })
        .fold(0.0, f64::max);
    (fit, worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Perpendicular to the velocity the shift is β (20.6″ for 1e-4).
    #[test]
    fn aberration_round_trip_and_size() {
        let u = unit([0.3, -0.5, 0.8]);
        let beta = [0.0, 1e-4, 0.0];
        let back = aberrate(aberrate(u, beta), scale(beta, -1.0));
        assert!(angle(u, back) < 1e-15);
        let perp = [1.0, 0.0, 0.0];
        let shift = angle(perp, aberrate(perp, beta)).to_degrees() * 3600.0;
        assert!((shift - 20.626).abs() < 0.01, "{shift}");
    }

    /// A ground telescope at 2,225 m looking about 60° up (the geometry of a
    /// real tracking frame: header altitude 60.07°). A star near the
    /// boresight moves by the refraction at its altitude; the header WCS
    /// still maps the image to catalogue positions and the exact mapping
    /// inverts. An impossible temperature is refused rather than giving
    /// NaN, and a ground site needs a pointing. That pointing is above the
    /// horizon and its opposite is not. Extinction: 0.2 mag per airmass at
    /// airmass 1.15.
    #[test]
    fn ground_site_refraction_lifts_stars_towards_the_zenith() {
        let cam = Camera {
            width: 4096,
            height: 4096,
            pixel_scale_arcsec: 1.55,
            ..Camera::default()
        };
        let spec = PointingSpec::from(Pointing {
            ra_deg: 43.3529,
            dec_deg: 61.2228,
            roll_deg: 0.0,
        });
        let time = TimeSpec {
            start_utc: "2020-12-24T02:46:18.293".into(),
        };
        let site = |refraction| {
            Orbit::Site(crate::orbit::Site {
                latitude_deg: 32.903_056,
                longitude_deg: -105.529_556,
                height_m: 2225.0,
                dut1_s: -0.17,
                refraction,
                pressure_hpa: None,
                temperature_c: None,
                extinction_mag_per_airmass: 0.2,
            })
        };
        let geom = |r| {
            Geometry::new(Some(&spec), Some(&time), Some(&site(r)), &[], &cam)
                .unwrap()
                .unwrap()
        };
        let (g, g0) = (geom(true), geom(false));
        let atm = g.atmosphere.unwrap();
        let b = radec_to_unit(g.pointing.ra_deg, g.pointing.dec_deg);
        let alt = atm.altitude_deg(b);
        assert!((alt - 60.07).abs() < 0.2, "{alt}");
        let (ra, dec) = (43.40, 61.25);
        let (x1, y1) = g.sky_to_pixel(ra, dec).unwrap();
        let (x0, y0) = g0.sky_to_pixel(ra, dec).unwrap();
        let shift = libm::hypot(x1 - x0, y1 - y0) * cam.pixel_scale_arcsec;
        let u = g0.apparent(radec_to_unit(ra, dec), 0.0);
        let want = atm.refraction_deg(atm.altitude_deg(u)) * 3600.0;
        assert!((20.0..40.0).contains(&want), "{want}");
        assert!((shift - want).abs() < 0.01 * want, "{shift} {want}");
        assert!(g.sky_wcs_fit_px < 0.05, "{}", g.sky_wcs_fit_px);
        let (ra2, dec2) = g.pixel_to_sky(x1, y1);
        assert!(((ra2 - ra) * 3600.0).abs() < 1e-6 && ((dec2 - dec) * 3600.0).abs() < 1e-6);
        let (xs, ys) = g.sky_wcs.world_to_pixel(ra, dec).unwrap();
        assert!(libm::hypot(xs - x1, ys - y1) < 0.05);
        let mut bad = site(true);
        if let Orbit::Site(s) = &mut bad {
            s.temperature_c = Some(-273.0);
        }
        assert!(Geometry::new(Some(&spec), Some(&time), Some(&bad), &[], &cam).is_err());
        assert!(Geometry::new(None, Some(&time), Some(&site(true)), &[], &cam).is_err());
        assert!(!g.below_horizon(g.apparent(b, 0.0)));
        assert!(g.below_horizon(scale(atm.up, -1.0)));
        let x = g.airmass().unwrap();
        assert!((x - 1.153).abs() < 0.01, "{x}");
        assert!((g.transmission() - libm::pow(10.0, -0.08 * x)).abs() < 1e-12);
        assert_eq!(g0.transmission(), g.transmission());
    }

    #[test]
    fn sky_wcs_fits_and_differs_from_camera() {
        let cam = Camera::default();
        let spec = PointingSpec::from(Pointing {
            ra_deg: 172.5,
            dec_deg: 2.0,
            roll_deg: 30.0,
        });
        let time = TimeSpec {
            start_utc: "2026-11-17T03:00:00".into(),
        };
        let g = Geometry::new(Some(&spec), Some(&time), None, &[], &cam)
            .unwrap()
            .unwrap();
        assert!(g.sky_wcs_fit_px < 1e-3, "{}", g.sky_wcs_fit_px);
        let ab = g.aberration_arcsec();
        assert!((1.0..21.0).contains(&ab), "{ab}");
        let (ra, dec) = g.pixel_to_sky(0.0, 0.0);
        let (x, y) = g.sky_wcs.world_to_pixel(ra, dec).unwrap();
        assert!(x.abs() < 1e-3 && y.abs() < 1e-3);
        let (x, y) = g.sky_to_pixel(ra, dec).unwrap();
        assert!(x.abs() < 1e-9 && y.abs() < 1e-9);
    }

    #[test]
    fn sun_deflection_is_4_mas_at_90_degrees() {
        let l = Light {
            sun_to_obs: [-1.0, 0.0, 0.0],
            sun_dist_km: crate::ephem::AU_KM,
            obs_bary_au: [0.0; 3],
        };
        let p = [0.0, 1.0, 0.0];
        let q = deflect(p, &l, 1.0);
        let shift = angle(p, q).to_degrees() * 3_600_000.0;
        assert!((shift - 4.07).abs() < 0.01, "{shift}");
        assert!(q[0] < 0.0, "pushed away from the Sun");
        assert!(angle(p, deflect(q, &l, -1.0)) < 1e-14);
    }

    /// A star along +y with 1″ parallax, observer 1 au along +x: 1″ shift.
    #[test]
    fn parallax_shifts_a_near_star() {
        let l = Light {
            sun_to_obs: [0.0, 0.0, 1.0],
            sun_dist_km: crate::ephem::AU_KM,
            obs_bary_au: [1.0, 0.0, 0.0],
        };
        let p = apparent_dir([0.0, 1.0, 0.0], 1000.0, Some(&l), [0.0; 3]);
        let q = apparent_dir([0.0, 1.0, 0.0], 0.0, Some(&l), [0.0; 3]);
        let shift = angle(p, q).to_degrees() * 3600.0;
        assert!((shift - 1.0).abs() < 1e-3, "{shift}");
    }

    #[test]
    fn observer_needs_time() {
        let spec = PointingSpec::from(Pointing {
            ra_deg: 1.0,
            dec_deg: 1.0,
            roll_deg: 0.0,
        });
        let r = Geometry::new(
            Some(&spec),
            None,
            Some(&Orbit::Geocenter),
            &[],
            &Camera::default(),
        );
        assert!(matches!(r, Err(SimError::BadTime(_))));
    }
}
