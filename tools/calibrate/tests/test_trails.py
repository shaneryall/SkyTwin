"""Trailed frames: stars drawn as segments with a known PSF -> trail length and angle, PSF, photometry and
radiation back."""

import numpy as np
from astropy.io import fits
from scipy.special import erf

from skytwin_calibrate import trails
from skytwin_calibrate.util import FWHM_PER_SIGMA


def draw_trail(img, x, y, flux, length, angle_deg, comps, sub=3):
    """Add a trail of `flux` e centred at (x, y) to `img`: each PSF component's density across the trail
    times its erf ends along it, sampled `sub` × `sub` per pixel."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    ext = int(length / 2 + 6 * max(w for _, w in comps)) + 2
    xi, yi = int(np.rint(x)), int(np.rint(y))
    y0, y1 = max(yi - ext, 0), min(yi + ext + 1, img.shape[0])
    x0, x1 = max(xi - ext, 0), min(xi + ext + 1, img.shape[1])
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    acc = np.zeros_like(xx)
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    for dy in offs:
        for dx in offs:
            u = (xx + dx - x) * c + (yy + dy - y) * s
            v = -(xx + dx - x) * s + (yy + dy - y) * c
            for f, w in comps:
                sg = w / FWHM_PER_SIGMA
                along = 0.5 * (erf((u + length / 2) / (sg * np.sqrt(2))) - erf((u - length / 2) / (sg * np.sqrt(2))))
                acc += f * np.exp(-0.5 * (v / sg) ** 2) / (sg * np.sqrt(2 * np.pi)) * along / max(length, 1e-9)
    img[y0:y1, x0:x1] += flux * acc / (sub * sub)


def trail_frame(n=1024, length=60.0, angle=25.0, comps=((0.6, 2.0), (0.4, 5.0)), stars=120, sky=200.0, seed=1):
    """(frame in e, star list [(x, y, flux)]): `stars` trails with fluxes 2e4 to 1e6 e, Poisson noise on a
    flat sky, read noise 5 e."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), sky)
    lst = []
    for _ in range(stars):
        x, y = rng.uniform(60, n - 60, 2)
        f = float(10 ** rng.uniform(4.3, 6))
        draw_trail(img, x, y, f, length, angle, comps)
        lst.append((x, y, f))
    return rng.poisson(img) + rng.normal(0, 5.0, img.shape), lst


def test_geometry_finds_the_trail_length_and_angle():
    """A trail's length and angle are found, and an untrailed star is not a trail: a wide halo widens
    the ridge, but not to a trail's length."""
    d, _ = trail_frame(length=60.0, angle=25.0)
    L, ang, _ = trails.geometry(d)
    assert abs(L / 60 - 1) < 0.02 and abs(ang - 25.0) < 1.5, (L, ang)
    d, _ = trail_frame(length=0.0, angle=0.0, seed=2)
    L, _, _ = trails.geometry(d)
    assert L < trails.MIN_TRAIL_PX, L
    for true in (15.0, 30.0):
        d, _ = trail_frame(n=512, length=true, angle=33.0, comps=((1.0, 2.2),), stars=60, seed=int(true))
        L, ang, _ = trails.geometry(d)
        assert abs(L / true - 1) < 0.06 and abs(ang - 33.0) < 2.0, (true, L, ang)


def test_psf_comes_from_the_profile_across_the_trails():
    comps = ((0.6, 2.0), (0.4, 5.0))
    Vs, Ys = [], []
    for seed in range(4):
        d, _ = trail_frame(comps=comps, seed=seed + 10)
        V, Y, n = trails.cross_profile(d, 60.0, 25.0)
        assert n >= 5
        Vs.append(V)
        Ys.append(Y)
    fit, notes = trails.fit_marginal(np.concatenate(Vs), np.concatenate(Ys))
    assert abs(fit["core_fwhm_px"] / 2.0 - 1) < 0.1, (fit, notes)
    wing = max(fit["wings"], key=lambda w: w["fraction"])
    assert abs(wing["fwhm_px"] / 5.0 - 1) < 0.15 and abs(wing["fraction"] - 0.4) < 0.1, fit


def test_trail_photometry_recovers_the_flux():
    comps = [(0.6, 2.0), (0.4, 5.0)]
    assert 0.9 < trails.aperture_fraction(comps, 60.0) < 1.0
    d, lst = trail_frame(comps=comps, stars=80, seed=3)
    px, py, fl = (np.array(v) for v in zip(*lst))
    mag = -2.5 * np.log10(fl) + 25.0
    rows = trails.photometry(d, px, py, mag, 1.0, 1.0, comps, 60.0, 25.0)
    zp = np.median([z for m, z, snr, _ in rows if snr > 20])
    assert len(rows) > 20 and abs(zp - 25.0) < 0.03, (len(rows), zp)


def test_radiation_outside_the_trails(tmp_path):
    """Trails move between frames, as they do when the telescope follows a satellite; hits rise toward the
    top rows (row 0 read first, 4 s to read); a hot pixel sits in every frame."""
    rng = np.random.default_rng(4)
    n, t, ro, rate_px = 512, 3.0, 4.0, 4.0
    files, geo = [], {}
    for k in range(20):
        d, _ = trail_frame(n=n, stars=40, seed=100 + k)
        d[100, 200] += 3000
        for _ in range(rng.poisson(rate_px * (t + ro / 2))):
            u = (-t + np.sqrt(t * t + 2 * ro * rng.uniform() * (t + ro / 2))) / ro
            yy, xx = min(int(u * n), n - 1), int(rng.integers(0, n - 2))
            d[yy, xx : xx + 2] += 800.0
        f = str(tmp_path / f"t{k}.fits")
        fits.writeto(f, d.astype(np.float32))
        files.append(f)
        geo[f] = (60.0, 25.0, 1.0)
    out, notes = trails.radiation(files, geo, t, 1.0)
    assert abs(out["rate_per_cm2_s"] / rate_px - 1) < 0.25, (out, notes)
    assert abs(out["readout_s"] / ro - 1) < 0.5 and out["first_row"] == "bottom", (out, notes)
    assert "20 more at pixels" in notes[0] or "more at pixels" in notes[0], notes
