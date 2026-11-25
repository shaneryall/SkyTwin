"""Zeropoints: forced photometry with a wide PSF, and ground frames moved above the atmosphere."""

import numpy as np
from astropy.io import fits
from scipy.special import erf

from skytwin_calibrate import astrometry, keywords
from skytwin_calibrate.util import FWHM_PER_SIGMA


def star_frame(comps, n=512, stars=150, sky=300.0, seed=1):
    """(frame in e, x, y, flux): round stars of 3e4 to 1e6 e on a flat sky, a sum of pixel-integrated
    Gaussians `comps` [(fraction, fwhm_px)]."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), sky)
    xs, ys = rng.uniform(40, n - 40, stars), rng.uniform(40, n - 40, stars)
    fl = 10 ** rng.uniform(4.5, 6, stars)
    edges = np.arange(n + 1) - 0.5
    for x, y, f in zip(xs, ys, fl):
        for frac, fwhm in comps:
            s = fwhm / FWHM_PER_SIGMA
            gx = 0.5 * np.diff(erf((edges - x) / (s * np.sqrt(2))))
            gy = 0.5 * np.diff(erf((edges - y) / (s * np.sqrt(2))))
            img += frac * f * np.outer(gy, gx)
    return rng.poisson(img) + rng.normal(0, 8.0, img.shape), xs, ys, fl


def test_apertures_follow_a_wide_psf():
    """A 4.7 px core with a 20% halo the PSF model lacks: the aperture grows to 2.5 FWHM and its annulus
    clears the wings, so the zeropoint comes back within 0.05 mag; a fixed 4 px aperture with a 6-10 px
    annulus leans on the model and reads it over 0.1 mag too faint."""
    d, xs, ys, fl = star_frame([(0.8, 4.7), (0.2, 12.0)])
    mag = 25.0 - 2.5 * np.log10(fl)
    model = [(1.0, 4.7)]
    rows = astrometry.forced_photometry(d, xs, ys, mag, 1.0, 1.0, model)
    zp = np.median([z for _, z, snr, _ in rows if snr > 30])
    assert len(rows) > 30 and abs(zp - 25.0) < 0.05, (len(rows), zp)
    old = astrometry.forced_photometry(d, xs, ys, mag, 1.0, 1.0, model, r_ap=4.0)
    assert np.median([z for _, z, snr, _ in old if snr > 30]) < 24.9


def _frames(airmasses, k=0.29, zp0=22.84, coef=0.86, stars=40, seed=2):
    """Per-frame photometry rows [(airmass, rows)]: stars with colours, zp = zp0 - k X - coef × colour."""
    rng = np.random.default_rng(seed)
    out = []
    for x in airmasses:
        c = rng.uniform(0.3, 2.0, stars)
        m = rng.uniform(10.0, 14.0, stars)
        z = zp0 - (k * x if x else 0.0) - coef * c + rng.normal(0, 0.05, stars)
        out.append((x, [(float(mi), float(zi), 100.0, float(ci)) for mi, zi, ci in zip(m, z, c)]))
    return out


def test_extinction_moves_the_zeropoint_above_the_atmosphere():
    pf = _frames(np.linspace(1.45, 3.5, 27))
    cam, notes = astrometry.zeropoint_and_extinction(pf)
    atm = cam.pop("_atmosphere")
    assert atm["measured"] and abs(atm["extinction_mag_per_airmass"] - 0.29) < 0.02, (atm, notes)
    assert abs(cam["zeropoint_mag"] - 22.84) < 0.03 and abs(cam["color_term"]["coefficient"] - 0.86) < 0.03, cam


def test_extinction_is_assumed_when_the_airmass_barely_changes():
    pf = _frames(np.full(20, 1.155) + np.linspace(0, 0.1, 20))
    cam, notes = astrometry.zeropoint_and_extinction(pf, extinction=None)
    atm = cam.pop("_atmosphere")
    assert not atm["measured"] and atm["extinction_mag_per_airmass"] == 0.2
    through = 22.84 - 0.29 * 1.205
    assert abs(cam["zeropoint_mag"] - (through + 0.2 * 1.205)) < 0.03, (cam, notes)
    assert any("assumed extinction" in n for n in notes)
    cam, _ = astrometry.zeropoint_and_extinction(_frames([None] * 10))
    assert "_atmosphere" not in cam and abs(cam["zeropoint_mag"] - 22.84) < 0.03


def test_altitude_gives_the_airmass():
    h = fits.Header({"ALT_PNT": 60.0, "EXPOSURE": 3.0})
    m = keywords.frame_meta(h, {"altitude_deg": "ALT_PNT", "exposure_s": "EXPOSURE"})
    assert abs(m["airmass"] - 1.154) < 0.002, m
    m = keywords.frame_meta(fits.Header({"AIRMASS": 2.1}), {"airmass": "AIRMASS"})
    assert m["airmass"] == 2.1


def test_a_cloud_over_some_frames_does_not_move_the_zeropoint():
    """A fifth of the frames 0.4 mag dimmer: the median of the frames' zeropoints keeps the clear-sky
    value, where a median over all stars would drop by about 0.1 mag."""
    pf = _frames([None] * 20)
    for i in range(4):
        pf[i] = (None, [(m, z - 0.4, s, c) for m, z, s, c in pf[i][1]])
    cam, notes = astrometry.zeropoint_and_extinction(pf)
    assert abs(cam["zeropoint_mag"] - 22.84) < 0.03, (cam, notes)
    assert any("median of frames is used" in n for n in notes)
