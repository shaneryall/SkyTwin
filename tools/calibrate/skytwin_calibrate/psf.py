"""Stack isolated stars and fit a multi-Gaussian PSF by forward modelling.

The forward model renders pixel-integrated Gaussians at random sub-pixel
positions and measures them with the same code as the real stars, so the
pixel width and the stacking are counted exactly once.
"""

import numpy as np
from astropy.io import fits
from scipy import ndimage, optimize, special

from .util import FWHM_PER_SIGMA, mad_sigma, write_json

R = 40
"""Profile radius, px; smaller for small frames (see `radius_for`)."""


def radius_for(shape):
    """Profile radius for a frame: 40 px, or a sixteenth of a small frame's
    side (at least 10), so stamps (radius + 20 px of background) fit."""
    return int(min(R, max(10, min(shape) // 16)))


def _profile(cut, rr_off, R=R):
    """Radial profile of a background-subtracted stamp, normalised to the flux within `R`, about its 5×5
    centroid; None when that flux is not positive.

    The residual background is the mean of the ring R + 5 to R + 20 px, a mean like the profile's rings.
    """
    c = cut[R + 18 : R + 23, R + 18 : R + 23]
    cy = (c.sum(1) @ np.arange(-2, 3)) / c.sum()
    cx = (c.sum(0) @ np.arange(-2, 3)) / c.sum()
    yy, xx = rr_off
    r = np.hypot(yy - cy, xx - cx)
    cut = cut - np.mean(cut[(r > R + 5) & (r < R + 20)])
    tot = cut[r < R].sum()
    if tot <= 0:
        return None
    return [np.mean(cut[(r >= a) & (r < a + 1)]) / tot for a in range(R)]


def _fixed_pixels(files, nsig=5.0):
    """Pixels bright in most frames (hot pixels): high in the median over
    frames of each frame's excess over a local median. Stars move between
    pointings and drop out of the median. None with under 10 frames."""
    if len(files) < 10:
        return None
    res = []
    for f in files[:150]:
        d = fits.getdata(f).astype(float)
        res.append(d - ndimage.median_filter(d, size=5))
    try:
        med = np.median(np.array(res), axis=0)
    except ValueError:
        return None
    s = mad_sigma(med, 0.0) + 1e-9
    return ndimage.binary_dilation(med > nsig * s, iterations=1)


def stack(files, x0=0, peak_range=(3000, 55000), radius=None):
    """Median radial profile of isolated unsaturated stars, each against its own local background.

    Candidates are local maxima (9×9) with peaks in `peak_range` ADU, away from hot pixels and from the
    first `x0` columns. A star is isolated when nothing beyond 6 px of it exceeds 20% of its peak.
    Returns (profile, number of stars).
    """
    profs = []
    R = radius
    fixed = _fixed_pixels(files)
    for f in files:
        d = fits.getdata(f).astype(float)
        if R is None:
            R = radius_for(d.shape)
        yy, xx = np.mgrid[-R - 20 : R + 21, -R - 20 : R + 21]
        rr = np.hypot(yy, xx)
        m = (d == ndimage.maximum_filter(d, 9)) & (d > peak_range[0]) & (d < peak_range[1])
        if fixed is not None and fixed.shape == d.shape:
            m &= ~fixed
        for y, x in zip(*np.nonzero(m)):
            if y < R + 20 or x < R + 20 + x0 or y >= d.shape[0] - R - 20 or x >= d.shape[1] - R - 20:
                continue
            cut = d[y - R - 20 : y + R + 21, x - R - 20 : x + R + 21]
            cut = cut - np.median(cut[rr > R + 5])
            if ((cut > 0.2 * cut[R + 20, R + 20]) & (rr > 6)).any():
                continue
            p = _profile(cut, (yy, xx), R)
            if p is not None:
                profs.append(p)
    if not profs:
        raise ValueError("no isolated unsaturated stars found")
    return np.median(np.array(profs), axis=0), len(profs)


def _model(params, offs, R=R):
    """Median profile of a three-Gaussian star (σ px and fractions in `params`) rendered at the sub-pixel
    offsets `offs` and measured like a real one."""
    s1, f2, s2, f3, s3 = params
    comps = [(1 - f2 - f3, s1), (f2, s2), (f3, s3)]
    yy, xx = np.mgrid[-R - 20 : R + 21, -R - 20 : R + 21]

    def pixel_integral(a, b, c, s):
        return 0.5 * (special.erf((b - c) / (np.sqrt(2) * s)) - special.erf((a - c) / (np.sqrt(2) * s)))

    def g(cx, cy, s):
        return pixel_integral(xx - 0.5, xx + 0.5, cx, s) * pixel_integral(yy - 0.5, yy + 0.5, cy, s)

    return np.median([_profile(sum(f * g(ox, oy, s) for f, s in comps), (yy, xx), R) for ox, oy in offs], axis=0)


def fit(profile, n_fit=20):
    """Core FWHM and two wings (fraction, FWHM in px) fitted to the inner `n_fit` px of a stacked profile.

    Residuals are weighted by 10% of the profile plus a floor; parameters outside 0.3 < σ1 < σ2 < σ3 < 20,
    non-negative fractions and wings under 80% of the flux are rejected. Forty fixed sub-pixel offsets
    (seed 1) make the model deterministic.
    """
    R = len(profile)
    n_fit = min(n_fit, R)
    offs = np.random.default_rng(1).uniform(-0.5, 0.5, (40, 2))

    def resid(p):
        s1, f2, s2, f3, s3 = p
        if not (0.3 < s1 < s2 < s3 < 20 and f2 >= 0 and f3 >= 0 and f2 + f3 < 0.8):
            return np.full(n_fit, 1e3)
        return (_model(p, offs, R)[:n_fit] - profile[:n_fit]) / (np.abs(profile[:n_fit]) * 0.1 + 2e-6)

    sol = optimize.least_squares(resid, [1.0, 0.15, 2.7, 0.02, 6.7], diff_step=1e-3)
    s1, f2, s2, f3, s3 = sol.x
    k = FWHM_PER_SIGMA
    return {
        "core_fwhm_px": float(k * s1),
        "wings": [{"fraction": float(f2), "fwhm_px": float(k * s2)}, {"fraction": float(f3), "fwhm_px": float(k * s3)}],
    }


def run(files, x0, out_path):
    """Stack and fit the stars in `files` (see `stack` and `fit`), write the result to `out_path` and return it."""
    prof, n = stack(files, x0)
    res = fit(prof)
    res.update(stars=n, profile=[float(v) for v in prof])
    write_json(res, out_path, indent=1)
    return res
