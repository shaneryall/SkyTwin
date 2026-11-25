"""Per-frame statistics, the same for real and simulated frames.

Every estimator holds up against stars and against clipping at zero: block
medians for the background shape, the median absolute deviation for noise
(from quantiles above the clip when the clip reaches it), and an FFT peak
(away from the axes) for coherent pattern noise.
"""

import warnings

import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.special import ndtri

from . import keywords
from .util import mad_sigma, write_rows


def clipped_noise(a):
    """Gaussian σ of the background from two quantiles above the lower clip.

    Above the clip the observed quantiles equal the true ones, so the
    estimate holds when most pixels sit at the clip (a dark frame with a
    bias below zero), where the MAD collapses to 0. Both quantiles stay
    below the upper tail that stars and hits fill. Returns None when more
    than 95% of pixels are clipped."""
    qs = _clip_quantiles(a, 0.16)
    if qs is None:
        return None
    lo, hi = np.quantile(a, qs)
    return float((hi - lo) / (ndtri(qs[1]) - ndtri(qs[0])))


def _clip_quantiles(a, floor):
    """Two quantiles above the lower clip (None if over 95% is clipped)."""
    f = float((a <= a.min()).mean())
    if f > 0.95:
        return None
    qa = max(floor, f + 0.03)
    return qa, min(qa + 0.3, (qa + 1) / 2)


def _row_levels(a):
    """Each row's level from a quantile above the clip (None if over 95%
    of the frame is clipped)."""
    qs = _clip_quantiles(a, 0.25)
    return None if qs is None else np.quantile(a, qs[0], axis=1)


def row_alternation(a):
    """Odd/even row offset, ADU: even rows' level minus the mean of their
    odd neighbours (a read-out fault some frames show as a comb). 0 when
    it cannot be measured."""
    lo = _row_levels(a)
    if lo is None or len(lo) < 5:
        return 0.0
    e = np.arange(2, len(lo) - 1, 2)
    return float(np.median(lo[e] - 0.5 * (lo[e - 1] + lo[e + 1])))


def _smooth_rows(v, iters=3):
    """Degree-5 polynomial in row number, refitted `iters` times without 4σ outliers.

    Degree 5 follows a steep glare edge out to the frame's edge.
    """
    x = np.linspace(-1, 1, len(v))
    keep = np.ones(len(v), bool)
    for _ in range(iters):
        c = np.polynomial.polynomial.polyfit(x[keep], v[keep], 5)
        r = v - np.polynomial.polynomial.polyval(x, c)
        mad = mad_sigma(r[keep], 0.0) or 1.0
        keep = np.abs(r) < 4 * mad
    return np.polynomial.polynomial.polyval(x, c)


def bad_rows(a, sigma):
    """Rows the read-out corrupted: bands of salt-and-pepper or saturated
    pixels, and dropped (black) rows.

    A row's spread is the median absolute difference of neighbouring
    pixels both above the clip, which a gradient along the row cancels;
    a row is bad when it is over twice or under 0.4 times the typical
    row's (a dropped row has no unclipped pairs: spread 0). A row is also
    bad when its level jumps from its neighbours': levels are taken after
    removing the odd/even comb, an outlier-clipped polynomial and a 31-row running
    median, against a threshold of max(σ/2, 6 × their own scatter, 3 ADU).
    Rows with no unclipped pairs give no spread (NaN median, warning
    silenced) and count as dropped."""
    lo = _row_levels(a)
    if lo is None or not sigma:
        return np.zeros(a.shape[0], bool)
    clip = a.min()
    left, right = a[:, :-1], a[:, 1:]
    both = (left > clip) & (right > clip)
    dif = np.where(both, np.abs(right - left), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nanmedian(dif, axis=1)
    spread = np.where(both.sum(axis=1) >= 0.05 * a.shape[1], spread, 0.0)
    typ = np.median(spread)
    bad = np.zeros(len(lo), bool)
    if typ > 0:
        bad |= (spread > 2 * typ) | (spread < 0.4 * typ)
    alt = row_alternation(a)
    lo = lo - np.where(np.arange(len(lo)) % 2 == 0, 0.5, -0.5) * alt
    res = lo - _smooth_rows(lo)
    dev = res - ndimage.median_filter(res, 31, mode="nearest")
    scatter = mad_sigma(dev, 0.0)
    bad |= np.abs(dev) > max(0.5 * sigma, 6 * scatter, 3.0)
    return bad


def band_list(a, bad, high):
    """Corrupted bands as [first, last, kind]: `dropped` when over 90% of
    the band sits at the low clip, `saturated` when over 90% is at or
    above 95% of `high`, otherwise `noise`."""
    idx = np.flatnonzero(bad)
    if not len(idx):
        return []
    out = []
    for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        px = a[g[0] : g[-1] + 1]
        if (px <= a.min()).mean() > 0.9:
            kind = "dropped"
        elif (px >= 0.95 * high).mean() > 0.9:
            kind = "saturated"
        else:
            kind = "noise"
        out.append([int(g[0]), int(g[-1]), kind])
    return out


def background(d, x0):
    """Background statistics of frame `d` right of column `x0`, in ADU.

    `median` and `noise` (MAD σ; from `clipped_noise` when over 20% of pixels sit at the lower clip, where
    the MAD reads low, down to 0); `noise_dx`, the noise from neighbours along rows (MAD of the difference
    over √2), blind to row offsets, gradients and structure larger than a pixel; `zero_frac`, the fraction
    of pixels at exactly 0; `ramp_x` and `ramp_y`, the change across the frame of a plane fitted to 16×16
    block medians; `structure`, the RMS of the block medians about that plane.
    """
    a = d[:, x0:]
    med = float(np.median(a))
    mad = float(mad_sigma(a, med))
    if (a <= a.min()).mean() > 0.2:
        mad = clipped_noise(a) or mad
    ny, nx = a.shape
    by, bx = ny // 16, nx // 16
    blocks = np.array(
        [[np.median(a[i * by : (i + 1) * by, j * bx : (j + 1) * bx]) for j in range(16)] for i in range(16)]
    )
    yy, xx = np.mgrid[0:16, 0:16]
    A = np.c_[np.ones(256), (xx * bx + bx / 2).ravel(), (yy * by + by / 2).ravel()]
    c, *_ = np.linalg.lstsq(A, blocks.ravel(), rcond=None)
    resid = blocks.ravel() - A @ c
    dx = np.diff(a, axis=1)
    noise_dx = float(mad_sigma(dx) / np.sqrt(2))
    return dict(
        median=med,
        noise=mad,
        noise_dx=noise_dx,
        zero_frac=float((d == 0).mean()),
        ramp_x=float(c[1] * nx),
        ramp_y=float(c[2] * ny),
        structure=float(resid.std()),
    )


def _block_resid(a, mask, n):
    """Residuals of masked block medians on an n×n grid about a quadratic surface, and the median count of
    unmasked pixels per block.

    Blocks under 30% unmasked are NaN; (None, 0) when fewer than 10 blocks remain. The quadratic takes
    out vignetting and glare curvature, which are frame-sized; the field's features are much smaller.
    """
    ny, nx = a.shape
    by, bx = ny // n, nx // n
    b = np.full((n, n), np.nan)
    cnt = []
    for i in range(n):
        for j in range(n):
            blk = a[i * by : (i + 1) * by, j * bx : (j + 1) * bx][~mask[i * by : (i + 1) * by, j * bx : (j + 1) * bx]]
            cnt.append(blk.size)
            if blk.size > 0.3 * by * bx:
                b[i, j] = np.median(blk)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1) - 0.5
    X, Y = xx.ravel(), yy.ravel()
    A = np.c_[np.ones(n * n), X, Y, X * X, Y * Y, X * Y]
    ok = np.isfinite(b.ravel())
    if ok.sum() < 10:
        return None, 0
    c, *_ = np.linalg.lstsq(A[ok], b.ravel()[ok], rcond=None)
    r = np.full(n * n, np.nan)
    r[ok] = b.ravel()[ok] - A[ok] @ c
    return r.reshape(n, n), float(np.median(cnt))


def lag_cov(r):
    """Covariance of a 2-D residual grid at lags 1 and 2 (x and y pooled,
    NaN cells skipped). Pixel noise is independent between blocks, so these
    hold only structure."""
    out = []
    m = np.nanmean(r)
    for k in (1, 2):
        a = np.r_[(r[:, :-k] - m).ravel(), (r[:-k] - m).ravel()]
        b = np.r_[(r[:, k:] - m).ravel(), (r[k:] - m).ravel()]
        ok = np.isfinite(a) & np.isfinite(b)
        out.append(float(np.mean(a[ok] * b[ok])) if ok.sum() > 10 else 0.0)
    return out


def field(d, x0, med, mad, valid=None):
    """Background structure with sources masked (pixels 5σ above the
    background in a 3×3 mean, grown by 8 px): the RMS of 16×16 block
    medians about a quadratic surface with the RMS noise alone would give,
    and on a 32×32 grid the covariances at lags 1 and 2 (noise-free; they
    give the smooth field's size and strength)."""
    a = d[:, x0:]
    mask = ndimage.binary_dilation(ndimage.uniform_filter(a, 3) > med + 5 * mad / 3, iterations=8)
    if valid is not None:
        mask |= ~valid
    r16, cnt = _block_resid(a, mask, 16)
    r32, _ = _block_resid(a, mask, 32)
    if r16 is None or r32 is None:
        return None
    c1, c2 = lag_cov(r32)
    return {
        "rms": float(np.nanstd(r16)),
        "floor": float(1.2533 * mad / np.sqrt(max(cnt, 1))),
        "cov1": c1,
        "cov2": c2,
        "block_fine": float(a.shape[1] / 32),
        "mask_frac": float(mask.mean()),
    }


def pattern(d, x0, med, mad, size=256):
    """The strongest coherent pattern in the central `size`² pixels: period (px), angle (deg) and amplitude
    (ADU) of the highest FFT peak away from the axes (|f| < 0.02 cycles/px excluded).

    Pixels are clipped to ±5 `mad` about `med` first, so stars do not dominate.
    """
    a = d[:, x0:]
    ny, nx = a.shape
    s = min(size, ny, nx)
    cy, cx = ny // 2, nx // 2
    p = np.clip(a[cy - s // 2 : cy + s // 2, cx - s // 2 : cx + s // 2], med - 5 * mad, med + 5 * mad) - med
    F = np.abs(np.fft.fft2(p)) ** 2
    f = np.fft.fftfreq(p.shape[0])
    FY, FX = np.meshgrid(f, np.fft.fftfreq(p.shape[1]), indexing="ij")
    F[(np.abs(FY) < 0.02) | (np.abs(FX) < 0.02)] = 0
    F[FY < 0] = 0
    i = np.unravel_index(np.argmax(F), F.shape)
    if F[i] <= 0:
        return dict(period=None, angle=None, amplitude=0.0)
    return dict(
        period=float(1 / np.hypot(FX[i], FY[i])),
        angle=float(np.degrees(np.arctan2(FY[i], FX[i]))),
        amplitude=float(2 * np.sqrt(F[i]) / p.size),
    )


def sharp_mask(d, med, mad, skip_rows=None):
    """Single-pixel events: 8σ above background and 2.5× their neighbours,
    outside `skip_rows` (corrupted rows and the rows either side)."""
    thr = med + 8 * max(mad, 1.0)
    nb = (ndimage.uniform_filter(d, 3) * 9 - d) / 8
    ev = (d > thr) & ((d - med) > 2.5 * np.maximum(nb - med, 1.0))
    if skip_rows is not None and skip_rows.any():
        ev[ndimage.binary_dilation(skip_rows), :] = False
    return ev


def sharp_events(d, med, mad, skip_rows=None):
    """Number of single-pixel events (see `sharp_mask`)."""
    return int(sharp_mask(d, med, mad, skip_rows).sum())


def frame(path, km):
    """Per-frame statistics of the FITS file at `path`, with header values read through keyword map `km`.

    On frames nearly all at the clip there is no noise to set a threshold by, so `sharp`, `bad_rows`,
    `row_alt` and `bands` are None. `sharp_upper` counts events in the upper half of the rows: hits
    collected while rows wait to be read make it differ from the lower half.
    """
    d = fits.getdata(path).astype(np.float64)
    h = fits.getheader(path)
    x0 = int(km.get("skip_left_columns", 0))
    b = background(d, x0)
    out = dict(file=str(path), shape=list(d.shape), **b)
    out["pattern"] = pattern(d, x0, b["median"], max(b["noise"], 1.0))
    a = d[:, x0:]
    if clipped_noise(a) is None:
        out["sharp"] = out["bad_rows"] = out["row_alt"] = out["bands"] = None
    else:
        bad = bad_rows(a, b["noise"])
        out["bad_rows"] = int(bad.sum())
        out["bands"] = band_list(a, bad, km.get("saturation_adu") or float(a.max()))
        out["row_alt"] = row_alternation(a)
        ev = sharp_mask(a, b["median"], b["noise"], bad)
        out["sharp"] = int(ev.sum())
        out["sharp_upper"] = int(ev[a.shape[0] // 2 :].sum())
    out["field"] = field(d, x0, b["median"], max(b["noise"], 1.0))
    out["saturated"] = int((d >= km.get("saturation_adu", np.inf)).sum())
    out["meta"] = keywords.frame_meta(h, km)
    return out


def run(files, km, out_path, progress=True):
    """Measure every frame in `files` (see `frame`), write the rows to `out_path` as JSON and return them.

    An unreadable frame gets a row with `file` and `error` instead, and the run carries on.
    """
    rows = []
    for i, f in enumerate(files):
        try:
            rows.append(frame(f, km))
        except Exception as e:
            rows.append(dict(file=str(f), error=str(e)))
        if progress and (i + 1) % 100 == 0:
            print(f"  measured {i + 1}/{len(files)}")
    write_rows(rows, out_path)
    return rows
