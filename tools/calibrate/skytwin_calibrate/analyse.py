"""Detector properties measured from a pile of ordinary frames.

No calibration frames are needed: everything here comes from science
frames at varied pointings, so stars move between frames while the
detector stays put.

- `camera_basics`: size, binning, pixel type, gain, read noise, pixel scale
  and exposure, from headers where they exist.
- `edge_columns`: dead or overscan columns at the left edge.
- `saturation`: the level where pixels pile up, and whether it is the
  converter's limit or the full well.
- `hot_pixels`: pixels high in most frames, their dark rates, and how the
  rates change with temperature.
- `flat`: each pixel's response from how it follows the frame level across
  frames (vignetting and pixel-to-pixel non-uniformity); two independent
  halves of the frames give the non-uniformity free of noise.
- `point_frames`: frames whose stars are round, for the PSF fit.

Further down: read-out effects (serial lag, smear, corrupted rows, read-out
time from radiation hits), gain and read noise by photon transfer, Earth
glare, smooth and fixed background structure, and a circular field stop.
"""

import os
import re
from collections import Counter
from itertools import pairwise

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.special import erfc

from . import keywords, measure
from .psf import _fixed_pixels
from .util import header_number, mad_sigma, percentiles, read_json, run_skytwin, spread, utc_timestamp, write_json


def _mode(vals):
    """Most common non-None value (the first seen on a tie), or None."""
    vals = [v for v in vals if v is not None]
    return Counter(vals).most_common(1)[0][0] if vals else None


def output_type(h):
    """Skytwin's `processing.output` pixel type for a header's BITPIX and BZERO (u16 when unrecognised)."""
    bitpix, bzero = int(h.get("BITPIX", 16)), float(h.get("BZERO", 0))
    if bitpix == 16:
        return "u16" if bzero == 32768 else "i16"
    if bitpix == 32:
        return "i32"
    if bitpix == -32:
        return "f32"
    return "f32" if bitpix == -64 else "u16"


def pixel_scale_from_wcs(h):
    """Pixel scale (arcsec) from a celestial WCS in any of its forms (CD,
    PC with CDELT, or CDELT alone); None without one."""
    if not any(k in h for k in ("CD1_1", "CDELT1", "PC1_1")) or "CTYPE1" not in h:
        return None
    try:
        m = WCS(h, naxis=2).pixel_scale_matrix
    except Exception:
        return None
    det = float(abs(np.linalg.det(m)))
    return 3600 * np.sqrt(det) if det > 0 else None


def camera_basics(headers, km):
    """Camera block from headers through keyword map `km`, binning (x, y), and notes [(setting, value, source)].

    Gain, read noise, binning and pixel scale are the most common header values (the pixel scale from the
    header WCS when no keyword gives it); exposure is the median. The pixel pitch is the header's pixel
    size over the x binning, unless the card's comment says the size is unbinned.
    """
    notes = []
    cam = {}
    g = _mode([header_number(h, km.get("gain_e_per_adu")) for h in headers])
    r = _mode([header_number(h, km.get("read_noise_e")) for h in headers])
    if g:
        cam["gain_e_per_adu"] = g
        notes.append(("camera.gain_e_per_adu", g, f"header {km['gain_e_per_adu']}"))
    if r:
        cam["read_noise_e"] = r
        notes.append(("camera.read_noise_e", r, f"header {km['read_noise_e']}"))
    s = _mode([header_number(h, km.get("pixel_scale_arcsec")) for h in headers])
    src = f"header {km.get('pixel_scale_arcsec')}"
    if s is None:
        s = _mode([round(pixel_scale_from_wcs(h), 4) if pixel_scale_from_wcs(h) else None for h in headers])
        src = "header WCS (not plate-solved)"
    if s:
        cam["pixel_scale_arcsec"] = float(s)
        notes.append(("camera.pixel_scale_arcsec", s, src))
    e = [header_number(h, km.get("exposure_s")) for h in headers]
    e = [v for v in e if v]
    if e:
        cam["exposure_s"] = float(np.median(e))
        notes.append(("camera.exposure_s", cam["exposure_s"], "median of the headers (overridden per twin)"))
    bx = _mode([header_number(h, km.get("binning_x")) for h in headers]) or 1
    by = _mode([header_number(h, km.get("binning_y")) for h in headers]) or 1
    key = km.get("pixel_size_um")
    p = _mode([header_number(h, key) for h in headers])
    if p:
        com = next((str(h.comments[key]) for h in headers if key in h), "").lower()
        unbinned = re.search(r"unbinned|before bin|physical", com) is not None
        cam["pixel_pitch_um"] = round(p if unbinned else p / bx, 3)
        notes.append(
            (
                "camera.pixel_pitch_um",
                cam["pixel_pitch_um"],
                f"header {key}" + ("" if unbinned or bx == 1 else f" / binning {bx:g}"),
            )
        )
    return cam, (int(bx), int(by)), notes


def edge_columns(files, max_frames=40):
    """Leading columns that are not sky (overscan, masked or dead): their
    median over a stack sits far from the image's, or they almost never
    hold a bright pixel (no light reaches them)."""
    stack = np.array([fits.getdata(f).astype(np.float32) for f in files[:max_frames]])
    col = np.median(np.median(stack, axis=0), axis=0)
    ref = np.median(col)
    col_sig = mad_sigma(col, ref) + 0.5
    med = np.median(stack, axis=(1, 2), keepdims=True)
    sig = mad_sigma(stack, med, axis=(1, 2), keepdims=True) + 1e-6
    bright = (stack > med + 10 * sig).mean(axis=(0, 1))
    dark = bright < 0.1 * np.median(bright) if np.median(bright) > 0 else np.zeros_like(bright, bool)
    bad = (np.abs(col - ref) > 8 * col_sig) | dark
    n = 0
    while n < len(col) // 4 and bad[n : n + 3].any():
        n += 1 + int(np.argmax(bad[n : n + 3]))
    return int(n)


def saturation(files, max_frames=300):
    """(level, frames): where flat-topped frame maxima cluster. A frame
    counts when at least 3 pixels lie within 0.5% of its maximum; the
    highest such maxima within 1% of each other form the cluster, and the
    level is their median (converters often saturate a little differently
    frame to frame)."""
    tops = []
    for f in files[:max_frames]:
        d = fits.getdata(f)
        m = float(d.max())
        if m > 0 and (d >= 0.995 * m).sum() >= 3:
            tops.append(m)
    if not tops:
        return None, 0
    hi = max(tops)
    cluster = [t for t in tops if t >= 0.99 * hi]
    if len(cluster) < 3:
        return None, 0
    return float(np.median(cluster)), len(cluster)


def _residual(d):
    """Frame `d` less its 5×5 median, and that residual's MAD σ."""
    r = d - ndimage.median_filter(d, size=5)
    sig = mad_sigma(r) + 1e-6
    return r, sig


def exposure_group(exposures, min_frames=10, ratio=1.25):
    """Indices of the frames to measure hot pixels from, and a note (None when nothing is left out).

    A row keeps collecting dark charge while it waits to be read, and that charge does not scale with
    the exposure: divided by 0.1 s it reads as a far higher rate than in a 3 s frame, and a median over
    both exposures inflates rates and counts. Exposures within `ratio` of their neighbour in sorted
    order form one group; the group of the longest exposures with at least `min_frames` frames is used,
    where the wait matters least. Frames without an exposure are left out of the grouping.
    """
    timed = sorted((e, i) for i, e in enumerate(exposures) if e)
    groups = []
    for e, i in timed:
        if groups and e <= ratio * groups[-1][-1][0]:
            groups[-1].append((e, i))
        else:
            groups.append([(e, i)])
    big = [g for g in groups if len(g) >= min_frames]
    if len(groups) < 2 or not big:
        return [i for _, i in timed], None
    best = max(big, key=lambda g: float(np.median([e for e, _ in g])))
    left = len(timed) - len(best)
    lo, hi = best[0][0], best[-1][0]
    span = f"{lo:g} s" if lo == hi else f"{lo:g} to {hi:g} s"
    return [i for _, i in best], (
        f"from the {len(best)} frames of {span}; {left} frames of other exposures left out (a row's dark "
        "charge while it waits to be read does not scale with the exposure)"
    )


def hot_pixels(files, metas, gain, x0=0, sat=None, max_frames=150):
    """Hot pixels from the median, over many frames, of each pixel's excess
    over its neighbours: stars and radiation hits move or come and go, hot
    pixels stay. Returns ({hot_pixels, dark} or None, notes): density, rate
    distribution (e-/s) at a reference temperature, and the temperature doubling
    if the frames span enough temperature.

    The threshold is 6 times the larger of the stack's MAD σ and the noise of
    a median of N frames, about 1.25 σ / √N: integer data can make the MAD
    of a median stack zero. With 3 °C or more of temperature range, each
    frame's rates over the per-pixel medians give one ratio per frame; a
    log2 slope over 0.02 per °C gives the doubling, and the rates are moved
    to the median temperature."""
    pairs = [(f, m) for f, m in zip(files, metas) if m.get("exposure_s")][:max_frames]
    if len(pairs) < 10:
        return None, ["fewer than 10 frames with an exposure time"]
    res = None
    sigs = []
    for i, (f, m) in enumerate(pairs):
        d = fits.getdata(f).astype(np.float32)
        r, sig = _residual(d)
        r = r * (gain / m["exposure_s"])
        sigs.append(sig * gain / m["exposure_s"])
        if sat is not None:
            r[d >= sat] = np.nan
        if res is None:
            res = np.empty((len(pairs), *r.shape), np.float32)
        res[i] = r
    stack = np.nanmedian(res, axis=0)
    stack[:, :x0] = 0
    body = stack[:, x0:]
    mad = mad_sigma(body)
    floor = 1.2533 * float(np.median(sigs)) / np.sqrt(len(pairs))
    thr = 6 * max(mad, floor)
    hot = stack > thr
    ys, xs = np.nonzero(hot)
    if len(ys) < 5:
        return None, [f"fewer than 5 pixels above {thr:.2f} e/s in the median of {len(pairs)} frames"]
    per_frame = res[:, ys, xs]
    per_pixel = stack[ys, xs]
    temps = np.array([m.get("temperature_c") if m.get("temperature_c") is not None else np.nan for _, m in pairs])
    notes = []
    tref = float(np.nanmedian(temps)) if np.isfinite(temps).any() else None
    doubling = None
    if tref is not None and np.nanmax(temps) - np.nanmin(temps) >= 3:
        ratio = np.nanmedian(per_frame / per_pixel, axis=1)
        ok = np.isfinite(temps) & (ratio > 0)
        if ok.sum() >= 10:
            slope, icpt = np.polyfit(temps[ok] - tref, np.log2(ratio[ok]), 1)
            if slope > 0.02:
                doubling = float(1 / slope)
                per_pixel = per_pixel * 2**icpt
        notes.append(
            f"temperature {np.nanmin(temps):.1f} to {np.nanmax(temps):.1f} C over {ok.sum()} frames"
            + (f": rates double every {doubling:.1f} C" if doubling else ": no clear trend")
        )
    vals = np.sort(per_pixel[per_pixel > 0])
    h, w = stack.shape
    block = {
        "density_per_mpix": round(len(vals) / ((w - x0) * h) * 1e6, 2),
        "e_per_s_at_ref": {"empirical": percentiles(vals, 3)},
    }
    dark = None
    if tref is not None:
        dark = {"e_per_s_at_ref": 0.0, "ref_temp_c": round(tref, 2)}
        if doubling:
            dark["doubling_c"] = round(doubling, 2)
    notes.insert(
        0,
        f"{len(vals)} hot pixels ({block['density_per_mpix']} per Mpix) above {thr:.2f} e/s in the "
        f"median of {len(pairs)} frames; rates {vals[0]:.2f} to {vals[-1]:.0f} e/s, median {np.median(vals):.1f}"
        + (f" at {tref:.1f} C" if tref is not None else ""),
    )
    return {"hot_pixels": block, "dark": dark}, notes


def flat(files, gain, x0=0, sat=None, axis=None, max_frames=400):
    """Response per pixel from how it follows the frame level. Returns
    (flat block or None, notes).

    Each pixel's slope against the frame median is accumulated over
    unsaturated pixels within 5σ of the level, separately for even and odd
    frames. Frame levels must span 30 ADU (p10 to p90) over 20 frames, or
    response cannot be told from additive background. A field stop, if
    found, masks the response map 3 edge widths inside its radius.

    Vignetting comes from 16×16 block medians fitted with a radial
    polynomial in (r / half-diagonal)² plus a plane (the plane takes up
    stray-light gradients that scale with level); kept when a2 is over 3σ
    and 0.01. Non-uniformity: the two halves' small-scale structure (over a
    9 px median) agrees only where it is real, so their covariance is the
    PRNU variance; measured over the unmasked map, only where the 9 px
    window is well inside any field stop, so its edge does not pass for
    non-uniformity."""
    acc = [None, None]
    levels = []
    for i, f in enumerate(files[:max_frames]):
        d = fits.getdata(f).astype(np.float64)[:, x0:]
        L = float(np.median(d))
        sig = mad_sigma(d, L) + 1e-6
        use = d < L + 5 * sig
        if sat is not None:
            use &= d < sat
        levels.append(L)
        k = i % 2
        if acc[k] is None:
            acc[k] = [np.zeros_like(d) for _ in range(5)]
        n, sx, sy, sxx, sxy = acc[k]
        u = use.astype(np.float64)
        n += u
        sx += u * L
        sy += u * d
        sxx += u * L * L
        sxy += u * d * L
    levels = np.array(levels)
    span = np.percentile(levels, 90) - np.percentile(levels, 10) if len(levels) else 0
    if len(levels) < 20 or span < 30:
        return None, [
            f"frame levels span only {span:.0f} ADU over {len(levels)} frames: "
            "the response cannot be told from additive background"
        ]

    def slope(a):
        n, sx, sy, sxx, sxy = a
        den = n * sxx - sx * sx
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, (n * sxy - sx * sy) / den, np.nan)

    b1, b2 = slope(acc[0]), slope(acc[1])
    b = np.nanmean([b1, b2], axis=0)
    h, w = b.shape
    out = {}
    stop, stop_notes = field_stop(b, x0, axis)
    inside = np.ones_like(b, dtype=bool)
    b_all = b
    if stop:
        out["field_stop"] = stop
        yy0, xx0 = np.mgrid[0:h, 0:w]
        r = np.hypot(xx0 + x0 - stop["centre_px"][0], yy0 - stop["centre_px"][1])
        inside = r < stop["radius_px"] - 3 * stop["edge_px"]
        b = np.where(inside, b, np.nan)
    by, bx = h // 16, w // 16
    blk = np.array(
        [[np.nanmedian(b[i * by : (i + 1) * by, j * bx : (j + 1) * bx]) for j in range(16)] for i in range(16)]
    )
    yy, xx = np.mgrid[0:16, 0:16]
    X = (xx * bx + bx / 2 + x0).ravel()
    Y = (yy * by + by / 2).ravel()
    ax = axis or ((w + x0 - 1) / 2, (h - 1) / 2)
    half = 0.5 * np.hypot(w + x0, h)
    r2 = ((X - ax[0]) ** 2 + (Y - ax[1]) ** 2) / half**2
    A = np.c_[np.ones_like(r2), r2, r2 * r2, X / half, Y / half]
    ok = np.isfinite(blk.ravel())
    c, *_ = np.linalg.lstsq(A[ok], blk.ravel()[ok], rcond=None)
    resid = blk.ravel()[ok] - A[ok] @ c
    cov = np.linalg.pinv(A[ok].T @ A[ok]) * resid.var()
    a2, a4 = c[1] / c[0], c[2] / c[0]
    err2 = np.sqrt(cov[1, 1]) / abs(c[0])
    notes = [f"{len(levels)} frames, levels spanning {span:.0f} ADU; mean response {c[0]:.3f}", *stop_notes]
    if abs(a2) > 3 * err2 and abs(a2) > 0.01:
        out["vignetting"] = [round(float(a2), 4), round(float(a4), 4)]
        notes.append(f"vignetting a2 = {a2:.3f} ± {err2:.3f}, a4 = {a4:.3f}")
    else:
        notes.append(f"no significant vignetting (a2 = {a2:.3f} ± {err2:.3f})")
    smooth = ndimage.median_filter(np.nan_to_num(b_all, nan=np.nanmedian(b_all)), size=9)
    e1, e2 = b1 / smooth - 1, b2 / smooth - 1
    deep = ndimage.binary_erosion(inside, iterations=6) if stop else inside
    m = deep & np.isfinite(e1) & np.isfinite(e2) & (np.abs(e1) < 0.5) & (np.abs(e2) < 0.5)
    cv = float(np.mean((e1[m] - e1[m].mean()) * (e2[m] - e2[m].mean())))
    if cv > 0 and np.sqrt(cv) > 0.002:
        out["prnu_rms"] = round(float(np.sqrt(cv)), 4)
        notes.append(f"pixel response non-uniformity {np.sqrt(cv) * 100:.2f}% RMS")
    else:
        notes.append("pixel response non-uniformity below 0.2% (or not measurable)")
    return (out or None), notes


def elongation(d, sat=None, x0=0, n_max=30):
    """Median elongation (major/minor axis, from second moments) of up to `n_max` of the brightest peaks over
    30σ and under 80% of saturation; None with fewer than 3."""
    med = np.median(d[:, x0:])
    sig = mad_sigma(d[:, x0:], med) + 1e-6
    pk = (d == ndimage.maximum_filter(d, 11)) & (d > med + 30 * sig)
    if sat is not None:
        pk &= d < 0.8 * sat
    pk[:, : x0 + 8] = False
    ys, xs = np.nonzero(pk)
    order = np.argsort(-d[ys, xs])[:n_max]
    el = []
    yy, xx = np.mgrid[-6:7, -6:7]
    for y, x in zip(ys[order], xs[order]):
        if y < 8 or x < 8 or y >= d.shape[0] - 8 or x >= d.shape[1] - 8:
            continue
        c = d[y - 6 : y + 7, x - 6 : x + 7] - med
        c = np.where(c > 3 * sig, c, 0)
        t = c.sum()
        if t <= 0:
            continue
        mx, my = (c * xx).sum() / t, (c * yy).sum() / t
        sxx = (c * (xx - mx) ** 2).sum() / t
        syy = (c * (yy - my) ** 2).sum() / t
        sxy = (c * (xx - mx) * (yy - my)).sum() / t
        ev = np.linalg.eigvalsh([[sxx, sxy], [sxy, syy]])
        if ev[0] > 0.05:
            el.append(np.sqrt(ev[1] / ev[0]))
    return float(np.median(el)) if len(el) >= 3 else None


def still_by_header(files, km, scale, max_headers=60000, want=300, max_trail_px=5.0):
    """Frames ranked by the trail their header tracking rate implies
    (`trail_rate_arcsec_per_s` × `exposure_s`), least first, up to `want`.
    Body rates in headers include jitter, so this only ranks candidates;
    `point_frames` decides from the stars themselves."""
    if not (km.get("trail_rate_arcsec_per_s") and km.get("exposure_s") and scale):
        return []
    ranked = []
    for f in files[:max_headers]:
        try:
            h = fits.getheader(f)
        except Exception:
            continue
        r, e = header_number(h, km["trail_rate_arcsec_per_s"]), header_number(h, km["exposure_s"])
        if r is not None and e and r * e / scale <= max_trail_px:
            ranked.append((r * e / scale, f))
    return [f for _, f in sorted(ranked)[:want]]


def point_frames(files, sat=None, x0=0, max_frames=300, limit=1.2):
    """Frames whose stars are round (stars not trailed)."""
    out = []
    for f in files[:max_frames]:
        e = elongation(fits.getdata(f).astype(np.float64), sat, x0)
        if e is not None and e < limit:
            out.append(f)
    return out


def metas(headers, km):
    """`keywords.frame_meta` for each header."""
    return [keywords.frame_meta(h, km) for h in headers]


def smear_sources(d, med, sig, sat=None, x0=0, min_flux=1e5, max_size=40):
    """Bright, compact, unsaturated sources as boxes (x0, x1, y0, y1) grown by 3 px, with flux in ADU.

    Sources are 3×3-mean pixels over 10σ, at most `max_size` px across, 30 px clear of column `x0`,
    under 90% of saturation and with at least `min_flux`. The local background is the median of a frame
    5 px wide around the box.
    """
    m = ndimage.uniform_filter(d, 3) > med + 10 * sig
    lab, _ = ndimage.label(m)
    out = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        ys, xs = sl
        if ys.stop - ys.start > max_size or xs.stop - xs.start > max_size or xs.start < x0 + 30:
            continue
        y0, y1 = max(ys.start - 3, 0), min(ys.stop + 3, d.shape[0])
        xa, xb = max(xs.start - 3, 0), min(xs.stop + 3, d.shape[1])
        cut = d[y0:y1, xa:xb]
        if sat is not None and cut.max() >= 0.9 * sat:
            continue
        ring = d[max(y0 - 5, 0) : y1 + 5, max(xa - 5, 0) : xb + 5].copy()
        ring[
            (y0 - max(y0 - 5, 0)) : (y0 - max(y0 - 5, 0)) + (y1 - y0),
            (xa - max(xa - 5, 0)) : (xa - max(xa - 5, 0)) + (xb - xa),
        ] = np.nan
        flux = float((cut - np.nanmedian(ring)).sum())
        if flux >= min_flux:
            out.append((xa, xb, y0, y1, flux))
    return out


def smear_measure(files, exposures, sat=None, x0=0, max_frames=300):
    """Median time per row that a source's light falls on the rows above
    and below it during read-out: the column excess per row, summed over
    the source's columns, over its flux rate. Returns (above, below, n) in
    seconds per row, or None for both with under 10 sources either side.

    Column means are 3σ-clipped against stars; the background under the
    source columns comes from a parabola through reference columns 8-25 px
    either side, since vignetting and glare curve it across the columns."""
    above, below = [], []
    for f, e in list(zip(files, exposures))[:max_frames]:
        if not e:
            continue
        d = fits.getdata(f).astype(np.float64)
        med = float(np.median(d[:, x0:]))
        sig = float(mad_sigma(d[:, x0:], med)) + 1e-6
        h, w = d.shape
        for xa, xb, y0, y1, flux in smear_sources(d, med, sig, sat, x0):
            cols = np.arange(xa, xb)
            ref = np.r_[max(xa - 25, x0) : max(xa - 8, x0), min(xb + 8, w) : min(xb + 25, w)]
            if len(ref) < 20 or xa - 25 < x0 or xb + 25 > w:
                continue
            for lo, hi, acc in ((y1 + 25, h, above), (0, y0 - 25, below)):
                if hi - lo < 60:
                    continue
                cm = sigma_clipped_stats(d[lo:hi], sigma=3, axis=0)[0]
                c = np.polyfit(ref - xa, cm[ref], 2)
                acc.append(float((cm[cols] - np.polyval(c, cols - xa)).sum()) / (flux / e))
    if len(above) < 10 or len(below) < 10:
        return None, None, min(len(above), len(below))
    return float(np.median(above)), float(np.median(below)), min(len(above), len(below))


def earth_probe(rows, skytwin, work, scale):
    """Skytwin's own Earth geometry for each real frame: render a 16×16 twin
    with a constant stray-light table and read the truth (angle from the
    limb, whether the nearest limb point is sunlit, and the Earth's
    direction in the image). Frames need a time, a spacecraft state and a
    pointing. Returns {file: (angle, lit, direction or None)}."""
    os.makedirs(work, exist_ok=True)
    prof = {
        "camera": {
            "width": 16,
            "height": 16,
            "pixel_scale_arcsec": scale or 3.0,
            "sky": {"e_per_s": 0.0},
            "read_noise_e": 0.0,
            "dark_e_per_s": 0.0,
        },
        "stray_light": [{"body": "earth", "angle_deg": [-90.0, 180.0], "e_per_s": [1.0, 1.0]}],
    }
    pp = os.path.join(work, "probe.json")
    write_json(prof, pp, allow_nan=False)
    out = {}
    for i, r in enumerate(rows):
        m = r.get("meta") or {}
        p = m.get("pointing")
        if "error" in r or not (m.get("time_utc") and m.get("position_km") and m.get("velocity_km_s") and p):
            continue
        sc = {
            "seed": i,
            "instrument": os.path.abspath(pp),
            "overrides": {"camera": {"exposure_s": m.get("exposure_s") or 1.0}},
            "scene": {
                "time": {"start_utc": m["time_utc"]},
                "observer": {
                    "state_vector": {
                        "epoch_utc": m["time_utc"],
                        "frame": m["state_frame"],
                        "position_km": m["position_km"],
                        "velocity_km_s": m["velocity_km_s"],
                    }
                },
                "pointing": dict(p, roll_deg=p["roll_deg"] if p.get("roll_deg") is not None else 0.0),
            },
        }
        f = os.path.join(work, f"p{i}")
        if not run_skytwin(skytwin, sc, f + ".fits", check=False):
            continue
        t = read_json(f + ".truth.json")
        st = [x for x in (t.get("instrument", {}).get("stray_light") or []) if x.get("body") == "earth"]
        if st:
            out[r["file"]] = (st[0]["angle_deg"], st[0]["lit"], st[0].get("direction"))
    return out


def earth_glare(rows, gain, geom, roll_known, min_bin=8):
    """Earth glare from real frames and their geometry (`earth_probe`):
    the background excess of frames with a sunlit limb against the angle
    from the limb (equal-count bins), the factor for a dark limb, and, if
    the roll is known, the gradient across the frame and its direction
    relative to the Earth. Returns (stray block or None, angle beyond which
    there is no glare, notes).

    The baseline is the median level of the 40% of frames farthest from the
    limb; the excess is in e-/s per pixel. Bins of `min_bin` frames count
    when their median excess exceeds 3 times the baseline scatter over
    √`min_bin` (and 1 e-/s); the glare runs from the closest bin out to the
    first bin back at the baseline. The gradient uses frames with clear
    glare (20 ADU or more): their ramps less the read-out ramps of
    glare-free frames, against the Earth's direction in the image."""
    pts = []
    for r in rows:
        g = geom.get(r.get("file"))
        e = (r.get("meta") or {}).get("exposure_s")
        if g is None or not e or "error" in r:
            continue
        pts.append((g[0], g[1], g[2], r["median"], e, r["ramp_x"], r["ramp_y"], r["shape"]))
    if len(pts) < 50:
        return None, None, [f"{len(pts)} frames with Earth geometry (need 50: time, spacecraft state and pointing)"]
    a = np.array([p[0] for p in pts])
    lit = np.array([p[1] > 0 for p in pts])
    lvl = np.array([p[3] for p in pts])
    ex = np.array([p[4] for p in pts])
    far = a >= np.percentile(a, 60)
    base = float(np.median(lvl[far]))
    exc = (lvl - base) * gain / ex
    noise = float(mad_sigma(exc[far]))

    def table(mask):
        idx = np.argsort(a[mask])
        aa, vv = a[mask][idx], exc[mask][idx]
        ang, val = [], []
        for k in range(0, len(aa) - min_bin + 1, min_bin):
            ang.append(float(np.median(aa[k : k + min_bin])))
            val.append(float(np.median(vv[k : k + min_bin])))
        return np.array(ang), np.array(val)

    ang, val = table(lit)
    sig = val > max(3 * noise / np.sqrt(min_bin), 1.0)
    if not sig.any() or not sig[0]:
        return (
            None,
            None,
            [
                f"no background rise towards a sunlit limb (closest {a[lit].min():.0f} deg; "
                f"{lit.sum()} lit, {(~lit).sum()} dark frames)"
            ],
        )
    last = int(np.argmin(sig)) - 1 if not sig.all() else len(sig) - 1
    free = float(ang[last + 1]) if last + 1 < len(ang) else float(ang[last] + 1.0)
    block = {
        "body": "earth",
        "angle_deg": [round(float(x), 2) for x in ang[: last + 1]] + [round(free, 2)],
        "e_per_s": [round(float(max(v, 0.01)), 2) for v in val[: last + 1]] + [0.0],
    }
    notes = [
        f"sunlit limb: background rises below {free:.1f} deg from the limb, to {max(block['e_per_s']):.0f} "
        f"e/s per pixel ({lit.sum()} lit and {(~lit).sum()} dark frames; baseline {base:.1f} ADU)"
    ]
    near = a < free
    if (near & ~lit).sum() >= min_bin and (near & lit).sum() >= min_bin:
        ratio = np.median(exc[near & ~lit]) / max(np.median(exc[near & lit]), 1e-9)
        block["dark_limb_factor"] = round(float(np.clip(ratio, 0.0, 1.0)), 3)
        notes.append(f"dark limb: {block['dark_limb_factor']} of the sunlit glare")
    if roll_known:
        rx0 = np.median([p[5] for p, f in zip(pts, far) if f])
        ry0 = np.median([p[6] for p, f in zip(pts, far) if f])
        gr, tw = [], []
        for p, e_, n_ in zip(pts, exc, near):
            if not n_ or p[2] is None or e_ * p[4] / gain < 20:
                continue
            gx, gy = p[5] - rx0, p[6] - ry0
            L = e_ * p[4] / gain
            mag = np.hypot(gx, gy)
            if mag <= 0:
                continue
            gr.append(np.expm1(mag / (2 * L)) if mag < 20 * L else np.nan)
            tw.append(np.degrees(np.arctan2(gy, gx) - np.arctan2(p[2][1], p[2][0])))
        gr = np.array(gr, float)
        if np.isfinite(gr).sum() >= 5:
            block["gradient"] = round(float(np.nanmedian(gr)), 3)
            t = np.radians(tw)
            block["toward_deg"] = round(
                float(np.degrees(np.arctan2(np.median(np.sin(t)), np.median(np.cos(t))))) % 360, 1
            )
            notes.append(
                f"gradient {block['gradient']} across half the frame, {block['toward_deg']} deg from the "
                f"Earth's direction ({np.isfinite(gr).sum()} frames)"
            )
    return block, free, notes


def _bspline_field(w, h, scale, rng):
    """The same field as Skytwin's `noise.smooth_field` (zero mean, unit RMS): Gaussian nodes every `scale` px
    interpolated with a cubic B-spline."""
    gw, gh = int(w / scale) + 4, int(h / scale) + 4
    g = rng.normal(size=(gh, gw))

    def n_nodes(n):
        return int(n / scale) + 4

    def weights(n):
        p = np.arange(n) / scale
        i = np.floor(p).astype(int)
        t = p - i
        u = 1 - t
        wts = np.stack([u**3 / 6, (3 * t**3 - 6 * t**2 + 4) / 6, (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6, t**3 / 6], 1)
        M = np.zeros((n, n_nodes(n)))
        for k in range(4):
            M[np.arange(n), i + k] += wts[:, k]
        return M

    f = weights(h) @ g[: n_nodes(h)] @ weights(w).T
    f -= f.mean()
    return f / f.std()


def field_scale_table(block, n_blocks=32, scales=None, seed=4, reps=4):
    """For a unit-RMS field of each scale, measured as `measure.field` does
    (block means on an n×n grid, quadratic surface removed): covariance at
    lags 1 and 2. Returns [(scale, cov1, cov2)]."""
    scales = scales or [float(s) for s in np.geomspace(4, 400, 24)]
    rng = np.random.default_rng(seed)
    size = int(np.round(block * n_blocks))
    out = []
    for s in scales:
        c = []
        for _ in range(reps):
            f = _bspline_field(size, size, s, rng)
            r, _ = measure._block_resid(f, np.zeros_like(f, bool), n_blocks)
            c.append(measure.lag_cov(r))
        c1, c2 = np.mean(c, axis=0)
        out.append((s, float(c1), float(c2)))
    return out


def smooth_field(rows):
    """`noise.smooth_field` from the masked structure of each frame. The
    lag-2 to lag-1 covariance ratio (median over frames) gives the scale;
    each frame's lag-1 covariance then gives its RMS, stored per frame as
    `field_rms` so profiles can link it. The ratio rises with scale; only the
    rising part of the table is used for the inversion. Returns (block or
    None, notes)."""
    have = [r for r in rows if "error" not in r and r.get("field") and "cov1" in r["field"]]
    fl = [r["field"] for r in have]
    if len(fl) < 20:
        return None, ["fewer than 20 frames with masked structure"]
    c1 = np.array([f["cov1"] for f in fl])
    c2 = np.array([f["cov2"] for f in fl])
    noise_c1 = np.median(np.abs(c1 - np.median(c1)))
    if np.median(c1) <= 2 * noise_c1 or np.median(c1) <= 0:
        return None, [f"no smooth structure beyond noise (lag-1 covariance {np.median(c1):.3f} ADU²)"]
    ratio = float(np.median(c2) / np.median(c1))
    table = field_scale_table(np.median([f["block_fine"] for f in fl]))
    s = np.array([t[0] for t in table])
    rt = np.array([t[2] / t[1] for t in table])
    top = int(np.argmax(rt))
    s, rt, k1 = s[: top + 1], rt[: top + 1], np.array([t[1] for t in table])[: top + 1]
    scale = float(np.interp(np.clip(ratio, rt[0], rt[-1]), rt, s))
    k = float(np.interp(scale, s, k1))
    rms = np.sqrt(np.clip(c1, 0, None) / k)
    for r, v in zip(have, rms):
        r["field_rms"] = float(v)
    block = {"rms_adu": {"empirical": percentiles(rms, 3)}, "scale_px": round(scale, 1)}
    return block, [
        f"{np.median(rms):.2f} ADU RMS (p90 {np.percentile(rms, 90):.2f}), feature size {scale:.0f} px "
        f"(lag-2/lag-1 covariance {ratio:.2f}), with sources masked; {len(fl)} frames"
    ]


def readout_faults(rows):
    """The `readout_faults` block from the corrupted bands `measure` found:
    the fraction of frames with any, bands per corrupted frame, rows per
    band, the mix of kinds, and how often a dropped band runs to the last
    row (a frame cut short). None when no frame has a band."""
    ok = [r for r in rows if "error" not in r and r.get("bands") is not None]
    if len(ok) < 20:
        return None, [f"only {len(ok)} frames measured"]
    hit = [r for r in ok if r["bands"]]
    if not hit:
        return None, [f"no corrupted rows in {len(ok)} frames"]
    nb, rows_per, kinds, trunc = [], [], Counter(), 0
    for r in hit:
        h = r["shape"][0]
        bands = r["bands"]
        if bands[-1][2] == "dropped" and bands[-1][1] == h - 1 and bands[-1][1] - bands[-1][0] >= 10:
            trunc += 1
            bands = bands[:-1]
        nb.append(max(len(bands), 1))
        for first, last, k in bands:
            rows_per.append(last - first + 1)
            kinds[k] += 1
    block = {
        "probability": round(len(hit) / len(ok), 4),
        "bands": {"empirical": percentiles(nb)},
        "band_rows": {"empirical": percentiles(rows_per)} if rows_per else 1,
        "kind_weights": [kinds["noise"], kinds["dropped"], kinds["saturated"]] if kinds else [1, 1, 1],
        "truncate_probability": round(trunc / len(hit), 3),
    }
    if not any(block["kind_weights"]):
        block["kind_weights"] = [1, 1, 1]
    nf = _noise_band_shape(hit)
    if nf:
        block.update(nf[0])
    notes = [
        f"{len(hit)} of {len(ok)} frames ({100 * len(hit) / len(ok):.1f}%) have corrupted rows: "
        f"median {np.median(nb):.0f} band(s) of {np.median(rows_per) if rows_per else 0:.0f} row(s); "
        f"kinds noise/dropped/saturated {kinds['noise']}/{kinds['dropped']}/{kinds['saturated']}; "
        f"{trunc} cut short"
    ] + (nf[1] if nf else [])
    return block, notes


def _noise_band_shape(hit, max_bands=300):
    """How `noise` bands look: the share of their pixels at the frame's
    low clip, and the extra scatter of the rest about the rows beside them
    (each band's RMS of row-minus-neighbours, less a clean row's). The
    scatter is a MAD σ, not a variance: a star crossing the row is not
    read-out scatter. Returns (settings, notes) or None."""
    low, extra = [], []
    for r in hit:
        if len(extra) >= max_bands:
            break
        bands = [b for b in r["bands"] if b[2] == "noise"]
        if not bands or not r.get("file"):
            continue
        d = fits.getdata(r["file"]).astype(np.float64)
        h = d.shape[0]
        lo = d.min()
        for first, last, _ in bands:
            if first < 3 or last > h - 4:
                continue
            nb = 0.5 * (d[first - 1] + d[last + 1])
            band = d[first : last + 1]
            low.append(float(np.mean(band <= lo)))
            ok = band > lo
            diff = (band - nb)[ok]
            base = d[first - 2] - 0.5 * (d[first - 3] + d[first - 1])
            ex2 = mad_sigma(diff) ** 2 - mad_sigma(base) ** 2 if diff.size > 10 else 0.0
            extra.append(float(np.sqrt(max(ex2, 0.0))))
    if len(extra) < 10:
        return None
    out = {"noise_low_fraction": round(float(np.median(low)), 3), "noise_rms_adu": {"empirical": percentiles(extra, 3)}}
    return out, [
        f"noise bands ({len(extra)} measured): {np.median(low):.1%} of pixels at the low clip, the rest "
        f"scattered {np.median(extra):.1f} ADU RMS about their neighbours (p90 {np.percentile(extra, 90):.0f})"
    ]


def radiation_readout(rows, min_hits=300):
    """Read-out time from how hits split between the frame's halves.

    A row collects hits for the exposure t plus the time until it is read,
    rising to R for the last row; the halves then hold hits in the ratio
    r = (t + 3R/4) / (t + R/4), so R = t (r − 1) / (3/4 − r/4). Frames with
    at least `min_hits` events are pooled (hits outnumber hot pixels
    there). Returns (readout_s, first_row, notes); readout_s None when
    the split is within 2σ (Poisson) of even, where no read-out effect is
    seen. The ratio is capped at 2.9, below the 3 that an infinite read-out
    gives."""
    use = [
        r
        for r in rows
        if "error" not in r
        and r.get("sharp_upper") is not None
        and (r.get("sharp") or 0) >= min_hits
        and r["meta"].get("exposure_s")
    ]
    if len(use) < 5:
        return None, None, [f"only {len(use)} frames with {min_hits}+ hits"]
    up = sum(r["sharp_upper"] for r in use)
    tot = sum(r["sharp"] for r in use)
    lo = tot - up
    first = "bottom"
    if up < lo:
        up, lo, first = lo, up, "top"
    ratio = up / max(lo, 1)
    if ratio - 1 < 2 * ratio * np.sqrt(1 / max(up, 1) + 1 / max(lo, 1)):
        return None, None, [f"hits split evenly over {len(use)} frames (ratio {ratio:.2f})"]
    ratio = min(ratio, 2.9)
    t = float(np.median([r["meta"]["exposure_s"] for r in use]))
    ro = t * (ratio - 1) / (0.75 - ratio / 4)
    return (
        round(ro, 3),
        first,
        [
            f"{len(use)} frames with {min_hits}+ hits: {ratio:.2f}x more in the "
            f"half read last ({first} read first) -> read-out {ro:.2f} s on {t:.2f} s exposures"
        ],
    )


def _bg_residual(d, size=7):
    """The frame minus a local median, with sources and outliers zeroed,
    and each row's median removed (row offsets are a separate setting)."""
    r = d - ndimage.median_filter(d, size=size)
    s = mad_sigma(r, 0.0) + 1e-9
    r = np.where(np.abs(r) < 4 * s, r, 0.0)
    return r - np.median(r, axis=1, keepdims=True)


def serial_lag(files, sat=None, x0=0, max_frames=80):
    """A single-pole low-pass along rows in the read-out chain (Skytwin's
    `noise.serial_lag`): background noise correlated along rows as aᵏ at
    lag k, but not down columns, and sources trailing to one side. The
    pole is the ratio of the lag-2 to the lag-1 correlation (white noise
    added after the filter lowers both alike); the side is where bright
    isolated stars carry more flux: the share of flux 1-3 px left minus
    right of the peak, from bright resolved stars (single hot pixels have
    neither) against a 15 px median along the row. Returns (block or None,
    notes)."""
    rat, l1s, cols, sides = [], [], [], []
    for f in files[:max_frames]:
        d = fits.getdata(f).astype(np.float64)[:, x0:]
        r = _bg_residual(d)
        v = float((r * r).mean())
        if v <= 0:
            continue
        l1 = float((r[:, 1:] * r[:, :-1]).mean()) / v
        l2 = float((r[:, 2:] * r[:, :-2]).mean()) / v
        c1 = float((r[1:, :] * r[:-1, :]).mean()) / v
        l1s.append(l1)
        cols.append(c1)
        if l1 > 0.05:
            rat.append(l2 / l1)
        loc = d - ndimage.median_filter(d, size=(1, 15))
        sig = mad_sigma(loc, 0.0) + 1e-9
        pk = (loc == ndimage.maximum_filter(loc, 7)) & (loc > 12 * sig)
        if sat is not None:
            pk &= d < 0.8 * sat
        for y, x in zip(*np.nonzero(pk)):
            if 5 <= x < d.shape[1] - 5:
                row = loc[y]
                if row[x - 1] + row[x + 1] < 0.3 * row[x]:
                    continue
                sides.append((row[x - 3 : x].sum() - row[x + 1 : x + 4].sum()) / row[x])
    if not l1s:
        return None, ["no frames"]
    l1, c1 = float(np.median(l1s)), float(np.median(cols))
    if len(rat) < 5 or l1 < 0.1 or l1 < 2 * abs(c1):
        return None, [f"noise along rows not correlated (lag-1 {l1:.2f}, down columns {c1:.2f})"]
    a = float(np.clip(np.median(rat), 0.0, 0.95))
    side = float(np.median(sides)) if sides else 0.0
    toward = "left" if side >= 0 else "right"
    return {"coefficient": round(a, 3), "toward": toward}, [
        f"noise correlation along rows: lag-1 {l1:.2f}, lag-2/lag-1 {a:.2f} (down columns {c1:.2f}) "
        f"in {len(l1s)} frames; sources trail {toward} (median left-minus-right flux {side:+.2f} of the peak, "
        f"{len(sides)} stars)"
    ]


def lag_variance_factor(a):
    """Variance of white noise after the single-pole filter, over before."""
    return (1.0 - a) / (1.0 + a)


def _blocks(a, block):
    """The `block` × `block` tiles of 2-D array `a` (partial tiles at the edges left out), one per row."""
    ny, nx = a.shape[0] // block, a.shape[1] // block
    t = a[: ny * block, : nx * block].reshape(ny, block, nx, block).transpose(0, 2, 1, 3)
    return t.reshape(ny * nx, block * block)


def clipped_variance(x, k=4.0, iterations=3):
    """Variance of each row of `x` after clipping at `k` σ about the mean, iterated from the MAD, and
    divided by the variance a Gaussian keeps inside ±k σ. Unlike the MAD, it is not stepped on integer
    data: a MAD of 9 or 10 ADU is a 10% step in σ, which biased photon-transfer gains by 15%."""
    x = np.asarray(x, dtype=np.float64)
    c = np.median(x, axis=1, keepdims=True)
    s = mad_sigma(x, axis=1, keepdims=True)
    s = np.where(s > 0, s, x.std(axis=1, keepdims=True) + 1e-12)
    for _ in range(iterations):
        m = np.abs(x - c) < k * s
        n = m.sum(axis=1, keepdims=True)
        c = (x * m).sum(axis=1, keepdims=True) / np.maximum(n, 1)
        s = np.sqrt((((x - c) * m) ** 2).sum(axis=1, keepdims=True) / np.maximum(n - 1, 1))
    kept = 1.0 - 2.0 * k * np.exp(-0.5 * k * k) / np.sqrt(2.0 * np.pi) / (1.0 - erfc(k / np.sqrt(2.0)))
    return (s[:, 0] ** 2) / kept


def photon_transfer(files, times, x0=0, lag=0.0, max_pairs=400, max_dt_s=None, block=16, central=0.5, exposures=None):
    """Gain and read noise from pairs of consecutive frames with the same
    exposure (when `exposures` are given): in blocks of each pair, the
    variance of the difference (halved) against the mean level, variance =
    level / gain + read². The difference removes stars, fixed patterns and
    the flat; row medians are removed (row offsets); the serial lag's
    variance reduction is undone. Only the central `central` fraction of
    each axis is used. Levels outside p2-p98 are dropped; the line is fitted
    to medians in 20 equal-count level bins, which stars that moved do not
    shift.

    Pairs are at most `max_dt_s` apart; by default 1.5 times the median
    spacing of consecutive frames, at least 5 s and at most 60 s, so an
    archive taken every 7 s still pairs. The intercept is the read variance
    when the frames are bias-subtracted. A negative intercept means they are
    not (variance = (level - bias) / gain + read²): the read noise then comes
    from the darkest level bin, an upper bound if those frames hold light.
    Returns (gain e/ADU, read ADU, notes) or (None, None, notes)."""
    if exposures is None:
        exposures = [None] * len(files)
    items = sorted(
        [(utc_timestamp(s), f, e) for f, s, e in zip(files, times, exposures) if s and utc_timestamp(s) is not None]
    )
    if max_dt_s is None:
        gaps = [tb - ta for (ta, _, _), (tb, _, _) in pairwise(items) if tb > ta]
        max_dt_s = min(60.0, max(5.0, 1.5 * float(np.median(gaps)))) if gaps else 5.0
    pts = []
    pairs = 0
    fac = lag_variance_factor(lag)
    for (ta, a, ea), (tb, b, eb) in pairwise(items):
        if not (0 < tb - ta <= max_dt_s) or pairs >= max_pairs:
            continue
        if ea and eb and abs(ea - eb) > 0.01 * max(ea, eb):
            continue
        A = fits.getdata(a).astype(np.float64)[:, x0:]
        B = fits.getdata(b).astype(np.float64)[:, x0:]
        if A.shape != B.shape:
            continue
        pairs += 1
        ny0, nx0 = A.shape
        cy, cx = int(ny0 * (1 - central) / 2), int(nx0 * (1 - central) / 2)
        A, B = A[cy : ny0 - cy, cx : nx0 - cx], B[cy : ny0 - cy, cx : nx0 - cx]
        D = A - B
        D = D - np.median(D, axis=1, keepdims=True)
        M = 0.5 * (A + B)
        Db, Mb = _blocks(D, block), _blocks(M, block)
        pts.extend(zip(np.median(Mb, axis=1).tolist(), (clipped_variance(Db) / 2 / fac).tolist()))
    if pairs < 5 or len(pts) < 50:
        return None, None, [f"only {pairs} frame pairs within {max_dt_s:.0f} s"]
    p = np.array(pts)
    lo, hi = np.percentile(p[:, 0], [2, 98])
    p = p[(p[:, 0] >= lo) & (p[:, 0] <= hi)]
    if hi - lo < 10:
        return None, None, [f"levels span only {hi - lo:.1f} ADU in {pairs} pairs"]
    edges = np.quantile(p[:, 0], np.linspace(0, 1, 21))
    L, V = [], []
    for e0, e1 in pairwise(edges):
        m = (p[:, 0] >= e0) & (p[:, 0] <= e1)
        if m.sum() > 10:
            L.append(np.median(p[m, 0]))
            V.append(np.median(p[m, 1]))
    slope, icpt = np.polyfit(L, V, 1)
    if slope <= 0:
        return None, None, [f"variance does not grow with level ({pairs} pairs)"]
    gain = 1.0 / slope
    note = (
        f"{pairs} frame pairs up to {max_dt_s:.1f} s apart, {len(p)} blocks, levels {lo:.0f} to {hi:.0f} ADU: "
        f"variance = level / {gain:.2f} + {icpt:.2f}" + (f" (serial lag {lag:.2f} undone)" if lag else "")
    )
    if icpt >= 0:
        read = float(np.sqrt(icpt))
    else:
        read = float(np.sqrt(max(V[0], 0.0)))
        note += (
            f"; the intercept is negative, so the frames are not bias-subtracted: read noise from the darkest "
            f"level bin ({L[0]:.0f} ADU), an upper bound if those frames hold light"
        )
    return float(gain), read, [note]


def field_stop(b, x0=0, axis=None):
    """A circular field stop in a response map `b` (from `flat`): fits
    centre, radius, edge blur and leak beyond it, with a radial polynomial
    inside. Kept when the response beyond the edge is under half of
    inside and the circle lies mostly within the frame. The fit starts at the
    radius where the azimuthal median response falls through half. Returns
    (block or None, notes)."""
    h, w = b.shape
    ok = np.isfinite(b)
    if ok.mean() < 0.5:
        return None, ["response map mostly undefined"]
    inner = np.nanmedian(b[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3])
    if not inner > 0:
        return None, ["no response at the centre"]
    bn = np.where(ok, b / inner, np.nan)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx += x0
    cx0, cy0 = axis or ((w + x0 - 1) / 2, (h - 1) / 2)
    r0 = np.hypot(xx - cx0, yy - cy0)
    prof = [
        np.nanmedian(bn[(r0 >= k) & (r0 < k + 1)]) if ((r0 >= k) & (r0 < k + 1)).any() else np.nan
        for k in range(int(r0.max()))
    ]
    below = [k for k, v in enumerate(prof) if k > 10 and np.isfinite(v) and v < 0.5]
    if not below:
        return None, ["no edge: the response stays above half out to the corners"]
    R0 = float(below[0])
    sel = ok & (r0 < R0 + 25)
    X, Y, B = xx[sel], yy[sel], bn[sel]

    def model(p, X, Y):
        cx, cy, R, s, out, a2 = p
        r = np.hypot(X - cx, Y - cy)
        t = out + (1 - out) * 0.5 * erfc((r - R) / (np.sqrt(2) * abs(s)))
        return t * (1 + a2 * (r / R) ** 2)

    res = least_squares(
        lambda p: model(p, X, Y) - B,
        [cx0, cy0, R0, 2.0, 0.05, 0.0],
        loss="soft_l1",
        f_scale=0.05,
        bounds=([cx0 - 30, cy0 - 30, 5, 0.2, 0, -0.9], [cx0 + 30, cy0 + 30, 2 * R0 + 10, 30, 0.6, 2]),
    )
    cx, cy, R, s, out, _ = res.x
    frac_in = np.mean((np.hypot(xx - cx, yy - cy) < R)[ok])
    if out > 0.5 or frac_in < 0.3:
        return None, [f"edge fit not a field stop (leak {out:.2f}, {frac_in:.0%} of the frame inside)"]
    blk = {
        "centre_px": [round(float(cx), 2), round(float(cy), 2)],
        "radius_px": round(float(R), 2),
        "edge_px": round(float(abs(s)), 2),
        "outside": round(float(out), 3),
    }
    return blk, [
        f"field stop centred ({cx:.1f}, {cy:.1f}), radius {R:.1f} px, edge σ {abs(s):.1f} px, "
        f"{out:.1%} response beyond it"
    ]


def unlag(d, a, toward="left"):
    """Undo Skytwin's `noise.serial_lag` exactly: in read order, in = (out −
    a · previous out) / (1 − a)."""
    d = np.asarray(d, dtype=np.float64)
    if toward == "left":
        d = d[:, ::-1]
    prev = np.concatenate([d[:, :1], d[:, :-1]], axis=1)
    u = (d - a * prev) / (1.0 - a)
    return u[:, ::-1] if toward == "left" else u


def unlag_files(files, lag, work):
    """Copies of `files` (float32) with the serial lag undone, for PSF work."""
    os.makedirs(work, exist_ok=True)
    out = []
    for i, f in enumerate(files):
        with fits.open(f) as hl:
            d = unlag(hl[0].data, lag["coefficient"], lag.get("toward", "left"))
            p = os.path.join(work, f"u{i:05d}.fits")
            fits.PrimaryHDU(d.astype(np.float32), hl[0].header).writeto(p, overwrite=True)
            out.append(p)
    return out


def stop_masks(shape, stop, x0=0):
    """(inside, just-inside annulus, just-outside annulus) of a field stop,
    clear of its blurred edge."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx + x0 - stop["centre_px"][0], yy - stop["centre_px"][1])
    R, e = stop["radius_px"], max(stop["edge_px"], 1.0)
    inside = r < R - 3 * e
    ring_in = (r >= R - 3 * e - 12) & (r < R - 3 * e)
    ring_out = (r > R + 3 * e) & (r <= R + 3 * e + 15)
    return inside, ring_in, ring_out


def sky_split(rows, stop, x0=0):
    """Splits each frame's level into light (through the stop) and an added
    level (dark residue, read-out offsets: not through it), from the
    medians of rings just inside and just outside the stop, where the
    response is 1 and `outside`. Rings around the whole circle cancel
    linear ramps. Adds `sky_adu` (light level at full response) and
    `add_adu` to each row; returns notes."""
    out = stop.get("outside", 0.0)
    n = 0
    for r in rows:
        if "error" in r:
            continue
        d = fits.getdata(r["file"]).astype(np.float64)[:, x0:]
        _, ri, ro = stop_masks(d.shape, stop, x0)
        if ro.sum() < 50:
            continue
        li, lo = float(np.median(d[ri])), float(np.median(d[ro]))
        sky = max((li - lo) / max(1.0 - out, 1e-3), 0.0)
        r["sky_adu"] = sky
        r["add_adu"] = lo - sky * out
        n += 1
    if not n:
        return ["no frames with enough pixels beyond the stop"]
    s = np.array([r["sky_adu"] for r in rows if "sky_adu" in r])
    a = np.array([r["add_adu"] for r in rows if "add_adu" in r])
    return [
        f"{n} frames: light through the stop median {np.median(s):.1f} ADU (p10 {np.percentile(s, 10):.1f}, "
        f"p90 {np.percentile(s, 90):.1f}); added level median {np.median(a):.1f} ADU"
    ]


def refield(rows, stop, x0=0):
    """Smooth-field statistics again, from inside the stop only (its edge
    is not background structure)."""
    for r in rows:
        if "error" in r:
            continue
        d = fits.getdata(r["file"]).astype(np.float64)
        inside, _, _ = stop_masks(d[:, x0:].shape, stop, x0)
        r["field"] = measure.field(d, x0, r["median"], max(r["noise"], 1.0), valid=inside)


def transient_sharp(rows, files_fixed, x0=0, max_dt_s=5.0, near_px=2):
    """Radiation hits only: sharp events that are neither fixed (hot
    pixels: high in the median over frames) nor present in a frame taken
    within `max_dt_s` (stars, which stay). Rewrites `sharp` and
    `sharp_upper` in rows that have a neighbour in time; returns notes.
    A neighbour's sources are its pixels 5σ over its background, grown by
    `near_px`."""
    fixed = _fixed_pixels(files_fixed)

    def t(r):
        return utc_timestamp((r.get("meta") or {}).get("time_utc"))

    ok = sorted(
        [(t(r), i) for i, r in enumerate(rows) if "error" not in r and r.get("sharp") is not None and t(r) is not None]
    )
    before = after = n = 0
    for k, (ti, i) in enumerate(ok):
        nb = [j for tj, j in ok[max(0, k - 1) : k + 2] if j != i and abs(tj - ti) <= max_dt_s]
        if not nb:
            continue
        r = rows[i]
        d = fits.getdata(r["file"]).astype(np.float64)[:, x0:]
        bad = measure.bad_rows(d, r["noise"])
        ev = measure.sharp_mask(d, r["median"], r["noise"], bad)
        if fixed is not None and fixed[:, x0:].shape == ev.shape:
            ev &= ~fixed[:, x0:]
        for j in nb:
            o = fits.getdata(rows[j]["file"]).astype(np.float64)[:, x0:]
            lit = o > rows[j]["median"] + 5 * max(rows[j]["noise"], 1.0)
            ev &= ~ndimage.binary_dilation(lit, iterations=near_px)
        before += r["sharp"]
        r["sharp"] = int(ev.sum())
        r["sharp_upper"] = int(ev[ev.shape[0] // 2 :].sum())
        after += r["sharp"]
        n += 1
    if not n:
        return ["no frames with a neighbour in time: sharp events kept as counted"]
    return [
        f"{n} frames with a neighbour within {max_dt_s:.0f} s: {before} sharp events -> {after} transient "
        f"(hot pixels and sources seen in the neighbour removed)"
    ]


def flat_model(flat, shape, axis=None):
    """Smooth part of Skytwin's flat (vignetting × field stop), as a map of `shape`."""
    h, w = shape
    ax = axis or ((w - 1) / 2, (h - 1) / 2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.ones(shape)
    if flat and flat.get("vignetting"):
        a2, a4 = flat["vignetting"]
        r2 = ((xx - ax[0]) ** 2 + (yy - ax[1]) ** 2) / (0.5 * np.hypot(w, h)) ** 2
        out *= np.clip(1 + a2 * r2 + a4 * r2 * r2, 0, None)
    fs = (flat or {}).get("field_stop")
    if fs:
        c = fs.get("centre_px") or ax
        r = np.hypot(xx - c[0], yy - c[1])
        o = fs.get("outside", 0.0)
        out *= o + (1 - o) * 0.5 * erfc((r - fs["radius_px"]) / (np.sqrt(2) * max(fs.get("edge_px", 1.0), 1e-6)))
    return out


def fixed_field(rows, flat, x0=0, block=8, iters=4, max_frames=3000):
    """Additive structure that is the same in every frame (Skytwin's
    `noise.fixed_field`), fitted jointly with each frame's pedestal and
    light: block medians b_i = p_i + F + s_i · flat. F is the median over
    frames of b_i − p_i − s_i flat; p_i and s_i a per-frame regression on
    b_i − F. F is kept orthogonal to a constant and to the flat's shape, so
    common light stays in s and offsets in p. Rewrites `add_adu`,
    `sky_adu`, `ramp_x` and `ramp_y` (ramps become deviations from F) in
    the rows used; returns (block or None, notes). Per-frame ramps are
    fitted about F."""
    use = spread([r for r in rows if "error" not in r], max_frames)
    if len(use) < 20:
        return None, ["fewer than 20 frames"]
    B = []
    for r in use:
        d = fits.getdata(r["file"]).astype(np.float64)[:, x0:]
        h, w = d.shape
        ny, nx = h // block, w // block
        B.append(np.median(d[: ny * block, : nx * block].reshape(ny, block, nx, block), axis=(1, 3)))
    B = np.array(B)
    fm = flat_model(flat, (h, w + x0))[:, x0:]
    fb = fm[: ny * block, : nx * block].reshape(ny, block, nx, block).mean(axis=(1, 3))
    n = len(use)
    p = np.array([r.get("add_adu", r["median"]) for r in use], float)
    s = np.array([r.get("sky_adu", 0.0) for r in use], float)
    one, fv = np.ones(fb.size), fb.ravel()
    Q, _ = np.linalg.qr(np.c_[one, fv])
    F = np.zeros_like(fb)
    for _ in range(iters):
        R = B - p[:, None, None] - s[:, None, None] * fb[None]
        F = np.median(R, axis=0)
        f = F.ravel()
        F = (f - Q @ (Q.T @ f)).reshape(F.shape)
        A = np.c_[one, fv]
        for i in range(n):
            (p[i], s[i]), *_ = np.linalg.lstsq(A, (B[i] - F).ravel(), rcond=None)
    s = np.maximum(s, 0.0)
    cy = (np.arange(ny) + 0.5) * block
    cx = (np.arange(nx) + 0.5) * block + x0
    YY, XX = np.meshgrid(cy, cx, indexing="ij")
    P = np.c_[np.ones(YY.size), XX.ravel() / w, YY.ravel() / h]
    for i, r in enumerate(use):
        res = (B[i] - F - p[i] - s[i] * fb).ravel()
        c, *_ = np.linalg.lstsq(P, res, rcond=None)
        r["add_adu"], r["sky_adu"] = float(p[i]), float(s[i])
        r["ramp_x"], r["ramp_y"] = float(c[1]), float(c[2])
    rms = float(np.std(F))
    blk = {
        "origin_px": [float(x0 + block / 2 - 0.5), float(block / 2 - 0.5)],
        "spacing_px": float(block),
        "values": [[round(float(v), 3) for v in row] for row in F],
    }
    resid = B - p[:, None, None] - F[None] - s[:, None, None] * fb[None]
    return blk, [
        f"{n} frames, {ny}×{nx} grid every {block} px: fixed structure {rms:.2f} ADU RMS "
        f"(range {F.min():.1f} to {F.max():.1f}); per-frame light median {np.median(s):.1f} ADU, "
        f"pedestal median {np.median(p):.1f} ADU; block residual {mad_sigma(resid, 0.0):.2f} ADU"
    ]
