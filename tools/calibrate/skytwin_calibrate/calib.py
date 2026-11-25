"""Detector properties from calibration frames: the bias, dark and flat frames taken with the science frames.

`skytwin-calibrate learn --calibration DIR` sorts the frames by their header frame type (`IMAGETYP` and
kin), else by the name of the file or the nearest folder, and keeps those of the science frames' size and
binning (flats are often taken at another binning). Frames more than 8 hours apart fall in different nights.

- `read_noise`: half the clipped variance of the difference of consecutive bias frames over the central
  half (the fixed bias structure cancels), and a master bias per night.
- `flat_gain`: photon transfer from pairs of flats with the same exposure, the second scaled to the first's
  level so a drifting lamp or twilight sky adds no variance; half the difference's clipped variance against
  the bias-subtracted level, fitted as level / gain + read² with the read noise from the bias pairs.
- `dark_current`: the rate map from master darks of each night, the longest exposure less the shortest
  (bias and read-out wait cancel) or, with one exposure, less the master bias; its clipped median is the
  dark current at the darks' temperature, and `hot_from_darks` finds the pixels far above it.
- `cosmic_rays`: each dark less the median dark of its night and exposure leaves read noise and transient
  events. Groups of pixels 8σ above zero are hits, less those at a pixel that fires in two or more darks
  (flickering pixels). Counts against exposure give the rate (slope) and the hits during read-out
  (intercept); Skytwin collects hits for the exposure plus half the read-out time on average, so the
  read-out time is twice the intercept over the slope.
- `master_flat` and `dust`: ring-shaped shadows of dust in the master flat, found by matched filters of
  annuli over a range of radii and measured from each one's radial profile.
"""

import os
import re
from collections import Counter, defaultdict

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy import ndimage, signal

from . import keywords
from .analyse import clipped_variance
from .util import mad_sigma, percentiles, spread, utc_timestamp

TYPE_KEYS = ("IMAGETYP", "IMAGETYPE", "FRAMETYP", "FRAMETYPE", "OBSTYPE", "EXPTYPE", "FRAME")
KINDS = (("bias", r"bias|zero"), ("dark", r"dark"), ("flat", r"flat"), ("light", r"light|object|science"))
NIGHT_GAP_S = 8 * 3600
RADII_PX = (22, 26, 30, 34, 40, 48, 58, 70, 85, 100, 120, 140)
"""Outer radii of the annulus filters for dust shadows, image pixels."""


def kind_of(path, h):
    """'bias', 'dark', 'flat', 'light' or None: the header's frame type, else the file name or the nearest
    folder that names one of the first three."""
    for k in TYPE_KEYS:
        v = h.get(k)
        if isinstance(v, str):
            for kind, pat in KINDS:
                if re.search(pat, v, re.IGNORECASE):
                    return kind
    for part in reversed(os.path.normpath(path).split(os.sep)):
        for kind, pat in KINDS[:3]:
            if re.search(pat, part, re.IGNORECASE):
                return kind
    return None


def sort_frames(files, km, shape, shape_of):
    """Calibration frames by kind, {'bias': [...], 'dark': [...], 'flat': [...]}, and notes.

    Each frame is a dict of `path`, `exposure_s` (0 when absent), `t` (POSIX seconds or None),
    `temperature_c` and `night` (frames more than `NIGHT_GAP_S` apart start a new night; frames without a
    time are night 0), sorted by time. Only frames with `shape_of(header) == shape` are kept.
    """
    out = {"bias": [], "dark": [], "flat": []}
    other, light, unknown = Counter(), 0, 0
    recs = []
    for f in files:
        try:
            h = fits.getheader(f)
        except Exception:
            continue
        kind = kind_of(f, h)
        if kind == "light":
            light += 1
            continue
        if kind is None:
            unknown += 1
            continue
        if shape_of(h) != shape:
            other[kind] += 1
            continue
        m = keywords.frame_meta(h, km)
        t = utc_timestamp(m["time_utc"]) if m["time_utc"] else None
        recs.append(
            {
                "kind": kind,
                "path": f,
                "exposure_s": m["exposure_s"] or 0.0,
                "t": t,
                "temperature_c": m["temperature_c"],
            }
        )
    recs.sort(key=lambda r: (r["t"] is None, r["t"] or 0.0, r["path"]))
    night, last = 0, None
    for r in recs:
        if r["t"] is not None:
            if last is not None and r["t"] - last > NIGHT_GAP_S:
                night += 1
            last = r["t"]
        r["night"] = night if r["t"] is not None else 0
        out[r.pop("kind")].append(r)
    nights = len({r["night"] for v in out.values() for r in v})
    note = ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f" frames over {nights} night(s)"
    if other:
        note += "; " + ", ".join(f"{n} {k}" for k, n in other.items()) + " of another size or binning left out"
    if light or unknown:
        note += f"; {light + unknown} frames not bias, dark or flat ignored"
    return out, [note]


def _load(path, x0=0):
    """Frame at `path` as float32, without its first `x0` columns."""
    return fits.getdata(path).astype(np.float32)[:, x0:]


def _centre(a, central=0.5):
    """The central `central` fraction of each axis of `a`."""
    h, w = a.shape
    cy, cx = int(h * (1 - central) / 2), int(w * (1 - central) / 2)
    return a[cy : h - cy, cx : w - cx]


def _var(a):
    """Clipped variance of all of `a` (stars, hits and hot pixels left out)."""
    return float(clipped_variance(np.asarray(a, np.float64).reshape(1, -1))[0])


def _master(recs, x0=0, n=15):
    """Median of up to `n` of the frames `recs`."""
    return np.median(np.stack([_load(r["path"], x0) for r in spread(recs, n)]), axis=0)


def _by(recs, key):
    """`recs` grouped by `key(rec)`, in first-seen order."""
    out = defaultdict(list)
    for r in recs:
        out[key(r)].append(r)
    return out


def bias_for(masters, night):
    """The master bias of `night`, else the median of all nights' masters, else None."""
    if night in masters:
        return masters[night]
    if not masters:
        return None
    if "all" not in masters:
        masters["all"] = np.median(np.stack([m for k, m in masters.items() if k != "all"]), axis=0)
    return masters["all"]


def read_noise(bias, x0=0, max_pairs=20):
    """(read noise ADU or None, {night: master bias}, notes) from consecutive pairs of bias frames, at most
    `max_pairs` a night: half the clipped variance of their difference over the central half."""
    v, masters, levels = [], {}, []
    for night, recs in _by(bias, lambda r: r["night"]).items():
        masters[night] = _master(recs, x0)
        levels.append(float(np.median(_centre(masters[night]))))
        prev = None
        for r in recs[: max_pairs + 1]:
            d = _load(r["path"], x0)
            if prev is not None:
                v.append(_var(_centre(d - prev)) / 2)
            prev = d
    if not v:
        return None, masters, [f"{len(bias)} bias frame(s): no pairs for the read noise"]
    rn = float(np.sqrt(np.median(v)))
    lv = ", ".join(f"{x:.1f}" for x in levels)
    return rn, masters, [f"read noise {rn:.2f} ADU from {len(v)} pairs of bias frames; bias level {lv} ADU"]


def flat_gain(flats, masters, read_adu=None, sat=None, top=65535, x0=0, max_pairs=60):
    """(gain e/ADU or None, notes) by photon transfer from pairs of flats.

    Consecutive flats of one night with the same exposure pair up (each flat in one pair at most). With a
    master bias, both are bias-subtracted, the second scaled to the first's level, and the halved clipped
    variance of their difference over the central half is fitted as level / gain + read², the read noise
    fixed at `read_adu` from the bias pairs; without one, slope and intercept are both fitted on the raw
    levels (the slope does not need the bias). Pairs above 60% of `sat` (or of `top`) are left out, where
    the full well and the converter bend the curve.
    """
    limit = 0.6 * (sat or top)
    pts, exps = [], set()
    i = 0
    while i + 1 < len(flats) and len(pts) < max_pairs:
        a, b = flats[i], flats[i + 1]
        ea, eb = a["exposure_s"], b["exposure_s"]
        if a["night"] != b["night"] or abs(ea - eb) > 0.01 * max(ea, eb, 1e-9):
            i += 1
            continue
        i += 2
        A, B = _centre(_load(a["path"], x0)), _centre(_load(b["path"], x0))
        if 0.5 * (np.median(A) + np.median(B)) > limit:
            continue
        bias = bias_for(masters, a["night"])
        if bias is not None:
            bc = _centre(bias)
            A, B = A - bc, B - bc
        ma, mb = float(np.median(A)), float(np.median(B))
        if ma <= 0 or mb <= 0:
            continue
        pts.append((0.5 * (ma + mb), _var(A - B * (ma / mb)) / 2))
        exps.add(ea)
    if len(pts) < 3:
        return None, [f"only {len(pts)} flat pairs below {limit:.0f} ADU"]
    P = np.array(pts)
    if read_adu is not None and masters:
        y = P[:, 1] - read_adu**2
        slope = float(np.sum(P[:, 0] * y) / np.sum(P[:, 0] ** 2))
        icpt = read_adu**2
        how = f"read² {icpt:.1f} from the bias pairs"
    else:
        slope, icpt = (float(c) for c in np.polyfit(P[:, 0], P[:, 1], 1))
        how = f"intercept {icpt:.1f} fitted (no bias frames, levels include the bias)"
    if slope <= 0:
        return None, [f"variance does not grow with level over {len(P)} flat pairs"]
    gain = 1.0 / slope
    rms = float(np.std(P[:, 1] - (P[:, 0] * slope + icpt)) / np.mean(P[:, 1]))
    ex = ", ".join(f"{e:g}" for e in sorted(exps))
    return gain, [
        f"{len(P)} flat pairs at {ex} s, levels {P[:, 0].min():.0f} to {P[:, 0].max():.0f} ADU: "
        f"variance = level / {gain:.3f} + read² ({how}); scatter {rms:.1%}"
    ]


def dark_current(darks, masters, gain, x0=0, min_frames=3):
    """(rate map e/s or None, dark e/s, temperature °C or None, notes).

    Per night, a master dark (median) of each exposure with at least `min_frames` frames; the rate is the
    longest exposure less the shortest over their difference when the shortest is under 70% of the longest
    (the bias and the charge a row collects while it waits to be read both cancel), else the longest less
    the night's master bias. Nights are combined by their median. The dark current is the 4σ-clipped
    median of the map.
    """
    maps, hows, fixed = [], set(), []
    for night, recs in _by(darks, lambda r: r["night"]).items():
        groups = {e: g for e, g in _by(recs, lambda r: round(r["exposure_s"], 3)).items() if e > 0}
        exps = sorted(e for e, g in groups.items() if len(g) >= min_frames)
        if not exps:
            continue
        tl = exps[-1]
        if exps[0] < 0.7 * tl:
            ts = exps[0]
            ml = _master(groups[tl], x0)
            maps.append((ml - _master(groups[ts], x0)) / (tl - ts))
            hows.add(f"{tl:g} s less {ts:g} s")
            bias = bias_for(masters, night)
            if bias is not None:
                fixed.append(float(np.median(ml - bias - maps[-1] * tl)))
        else:
            bias = bias_for(masters, night)
            if bias is None:
                continue
            maps.append((_master(groups[tl], x0) - bias) / tl)
            hows.add(f"{tl:g} s less the master bias")
    if not maps:
        return None, None, None, [f"{len(darks)} darks: no exposure with {min_frames}+ frames to rate (or no bias)"]
    rate = (np.median(np.stack(maps), axis=0) if len(maps) > 1 else maps[0]) * gain
    _, med, _ = sigma_clipped_stats(rate, sigma=4.0, maxiters=5)
    temps = [r["temperature_c"] for r in darks if r["temperature_c"] is not None]
    tc = float(np.median(temps)) if temps else None
    return (
        rate,
        float(med),
        tc,
        [
            f"dark current {med:.4f} e/s"
            + (f" at {tc:.1f} C" if tc is not None else "")
            + f" from master darks of {len(maps)} night(s) ({'; '.join(sorted(hows))})"
            + (
                f"; {np.median(fixed) * gain:.2f} e per dark above the bias does not grow with exposure (a bias "
                "shift or charge collected while waiting to be read) and is left out"
                if fixed
                else ""
            )
        ],
    )


def hot_from_darks(rate, dark, threshold=None):
    """Skytwin's `hot_pixels` block (density per Mpix of the image, rates e/s above the dark current) from
    a dark-rate map, and the threshold used: pixels above `threshold` e/s over `dark`, and never under 6
    times the map's MAD σ, where the masters' noise would pass for hot pixels."""
    exc = rate - dark
    floor = 6.0 * float(mad_sigma(exc))
    thr = max(threshold, floor) if threshold is not None else floor
    vals = np.sort(exc[exc > thr])
    if len(vals) < 5:
        return None, thr
    return {
        "density_per_mpix": round(len(vals) / exc.size * 1e6, 2),
        "e_per_s_at_ref": {"empirical": percentiles(vals, 3)},
    }, thr


def cosmic_rays(darks, area_cm2, x0=0, k=8.0, min_frames=3, max_frames=25):
    """Radiation from darks: ({rate_per_cm2_s, readout_s, first_row, measured_readout} or None, notes).

    Hits are counted in each dark of a night-and-exposure group of at least `min_frames` (up to
    `max_frames`), less the group's median. Hits whose peak lies within 1 px of a peak in another dark are
    a flickering pixel, not radiation, and are dropped. With two or more exposures the counts are fitted
    as intercept + slope × exposure: the rate is the slope over `area_cm2` (the unbinned chip), the
    read-out time twice the intercept over the slope when the intercept is 2σ above zero, and the first row
    read is the half with fewer hits when they differ by 2σ (Poisson). With one exposure the rate is the
    median count over the exposure, read-out hits included.
    """
    peaks, count = [], None
    for (_, t), recs in _by(darks, lambda r: (r["night"], round(r["exposure_s"], 3))).items():
        if t <= 0 or len(recs) < min_frames:
            continue
        stack = np.stack([_load(r["path"], x0) for r in spread(recs, max_frames)])
        med = np.median(stack, axis=0)
        for d in stack:
            res = d - med
            sig = float(mad_sigma(res[::2, ::2])) + 1e-6
            lab, n = ndimage.label(res > k * sig, structure=np.ones((3, 3)))
            pos = np.array(ndimage.maximum_position(res, lab, range(1, n + 1)), int).reshape(-1, 2)
            peaks.append((t, pos))
            m = np.zeros(d.shape, bool)
            m[pos[:, 0], pos[:, 1]] = True
            if count is None:
                count = np.zeros(d.shape, np.uint16)
            count += ndimage.binary_dilation(m, structure=np.ones((3, 3)))
    if len(peaks) < 5:
        return None, [f"only {len(peaks)} darks in groups of {min_frames}+ of one night and exposure"]
    h = count.shape[0]
    T, N, up, lo, rec = [], [], 0, 0, 0
    for t, pos in peaks:
        again = count[pos[:, 0], pos[:, 1]] >= 2
        keep = pos[~again]
        T.append(t)
        N.append(len(keep))
        rec += int(again.sum())
        up += int((keep[:, 0] >= h / 2).sum())
        lo += int((keep[:, 0] < h / 2).sum())
    T, N = np.array(T, float), np.array(N, float)
    exps = sorted(set(T))
    note = (
        f"{int(N.sum())} hits in {len(N)} darks of {', '.join(f'{e:g}' for e in exps)} s "
        f"({rec} more at pixels that fire in two or more darks left out as flickering pixels)"
    )
    out = {"measured_readout": len(exps) >= 2}
    if len(exps) < 2:
        a = float(np.median(N / T))
        out["rate_per_cm2_s"] = round(a / area_cm2, 4)
        return out, [note + f": {a:.3f} hits/s, read-out hits included (one exposure)"]
    A = np.c_[np.ones_like(T), T]
    (b, a), *_ = np.linalg.lstsq(A, N, rcond=None)
    if a <= 0:
        return None, [note + ": counts do not grow with exposure"]
    s2 = float(np.sum((N - A @ [b, a]) ** 2) / max(len(N) - 2, 1))
    sb = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0]))
    out["rate_per_cm2_s"] = round(float(a) / area_cm2, 4)
    note += f": {a:.3f} hits/s on {area_cm2:.2f} cm2, {b:.2f} ± {sb:.2f} at zero exposure"
    if b > 2 * sb:
        out["readout_s"] = round(2 * float(b) / float(a), 2)
        note += f" -> read-out {out['readout_s']} s"
        if abs(up - lo) > 2 * np.sqrt(up + lo):
            out["first_row"] = "bottom" if up > lo else "top"
            note += f" ({up} hits in the upper half, {lo} in the lower: {out['first_row']} read first)"
    else:
        note += ": no read-out hits seen"
    return out, [note]


def master_flat(flats, masters, sat=None, top=65535, x0=0, n=40):
    """(normalised master flat or None, frames used): the median of up to `n` bias-subtracted flats, each
    divided by its median, keeping those whose median is 8% to 70% of `sat` (or of `top`)."""
    lo, hi = 0.08 * (sat or top), 0.7 * (sat or top)
    use = []
    for r in spread(flats, n):
        d = _load(r["path"], x0)
        bias = bias_for(masters, r["night"])
        if bias is not None:
            d = d - bias
        m = float(np.median(d))
        if lo < m < hi:
            use.append(d / m)
    if len(use) < 3:
        return None, len(use)
    return np.median(np.stack(use), axis=0), len(use)


def _poly_terms(u, v, degree=4):
    """Columns uⁱ vʲ for i + j ≤ `degree`."""
    return np.stack([u**i * v**j for i in range(degree + 1) for j in range(degree + 1 - i)], axis=-1)


def _background(F, block=32, size=5):
    """The large-scale level of `F`, everywhere: a 2-D polynomial of degree 4 fitted to the medians of
    `block` × `block` tiles (tiles 3σ off the fit, where shadows pull them down, are clipped), times the
    tiles' remaining structure median-filtered over `size` × `size` tiles and interpolated back. A median
    filter of the flat itself is biased on curved vignetting by up to a few tenths of a percent, as deep as
    the shadows sought."""
    H, W = F.shape
    ny, nx = H // block, W // block
    t = F[: ny * block, : nx * block].reshape(ny, block, nx, block).transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    t = np.median(t, axis=2)
    ty, tx = (np.arange(ny) + 0.5) * block - 0.5, (np.arange(nx) + 0.5) * block - 0.5
    sy, sx = 2.0 / H, 2.0 / W
    ty, tx = np.meshgrid(ty * sy - 1, tx * sx - 1, indexing="ij")
    A = _poly_terms(tx.ravel(), ty.ravel())
    ok = np.ones(t.size, bool)
    for _ in range(5):
        c, *_ = np.linalg.lstsq(A[ok], t.ravel()[ok], rcond=None)
        r = t.ravel() - A @ c
        ok = np.abs(r) < 3 * mad_sigma(r[ok]) + 1e-12
    rest = ndimage.median_filter(t / (A @ c).reshape(ny, nx), size=size, mode="nearest")
    yy, xx = np.meshgrid((np.arange(H) + 0.5) / block - 0.5, (np.arange(W) + 0.5) / block - 0.5, indexing="ij")
    rest = ndimage.map_coordinates(rest, [yy, xx], order=1, mode="nearest")
    v, u = np.meshgrid(np.arange(H) * sy - 1, np.arange(W) * sx - 1, indexing="ij")
    return (_poly_terms(u, v) @ c) * rest


def _ring(sm, x, y, R):
    """(outer radius, hole fraction, depth) of a shadow at (`x`, `y`) from its radial profile in 2 px bins of
    the smoothed residual `sm`: the depth is the level outside less the profile's minimum, the radii are
    where the profile, followed out and in from its minimum, crosses half depth. None when the depth is
    under 0.0007."""
    H, W = sm.shape
    rmax = int(2.2 * R) + 10
    y0, y1, x0, x1 = max(y - rmax, 0), min(y + rmax + 1, H), max(x - rmax, 0), min(x + rmax + 1, W)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(yy - y, xx - x)
    edges = np.arange(0, rmax, 2)
    inside = r < edges[-1] + 2
    idx = (r[inside] / 2).astype(int)
    sums = np.bincount(idx, sm[y0:y1, x0:x1][inside], len(edges))
    ns = np.bincount(idx, minlength=len(edges))
    prof = np.where(ns > 0, sums / np.maximum(ns, 1), np.nan)
    k = int(np.nanargmin(prof))
    far = edges > 1.6 * edges[k] + 10
    if not far.any():
        return None
    outside = float(np.nanmedian(prof[far]))
    depth = outside - float(prof[k])
    if not np.isfinite(depth) or depth < 0.0007:
        return None
    half = outside - depth / 2
    i = j = k
    while j + 1 < len(prof) and prof[j + 1] < half:
        j += 1
    while i > 0 and prof[i - 1] < half:
        i -= 1
    r_out = float(edges[j] + 2)
    return r_out, float(edges[i]) / r_out, depth


def dust(F, binning=(1, 1), margin=60, min_depth=0.0007, snr=5.0):
    """Skytwin's `flat.dust` block (chip pixels) from a normalised master flat `F`, or None, and notes.

    The flat over its large-scale level (`_background`) leaves the shadows. For each radius R in
    `RADII_PX` (up to an eighth of the image), the depth at each pixel is the mean deficit in an annulus
    (0.5 R to R) against the ring just outside it; local maxima over `min_depth` and `snr` σ, `margin` + R
    px clear of the edges, are candidates, kept deepest first unless they overlap one already kept. Each is
    then measured from its own radial profile (`_ring`). The hole fraction is the median of the deeper half
    of the shadows (noise fills the hole of a faint one). The density is over the area searched.
    """
    res = F / _background(F) - 1
    H, W = res.shape
    noise = float(mad_sigma(res))
    cands = []
    for R in (r for r in RADII_PX if r <= min(H, W) / 8):
        n = int(1.5 * R) + 2
        yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
        r = np.hypot(yy, xx)
        ann = (r <= R) & (r >= 0.5 * R)
        ring = (r > R + 2) & (r <= 1.4 * R + 2)
        kern = ann / ann.sum() - ring / ring.sum()
        depth = -signal.fftconvolve(res, kern[::-1, ::-1], mode="same")
        sig = noise * np.sqrt(1 / ann.sum() + 1 / ring.sum())
        pk = (depth == ndimage.maximum_filter(depth, size=R)) & (depth > min_depth) & (depth > snr * sig)
        for y, x in zip(*np.nonzero(pk)):
            if R + margin < x < W - R - margin and R + margin < y < H - R - margin:
                cands.append((float(depth[y, x]), int(x), int(y), R))
    cands.sort(reverse=True)
    kept = []
    for d, x, y, R in cands:
        if all(np.hypot(x - kx, y - ky) > 0.8 * (R + kR) for _, kx, ky, kR in kept):
            kept.append((d, x, y, R))
    sm = ndimage.gaussian_filter(res, 2)
    rings = [m for m in (_ring(sm, x, y, R) for _, x, y, R in kept) if m]
    if len(rings) < 3:
        return None, [f"{len(rings)} dust shadows found in the master flat (noise {noise:.2%})"]
    bx, by = binning
    scale = 0.5 * (bx + by)
    m = margin + float(np.median([R for *_, R in kept]))
    area = max(H - 2 * m, 1) * max(W - 2 * m, 1)
    radii = sorted(round(r * scale, 1) for r, _, _ in rings)
    depths = sorted(round(d, 4) for *_, d in rings)
    deep = [h for _, h, d in rings if d >= np.median(depths)]
    hole = round(float(np.median(deep)), 2)
    block = {
        "density_per_mpix": round(len(rings) / area * 1e6 / (bx * by), 2),
        "radius_px": {"empirical": radii},
        "depth": {"empirical": depths},
        "hole_fraction": hole,
    }
    q = np.quantile
    return block, [
        f"{len(rings)} dust shadows in the master flat (noise {noise:.2%}): outer radius "
        f"{q(radii, 0.1):.0f}/{q(radii, 0.5):.0f}/{q(radii, 0.9):.0f} chip px (p10/p50/p90), depth "
        f"{q(depths, 0.1):.4f}/{q(depths, 0.5):.4f}/{q(depths, 0.9):.4f}, hole {hole} (deeper half); "
        f"{block['density_per_mpix']} per Mpix of the chip over the {area / 1e6:.2f} Mpix searched; "
        "shallower or larger shadows are not found"
    ]
