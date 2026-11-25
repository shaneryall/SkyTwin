"""Frames whose stars trail: a telescope tracking a satellite, or a camera on a moving platform.

Every star in such a frame is the same short segment, so the frame's own pixels give the trail.

- `geometry`: length and angle of the trails from the frame's autocorrelation, which for segments of
  length L is a ridge whose profile along the trail is a triangle of half-width L, smoothed by the PSF.
- `centres`: isolated trails, found by a matched filter along the trail (a line of length L).
- `psf_from_trails`: the PSF from the profile across the trails, which the motion does not blur; for a
  sum of round Gaussians the profile across is the same sum of one-dimensional Gaussians, with the same
  FWHMs and flux fractions.
- `aperture_fraction`: the fraction of a trail's flux that a rectangular aperture along it measures, from
  a rendered trail with the fitted PSF, background ring included.
- `radiation`: hits outside the trails, from frames of one exposure: rate per cm² and the read-out time
  from how the hits split between the chip's halves.
"""

import numpy as np
from astropy.io import fits
from scipy import ndimage, signal
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf, ndtr

from .calib import _background
from .util import FWHM_PER_SIGMA, mad_sigma, spread

MIN_TRAIL_PX = 8.0
"""Shorter trails are treated as round stars."""


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def _ramp(u, t1, t2, a, b, s):
    """∫ from t1 to t2 of (a + b t) N(u - t; s) dt."""
    z1, z2 = (t1 - u) / s, (t2 - u) / s
    return (a + b * u) * (ndtr(z2) - ndtr(z1)) - b * s * (_norm_pdf(z2) - _norm_pdf(z1))


def _triangle(u, length, s):
    """A triangle of half-width `length` and peak `length`, smoothed by a Gaussian of σ `s`: the
    autocorrelation of a segment of that length along it."""
    return _ramp(u, -length, 0.0, length, 1.0, s) + _ramp(u, 0.0, length, length, -1.0, s)


def _sky(d):
    """Frame `d` less its large-scale level, and that residual's MAD σ."""
    r = d - _background(d)
    return r, float(mad_sigma(r[::2, ::2])) + 1e-9


def geometry(d, x0=0, max_len=600.0):
    """(length px, angle deg in [0, 180), PSF σ px) of the trails in frame `d`, from its autocorrelation.

    The frame (2 × 2 binned for speed) less its large-scale level is used as it is: every star has the
    same trail, so the autocorrelation keeps its shape however the fluxes are spread (a threshold or a
    clip would stretch bright trails by their wings). Only a saturated plateau (pixels at 98% of the
    maximum, grown by 5 px) is zeroed and the top 0.005% of pixels capped. The autocorrelation's zero
    lag (pixel noise, hot pixels) is replaced by its neighbours' mean and the level far from the centre
    is removed. The angle comes from the second moments of the part above 30% of the peak; along that
    angle the profile is fitted as a smoothed triangle (`_triangle`). Stars whose ridge falls to half
    its peak within 4 px are round: length 0. A fit at the longest length the window allows or at the
    narrowest width (noise, not stars) gives (None, None, None).
    """
    a = d[:, x0:].astype(np.float32)
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    a = a[:h, :w].reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
    r, sig = _sky(a)
    top = a >= 0.98 * a.max()
    if top.sum() >= 3:
        r[ndimage.binary_dilation(top, iterations=5)] = 0.0
    m = np.clip(r, -5 * sig, float(np.percentile(r, 99.995)))
    acf = np.fft.irfft2(np.abs(np.fft.rfft2(m)) ** 2, s=m.shape)
    acf = np.fft.fftshift(acf)
    cy, cx = m.shape[0] // 2, m.shape[1] // 2
    W = int(min(max_len / 2, cy - 1, cx - 1))
    win = acf[cy - W : cy + W + 1, cx - W : cx + W + 1].astype(np.float64)
    win[W, W] = 0.25 * (win[W - 1, W] + win[W + 1, W] + win[W, W - 1] + win[W, W + 1])
    yy, xx = np.mgrid[-W : W + 1, -W : W + 1]
    rr = np.hypot(yy, xx)
    win -= np.median(win[(rr > 0.8 * W) & (rr <= W)])
    if win[W, W] <= 0:
        return 0.0, None, None
    thr = 0.3 * win[W, W]
    lab, _ = ndimage.label(win > thr)
    core = lab == lab[W, W]
    wt = win * core
    mxx, myy, mxy = (wt * xx * xx).sum(), (wt * yy * yy).sum(), (wt * xx * yy).sum()
    ang = 0.5 * np.arctan2(2 * mxy, mxx - myy)
    u = np.arange(-W, W + 1, dtype=float)
    prof = np.mean(
        [
            ndimage.map_coordinates(
                win, [W + u * np.sin(ang) + dv * np.cos(ang), W + u * np.cos(ang) - dv * np.sin(ang)]
            )
            for dv in (-1.0, 0.0, 1.0)
        ],
        axis=0,
    )
    peak = prof[W]

    def resid(p):
        c, length, s, b = p
        return c * _triangle(u, length, s) / max(length, 1e-3) + b - prof

    half = float(np.argmax(prof[W:] < 0.5 * peak)) or float(W)
    if half <= 2:
        return 0.0, float(np.degrees(ang) % 180.0), None
    fit = least_squares(
        resid,
        [peak, max(2 * half, 1.0), 1.5, 0.0],
        bounds=([0, 0.5, 0.3, -np.inf], [np.inf, 2 * W, 20.0, np.inf]),
    )
    _, length, s, _ = fit.x
    if length > 0.9 * W or s < 0.31:
        return None, None, None
    return float(2 * length), float(np.degrees(ang) % 180.0), float(2 * s / np.sqrt(2))


def survey(files, x0=0):
    """{file: (length px, angle deg, PSF σ px)} for `files`, from `geometry`; frames where it fails are left
    out."""
    out = {}
    for f in files:
        try:
            g = geometry(fits.getdata(f), x0)
        except Exception:
            continue
        if g[0] is not None:
            out[f] = g
    return out


def line_kernel(length, angle_deg):
    """A unit-sum line of `length` px at `angle_deg`, drawn with bilinear weights every 0.25 px."""
    t = np.radians(angle_deg)
    n = int(np.ceil(length / 2)) + 2
    k = np.zeros((2 * n + 1, 2 * n + 1))
    for s in np.arange(-length / 2, length / 2 + 1e-9, 0.25):
        x, y = n + s * np.cos(t), n + s * np.sin(t)
        i, j = int(np.floor(y)), int(np.floor(x))
        fy, fx = y - i, x - j
        k[i, j] += (1 - fy) * (1 - fx)
        k[i, j + 1] += (1 - fy) * fx
        k[i + 1, j] += fy * (1 - fx)
        k[i + 1, j + 1] += fy * fx
    return k / k.sum()


def _rotated(dx, dy, angle_deg):
    """Offsets (along, across) the trail direction."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return dx * c + dy * s, -dx * s + dy * c


def centres(d, length, angle_deg, x0=0, sat=None, nsig=6.0, isolate=40.0):
    """Trail centres (x, y, mean level along the trail) in frame `d`, brightest first, to a fraction of a
    pixel from the 3 × 3 centroid of the filtered frame.

    The frame less its large-scale level is filtered with `line_kernel`; local maxima over `nsig` σ of the
    filtered frame are centres. With `isolate`, a centre is dropped when another with over 1/16 of its
    level lies within `length` + `isolate` px along the trail and `isolate` px across, or when any pixel
    within its trail reaches 85% of `sat`.
    """
    r, _ = _sky(d.astype(np.float64))
    r[:, :x0] = 0.0
    s = signal.fftconvolve(r, line_kernel(length, angle_deg), mode="same")
    ss = float(mad_sigma(s[::4, ::4])) + 1e-9
    size = int(min(length, 30)) + 5
    pk = (s == ndimage.maximum_filter(s, size=size)) & (s > nsig * ss)
    ys, xs = np.nonzero(pk)
    lv = s[ys, xs]
    order = np.argsort(-lv)
    ys, xs, lv = ys[order], xs[order], lv[order]
    H, W = d.shape

    def sub(k):
        y, x = ys[k], xs[k]
        if not (0 < y < H - 1 and 0 < x < W - 1):
            return float(x), float(y)
        c = s[y - 1 : y + 2, x - 1 : x + 2]
        c = np.clip(c - c.min(), 0, None)
        t = c.sum() or 1.0
        oy, ox = np.mgrid[-1:2, -1:2]
        return float(x + (c * ox).sum() / t), float(y + (c * oy).sum() / t)

    if not isolate:
        return [(*sub(k), float(lv[k])) for k in range(len(xs))]
    out = []
    for k in range(len(xs)):
        du, dv = _rotated(xs - xs[k], ys - ys[k], angle_deg)
        near = (np.abs(du) < length + isolate) & (np.abs(dv) < isolate) & (lv > lv[k] / 16)
        near[k] = False
        if near.any():
            continue
        if sat is not None:
            half = int(length / 2 + 4)
            y0, y1 = max(ys[k] - half, 0), min(ys[k] + half + 1, H)
            x0_, x1 = max(xs[k] - half, 0), min(xs[k] + half + 1, W)
            if d[y0:y1, x0_:x1].max() >= 0.85 * sat:
                continue
        out.append((*sub(k), float(lv[k])))
    return out


def cross_profile(d, length, angle_deg, x0=0, sat=None, max_stars=200, min_snr=30.0, reach=30.0):
    """Offsets across the trail and pixel values over the star's flux per unit length, (V, Y, stars), from
    the middle of isolated trails in frame `d` (`centres`).

    Pixels within `length`/2 - 6 px of the centre along the trail and `reach` px across are kept; the
    background is the median 2 to 12 px beyond `reach`; each star is recentred across the trail on its
    core (6 px) and normalised by its flux per unit length within 12 px. Stars whose flux per unit length
    is under `min_snr` σ of the background are left out.
    """
    if length < MIN_TRAIL_PX + 12:
        return np.zeros(0), np.zeros(0), 0
    d = d.astype(np.float64)
    H, W = d.shape
    ext = int(length / 2 + reach + 15)
    yy, xx = np.mgrid[-ext : ext + 1, -ext : ext + 1]
    fv, fy = [], []
    for x, y, _ in centres(d, length, angle_deg, x0, sat):
        xi, yi = int(np.rint(x)), int(np.rint(y))
        if not (x0 + ext < xi < W - ext and ext < yi < H - ext):
            continue
        cut = d[yi - ext : yi + ext + 1, xi - ext : xi + ext + 1]
        u, v = _rotated(xx - (x - xi), yy - (y - yi), angle_deg)
        mid = np.abs(u) <= length / 2 - 6
        ring = mid & (np.abs(v) > reach + 2) & (np.abs(v) <= reach + 12)
        bg = float(np.median(cut[ring]))
        sel = mid & (np.abs(v) <= reach)
        vv, yv = v[sel], cut[sel] - bg
        core = np.abs(vv) <= 6
        w_ = np.clip(yv[core], 0, None)
        if w_.sum() <= 0:
            continue
        vv = vv - float(np.sum(vv[core] * w_) / w_.sum())
        per_len = yv[np.abs(vv) <= 12].sum() / (length - 12)
        sig = float(mad_sigma(cut[ring], bg))
        if per_len <= 0 or per_len < min_snr * sig:
            continue
        fv.append(vv)
        fy.append(yv / per_len)
        if len(fv) >= max_stars:
            break
    if not fv:
        return np.zeros(0), np.zeros(0), 0
    return np.concatenate(fv), np.concatenate(fy), len(fv)


def _pixel_gauss(v, w):
    """A unit-flux Gaussian of FWHM `w` integrated over pixels 1 px wide centred at `v`."""
    s = w / FWHM_PER_SIGMA
    return 0.5 * (erf((v + 0.5) / (s * np.sqrt(2))) - erf((v - 0.5) / (s * np.sqrt(2))))


def fit_marginal(V, Y, reach=30.0, step=0.25):
    """Multi-Gaussian PSF from a stacked profile across trails (`cross_profile`): ({core_fwhm_px, wings,
    chi2, halos}, notes), or (None, notes).

    The profile is the 3σ-clipped mean in `step` px bins (a median of integer pixel values sticks at zero
    in a faint halo). It is fitted with a core and one halo, then a core and two; the second halo is kept
    when it lowers χ² by over 20%. Each Gaussian is integrated over a pixel 1 px wide, as Skytwin renders.
    The two-halo fit starts its second halo at 3 × the first one's width, clipped into the fit's bounds:
    a faint first halo can fit wide (28 px at fraction 0.01), and curve_fit refuses a start outside them.
    """
    edges = np.arange(-reach, reach + 1e-9, step)
    idx = np.digitize(V, edges) - 1
    cen = 0.5 * (edges[1:] + edges[:-1])
    prof, err = np.full(len(cen), np.nan), np.full(len(cen), np.nan)
    for i in range(len(cen)):
        y = Y[idx == i]
        if len(y) < 10:
            continue
        for _ in range(3):
            m, s = y.mean(), y.std()
            y = y[np.abs(y - m) < 3 * s] if s > 0 else y
        prof[i], err[i] = y.mean(), max(y.std(), 1e-6) / np.sqrt(len(y))
    ok = np.isfinite(prof)
    if ok.sum() < 40:
        return None, [f"only {ok.sum()} bins across the trails hold 10+ pixels"]
    c, p, e = cen[ok], prof[ok], err[ok]
    w0 = max(float(np.sum(np.clip(p, 0, None) * c * c) / np.sum(np.clip(p, 0, None))) ** 0.5 * 2.355, 1.0)

    def m1(v, amp, f1, a, b):
        return amp * ((1 - f1) * _pixel_gauss(v, a) + f1 * _pixel_gauss(v, b))

    def m2(v, amp, f1, f2, a, b, cc):
        return amp * ((1 - f1 - f2) * _pixel_gauss(v, a) + f1 * _pixel_gauss(v, b) + f2 * _pixel_gauss(v, cc))

    try:
        p1, _ = curve_fit(
            m1, c, p, sigma=e, p0=[1.0, 0.2, 0.7 * w0, 2 * w0], bounds=([0.3, 0, 0.5, 1.0], [2, 0.95, 10, 40])
        )
    except RuntimeError:
        return None, ["the profile across the trails does not fit a core and halo"]
    chi1 = float(np.sum(((p - m1(c, *p1)) / e) ** 2))
    best = {"core_fwhm_px": p1[2], "wings": [{"fraction": p1[1], "fwhm_px": p1[3]}], "chi2": chi1, "halos": 1}
    lo2, hi2 = [0.3, 0, 0, 0.5, 1.0, 3.0], [2, 0.95, 0.5, 10, 40, 80]
    try:
        p2, _ = curve_fit(
            m2,
            c,
            p,
            sigma=e,
            p0=np.clip([1.0, 0.3, 0.05, p1[2], p1[3], 3 * p1[3]], lo2, hi2),
            bounds=(lo2, hi2),
        )
        chi2 = float(np.sum(((p - m2(c, *p2)) / e) ** 2))
        if chi2 < 0.8 * chi1 and p2[2] > 0.01:
            best = {
                "core_fwhm_px": p2[3],
                "wings": [{"fraction": p2[1], "fwhm_px": p2[4]}, {"fraction": p2[2], "fwhm_px": p2[5]}],
                "chi2": chi2,
                "halos": 2,
            }
    except RuntimeError:
        pass
    best["wings"].sort(key=lambda wg: wg["fwhm_px"])
    return best, [f"χ² {chi1:.0f} with one halo" + (f", {best['chi2']:.0f} with two" if best["halos"] == 2 else "")]


def psf_from_trails(files, geo, x0=0, sat=None, max_frames=30):
    """PSF from the profile across isolated trails in up to `max_frames` of `files` with a trail
    (`geo` from `survey`): (fit or None, stars, frames, notes). See `cross_profile` and `fit_marginal`."""
    use = [f for f in files if f in geo and geo[f][0] >= MIN_TRAIL_PX + 12]
    Vs, Ys, frames, stars = [], [], 0, 0
    for f in spread(use, max_frames):
        L, ang, _ = geo[f]
        V, Y, n = cross_profile(fits.getdata(f), L, ang, x0, sat)
        if n:
            Vs.append(V)
            Ys.append(Y)
            frames += 1
            stars += n
    if frames < 3 or stars < 20:
        return None, stars, frames, [f"only {stars} isolated bright trails in {frames} frames"]
    fit, notes = fit_marginal(np.concatenate(Vs), np.concatenate(Ys))
    return fit, stars, frames, notes


def marginal_fwhm(comps):
    """FWHM of the profile across a trail for a PSF [(fraction, fwhm_px)]."""
    v = np.linspace(0, 30, 3001)
    p = sum(f * np.exp(-0.5 * (v / (w / FWHM_PER_SIGMA)) ** 2) / (w / FWHM_PER_SIGMA) for f, w in comps)
    return float(2 * v[np.argmax(p < 0.5 * p[0])])


def trail_box(length, comps, pad=12.0):
    """(half length, half width, ring pad) of the photometry box along a trail: 2 FWHM across each side
    of the trail and 2 FWHM beyond each end, the background in a ring `pad` px wide around it."""
    fw = marginal_fwhm(comps)
    return length / 2 + 2 * fw, 2 * fw, pad


def aperture_fraction(comps, length, sub=5):
    """Fraction of a trail's flux that `trail_box` photometry measures: a unit trail of `length` px with
    PSF `comps` is rendered (`sub` × `sub` samples per pixel) and measured the same way, so both the halo
    outside the box and the halo that raises the ring's median count."""
    half_l, half_w, pad = trail_box(length, comps)
    ext = int(half_l + pad + 4)
    yy, xx = np.mgrid[-ext : ext + 1, -ext : ext + 1].astype(float)
    img = np.zeros_like(xx)
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    for dy in offs:
        for dx in offs:
            u, v = xx + dx, yy + dy
            for f, w in comps:
                s = w / FWHM_PER_SIGMA
                along = 0.5 * (erf((u + length / 2) / (s * np.sqrt(2))) - erf((u - length / 2) / (s * np.sqrt(2))))
                img += f * _norm_pdf(v / s) / s * along / max(length, 1e-9)
    img /= sub * sub
    ap = (np.abs(xx) <= half_l) & (np.abs(yy) <= half_w)
    ring = (np.abs(xx) <= half_l + pad) & (np.abs(yy) <= half_w + pad) & ~ap
    return float((img[ap] - np.median(img[ring])).sum())


def mask(d, length, angle_deg, sat=None, x0=0):
    """Pixels of `d` that belong to star trails: connected groups over 2.5σ in the frame (less its
    large-scale level, smoothed by 1 px) that reach over half a trail along the trail direction, grown
    by half their width plus 3 px; and pixels at 95% of `sat`, grown by 15 px."""
    r, _ = _sky(d.astype(np.float64))
    sm = ndimage.gaussian_filter(r, 1.0)
    ss = float(mad_sigma(sm[::2, ::2])) + 1e-9
    lab, n = ndimage.label(sm > 2.5 * ss, structure=np.ones((3, 3)))
    out = np.zeros(d.shape, bool)
    if n:
        sl = ndimage.find_objects(lab)
        t = np.radians(angle_deg)
        for i, s in enumerate(sl, start=1):
            if s is None:
                continue
            ys, xs = np.nonzero(lab[s] == i)
            if len(ys) < 6:
                continue
            u = xs * np.cos(t) + ys * np.sin(t)
            v = -xs * np.sin(t) + ys * np.cos(t)
            if np.ptp(u) >= 0.5 * length:
                grow = int(0.5 * np.ptp(v)) + 3
                sub = np.zeros((s[0].stop - s[0].start + 2 * grow, s[1].stop - s[1].start + 2 * grow), bool)
                sub[ys + grow, xs + grow] = True
                sub = ndimage.binary_dilation(sub, iterations=grow)
                y0, x0_ = s[0].start - grow, s[1].start - grow
                ya, xa = max(y0, 0), max(x0_, 0)
                yb, xb = min(y0 + sub.shape[0], d.shape[0]), min(x0_ + sub.shape[1], d.shape[1])
                out[ya:yb, xa:xb] |= sub[ya - y0 : yb - y0, xa - x0_ : xb - x0_]
    if sat is not None:
        out |= ndimage.binary_dilation(d >= 0.95 * sat, iterations=15)
    out[:, :x0] = True
    return out


def radiation(files, geo, exposure, area_cm2, x0=0, sat=None, k=8.0, max_frames=40):
    """Radiation from frames of one `exposure` with trails: ({rate_per_cm2_s, readout_s, first_row} or None,
    notes).

    In each frame less its 5 × 5 median, groups of pixels `k` σ high outside the trail `mask` are events;
    events whose peak lies within 1 px of one in another frame (hot and flickering pixels, a tracked
    target) are left out. Counts are scaled to the whole chip by the unmasked fraction of each half. A row
    collects hits for the exposure t plus its wait to be read, up to R for the last row, so the halves hold
    them in the ratio r = (t + 3R/4) / (t + R/4), R = t (r − 1) / (3/4 − r/4), kept when the split is 2σ
    (Poisson) from even; the rate is then the hits per chip over the area and t + R/2.
    """
    use = [f for f in files if f in geo][:max_frames]
    if len(use) < 5:
        return None, [f"only {len(use)} frames of {exposure:g} s with a trail measured"]
    peaks, count, frees = [], None, []
    for f in use:
        d = fits.getdata(f).astype(np.float64)
        L, ang, _ = geo[f]
        m = mask(d, max(L, 1.0), ang or 0.0, sat, x0)
        res = d - ndimage.median_filter(d, size=5)
        sig = float(mad_sigma(res[::2, ::2])) + 1e-9
        lab, n = ndimage.label((res > k * sig) & ~m, structure=np.ones((3, 3)))
        pos = np.array(ndimage.maximum_position(res, lab, range(1, n + 1)), int).reshape(-1, 2)
        h = d.shape[0]
        frees.append((1 - m[h // 2 :].mean(), 1 - m[: h // 2].mean()))
        peaks.append(pos)
        hit = np.zeros(d.shape, bool)
        hit[pos[:, 0], pos[:, 1]] = True
        if count is None:
            count = np.zeros(d.shape, np.uint16)
        count += ndimage.binary_dilation(hit, structure=np.ones((3, 3)))
    h = count.shape[0]
    up = lo = rec = 0.0
    n_up = n_lo = 0
    for pos, (fu, fl) in zip(peaks, frees):
        again = count[pos[:, 0], pos[:, 1]] >= 2
        keep = pos[~again]
        rec += int(again.sum())
        u, lw = int((keep[:, 0] >= h / 2).sum()), int((keep[:, 0] < h / 2).sum())
        n_up += u
        n_lo += lw
        up += u / max(fu, 0.05)
        lo += lw / max(fl, 0.05)
    nf = len(peaks)
    per_chip = (up + lo) / nf
    first, hi_, lo_ = ("bottom", up, lo) if up >= lo else ("top", lo, up)
    ratio = hi_ / max(lo_, 1e-9)
    n_hi, n_lo_ = (n_up, n_lo) if up >= lo else (n_lo, n_up)
    notes = [
        f"{n_up + n_lo} hits outside the trails in {nf} frames of {exposure:g} s ({rec} more at pixels that fire "
        f"in two or more frames left out): {per_chip:.1f} per chip and frame, {up / max(up + lo, 1e-9):.0%} in "
        "the upper half"
    ]
    ro = 0.0
    if ratio - 1 > 2 * ratio * np.sqrt(1 / max(n_hi, 1) + 1 / max(n_lo_, 1)):
        ratio = min(ratio, 2.9)
        ro = exposure * (ratio - 1) / (0.75 - ratio / 4)
        notes.append(f"{ratio:.2f}x more in the half read last ({first} read first) -> read-out {ro:.2f} s")
    else:
        notes.append("hits split evenly between the halves: no read-out time")
    out = {"rate_per_cm2_s": round(per_chip / (area_cm2 * (exposure + ro / 2)), 4)}
    if ro:
        out.update(readout_s=round(ro, 2), first_row=first)
    notes.append("every event outside the masks counts, so the rate is an upper bound")
    return out, notes


def photometry(d, px, py, mag, gain, exposure, comps, length, angle_deg, x0=0, sat=None, col=None):
    """Forced photometry of trailed catalogue stars: a `trail_box` along the trail about each predicted
    centre, the background the median of the ring around it, the flux corrected by `aperture_fraction`.
    Stars with a catalogue neighbour under 3 mag fainter whose box overlaps, or a pixel in the box at 95%
    of `sat`, are left out. Returns [(mag, zeropoint, S/N, colour index or NaN)], as
    `astrometry.forced_photometry`."""
    half_l, half_w, pad = trail_box(length, comps)
    frac = aperture_fraction(comps, length)
    ext = int(half_l + pad + 2)
    H, W = d.shape
    yy, xx = np.mgrid[-ext : ext + 1, -ext : ext + 1]
    out = []
    for k in range(len(mag)):
        x, y = px[k], py[k]
        xi, yi = int(np.rint(x)), int(np.rint(y))
        if not (x0 + ext <= xi < W - ext and ext <= yi < H - ext):
            continue
        du, dv = _rotated(px - x, py - y, angle_deg)
        near = (np.abs(du) < 2 * half_l + pad) & (np.abs(dv) < 2 * half_w + pad) & (mag < mag[k] + 3)
        near[k] = False
        if near.any():
            continue
        cut = d[yi - ext : yi + ext + 1, xi - ext : xi + ext + 1]
        u, v = _rotated(xx - (x - xi), yy - (y - yi), angle_deg)
        ap = (np.abs(u) <= half_l) & (np.abs(v) <= half_w)
        if sat is not None and cut[ap].max() >= 0.95 * sat:
            continue
        ring = (np.abs(u) <= half_l + pad) & (np.abs(v) <= half_w + pad) & ~ap
        bg = float(np.median(cut[ring]))
        sig = float(mad_sigma(cut[ring], bg))
        flux = float((cut[ap] - bg).sum())
        if flux > 0:
            out.append(
                (
                    float(mag[k]),
                    float(mag[k] + 2.5 * np.log10(flux * gain / exposure / frac)),
                    flux / (sig * np.sqrt(ap.sum()) + 1e-9),
                    float(col[k]) if col is not None else np.nan,
                )
            )
    return out
