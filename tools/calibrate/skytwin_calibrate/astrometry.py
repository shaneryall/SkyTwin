"""Pointing calibration from plate solves (Fayde).

Solving frames with round stars against a catalogue gives, per frame, the
true pixel scale, parity and roll, and where on the image the header's
RA/Dec actually falls. Over many frames that fixes:

- `camera.pixel_scale_arcsec` and `camera.parity`;
- the roll keyword's convention: solved roll = offset + sign × keyword,
  whichever sign leaves the smaller spread;
- `pointing.header`: the pixel the header pointing refers to, and the
  random error of the header's RA/Dec and roll.

Fayde reports roll in Skytwin's convention (position angle of image +y,
east of north), so no conversion is needed.

Archives often carry a fitted WCS per frame (an Astrometry.net solution,
say) where a fresh plate solve would fail: very wide fields, few bright
stars. `from_header_wcs` uses that WCS to match detected stars to the
catalogue; a joint fit over many frames then separates what changes per
frame (pointing: an affine map of tangent-plane coordinates) from what the
lens fixes (radial distortion about the optical axis). The matched stars
also give the photometric zeropoint.
"""

import csv
import json
import os
import shutil
import subprocess

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from scipy.optimize import least_squares

from . import trails
from .util import FWHM_PER_SIGMA, header_number, is_csv, mad_sigma, most_common, open_text, read_json, run_skytwin


def find_fayde(path=None):
    """The fayde binary: `path` if given, else the one on the PATH (None if there is none)."""
    return path or shutil.which("fayde")


def _wrap(a):
    """Angles in degrees wrapped to [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def _circ_median(a):
    """Median of angles in degrees, taken about their circular mean so the wrap does not split them."""
    a = np.asarray(a, float)
    ref = np.degrees(np.arctan2(np.mean(np.sin(np.radians(a))), np.mean(np.cos(np.radians(a)))))
    return float(ref + np.median(_wrap(a - ref)))


def solve_one(fayde, f, ucac4, ra, dec, scale, deadline=20):
    """Fayde's plate solve of frame `f` (a dict), or None when it fails.

    The search is within 2° of `ra`, `dec` (deg) when given, and at `scale` (arcsec/px, ±3%) when given,
    ignoring any WCS in the header; `deadline` is in seconds.
    """
    cmd = [
        fayde,
        "solve",
        "--image",
        f,
        "--ucac4",
        ucac4,
        "--no-header-wcs",
        "--radius",
        "2",
        "--deadline",
        str(deadline),
    ]
    if ra is not None and dec is not None:
        cmd += ["--ra", str(ra), "--dec", str(dec)]
    if scale:
        cmd += ["--scale", str(scale), "--scale-sigma", "0.03"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        r = json.loads(p.stdout)
    except ValueError:
        return None
    return r if r.get("solved") else None


def enclosed(psf_comps, r):
    """Fraction of a multi-Gaussian PSF [(fraction, fwhm_px)] within radius r."""
    k = FWHM_PER_SIGMA
    return sum(f * (1 - np.exp(-r * r / (2 * (w / k) ** 2))) for f, w in psf_comps)


def star_zeropoints(f, pairs, gain, exposure, sat, psf_comps, r_ap=8.0, mag_range=(8.5, 14.5)):
    """Zeropoint (mag giving 1 e-/s) from each matched catalogue star in `pairs`.

    Aperture flux within `r_ap` px less the median of the 12-20 px annulus, corrected to total flux with
    the PSF model; stars outside `mag_range`, with inlier probability under 0.9, near the edge or
    peaking within 5% of saturation `sat` are skipped. Fayde's pixel positions are FITS (1-based).
    """
    d = fits.getdata(f).astype(np.float64)
    h, w = d.shape
    yy, xx = np.mgrid[-20:21, -20:21]
    rr = np.hypot(yy, xx)
    frac = enclosed(psf_comps, r_ap)
    out = []
    for p in pairs:
        m = p.get("mag")
        if m is None or not (mag_range[0] <= m <= mag_range[1]) or p.get("inlier_prob", 1) < 0.9:
            continue
        x, y = round(p["x"] - 1), round(p["y"] - 1)
        if x < 20 or y < 20 or x >= w - 20 or y >= h - 20:
            continue
        cut = d[y - 20 : y + 21, x - 20 : x + 21]
        if sat is not None and cut[rr <= 3].max() >= 0.95 * sat:
            continue
        bg = np.median(cut[(rr >= 12) & (rr <= 20)])
        flux = (cut[rr <= r_ap] - bg).sum() * gain / exposure / frac
        if flux > 0:
            out.append(m + 2.5 * np.log10(flux))
    return out


def calibrate(
    files,
    headers,
    metas,
    km,
    fayde,
    ucac4,
    scale_guess=None,
    max_frames=60,
    gain=1.0,
    sat=None,
    psf_comps=None,
    skytwin=None,
    catalog=None,
    work=None,
    extinction=None,
    x0=0,
):
    """Pointing calibration from Fayde plate solves of up to `max_frames` frames.

    Returns (camera updates, `pointing.header` block or None, roll convention or None, notes). Pixel scale
    is the median and parity the most common over solved frames (at least 5); the zeropoint needs 20
    usable stars. The header pointing's reference pixel and error (arcsec per axis) come from where the
    header RA/Dec falls in each solution. The roll keyword's convention is the sign whose offset leaves
    the smaller MAD spread, kept when that spread is under 5°.

    With `skytwin`, `catalog` and `work`, the zeropoint comes instead from photometry of the catalogue's
    stars through each solution (`frame_photometry`, `zeropoint_and_extinction`): every catalogue star,
    detected or not, with the colour term and, for frames with an airmass, the extinction.
    """
    rows = []
    zps, per_frame = [], []
    lim, failed = 8.0, []
    if work:
        os.makedirs(work, exist_ok=True)
    rk = (km.get("roll") or {}).get("keyword")
    for f, h, m in list(zip(files, headers, metas))[:max_frames]:
        p = m.get("pointing") or {}
        r = solve_one(fayde, f, ucac4, p.get("ra_deg"), p.get("dec_deg"), scale_guess)
        if r is None:
            continue
        w = WCS(fits.Header.fromstring("\n".join(c.ljust(80) for c in r["wcs_cards"]), sep="\n"))
        ref = None
        if p.get("ra_deg") is not None:
            x, y = w.all_world2pix(p["ra_deg"], p["dec_deg"], 0)
            ref = (float(x), float(y))
        raw = header_number(h, rk)
        if skytwin and catalog and work and m.get("exposure_s"):
            try:
                d = fits.getdata(f).astype(np.float64)
                hh, ww = d.shape
                cat, _, _, lim = catalogue_to_count(skytwin, work, ww, hh, w, catalog, lim, len(rows))
                ra, dec, mag, ids = (np.array(v) for v in zip(*cat))
                colours = catalogue_colours(catalog)
                col = np.array([colours.get(int(i), np.nan) if i is not None else np.nan for i in ids], float)
                px, py = w.all_world2pix(ra, dec, 0)
                ph = frame_photometry(d, px, py, mag, col, gain, m["exposure_s"], psf_comps, x0, sat)
                per_frame.append((m.get("airmass"), ph))
            except Exception as e:
                failed.append(f"{type(e).__name__}: {e}")
        elif psf_comps and m.get("exposure_s"):
            zps += star_zeropoints(f, r.get("pairs", []), gain, m["exposure_s"], sat, psf_comps)
        rows.append(
            dict(scale=r["pixel_scale_arcsec"], parity=r["parity"].lower(), roll=r["roll_deg"], ref=ref, raw_roll=raw)
        )
    if len(rows) < 5:
        return {}, None, None, [f"only {len(rows)} frames solved (need 5)"]
    cam = {"pixel_scale_arcsec": round(float(np.median([r["scale"] for r in rows])), 5)}
    par = [r["parity"] for r in rows]
    cam["parity"] = most_common(par)
    notes = [
        f"{len(rows)} frames solved: {cam['pixel_scale_arcsec']} arcsec/px, {cam['parity']} parity "
        f"({par.count(cam['parity'])} of {len(par)})"
    ]
    if failed:
        notes.append(f"catalogue photometry failed in {len(failed)} frames (first: {failed[0][:200]})")
    if per_frame:
        zcam, znotes = zeropoint_and_extinction(per_frame, extinction)
        cam.update(zcam)
        notes += znotes
    elif len(zps) >= 20:
        cam["zeropoint_mag"] = round(float(np.median(zps)), 3)
        mad = mad_sigma(zps, cam["zeropoint_mag"])
        notes.append(
            f"zeropoint {cam['zeropoint_mag']} mag (1 e-/s in the catalogue band) from {len(zps)} stars, "
            f"scatter {mad:.2f} mag"
        )
    else:
        notes.append(f"zeropoint not measured: {len(zps)} usable matched stars")
    header = {}
    refs = np.array([r["ref"] for r in rows if r["ref"] is not None])
    if len(refs) >= 5:
        c = np.median(refs, axis=0)
        mad = mad_sigma(refs, c, axis=0) * cam["pixel_scale_arcsec"]
        header["reference_px"] = [round(float(c[0]), 1), round(float(c[1]), 1)]
        header["error_arcsec"] = round(float(np.mean(mad)), 2)
        notes.append(
            f"the header RA/Dec refers to pixel ({c[0]:.1f}, {c[1]:.1f}), "
            f"error {header['error_arcsec']} arcsec per axis"
        )
    roll = None
    pr = [(r["roll"], r["raw_roll"]) for r in rows if r["raw_roll"] is not None]
    if len(pr) >= 5:
        s, v = (np.array(x, float) for x in zip(*pr))
        best = None
        for sign in (1, -1):
            off = _circ_median(s - sign * v)
            spread = float(mad_sigma(_wrap(s - sign * v - off), 0.0))
            if best is None or spread < best[2]:
                best = (sign, off, spread)
        sign, off, spread = best
        if spread < 5:
            roll = {"keyword": rk, "offset_deg": round(off % 360.0, 3), "sign": sign}
            header["roll_error_deg"] = round(spread, 3)
            notes.append(f"roll = {roll['offset_deg']} {'+' if sign > 0 else '-'} {rk}, error {spread:.2f} deg")
        else:
            notes.append(f"{rk} does not follow the solved roll (spread {spread:.1f} deg)")
    return cam, (header or None), roll, notes


def has_celestial_wcs(h):
    """True when header `h` has an RA/Dec WCS."""
    return str(h.get("CTYPE1", "")).startswith("RA---") and str(h.get("CTYPE2", "")).startswith("DEC--")


def pointing_from_wcs(h):
    """Skytwin's pointing {ra_deg, dec_deg, roll_deg} of the frame whose header `h` holds a celestial WCS:
    the image centre, and the position angle (east of north) of the image +y axis there. None when the
    header has no celestial WCS or it does not convert."""
    if not has_celestial_wcs(h):
        return None
    try:
        w = WCS(h, naxis=2)
        cx, cy = (int(h["NAXIS1"]) - 1) / 2, (int(h["NAXIS2"]) - 1) / 2
        (r0, d0), (r1, d1) = w.all_pix2world([[cx, cy], [cx, cy + 10.0]], 0)
    except Exception:
        return None
    if not (np.isfinite([r0, d0, r1, d1]).all() and abs(d0) <= 90.0):
        return None
    dra = np.radians(((r1 - r0 + 180.0) % 360.0) - 180.0) * np.cos(np.radians(d0))
    roll = float(np.degrees(np.arctan2(dra, np.radians(d1 - d0))) % 360.0)
    return {"ra_deg": float(r0), "dec_deg": float(d0), "roll_deg": round(roll, 4)}


def _detect(d, x0=0, nsig=6.0, sat=None, trail=None):
    """Peaks over `nsig` σ of a frame less its 15 px median, as (x, y, peak, saturated) with 3×3 centroids.

    The first `x0` + 2 columns and a 2 px border are left out; a peak at or above 95% of `sat` is marked
    saturated. With `trail` (length px, angle deg) of at least `trails.MIN_TRAIL_PX`, the peaks are the
    centres of trails (`trails.centres`), marked saturated when the centre pixel is.
    """
    if trail and trail[0] >= trails.MIN_TRAIL_PX:
        out = []
        for x, y, lv in trails.centres(d, trail[0], trail[1], x0, nsig=nsig, isolate=0):
            xi, yi = int(np.rint(x)), int(np.rint(y))
            out.append((x, y, lv, sat is not None and d[yi, xi] >= 0.95 * sat))
        return out
    loc = d - ndimage.median_filter(d, size=15)
    sig = mad_sigma(loc, 0.0) + 1e-9
    pk = (loc == ndimage.maximum_filter(loc, 5)) & (loc > nsig * sig)
    pk[:, : x0 + 2] = False
    pk[:, -2:] = False
    pk[:2, :] = False
    pk[-2:, :] = False
    out = []
    for y, x in zip(*np.nonzero(pk)):
        c = np.clip(loc[y - 1 : y + 2, x - 1 : x + 2], 0, None)
        s = c.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[-1:2, -1:2]
        sat_px = sat is not None and d[y, x] >= 0.95 * sat
        out.append((x + (c * xx).sum() / s, y + (c * yy).sum() / s, float(loc[y, x]), sat_px))
    return out


def scene_catalog(cat):
    """Skytwin's `scene.catalog` source for a UCAC4 directory or a CSV catalogue file (a Gaia extract with
    Skytwin's default column names); a dict is copied as it is.

    Not to be confused with `contract.catalog_block`, the profile's record of the catalogue.
    """
    if isinstance(cat, dict):
        return dict(cat)
    if is_csv(cat):
        return {"csv": {"path": str(cat)}}
    return {"ucac4": str(cat)}


def _catalogue(skytwin, work, w_, h_, wcs, ucac4, mag_limit, tag):
    """Catalogue stars [(ra, dec, mag, id)] on a `w_` × `h_` frame with WCS `wcs`, plus its pixel scale
    (arcsec) and parity.

    The stars come through Skytwin's own catalogue reader: a noiseless render at the frame's pointing,
    roll, scale and parity, whose truth lists them. `ucac4` is a UCAC4 directory or a CSV catalogue
    (see `scene_catalog`).
    """
    cx, cy = (w_ - 1) / 2, (h_ - 1) / 2
    c = wcs.pixel_to_world(cx, cy)
    up = wcs.pixel_to_world(cx, cy + 1)
    roll = float(c.position_angle(up).deg) % 360
    m = wcs.pixel_scale_matrix
    scale = 3600 * float(np.sqrt(abs(np.linalg.det(m))))
    parity = "positive" if np.linalg.det(m) > 0 else "negative"
    scene = {
        "seed": 1,
        "camera": {
            "width": w_,
            "height": h_,
            "pixel_scale_arcsec": scale,
            "fwhm_px": 1.0,
            "exposure_s": 1.0,
            "zeropoint_mag": 20.0,
            "sky": {"e_per_s": 0.0},
            "dark_e_per_s": 0.0,
            "read_noise_e": 0.0,
            "gain_e_per_adu": 1.0,
            "bias_adu": 0.0,
            "full_well_e": 1e12,
            "parity": parity,
        },
        "scene": {
            "pointing": {"ra_deg": float(c.ra.deg), "dec_deg": float(c.dec.deg), "roll_deg": roll},
            "catalog": dict(scene_catalog(ucac4), mag_limit=mag_limit),
        },
    }
    out = os.path.join(work, f"cat{tag}.fits")
    run_skytwin(skytwin, scene, out)
    t = read_json(out[:-5] + ".truth.json")
    return (
        [(s["ra_deg"], s["dec_deg"], s["mag"], s.get("id")) for s in t["stars"] if s.get("ra_deg") is not None],
        scale,
        parity,
    )


def catalogue_to_count(skytwin, work, w_, h_, wcs, ucac4, mag_limit, tag, min_stars=40, max_limit=18.0, step=2.0):
    """`_catalogue` with its magnitude limit made fainter, `step` at a time up to `max_limit`, until the
    frame holds `min_stars` catalogue stars: a limit that suits a field tens of degrees wide leaves a
    sub-degree field with a handful. Returns (stars, scale, parity, limit used); callers pass the limit
    back for the next frame, so similar fields do not search again."""
    lim = mag_limit
    while True:
        cat, scale, parity = _catalogue(skytwin, work, w_, h_, wcs, ucac4, lim, tag)
        if len(cat) >= min_stars or lim >= max_limit:
            return cat, scale, parity, lim
        lim = min(lim + step, max_limit)


_COLOURS = {}


def catalogue_colours(cat):
    """Colour index (BP-RP) by source id from a CSV catalogue's `bp_rp` column, cached per file; empty for
    UCAC4 and for rows without one."""
    if isinstance(cat, dict) or not is_csv(cat):
        return {}
    if cat not in _COLOURS:
        out = {}
        with open_text(cat) as fh:
            r = csv.DictReader(fh)
            if "bp_rp" in (r.fieldnames or []):
                for row in r:
                    try:
                        out[int(row["source_id"])] = float(row["bp_rp"])
                    except (ValueError, TypeError, KeyError):
                        continue
        _COLOURS[cat] = out
    return _COLOURS[cat]


def _tan(ra, dec, ra0, dec0):
    """Gnomonic (tangent-plane) coordinates (xi, eta) in degrees of `ra`, `dec` about `ra0`, `dec0`."""
    a, d, a0, d0 = map(np.radians, (ra, dec, ra0, dec0))
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.degrees(xi), np.degrees(eta)


def from_header_wcs(
    files,
    skytwin,
    ucac4,
    work,
    x0=0,
    sat=None,
    gain=1.0,
    exposures=None,
    psf_comps=None,
    max_frames=80,
    mag_limit=8.0,
    match_px=3.0,
    trail_geo=None,
    airmasses=None,
    extinction=None,
):
    """Camera calibration from frames that carry their own celestial WCS.

    Detected stars are matched (within `match_px`, brightest catalogue star first, saturated peaks left
    out) to catalogue stars to `mag_limit` projected through each frame's WCS, the limit made fainter
    where a field holds too few such stars (`catalogue_to_count`); frames need 6 matches, and 5 such
    frames are needed. A joint fit (soft-L1 loss) gives each frame an affine map of tangent-plane
    coordinates and all frames one radial distortion, 1 + k1 ρ² + k2 ρ⁴ with ρ in units of half the image
    diagonal. The distortion is centred on the image centre, the pixel the header pointing refers to: a
    free centre trades against every frame's offset. It is kept when k1 lies 5σ from zero (σ from the
    fit's Jacobian) and the median residual drops. Pixel scale and parity are read from the WCS; the
    zeropoint and colour term come from `forced_photometry` and `zeropoint_from_forced`. Frames in
    `trail_geo` ({file: (length px, angle deg, ...)}) with trailed stars are matched by their trail centres
    and measured with boxes along the trails (`trails.photometry`); when `trail_geo` is given, frames missing
    from it (their stars may trail, unmeasured) are left out of the photometry. With `airmasses` (per file,
    ground frames) the zeropoint is moved above the atmosphere (`zeropoint_and_extinction`, `extinction` the
    assumed coefficient when it cannot be fitted), reported under the camera updates' `_atmosphere` key.
    Returns (camera updates, notes, matched frames).
    """
    os.makedirs(work, exist_ok=True)
    frames = []
    scales, parities, per_frame = [], [], []
    no_trail = 0
    failed, first_error = 0, None
    lim = mag_limit
    for i, f in enumerate(files[:max_frames]):
        h = fits.getheader(f)
        if not has_celestial_wcs(h):
            continue
        d = fits.getdata(f).astype(np.float64)
        hh, ww = d.shape
        try:
            wcs = WCS(h, naxis=2)
            cat, scale, parity, lim = catalogue_to_count(skytwin, work, ww, hh, wcs, ucac4, lim, i)
        except Exception as e:
            failed += 1
            first_error = first_error or f"{os.path.basename(f)}: {e}"
            continue
        if len(cat) < 4:
            continue
        ra, dec, mag, ids = (np.array(v) for v in zip(*cat))
        colours = catalogue_colours(ucac4)
        col = np.array([colours.get(int(i), np.nan) if i is not None else np.nan for i in ids], float)
        px, py = wcs.all_world2pix(ra, dec, 0)
        tg = (trail_geo or {}).get(f)
        src = _detect(d, x0, sat=sat, trail=tg)
        if not src:
            continue
        S = np.array([(s[0], s[1]) for s in src])
        used, rows = set(), []
        for k in np.argsort(mag):
            dist = np.hypot(S[:, 0] - px[k], S[:, 1] - py[k])
            j = int(np.argmin(dist))
            if dist[j] < match_px and j not in used and not src[j][3]:
                used.add(j)
                rows.append((S[j, 0], S[j, 1], ra[k], dec[k], mag[k]))
        if len(rows) < 6:
            continue
        r = np.array(rows)
        c = wcs.pixel_to_world((ww - 1) / 2, (hh - 1) / 2)
        xi, eta = _tan(r[:, 2], r[:, 3], c.ra.deg, c.dec.deg)
        frames.append(dict(x=r[:, 0], y=r[:, 1], xi=xi, eta=eta, mag=r[:, 4], shape=(hh, ww), file=f))
        scales.append(scale)
        parities.append(parity)
        if exposures is not None and exposures[i]:
            if trail_geo and not tg:
                no_trail += 1
            else:
                rows_ph = frame_photometry(d, px, py, mag, col, gain, exposures[i], psf_comps, x0, sat, tg)
                per_frame.append((airmasses[i] if airmasses is not None else None, rows_ph))
    if len(frames) < 5:
        why = f"; {failed} frames failed, first: {first_error}" if failed else ""
        return {}, [f"only {len(frames)} frames with 6+ catalogue matches{why}"], frames
    hh, ww = frames[0]["shape"]
    R = 0.5 * np.hypot(ww, hh)
    nf = len(frames)

    def unpack(p):
        return p[:4], p[4:].reshape(nf, 6)

    def resid(p, radial=True):
        (k1, k2, cx, cy), A = unpack(p)
        out = []
        for fr, a in zip(frames, A):
            u = a[0] * fr["xi"] + a[1] * fr["eta"] + a[2]
            v = a[3] * fr["xi"] + a[4] * fr["eta"] + a[5]
            if radial:
                rho2 = (u * u + v * v) / (R * R)
                fac = 1 + k1 * rho2 + k2 * rho2 * rho2
                u, v = u * fac, v * fac
            out += [cx + u - fr["x"], cy + v - fr["y"]]
        return np.concatenate(out)

    cx0, cy0 = (ww - 1) / 2, (hh - 1) / 2
    A0 = []
    for fr in frames:
        M = np.c_[fr["xi"], fr["eta"], np.ones_like(fr["xi"])]
        ax = np.linalg.lstsq(M, fr["x"] - cx0, rcond=None)[0]
        ay = np.linalg.lstsq(M, fr["y"] - cy0, rcond=None)[0]
        A0.append(np.r_[ax, ay])
    p0 = np.r_[0.0, 0.0, cx0, cy0, np.ravel(A0)]
    flat_fit = least_squares(
        lambda p: resid(np.r_[0, 0, cx0, cy0, p[4:]], radial=False), p0, loss="soft_l1", f_scale=0.5
    )
    fs = least_squares(
        lambda p: resid(np.r_[p[0], p[1], cx0, cy0, p[2:]]),
        np.r_[0.0, 0.0, flat_fit.x[4:]],
        loss="soft_l1",
        f_scale=0.5,
    )
    x_full = np.r_[fs.x[:2], cx0, cy0, fs.x[2:]]
    rms0 = float(np.sqrt(np.median(resid(np.r_[0, 0, cx0, cy0, flat_fit.x[4:]], radial=False) ** 2)))
    rms1 = float(np.sqrt(np.median(fs.fun**2)))
    k1, k2 = x_full[:2]
    n = sum(len(fr["x"]) for fr in frames)
    cam = {"pixel_scale_arcsec": round(float(np.median(scales)), 4), "parity": most_common(parities)}
    notes = [
        f"{nf} frames, {n} stars matched to the header WCS: {cam['pixel_scale_arcsec']} arcsec/px, "
        f"{cam['parity']} parity"
    ]
    try:
        J = fs.jac
        s2 = float(np.sum(fs.fun**2) / max(len(fs.fun) - len(x_full), 1))
        cov = np.linalg.pinv(J.T @ J) * s2
        k1_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    except Exception:
        k1_err = np.inf
    if abs(k1) > 5 * k1_err and rms1 < rms0:
        cam["distortion"] = {"k1": round(float(k1), 5), "k2": round(float(k2), 5), "radius_px": round(float(R), 2)}
        notes.append(
            f"radial distortion k1 {k1:+.4f} ± {k1_err:.4f}, k2 {k2:+.4f} at {R:.0f} px "
            f"({'pincushion' if k1 > 0 else 'barrel'}) about the image centre; "
            f"residual RMS {rms0:.2f} -> {rms1:.2f} px"
        )
    else:
        notes.append(
            f"no significant distortion: residual RMS {rms0:.2f} px, {rms1:.2f} with the best radial "
            f"model (k1 {k1:+.4f} ± {k1_err:.4f}, k2 {k2:+.4f}); the frames may be distortion-corrected already"
        )
    zcam, zp_notes = zeropoint_and_extinction(per_frame, extinction)
    if no_trail:
        zp_notes.append(f"{no_trail} frames whose trails were not measured left out of the photometry")
    cam.update(zcam)
    notes += zp_notes
    return cam, notes, frames


def forced_photometry(d, px, py, mag, gain, exposure, psf_comps, x0=0, sat=None, r_ap=None, col=None):
    """Aperture photometry at every catalogue star's predicted position,
    detected or not (no selection bias), skipping stars with a neighbour
    under 2 mag fainter within 1.5 apertures or a peak within 5% of saturation `sat`.

    The aperture is `r_ap` px about the predicted position (default 2.5 times the PSF's FWHM, at least
    4 px), the background the median of the annulus from 1.36 to 2 apertures (at least 6 to 10 px), and
    the flux is corrected to total with the PSF model `psf_comps` [(fraction, fwhm_px)]. A fixed aperture
    and annulus sit inside the wings of a wide PSF, where the background takes star light.
    Returns [(mag, zeropoint, S/N, colour index or NaN)] for stars with positive flux.
    """
    h, w = d.shape
    r_ap = r_ap or max(4.0, 2.5 * trails.marginal_fwhm(psf_comps))
    a_in, a_out = max(6.0, 1.36 * r_ap), max(10.0, 2.0 * r_ap)
    n = int(np.ceil(a_out))
    frac = enclosed(psf_comps, r_ap)
    yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
    out = []
    for k in range(len(mag)):
        x, y = px[k], py[k]
        xi, yi = int(np.round(x)), int(np.round(y))
        if not (x0 + n + 1 <= xi < w - n - 1 and n + 1 <= yi < h - n - 1):
            continue
        dd = np.hypot(px - x, py - y)
        if np.any((dd < 1.5 * r_ap) & (dd > 0) & (mag < mag[k] + 2)):
            continue
        cut = d[yi - n : yi + n + 1, xi - n : xi + n + 1]
        r = np.hypot(yy - (y - yi), xx - (x - xi))
        if sat is not None and cut[r <= r_ap].max() >= 0.95 * sat:
            continue
        ann = cut[(r > a_in) & (r < a_out)]
        bg = np.median(ann)
        sig = mad_sigma(ann, bg)
        ap = r <= r_ap
        flux = (cut[ap] - bg).sum()
        snr = flux / (sig * np.sqrt(ap.sum()) + 1e-9)
        if flux > 0:
            out.append(
                (
                    float(mag[k]),
                    float(mag[k] + 2.5 * np.log10(flux * gain / exposure / frac)),
                    float(snr),
                    float(col[k]) if col is not None else np.nan,
                )
            )
    return out


def frame_photometry(d, px, py, mag, col, gain, exposure, psf_comps=None, x0=0, sat=None, trail=None):
    """Photometry rows [(mag, zeropoint, S/N, colour)] of the catalogue stars at (`px`, `py`) in frame `d`:
    boxes along the trails (`trails.photometry`) when `trail` (length px, angle deg) is a trail, else
    `forced_photometry`."""
    comps = psf_comps or [(1.0, 2.0)]
    if trail and trail[0] >= trails.MIN_TRAIL_PX:
        return trails.photometry(d, px, py, mag, gain, exposure, comps, trail[0], trail[1], x0, sat, col)
    return forced_photometry(d, px, py, mag, gain, exposure, comps, x0, sat, col=col)


def good_bins(rows, min_snr=10.0, step=0.5):
    """[(lo, hi)] magnitude bins of `step` holding 10 or more of photometry `rows` whose median S/N reaches
    `min_snr`: the stars `zeropoint_from_forced` uses. Choosing by the bin's median, not each star's S/N,
    keeps noise from selecting the measures that came out bright."""
    if not rows:
        return []
    z = np.array([r[:3] for r in rows], float)
    out = []
    for lo in np.arange(np.floor(z[:, 0].min()), z[:, 0].max(), step):
        m = (z[:, 0] >= lo) & (z[:, 0] < lo + step)
        if m.sum() >= 10 and np.median(z[m, 2]) >= min_snr:
            out.append((float(lo), float(lo + step)))
    return out


def frame_zeropoints(per_frame, cterm=None, bins=None, min_stars=10):
    """[(airmass or None, zeropoint at colour 0)] of the frames in `per_frame` [(airmass, rows)] with
    `min_stars` stars in `bins` (default `good_bins` of all the rows): the median over those stars, with
    `cterm` applied."""
    if bins is None:
        bins = good_bins([r for _, rows in per_frame for r in rows])
    coef = cterm["coefficient"] if cterm else 0.0
    dflt = cterm["default_index"] if cterm else 0.0
    out = []
    for x, rows in per_frame:
        z = [
            zp + coef * (c if np.isfinite(c) else dflt) for m, zp, _, c in rows if any(lo <= m < hi for lo, hi in bins)
        ]
        if len(z) >= min_stars:
            out.append((float(x) if x else None, float(np.median(z))))
    return out


def extinction_fit(per_frame, cterm=None, min_stars=10, min_frames=8, min_span=0.3):
    """Zeropoint above the atmosphere and extinction from per-frame photometry [(airmass, rows)]:
    ({zp0, k, k_err, frames, airmass}, notes) or (None, notes).

    The frames' zeropoints at colour 0 (`frame_zeropoints`) are fitted as zp = zp0 - k · airmass, clipped
    at 3σ twice. The fit needs `min_frames` frames spanning `min_span` in airmass, and k positive and 3σ
    from zero.
    """
    fz = [(x, z) for x, z in frame_zeropoints(per_frame, cterm, min_stars=min_stars) if x]
    X, Z = np.array([x for x, _ in fz]), np.array([z for _, z in fz])
    if len(X) < min_frames or np.ptp(X) < min_span:
        span = f"{X.min():.2f} to {X.max():.2f}" if len(X) else "none"
        return None, [f"extinction not fitted: {len(X)} frames with {min_stars}+ stars, airmass {span}"]
    keep = np.ones(len(X), bool)
    for _ in range(3):
        A = np.c_[np.ones(keep.sum()), -X[keep]]
        (zp0, k), *_ = np.linalg.lstsq(A, Z[keep], rcond=None)
        res = Z - (zp0 - k * X)
        sd = float(np.std(res[keep])) or 1e-6
        keep = np.abs(res) < 3 * sd
    A = np.c_[np.ones(keep.sum()), -X[keep]]
    k_err = float(np.sqrt(np.linalg.inv(A.T @ A)[1, 1]) * sd)
    note = (
        f"zeropoint against airmass over {keep.sum()} of {len(X)} frames (airmass {X.min():.2f} to "
        f"{X.max():.2f}): extinction {k:.3f} ± {k_err:.3f} mag/airmass, {sd:.3f} mag rms per frame"
    )
    if k <= 0 or k < 3 * k_err:
        return None, [note + ": not significant"]
    return {
        "zp0": float(zp0),
        "k": float(k),
        "k_err": k_err,
        "frames": int(keep.sum()),
        "airmass": [float(X.min()), float(X.max())],
    }, [note]


def zeropoint_and_extinction(per_frame, extinction=None):
    """Zeropoint and colour term from per-frame photometry [(airmass or None, rows)]: (camera updates,
    notes). The colour term comes from `zeropoint_from_forced` over all stars. The zeropoint is the median
    of the frames' own (`frame_zeropoints`) when 5 or more frames have enough stars: a passing cloud dims
    some frames, which moves a median over all stars but not a median over frames; else it is
    `zeropoint_from_forced`'s. For ground frames (with airmass) it is moved above the atmosphere, by
    `extinction_fit` when the frames allow it, else by `extinction` mag/airmass (0.2, typical at visible
    wavelengths, when not given) at the frames' median airmass. The updates then carry `_atmosphere`
    ({extinction_mag_per_airmass, measured, airmass}) for the caller to take out.
    """
    zps = [r for _, rows in per_frame for r in rows]
    zp, notes, cterm = zeropoint_from_forced(zps, with_colour=True)
    cam = {}
    if cterm:
        cam["color_term"] = cterm
    if zp is None:
        return cam, notes
    fz = frame_zeropoints(per_frame, cterm)
    if len(fz) >= 5:
        z = np.array([v for _, v in fz])
        lo, hi = np.percentile(z, [10, 90])
        notes.append(
            f"frames' own zeropoints at colour 0: median {np.median(z):.3f} over {len(z)} frames (p10 {lo:.3f}, "
            f"p90 {hi:.3f}); the median of frames is used, which a cloud over some frames does not move"
        )
        zp = round(float(np.median(z)), 3)
    X = [x for x, rows in per_frame if x and rows]
    if not X:
        cam["zeropoint_mag"] = zp
        return cam, notes
    fit, fnotes = extinction_fit(per_frame, cterm)
    notes += fnotes
    if fit:
        cam["zeropoint_mag"] = round(fit["zp0"], 3)
        cam["_atmosphere"] = {
            "extinction_mag_per_airmass": round(fit["k"], 3),
            "measured": True,
            "airmass": [round(v, 2) for v in fit["airmass"]],
        }
        notes.append(f"zeropoint {cam['zeropoint_mag']} above the atmosphere (the fit's intercept)")
    else:
        k = 0.2 if extinction is None else float(extinction)
        xm = float(np.median(X))
        cam["zeropoint_mag"] = round(zp + k * xm, 3)
        cam["_atmosphere"] = {"extinction_mag_per_airmass": k, "measured": False, "airmass": [round(xm, 3)]}
        notes.append(
            f"zeropoint {zp} through the atmosphere at median airmass {xm:.2f}, {cam['zeropoint_mag']} above it "
            f"with an assumed extinction of {k} mag/airmass"
            + (" (--extinction)" if extinction is not None else " (typical at visible wavelengths; give --extinction)")
        )
    return cam, notes


def zeropoint_from_forced(rows, min_snr=10.0, step=0.5, min_bin=30, with_colour=False):
    """Zeropoint from forced photometry rows [(mag, zeropoint, S/N[, colour])].

    Per magnitude bin of `step` mag the median S/N and zeropoint; bins of at least 10 stars whose median
    star reaches `min_snr` (no noise bias, no detection bias) give the zeropoint. A drift of over 0.3 mag
    across bins of `min_bin` stars means the catalogue's photometry or the response is not linear, and
    is flagged with a WARNING note.

    With `with_colour` and at least 50 stars spanning over 0.3 in colour (p10 to p90), per-star zeropoints
    are fitted against colour, zp_i = A + B · colour, clipped at 3σ in three passes. A slope 5σ from zero
    becomes Skytwin's colour term (coefficient −B: camera magnitude = m − B · colour) with zeropoint A. The
    drift check then uses zeropoints after the colour term, since bright stars may differ in colour from
    faint ones.

    Returns (zeropoint or None, notes), or (zeropoint or None, notes, colour term or None) `with_colour`.
    """

    def fail(msg):
        return (None, [msg], None) if with_colour else (None, [msg])

    if len(rows) < 20:
        return fail(f"zeropoint not measured: {len(rows)} stars")
    z = np.array([tuple(r) + (np.nan,) * (4 - len(r)) for r in rows], float)
    bins = []
    for lo in np.arange(np.floor(z[:, 0].min()), z[:, 0].max(), step):
        m = (z[:, 0] >= lo) & (z[:, 0] < lo + step)
        if m.sum() >= 10 and np.median(z[m, 2]) >= min_snr:
            bins.append((lo, float(np.median(z[m, 1])), int(m.sum())))
    if not bins:
        return fail(f"zeropoint not measured: no magnitude bin with median S/N {min_snr:.0f}")
    sel = np.concatenate([z[(z[:, 0] >= lo) & (z[:, 0] < lo + step)] for lo, _, _ in bins])
    use = sel[:, 1]
    zp = round(float(np.median(use)), 3)
    cterm, cnote = None, []
    ok = np.isfinite(sel[:, 3])
    if with_colour and ok.sum() >= 50 and np.ptp(np.percentile(sel[ok, 3], [10, 90])) > 0.3:
        c, y = sel[ok, 3], sel[ok, 1]
        keep = np.ones(len(c), bool)
        for _ in range(3):
            B, A = np.polyfit(c[keep], y[keep], 1)
            r = y - (A + B * c)
            sd = mad_sigma(r[keep], 0.0)
            keep = np.abs(r) < 3 * sd
        n = keep.sum()
        Berr = sd / np.sqrt(n) / (np.std(c[keep]) + 1e-9)
        if abs(B) > 5 * Berr:
            zp = round(float(A), 3)
            cterm = {"coefficient": round(float(-B), 4), "default_index": round(float(np.median(c)), 3)}
            use = y[keep] - B * c[keep]
            cnote = [
                f"colour term: camera mag = catalogue mag {-B:+.3f} × colour (± {Berr:.3f}) from {n} stars; "
                f"scatter {sd:.2f} mag after it"
            ]
        else:
            cnote = [f"no significant colour term ({-B:+.3f} ± {Berr:.3f})"]
    mad = mad_sigma(use, zp)
    notes = [
        f"zeropoint {zp} mag (1 e-/s, catalogue band) by forced photometry of {len(use)} stars, mag "
        f"{bins[0][0]:.1f}-{bins[-1][0] + step:.1f} (median S/N >= {min_snr:.0f}), scatter {mad:.2f} mag"
    ]
    zc = sel[:, 1].copy()
    if cterm:
        cc = np.where(np.isfinite(sel[:, 3]), sel[:, 3], cterm["default_index"])
        zc = zc + cterm["coefficient"] * cc
    full = []
    for lo, _, nb in bins:
        m = (sel[:, 0] >= lo) & (sel[:, 0] < lo + step)
        if nb >= min_bin:
            full.append((lo, float(np.median(zc[m])), nb))
    zb = [b[1] for b in full]
    if len(full) >= 3 and max(zb) - min(zb) > 0.3:
        notes.append(
            "WARNING: zeropoint by magnitude "
            + ", ".join(f"{lo:.1f}: {v:.2f}" for lo, v, _ in full)
            + ": the catalogue's photometry or the response is not linear over these stars (UCAC4's own "
            "magnitudes saturate brighter than ~8; use a Gaia extract with --catalog)"
        )
    notes += cnote
    return (zp, notes, cterm) if with_colour else (zp, notes)
