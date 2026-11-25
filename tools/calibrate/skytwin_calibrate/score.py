"""Score a detector's source list against an Skytwin truth file.

Matches detections to true stars (nearest first, within a radius), then
reports completeness by magnitude, false detections, centroid errors in
pixels, and centroid errors in units of each star's Cramér–Rao bound
(1.0 means the detector reaches the best possible precision).
"""

import csv
from itertools import pairwise

import numpy as np
from scipy.spatial import cKDTree

from .util import read_json

COLUMNS = (
    ("x", "y", None),
    ("x_image", "y_image", 1),
    ("xwin_image", "ywin_image", 1),
    ("xcentroid", "ycentroid", 0),
    ("x_centroid", "y_centroid", 0),
)
"""Position columns a source list may use, first match wins, with the pixel origin each implies:
Skytwin's and Fayde's `X`, `Y` (the file type's default origin), SExtractor's `X_IMAGE` and
`XWIN_IMAGE` (1-based), and photutils' `xcentroid` (0-based). Names match without regard to case."""

FLUX_COLUMNS = ("flux", "flux_auto", "segment_flux")
"""Flux columns a source list may use, first match wins."""


def _pick(names, path):
    """The x and y column names in `names` and the origin they imply; ValueError if none match."""
    lower = {n.lower(): n for n in names}
    for x, y, origin in COLUMNS:
        if x in lower and y in lower:
            return lower[x], lower[y], origin
    accepted = ", ".join(f"{x}/{y}" for x, y, _ in COLUMNS)
    raise ValueError(f"{path}: no position columns; expected one of {accepted} (any case), found {list(names)}")


def _flux_name(names):
    """The flux column in `names`, or None."""
    lower = {n.lower(): n for n in names}
    return next((lower[f] for f in FLUX_COLUMNS if f in lower), None)


def load_sources(path, origin=None):
    """Detections as an (n, 2) array of 0-based pixel positions, plus flux if present (else None).

    Reads a FITS table (the first table extension), a JSON list of objects, or a CSV with a header.
    The position columns are any pair in `COLUMNS`. A FITS table's `X`, `Y` (as in Fayde's SOURCES)
    are taken as 1-based and a CSV's or JSON's `x`, `y` as 0-based; SExtractor and photutils
    columns carry their own origin. `origin` overrides all of these. An empty list gives no
    detections.
    """
    p = str(path)
    if p.lower().endswith((".fits", ".fit", ".fts")):
        from astropy.io import fits

        with fits.open(p) as h:
            table = next((x for x in h[1:] if hasattr(x, "columns")), None)
            if table is None:
                raise ValueError(f"{p}: no table extension")
            names = list(table.columns.names)
            if table.data is None or len(table.data) == 0:
                return np.zeros((0, 2)), None
            cx, cy, implied = _pick(names, p)
            xy = np.c_[table.data[cx], table.data[cy]].astype(float)
            fn = _flux_name(names)
            flux = np.array(table.data[fn], float) if fn else None
        default = 1
    elif p.lower().endswith(".json"):
        rows = read_json(p)
        if not rows:
            return np.zeros((0, 2)), None
        cx, cy, implied = _pick(list(rows[0]), p)
        xy = np.array([[r[cx], r[cy]] for r in rows], float)
        fn = _flux_name(list(rows[0]))
        flux = np.array([r[fn] for r in rows], float) if fn else None
        default = 0
    else:
        with open(p, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            names = reader.fieldnames or []
        if not rows:
            return np.zeros((0, 2)), None
        cx, cy, implied = _pick(names, p)
        xy = np.array([[float(r[cx]), float(r[cy])] for r in rows])
        fn = _flux_name(names)
        flux = np.array([float(r[fn]) for r in rows]) if fn else None
        default = 0
    o = origin if origin is not None else (implied if implied is not None else default)
    return xy - o, flux


def match(truth_xy, det_xy, radius):
    """Greedy nearest-pair matching; returns (truth index, detection index) pairs."""
    if len(truth_xy) == 0 or len(det_xy) == 0:
        return []
    tree = cKDTree(det_xy)
    cand = []
    for i, p in enumerate(truth_xy):
        for j in tree.query_ball_point(p, radius):
            cand.append((float(np.hypot(*(det_xy[j] - p))), i, j))
    cand.sort()
    used_t, used_d, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in used_t and j not in used_d:
            used_t.add(i)
            used_d.add(j)
            pairs.append((i, j))
    return pairs


def _along(x0, y0, x1, y1):
    """Points every 2 px or closer along a segment, both ends included."""
    n = max(int(np.hypot(x1 - x0, y1 - y0) / 2), 1)
    return [[x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n] for i in range(n + 1)]


def _lit_sections(t):
    """The halves of target `t`'s track (start to mid, mid to end) during which it gave light.

    A half counts when its light curve has a positive rate at any sample within it; without a light
    curve every half counts.
    """
    curve = t.get("light_curve") or []
    dur = curve[-1][0] if curve else None
    out = []
    for (ka, kb), (ta, tb) in zip([("start", "mid"), ("mid", "end")], [(0.0, 0.5), (0.5, 1.0)]):
        a, b = t[ka].get("xy"), t[kb].get("xy")
        if not (a and b):
            continue
        if dur and not any(r > 0 for s, r in curve if ta * dur <= s <= tb * dur):
            continue
        out.append((a, b))
    return out


def score(truth, det_xy, radius=2.0, mag_bins=None):
    """Score detections `det_xy` (0-based px) against an Skytwin truth dict, matching within `radius` px.

    Only stars on the image count. Completeness is by magnitude bin (`mag_bins` edges, default whole
    magnitudes from floor(brightest) to floor(faintest) + 1, so a star at an integer magnitude such as
    12.0 lands in [12, 13); the last bin is closed, so a star on its top edge still counts). An
    unmatched detection within `radius` of a target, a hot or flickering pixel, a radiation hit, or a
    point along a streak or the lit part of a target's track is counted apart from the false detections;
    targets off the image or giving no light explain nothing. Centroid errors are in px and in units of
    each star's Cramér-Rao bound, |error| / √(σx² + σy²).
    """
    w, h = truth["camera"]["width"], truth["camera"]["height"]
    stars = [s for s in truth["stars"] if -0.5 <= s["x"] < w - 0.5 and -0.5 <= s["y"] < h - 0.5]
    txy = np.array([[s["x"], s["y"]] for s in stars], float).reshape(-1, 2)
    pairs = match(txy, det_xy, radius)
    found = np.zeros(len(stars), bool)
    for i, _ in pairs:
        found[i] = True
    mags = np.array([s["mag"] for s in stars])
    bins = mag_bins or list(np.arange(np.floor(mags.min()), np.floor(mags.max()) + 2, 1.0) if len(mags) else [0.0, 1.0])
    comp = []
    last = len(bins) - 2
    for k, (a, b) in enumerate(pairwise(bins)):
        m = (mags >= a) & ((mags < b) | ((mags == b) if k == last else False))
        if m.sum():
            comp.append(
                {
                    "mag_from": float(a),
                    "mag_to": float(b),
                    "stars": int(m.sum()),
                    "found": int(found[m].sum()),
                    "completeness": float(found[m].mean()),
                }
            )
    d = np.array([det_xy[j] - txy[i] for i, j in pairs]).reshape(-1, 2)
    norm = [
        np.hypot(*(det_xy[j] - txy[i])) / np.hypot(*stars[i]["crlb_px"]) for i, j in pairs if stars[i].get("crlb_px")
    ]
    lit_targets = [t for t in truth.get("targets", []) if t.get("on_image", True) and t.get("flux_e", 1.0) > 0]
    known = [[t["mid"]["xy"][0], t["mid"]["xy"][1]] for t in lit_targets if t["mid"].get("xy")]
    inst = truth.get("instrument", {})
    known += [[p["x"], p["y"]] for p in inst.get("hot_pixels", []) + inst.get("flickering_pixels", [])]
    rad = inst.get("radiation") or {}
    known += [[0.5 * (k["start"][0] + k["end"][0]), 0.5 * (k["start"][1] + k["end"][1])] for k in rad.get("hits", [])]
    for k in truth.get("streaks", []):
        known += _along(*k["start"], *k["end"])
    for t in lit_targets:
        for a, b in _lit_sections(t):
            known += _along(*a, *b)
    matched = {j for _, j in pairs}
    rest = [j for j in range(len(det_xy)) if j not in matched]
    explained = 0
    if rest and known:
        kt = cKDTree(np.array(known, float))
        explained = sum(1 for j in rest if kt.query(det_xy[j])[0] <= radius)
    return {
        "stars_in_frame": len(stars),
        "detections": len(det_xy),
        "matched": len(pairs),
        "completeness_by_mag": comp,
        "false_detections": len(rest) - explained,
        "detections_on_defects_or_targets": explained,
        "centroid_error_px": {
            "median": float(np.median(np.hypot(*d.T))) if len(d) else None,
            "rms_x": float(np.sqrt(np.mean(d[:, 0] ** 2))) if len(d) else None,
            "rms_y": float(np.sqrt(np.mean(d[:, 1] ** 2))) if len(d) else None,
            "bias_x": float(np.mean(d[:, 0])) if len(d) else None,
            "bias_y": float(np.mean(d[:, 1])) if len(d) else None,
        },
        "error_over_crlb": {
            "median": float(np.median(norm)) if norm else None,
            "rms": float(np.sqrt(np.mean(np.square(norm)))) if norm else None,
            "stars": len(norm),
        },
    }


def markdown(res, title="Detection score"):
    """The `score` result as a Markdown summary."""
    c = res["centroid_error_px"]
    e = res["error_over_crlb"]
    lines = [
        f"# {title}",
        "",
        f"Stars in frame: {res['stars_in_frame']}. Detections: {res['detections']}. Matched: {res['matched']}.",
        f"False detections: {res['false_detections']} (another {res['detections_on_defects_or_targets']} sit on "
        "streaks, targets, hot or flickering pixels, or radiation hits).",
        "",
        "| Magnitude | Stars | Found | Completeness |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {b['mag_from']:.0f}–{b['mag_to']:.0f} | {b['stars']} | {b['found']} | {b['completeness']:.0%} |"
        for b in res["completeness_by_mag"]
    ]
    if c["median"] is not None:
        lines += [
            "",
            f"Centroid error: median {c['median']:.3f} px; RMS x {c['rms_x']:.3f}, y {c['rms_y']:.3f} px; "
            f"bias ({c['bias_x']:+.3f}, {c['bias_y']:+.3f}) px.",
        ]
    if e["median"] is not None:
        lines += [
            f"Centroid error over its Cramér–Rao bound, |error| / √(σx² + σy²) ({e['stars']} stars): "
            f"median {e['median']:.2f}, RMS {e['rms']:.2f}. "
            "A detector that reaches the bound has an RMS of 1; above 1 it loses precision."
        ]
    return "\n".join(lines) + "\n"
