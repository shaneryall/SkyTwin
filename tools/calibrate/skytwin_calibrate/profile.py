"""First-pass profile blocks from measured statistics.

Fills, on top of a base profile (camera basics, and any blocks measured by
hand): pedestal, extra white noise and pattern amplitude as rows linked per
real frame (so their correlations survive), pattern period and angle,
temperature, read-out ramps, and a radiation rate map by sub-satellite
point. Settings that measurements recover only partly (pattern amplitude,
the radiation map) are corrected by `refine`.
"""

import copy

import numpy as np

from .util import mad_sigma, percentiles, read_json, write_json

LAT_EDGES = [-90, -60, -40, -20, 0, 20, 40, 60, 90]
LON_EDGES = list(range(-180, 181, 40))


GLARE_FLOOR_ADU = 1000.0


def usable(rows, max_background=None):
    """Rows measured without error whose background median is under `max_background` ADU (default:
    `background_cap(rows)`)."""
    if max_background is None:
        max_background = background_cap(rows)
    return [r for r in rows if "error" not in r and r["median"] < max_background]


def background_cap(rows):
    """Background (ADU) above which a frame counts as flooded by glare and is left out of the per-frame
    statistics: 1000 ADU, or twice the median frame background when that is higher. Raw frames whose
    bias alone is near or above 1000 ADU (bias not subtracted) are then kept, while frames far above the
    archive's usual level are still left out."""
    med = [r["median"] for r in rows if "error" not in r]
    return max(GLARE_FLOOR_ADU, 2.0 * float(np.median(med))) if med else GLARE_FLOOR_ADU


def pct(v):
    """Percentile table (see `util.percentiles`, 4 decimals) of the finite, non-None values in `v`; None
    when there are none."""
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return percentiles(v, 4) if len(v) else None


def area_cm2(profile):
    """Detector area in cm² (unbinned chip: width × binning × pixel pitch, likewise for height)."""
    cam = profile.get("camera", {})
    det = profile.get("detector", {})
    bx, by = det.get("binning", [1, 1])
    pitch = cam.get("pixel_pitch_um", 13.0) * 1e-4
    return cam.get("width", 1024) * bx * pitch * cam.get("height", 1024) * by * pitch


def rate_spread(ratios, counts):
    """Frame-to-frame spread of the radiation rate, as an empirical factor
    with median 1. Each measured ratio carries Poisson noise from its own
    count (variance about 1/count in the log), which the simulation adds
    again when it draws hits, so the log-ratios are shrunk toward the
    median by sqrt(1 - mean(1/count) / variance), with the variance taken
    from the MAD. Returns ({"empirical": table}, shrink factor)."""
    lr = np.log(np.asarray(ratios, float))
    lr -= np.median(lr)
    var = mad_sigma(lr, 0.0) ** 2
    pois = float(np.mean(1.0 / np.maximum(np.asarray(counts, float), 1.0)))
    k = np.sqrt(max(0.0, 1.0 - pois / var)) if var > 0 else 0.0
    return {"empirical": pct(np.exp(k * lr))}, k


def rate_map(rows, area):
    """Radiation rate (hits/cm²/s) per sub-satellite cell of `LAT_EDGES` × `LON_EDGES`, and the typical
    rate over all frames.

    Cells with under 3 frames take the median of their latitude band's known cells (or of all frames).
    """
    lat = np.array([r["meta"]["subpoint_lat_deg"] for r in rows], float)
    lon = np.array([r["meta"]["subpoint_lon_deg"] for r in rows], float)
    rate = np.array([r["sharp"] / r["meta"]["exposure_s"] / area for r in rows], float)
    cnt = np.array([r["sharp"] for r in rows], float)
    table = []
    for i in range(len(LAT_EDGES) - 1):
        row = []
        for j in range(len(LON_EDGES) - 1):
            m = (lat >= LAT_EDGES[i]) & (lat < LAT_EDGES[i + 1]) & (lon >= LON_EDGES[j]) & (lon < LON_EDGES[j + 1])
            row.append(_typical(rate[m], cnt[m]) if m.sum() >= 3 else None)
        known = [v for v in row if v is not None]
        fill = float(np.median(known)) if known else float(np.median(rate))
        table.append([round(v if v is not None else fill, 3) for v in row])
    return {"lat_edges_deg": LAT_EDGES, "lon_edges_deg": LON_EDGES, "rate_per_cm2_s": table}, _typical(rate, cnt)


def _typical(rate, counts):
    """A cell's rate: the median (unmoved by storms) when frames hold
    several hits each, else the mean (a median of a few Poisson counts is
    mostly 0 or 1)."""
    if np.median(counts) >= 5:
        return float(np.median(rate))
    return float(np.mean(rate))


def map_rate(table, lat, lon):
    """The map's rate at a sub-satellite point (as Skytwin's RateMap.rate): edge cells extend outwards, and a
    map spanning 360° of longitude wraps."""

    def cell(edges, v):
        return int(np.clip(np.searchsorted(edges, v, side="right"), 1, len(edges) - 1)) - 1

    lo = table["lon_edges_deg"]
    if lo[-1] - lo[0] >= 359.999:
        lon = lo[0] + (lon - lo[0]) % 360.0
    return table["rate_per_cm2_s"][cell(table["lat_edges_deg"], lat)][cell(lo, lon)]


def _linked(v):
    """Values tied to the real frame they came from (`empirical_linked`, group `real_frame`), 3 decimals."""
    return {"empirical_linked": {"group": "real_frame", "values": [round(float(x), 3) for x in v]}}


def build(rows, base, max_background=None, amp_cal=1.0):
    """Profile blocks from per-frame statistics `rows`, on top of a copy of `base`.

    Only error-free frames of the camera's size with background under `max_background` ADU (default:
    `background_cap`) are used.
    Each frame's extra white noise (ADU) is its noise less read noise, sky shot noise and the pattern's
    share (amplitude/√2), floored at 0.1 ADU. With a serial lag of coefficient a the white noise is
    filtered: neighbours along a row differ by 2σ²(1 − a) after it, and σ² shrinks by (1 − a)/(1 + a), so
    the pre-filter variance comes from `noise_dx`. With a field stop (`sky_adu` on every row) each frame's
    level is split into light (sky, through the flat, with shot noise) and an added level (the pedestal).
    Pedestal, sky, extra noise, pattern amplitude (times `amp_cal`), ramps and the smooth field's RMS are
    linked per real frame, so a quiet frame never gets a flare frame's structure. The radiation rate map
    holds each cell's median; storms and the edges of the belts put single frames far above it, which
    `radiation.map_scale` keeps (see `rate_spread`).
    """
    prof = copy.deepcopy(base)
    cam = prof.setdefault("camera", {})
    shape = [cam.get("height", 1024), cam.get("width", 1024)]
    rows = [r for r in usable(rows, max_background) if r["shape"] == shape]
    if not rows:
        raise ValueError(f"no usable frames of size {shape[1]} x {shape[0]}")
    read_adu = cam.get("read_noise_e", 8.0) / cam.get("gain_e_per_adu", 1.0)
    lag = ((prof.get("noise") or {}).get("serial_lag") or {}).get("coefficient", 0.0)
    gain = cam.get("gain_e_per_adu", 1.0)
    exposure = cam.get("exposure_s", 1.0) or 1.0
    split = all("sky_adu" in r for r in rows)
    keep = []
    for r in rows:
        amp = r["pattern"]["amplitude"]
        shot = r["sky_adu"] / gain if split else 0.0
        if lag and r.get("noise_dx"):
            pre2 = r["noise_dx"] ** 2 / (1 - lag) * (1 + lag) / (1 - lag)
            extra2 = pre2 - read_adu**2 - shot - (amp / np.sqrt(2)) ** 2
        else:
            extra2 = r["noise"] ** 2 - read_adu**2 - shot - (amp / np.sqrt(2)) ** 2
        keep.append(
            (
                r["add_adu"] if split else r["median"],
                float(np.sqrt(max(extra2, 0.01))),
                amp,
                r["ramp_x"],
                r["ramp_y"],
                r["sky_adu"] * gain / exposure if split else 0.0,
            )
        )
    k = np.array(keep)
    prof.setdefault("processing", {})["pedestal_adu"] = _linked(k[:, 0])
    if split:
        prof["sky_e_per_s"] = _linked(k[:, 5])
    sf = (prof.get("noise") or {}).get("smooth_field")
    if sf and any("field_rms" in r for r in rows):
        fr = [r["field_rms"] for r in rows if "field_rms" in r]
        sf["rms_adu"] = _linked([r.get("field_rms", float(np.median(fr))) for r in rows])
    noise = prof.setdefault("noise", {})
    noise["extra_white_adu"] = _linked(k[:, 1])
    per = [r["pattern"]["period"] for r in rows]
    ang = [r["pattern"]["angle"] for r in rows]
    if pct(per):
        noise["patterns"] = [
            {
                "period_px": {"empirical": pct(per)},
                "angle_deg": {"empirical": pct(ang)},
                "amplitude_adu": _linked(k[:, 2] * amp_cal),
            }
        ]
    temps = [r["meta"]["temperature_c"] for r in rows]
    if pct(temps):
        prof["temperature_c"] = {"empirical": pct(temps)}
    noise["ramp_x_adu"] = _linked(k[:, 3])
    noise["ramp_y_adu"] = _linked(k[:, 4])
    noise.pop("ramp_y_adu_per_s", None)
    pos = [
        r
        for r in rows
        if r.get("sharp") is not None and r["meta"]["exposure_s"] and r["meta"]["subpoint_lat_deg"] is not None
    ]
    if len(pos) > 30:
        area = area_cm2(prof)
        table, med = rate_map(pos, area)
        rad = prof.setdefault("radiation", {})
        rad["rate_per_cm2_s"] = round(med, 3)
        rad["rate_map"] = table
        cell = [map_rate(table, r["meta"]["subpoint_lat_deg"], r["meta"]["subpoint_lon_deg"]) for r in pos]
        ratios = [r["sharp"] / r["meta"]["exposure_s"] / area / c if c > 0 else 0.0 for r, c in zip(pos, cell)]
        ok = [(x, r["sharp"]) for x, r in zip(ratios, pos) if x > 0]
        if len(ok) > 30:
            rad["map_scale"], _ = rate_spread(*zip(*ok))
    return prof


def run(stats_path, base_path, out_path, max_background=None, amp_cal=1.0):
    """`build` from the statistics and base profile files; write the profile to `out_path` and return it."""
    prof = build(read_json(stats_path), read_json(base_path), max_background, amp_cal)
    write_json(prof, out_path, indent=1, allow_nan=False)
    return prof
