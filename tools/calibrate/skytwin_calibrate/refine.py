"""Close the loop: simulate with a profile, measure the same way, correct.

Two corrections:
- global: the pattern amplitude, which the FFT measure recovers only partly,
  is scaled by real/simulated medians;
- twins: when the keyword map gives a time and a spacecraft state, each
  sampled real frame gets a simulated twin at the same time, position,
  exposure and temperature; the radiation map is corrected cell by cell by
  the real/simulated ratio of sharp events.
"""

import contextlib
import os
import random

import numpy as np

from . import measure
from . import profile as prof_mod
from .util import read_json, run_skytwin, write_json

RANDOM_STARS = {"random_stars": {"count": 150, "max_flux_e": 3e6, "min_flux_e": 2e4}}
"""Stars for simulated frames when no catalogue is given."""


def scale_linked(block, factor):
    """Multiply the values of an `empirical_linked` block by `factor` in place (3 decimals)."""
    vals = block["empirical_linked"]["values"]
    block["empirical_linked"]["values"] = [round(v * factor, 3) for v in vals]


def _state_vector(m):
    """The scene's `observer.state_vector` from a frame's header values."""
    return {
        "state_vector": {
            "epoch_utc": m["time_utc"],
            "frame": m["state_frame"],
            "position_km": m["position_km"],
            "velocity_km_s": m["velocity_km_s"],
        }
    }


def iterate(profile_path, real_rows, km, skytwin, work, n_global=60, n_twins=200, stars=None, seed=1):
    """One refinement pass over the profile at `profile_path`, rewritten in place; returns what changed.

    The pattern amplitude is scaled by the real/simulated ratio of median measured amplitudes over
    `n_global` frames without radiation or hot pixels. With a radiation rate map and over 30 real frames
    with time, state, exposure, sub-satellite point and a sharp count, up to `n_twins` of them get twins
    with radiation only (no hot pixels, which the real count leaves out) at the real frame's levels; each
    map cell is scaled by the median real/simulated ratio of sharp events of its twins (at least 3, else
    the overall ratio).
    """
    os.makedirs(work, exist_ok=True)
    prof = read_json(profile_path)
    real = prof_mod.usable(real_rows)
    report = {}
    pats = prof.get("noise", {}).get("patterns", [])
    if pats and "empirical_linked" in pats[0]["amplitude_adu"]:
        sim = []
        for s in range(n_global):
            f = os.path.join(work, f"g{s}.fits")
            scene = {
                "seed": seed * 1000 + s,
                "instrument": os.path.abspath(profile_path),
                "overrides": {"radiation": None, "hot_pixels": None},
                "scene": stars or RANDOM_STARS,
            }
            run_skytwin(skytwin, scene, f)
            sim.append(measure.frame(f, km))
        r_amp = np.median([r["pattern"]["amplitude"] for r in real])
        s_amp = np.median([r["pattern"]["amplitude"] for r in sim])
        factor = float(r_amp / s_amp) if s_amp > 0 else 1.0
        scale_linked(pats[0]["amplitude_adu"], factor)
        report["pattern_amplitude_factor"] = factor
    rad = prof.get("radiation", {})
    twins = [
        r
        for r in real
        if r["meta"]["time_utc"]
        and r["meta"]["position_km"]
        and r["meta"]["velocity_km_s"]
        and r["meta"]["exposure_s"]
        and r["meta"]["subpoint_lat_deg"] is not None
        and r.get("sharp") is not None
    ]
    if rad.get("rate_map") and len(twins) > 30:
        random.seed(seed)
        twins = random.sample(twins, min(n_twins, len(twins)))
        pairs = []
        write_json(prof, os.path.join(work, "profile.json"), allow_nan=False)
        for i, r in enumerate(twins):
            m = r["meta"]
            f = os.path.join(work, f"t{i}.fits")
            ov = {
                "camera": {"exposure_s": m["exposure_s"]},
                "hot_pixels": None,
                "processing": {"pedestal_adu": r.get("add_adu", r["median"])},
            }
            if "sky_adu" in r:
                gain = prof.get("camera", {}).get("gain_e_per_adu", 1.0)
                ov["sky_e_per_s"] = r["sky_adu"] * gain / m["exposure_s"]
            if m["temperature_c"] is not None:
                ov["temperature_c"] = m["temperature_c"]
            scene = {
                "seed": seed * 100000 + i,
                "instrument": os.path.abspath(os.path.join(work, "profile.json")),
                "overrides": ov,
                "scene": {"time": {"start_utc": m["time_utc"]}, "observer": _state_vector(m)},
            }
            run_skytwin(skytwin, scene, f)
            sim_sharp = measure.frame(f, km)["sharp"]
            if sim_sharp is not None:
                pairs.append((m["subpoint_lat_deg"], m["subpoint_lon_deg"], r["sharp"], sim_sharp))
        if not pairs:
            report["radiation"] = "no twin gave a sharp-event count; the rate map is unchanged"
        else:
            table = rad["rate_map"]["rate_per_cm2_s"]
            late, lone = rad["rate_map"]["lat_edges_deg"], rad["rate_map"]["lon_edges_deg"]
            la, lo, re_, si = (np.array(v, float) for v in zip(*pairs))
            glob = float(np.median(np.maximum(re_, 1) / np.maximum(si, 1)))
            for i in range(len(late) - 1):
                for j in range(len(lone) - 1):
                    m = (la >= late[i]) & (la < late[i + 1]) & (lo >= lone[j]) & (lo < lone[j + 1])
                    ratio = float(np.median(np.maximum(re_[m], 1) / np.maximum(si[m], 1))) if m.sum() >= 3 else glob
                    table[i][j] = round(table[i][j] * ratio, 3)
            report["radiation_ratio_before"] = glob
    write_json(prof, profile_path, indent=1, allow_nan=False)
    return report


def twins(profile_path, real_rows, km, skytwin, work, n=300, seed=7, stars=None, catalog=None, keep=10):
    """Simulated twins of up to `n` real frames, measured; for a realism report.

    Each twin takes only the real frame's time, spacecraft state, exposure and temperature; everything
    else is drawn from the profile, so nothing is matched by construction. Stars trail at the frame's
    tracking rate when its meta has one (from the header, or measured by `learn`), in a random direction.
    With `catalog` and a full pointing the twin shows the real sky: catalogue stars at the real pointing.
    Without a catalogue the real pointing (roll 0 if its convention is unknown) still sets the Earth, Sun
    and Moon angles for stray light. The first `keep` twins stay on disk to look at; the rest are deleted
    once measured, since real-sky truth files list every catalogue star and run to tens of MB each.
    """
    os.makedirs(work, exist_ok=True)
    real = [r for r in prof_mod.usable(real_rows, 1e9) if r["meta"]["exposure_s"]]
    random.seed(seed)
    pick = random.sample(real, min(n, len(real)))
    out = []
    for i, r in enumerate(pick):
        m = r["meta"]
        ov = {"camera": {"exposure_s": m["exposure_s"]}}
        if m["temperature_c"] is not None:
            ov["temperature_c"] = m["temperature_c"]
        sc = dict(stars or RANDOM_STARS)
        p = m.get("pointing")
        if m.get("trail_rate_arcsec_per_s"):
            ov["pointing"] = {"drift_arcsec_per_s": m["trail_rate_arcsec_per_s"]}
        if catalog and p and p.get("roll_deg") is not None and m["time_utc"]:
            sc = {"pointing": p, "catalog": dict(catalog)}
        elif p and m["time_utc"]:
            sc["pointing"] = dict(p, roll_deg=p["roll_deg"] if p.get("roll_deg") is not None else 0.0)
        if m["time_utc"] and m["position_km"] and m["velocity_km_s"]:
            sc["time"] = {"start_utc": m["time_utc"]}
            sc["observer"] = _state_vector(m)
        f = os.path.join(work, f"tw{i}.fits")
        scene = {"seed": seed * 100000 + i, "instrument": os.path.abspath(profile_path), "overrides": ov, "scene": sc}
        run_skytwin(skytwin, scene, f)
        out.append(measure.frame(f, km))
        if i >= keep:
            _remove_outputs(f)
    return out


def _remove_outputs(f):
    """Delete a simulated frame `f` and the truth, provenance and scene files beside it."""
    stem = f[:-5]
    for p in (f, stem + ".truth.json", stem + ".provenance.json", f + ".json"):
        with contextlib.suppress(OSError):
            os.remove(p)
