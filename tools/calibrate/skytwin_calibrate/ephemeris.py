"""Spacecraft states from an ephemeris table, for frames whose headers carry none.

`skytwin-calibrate learn --ephemeris FILE` takes either

- a CSV with columns `time_utc` (ISO 8601, UTC), `x_km`, `y_km`, `z_km`, `vx_km_s`, `vy_km_s`, `vz_km_s`
  (geocentric, GCRS), or
- a JPL Horizons vector table, as the Horizons API returns it (JSON or plain text) or as the web page
  shows it: CSV rows between `$$SOE` and `$$EOE`, geocentric (centre 500@399), ICRF, in km and km/s or
  au and au/day, times as JDTDB or JDUT. Geocentric ICRF and GCRS axes agree to well under a
  milliarcsecond, which no frame resolves.

Each frame gets the state at its keyword map's `time_utc` by cubic Hermite interpolation between the
table rows on either side, positions and velocities both, and, when its headers give no sub-satellite
point, the geodetic latitude and longitude under the spacecraft (radiation maps are binned by it).
Frames outside the table keep no state.
"""

import csv
import json
import re

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy.time import Time

AU_KM = 149597870.7
DAY_S = 86400.0


def _horizons_rows(text):
    """(times POSIX s, states N×6 km and km/s, notes) from a Horizons vector table's text."""
    head, _, rest = text.partition("$$SOE")
    body = rest.partition("$$EOE")[0]
    au = re.search(r"Output units\s*:\s*AU-D", head, re.IGNORECASE) is not None
    scale = "utc" if re.search(r"JDUT", head) else "tdb"
    notes = []
    if not re.search(r"Center body name:\s*Earth|Center-site name:\s*GEOCENTRIC|500@399", head, re.IGNORECASE):
        notes.append("WARNING: the table does not say it is geocentric; Skytwin needs geocentric states")
    jd, st = [], []
    for line in body.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 8:
            continue
        jd.append(float(f[0]))
        st.append([float(x) for x in f[2:8]])
    st = np.array(st, float)
    if au:
        st[:, :3] *= AU_KM
        st[:, 3:] *= AU_KM / DAY_S
    t = Time(jd, format="jd", scale=scale).utc.unix
    notes.insert(
        0, f"JPL Horizons table, {len(jd)} rows, times JD{'UT' if scale == 'utc' else 'TDB'}, {'au' if au else 'km'}"
    )
    return np.asarray(t, float), st, notes


def load(path):
    """(times POSIX s UTC, states N×6, notes) from the table at `path` (see the module notes), sorted by time."""
    with open(path) as fh:
        text = fh.read()
    if text.lstrip().startswith("{"):
        text = json.loads(text).get("result", "")
    if "$$SOE" in text:
        t, st, notes = _horizons_rows(text)
    else:
        rows = list(csv.DictReader(text.splitlines()))
        t = Time([r["time_utc"].replace("Z", "") for r in rows], format="isot", scale="utc").unix
        keys = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")
        st = np.array([[float(r[k]) for k in keys] for r in rows], float)
        notes = [f"CSV table, {len(rows)} rows"]
    order = np.argsort(t)
    return np.asarray(t, float)[order], st[order], notes


def state_at(times, states, t, slack_s=1.0):
    """(position km, velocity km/s) at POSIX time `t` by cubic Hermite interpolation, or None outside the
    table (by more than `slack_s`)."""
    if len(times) < 2 or t < times[0] - slack_s or t > times[-1] + slack_s:
        return None
    i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
    t0, t1 = times[i], times[i + 1]
    dt = t1 - t0
    s = (t - t0) / dt
    p0, v0, p1, v1 = states[i, :3], states[i, 3:], states[i + 1, :3], states[i + 1, 3:]
    h = (2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2)
    dh = (6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s)
    pos = h[0] * p0 + h[1] * dt * v0 + h[2] * p1 + h[3] * dt * v1
    vel = (dh[0] * p0 + dh[1] * dt * v0 + dh[2] * p1 + dh[3] * dt * v1) / dt
    return pos, vel


def subpoint(pos_km, time_utc):
    """(geodetic latitude, longitude) in degrees under a GCRS position (km) at ISO time `time_utc`."""
    t = Time(time_utc, scale="utc")
    g = GCRS(CartesianRepresentation(*(np.asarray(pos_km) * u.km)), obstime=t)
    itrs = g.transform_to(ITRS(obstime=t))
    loc = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
    return float(loc.lat.deg), float(((loc.lon.deg + 180.0) % 360.0) - 180.0)


def apply(metas, times, states):
    """Give each frame meta (dicts from `keywords.frame_meta`) without a state the table's state at its
    `time_utc`, in GCRS, and its sub-satellite point when it has none. Returns how many got a state."""
    n = 0
    for m in metas:
        if m is None or m.get("position_km") or not m.get("time_utc"):
            continue
        try:
            t = float(Time(str(m["time_utc"]).replace("Z", ""), scale="utc").unix)
        except ValueError:
            continue
        sv = state_at(times, states, t)
        if sv is None:
            continue
        m["position_km"] = [round(float(x), 4) for x in sv[0]]
        m["velocity_km_s"] = [round(float(x), 7) for x in sv[1]]
        m["state_frame"] = "gcrs"
        if m.get("subpoint_lat_deg") is None:
            m["subpoint_lat_deg"], m["subpoint_lon_deg"] = subpoint(sv[0], str(m["time_utc"]).replace("Z", ""))
        n += 1
    return n
