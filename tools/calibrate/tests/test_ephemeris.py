"""Spacecraft states from an ephemeris table: interpolation, the Horizons format, and frames given states."""

import json

import numpy as np
from astropy.time import Time, TimeDelta

from skytwin_calibrate import ephemeris

MU = 398600.4418
R = 7000.0
W = np.sqrt(MU / R**3)


def orbit(t):
    """A circular orbit in the x-y plane inclined 50 degrees about x: (position km, velocity km/s)."""
    c, s, i = np.cos(W * t), np.sin(W * t), np.radians(50.0)
    p = R * np.array([c, s * np.cos(i), s * np.sin(i)])
    v = R * W * np.array([-s, c * np.cos(i), c * np.sin(i)])
    return p, v


T0 = Time("2010-03-01T00:00:00", scale="utc")


def write_csv(path, step=60.0, n=61):
    lines = ["time_utc,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s"]
    for k in range(n):
        p, v = orbit(k * step)
        t = (T0 + TimeDelta(k * step, format="sec")).isot
        lines.append(",".join([t, *(f"{x:.6f}" for x in p), *(f"{x:.9f}" for x in v)]))
    path.write_text("\n".join(lines))


def test_interpolation_between_rows_is_under_a_metre(tmp_path):
    """Rows a minute apart on a 7000 km orbit: positions to a metre, velocities to 10 cm/s."""
    write_csv(tmp_path / "e.csv")
    times, states, notes = ephemeris.load(tmp_path / "e.csv")
    assert len(times) == 61 and "CSV" in notes[0]
    for dt in (13.7, 600.0, 1234.5, 3599.0):
        p, v = ephemeris.state_at(times, states, T0.unix + dt)
        tp, tv = orbit(dt)
        assert np.linalg.norm(p - tp) < 1e-3 and np.linalg.norm(v - tv) < 1e-4, (dt, p - tp, v - tv)
    assert ephemeris.state_at(times, states, T0.unix - 100) is None


def _horizons(rows_jd, states, scale="TDB", units="KM-S"):
    """A Horizons vector table's text around the given rows."""
    head = (
        "*******************************************************************************\n"
        " Revised: Jan 01, 2020            WISE (spacecraft)                  -163\n"
        f"Center body name: Earth (399)                     {{source: DE441}}\n"
        f"Output units    : {units}\n"
        "*******************************************************************************\n"
        f"            JD{scale},            Calendar Date ({scale}),                      X,                      Y,"
        "                      Z,                     VX,                     VY,                     VZ,\n"
    )
    body = "".join(
        f"{jd:.9f}, A.D. 2010-Mar-01 00:00:00.0000, " + ", ".join(f"{x:.12E}" for x in st) + ",\n"
        for jd, st in zip(rows_jd, states)
    )
    return head + "$$SOE\n" + body + "$$EOE\n"


def test_horizons_tables_in_tdb_and_au(tmp_path):
    t = [T0 + TimeDelta(k * 60.0, format="sec") for k in range(11)]
    st = [np.r_[orbit(k * 60.0)] for k in range(11)]
    (tmp_path / "h.txt").write_text(_horizons([x.tdb.jd for x in t], st))
    times, states, notes = ephemeris.load(tmp_path / "h.txt")
    assert abs(times[0] - T0.unix) < 1e-3 and np.allclose(states[3], np.r_[orbit(180.0)]), notes
    au = [np.r_[s[:3] / ephemeris.AU_KM, s[3:] / ephemeris.AU_KM * ephemeris.DAY_S] for s in st]
    res = json.dumps({"result": _horizons([x.utc.jd for x in t], au, scale="UT", units="AU-D")})
    (tmp_path / "h.json").write_text(res)
    times, states, notes = ephemeris.load(tmp_path / "h.json")
    assert abs(times[5] - (T0.unix + 300)) < 1e-3 and np.allclose(states[5], np.r_[orbit(300.0)], rtol=1e-9), notes
    assert "JDUT" in notes[0] and not any("WARNING" in n for n in notes)


def test_frames_get_states_and_subpoints(tmp_path):
    write_csv(tmp_path / "e.csv")
    times, states, _ = ephemeris.load(tmp_path / "e.csv")
    metas = [
        {"time_utc": (T0 + TimeDelta(dt, format="sec")).isot, "position_km": None, "subpoint_lat_deg": None}
        for dt in (100.0, 2000.0, 9000.0)
    ]
    assert ephemeris.apply(metas, times, states) == 2
    p, _ = orbit(100.0)
    m = metas[0]
    assert np.allclose(m["position_km"], p, atol=1e-3) and m["state_frame"] == "gcrs"
    geocentric = np.degrees(np.arcsin(p[2] / np.linalg.norm(p)))
    assert abs(m["subpoint_lat_deg"] - geocentric) < 0.3 and -180 <= m["subpoint_lon_deg"] <= 180, m
    assert metas[2]["position_km"] is None
