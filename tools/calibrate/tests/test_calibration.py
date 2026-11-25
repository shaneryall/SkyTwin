"""Calibration frames: bias, darks and flats of a known camera -> its gain, read noise, dark current,
radiation and dust back."""

from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits

from skytwin_calibrate import calib

KM = {"exposure_s": "EXPTIME", "temperature_c": "CCD-TEMP", "time_utc": "DATE-OBS"}


def ring_flat(n, rings, vignetting=-0.1):
    """Response map of an `n` × `n` chip: radial vignetting and dust rings (x, y, outer radius, depth, hole
    at half the radius)."""
    yy, xx = np.mgrid[0:n, 0:n]
    rho2 = ((xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2) / (0.5 * np.hypot(n, n)) ** 2
    f = 1 + vignetting * rho2
    for x, y, R, depth in rings:
        r = np.hypot(xx - x, yy - y)
        f = f * np.where((r <= R) & (r >= 0.5 * R), 1 - depth, 1.0)
    return f


def write_calibration(
    d,
    n=256,
    gain=1.7,
    read_adu=10.0,
    bias=1000.0,
    dark_e_s=0.5,
    hits_per_s=3.0,
    readout_s=4.0,
    temp=-20.0,
    rings=(),
    nights=2,
    seed=1,
):
    """Bias, dark and flat frames of a camera, written under `d` (`Bias`, `Darks`, `Flats` folders, with
    IMAGETYP, EXPTIME and EXPOSURE, CCD-TEMP and DATE-OBS). Per night: 6 bias frames, 5 darks each of 2, 5 and 10 s, and
    flats in pairs at four levels. Darks hold dark current, 20 hot pixels at 40 e/s, radiation hits whose
    density rises toward the top rows (row 0 read first, `readout_s` to read the frame), and 5 flickering
    pixels high in two darks each."""
    rng = np.random.default_rng(seed)
    cols = rng.normal(0.0, 2.0, n)
    hy, hx = rng.integers(0, n, 20), rng.integers(0, n, 20)
    resp = ring_flat(n, rings)
    for sub in ("Bias", "Darks", "Flats"):
        (d / sub).mkdir(parents=True, exist_ok=True)

    def frame(sub, name, kind, exp, t, e):
        a = bias + cols[None, :] + e / gain + rng.normal(0.0, read_adu, (n, n))
        h = fits.Header(
            {
                "IMAGETYP": kind,
                "EXPTIME": exp,
                "EXPOSURE": exp,
                "CCD-TEMP": temp,
                "DATE-OBS": t.isoformat(timespec="seconds"),
            }
        )
        fits.writeto(d / sub / name, np.clip(np.round(a), 0, 65535).astype(np.uint16), h)

    for night in range(nights):
        t = datetime(2022, 11, 8 + 2 * night, 23, 0, 0)
        for k in range(6):
            frame("Bias", f"n{night}-bias-{k}.fit", "BIAS", 0.0, t, 0.0)
            t += timedelta(seconds=20)
        flick = [(rng.integers(0, n), rng.integers(0, n), set(rng.choice(5, 2, replace=False))) for _ in range(5)]
        for exp in (2.0, 5.0, 10.0):
            for k in range(5):
                e = rng.poisson(dark_e_s * exp, (n, n)).astype(float)
                e[hy, hx] += rng.poisson(40.0 * exp, 20)
                for _ in range(rng.poisson(hits_per_s * (exp + 0.5 * readout_s))):
                    u = (-exp + np.sqrt(exp**2 + 2 * readout_s * rng.uniform() * (exp + 0.5 * readout_s))) / readout_s
                    y, x = min(int(u * n), n - 1), int(rng.integers(0, n - 1))
                    e[y, x : x + 2] += 400.0
                if exp == 10.0:
                    for y, x, which in flick:
                        if k in which:
                            e[y, x] += 300.0 * gain
                frame("Darks", f"n{night}-dark-{exp:04.1f}s-{k}.fit", "DARK", exp, t, e)
                t += timedelta(seconds=30)
        for level in (3000.0, 8000.0, 15000.0, 25000.0):
            for k in range(4):
                e = rng.poisson(level * gain * resp * (1 + 0.003 * k)).astype(float)
                frame("Flats", f"n{night}-flat-{level:05.0f}-{k}.fit", "FLAT", level / 1000.0, t, e)
                t += timedelta(seconds=15)


def _sorted(d, n=256):
    files = sorted(str(p) for p in d.rglob("*.fit"))
    return calib.sort_frames(files, KM, (n, n), lambda h: (h["NAXIS1"], h["NAXIS2"]))


def test_frames_are_sorted_by_header_then_folder(tmp_path):
    assert calib.kind_of("/x/a.fit", {"IMAGETYP": "Bias Frame"}) == "bias"
    assert calib.kind_of("/x/a.fit", {"IMAGETYP": "Light Frame"}) == "light"
    assert calib.kind_of("/data/Darks/a-001.fit", {}) == "dark"
    assert calib.kind_of("/data/night1/flat_g_01.fits", {}) == "flat"
    assert calib.kind_of("/data/night1/m31_01.fits", {}) is None
    write_calibration(tmp_path, n=64, nights=2)
    fits.writeto(tmp_path / "Flats" / "big.fit", np.zeros((128, 128), np.uint16), fits.Header({"IMAGETYP": "FLAT"}))
    frames, notes = _sorted(tmp_path, 64)
    assert [len(frames[k]) for k in ("bias", "dark", "flat")] == [12, 30, 32]
    assert {r["night"] for r in frames["bias"]} == {0, 1}
    assert "2 night(s)" in notes[0] and "1 flat of another size" in notes[0]


def test_bias_and_flats_give_read_noise_and_gain(tmp_path):
    write_calibration(tmp_path)
    frames, _ = _sorted(tmp_path)
    rn, masters, notes = calib.read_noise(frames["bias"])
    assert abs(rn / 10.0 - 1) < 0.03, (rn, notes)
    assert set(masters) == {0, 1}
    g, notes = calib.flat_gain(frames["flat"], masters, rn)
    assert abs(g / 1.7 - 1) < 0.04, (g, notes)
    g_free, notes = calib.flat_gain(frames["flat"], {}, None)
    assert abs(g_free / 1.7 - 1) < 0.06 and "no bias frames" in notes[0], (g_free, notes)


def test_darks_give_dark_current_hot_pixels_and_radiation(tmp_path):
    write_calibration(tmp_path, dark_e_s=0.5, hits_per_s=3.0, readout_s=4.0)
    frames, _ = _sorted(tmp_path)
    _, masters, _ = calib.read_noise(frames["bias"])
    rate, dark, temp, notes = calib.dark_current(frames["dark"], masters, 1.7)
    assert abs(dark / 0.5 - 1) < 0.1 and temp == -20.0, (dark, notes)
    assert "10 s less 2 s" in notes[0]
    blk, _ = calib.hot_from_darks(rate, dark, 20.0)
    assert blk["density_per_mpix"] == round(20 / 256**2 * 1e6, 2), blk
    assert abs(np.median(blk["e_per_s_at_ref"]["empirical"]) - 40.0) < 3.0

    cr, notes = calib.cosmic_rays(frames["dark"], 1.0)
    assert abs(cr["rate_per_cm2_s"] / 3.0 - 1) < 0.15, notes
    assert abs(cr["readout_s"] / 4.0 - 1) < 0.35, notes
    assert cr["first_row"] == "bottom", notes
    assert "fire in two or more darks" in notes[0] and not notes[0].startswith("0 ")


def test_one_dark_exposure_counts_read_out_hits_in_the_rate(tmp_path):
    write_calibration(tmp_path, n=128, hits_per_s=3.0, readout_s=4.0, nights=1)
    frames, _ = _sorted(tmp_path, 128)
    tens = [r for r in frames["dark"] if r["exposure_s"] == 10.0]
    cr, notes = calib.cosmic_rays(tens, 1.0)
    assert not cr["measured_readout"] and "readout_s" not in cr
    assert abs(cr["rate_per_cm2_s"] / (3.0 * 12 / 10) - 1) < 0.25, notes


def test_dust_shadows_in_the_master_flat(tmp_path):
    """All 12 flats are used, since 3000 ADU is under 8% of the 16-bit range, and the four dust rings
    of 30 and 40 image px come back as 60 to 80 chip px at 2 × 2 binning."""
    rings = [(150, 150, 30, 0.006), (350, 160, 30, 0.004), (250, 300, 40, 0.008), (140, 360, 30, 0.005)]
    write_calibration(tmp_path, n=512, rings=rings, nights=1)
    frames, _ = _sorted(tmp_path, 512)
    _, masters, _ = calib.read_noise(frames["bias"])
    F, used = calib.master_flat(frames["flat"], masters)
    assert used == 12
    blk, notes = calib.dust(F, (2, 2))
    assert blk is not None, notes
    radii = blk["radius_px"]["empirical"]
    assert len(radii) == 4, notes
    assert all(50 <= r <= 90 for r in radii), radii
    assert 0.003 < np.median(blk["depth"]["empirical"]) < 0.009, blk["depth"]
    assert 0.3 <= blk["hole_fraction"] <= 0.7, blk
