"""Measurements over an archive of frames: photon transfer and hot pixels."""

from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits

from skytwin_calibrate import analyse


def _write(path, d):
    fits.writeto(path, np.clip(np.round(d), 0, 65535).astype(np.uint16))
    return str(path)


def _raw_archive(tmp_path, gain=1.07, read_e=10.0, bias=1014.0, n=256, seed=3):
    """Two nights of raw frames (bias not subtracted), as a satellite-tracking archive takes them:
    12 bias-only 0.1 s frames 4.6 s apart, and 12 frames of 3 s with 40 ADU of sky 7.4 s apart."""
    rng = np.random.default_rng(seed)
    files, times, exps = [], [], []
    for night, (count, exp, sky_adu, step) in enumerate([(12, 0.1, 0.0, 4.6), (12, 3.0, 40.0, 7.4)]):
        t0 = datetime(2020, 12, 20 + 4 * night, 2, 0, 0)
        for k in range(count):
            e = rng.poisson(sky_adu * gain, (n, n)) + rng.normal(0.0, read_e, (n, n))
            files.append(_write(tmp_path / f"n{night}_{k:02d}.fits", bias + e / gain))
            times.append((t0 + timedelta(seconds=k * step)).isoformat())
            exps.append(exp)
    return files, times, exps


def test_photon_transfer_pairs_frames_at_the_archive_cadence(tmp_path):
    """Frames 7.4 s apart pair (the limit follows the archive's spacing, not a fixed 5 s), and on raw
    frames the read noise comes from the darkest frames: the fit's intercept there is
    read² - bias / gain, far below zero."""
    files, times, exps = _raw_archive(tmp_path)
    gain, read, notes = analyse.photon_transfer(files, times, exposures=exps)
    assert gain is not None, notes
    assert abs(gain / 1.07 - 1) < 0.08, (gain, notes)
    assert abs(read / (10.0 / 1.07) - 1) < 0.1, (read, notes)
    assert "not bias-subtracted" in notes[0]

    g5, _, notes5 = analyse.photon_transfer(files, times, max_dt_s=5.0, exposures=exps)
    assert g5 is None and "levels span only" in notes5[0]


def test_exposure_group_keeps_the_longest_exposures():
    keep, note = analyse.exposure_group([0.1] * 66 + [3.0] * 93)
    assert keep == list(range(66, 159))
    assert "93 frames of 3 s" in note and "66 frames of other exposures" in note

    keep, note = analyse.exposure_group([0.1] * 66 + [3.0] * 5)
    assert keep == list(range(66)) and "5 frames of other exposures" in note

    spread = [1.0 + 0.05 * k for k in range(21)]
    assert analyse.exposure_group(spread) == (list(range(21)), None)
    assert analyse.exposure_group([None] + [3.0] * 12) == (list(range(1, 13)), None)


def test_hot_pixels_come_from_one_exposure(tmp_path):
    """A row keeps collecting dark charge while it waits to be read. In a 0.1 s frame that charge,
    divided by 0.1 s, reads as a rate ten times too high, and a median over both exposures
    inflates every hot pixel; the 3 s group gives the rate the 3 s frames show."""
    rng = np.random.default_rng(5)
    n, rate, wait_e = 64, 20.0, 40.0
    ys, xs = rng.integers(8, n - 8, 8), rng.integers(8, n - 8, 8)
    files, metas = [], []
    for k, exp in enumerate([0.1] * 12 + [3.0] * 12):
        d = 1000.0 + rng.normal(0.0, 5.0, (n, n))
        d[ys, xs] += rate * exp + wait_e
        files.append(_write(tmp_path / f"f{k:02d}.fits", d))
        metas.append({"exposure_s": exp, "temperature_c": -20.0})

    keep, note = analyse.exposure_group([m["exposure_s"] for m in metas])
    assert keep == list(range(12, 24)) and note
    hp, notes = analyse.hot_pixels([files[i] for i in keep], [metas[i] for i in keep], 1.0)
    rates = hp["hot_pixels"]["e_per_s_at_ref"]["empirical"]
    expected = (rate * 3.0 + wait_e) / 3.0
    assert all(abs(r - expected) < 3.0 for r in rates), (rates, notes)

    mixed, _ = analyse.hot_pixels(files, metas, 1.0)
    assert np.median(mixed["hot_pixels"]["e_per_s_at_ref"]["empirical"]) > 3 * expected


def test_hot_pixel_rates_move_to_the_median_temperature(tmp_path):
    """Frames at two temperatures, half at 0 °C and half at 10 °C, with rates doubling every 6 °C:
    the rates come back at the median temperature, 5 °C. An even spread of temperatures hides a
    wrong sign in that move; two groups don't."""
    rng = np.random.default_rng(7)
    n, exp, doubling = 96, 10.0, 6.0
    ys, xs = rng.integers(8, n - 8, 60), rng.integers(8, n - 8, 60)
    base = rng.uniform(200.0, 400.0, 60)
    files, metas = [], []
    for k, temp in enumerate([0.0] * 10 + [10.0] * 10):
        d = 1000.0 + rng.normal(0.0, 5.0, (n, n))
        d[ys, xs] += base * 2 ** (temp / doubling) * exp
        files.append(_write(tmp_path / f"t{k:02d}.fits", d))
        metas.append({"exposure_s": exp, "temperature_c": temp})
    hp, notes = analyse.hot_pixels(files, metas, 1.0)
    rates = np.array(hp["hot_pixels"]["e_per_s_at_ref"]["empirical"])
    expected = np.median(base * 2 ** (5.0 / doubling))
    assert abs(np.median(rates) / expected - 1) < 0.05, (np.median(rates), expected, notes)
