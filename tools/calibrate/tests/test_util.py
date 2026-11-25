import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from skytwin_calibrate.util import most_common, utc_timestamp

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset")
def test_times_without_an_offset_are_utc_in_any_time_zone(monkeypatch):
    """Frame times read the same in every zone, across a daylight-saving change too."""
    for tz in ("UTC", "America/New_York", "Australia/Lord_Howe"):
        monkeypatch.setenv("TZ", tz)
        time.tzset()
        assert utc_timestamp("2024-11-03T05:30:00") == 1730611800.0
        assert utc_timestamp("2024-11-03T06:30:00") - utc_timestamp("2024-11-03T05:30:00") == 3600.0
    monkeypatch.delenv("TZ")
    time.tzset()
    assert utc_timestamp("2024-11-03T05:30:00Z") == utc_timestamp("2024-11-03T05:30:00+00:00") == 1730611800.0
    assert utc_timestamp("2024-11-03T07:30:00+02:00") == 1730611800.0
    assert utc_timestamp("not a time") is None
    assert utc_timestamp(None) is None


def test_most_common_breaks_ties_by_first_seen():
    """A tie gives the value seen first, whatever the string hash seed."""
    assert most_common(["flipped", "normal", "normal", "flipped"]) == "flipped"
    assert most_common(["normal", "flipped", "flipped"]) == "flipped"
    code = "from skytwin_calibrate.util import most_common; print(most_common(['b', 'a', 'a', 'b', 'c']))"
    out = {
        subprocess.run(
            [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": str(s)}, capture_output=True, text=True
        ).stdout.strip()
        for s in range(8)
    }
    assert out == {"b"}


def test_module_entry_point_returns_the_exit_status(tmp_path):
    """`python -m skytwin_calibrate.cli check` exits 1 when the catalogue does not match."""
    cat = tmp_path / "cat.csv"
    cat.write_text("a,b\n1,2\n")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "skytwin_calibrate.cli",
            "check",
            str(ROOT / "profiles/wise-w1.json"),
            "--catalog",
            str(cat),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "problem:" in r.stdout


def test_score_reads_sextractor_photutils_and_upper_case_columns(tmp_path):
    """Source lists from SExtractor (1-based X_IMAGE), photutils (0-based xcentroid) and a CSV with
    upper-case X, Y give the same 0-based positions; a FITS file without a table is a clear error."""
    from astropy.io import fits
    from astropy.table import Table

    from skytwin_calibrate import score

    xy = np.array([[10.0, 20.0], [30.5, 40.25]])
    Table({"X_IMAGE": xy[:, 0] + 1, "Y_IMAGE": xy[:, 1] + 1}).write(tmp_path / "sex.fits")
    Table({"xcentroid": xy[:, 0], "ycentroid": xy[:, 1]}).write(tmp_path / "phot.fits")
    (tmp_path / "upper.csv").write_text("X,Y\n10,20\n30.5,40.25\n")
    for name in ("sex.fits", "phot.fits", "upper.csv"):
        got, _ = score.load_sources(tmp_path / name)
        assert np.allclose(got, xy), name
    fits.PrimaryHDU(np.zeros((4, 4))).writeto(tmp_path / "image.fits")
    with pytest.raises(ValueError, match="no table"):
        score.load_sources(tmp_path / "image.fits")
    (tmp_path / "other.csv").write_text("ra,dec\n1,2\n")
    with pytest.raises(ValueError, match="no position columns"):
        score.load_sources(tmp_path / "other.csv")


def test_learn_refuses_a_missing_catalogue(tmp_path):
    """A `--ucac4` that is not a UCAC4 directory stops `learn` before any work."""
    from skytwin_calibrate import learn

    with pytest.raises(ValueError, match="not a UCAC4 directory"):
        learn.learn([str(tmp_path)], str(tmp_path / "p.json"), ucac4=str(tmp_path / "nowhere"))
    with pytest.raises(ValueError, match="no such file"):
        learn.learn([str(tmp_path)], str(tmp_path / "p.json"), catalog=str(tmp_path / "gaia.csv"))
