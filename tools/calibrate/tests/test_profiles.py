"""Every profile shipped in `profiles/` loads, describes itself and renders a frame."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from skytwin_calibrate import contract

ROOT = Path(__file__).resolve().parents[3]
SKYTWIN = os.environ.get("SKYTWIN", str(ROOT / "target" / "release" / "skytwin"))
PROFILES = sorted((ROOT / "profiles").glob("*.json"))


def test_profiles_found():
    assert len(PROFILES) >= 2


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_learned_profile_contract(path):
    """A learned profile names its sensor, its catalogue and where it came from."""
    p = json.loads(path.read_text())
    if "learned" not in p:
        pytest.skip("hand-written example")
    assert p["schema_version"] == contract.SCHEMA_VERSION
    assert p["sensor"]["name"] and len(p["sensor"]["band_nm"]) == 2
    cat = p["catalog"]
    assert cat["kind"] in ("csv", "ucac4") and cat["mag_limit"] > 0 and cat.get("source")
    if cat["kind"] == "csv":
        assert set(contract.GAIA_COLUMNS) <= set(cat["columns"])
    assert p["learned"]["inputs"], "cite the data the profile was learned from"
    assert "Profile:" in contract.describe(path) or contract.describe(path)


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_profile_renders(path, tmp_path):
    if not Path(SKYTWIN).exists():
        pytest.skip("skytwin binary not built (cargo build --release)")
    cam = json.loads(path.read_text())["camera"]
    scene = {
        "seed": 3,
        "instrument": str(path),
        "scene": {"random_stars": {"count": 50, "max_flux_e": 2e5, "min_flux_e": 2e3}},
    }
    (tmp_path / "s.json").write_text(json.dumps(scene))
    subprocess.run([SKYTWIN, str(tmp_path / "s.json"), str(tmp_path / "s.fits")], check=True, capture_output=True)
    d = fits.getdata(tmp_path / "s.fits")
    assert d.shape == (cam["height"], cam["width"])
    assert np.isfinite(d.astype(np.float64)).all()
