"""Learn round trip: frames from a known profile, with header keywords
Skytwin does not use itself -> `learn` -> the same settings back."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from skytwin_calibrate import learn

ROOT = Path(__file__).resolve().parents[3]
SKYTWIN = os.environ.get("SKYTWIN", str(ROOT / "target" / "release" / "skytwin"))

TRUE = {
    "detector_seed": 3,
    "camera": {
        "width": 512,
        "height": 512,
        "fwhm_px": 2.2,
        "exposure_s": 2.0,
        "sky": {"e_per_s": 100.0},
        "dark_e_per_s": 0.0,
        "read_noise_e": 8.0,
        "gain_e_per_adu": 2.0,
        "bias_adu": 0.0,
        "full_well_e": 1e7,
    },
    "processing": {"pedestal_adu": 20.0, "clip_max_adu": 60000.0},
    "temperature_c": {"uniform": [-35.0, -20.0]},
    "dark": {"e_per_s_at_ref": 0.0, "ref_temp_c": -27.5, "doubling_c": 6.0},
    "hot_pixels": {"density_per_mpix": 2000.0, "e_per_s_at_ref": {"uniform": [40.0, 80.0]}},
    "flat": {"vignetting": [-0.3, 0.0]},
    "smear": {"above_s_per_row": 2e-4, "below_s_per_row": 5e-5},
    "noise": {"smooth_field": {"rms_adu": 4.0, "scale_px": 40.0}},
    "header": {
        "wcs": False,
        "skytwin_keywords": False,
        "cards": [
            {"key": "EXPOSURE", "value": "{exposure_s}", "comment": "[s] Actual length of exposure"},
            {"key": "CCD-TEMP", "value": "{ccd_temp_c}", "comment": "[C] CCD temperature"},
            {"key": "GAIN", "value": "2.00", "comment": "[electrons/ADU] amplifier gain"},
            {"key": "TELESCOP", "value": "TESTSAT", "comment": "spacecraft"},
        ],
    },
}


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    """48 frames from `TRUE`; the background level varies between frames (as stray light makes it), so the
    response can be told from additive offsets."""
    if not Path(SKYTWIN).exists():
        pytest.skip("skytwin binary not built (cargo build --release)")
    d = tmp_path_factory.mktemp("learn")
    (d / "true.json").write_text(json.dumps(TRUE))
    rng = np.random.default_rng(5)
    out = []
    for s in range(48):
        sky = float(rng.uniform(50.0, 2000.0))
        scene = {
            "seed": s,
            "instrument": str(d / "true.json"),
            "overrides": {"camera": {"sky": {"e_per_s": sky}}},
            "scene": {"random_stars": {"count": 30, "max_flux_e": 3e6, "min_flux_e": 3e4}},
        }
        (d / f"s{s}.json").write_text(json.dumps(scene))
        subprocess.run([SKYTWIN, str(d / f"s{s}.json"), str(d / f"f{s}.fits")], check=True, capture_output=True)
        out.append(str(d / f"f{s}.fits"))
    return d, out


def test_learn_recovers_a_known_profile(frames, tmp_path):
    """`learn` finds the keywords and recovers the settings; smear is the raw measure (no --skytwin to
    correct it), so it only has to fall within a factor of 2.5."""
    d, _ = frames
    prof, _ = learn.learn([str(d)], str(tmp_path / "p.json"), verbose=False)
    km = json.loads((tmp_path / "p.keywords.json").read_text())
    assert km["exposure_s"] == "EXPOSURE" and km["temperature_c"] == "CCD-TEMP" and km["gain_e_per_adu"] == "GAIN"
    cam = prof["camera"]
    assert cam["gain_e_per_adu"] == 2.0 and cam["width"] == 512 and abs(cam["exposure_s"] - 2.0) < 1e-9
    assert prof["processing"]["clip_max_adu"] == 60000.0
    hp = prof["hot_pixels"]
    assert 1400 < hp["density_per_mpix"] < 2400, hp["density_per_mpix"]
    med = np.median(hp["e_per_s_at_ref"]["empirical"])
    assert 45 < med < 75, med
    assert 4.0 < prof["dark"]["doubling_c"] < 8.5, prof["dark"]
    a2 = prof["flat"]["vignetting"][0]
    assert -0.4 < a2 < -0.2, prof["flat"]
    assert abs(cam["fwhm_px"] - 2.2) < 0.25, cam["fwhm_px"]
    sm = prof["smear"]
    assert 0.8e-4 < sm["above_s_per_row"] < 5e-4 and 2e-5 < sm["below_s_per_row"] < 1.25e-4, sm
    sf = prof["noise"]["smooth_field"]
    ra = sf["rms_adu"]
    rms = np.median(ra["empirical"] if "empirical" in ra else ra["empirical_linked"]["values"])
    assert 2.5 < rms < 5.5 and 20 < sf["scale_px"] < 80, sf
    hdr = {c["key"]: c["value"] for c in prof["header"]["cards"]}
    assert hdr["EXPOSURE"] == "{exposure_s}" and hdr["CCD-TEMP"] == "{ccd_temp_c}" and hdr["TELESCOP"] == "TESTSAT"
    assert prof["header"]["wcs"] is False
    text = (tmp_path / "p.learn.md").read_text()
    assert "| hot_pixels | measured |" in text and "| smear | measured |" in text and "| cti | not learned |" in text


def test_earth_glare_table():
    """Glare of 400 ADU × exp(-(angle - 2) / 3) within 15° of a sunlit limb, a tenth of that by a dark one;
    at 2 e/ADU and 2 s the table's e/s equal the ADU."""
    from skytwin_calibrate import analyse

    rng = np.random.default_rng(2)
    rows, geom = [], {}
    for i in range(800):
        a = float(rng.uniform(2, 80))
        lit = bool(rng.uniform() < 0.5)
        glare = 400.0 * np.exp(-(a - 2) / 3) if a < 15 else 0.0
        glare *= 1.0 if lit else 0.1
        f = f"f{i}"
        rows.append(
            {
                "file": f,
                "median": 20 + glare + rng.normal(0, 2),
                "ramp_x": 0.0,
                "ramp_y": 0.0,
                "shape": [256, 256],
                "meta": {"exposure_s": 2.0},
            }
        )
        geom[f] = (a, 1.0 if lit else 0.0, [1.0, 0.0])
    block, free, _ = analyse.earth_glare(rows, 2.0, geom, roll_known=False)
    assert block["body"] == "earth" and 8 < free < 17, (free, block)
    a0, v0 = block["angle_deg"][0], block["e_per_s"][0]
    want = 400.0 * np.exp(-(a0 - 2) / 3)
    assert 0.6 * want < v0 < 1.4 * want, (a0, v0, want)
    assert 0.05 < block["dark_limb_factor"] < 0.2, block
    assert block["e_per_s"][-1] == 0.0


def test_learn_takes_camera_numbers_from_calibration_frames(frames, tmp_path):
    """With `--calibration`, gain and read noise come from flat and bias pairs (replacing the header's
    gain), the dark current, radiation rate and read-out time from the darks, and dust from the master
    flat; hot pixels stay the science frames' own. The headers give no pixel size, so 13 µm is assumed
    for the detector area."""
    from test_calibration import write_calibration

    d, _ = frames
    cal = tmp_path / "calibration"
    area = (512 * 13e-4) ** 2
    rings = [(150, 150, 30, 0.006), (350, 160, 30, 0.004), (250, 300, 40, 0.008), (140, 360, 30, 0.005)]
    write_calibration(
        cal,
        n=512,
        gain=2.0,
        read_adu=4.0,
        bias=20.0,
        dark_e_s=0.5,
        hits_per_s=3.0,
        readout_s=4.0,
        temp=-27.5,
        rings=rings,
        nights=1,
    )
    prof, _ = learn.learn([str(d)], str(tmp_path / "p.json"), verbose=False, calibration=[str(cal)])
    cam = prof["camera"]
    assert abs(cam["gain_e_per_adu"] / 2.0 - 1) < 0.05 and abs(cam["read_noise_e"] / 8.0 - 1) < 0.05, cam
    assert abs(prof["dark"]["e_per_s_at_ref"] / 0.5 - 1) < 0.2 and prof["dark"]["doubling_c"], prof["dark"]
    rad = prof["radiation"]
    assert abs(rad["rate_per_cm2_s"] * area / 3.0 - 1) < 0.2 and abs(rad["readout_s"] / 4.0 - 1) < 0.4, rad
    assert rad["first_row"] == "bottom" and "map_scale" not in rad, rad
    assert len(prof["flat"]["dust"]["radius_px"]["empirical"]) == 4 and "vignetting" in prof["flat"], prof["flat"]
    assert 1400 < prof["hot_pixels"]["density_per_mpix"] < 2400
    text = (tmp_path / "p.learn.md").read_text()
    assert "replaces the header's 2.0 e/ADU" in text and "| flat.dust | measured |" in text
    assert "science frames' lowest rate, or 6σ of the dark map" in text


def test_learn_on_frames_whose_stars_trail(tmp_path):
    """Stars trail 30 px (a telescope following a satellite): `learn` measures the trails, fits the PSF
    across them, counts radiation outside them and gives each frame's twin its trail rate."""
    if not Path(SKYTWIN).exists():
        pytest.skip("skytwin binary not built (cargo build --release)")
    prof = json.loads(json.dumps(TRUE))
    prof["camera"]["pixel_scale_arcsec"] = 2.0
    prof["pointing"] = {"drift_arcsec_per_s": 30.0}
    prof["radiation"] = {"rate_per_cm2_s": 20.0}
    prof["header"]["cards"].append({"key": "PIXSCALE", "value": "2.0", "comment": "[arcsec/pix] pixel scale"})
    (tmp_path / "true.json").write_text(json.dumps(prof))
    frames = tmp_path / "frames"
    frames.mkdir()
    for s in range(24):
        scene = {
            "seed": 100 + s,
            "instrument": str(tmp_path / "true.json"),
            "scene": {"random_stars": {"count": 60, "max_flux_e": 3e6, "min_flux_e": 3e4}},
        }
        (tmp_path / f"s{s}.json").write_text(json.dumps(scene))
        subprocess.run(
            [SKYTWIN, str(tmp_path / f"s{s}.json"), str(frames / f"f{s}.fits")], check=True, capture_output=True
        )
    out, _ = learn.learn([str(frames)], str(tmp_path / "p.json"), verbose=False)
    text = (tmp_path / "p.learn.md").read_text()
    assert "| trails | measured |" in text and "| psf | measured |" in text, text
    from skytwin_calibrate import trails

    comps = [(1 - sum(w["fraction"] for w in out["psf"]["wings"]), out["camera"]["fwhm_px"])] + [
        (w["fraction"], w["fwhm_px"]) for w in out["psf"]["wings"]
    ]
    assert abs(trails.marginal_fwhm(comps) / 2.2 - 1) < 0.15, out["psf"]
    assert abs(out["radiation"]["rate_per_cm2_s"] / 20.0 - 1) < 0.35, out["radiation"]
    real = json.loads((tmp_path / "p-work" / "real.json").read_text())
    rates = [r["meta"]["trail_rate_arcsec_per_s"] for r in real if "error" not in r]
    assert abs(np.median(rates) / 30.0 - 1) < 0.05, rates


@pytest.mark.parametrize("parity", ["negative", "positive"])
def test_pointing_from_wcs_reads_skytwins_own_convention(tmp_path, parity):
    """Roll is the position angle of the image +y axis east of north; a frame Skytwin writes with its WCS
    reads back to the pointing it was drawn at, for either parity."""
    if not Path(SKYTWIN).exists():
        pytest.skip("skytwin binary not built (cargo build --release)")
    from astropy.io import fits

    from skytwin_calibrate import astrometry

    prof = json.loads(json.dumps(TRUE))
    prof["camera"].update(pixel_scale_arcsec=2.0, parity=parity)
    prof["header"]["wcs"] = True
    (tmp_path / "p.json").write_text(json.dumps(prof))
    scene = {
        "seed": 1,
        "instrument": str(tmp_path / "p.json"),
        "scene": {
            "pointing": {"ra_deg": 120.0, "dec_deg": 30.0, "roll_deg": 40.0},
            "random_stars": {"count": 5, "max_flux_e": 1e5, "min_flux_e": 1e4},
        },
    }
    (tmp_path / "s.json").write_text(json.dumps(scene))
    subprocess.run([SKYTWIN, str(tmp_path / "s.json"), str(tmp_path / "f.fits")], check=True, capture_output=True)
    p = astrometry.pointing_from_wcs(fits.getheader(tmp_path / "f.fits"))
    assert abs(p["ra_deg"] - 120.0) < 1e-3 and abs(p["dec_deg"] - 30.0) < 1e-3 and abs(p["roll_deg"] - 40.0) < 0.01, p
