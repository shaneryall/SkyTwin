"""Round trip: known settings -> Skytwin frames -> measure -> profile -> same settings."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from skytwin_calibrate import measure, profile, report

ROOT = Path(__file__).resolve().parents[3]
SKYTWIN = os.environ.get("SKYTWIN", str(ROOT / "target" / "release" / "skytwin"))

BASE = {
    "camera": {
        "width": 256,
        "height": 256,
        "sky": {"e_per_s": 0.0},
        "dark_e_per_s": 0.0,
        "read_noise_e": 8.0,
        "gain_e_per_adu": 1.0,
        "bias_adu": 1000.0,
        "exposure_s": 3.0,
    },
    "processing": {"pedestal_adu": 30.0},
    "noise": {"extra_white_adu": 20.0, "patterns": [{"period_px": 5.0, "angle_deg": 30.0, "amplitude_adu": 10.0}]},
}


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    """12 frames rendered from `BASE`."""
    if not Path(SKYTWIN).exists():
        pytest.skip("skytwin binary not built (cargo build --release)")
    d = tmp_path_factory.mktemp("frames")
    (d / "profile.json").write_text(json.dumps(BASE))
    out = []
    for s in range(12):
        scene = {"seed": s, "instrument": str(d / "profile.json"), "camera": {"exposure_s": 3.0}}
        (d / f"s{s}.json").write_text(json.dumps(scene))
        subprocess.run([SKYTWIN, str(d / f"s{s}.json"), str(d / f"s{s}.fits")], check=True, capture_output=True)
        out.append(str(d / f"s{s}.fits"))
    return out


def test_measure_recovers_known_settings(frames, tmp_path):
    """Pedestal, total noise (read, extra white and pattern/√2) and the pattern come back from `measure`."""
    rows = measure.run(frames, {"exposure_s": "EXPTIME"}, tmp_path / "s.json", progress=False)
    med = np.median([r["median"] for r in rows])
    noise = np.median([r["noise"] for r in rows])
    per = np.median([r["pattern"]["period"] for r in rows])
    ang = np.median([r["pattern"]["angle"] for r in rows])
    want_noise = np.sqrt(8.0**2 + 20.0**2 + 10.0**2 / 2)
    assert abs(med - 30.0) < 1.5, med
    assert abs(noise - want_noise) < 0.05 * want_noise, (noise, want_noise)
    assert abs(per - 5.0) < 0.2 and abs(ang - 30.0) < 2.0, (per, ang)
    assert all(r["meta"]["exposure_s"] == 3.0 for r in rows)


def test_profile_and_report(frames, tmp_path):
    """`profile` recovers pedestal and extra noise (a little high, as the FFT under-reads the pattern), and
    a report of identical statistics shows KS distance 0."""
    rows = measure.run(frames, {"exposure_s": "EXPTIME"}, tmp_path / "s.json", progress=False)
    prof = profile.build(rows, {"camera": BASE["camera"]})
    ped = prof["processing"]["pedestal_adu"]["empirical_linked"]["values"]
    extra = prof["noise"]["extra_white_adu"]["empirical_linked"]["values"]
    assert abs(np.median(ped) - 30.0) < 1.5
    assert abs(np.median(extra) - 20.0) < 3.0, np.median(extra)
    text = report.run(tmp_path / "s.json", tmp_path / "s.json", tmp_path / "r.md")
    assert "| Background noise (MAD) |" in text and "| 0.00 |" in text


def test_score_matches_a_perfect_detector():
    """One star matched 0.1 px off, one detection on a hot pixel, one false."""
    from skytwin_calibrate import score

    truth = {
        "camera": {"width": 100, "height": 100},
        "stars": [
            {"x": 10.0, "y": 20.0, "mag": 12.0, "crlb_px": [0.1, 0.1]},
            {"x": 50.0, "y": 60.0, "mag": 15.5, "crlb_px": [0.3, 0.3]},
        ],
        "streaks": [],
        "targets": [],
        "instrument": {"hot_pixels": [{"x": 80, "y": 80, "e_per_s": 5.0}]},
    }
    det = np.array([[10.1, 20.0], [80.0, 80.0], [5.0, 90.0]])
    res = score.score(truth, det, radius=2.0)
    assert res["matched"] == 1 and res["false_detections"] == 1 and res["detections_on_defects_or_targets"] == 1
    assert abs(res["centroid_error_px"]["median"] - 0.1) < 1e-9
    assert abs(res["error_over_crlb"]["median"] - 0.1 / np.hypot(0.1, 0.1)) < 1e-9


def test_completeness_counts_a_star_at_an_integer_magnitude():
    """A lone mag-12.0 star falls in [12, 13), and the bins hold every star."""
    from skytwin_calibrate import score

    truth = {
        "camera": {"width": 100, "height": 100},
        "stars": [{"x": 10.0, "y": 20.0, "mag": 12.0}],
        "streaks": [],
        "targets": [],
    }
    res = score.score(truth, np.array([[10.0, 20.0]]), radius=2.0)
    assert res["completeness_by_mag"] == [
        {"mag_from": 12.0, "mag_to": 13.0, "stars": 1, "found": 1, "completeness": 1.0}
    ]
    two = dict(truth, stars=truth["stars"] + [{"x": 50.0, "y": 50.0, "mag": 14.0}])
    res = score.score(two, np.array([[10.0, 20.0]]), radius=2.0, mag_bins=[12.0, 13.0, 14.0])
    assert sum(b["stars"] for b in res["completeness_by_mag"]) == 2


def test_a_target_giving_no_light_explains_no_detection():
    """A detection on a target that is off the image or dark counts as false; on a target lit only in
    the second half of the exposure, a detection on the first half of its track counts as false too."""
    from skytwin_calibrate import score

    def target(on_image, flux, rates):
        return {
            "on_image": on_image,
            "flux_e": flux,
            "start": {"xy": [40.0, 50.0]},
            "mid": {"xy": [50.0, 50.0]},
            "end": {"xy": [60.0, 50.0]},
            "light_curve": [[0.0, rates[0]], [5.0, rates[1]], [10.0, rates[2]]],
        }

    def false_count(t, det):
        truth = {"camera": {"width": 100, "height": 100}, "stars": [], "streaks": [], "targets": [t]}
        return score.score(truth, np.array([det]), radius=2.0)["false_detections"]

    assert false_count(target(False, 0.0, [0, 0, 0]), [50.0, 50.0]) == 1
    assert false_count(target(True, 0.0, [0, 0, 0]), [50.0, 50.0]) == 1
    assert false_count(target(True, 500.0, [1, 1, 1]), [50.0, 50.0]) == 0
    half = target(True, 500.0, [0, 0, 1])
    assert false_count(half, [42.0, 50.0]) == 1
    assert false_count(half, [58.0, 50.0]) == 0
