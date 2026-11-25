"""Detector labels: what is visible, where its boxes are, and the exported training set."""

import json
import struct
import zlib

import numpy as np
import pytest

import skytwin
from skytwin import labels as lab
from skytwin.export import main as export

CAMERA = {
    "width": 128,
    "height": 96,
    "exposure_s": 2.0,
    "pixel_scale_arcsec": 3.0,
    "fwhm_px": 2.0,
    "read_noise_e": 5.0,
    "sky": {"e_per_s": 10.0},
}

SCENE = {
    "seed": 3,
    "camera": CAMERA,
    "scene": {
        "stars": [
            {"x": 60.2, "y": 40.7, "flux_e": 2e4},
            {"x": 20.0, "y": 70.0, "flux_e": 30.0},
            {"x": -1.5, "y": 50.0, "flux_e": 2e4},
        ],
        "streaks": [{"start": [10.0, 10.0], "end": [110.0, 80.0], "flux_e": 3e4}],
    },
}


def read_png(path):
    """Pixels of a greyscale PNG written without row filters."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, head = 8, b"", None
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos : pos + 4])
        tag, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + n]
        if tag == b"IHDR":
            head = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    w, h, depth = head[:3]
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, -1)
    assert (rows[:, 0] == 0).all()
    return rows[:, 1:].copy().view(">u2" if depth == 16 else np.uint8).reshape(h, w)


def test_objects_keep_what_a_detector_could_see():
    """The bright star and the streak pass; the faint star and the one
    mostly off the image do not; boxes sit on the pixels they mark."""
    f = skytwin.Simulator(SCENE).render(0)
    objs = lab.objects(f.truth)
    assert [(o.kind, o.form) for o in objs] == [("star", "point")] * 3 + [("satellite", "streak")]
    assert objs[1].snr < 5 < objs[0].snr
    kept = lab.boxes(objs, (0, 0, 128, 96), (128, 96))
    assert [b.obj.source for b in kept] == ["stars[0]", "streaks[0]"]
    star, streak = kept
    x0, y0, x1, y1 = star.xyxy
    assert x0 < 60.7 < x1 and y0 < 41.2 < y1
    assert abs((x0 + x1) / 2 - 60.7) < 1e-9 and abs((y0 + y1) / 2 - 41.2) < 1e-9
    r = streak.obj.radius
    assert np.allclose(streak.xyxy, (10.5 - r, 10.5 - r, 110.5 + r, 80.5 + r))
    peak = np.unravel_index(np.argmax(f.image), f.image.shape)
    assert x0 <= peak[1] + 0.5 <= x1 and y0 <= peak[0] + 0.5 <= y1
    line = lab.yolo_lines([star], lab.class_names(), (128, 96))[0].split()
    assert line[0] == "0" and abs(float(line[1]) - 60.7 / 128) < 1e-6


def _inside(corners, q, tol=1e-6):
    """Whether point ``q`` lies in the convex polygon ``corners`` (or on its edge)."""
    signs = set()
    for k in range(len(corners)):
        a, b = corners[k], corners[(k + 1) % len(corners)]
        cross = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
        if abs(cross) > tol:
            signs.add(cross > 0)
    return len(signs) <= 1


def test_rotated_boxes_hold_streaks_that_cross_the_tile_edge():
    """A streak leaving a tile at a shallow angle keeps all of its in-tile
    light in its rotated box, which stays inside the tile."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        p0 = rng.uniform(-40, 104, 2)
        ang = rng.uniform(0, 2 * np.pi)
        p1 = p0 + rng.uniform(5, 200) * np.array([np.cos(ang), np.sin(ang)])
        r = rng.uniform(0.5, 6)
        path = np.linspace(p0, p1, 200)
        path = path[((path >= 0) & (path <= 64)).all(axis=1)]
        if len(path) < 2:
            continue
        c = np.array(lab._rotated(path, r, 64.0, 64.0))
        assert c.shape == (4, 2) and (c >= 0).all() and (c <= 64).all()
        u = (path[-1] - path[0]) / max(np.hypot(*(path[-1] - path[0])), 1e-9)
        v = np.array([-u[1], u[0]])
        for q in path[::10]:
            for s in (-0.99 * r, 0.0, 0.99 * r):
                x = q + s * v
                if (x >= 0).all() and (x <= 64).all():
                    assert _inside(c, x)


def test_tiles_cut_streaks_and_scale_their_signal():
    """A streak crossing four overlapping tiles is boxed in each it passes
    through, with its signal-to-noise scaled by the square root of the
    share of its length there; the rotated box lies along it, inside the
    tile, and holds every point of the streak's path in the tile."""
    assert lab.tiles(128, 96) == [(0, 0, 128, 96)]
    tiles = lab.tiles(128, 96, 64, 16)
    assert tiles == [
        (0, 0, 64, 64),
        (48, 0, 64, 64),
        (64, 0, 64, 64),
        (0, 32, 64, 64),
        (48, 32, 64, 64),
        (64, 32, 64, 64),
    ]
    f = skytwin.Simulator(SCENE).render(0)
    streak = [o for o in lab.objects(f.truth) if o.kind == "satellite"]
    total = 0.0
    for t in tiles:
        for b in lab.boxes(streak, t, (128, 96), min_snr=0.0):
            assert 0 <= b.xyxy[0] < b.xyxy[2] <= 64 and 0 <= b.xyxy[1] < b.xyxy[3] <= 64
            assert b.snr < streak[0].snr
            total += (b.snr / streak[0].snr) ** 2
            c = np.array(b.corners)
            assert c.shape == (4, 2) and (c >= 0).all() and (c <= 64).all()
            path = streak[0].path - np.array([t[0] - 0.5, t[1] - 0.5])
            path = path[((path >= 0) & (path <= 64)).all(axis=1)]
            assert all(_inside(c, q) for q in path)
    assert total > 1.0
    obb = lab.yolo_obb_lines(lab.boxes(streak, tiles[0], (128, 96), min_snr=0.0), lab.class_names(), (64, 64))
    values = [float(v) for v in obb[0].split()[1:]]
    assert len(values) == 8 and all(0.0 <= v <= 1.0 for v in values)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_png_round_trip(tmp_path, dtype):
    a = (np.arange(35 * 17).reshape(17, 35) * 97 % np.iinfo(dtype).max).astype(dtype)
    lab.write_png(tmp_path / "a.png", a)
    assert np.array_equal(read_png(tmp_path / "a.png"), a)


def test_stretch_depends_on_signal_to_noise_alone():
    """The same sources over twice the sky noise level land on the same grey levels."""
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (64, 64))
    noise[30:33, 30:33] += 50
    a = lab.stretch(1000 + 10 * noise)
    b = lab.stretch(5000 + 20 * noise)
    assert np.abs(a.astype(int) - b).max() <= 1
    want = 255 * (np.arcsinh(50 / 3) + np.arcsinh(2 / 3)) / (np.arcsinh(1000 / 3) + np.arcsinh(2 / 3))
    assert abs(int(a[31, 31]) - want) <= 2
    sky = 255 * np.arcsinh(2 / 3) / (np.arcsinh(1000 / 3) + np.arcsinh(2 / 3))
    assert abs(np.median(a) - sky) <= 1


def test_export_writes_a_yolo_and_coco_set(tmp_path):
    """Five frames in 64 px tiles: every tile has an image and a label
    file, the validation split holds the last two frames whole, COCO
    holds the same objects as YOLO, and the images are the stretched
    frames."""
    scene = dict(SCENE, scene=dict(SCENE["scene"], random_stars={"count": 20, "max_flux_e": 1e5, "min_flux_e": 50}))
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    out = tmp_path / "set"
    args = [str(path), str(out), "--frames", "5", "--format", "yolo", "--format", "coco"]
    assert export([*args, "--tile", "64", "--overlap", "16", "--val", "0.4"]) == 0
    images = {s: sorted(p.stem for p in (out / "images" / s).glob("*.png")) for s in ("train", "val")}
    assert len(images["train"]) == 18 and len(images["val"]) == 12
    assert {s.split("_")[0] for s in images["val"]} == {"000003", "000004"}
    lines = 0
    for s in ("train", "val"):
        for stem in images[s]:
            for row in (out / "labels" / s / f"{stem}.txt").read_text().splitlines():
                k, *xywh = row.split()
                assert k in ("0", "1") and all(0.0 <= float(v) <= 1.0 for v in xywh)
                lines += 1
        coco = json.loads((out / "annotations" / f"instances_{s}.json").read_text())
        assert len(coco["images"]) == len(images[s])
        assert [c["name"] for c in coco["categories"]] == ["star", "satellite"]
        lines -= len(coco["annotations"])
    assert lines == 0
    yaml = (out / "data.yaml").read_text()
    assert "train: images/train" in yaml and "val: images/val" in yaml and "  1: satellite" in yaml
    manifest = json.loads((out / "skytwin.json").read_text())
    assert manifest["counts"]["val"]["frames"] == 2 and manifest["frames"]["first_val"] == 3
    frame = skytwin.Simulator(scene).render(4)
    assert np.array_equal(read_png(out / "images" / "val" / "000004_05.png"), lab.stretch(frame.image)[32:96, 64:128])


def test_export_rejects_two_yolo_formats(tmp_path, capsys):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(SCENE))
    assert export([str(path), str(tmp_path / "o"), "--format", "yolo", "--format", "yolo-obb"]) == 2
    assert "choose one" in capsys.readouterr().err


def test_export_refuses_an_earlier_export_unless_told_to_replace_it(tmp_path, capsys):
    """A second export into the same folder is refused; with ``--overwrite`` it
    replaces the first, so no frame lands in both splits or lingers from it."""
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(SCENE))
    out = tmp_path / "set"
    assert export([str(path), str(out), "--frames", "10", "--val", "0.3"]) == 0
    assert export([str(path), str(out), "--frames", "5", "--val", "0.4"]) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert export([str(path), str(out), "--frames", "5", "--val", "0.4", "--overwrite"]) == 0
    train = {p.name for p in (out / "images" / "train").glob("*.png")}
    val = {p.name for p in (out / "images" / "val").glob("*.png")}
    assert not train & val
    assert train | val == {f"{k:06d}.png" for k in range(5)}


@pytest.mark.parametrize("bad", [["--tile", "0"], ["--tile", "64", "--overlap", "64"], ["--frames", "-1"]])
def test_export_rejects_impossible_tiles_and_counts(tmp_path, bad):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(SCENE))
    assert export([str(path), str(tmp_path / "o"), *bad]) == 2


def test_detection_targets_batch_in_pytorch():
    """Frames with different numbers of objects batch with ``collate``."""
    torch = pytest.importorskip("torch")
    from skytwin.torch import FrameDataset, collate, detection_target

    scene = dict(SCENE, batch={"frames": 4})
    scene["scene"] = dict(
        SCENE["scene"],
        random_stars={"count": 10, "max_flux_e": 1e5, "min_flux_e": 500},
        random_streaks={"count": {"uniform": [0, 3]}, "length_px": 40, "mag": 9},
    )
    loader = torch.utils.data.DataLoader(
        FrameDataset(scene, transform=detection_target), batch_size=2, collate_fn=collate
    )
    batches = list(loader)
    assert len(batches) == 2
    images, targets = batches[0]
    assert images.shape == (2, 1, 96, 128) and images.dtype == torch.float32 and 0 <= images.min() <= images.max() <= 1
    for t in targets:
        assert t["boxes"].shape == (len(t["labels"]), 4) and t["boxes"].dtype == torch.float32
        assert (t["boxes"][:, 2:] > t["boxes"][:, :2]).all()
    _, t = detection_target(skytwin.Simulator(SCENE).render(0), box_format="cxcywh")
    assert torch.allclose(t["boxes"][0, :2], torch.tensor([60.7 / 128, 41.2 / 96]))


TRACKING = {
    "seed": 8,
    "batch": {"frames": 6},
    "camera": dict(CAMERA, width=256, height=256, exposure_s=3.0),
    "overrides": {"pointing": {"wobble_rms_arcsec": 1.0}},
    "scene": {
        "time": {"start_utc": "2026-11-17T03:00:00.000"},
        "observer": {
            "tle": [
                "1 99001U          26321.00000000  .00000000  00000-0  10000-4 0    01",
                "2 99001  98.5500 325.0000 0012000  95.0000  10.0000 14.30000000    04",
            ]
        },
        "pointing": {"track": "GEO-1", "roll_deg": 30.0},
        "random_stars": {"count": 30, "max_flux_e": 2e5, "min_flux_e": 2e4},
        "targets": [
            {
                "name": "GEO-1",
                "orbit": {
                    "tle": [
                        "1 99101U          26321.00000000  .00000000  00000-0  00000+0 0    06",
                        "2 99101   0.0500  80.0000 0002000  90.0000 100.0000  1.00273791    07",
                    ]
                },
                "brightness": {"mag": 9.0},
            }
        ],
    },
}


def test_tracking_trails_stars_and_keeps_the_target_a_point():
    """Tracking a GEO target at its rate trails every star about 45 px and
    keeps the target a point, though the pointing wobbles."""
    for f in skytwin.Simulator(TRACKING).stream():
        objs = lab.objects(f.truth)
        stars = [o for o in objs if o.kind == "star"]
        (sat,) = [o for o in objs if o.kind == "satellite"]
        assert len(stars) == 30 and all(o.form == "streak" for o in stars)
        assert all(40 < np.hypot(*(o.path[-1] - o.path[0])) < 50 for o in stars)
        assert sat.form == "point" and sat.snr > 5
    names = lab.class_names(by_form=True)
    lines = lab.yolo_lines(lab.boxes(objs, (0, 0, 256, 256), (256, 256)), names, (256, 256))
    assert {line.split()[0] for line in lines} == {"1", "2"}


def test_jitter_does_not_make_a_streak():
    """A path that zigzags 10 px in all within 1 px is a point; one that
    runs 5 px is a streak at the default threshold (twice a 2 px FWHM)."""
    zigzag = [[50.0 + 0.5 * (k % 2), 50.0 + 0.1 * k] for k in range(10)]
    label = {"extent_px": 1.0, "snr": 20.0, "peak_snr": 5.0, "saturated": False}
    truth = {
        "camera": {"fwhm_px": 2.0},
        "stars": [
            {"label": dict(label, path=zigzag)},
            {"label": dict(label, path=[[10.0, 10.0], [15.0, 10.0]])},
        ],
    }
    assert [o.form for o in lab.objects(truth)] == ["point", "streak"]
    assert [o.form for o in lab.objects(truth, streak_px=6.0)] == ["point", "point"]


def test_one_frame_export_keeps_its_frame_for_training():
    """The validation share never takes every frame, so `images/train` is never empty."""
    from skytwin.export import val_count

    assert val_count(1, 0.1) == 0
    assert val_count(2, 0.1) == 1
    assert val_count(10, 0.1) == 1
    assert val_count(10, 0.25) == 3
    assert val_count(10, 0.0) == 0
    assert val_count(0, 0.5) == 0
