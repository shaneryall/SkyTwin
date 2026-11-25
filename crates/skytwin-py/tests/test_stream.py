"""Python bindings: repeatable, ordered frames that match the command line, and the PyTorch dataset."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import skytwin

ROOT = Path(__file__).resolve().parents[3]

SCENE = {
    "seed": 11,
    "batch": {"frames": 24},
    "camera": {
        "width": 160,
        "height": 120,
        "exposure_s": 2.0,
        "pixel_scale_arcsec": 3.0,
        "fwhm_px": 2.0,
        "read_noise_e": 5.0,
        "sky": {"e_per_s": 10.0},
    },
    "overrides": {"pointing": {"wobble_rms_arcsec": 1.0}},
    "scene": {
        "random_stars": {"count": 40, "max_flux_e": 2e5, "min_flux_e": 500.0},
        "random_streaks": {
            "count": {"uniform": [0, 3]},
            "length_px": {"uniform": [10, 80]},
            "mag": {"uniform": [9, 12]},
        },
    },
}


def same(a, b):
    """True when two frames have the same index, pixels and truth."""
    return a.index == b.index and np.array_equal(a.image, b.image) and a.truth == b.truth


def test_render_is_repeatable_and_labelled():
    """The same index gives the same frame; seeds count up from the file's."""
    sim = skytwin.Simulator(SCENE)
    assert len(sim) == 24
    a, b = sim.render(3), sim.render(3)
    assert same(a, b) and a.seed == 14
    assert a.image.dtype == np.uint16 and a.image.shape == (120, 160)
    assert len(a.truth["stars"]) == 40
    assert not np.array_equal(sim.render(4).image, a.image)


def test_stream_is_ordered_and_independent_of_workers():
    """Frames come in order and match whatever the number of workers."""
    sim = skytwin.Simulator(SCENE)
    one = list(sim.stream(workers=1))
    many = list(sim.stream(workers=8, prefetch=3))
    assert [f.index for f in many] == list(range(24))
    assert all(same(a, b) for a, b in zip(one, many, strict=True))
    assert same(many[5], sim.render(5))
    part = list(sim.stream(10, 4, workers=3))
    assert [f.index for f in part] == [10, 11, 12, 13]


def test_expected_is_the_noise_free_image():
    """Away from sources the expected image is the sky, 10 e/s for 2 s."""
    f = skytwin.Simulator(SCENE).render(2, expected=True)
    assert f.expected.dtype == np.float32 and f.expected.shape == f.image.shape
    assert abs(np.median(f.expected) - 20.0) < 1.0
    assert f.expected.sum() > 20.0 * f.expected.size


def test_stopping_early_and_errors():
    """Leaving a stream early is fine; bad scenes raise ValueError."""
    sim = skytwin.Simulator(SCENE)
    for f in sim.stream(workers=4):
        if f.index == 2:
            break
    with pytest.raises(ValueError):
        skytwin.Simulator({"scene": {}, "nonsense": 1})
    with pytest.raises(ValueError):
        skytwin.Simulator("no/such/file.json")


def _card(head, k, key):
    """Integer value of card `key` in the first `k` bytes of a FITS header."""
    return int(next(head[i + 10 : i + 30] for i in range(0, k, 80) if head[i : i + 8].strip() == key))


def read_fits(path):
    """Pixels of a 16-bit unsigned FITS image (BZERO 32768), read without astropy."""
    raw = Path(path).read_bytes()
    k = 0
    while True:
        block = raw[k : k + 2880]
        k += 2880
        if any(block[i : i + 8].startswith(b"END     ") for i in range(0, 2880, 80)):
            break
    head = raw[:k].decode()
    w, h = _card(head, k, "NAXIS1"), _card(head, k, "NAXIS2")
    data = np.frombuffer(raw[k : k + 2 * w * h], dtype=">i2").astype(np.int32) + 32768
    return data.reshape(h, w)


@pytest.mark.skipif(not (ROOT / "target/release/skytwin").exists(), reason="build the CLI first")
def test_python_frames_match_the_command_line(tmp_path):
    """Python frames equal the command line's FITS pixels and compact truth."""
    scene = tmp_path / "s.json"
    scene.write_text(json.dumps(SCENE))
    out = str(tmp_path / "f.fits")
    subprocess.run(
        [str(ROOT / "target/release/skytwin"), str(scene), out, "--frames", "4", "--compact-truth"],
        check=True,
        capture_output=True,
    )
    sim = skytwin.Simulator(str(scene))
    for k in range(4):
        f = sim.render(k)
        assert np.array_equal(read_fits(tmp_path / f"f_{k:05d}.fits"), f.image.astype(np.int32))
        assert json.loads((tmp_path / f"f_{k:05d}.truth.json").read_text()) == f.truth


def labels(f):
    """Image tensor, index and streak count of a frame (module level, so it pickles)."""
    import torch

    return torch.from_numpy(f.image.astype("float32")), f.index, len(f.truth["streaks"])


def as_list(batch):
    """Collate that keeps the batch as a list (module level, so it pickles)."""
    return batch


def test_torch_dataset_splits_frames_across_workers():
    """DataLoader workers (new processes, so transform and collate must pickle) share the frames."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from skytwin.torch import FrameDataset

    ds = FrameDataset(SCENE, transform=labels)
    assert len(ds) == 24
    seen = []
    loader = DataLoader(ds, batch_size=4, num_workers=3, collate_fn=as_list)
    for batch in loader:
        seen += [i for _, i, _ in batch]
    assert sorted(seen) == list(range(24))
    sim = skytwin.Simulator(SCENE)
    img, idx, _ = next(iter(FrameDataset(SCENE, start=7, count=1, transform=labels)))
    assert idx == 7 and torch.equal(img, torch.from_numpy(sim.render(7).image.astype("float32")))


def test_default_dataset_batches_with_the_default_collation():
    """A plain DataLoader, no collate_fn, batches the default (image, index) pairs."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from skytwin.torch import FrameDataset

    batches = list(DataLoader(FrameDataset(SCENE, count=4), batch_size=2))
    assert len(batches) == 2
    images, index = batches[1]
    assert images.dtype == torch.float32 and images.shape[0] == 2
    assert index.tolist() == [2, 3]


def test_slow_reader_gets_every_frame_in_order():
    """With workers held to a window ahead of a slow reader, every frame still arrives, in order."""
    import time

    sim = skytwin.Simulator(SCENE)
    seen = []
    for f in sim.stream(0, 24, workers=4, prefetch=1):
        seen.append(f.index)
        if f.index < 3:
            time.sleep(0.2)
    assert seen == list(range(24))
