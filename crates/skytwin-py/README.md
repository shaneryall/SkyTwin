# skytwin (Python)

Synthetic star-field images with satellite streaks, instrument defects and an exact answer key
for every frame, rendered in memory and streamed to Python. See the
[Skytwin README](https://github.com/shaneryall/SkyTwin#training-sets).

```bash
pip install skytwin      # or `pip install .` from this folder to build from source
```

```python
import skytwin

sim = skytwin.Simulator("scene.json")
for f in sim.stream(count=1000, expected=True):
    f.image, f.expected, f.truth
```

Replace `scene.json` with the path to any Skytwin
[scene file](https://github.com/shaneryall/SkyTwin/blob/v0.1.0/docs/scene-format.md).

Tests: `pip install -e '.[test]' && pytest tests` (the command-line match
test needs `cargo build --release` at the repository root; the PyTorch test
needs `torch`).

To write a training set for Ultralytics YOLO or for COCO tools such as DETR, run:

```bash
skytwin-export scene.json out/ --frames 5000 --format yolo --format coco --tile 640
```

See [docs/labels.md](https://github.com/shaneryall/SkyTwin/blob/v0.1.0/docs/labels.md) for the labels, the options and
the PyTorch targets.
