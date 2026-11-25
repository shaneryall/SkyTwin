# Export a training set

`skytwin-export` renders frames of a scene and writes them with YOLO or COCO labels, ready for
Ultralytics, DETR or another detector. This page exports 20 frames of an example scene, cut into
640 px tiles, in both formats.

Before you start, you need:

- Python 3.10 or later and a Rust toolchain, to build the `skytwin` package.
- A checkout of this repository, for `examples/dirty-frame.json`.

## Steps

1. Install the package from the repository root:

   ```bash
   pip install ./crates/skytwin-py
   ```

2. Export the frames:

   ```bash
   skytwin-export examples/dirty-frame.json out/ --frames 20 --format yolo --format coco --tile 640 --overlap 64
   ```

   Frame k is the scene's frame k with seed + k, so running the command again with a larger
   `--frames` extends the same set.

3. Point your trainer at the output. For Ultralytics, give it `out/data.yaml`; for COCO tools,
   `out/annotations/instances_train.json` and `instances_val.json`.

## Result

The scene's 1024 × 1024 frames each give four tiles. The last 10% of the frames form the
validation split, so `out/images/train/` holds 72 PNG tiles and `out/images/val/` holds 8, each
with a label file in `out/labels/`. Stars are class 0 and satellites class 1.

The [labels reference](labels.md) lists every option, the output files and the label formats.

## Load frames in PyTorch

To train without writing files, `skytwin.torch` renders frames on the fly. It needs PyTorch
installed. `count` sets how many frames the dataset holds; without it, the scene's
`batch.frames` or 1:

```python
from torch.utils.data import DataLoader
from skytwin.torch import FrameDataset, collate, detection_target

dataset = FrameDataset("examples/dirty-frame.json", count=1000, transform=detection_target)
loader = DataLoader(dataset, batch_size=16, num_workers=4, collate_fn=collate)
for images, targets in loader:
    targets[0]["boxes"], targets[0]["labels"]
```

Frames hold different numbers of objects, so `collate` stacks the images and keeps the targets
as a list, one per frame.
