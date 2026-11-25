"""A PyTorch dataset of simulated frames, rendered on the fly.

    from skytwin.torch import FrameDataset, collate, detection_target
    ds = FrameDataset("scene.json", transform=detection_target)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4, collate_fn=collate)
    for images, targets in loader:
        ...                                        # targets[i]["boxes"], targets[i]["labels"]

Each DataLoader worker renders its own share of the indices, on
``threads`` Rust threads. ``transform(frame)`` turns a :class:`skytwin.Frame`
into whatever the model takes; by default it gives the image as a float32
tensor and the frame's index, which PyTorch's default collation batches.
:func:`detection_target` gives the stretched
image and boxes in the form torchvision detectors (``xyxy``) or DETR
(``cxcywh``) take. Frames hold different numbers of objects, so batch them
with :func:`collate`, which stacks the images and keeps one target per
image; PyTorch's default collation cannot.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import Frame, Simulator
from . import labels as lab

__all__ = ["FrameDataset", "collate", "default_transform", "detection_target"]


def default_transform(frame: Frame) -> tuple[torch.Tensor, int]:
    """The image as a float32 tensor, and the frame's index.

    The index, not the truth dict, so a plain ``DataLoader`` can batch the
    pairs; frames hold different numbers of objects, which PyTorch's default
    collation cannot stack. For labels, use :func:`detection_target` with
    :func:`collate`.
    """
    return torch.from_numpy(frame.image.astype("float32")), frame.index


def detection_target(
    frame: Frame,
    *,
    kinds: Sequence[str] = lab.CLASSES,
    by_form: bool = False,
    streak_px: float | None = None,
    min_snr: float = 5.0,
    box_format: str = "xyxy",
) -> tuple[torch.Tensor, dict]:
    """The frame as a detector's input and target.

    The image is :func:`skytwin.labels.stretch` scaled to 0–1, shape
    ``(1, rows, columns)``. The target holds ``boxes`` (``xyxy`` in pixels
    from the image corner, or ``cxcywh`` as fractions of the image), class
    indices ``labels`` into :func:`skytwin.labels.class_names`, each
    object's ``snr``, and the frame's ``index``.
    """
    if box_format not in ("xyxy", "cxcywh"):
        raise ValueError("box_format is 'xyxy' or 'cxcywh'")
    img = lab.stretch(frame.image)
    h, w = img.shape
    names = lab.class_names(by_form, kinds)
    objs = lab.objects(frame.truth, streak_px=streak_px)
    bxs = lab.boxes(objs, (0, 0, w, h), (w, h), min_snr=min_snr)
    rows, classes, snr = [], [], []
    for b in bxs:
        k = lab.class_id(b.obj, names)
        if k is None:
            continue
        x0, y0, x1, y1 = b.xyxy
        if box_format == "xyxy":
            rows.append([x0, y0, x1, y1])
        else:
            rows.append([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h])
        classes.append(k)
        snr.append(b.snr)
    target = {
        "boxes": torch.tensor(rows, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(classes, dtype=torch.int64),
        "snr": torch.tensor(snr, dtype=torch.float32),
        "index": frame.index,
    }
    return torch.from_numpy(img).float().div_(255.0).unsqueeze(0), target


def collate(batch: list[tuple[torch.Tensor, object]]) -> tuple[torch.Tensor, list]:
    """Stack the images of ``(image, target)`` pairs and keep the targets as a list."""
    images, targets = zip(*batch, strict=True)
    return torch.stack(images), list(targets)


class FrameDataset(IterableDataset):
    """Frames ``start`` to ``start + count - 1`` of a scene file's batch.

    Under a DataLoader each worker renders one contiguous share of the
    indices, on ``threads`` Rust threads (default 1 per worker; every core
    without workers).
    """

    def __init__(
        self, scene, *, start=0, count=None, seed=None, expected=False, transform=default_transform, threads=None
    ):
        self.scene, self.seed, self.expected = scene, seed, expected
        self.start = start
        self.count = count if count is not None else len(Simulator(scene, seed)) - start
        self.transform, self.threads = transform, threads

    def __len__(self):
        return self.count

    def __iter__(self):
        sim = Simulator(self.scene, self.seed)
        info = get_worker_info()
        lo, hi = self.start, self.start + self.count
        threads = self.threads
        if info is not None:
            per = -(-self.count // info.num_workers)
            lo = self.start + info.id * per
            hi = min(lo + per, self.start + self.count)
            threads = threads or 1
        for f in sim.stream(lo, max(hi - lo, 0), workers=threads, expected=self.expected):
            yield self.transform(f)
