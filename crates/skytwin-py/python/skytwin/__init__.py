"""Skytwin from Python: simulated frames and exact labels, in memory.

    import skytwin
    sim = skytwin.Simulator("scene.json")      # a scene file, or a dict
    for frame in sim.stream(count=1000):        # rendered ahead on all cores
        frame.image                             # numpy array (rows, columns)
        frame.truth["stars"], frame.truth["streaks"]

Frame k uses seed + k and draws everything random anew; with
``batch.random_pointing`` it also points somewhere new. A frame depends only
on its index, so the same index always gives the same pixels, whatever the
order or the number of workers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from . import _native

__all__ = ["Frame", "Simulator", "__version__"]
__version__ = _native.__version__


class Frame(NamedTuple):
    """One simulated frame."""

    index: int
    """Index in the batch."""
    seed: int
    """Seed it was drawn with (the file's seed + index)."""
    image: np.ndarray
    """Pixels as delivered (the profile's output type), shape (rows, columns)."""
    truth: dict
    """Everything drawn: stars, streaks, targets, WCS, pointing, detector
    effects (the same as the ``.truth.json`` file)."""
    expected: np.ndarray | None
    """Noise-free electrons per pixel (float32), when asked for: a clean
    target for denoising or detection."""


def _frame(t: tuple) -> Frame:
    """A :class:`Frame` from the extension's raw tuple (pixel and expected buffers wrapped without a copy)."""
    index, seed, w, h, dtype, image, truth, expected = t
    img = np.frombuffer(image, dtype=dtype).reshape(h, w)
    exp = None if expected is None else np.frombuffer(expected, dtype="<f4").reshape(h, w)
    return Frame(index, seed, img, json.loads(truth), exp)


class Simulator:
    """A scene file loaded once, to render many frames.

    ``scene`` is a path to a scene file or the same content as a dict.
    Instrument profile paths inside it are read relative to the working
    directory. ``seed`` replaces the file's seed.
    """

    def __init__(self, scene: str | os.PathLike | dict[str, Any], seed: int | None = None):
        if isinstance(scene, dict):
            self._native = _native.Native.from_json(json.dumps(scene), seed)
        else:
            self._native = _native.Native.from_path(str(Path(scene)), seed)

    def __len__(self) -> int:
        """Frames in the batch (``batch.frames``, else 1)."""
        return self._native.frames()

    def render(self, index: int = 0, *, expected: bool = False) -> Frame:
        """Render frame ``index`` (thread-safe; the GIL is released)."""
        return _frame(self._native.render(index, expected))

    def stream(
        self,
        start: int = 0,
        count: int | None = None,
        *,
        workers: int | None = None,
        prefetch: int | None = None,
        expected: bool = False,
    ) -> Iterator[Frame]:
        """Frames ``start`` to ``start + count - 1`` in order, rendered ahead
        on ``workers`` threads (default: every core), at most ``prefetch``
        finished frames waiting (default: 2 per worker). ``count`` defaults
        to the rest of the batch."""
        if count is None:
            count = max(len(self) - start, 0)
        workers = workers or os.cpu_count() or 1
        prefetch = prefetch or 2 * workers
        s = self._native.stream(start, count, workers, prefetch, expected)
        try:
            for t in s:
                yield _frame(t)
        finally:
            s.close()

    def __iter__(self) -> Iterator[Frame]:
        """All frames of the batch (see :meth:`stream`)."""
        return self.stream()
