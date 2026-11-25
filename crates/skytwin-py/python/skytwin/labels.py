"""Detector labels from a frame's truth: boxes, rotated boxes, YOLO and COCO.

Every star, streak and target in the truth carries a ``label``: the path it
was drawn along, the signal-to-noise of an ideal matched filter, the peak
pixel signal-to-noise, how far from the path it stands above the pixel
noise, and whether it saturates. :func:`objects` turns those into labelled
objects; :func:`boxes` cuts them to a tile of the image (or the whole of
it) as axis-aligned and rotated boxes; :func:`yolo_lines`,
:func:`yolo_obb_lines` and :func:`coco_annotations` write them out.

Coordinates. Skytwin puts pixel centres on whole numbers: ``(0, 0)`` is the
centre of ``image[0, 0]``. YOLO and COCO measure from the outer corner of
that pixel, so every label here is shifted by half a pixel. ``x`` runs
along a row and ``y`` down the array; images are written with array row 0
at the top, so a FITS viewer that draws row 0 at the bottom shows the same
frame flipped vertically.

Classes. Stars are ``star``; targets and pixel streaks are ``satellite``.
With ``by_form`` each splits into ``_point`` and ``_streak`` by how far
apart the farthest two points of its drawn path are: a streak spans more
than twice the PSF FWHM. Pointing jitter and small jumps leave a source a
point; tracking at a target's rate trails every star into a streak and
keeps the target a point.
"""

from __future__ import annotations

import itertools
import math
import struct
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "CLASSES",
    "FORM_CLASSES",
    "Box",
    "Obj",
    "boxes",
    "class_id",
    "class_names",
    "coco_annotations",
    "objects",
    "raw16",
    "stretch",
    "tiles",
    "write_png",
    "yolo_lines",
    "yolo_obb_lines",
]

CLASSES = ("star", "satellite")
"""Class names, in YOLO order."""
FORM_CLASSES = ("star_point", "star_streak", "satellite_point", "satellite_streak")
"""Class names when split by form, in YOLO order."""


@dataclass(frozen=True)
class Obj:
    """One labelled source, in Skytwin pixel coordinates."""

    kind: str
    """``star`` or ``satellite``."""
    form: str
    """``point`` or ``streak``."""
    path: np.ndarray
    """Drawn path, shape (n, 2), ``[x, y]`` with pixel centres on whole numbers."""
    radius: float
    """Half-width of the labelled region about the path, px (to pixel edges)."""
    snr: float
    """Matched-filter signal-to-noise over the whole image."""
    peak_snr: float
    """Highest single-pixel signal-to-noise."""
    saturated: bool
    """Whether it reaches full well."""
    source: str
    """Where it is in the truth: ``stars[3]``, ``streaks[0]`` or ``targets[0]``."""

    @property
    def name(self) -> str:
        """Class name with the form: ``star_point``, ``satellite_streak``, …"""
        return f"{self.kind}_{self.form}"


@dataclass(frozen=True)
class Box:
    """An object cut to a tile, in that tile's corner coordinates (pixels)."""

    obj: Obj
    xyxy: tuple[float, float, float, float]
    """Axis-aligned box ``(x0, y0, x1, y1)``."""
    corners: tuple[tuple[float, float], ...]
    """Rotated box: four corners in order around it, inside the tile. A
    streak's box is a rectangle along it, cut by the tile's edge where it
    crosses one (see ``boxes``)."""
    snr: float
    """Matched-filter signal-to-noise of the part inside the tile (for a
    streak, the whole value scaled by the square root of the share of its
    on-image length inside)."""


def class_names(by_form: bool = False, kinds: Sequence[str] = CLASSES) -> list[str]:
    """Class names in YOLO order for the chosen kinds."""
    names = FORM_CLASSES if by_form else CLASSES
    return [n for n in names if n.split("_")[0] in kinds]


def objects(truth: dict, *, min_radius_px: float | None = None, streak_px: float | None = None) -> list[Obj]:
    """Every source in a frame's truth that has a drawn path.

    ``radius`` is the label's ``extent_px`` or ``min_radius_px`` (default:
    the camera FWHM), whichever is larger, plus half a pixel to reach the
    edges of the outermost pixels. A source is a streak when the farthest
    two points of its path are more than ``streak_px`` apart (default:
    twice the FWHM), and otherwise a point.
    """
    fwhm = float(truth["camera"].get("fwhm_px", 2.0))
    floor = fwhm if min_radius_px is None else min_radius_px
    streak = 2.0 * fwhm if streak_px is None else streak_px
    out = []
    groups = (("stars", "star"), ("streaks", "satellite"), ("targets", "satellite"))
    for key, kind in groups:
        for i, src in enumerate(truth.get(key, [])):
            lab = src.get("label")
            if not lab or not lab["path"]:
                continue
            path = np.asarray(lab["path"], dtype=np.float64).reshape(-1, 2)
            form = "streak" if _span(path) > streak else "point"
            out.append(
                Obj(
                    kind=kind,
                    form=form,
                    path=path,
                    radius=max(float(lab["extent_px"]), floor) + 0.5,
                    snr=float(lab["snr"]),
                    peak_snr=float(lab["peak_snr"]),
                    saturated=bool(lab["saturated"]),
                    source=f"{key}[{i}]",
                )
            )
    return out


def _length(path: np.ndarray) -> float:
    """Length of a polyline."""
    return float(np.hypot(*np.diff(path, axis=0).T).sum()) if len(path) > 1 else 0.0


def _span(path: np.ndarray) -> float:
    """Largest distance between two points of a path."""
    d = path[:, None, :] - path[None, :, :]
    return float(np.sqrt((d * d).sum(axis=-1)).max())


def _clip(path: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> list[np.ndarray]:
    """The parts of a polyline inside a rectangle (Liang–Barsky per piece).

    A one-point path is kept when the point is inside. Each part is an
    array of points; touching pieces join into one part.
    """
    if len(path) == 1:
        x, y = path[0]
        return [path.copy()] if x0 <= x <= x1 and y0 <= y <= y1 else []
    parts: list[list[tuple[float, float]]] = []
    for (ax, ay), (bx, by) in itertools.pairwise(path):
        dx, dy = bx - ax, by - ay
        lo, hi = 0.0, 1.0
        for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
            if p == 0:
                if q < 0:
                    lo, hi = 1.0, 0.0
                    break
                continue
            t = q / p
            if p < 0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
        if lo > hi:
            continue
        a = (float(ax), float(ay)) if lo == 0.0 else (ax + lo * dx, ay + lo * dy)
        b = (float(bx), float(by)) if hi == 1.0 else (ax + hi * dx, ay + hi * dy)
        if parts and parts[-1][-1] == a:
            parts[-1].append(b)
        else:
            parts.append([a, b])
    return [np.asarray(p) for p in parts]


def boxes(
    objs: Iterable[Obj],
    tile: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    min_snr: float = 5.0,
    min_visible: float = 0.5,
) -> list[Box]:
    """Objects cut to ``tile`` = ``(x0, y0, width, height)`` in pixels of
    an image of ``image_size`` = ``(width, height)``.

    A point is kept when at least ``min_visible`` of its box area lies in
    the tile; a streak when part of its path does. Either needs a
    signal-to-noise of at least ``min_snr`` inside the tile. A point's box
    is its path widened by its radius; a streak's is the part of its path
    within a radius of the tile, widened by the radius; both are clamped to
    the tile. A streak's rotated box runs along its part in the tile from
    the first to the last point, cut by the tile's edge where it crosses one,
    so it holds all of the streak inside the tile; a point's is its box.
    """
    tx, ty, tw, th = tile
    iw, ih = image_size
    rect = (tx - 0.5, ty - 0.5, tx + tw - 0.5, ty + th - 0.5)
    image = (-0.5, -0.5, iw - 0.5, ih - 0.5)
    out = []
    for o in objs:
        r = o.radius
        if o.form == "point":
            lo, hi = o.path.min(axis=0), o.path.max(axis=0)
            full = (lo[0] - r, lo[1] - r, hi[0] + r, hi[1] + r)
            cut = _intersect(full, rect)
            if cut is None or _area(cut) < min_visible * _area(full) or o.snr < min_snr:
                continue
            xyxy = (cut[0] - rect[0], cut[1] - rect[1], cut[2] - rect[0], cut[3] - rect[1])
            corners = ((xyxy[0], xyxy[1]), (xyxy[2], xyxy[1]), (xyxy[2], xyxy[3]), (xyxy[0], xyxy[3]))
            out.append(Box(o, tuple(float(v) for v in xyxy), corners, o.snr))
            continue
        inside = _clip(o.path, *rect)
        on_image = sum(_length(p) for p in _clip(o.path, *image))
        length = sum(_length(p) for p in inside)
        if length <= 0.0 or on_image <= 0.0:
            continue
        snr = o.snr * math.sqrt(min(length / on_image, 1.0))
        near = np.concatenate(_clip(o.path, rect[0] - r, rect[1] - r, rect[2] + r, rect[3] + r))
        lo, hi = near.min(axis=0), near.max(axis=0)
        cut = _intersect((lo[0] - r, lo[1] - r, hi[0] + r, hi[1] + r), rect)
        if cut is None or snr < min_snr:
            continue
        xyxy = (cut[0] - rect[0], cut[1] - rect[1], cut[2] - rect[0], cut[3] - rect[1])
        corners = _rotated(np.concatenate(inside) - np.array(rect[:2]), r, tw, th)
        out.append(Box(o, tuple(float(v) for v in xyxy), corners, snr))
    return out


def _intersect(a, b):
    """Overlap of two ``(x0, y0, x1, y1)`` rectangles, or None."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _area(b) -> float:
    """Area of a ``(x0, y0, x1, y1)`` rectangle."""
    return (b[2] - b[0]) * (b[3] - b[1])


def _rotated(pts: np.ndarray, r: float, w: float, h: float) -> tuple[tuple[float, float], ...]:
    """Four corners, in order around them, of a box along ``pts`` (first to
    last point) that holds them all widened by ``r`` and lies inside a
    ``w × h`` tile.

    The box starts as the rectangle along the points, widened by ``r``. Where
    it crosses the tile's edge it is cut by the edge, so it keeps every part
    of the source inside the tile; a cut leaves up to eight corners, and the
    shortest edge is replaced by extending its neighbours until they meet,
    provided they meet inside the tile, until four remain. A box that can't
    be brought to four that way, or that the tile cuts to a sliver, gets the
    corners of its axis-aligned bounds instead.
    """
    d = pts[-1] - pts[0]
    n = math.hypot(*d)
    u = d / n if n > 1e-9 else np.array([1.0, 0.0])
    v = np.array([-u[1], u[0]])
    a, b = pts @ u, pts @ v
    (a0, a1), (b0, b1) = (a.min() - r, a.max() + r), (b.min() - r, b.max() + r)
    rect = [a0 * u + b0 * v, a1 * u + b0 * v, a1 * u + b1 * v, a0 * u + b1 * v]
    poly = _clean(_clip_polygon(rect, w, h))
    while len(poly) > 4:
        merged = _merge_shortest(poly, w, h)
        if merged is None:
            break
        poly = _clean(merged)
    if len(poly) == 4:
        return tuple((float(np.clip(x, 0, w)), float(np.clip(y, 0, h))) for x, y in poly)
    pts_all = np.array(poly if len(poly) >= 3 else rect)
    x0, y0 = np.clip(pts_all.min(axis=0), 0, [w, h])
    x1, y1 = np.clip(pts_all.max(axis=0), 0, [w, h])
    return ((float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1)))


def _clip_polygon(poly, w: float, h: float) -> list[np.ndarray]:
    """``poly`` (corners in order) cut to the tile ``[0, w] × [0, h]``
    (Sutherland–Hodgman)."""
    out = [np.asarray(p, float) for p in poly]
    for axis, limit, keep_above in ((0, 0.0, True), (0, w, False), (1, 0.0, True), (1, h, False)):
        if not out:
            break
        src, out = out, []
        for i, cur in enumerate(src):
            prev = src[i - 1]
            cur_in = cur[axis] >= limit if keep_above else cur[axis] <= limit
            prev_in = prev[axis] >= limit if keep_above else prev[axis] <= limit
            if cur_in != prev_in:
                t = (limit - prev[axis]) / (cur[axis] - prev[axis])
                out.append(prev + t * (cur - prev))
            if cur_in:
                out.append(cur)
    return out


def _clean(poly: list[np.ndarray], tol: float = 1e-9) -> list[np.ndarray]:
    """``poly`` without repeated corners or corners on a straight edge."""
    pts = [p for i, p in enumerate(poly) if np.hypot(*(p - poly[i - 1])) > tol]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            if abs(cross) <= tol * max(1.0, np.hypot(*(p2 - p0))):
                del pts[i]
                changed = True
                break
    return pts


def _merge_shortest(poly: list[np.ndarray], w: float, h: float) -> list[np.ndarray] | None:
    """``poly`` with one edge removed by extending its two neighbours to
    their crossing, choosing the edge that adds the least area and keeps the
    crossing inside the tile; None when no edge can go."""
    n = len(poly)
    best = None
    for i in range(n):
        p_prev, p0, p1, p_next = poly[i - 1], poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        d0, d1 = p0 - p_prev, p1 - p_next
        den = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(den) < 1e-12:
            continue
        diff = p1 - p0
        t = (diff[0] * d1[1] - diff[1] * d1[0]) / den
        u = (diff[0] * d0[1] - diff[1] * d0[0]) / den
        if t <= 0 or u <= 0:
            continue
        x = p0 + t * d0
        if not (-1e-9 <= x[0] <= w + 1e-9 and -1e-9 <= x[1] <= h + 1e-9):
            continue
        added = 0.5 * abs((p1[0] - p0[0]) * (x[1] - p0[1]) - (p1[1] - p0[1]) * (x[0] - p0[0]))
        if best is None or added < best[0]:
            best = (added, i, x)
    if best is None:
        return None
    _, i, x = best
    j = (i + 1) % n
    out = [p for k, p in enumerate(poly) if k not in (i, j)]
    out.insert(i if j > i else i - 1, x)
    return out


def class_id(o: Obj, names: Sequence[str]) -> int | None:
    """Index of an object's class in ``names`` (with or without its form), or None."""
    for key in (o.name, o.kind):
        if key in names:
            return list(names).index(key)
    return None


def yolo_lines(bxs: Iterable[Box], names: Sequence[str], tile_size: tuple[int, int]) -> list[str]:
    """Ultralytics detection labels: ``class cx cy w h``, normalised."""
    w, h = tile_size
    out = []
    for b in bxs:
        k = class_id(b.obj, names)
        if k is None:
            continue
        x0, y0, x1, y1 = b.xyxy
        out.append(f"{k} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return out


def yolo_obb_lines(bxs: Iterable[Box], names: Sequence[str], tile_size: tuple[int, int]) -> list[str]:
    """Ultralytics oriented-box labels: ``class x1 y1 x2 y2 x3 y3 x4 y4``, normalised."""
    w, h = tile_size
    out = []
    for b in bxs:
        k = class_id(b.obj, names)
        if k is None:
            continue
        xy = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in b.corners)
        out.append(f"{k} {xy}")
    return out


def coco_annotations(bxs: Iterable[Box], names: Sequence[str], image_id: int, first_id: int) -> list[dict]:
    """COCO annotations (``bbox`` as ``[x, y, w, h]``, category ids from 1),
    each with a ``skytwin`` record: form, signal-to-noise, saturation,
    rotated box and source in the truth."""
    out = []
    for b in bxs:
        k = class_id(b.obj, names)
        if k is None:
            continue
        x0, y0, x1, y1 = b.xyxy
        out.append(
            {
                "id": first_id + len(out),
                "image_id": image_id,
                "category_id": k + 1,
                "bbox": [round(x0, 3), round(y0, 3), round(x1 - x0, 3), round(y1 - y0, 3)],
                "area": round((x1 - x0) * (y1 - y0), 3),
                "iscrowd": 0,
                "skytwin": {
                    "form": b.obj.form,
                    "snr": round(b.snr, 3),
                    "peak_snr": round(b.obj.peak_snr, 3),
                    "saturated": b.obj.saturated,
                    "rotated_box": [[round(x, 3), round(y, 3)] for x, y in b.corners],
                    "source": b.obj.source,
                },
            }
        )
    return out


def tiles(width: int, height: int, size: int | None = None, overlap: int = 0) -> list[tuple[int, int, int, int]]:
    """Tiles ``(x0, y0, w, h)`` of at most ``size`` px covering the image,
    overlapping by at least ``overlap`` px, the last in each row and column
    flush with the edge. One tile (the image) without a size."""

    def starts(n: int) -> list[int]:
        if size is None or size >= n:
            return [0]
        step = max(size - overlap, 1)
        s = list(range(0, n - size, step))
        return [*s, n - size]

    tw = width if size is None else min(size, width)
    th = height if size is None else min(size, height)
    return [(x, y, tw, th) for y in starts(height) for x in starts(width)]


def stretch(image: np.ndarray, *, soft: float = 3.0, top: float = 1000.0) -> np.ndarray:
    """8-bit image for a detector: noise-normalised, then an asinh stretch.

    The image is measured in units of its noise, ``z = (image − median) /
    σ`` with σ from the median absolute deviation, and ``z`` from −2 to
    ``top`` maps through ``asinh(z / soft)`` onto 0–255. A source's grey
    level then depends on its signal-to-noise alone, the same in every
    frame however bright the sky.
    """
    a = np.nan_to_num(image.astype(np.float64))
    med = float(np.median(a))
    sigma = 1.4826 * float(np.median(np.abs(a - med)))
    if sigma <= 0:
        sigma = float(a.std()) or 1.0
    lo = math.asinh(-2.0 / soft)
    span = math.asinh(top / soft) - lo
    z = np.clip((a - med) / sigma, -2.0, top)
    return np.round((np.arcsinh(z / soft) - lo) / span * 255.0).astype(np.uint8)


def raw16(image: np.ndarray) -> np.ndarray:
    """The image as delivered, as 16-bit unsigned (other types rounded and clipped)."""
    if image.dtype == np.uint16:
        return image
    return np.clip(np.round(np.nan_to_num(image.astype(np.float64))), 0, 65535).astype(np.uint16)


def write_png(path: str | Path, image: np.ndarray, level: int = 3) -> None:
    """Write a 2-D ``uint8`` or ``uint16`` array as a greyscale PNG, row 0 at the top."""
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError("write_png takes a 2-D uint8 or uint16 array")
    h, w = image.shape
    depth = 8 * image.dtype.itemsize
    data = np.ascontiguousarray(image, dtype=">u2" if depth == 16 else np.uint8).view(np.uint8)
    rows = np.zeros((h, 1 + w * image.dtype.itemsize), np.uint8)
    rows[:, 1:] = data.reshape(h, -1)

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    header = struct.pack(">IIBBBBB", w, h, depth, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    png += chunk(b"IDAT", zlib.compress(rows.tobytes(), level)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)
