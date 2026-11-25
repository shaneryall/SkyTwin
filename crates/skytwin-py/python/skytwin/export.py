"""Write a training set: images with YOLO, YOLO-OBB or COCO labels.

    skytwin-export scene.json out/ --frames 5000 --format yolo --format coco --tile 640

Layout, as Ultralytics and COCO tools expect it::

    out/images/{train,val}/000123.png              000123_04.png for tile 4
    out/labels/{train,val}/000123.txt              YOLO or YOLO-OBB
    out/data.yaml                                  Ultralytics dataset file
    out/annotations/instances_{train,val}.json     COCO; file names relative to images/<split>/
    out/skytwin.json                               what was rendered, with which options

Frame k is the scene's frame k (seed + k), so the set can be rebuilt or
extended exactly. The last ``--val`` share of the frames (rounded up) is
the validation split, and every tile of a frame stays in its frame's split.
Images are the delivered pixels through :func:`skytwin.labels.stretch`
(8-bit, noise-normalised asinh) or, with ``--image raw16``, as delivered in
16 bits. Labels are those of :mod:`skytwin.labels`: objects at or above
``--min-snr`` (matched-filter signal-to-noise) inside each tile.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from . import Simulator, __version__
from . import labels as lab

__all__ = ["main"]


_OUTPUTS = ("images", "labels", "annotations", "truth", "data.yaml", "skytwin.json")
"""What an export writes in its output folder, and what ``--overwrite`` removes."""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skytwin-export",
        description="Render frames of a Skytwin scene into a YOLO or COCO training set.",
    )
    p.add_argument("scene", help="scene file (JSON)")
    p.add_argument("out", help="output folder")
    p.add_argument("--frames", type=int, help="frames to render (default: the scene's batch.frames, else 1)")
    p.add_argument("--start", type=int, default=0, help="first frame index (default 0)")
    p.add_argument("--seed", type=int, help="replace the scene's seed")
    p.add_argument(
        "--format",
        action="append",
        choices=["yolo", "yolo-obb", "coco"],
        help="label format, repeatable (default yolo); yolo and yolo-obb share labels/, so pick one",
    )
    p.add_argument(
        "--classes",
        default="star,satellite",
        help="kinds to label, separated by commas: star, satellite or star,satellite (default)",
    )
    p.add_argument("--by-form", action="store_true", help="split each kind into _point and _streak classes")
    p.add_argument("--min-snr", type=float, default=5.0, help="least matched-filter signal-to-noise (default 5)")
    p.add_argument("--min-visible", type=float, default=0.5, help="least share of a point's box in a tile (0.5)")
    p.add_argument("--min-radius", type=float, help="least box half-width before the half pixel, px (default FWHM)")
    p.add_argument("--streak-px", type=float, help="least span of a streak's path, px (default 2 × FWHM)")
    p.add_argument("--tile", type=int, help="cut frames into tiles of this many pixels")
    p.add_argument("--overlap", type=int, default=0, help="least overlap between tiles, px (default 0)")
    p.add_argument("--val", type=float, default=0.1, help="share of frames for validation (default 0.1)")
    p.add_argument("--image", choices=["asinh", "raw16"], default="asinh", help="image encoding (default asinh)")
    p.add_argument("--workers", type=int, help="render threads (default: every core)")
    p.add_argument("--keep-truth", action="store_true", help="also write each frame's truth to truth/")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an earlier export in the output folder (its images, labels, annotations and truth)",
    )
    return p


def _yaml(out: Path, names: list[str], has_val: bool) -> str:
    """Ultralytics dataset file."""
    lines = [
        "# Skytwin training set; see skytwin.json",
        f"path: {out.resolve()}",
        "train: images/train",
        f"val: images/{'val' if has_val else 'train'}",
        "names:",
        *(f"  {k}: {n}" for k, n in enumerate(names)),
    ]
    return "\n".join(lines) + "\n"


def val_count(count: int, val: float) -> int:
    """Frames held out for validation: a `val` share of `count`, rounded up, but never all of them.

    The training split keeps at least one frame, so a one-frame export is all training data.
    """
    return min(math.ceil(val * count), max(count - 1, 0))


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit status."""
    args = _parser().parse_args(argv)
    formats = set(args.format or ["yolo"])
    if {"yolo", "yolo-obb"} <= formats:
        print("skytwin-export: yolo and yolo-obb both write labels/; choose one", file=sys.stderr)
        return 2
    kinds = [k.strip() for k in args.classes.split(",") if k.strip()]
    if not kinds or any(k not in lab.CLASSES for k in kinds):
        print(f"skytwin-export: --classes takes {', '.join(lab.CLASSES)}", file=sys.stderr)
        return 2
    if not 0.0 <= args.val < 1.0:
        print("skytwin-export: --val must be at least 0 and below 1", file=sys.stderr)
        return 2
    if args.frames is not None and args.frames < 0:
        print("skytwin-export: --frames must be 0 or more", file=sys.stderr)
        return 2
    if args.tile is not None and not 0 <= args.overlap < args.tile:
        print("skytwin-export: --tile must be above 0 and --overlap at least 0 and below --tile", file=sys.stderr)
        return 2
    if args.tile is None and args.overlap:
        print("skytwin-export: --overlap needs --tile", file=sys.stderr)
        return 2
    earlier = [n for n in _OUTPUTS if (Path(args.out) / n).exists()]
    if earlier and not args.overwrite:
        print(
            f"skytwin-export: {args.out} holds an earlier export ({', '.join(earlier)}); "
            "pass --overwrite to replace it, or choose another folder",
            file=sys.stderr,
        )
        return 2
    for n in earlier:
        target = Path(args.out) / n
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    names = lab.class_names(args.by_form, kinds)
    sim = Simulator(args.scene, seed=args.seed)
    count = args.frames if args.frames is not None else max(len(sim) - args.start, 0)
    n_val = val_count(count, args.val)
    first_val = args.start + count - n_val
    out = Path(args.out)
    splits = ["train", "val"] if n_val else ["train"]
    yolo = formats & {"yolo", "yolo-obb"}
    for s in splits:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        if yolo:
            (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    if "coco" in formats:
        (out / "annotations").mkdir(parents=True, exist_ok=True)
    if args.keep_truth:
        (out / "truth").mkdir(parents=True, exist_ok=True)
    categories = [{"id": k + 1, "name": n, "supercategory": n.split("_")[0]} for k, n in enumerate(names)]
    coco = {s: {"images": [], "annotations": [], "categories": categories} for s in splits}
    tally = {s: {"frames": 0, "images": 0, "objects": dict.fromkeys(names, 0)} for s in splits}
    writes: list[Future] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in sim.stream(args.start, count, workers=args.workers):
            split = "val" if f.index >= first_val else "train"
            img = lab.stretch(f.image) if args.image == "asinh" else lab.raw16(f.image)
            h, w = img.shape
            objs = lab.objects(f.truth, min_radius_px=args.min_radius, streak_px=args.streak_px)
            frame_tiles = lab.tiles(w, h, args.tile, args.overlap)
            tally[split]["frames"] += 1
            for t, tile in enumerate(frame_tiles):
                stem = f"{f.index:06d}" if len(frame_tiles) == 1 else f"{f.index:06d}_{t:02d}"
                x0, y0, tw, th = tile
                bxs = lab.boxes(objs, tile, (w, h), min_snr=args.min_snr, min_visible=args.min_visible)
                crop = img[y0 : y0 + th, x0 : x0 + tw].copy()
                writes.append(pool.submit(lab.write_png, out / "images" / split / f"{stem}.png", crop))
                if yolo:
                    to_lines = lab.yolo_lines if "yolo" in formats else lab.yolo_obb_lines
                    lines = to_lines(bxs, names, (tw, th))
                    text = "\n".join(lines) + ("\n" if lines else "")
                    (out / "labels" / split / f"{stem}.txt").write_text(text)
                if "coco" in formats:
                    c = coco[split]
                    image_id = len(c["images"]) + 1
                    c["images"].append(
                        {
                            "id": image_id,
                            "file_name": f"{stem}.png",
                            "width": tw,
                            "height": th,
                            "skytwin": {"frame": f.index, "seed": f.seed, "tile": [x0, y0]},
                        }
                    )
                    c["annotations"] += lab.coco_annotations(bxs, names, image_id, len(c["annotations"]) + 1)
                tally[split]["images"] += 1
                for b in bxs:
                    for key in (b.obj.name, b.obj.kind):
                        if key in tally[split]["objects"]:
                            tally[split]["objects"][key] += 1
            if args.keep_truth:
                (out / "truth" / f"{f.index:06d}.truth.json").write_text(json.dumps(f.truth))
            done = f.index - args.start + 1
            if done % 100 == 0 or done == count:
                print(f"{done}/{count} frames", file=sys.stderr)
            if len(writes) > 64:
                for job in writes:
                    job.result()
                writes.clear()
        for job in writes:
            job.result()

    if yolo:
        (out / "data.yaml").write_text(_yaml(out, names, bool(n_val)))
    if "coco" in formats:
        for s in splits:
            (out / "annotations" / f"instances_{s}.json").write_text(json.dumps(coco[s]))
    manifest = {
        "skytwin_version": __version__,
        "scene": str(args.scene),
        "seed": args.seed,
        "frames": {"start": args.start, "count": count, "first_val": first_val if n_val else None},
        "formats": sorted(formats),
        "classes": names,
        "options": {
            "min_snr": args.min_snr,
            "min_visible": args.min_visible,
            "min_radius_px": args.min_radius,
            "streak_px": args.streak_px,
            "tile": args.tile,
            "overlap": args.overlap,
            "image": args.image,
        },
        "counts": tally,
    }
    (out / "skytwin.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
