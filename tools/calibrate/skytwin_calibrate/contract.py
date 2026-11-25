"""The profile as a contract: what a learned profile needs to be used.

A learned profile is one JSON file that holds everything tuned to a
sensor, plus what it depends on:

- `schema_version`: the profile format (1);
- `sensor`: what it describes (name, platform, passband);
- `catalog`: the star catalogue it was calibrated against. Its zeropoint,
  colour term and magnitude limit hold for that catalogue only, so Skytwin
  fills a scene's `catalog` from it and refuses another kind; the scene
  gives just `"catalog": {"path": "..."}`. A CSV catalogue is pinned by
  SHA-256 and row count;
- `learned`: provenance (tool, version, inputs, date).

`check` verifies a local catalogue against a profile; `describe` prints
what a profile is and a scene to start from.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
from collections.abc import Iterable, Mapping
from typing import Any

from . import __version__
from .util import is_csv, most_common, open_text, read_json, sha256

SCHEMA_VERSION = 1

GAIA_COLUMNS = {
    "id": "source_id",
    "ra": "ra",
    "dec": "dec",
    "mag": "phot_g_mean_mag",
    "pmra": "pmra",
    "pmdec": "pmdec",
    "parallax": "parallax",
    "color": "bp_rp",
}
"""Skytwin catalogue column -> Gaia DR3 column name."""


def csv_rows(path: str | os.PathLike) -> tuple[list[str], int]:
    """Header columns and data row count of a CSV file (optionally gzipped)."""
    with open_text(path) as fh:
        head = fh.readline().strip().split(",")
        return head, sum(1 for _ in fh)


def catalog_block(
    src: str | os.PathLike,
    mag_limit: float,
    name: str | None = None,
    source: str | None = None,
    with_colour: bool = False,
) -> dict[str, Any]:
    """The profile's `catalog` requirement for catalogue `src` (a UCAC4 directory or a CSV file).

    A CSV is pinned by its Gaia columns present, epoch J2016.0, SHA-256 and row count; the colour column is
    recorded only `with_colour` (when the profile has a colour term). `name` defaults to the file or
    directory name; `source` says how to get the catalogue again.
    """
    csv = is_csv(src)
    blk = {
        "name": name or os.path.basename(str(src).rstrip("/")),
        "kind": "csv" if csv else "ucac4",
        "mag_limit": round(float(mag_limit), 1),
    }
    if csv:
        head, n = csv_rows(src)
        cols = {k: v for k, v in GAIA_COLUMNS.items() if v in head}
        if not with_colour:
            cols.pop("color", None)
        blk.update(columns=cols, epoch_jyear=2016.0, sha256=sha256(src), rows=n)
    else:
        blk["band"] = "default"
    if source:
        blk["source"] = source
    return blk


def _git_commit() -> str | None:
    """Short commit hash of the checkout this package runs from, or None outside one."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except OSError:
        return None


def stamp(
    prof: dict[str, Any],
    heads: Iterable[Mapping[str, Any]],
    inputs: Iterable[str | os.PathLike],
    n_frames: int,
    catalog: dict[str, Any] | None = None,
    sensor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the contract fields into a learned profile in place, and return it.

    The sensor name, unless given, is the most common INSTRUME, TELESCOP or DETECTOR value in `heads`.
    Inputs that exist on disk are recorded by base name only.
    """
    prof["schema_version"] = SCHEMA_VERSION
    sen = dict(sensor or {})
    if "name" not in sen:
        for k in ("INSTRUME", "TELESCOP", "DETECTOR"):
            v = [str(h.get(k)) for h in heads if h.get(k)]
            if v:
                sen["name"] = most_common(v)
                break
        sen.setdefault("name", "unnamed sensor")
    prof["sensor"] = sen
    if catalog:
        prof["catalog"] = catalog
    prof["learned"] = {
        "tool": "skytwin-calibrate learn",
        "version": __version__,
        "commit": _git_commit(),
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frames": n_frames,
        "inputs": [os.path.basename(str(i).rstrip("/")) if os.path.exists(str(i)) else str(i) for i in inputs],
        "report": "beside the profile: <name>.learn.md",
    }
    return prof


def check(profile_path: str | os.PathLike, catalog_path: str | os.PathLike) -> list[str]:
    """Problems with using the catalogue at `catalog_path` with the profile; empty when it fits."""
    prof = read_json(profile_path)
    req = prof.get("catalog")
    if not req:
        return ["the profile names no catalogue (made before contracts, or learned without one)"]
    out = []
    if req["kind"] == "csv":
        if not os.path.isfile(catalog_path):
            return [f"{catalog_path}: not a file (the profile wants a CSV: {req['name']})"]
        head, n = csv_rows(catalog_path)
        missing = [v for v in (req.get("columns") or {}).values() if v not in head]
        if missing:
            out.append(f"columns missing: {', '.join(missing)}")
        if req.get("rows") is not None and n != req["rows"]:
            out.append(f"{n} rows, the profile was calibrated with {req['rows']}")
        if req.get("sha256") and sha256(catalog_path) != req["sha256"]:
            out.append(
                "SHA-256 differs from the file the profile was calibrated with "
                "(a different extract: the zeropoint and colour term may not hold)"
            )
    elif not os.path.isdir(os.path.join(catalog_path, "u4b")):
        out.append(f"{catalog_path}: no u4b/ inside (the profile wants a UCAC4 directory)")
    return out


def describe(profile_path: str | os.PathLike) -> str:
    """Markdown summary of a profile (sensor, image, blocks, catalogue, provenance) and a scene to start from."""
    prof = read_json(profile_path)
    cam = prof.get("camera", {})
    sen = prof.get("sensor") or {}
    req = prof.get("catalog")
    lines = [f"# {sen.get('name', prof.get('name', 'profile'))}", ""]
    if sen.get("platform"):
        lines.append(f"Platform: {sen['platform']}")
    if sen.get("band_nm"):
        lines.append(f"Passband: {sen['band_nm'][0]:.0f}-{sen['band_nm'][1]:.0f} nm")
    lines.append(f"Schema version: {prof.get('schema_version', 'none (hand-written or older)')}")
    lines.append(
        f"Image: {cam.get('width')} x {cam.get('height')} px, {cam.get('pixel_scale_arcsec')} arcsec/px, "
        f"exposure {cam.get('exposure_s')} s, zeropoint {cam.get('zeropoint_mag')}"
    )
    meta = (
        "name",
        "description",
        "camera",
        "schema_version",
        "sensor",
        "catalog",
        "learned",
        "detector_seed",
        "header",
    )
    blocks = [k for k in prof if k not in meta]
    lines.append(f"Blocks: {', '.join(blocks)}")
    if req:
        lines += ["", f"Catalogue: {req['name']} ({req['kind']}), to magnitude {req['mag_limit']}"]
        if req.get("sha256"):
            lines.append(f"  file SHA-256 {req['sha256'][:16]}..., {req.get('rows')} rows")
        if req.get("source"):
            lines.append(f"  get it: {req['source']}")
    else:
        lines += ["", "Catalogue: not named (scenes must give a full scene.catalog)"]
    if prof.get("learned"):
        lr = prof["learned"]
        lines += [
            "",
            f"Learned by {lr.get('tool')} {lr.get('version')} ({lr.get('commit') or 'commit unknown'}) on "
            f"{lr.get('date_utc')} from {lr.get('frames')} frames: {', '.join(lr.get('inputs', []))}",
        ]
    scene = {
        "instrument": os.path.abspath(profile_path),
        "scene": {
            "time": {"start_utc": "2026-01-01T00:00:00Z"},
            "pointing": {"ra_deg": 0.0, "dec_deg": 0.0, "roll_deg": 0.0},
        },
    }
    if req:
        scene["scene"]["catalog"] = {"path": "/path/to/" + ("catalogue.csv" if req["kind"] == "csv" else "ucac4")}
    lines += ["", "A scene to start from:", "", "```json", json.dumps(scene, indent=2), "```"]
    return "\n".join(lines)
