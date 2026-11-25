"""Helpers shared by the measuring modules: MAD statistics, header values, files and the skytwin runner."""

import gzip
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone

import numpy as np

MAD_TO_SIGMA = 1.4826
"""Gaussian σ per median absolute deviation (1 / Φ⁻¹(3/4))."""

FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
"""FWHM of a Gaussian in units of its σ."""


def mad_sigma(x, centre=None, axis=None, keepdims=False):
    """Gaussian σ from the median absolute deviation: 1.4826 × median |x − centre|.

    `centre` defaults to the median of `x` over `axis`; pass 0 for data already centred.
    """
    x = np.asarray(x)
    if centre is None:
        centre = np.median(x, axis=axis, keepdims=True)
    return MAD_TO_SIGMA * np.median(np.abs(x - centre), axis=axis, keepdims=keepdims)


def percentiles(v, digits=None):
    """The 0th to 100th percentiles of `v` in steps of 1 (101 floats), rounded to `digits` when given.

    This is the table Skytwin's `{"empirical": [...]}` distributions take.
    """
    q = np.percentile(v, np.linspace(0, 100, 101))
    return [float(x) if digits is None else round(float(x), digits) for x in q]


def spread(items, n):
    """At most `n` items, evenly spaced through `items` and always the same ones."""
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def number(v):
    """A header value as a float, or None when it is not a number (booleans are not)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def header_number(h, key):
    """Header card `key` as a float (see `number`), or None when `key` is empty, absent or not numeric."""
    if not key or key not in h:
        return None
    return number(h[key])


def utc_timestamp(s):
    """POSIX seconds of an ISO 8601 time string (a trailing Z allowed), or None when it does not parse.

    A time with no UTC offset is read as UTC, never as the machine's local time, so frame spacing
    does not depend on the time zone or jump at a daylight-saving change.
    """
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp()


def most_common(values):
    """The most frequent of `values`; on a tie, the one seen first, so the result is the same on every run."""
    return Counter(values).most_common(1)[0][0]


def read_json(path):
    """The JSON document at `path`."""
    with open(path) as fh:
        return json.load(fh)


def write_json(obj, path, **kwargs):
    """Write `obj` as JSON to `path`; `kwargs` go to `json.dump`."""
    with open(path, "w") as fh:
        json.dump(obj, fh, **kwargs)


def write_rows(rows, path):
    """Write per-frame measurement rows: NaN refused, anything not serialisable written as null."""
    write_json(rows, path, allow_nan=False, default=lambda o: None)


def is_csv(path):
    """True for a CSV star catalogue (`.csv` or `.csv.gz`), False for a UCAC4 directory."""
    return str(path).lower().endswith((".csv", ".csv.gz"))


def open_text(path):
    """`path` opened for reading as text, through gzip when it ends in `.gz`."""
    return (gzip.open if str(path).endswith(".gz") else open)(path, "rt")


def sha256(path, chunk=1 << 20):
    """Hex SHA-256 of the file at `path`."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def run_skytwin(skytwin, scene, out_fits, check=True):
    """Render `scene` (a dict) to `out_fits` with the `skytwin` binary; the scene is written beside it as `.json`.

    Returns True on success. A failure raises RuntimeError with skytwin's message, or returns False when
    `check` is false. The truth file lands at `out_fits` with `.fits` replaced by `.truth.json`.
    """
    write_json(scene, out_fits + ".json", allow_nan=False)
    r = subprocess.run([skytwin, out_fits + ".json", out_fits], capture_output=True, text=True)
    if r.returncode != 0:
        if check:
            raise RuntimeError(f"skytwin failed: {r.stderr.strip()}")
        return False
    return True
