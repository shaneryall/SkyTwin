"""Realism report: real and simulated statistics side by side."""

from pathlib import Path

import numpy as np
from scipy import stats

from .util import read_json

METRICS = [
    ("median", "Background level", "ADU"),
    ("noise", "Background noise (MAD)", "ADU"),
    ("zero_frac", "Pixels at the lower clip", "fraction"),
    ("ramp_y", "Ramp along y", "ADU"),
    ("ramp_x", "Ramp along x", "ADU"),
    ("structure", "Residual large-scale structure", "ADU"),
    ("field.rms", "Structure with sources masked", "ADU"),
    ("field.mask_frac", "Area covered by sources", "fraction"),
    ("pattern.amplitude", "Pattern amplitude", "ADU"),
    ("pattern.period", "Pattern period", "px"),
    ("pattern.angle", "Pattern angle", "deg"),
    ("sharp_per_s", "Sharp single-pixel events", "per s"),
    ("bad_rows", "Corrupted read-out rows", "count"),
    ("row_alt", "Odd/even row offset", "ADU"),
    ("saturated", "Saturated pixels", "count"),
]


def value(r, key):
    """Quantity `key` of a measurement row (dotted keys reach into nested dicts); None when absent.

    `sharp_per_s` is the sharp-event count over the exposure time.
    """
    if key == "sharp_per_s":
        e = r["meta"]["exposure_s"]
        return r["sharp"] / e if e and r.get("sharp") is not None else None
    v = r
    for part in key.split("."):
        v = v.get(part) if isinstance(v, dict) else None
    return v


def column(rows, key):
    """Finite values of `key` over the error-free rows, as a float array."""
    v = [value(r, key) for r in rows if "error" not in r]
    return np.array([x for x in v if x is not None and np.isfinite(x)], float)


def _spread(v):
    """p5 / median / p95 of `v` to 3 significant figures."""
    return " / ".join(f"{x:.3g}" for x in np.percentile(v, [5, 50, 95]))


def table(real, sim):
    """Markdown table of every `METRICS` quantity with at least 5 values on each side: percentiles of both
    and their two-sample Kolmogorov-Smirnov distance."""
    lines = [
        "| Quantity | Unit | Real p5 / median / p95 | Simulated p5 / median / p95 | KS distance |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, name, unit in METRICS:
        a, b = column(real, key), column(sim, key)
        if len(a) < 5 or len(b) < 5:
            continue
        ks = stats.ks_2samp(a, b).statistic
        lines.append(f"| {name} | {unit} | {_spread(a)} | {_spread(b)} | {ks:.2f} |")
    return "\n".join(lines)


def run(real_path, sim_path, out_path, title="Realism report"):
    """Write the realism report for real and simulated statistics files to `out_path` and return its text.

    Only real frames of the simulated frame sizes are compared, so like is compared with like.
    """
    real, sim = read_json(real_path), read_json(sim_path)
    shapes = {tuple(r["shape"]) for r in sim if "error" not in r}
    real = [r for r in real if "error" not in r and tuple(r["shape"]) in shapes]
    n_r = sum("error" not in r for r in real)
    n_s = sum("error" not in r for r in sim)
    text = f"""# {title}

Real frames: {n_r}. Simulated frames: {n_s}. Both measured with the same code
(`skytwin-calibrate measure`).

The KS distance is the largest gap between the two cumulative
distributions: 0 means identical, 1 means no overlap. Below about 0.2 the
distributions are close; above 0.4 they clearly differ.

{table(real, sim)}
"""
    Path(out_path).write_text(text)
    return text
