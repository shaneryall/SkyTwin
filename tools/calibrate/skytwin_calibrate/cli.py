"""skytwin-calibrate: measure an archive, build and refine a profile, report."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from . import contract, keywords, learn, measure, profile, psf, refine, report, score
from .util import read_json, write_json, write_rows


def files_from(args: argparse.Namespace) -> list[str]:
    """Frame paths from `--list` (one per line) or else from the glob patterns in `frames`, each sorted."""
    if args.list:
        return [line for line in Path(args.list).read_text().splitlines() if line.strip()]
    out = []
    for g in args.frames:
        out += sorted(glob.glob(g, recursive=True))
    return out


def main(argv: Sequence[str] | None = None) -> int | None:
    """Run the `skytwin-calibrate` command line on `argv` (default: `sys.argv[1:]`).

    Returns 1 when `check` finds problems, else None.
    """
    p = argparse.ArgumentParser(prog="skytwin-calibrate")
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("learn", help="learn a whole profile from sample frames, in one step")
    lp.add_argument("inputs", nargs="+", help="directories, FITS files, glob patterns or .txt lists")
    lp.add_argument(
        "-o", "--out", required=True, help="profile to write (a .learn.md report and .keywords.json go beside it)"
    )
    lp.add_argument("--keywords", help="header keyword map (default: worked out from the headers)")
    lp.add_argument("--skytwin", help="skytwin binary: refine the profile and write a realism report")
    lp.add_argument(
        "--ucac4",
        help="UCAC4 directory: plate solves for scale, parity and header pointing errors, and real-sky twins",
    )
    lp.add_argument(
        "--catalog",
        help="star catalogue for matching, photometry and real-sky twins instead of --ucac4: a "
        "UCAC4 directory or a CSV (Gaia extract, Skytwin column names). Use one with good photometry for "
        "the stars the camera sees: UCAC4's own magnitudes saturate brighter than about 8",
    )
    lp.add_argument("--catalog-name", help="name recorded for the catalogue in the profile (default: the file name)")
    lp.add_argument("--catalog-source", help="how to get or rebuild the catalogue, recorded in the profile")
    lp.add_argument("--sensor-name", help="sensor name recorded in the profile (default: INSTRUME/TELESCOP)")
    lp.add_argument("--platform", help="spacecraft or mission recorded in the profile")
    lp.add_argument("--band-nm", nargs=2, type=float, metavar=("FROM", "TO"), help="passband recorded in the profile")
    lp.add_argument(
        "--ephemeris",
        help="spacecraft states for frames whose headers carry none: a CSV (time_utc, x_km, y_km, z_km, vx_km_s, "
        "vy_km_s, vz_km_s; geocentric GCRS) or a JPL Horizons vector table (geocentric, in km and km/s or "
        "in au and au/day)",
    )
    lp.add_argument(
        "--extinction",
        type=float,
        help="ground frames: extinction, mag/airmass, to move the zeropoint above the atmosphere when the frames "
        "do not span enough airmass to fit it (default 0.2)",
    )
    lp.add_argument(
        "--pixel-pitch-um",
        type=float,
        help="unbinned pixel size, um, when the headers do not give it (radiation rates per cm² need it)",
    )
    lp.add_argument(
        "--calibration",
        nargs="+",
        help="bias, dark and flat frames of the same camera (directories, files, glob patterns or .txt lists): "
        "gain, read noise, dark current, cosmic rays and dust shadows from them",
    )
    lp.add_argument("--fayde", help="fayde binary for the plate solves (default: on the PATH)")
    lp.add_argument("--max-frames", type=int, default=1500, help="frames to sample from the inputs (default 1500)")
    lp.add_argument("--iterations", type=int, default=2, help="refinement passes (default 2)")
    lp.add_argument(
        "--twins",
        type=int,
        default=200,
        help="simulated twins per refinement pass (default 200); the realism report uses up to 300",
    )
    lp.add_argument("--work", help="working directory (default: beside the profile)")

    k = sub.add_parser("check", help="check a local star catalogue against what a profile was calibrated with")
    k.add_argument("profile", help="profile (JSON)")
    k.add_argument("--catalog", required=True, help="the catalogue you will give scenes (CSV file or UCAC4 directory)")

    d = sub.add_parser("describe", help="what a profile is, what it needs, and a scene to start from")
    d.add_argument("profile", help="profile (JSON)")

    m = sub.add_parser("measure", help="per-frame statistics of real or simulated frames")
    m.add_argument("frames", nargs="*", help="FITS files or glob patterns")
    m.add_argument("--list", help="file with one FITS path per line")
    m.add_argument("--keywords", help="header keyword map (JSON)")
    m.add_argument("-o", "--out", required=True, help="statistics file to write (JSON)")

    s = sub.add_parser("psf", help="stack isolated stars and fit a multi-Gaussian PSF")
    s.add_argument("frames", nargs="*", help="FITS files or glob patterns")
    s.add_argument("--list", help="file with one FITS path per line")
    s.add_argument(
        "--skip-left-columns",
        type=int,
        default=0,
        help="columns at the left edge to leave out, such as overscan (default 0)",
    )
    s.add_argument("-o", "--out", required=True, help="fit result to write (JSON)")

    b = sub.add_parser("profile", help="build profile blocks from statistics")
    b.add_argument("stats", help="statistics of the real frames, from measure")
    b.add_argument("--base", required=True, help="base profile (camera basics and hand-measured blocks)")
    b.add_argument(
        "--max-background",
        type=float,
        help="frames above this background (ADU) count as glare and are left out "
        "(default: 1000, or twice the median frame background if higher)",
    )
    b.add_argument("-o", "--out", required=True, help="profile to write")

    r = sub.add_parser("refine", help="simulate, measure and correct a profile in place")
    r.add_argument("profile", help="profile to correct in place")
    r.add_argument("--real", required=True, help="statistics of the real frames")
    r.add_argument("--keywords", help="header keyword map for simulated frames (default: none)")
    r.add_argument("--skytwin", default="skytwin", help="path to the skytwin binary")
    r.add_argument("--work", default="refine-work", help="working directory (default refine-work)")
    r.add_argument("--iterations", type=int, default=2, help="refinement passes (default 2)")
    r.add_argument("--twins", type=int, default=200, help="simulated twins per pass (default 200)")

    w = sub.add_parser("twins", help="simulate twins of real frames and measure them")
    w.add_argument("profile", help="profile (JSON)")
    w.add_argument("--real", required=True, help="statistics of the real frames")
    w.add_argument("--keywords", help="header keyword map for simulated frames")
    w.add_argument("--skytwin", default="skytwin", help="path to the skytwin binary")
    w.add_argument("--work", default="twins-work", help="working directory (default twins-work)")
    w.add_argument("-n", type=int, default=300, help="twins to simulate (default 300)")
    w.add_argument("--ucac4", help="UCAC4 directory: twins get catalogue stars at the real pointing")
    w.add_argument("--mag-limit", type=float, default=15.5, help="faintest catalogue star drawn (default 15.5)")
    w.add_argument("-o", "--out", required=True, help="statistics of the twins to write (JSON)")

    t = sub.add_parser("report", help="compare real and simulated statistics")
    t.add_argument("--real", required=True, help="statistics of the real frames")
    t.add_argument("--sim", required=True, help="statistics of the simulated frames")
    t.add_argument("--title", default="Realism report", help="report title (default: Realism report)")
    t.add_argument("-o", "--out", required=True, help="report to write (Markdown)")

    c = sub.add_parser("score", help="score a detector's source list against a truth file")
    c.add_argument("--truth", required=True, help="truth file of the frame the detector ran on")
    c.add_argument(
        "--sources",
        required=True,
        help="FITS table, CSV or JSON with X/Y, X_IMAGE/Y_IMAGE or xcentroid/ycentroid columns (any case)",
    )
    c.add_argument(
        "--origin",
        type=int,
        choices=[0, 1],
        help="pixel origin of the source list (default: 1 for X_IMAGE and for a FITS table's X, 0 otherwise)",
    )
    c.add_argument("--radius", type=float, default=2.0, help="match radius, pixels (default 2)")
    c.add_argument("-o", "--out", help="write the result as JSON (and .md beside it)")

    a = p.parse_args(argv)
    if a.cmd == "learn":
        km = keywords.load(a.keywords) if a.keywords else None
        sensor = {k: v for k, v in (("name", a.sensor_name), ("platform", a.platform), ("band_nm", a.band_nm)) if v}
        learn.learn(
            a.inputs,
            a.out,
            km=km,
            skytwin=a.skytwin,
            ucac4=a.ucac4,
            catalog=a.catalog,
            max_frames=a.max_frames,
            iterations=a.iterations,
            twins=a.twins,
            work=a.work,
            fayde=a.fayde,
            catalog_name=a.catalog_name,
            catalog_source=a.catalog_source,
            sensor=sensor,
            calibration=a.calibration,
            pixel_pitch_um=a.pixel_pitch_um,
            extinction=a.extinction,
            ephemeris=a.ephemeris,
        )
        base = os.path.splitext(a.out)[0]
        print(f"profile -> {a.out}\nwhat was measured -> {base}.learn.md\nkeyword map -> {base}.keywords.json")
    elif a.cmd == "check":
        probs = contract.check(a.profile, a.catalog)
        if probs:
            print("\n".join(f"problem: {x}" for x in probs))
            return 1
        print(f"{a.catalog}: matches what {a.profile} was calibrated with")
    elif a.cmd == "describe":
        print(contract.describe(a.profile))
    elif a.cmd == "measure":
        fs = files_from(a)
        rows = measure.run(fs, keywords.load(a.keywords), a.out)
        print(f"measured {sum('error' not in r for r in rows)} of {len(fs)} frames -> {a.out}")
    elif a.cmd == "psf":
        res = psf.run(files_from(a), a.skip_left_columns, a.out)
        print(
            f"{res['stars']} stars: core FWHM {res['core_fwhm_px']:.3f} px, wings "
            + ", ".join(f"{w['fraction']:.3f} @ {w['fwhm_px']:.2f} px" for w in res["wings"])
        )
    elif a.cmd == "profile":
        profile.run(a.stats, a.base, a.out, a.max_background)
        print(f"profile -> {a.out}")
    elif a.cmd == "refine":
        real = read_json(a.real)
        km = keywords.load(a.keywords)
        for i in range(a.iterations):
            rep = refine.iterate(a.profile, real, km, a.skytwin, f"{a.work}/it{i}", n_twins=a.twins, seed=i + 1)
            print(f"iteration {i + 1}: {json.dumps(rep)}")
    elif a.cmd == "twins":
        cat = {"ucac4": a.ucac4, "mag_limit": a.mag_limit} if a.ucac4 else None
        km = keywords.load(a.keywords)
        rows = refine.twins(a.profile, read_json(a.real), km, a.skytwin, a.work, a.n, catalog=cat)
        write_rows(rows, a.out)
        print(f"{len(rows)} twins measured -> {a.out}")
    elif a.cmd == "score":
        try:
            xy, _ = score.load_sources(a.sources, a.origin)
        except ValueError as e:
            print(f"skytwin-calibrate score: {e}", file=sys.stderr)
            return 2
        res = score.score(read_json(a.truth), xy, a.radius)
        md = score.markdown(res)
        if a.out:
            write_json(res, a.out, indent=1)
            Path(os.path.splitext(a.out)[0] + ".md").write_text(md)
        print(md)
    elif a.cmd == "report":
        print(report.run(a.real, a.sim, a.out, a.title))
    return None


if __name__ == "__main__":
    raise SystemExit(main())
