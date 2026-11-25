"""Add the stars a Gaia extract lacks (Gaia DR3 is incomplete brighter than
G ~ 3) from the Yale Bright Star Catalogue (BSC5, VizieR V/50), so wide
fields keep their brightest stars.

    python add_bright_stars.py gaia.csv bsc5/catalog.gz out.csv

A BSC5 star with no Gaia source within 30" and 1.5 mag is appended, moved
from epoch 2000 to 2016 by its proper motion (the CSV epoch), with
approximate Gaia photometry from V and B-V:

    G     = V - 0.0291 - 0.0239 (B-V) - 0.2297 (B-V)^2     (Riello et al. 2021, Table C.2 form)
    BP-RP = 0.019 + 1.248 (B-V) + 0.089 (B-V)^2            (fitted to Gaia-BSC5 pairs, rough)

Rows added get source_id 0 - HR (negative ids mark them).
"""

import csv
import gzip
import sys

import numpy as np


def bsc5(path):
    """BSC5 stars as (HR, RA deg, Dec deg, V, B-V, pmRA, pmDec arcsec/yr) at J2000, from the fixed-width
    catalogue file. Entries without a J2000 position or a V magnitude (novae, extragalactic objects) are
    skipped; a missing B-V is taken as 0.6 and a missing proper motion as 0."""
    out = []
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\n").ljust(197)
            try:
                hr = int(line[0:4])
                ra = 15 * (int(line[75:77]) + int(line[77:79]) / 60 + float(line[79:83]) / 3600)
                de = int(line[84:86]) + int(line[86:88]) / 60 + int(line[88:90]) / 3600
                de = -de if line[83] == "-" else de
                v = float(line[102:107])
            except ValueError:
                continue
            try:
                bv = float(line[109:114])
            except ValueError:
                bv = 0.6
            try:
                pma, pmd = float(line[148:154]), float(line[154:160])
            except ValueError:
                pma = pmd = 0.0
            out.append((hr, ra, de, v, bv, pma, pmd))
    return out


def main():
    """Read the Gaia CSV, append the BSC5 stars it lacks, and write the result."""
    gaia, bsc, dst = sys.argv[1:4]
    with open(gaia, newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    ix = {k: head.index(k) for k in ("ra", "dec", "phot_g_mean_mag")}
    g = np.array(
        [
            [float(r[ix["ra"]]), float(r[ix["dec"]]), float(r[ix["phot_g_mean_mag"]])]
            for r in body
            if float(r[ix["phot_g_mean_mag"]]) < 8.5
        ]
    )
    gu = np.c_[
        np.cos(np.radians(g[:, 1])) * np.cos(np.radians(g[:, 0])),
        np.cos(np.radians(g[:, 1])) * np.sin(np.radians(g[:, 0])),
        np.sin(np.radians(g[:, 1])),
    ]
    added = []
    for hr, ra, de, v, bv, pma, pmd in bsc5(bsc):
        de16 = de + 16 * pmd / 3600
        ra16 = ra + 16 * pma / 3600 / max(np.cos(np.radians(de)), 1e-6)
        G = v - 0.0291 - 0.0239 * bv - 0.2297 * bv * bv
        u = np.array(
            [
                np.cos(np.radians(de16)) * np.cos(np.radians(ra16)),
                np.cos(np.radians(de16)) * np.sin(np.radians(ra16)),
                np.sin(np.radians(de16)),
            ]
        )
        sep = np.degrees(np.arccos(np.clip(gu @ u, -1, 1))) * 3600
        if np.any((sep < 30) & (np.abs(g[:, 2] - G) < 1.5)):
            continue
        r = [""] * len(head)
        r[head.index("source_id")] = str(-hr)
        r[ix["ra"]], r[ix["dec"]], r[ix["phot_g_mean_mag"]] = f"{ra16 % 360:.7f}", f"{de16:.7f}", f"{G:.3f}"
        if "pmra" in head:
            r[head.index("pmra")], r[head.index("pmdec")] = f"{pma * 1000:.2f}", f"{pmd * 1000:.2f}"
        if "bp_rp" in head:
            r[head.index("bp_rp")] = f"{0.019 + 1.248 * bv + 0.089 * bv * bv:.3f}"
        added.append(r)
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(body)
        w.writerows(added)
    print(
        f"{len(added)} BSC5 stars added to {len(body)} Gaia rows; brightest: "
        + ", ".join(
            f"HR {-int(r[0])} G {r[ix['phot_g_mean_mag']]}"
            for r in sorted(added, key=lambda r: float(r[ix["phot_g_mean_mag"]]))[:5]
        )
    )


if __name__ == "__main__":
    main()
