"""Write FITS frames (real DUST conversions or Skytwin frames from a DUST
profile) as 8-bit PNGs the way the DUST dataset made its Images/:

    png = floor(255 * DN / P99.9), clipped to 0..255

per frame: black at 0 DN, white at the frame's 99.9th percentile. Measured
on the dataset itself (12 of 14 sampled sessions to 0.3 grey levels RMS;
the others hold PNGs that do not match their HDF5 files). Greyscale saved
as RGB, as in DUST.

    python dust_png.py frame.fits [more.fits ...] -o out_dir
"""

import argparse
import os

import numpy as np
from astropy.io import fits
from PIL import Image


def to_png(d):
    """Frame `d` as an RGB image: floor(255 × DN / P99.9), clipped to 0..255 (all black when P99.9 <= 0)."""
    d = np.asarray(d, dtype=np.float64)
    top = np.percentile(d, 99.9)
    q = np.floor(255.0 * d / top) if top > 0 else np.zeros_like(d)
    g = np.clip(q, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([g, g, g]), "RGB")


def main():
    """Convert each FITS file given to a PNG of the same name in the output directory."""
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in a.files:
        name = os.path.splitext(os.path.basename(f))[0] + ".png"
        to_png(fits.getdata(f)).save(os.path.join(a.out, name))


if __name__ == "__main__":
    main()
