"""Convert the DUST dataset (Zenodo 20255672, CASSIOPE FAI near-infrared) to
FITS frames that `skytwin-calibrate learn` reads.

    python dust_to_fits.py /path/to/DUST out_dir

Per frame:

- Pixels: the HDF5 "FAI Image" flipped vertically and cropped to rows
  12..267, which is exactly the 256x256 PNG the dataset's WCS, YOLO boxes
  and MOT tracks refer to. Values are converted from Rayleighs back to DN
  (image x exposure / R-per-DN-per-s), float32.
- WCS: the dataset's Astrometry.net solution (TAN-SIP), copied in.
- Pointing: RA/Dec of the image centre and roll (position angle of the
  image +y axis, east of north) from that WCS, as ATT_RA / ATT_DEC /
  ATT_ROLL.
- Spacecraft: the HDF5 "GEO" position (ECEF; labelled km but in metres)
  turned into GCRS km, with velocity from neighbouring frames of the same
  session.
"""

import glob
import os
import sys

import h5py
import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

ROW0, ROWS = 12, 256


def roll_of(w, cx, cy):
    """Position angle (deg, east of north) of the image +y axis at the centre."""
    c0 = w.pixel_to_world(cx, cy)
    c1 = w.pixel_to_world(cx, cy + 1.0)
    return float(c0.position_angle(c1).deg) % 360.0


def gcrs_km(xyz_m, t):
    """ITRS (ECEF) positions in metres at times `t` as GCRS positions in km, shape (n, 3)."""
    itrs = ITRS(CartesianRepresentation(*(np.asarray(xyz_m) * u.m)), obstime=t)
    g = itrs.transform_to(GCRS(obstime=t))
    return g.cartesian.xyz.to_value(u.km).T


def session(sdir, out):
    """Convert one session directory (H5/ and WCS/) into FITS frames under `out`; returns the frame count.

    Velocities are the time gradient of the GCRS positions over the session (needs 3 frames, else the
    velocity cards are left out).
    """
    h5s = sorted(glob.glob(os.path.join(sdir, "H5", "*.h5")))
    recs = []
    for p in h5s:
        name = os.path.basename(p)[:-3]
        wcs_path = os.path.join(sdir, "WCS", name + ".fits")
        with h5py.File(p, "r") as f:
            a = dict(f.attrs)
            img = np.asarray(f["FAI Image"], dtype=np.float64)
        recs.append((name, a, img, wcs_path if os.path.exists(wcs_path) else None))
    if not recs:
        return 0
    t = Time([str(r[1]["UTC"]) for r in recs], scale="utc")
    xyz = np.array(
        [
            [r[1]["S/C GEO X Position (km)"], r[1]["S/C GEO Y Position (km)"], r[1]["S/C GEO Z Position (km)"]]
            for r in recs
        ]
    ).T
    pos = gcrs_km(xyz, t)
    sec = (t - t[0]).to_value(u.s)
    vel = np.full_like(pos, np.nan)
    if len(recs) >= 3:
        for k in range(3):
            vel[:, k] = np.gradient(pos[:, k], sec)
    n = 0
    for i, (name, a, img, wcs_path) in enumerate(recs):
        exp = float(a["Exposure Time (s)"])
        dn = np.flipud(img)[ROW0 : ROW0 + ROWS] * exp / float(a["DN to Rayleigh Conversion Factor (R/DN/s)"])
        h = fits.Header()
        h["INSTRUME"] = ("FAI-NIR", "CASSIOPE Fast Auroral Imager, NIR channel")
        h["ORIGIN"] = "DUST (Zenodo 20255672), CC BY 4.0"
        h["DATE-OBS"] = (t[i].isot, "UTC")
        h["EXPTIME"] = (exp, "s")
        h["CCDTEMP"] = (float(a["CCD Temperature (°C)"]), "C")
        h["SUB_LAT"] = (float(a["S/C Latitude (°)"]), "deg")
        h["SUB_LON"] = (float(a["S/C Longitude (°)"]), "deg")
        h["SC_ALT"] = (float(a["S/C Altitude (km)"]), "km")
        h["ADMODE"] = (str(a["Attitude Determination Mode"]), "attitude determination mode")
        h["SC_YAW"], h["SC_PITCH"], h["SC_ROLL"] = (float(a["Yaw (°)"]), float(a["Pitch (°)"]), float(a["Roll (°)"]))
        for k, v in zip("XYZ", pos[i]):
            h[f"OBS_{k}"] = (float(v), "GCRS km")
        if np.all(np.isfinite(vel[i])):
            for k, v in zip("XYZ", vel[i]):
                h[f"OBS_V{k}"] = (float(v), "GCRS km/s")
        if wcs_path:
            wh = fits.getheader(wcs_path)
            w = WCS(wh)
            cx, cy = (ROWS - 1) / 2.0, (ROWS - 1) / 2.0
            c = w.pixel_to_world(cx, cy)
            h["ATT_RA"] = (float(c.ra.deg), "deg, image centre, from the WCS")
            h["ATT_DEC"] = (float(c.dec.deg), "deg, image centre, from the WCS")
            h["ATT_ROLL"] = (roll_of(w, cx, cy), "deg, PA of image +y, from the WCS")
            h.update(w.to_header(relax=True))
        d = os.path.join(
            out,
            os.path.basename(os.path.dirname(sdir.rstrip("/"))).replace(" ", ""),
            os.path.basename(sdir.rstrip("/")),
        )
        os.makedirs(d, exist_ok=True)
        fits.PrimaryHDU(dn.astype(np.float32), h).writeto(os.path.join(d, name + ".fits"), overwrite=True)
        n += 1
    return n


def main():
    """Convert every session under the DUST root given on the command line."""
    src, out = sys.argv[1], sys.argv[2]
    total = 0
    for s in sorted(glob.glob(os.path.join(src, "SET *", "*"))):
        k = session(s, out)
        total += k
        print(f"{s}: {k}", flush=True)
    print(total, "frames")


if __name__ == "__main__":
    main()
