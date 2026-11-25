"""Header keyword map: where an instrument keeps the values calibration needs.

A map is a JSON object. Every entry is optional:

    {
      "exposure_s": "EXPOSURE",
      "temperature_c": "CCD-TEMP",
      "time_utc": "DATE-OBS",
      "subpoint_lat_deg": "SUB_LAT",
      "subpoint_lon_deg": "SUB_LON",
      "earth_limb_deg": "LIMB_ANG",
      "position_km": ["SC_X", "SC_Y", "SC_Z"],
      "velocity_km_s": ["SC_VX", "SC_VY", "SC_VZ"],
      "state_frame": "itrs",
      "ra": "RA", "dec": "DEC", "radec_sexagesimal": true,
      "roll": {"keyword": "ROLLANG", "offset_deg": 90.0, "sign": -1},
      "trail_rate_arcsec_per_s": "TRACKRAT",
      "skip_left_columns": 16,
      "saturation_adu": 60000,
      "pixel_size_um": "XPIXSZ",
      "airmass": "AIRMASS"
    }

`temperature_k` may stand in for `temperature_c`, and `altitude_deg` (the
telescope's altitude) for `airmass`, through Kasten and Young (1989). `skytwin-calibrate learn`
writes a map like this from the headers themselves.

`pixel_size_um` is the pixel size after binning unless the card's comment
says it is unbinned; `learn` divides it by the binning for
`camera.pixel_pitch_um`.

`roll` converts the instrument's roll keyword to Skytwin's convention:
roll = offset_deg + sign * value.
"""

import math

from .util import header_number, read_json


def kasten_young(alt_deg):
    """Airmass at altitude `alt_deg` (Kasten and Young 1989)."""
    z = 90.0 - alt_deg
    return 1.0 / (math.cos(math.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)


SCALARS = [
    "exposure_s",
    "temperature_c",
    "subpoint_lat_deg",
    "subpoint_lon_deg",
    "earth_limb_deg",
    "trail_rate_arcsec_per_s",
    "airmass",
    "altitude_deg",
]


def load(path):
    """The keyword map at `path`, or an empty map when `path` is empty."""
    return read_json(path) if path else {}


def frame_meta(h, km):
    """Values from header `h` through keyword map `km`, None where absent.

    Temperatures are °C (converted from `temperature_k` when only that is mapped); the state frame defaults
    to GCRS. `pointing` holds RA, Dec and roll in degrees, with roll None while the map's roll convention
    is unverified.
    """
    out = {name: header_number(h, km[name]) if name in km else None for name in SCALARS}
    if out["temperature_c"] is None and "temperature_k" in km:
        k = header_number(h, km["temperature_k"])
        out["temperature_c"] = k - 273.15 if k is not None else None
    if out["airmass"] is None and out["altitude_deg"]:
        out["airmass"] = kasten_young(out["altitude_deg"])
    out["time_utc"] = str(h.get(km["time_utc"])) if "time_utc" in km and km["time_utc"] in h else None
    for name in ["position_km", "velocity_km_s"]:
        if name in km:
            vals = [header_number(h, k) for k in km[name]]
            out[name] = vals if all(v is not None for v in vals) else None
        else:
            out[name] = None
    out["state_frame"] = km.get("state_frame", "gcrs")
    out["pointing"] = None
    if "ra" in km and "dec" in km and km["ra"] in h and km["dec"] in h:
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord

            if km.get("radec_sexagesimal", False):
                ra, dec = str(h[km["ra"]]).replace(" ", ":"), str(h[km["dec"]]).replace(" ", ":")
                c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
            else:
                c = SkyCoord(float(h[km["ra"]]), float(h[km["dec"]]), unit="deg")
            roll = None
            if "roll" in km and not km["roll"].get("unverified"):
                r = km["roll"]
                v = header_number(h, r["keyword"])
                if v is not None:
                    roll = (r.get("offset_deg", 0.0) + r.get("sign", 1.0) * v) % 360.0
            out["pointing"] = {"ra_deg": float(c.ra.deg), "dec_deg": float(c.dec.deg), "roll_deg": roll}
        except Exception:
            out["pointing"] = None
    return out
