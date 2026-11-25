"""Read an instrument's headers without being told its keyword names.

`infer` builds a keyword map (see `keywords.py`) from sample headers. Each
entry is chosen by scoring every card on its name and on its comment, so an
unfamiliar header still maps if it says what a value is ("[C] CCD
Temperature", "ECEF Position at Exp start, X-component"). A card only
counts if its value parses and varies sensibly across the sample.

`style` writes the profile's `header` block so simulated frames carry the
same keywords as the real ones: values that never change are copied,
values Skytwin can compute (time, exposure, temperature, pointing,
sub-satellite point, Earth-limb angle, binning) become templates, and the
rest are left out.
"""

import re
from collections import Counter

import numpy as np

from .util import number as _num

STRUCTURAL = re.compile(
    r"^(SIMPLE|BITPIX|NAXIS\d*|EXTEND|BZERO|BSCALE|BLANK|COMMENT|HISTORY|CHECKSUM|DATASUM|END|"
    r"CTYPE\d|CRPIX\d|CRVAL\d|CDELT\d|CUNIT\d|CROTA\d|CD\d_\d|PC\d_\d|WCSAXES|WCSNAME|EQUINOX|RADESYS|"
    r"LONPOLE|LATPOLE|[AB]_\w+|[AB]P_\w+|PV\d_\d+|PROJP\d)$"
)


def _sexa(v):
    """True for a sexagesimal string such as '00:49:39.0' or '00 49 39.0'."""
    return isinstance(v, str) and re.fullmatch(r"\s*[+-]?\d{1,3}[: ]\d{1,2}[: ]\d{1,2}(\.\d*)?\s*", v) is not None


def _cards(headers, min_present=0.5):
    """Map keyword -> (values, comment) for keywords in at least `min_present` of the sample headers.

    Values come only from the headers that have the keyword; the comment from the first that does.
    """
    out = {}
    for h in headers:
        for k in h:
            if not k or STRUCTURAL.match(k):
                continue
            out.setdefault(k, ([], h.comments[k] or ""))[0].append(h[k])
    return {k: v for k, v in out.items() if len(v[0]) >= min_present * len(headers)}


SCALARS = {
    "exposure_s": dict(
        names=r"^(EXPTIME|EXPOSURE|AEXPTIME|ITIME|EXP_TIME|EXPOSURE_TIME|TEXPOSE|ELAPTIME)$",
        comment=[r"expos"],
        ok=lambda v: 0 < v < 1e5,
        prefer=r"actual",
    ),
    "temperature_c": dict(
        names=r"^(CCD-TEMP|CCDTEMP|CCD_TEMP|DET-TEMP|DETTEMP|TEMPERAT|SET-TEMP|CCDTEMP1)$",
        comment=[r"(ccd|detector|sensor|focal).*temp|temp.*(ccd|detector|sensor)", r"\[\s*c\s*\]|celsius|deg\s*c\b"],
        ok=lambda v: -150 < v < 60,
    ),
    "temperature_k": dict(
        names=r"^(TEMP_CCD|CCDTEMPK|DETTEMPK)$",
        comment=[r"(ccd|detector|sensor|focal).*temp|temp.*(ccd|detector|sensor)", r"\[\s*k\s*\]|kelvin"],
        ok=lambda v: 120 < v < 335,
    ),
    "subpoint_lat_deg": dict(
        names=r"^(GEO_LAT|SUB_LAT|SUBLAT|SAT_LAT|LATITUDE|SC_LAT)$", comment=[r"latitude"], ok=lambda v: -90 <= v <= 90
    ),
    "subpoint_lon_deg": dict(
        names=r"^(GEO_LONG|GEO_LON|SUB_LON|SUBLON|SAT_LON|LONGITUD|SC_LON)$",
        comment=[r"longitude"],
        ok=lambda v: -360 <= v <= 360,
    ),
    "earth_limb_deg": dict(
        names=r"^(EARTHLIM|ELIMB|LIMB_ANG)$",
        comment=[r"earth.*limb|limb.*angle"],
        ok=lambda v: -90 <= v <= 180,
        prefer=r"used|min",
    ),
    "trail_rate_arcsec_per_s": dict(
        names=r"^(SLEWRATE|TRACKRAT)$",
        comment=[r"(slew|track|angular).*(vel|rate)", r"arcsec"],
        ok=lambda v: 0 <= v < 1e5,
    ),
    "gain_e_per_adu": dict(
        names=r"^(GAIN|EGAIN|CCDGAIN|GAIN1)$", comment=[r"e(lectrons?)?\s*-?\s*/\s*adu|gain"], ok=lambda v: 0 < v < 1000
    ),
    "read_noise_e": dict(
        names=r"^(RDNOISE|READNOIS|RON|READNOISE|RDNOISE1)$",
        comment=[r"read\s*-?out|read noise"],
        ok=lambda v: 0 < v < 1000,
    ),
    "saturation_adu": dict(names=r"^(SATURATE|SATLEVEL|DATAMAX)$", comment=[r"saturat"], ok=lambda v: v > 0),
    "binning_x": dict(
        names=r"^(XBINNING|CCDBIN1|BINX|XBIN|HBIN)$",
        comment=[r"bin.*\b(x|horizontal|col)"],
        ok=lambda v: 1 <= v <= 64 and v == int(v),
    ),
    "binning_y": dict(
        names=r"^(YBINNING|CCDBIN2|BINY|YBIN|VBIN)$",
        comment=[r"bin.*\b(y|vertical|row)"],
        ok=lambda v: 1 <= v <= 64 and v == int(v),
    ),
    "pixel_scale_arcsec": dict(
        names=r"^(PIXSCALE|SECPIX|PIXSCAL1|SCALE)$", comment=[r"arcsec.*pix|\"/pix"], ok=lambda v: 0 < v < 3600
    ),
    "airmass": dict(names=r"^(AIRMASS|SECZ|AMSTART|AIRMASS1)$", comment=[r"air\s*mass"], ok=lambda v: 1 <= v < 40),
    "altitude_deg": dict(
        names=r"^(ALT_PNT|ALTITUDE|TELALT|CENTALT|OBJCTALT|ELEVATIO|ALT-OBJ)$",
        comment=[r"altitude|elevation"],
        ok=lambda v: 0 < v <= 90,
    ),
    "pixel_size_um": dict(
        names=r"^(XPIXSZ|PIXSIZE|PIXSIZE1|PIXSZ|XPIXELSZ|PIXELSZ)$",
        comment=[r"pixel", r"micron|\bum\b|µm"],
        ok=lambda v: 1 < v < 200,
    ),
}
"""Scalar quantities: name pattern, comment patterns (all must match), a sanity check on the values, and an
optional word that marks the preferred card among several (e.g. actual over requested exposure)."""


def _score_scalar(k, vals, com, spec):
    """Score of card `k` for a `SCALARS` entry: 0 unless 90% of its values parse and pass the check.

    Name match +3, all comment patterns +2 plus one per pattern, preference word +1.
    """
    nums = [_num(v) for v in vals]
    good = [n for n in nums if n is not None and spec["ok"](n)]
    if len(good) < 0.9 * len(vals):
        return 0
    c = com.lower()
    s = 0
    if re.match(spec["names"], k):
        s += 3
    if all(re.search(p, c) for p in spec["comment"]):
        s += 2 + len(spec["comment"])
    if s and spec.get("prefer") and re.search(spec["prefer"], c):
        s += 1
    return s


TIME_NAMES = r"^(DATE-OBS|DATE_OBS|DATEOBS|OBS-DATE|TIME-OBS|A_EXP_S|UTCSTART|DATE-BEG|T_START)$"


def _iso(v):
    """True for an ISO 8601 date-time string."""
    return isinstance(v, str) and re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d*)?Z?\s*", v) is not None


def _score_time(k, vals, com):
    """Score of card `k` as the exposure start time: all values ISO 8601; comments naming an end or a
    request count against it."""
    if not all(_iso(v) for v in vals):
        return 0
    c = com.lower()
    s = 3 if re.match(TIME_NAMES, k) else 0
    if re.search(r"start|begin|observation|obs", c):
        s += 2
    if re.search(r"request|end|stop|mid", c):
        s -= 2
    return s


def _vectors(cards):
    """Families of three cards that hold a position (km) or velocity (km/s).

    Names differ only in a trailing 1/2/3 or X/Y/Z, comments say what they are, and the vector's median
    magnitude is plausible for an Earth orbit. Each candidate is a dict with its `kind`, `keys`, reference
    `frame` (gcrs, itrs, teme or None) and `when` in the exposure it was taken (start, mid, end, unknown).
    """
    groups = {}
    for k in cards:
        m = re.match(r"^(.*?)[_-]?([123XYZ])$", k)
        if m:
            groups.setdefault(m.group(1), {})[m.group(2)] = k
    out = []
    for stem, mem in groups.items():
        keys = [mem.get(a) or mem.get(b) for a, b in (("1", "X"), ("2", "Y"), ("3", "Z"))]
        if None in keys:
            continue
        vals = [cards[k][0] for k in keys]
        if len({len(c) for c in vals}) != 1:
            continue
        nums = [[_num(v) for v in col] for col in vals]
        if any(n is None for col in nums for n in col):
            continue
        com = " ".join(cards[k][1] for k in keys).lower()
        mag = np.median(np.linalg.norm(np.array(nums, float), axis=0))
        kind = None
        if re.search(r"pos", com + stem.lower()) and 6000 < mag < 5e5:
            kind = "position_km"
        elif re.search(r"vel", com + stem.lower()) and 0.5 < mag < 12:
            kind = "velocity_km_s"
        if kind is None:
            continue
        words = com + " " + stem.lower()
        frame = None
        if re.search(r"ecef|itrf|itrs|earth[- ]fixed|\becr", words):
            frame = "itrs"
        if re.search(r"j2000|gcrs|\beci|icrf|inertial|eme2000", words):
            frame = "gcrs"
        if re.search(r"teme", words):
            frame = "teme"
        when = (
            "start"
            if re.search(r"start|begin", com)
            else "mid"
            if re.search(r"mid", com)
            else "end"
            if re.search(r"\bend\b", com)
            else "unknown"
        )
        out.append(dict(kind=kind, keys=keys, frame=frame, when=when, stem=stem))
    return out


def _pointing(cards):
    """Best RA and Dec cards as {axis: (score, keyword, sexagesimal)}: 'actual' beats 'commanded'."""
    best = {}
    for k, (vals, com) in cards.items():
        c = com.lower()
        for axis, names, words in (
            ("ra", r"^(RA|OBJCTRA|RA_OBJ|RA_PNT|TELRA|RA-OBS|PNT_RA|RA_NOM)$", r"right ascension|\bra\b"),
            ("dec", r"^(DEC|OBJCTDEC|DEC_OBJ|DEC_PNT|TELDEC|DEC-OBS|PNT_DEC|DEC_NOM)$", r"declination|\bdec\b"),
        ):
            sexa = all(_sexa(v) for v in vals)
            nums = [_num(v) for v in vals]
            numeric = all(n is not None for n in nums)
            if not (sexa or numeric):
                continue
            s = (3 if re.match(names, k) else 0) + (2 if re.search(words, c) else 0)
            if not s:
                continue
            s += 1 if re.search(r"actual|measured|estimated", c) else 0
            s -= 2 if re.search(r"command|request|target", c) else 0
            if s > best.get(axis, (0,))[0]:
                best[axis] = (s, k, sexa)
    return best


def _roll(cards):
    """The card most likely to hold the roll or position angle, or None; rates and commanded values lose."""
    best = (0, None)
    for k, (vals, com) in cards.items():
        c = com.lower()
        nums = [_num(v) for v in vals]
        if any(n is None for n in nums) or not re.search(r"roll|position angle|\bpa\b", c + " " + k.lower()):
            continue
        s = 2 + (1 if re.search(r"actual|measured", c) else 0) - (2 if re.search(r"command|request|vel|rate", c) else 0)
        if s > best[0]:
            best = (s, k)
    return best[1]


def infer(headers):
    """A keyword map for sample `headers`, and findings [(entry, keyword, why)].

    The state vector is a position and velocity pair in one frame, preferring cards taken at the exposure
    start (the map's time is the start) and then GCRS. A roll card is marked `unverified` until a plate
    solve fixes its offset and sign.
    """
    cards = _cards(headers)
    km, notes = {}, []
    for name, spec in SCALARS.items():
        scored = sorted(((_score_scalar(k, v, c, spec), k) for k, (v, c) in cards.items()), reverse=True)
        if scored and scored[0][0] >= 3:
            km[name] = scored[0][1]
            notes.append((name, scored[0][1], cards[scored[0][1]][1]))
    if "temperature_c" in km:
        km.pop("temperature_k", None)
        notes = [n for n in notes if n[0] != "temperature_k"]
    scored = sorted(((_score_time(k, v, c), k) for k, (v, c) in cards.items()), reverse=True)
    if scored and scored[0][0] >= 3:
        km["time_utc"] = scored[0][1]
        notes.append(("time_utc", scored[0][1], cards[scored[0][1]][1]))
    vecs = _vectors(cards)
    rank = {"start": 0, "unknown": 1, "mid": 2, "end": 3}
    pref = {"gcrs": 0, "itrs": 1, "teme": 2, None: 9}
    pairs = []
    for p in (v for v in vecs if v["kind"] == "position_km"):
        for q in (v for v in vecs if v["kind"] == "velocity_km_s"):
            if p["frame"] == q["frame"] and p["frame"] is not None:
                pairs.append((rank[p["when"]] + rank[q["when"]], pref[p["frame"]], p, q))
    if pairs:
        _, _, p, q = min(pairs, key=lambda t: (t[0], t[1]))
        km["position_km"], km["velocity_km_s"], km["state_frame"] = p["keys"], q["keys"], p["frame"]
        notes.append(("state vector", ", ".join(p["keys"] + q["keys"]), f"{p['frame']}, at exposure {p['when']}"))
    pt = _pointing(cards)
    if "ra" in pt and "dec" in pt:
        km["ra"], km["dec"] = pt["ra"][1], pt["dec"][1]
        if pt["ra"][2]:
            km["radec_sexagesimal"] = True
        notes.append(("pointing", f"{km['ra']}, {km['dec']}", "sexagesimal" if pt["ra"][2] else "degrees"))
        r = _roll(cards)
        if r:
            km["roll"] = {"keyword": r, "offset_deg": 0.0, "sign": 1, "unverified": True}
            notes.append(("roll", r, "convention not checked: plate-solve frames to fix offset and sign"))
    return km, notes


TEMPLATES = {
    "time_utc": "{date_obs}",
    "exposure_s": "{exposure_s}",
    "temperature_c": "{ccd_temp_c}",
    "temperature_k": "{ccd_temp_k}",
    "subpoint_lat_deg": "{sub_lat_deg}",
    "subpoint_lon_deg": "{sub_lon_deg}",
    "earth_limb_deg": "{earth_limb_deg}",
    "binning_x": "{binning_x}",
    "binning_y": "{binning_y}",
}


def style(headers, km):
    """The profile's `header` block, imitating the real headers.

    Cards with a `TEMPLATES` entry become templates (kept as strings where the instrument writes numbers as
    strings); cards with one value in 95% of the headers are copied; the rest are left out.
    """
    cards = _cards(headers)
    has_wcs = all("CTYPE1" in h and "CTYPE2" in h for h in headers)
    template_of = {}
    for name, t in TEMPLATES.items():
        k = km.get(name)
        if isinstance(k, str):
            template_of[k] = t
    if km.get("ra"):
        sexa = km.get("radec_sexagesimal")
        sep = "colon" if any(":" in str(v) for v in cards.get(km["ra"], ([], ""))[0]) else "space"
        template_of[km["ra"]] = f"{{ra_hms:{sep}}}" if sexa else "{ra_deg}"
        template_of[km["dec"]] = f"{{dec_dms:{sep}}}" if sexa else "{dec_deg}"
    out_cards = []
    for k, (vals, com) in cards.items():
        if len(k) > 8 or not re.fullmatch(r"[A-Z0-9_-]+", k):
            continue
        strs = [isinstance(v, str) for v in vals]
        if k in template_of:
            t = template_of[k]
            if all(strs) and not t.startswith(("{date_obs", "{ra_hms", "{dec_dms")):
                t = t[:-1] + ":str}"
            out_cards.append({"key": k, "value": t, "comment": com})
            continue
        if any(v is None or (isinstance(v, str) and v == "") for v in vals):
            continue
        common, n = Counter(map(repr, vals)).most_common(1)[0]
        if n >= 0.95 * len(vals):
            v = next(v for v in vals if repr(v) == common)
            if isinstance(v, (bool, np.bool_)):
                v = bool(v)
            elif isinstance(v, (int, np.integer)):
                v = int(v)
            elif isinstance(v, (float, np.floating)):
                v = float(v)
            else:
                v = str(v)
            if isinstance(v, str) and re.fullmatch(r"\{.*\}", v):
                continue
            out_cards.append({"key": k, "value": v, "comment": com})
    block = {"wcs": has_wcs, "skytwin_keywords": False, "cards": out_cards}
    if km.get("roll") and not km["roll"].get("unverified"):
        r = km["roll"]
        block["roll"] = {"keyword": r["keyword"], "offset_deg": r.get("offset_deg", 0.0), "sign": r.get("sign", 1)}
        block["cards"] = [c for c in out_cards if c["key"] != r["keyword"]]
    return block
