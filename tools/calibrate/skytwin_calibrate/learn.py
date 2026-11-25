"""Learn a whole instrument profile from sample frames, in one step.

    skytwin-calibrate learn /path/to/frames -o profile.json

What it does, each part reported in `profile.learn.md` beside the profile:

- the keyword map from the headers (`headers.infer`), unless one is given;
- the frames of the most common size and binning;
- camera basics from headers, leading columns that are not sky, the
  saturation level, the serial lag, and gain and read noise by photon
  transfer when the headers lack them;
- per-frame statistics (`measure`): background, noise, ramps, pattern,
  sharp events;
- hot pixels and their temperature trend, the flat field, the PSF from
  frames whose stars are round, and the pointing from the frames' WCS or
  plate solves;
- for frames whose stars trail (a telescope tracking a satellite): the
  trail length and angle per frame, the PSF from the profile across the
  trails, photometry along them, radiation outside them, and each frame's
  trail rate for its twin (`trails`);
- with `--calibration`: gain and read noise from flat and bias pairs, dark
  current, cosmic rays and their read-out time from the darks, and dust
  shadows from the master flat (`calib`);
- header style, so simulated frames carry the instrument's keywords;
- Earth glare, smear, smooth background structure and read-out faults;
- profile blocks from the statistics (`profile.build`): pedestal, noise,
  pattern, ramps, temperature, radiation map;
- with `--skytwin`: corrections against simulated frames, then a realism
  report from simulated twins.

Blocks it cannot learn yet are listed in the report as defaults or left
out, so nothing pretends to be measured.
"""

import glob
import json
import os
import shutil
from collections import Counter, defaultdict

import numpy as np
from astropy.io import fits

from . import analyse, astrometry, calib, contract, keywords, measure, refine, report, trails
from . import ephemeris as ephem
from . import headers as hdr
from . import profile as prof_mod
from . import psf as psf_mod
from .util import header_number, read_json, run_skytwin, spread, write_json, write_rows

NOT_YET = {
    "stray_light (Sun, Moon)": "Sun and Moon glare: set by hand",
    "pointing motion": "wobble, drift and jumps: set by hand",
    "cti": "charge-transfer inefficiency: set by hand",
    "rts_pixels": "flickering pixels: set by hand",
    "psf.field": "field-dependent PSF: set by hand",
}


def find_files(inputs):
    """FITS files from `inputs`: directories (searched recursively), glob patterns, `.txt` lists of paths,
    or paths; sorted, without duplicates."""
    out = []
    for i in inputs:
        if os.path.isdir(i):
            for ext in ("fits", "fit", "fts", "FITS", "FIT"):
                out += glob.glob(os.path.join(i, "**", f"*.{ext}"), recursive=True)
        elif any(c in i for c in "*?["):
            out += glob.glob(i, recursive=True)
        elif i.endswith(".txt"):
            with open(i) as fh:
                out += [line.strip() for line in fh if line.strip()]
        else:
            out.append(i)
    return sorted(set(out))


class Log:
    """What `learn` found, as (setting, status, detail) rows, echoed to stdout when verbose."""

    def __init__(self, verbose=True):
        self.rows, self.verbose = [], verbose

    def step(self, msg):
        """Print a progress line when verbose."""
        if self.verbose:
            print(msg, flush=True)

    def block(self, name, status, detail):
        """Record a finding for the report."""
        self.rows.append((name, status, detail))
        self.step(f"  {name}: {status} - {detail}")


def chip_density(per_image_mpix, bx, by):
    """A density measured per million pixels of a `bx` × `by` binned image, per million chip pixels:
    Skytwin places hot pixels, flickering pixels and dust on the unbinned chip, `bx` × `by` of them to
    each image pixel."""
    return round(per_image_mpix / (bx * by), 2)


def _psf_comps(base, cam):
    """The learned PSF as [(fraction, fwhm_px)], core first; None without a fitted PSF."""
    if "psf" not in base:
        return None
    wings = base["psf"]["wings"]
    return [(1 - sum(wg["fraction"] for wg in wings), cam["fwhm_px"])] + [
        (wg["fraction"], wg["fwhm_px"]) for wg in wings
    ]


def check_catalogues(ucac4, catalog):
    """Refuse a `--ucac4` that is not a UCAC4 directory (one holding `u4b/`) or a `--catalog` that does
    not exist, before any work: a missing catalogue would otherwise only show as frames without matches."""
    if ucac4 and not os.path.isdir(os.path.join(ucac4, "u4b")):
        raise ValueError(f"--ucac4 {ucac4}: not a UCAC4 directory (no u4b/ in it)")
    if catalog and not os.path.exists(catalog):
        raise ValueError(f"--catalog {catalog}: no such file or directory")
    if catalog and os.path.isdir(catalog) and not os.path.isdir(os.path.join(catalog, "u4b")):
        raise ValueError(f"--catalog {catalog}: a directory, but not a UCAC4 one (no u4b/ in it)")


def learn(
    inputs,
    out_path,
    km=None,
    skytwin=None,
    ucac4=None,
    max_frames=1500,
    iterations=2,
    twins=200,
    work=None,
    verbose=True,
    fayde=None,
    catalog=None,
    catalog_name=None,
    catalog_source=None,
    sensor=None,
    calibration=None,
    pixel_pitch_um=None,
    extinction=None,
    ephemeris=None,
):
    """Learn a profile from the frames in `inputs` and write it to `out_path`, with `.learn.md` and
    `.keywords.json` beside it. Returns (profile, report rows).

    Matching, photometry and twins use `catalog` (UCAC4 directory or CSV) when given, else `ucac4`; Fayde
    plate solves need UCAC4 itself. Working files go to `work` (default: beside the profile).

    `calibration` (directories, files, globs or lists, as `inputs`) holds bias, dark and flat frames of the
    same camera. What they measure replaces what the science frames would give: gain and read noise, the
    dark current, the radiation rate and read-out time (darks have no stars to mistake for hits); dust
    shadows come only from them. Hot pixels stay those the science frames show, the darks' own being the
    fallback when the science frames find none.

    `pixel_pitch_um`, the unbinned pixel size, overrides the headers' (radiation rates per cm² need it).
    For ground frames (an airmass or altitude in the headers) the zeropoint is moved above the
    atmosphere, with the extinction fitted against airmass when the frames span enough of it, else
    `extinction` (mag/airmass, default 0.2); the profile's description says what a scene's ground site
    should use. `ephemeris` (a CSV or JPL Horizons vector table, see `ephemeris`) gives frames whose
    headers carry no spacecraft state their state and sub-satellite point.

    Order matters in places. The serial lag is measured before the noise is used for anything. Later
    steps rewrite some per-frame values (ramps about the fixed field, structure inside a field stop,
    radiation-only events), so the raw measurements are kept apart for the realism report, which compares
    them with twins measured the same way. With a field stop, each frame's level is split into light and
    an added level, then the structure every frame shares is fitted with each frame's light and pedestal.
    PSF candidates are frames the headers say were still (searched over all inputs, since such frames can
    be rare), else a sample; either way they are kept only if their stars measure round, and the PSF is
    fitted with any serial lag undone (Skytwin applies it again), on peaks 30 times the noise (frames may be
    scaled anywhere) and under 85% of saturation. A roll convention fixed by plate solves re-reads every
    frame's pointing, which real-sky twins need. Frames inside the Earth glare zone leave the pedestal
    sampling, which would otherwise count their glare twice.

    With `skytwin`, measures that are biased are corrected by simulating with the profile and measuring
    the same way: smear (wings and halos fall in the source columns too) and the smooth field (its masked
    lag-1 covariance also picks up faint source wings), twice each; then `iterations` refinement passes
    and the realism report. The smooth field is compared with twins of the real sky when a catalogue and
    most frames' pointing are known (from a roll keyword or each frame's own WCS): frames of a few random
    stars lack the faint stars of a real field, and the correction would book their light as field.
    """
    log = Log(verbose)
    check_catalogues(ucac4, catalog)
    cat_src = catalog or ucac4
    files = find_files(inputs)
    if not files:
        raise ValueError("no FITS files found")
    work = work or os.path.splitext(out_path)[0] + "-work"
    os.makedirs(work, exist_ok=True)
    sample = spread(files, max_frames)
    log.step(f"{len(files)} files, using {len(sample)}")

    heads, unreadable, first_error = [], 0, None
    for f in sample:
        try:
            heads.append((f, fits.getheader(f)))
        except Exception as e:
            unreadable += 1
            first_error = first_error or f"{f}: {e}"
    if not heads:
        raise ValueError(f"none of the {len(sample)} files has a readable FITS header (first error: {first_error})")
    if unreadable:
        log.block("inputs", "skipped", f"{unreadable} files without a readable header (first: {first_error})")
    if km is None:
        km, notes = hdr.infer([h for _, h in spread(heads, 300)])
        for name, key, why in notes:
            log.block(f"keyword {name}", "found", f"{key} ({why})" if why else key)
        if "exposure_s" not in km:
            log.block("keyword exposure_s", "missing", "rates per second assume 1 s exposures")

    def shape_of(h):
        bx = header_number(h, km.get("binning_x")) or 1
        by = header_number(h, km.get("binning_y")) or 1
        return (int(h.get("NAXIS1", 0)), int(h.get("NAXIS2", 0)), int(bx), int(by))

    groups = Counter(shape_of(h) for _, h in heads)
    (w, h_, bx, by), n_main = groups.most_common(1)[0]
    main = [(f, h) for f, h in heads if shape_of(h) == (w, h_, bx, by)]
    others = ", ".join(f"{a}×{b} bin {c}×{d} ({n})" for (a, b, c, d), n in groups.most_common()[1:4])
    log.block(
        "frames",
        "used",
        f"{n_main} of {len(heads)} at {w}×{h_}, binning {bx}×{by}"
        + (f"; other modes: {others} (learn them separately)" if others else ""),
    )
    mfiles = [f for f, _ in main]
    mheads = [h for _, h in main]
    metas = analyse.metas(mheads, km)

    cam, _, notes = analyse.camera_basics(mheads, km)
    for k, v, src in notes:
        log.block(k, "header", f"{v} ({src})")
    cam.update(width=w, height=h_, sky={"e_per_s": 0.0}, dark_e_per_s=0.0)
    if pixel_pitch_um:
        cam["pixel_pitch_um"] = float(pixel_pitch_um)
        log.block("camera.pixel_pitch_um", "given", f"{pixel_pitch_um} um (--pixel-pitch-um)")
    cam.setdefault("gain_e_per_adu", 1.0)
    cam.setdefault("read_noise_e", 8.0)
    if "gain_e_per_adu" not in km:
        log.block("camera.gain_e_per_adu", "default", "1.0: no gain in the headers")
    gain = cam["gain_e_per_adu"]
    x0 = analyse.edge_columns(spread(mfiles, 40))
    if x0:
        km["skip_left_columns"] = x0
        log.block("edge columns", "measured", f"{x0} leading columns are not sky; left out of every measurement")
    out_type = analyse.output_type(mheads[0])
    proc = {"bias_subtracted": True, "output": out_type}
    sat, n_sat = analyse.saturation(spread(mfiles, 300))
    if sat is not None:
        km["saturation_adu"] = sat
        top = {"u16": 65535, "i16": 32767, "i32": 2**31 - 1}.get(out_type)
        if top is not None and sat >= top:
            log.block("saturation", "measured", f"{sat:.0f} ADU in {n_sat} frames: the converter's limit")
        else:
            proc["clip_max_adu"] = sat
            log.block("saturation", "measured", f"{sat:.0f} ADU in {n_sat} frames: pixels clip here")
    else:
        log.block("saturation", "not seen", "no level where pixels pile up")

    lag_blk, notes = analyse.serial_lag(spread(mfiles, 80), sat, x0)
    log.block("noise.serial_lag", "measured" if lag_blk else "off", "; ".join(notes))
    top_adu = {"u16": 65535, "i16": 32767, "i32": 2**31 - 1}.get(out_type, 65535)
    cal, cal_rn = None, None
    if calibration:
        log.step("calibration frames")
        frames, notes = calib.sort_frames(find_files(calibration), km, (w, h_, bx, by), shape_of)
        log.block("calibration frames", "used" if any(frames.values()) else "none found", "; ".join(notes))
        cal = {"frames": frames, "masters": {}}
        notes, g = [], None
        if frames["bias"]:
            cal_rn, cal["masters"], n = calib.read_noise(frames["bias"], x0)
            notes += n
        if frames["flat"]:
            g, n = calib.flat_gain(frames["flat"], cal["masters"], cal_rn, sat, top_adu, x0)
            notes += n
        if g:
            if "gain_e_per_adu" in km:
                notes.append(f"replaces the header's {cam['gain_e_per_adu']} e/ADU")
            cam["gain_e_per_adu"] = round(g, 3)
            gain = cam["gain_e_per_adu"]
        if cal_rn is not None:
            cam["read_noise_e"] = round(cal_rn * gain, 3)
        if notes:
            log.block(
                "camera.gain_e_per_adu, read_noise_e",
                "measured" if g or cal_rn is not None else "off",
                f"from calibration frames: {gain} e/ADU, {cam['read_noise_e']} e; " + "; ".join(notes),
            )
        cal["gain"] = g
    if "gain_e_per_adu" not in km and not (cal and cal["gain"]):
        g, rn, notes = analyse.photon_transfer(
            mfiles,
            [m.get("time_utc") for m in metas],
            x0,
            lag_blk["coefficient"] if lag_blk else 0.0,
            exposures=[m.get("exposure_s") for m in metas],
        )
        if g:
            cam["gain_e_per_adu"] = round(g, 3)
            cam["read_noise_e"] = round((cal_rn if cal_rn is not None else rn) * g, 3)
            gain = cam["gain_e_per_adu"]
            if cal_rn is not None:
                notes.append("read noise from the bias pairs")
            log.block(
                "camera.gain_e_per_adu, read_noise_e",
                "measured",
                f"{gain} e/ADU, {cam['read_noise_e']} e; " + "; ".join(notes),
            )
        else:
            log.block("camera.gain_e_per_adu, read_noise_e", "default", "; ".join(notes))
    elif "read_noise_e" not in km and cal_rn is None:
        g, rn, notes = analyse.photon_transfer(
            mfiles,
            [m.get("time_utc") for m in metas],
            x0,
            lag_blk["coefficient"] if lag_blk else 0.0,
            exposures=[m.get("exposure_s") for m in metas],
        )
        if g:
            cam["read_noise_e"] = round(rn * gain, 3)
            log.block(
                "camera.read_noise_e",
                "measured",
                f"{cam['read_noise_e']} e with the header's gain {gain} e/ADU (photon transfer gives {g:.2f}); "
                + "; ".join(notes),
            )
        else:
            log.block("camera.read_noise_e", "default", f"{cam['read_noise_e']} e; " + "; ".join(notes))

    log.step("measuring frames")
    real_path = os.path.join(work, "real.json")
    rows = measure.run(mfiles, km, real_path, progress=verbose)
    shutil.copy(real_path, os.path.join(work, "real_raw.json"))
    if ephemeris:
        times, states, enotes = ephem.load(ephemeris)
        n_st = ephem.apply([r["meta"] for r in rows if "error" not in r], times, states)
        ephem.apply(metas, times, states)
        write_rows(rows, real_path)
        log.block(
            "spacecraft states",
            "given" if n_st else "off",
            f"{n_st} of {len(rows)} frames from {os.path.basename(ephemeris)} at their start time; "
            + "; ".join(enotes),
        )

    base = {
        "name": _name(mheads),
        "description": f"Learned by skytwin-calibrate learn from {n_main} frames.",
        "detector_seed": 1,
        "camera": cam,
        "processing": proc,
    }
    if (bx, by) != (1, 1):
        base["detector"] = {"chip_px": [w * bx, h_ * by], "binning": [bx, by]}
    if lag_blk:
        base["noise"] = {"serial_lag": lag_blk}
    log.step("hot pixels")
    keep, group_note = analyse.exposure_group([m.get("exposure_s") for m in metas])
    hfiles, hmetas = [mfiles[i] for i in keep], [metas[i] for i in keep]
    hp, notes = analyse.hot_pixels(spread(hfiles, 150), spread(hmetas, 150), gain, x0, sat)
    if group_note:
        notes.append(group_note)
    dk = None
    if cal and cal["frames"]["dark"]:
        rate, dark_e, dark_t, dnotes = calib.dark_current(cal["frames"]["dark"], cal["masters"], gain, x0)
        if rate is not None:
            dk = (dark_e, dark_t, dnotes)
            if hp is None:
                blk, thr = calib.hot_from_darks(rate, dark_e)
                if blk:
                    hp = {"hot_pixels": blk, "dark": None}
                    notes.append(
                        f"from the darks instead: {blk['density_per_mpix']} per Mpix more than {thr:.2f} e/s "
                        "above the dark current"
                    )
            else:
                low = hp["hot_pixels"]["e_per_s_at_ref"]["empirical"][0]
                blk, thr = calib.hot_from_darks(rate, dark_e, low)
                notes.append(
                    f"the darks hold {blk['density_per_mpix'] if blk else 0} per Mpix above {thr:.2f} e/s (the "
                    "science frames' lowest rate, or 6σ of the dark map if higher)"
                )
    if hp and (bx, by) != (1, 1):
        dens = hp["hot_pixels"]["density_per_mpix"]
        hp["hot_pixels"]["density_per_mpix"] = chip_density(dens, bx, by)
        notes.append(
            f"{dens} per Mpix of the {bx}×{by} binned image is {chip_density(dens, bx, by)} per Mpix of the chip"
        )
    if hp:
        base["hot_pixels"] = hp["hot_pixels"]
        if hp["dark"]:
            base["dark"] = hp["dark"]
        log.block("hot_pixels", "measured", "; ".join(notes))
    else:
        log.block("hot_pixels", "off", "; ".join(notes))
    if dk:
        dark_e, dark_t, dnotes = dk
        ref = dict(base.get("dark") or {})
        tref = ref.get("ref_temp_c", dark_t)
        if tref is not None and dark_t is not None and abs(tref - dark_t) >= 0.5:
            if ref.get("doubling_c"):
                dark_e *= 2 ** ((tref - dark_t) / ref["doubling_c"])
                dnotes.append(f"moved to the science frames' {tref:.1f} C with the hot pixels' doubling")
            else:
                dnotes.append(f"the science frames are at {tref:.1f} C: no temperature trend to move it with")
        if tref is None:
            cam["dark_e_per_s"] = round(dark_e, 4)
        else:
            base["dark"] = dict(ref, e_per_s_at_ref=round(dark_e, 4), ref_temp_c=round(tref, 2))
        log.block("dark", "measured", "; ".join(dnotes))
    log.step("flat field")
    fl, notes = analyse.flat(spread(mfiles, 400), gain, x0, sat)
    log.block("flat", "measured" if fl else "off", "; ".join(notes))
    if cal and cal["frames"]["flat"]:
        mf, nf = calib.master_flat(cal["frames"]["flat"], cal["masters"], sat, top_adu, x0)
        db, notes = calib.dust(mf, (bx, by)) if mf is not None else (None, [f"only {nf} flats at 8-70% of saturation"])
        if db:
            fl = dict(fl or {}, dust=db)
        log.block("flat.dust", "measured" if db else "off", f"master flat of {nf} flats; " + "; ".join(notes))
    if fl:
        base["flat"] = fl
    stop = (fl or {}).get("field_stop")
    if stop:
        notes = analyse.sky_split(rows, stop, x0)
        log.block("sky_e_per_s, processing.pedestal_adu", "measured", "; ".join(notes))
        ff, notes = analyse.fixed_field(rows, fl, x0)
        if ff:
            base.setdefault("noise", {})["fixed_field"] = ff
        log.block("noise.fixed_field", "measured" if ff else "off", "; ".join(notes))
        analyse.refield(rows, stop, x0)
    notes = analyse.transient_sharp(rows, spread(mfiles, 150), x0)
    log.block("radiation events", "measured", "; ".join(notes))
    write_rows(rows, real_path)
    log.step("trails")
    geo = trails.survey(spread(mfiles, 20), x0)
    if sum(g[0] >= trails.MIN_TRAIL_PX for g in geo.values()) >= 3:
        geo.update(trails.survey([f for f in spread(mfiles, 300) if f not in geo], x0))
    trailed = {f: g for f, g in geo.items() if g[0] >= trails.MIN_TRAIL_PX}
    if len(trailed) >= 3:
        q = np.percentile([g[0] for g in trailed.values()], [10, 50, 90])
        log.block(
            "trails",
            "measured",
            f"{len(trailed)} of {len(geo)} frames have trailed stars (autocorrelation): length "
            f"{q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} px (p10/p50/p90)",
        )
    else:
        trailed = {}
        log.block("trails", "none", f"stars round in {len(geo)} sampled frames")
    log.step("PSF")
    main_set = set(mfiles)
    cands = [
        f
        for f in analyse.still_by_header(spread(files, 60000), km, cam.get("pixel_scale_arcsec"))
        if f in main_set or _same_shape(f, w, h_)
    ]
    src = "header tracking rate" if cands else "a sample"
    cands = cands or spread(mfiles, 300)
    if lag_blk:
        cands = analyse.unlag_files(cands, lag_blk, os.path.join(work, "unlag"))
        src += ", serial lag undone"
    pts = analyse.point_frames(cands, sat, x0)
    if len(pts) >= 3:
        try:
            nz = np.median([r["noise_dx"] for r in rows if "error" not in r and r.get("noise_dx")] or [10.0])
            lo = 30 * nz
            pk = (min(lo, 0.05 * sat) if sat else lo, 0.85 * sat if sat else np.inf)
            prof, n = psf_mod.stack(pts[:80], x0, peak_range=pk)
            if n < 20:
                raise ValueError(f"only {n} isolated unsaturated stars")
            fit = psf_mod.fit(prof)
            cam["fwhm_px"] = round(fit["core_fwhm_px"], 3)
            base["psf"] = {"wings": [{k: round(v, 4) for k, v in wg.items()} for wg in fit["wings"]]}
            log.block(
                "psf",
                "measured",
                f"{n} stars from {len(pts)} frames with round stars (chosen by {src}): core FWHM "
                f"{fit['core_fwhm_px']:.2f} px, wings "
                + ", ".join(f"{wg['fraction']:.3f} at {wg['fwhm_px']:.1f} px" for wg in fit["wings"]),
            )
        except ValueError as e:
            log.block("psf", "default", f"{e}; camera.fwhm_px stays {cam.get('fwhm_px', 2.5)}")
    else:
        tfit, tnotes = None, []
        if trailed:
            log.step("PSF across the trails")
            tfit, n_st, n_fr, tnotes = trails.psf_from_trails(list(trailed), trailed, x0, sat)
        if tfit:
            cam["fwhm_px"] = round(float(tfit["core_fwhm_px"]), 3)
            wings = [
                {"fraction": round(float(w["fraction"]), 4), "fwhm_px": round(float(w["fwhm_px"]), 3)}
                for w in tfit["wings"]
            ]
            base["psf"] = {"wings": wings}
            log.block(
                "psf",
                "measured",
                f"profile across {n_st} isolated trails in {n_fr} frames (the motion does not blur it): core FWHM "
                f"{cam['fwhm_px']:.2f} px, wings "
                + ", ".join(f"{w['fraction']:.3f} at {w['fwhm_px']:.1f} px" for w in wings)
                + "; "
                + "; ".join(tnotes),
            )
        else:
            log.block("psf", "default", "; ".join([f"only {len(pts)} frames with round stars", *tnotes]))

    atm = None
    fayde = astrometry.find_fayde(fayde)
    wcs_files = [f for f in spread(mfiles, 400) if astrometry.has_celestial_wcs(fits.getheader(f))]
    if cat_src and skytwin and len(wcs_files) >= 20:
        log.step("astrometry from the header WCS")
        comps = _psf_comps(base, cam)
        sub = spread(wcs_files, 80)
        sub_meta = [keywords.frame_meta(fits.getheader(f), km) for f in sub]
        exps = [m.get("exposure_s") for m in sub_meta]
        keep, ph_note = analyse.exposure_group(exps)
        exps = [e if i in set(keep) else None for i, e in enumerate(exps)]
        cupd, notes, _ = astrometry.from_header_wcs(
            sub,
            skytwin,
            cat_src,
            os.path.join(work, "astrometry"),
            x0=x0,
            sat=sat,
            gain=gain,
            exposures=exps,
            psf_comps=comps,
            trail_geo=geo if trailed else None,
            airmasses=[m.get("airmass") for m in sub_meta],
            extinction=extinction,
        )
        atm = cupd.pop("_atmosphere", None)
        cam.update(cupd)
        if ph_note:
            notes.append(
                "photometry " + ph_note.split(";")[0] + "; other exposures left out (a short exposure's shutter "
                "timing errs by more of its length)"
            )
        log.block(
            "camera.pixel_scale_arcsec, parity, distortion, zeropoint_mag",
            "measured" if cupd else "off",
            "; ".join(notes),
        )
    elif ucac4 and fayde and pts:
        log.step("plate solves")
        ph = [fits.getheader(f) for f in pts[:60]]
        comps = _psf_comps(base, cam)
        cupd, hblock, roll, notes = astrometry.calibrate(
            pts[:60],
            ph,
            analyse.metas(ph, km),
            km,
            fayde,
            ucac4,
            cam.get("pixel_scale_arcsec"),
            gain=gain,
            sat=sat,
            psf_comps=comps,
            skytwin=skytwin,
            catalog=cat_src,
            work=os.path.join(work, "photometry"),
            extinction=extinction,
            x0=x0,
        )
        atm = cupd.pop("_atmosphere", None)
        if cupd:
            cam.update(cupd)
        if hblock:
            base.setdefault("pointing", {})["header"] = hblock
        if roll:
            km["roll"] = roll
            for r, hd in zip(rows, mheads):
                if "error" not in r:
                    r["meta"] = keywords.frame_meta(hd, km)
            write_rows(rows, real_path)
        log.block("pointing.header, scale, parity", "measured" if cupd else "off", "; ".join(notes))
    else:
        why = (
            "no --ucac4"
            if not ucac4
            else "no fayde command (pip install fayde)"
            if not fayde
            else "no frames with round stars"
        )
        log.block("pointing.header, scale, parity", "not learned", f"plate solves need Fayde and UCAC4: {why}")

    if atm:
        k, how = atm["extinction_mag_per_airmass"], atm["airmass"]
        src = f"measured over airmass {how[0]}-{how[1]}" if atm["measured"] else "assumed"
        base["description"] += (
            f" The zeropoint is above the atmosphere: give the scene's ground site "
            f"extinction_mag_per_airmass {k} ({src})."
        )
        log.block("atmosphere", "measured" if atm["measured"] else "assumed", f"extinction {k} mag/airmass ({src})")
    if cat_src and not km.get("roll"):
        n_pt = 0
        for r, hd in zip(rows, mheads):
            if "error" not in r and astrometry.has_celestial_wcs(hd):
                p = astrometry.pointing_from_wcs(hd)
                if p:
                    r["meta"]["pointing"] = p
                    n_pt += 1
        if n_pt:
            log.block("pointing", "measured", f"{n_pt} frames' centre and roll from their own WCS, for real-sky twins")
    scale = cam.get("pixel_scale_arcsec")
    if trailed and scale and "trail_rate_arcsec_per_s" not in km:
        by_exp = defaultdict(list)
        for r in rows:
            if "error" not in r and r.get("file") in geo and r["meta"].get("exposure_s"):
                by_exp[round(r["meta"]["exposure_s"], 3)].append(geo[r["file"]][0])
        n_set = 0
        for r in rows:
            e = None if "error" in r else r["meta"].get("exposure_s")
            if not e:
                continue
            L = geo[r["file"]][0] if r.get("file") in geo else float(np.median(by_exp.get(round(e, 3)) or [0.0]))
            if L >= trails.MIN_TRAIL_PX:
                r["meta"]["trail_rate_arcsec_per_s"] = round(L * scale / e, 3)
                n_set += 1
        log.block(
            "trail rates",
            "measured",
            f"{n_set} frames' twins trail at the frame's rate (its own trail, or the median of its exposure's), "
            "in a random direction",
        )
    base["header"] = hdr.style(spread(mheads, 300), km)
    log.block(
        "header",
        "measured",
        f"{len(base['header']['cards'])} cards copied or templated; "
        f"WCS {'written' if base['header']['wcs'] else 'not written'}"
        + ("; roll keyword not used until its convention is checked" if km.get("roll", {}).get("unverified") else ""),
    )

    stat_rows = rows
    if skytwin:
        log.step("Earth geometry")
        geom = analyse.earth_probe(rows, skytwin, os.path.join(work, "probe"), cam.get("pixel_scale_arcsec"))
        stray, free, notes = analyse.earth_glare(
            rows, gain, geom, roll_known=bool(km.get("roll")) and not km["roll"].get("unverified")
        )
        if stray:
            base["stray_light"] = [stray]
            inside = {f for f, g in geom.items() if g[0] < free}
            stat_rows = [r for r in rows if r.get("file") not in inside]
            notes.append(f"{len(inside)} frames inside it left out of the pedestal and noise sampling")
        log.block("stray_light (Earth)", "measured" if stray else "off", "; ".join(notes))
    else:
        log.block("stray_light (Earth)", "not learned", "needs --skytwin for each frame's Earth geometry")
    log.step("smear")
    exps = [m.get("exposure_s") for m in metas]
    sm_a, sm_b, n_sm = analyse.smear_measure(spread(mfiles, 300), spread(exps, 300), sat, x0)
    if sm_a is not None:
        base["smear"] = {"above_s_per_row": max(sm_a, 0.0), "below_s_per_row": max(sm_b, 0.0)}
        log.block(
            "smear",
            "measured",
            f"{n_sm} bright unsaturated sources: {sm_a:.2e} s/row above, "
            f"{sm_b:.2e} below (raw; corrected below with --skytwin)",
        )
    else:
        log.block("smear", "off", f"only {n_sm} bright unsaturated sources with clear columns")

    sf, notes = analyse.smooth_field(stat_rows)
    if sf:
        base.setdefault("noise", {})["smooth_field"] = sf
    log.block("noise.smooth_field", "measured" if sf else "off", "; ".join(notes))

    rf, notes = analyse.readout_faults(rows)
    if rf:
        base["readout_faults"] = rf
    log.block("readout_faults", "measured" if rf else "off", "; ".join(notes))

    prof = prof_mod.build(stat_rows, base)
    req = None
    if cat_src:
        req = contract.catalog_block(
            cat_src,
            _mag_limit(prof, rows),
            name=catalog_name,
            source=catalog_source,
            with_colour="color_term" in prof.get("camera", {}),
        )
    contract.stamp(prof, mheads, inputs, n_main, catalog=req, sensor=sensor)
    log.block(
        "catalog",
        "recorded" if req else "not learned",
        f"{req['name']} ({req['kind']}) to magnitude {req['mag_limit']}"
        + (f", SHA-256 {req['sha256'][:12]}..., {req['rows']} rows" if req and req.get("sha256") else "")
        if req
        else "no --catalog or --ucac4: scenes must give a full scene.catalog",
    )
    write_rows(rows, real_path)
    n_used = len(prof_mod.usable(rows))
    log.block(
        "processing.pedestal_adu, noise",
        "measured",
        f"pedestal, extra white noise, pattern and ramps linked per frame, from {n_used} frames",
    )
    if "temperature_c" in prof:
        log.block("temperature_c", "measured", "distribution of the header temperatures")
    cr = cal_cr = None
    if cal and cal["frames"]["dark"]:
        cr, crnotes = calib.cosmic_rays(cal["frames"]["dark"], prof_mod.area_cm2(prof), x0)
        if "pixel_pitch_um" not in cam:
            crnotes.append("pixel pitch unknown: 13 um assumed for the chip area (give --pixel-pitch-um)")
        if cr:
            cal_cr = cr
            prof["radiation"] = {"rate_per_cm2_s": cr["rate_per_cm2_s"]}
            log.block("radiation", "measured", "from the darks: " + "; ".join(crnotes))
        else:
            log.block("radiation (darks)", "off", "; ".join(crnotes))
    if trailed and not cr:
        exp_of = {f: m.get("exposure_s") for f, m in zip(mfiles, metas)}
        groups = defaultdict(list)
        for f in trailed:
            if exp_of.get(f):
                groups[round(exp_of[f], 3)].append(f)
        long = [e for e, g in groups.items() if len(g) >= 5]
        if long:
            e = max(long)
            log.step("radiation outside the trails")
            cr, crnotes = trails.radiation(sorted(groups[e]), trailed, e, prof_mod.area_cm2(prof), x0, sat)
            if "pixel_pitch_um" not in cam:
                crnotes.append("pixel pitch unknown: 13 um assumed for the chip area (give --pixel-pitch-um)")
            if cr:
                cr["measured_readout"] = True
                prof["radiation"] = {"rate_per_cm2_s": cr["rate_per_cm2_s"]}
                log.block("radiation", "measured", "from frames with trails: " + "; ".join(crnotes))
            else:
                log.block("radiation (trails)", "off", "; ".join(crnotes))
        else:
            log.block("radiation (trails)", "off", "no exposure with 5+ frames with trails")
    if not cr and "radiation" in prof:
        sc = (prof["radiation"].get("map_scale") or {}).get("empirical")
        tail = f"; frame-to-frame factor p5 {sc[5]:.2g}, p95 {sc[95]:.2g}" if sc else ""
        log.block(
            "radiation",
            "measured",
            f"rate map by sub-satellite point; median {prof['radiation']['rate_per_cm2_s']} /cm²/s{tail}",
        )
    elif not cr:
        ok = [r for r in rows if "error" not in r and r.get("sharp") and r["meta"]["exposure_s"]]
        if ok:
            area = prof_mod.area_cm2(prof)
            rates = np.array([r["sharp"] / r["meta"]["exposure_s"] / area for r in ok])
            med = float(np.median(rates))
            prof["radiation"] = {"rate_per_cm2_s": round(med, 3)}
            if len(ok) > 30:
                prof["radiation"]["map_scale"], _ = prof_mod.rate_spread(rates / med, [r["sharp"] for r in ok])
            log.block(
                "radiation",
                "measured",
                "one rate for all frames, with its frame-to-frame spread (no sub-satellite point in the headers)",
            )
    if "radiation" in prof:
        ro, first, notes = analyse.radiation_readout(rows)
        if cr and cr["measured_readout"]:
            ro = cr.get("readout_s")
            first = cr.get("first_row") or first
            src = "the frames with trails" if trailed and not cal_cr else "the darks"
            if not ro:
                notes = [f"{src} show no read-out hits"]
            elif cr.get("first_row"):
                notes = [f"{ro} s from {src}, {first} read first (see radiation)"]
            else:
                where = f"{first} read first from the science frames" if first else "first row unknown (bottom)"
                notes = [f"{ro} s from {src} (see radiation); {where}"]
        if ro:
            prof["radiation"]["readout_s"] = ro
            if first:
                prof["radiation"]["first_row"] = first
        log.block("radiation.readout_s", "measured" if ro else "off", "; ".join(notes))
    for k, why in NOT_YET.items():
        log.block(k, "not learned", why)
    write_json(prof, out_path, indent=1, allow_nan=False)
    write_json(km, os.path.splitext(out_path)[0] + ".keywords.json", indent=1)

    rep_md = None
    if skytwin and sm_a is not None:
        for i in range(2):
            sim_a, sim_b = _sim_smear(out_path, skytwin, os.path.join(work, f"smear{i}"), sat, x0, cam)
            pr = read_json(out_path)
            fa = sm_a / sim_a if sim_a and sim_a > 0 and sm_a > 0 else 1.0
            fb = sm_b / sim_b if sim_b and sim_b > 0 and sm_b > 0 else 1.0
            pr["smear"]["above_s_per_row"] = float(f"{pr['smear']['above_s_per_row'] * fa:.3e}")
            pr["smear"]["below_s_per_row"] = float(f"{pr['smear']['below_s_per_row'] * fb:.3e}")
            write_json(pr, out_path, indent=1, allow_nan=False)
            log.block(
                f"smear correction {i + 1}",
                "done",
                f"real/simulated {fa:.2f} above, {fb:.2f} below -> "
                f"{pr['smear']['above_s_per_row']:.2e}, {pr['smear']['below_s_per_row']:.2e} s/row",
            )
    if skytwin and sf:
        real_rows = [r for r in stat_rows if "error" not in r and r.get("field")]

        def exc(rs):
            return np.sqrt(max(np.median([r["field"]["cov1"] for r in rs]), 1e-9))

        rates = [r["meta"].get("trail_rate_arcsec_per_s") for r in real_rows]
        rates = [v for v in rates if v]
        drift = float(np.median(rates)) if len(rates) > len(real_rows) / 2 else None
        sky = sum((r["meta"].get("pointing") or {}).get("roll_deg") is not None for r in real_rows)
        real_sky = bool(cat_src) and sky > len(real_rows) / 2
        for i in range(2):
            fw = os.path.join(work, f"field{i}")
            if real_sky:
                cat = dict(astrometry.scene_catalog(cat_src), mag_limit=_mag_limit(read_json(out_path), rows))
                sky_rows = [r for r in real_rows if (r["meta"].get("pointing") or {}).get("roll_deg") is not None]
                sims = refine.twins(out_path, sky_rows, km, skytwin, fw, 30, seed=11 + i, catalog=cat, keep=0)
                stop = (read_json(out_path).get("flat") or {}).get("field_stop")
                if stop:
                    analyse.refield(sims, stop, x0)
                sims = [r for r in sims if r.get("field")]
            else:
                sims = _sim_frames(out_path, skytwin, fw, km, n=30, drift=drift)
            fac = exc(real_rows) / max(exc(sims), 1e-6)
            pr = read_json(out_path)
            ra = pr["noise"]["smooth_field"]["rms_adu"]
            if "empirical_linked" in ra:
                refine.scale_linked(ra, fac)
            else:
                ra["empirical"] = [round(x * fac, 3) for x in ra["empirical"]]
            write_json(pr, out_path, indent=1, allow_nan=False)
            log.block(
                f"smooth field correction {i + 1}",
                "done",
                f"real/simulated {fac:.2f}"
                + (" against twins of the real sky" if real_sky else " against frames of random stars"),
            )
    if skytwin:
        real = read_json(real_path)
        for i in range(iterations):
            log.step(f"refine pass {i + 1}")
            r = refine.iterate(out_path, real, km, skytwin, os.path.join(work, f"refine{i}"), n_twins=twins, seed=i + 1)
            log.block(f"refine pass {i + 1}", "done", json.dumps(r) if r else "nothing to correct")
        log.step("twins for the realism report")
        cat = None
        if cat_src:
            cat = dict(astrometry.scene_catalog(cat_src), mag_limit=_mag_limit(read_json(out_path), rows))
        sim = refine.twins(out_path, real, km, skytwin, os.path.join(work, "twins"), min(300, len(real)), catalog=cat)
        write_rows(sim, os.path.join(work, "sim.json"))
        rep_md = report.run(
            os.path.join(work, "real_raw.json"),
            os.path.join(work, "sim.json"),
            os.path.join(work, "report.md"),
            "Realism report",
        )
    md = _markdown(log.rows, out_path, rep_md)
    with open(os.path.splitext(out_path)[0] + ".learn.md", "w") as fh:
        fh.write(md)
    return read_json(out_path), log.rows


def _mag_limit(prof, rows, floor_sigma=0.5, cap=17.0):
    """Faintest catalogue magnitude worth drawing: a star whose whole flux
    is `floor_sigma` of one pixel's noise. Fainter ones change nothing a
    measure sees, and on wide fields there are millions of them."""
    cam = prof.get("camera", {})
    zp, g, t = cam.get("zeropoint_mag"), cam.get("gain_e_per_adu", 1.0), cam.get("exposure_s", 1.0)
    nz = [r.get("noise_dx") or r.get("noise") for r in rows if "error" not in r]
    nz = [v for v in nz if v]
    if zp is None or not nz or not t:
        return cap
    noise_e = float(np.median(nz)) * g
    return float(min(cap, round(zp - 2.5 * np.log10(floor_sigma * noise_e / t), 1)))


def _sim_smear(profile_path, skytwin, work, sat, x0, cam, n=30):
    """Smear (above, below; s/row) measured on `n` frames simulated with the profile, bright stars only, no
    radiation or hot pixels."""
    os.makedirs(work, exist_ok=True)
    files = []
    for s in range(n):
        f = os.path.join(work, f"s{s}.fits")
        scene = {
            "seed": 9000 + s,
            "instrument": os.path.abspath(profile_path),
            "overrides": {"radiation": None, "hot_pixels": None},
            "scene": {"random_stars": {"count": 40, "max_flux_e": 3e6, "min_flux_e": 1e5}},
        }
        run_skytwin(skytwin, scene, f)
        files.append(f)
    a, b, _ = analyse.smear_measure(files, [cam.get("exposure_s", 1.0)] * n, sat, x0)
    return a, b


def _sim_frames(profile_path, skytwin, work, km, n=30, drift=None):
    """Statistics of `n` frames simulated with the profile without radiation, measured as the real frames
    were (inside a field stop if the profile has one); only rows with a `field` measurement. With `drift`
    (arcsec/s) the stars trail, as in frames taken tracking a satellite."""
    os.makedirs(work, exist_ok=True)
    out = []
    for s in range(n):
        f = os.path.join(work, f"f{s}.fits")
        scene = {
            "seed": 7000 + s,
            "instrument": os.path.abspath(profile_path),
            "overrides": {"radiation": None, **({"pointing": {"drift_arcsec_per_s": drift}} if drift else {})},
            "scene": {"random_stars": {"count": 60, "max_flux_e": 1e6, "min_flux_e": 1e4}},
        }
        run_skytwin(skytwin, scene, f)
        out.append(measure.frame(f, km))
    stop = (read_json(profile_path).get("flat") or {}).get("field_stop")
    if stop:
        analyse.refield(out, stop, int(km.get("skip_left_columns", 0)))
    return [r for r in out if r.get("field")]


def _same_shape(f, w, h):
    """True when FITS file `f` is `w` × `h` px (False when unreadable)."""
    try:
        hd = fits.getheader(f)
    except Exception:
        return False
    return int(hd.get("NAXIS1", 0)) == w and int(hd.get("NAXIS2", 0)) == h


def _name(heads):
    """Profile name from the most common INSTRUME, TELESCOP or DETECTOR header value."""
    for k in ("INSTRUME", "TELESCOP", "DETECTOR"):
        v = [str(h.get(k)) for h in heads if h.get(k)]
        if v:
            return f"{Counter(v).most_common(1)[0][0]} (learned)"
    return "learned"


def _markdown(rows, out_path, rep_md):
    """The `.learn.md` report: one table row per finding, then the realism report if there is one."""
    lines = [
        f"# What `learn` measured for {os.path.basename(out_path)}",
        "",
        "| Setting | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for name, status, detail in rows:
        lines.append(f"| {name} | {status} | {str(detail).replace('|', '/')} |")
    lines += [
        "",
        "Status: *measured* from the images, *header* read from headers, *default* not measurable from "
        "these frames, *not learned* outside what `learn` does yet (set by hand if it matters).",
    ]
    if rep_md:
        lines += ["", rep_md]
    return "\n".join(lines) + "\n"
