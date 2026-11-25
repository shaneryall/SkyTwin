# Changelog

All notable changes to Skytwin are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-25

### Added

- `skytwin --version` prints the version, for bug reports.
- Training labels: every star, streak and target in the truth carries a
  `label` with its drawn path, matched-filter and peak signal-to-noise
  ratios against the frame's noise, extent above the noise and saturation.
  Rendered pixels are unchanged.
- `skytwin-export` writes YOLO, YOLO-OBB and COCO training sets with
  tiling, a signal-to-noise threshold and a noise-normalised asinh stretch;
  `skytwin.labels` holds the conversions, and `skytwin.torch` gains
  `detection_target` and `collate` for batching frames with different
  numbers of objects. See `docs/labels.md`.
- `examples/track-rate.json`: tracking a GEO satellite at its rate from
  a 790 km orbit through the example profile, so the stars trail into
  streaks and the satellite stays a point.
- Rendering: point stars with a pixel-integrated Gaussian PSF, satellite
  streaks, FITS output (u16, i16, i32 or f32) and a truth file for every
  frame.
- Sky: UCAC4 and CSV (Gaia) catalogues with proper motion and parallax;
  TAN and TAN-SIP WCS; stellar aberration and solar light deflection;
  passbands with AB zeropoints and colour terms.
- Time and orbits: UTC/TT with leap seconds; observers and targets from
  TLEs (SGP4), state vectors or ephemeris tables; Earth shadow and
  occultation; Lambertian and faceted targets with glints and tumbling;
  Earthshine and Moonshine; target tracking.
- Instrument profiles: processing, read-out noise, serial lag, fixed and
  smooth background fields, per-frame sky light, temperature and dark
  current, hot and flickering pixels, radiation hits with a rate map,
  CTI, PSF wings, Moffat tail and field variation, smear, bleeding,
  serial overflow, flat field with vignetting, PRNU, dust and field stop,
  radial lens distortion, pointing motion and header pointing errors,
  stray light, zodiacal light, chip layout and instrument-style headers,
  read-out faults.
- Curved drift and image roll during the exposure
  (`pointing.drift_turn_deg_per_s`, `roll_rate_deg_per_s`,
  `roll_centre_px`): stars trail in arcs, and the reported roll follows
  the image.
- Profiles as contracts: `schema_version`, `sensor`, `learned` provenance
  and the `catalog` a profile was calibrated against; scenes give only the
  catalogue's path.
- Sequences, `only_visible` target filtering, and batches for training
  sets on all cores; per-pixel random streams for exact clean/dirty pairs.
- Truth: Cramér–Rao bounds, trail ends, light curves, photon counts.
- A provenance manifest for every run.
- The `skytwin` Python package streaming frames to numpy and PyTorch.
- `skytwin-calibrate`: learns a profile from real frames (headers, noise,
  photon-transfer gain, serial lag, hot pixels, flat and field stop, PSF,
  astrometry and distortion from the frames' own WCS, forced-photometry
  zeropoint and colour term, radiation, stray light, read-out faults),
  refines it against simulated twins, reports realism, scores detectors,
  and checks a catalogue against a profile.
- `learn --calibration` reads the camera's bias, dark and flat frames for gain, read noise, dark
  current, radiation rate, read-out time and dust shadows; `learn` also reads the pixel size
  from the headers (`XPIXSZ` and kin).
- `learn` handles frames whose stars trail, as when the telescope tracks a satellite: it measures
  each frame's trail and fits the PSF, zeropoint and radiation around the trails.
- `learn` gives ground frames a zeropoint above the atmosphere, fitted against airmass, or moved
  with `--extinction` (default 0.2 mag/airmass) when the frames span too little airmass.
- `learn --ephemeris` gives spacecraft states and sub-satellite points to frames whose headers
  carry none, from a CSV or a JPL Horizons vector table.
- Profiles learned from public archives: `profiles/cassiope-fai-nir.json` (the DUST dataset),
  `profiles/wise-w1.json` (WISE's 3.4 µm band), `profiles/astrianet-pl16803.json` (a ground
  camera tracking satellites) and `profiles/ckoirama-cdk24.json` (a ground 0.6 m telescope).
  `profiles/README.md` gives each one's source and how closely its twins match.
- A documentation site built from `docs/`, versioned with `mike`, with the command-line reference
  from `--help` and the Python API from docstrings.
- Ground-site observers: a telescope at a WGS84 latitude, longitude and
  height, turning with the Earth (UT1 = UTC + DUT1), seen through the
  atmosphere (refraction from pressure and temperature, extinction per
  airmass, the horizon as the occulting edge); `SITELAT`, `SITELONG`,
  `SITEELEV` and `AIRMASS` header keywords and the site and atmosphere in
  the truth. Example `examples/ground-tracks-meo.json`.

[Unreleased]: https://github.com/shaneryall/SkyTwin/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shaneryall/SkyTwin/releases/tag/v0.1.0
