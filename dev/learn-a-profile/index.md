# Learn a profile from real frames

`skytwin-calibrate learn` turns a folder of an instrument's FITS frames into an instrument profile, with a report of where each setting came from and how closely simulated frames match the real ones.

Before you start, you need:

- Python 3.10 or later and a Rust toolchain.
- A checkout of this repository.
- Sample FITS frames from one camera, as a folder, files, glob patterns or a `.txt` list.
- Optional: a star catalogue covering the frames' fields, either a Gaia CSV with Skytwin's column names or a UCAC4 directory. Without one, the zeropoint, colour term and real-sky twins are not learned. [design/calibration.md](https://shaneryall.github.io/SkyTwin/dev/design/calibration/#choosing-a-catalogue) says which to choose.
- Optional: [Fayde](https://pypi.org/project/fayde/) (`pip install fayde`) and a UCAC4 directory (`--ucac4`), to plate solve frames that carry no celestial WCS of their own. A Gaia CSV alone isn't enough for the plate solves.

## Steps

1. Build `skytwin` and install the tool from the repository root:

```
cargo build --release
pip install -e tools/calibrate
```

1. Learn the profile:

```
skytwin-calibrate learn FRAMES_DIR -o profile.json --skytwin target/release/skytwin --catalog GAIA_CSV
```

Leave out `--skytwin` to skip the refinement and the realism report, and `--catalog` to skip the steps that need stars.

1. Open `profile.learn.md`. Each setting is marked *measured*, *header*, *default* or *not learned*, with its evidence, and the realism report follows.
1. If the report shows a header card read wrongly, such as the exposure time taken from the wrong keyword, correct it in `profile.keywords.json` and learn again with `--keywords profile.keywords.json`.
1. Print the profile's summary and a scene to start from:

```
skytwin-calibrate describe profile.json
```

## Result

`profile.json` is ready for a scene's `instrument`. Blocks that `learn` does not fit, such as pointing motion and CTI, are listed in the report; add them by hand from the [profile reference](https://shaneryall.github.io/SkyTwin/dev/reference/profile/index.md). [Use and override an instrument profile](https://shaneryall.github.io/SkyTwin/dev/override-a-profile/index.md) shows how a scene uses it.

## Add calibration frames

Give the camera's bias, dark and flat frames with `--calibration DIR`. Gain, read noise, dark current, cosmic rays and dust shadows then come from them instead of the science frames.

## Learn from ground frames

Frames with an airmass or altitude in their headers are treated as taken from the ground, and the profile's zeropoint is the one above the atmosphere. When the frames span too little airmass to fit the extinction, give it with `--extinction` (mag/airmass); the report marks it as assumed. Give the scene's ground site the extinction that the profile's description names.

## Give spacecraft states

When the headers carry no spacecraft position, give `--ephemeris` a CSV or a JPL Horizons vector table; the [command-line reference](https://shaneryall.github.io/SkyTwin/dev/reference/cli/#skytwin-calibrate-learn) gives the format. Each frame gets the state at its time and, when its headers give none, its sub-satellite point. Radiation maps and twins matched in orbit need it.

## Run the steps one at a time

The commands `learn` runs can also be run separately, for example to build on a base profile whose smear, stray light, layout or header blocks you measured by hand.

1. Write a keyword map for your instrument's headers, starting from [`examples/keywords.json`](https://github.com/shaneryall/SkyTwin/blob/main/tools/calibrate/examples/keywords.json).
1. Measure the archive:

```
skytwin-calibrate measure --list frames.txt --keywords keywords.json -o real.json
```

1. Fit the PSF from frames where stars are points, and put the result in the base profile's `camera.fwhm_px` and `psf.wings`:

```
skytwin-calibrate psf --list point_frames.txt -o psf.json
```

1. Build the profile on top of the base profile:

```
skytwin-calibrate profile real.json --base base.json -o profile.json
```

1. Refine it in place (simulate, measure, correct; two passes by default):

```
skytwin-calibrate refine profile.json --real real.json --keywords keywords.json --skytwin target/release/skytwin
```

1. Simulate twins of real frames and compare them:

```
skytwin-calibrate twins profile.json --real real.json --keywords keywords.json --skytwin target/release/skytwin --ucac4 /path/to/ucac4 -o sim.json
skytwin-calibrate report --real real.json --sim sim.json -o report.md
```

## Score a detector

Render a frame, run a detector on it, and score its source list against the truth:

```
skytwin scene.json frame.fits
fayde detect --image frame.fits -o frame.sources.fits
skytwin-calibrate score --truth frame.truth.json --sources frame.sources.fits -o score.json
```

`score.json` holds completeness by magnitude, false detections, and centroid error in units of each star's Cramér–Rao bound; [design/calibration.md](https://shaneryall.github.io/SkyTwin/dev/design/calibration/#the-cram%C3%A9rrao-bound-in-scores) explains that bound.
