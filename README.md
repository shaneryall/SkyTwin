<div align="center" markdown>

# Skytwin

**Synthetic star-field images with satellite streaks, instrument defects and an exact answer key
for every frame.**

[![CI](https://github.com/shaneryall/SkyTwin/actions/workflows/ci.yml/badge.svg)](https://github.com/shaneryall/SkyTwin/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/skytwin?label=pypi)](https://pypi.org/project/skytwin/)
[![crates.io](https://img.shields.io/crates/v/skytwin-cli)](https://crates.io/crates/skytwin-cli)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24544458.svg)](https://doi.org/10.5281/zenodo.24544458)

[Install](#install) · [Quick start](#quick-start) ·
[Tutorial](docs/tutorials/first-training-set.ipynb) · [Profiles](docs/reference/profile.md) ·
[Labels](docs/labels.md) · [Documentation](https://shaneryall.github.io/SkyTwin/) ·
[Changelog](CHANGELOG.md)

</div>

Skytwin is developed and maintained by [Shane Ryall](https://shaneryall.com/).

Skytwin is a simulator, written in Rust with Python bindings, of what a camera sees when it looks
for satellites and other resident space objects, from orbit or from the ground. A frame from a
real camera comes with no list of what is in it. A frame from Skytwin comes with a truth file
that gives the position, path, flux and signal-to-noise ratio of every star, streak and target.
That makes it a test bench for detectors, trackers and plate solvers (such as
[Fayde](https://github.com/shaneryall/Fayde)), and a source of labelled training data for space
domain awareness.

Stars are placed by hand, at random, or read from UCAC4 or a Gaia extract and moved to the
observation epoch; with a sky pointing, the FITS header carries a TAN or TAN-SIP WCS. Observers
and targets come from TLEs, state vectors or ephemeris tables, with light-time, Earth shadow,
glints from tumbling facets, and refraction and extinction for a ground site. An instrument
profile holds every detector and optics defect as a setting, fixed or drawn per frame. The same
scene file and seed give the same pixels on every machine.

The pictures in this file are synthetic frames made by
[`scripts/make_readme_figures.py`](scripts/make_readme_figures.py). Each frame is shown with the
asinh stretch that `skytwin-export` writes, so its grey levels follow its own noise and can't be
compared across frames. The first picture shows random stars through each shipped
[instrument profile](profiles/README.md), a 164-pixel crop from the centre of each frame at the
camera's own pixel scale:

![Five crops of simulated star fields, one per shipped profile: an example small-satellite camera,
WISE W1, the CASSIOPE Fast Auroral Imager, and ground cameras at Ckoirama and AstriaNet, with
pixel scales from 0.934 to 380 arcseconds. Each has its own noise, star shapes and
background.](docs/images/profiles.png)

## Install

The command needs Rust 1.85 or later. To install it from a checkout:

```bash
cargo install --path crates/skytwin-cli
```

The Python package, which also provides the `skytwin-export` command, and `skytwin-calibrate`
need Python 3.10 or later. In a virtual environment, from the repository root:

```bash
pip install ./crates/skytwin-py
```

```bash
pip install -e tools/calibrate
```

A star catalogue, a UCAC4 directory or a Gaia CSV extract, is optional; scenes need one only for
real skies.

## Quick start

Render the example scene from the repository root:

```bash
cargo run --release -p skytwin-cli -- examples/two-streaks.json out/two-streaks.fits
```

This writes `out/two-streaks.fits`, its answer key `out/two-streaks.truth.json`, and
`out/two-streaks.provenance.json`. In the truth file, `streaks[0]` gives the first streak's
`start` and `end` pixels and its `flux_e`. `--seed N` overrides the scene's seed.

More scenes, from a camera in a 790 km orbit to a ground telescope tracking a MEO satellite, are
in [`examples/`](examples/). Four of them, `leo-observer`, `leo-stares-geo`, `leo-tracks-geo`, and
`ucac4-field`, read their stars from a UCAC4 catalogue at `data/ucac4`. To render them, copy
UCAC4 to `data/ucac4`, or set `scene.catalog.ucac4` in the scene file to the path of your copy.
`examples/tumbling-glint.json` renders a tumbling target whose panels reflect the Sun towards the
camera once in the exposure. The truth file records the signal along its streak:

![A satellite streak with a bright spot halfway along it, and a plot of the target's signal
against time: about 500 electrons per second, rising briefly to 1,253 times that
mid-exposure.](docs/images/glint.png)

Each pixel draws its noise from its own random stream, so the same seed with and without a
satellite gives an exact clean and dirty pair. In the next picture, made with the example profile
without its frame-transfer smear, the two whole 1024 × 1024 frames differ in 4,565 pixels, all
where the streak's light falls; the difference panel is on a linear scale:

![Two identical star fields, the second with a satellite streak, and their absolute difference,
which is black except for the streak.](docs/images/clean-dirty.png)

## Training sets

The `--frames` option renders independent frames on all cores, and frame k uses the scene's seed
plus k. To render 20 frames of `examples/dirty-frame.json` from the repository root:

```bash
skytwin examples/dirty-frame.json out/batch/f.fits --frames 20 --compact-truth
```

Each frame of this scene is a 2 MB FITS file, so a set of 50,000 frames takes about 105 GB.

From Python, frames arrive as numpy arrays with their truth, with exactly the pixels of the same
frame from `--frames`:

```python
import skytwin

sim = skytwin.Simulator("examples/dirty-frame.json")
for frame in sim.stream(count=20):
    image, stars = frame.image, frame.truth["stars"]
```

To write a set that Ultralytics YOLO and COCO tools such as DETR read directly:

```bash
skytwin-export examples/dirty-frame.json out/yolo/ --frames 20 --format yolo --format coco --tile 640
```

Stars are class `star`, and targets and streaks are class `satellite`. An object is labelled
when its matched-filter signal-to-noise ratio in the tile reaches 5, the default. The next picture
draws the rotated boxes that `--format yolo-obb` writes, blue for stars and orange for satellites.
`--format yolo` writes axis-aligned boxes around the same objects, and COCO records both:

![Two 212-pixel tiles with their label boxes. On the left, small blue squares around point stars
and a long orange box along a satellite streak. On the right, long blue boxes along trailed stars
and a small orange square on the tracked satellite.](docs/images/labels.png)

## Instrument profiles

A profile is one JSON file holding everything tuned to a sensor: noise, pattern noise, dark
current, hot and flickering pixels, radiation hits, charge-transfer trails, smear, saturation,
flat field, stray light, pointing motion and the FITS header style. The next picture turns on one
defect at a time in a plain camera, with settings strong enough to see:

![Eight crops, each with one defect: radiation hits, hot pixels, a charge-transfer trail behind a
bright star, diagonal pattern noise, ring-shaped dust shadows, a saturated star whose charge
spreads along its column and row, a smear column through a bright star, and stars repeated where
the pointing jumped.](docs/images/defects.png)

To learn a profile from sample frames, with the installed `skytwin` refining it, and print what it
holds:

```bash
skytwin-calibrate learn FRAMES_DIR -o profile.json --skytwin skytwin --catalog GAIA_CSV
```

```bash
skytwin-calibrate describe profile.json
```

`FRAMES_DIR` is a folder of FITS frames and `GAIA_CSV` a Gaia extract.
[`profiles/cassiope-fai-nir.json`](profiles/cassiope-fai-nir.json) was learned this way from the
public DUST dataset, frames of the CASSIOPE Fast Auroral Imager's near-infrared channel.

## Documentation

The [documentation site](https://shaneryall.github.io/SkyTwin/) holds every page below.

| Document | Covers |
| --- | --- |
| [docs/tutorials/first-training-set.ipynb](docs/tutorials/first-training-set.ipynb) | Tutorial: describe a camera, render a frame, read its labels and export a YOLO training set |
| [docs/scene-format.md](docs/scene-format.md) | Scene files, the truth file and FITS header keywords |
| [docs/override-a-profile.md](docs/override-a-profile.md) | Using an instrument profile in a scene and changing its settings |
| [docs/reference/profile.md](docs/reference/profile.md) | Every key of an instrument profile |
| [docs/design/realism.md](docs/design/realism.md) | How profiles model real defects, and the limits of the models |
| [docs/export-training-data.md](docs/export-training-data.md) | Exporting a YOLO or COCO training set, and loading frames in PyTorch |
| [docs/labels.md](docs/labels.md) | Every label field, class and box rule, export option and output file |
| [docs/design/labels.md](docs/design/labels.md) | How the signal-to-noise ratio and the label rules are computed |
| [docs/learn-a-profile.md](docs/learn-a-profile.md) | Learning a profile from real frames; scoring a detector |
| [docs/reference/calibrate.md](docs/reference/calibrate.md) | Every `skytwin-calibrate` command and the files `learn` writes |
| [Command-line reference](https://shaneryall.github.io/SkyTwin/latest/reference/cli/) | Every option of `skytwin`, `skytwin-export` and `skytwin-calibrate`, from their `--help` |
| [docs/reference/python-api.md](docs/reference/python-api.md) | The `skytwin` Python package: simulator, labels, PyTorch dataset, export |
| [docs/design/calibration.md](docs/design/calibration.md) | How `skytwin-calibrate` measures each block |
| [docs/validation.md](docs/validation.md) | Checks against independent tools, reproducibility and speed |
| [docs/references.md](docs/references.md) | The papers and standards the models follow |
| [docs/versioning.md](docs/versioning.md) | What counts as a breaking change |
| [CHANGELOG.md](CHANGELOG.md) | What changed in each release |

The library's API documentation: `cargo doc --open -p skytwin-core`.

## Help

Ask questions in [Discussions](https://github.com/shaneryall/SkyTwin/discussions) and report bugs
in [Issues](https://github.com/shaneryall/SkyTwin/issues). Report a vulnerability privately, as
[SECURITY.md](SECURITY.md) describes. To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use Skytwin in published work, cite it as [CITATION.cff](CITATION.cff) gives. The DOI
[10.5281/zenodo.24544458](https://doi.org/10.5281/zenodo.24544458) covers every version; each
release also has a DOI of its own on Zenodo.

## Licence

Skytwin uses the GNU General Public License, version 3 or later; see [LICENSE](LICENSE).
Third-party data and code notices are in [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).

## Author

[Shane Ryall](https://shaneryall.com/) is a Canadian AI researcher and engineer based in Ottawa.
Find his other projects on [GitHub](https://github.com/shaneryall).
