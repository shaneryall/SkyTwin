# skytwin-calibrate reference

Every command, the files `learn` writes, and what it leaves out. [learn-a-profile.md](https://shaneryall.github.io/SkyTwin/dev/learn-a-profile/index.md) gives the steps; [design/calibration.md](https://shaneryall.github.io/SkyTwin/dev/design/calibration/index.md) explains how each block is measured.

## Commands

| Command    | Does                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `learn`    | Learns a whole profile from sample frames in one step: the commands below, in order, with header and catalogue handling around them                                                                                                                                                                                                                                                                       |
| `check`    | Compares a local star catalogue with the one a profile was calibrated against: columns, rows and SHA-256                                                                                                                                                                                                                                                                                                  |
| `describe` | Prints what a profile is, what it needs, and a scene to start from                                                                                                                                                                                                                                                                                                                                        |
| `measure`  | Per frame: background level, noise (MAD, or quantiles above the clip on mostly clipped frames), clipped fraction, ramps along x and y, residual structure, the strongest coherent pattern (period, angle, amplitude), sharp single-pixel events, corrupted read-out rows, the odd/even row offset, saturated pixels, and header values through the keyword map                                            |
| `psf`      | Stacks isolated, unsaturated stars and fits a core plus two wider Gaussians by forward modelling (pixel-integrated stars at random sub-pixel positions, stacked with the same code)                                                                                                                                                                                                                       |
| `profile`  | Sets pedestal, extra white noise, pattern amplitude and read-out ramps as rows linked per real frame; pattern period and angle; temperature; a radiation rate map by sub-satellite point, and the frame-to-frame spread around it (`radiation.map_scale`, with the Poisson scatter of the counts taken out)                                                                                               |
| `refine`   | Scales the pattern amplitude by real/simulated medians; corrects the radiation map cell by cell from twins matched in time, position, exposure and temperature                                                                                                                                                                                                                                            |
| `twins`    | Simulates frames that take only time, spacecraft state, exposure and temperature from real frames; everything else is drawn from the profile. With `--ucac4` and a pointing in the keyword map, twins show the real sky: catalogue stars at the real pointing, trailed at the recorded tracking rate (`trail_rate_arcsec_per_s`). Without it they use random stars, which saturate more than a real field |
| `report`   | Real and simulated distributions side by side, with the Kolmogorov–Smirnov distance per quantity                                                                                                                                                                                                                                                                                                          |
| `score`    | Matches a detector's source list (FITS table with `X`, `Y`, such as Fayde's; or CSV/JSON with `x`, `y`) to a truth file: completeness by magnitude, false detections (those on streaks, targets, hot or flickering pixels and radiation hits are counted apart), centroid error and bias, and error over each star's Cramér–Rao bound                                                                     |

## `learn` options

`skytwin-calibrate learn INPUTS -o profile.json` takes frames as directories, files, glob patterns or `.txt` lists. `skytwin-calibrate learn --help` lists every option with its default, and the [command-line reference](https://shaneryall.github.io/SkyTwin/dev/reference/cli/#skytwin-calibrate-learn) shows the same.

## Files `learn` writes

| File                    | Holds                                                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `profile.json`          | The instrument profile, ready for `skytwin`                                                                                      |
| `profile.learn.md`      | Every setting with where it came from (*measured*, *header*, *default*, *not learned*) and the evidence, then the realism report |
| `profile.keywords.json` | The header keyword map it worked out; edit and pass back with `--keywords` if a choice is wrong                                  |

The profile carries its contract: `schema_version`, `sensor`, `learned` (provenance) and `catalog`, the catalogue it was calibrated against; a CSV is pinned by SHA-256 and row count. The [profile reference](https://shaneryall.github.io/SkyTwin/dev/reference/profile/#profiles-as-contracts) lists the keys.

## What `learn` leaves out

These are not learned, so they are left out of the profile and listed in the report: Sun and Moon glare, pointing motion (wobble, drift, jumps), CTI, flickering pixels and the field-dependent PSF. Add them by hand from the [profile reference](https://shaneryall.github.io/SkyTwin/dev/reference/profile/index.md).

Some settings need the data to vary. The flat field needs frame levels spanning at least 30 ADU; the temperature trend needs at least 3 °C. With too little variation, `learn` says so and leaves the block out.

`profile` keeps the blocks it does not fit from the base profile: smear, stray light, pointing motion, hot pixels, CTI, layout and header style.

## Keyword map

A keyword map names the header cards that hold each value. All entries are optional; with a time and a spacecraft state, refinement can match each real frame with a simulated twin. [`examples/keywords.json`](https://github.com/shaneryall/SkyTwin/blob/main/tools/calibrate/examples/keywords.json) reads Skytwin's own keywords.
