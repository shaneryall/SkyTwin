# Use and override an instrument profile

A scene renders through an instrument profile and can change any of the profile's settings without editing the profile file. In this guide, you render a frame through the example profile with its radiation switched off and its dark current drawn anew for each frame.

Before you start, you need:

- The `skytwin` command (see the [README](https://github.com/shaneryall/SkyTwin/blob/main/README.md)).
- A checkout of this repository, for `profiles/example-microsat.json`.

## Steps

1. Create `quiet-frame.json` and name the profile with `instrument`:

```
{
  "seed": 3,
  "instrument": "profiles/example-microsat.json",
  "scene": {
    "random_stars": { "count": 300, "max_flux_e": 500000.0, "min_flux_e": 300.0 }
  }
}
```

Without an `instrument`, the scene renders through a defect-free default camera.

1. Add an `overrides` object to change settings. It is a JSON merge patch (RFC 7386): objects merge key by key, and any other value replaces the profile's.

```
"overrides": { "camera": { "exposure_s": 3.0 } }
```

A top-level `camera` object in the scene is merged into the profile's camera the same way.

1. Switch a defect off by setting its block to `null`, which deletes it from the profile:

```
"overrides": { "camera": { "exposure_s": 3.0 }, "radiation": null }
```

1. To vary a setting from frame to frame, give a distribution where the profile has a number. The example profile fixes its dark current at 0.3 e⁻/s at −30 °C. To draw it for each frame instead, add this `dark` entry inside `overrides`, after `"radiation": null`:

```
"dark": { "e_per_s_at_ref": { "uniform": [0.1, 0.5] } }
```

The [profile reference](https://shaneryall.github.io/SkyTwin/dev/reference/profile/#values) lists every distribution and how to link settings so they are drawn together.

1. Render the frame from the repository root:

```
skytwin quiet-frame.json out/quiet-frame.fits
```

## Result

`out/quiet-frame.fits` has no radiation hits. `out/quiet-frame.truth.json` records the profile name and, under `instrument.settings`, every value drawn for the frame. Here, `radiation_rate_per_cm2_s` is 0, and `dark_e_per_s` is the dark current drawn from 0.1 to 0.5 e⁻/s and scaled to the frame's temperature.

## Use a learned profile with its catalogue

A learned profile names the star catalogue it was calibrated against, so the scene gives only the catalogue file:

1. Print the profile's summary and a scene to start from:

```
skytwin-calibrate describe profiles/wise-w1.json
```

1. Check that your copy of the catalogue matches the profile's (columns, rows, SHA-256):

```
skytwin-calibrate check profiles/wise-w1.json --catalog /data/gaia_dr3_g17.csv
```

1. Give the file in the scene: `"catalog": {"path": "/data/gaia_dr3_g17.csv"}`. Skytwin fills in the kind, columns, epoch and magnitude limit from the profile.

[profiles/README.md](https://github.com/shaneryall/SkyTwin/blob/main/profiles/README.md) lists the learned profiles and what each needs, such as a ground site's extinction.
