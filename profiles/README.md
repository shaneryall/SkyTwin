# Instrument profiles

Each file here is an instrument profile: everything Skytwin needs to make
frames that look like one camera's, including its defects. A scene names
one with `"instrument": "profiles/<file>.json"` and may change any part of
it with `overrides` (see
[docs/override-a-profile.md](../docs/override-a-profile.md)).

| Profile | Camera | Observer | Learned from | Source licence |
| --- | --- | --- | --- | --- |
| [`example-microsat.json`](example-microsat.json) | Illustrative small-satellite camera with every defect switched on | Space | Hand-written, round numbers | — |
| [`cassiope-fai-nir.json`](cassiope-fai-nir.json) | CASSIOPE Fast Auroral Imager, near-infrared channel, 256 × 256 | Space, polar orbit | 1,378 frames of the DUST dataset | CC BY 4.0 |
| [`wise-w1.json`](wise-w1.json) | WISE band W1 (3.4 µm), 1016 × 1016 single exposures | Space, 525 km Sun-synchronous orbit | 150 All-Sky frames | NASA/IPAC public data |
| [`astrianet-pl16803.json`](astrianet-pl16803.json) | FLI ProLine PL16803, 4096 × 4096, unfiltered, raw frames | Ground, New Mexico, tracking satellites | 159 frames of two nights | CC0 1.0 |
| [`ckoirama-cdk24.json`](ckoirama-cdk24.json) | FLI camera on a PlaneWave CDK24, 2048 × 2048 (binned 2 × 2), g′, raw frames | Ground, Chile, sidereal | 44 frames plus 99 calibration frames | CC BY 4.0 |

The learned profiles were made with `skytwin-calibrate learn` (see
[docs/learn-a-profile.md](../docs/learn-a-profile.md)). What it was told
rather than measured is a physical fact the frames do not show, and each
profile's `sensor.notes` says what: a pixel size the headers lack, WISE's
read noise and its array's lack of charge transfer, and the extinction where
the frames span too little airmass to fit it. Each records its sensor, the
catalogue its photometry holds for, its sources and the frames it was
learned from in the `sensor`, `catalog` and `learned` blocks. No source
images are included; see
[THIRD-PARTY-LICENSES.md](../THIRD-PARTY-LICENSES.md) for the citations.

## Using a learned profile

The learned profiles' photometry holds for Gaia DR3 G magnitudes with a
BP−RP colour, so a scene gives a Gaia CSV with Skytwin's column names
(`source_id, ra, dec, phot_g_mean_mag, pmra, pmdec, parallax, bp_rp`) that
covers its field:

```json
{
  "instrument": "profiles/wise-w1.json",
  "scene": {
    "time": { "start_utc": "2010-05-01T12:00:00Z" },
    "pointing": { "ra_deg": 83.8, "dec_deg": -5.4, "roll_deg": 0.0 },
    "catalog": { "path": "/data/gaia_dr3_g17.csv" }
  }
}
```

`skytwin-calibrate describe profiles/<file>.json` prints a profile's
summary and a scene to start from. The ground profiles' zeropoints are
above the atmosphere; give the scene a ground site (`scene.observer.site`)
with the extinction the profile's description names.

## How close they are

Each profile was checked by simulating twins of its real frames (same
exposure, time and trail rate, the catalogue's stars at the frame's
pointing where it is known, everything else drawn from the profile) and
measuring both sets with the same code. The table gives the
Kolmogorov–Smirnov distance between the real and simulated distributions:
0 is identical, above about 0.4 the two clearly differ.

| Quantity | WISE W1 | AstriaNet | Ckoirama |
| --- | --- | --- | --- |
| Background level | 0.12 | 0.19 | 0.16 |
| Background noise | 0.35 | 0.27 | 0.11 |
| Pattern noise amplitude | 0.19 | 0.32 | 0.18 |
| Sharp single-pixel events (radiation, hot pixels) | 0.07 | 0.30 | 0.14 |
| Background structure with stars masked | 0.33 | 0.25 | 1.00 (see the profile's known gap) |
| Area covered by sources | 0.37 | 0.54 | 0.77 |

## Profiles

### `cassiope-fai-nir.json`

The near-infrared channel of the Fast Auroral Imager on CASSIOPE, from the
DUST dataset (doi:[10.5281/zenodo.20255672](https://doi.org/10.5281/zenodo.20255672)):
256 × 256 frames of 0.1 s, about 380″ pixels, in level-1 DN. Learned from
1,378 frames of 22 sessions.

To convert the dataset's HDF5 files to FITS frames, use
[`tools/calibrate/examples/dust_to_fits.py`](../tools/calibrate/examples/dust_to_fits.py),
which needs `h5py`. It flips the pixels and crops them to the dataset's PNG
frame, copies in the dataset's Astrometry.net solution as the WCS, and
converts the spacecraft state to GCRS.
[`dust_png.py`](../tools/calibrate/examples/dust_png.py), which needs
Pillow, writes FITS frames, real or simulated, as 8-bit PNGs the same way
the dataset made its images.

### `wise-w1.json`

WISE's 3.4 µm band as IRSA serves its single exposures (Level 1b): dark,
flat and nonlinearity already corrected, 7.7 s, 2.75″ pixels. Learned from
150 frames spread evenly over the 2010 cryogenic survey, with spacecraft
states from JPL Horizons (`--ephemeris`).

Measured from the frames: the frame-to-frame pedestal, noise and pattern
noise; smooth background structure (its correction against twins of the
real sky came out 0.97, then 1.03); a 6.0″ PSF (WISE's published W1 PSF
is 6.1″); hot pixels; and the sharp events that survive the pipeline,
mapped by sub-satellite point, with a frame-to-frame spread of 0.44× to
3.6× the map (5th to 95th percentile). The zeropoint is 20.03 (1 e⁻/s in
Gaia G), the median of 80 frames' own, with colour term W1 = G − 1.75
(BP−RP) and 0.13 mag scatter per star over 14,089 stars; WISE's own
`MAGZP` converts to 19.97 in the same units.

Given to the learner: 18 µm pixels (`--pixel-pitch-um`), and read noise
11.4 e⁻ from photon transfer over 305 more high-latitude frames (the 150
hold no pairs close in time). Set by hand after learning: no smear and no
read-out hits, since HgCdTe arrays are read in place; the learner sees a
1.15× excess of hits in one half and would read it as 2.4 s of read-out.
Not modelled:
saturation (W1 saturates near magnitude 8, so stars brighter than about
G 9 come out fainter in real frames than the zeropoint predicts) and
masked pixels (1.2% of each real frame is NaN). Twins of the real sky come
out a little noisier than the real frames (KS 0.35): each frame's extra
noise was measured with its faint stars in it, and the twins add the
catalogue's stars again.

### `ckoirama-cdk24.json`

Raw frames of the camera on Ckoirama Observatory's 0.6 m PlaneWave CDK24,
taken for the BlueWalker 3 campaign: g′, 10 s, sidereal, 0.934″ pixels.

The calibration frames (`--calibration`) give the gain, 1.71 e⁻/ADU, and
read noise, 18.6 e⁻ (photon transfer on flat and bias pairs); dark current,
0.086 e⁻/s at −20 °C (10 s master darks less 2 s and 5 s ones, so the
0.64 e⁻ per dark that does not grow with exposure is left out); cosmic
rays, 0.026 /cm²/s with no read-out hits (transient events in 45 darks,
less 88 at pixels that fire in two or more darks, which are flickering
pixels); and 70 dust shadows from the master flat (outer radius 28–104
chip px, 0.09–0.36% deep, hole 0.38 of the radius). The science frames
give the pedestal, noise and pattern per frame, hot pixels (786 per Mpix
of the binned image above 2.9 e⁻/s), a 4.4″ PSF with a wide seeing halo,
vignetting (17% at the corners), and the plate scale and parity (Fayde
solved 30 of 30 frames).

The zeropoint is 23.01 above the atmosphere with colour term
g′ = G + 0.98 (BP−RP), from Gaia photometry through the 30 solutions,
0.18 mag per star; extinction is 0.31 mag per airmass, fitted over
airmass 1.45–3.54. Give the scene's site
`"extinction_mag_per_airmass": 0.309`.

Known gap: real frames carry more large-scale flat-field structure (the
square fall-off at the chip's corners, broad features, faint and very
large dust shadows) than the vignetting and dust model reproduces.

### `astrianet-pl16803.json`

Raw frames of the FLI ProLine PL16803 at the AstriaNet node at New Mexico
Skies, from the data published with PyCIS: unfiltered, 1.55″ pixels, the
telescope tracking satellites so the stars trail. Learned from 159 frames
of two nights: Starlink-1422 (66 × 0.1 s) and NAVSTAR 48 (93 × 3 s).

Measured from the frames: gain 1.09 e⁻/ADU and read noise 10.3 e⁻ (photon
transfer on frame pairs up to 11 s apart; the read noise from the darkest,
bias-only frames); the pedestal, noise and pattern per frame; hot pixels,
119 per Mpix above 2.9 e⁻/s, from the 3 s frames (their rates include the
dark charge a row collects while it waits to be read); and the plate scale
and parity.

The star trails, 64–66 px in the 3 s frames, give the PSF and the
photometry. Across a trail the tracking does not blur the PSF: from 2,935
trails, a 2.02 px core with halos (2.5 px, 3.9″, across the trail).
Photometry in boxes along the trails gives 21.45 at colour 0 through the
atmosphere, the median of the 65 frames of 3 s (the night's first frames
passed under cloud), and colour term G + 0.13 (BP−RP). It drifts about
0.3 mag from G 7 to G 14 (fainter stars read fainter), so it holds to
about 0.1 mag. The extinction is assumed, 0.2 mag per airmass: the 3 s
frames all sit near airmass 1.15, so it could not be measured. The
zeropoint above the atmosphere is 21.68.

Radiation is 0.22 hits/cm²/s and 4.4 s of read-out, bottom row first, from
events in the 3 s frames outside the star trails, less those at pixels
that fire in two or more frames. Every event outside the masks counts, so
the rate is an upper bound. The headers lack the pixel size, so 9 µm was
given to the learner.

The twins trail at each frame's rate across the real sky, so the area
covered by sources compares (7.4% real, 9.4% simulated). Their noise and
pattern amplitude come out a little high: each frame's extra noise was
measured with its faint stars in it, and the twins add the catalogue's
stars again.
