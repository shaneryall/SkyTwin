# How skytwin-calibrate learns a profile

Every number comes from measuring frames: real frames to set the profile,
and simulated frames, measured with the same code, to check and correct
it. Nothing about a particular instrument is built in. Its header names go
in a keyword map; its measured profile stays with you.

The steps for running it are in [learn-a-profile.md](../learn-a-profile.md), its commands in
the [calibrate reference](../reference/calibrate.md), and every option in the
[command-line reference](https://shaneryall.github.io/SkyTwin/latest/reference/cli/#skytwin-calibrate-learn).

## The learn pipeline

`learn` runs these steps in order:

1. Headers: finds exposure, time, temperature (°C or K), sub-satellite
   point, Earth-limb angle, spacecraft position and velocity (and their
   frame, preferring the exposure start), pointing, roll, tracking rate,
   gain, read noise, binning, pixel size and pixel scale, by scoring every
   card on its name and its comment. The pixel size is taken as after
   binning (as `XPIXSZ` is) unless its comment says otherwise.
2. Frames: keeps the most common size and binning; other modes are
   listed, to learn separately.
3. Camera: gain, read noise, pixel scale and exposure from headers;
   leading columns that get no light; pixel type; the saturation level
   (where frame maxima pile up).
   Then the read-out chain: a serial lag (noise correlated along rows
   as aᵏ, sources trailing one way: `noise.serial_lag`), and, when the
   headers give no gain, gain and read noise by photon transfer from
   pairs of frames with the same exposure, taken up to 1.5 times the
   archive's usual spacing apart (5 to 60 s): the clipped block variance of
   the difference against level, over the central half of the frame, with
   the lag's variance reduction undone. On frames with the bias still in,
   the read noise comes from the darkest frames.
4. Statistics: background, noise, ramps, pattern noise and sharp events
   per frame (`measure`). On frames whose bias sits below the clip, noise
   comes from quantiles above the clip, where the MAD would read 0. Rows
   the read-out corrupted (salt-and-pepper bands, dropped rows) are counted
   and kept out of the sharp events; each row's spread comes from
   neighbouring-pixel differences, so glare gradients and dense radiation
   storms are not mistaken for corruption. The odd/even row offset (even
   rows' level minus their odd neighbours') is measured per frame. The
   corrupted bands found (rows, and whether noise, dropped or saturated)
   become `readout_faults`.
5. Detector: hot pixels from the median, over 150 frames, of each pixel's
   excess over its neighbours, and how their rates follow temperature, from
   the longest exposures that number at least 10 frames (a row's dark charge
   while it waits to be read does not scale with the exposure, so short
   frames would inflate the rates); the flat field (a circular field stop if
   there is one, vignetting inside it, and pixel response non-uniformity)
   from how each pixel follows the frame level; the PSF from frames whose
   stars measure round (with any serial lag undone first), picked from all
   the inputs by their header tracking rate; frame-transfer smear from the
   column excess above and below bright unsaturated stars (corrected against
   simulated frames with `--skytwin`).
6. Pointing: when the frames carry their own celestial WCS (an
   archive's Astrometry.net solutions, say), at least 20 of them do, and
   `--skytwin` and a catalogue (`--catalog` or `--ucac4`) are given, stars
   are matched to the catalogue through it, and a joint
   fit over up to 80 frames (per frame an affine map of tangent-plane
   coordinates, shared radial distortion about the image centre) gives the
   pixel scale, parity, `camera.distortion` (kept when k1 is 5σ from
   zero) and the photometric zeropoint. This works where a fresh plate
   solve fails (fields tens of degrees wide, a few dozen bright stars).
   Otherwise, with `--ucac4` (a CSV catalogue is not enough) and Fayde
   (`pip install fayde`): plate
   solves of the frames with round stars give the true pixel scale and
   parity, the roll keyword's convention, the pixel the header RA/Dec
   refers to, the header's pointing errors, and the photometric zeropoint
   (aperture photometry of the matched catalogue stars, corrected to total
   flux with the fitted PSF).
7. Earth glare, with `--skytwin`: a tiny twin of each frame gives
   Skytwin's own geometry (angle from the limb, sunlit or dark limb,
   Earth's direction in the image); the background of sunlit-limb frames
   against angle gives the table, dark-limb frames the dark factor, and the
   frame ramps the gradient and its direction. Frames inside the glare zone
   leave the pedestal sampling so their glare is not counted twice.
   Then `noise.smooth_field`: smooth background structure with sources
   masked, from block-to-block covariances that hold no pixel noise. With
   `--skytwin` it is corrected against simulated frames, twins of the real
   sky when a catalogue and the frames' pointing are known (frames with
   their own WCS give it), since a real field's faint stars below the mask
   add to the covariance too.
8. Header style, so simulated frames carry the same keywords: constant
   cards copied, computable ones templated.
9. Profile blocks from the statistics (`profile`).
10. With `--skytwin`: two refinement passes, then simulated twins and a
   realism report. Add `--catalog` or `--ucac4` for twins of the real sky.

## Blocks fitted indirectly

`skytwin-calibrate learn` fits most blocks, the read-out faults among them, directly from frame
statistics. The blocks below are fitted indirectly.

### Background light, pedestal and fixed field

`skytwin-calibrate learn` splits each frame's level into `sky_e_per_s`, the
pedestal and `noise.fixed_field` when the flat has a field stop: block
medians are fitted as pedestal + fixed field + light × flat, the fixed
field kept orthogonal to a constant and to the flat's shape.

### Serial lag

`skytwin-calibrate learn` takes `a` as the ratio of the lag-2 to the lag-1
correlation of the background along rows (white noise added after the
filter lowers both alike), the side from where resolved stars carry more
flux, and undoes the lag before choosing frames and fitting the PSF, so the
PSF is the optical one.

### Smooth background field

`skytwin-calibrate learn` measures `noise.smooth_field` with sources masked: block medians
about a quadratic surface, whose covariances between neighbouring blocks
(lags 1 and 2) hold no pixel noise. Their ratio gives the scale and their
size the RMS, through a table made by measuring simulated fields the same
way.

### Colour term

`skytwin-calibrate learn` fits `camera.color_term` from forced photometry
when its catalogue carries a colour (a Gaia CSV): per-star zeropoint
against colour, clipped, kept when 5σ from zero.

## Choosing a catalogue

The catalogue sets the photometry, so use one whose magnitudes hold for the stars the camera
sees: UCAC4's own magnitudes saturate brighter than about 8, which a
wide-field camera's stars often are. The zeropoint comes from forced
photometry at every catalogue position (no detection bias), and `learn`
warns when it drifts with magnitude. With a colour column (`bp_rp`) it
also fits `camera.color_term`. Gaia DR3 lacks many stars brighter than
G ~ 3, often the brightest in a wide field;
[`add_bright_stars.py`](../../tools/calibrate/examples/add_bright_stars.py) adds them to
a Gaia CSV from the Yale Bright Star Catalogue (VizieR V/50).

## Calibration frames

With `--calibration`, `learn` also reads the camera's bias, dark and
flat frames. They are sorted by their header frame type (`IMAGETYP` and
kin), else by the folder or file name, and only those of the science
frames' size and binning are used. What they measure replaces what the
science frames would give:

- gain by photon transfer from pairs of flats at the same exposure, with
  the read noise from pairs of bias frames;
- the dark current from master darks of the longest and shortest exposure,
  so the bias and the charge a row collects while it waits to be read
  cancel (the part that does not grow with exposure is reported, not kept);
- the radiation rate from hits in the darks, each dark less the median of
  its night and exposure, leaving out hits at pixels that fire in two or
  more darks (flickering pixels); with two or more exposures, the hits at
  zero exposure give the read-out time;
- dust shadows from the master flat, found with ring-shaped matched
  filters and measured from their radial profiles, in chip pixels as
  Skytwin draws them.

Hot pixels stay the ones the science frames show; the darks' own are the
fallback when the science frames find none, and `profile.learn.md`
compares the two.

## Frames whose stars trail

When the stars in most sampled frames trail (a telescope tracking a
satellite, say), `learn` measures each frame's trail length and angle from
the frame's autocorrelation, which for segments of length L is a ridge
whose profile along the trail is a triangle of half-width L. Then:

- the PSF comes from the profile across isolated trails, which the motion
  does not blur; for a sum of round Gaussians the profile across is the
  same sum of one-dimensional Gaussians, so the fit gives `camera.fwhm_px`
  and `psf.wings` directly;
- frames with their own WCS are matched by trail centres, found with a
  filter along the trail, and the zeropoint comes from boxes along the
  trails, corrected for the PSF halo outside the box and in its
  background ring;
- radiation is counted outside the trails in frames of the longest
  exposure, leaving out events at a pixel that fires in two or more
  frames (hot and flickering pixels, a tracked target); the read-out time
  comes from how the hits split between the chip's halves, and the rate
  is an upper bound, since every other event counts too;
- each frame's twin trails at that frame's rate, so the smooth-field
  correction and the realism report compare like with like.

## Ground frames

Frames with an airmass, or the telescope's altitude, in their headers are
taken from the ground. Their zeropoint is fitted against airmass, frame by
frame, so the profile's zeropoint is the one above the atmosphere and its
description gives the extinction a scene's ground site should use (a site
applies the atmosphere itself). When the frames span too little airmass to
fit it, `--extinction` (mag/airmass, 0.2 if not given) moves the zeropoint
up from the frames' median airmass, and the report says it was assumed.
Forced photometry uses an aperture of 2.5 times the PSF's FWHM, with the
background 1.36 to 2 apertures out, so a wide PSF's wings do not land in
the background. With plate solves, a `--catalog` (a Gaia extract, say) is
measured through each solution, so the colour term and the extinction come
from it rather than from UCAC4's magnitudes.

## The Cramér–Rao bound in scores

Truth files of static frames (no tracking, no pointing motion) carry each
star's Cramér–Rao bound (`crlb_px`, `crlb_flux_e`), from the actual PSF and
per-pixel variance. A detector whose centroid errors have an RMS of 1 in
units of the bound reaches the best precision any method can.
