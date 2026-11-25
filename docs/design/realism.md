# Realism of simulated frames

Simulated frames are usually far cleaner than real ones. Detectors and
solvers tuned on clean frames then fail on real data. Skytwin's aim is to
reproduce a real instrument's defects closely enough that a detector cannot
tell the difference by its statistics.

## Principles

1. Every defect is a setting. Nothing about an instrument is hard-coded.
   An **instrument profile** (a JSON file, one per satellite or camera) holds
   every setting. A scene names a profile and may override any field.
2. Every defect can be switched off. Leaving a block out of a profile
   turns that defect off, so each one can be studied alone.
3. Settings vary from frame to frame. Any defect setting in a profile can be
   fixed or drawn per frame from a distribution, so a simulated set spreads
   the way a real archive does.
4. Profiles are measured, not guessed. A calibration step measures a real
   archive (pedestal, noise, pattern noise, hit rates by orbit position, hot
   pixels by temperature, gradients by Earth-limb angle, and more) and writes
   the profile.
5. Realism is checked. The same measurements run on simulated frames, and
   a report compares the two distributions;
   [profiles/README.md](../../profiles/README.md#how-close-they-are) gives the
   comparison for each shipped profile.

The repository ships a generic example profile and profiles learned from
public archives (a space-based near-infrared imager, WISE's 3.4 µm band and
two ground telescopes); [profiles/README.md](../../profiles/README.md) lists
them with their sources. A profile learned from your own instrument's frames
stays with you.

The keys are in the [instrument profile reference](../reference/profile.md), and
[calibration.md](calibration.md) describes how `skytwin-calibrate` learns them.

## Model notes

### Moving PSFs

A field-dependent PSF is an elliptical Gaussian. It stays closed-form when moving, because in
coordinates whitened by its covariance it is a round Gaussian moving in a straight line. A real
PSF's halo is usually fitted well by two or three Gaussians out to a few core widths. Fit the
components with pixel integration included: fitting point-sampled Gaussians to a stacked star
profile counts the pixel width twice and gives too wide a core.

### Passband colours

`camera.passband` treats every source as a blackbody. Blackbodies miss absorption bands, so
colour offsets for cool stars are rough (a few tenths of a magnitude across wide bands).
