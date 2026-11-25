# Instrument profile reference

Every key an instrument profile may hold. Every block is optional, and leaving one out switches
that defect off. [Use and override an instrument profile](../override-a-profile.md) shows how a
scene names a profile and changes it; [design/realism.md](../design/realism.md) explains the
models.

## Values

Wherever a defect block takes a number, it also takes a distribution; the numbers in the
[`camera`](#camera) block are fixed. A defect setting such as `pedestal_adu` takes any of these
forms:

```json
"pedestal_adu": 24
"pedestal_adu": { "normal": [24, 8] }
"pedestal_adu": { "uniform": [12, 78] }
"pedestal_adu": { "log_uniform": [12, 78] }
"pedestal_adu": { "empirical": [12, 18, 24, 24, 31, 78] }
```

To keep settings that go together together, draw them from the same row:

```json
"pedestal_adu":    { "empirical_linked": { "group": "real_frame", "values": [24, 31, 78] } },
"extra_white_adu": { "empirical_linked": { "group": "real_frame", "values": [26, 30, 43] } }
```

Every setting in one group takes the entry at the same index, so a frame that
was measured warm and noisy stays warm and noisy. Lists in one group must
have the same length.

A distribution is drawn once per frame from the frame's seed, in its own
stream, so changing a profile setting does not change the pixel noise. Detector maps
(hot pixels, bad columns, flat field) use a separate, fixed `detector_seed`,
so they stay the same from frame to frame, as on a real chip.

## Profiles as contracts

A profile is one JSON file holding everything tuned to a sensor, so it can
be handed to someone else and used as is. Learned profiles also say what
they are and what they need:

| Key | Holds |
| --- | --- |
| `schema_version` | Profile format (1) |
| `sensor` | `name`, `platform`, `band_nm` `[from, to]`, `notes` |
| `catalog` | The star catalogue the profile was calibrated against: `name`, `kind` (`csv` or `ucac4`), `mag_limit`, and for a CSV `columns`, `epoch_jyear`, `sha256`, `rows`; `source` says how to get or rebuild it |
| `learned` | Provenance: tool, version, commit, date, frames, inputs |

The zeropoint, colour term and magnitude limit hold for that catalogue
only. A scene that gives only the catalogue file gets the rest from the profile (anything the
scene gives wins). A catalogue of the other kind is refused unless the scene sets
`"ignore_profile": true`. Each run's provenance file records the catalogue used (with the CSV's
SHA-256) beside the one the profile names.

## Profile layout

Every key below is optional. Units are in the key names.

| Key | Settings |
| --- | --- |
| `name`, `description` | Name (written to the truth file) and free text |
| `detector_seed` | Seed for everything fixed to the chip: column offsets, hot and flickering pixels, pixel response, dust (default 0) |
| `camera` | Image size, pixel scale and parity, PSF core, exposure, zeropoint, sky, gain, read noise, bias, full well, pixel pitch, distortion, colour term, passband |
| `sky_e_per_s` | Diffuse light per pixel, drawn per frame (replaces `camera.sky`; a zodiacal table still takes precedence) |
| `processing` | What the ground pipeline did: bias subtraction, pedestal, clipping, output type |
| `noise` | Extra white noise, coherent patterns, row and column offsets, ramps, serial lag, fixed and smooth fields |
| `temperature_c` | CCD temperature per frame |
| `dark` | Dark current at a reference temperature and how it scales with temperature |
| `hot_pixels`, `rts_pixels` | Hot pixels, and flickering (random telegraph) pixels, fixed to the chip |
| `radiation` | Hit rate, its map by sub-satellite position, charge per µm of track, charge spread |
| `cti` | Serial and parallel charge-transfer inefficiency |
| `readout_faults` | Rows lost or garbled in read-out |
| `psf` | Wings and a power-law tail beyond the core, and growth across the field |
| `flat` | Vignetting, pixel response non-uniformity, dust shadows, field stop |
| `smear`, `saturation` | Smear while charge moves; what happens above full well |
| `pointing` | Wobble, drift, jumps, oscillations and roll during the exposure, unresolved jitter, and the error of the header pointing |
| `stray_light`, `zodiacal` | Light from the Earth, Sun and Moon by angle; zodiacal light table |
| `detector` | Where the image sits on the chip: chip size, window, binning |
| `header` | FITS header written in the instrument's style |
| `sampling` | How finely moving sources are sampled |
| `schema_version`, `sensor`, `catalog`, `learned` | What a learned profile is and needs (see [Profiles as contracts](#profiles-as-contracts)) |

## Blocks

### `processing`

| Key | Meaning | Default |
| --- | --- | --- |
| `bias_subtracted` | The bias level was removed before delivery | `true` |
| `pedestal_adu` | Offset added after processing | 0 |
| `output` | Pixel type written: `u16`, `i16`, `i32` or `f32` | `u16` |
| `clip_min_adu`, `clip_max_adu` | Values outside are set to these | the output type's range |

`u16` (0 to 65535) clips everything below 0, as many delivered products do.
To keep the negative side of a bias-subtracted image, use `i16`, `i32` or
`f32` and leave `clip_min_adu` unset. `f32` also keeps fractional values;
the integer types round.

### `noise`

| Key | Meaning |
| --- | --- |
| `extra_white_adu` | White Gaussian noise on top of the camera read noise, ADU RMS |
| `patterns` | Coherent plane waves `A cos(2π(x cos θ + y sin θ)/P + φ)`, each with `period_px`, `angle_deg`, `amplitude_adu`; φ is random per frame |
| `row_noise_adu` | A new offset per row each frame, ADU RMS |
| `column_noise_adu` | An offset per column fixed to the detector (`detector_seed`), ADU RMS |
| `ramp_y_adu`, `ramp_y_adu_per_s` | Linear ramp from bottom to top: a fixed read-out part plus a part growing with exposure, ADU |
| `ramp_x_adu` | Linear ramp from left to right, ADU |
| `serial_lag` | Slow settling of the read-out chain: `coefficient` `a` (0 to below 1) and `toward` (`left`, the default, or `right`, the side trails extend to). See below |
| `fixed_field` | Additive structure that is the same every frame (a curved read-out ramp, glow near an edge, a bias pattern): `values` (ADU, rows of nodes), `spacing_px`, `origin_px` (pixel of node `[0][0]`); interpolated bilinearly, held at the edge value beyond the grid |

`sky_e_per_s` (top level, any per-frame value) is diffuse light per pixel
that replaces `camera.sky`: unlike `processing.pedestal_adu` it passes
through the flat (vignetting, field stop) and carries shot noise.

### `noise.serial_lag`

A single-pole low-pass along each row of the read-out signal, in read
order: out = (1 − a) · in + a · previous out. It acts on charge and read
noise alike, so it trails every source by aᵈ at `d` pixels and correlates
the noise along rows (lag-k correlation aᵏ) but not down columns. CTI moves
charge only and leaves the noise white; this is the analogue chain (a slow
preamplifier or video filter). Flux is kept, and white noise variance
falls by (1 − a)/(1 + a).

### `temperature_c`

The CCD temperature, °C, drawn per frame. Scales `dark` and `hot_pixels`.

### `dark`

| Key | Meaning |
| --- | --- |
| `e_per_s_at_ref` | Dark current at the reference temperature, e⁻/pixel/s (replaces `camera.dark_e_per_s`) |
| `ref_temp_c` | Reference temperature, °C |
| `doubling_c` | Optional: double every this many °C instead of following the silicon band-gap law |

### `hot_pixels`

| Key | Meaning |
| --- | --- |
| `density_per_mpix` | Hot pixels per million pixels |
| `e_per_s_at_ref` | Each pixel's dark rate at the `dark` reference temperature, drawn once from `detector_seed` |

The map is fixed to the detector; rates scale with the frame temperature
like dark current.

### `rts_pixels`

Flickering pixels: dark current that switches at random between a low and
a high level (random telegraph signal), common after radiation damage.

| Key | Meaning |
| --- | --- |
| `density_per_mpix` | Flickering pixels per million chip pixels (fixed by `detector_seed`) |
| `high_e_per_s_at_ref` | Extra rate in the high state at `ref_temp_c`, e⁻/s; scaled with temperature like dark current |
| `mean_high_s`, `mean_low_s` | Mean time in each state at `ref_temp_c` |
| `activation_high_ev`, `activation_low_ev` | Arrhenius activation energies of the switching: time constants shrink as the chip warms; different values change the fraction of time spent high |

Each frame simulates the switching over the exposure. The truth file lists
the pixels that were on and their mean rate.

### `radiation`

| Key | Meaning | Default |
| --- | --- | --- |
| `rate_per_cm2_s` | Hit rate when there is no map or no orbiting observer | |
| `rate_map` | Rate table by sub-satellite latitude and longitude (`lat_edges_deg`, `lon_edges_deg`, `rate_per_cm2_s` rows) | none |
| `map_scale` | Factor on the map, drawn per frame: an empirical distribution gives storms and belt edges, frames far above their cell's usual rate | 1 |
| `collect_time_s` | Time hits accumulate | the exposure |
| `readout_s` | Read-out time: each row keeps collecting hits until it is read, so hit density rises linearly to the last row read, which gets this much more time | 0 |
| `first_row` | Row read first: `bottom` (row 0) or `top` | `bottom` |
| `depth_um` | Sensitive silicon depth: sets track length from the incidence angle | 15 |
| `e_per_um` | Most probable charge per µm of track | 80 |
| `landau_scale` | Relative width of the charge straggling (Moyal) | 0.15 |
| `diffusion_px` | Charge spread, Gaussian σ | 0.4 |

The sub-satellite point needs `scene.time` and an orbiting `scene.observer`.
State vectors may be Earth-fixed (`"frame": "itrs"`), as spacecraft headers
often give them. The truth file lists every hit and the sub-satellite point.

### `readout_faults`

Rows lost or garbled after digitisation, as a link or memory fault leaves
them.

| Key | Meaning | Default |
| --- | --- | --- |
| `probability` | Fraction of frames with any corruption | |
| `bands` | Bands in a corrupted frame | 1 |
| `band_rows` | Rows per band | 1 |
| `kind_weights` | Relative weights of `[noise, dropped, saturated]`: salt and pepper (each pixel at the low clip or anywhere up to the high one), rows at the low clip, rows at the high clip | `[1, 1, 1]` |
| `truncate_probability` | Chance that a corrupted frame is also cut short: every row from a random row to the last reads at the low clip | 0 |
| `noise_low_fraction` | In a `noise` band, the share of pixels read at the low clip | 0.5 |
| `noise_rms_adu` | In a `noise` band, the other pixels: unset, uniform over the whole output range (salt and pepper); set, the row's own value plus Gaussian noise of this RMS (drawn per band). With float output and no clip, garbled rows span the frame's own values | unset |

The truth file lists each band (`readout_bands`: `first`, `last`, `kind`).

### `cti`

| Key | Meaning |
| --- | --- |
| `serial`, `parallel` | Each with `loss_per_transfer` and `trail_px` (mean release distance) |
| `amp_right`, `register_bottom` | Where charge is read out; trails point away from it |

### `noise.smooth_field`

Smooth random background structure, new every frame: Gaussian values on a
grid `scale_px` apart, interpolated with cubic B-splines, then set to zero
mean and exactly `rms_adu` RMS over the frame. Real frames carry it from
unresolved sources, scattered light and read-out drifts; without it,
simulated backgrounds are too smooth.

| Key | Meaning |
| --- | --- |
| `rms_adu` | RMS over the frame, ADU |
| `scale_px` | Grid spacing (feature size), pixels |

### `camera`

The numbers in `camera` are fixed; they don't take a distribution.

| Key | Meaning | Default |
| --- | --- | --- |
| `width`, `height` | Image size, pixels | 1024, 1024 |
| `pixel_scale_arcsec` | Pixel scale, arcsec per pixel | 3.0 |
| `parity` | Handedness on the sky: `negative` (north up, east left) or `positive` (mirrored) | `negative` |
| `optical_axis_px` | Pixel `[x, y]` (0-based) on the optical axis, where the pointing is | image centre |
| `distortion` | Radial distortion about the optical axis: `k1`, `k2` (coefficients of ρ² and ρ⁴, ρ in units of `radius_px`), `radius_px`; negative `k1` is barrel | none (pure TAN) |
| `fwhm_px` | PSF core full width at half maximum, pixels | 2.5 |
| `exposure_s` | Exposure time, s; 0 makes a bias frame, which can hold no stars, streaks or targets | 10.0 |
| `zeropoint_mag` | Magnitude giving 1 e⁻/s in total, all losses folded in | 20.0 |
| `sky` | Sky background: `{"e_per_s": …}` per pixel or `{"mag_per_arcsec2": …}` | `{"mag_per_arcsec2": 22.5}` |
| `dark_e_per_s` | Dark current, e⁻ per pixel per second (replaced by `dark`) | 0.1 |
| `read_noise_e` | Read noise, e⁻ RMS | 8.0 |
| `gain_e_per_adu` | Gain, e⁻ per ADU | 1.5 |
| `bias_adu` | Bias level, ADU | 1000.0 |
| `full_well_e` | Full well, e⁻; pixels clip here before digitising | 60,000 |
| `pixel_pitch_um` | Pixel pitch, µm; sets the area radiation hits fall on | 13.0 |
| `quantum_efficiency` | Electrons per incident photon, for photon counts in the truth; the zeropoint stays in electrons | 1.0 |
| `sun_mag` | The Sun's magnitude in the camera band, for light reflected by targets | −26.74 (V) |
| `earth_limb_margin_km` | Height above the surface below which a line of sight counts as blocked, km | 100 |
| `color_term` | Catalogue-to-camera colour term (below) | none |
| `passband` | Passband and collecting area (below); replaces `zeropoint_mag` and `sun_mag` | none |

### `camera.color_term`

Moves catalogue magnitudes to the camera band:
`m_camera = m_catalogue + coefficient × colour index`. `bands` names the two
catalogue bands of the index (first minus second, for example
`["apass_b", "apass_v"]`); catalogue stars get their index from those
bands, and `default_index` serves stars without both. Without `bands`,
stars take the catalogue's own colour column (a CSV catalogue's `bp_rp`).
Scene stars may give `color_index`. The truth file records each star's
index.

### `camera.passband`

A passband replaces `camera.zeropoint_mag` and `camera.sun_mag` with values
computed by synthetic photometry in AB magnitudes:

| Key | Meaning |
| --- | --- |
| `name` | Written as `FILTER` |
| `response` | `wavelength_nm` and `throughput` (0–1): optics × filter × quantum efficiency |
| `aperture_area_m2` | Collecting area |
| `reference` | Optional curve of the band magnitudes are given in, such as the catalogue's; stars move to the camera band by their temperature |
| `color_index` | `b_v` or `bp_rp`: what a star's `color_index` is, to get its temperature |
| `color_bands` | UCAC4 bands of that index (first minus second) |
| `default_teff_k` | Temperature for stars without one (default 5800 K) |

The zeropoint is the AB magnitude giving 1 e⁻/s,
`2.5 log10(A/h ∫ 3631 Jy T(λ) dλ/λ)`. Sources are blackbodies: stars at
their temperature (scene `teff_k`, else from the colour index, else the
default), the Sun at 5772 K and 1 au, so reflected light from targets
follows the band. A star's camera magnitude is its reference magnitude plus
the difference of the two bands' AB magnitudes for its blackbody, tabulated
over 2000–50,000 K. Give either `color_term` or `reference`, not both. The
header adds `MAGSYS = 'AB'`; `MAGZERO` is the computed zeropoint.

### `psf`

| Key | Meaning |
| --- | --- |
| `wings` | Wider Gaussian components, each with `fraction` (share of the flux) and `fwhm_px`; the core (`camera.fwhm_px`) keeps the rest |
| `tail` | A power-law (Moffat) tail: `fraction`, `fwhm_px`, `beta` (wings fall as r^−2β, β > 1); expanded into ten Gaussians fitted to the profile out to 12 FWHM |
| `field.radial_growth`, `field.tangential_growth` | Fractional growth of every component's σ along and across the radius from the optical axis, reached at the image's half-diagonal (quadratic in radius, like coma and astigmatism) |

Every component is pixel-integrated and applies to stars, trails, streaks
and targets alike. A field-dependent PSF is an elliptical Gaussian.

### `flat`

| Key | Meaning |
| --- | --- |
| `vignetting` | `[a2, a4]`: response `1 + a2 ρ² + a4 ρ⁴`, ρ the distance from the optical axis in half-diagonals |
| `prnu_rms` | Pixel response non-uniformity, fractional RMS, fixed to each chip pixel |
| `dust` | Out-of-focus dust shadows (annular "donuts") fixed to the chip: `density_per_mpix`, `radius_px`, `depth` (fraction of light removed), `hole_fraction` (inner radius over outer) |
| `field_stop` | A circular aperture imaged on the chip: `radius_px` (half response), `edge_px` (Gaussian σ of the edge), `outside` (response beyond, as a fraction of inside), `centre_px` (default: the optical axis). Response `outside + (1 − outside) · ½ erfc((r − radius) / (√2 edge))` |

The flat multiplies light only. `skytwin … --write-flat` also writes the
noiseless flat as `name.flat.fits` (32-bit floats), for testing
flat-fielding. The truth file lists the dust shadows. With
`hole_fraction` 0 the shadows are solid dark blobs, as dust closer to
focus casts.

### `smear`

| Key | Meaning |
| --- | --- |
| `above_s_per_row` | Seconds each row's charge spends under a source while moving, for pixels above it (higher row numbers) |
| `below_s_per_row` | The same for pixels below it |

Without a shutter, light keeps falling while charge is flushed before the
exposure and transferred after it. A source of rate R e⁻/s then adds
R × dwell to every pixel of its column, on each side. Only light smears:
dark current and hot pixels are added afterwards.

### `saturation`

| Key | Meaning | Default |
| --- | --- | --- |
| `bleed` | Charge above `camera.full_well_e` spills along the column, half each way, conserving charge | `false` (excess lost) |
| `serial_overflow` | A share of the excess spills along the row first, as when the serial register overflows, drawing a horizontal line from a saturated star: `fraction` of the excess, exponential `length_px`, `toward` `left` (default) or `right` | none |

If the ADC saturates before the pixel's full well, as on some cameras, set
`processing.clip_max_adu` to the ADC limit and leave `bleed` off.

### `pointing`

Motion of the whole image during the exposure, and the error of the
pointing written to the header.

| Key | Meaning |
| --- | --- |
| `wobble_rms_arcsec`, `wobble_time_s` | Smooth random wander per axis (Ornstein–Uhlenbeck), RMS and correlation time (default 1 s) |
| `drift_arcsec_per_s` | Steady rate in a random direction, zero at mid-exposure |
| `drift_turn_deg_per_s` | Rate at which the drift direction turns, in a random sense: the image moves along a circular arc of radius rate / turn rate, and every star trails in the same arc |
| `roll_rate_deg_per_s`, `roll_centre_px` | Rotation of the image about a pixel (default: the image centre), in a random sense, zero at mid-exposure: stars trail in arcs centred there, longer the farther they are from it |
| `jump_rate_per_s`, `jump_arcsec` | Sudden offsets at random times, random directions |
| `blur_arcsec` | Unresolved fast jitter: a Gaussian σ added to every PSF component |
| `oscillations` | Sinusoids, each `amplitude_arcsec`, `period_s` and `angle_deg` on the image (default random per frame), random phase. The image dwells at the turning points: along a trail they draw dashes, across it a zigzag or, when fast, two parallel trails |
| `header.reference_px` | Pixel whose sky position the reported RA/Dec describe (default: image centre) |
| `header.error_arcsec`, `header.roll_error_deg` | Random error of the reported pointing (σ per axis) and roll |
| `header.epoch` | When the reported pointing applies: `start`, `mid` (default) or `end` of the exposure |

Motion moves stars, targets and streaks; it does not move radiation hits or
hot pixels, which belong to the detector. The path is measured from the
image at mid-exposure: whatever wobble, oscillation or jumps came before,
the image is then where it would be without motion, so the star positions
in the truth and the header WCS describe that moment. Trail ends, streak
ends and target samples in the truth are where each source was drawn,
motion included. The truth file keeps the offset path (thinned), every
jump, and the turn and roll rates. The header's `ATT_RA`, `ATT_DEC`,
`ATT_ROLL` carry the reported pointing; the truth file keeps both the true
and the reported pointing.

`camera.parity` (`negative`, the default, or `positive`) mirrors the image
for optics that flip it; roll stays the position angle of +y.

`camera.distortion` adds radial lens distortion about the optical axis: a
point the TAN projection puts `r` pixels from the axis lands at
`r (1 + k1 ρ² + k2 ρ⁴)`, ρ = r / `radius_px` (`k1` negative: barrel;
positive: pincushion). Every star, target, streak and truth position goes
through it, and the header WCS becomes TAN-SIP with order-5 `A`/`B` and
`AP`/`BP` terms fitted to the model over the disc of `radius_px` (use the
half-diagonal so the whole frame is covered).

### `stray_light`

A list, one entry per body. Needs `scene.time`, an orbiting
`scene.observer` and a pointing.

| Key | Meaning |
| --- | --- |
| `body` | `earth` (angle to the nearest limb), `sun` or `moon` (angle to the centre) |
| `angle_deg`, `e_per_s` | Level at the image centre against angle, e⁻/pixel/s; log-interpolated, held below the table, zero beyond it |
| `gradient` | How much brighter the bright edge is than the centre, across half the frame width (0.2 = 20%) |
| `toward_deg` | Direction of the bright side relative to the body's direction in the image: 0 towards it, 180 away from it |
| `dark_limb_factor` | Earth: factor when the nearest limb point is in darkness |
| `phase_scaled` | Moon: scale by the illuminated fraction (default true) |

The truth file records each source's angle, level, image direction and
whether the limb was lit (Moon: lit fraction).

### `zodiacal`

A table of zodiacal surface brightness, V mag/arcsec², with
`elongation_deg` (|λ − λ☉|), `latitude_deg` (|β|) and `v_mag_arcsec2` rows
by elongation. With a time and an observer it replaces `camera.sky`; the V
value is moved to the camera band through `camera.sun_mag`, since zodiacal
light has the Sun's colour. Skytwin ships no table: fill it from a
published reference such as Leinert et al. (1998), table 17.

### `sampling`

How finely moving sources (streaks, trails, targets) are cut into exact
straight pieces.

| Key | Meaning | Default |
| --- | --- | --- |
| `pixel_tolerance` | Largest distance between the true path and a piece, px | 0.01 |
| `flux_relative_tolerance` | Largest relative brightness error over a piece | 0.01 |
| `max_step_s` | Longest piece, s | none |
| `max_depth` | Deepest halving of a piece | 14 |

### `detector`

Where the delivered image sits on the chip.

| Key | Meaning | Default |
| --- | --- | --- |
| `chip_px` | Chip size, unbinned pixels | the image size |
| `window_origin_px` | Chip pixel of the image's first pixel, unbinned, 0-based | `[0, 0]` |
| `binning` | Binning `[x, y]` | `[1, 1]` |

`camera` describes the delivered image (its size, pixel scale and PSF are in
binned pixels). The layout moves the chip's own effects into place: hot
pixels belong to the chip, so two windows of one chip share them (binned
pixels sum theirs); radiation hits see the binned pixel area and track
length; CTI counts transfers from the window's real distance to the
amplifier and register. `camera.optical_axis_px` puts the pointing on a
given image pixel instead of the centre.

### `header`

| Key | Meaning | Default |
| --- | --- | --- |
| `wcs` | Write the WCS | `true` |
| `skytwin_keywords` | Write Skytwin's own keywords (times, observer, attitude, seed) | `true` |
| `roll` | `keyword`, `offset_deg`, `sign`: the roll in the instrument's convention, `offset + sign × roll` | none |
| `cards` | Extra cards in order: `key`, `value`, `comment` | none |
| `sexagesimal_carry` | Carry rounded seconds into the minutes. `false` copies a common formatter bug: 59.96 s is written `60.0`, about one frame in 600 per field | `true` |
| `wrong_wcs` | Write a wrong WCS on some frames, as another solver's failed solution: `probability`, `offset_arcmin` (tangent point moved, random direction), `roll_deg` (axes turned, either way), `scale_error` | none |

A card value is a number, a boolean, a string, or a template computed from
the frame: `{date_obs}`, `{jd_obs}`, `{mjd_obs}`, `{exposure_s}`,
`{ccd_temp_c}`, `{ccd_temp_k}`, `{ra_deg}`, `{dec_deg}`, `{roll_deg}`,
`{ra_hms}`, `{dec_dms}` (option `:space` for `hh mm ss.s`),
`{sub_lat_deg}`, `{sub_lon_deg}`, `{earth_limb_deg}`, `{binning_x}`,
`{binning_y}`. Option `:str` writes a number as a quoted string. A
template whose value the frame lacks (no time, no observer) is left out. A
card named like one of Skytwin's own keywords replaces it.

With `wrong_wcs`, the truth file keeps the true `wcs` and adds the
`header_wcs` that was written instead. A scene's `header_bias_arcsec`
(`[east, north]`) shifts the reported RA/Dec from the true pointing, on top
of `pointing.header.error_arcsec`: a header that lags the boresight.
