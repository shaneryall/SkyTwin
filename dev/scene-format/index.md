# Scene, truth and header reference

What a scene file may hold, what the truth file records, and the FITS keywords Skytwin writes. Instrument profiles are described in [reference/profile.md](https://shaneryall.github.io/SkyTwin/dev/reference/profile/index.md).

## Scene file

A scene file is a JSON object. All its keys are optional: `seed`, `instrument` (a profile path or object), `camera` and `overrides` (merged into the profile), `scene`, and either `batch` or `sequence`.

`batch` makes independent frames for a training set. `frames` is the number of frames. Frame k uses seed + k and draws everything random anew: stars, streaks, noise, and every profile distribution. With `random_pointing`, each frame also gets a new pointing. Right ascension and roll are uniform, and declination is uniform on the sphere within `dec_deg` (default `[-90, 90]`).

`sequence` (`frames`, `cadence_s` start to start, and a `scene.time`) makes several frames in a row: frame k starts k × cadence later with seed + k, writes `name_000.fits`, `name_001.fits`, …, and the run writes `name.sequence.json` listing them. Detector maps (hot and flickering pixels, column offsets) stay the same in every frame. With `"only_visible": "<target>"`, frames are kept only when that target is sunlit and not blocked by the Earth; the run steps on at the cadence (up to `max_steps`, default 100 × frames) and the index records how many steps were skipped. Two settings make the pointing change slowly through a sequence (they need `scene.pointing` as RA/Dec): `pointing_walk` (`step_arcsec` per axis and `roll_step_deg`, σ per frame) moves the boresight and the header with it; `settling` (`offset_arcsec`, `time_s`) starts the boresight off the commanded pointing, in a direction drawn once, and converges on it as exp(−t / time_s), while the header reports the commanded pointing, so the header lags. The index lists each frame's walk and settling offsets. See the files in [`examples/`](https://github.com/shaneryall/SkyTwin/blob/main/examples). Unknown keys are errors.

- Pixel positions are 0-based, with pixel centres on whole numbers.
- Sky positions are ICRS degrees and need `scene.pointing`.
- `scene.catalog` reads `ucac4` (a UCAC4 directory) or `csv`: a local catalogue file such as a Gaia archive extract, with `path`, `columns` (default Gaia DR3 names: `source_id`, `ra`, `dec`, `phot_g_mean_mag`, `pmra`, `pmdec`, `parallax`, `bp_rp`) and `epoch_jyear` (default 2016.0). Stars move by proper motion to the observation epoch, carry their parallax, and use `bp_rp` as the colour index. With an instrument profile that names its catalogue, `{"path": "..."}` is enough: the profile supplies the kind, columns, epoch and magnitude limit (see [reference/profile.md](https://shaneryall.github.io/SkyTwin/dev/reference/profile/#profiles-as-contracts)).
- Roll is the position angle of the image +y axis, east of north, as in Fayde.
- Brightness is `flux_e` (total electrons over the exposure) or `mag` (in the camera band, through `camera.zeropoint_mag`, the magnitude giving 1 e⁻/s).
- `scene.stars` lists stars. Each has a pixel position (`x`, `y`) or a sky position (`ra_deg`, `dec_deg`), a brightness, and optionally `id`, `color_index`, `teff_k`, and `parallax_mas`. `parallax_mas` applies only to a sky position in a scene with a time.
- `scene.random_stars` (`count`, `min_flux_e`, `max_flux_e`) adds `count` stars placed uniformly over the image, with fluxes between `min_flux_e` and `max_flux_e` that follow N(>F) ∝ F^−0.6.
- `scene.streaks` lists streaks in pixels. Each has a `start` and an `end` (`[x, y]` at shutter open and at shutter close) and a brightness.
- `scene.time.start_utc` is the shutter-open time. With a time, catalogue stars move to mid-exposure and aberration is applied.
- `scene.observer` is `"geocenter"` (default), `{"tle": [line1, line2]}`, `{"state_vector": {...}}` or `{"ephemeris": [...]}`, all in GCRS km and km/s, or a ground site: `{"site": {"latitude_deg", "longitude_deg", "height_m", "dut1_s", "refraction", "pressure_hpa", "temperature_c", "extinction_mag_per_airmass"}}` (WGS84, turning with the Earth at UT1 = UTC + `dut1_s`; pressure and temperature default to the standard atmosphere at the site's height). A ground site needs a pointing: every source is refracted towards the zenith and dimmed by the extinction at the boresight's airmass, and what is below the horizon is hidden. See [`examples/ground-tracks-meo.json`](https://github.com/shaneryall/SkyTwin/blob/main/examples/ground-tracks-meo.json).
- `scene.pointing.quaternion` is `[w, x, y, z]`, rotating camera vectors into GCRS. The camera frame has +z along the boresight, +x along increasing columns and +y along decreasing rows.
- `scene.pointing.track` names a target to follow, with `roll_deg`.
- `scene.targets` lists targets: `name`, `orbit` (same forms as the observer) and `brightness`: `{"mag": m}` (when fully sunlit), `{"lambertian_sphere": {"diameter_m": d, "albedo": a}}`, or `{"facets": {...}}`: flat panels, each with a body-frame `normal`, `area_m2`, `diffuse_albedo`, `specular_albedo` and `specular_exponent`, in an `inertial` attitude (quaternion) or `tumbling` about an axis at `rate_deg_s`. Panels glint when their normal bisects the Sun and the observer; the path tracer starts fine enough to catch the shortest glint. See [`examples/tumbling-glint.json`](https://github.com/shaneryall/SkyTwin/blob/main/examples/tumbling-glint.json). Set `"earthshine": true` or `"moonshine": true` on a target to add sunlight reflected by the Earth (Bond albedo 0.306) or the Moon (0.11), each treated as a sunlit Lambertian sphere seen from the target (a far-field approximation, rough within a few Earth radii). The truth records the share that is direct sunlight. `scene.target_file` reads a two- or three-line element file, with one brightness model for all of it.
- `camera.sun_mag` (default −26.74, V) sets the Sun's magnitude in the camera band, or `camera.passband` computes it, with an AB zeropoint, from a throughput curve and aperture (see [reference/profile.md](https://shaneryall.github.io/SkyTwin/dev/reference/profile/#camerapassband)).
- `camera.earth_limb_margin_km` (default 100) is the height below which a line of sight counts as blocked.
- `scene.random_streaks` (`count`, `length_px`, `mag`, each a fixed value or a distribution) adds straight streaks anywhere on the image in random directions, drawn anew each frame, for training sets.
- `scene.header_bias_arcsec` (`[east, north]`) is the reported pointing minus the true pointing, as from a header that lags the boresight. It adds to the profile's random header error.

## Truth file

Each star records `photons` (reaching the detector, before `camera.quantum_efficiency` and the flat) and `detected_e` (electrons expected after the flat, averaged over the pixels the star's light falls on); each target records `photons` too.

For each star in a static frame (no tracking, no pointing motion), the truth file gives the Cramér–Rao bound on its centroid (`crlb_px`) and flux (`crlb_flux_e`): the best precision any method could reach, from the actual PSF and noise at that spot. `tools/calibrate`'s `score` command uses it to grade detectors.

When stars trail (tracking a target, or pointing drift, wobble and jumps), each star also gives `trail_start` and `trail_end`: its pixel position at shutter open and close. With `x`, `y` (the mid-exposure position) they let a solver's along-trail and cross-trail errors be measured directly.

Pixel positions in the truth are where each source was drawn. Pointing motion is measured from the image at mid-exposure, so it leaves `x`, `y` in place and moves the trail ends, a streak's `start` and `end`, and a target's `start` and `end` samples. A streak's `start_radec` and `end_radec` are sky positions, which pointing motion does not change.

For each target, `start`, `mid` and `end` give the pixel position, the astrometric RA/Dec (light-time corrected; what a reduction against the stars reports, consistent with the header WCS), range, phase angle, lit fraction, whether the Earth blocks it, and magnitude. `flux_e` is the signal drawn and `on_image` says whether any lit part of the path falls on the image, and `light_curve` gives the signal rate along the traced path.

Every star, streak and target also has a `label` for detector training: the path along which it was drawn, its matched-filter and peak signal-to-noise ratios against the frame's noise, how far from the path it stands above the noise, and whether it saturates. See [labels.md](https://shaneryall.github.io/SkyTwin/dev/labels/index.md).

## Header keywords

| Keyword                                                              | Meaning                                                                                                           |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `DATE-OBS`, `DATE-BEG`, `DATE-AVG`, `DATE-END`, `MJD-OBS`, `MJD-AVG` | Exposure times, UTC                                                                                               |
| `OBS_X` … `OBS_VZ`                                                   | Observer GCRS position (km) and velocity (km/s) at mid-exposure                                                   |
| `SITELAT`, `SITELONG`, `SITEELEV`, `AIRMASS`                         | Ground site (WGS84 degrees, metres) and airmass at the boresight                                                  |
| `ATT_RA`, `ATT_DEC`, `ATT_ROLL`, `ATT_QW` … `ATT_QZ`                 | Apparent boresight and camera quaternion                                                                          |
| `CRVAL`, `CRPIX`, `CD`                                               | TAN WCS to astrometric (catalogue) positions; TAN-SIP with `A`/`B`/`AP`/`BP` terms when the camera has distortion |
