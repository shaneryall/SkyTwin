# Command-line reference

Each command's options and defaults, as printed by its `--help` option in `skytwin 0.1.0`. For tasks and examples, see the guides.

## skytwin

```
usage: skytwin <scene.json> <out.fits> [--seed N] [--write-flat] [--frames N] [--jobs N] [--compact-truth]
       skytwin --version
Writes <out.fits> and <out>.truth.json.
The scene file may name an instrument profile: "instrument": "profiles/x.json".
--frames N renders N independent frames (a "batch"), on --jobs threads (default: all cores).
--compact-truth writes the truth JSON without indentation (about 30% smaller, faster).
```

## skytwin-export

```
usage: skytwin-export [-h] [--frames FRAMES] [--start START] [--seed SEED]
                      [--format {yolo,yolo-obb,coco}] [--classes CLASSES] [--by-form]
                      [--min-snr MIN_SNR] [--min-visible MIN_VISIBLE] [--min-radius MIN_RADIUS]
                      [--streak-px STREAK_PX] [--tile TILE] [--overlap OVERLAP] [--val VAL]
                      [--image {asinh,raw16}] [--workers WORKERS] [--keep-truth] [--overwrite]
                      scene out

Render frames of a Skytwin scene into a YOLO or COCO training set.

positional arguments:
  scene                 scene file (JSON)
  out                   output folder

options:
  -h, --help            show this help message and exit
  --frames FRAMES       frames to render (default: the scene's batch.frames, else 1)
  --start START         first frame index (default 0)
  --seed SEED           replace the scene's seed
  --format {yolo,yolo-obb,coco}
                        label format, repeatable (default yolo); yolo and yolo-obb share labels/,
                        so pick one
  --classes CLASSES     kinds to label, separated by commas: star, satellite or star,satellite
                        (default)
  --by-form             split each kind into _point and _streak classes
  --min-snr MIN_SNR     least matched-filter signal-to-noise (default 5)
  --min-visible MIN_VISIBLE
                        least share of a point's box in a tile (0.5)
  --min-radius MIN_RADIUS
                        least box half-width before the half pixel, px (default FWHM)
  --streak-px STREAK_PX
                        least span of a streak's path, px (default 2 × FWHM)
  --tile TILE           cut frames into tiles of this many pixels
  --overlap OVERLAP     least overlap between tiles, px (default 0)
  --val VAL             share of frames for validation (default 0.1)
  --image {asinh,raw16}
                        image encoding (default asinh)
  --workers WORKERS     render threads (default: every core)
  --keep-truth          also write each frame's truth to truth/
  --overwrite           replace an earlier export in the output folder (its images, labels,
                        annotations and truth)
```

## skytwin-calibrate

```
usage: skytwin-calibrate [-h]
                         {learn,check,describe,measure,psf,profile,refine,twins,report,score} ...

positional arguments:
  {learn,check,describe,measure,psf,profile,refine,twins,report,score}
    learn               learn a whole profile from sample frames, in one step
    check               check a local star catalogue against what a profile was calibrated with
    describe            what a profile is, what it needs, and a scene to start from
    measure             per-frame statistics of real or simulated frames
    psf                 stack isolated stars and fit a multi-Gaussian PSF
    profile             build profile blocks from statistics
    refine              simulate, measure and correct a profile in place
    twins               simulate twins of real frames and measure them
    report              compare real and simulated statistics
    score               score a detector's source list against a truth file

options:
  -h, --help            show this help message and exit
```

## skytwin-calibrate learn

```
usage: skytwin-calibrate learn [-h] -o OUT [--keywords KEYWORDS] [--skytwin SKYTWIN]
                               [--ucac4 UCAC4] [--catalog CATALOG] [--catalog-name CATALOG_NAME]
                               [--catalog-source CATALOG_SOURCE] [--sensor-name SENSOR_NAME]
                               [--platform PLATFORM] [--band-nm FROM TO] [--ephemeris EPHEMERIS]
                               [--extinction EXTINCTION] [--pixel-pitch-um PIXEL_PITCH_UM]
                               [--calibration CALIBRATION [CALIBRATION ...]] [--fayde FAYDE]
                               [--max-frames MAX_FRAMES] [--iterations ITERATIONS] [--twins TWINS]
                               [--work WORK]
                               inputs [inputs ...]

positional arguments:
  inputs                directories, FITS files, glob patterns or .txt lists

options:
  -h, --help            show this help message and exit
  -o, --out OUT         profile to write (a .learn.md report and .keywords.json go beside it)
  --keywords KEYWORDS   header keyword map (default: worked out from the headers)
  --skytwin SKYTWIN     skytwin binary: refine the profile and write a realism report
  --ucac4 UCAC4         UCAC4 directory: plate solves for scale, parity and header pointing
                        errors, and real-sky twins
  --catalog CATALOG     star catalogue for matching, photometry and real-sky twins instead of
                        --ucac4: a UCAC4 directory or a CSV (Gaia extract, Skytwin column names).
                        Use one with good photometry for the stars the camera sees: UCAC4's own
                        magnitudes saturate brighter than about 8
  --catalog-name CATALOG_NAME
                        name recorded for the catalogue in the profile (default: the file name)
  --catalog-source CATALOG_SOURCE
                        how to get or rebuild the catalogue, recorded in the profile
  --sensor-name SENSOR_NAME
                        sensor name recorded in the profile (default: INSTRUME/TELESCOP)
  --platform PLATFORM   spacecraft or mission recorded in the profile
  --band-nm FROM TO     passband recorded in the profile
  --ephemeris EPHEMERIS
                        spacecraft states for frames whose headers carry none: a CSV (time_utc,
                        x_km, y_km, z_km, vx_km_s, vy_km_s, vz_km_s; geocentric GCRS) or a JPL
                        Horizons vector table (geocentric, in km and km/s or in au and au/day)
  --extinction EXTINCTION
                        ground frames: extinction, mag/airmass, to move the zeropoint above the
                        atmosphere when the frames do not span enough airmass to fit it (default
                        0.2)
  --pixel-pitch-um PIXEL_PITCH_UM
                        unbinned pixel size, um, when the headers do not give it (radiation rates
                        per cm² need it)
  --calibration CALIBRATION [CALIBRATION ...]
                        bias, dark and flat frames of the same camera (directories, files, glob
                        patterns or .txt lists): gain, read noise, dark current, cosmic rays and
                        dust shadows from them
  --fayde FAYDE         fayde binary for the plate solves (default: on the PATH)
  --max-frames MAX_FRAMES
                        frames to sample from the inputs (default 1500)
  --iterations ITERATIONS
                        refinement passes (default 2)
  --twins TWINS         simulated twins per refinement pass (default 200); the realism report uses
                        up to 300
  --work WORK           working directory (default: beside the profile)
```

## skytwin-calibrate check

```
usage: skytwin-calibrate check [-h] --catalog CATALOG profile

positional arguments:
  profile            profile (JSON)

options:
  -h, --help         show this help message and exit
  --catalog CATALOG  the catalogue you will give scenes (CSV file or UCAC4 directory)
```

## skytwin-calibrate describe

```
usage: skytwin-calibrate describe [-h] profile

positional arguments:
  profile     profile (JSON)

options:
  -h, --help  show this help message and exit
```

## skytwin-calibrate measure

```
usage: skytwin-calibrate measure [-h] [--list LIST] [--keywords KEYWORDS] -o OUT [frames ...]

positional arguments:
  frames               FITS files or glob patterns

options:
  -h, --help           show this help message and exit
  --list LIST          file with one FITS path per line
  --keywords KEYWORDS  header keyword map (JSON)
  -o, --out OUT        statistics file to write (JSON)
```

## skytwin-calibrate psf

```
usage: skytwin-calibrate psf [-h] [--list LIST] [--skip-left-columns SKIP_LEFT_COLUMNS] -o OUT
                             [frames ...]

positional arguments:
  frames                FITS files or glob patterns

options:
  -h, --help            show this help message and exit
  --list LIST           file with one FITS path per line
  --skip-left-columns SKIP_LEFT_COLUMNS
                        columns at the left edge to leave out, such as overscan (default 0)
  -o, --out OUT         fit result to write (JSON)
```

## skytwin-calibrate profile

```
usage: skytwin-calibrate profile [-h] --base BASE [--max-background MAX_BACKGROUND] -o OUT stats

positional arguments:
  stats                 statistics of the real frames, from measure

options:
  -h, --help            show this help message and exit
  --base BASE           base profile (camera basics and hand-measured blocks)
  --max-background MAX_BACKGROUND
                        frames above this background (ADU) count as glare and are left out
                        (default: 1000, or twice the median frame background if higher)
  -o, --out OUT         profile to write
```

## skytwin-calibrate refine

```
usage: skytwin-calibrate refine [-h] --real REAL [--keywords KEYWORDS] [--skytwin SKYTWIN]
                                [--work WORK] [--iterations ITERATIONS] [--twins TWINS]
                                profile

positional arguments:
  profile               profile to correct in place

options:
  -h, --help            show this help message and exit
  --real REAL           statistics of the real frames
  --keywords KEYWORDS   header keyword map for simulated frames (default: none)
  --skytwin SKYTWIN     path to the skytwin binary
  --work WORK           working directory (default refine-work)
  --iterations ITERATIONS
                        refinement passes (default 2)
  --twins TWINS         simulated twins per pass (default 200)
```

## skytwin-calibrate twins

```
usage: skytwin-calibrate twins [-h] --real REAL [--keywords KEYWORDS] [--skytwin SKYTWIN]
                               [--work WORK] [-n N] [--ucac4 UCAC4] [--mag-limit MAG_LIMIT] -o OUT
                               profile

positional arguments:
  profile               profile (JSON)

options:
  -h, --help            show this help message and exit
  --real REAL           statistics of the real frames
  --keywords KEYWORDS   header keyword map for simulated frames
  --skytwin SKYTWIN     path to the skytwin binary
  --work WORK           working directory (default twins-work)
  -n N                  twins to simulate (default 300)
  --ucac4 UCAC4         UCAC4 directory: twins get catalogue stars at the real pointing
  --mag-limit MAG_LIMIT
                        faintest catalogue star drawn (default 15.5)
  -o, --out OUT         statistics of the twins to write (JSON)
```

## skytwin-calibrate report

```
usage: skytwin-calibrate report [-h] --real REAL --sim SIM [--title TITLE] -o OUT

options:
  -h, --help     show this help message and exit
  --real REAL    statistics of the real frames
  --sim SIM      statistics of the simulated frames
  --title TITLE  report title (default: Realism report)
  -o, --out OUT  report to write (Markdown)
```

## skytwin-calibrate score

```
usage: skytwin-calibrate score [-h] --truth TRUTH --sources SOURCES [--origin {0,1}]
                               [--radius RADIUS] [-o OUT]

options:
  -h, --help         show this help message and exit
  --truth TRUTH      truth file of the frame the detector ran on
  --sources SOURCES  FITS table, CSV or JSON with X/Y, X_IMAGE/Y_IMAGE or xcentroid/ycentroid
                     columns (any case)
  --origin {0,1}     pixel origin of the source list (default: 1 for X_IMAGE and for a FITS
                     table's X, 0 otherwise)
  --radius RADIUS    match radius, pixels (default 2)
  -o, --out OUT      write the result as JSON (and .md beside it)
```
