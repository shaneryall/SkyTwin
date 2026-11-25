# Training labels reference

The `label` record in the truth file, the classes and boxes built from it, and the files
`skytwin-export` writes for YOLO (Ultralytics) and COCO tools such as DETR.
[Export a training set](export-training-data.md) gives the steps;
[design/labels.md](design/labels.md) explains how the signal-to-noise ratio is computed and why
the rules below were chosen.

## Labels in the truth file

Each star, streak and target in the truth file has a `label` that records
what a detector could see of it.

| Field | Meaning |
| --- | --- |
| `path` | Pixel positions along which the source was drawn, pointing motion included. A still source has one point. A moving one has the path from its first to its last lit point, simplified to within 0.5 px and to at most 65 points, with both ends kept. Empty when nothing was drawn. |
| `snr` | Signal-to-noise ratio of an ideal matched filter, √(Σ sᵢ² / vᵢ) over the source's pixels on the image, where sᵢ is its expected signal in electrons after the flat field and vᵢ the pixel's variance. No detector can do better. |
| `peak_snr` | Highest single-pixel signal-to-noise ratio, max sᵢ / √vᵢ. |
| `extent_px` | Distance from `path` of the farthest pixel centre where the source is at least the pixel's noise σ, in pixels. |
| `saturated` | Whether a pixel where the source is at least the noise σ saturates, counting everything on that pixel: it reaches full well, or the level where the delivered pixel clips at the output's maximum, whichever is lower. |

## Classes

`skytwin.labels.objects` reads the labels of a frame. Stars are class
`star`. Targets and pixel streaks are class `satellite`. A source is a
streak when the farthest two points of its path are more than twice the
camera FWHM apart, and otherwise a point. With `by_form`, each class splits
into `star_point`, `star_streak`, `satellite_point` and `satellite_streak`.

## Boxes

The labelled region of an object is its path widened by a radius. The
radius is `extent_px` or the FWHM, whichever is larger, plus 0.5 px to
reach the edges of the outermost pixels. An object is labelled when its
`snr` is at least `min_snr`, 5 by default.

Skytwin puts pixel centres on whole numbers, while YOLO and COCO measure
from the outer corner of the first pixel. Labels therefore add 0.5 px to
each axis. `x` runs along a row of the image array and `y` down its rows.
Images are written with array row 0 at the top, so a FITS viewer that draws
row 0 at the bottom shows the same frame flipped vertically.

## Tiles

A large frame can be cut into tiles. A point is kept in a tile when at
least half of its box lies inside. A streak is kept when part of its path
lies inside. Its signal-to-noise ratio in the tile is its full value times
the square root of the share of its on-image length inside. Axis-aligned boxes are
clamped to the tile. A streak's rotated box runs along its part in the tile
from the first to the last point, with the half-width of its radius. Where
it crosses the edge of the tile, the edge cuts it, so the box holds all of
the streak's light inside the tile; a cut box has four corners but is no
longer a rectangle (YOLO-OBB and COCO's `rotated_box` give the corners).

## Export options

`skytwin-export SCENE OUT_DIR` takes these options:

| Option | Effect | Default |
| --- | --- | --- |
| `--frames` | Frames to render | the scene's `batch.frames`, else 1 |
| `--start` | First frame index | 0 |
| `--seed` | Replaces the scene's seed | the scene's |
| `--format` | `yolo`, `yolo-obb` or `coco`; repeat for several. `yolo` and `yolo-obb` both write `labels/`, so choose one. | `yolo` |
| `--classes` | Kinds to label, separated by commas: `star`, `satellite`, or `star,satellite` | `star,satellite` |
| `--by-form` | Split each kind into point and streak classes | off |
| `--min-snr` | Least matched-filter signal-to-noise ratio in the tile | 5 |
| `--min-visible` | Least share of a point's box inside a tile | 0.5 |
| `--min-radius` | Least box half-width before the half pixel, px | FWHM |
| `--streak-px` | Least span of a streak's path, px | 2 × FWHM |
| `--tile`, `--overlap` | Tile size and least overlap, px; the overlap is at least 0 and below the tile size | whole frame, 0 |
| `--val` | Share of frames for validation | 0.1 |
| `--image` | `asinh` (8-bit) or `raw16` (the delivered pixels in 16 bits) | `asinh` |
| `--workers` | Render threads | every core |
| `--keep-truth` | Also write each frame's truth file to `truth/` | off |
| `--overwrite` | Replace an earlier export in the output folder; without it, a folder that holds one is refused | off |

Frame k is the scene's frame k, with seed + k, so a set can be rebuilt or
extended exactly. The last `--val` share of the frames
forms the validation split, and all tiles of a frame stay in its split.

## Output files

The output folder holds:

```text
images/{train,val}/000123.png              000123_04.png for tile 4
labels/{train,val}/000123.txt              YOLO or YOLO-OBB
data.yaml                                  Ultralytics dataset file
annotations/instances_{train,val}.json     COCO
skytwin.json                               scene, options and counts
```

YOLO label lines are `class cx cy w h`, and YOLO-OBB lines are
`class x1 y1 x2 y2 x3 y3 x4 y4`, all as fractions of the tile size. COCO
category ids start at 1, image file names are relative to
`images/<split>/`, and each annotation carries a `skytwin` record with the
form, signal-to-noise ratio, saturation, rotated box and the source's place
in the truth file. COCOeval keeps 100 detections per image by default, and
a tile of stars can hold more. Raise `params.maxDets` when stars are a
class.

## Image stretch

The `asinh` image measures each pixel in units of the frame's noise,
z = (image − median) / σ, with σ = 1.4826 times the median absolute
deviation. z is clipped to [−2, 1000] and mapped through asinh(z / 3)
linearly onto 0–255. The sky sits at grey level 22, ±1σ of noise spans 11
to 34, a 5σ pixel is at 68, and a 50σ pixel at 148.

The `raw16` image keeps the delivered pixel values, so most of a frame sits
in a narrow band near its sky level. A reader that converts 16-bit PNG files
to 8 bits on loading, as OpenCV's `cv2.imread` does by default, leaves the
sky in a few grey levels and faint sources gone; give such a reader `asinh`
images, or load `raw16` files with `cv2.IMREAD_UNCHANGED`.

## PyTorch

`skytwin.torch.detection_target` turns a frame into the stretched image,
scaled to 0–1, and a target with `boxes` (`xyxy` in pixels, as torchvision
detectors take, or `cxcywh` as fractions, as DETR takes), `labels` and
`snr`. Frames hold different numbers of objects, so batch them with
`skytwin.torch.collate`.

## Limits

- Hot pixels and radiation hits are in the truth file
  (`instrument.hot_pixels`, `instrument.radiation.hits`) but are not
  exported as classes.
- Labels give boxes and paths. There are no per-pixel masks.
- The footprint leaves out smear and bleeding along columns. `saturated`
  marks the sources that bleed.
