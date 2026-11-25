# Python API

The `skytwin` package renders frames in memory and turns their truth into labels. The [export guide](https://shaneryall.github.io/SkyTwin/0.1/export-training-data/index.md) shows the package in use; this page lists every public name, from the package's docstrings.

## `skytwin`

Skytwin from Python: simulated frames and exact labels, in memory.

```
import skytwin
sim = skytwin.Simulator("scene.json")      # a scene file, or a dict
for frame in sim.stream(count=1000):        # rendered ahead on all cores
    frame.image                             # numpy array (rows, columns)
    frame.truth["stars"], frame.truth["streaks"]
```

Frame k uses seed + k and draws everything random anew; with `batch.random_pointing` it also points somewhere new. A frame depends only on its index, so the same index always gives the same pixels, whatever the order or the number of workers.

### `Simulator`

A scene file loaded once, to render many frames.

`scene` is a path to a scene file or the same content as a dict. Instrument profile paths inside it are read relative to the working directory. `seed` replaces the file's seed.

#### `render(index=0, *, expected=False)`

Render frame `index` (thread-safe; the GIL is released).

#### `stream(start=0, count=None, *, workers=None, prefetch=None, expected=False)`

Frames `start` to `start + count - 1` in order, rendered ahead on `workers` threads (default: every core), at most `prefetch` finished frames waiting (default: 2 per worker). `count` defaults to the rest of the batch.

### `Frame`

Bases: `NamedTuple`

One simulated frame.

#### `index`

Index in the batch.

#### `seed`

Seed it was drawn with (the file's seed + index).

#### `image`

Pixels as delivered (the profile's output type), shape (rows, columns).

#### `truth`

Everything drawn: stars, streaks, targets, WCS, pointing, detector effects (the same as the `.truth.json` file).

#### `expected`

Noise-free electrons per pixel (float32), when asked for: a clean target for denoising or detection.

## `skytwin.labels`

Detector labels from a frame's truth: boxes, rotated boxes, YOLO and COCO.

Every star, streak and target in the truth carries a `label`: the path it was drawn along, the signal-to-noise of an ideal matched filter, the peak pixel signal-to-noise, how far from the path it stands above the pixel noise, and whether it saturates. :func:`objects` turns those into labelled objects; :func:`boxes` cuts them to a tile of the image (or the whole of it) as axis-aligned and rotated boxes; :func:`yolo_lines`, :func:`yolo_obb_lines` and :func:`coco_annotations` write them out.

Coordinates. Skytwin puts pixel centres on whole numbers: `(0, 0)` is the centre of `image[0, 0]`. YOLO and COCO measure from the outer corner of that pixel, so every label here is shifted by half a pixel. `x` runs along a row and `y` down the array; images are written with array row 0 at the top, so a FITS viewer that draws row 0 at the bottom shows the same frame flipped vertically.

Classes. Stars are `star`; targets and pixel streaks are `satellite`. With `by_form` each splits into `_point` and `_streak` by how far apart the farthest two points of its drawn path are: a streak spans more than twice the PSF FWHM. Pointing jitter and small jumps leave a source a point; tracking at a target's rate trails every star into a streak and keeps the target a point.

### `CLASSES = ('star', 'satellite')`

Class names, in YOLO order.

### `FORM_CLASSES = ('star_point', 'star_streak', 'satellite_point', 'satellite_streak')`

Class names when split by form, in YOLO order.

### `Obj`

One labelled source, in Skytwin pixel coordinates.

#### `kind`

`star` or `satellite`.

#### `form`

`point` or `streak`.

#### `path`

Drawn path, shape (n, 2), `[x, y]` with pixel centres on whole numbers.

#### `radius`

Half-width of the labelled region about the path, px (to pixel edges).

#### `snr`

Matched-filter signal-to-noise over the whole image.

#### `peak_snr`

Highest single-pixel signal-to-noise.

#### `saturated`

Whether it reaches full well.

#### `source`

Where it is in the truth: `stars[3]`, `streaks[0]` or `targets[0]`.

#### `name`

Class name with the form: `star_point`, `satellite_streak`, …

### `Box`

An object cut to a tile, in that tile's corner coordinates (pixels).

#### `xyxy`

Axis-aligned box `(x0, y0, x1, y1)`.

#### `corners`

Rotated box: four corners in order around it, inside the tile. A streak's box is a rectangle along it, cut by the tile's edge where it crosses one (see `boxes`).

#### `snr`

Matched-filter signal-to-noise of the part inside the tile (for a streak, the whole value scaled by the square root of the share of its on-image length inside).

### `class_names(by_form=False, kinds=CLASSES)`

Class names in YOLO order for the chosen kinds.

### `objects(truth, *, min_radius_px=None, streak_px=None)`

Every source in a frame's truth that has a drawn path.

`radius` is the label's `extent_px` or `min_radius_px` (default: the camera FWHM), whichever is larger, plus half a pixel to reach the edges of the outermost pixels. A source is a streak when the farthest two points of its path are more than `streak_px` apart (default: twice the FWHM), and otherwise a point.

### `boxes(objs, tile, image_size, *, min_snr=5.0, min_visible=0.5)`

Objects cut to `tile` = `(x0, y0, width, height)` in pixels of an image of `image_size` = `(width, height)`.

A point is kept when at least `min_visible` of its box area lies in the tile; a streak when part of its path does. Either needs a signal-to-noise of at least `min_snr` inside the tile. A point's box is its path widened by its radius; a streak's is the part of its path within a radius of the tile, widened by the radius; both are clamped to the tile. A streak's rotated box runs along its part in the tile from the first to the last point, cut by the tile's edge where it crosses one, so it holds all of the streak inside the tile; a point's is its box.

### `class_id(o, names)`

Index of an object's class in `names` (with or without its form), or None.

### `yolo_lines(bxs, names, tile_size)`

Ultralytics detection labels: `class cx cy w h`, normalised.

### `yolo_obb_lines(bxs, names, tile_size)`

Ultralytics oriented-box labels: `class x1 y1 x2 y2 x3 y3 x4 y4`, normalised.

### `coco_annotations(bxs, names, image_id, first_id)`

COCO annotations (`bbox` as `[x, y, w, h]`, category ids from 1), each with a `skytwin` record: form, signal-to-noise, saturation, rotated box and source in the truth.

### `tiles(width, height, size=None, overlap=0)`

Tiles `(x0, y0, w, h)` of at most `size` px covering the image, overlapping by at least `overlap` px, the last in each row and column flush with the edge. One tile (the image) without a size.

### `stretch(image, *, soft=3.0, top=1000.0)`

8-bit image for a detector: noise-normalised, then an asinh stretch.

The image is measured in units of its noise, `z = (image − median) / σ` with σ from the median absolute deviation, and `z` from −2 to `top` maps through `asinh(z / soft)` onto 0–255. A source's grey level then depends on its signal-to-noise alone, the same in every frame however bright the sky.

### `raw16(image)`

The image as delivered, as 16-bit unsigned (other types rounded and clipped).

### `write_png(path, image, level=3)`

Write a 2-D `uint8` or `uint16` array as a greyscale PNG, row 0 at the top.

## `skytwin.torch`

A PyTorch dataset of simulated frames, rendered on the fly.

```
from skytwin.torch import FrameDataset, collate, detection_target
ds = FrameDataset("scene.json", transform=detection_target)
loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4, collate_fn=collate)
for images, targets in loader:
    ...                                        # targets[i]["boxes"], targets[i]["labels"]
```

Each DataLoader worker renders its own share of the indices, on `threads` Rust threads. `transform(frame)` turns a :class:`skytwin.Frame` into whatever the model takes; by default it gives the image as a float32 tensor and the frame's index, which PyTorch's default collation batches. :func:`detection_target` gives the stretched image and boxes in the form torchvision detectors (`xyxy`) or DETR (`cxcywh`) take. Frames hold different numbers of objects, so batch them with :func:`collate`, which stacks the images and keeps one target per image; PyTorch's default collation cannot.

### `FrameDataset`

Bases: `IterableDataset`

Frames `start` to `start + count - 1` of a scene file's batch.

Under a DataLoader each worker renders one contiguous share of the indices, on `threads` Rust threads (default 1 per worker; every core without workers).

### `default_transform(frame)`

The image as a float32 tensor, and the frame's index.

The index, not the truth dict, so a plain `DataLoader` can batch the pairs; frames hold different numbers of objects, which PyTorch's default collation cannot stack. For labels, use :func:`detection_target` with :func:`collate`.

### `detection_target(frame, *, kinds=lab.CLASSES, by_form=False, streak_px=None, min_snr=5.0, box_format='xyxy')`

The frame as a detector's input and target.

The image is :func:`skytwin.labels.stretch` scaled to 0–1, shape `(1, rows, columns)`. The target holds `boxes` (`xyxy` in pixels from the image corner, or `cxcywh` as fractions of the image), class indices `labels` into :func:`skytwin.labels.class_names`, each object's `snr`, and the frame's `index`.

### `collate(batch)`

Stack the images of `(image, target)` pairs and keep the targets as a list.

## `skytwin.export`

Write a training set: images with YOLO, YOLO-OBB or COCO labels.

```
skytwin-export scene.json out/ --frames 5000 --format yolo --format coco --tile 640
```

Layout, as Ultralytics and COCO tools expect it::

```
out/images/{train,val}/000123.png              000123_04.png for tile 4
out/labels/{train,val}/000123.txt              YOLO or YOLO-OBB
out/data.yaml                                  Ultralytics dataset file
out/annotations/instances_{train,val}.json     COCO; file names relative to images/<split>/
out/skytwin.json                               what was rendered, with which options
```

Frame k is the scene's frame k (seed + k), so the set can be rebuilt or extended exactly. The last `--val` share of the frames (rounded up) is the validation split, and every tile of a frame stays in its frame's split. Images are the delivered pixels through :func:`skytwin.labels.stretch` (8-bit, noise-normalised asinh) or, with `--image raw16`, as delivered in 16 bits. Labels are those of :mod:`skytwin.labels`: objects at or above `--min-snr` (matched-filter signal-to-noise) inside each tile.

### `main(argv=None)`

Command-line entry point; returns the exit status.
