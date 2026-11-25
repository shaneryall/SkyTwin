# Your first training set[¶](#your-first-training-set)

In this tutorial you describe a camera with a few real-world defects, render a frame through it, look at the labels Skytwin gives every star and satellite streak, and export 200 frames as a YOLO training set. It takes about five minutes and needs no data of your own: every frame is made here.

In \[ \]:

Copied!

```
%pip install -q skytwin==0.1.0 matplotlib
```

%pip install -q skytwin==0.1.0 matplotlib

## Describe the camera[¶](#describe-the-camera)

A scene renders through an instrument profile. Here the profile is written inline: a 512 × 512 camera with 3″ pixels, plus three defects that real frames carry and clean simulations usually lack. Hot pixels and radiation hits look like faint stars to a detector; the pattern noise is a faint diagonal ripple. Values written as `uniform` or `log_uniform` are drawn anew for each frame, so the set spreads the way a real archive does.

The scene scatters 300 stars and one to three satellite streaks over the field.

In \[1\]:

Copied!

```
camera = {
    "width": 512, "height": 512, "pixel_scale_arcsec": 3.0, "fwhm_px": 2.0,
    "exposure_s": 5.0, "zeropoint_mag": 20.0, "sky": {"mag_per_arcsec2": 22.5},
    "dark_e_per_s": 0.5, "read_noise_e": 8.0, "gain_e_per_adu": 1.2,
    "bias_adu": 1000.0, "full_well_e": 70000.0, "pixel_pitch_um": 13.0,
}
instrument = {
    "name": "tutorial-camera",
    "camera": camera,
    "hot_pixels": {"density_per_mpix": 150, "e_per_s_at_ref": {"log_uniform": [0.5, 500]}},
    "radiation": {"rate_per_cm2_s": 2.0},
    "noise": {
        "extra_white_adu": {"log_uniform": [12, 40]},
        "patterns": [{"period_px": 5.0, "angle_deg": 40.0, "amplitude_adu": {"uniform": [3, 15]}}],
    },
}
scene = {
    "seed": 1,
    "instrument": instrument,
    "scene": {
        "random_stars": {"count": 300, "min_flux_e": 300.0, "max_flux_e": 500000.0},
        "random_streaks": {
            "count": {"uniform": [1, 3]},
            "length_px": {"uniform": [40, 200]},
            "mag": {"uniform": [8, 11]},
        },
    },
}
```

camera = { "width": 512, "height": 512, "pixel_scale_arcsec": 3.0, "fwhm_px": 2.0, "exposure_s": 5.0, "zeropoint_mag": 20.0, "sky": {"mag_per_arcsec2": 22.5}, "dark_e_per_s": 0.5, "read_noise_e": 8.0, "gain_e_per_adu": 1.2, "bias_adu": 1000.0, "full_well_e": 70000.0, "pixel_pitch_um": 13.0, } instrument = { "name": "tutorial-camera", "camera": camera, "hot_pixels": {"density_per_mpix": 150, "e_per_s_at_ref": {"log_uniform": [0.5, 500]}}, "radiation": {"rate_per_cm2_s": 2.0}, "noise": { "extra_white_adu": {"log_uniform": [12, 40]}, "patterns": \[{"period_px": 5.0, "angle_deg": 40.0, "amplitude_adu": {"uniform": [3, 15]}}\], }, } scene = { "seed": 1, "instrument": instrument, "scene": { "random_stars": {"count": 300, "min_flux_e": 300.0, "max_flux_e": 500000.0}, "random_streaks": { "count": {"uniform": [1, 3]}, "length_px": {"uniform": [40, 200]}, "mag": {"uniform": [8, 11]}, }, }, }

## Render a frame[¶](#render-a-frame)

`Simulator` loads the scene once; `render(k)` gives frame k, drawn with seed + k. The image comes back as the camera would deliver it, and the truth holds everything that was drawn.

In \[2\]:

Copied!

```
import skytwin

sim = skytwin.Simulator(scene)
frame = sim.render(0)
truth = frame.truth
print("image:", frame.image.shape, frame.image.dtype)
print("stars:", len(truth["stars"]), " streaks:", len(truth["streaks"]))
print("hot pixels:", len(truth["instrument"]["hot_pixels"]),
      " radiation hits:", len(truth["instrument"]["radiation"]["hits"]))
```

import skytwin sim = skytwin.Simulator(scene) frame = sim.render(0) truth = frame.truth print("image:", frame.image.shape, frame.image.dtype) print("stars:", len(truth["stars"]), " streaks:", len(truth["streaks"])) print("hot pixels:", len(truth["instrument"]["hot_pixels"]), " radiation hits:", len(truth["instrument"]["radiation"]["hits"]))

```
image: (512, 512) uint16
stars: 300  streaks: 2
hot pixels: 39  radiation hits: 8
```

Raw counts span a range far wider than a screen shows. `skytwin.labels.stretch` measures each pixel in units of the frame's noise and maps it through an asinh curve, so a source's grey level depends on its signal-to-noise ratio alone. It is the same stretch the exported images use.

In \[3\]:

Copied!

```
import matplotlib.pyplot as plt
from skytwin import labels

image8 = labels.stretch(frame.image)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(image8, cmap="gray")
ax.set_title("Frame 0, asinh stretch")
ax.axis("off")
plt.show()
```

import matplotlib.pyplot as plt from skytwin import labels image8 = labels.stretch(frame.image) fig, ax = plt.subplots(figsize=(7, 7)) ax.imshow(image8, cmap="gray") ax.set_title("Frame 0, asinh stretch") ax.axis("off") plt.show()

## Read the labels[¶](#read-the-labels)

Every star and streak in the truth carries a `label`: the path it was drawn along and its matched-filter signal-to-noise ratio against this frame's noise. `labels.objects` turns them into objects with a class and a form. A source is labelled for training when its ratio reaches 5, so faint stars a detector cannot find are not counted against it.

In \[4\]:

Copied!

```
from collections import Counter

objs = labels.objects(truth)
print(Counter(f"{o.name}, {'labelled' if o.snr >= 5 else 'too faint'}" for o in objs))
for o in objs:
    if o.kind == "satellite":
        print(f"{o.source}: {o.form}, signal-to-noise {o.snr:.0f}, path {len(o.path)} points")
```

from collections import Counter objs = labels.objects(truth) print(Counter(f"{o.name}, {'labelled' if o.snr >= 5 else 'too faint'}" for o in objs)) for o in objs: if o.kind == "satellite": print(f"{o.source}: {o.form}, signal-to-noise {o.snr:.0f}, path {len(o.path)} points")

```
Counter({'star_point, labelled': 202, 'star_point, too faint': 98, 'satellite_streak, labelled': 2})
streaks[0]: streak, signal-to-noise 101, path 2 points
streaks[1]: streak, signal-to-noise 130, path 2 points
```

`labels.boxes` cuts the objects to a tile, here the whole frame, and keeps those at signal-to-noise 5 or more. Stars get axis-aligned boxes; streaks also get a rotated box along their path, drawn here in orange. Hot pixels and radiation hits get no box: they are detector defects, not sky sources, and a detector should learn to ignore them.

In \[5\]:

Copied!

```
from matplotlib.patches import Polygon, Rectangle

h, w = frame.image.shape
kept = labels.boxes(objs, (0, 0, w, h), (w, h))
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(image8, cmap="gray")
for b in kept:
    x0, y0, x1, y1 = b.xyxy
    if b.obj.kind == "star":
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                               fill=False, color="cyan", lw=0.6))
    else:
        corners = [(x - 0.5, y - 0.5) for x, y in b.corners]
        ax.add_patch(Polygon(corners, fill=False, color="orange", lw=1.2))
ax.set_title(f"{len(kept)} labelled objects")
ax.axis("off")
plt.show()
```

from matplotlib.patches import Polygon, Rectangle h, w = frame.image.shape kept = labels.boxes(objs, (0, 0, w, h), (w, h)) fig, ax = plt.subplots(figsize=(7, 7)) ax.imshow(image8, cmap="gray") for b in kept: x0, y0, x1, y1 = b.xyxy if b.obj.kind == "star": ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, color="cyan", lw=0.6)) else: corners = [(x - 0.5, y - 0.5) for x, y in b.corners] ax.add_patch(Polygon(corners, fill=False, color="orange", lw=1.2)) ax.set_title(f"{len(kept)} labelled objects") ax.axis("off") plt.show()

## Export a training set[¶](#export-a-training-set)

`skytwin-export` renders frames 0 to 199 of the same scene on every core and writes them as PNG tiles with YOLO label files. The scene goes in a JSON file first. The last 10% of the frames become the validation split.

In \[6\]:

Copied!

```
import json
import subprocess

with open("first-set.json", "w") as f:
    json.dump(scene, f)

out = subprocess.run(
    ["skytwin-export", "first-set.json", "first-set", "--frames", "200", "--tile", "256"],
    capture_output=True, text=True, check=True,
)
print(out.stderr.strip().splitlines()[-1])
```

import json import subprocess with open("first-set.json", "w") as f: json.dump(scene, f) out = subprocess.run( ["skytwin-export", "first-set.json", "first-set", "--frames", "200", "--tile", "256"], capture_output=True, text=True, check=True, ) print(out.stderr.strip().splitlines()[-1])

```
200/200 frames
```

In \[7\]:

Copied!

```
from pathlib import Path

root = Path("first-set")
for split in ("train", "val"):
    n_img = len(list((root / "images" / split).glob("*.png")))
    n_lab = len(list((root / "labels" / split).glob("*.txt")))
    print(f"{split}: {n_img} images, {n_lab} label files")
print("classes:", [l for l in (root / "data.yaml").read_text().splitlines() if ": " in l and l.startswith("  ")])
first = sorted((root / "labels" / "train").glob("*.txt"))[0]
print(first.name, "begins:")
print("\n".join(first.read_text().splitlines()[:3]))
```

from pathlib import Path root = Path("first-set") for split in ("train", "val"): n_img = len(list((root / "images" / split).glob("\*.png"))) n_lab = len(list((root / "labels" / split).glob("\*.txt"))) print(f"{split}: {n_img} images, {n_lab} label files") print("classes:", [l for l in (root / "data.yaml").read_text().splitlines() if ": " in l and l.startswith(" ")]) first = sorted((root / "labels" / "train").glob("\*.txt"))[0] print(first.name, "begins:") print("\\n".join(first.read_text().splitlines()[:3]))

```
train: 720 images, 720 label files
val: 80 images, 80 label files
classes: ['  0: star', '  1: satellite']
000000_00.txt begins:
0 0.317449 0.826934 0.021151 0.021151
0 0.940648 0.994064 0.019531 0.011871
0 0.622531 0.310500 0.019531 0.019531
```

Each label line is `class cx cy w h`, as fractions of the tile size. Drawing one tile's labels back onto its image checks that they line up.

In \[8\]:

Copied!

```
from PIL import Image

name = "000005_02"
tile = Image.open(root / "images" / "train" / f"{name}.png")
tw, th = tile.size
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(tile, cmap="gray")
rows = (root / "labels" / "train" / f"{name}.txt").read_text().split("\n")
for row in filter(None, rows):
    c, cx, cy, bw, bh = row.split()
    cx, cy, bw, bh = float(cx) * tw, float(cy) * th, float(bw) * tw, float(bh) * th
    colour = "cyan" if c == "0" else "orange"
    ax.add_patch(Rectangle((cx - bw / 2 - 0.5, cy - bh / 2 - 0.5), bw, bh,
                           fill=False, color=colour, lw=0.8))
ax.set_title(f"Tile {name} with its YOLO labels")
ax.axis("off")
plt.show()
```

from PIL import Image name = "000005_02" tile = Image.open(root / "images" / "train" / f"{name}.png") tw, th = tile.size fig, ax = plt.subplots(figsize=(6, 6)) ax.imshow(tile, cmap="gray") rows = (root / "labels" / "train" / f"{name}.txt").read_text().split("\\n") for row in filter(None, rows): c, cx, cy, bw, bh = row.split() cx, cy, bw, bh = float(cx) * tw, float(cy) * th, float(bw) * tw, float(bh) * th colour = "cyan" if c == "0" else "orange" ax.add_patch(Rectangle((cx - bw / 2 - 0.5, cy - bh / 2 - 0.5), bw, bh, fill=False, color=colour, lw=0.8)) ax.set_title(f"Tile {name} with its YOLO labels") ax.axis("off") plt.show()

## What you built[¶](#what-you-built)

`first-set/` is a YOLO dataset of 800 tiles from 200 frames of a camera with hot pixels, radiation hits and pattern noise, with every star and streak above signal-to-noise 5 labelled. Ultralytics reads it as is:

```
pip install ultralytics
yolo detect train data=first-set/data.yaml model=yolo11n.pt imgsz=256
```

To go further:

- [Export a training set](https://shaneryall.github.io/SkyTwin/0.1/tutorials/export-training-data.md) covers COCO output, rotated boxes and loading frames straight into PyTorch.
- [Use and override an instrument profile](https://shaneryall.github.io/SkyTwin/0.1/tutorials/override-a-profile.md) renders through the profiles learned from real cameras.
- The [labels reference](https://shaneryall.github.io/SkyTwin/0.1/tutorials/labels.md) lists every option and file.
