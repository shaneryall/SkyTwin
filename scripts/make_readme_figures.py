r"""README figures: frames rendered by Skytwin, stretched as `skytwin-export` stretches them.

    pip install ./crates/skytwin-py pillow
    python scripts/make_readme_figures.py docs/images

Run from the repository root, so the scenes find `profiles/` and `examples/`. Every frame is
synthetic: random stars, explicit streaks and targets from the scenes below, rendered by the
`skytwin` package with fixed seeds, so the frames are the same on every machine. The PNG files
were made with numpy 2.5 and Pillow 13.0; another Pillow may draw the captions or encode the
files differently.

Each frame is stretched whole with `skytwin.labels.stretch`, the asinh stretch that
`skytwin-export` writes, and then cropped, with row 0 at the top as in the exported PNG files.
A crop's grey levels depend on its own frame's noise, so the backgrounds of different frames
cannot be compared by eye. Crops are shown at their own pixels or enlarged by a whole number, so
no pixel is dropped. Every box in `labels.png` comes from `skytwin.labels.boxes`, the function
the exporter calls; none is drawn by hand.

The script writes five PNG files, each at most 880 px wide:

- `profiles.png`: random stars through each shipped profile, a 164 px crop from the centre of
  each frame at the camera's own pixel scale.
- `clean-dirty.png`: one seed with and without a satellite streak, both stretched like the clean
  frame, and the absolute difference on a linear scale.
- `labels.png`: two 212 px tiles with the rotated boxes `--format yolo-obb` writes for them.
- `defects.png`: eight defects, each switched on alone in a plain camera, with settings strong
  enough to see.
- `glint.png`: the tumbling-glint example's target streak and its light curve from the truth.

It prints the numbers the README quotes: how many pixels differ between the clean and dirty
frames, how many pixels the streak's light reaches (where the noise-free frames differ), and how
many differing pixels lie outside those; and the glint's peak over the target's median signal.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import skytwin
from PIL import Image, ImageDraw, ImageFont
from skytwin import labels as lab

BG = (13, 17, 23)
EDGE = (48, 54, 61)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)
STAR = (86, 180, 233)
SAT = (230, 159, 0)
GAP = 8


def font(size):
    """Pillow's bundled font at `size` px (Pillow 10.1 or later)."""
    return ImageFont.load_default(size=size)


def render(scene, seed=1, expected=False):
    """Frame 0 of `scene` (a scene dict) with `seed`."""
    return skytwin.Simulator(scene, seed=seed).render(0, expected=expected)


def to_rgb(grey8):
    """An 8-bit greyscale array as an RGB Pillow image."""
    return Image.fromarray(grey8).convert("RGB")


def stretch_like(image, reference, soft=3.0, top=1000.0):
    """`image` stretched with the median and noise of `reference`, by the formula of
    `skytwin.labels.stretch`, so two frames share one grey scale."""
    ref = np.nan_to_num(reference.astype(np.float64))
    med = float(np.median(ref))
    sigma = 1.4826 * float(np.median(np.abs(ref - med))) or float(ref.std()) or 1.0
    lo = math.asinh(-2.0 / soft)
    span = math.asinh(top / soft) - lo
    z = np.clip((np.nan_to_num(image.astype(np.float64)) - med) / sigma, -2.0, top)
    return np.round((np.arcsinh(z / soft) - lo) / span * 255.0).astype(np.uint8)


def crop(a, x0, y0, size):
    """A `size` × `size` crop of array `a` with its top-left corner at column `x0`, row `y0`."""
    return a[y0 : y0 + size, x0 : x0 + size]


def enlarge(im, panel_px):
    """`im` enlarged to `panel_px` by a whole number, nearest-neighbour."""
    if panel_px % im.width or panel_px % im.height:
        raise ValueError(f"{im.size} does not divide {panel_px}")
    return im.resize((panel_px, panel_px), Image.NEAREST)


def sheet(panels, columns, panel_px, caption_px=30):
    """Panels laid out in `columns`, each `panel_px` square with a thin border and its caption
    (one or two lines) below. `panels` holds (Pillow image, caption) pairs."""
    rows = (len(panels) + columns - 1) // columns
    w = columns * panel_px + (columns + 1) * GAP
    h = rows * (panel_px + caption_px) + (rows + 1) * GAP
    out = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(out)
    for k, (im, text) in enumerate(panels):
        r, c = divmod(k, columns)
        x = GAP + c * (panel_px + GAP)
        y = GAP + r * (panel_px + caption_px + GAP)
        out.paste(enlarge(im, panel_px), (x, y))
        draw.rectangle((x - 1, y - 1, x + panel_px, y + panel_px), outline=EDGE)
        first, _, second = text.partition("\n")
        draw.text((x + panel_px / 2, y + panel_px + 7), first, fill=TEXT, font=font(15), anchor="ma")
        if second:
            draw.text((x + panel_px / 2, y + panel_px + 26), second, fill=MUTED, font=font(13), anchor="ma")
    return out


def stars(count, flux=(300.0, 1e6)):
    """A `random_stars` block of `count` stars between the two fluxes, electrons."""
    return {"count": count, "min_flux_e": flux[0], "max_flux_e": flux[1]}


def profiles():
    """Random stars through each shipped profile: a 164 px crop from the centre of each frame."""
    shown = [
        ("example-microsat", "example-microsat"),
        ("wise-w1", "WISE W1"),
        ("cassiope-fai-nir", "CASSIOPE FAI NIR"),
        ("ckoirama-cdk24", "Ckoirama CDK24"),
        ("astrianet-pl16803", "AstriaNet PL16803"),
    ]
    size = 164
    panels = []
    for name, caption in shown:
        cam = json.loads(Path(f"profiles/{name}.json").read_text())["camera"]
        w, h = cam["width"], cam["height"]
        scene = {
            "instrument": f"profiles/{name}.json",
            "scene": {"random_stars": stars(max(int(4000e-6 * w * h), 60))},
        }
        g = lab.stretch(render(scene, seed=2).image)
        scale = f"{cam['pixel_scale_arcsec']:.3g} arcsec/px"
        panels.append((to_rgb(crop(g, (w - size) // 2, (h - size) // 2, size)), f"{caption}\n{scale}"))
    return sheet(panels, 5, size, caption_px=46)


def clean_dirty():
    """One seed with and without a streak, and their absolute difference.

    The example profile is used without its frame-transfer smear, so the streak's light stays
    near the streak. Returns the figure, the number of pixels that differ in the whole frame, the
    number the streak's light reaches, and the number of differing pixels outside those.
    """
    base = {
        "instrument": "profiles/example-microsat.json",
        "overrides": {"smear": None},
        "scene": {"random_stars": stars(300, (300.0, 500000.0))},
    }
    streak = {"start": [30.0, 230.0], "end": [250.0, 70.0], "mag": 8.0}
    dirty = {**base, "scene": {**base["scene"], "streaks": [streak]}}
    c, d = render(base, seed=3, expected=True), render(dirty, seed=3, expected=True)
    diff = np.abs(d.image.astype(np.float64) - c.image.astype(np.float64))
    reached = d.expected != c.expected
    counts = (int(np.count_nonzero(diff)), int(np.count_nonzero(reached)), int(np.count_nonzero(diff[~reached])))
    size = 280
    scale = max(float(np.percentile(diff[diff > 0], 99)), 1e-9)
    linear = (np.clip(diff / scale, 0, 1) * 255).astype(np.uint8)
    panels = [
        (to_rgb(crop(stretch_like(c.image, c.image), 0, 0, size)), "clean"),
        (to_rgb(crop(stretch_like(d.image, c.image), 0, 0, size)), "dirty: same seed, plus a streak"),
        (to_rgb(crop(linear, 0, 0, size)), "|dirty minus clean|"),
    ]
    return sheet(panels, 3, size), counts


def tile_with_boxes(scene, seed, tile, scale=2):
    """One tile of `scene`, enlarged `scale` times, with the rotated boxes the exporter writes."""
    f = render(scene, seed=seed)
    h, w = f.image.shape
    x0, y0, tw, th = tile
    bxs = lab.boxes(lab.objects(f.truth), tile, (w, h))
    im = enlarge(to_rgb(lab.stretch(f.image)).crop((x0, y0, x0 + tw, y0 + th)), tw * scale)
    draw = ImageDraw.Draw(im)
    for b in bxs:
        pts = [(min(x * scale, tw * scale - 1), min(y * scale, th * scale - 1)) for x, y in b.corners]
        draw.line([*pts, pts[0]], fill=STAR if b.obj.kind == "star" else SAT, width=2)
    return im, bxs


def _count(bxs, kind):
    """How many of the boxes `bxs` hold an object of `kind`."""
    return sum(b.obj.kind == kind for b in bxs)


def _example(name):
    """The scene `examples/<name>.json` as a dict."""
    return json.loads(Path(f"examples/{name}.json").read_text())


def labels():
    """Two 212 px tiles with their boxes: point stars and a streak; trailed stars and a point."""
    a, ba = tile_with_boxes(
        {
            "instrument": "profiles/example-microsat.json",
            "scene": {
                "random_stars": stars(700, (300.0, 300000.0)),
                "streaks": [{"start": [470.0, 600.0], "end": [760.0, 440.0], "mag": 9.5}],
            },
        },
        5,
        (500, 400, 212, 212),
    )
    trailed = _example("track-rate")
    trailed["scene"]["random_stars"]["count"] = 1500
    trailed["scene"]["targets"][0]["brightness"]["lambertian_sphere"]["diameter_m"] = 20.0
    b, bb = tile_with_boxes(trailed, 1, (406, 406, 212, 212))
    panels = [
        (a, f"point stars ({_count(ba, 'star')}) and a streak"),
        (b, f"trailed stars ({_count(bb, 'star')}) and a tracked target"),
    ]
    return sheet(panels, 2, 424)


def defect(block, scene_stars=None, camera=None):
    """A plain camera with one profile `block` added, as a scene dict."""
    prof = {"name": "plain", "camera": {"exposure_s": 5.0, "read_noise_e": 6.0, **(camera or {})}, **block}
    scene = {"random_stars": stars(150, (500.0, 200000.0))}
    if scene_stars:
        scene["stars"] = scene_stars
    return {"instrument": prof, "scene": scene}


def defects():
    """Eight defects, each alone: a 208 px crop, or a 104 px crop enlarged twice."""
    bright = [{"x": 100.0, "y": 100.0, "flux_e": 3e6}]
    radiation = {"radiation": {"rate_per_cm2_s": 40.0, "depth_um": 15.0, "e_per_um": 300.0}}
    hot = {"hot_pixels": {"density_per_mpix": 4000, "e_per_s_at_ref": {"log_uniform": [5, 500]}}}
    cti = {"cti": {"serial": {"loss_per_transfer": 4e-4, "trail_px": 4.0}, "amp_right": True}}
    pattern = {"noise": {"patterns": [{"period_px": 7.0, "angle_deg": 35.0, "amplitude_adu": 12.0}]}}
    dust = {"flat": {"dust": {"density_per_mpix": 200, "radius_px": 22, "depth": 0.035, "hole_fraction": 0.35}}}
    bleed = {"saturation": {"bleed": True, "serial_overflow": {"fraction": 0.3, "length_px": 25.0}}}
    smear = {"smear": {"above_s_per_row": 2e-4, "below_s_per_row": 5e-5}}
    jumps = {"pointing": {"jump_rate_per_s": 0.4, "jump_arcsec": 15.0}}
    cases = [
        ("radiation hits", defect(radiation), 0, 0, 208),
        ("hot pixels", defect(hot), 0, 0, 104),
        ("charge-transfer trail", defect(cti, bright), 48, 48, 104),
        ("pattern noise", defect(pattern), 0, 0, 208),
        ("dust shadows", defect(dust, camera={"sky": {"e_per_s": 400.0}}), 0, 0, 208),
        ("bleeding and overflow", defect(bleed, [{"x": 100.0, "y": 100.0, "flux_e": 1.2e7}]), 0, 0, 208),
        ("frame-transfer smear", defect(smear, bright), 0, 0, 208),
        ("pointing jumps", defect(jumps), 0, 0, 104),
    ]
    panels = []
    for caption, scene, x0, y0, size in cases:
        g = lab.stretch(render(scene, seed=4).image)
        panels.append((to_rgb(crop(g, x0, y0, size)), caption))
    return sheet(panels, 4, 208)


def glint():
    """The tumbling-glint example: its target's streak, and the signal along it from the truth.

    Returns the figure and the glint's peak over the median of the light curve.
    """
    f = render(_example("tumbling-glint"), seed=31)
    tgt = f.truth["targets"][0]
    (sx, sy), (ex, ey) = tgt["start"]["xy"], tgt["end"]["xy"]
    size = 212
    x0, y0 = int((sx + ex) / 2) - size // 2, int((sy + ey) / 2) - size // 2
    image = to_rgb(crop(lab.stretch(f.image), x0, y0, size))
    t, rate = np.array(tgt["light_curve"]).T
    peak = float(rate.max() / np.median(rate))
    plot = Image.new("RGB", (size * 2, size * 2), BG)
    draw = ImageDraw.Draw(plot)
    left, top, right, bottom = 62, 16, size * 2 - 14, size * 2 - 46
    draw.rectangle((left, top, right, bottom), outline=EDGE)
    lo, hi = math.log10(rate.min()) - 0.2, math.log10(rate.max()) + 0.2
    for k in range(math.ceil(lo), math.floor(hi) + 1):
        y = bottom - (k - lo) / (hi - lo) * (bottom - top)
        draw.line((left, y, right, y), fill=(30, 36, 44))
        draw.text((left - 6, y), f"1e{k}", fill=MUTED, font=font(12), anchor="rm")
    for s in range(0, int(t.max()) + 1, 2):
        x = left + s / t.max() * (right - left)
        draw.text((x, bottom + 6), str(s), fill=MUTED, font=font(12), anchor="ma")
    pts = [
        (left + ti / t.max() * (right - left), bottom - (math.log10(ri) - lo) / (hi - lo) * (bottom - top))
        for ti, ri in zip(t, rate, strict=True)
    ]
    draw.line(pts, fill=SAT, width=2)
    draw.text(((left + right) / 2, bottom + 24), "seconds after shutter open", fill=MUTED, font=font(13), anchor="ma")
    draw.text((left + 8, top + 6), "signal, electrons per second", fill=MUTED, font=font(13), anchor="la")
    panels = [(image, "a tumbling target's streak"), (plot, "its light curve, from the truth file")]
    return sheet(panels, 2, size * 2), peak


def main(out_dir):
    """Write the five figures to `out_dir` and print the numbers the README quotes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profiles().save(out / "profiles.png", optimize=True)
    fig, (changed, reached, outside) = clean_dirty()
    fig.save(out / "clean-dirty.png", optimize=True)
    labels().save(out / "labels.png", optimize=True)
    defects().save(out / "defects.png", optimize=True)
    fig, peak = glint()
    fig.save(out / "glint.png", optimize=True)
    print(f"pixels that differ: {changed}; reached by the streak's light: {reached}; differing outside: {outside}")
    print(f"glint peak over median signal: {peak:.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/images")
