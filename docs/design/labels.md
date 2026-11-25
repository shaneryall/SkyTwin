# How training labels are computed

The [labels reference](../labels.md) lists the fields and rules. This page explains how the
signal-to-noise ratio is computed and why the classes, tiles and image stretch work as they do.

## Signal-to-noise ratio

While the renderer draws a source, it keeps the electrons the source adds
near its path. The reach covers every PSF component that carries at least
5% of the light, out to 5σ + 2 px, grown by the field PSF at the farthest
corner of the image. After the sky, the other sources, the flat field,
smear, dark current and hot pixels are in place, the renderer weighs each
source against the variance of the pixels under it. That variance is the
expected signal plus read noise and extra white noise, the same as for the
Cramér–Rao bounds. Radiation hits, pattern noise and read-out faults are
not part of it.

The signal-to-noise ratio is the expected value for the frame, computed
from the noise-free image. Through the example profile it agrees to 0.01%
with the value computed from the difference of two noise-free frames
rendered with and without the source.

## Point or streak

A source is a streak when the farthest two points of its path are more than twice the camera
FWHM apart. The span, rather than the path length, keeps pointing jitter and small jumps from
turning a point into a streak.

In a frame that tracks a target at its rate, every star trails into a
`star_streak` and the target stays a `satellite_point`. In a frame that
stares at the stars, satellites that cross the field are
`satellite_streak`. [`examples/track-rate.json`](../../examples/track-rate.json)
tracks a GEO satellite from a 790 km orbit through the example profile.
In 3 s its stars trail about 45 px.

## Signal-to-noise ratio in a tile

A streak cut by a tile edge keeps its full signal-to-noise ratio times the square root of the
share of its on-image length inside the tile, since the squared matched-filter ratio adds over
pixels.

## Image stretch

The `asinh` stretch measures each pixel in units of the frame's own noise before mapping it to
grey levels. The grey level of a pixel then depends on its signal-to-noise ratio alone, in every
frame and through every profile.

## Checked with

Ultralytics 8.4 reads both YOLO layouts with no corrupt labels and trains
on them, and pycocotools reads the COCO files.
