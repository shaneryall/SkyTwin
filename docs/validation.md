# Validation and reproducibility

## Checks against independent tools

- A frame rendered from UCAC4 is plate-solved by Fayde to 0.01″ of the true centre; one from a
  790 km orbit to 0.044″ of the header WCS.
- A GEO satellite seen from LEO matches an independent Python calculation to 0.024″, 0.0 m in
  range and 0.0001 mag.
- Time scales and orbits are checked against astropy and the Python `sgp4` package.
- For each profile learned from a real archive,
  [How close they are](../profiles/README.md#how-close-they-are) in `profiles/README.md` gives
  the Kolmogorov–Smirnov distance between the statistics of its real frames and those of their
  simulated twins.

## Reproducibility

The same scene file and seed give the same pixels on every machine; continuous integration
renders reference scenes on Linux, macOS and Windows and compares the pixel hashes. Frames of a
batch run on all cores, and the output doesn't depend on the thread count.

Each pixel's noise comes from its own random stream, so adding or moving a source changes no
pixel outside its footprint: the same seed with and without a satellite gives an exact
clean/dirty pair. Every run writes `name.provenance.json` with the Skytwin version, command,
seed, status, and SHA-256 hashes of the scene, profile, catalogue and every output.

## Speed

On a 16-core Apple M4 Max, a 1024 × 1024 frame through the example profile renders at about 43
frames a second.
