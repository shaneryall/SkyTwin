# skytwin-cli

Synthetic star-field images with satellite streaks, instrument defects and an exact answer key
for every frame.

The `skytwin-cli` crate provides the `skytwin` command for the
[Skytwin](https://github.com/shaneryall/SkyTwin) simulator. You provide a scene file that
describes the camera, the stars and the targets. The command writes a FITS image and a truth
file that lists every star, streak and target with its position, flux and signal-to-noise
ratio.

## Install the command

You need Rust 1.85 or later to build the command. In a terminal, build and install it with Cargo:

```bash
cargo install --locked skytwin-cli
```

## Use the command

To render a frame, pass the scene file and the output path to `skytwin`:

```bash
skytwin SCENE OUTPUT
```

Replace the following:

- `SCENE`: The path to the scene file, for example `two-streaks.json`
- `OUTPUT`: The path to the FITS image to write, for example `two-streaks.fits`

The command also writes the truth file, for example `two-streaks.truth.json`. To list the
options, such as `--seed` and `--frames`, run `skytwin --help`.

For more information, see the following pages:

- Example scenes:
  [`examples/`](https://github.com/shaneryall/SkyTwin/tree/v0.1.0/examples)
- Scene files and the truth file:
  [Scene format](https://github.com/shaneryall/SkyTwin/blob/v0.1.0/docs/scene-format.md)
- Instrument profiles:
  [Override a profile](https://github.com/shaneryall/SkyTwin/blob/v0.1.0/docs/override-a-profile.md)
- The Python package and training sets:
  [Skytwin README](https://github.com/shaneryall/SkyTwin#readme)
- Versioning:
  [Versioning](https://github.com/shaneryall/SkyTwin/blob/v0.1.0/docs/versioning.md)

All Skytwin crates share one release version. Each crate requires exactly that version of the
other Skytwin crates.

For more information about the models, see the
[references](https://github.com/shaneryall/SkyTwin/blob/v0.1.0/docs/references.md).

## Licence

Skytwin uses the GNU General Public License, version 3 or later. For the licence terms, see
[`LICENSE`](https://github.com/shaneryall/SkyTwin/blob/v0.1.0/LICENSE); for third-party data and
code notices, see
[`THIRD-PARTY-LICENSES.md`](https://github.com/shaneryall/SkyTwin/blob/v0.1.0/THIRD-PARTY-LICENSES.md).
If you use Skytwin in a publication, cite it as described in
[`CITATION.cff`](https://github.com/shaneryall/SkyTwin/blob/v0.1.0/CITATION.cff).
