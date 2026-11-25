# skytwin-core

Synthetic star-field images with satellite streaks, instrument defects and an exact answer key
for every frame.

The `skytwin-core` crate is the library behind the
[Skytwin](https://github.com/shaneryall/SkyTwin) simulator. It turns a scene (stars, streaks
and orbiting targets) and a camera or instrument profile into a frame: an image in ADU and a
truth record of exactly what was drawn, so detectors, trackers and plate solvers can be scored
against known answers. The same scene and seed give the same pixels on every machine.

For more information about Skytwin and how to install the `skytwin` command or the Python
package, see the [Skytwin README](https://github.com/shaneryall/SkyTwin#readme). For scene
files and the truth file, see
[Scene format](https://github.com/shaneryall/SkyTwin/blob/v0.1.0/docs/scene-format.md).

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
