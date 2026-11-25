# Versioning

Skytwin follows semantic versioning. While the version is 0.x, a breaking change increases the
minor number, and anything else increases the patch number.

## Breaking changes

The following changes are breaking:

- Removing a command-line option or a Python argument, or changing its meaning
- Removing or renaming a key of a scene file or an instrument profile, or changing what an
  existing key means
- Removing or renaming a field of the truth file, the provenance record or an exported label set
- Changing the pixels a scene and seed give, outside a documented fix
- Changing a published package's public API so that existing code no longer works

Adding a truth field or an optional scene or profile key whose default keeps old files rendering
the same isn't a breaking change. A fix that changes pixels is listed in the changelog with the
scenes it affects.

A fix that leaves outputs and file formats unchanged is a patch release. So is a new command,
option or truth field that changes nothing that already exists.

## Deprecations

Before Skytwin removes or renames a command-line option, a Python argument, or a scene, profile
or truth key, one release deprecates it:

1. The release that deprecates it still accepts the old name and behaves as before. Skytwin
   prints a warning that names the replacement, and the release notes list the deprecation.
2. A later breaking release removes it, no earlier than the next minor version.

A change that can't keep the old behavior can't be deprecated first. Its release notes say what
to do, for example to update your scene files.

## Releases

Each release has a section in the [changelog](https://github.com/shaneryall/SkyTwin/blob/main/CHANGELOG.md)
that lists its changes and marks the breaking ones. Each release publishes the same version on
GitHub, on PyPI as the `skytwin` package, and on crates.io as the `skytwin-core` and
`skytwin-cli` crates. You install `skytwin-calibrate` from the repository; it carries the same
version.
