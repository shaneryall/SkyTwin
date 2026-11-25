# Releasing Skytwin

The maintainer's runbook for a release. A version on crates.io or PyPI is permanent, so the
registries come last, after the GitHub release has been checked.

## Once, before the first release

1. In the GitHub repository settings, create the environment `pypi` with yourself as a required
   reviewer and deployments limited to `v*` tags.
2. On PyPI, add a trusted publisher for project `skytwin`: owner `shaneryall`, repository
   `SkyTwin`, workflow `release.yml`, environment `pypi`. No API token is stored.
3. Link the repository to Zenodo, so each GitHub release gets a DOI.

## Each release

1. Set the version in `[workspace.package]` of the root `Cargo.toml`, in the `skytwin-core` pin
   under `[workspace.dependencies]`, in `tools/calibrate/pyproject.toml`, and in
   `tools/calibrate/skytwin_calibrate/__init__.py`. The Python package `skytwin` reads its
   version from the root `Cargo.toml`.
   In the package READMEs (`crates/*/README.md`), point the links at the new tag
   (`blob/vX.Y.Z`), so each registry page links to the files of its own version.
2. Set `version` and `date-released` in `CITATION.cff`, and rename `## [Unreleased]` in
   `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD` with the release date.
3. Run every check in [CONTRIBUTING.md](CONTRIBUTING.md#checks) at the release commit.

4. Push the release commit and its tag `vX.Y.Z` to GitHub from a clean clone of the public
   repository, never with `--tags` or `--all`.
5. The `release` workflow builds the wheels for Linux, macOS and Windows and the source
   distribution, and tests the Linux wheel. Check the run, then approve the `pypi` deployment to
   upload them.
6. Publish the crates, core first:

   ```bash
   cargo publish -p skytwin-core
   ```

   ```bash
   cargo publish -p skytwin-cli
   ```

   `skytwin-py` isn't published to crates.io (`publish = false`).
7. Install from each channel and render the quick-start scene: `pip install skytwin`, `cargo
   install skytwin-cli`.
8. Publish the GitHub release with the changelog section as its notes.
