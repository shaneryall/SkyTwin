# skytwin-calibrate

Build a Skytwin instrument profile from a real image archive, and report how
closely simulated frames match it.

## Install

Requires Python 3.10 or later and a built `skytwin` binary
(`cargo build --release` at the repository root).

```bash
pip install -e tools/calibrate
```

## Quick start

Point `learn` at sample frames (a directory, files, glob patterns or a
`.txt` list). Nothing about the instrument needs to be known in advance:

```bash
skytwin-calibrate learn /path/to/frames -o profile.json \
    --skytwin target/release/skytwin --ucac4 /path/to/ucac4
```

It writes `profile.json`, a report `profile.learn.md` of where each setting came from, and the
header keyword map `profile.keywords.json`. `skytwin-calibrate describe profile.json` prints
the profile's summary and a scene to start from.

## Documentation

| Document | Covers |
| --- | --- |
| [Learn a profile](../../docs/learn-a-profile.md) | Steps: learning, calibration frames, ground frames, spacecraft states, the separate commands, scoring a detector |
| [Calibrate reference](../../docs/reference/calibrate.md) | Every command, the files `learn` writes, and what it leaves out |
| [How a profile is learned](../../docs/design/calibration.md) | What each step measures and how |
| [Profile reference](../../docs/reference/profile.md) | Every key of an instrument profile |

## Test

```bash
pip install -e 'tools/calibrate[test]'
pytest tools/calibrate/tests
```

The round-trip test renders frames with known settings through Skytwin and
checks that `measure` and `profile` recover them.
