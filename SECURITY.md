# Security policy

## Supported versions

Security fixes go into the latest release. While Skytwin is at 0.x, update to the latest version
to get a fix.

## Report a vulnerability

Report a vulnerability privately, not in a public issue:

1. On the repository's GitHub page, click the **Security** tab.
2. Click **Report a vulnerability**.
3. Describe the problem, the Skytwin version (`skytwin --version`, or `skytwin.__version__` in
   Python), and how to reproduce it. If a file triggers it, attach the file or say how to make
   one.

Only the maintainers see the report, and they aim to answer within seven days. When a fix is
released, the release notes credit you, unless you ask not to be named.

## Scope

Skytwin reads files that may come from other people: scene files and instrument profiles (JSON),
TLE files, ephemeris tables, star catalogues (a UCAC4 directory or a Gaia CSV extract), header
keyword maps, and, in `skytwin-calibrate`, FITS frames. Any of these that makes Skytwin crash,
hang, use unbounded memory or time, or write outside the output paths you gave it is a
vulnerability.

A wrong result for valid input is a bug, not a vulnerability. Report it in a public issue.
