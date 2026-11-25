# Contributing to Skytwin

## Setup

You need the following tools:

- Rust 1.85 or later (`rustup`), with `rustfmt`, `clippy`, and `cargo-deny`
  (`cargo install cargo-deny`).
- Python 3.10 or later for `tools/calibrate` and `crates/skytwin-py`.

```bash
cargo build --release
python3 -m venv .venv
.venv/bin/pip install -e 'tools/calibrate[test]' ruff maturin pytest
```

## Checks

Every change passes what CI runs:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace
.venv/bin/ruff check tools/calibrate crates/skytwin-py
.venv/bin/ruff format --check tools/calibrate crates/skytwin-py
PATH=$PWD/target/release:$PATH .venv/bin/python -m pytest tools/calibrate/tests
cargo deny check
(cd crates/skytwin-py && VIRTUAL_ENV=$PWD/../../.venv ../../.venv/bin/maturin develop --release --locked)
PATH=$PWD/target/release:$PATH .venv/bin/python -m pytest crates/skytwin-py/tests
```

The lint policy lives in the root `Cargo.toml`: clippy pedantic, with
numeric casts, maths-style short names and exact float comparison allowed.
Fix other findings rather than allowing them; where an exception is
needed, use `#[expect(lint, reason = "...")]`.

### Documentation site

The site is built from `docs/` with MkDocs, and its command-line reference from each command's
`--help`, so the commands must be installed first. To build it with warnings as errors, from the
repository root:

```bash
.venv/bin/pip install --require-hashes -r scripts/docs/requirements.txt
.venv/bin/pip install ./crates/skytwin-py ./tools/calibrate
cargo build --release -p skytwin-cli
PATH=$PWD/.venv/bin:$PATH SKYTWIN_BIN=target/release/skytwin mkdocs build --strict
```

Use `mkdocs serve` in place of `mkdocs build --strict` to preview it at
`http://127.0.0.1:8000`.

## Conventions

- Reproducibility is a contract: the same scene and seed give the same
  pixels on every machine. Do not change a random stream, the order random
  numbers are drawn, or the order of floating-point operations without
  saying so in the commit; tests check frames bit for bit.
- Profile and scene keys are public API, a file format of their own.
  Add keys with defaults that keep old files rendering the same; unknown
  keys stay errors.
- Every public item has a doc comment: a one-line
  summary, then units, frames and conventions, then `# Errors` and
  `# Panics` where they apply, and the source (paper, standard) for any
  model. Explain *why* in doc comments; code does not carry narrating
  `//` comments.
- New physics comes with a test against an independent value
  (a reference implementation, a paper's worked example, or a closed form).
  New profile settings come with a round-trip test in `tools/calibrate` when
  `learn` measures them.

## Commits and branches

Use the same type word for branches, commit subjects and pull request
titles:

| Type | For |
| --- | --- |
| `feat` | a new feature or user-visible behaviour |
| `bug` | a bug fix |
| `docs` | documentation only |
| `refactor` | a code change with no behaviour change |
| `test` | tests only |
| `chore` | tooling, dependencies, configuration, cleanup |
| `ci` | CI pipeline changes |
| `perf` | performance |

Branches are `<type>/<short-kebab-topic>`; commit subjects are
`<type>: <imperative summary>`, with an optional scope such as
`feat(calibrate): ...`. Add a line to the `Unreleased` section of
[CHANGELOG.md](CHANGELOG.md) for anything a user would notice.

## Releasing

The maintainer's steps are in [RELEASING.md](RELEASING.md).
