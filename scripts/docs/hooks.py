"""MkDocs hooks for the Skytwin documentation site.

- The home page is the repository README and the changelog page is CHANGELOG.md, both added at
  build time so neither is written twice.
- Links from a page to a file outside `docs/` (a profile, an example scene, a source file) point
  to that file on GitHub, at the ref in the `DOCS_REF` environment variable (default `main`).
- Links to a page of the published site's latest version (`SITE` + `latest/`), which work on
  GitHub, become links within the version being built.
- The command-line reference is each command's `--help` output. The commands must be on the
  path: `skytwin` (or the binary named by `SKYTWIN_BIN`), `skytwin-export` and
  `skytwin-calibrate`.
"""

from __future__ import annotations

import os
import posixpath
import re
import subprocess

from mkdocs.structure.files import File

REPO = "https://github.com/shaneryall/SkyTwin"
SITE = "https://shaneryall.github.io/SkyTwin/"
LINK = re.compile(r"(\]\()([^)\s]+)(\))")
FROM_ROOT = {"index.md": "README.md", "reference/changelog.md": "CHANGELOG.md"}
"""Pages generated from a file at the repository root, whose links are relative to the root."""
CALIBRATE = [
    "learn",
    "check",
    "describe",
    "measure",
    "psf",
    "profile",
    "refine",
    "twins",
    "report",
    "score",
]


def _ref() -> str:
    return os.environ.get("DOCS_REF", "main")


def _outside(page_src: str, target: str, root_relative: bool) -> str | None:
    """The repository path of `target` when the site does not serve it, else None.

    Pages and images under docs/ stay on the site. A page generated from a root file (the README)
    links any other file under docs/ to the repository.
    """
    if re.match(r"^[a-z]+:|^#|^/", target):
        return None
    path, _, frag = target.partition("#")
    base = "" if root_relative else posixpath.join("docs", posixpath.dirname(page_src))
    repo_path = posixpath.normpath(posixpath.join(base, path))
    served = repo_path.endswith((".md", ".ipynb", ".png", ".gif", ".svg", ".jpg"))
    if repo_path.startswith("docs/") and (served or not root_relative):
        return None
    return repo_path + (f"#{frag}" if frag else "")


def on_page_markdown(markdown: str, page, config, files) -> str:
    src = page.file.src_uri
    root_relative = src in FROM_ROOT
    depth = src.count("/")

    def fix(m: re.Match) -> str:
        target = m.group(2)
        if target.startswith(SITE + "latest/"):
            path, _, frag = target[len(SITE + "latest/") :].partition("#")
            page_md = path.rstrip("/") + ".md" if path else "index.md"
            rel = posixpath.relpath(page_md, posixpath.dirname(src) or ".")
            return f"{m.group(1)}{rel}{'#' + frag if frag else ''}{m.group(3)}"
        repo_path = _outside(src, target, root_relative)
        if repo_path is None:
            if root_relative and target.startswith("docs/"):
                return (
                    f"{m.group(1)}{'../' * depth}{target[len('docs/') :]}{m.group(3)}"
                )
            return m.group(0)
        kind = "tree" if repo_path.endswith("/") else "blob"
        return f"{m.group(1)}{REPO}/{kind}/{_ref()}/{repo_path}{m.group(3)}"

    return LINK.sub(fix, markdown)


def _help(args: list[str]) -> str:
    run = subprocess.run(
        [*args, "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "100"},
    )
    return (run.stdout or run.stderr).rstrip()


def _cli_reference() -> str:
    skytwin = os.environ.get("SKYTWIN_BIN", "skytwin")
    version = subprocess.run(
        [skytwin, "--version"], capture_output=True, text=True, check=True
    )
    commands = [
        ("skytwin", [skytwin]),
        ("skytwin-export", ["skytwin-export"]),
        ("skytwin-calibrate", ["skytwin-calibrate"]),
        *((f"skytwin-calibrate {c}", ["skytwin-calibrate", c]) for c in CALIBRATE),
    ]
    out = [
        "# Command-line reference",
        "",
        (
            "Each command's options and defaults, as printed by its `--help` option in "
            f"`{version.stdout.strip()}`. For tasks and examples, see the guides."
        ),
        "",
    ]
    for title, args in commands:
        out += [f"## {title}", "", "```text", _help(args), "```", ""]
    return "\n".join(out)


def on_post_page(output: str, page, config) -> str:
    """Point a notebook's links to other pages at those pages' URLs.

    MkDocs rewrites `.md` links in Markdown pages only; a notebook is rendered by mkdocs-jupyter.
    """
    src = page.file.src_uri
    if not src.endswith(".ipynb"):
        return output

    def fix(m: re.Match) -> str:
        path = posixpath.normpath(posixpath.join(posixpath.dirname(src), m.group(2)))
        target = _FILES.get_file_from_path(path) if _FILES else None
        if target is None:
            return m.group(0)
        rel = posixpath.relpath("/" + target.url, "/" + page.url.rstrip("/"))
        slash = "/" if target.url.endswith("/") else ""
        return f"{m.group(1)}{rel.rstrip('.') or '.'}{slash}{m.group(3)}"

    return re.sub(r'(href=")([^":#]+\.md)((?:#[^"]*)?")', fix, output)


_FILES = None


def on_files(files, config):
    global _FILES
    root = os.path.dirname(config.config_file_path)
    for page, source in FROM_ROOT.items():
        with open(os.path.join(root, source), encoding="utf-8") as f:
            files.append(File.generated(config, page, content=f.read()))
    files.append(File.generated(config, "reference/cli.md", content=_cli_reference()))
    _FILES = files
    return files
