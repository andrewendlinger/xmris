#!/usr/bin/env python3
"""Check hand-authored xmris docs against the house rules.

Run from the repo root::

    uv run python .claude/skills/docs-page/check_docs.py [PATH ...]

With no PATH it checks every hand-authored page: ``docs/**/*.md`` minus the
generated (``api_reference/``) and built (``_build/``) trees.

Errors exit 1 -- each one renders wrong or produces a dead link, and
``myst build`` is silent about all of them. Warnings exit 0: real drift, but
too judgment-dependent to gate on.

Stdlib only, and deliberately fence-aware: a bare ``#`` comment at column 0
inside a ``code-cell`` looks exactly like a Markdown header to a naive grep,
which is how the first pass of this audit overcounted headers three-fold.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# --- genre ------------------------------------------------------------------
# Location is genre. Diary entries belong to the `dev-diary` skill and
# api_reference/ is generated + gitignored, so neither is checked here.
GENRES = {
    "docs/notebooks": "tutorial",
    "docs/explanation": "explainer",
    "docs/contributing": "guide",
}
SKIP_DIRS = ("_build", "api_reference", "diary")

KERNEL_DISPLAY_NAME = "Python 3 (xmris)"

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADER_RE = re.compile(r"^(#{1,6})\s+\S")
TARGET_RE = re.compile(r"^\(.+\)=$")
IPYNB_LINK_RE = re.compile(r"\]\([^)]*\.ipynb[^)]*\)")
CONFIG_SINGLETON_RE = re.compile(r"\b(ATTRS|DIMS|COORDS|VARS)\.")


class Page:
    """A doc page split into fenced and unfenced regions, once."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines = path.read_text().split("\n")
        self.genre = next(
            (g for prefix, g in GENRES.items() if str(path).startswith(prefix)), "other"
        )
        self.frontmatter = self._frontmatter()
        self.headers: list[tuple[int, int, str]] = []  # (lineno, depth, text)
        self.cells: list[tuple[int, str, str]] = []  # (lineno, info, body)
        self._scan()

    def _frontmatter(self) -> str:
        if not self.lines or self.lines[0].strip() != "---":
            return ""
        for i, line in enumerate(self.lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(self.lines[1:i])
        return ""

    def _scan(self) -> None:
        start = 0
        if self.frontmatter:
            start = self.frontmatter.count("\n") + 3  # past the closing ---
        fence: str | None = None
        info = ""
        body: list[str] = []
        cell_start = 0
        for i in range(start, len(self.lines)):
            line = self.lines[i]
            m = FENCE_RE.match(line)
            if m:
                tok = m.group(1)[0] * 3
                if fence is None:
                    fence, info, body, cell_start = tok, line.strip().lstrip("`~"), [], i
                elif line.strip().startswith(fence):
                    self.cells.append((cell_start, info, "\n".join(body)))
                    fence = None
                continue
            if fence is not None:
                body.append(line)
                continue
            hm = HEADER_RE.match(line)
            if hm:
                self.headers.append((i, len(hm.group(1)), line.lstrip("#").strip()))

    def first_content_line(self) -> int | None:
        """Index of the first non-blank, non-frontmatter line."""
        start = self.frontmatter.count("\n") + 3 if self.frontmatter else 0
        for i in range(start, len(self.lines)):
            if self.lines[i].strip():
                return i
        return None

    def has_target_above(self, lineno: int) -> bool:
        """Report whether an explicit ``(target)=`` sits on the preceding line."""
        return lineno > 0 and bool(TARGET_RE.match(self.lines[lineno - 1].strip()))


def toc_files(root: Path) -> set[str]:
    """Collect every ``file:`` path listed in the hand-maintained myst.yml TOC."""
    myst = root / "docs" / "myst.yml"
    if not myst.exists():
        return set()
    entries = re.finditer(r"^\s*-?\s*file:\s*(\S+)", myst.read_text(), re.M)
    return {m.group(1).strip() for m in entries}


def check(page: Page, toc: set[str]) -> tuple[list[str], list[str]]:
    """Check one page, returning its (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    rel = str(page.path)
    name = page.path.name
    testonly = name.startswith("testonly_")

    def err(line: int, msg: str) -> None:
        errors.append(f"{rel}:{line + 1}: {msg}")

    def warn(line: int, msg: str) -> None:
        warnings.append(f"{rel}:{line + 1}: {msg}")

    # --- H1: exactly one, and it must be the first content node -------------
    # mystmd lifts the first heading into frontmatter.title, but only *removes*
    # it from the body when it leads the page. Anything before it (even a
    # hidden remove-cell) and the title renders twice.
    h1s = [h for h in page.headers if h[1] == 1]
    first = page.first_content_line()
    if not h1s:
        errors.append(f"{rel}:1: no H1 -- the page title is lifted from an H2 and rendered twice")
    else:
        for lineno, _, text in h1s[1:]:
            err(lineno, f"second H1 {text!r} -- exactly one per page")
        lineno = h1s[0][0]
        expected = lineno - 1 if page.has_target_above(lineno) else lineno
        if first is not None and expected != first:
            err(lineno, "content precedes the H1 -- the title will render twice")

    # --- explicit MyST targets (Commandment 8) ------------------------------
    # Auto-generated slugs are numbered by document position (id-1-, id-2-),
    # so inserting one section silently renumbers every anchor below it.
    for lineno, _, text in page.headers:
        if not page.has_target_above(lineno):
            err(lineno, f"header {text!r} has no explicit (target)= above it")

    # --- dead .ipynb links --------------------------------------------------
    # myst.yml excludes notebooks/**/*.ipynb, so these resolve to null -- and
    # the build emits no warning.
    # Commented-out links never render, so they are dead weight rather than a
    # broken page -- and HTML comments span lines, so track the block.
    infence = False
    fence = ""
    incomment = False
    for i, line in enumerate(page.lines):
        m = FENCE_RE.match(line)
        if m and not incomment:
            tok = m.group(1)[0] * 3
            if not infence:
                infence, fence = True, tok
            elif line.strip().startswith(fence):
                infence = False
            continue
        if infence:
            continue
        if incomment:
            if "-->" in line:
                incomment = False
            continue
        if "<!--" in line and "-->" not in line:
            incomment = True
            continue
        if IPYNB_LINK_RE.search(line):
            err(i, "links a .ipynb -- excluded from the build, so the link is dead")

    # --- frontmatter / kernel ----------------------------------------------
    if page.frontmatter:
        m = re.search(r"display_name:\s*(.*)", page.frontmatter)
        got = m.group(1).strip() if m else ""
        if got != KERNEL_DISPLAY_NAME:
            errors.append(
                f"{rel}:1: kernel display_name is {got or '(missing)'!r}, "
                f"expected {KERNEL_DISPLAY_NAME!r}"
            )
    elif page.genre in ("tutorial", "explainer"):
        warnings.append(f"{rel}:1: no jupytext frontmatter -- this page cannot use code-cells")

    # --- TOC membership -----------------------------------------------------
    if not testonly and page.genre != "other":
        rel_to_docs = str(page.path).removeprefix("docs/")
        if rel_to_docs not in toc:
            errors.append(f"{rel}:1: not in docs/myst.yml -- the page never renders")

    # --- warnings -----------------------------------------------------------
    for lineno, info, body in page.cells:
        if info.strip() == "mermaid":
            warn(lineno, "bare ```mermaid fence -- prefer the ```{mermaid} directive")
        if not info.startswith("{code-cell}"):
            continue
        # Config singletons are for library internals and hidden test cells.
        # Contributor-facing passages (architecture.md) legitimately show them,
        # so this can only ever be a warning.
        if "remove-cell" not in body and CONFIG_SINGLETON_RE.search(body):
            warn(lineno, "config singleton in a visible cell -- readers see plain strings")

    if page.genre == "tutorial" and not testonly:
        code = [b for _, i, b in page.cells if i.startswith("{code-cell}")]
        if code and not any(re.search(r"^\s*(assert |np\.testing\.assert)", b, re.M) for b in code):
            warnings.append(f"{rel}:1: tutorial runs code but asserts nothing -- a doc, not a test")

    return errors, warnings


def collect(args: list[str]) -> list[Path]:
    """Resolve CLI paths, defaulting to every hand-authored page under docs/."""
    if args:
        return [Path(a) for a in args]
    return sorted(
        p for p in Path("docs").rglob("*.md") if not any(part in SKIP_DIRS for part in p.parts)
    )


def main(argv: list[str]) -> int:
    """Check the requested pages and return the process exit code."""
    root = Path.cwd()
    if not (root / "docs").is_dir():
        print("error: run from the repo root (no ./docs found)", file=sys.stderr)
        return 2

    toc = toc_files(root)
    pages = collect(argv)
    all_errors: list[str] = []
    all_warnings: list[str] = []
    for path in pages:
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
        errors, warnings = check(Page(path), toc)
        all_errors += errors
        all_warnings += warnings

    for line in all_warnings:
        print(f"warning: {line}")
    if all_warnings and all_errors:
        print()
    for line in all_errors:
        print(f"error:   {line}")

    print(f"\n{len(pages)} page(s): {len(all_errors)} error(s), {len(all_warnings)} warning(s)")
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
