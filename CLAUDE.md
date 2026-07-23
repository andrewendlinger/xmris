# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`xmris` is an xarray-based, purely functional toolbox for MRI/MRS ("xarray in, xarray out"). Users interact through the `.xmr` xarray accessor (e.g. `da.xmr.zero_fill().xmr.to_spectrum().xmr.autophase()`).

## Architecture — read before writing library code

@docs/contributing/contract.md is the Architecture Contract — the numbered Commandments every change under `src/xmris/` must obey, each with its enforcement and live exemplars. Follow it for any library code; where another page differs, the contract wins.

## Documentation style

These four rules govern everything under `docs/` — explanation articles, tutorials, diary entries, and edits to any of them. This section is their single source of truth; the `dev-diary` and `docs-page` skills route here for the rules and restate only the one exception that binds their own genre. *Exception: guides under `docs/contributing/` are exempt from the motivated-narrative rule — a numbered list of commands is the right shape for a setup page.*

- **Motivated narrative, never a FAQ.** One driving question the reader already has, with every decision arriving as the answer to a tension they just felt. A cold "Why X?" heading makes a sound decision read as an assertion to accept. Concise and conversational; deep or tangential rationale goes in a `:::{dropdown}`, off the main line of reasoning.
- **One home per concept.** Consolidate in whichever direction fits — "where does this belong?" beats "who had it first." Editing and thinning existing pages is expected work, not scope creep. Say what you moved in the PR body. *Exception: a `docs/diary/` entry is a decision record, not a concept home — it owns one decision and is rewritten in place as that decision evolves; two entries may touch one concept when their decisions differ.*
- **Every article stands alone.** Readers arrive from search and deep links, not by walking the TOC. Each page must read start to finish on its own, so cross-reference rather than depend silently, and keep the orienting recap when you thin a page. Declare a hard prerequisite in a `seealso` at the top.
- **The MyST palette carries the argument.** Mermaid, admonitions, dropdowns, tables, LaTeX, executable `code-cell`s — reach for the one that does real work (a decision tree drawn as a flowchart is checkable at a glance; the same tree in prose is not). Nothing decorative. Stay inside the palette the docs already use.

Reader-facing prose uses plain strings (`"time"`, `"frequency"`), per Commandment 4's library-internals-only rule. `ATTRS`/`DIMS`/`COORDS` appear only in passages explicitly addressed to contributors, or inside hidden test cells.

## Significant changes get a diary entry

If a change picks between ≥2 viable approaches, adds conceptual surface (a new rule, decorator, or namespace — not a vocabulary term that follows an existing pattern), or spans multiple PRs, invoke the `dev-diary` skill — at the **start** (a one-screen article written from the approved plan, as the branch's first commit) and again at the **end** (rewritten into the story of how it is now; a "what changed from the plan" note only where the divergence teaches). When an existing entry already tells the decision's story, propose updating that entry instead of adding a sibling.

Pass 1 is the change's **master overview** and its review gate: the plan file is right for executing and too heavy for approving, so the entry is what gets read on the rendered site (`uv run docs`) before work starts. It never restates the plan's steps. The skill always asks before writing anything — never decide that autonomously — and after committing the draft the turn **ends** so the user can review the page; implementation waits for their go-ahead.

The `Dev Diary` section opens with one evergreen intro (`docs/diary/about.md`, pinned first) that explains what the diary *is*; dated entries follow it chronologically and carry a muted `Last edited` line rather than a status banner.

## Environment & commands

Package manager is `uv` — never use pip. Add deps with `uv add <pkg>`; sync with `uv sync --all-extras --dev`.

- Tests: `uv run test` (regenerates notebook tests from MyST, then runs pytest). Regenerate notebooks only: `uv run test-gen`.
- Architecture tests only, fast iteration: `uv run pytest tests/test_core.py -n0 --no-cov` (pytest `addopts` otherwise forces `-n auto --nbmake --cov`). Single test: append `::TestClass::test_name`.
- Lint: `uv run ruff check .` (`--fix` to auto-fix). Format: `uv run ruff format .`.
- Type-check: `uv run mypy src/xmris` — run it and fix clear type errors before finishing. It is not in CI and not configured, so xarray typing can be noisy; fix real issues, don't chase false positives.
- Docs API stubs: `uv run docs-api`. Check a page renders: `myst build --html` from `docs/` — one-shot, ~10 s warm, exit 0 (add `--execute` to run notebooks too).
- `uv run docs` and `uv run docs-notebooks` **launch a blocking preview server** (`myst start --execute`) and never exit. They are for a human reading the site — never put them in a verification step. In general, check what a `uv run <name>` alias does in `src/xmris/_scripts.py`; the inline comments in `pyproject.toml` describe intent, not blocking behaviour.

Ruff: line length 100, NumPy docstring convention. Public functions need fully-typed signatures + NumPy-format docstrings (they feed the quartodoc API docs).

## Testing strategy

- Architecture/config invariants: standard pytest in `tests/test_core.py`.
- Math/science behavior: MyST `.md` notebooks **are** the tests. `uv run test` converts them to `.ipynb` under `tests/autogen_notebooks/` (gitignored) and runs them via nbmake. `test-gen` walks `docs/notebooks/` (tutorials) **and** `docs/explanation/` (explainers that carry a jupytext kernelspec, landing under `autogen_notebooks/explanation/`) — so a live claim in an explainer is executed and asserted like any tutorial cell.
  - Tag pure-assert cells `# %% tags=["remove-cell"]` so nbmake executes them but the rendered docs hide them.
  - Files prefixed `testonly_` are test-only (never rendered).

## Gotchas

- `pyamares` comes from a git fork (`[tool.uv.sources]`, branch `xmris-compatible`), not PyPI. For local pyAMARES dev, swap the `git` line for the commented `path` line. Publish builds use `uv build --no-sources`.
- Jupytext syncs `.md` ↔ `.ipynb`; edit either, but commit ONLY `.md` — `docs/**/*.ipynb` are gitignored. The frozen kernel label is `display_name: Python 3 (xmris)` (jupytext has no `--display-name` flag). To bulk-fix drifted kernels, register the venv kernelspec once with `uv run python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (xmris)"`, then `uv run jupytext --set-kernel python3 ./docs/notebooks/**/*.md`.
- Python is capped at ≤3.13 (pyamares constraint). The pins `xarray<2025.11.0`, `griffe<0.40.0`, `pytest-cov<7.0` are deliberate — don't unpin without a reason.

## Commits & releases

- Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `add:`). Work on `main`.
- Release (see `/release`): never bump version until CI is green. Releases go through a `release/vX.Y.Z` branch (full test matrix); a `vX.Y.Z` tag triggers the PyPI publish. Bump with `uv version --bump patch|minor`.
