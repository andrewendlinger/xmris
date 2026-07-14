# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`xmris` is an xarray-based, purely functional toolbox for MRI/MRS ("xarray in, xarray out"). Users interact through the `.xmr` xarray accessor (e.g. `da.xmr.zero_fill().xmr.to_spectrum().xmr.autophase()`).

## Architecture — read before writing library code

@docs/contributing/ai_context.md defines the strict architectural rules (the "8 Commandments") and code templates. Follow it for any code under `src/xmris/`. Key points:

- Use the config singletons `ATTRS`/`DIMS`/`COORDS`/`VARS` from `xmris.core.config` — never hardcode dim/attr strings inside the package. This applies to library internals only; user-facing examples may use plain strings like `"time"`.
- Do NOT use the legacy `xmris.config` `DEFAULTS` in new code — it is a deprecated shim (importing it emits a `DeprecationWarning`). `core.config` is the single source of truth.
- If a new function needs a dim/coord/attr not already in the vocabulary, add it to `config.py` and explicitly flag the new term to the user.

## Environment & commands

Package manager is `uv` — never use pip. Add deps with `uv add <pkg>`; sync with `uv sync --all-extras --dev`.

- Tests: `uv run test` (regenerates notebook tests from MyST, then runs pytest). Regenerate notebooks only: `uv run test-gen`.
- Architecture tests only, fast iteration: `uv run pytest tests/test_core.py -n0 --no-cov` (pytest `addopts` otherwise forces `-n auto --nbmake --cov`). Single test: append `::TestClass::test_name`.
- Lint: `uv run ruff check .` (`--fix` to auto-fix). Format: `uv run ruff format .`.
- Type-check: `uv run mypy src/xmris` — run it and fix clear type errors before finishing. It is not in CI and not configured, so xarray typing can be noisy; fix real issues, don't chase false positives.
- Docs API stubs: `uv run docs-api`.

Ruff: line length 100, NumPy docstring convention. Public functions need fully-typed signatures + NumPy-format docstrings (they feed the quartodoc API docs).

## Testing strategy

- Architecture/config invariants: standard pytest in `tests/test_core.py`.
- Math/science behavior: MyST `.md` notebooks in `docs/notebooks/` **are** the tests. `uv run test` converts them to `.ipynb` under `tests/autogen_notebooks/` (gitignored) and runs them via nbmake.
  - Tag pure-assert cells `# %% tags=["remove-cell"]` so nbmake executes them but the rendered docs hide them.
  - Files prefixed `testonly_` are test-only (never rendered).

## Gotchas

- `pyamares` comes from a git fork (`[tool.uv.sources]`, branch `xmris-compatible`), not PyPI. For local pyAMARES dev, swap the `git` line for the commented `path` line. Publish builds use `uv build --no-sources`.
- Jupytext syncs `.md` ↔ `.ipynb`; edit either, but commit ONLY `.md` — `docs/**/*.ipynb` are gitignored. The frozen kernel label is `display_name: Python 3 (xmris)` (jupytext has no `--display-name` flag). To bulk-fix drifted kernels, register the venv kernelspec once with `uv run python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (xmris)"`, then `uv run jupytext --set-kernel python3 ./docs/notebooks/**/*.md`.
- Python is capped at ≤3.13 (pyamares constraint). The pins `xarray<2025.11.0`, `griffe<0.40.0`, `pytest-cov<7.0` are deliberate — don't unpin without a reason.

## Commits & releases

- Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `add:`). Work on `main`.
- Release (see `/release`): never bump version until CI is green. Releases go through a `release/vX.Y.Z` branch (full test matrix); a `vX.Y.Z` tag triggers the PyPI publish. Bump with `uv version --bump patch|minor`.
