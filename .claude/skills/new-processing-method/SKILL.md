---
name: new-processing-method
description: Add a new xmris processing/accessor function end-to-end following the project's architectural rules. Use when adding a new `.xmr` method or transform (e.g. a new apodization, phasing, Fourier, baseline, or fitting operation), or when asked to "add a processing function/method" to xmris.
---

# Add a new xmris processing method

Scaffold a new library function and wire it into the `.xmr` accessor following the project's strict architecture, then prove it with a notebook test.

## 0. Read the rules first

Read `docs/contributing/ai_context.md` (the "8 Commandments" + code templates) before writing anything. All of it applies. The steps below are the workflow; that file is the source of truth for *how* each piece is written.

## 1. Place the transform

Add the pure math to the right domain module under `src/xmris/` — `processing/` (`fid.py`, `fourier.py`, `phasing.py`, `baseline.py`), `vendor/` (hardware sanitization), or `fitting/`. Match the surrounding file's style.

Rules that are easy to get wrong here:
- **Functional purity** — never mutate in place; always return a new `xr.DataArray`/`xr.Dataset`.
- **No magic strings** — import `ATTRS`/`DIMS`/`COORDS`/`VARS` from `xmris.core.config`; never hardcode `"time"`, `"reference_frequency"`, etc. inside the package.
- **Config-constant defaults** — a `dim` argument defaults to the config constant (`dim: str = DIMS.time`), never `None`.
- **Validation** — guard required attrs with `@requires_attrs(...)`; validate dims with `_check_dims(self._obj, dim, "func_name")`.
- **Coordinates** — build new coords with `as_variable(TERM, dim, data)` + `.assign_coords(...)`, don't hand-mutate `.attrs`.
- **Lineage** — append only quantitative parameters to `.attrs` (e.g. `phase_p0=15.0`); no boolean/string flags.
- **Docstring** — NumPy convention, fully-typed signature (feeds the quartodoc API docs).

## 2. Extend the vocabulary if needed

If the function needs a dim/coord/attr not already in `src/xmris/core/config.py`, add the `XmrisTerm` there (with `unit`/`long_name`) and **explicitly tell the user** every new `ATTRS`/`DIMS`/`COORDS`/`VARS` term you introduced so it can be tracked.

## 3. Register on the accessor

Expose the function via the `.xmr` namespace on `XmrisAccessor` (or `XmrisDatasetAccessor`) in `src/xmris/core/accessor.py`, and add it to the public API in `src/xmris/__init__.py` (`__all__`) if it's user-facing.

## 4. Write the notebook test (this is how math is tested here)

Math/science is tested via MyST `.md` notebooks in `docs/notebooks/<area>/` (basics/pipeline/fitting/vendor/visualization), not `test_*.py`. Create or extend the relevant notebook with:
1. Markdown explaining the math/physics (use rich MyST: LaTeX, mermaid, dropdowns where useful).
2. A cell generating synthetic, noisy `xarray` data.
3. A cell applying the new function and plotting the result.
4. **Assertion cells** (`assert` / `np.testing.assert_allclose`) proving both the numeric result **and** that dims/coords/attrs were preserved — each tagged `# %% tags=["remove-cell"]` so nbmake runs them but the rendered docs hide them.

Use a `testonly_`-prefixed file for pure test notebooks that shouldn't render.

## 5. Verify

Run the affected notebook and the architecture tests:

```
uv run test-gen                                  # regenerate .ipynb from the .md notebooks
uv run pytest tests/test_core.py -n0 --no-cov    # architecture invariants
uv run ruff format . && uv run ruff check . --fix
uv run mypy src/xmris                            # fix clear type errors
```

Then `uv run test` for the full pipeline (regenerates notebooks + runs everything) before wrapping up.

## 6. Report

Summarize the new function, any new `config.py` terms (call these out explicitly), and where its notebook test lives.
