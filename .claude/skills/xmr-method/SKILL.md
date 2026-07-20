---
name: xmr-method
description: Add or modify a function in the xmris library and wire it into the `.xmr` accessor — transforms (processing/), vendor loaders (vendor/), or fitting (fitting/). Use when adding a new `.xmr` method, changing an existing one's signature, defaults, or domain contract, fixing a transform's math, or moving a function between modules.
---

# Add or change an xmris `.xmr` method

Most work here is **changing an existing function**, not adding one — signatures, defaults, domain
contracts, a fix to the math. Both paths use the same rules; jump to
[Editing an existing method](#editing) for the change map.

## 0. Read the rules first

Two documents own the rules, and this skill routes to them rather than restating:

- **`docs/contributing/ai_context.md`** — the 8 Commandments, `_check_dims`, `as_variable`, and the
  annotated `example_func` template. All of it applies.
- **`docs/explanation/domains.md`** — which decorator your function gets. It carries the full
  taxonomy and a mermaid decision tree (§ *For contributors: declaring a function's domain*). Read
  it before choosing; do not guess from the table below.

## 1. Route by kind

| | **Transform** | **Vendor loader** | **Fitting** |
|---|---|---|---|
| Lives in | `processing/` | `vendor/` | `fitting/` |
| Returns | `DataArray` | `DataArray` / `Dataset` | `Dataset` |
| Domain decorator | **yes** — funnel or preserving | no | no |
| `dim` default | config constant, or `None` iff multi-label | config constant | config constant |
| Accessor home | one of the **4 mixins** (§3) | direct on `XmrisAccessor` | direct on `XmrisAccessor` |
| Template | `templates/transform.md` | `templates/transform.md`, minus the decorator | ↑ |

**Converters and primitives are transforms that stay undecorated by design** — `to_spectrum`,
`to_fid`, the `fft` family, `to_ppm`/`to_hz`, `to_complex`. Their transforms are explicit on
purpose. `domains.md` explains why; never inline `fft`/`ifft` to fake a conversion, route through
the converters.

**Doesn't route here:**

- Widgets (`visualization/widget/`) → the **`xmr-widget`** skill.
- Plots (`visualization/plot/`) → **no skill yet**, overhaul pending. The `*Config`/`PlotParam`
  dataclass convention in `_base_config.py` is unlike anything below; don't apply this skill to it.
- Docs pages → **`docs-page`**. Diary entries → **`dev-diary`**.

## 2. Place the function

The full `processing/` module map — the two most often missed are at the bottom:

| Module | Holds |
|---|---|
| `processing/fid.py` | apodization, zero-fill, `to_spectrum`/`to_fid` |
| `processing/fourier.py` | the six FFT primitives |
| `processing/phasing.py` | `phase`, `autophase` |
| `processing/baseline.py` | baseline correction |
| `processing/referencing.py` | `to_ppm`/`to_hz` — **axis referencing lives here**, not in `fid.py` |
| `processing/utils.py` | complex-layout primitives (`to_complex`, `to_real_imag`) |

If the function needs a dim/coord/attr not in `src/xmris/core/config.py`, add the `XmrisTerm` there
(with `unit`/`long_name`) and **tell the user explicitly** — every new `ATTRS`/`DIMS`/`COORDS`/`VARS`
term must be tracked, not slipped in.

## 3. Wire it into the accessor

`XmrisAccessor` is **composed of four mixins** (`src/xmris/core/accessor.py:694`), so "add it to the
accessor" means picking one:

| Mixin | Line | Holds |
|---|---|---|
| `XmrisSpectrumCoordsMixin` | `:340` | `to_ppm`, `to_hz` |
| `XmrisFourierMixin` | `:352` | the six FFT primitives |
| `XmrisProcessingMixin` | `:432` | apodize, `to_spectrum`/`to_fid`, `zero_fill`, `baseline_als` |
| `XmrisPhasingMixin` | `:580` | `phase`, `autophase` |

Vendor and fitting methods go **directly on `XmrisAccessor`** (`fit_amares` `:733`,
`remove_digital_filter` `:816`, `estimate_group_delay` `:855`) — there is no mixin for them.

`XmrisDatasetAccessor` (`:678`) is **not** a home for transforms. It inherits from nothing and has
one `plot` property; putting a method there gives it no `.xmr` parity on DataArrays.

Then export it: import the function in `src/xmris/__init__.py` and add it to `__all__`, under the
matching labelled section. `templates/transform.md` has the delegator patterns.

(coverage)=
## 4. What the toolchain catches — and what it doesn't

**Read this before trusting a green test run.** `uv run pytest tests/test_core.py` passes 192 tests
whether or not your new function is covered by any of them.

| Rule | Enforced by |
|---|---|
| `dim=None` iff multi-label domain | `TestDomainDimRule` — **automatic** ✅ |
| Domain semantics (coercion, restore, attrs across transforms) | **automatic** ✅ — but only if you used the decorators |
| NumPy docstrings, typed signatures | **ruff** in CI ✅ |
| Lowercase vocabulary, description on every config field | **automatic** ✅ |
| No magic strings (`DIMS.time`, not `"time"`) | **nothing — you** ❌ |
| Functional purity / no in-place mutation | **nothing — you** ❌ |
| Free function ⟷ accessor parity (exists, signature matches) | **nothing — you** ❌ |
| `__all__` membership | **nothing — you** ❌ |

The magic-string gap is not obvious, so it is worth knowing *why*: `XmrisTerm` subclasses `str`, so
`DIMS.time == "time"` is `True`. `TestAccessorDefaults` compares with `==`, and its own docstring
admits it "will still pass if the default is a bare string that happens to match today." Only
`type(default) is XmrisTerm` tells them apart. There are live violations in-tree today.

**`TestDomainDimRule` is the only auto-discovering test, and it walks a hardcoded list of 8
modules** (`tests/test_core.py:766-775`). A function in a **brand-new module** is invisible to it —
and its `checked >= 10` floor still passes. Adding a module means editing that list.

Three more lists need a hand edit for every new accessor method. They are precise and easy to miss:
**see `templates/tests.md`.** Skipping them is how a function ships with zero architecture coverage.

## 5. Document it

Math and science are tested in MyST notebooks under `docs/notebooks/`, not `test_*.py`. **Use the
`docs-page` skill** (tutorial genre) — it owns the cell structure, the hidden-assert convention,
and the TOC step. Don't hand-roll a notebook from memory here.

(editing)=
## 6. Editing an existing method

| Change | Files to touch together |
|---|---|
| Signature, defaults, new parameter | free function **and** its accessor delegator — keep them in lock-step, nothing checks this |
| Domain contract (`ensures_domain` ⇄ `computes_in` ⇄ none) | the decorator, the `dim` default (the biconditional flips it), and the tuple in `TestDomainRollout` |
| The math only | the function; re-run its notebook |
| Renaming / moving between modules | the function, `accessor.py`, `__init__.py`'s `__all__`, and every hardcoded test list naming it |
| A new `config.py` term | `config.py`, the function, and call it out to the user |

Guardrails: changing a `dim` default means checking `TestAccessorDefaults`' parametrize list for
that method name. Changing a domain decorator means re-reading `domains.md` — a funnel and a
domain-preserving op behave differently on an axis outside their domain, and only one of them
passes data through untouched.

## 7. Verify

```bash
uv run pytest tests/test_core.py -n0 --no-cov    # architecture invariants
uv run ruff format . && uv run ruff check . --fix
uv run mypy src/xmris                            # fix clear type errors
uv run test-gen                                  # regenerate .ipynb from the .md notebooks
uv run pytest "tests/autogen_notebooks/<area>/<name>.ipynb" -n0 --no-cov
```

Then `uv run test` for the full pipeline before wrapping up.

Because the suite cannot check them, verify the ❌ rows of the [coverage table](#coverage) by hand:
grep your diff for bare dim strings, confirm the function returns a new object rather than mutating
its input, and diff the free function's signature against the accessor delegator's.

## 8. Report

Summarize the function, its domain contract, **every new `config.py` term** (call these out
explicitly), which test lists you edited, and where its notebook test lives.

## Checklist

- [ ] Kind routed; `domains.md` consulted for the decorator (or its deliberate absence)
- [ ] Correct module — `referencing.py` for axis referencing, `utils.py` for complex layout
- [ ] `dim` default is a config constant, or `None` iff multi-label domain
- [ ] Returns a new object; no in-place mutation
- [ ] Delegator added to the right mixin (or directly on `XmrisAccessor`), signature matching
- [ ] Imported and listed in `__all__`
- [ ] The three test lists updated (`templates/tests.md`) — plus the module list if new module
- [ ] Notebook written via the `docs-page` skill
- [ ] Architecture tests, ruff, mypy, and the notebook all green
