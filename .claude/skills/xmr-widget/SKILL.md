---
name: xmr-widget
description: Add or modify an xmris interactive widget (AnyWidget) — the Python widget class, its JS/CSS, the `.xmr.widget` accessor wiring, or its static-export docs cell. Use when adding a new `.xmr.widget.*` component, changing an existing widget's traits, signature or defaults, fixing its canvas rendering or theming, or updating the reproducible snippet it emits.
---

# Add or change an xmris widget

Most work here is **changing an existing widget** — a trait, a default, canvas drawing, theming,
the snippet it emits. Nothing new has been created in this layer since the apodizer; the last three
widget commits touched only existing files. Both paths use the same rules; jump to
[Editing an existing widget](#editing) for the change map.

## 0. Read the rules first

**`docs/contribute/static_widgets.md` is the canonical widget reference** and this skill routes
to it rather than restating. It owns the anatomy (§1, with full `.py`, `.js` and accessor-method
skeletons), the authoring conventions (§2), the docs-rendering pattern (§3), the size limits (§4),
and testing (§5). Read it before writing anything.

`docs/contribute/contract.md` — the Architecture Contract's Commandments — still applies
underneath, with one deliberate exception noted below.

## 1. What routes here

Widgets under `src/xmris/visualization/widget/`: the widget class, its JS/CSS, the factory, the
`XmrisWidgetAccessor` method, and the notebook that renders it.

**Doesn't route here:**

- The `.xmr` method a widget *wraps* → the **`xmr-method`** skill. A widget is a UI over real math,
  never a home for new math. If the method doesn't exist, add it first.
- Static plots (`visualization/plot/`) → **no skill yet**, overhaul pending.
- Docs pages → **`docs-page`**. Diary entries → **`dev-diary`**.

## 2. The triplet, and the naming trap

Three files, always: `src/xmris/visualization/widget/<folder>/<folder>.{py,js,css}`. All three
existing widgets follow it exactly — no extras, no missing CSS. Copy `phase/` and adapt.

But **the folder name is not the class name and not the factory name**, in two of three cases:

| Folder | Class | Factory | Accessor |
|---|---|---|---|
| `phase/` | `PhaseWidget` | `phase_spectrum` | `.xmr.widget.phase_spectrum` |
| `scroller/` | `ScrollWidget` | `scroll_spectra` | `.xmr.widget.scroll_spectra` |
| `apodizer/` | `ApodizerWidget` | `apodize` | `.xmr.widget.apodize` |

So the import path doubles (`xmris.visualization.widget.phase.phase`), and `<name>` in the
reference doc means the *folder*. Pick the three names deliberately; nothing derives one from
another.

## 3. Resolving `dim` — the branch the reference implementation doesn't show

`static_widgets.md` §2 covers this, but the choice is worth having in front of you because copying
the wrong line is silent:

| Widget operates on | Resolve with | Exemplar |
|---|---|---|
| A spectrum | `_resolve_dim(da, SPECTRAL_DIMS)` | `phase.py:124`, `scroller.py:126` |
| A time-domain FID | `DIMS.time` | `apodizer.py:152` |

Then always `_check_dims(da, dim, "<factory>")`. Axis labels come from
`_spectral_axis_label(dim, coord)` (`src/xmris/core/utils.py:87`) — all three widgets call it;
don't re-inline the `long_name`/`units` f-string.

:::{important}
**`dim=None` means something different here than in `xmr-method`.** In library code, `dim` defaults
to `None` *iff* the function carries a multi-label domain decorator. **Every widget factory takes
`dim: str | None = None`** — including the apodizer, whose fallback is single-label `DIMS.time` —
because widgets carry no domain decorator and resolve at call time. Don't read a widget's `dim=None`
as a domain claim, and don't "fix" it to a config constant.
:::

Note also that `dim` is not always the second parameter: `scroll_spectra(da, scroll_axis, dim, …)`
puts the scroll axis first, so `scroll_spectra(da, "repetition")` binds `scroll_axis`. Match the
accessor to whatever order the factory uses.

## 4. Wire it in

Three edits, two of them code:

1. Re-export the factory in `src/xmris/visualization/widget/__init__.py` and add it to `__all__`.
2. Add a thin, lazily-importing method to `XmrisWidgetAccessor` (`src/xmris/core/accessor.py:104`)
   with a full NumPy docstring and **defaults mirroring the factory exactly**. `.xmr.widget` is
   already wired on `XmrisAccessor` via a manually-cached property (`:724`) — no change there.
3. Add the notebook to the `myst.yml` TOC under the *Widgets* group (`docs/myst.yml:73-80`).

Document only parameters the method actually accepts. All three current methods declare `**kwargs`
and describe it as "passed to the underlying …Widget"; it is forwarded to the *factory*, and no
factory accepts `**kwargs`, so that documented capability raises `TypeError`. Don't copy it.

If the widget needs a dim/coord/attr not in `src/xmris/core/config.py`, add the `XmrisTerm` there
and **tell the user explicitly** — every new `ATTRS`/`DIMS`/`COORDS`/`VARS` term gets tracked.

(coverage)=
## 5. What the toolchain catches — and what it doesn't

**Read this before trusting a green test run.** `tests/test_core.py` does not touch widgets. The
word appears in it once, in a docstring (`:979`). No test imports, instantiates, or exercises any
widget; `TestAccessorDefaults` parametrizes 17 methods and none is a widget, and
`TestAccessorRegistration` checks `.xmr.plot` caching but not `.xmr.widget`.

| Rule | Enforced by |
|---|---|
| NumPy docstrings, typed signatures | **ruff** in CI ✅ |
| Renders, exports, and the emitted snippet reproduces the math | **its notebook**, via nbmake ✅ — the only real test |
| Synced payload within limits | `export_widget_static` **raises** ⚠️ — but only when the notebook runs |
| Accessor ⟷ factory parity (params, defaults, order) | **nothing — you** ❌ |
| `nmr-*` class namespace, `--nmr-*` tokens | **nothing — you** ❌ |
| `remove-me-close-btn` on kernel-dependent buttons | **nothing — you** ❌ |
| Factory re-exported in `widget/__init__.py` | **nothing — you** ❌ |

Two of those ❌ rows fail *invisibly in the rendered docs* rather than at runtime: a missing
`remove-me-close-btn` leaves a button that needs a live kernel, trapping the reader; an unprefixed
CSS class can be restyled by the host page. The reference widget itself violates the namespace rule
— `phase.css:33-34` defines `.leg-re` / `.leg-im` — so copying `phase/` wholesale propagates it.

The size limits are **two independent guards**, both in `_static_exporter.py`: `max_points`
(default `100_000`) is a per-trait `arr.size` check (`:84-89`); the ~2.5 MB cap is on the
serialized JSON (`:127-132`). Downsample before syncing if either is close.

## 6. Document & test

The notebook **is** the test. Add it under `docs/visualization/` — **use the
`docs-page` skill** (tutorial genre) for the cell structure and TOC step, and follow
`static_widgets.md` §3 for the widget-specific two-cell pattern (live call `remove-output`,
`export_widget_static` `remove-input`).

The `remove-cell` assertion block is what makes it a test: run the reproducible snippet the widget
emits and prove the result is right — values **and** preserved dims/coords/attrs. A widget whose
notebook only renders it has been shown to work, not tested.

`export_widget_static` is not re-exported from the package; import it from
`xmris.visualization.widget._static_exporter`.

(editing)=
## 7. Editing an existing widget

| Change | Files to touch together |
|---|---|
| Canvas look, interaction, drawing | `<folder>/<folder>.js`, `<folder>/<folder>.css` |
| Synced state, validation, data prep | `<folder>/<folder>.py` (class + factory) |
| A new synced trait | the class **and** the JS that reads it — `.tag(sync=True)` or the frontend never sees it |
| User-facing params, defaults, docstring | factory **and** the `XmrisWidgetAccessor` method, in lock-step |
| The snippet it emits | the JS Close handler **and** the `.xmr` method it targets |
| Renaming the widget | folder, class, factory, `widget/__init__.py`, accessor method, notebook, `myst.yml` |

Guardrails: if the JS reimplements a transform for live preview (the apodizer's in-browser FFT),
keep it consistent with the Python method it mirrors. Regenerate and run the widget's notebook after
any change — it is the only thing that will catch a break.

## 8. Verify

The notebook is the real check, so run it first:

```bash
uv run test-gen                                                    # .md → .ipynb
uv run pytest "tests/autogen_notebooks/visualization/<nb>.ipynb" -n0 --no-cov
uv run ruff format . && uv run ruff check . --fix
uv run mypy src/xmris                                              # fix clear type errors
uv run pytest tests/test_core.py -n0 --no-cov                      # nothing else broke — not widget coverage
```

Then `uv run test` for the full pipeline before wrapping up.

Because nothing checks them, verify the ❌ rows of the [coverage table](#coverage) by hand: diff the
factory signature against the accessor method's (names, defaults **and** order), grep the diff for
CSS classes lacking the `nmr-` prefix, and confirm every kernel-dependent button carries
`remove-me-close-btn`.

## 9. Report

Summarize the widget, the `.xmr` method its snippet reproduces, **every new `config.py` term**
(call these out explicitly), and where its notebook lives.

## Checklist

<!-- excerpt:start -->
- [ ] `static_widgets.md` read; `phase/` copied rather than hand-written
- [ ] Folder, class and factory names chosen deliberately
- [ ] `dim` resolved by domain — `_resolve_dim(da, SPECTRAL_DIMS)` or `DIMS.time` — then `_check_dims`
- [ ] Axis label via `_spectral_axis_label`, not a hardcoded string
- [ ] Every kernel-dependent button carries `remove-me-close-btn`
- [ ] Classes `nmr-*`, colors `var(--nmr-*)`, canvas colors via `themeColors(el)`
- [ ] Factory re-exported; accessor method mirrors it exactly; `myst.yml` TOC entry added
- [ ] Notebook written via `docs-page`, with a `remove-cell` block asserting the emitted snippet
- [ ] Notebook, ruff, mypy green — and the architecture suite still passes
<!-- excerpt:end -->
