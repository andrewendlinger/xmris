---
name: new-widget
description: Add or modify an xmris interactive visualization widget (AnyWidget) end-to-end following the project's conventions. Use when adding a new `.xmr.widget.*` component (e.g. a phaser, scroller, apodizer, or other canvas UI), or when working on an existing widget's Python/JS/CSS, accessor wiring, or docs.
---

# Add a new xmris widget

Scaffold an interactive AnyWidget, wire it into the `.xmr.widget` accessor, and
document it — following the project's architecture and the widget-specific
conventions. For edits to an existing widget, jump to
[Working on an existing widget](#existing).

## 0. Read the rules first

Read `docs/contributing/static_widgets.md` — the **canonical widget reference**
(anatomy, conventions, the static-docs exporter). It builds on the "8
Commandments" in `docs/contributing/ai_context.md`; both apply. The reference
implementation to copy is `src/xmris/visualization/widget/phase/`.

## 1. Scaffold the triplet

A widget is a folder `src/xmris/visualization/widget/<name>/` holding
`<name>.py` + `<name>.js` + `<name>.css`. **Copy the `phase/` widget** and rename
— don't hand-write the AnyWidget/canvas plumbing. Keep the `_HERE = pathlib.Path(__file__).parent`
wiring and the `_esm = load_esm(_HERE / "<name>.js")` / `_css = load_css(...)`
calls intact — these concatenate the shared `_shared/{canvas.js,theme.css}`
layer ahead of your files (see `docs/contributing/static_widgets.md`).

## 2. Python: widget class + factory

In `<name>.py` define:
- `class <Name>Widget(anywidget.AnyWidget)` — `_esm`/`_css` wrap the sibling
  JS/CSS with `load_esm`/`load_css` (imported `from .._shared import ...`); every
  value the frontend reads is a `traitlets.*(...).tag(sync=True)` trait (always
  `width`/`height`). Full NumPy docstring with an `Attributes` section for the
  synced traits.
- A factory `<verb>_<noun>(da, dim=None, ...) -> <Name>Widget` that validates,
  extracts numpy arrays, and returns the instance.

Rules that are easy to get wrong here:
- **Wrap a real method** — a widget returns a widget, not a DataArray; its Close
  button must emit a reproducible `.xmr.<method>(...)` snippet. If that method
  doesn't exist yet, add it first (use the `new-processing-method` skill).
- **No name sniffing / no magic strings** — resolve the axis with
  `if dim is None: dim = _resolve_spectral_dim(da)` then `_check_dims(da, dim, "<factory>")`;
  import `DIMS`/`ATTRS`/`COORDS` from `xmris.core.config` when you need vocabulary.
- **Labels from lineage** — build axis labels from `da.coords[dim].attrs`
  (`long_name`/`units`), never a hardcoded string.
- **Functional purity** — never mutate the input; read from it only.

## 3. Frontend: `<name>.js` + `<name>.css`

- One `export function render({ model, el })`: build the DOM, draw to a
  `<canvas>`, and redraw via `model.on("change:…", () => requestAnimationFrame(draw))`.
- **Reuse the shared helpers** (in scope from `_shared/canvas.js`, no import):
  `ticks`/`nfmt`, `setupCanvas`/`resizeCanvas`, `watchTheme`, and `themeColors(el)`
  for canvas colors — don't re-inline them.
- **Close button** carries the class `remove-me-close-btn` (keep the
  `// CONVENTION:` comment) and calls `showSnippetBanner(root, {title, subtitle,
  hint, target})` with the copyable `.xmr.<method>(...)` snippet on click.
- Style with the `nmr-*` CSS namespace and the `--nmr-*` theme tokens (never
  hardcode colors); a widget's `.css` holds only its widget-specific classes.

## 4. Wire it in (2 edits)

1. Re-export the factory in `src/xmris/visualization/widget/__init__.py` and add
   it to `__all__`.
2. Add a thin, lazily-importing method to `XmrisWidgetAccessor` in
   `src/xmris/core/accessor.py`, with a full NumPy docstring and **defaults that
   mirror the factory exactly**. (`.xmr.widget` is already wired via a cached
   property — no change to `XmrisAccessor`.)

## 5. Extend the vocabulary if needed

If the widget needs a dim/coord/attr not in `src/xmris/core/config.py`, add the
`XmrisTerm` there and **explicitly tell the user** every new `ATTRS`/`DIMS`/
`COORDS`/`VARS` term introduced.

## 6. Document & test

Add a MyST notebook under `docs/notebooks/visualization/widget/` and a `myst.yml`
TOC entry. **Use the `new-doc-notebook` skill** for the notebook structure; follow
`docs/contributing/static_widgets.md` for the widget-specific pieces:
- the live-call (`remove-output`) + `export_widget_static` (`remove-input`)
  two-cell pattern;
- a hidden `remove-cell` assertion block that runs the reproducible snippet the
  widget emits and proves the result is correct — values **and** preserved
  dims/coords/attrs;
- keep synced arrays under `max_points` / ~2.5 MB (downsample if needed).

## 7. Verify

```
uv run test-gen                                  # regenerate .ipynb from the .md notebooks
uv run pytest tests/test_core.py -n0 --no-cov    # architecture invariants
uv run ruff format . && uv run ruff check . --fix
uv run mypy src/xmris                            # fix clear type errors
```

Smoke-test the widget instantiates and exports:
`da.xmr.widget.<name>(...)` and `export_widget_static(<factory>, da, ...)`.
Then `uv run test` for the full pipeline before wrapping up.

## 8. Report

Summarize the widget, the `.xmr.<method>` it reproduces, any new `config.py`
terms (call these out explicitly), and where its notebook test lives.

(existing)=
## Working on an existing widget

Same conventions apply. Use this change map to touch the right files:

| Change | Files |
| :-- | :-- |
| Canvas look / interaction / drawing | `<name>/<name>.js`, `<name>/<name>.css` |
| Synced state, validation, data prep | `<name>/<name>.py` (widget class + factory) |
| User-facing params / docstring | factory **and** the `XmrisWidgetAccessor` method — keep signatures and defaults in lock-step |
| Reproducible snippet it emits | the JS Close handler **and** the `.xmr` method it targets |

Guardrails: accessor defaults must mirror the factory; if the JS reimplements a
transform for live preview, keep it consistent with the Python method it mirrors;
regenerate and run the widget's notebook (`uv run test-gen` + nbmake) after any
change.
