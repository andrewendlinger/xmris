---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (xmris)
  language: python
  name: python3
---

# Building & Documenting Interactive Widgets

`xmris` uses [AnyWidget](https://anywidget.dev/) to provide interactive,
browser-based UI components (phase correction, spectra scrolling, apodization)
directly inside Jupyter Notebooks. This page is the **canonical reference for
authoring widgets** — the `xmr-widget` skill defers to it, the same way the
`xmr-method` skill defers to [AI Context](./ai_context.md).

Widgets sit in the visualization layer but must still respect the project's
["8 Commandments"](./ai_context.md). This document adds the widget-specific
conventions on top.

(widget-anatomy)=
## 1. Anatomy of a widget

Every widget is a **triplet** under `src/xmris/visualization/widget/<name>/`,
plus two small wiring edits. The `phase/` widget is the reference implementation
— copy it and adapt.

```text
src/xmris/visualization/widget/
├── __init__.py              # re-exports each factory (edit 1)
├── _static_exporter.py      # export_widget_static — shared, don't touch
├── _shared/                 # shared frontend layer — don't fork per widget
│   ├── __init__.py          # load_esm() / load_css() — concatenate shared + widget
│   ├── canvas.js            # ticks, nfmt, setupCanvas, themeColors, showSnippetBanner…
│   └── theme.css            # `--nmr-*` design tokens (light + dark) + common chrome
└── <name>/
    ├── <name>.py            # AnyWidget subclass + factory function
    ├── <name>.js            # render({ model, el }) — canvas frontend
    └── <name>.css           # widget-specific `nmr-*` styles only
```

**The shared layer.** There is no JS bundler, so code sharing happens in Python:
`load_esm`/`load_css` (in `_shared/__init__.py`) read the common `canvas.js` /
`theme.css` and concatenate them **ahead** of a widget's own file, returning a
single source string. `canvas.js` is deliberately `import`/`export`-free so it
sits in the same module scope as the widget's one `export function render`.
`_static_exporter.py` already accepts `_esm`/`_css` as strings, so this works in
both live AnyWidget and the static docs. Don't reintroduce per-widget copies of
`ticks`/`nfmt`/the close-banner — call the shared helpers.

**Python (`<name>.py`)** — an `anywidget.AnyWidget` subclass declaring the
synchronized state, plus a module-level **factory** that prepares the data and
returns an instance:

```python
import pathlib

import anywidget
import numpy as np
import traitlets
import xarray as xr

from xmris.core.config import SPECTRAL_DIMS
from xmris.core.utils import _check_dims, _resolve_dim, _spectral_axis_label

from .._shared import load_css, load_esm

_HERE = pathlib.Path(__file__).parent


class MyWidget(anywidget.AnyWidget):
    """One-line summary.

    Attributes
    ----------
    width, height : int
        Canvas size in pixels.
    x_coords, reals, imags : list of float
        Synchronized data arrays consumed by the frontend.
    ...
    """

    # load_esm/load_css concatenate the shared _shared/{canvas.js,theme.css}
    # ahead of this widget's own files.
    _esm = load_esm(_HERE / "my_widget.js")
    _css = load_css(_HERE / "my_widget.css")

    width = traitlets.Int(740).tag(sync=True)
    height = traitlets.Int(400).tag(sync=True)
    # ... every piece of state the JS reads is a `.tag(sync=True)` trait


def my_widget(da: xr.DataArray, dim: str | None = None, width: int = 740) -> MyWidget:
    """Factory: validate `da`, extract numpy arrays, return the widget."""
    ...
```

**Frontend (`<name>.js`)** — a single `export function render({ model, el })`.
It builds the DOM, draws to a `<canvas>`, and redraws on trait changes. Use the
shared helpers from `_shared/canvas.js` (already in scope — no import): `ticks`,
`nfmt`, `setupCanvas`/`resizeCanvas`, `showSnippetBanner`, `watchTheme`, and
`themeColors(el)` for canvas colors so the drawing follows light/dark:

```javascript
export function render({ model, el }) {
    const root = document.createElement("div"); // ... build DOM
    const ctx = setupCanvas(canvas, W, H, window.devicePixelRatio || 1);
    function draw() {
        const C = themeColors(root);            // palette from the --nmr-* CSS vars
        // read model.get(...), paint with C.grid / C.real / C.imag …
    }
    model.on("change:p0 change:p1", () => requestAnimationFrame(draw));
    watchTheme(() => requestAnimationFrame(draw)); // redraw on OS theme flip
    draw();
}
```

**Wiring (2 edits):**

1. Re-export the factory in `src/xmris/visualization/widget/__init__.py` and add
   it to `__all__`.
2. Add a thin, lazily-importing method to `XmrisWidgetAccessor` in
   `src/xmris/core/accessor.py`. The `.xmr.widget` namespace is already wired via
   a cached property, so `XmrisAccessor` itself needs no change:

   ```python
   def my_widget(self, dim: str | None = None, width: int = 740, **kwargs):
       """Full NumPy docstring (feeds the quartodoc API reference)."""
       from xmris.visualization.widget import my_widget  # lazy import
       return my_widget(self._obj, dim=dim, width=width, **kwargs)
   ```

(widget-conventions)=
## 2. Authoring conventions

These are the rules a new widget must follow. The first three are what set
widgets apart from the copy-paste heuristics in the older code.

### Reproducibility: return the widget, wrap a real method

Widgets are the **one deliberate exception** to "xarray in, xarray out": a
factory returns a *widget instance*, not a DataArray. Provenance is preserved a
different way — the widget's **Close** button emits a copyable snippet that maps
to a genuine `.xmr` processing call. Therefore **every widget must wrap an
existing `.xmr` method** so its output is reproducible:

| Widget | Reproducible call it emits |
| :-- | :-- |
| `phase_spectrum` | `.xmr.phase(p0=…, p1=…, pivot=…)` |
| `apodize` | `.xmr.apodize_exp(lb=…)` / `.xmr.apodize_lg(lb=…, gb=…)` |
| `scroll_spectra` | `.isel({dim: idx})` |

If there is no processing method to reproduce, add that method first (see the
`xmr-method` skill) — a widget is a UI over real math, never a home for new math.

### Resolve dimensions from the vocabulary — no name sniffing

Do **not** guess the axis with substring checks like `"ppm" in dim`. Resolve it
from the vocabulary, then validate. **Pick the one branch matching your widget's
domain — these are alternatives, not steps.**

A **spectral** widget (`phase`, `scroller`) resolves across the multi-label
spectral domain, which is what `_resolve_dim` exists for:

```python
if dim is None:
    dim = _resolve_dim(da, SPECTRAL_DIMS)   # frequency / chemical_shift
_check_dims(da, dim, "my_widget")
```

A **time-domain** widget (`apodizer`) receives an FID, which has no spectral axis
at all — running the resolver on one raises before it can find anything. Use the
canonical time dimension instead:

```python
if dim is None:
    dim = DIMS.time
_check_dims(da, dim, "my_widget")
```

`_resolve_dim` raises a helpful error for non-standard axis names; a single-label
domain needs no resolver, just the constant. Either way the factory exposes
`dim: str | None = None` as an escape hatch — that `None` is a widget convention,
not the domain-decorator biconditional that governs library functions.

Derive axis labels from the coordinate's **lineage metadata**, not a hardcoded
string. The shared helper does this, including the fallback when the metadata is
absent — call it rather than reassembling the f-string:

```python
coord = da.coords[dim]
label = _spectral_axis_label(dim, coord)   # xmris.core.utils
```

### The Close-button rule (static-docs safety)

The static docs have no Python backend, so any button that needs a live kernel
(Close, Save, Apply) would trap the reader in a broken state. **Add the CSS class
`remove-me-close-btn`** to every such button — `export_widget_static` hides it
automatically.

```javascript
// CONVENTION: Always add 'remove-me-close-btn' to buttons that finalize, close,
// or need a live Jupyter kernel, so the static-docs exporter hides them.
// Keep this comment if you copy this widget as a template.
const closeBtn = document.createElement("button");
closeBtn.className = "nmr-btn nmr-btn-outline remove-me-close-btn";
```

Need to hide extra elements? Pass `hide_selectors=["#save-tooltip", ".menu"]` to
`export_widget_static`.

### Other conventions

- **CSS namespace:** style everything with the `nmr-*` prefix (`nmr-viewer`,
  `nmr-btn`, `nmr-bar`, …) to avoid clashing with notebook/host styles.
- **Theming:** never hardcode colors. In CSS use the `--nmr-*` design tokens
  from `_shared/theme.css` (`var(--nmr-accent)`, `var(--nmr-real)`, …); the
  shared `nmr-*` chrome already lives there, so a widget's own `.css` holds only
  its widget-specific/layout classes. In JS read the same palette via
  `themeColors(el)`. Both light and dark are defined via
  `@media (prefers-color-scheme: dark)` — a widget gets dark mode for free by
  using the tokens, and `watchTheme(redraw)` keeps the canvas in sync when the
  OS preference flips.
- **Accessor mirrors the factory:** the `.xmr.widget.*` method must expose the
  same parameters and the **same default values** as the factory — they are one
  contract in two places. Keep them in lock-step when either changes.
- **Docstrings:** give both the widget class (an `Attributes` section for the
  synced traits) and the factory/accessor method full NumPy docstrings — they
  feed the quartodoc API reference.
- **Client-side math stays in sync:** if the frontend reimplements a transform
  for live preview (e.g. the apodizer's in-browser FFT), it mirrors a Python
  method — keep the two definitions consistent when the math changes.

(widget-render-docs)=
## 3. Rendering a widget in the docs

In a tutorial notebook, show the reader the *live* call but render the *static*
export. Use `:tags: [remove-output]` on the live cell and `:tags: [remove-input]`
on the export cell.

```{code-cell} ipython3
import numpy as np
import xarray as xr

import xmris

# Dummy complex spectrum on a canonical spectral axis (auto-resolves).
f = np.linspace(-20, 20, 1024)
da = xr.DataArray(
    10 / (1 + 1j * f) + np.random.randn(1024) * 0.05,
    dims=["frequency"],
    coords={"frequency": f},
)
```

The reader sees this (its output is stripped with `remove-output`):

```python
da.xmr.widget.phase_spectrum()
```

…while this cell (hidden with `remove-input`) renders the interactive canvas by
passing the **factory** and its arguments to `export_widget_static`:

```{code-cell} ipython3
from xmris.visualization.widget._static_exporter import export_widget_static
from xmris.visualization.widget.phase.phase import phase_spectrum

export_widget_static(
    phase_spectrum,     # the widget factory function
    da,                 # positional args forwarded to the factory
    width=700,          # keyword args forwarded to the factory
)
```

For a non-standard axis name, forward the escape hatch too, e.g.
`export_widget_static(phase_spectrum, da, dim="ppm")`.

(widget-large-data)=
## 4. Handling large datasets & debugging

Browsers cap standalone HTML iframes; above ~2.5 MB they silently render a blank
box. `export_widget_static` guards against this with float compression and hard
limits: exporting any synced array longer than `max_points` (default `100_000`)
raises a `ValueError` at docs-build time.

To inspect the payload, pass `debug=True`:

```{code-cell} ipython3
export_widget_static(
    phase_spectrum,
    da,
    debug=True,
)
```

```text
--- Static Export Debug: PhaseWidget ---
  [Sync] width           : int = 700
  [Sync] x_coords        : Array/List (Size: 1024)
  [Sync] reals           : Array/List (Size: 1024)

  JSON Payload Size : 18.21 KB (0.02 MB)
  Base64 URI Size   : 25.12 KB (0.02 MB)
--------------------------------------------------
```

If a widget's arrays are too large, slice or downsample the `DataArray` before
exporting.

## 5. Documenting & testing

Each widget ships with a MyST notebook under
`docs/notebooks/visualization/widget/` and a `myst.yml` TOC entry. The notebook
*is* the test (nbmake executes it). Use the **`docs-page` skill** for the
notebook structure; the widget-specific pieces it must include are the
live/static two-cell pattern above, and a hidden `remove-cell` assertion block
that runs the reproducible snippet the widget emits and proves the resulting
DataArray is correct (values **and** preserved dims/coords/attrs).
