---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Documenting Interactive Widgets


The `xmris` package heavily utilizes [AnyWidget](https://anywidget.dev/) to provide interactive, browser-based UI components (like phase correction and spectra scrolling) directly inside Jupyter Notebooks. 

However, standard Jupyter widgets require a live Python kernel to handle two-way communication. This creates a problem for our static documentation (built with Sphinx/MyST), which consists of pre-rendered HTML without an active Python backend.



To solve this, we created the **Universal Static Exporter** (`export_widget_static`). This utility intercepts a live widget, extracts its synchronized state, bundles its JavaScript and CSS, and injects a mock Jupyter model to render a fully functional, standalone HTML iframe.

## 1. How to Use the Static Exporter

When writing documentation tutorials, you will typically show the user how to instantiate the live widget, but then actually render the *static* version so it appears on the website.

To do this, use the `:tags: [remove-input]` MyST directive to hide the static export code from the reader.

```{code-cell} ipython3
import xarray as xr
import numpy as np

# 1. Create your dummy data
da = xr.DataArray(10 / (1 + 1j * np.linspace(-20, 20, 1024)) + np.random.randn(1024) * 0.05, dims=["ppm"])
```

To display the widget in the docs, pass the **widget factory function** and **all its arguments** directly to `export_widget_static`:

```{code-cell} ipython3
:tags: [remove-input]

from xmris.visualization.widget._static_exporter import export_widget_static
from xmris.visualization.widget.phase.phase import phase_spectrum

# This will render the interactive canvas in the docs!
export_widget_static(
    phase_spectrum,     # The widget generating function
    da,                 # Positional arguments
    width=700,          # Keyword arguments
    show_grid=True      # Keyword arguments
)
```

## 2. Widget Development Conventions

Because the static documentation version has no Python backend, buttons that trigger Python code (e.g., "Save to Workspace", "Close", "Apply Phase") will not work and can trap the user in a broken state.

**The Rule:** If your widget has a button that relies on the live Jupyter kernel, you must add the CSS class `xmris-close-btn` to it in your Javascript.

```javascript
// Example in my_widget.js
const closeBtn = document.createElement("button");

// Add 'xmris-close-btn' so the documentation exporter knows to hide it!
closeBtn.className = "nmr-btn nmr-btn-outline xmris-close-btn"; 
closeBtn.textContent = "Finalize & Close";

closeBtn.onclick = () => { model.send({ event: "save" }); };

```

The `export_widget_static` utility automatically searches for and hides any DOM elements with the `.xmris-close-btn` class.

If you need to hide additional, specific elements, you can pass them via the `hide_selectors` argument:

```python
export_widget_static(
    my_widget, data, 
    hide_selectors=["#save-tooltip", ".advanced-menu"]
)

```

## 3. Handling Large Datasets & Debugging

Web browsers enforce strict size limits on standalone HTML iframes. If our static widget payload exceeds ~2.5 MB, the browser will silently fail and render a blank white box.

To prevent this, `export_widget_static` includes automatic float compression and aggressive safety checks. If you attempt to export an array larger than `max_points` (default `100_000`), Python will raise a `ValueError` during the documentation build.

**Debugging your Export:**
If your widget is failing to render or you want to inspect the payload size, set `debug=True`.

```{code-cell} ipython3
export_widget_static(
    phase_spectrum,
    da,
    debug=True
)
```

**Output:**

```text
--- Static Export Debug: PhaseWidget ---
  [Sync] width           : int = 700
  [Sync] height          : int = 400
  [Sync] x_coords        : Array/List (Size: 1024)
  [Sync] reals           : Array/List (Size: 1024)
  
  JSON Payload Size : 18.21 KB (0.02 MB)
  Base64 URI Size   : 25.12 KB (0.02 MB)
--------------------------------------------------

```
