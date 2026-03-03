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

# Introduction to Config-Based Plotting

Generating publication-ready figures directly from code often results in function signatures cluttered with dozens of keyword arguments (`figsize`, `cmap`, `linewidth`, etc.).

To solve this, `xmris` utilizes a **Config-Based Plotting Architecture**. Every complex plotting function is accessed via the `xarray` accessor (e.g., `da.xmr.plot.ridge()`) and accepts a single, dedicated Configuration Object. This provides a clean API while maintaining infinite customizability, perfectly aligning with our "Xarray in, xarray out" philosophy.

## 1. Environment Setup

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

# 1. Use retina for crisp, PDF-like text
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

# 2. Set a high baseline DPI
plt.rcParams["figure.dpi"] = 150
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import xarray as xr

# Import the global configuration and the RidgePlotConfig object
import xmris
from xmris.visualization.plot import PlotRidgeConfig
```

## 2. The Configuration Object

Instead of passing arguments directly to the plotting functions, we instantiate a specific configuration object, such as `PlotRidgeConfig`. Because `xmris` configs inherit from a rich base class, simply outputting the object in a Jupyter cell renders a beautiful HTML table detailing every available parameter, its default value, and its description.

```{code-cell} ipython3
# Instantiate the config
CFG = PlotRidgeConfig()

# Display it in the notebook to view all available styling options
CFG
```

## 3. Customizing the Configuration

You can easily modify the config object using standard `matplotlib` terminology before passing it to the accessor.

```{code-cell} ipython3
# Update generic plot parameters
CFG.annotation = None  # Remove the top-left annotation
CFG.xlabel = "Chemical Shift"
CFG.cmap = "viridis"  # Change the colormap

print(f"Updated colormap: {CFG.cmap}")
```

In the next tutorial, we will apply this configuration object to real kinetic spectroscopy data.
