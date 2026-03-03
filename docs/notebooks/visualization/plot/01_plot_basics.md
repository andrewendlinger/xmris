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

Generating publication-ready figures directly from code often results in function signatures cluttered with dozens of keyword arguments (`figsize`, `cmap`, `linewidth`, `alpha`, etc.). This leads to unreadable code and makes it difficult to maintain consistent styling across a project.

To solve this, `xmris` utilizes a **Config-Based Plotting Architecture**. Every complex plotting function accessed via the `xarray` accessor (e.g., `da.xmr.plot.waterfall()`) accepts a single, dedicated Configuration Object. 

This approach provides three major benefits:
1. **Clean API:** Your core processing scripts remain highly readable.
2. **Discoverability:** Because the configs are built on Python `dataclasses`, your IDE (like VSCode or PyCharm) will provide instant autocompletion and type hints for every available styling parameter.
3. **Reproducibility:** You can define a single configuration object at the top of your notebook and pass it to multiple plots to guarantee uniform aesthetics.

::: {note}
To get familiar with this concept we will use the [waterfall plot](./02_plot_waterfall.md) as an example here.
:::


## 1. Environment Setup

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

# Use retina for crisp, PDF-like text in the notebook
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

# Set a high baseline DPI for inline visualizations
plt.rcParams["figure.dpi"] = 150
```

```{code-cell} ipython3
import xarray as xr
import matplotlib.pyplot as plt

# Import the specific configuration object you need
from xmris.visualization import WaterfallConfig
```

## 2. The Configuration Object

Instead of passing styling arguments directly to the plotting function, we instantiate a specific configuration object.

Because `xmris` configs inherit from a rich display base class, simply outputting the object in a Jupyter cell renders a beautifully formatted table. This table details every available parameter, logically groups them, and shows their current values and data types.

```{code-cell} ipython3
# Instantiate the default configuration
CFG = WaterfallConfig()

# Display it in the notebook to explore all available styling options
CFG
```

## 3. Customizing the Configuration

You can easily modify the config object using standard attribute assignment. Thanks to the underlying architecture, your IDE will catch typos and warn you if you try to pass a string to a parameter that expects a float.

```{code-cell} ipython3
# Update general plot parameters
CFG.annotation = None  # Remove the top-left text annotation entirely
CFG.xlabel = "Chemical Shift"

# Update stack aesthetics
CFG.cmap = "viridis"   # Change the colormap
CFG.alpha = 0.85       # Make the filled areas more opaque

print(f"Current colormap: {CFG.cmap}")
print(f"Current alpha: {CFG.alpha}")
```

## 4. Applying the Configuration

Once your configuration object is tailored to your liking, simply pass it to the corresponding `xarray` accessor.

```python
# Conceptual Example:
# ax = my_dataset.xmr.plot.waterfall(config=CFG)
# plt.show()

```

In the next tutorial, we will generate real kinetic spectroscopy data and apply this exact configuration object to visualize a dynamic 2D time-series.
