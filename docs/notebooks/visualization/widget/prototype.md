---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import xarray as xr
import xmris
import numpy as np

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150
```

## 2. Launching the Widget

You can launch the interactive viewer directly from the `xmris` package.

In a **live Jupyter notebook**, use the full interactive widget:

```{code-cell} ipython3
# Generate synthetic FID
dwell_time = 0.001
n_points = 1024
t = np.arange(n_points) * dwell_time

rng = np.random.default_rng(42)
clean_fid = np.exp(-t / 0.05) * (
    np.exp(1j * 2 * np.pi * 50 * t) + 0.6 * np.exp(1j * 2 * np.pi * -150 * t)
)
noise = rng.normal(scale=0.08, size=n_points) + 1j * rng.normal(scale=0.08, size=n_points)

da_fid = xr.DataArray(
    clean_fid + noise, dims=["time"], coords={"time": t}
)
da_spec = da_fid.xmr.to_spectrum()

# Intentionally ruin the phase
da_ruined = da_spec.xmr.phase(p0=120.0, p1=-45.0)
```

```{code-cell} ipython3
from xmris.visualization.widget._static_exporter import export_widget_static
from xmris.visualization.widget.phase.phase import phase_spectrum

# Call export_widget_static by passing the original function and all desired arguments
export_widget_static(
    phase_spectrum,     # The widget factory function
    da_ruined,          # Positional arg passed to phase_spectrum
    width=700,          # Kwarg passed to phase_spectrum
    height_padding=120,
    max_points=100_000,     # Kwarg caught by export_widget_static for safety checking
    debug=True
)
```

```{code-cell} ipython3
# Generate synthetic 2D data
n_reps = 10
n_points = 1024
ppm = np.linspace(10, -2, n_points)
repetitions = np.arange(n_reps)

# Simulate a decaying peak at 4.7 ppm (Water) and a stable peak at 2.0 ppm (NAA)
ppm_mesh, rep_mesh = np.meshgrid(ppm, repetitions)

# Peak 1: Decaying
peak_water = 10.0 * np.exp(-rep_mesh / 15.0) / (1 + ((ppm_mesh - 4.7) / 0.1)**2)
# Peak 2: Stable
peak_naa = 3.0 / (1 + ((ppm_mesh - 2.0) / 0.05)**2)

clean_data = peak_water + peak_naa

rng = np.random.default_rng(42)
noise = rng.normal(scale=0.2, size=(n_reps, n_points))
data_2d = clean_data + noise

# Inject an artifact at index 25
data_2d[8, :] += 5.0 * np.sin(2 * np.pi * 5 * ppm)

# Build the xarray DataArray
da_series = xr.DataArray(
    data_2d,
    dims=["repetitions", "ppm"],
    coords={"repetitions": repetitions, "ppm": ppm},
    attrs={"xmris_synthetic": True}
)
```

```{code-cell} ipython3
:tags: [remove-input]

from xmris.visualization.widget._static_exporter import export_widget_static
from xmris.visualization.widget.scroller.scroller import scroll_spectra

# Call export_widget_static by passing the original function and all desired arguments
export_widget_static(
    scroll_spectra,     # The widget factory function
    da_series,          # Positional arg passed to phase_spectrum
    width=700,          # Kwarg passed to phase_spectrum
    height_padding=120,
    max_points=100_000,     # Kwarg caught by export_widget_static for safety checking
    debug=True
)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
