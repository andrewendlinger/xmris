# %% [markdown] vscode={"languageId": "plaintext"}
# ---
# title: Core Naming Conventions
# ---

# %% tags=["remove-cell"]
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150

# %% [markdown]
# ## The Magic String Problem
#
# In Magnetic Resonance, data formats are notoriously fragmented. Vendors use entirely different coordinate systems, units, and metadata keys. If a processing package hardcodes "magic strings" like `"Time"` or `"MHz"` directly into its functions, it inevitably breaks when a user imports data with a `"t"` axis or a `"tx_freq"` attribute.
#
# It also forces users to continually supply repetitive arguments for every step of their pipeline:
#
# ```python
# # The old, exhaustive way
# da_phased = phase(da, time_dim="Time", freq_dim="Frequency")
# da_fit = fit(da_phased, time_dim="Time", mhz_key="MHz")
# ```
#
# ## The Backend `DEFAULTS` Solution
#
# To solve this, `xmris` utilizes a **Global Configuration Architecture** under the hood. We encapsulate standard `xarray` anatomy (`Dimensions`, `Coordinates`, and `Attributes`) into a single `DEFAULTS` object used by the backend.
#
# ```mermaid
# graph TD
#     A[xmris.config.DEFAULTS] -->|Defines Dimensions| B(Data Loaders)
#     A -->|Defines Attributes| C(Processing Functions)
#     A -->|Defines Units| D(Plotting / UI)
#
#     style A fill:#f9f,stroke:#333,stroke-width:2px
# ```
#
# :::{important}
# For the end-user, this means you **do not need to interact with the configuration object**. You simply name your dimensions using standard conventions (e.g., `"time"`, `"chemical_shift"`), and the backend functions will automatically discover them. You only ever need to import the configuration if you want to override the default naming scheme for a custom, non-standard dataset.
# :::
#
# Let's see this backend architecture in action using the `to_real_imag()` complex-splitting utility.

# %%
import numpy as np
import xarray as xr

# Ensure the accessor is registered
import xmris

# %% [markdown]
# ### 1. Default Behavior
#
# We generate a dataset using standard lowercase dimension names (e.g., `"time"`). When we call an `xmris` function without any arguments, it successfully executes because it queries the backend `DEFAULTS` object to figure out what to do.

# %%
# Generate a synthetic complex FID
t = np.linspace(0, 1, 512)
complex_fid = np.exp(-t * 3.0) * np.exp(1j * 2 * np.pi * 15.0 * t)

da_complex = xr.DataArray(
    complex_fid,
    dims=["time"],
    coords={"time": t},
)

# Split the array.
# The backend dynamically queries the config to determine that the
# new dimension should be named "component" with coordinates "real" and "imag".
da_split = da_complex.xmr.to_real_imag()

print("Automatically Generated Dimension:", da_split.dims)

# %% [markdown]
# ### 2. Overriding the Defaults Globally
#
# Suppose you are working in a highly specialized pipeline where the channel dimension must *always* be called `"feature_map"` instead of `"component"`. Passing `dim="feature_map"` into every single processing function would be tedious.
#
# Instead, you can import the `DEFAULTS` object at the top of your script and change it globally. From that point on, every `xmris` function will instantly adapt to your new convention.

# %%
from xmris.config import DEFAULTS

# Override the naming convention globally
DEFAULTS.component.dim = "feature_map"
DEFAULTS.component.coords = ("r_map", "i_map")

# Run the exact same function with no arguments
da_global = da_complex.xmr.to_real_imag()

print("Global Override Dimension:", da_global.dims)
print("Global Override Coordinates:", da_global.coords["feature_map"].values)

# %% tags=["remove-cell"]
# Reset for CI tests
DEFAULTS.component.dim = "component"
DEFAULTS.component.coords = ("real", "imag")

# STRICT TESTS FOR CI
assert "feature_map" in da_global.dims, "Global override failed to apply dimension."
assert list(da_global.coords["feature_map"].values) == ["r_map", "i_map"], (
    "Global override failed to apply coordinates."
)
