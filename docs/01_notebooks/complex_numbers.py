# %% [markdown] vscode={"languageId": "plaintext"}
# ---
# title: Utilities - Complex and Real Formats
# ---

# %% tags=["remove-cell"]
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

# 1. Use retina for crisp, PDF-like text that never disappears in HTML
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

# 2. Set a high baseline DPI
plt.rcParams["figure.dpi"] = 150

# %% [markdown]
#
# Magnetic Resonance signals are inherently complex-valued, consisting of both Real and Imaginary components (representing the X and Y magnetization vectors in the rotating frame).
#
# However, many modern data pipelines—specifically **Machine Learning frameworks like PyTorch and TensorFlow**—do not uniformly support complex data types across all of their layers and loss functions.
#
# To feed MR data into these networks, a standard workaround is to split the 1D complex array into a 2D real-valued array, essentially treating the Real and Imaginary components as two separate "channels" (much like RGB channels in an image).
#
# `xmris` provides fluid utilities to transition back and forth between these representations while preserving all coordinates and metadata.

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Ensure the accessor is imported so .xmr is registered
import xmris

# %% [markdown]
# ## 1. Creating a Complex Array
# Let's start by simulating a standard, complex-valued Free Induction Decay (FID).

# %%
# Generate a simple synthetic complex FID
t = np.linspace(0, 1, 512)
freq = 15.0  # Hz
decay = 3.0
complex_fid = np.exp(-t * decay) * np.exp(1j * 2 * np.pi * freq * t)

da_complex = xr.DataArray(
    complex_fid,
    dims=["Time"],
    coords={"Time": t},
    attrs={"sequence": "FID", "B0": 3.0},
    name="Signal",
)

print("Original DataArray:")
print(f"Shape: {da_complex.shape}")
print(f"Data Type: {da_complex.dtype}")
da_complex

# %% [markdown]
# ## 2. Splitting into Real and Imaginary Channels
#
# Using `.xmr.to_real_imag()`, we can expand this array. The function will stack the real and imaginary components along a brand new dimension (defaulting to `"Complex"`).
#
# Notice how the data type changes from `complex128` to `float64`, and the shape expands from `(512,)` to `(512, 2)`.

# %%
da_split = da_complex.xmr.to_real_imag()

print("Split DataArray:")
print(f"Shape: {da_split.shape}")
print(f"Data Type: {da_split.dtype}")
da_split

# %% [markdown]
# Because this is still an `xarray.DataArray`, we can easily plot the two channels side-by-side using standard xarray plotting utilities.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
da_split.plot.line(ax=ax, x="Time", hue="Complex")
ax.set_title("FID Split into Real and Imaginary Channels")
plt.show()

# %% [markdown]
# ## 3. Reconstructing the Complex Array
#
# After running your data through a neural network or saving it to a legacy file format, you will likely need to reconstruct it into a proper complex array to perform standard signal processing (like an FFT or phase correction).
#
# We use `.xmr.to_complex()` to collapse the dimension back down.

# %%
da_reconstructed = da_split.xmr.to_complex()

print("Reconstructed DataArray:")
print(f"Shape: {da_reconstructed.shape}")
print(f"Data Type: {da_reconstructed.dtype}")

# Prove mathematically that we recovered the exact original data
is_identical = np.allclose(da_complex.values, da_reconstructed.values)
print(f"\nExact recovery successful: {is_identical}")

# %% tags=["remove-cell"]
# STRICT TESTS FOR CI
# 1. Test splitting dimensionality and types
assert da_split.ndim == da_complex.ndim + 1, "Dimension was not added."
assert da_split.sizes["Complex"] == 2, "Complex dimension should have size 2."
assert not np.iscomplexobj(da_split.values), "Split array should be strictly real."
assert da_split.name == da_complex.name, "Name attribute lost."
assert da_split.attrs["sequence"] == "FID", "Attributes were lost."

# 2. Test coordinate labels
assert list(da_split.coords["Complex"].values) == ["Real", "Imag"], (
    "Coordinate labels incorrect."
)

# 3. Test mathematical accuracy of split
np.testing.assert_array_equal(da_split.sel(Complex="Real").values, da_complex.real.values)
np.testing.assert_array_equal(da_split.sel(Complex="Imag").values, da_complex.imag.values)

# 4. Test reconstruction dimensionality and types
assert da_reconstructed.ndim == da_complex.ndim, (
    "Reconstruction failed to drop dimension."
)
assert np.iscomplexobj(da_reconstructed.values), "Reconstructed array should be complex."
assert "Complex" not in da_reconstructed.dims, "Complex dimension was not fully dropped."

# 5. Test reconstruction mathematical purity
np.testing.assert_array_equal(
    da_reconstructed.values,
    da_complex.values,
    err_msg="Reconstructed complex values do not match original.",
)
assert da_reconstructed.attrs["B0"] == 3.0, "Reconstruction lost attributes."

# 6. Test custom dimension and labels
da_custom = da_complex.xmr.to_real_imag(dim="Channel", labels=("R", "I"))
assert "Channel" in da_custom.dims
assert list(da_custom.coords["Channel"].values) == ["R", "I"]
da_custom_recon = da_custom.xmr.to_complex(dim="Channel", labels=("R", "I"))
np.testing.assert_array_equal(da_custom_recon.values, da_complex.values)
