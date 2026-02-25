# %% [markdown] vscode={"languageId": "plaintext"}
# ---
# title: FID - Transformations
# ---

# %% tags=["remove-cell"]
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

# 1. Use retina for crisp, PDF-like text that never disappears in HTML
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

# 2. Set a high baseline DPI
plt.rcParams["figure.dpi"] = 150
# %% [markdown]
# In Magnetic Resonance Spectroscopy (MRS), the raw data acquired by the scanner is a time-domain Free Induction Decay (FID).
#
# To visualize the chemical resonances, this digital FID signal is processed by a discrete Fourier transformation (DFT) to produce a digital MR spectrum. Because an FID conventionally starts at $t=0$, we perform a standard Fast Fourier Transform (FFT) followed by a frequency-domain shift (`fftshift`) to center the zero-frequency (DC) component.
#
# ```mermaid
# flowchart LR
#     A[Time-Domain / FID] --> B(FFT) --> C(fftshift) --> D[Frequency-Domain / Spectrum]
#
#     style A fill:#e1f5fe,stroke:#01579b,stroke-width:2px
#     style D fill:#e8f5e9,stroke:#2e7d32,s
#
# ```
# ```mermaid
# flowchart LR
#     D[Frequency-Domain / Spectrum] --> E(ifftshift) --> F(IFFT) --> A[Time-Domain / FID]
#
#     style D fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
#     style A fill:#e1f5fe,stroke:#01579b,stroke-width:2px
# ```

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Ensure the accessor is registered
import xmris.core.accessor

# %% [markdown]
# ## 1. Generate a Synthetic FID
# Let's create an FID with two distinct resonances (at 50 Hz and -150 Hz).

# %%
n_points = 1024
dwell_time = 0.001  # 1 ms dwell time implies a spectral width of 1000 Hz
t = np.arange(n_points) * dwell_time

# Two peaks with different frequencies and decay rates
peak1 = np.exp(-t / 0.05) * np.exp(1j * 2 * np.pi * 50 * t)
peak2 = 0.5 * np.exp(-t / 0.03) * np.exp(1j * 2 * np.pi * -150 * t)

da_fid = xr.DataArray(
    peak1 + peak2,
    dims=["Time"],
    coords={"Time": t},
    attrs={"sequence": "FID", "B0": 3.0},
)

da_fid.real.plot(figsize=(8, 3))
plt.title("Synthetic FID (Real Part)")
plt.show()

# %% [markdown]
# ## 2. Convert to Spectrum
# We use `.xmr.to_spectrum()` to perform the FFT and automatically rename the dimension to "Frequency". Notice how the resulting coordinates automatically represent the correct centered frequency axis (ranging from -500 Hz to 500 Hz).

# %%
da_spec = da_fid.xmr.to_spectrum(dim="Time", out_dim="Frequency")

# Plot the magnitude spectrum
np.abs(da_spec).plot(figsize=(8, 3))
plt.title("MR Spectrum (Magnitude)")
plt.xlim(250, -250)  # Standard MRS convention: reverse x-axis
plt.show()

# %% tags=["remove-cell"]
# STRICT TESTS: FID to Spectrum
# 1. Prove metadata preservation
assert "Frequency" in da_spec.dims, "Dimension was not renamed."
assert da_spec.attrs == da_fid.attrs, "Attributes were dropped."

# 2. Prove math and coordinate alignment
_expected_freqs = np.fft.fftshift(np.fft.fftfreq(n_points, d=dwell_time))
_expected_spec = np.fft.fftshift(np.fft.fft(da_fid.values, norm="ortho"))

np.testing.assert_allclose(
    da_spec.coords["Frequency"].values,
    _expected_freqs,
    err_msg="Frequency coordinates are incorrect.",
)
np.testing.assert_allclose(
    da_spec.values, _expected_spec, err_msg="Spectrum values are incorrect."
)

# %% [markdown]
# ## 3. Convert back to FID
# We can accurately recover the original time-domain signal using `.xmr.to_fid()`.

# %%
da_recovered = da_spec.xmr.to_fid(dim="Frequency", out_dim="Time")

da_recovered.real.plot(figsize=(8, 3), linestyle="-", color="orange")
plt.title("Recovered FID (Real Part)")
plt.show()

# %% tags=["remove-cell"]
# STRICT TESTS: Spectrum back to FID
assert "Time" in da_recovered.dims, "Dimension was not renamed back to Time."
np.testing.assert_allclose(
    da_recovered.coords["Time"].values,
    da_fid.coords["Time"].values,
    err_msg="Time coordinates shifted.",
)
np.testing.assert_allclose(
    da_recovered.values,
    da_fid.values,
    err_msg="Data was not accurately recovered.",
    atol=1e-10,
)
