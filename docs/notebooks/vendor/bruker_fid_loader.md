# Visual Verification: FID Loader and Processing Pipeline

**Goal:** Visually verify the strict 1D-to-ND Bruker memory reshaping, physical coordinate calculation, and the newly refactored `xmris` processing pipeline. We will explicitly use `xarray`'s native plotting to ensure units and axis names are automatically and correctly resolved.

```{code-cell} ipython3
:tags: [remove-cell]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

# High-res output for crisp visual inspection
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150
```

```{code-cell} ipython3
from pathlib import Path
import xarray as xr

from xmris.vendor.bruker import reshape_bruker_raw, build_fid
from xmris.core.config import DIMS, COORDS

# Configuration
xr.set_options(display_expand_data=False)
```

## 1. Data Loading

```{code-cell} ipython3
DATA_DIR = Path("../../../tests/data/")
FILE_PATH = Path(DATA_DIR / "nspect_slab_1H" / "rawdatajob0.nc")
```

```{code-cell} ipython3
raw_1d_data = xr.load_dataarray(FILE_PATH).xmr.to_complex()
raw_1d_data
```

```{code-cell} ipython3
raw_1d_data.attrs
```

```{code-cell} ipython3
# 2. Reshape into C-contiguous N-dimensional numpy array
reshaped_nd, valid_dims = reshape_bruker_raw(raw_1d_data.values, raw_1d_data.attrs)
```

```{code-cell} ipython3
fid_xr = build_fid(reshaped_nd, valid_dims, raw_1d_data.attrs)

print(f"Constructed FID Shape: {fid_xr.shape}")
print(f"Assigned Attributes: {list(fid_xr.attrs.keys())}")
```

## 2. Apply the `xmris` Pipeline

Transform the time-domain FID into the frequency domain along the `time` dimension, and convert the relative `frequency` coordinates to absolute `chemical_shift` using the Bouncer-enforced metadata.

```{code-cell} ipython3
# Select the first repetition/channel/average for basic visual inspection if multi-dimensional
if fid_xr.ndim > 1:
    fid_inspect = fid_xr.isel({d: 0 for d in fid_xr.dims if d != DIMS.time})
else:
    fid_inspect = fid_xr

fid_inspect
```

## 3. Visualizing the Time Domain (FID)

Verify signal decay shape, complex quadrature, and absence of truncation artifacts. By using `xarray.plot()`, the x-axis should automatically read "Time [s]".

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 4))

# Use xarray's native plotting to verify automatic labels
fid_inspect.real.plot(ax=ax, label="Real", alpha=0.8)
fid_inspect.imag.plot(ax=ax, label="Imaginary", alpha=0.8)

ax.set_title("Time Domain: Ground Truth FID")
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

```{code-cell} ipython3
fid_corrected = fid_inspect.xmr.remove_digital_filter(group_delay=raw_1d_data.attrs["groupDelay"], keep_length=False)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 4))

# Use xarray's native plotting to verify automatic labels
fid_corrected.real.plot(ax=ax, label="Real", alpha=0.8, marker='.')
fid_corrected.imag.plot(ax=ax, label="Imaginary", alpha=0.8, marker='.')

ax.set_title("Time Domain: Ground Truth FID")
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

```{code-cell} ipython3
spectrum = fid_corrected.xmr.to_spectrum().xmr.autophase().xmr.autophase()
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

# --- Subplot 1: Frequency (Hz) ---
spectrum.real.plot(ax=ax1, color="red", marker='.')
# spectrum.imag.plot(x=DIMS.frequency, ax=ax1, color="blue")
ax1.set_title("Frequency Domain")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-50, 50)
ax1.set_ylim(-1e8, 4e8)
ax1.xaxis.set_inverted(True)
# --- Subplot 2: ---
spectrum.imag.plot(ax=ax2, color="navy", marker='.')
ax2.set_title("Chemical Shift Domain")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-50, 50)
# ax2.set_ylim(-1e8, 4e8)
ax2.xaxis.set_inverted(True)

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# Apply the strict xmris pipeline
spectrum = (
    fid_corrected
    .xmr.apodize_exp(lb=30)
    .xmr.to_spectrum()
    .xmr.autophase()
)

# Apply the strict xmris pipeline
# spectrum = (
#     fid_inspect
#     .xmr.to_spectrum()
#     .xmr.phase(p1=-p.acqp["ACQ_RxFilterInfo"][0][0] * 360)
# )

spectrum
```

## 4. Visualizing the Frequency Domain (Spectrum)

Verify that the Fourier transform executed correctly. By passing the explicit x-coordinate to `xarray.plot()`, the axes should automatically label themselves "Frequency [Hz]" and "Chemical Shift [ppm]".

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

# --- Subplot 1: Frequency (Hz) ---
spectrum.real.plot(ax=ax1, color="red", marker='.')
# spectrum.imag.plot(x=DIMS.frequency, ax=ax1, color="blue")
ax1.set_title("Frequency Domain")
ax1.grid(True, alpha=0.3)
# ax1.set_xlim(-200, 200)
# --- Subplot 2: ---
spectrum.imag.plot(ax=ax2, color="navy", marker='.')
ax2.set_title("Chemical Shift Domain")
ax2.grid(True, alpha=0.3)
# ax2.set_xlim(-200, 200)


plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

spectrum_ppm = spectrum.xmr.to_ppm()
# --- Subplot 1: Frequency (Hz) ---
spectrum_ppm.real.plot(ax=ax1, x='chemical_shift', color="red", marker='.')
# spectrum.imag.plot(x=DIMS.frequency, ax=ax1, color="blue")
ax1.set_title("Frequency Domain")
ax1.grid(True, alpha=0.3)
# ax1.set_xlim(3, 6.4)
ax1.xaxis.set_inverted(True)
# --- Subplot 2: ---
spectrum_ppm.imag.plot(ax=ax2, x='chemical_shift', color="navy", marker='.')
ax2.set_title("Chemical Shift Domain")
ax2.grid(True, alpha=0.3)
# ax2.set_xlim(3, 6.4)
ax2.xaxis.set_inverted(True)

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
fid_final = spectrum.xmr.to_fid()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 4))

# Use xarray's native plotting to verify automatic labels
fid_final.real.plot(ax=ax, label="Real", alpha=0.8, marker='.')
fid_final.imag.plot(ax=ax, label="Imaginary", alpha=0.8, marker='.')

ax.set_title("Time Domain: Ground Truth FID")
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
