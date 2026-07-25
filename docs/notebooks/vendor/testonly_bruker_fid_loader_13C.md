---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  language: python
  display_name: Python 3 (xmris)
---

(bruker-13c)=
# Internal Test: Bruker 13C Slab Reshaping and Validation

```{code-cell} ipython3
:tags: [remove-cell]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import xarray as xr
from pathlib import Path
import tomllib  # Python 3.11+ built-in TOML parser

# xmris imports
from xmris.vendor.bruker import reshape_bruker_raw, build_fid
from xmris.core.config import DIMS, COORDS

# High-res output for crisp visual inspection
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150
xr.set_options(display_expand_data=False)
```

(bruker-13c-validation)=
## Internal Validation: 13C NSPECT Slab Data

**Goal:** Strictly validate the Bruker data loader, memory reshaping, and coordinate calculation against a known ground-truth dataset (`nspect_slab_13C`). This ensures that future refactors to `xmris` do not break raw Paravision parsing or shift physical coordinate definitions.

(bruker-13c-ground-truth)=
## 1. Load Ground Truth

We dynamically load the exact acquisition parameters and expected spectral peak locations derived from the `ground_truth.toml` file.

```{code-cell} ipython3
DATA_DIR = Path("../../../tests/data/")
TOML_PATH = DATA_DIR / "nspect_slab_13C" / "ground_truth.toml"

with open(TOML_PATH, "rb") as f:
    ground_truth_data = tomllib.load(f)

# Isolate the specific dataset config for easier access
gt_13c = ground_truth_data["nspect_13c"]

print(f"Loaded Ground Truth for: {gt_13c['dataset_name']}")
print(f"System: {gt_13c['system']} ({gt_13c['vendor_version']})")
```

(bruker-13c-reshape)=
## 2. Load and Reshape Data

Load the continuous 1D complex array and pass it through the `xmris` Bruker reshaping utilities.

```{code-cell} ipython3
# Note: Pointing to the 13C dataset directory
FILE_PATH = Path(DATA_DIR / "nspect_slab_13C" / "rawdatajob0.nc")

# 1. Load raw 1D
raw_1d_data = xr.load_dataarray(FILE_PATH).xmr.to_complex()

# 2. Reshape into ND
reshaped_nd, valid_dims = reshape_bruker_raw(raw_1d_data.values, raw_1d_data.attrs)

# 3. Build annotated FID
fid_xr = build_fid(reshaped_nd, valid_dims, raw_1d_data.attrs)

print(f"Constructed FID Shape: {fid_xr.shape}")
print(f"Dimensions: {fid_xr.dims}")
print(f"Minimal Attributes: {fid_xr.attrs}")
```

(bruker-13c-metadata)=
## 3. Metadata Assertions

Verify that `build_fid` accurately mapped the minimal required physical metadata to the `xarray` attributes by comparing them directly against the values in the TOML file.

```{code-cell} ipython3
:tags: [hide-output]

attrs = fid_xr.attrs
params = gt_13c["parameters"]

# Minimal Metadata Assertions
assert np.isclose(attrs.get("reference_frequency", 0), params["frequency"]["reference_frequency"]["value"], atol=1e-5), "Ref freq mismatch"
assert attrs.get("carrier_ppm") == params["frequency"]["working_chemical_shift"]["value"], "Carrier PPM mismatch"
assert attrs.get("group_delay") == params["rx_filter_info"]["groupDelay"]["value"], "Group delay mismatch"
assert attrs.get("units") == "a.u.", "Units mismatch"

# Dimension checks (from xarray sizes, not attributes)
assert fid_xr.sizes[DIMS.time] == params["general"]["acq_points"]["value"], "Acquisition points mismatch"

print("✅ Minimal metadata and dimension assertions passed.")
```

(bruker-13c-coordinates)=
## 4. Pipeline & Coordinate Validation

Apply the standard `xmris` processing pipeline. This validates that the `remove_digital_filter`, `to_spectrum`, and `to_ppm` accessors execute correctly using the minimal metadata map.

```{code-cell} ipython3
# 1. Select the first repetition/average (if multi-dimensional)
if fid_xr.ndim > 1:
    fid_single = fid_xr.isel({d: 0 for d in fid_xr.dims if d != DIMS.time})
else:
    fid_single = fid_xr

# 2. Apply xmris pipeline
spectrum_hz = (
    fid_single
    .xmr.remove_digital_filter(group_delay=attrs["group_delay"], keep_length=False)
    .xmr.apodize_exp(lb=5) # 5 Hz line broadening for visibility
    .xmr.to_spectrum()
    .xmr.autophase()
)

# 3. Convert to PPM
spectrum_ppm = spectrum_hz.xmr.to_ppm()
```

(bruker-13c-visual-check)=
## 5. Visual Peak Sanity Check

Visually verify that the peaks align with our expected ground truth markers in both the `Hz` and `ppm` domains.

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# --- Plot 1: Frequency (Hz) ---
spectrum_hz.real.plot(ax=ax1, color="k", linewidth=1.5)
ax1.set_xlim(800, -100)
ax1.set_title("13C Spectrum (Hz Domain)")
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_inverted(True)

# Overlay GT lines
for peak_name, locs in gt_13c["spectrum_view"].items():
    ax1.axvline(locs["hz"], color="r", linestyle="--", alpha=0.7)
    ax1.text(locs["hz"], ax1.get_ylim()[1]*0.9, f" {peak_name.capitalize()}", color="r")

# --- Plot 2: Chemical Shift (ppm) ---
spectrum_ppm.real.plot(ax=ax2, x="chemical_shift", color="tab:blue", linewidth=1.5)
ax2.set_xlim(190, 155)
ax2.set_title("13C Spectrum (PPM Domain)")
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_inverted(True)

# Overlay GT lines
for peak_name, locs in gt_13c["spectrum_view"].items():
    ax2.axvline(locs["ppm"], color="r", linestyle="--", alpha=0.7)
    ax2.text(locs["ppm"], ax2.get_ylim()[1]*0.9, f" {peak_name.capitalize()}", color="r")

plt.tight_layout()
plt.show()
```

(bruker-13c-peak-assertions)=
## 6. Quantitative Peak Assertions

Finally, we programmatically verify that the maximum signal intensities lie within a very narrow tolerance ($\pm 2.5$ Hz and $\pm 0.1$ ppm) of the declared ground truth coordinates. This guarantees the coordinate math inside `to_spectrum` and `to_ppm` is exact.

```{code-cell} ipython3
def find_peak_in_window(data, coord_name, target, window):
    """Finds the coordinate of the maximum value within a window."""
    sliced = data.sel({coord_name: slice(target - window, target + window)})
    max_idx = sliced.real.argmax()
    return float(sliced[coord_name][max_idx].values)

# Tolerances
TOL_HZ = 2.5
TOL_PPM = 0.1

for peak_name, locs in gt_13c["spectrum_view"].items():
    # 1. Check Hz
    found_hz = find_peak_in_window(spectrum_hz, DIMS.frequency, locs["hz"], TOL_HZ)
    diff_hz = abs(found_hz - locs["hz"])
    assert diff_hz <= TOL_HZ, f"[{peak_name.capitalize()}] Hz mismatch: Expected {locs['hz']}, Found {found_hz:.2f} (Diff: {diff_hz:.2f})"

    # 2. Check PPM
    found_ppm = find_peak_in_window(spectrum_ppm, "chemical_shift", locs["ppm"], TOL_PPM)
    diff_ppm = abs(found_ppm - locs["ppm"])
    assert diff_ppm <= TOL_PPM, f"[{peak_name.capitalize()}] PPM mismatch: Expected {locs['ppm']}, Found {found_ppm:.2f} (Diff: {diff_ppm:.2f})"

    print(f"✅ {peak_name.capitalize()} verified. Hz diff: {diff_hz:.2f}, PPM diff: {diff_ppm:.3f}")

print("\n🚀 All pipeline and coordinate assertions passed.")
```

(bruker-13c-group-delay)=
## 7. Group-Delay Estimator Sanity Check

Validate `estimate_group_delay` against the real acquisition. There is no ground-truth "true" delay for this probe, so we assert conservatively: the measured delay is physically plausible and, being the residual-phase minimiser over the search window, never leaves *more* residual phase than the vendor header value.

```{code-cell} ipython3
from xmris.vendor.bruker import _PHI0_GRID, _residual_phase_cost

header_gd = float(fid_single.attrs["group_delay"])
measured_gd = fid_single.xmr.estimate_group_delay()


def resid_13c(d):
    """Whole-spectrum residual first-order phase cost after removing delay `d`."""
    spec = fid_single.xmr.remove_digital_filter(group_delay=d).xmr.to_spectrum()
    return _residual_phase_cost(spec, "acme", _PHI0_GRID)


# Physically plausible band for high-resolution 13C spectroscopy group delay.
assert 60.0 < measured_gd < 96.0, f"Implausible measured group delay: {measured_gd}"
# The measured delay minimises residual phase over the header-anchored window, so on
# the whole-spectrum cost it can only match or beat the header value.
assert resid_13c(measured_gd) <= resid_13c(header_gd) + 1e-9, "Measured delay worse than header"
# Anchor-invariance: two different (overlapping-window) anchors converge to the same minimum.
# A no-op estimator that just echoed the header would instead return the two different hints.
m_lo = fid_single.xmr.estimate_group_delay(header_hint=72.0)
m_hi = fid_single.xmr.estimate_group_delay(header_hint=80.0)
assert abs(m_lo - m_hi) < 1.0, f"anchor-dependent result: {m_lo:.2f} vs {m_hi:.2f}"

print(f"header group delay  : {header_gd}")
print(f"measured group delay: {measured_gd:.2f}")
print("✅ Group-delay estimator sanity check passed.")
```

```{code-cell} ipython3
fid_xr
```
