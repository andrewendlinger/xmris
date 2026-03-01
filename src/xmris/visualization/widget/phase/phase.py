import pathlib

import anywidget
import numpy as np
import traitlets
import xarray as xr

_HERE = pathlib.Path(__file__).parent


# ============================================================================
# 1. AnyWidget definition
# ============================================================================
class PhaseWidget(anywidget.AnyWidget):
    """Interactive zero and first-order phase correction widget."""

    _esm = _HERE / "phase.js"
    _css = _HERE / "phase.css"

    # Abstracted coordinate traits
    x_coords = traitlets.List().tag(sync=True)
    x_label = traitlets.Unicode("Chemical Shift [ppm]").tag(sync=True)
    reals = traitlets.List().tag(sync=True)
    imags = traitlets.List().tag(sync=True)
    mag = traitlets.List().tag(sync=True)

    p0 = traitlets.Float(0.0).tag(sync=True)
    p1 = traitlets.Float(0.0).tag(sync=True)
    pivot_val = traitlets.Float(0.0).tag(sync=True)


# ============================================================================
# 2. Convenience wrapper for xarray
# ============================================================================
def phase_spectrum(da: xr.DataArray) -> PhaseWidget:
    """Create an interactive phase correction viewer for a 1-D complex xarray.

    Automatically detects if the spectral dimension is in ppm or Hz,
    falling back to the raw dimension name if neither is found.
    """
    if da.ndim != 1:
        raise ValueError(f"Need a 1-D DataArray, got {da.ndim}-D. Slice first.")

    if not np.iscomplexobj(da.values):
        raise ValueError("The spectrum must be complex-valued to perform phasing.")

    # 1. Identify the spectral dimension and its unit
    spec_dim = None
    x_label = ""

    # We sniff the dimension names to match xmris's conventions
    for d in da.dims:
        d_str = str(d).lower()
        if any(k in d_str for k in ("ppm", "chem", "shift")):
            spec_dim = d
            x_label = "Chemical Shift [ppm]"
            break
        elif any(k in d_str for k in ("hz", "freq")):
            spec_dim = d
            x_label = "Frequency [Hz]"
            break

    # Fallback if no explicit string matched: use the raw dimension name
    if spec_dim is None:
        spec_dim = da.dims[0]
        x_label = str(spec_dim)  # e.g., "points" or "time"

    x_vals = da.coords[spec_dim].values.astype(float)
    vals = da.values

    # 2. Calculate magnitude for axis scaling
    mag_vals = np.abs(vals).astype(float)

    # 3. Find the pivot (x-coordinate of the maximum magnitude peak)
    max_idx = np.argmax(mag_vals)
    pivot = float(x_vals[max_idx])

    return PhaseWidget(
        x_coords=x_vals.tolist(),
        x_label=x_label,  # <-- Pass the fully formatted label string
        reals=np.real(vals).astype(float).tolist(),
        imags=np.imag(vals).astype(float).tolist(),
        mag=mag_vals.tolist(),
        pivot_val=pivot,
        p0=0.0,
        p1=0.0,
    )
