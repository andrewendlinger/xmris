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

    # Layout traits
    width = traitlets.Int(740).tag(sync=True)
    height = traitlets.Int(400).tag(sync=True)
    show_grid = traitlets.Bool(True).tag(sync=True)

    # Coordinate traits
    x_coords = traitlets.List().tag(sync=True)
    x_label = traitlets.Unicode("Chemical Shift [ppm]").tag(sync=True)

    # Data traits
    reals = traitlets.List().tag(sync=True)
    imags = traitlets.List().tag(sync=True)
    mag = traitlets.List().tag(sync=True)

    # Phase traits
    p0 = traitlets.Float(0.0).tag(sync=True)
    p1 = traitlets.Float(0.0).tag(sync=True)
    pivot_val = traitlets.Float(0.0).tag(sync=True)


# ============================================================================
# 2. Convenience wrapper for xarray
# ============================================================================
def phase_spectrum(
    da: xr.DataArray, width: int = 740, height: int = 400, show_grid: bool = True
) -> PhaseWidget:
    """Create an interactive phase correction viewer for a 1-D complex xarray.

    Automatically detects if the spectral dimension is in ppm or Hz,
    falling back to the raw dimension name if neither is found.
    """
    if da.ndim != 1:
        raise ValueError(f"Need a 1-D DataArray, got {da.ndim}-D. Slice first.")

    if not np.iscomplexobj(da.values):
        raise ValueError("The spectrum must be complex-valued to perform phasing.")

    spec_dim = None
    x_label = ""

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

    if spec_dim is None:
        spec_dim = da.dims[0]
        x_label = str(spec_dim)

    x_vals = da.coords[spec_dim].values.astype(float)
    vals = da.values

    mag_vals = np.abs(vals).astype(float)
    max_idx = np.argmax(mag_vals)
    pivot = float(x_vals[max_idx])

    return PhaseWidget(
        width=width,
        height=height,
        show_grid=show_grid,
        x_coords=x_vals.tolist(),
        x_label=x_label,
        reals=np.real(vals).astype(float).tolist(),
        imags=np.imag(vals).astype(float).tolist(),
        mag=mag_vals.tolist(),
        pivot_val=pivot,
        p0=0.0,
        p1=0.0,
    )
