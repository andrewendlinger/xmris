import pathlib

import anywidget
import numpy as np
import traitlets
import xarray as xr

_HERE = pathlib.Path(__file__).parent


class ApodizerWidget(anywidget.AnyWidget):
    """Interactive widget for NMR spectra apodization."""

    _esm = _HERE / "apodizer.js"
    _css = _HERE / "apodizer.css"

    width = traitlets.Int(740).tag(sync=True)
    height = traitlets.Int(550).tag(sync=True)

    t_coords = traitlets.List().tag(sync=True)
    x_coords = traitlets.List().tag(sync=True)
    x_label = traitlets.Unicode("Chemical Shift [ppm]").tag(sync=True)

    reals_t = traitlets.List().tag(sync=True)
    imags_t = traitlets.List().tag(sync=True)

    lb = traitlets.Float(0.0).tag(sync=True)
    gb = traitlets.Float(0.0).tag(sync=True)

    # Configurable limits
    lb_min = traitlets.Float(0.0).tag(sync=True)
    lb_max = traitlets.Float(50.0).tag(sync=True)
    gb_min = traitlets.Float(0.0).tag(sync=True)
    gb_max = traitlets.Float(50.0).tag(sync=True)

    method = traitlets.Unicode("exp").tag(sync=True)
    display_mode = traitlets.Unicode("real").tag(sync=True)
    show_orig = traitlets.Bool(False).tag(sync=True)
    show_grid = traitlets.Bool(True).tag(sync=True)


def apodize_interactive(
    da: xr.DataArray,
    dim: str | None = None,
    unit: str = "ppm",
    width: int = 740,
    height: int = 550,
    lb_range: tuple[float, float] = (0.0, 50.0),
    gb_range: tuple[float, float] = (0.0, 50.0),
) -> ApodizerWidget:
    """
    Instantiate an interactive viewer to apply and visualize apodization.

    Parameters
    ----------
    da : xr.DataArray
        A 1-dimensional, complex-valued DataArray in the time domain.
    dim : str, optional
        The time dimension. Auto-detected if not provided.
    unit : {'ppm', 'hz'}, optional
        The unit for the spectral x-axis. Default is 'ppm'.
    width : int, optional
        Width of the widget in pixels. Default is 740.
    height : int, optional
        Height of the widget in pixels. Default is 550.
    lb_range : tuple of float, optional
        The (min, max) range for the Line Broadening slider. Default is (0, 50).
    gb_range : tuple of float, optional
        The (min, max) range for the Gaussian Broadening slider. Default is (0, 50).
    """
    if da.ndim != 1:
        raise ValueError(f"Input must be 1-D, but has shape {da.shape}.")

    time_dim = dim or da.dims[0]

    # 1. Zero-filling to the next power of two
    n = len(da)
    n2 = 1 << (n - 1).bit_length()

    if n2 > n:
        pad_len = n2 - n
        vals = np.pad(da.values, (0, pad_len), "constant", constant_values=0)

        t_vals = da.coords[time_dim].values.astype(float)
        dt = t_vals[1] - t_vals[0] if len(t_vals) > 1 else 1.0
        t_pad = t_vals[-1] + np.arange(1, pad_len + 1) * dt
        t_vals = np.concatenate([t_vals, t_pad])

        # Reconstruct the padded array preserving metadata
        da = xr.DataArray(
            vals, coords={time_dim: t_vals}, dims=[time_dim], attrs=da.attrs
        )

    # 2. Extract accurate Spectral Axis via xmris pipeline
    da_spec = da.xmr.to_spectrum(dim=time_dim)

    if unit.lower() == "ppm":
        da_spec = da_spec.xmr.to_ppm()
        x_label = "Chemical Shift [ppm]"
    else:
        # Fallback to Hz if user specifies it
        if (
            "ppm" in str(da_spec.dims[0]).lower()
            or "shift" in str(da_spec.dims[0]).lower()
        ):
            da_spec = da_spec.xmr.to_hz()
        x_label = "Frequency [Hz]"

    spec_dim = da_spec.dims[0]
    x_vals = da_spec.coords[spec_dim].values.astype(float)

    # 3. Extract purely prepared time-domain data for the JS Math Engine
    t_vals = da.coords[time_dim].values.astype(float)
    vals = da.values
    if not np.iscomplexobj(vals):
        vals = vals.astype(complex)

    return ApodizerWidget(
        width=width,
        height=height,
        t_coords=t_vals.tolist(),
        x_coords=x_vals.tolist(),
        x_label=x_label,
        reals_t=np.real(vals).tolist(),
        imags_t=np.imag(vals).tolist(),
        lb_min=lb_range[0],
        lb_max=lb_range[1],
        gb_min=gb_range[0],
        gb_max=gb_range[1],
    )
