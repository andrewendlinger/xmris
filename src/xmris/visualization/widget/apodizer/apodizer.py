import pathlib

import anywidget
import numpy as np
import traitlets
import xarray as xr

from xmris.core.config import DIMS
from xmris.core.utils import _check_dims, _spectral_axis_label

from .._shared import load_css, load_esm

_HERE = pathlib.Path(__file__).parent


class ApodizerWidget(anywidget.AnyWidget):
    """Interactive widget for NMR/MRS spectra apodization.

    Displays the time-domain FID (top) alongside its live-recomputed spectrum
    (bottom) while the user adjusts exponential or Lorentz-to-Gauss window
    parameters.

    Attributes
    ----------
    width : int
        Pixel width of the rendering canvas.
    height : int
        Pixel height of the (stacked) rendering canvases.
    t_coords : list of float
        Time-domain axis coordinates (seconds) of the FID.
    x_coords : list of float
        The spectral axis coordinates (ppm or Hz) of the transformed spectrum.
    x_label : str
        The label displayed on the spectrum X-axis.
    reals_t : list of float
        Real component of the time-domain FID.
    imags_t : list of float
        Imaginary component of the time-domain FID.
    lb : float
        Current line-broadening (Lorentzian) parameter in Hz.
    gb : float
        Current Gaussian-broadening parameter in Hz (Lorentz-Gauss only).
    lb_min, lb_max : float
        Slider bounds for the line-broadening control.
    gb_min, gb_max : float
        Slider bounds for the Gaussian-broadening control.
    method : str
        Active apodization method, ``"exp"`` or ``"lg"``.
    display_mode : str
        Which spectral component to show: ``"real"``, ``"imag"`` or ``"mag"``.
    show_orig : bool
        If True, overlays the un-apodized data as a reference trace.
    show_grid : bool
        If True, renders background grid lines.
    """

    _esm = load_esm(_HERE / "apodizer.js")
    _css = load_css(_HERE / "apodizer.css")

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


def apodize(
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

    The widget mirrors the :meth:`~xmris.core.accessor.XmrisProcessingMixin.apodize_exp`
    and :meth:`~xmris.core.accessor.XmrisProcessingMixin.apodize_lg` methods: it
    previews line broadening / resolution enhancement in the browser, and its
    Close button emits the matching ``.xmr.apodize_exp(...)`` / ``.xmr.apodize_lg(...)``
    snippet to reproduce the chosen parameters on the real data.

    Parameters
    ----------
    da : xr.DataArray
        A 1-dimensional, time-domain DataArray (an FID). Real-valued input is
        promoted to complex for the display.
    dim : str, optional
        The time dimension to apodize along. If None (default), the canonical
        time dimension (``time``) is used; pass it explicitly for non-standard
        axis names.
    unit : {'ppm', 'hz'}, optional
        The unit for the spectral (bottom) x-axis. Default is 'ppm'.
    width : int, optional
        Width of the widget in pixels. Default is 740.
    height : int, optional
        Height of the widget in pixels. Default is 550.
    lb_range : tuple of float, optional
        The (min, max) range for the Line Broadening slider. Default is (0, 50).
    gb_range : tuple of float, optional
        The (min, max) range for the Gaussian Broadening slider. Default is (0, 50).

    Returns
    -------
    ApodizerWidget
        An interactive widget instance synchronized with the provided data.

    Raises
    ------
    ValueError
        If the input `da` is not 1-dimensional, if `unit` is not one of
        ``{'ppm', 'hz'}``, or if the resolved time dimension is not present
        (pass `dim` explicitly in that case).

    Notes
    -----
    The FID is zero-filled to the next power of two before the browser's
    Radix-2 FFT, which requires power-of-two lengths. This is mathematically
    equivalent to interpolation in the frequency domain and introduces no
    artifacts.
    """
    if da.ndim != 1:
        raise ValueError(f"Input must be 1-D, but has shape {da.shape}.")

    if unit.lower() not in ("ppm", "hz"):
        raise ValueError(f"Unknown unit '{unit}'. Use 'ppm' or 'hz'.")

    # Resolve the time axis: an explicit `dim` wins; otherwise use the canonical
    # time dimension. This mirrors the `DIMS.time` default of the apodize_exp /
    # apodize_lg methods the widget wraps (the input is a time-domain FID, not a
    # spectrum — so the spectral resolver does not apply here).
    if dim is None:
        dim = DIMS.time
    _check_dims(da, dim, "apodize")
    time_dim = dim

    # 1. Zero-filling to the next power of two (required by the browser Radix-2 FFT)
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
        da = xr.DataArray(vals, coords={time_dim: t_vals}, dims=[time_dim], attrs=da.attrs)

    # 2. Extract the spectral axis via the xmris pipeline and derive its label
    #    from the resulting coordinate's lineage metadata (no name-sniffing).
    da_spec = da.xmr.to_spectrum(dim=time_dim)
    if unit.lower() == "ppm":
        da_spec = da_spec.xmr.to_ppm()

    spec_dim = da_spec.dims[0]
    coord = da_spec.coords[spec_dim]
    x_vals = coord.values.astype(float)
    x_label = _spectral_axis_label(str(spec_dim), coord)

    # 3. Extract the prepared time-domain data for the JS math engine
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
