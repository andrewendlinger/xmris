import pathlib

import anywidget
import numpy as np
import traitlets
import xarray as xr

from xmris.core.utils import _check_dims, _resolve_spectral_dim, _spectral_axis_label

from .._shared import load_css, load_esm

_HERE = pathlib.Path(__file__).parent


class ScrollWidget(anywidget.AnyWidget):
    """Interactive widget for browsing a series of NMR spectra.

    Attributes
    ----------
    width : int
        Pixel width of the rendering canvas.
    height : int
        Pixel height of the rendering canvas.
    x_coords : list of float
        The spectral axis coordinates (e.g., ppm or Hz).
    x_label : str
        The label displayed on the X-axis.
    spectra : list of list of float
        The 2D matrix of spectra to scroll through (e.g., [scroll, points]).
    scroll_dim : str
        The name of the dimension being scrolled through.
    current_index : int
        The currently displayed index of the `scroll_dim`.
    show_trace : bool
        If True, displays fading historical traces behind the current spectrum.
    trace_count : int
        Number of historical traces to display when `show_trace` is True.
    xlim : list of float
        Optional static bounds for the X-axis.
    ylim : list of float
        Optional static bounds for the Y-axis.
    """

    _esm = load_esm(_HERE / "scroller.js")
    _css = load_css(_HERE / "scroller.css")

    width = traitlets.Int(740).tag(sync=True)
    height = traitlets.Int(400).tag(sync=True)
    x_coords = traitlets.List().tag(sync=True)
    x_label = traitlets.Unicode("Chemical Shift [ppm]").tag(sync=True)
    spectra = traitlets.List().tag(sync=True)
    scroll_dim = traitlets.Unicode("").tag(sync=True)
    current_index = traitlets.Int(0).tag(sync=True)
    show_trace = traitlets.Bool(True).tag(sync=True)
    trace_count = traitlets.Int(10).tag(sync=True)
    xlim = traitlets.List(default_value=[]).tag(sync=True)
    ylim = traitlets.List(default_value=[]).tag(sync=True)


def scroll_spectra(
    da: xr.DataArray,
    scroll_axis: str | None = None,
    dim: str | None = None,
    part: str = "real",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_trace: bool = True,
    trace_count: int = 10,
    width: int = 740,
    height: int = 400,
) -> ScrollWidget:
    """
    Instantiate an interactive viewer for a 2-D xarray of spectra.

    This generates a UI allowing the user to scroll through repetitions,
    averages, or any specified dimension. The widget includes an "Extract Slice"
    button that closes the widget and emits the exact ``.isel({dim: idx})``
    snippet needed to isolate a specific trace, preserving pipeline lineage.

    Parameters
    ----------
    da : xr.DataArray
        A 2-dimensional DataArray. Must contain one spectral dimension and one
        scrolling dimension.
    scroll_axis : str, optional
        The dimension to scroll through. If None (default), it is derived as the
        non-spectral dimension of the 2-D array.
    dim : str, optional
        The spectral (display) dimension. If None (default), the canonical
        spectral dimension (``frequency`` or ``chemical_shift``) is resolved
        automatically; pass it explicitly for non-standard axis names.
    part : {'real', 'imag', 'abs'}, optional
        Which component of the complex data to display. Defaults to 'real'.
    xlim : tuple of float, optional
        Static (min, max) bounds for the spectral axis.
    ylim : tuple of float, optional
        Static (min, max) bounds for intensity. If None, auto-ranges globally.
    show_trace : bool, optional
        Show fading historical traces behind the current scan. Defaults to True.
    trace_count : int, optional
        The number of historical traces to overlay. Defaults to 10.
    width : int, optional
        Width of the widget in pixels. The default is 740.
    height : int, optional
        Height of the widget in pixels. The default is 400.

    Returns
    -------
    ScrollWidget
        An interactive widget instance synchronized with the provided data.

    Raises
    ------
    ValueError
        If the input `da` is not exactly 2-dimensional, if no spectral dimension
        can be resolved (pass `dim` explicitly in that case), if `scroll_axis`
        coincides with the spectral dimension, or if the requested `part` is
        invalid.
    """
    if da.ndim != 2:
        raise ValueError(f"Input must be exactly 2-D, but has shape {da.shape}.")

    # 1. Resolve the spectral (display) axis from the vocabulary — an explicit
    #    `dim` wins; otherwise auto-detect the canonical spectral dimension.
    if dim is None:
        spec_dim = _resolve_spectral_dim(da)
    else:
        spec_dim = dim
    _check_dims(da, spec_dim, "scroll_spectra")

    # 2. Derive the scroll axis as the other (non-spectral) dimension, unless the
    #    caller pins it explicitly.
    others = [str(d) for d in da.dims if d != spec_dim]
    scroll_dim = scroll_axis if scroll_axis is not None else others[0]
    _check_dims(da, scroll_dim, "scroll_spectra")
    if scroll_dim == spec_dim:
        raise ValueError(
            f"scroll_axis '{scroll_dim}' must differ from the spectral dimension '{spec_dim}'."
        )

    # 3. Extract the targeted mathematical component
    vals = da.values
    if np.iscomplexobj(vals):
        part = part.lower()
        if part in ("real", "re"):
            vals = np.real(vals)
        elif part in ("imag", "im"):
            vals = np.imag(vals)
        elif part in ("abs", "mag", "magnitude"):
            vals = np.abs(vals)
        else:
            raise ValueError(f"Unknown part '{part}'. Use 'real', 'imag', or 'abs'.")

    vals = vals.astype(float)

    # 4. Transpose if necessary so shape is (Scroll, Spectral)
    if da.dims.index(scroll_dim) > da.dims.index(spec_dim):
        vals = vals.T

    # 5. Build the display axis from the coordinate and its lineage metadata.
    if spec_dim in da.coords:
        coord = da.coords[spec_dim]
        x_vals = coord.values.astype(float)
        x_label = _spectral_axis_label(str(spec_dim), coord)
    else:
        x_vals = np.arange(da.sizes[spec_dim], dtype=float)
        x_label = str(spec_dim)

    return ScrollWidget(
        width=width,
        height=height,
        x_coords=x_vals.tolist(),
        x_label=x_label,
        spectra=vals.tolist(),
        scroll_dim=str(scroll_dim),
        current_index=0,
        show_trace=show_trace,
        trace_count=int(trace_count),
        xlim=list(xlim) if xlim is not None else [],
        ylim=list(ylim) if ylim is not None else [],
    )
