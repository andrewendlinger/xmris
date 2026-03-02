import pathlib

import anywidget
import numpy as np
import traitlets
import xarray as xr

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
        The 2D matrix of spectra to scroll through (e.g., [repetitions, points]).
    scroll_dim : str
        The name of the dimension being scrolled through (for extraction logic).
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

    _esm = _HERE / "scroller.js"
    _css = _HERE / "scroller.css"

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
    averages, or any specified dimension. The widget includes an extraction
    button to generate the exact `.isel()` snippet needed to isolate a specific
    trace, preserving pipeline lineage.

    Parameters
    ----------
    da : xr.DataArray
        A 2-dimensional DataArray. Must contain one spectral dimension and one
        scrolling dimension.
    scroll_axis : str, optional
        The specific dimension to scroll through. If None, it defaults to
        looking for 'repetitions', then 'averages', or falls back to the
        non-spectral dimension.
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
        If the input `da` is not exactly 2-dimensional, or if the requested
        `part` is invalid.
    """
    if da.ndim != 2:
        raise ValueError(f"Input must be exactly 2-D, but has shape {da.shape}.")

    # 1. Identify spectral vs scroll dimensions
    spec_dim = None
    x_label = "Frequency"

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
        # Fallback: assume the last dimension is the spectral axis
        spec_dim = da.dims[-1]
        x_label = str(spec_dim)

    # 2. Determine the scroll dimension
    if scroll_axis is not None:
        if scroll_axis not in da.dims:
            raise ValueError(
                f"Requested scroll_axis '{scroll_axis}' not found in dimensions: {da.dims}"  # noqa: E501
            )
        scroll_dim = scroll_axis
    else:
        # Auto-detect common scroll dimensions if not explicitly provided
        available_dims = [d for d in da.dims if d != spec_dim]
        scroll_dim = available_dims[0]  # Default to whatever is left
        for candidate in ("repetitions", "averages", "time"):
            if candidate in available_dims:
                scroll_dim = candidate
                break

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

    x_vals = da.coords[spec_dim].values.astype(float)

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
