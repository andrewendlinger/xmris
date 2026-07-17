import pathlib

import anywidget
import numpy as np
import traitlets
import xarray as xr

from xmris.core.config import SPECTRAL_DIMS
from xmris.core.utils import _check_dims, _resolve_dim, _spectral_axis_label

from .._shared import load_css, load_esm

_HERE = pathlib.Path(__file__).parent


class PhaseWidget(anywidget.AnyWidget):
    """Interactive widget for manual NMR spectra phase correction.

    Provides a graphical interface for adjusting zero-order (p0) and
    first-order (p1) phase terms.

    Attributes
    ----------
    width : int
        Pixel width of the rendering canvas.
    height : int
        Pixel height of the rendering canvas.
    show_grid : bool
        If True, renders background grid lines.
    show_pivot : bool
        If True, displays a dashed vertical line at the `pivot_val` location.
    x_coords : list of float
        The spectral axis coordinates (e.g., ppm or Hz).
    x_label : str
        The label displayed on the X-axis.
    reals : list of float
        Real component of the complex spectrum.
    imags : list of float
        Imaginary component of the complex spectrum.
    mag : list of float
        Magnitude ($|S|$) used for initial auto-scaling of the view.
    p0 : float
        Current zero-order phase correction in degrees.
    p1 : float
        Current first-order phase correction in degrees.
    pivot_val : float
        The frequency/coordinate where the $p_1$ phase shift is zero.
    """

    _esm = load_esm(_HERE / "phase.js")
    _css = load_css(_HERE / "phase.css")

    width = traitlets.Int(740).tag(sync=True)
    height = traitlets.Int(400).tag(sync=True)
    show_grid = traitlets.Bool(True).tag(sync=True)
    show_pivot = traitlets.Bool(True).tag(sync=True)
    x_coords = traitlets.List().tag(sync=True)
    x_label = traitlets.Unicode("Chemical Shift [ppm]").tag(sync=True)
    reals = traitlets.List().tag(sync=True)
    imags = traitlets.List().tag(sync=True)
    mag = traitlets.List().tag(sync=True)
    p0 = traitlets.Float(0.0).tag(sync=True)
    p1 = traitlets.Float(0.0).tag(sync=True)
    pivot_val = traitlets.Float(0.0).tag(sync=True)


def phase_spectrum(
    da: xr.DataArray,
    dim: str | None = None,
    width: int = 740,
    height: int = 400,
    show_grid: bool = True,
    show_pivot: bool = True,
) -> PhaseWidget:
    """
    Instantiate an interactive phase correction viewer for a 1-D complex xarray.

    The spectral axis is resolved from the project vocabulary and the pivot is
    placed at the maximum signal intensity for physically sensible first-order
    phasing.

    Parameters
    ----------
    da : xr.DataArray
        A 1-dimensional, complex-valued DataArray carrying a spectral coordinate.
    dim : str, optional
        Spectral dimension to plot along. If None (default), the canonical
        spectral dimension (``frequency`` or ``chemical_shift``) is resolved
        automatically; pass it explicitly for non-standard axis names.
    width : int, optional
        Width of the widget in pixels. The default is 740.
    height : int, optional
        Height of the widget in pixels. The default is 400.
    show_grid : bool, optional
        Toggle the background grid visibility. The default is True.
    show_pivot : bool, optional
        Toggle the visibility of the $p_1$ pivot indicator. The default is True.

    Returns
    -------
    PhaseWidget
        An interactive widget instance synchronized with the provided data.

    Raises
    ------
    ValueError
        If the input `da` is not 1-dimensional, contains non-complex data, or no
        spectral dimension can be resolved (pass `dim` explicitly in that case).

    Notes
    -----
    The pivot point is crucial for first-order phasing ($p_1$). This function
    sets the pivot to the maximum of the magnitude spectrum to simplify
    local phase adjustments.
    """
    if da.ndim != 1:
        raise ValueError(f"Input must be 1-D, but has shape {da.shape}.")

    if not np.iscomplexobj(da.values):
        raise ValueError("Phasing requires complex-valued data (Real + Imaginary).")

    # Resolve the spectral axis: an explicit `dim` wins; otherwise auto-detect the
    # canonical spectral dimension (frequency / chemical_shift) via the shared resolver.
    if dim is None:
        dim = _resolve_dim(da, SPECTRAL_DIMS)
    _check_dims(da, dim, "phase_spectrum")

    # Build the display axis from the coordinate and its lineage metadata (no name sniffing).
    if dim in da.coords:
        coord = da.coords[dim]
        x_vals = coord.values.astype(float)
        x_label = _spectral_axis_label(dim, coord)
    else:
        x_vals = np.arange(da.sizes[dim], dtype=float)
        x_label = str(dim)

    vals = da.values
    mag_vals = np.abs(vals).astype(float)

    # Heuristic: Pivot at the highest peak
    pivot = float(x_vals[np.argmax(mag_vals)])

    return PhaseWidget(
        width=width,
        height=height,
        show_grid=show_grid,
        show_pivot=show_pivot,
        x_coords=x_vals.tolist(),
        x_label=x_label,
        reals=np.real(vals).astype(float).tolist(),
        imags=np.imag(vals).astype(float).tolist(),
        mag=mag_vals.tolist(),
        pivot_val=pivot,
    )
