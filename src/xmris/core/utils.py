# src/xmris/core/utils.py
import numpy as np
import xarray as xr

from xmris.core.config import SPECTRAL_DIMS, XmrisTerm


def _check_dims(da: xr.DataArray, dims: str | list[str], method_name: str) -> None:
    """Validate that required dimensions exist in the DataArray."""
    dims_to_check = [dims] if isinstance(dims, str) else dims
    missing = [d for d in dims_to_check if d not in da.dims]

    if missing:
        raise ValueError(
            f"Method '{method_name}' attempted to operate on missing "
            f"dimension(s): {missing}.\n"
            f"Available dimensions are: {list(da.dims)}.\n\n"
            f"To fix this, either pass the correct `dim` string argument to the function,"
            f" or rename your data's axes using xarray:\n"
            f"    >>> obj = obj.rename({{{repr(missing[0])}: 'correct_name'}})"
        )


def _resolve_spectral_dim(da: xr.DataArray) -> str:
    """Identify the single spectral dimension present on ``da``.

    Scans ``da.dims`` for a dimension belonging to the spectral domain
    (``SPECTRAL_DIMS`` — ``frequency`` [Hz] or ``chemical_shift`` [ppm]). Used
    by the ``@resolves_spectral_dim`` decorator to fill a ``dim=None`` argument.

    Parameters
    ----------
    da : xr.DataArray
        The data to inspect.

    Returns
    -------
    str
        The name of the unique spectral dimension found.

    Raises
    ------
    ValueError
        If no spectral dimension is present, or if more than one is present
        (ambiguous — the caller must pass ``dim`` explicitly).
    """
    found = [d for d in da.dims if d in SPECTRAL_DIMS]

    if not found:
        raise ValueError(
            f"Could not resolve a spectral dimension: none of "
            f"{sorted(SPECTRAL_DIMS)} are present in {list(da.dims)}.\n\n"
            f"If this is time-domain data, transform it first:\n"
            f"    >>> obj = obj.xmr.to_spectrum()\n"
            f"Otherwise pass the dimension explicitly, e.g. `dim='frequency'`."
        )

    if len(found) > 1:
        raise ValueError(
            f"Ambiguous spectral dimension: multiple spectral dims {found} are "
            f"present in {list(da.dims)}.\n\n"
            f"Pass the target dimension explicitly, e.g. `dim={found[0]!r}`."
        )

    return str(found[0])


def _spectral_axis_label(dim: str, coord: xr.DataArray) -> str:
    """Build an axis label from a coordinate's lineage metadata.

    Prefers the coordinate's ``long_name``/``units`` attrs (set by xmris when it
    builds spectral coordinates) over any hardcoded string, falling back to the
    bare dimension name when that metadata is absent. Shared by the visualization
    widgets so no widget hardcodes an axis label.

    Parameters
    ----------
    dim : str
        Name of the spectral dimension (used as the fallback label).
    coord : xr.DataArray
        The coordinate carrying the axis values and its ``.attrs`` metadata.

    Returns
    -------
    str
        A display label such as ``"Chemical Shift [ppm]"``.
    """
    long_name = coord.attrs.get("long_name")
    units = coord.attrs.get("units")
    if long_name and units:
        return f"{long_name} [{units}]"
    if long_name:
        return str(long_name)
    return str(dim)


def as_variable(term: XmrisTerm, dims: str | tuple, data: np.ndarray) -> xr.Variable:
    """Wrap a numpy array into an xarray Variable.

    Automatically apply the correct units and long_name from the provided XmrisTerm.
    """
    attrs = {"long_name": term.long_name}
    if term.unit:
        attrs["units"] = term.unit

    return xr.Variable(dims, data, attrs=attrs)
