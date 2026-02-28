# src/xmris/core/utils.py
import numpy as np
import xarray as xr

from xmris.core.config import XmrisTerm


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


def as_variable(term: XmrisTerm, dims: str | tuple, data: np.ndarray) -> xr.Variable:
    """Wrap a numpy array into an xarray Variable.

    Automatically apply the correct units and long_name from the provided XmrisTerm.
    """
    attrs = {"long_name": term.long_name}
    if term.unit:
        attrs["units"] = term.unit

    return xr.Variable(dims, data, attrs=attrs)
