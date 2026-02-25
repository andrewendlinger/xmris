import numpy as np
import xarray as xr

from xmris.core.accessor import _check_dims
from xmris.core.config import DIMS


def to_real_imag(
    da: xr.DataArray,
    dim: str = DIMS.component,
    coords: tuple[str, str] = ("real", "imag"),
) -> xr.DataArray:
    """
    Convert a complex DataArray into a real-valued DataArray.

    Stacks the real and imaginary components along a new dimension. This is
    particularly useful when passing data to machine learning models or
    exporters that do not support complex numbers natively.

    Parameters
    ----------
    da : xr.DataArray
        The complex-valued input DataArray.
    dim : str, optional
        The name of the new dimension for components. Defaults to DIMS.component.
    coords : tuple of str, optional
        The labels for the real and imaginary coordinates. Defaults to ("real", "imag").

    Returns
    -------
    xr.DataArray
        Real-valued DataArray with an additional trailing dimension.
    """
    new_da = xr.DataArray(
        np.stack([da.real.values, da.imag.values], axis=-1),
        dims=tuple(da.dims) + (dim,),
        coords={**da.coords, dim: list(coords)},
        name=da.name,
    )
    return new_da.assign_attrs(da.attrs)


def to_complex(
    da: xr.DataArray,
    dim: str = DIMS.component,
    coords: tuple[str, str] = ("real", "imag"),
) -> xr.DataArray:
    """
    Merge separated real and imag components back into complex numbers.

    Reconstructs complex values by selecting specific coordinates from
    the component dimension.

    Parameters
    ----------
    da : xr.DataArray
        The real-valued input DataArray containing component parts.
    dim : str, optional
        The dimension name to reduce. Defaults to DIMS.component.
    coords : tuple of str, optional
        The coordinate names corresponding to (real, imag). Defaults to ("real", "imag").

    Returns
    -------
    xr.DataArray
        Complex-valued DataArray with the component dimension removed.
    """
    # Use our UX-friendly bouncer to ensure the dimension exists!
    _check_dims(da, dim, "to_complex")

    # Select the specified coordinates to rebuild the complex array
    real_part = da.sel({dim: coords[0]})
    imag_part = da.sel({dim: coords[1]})

    new_da = real_part + 1j * imag_part
    new_da.name = da.name

    return new_da.assign_attrs(da.attrs)
