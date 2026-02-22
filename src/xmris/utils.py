import numpy as np
import xarray as xr

from xmris.config import DEFAULTS


def to_real_imag(
    da: xr.DataArray,
    dim: str | None = None,
    coords: tuple[str, str] | None = None,
) -> xr.DataArray:
    """
    Convert a complex DataArray into a real-valued DataArray.

    Stacks the real and imaginary components along a new dimension.

    Parameters
    ----------
    da : xr.DataArray
        The complex-valued input DataArray.
    dim : str, optional
        The name of the new dimension for components. Defaults to
        DEFAULTS.component.dim.
    coords : tuple of str, optional
        The labels for the real and imaginary components. Defaults to
        DEFAULTS.component.coords (e.g., ("real", "imag")).

    Returns
    -------
    xr.DataArray
        Real-valued DataArray with an additional trailing dimension.
    """
    dim = dim or DEFAULTS.component.dim
    coords = coords or DEFAULTS.component.coords

    new_da = xr.DataArray(
        np.stack([da.real.values, da.imag.values], axis=-1),
        dims=tuple(da.dims) + (dim,),
        coords={**da.coords, dim: list(coords)},
        name=da.name,
    )
    return new_da.assign_attrs(da.attrs)


def to_complex(
    da: xr.DataArray,
    dim: str | None = None,
    coords: tuple[str, str] | None = None,
) -> xr.DataArray:
    """
    Merge separated real and imag components back into complex numbers.

    Reconstructs complex values by selecting specific coordinates from
    a component dimension.

    Parameters
    ----------
    da : xr.DataArray
        The real-valued input DataArray containing component parts.
    dim : str, optional
        The dimension name to reduce. Defaults to DEFAULTS.component.dim.
    coords : tuple of str, optional
        The coordinate names corresponding to (real, imag). Defaults to
        DEFAULTS.component.coords.

    Returns
    -------
    xr.DataArray
        Complex-valued DataArray with the component dimension removed.

    Raises
    ------
    ValueError
        If the specified `dim` is not present in the input DataArray.
    """
    dim = dim or DEFAULTS.component.dim
    coords = coords or DEFAULTS.component.coords

    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray.")

    real_part = da.sel({dim: coords[0]})
    imag_part = da.sel({dim: coords[1]})

    new_da = real_part + 1j * imag_part
    new_da.name = da.name

    return new_da.assign_attrs(da.attrs)
