import numpy as np
import xarray as xr

from xmris.config import DEFAULTS


def to_real_imag(
    da: xr.DataArray,
    dim: str | None = None,
    coords: tuple[str, str] | None = None,
) -> xr.DataArray:
    """
    Convert a complex-valued DataArray into a real-valued DataArray
    with an extra dimension for the real and imaginary components.
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
    Convert an xarray DataArray with separate real and imaginary components
    back into a complex-valued DataArray.
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
