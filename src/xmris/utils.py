import numpy as np
import xarray as xr


def to_real_imag(
    da: xr.DataArray, dim: str = "Complex", labels: tuple[str, str] = ("Real", "Imag")
) -> xr.DataArray:
    """Convert a complex-valued xarray DataArray into a real-valued DataArray
    with an extra dimension for the real and imaginary components.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with complex values.
    dim : str, optional
        Name of the new dimension to store the components, by default "Complex".
    labels : tuple[str, str], optional
        The coordinate labels for the new dimension, by default ("Real", "Imag").

    Returns
    -------
    xr.DataArray
        DataArray with an additional dimension, containing real and imaginary components.
    """
    # Use numpy stacking for speed, then reconstruct the xarray
    new_da = xr.DataArray(
        np.stack([da.real.values, da.imag.values], axis=-1),
        dims=tuple(da.dims) + (dim,),
        coords={**da.coords, dim: list(labels)},
        name=da.name,
    )
    # Safely preserve attributes
    return new_da.assign_attrs(da.attrs)


def to_complex(
    da: xr.DataArray, dim: str = "Complex", labels: tuple[str, str] = ("Real", "Imag")
) -> xr.DataArray:
    """Convert an xarray DataArray with separate real and imaginary components
    back into a complex-valued DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray containing the separated components.
    dim : str, optional
        Name of the dimension that stores the components, by default "Complex".
    labels : tuple[str, str], optional
        The coordinate labels corresponding to the real and imaginary parts,
        by default ("Real", "Imag").

    Returns
    -------
    xr.DataArray
        Complex-valued DataArray reconstructed from the real and imaginary parts.
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray.")

    # Using .sel() cleanly extracts the data and naturally drops the selected dimension
    real_part = da.sel({dim: labels[0]})
    imag_part = da.sel({dim: labels[1]})

    new_da = real_part + 1j * imag_part
    new_da.name = da.name

    return new_da.assign_attrs(da.attrs)
