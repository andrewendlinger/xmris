import numpy as np
import xarray as xr

from xmris.core.config import COORDS, DIMS, XmrisTerm
from xmris.core.utils import _check_dims, as_variable

# --- 1. Shifting Utilities ---


def fftshift(da: xr.DataArray, dim: str | list[str]) -> xr.DataArray:
    """
    Apply fftshift by rolling data and coordinates along the given dimension(s).

    This shifts the zero-frequency component to the center of the spectrum.

    Parameters
    ----------
    da : xr.DataArray
        The input xarray DataArray.
    dim : str or list of str
        The dimension(s) along which to apply the shift.

    Returns
    -------
    xr.DataArray
        A new DataArray with the data and coordinates rolled.
    """
    dims = [dim] if isinstance(dim, str) else dim
    _check_dims(da, dims, "fftshift")

    shifts = {d: da.sizes[d] // 2 for d in dims}
    return da.roll(shifts, roll_coords=True)


def ifftshift(da: xr.DataArray, dim: str | list[str]) -> xr.DataArray:
    """
    Apply ifftshift by rolling data and coordinates along the given dimension(s).

    This is the exact inverse of `fftshift`, moving the zero-frequency component
    from the center back to the original position.

    Parameters
    ----------
    da : xr.DataArray
        The input xarray DataArray.
    dim : str or list of str
        The dimension(s) along which to apply the inverse shift.

    Returns
    -------
    xr.DataArray
        A new DataArray with the data and coordinates rolled.
    """
    dims = [dim] if isinstance(dim, str) else dim
    _check_dims(da, dims, "ifftshift")

    shifts = {d: (da.sizes[d] + 1) // 2 for d in dims}
    return da.roll(shifts, roll_coords=True)


# --- 2. Coordinate Math ---


def _convert_fft_coords(
    da: xr.DataArray, dim: str, out_dim: str | None = None, term: XmrisTerm | None = None
) -> xr.DataArray:
    """
    Calculate unshifted reciprocal coordinates and safely rebuild the DataArray.

    Computes the standard discrete Fourier Transform sample frequencies
    (or time periods) and assigns them to the transformed dimension. If an
    `XmrisTerm` is provided, it injects the associated unit and long_name metadata.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray (already transformed in the data domain).
    dim : str
        The original dimension name that was transformed.
    out_dim : str, optional
        The new name for the transformed dimension. If None, the original
        dimension name is kept.
    term : XmrisTerm, optional
        The configuration term (e.g., `COORDS.frequency`) used to inject
        physical metadata (units, long_name) into the new coordinate.

    Returns
    -------
    xr.DataArray
        A DataArray with updated reciprocal coordinates and renamed dimensions.
    """
    n_points = da.sizes[dim]
    old_coords = da.coords[dim].values

    delta = (old_coords[1] - old_coords[0]) if len(old_coords) > 1 else 1.0

    # Calculate UNSHIFTED reciprocal coordinates (frequencies or time periods)
    new_coords = np.fft.fftfreq(n_points, d=delta)

    target_dim = out_dim if out_dim is not None else dim

    # Use the helper to bundle data and metadata if we have a known XmrisTerm
    if term is not None:
        new_var = as_variable(term, target_dim, new_coords)
    else:
        new_var = xr.Variable(target_dim, new_coords)

    if out_dim is not None and out_dim != dim:
        da = da.rename({dim: out_dim})

    return da.assign_coords({target_dim: new_var})


# --- 3. Pure Transforms ---


def fft(
    da: xr.DataArray,
    dim: str | list[str] = DIMS.time,
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """
    Perform an N-dimensional Fast Fourier Transform (FFT).

    Applies an ortho-normalized, unshifted FFT. Metadata and unaffected
    dimensions are strictly preserved.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain DataArray.
    dim : str or list of str, optional
        The dimension(s) to transform. Defaults to `DIMS.time`.
    out_dim : str or list of str, optional
        The resulting dimension name(s). Must match the length of `dim`.
        If None, the original dimension names are retained.

    Returns
    -------
    xr.DataArray
        The frequency-domain DataArray with updated reciprocal coordinates.
    """
    dims = [dim] if isinstance(dim, str) else dim
    _check_dims(da, dims, "fft")

    # Handle dimension name mapping
    out_dims = [out_dim] if isinstance(out_dim, str) else out_dim
    if out_dims is not None and len(dims) != len(out_dims):
        raise ValueError("`dim` and `out_dim` lists must have the same length.")

    # 1. Perform standard numpy FFT
    axes = tuple(da.get_axis_num(d) for d in dims)
    arr_fft = np.fft.fftn(da.values, axes=axes, norm="ortho")

    # Safely rebuild DataArray to preserve untouched lineage/attrs exactly
    da_transformed = da.copy(data=arr_fft)

    # 2. Assign unshifted reciprocal coordinates
    for i, d in enumerate(dims):
        o_dim = out_dims[i] if out_dims else None

        # Smart mapping: If converting standard time, automatically apply
        # frequency metadata
        term = (
            COORDS.frequency
            if (d == DIMS.time and o_dim in (None, DIMS.frequency))
            else None
        )
        da_transformed = _convert_fft_coords(
            da_transformed, dim=d, out_dim=o_dim, term=term
        )

    return da_transformed


def ifft(
    da: xr.DataArray,
    dim: str | list[str] = DIMS.frequency,
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """
    Perform an N-dimensional Inverse Fast Fourier Transform (IFFT).

    Applies an ortho-normalized, unshifted IFFT. Metadata and unaffected
    dimensions are strictly preserved.

    Parameters
    ----------
    da : xr.DataArray
        The input frequency-domain DataArray.
    dim : str or list of str, optional
        The dimension(s) to transform. Defaults to `DIMS.frequency`.
    out_dim : str or list of str, optional
        The resulting dimension name(s). Must match the length of `dim`.
        If None, the original dimension names are retained.

    Returns
    -------
    xr.DataArray
        The time-domain DataArray with updated reciprocal coordinates.
    """
    dims = [dim] if isinstance(dim, str) else dim
    _check_dims(da, dims, "ifft")

    out_dims = [out_dim] if isinstance(out_dim, str) else out_dim
    if out_dims is not None and len(dims) != len(out_dims):
        raise ValueError("`dim` and `out_dim` lists must have the same length.")

    axes = tuple(da.get_axis_num(d) for d in dims)
    arr_ifft = np.fft.ifftn(da.values, axes=axes, norm="ortho")

    da_transformed = da.copy(data=arr_ifft)

    for i, d in enumerate(dims):
        o_dim = out_dims[i] if out_dims else None

        # Smart mapping: If converting standard frequency, automatically apply
        # time metadata
        term = (
            COORDS.time if (d == DIMS.frequency and o_dim in (None, DIMS.time)) else None
        )
        da_transformed = _convert_fft_coords(
            da_transformed, dim=d, out_dim=o_dim, term=term
        )

    return da_transformed


# --- 4. Centered Transforms (Convenience Wrappers) ---


def fftc(
    da: xr.DataArray,
    dim: str | list[str] = DIMS.time,
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """
    Perform a centered N-dimensional FFT.

    This executes an `ifftshift -> fft -> fftshift` pipeline, which is the
    standard in MRI/MRS processing to ensure the 0 Hz frequency remains at
    the center of the spectral axis.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain DataArray.
    dim : str or list of str, optional
        The dimension(s) to transform. Defaults to `DIMS.time`.
    out_dim : str or list of str, optional
        The resulting dimension name(s). If None, keeps the original names.

    Returns
    -------
    xr.DataArray
        The centered, frequency-domain DataArray.
    """
    new_dims = out_dim if out_dim is not None else dim

    return (
        ifftshift(da, dim=dim)
        .pipe(fft, dim=dim, out_dim=out_dim)
        .pipe(fftshift, dim=new_dims)
    )


def ifftc(
    da: xr.DataArray,
    dim: str | list[str] = DIMS.frequency,
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """
    Perform a centered N-dimensional IFFT.

    This executes an `ifftshift -> ifft -> fftshift` pipeline, which correctly
    inverts a centered frequency-domain spectrum back to the time domain.

    Parameters
    ----------
    da : xr.DataArray
        The input centered frequency-domain DataArray.
    dim : str or list of str, optional
        The dimension(s) to transform. Defaults to `DIMS.frequency`.
    out_dim : str or list of str, optional
        The resulting dimension name(s). If None, keeps the original names.

    Returns
    -------
    xr.DataArray
        The centered, time-domain DataArray.
    """
    new_dims = out_dim if out_dim is not None else dim

    return (
        ifftshift(da, dim=dim)
        .pipe(ifft, dim=dim, out_dim=out_dim)
        .pipe(fftshift, dim=new_dims)
    )
