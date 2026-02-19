from typing import Union

import numpy as np
import xarray as xr

# --- 1. Shifting Utilities ---


def fftshift(da: xr.DataArray, dim: str | list[str]) -> xr.DataArray:
    """Apply fftshift by rolling data and coordinates along the given dimension(s)."""
    dims = [dim] if isinstance(dim, str) else dim
    shifts = {d: da.sizes[d] // 2 for d in dims}
    return da.roll(shifts, roll_coords=True)


def ifftshift(da: xr.DataArray, dim: str | list[str]) -> xr.DataArray:
    """Apply ifftshift by rolling data and coordinates along the given dimension(s)."""
    dims = [dim] if isinstance(dim, str) else dim
    shifts = {d: (da.sizes[d] + 1) // 2 for d in dims}
    return da.roll(shifts, roll_coords=True)


# --- 2. Coordinate Math ---


def _convert_fft_coords(
    da: xr.DataArray, dim: str, out_dim: str = None
) -> xr.DataArray:
    """Calculate unshifted reciprocal coordinates and optionally renames dimension."""
    n_points = da.sizes[dim]
    old_coords = da.coords[dim].values

    delta = (old_coords[1] - old_coords[0]) if len(old_coords) > 1 else 1.0

    # Calculate UNSHIFTED frequencies
    new_coords = np.fft.fftfreq(n_points, d=delta)

    da = da.assign_coords({dim: new_coords})
    if out_dim is not None and out_dim != dim:
        da = da.rename({dim: out_dim})
    return da


# --- 3. Pure Transforms ---


def fft(
    da: xr.DataArray,
    dim: str | list[str] = "Time",
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """Perform N-dimensional FFT (Ortho normalized, no shifts)."""
    dims = [dim] if isinstance(dim, str) else dim

    # Handle dimension name mapping
    out_dims = [out_dim] if isinstance(out_dim, str) else out_dim
    if out_dims is not None and len(dims) != len(out_dims):
        raise ValueError("`dim` and `out_dim` lists must have the same length.")
    dim_map = dict(zip(dims, out_dims)) if out_dims else {d: d for d in dims}

    for d in dims:
        if d not in da.dims:
            raise ValueError(f"Dimension '{d}' not found. Available: {list(da.dims)}")

    # 1. Perform standard numpy FFT
    axes = tuple(da.get_axis_num(d) for d in dims)
    arr_fft = np.fft.fftn(da.values, axes=axes, norm="ortho")

    da_transformed = xr.DataArray(
        arr_fft, dims=da.dims, coords=da.coords, attrs=da.attrs, name=da.name
    )

    # 2. Assign unshifted frequency coordinates
    for d in dims:
        da_transformed = _convert_fft_coords(da_transformed, dim=d, out_dim=dim_map[d])

    return da_transformed


def ifft(
    da: xr.DataArray,
    dim: str | list[str] = "Time",
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """Perform N-dimensional IFFT (Ortho normalized, no shifts)."""
    # (Implementation is identical to fft, just using np.fft.ifftn)
    dims = [dim] if isinstance(dim, str) else dim
    out_dims = [out_dim] if isinstance(out_dim, str) else out_dim
    dim_map = dict(zip(dims, out_dims)) if out_dims else {d: d for d in dims}

    axes = tuple(da.get_axis_num(d) for d in dims)
    arr_ifft = np.fft.ifftn(da.values, axes=axes, norm="ortho")

    da_transformed = xr.DataArray(
        arr_ifft, dims=da.dims, coords=da.coords, attrs=da.attrs, name=da.name
    )
    for d in dims:
        da_transformed = _convert_fft_coords(da_transformed, dim=d, out_dim=dim_map[d])
    return da_transformed


# --- 4. Centered Transforms (Convenience Wrappers) ---


def fftc(
    da: xr.DataArray,
    dim: str | list[str] = "Time",
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """Centered N-dimensional FFT (ifftshift -> fft -> fftshift)."""
    # Look how beautiful and explicit this functional chain is!
    new_dims = out_dim if out_dim is not None else dim

    return (
        ifftshift(da, dim=dim)
        .pipe(fft, dim=dim, out_dim=out_dim)
        .pipe(fftshift, dim=new_dims)
    )


def ifftc(
    da: xr.DataArray,
    dim: str | list[str] = "Time",
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """Centered N-dimensional IFFT (ifftshift -> ifft -> fftshift)."""
    new_dims = out_dim if out_dim is not None else dim

    return (
        ifftshift(da, dim=dim)
        .pipe(ifft, dim=dim, out_dim=out_dim)
        .pipe(fftshift, dim=new_dims)
    )
