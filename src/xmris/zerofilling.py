import numpy as np
import xarray as xr


def zero_fill(
    da: xr.DataArray,
    dim: str = "Time",
    target_points: int = 1024,
    position: str = "end",
) -> xr.DataArray:
    """
    Pad the specified dimension with zero amplitude points.

    Artificially extend the data with zeros and increase digital resolution.

    Parameters
    ----------
    da : xr.DataArray
        The input data.
    dim : str, optional
        The dimension along which to pad zeros, by default "Time".
    target_points : int, optional
        The total number of points desired after padding, by default 1024.
    position : {"end", "symmetric"}, optional
        Where to apply the zeros. Use "end" for time-domain FIDs, and
        "symmetric" for spatial frequency domains like k-space. By default "end".

    Returns
    -------
    xr.DataArray
        A new DataArray padded with zeros to the target length, preserving metadata.
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray.")

    current_points = da.sizes[dim]
    if target_points <= current_points:
        return da.copy()

    pad_size = target_points - current_points

    # Determine padding distribution based on position
    if position == "end":
        pad_width = (0, pad_size)
    elif position == "symmetric":
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        pad_width = (pad_left, pad_right)
    else:
        raise ValueError("`position` must be either 'end' or 'symmetric'.")

    # Pad with constant 0s
    da_padded = da.pad({dim: pad_width}, mode="constant", constant_values=0)

    # Extrapolate coordinates linearly if they exist
    if dim in da.coords:
        old_coords = da.coords[dim].values
        if len(old_coords) > 1:
            delta = old_coords[1] - old_coords[0]

            if position == "end":
                new_coords = old_coords[0] + np.arange(target_points) * delta
            else:  # symmetric
                start_coord = old_coords[0] - (pad_width[0] * delta)
                new_coords = start_coord + np.arange(target_points) * delta

            da_padded = da_padded.assign_coords({dim: new_coords})

    return da_padded.assign_attrs(da.attrs)
