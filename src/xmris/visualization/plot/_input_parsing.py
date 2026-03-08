import xarray as xr

from xmris.core import DIMS


def parse_input_dims_timeseries(
    da: xr.DataArray,
    user_x_dim: str | None = None,
    user_stack_dim: str | None = None,
) -> tuple[str, str]:
    """Resolve and validate the user input dimension for a 1D + time series plot.

    This function attempts to automatically identify the most logical dimensions for
    a 2D (1D spectra + time) contour or ridge plot if they are not explicitly provided.
    It prefers 'chemical_shift' or 'frequency' for the x-axis and 'averages' or
    'repetitions' for the stacked axis .

    Parameters
    ----------
    da : xr.DataArray
        The N-dimensional DataArray being plotted.
    user_x_dim : str, optional
        Explicitly requested x-axis dimension. If None, the function will attempt
        to auto-resolve it.
    user_stack_dim : str, optional
        Explicitly requested stacking dimension. If None, the function will attempt
        to auto-resolve it.

    Returns
    -------
    tuple[str, str]
        A tuple containing the validated `(x_dim, stack_dim)`.

    Raises
    ------
    ValueError
        If the required dimensions cannot be found or unambiguously resolved.
    """
    dims = list(da.dims)

    # 1. Resolve X-Axis
    if user_x_dim:
        if user_x_dim not in dims:
            raise ValueError(
                f"Requested x-axis dimension '{user_x_dim}' not found in DataArray."
            )
        x_dim = user_x_dim
    else:
        # Auto-detect common spectral axes
        if DIMS.chemical_shift in dims:
            x_dim = DIMS.chemical_shift
        elif DIMS.frequency in dims:
            x_dim = DIMS.frequency
        else:
            raise ValueError(
                "Could not automatically resolve x-axis dimension. DataArray must contain "
                "'chemical_shift' or 'frequency', or `x_dim` must be explicitly provided."
            )

    # Remove the resolved x-axis from the pool of available dimensions
    remaining_dims = [d for d in dims if d != x_dim]

    # 2. Resolve Stacking Axis
    if user_stack_dim:
        if user_stack_dim not in dims:
            raise ValueError(
                f"Requested stacking dimension '{user_stack_dim}' not found in DataArray."
            )
        stack_dim = user_stack_dim
    else:
        if not remaining_dims:
            raise ValueError(
                f"DataArray only has one dimension ('{x_dim}'). Ridge/Contour plots require "
                "at least two dimensions."
            )
        elif len(remaining_dims) == 1:
            stack_dim = remaining_dims[0]
        else:
            # Try to find a logical secondary dimension
            if DIMS.averages in remaining_dims:
                stack_dim = DIMS.averages
            elif DIMS.repetitions in remaining_dims:
                stack_dim = DIMS.repetitions
            else:
                # Fallback to the first available non-x dimension
                stack_dim = remaining_dims[0]

    return str(x_dim), str(stack_dim)
