import numpy as np
import xarray as xr

from xmris.core.config import ATTRS, DIMS, VARS


def remove_digital_filter(
    da: xr.DataArray, group_delay: float, dim: str = "time", keep_length: bool = True
) -> xr.DataArray:
    """
    Remove the hardware digital filter group delay from Bruker FID data.

    Bruker consoles use a cascade of digital FIR filters during analog-to-digital
    conversion. Because these filters calculate a moving average, they require time
    to "wake up", introducing a causality delay at the start of the Free Induction Decay
    (FID). This manifests as a time-shift, effectively prepending the actual signal with
    a specific number of filter transient points (often appearing as a flatline or
    wavy noise).

    If left uncorrected, this time shift causes a massive, rolling first-order phase error
    in the frequency-domain spectrum.

    This function realigns the signal to t=0 by:
      1. Truncating the integer portion of the delay.
      2. Applying a first-order phase correction to exactly compensate for the remaining
         fractional sub-point delay.

    Parameters
    ----------
    da : xr.DataArray
        Input free induction decay (FID) data in the time domain.
    group_delay : float
        The exact delay value to remove. This should be read directly from the
        Bruker `ACQ_RxFilterInfo` parameter array (index 0: 'groupDelay')].
        Typical values:
          - ~76.0 for standard high-resolution Spectroscopy.
          - ~0.0 to 16.0 for Fast Imaging or ZTE (where hardware pre-compensation
            or short filters are used).
    dim : str, optional
        The time dimension along which to apply the correction, by default "time".
    keep_length : bool, optional
        If True, appends pure zeros to the end of the FID to replace the truncated
        startup points. This ensures the returned DataArray maintains the exact same
        length as the input (critical for FFT radix sizes), avoiding the confusion of
        traditional spectral "zero-filling". By default True.

    Returns
    -------
    xr.DataArray
        The corrected FID data with the filter transient stripped, phase aligned,
        and lineage metadata preserved.
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' missing in DataArray.")

    if group_delay <= 0:
        return da.copy()

    # 1. Separate the delay into integer (points) and fractional (phase) components
    int_delay = int(np.floor(group_delay))
    frac_delay = group_delay - int_delay
    axis_idx = da.get_axis_num(dim)

    # 2. Remove the Integer Delay (Slicing)
    if int_delay > 0:
        cut_data = da.isel({dim: slice(int_delay, None)})
    else:
        cut_data = da

    # 3. Correct the Fractional Delay (Phase Correction)
    if not np.isclose(frac_delay, 0.0):
        n_points = cut_data.sizes[dim]
        freqs = np.fft.fftfreq(n_points)

        # Broadcast frequencies to match N-dimensional data shape
        shape_ones = [1] * cut_data.ndim
        shape_ones[axis_idx] = -1
        freqs_reshaped = freqs.reshape(shape_ones)

        spectrum = np.fft.fft(cut_data.values, axis=axis_idx)

        # Shift signal "left" by multiplying by exp(+j * 2pi * f * dt)
        phase_corrector = np.exp(1j * 2 * np.pi * freqs_reshaped * frac_delay)
        corrected_values = np.fft.ifft(spectrum * phase_corrector, axis=axis_idx)
    else:
        corrected_values = cut_data.values

    # 4. Restore Original Array Length
    if int_delay > 0 and keep_length:
        pad_shape = list(corrected_values.shape)
        pad_shape[axis_idx] = int_delay
        zeros_padding = np.zeros(pad_shape, dtype=corrected_values.dtype)
        final_values = np.concatenate((corrected_values, zeros_padding), axis=axis_idx)
        # Use original DataArray as template to preserve shape sizes
        template_da = da
    else:
        final_values = corrected_values
        # Use the truncated DataArray as template
        template_da = cut_data

    # 5. Rebuild DataArray Safely (Functional Purity)
    da_new = template_da.copy(data=final_values)

    # Ensure Time coordinate starts exactly at 0 after shifting
    time_coords = da_new.coords[dim].values
    da_new = da_new.assign_coords({dim: time_coords - time_coords[0]})

    # 6. Preserve Lineage
    new_attrs = da.attrs.copy()
    new_attrs.update(
        {
            "digital_filter_removed": True,
            "group_delay_removed": group_delay,
            "length_retained_with_zeros": keep_length,
        }
    )

    return da_new.assign_attrs(new_attrs)


import xarray as xr


def _get_val(pv_params: dict, key: str, default=None):
    """Helper to cleanly extract scalar values from Bruker's array-wrapped params."""  # noqa: D401
    val = pv_params.get(key, default)
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        return val[0]
    return val


def reshape_bruker_raw(
    raw_data_1d: np.ndarray, pv_params: dict
) -> tuple[np.ndarray, list[str]]:
    """
    Reshape a flat Bruker rawdata.job0 array into a squeezed N-dimensional numpy array.

    Bruker stores multi-dimensional data sequentially. This function parses the
    method parameters to determine the shape, filters out empty dimensions, and
    reshapes the data to match xmris conventions.

    Expected Bruker Parameters in `pv_params`:
    ------------------------------------------
    * PVM_SpecMatrix    (int):   Number of points in the FID.
    * PVM_EncNReceivers (int):   Number of receive channels (Default: 1).
    * PVM_NAverages     (int):   Number of averages (Default: 1).
    * PVM_NRepetitions  (int):   Number of repetitions (Default: 1).

    Spatial encoding is currently unsupported, we assume:
    --------------------------------------------
    n_slices = 1        # number of slices / slabs
    n_ph1 = 1           # phase encoding direction A
    n_ph2 = 1           # phase encoding direction B


    Parameters
    ----------
    raw_data_1d : np.ndarray
        The flat, 1D complex numpy array loaded directly from the binary file.
    pv_params : dict
        The parsed Bruker parameter dictionary.

    Returns
    -------
    reshaped_data : np.ndarray
        The N-dimensional numpy array.
    valid_dims : list[str]
        A list of dimension names matching the axes of `reshaped_data`.
    """
    # 1. Extract structural sizes
    try:
        n_points = int(_get_val(pv_params, "PVM_SpecMatrix"))
    except TypeError:
        raise ValueError("Missing required structural parameter 'PVM_SpecMatrix'.")

    n_channels = int(_get_val(pv_params, "PVM_EncNReceivers", 1))
    n_averages = int(_get_val(pv_params, "PVM_NAverages", 1))
    n_rep = int(_get_val(pv_params, "PVM_NRepetitions", 1))

    # Spatial dimensions, defaulting to 1 for now. Unlocalized spectroscopy only.
    n_slices = 1
    n_ph1 = 1
    n_ph2 = 1

    # 2. Map standard Bruker order
    dims = [DIMS.time, "channels", "slices", "averages", "ph1", "ph2", "repetitions"]
    sizes = [n_points, n_channels, n_slices, n_averages, n_ph1, n_ph2, n_rep]

    # 3. Filter out empty dimensions (size == 1)
    valid_dims = [d for d, s in zip(dims, sizes) if s > 1]
    valid_sizes = [s for s in sizes if s > 1]

    # 4. Reshape and Transpose
    # Bruker stores Time as the fastest changing dimension.
    # We reshape to reversed sizes to match C-contiguous memory, then transpose.
    try:
        reshaped_data = raw_data_1d.reshape(valid_sizes[::-1]).T
    except ValueError as e:
        raise ValueError(
            f"Cannot reshape raw data of size {raw_data_1d.size} into expected "
            f"valid sizes {valid_sizes}. Check Bruker parameters."
        ) from e

    print(f"Reshaped Bruker data to dims: [ {' | '.join(valid_dims)} ]")

    return reshaped_data, valid_dims


def build_fid(
    data: np.ndarray,
    dims: list[str],
    pv_params: dict,
) -> xr.DataArray:
    """
    Construct a strict xmris FID DataArray from an N-dimensional numpy array.

    Expected Bruker Parameters in `pv_params`:
    ------------------------------------------
    * PVM_SpecSWH        (float): Spectral width in Hz. Used to calculate dwell time.
    * PVM_RepetitionTime (float): TR in ms. Used to calculate the repetitions coordinate.
    * PVM_FrqRef         (float): Reference Larmor frequency in MHz. (Required for to_ppm)
    * PVM_FrqWorkPpm     (float): Carrier chemical shift in ppm. (Required for to_ppm)
    * groupDelay         (float): Bruker specific FID delay. In `ACQ_RxFilterInfo`.

    Parameters
    ----------
    data : np.ndarray
        The squeezed, N-dimensional complex data array.
    dims : list[str]
        The dimension names matching the `data` axes. Must contain DIMS.time.
    pv_params : dict
        The parsed Bruker parameter dictionary.

    Returns
    -------
    xr.DataArray
        A fully compliant DataArray ready for the xmris processing pipeline.
    """
    if data.ndim != len(dims):
        raise ValueError(f"Data ndim ({data.ndim}) must match len(dims) ({len(dims)}).")

    if DIMS.time not in dims:
        raise ValueError(f"Provided dimensions must contain '{DIMS.time}'.")

    def _get_strict(key: str):
        val = _get_val(pv_params, key)
        if val is None:
            raise ValueError(f"Missing required Bruker parameter for physics: '{key}'")
        return float(val)

    # 1. Extract physical parameters
    sw_hz = _get_strict("PVM_SpecSWH")
    tr_ms = _get_strict("PVM_RepetitionTime")
    f0_mhz = _get_strict("PVM_FrqRef")
    carrier_ppm = _get_strict("PVM_FrqWorkPpm")
    groupDelay = _get_strict("groupDelay")

    # 2. Build explicit coordinates
    coords = {}

    # Time Coordinate
    time_idx = dims.index(DIMS.time)
    n_points = data.shape[time_idx]
    dt_s = 1.0 / sw_hz
    coords[DIMS.time] = (
        DIMS.time,
        np.arange(n_points) * dt_s,
        {"units": "s", "long_name": "Time"},
    )

    # Repetition Coordinate (if present)
    if "repetitions" in dims:
        rep_idx = dims.index("repetitions")
        n_rep = data.shape[rep_idx]
        tr_s = tr_ms * 1e-3
        coords["repetitions"] = (
            "repetitions",
            np.arange(n_rep) * tr_s + tr_s,
            {"units": "s", "long_name": "Elapsed Repetition Time"},
        )

    # Simple index coordinates for remaining dimensions
    for d in dims:
        if d not in coords:
            axis_len = data.shape[dims.index(d)]
            coords[d] = (d, np.arange(axis_len))

    # 3. Construct the DataArray
    da = xr.DataArray(data, dims=dims, coords=coords, name=VARS.original_data)

    # 4. Attach ONLY the metadata required by the core processing decorators
    return da.assign_attrs(
        {
            ATTRS.reference_frequency: f0_mhz,
            ATTRS.carrier_ppm: carrier_ppm,
            "bruker_group_delay": groupDelay,
            "units": "a.u.",
        }
    )
