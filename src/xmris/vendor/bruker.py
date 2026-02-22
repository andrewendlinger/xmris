import numpy as np
import xarray as xr


def remove_digital_filter(
    da: xr.DataArray, group_delay: float, dim: str = "Time", keep_length: bool = True
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
        The time dimension along which to apply the correction, by default "Time".
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
