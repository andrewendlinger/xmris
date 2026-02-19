import numpy as np
import xarray as xr

from xmris.fourier import fft, fftshift, ifft, ifftshift


def to_spectrum(
    da: xr.DataArray, dim: str = "Time", out_dim: str = "Frequency"
) -> xr.DataArray:
    """
    Convert a time-domain Free Induction Decay (FID) to a frequency-domain spectrum.

    The stored, digital FID signal can be processed by a discrete Fourier
    [cite_start]transformation (DFT) to produce a digital MR spectrum[cite: 17, 18].
    This function applies the FFT along the specified time dimension and shifts
    the zero-frequency component to the center of the spectrum.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain FID data.
    dim : str, optional
        The time dimension to transform, by default "Time".
    out_dim : str, optional
        The name of the resulting frequency dimension, by default "Frequency".

    Returns
    -------
    xr.DataArray
        The frequency-domain spectrum with centered zero-frequency coordinates.
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray.")

    # 1. Standard FFT (handles transform and creates unshifted frequency coords)
    da_freq = fft(da, dim=dim, out_dim=out_dim)

    # 2. Shift the frequency domain to center the DC component
    return fftshift(da_freq, dim=out_dim)


def to_fid(
    da: xr.DataArray, dim: str = "Frequency", out_dim: str = "Time"
) -> xr.DataArray:
    """Convert a frequency-domain spectrum back to a time-domain FID."""
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray.")

    # 1. Inverse shift the frequency domain to put the DC component at index 0
    # This prepares the data for the standard IFFT algorithm
    da_unshifted = ifftshift(da, dim=dim)

    # 2. Apply IFFT
    # The output is naturally ordered [t=0, t=1, ... t=N-1]
    da_fid = ifft(da_unshifted, dim=dim, out_dim=out_dim)

    # 3. Reconstruct the strictly positive time coordinates [0, T_acq]
    if dim in da.coords:
        freqs = da.coords[dim].values
        n_points = len(freqs)
        if n_points > 1:
            # Calculate dwell time (dt) based on the sampling theorem:
            # Spectral Width (SW) = 1/dt
            # SW = n_points * df
            df = abs(freqs[1] - freqs[0])
            dt = 1.0 / (n_points * df)

            t_coords = np.arange(n_points) * dt
            da_fid = da_fid.assign_coords({out_dim: t_coords})

    return da_fid


def apodize_exp(da: xr.DataArray, dim: str = "Time", lb: float = 1.0) -> xr.DataArray:
    """
    Apply an exponential weighting filter function for line broadening.

    During apodization, the time-domain FID signal $f(t)$ is multiplied with a filter
    function $f_{filter}(t) = e^{-t/T_L}$. This improves the
    Signal-to-Noise Ratio (SNR) because data points at the end of the FID, which
    primarily contain noise, are attenuated. The time constant $T_L$ is calculated
    from the desired line broadening in Hz.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain data.
    dim : str, optional
        The dimension corresponding to time, by default "Time".
    lb : float, optional
        The desired line broadening factor in Hz, by default 1.0.

    Returns
    -------
    xr.DataArray
        A new apodized DataArray, preserving coordinates and attributes.
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray.")

    t = da.coords[dim]

    # Calculate exponential filter: exp(-t / T_L) where T_L = 1 / (pi * lb)
    # This simplifies to: exp(-pi * lb * t)
    weight = np.exp(-np.pi * lb * t)

    # Functional application
    da_apodized = da * weight
    return da_apodized.assign_attrs(da.attrs)


def apodize_lg(
    da: xr.DataArray, dim: str = "Time", lb: float = 1.0, gb: float = 1.0
) -> xr.DataArray:
    """
    Apply a Lorentzian-to-Gaussian transformation filter.

    This filter converts a Lorentzian line shape to a Gaussian line shape, which decays
    to the baseline in a narrower frequency range. The time-domain FID
    is multiplied by $e^{+t/T_L}e^{-t^2/T_G^2}$. The time constants $T_L$ and $T_G$
    are derived from the `lb` and `gb` frequency-domain parameters.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain data.
    dim : str, optional
        The dimension corresponding to time, by default "Time".
    lb : float, optional
        The Lorentzian line broadening to cancel in Hz, by default 1.0.
    gb : float, optional
        The Gaussian line broadening to apply in Hz, by default 1.0.

    Returns
    -------
    xr.DataArray
        A new apodized DataArray, preserving coordinates and attributes.
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray.")

    t = da.coords[dim]

    # Calculate Lorentzian cancellation: exp(+t / T_L)
    # T_L = 1 / (pi * lb)
    weight_lorentzian = np.exp(np.pi * lb * t)

    # Calculate Gaussian broadening: exp(-t^2 / T_G^2)
    # T_G = 2 * sqrt(ln(2)) / (pi * gb)
    if gb != 0:
        t_g = (2 * np.sqrt(np.log(2))) / (np.pi * gb)
        weight_gaussian = np.exp(-(t**2) / (t_g**2))
    else:
        weight_gaussian = 1.0

    da_apodized = da * (weight_lorentzian * weight_gaussian)
    return da_apodized.assign_attrs(da.attrs)


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
