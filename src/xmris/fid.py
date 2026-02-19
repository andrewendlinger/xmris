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
