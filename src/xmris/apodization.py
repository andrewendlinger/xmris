import numpy as np
import xarray as xr


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
