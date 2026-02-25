import nmrglue as ng
import xarray as xr

from xmris.processing.fid import apodize_exp, to_fid, to_spectrum


def phase(da: xr.DataArray, p0: float = 0.0, p1: float = 0.0) -> xr.DataArray:
    """
    Apply zero- and first-order phase correction to a spectrum.

    Parameters
    ----------
    da : xr.DataArray
        The input frequency-domain spectrum.
    p0 : float, optional
        Zero-order phase angle in degrees, by default 0.0.
    p1 : float, optional
        First-order phase angle in degrees, by default 0.0.

    Returns
    -------
    xr.DataArray
        The phase-corrected spectrum. Phase angles are appended to the attributes.
    """
    # nmrglue's ps expects the data array and angles in degrees
    phased_values = ng.process.proc_base.ps(da.values, p0=p0, p1=p1)

    # Reconstruct the DataArray and embed the exact angles used
    da_phased = da.copy(data=phased_values)

    # Merge existing attributes and add the phase angles
    new_attrs = da.attrs.copy()
    new_attrs.update({"p0": p0, "p1": p1})

    return da_phased.assign_attrs(new_attrs)


def autophase(
    da: xr.DataArray,
    dim: str = "frequency",
    lb: float = 10.0,
    temp_time_dim: str = "time",
) -> xr.DataArray:
    """
    Automatically calculate and apply phase correction to a spectrum.

    This function is optimized for noisy MRIS data. It temporarily converts the
    spectrum back to the time domain, applies heavy exponential apodization to
    artificially boost the Signal-to-Noise Ratio (SNR), and uses an entropy
    minimization algorithm (ACME) on the smoothed data to find the global
    phase minimum. These angles are then applied to the original, untouched spectrum.

    Parameters
    ----------
    da : xr.DataArray
        The input frequency-domain spectrum.
    dim : str, optional
        The frequency dimension, by default "frequency".
    lb : float, optional
        The exponential line broadening factor (in Hz) applied during the
        temporary SNR-boosting step. Higher values suppress more noise.
        By default 10.0.
    temp_time_dim : str, optional
        The name used for the temporary time dimension during the inverse
        transform. By default "time".

    Returns
    -------
    xr.DataArray
        The phased spectrum. Angles are stored in `attrs['p0']` and `attrs['p1']`.
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray.")

    # 1. Transform spectrum back to a temporary FID
    temp_fid = to_fid(da, dim=dim, out_dim=temp_time_dim)

    # 2. Apply heavy sacrificial apodization to crush the noise
    temp_apodized_fid = apodize_exp(temp_fid, dim=temp_time_dim, lb=lb)

    # 3. Transform back to a smooth, high-SNR spectrum for the optimizer
    temp_smooth_spec = to_spectrum(temp_apodized_fid, dim=temp_time_dim, out_dim=dim)

    # 4. Calculate phase angles using nmrglue's ACME algorithm on the smooth data
    _, (p0, p1) = ng.process.proc_autophase.autops(
        temp_smooth_spec.values, "acme", return_phases=True
    )

    # 5. Apply the calculated angles to the ORIGINAL, untouched spectrum
    return phase(da, p0=p0, p1=p1)
