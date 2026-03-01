import nmrglue as ng
import xarray as xr

from xmris.core.config import ATTRS, DIMS
from xmris.core.utils import _check_dims
from xmris.processing.fid import apodize_exp, to_fid, to_spectrum


def phase(
    da: xr.DataArray,
    dim: str = DIMS.frequency,
    p0: float = 0.0,
    p1: float = 0.0,
    pivot: float = None,
) -> xr.DataArray:
    """
    Apply zero- and first-order phase correction to a spectrum.

    Parameters
    ----------
    da : xr.DataArray
        The input frequency-domain spectrum.
    dim : str, optional
        The frequency dimension along which to apply phase correction,
        by default `DIMS.frequency`.
    p0 : float, optional
        Zero-order phase angle in degrees, by default 0.0.
    p1 : float, optional
        First-order phase angle in degrees, by default 0.0.
    pivot : float, optional
        The coordinate value (e.g., ppm or Hz) around which p1 is pivoted.
        If None, standard nmrglue index-0 pivoting is used.

    Returns
    -------
    xr.DataArray
        The phase-corrected spectrum. Phase angles are appended to the attributes.
    """
    _check_dims(da, dim, "phase")

    # If a pivot is provided, translate human UI angles to nmrglue array angles
    if pivot is not None:
        coords = da.coords[dim].values
        x_range = coords[-1] - coords[0]

        if x_range != 0:
            # Shift p0 to match what nmrglue expects at index 0
            p0_ng = p0 + p1 * ((coords[0] - pivot) / x_range)
            # nmrglue expects p1 to be the total phase twist across the array
            p1_ng = p1
        else:
            p0_ng, p1_ng = p0, p1
    else:
        p0_ng, p1_ng = p0, p1

    da_transposed = da.transpose(..., dim)

    # Pass the translated angles to nmrglue
    phased_values = ng.process.proc_base.ps(da_transposed.values, p0=p0_ng, p1=p1_ng)

    da_phased = da_transposed.copy(data=phased_values)
    da_phased = da_phased.transpose(*da.dims)

    da_phased = da_phased.assign_attrs(da.attrs)
    # Save the exact numbers the user input for reproducibility
    da_phased.attrs[ATTRS.phase_p0] = p0
    da_phased.attrs[ATTRS.phase_p1] = p1
    if pivot is not None:
        da_phased.attrs["phase_pivot"] = pivot

    return da_phased


def autophase(
    da: xr.DataArray,
    dim: str = DIMS.frequency,
    lb: float = 10.0,
    temp_time_dim: str = DIMS.time,
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
        The frequency dimension, by default `DIMS.frequency`.
    lb : float, optional
        The exponential line broadening factor (in Hz) applied during the
        temporary SNR-boosting step. Higher values suppress more noise.
        By default 10.0.
    temp_time_dim : str, optional
        The name used for the temporary time dimension during the inverse
        transform. By default `DIMS.time`.

    Returns
    -------
    xr.DataArray
        The phased spectrum. Applied angles are stored in `attrs[ATTRS.phase_p0]`
        and `attrs[ATTRS.phase_p1]`.
    """
    _check_dims(da, dim, "autophase")

    # 1. Transform spectrum back to a temporary FID
    temp_fid = to_fid(da, dim=dim, out_dim=temp_time_dim)

    # 2. Apply heavy sacrificial apodization to crush the noise
    temp_apodized_fid = apodize_exp(temp_fid, dim=temp_time_dim, lb=lb)

    # 3. Transform back to a smooth, high-SNR spectrum for the optimizer
    temp_smooth_spec = to_spectrum(temp_apodized_fid, dim=temp_time_dim, out_dim=dim)

    # 4. Calculate phase angles using nmrglue's ACME algorithm on the smooth data
    # Ensure the target dimension is last for nmrglue's underlying routines
    temp_smooth_transposed = temp_smooth_spec.transpose(..., dim)
    _, (p0, p1) = ng.process.proc_autophase.autops(
        temp_smooth_transposed.values, "acme", return_phases=True
    )

    # 5. Apply the calculated angles to the ORIGINAL, untouched spectrum
    # The phase() function will automatically handle the lineage stamping for p0 and p1
    return phase(da, dim=dim, p0=p0, p1=p1)
