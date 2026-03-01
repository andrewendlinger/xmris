import numpy as np
import scipy.optimize
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
    Apply zero- and first-order (linear) phase correction to a spectrum.

    Parameters
    ----------
    da : xr.DataArray
        The input frequency-domain spectrum. Must be complex-valued.
    dim : str, optional
        The coordinate dimension along which to apply phase correction,
        by default `DIMS.frequency`.
    p0 : float, optional
        Zero-order phase angle in degrees. This is a constant phase shift
        applied uniformly to all coordinates. By default 0.0.
    p1 : float, optional
        First-order phase angle in degrees. This represents the total phase
        twist applied across the entire spectral range (`max_coord - min_coord`).
        By default 0.0.
    pivot : float, optional
        The coordinate value (e.g., ppm or Hz) around which `p1` is anchored.
        At this exact coordinate, the first-order phase contribution is 0.0.
        If None, standard maximum-magnitude pivoting is used.

    Returns
    -------
    xr.DataArray
        The phase-corrected spectrum. Phase parameters (p0, p1, pivot, and
        pivot_coord) are appended to the dataset attributes to preserve lineage.
    """
    _check_dims(da, dim, "phase")

    # If pivot isn't explicitly provided, default to the max magnitude
    # to perfectly match the initial behavior of the JS widget
    if pivot is None:
        pivot = float(da.coords[dim].values[np.argmax(np.abs(da.values))])

    # Extract coordinates and determine the absolute range (matching JS: max - min)
    coords = da.coords[dim]
    x_min = float(coords.min())
    x_max = float(coords.max())
    x_range = x_max - x_min

    # Convert degrees to radians
    p0_rad = np.radians(p0)
    p1_rad = np.radians(p1)

    # Calculate the phase array using physical coordinates (matches JS 1:1)
    if x_range == 0:
        phase_array = p0_rad
    else:
        phase_array = p0_rad + p1_rad * ((coords - pivot) / x_range)

    # Apply the complex phase shift: data * exp(i * phase)
    # xarray automatically broadcasts the coordinate math across the correct dimension
    da_phased = da * np.exp(1.0j * phase_array)

    # 1. Explicitly copy original attributes to the new object
    da_phased.attrs = da.attrs.copy()

    # 2. Safety Check: Warn if applying phase in a new coordinate space
    if pivot is not None and ATTRS.phase_pivot_coord in da_phased.attrs:
        old_coord = da_phased.attrs[ATTRS.phase_pivot_coord]
        if old_coord != dim:
            import warnings

            warnings.warn(
                f"Applying phase in '{dim}', but previous phase operations "
                f"were recorded in '{old_coord}'. Ensure your pivot value "
                f"({pivot}) matches the current dimension's units."
            )

    # 3. Append new processing parameters to the copied dict
    da_phased.attrs[ATTRS.phase_p0] = p0
    da_phased.attrs[ATTRS.phase_p1] = p1
    da_phased.attrs[ATTRS.phase_pivot] = pivot
    da_phased.attrs[ATTRS.phase_pivot_coord] = dim

    return da_phased


# --- Private Scoring Functions ---
def _acme_score(ph, da, dim, pivot):
    """ACME objective function natively using xmris coordinates."""
    p0 = ph[0]
    p1 = ph[1] if len(ph) > 1 else 0.0

    phased_da = phase(da, dim=dim, p0=p0, p1=p1, pivot=pivot)
    data = np.real(phased_da.values)

    stepsize = 1
    ds1 = np.abs((data[1:] - data[:-1]) / (stepsize * 2))
    p1_prob = ds1 / np.sum(ds1)
    p1_prob[p1_prob == 0] = 1

    h1 = -p1_prob * np.log(p1_prob)
    h1s = np.sum(h1)

    as_ = data - np.abs(data)
    sumas = np.sum(as_)
    pfun = 0.0
    if sumas < 0:
        pfun = np.sum((as_ / 2) ** 2)

    return (h1s + 1000 * pfun) / data.shape[-1] / np.max(data)


def _peak_minima_score(ph, da, dim, pivot, target_idx, index_width):
    """Minima-minimization around the target peak for sparse spectra."""
    p0 = ph[0]
    p1 = ph[1] if len(ph) > 1 else 0.0

    phased_da = phase(da, dim=dim, p0=p0, p1=p1, pivot=pivot)
    data = np.real(phased_da.values)

    start = max(0, target_idx - index_width)
    end = min(len(data), target_idx + index_width)

    mina = np.min(data[start:target_idx]) if start < target_idx else data[target_idx]
    minb = np.min(data[target_idx:end]) if end > target_idx else data[target_idx]

    return np.abs(mina - minb)


def _roi_positivity_score(ph, da, dim, pivot, target_idx, index_width):
    """Maximizes positive real signal and penalizes negative signal within an ROI."""
    p0 = ph[0]
    p1 = ph[1] if len(ph) > 1 else 0.0

    phased_da = phase(da, dim=dim, p0=p0, p1=p1, pivot=pivot)
    data = np.real(phased_da.values)

    start = max(0, target_idx - index_width)
    end = min(len(data), target_idx + index_width)
    data_roi = data[start:end]

    pos_reward = np.sum(data_roi[data_roi > 0])
    neg_penalty = np.sum(np.abs(data_roi[data_roi < 0])) * 5.0

    return neg_penalty - pos_reward


# --- Public API ---
def autophase(
    da: xr.DataArray,
    dim: str = DIMS.frequency,
    method: str = "acme",
    peak_width: float = 0.5,
    target_coord: float | None = None,
    p0_only: bool = False,
    lb: float = 0.0,
    temp_time_dim: str = DIMS.time,
    **kwargs,
) -> xr.DataArray:
    """
    Automatically calculate and apply phase correction to a spectrum.

    Parameters
    ----------
    da : xr.DataArray
        The input frequency-domain spectrum.
    dim : str, optional
        The coordinate dimension to operate on, by default `DIMS.frequency`.
    method : {"acme", "peak_minima", "positivity"}, optional
        The scoring algorithm to use. "acme" relies on entropy and is best for
        multi-peak high SNR spectra. "positivity" and "peak_minima" are optimized
        for sparse/noisy spectra. By default "acme".
    peak_width : float, optional
        Width of the ROI (in units of `dim`, e.g., Hz or ppm) for the local methods.
        Concentrates the solver on the region surrounding the target peak.
        By default 0.5.
    target_coord : float | None, optional
        The explicit coordinate (e.g. 171.0 ppm) to target for local methods.
        If None, the coordinate of the maximum absolute magnitude is used.
    p0_only : bool, optional
        If True, locks p1=0 and only optimizes the zero-order phase. Highly
        recommended for sparse spectra evaluated over a narrow `peak_width`.
    lb : float, optional
        Optional exponential line broadening (in Hz). Can help smooth extreme
        noise for ACME, but usually unnecessary for local methods. By default 0.0.
    temp_time_dim : str, optional
        The name used for the temporary time dimension if lb > 0.
    **kwargs :
        Additional keyword arguments passed to `scipy.optimize.differential_evolution`.

    Returns
    -------
    xr.DataArray
        The phased spectrum.
    """
    _check_dims(da, dim, "autophase")
    kwargs.setdefault("disp", False)

    coords = da.coords[dim].values

    # 1. Determine the target coordinate/index and pivot
    if target_coord is not None:
        target_idx = int(np.argmin(np.abs(coords - target_coord)))
        pivot = float(target_coord)
    else:
        target_idx = int(np.argmax(np.abs(da.values)))
        pivot = float(coords[target_idx])

    # 2. Convert physical peak_width to index points
    step_size = np.abs(coords[1] - coords[0])
    index_width = int(round((peak_width / 2.0) / step_size))
    index_width = max(1, index_width)

    # 3. Optional preprocessing
    if lb > 0:
        temp_fid = to_fid(da, dim=dim, out_dim=temp_time_dim)
        temp_apodized_fid = apodize_exp(temp_fid, dim=temp_time_dim, lb=lb)
        work_da = to_spectrum(temp_apodized_fid, dim=temp_time_dim, out_dim=dim)
    else:
        work_da = da

    # 4. Routing
    if method == "acme":
        score_fn = _acme_score
        args = (work_da, dim, pivot)
    elif method == "peak_minima":
        score_fn = _peak_minima_score
        args = (work_da, dim, pivot, target_idx, index_width)
    elif method == "positivity":
        score_fn = _roi_positivity_score
        args = (work_da, dim, pivot, target_idx, index_width)
    else:
        raise ValueError("Method must be 'acme', 'peak_minima', or 'positivity'")

    # 5. Bounded Global Optimization
    if p0_only:
        bounds = [(-180.0, 180.0)]
    else:
        bounds = [(-180.0, 180.0), (-4000.0, 4000.0)]

    opt = scipy.optimize.differential_evolution(
        score_fn,
        bounds=bounds,
        args=args,
        strategy="best1bin",
        tol=0.01,
        seed=42,  # Consider exposing this to the user later if stochasticity is desired
        disp=kwargs.get("disp"),
    )

    p0_opt = opt.x[0]
    p1_opt = opt.x[1] if not p0_only else 0.0

    return phase(da, dim=dim, p0=p0_opt, p1=p1_opt, pivot=pivot)
