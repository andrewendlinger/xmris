import warnings

import numpy as np
import scipy.optimize
import xarray as xr

from xmris.core.config import ATTRS, DIMS, VARS
from xmris.core.utils import _check_dims
from xmris.processing.fid import to_spectrum
from xmris.processing.phasing import _acme_cost


def remove_digital_filter(
    da: xr.DataArray,
    group_delay: float | str = "header",
    dim: str = DIMS.time,
    keep_length: bool = True,
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
    group_delay : float or {"header", "measure"}, optional
        The delay value (in samples) to remove. By default ``"header"``, which
        reads the vendor-reported value from ``da.attrs`` (the ``group_delay``
        attribute written by :func:`build_fid`, mapping Bruker
        ``ACQ_RxFilterInfo``[0].groupDelay). Pass an explicit float to force a
        value, or ``"measure"`` to estimate it from the data via
        :func:`estimate_group_delay` — robust when the header under-counts the
        true digital-filter delay. Typical header values:
          - ~76.0 for standard high-resolution Spectroscopy.
          - ~0.0 to 16.0 for Fast Imaging or ZTE (where hardware pre-compensation
            or short filters are used).
    dim : str, optional
        The time dimension along which to apply the correction, by default
        ``DIMS.time``.
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

    # Resolve string sentinels ("header"/"measure") to a numeric delay before any
    # arithmetic — the guards below assume a float.
    group_delay = _resolve_group_delay(da, group_delay, dim)

    if group_delay <= 0:
        return da.copy()

    # 1. Separate the delay into integer (points) and fractional (phase) components
    int_delay = int(np.floor(group_delay))
    if int_delay >= da.sizes[dim]:
        raise ValueError(
            f"group_delay ({group_delay}) removes {int_delay} points, but '{dim}' has only "
            f"{da.sizes[dim]}. Pass a delay smaller than the acquisition length."
        )
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

    # 6. Preserve Lineage — record only the quantifiable parameter applied (Commandment 3).
    new_attrs = da.attrs.copy()
    new_attrs[ATTRS.group_delay_removed] = group_delay

    return da_new.assign_attrs(new_attrs)


# Legacy attr key, read-only, for backward compatibility with FIDs saved before the
# vendor-agnostic ATTRS.group_delay ("group_delay") rename (#85 metadata tidy).
_LEGACY_GROUP_DELAY_KEY = "bruker_group_delay"


def _read_group_delay_attr(da: xr.DataArray) -> float | None:
    """Read the stored group-delay attr, falling back to the legacy Bruker key."""
    val = da.attrs.get(ATTRS.group_delay, da.attrs.get(_LEGACY_GROUP_DELAY_KEY))
    return None if val is None else float(val)


def _resolve_group_delay(da: xr.DataArray, group_delay: float | str, dim: str) -> float:
    """Resolve a ``group_delay`` argument (float or ``"header"``/``"measure"``) to samples."""
    if not isinstance(group_delay, str):
        return float(group_delay)

    if group_delay == "header":
        header = _read_group_delay_attr(da)
        if header is None:
            raise ValueError(
                f"remove_digital_filter(group_delay='header') needs the "
                f"'{ATTRS.group_delay}' attribute (written by build_fid), which is "
                f"absent. Pass an explicit float, or use group_delay='measure'."
            )
        return header

    if group_delay == "measure":
        measured = estimate_group_delay(da, dim=dim)
        assert not isinstance(measured, tuple)  # return_profile defaults False
        return measured

    raise ValueError(f"group_delay string must be 'header' or 'measure', got {group_delay!r}.")


# φ0 search grid (degrees) for the ACME residual. -180 and +180 are the same phase, so the grid
# stops one step short of +180 to avoid scoring a duplicate.
_PHI0_GRID = np.linspace(-180.0, 175.0, 72)
# Fraction of the best-to-median finite-cost spread within which a local minimum counts as a
# competing candidate during aliasing disambiguation.
_NEAR_MINIMA_BAND_FRAC = 0.12


def estimate_group_delay(
    da: xr.DataArray,
    dim: str = DIMS.time,
    *,
    search_range: tuple[float, float] | None = None,
    header_hint: float | None = None,
    window: float = 16.0,
    metric: str = "acme",
    refine: bool = True,
    return_profile: bool = False,
) -> float | tuple[float, xr.DataArray]:
    """
    Measure the true digital-filter group delay by minimizing residual phase.

    The vendor header value (Bruker ``ACQ_RxFilterInfo``/``GRPDLY``) can *under-count*
    the real receiver digital-filter group delay for some ParaVision/probe
    combinations, leaving a residual first-order (frequency-dependent) phase error
    after :func:`remove_digital_filter` — negligible near the carrier but large for
    peaks far from it. This estimator finds the delay that removes that residual.

    An incorrect delay is mathematically a residual *linear phase* across the spectrum,
    so the correct delay is the one that, after removal, makes the spectrum maximally
    absorptive under a *single* global zero-order phase (``φ0``). The discriminating
    power comes from forbidding first-order phase (``φ1``): delay and ``φ1`` are the
    same linear-phase degree of freedom, so any ``φ1`` freedom would make every delay
    look equally good. This is the peak-agnostic generalization of a tied-phase model
    residual. ``argmax(|FID|)`` is deliberately **not** used — it lands on the filter's
    ringing, not the true delay.

    Parameters
    ----------
    da : xr.DataArray
        Input free induction decay (FID) in the time domain. May be N-dimensional;
        the single highest-energy 1-D slice is used for estimation.
    dim : str, optional
        The time dimension, by default ``DIMS.time``.
    search_range : tuple of float, optional
        Explicit ``(low, high)`` delay bounds (samples) to search. If ``None``
        (default), the window is anchored on the header: ``header ± window``.
    header_hint : float, optional
        Vendor-reported delay to anchor the search on. If ``None``, falls back to
        ``da.attrs[group_delay]`` and then to a broad default range.
    window : float, optional
        Half-width (samples) of the header-anchored search window, by default 16.0.
    metric : {"acme", "coherence"}, optional
        Residual-phase cost. ``"acme"`` (default) minimizes the ACME entropy over a
        ``φ0`` grid (whole-spectrum, robust to linear-phase aliasing). ``"coherence"``
        is a cheaper ``φ0``-invariant phase-coherence proxy.
    refine : bool, optional
        If True (default), refine the best integer delay to sub-sample precision.
    return_profile : bool, optional
        If True, also return the cost-vs-delay profile (an ``xr.DataArray`` over a
        ``trial_delay`` axis) for diagnosing multimodality/aliasing. By default False.

    Returns
    -------
    float or tuple[float, xr.DataArray]
        The measured group delay in samples, or ``(delay, profile)`` when
        ``return_profile=True``.

    Warns
    -----
    UserWarning
        When the measured delay deviates strongly from the header (a likely
        under-counting header), or when the cost profile is ambiguous (several
        delays give a near-minimal residual — linear-phase aliasing).
    """
    _check_dims(da, dim, "estimate_group_delay")
    if metric not in ("acme", "coherence"):
        raise ValueError(f"metric must be 'acme' or 'coherence', got {metric!r}.")

    fid = _pick_representative_slice(da, dim)

    # Resolve the header anchor (explicit hint wins, else the stored attr).
    header = header_hint
    if header is None:
        header = _read_group_delay_attr(da)

    # Resolve the search window.
    if search_range is not None:
        lo, hi = float(search_range[0]), float(search_range[1])
    elif header is not None:
        lo, hi = header - window, header + window
    else:
        lo, hi = 0.0, 96.0
    lo = max(0.0, lo)
    max_delay = float(fid.sizes[dim] - 1)  # a delay must leave >= 1 point after slicing
    hi = min(hi, max_delay)  # never probe a delay >= the FID length
    if hi <= lo:
        raise ValueError(f"Invalid search range ({lo}, {hi}).")

    freq_dim = str(DIMS.frequency)

    def cost(delay: float) -> float:
        cleaned = remove_digital_filter(fid, group_delay=float(delay), dim=dim, keep_length=True)
        spec = to_spectrum(cleaned, dim=dim)
        return _residual_phase_cost(spec, metric, _PHI0_GRID)

    # 1. Coarse integer grid search.
    candidates = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1, dtype=float)
    costs = np.array([cost(d) for d in candidates])
    best_i = int(np.argmin(costs))

    # 2. Disambiguate near-equal minima (linear-phase aliasing) with an un-aliased phase-slope
    #    seed. Competitors are *local minima* within a band scaled on the finite cost range;
    #    non-finite (degenerate) costs are excluded so they cannot poison the band or the seed.
    gmin = float(costs[best_i])
    finite = np.isfinite(costs)
    safe = np.where(finite, costs, np.inf)
    is_local_min = np.r_[True, safe[1:] <= safe[:-1]] & np.r_[safe[:-1] <= safe[1:], True] & finite
    # Scale the band on the median finite cost, not the max: a garbage tail (very wide ranges)
    # or a no-op endpoint must not widen it enough to sweep in tail local minima.
    if finite.any():
        band = _NEAR_MINIMA_BAND_FRAC * max(float(np.median(costs[finite])) - gmin, 0.0)
    else:
        band = 0.0
    near = np.where(is_local_min & (costs <= gmin + band))[0]
    ambiguous = near.size > 1  # several competing local minima -> possible linear-phase aliasing
    if ambiguous:
        seed = _seed_absolute_delay(fid, dim, freq_dim, header, float(candidates[best_i]))
        if seed is not None and lo <= seed <= hi:
            best_i = int(near[np.argmin(np.abs(candidates[near] - seed))])
    best = float(candidates[best_i])

    # 3. Sub-sample fractional refinement around the chosen integer.
    if refine:
        r_lo, r_hi = max(0.0, best - 1.0), min(best + 1.0, max_delay)
        if r_hi > r_lo:
            res = scipy.optimize.minimize_scalar(
                cost,
                bounds=(r_lo, r_hi),
                method="bounded",
                options={"xatol": 1e-2},
            )
            if res.success and res.fun <= costs[best_i]:
                best = max(0.0, float(res.x))

    # 4. Advisory warnings.
    if header is not None and abs(best - header) > 2.0:
        warnings.warn(
            f"Measured group delay ({best:.3f}) deviates from the header value "
            f"({header:.3f}) by {best - header:+.3f} samples; the header may "
            f"under-count the true digital-filter delay for this acquisition.",
            stacklevel=2,
        )
    if ambiguous and np.any(np.abs(candidates[near] - best) > 2.0):
        warnings.warn(
            "Group-delay estimate is ambiguous: several candidate delays give a "
            "near-minimal residual (linear-phase aliasing). Inspect the profile via "
            "return_profile=True and consider constraining search_range.",
            stacklevel=2,
        )

    if return_profile:
        # xmris-diagnostic-dim: "trial_delay" is a deliberate LOCAL dimension label for this
        # debug-only profile. It is intentionally NOT added to xmris.core.config DIMS/COORDS,
        # to keep that vocabulary limited to physical/acquisition axes. Revocable: if a
        # diagnostic-axis vocabulary is later introduced, grep "xmris-diagnostic-dim" to
        # migrate every such site (e.g. amares' "spectrum"/"Metabolite" output axes).
        profile = xr.DataArray(
            costs,
            dims=["trial_delay"],
            coords={"trial_delay": candidates},
            name="residual_phase_cost",
            attrs={"long_name": "Residual first-order phase cost", "metric": metric},
        )
        return best, profile
    return best


def _pick_representative_slice(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Reduce an N-D FID to the single 1-D slice carrying the most signal energy.

    Group-delay estimation needs one high-SNR FID; the max-energy slice (summed over
    the time axis) is a robust choice for multi-repetition / multi-coil inputs.
    """
    if da.ndim == 1:
        return da
    energy = (da * da.conj()).real.sum(dim=dim)
    unraveled = np.unravel_index(int(np.argmax(energy.values)), energy.shape)
    sel = {d: int(unraveled[i]) for i, d in enumerate(energy.dims)}
    return da.isel(sel)


def _residual_phase_cost(spec: xr.DataArray, metric: str, p0_grid: np.ndarray) -> float:
    """Score residual first-order phase in a spectrum, invariant to zero-order phase."""
    if metric == "acme":
        # Minimize the ACME score over a coarse φ0 grid (deterministic). Rotating by each φ0 is a
        # scalar op on the raw real/imag arrays — far cheaper than rebuilding an xarray per grid
        # point — and a degenerate spectrum yields inf, which we drop.
        re, im = spec.real.values, spec.imag.values
        scores = (_acme_cost(re * np.cos(a) - im * np.sin(a)) for a in np.radians(p0_grid))
        return min((c for c in scores if np.isfinite(c)), default=np.inf)
    # "coherence": φ0-invariant phase coherence, 0 when all points share one phase; a degenerate
    # all-zero spectrum has no coherence to speak of and scores as worst (inf), not best (0).
    s = spec.values
    denom = float(np.sum(np.abs(s)))
    if denom == 0.0:
        return np.inf
    return 1.0 - float(np.abs(np.sum(s))) / denom


def _seed_absolute_delay(
    fid: xr.DataArray, dim: str, freq_dim: str, header: float | None, fallback: float
) -> float | None:
    """Un-aliased absolute-delay seed from the magnitude-weighted phase slope.

    Removes an anchor delay, then reads the residual delay from the slope of unwrapped
    phase vs frequency: a residual Δd imprints ``φ(f) = -2π·Δd·f/f_s``. Returns
    ``anchor + Δd``, used only to break near-equal ACME minima.
    """
    anchor = header if header is not None else fallback
    try:
        cleaned = remove_digital_filter(fid, group_delay=float(anchor), dim=dim, keep_length=True)
        spec = to_spectrum(cleaned, dim=dim)
        s = spec.values
        f = spec.coords[freq_dim].values.astype(float)
        mag = np.abs(s)
        if mag.max() == 0.0 or f.size < 2:
            return None
        ang = np.unwrap(np.angle(s))
        # Magnitude-weighted least-squares slope. np.polyfit applies the weights inside the
        # squared residual, so w=mag reproduces the intended mag**2 weighting.
        slope = np.polyfit(f, ang, 1, w=mag)[0]  # rad/Hz
        if not np.isfinite(slope):
            return None
        fs = spec.sizes[freq_dim] * abs(f[1] - f[0])  # spectral width [Hz]
        residual = -slope * fs / (2.0 * np.pi)
        return float(anchor + residual)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        # e.g. remove_digital_filter rejecting a too-large anchor (Fix 1) — skip the seed.
        return None


def _get_val(pv_params: dict, key: str, default=None):
    """Helper to cleanly extract scalar values from Bruker's array-wrapped params."""  # noqa: D401
    val = pv_params.get(key, default)
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        return val[0]
    return val


def reshape_bruker_raw(raw_data_1d: np.ndarray, pv_params: dict) -> tuple[np.ndarray, list[str]]:
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
            ATTRS.group_delay: groupDelay,
            "units": "a.u.",
        }
    )
