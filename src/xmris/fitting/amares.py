import contextlib
import logging
import os
import sys
import tempfile
import warnings
from collections.abc import Iterator, Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from pyAMARES import (
    initialize_FID,
    multieq6,
    result_pd_to_params,
    uninterleave,
)
from pyAMARES.kernel.lmfit import fitAMARES as pyamares_fitAMARES
from pyAMARES.libs import logger as _pa_logger
from pyAMARES.libs.logger import set_log_level
from tqdm.auto import tqdm

from xmris.core.config import ATTRS, DIMS, SPECTRAL_DIMS, TIME_DIMS, VARS
from xmris.core.options import OPTIONS
from xmris.core.utils import _check_dims
from xmris.core.validation import (
    _coerce_to_domain,
    _domain_of,
    _restore_domain,
    _strict_domain_error,
)
from xmris.fitting.prior_knowledge import build_prior_knowledge

# --- Logging & verbosity (BUG-010) -------------------------------------------
# pyAMARES emits noise on four channels; `verbose=False` must silence all of them,
# and it must hold inside joblib workers (which re-import this module fresh), not
# only the main process. So verbosity is (re)applied per fit call rather than once
# — replacing the in-process stdout redirect that used to force `num_workers=1`.

logger = logging.getLogger("xmris.fitting")
if not logger.handlers:  # give xmris its own stdout handler, as pyAMARES does
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[xmris | %(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


def _set_verbosity(verbose: bool) -> None:
    """Set the pyAMARES + xmris fitting log levels for the current process."""
    level = "info" if verbose else "error"
    # pyAMARES creates some loggers lazily *during* a fit; setting the module
    # default makes those honor the level too, not only loggers that already exist.
    _pa_logger.DEFAULT_LOG_LEVEL = level
    set_log_level(level, verbose=False)
    logger.setLevel(logging.INFO if verbose else logging.ERROR)


@contextlib.contextmanager
def _muted_warnings(verbose: bool):
    """Mute the routine warnings a fit emits unless ``verbose``.

    The scipy ``xtol``/``ftol`` UserWarning (from our magnitude-normalized
    tolerance) and the pyAMARES ``fid.py`` divide-by-zero RuntimeWarning (an
    exactly-zero spectrum) are expected here and would otherwise flood a batch fit.
    """
    if verbose:
        yield
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


# pyAMARES result-table column labels (its stateful API speaks these, not the config
# vocabulary). Kept local so the mapping to VARS lives in exactly one place.

# Named per-metabolite value variables: VARS name -> pyAMARES value column.
_VALUE_COLS = {
    VARS.amplitude: "amplitude",
    VARS.chem_shift: "chem shift(ppm)",
    VARS.linewidth: "LW(Hz)",
    VARS.phase: "phase(deg)",
    VARS.snr: "SNR",
}

# Per-parameter uncertainties, gathered along the `parameter` dim as `crlb`/`sd`
# variables: parameter -> (sd column, CRLB% column). Only these four parameters
# have a named value var to pair with (g is typically fixed and unreported).
# Amplitude's columns are unqualified in pyAMARES and `CRLB(cs%) ` carries a
# trailing space, so every label is pinned here explicitly.
_UNCERTAINTY_COLS = {
    VARS.amplitude: ("sd", "CRLB(%)"),
    VARS.chem_shift: ("sd(ppm)", "CRLB(cs%) "),
    VARS.linewidth: ("sd(Hz)", "CRLB(LW%)"),
    VARS.phase: ("sd(deg)", "CRLB(phase%)"),
}

# The `parameter` coordinate values, in a fixed order.
_PARAMETERS = list(_UNCERTAINTY_COLS)


def _fit_dataset_safe(
    fid_current,
    FIDobj_shared,
    initial_params,
    method="leastsq",
    initialize_with_lm=False,
    verbose=False,
):
    """
    Safely fit a single FID dataset using the pyAMARES algorithm.

    This internal helper function performs the fitting of a single spectrum. It
    deep copies the shared FID object to avoid race conditions and state corruption
    during multiprocessing. If the fitting process raises an exception (e.g., due
    to non-convergence or bad data), it catches the error and returns a DataFrame
    populated with NaNs. This ensures that downstream array concatenations in
    N-dimensional datasets do not fail due to mismatched shapes or `None` types.

    Parameters
    ----------
    fid_current : numpy.ndarray
        The 1D complex array representing the current Free Induction Decay (FID)
        dataset to be fitted.
    FIDobj_shared : argparse.Namespace
        A shared pyAMARES FID object template containing common settings,
        such as spectrometer frequency, spectral width, and dead time.
    initial_params : lmfit.Parameters
        The initialized fitting parameters and prior knowledge constraints
        used for the AMARES algorithm.
    method : {"leastsq", "least_squares"}, optional
        The minimization method to be passed to `lmfit`. Defaults to "leastsq"
        (Levenberg-Marquardt).
    initialize_with_lm : bool, optional
        If True, an internal Levenberg-Marquardt initializer is executed to
        refine starting values before the main fitting routine. Defaults to False.
    verbose: bool, optional
        If True, sets logging level to INFO. Default is False -> log level ERROR.


    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the fitting results (e.g., amplitude, linewidth,
        chemical shift, phase, CRLB, SNR) for the current dataset. If the fit
        fails, returns a structurally identical DataFrame filled with NaNs.
    """
    # Re-apply verbosity here so it also holds in joblib worker processes.
    _set_verbosity(verbose)
    try:
        with _muted_warnings(verbose):
            FIDobj_current = deepcopy(FIDobj_shared)
            FIDobj_current.fid = fid_current

            out = pyamares_fitAMARES(
                fid_parameters=FIDobj_current,
                fitting_parameters=initial_params,
                method=method,
                initialize_with_lm=initialize_with_lm,
                ifplot=False,
                inplace=True,
            )

            result_table = FIDobj_current.result_multiplets

            # Explicit memory cleanup in the worker process
            del FIDobj_current
            del out

            return result_table

    except Exception as e:
        # Silent unless verbose — the NaN sentinel is the signal to check (isnan).
        logger.warning("AMARES fit failed on a voxel; returning NaNs. Error: %s", e)
        if hasattr(FIDobj_shared, "peaklist"):
            cols = [
                "amplitude",
                "sd",
                "CRLB(%)",
                "chem shift(ppm)",
                "sd(ppm)",
                "CRLB(cs%) ",
                "LW(Hz)",
                "sd(Hz)",
                "CRLB(LW%)",
                "phase(deg)",
                "sd(deg)",
                "CRLB(phase%)",
                "g",
                "g_sd",
                "g (%)",
                "SNR",
            ]
            dummy_df = pd.DataFrame(np.nan, index=FIDobj_shared.peaklist, columns=cols)
            dummy_df.index.name = "name"
            return dummy_df
        return None


def _run_parallel_fitting_optimal(
    fid_arrs,
    FIDobj_shared,
    initial_params,
    method="leastsq",
    initialize_with_lm=False,
    num_workers=8,
    verbose=False,
):
    """
    Execute parallel AMARES fitting across multiple FID datasets using `joblib`.

    This internal execution engine replaces the legacy `multiprocessing` approach.
    It uses the `loky` backend from `joblib` to efficiently manage worker pools
    and minimize memory overhead when passing large NumPy arrays (via memory mapping).
    It also utilizes joblib's generator return style to provide a completely accurate,
    non-blocking progress bar.

    Parameters
    ----------
    fid_arrs : numpy.ndarray
        A 2D array of shape (n_spectra, n_time_points) containing the stacked
        complex FID data to be fitted.
    FIDobj_shared : argparse.Namespace
        A shared pyAMARES FID object template containing common settings.
        Heavy visualization attributes (like `styled_df`) are stripped internally
        to avoid serialization overhead across processes.
    initial_params : lmfit.Parameters
        The initialized fitting parameters and prior knowledge constraints.
    method : {"leastsq", "least_squares"}, optional
        The minimization method to be passed to `lmfit`. Defaults to "leastsq".
    initialize_with_lm : bool, optional
        If True, an internal Levenberg-Marquardt initializer is executed before
        the main fitting routine. Defaults to False.
    num_workers : int, optional
        The number of concurrent worker processes to spawn. Defaults to 8.
    verbose: bool, optional
        If True, sets logging level to INFO and prints timing. Default is False.

    Returns
    -------
    numpy.ndarray
        A 1D object array of length `n_spectra`, where each element is a
        pandas DataFrame containing the fit results for the corresponding spectrum.
    """
    # Create a safe copy and strip heavy/unpicklable visualization attributes
    FIDobj_shared_clean = deepcopy(FIDobj_shared)
    for attr in ("styled_df", "simple_df", "out_obj", "fitted_fid"):
        if hasattr(FIDobj_shared_clean, attr):
            delattr(FIDobj_shared_clean, attr)

    timebefore = datetime.now()
    n_spectra = fid_arrs.shape[0]

    # Generate the task arguments
    args_list = [
        (
            fid_arrs[i, :],
            FIDobj_shared_clean,
            initial_params,
            method,
            initialize_with_lm,
            verbose,
        )
        for i in range(n_spectra)
    ]
    # Pre-allocate an object array to hold the resulting DataFrames
    result_array = np.empty(n_spectra, dtype=object)

    # Yield results immediately as they finish
    parallel_gen = Parallel(n_jobs=num_workers, backend="loky", return_as="generator")(
        delayed(_fit_dataset_safe)(*args) for args in args_list
    )

    # Process and assign back to the correct index
    with tqdm(total=n_spectra, desc="Fitting Spectra", disable=not verbose) as pbar:
        for i, res in enumerate(parallel_gen):
            result_array[i] = res
            pbar.update(1)

    logger.info(
        "Fitting %d spectra with %d workers took %.2f seconds.",
        n_spectra,
        num_workers,
        (datetime.now() - timebefore).total_seconds(),
    )

    return result_array


def _resolve_fit_domain(da: xr.DataArray, dim: str) -> tuple[xr.DataArray, str, tuple | None]:
    """Return a time-domain FID to fit, its time dim, and how to restore the input.

    AMARES fits in the time domain. This funnels the caller's data there while
    remembering its representation, so the result can be handed back in the same
    domain (FID in -> FID out, ppm in -> ppm out). It reuses the domain engine's
    own converter routing (``_coerce_to_domain``), so an inserted transform is
    bit-identical to an explicit ``to_fid()`` and a real-valued spectrum earns the
    same refusal.

    Parameters
    ----------
    da : xr.DataArray
        The input data — a FID (``dim`` present) or a complex spectrum.
    dim : str
        The requested time dimension.

    Returns
    -------
    tuple[xr.DataArray, str, tuple | None]
        The FID to fit, the time dimension to fit along, and the
        ``_RestoreState`` to invert the coercion (``None`` if no conversion ran).
    """
    if dim in da.dims and dim not in SPECTRAL_DIMS:
        # Already a FID along the requested axis (canonical `time` or a custom name).
        return da, dim, None

    if _domain_of(da, SPECTRAL_DIMS) is not None:
        # Spectral input: convert to a FID for the fit (unless strict mode forbids it).
        if not OPTIONS["auto_convert"]:
            raise _strict_domain_error(da, TIME_DIMS, "fit_amares", "time")
        fid, state = _coerce_to_domain(da, TIME_DIMS)
        return fid, str(DIMS.time), state

    # Neither the requested time dim nor a convertible spectral dim is present.
    _check_dims(da, dim, "fit_amares")
    return da, dim, None  # pragma: no cover — _check_dims raises above


@contextlib.contextmanager
def _resolve_pk_file(
    prior_knowledge: Mapping[str, Any] | pd.DataFrame | str | Path,
) -> Iterator[str]:
    """Yield a filesystem path to a pyAMARES prior-knowledge file.

    pyAMARES's parser takes only a CSV/XLSX *path*, so an in-memory spec is written
    to a temporary CSV (removed on exit). A dict is validated and built via
    :func:`build_prior_knowledge`; a DataFrame (already in pyAMARES's positional
    layout) is serialized as-is; a path is checked for existence and used directly.
    """
    text: str | None = None
    if isinstance(prior_knowledge, Mapping):
        text = build_prior_knowledge(prior_knowledge)
    elif isinstance(prior_knowledge, pd.DataFrame):
        text = prior_knowledge.to_csv()

    if text is None:
        path = str(prior_knowledge)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Prior-knowledge file not found: {path!r}. Pass a path to a CSV/XLSX "
                f"file, or an in-memory spec (see xmris.fitting.build_prior_knowledge)."
            )
        yield path
        return

    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    try:
        tmp.write(text)
        tmp.close()
        yield tmp.name
    finally:
        os.unlink(tmp.name)


def fit_amares(
    da: xr.DataArray,
    prior_knowledge: Mapping[str, Any] | pd.DataFrame | str | Path,
    dim: str = DIMS.time,
    mhz: float | None = None,
    sw: float | None = None,
    deadtime: float | None = None,
    g_global: float | bool = 0.0,
    method: str = "leastsq",
    initialize_with_lm: bool = False,
    num_workers: int = 4,
    init_fid: np.ndarray | None = None,
    verbose: bool = False,
) -> xr.Dataset:
    """
    Apply AMARES time-domain fitting to an N-dimensional signal.

    This function isolates the stateful pyAMARES API to perform parallelized batch
    fitting across spatial or repetition dimensions. It automatically scans the
    dataset to initialize the fitting template using the voxel with the highest
    Signal-to-Noise Ratio (SNR), ensuring robust prior knowledge instantiation.

    AMARES fits in the time domain. Following the domain-preserving contract, a
    spectrum handed to `fit_amares` is round-tripped through the FID for the fit and
    the returned time-domain variables (`data`, `fit`, `residuals`) are restored to
    the representation that was passed in (ppm in -> ppm out); a FID is fitted and
    returned as-is. The quantified parameters are domain-independent.

    Robustness: the FID is normalized by a single global factor before fitting — so
    pyAMARES's magnitude-derived optimizer tolerance behaves at any signal scale (a
    Bruker-scale FID no longer "converges" on the prior) — and the fitted amplitudes
    are rescaled back into the input units. A fit that fails is recorded as `NaN`, so
    it is distinguishable from a genuine zero-signal spectrum.

    Parameters
    ----------
    da : xr.DataArray
        Input data. A FID with the specified time dimension, or a complex spectrum
        (`frequency`/`chemical_shift`) that is converted to a FID for the fit.
    prior_knowledge : Mapping | pandas.DataFrame | str | Path
        The prior-knowledge constraints, either in memory or on disk. A mapping of
        peak name to parameters is built and validated via
        :func:`~xmris.fitting.build_prior_knowledge` (phase bounds, peak-name and
        tie-order traps handled for you); a ``str``/``Path`` is a pyAMARES CSV/XLSX
        file used directly; a DataFrame in pyAMARES's positional layout is accepted
        as-is.
    dim : str, optional
        The time dimension along which to fit, by default ``DIMS.time``.
    mhz : float, optional
        Spectrometer frequency in MHz. If None, read from
        ``da.attrs['reference_frequency']``.
    sw : float, optional
        Spectral width in Hz. If None, calculated from the `dim` coordinate spacing.
    deadtime : float, optional
        Acquisition time origin in seconds. If None, taken from the first `dim`
        coordinate value (the single source of truth for the time axis).
    g_global : float or bool, optional
        Global lineshape held for every peak: 0.0 = pure Lorentzian (default),
        1.0 = pure Gaussian, in between = pseudo-Voigt. Pass ``False`` instead to
        let each peak's ``g`` vary, fitted from the prior-knowledge value.
    method : {"leastsq", "least_squares"}, optional
        Fitting method. Defaults to 'leastsq' (Levenberg-Marquardt).
    initialize_with_lm : bool, optional
        Run an internal Levenberg-Marquardt initializer before fitting. Defaults to
        False (True can diverge on real data).
    num_workers : int, optional
        Number of parallel processes to spawn. Defaults to 4.
    init_fid : np.ndarray, optional
        A 1D complex array to use as the template for pyAMARES initialization. If None,
        the function automatically selects the spectrum with the highest SNR.
    verbose : bool, optional
        If True, sets logging level to INFO and prints progress. Default is False.

    Returns
    -------
    xr.Dataset
        A dataset containing the original data, the fitted model, the residuals, and
        the quantified parameters (amplitude, chem_shift, linewidth, phase, CRLB, SNR)
        mapped across the original dimensions and the new ``metabolite`` dimension.
    """
    _set_verbosity(verbose)

    # 1. Domain handling: obtain a FID to fit and remember how to restore the input.
    da_fid, dim, restore_state = _resolve_fit_domain(da, dim)

    # 2. Extract/infer physical parameters from the FID.
    if mhz is None:
        mhz = da_fid.attrs.get(ATTRS.reference_frequency)
        if mhz is None:
            raise ValueError(
                f"mhz must be provided or present in da.attrs[{ATTRS.reference_frequency!r}]."
            )

    if sw is None:
        dt = float(da_fid.coords[dim].values[1] - da_fid.coords[dim].values[0])
        sw = 1.0 / dt

    if deadtime is None:
        deadtime = float(da_fid.coords[dim].values[0])

    # 3. Flatten the N-dimensional FID to a 2D array (n_spectra x time).
    other_dims = [d for d in da_fid.dims if d != dim]
    if other_dims:
        stacked_da = da_fid.stack(spectrum=other_dims).transpose("spectrum", dim)
        fid_arrs = stacked_da.values
        stacked_coords = stacked_da.coords["spectrum"]
    else:
        fid_arrs = np.atleast_2d(da_fid.values)
        stacked_coords = None
    n_spectra, n_time = fid_arrs.shape

    # 4. Normalize by a single global factor so the optimizer's magnitude-derived
    #    tolerance is well-behaved. One factor for the whole array — never per
    #    spectrum, which would flatten a dynamic series.
    global_scale = float(np.abs(fid_arrs).max())
    if not np.isfinite(global_scale) or global_scale == 0.0:
        global_scale = 1.0  # nothing to normalize; degenerate fits fall through to NaN
    fid_norm = fid_arrs / global_scale
    spectrum_max = np.abs(fid_arrs).max(axis=1)  # per-spectrum: 0 => no signal => NaN

    # 5. Smart initialization: pick the highest-SNR (normalized) FID as the template.
    if init_fid is not None:
        template_fid = np.asarray(init_fid) / global_scale
    else:
        signal_region = np.mean(np.abs(fid_norm[:, 0:10]), axis=1)
        noise_pts = max(10, n_time // 5)
        noise_region = np.std(fid_norm[:, -noise_pts:], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_array = np.where(noise_region == 0, 0, signal_region / noise_region)
        best_idx = int(np.nanargmax(snr_array))
        template_fid = fid_norm[best_idx]
        logger.info(
            "Auto-selected FID index %d for initialization (SNR: %.2f)",
            best_idx,
            snr_array[best_idx],
        )

    # 6. Setup the shared pyAMARES state (normalize_fid=False — we normalize ourselves
    #    and rescale the amplitudes back, so results come out in the input units).
    #    An in-memory prior-knowledge spec is materialized to a temp CSV for the
    #    duration of this call (pyAMARES's parser takes only a file path).
    with _resolve_pk_file(prior_knowledge) as pk_path:
        shared_obj = initialize_FID(
            fid=template_fid,
            priorknowledgefile=pk_path,
            MHz=mhz,
            sw=sw,
            deadtime=deadtime,
            g_global=g_global,
            normalize_fid=False,
            preview=False,
        )

    # 7. Fit every spectrum.
    if num_workers == 1:
        result_list = [
            _fit_dataset_safe(
                fid_norm[i, :],
                FIDobj_shared=shared_obj,
                initial_params=shared_obj.initialParams,
                method=method,
                initialize_with_lm=initialize_with_lm,
                verbose=verbose,
            )
            for i in tqdm(range(n_spectra), desc="Fitting Spectra", disable=not verbose)
        ]
    else:
        result_list = _run_parallel_fitting_optimal(
            fid_arrs=fid_norm,
            FIDobj_shared=shared_obj,
            initial_params=shared_obj.initialParams,
            method=method,
            initialize_with_lm=initialize_with_lm,
            num_workers=num_workers,
            verbose=verbose,
        )

    # 8. Extract parameters (NaN sentinel for failed fits) and reconstruct the model.
    first_ok = next(
        (df for df in result_list if df is not None and not df.isna().all().all()), None
    )
    metabolites = np.asarray(first_ok.index.values if first_ok is not None else shared_obj.peaklist)
    n_metab = len(metabolites)

    value_out = {key: np.full((n_spectra, n_metab), np.nan) for key in _VALUE_COLS}
    n_param = len(_PARAMETERS)
    crlb_out = np.full((n_spectra, n_metab, n_param), np.nan)
    sd_out = np.full((n_spectra, n_metab, n_param), np.nan)
    fit_norm = np.full((n_spectra, n_time), np.nan, dtype=complex)

    dwelltime = 1.0 / sw
    timeaxis = np.arange(0, dwelltime * n_time, dwelltime) + deadtime

    for i, df in enumerate(result_list):
        # No signal to fit, a hard failure, or an all-NaN fit -> keep the NaN sentinel.
        if spectrum_max[i] == 0 or df is None or df.isna().all().all():
            continue
        for key, col in _VALUE_COLS.items():
            if col in df.columns:
                value_out[key][i, :] = df[col].values
        for p, param in enumerate(_PARAMETERS):
            sd_col, crlb_col = _UNCERTAINTY_COLS[param]
            if sd_col in df.columns:
                sd_out[i, :, p] = df[sd_col].values
            if crlb_col in df.columns:
                crlb_out[i, :, p] = df[crlb_col].values
        params = result_pd_to_params(df, MHz=mhz)
        fit_norm[i, :] = uninterleave(multieq6(params, timeaxis))

    # 9. Rescale amplitude + reconstructed model back into the input units. The
    #    amplitude *sd* scales with it; CRLB is relative (%), and the other
    #    parameters' uncertainties are independent of amplitude scale.
    value_out[VARS.amplitude] *= global_scale
    sd_out[:, :, _PARAMETERS.index(VARS.amplitude)] *= global_scale
    fit_arrs = fit_norm * global_scale

    # 10. Assemble the output Dataset in the caller's representation.
    ds = xr.Dataset()
    ds[VARS.original_data] = da  # the input, exactly as passed (FID or spectrum)

    if other_dims:
        fit_da = (
            xr.DataArray(
                fit_arrs,
                dims=["spectrum", dim],
                coords={"spectrum": stacked_coords, dim: da_fid.coords[dim]},
            )
            .unstack("spectrum")
            .transpose(*da_fid.dims)
        )
        param_dims = ["spectrum", DIMS.metabolite]
        param_coords = {"spectrum": stacked_coords, DIMS.metabolite: metabolites}
        out_param_dims = tuple(other_dims) + (DIMS.metabolite,)

        def _param_var(arr: np.ndarray) -> xr.DataArray:
            return (
                xr.DataArray(arr, dims=param_dims, coords=param_coords)
                .unstack("spectrum")
                .transpose(*out_param_dims)
            )

        def _uncertainty_var(arr: np.ndarray) -> xr.DataArray:
            return (
                xr.DataArray(
                    arr,
                    dims=[*param_dims, DIMS.parameter],
                    coords={**param_coords, DIMS.parameter: _PARAMETERS},
                )
                .unstack("spectrum")
                .transpose(*out_param_dims, DIMS.parameter)
            )
    else:
        fit_da = xr.DataArray(fit_arrs[0], dims=[dim], coords={dim: da_fid.coords[dim]})
        param_coords = {DIMS.metabolite: metabolites}

        def _param_var(arr: np.ndarray) -> xr.DataArray:
            return xr.DataArray(arr[0], dims=[DIMS.metabolite], coords=param_coords)

        def _uncertainty_var(arr: np.ndarray) -> xr.DataArray:
            return xr.DataArray(
                arr[0],
                dims=[DIMS.metabolite, DIMS.parameter],
                coords={DIMS.metabolite: metabolites, DIMS.parameter: _PARAMETERS},
            )

    # Restore the fit to the caller's representation (ppm in -> ppm out). The model
    # carries the input attrs so the ppm leg (`to_ppm`) finds `reference_frequency`.
    fit_da = fit_da.assign_attrs(dict(da.attrs))
    if restore_state is not None:
        fit_da = _restore_domain(fit_da, TIME_DIMS, restore_state)
    fit_da.attrs = {}

    ds[VARS.fit] = fit_da
    ds[VARS.residuals] = ds[VARS.original_data] - ds[VARS.fit]

    for var, arr in value_out.items():
        ds[var] = _param_var(arr)
    ds[VARS.crlb] = _uncertainty_var(crlb_out)
    ds[VARS.sd] = _uncertainty_var(sd_out)

    # 11. Preserve lineage: input attrs + the one quantitative fitting parameter.
    ds.attrs = dict(da.attrs)
    ds.attrs[ATTRS.amares_amplitude_scale] = global_scale
    for coord in da.coords:
        if coord in ds.coords:
            ds.coords[coord].attrs.update(da.coords[coord].attrs)

    return ds
