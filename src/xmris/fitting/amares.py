from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Import pyAMARES and its core utilities
import pyAMARES
import xarray as xr
from joblib import Parallel, delayed
from pyAMARES import (
    initialize_FID,
    multieq6,
    result_pd_to_params,
    uninterleave,
)
from pyAMARES.kernel.lmfit import fitAMARES as pyamares_fitAMARES
from pyAMARES.libs.logger import set_log_level
from tqdm.auto import tqdm


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
    set_log_level("info" if verbose else "error", verbose=False)
    try:
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
        print(f"Warning: AMARES fit failed on a voxel. Returning NaNs. Error: {e}")
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
        If True, sets logging level to INFO. Default is False -> log level ERROR.

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
            verbose,  # <-- Pass to worker
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
    with tqdm(total=n_spectra, desc="Fitting Spectra") as pbar:
        for i, res in enumerate(parallel_gen):
            result_array[i] = res
            pbar.update(1)

    timeafter = datetime.now()
    print(
        f"Fitting {n_spectra} spectra with {num_workers} workers took "
        f"{(timeafter - timebefore).total_seconds():.2f} seconds."
    )

    return result_array


def fit_amares(
    da: xr.DataArray,
    prior_knowledge_file: str | Path,
    dim: str = "time",
    mhz: float | None = None,
    sw: float | None = None,
    deadtime: float | None = None,
    method: str = "leastsq",
    initialize_with_lm: bool = True,
    num_workers: int = 4,
    init_fid: np.ndarray | None = None,
    verbose: bool = False,
) -> xr.Dataset:
    """
    Apply AMARES time-domain fitting to an N-dimensional Free Induction Decay (FID).

    This function isolates the stateful pyAMARES API to perform parallelized batch
    fitting across spatial or repetition dimensions. It automatically scans the
    dataset to initialize the fitting template using the voxel with the highest
    Signal-to-Noise Ratio (SNR), ensuring robust prior knowledge instantiation.

    The numerical results and the reconstructed time-domain fits are packed into
    an aligned xarray Dataset, preserving all physical coordinates.

    Parameters
    ----------
    da : xr.DataArray
        Input FID data. Must contain the specified time dimension.
    prior_knowledge_file : str | Path
        Path to the CSV or XLSX file containing the prior knowledge constraints.
    dim : str, optional
        The time dimension along which to fit, by default "time".
    mhz : float, optional
        Spectrometer frequency in MHz. If None, attempts to read from da.attrs['MHz'].
    sw : float, optional
        Spectral width in Hz. If None, attempts to calculate from `dim` coordinates.
    deadtime : float, optional
        Time delay before the first point in seconds. If None, defaults to 0.0.
    method : {"leastsq", "least_squares"}, optional
        Fitting method. Defaults to 'leastsq' (Levenberg-Marquardt).
    initialize_with_lm : bool, optional
        Run an internal Levenberg-Marquardt initializer before fitting. Defaults to True.
    num_workers : int, optional
        Number of parallel processes to spawn. Defaults to 4.
    init_fid : np.ndarray, optional
        A 1D complex array to use as the template for pyAMARES initialization. If None,
        the function automatically selects the spectrum with the highest SNR.
    verbose: bool, optional
        If True, sets logging level to INFO. Default is False -> log level ERROR.

    Returns
    -------
    xr.Dataset
        A dataset containing the original data, the fitted FIDs, the residuals,
        and quantified parameters (amplitude, chem_shift, linewidth, phase, CRLB, SNR)
        mapped across the original dimensions and the new 'Metabolite' dimension.
    """
    set_log_level("info" if verbose else "error", verbose=False)

    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' missing in DataArray.")

    # 1. Extract/Infer Physical Parameters
    if mhz is None:
        mhz = da.attrs.get("MHz")
        if mhz is None:
            raise ValueError("mhz must be provided or present in da.attrs['MHz']")

    if sw is None:
        # Assuming the coordinate is in seconds
        dt = float(da.coords[dim].values[1] - da.coords[dim].values[0])
        sw = 1.0 / dt

    if deadtime is None:
        deadtime = float(da.coords[dim].values[0])

    # 2. Flatten N-dimensional DataArray to 2D NumPy array (N_spectra x Time)
    other_dims = [d for d in da.dims if d != dim]

    if len(other_dims) > 0:
        stacked_da = da.stack(spectrum=other_dims).transpose("spectrum", dim)
        fid_arrs = stacked_da.values
        stacked_coords = stacked_da.coords["spectrum"]
    else:
        fid_arrs = np.atleast_2d(da.values)

    n_spectra, n_time = fid_arrs.shape

    # 3. Smart Initialization: Find the best FID to initialize pyAMARES
    if init_fid is not None:
        template_fid = np.asarray(init_fid)
    else:
        # Vectorized SNR Calculation (matches pyAMARES.fidSNR logic)
        signal_region = np.mean(np.abs(fid_arrs[:, 0:10]), axis=1)
        noise_pts = max(10, n_time // 5)
        noise_region = np.std(fid_arrs[:, -noise_pts:], axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            snr_array = np.where(noise_region == 0, 0, signal_region / noise_region)

        best_idx = np.nanargmax(snr_array)
        template_fid = fid_arrs[best_idx]
        print(
            f"Auto-selected FID index {best_idx} for initialization "
            f"(SNR: {snr_array[best_idx]:.2f})"
        )

    # 4. Setup the Shared pyAMARES State
    # We suppress plotting and previewing to maintain functional purity
    shared_obj = initialize_FID(
        fid=template_fid,
        priorknowledgefile=str(prior_knowledge_file),
        MHz=mhz,
        sw=sw,
        deadtime=deadtime,
        normalize_fid=False,
        preview=False,
    )

    # 5. Execute Fitting natively via xmris
    if num_workers == 1:
        # BYPASS multiprocessing entirely for testing/single-core execution
        result_list = []
        for i in tqdm(range(n_spectra), desc="Fitting Spectra (Single Core)"):
            res = _fit_dataset_safe(
                fid_arrs[i, :],
                FIDobj_shared=shared_obj,
                initial_params=shared_obj.initialParams,
                method=method,
                initialize_with_lm=initialize_with_lm,
            )
            result_list.append(res)
    else:
        # Use our optimized joblib executor
        result_list = _run_parallel_fitting_optimal(
            fid_arrs=fid_arrs,
            FIDobj_shared=shared_obj,
            initial_params=shared_obj.initialParams,
            method=method,
            initialize_with_lm=initialize_with_lm,
            num_workers=num_workers,
        )

    # 6. Extract Parameters and Reconstruct Time-Domain Fits
    metabolites = result_list[0].index.values
    n_metab = len(metabolites)

    # Allocate parameter arrays
    amplitudes = np.zeros((n_spectra, n_metab))
    chem_shifts = np.zeros((n_spectra, n_metab))
    linewidths = np.zeros((n_spectra, n_metab))
    phases = np.zeros((n_spectra, n_metab))
    crlbs = np.zeros((n_spectra, n_metab))
    snrs = np.zeros((n_spectra, n_metab))

    # Allocate time-domain array
    fit_data = np.zeros((n_spectra, n_time), dtype=complex)

    # Calculate the time axis exactly as pyAMARES does it internally
    dwelltime = 1.0 / sw
    timeaxis = np.arange(0, dwelltime * n_time, dwelltime) + deadtime

    for i, df in enumerate(result_list):
        if df is None or df.isna().all().all():
            # Handle cases where the fit failed completely and returned NaNs
            # The zeros allocated above will naturally persist for these voxels
            continue

        amplitudes[i, :] = df["amplitude"].values
        chem_shifts[i, :] = df["chem shift(ppm)"].values
        linewidths[i, :] = df["LW(Hz)"].values
        phases[i, :] = df["phase(deg)"].values
        crlbs[i, :] = df["CRLB(%)"].values
        if "SNR" in df.columns:
            snrs[i, :] = df["SNR"].values

        # Reconstruct the time-domain model from the resulting DataFrame
        params = result_pd_to_params(df, MHz=mhz)
        fit_data[i, :] = uninterleave(multieq6(params, timeaxis))

    # 7. Construct the xarray Dataset
    ds = xr.Dataset()

    if len(other_dims) > 0:
        # A) Assemble the Time-Domain Variables (Restore exact original order)
        ds["raw_data"] = (
            xr.DataArray(
                fid_arrs,
                dims=["spectrum", dim],
                coords={"spectrum": stacked_coords, dim: da.coords[dim]},
            )
            .unstack("spectrum")
            .transpose(*da.dims)
        )

        ds["fit_data"] = (
            xr.DataArray(
                fit_data,
                dims=["spectrum", dim],
                coords={"spectrum": stacked_coords, dim: da.coords[dim]},
            )
            .unstack("spectrum")
            .transpose(*da.dims)
        )

        # B) Assemble the Parameter Variables
        param_coords = {"spectrum": stacked_coords, "Metabolite": metabolites}
        param_dims = ["spectrum", "Metabolite"]

        # Define the strict output dimension order (e.g., ["x", "y", "Metabolite"])
        out_param_dims = tuple(other_dims) + ("Metabolite",)

        ds["amplitude"] = (
            xr.DataArray(amplitudes, dims=param_dims, coords=param_coords)
            .unstack("spectrum")
            .transpose(*out_param_dims)
        )
        ds["chem_shift"] = (
            xr.DataArray(chem_shifts, dims=param_dims, coords=param_coords)
            .unstack("spectrum")
            .transpose(*out_param_dims)
        )
        ds["linewidth"] = (
            xr.DataArray(linewidths, dims=param_dims, coords=param_coords)
            .unstack("spectrum")
            .transpose(*out_param_dims)
        )
        ds["phase"] = (
            xr.DataArray(phases, dims=param_dims, coords=param_coords)
            .unstack("spectrum")
            .transpose(*out_param_dims)
        )
        ds["crlb"] = (
            xr.DataArray(crlbs, dims=param_dims, coords=param_coords)
            .unstack("spectrum")
            .transpose(*out_param_dims)
        )
        ds["snr"] = (
            xr.DataArray(snrs, dims=param_dims, coords=param_coords)
            .unstack("spectrum")
            .transpose(*out_param_dims)
        )

    else:
        # Handle the 1D case directly without unstacking
        ds["raw_data"] = da
        ds["fit_data"] = xr.DataArray(
            fit_data[0], dims=[dim], coords={dim: da.coords[dim]}
        )

        param_coords = {"Metabolite": metabolites}
        ds["amplitude"] = xr.DataArray(
            amplitudes[0], dims=["Metabolite"], coords=param_coords
        )
        ds["chem_shift"] = xr.DataArray(
            chem_shifts[0], dims=["Metabolite"], coords=param_coords
        )
        ds["linewidth"] = xr.DataArray(
            linewidths[0], dims=["Metabolite"], coords=param_coords
        )
        ds["phase"] = xr.DataArray(phases[0], dims=["Metabolite"], coords=param_coords)
        ds["crlb"] = xr.DataArray(crlbs[0], dims=["Metabolite"], coords=param_coords)
        ds["snr"] = xr.DataArray(snrs[0], dims=["Metabolite"], coords=param_coords)

    # 8. Calculate Residuals
    ds["residuals"] = ds["raw_data"] - ds["fit_data"]

    # 9. Preserve Lineage & Add Fit Metadata
    ds.attrs = da.attrs.copy()
    ds.attrs.update(
        {
            "fit_method": method,
            "prior_knowledge_file": str(prior_knowledge_file),
            "amares_version": pyAMARES.__version__,
        }
    )

    for coord in da.coords:
        if coord in ds.coords:
            ds.coords[coord].attrs.update(da.coords[coord].attrs)

    return ds
