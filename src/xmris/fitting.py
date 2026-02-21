import os
from pathlib import Path

import numpy as np

# Import pyAMARES and its core utilities
import pyAMARES
import xarray as xr
from pyAMARES import (
    initialize_FID,
    multieq6,
    result_pd_to_params,
    run_parallel_fitting_with_progress,
    uninterleave,
)


def fit_amares(
    da: xr.DataArray,
    prior_knowledge_file: str | Path,
    dim: str = "Time",
    mhz: float | None = None,
    sw: float | None = None,
    deadtime: float | None = None,
    method: str = "leastsq",
    initialize_with_lm: bool = True,
    num_workers: int = 4,
    init_fid: np.ndarray | None = None,
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
        The time dimension along which to fit, by default "Time".
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

    Returns
    -------
    xr.Dataset
        A dataset containing the original data, the fitted FIDs, the residuals,
        and quantified parameters (amplitude, chem_shift, linewidth, phase, CRLB, SNR)
        mapped across the original dimensions and the new 'Metabolite' dimension.
    """
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
            f"Auto-selected FID index {best_idx} for initialization"
            + f"(SNR: {snr_array[best_idx]:.2f})"
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

    # 5. Execute Parallel Batch Fitting
    result_list = run_parallel_fitting_with_progress(
        fid_arrs,
        FIDobj_shared=shared_obj,
        initial_params=shared_obj.initialParams,
        method=method,
        initialize_with_lm=initialize_with_lm,
        num_workers=num_workers,
        notebook=False,
        logfilename=os.devnull,
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
        if df is None:
            # Handle rare cases where the fit fails completely on a voxel
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

    return ds
