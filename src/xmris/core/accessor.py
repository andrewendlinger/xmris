from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Import our new core architecture
from xmris.core.config import ATTRS, COORDS, DIMS
from xmris.core.validation import requires_attrs
from xmris.processing.fid import apodize_exp, apodize_lg, to_fid, to_spectrum, zero_fill
from xmris.processing.fourier import fft, fftc, fftshift, ifft, ifftc, ifftshift
from xmris.processing.phase import autophase, phase
from xmris.vendor.bruker import remove_digital_filter

# Import the config type for type-hinting, but defer the actual plotting function
from xmris.visualization.plot import PlotHeatmapConfig, PlotRidgeConfig

# (Assuming imports for fid, fourier, phase, etc. remain the same)


def _check_dims(obj: xr.DataArray, dims: str | list[str], func_name: str):
    """Internal helper to validate explicit dimension arguments with UX-friendly errors."""
    dim_list = [dims] if isinstance(dims, str) else dims
    missing = [d for d in dim_list if d not in obj.dims]

    if missing:
        available = list(obj.dims)
        raise ValueError(
            f"Method '{func_name}' attempted to operate on missing dimension(s): {missing}.\n"
            f"Available dimensions are: {available}.\n\n"
            f"To fix this, either pass the correct `dim` string argument to the function, "
            f"or rename your data's axes using xarray:\n"
            f"    >>> obj = obj.rename({{{repr(missing[0])}: DIMS.time}})"
        )


class XmrisDatasetPlotAccessor:
    """Sub-accessor for xmris xr.Datasets plotting functionalities."""

    def __init__(self, obj: xr.Dataset):
        self._obj = obj

    def trajectory(
        self,
        dim: str,
        metabolites: list[str] | None = None,
        ax: plt.Axes | None = None,
        config=None,
    ):
        """Plot kinetic trajectories with CRLB shading."""
        from xmris.visualization.plot.plot_trajectory import plot_trajectory

        return plot_trajectory(
            self._obj, dim=dim, metabolites=metabolites, ax=ax, config=config
        )

    def qc_grid(self, dim: str, config=None):
        """Plot a grid of spectra and fits to quickly visually inspect quality."""
        from xmris.visualization.plot.plot_qc_grid import plot_qc_grid

        return plot_qc_grid(self._obj, dim=dim, config=config)


@xr.register_dataset_accessor("xmr")
class XmrisDatasetAccessor:
    """Accessor for xmris xr.Datasets (e.g., AMARES fitting results)."""

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._plot = None

    @property
    def plot(self) -> XmrisDatasetPlotAccessor:
        """Access xmris plotting functionalities."""
        if self._plot is None:
            self._plot = XmrisDatasetPlotAccessor(self._obj)
        return self._plot


class XmrisPlotAccessor:
    """Sub-accessor for xmris plotting functionalities (accessed via .xmr.plot)."""

    def __init__(self, obj: xr.DataArray):
        self._obj = obj

    def ridge(
        self,
        x_dim: str | None = None,
        stack_dim: str | None = None,
        ax: plt.Axes | None = None,
        config: PlotRidgeConfig | None = None,
    ) -> plt.Axes:
        """Generate a ridge plot (2D waterfall) of stacked 1D spectra."""
        # Deferred import to keep main package load times fast
        from xmris.visualization.plot import plot_ridge as _plot_ridge

        return _plot_ridge(
            da=self._obj,
            x_dim=x_dim,
            stack_dim=stack_dim,
            ax=ax,
            config=config,
        )

    def heatmap(
        self,
        x_dim: str | None = None,
        stack_dim: str | None = None,
        ax: plt.Axes | None = None,
        config: PlotHeatmapConfig | None = None,
    ) -> plt.Axes:
        """Generate a 2D heatmap plot of stacked 1D spectra."""
        from xmris.visualization.plot.plot_heatmap import plot_heatmap as _plot_heatmap

        return _plot_heatmap(
            da=self._obj, x_dim=x_dim, stack_dim=stack_dim, ax=ax, config=config
        )


@xr.register_dataarray_accessor("xmr")
class XmrisAccessor:
    """
    Accessor for xarray DataArrays to perform MRI and MRS operations.

    This class is registered under the `.xmr` namespace. It provides a
    fluent, method-chaining API for signal processing, spectroscopy, and
    imaging functions directly on xarray objects while preserving coordinates
    and metadata.

    Attributes
    ----------
    _obj : xr.DataArray
        The underlying xarray DataArray object being operated on.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        """Initialize the accessor with the xarray object."""
        self._obj = xarray_obj
        self._plot = None  # Cache for the plot sub-accessor

    # --- Plotting Sub-Accessor ---
    @property
    def plot(self) -> XmrisPlotAccessor:
        """Access xmris plotting functionalities."""
        # Lazy initialization: only create the object if the user asks for it
        if self._plot is None:
            self._plot = XmrisPlotAccessor(self._obj)
        return self._plot

    # --- Shifts ---

    def fftshift(self, dim: str | list[str]) -> xr.DataArray:
        """
        Apply fftshift by rolling data and coordinates along specified dimensions.

        Moves the zero-frequency component to the center of the spectrum.
        """
        return fftshift(self._obj, dim=dim)

    def ifftshift(self, dim: str | list[str]) -> xr.DataArray:
        """
        Apply ifftshift by rolling data and coordinates along specified dimensions.

        The inverse of :meth:`fftshift`.
        """
        return ifftshift(self._obj, dim=dim)

    # --- Pure Transforms ---

    def fft(
        self,
        dim: str | list[str] = "time",
        out_dim: str | list[str] | None = None,
    ) -> xr.DataArray:
        """
        Perform a standard N-dimensional FFT (no shifts).

        Optionally renames the transformed dimension(s) using `out_dim`.
        """
        return fft(self._obj, dim=dim, out_dim=out_dim)

    def ifft(
        self,
        dim: str | list[str] = "time",
        out_dim: str | list[str] | None = None,
    ) -> xr.DataArray:
        """
        Perform a standard N-dimensional Inverse FFT (no shifts).

        Optionally renames the transformed dimension(s) using `out_dim`.
        """
        return ifft(self._obj, dim=dim, out_dim=out_dim)

    # --- Centered Transforms ---

    def fftc(
        self,
        dim: str | list[str] = "time",
        out_dim: str | list[str] | None = None,
    ) -> xr.DataArray:
        """
        Perform an N-dimensional centered FFT.

        This applies necessary shifts before and after the transform to ensure
        the DC component is centered. Optionally renames dimensions using `out_dim`.
        """
        return fftc(self._obj, dim=dim, out_dim=out_dim)

    def ifftc(
        self,
        dim: str | list[str] = "time",
        out_dim: str | list[str] | None = None,
    ) -> xr.DataArray:
        """
        Perform an N-dimensional centered Inverse FFT.

        This applies necessary shifts before and after the transform to ensure
        the DC component is centered. Optionally renames dimensions using `out_dim`.
        """
        return ifftc(self._obj, dim=dim, out_dim=out_dim)

    # --- Apodization ---

    def apodize_exp(self, dim: str = "time", lb: float = 1.0) -> xr.DataArray:
        """
        Multiply the time-domain signal by a decreasing mono-exponential filter.

        This improves the Signal-to-Noise Ratio (SNR) by applying a line
        broadening factor parameterized in Hz.

        Parameters
        ----------
        dim : str, optional
            The dimension corresponding to time, by default "time".
        lb : float, optional
            The desired line broadening factor in Hz, by default 1.0.

        Returns
        -------
        xr.DataArray
            A new apodized DataArray, preserving coordinates and attributes.
        """
        return apodize_exp(self._obj, dim=dim, lb=lb)

    def apodize_lg(
        self, dim: str = "time", lb: float = 1.0, gb: float = 1.0
    ) -> xr.DataArray:
        """
        Apply a Lorentzian-to-Gaussian transformation filter.

        This applies the filter to the time-domain signal for resolution
        enhancement, parameterized in Hz.

        Parameters
        ----------
        dim : str, optional
            The dimension corresponding to time, by default "time".
        lb : float, optional
            The Lorentzian line broadening to cancel in Hz, by default 1.0.
        gb : float, optional
            The Gaussian line broadening to apply in Hz, by default 1.0.

        Returns
        -------
        xr.DataArray
            A new apodized DataArray, preserving coordinates and attributes.
        """
        return apodize_lg(self._obj, dim=dim, lb=lb, gb=gb)

    # --- FID Specific Operations ---

    def to_spectrum(self, dim: str = "time", out_dim: str = "frequency") -> xr.DataArray:
        """
        Convert a time-domain FID to a frequency-domain spectrum.

        Applies a Fast Fourier Transform (FFT) and centers the zero-frequency component.

        Parameters
        ----------
        dim : str, optional
            The time dimension to transform, by default "time".
        out_dim : str, optional
            The name of the resulting frequency dimension, by default "frequency".

        Returns
        -------
        xr.DataArray
            The frequency-domain spectrum.
        """
        return to_spectrum(self._obj, dim=dim, out_dim=out_dim)

    def to_fid(self, dim: str = "frequency", out_dim: str = "time") -> xr.DataArray:
        """
        Convert a frequency-domain spectrum to a time-domain FID.

        Applies an inverse shift and Inverse Fast Fourier Transform (IFFT).

        Parameters
        ----------
        dim : str, optional
            The frequency dimension to transform, by default "frequency".
        out_dim : str, optional
            The name of the resulting time dimension, by default "time".

        Returns
        -------
        xr.DataArray
            The time-domain FID data.
        """
        return to_fid(self._obj, dim=dim, out_dim=out_dim)

    def zero_fill(
        self,
        dim: str = "time",
        target_points: int = 1024,
        position: str = "end",
    ) -> xr.DataArray:
        """
        Pad the specified dimension with zero amplitude points.

        Artificially extend the data with zeros and increase digital resolution.

        Parameters
        ----------
        dim : str, optional
            The dimension along which to pad zeros, by default "time".
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
        return zero_fill(
            self._obj, dim=dim, target_points=target_points, position=position
        )

    # --- Phase Correction ---

    def phase(self, p0: float = 0.0, p1: float = 0.0) -> xr.DataArray:
        """
        Apply zero- and first-order phase correction to the spectrum.

        Parameters
        ----------
        p0 : float, optional
            Zero-order phase angle in degrees, by default 0.0.
        p1 : float, optional
            First-order phase angle in degrees, by default 0.0.

        Returns
        -------
        xr.DataArray
            The phase-corrected spectrum with p0 and p1 stored in the attributes.
        """
        return phase(self._obj, p0=p0, p1=p1)

    def autophase(
        self, dim: str = "frequency", lb: float = 5.0, temp_time_dim: str = "time"
    ) -> xr.DataArray:
        """
        Automatically calculate and apply phase correction to a spectrum.

        Uses a hidden "sacrificial apodization" step to improve SNR temporarily
        for the optimizer, calculating the correct phase angles, and applying
        them to the raw, input spectrum.

        Parameters
        ----------
        dim : str, optional
            The frequency dimension, by default "frequency".
        lb : float, optional
            The line broadening (in Hz) used for the sacrificial apodization.
            Higher values suppress more noise. By default 10.0.
        temp_time_dim : str, optional
            The name used for the temporary time dimension during the inverse
            transform. By default "time".

        Returns
        -------
        xr.DataArray
            The phased high-resolution spectrum. Phase angles are stored in
            `DataArray.attrs['p0']` and `DataArray.attrs['p1']`.
        """
        return autophase(self._obj, dim=dim, lb=lb, temp_time_dim=temp_time_dim)

    # --- Fitting ---

    def fit_amares(
        self,
        prior_knowledge_file: str | Path,
        dim: str = "time",
        mhz: float | None = None,
        sw: float | None = None,
        deadtime: float | None = None,
        method: str = "leastsq",
        initialize_with_lm: bool = True,
        num_workers: int = 4,
        init_fid: np.ndarray | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Apply AMARES time-domain fitting to an N-dimensional FID.

        This method wraps `pyAMARES` to perform parallelized batch fitting
        across spatial or repetition dimensions. The numerical results and
        the reconstructed time-domain fits are packed into an aligned xarray Dataset.

        Requires the optional `pyAMARES` package to be installed.

        Parameters
        ----------
        prior_knowledge_file : str | Path
            Path to the CSV or XLSX file containing the prior knowledge constraints.
        dim : str, optional
            The time dimension along which to fit, by default "time".
        mhz : float, optional
            Spectrometer frequency in MHz. If None, attempts to read from attrs['MHz'].
        sw : float, optional
            Spectral width in Hz. If None, attempts to calculate from `dim` coordinates.
        deadtime : float, optional
            Time delay before the first point in seconds. If None, defaults to 0.0.
        method : {"leastsq", "least_squares"}, optional
            Fitting method. Defaults to 'leastsq' (Levenberg-Marquardt).
        initialize_with_lm : bool, optional
            Run an internal Levenberg-Marquardt initializer before fitting.
            Defaults to True.
        num_workers : int, optional
            Number of parallel processes to spawn. Defaults to 4.
        init_fid : np.ndarray, optional
            A 1D complex array to use as the template for pyAMARES initialization.
            If None, the function automatically selects the spectrum with the highest SNR.

        Returns
        -------
        xr.Dataset
            A dataset containing the original data, the fitted FIDs, the residuals,
            and the quantified parameters (amplitude, chem_shift, linewidth, phase,
            CRLB, SNR) mapped across the original dimensions and the new 'Metabolite'
            dimension.

        Raises
        ------
        ImportError
            If the `pyAMARES` package is not installed.
        """
        try:
            from xmris.fitting.amares import fit_amares as _internal_fit_amares
        except ImportError as e:
            raise ImportError(
                "The '.fit_amares()' method requires the optional 'pyAMARES' package. "
                "Please install it using 'pip install pyAMARES' or 'uv add pyAMARES'."
            ) from e

        return _internal_fit_amares(
            self._obj,
            prior_knowledge_file=prior_knowledge_file,
            dim=dim,
            mhz=mhz,
            sw=sw,
            deadtime=deadtime,
            method=method,
            initialize_with_lm=initialize_with_lm,
            num_workers=num_workers,
            init_fid=init_fid,
            **kwargs,
        )

    # --- Vendor Specific ---

    def remove_digital_filter(
        self, group_delay: float, dim: str = "time", keep_length: bool = True
    ) -> xr.DataArray:
        """
        Remove the hardware digital filter group delay from Bruker FID data.

        Bruker consoles use a cascade of digital FIR filters during analog-to-digital
        conversion. Because these filters calculate a moving average, they require time
        to "wake up", introducing a causality delay at the start of the Free Induction
        Decay (FID). This manifests as a time-shift, effectively prepending the actual
        signal with a specific number of filter transient points.

        Parameters
        ----------
        group_delay : float
            The exact delay value to remove. This should be read directly from the
            Bruker `ACQ_RxFilterInfo` parameter array.
        dim : str, optional
            The time dimension along which to apply the correction, by default "time".
        keep_length : bool, optional
            If True, appends pure zeros to the end of the FID to replace the truncated
            startup points, maintaining the original length. By default True.

        Returns
        -------
        xr.DataArray
            The corrected FID data with the filter transient stripped and phase aligned.
        """
        return remove_digital_filter(
            self._obj, group_delay=group_delay, dim=dim, keep_length=keep_length
        )

    # --- Coordinate Math ---

    @requires_attrs(ATTRS.b0_field, ATTRS.reference_frequency)
    def to_ppm(self, dim: str = DIMS.frequency) -> xr.DataArray:
        """
        Convert the frequency axis coordinates from Hz to parts-per-million (ppm).

        This relies on the spectrometer reference frequency and the static B0 field.
        """
        _check_dims(self._obj, dim, "to_ppm")

        # We can safely access this without KeyError fears thanks to the bouncer!
        mhz = self._obj.attrs[ATTRS.reference_frequency]
        hz_coords = self._obj.coords[dim].values

        # Calculate ppm: (Hz / MHz)
        ppm_coords = hz_coords / mhz

        # Assign new coordinate using our config standard
        return self._obj.assign_coords({COORDS.ppm: (dim, ppm_coords)})

    # --- Utility / Formatting ---

    def to_real_imag(
        self, dim: str = DIMS.component, coords: tuple[str, str] = ("real", "imag")
    ) -> xr.DataArray:
        """Split a complex array into a real-valued array with an extra component dimension."""  # noqa: E501
        # Kept the deferred import to match your original package load-time strategy
        from xmris.processing.utils import to_real_imag as _to_real_imag

        return _to_real_imag(self._obj, dim=dim, coords=coords)

    def to_complex(
        self, dim: str = DIMS.component, coords: tuple[str, str] = ("real", "imag")
    ) -> xr.DataArray:
        """Reconstruct a real-valued split array back into a standard complex array."""
        from xmris.processing.utils import to_complex as _to_complex

        return _to_complex(self._obj, dim=dim, coords=coords)
