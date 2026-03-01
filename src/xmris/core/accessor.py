"""
The primary xarray accessor namespace for the xmris package.

This module exposes the `.xmr` namespace to xarray DataArrays and Datasets.
It uses a "Hybrid Mixin" pattern: the user-facing API remains perfectly flat
for fluent method chaining (e.g., `da.xmr.apodize_exp().xmr.fft()`), while
the underlying developer API is strictly modularized into Mixin classes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Import our core architecture
from xmris.core.config import ATTRS, COORDS, DIMS
from xmris.core.utils import _check_dims, as_variable
from xmris.core.validation import requires_attrs

# Processing imports
from xmris.processing.fid import apodize_exp, apodize_lg, to_fid, to_spectrum, zero_fill
from xmris.processing.fourier import fft, fftc, fftshift, ifft, ifftc, ifftshift
from xmris.processing.phasing import autophase, phase

# =============================================================================
# Sub-Accessors (Terminal / Visualization tools)
# =============================================================================
from xmris.vendor.bruker import remove_digital_filter

# Deferred plot configs
from xmris.visualization.plot import PlotHeatmapConfig, PlotRidgeConfig


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
        from xmris.visualization.plot import plot_ridge as _plot_ridge

        return _plot_ridge(
            da=self._obj, x_dim=x_dim, stack_dim=stack_dim, ax=ax, config=config
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


class XmrisWidgetAccessor:
    """Sub-accessor for xmris interactive widget functionalities.

    This class provides a dedicated namespace for interactive UI components
    powered by AnyWidget. It is accessed via the `.xmr.widget` attribute
    on an xarray DataArray.
    """

    def __init__(self, obj: xr.DataArray):
        """
        Initialize the widget sub-accessor.

        Parameters
        ----------
        obj : xr.DataArray
            The underlying xarray DataArray object being operated on.
        """
        self._obj = obj

    def phase_spectrum(self):
        """
        Open an interactive zero- and first-order phase correction widget.

        This method launches an AnyWidget-based user interface directly in the
        Jupyter Notebook. It allows for manual, real-time adjustment of the
        zero-order ($p_0$) and first-order ($p_1$) phase angles of a 1-D
        complex-valued NMR/MRS spectrum.

        Returns
        -------
        NMRPhaseWidget
            The interactive widget instance. Assigning this to a variable allows
            you to programmatically extract the optimized phase angles after
            interacting with the UI.

        Raises
        ------
        ValueError
            If the underlying DataArray is not 1-dimensional or does not contain
            complex-valued data.

        Notes
        -----
        - **Zero-order phase ($p_0$)**: Click and drag vertically on the canvas.
        - **First-order phase ($p_1$)**: Hold `Shift`, then click and drag vertically.
        - The pivot point for $p_1$ is automatically set to the chemical shift (ppm)
          of the maximum magnitude peak.

        Examples
        --------
        >>> widget = da.xmr.widget.phase_spectrum()
        >>> display(widget)

        Once you have phased the spectrum by eye in the UI, you can extract
        the exact angles back into Python:

        >>> p0_opt = widget.p0
        >>> p1_opt = widget.p1
        """
        # Lazy import to avoid loading AnyWidget/frontend assets unless requested
        from xmris.visualization.widget import phase_spectrum

        # Return the widget instance so it renders and can be assigned
        return phase_spectrum(self._obj)


# =============================================================================
# Mixins (Developer API Modularity)
# =============================================================================


class XmrisSpectrumCoordsMixin:
    """Mixin providing operations to translate physical coordinate systems."""

    @requires_attrs(ATTRS.reference_frequency, ATTRS.carrier_ppm)
    def to_ppm(self, dim: str = DIMS.frequency) -> xr.DataArray:
        """Convert relative frequency axis [Hz] to absolute chemical shift axis [ppm]."""
        _check_dims(self._obj, dim, "to_ppm")

        mhz = self._obj.attrs[ATTRS.reference_frequency]
        carrier_ppm = self._obj.attrs[ATTRS.carrier_ppm]
        hz_coords = self._obj.coords[dim].values

        # 1. Calculate the math
        ppm_coords = carrier_ppm + (hz_coords / mhz)

        # 2. Build the fully-formed xarray Variable (data + metadata)
        shift_var = as_variable(DIMS.chemical_shift, dim, ppm_coords)

        # 3. Assign and swap in one clean sweep
        obj = self._obj.assign_coords({DIMS.chemical_shift: shift_var})
        return obj.swap_dims({dim: DIMS.chemical_shift})

    @requires_attrs(ATTRS.reference_frequency, ATTRS.carrier_ppm)
    def to_hz(self, dim: str = DIMS.chemical_shift) -> xr.DataArray:
        """Convert absolute chemical shift axis [ppm] to relative frequency axis [Hz]."""
        _check_dims(self._obj, dim, "to_hz")

        mhz = self._obj.attrs[ATTRS.reference_frequency]
        carrier_ppm = self._obj.attrs[ATTRS.carrier_ppm]
        ppm_coords = self._obj.coords[dim].values

        hz_coords = (ppm_coords - carrier_ppm) * mhz

        # Pack the data and metadata together instantly
        freq_var = as_variable(COORDS.frequency, dim, hz_coords)

        obj = self._obj.assign_coords({COORDS.frequency: freq_var})
        return obj.swap_dims({dim: DIMS.frequency})


class XmrisFourierMixin:
    """Mixin providing generalized N-dimensional Fourier transforms and shifts."""

    def fftshift(self, dim: str | list[str]) -> xr.DataArray:
        """
        Apply fftshift by rolling data and coordinates along specified dimensions.

        Moves the zero-frequency component to the center of the spectrum.
        """
        return fftshift(self._obj, dim=dim)

    def ifftshift(self, dim: str | list[str]) -> xr.DataArray:
        """
        Apply ifftshift by rolling data and coordinates along specified dimensions.

        The exact inverse of :meth:`fftshift`.
        """
        return ifftshift(self._obj, dim=dim)

    def fft(
        self,
        dim: str | list[str] = DIMS.time,
        out_dim: str | list[str] | None = None,
    ) -> xr.DataArray:
        """
        Perform a standard N-dimensional Fast Fourier Transform (no shifts).

        Parameters
        ----------
        dim : str or list of str, optional
            Dimension(s) to transform, by default `DIMS.time`.
        out_dim : str or list of str, optional
            Optional new dimension name(s), by default None.

        Returns
        -------
        xr.DataArray
            The transformed DataArray.
        """
        return fft(self._obj, dim=dim, out_dim=out_dim)

    def ifft(
        self,
        dim: str | list[str] = DIMS.frequency,
        out_dim: str | list[str] | None = None,
    ) -> xr.DataArray:
        """
        Perform a standard N-dimensional Inverse FFT (no shifts).

        Parameters
        ----------
        dim : str or list of str, optional
            Dimension(s) to transform, by default `DIMS.frequency`.
        out_dim : str or list of str, optional
            Optional new dimension name(s), by default None.

        Returns
        -------
        xr.DataArray
            The transformed DataArray.
        """
        return ifft(self._obj, dim=dim, out_dim=out_dim)

    def fftc(
        self,
        dim: str | list[str] = DIMS.time,
        out_dim: str | list[str] | None = None,
    ) -> xr.DataArray:
        """Perform a centered N-dimensional FFT (ifftshift -> fft -> fftshift)."""
        return fftc(self._obj, dim=dim, out_dim=out_dim)

    def ifftc(
        self,
        dim: str | list[str] = DIMS.frequency,
        out_dim: str | list[str] | None = None,
    ) -> xr.DataArray:
        """Perform a centered N-dimensional Inverse FFT (ifftshift -> ifft -> fftshift)."""  # noqa: E501
        return ifftc(self._obj, dim=dim, out_dim=out_dim)


class XmrisProcessingMixin:
    """Mixin providing common NMR/MRI Free Induction Decay processing tools."""

    def apodize_exp(self, dim: str = DIMS.time, lb: float = 1.0) -> xr.DataArray:
        """
        Multiply the time-domain signal by a decreasing mono-exponential filter.

        Parameters
        ----------
        dim : str, optional
            The dimension corresponding to time, by default `DIMS.time`.
        lb : float, optional
            The desired line broadening factor in Hz, by default 1.0.

        Returns
        -------
        xr.DataArray
            A new apodized DataArray, preserving coordinates and attributes.
        """
        return apodize_exp(self._obj, dim=dim, lb=lb)

    def apodize_lg(
        self, dim: str = DIMS.time, lb: float = 1.0, gb: float = 1.0
    ) -> xr.DataArray:
        """
        Apply a Lorentzian-to-Gaussian transformation filter.

        Parameters
        ----------
        dim : str, optional
            The dimension corresponding to time, by default `DIMS.time`.
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

    def to_spectrum(
        self, dim: str = DIMS.time, out_dim: str = DIMS.frequency
    ) -> xr.DataArray:
        """
        Convert a time-domain FID to a frequency-domain spectrum.

        Parameters
        ----------
        dim : str, optional
            The time dimension to transform, by default `DIMS.time`.
        out_dim : str, optional
            The name of the resulting frequency dimension, by default `DIMS.frequency`.

        Returns
        -------
        xr.DataArray
            The centered frequency-domain spectrum.
        """
        return to_spectrum(self._obj, dim=dim, out_dim=out_dim)

    def to_fid(self, dim: str = DIMS.frequency, out_dim: str = DIMS.time) -> xr.DataArray:
        """
        Convert a frequency-domain spectrum to a time-domain FID.

        Parameters
        ----------
        dim : str, optional
            The frequency dimension to transform, by default `DIMS.frequency`.
        out_dim : str, optional
            The name of the resulting time dimension, by default `DIMS.time`.

        Returns
        -------
        xr.DataArray
            The un-shifted time-domain FID data.
        """
        return to_fid(self._obj, dim=dim, out_dim=out_dim)

    def zero_fill(
        self,
        dim: str = DIMS.time,
        target_points: int = 1024,
        position: str = "end",
    ) -> xr.DataArray:
        """
        Pad the specified dimension with zero amplitude points.

        Parameters
        ----------
        dim : str, optional
            The dimension along which to pad zeros, by default `DIMS.time`.
        target_points : int, optional
            The total number of points desired after padding, by default 1024.
        position : {"end", "symmetric"}, optional
            Where to apply the zeros. Use "end" for time-domain FIDs, and
            "symmetric" for spatial frequency domains like k-space. By default "end".

        Returns
        -------
        xr.DataArray
            A new DataArray padded with zeros to the target length.
        """
        return zero_fill(
            self._obj, dim=dim, target_points=target_points, position=position
        )


class XmrisPhasingMixin:
    """Mixin providing common MR spectra phasing tools."""

    def phase(
        self, dim: str = DIMS.frequency, p0: float = 0.0, p1: float = 0.0
    ) -> xr.DataArray:
        """
        Apply zero- and first-order phase correction to the spectrum.

        Parameters
        ----------
        dim : str, optional
            The frequency dimension along which to apply phase correction,
            by default `DIMS.frequency`.
        p0 : float, optional
            Zero-order phase angle in degrees, by default 0.0.
        p1 : float, optional
            First-order phase angle in degrees, by default 0.0.

        Returns
        -------
        xr.DataArray
            The phase-corrected spectrum with phase_p0 and phase_p1 stored
            in the attributes.
        """
        return phase(self._obj, dim=dim, p0=p0, p1=p1)

    def autophase(
        self, dim: str = DIMS.frequency, lb: float = 10.0, temp_time_dim: str = DIMS.time
    ) -> xr.DataArray:
        """
        Automatically calculate and apply phase correction to a spectrum.

        Uses a hidden "sacrificial apodization" step to improve SNR temporarily
        for the optimizer, calculating the correct phase angles, and applying
        them to the raw, input spectrum.

        Parameters
        ----------
        dim : str, optional
            The frequency dimension, by default `DIMS.frequency`.
        lb : float, optional
            The line broadening (in Hz) used for the sacrificial apodization.
            Higher values suppress more noise. By default 10.0.
        temp_time_dim : str, optional
            The name used for the temporary time dimension during the inverse
            transform. By default `DIMS.time`.

        Returns
        -------
        xr.DataArray
            The phased high-resolution spectrum. Phase angles are stored in
            `DataArray.attrs['phase_p0']` and `DataArray.attrs['phase_p1']`.
        """
        return autophase(self._obj, dim=dim, lb=lb, temp_time_dim=temp_time_dim)


# =============================================================================
# Main User API Registration
# =============================================================================


@xr.register_dataset_accessor("xmr")
class XmrisDatasetAccessor:
    """Accessor for xmris xr.Datasets (e.g., fitting results)."""

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._plot = None

    @property
    def plot(self) -> XmrisDatasetPlotAccessor:
        """Access xmris plotting functionalities."""
        if self._plot is None:
            self._plot = XmrisDatasetPlotAccessor(self._obj)
        return self._plot


@xr.register_dataarray_accessor("xmr")
class XmrisAccessor(
    XmrisSpectrumCoordsMixin, XmrisFourierMixin, XmrisProcessingMixin, XmrisPhasingMixin
):
    """
    Main Accessor for xarray DataArrays to perform MRI and MRS operations.

    This class is registered under the `.xmr` namespace. It inherits from
    several domain-specific Mixins to provide a fluent, method-chaining API
    (e.g., `da.xmr.apodize_exp().xmr.to_spectrum().xmr.to_ppm()`) without
    creating an unmanageable monolithic class.

    Attributes
    ----------
    _obj : xr.DataArray
        The underlying xarray DataArray object being operated on.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        """Initialize the accessor with the xarray object."""
        self._obj = xarray_obj
        self._plot = None  # Cache for the plot sub-accessor
        self._widget = None  # Cache for the widget sub-accessor

    @property
    def plot(self) -> XmrisPlotAccessor:
        """Access xmris plotting functionalities for DataArrays."""
        if self._plot is None:
            self._plot = XmrisPlotAccessor(self._obj)
        return self._plot

    @property
    def widget(self) -> XmrisWidgetAccessor:
        """Access xmris plotting functionalities for DataArrays."""
        if self._widget is None:
            self._widget = XmrisWidgetAccessor(self._obj)
        return self._widget

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
