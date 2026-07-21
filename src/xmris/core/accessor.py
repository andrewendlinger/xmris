"""
The primary xarray accessor namespace for the xmris package.

This module exposes the `.xmr` namespace to xarray DataArrays and Datasets.
It uses a "Hybrid Mixin" pattern: the user-facing API remains perfectly flat
for fluent method chaining (e.g., `da.xmr.apodize_exp().xmr.fft()`), while
the underlying developer API is strictly modularized into Mixin classes.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Import our core architecture
from xmris.core.config import DIMS
from xmris.core.utils import _check_dims, as_variable  # noqa: F401  (re-exported)
from xmris.processing.baseline import baseline_als

# Processing imports
from xmris.processing.fid import apodize_exp, apodize_lg, to_fid, to_spectrum, zero_fill
from xmris.processing.fourier import fft, fftc, fftshift, ifft, ifftc, ifftshift
from xmris.processing.phasing import autophase, phase
from xmris.processing.referencing import to_hz, to_ppm

# =============================================================================
# Sub-Accessors (Terminal / Visualization tools)
# =============================================================================
from xmris.vendor.bruker import estimate_group_delay, remove_digital_filter

# Deferred plot configs
from xmris.visualization.plot import CarpetConfig, WaterfallConfig


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

        return plot_trajectory(self._obj, dim=dim, metabolites=metabolites, ax=ax, config=config)

    def qc_grid(self, dim: str, config=None):
        """Plot a grid of spectra and fits to quickly visually inspect quality."""
        from xmris.visualization.plot.plot_qc_grid import plot_qc_grid

        return plot_qc_grid(self._obj, dim=dim, config=config)


class XmrisPlotAccessor:
    """Sub-accessor for xmris plotting functionalities (accessed via .xmr.plot)."""

    def __init__(self, obj: xr.DataArray):
        self._obj = obj

    def waterfall(
        self,
        x_dim: str | None = None,
        stack_dim: str | None = None,
        ax: plt.Axes | None = None,
        config: "WaterfallConfig | None" = None,
    ):
        """Generate a ridge plot (2D waterfall) of stacked 1D spectra."""
        from xmris.visualization.plot import plot_waterfall as _plot_waterfall

        return _plot_waterfall(
            da=self._obj,
            x_dim=x_dim,
            stack_dim=stack_dim,
            ax=ax,
            config=config,
        )

    def carpet(
        self,
        x_dim: str | None = None,
        stack_dim: str | None = None,
        ax: plt.Axes | None = None,
        config: "CarpetConfig | None" = None,
    ):
        """Generate a 2D carpet plot of stacked 1D spectra."""
        from xmris.visualization.plot import plot_carpet as _plot_carpet

        return _plot_carpet(
            da=self._obj,
            x_dim=x_dim,
            stack_dim=stack_dim,
            ax=ax,
            config=config,
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

    def phase_spectrum(
        self,
        dim: str | None = None,
        width: int = 740,
        height: int = 400,
        show_grid: bool = True,
        show_pivot: bool = True,
        **kwargs,
    ):
        """Open an interactive zero- and first-order phase correction widget.

        This method launches an AnyWidget-based user interface directly in the
        Jupyter Notebook. It allows for manual, real-time adjustment of the
        zero-order (p0) and first-order (p1) phase angles of a 1-D
        complex-valued NMR/MRS spectrum.

        Parameters
        ----------
        dim : str, optional
            Spectral dimension to plot along. If None (default), the canonical
            spectral dimension (frequency or chemical_shift) is auto-detected;
            pass it explicitly for non-standard axis names.
        width : int, optional
            Width of the widget in pixels. Default is 740.
        height : int, optional
            Height of the widget in pixels. Default is 400.
        show_grid : bool, optional
            Toggle the background grid visibility. Default is True.
        show_pivot : bool, optional
            Toggle the visibility of the p1 pivot indicator. Default is True.
        **kwargs
            Additional arguments passed to the underlying PhaseWidget.

        Returns
        -------
        PhaseWidget
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
        - **Zero-order phase (p0)**: Adjusts the phase uniformly across the spectrum.
        - **First-order phase (p1)**: Adjusts phase linearly relative to a pivot point.
        - The pivot point (p_pivot) is automatically set to the coordinate
          corresponding to the maximum magnitude peak.
        """
        # Lazy import to avoid loading AnyWidget/frontend assets unless requested
        from xmris.visualization.widget import phase_spectrum

        # The underlying function handles the 1-D, complex-type, and dim validation
        return phase_spectrum(
            self._obj,
            dim=dim,
            width=width,
            height=height,
            show_grid=show_grid,
            show_pivot=show_pivot,
            **kwargs,
        )

    def scroll_spectra(
        self,
        scroll_axis: str | None = None,
        dim: str | None = None,
        part: str = "real",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        show_trace: bool = True,
        trace_count: int = 10,
        width: int = 740,
        height: int = 400,
        **kwargs,
    ):
        """Open an interactive widget to scroll through a 2-D series of spectra.

        This method launches a user interface for exploring multi-dimensional
        spectroscopy data (e.g., transient repetitions, averages). It includes
        a timeline scrubber, animation playback, and fading historical traces.
        Clicking "Extract Slice" provides a copyable `.isel(...)` code snippet
        to isolate the current view while preserving pipeline lineage.

        Parameters
        ----------
        scroll_axis : str, optional
            The dimension to scroll through. If None, it is derived as the
            non-spectral dimension of the 2-D array.
        dim : str, optional
            The spectral (display) dimension. If None, the canonical spectral
            dimension (frequency or chemical_shift) is auto-detected; pass it
            explicitly for non-standard axis names.
        part : {'real', 'imag', 'abs'}, optional
            Which mathematical component of complex data to display. Default is 'real'.
        xlim : tuple of float, optional
            Static (min, max) bounds for the spectral axis.
        ylim : tuple of float, optional
            Static (min, max) bounds for intensity. If None, auto-ranges to the
            global minimum and maximum of the dataset.
        show_trace : bool, optional
            Show fading historical traces behind the current scan. Default is True.
        trace_count : int, optional
            The number of historical traces to overlay. Default is 10.
        width : int, optional
            Width of the widget in pixels. Default is 740.
        height : int, optional
            Height of the widget in pixels. Default is 400.
        **kwargs
            Additional arguments passed to the underlying ScrollWidget.

        Returns
        -------
        ScrollWidget
            The interactive widget instance.

        Raises
        ------
        ValueError
            If the input DataArray is not exactly 2-dimensional.
        """
        # Lazy import to avoid loading AnyWidget/frontend assets unless requested
        from xmris.visualization.widget import scroll_spectra

        return scroll_spectra(
            self._obj,
            scroll_axis=scroll_axis,
            dim=dim,
            part=part,
            xlim=xlim,
            ylim=ylim,
            show_trace=show_trace,
            trace_count=trace_count,
            width=width,
            height=height,
            **kwargs,
        )

    def apodize(
        self,
        dim: str | None = None,
        unit: str = "ppm",
        width: int = 740,
        height: int = 550,
        lb_range: tuple[float, float] = (0.0, 50.0),
        gb_range: tuple[float, float] = (0.0, 50.0),
        **kwargs,
    ):
        """Open an interactive widget for NMR/MRS spectrum apodization.

        This method launches an AnyWidget-based user interface to interactively
        apply and visualize line broadening (exponential) or resolution enhancement
        (Lorentz-to-Gauss) filters. It displays the modified time-domain FID alongside
        the resulting frequency-domain spectrum in real-time.

        Parameters
        ----------
        dim : str, optional
            The time dimension to apply the filter along. If None, it will be
            auto-detected by the underlying function.
        unit : {'ppm', 'hz'}, optional
            The unit for the spectral x-axis display. Default is 'ppm'.
        width : int, optional
            Width of the widget in pixels. Default is 740.
        height : int, optional
            Height of the widget in pixels. Default is 550.
        lb_range : tuple of float, optional
            The (min, max) range for the Line Broadening slider. Default is (0.0, 50.0).
        gb_range : tuple of float, optional
            The (min, max) range for the Gaussian Broadening slider. Default is (0.0, 50.0).
        **kwargs
            Additional arguments passed to the underlying ApodizerWidget.

        Returns
        -------
        ApodizerWidget
            The interactive widget instance. Closing the widget UI generates a
            copyable code snippet to apply the finalized filter parameters
            programmatically.

        Raises
        ------
        ValueError
            If the underlying DataArray is not 1-dimensional.

        Notes
        -----
        - **Exponential (exp)**: Multiplies the FID by an exponential decay,
          improving Signal-to-Noise Ratio (SNR) at the cost of line broadening (lb).
        - **Lorentz-Gauss (lg)**: Applies a combination of a rising exponential
          and a decaying Gaussian. It cancels natural Lorentzian broadening (lb)
          and imposes a Gaussian shape (gb) for resolution enhancement.
        """  # noqa: E501
        # Lazy import to avoid loading AnyWidget/frontend assets unless requested
        from xmris.visualization.widget import apodize

        return apodize(
            da=self._obj,
            dim=dim,
            unit=unit,
            width=width,
            height=height,
            lb_range=lb_range,
            gb_range=gb_range,
            **kwargs,
        )


# =============================================================================
# Mixins (Developer API Modularity)
# =============================================================================


class XmrisSpectrumCoordsMixin:
    """Mixin providing operations to translate physical coordinate systems."""

    def to_ppm(self, dim: str = DIMS.frequency) -> xr.DataArray:
        """Convert relative frequency axis [Hz] to absolute chemical shift axis [ppm]."""
        return to_ppm(self._obj, dim=dim)

    def to_hz(self, dim: str = DIMS.chemical_shift) -> xr.DataArray:
        """Convert absolute chemical shift axis [ppm] to relative frequency axis [Hz]."""
        return to_hz(self._obj, dim=dim)


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

    def apodize_lg(self, dim: str = DIMS.time, lb: float = 1.0, gb: float = 1.0) -> xr.DataArray:
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

    def to_spectrum(self, dim: str = DIMS.time, out_dim: str = DIMS.frequency) -> xr.DataArray:
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
        return zero_fill(self._obj, dim=dim, target_points=target_points, position=position)

    def baseline_als(
        self,
        dim: str | None = None,
        lam: float = 1e5,
        p: float = 0.001,
        n_iter: int = 10,
    ) -> xr.DataArray:
        r"""Apply Asymmetric Least Squares (AsLS) baseline correction to a spectrum.

        This method automatically estimates and subtracts a smooth baseline without
        requiring user-defined signal-free regions. It operates strictly on the
        real (absorption) component of the data.

        .. warning::
            **Real-Valued Output Only:** This function discards the imaginary
            (dispersion) component of the data. AsLS relies on the asymmetry of
            absorption-mode peaks and cannot be applied to complex data without
            breaking Kramers-Kronig relations. The resulting real-valued spectrum
            cannot be inverse-Fourier transformed back to the time domain.

        Parameters
        ----------
        dim : str or None, optional
            The spectral dimension along which to apply correction. If ``None``
            (default) it is resolved automatically to the spectral dim present
            (Hz or ppm); time-domain input is transformed to a spectrum first.
        lam : float, optional
            The smoothness penalty ($\lambda$). Higher values result in a stiffer,
            flatter baseline. Typical NMR ranges are 10,000 to 10,000,000.
            Defaults to 100,000.
        p : float, optional
            The asymmetry parameter. Controls how aggressively positive peaks are
            ignored during the fit. Typical ranges are 0.001 to 0.05.
            Defaults to 0.001.
        n_iter : int, optional
            Maximum number of iterations for the sparse solver. Defaults to 10.

        Returns
        -------
        xr.DataArray
            The strictly real-valued, baseline-corrected spectrum.
        """
        return baseline_als(self._obj, dim=dim, lam=lam, p=p, n_iter=n_iter)


class XmrisPhasingMixin:
    """Mixin providing common MR spectra phasing tools."""

    def phase(
        self,
        dim: str = DIMS.frequency,
        p0: float = 0.0,
        p1: float = 0.0,
        pivot: float = None,
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
        pivot : float, optional
            The coordinate value (e.g., ppm or Hz) around which p1 is pivoted.
            If None, standard nmrglue index-0 pivoting is used.

        Returns
        -------
        xr.DataArray
            The phase-corrected spectrum with phase_p0 and phase_p1 stored
            in the attributes.
        """
        return phase(self._obj, dim=dim, p0=p0, p1=p1, pivot=pivot)

    def autophase(
        self,
        dim: str | None = None,
        method: str = "acme",
        peak_width: float = 0.5,
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
        dim : str or None, optional
            The spectral dimension to operate on. If ``None`` (default) it is
            resolved automatically to the spectral dim present (Hz or ppm);
            time-domain input is transformed to a spectrum first.
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
        """  # noqa: E501
        return autophase(
            self._obj,
            dim=dim,
            method=method,
            peak_width=peak_width,
            lb=lb,
            temp_time_dim=temp_time_dim,
            **kwargs,
        )


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
        prior_knowledge: Mapping[str, Any] | pd.DataFrame | str | Path,
        dim: str = DIMS.time,
        mhz: float | None = None,
        sw: float | None = None,
        deadtime: float | None = None,
        carrier: float | None = None,
        g_global: float | bool = 0.0,
        method: str = "leastsq",
        initialize_with_lm: bool = False,
        num_workers: int = 4,
        init_fid: np.ndarray | None = None,
        verbose: bool = False,
    ) -> xr.Dataset:
        """
        Apply AMARES time-domain fitting to an N-dimensional FID.

        This method wraps `pyAMARES` to perform parallelized batch fitting
        across spatial or repetition dimensions. The numerical results and
        the reconstructed time-domain fits are packed into an aligned xarray Dataset.

        Requires the optional `pyAMARES` package to be installed.

        Parameters
        ----------
        prior_knowledge : Mapping | pandas.DataFrame | str | Path
            The prior-knowledge constraints, in memory or on disk. A mapping of peak
            name to parameters is built and validated via
            :func:`~xmris.fitting.build_prior_knowledge`; a ``str``/``Path`` is a
            pyAMARES CSV/XLSX file; a DataFrame in pyAMARES's positional layout is
            used as-is.
        dim : str, optional
            The time dimension along which to fit, by default ``DIMS.time``. A
            complex spectrum is accepted too and converted to a FID for the fit.
        mhz : float, optional
            Spectrometer frequency in MHz. If None, read from
            ``attrs['reference_frequency']``.
        sw : float, optional
            Spectral width in Hz. If None, calculated from the `dim` coordinate spacing.
        deadtime : float, optional
            Acquisition time origin in seconds. If None, taken from the first `dim`
            coordinate value.
        carrier : float, optional
            Transmitter carrier position on the absolute ppm scale, letting
            prior-knowledge and reported shifts be absolute/literature ppm. If None,
            taken from ``da.attrs['carrier_ppm']`` (default 0.0 = carrier-relative).
        g_global : float or bool, optional
            Global lineshape held for every peak: 0.0 = Lorentzian (default),
            1.0 = Gaussian, in between = pseudo-Voigt. Pass ``False`` to let each
            peak's ``g`` vary, fitted from the prior-knowledge value.
        method : {"leastsq", "least_squares"}, optional
            Fitting method. Defaults to 'leastsq' (Levenberg-Marquardt).
        initialize_with_lm : bool, optional
            Run an internal Levenberg-Marquardt initializer before fitting.
            Defaults to False.
        num_workers : int, optional
            Number of parallel processes to spawn. Defaults to 4.
        init_fid : np.ndarray, optional
            A 1D complex array to use as the template for pyAMARES initialization.
            If None, the function automatically selects the spectrum with the highest SNR.
        verbose : bool, optional
            If True, sets logging level to INFO and prints progress. Default is False.

        Returns
        -------
        xr.Dataset
            A dataset containing the original data, the fitted model, the residuals,
            and the quantified parameters (amplitude, chem_shift, linewidth, phase,
            CRLB, SNR) mapped across the original dimensions and the new ``metabolite``
            dimension. Time-domain variables are returned in the input's domain
            (ppm in -> ppm out).

        Raises
        ------
        ImportError
            If the `pyAMARES` package is not installed.
        """
        try:
            from xmris.fitting.amares import fit_amares as _internal_fit_amares
        except ImportError as e:
            from xmris.fitting import MISSING_FITTING_DEP_MSG

            raise ImportError(MISSING_FITTING_DEP_MSG) from e

        return _internal_fit_amares(
            self._obj,
            prior_knowledge=prior_knowledge,
            dim=dim,
            mhz=mhz,
            sw=sw,
            deadtime=deadtime,
            carrier=carrier,
            g_global=g_global,
            method=method,
            initialize_with_lm=initialize_with_lm,
            num_workers=num_workers,
            init_fid=init_fid,
            verbose=verbose,
        )

    # --- Vendor Specific ---

    def remove_digital_filter(
        self,
        group_delay: float | str = "header",
        dim: str = DIMS.time,
        keep_length: bool = True,
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
        group_delay : float or {"header", "measure"}, optional
            The delay (in samples) to remove. By default ``"header"``, which reads the
            vendor-reported value from ``.attrs`` (written by the Bruker loader). Pass
            a float to force a value, or ``"measure"`` to estimate it from the data via
            :meth:`estimate_group_delay` — robust when the header under-counts the true
            delay.
        dim : str, optional
            The time dimension along which to apply the correction, by default
            ``DIMS.time``.
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

    def estimate_group_delay(
        self,
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

        The vendor header value (Bruker ``ACQ_RxFilterInfo``/``GRPDLY``) can under-count
        the real receiver digital-filter group delay for some ParaVision/probe
        combinations, leaving a residual first-order phase error after
        :meth:`remove_digital_filter`. This finds the delay that removes that residual
        by locating the value that makes the spectrum maximally absorptive under a single
        global zero-order phase (``argmax(|FID|)`` is deliberately not used — it lands on
        the filter's ringing).

        Parameters
        ----------
        dim : str, optional
            The time dimension, by default ``DIMS.time``.
        search_range : tuple of float, optional
            Explicit ``(low, high)`` delay bounds (samples). If ``None`` (default), the
            window is anchored on the header: ``header ± window``.
        header_hint : float, optional
            Vendor-reported delay to anchor the search on. If ``None``, falls back to the
            stored ``group_delay`` attribute, then to a broad default range.
        window : float, optional
            Half-width (samples) of the header-anchored search window, by default 16.0.
        metric : {"acme", "coherence"}, optional
            Residual-phase cost, by default ``"acme"`` (whole-spectrum, alias-robust).
        refine : bool, optional
            If True (default), refine the best integer delay to sub-sample precision.
        return_profile : bool, optional
            If True, also return the cost-vs-delay profile for diagnosing multimodality.

        Returns
        -------
        float or tuple[float, xr.DataArray]
            The measured group delay in samples, or ``(delay, profile)`` when
            ``return_profile=True``.
        """
        return estimate_group_delay(
            self._obj,
            dim=dim,
            search_range=search_range,
            header_hint=header_hint,
            window=window,
            metric=metric,
            refine=refine,
            return_profile=return_profile,
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
