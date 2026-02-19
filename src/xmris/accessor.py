"""
Xarray accessor for the xmris toolbox.

This module registers the `.xmr` namespace on xarray DataArrays.
"""

import xarray as xr

from xmris.fid import apodize_exp, apodize_lg, to_fid, to_spectrum, zero_fill
from xmris.fourier import fft, fftc, fftshift, ifft, ifftc, ifftshift
from xmris.phase import autophase, phase


@xr.register_dataarray_accessor("xmr")
class XmrisAccessor:
    """
    Accessor for xarray DataArrays to perform MRI and MRS operations.

    This class is registered under the ``.xmr`` namespace. It provides a
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
        dim: str | list[str] = "Time",
        out_dim: str | list[str] | None = None,
    ) -> xr.DataArray:
        """
        Perform a standard N-dimensional FFT (no shifts).

        Optionally renames the transformed dimension(s) using `out_dim`.
        """
        return fft(self._obj, dim=dim, out_dim=out_dim)

    def ifft(
        self,
        dim: str | list[str] = "Time",
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
        dim: str | list[str] = "Time",
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
        dim: str | list[str] = "Time",
        out_dim: str | list[str] | None = None,
    ) -> xr.DataArray:
        """
        Perform an N-dimensional centered Inverse FFT.

        This applies necessary shifts before and after the transform to ensure
        the DC component is centered. Optionally renames dimensions using `out_dim`.
        """
        return ifftc(self._obj, dim=dim, out_dim=out_dim)

    # --- Apodization ---

    def apodize_exp(self, dim: str = "Time", lb: float = 1.0) -> xr.DataArray:
        """
        Multiply the time-domain signal by a decreasing mono-exponential filter.

        This improves the Signal-to-Noise Ratio (SNR) by applying a line
        broadening factor parameterized in Hz.

        Parameters
        ----------
        dim : str, optional
            The dimension corresponding to time, by default "Time".
        lb : float, optional
            The desired line broadening factor in Hz, by default 1.0.

        Returns
        -------
        xr.DataArray
            A new apodized DataArray, preserving coordinates and attributes.
        """
        return apodize_exp(self._obj, dim=dim, lb=lb)

    def apodize_lg(
        self, dim: str = "Time", lb: float = 1.0, gb: float = 1.0
    ) -> xr.DataArray:
        """
        Apply a Lorentzian-to-Gaussian transformation filter.

        This applies the filter to the time-domain signal for resolution
        enhancement, parameterized in Hz.

        Parameters
        ----------
        dim : str, optional
            The dimension corresponding to time, by default "Time".
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

    def to_spectrum(
        self, dim: str = "Time", out_dim: str = "Frequency"
    ) -> xr.DataArray:
        """
        Convert a time-domain FID to a frequency-domain spectrum.

        Applies a Fast Fourier Transform (FFT) and centers the zero-frequency component.

        Parameters
        ----------
        dim : str, optional
            The time dimension to transform, by default "Time".
        out_dim : str, optional
            The name of the resulting frequency dimension, by default "Frequency".

        Returns
        -------
        xr.DataArray
            The frequency-domain spectrum.
        """
        return to_spectrum(self._obj, dim=dim, out_dim=out_dim)

    def to_fid(self, dim: str = "Frequency", out_dim: str = "Time") -> xr.DataArray:
        """
        Convert a frequency-domain spectrum to a time-domain FID.

        Applies an inverse shift and Inverse Fast Fourier Transform (IFFT).

        Parameters
        ----------
        dim : str, optional
            The frequency dimension to transform, by default "Frequency".
        out_dim : str, optional
            The name of the resulting time dimension, by default "Time".

        Returns
        -------
        xr.DataArray
            The time-domain FID data.
        """
        return to_fid(self._obj, dim=dim, out_dim=out_dim)

    def zero_fill(
        self,
        dim: str = "Time",
        target_points: int = 1024,
        position: str = "end",
    ) -> xr.DataArray:
        """
        Pad the specified dimension with zero amplitude points.

        Artificially extend the data with zeros and increase digital resolution.

        Parameters
        ----------
        dim : str, optional
            The dimension along which to pad zeros, by default "Time".
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
        self, dim: str = "Frequency", lb: float = 5.0, temp_time_dim: str = "Time"
    ) -> xr.DataArray:
        """
        Automatically calculate and apply phase correction to a spectrum.

        Uses a hidden "sacrificial apodization" step to improve SNR temporarily
        for the optimizer, calculating the correct phase angles, and applying
        them to the raw, input spectrum.

        Parameters
        ----------
        dim : str, optional
            The frequency dimension, by default "Frequency".
        lb : float, optional
            The line broadening (in Hz) used for the sacrificial apodization.
            Higher values suppress more noise. By default 10.0.
        temp_time_dim : str, optional
            The name used for the temporary time dimension during the inverse
            transform. By default "Time".

        Returns
        -------
        xr.DataArray
            The phased high-resolution spectrum. Phase angles are stored in
            `DataArray.attrs['p0']` and `DataArray.attrs['p1']`.
        """
        return autophase(self._obj, dim=dim, lb=lb, temp_time_dim=temp_time_dim)
