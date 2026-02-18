"""
Xarray accessor for the xmris toolbox.

This module registers the `.xmr` namespace on xarray DataArrays.
"""

from typing import Union

import xarray as xr

from xmris.signal import fft, fftc, fftshift, ifft, ifftc, ifftshift


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

    def fftshift(self, dim: Union[str, list[str]]) -> xr.DataArray:
        """
        Apply fftshift by rolling data and coordinates along specified dimensions.

        Moves the zero-frequency component to the center of the spectrum.
        """
        return fftshift(self._obj, dim=dim)

    def ifftshift(self, dim: Union[str, list[str]]) -> xr.DataArray:
        """
        Apply ifftshift by rolling data and coordinates along specified dimensions.

        The inverse of :meth:`fftshift`.
        """
        return ifftshift(self._obj, dim=dim)

    # --- Pure Transforms ---

    def fft(
        self,
        dim: Union[str, list[str]] = "Time",
        out_dim: Union[str, list[str], None] = None,
    ) -> xr.DataArray:
        """
        Perform a standard N-dimensional FFT (no shifts).

        Optionally renames the transformed dimension(s) using `out_dim`.
        """
        return fft(self._obj, dim=dim, out_dim=out_dim)

    def ifft(
        self,
        dim: Union[str, list[str]] = "Time",
        out_dim: Union[str, list[str], None] = None,
    ) -> xr.DataArray:
        """
        Perform a standard N-dimensional Inverse FFT (no shifts).

        Optionally renames the transformed dimension(s) using `out_dim`.
        """
        return ifft(self._obj, dim=dim, out_dim=out_dim)

    # --- Centered Transforms ---

    def fftc(
        self,
        dim: Union[str, list[str]] = "Time",
        out_dim: Union[str, list[str], None] = None,
    ) -> xr.DataArray:
        """
        Perform an N-dimensional centered FFT.

        This applies necessary shifts before and after the transform to ensure
        the DC component is centered. Optionally renames dimensions using `out_dim`.
        """
        return fftc(self._obj, dim=dim, out_dim=out_dim)

    def ifftc(
        self,
        dim: Union[str, list[str]] = "Time",
        out_dim: Union[str, list[str], None] = None,
    ) -> xr.DataArray:
        """
        Perform an N-dimensional centered Inverse FFT.

        This applies necessary shifts before and after the transform to ensure
        the DC component is centered. Optionally renames dimensions using `out_dim`.
        """
        return ifftc(self._obj, dim=dim, out_dim=out_dim)
