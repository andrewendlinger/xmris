"""Spectral-axis referencing: conversion between relative [Hz] and absolute [ppm] axes.

These converters are pure coordinate swaps (no Fourier transform): they build the
sibling spectral coordinate from the ``reference_frequency`` / ``carrier_ppm``
metadata and swap the indexing dimension. Together with ``to_spectrum`` /
``to_fid`` they are the only functions in xmris that change a DataArray's
representation on purpose — the domain decorators route every automatic
conversion through them.
"""

import xarray as xr

from xmris.core.config import ATTRS, COORDS, DIMS
from xmris.core.utils import _check_dims, as_variable
from xmris.core.validation import requires_attrs


@requires_attrs(ATTRS.reference_frequency, ATTRS.carrier_ppm)
def to_ppm(da: xr.DataArray, dim: str = DIMS.frequency) -> xr.DataArray:
    """
    Convert a relative frequency axis [Hz] to an absolute chemical shift axis [ppm].

    Computes ``carrier_ppm + hz / reference_frequency`` for every point of the
    frequency coordinate, assigns the result as a new ``chemical_shift``
    coordinate, and swaps the indexing dimension to it. The original Hz
    coordinate is kept alongside, so both views remain available.

    Parameters
    ----------
    da : xr.DataArray
        The input frequency-domain data with a coordinate on ``dim``.
    dim : str, optional
        The relative frequency dimension to convert, by default `DIMS.frequency`.

    Returns
    -------
    xr.DataArray
        The same data indexed by an absolute ``chemical_shift`` [ppm] dimension.
    """
    _check_dims(da, dim, "to_ppm")

    mhz = da.attrs[ATTRS.reference_frequency]
    carrier_ppm = da.attrs[ATTRS.carrier_ppm]
    hz_coords = da.coords[dim].values

    # 1. Calculate the math
    ppm_coords = carrier_ppm + (hz_coords / mhz)

    # 2. Build the fully-formed xarray Variable (data + metadata). Use the
    #    COORDS term (carries unit="ppm") so the coordinate's lineage is
    #    complete — mirrors `to_hz` using COORDS.frequency (unit="Hz").
    shift_var = as_variable(COORDS.chemical_shift, dim, ppm_coords)

    # 3. Assign and swap in one clean sweep
    obj = da.assign_coords({DIMS.chemical_shift: shift_var})
    return obj.swap_dims({dim: DIMS.chemical_shift})


@requires_attrs(ATTRS.reference_frequency, ATTRS.carrier_ppm)
def to_hz(da: xr.DataArray, dim: str = DIMS.chemical_shift) -> xr.DataArray:
    """
    Convert an absolute chemical shift axis [ppm] to a relative frequency axis [Hz].

    Computes ``(ppm - carrier_ppm) * reference_frequency`` for every point of
    the chemical shift coordinate, assigns the result as a new ``frequency``
    coordinate, and swaps the indexing dimension to it.

    Parameters
    ----------
    da : xr.DataArray
        The input chemical-shift-domain data with a coordinate on ``dim``.
    dim : str, optional
        The chemical shift dimension to convert, by default `DIMS.chemical_shift`.

    Returns
    -------
    xr.DataArray
        The same data indexed by a relative ``frequency`` [Hz] dimension.
    """
    _check_dims(da, dim, "to_hz")

    mhz = da.attrs[ATTRS.reference_frequency]
    carrier_ppm = da.attrs[ATTRS.carrier_ppm]
    ppm_coords = da.coords[dim].values

    hz_coords = (ppm_coords - carrier_ppm) * mhz

    # Pack the data and metadata together instantly
    freq_var = as_variable(COORDS.frequency, dim, hz_coords)

    obj = da.assign_coords({COORDS.frequency: freq_var})
    return obj.swap_dims({dim: DIMS.frequency})
