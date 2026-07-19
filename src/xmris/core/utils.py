# src/xmris/core/utils.py
from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr

from xmris.core.config import SPECTRAL_DIMS, TIME_DIMS, XmrisTerm


def _check_dims(da: xr.DataArray, dims: str | list[str], method_name: str) -> None:
    """Validate that required dimensions exist in the DataArray."""
    dims_to_check = [dims] if isinstance(dims, str) else dims
    missing = [d for d in dims_to_check if d not in da.dims]

    if missing:
        raise ValueError(
            f"Method '{method_name}' attempted to operate on missing "
            f"dimension(s): {missing}.\n"
            f"Available dimensions are: {list(da.dims)}.\n\n"
            f"To fix this, either pass the correct `dim` string argument to the function,"
            f" or rename your data's axes using xarray:\n"
            f"    >>> obj = obj.rename({{{repr(missing[0])}: 'correct_name'}})"
        )


def _domain_label(candidates: frozenset[str]) -> str:
    """Human-readable name of a domain dim-group, for error messages and docs."""
    if candidates == SPECTRAL_DIMS:
        return "spectral"
    if candidates == TIME_DIMS:
        return "time"
    return "target"


def _resolve_dim(da: xr.DataArray, candidates: frozenset[str]) -> str:
    """Identify the single dimension of ``da`` belonging to ``candidates``.

    Scans ``da.dims`` for a dimension in the given domain group (e.g.
    ``SPECTRAL_DIMS`` — ``frequency`` [Hz] or ``chemical_shift`` [ppm]). Used
    by the domain decorators in ``validation.py`` to fill a ``dim=None``
    argument, and by the visualization widgets to auto-detect the display axis.

    Parameters
    ----------
    da : xr.DataArray
        The data to inspect.
    candidates : frozenset of str
        The candidate dimension names constituting the domain — use
        ``SPECTRAL_DIMS`` or ``TIME_DIMS`` from :mod:`xmris.core.config`.

    Returns
    -------
    str
        The name of the unique matching dimension found.

    Raises
    ------
    ValueError
        If no candidate dimension is present, or if more than one is present
        (ambiguous — the caller must pass ``dim`` explicitly).
    """
    label = _domain_label(candidates)
    found = [d for d in da.dims if d in candidates]

    if not found:
        if candidates == SPECTRAL_DIMS:
            hint = (
                "If this is time-domain data, transform it first:\n"
                "    >>> obj = obj.xmr.to_spectrum()\n"
                "Otherwise pass the dimension explicitly, e.g. `dim='frequency'`."
            )
        else:
            hint = f"Pass the dimension explicitly, e.g. `dim={sorted(candidates)[0]!r}`."
        raise ValueError(
            f"Could not resolve a {label} dimension: none of "
            f"{sorted(candidates)} are present in {list(da.dims)}.\n\n{hint}"
        )

    if len(found) > 1:
        raise ValueError(
            f"Ambiguous {label} dimension: multiple {label} dims {found} are "
            f"present in {list(da.dims)}.\n\n"
            f"Pass the target dimension explicitly, e.g. `dim={found[0]!r}`."
        )

    return str(found[0])


def _spectral_axis_label(dim: str, coord: xr.DataArray) -> str:
    """Build an axis label from a coordinate's lineage metadata.

    Prefers the coordinate's ``long_name``/``units`` attrs (set by xmris when it
    builds spectral coordinates) over any hardcoded string, falling back to the
    bare dimension name when that metadata is absent. Shared by the visualization
    widgets so no widget hardcodes an axis label.

    Parameters
    ----------
    dim : str
        Name of the spectral dimension (used as the fallback label).
    coord : xr.DataArray
        The coordinate carrying the axis values and its ``.attrs`` metadata.

    Returns
    -------
    str
        A display label such as ``"Chemical Shift [ppm]"``.
    """
    long_name = coord.attrs.get("long_name")
    units = coord.attrs.get("units")
    if long_name and units:
        return f"{long_name} [{units}]"
    if long_name:
        return str(long_name)
    return str(dim)


def as_variable(term: XmrisTerm, dims: str | tuple, data: np.ndarray) -> xr.Variable:
    """Wrap a numpy array into an xarray Variable.

    Automatically apply the correct units and long_name from the provided XmrisTerm.
    """
    attrs = {"long_name": term.long_name}
    if term.unit:
        attrs["units"] = term.unit

    return xr.Variable(dims, data, attrs=attrs)


def read_attr(obj: xr.DataArray | xr.Dataset, term: XmrisTerm, default: Any = None) -> Any:
    """Read an attribute from ``obj.attrs``, falling back to the term's legacy aliases.

    Looks up the canonical key first, then each entry of ``term.aliases`` in
    declaration order. This is the single shared resolver for attribute keys
    that were renamed at some point (e.g. legacy ``"MHz"`` →
    ``"reference_frequency"``): the rename is declared once on the term in
    :mod:`xmris.core.config` and every reader inherits the fallback.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Object whose ``.attrs`` to read.
    term : XmrisTerm
        Vocabulary term naming the attribute (e.g. ``ATTRS.reference_frequency``).
    default : Any, optional
        Value returned when neither the canonical key nor any alias is present,
        by default None.

    Returns
    -------
    Any
        The stored attribute value, or ``default`` if absent.
    """
    for key in (term, *term.aliases):
        if key in obj.attrs:
            return obj.attrs[key]
    return default


def _first_matching_name(term: XmrisTerm, names: Iterable[str]) -> str | None:
    """Return the first of (canonical value, *aliases) present in ``names``, else None."""
    available = set(names)
    for key in (term, *term.aliases):
        if key in available:
            return str(key)
    return None
