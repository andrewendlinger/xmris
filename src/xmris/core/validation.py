"""Decorator engine for runtime validation, domain contracts, and docstring generation.

Two decorator tiers guard the processing functions:

- ``@requires_attrs(...)`` — *gate*: raise if required ``.attrs`` are missing.
- ``@ensures_domain(...)`` / ``@computes_in(...)`` — *domain*: establish the
  function's working domain (time vs. spectral), transforming the input through
  the standard converters when needed and resolving a ``dim=None`` argument.
  ``ensures_domain`` implements the *funnel* contract (the result is left in
  the working domain); ``computes_in`` implements the *domain-preserving*
  contract (the result is transformed back to the input's representation).

The two domain decorators share one private engine and differ only in whether
the coercion is inverted after the wrapped function runs.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import xarray as xr

from .config import ATTRS, DIMS, SPECTRAL_DIMS, TIME_DIMS
from .options import OPTIONS
from .utils import _domain_label, _resolve_dim


def _append_to_docstring(doc: str | None, title: str, keys: tuple[str, ...], vocab: Any) -> str:
    """Helper to cleanly append a new NumPy-style section to an existing docstring."""  # noqa: D401
    base_doc = doc or ""
    if base_doc and not base_doc.endswith("\n\n"):
        base_doc += "\n\n" if base_doc.endswith("\n") else "\n\n"

    lines = [f"    {title}", f"    {'-' * len(title)}"]
    for k in keys:
        desc = vocab.get_description(k)
        lines.append(f"    * ``{k}``: {desc}")

    return base_doc + "\n".join(lines) + "\n"


def _append_note_to_docstring(doc: str | None, title: str, body: str) -> str:
    """Append a free-text NumPy-style section (e.g. ``Notes``) to a docstring."""
    base_doc = doc or ""
    if base_doc and not base_doc.endswith("\n\n"):
        base_doc += "\n" if base_doc.endswith("\n") else "\n\n"

    lines = [f"    {title}", f"    {'-' * len(title)}", f"    {body}"]
    return base_doc + "\n".join(lines) + "\n"


def requires_attrs(*keys: str) -> Callable:
    """Decorator to enforce that specific attributes exist in the input's ``.attrs``.

    The *gate* tier of the validation taxonomy. The wrapped callable's first
    positional argument must be the ``xarray.DataArray`` (free-function
    convention). If attributes are missing at runtime, it raises a clear
    ValueError with instructions on how to fix it using standard xarray
    methods. At import time, it dynamically appends the required attributes to
    the function's docstring.

    Parameters
    ----------
    *keys : str
        The attribute string keys required by the function (e.g., ATTRS.b0_field).
    """  # noqa: D401

    def decorator(func: Callable) -> Callable:
        # 1. Modify the docstring at import time
        func.__doc__ = _append_to_docstring(
            doc=func.__doc__, title="Required Attributes", keys=keys, vocab=ATTRS
        )

        sig = inspect.signature(func)
        first_param = next(iter(sig.parameters))

        # 2. Wrap the runtime execution
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            da = bound.arguments[first_param]
            missing = [k for k in keys if k not in da.attrs]
            if missing:
                raise ValueError(
                    f"'{func.__name__}' requires the following missing attributes "
                    f"in `obj.attrs`: {missing}.\n\n"
                    f"To fix this, assign them using standard xarray methods:\n"
                    f"    >>> obj = obj.assign_attrs({{{repr(missing[0])}: value}})"
                )
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


class _RestoreState(NamedTuple):
    """What a ``computes_in`` round trip must hand back after the function runs."""

    dim: str  # original domain dim name (e.g. "chemical_shift")
    coord: xr.Variable | None  # original coordinate, restored verbatim when possible
    size: int  # original length along `dim`


def _domain_of(da: xr.DataArray, domain: frozenset[str]) -> str | None:
    """Return the first dimension of ``da`` belonging to ``domain``, or ``None``."""
    return next((str(d) for d in da.dims if d in domain), None)


def _coerce_to_domain(
    da: xr.DataArray, domain: frozenset[str]
) -> tuple[xr.DataArray, _RestoreState]:
    """Transform ``da`` into the physical domain ``domain`` via the standard converters.

    Only called when ``da`` carries no dimension of ``domain``. Routes strictly
    through the converter functions (`to_spectrum`/`to_fid`, with the ppm leg
    handled by `to_hz`) so an auto-inserted transform is bit-identical to an
    explicit one. Returns the converted array plus the :class:`_RestoreState`
    describing the original representation. The converters are imported lazily
    to keep ``validation`` free of an import-time dependency on ``processing``.
    """
    from xmris.processing.fid import to_fid, to_spectrum

    if domain == SPECTRAL_DIMS:
        source = _domain_of(da, TIME_DIMS)
        if source is not None:
            state = _RestoreState(
                dim=source,
                coord=da.coords[source].variable if source in da.coords else None,
                size=da.sizes[source],
            )
            return to_spectrum(da, dim=source), state

    elif domain == TIME_DIMS:
        source = _domain_of(da, SPECTRAL_DIMS)
        if source is not None:
            if not np.iscomplexobj(da.values):
                raise ValueError(
                    f"Cannot transform real-valued spectral data (dim {source!r}) "
                    f"into the time domain: the imaginary component is gone (e.g. "
                    f"discarded by `baseline_als`), so no valid FID exists behind "
                    f"this spectrum.\n\n"
                    f"Apply time-domain operations before the step that discarded "
                    f"the imaginary part, or pass an explicit existing dimension "
                    f"to operate on."
                )
            state = _RestoreState(
                dim=source,
                coord=da.coords[source].variable if source in da.coords else None,
                size=da.sizes[source],
            )
            if source == DIMS.chemical_shift:
                # ppm leg: reference to Hz first so `to_fid` reconstructs a
                # physical dwell time (its math assumes Hz coordinate spacing).
                from xmris.processing.referencing import to_hz

                da = to_hz(da, dim=source)
                source = str(DIMS.frequency)
            return to_fid(da, dim=source), state

    raise ValueError(
        f"Cannot ensure domain {sorted(domain)}: found no convertible "
        f"time/spectral dimension in {list(da.dims)}."
    )


def _restore_domain(
    result: xr.DataArray, domain: frozenset[str], state: _RestoreState
) -> xr.DataArray:
    """Invert a ``computes_in`` coercion, returning ``result`` in its original representation.

    Routes back through the standard converters; when the wrapped function
    preserved the length along the working dimension, the original coordinate
    variable is reassigned verbatim (protecting e.g. a dead-time offset that
    `to_fid`'s synthesized ``[0, T)`` axis would drop). Length-changing
    operations keep the converter-recomputed coordinates instead.
    """
    from xmris.processing.fid import to_fid, to_spectrum

    working = _domain_of(result, domain)
    if working is None:
        raise ValueError(
            f"Cannot restore the input representation: the wrapped function "
            f"returned no {_domain_label(domain)} dimension "
            f"(got {list(result.dims)}). Functions under `computes_in` must "
            f"keep their working dimension."
        )

    if domain == TIME_DIMS:
        out = to_spectrum(result, dim=working)
        if state.dim == DIMS.chemical_shift:
            # The inbound ppm leg went through `to_hz`, so the attrs needed by
            # `to_ppm` are guaranteed present on the result.
            from xmris.processing.referencing import to_ppm

            out = to_ppm(out)
    else:
        out = to_fid(result, dim=working)

    if state.coord is not None and out.sizes.get(state.dim) == state.size:
        out = out.assign_coords({state.dim: state.coord})
    return out


def _domain_decorator(domain: frozenset[str], *, restore: bool) -> Callable:
    """Shared engine behind ``ensures_domain`` (funnel) and ``computes_in`` (restore)."""
    label = _domain_label(domain)
    if restore:
        note = (
            f"The computation runs in the {label} domain: input in another "
            f"domain is transformed in via the standard converters and the "
            f"result is transformed back, preserving the input's representation "
            f"and (where the length is unchanged) its exact coordinates. An "
            f"explicitly requested ``dim`` outside the {label} domain passes "
            f"through untouched."
        )
    else:
        note = (
            f"If the input is not already in the {label} domain it is "
            f"transformed into it via the standard converters and the result "
            f"is left there (funnel contract). A ``dim`` left as ``None`` is "
            f"resolved to the unique {label} dimension present."
        )

    def decorator(func: Callable) -> Callable:
        func.__doc__ = _append_note_to_docstring(func.__doc__, "Notes", note)

        sig = inspect.signature(func)
        first_param = next(iter(sig.parameters))
        has_dim = "dim" in sig.parameters

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            da = bound.arguments[first_param]
            requested = bound.arguments.get("dim") if has_dim else None

            state = None
            if _domain_of(da, domain) is None:
                # Domain-preserving ops honor an explicit request for a foreign
                # axis (e.g. zero_fill(dim="kx")) by not converting at all.
                foreign_request = restore and requested is not None and requested not in domain
                if not foreign_request:
                    if not OPTIONS["auto_convert"]:
                        converter = "to_spectrum" if domain == SPECTRAL_DIMS else "to_fid"
                        raise ValueError(
                            f"'{func.__name__}' requires a {label} dimension, but "
                            f"none of {sorted(domain)} are present in "
                            f"{list(da.dims)}, and automatic conversion is "
                            f"disabled (xmris.set_options(auto_convert=False)).\n\n"
                            f"Convert explicitly first:\n"
                            f"    >>> obj = obj.xmr.{converter}()"
                        )
                    da, state = _coerce_to_domain(da, domain)
                    bound.arguments[first_param] = da

            if has_dim and bound.arguments.get("dim") is None:
                bound.arguments["dim"] = _resolve_dim(da, domain)

            result = func(*bound.args, **bound.kwargs)

            if restore and state is not None:
                result = _restore_domain(result, domain, state)
            return result

        wrapper.__xmris_domain__ = (domain, restore)  # type: ignore[attr-defined]
        return wrapper

    return decorator


def ensures_domain(domain: frozenset[str]) -> Callable:
    """Funnel contract: ensure the input is in a physical domain, leaving the result there.

    The *domain* tier of the validation taxonomy, for operations that are only
    meaningful in one domain and whose result is consumed there (e.g.
    ``autophase``, ``baseline_als``). Before the wrapped function runs, its
    DataArray (first positional argument) is transformed into ``domain`` via
    the standard converters — a no-op if it is already there — and the result
    is *left in that domain* (no round-trip restore). A ``dim`` argument left
    as ``None`` is resolved to the unique domain dimension present; an
    explicitly supplied ``dim`` is never overridden.

    Parameters
    ----------
    domain : frozenset of str
        The dimension names constituting the required domain — use
        ``SPECTRAL_DIMS`` or ``TIME_DIMS`` from :mod:`xmris.core.config`.

    See Also
    --------
    computes_in : The domain-preserving contract (round trip, representation restored).
    """
    return _domain_decorator(domain, restore=False)


def computes_in(domain: frozenset[str]) -> Callable:
    """Domain-preserving contract: compute in a physical domain, restore the representation.

    The *domain* tier of the validation taxonomy, for operations whose physics
    is identical seen from either domain (e.g. ``apodize_exp``, ``zero_fill``).
    Input already in ``domain`` is processed directly. Input in the sibling
    domain takes a round trip: it is transformed in via the standard
    converters, processed, and transformed back — so the output keeps the
    input's representation, with the original coordinates reassigned verbatim
    whenever the operation preserved the length. Real-valued spectral input is
    rejected (its inverse transform is undefined), and an explicitly requested
    ``dim`` outside ``domain`` passes through untouched.

    Parameters
    ----------
    domain : frozenset of str
        The dimension names constituting the working domain — use
        ``SPECTRAL_DIMS`` or ``TIME_DIMS`` from :mod:`xmris.core.config`.

    See Also
    --------
    ensures_domain : The funnel contract (result is left in the working domain).
    """
    return _domain_decorator(domain, restore=True)
