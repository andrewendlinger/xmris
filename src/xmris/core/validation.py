"""Decorator engine for runtime validation and dynamic docstring generation."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

import xarray as xr

from .config import ATTRS
from .utils import _resolve_spectral_dim


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


def requires_attrs(*keys: str) -> Callable:
    """Decorator to enforce that specific attributes exist in `self._obj.attrs`.

    If attributes are missing at runtime, it raises a clear ValueError with
    instructions on how to fix it using standard xarray methods. At import time,
    it dynamically appends the required attributes to the method's docstring.

    Parameters
    ----------
    *keys : str
        The attribute string keys required by the method (e.g., ATTRS.b0_field).
    """  # noqa: D401

    def decorator(func: Callable) -> Callable:
        # 1. Modify the docstring at import time
        func.__doc__ = _append_to_docstring(
            doc=func.__doc__, title="Required Attributes", keys=keys, vocab=ATTRS
        )

        # 2. Wrap the runtime execution
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            missing = [k for k in keys if k not in self._obj.attrs]
            if missing:
                raise ValueError(
                    f"Method '{func.__name__}' requires the following missing attributes "
                    f"in `obj.attrs`: {missing}.\n\n"
                    f"To fix this, assign them using standard xarray methods:\n"
                    f"    >>> obj = obj.assign_attrs({{{repr(missing[0])}: value}})"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def _append_note_to_docstring(doc: str | None, title: str, body: str) -> str:
    """Append a free-text NumPy-style section (e.g. ``Notes``) to a docstring."""
    base_doc = doc or ""
    if base_doc and not base_doc.endswith("\n\n"):
        base_doc += "\n" if base_doc.endswith("\n") else "\n\n"

    lines = [f"    {title}", f"    {'-' * len(title)}", f"    {body}"]
    return base_doc + "\n".join(lines) + "\n"


def resolves_spectral_dim(func: Callable) -> Callable:
    """Fill a missing ``dim`` argument with the data's spectral dimension.

    The *resolve* tier of the validation taxonomy (``requires`` → ``resolves``
    → ``ensures``). If the caller leaves ``dim`` as ``None`` (or omits it), the
    DataArray (first positional argument) is introspected and the unique
    spectral dimension is injected via :func:`_resolve_spectral_dim`. An
    explicitly supplied ``dim`` is never overridden. Zero-cost — no data is
    transformed.
    """
    func.__doc__ = _append_note_to_docstring(
        func.__doc__,
        "Notes",
        "If ``dim`` is left as ``None`` it is resolved automatically to the "
        "spectral dimension present (``frequency`` or ``chemical_shift``).",
    )

    sig = inspect.signature(func)
    first_param = next(iter(sig.parameters))

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)
        if bound.arguments.get("dim") is None:
            bound.arguments["dim"] = _resolve_spectral_dim(bound.arguments[first_param])
        return func(*bound.args, **bound.kwargs)

    return wrapper


def _coerce_to_domain(da: xr.DataArray, target_dims: frozenset[str]) -> xr.DataArray:
    """Return ``da`` transformed into the physical domain named by ``target_dims``.

    A no-op if ``da`` already carries a dimension in ``target_dims``; otherwise a
    Fourier transform (time ↔ spectral) moves it into the target domain and the
    result is left there. The conversion routines are imported lazily to keep
    ``validation`` free of an import-time dependency on ``processing``.
    """
    if any(d in target_dims for d in da.dims):
        return da

    from xmris.core.config import SPECTRAL_DIMS, TIME_DIMS
    from xmris.processing.fid import to_fid, to_spectrum

    if target_dims == SPECTRAL_DIMS:
        source = next((str(d) for d in da.dims if d in TIME_DIMS), None)
        if source is not None:
            return to_spectrum(da, dim=source)
    elif target_dims == TIME_DIMS:
        source = next((str(d) for d in da.dims if d in SPECTRAL_DIMS), None)
        if source is not None:
            return to_fid(da, dim=source)

    raise ValueError(
        f"Cannot ensure domain {sorted(target_dims)}: found no convertible "
        f"time/spectral dimension in {list(da.dims)}."
    )


def ensures_domain(target_dims: frozenset[str]) -> Callable:
    """Ensure the input is in a target physical domain, transforming if needed.

    The *ensures* tier of the validation taxonomy (``requires`` → ``resolves``
    → ``ensures``). Before the wrapped function runs, its DataArray (first
    positional argument) is coerced into ``target_dims``: a no-op if it is
    already there, otherwise an FFT/IFFT (O(N log N)) moves it into the domain
    and the result is *left in that domain* (no round-trip restore).

    Parameters
    ----------
    target_dims : frozenset of str
        The dimension names constituting the required domain — use
        ``SPECTRAL_DIMS`` or ``TIME_DIMS`` from :mod:`xmris.core.config`.
    """

    def decorator(func: Callable) -> Callable:
        # NB: no docstring-note injection here. When stacked with
        # @resolves_spectral_dim (which does inject a Notes section) a second
        # injection would produce a duplicate "Notes" block; the auto-transform
        # behaviour is instead documented in each function's own parameter docs.
        sig = inspect.signature(func)
        first_param = next(iter(sig.parameters))

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.arguments[first_param] = _coerce_to_domain(
                bound.arguments[first_param], target_dims
            )
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
