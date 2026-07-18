"""Global runtime options for xmris, mirroring :func:`xarray.set_options`."""

from collections.abc import Callable
from typing import Any

OPTIONS: dict[str, bool] = {
    "auto_convert": True,
}

# Per-option value validators, checked up front before any option is applied
# (mirrors the ``xarray.set_options`` pattern). Extend alongside ``OPTIONS``.
_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "auto_convert": lambda value: isinstance(value, bool),
}


class set_options:
    """Set global xmris options, either permanently or within a context.

    Parameters
    ----------
    auto_convert : bool, optional
        When ``True`` (the default), domain-decorated operations transform
        their input into the required physical domain automatically — the
        funnel (``@ensures_domain``) and domain-preserving (``@computes_in``)
        contracts. When ``False``, xmris runs *strict*: a domain mismatch
        raises an actionable error instead of converting, so every Fourier
        transform in a pipeline is written explicitly. Recommended for
        quantitative work.

    Examples
    --------
    Temporarily, as a context manager::

        with xmris.set_options(auto_convert=False):
            fid.xmr.autophase()   # raises: convert with .xmr.to_spectrum() first

    Or globally::

        xmris.set_options(auto_convert=False)
    """

    def __init__(self, **kwargs: bool):
        # Validate every key and value up front, then apply atomically — a raise
        # mid-call must never leave a partial change behind (the object is never
        # entered, so ``__exit__`` could not restore it). Mirrors xarray.set_options.
        for key, value in kwargs.items():
            if key not in OPTIONS:
                raise ValueError(
                    f"Unknown xmris option {key!r}. Available options: {sorted(OPTIONS)}"
                )
            validator = _VALIDATORS.get(key)
            if validator is not None and not validator(value):
                raise ValueError(
                    f"Invalid value {value!r} for xmris option {key!r}: expected a bool."
                )

        self.old: dict[str, bool] = {key: OPTIONS[key] for key in kwargs}
        OPTIONS.update(kwargs)

    def __enter__(self) -> "set_options":
        """Enter the context; options were already applied in ``__init__``."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Restore the option values that were active before this context."""
        OPTIONS.update(self.old)
