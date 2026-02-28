"""Decorator engine for runtime validation and dynamic docstring generation."""

import functools
from collections.abc import Callable
from typing import Any

from .config import ATTRS


def _append_to_docstring(
    doc: str | None, title: str, keys: tuple[str, ...], vocab: Any
) -> str:
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
