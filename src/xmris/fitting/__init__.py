"""Modeling and simulation.

``simulate_fid`` is dependency-light and always importable. ``fit_amares`` needs
the optional ``fitting`` extra (pyAMARES); it is resolved lazily (PEP 562) so a
bare ``import xmris`` succeeds even when pyAMARES is not installed.
"""

from typing import TYPE_CHECKING

from .simulation import simulate_fid

if TYPE_CHECKING:
    from .amares import fit_amares

__all__ = ["fit_amares", "simulate_fid"]

MISSING_FITTING_DEP_MSG = (
    "AMARES fitting requires the optional 'fitting' extra (pyAMARES). "
    "Install it with `pip install 'xmris[fitting]'` or `uv add 'xmris[fitting]'`."
)


def __getattr__(name: str):
    """Resolve the optional ``fit_amares`` export lazily (PEP 562)."""
    if name == "fit_amares":
        try:
            from .amares import fit_amares
        except ImportError as exc:
            raise ImportError(MISSING_FITTING_DEP_MSG) from exc
        return fit_amares
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
