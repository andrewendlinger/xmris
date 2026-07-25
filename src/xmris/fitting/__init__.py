"""Modeling and simulation.

``simulate_fid`` is dependency-light and always importable. ``fit_amares`` needs
the optional ``fitting`` extra (pyAMARES); it is resolved lazily (PEP 562) so a
bare ``import xmris`` succeeds even when pyAMARES is not installed.
"""

import importlib.util
import sys
from typing import TYPE_CHECKING

from .prior_knowledge import build_prior_knowledge
from .simulation import simulate_fid

if TYPE_CHECKING:
    from .amares import fit_amares

MISSING_FITTING_DEP_MSG = (
    "AMARES fitting requires the optional 'fitting' extra (pyAMARES). "
    "Install it with `pip install 'xmris[fitting]'` or `uv add 'xmris[fitting]'`."
)


def _pyamares_installed() -> bool:
    """Report whether pyAMARES is importable, without importing it.

    Keeps ``from xmris.fitting import *`` (and the top-level star-import) working on
    a fitting-free install, and lets a genuinely-absent pyAMARES be told apart from
    a present-but-broken one.
    """
    if "pyAMARES" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("pyAMARES") is not None
    except ImportError:
        return False


# ``fit_amares`` is only a real runtime attribute when pyAMARES is installed; drop
# it from the eagerly-iterated ``__all__`` otherwise so a star-import does not force
# the lazy resolver and raise. Explicit access still resolves via __getattr__.
__all__ = ["build_prior_knowledge", "fit_amares", "simulate_fid"]
if not _pyamares_installed():
    __all__.remove("fit_amares")


def __getattr__(name: str):
    """Resolve the optional ``fit_amares`` export lazily (PEP 562)."""
    if name == "fit_amares":
        try:
            from .amares import fit_amares
        except ImportError as exc:
            # Only re-map the genuine "pyAMARES absent" case to the friendly install
            # hint; let a real failure inside a present-but-broken pyAMARES (renamed
            # symbol, numpy>=2 incompat) surface its true cause instead of a message
            # that points at an already-installed package.
            if not _pyamares_installed():
                raise ImportError(MISSING_FITTING_DEP_MSG) from exc
            raise
        return fit_amares
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
