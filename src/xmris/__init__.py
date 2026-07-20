from typing import TYPE_CHECKING

# =============================================================================
# 0. Submodules (Required to expose the namespace for quartodoc / Griffe)
# =============================================================================
from . import config, core, fitting, processing, vendor, visualization

# =============================================================================
# 1. Global Configuration & Singletons (The Central Nervous System)
# =============================================================================
from .core import (
    ATTRS,
    COORDS,
    DIMS,
    VARS,
)

# =============================================================================
# 2. Xarray Accessors (Importing these automatically registers the .xmr namespace)
# =============================================================================
from .core.accessor import XmrisAccessor, XmrisDatasetAccessor
from .core.options import set_options

# =============================================================================
# 4. Modeling & Fitting
# =============================================================================
# `fit_amares` needs the optional `fitting` extra (pyAMARES); it is exposed
# lazily via __getattr__ below so `import xmris` works without it. `simulate_fid`
# is dependency-light and stays eager.
from .fitting.simulation import simulate_fid
from .processing.baseline import baseline_als

if TYPE_CHECKING:
    from .fitting.amares import fit_amares

# =============================================================================
# 3. Core Signal Processing & Utilities
# =============================================================================
from .processing.fid import apodize_exp, apodize_lg, to_fid, to_spectrum, zero_fill
from .processing.fourier import fft, fftc, fftshift, ifft, ifftc, ifftshift
from .processing.phasing import autophase, phase
from .processing.referencing import to_hz, to_ppm
from .processing.utils import to_complex, to_real_imag

# =============================================================================
# 5. Vendor Integrations
# =============================================================================
from .vendor.bruker import remove_digital_filter

# =============================================================================
# 6. Visualization & Aesthetics
# =============================================================================
from .visualization.plot import (
    CarpetConfig,
    PlotQCGridConfig,
    PlotTrajectoryConfig,
    WaterfallConfig,
)

# =============================================================================
# Explicitly define the public API
# =============================================================================
__all__ = [
    # --- Submodules ---
    "core",
    "config",
    "fitting",
    "processing",
    "vendor",
    "visualization",
    # --- 1. Config & Singletons ---
    "ATTRS",
    "COORDS",
    "DIMS",
    "VARS",
    # --- 2. Accessors ---
    "XmrisAccessor",
    "XmrisDatasetAccessor",
    "set_options",
    # --- 3. Core Processing & Utilities ---
    "to_complex",
    "to_real_imag",
    "apodize_exp",
    "apodize_lg",
    "to_fid",
    "to_spectrum",
    "to_ppm",
    "to_hz",
    "zero_fill",
    "fft",
    "fftc",
    "fftshift",
    "ifft",
    "ifftc",
    "ifftshift",
    "autophase",
    "phase",
    "baseline_als",
    # --- 4. Fitting ---
    "fit_amares",
    "simulate_fid",
    # --- 5. Vendor ---
    "remove_digital_filter",
    # --- 6. Visualization Configs ---
    "WaterfallConfig",
    "CarpetConfig",
    "PlotTrajectoryConfig",
    "PlotQCGridConfig",
]


def __getattr__(name: str):
    """Resolve optional-dependency exports lazily (PEP 562).

    ``fit_amares`` lives behind the optional ``fitting`` extra (pyAMARES). Keeping
    it out of the eager imports lets ``import xmris`` succeed without pyAMARES,
    deferring the friendly ImportError to the moment fitting is actually used.
    """
    if name == "fit_amares":
        from .fitting import fit_amares

        return fit_amares
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
