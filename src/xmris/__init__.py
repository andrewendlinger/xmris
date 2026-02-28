# =============================================================================
# 0. Submodules (Required to expose the namespace for quartodoc / Griffe)
# =============================================================================
from . import config, core, fitting, processing, vendor, visualization

# =============================================================================
# 1. Global Configuration & Singletons (The Central Nervous System)
# =============================================================================
from .config import DEFAULTS  # Legacy patch (branch refactor/18)
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

# =============================================================================
# 4. Modeling & Fitting
# =============================================================================
from .fitting.amares import fit_amares

# =============================================================================
# 3. Core Signal Processing & Utilities
# =============================================================================
from .processing.fid import apodize_exp, apodize_lg, to_fid, to_spectrum, zero_fill
from .processing.fourier import fft, fftc, fftshift, ifft, ifftc, ifftshift
from .processing.phasing import autophase, phase
from .processing.utils import to_complex, to_real_imag

# =============================================================================
# 5. Vendor Integrations
# =============================================================================
from .vendor.bruker import remove_digital_filter

# =============================================================================
# 6. Visualization & Aesthetics
# =============================================================================
from .visualization.plot import (
    PlotHeatmapConfig,
    PlotQCGridConfig,
    PlotRidgeConfig,
    PlotTrajectoryConfig,
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
    "DEFAULTS",  # Legacy patch (branch refactor/18)
    # --- 2. Accessors ---
    "XmrisAccessor",
    "XmrisDatasetAccessor",
    # --- 3. Core Processing & Utilities ---
    "to_complex",
    "to_real_imag",
    "apodize_exp",
    "apodize_lg",
    "to_fid",
    "to_spectrum",
    "zero_fill",
    "fft",
    "fftc",
    "fftshift",
    "ifft",
    "ifftc",
    "ifftshift",
    "autophase",
    "phase",
    # --- 4. Fitting ---
    "fit_amares",
    # --- 5. Vendor ---
    "remove_digital_filter",
    # --- 6. Visualization Configs ---
    "PlotRidgeConfig",
    "PlotHeatmapConfig",
    "PlotTrajectoryConfig",
    "PlotQCGridConfig",
]
