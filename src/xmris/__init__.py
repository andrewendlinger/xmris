# 0. Expose the submodules for quartodoc (Griffe) traversal
from . import fitting, processing, vendor  # noqa: I001

# 1. Configuration (The Central Nervous System)
from .core import ATTRS, COORDS, DIMS, VARS, accessor

# 2. Accessor (Importing this automatically registers the .xmr namespace)
from .core.accessor import XmrisAccessor, XmrisDatasetAccessor

# legacy -- patch for now (branch refactor/18)
from .config import DEFAULTS

# 6. Fitting
from .fitting.amares import fit_amares
from .processing import utils

# 4. Core Processing
from .processing.fid import apodize_exp, apodize_lg, to_fid, to_spectrum, zero_fill
from .processing.fourier import fft, fftc, fftshift, ifft, ifftc, ifftshift
from .processing.phasing import autophase, phase

# 3. Utilities
from .processing.utils import to_complex, to_real_imag

# 5. Vendor Specific
from .vendor.bruker import remove_digital_filter

# 7. Visualisation
from .visualization.plot import (
    PlotHeatmapConfig,
    PlotQCGridConfig,
    PlotRidgeConfig,
    PlotTrajectoryConfig,
)

# Explicitly define the public API.
__all__ = [
    # --- Submodules (Required for quartodoc to build the pages) ---
    "accessor",
    "config",
    "fitting",
    "processing",
    "utils",
    "vendor",
    # --- Flat API (For the users) ---
    # Config
    "ATTRS",
    "COORDS",
    "DIMS",
    "VARS",
    "DEFAULTS",  # legacy -- patch for now (branch refactor/18)
    # Accessor
    "XmrisAccessor",
    "XmrisDatasetAccessor",
    # Utilities
    "to_complex",
    "to_real_imag",
    # FID Operations
    "apodize_exp",
    "apodize_lg",
    "to_fid",
    "to_spectrum",
    "zero_fill",
    # Fourier Transforms
    "fft",
    "fftc",
    "fftshift",
    "ifft",
    "ifftc",
    "ifftshift",
    # Phase Correction
    "autophase",
    "phase",
    # Vendor
    "remove_digital_filter",
    # Fitting
    "fit_amares",
    # Visualization - Plotting Configs
    "PlotRidgeConfig",
    "PlotHeatmapConfig",
    "PlotTrajectoryConfig",
    "PlotQCGridConfig",
]
