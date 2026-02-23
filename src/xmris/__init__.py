# 0. Expose the submodules for quartodoc (Griffe) traversal
from . import accessor, config, fitting, processing, utils, vendor  # noqa: I001

# 1. Configuration (The Central Nervous System)
from .config import DEFAULTS

# 2. Accessor (Importing this automatically registers the .xmr namespace)
from .accessor import XmrisAccessor, XmrisDatasetAccessor

# 3. Utilities
from .utils import to_complex, to_real_imag

# 4. Core Processing
from .processing.fid import apodize_exp, apodize_lg, to_fid, to_spectrum, zero_fill
from .processing.fourier import fft, fftc, fftshift, ifft, ifftc, ifftshift
from .processing.phase import autophase, phase

# 5. Vendor Specific
from .vendor.bruker import remove_digital_filter

# 6. Fitting
from .fitting.amares import fit_amares

# 7. Visualisation
from .visualization.plot import (
    PlotHeatmapConfig,
    PlotRidgeConfig,
    PlotTrajectoryConfig,
    PlotQCGridConfig,
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
    "DEFAULTS",
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
