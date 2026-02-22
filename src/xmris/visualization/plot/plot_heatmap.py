from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator

from xmris.config import DEFAULTS

from ._base_config import BasePlotConfig


@dataclass
class PlotHeatmapConfig(BasePlotConfig):
    """Configuration object for controlling the aesthetics of xmris Heatmap Plots.

    Modify these attributes to customize the publication-ready output.
    """

    # --- Figure & Canvas ---
    figsize: tuple[float, float] = field(
        default=(8, 5),
        metadata={
            "group": "Figure & Canvas",
            "description": "Dimensions of the figure (width, height).",
        },
    )
    style: str = field(
        default="seaborn-v0_8-white",
        metadata={
            "group": "Figure & Canvas",
            "description": "Matplotlib style sheet used for the underlying canvas.",
        },
    )
    fontfamily: str = field(
        default="sans-serif",
        metadata={
            "group": "Figure & Canvas",
            "description": "Font family used for all plot text.",
        },
    )

    # --- Colormap ---
    cmap: str = field(
        default="magma",
        metadata={
            "group": "Colormap",
            "description": "Matplotlib colormap used for the 2D intensity.",
        },
    )
    cmap_start: float = field(
        default=0.1,
        metadata={
            "group": "Colormap",
            "description": "Lower color boundary from the colormap (0 to 1) to avoid extremes.",  # noqa: E501
        },
    )
    cmap_end: float = field(
        default=0.8,
        metadata={
            "group": "Colormap",
            "description": "Upper color boundary from the colormap (0 to 1) to avoid extremes.",  # noqa: E501
        },
    )

    # --- Labels & Fonts ---
    labelsize: int = field(
        default=12,
        metadata={
            "group": "Labels & Fonts",
            "description": "Font size applied to the axis labels.",
        },
    )
    ticklabelsize: int = field(
        default=10,
        metadata={
            "group": "Labels & Fonts",
            "description": "Font size applied to the axis tick marks.",
        },
    )
    xlabel: str | None = field(
        default=None,
        metadata={
            "group": "Labels & Fonts",
            "description": "x-axis label. If None (default), uses the xarray dimension name.",  # noqa: E501
        },
    )
    ylabel: str | None = field(
        default=None,
        metadata={
            "group": "Labels & Fonts",
            "description": "y-axis label. If None (default), uses the stacking dimension name.",  # noqa: E501
        },
    )

    # --- Ticks & Grid ---
    tick_color: str = field(
        default="lightgray",
        metadata={
            "group": "Ticks & Grid",
            "description": "Color of the inward-facing tick marks.",
        },
    )
    tick_len_major: float = field(default=5.0, metadata={"group": "Ticks & Grid"})
    tick_wid_major: float = field(default=1.0, metadata={"group": "Ticks & Grid"})
    tick_len_minor: float = field(default=2.0, metadata={"group": "Ticks & Grid"})
    tick_wid_minor: float = field(default=0.8, metadata={"group": "Ticks & Grid"})

    xminor_locator: int = field(
        default=5,
        metadata={
            "group": "Ticks & Grid",
            "description": "Number of sub-intervals between major x-axis ticks.",
        },
    )
    yminor_locator: int = field(
        default=4,
        metadata={
            "group": "Ticks & Grid",
            "description": "Number of sub-intervals between major y-axis ticks.",
        },
    )

    grid_on: bool = field(
        default=True,
        metadata={
            "group": "Ticks & Grid",
            "description": "Toggle the visibility of the overlay grid.",
        },
    )
    grid_color: str = field(default="lightgray", metadata={"group": "Ticks & Grid"})
    grid_alpha: float = field(default=0.2, metadata={"group": "Ticks & Grid"})
    grid_linewidth: float = field(default=0.6, metadata={"group": "Ticks & Grid"})

    # --- Colorbar ---
    cbar_on: bool = field(
        default=True,
        metadata={
            "group": "Colorbar",
            "description": "Toggle the visibility of the colorbar.",
        },
    )
    cbar_label: str | None = field(
        default="Signal Intensity (a.u.)",
        metadata={
            "group": "Colorbar",
            "description": "Label appended to the side of the colorbar.",
        },
    )


def plot_heatmap(
    da: xr.DataArray,
    x_dim: str | None = None,
    stack_dim: str | None = None,
    ax: plt.Axes | None = None,
    config: PlotHeatmapConfig | None = None,
) -> plt.Axes:
    """Generate a publication-ready 2D heatmap plot of stacked 1D spectra."""
    # 1. Resolve Configuration
    cfg = config or PlotHeatmapConfig()

    # 2. Resolve Data & Dimensions
    x_dim = x_dim or DEFAULTS.chemical_shift.dim
    stack_dim = stack_dim or DEFAULTS.time.dim

    if x_dim not in da.dims or stack_dim not in da.dims:
        raise ValueError(f"DataArray must contain '{x_dim}' and '{stack_dim}'.")

    da_plot = da.transpose(stack_dim, x_dim)
    x_vals = da_plot.coords[x_dim].values
    stack_vals = da_plot.coords[stack_dim].values
    spectra = da_plot.values

    x_unit = da_plot.coords[x_dim].attrs.get("units", "ppm")
    stack_unit = da_plot.coords[stack_dim].attrs.get("units", "s")

    # 3. Create Axes with Context Managers
    custom_rc = {"font.family": cfg.fontfamily, "axes.linewidth": 1.2}

    with plt.style.context(cfg.style), plt.rc_context(custom_rc):
        if ax is None:
            fig, ax = plt.subplots(figsize=cfg.figsize)
        else:
            fig = ax.get_figure()

        # 4. Truncated Colormap Setup
        try:
            base_cmap = plt.colormaps[cfg.cmap]
        except AttributeError:
            base_cmap = cm.get_cmap(cfg.cmap)

        custom_cmap = ListedColormap(
            base_cmap(np.linspace(cfg.cmap_start, cfg.cmap_end, 256))
        )

        # 5. Core Plotting (pcolormesh)
        c = ax.pcolormesh(x_vals, stack_vals, spectra, cmap=custom_cmap, shading="auto")

        # 6. Formatting & Cleanup
        x_label_str = cfg.xlabel if cfg.xlabel else x_dim.replace("_", " ").title()
        y_label_str = cfg.ylabel if cfg.ylabel else stack_dim.replace("_", " ").title()

        ax.set_xlabel(
            f"{x_label_str} [{x_unit}]", fontsize=cfg.labelsize, fontweight="bold"
        )
        ax.set_ylabel(
            f"{y_label_str} [{stack_unit}]", fontsize=cfg.labelsize, fontweight="bold"
        )

        if not ax.xaxis_inverted():
            ax.invert_xaxis()

        # Force ticks and grid to draw OVER the pcolormesh
        ax.set_axisbelow(False)

        if cfg.grid_on:
            ax.grid(
                True,
                which="major",
                color=cfg.grid_color,
                alpha=cfg.grid_alpha,
                linewidth=cfg.grid_linewidth,
                linestyle="-",
            )

        ax.xaxis.set_minor_locator(AutoMinorLocator(cfg.xminor_locator))
        ax.yaxis.set_minor_locator(AutoMinorLocator(cfg.yminor_locator))

        # Tick Styling
        ax.tick_params(
            which="major",
            direction="in",
            color=cfg.tick_color,
            labelcolor="black",
            length=cfg.tick_len_major,
            width=cfg.tick_wid_major,
            top=True,
            right=True,
            labelsize=cfg.ticklabelsize,
        )
        ax.tick_params(
            which="minor",
            direction="in",
            color=cfg.tick_color,
            length=cfg.tick_len_minor,
            width=cfg.tick_wid_minor,
            top=True,
            right=True,
        )

        # Colorbar Handling
        if cfg.cbar_on:
            cbar = fig.colorbar(c, ax=ax, pad=0.02)
            if cfg.cbar_label:
                cbar.set_label(cfg.cbar_label, fontsize=cfg.labelsize, fontweight="bold")
            cbar.ax.tick_params(
                which="major",
                direction="in",
                length=cfg.tick_len_major - 2,
                width=1.2,
                labelsize=cfg.ticklabelsize,
            )

        if ax.get_figure() is fig:
            fig.tight_layout()

    return ax
