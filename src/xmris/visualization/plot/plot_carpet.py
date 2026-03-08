from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator

from xmris.visualization.plot._base_config import BasePlotConfig, PlotParam
from xmris.visualization.plot._input_parsing import parse_input_dims_timeseries


@dataclass
class CarpetConfig(BasePlotConfig):
    """Configuration object for controlling the aesthetics of xmris Carpet Plots."""

    # --- Figure Setup ---
    figsize: tuple[float, float] = PlotParam(
        default=(8, 5),
        group="Figure Setup",
        description="Dimensions of the figure in inches (width, height).",
    )
    style: str = PlotParam(
        default="seaborn-v0_8-white",
        group="Figure Setup",
        description="Matplotlib style sheet used for the underlying canvas.",
    )
    fontfamily: str = PlotParam(
        default="sans-serif",
        group="Figure Setup",
        description="Font family used for all plot text (e.g., 'sans-serif', 'serif').",
    )

    # --- Colormap Aesthetics ---
    cmap: str = PlotParam(
        default="magma",
        group="Colormap Aesthetics",
        description="Matplotlib colormap string used for the 2D intensity.",
    )
    cmap_start: float = PlotParam(
        default=0.1,
        group="Colormap Aesthetics",
        description="Lower color boundary sampled from the colormap (0.0 to 1.0) "
        + "to avoid extremes.",
    )
    cmap_end: float = PlotParam(
        default=0.8,
        group="Colormap Aesthetics",
        description="Upper color boundary sampled from the colormap (0.0 to 1.0) "
        + "to avoid extremes.",
    )

    # --- Axes & Labels ---
    xlabel: str | None = PlotParam(
        default=None,
        group="Axes & Labels",
        description="Explicit x-axis string label. If None, uses the formatted "
        + "xarray dimension name.",
    )
    ylabel: str | None = PlotParam(
        default=None,
        group="Axes & Labels",
        description="Explicit y-axis string label. If None, uses the stacking dimension name.",
    )
    labelsize: int = PlotParam(
        default=12,
        group="Axes & Labels",
        description="Font size in points applied to the primary axis labels.",
    )
    ticklabelsize: int = PlotParam(
        default=10,
        group="Axes & Labels",
        description="Font size in points applied to the axis tick marks.",
    )

    # --- Ticks & Grid ---
    tick_color: str = PlotParam(
        default="lightgray",
        group="Ticks & Grid",
        description="Color string for the inward-facing tick marks.",
    )
    tick_len_major: float = PlotParam(
        default=5.0, group="Ticks & Grid", description="Length of major ticks in points."
    )
    tick_wid_major: float = PlotParam(
        default=1.0, group="Ticks & Grid", description="Width of major ticks in points."
    )
    tick_len_minor: float = PlotParam(
        default=2.0, group="Ticks & Grid", description="Length of minor ticks in points."
    )
    tick_wid_minor: float = PlotParam(
        default=0.8, group="Ticks & Grid", description="Width of minor ticks in points."
    )
    xminor_locator: int = PlotParam(
        default=5,
        group="Ticks & Grid",
        description="Number of sub-intervals between major x-axis ticks.",
    )
    yminor_locator: int = PlotParam(
        default=4,
        group="Ticks & Grid",
        description="Number of sub-intervals between major y-axis ticks.",
    )
    grid_on: bool = PlotParam(
        default=True,
        group="Ticks & Grid",
        description="Toggle the visibility of the overlay coordinate grid.",
    )
    grid_color: str = PlotParam(
        default="lightgray", group="Ticks & Grid", description="Color of the overlay grid lines."
    )
    grid_alpha: float = PlotParam(
        default=0.2, group="Ticks & Grid", description="Transparency of the overlay grid lines."
    )
    grid_linewidth: float = PlotParam(
        default=0.6, group="Ticks & Grid", description="Thickness of the overlay grid lines."
    )

    # --- Colorbar ---
    cbar_on: bool = PlotParam(
        default=True,
        group="Colorbar",
        description="Toggle the visibility of the colorbar legend.",
    )
    cbar_label: str | None = PlotParam(
        default="Signal Intensity (a.u.)",
        group="Colorbar",
        description="Text label appended to the side of the colorbar.",
    )


def plot_carpet(
    da: xr.DataArray,
    x_dim: str | None = None,
    stack_dim: str | None = None,
    ax: plt.Axes | None = None,
    config: CarpetConfig | None = None,
) -> plt.Axes:
    """
    Generate a publication-ready 2D carpet plot of stacked 1D spectra.

    This function automatically resolves missing dimensions, applies a truncated
    colormap to avoid print saturation, and overlays a highly readable measurement grid.

    Parameters
    ----------
    da : xr.DataArray
        The N-dimensional DataArray containing the spectroscopic data.
    x_dim : str, optional
        The dimension to plot along the horizontal axis. If None, auto-resolves.
    stack_dim : str, optional
        The dimension to plot along the vertical axis. If None, auto-resolves.
    ax : plt.Axes, optional
        A pre-existing matplotlib Axes object to plot onto.
    config : CarpetConfig, optional
        A configuration dataclass defining all aesthetic parameters.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the rendered carpet plot.
    """
    # 1. Resolve Configuration
    cfg = config or CarpetConfig()

    # 2. Resolve Data & Dimensions securely
    x_dim_str, stack_dim_str = parse_input_dims_timeseries(da, x_dim, stack_dim)

    da_plot = da.transpose(stack_dim_str, x_dim_str)
    x_vals = da_plot.coords[x_dim_str].values
    stack_vals = da_plot.coords[stack_dim_str].values
    spectra = da_plot.values

    x_unit = da_plot.coords[x_dim_str].attrs.get("units", "ppm")
    stack_unit = da_plot.coords[stack_dim_str].attrs.get("units", "s")

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

        # Truncate the colormap to prevent extreme whites/blacks from hiding data
        custom_cmap = ListedColormap(base_cmap(np.linspace(cfg.cmap_start, cfg.cmap_end, 256)))

        # 5. Core Plotting
        # (pcolormesh is faster and more accurate than imshow for non-uniform grids)
        c = ax.pcolormesh(x_vals, stack_vals, spectra, cmap=custom_cmap, shading="auto")

        # 6. Formatting & Cleanup
        x_label_str = cfg.xlabel if cfg.xlabel else x_dim_str.replace("_", " ").title()
        y_label_str = cfg.ylabel if cfg.ylabel else stack_dim_str.replace("_", " ").title()

        ax.set_xlabel(f"{x_label_str} [{x_unit}]", fontsize=cfg.labelsize, fontweight="bold")
        ax.set_ylabel(f"{y_label_str} [{stack_unit}]", fontsize=cfg.labelsize, fontweight="bold")

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
