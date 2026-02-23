from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import AutoMinorLocator

from ._base_config import BasePlotConfig


@dataclass
class PlotTrajectoryConfig(BasePlotConfig):
    """Configuration for AMARES 1D Trajectory Plots with CRLB Shading."""

    # --- Figure & Canvas ---
    figsize: tuple[float, float] = field(
        default=(8, 5),
        metadata={
            "group": "Figure & Canvas",
            "description": "Dimensions (width, height).",
        },
    )
    style: str = field(
        default="seaborn-v0_8-white",
        metadata={
            "group": "Figure & Canvas",
            "description": "Matplotlib style sheet.",
        },
    )
    fontfamily: str = field(
        default="sans-serif",
        metadata={
            "group": "Figure & Canvas",
            "description": "Font family used for all text.",
        },
    )
    axes_linewidth: float = field(
        default=1.2,
        metadata={
            "group": "Figure & Canvas",
            "description": "Line thickness for the plot bounding box.",
        },
    )

    # --- Aesthetics ---
    palette: str | tuple[str, ...] = field(
        default="tab10",
        metadata={
            "group": "Aesthetics",
            "description": "Colormap name or sequence of explicit colors to cycle through.",
        },
    )
    markers: tuple[str, ...] = field(
        default=("o", "s", "^", "D", "v", "p", "*", "h", "X"),
        metadata={
            "group": "Aesthetics",
            "description": "Sequence of marker styles to cycle through for each metabolite.",
        },
    )
    linewidth: float = field(
        default=2.0,
        metadata={
            "group": "Aesthetics",
            "description": "Line thickness for the central trajectories.",
        },
    )
    markersize: float = field(
        default=6.0,
        metadata={
            "group": "Aesthetics",
            "description": "Marker size for trajectory points.",
        },
    )
    fill_alpha: float = field(
        default=0.2,
        metadata={
            "group": "Aesthetics",
            "description": "Transparency of the CRLB shaded error band.",
        },
    )
    fill_linewidth: float = field(
        default=0.0,
        metadata={
            "group": "Aesthetics",
            "description": "Outline thickness of the CRLB shaded region.",
        },
    )

    # --- Labels & Fonts ---
    title: str | None = field(
        default="Metabolite Trajectories (Shading = CRLB Error)",
        metadata={
            "group": "Labels & Fonts",
            "description": "Plot title. Set to None to hide.",
        },
    )
    xlabel: str | None = field(
        default=None,
        metadata={
            "group": "Labels & Fonts",
            "description": "x-axis label. If None, auto-derives from the xarray dimension.",
        },
    )
    ylabel: str = field(
        default="Amplitude [a.u.]",
        metadata={
            "group": "Labels & Fonts",
            "description": "y-axis label.",
        },
    )
    labelsize: int = field(
        default=12,
        metadata={
            "group": "Labels & Fonts",
            "description": "Font size applied to the axis labels.",
        },
    )
    fontweight: str = field(
        default="bold",
        metadata={
            "group": "Labels & Fonts",
            "description": "Font weight for titles and labels (e.g., 'normal', 'bold').",
        },
    )

    # --- Grid, Ticks & Legend ---
    tick_direction: str = field(
        default="in",
        metadata={
            "group": "Grid & Ticks",
            "description": "Direction of the tick marks ('in', 'out', or 'inout').",
        },
    )
    grid_on: bool = field(
        default=True,
        metadata={
            "group": "Grid & Ticks",
            "description": "Toggle visibility of the background grid.",
        },
    )
    grid_alpha: float = field(
        default=0.3,
        metadata={
            "group": "Grid & Ticks",
            "description": "Transparency of the grid lines.",
        },
    )
    grid_linestyle: str = field(
        default="--",
        metadata={
            "group": "Grid & Ticks",
            "description": "Style of the grid lines (e.g., '--', '-').",
        },
    )
    legend_on: bool = field(
        default=True,
        metadata={
            "group": "Legend",
            "description": "Toggle visibility of the legend.",
        },
    )
    legend_frameon: bool = field(
        default=True,
        metadata={
            "group": "Legend",
            "description": "Toggle the bounding box frame around the legend.",
        },
    )


def plot_trajectory(
    ds: xr.Dataset,
    dim: str,
    metabolites: list[str] | None = None,
    ax: plt.Axes | None = None,
    config: PlotTrajectoryConfig | None = None,
) -> plt.Axes:
    """
    Plot metabolite amplitude trajectories along a specified dimension.

    The Cramér-Rao Lower Bound (CRLB) is mathematically translated into an absolute
    error and used to draw a shaded confidence interval around the trajectory line,
    visually indicating the fit uncertainty.
    """
    cfg = config or PlotTrajectoryConfig()

    # 1. Validate Dataset
    required_vars = ["amplitude", "crlb"]
    for v in required_vars:
        if v not in ds.data_vars:
            raise ValueError(f"Dataset missing required AMARES variable: {v}")

    if dim not in ds.dims:
        raise ValueError(f"Dimension '{dim}' not found in Dataset.")

    if metabolites is not None:
        ds = ds.sel(Metabolite=metabolites)
    metab_list = ds.coords["Metabolite"].values

    # 2. Setup Context & Palette
    custom_rc = {"font.family": cfg.fontfamily, "axes.linewidth": cfg.axes_linewidth}

    with plt.style.context(cfg.style), plt.rc_context(custom_rc):
        if ax is None:
            fig, ax = plt.subplots(figsize=cfg.figsize)
        else:
            fig = ax.get_figure()

        # Parse categorical colors safely
        if isinstance(cfg.palette, str):
            cmap = plt.get_cmap(cfg.palette)
            colors = (
                cmap.colors
                if hasattr(cmap, "colors")
                else [cmap(i) for i in np.linspace(0, 1, len(metab_list))]
            )
        else:
            colors = cfg.palette

        # 3. Extract Coordinates and Units robustly
        series_coords = ds.coords[dim].values
        unit = ds.coords[dim].attrs.get("units")

        # 4. Plot Trajectories
        for i, metab in enumerate(metab_list):
            amps = ds["amplitude"].sel(Metabolite=metab).values
            crlbs = ds["crlb"].sel(Metabolite=metab).values

            # Calculate absolute error from CRLB percentage: Error = Amp * (CRLB / 100)
            crlbs_clean = np.nan_to_num(crlbs, nan=0.0)
            abs_error = amps * (crlbs_clean / 100.0)

            # Cycle through colors and markers
            color = colors[i % len(colors)]
            marker = cfg.markers[i % len(cfg.markers)]

            # Plot the central trajectory
            ax.plot(
                series_coords,
                amps,
                color=color,
                linewidth=cfg.linewidth,
                marker=marker,
                markersize=cfg.markersize,
                label=f"{metab}",
            )

            # Plot the CRLB shaded error band
            ax.fill_between(
                series_coords,
                amps - abs_error,
                amps + abs_error,
                color=color,
                alpha=cfg.fill_alpha,
                linewidth=cfg.fill_linewidth,
            )

        # 5. Format Labels and Title
        if cfg.xlabel is not None:
            ax.set_xlabel(cfg.xlabel, fontweight=cfg.fontweight, fontsize=cfg.labelsize)
        else:
            label_text = f"{dim} [{unit}]" if unit else f"{dim}"
            ax.set_xlabel(label_text, fontweight=cfg.fontweight, fontsize=cfg.labelsize)

        ax.set_ylabel(cfg.ylabel, fontweight=cfg.fontweight, fontsize=cfg.labelsize)

        if cfg.title:
            ax.set_title(cfg.title, fontweight=cfg.fontweight)

        # 6. Format Axes, Grid, and Legend
        if cfg.grid_on:
            ax.grid(True, alpha=cfg.grid_alpha, linestyle=cfg.grid_linestyle)

        if cfg.legend_on:
            ax.legend(frameon=cfg.legend_frameon)

        # Force inward ticks & mirroring for a publishable look
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", direction=cfg.tick_direction, top=True, right=True)

        if ax.get_figure() is fig:
            fig.tight_layout()

    return ax
