from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator

from xmris.config import DEFAULTS

from ._base_config import BasePlotConfig


@dataclass
class PlotRidgeConfig(BasePlotConfig):
    """Configuration object for controlling the aesthetics of xmris Ridge Plots.

    Modify these attributes to customize the publication-ready output.
    """

    # --- Figure & Canvas ---
    fig_size: tuple[float, float] = field(
        default=(8, 6),
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
    font_family: str = field(
        default="sans-serif",
        metadata={
            "group": "Figure & Canvas",
            "description": "Font family used for all plot text.",
        },
    )

    # --- Stacking & Colors ---
    offset_step: float = field(
        default=1.5,
        metadata={
            "group": "Stacking & Colors",
            "description": "Vertical shift between consecutive stacked spectra.",
        },
    )
    cmap_name: str = field(
        default="magma",
        metadata={
            "group": "Stacking & Colors",
            "description": "Matplotlib colormap used for the time/stack gradient.",
        },
    )
    cmap_start: float = field(
        default=0.8,
        metadata={
            "group": "Stacking & Colors",
            "description": "Top color boundary from the colormap (0 to 1).",
        },
    )
    cmap_end: float = field(
        default=0.1,
        metadata={
            "group": "Stacking & Colors",
            "description": "Bottom color boundary from the colormap (0 to 1).",
        },
    )
    fill_alpha: float = field(
        default=0.75,
        metadata={
            "group": "Stacking & Colors",
            "description": "Transparency of the colored filled area under spectra.",
        },
    )

    # --- Line Aesthetics ---
    line_width: float = field(
        default=0.8,
        metadata={
            "group": "Line Aesthetics",
            "description": "Thickness of standard spectrum outlines.",
        },
    )
    line_width_highlight: float = field(
        default=1.3,
        metadata={
            "group": "Line Aesthetics",
            "description": "Thickness for labeled/highlighted timepoint outlines.",
        },
    )

    # --- Timeline Labels ---
    label_every_n: int = field(
        default=10,
        metadata={
            "group": "Timeline Labels",
            "description": "Interval at which to add a time label (every Nth spectrum).",
        },
    )
    label_x_nudge: float = field(
        default=-0.25,
        metadata={
            "group": "Timeline Labels",
            "description": "Horizontal adjustment/offset for time labels.",
        },
    )
    label_y_nudge_frac: float = field(
        default=0.0,
        metadata={
            "group": "Timeline Labels",
            "description": "Vertical adjustment for time labels (fraction of offset_step).",  # noqa: E501
        },
    )
    label_fontsize: int = field(
        default=12,
        metadata={
            "group": "Timeline Labels",
            "description": "Font size applied to the axis labels.",
        },
    )
    tick_fontsize: int = field(
        default=10,
        metadata={
            "group": "Timeline Labels",
            "description": "Font size applied to the axis tick marks.",
        },
    )

    # --- Axes Padding & Ticks ---
    pad_left: float = field(
        default=0.0,
        metadata={
            "group": "Axes Padding & Ticks",
            "description": "Empty padding space on the high-ppm (left) side.",
        },
    )
    pad_right: float = field(
        default=0.0,
        metadata={
            "group": "Axes Padding & Ticks",
            "description": "Empty padding space on the low-ppm (right) side.",
        },
    )
    minor_locator_x: int = field(
        default=5,
        metadata={
            "group": "Axes Padding & Ticks",
            "description": "Number of sub-intervals between major X-axis ticks.",
        },
    )


def plot_ridge(
    da: xr.DataArray,
    x_dim: str | None = None,
    stack_dim: str | None = None,
    ax: plt.Axes | None = None,
    config: PlotRidgeConfig | None = None,
) -> plt.Axes:
    """Generate a publication-ready ridge plot (2D waterfall) of stacked 1D spectra."""
    # 1. Resolve Configuration safely
    cfg = config or PlotRidgeConfig()

    # 2. Resolve Data & Dimensions
    x_dim = x_dim or DEFAULTS.chemical_shift.dim
    stack_dim = stack_dim or DEFAULTS.time.dim

    if x_dim not in da.dims or stack_dim not in da.dims:
        raise ValueError(f"DataArray must contain '{x_dim}' and '{stack_dim}'.")

    da_plot = da.transpose(stack_dim, x_dim)
    x_vals = da_plot.coords[x_dim].values
    stack_vals = da_plot.coords[stack_dim].values
    spectra = da_plot.values

    x_unit = da_plot.coords[x_dim].attrs.get("units", "p.p.m.")
    stack_unit = da_plot.coords[stack_dim].attrs.get("units", "s")

    # 3. Create Axes with Context Managers
    custom_rc = {"font.family": cfg.font_family, "axes.linewidth": 1.2}

    with plt.style.context(cfg.style), plt.rc_context(custom_rc):
        if ax is None:
            fig, ax = plt.subplots(figsize=cfg.fig_size)
        else:
            fig = ax.get_figure()

        # 4. Color Mapping Setup
        try:
            base_cmap = plt.colormaps[cfg.cmap_name]
        except AttributeError:
            base_cmap = cm.get_cmap(cfg.cmap_name)

        line_colors = base_cmap(
            np.linspace(cfg.cmap_start, cfg.cmap_end, len(stack_vals))
        )

        # 5. Core Plotting Loop
        for i in range(len(stack_vals) - 1, -1, -1):
            y_baseline = i * cfg.offset_step
            shifted_spectrum = spectra[i, :] + y_baseline

            is_labeled = (i % cfg.label_every_n == 0) or (i == len(stack_vals) - 1)
            current_lw = cfg.line_width_highlight if is_labeled else cfg.line_width

            ax.fill_between(
                x_vals,
                y_baseline,
                shifted_spectrum,
                color=line_colors[i],
                alpha=cfg.fill_alpha,
                linewidth=0,
                zorder=len(stack_vals) - i,
            )

            ax.plot(
                x_vals,
                shifted_spectrum,
                color="black",
                linewidth=current_lw,
                zorder=len(stack_vals) - i + 0.1,
            )

            if is_labeled:
                label_x = x_vals.min() + cfg.label_x_nudge
                label_y = y_baseline + (cfg.offset_step * cfg.label_y_nudge_frac)

                ax.text(
                    label_x,
                    label_y,
                    f"{stack_vals[i]:.0f} {stack_unit}",
                    fontsize=cfg.tick_fontsize,
                    color="black",
                    ha="left",
                    va="center",
                    fontweight="bold",
                )

        # 6. Formatting & Cleanup
        x_label_str = (
            r"$^{13}$C Chemical Shift"
            if x_dim == DEFAULTS.chemical_shift.dim
            else x_dim.replace("_", " ").title()
        )
        ax.set_xlabel(
            f"{x_label_str} ({x_unit})", fontsize=cfg.label_fontsize, fontweight="bold"
        )

        if not ax.xaxis_inverted():
            ax.invert_xaxis()

        ax.set_xlim(x_vals.max() + cfg.pad_left, x_vals.min() - cfg.pad_right)

        ax.xaxis.set_minor_locator(AutoMinorLocator(cfg.minor_locator_x))
        ax.tick_params(
            axis="x",
            which="major",
            direction="out",
            length=6,
            width=1.2,
            labelsize=cfg.tick_fontsize,
        )
        ax.tick_params(axis="x", which="minor", direction="out", length=3, width=1)

        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.2)

        ax.text(
            0.0,
            0.97,
            "Absorption Mode",
            transform=ax.transAxes,
            fontsize=cfg.label_fontsize - 1,
            fontstyle="italic",
            color="gray",
            ha="left",
            va="top",
        )

        if ax.get_figure() is fig:
            fig.tight_layout()

    return ax
