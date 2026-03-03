from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator

from xmris.visualization.plot._base_config import BasePlotConfig, PlotParam
from xmris.visualization.plot._input_parsing import parse_input_dims_timeseries


@dataclass
class WaterfallConfig(BasePlotConfig):
    """Configuration object for controlling the aesthetics of xmris Waterfall Plots."""

    # --- Figure Setup ---
    figsize: tuple[float, float] = PlotParam(
        default=(8, 6),
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

    # --- Stack Geometry ---
    stack_offset: float = PlotParam(
        default=0.5,
        group="Stack Geometry",
        description="Vertical baseline shift between stacked spectra"
        + "(in normalized amplitude units).",
    )
    stack_scale: float = PlotParam(
        default=10.0,
        group="Stack Geometry",
        description="Height multiplier for individual spectra to control visual "
        + "overlap (baseline max is 1.0).",
    )
    stack_skew: float = PlotParam(
        default=-20.0,
        group="Stack Geometry",
        description="Horizontal skew angle in degrees (strictly between -89.0 and 89.0). "
        + "0 is vertically straight.",
    )

    # --- Stack Aesthetics ---
    cmap: str | None = PlotParam(
        default="magma",
        group="Stack Aesthetics",
        description="Matplotlib colormap string. Set to None to disable filled areas.",
    )
    cmap_start: float = PlotParam(
        default=0.8,
        group="Stack Aesthetics",
        description="Top color boundary sampled from the colormap (0.0 to 1.0).",
    )
    cmap_end: float = PlotParam(
        default=0.1,
        group="Stack Aesthetics",
        description="Bottom color boundary sampled from the colormap (0.0 to 1.0).",
    )
    alpha: float = PlotParam(
        default=0.75,
        group="Stack Aesthetics",
        description="Transparency (alpha) of the colored filled area under spectra.",
    )
    linewidth: float = PlotParam(
        default=0.8,
        group="Stack Aesthetics",
        description="Thickness in points of standard spectrum outlines.",
    )
    linewidth_highlight: float = PlotParam(
        default=1.3,
        group="Stack Aesthetics",
        description="Thickness in points for labeled/highlighted spectrum outlines.",
    )

    # --- Stack Labels ---
    stack_label_step: int = PlotParam(
        default=10,
        group="Stack Labels",
        description="Interval at which to add a stack y-axis label "
        + "(e.g., every 10th spectrum).",
    )
    stack_label_x_offset: float = PlotParam(
        default=-0.25,
        group="Stack Labels",
        description="Horizontal adjustment for stack labels (in x-axis data units).",
    )
    stack_label_y_offset: float = PlotParam(
        default=0.0,
        group="Stack Labels",
        description="Vertical adjustment for stack labels (as a fraction of stack_offset).",  # noqa: E501
    )

    # --- Axes & Ticks ---
    xlabel: str | None = PlotParam(
        default=None,
        group="Axes & Ticks",
        description="Explicit x-axis string label. If None, uses the formatted "
        + "xarray dimension name.",
    )
    labelsize: int = PlotParam(
        default=12,
        group="Axes & Ticks",
        description="Font size in points applied to the primary axis labels.",
    )
    ticklabelsize: int = PlotParam(
        default=10,
        group="Axes & Ticks",
        description="Font size in points applied to the axis tick marks and stack labels.",  # noqa: E501
    )
    pad_left: float = PlotParam(
        default=0.0,
        group="Axes & Ticks",
        description="Empty padding space on the high-value (left) side in x-axis " + "data units.",
    )
    pad_right: float = PlotParam(
        default=0.0,
        group="Axes & Ticks",
        description="Empty padding space on the low-value (right) side in x-axis " + "data units.",
    )
    xminor_locator: int = PlotParam(
        default=5,
        group="Axes & Ticks",
        description="Number of sub-intervals between major x-axis ticks.",
    )

    # --- Annotations ---
    annotation: str | None = PlotParam(
        default="Absorption Mode",
        group="Annotations",
        description="Text annotation placed in the top left corner. "
        + "Set to None to hide entirely.",
    )


def plot_waterfall(
    da: xr.DataArray,
    x_dim: str | None = None,
    stack_dim: str | None = None,
    ax: plt.Axes | None = None,
    config: WaterfallConfig | None = None,
) -> plt.Axes:
    """
    Generate a publication-ready waterfall plot (2D stacked series) of 1D spectra.

    This function automatically normalizes the array amplitude, applies configured
    styling, and safely resolves missing dimensions.

    Parameters
    ----------
    da : xr.DataArray
        The N-dimensional DataArray containing the spectroscopic data.
    x_dim : str, optional
        The dimension to plot along the horizontal axis. If None, attempts to
        auto-resolve (prefers 'chemical_shift' or 'frequency').
    stack_dim : str, optional
        The dimension to stack vertically. If None, attempts to auto-resolve
        based on remaining dimensions.
    ax : plt.Axes, optional
        A pre-existing matplotlib Axes object to plot onto. If None, a new Figure
        and Axes will be created.
    config : PlotWaterfallConfig, optional
        A configuration dataclass defining all aesthetic parameters. If None,
        default styling is applied.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the rendered waterfall plot.
    """
    # 1. Resolve Configuration safely
    cfg = config or WaterfallConfig()

    # Validate skew angle to prevent infinity/math errors
    if not (-89.0 <= cfg.stack_skew <= 89.0):
        raise ValueError("stack_skew must be an angle in degrees strictly between -89.0 and 89.0.")

    # 2. Resolve Data & Dimensions
    x_dim_str, stack_dim_str = parse_input_dims_timeseries(da, x_dim, stack_dim)

    da_plot = da.transpose(stack_dim_str, x_dim_str)
    x_vals = da_plot.coords[x_dim_str].values
    stack_vals = da_plot.coords[stack_dim_str].values

    # Normalize the entire 2D array by the global absolute maximum
    spectra = da_plot.values.copy()
    max_val = np.max(np.abs(spectra))
    if max_val > 0:
        spectra = spectra / max_val

    x_unit = da_plot.coords[x_dim_str].attrs.get("units", "ppm")
    stack_unit = da_plot.coords[stack_dim_str].attrs.get("units", "s")

    # 3. Create Axes with Context Managers
    custom_rc = {"font.family": cfg.fontfamily, "axes.linewidth": 1.2}

    with plt.style.context(cfg.style), plt.rc_context(custom_rc):
        if ax is None:
            _fig, ax = plt.subplots(figsize=cfg.figsize)
        else:
            _fig = ax.get_figure()

        # 4. Color Mapping Setup (Skip if cmap is None)
        line_colors = None
        if cfg.cmap is not None:
            try:
                base_cmap = plt.colormaps[cfg.cmap]
            except AttributeError:
                base_cmap = cm.get_cmap(cfg.cmap)

            line_colors = base_cmap(np.linspace(cfg.cmap_start, cfg.cmap_end, len(stack_vals)))

        # Pre-calculate the tangent of the skew angle for horizontal displacement
        skew_tan = np.tan(np.radians(cfg.stack_skew))

        # 5. Core Plotting Loop
        for i in range(len(stack_vals) - 1, -1, -1):
            y_baseline = i * cfg.stack_offset

            # Apply amplitude scaling and vertical offset
            shifted_spectrum = (spectra[i, :] * cfg.stack_scale) + y_baseline

            # Apply geometric horizontal skew based on the vertical offset
            shifted_x = x_vals + (y_baseline * skew_tan)

            is_labeled = (i % cfg.stack_label_step == 0) or (i == len(stack_vals) - 1)
            current_lw = cfg.linewidth_highlight if is_labeled else cfg.linewidth

            # Only draw the filled area if a colormap was provided
            if cfg.cmap is not None and line_colors is not None:
                ax.fill_between(
                    shifted_x,
                    y_baseline,
                    shifted_spectrum,
                    color=line_colors[i],
                    alpha=cfg.alpha,
                    linewidth=0,
                    zorder=len(stack_vals) - i,
                    clip_on=False,
                )

            ax.plot(
                shifted_x,
                shifted_spectrum,
                color="black",
                linewidth=current_lw,
                zorder=len(stack_vals) - i + 0.1,
                clip_on=False,
            )

            if is_labeled:
                label_x = shifted_x.min() + cfg.stack_label_x_offset
                label_y = y_baseline + (cfg.stack_offset * cfg.stack_label_y_offset)

                ax.text(
                    label_x,
                    label_y,
                    f"{stack_vals[i]:.0f} {stack_unit}",
                    fontsize=cfg.ticklabelsize,
                    color="black",
                    ha="left",
                    va="center",
                    fontweight="bold",
                    clip_on=False,
                )

        # 6. Formatting & Cleanup
        x_label_str = cfg.xlabel if cfg.xlabel else x_dim_str.replace("_", " ").title()

        ax.set_xlabel(f"{x_label_str} [{x_unit}]", fontsize=cfg.labelsize, fontweight="bold")

        if not ax.xaxis_inverted():
            ax.invert_xaxis()

        # Enforce limits strictly to the unskewed original data range
        ax.set_xlim(x_vals.max() + cfg.pad_left, x_vals.min() - cfg.pad_right)

        ax.xaxis.set_minor_locator(AutoMinorLocator(cfg.xminor_locator))
        ax.tick_params(
            axis="x",
            which="major",
            direction="out",
            length=6,
            width=1.2,
            labelsize=cfg.ticklabelsize,
        )
        ax.tick_params(axis="x", which="minor", direction="out", length=3, width=1)

        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.2)

        if cfg.annotation:
            ax.text(
                0.0,
                0.97,
                cfg.annotation,
                transform=ax.transAxes,
                fontsize=cfg.labelsize - 1,
                fontstyle="italic",
                color="gray",
                ha="left",
                va="top",
            )

    return ax
