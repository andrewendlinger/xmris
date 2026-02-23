from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

from ._base_config import BasePlotConfig


@dataclass
class PlotQCGridConfig(BasePlotConfig):
    """Configuration for AMARES Fit Quality Control Grids."""

    # --- Figure & Canvas ---
    style: str = field(
        default="seaborn-v0_8-white",
        metadata={"group": "Figure & Canvas", "description": "Matplotlib style sheet."},
    )
    fontfamily: str = field(
        default="sans-serif",
        metadata={
            "group": "Figure & Canvas",
            "description": "Font family used for all text.",
        },
    )

    # --- Grid Layout ---
    max_cols: int = field(
        default=10,
        metadata={
            "group": "Grid Layout",
            "description": "Maximum number of columns in the grid.",
        },
    )
    max_plots: int | None = field(
        default=None,
        metadata={
            "group": "Grid Layout",
            "description": "Max subplots. If None (default), plots all spectra. If N > max, samples evenly.",
        },
    )
    sharey: bool = field(
        default=False,
        metadata={
            "group": "Grid Layout",
            "description": "Share Y-axis limits across all plots to accurately compare absolute amplitudes.",
        },
    )

    # --- Quality Control ---
    crlb_threshold: float = field(
        default=20.0,
        metadata={
            "group": "Quality Control",
            "description": "CRLB % threshold to flag a bad fit.",
        },
    )
    fail_color: str = field(
        default="#ffe6e6",
        metadata={
            "group": "Quality Control",
            "description": "Background color for failed fits (default: light red).",
        },
    )

    # --- Aesthetics & Zooming ---
    plot_residuals: bool = field(
        default=True,
        metadata={
            "group": "Aesthetics",
            "description": "Whether to plot the residual line below the fit.",
        },
    )
    xlim: tuple[float, float] | None = field(
        default=None,
        metadata={
            "group": "Aesthetics",
            "description": "X-axis limits for zooming (e.g., (10.0, -20.0)).",
        },
    )
    ylim: tuple[float, float] | None = field(
        default=None,
        metadata={
            "group": "Aesthetics",
            "description": "Y-axis limits for zooming (e.g., (-10, 100)).",
        },
    )


def plot_qc_grid(
    ds: xr.Dataset,
    dim: str,
    config: PlotQCGridConfig | None = None,
) -> plt.Figure:
    """
    Generate a Quality Control grid of spectra with overlaid AMARES fits.

    If the dataset dimension exceeds `max_plots`, it will linearly downsample
    the indices to provide an evenly spaced overview. Subplots where ANY metabolite
    exceeds the `crlb_threshold` will have their background shaded `fail_color` and
    the maximum CRLB printed.
    """
    cfg = config or PlotQCGridConfig()

    required_vars = ["fit_data", "raw_data", "crlb"]
    for v in required_vars:
        if v not in ds.data_vars:
            raise ValueError(f"Dataset missing required AMARES variable: {v}")

    if dim not in ds.dims:
        raise ValueError(f"Dimension '{dim}' not found in Dataset.")

    # Determine which indices to plot
    n_total = ds.sizes[dim]
    if cfg.max_plots is None or n_total <= cfg.max_plots:
        indices = np.arange(n_total)
    else:
        # Linearly sample the array to fit within max_plots
        indices = np.linspace(0, n_total - 1, cfg.max_plots, dtype=int)

    n_plots = len(indices)
    cols = min(n_plots, cfg.max_cols)
    rows = int(np.ceil(n_plots / cols))

    # Base subplot size: 3.5 width, 2.5 height
    figsize = (cols * 3.5, rows * 2.5)

    custom_rc = {"font.family": cfg.fontfamily, "axes.linewidth": 1.0}
    with plt.style.context(cfg.style), plt.rc_context(custom_rc):
        # Create grid with NO spacing
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=figsize,
            sharex=True,
            sharey=cfg.sharey,
            squeeze=False,
            gridspec_kw={"wspace": 0.0, "hspace": 0.0},
        )
        axes_flat = axes.flatten()

        freq_dim = "Frequency"
        dim_coords = ds.coords[dim].values
        dim_unit = ds.coords[dim].attrs.get("units", "")

        # Pre-convert needed slices to frequency domain to save time
        ds_subset = ds.isel({dim: indices})
        spec_raw = ds_subset["raw_data"].xmr.to_spectrum(out_dim=freq_dim).real
        spec_fit = ds_subset["fit_data"].xmr.to_spectrum(out_dim=freq_dim).real
        if cfg.plot_residuals:
            spec_res = ds_subset["residuals"].xmr.to_spectrum(out_dim=freq_dim).real

        freq_coords = spec_raw.coords[freq_dim].values

        for i, idx_val in enumerate(indices):
            ax = axes_flat[i]

            # Plot Data
            ax.plot(
                freq_coords,
                spec_raw.isel({dim: i}),
                color="black",
                alpha=0.4,
                label="Raw",
            )
            ax.plot(
                freq_coords,
                spec_fit.isel({dim: i}),
                color="red",
                linewidth=1.2,
                label="Fit",
            )

            if cfg.plot_residuals:
                # Offset residuals dynamically based on the spectrum max
                offset = spec_raw.isel({dim: i}).max().values * 0.2
                ax.plot(
                    freq_coords,
                    spec_res.isel({dim: i}) - offset,
                    color="green",
                    alpha=0.6,
                    linewidth=1.0,
                )

            # Check CRLB Quality
            crlbs = ds_subset["crlb"].isel({dim: i}).values
            max_crlb = np.nanmax(np.nan_to_num(crlbs, nan=np.inf))

            # In-plot Annotation (Top-Left corner)
            coord_val = dim_coords[idx_val]
            if isinstance(coord_val, (int, np.integer)):
                label_text = f"{coord_val}{dim_unit}"
            else:
                label_text = f"{coord_val:.1f}{dim_unit}"

            if max_crlb > cfg.crlb_threshold:
                ax.set_facecolor(cfg.fail_color)
                text_color = "darkred"
                if np.isinf(max_crlb):
                    label_text += "\nCRLB: NaN"
                else:
                    label_text += f"\nCRLB: {max_crlb:.1f}%"
            else:
                text_color = "black"

            # Add stylish box behind text so it's readable over peaks
            ax.text(
                0.04,
                0.94,
                label_text,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                color=text_color,
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

            # Apply Zooming (xlim / ylim)
            if cfg.xlim is not None:
                ax.set_xlim(cfg.xlim)
            elif not ax.xaxis_inverted():
                ax.invert_xaxis()

            if cfg.ylim is not None:
                ax.set_ylim(cfg.ylim)

            ax.set_yticks([])  # Hide y-axis ticks for a clean grid

        # Clean up empty subplots
        for j in range(n_plots, len(axes_flat)):
            axes_flat[j].axis("off")

        # Apply the locator only to the bottom row of axes where x-ticks are visible
        for ax in axes[-1, :]:
            # nbins limits total ticks, prune='both' removes the outermost edge ticks
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

        # Dynamic X-Axis Label based on the coordinate's actual name and units
        x_name = freq_dim.replace("_", " ").title()
        x_unit = spec_raw.coords[freq_dim].attrs.get("units", "Hz")
        fig.supxlabel(f"{x_name} [{x_unit}]", fontweight="bold", fontsize=12)

        # Adjust layout to remove outer white space
        fig.tight_layout()

    return fig
