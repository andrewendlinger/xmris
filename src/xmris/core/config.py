"""
Core configuration and vocabulary definitions for xmris xarray objects.

This module defines the single source of truth for all metadata attributes,
dimensions, coordinates, and data variables expected by the xmris package.
"""

import dataclasses
from dataclasses import dataclass, field


class BaseVocabulary:
    """
    Base class for xmris xarray vocabularies.

    Provides rich HTML display for Jupyter Notebooks and utility
    methods to fetch metadata for validation decorators.
    """

    def get_description(self, target_value: str) -> str:
        """
        Fetch the description for a given xarray key value.

        Used by the validation decorators to build dynamic docstrings.

        Parameters
        ----------
        target_value : str
            The actual string value of the attribute/dimension/coordinate
            (e.g., "MHz", "Time").

        Returns
        -------
        str
            The description string, or a fallback message if not found.
        """
        for f in dataclasses.fields(self):
            if getattr(self, f.name) == target_value:
                return f.metadata.get("description", "No description provided.")
        return "Unknown xarray key."

    def _repr_html_(self) -> str:
        """
        Render a clean HTML table of the vocabulary for Jupyter Notebooks.

        Returns
        -------
        str
            HTML string representing the class fields and metadata.
        """
        cls_name = self.__class__.__name__
        doc = self.__class__.__doc__ or ""
        desc_text = doc.strip().split("\n")[0] if doc else f"Vocabulary for {cls_name}:"

        html = [
            "<div style='font-family: sans-serif; max-width: 800px;'>",
            f"<h3 style='margin-bottom: 4px;'>{cls_name}</h3>",
            f"<p style='margin-top: 0; color: #555;'><em>{desc_text}</em></p>",
            "<table style='width: 100%; border-collapse: collapse; text-align: left;'>",
            "<tr style='border-bottom: 2px solid #ccc;'>",
            "<th style='padding: 8px;'>Property</th>",
            "<th style='padding: 8px;'>xarray String Key</th>",
            "<th style='padding: 8px;'>Description</th>",
            "</tr>",
        ]

        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            desc = f.metadata.get("description", "")
            unit = f.metadata.get("unit", "")

            desc_str = f"{desc} <strong>[{unit}]</strong>" if unit else desc

            html.append(
                "<tr style='border-bottom: 1px solid #eee;'>"
                f"<td style='padding: 8px;'><code>{f.name}</code></td>"
                f"<td style='padding: 8px;'><strong><code>\"{val}\"</code></strong></td>"
                f"<td style='padding: 8px;'>{desc_str}</td>"
                "</tr>"
            )

        html.append("</table></div>")
        return "".join(html)


@dataclass(frozen=True)
class XmrisAttributes(BaseVocabulary):
    """Official metadata attribute keys for xmris xarray objects (`.attrs`)."""

    b0_field: str = field(
        default="b0_field",
        metadata={"description": "Static main magnetic field strength.", "unit": "T"},
    )
    reference_frequency: str = field(
        default="MHz",
        metadata={
            "description": "Spectrometer working/reference frequency.",
            "unit": "MHz",
        },
    )
    p0: str = field(
        default="p0",
        metadata={"description": "Zero-order phase angle.", "unit": "degrees"},
    )
    p1: str = field(
        default="p1",
        metadata={"description": "First-order phase angle.", "unit": "degrees"},
    )


@dataclass(frozen=True)
class XmrisDimensions(BaseVocabulary):
    """Official dimension names for xmris xarray objects (`.dims`)."""

    time: str = field(
        default="Time",
        metadata={
            "description": "Time-domain dimension for Free Induction Decay (FID) data."
        },
    )

    frequency: str = field(
        default="Frequency",
        metadata={"description": "Frequency-domain dimension for spectral data."},
    )

    metabolite: str = field(
        default="Metabolite",
        metadata={"description": "Dimension representing quantified metabolites."},
    )

    component: str = field(
        default="component",
        metadata={"description": "Dimension separating real and imaginary parts."},
    )


@dataclass(frozen=True)
class XmrisCoordinates(BaseVocabulary):
    """Official coordinate names for xmris xarray objects (`.coords`)."""

    time: str = field(
        default="Time", metadata={"description": "Time coordinates.", "unit": "s"}
    )
    frequency: str = field(
        default="Frequency",
        metadata={"description": "Frequency coordinates.", "unit": "Hz"},
    )
    ppm: str = field(
        default="ppm",
        metadata={"description": "Chemical shift coordinates.", "unit": "ppm"},
    )


@dataclass(frozen=True)
class XmrisDataVars(BaseVocabulary):
    """Official data variable names for xmris xarray Datasets (`.data_vars`)."""

    # --- Time/Frequency Domain Data ---
    original_data: str = field(
        default="data",
        metadata={"description": "The original experimental data (FID or Spectrum)."},
    )
    fit: str = field(
        default="fit",
        metadata={
            "description": "The reconstructed time-domain or frequency-domain fit."
        },
    )
    residuals: str = field(
        default="residuals",
        metadata={"description": "The difference between the original data and the fit."},
    )
    baseline: str = field(
        default="baseline",
        metadata={"description": "The calculated baseline of the spectrum."},
    )

    # --- Quantified Parameters ---
    amplitude: str = field(
        default="amplitude", metadata={"description": "Fitted peak amplitude."}
    )
    chem_shift: str = field(
        default="chem_shift",
        metadata={"description": "Fitted chemical shift.", "unit": "ppm"},
    )
    linewidth: str = field(
        default="linewidth",
        metadata={"description": "Fitted linewidth (damping factor).", "unit": "Hz"},
    )
    phase: str = field(
        default="phase", metadata={"description": "Fitted phase.", "unit": "degrees"}
    )
    crlb: str = field(
        default="crlb",
        metadata={
            "description": "Cramer-Rao Lower Bound (fitting error estimation).",
            "unit": "%",
        },
    )
    snr: str = field(default="snr", metadata={"description": "Signal-to-Noise Ratio."})


# =============================================================================
# Global Singletons
# =============================================================================
ATTRS = XmrisAttributes()
DIMS = XmrisDimensions()
COORDS = XmrisCoordinates()
VARS = XmrisDataVars()
