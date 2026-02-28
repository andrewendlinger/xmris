"""
Core configuration and vocabulary definitions for xmris xarray objects.

This module defines the single source of truth for all metadata attributes,
dimensions, coordinates, and data variables expected by the xmris package.
"""


class XmrisTerm(str):
    """A string subclass that holds metadata attributes.

    This allows xarray to treat it as a standard dimension/coordinate name,
    while allowing developers to access `.unit` and `.description` directly.
    """

    def __new__(cls, value: str, description: str = "", unit: str = ""):
        """Create a new :class:`XmrisTerm` instance with metadata.

        Parameters
        ----------
        value : str
            The string value to use for the term.
        description : str, optional
            A human‑readable description of the term (default is empty).
        unit : str, optional
            The unit associated with the term, if any (default is empty).

        Returns
        -------
        XmrisTerm
            A new string instance with ``description`` and ``unit`` attributes.
        """
        obj = str.__new__(cls, value)
        obj.description = description
        obj.unit = unit
        return obj

    @property
    def long_name(self) -> str:
        """Automatically generates a display-friendly long name.

        Example: 'chemical_shift' -> 'Chemical Shift'
        """
        return self.replace("_", " ").title()


class BaseVocabulary:
    """
    Base class for xmris xarray vocabularies.

    Provides rich HTML display for Jupyter Notebooks and utility
    methods to fetch metadata for validation decorators.
    """

    def _get_terms(self) -> dict:
        """Help extract all XmrisTerm attributes from the class."""
        return {
            key: val
            for key, val in vars(self.__class__).items()
            if isinstance(val, XmrisTerm)
        }

    def get_description(self, target_value: str) -> str:
        """
        Fetch the description for a given xarray key value.

        Used by the validation decorators to build dynamic docstrings.

        Parameters
        ----------
        target_value : str
            The actual string value of the attribute/dimension/coordinate
            (e.g., "MHz", "time").

        Returns
        -------
        str
            The description string, or a fallback message if not found.
        """
        for term in self._get_terms().values():
            if term == target_value:
                return term.description or "No description provided."
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
            "<div style='font-family: sans-serif; max-width: 900px;'>",
            f"<h3 style='margin-bottom: 4px;'>{cls_name}</h3>",
            f"<p style='margin-top: 0; color: #555;'><em>{desc_text}</em></p>",
            "<table style='width: 100%; border-collapse: collapse; text-align: left;'>",
            "<tr style='border-bottom: 2px solid #ccc;'>",
            "<th style='padding: 8px;'>Property</th>",
            "<th style='padding: 8px;'>xarray String Key</th>",
            "<th style='padding: 8px;'>Unit</th>",
            "<th style='padding: 8px;'>Description</th>",
            "</tr>",
        ]

        for prop_name, term in self._get_terms().items():
            # Format unit cleanly: bold if present, faint dash if empty
            unit_str = (
                f"<strong>{term.unit}</strong>"
                if term.unit
                else "<span style='color: #999;'>-</span>"
            )

            html.append(
                "<tr style='border-bottom: 1px solid #eee;'>"
                f"<td style='padding: 8px; white-space: nowrap;'><code>{prop_name}</code></td>"  # noqa: E501
                f"<td style='padding: 8px; white-space: nowrap;'><strong><code>\"{term}\"</code></strong></td>"  # noqa: E501
                f"<td style='padding: 8px; white-space: nowrap;'>{unit_str}</td>"
                f"<td style='padding: 8px;'>{term.description}</td>"
                "</tr>"
            )

        html.append("</table></div>")
        return "".join(html)


class XmrisAttributes(BaseVocabulary):
    """Official metadata attribute keys for xmris xarray objects (`.attrs`)."""

    reference_frequency = XmrisTerm(
        "reference_frequency",
        description=(
            "The measured Larmor frequency of the target nucleus. This reflects the"
            "actual B0 field during the scan, not a theoretical constant. It serves as"
            "the denominator to convert frequency shifts (Hz) to parts-per-million (ppm)."
            "Maps to Bruker 'PVM_FrqRef' and potentially DICOM 'ImagingFrequency'"
            "(0018,0084) or 'TransmitterFrequency' (0018,9098)."
        ),
        unit="MHz",
    )

    carrier_ppm = XmrisTerm(
        "carrier_ppm",
        description=(
            "The absolute chemical shift at the center of the RF excitation bandwidth. "
            "In the digitized baseband signal, this is the exact chemical shift located"
            "at 0 Hz. For standard 1H MRS, this is typically water (4.7 ppm). Maps to"
            "Bruker 'PVM_FrqWorkPpm'."
        ),
        unit="ppm",
    )

    # --- Phase Parameters ---
    phase_p0 = XmrisTerm(
        "phase_p0", description="Zero-order phase angle applied.", unit="degrees"
    )
    phase_p1 = XmrisTerm(
        "phase_p1", description="First-order phase angle applied.", unit="degrees"
    )

    # --- Apodization Parameters ---
    apodization_lb = XmrisTerm(
        "apodization_lb", description="Line broadening factor applied.", unit="Hz"
    )
    apodization_gb = XmrisTerm(
        "apodization_gb", description="Gaussian broadening factor applied.", unit="Hz"
    )

    # --- Zero Fill Parameters ---
    zero_fill_target = XmrisTerm(
        "zero_fill_target", description="Total number of points after zero-filling."
    )
    zero_fill_position = XmrisTerm(
        "zero_fill_position", description="Position of padding ('end' or 'symmetric')."
    )


class XmrisDimensions(BaseVocabulary):
    """Official dimension names for xmris xarray objects (`.dims`)."""

    time = XmrisTerm(
        "time", description="Time-domain dimension for Free Induction Decay (FID) data."
    )

    frequency = XmrisTerm(
        "frequency", description="Frequency-domain dimension for spectral data."
    )

    metabolite = XmrisTerm(
        "metabolite", description="Dimension representing quantified metabolites."
    )

    component = XmrisTerm(
        "component", description="Dimension separating real and imaginary parts."
    )
    # --- Standard Acquisition Dimensions ---
    average = XmrisTerm(
        "average", description="Dimension for multiple signal acquisitions/averages."
    )
    coil = XmrisTerm("coil", description="Dimension for multi-coil phased array data.")
    echo = XmrisTerm("echo", description="Dimension for multi-echo acquisitions.")

    # --- Spatial Frequency (k-space) ---
    kx = XmrisTerm("kx", description="k-space frequency dimension along the x-axis.")
    ky = XmrisTerm("ky", description="k-space frequency dimension along the y-axis.")
    kz = XmrisTerm("kz", description="k-space frequency dimension along the z-axis.")

    # --- Image Space ---
    x = XmrisTerm("x", description="Image space dimension along the x-axis.")
    y = XmrisTerm("y", description="Image space dimension along the y-axis.")
    z = XmrisTerm("z", description="Image space dimension along the z-axis (slice).")


class XmrisCoordinates(BaseVocabulary):
    """Official coordinate names for xmris xarray objects (`.coords`)."""

    time = XmrisTerm("time", description="Time coordinates.", unit="s")

    frequency = XmrisTerm("frequency", description="Frequency coordinates.", unit="Hz")

    chemical_shift = XmrisTerm(
        "chemical_shift", description="Chemical shift coordinates.", unit="ppm"
    )

    # --- Spatial Frequency (k-space) ---
    kx = XmrisTerm("kx", description="k-space coordinates along x.", unit="1/m")
    ky = XmrisTerm("ky", description="k-space coordinates along y.", unit="1/m")
    kz = XmrisTerm("kz", description="k-space coordinates along z.", unit="1/m")

    # --- Image Space ---
    x = XmrisTerm("x", description="Spatial coordinates along x.", unit="mm")
    y = XmrisTerm("y", description="Spatial coordinates along y.", unit="mm")
    z = XmrisTerm("z", description="Spatial coordinates along z.", unit="mm")


class XmrisDataVars(BaseVocabulary):
    """Official data variable names for xmris xarray Datasets (`.data_vars`)."""

    # --- Time/Frequency Domain Data ---
    original_data = XmrisTerm(
        "data", description="The original experimental data (FID or Spectrum)."
    )

    fit = XmrisTerm(
        "fit", description="The reconstructed time-domain or frequency-domain fit."
    )

    residuals = XmrisTerm(
        "residuals", description="The difference between the original data and the fit."
    )

    baseline = XmrisTerm(
        "baseline", description="The calculated baseline of the spectrum."
    )

    # --- Quantified Parameters ---
    amplitude = XmrisTerm("amplitude", description="Fitted peak amplitude.")

    chem_shift = XmrisTerm("chem_shift", description="Fitted chemical shift.", unit="ppm")

    linewidth = XmrisTerm(
        "linewidth", description="Fitted linewidth (damping factor).", unit="Hz"
    )

    phase = XmrisTerm("phase", description="Fitted phase.", unit="degrees")

    crlb = XmrisTerm(
        "crlb", description="Cramer-Rao Lower Bound (fitting error estimation).", unit="%"
    )

    snr = XmrisTerm("snr", description="Signal-to-Noise Ratio.")


# =============================================================================
# Global Singletons
# =============================================================================
ATTRS = XmrisAttributes()
DIMS = XmrisDimensions()
COORDS = XmrisCoordinates()
VARS = XmrisDataVars()
