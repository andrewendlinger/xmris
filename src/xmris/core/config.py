"""
Core configuration and vocabulary definitions for xmris xarray objects.

This module defines the single source of truth for all metadata attributes,
dimensions, coordinates, and data variables expected by the xmris package.
"""


class XmrisTerm(str):
    """A string subclass that holds metadata attributes.

    This allows xarray to treat it as a standard dimension/coordinate name,
    while allowing developers to access `.unit` and `.description` directly.
    Instances are immutable: all metadata is fixed at construction time.
    """

    description: str
    unit: str
    aliases: tuple[str, ...]

    def __new__(
        cls,
        value: str,
        description: str = "",
        unit: str = "",
        aliases: tuple[str, ...] = (),
    ):
        """Create a new :class:`XmrisTerm` instance with metadata.

        Parameters
        ----------
        value : str
            The string value to use for the term.
        description : str, optional
            A human‑readable description of the term (default is empty).
        unit : str, optional
            The unit associated with the term, if any (default is empty).
        aliases : tuple of str, optional
            Legacy string values this term was previously known under (default
            is empty). Readers resolve the canonical value first, then each
            alias in order — see :func:`xmris.core.utils.read_attr`.

        Returns
        -------
        XmrisTerm
            A new string instance with ``description``, ``unit`` and
            ``aliases`` attributes.
        """
        obj = str.__new__(cls, value)
        # Instances are frozen (__setattr__ raises), so bypass it here.
        object.__setattr__(obj, "description", description)
        object.__setattr__(obj, "unit", unit)
        object.__setattr__(obj, "aliases", tuple(str(a) for a in aliases))
        return obj

    def __setattr__(self, name: str, value: object) -> None:
        """Reject attribute mutation — terms are frozen at construction."""
        raise AttributeError(
            f"XmrisTerm is immutable: cannot set {name!r} on {str(self)!r}. "
            "Define a new term in xmris.core.config instead."
        )

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion — terms are frozen at construction."""
        raise AttributeError(f"XmrisTerm is immutable: cannot delete {name!r} from {str(self)!r}.")

    def __getnewargs_ex__(self) -> tuple[tuple[str], dict[str, object]]:
        """Route pickle/copy through ``__new__`` so metadata survives round-trips."""
        return (
            (str(self),),
            {"description": self.description, "unit": self.unit, "aliases": self.aliases},
        )

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

    def __init_subclass__(cls, **kwargs) -> None:
        """Enforce key uniqueness (canonical values and aliases) at import time.

        Within one vocabulary class, every canonical term value and every alias
        must be distinct — a duplicate would make attribute lookups ambiguous.
        Cross-vocabulary duplicates (e.g. ``time`` in both DIMS and COORDS)
        remain intentionally allowed.
        """
        super().__init_subclass__(**kwargs)
        seen: dict[str, str] = {}
        for prop, term in vars(cls).items():
            if not isinstance(term, XmrisTerm):
                continue
            keys = [(str(term), "canonical value")] + [(a, "alias") for a in term.aliases]
            for key, kind in keys:
                if key in seen:
                    raise ValueError(
                        f"{cls.__name__}: duplicate vocabulary key {key!r} — "
                        f"{kind} of {prop!r} collides with the {seen[key]}."
                    )
                seen[key] = f"{kind} of {prop!r}"

    def _get_terms(self) -> dict:
        """Help extract all XmrisTerm attributes from the class."""
        return {key: val for key, val in vars(self.__class__).items() if isinstance(val, XmrisTerm)}

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
        for term in self._get_terms().values():
            if target_value in term.aliases:
                return f"Legacy alias of {str(term)!r}. " + (
                    term.description or "No description provided."
                )
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

            key_str = f'<strong><code>"{term}"</code></strong>'
            if term.aliases:
                legacy = ", ".join(f'"{a}"' for a in term.aliases)
                key_str += f"<br><small style='color: #999;'>legacy: {legacy}</small>"

            html.append(
                "<tr style='border-bottom: 1px solid #eee;'>"
                f"<td style='padding: 8px; white-space: nowrap;'><code>{prop_name}</code></td>"  # noqa: E501
                f"<td style='padding: 8px; white-space: nowrap;'>{key_str}</td>"
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
            "The measured Larmor frequency of the target nucleus. This reflects the "
            "actual B0 field during the scan, not a theoretical constant. It serves as "
            "the denominator to convert frequency shifts (Hz) to parts-per-million (ppm)."
            " Maps to Bruker 'PVM_FrqRef' and potentially DICOM 'ImagingFrequency'"
            "(0018,0084) or 'TransmitterFrequency' (0018,9098)."
        ),
        unit="MHz",
        aliases=("MHz",),
    )

    carrier_ppm = XmrisTerm(
        "carrier_ppm",
        description=(
            "The absolute chemical shift at the center of the RF excitation bandwidth. "
            "In the digitized baseband signal, this is the exact chemical shift located "
            "at 0 Hz. For standard 1H MRS, this is typically water (4.7 ppm). Maps to "
            "Bruker 'PVM_FrqWorkPpm'."
        ),
        unit="ppm",
    )

    group_delay = XmrisTerm(
        "group_delay",
        description=(
            "The receiver digital-filter group delay, in samples. Modern consoles "
            "oversample then decimate through an FIR filter cascade, delaying the FID "
            "by this many points; it must be removed before the Fourier transform to "
            "avoid a first-order phase roll. Vendor-reported and, for some ParaVision/"
            "probe combinations, an under-count of the true delay (see "
            "`estimate_group_delay`). Maps to Bruker 'ACQ_RxFilterInfo'[0].groupDelay "
            "(TopSpin 'GRPDLY')."
        ),
        unit="samples",
        aliases=("bruker_group_delay",),
    )

    group_delay_removed = XmrisTerm(
        "group_delay_removed",
        description=(
            "Lineage: the digital-filter group delay actually removed from the FID by "
            "`remove_digital_filter`, in samples. For `group_delay='measure'` this is the "
            "sole record of the auto-measured value."
        ),
        unit="samples",
    )

    # --- mostly used for demo of attributes ---
    b0_field = XmrisTerm("b0_field", description="Magnetic field strength B0", unit="Tesla")

    # --- Phase Parameters ---
    phase_p0 = XmrisTerm(
        "phase_p0",
        description=(
            "Zero-order (frequency-independent) phase angle applied uniformly across the spectrum."
        ),
        unit="degrees",
    )
    phase_p1 = XmrisTerm(
        "phase_p1",
        description=(
            "First-order (frequency-dependent) phase angle. "
            "Represents the total phase twist applied across the entire spectral sweep "
            "width, anchored at the pivot."
        ),
        unit="degrees",
    )
    phase_pivot = XmrisTerm(
        "phase_pivot",
        description=(
            "Coordinate value where the first-order phase contribution evaluates to exactly 0.0."
        ),
        unit="dimension-dependent",
    )
    phase_pivot_coord = XmrisTerm(
        "phase_pivot_coord",
        description="The coordinate dimension in which the phase pivot was defined.",
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
    # --- Baseline Correction Parameters ---
    baseline_method = XmrisTerm(
        "baseline_method",
        description="The algorithm used to estimate and remove the spectral baseline.",
    )
    baseline_lam = XmrisTerm(
        "baseline_lam",
        description=(
            "The smoothness penalty (lambda) applied during Asymmetric Least Squares "
            "(AsLS) baseline correction. Higher values yield stiffer baselines."
        ),
    )
    baseline_p = XmrisTerm(
        "baseline_p",
        description=(
            "The asymmetry parameter applied during AsLS baseline correction. "
            "Controls how aggressively the solver ignores positive absorption peaks."
        ),
    )
    baseline_iter = XmrisTerm(
        "baseline_iter",
        description="The number of sparse solver iterations used to calculate the baseline.",
    )


class XmrisDimensions(BaseVocabulary):
    """Official dimension names for xmris xarray objects (`.dims`)."""

    time = XmrisTerm(
        "time", description="Time-domain dimension for Free Induction Decay (FID) data."
    )

    frequency = XmrisTerm(
        "frequency",
        description=(
            "Relative frequency dimension in Hertz (Hz). Generated directly by the "
            "Fourier transform, or calculated from chemical shift using the "
            "`carrier_ppm` and `reference_frequency` (MHz) attributes."
        ),
    )

    chemical_shift = XmrisTerm(
        "chemical_shift",
        description=(
            "Absolute chemical shift dimension in parts-per-million (ppm). "
            "Calculated mathematically from the relative frequency (Hz) by dividing "
            "by `reference_frequency` (MHz) and adding the `carrier_ppm` offset."
        ),
    )

    metabolite = XmrisTerm(
        "metabolite", description="Dimension representing quantified metabolites."
    )

    component = XmrisTerm("component", description="Dimension separating real and imaginary parts.")
    # --- Standard Acquisition Dimensions ---
    # Dimension names are uniformly singular; legacy plural spellings are aliases.
    average = XmrisTerm(
        "average",
        description="Dimension for multiple signal acquisitions/averages.",
        aliases=("averages",),
    )
    repetition = XmrisTerm(
        "repetition",
        description=(
            "Dimension for repeated acquisitions over time (dynamic series), "
            "one entry per TR block."
        ),
        aliases=("repetitions",),
    )
    coil = XmrisTerm(
        "coil",
        description="Dimension for multi-coil phased array data.",
        aliases=("channels",),
    )
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

    fit = XmrisTerm("fit", description="The reconstructed time-domain or frequency-domain fit.")

    residuals = XmrisTerm(
        "residuals", description="The difference between the original data and the fit."
    )

    baseline = XmrisTerm("baseline", description="The calculated baseline of the spectrum.")

    # --- Quantified Parameters ---
    amplitude = XmrisTerm("amplitude", description="Fitted peak amplitude.")

    chem_shift = XmrisTerm("chem_shift", description="Fitted chemical shift.", unit="ppm")

    linewidth = XmrisTerm("linewidth", description="Fitted linewidth (damping factor).", unit="Hz")

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


# =============================================================================
# Domain Groupings
# =============================================================================
# Semantic groupings of *existing* dimension terms (not new vocabulary) used by
# the ``ensures_domain`` and ``computes_in`` domain decorators in
# ``validation.py``. They bundle the already-defined ``DIMS`` into the two
# physical domains xmris operates in: time-domain FIDs vs. frequency-domain
# spectra, where the spectral axis may be labelled in Hz (``frequency``) or
# ppm (``chemical_shift``).
TIME_DIMS = frozenset({DIMS.time})
SPECTRAL_DIMS = frozenset({DIMS.frequency, DIMS.chemical_shift})
