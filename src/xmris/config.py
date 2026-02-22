from dataclasses import dataclass


@dataclass
class Dimension:
    """Defines an xarray Dimension, its optional Coordinates, and standard units."""

    dim: str
    coords: tuple[str, ...] | None = None
    units: str | None = None


@dataclass
class Attribute:
    """Defines a standard xarray Attribute (metadata key) and its expected units."""

    key: str
    units: str | None = None


class XmrisConfig:
    """Global configuration and standard nomenclature for xmris."""

    def __init__(self):
        # --- Dimensions & Units ---
        self.time = Dimension(dim="time", units="s")
        self.frequency = Dimension(dim="frequency", units="Hz")
        self.chemical_shift = Dimension(dim="chemical_shift", units="ppm")
        self.component = Dimension(dim="component", coords=("real", "imag"))

        # --- Metadata Attributes ---
        self.b0 = Attribute(key="B0", units="T")
        self.mhz = Attribute(key="MHz", units="MHz")
        self.te = Attribute(key="TE", units="s")
        self.tr = Attribute(key="TR", units="s")


# The single global instance users interact with
DEFAULTS = XmrisConfig()
