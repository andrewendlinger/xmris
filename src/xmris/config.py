import warnings
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


# 1. Rename the instance to be private
_DEFAULTS = XmrisConfig()


# 2. Use module-level __getattr__ to intercept access to DEFAULTS
def __getattr__(name):
    if name == "DEFAULTS":
        warnings.warn(
            "The `DEFAULTS` configuration object is deprecated and will be removed "
            "in a future release. Please use the new singletons `ATTRS`, `DIMS`, "
            "`COORDS`, and `VARS` from `xmris.core.config` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEFAULTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 3. (Optional) Define __dir__ so autocomplete tools still see DEFAULTS
def __dir__():
    return ["Dimension", "Attribute", "XmrisConfig", "DEFAULTS"]
