# 1. Import the accessor so it registers with xarray immediately
from . import accessor as accessor

# 2. Optionally, expose core functions at the top level for users who
# prefer functional programming over method chaining.
from .signal import fftc, ifftc

__all__ = [
    "fftshift",
    "ifftshift",
    "fft",
    "ifft",
    "fftc",
    "ifftc",
]
