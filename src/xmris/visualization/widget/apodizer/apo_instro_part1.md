**System Role / Objective:**
You are an expert in Digital Signal Processing (DSP) and JavaScript performance optimization. Your task is to translate a specific set of Python (NumPy/xarray) Magnetic Resonance Spectroscopy (MRS) mathematical functions into **pure, dependency-free JavaScript**.

I am building a web-based UI, and I need a standalone JavaScript math module that perfectly replicates my Python backend's output.

you must output just the javascript as one large code snippet.

**Requirements:**

1. **No External Libraries:** Do not use `math.js`, `numeric.js`, or any other dependencies. Use standard JavaScript `Math` and TypedArrays (`Float64Array`).
2. **Complex Number Handling:** Since JS lacks a native complex type, represent complex 1D arrays by passing and returning two separate `Float64Array` objects (`real` and `imag`).
3. **Apodization Functions:** Implement JS equivalents of `apodize_exp` and `apodize_lg`. They should take `real`, `imag`, a time coordinate array `t`, and the parameters (`lb`, `gb`), and return the modified arrays.
4. **Shifting (`fftshift` / `ifftshift`):** Implement 1D array shifting logic that exactly mimics `numpy.roll`.
* `fftshift` shifts by `Math.floor(N / 2)`
* `ifftshift` shifts by `Math.floor((N + 1) / 2)`


5. **Fast Fourier Transform (`fft_ortho`):** Write a custom Cooley-Tukey `fft_radix2` function.
* You can assume the input array length `N` is strictly a power of 2 (this is enforced on the Python side).
* Scale the final FFT output by `1 / Math.sqrt(N)` so that it perfectly matches NumPy's `norm="ortho"` energy preservation.
* do the inverse, ifft, too.

6. **The Centered FFT Pipeline (`fftc`):** Create a wrapper function that executes the exact standard NMR pipeline: `ifftshift` -> `fft_ortho` -> `fftshift`.
* do the `ifftc` too

**Context:**
Below is the exact Python implementation you need to mirror. Please output the clean, documented JavaScript code.

---

1. **From `processing/fid.py` (Apodization):**

```python
def to_spectrum(
    da: xr.DataArray, dim: str = DIMS.time, out_dim: str = DIMS.frequency
) -> xr.DataArray:
    """
    Convert a time-domain Free Induction Decay (FID) to a frequency-domain spectrum.

    The stored, digital FID signal can be processed by a discrete Fourier
    transformation (DFT) to produce a digital MR spectrum.
    This function applies the FFT along the specified time dimension and shifts
    the zero-frequency component to the center of the spectrum.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain FID data.
    dim : str, optional
        The time dimension to transform, by default `DIMS.time`.
    out_dim : str, optional
        The name of the resulting frequency dimension, by default `DIMS.frequency`.

    Returns
    -------
    xr.DataArray
        The frequency-domain spectrum with centered zero-frequency coordinates.
    """
    _check_dims(da, dim, "to_spectrum")

    # 1. Standard FFT (handles transform and creates unshifted frequency coords)
    da_freq = fft(da, dim=dim, out_dim=out_dim)

    # 2. Shift the frequency domain to center the DC component
    da_spectrum = fftshift(da_freq, dim=out_dim)

    return da_spectrum


def to_fid(
    da: xr.DataArray, dim: str = DIMS.frequency, out_dim: str = DIMS.time
) -> xr.DataArray:
    """
    Convert a frequency-domain spectrum back to a time-domain FID.

    This is the mathematical inverse of `to_spectrum`. It inverse-shifts the
    data to position 0 Hz at the array boundary, computes the IFFT, and
    reconstructs strictly positive time coordinates.

    Parameters
    ----------
    da : xr.DataArray
        The input frequency-domain data.
    dim : str, optional
        The frequency dimension to transform, by default `DIMS.frequency`.
    out_dim : str, optional
        The name of the resulting time dimension, by default `DIMS.time`.

    Returns
    -------
    xr.DataArray
        The reconstructed time-domain FID.
    """
    _check_dims(da, dim, "to_fid")

    # 1. Inverse shift the frequency domain to put the DC component at index 0
    # This prepares the data for the standard IFFT algorithm
    da_unshifted = ifftshift(da, dim=dim)

    # 2. Apply IFFT
    # The output is naturally ordered [t=0, t=1, ... t=N-1]
    da_fid = ifft(da_unshifted, dim=dim, out_dim=out_dim)

    # 3. Reconstruct the strictly positive time coordinates [0, T_acq]
    if dim in da.coords:
        freqs = da.coords[dim].values
        n_points = len(freqs)
        if n_points > 1:
            # Calculate dwell time (dt) based on the sampling theorem:
            # Spectral Width (SW) = 1/dt
            # SW = n_points * df
            df = abs(freqs[1] - freqs[0])
            dt = 1.0 / (n_points * df)

            t_coords = np.arange(n_points) * dt

            # Re-inject metadata if mapping to standard DIMS.time
            term = COORDS.time if out_dim == DIMS.time else None

            if term:
                time_var = as_variable(term, out_dim, t_coords)
            else:
                time_var = xr.Variable(out_dim, t_coords)

            da_fid = da_fid.assign_coords({out_dim: time_var})

    return da_fid


def apodize_exp(da: xr.DataArray, dim: str = DIMS.time, lb: float = 1.0) -> xr.DataArray:
    """
    Apply an exponential weighting filter function for line broadening.

    During apodization, the time-domain FID signal $f(t)$ is multiplied with a filter
    function $f_{filter}(t) = e^{-t/T_L}$. This improves the Signal-to-Noise Ratio (SNR)
    because data points at the end of the FID, which primarily contain noise, are
    attenuated. The time constant $T_L$ is calculated from the desired line broadening
    in Hz.


    Parameters
    ----------
    da : xr.DataArray
        The input time-domain data.
    dim : str, optional
        The dimension corresponding to time, by default `DIMS.time`.
    lb : float, optional
        The desired line broadening factor in Hz, by default 1.0.

    Returns
    -------
    xr.DataArray
        A new apodized DataArray, preserving coordinates and attributes.
    """
    _check_dims(da, dim, "apodize_exp")

    t = da.coords[dim]

    # Calculate exponential filter: exp(-t / T_L) where T_L = 1 / (pi * lb)
    # This simplifies to: exp(-pi * lb * t)
    weight = np.exp(-np.pi * lb * t)

    # Functional application (transpose ensures broadcasting doesn't scramble axis order)
    da_apodized = (da * weight).transpose(*da.dims).assign_attrs(da.attrs)

    # Record lineage
    da_apodized.attrs[ATTRS.apodization_lb] = lb

    return da_apodized


def apodize_lg(
    da: xr.DataArray, dim: str = DIMS.time, lb: float = 1.0, gb: float = 1.0
) -> xr.DataArray:
    """
    Apply a Lorentzian-to-Gaussian transformation filter.

    This filter converts a Lorentzian line shape to a Gaussian line shape, which decays
    to the baseline in a narrower frequency range. The time-domain FID
    is multiplied by $e^{+t/T_L}e^{-t^2/T_G^2}$. The time constants $T_L$ and $T_G$
    are derived from the `lb` and `gb` frequency-domain parameters.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain data.
    dim : str, optional
        The dimension corresponding to time, by default `DIMS.time`.
    lb : float, optional
        The Lorentzian line broadening to cancel in Hz, by default 1.0.
    gb : float, optional
        The Gaussian line broadening to apply in Hz, by default 1.0.

    Returns
    -------
    xr.DataArray
        A new apodized DataArray, preserving coordinates and attributes.
    """
    _check_dims(da, dim, "apodize_lg")

    t = da.coords[dim]

    # Calculate Lorentzian cancellation: exp(+t / T_L)
    # T_L = 1 / (pi * lb)
    weight_lorentzian = np.exp(np.pi * lb * t)

    # Calculate Gaussian broadening: exp(-t^2 / T_G^2)
    # T_G = 2 * sqrt(ln(2)) / (pi * gb)
    if gb != 0:
        t_g = (2 * np.sqrt(np.log(2))) / (np.pi * gb)
        weight_gaussian = np.exp(-(t**2) / (t_g**2))
    else:
        weight_gaussian = 1.0

    weight = weight_lorentzian * weight_gaussian

    da_apodized = (da * weight).transpose(*da.dims).assign_attrs(da.attrs)

    # Record lineage
    da_apodized.attrs[ATTRS.apodization_lb] = lb
    da_apodized.attrs[ATTRS.apodization_gb] = gb

    return da_apodized
```


2. **From your core/math modules (FFT & Shifting):**

```python
import numpy as np
import xarray as xr

from xmris.core.config import COORDS, DIMS, XmrisTerm
from xmris.core.utils import _check_dims, as_variable

# --- 1. Shifting Utilities ---


def fftshift(da: xr.DataArray, dim: str | list[str]) -> xr.DataArray:
    """
    Apply fftshift by rolling data and coordinates along the given dimension(s).

    This shifts the zero-frequency component to the center of the spectrum.

    Parameters
    ----------
    da : xr.DataArray
        The input xarray DataArray.
    dim : str or list of str
        The dimension(s) along which to apply the shift.

    Returns
    -------
    xr.DataArray
        A new DataArray with the data and coordinates rolled.
    """
    dims = [dim] if isinstance(dim, str) else dim
    _check_dims(da, dims, "fftshift")

    shifts = {d: da.sizes[d] // 2 for d in dims}
    return da.roll(shifts, roll_coords=True)


def ifftshift(da: xr.DataArray, dim: str | list[str]) -> xr.DataArray:
    """
    Apply ifftshift by rolling data and coordinates along the given dimension(s).

    This is the exact inverse of `fftshift`, moving the zero-frequency component
    from the center back to the original position.

    Parameters
    ----------
    da : xr.DataArray
        The input xarray DataArray.
    dim : str or list of str
        The dimension(s) along which to apply the inverse shift.

    Returns
    -------
    xr.DataArray
        A new DataArray with the data and coordinates rolled.
    """
    dims = [dim] if isinstance(dim, str) else dim
    _check_dims(da, dims, "ifftshift")

    shifts = {d: (da.sizes[d] + 1) // 2 for d in dims}
    return da.roll(shifts, roll_coords=True)


# --- 2. Coordinate Math ---


def _convert_fft_coords(
    da: xr.DataArray, dim: str, out_dim: str | None = None, term: XmrisTerm | None = None
) -> xr.DataArray:
    """
    Calculate unshifted reciprocal coordinates and safely rebuild the DataArray.

    Computes the standard discrete Fourier Transform sample frequencies
    (or time periods) and assigns them to the transformed dimension. If an
    `XmrisTerm` is provided, it injects the associated unit and long_name metadata.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray (already transformed in the data domain).
    dim : str
        The original dimension name that was transformed.
    out_dim : str, optional
        The new name for the transformed dimension. If None, the original
        dimension name is kept.
    term : XmrisTerm, optional
        The configuration term (e.g., `COORDS.frequency`) used to inject
        physical metadata (units, long_name) into the new coordinate.

    Returns
    -------
    xr.DataArray
        A DataArray with updated reciprocal coordinates and renamed dimensions.
    """
    n_points = da.sizes[dim]
    old_coords = da.coords[dim].values

    delta = (old_coords[1] - old_coords[0]) if len(old_coords) > 1 else 1.0

    # Calculate UNSHIFTED reciprocal coordinates (frequencies or time periods)
    new_coords = np.fft.fftfreq(n_points, d=delta)

    target_dim = out_dim if out_dim is not None else dim

    # Use the helper to bundle data and metadata if we have a known XmrisTerm
    if term is not None:
        new_var = as_variable(term, target_dim, new_coords)
    else:
        new_var = xr.Variable(target_dim, new_coords)

    if out_dim is not None and out_dim != dim:
        da = da.rename({dim: out_dim})

    return da.assign_coords({target_dim: new_var})


# --- 3. Pure Transforms ---


def fft(
    da: xr.DataArray,
    dim: str | list[str] = DIMS.time,
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """
    Perform an N-dimensional Fast Fourier Transform (FFT).

    Applies an ortho-normalized, unshifted FFT. Metadata and unaffected
    dimensions are strictly preserved.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain DataArray.
    dim : str or list of str, optional
        The dimension(s) to transform. Defaults to `DIMS.time`.
    out_dim : str or list of str, optional
        The resulting dimension name(s). Must match the length of `dim`.
        If None, the original dimension names are retained.

    Returns
    -------
    xr.DataArray
        The frequency-domain DataArray with updated reciprocal coordinates.
    """
    dims = [dim] if isinstance(dim, str) else dim
    _check_dims(da, dims, "fft")

    # Handle dimension name mapping
    out_dims = [out_dim] if isinstance(out_dim, str) else out_dim
    if out_dims is not None and len(dims) != len(out_dims):
        raise ValueError("`dim` and `out_dim` lists must have the same length.")

    # 1. Perform standard numpy FFT
    axes = tuple(da.get_axis_num(d) for d in dims)
    arr_fft = np.fft.fftn(da.values, axes=axes, norm="ortho")

    # Safely rebuild DataArray to preserve untouched lineage/attrs exactly
    da_transformed = da.copy(data=arr_fft)

    # 2. Assign unshifted reciprocal coordinates
    for i, d in enumerate(dims):
        o_dim = out_dims[i] if out_dims else None

        # Smart mapping: If converting standard time, automatically apply
        # frequency metadata
        term = (
            COORDS.frequency
            if (d == DIMS.time and o_dim in (None, DIMS.frequency))
            else None
        )
        da_transformed = _convert_fft_coords(
            da_transformed, dim=d, out_dim=o_dim, term=term
        )

    return da_transformed


def ifft(
    da: xr.DataArray,
    dim: str | list[str] = DIMS.frequency,
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """
    Perform an N-dimensional Inverse Fast Fourier Transform (IFFT).

    Applies an ortho-normalized, unshifted IFFT. Metadata and unaffected
    dimensions are strictly preserved.

    Parameters
    ----------
    da : xr.DataArray
        The input frequency-domain DataArray.
    dim : str or list of str, optional
        The dimension(s) to transform. Defaults to `DIMS.frequency`.
    out_dim : str or list of str, optional
        The resulting dimension name(s). Must match the length of `dim`.
        If None, the original dimension names are retained.

    Returns
    -------
    xr.DataArray
        The time-domain DataArray with updated reciprocal coordinates.
    """
    dims = [dim] if isinstance(dim, str) else dim
    _check_dims(da, dims, "ifft")

    out_dims = [out_dim] if isinstance(out_dim, str) else out_dim
    if out_dims is not None and len(dims) != len(out_dims):
        raise ValueError("`dim` and `out_dim` lists must have the same length.")

    axes = tuple(da.get_axis_num(d) for d in dims)
    arr_ifft = np.fft.ifftn(da.values, axes=axes, norm="ortho")

    da_transformed = da.copy(data=arr_ifft)

    for i, d in enumerate(dims):
        o_dim = out_dims[i] if out_dims else None

        # Smart mapping: If converting standard frequency, automatically apply
        # time metadata
        term = (
            COORDS.time if (d == DIMS.frequency and o_dim in (None, DIMS.time)) else None
        )
        da_transformed = _convert_fft_coords(
            da_transformed, dim=d, out_dim=o_dim, term=term
        )

    return da_transformed


# --- 4. Centered Transforms (Convenience Wrappers) ---


def fftc(
    da: xr.DataArray,
    dim: str | list[str] = DIMS.time,
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """
    Perform a centered N-dimensional FFT.

    This executes an `ifftshift -> fft -> fftshift` pipeline, which is the
    standard in MRI/MRS processing to ensure the 0 Hz frequency remains at
    the center of the spectral axis.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain DataArray.
    dim : str or list of str, optional
        The dimension(s) to transform. Defaults to `DIMS.time`.
    out_dim : str or list of str, optional
        The resulting dimension name(s). If None, keeps the original names.

    Returns
    -------
    xr.DataArray
        The centered, frequency-domain DataArray.
    """
    new_dims = out_dim if out_dim is not None else dim

    return (
        ifftshift(da, dim=dim)
        .pipe(fft, dim=dim, out_dim=out_dim)
        .pipe(fftshift, dim=new_dims)
    )


def ifftc(
    da: xr.DataArray,
    dim: str | list[str] = DIMS.frequency,
    out_dim: str | list[str] | None = None,
) -> xr.DataArray:
    """
    Perform a centered N-dimensional IFFT.

    This executes an `ifftshift -> ifft -> fftshift` pipeline, which correctly
    inverts a centered frequency-domain spectrum back to the time domain.

    Parameters
    ----------
    da : xr.DataArray
        The input centered frequency-domain DataArray.
    dim : str or list of str, optional
        The dimension(s) to transform. Defaults to `DIMS.frequency`.
    out_dim : str or list of str, optional
        The resulting dimension name(s). If None, keeps the original names.

    Returns
    -------
    xr.DataArray
        The centered, time-domain DataArray.
    """
    new_dims = out_dim if out_dim is not None else dim

    return (
        ifftshift(da, dim=dim)
        .pipe(ifft, dim=dim, out_dim=out_dim)
        .pipe(fftshift, dim=new_dims)
    )
```