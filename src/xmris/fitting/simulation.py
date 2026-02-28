import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

# Assuming your core vocabulary is accessible here
from xmris.core import ATTRS, COORDS, DIMS


def _simulate_fid_ndarray(
    amplitudes: ArrayLike,
    *,
    frequencies: ArrayLike | None = None,
    chemical_shifts: ArrayLike | None = None,
    reference_frequency: float | None = None,
    carrier_ppm: float = 0.0,
    spectral_width: float = 10000.0,
    n_points: int = 1024,
    dampings: float | ArrayLike = 50.0,
    phases: float | ArrayLike = 0.0,
    lineshape_g: float | ArrayLike = 0.0,
    dead_time: float = 0.0,
) -> np.ndarray:
    """Internal pure-NumPy implementation of the AMARES FID simulation.

    This function avoids xarray overhead for use inside iterative fitting loops.
    It generates a time-domain FID signal based on Equation 6 of the AMARES
    algorithm (Vanhamme, L. et al., J Magn Reson 1997, 129 (1), 35-43).

    Parameters
    ----------
    amplitudes : ArrayLike
        The amplitudes (a_k) of the peaks.
    frequencies : ArrayLike | None, optional
        The frequencies (f_k) of the peaks in Hz.
    chemical_shifts : ArrayLike | None, optional
        The chemical shifts of the peaks in ppm. Must be accompanied by
        `reference_frequency`.
    reference_frequency : float | None, optional
        The spectrometer operating frequency in MHz.
    carrier_ppm : float, optional
        The transmitter carrier frequency in ppm. The observable frequency window
        is centered around this value to prevent spectral aliasing (wrapping).
        Default is 0.0.
    spectral_width : float, optional
        The spectral width in Hz. Default is 10000.0.
    n_points : int, optional
        The number of data points (N). Default is 1024.
    dampings : float | ArrayLike, optional
        The damping factor(s) (d_k). Default is 50.0.
    phases : float | ArrayLike, optional
        The phase(s) (phi_k) in radians. Default is 0.0.
    lineshape_g : float | ArrayLike, optional
        The lineshape parameter(s) (g_k) between 0 (Lorentzian) and 1 (Gaussian).
    dead_time : float, optional
        The time origin offset in seconds. Default is 0.0.

    Returns
    -------
    numpy.ndarray
        A 1D complex array representing the combined time-domain FID signal.
    """  # noqa: D401
    amplitudes = np.atleast_1d(amplitudes)
    n_peaks = len(amplitudes)

    if frequencies is not None and chemical_shifts is not None:
        raise ValueError("Provide either 'frequencies' or 'chemical_shifts', not both.")
    elif chemical_shifts is not None:
        if reference_frequency is None:
            raise ValueError(
                "reference_frequency (MHz) must be provided when using chemical shifts."
            )
        chemical_shifts = np.atleast_1d(chemical_shifts)
        # Shift the peaks relative to the carrier before converting to Hz
        freqs = (chemical_shifts - carrier_ppm) * reference_frequency
    elif frequencies is not None:
        freqs = np.atleast_1d(frequencies)
    else:
        raise ValueError("Either 'frequencies' or 'chemical_shifts' must be provided.")

    if len(freqs) != n_peaks:
        raise ValueError("Length of frequencies/chemical_shifts must match amplitudes.")

    dampings = np.broadcast_to(dampings, n_peaks)
    phases = np.broadcast_to(phases, n_peaks)
    g_arr = np.clip(np.broadcast_to(lineshape_g, n_peaks), 0.0, 1.0)

    dwelltime = 1.0 / spectral_width
    t = np.arange(0, dwelltime * n_points, dwelltime) + dead_time
    t_col = t[:, np.newaxis]

    complex_phase = np.exp(1j * phases)
    decay = np.exp(-dampings * (1 - g_arr + g_arr * t_col) * t_col)
    oscillation = np.exp(1j * 2 * np.pi * freqs * t_col)

    fid_matrix = amplitudes * complex_phase * decay * oscillation
    return np.sum(fid_matrix, axis=1)


def simulate_fid(
    amplitudes: ArrayLike,
    *,
    frequencies: ArrayLike | None = None,
    chemical_shifts: ArrayLike | None = None,
    reference_frequency: float | None = None,
    carrier_ppm: float = 0.0,
    spectral_width: float = 10000.0,
    n_points: int = 1024,
    dampings: float | ArrayLike = 50.0,
    phases: float | ArrayLike = 0.0,
    lineshape_g: float | ArrayLike = 0.0,
    dead_time: float = 0.0,
    target_snr: float | None = None,
) -> xr.DataArray:
    """Simulate a complex Free Induction Decay (FID) signal.

    Returns a formatted array.DataArray compliant with xmris.core vocabularies.

    This function relies on the AMARES algorithm formulation. The generated data
    is a time-domain signal, meaning its primary dimension and coordinate will
    always be `time`. Simulation parameters (like the input ppm/Hz peaks and noise
    targets) are preserved in the DataArray's attributes for downstream tracking.

    If `target_snr` is provided, complex Gaussian white noise is added to the
    ideal signal. The total noise variance is split equally between the real and
    imaginary receiver channels to physically mimic quadrature detection.

    Parameters
    ----------
    amplitudes : ArrayLike
        The amplitudes (a_k) of the peaks.
    frequencies : ArrayLike | None, optional
        The frequencies (f_k) of the peaks in Hz.
    chemical_shifts : ArrayLike | None, optional
        The chemical shifts of the peaks in ppm. Must be accompanied by
        `reference_frequency`.
    reference_frequency : float | None, optional
        The spectrometer operating frequency in MHz. Maps to ATTRS.reference_frequency.
    carrier_ppm : float, optional
        The transmitter carrier frequency in ppm. The observable frequency window
        is centered around this value to prevent spectral aliasing. Default is 0.0.
    spectral_width : float, optional
        The spectral width in Hz. Determines the dwell time. Default is 10000.0.
    n_points : int, optional
        The number of data points (N). Default is 1024.
    dampings : float | ArrayLike, optional
        The damping factor(s) (d_k). Default is 50.0.
    phases : float | ArrayLike, optional
        The phase(s) (phi_k) in radians. Default is 0.0.
    lineshape_g : float | ArrayLike, optional
        The lineshape parameter(s) (g_k) between 0 (Lorentzian) and 1 (Gaussian).
    dead_time : float, optional
        The time origin offset in seconds. Default is 0.0.
    target_snr : float | None, optional
        The target Signal-to-Noise Ratio. If provided, complex Gaussian white
        noise is added to the FID. Signal power is calculated from the first
        10 points of the FID. Default is None (returns ideal, noiseless FID).

    Returns
    -------
    xarray.DataArray
        A 1D DataArray containing the complex FID signal, dimensioned by `DIMS.time`,
        with coordinates `COORDS.time`, and rich simulation metadata.
    """
    # 1. Compute the high-performance raw data (ideal signal)
    fid_data = _simulate_fid_ndarray(
        amplitudes=amplitudes,
        frequencies=frequencies,
        chemical_shifts=chemical_shifts,
        reference_frequency=reference_frequency,
        carrier_ppm=carrier_ppm,
        spectral_width=spectral_width,
        n_points=n_points,
        dampings=dampings,
        phases=phases,
        lineshape_g=lineshape_g,
        dead_time=dead_time,
    )

    # 2. Add Complex Gaussian White Noise if requested
    if target_snr is not None:
        # Calculate signal power based on the first 10 points (legacy pyAMARES behavior)
        # Bounding the slice safely in case n_points < 10
        signal_slice = fid_data[0 : min(10, n_points)]
        signal_p = np.mean(np.abs(signal_slice))

        # Calculate standard deviation for total noise
        noise_std_total = signal_p / target_snr

        # Split variance for complex noise: scale std by 1/sqrt(2)
        noise_std_channel = noise_std_total / np.sqrt(2)
        # use the new Generator API instead of legacy np.random functions
        rng = np.random.default_rng()
        noise_real = rng.normal(0, noise_std_channel, fid_data.shape)
        noise_imag = rng.normal(0, noise_std_channel, fid_data.shape)

        fid_data = fid_data + (noise_real + 1j * noise_imag)

    # 3. Reconstruct the physical time axis
    dwelltime = 1.0 / spectral_width
    time_coords = np.arange(0, dwelltime * n_points, dwelltime) + dead_time

    # 4. Build compliant metadata attributes
    attrs = {
        "spectral_width": spectral_width,
        "dead_time": dead_time,
        "sim_amplitudes": np.atleast_1d(amplitudes).tolist(),
        "sim_dampings": np.atleast_1d(dampings).tolist(),
        ATTRS.carrier_ppm: carrier_ppm,
        "units": "a.u.",
    }

    if target_snr is not None:
        attrs["target_snr"] = target_snr

    if reference_frequency is not None:
        attrs[ATTRS.reference_frequency] = reference_frequency

    if frequencies is not None:
        attrs["sim_frequencies_hz"] = np.atleast_1d(frequencies).tolist()
    if chemical_shifts is not None:
        attrs["sim_chemical_shifts_ppm"] = np.atleast_1d(chemical_shifts).tolist()

    # 5. Construct and return the xarray object
    return xr.DataArray(
        data=fid_data,
        dims=[DIMS.time],
        coords={
            COORDS.time: (DIMS.time, time_coords, {"units": "s", "long_name": "Time"})
        },
        attrs=attrs,
        name="FID Signal",
    )
