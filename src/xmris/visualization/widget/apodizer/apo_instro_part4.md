---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
---

Next, write a demonstration and documentation notebook, similar to the reference notebook below. make sure to be objective and concise.

unlike the reference, use `simulate_fid` for simulating the fid.

YOU MUST NOT USE CANVAS. YOU MUST OUTPUT ONE COHERENT NOTEBOOK IN JUPYTEXT MYSTMD MARKDOWN FORMAT. YOU MUST NOT USE CANVAS.


# FID Simulation

you can use this snippet here:

```python
from xmris import simulate_fid

da_fid = simulate_fid(
    amplitudes=[10.0, 8.0],
    frequencies=[150.0, 165.0],
    dampings=[15.0, 15.0],
    reference_frequency=298.0,
    carrier_ppm=4.7,
    spectral_width=2000.0,
    n_points=2048,
    target_snr=15.0
)
```

based on the following function:

```python
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
```


# reference notebook

make a reference to the classical apodize_exp and apodize_lg functions explained in `../../pipeline/apodization.ipynb`.

```markdown
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150
```

While automated phasing algorithms (like ACME) are incredibly powerful, heavily distorted baselines, massive solvent peaks, or extreme noise can sometimes trick the optimizer. In these cases, falling back to manual phase correction is necessary.

As discussed in the [Phase Correction Pipeline](../../pipeline/phase.ipynb) documentation, applying a phase correction mathematically is straightforward. However, guessing the exact $p_0$ and $p_1$ angles blindly is nearly impossible.

To solve this, `xmris` provides an interactive, browser-based AnyWidget that allows you to click and drag to fix the phase in real-time, and then generates the exact Python code needed to reproduce your manual adjustments.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import xmris
```

## 1. Generating Unphased Data

Let's generate the same ruined synthetic spectrum we used in the pipeline documentation.

```{code-cell} ipython3
:tags: [hide-input]

# Generate synthetic FID
dwell_time = 0.001
n_points = 1024
t = np.arange(n_points) * dwell_time

rng = np.random.default_rng(42)
clean_fid = np.exp(-t / 0.05) * (
    np.exp(1j * 2 * np.pi * 50 * t) + 0.6 * np.exp(1j * 2 * np.pi * -150 * t)
)
noise = rng.normal(scale=0.08, size=n_points) + 1j * rng.normal(scale=0.08, size=n_points)

da_fid = xr.DataArray(
    clean_fid + noise, dims=["time"], coords={"time": t}
)
da_spec = da_fid.xmr.to_spectrum()

# Intentionally ruin the phase
da_ruined = da_spec.xmr.phase(p0=120.0, p1=-45.0)
```

## 2. Launching the Widget

You can launch the interactive viewer directly from the `xmris` package. Pass your complex-valued 1D frequency-domain `DataArray` to the `phase_spectrum` function.

The widget will automatically detect your spectral dimension (e.g., `frequency`, `ppm`), set up the coordinates, and calculate the optimal pivot point based on the maximum signal magnitude.

```{code-cell} ipython3
:tags: [remove-output]

# Launch the interactive widget
da_ruined.xmr.widget.phase_spectrum()
```

```{code-cell} ipython3
:tags: [remove-input]

from xmris.visualization.widget._static_exporter import export_widget_static
from xmris.visualization.widget.phase.phase import phase_spectrum

# This will render the interactive canvas in the docs!
export_widget_static(
    phase_spectrum,     # The widget generating function
    da_ruined,          # Positional arguments
    width=700,          # Keyword arguments
)
```

### Using the Widget

Once the widget is rendered in your notebook, you can interact with it using the following controls:

* **Zero-Order Phase ($p_0$):** Click and drag vertically on the canvas to adjust the global, frequency-independent phase.
* **First-Order Phase ($p_1$):** Hold `Shift` while clicking and dragging vertically. This twists the phase linearly across the spectrum, anchored perfectly at the `pivot` point (indicated by a small gray marker on the top axis).
* **Fine-Tuning:** You can manually type exact degree values into the input boxes in the control bar.
* **Visual Feedback:** The real component is rendered in blue, and the imaginary component is rendered in red. Your goal is usually to maximize the symmetry and positivity of the blue peaks while zeroing out the red dispersive twists at the peak centers.

## 3. Extracting the Parameters

Interactive widgets are great for exploration, but they are generally bad for reproducible science if the parameters stay trapped in the UI.

When you have achieved the desired phase, click the **Close** button in the widget control bar. The canvas will unmount, and the widget will generate a strict, reproducible code snippet reflecting your final parameters.

(Screenshot only)

![Screenshot of the widget (part 2)](../../../assets/notebook-assets/screenshot_widget_phase_spectrum_II.png)

+++

Click the **Copy Code** button, and paste it into the next cell in your notebook:

```{code-cell} ipython3
# This code is pasted directly from the widget's completion screen!
phased_da = da_ruined.xmr.phase(p0=-120.00, p1=45.00, pivot=50.000)

# Verify the correction
fig, ax = plt.subplots(figsize=(8, 3))
phased_da.real.plot(ax=ax, color="tab:blue")
plt.title("Spectrum Phased via Widget Parameters")
plt.show()
```

Because the `pivot` value is explicitly hardcoded into the generated snippet, your manual $p_1$ adjustment is guaranteed to produce the exact same mathematical output even if the array is later cropped, padded, or resampled!

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: Widget Output Integration
assert "phase_p0" in phased_da.attrs
assert "phase_p1" in phased_da.attrs
assert "phase_pivot" in phased_da.attrs

np.testing.assert_allclose(
    phased_da.values,
    da_spec.values,
    rtol=1e-3,
    atol=1e-3,
    err_msg="Widget phase parameters did not successfully invert the synthetic phase error.",
)
```

```


YOU MUST NOT USE CANVAS. YOU MUST OUTPUT ONE COHERENT NOTEBOOK IN JUPYTEXT MYSTMD MARKDOWN FORMAT. YOU MUST NOT USE CANVAS.
