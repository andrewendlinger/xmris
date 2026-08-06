---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (xmris)
  language: python
  name: python3
---

(bruker-grpdly)=
# Bruker — The Digital Filter Group Delay

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

# Crisp retina output + sane default DPI for the rendered docs
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150
```

If you have ever loaded a raw Bruker FID and found it starts with a strange, wavy flat stretch instead of a sharp spike — or watched an uncorrected spectrum spin like a corkscrew — you have met the **digital-filter group delay**. It is not a bad acquisition; it is a predictable byproduct of how the console digitizes and filters the signal.

This page explains where the delay comes from, removes it with `remove_digital_filter`, and — when the vendor header value turns out to be wrong — *measures* the true delay from the data with `estimate_group_delay`.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

import xmris  # registers the .xmr accessor
from xmris.fitting.simulation import simulate_fid
```

(bruker-grpdly-pipeline)=
## 1. The hardware pipeline

Modern Bruker consoles (the AVANCE NEO and relatives) are fully digital receivers. After the coil and preamplifier the signal is mixed down to an intermediate frequency and handed to a fast **oversampling ADC** (hundreds of MHz). The digital stream is then **down-converted to a complex I/Q baseband** — digital quadrature detection — and **decimated** to your requested sweep width through a cascade of CIC and FIR filters. The exact sampling rate and IF are hardware-dependent.

```{mermaid}
graph LR
    A["RF Coil"] --> B["Preamplifier"]
    B --> C["Analog Mix<br>to IF"]
    C --> D["Oversampling<br>ADC"]
    D --> E["Digital Down-Conversion<br>(I/Q quadrature)"]
    E --> F["CIC + FIR<br>Decimation"]
    F --> G["Raw FID<br>(shifted, delayed)"]

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
```

:::{note}
Older write-ups label the real→complex step a "Hilbert transform." Bruker's pipeline actually performs **digital down-conversion**: it mixes the real ADC stream with a numerically-controlled oscillator, then low-pass filters and decimates. A Hilbert transform is a different, perfectly valid way to build a complex (analytic) signal — just not the one the console runs.
:::

The decimation stage is where the delay is born.

(bruker-grpdly-constant-delay)=
## 2. Why the group delay is constant

The FIR filters are designed to be **linear-phase** — their impulse response is symmetric. Linear phase means a **constant group delay**: every frequency in the FID is held back by the *same* number of points, so the signal's shape survives intact and is merely shifted in time.

:::{dropdown} Deep dive: group delay and linear phase
Group delay is $-\,\mathrm{d}\varphi/\mathrm{d}\omega$, the derivative of a filter's phase response. When that derivative is *constant*, the phase is **linear** in frequency and every spectral component is delayed by the same amount. A symmetric FIR of length $L$ has exactly this property, with a group delay of $(L-1)/2$ samples. (A CIC filter is symmetric too, so the whole chain stays linear-phase.)

If the delay were *not* constant — high frequencies emerging before low ones — the lineshape would smear and distort. Bruker avoids that by construction; the price is a single, well-defined time shift that we can undo exactly.
:::

But a causal filter cannot emit its centre tap until its window has filled, so the output ramps up through a short, ringing transient before the true signal arrives. That is the "wavy flatline" at the start of a raw FID.

A pure time shift of $d$ samples is a **linear phase** across the spectrum. Over the full sweep width the phase winds through a complete $2\pi$ turn **$d$ times** — for a typical high-resolution delay ($d\approx 76$) that is roughly 76 turns of first-order phase. Left uncorrected, the real spectrum is an unusable corkscrew.

::: {seealso}
[Phase Correction](#phase) derives the zero- and first-order phase "twist" this section relies on. [The FFT](#fft) and [FID Transformations](#fid-transforms) cover the Fourier transform and the $t=0$ convention.
:::

(bruker-grpdly-simulate)=
## 3. Simulate a raw Bruker FID

To see the effect cleanly we build our own "raw" FID: an ideal multi-peak signal from `simulate_fid`, pushed through a symmetric windowed-sinc FIR (the integer delay + startup transient) plus a fractional sub-point phase ramp. Three resonances are spread across the sweep width so the frequency-dependent phase error is visible.

```{code-cell} ipython3
:tags: [hide-input]

def make_raw_fid(peaks_hz, amps, delay, *, sw=5000.0, n=2048, damping=30.0, header=None):
    """Mimic a raw Bruker acquisition.

    Convolve an ideal multi-peak FID with a symmetric (linear-phase) windowed-sinc
    FIR to inject an integer group delay and its startup transient, then add the
    fractional sub-point delay via a Fourier phase ramp. ``header`` sets the value
    the console *reports* (which need not equal the true ``delay``).
    """
    ideal = simulate_fid(
        amplitudes=amps,
        frequencies=peaks_hz,
        spectral_width=sw,
        n_points=n,
        dampings=damping,
        reference_frequency=32.09,
        carrier_ppm=171.0,
    )
    int_delay = int(np.floor(delay))
    frac = delay - int_delay
    taps = 2 * int_delay + 1
    k = np.arange(taps)
    fir = np.sinc(0.5 * (k - int_delay)) * np.hamming(taps)  # symmetric -> linear phase
    fir /= fir.sum()
    raw = np.convolve(ideal.values, fir, mode="full")[:n]
    if frac:  # fractional sub-point delay via a Fourier phase ramp
        freqs = np.fft.fftfreq(n)
        raw = np.fft.ifft(np.fft.fft(raw) * np.exp(-1j * 2 * np.pi * freqs * frac))
    da = ideal.copy(data=raw)
    if header is not None:
        da.attrs["group_delay"] = header  # what the console reports
    return da
```

```{code-cell} ipython3
peaks_hz = [20.0, 436.0, 651.0]  # three resonances across the sweep width
amps = [1.0, 0.6, 0.8]

# A well-behaved acquisition: the reported delay matches the true one.
raw = make_raw_fid(peaks_hz, amps, delay=76.125, header=76.125)

fig, ax = plt.subplots(figsize=(8, 3.2))
raw.real.plot(ax=ax, label="raw (real)")
ax.set_xlim(0, 0.03)
ax.set_title("Raw Bruker FID — the filter transient before the true signal")
ax.legend()
plt.show()
```

(bruker-grpdly-correct)=
## 4. Correct it with `remove_digital_filter`

`remove_digital_filter` undoes the delay in three moves: it **slices off** the integer part of the delay (the transient points), applies a **first-order phase** ramp for the leftover fractional sub-point, and — with `keep_length=True` — **zero-pads** the tail so the array length (and FFT radix) is unchanged.

Because the Bruker loader stores the reported delay in `attrs`, the default `group_delay="header"` simply reads it:

```{code-cell} ipython3
clean = raw.xmr.remove_digital_filter()  # group_delay="header" reads the stored value
# equivalent to: raw.xmr.remove_digital_filter(group_delay=76.125)
```

```{code-cell} ipython3
fig, (ax_start, ax_end) = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)

raw.real.plot(ax=ax_start, alpha=0.5, label="raw")
clean.real.plot(ax=ax_start, lw=2, label="corrected")
ax_start.set_xlim(0, 0.03)
ax_start.set_title("FID start — transient removed")
ax_start.legend()

raw.real.plot(ax=ax_end, alpha=0.5, label="raw")
clean.real.plot(ax=ax_end, lw=2, label="corrected")
ax_end.set_xlim(0.39, 0.41)
ax_end.set_title("FID tail — zero-padded")
ax_end.legend()

plt.tight_layout()
plt.show()
```

A time shift is a linear phase, so the naive real spectrum is a corkscrew. After removal the peaks are cleanly absorptive:

```{code-cell} ipython3
spec_naive = raw.xmr.to_spectrum()
spec_clean = clean.xmr.to_spectrum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
spec_naive.real.plot(ax=ax1, color="tab:red")
ax1.set_title("Naive FT (uncorrected)\nfirst-order phase corkscrew")
spec_clean.real.plot(ax=ax2, color="tab:blue")
ax2.set_title("After remove_digital_filter\nclean absorption")
plt.tight_layout()
plt.show()
```

:::{note}
The corrected FID's first point sits a little below the ideal sum of amplitudes — the FIR smooths the synthetic's abrupt onset. That rounding is a detail of *this simulation*, not an effect of the group delay itself.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: remove_digital_filter (header default)
# Purity + lineage: a new object, original attrs survive, the removed delay is recorded.
assert clean is not raw, "function mutated data in place"
assert set(raw.attrs).issubset(clean.attrs), "original attrs were dropped"
assert clean.attrs["group_delay_removed"] == 76.125

# Coordinates: length preserved (keep_length default), time reset to 0, tail zero-filled.
assert clean.sizes["time"] == raw.sizes["time"], "keep_length failed to preserve length"
np.testing.assert_allclose(
    clean.coords["time"].values[0], 0.0, err_msg="time coordinate not reset to 0"
)
# Regression for #83: the origin reset must keep the coordinate's units metadata
# (a bare-array assign_coords dropped it, mislabelling the time axis downstream).
assert clean.coords["time"].attrs.get("units") == "s", (
    "time coordinate lost its units metadata during remove_digital_filter"
)
np.testing.assert_allclose(
    clean.values[-76:], 0.0, atol=1e-9, err_msg="integer-delay tail not zero-filled"
)

# The "header" default reads the attr -> identical to passing the float explicitly.
_explicit = raw.xmr.remove_digital_filter(group_delay=76.125)
np.testing.assert_allclose(
    clean.values, _explicit.values, err_msg="header default != explicit float"
)

# Math: every peak is absorptive (real-dominant) after correction, no autophase needed.
for _f in peaks_hz:
    _seg = spec_clean.sel(frequency=slice(_f - 40, _f + 40))
    _pk = _seg.values[int(np.abs(_seg).values.argmax())]
    assert _pk.real > 0 and abs(_pk.imag) < 0.4 * _pk.real, f"peak at {_f} Hz not absorptive"
```

(bruker-grpdly-measure)=
## 5. When the header lies: `estimate_group_delay`

So far the header told the truth. For some ParaVision / probe combinations, though, the reported delay **under-counts** the real one — the console removes fewer transient points than it inserted. (This is an empirically-observed caveat, not documented Bruker behaviour.) What is left behind is a few samples of un-removed transient, i.e. a residual **first-order phase**

$$\varphi(f) \;=\; \varphi_0 \;-\; 2\pi\,\Delta d\,\frac{f}{f_s}\,,$$

zero at the carrier but growing with offset, so peaks far from the centre are quietly twisted — enough to bias peak areas and tied-phase fits.

`estimate_group_delay` finds the delay that removes it: the value that makes the *whole* spectrum absorptive under a **single** zero-order phase. (`argmax(|FID|)` is not usable — it lands on the transient's ringing, not the true delay.)

```{code-cell} ipython3
# A pathological acquisition: true delay 84, but the console reports only 76.125.
bad = make_raw_fid(peaks_hz, amps, delay=84.0, header=76.125)

# Reads the header as its search anchor; warns because the measurement contradicts it.
measured, profile = bad.xmr.estimate_group_delay(return_profile=True)

print(f"reported (header): {bad.attrs['group_delay']}")
print(f"measured (true)  : {measured:.2f} samples")
```

The cost profile has a single sharp minimum at the true delay. The header sits ~8 samples short of it, and `argmax(|FID|)` lands on a ringing lobe:

```{code-cell} ipython3
argmax_fid = int(np.argmax(np.abs(bad.values)))

fig, ax = plt.subplots(figsize=(8, 3.6))
profile.plot(ax=ax, marker=".", color="tab:blue", label="residual-phase cost")
ax.axvline(measured, color="tab:green", lw=2, label=f"measured ({measured:.1f})")
ax.axvline(76.125, color="tab:red", ls="--", label="header (76.125)")
ax.axvline(argmax_fid, color="0.5", ls=":", label=f"argmax|FID| ({argmax_fid})")
ax.set_xlabel("trial group delay (samples)")
ax.set_ylabel("residual-phase cost")
ax.set_title("estimate_group_delay: cost vs. trial delay")
ax.legend()
plt.show()
```

Remove each delay, transform, and apply **only** a zero-order phase (`p0_only=True`) so any residual *first-order* phase stays visible. The header value leaves the far peaks twisted; the measured value is absorptive everywhere. In a pipeline, `group_delay="measure"` does both steps at once.

```{code-cell} ipython3
spec_header = (
    bad.xmr.remove_digital_filter(group_delay=76.125)
    .xmr.to_spectrum()
    .xmr.autophase(p0_only=True)
)
spec_measured = (
    bad.xmr.remove_digital_filter(group_delay="measure")  # measure, then remove
    .xmr.to_spectrum()
    .xmr.autophase(p0_only=True)
)

fig, (axh, axm) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
spec_header.real.plot(ax=axh, color="tab:red")
axh.set_title("Header delay (76.125)\nfar peaks twisted")
spec_measured.real.plot(ax=axm, color="tab:blue")
axm.set_title("Measured delay (~84)\nabsorptive everywhere")
for ax in (axh, axm):
    for _f in peaks_hz:
        ax.axvline(_f, color="0.8", lw=0.7, zorder=0)
plt.tight_layout()
plt.show()
```

:::{note}
In the header panel the **436 Hz** peak is badly twisted, yet the **651 Hz** peak looks almost fine — at that offset the residual phase happens to wrap by nearly a full $2\pi$. A naive check on the wrong peak pair would wrongly declare the delay correct; `estimate_group_delay` scores the **whole** spectrum, so it is not fooled by this aliasing.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: estimate_group_delay
from xmris.vendor.bruker import _PHI0_GRID, _residual_phase_cost

_TRUE = 84.0


def _resid(d):
    """Whole-spectrum residual first-order phase cost after removing delay ``d``."""
    _s = bad.xmr.remove_digital_filter(group_delay=d).xmr.to_spectrum()
    return _residual_phase_cost(_s, "acme", _PHI0_GRID)


def _peak_phase_deg(d, f0, half=40.0):
    _s = bad.xmr.remove_digital_filter(group_delay=d).xmr.to_spectrum()
    _seg = _s.sel(frequency=slice(f0 - half, f0 + half))
    return np.degrees(np.angle(_seg.values[int(np.abs(_seg).values.argmax())]))


def _wrap(x):
    return (x + 180.0) % 360.0 - 180.0


# Recovery: sub-sample precision, profile minimum on the truth.
np.testing.assert_allclose(measured, _TRUE, atol=0.5, err_msg="did not recover true delay")
assert abs(float(profile.trial_delay[int(profile.argmin())]) - _TRUE) <= 1.0

# Measured beats the header on whole-spectrum residual phase.
assert _resid(measured) < 0.3 * _resid(76.125), "measured did not beat the header"

# argmax(|FID|) is unreliable: different integer, far worse residual.
_argmax = int(np.argmax(np.abs(bad.values)))
assert _resid(float(_argmax)) > 5.0 * _resid(measured), "argmax not clearly worse than measured"

# Aliasing: the wrong header exposes the mid peak but the far peak looks benign.
assert abs(_wrap(_peak_phase_deg(76.125, 436.0) - _peak_phase_deg(76.125, 20.0))) > 50.0
assert abs(_wrap(_peak_phase_deg(76.125, 651.0) - _peak_phase_deg(76.125, 20.0))) < 30.0

# With the measured delay, all peaks share one phase (no first-order residual).
assert abs(_wrap(_peak_phase_deg(measured, 436.0) - _peak_phase_deg(measured, 20.0))) < 40.0

# Lineage: the "measure" sentinel records the measured (not the header) delay.
_meas_attr = bad.xmr.remove_digital_filter(group_delay="measure").attrs["group_delay_removed"]
np.testing.assert_allclose(_meas_attr, _TRUE, atol=0.5)
```
