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

(pyamares)=

# AMARES Fitting in Depth

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

# Crisp retina output + sane default DPI for the rendered docs
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150
```

[Fitting one clean spectrum](#fitting-quickstart) is three lines. Fitting a real dataset raises
three questions that page never had to answer: what happens to the _other_ voxels, what is actually
inside the object that comes back, and how do you know which of those numbers to believe?

This page answers them on data built to misbehave — peaks that drift, signal that fades, a voxel
with nothing in it at all — using the same one-line call throughout.

:::{dropdown} Why AMARES?
From the pyAMARES [paper](https://doi.org/10.3390/diagnostics14232668):

AMARES models the MRS signal as a sum of exponentially damped sinusoids. It uses parameters such as
chemical shift, linewidth, amplitude, phase, and spectral lineshape, which can be constrained by
prior knowledge. This knowledge includes initial parameters, parameter ranges, and relationships
between different peaks and can be readily obtained from published literature. Peaks outside the
region of interest can be filtered out, and parameters without prior knowledge can be fitted.

In contrast, frequency-domain fitting methods like LCModel require all metabolites to be modeled as
basis set spectra. While this approach reduces the number of parameters to fit, it requires
additional effort to obtain basis set spectra through experiments or numerical simulations.
Moreover, frequency-domain fitting strategies typically require well-phased absorptive spectra.
AMARES circumvents the sometimes subjective and complicated phasing procedure, making it
particularly effective for analyzing data with distorted phases due to long receiver dead times.

LCModel and AMARES have been compared directly and proven to be comparable, each with its own
advantages. However, AMARES is often the preferred method for quantifying X-nuclei MRS data, such
as 13C and 31P MRS, where spectra typically exhibit fewer peaks and less J-coupling compared to 1H
MRS.
:::

| Function                                                                          | What it does here                                             |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| [`simulate_fid()`](#xmris.fitting.simulation.simulate_fid)                        | builds one voxel of synthetic ³¹P signal at a time            |
| [`build_prior_knowledge()`](#xmris.fitting.prior_knowledge.build_prior_knowledge) | validates the peak dict below — `fit_amares` calls it for you |
| [`.xmr.fit_amares()`](#xmris.core.accessor.XmrisAccessor.fit_amares)              | fits every voxel in one call, returns a `Dataset`             |
| [`.xmr.to_spectrum()`](#xmris.core.accessor.XmrisProcessingMixin.to_spectrum)     | FID → spectrum, across all voxels at once                     |
| [`.xmr.to_ppm()`](#xmris.core.accessor.XmrisSpectrumCoordsMixin.to_ppm)           | relabels the frequency axis in ppm                            |
| [`.xmr.to_hz()`](#xmris.core.accessor.XmrisSpectrumCoordsMixin.to_hz)             | and back again                                                |

```{code-cell} ipython3
import numpy as np
import pandas as pd
import xarray as xr

import xmris  # registers the .xmr accessor  # noqa: F401
from xmris.fitting.simulation import simulate_fid
```

(pyamares-data)=

## 1. A dataset worth fitting

A textbook spectrum is a bad test. Real spectroscopic imaging goes wrong in ways that are boring
individually and awkward together: **B0 inhomogeneity** shifts every peak in a voxel by a fraction
of a ppm, **coil sensitivity** falls off with distance so the noise floor is not the same twice,
and some voxels sit **outside the object** and contain nothing to fit.

So we simulate a row of eight voxels with all three. `simulate_fid` makes one voxel at a time —
each with its own concentration, its own frequency offset and its own noise level — and `xr.concat`
stacks them into the 2-D `(voxel, time)` array that fitting expects.

```{code-cell} ipython3
n_voxels = 8
mhz = 120.6  # 31P at 7 T

pcr_amplitude = np.linspace(10.0, 45.0, n_voxels)  # concentration rises along the row
b0_drift = np.linspace(-0.25, 0.25, n_voxels)  # ppm — the shim degrades across it
snr = np.geomspace(250.0, 4.0, n_voxels)  # coil sensitivity falls off with distance

voxels = [
    simulate_fid(
        amplitudes=[pcr_amplitude[i], 5.0],  # PCr varies, ATP does not
        chemical_shifts=[0.0 + b0_drift[i], -7.5 + b0_drift[i]],
        reference_frequency=mhz,
        spectral_width=8000.0,
        n_points=512,
        dampings=[np.pi * 15.0, np.pi * 20.0],
        target_snr=float(snr[i]),
        seed=i,  # a different noise draw per voxel, still reproducible
    )
    for i in range(n_voxels)
]

voxels[3] = xr.zeros_like(voxels[3])  # this one is outside the object: no signal at all

grid = xr.concat(voxels, dim="voxel").assign_coords(voxel=np.arange(n_voxels))
grid.attrs = {"reference_frequency": mhz, "carrier_ppm": 0.0}
grid
```

:::{note} `xr.concat` keeps only the _first_ voxel's attributes
Which is exactly wrong here — voxel 0's `target_snr` and `sim_amplitudes` describe one voxel, not
the stack. So we replace them with the two the fit actually needs: the spectrometer frequency and
the carrier position. Everything else (the dwell time, the number of points) `fit_amares` reads off
the `time` coordinate.
:::

Plotted, the three effects are hard to miss — and hard to fit:

```{code-cell} ipython3
spectra = grid.xmr.to_spectrum().xmr.to_ppm()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
spectra.real.plot.line(x="chemical_shift", hue="voxel", ax=axes[0], add_legend=False)
axes[0].set_title("The row: PCr grows, noise grows with it")
axes[0].set_xlim(4, -12)

spectra.real.plot.line(x="chemical_shift", hue="voxel", ax=axes[1])
axes[1].set_title("Zoom on PCr: every voxel sits somewhere else")
axes[1].set_xlim(0.6, -0.6)

for ax in axes:
    ax.set_ylabel("intensity [a.u.]")
plt.tight_layout()
plt.show()
```

(pyamares-fit)=

## 2. One call, every voxel

The prior knowledge is the same as for a single spectrum, with one adjustment: the default ±0.5 ppm
search window is generous here, but stating the window explicitly documents how much drift you
expect the fit to absorb.

```{code-cell} ipython3
pk = {
    "PCr": {
        "amplitude": 10.0,
        "chem_shift": 0.0,
        "linewidth": 15.0,
        "chem_shift_bounds": (-0.6, 0.6),
    },
    "ATP": {
        "amplitude": 5.0,
        "chem_shift": -7.5,
        "linewidth": 20.0,
        "chem_shift_bounds": (-8.2, -6.8),
    },
}
```

Now the part that would otherwise be a `for` loop. `fit_amares` flattens every non-time dimension
into a single list of spectra, scans it for the highest-SNR one to initialize the pyAMARES template
from, fits each in turn, and reassembles the results onto the axes you started with.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
graph LR
    A["DataArray<br>dims: voxel, time"] --> B("flatten to a list of spectra")
    B --> C{"fit each"}
    C -->|"spectrum 0"| D["AMARES fit<br>voxel 0"]
    C -->|"spectrum N"| E["AMARES fit<br>voxel N"]
    D --> F("reassemble on the original axes")
    E --> F
    F --> G["Dataset<br>signals on time<br>parameters on metabolite"]
```

```{code-cell} ipython3
ds = grid.xmr.fit_amares(pk)
ds
```

Two arguments worth knowing about, neither of them required:

- **`num_workers`** decides whether those fits are spread over worker processes. It defaults to `1`
  — everything in-process — because a pool costs a second or two to start, which eight 512-point
  spectra never earn back. On a real grid the arithmetic reverses: pass `-1` for every core, or
  `-2` to leave one free. The [diary entry](#diary-amares-fitting-workers) has the measurements,
  including the two very different break-even points, and the reason the _default_ stays serial.
- **`method`** picks the optimizer, and the default `"least_squares"` (SciPy's trust-region solver)
  is the one to keep. `"leastsq"` — Levenberg–Marquardt — is faster per fit, but on drifted data
  like this it has a second, shallower minimum it settles into unpredictably, so the same signal
  fitted twice can give amplitudes 22% apart with both fits reporting success. The
  [diary entry](#diary-amares-fitting-optimizer) has the measurements.

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: the fit ran over every voxel and produced the documented structure.
from pathlib import Path

from xmris import build_prior_knowledge

# 1. Every variable the sections below refer to is present.
for _v in (
    "data",
    "fit",
    "fit_components",
    "residuals",
    "amplitude",
    "chem_shift",
    "linewidth",
    "phase",
    "lineshape_g",
    "snr",
    "crlb",
    "sd",
    "fit_status",
):
    assert _v in ds.data_vars, f"{_v} missing from the fit Dataset"

# 2. The four shapes: summed signals, per-peak signals, values, uncertainties.
assert ds["fit"].dims == ("voxel", "time")
assert ds["fit_components"].dims == ("voxel", "metabolite", "time")
assert ds["amplitude"].dims == ("voxel", "metabolite")
assert ds["crlb"].dims == ("voxel", "metabolite", "parameter")
assert list(ds["parameter"].values) == ["amplitude", "chem_shift", "linewidth", "phase"]
assert list(ds["metabolite"].values) == ["PCr", "ATP"]

# 2b. The one invariant that could silently rot: the components ARE the fit, split
#     up. Anything that desynchronizes the two reconstructions breaks here.
#     skipna=False, or the empty voxel 3's all-NaN components would sum to a 0.0
#     that no longer matches `fit`'s NaN.
xr.testing.assert_allclose(ds["fit_components"].sum("metabolite", skipna=False), ds["fit"])

# 3. Regression for #68: the spectrometer frequency is resolved from the modern
#    `reference_frequency` attr, not the legacy "MHz" key. `grid` carries only the
#    former, and no explicit `mhz=` was passed -- this raised a ValueError before the fix.
assert "MHz" not in grid.attrs

# 4. The file route through `_resolve_pk_file` still works alongside the dict above.
_pk_path = Path("example_pk.csv")
_pk_path.write_text(build_prior_knowledge(pk))
try:
    _ds_file = grid.isel(voxel=0).xmr.fit_amares(_pk_path)
finally:
    _pk_path.unlink(missing_ok=True)
np.testing.assert_allclose(
    _ds_file["amplitude"].values,
    ds["amplitude"].isel(voxel=0).values,
    rtol=1e-3,
    err_msg="prior knowledge from a file disagreed with the same spec passed as a dict",
)
```

(pyamares-dataset)=

## 3. Anatomy of the returned Dataset

Fitting is the one xmris operation that does not hand back a `DataArray`. It cannot: a fit produces
signals that live on `time`, the same signals split per metabolite, quantities that live per
metabolite, and uncertainties that live per metabolite _and_ per parameter. Four different shapes,
one aligned container.

```{code-cell} ipython3
ds
```

| Shape                            | Variables                                                             | What it is                                                                         |
| -------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `(voxel, time)`                  | `data`, `fit`, `residuals`                                            | the signal you passed in, the model AMARES built, and what is left over            |
| `(voxel, metabolite, time)`      | `fit_components`                                                      | that same model _before_ the peaks were added together — one signal per metabolite |
| `(voxel, metabolite)`            | `amplitude`, `chem_shift`, `linewidth`, `phase`, `lineshape_g`, `snr` | the quantified answer, one number per peak per voxel                               |
| `(voxel, metabolite, parameter)` | `crlb`, `sd`                                                          | the uncertainty on _every_ fitted parameter, relative (%) and absolute             |
| `(voxel,)`                       | `fit_status`                                                          | whether that voxel was fitted at all                                               |

`fit_components` sits directly under `fit` because it is the next thing you reach for: `fit` tells
you whether the model matched the data, the components tell you _which peak_ accounts for which
part of it. They are not a second, independent reconstruction — summing them over `metabolite`
gives back `fit` exactly, which is what makes plotting them underneath the fit honest rather than
illustrative. Everything true of `fit` is true of them: same domain, same axis, same calibration
attrs, so `ds["fit_components"].xmr.to_ppm()` works wherever `ds["fit"].xmr.to_ppm()` does.

The point of the `metabolite` and `parameter` axes is that they carry names, so selection reads
like the question you are asking:

```{code-cell} ipython3
print("PCr amplitude, all voxels:", ds["amplitude"].sel(metabolite="PCr").values.round(2))
print(
    "its uncertainty (%)      :",
    ds["crlb"].sel(metabolite="PCr", parameter="amplitude").values.round(2),
)
print(
    "worst parameter per voxel:", ds["crlb"].max("parameter").sel(metabolite="PCr").values.round(2)
)
```

Lineage survives the fit, and the fit adds its own:

```{code-cell} ipython3
ds.attrs
```

`amares_amplitude_scale` is the single factor the whole array was divided by before fitting.
pyAMARES derives its optimizer tolerance from signal magnitude, so a Bruker-scale FID would
otherwise "converge" before the solver moved; xmris normalizes once, fits, and multiplies the
amplitudes back. Recording the factor keeps that auditable rather than hidden.

(pyamares-status)=

### Which voxels were actually fitted

Voxel 3 was empty. It could not be fitted, and the honest record of that is not a zero — a zero is
a _measurement_, and a downstream mean would fold it in as though somebody had measured nothing
there.

```{code-cell} ipython3
print("amplitude:", ds["amplitude"].sel(metabolite="PCr").values.round(2))
print("fit_status:", ds["fit_status"].values)
print(ds["fit_status"].attrs)
```

`NaN` says _there is no value here_. `fit_status` says why: `0` fitted, `1` no signal to fit,
`2` the solver was handed a real spectrum and failed on it. The first two are visible above; the
third is the one worth grepping for on real data, because it means a spectrum that looked fittable
was not.

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: absence is recorded as NaN + a status, never as a zero.
# 1. The empty voxel is NaN across every quantified variable.
assert np.all(np.isnan(ds["amplitude"].isel(voxel=3).values))
assert np.all(np.isnan(ds["crlb"].isel(voxel=3).values))
# 2. Its neighbours are finite, so the failure did not spread.
assert np.all(np.isfinite(ds["amplitude"].isel(voxel=[2, 4]).values))
# 3. fit_status separates the two kinds of absence, with CF-style metadata.
np.testing.assert_array_equal(ds["fit_status"].values, [0, 0, 0, 1, 0, 0, 0, 0])
assert ds["fit_status"].attrs["flag_meanings"] == "fitted no_signal failed"
# 4. The normalization factor was stamped, and it is the array-wide maximum.
np.testing.assert_allclose(
    ds.attrs["amares_amplitude_scale"],
    float(np.abs(grid).max()),
    err_msg="the recorded scale factor is not the array-wide magnitude",
)
```

(pyamares-quality)=

## 4. Is the fit trustworthy?

Every fitted voxel returns numbers. Whether those numbers _mean_ anything is a separate question,
and the `Dataset` carries three independent ways to answer it.

(pyamares-crlb)=

### The Cramér–Rao lower bound

`crlb` is the standard one. It is the theoretical floor on the uncertainty of a fitted parameter —
the best any unbiased algorithm could do given this data's noise and this model's constraints —
reported as a percentage of the fitted value.

:::{dropdown} Deep dive: where the CRLB comes from, and how to use it
Think of a standard NMR spectrum. If a peak is sharp (narrow linewidth) and your baseline is clean
(low noise), the fitting algorithm can pinpoint the peak's amplitude and position with high
precision. Conversely, if the signal is a broad lump buried in baseline noise, any estimate of its
area will carry significant uncertainty.

In MRS, the CRLB calculates the mathematical **"best-case scenario"** for this uncertainty. It
represents the absolute minimum variance (error) that _any_ unbiased fitting algorithm can possibly
achieve, based purely on data quality and model constraints.

Mathematically, the variance of your estimated amplitude ($\hat{A}$) will always be greater than or
equal to the CRLB variance:

$$\text{Var}(\hat{A}) \ge \text{CRLB}_{\text{var}}$$

In practice, we look at the standard deviation: $\sigma \ge \sqrt{\text{CRLB}_{\text{var}}}$.
pyAMARES estimates this theoretical floor by analyzing the
[Fisher Information Matrix](https://en.wikipedia.org/wiki/Fisher_information) alongside the noise
variance of your raw data.

**Comparing %CRLB values.** Be careful when comparing them across studies or peaks, as the CRLB is
highly sensitive to:

1. **Signal-to-noise ratio.** A massive Peak A might have a 2% CRLB, while a tiny Peak B has 15%.
   However, a 2% error on a huge peak can still represent a larger _absolute_ error than a 15%
   error on a small one.
2. **Linewidth and overlap.** Broad or heavily overlapping peaks increase statistical covariance
   (the algorithm struggles to unambiguously assign the signal), which intrinsically drives up the
   CRLB.
3. **Prior knowledge constraints.** Tightly constraining a fit (e.g. fixing linewidths or
   frequencies) restricts the algorithm's freedom, mathematically forcing the calculated CRLB down.

**The takeaway:** the historical standard of rejecting fits with a %CRLB > 20% is now discouraged
[](https://doi.org/10.1002/mrm.25568) (see also [](https://doi.org/10.1002/mrm.27742)). Because
%CRLB scales inversely with amplitude, discarding high-%CRLB fits disproportionately removes valid
low-concentration values. This introduces a "selection bias" that artificially inflates group
averages. Instead, use the %CRLB to **weight** statistical analyses, or treat the >20% threshold as
a flag to manually inspect spectra for severe artifacts.
:::

Both flavours come back: `crlb` as a percentage, `sd` as a standard deviation in the units of the
data. Relative numbers compare peaks with each other; absolute ones propagate into arithmetic.

```{code-cell} ipython3
atp = ds.sel(metabolite="ATP")
qc = pd.DataFrame(
    {
        "amplitude": atp["amplitude"].values,
        "snr": atp["snr"].values,
        "sd (abs)": atp["sd"].sel(parameter="amplitude").values,
        "crlb (%)": atp["crlb"].sel(parameter="amplitude").values,
    },
    index=pd.Index(ds["voxel"].values, name="voxel"),
)


def flag_crlb(row):
    """Green = fitted precisely; amber = worth a look; grey = nothing was fitted."""
    if pd.isna(row["crlb (%)"]):
        color = "background-color: rgba(128, 128, 128, 0.2)"
    elif row["crlb (%)"] > 20.0:
        color = "background-color: rgba(255, 170, 0, 0.25)"
    else:
        color = "background-color: rgba(0, 255, 0, 0.15)"
    return [color] * len(row)


qc.style.apply(flag_crlb, axis=1).format("{:.2f}", na_rep="—")
```

ATP's amplitude never changes — it is the same 5 a.u. in every voxel. Its _uncertainty_ grows by
more than two orders of magnitude down the column, tracking the noise, and the last voxels cross
the traditional 20% line. Nothing there is wrong with the fit; the data simply stopped supporting a
precise answer, and the CRLB is what says so.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 3.5))
for name, marker in (("PCr", "o"), ("ATP", "s")):
    ds["crlb"].sel(metabolite=name, parameter="amplitude").plot(ax=ax, marker=marker, label=name)
ax.axhline(20.0, color="grey", ls="--", lw=1)
ax.text(0.1, 21, "20% — inspect, do not discard", color="grey", fontsize=8)
ax.set_yscale("log")
ax.set_ylabel("amplitude CRLB [%]")
ax.set_title("Uncertainty follows the noise floor, not the concentration")
ax.legend()
plt.show()
```

(pyamares-drift)=

### Did it follow the drift?

The second check is one the CRLB cannot give you: whether the fit found the peaks where they
actually were. We know the answer, because we put the drift there.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(ds["voxel"], b0_drift, "k--", lw=1, label="simulated B0 drift")
ds["chem_shift"].sel(metabolite="PCr").plot(ax=ax, marker="o", ls="none", label="fitted PCr")
(ds["chem_shift"].sel(metabolite="ATP") + 7.5).plot(
    ax=ax, marker="s", ls="none", label="fitted ATP (+7.5 ppm)"
)
ax.set_ylabel("shift from nominal [ppm]")
ax.set_title("Both peaks were tracked across the drift")
ax.legend()
plt.show()
```

Each peak was found within a hundredth of a ppm of where it was put — including in the noisiest
voxels, where the _amplitude_ has already become uncertain. Position is a much better-conditioned
quantity than area, which is worth remembering when a fit "looks bad".

(pyamares-residuals)=

### Look at the spectrum

The last check is the one no summary statistic replaces. `data`, `fit` and `residuals` come back in
the same domain you passed in, so plotting them is the same code for the best and worst voxel:

```{code-cell} ipython3
ds.isel(voxel=0)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

for ax, v in zip(axes, [0, 7]):
    voxel = ds.isel(voxel=v)
    offset = -0.25 * float(np.abs(voxel["data"].xmr.to_spectrum().real).max())
    component_offset = -0.15 * float(np.abs(voxel["data"].xmr.to_spectrum().real).max())
    for var, color, label in (
        ("data", "black", "data"),
        ("fit", "tab:red", "fit"),
    ):
        voxel[var].xmr.to_spectrum().xmr.to_ppm().real.plot(
            ax=ax, color=color, alpha=0.5 if var == "data" else 1.0, label=label
        )
    (voxel["residuals"].xmr.to_spectrum().xmr.to_ppm().real + offset).plot(
        ax=ax, color="tab:green", lw=0.8, label="residual (offset)"
    )
    for metabolite in voxel.metabolite:
        (
            voxel["fit_components"].sel(metabolite=metabolite).xmr.to_spectrum().xmr.to_ppm().real
            + component_offset
        ).plot(
            ax=ax,
            label=f"{metabolite.values} (fit component, offset)",
            alpha=1,
            lw=2,
            linestyle=(0, (1, 2)),
        )

    crlb = float(voxel["crlb"].sel(metabolite="ATP", parameter="amplitude"))
    ax.set_title(f"voxel {v} — ATP CRLB {crlb:.1f}%")
    ax.set_xlim(4, -12)
    ax.set_ylabel("intensity [a.u.]")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

Both residuals are structureless — the model fits both voxels as well as the data allows. The
difference between them is not fit quality, it is how much data there was. That distinction is
exactly what a CRLB column tells you and a glance at a spectrum does not.

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: the quality story the three plots above tell.
# 1. Amplitudes recover the simulated concentration gradient, ATP included, and the
#    tolerance is looser where the noise is: this is a fit, not an interpolation.
_signal = [0, 1, 2, 4, 5, 6, 7]  # every voxel except the empty one
np.testing.assert_allclose(
    ds["amplitude"].sel(metabolite="PCr").isel(voxel=_signal).values,
    pcr_amplitude[_signal],
    rtol=0.1,
    err_msg="PCr amplitudes did not track the simulated gradient",
)
np.testing.assert_allclose(
    ds["amplitude"].sel(metabolite="ATP").isel(voxel=_signal).values,
    5.0,
    rtol=0.2,
    err_msg="ATP amplitude should be constant across the row",
)
# 2. Linewidths stayed inside the prior's window.
_lw = ds["linewidth"].isel(voxel=_signal).values
assert np.all((_lw > 5.0) & (_lw < 40.0)), "linewidths escaped the prior knowledge bounds"
# 3. The drift was tracked, both peaks, to well under the search window.
np.testing.assert_allclose(
    ds["chem_shift"].sel(metabolite="PCr").isel(voxel=_signal).values,
    b0_drift[_signal],
    atol=0.05,
    err_msg="fitted PCr shift did not follow the simulated drift",
)
np.testing.assert_allclose(
    ds["chem_shift"].sel(metabolite="ATP").isel(voxel=_signal).values + 7.5,
    b0_drift[_signal],
    atol=0.05,
    err_msg="fitted ATP shift did not follow the simulated drift",
)
# 4. Uncertainty rises monotonically with the noise floor, and crosses 20% at the end
#    -- the claim the table and the log-scale plot both make.
_crlb_atp = ds["crlb"].sel(metabolite="ATP", parameter="amplitude").isel(voxel=_signal).values
assert np.all(np.diff(_crlb_atp) > 0), "ATP CRLB should grow monotonically down the row"
assert _crlb_atp[0] < 1.0 and _crlb_atp[-1] > 20.0
# 5. `sd` is the same story in absolute units: it rises down the row too, and it is
#    small next to the amplitude it qualifies.
_amp_atp = ds["amplitude"].sel(metabolite="ATP").isel(voxel=_signal).values
_sd_atp = ds["sd"].sel(metabolite="ATP", parameter="amplitude").isel(voxel=_signal).values
assert np.all(np.diff(_sd_atp) > 0), "absolute sd should grow with the noise floor"
assert np.all(_sd_atp < _amp_atp), "sd should be a fraction of the amplitude it qualifies"
# 6. residuals are exactly data - fit.
xr.testing.assert_allclose(ds["residuals"], ds["data"] - ds["fit"])
```

(pyamares-deuterium)=

## 5. A harder spectrum: deuterium with a baseline

The ³¹P row was chosen to be tractable: two well-separated peaks, a flat baseline, and no phase to
speak of. Deuterium metabolic imaging offers none of the three. ²H resonates at 45.7 MHz at 7 T, so
water, the glucose that labels it and the Glx pool downstream are packed into roughly two and a
half ppm; the lines are broad; a macromolecular component lifts the whole spectrum off zero; and
both the receiver phase and the B0 offset drift from voxel to voxel.

That last pair is what earns this section its own dataset. A _single_ peak wandering is what
[§4](#pyamares-drift) already covered. Here the entire spectrum shifts and twists as one — and the
prior knowledge has to say so, or the fit quietly gives up on the smallest peak.

(pyamares-deuterium-grid)=

### Four voxels that move together

Two numbers per voxel produce that motion, and `simulate_fid` takes both directly. A scalar
`phases` is applied to every peak, which is precisely what a global receiver phase is; adding one
offset to every entry of `chemical_shifts` is precisely a B0 shift.

The baseline is the third new ingredient, and it is not a separate concept: it is a peak like any
other, just a very broad one. A 200 Hz line at 45.7 MHz spans more than four ppm, which is wider
than the metabolic spectrum sitting on top of it — so it reads as a rolling background rather than
as a peak.

```{code-cell} ipython3
mhz_2h = 45.7  # 2H at 7 T
carrier = 4.8  # transmit on the water line

# name: (amplitude, chemical shift [ppm], linewidth [Hz])
dmi_truth = {
    "Water": (100.0, 4.80, 14.0),
    "Glc": (22.0, 3.80, 16.0),
    "Glx": (10.0, 2.35, 18.0),
    "Baseline": (400.0, 3.50, 200.0),  # broad enough to read as a background
}
dmi_names = list(dmi_truth)
dmi_amps = np.array([dmi_truth[n][0] for n in dmi_names])
dmi_ppms = np.array([dmi_truth[n][1] for n in dmi_names])
dmi_lws = np.array([dmi_truth[n][2] for n in dmi_names])

n_dmi = 4
dmi_b0 = np.linspace(-0.12, 0.18, n_dmi)  # ppm — the whole spectrum, not one peak
dmi_phase = np.linspace(-35.0, 50.0, n_dmi)  # degrees — likewise
dmi_snr = np.linspace(180.0, 60.0, n_dmi)

dmi = xr.concat(
    [
        simulate_fid(
            amplitudes=dmi_amps,
            chemical_shifts=dmi_ppms + dmi_b0[i],  # every peak moves by the same amount
            reference_frequency=mhz_2h,
            carrier_ppm=carrier,
            spectral_width=2000.0,
            n_points=512,
            dampings=np.pi * dmi_lws,  # damping = pi x linewidth
            phases=np.deg2rad(dmi_phase[i]),  # one phase, applied to every peak
            target_snr=float(dmi_snr[i]),
            seed=100 + i,
        )
        for i in range(n_dmi)
    ],
    dim="voxel",
).assign_coords(voxel=np.arange(n_dmi))
dmi.attrs = {"reference_frequency": mhz_2h, "carrier_ppm": carrier}
```

`carrier_ppm` is doing real work here for the first time on this page. The ³¹P grid transmitted at
0 ppm, so its peak positions and its prior knowledge were the same numbers. Deuterium is normally
transmitted on the water line, and stating that in the attrs is what lets you keep writing
literature ppm values — 4.8 for water, 3.8 for glucose — everywhere else.

```{code-cell} ipython3
dmi_spec = dmi.xmr.to_spectrum().xmr.to_ppm()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
dmi_spec.real.plot.line(x="chemical_shift", hue="voxel", ax=axes[0])
axes[0].set_title("Real part — every voxel is phased differently")
np.abs(dmi_spec).plot.line(x="chemical_shift", hue="voxel", ax=axes[1], add_legend=False)
axes[1].set_title("Magnitude — phase removed, the B0 shift is what is left")
for ax in axes:
    ax.set_xlim(7.5, 0.5)
    ax.set_ylabel("intensity [a.u.]")
plt.tight_layout()
plt.show()
```

Neither panel has a flat stretch to call a baseline: the broad component carries the whole spectrum
up to a third of water's height, and glucose and Glx sit on its flank rather than on zero.

(pyamares-deuterium-tie)=

### One phase for the whole spectrum

The prior knowledge follows the same shape as before, with two differences that the previous
section did not need. Linewidths get explicit bounds, and the baseline is listed as a peak.

Bounding the linewidths is not decoration. Left open, the narrow peaks and the broad one describe
overlapping things — a 14 Hz water line and a 200 Hz background can trade area back and forth
without changing the sum much — and the optimizer will happily widen water until it _is_ the
baseline. Saying "water is between 5 and 40 Hz, the background is between 120 and 400" is the
statement that keeps them distinguishable.

```{code-cell} ipython3
dmi_pk = {
    "Water": {
        "amplitude": 100.0,
        "chem_shift": 4.8,
        "linewidth": 14.0,
        "chem_shift_bounds": (4.5, 5.1),
        "linewidth_bounds": (5.0, 40.0),
    },
    "Glc": {
        "amplitude": 20.0,
        "chem_shift": 3.8,
        "linewidth": 16.0,
        "chem_shift_bounds": (3.5, 4.1),
        "linewidth_bounds": (5.0, 40.0),
    },
    "Glx": {
        "amplitude": 10.0,
        "chem_shift": 2.35,
        "linewidth": 18.0,
        "chem_shift_bounds": (2.05, 2.65),
        "linewidth_bounds": (5.0, 45.0),
    },
    "Baseline": {
        "amplitude": 400.0,
        "chem_shift": 3.5,
        "linewidth": 200.0,
        "chem_shift_bounds": (3.0, 4.0),
        "linewidth_bounds": (120.0, 400.0),
    },
}

ds_free = dmi.xmr.fit_amares(dmi_pk)
```

That fit runs, reports `fit_status` 0 everywhere, and is wrong about Glx. Nothing in the spec ties
the four phases to one another, so AMARES gives each peak its own — four free parameters where the
physics has one. Water, glucose and the background are strong enough to find the right value
anyway. Glx is a tenth of water's amplitude, sitting on the background's flank, and with a free
phase it has a second, wrong minimum available: rotate its phase far enough and its amplitude can
collapse toward zero with barely a penalty in the residual.

[`build_prior_knowledge`](#xmris.fitting.prior_knowledge.build_prior_knowledge) has
`tie_phase_to` for exactly this. It nominates one peak as the anchor and writes every other peak's
phase as an lmfit expression referencing it, so the four phases become one fitted number.

```{code-cell} ipython3
import io

from xmris import build_prior_knowledge

# `tie_phase_to` is a builder argument, so go through the CSV rather than the dict:
# build the table, read it back as a DataFrame, and hand that to `fit_amares`.
dmi_pk_tied = pd.read_csv(
    io.StringIO(build_prior_knowledge(dmi_pk, tie_phase_to="Water")), index_col=0
)
dmi_pk_tied.loc[["phase"]]
```

The anchor keeps a number; everyone else carries the anchor's name. `fit_amares` accepts that
`DataFrame` wherever it accepts the dict:

```{code-cell} ipython3
ds_dmi = dmi.xmr.fit_amares(dmi_pk_tied)

pd.DataFrame(
    {
        "true": [dmi_truth[n][0] for n in dmi_names],
        "free phase": ds_free["amplitude"].mean("voxel").sel(metabolite=dmi_names).values,
        "tied phase": ds_dmi["amplitude"].mean("voxel").sel(metabolite=dmi_names).values,
        "worst CRLB% free": ds_free["crlb"]
        .sel(parameter="amplitude")
        .max("voxel")
        .sel(metabolite=dmi_names)
        .values,
        "worst CRLB% tied": ds_dmi["crlb"]
        .sel(parameter="amplitude")
        .max("voxel")
        .sel(metabolite=dmi_names)
        .values,
    },
    index=dmi_names,
).round(1)
```

Three parameters removed, and every peak got better. Note which column would have caught this
without knowing the truth: the free-phase CRLB on Glx is the one number in that table that is
obviously not a measurement.

```{code-cell} ipython3
print("global phase, per voxel :", ds_dmi["phase"].sel(metabolite="Water").values.round(1))
print("            simulated   :", dmi_phase.round(1))
print(
    "B0 offset, per voxel    :",
    (ds_dmi["chem_shift"].sel(metabolite="Water").values - 4.8).round(3),
)
print("            simulated   :", dmi_b0.round(3))
```

Both nuisance parameters come back per voxel to within a tenth of a degree and a thousandth of a
ppm — and because the phase is tied, reading it off `Water` reads it off the whole spectrum.

(pyamares-deuterium-components)=

### What the components are for

With four overlapping contributions and no flat baseline, "does the fit look right?" is no longer a
question the summed model can answer on its own — a good total can hide a background that has
eaten a metabolite. `fit_components` is what makes the decomposition inspectable.

```{code-cell} ipython3
dmi_colors = {
    "Water": "tab:blue",
    "Glc": "tab:orange",
    "Glx": "tab:green",
    "Baseline": "tab:purple",
}

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, v in zip(axes.ravel(), ds_dmi.voxel.values):
    vox = ds_dmi.sel(voxel=v)
    scale = float(np.abs(vox["data"].xmr.to_spectrum().real).max())

    vox["data"].xmr.to_spectrum().xmr.to_ppm().real.plot(
        ax=ax, color="black", alpha=0.4, lw=2.0, label="data"
    )
    vox["fit"].xmr.to_spectrum().xmr.to_ppm().real.plot(ax=ax, color="tab:red", lw=1.2, label="fit")
    for m in vox.metabolite.values:
        vox["fit_components"].sel(metabolite=m).xmr.to_spectrum().xmr.to_ppm().real.plot(
            ax=ax, color=dmi_colors[str(m)], lw=1.0, ls=(0, (4, 1.5)), label=str(m)
        )
    (vox["residuals"].xmr.to_spectrum().xmr.to_ppm().real - 0.25 * scale).plot(
        ax=ax, color="tab:green", lw=0.7, label="residual (offset)"
    )

    ax.set_title(
        f"voxel {int(v)} — phase {float(vox['phase'].sel(metabolite='Water')):+.0f}°, "
        f"B0 {float(vox['chem_shift'].sel(metabolite='Water')) - 4.8:+.2f} ppm"
    )
    ax.set_xlim(7.5, 0.5)
    ax.set_ylabel("intensity [a.u.]")
axes[0, 0].legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()
```

Every component carries the voxel's phase, because the phase was fitted rather than corrected
away — which is why water's dashed line goes dispersive in voxel 0 and voxel 3 exactly as the data
does. The background is a single broad component that stays under the metabolites instead of
absorbing them, and the residual is flat in all four voxels.

Having the background as a named component also means removing it is selection, not arithmetic:

```{code-cell} ipython3
# `.sum()` drops attrs, and `to_ppm` needs the calibration back to find its axis.
metabolites_only = (
    ds_dmi["fit_components"]
    .drop_sel(metabolite="Baseline")
    .sum("metabolite")
    .assign_attrs(ds_dmi["fit_components"].attrs)
)

fig, ax = plt.subplots(figsize=(7, 3.5))
ds_dmi["data"].isel(voxel=1).xmr.to_spectrum().xmr.to_ppm().real.plot(
    ax=ax, color="black", alpha=0.4, lw=2.0, label="data"
)
metabolites_only.isel(voxel=1).xmr.to_spectrum().xmr.to_ppm().real.plot(
    ax=ax, color="tab:red", lw=1.2, label="fit, baseline component removed"
)
ax.set_title("voxel 1 — the same fit, with the background component dropped")
ax.set_xlim(7.5, 0.5)
ax.set_ylabel("intensity [a.u.]")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: the deuterium fit recovered what was simulated.
# 1. Every peak, the broad background included, comes back at its simulated size.
for _n, (_amp, _ppm, _lw) in dmi_truth.items():
    np.testing.assert_allclose(
        ds_dmi["amplitude"].sel(metabolite=_n).values,
        _amp,
        rtol=0.25,
        err_msg=f"{_n} amplitude was not recovered",
    )
    np.testing.assert_allclose(
        ds_dmi["linewidth"].sel(metabolite=_n).values,
        _lw,
        rtol=0.25,
        err_msg=f"{_n} linewidth was not recovered -- did the peaks trade area with the baseline?",
    )
assert (ds_dmi["fit_status"].values == 0).all()

# 2. The two per-voxel nuisance parameters are recovered.
np.testing.assert_allclose(
    ds_dmi["phase"].sel(metabolite="Water").values,
    dmi_phase,
    atol=3.0,
    err_msg="the global receiver phase was not recovered",
)
np.testing.assert_allclose(
    ds_dmi["chem_shift"].sel(metabolite="Water").values - carrier,
    dmi_b0,
    atol=0.03,
    err_msg="the global B0 offset was not recovered",
)

# 3. The tie held: one phase per voxel, shared by all four peaks. Without this the
#    prose above ("reading it off Water reads it off the whole spectrum") is false.
_phases = ds_dmi["phase"].transpose("voxel", "metabolite").values
assert np.abs(_phases - _phases[:, :1]).max() < 1e-6, (
    f"the phases were not tied: spread {np.abs(_phases - _phases[:, :1]).max():.3f} deg"
)


# 4. The point of the section: tying beats four free phases on the weakest peak.
def _err(d):
    return float(np.abs(d["amplitude"].sel(metabolite="Glx").values - 10.0).sum())


assert _err(ds_dmi) < _err(ds_free), (
    f"tied phase ({_err(ds_dmi):.2f}) did not beat free phase ({_err(ds_free):.2f}) on Glx"
)

# 5. The components still decompose the fit exactly, four peaks and a carrier included.
xr.testing.assert_allclose(ds_dmi["fit_components"].sum("metabolite", skipna=False), ds_dmi["fit"])
# ...and the baseline-removed model of the cell above is exactly `fit` minus the
# background, so what that plot plots is a decomposition, not an approximation.
# (`sel` leaves a scalar `metabolite` coord behind; drop it before the comparison.)
xr.testing.assert_allclose(
    ds_dmi["fit_components"].drop_sel(metabolite="Baseline").sum("metabolite")
    + ds_dmi["fit_components"].sel(metabolite="Baseline").drop_vars("metabolite"),
    ds_dmi["fit"],
)
```

(pyamares-guarantees)=

## 6. What holds when the data is real

The data above was synthetic, and synthetic data is forgiving in one specific way: it arrives at
unit scale, in the domain the algorithm wants, with no missing voxels. Three guarantees cover what
happens when it does not.

**A non-answer is `NaN`, never `0`** — shown in [§3](#pyamares-status), with `fit_status` recording
which kind of absence it was.

**Safe at any signal scale.** A Bruker FID peaks around `1e7`, and pyAMARES derives its optimizer
tolerance from signal magnitude — at that scale the tolerance balloons and the solver stops before
it has moved, handing back your prior guess dressed as a result. `fit_amares` normalizes the array,
fits where the tolerance behaves, and rescales:

```{code-cell} ipython3
scanner_scale = (grid.isel(voxel=0) * 1e7).assign_attrs(grid.attrs)
ds_scaled = scanner_scale.xmr.fit_amares(pk)

print("unit scale :", ds["amplitude"].isel(voxel=0).values.round(4))
print("x 1e7      :", ds_scaled["amplitude"].values)
print("ratio      :", (ds_scaled["amplitude"] / ds["amplitude"].isel(voxel=0)).values)
```

**Either domain in, the same domain out.** AMARES fits the FID, but you rarely hold one — you hold
a phased spectrum in ppm. Hand that over and `fit_amares` round-trips it for you, returning
`data`, `fit`, `fit_components` and `residuals` in the representation you passed:

```{code-cell} ipython3
as_ppm = grid.isel(voxel=0).xmr.to_spectrum().xmr.to_ppm()
ds_ppm = as_ppm.xmr.fit_amares(pk)

print("fitted from a FID     :", ds["amplitude"].isel(voxel=0).values.round(4))
print("fitted from a spectrum:", ds_ppm["amplitude"].values.round(4))
print("signals came back on  :", ds_ppm["fit"].dims)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: the two guarantees demonstrated above.
# 1. Scale invariance: a 1e7 FID gives the same physics, in its own units.
np.testing.assert_allclose(
    ds_scaled["amplitude"].values / 1e7,
    ds["amplitude"].isel(voxel=0).values,
    rtol=1e-4,
    err_msg="a Bruker-scale fit did not reproduce the unit-scale answer",
)
assert ds_scaled.attrs["amares_amplitude_scale"] > 1e6
# 2. Domain preservation: identical parameters, spectral signals out.
np.testing.assert_allclose(
    ds_ppm["amplitude"].values,
    ds["amplitude"].isel(voxel=0).values,
    rtol=1e-3,
    err_msg="fitting a spectrum disagreed with fitting the same signal as a FID",
)
assert ds_ppm["fit"].dims == ("chemical_shift",)
assert ds_ppm["residuals"].dims == ("chemical_shift",)
# every signal variable rides along -- a FID left inside a ppm Dataset would break
# `ds_ppm["fit_components"].xmr.to_ppm()` with a missing-reference_frequency error.
assert ds_ppm["fit_components"].dims == ("metabolite", "chemical_shift")
# and the round trip is reversible from there
assert ds_ppm["fit"].xmr.to_hz().dims == ("frequency",)
```

:::{seealso}
The reasoning behind each guarantee — the optimizer trap that makes a Bruker-scale fit hand back
your prior guess unchanged, why a failed voxel is not a zero, and why fitting hand-rolls its own
domain round trip — is in the diary entry
[pyAMARES now behaves like the rest of the pipeline](#diary-amares-fitting).

For fitting a dynamic series rather than a spatial one,
[Visualizing Dynamic AMARES Fits](#dynamic-fits) turns the same `Dataset` into trajectory and
quality-control plots.
:::
