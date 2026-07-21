(fitting)=
# Fitting on Real Data

:::{seealso}
New to AMARES fitting? The [pyAMARES tutorial](../notebooks/fitting/pyamares.md) runs
a fit end to end on synthetic data. This article is the *why* underneath it: what
changes when the data is real, and how `fit_amares` stays trustworthy when it does.
:::

Fitting inverts the forward model. `fit_amares` takes a signal and finds the set of
damped sinusoids — one per metabolite — whose sum best reproduces it, handing back
amplitudes, chemical shifts, linewidths and phases. AMARES does this in the *time*
domain, on the FID — but you needn't hand it one: like the rest of the pipeline,
fitting [meets your data in either domain](#fitting-domains), converting as needed and
returning your results in the representation you started from.

On the clean, unit-scale synthetic data of the tutorial, it just works. Point it at a
Bruker FID off a real scanner and something unnerving can happen instead: the fit
reports success, every status says *converged*, and the numbers it returns are —
exactly, to the digit — the prior knowledge you started from.

(fitting-scale-trap)=
## The problem: a fit that hands back your guess

An optimizer stops when its steps get small *relative* to the values it is moving.
pyAMARES sets that stopping tolerance from the signal's magnitude,

$$\text{tol} = \sqrt{\max|\text{fid}|}\times 10^{-6},$$

and passes it to SciPy's least-squares solver as `xtol`/`ftol` — which are *relative*
tolerances. The two don't compose. On a signal near unit scale the tolerance is a
sensible `~1e-6`. On a Bruker FID peaking around `1e7` it balloons:

| `max|fid|` | derived `tol` | what the solver does |
|---|---|---|
| `1e0` (synthetic) | `~1e-6` | iterates to a real minimum ✅ |
| `1e4` | `~1e-4` | stops early, near the prior |
| `1e7` (Bruker) | `~3e-3` | "converges" on step one ❌ |

At `~3e-3` the very first step already looks small enough to quit — before the solver
has moved off its starting point. It returns the initial guess and reports success.
There is no exception, no warning, no failed status: just plausible numbers that
happen to be your prior. On synthetic data the trap stays invisible because the scale
hides it.

(fitting-normalize)=
## The fix: normalize, fit, rescale

Scale is not physics. Multiply an FID by a constant and the *concentrations* it
encodes are unchanged — only the units move. So `fit_amares` takes scale out of the
solver's way: it divides the data by a single factor, fits in that normalized space
where the tolerance behaves, then multiplies the fitted amplitudes (and the
reconstructed fit) back by the same factor.

```python
# What goes wrong on raw Bruker-scale data:
#   fid (peaks ~1e7) → fit → tolerance ~3e-3 → "converged" on the prior   ❌

# What fit_amares does internally to prevent it:
scale = abs(fid).max()
ds = (fid / scale).xmr.fit_amares(pk)   # fit where the tolerance behaves
ds["amplitude"] *= scale                # ...then restore your input units   ✅
```

At the call site you just write `fid.xmr.fit_amares(pk)` — the round trip happens
inside, so the result is already in your units and safe at any scale.

```{important}
`fit_amares` is safe at **any** signal scale. Whether your FID peaks at `1` or at
`1e7`, the fit converges to the same physical answer — you never rescale your data by
hand to make fitting work.
```

One factor, applied to the *whole* array — never one per spectrum. A dynamic series
(hyperpolarized signal decaying across repetitions, say) carries its real information
in how the amplitude changes from one spectrum to the next. Normalizing each spectrum
to its own maximum would divide that away, flattening the very time course you are
trying to measure. A single global factor rescales everything together and leaves the
dynamics intact.

The factor isn't thrown away — `fit_amares` records it in `attrs` as
`amares_amplitude_scale`, so the normalization is auditable rather than hidden.

:::{dropdown} Why not pyAMARES's own `normalize_fid`?
pyAMARES has a `normalize_fid` switch, but it only touches the single template FID
handed to its initializer (`opts.fid = fid / np.max(fid)`) — not every spectrum in an
N-dimensional series, and not the reported amplitudes, which stay in normalized units.
`fit_amares` instead applies one magnitude factor across the whole array (so a dynamic
series keeps its relative scale) and multiplies the amplitudes back, so your results
land in input units with no per-call bookkeeping.
:::

(fitting-nan)=
## A failed fit is not a zero

Fit an N-dimensional dataset and some voxels will not fit — noise-only corners,
degenerate spectra, a solver that genuinely gives up. What belongs in their slot in
the output? Zero is the tempting default, and the wrong one: a real voxel can *also*
be legitimately near zero, because there is no metabolite there. Write failures as
zeros and the two become indistinguishable — a downstream mean or a concentration map
quietly folds the give-ups in as though they were measurements of nothing.

So `fit_amares` writes **`NaN`** for a fit that failed, and for any spectrum with no
signal to normalize (`max|fid| = 0`). `NaN` is not a value pretending to be data; it
is the honest absence of one. It shows up as a hole in a map, forces an explicit
`nanmean` instead of silently biasing an average, and makes "did this voxel fit?" as
simple as `isnan`. A genuine zero stays zero; a non-answer reads as one.

(fitting-domains)=
## Fit a FID or a spectrum

AMARES is a time-domain method — it models the FID directly. An earlier design took
that literally: `fit_amares` demanded a FID and left the `to_fid()` to you, on the
principle that the Fourier transform is part of the model and hiding it would hide the
model. Defensible, but it made fitting the *one* pipeline step that refused your data.
Everything else — `apodize_exp`, `autophase`, `baseline_als` — meets you in whichever
domain you already hold.

So fitting joins them. Hand `fit_amares` a spectrum and it round-trips through the FID
for you, fits, and returns `data`, `fit` and `residuals` **as spectra** — in the same
representation (Hz or ppm) you passed in. Hand it a FID and you get FIDs back. The
fitted parameters are identical either way.

```python
ds_fid  = fid.xmr.fit_amares(pk)                      # data/fit/residuals are FIDs
ds_spec = fid.xmr.to_spectrum().xmr.fit_amares(pk)    # ...come back as spectra

np.testing.assert_allclose(ds_fid.amplitude, ds_spec.amplitude, rtol=1e-3)  # same fit
```

This doesn't hide the model — it makes it *legible*. xmris's [domain contract](domains.md)
keeps the domain readable on every axis (the `repr` says `time` vs `frequency` vs
`chemical_shift`), and an inserted transform is bit-identical to the `to_fid()` you
would have typed. The model still lives in the time domain; you are simply no longer
the one shuttling data into it. Prefer to shuttle it yourself? `set_options(auto_convert=False)`
turns the convenience off and asks for an explicit `to_fid()`.

(fitting-adapter)=
## Why these fixes live in xmris, not pyAMARES

You may notice `pyproject.toml`'s `fitting` extra depends on `pyamares-xmris` rather
than `pyamares`, and wonder whether the robustness work belongs upstream instead. It
doesn't — and the reason is the same one that shapes the fix.

The scale trap is not a flaw in pyAMARES's mathematics; it is a property of the
*scale at which data is handed to it*. Normalize-and-rescale lives at that boundary —
the xmris adapter around pyAMARES — not in the AMARES algorithm. The `NaN` sentinel is
likewise about how xmris assembles results into an xarray Dataset. Both are
xmris-shaped, so both live in xmris, in code you can read.

That leaves `pyamares-xmris` with nothing *algorithmic* to carry. It is a faithful BSD
repackage of pyAMARES with **zero** kernel changes — its only difference is *packaging*.
`hlsvdpro` ships no wheel (or sdist) for Apple Silicon, so stock pyAMARES — which
requires it unconditionally — cannot install on an arm64 Mac at all; `pyamares-xmris`
adds an `hlsvdpro` platform marker that skips it there (pyAMARES falls back to a bundled
pure-Python HLSVD) and pins the `numpy<2` / `pandas<2.2` limits pyAMARES's own metadata
under-declares. The import name is unchanged (`import pyAMARES`).

So every *robustness* fix stays in the adapter, where it belongs, and the packaging fix
ships as its own small PyPI package (the upstream maintainer was unresponsive to the
one-line marker PR). That is what makes fitting an installable **optional extra**: a bare
`pip install xmris` never pulls pyAMARES — clean on every platform — and
`pip install "xmris[fitting]"` pulls `pyamares-xmris`, which installs cleanly on Apple
Silicon straight from PyPI.

:::{seealso}
The [pyAMARES tutorial](../notebooks/fitting/pyamares.md) demonstrates a full fit and
its Dataset output. The scale trap, the `NaN` sentinel, and the domain round trip are
pinned by the `TestFittingDomain` suite in `tests/test_core.py`.
:::
