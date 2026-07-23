(diary-amares-fitting)=
# pyAMARES now behaves like the rest of the pipeline

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-07-24</span>

Point `fit_amares` at a real Bruker FID and something unnerving happens: every status
reads *converged*, and the numbers it hands back are — exactly, to the digit — the prior
knowledge you started from. No exception, no warning. That was the sharpest edge of a
broader mismatch. pyAMARES is a stateful, file-driven optimizer with its own scale
assumptions; xmris is a pure `xarray in, xarray out` pipeline. This arc (workstreams A–G
on one branch, closing the issue tree #67/#69/#70/#80/#81/#82) makes the former behave
like the latter — without touching the AMARES mathematics.

:::{important}
Every robustness and ergonomics fix lives in the **xmris adapter** around pyAMARES, not
in the algorithm: a scale-normalizing wrapper, a `NaN` sentinel, a domain-preserving
round trip, and an in-memory prior-knowledge builder — so `fit_amares` is one more `.xmr`
method, and pyAMARES stays a faithful upstream we only repackage.
:::

(diary-amares-fitting-decisions)=
## The decisions that could have gone another way

The mechanics live in [Fitting on Real Data](../explanation/fitting.md); this is *why*
each went the way it did.

- **Scale.** pyAMARES derives its optimizer tolerance from `max|fid|`, so a Bruker-scale
  FID (~`1e7`) quits on step one. The fix normalizes by a *single global* factor, fits
  where the tolerance behaves, and rescales the amplitudes back — recorded as
  `amares_amplitude_scale` so the normalization is auditable, not hidden.
- **Failure.** A voxel that will not fit is written `NaN`, never `0`: a give-up must stay
  distinguishable from a genuine near-zero measurement, or a downstream `mean` folds it in.
- **Domain.** Fitting now meets your data in either representation and returns it in the
  one you passed, like every other step — reversing an earlier stance that `fit_amares`
  should demand a FID and keep the Fourier transform explicit.
- **Prior knowledge.** `build_prior_knowledge` takes a named-peak dict and emits
  pyAMARES's trap-prone positional CSV, refusing each footgun — blank phase bounds,
  digit-in-name multiplets, tie ordering — at the door.

```python
# One method: any scale, either domain, prior knowledge built in memory.
ds = spectrum.xmr.fit_amares(
    {"PCr": {"amplitude": 1, "chem_shift": 0.0, "linewidth": 15}},
)
ds["amplitude"]                        # quantified, per metabolite
ds["crlb"].sel(parameter="amplitude")  # per-parameter uncertainty
```

:::{dropdown} Why not the `@computes_in` decorator for the round trip?
The domain engine's decorators assume `DataArray` in, `DataArray` out. `fit_amares`
returns a `Dataset` whose parameter variables (`amplitude`, `crlb`, …) live on a
`metabolite` axis, not a spectral one — `@computes_in`'s restore leg would FFT them as if
they were signal. So fitting hand-rolls the round trip with the same converter helpers the
decorator uses, restoring only the signal variables. Commandment 6 records this as the one
domain-preserving function that carries no decorator.
:::

:::{dropdown} Why not pyAMARES's own `normalize_fid`?
`normalize_fid` scales only the single template FID handed to the initializer — not every
spectrum in an N-dimensional series — and leaves the reported amplitudes in normalized
units. It neither fixes the trap across a dataset nor returns input units, so the adapter
applies one magnitude factor across the whole array and rescales the amplitudes back.
:::

(diary-amares-fitting-vocab)=
## New vocabulary

Fitting output needed terms the pipeline did not have: a `metabolite` dimension indexing
the quantified peaks, and a `parameter` dimension indexing the fitted parameters that the
uncertainty variables span. The uncertainties took **Shape B** — the values (`amplitude`,
`chem_shift`, …) stay named data variables, and only `crlb` and the new `sd` carry the
`parameter` axis. So the common case (`ds["amplitude"]`) stays a plain named array, and
only an uncertainty comparison pays for the extra dimension.

(diary-amares-fitting-changed)=
## What changed from the plan

- **The fork exit reversed.** The plan assumed adopting *official* pyAMARES from PyPI.
  But official pyAMARES hard-requires `hlsvdpro`, which ships no Apple-Silicon wheel — so
  it cannot install on an arm64 Mac at all. The fix instead ships as `pyamares-xmris`, a
  zero-algorithm BSD repackage that adds the `hlsvdpro` platform marker and the
  `numpy<2`/`pandas<2.2` pins, pulled in only by the optional `[fitting]` extra.
- **Fitting's domain stance reversed.** The pre-arc position, recorded in
  [The Two Domains](../explanation/domains.md), was that fitting should demand a FID and
  leave the transform explicit. Making it domain-preserving like the rest of the pipeline
  overturned that, and `domains.md` and Commandment 6 were rewritten to match.
