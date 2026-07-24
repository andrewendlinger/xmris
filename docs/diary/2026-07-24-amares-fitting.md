(diary-amares-fitting)=
# pyAMARES now behaves like the rest of the pipeline

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-07-24</span>

Point `fit_amares` at a real Bruker FID and something unnerving happens: every status
reads *converged*, and the numbers it hands back are — exactly, to the digit — the prior
knowledge you started from. No exception, no warning, no failed status: just plausible
numbers that happen to be your guess. That was the sharpest edge of a broader mismatch.
pyAMARES is a stateful, file-driven optimizer with its own scale assumptions; xmris is a
pure `xarray in, xarray out` pipeline. This arc (workstreams A–G on one branch, closing
the issue tree #67/#69/#70/#80/#81/#82) makes the former behave like the latter — without
touching the AMARES mathematics.

:::{important}
Every robustness and ergonomics fix lives in the **xmris adapter** around pyAMARES, not
in the algorithm: a scale-normalizing wrapper, a `NaN` sentinel with a `fit_status` label,
a domain-preserving round trip, and an in-memory prior-knowledge builder — so `fit_amares`
is one more `.xmr` method, and pyAMARES stays a faithful upstream we only repackage.
:::

(diary-amares-fitting-scale)=
## The scale trap, and why the fix is a wrapper

An optimizer stops when its steps get small *relative* to the values it is moving.
pyAMARES sets that stopping tolerance from the signal's magnitude,

$$\text{tol} = \sqrt{\max|\text{fid}|}\times 10^{-6},$$

and passes it to SciPy's least-squares solver as `xtol`/`ftol` — which are themselves
*relative* tolerances. The two don't compose. On a signal near unit scale the tolerance is
a sensible `~1e-6`; on a Bruker FID peaking around `1e7` it balloons:

| `max|fid|` | derived `tol` | what the solver does |
|---|---|---|
| `1e0` (synthetic) | `~1e-6` | iterates to a real minimum ✅ |
| `1e4` | `~1e-4` | stops early, near the prior |
| `1e7` (Bruker) | `~3e-3` | "converges" on step one ❌ |

At `~3e-3` the very first step already looks small enough to quit, before the solver has
moved off its starting point. On synthetic data the trap stays invisible, because the
scale hides it — which is why it survived every tutorial we had.

Scale, though, is not physics. Multiply an FID by a constant and the *concentrations* it
encodes are unchanged; only the units move. So `fit_amares` takes scale out of the
solver's way: divide by a single factor, fit in that normalized space where the tolerance
behaves, then multiply the fitted amplitudes (and the reconstructed fit) back by the same
factor. At the call site you still just write `fid.xmr.fit_amares(pk)`.

That factor is deliberately **one number for the whole array**, never one per spectrum. A
dynamic series — hyperpolarized signal decaying across repetitions, say — carries its real
information in how the amplitude changes from one spectrum to the next. Normalizing each
spectrum to its own maximum would divide that away, flattening the very time course you
are trying to measure. A single global factor rescales everything together and leaves the
dynamics intact. It is recorded in `attrs` as `amares_amplitude_scale`, so the
normalization is auditable rather than hidden.

:::{dropdown} Why not pyAMARES's own `normalize_fid`?
pyAMARES has a `normalize_fid` switch, but it only touches the single template FID handed
to its initializer (`opts.fid = fid / np.max(fid)`) — not every spectrum in an
N-dimensional series, and not the reported amplitudes, which stay in normalized units. It
neither defeats the trap across a dataset nor returns input units, so the adapter applies
one magnitude factor across the whole array and rescales the amplitudes back.
:::

(diary-amares-fitting-absence)=
## A failed fit is not a zero

Fit an N-dimensional dataset and some voxels will not fit — noise-only corners, degenerate
spectra, a solver that genuinely gives up. What belongs in their slot in the output? Zero
is the tempting default, and the wrong one: a real voxel can *also* be legitimately near
zero, because there is no metabolite there. Write failures as zeros and the two become
indistinguishable — a downstream mean or a concentration map quietly folds the give-ups in
as though they were measurements of nothing.

So `fit_amares` writes **`NaN`**: not a value pretending to be data, but the honest absence
of one, forcing an explicit `nanmean` where a zero would have passed silently.

That leaves a second question the `NaN` cannot answer: *why* is the value absent — was
there no signal to fit, or did a real spectrum defeat the solver? Both read alike, yet one
is an empty background voxel you expected and the other is a fit you should look at. So
beside the science variables `fit_amares` returns a per-spectrum **`fit_status`** flag —
`0` fitted, `1` no_signal, `2` failed — carrying CF-style `flag_values`/`flag_meanings`.
It beat a boolean (there are three states, not two) and beat overloading the float arrays
with a second sentinel (the outcome is categorical, not numeric). Because it rides beside
the science variables rather than inside them, consumers that select by name
(`plot_qc_grid`, `plot_trajectory`) were untouched.

(diary-amares-fitting-domain)=
## Fit a FID or a spectrum

AMARES is a time-domain method — it models the FID directly. An earlier design took that
literally: `fit_amares` demanded a FID and left the `to_fid()` to you, on the principle
that the Fourier transform is part of the model and hiding it would hide the model.
Defensible, but it made fitting the *one* pipeline step that refused your data. Everything
else — `apodize_exp`, `autophase`, `baseline_als` — meets you in whichever domain you
already hold.

So fitting joined them. Hand `fit_amares` a spectrum and it round-trips through the FID,
fits, and returns `data`, `fit` and `residuals` **as spectra**, in the same representation
(Hz or ppm) you passed in. Hand it a FID and you get FIDs back. The fitted parameters are
identical either way.

```python
ds_fid  = fid.xmr.fit_amares(pk)                      # data/fit/residuals are FIDs
ds_spec = fid.xmr.to_spectrum().xmr.fit_amares(pk)    # ...come back as spectra
```

This doesn't hide the model — it makes it *legible*. The [domain
contract](../explanation/domains.md) keeps the domain readable on every axis (the `repr`
says `time` vs `frequency` vs `chemical_shift`), and an inserted transform is bit-identical
to the `to_fid()` you would have typed. The model still lives in the time domain; you are
simply no longer the one shuttling data into it. Prefer to shuttle it yourself?
`set_options(auto_convert=False)` turns the convenience off and asks for an explicit
`to_fid()`.

:::{dropdown} Why not the `@computes_in` decorator for the round trip?
The domain engine's decorators assume `DataArray` in, `DataArray` out. `fit_amares`
returns a `Dataset` whose parameter variables (`amplitude`, `crlb`, …) live on a
`metabolite` axis, not a spectral one — `@computes_in`'s restore leg would FFT them as if
they were signal. So fitting hand-rolls the round trip with the same converter helpers the
decorator uses, restoring only the signal variables. Commandment 6 records this as the one
domain-preserving function that carries no decorator.
:::

(diary-amares-fitting-vocab)=
## The vocabulary fitting needed

Fitting output needed terms the pipeline did not have: a `metabolite` dimension indexing
the quantified peaks, and a `parameter` dimension indexing the fitted parameters that the
uncertainty variables span. The uncertainties took **Shape B** — the values (`amplitude`,
`chem_shift`, …) stay named data variables, and only `crlb` and the new `sd` carry the
`parameter` axis. So the common case (`ds["amplitude"]`) stays a plain named array, and
only an uncertainty comparison pays for the extra dimension.

(diary-amares-fitting-prior)=
## Prior knowledge, built in memory

pyAMARES reads prior knowledge from a positional CSV whose row order and bound syntax are
easy to get subtly — and silently — wrong. `build_prior_knowledge` takes a named-peak dict
instead and emits that file, refusing each footgun at the door rather than writing
something that fits to garbage: a blank phase bound (`-inf` to pyAMARES, which `NaN`s the
fit), a trailing digit in a peak name (folded into a J-coupling multiplet), a tie anchor
that is not one of the peaks. The dict goes straight to `fit_amares` — no file need touch
disk.

```python
# One method: any scale, either domain, prior knowledge built in memory.
ds = spectrum.xmr.fit_amares(
    {"PCr": {"amplitude": 1, "chem_shift": 0.0, "linewidth": 15}},
)
ds["amplitude"]                        # quantified, per metabolite
ds["crlb"].sel(parameter="amplitude")  # per-parameter uncertainty
```

(diary-amares-fitting-adapter)=
## Why these fixes live in xmris, not pyAMARES

You may notice `pyproject.toml`'s `fitting` extra depends on `pyamares-xmris` rather than
`pyamares`, and wonder whether the robustness work belongs upstream instead. It doesn't —
and the reason is the same one that shapes the fix. The scale trap is not a flaw in
pyAMARES's mathematics; it is a property of the *scale at which data is handed to it*.
Normalize-and-rescale lives at that boundary — the xmris adapter — not in the AMARES
algorithm. The `NaN` sentinel is likewise about how xmris assembles results into an xarray
Dataset. Both are xmris-shaped, so both live in xmris, in code you can read.

That leaves `pyamares-xmris` with nothing *algorithmic* to carry: it is a faithful BSD
repackage with **zero** kernel changes, differing only in packaging. `hlsvdpro` ships no
wheel or sdist for Apple Silicon, so stock pyAMARES — which requires it unconditionally —
cannot install on an arm64 Mac at all; the repackage adds a platform marker that skips it
there (pyAMARES falls back to a bundled pure-Python HLSVD) and pins the `numpy<2` /
`pandas<2.2` limits pyAMARES's own metadata under-declares. The import name is unchanged.
It ships as its own small PyPI package because the upstream maintainer was unresponsive to
the one-line marker PR — and that is what makes fitting an installable **optional extra**,
where a bare `pip install xmris` never pulls pyAMARES at all.

:::{seealso}
The [pyAMARES tutorial](../notebooks/fitting/pyamares.md) runs a fit end to end and shows
the Dataset it returns. The guarantees above are pinned by `TestFittingDomain` in
`tests/test_core.py` and, end to end through the public pipeline, by
`docs/notebooks/fitting/testonly_amares_robustness.md`.
:::

(diary-amares-fitting-changed)=
## What changed from the plan

- **The fork exit reversed.** The plan assumed the arc would end by adopting *official*
  pyAMARES from PyPI, the fork finally unnecessary. It can't: official pyAMARES does not
  install on an arm64 Mac at all, so the repackage described above is a permanent
  dependency, not a waypoint.
- **Fitting's domain stance reversed.** The pre-arc position was the one recorded in
  [The Two Domains](../explanation/domains.md) — fitting demands a FID, the transform stays
  explicit. Overturning it meant rewriting that page and Commandment 6 as part of this arc,
  not just changing `fit_amares`.
- **The explainer folded back in.** The plan put the reasoning above on a separate page,
  `docs/explanation/fitting.md` ("Fitting on Real Data"). It shipped and was read — and
  turned out to be a decision record wearing an explainer's clothes, duplicating this entry
  closely enough that an answer had to be assembled from two articles. It is now folded
  into this one. The lesson generalizes: *how it works* earns an explainer, *why it works
  that way* is a diary entry, and the split is worth making before the second page exists.
