(diary-amares-fitting)=
# pyAMARES now behaves like the rest of the pipeline

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-07-25 · #105</span>

Quantifying a spectrum in xmris means `da.xmr.fit_amares(...)`: one accessor method that
wraps [pyAMARES](https://github.com/HawkMRS/pyAMARES), fits every voxel of an N-dimensional
array, and hands back a `Dataset` of amplitudes, chemical shifts and linewidths. Point that
method at a real Bruker FID, though, and something unnerving happens. Every status reads
*converged*, and the numbers are — exactly, to the digit — the prior knowledge you started
from. No exception, no warning, no failed status: just plausible numbers that happen to be
your guess.

That was the sharpest edge of a broader mismatch. pyAMARES is a stateful, file-driven
optimizer with its own scale assumptions; xmris is a pure `xarray in, xarray out` pipeline.
[PR #105](https://github.com/andrewendlinger/xmris/pull/105) makes the former behave like
the latter — without touching the AMARES mathematics.

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

| peak magnitude | derived `tol` | what the solver does |
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

(diary-amares-fitting-optimizer)=
## Two minima, and the default that picks one

A `NaN` tells you when a number is missing. It cannot tell you when a number is *wrong* —
and one was. Fit the same ³¹P signal twice, byte for byte identical, and γ-ATP comes back
either 5.00 or 6.12: one input, a 22 % spread, both runs reporting *converged*. Again the
difference between a publishable number and a wrong one, this time arriving with no
warning at all.

Everything above was an xmris-shaped fix, so the adapter is where we looked. It is not
there. Hashing every argument that reaches the optimizer across six calls — the FID, the
parameter values, bounds, `vary` flags, names, the time axis, `MHz`/`sw`/`deadtime` —
gives one distinct hash apiece, all six times. That clears the normalization factor, the
temporary prior-knowledge CSV and the verbosity context, and separate runs clear BLAS
threading, the global NumPy RNG, and a loose stopping tolerance (the shallow minimum
survives at `xtol = 1e-14`, so it is a real minimum, not an early exit). Handed those
bit-identical inputs and freshly copied parameters, raw `lmfit` still returns χ² = 0.004
or χ² = 3.71, unpredictably. The coin flip lives inside MINPACK — in
`scipy.optimize.leastsq` itself, which xmris cannot repair from the outside.

Which leaves no bug to fix and a default to choose. `"least_squares"` — SciPy's
trust-region solver, which handles the parameter bounds natively where Levenberg–Marquardt
reaches them through a transform — lands in the deep basin on every run measured:

| | `"leastsq"` (was) | `"least_squares"` (now) |
|---|---|---|
| 8 runs, ill-conditioned 2-peak signal | 3 distinct answers | 1 |
| γ-ATP amplitude (true: 5.0) | 5.00 **or** 6.12 | 5.00 |
| cost per fit (512 points) | ~28 ms | ~44 ms |
| 8-voxel grid, wall clock | 0.16 s | 0.16 s |

Both fitting tutorials had already pinned `method="least_squares"` at every call site, and
a default that every page in the docs overrides is the wrong default. So it moved, and the
pins came out with it — ten call sites across the docs and eleven more in the test suite,
which had been exercising a solver the library did not ship. Nothing rendered changed:
where LM converged at all it converged to the same place, agreeing on amplitudes, CRLBs and
SNR to every digit the pages print. The switch costs about half again per fit on a clean
512-point signal and nothing measurable across a grid, where process setup dominates — both
measured on synthetic data, and a scanner-scale multi-peak fit remains unmeasured.
`"leastsq"` is still one keyword away.

:::{warning}
Basin-stable is not bit-identical. `least_squares` still jitters in the last digits (χ²
agrees run to run to ~1e-12), so the guarantee is *the same minimum* — and
`TestFittingDomain::test_default_method_is_reproducible` pins exactly that, ten consecutive
default fits compared with `assert_allclose`, never `==`. Forced back to `"leastsq"` it
fails on its fourth run, which is the only reason to believe it when it passes.
:::

:::{dropdown} Why not keep LM and fit twice, keeping the better χ²?
It turns an unpredictable answer into a predictable one at double the cost — and only if
there are exactly two minima. Nothing promises that. On a real ³¹P spectrum with seven
peaks and heavier overlap there may be more, and "best of two" would then be quietly
choosing among several wrong ones while looking just as convergent. A solver that does not
flip basins is both cheaper and honest about what it guarantees.
:::

(diary-amares-fitting-workers)=
## The worker pool that made fitting slower

The optimizer default had a tell: every page in the docs overrode it. So did this one. `fit_amares`
shipped `num_workers=4`, and all ten fitting call sites across the tutorials and the test suite
passed `num_workers=1` instead. The second time that pattern shows up it stops being a coincidence
and starts being a measurement.

It is one — but not the single number it first looked like. On the 512-point two-peak signal these
pages fit, timed both cold (one fit in a fresh process) and warm (the fourth consecutive fit in the
same session):

| spectra | cold, serial | cold, `-1` | warm, serial | warm, `-1` |
|---|---|---|---|---|
| 2 | 0.79 s | 2.17 s | 0.06 s | 0.09 s |
| 8 | 0.94 s | 2.68 s | 0.20 s | 0.08 s |
| 32 | 1.64 s | 3.14 s | 0.81 s | 0.19 s |
| 64 | 3.21 s | 3.25 s | 2.45 s | 0.50 s |

Two break-evens, an order of magnitude apart. Cold, the pool pays ~1.5–2 s before it fits anything
— ten processes each re-importing NumPy, SciPy and pyAMARES, because macOS spawns rather than forks
— and does not draw level until about **64** spectra. Warm, loky keeps its executor alive between
calls, so that startup is paid once and the pool is ahead from about **8**.

Which of the two should a *default* serve? The cold one. A fitting script is a process that starts,
fits, and exits: it pays the startup and never amortizes it. And the person who *is* fitting
repeatedly in a long session — the one the warm column rewards — is precisely the person in a
position to type `num_workers=-1` once and mean it.

Neither number is hardcoded anywhere, because both move with the data: a real ³¹P fit, more peaks
over more points, runs 0.5–2 s per spectrum and drags both crossovers down to a handful of voxels.
That the threshold moves is the smaller half of the problem. The larger half is that `fit_amares`
cannot see the *machine*: not the container's CPU quota, not the SLURM allocation, not the outer
pool it may already be running inside. `os.cpu_count()` still reports the host's twenty cores from
inside a container limited to two. A default that starts processes is a decision made without the
information the decision needs — which is why SciPy ships `workers=1`, scikit-learn `n_jobs=None`,
and joblib `n_jobs=1`. The caller knows the machine; the library does not.

So fitting joins them. `num_workers` defaults to **1** and fits in-process, and the pool is one
keyword away in joblib's spelling:

```python
ds = grid.xmr.fit_amares(pk)                  # in-process — the default
ds = grid.xmr.fit_amares(pk, num_workers=-1)  # every core
ds = grid.xmr.fit_amares(pk, num_workers=-2)  # all but one
```

Those negative spellings already worked, by inheritance — `num_workers` has always been handed
straight to joblib's `n_jobs` — but nothing named or tested them, so they were a feature only a
reader of the source could find. Making them the official opt-in meant pinning them, and pinning
them turned up that the knob was never bounded by the work: asking for eight workers to fit two
spectra started eight processes, six of which would never receive a task. Dispatch now resolves the
count through `effective_n_jobs` *first* and then caps it at the spectra that actually have signal,
so `-1` on a two-voxel grid starts two workers rather than ten, and a request that resolves to a
single worker collapses into the in-process loop instead of paying a startup it cannot use. That
last case now also catches `-1` on a single-core host, which the earlier `n == 1` collapse did not.

One count is refused outright: `num_workers=0`, which joblib would otherwise reject from deep inside
dispatch with `"n_jobs == 0 in Parallel has no meaning"` — after the entire setup had been paid for,
and a plausible typo now that `1` is the default.

:::{warning}
`-1` means *every core this process can see* — which, inside a container, a SLURM job or an
enclosing pool, is the number the new default exists to avoid guessing. It is the right answer on
your own workstation and the wrong one on a shared node. On a real grid the win is large; ask for
it deliberately.
:::

The timings above are macOS `spawn` figures. Linux `fork` is cheaper today and Python 3.14 moves it
to `forkserver`, so the cold column should shrink there — by how much is unmeasured, which is the
one claim on this page that rests on a single platform.

:::{dropdown} Why not size the pool automatically?
The tempting version times the first fit and starts a pool only if the remaining ones would outrun
its startup — self-calibrating, and the first fit isn't wasted. But it calibrates the *work* when
the missing information is the *machine*, so it would still oversubscribe a two-CPU container. It
also makes which code path ran depend on how loaded the host was, and the branch is precisely what
`test_two_active_voxels_still_use_the_pool` and `test_pool_is_capped_at_active_spectra` exist to
pin. A knob the caller sets in one keystroke beats a heuristic nobody can predict or test.
:::

As with the optimizer, the pins came out with the default: nineteen `num_workers=1` arguments across
the architecture suite and six across the fitting notebooks, every one of them now redundant. Both
now exercise the shipped path rather than a configuration no user would have. Two pins stayed on
purpose — the parallel-equivalence tests need a pool to compare against, and
`TestFittingVerbosity`'s `warnings.catch_warnings` can only observe warnings raised in its own
process, so that one documents a real dependency rather than a habit.

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
- **"The pool is slower" turned out to be half a sentence.** The worker-pool decision was proposed
  on a single measured break-even — around fifty short spectra before a pool repays its startup —
  which made it look like plain arithmetic. Timing repeat fits in one process showed a second
  regime: loky keeps its executor alive between calls, so a warm session is ahead from about eight
  spectra while a cold one needs about sixty-four. The default did not change, but the reason it is
  right did. It rests on which regime a *default* should serve, and on the library's inability to
  see the machine — not on one number that happened to favour serial.
- **The explainer folded back in.** The plan put the reasoning above on a separate page,
  `docs/explanation/fitting.md` ("Fitting on Real Data"). It shipped and was read — and
  turned out to be a decision record wearing an explainer's clothes, duplicating this entry
  closely enough that an answer had to be assembled from two articles. It is now folded
  into this one. The lesson generalizes: *how it works* earns an explainer, *why it works
  that way* is a diary entry, and the split is worth making before the second page exists.
