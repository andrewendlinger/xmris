# PR #105 follow-ups — findings from the fitting-docs rewrite

Written 2026-07-25 on branch `amares`, after splitting `docs/notebooks/fitting/pyamares.md` into a
quick start plus a deep dive. Everything below is **library-side** and was deliberately left out of
that docs pass. Items 1–3 are real defects with repros; item 4 is accumulated debt.

Repro snippets assume `uv run python -c ...` from the repo root with the `fitting` extra installed.
Never run fitting under the local coverage tracer (`--no-cov`); CI is the coverage authority.

---

## 1. `method="leastsq"` fits are not reproducible — HIGH

The default optimizer returns **one of two different minima, unpredictably**, for byte-identical
input. On the 31P test signal below the two answers differ by 22% in ATP amplitude — the difference
between a publishable number and a wrong one.

### Repro

```python
import numpy as np
import xmris
from xmris.fitting.simulation import simulate_fid

fid = simulate_fid(
    amplitudes=[10.0, 5.0], chemical_shifts=[-0.25, -7.75], reference_frequency=120.6,
    spectral_width=8000.0, n_points=512, dampings=[np.pi * 15.0, np.pi * 20.0],
    target_snr=250.0, seed=0,
).assign_attrs(reference_frequency=120.6, carrier_ppm=0.0)

pk = {
    "PCr": {"amplitude": 10.0, "chem_shift": 0.0, "linewidth": 15.0,
            "chem_shift_bounds": (-0.6, 0.6)},
    "ATP": {"amplitude": 5.0, "chem_shift": -7.5, "linewidth": 20.0,
            "chem_shift_bounds": (-8.2, -6.8)},
}

for i in range(5):                       # default method="leastsq"
    ds = fid.xmr.fit_amares(pk, num_workers=1)
    print(i, ds["amplitude"].values.round(4))
```

Observed (two consecutive processes, same machine, same input):

```
run 1: [10.1198 6.1211] [10.0044 5.0007] [10.1198 6.1211] [10.1197 6.1331] [10.1198 6.1211]
run 2: [10.1198 6.1211] [10.0044 5.0007] [10.1198 6.1211] [10.0044 5.0007] [10.0044 5.0007]
```

`[10.0044, 5.0007]` is the true answer (simulated amplitudes 10 and 5). `[10.1198, 6.1211]` is a
shallower second minimum. Note also the jitter *within* a minimum (`6.1211` vs `6.1331`).

### What was ruled out

- **Not** input variation — the FID is seeded and identical (`np.allclose` across calls).
- **Not** `initialParams` mutation. `lmfit` does not modify the shared `Parameters` object between
  fits (`amares.py:597` passes `shared_obj.initialParams` to every spectrum; a before/after diff
  over all parameter values is empty after four fits).
- **Not** `initialize_FID` — calling it three times on identical inputs yields
  bit-identical `initialParams`.
- **Not** the parallel path — this reproduces at `num_workers=1`.
- Reusing **one** `shared_obj` across repeated direct `_fit_dataset_safe` calls *is* stable (4/4
  land on the same minimum). So the variation appears to enter with something `fit_amares` does
  per call around the fit, not with the optimizer in isolation.

### Where to look next

Bisect what `fit_amares` adds around `_fit_dataset_safe` between calls: the global normalization
factor (`amares.py:541-549`), the per-call temp prior-knowledge CSV (`_resolve_pk_file`), and the
`_set_verbosity` / `_muted_warnings` context (`amares.py:54-95`). Then check whether lmfit's
`leastsq` (MINPACK) path picks up any process- or thread-level state.

### Workaround already in place

`method="least_squares"` (SciPy trust-region) is stable: 6/6 identical on the single spectrum
above, and 5/5 **bit-identical** on the 8-voxel grid in `docs/notebooks/fitting/pyamares.md`. Both
new fitting pages pin it, and the deep dive says why in prose. That prose should be revisited when
this is fixed.

### Acceptance

Ten consecutive `fit_amares` calls on the repro above, with the default `method`, return identical
amplitudes — ideally pinned by a test in `TestFittingDomain`.

---

## 2. Empty voxels are dispatched to the worker pool — ~~MEDIUM~~ **DONE 2026-07-25**

Fixed: `fit_amares` now masks the dispatch (`active_idx = np.flatnonzero(spectrum_max != 0)`) and
scatters results back by absolute index, so an all-zero spectrum never reaches the optimizer in
either branch. `fit_status` still reads `1` for it and the values stay `NaN` — the output is
byte-identical, only the wasted work and the stderr leak are gone. Pinned by
`TestFittingDomain::{test_empty_voxel_not_dispatched, test_parallel_empty_voxel_keeps_order,
test_empty_voxel_leaks_no_stderr}`.

Two things the fix turned up that the notes below got wrong or missed:

- The stderr leak was **not** parallel-only. Measured on this branch pre-fix, the same 8-voxel grid
  leaked 4 lines at `num_workers=1` *and* at `num_workers=4`; post-fix it is 0 in both. The table
  below is left as originally recorded.
- `TestFittingVerbosity::test_fit_silent_when_not_verbose` built its warning trigger from an
  exactly-zero voxel, which is precisely what is no longer dispatched — the fix would have left it
  passing vacuously. It now uses a `fid * 1e-30` voxel under `least_squares` (which trips the muted
  scipy `xtol`/`ftol` UserWarning) and asserts **both** directions, so the trigger going dead is
  itself a failure. `_muted_warnings`' "(an exactly-zero spectrum)" note was corrected to match.

<details>
<summary>Original report</summary>

`fit_amares` computes `spectrum_max` up front (`amares.py:549`) but only consults it when
*unpacking* results (`amares.py:650`, `if spectrum_max[i] == 0: status[i] = 1; continue`). Every
spectrum — including all-zero ones — is fitted first, in both the serial branch
(`amares.py:592-605`) and the parallel one (`amares.py:606-615`).

Two consequences:

1. **Wasted work** proportional to the number of empty voxels. On a real MRSI grid the background
   is often most of the array.
2. **Warning noise the user sees.** Fitting an all-zero spectrum gives a degenerate covariance, and
   with `num_workers > 1` lmfit's warning escapes the worker onto stderr — including absolute
   `.venv` paths, which then render inside notebook output on the docs site:

   ```
   .../lmfit/minimizer.py:819: RuntimeWarning: invalid value encountered in sqrt
     par.stderr = float(np.sqrt(self.result.covar[ivar, ivar]))
   ```

### Repro

Build any grid containing `xr.zeros_like(fid)` and fit it four ways:

| `num_workers` | empty voxel | stderr lines |
|---|---|---|
| 1 | yes | 0 |
| 1 | no | 0 |
| 4 | yes | **4** |
| 4 | no | 0 |

(8 voxels, 512 points, `method="least_squares"`; run each in a subprocess and count `p.stderr`.)

### Fix sketch

Skip dispatch for `spectrum_max[i] == 0` — substitute `None` into `result_list` at that index in
both branches, so the existing status logic at `amares.py:648-655` is untouched. Cheap, removes the
wasted fits *and* the stderr leak.

### Acceptance

A grid with an empty voxel fits with zero stderr output at `num_workers=4`, `fit_status` still
reads `1` for that voxel, and the parallel/serial equivalence test still passes.

### Docs coupling

`docs/notebooks/fitting/pyamares.md` currently passes `num_workers=1` for its main fit partly
because of this. The stated reason (pool startup exceeds the cost of eight short fits) remains true
and measured, so the page needs no change — but once this is fixed, executing the default parallel
path there becomes an option again, which would close the last open to-do in
`docs/plans/2026-07_amares_hardening.md:285-287`.

</details>

### Still open, downstream of this fix

Switching `docs/notebooks/fitting/pyamares.md` to the default parallel path — now unblocked, but a
separate call: the page's stated `num_workers=1` rationale (pool startup exceeds eight short fits)
is measured and still true, so this is a decision about what the page should *demonstrate*, not a
defect. It would close `docs/plans/2026-07_amares_hardening.md:285-287`.

---

## 3. `simulate_fid` can return `n_points + 1` samples — MEDIUM

`simulation.py:89` and `:209` build the time axis as
`np.arange(0, dwelltime * n_points, dwelltime)`, whose length is float-rounding dependent:

```python
from xmris.fitting.simulation import simulate_fid
simulate_fid(amplitudes=[1.0], frequencies=[10.0], spectral_width=3001.2, n_points=60)
# -> 61 samples
```

Verified: `sw=3001.2, n=60 → 61`; `sw=8000, n=512 → 512`; `sw=10000, n=1024 → 1024`;
`sw=2999.7, n=100 → 100`. Data and coordinate stay mutually consistent (same expression), so
nothing crashes — the array is just silently one longer than requested, which quietly breaks any
assert on `n_points`.

`fit_amares` was already hardened against exactly this expression — see the comment at
`amares.py:643-646`, which uses `deadtime + np.arange(n_time) * dwelltime` instead. `simulate_fid`
should use the same form.

### Acceptance

`simulate_fid(..., n_points=n).sizes["time"] == n` for a parametrized sweep of awkward
`(spectral_width, n_points)` pairs, pinned in `tests/test_core.py`.

---

## 4. Debt found but not addressed

Ordered roughly by value.

- **`docs/index.md` fails the docs checker with 12 errors** — 10 headers without an explicit
  `(target)=` (Commandment 8) and 2 dead `.ipynb` links (`:61`, `:132`; `myst.yml` excludes
  `notebooks/**/*.ipynb`, so both resolve to nothing with no build warning). Also one bare
  ```` ```mermaid ```` fence at `:91` where house style is the `{mermaid}` directive. The
  landing page is the most-read page in the docs.
- **`docs/notebooks/visualization/plot/03_plotting_1dfid.md` still hand-rolls its data.** A
  60-repetition `for` loop of `np.exp(...)` plus a raw hand-written prior-knowledge CSV — the exact
  pattern just removed from the fitting pages. It should use `simulate_fid` (vary `chemical_shifts`
  / `target_snr` per repetition) and `build_prior_knowledge`. It also still carries a
  `skip-execution` cell whose `num_workers=4` twin never runs, and two blocks of commented-out
  assertions (`:243-245`, `:252-254`) that should be restored or deleted.
- **Three more pages hand-roll FIDs**: `pipeline/apodization.md`, `pipeline/phase.md`,
  `visualization/widget/01_widget_phase.md`. All four hits are now reported by
  `.claude/skills/docs-page/check_docs.py` (`hand-rolled FID -- build MRS signals with
  simulate_fid`), so the backlog is greppable: `uv run python
  .claude/skills/docs-page/check_docs.py`.
- **`simulate_fid` is 1-D only.** Four pages now repeat
  `xr.concat([simulate_fid(...) for ...], dim=...)`. An N-D entry point (or a documented helper)
  would delete that boilerplate; the recipe is currently taught in
  `.claude/skills/docs-page/templates/tutorial.md` § Data instead. Design question: who names the
  stacking dimension, and does it belong in the controlled vocabulary?
- **`ruff format` drift on two untouched files**: `src/xmris/processing/fourier.py` and
  `src/xmris/visualization/widget/_static_exporter.py` (`uv run ruff format --check .`). Pre-existing
  on `main`; a formatting-only commit would clear it.
- **`fit_amares(pk)` on a single 1-D FID starts a 4-process pool by default**, costing ~2.0 s where
  the serial path takes ~0.2 s. Worth considering whether `num_workers` should collapse to serial
  when there is only one spectrum to fit.
