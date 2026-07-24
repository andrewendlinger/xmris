# AMARES hardening — code review findings (2026-07-22)

Review of branch `amares` vs `main` (merge-base `960c459`), max-effort recall pass
(10 finder angles + sweep, cross-checked against the installed pyAMARES source and by
local reproduction). Scope: **core fitting only** — `src/xmris/fitting/**`,
`src/xmris/core/{accessor,config}.py`, packaging. **Plotting / visualization is out of
scope** (see [Out of scope](#out-of-scope) for the known viz breakage, tracked separately).

> **Status — all 12 in-scope items RESOLVED (2026-07-22).** Fixed across
> `fitting/amares.py`, `fitting/prior_knowledge.py`, and `core/accessor.py`, with 8 new
> regression tests in `tests/test_core.py`. Verified: full `test_core.py` (235 passed),
> the `pyamares.ipynb` end-to-end fit (passed), and `ruff`/`mypy` clean on the changed
> files. Per-item resolution is in the **Status** column; line numbers cite the original
> pre-fix code.

## Triage summary

| # | Sev | Finding | Location | Verdict | Status |
|---|-----|---------|----------|---------|--------|
| 1 | **High** | `np.arange` time axis yields `n±1` samples → whole fit crashes | `fitting/amares.py:577` | Confirmed | ✅ Fixed — length-exact axis |
| 2 | Medium | DataFrame prior-knowledge serialized with default `index=True` → pyAMARES misparse | `fitting/amares.py:350` | Confirmed | ✅ Fixed — validate layout, clear error |
| 3 | Medium | `np.nanargmax` on all-NaN SNR → crash on masked N-D data | `fitting/amares.py:511` | Confirmed | ✅ Fixed — fall back to spectrum 0 |
| 4 | Medium | `fit`/`residuals` lose `reference_frequency` → spectral fit vars unconvertible | `fitting/amares.py:661` | Confirmed | ✅ Fixed — keep calibration attrs |
| 5 | Medium | `None` bound endpoint / value raises `TypeError`, not the friendly error | `fitting/prior_knowledge.py:114` | Confirmed | ✅ Fixed — `_as_bound`/`_num` |
| 6 | Medium | Default path blanket-mutes **all** `UserWarning` + `RuntimeWarning` | `fitting/amares.py:74` | Plausible | ✅ Fixed — message+module targeted |
| 7 | Low | Hardcoded `"spectrum"` stack dim collides with a user dim of that name | `fitting/amares.py:485` | Confirmed | ✅ Fixed — collision-free `stack_dim` |
| 8 | Low | Every-voxel failure returns a silent all-NaN Dataset when quiet | `fitting/amares.py:180` | Plausible | ✅ Fixed — `logger.error` escalation |
| 9 | Low | Positional metabolite fill trusts identical DataFrame row order | `fitting/amares.py:585` | Plausible | ✅ Fixed — `df.reindex(metabolites)` |
| 10 | Cleanup | Dead all-NaN `dummy_df` built but never read (+ duplicates label maps) | `fitting/amares.py:200` | Confirmed | ✅ Fixed — `return None` |
| 11 | Conventions | `metabolite`/`parameter` coords bypass `as_variable` (drop `long_name`) | `fitting/amares.py:622` | Confirmed | ✅ Fixed — built via `as_variable` |
| 12 | Cleanup | Accessor re-implements `fitting.__getattr__`'s lazy-import error path | `core/accessor.py:813` | Confirmed | ✅ Fixed — import via package |

---

## High

### 1. `np.arange` time axis can yield `n_time ± 1` samples, crashing the whole fit
`fitting/amares.py:577` (assignment at `:593`)

```python
timeaxis = np.arange(0, dwelltime * n_time, dwelltime) + deadtime
...
fit_norm[i, :] = uninterleave(multieq6(params, timeaxis))
```

`np.arange` with a float step computes its length as `ceil((stop-start)/step)`; the
product `dwelltime * n_time` can round just above the last multiple, so `timeaxis` has
`n_time + 1` points. `multieq6` then returns `n_time + 1` samples and the assignment into
the fixed-width `fit_norm[i, :]` raises `ValueError: could not broadcast input array from
shape (n+1,) into shape (n,)`. This is in the **main-process** reconstruction loop, *outside*
`_fit_dataset_safe`'s `try/except`, so it aborts the entire `fit_amares` call — after all
fitting has completed.

- **Repro:** `sw=3001.2 Hz, n_time=60` → `len(timeaxis)==61`; `sw=3906.25, n_time=1000` → 1001.
  `sw` is derived from coordinate spacing (or a synthesized FID), so a plausible
  `simulate_fid(spectral_width=3001.2, n_points=60)` triggers it — the user never picks the
  fragile value directly.
- **Fix:** build a length-exact axis, e.g. `timeaxis = deadtime + np.arange(n_time) * dwelltime`.
- Pre-existing formula, but load-bearing in the rewritten reconstruction and squarely in a
  touched function.

---

## Medium

### 2. DataFrame prior-knowledge serialized with pandas' default `index=True`
`fitting/amares.py:350`

```python
elif isinstance(prior_knowledge, pd.DataFrame):
    text = prior_knowledge.to_csv()   # index=True by default
```

pyAMARES reads the file with `pd.read_csv(fname, index_col=0)` (`PriorKnowledge.py:323`), so
column 0 **must** be the row labels. A DataFrame carrying a default `RangeIndex` serializes an
extra leading `0,1,2,…` column, shifting every positional column one to the right.

- **Repro:** `df = pd.read_csv("pk.csv")` (no `index_col`) then `fit_amares(df)` → pyAMARES reads
  the numeric index as the label column, the real labels/values shift right → wrong
  initials/bounds or a parse error. Exactly the positional trap the module docstring warns about.
- The DataFrame branch has **no test**.
- **Fix:** `to_csv(index=False)`, or document/require that the row labels live in the DataFrame index
  and validate it. Add a DataFrame-path test either way.

### 3. `np.nanargmax` on an all-NaN SNR array crashes template selection
`fitting/amares.py:511`

```python
snr_array = np.where(noise_region == 0, 0, signal_region / noise_region)
best_idx = int(np.nanargmax(snr_array))
```

If every spectrum's signal region contains NaN while its noise tail is non-zero, the
`np.where(..., 0, ...)` zero-branch is not taken and `snr_array` is all-NaN, so
`np.nanargmax` raises `ValueError: All-NaN slice encountered`.

- **Repro:** a masked N-D dataset (spatial voxels outside the object filled with NaN) fit with
  default `init_fid=None` → crash *before any fit runs*, defeating the function's own
  "failed fit → NaN sentinel" contract (the caller expected an all-NaN Dataset, gets an exception).
- **Fix:** guard the all-NaN case (e.g. `np.nan_to_num` the SNR, or fall back to index 0 / raise a
  clear "no finite-SNR spectrum to initialize from" error).

### 4. `fit`/`residuals` variables lose `reference_frequency`
`fitting/amares.py:661`

```python
fit_da = fit_da.assign_attrs(dict(da.attrs))
if restore_state is not None:
    fit_da = _restore_domain(fit_da, TIME_DIMS, restore_state)
fit_da.attrs = {}          # wipes reference_frequency / carrier_ppm too
```

The wholesale clear leaves the `fit` (and hence `residuals`) variable with empty `.attrs`,
while `ds["data"]` (assigned straight from `da`) keeps the physical calibration. The new
spectrum-in → spectrum-out feature makes this asymmetry reachable.

- **Failure:** for a `chemical_shift`/`frequency` input, `ds["fit"].xmr.to_fid()` / `.to_ppm()` (or any
  `@requires_attrs` op) raises `requires missing attributes ['reference_frequency']`, even though the
  identically-shaped `ds["data"]` converts fine (Dataset-level attrs don't propagate to variables).
- Clearing stale *processing* lineage (`phase_p0`, `apodization_lb`) is defensible; dropping the
  calibration needed to interpret the fit's own axis is not.
- **Fix:** keep the referencing attrs on the fit variable (whitelist `reference_frequency`,
  `carrier_ppm`) instead of clearing everything.

### 5. `None` bound endpoint / value raises `TypeError` instead of the promised friendly error
`fitting/prior_knowledge.py:114` (also `:214`, `:219`)

```python
out[VARS.chem_shift] = (
    (float(cs_bounds[0]), float(cs_bounds[1]))   # no _as_bound → float(None) explodes
    if cs_bounds is not None
    else (chem_shift - shift_window, chem_shift + shift_window)
)
```

The `chem_shift` branch bypasses `_as_bound`, so a `None` endpoint raises `TypeError` — unlike
amplitude/linewidth/phase/g, which support open sides via `_as_bound`. This contradicts the
docstring's "`None` opens a side." Similarly `float(spec.get("phase", 0.0))` / `float(spec.get("g", 0.0))`
default only on an **absent** key; an explicit `None` value hits `float(None)`.

- **Repro:** `chem_shift_bounds=(None, 1.0)` → `TypeError: float() argument must be … not 'NoneType'`,
  while the identical `amplitude_bounds=(None, 100.0)` is accepted; `{"phase": None}` (meaning "default")
  likewise `TypeError`s instead of the builder's promised at-the-door `ValueError`.
- **Fix:** route the chem_shift endpoints through `_as_bound`; treat an explicit `None` value the same as
  an absent key (or reject it with a named-peak `ValueError`).

### 6. Default (`verbose=False`) path blanket-mutes all `UserWarning` + `RuntimeWarning`
`fitting/amares.py:74`

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)
    yield
```

The two named nuisances are the scipy `xtol`/`ftol` `UserWarning` and pyAMARES's divide-by-zero
`RuntimeWarning`, but the mute suppresses the **entire** categories around the whole fit body — and
`verbose=False` is the default, so this is the default path.

- **Risk:** a genuine numerical instability (overflow, "invalid value encountered") during
  `deepcopy`/`pyamares_fitAMARES` on production data emits a `RuntimeWarning` that is silently dropped, so
  a subtly-wrong fit yields no diagnostic.
- **Fix:** filter on the specific message/module (`warnings.filterwarnings("ignore", message=..., module=...)`)
  rather than muting the whole categories.

---

## Low

### 7. Hardcoded `"spectrum"` stack dim collides with a user dimension of that name
`fitting/amares.py:485`

```python
stacked_da = da_fid.stack(spectrum=other_dims).transpose("spectrum", dim)
```

- **Repro:** a DataArray with dims `("spectrum", "time")` (plausible for a series of MR spectra) →
  `ValueError: cannot create a new dimension with the same name as an existing dimension`.
- **Fix:** use a collision-resistant private constant for the transient stack dim, or assert it is absent
  from `da.dims` first.

### 8. Total (every-voxel) fit failure returns a silent all-NaN Dataset
`fitting/amares.py:180`

The per-voxel failure notice is `logger.warning(...)`, but `_set_verbosity(False)` sets the logger to
`ERROR`, so it is dropped, and there is no aggregate escalation when the **whole** batch fails.

- **Failure:** a misconfigured prior knowledge / wrong `mhz` makes every fit raise; `first_ok` stays `None`,
  metabolites fall back to the peaklist, and the caller gets a structurally-valid but entirely all-NaN
  Dataset with no exception and no log output — easily mistaken for a genuine zero-signal result.
- **Fix:** escalate the "0 of N fits succeeded" case to a single `logger.error`/`warning` that survives the
  default level (or raise).

### 9. Positional metabolite fill trusts identical DataFrame row order
`fitting/amares.py:585` (labels sourced at `:564`–`:567`)

`metabolites` is taken from `first_ok.index` (the first successful fit); every other spectrum's
`df[col].values` is then written **positionally** into that slot order with no index alignment.

- **Risk:** if any voxel's pyAMARES `result_multiplets` returns rows in a different order than `first_ok`
  (re-sorted output, or differing from the peaklist used by the all-NaN fallback), that voxel's parameters
  are silently attached to the wrong metabolite — a scientific mislabel with no error.
- Low probability given a shared prior-knowledge peaklist, but it is silent corruption.
- **Fix:** align each df to the canonical order (`df.reindex(metabolites)`) before extracting `.values`.

---

## Cleanup & conventions

### 10. Dead all-NaN `dummy_df` built but never read
`fitting/amares.py:200` (except branch `:178`–`:203`)

`_fit_dataset_safe`'s except branch constructs a 16-column all-NaN `dummy_df`, but the consumer
(`:564`–`:566`, `:581`) treats `df is None`, an all-NaN df, and `spectrum_max == 0` identically (skip,
keep the NaN sentinel) — so `return None` behaves the same. The dead branch also re-hardcodes the exact
pyAMARES column labels (incl. the trailing-space `"CRLB(cs%) "`) that already live in
`_VALUE_COLS`/`_UNCERTAINTY_COLS`, a second copy that can drift.

- **Fix:** `return None` in the except branch; delete the dummy construction and its duplicate label list.

### 11. `metabolite` / `parameter` coordinates bypass `as_variable` (drop `long_name`)
`fitting/amares.py:622` (and `:637`, `:644`, `:653`)

Commandment 7 (`ai_context.md`): *"When creating new coordinates … use the internal `as_variable(TERM, dim, data)`
helper to bundle data and metadata."* The new coords are built via plain `coords={DIMS.metabolite: …}` /
`{DIMS.parameter: …}`, so `DIMS.metabolite`/`DIMS.parameter`'s `long_name` metadata is dropped.

- Note: the per-parameter **data vars** (`amplitude`, `crlb`, …) are data vars, not coords, so Commandment 7
  does not bind them — this is strictly the two coordinates.
- **Fix:** wrap both coordinate arrays with `as_variable` before `.assign_coords`.

### 12. Accessor re-implements the fitting package's lazy-import error path
`core/accessor.py:813`

```python
try:
    from xmris.fitting.amares import fit_amares as _internal_fit_amares
except ImportError as e:
    from xmris.fitting import MISSING_FITTING_DEP_MSG
    raise ImportError(MISSING_FITTING_DEP_MSG) from e
```

This duplicates exactly what `xmris.fitting.__getattr__` already does (catch the pyAMARES `ImportError`,
re-raise `MISSING_FITTING_DEP_MSG`). Two copies of the friendly-error wrapping can diverge.

- **Fix:** `from xmris.fitting import fit_amares as _internal_fit_amares`, routing through the single
  package resolver.

---

## Checked and cleared (not findings)

Verified against the installed pyAMARES source and by local reproduction:

- **`g_global` `0.0` vs `False`** — pyAMARES uses `if g_global is False:` (`PriorKnowledge.py:359`), so the
  two are correctly distinguished; forwarding the value as-is is fine.
- **Result-column labels** — `_VALUE_COLS` / `_UNCERTAINTY_COLS` (incl. the trailing-space `"CRLB(cs%) "`)
  match pyAMARES's `report.py` exactly; extraction is correct.
- **ppm/Hz round trip** — `_restore_domain` reassigns the original coordinate verbatim
  (`validation.py:212`), so `ds["data"] - ds["fit"]` aligns element-wise (coords identical, no NaN pad).
- **Lazy-import guards** — `import xmris` does not pull in pyAMARES; `xmris.fit_amares` /
  `xmris.fitting.fit_amares` / the accessor all raise the friendly `xmris[fitting]` error when absent;
  `os.unlink`-in-`finally` and both PEP-562 `__getattr__` hooks are correct.
- **Packaging** — `scipy` is genuinely used by core (`baseline`/`phasing`/`bruker`), `joblib` appears only
  under `fitting`, and `pandas` (newly imported eagerly in `accessor.py`) is guaranteed transitively by xarray.
- **carrier / absolute-ppm** — the `ppm_offset=-carrier` + add-back is consistent with the pinned pyAMARES
  (pinned by `TestFittingDomain.test_carrier_enables_absolute_ppm`); flagged only as version-coupled, not a bug.

## New vocabulary introduced (flagging per CLAUDE.md)

- `ATTRS.amares_amplitude_scale` (`config.py:299`, unit `a.u.`) — numeric normalization factor; a valid
  quantitative lineage attr (Commandment 3 clean).
- `DIMS.parameter` (`config.py:340`) — axis the `crlb`/`sd` uncertainty vars span.
- `VARS.sd` (`config.py:432`) — per-parameter standard deviation.

All three are correctly declared (unique, frozen) and consumed via the singletons. Removal of the old
string-flag attrs (`fit_method` / `prior_knowledge_file` / `amares_version`) is a correct Commandment 3
improvement; nothing descriptive was re-added.

<a id="out-of-scope"></a>
## Out of scope — plotting / visualization (tracked separately)

Excluded from this review at the reviewer's request, but recorded so the state is on file. The
`fit_amares` output-schema rename (`raw_data`/`fit_data` → `data`/`fit`, `Metabolite` → `metabolite`,
`crlb` gained a `parameter` axis) was **not** propagated to its consumers:

- `visualization/plot/plot_qc_grid.py` (`.xmr.plot.qc_grid()`) — `required_vars` still lists `fit_data`;
  also assumes FIDs (breaks spectrum-in) and `np.nanmax`es CRLB over the new `parameter` axis.
- `visualization/plot/plot_trajectory.py` (`.xmr.plot.trajectory()`) — reads the removed `Metabolite`
  dim; the `crlb` `parameter` axis breaks the error-band math.
- `docs/notebooks/visualization/plot/03_plotting_1dfid.md` — exercises both → `uv run test` goes red.

`docs/plans/amares_handoff.md` already flags this as **known and deliberately deferred** ("rewrite-not-patch").
Note the handoff's claim that `plot_trajectory.py` is "not a consumer" is inaccurate — it uses
`.sel(Metabolite=…)` and is broken too.

**Resolved (`40f84a4`).** The rename *was* propagated to both consumers and the notebook assert —
the touched lines now route through `DIMS`/`VARS` and select `parameter="amplitude"` off the new
`crlb` axis, restoring the old error-band / QC-flag semantics. `03_plotting_1dfid.md` is green again.
The one genuinely separable piece — `plot_qc_grid`'s pre-existing spectrum-in assumption (unconditional
`to_spectrum`, broken on `main` too) — stays deferred, now tracked as
[#106](https://github.com/andrewendlinger/xmris/issues/106).
