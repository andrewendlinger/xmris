# Bring pyAMARES integration together: harden the adapter, go official, close the issue tree

## Context

Integrating pyAMARES into xmris has been the project's biggest pain point. The
fitting layer (`src/xmris/fitting/amares.py`) works on clean synthetic data but
is unsafe on real Bruker data: it can **silently return the prior knowledge as
if it were a converged fit**, its output ignores the config vocabulary, its
prior-knowledge input is a trap-laden positional CSV, and it depends on a
personal **git fork** of pyAMARES that blocks a clean `pip install xmris`.

The evidence is unusually complete:

- A real downstream project (`/Users/andre/PhD/projects/11_EAE_mice`, HP-¹³C EAE
  mice) treats `.xmr.fit_amares()` as *unsafe to call directly* and routes
  everything through a local wrapper `eae_mice/amares.py`, backed by a 16-entry
  bug tracker `XMRIS_BUGS.md`. This wrapper is **inspiration, not a blueprint**:
  it precisely documents *what* breaks and *why*, and proves every fix can live in
  xmris-space (none patch pyAMARES internals) — but where a cleaner in-library
  approach exists, take that rather than transcribing the workaround.
- The maintainer has already triaged this into a clean issue tree:
  **#19** (parent: "fitting must become more robust") → **#68**, **#69**,
  **#80**, **#81**, **#82**; plus **#70** (packaging/fork), **#67** (release
  epic incl. coverage 11%→70%), and tracking **#71 / #88 / #34**.

**The fork question — resolved.** The `xmris-compatible` branch is exactly
**3 commits touching `setup.py` only** (an `hlsvdpro` PEP 508 marker + `numpy<2`/
`pandas<2.2` caps); **zero algorithm/kernel changes**, and it fixes **none** of
the robustness bugs. It buys xmris nothing behaviorally and blocks PyPI
publishing. **Decision (user-confirmed): go official pyAMARES + harden the xmris
adapter.** Every real fix belongs in the adapter regardless of fork-vs-official.

**Intended outcome:** `da.xmr.fit_amares(...)` is safe and ergonomic on real
data out of the box; output speaks the config vocabulary; priors are built
in-memory with validation; fitting is an optional extra on official pyAMARES;
and `fitting/amares.py` coverage clears the release bar.

## Foundation: the gating PRs have merged (status 2026-07-20)

The `amares` branch is now fast-forwarded to `main` (`960c459`); everything below
branches from current `main`, not an open PR. What landed **inverts one
load-bearing assumption this plan was originally written on.**

**#96 shipped a canonical-only vocabulary — the *opposite* of the alias machinery
this plan assumed.** The team built exactly the `aliases=(...)` /
`read_attr(obj, term)` approach, watched it break on the `MHz` ↔
`reference_frequency` case (one reader checked the alias, another didn't), and
**deliberately rejected it.** `docs/explanation/vocabulary.md` (#98/#100) is the
published decision record. So there is **no `read_attr`, no `_first_matching_name`,
no `aliases`** anywhere. What actually exists in `core/config.py`:

- **Frozen `XmrisTerm`** (`__setattr__`/`__delattr__` raise) + **import-time
  canonical-value uniqueness** (`BaseVocabulary.__init_subclass__`). Declare new
  terms in `config.py`; never mutate one.

**Consequence for every workstream below (a reversal of the old guidance):** read
canonical keys **directly** — `da.attrs[ATTRS.reference_frequency]`, gated by
`@requires_attrs(...)` — and do **not** add any alias / back-compat tolerance for
foreign attr names. That is now forbidden by design: the vocabulary is a fixed
point, and conforming foreign data onto it is the *user's* one-time job
(`fid.rename(...)` / `assign_attrs(...)`, and eventually the planned-but-unshipped
`da.xmr.map_vocab(...)` helper — out of scope here).

**Already done — remove from scope:**
- **#68 / BUG-006** — done in **#93** (its own PR, not folded into #96):
  `fit_amares` reads `da.attrs.get(ATTRS.reference_frequency)` and the error points
  at the modern key. The current code has **no `MHz` fallback** — already
  canonical-only clean.
- **BUG-013** — `DIMS.average / repetition / coil` are singular in `config.py`
  (no plural alias — canonical-only), retiring the `DIMS.averages` `AttributeError`.
- **BUG-002** — measured group-delay (`remove_digital_filter(group_delay="measure")`
  + `estimate_group_delay()`), shipped in #85.

**Also merged, adjacent (not in this plan's scope, but they touch nearby files):**
**#92** (part of #67 — killed the import-time `DeprecationWarning`, dropped the
legacy `DEFAULTS` shim from `__init__.py`, gated ruff in CI) and **#94** (#83 —
plotting now guards complex input + preserves the time-axis units).

## Pivot (2026-07-20): fitting meets your data in either domain

Reading the draft design doc, the user reversed a stance it took. Fitting should
**not** be the lone pipeline step that refuses a spectrum and makes you `to_fid()`
by hand. `fit_amares` will **auto-convert domains** and **return results in the
representation you handed it**: FID in → time-domain `data`/`fit`/`residuals`;
spectrum in → those three vars come back as spectra (ppm in → ppm out). The fitted
parameters (`amplitude`, `chem_shift`, …) are domain-independent throughout.

**This is the domain-*preserving* contract** (`computes_in`, like `apodize_exp` /
`zero_fill`) — compute in the time domain, restore the input's representation — not
the funnel. The forward model is still time-domain; the contract only removes the
manual bookkeeping, exactly as it does for apodization. So the design doc's "fitting
is explicit, the transform is part of the model" framing **flips**: the domain
contract makes that transform *legible* (readable per-axis in the `repr`), not hidden.

**Why it can't just wear `@computes_in`:** the decorator's `_restore_domain`
(`core/validation.py`) operates on a single DataArray; `fit_amares` returns a
**Dataset**, and transforming it back wholesale would FFT the parameter vars
(`amplitude` over `metabolite`) into nonsense. So `fit_amares` orchestrates the round
trip itself — reusing the engine's converter-routing so inserted transforms stay
bit-identical — and restores **only** the time-domain data vars. Detailed in
workstream **G**.

## Bug → issue → fix-location map

| EAE bug | Issue | Fix location | Status |
|---|---|---|---|
| BUG-001 scale trap (returns prior, no error) | #80 §1 | adapter: normalize/rescale internally | **in scope** |
| BUG-001 sub: failed fits look like zero-signal | #80 §1 | adapter: NaN sentinel (not `np.zeros`) | **in scope** |
| BUG-009 `initialize_with_lm=True` diverges | #80 §2 | adapter: default `False` | **in scope** |
| BUG-005 phase-NaN / iloc / TIE-order traps | #82 | new PK builder + validation | **in scope** |
| BUG-008 trailing-digit peak name → multiplet | #82 | PK builder validation | **in scope** |
| BUG-004 carrier / absolute-ppm unusable | #81 | adapter: anchor ppm + forward `carrier` | **in scope** |
| BUG-007 `deadtime` default ≠ docstring | #81 | adapter: align default/docs | **in scope** |
| BUG-010 `verbose=False` leaks 3 of 4 channels | #81 | adapter: logging + call-time tqdm | **in scope** |
| BUG-011 export/kwargs/`g_global`/CRLB cols | #81 | adapter + `fitting/__init__` | **in scope** |
| vocab drift `raw_data`/`fit_data`/`Metabolite` | #69 | rename to `VARS`/`DIMS` + consumers | **in scope** |
| `dim="time"` literal, literal lineage attrs | #71/#88 | `DIMS.time` default, drop string attrs | **in scope** |
| BUG-003 `simulate_fid` phases radians-not-deg | (unfiled) | docstring/units fix (stays top-level fn) | **in scope** |
| git-fork dependency, no optional extra | #70 | pyproject + import guards + upstream | **in scope** |
| coverage 11%; parallel path untested | #67/#19 | notebook + arch tests | **in scope** |
| fitting rejects spectral input (papercut) | (pivot) | G: domain-preserving auto-convert | **in scope (new)** |
| BUG-006 MHz vs reference_frequency | #68 | — | **done (#93)** |
| BUG-013 `DIMS.averages` AttributeError | — | — | **done (singular dims, no alias)** |
| BUG-002 group-delay under-count | — | — | **done (#85)** |
| BUG-012/014/015 plotting; BUG-016 AUC | — | separate subsystems | **out of scope** |

## The work (stacked PRs off current `main`, issue-aligned)

All the target bugs are still live in `fit_amares` as read on `main` today (magic
`dim="time"`, `initialize_with_lm=True`, `normalize_fid=False`, `np.zeros`
pre-alloc + `continue` on failure, `raw_data`/`fit_data`/`"Metabolite"` outputs,
`fit_method`/`prior_knowledge_file`/`amares_version` lineage strings). #96 only
tidied it (dropped some `str()` wrappers). Nothing here is pre-empted.

**Legacy code is rewritten, not patched (user, 2026-07-20).** The visualization /
plotting layer (`visualization/plot/*`) and the notebooks that drive it are legacy.
The priority is a **well-written amares core** (`fit_amares` + PK builder + robustness
+ domain pivot + output vocab); the surrounding consumers get **rewritten against the
new output contract**, not delicately patched to keep the old names/domains alive.
Concretely, in E and G "update consumers" means *land the clean core, then rewrite the
legacy viz separately* — keeping only the tests/docs that must stay green in step.
Never let a legacy plotter's assumptions (e.g. `plot_qc_grid` FFT-ing `data`/`fit`)
shape the core API.

Workstreams **B, D, E all rewrite the body of `fit_amares`**, so develop them as
a rebased stack (squash-merge, `--onto` per the merge-workflow note) to avoid
re-resolving the same conflicts. `eae_mice/amares.py` is **inspiration, not ground
truth** for B, C, D — mine it for the failure modes and constraints it documents,
then design the cleanest xmris-native wrapper (e.g. prefer pyAMARES's own
`normalize_fid`/`scale_amplitude` remedies where they beat a manual
normalize-and-rescale).

**Follow the project skills (do not hand-roll the pattern):**
- **`new-processing-method`** governs every code change to `fit_amares`, the PK
  builder, and `config.py`: functional purity, config-vocab only (no magic
  strings) inside the package, `dim: str = DIMS.time` default (per workstream G,
  fitting now honors the **domain-preserving** contract — it is no longer
  "undecorated"; the biconditional keeps the default a config constant because
  `TIME_DIMS` is single-label), `_check_dims` validation, lineage attrs =
  quantitative parameters only,
  `as_variable` for coords, fully-typed NumPy docstrings, accessor + `__all__`
  registration, and **explicitly flag every new `ATTRS`/`DIMS`/`COORDS`/`VARS`
  term** to the user.
- **`new-doc-notebook`** governs all notebook work (workstream F): `.md` under
  `docs/notebooks/fitting/`, reader-facing cells use **plain strings**
  (`"time"`) with config singletons only inside hidden `remove-cell` assert
  cells, seeded RNG for noisy asserts, exact frontmatter/H1/target rules, and a
  **mandatory `docs/myst.yml` TOC entry** (except `testonly_` files).
- **`design-doc`** (new, from #97) governs the decision record. Triggers hit: new
  vocabulary (`amares_amplitude_scale`, CRLB `VARS`), a ≥2-approach call (official
  vs fork; normalize-and-rescale vs pyAMARES's own `normalize_fid`), a multi-PR
  chain. **User decision: a separate prose explainer `docs/explanation/fitting.md`
  — NOT folded into `pyamares.md`.** The tutorial stays a concise usage
  walkthrough (design rationale would frustrate a reader who just wants to fit);
  the "why" lives in its own article and cross-links to workstream F's robustness
  notebook, which *proves* what the explainer *states*. Style exemplar:
  `docs/explanation/vocabulary.md`. Two-pass discipline: **pass 1** drafted from
  this plan as the chain's first commit, **pass 2** reconciled against the built
  code as the last (`ASSUMPTION:` grep clean, every snippet checked against the
  shipped API). Detailed in workstream F.

Note on surface shape: `simulate_fid` and the PK builder are **constructors**
(they don't take a DataArray), so they stay **top-level `xmris.*` / `xmris.fitting.*`
functions, not `.xmr` methods**; `fit_amares` remains the only fitting accessor
method.

### A — Packaging & fork exit (#70) — foundation, low risk
- `pyproject.toml`: add `[project.optional-dependencies] fitting = ["pyamares>=…"]`;
  remove bare `pyamares` from core `dependencies`. Move the fork's caps into
  xmris's own deps: `numpy>=1.26,<2.0`, add `pandas<2.2` (xmris already pins
  `xarray<2025.11.0`).
- Guard imports: `src/xmris/__init__.py` currently does an **unconditional**
  `from .fitting.amares import fit_amares` (hard-fails `import xmris` with no
  pyAMARES). Make it lazy/guarded; keep `simulate_fid` eager (no pyAMARES dep).
  Export `fit_amares` from `src/xmris/fitting/__init__.py` (`__all__` today lists
  only `simulate_fid`) behind an import guard. Accessor already guards
  (`core/accessor.py:791-797`).
- Fork exit (2 steps): **(a) now** — upstream the one-line `hlsvdpro` PEP 508
  marker to `HawkMRS/pyAMARES` (fork commit `cfb3a8c`, "Resolves #15"); it's the
  only fork change that must live in pyAMARES's `setup.py`, and `hlsvdpro` is
  optional (bundled `hlsvdpropy` fallback). **(b) later** — once a PyPI
  `pyamares ≥0.3.33` includes it, drop `[tool.uv.sources]` and pin the official
  package. Until then the git source stays but is no longer load-bearing for
  behavior. Update `CLAUDE.md`'s pyamares gotcha + `requires-python` rationale.

### B — Robustness: kill the silent-failure modes (#80) — the critical safety fix
Prior art to learn from (not copy): `fit_amares_scaled` + `amares_failed_mask` (`eae_mice/amares.py:254-327`).
- **Internal normalization** (BUG-001): normalize the FID to unit max **as a
  single global factor across the whole array** (never per-spectrum — that would
  flatten a dynamic series), fit, then rescale `amplitude` + the time-domain vars
  back into input units. Record the factor in a **new registered attr**
  (`ATTRS.amares_amplitude_scale`, flag below). This defeats pyAMARES's
  scale-dependent `tol = sqrt(max|fid|)*1e-6` (`kernel/lmfit.py:440`) without
  forking.
- **NaN sentinel** (BUG-001 sub): pre-allocate outputs with `np.nan`, not
  `np.zeros`, so a failed/degenerate fit is distinguishable from a real
  zero-signal spectrum (downstream detection collapses to `isnan`, retiring
  `amares_failed_mask`). Handle `max|fid|=0` spectra explicitly → NaN, no
  divide-by-zero.
- **Default `initialize_with_lm=False`** (BUG-009) in both free fn and accessor.
- Optional: detect "terminated at the initial guess with prior's exact values"
  and emit a warning.

### C — Prior-knowledge builder + validation (#82) — usability (user chose "builder")
Prior art to learn from (not copy): `write_prior_knowledge` (`eae_mice/amares.py:90-191`).
- New **top-level `xmris.fitting` free function** (a constructor: spec → validated
  pyAMARES params, *not* a `.xmr` method) that takes a dict / DataFrame /
  lightweight spec, so users never hand-write the positional CSV. `fit_amares`
  gains a `prior_knowledge=<dict | DataFrame | path>` argument that routes
  in-memory specs through it; keep CSV/XLSX-path input but **validate at the door**.
- Bake in the hard-won rules: always emit explicit `(-180,180)` phase bounds
  (BUG-005 phase-NaN trap); reject non-letter peak names / require explicit
  multiplet syntax (BUG-008); write the TIE anchor peak first (column-order
  `UnboundLocalError`); document the `iloc` positional format.
- No new vocabulary needed here: metabolite/peak names are the **values** of the
  `DIMS.metabolite` coordinate, not a term (see the Vocabulary section).

### D — API / UX surface (#81)
- **carrier / absolute-ppm** (BUG-004): let users pass literature absolute ppm;
  xmris anchors internally (EAE's `anchor_hz/mhz + (ppm-ppm_anchor)` approach) and
  forwards `carrier` where valid — at minimum document the carrier-relative axis.
  **Reuse the existing `ATTRS.carrier_ppm` + `ATTRS.reference_frequency`** for the
  anchoring; do not mint a new carrier term (vocab-overlap check).
- **`deadtime`** (BUG-007): keep the coordinate-derived default (`coords[dim][0]`
  — the single source of truth) and **fix the docstring** to match; do **not**
  mint an `ATTRS.deadtime`. See the Vocabulary section for why (group_delay
  overlap). Likewise derive `sw` from the coordinate, not a stored attr.
- **`verbose=False` silences all 4 channels** (BUG-010): replace `print()` with
  module logging (`amares.py:92,199,309`); disable tqdm at **call time** (pass
  `disable=`, don't rely on import-time `TQDM_DISABLE`); set the `pyAMARES` logger
  level; filter the routine scipy/pyAMARES warnings. Make it hold for
  `num_workers>1` via a joblib worker initializer (EAE's suppressor is in-process
  only, forcing `num_workers=1`).
- **Bundle** (BUG-011): reconcile accessor `**kwargs` vs free-fn `verbose`
  (real signature, no silent `TypeError`); forward `g_global`; surface the
  per-parameter sd/CRLB columns pyAMARES already computes (new `VARS`, flag below).

### E — Output vocabulary compliance (#69, #71, #88) — user chose "clean rename now"
- Rename output data vars: `raw_data` → `VARS.original_data` (**value `"data"`**),
  `fit_data` → `VARS.fit` (`"fit"`), keep `VARS.residuals`; params already match
  `VARS.amplitude/chem_shift/linewidth/phase/crlb/snr`.
- Rename the `"Metabolite"` dim → `DIMS.metabolite` (`"metabolite"`).
- `dim` default `"time"` → `DIMS.time` (#71 item 8). Drop the descriptive
  lineage attrs (`fit_method` / `prior_knowledge_file` / `amares_version`) per
  Commandment 3 — see the Vocabulary section (#71 item 9).
- **Update every consumer in the same PR** (canonical-only means no fallback —
  `.sel()`/`ds[...]` raise on the old key). Verified consumers on `main`:
  `visualization/plot/plot_qc_grid.py:112` (`required_vars = ["fit_data",
  "raw_data", "crlb"]` + the `to_spectrum` calls at 155–156); the `'Metabolite'`
  docstring at `core/accessor.py:784`; and two notebooks —
  `docs/notebooks/fitting/pyamares.md` (many `raw_data`/`fit_data`/`Metabolite`
  refs) and `docs/notebooks/visualization/plot/03_plotting_1dfid.md:268` (`.sel(
  Metabolite="PCr")`). (`plot_trajectory.py` reads only param vars, which keep
  their names — not a consumer.) Reader-facing notebook cells become plain
  lowercase `ds.data` / `ds.fit` / `.sel(metabolite=...)` — the new canonical
  values *are* those strings, so no config import leaks into visible cells. The
  out-of-repo `eae_mice/amares.py` + `XMRIS_BUGS.md` also need a matching update
  (hand off to the user).
- Scope note: the `plot_*.py` **source** is legacy (rewrite per the principle above,
  not patch). What must stay green in-PR is the arch/notebook **tests** and the
  tutorial; the QC/trajectory plotters are rewritten against the new contract after
  the core lands, so E's rename doesn't drag a legacy-plotter refactor into its PR.

### F — Tests, docs, coverage (#67, #19, #34)
- Raise `fitting/amares.py` coverage **11% → ≥70%**; **un-skip the parallel
  path** (`pyamares.md` currently runs only `num_workers=1`; the default `loky`
  path is `skip-execution`).
- Per `design-doc` + `new-doc-notebook`, three doc artifacts:
  - **New explainer `docs/explanation/fitting.md`** (the design record — user
    asked for it separate so the tutorial stays lean). Prose, no kernel. Driving
    question: *"I fit my Bruker FID and got back my prior — why?"* → the scale trap
    (`tol=sqrt(max|fid|)*1e-6` fed to scipy as a relative tol), normalize-and-rescale,
    the NaN-vs-`np.zeros` sentinel, and the official-vs-fork call — walked through
    as pedagogy (paired ❌/✅), never a dry "Alternatives" list. Cross-link ↔ the
    robustness notebook below; register in `docs/myst.yml`.
  - **Extend `docs/notebooks/fitting/pyamares.md`** — keep it a *concise usage
    tutorial* (reader-facing plain strings): demo the PK builder, the parallel
    path, and safe-on-real-scale usage; push the *why* to the explainer with a
    one-line `seealso`, don't inline the rationale. Hidden `remove-cell` asserts
    (config singletons only) prove math + coords + `VARS`/`DIMS` keys + lineage attrs.
  - **Add a `testonly_` notebook** (`docs/notebooks/fitting/testonly_amares_robustness.md`,
    excluded from the `docs/myst.yml` TOC) — the explainer's *proof*: scale-trap
    (Bruker-scale ~1e7 synthetic FID converges, does *not* echo the prior), NaN
    sentinel on an all-zero spectrum, PK-builder validation (explicit phase bounds,
    digit-name rejected, TIE order), `verbose=False` silence. Seed all noise
    (`np.random.default_rng(42)` / fixed `target_snr` seed).
  - Scalar invariants (new `config.py` terms, uniqueness, the `fit_amares`
    dim-default rule) go in `tests/test_core.py`.
- **BUG-003**: fix `simulate_fid`'s `phases` doc/units (it's radians; fit output
  phase is degrees). It stays a **top-level constructor** (`xmris.simulate_fid`,
  correctly *not* a `.xmr` method) — doc/units fix + a hidden assert, no accessor
  change.
- Optional consolidation: reconstruct the model FID via xmris's own
  `_simulate_fid_ndarray` instead of pyAMARES `multieq6`/`uninterleave`, removing a
  coupling point and guaranteeing simulate/fit agree (verify numerically equal
  first; keep low-priority).
- #34: de-duplicate the accessor vs free-fn docstring (single canonical source).

### G — Fitting meets either domain (domain-preserving pivot) — NEW (2026-07-20)
Rewrites the same `fit_amares` body as B/D/E — land it in that stack. User decided:
auto-convert **and match the input representation** (see the Pivot section above).
- **Behavior:** `fit_amares` accepts a FID **or** a spectrum. Detect the input
  representation; if spectral, round-trip to a FID for the fit (ppm → `to_hz` →
  `to_fid`; Hz → `to_fid`), then convert `data`/`fit`/`residuals` back to that
  representation (`to_spectrum` → `to_ppm` if ppm). Leave the parameter vars
  (`amplitude`/`chem_shift`/`linewidth`/`phase`/`crlb`/`snr`) untouched. FID in →
  unchanged (time-domain out), so the tutorial and existing callers are unaffected.
- **Reuse, don't reinvent:** route through the engine's converter helpers
  (`_coerce_to_domain` / `_restore_domain` / `_strict_domain_error` in
  `core/validation.py`, applied per-DataArray-var) or the public converters — never
  inline `fft`/`ifft` (Commandment 6). Real-valued spectral input → the shared
  `_real_valued_spectral_error`; `set_options(auto_convert=False)` → the shared
  strict recipe. `residuals` stays `data - fit` in whichever domain is returned.
- **Keep** `dim: str = DIMS.time` (single-label domain → the biconditional keeps the
  config-constant default; `test_dim_defaults_follow_biconditional` needs no change).
- **Consumer ripple (legacy — rewrite, don't patch):** `plot_qc_grid.py` FFTs
  `data`/`fit` assuming they are FIDs, which breaks for a spectrum-in fit. Per the
  legacy-rewrite principle above, the domain-aware handling arrives with the viz
  **rewrite**, not a patch — don't contort the plotter to limp along. The core lands
  correct; the viz follows.
- **Tests** (`tests/test_core.py`): remove `fit_amares` from
  `test_undecorated_by_design` (1348) and fix that class's docstring ("fitting stay
  undecorated"); add a `TestFittingDomain` with behavior smokes — FID→time-domain
  vars, Hz-spectrum→`frequency` vars, ppm-spectrum→`chemical_shift` vars,
  real-spectrum refused, strict-mode raises the recipe, params present in every case,
  and a fit-a-FID-vs-fit-its-spectrum numerical-equivalence check.
- **Docs — the contract reversal (same PR):**
  - `docs/explanation/domains.md`: move `fit_amares` out of "Explicit operations —
    no magic" into the **domain-preserving** class (keep `fft`/`ifft`/`phase` as the
    explicit primitives); add a `fit_amares()` row to the "at a glance" table; add a
    contributor aside that it hand-rolls the round trip because it returns a Dataset.
  - `docs/contributing/ai_context.md` Commandment 6 + `.claude/skills/new-processing-method/SKILL.md`:
    drop "fitting stays undecorated"; fitting is domain-preserving (hand-rolled for
    its Dataset output).
  - `docs/explanation/fitting.md` (the design doc): flip the opening ("never converts"
    → "meets your data in either domain") and add a **"Pass a FID or a spectrum"**
    section reconciling the old "transform is part of the model" worry with the domain
    contract, cross-linking `domains.md`. Since the draft is uncommitted, this is a
    plain edit now, then the pass-2 reconcile as usual.

## Vocabulary: reconcile with existing terms before adding (Commandment 4 + #96 uniqueness)

Every candidate term is checked against the existing `ATTRS`/`DIMS`/`COORDS`/`VARS`
**and** must pass #96's import-time uniqueness (each canonical value distinct
within its vocabulary — there are no aliases to also deconflict) before it's
minted. Guiding rule: **derive, don't duplicate.**

**The `deadtime` ↔ `group_delay` overlap (user-flagged) — resolved by deriving from the coordinate.**
- `ATTRS.group_delay` / `group_delay_removed` (**samples**, from #85/#89) are a
  *DSP-filter* concept: the receiver digital-filter delay removed by
  `remove_digital_filter` **before** fitting. AMARES `deadtime` (**seconds**) is
  the *acquisition* time-origin the fit models. They are **physically distinct** —
  `deadtime` is *not* `group_delay` in other units — so they must not be merged or
  made to shadow one another.
- Both `deadtime` and `sw` (`=1/dt`) are **fully determined by the `time`
  coordinate** (`COORDS.time`, unit s), which is xmris's "xarray-in/out" source of
  truth and which `remove_digital_filter` already resets. So keep them as
  **override *parameters* on `fit_amares`, defaulting to values read from the
  coordinate** — do **not** create `ATTRS.deadtime` or `ATTRS.spectral_width`
  (that would duplicate the coordinate and risk drifting from the group_delay
  lineage). This is also the principled fix for **BUG-007** (keep coord-derived
  default, correct the docstring).
- Apply the same reconciliation to `simulate_fid`'s ad-hoc `"spectral_width"` /
  `"dead_time"` attrs: derive on read; keep only genuine sim provenance, clearly
  namespaced (e.g. `sim_*`), never colliding with a real vocabulary key.

**Terms genuinely worth adding** (declared in `config.py`, each flagged to the user):
- `ATTRS.amares_amplitude_scale` (a.u.) — the normalization factor (workstream B);
  new concept, no overlap.
- `VARS` per-parameter uncertainties pyAMARES already computes, e.g.
  `amplitude_sd` / `crlb_*` (workstream D) — genuinely new outputs.

**Terms to drop rather than add** (Commandment 3 — no string/flag bloat):
- `fit_method` / `prior_knowledge_file` / `amares_version` are provenance strings,
  not quantitative parameters — omit them (at most keep `amares_version` as a
  deliberate provenance exception). Metabolite names live as the **values of the
  `DIMS.metabolite` coordinate**, not an attr, so the PK builder needs no
  `ATTRS.prior_knowledge` term.

## Open questions & risks (flag before implementation)

Surfaced reviewing the plan — settle each before/early in its workstream:

1. **In-memory PK vs pyAMARES's file API (C).** `initialize_FID` takes a
   `priorknowledgefile` *path* today, and EAE's `write_prior_knowledge` writes a CSV —
   so verify whether pyAMARES accepts an in-memory table (DataFrame / `lmfit.Parameters`)
   or the "in-memory builder" must still serialize to a temp file internally. Decides C's
   whole shape.
2. **Fitting tests need the optional extra (A × F).** Making `pyamares` an optional
   `[fitting]` extra means the notebook + `TestFittingDomain` tests only run with it
   installed — CI must `uv sync --extra fitting` for those, **plus** a separate
   no-extra `import xmris` smoke so the guard is actually exercised.
3. **Per-parameter uncertainty output shape (D + vocab).** pyAMARES computes `sd` and
   CRLB for *every* parameter (amp/cs/LW/phase/g). Flattening all of them into `VARS`
   (`crlb_amplitude`, `crlb_chem_shift`, …, `sd_*`) bloats the vocabulary — decide
   explode-into-terms vs. a `parameter`/`statistic` dim carrying one `crlb`/`sd` var,
   *before* minting terms.
4. **`initialize_with_lm=False` is a behavior-changing default (B).** Flipping it shifts
   results for existing callers (incl. EAE pilots) — intended, but call it out in the
   changelog and the real-data sanity check.
5. **Complex-input / `component`-split data.** AMARES needs complex FIDs; the domain
   funnel covers time↔spectral but not real/imag-split (`DIMS.component`) input. Decide
   whether `fit_amares` requires complex and errors clearly, or auto-`to_complex`s.
6. **Docs paused (user).** `fitting.md` review + the robustness notebook are on hold;
   focus is the plan + core code. Note: per *author-for-the-render*, the `ASSUMPTION:`
   HTML comments in `fitting.md` are invisible on the built site — pass-2 bookkeeping,
   not review annotations.

## Verification (end-to-end)
- **Architecture:** `uv run pytest tests/test_core.py -n0 --no-cov` — incl.
  `TestDomainDimRule` and new fitting-vocabulary/PK-builder tests.
- **Science/notebooks:** iterate fast with `uv run test-gen` then
  `uv run pytest "tests/autogen_notebooks/fitting/<name>.ipynb" -n0 --no-cov`;
  full pass with `uv run test`. The extended `pyamares.md` + the new
  `testonly_amares_robustness.md` are the integration tests (parallel path, scale
  trap, NaN sentinel, PK builder, silence). Commit only the `.md`.
- **Design doc (pass 2):** `git grep -n "ASSUMPTION:" -- 'docs/**/*.md'` is empty;
  every snippet in `docs/explanation/fitting.md` matches the shipped `fit_amares`
  signature; the explainer is in the `docs/myst.yml` TOC and cross-links the
  robustness notebook both ways.
- **Domain pivot (G):** `uv run pytest tests/test_core.py -n0 --no-cov` covers the
  updated `TestDomainRollout` + new `TestFittingDomain`; smoke that
  `spec.xmr.fit_amares(pk)` returns spectral `data`/`fit` while
  `fid.xmr.fit_amares(pk)` returns time-domain, with **identical parameters**; and
  that `set_options(auto_convert=False)` makes a spectral fit raise the `to_fid()`
  recipe. Confirm `plot_qc_grid` renders for both a FID-fit and a spectrum-fit Dataset.
- **Coverage:** confirm `fitting/amares.py ≥70%` in the cov report.
- **Types/lint:** `uv run mypy src/xmris`; `uv run ruff check . && uv run ruff format .`.
- **Packaging:** `uv sync --extra fitting` works; **`import xmris` succeeds with
  pyAMARES absent** (guard) and `.xmr.fit_amares` raises the friendly `ImportError`;
  `uv build --no-sources` yields a wheel whose METADATA carries `pyamares` under
  the `fitting` extra only.
- **Real-data sanity (optional, out-of-repo):** re-run an EAE pilot through the
  hardened `.xmr.fit_amares` and confirm it reproduces the wrapper's numbers
  (normalized Lac/Pyr ≈ 0.5, CRLB ≈ 2%), then delete the corresponding workarounds
  from `eae_mice/amares.py`.
