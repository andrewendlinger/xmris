# AMARES hardening — handoff

Branch `amares` (worktree `/Users/andre/worktrees/xmris/amares`), **11 commits off `main`**,
**227 arch tests green** (`uv run pytest tests/test_core.py -n0 --no-cov`). Local only — not pushed.

- Approved plan: [`docs/plans/2026-07_amares_hardening.md`](2026-07_amares_hardening.md), mirrored
  verbatim from `~/.claude/plans/a-major-struggle-has-calm-seahorse.md`. Deviations are in §b — the
  plan is the historical intent (e.g. it says "go official pyAMARES"; we shipped `pyamares-xmris`).
- Fitting backend: **`pyamares-xmris`** (PyPI 0.3.33, import name still `pyAMARES`).
  `pip install "xmris[fitting]"` installs on Apple Silicon from PyPI.
- Only workstream **F** remains. Do **not** run full `uv run test` yet — a legacy viz notebook fails (see §e).

## (a) Status per workstream + commit SHAs

| WS | Scope (issue) | Status | Commits |
|---|---|---|---|
| **A** | optional `[fitting]` extra + import guards + fork exit (#70) | DONE | `75c7f87` (extra, PEP-562 guards, `scipy`→core), `e60c15e` (pyamares-xmris swap, drop git source) |
| **B** | scale-trap normalize/rescale, NaN sentinel, `initialize_with_lm=False` (#80) | DONE | `8eb99a2` (core), `32bb61e` (tests) |
| **C** | in-memory prior-knowledge builder (#82) | DONE | `3d20e6c` |
| **D** | API/UX: CRLB/sd, verbose, `g_global`, carrier, deadtime (#81) | DONE | `a5bfc5f` (CRLB Shape B), `329e7e6` (verbose BUG-010), `c247a4f` (`g_global`), `9dbd7f7` (carrier BUG-004); deadtime BUG-007 already in `8eb99a2` |
| **E** | output vocab `raw_data`/`fit_data`/`Metabolite` → `VARS`/`DIMS` (#69/#71/#88) | DONE | `8eb99a2` |
| **F** | ≥70% cov, un-skip parallel path, `testonly_` robustness notebook, #34 docstring de-dup (#67/#19/#34) | **NOT STARTED** | — |
| **G** | domain pivot: `fit_amares` domain-preserving | DONE | `8eb99a2` (code), `e38063c` (design note p1), `08bbbce` (contract reversal + design note p2) |

## (b) Deviations from the plan + why

1. **Fork exit is `pyamares-xmris` on PyPI, NOT "official pyAMARES".** Official pyAMARES 0.3.28
   hard-requires `hlsvdpro` (no arm64 wheel, no sdist) → uninstallable on Apple Silicon; upstream
   HawkMRS unresponsive to the one-line marker PR. Decision (with user): republish the fork as a
   faithful BSD package (zero kernel changes: `hlsvdpro` platform marker + numpy/pandas caps). `e60c15e`.
2. **numpy/pandas caps → `[fitting]` extra, not core deps** (plan said core). They are purely
   pyAMARES-driven; scoping to the extra keeps `pip install xmris` numpy-2-friendly. `75c7f87`.
3. **`scipy` added as a core dep** (not in plan). Latent under-declaration — core baseline/phasing/
   bruker import scipy but it only resolved transitively via pyamares. `75c7f87`.
4. **verbose: per-call re-apply, not a joblib worker initializer** (plan said initializer). joblib
   has no clean initializer API; `_set_verbosity` + `_muted_warnings` run inside `_fit_dataset_safe`
   (executes in every worker). Also had to set pyAMARES `DEFAULT_LOG_LEVEL` to silence *lazily-created*
   loggers (the `pm_index` leak) — `set_log_level` only sweeps existing ones. `329e7e6`.
5. **carrier via `ppm_offset`, not pyAMARES `carrier=`** (plan said forward `carrier`). Forwarding
   `carrier` is a no-op: pyAMARES's `carrier` shifts only the *template* FID, which `fit_amares`
   overwrites per spectrum. Instead shift the shared prior knowledge with `ppm_offset=-carrier`, then
   add the carrier back to the reported `chem_shift`. `9dbd7f7`.
6. **PK builder returns CSV *text* and is eager (pyAMARES-free).** `build_prior_knowledge(dict) -> str`;
   `fit_amares(prior_knowledge=<dict|DataFrame|path>)` routes it. Builder needs no pyAMARES, so it is
   an eager export (usable without the `fitting` extra). `3d20e6c`.

## (c) Six open questions — resolutions

1. **In-memory PK vs pyAMARES file API (C):** pyAMARES `generateparameter` is **file-path-only**
   (`fname.endswith(".csv"/"xlsx")`, else `NotImplementedError`; `initialize_FID` takes no pre-built
   `lmfit.Parameters`). → builder writes a `tempfile` CSV per fit call (`_resolve_pk_file`), removed on
   exit. `3d20e6c`.
2. **CI extras for fitting tests (A×F):** CI already `uv sync --all-extras --dev` → fitting tests run.
   The no-extra guard is exercised by `TestFittingPackaging` (subprocess blocks pyAMARES via `meta_path`,
   no `importorskip`), so **no dedicated no-pyamares CI job was added**. Add one in F if a true
   import-without-extra job is wanted. `75c7f87`.
3. **CRLB/sd output shape (D):** DECIDED **Shape B** (with user): parameter *values* stay named vars
   (`ds.amplitude`); `crlb` + `sd` gain a `parameter` dim `[amplitude, chem_shift, linewidth, phase]`.
   Not explode-into-`VARS`. `g` excluded (no value var, usually fixed). `a5bfc5f`.
4. **`initialize_with_lm=False` default flip (B):** DONE in the core rewrite (was `True`).
   Behavior-changing — intended (`True` can diverge on real data). `8eb99a2`.
5. **Complex / `component`-split input (Q#5):** **NOT ADDRESSED.** `fit_amares` requires a complex FID;
   real/imag-split (`DIMS.component`) input is not handled or auto-`to_complex`'d. Still open.
6. **Docs (Q#6):** `docs/explanation/fitting.md` written (p1 `e38063c`, p2 `08bbbce`) and reconciled
   again for the fork exit (`e60c15e`). The `testonly_` robustness notebook, ≥70% coverage, and
   un-skipping the parallel notebook path are **F, not done**.

## (d) New vocabulary minted (all flagged to user)

- `ATTRS.amares_amplitude_scale` (a.u.) — the single global normalization factor. `8eb99a2`.
- `DIMS.parameter` — fitted-parameter axis for `crlb`/`sd`. `a5bfc5f`.
- `VARS.sd` — per-parameter standard deviation (unit follows the parameter). `a5bfc5f`.

No new `COORDS`. carrier reused `ATTRS.carrier_ppm` + `ATTRS.reference_frequency`. `g` (lineshape) is a
local string, deliberately **not** a `VARS` term.

## (e) Half-decided / fragile — a fresh session must know

- **F is all that remains:** ≥70% cov on `fitting/amares.py`; un-skip the parallel path in
  `docs/notebooks/fitting/pyamares.md` (currently only `num_workers=1` executes; the loky path is
  `skip-execution`); add `docs/notebooks/fitting/testonly_amares_robustness.md`; #34 de-dup accessor vs
  free-fn docstring.
- **Legacy viz vs the new output — RESOLVED (`40f84a4`):**
  `plot_trajectory.py` / `plot_qc_grid.py` and `03_plotting_1dfid.md`'s assert now read the new
  contract (`DIMS`/`VARS`, `crlb` selected at `parameter="amplitude"`), and the notebook is green.
  Only `plot_qc_grid.py`'s pre-existing spectrum-in assumption (unconditional `to_spectrum`, broken on
  `main` too) stays deferred — tracked as
  [#106](https://github.com/andrewendlinger/xmris/issues/106). *(Original note: it still used
  `Metabolite` and failed full `uv run test`; the `prior_knowledge_file`→`prior_knowledge` rename had
  been propagated mechanically only.)*
- **Verify fitting notebooks individually, not via full `uv run test`:**
  `uv run test-gen && uv run pytest "tests/autogen_notebooks/fitting/pyamares.ipynb" -n0 --no-cov`.
- **mypy: 55 pre-existing errors** (accessor xarray-typing + `amares.py:540` `restore_state`
  `_RestoreState`-vs-tuple). Not in CI; **zero introduced by this branch**.
- **Behavior-loaded defaults:** `initialize_with_lm=False`, `g_global=0.0` (fixed Lorentzian),
  `carrier=None` → auto-read `attrs['carrier_ppm']` (default 0.0 = carrier-relative, back-compatible).
- **Fork-exit provenance:** `pyamares-xmris` source = fork `andrewendlinger/pyAMARES`, branch
  `pyamares-xmris`, tag `v0.3.33`. Publish/maintain brief:
  `/Users/andre/PhD/dev/python/xmris-compatible/CLAUDE.md`. Upstream sync = rebase the packaging
  commits, bump, republish.
- **Not reviewed, not pushed.** Recommend `/code-review ultra` on the branch **before** F (all logic is
  in place; F is tests/docs that should absorb review feedback).
