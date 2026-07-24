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

(testonly-amares-robustness)=
# AMARES robustness — end-to-end integration checks

This notebook is the contributor-facing, end-to-end proof that `fit_amares` and
`build_prior_knowledge` hold up under the conditions unit tests abstract away: real
Bruker signal scales, empty voxels, and the trap-prone prior-knowledge inputs that
pyAMARES would otherwise misread silently. It is a `testonly_` page — executed by the
test suite (nbmake) but never rendered on the docs site — so it speaks in config
singletons and asserts rather than reader-facing prose. Its unit-level companions are
`TestFittingDomain` and `TestPriorKnowledgeBuilder` in `tests/test_core.py`; here the
same guarantees are exercised through the whole public pipeline at once, with
`simulate_fid` standing in for a scanner. Every fit uses `num_workers=1` (loky under
nbmake's `-n auto` xdist would nest parallel pools).

```{code-cell} ipython3
import contextlib
import io
import logging

import numpy as np
import pytest
import xarray as xr

import xmris  # registers the .xmr accessor
from xmris.core.config import ATTRS, DIMS, VARS
from xmris.fitting import build_prior_knowledge
from xmris.fitting.simulation import simulate_fid
```

(testonly-amares-robustness-scale)=
## 1. A Bruker-scale fit converges — and does not echo the prior

A real 31P FID off a scanner peaks around `1e7`. On such data pyAMARES's
magnitude-derived optimizer tolerance balloons and the solver "converges" on step one,
handing back the prior guess unchanged. `fit_amares` defeats that by normalizing to a
single global factor, fitting where the tolerance behaves, and rescaling the amplitudes
back (issue #80). We prove it end to end with a signal whose true amplitude ratio (2:1)
differs from a deliberately **equal** prior guess: if the trap bit, the fit would report
the guess's 1:1 ratio.

```{code-cell} ipython3
# Bruker-scale 31P FID: PCr and ATP at ~1e7 a.u., true amplitude ratio 2:1.
# dampings are pi * linewidth(Hz) so the fitted linewidths land near the prior.
fid = simulate_fid(
    amplitudes=[1.0e7, 5.0e6],
    chemical_shifts=[0.0, -7.5],
    reference_frequency=49.0,
    spectral_width=8000.0,
    n_points=512,
    dampings=[np.pi * 15.0, np.pi * 20.0],
    target_snr=300.0,
    seed=0,
)

# A deliberately wrong, *equal* amplitude guess for both peaks: echoing it → 1:1.
pk = {
    "PCr": {"amplitude": 3.0, "chem_shift": 0.0, "linewidth": 15.0},
    "ATP": {"amplitude": 3.0, "chem_shift": -7.5, "linewidth": 20.0},
}
ds = fid.xmr.fit_amares(prior_knowledge=pk, num_workers=1)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: the normalize/rescale fix (#80) defeats the magnitude scale trap.
_amps = ds[VARS.amplitude].values
# 1. Amplitudes return in input units at Bruker scale — not the ~3 prior guess.
np.testing.assert_allclose(_amps, [1.0e7, 5.0e6], rtol=0.1, err_msg="scale not recovered")
# 2. The true 2:1 ratio is recovered; echoing the equal prior would give 1:1.
np.testing.assert_allclose(_amps[0] / _amps[1], 2.0, rtol=0.1, err_msg="prior was echoed")
# 3. The global normalization factor is stamped as lineage (~1e7 scale).
assert ds.attrs[ATTRS.amares_amplitude_scale] > 1e6
# 4. The per-parameter amplitude uncertainty came back finite.
_crlb_amp = ds[VARS.crlb].sel({DIMS.parameter: VARS.amplitude}).values
assert np.all(np.isfinite(_crlb_amp))
```

(testonly-amares-robustness-nan)=
## 2. An empty voxel stays `NaN` — and `fit_status` says why

Fit an N-dimensional dataset and some voxels have no signal. Writing them as `0` makes
a give-up indistinguishable from a genuine near-zero measurement — a downstream mean
folds them in. `fit_amares` writes `NaN` instead: the honest absence of a value. But one
`NaN` cannot say *which* absence it is — an empty (no-signal) voxel and a failed fit read
alike. The `fit_status` flag (0=fitted, 1=no_signal, 2=failed) records the distinction the
values cannot.

```{code-cell} ipython3
# Stack the real FID with an all-zero voxel.
stack = xr.concat([fid, xr.zeros_like(fid)], dim="voxel").assign_attrs(fid.attrs)
ds_stack = stack.xmr.fit_amares(prior_knowledge=pk, num_workers=1)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: the empty voxel's values stay NaN (never a spurious zero); the real one
# still fits; and fit_status separates the two states the shared NaN cannot.
assert np.all(np.isnan(ds_stack[VARS.amplitude].isel(voxel=1).values))
assert np.all(np.isfinite(ds_stack[VARS.amplitude].isel(voxel=0).values))
assert int(ds_stack[VARS.fit_status].isel(voxel=1)) == 1  # empty voxel -> no_signal
assert int(ds_stack[VARS.fit_status].isel(voxel=0)) == 0  # real voxel  -> fitted
assert ds_stack[VARS.fit_status].attrs["flag_meanings"] == "fitted no_signal failed"
```

(testonly-amares-robustness-traps)=
## 3. `build_prior_knowledge` refuses each pyAMARES footgun at the door

pyAMARES reads prior knowledge from a positional CSV whose row order and bound syntax
are easy to get subtly — and silently — wrong. `build_prior_knowledge` refuses each trap
loudly instead of emitting a file that fits to garbage.

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: every refusal is a clear ValueError, not a silent mis-fit.
_ok = {"amplitude": 10.0, "chem_shift": 0.0, "linewidth": 15.0}

# (a) A trailing digit is a J-coupling multiplet component in pyAMARES ('ATP2' folds
#     into 'ATP') — rejected as a peak name.
with pytest.raises(ValueError, match="letters only"):
    build_prior_knowledge({"ATP2": _ok})

# (b) An empty spec has nothing to fit.
with pytest.raises(ValueError, match="empty"):
    build_prior_knowledge({})

# (c) A tie anchor that is not one of the peaks is a typo, not a silent no-op.
with pytest.raises(ValueError, match="not one of the peaks"):
    build_prior_knowledge({"PCr": _ok}, tie_phase_to="XYZ")

# (d) Phase is ALWAYS bounded (-180, 180): a blank bound is -inf in pyAMARES and NaNs
#     the fit, so the emitted CSV carries the bound explicitly.
_csv = build_prior_knowledge({"PCr": _ok})
assert "(-180.0, 180.0)" in _csv

# (e) The tie anchor is written *first*, so lmfit sees it defined before the peaks that
#     reference it (it resolves columns left to right).
_tied = build_prior_knowledge(
    {"ATP": {**_ok, "chem_shift": -7.5}, "PCr": _ok}, tie_phase_to="PCr"
)
assert _tied.splitlines()[0].split(",")[1] == "PCr"
```

(testonly-amares-robustness-quiet)=
## 4. `verbose=False` is silent on every channel

A batch fit must not flood the console. `verbose=False` sets the pyAMARES and xmris log
levels to ERROR — and does so inside each worker, not only the main process — so nothing
reaches stdout, stderr, or the `xmris.fitting` logger on a clean run.

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: capture all three channels around a quiet fit and assert silence.
_out = io.StringIO()
_log = io.StringIO()
_handler = logging.StreamHandler(_log)
_logger = logging.getLogger("xmris.fitting")
_logger.addHandler(_handler)
try:
    with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(_out):
        _ = fid.xmr.fit_amares(prior_knowledge=pk, num_workers=1, verbose=False)
finally:
    _logger.removeHandler(_handler)
_captured = _out.getvalue() + _log.getvalue()
assert _captured.strip() == "", f"expected silence with verbose=False, captured: {_captured!r}"
```
