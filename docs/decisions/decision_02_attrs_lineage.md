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

(attrs-nb)=
# The attrs decision — a design notebook

:::{note} A frozen exploration record
This notebook explored the option space for roadmap decision **02 — the lineage record**
(tracker [#64](https://github.com/andrewendlinger/xmris/issues/64)) and froze when the
decision landed on 2026-08-02: **option B, the physics/record split**. It is the record of
*why*, not the spec of *what* — the aimed-solution notebook will be the spec, and the
[decision board](#roadmap-decisions) carries the summary. The "Option B" cells run a
**prototype defined inside this notebook**; nothing here is implemented in the library yet.
:::

The roadmap promises that the object handed back *can answer for every step that produced it*.
This notebook holds every candidate attrs strategy against that sentence — first
the problem, demonstrated live against today's `main`; then each option as code, the user
experience and the contributor's side of the same coin.

(attrs-nb-problem)=
## The problem, live

Today's law is Commandment 3: preserve inbound attrs, then append the parameter you applied —
`apodization_lb = 5.0`, not `apodized = True`. A short chain looks well-recorded:

```{code-cell} ipython3
import numpy as np
import xarray as xr

import xmris

fid = xmris.simulate_fid(
    amplitudes=[1.0, 0.6],
    chemical_shifts=[0.0, 5.2],
    reference_frequency=120.66,  # MHz
    n_points=1024,
)

spectrum = (
    fid.xmr.zero_fill(target_points=2048)
    .xmr.apodize_exp(lb=5.0)
    .xmr.to_spectrum()
    .xmr.phase(p0=20.0)
)
spectrum.attrs
```

Fifteen keys, physics and lineage shoulder to shoulder — already showing #64's second worry,
sprawl. But the record *reads* complete: the line broadening, the zero-fill, the phase are all
there. It stays complete exactly as long as no step happens twice.

```{code-cell} ipython3
twice = fid.xmr.apodize_exp(lb=5.0).xmr.apodize_exp(lb=5.0)

print("the record claims lb =", twice.attrs["apodization_lb"])
print("the data says lb = 10:", bool(np.allclose(twice, fid.xmr.apodize_exp(lb=10.0))))
```

The second application silently overwrote the first — no trace remains that this FID was
apodized twice. With phasing, the record graduates from forgetting to lying:

```{code-cell} ipython3
spec = fid.xmr.to_spectrum()
rephased = spec.xmr.phase(p0=20.0).xmr.phase(p0=-5.0)

print("the record claims p0 =", rephased.attrs["phase_p0"])
print("the data carries 15°:", bool(np.allclose(rephased, spec.xmr.phase(p0=15.0))))
```

Phases add: the object carries 15° of correction and its own record claims −5°. Anyone
reproducing this result from the attrs reproduces the wrong spectrum.

The library already feels this strain. `phase` hand-rolls a one-off cross-step consistency
check — it remembers which coordinate space the last phase was applied in, and warns when the
next one happens somewhere else:

```{code-cell} ipython3
spec_ppm = spec.xmr.phase(p0=20.0).xmr.to_ppm()
_ = spec_ppm.xmr.phase(p0=5.0, dim="chemical_shift")  # ← warns
```

That warning is proto-history: one function keeping a private memory of the step before it,
because the object has no shared one. And finally, one step *outside* xmris loses everything —
xarray drops attrs on most operations by default
([#21](https://github.com/andrewendlinger/xmris/issues/21)):

```{code-cell} ipython3
(spectrum * 2).attrs
```

So the question splits in two: **what shape should the record have** (Options A–C below), and
**what preservation guarantee can the library honestly make** (its own
[section](#attrs-nb-guarantee) — the answer is the same under every option).

(attrs-nb-reframe)=
## Two kinds of attrs

Before comparing options, one observation that every option benefits from. Today's `ATTRS`
vocabulary mixes two populations that behave nothing alike:

| Class | Keys today | Who reads them |
|---|---|---|
| **Physics / calibration** — describe the data | `reference_frequency`, `carrier_ppm`, `b0_field`, `group_delay`, `spectral_width`, `dead_time`, `units` | Code: `to_ppm` and `fit_amares` gate and read them (`@requires_attrs`); [#22](https://github.com/andrewendlinger/xmris/issues/22)'s type/range validation targets them |
| **Lineage / audit** — describe what was done | `phase_p0/p1/pivot/pivot_coord`, `apodization_lb/gb`, `zero_fill_target/position`, `baseline_*`, `group_delay_removed`, `amares_amplitude_scale`, `sim_*` | **Nobody.** A grep of `src/` finds one consumer: `phase`'s own advisory warning above. The record is write-only |

Physics attrs are healthy: individually addressable, typed, gated at the door. Everything wrong
in the previous section is confined to the lineage class. netCDF's CF conventions draw exactly
this line — physical metadata that must stay valid, separate from an append-only `history`
attribute of processing steps. Every option below leaves physics attrs untouched; the decision
is only about the record.

(attrs-nb-option-a)=
## Option A — flat keys stay law

Keep Commandment 3 as written. The overwrite becomes documented semantics — *last application
wins* — and [#23](https://github.com/andrewendlinger/xmris/issues/23) (provenance tracking)
closes as won't-do.

**The user experience** is the one already shown live above: readable single keys
(`apodization_lb: 5.0`) that answer "what was applied last?" and nothing else — not how many
times, not in what order, not what the effective total is. The re-phasing cell stays a lie.

**The contributor side** is today's hand-rolled pattern, repeated in every function
(`src/xmris/processing/fid.py`):

```python
da_apodized = (da * weight).transpose(*da.dims).assign_attrs(da.attrs)

# Record lineage
da_apodized.attrs[ATTRS.apodization_lb] = lb

return da_apodized
```

Twenty functions, twenty appends, and any new function's author must remember both halves —
the `assign_attrs` copy *and* the append. Cheapest option by far: zero migration, no new
machinery, `TestAttrsPreservation` untouched. Its price is that the hero sentence is false and
stays false — the object answers for the *last* application of each step, unordered.

(attrs-nb-option-b)=
## Option B — one structured history

Lineage moves into a single attr: `xmr_history`, one JSON string holding an append-only event
log. Flat lineage keys are deleted from the vocabulary. One central decorator does all the
bookkeeping — preservation *and* the event — so function bodies stop handling attrs entirely.

To make this option feelable rather than imagined, the cell below prototypes that decorator and
grafts it onto today's functions. These ~40 lines are also the honest size estimate of the
central machinery:

```{code-cell} ipython3
import functools
import inspect
import json
from importlib.metadata import version

HISTORY_KEY = "xmr_history"  # would live in the vocabulary as ATTRS.history
EMPTY = json.dumps({"schema": 1, "events": []})

# Keys today's functions hand-append. Under option B the appends are deleted from
# the source; the prototype strips them from wrapped outputs to emulate that.
FLAT_LINEAGE = {
    "phase_p0", "phase_p1", "phase_pivot", "phase_pivot_coord",
    "apodization_lb", "apodization_gb", "zero_fill_target", "zero_fill_position",
    "baseline_method", "baseline_lam", "baseline_p", "baseline_iter",
    "group_delay_removed", "amares_amplitude_scale",
    "sim_amplitudes", "sim_dampings", "sim_frequencies_hz", "sim_chemical_shifts_ppm",
}


def records_history(func):
    """Prototype of the one central decorator: preserve attrs, append the event."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        data_args = [
            v for v in bound.arguments.values() if isinstance(v, (xr.DataArray, xr.Dataset))
        ]
        params = {
            k: v
            for k, v in bound.arguments.items()
            if not isinstance(v, (xr.DataArray, xr.Dataset))
        }

        result = func(*args, **kwargs)

        inbound = dict(data_args[0].attrs) if data_args else {}
        added = {k: v for k, v in result.attrs.items() if k not in FLAT_LINEAGE}
        result.attrs = {**inbound, **added}  # preservation by construction

        envelope = json.loads(result.attrs.get(HISTORY_KEY, EMPTY))
        envelope["events"].append({"op": func.__name__, "params": params, "v": version("xmris")})
        result.attrs[HISTORY_KEY] = json.dumps(envelope, default=str)
        return result

    return wrapper
```

```{code-cell} ipython3
# Graft the prototype onto the live library, notebook-locally: the .xmr methods are
# thin delegators to these module-level names (Commandment 9), so rebinding them
# gives the real chained UX without touching src/.
import xmris.core.accessor as _accessor

for _name in ("zero_fill", "apodize_exp", "to_spectrum", "phase"):
    setattr(_accessor, _name, records_history(getattr(_accessor, _name)))

simulate_fid = records_history(xmris.simulate_fid)  # history starts at birth
```

(attrs-nb-option-b-ux)=
### What the user sees

The same chain as the opening — with the double apodization and the re-phasing left in
deliberately:

```{code-cell} ipython3
fid_b = simulate_fid(
    amplitudes=[1.0, 0.6],
    chemical_shifts=[0.0, 5.2],
    reference_frequency=120.66,
    n_points=1024,
)

spectrum_b = (
    fid_b.xmr.zero_fill(target_points=2048)
    .xmr.apodize_exp(lb=5.0)
    .xmr.apodize_exp(lb=5.0)  # the same step twice — now on the record
    .xmr.to_spectrum()
    .xmr.phase(p0=20.0)
    .xmr.phase(p0=-5.0)  # re-phased — both calls visible, order kept
)
spectrum_b.attrs
```

The attrs dict collapses to its physics plus one record key. That `xmr_history` string is the
honest cost of this option: at a glance it is a JSON blob. The reading surface would be a
`da.xmr.history()` method — prototyped here as a function:

```{code-cell} ipython3
import pandas as pd


def history(da: xr.DataArray) -> pd.DataFrame:
    """What `da.xmr.history()` would return."""
    events = json.loads(da.attrs[HISTORY_KEY])["events"]
    return pd.DataFrame(
        {
            "op": [e["op"] for e in events],
            "params": [
                ", ".join(f"{k}={v!r}" for k, v in e["params"].items() if v is not None)
                for e in events
            ],
            "xmris": [e["v"] for e in events],
        }
    )


history(spectrum_b)
```

Every step, in order, with its parameters — including the repeated ones the flat record
swallowed. The hero sentence, answered by a method call.

(attrs-nb-option-b-file)=
### What survives a file

The record must travel with the data, so the shape is chosen for serialization: netCDF attrs
cannot hold nested dicts, and a length-1 list-of-strings attr silently collapses to a scalar on
an xarray round-trip — one JSON string is the engine-proof form.

```{code-cell} ipython3
import tempfile
from pathlib import Path

# netCDF holds no complex values natively (that is the component dim's business,
# out of scope here) — the attrs are what we are round-tripping.
magnitude = spectrum_b.copy(data=np.abs(spectrum_b.values))

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "spectrum.nc"
    magnitude.to_netcdf(path)
    loaded = xr.load_dataarray(path)

assert loaded.attrs[HISTORY_KEY] == spectrum_b.attrs[HISTORY_KEY]
history(loaded)
```

A colleague opening this file next year gets the full processing story out of the `.nc` itself.

One deliberate absence: **no timestamps**. Events record `op`, `params` and the xmris version —
nothing wall-clock. Identical inputs must produce identical objects, or every
`assert_identical` in the executable docs (this project's differentiator) starts failing:

```{code-cell} ipython3
rerun = (
    fid_b.xmr.zero_fill(target_points=2048)
    .xmr.apodize_exp(lb=5.0)
    .xmr.apodize_exp(lb=5.0)
    .xmr.to_spectrum()
    .xmr.phase(p0=20.0)
    .xmr.phase(p0=-5.0)
)
xr.testing.assert_identical(spectrum_b, rerun)
print("bit-identical rerun — the record is deterministic")
```

(attrs-nb-option-b-contrib)=
### What the contributor writes

Function bodies lose all attrs bookkeeping. Today's `apodize_exp` closes with three lines of
hand-rolled lineage; under option B it becomes:

```python
@computes_in(TIME_DIMS)
@records_history  # ← preservation + the event, centrally
def apodize_exp(da: xr.DataArray, dim: str = DIMS.time, lb: float = 1.0) -> xr.DataArray:
    _check_dims(da, dim, "apodize_exp")
    weight = np.exp(-np.pi * lb * da.coords[dim])
    return (da * weight).transpose(*da.dims)  # no attrs handling at all
```

The wider footprint of the change:

- **Vocabulary**: the 14 flat lineage terms in `ATTRS` (plus `simulate_fid`'s literal `sim_*`
  keys) are deleted; one term is added (`history = "xmr_history"` — namespaced, so it cannot
  collide with user or CF attrs the way `baseline_method` can today).
- **Commandment 3 rewrite** (draft): *Preserve inbound coordinates and attributes. Physics
  attrs stay flat and typed. What the function did is one appended event in the history —
  written by the decorator, never by hand. Banned: state flags, hand-appended lineage keys, and
  any math that branches on the history.* The last clause makes the record write-only for
  library logic; advisory reads stay legal — `phase`'s coordinate-space warning becomes a peek
  at the history's tail instead of a private key.
- **Tests**: `TestAttrsPreservation` survives unchanged (preservation is now structural, the
  test becomes a regression guard); one new test pins the round trip and event order.
- **`fit_amares`** appends a single `fit_amares` event to the Dataset — `amares_amplitude_scale`
  becomes one of its params instead of a vocabulary term (the fit-Dataset chapter of the
  data-model schema, [#28](https://github.com/andrewendlinger/xmris/issues/28)).

:::{dropdown} Fine print — where the prototype cheats
- **Applied vs. passed.** The decorator records arguments *as bound*. `phase(pivot=None)`
  resolves the pivot internally, and `autophase` *finds* p0/p1 — today's flat keys record the
  resolved values, which is genuinely better. The real decorator needs one opt-in channel for a
  function to override its recorded params with what it actually applied; most functions need
  nothing. This is the main piece of design work left inside option B.
- **Functions that add physics attrs** (vendor loaders, `simulate_fid`) keep doing so — the
  decorator preserves inbound attrs and function-added keys alike; only the event goes through
  the append. The prototype emulates "deleted appends" by stripping `FLAT_LINEAGE`.
- Params must be JSON-representable; the prototype falls back to `str`. The real rule: record
  scalars, strings and small lists; data-shaped arguments are omitted.
- The monkeypatch wraps four functions for the demo; the real change decorates all of them.
:::

(attrs-nb-option-c)=
## Option C — hybrid: flat "latest" keys *plus* the history

Keep the history from option B as the audit record, and additionally keep flat keys as a
quick-glance surface for the *latest* application. The attrs would render like:

```python
>>> spectrum.attrs
{'reference_frequency': 120.66,
 'carrier_ppm': 0.0,
 'apodization_lb': 5.0,        # ← latest application only
 'phase_p0': -5.0,             # ← still claims −5° while the data carries 15°
 'xmr_history': '{"schema": 1, "events": [...]}'}
```

The contributor delta over B is small — the decorator mirrors each event's params into flat
keys. The problems are what the mirror *means*:

- "Latest" is not "effective state": after two `lb=5` apodizations the mirror reads `5.0`; after
  the re-phasing it reads `-5.0`. The quick-glance surface keeps the lie that motivated the
  change, now beside the record that contradicts it.
- The mirror duplicates the history's last event — redundant information that must be kept
  consistent, and a second record format frozen at 1.0.
- Its only justification would be readers — and today there are none (the
  [grep above](#attrs-nb-reframe)). A consumer that appears later can be served by *adding* the
  mirror back, a non-breaking change; removing it later is the breaking one.

:::{note} Option D — provenance outside attrs
A sidecar object (a ledger next to the DataArray) fails the first constraint: the record must
travel with the data. Any plain xarray op or netCDF round trip orphans it. Not considered
further.
:::

(attrs-nb-guarantee)=
## The preservation guarantee — the same answer under every option

Whatever the record's shape, what may a user actually rely on? Inside xmris the answer can be
structural: the decorator restores inbound attrs onto every result, so even attrs-dropping
operations *inside* a function body cannot lose them (this is
[#21](https://github.com/andrewendlinger/xmris/issues/21)'s "systematic mechanism"). Outside
xmris, the library cannot patch xarray, and pretending otherwise would be a false promise. The
escape hatch already exists upstream:

```{code-cell} ipython3
with xr.set_options(keep_attrs=True):
    doubled = spectrum_b * 2

doubled.attrs == spectrum_b.attrs
```

:::{important} The honest guarantee, stated once
Every public xmris function preserves inbound attrs and coordinates and appends its record —
structurally, via the central decorator; `TestAttrsPreservation` stays as the regression guard.
Outside xmris functions, xarray's rules apply: plain arithmetic, `concat`, `groupby` drop attrs
unless you opt in with `xr.set_options(keep_attrs=True)`. xmris documents that boundary and
never sets global options itself; histories of *combined* objects follow xarray's
`combine_attrs`, and xmris invents no merge semantics. The history records xmris steps only —
math done outside the library is invisible to it under every option.
:::

(attrs-nb-verdict)=
## Side by side

| Can the object answer… | A — flat | B — history | C — hybrid |
|---|---|---|---|
| what was applied last? | yes | yes (last event) | yes |
| how many times, in what order? | no | yes | yes (history half) |
| for the effective total (re-phasing)? | no — record lies | yes (sum of events) | history yes, mirror lies |
| after a netCDF round trip? | yes | yes | yes |
| at a glance, in the attrs repr? | yes — readable keys | JSON blob; `history()` is the surface | readable keys + blob |
| Frozen surfaces at 1.0 | 14 flat keys | 1 envelope format | both |
| Per-function cost | hand-rolled ×20 | one decorator | one decorator + mirror invariant |
| Readers served today | 0 | 0 (writes a new surface) | 0 |

:::{tip} Recommendation: option B
The flat record fails the hero sentence in the first repeated step, and its keys have no
readers to protect. One JSON-string history — envelope-versioned, timestamp-free, written by
the same decorator that makes preservation structural — answers every row above except the
glance, and the glance is a display problem (`history()`), not a storage problem. Revisit if a
real programmatic consumer of "latest params" appears (re-add the mirror, additively), or if
CF/NIfTI-MRS interop later wants a timestamped free-text `history` twin.

Deciding this fixes half of the data-model schema (attrs = physics + record,
[#28](https://github.com/andrewendlinger/xmris/issues/28)) and shapes its fit-Dataset chapter
(one `fit_amares` event). Decided 2026-08-02: the resolution is recorded on the
[decision board](#roadmap-decisions) and in
[#64](https://github.com/andrewendlinger/xmris/issues/64), which tracks the implementation.
This page stays frozen as the record of *why*.
:::
