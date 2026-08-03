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

(constants-nb)=
# The constants decision — a design notebook

:::{note} A frozen exploration record
This notebook explored the option space for roadmap decision **02b — the physical constants**
(tracker [#21](https://github.com/andrewendlinger/xmris/issues/21)) and froze when the
decision landed on 2026-08-03: **the container coordinates, `xmr_acquisition` +
`xmr_history`**. It is the record of *why*, not the spec of *what* — the aimed-solution
notebook will be the spec, and the [decision board](#roadmap-decisions) carries the summary.
Every option below runs a **prototype defined inside this notebook**; nothing here is
implemented in the library yet. Sibling to [the attrs exploration](#attrs-nb), whose decision
is settled law underneath this page.
:::

:::{note} What changed across review rounds
Pass one compared boilerplate, an import-time global, and one-coordinate-per-constant storage;
review rejected all three. Pass two elevated **repr readability to a first-class constraint**,
prototyped the `keep_attrs` ergonomics (scoped sugar, notebook auto-detection), and added two
structural options — the container coordinate won. This pass folds in the convergence round:
the **history rides the container block too** (amending dossier 02's storage home, not its
format), both containers are **pinned to the bottom** of the coordinate list with the
enforcement measured, and the **names** are fixed (`xmr_acquisition`, `xmr_history`).
:::

The [attrs decision](#attrs-nb) split the metadata world in two: lineage became the
`xmr_history` record, and the **physical constants** — the numbers a measurement cannot be
interpreted without — stayed flat, typed, individually addressable. Two promises about those
constants are still broken or unmade: they do not reliably **travel** (one ordinary xarray
operation strips them, and the failure surfaces steps later), and they are not
**discoverable** (every constant carries curated prose and a unit in the vocabulary, yet
nothing on the object surfaces it). This notebook's thesis, reached the long way round: the
library should stop trying to patch xarray's metadata rules and instead stop keeping physics
*as* metadata — the options differ in **where in the object's structure** the constants live,
and what each home costs in the repr, at the gates, and in a file.

(constants-nb-problem)=
## The problem, live

`simulate_fid` stamps its output with the calibration the physics needs — the same keys a
vendor loader writes. Inside `.xmr` chains they are safe; a notebook is not an `.xmr` chain:

```{code-cell} ipython3
import numpy as np
import pandas as pd
import xarray as xr

import xmris

fid = xmris.simulate_fid(
    amplitudes=[1.0, 0.6],
    chemical_shifts=[0.0, 5.2],
    reference_frequency=120.66,  # MHz
    n_points=1024,
)
spectrum = fid.xmr.apodize_exp(lb=5.0).xmr.to_spectrum()

scaled = spectrum * 2  # any plain xarray operation
scaled.attrs
```

Everything is gone — including `reference_frequency` and `carrier_ppm`, without which a ppm
axis cannot exist. Nothing says so; the failure arrives later, in a different cell, possibly a
different day:

```{code-cell} ipython3
normalized = scaled / scaled.size  # ...a few innocent steps later...
try:
    normalized.xmr.to_ppm()
except ValueError as err:
    print(err)
```

The error is honest but names the *symptom*: the user is told to re-enter a value they already
provided, with no hint which upstream line dropped it —
[#21](https://github.com/andrewendlinger/xmris/issues/21)'s insidious half. And the second
promise breaks silently in the first cell's output: to a scientist arriving from any vendor,
`'reference_frequency': 120.66` is a bare float, while the curation written for exactly that
reader is reachable only by importing library internals:

```{code-cell} ipython3
from xmris.core.config import ATTRS  # a user never types this line

print("unit:", ATTRS.reference_frequency.unit)
print(ATTRS.reference_frequency.description)
```

(constants-nb-class)=
## What must travel — and what already does

Not every key on today's objects is a constant this dossier must carry:

| Key on today's objects | What it is | In this dossier's class? |
|---|---|---|
| `reference_frequency`, `carrier_ppm`, `group_delay`, `b0_field` | Non-derivable acquisition constants — gates and converters need them | **Yes** |
| `spectral_width`, `dead_time` | Derivable: the time coordinate's spacing and first sample — `fit_amares` already infers both from it | No — duplication to retire |
| `units` | Describes the data values (CF-style variable attr) | No — stays an attr |
| `sim_*`, `target_snr`, `apodization_lb`, … | Lineage — becomes `xmr_history` | No — dossier 02's law |

The second row is the thesis in miniature: `spectral_width` never had a travel problem,
because it is not an annotation *about* the data — it **is structure**, the spacing of a
coordinate, and xarray protects structure with everything it has. Four constants need a home
today; the MRSI horizon multiplies the family (echo/repetition timing, flip angle, voxel
geometry), so the mechanism must stay readable at, say, ten constants on a four-dimensional
object. That object is this page's measuring stick:

```{code-cell} ipython3
HORIZON = {  # the four real constants of today + a plausible MRSI-era family
    "reference_frequency": 120.66, "carrier_ppm": 4.7, "group_delay": 68.0, "b0_field": 7.0,
    "echo_time": 0.012, "repetition_time": 2.0, "flip_angle": 90.0,
    "voxel_dx": 2.5, "voxel_dy": 2.5, "voxel_dz": 8.0,
}

mrsi = xr.DataArray(
    np.zeros((8, 8, 256, 4), dtype=complex),
    dims=["kx", "ky", "time", "coil"],
    coords={
        "kx": np.arange(8), "ky": np.arange(8), "coil": np.arange(4),
        "time": ("time", np.arange(256) * 1e-4, {"units": "s"}),
    },
    name="mrsi",
)
```

(constants-nb-keepattrs)=
## The `keep_attrs` family — ergonomics for the flag

The constants could stay attrs if the flag that protects attrs stopped being ceremony. Two
ergonomic forms, prototyped. First, scoped sugar — `with xmris.keep():`

```{code-cell} ipython3
import contextlib


@contextlib.contextmanager
def keep():
    """What `xmris.keep()` would be: three lines of sugar over xarray's option."""
    with xr.set_options(keep_attrs=True):
        yield


with keep():
    doubled = spectrum * 2

doubled.attrs == spectrum.attrs
```

It works, and the name is friendlier — but count the ceremony. Scoped honestly, *every* cell
that touches the data needs the wrapper, which is more ritual than the line it replaced;
hoisted to one call at the top of the notebook, it *is* the anti-goal line wearing a nicer
name. Sugar relocates the ceremony; it cannot remove it.

Second form: the library detects a notebook and sets the flag there, leaving scripts alone.
Detection is real — `get_ipython()` exists exactly when running under Jupyter/IPython (it does
right now, in this page's kernel). But watch what the *same three lines* then do in the two
environments:

```{code-cell} ipython3
import subprocess
import sys

xr.set_options(keep_attrs=True)  # what auto-detecting xmris would have done at import — here

lines = (
    "import numpy as np, xarray as xr\n"
    "da = xr.DataArray(np.ones(4), dims='t', attrs={'reference_frequency': 120.66})\n"
    "print('as a script:  ', (da * 2).attrs)\n"
)
da = xr.DataArray(np.ones(4), dims="t", attrs={"reference_frequency": 120.66})
print("in the notebook:", (da * 2).attrs)
print(subprocess.run([sys.executable, "-c", lines], capture_output=True, text=True).stdout, end="")
```

The pipeline that worked all week in Jupyter loses its constants the day it is promoted to a
script — the failure mode moves to exactly the moment nobody is watching. And in the notebook,
the flag is a **session global**: every other library's objects in the same kernel now behave
differently because xmris was imported — the third tenet inverted, and a direct contradiction
of the [attrs Resolution's guarantee](#attrs-nb-guarantee) that xmris never sets global
options. Worse, the whole family shares one deeper flaw — under the flag, attrs of *combined*
objects are first-wins:

```{code-cell} ipython3
colleague = spectrum.copy(deep=True)
colleague.attrs["reference_frequency"] = 500.13  # same molecule, different spectrometer

merged = spectrum + colleague  # flag still on, from the cell above
print(merged.attrs["reference_frequency"], "— first operand wins, silently")

xr.set_options(keep_attrs="default")  # undo: this notebook must not own the session either
```

A later `to_ppm` draws an axis that is wrong for half the signal, without a murmur. Whatever
the ergonomics, `keep_attrs` converts silent loss into silent lies. The family is not the
answer at any level of sugar.

(constants-nb-structure)=
## The structural turn

xarray drops *annotations*; it protects *structure* — coordinates survive because science
depends on them. So measure exactly what a coordinate-borne constant can rely on. The probe: a
scalar (dimensionless) coordinate whose **value** is the constant and whose own attrs carry
its unit — pushed through the operations a real analysis performs:

```{code-cell} ipython3
probe = xr.DataArray(
    np.linspace(-1, 1, 8),
    dims=["frequency"],
    coords={
        "frequency": ("frequency", np.arange(8.0), {"units": "Hz"}),
        "reference_frequency": xr.Variable((), 120.66, attrs={"units": "MHz"}),
    },
)

ops = {
    "da * 2": lambda d: d * 2,
    "np.abs(da)": lambda d: np.abs(d),
    "da - da.mean()": lambda d: d - d.mean(),
    "da.mean('frequency')": lambda d: d.mean("frequency"),
    "da.isel(frequency=slice(2, 6))": lambda d: d.isel(frequency=slice(2, 6)),
    "da.where(da > 0)": lambda d: d.where(d > 0),
    "xr.where(da > 0, da, 0)": lambda d: xr.where(d > 0, d, 0),
    "da.groupby('band').mean()": lambda d: d.assign_coords(
        band=("frequency", [0, 0, 0, 0, 1, 1, 1, 1])
    ).groupby("band").mean(),
    "xr.concat([da, da], 'rep')": lambda d: xr.concat([d, d], dim="rep"),
}

matrix = {}
for label, op in ops.items():
    out = op(probe)
    coord = out.coords.get("reference_frequency")
    matrix[label] = {
        "object attrs survive": bool(out.attrs),
        "coord value survives": coord is not None and 120.66 in np.atleast_1d(coord.values),
        "coord attrs survive": coord is not None and coord.attrs.get("units") == "MHz",
    }
pd.DataFrame(matrix).T
```

Three facts, and they shape everything below. Object attrs survive almost nothing — the
problem section, quantified. A coordinate's **value** survives *everything on this list*,
including the reduction that removes the spectral axis. A coordinate's **attrs** survive
everything except one function — `xr.where` (the function form; the everyday method
`da.where` is safe), which rebuilds coordinates bare. So a constant stored as a coordinate
*value* is bulletproof; a constant stored in a coordinate's *attrs* has exactly one known
leak. Three homes follow from this table — they differ in which side of it they sit on, and
in what they do to the repr.

(constants-nb-perconstant)=
## Option P — one coordinate per constant

The maximal-robustness corner: every constant is its own scalar coordinate (value = the
constant, attrs = its unit via `as_variable`, Commandment 7's machinery). Travel is the
"value" column above — no leak at all — and disagreement gets the best semantics on this
page:

```{code-cell} ipython3
ge = probe.copy(deep=True)
siemens = probe.copy(deep=True).assign_coords(reference_frequency=500.13)

summed = ge + siemens
drifting = xr.concat([ge, ge.assign_coords(reference_frequency=120.68)], dim="repetition")

print("cross-field sum:  constant kept?", "reference_frequency" in summed.coords)
print("drifting series: ", drifting.reference_frequency.values, drifting.reference_frequency.dims)
```

Conflicting values are *dropped* — the combined object honestly has no single reference
frequency, and the next gate refuses loudly. A drifting series is *promoted* to a per-scan
array — the truth, recorded. This option fails somewhere else entirely. Put the horizon
family on the measuring stick:

```{code-cell} ipython3
mrsi_per = mrsi.assign_coords({k: xr.Variable((), v) for k, v in HORIZON.items()})
mrsi_per
```

Ten constants shoulder-to-shoulder with four real axes, indistinguishable from data
coordinates at a glance. This is the review objection, on screen: at MRSI scale the repr —
the single most-read surface in a notebook — becomes a wall. Mitigations were considered and
don't rescue it: insertion order *is* preserved through operations (the constants stay
grouped at the end), but ten lines are ten lines; an `xmr_` name prefix makes the grouping
visible at the cost of uglier names everywhere (`da.xmr_reference_frequency`), and xarray
offers no per-object way to fold coordinates away. Right physics, unreadable at scale.

(constants-nb-container)=
## Option C — the container coordinates

The geoscience stacks hit this exact wall and left a pattern: rioxarray carries an entire
coordinate-reference system as **one** scalar coordinate (`spatial_ref`) whose attrs hold the
fields — CF's grid-mapping design, netCDF-native, proven at survey scale. The xmris shape is
that pattern, twice — a two-line block that ends the coordinate list:

- **`xmr_acquisition`** — attrs are the physical constants, flat and typed as 02's law
  demands; its **value** is a deterministic fingerprint of them, which buys back the honesty
  the attrs side of the matrix loses (xarray compares values, never attrs).
- **`xmr_history`** — the record rides the same block: dossier 02's JSON envelope, format
  untouched, but its **home** moves from a droppable object attr into a coordinate whose
  *value is the envelope itself* — so identical histories travel and diverged histories drop
  honestly, by xarray's own rules.

```{code-cell} ipython3
import functools
import hashlib
import json

PHYSICS = (ATTRS.reference_frequency, ATTRS.carrier_ppm, ATTRS.group_delay, ATTRS.b0_field)
CONTAINERS = ["xmr_acquisition", "xmr_history"]
EMPTY_HISTORY = json.dumps({"schema": 1, "events": []})


def fingerprint(constants: dict) -> str:
    """Deterministic identity of a calibration (no timestamps — reruns stay identical)."""
    return hashlib.sha256(json.dumps(constants, sort_keys=True).encode()).hexdigest()[:8]


def mint_containers(da: xr.DataArray) -> xr.DataArray:
    """Prototype of the central bookkeeping: home the constants and the record,
    and pin the block to the bottom of the coordinate list."""
    carried = (
        dict(da.coords["xmr_acquisition"].attrs) if "xmr_acquisition" in da.coords else {}
    )
    loose = {str(t): da.attrs[t] for t in PHYSICS if t in da.attrs}
    constants = {**carried, **loose}
    history = (
        str(da.coords["xmr_history"].values) if "xmr_history" in da.coords else EMPTY_HISTORY
    )

    # assign_coords on an existing key updates in place (measured) — dropping
    # first and re-assigning is what moves the block to the end.
    out = da.drop_vars([c for c in CONTAINERS if c in da.coords])
    if constants:
        out = out.assign_coords(
            xmr_acquisition=xr.Variable((), fingerprint(constants), constants)
        )
    out = out.assign_coords(xmr_history=xr.Variable((), history))
    out.attrs = {k: v for k, v in out.attrs.items() if k not in loose}
    return out


def reads_constants(func):
    """Prototype of the gate change: constants are found in the container, then attrs,
    and the call appends its event to the record (dossier 02's decorator, one day)."""

    @functools.wraps(func)
    def wrapper(da, *args, **kwargs):
        found = {}
        if "xmr_acquisition" in da.coords and da.coords["xmr_acquisition"].ndim == 0:
            found = {
                k: v for k, v in da.coords["xmr_acquisition"].attrs.items() if k in PHYSICS
            }
        result = func(da.assign_attrs(found), *args, **kwargs)
        result.attrs = {k: v for k, v in result.attrs.items() if k not in found}
        result = mint_containers(result)
        envelope = json.loads(str(result.coords["xmr_history"].values))
        envelope["events"].append({"op": func.__name__})
        return result.assign_coords(xmr_history=xr.Variable((), json.dumps(envelope)))

    return wrapper


# Graft onto the live library, notebook-locally: the .xmr methods are thin
# delegators to these module-level names, so rebinding gives the real chained
# UX without touching src/.
import xmris.core.accessor as _accessor

for _name in ("to_ppm", "to_hz", "to_spectrum", "apodize_exp"):
    setattr(_accessor, _name, reads_constants(getattr(_accessor, _name)))
```

(constants-nb-container-ux)=
### The measuring stick, again

```{code-cell} ipython3
mrsi_one = mrsi.assign_coords(
    xmr_acquisition=xr.Variable((), fingerprint(HORIZON), attrs=HORIZON),
    xmr_history=xr.Variable((), EMPTY_HISTORY),
)
mrsi_one
```

Ten constants — and the processing record beside them — in a **two-line block** that stays
two lines whether the family grows to fifteen or fifty. In a Jupyter repr the
`xmr_acquisition` entry expands to the full constants dict on one click; programmatically it
is a plain mapping, one hop away, with `explain()` ([below](#constants-nb-explain)) as the
curated view:

```{code-cell} ipython3
dict(mrsi_one.xmr_acquisition.attrs)
```

Travel is the matrix's "attrs" column — everything except `xr.where` — plus the block itself
surviving axis-dropping reductions (it outlives `mean("time")`, where axis-carried constants
below die). The opening section's failure, replayed end-to-end:

```{code-cell} ipython3
spectrum_c = mint_containers(fid).xmr.apodize_exp(lb=5.0).xmr.to_spectrum()
detour = np.abs(spectrum_c * 2) / spectrum_c.size  # plain xarray, twice over
detour.xmr.to_ppm().coords["chemical_shift"]
```

No flag, no ceremony, no failure. Entry for an outsider is today's documented plain-string
path, unchanged — the first gate re-homes loose attrs into the container, and from then on
they travel:

```{code-cell} ipython3
outsider = xr.DataArray(
    np.random.default_rng(0).normal(size=128),
    dims=["frequency"],
    coords={"frequency": np.linspace(-500, 500, 128)},
    attrs={"reference_frequency": 120.66, "carrier_ppm": 4.7},  # plain strings, as today
)
referenced = outsider.xmr.to_ppm()
print("re-homed:", dict(referenced.xmr_acquisition.attrs))
print("recorded:", str(referenced.xmr_history.values))
print("survives the next plain op?", "xmr_acquisition" in (referenced * 2).coords)
```

(constants-nb-container-pinned)=
### Pinned to the bottom, named as a block

The review asked for a guarantee: these must not read as data coordinates, and they must sit
at the end of the list. Both are enforceable, and the enforcement was measured. xarray
displays coordinates in **insertion order** — there is no auto-sorting — so `mint_containers`
pins the block by dropping and re-assigning it last on every xmris return (a plain
re-assignment updates in place and would *not* move it). Ordinary operations then preserve
that order:

```{code-cell} ipython3
after = {
    "spectrum_c * 2": spectrum_c * 2,
    "np.abs(spectrum_c)": np.abs(spectrum_c),
    "spectrum_c + spectrum_c": spectrum_c + spectrum_c,
    "spectrum_c.where(|s| > 0.1)": spectrum_c.where(np.abs(spectrum_c) > 0.1),
    "spectrum_c.isel(slice)": spectrum_c.isel(frequency=slice(0, 512)),
    "xr.concat([...], 'rep')": xr.concat([spectrum_c, spectrum_c], dim="rep"),
}
pd.DataFrame(
    {"xmr_* block is still last": {k: list(v.coords)[-2:] == CONTAINERS for k, v in after.items()}}
)
```

Three cases can push the block off the bottom — a user's own later `assign_coords`, `groupby`
re-appending the grouped coordinate, and netCDF reload order (engine-determined). None of
them is silent damage, and every one self-heals at the next xmris touch:

```{code-cell} ipython3
slipped = spectrum_c.assign_coords(mask=("frequency", np.ones(spectrum_c.sizes["frequency"])))
print("after a user assign_coords:", list(slipped.coords))
print("after the next xmris call: ", list(slipped.xmr.to_ppm().coords))
```

The names do the other half of the review's ask. The `xmr_` prefix marks ownership at a
glance — it is the accessor's own name, so the mental link "`xmr_*` belongs to `.xmr`" is
free — it renders the pair as one visual block above the Attributes section, and it cannot
collide with any vendor's axis names. The words after the prefix are the domain's own:

| Coordinate | Value (what xarray compares) | Attrs | Read surface |
|---|---|---|---|
| `xmr_acquisition` | calibration fingerprint | the constants, flat + typed | one click in the repr · `explain()` · `da.xmr_acquisition.attrs` |
| `xmr_history` | the JSON envelope — 02's format, verbatim | — | `da.xmr.history()` |

(constants-nb-container-conflict)=
### When calibrations — or histories — disagree

The fingerprint value is what xarray compares, so conflicts behave like Option P's — honest,
and each container answers independently:

```{code-cell} ipython3
other_cal = mint_containers(
    fid.assign_attrs(reference_frequency=500.13)
).xmr.apodize_exp(lb=5.0).xmr.to_spectrum()

mixed = spectrum_c + other_cal
print("calibration kept?", "xmr_acquisition" in mixed.coords,
      "| record kept?", "xmr_history" in mixed.coords)
try:
    mixed.xmr.to_ppm()
except ValueError as err:
    print(str(err).splitlines()[0])
```

Different constants → different fingerprints → the calibration is dropped whole and the next
gate refuses loudly, instead of `keep_attrs`' confidently wrong axis — while the record
survives, because both operands carry the *same* history. The mirror case: same calibration,
diverged processing —

```{code-cell} ipython3
blend = spectrum_c + spectrum_c.xmr.apodize_exp(lb=0.5)  # one more recorded step on one side

print("calibration kept?", "xmr_acquisition" in blend.coords,
      "| record kept?", "xmr_history" in blend.coords)
```

The blend keeps its constants (they still hold) and honestly loses its lineage (no single
history describes it) — exactly the right answer in both columns, and nothing in the library
invented merge semantics to get it. A drifting series promotes the fingerprint to a per-scan
array — visibly non-uniform, so a gate meeting a non-scalar container can refuse with "this
series is not uniformly calibrated" (the exact message is dossier 03's schema work):

```{code-cell} ipython3
drifted = xr.concat(
    [mint_containers(fid), mint_containers(fid.assign_attrs(reference_frequency=120.68))],
    dim="repetition",
)
drifted.xmr_acquisition.values
```

One honest cost, from the matrix's one leak: `xr.where` (function form) strips the
container's attrs — but keeps its value, so the damage is *detectable*, not silent:

```{code-cell} ipython3
holed = xr.where(np.abs(spectrum_c) > 0.1, spectrum_c, 0)
print("container present:", "xmr_acquisition" in holed.coords,
      "| constants left:", dict(holed.xmr_acquisition.attrs))
```

A gate that finds the container present but empty can say precisely what happened and how to
fix it — a named, loud, one-function edge case (today `xr.where` already strips the
`units` metadata Commandment 7 puts on every ppm axis, so this is an upstream wart worth an
xarray issue regardless of this decision).

(constants-nb-container-file)=
### What survives a file

```{code-cell} ipython3
import tempfile
from pathlib import Path

magnitude = spectrum_c.copy(data=np.abs(spectrum_c.values))  # netCDF holds no complex values

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "spectrum.nc"
    magnitude.rename("spectrum").to_netcdf(path)
    loaded = xr.load_dataarray(path)

print("coordinate order after reload:", list(loaded.coords))
loaded.xmr_acquisition
```

The container is a real netCDF variable carrying the constants as its attributes — the CF
grid-mapping shape, self-describing under any tool's `ncdump`, no xmris required to read it.
(Reload order is the netCDF engine's choice — one of the three slip cases above; the first
xmris call re-pins the block.)

:::{dropdown} Fine print — costs and open ends, honestly
- **The 02 amendment, stated precisely.** Dossier 02 closed the record's *format* (one JSON
  envelope, versioned, timestamp-free) and named the key `xmr_history`; this dossier moves
  its *home* from `da.attrs["xmr_history"]` to the coordinate's value — same name, same
  envelope, one line of 02's Resolution to amend at the harvest. Everything else 02 decided
  (write-only for library math, `da.xmr.history()` as the reading surface, no invented merge
  semantics) carries over unchanged.
- **Stale attrs on a promoted container.** After the drift `concat`, the per-scan
  fingerprints are truthful but the container's attrs still show scan 1's constants
  (first-wins on attrs is unfixable from library code). The gate's non-scalar check is what
  makes this safe: a promoted container is *unreadable* until sliced back to one scan —
  `drifted.isel(repetition=0)` — which restores that scan's fingerprint.
- **Contributor footprint.** Two writers (`simulate_fid`, the Bruker loader) mint the block
  instead of `assign_attrs`; dossier 02's central decorator gains the re-pin (drop + assign
  last) and the container read; one reader helper behind `requires_attrs` (and the direct
  `attrs.get` reads in `to_ppm`, `to_hz`, `fit_amares`, `remove_digital_filter`) looks
  container-first-then-attrs, with the attrs fallback kept indefinitely — data written under
  today's law keeps working, and outsider entry stays plain `assign_attrs` (the lazy
  re-homing makes the existing fix-lines correct). `TestAttrsPreservation` is untouched; one
  new parametrized test pins "the block is the last two coordinates" across the public API.
- **Prototype cheats.** Only four functions are wrapped here (the real change is the central
  decorator, so every function records and re-pins); events carry just `op` (the real
  envelope carries params and version, per 02); `mint_containers` runs at the chain head
  (real writers mint at the source).
:::

(constants-nb-axis)=
## Option X — constants ride the axis they calibrate

The zero-repr-cost corner: `reference_frequency` and `carrier_ppm` calibrate the spectral
axis, `group_delay` the time axis — so store each constant in the attrs of the dimension
coordinate it belongs to. No new repr line at all, and a certain semantic beauty: the
calibration lives on the thing it calibrates.

```{code-cell} ipython3
axis_da = xr.DataArray(
    np.linspace(-1, 1, 8),
    dims=["frequency"],
    coords={
        "frequency": (
            "frequency",
            np.arange(8.0),
            {"units": "Hz", "reference_frequency": 120.66, "carrier_ppm": 4.7},
        )
    },
)
print("survives da*2:", (axis_da * 2).frequency.attrs.get("reference_frequency"))
print("survives mean('frequency'):",
      "frequency" in axis_da.mean("frequency").coords)
```

Both edges cut against it. The constants die whenever their axis does — and an axis dies in
legitimate workflows (integrating a peak, collapsing a spectral region into a map: exactly
the fit-derived amplitude maps MRSI produces, which still deserve their acquisition context).
It shares `xr.where`'s attrs leak *without* a surviving value to detect it by, and binary-op
conflicts are first-wins (measured — the axis values agree, so xarray keeps the first
operand's attrs wholesale). And the class has members with no axis at all: `b0_field` today,
echo/repetition timing tomorrow — which forces a second home and makes the whole design
two mechanisms instead of one. The converters would also need to hand-carry constants across
every axis swap (`to_spectrum` builds its output axis from scratch). Elegant for the axis
constants it fits, structurally partial for the family.

(constants-nb-explain)=
## Discoverability — `explain()`

Whatever the home, discoverability is one accessor method: walk the object, match every name
against the vocabulary, surface the curation. Under the container it reads one level deeper —
and the native repr already carries the constants one click away:

```{code-cell} ipython3
from xmris.core.config import COORDS, DIMS


def _term_for(key: str):
    for vocab in (ATTRS, COORDS, DIMS):
        for term in vocab._get_terms().values():
            if term == key:
                return term
    return None


def explain(da: xr.DataArray) -> pd.DataFrame:
    """What `da.xmr.explain()` would return."""
    rows = []
    if "xmr_acquisition" in da.coords:
        for key, value in da.coords["xmr_acquisition"].attrs.items():
            rows.append((str(key), "constant", value, _term_for(key), None))
    if "xmr_history" in da.coords:
        events = json.loads(str(da.coords["xmr_history"].values))["events"]
        rows.append((
            "xmr_history", "record", f"{len(events)} events", None,
            "The append-only processing record — read it with da.xmr.history().",
        ))
    for name, coord in da.coords.items():
        if coord.ndim == 0:
            continue
        span = f"{coord.size} points, {coord.values.min():g} … {coord.values.max():g}"
        rows.append((str(name), "coordinate", span, _term_for(name), None))
    for key, value in da.attrs.items():
        rows.append((str(key), "attr", value, _term_for(key), None))

    def describe(term, override):
        if override:
            return override
        if term is None:
            return "⚠ not in the xmris vocabulary"
        return term.description if len(term.description) <= 110 else term.description[:110] + "…"

    return pd.DataFrame(
        {
            "lives as": [r[1] for r in rows],
            "value": [r[2] for r in rows],
            "unit": [r[3].unit if r[3] else "" for r in rows],
            "description": [describe(r[3], r[4]) for r in rows],
        },
        index=pd.Index([r[0] for r in rows], name="name"),
    )


explain(spectrum_c.xmr.to_ppm())
```

Every recognized name answers for itself — readable wording, unit, inline — and the table
doubles as an audit: the literal keys `simulate_fid` leaks past the vocabulary today light up
as *not in the xmris vocabulary*, which is how a growing constants family stays curated
instead of accumulating folklore.

(constants-nb-verdict)=
## Side by side

| | `keep_attrs` family | P — per-constant coords | C — container block | X — axis-carried |
|---|---|---|---|---|
| Survives plain ops | only under the flag | everything measured | everything but `xr.where` | dies with its axis |
| Cross-calibration sum | first wins — silent lie | dropped → loud gate | dropped → loud gate | first wins — silent lie |
| Drifting series (`concat`) | first scan's values | per-scan truth | per-scan fingerprints, gate-detectable | first scan's values |
| Repr cost at MRSI scale | none | **+10 lines and growing** | +2 pinned lines, constant | none |
| The record can share the home | no | no — needs its own coord anyway | yes — sibling coordinate | no |
| Ceremony / session | per-notebook or owns session | none | none | none |
| Covers axis-less constants | yes | yes | yes | no — needs a second home |
| netCDF shape | header attrs | 10 scalar variables | one CF grid-mapping-style block | axis attrs |
| Script ≡ notebook | not under auto-detect | yes | yes | yes |
| Migration | docs only | writers + gates | writers + gates + decorator re-pin | writers + gates + converters |

:::{tip} The decision — converged 2026-08-03: the container block, plus `xmr.explain()`
Two scalar coordinates, pinned as the last entries of every xmris return.
**`xmr_acquisition`** holds the physical constants as its attrs — flat, typed, one click open
in the repr — with a deterministic calibration fingerprint as its value, so disagreement is
dropped honestly instead of first-wins lied about. **`xmr_history`** carries dossier 02's
JSON envelope, format untouched, as its value — the record now travels every plain operation
the constants do, and diverged histories vanish honestly by xarray's own rules (an amendment
to 02's storage home, folded at the harvest). Writers mint the block; gates read
container-first with an indefinite plain-attrs fallback and lazily re-home outsider entry;
the central decorator re-pins the block last on every return, and the measured ordering rules
(insertion order displayed, preserved by ordinary ops, three self-healing slips) make
"always at the bottom" an enforceable contract rather than a hope. `xmr.explain()` surfaces
the vocabulary's curation over all of it and audits what falls outside.

Option P keeps the strictly strongest travel but fails the repr at exactly the scale the
roadmap targets; option X is beautiful for axis constants and structurally partial;
`keep_attrs` in every wrapper lies under combination and splits notebook from script. Feeds
dossier 03 the schema sentence — *an xmris object carries its constants and record in the
pinned `xmr_*` block* — and re-scopes
[#21](https://github.com/andrewendlinger/xmris/issues/21) (the outside-xmris half becomes
structural) and [#22](https://github.com/andrewendlinger/xmris/issues/22) (validation targets
one dict, in one reader). Revisit if xarray flips its default attrs propagation or fixes
`xr.where`'s coordinate-attrs strip, or if per-scan constants become a first-class workflow
(a promoted container then wants reading semantics, not a refusal). The Resolution lives in
`roadmap_issue_02b_physics_attrs.md`; this page retires at the harvest.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# Pin the load-bearing claims so this page fails loudly if behavior shifts.

# The problem, and the undone session flag.
assert scaled.attrs == {}
assert xr.get_options()["keep_attrs"] == "default"

# keep_attrs family: first-wins lie under the flag.
assert merged.attrs["reference_frequency"] == 120.66

# The matrix rows everything builds on.
assert matrix["da * 2"]["coord value survives"] and matrix["da * 2"]["coord attrs survive"]
assert matrix["np.abs(da)"]["coord attrs survive"]
assert matrix["da.where(da > 0)"]["coord attrs survive"]
assert matrix["da.mean('frequency')"]["coord value survives"]
assert not matrix["xr.where(da > 0, da, 0)"]["coord attrs survive"]
assert matrix["xr.where(da > 0, da, 0)"]["coord value survives"]
assert not any(row["object attrs survive"] for row in matrix.values())

# Option P: honest conflicts, per-scan truth, and the repr sprawl being objected to.
assert "reference_frequency" not in summed.coords
assert list(drifting.reference_frequency.values) == [120.66, 120.68]
assert sum(1 for c in mrsi_per.coords.values() if c.ndim == 0) == len(HORIZON)

# Option C: travel, re-homing, and the record riding along.
assert dict(mrsi_one.xmr_acquisition.attrs) == HORIZON
assert (np.abs(spectrum_c * 2)).xmr_acquisition.attrs["reference_frequency"] == 120.66
assert referenced.xmr_acquisition.attrs["reference_frequency"] == 120.66
assert "reference_frequency" not in referenced.attrs
assert "xmr_acquisition" in (referenced * 2).coords
assert json.loads(str(referenced.xmr_history.values))["events"] == [{"op": "to_ppm"}]
assert [e["op"] for e in json.loads(str(spectrum_c.xmr_history.values))["events"]] == [
    "apodize_exp", "to_spectrum",
]

# Pinned to the bottom: preserved by plain ops, healed after a slip.
assert all(list(v.coords)[-2:] == CONTAINERS for v in after.values())
assert list(slipped.coords)[-2:] != CONTAINERS  # the slip is real...
assert list(slipped.xmr.to_ppm().coords)[-2:] == CONTAINERS  # ...and heals

# Conflicts: each container answers independently.
assert "xmr_acquisition" not in mixed.coords and "xmr_history" in mixed.coords
assert "xmr_acquisition" in blend.coords and "xmr_history" not in blend.coords
assert drifted.xmr_acquisition.dims == ("repetition",)
assert len(set(drifted.xmr_acquisition.values)) == 2

# The xr.where residue is detectable, and the file keeps everything.
assert "xmr_acquisition" in holed.coords and dict(holed.xmr_acquisition.attrs) == {}
assert loaded.xmr_acquisition.attrs["reference_frequency"] == 120.66
assert str(loaded.xmr_acquisition.values) == str(spectrum_c.xmr_acquisition.values)
assert str(loaded.xmr_history.values) == str(spectrum_c.xmr_history.values)

# Determinism (02's assert_identical law): identical pipelines, identical objects.
rerun = mint_containers(fid).xmr.apodize_exp(lb=5.0).xmr.to_spectrum()
xr.testing.assert_identical(spectrum_c, rerun)

# Option X: dies with its axis; conflict is first-wins (the measured flaw).
assert "frequency" not in axis_da.mean("frequency").coords
_xa, _xb = axis_da, axis_da.copy(deep=True)
_xb.coords["frequency"].attrs["reference_frequency"] = 500.13
assert (_xa + _xb).frequency.attrs["reference_frequency"] == 120.66

# explain(): curated, container-aware, and auditing the leaks.
table = explain(spectrum_c.xmr.to_ppm())
assert table.loc["reference_frequency", "unit"] == "MHz"
assert table.loc["reference_frequency", "lives as"] == "constant"
assert table.loc["xmr_history", "value"] == "3 events"
assert (table["description"] == "⚠ not in the xmris vocabulary").any()
```
