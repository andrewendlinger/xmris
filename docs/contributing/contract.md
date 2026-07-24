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

(contract)=
# The Architecture Contract

This page is the law for everything under `src/xmris/`: eleven numbered rules — the
**Commandments** — that every library change must obey. Cite them by number; ordinals 1–8 are
stable (code and tooling reference them), and 9–11 codify patterns the test suite was already
enforcing. The *why* lives elsewhere: the
[architecture tour](../notebooks/basics/architecture.md) motivates the xarray-first design,
[The Two Domains](../explanation/domains.md) derives the domain decorators, and
[The Controlled Vocabulary](../explanation/vocabulary.md) explains the config singletons in
`xmris.core.config` (which sits beside `validation.py` and `utils.py` under `src/xmris/core/`).
The authoring skills and `CLAUDE.md` route here; where another contributor page differs from this
one, this page wins.

Each rule closes with its enforcement — the test class in `tests/test_core.py` that guards it, or
an honest *reviewer checks* where none does.

(contract-rules)=
## The Commandments

(contract-c1)=
### 1. Xarray in, xarray out

Every public function takes and returns `xr.DataArray` or `xr.Dataset`. Private numeric kernels
(e.g. `_simulate_fid_ndarray`) may drop to NumPy internally — the boundary stays xarray.
*Reviewer checks.*

(contract-c2)=
### 2. Functional purity

Never modify the input in place. Copy, transform, return a new object.
*Reviewer checks; `TestAttrsPreservation` guards the metadata half.*

(contract-c3)=
### 3. Lineage: preserve, append the parameters applied, never flag

Preserve inbound coordinates and attributes, then append what the function actually applied under
`ATTRS` keys — a scalar (`phase_p0=15.0`), a config-blessed string (`baseline_method="als"`,
`zero_fill_position="end"`), or a list (`simulate_fid`'s `sim_amplitudes`). Banned: state flags
(`phase_applied=True`) — the applied parameter's presence *is* the record. *Enforced:
`TestAttrsPreservation`.* The wider attrs strategy — preservation guarantees and structured
provenance (`xmr_history`) — is an open design decision
([#64](https://github.com/andrewendlinger/xmris/issues/64), with
[#21](https://github.com/andrewendlinger/xmris/issues/21) and
[#23](https://github.com/andrewendlinger/xmris/issues/23)); do not build ahead of it.

(contract-c4)=
### 4. No magic strings — the vocabulary is law

Inside `src/xmris/`, dimension, coordinate, attribute and variable names come from the
`ATTRS`/`DIMS`/`COORDS`/`VARS` singletons in `xmris.core.config`, never a bare `"time"`. (User
code and reader-facing examples use plain strings deliberately — the low entrance barrier is a
feature.) A missing term is added to `config.py` — lowercase keys, singular dim names — and every
new term is called out explicitly in the change. The legacy `xmris.config.DEFAULTS` shim is
deprecated: never in new code. The one sanctioned exception is Commandment 11. *Enforced:
`TestConfigNamingConventions`, `TestConfigMetadata`, `TestVocabularyUniqueness`; the
no-bare-strings half — reviewer greps, since an `XmrisTerm` equals its string.*

(contract-c5)=
### 5. The dim-default biconditional

A `dim` argument defaults to its config constant (`dim: str = DIMS.time`) — *except* it defaults
to `None` **iff** the function carries a *multi-label* domain decorator (today only
`SPECTRAL_DIMS`), whose merged resolution fills it at call time. *Enforced: `TestDomainDimRule`.*
Why: [The Two Domains](../explanation/domains.md).

(contract-c6)=
### 6. Declare the contract at the door

Gate hidden state with `@requires_attrs(...)`. Declare a domain-sensitive function's working
domain with `@ensures_domain` (funnel: the result stays there) or `@computes_in`
(domain-preserving: the representation is restored). Converters, FFT primitives, vendor loaders
and fitting stay undecorated by design — their transforms are explicit; never inline `fft`/`ifft`
for domain handling, route through the converters. Validate dimensions with
`_check_dims(da, dim, "func_name")`. *Enforced: `TestDomainRollout` pins every function's
contract; the semantics under `TestEnsuresDomain`/`TestComputesIn`.* Which decorator: the
[decision tree in The Two Domains](../explanation/domains.md).

(contract-c7)=
### 7. Coordinates are built by `as_variable`

Never hand-assemble a `{"units": ..., "long_name": ...}` dict. `as_variable(COORDS.term, dim,
data)` bundles data and term metadata into a fully formed `xr.Variable` for `.assign_coords()`.
*Reviewer greps `src/` for literal `"long_name"`.*

(contract-c8)=
### 8. Explicit MyST targets in docs

Every docs header carries an explicit `(kebab-target)=` and is linked via `[text](#target)`, never
an auto-generated slug — which mystmd numbers by *document position*, so inserting one section
silently renumbers every anchor below it. The target is kebab-case and prefixed with the page
topic (`baseline-visualizing-the-results`, not `visualizing-the-results`), since targets resolve
page-globally. The full documentation law lives with the
[docs-page workflow](#contribute-docs). *Enforced: `check_docs.py`, run over the whole tree by
the `Docs style` job in `ci-fast.yml`.*

(contract-c9)=
### 9. The accessor method is a thin delegator

An `.xmr` method contains no logic: `return free_func(self._obj, ...)`. Defaults are copied
verbatim from the free function, every parameter is forwarded explicitly (a keyword reachable only
through `**kwargs` makes the docstring lie), and the docstring documents the method — it takes
`self`, not `da`. *Enforced in part: `TestAccessorDefaults` pins `dim` defaults for its listed
methods; full signature parity is open
([#102](https://github.com/andrewendlinger/xmris/issues/102)).*

(contract-c10)=
### 10. Errors end with the fix

Every `raise` a user can hit ends with a copy-pasteable recovery line — `>>> obj =
obj.rename({...})`, `>>> obj = obj.assign_attrs({...})`. `_check_dims` and `requires_attrs`
(in `core/utils.py` and `core/validation.py`) are the house exemplars. *Enforced in part:
`TestCheckDims` pins the rename fix; new messages — reviewer checks.*

(contract-c11)=
### 11. Deliberately-local axes carry the marker

A diagnostic output axis deliberately kept out of the vocabulary is tagged
`# xmris-diagnostic-dim` at its definition site, so every escape hatch stays greppable and
revocable. Exemplar: `estimate_group_delay`'s `trial_delay` axis in `vendor/bruker.py`. *The
marker is the enforcement.*

(contract-exemplars)=
## The rules in real code

There is no hand-written template: the exemplars below are quoted from the live source at build
time, so they cannot drift. Read `apodize_exp` top to bottom as the walkthrough — the domain is
declared at the door (6), the dimension validated (`_check_dims`, 6), the math pure (2), and the
applied parameter appended to `.attrs` (3):

```{literalinclude} ../../src/xmris/processing/fid.py
:language: python
:start-at: @computes_in(TIME_DIMS)
:end-at: return da_apodized
:caption: apodize_exp — quoted from src/xmris/processing/fid.py at build time
```


`to_ppm` shows the other half: an attribute gate (`@requires_attrs`, 6) on a deliberately
undecorated converter, and a coordinate built by `as_variable` (7) before the `swap_dims`:

```{literalinclude} ../../src/xmris/processing/referencing.py
:language: python
:start-at: @requires_attrs(ATTRS.reference_frequency
:end-at: return obj.swap_dims({dim: DIMS.chemical_shift})
:caption: to_ppm — quoted from src/xmris/processing/referencing.py at build time
```

Which decorator stack a *new* function copies — including the case neither exemplar shows — is the
`xmr-method` skill's job ([Add a processing method](#contribute-methods)).

(contract-executed)=
## The contract, executed

The rules above are claims; this cell runs them. It executes on every PR build, together with
hidden asserts that hold the page to Commandments 2, 3 and 7 — and pin the two quotes above, since
a silently truncated `literalinclude` would otherwise only warn.

```{code-cell} python
import xmris

fid = xmris.simulate_fid(
    amplitudes=[1.0, 0.6],
    chemical_shifts=[0.0, 5.2],
    reference_frequency=120.66,  # MHz — enables ppm referencing
    n_points=1024,
)
fid_before = fid.copy(deep=True)

spectrum = fid.xmr.apodize_exp(lb=5.0).xmr.to_spectrum().xmr.to_ppm()
spectrum.attrs
```

```{code-cell} python
:tags: [remove-cell]
import inspect

import xarray as xr

from xmris.core.config import ATTRS, COORDS

# Commandment 2 — the input is untouched
xr.testing.assert_identical(fid, fid_before)

# Commandment 3 — the applied parameter is the record; no boolean flags anywhere
assert spectrum.attrs[ATTRS.apodization_lb] == 5.0
assert not any(isinstance(v, bool) for v in spectrum.attrs.values())

# Commandment 7 — the ppm axis carries as_variable's term metadata
assert spectrum.coords[COORDS.chemical_shift].attrs["units"] == "ppm"

# Pin the literalinclude anchors: mystmd only WARNS on an unmatched
# start-at/end-at, so a renamed decorator or return statement would silently
# truncate the quotes above. These asserts fail the PR build loudly instead.
src = inspect.getsource(xmris.apodize_exp)
assert src.splitlines()[0] == "@computes_in(TIME_DIMS)"
assert src.rstrip().splitlines()[-1].strip() == "return da_apodized"
assert "ATTRS.apodization_lb" in src

src = inspect.getsource(xmris.to_ppm)
assert src.splitlines()[0] == "@requires_attrs(ATTRS.reference_frequency, ATTRS.carrier_ppm)"
assert src.rstrip().splitlines()[-1].strip() == "return obj.swap_dims({dim: DIMS.chemical_shift})"
```

(contract-open)=
## Open questions

The contract has one moving edge: the attrs strategy.
[#64](https://github.com/andrewendlinger/xmris/issues/64) decides whether lineage stays flat
per-parameter keys (today's law) or becomes a structured `xmr_history` log. Commandment 3 states
today's law; when #64 resolves, it and this page change together.
