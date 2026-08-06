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

(domains)=
# The Two Domains

An MRS measurement is one signal seen from two sides. The scanner records a **free induction
decay** (FID) — a complex signal over *time*. Its Fourier transform is the **spectrum** — the same
information over *frequency*, where peaks become visible and integrable. Nothing is gained or lost
moving between them: xmris uses unitary transforms (`norm="ortho"`), so the round trip is exact to
floating-point precision.

In xmris, **the dimension name is the domain**. A DataArray with a `time` dim is time-domain along
that axis; one with a `frequency` (Hz) or `chemical_shift` (ppm) dim is spectral. There is no
hidden flag to drift out of sync — the domain is readable, per axis, right in the `repr`.

```{dropdown} Why per-axis domain state matters
A CSI dataset can be image-space along `(x, y, z)` while still time-domain along `t`. A single
object-level "is this a spectrum?" flag cannot represent that; per-dimension naming can. This is
the classic per-dimension domain bookkeeping of NMR processing software, expressed natively in
xarray.
```

(domains-problem)=
## The problem

Every processing operation has a *home domain* where its math lives. Phasing and baseline work are
spectral-domain jobs; apodization is a time-domain multiplication. But your data is wherever your
pipeline currently holds it — so what should happen when you call a spectral operation on a FID, or
a time-domain operation on a spectrum?

Left unmanaged, this ends one of two ways: either you pepper pipelines with manual conversions and
remember each function's home domain yourself, or the library converts silently and you stop being
able to predict what a call returns. Both are failure modes. The second is worse.

(domains-goal)=
## The goal

One sentence:

```{important}
**The output domain of every operation is a pure function of the operation and the input domain —
never of runtime values.** You can read a pipeline and know, line by line, which domain the data is
in.
```

To honor that sentence, operations cannot all follow the same rule — because they do not all have
the same relationship to their home domain. xmris distinguishes four kinds.

(domains-op-classes)=
## Four kinds of operations

```{mermaid}
flowchart LR
    T["Time domain<br/><code>time</code>"] -- "to_spectrum()" --> S["Spectral domain<br/><code>frequency</code> (Hz) or <code>chemical_shift</code> (ppm)"]
    S -- "to_fid()" --> T
    S -- "to_ppm() &harr; to_hz()" --> S
```

(domains-funnel)=
### Funnel operations — the result *lands* in the home domain

`autophase`, `baseline_als`. These are only meaningful on a spectrum, and their result is consumed
as a spectrum — you phase in order to look at peaks. So a FID input is transformed on the way in
and **stays there**:

```python
fid.xmr.autophase()        # → phased spectrum, ready to inspect
spec.xmr.autophase()       # → phased spectrum (already home — no transform)
```

For `baseline_als` there is no alternative even in principle: it discards the imaginary component,
and a real-valued spectrum has no valid FID behind it anymore.

(domains-preserving)=
### Domain-preserving operations — same physics, your representation kept

`apodize_exp`, `apodize_lg`, `zero_fill`. Their math is identical seen from either side —
multiplying an FID by $e^{-\pi\,\mathrm{lb}\,t}$ *is* convolving its spectrum with a Lorentzian of
width $\mathrm{lb}$ Hz; zero-filling an FID *is* interpolating its spectrum onto a finer grid. So
these never change what you hold: internally the values take a round trip through the time domain,
and the output comes back in the input's representation, original coordinates intact.

```python
fid.xmr.apodize_exp(lb=2)    # → FID   (home domain — no transform)
spec.xmr.apodize_exp(lb=2)   # → spectrum, smoothed (round trip inside)
```

**Fitting (`fit_amares`) is domain-preserving too**, with a twist. It models the FID, but you may
hand it a spectrum: it round-trips through the time domain to fit and returns its signal variables
(`data`/`fit`/`residuals`) in the representation you passed — ppm in, ppm out — while the quantified
parameters are domain-independent. It is the one such operation that carries **no decorator**: it
returns a `Dataset`, and the round trip must transform only the signals, never the parameter table,
so it hand-rolls the same converter routing the decorators use.

(domains-converters)=
### Converters — the only functions that change representation *on purpose*

`to_spectrum`, `to_fid`, `to_ppm`, `to_hz`. They own every convention: FFT centering, physical
time-axis reconstruction, ppm referencing. All automatic conversion inside xmris routes through
them — an inserted transform is bit-identical to one you write yourself. They are deliberately
strict about their input (`to_spectrum` on data without a `time` dim is an error, not a no-op):
that strictness is what makes "accidentally Fourier-transformed twice" impossible.

(domains-explicit)=
### Explicit operations — no magic, by design

`phase` (the low-level primitive under `autophase`) and the raw `fft`/`ifft` family never convert
for you: they are the sharp tools the converters are built from, so they act exactly where you
point them.

(domains-contract-table)=
## What you get, at a glance

| You call            | on a FID (`time`)       | on a spectrum (`frequency`/`chemical_shift`) |
|---------------------|-------------------------|----------------------------------------------|
| `autophase()`       | spectrum (phased)       | spectrum (phased)                            |
| `baseline_als()`    | real spectrum           | real spectrum                                |
| `apodize_exp()`     | FID                     | spectrum                                     |
| `zero_fill()`       | FID (longer)            | spectrum (finer grid)                        |
| `to_spectrum()`     | spectrum                | error — no `time` dim                        |
| `to_fid()`          | error — no spectral dim | FID                                          |
| `fit_amares()`      | fit `Dataset` (time-domain) | fit `Dataset` (spectral)                 |

Spectral outputs keep their input's labeling: ppm in, ppm out.

(domains-pipeline)=
## The canonical pipeline

```python
result = (
    fid.xmr.zero_fill(target_points=4096)   # time-domain home: no transform
       .xmr.apodize_exp(lb=3)               # time-domain home: no transform
       .xmr.autophase()                     # funnel: FID → spectrum, stays
       .xmr.baseline_als()                  # already spectral: no transform
)
```

Exactly **one** Fourier transform executes — at the funnel boundary, where you would have written
`.xmr.to_spectrum()` by hand. Writing the converter explicitly remains equivalent and equally
cheap; the contracts only remove the bookkeeping, not the option.

(domains-guardrails)=
## Guardrails

**One-way data fails loudly.** Downstream of `baseline_als` the spectrum is real-valued and cannot
be taken back to the time domain. Domain-preserving operations check this and raise a clear error
instead of inventing an FID:

```python
spec.xmr.baseline_als().xmr.apodize_exp(lb=2)
# ValueError: Cannot transform real-valued spectral data (dim 'frequency') into
# the time domain: the imaginary component is gone (e.g. discarded by
# `baseline_als`), so no valid FID exists behind this spectrum. …
```

**The ppm leg is metadata-gated.** Converting `chemical_shift` data through the time domain needs
`reference_frequency` and `carrier_ppm` in `attrs`; if they are missing you get the standard
copy-pasteable `assign_attrs` fix, not a wrong axis.

**Explicit foreign dims pass through — for domain-preserving ops.** A domain-preserving
(`@computes_in`) operation called on an axis outside its domain skips conversion entirely:
`kspace.xmr.zero_fill(dim="kx", position="symmetric")` names a different axis, so the data passes
through untouched. Funnel (`@ensures_domain`) operations have no such passthrough — they always
land in their home domain, converting (or raising) even when an explicit foreign `dim` is named.

**Strict mode.** Prefer zero magic — e.g. for quantitative work?

```python
with xmris.set_options(auto_convert=False):
    fid.xmr.autophase()   # ValueError with the explicit fix: .xmr.to_spectrum()
```

(domains-contributors)=
## For contributors: declaring a function's domain

The contracts above are not conventions to remember — they are declared, one line per function,
with two decorators sharing one engine (see the
[validation API](../api/core.validation.md)):

```python
@ensures_domain(SPECTRAL_DIMS)          # funnel: coerce in, leave there
def autophase(da, dim=None, ...): ...

@computes_in(TIME_DIMS)                 # preserve: round trip, restore representation
def apodize_exp(da, dim=DIMS.time, lb=1.0): ...
```

Both decorators also resolve the axis: when a spectral function's `dim` is left as `None`, the
unique spectral dim present (`frequency` or `chemical_shift`) is filled in — an explicitly passed
`dim` is never overridden. That yields the package-wide signature rule, enforced by an architecture
test: *`dim` defaults to `None` **iff** the function is domain-decorated with a multi-label domain;
otherwise it defaults to the config constant.*

```{mermaid}
flowchart TD
    A[New processing function] --> B{Only meaningful in one domain,<br/>result consumed there?}
    B -- yes --> C["@ensures_domain(&lt;DOMAIN&gt;)"]
    B -- no --> D{Same physics seen from<br/>either domain?}
    D -- yes --> E["@computes_in(&lt;DOMAIN&gt;)"]
    D -- no --> F[Undecorated: converter<br/>or primitive — validate<br/>with _check_dims]
```

`fit_amares` is the exception the tree can't draw: it is domain-preserving (the `computes_in`
branch) but returns a `Dataset`, so it hand-rolls the round trip instead of wearing the decorator —
see [domain-preserving operations](#domains-preserving) above.

Two hard rules keep the system honest: decorator-inserted transforms **must route through the
converters** (never inline `fft`/`ifft` — the converters own the conventions), and only converters
may change representation. Everything else follows from the one-sentence goal at the
[top](#domains-goal).
