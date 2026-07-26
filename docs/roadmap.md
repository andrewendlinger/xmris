(roadmap)=
# Where xmris is going

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-07-26 · v0.6.1 · visual draft — content not final</span>

::::{div}
:class: roadmap-hero

:::{div}
:class: roadmap-kicker

The vision
:::

:::{div}
:class: roadmap-statement

xmris is finished when the three lines you write to process a single FID are the same three lines
that process a whole volume — and when the object they hand back already carries everything a paper
needs.
:::

:::{div}
:class: roadmap-tenets

`xarray in, xarray out` `the notebook is the test` `the record travels with the data`
:::
::::

Today the package is honest about what it is. The processing chain genuinely broadcasts: zero-fill,
apodize, Fourier, phase and reference all run across a coil axis, a repetition axis or a voxel grid
without knowing they exist. The fitting does not — `fit_amares` is one-dimensional, and five
documentation pages hand-roll the same `xr.concat` stack to work around it. That gap between the
two halves is the shape of most of the near-term work.

The destination is that the dimension you did not think about is the one that costs nothing. And
that the record travels with the data: the reference frequency, the phase that was applied, the
prior knowledge a fit was given, the residual it produced — all attached to the object you already
hold, so a figure and the methods section describing it come out of the same file.

The last thing on this page is a promise rather than a feature. `v1.0` is the point at which the
[architecture contract](contributing/contract.md) stops moving and the API you wrote against stays
where it is.

:::{note} How to read this page
Four bands, from what already works to what is still an intention — and the spine beside them loses
its confidence as it descends. There are deliberately **no dates**. A milestone moves up a band when
the work is real, not when a quarter ends. Version numbers are the release a milestone is aimed at,
not a commitment to ship it there.

The issue tracker holds the detail; this page holds the argument.
:::

:::{div}
:class: roadmap-map

[Shipped](#roadmap-phase-shipped) [In motion](#roadmap-phase-motion) [Next](#roadmap-phase-next) [Horizon](#roadmap-phase-horizon)
:::

:::::{div}
:class: roadmap-band roadmap-band--shipped

(roadmap-shipped)=
## Shipped <span class="roadmap-ver">v0.1 – v0.6</span>

::::{div}
:class: roadmap-phase roadmap-phase--shipped
:label: roadmap-phase-shipped

What you can use today.

Released to PyPI, documented, and exercised by the test suite — the claims on these cards are live
notebook cells, so a broken one fails the build rather than quietly ageing.

Nothing here is a plan. If something in this band does not work, that is a bug worth reporting, not
a roadmap item waiting its turn.
::::

::::{div}
:class: roadmap-item

An architecture you can hold in your head

Eleven numbered rules — xarray in, xarray out; never mutate the input; the vocabulary is law — are
written down as [the architecture contract](contributing/contract.md) and *executed* against the
source on every build. The page quotes the live code it governs, so the contract cannot drift away
from it. [#72](https://github.com/andrewendlinger/xmris/issues/72)
::::

::::{div}
:class: roadmap-item

A processing chain that reads like the physics

`zero_fill` → `apodize_exp` → `to_spectrum` → `autophase` → `baseline_als` → `to_ppm`, each a pure
function and an `.xmr` method, each preserving the coordinates it was given and appending exactly
the parameter it applied. Domain mistakes are caught at the door by `@computes_in` and
`@ensures_domain` rather than in your results.
::::

::::{div}
:class: roadmap-item

Fitting, through pyAMARES

`fit_amares` returns a Dataset with the data, the model and the residual aligned on the same axes,
so the fit is an object you can keep processing rather than a table you have to reassemble.
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Interactive widgets for phasing, apodization and scrolling through spectra
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Documentation that executes itself — every claim is a live cell, every page previewed per pull
request [#104](https://github.com/andrewendlinger/xmris/issues/104)
::::

:::::

:::::{div}
:class: roadmap-band roadmap-band--motion

(roadmap-in-motion)=
## In motion <span class="roadmap-ver">v0.7</span>

::::{div}
:class: roadmap-phase roadmap-phase--motion
:label: roadmap-phase-motion

Being built right now.

The linked issues are open and have commits against them. This is the last point at which the shape
of an API is still cheap to change.

So it is the band worth arguing with. If you have a view on how a failed fit should announce itself,
or what a result Dataset ought to be called, saying so now costs nothing — saying so after v1.0
costs a deprecation cycle.
::::

::::{div}
:class: roadmap-item

Fitting you can trust

A fit that fails to converge can currently return quietly, and the Dataset it produces uses literal
keys instead of the controlled vocabulary. This is the work that makes fitting say *why* it stopped,
validate prior knowledge before it reaches the solver, and come back speaking the same language as
everything else.
[#80](https://github.com/andrewendlinger/xmris/issues/80)
[#81](https://github.com/andrewendlinger/xmris/issues/81)
[#82](https://github.com/andrewendlinger/xmris/issues/82)
[#69](https://github.com/andrewendlinger/xmris/issues/69)

:::{dropdown} What each piece covers
- **[#80](https://github.com/andrewendlinger/xmris/issues/80)** — silent-failure and convergence
  hardening: a fit reports its exit condition instead of returning a plausible-looking result.
- **[#81](https://github.com/andrewendlinger/xmris/issues/81)** — the API and docstring rough edges
  around `fit_amares`.
- **[#82](https://github.com/andrewendlinger/xmris/issues/82)** — the prior-knowledge file format
  has traps that are currently discovered at runtime; validate and document them.
- **[#69](https://github.com/andrewendlinger/xmris/issues/69)** — the result Dataset ignores
  `VARS`/`DIMS`, which breaks the one-vocabulary rule at exactly the point users look hardest.
:::
::::

::::{div}
:class: roadmap-item

One canonical API

A free function and its `.xmr` method should never disagree — but today only the `dim` defaults are
pinned by a test, and the accessor is registered as an import side-effect nobody chose. This is the
audit, the enforcement, and the decision underneath both.
[#71](https://github.com/andrewendlinger/xmris/issues/71)
[#102](https://github.com/andrewendlinger/xmris/issues/102)
[#62](https://github.com/andrewendlinger/xmris/issues/62)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Install without a git fork — pyAMARES becomes an optional extra
[#70](https://github.com/andrewendlinger/xmris/issues/70)
[#115](https://github.com/andrewendlinger/xmris/issues/115)
::::

:::::

:::::{div}
:class: roadmap-band roadmap-band--next

(roadmap-next)=
## Next <span class="roadmap-ver">v0.8 – v0.9</span>

::::{div}
:class: roadmap-phase roadmap-phase--next
:label: roadmap-phase-next

Decided, but not yet built.

Each of these has a design behind it — usually an open `design-decision` issue that has been argued
through — and no implementation. The version stamp is where it is aimed, not where it is promised.

A milestone leaves this band by being built, or by being dropped when the decision underneath it
turns out to be wrong. Both happen, and the second is not a failure.
::::

::::{div}
:class: roadmap-item

Provenance that survives a chain

The applied parameter is the record — `phase_p0=15.0`, never `phase_applied=True` — but attributes
survive a long chain by convention rather than by guarantee. The open decision is whether lineage
stays flat per-parameter keys or becomes a structured history log; what it unlocks is a spectrum
that can reconstruct its own methods section.
[#64](https://github.com/andrewendlinger/xmris/issues/64)
[#21](https://github.com/andrewendlinger/xmris/issues/21)
[#23](https://github.com/andrewendlinger/xmris/issues/23)
::::

::::{div}
:class: roadmap-item

A data model written down

The vocabulary is law, but the *schema* it forms has never been stated: which dimensions and
attributes an object must carry to be an xmris FID, and what a function may assume about one it is
handed. Writing that down is what lets other packages target xmris rather than guess at it.
[#28](https://github.com/andrewendlinger/xmris/issues/28)
[#65](https://github.com/andrewendlinger/xmris/issues/65)
[#88](https://github.com/andrewendlinger/xmris/issues/88)
::::

::::{div}
:class: roadmap-item

Lazy and chunked

dask support, so a dataset larger than memory is still the same three chained lines.
[#25](https://github.com/andrewendlinger/xmris/issues/25)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Fitting beyond one dimension — the `xr.concat` workaround becomes the library's job
[#113](https://github.com/andrewendlinger/xmris/issues/113)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

A changelog [#10](https://github.com/andrewendlinger/xmris/issues/10)
::::

:::::

:::::{div}
:class: roadmap-band roadmap-band--horizon

(roadmap-horizon)=
## Horizon <span class="roadmap-ver">v1.0 and past it</span>

::::{div}
:class: roadmap-phase roadmap-phase--horizon
:label: roadmap-phase-horizon

Direction, not commitment.

This is what xmris is being built *toward*. These may arrive in a different form, in a different
order, or not at all — the rail fades here because the confidence does.

They are on the page anyway, because a roadmap that only lists the safe things is not telling you
where it is going. If your work depends on one of them, say so in the tracker: that is the main way
something moves up a band.
::::

::::{div}
:class: roadmap-item

`v1.0` — the point the contract stops moving

Not a feature: a promise. The epic that gates it is public-release readiness — a JOSS paper with a
State of the Field against FSL-MRS, Osprey, suspect and spant; a citation file; a typing gate;
fitting coverage worth the name; a clean install from PyPI with no git sources. After it, the API
you wrote against stays where it is.
[#67](https://github.com/andrewendlinger/xmris/issues/67)
::::

::::{div}
:class: roadmap-item

The visualization ecosystem: `.plot`, `.widget`, `.ui`

Three lazily-loaded pillars, of which two exist in part. The third does not: a napari bridge for
scrolling a four-dimensional spectral volume and overlaying fitted metabolite maps on the anatomy
they came from.
[#4](https://github.com/andrewendlinger/xmris/issues/4)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

More vendors than Bruker [#46](https://github.com/andrewendlinger/xmris/issues/46)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Imaging, and not only spectroscopy — the `I` in the name is still aspirational
::::

:::::

:::{div}
:class: roadmap-end

and then it is someone else's turn to say what comes next
:::

(roadmap-changing)=
## How this page changes

This page is written by hand, and it is the argument rather than the record: the issue tracker holds
what is actually being worked on, and it moves faster. A milestone here earns its band from the
state of the work, so if the two disagree, the tracker is right and this page is stale — say so by
[opening an issue](https://github.com/andrewendlinger/xmris/issues/new).

:::{seealso}
The [dev diary](diary/about.md) is this page's backward-looking twin: one entry per decision already
taken, rewritten in place as that decision evolves.
:::
