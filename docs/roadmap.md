(roadmap)=
# Where xmris is going

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-07-26</span>

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

That sentence is closer to true than it looks. Your data stays an `xarray.DataArray` — `.xmr` is an
accessor, not a container — so the processing chain broadcasts by construction: zero-fill, apodize,
Fourier, phase and reference run across a coil axis, a repetition series or a voxel grid without
being told they exist. The fitting keeps up: `fit_amares` takes whatever dimensions you hand it,
fits every spectrum in parallel, and folds the results back into your layout. The honest remainder
is small and specific — the simulator builds one FID at a time, automatic phasing picks a single
phase for the whole array, a fit across a grid still normalises and initialises globally rather
than per voxel, and past a certain size, memory. That remainder is the far end of this page.

The near end is less glamorous and more important. Roughly a third of the open tracker sits behind
five unmade architecture decisions, and the version on PyPI does not install the package this
documentation describes. So the next two releases are not feature releases: one makes the install
honest, and one makes the ground stop moving. Features come after the promises hold.

The last thing on this page is a promise rather than a feature. `v1.0` is not a milestone with an
item list; it is the point at which the [architecture contract](contributing/contract.md) stops
moving and the API you wrote against stays where it is. What has to be true first is written in
the [horizon band](#roadmap-horizon).

:::{note} How to read this page
Four bands, from what already works to what is still an intention — and the spine beside them loses
its confidence as it descends. There are deliberately **no dates**: this is a one-maintainer
project beside a PhD, so order is real information and a schedule would be fiction. A milestone
moves up a band when the work is real, not when a quarter ends.

The [tracker's milestones](https://github.com/andrewendlinger/xmris/milestones) hold the item
lists and move faster; this page holds the argument. Version numbers are the release a milestone is
aimed at, not a commitment to ship it there.
:::

:::{div}
:class: roadmap-map

[Shipped](#roadmap-phase-shipped) [In motion](#roadmap-phase-motion) [Next](#roadmap-phase-next) [Horizon](#roadmap-phase-horizon)
:::

:::::{div}
:class: roadmap-band roadmap-band--shipped

(roadmap-shipped)=
## Shipped <span class="roadmap-ver">on `main` today</span>

::::{div}
:class: roadmap-phase roadmap-phase--shipped
:label: roadmap-phase-shipped

What you can use today.

Real on `main`, documented, and exercised on every pull request — the claims on these cards are
live notebook cells, so a broken one fails the build rather than quietly ageing.

One honesty note: `main` is ahead of PyPI, and the wheel PyPI serves carries the old, broken
packaging — closing that gap is the entire point of the band below. Nothing here is a plan. If
something in this band does not work, that is a bug worth reporting, not a roadmap item waiting
its turn.
::::

::::{div}
:class: roadmap-item

An architecture you can hold in your head

Eleven numbered rules — xarray in, xarray out; never mutate the input; the vocabulary is law — are
written down as [the architecture contract](contributing/contract.md) and *executed* against the
source on every build. The page quotes the live code it governs, so the contract cannot drift away
from it.
::::

::::{div}
:class: roadmap-item

A processing chain that reads like the physics

`zero_fill` → `apodize_exp` → `to_spectrum` → `autophase` → `baseline_als` → `to_ppm`, each a pure
function and an `.xmr` method, each preserving the coordinates it was given and appending exactly
the parameter it applied. Domain mistakes are caught at the door by `@computes_in` and
`@ensures_domain`: a function handed data in the wrong domain refuses, instead of computing
something plausible and wrong.
::::

::::{div}
:class: roadmap-item

Fitting that broadcasts

`fit_amares` meets your data in either domain, stacks whatever extra dimensions it finds, fits
every spectrum in parallel, and hands back a Dataset with the data, the model and the residual
aligned on the same axes — with per-parameter uncertainties, and a per-voxel `fit_status` instead
of a silent zero where a fit fails.
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Interactive widgets for phasing, apodization and scrolling through spectra
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Documentation that executes itself — every claim is a live cell, every page previewed per pull
request
::::

:::::

:::::{div}
:class: roadmap-band roadmap-band--motion

(roadmap-in-motion)=
## In motion <span class="roadmap-ver">v0.7</span>

::::{div}
:class: roadmap-phase roadmap-phase--motion
:label: roadmap-phase-motion

Being released right now.

Everything in this band is merged and working on `main`; what is in motion is the release itself.
v0.7 is the tag that makes the shipped band installable — months of fitting, contract and
documentation work that PyPI has not seen yet.

It is deliberately small. The only code it waits for is the install fix below, because tagging a
release whose bare `pip install` cannot even be imported would defeat the point of tagging at all.
::::

::::{div}
:class: roadmap-item

Fitting you can trust

The rework that headlines this release. Amplitude scale is normalised before fitting, killing a
failure mode where a fit could "converge" on its own prior and look completely fine; a failed
voxel comes back as `NaN` plus a `fit_status` that says why, never a plausible number; prior
knowledge is validated before it reaches the solver instead of failing inside it.
::::

::::{div}
:class: roadmap-item

An install that tells the truth

The wheel on PyPI cannot install on Apple Silicon, and today's `main` fixed that while leaving a
bare `pip install xmris` unimportable — a missing core dependency that CI never caught, because no
job installs the package the way a user does. v0.7 ships an install that works everywhere, AMARES
as the optional `xmris[fitting]` extra, and a CI job that installs like a stranger so this class
of bug cannot return. [#122](https://github.com/andrewendlinger/xmris/issues/122)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

A changelog begins — its first entry is this release
[#10](https://github.com/andrewendlinger/xmris/issues/10)
::::

:::::

:::::{div}
:class: roadmap-band roadmap-band--next

(roadmap-next)=
## Next <span class="roadmap-ver">v0.8 – v0.9</span>

::::{div}
:class: roadmap-phase roadmap-phase--next
:label: roadmap-phase-next

Committed, but not built.

v0.8 is one thing said five ways: the architecture settles. Five design decisions are open, each
an issue where the choice is argued before any code follows, and roughly a third of the tracker is
blocked behind them. v0.8 is the last release allowed to move the ground under a user — and saying
that publicly is most of the value of having a roadmap at all.

v0.9 then turns outward: everything a stranger needs that today requires reading the source, or
knowing the author. Its exit criterion is the JOSS submission.
::::

::::{div}
:class: roadmap-item

Five decisions, then the code that follows

The questions are argued in their issues, not here — what this page fixes is the order, and that
they resolve together in one release rather than leaking across several.

:::{dropdown} The five, in dependency order
- **[#65](https://github.com/andrewendlinger/xmris/issues/65)** — what a vocabulary term *is*
  (today: a `str` subclass carrying metadata), then
  **[#88](https://github.com/andrewendlinger/xmris/issues/88)** — where diagnostic and
  algorithm-output axes live. The cheapest pair, so it goes first.
- **[#64](https://github.com/andrewendlinger/xmris/issues/64)** — the attrs strategy: does lineage
  stay flat applied-parameter keys, or become a structured history? The highest-leverage decision
  on the board — four issues wait on it, Commandment 3 is rewritten by whatever it decides, and it
  is what turns "the record travels with the data" from a convention into a guarantee: a spectrum
  that can reconstruct its own methods section.
- **[#62](https://github.com/andrewendlinger/xmris/issues/62)** — accessor auto-registration and
  which API is canonical, so a free function and its `.xmr` method cannot drift apart again.
- **[#66](https://github.com/andrewendlinger/xmris/issues/66)** — the boundary between pytest and
  notebook tests, which decides how the architecture suite gets rebuilt to discover functions
  instead of trusting hand-maintained lists.
:::
::::

::::{div}
:class: roadmap-item roadmap-item--minor

`simulate_fid` learns to stack — the `xr.concat` boilerplate five pages currently teach becomes
one argument [#113](https://github.com/andrewendlinger/xmris/issues/113)
::::

::::{div}
:class: roadmap-item

Ready for a stranger

The first hour of a new user, made survivable without reading the source: a documented path from
raw vendor data or a bare numpy array into an xmris-ready object
[#46](https://github.com/andrewendlinger/xmris/issues/46), a README whose quick start actually
runs, a public API surface with no unreachable corners, and the correctness backlog in fitting and
plotting paid down.
::::

::::{div}
:class: roadmap-item

A data model written down

The vocabulary is law inside the library, but the *schema* it forms has never been stated: which
dimensions and attributes an object must carry to count as an xmris FID or spectrum, and what a
function may assume about one it is handed. Writing that down — after the attrs decision fixes
what the record looks like — is what lets other packages target xmris rather than guess at it.
[#28](https://github.com/andrewendlinger/xmris/issues/28)
::::

::::{div}
:class: roadmap-item

A JOSS paper

The submission is v0.9's definition of done — a deadline chosen, not imposed. The epic that gates
it is public-release readiness: a citation file, community files, a typing gate, fitting coverage
worth the name, and a State of the Field against FSL-MRS, Osprey, suspect and spant. JOSS happily
cites 0.x software, so *citable* does not wait for *frozen*.
[#67](https://github.com/andrewendlinger/xmris/issues/67)
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

Not a feature: a promise — and deliberately not a milestone in the tracker yet, because creating
one now would fake a certainty nobody has. What has to be true first: the five decisions shipped
and the contract stable across two consecutive releases; the stranger's first hour solved; a
written deprecation policy. When those hold, calling it 1.0 is a formality; until then, calling it
1.0 would be a lie with a version number.
::::

::::{div}
:class: roadmap-item

MRSI: space and scale

The vocabulary already declares `x`, `y`, `z` and their k-space twins, and nothing uses them yet —
a placed bet, not dead weight. Cashing it in, in order: CSI-shaped data, spatial plus spectral,
through the same three lines that process one FID; per-voxel initialisation and normalisation for
fits across a grid; lazy, chunked processing for volumes that outgrow memory
[#25](https://github.com/andrewendlinger/xmris/issues/25); and image coordinates, so a fitted
metabolite map can sit on the anatomical image it came from
[#4](https://github.com/andrewendlinger/xmris/issues/4).
::::

::::{div}
:class: roadmap-item roadmap-item--minor

More vendors than Bruker — Siemens, GE, Philips, NIfTI-MRS
::::

::::{div}
:class: roadmap-item roadmap-item--minor

What xmris is *not*: image reconstruction and quantitative-MRI parameter mapping stay out of
scope — the *i* in the name is the spectroscopic imaging above, not an MRI toolbox
::::

:::::

:::{div}
:class: roadmap-end

and then it is someone else's turn to say what comes next
:::

(roadmap-changing)=
## How this page changes

This page is written by hand, and it is the argument rather than the record: the
[milestones](https://github.com/andrewendlinger/xmris/milestones) hold what is actually in each
release, and they move faster. A milestone here earns its band from the state of the work, so if
the two disagree, the tracker is right and this page is stale — say so by
[opening an issue](https://github.com/andrewendlinger/xmris/issues/new).

:::{seealso}
The [dev diary](diary/about.md) is this page's backward-looking twin: one entry per decision already
taken, rewritten in place as that decision evolves.
:::
