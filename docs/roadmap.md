(roadmap)=
# Where xmris is going

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-03</span>

::::{div}
:class: roadmap-hero

:::{div}
:class: roadmap-kicker

The direction
:::

:::{div}
:class: roadmap-statement

The same lines that process one spectrum will process a whole spectroscopic-imaging dataset —
and the result can answer for every step that produced it.
:::

:::{div}
:class: roadmap-tenets

`xarray in, xarray out` `docs before code` `the physics travels with the data`
:::
::::

Three commitments shape everything below. Your data stays an `xarray.DataArray` — `.xmr` adds
the physics, and the whole xarray ecosystem keeps working. The docs come before the code: every
page executes on every pull request. And the bookkeeping — coordinates, frequency axes, scanner
metadata — travels with the data.

:::{note} How to read this page
Five bands, descending in confidence: from what already works, through the decision board we
would defend today, to a far end that is more a direction.

The page is aligned with the [tracker's milestones](https://github.com/andrewendlinger/xmris/milestones) and is supposed to give a clearer picture of what's ahead / planned for xmris.
:::

:::{div}
:class: roadmap-map

[Shipped](#roadmap-phase-shipped) [In motion](#roadmap-phase-motion) [Decisions](#roadmap-phase-decisions) [Outward](#roadmap-phase-outward) [Horizon](#roadmap-phase-horizon)
:::

:::::{div}
:class: roadmap-band roadmap-band--shipped

(roadmap-shipped)=
## Shipped <span class="roadmap-ver">on `main` today</span>

::::{div}
:class: roadmap-phase roadmap-phase--shipped
:label: roadmap-phase-shipped

What you can use today.

Real on `main` and exercised on every pull request — every claim on these cards is a live
notebook cell, so a broken one fails the build. (`main` runs ahead of PyPI; closing that gap is
the band below.)
::::

::::{div}
:class: roadmap-item

Spectrum processing that reads like the physics

[`zero_fill`](#zero-fill) → [`apodize_exp`](#apodization) → [`autophase`](#autophase-intro) →
[`baseline_als`](#baseline) → [`to_ppm`](#hz-ppm): each a pure function and an `.xmr` method,
and the FID-to-spectrum conversion [happens automatically](#domain-contracts) where the chain
needs it.
::::

::::{div}
:class: roadmap-item

Plotting, some of it interactive

Config-based [plotting](#plot-basics) from [waterfall](#waterfall) to [carpet](#carpet) plots,
plus live widgets for [phasing](#widget-phase), [apodization](#widget-apodization) and
[scrolling through spectra](#widget-scroller).
::::

::::{div}
:class: roadmap-item

pyAMARES fitting that broadcasts

`fit_amares` fits every spectrum across whatever dimensions you bring — data, model, residual
and per-parameter uncertainties come back aligned on the same axes
([quick start](#fitting-quickstart), [deep dive](#pyamares)).
::::

::::{div}
:class: roadmap-item

An architecture you can hold in your head

Eleven rules — xarray in, xarray out; never mutate the input; the vocabulary is law — written
down as [the contract](contributing/contract.md) and executed against the source on every
build. The why is the [architecture tour](#architecture).
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
:class: roadmap-band roadmap-band--decisions

(roadmap-decisions)=
## The decisions <span class="roadmap-ver">v0.8</span>

::::{div}
:class: roadmap-phase roadmap-phase--decisions
:label: roadmap-phase-decisions

The near horizon: the architecture settles, one argued decision at a time.

Roughly a third of the tracker is blocked behind the questions on this board, and v0.8 is the
last release allowed to move the ground under a user — so each question is decided before its
code is written. The board is ordered by how irreversible each decision is and how soon it
blocks: the full cards are load-bearing, the one-line rows can wait.

Every decision leaves a paper trail. It starts as an *exploration notebook* that demonstrates the
problem live and runs every serious option as code. When one option wins, that notebook freezes
as the record of *why*, an *aimed-solution notebook* details the *what*, and only then does the
implementation follow. A decided card links its trail.
::::

::::{div}
:class: roadmap-item

01 · The license <span class="roadmap-status roadmap-status--arguing">in discussion</span>

xmris is AGPL-3.0-only — the strongest copyleft on PyPI, the quietest gate on who may adopt it,
and a contradiction of the extension ecosystem this page promises further down. Whether that is a
value to state and keep, or an accident to fix while there is exactly one copyright holder, is
under argument now.
::::

::::{div}
:class: roadmap-item roadmap-item--decided

02 · The lineage record <span class="roadmap-status roadmap-status--decided">decided 2026-08-02</span>

Processing lineage leaves flat attrs keys entirely: every operation appends to one `xmr_history`
record — the parameters actually applied, in order, written by one central decorator and read
back as `history()`. Physics and calibration attrs stay flat, typed and individually addressable.
The repeat-application lie — the data carrying one apodization while the record claims another —
dies with the flat keys.

The exploration that got here is frozen as [a design notebook](#attrs-nb); the aimed-solution
notebook comes next, then the implementation.
[#64](https://github.com/andrewendlinger/xmris/issues/64)
::::

::::{div}
:class: roadmap-item roadmap-item--decided

02b · The physical constants <span class="roadmap-status roadmap-status--decided">decided 2026-08-03</span>

The constants a measurement cannot be interpreted without — reference frequency, carrier, group
delay — move out of droppable attrs into one scalar coordinate, `xmr_acquisition`, with the
history riding beside it as `xmr_history`. Surviving plain xarray operations becomes structural
rather than boilerplate; mixed calibrations drop the container whole and fail loudly at the next
gate; and `explain()` renders every constant with its unit and curated description.

Frozen exploration: [the constants design notebook](#constants-nb); the aimed-solution notebook
comes next. [#21](https://github.com/andrewendlinger/xmris/issues/21)
[#22](https://github.com/andrewendlinger/xmris/issues/22)
::::

::::{div}
:class: roadmap-item

03 · A data model written down

Which dimensions, coordinates and attributes make an object an xmris FID or spectrum — and what a
function may assume about one it is handed. The schema is what lets other packages target xmris
rather than guess at it; with 02 and 02b fixing what the record and the constants look like, it
is unblocked and next in line.
[#28](https://github.com/andrewendlinger/xmris/issues/28)
::::

::::{div}
:class: roadmap-item

04 · Core and extras

`[fitting]` exists — does `[plotting]`? do vendors? Install lines are a one-way door: the extras
boundary closes while the ground is still allowed to move, not after.
[#124](https://github.com/andrewendlinger/xmris/issues/124)
::::

::::{div}
:class: roadmap-item

05 · The fit schema, and the fork beneath it

The fit result — one Dataset carrying data, model, residual, parameters and uncertainties — is
the contract a second fitter would slot into, so it gets frozen deliberately rather than by
accident. And the liability underneath gets named: today's fitter rides `pyamares-xmris`, a
self-maintained fork on the critical path, whose exit strategy — upstream, maintain, or replace —
has never been written down.
::::

::::{div}
:class: roadmap-item

06 · The preprocessing middle

The vocabulary declares `average` and `coil`, the Bruker loader emits them, and no function
consumes them: averaging, coil combination and frequency-drift alignment are the unclaimed gap
between load and fit. Does xmris own that middle — the physics-aware, lineage-recording versions,
not the one-line mean — or say honestly that it does not?
::::

::::{div}
:class: roadmap-item

07 · The auto-convert default

Today a domain-mismatched call is converted silently by default, while the option's own docstring
recommends strict mode for quantitative work. Flipping a default after 1.0 is a breaking change,
so it gets decided now, while it is cheap.
::::

::::{div}
:class: roadmap-item

08 · Accessor parity

A free function and its `.xmr` method can drift apart today — signatures, defaults, docstrings.
The fix is mechanical (an introspection test, a registration decorator, or codegen); the decision
is which mechanism, and it retires a whole class of bug at once.
[#62](https://github.com/andrewendlinger/xmris/issues/62)
[#102](https://github.com/andrewendlinger/xmris/issues/102)
::::

::::{div}
:class: roadmap-item

09 · Vendor IO

How much loading xmris owns — "array plus params in, labeled DataArray out", or real file
readers — and whether NIfTI-MRS becomes the common on-ramp instead of N bespoke loaders.
[#125](https://github.com/andrewendlinger/xmris/issues/125)
[#46](https://github.com/andrewendlinger/xmris/issues/46)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

10 · What a vocabulary term *is* — the representation behind the strings
[#65](https://github.com/andrewendlinger/xmris/issues/65)
[#88](https://github.com/andrewendlinger/xmris/issues/88)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

11 · Where tests live — the pytest/notebook boundary, and coverage that discovers instead of
listing [#66](https://github.com/andrewendlinger/xmris/issues/66)
[#107](https://github.com/andrewendlinger/xmris/issues/107)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

12 · The plug-in promise — a platform, or extensibility by documented convention?
[#124](https://github.com/andrewendlinger/xmris/issues/124)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

13 · The Python ceiling — ≤3.13 for everyone, for the sake of an optional extra
::::

::::{div}
:class: roadmap-item roadmap-item--minor

14 · The typing promise — `py.typed` ships with no checker behind it
::::

::::{div}
:class: roadmap-item roadmap-item--minor

15 · The MRI non-goal, said exactly — MRSI yes; reconstruction, where is the line?
::::

:::::

:::::{div}
:class: roadmap-band roadmap-band--outward

(roadmap-outward)=
## Then outward <span class="roadmap-ver">v0.8 – v0.9</span>

::::{div}
:class: roadmap-phase roadmap-phase--outward
:label: roadmap-phase-outward

Committed, but not built.

v0.8 lands the code the decision board unblocks — roughly a third of the tracker waits there —
and the decisions resolve together in one release rather than leaking across several.

v0.9 then turns to everything a stranger needs that today requires reading the source, or knowing
the author. Its exit criterion is the JOSS submission.
::::

::::{div}
:class: roadmap-item roadmap-item--minor

A full review-and-simplify pass runs first, so its findings feed the decisions still open above
[#127](https://github.com/andrewendlinger/xmris/issues/127)
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
[#46](https://github.com/andrewendlinger/xmris/issues/46), a sidebar that separates hands-on
tutorials from concept explainers [#126](https://github.com/andrewendlinger/xmris/issues/126), a
README whose quick start actually runs, a public API surface with no unreachable corners, and the
correctness backlog in fitting and plotting paid down.
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

This is what xmris is being built *toward* — held loosely on purpose. These may arrive in a
different form, in a different order, or not at all; the rail fades here because the confidence
does, and the decision board above is allowed to rewrite any of it.

They are on the page anyway, because a roadmap that only lists the safe things is not telling you
where it is going. If your work depends on one of them, say so in the tracker: that is the main
way something moves up a band.
::::

::::{div}
:class: roadmap-item

`v1.0` — the point the contract stops moving

Not a feature: a promise — and deliberately not a milestone in the tracker yet, because creating
one now would fake a certainty nobody has. What has to be true first: the decision board above
emptied, the contract stable across two consecutive releases, the stranger's first hour solved,
and a written deprecation policy. When those hold, calling it 1.0 is a formality; until then,
calling it 1.0 would be a lie with a version number.
::::

::::{div}
:class: roadmap-item

MRSI: space and scale

The vocabulary already declares `x`, `y`, `z` and their k-space twins, and nothing uses them yet —
a placed bet, not dead weight. Cashing it in, in order: CSI-shaped data, spatial plus spectral,
through the same three lines that process one FID; per-voxel initialisation for fits across a
grid; lazy, chunked processing for volumes that outgrow memory
[#25](https://github.com/andrewendlinger/xmris/issues/25); and image coordinates, so a fitted
metabolite map can sit on the anatomical image it came from
[#4](https://github.com/andrewendlinger/xmris/issues/4).
::::

::::{div}
:class: roadmap-item

An xmris someone else can extend

Whether this is promised at all is decision 12 on the board: a plug-in platform is a
compatibility contract with parties who do not exist yet, and a written data model (decision 03)
plus documented conventions may serve an outside lab better than an API. If the promise survives,
it looks like a lab in another MR domain publishing its own functions and vocabulary as an
extension, vendor loaders that ship on their own cadence, and heavy capabilities that never weigh
down the core install.
[#124](https://github.com/andrewendlinger/xmris/issues/124)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

More vendors than Bruker — Siemens, GE, Philips, NIfTI-MRS; how much xmris owns is decision 09
[#125](https://github.com/andrewendlinger/xmris/issues/125)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Core xmris will not do image reconstruction or quantitative-MRI parameter mapping — the exact
wording of that refusal, and the fate of the unused k-space vocabulary, is decision 15
::::

:::::

(roadmap-landscape)=
## The issue landscape

<span style="color: gray; font-size: 0.9em;">A snapshot taken 2026-08-03 · the
[milestones](https://github.com/andrewendlinger/xmris/milestones) are the live source</span>

Forty-one issues are open, and their distribution is the argument for the release line above: the
largest cluster is not missing features — it is unmade decisions, plus the work standing behind
them. Two of those decisions are now made (02 and 02b on the board); their tracker issues stay
open until the implementations land, with
[#21](https://github.com/andrewendlinger/xmris/issues/21) and
[#22](https://github.com/andrewendlinger/xmris/issues/22) re-scoped by what was decided.

| Cluster | Issues | Lands in |
|---|---|---|
| Design decisions — the board above | [#62](https://github.com/andrewendlinger/xmris/issues/62) [#64](https://github.com/andrewendlinger/xmris/issues/64) [#65](https://github.com/andrewendlinger/xmris/issues/65) [#66](https://github.com/andrewendlinger/xmris/issues/66) [#88](https://github.com/andrewendlinger/xmris/issues/88) [#113](https://github.com/andrewendlinger/xmris/issues/113) [#124](https://github.com/andrewendlinger/xmris/issues/124) [#125](https://github.com/andrewendlinger/xmris/issues/125) | v0.8 |
| Blocked behind them | [#21](https://github.com/andrewendlinger/xmris/issues/21) [#22](https://github.com/andrewendlinger/xmris/issues/22) [#23](https://github.com/andrewendlinger/xmris/issues/23) [#28](https://github.com/andrewendlinger/xmris/issues/28) [#34](https://github.com/andrewendlinger/xmris/issues/34) [#71](https://github.com/andrewendlinger/xmris/issues/71) [#102](https://github.com/andrewendlinger/xmris/issues/102) [#107](https://github.com/andrewendlinger/xmris/issues/107) | v0.8 · the schema #28 in v0.9 |
| The tag itself | [#10](https://github.com/andrewendlinger/xmris/issues/10) [#115](https://github.com/andrewendlinger/xmris/issues/115) [#116](https://github.com/andrewendlinger/xmris/issues/116) [#122](https://github.com/andrewendlinger/xmris/issues/122) | v0.7 |
| Quality & tooling | [#87](https://github.com/andrewendlinger/xmris/issues/87) [#108](https://github.com/andrewendlinger/xmris/issues/108) [#111](https://github.com/andrewendlinger/xmris/issues/111) [#117](https://github.com/andrewendlinger/xmris/issues/117) [#127](https://github.com/andrewendlinger/xmris/issues/127) | v0.8 – v0.9 |
| The front door | [#27](https://github.com/andrewendlinger/xmris/issues/27) [#46](https://github.com/andrewendlinger/xmris/issues/46) [#67](https://github.com/andrewendlinger/xmris/issues/67) [#119](https://github.com/andrewendlinger/xmris/issues/119) [#120](https://github.com/andrewendlinger/xmris/issues/120) [#121](https://github.com/andrewendlinger/xmris/issues/121) [#126](https://github.com/andrewendlinger/xmris/issues/126) | v0.9 |
| Correctness & capability | [#29](https://github.com/andrewendlinger/xmris/issues/29) [#31](https://github.com/andrewendlinger/xmris/issues/31) [#80](https://github.com/andrewendlinger/xmris/issues/80) [#82](https://github.com/andrewendlinger/xmris/issues/82) [#83](https://github.com/andrewendlinger/xmris/issues/83) [#84](https://github.com/andrewendlinger/xmris/issues/84) [#128](https://github.com/andrewendlinger/xmris/issues/128) | v0.9 |
| Space & scale | [#4](https://github.com/andrewendlinger/xmris/issues/4) [#25](https://github.com/andrewendlinger/xmris/issues/25) | Horizon |

The decisions are not a flat list. The spine below is the board's dependency order — the decided
pair feeds the data model first, and the license decision quietly gates the plug-in promise:

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart TD
    d02["02 lineage record ✓"] --> d03["03 data model"]
    d02b["02b constants ✓"] --> d03
    d06["06 preprocessing middle"] --> d03
    d06 --> d15["15 MRI non-goal"]
    d01["01 license"] --> d12["12 plug-in promise"]
    d04["04 core & extras"] <--> d12
    d04 --> d09["09 vendor IO"]
    d08["08 accessor parity"] --> d11["11 test architecture"]
```

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

The decision board keeps its paper on this site: each decided card links an exploration notebook,
frozen the day the decision lands, and an aimed-solution notebook that becomes the spec the
implementation is checked against. Both live in the sidebar beneath this page.

:::{seealso}
The [dev diary](diary/about.md) is this page's backward-looking twin: one entry per decision
already taken, rewritten in place as that decision evolves.
:::
