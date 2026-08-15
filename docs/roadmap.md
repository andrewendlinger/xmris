(roadmap)=
# Where xmris is going

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-10</span>

::::{div}
:class: roadmap-hero

:::{div}
:class: roadmap-kicker

The vision
:::

:::{div}
:class: roadmap-statement

xmris is finished when the three lines you write to process a single FID are the same three lines
that process a whole volume — and when the object they hand back already carries everything you
need.
:::

:::{div}
:class: roadmap-tenets

`xarray in, xarray out` `docs before code` `the record travels with the data`
:::
::::

Three commitments shape everything below. Your data stays an `xarray.DataArray` — the physics
comes to it, never your data into a framework, and the whole xarray ecosystem keeps working. The
docs come before the code: every page executes on every pull request. And the record travels with
the data — the reference frequency, the phase that was applied, the prior knowledge a fit was
given, attached to the object you already hold.

:::{note} How to read this page
Five bands: from what already works, through the upcoming architecture decisions, to the near
and far future — kept in sync with the
[issues and milestones](https://github.com/andrewendlinger/xmris/milestones). Like any plan, it
will probably change when it meets reality.
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

Real on `main` and exercised on every pull request — though `main` still runs ahead of PyPI.
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
down as [the contract](#contract) and executed against the source on every
build. The why is the [architecture tour](#architecture).
::::

:::::

:::::{div}
:class: roadmap-band roadmap-band--motion

(roadmap-in-motion)=
## In motion <span class="roadmap-ver"><a href="https://github.com/andrewendlinger/xmris/milestone/1">v0.7</a></span>

::::{div}
:class: roadmap-phase roadmap-phase--motion
:label: roadmap-phase-motion

Moving right now.

v0.7 is the tag that finally makes the band above installable.
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Next in line: the docs get a floor plan — a sidebar you can scan, concepts apart from tutorials,
and a landing page that says where everything lives
[#137](https://github.com/andrewendlinger/xmris/issues/137)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

This very page — the roadmap you are reading is being written and argued right now
[#116](https://github.com/andrewendlinger/xmris/issues/116)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

[A changelog](#changelog) begins — its first entry is this release
[#10](https://github.com/andrewendlinger/xmris/issues/10)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

The docs stop recommending strict mode — automatic domain conversion is the default, deliberately
and permanently [#131](https://github.com/andrewendlinger/xmris/issues/131)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Bug fix: a fit could silently return the prior it was given and look completely fine — the rework
that kills it is merged; the last guards are tracked
[#80](https://github.com/andrewendlinger/xmris/issues/80)
[#82](https://github.com/andrewendlinger/xmris/issues/82)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

Bug fix: the wheel on PyPI cannot install on Apple Silicon, and a bare `pip install xmris` cannot
even be imported [#122](https://github.com/andrewendlinger/xmris/issues/122)
::::

:::::

:::::{div}
:class: roadmap-band roadmap-band--decisions

(roadmap-decisions)=
## The decisions <span class="roadmap-ver"><a href="https://github.com/andrewendlinger/xmris/milestone/2">v0.8</a></span>

::::{div}
:class: roadmap-phase roadmap-phase--decisions
:label: roadmap-phase-decisions

The near horizon: the architecture settles, one argued decision at a time.

v0.8 is the last release allowed to move the ground under a user, so every question here is
decided in writing before its code is written.
::::

::::{div}
:class: roadmap-item roadmap-item--decided

02 · The lineage record <span class="roadmap-status roadmap-status--decided">decided 2026-08-02</span>

Processing history becomes one `xmr_history` record — what each operation actually applied, in
order, so the result can answer for every step that produced it. Physics attrs stay flat and
typed.

<span class="roadmap-progress">[explored](#attrs-nb) ✓ → decided ✓ → <strong>next: solution spec</strong> → implementation</span>
[#64](https://github.com/andrewendlinger/xmris/issues/64)
::::

::::{div}
:class: roadmap-item roadmap-item--decided

02b · The physical constants <span class="roadmap-status roadmap-status--decided">decided 2026-08-03</span>

The constants a measurement cannot be interpreted without travel as one container coordinate,
`xmr_acquisition` — they survive plain xarray operations structurally, not by boilerplate.

<span class="roadmap-progress">[explored](#constants-nb) ✓ → decided ✓ → <strong>next: solution spec</strong> → implementation</span>
[#21](https://github.com/andrewendlinger/xmris/issues/21)
[#22](https://github.com/andrewendlinger/xmris/issues/22)
::::

::::{div}
:class: roadmap-item

03 · A data model written down

Which dimensions, coordinates and attributes make an object an xmris FID or spectrum — the
schema other packages target instead of guessing at. Chapter two is the fit-result Dataset.
Unblocked by 02 and 02b, next in line.
[#28](https://github.com/andrewendlinger/xmris/issues/28)
::::

::::{div}
:class: roadmap-item

04 · Core and extras

`[fitting]` exists — does `[plotting]`? do vendors? Install lines are a one-way door: the
boundary closes while the ground is still allowed to move.
[#124](https://github.com/andrewendlinger/xmris/issues/124)
::::

::::{div}
:class: roadmap-item

08 · Accessor parity

A free function and its `.xmr` method can drift apart — signatures, defaults, docstrings. The
decision is the mechanism that makes drift impossible.
[#62](https://github.com/andrewendlinger/xmris/issues/62)
[#102](https://github.com/andrewendlinger/xmris/issues/102)
::::

::::{div}
:class: roadmap-item

09 · Vendor IO

How much loading xmris owns — and whether NIfTI-MRS becomes the common on-ramp instead of N
bespoke loaders.
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
[#133](https://github.com/andrewendlinger/xmris/issues/133)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

14 · The typing promise — `py.typed` ships with no checker behind it
[#67](https://github.com/andrewendlinger/xmris/issues/67)
::::

::::{div}
:class: roadmap-item roadmap-item--minor

15 · The MRI non-goal, said exactly — MRSI yes; reconstruction, where is the line?
[#136](https://github.com/andrewendlinger/xmris/issues/136)
::::

:::::

:::::{div}
:class: roadmap-band roadmap-band--outward

(roadmap-outward)=
## Then outward <span class="roadmap-ver"><a href="https://github.com/andrewendlinger/xmris/milestone/2">v0.8</a> – <a href="https://github.com/andrewendlinger/xmris/milestone/3">v0.9</a></span>

::::{div}
:class: roadmap-phase roadmap-phase--outward
:label: roadmap-phase-outward

Committed, but not built.

v0.8 lands the code the decision board unblocks; v0.9 turns to what a stranger needs, and the
JOSS submission is its definition of done.
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

The preprocessing middle gets claimed

The gap between load and fit — averaging that aligns before it means, coil combination,
frequency-drift correction — becomes xmris code: the physics-aware, lineage-recording versions,
not the one-line mean. The vocabulary's `average` and `coil` dimensions finally get their
consumers; drafts exist and will be ported.
[#132](https://github.com/andrewendlinger/xmris/issues/132)
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

Where xmris is headed, held loosely — these may arrive in a different form, in a different order,
or not at all.
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

<span style="color: gray; font-size: 0.9em;">A snapshot taken 2026-08-05 · the
[milestones](https://github.com/andrewendlinger/xmris/milestones) are the live source</span>

Forty-seven issues are open, and their distribution is the argument for the release line above: the
largest cluster is not missing features — it is unmade decisions, plus the work standing behind
them. Two of those decisions are now made (02 and 02b on the board); their tracker issues stay
open until the implementations land, with
[#21](https://github.com/andrewendlinger/xmris/issues/21) and
[#22](https://github.com/andrewendlinger/xmris/issues/22) re-scoped by what was decided.

| Cluster | Issues | Lands in |
|---|---|---|
| Design decisions — the board above | [#62](https://github.com/andrewendlinger/xmris/issues/62) [#64](https://github.com/andrewendlinger/xmris/issues/64) [#65](https://github.com/andrewendlinger/xmris/issues/65) [#66](https://github.com/andrewendlinger/xmris/issues/66) [#88](https://github.com/andrewendlinger/xmris/issues/88) [#124](https://github.com/andrewendlinger/xmris/issues/124) [#125](https://github.com/andrewendlinger/xmris/issues/125) [#136](https://github.com/andrewendlinger/xmris/issues/136) | v0.8 |
| Blocked behind them | [#21](https://github.com/andrewendlinger/xmris/issues/21) [#22](https://github.com/andrewendlinger/xmris/issues/22) [#23](https://github.com/andrewendlinger/xmris/issues/23) [#28](https://github.com/andrewendlinger/xmris/issues/28) [#34](https://github.com/andrewendlinger/xmris/issues/34) [#71](https://github.com/andrewendlinger/xmris/issues/71) [#102](https://github.com/andrewendlinger/xmris/issues/102) [#107](https://github.com/andrewendlinger/xmris/issues/107) | v0.8 |
| The tag itself | [#10](https://github.com/andrewendlinger/xmris/issues/10) [#80](https://github.com/andrewendlinger/xmris/issues/80) [#82](https://github.com/andrewendlinger/xmris/issues/82) [#115](https://github.com/andrewendlinger/xmris/issues/115) [#116](https://github.com/andrewendlinger/xmris/issues/116) [#122](https://github.com/andrewendlinger/xmris/issues/122) [#131](https://github.com/andrewendlinger/xmris/issues/131) [#137](https://github.com/andrewendlinger/xmris/issues/137) [#138](https://github.com/andrewendlinger/xmris/issues/138) | v0.7 |
| Quality & tooling | [#87](https://github.com/andrewendlinger/xmris/issues/87) [#108](https://github.com/andrewendlinger/xmris/issues/108) [#111](https://github.com/andrewendlinger/xmris/issues/111) [#117](https://github.com/andrewendlinger/xmris/issues/117) [#127](https://github.com/andrewendlinger/xmris/issues/127) [#133](https://github.com/andrewendlinger/xmris/issues/133) | v0.8 – v0.9 |
| The front door | [#27](https://github.com/andrewendlinger/xmris/issues/27) [#46](https://github.com/andrewendlinger/xmris/issues/46) [#67](https://github.com/andrewendlinger/xmris/issues/67) [#119](https://github.com/andrewendlinger/xmris/issues/119) [#120](https://github.com/andrewendlinger/xmris/issues/120) [#121](https://github.com/andrewendlinger/xmris/issues/121) [#126](https://github.com/andrewendlinger/xmris/issues/126) | v0.9 |
| Correctness & capability | [#29](https://github.com/andrewendlinger/xmris/issues/29) [#31](https://github.com/andrewendlinger/xmris/issues/31) [#83](https://github.com/andrewendlinger/xmris/issues/83) [#84](https://github.com/andrewendlinger/xmris/issues/84) [#113](https://github.com/andrewendlinger/xmris/issues/113) [#128](https://github.com/andrewendlinger/xmris/issues/128) [#132](https://github.com/andrewendlinger/xmris/issues/132) | v0.9 |
| Space & scale | [#4](https://github.com/andrewendlinger/xmris/issues/4) [#25](https://github.com/andrewendlinger/xmris/issues/25) | Horizon |

The decisions are not a flat list. The spine below is the board's dependency order — the decided
pair feeds the data model first, and the core/extras boundary and the plug-in promise decide each
other:

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart TD
    d02["02 lineage record ✓"] --> d03["03 data model"]
    d02b["02b constants ✓"] --> d03
    d04["04 core & extras"] <--> d12["12 plug-in promise"]
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
[opening an issue](https://github.com/andrewendlinger/xmris/issues/new). The same route works
forward: if your work depends on something sitting in the horizon band, say so there too, because
that is the main way something moves up a band.

The decision board keeps its paper on this site: each decided card links an exploration notebook,
frozen the day the decision lands, and an aimed-solution notebook that becomes the spec the
implementation is checked against. Both live in the sidebar beneath this page.

:::{seealso}
The [dev diary](#diary-about) is this page's backward-looking twin: one entry per decision
already taken, rewritten in place as that decision evolves.
:::
