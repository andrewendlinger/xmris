(changelog)=
# Changelog

What shipped in each release; *why* is in the [dev diary](#diary-about), what is next in the
[roadmap](#roadmap).

(changelog-v0-7-0)=
## v0.7.0 — unreleased

The fitting subsystem, the domain-contract engine, a hardened vocabulary, and a documentation site
rebuilt around them.

**Breaking**

- The controlled vocabulary is **canonical-only** — no aliases. Bruker multi-receiver data now loads
  with the dimension `coil` (was `channels`), acquisition dimensions are singular (`average`,
  `repetition`), vocabulary terms are frozen against mutation, and the legacy `bruker_group_delay`
  attribute fallback is gone. Rename inbound dimensions with `obj.rename({...})` until
  `da.xmr.map_vocab` lands. — [#65](https://github.com/andrewendlinger/xmris/issues/65) ·
  [#96](https://github.com/andrewendlinger/xmris/pull/96) ·
  [The controlled vocabulary](#vocabulary)

**Added**

- `da.xmr.fit_amares` fits AMARES in the time domain and returns a `Dataset` of fitted parameters
  alongside the reconstructed signals. It meets data in either domain and returns it unchanged,
  normalises away the magnitude scale trap that made Bruker-scale FIDs "converge" on their prior,
  and writes `NaN` for a failed voxel rather than a spurious zero. `build_prior_knowledge` replaces
  pyAMARES's positional CSV with a named-peak dict, and the optimizer itself lives behind the
  optional `xmris[fitting]` extra, so a bare `import xmris` never pulls it in. —
  [#67](https://github.com/andrewendlinger/xmris/issues/67)
  [#69](https://github.com/andrewendlinger/xmris/issues/69)
  [#70](https://github.com/andrewendlinger/xmris/issues/70)
  [#80](https://github.com/andrewendlinger/xmris/issues/80)
  [#81](https://github.com/andrewendlinger/xmris/issues/81)
  [#82](https://github.com/andrewendlinger/xmris/issues/82) ·
  [#105](https://github.com/andrewendlinger/xmris/pull/105) ·
  [Quick start: fitting](#fitting-quickstart) · [AMARES in depth](#pyamares) ·
  [diary](#diary-amares-fitting)
- Every domain-sensitive transform now declares the domain it works in, via `@computes_in`
  (domain-preserving) or `@ensures_domain` (funnelling), so passing a FID where a spectrum is wanted
  converts instead of returning nonsense. — [#42](https://github.com/andrewendlinger/xmris/issues/42)
  [#63](https://github.com/andrewendlinger/xmris/issues/63) ·
  [#73](https://github.com/andrewendlinger/xmris/pull/73) ·
  [#77](https://github.com/andrewendlinger/xmris/pull/77) ·
  [#78](https://github.com/andrewendlinger/xmris/pull/78) ·
  [The Two Domains](#domains) · [Domain contracts in action](#domain-contracts)
- `xmris.set_options(auto_convert=False)` turns that automatic conversion into a loud error naming
  the converter to call — strict mode for quantitative work. Global or a context manager, mirroring
  `xr.set_options`. — [#63](https://github.com/andrewendlinger/xmris/issues/63) ·
  [#79](https://github.com/andrewendlinger/xmris/pull/79) · [The Two Domains](#domains)
- `da.xmr.baseline_als` corrects a spectral baseline by asymmetric least squares. —
  [#43](https://github.com/andrewendlinger/xmris/issues/43) ·
  [#44](https://github.com/andrewendlinger/xmris/pull/44) · [Baseline correction](#baseline)
- `da.xmr.estimate_group_delay` measures a Bruker group delay from the data by minimising the
  residual, instead of trusting the header value. —
  [#85](https://github.com/andrewendlinger/xmris/issues/85) ·
  [#89](https://github.com/andrewendlinger/xmris/pull/89) · [Bruker — group delay](#bruker-grpdly)
- `da.xmr.widget.scroll_spectra` scrolls through a stack of spectra, and `da.xmr.widget.apodize`
  tunes line broadening against a live plot. Both render in the built documentation, not only in a
  running kernel. — [#16](https://github.com/andrewendlinger/xmris/issues/16) ·
  [#37](https://github.com/andrewendlinger/xmris/pull/37) ·
  [#38](https://github.com/andrewendlinger/xmris/pull/38) ·
  [#40](https://github.com/andrewendlinger/xmris/pull/40) ·
  [Interactive scrolling](#widget-scroller) · [Interactive apodization](#widget-apodization)

**Changed**

- `xmris` is released under **BSD-3-Clause**, replacing AGPL-3.0. —
  [#130](https://github.com/andrewendlinger/xmris/pull/130)
- `autophase` was rewritten for robustness, and `nmrglue` is no longer a dependency. —
  [#30](https://github.com/andrewendlinger/xmris/issues/30) ·
  [#36](https://github.com/andrewendlinger/xmris/pull/36) ·
  [Automated phase correction](#autophase-intro)
- Plotting is configured by objects rather than keyword soup — `WaterfallConfig`, `CarpetConfig`,
  `PlotTrajectoryConfig`, `PlotQCGridConfig`, reached through `da.xmr.plot`. —
  [#39](https://github.com/andrewendlinger/xmris/issues/39) ·
  [#41](https://github.com/andrewendlinger/xmris/pull/41) ·
  [Config-based plotting](#plot-basics) · [Waterfall plots](#waterfall) · [Carpet plots](#carpet)
- `apodize_exp`, `apodize_lg` and `zero_fill` return the representation you handed them, while
  `baseline_als` funnels into the spectral domain and stays there. —
  [#63](https://github.com/andrewendlinger/xmris/issues/63) ·
  [#78](https://github.com/andrewendlinger/xmris/pull/78) · [The Two Domains](#domains)

**Fixed**

- A bare `pip install xmris` was unimportable: matplotlib is imported by every `import xmris` but
  arrived only transitively through pyAMARES, and vanished when pyAMARES moved to the `fitting`
  extra. `requires-python` also read `<=3.13`, which PEP 440 resolves to `<= 3.13.0` — admitting no
  3.13 patch release. Both are fixed, and CI now installs the way a user does. —
  [#122](https://github.com/andrewendlinger/xmris/issues/122) ·
  [#147](https://github.com/andrewendlinger/xmris/pull/147)
- `fit_amares` reads the canonical `reference_frequency` attribute, so FIDs produced by
  `simulate_fid` fit without a manual `mhz=`. —
  [#68](https://github.com/andrewendlinger/xmris/issues/68) ·
  [#93](https://github.com/andrewendlinger/xmris/pull/93)
- Waterfall and carpet plots reject complex input instead of silently plotting its real part, and
  keep the time axis's units. — [#83](https://github.com/andrewendlinger/xmris/issues/83) ·
  [#94](https://github.com/andrewendlinger/xmris/pull/94)
- `import xmris` no longer emits a `DeprecationWarning`. —
  [#67](https://github.com/andrewendlinger/xmris/issues/67) ·
  [#92](https://github.com/andrewendlinger/xmris/pull/92)

**Documentation**

- The site was rebuilt: one directory per chapter, hands-on tutorials split from concept explainers,
  and a landing page per chapter. — [#126](https://github.com/andrewendlinger/xmris/issues/126)
  [#137](https://github.com/andrewendlinger/xmris/issues/137) ·
  [#139](https://github.com/andrewendlinger/xmris/pull/139)
- The contributor documentation is new in full: [The Architecture Contract](#contract) — the eleven
  rules every change to `src/xmris/` obeys — a page per kind of contribution, and
  [the dev diary](#diary-about). —
  [#72](https://github.com/andrewendlinger/xmris/issues/72) ·
  [#103](https://github.com/andrewendlinger/xmris/pull/103) ·
  [#114](https://github.com/andrewendlinger/xmris/pull/114) ·
  [Contribute](#contribute-home) · [Open a pull request](#contribute-pr) ·
  [diary](#diary-architecture-contract)
- Two explainers for the design decisions users hit first: [The Two Domains](#domains) on why a
  function cares whether it is handed a FID or a spectrum, and
  [The controlled vocabulary](#vocabulary) on why the names are fixed. —
  [#76](https://github.com/andrewendlinger/xmris/pull/76) ·
  [#100](https://github.com/andrewendlinger/xmris/pull/100)
- [The roadmap](#roadmap) says what is shipped, in motion, and still being argued about. —
  [#116](https://github.com/andrewendlinger/xmris/issues/116) ·
  [#123](https://github.com/andrewendlinger/xmris/pull/123) ·
  [#145](https://github.com/andrewendlinger/xmris/pull/145)
- Every pull request now publishes a fully executed preview of the site it would produce. —
  [#112](https://github.com/andrewendlinger/xmris/pull/112) ·
  [How the documentation reaches the web](#contribute-pr-deployment) · [diary](#diary-docs-previews)

**Maintenance**

- The documentation pages *are* the maths tests — every tutorial and explainer is executed by
  `nbmake` on both ends of the supported Python range, and a whole-tree docs-style checker gates the
  merge. — [#104](https://github.com/andrewendlinger/xmris/issues/104) ·
  [#110](https://github.com/andrewendlinger/xmris/pull/110) ·
  [diary](#diary-authoring-skills)
- Seven dependency updates, now arriving weekly via Dependabot, plus CI hardening: the site is published from
  workflow artifacts rather than a 103 MB committed branch, the Codecov uploader comes from PyPI so
  an outage cannot block every merge, and `ruff format` is gated for the first time. —
  [#115](https://github.com/andrewendlinger/xmris/issues/115)
  [#141](https://github.com/andrewendlinger/xmris/issues/141)
  [#151](https://github.com/andrewendlinger/xmris/issues/151) ·
  [#142](https://github.com/andrewendlinger/xmris/pull/142) ·
  [#146](https://github.com/andrewendlinger/xmris/pull/146) ·
  [#152](https://github.com/andrewendlinger/xmris/pull/152)

(changelog-earlier)=
## Earlier releases

v0.1.0 – v0.6.1 predate this changelog. Their contents are the
[tag list](https://github.com/andrewendlinger/xmris/tags) and the commits between them.
