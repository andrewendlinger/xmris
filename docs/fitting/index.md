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

(fitting)=
# Fitting & simulation

Processing gets you a spectrum you can look at. Quantification gets you numbers you can defend —
amplitudes, linewidths, chemical shifts, each with an uncertainty attached. xmris does that through
[AMARES](#pyamares), wrapped so it behaves like the rest of the pipeline: xarray in, xarray out,
one voxel or a whole volume on the same line.

The same forward model runs backwards, too. `simulate_fid` builds synthetic data from a parameter
list, which is how every page in these docs makes its test signals.

:::{note} Fitting is an optional extra
`pip install xmris` does not pull in the fitting backend. Install `xmris[fitting]` before running
the pages below — `fit_amares` raises an `ImportError` telling you exactly that if you forget.
Simulation is core and always available.
:::

(fitting-pages)=
## The four pages

**Start with the quick start; the rest are independent.** Each later page assumes you have seen a
fit succeed once.

| | Page | What you walk away with |
|---|---|---|
| 1 | [Quick start: fitting a spectrum](#fitting-quickstart) | Five minutes from FID to a table of amplitudes with error bars |
| 2 | [FID simulation](#simufid) | `simulate_fid` and the damped-sinusoid model behind it, for ¹H and ¹³C |
| 3 | [AMARES in depth](#pyamares) | What comes back in the fitted `Dataset`, and how to tell which of those numbers to believe |
| 4 | [Visualizing dynamic fits](#dynamic-fits) | Kinetic trajectories and QC grids across a dynamic or CSI series |

(fitting-next)=
## Where to go next

Prior knowledge is where most fits are won or lost — [AMARES in depth](#pyamares) covers building
it, and the [API reference](#api-home) has the parameter-by-parameter detail.
