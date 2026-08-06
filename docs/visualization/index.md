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

(visualization)=
# Visualization

MR plots are unusually fussy: a stack of spectra, an axis that runs backwards, a colormap that has
to survive printing. Passing thirty keyword arguments to get there is how plotting code becomes
unreadable, so xmris hands you a **config object** instead — one dataclass per plot type, with
discoverable fields and a `repr` you can paste into a paper's methods section.

The other half of this chapter is interactive. Some decisions — where the phase actually looks
right, how much line broadening is too much — are faster to make with a slider than with a scoring
function. Those are widgets, and each one prints the reproducible `.xmr` call for what you settled
on, so the notebook stays runnable without it.

(visualization-pages)=
## Two families

**Read page 1 first, then pick what you need.** Everything after it is independent.

| | Page | When you reach for it |
|---|---|---|
| 1 | [Config-based plotting](#plot-basics) | The idea behind every plot here: why a dataclass beats thirty kwargs |
| 2 | [Waterfall plots](#waterfall) | Stacked, offset spectra across a series — shape at a glance |
| 3 | [Carpet plots](#carpet) | The same series top-down as a 2D image, when occlusion hides what matters |
| 4 | [Interactive phasing](#widget-phase) | Setting $p_0$ / $p_1$ by hand when the automated scorer is fooled |
| 5 | [Interactive scrolling](#widget-scroller) | Stepping through a dynamic or multi-echo dataset one spectrum at a time |
| 6 | [Interactive apodization](#widget-apodization) | Feeling the SNR-versus-resolution trade-off in both domains at once |

(visualization-next)=
## Where to go next

Plots of *fitted* data — kinetic trajectories and QC grids — live with the fits, in
[Visualizing dynamic fits](#dynamic-fits). The maths each widget is a front end for is in the
[Processing pipeline](#pipeline).
