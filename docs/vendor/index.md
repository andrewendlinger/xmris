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

(vendor)=
# Vendor formats

Everything in the other chapters starts from data that is already a labelled `DataArray`. Getting
there from what a scanner actually wrote to disk is its own problem — raw files carry acquisition
quirks that are not physics, and a loader that ignores them hands you a spectrum with a twist in it
nobody can explain.

This chapter covers those quirks, one vendor at a time. Bruker is the format xmris supports today.

(vendor-pages)=
## The pages

**Independent — read the one that matches your problem.**

| | Page | The quirk it handles |
|---|---|---|
| 1 | [Bruker — the digital filter group delay](#bruker-grpdly) | Removing the console's filter delay, and measuring the true delay from the data when the header lies |

(vendor-more)=
## Not in the sidebar

A second page, [FID loading](#bruker-fid), verifies the strict 1D→ND Paravision reshaping against
real fixtures. It is written for contributors rather than users and is still marked work in
progress, so it is kept out of the sidebar — but it renders, and the link above works.

(vendor-next)=
## Where to go next

Once your data is loaded, [Basics](#basics) covers the conventions xmris expects it to follow, and
the [Processing pipeline](#pipeline) picks it up from there.
