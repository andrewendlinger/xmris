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

(pipeline)=
# Processing pipeline

A raw FID is not a spectrum you can read. Between the two sits a short, fairly standard sequence:
pad it, damp its tail, transform it, straighten its phase, flatten what is left underneath. This
chapter walks that sequence in the order you would actually apply it, one operation per page.

Every page uses the same shape — the physics first, then the `.xmr` call, then the plot that shows
what changed. Nothing here needs real data: the signals come from `simulate_fid`, so you can run
any page start to finish.

(pipeline-pages)=
## The sequence

**Order matters up to step 4** — the chapter is chronological, and each page picks up the array the
previous one produced. Step 5 is different: it is the proof that the whole chapter's domain
behaviour is what it claims.

| | Page | What it does to your data |
|---|---|---|
| 1 | [Zero filling](#zero-fill) | Pads the FID for sinc interpolation — and why that adds resolution but no information |
| 2 | [Apodization](#apodization) | Damps the noisy tail: exponential line broadening and the Lorentz-to-Gauss filter, trading SNR against resolution |
| 3a | [Phase correction](#phase) | The manual $p_0$ / $p_1$ model, and the pivot convention that makes it reproducible |
| 3b | [Automated phase correction](#autophase-intro) | `autophase` and its three scoring methods, across dense, sparse and ultra-low-SNR spectra |
| 3c | [Domain-agnostic autophase](#domain-agnostic-autophase) | Phasing straight from a FID, and the decorator tier that makes that safe |
| 4 | [Baseline correction](#baseline) | Asymmetric least squares against macromolecule and ringing baselines |
| 5 | [Domain contracts in action](#domain-contracts) | The capstone: every operation class × every input domain, asserted rather than asserted-to |

(pipeline-next)=
## Where to go next

Interactive versions of two of these steps — phasing and apodization — live in
[Visualization](#visualization), for when the automated scorer picks the wrong answer and you want
a slider. The design story behind step 5 is [The two domains](#domains).
