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

Turning a raw FID into a spectrum you can read, one step per page.

1. [Zero filling](#zero-fill) — padding before the transform
2. [Apodization](#apodization) — damping the noisy tail
3. Phase correction — [by hand](#phase), [automatically](#autophase-intro), or
   [straight from a FID](#domain-agnostic-autophase)
4. [Baseline correction](#baseline) — flattening what is left underneath
5. [Domain contracts in action](#domain-contracts) — the whole chapter, asserted

Chronological up to step 4; step 5 is the proof.
