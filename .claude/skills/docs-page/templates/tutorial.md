# Tutorial — one of the hands-on chapters (`docs/basics/`, `pipeline/`, `fitting/`, `visualization/`, `vendor/`)

The genre where **the documentation is the test suite.** Every page here is executed twice in CI:
by `uv run test` (jupytext → `tests/autogen_notebooks/*.ipynb` → nbmake, parallel, with coverage)
and by `myst build --execute` (the docs deploy, on PRs too).

So every cell must run headless, deterministically, and reasonably fast. A page that only renders
but proves nothing is half done; a page that asserts but reads like a test file is also half done.

## Placement & naming

- Category dirs: `basics/`, `pipeline/`, `fitting/`, `vendor/`, `visualization/plot/`,
  `visualization/widget/`.
- Lowercase snake_case. Numeric `NN_` prefixes **only** inside the two `visualization/` dirs —
  elsewhere the TOC controls order.
- `testonly_` prefix = executed by the test suite, never in the TOC, never rendered. Use for
  internal validation against ground-truth data.

## Skeleton

````markdown
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

(my-topic)=
# Descriptive Title

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

# Crisp retina output + sane default DPI for the rendered docs
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150
```

One-paragraph hook: the problem this page solves, in plain language. Physics/math
motivation next — LaTeX, a table, or a mermaid diagram if it genuinely clarifies.

| Function | What it does here |
|---|---|
| [`simulate_fid()`](#xmris.fitting.simulation.simulate_fid) | stands in for the scanner |
| [`.xmr.some_method()`](#xmris.core.accessor.XmrisProcessingMixin.some_method) | the transform this page is about |

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

import xmris  # registers the .xmr accessor
from xmris.fitting.simulation import simulate_fid
```

(my-topic-generate-data)=
## 1. Generate a synthetic FID

```{code-cell} ipython3
fid = simulate_fid(...)
```

(my-topic-apply)=
## 2. Apply the transform

```{code-cell} ipython3
result = fid.xmr.some_method()
```

... plot the result with xarray's native plotting ...

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: some_method
from xmris.core.config import ATTRS, DIMS

np.testing.assert_allclose(..., err_msg="...")
```
````

Imports go in a plain visible cell; wrap them in a `:::{dropdown}` only when bundled with a longer
plotting helper. `+++` splits adjacent markdown into separate cells — use it when prose after a
`:::` block would otherwise be swallowed into it.

The **functions-used table** sits under the hook, before the first cell: every xmris call the page
makes, linked to its API entry, one line on what it does *here* (not what it does in general — the
API page says that). Anchors are quartodoc's dotted targets, which are project-global, so a bare
`#anchor` resolves from any page: `#xmris.fitting.amares.fit_amares` for the free function,
`#xmris.core.accessor.XmrisAccessor.fit_amares` for the accessor method. Find the exact one with
`grep -n "^(xmris" docs/api/<module>.md` after `uv run docs-api`. Skip the table on pure
concept pages that call nothing.

## Hidden assert cells

Recommended for any cell that demonstrates a computation. A tutorial that runs code and asserts
nothing is a doc, not a test — the checker warns about exactly that, and four pages trip it today.
Pure-concept and plotting-gallery pages may legitimately skip them; say so rather than drifting.

- Tagged `:tags: [remove-cell]`, placed **immediately after** the demonstration it verifies.
  nbmake still executes it; the site hides it.
- Opens with a `# STRICT TESTS: <what>` comment. Underscore-prefix throwaway variables
  (`_target_points`).
- `np.testing.assert_allclose` / `assert_array_equal` with an `err_msg=`, plus plain `assert`s for
  metadata.
- Prove three things: (a) the **math** is right, (b) **coordinates** were built or extrapolated
  correctly, (c) original **attrs survived** and the new lineage attrs were stamped — quantitative
  parameters only, e.g. `phase_p0`, never boolean flags.

Number the assertions to the claims they back, as `domain_agnostic_autophase.md` does:

```python
# 1. The auto-FFT ran: output is a spectrum, and the time axis is gone.
# 2. Metadata survived the auto-FFT.
# 3. Phase lineage was recorded.
```

## Data

**Synthetic MRS signals come from `simulate_fid`.** Not "prefer" — the checker warns on a
hand-written damped sinusoid anywhere outside `basics/`, whose subject genuinely *is* raw
construction. Re-deriving the forward model inline duplicates
`docs/fitting/simufid.md`, drifts from it, and usually does it in the `for` loop the
package exists to delete.

```python
from xmris.fitting.simulation import simulate_fid

fid = simulate_fid(
    amplitudes=[10.0, 5.0],
    chemical_shifts=[0.0, -7.5],            # or frequencies=[...] in Hz
    reference_frequency=120.6,              # MHz — required for the ppm route
    spectral_width=8000.0,
    n_points=512,
    dampings=[np.pi * 15.0, np.pi * 20.0],  # damping = pi * linewidth [Hz]
    target_snr=60.0,
    seed=0,
)
```

`simulate_fid` returns **one 1-D FID**. For anything N-dimensional, simulate per spectrum and
stack — this is the recipe pages used to hand-roll a loop to avoid:

```python
grid = xr.concat(
    [simulate_fid(amplitudes=[a, 5.0], ..., seed=i) for i, a in enumerate(amps)],
    dim="voxel",
).assign_coords(voxel=np.arange(len(amps)))
# xr.concat keeps only the FIRST FID's attrs, so per-spectrum simulation lineage
# (target_snr, sim_amplitudes) would be wrong for the stack. Replace it with the
# calibration downstream functions actually need:
grid.attrs = {"reference_frequency": 120.6, "carrier_ppm": 0.0}
```

Realism is a parameter, not a rewrite. Data that converges on the first try teaches nothing about
reading a result:

| Effect | How |
|---|---|
| Peak movement (B0 drift, shim) | vary `chemical_shifts` per spectrum |
| Noise, SNR gradient | `target_snr=` per spectrum, always with `seed=` |
| Phase distortion, receiver dead time | `phases=` (radians) / `dead_time=` |
| Broad macromolecular baseline | a second `simulate_fid` with `dampings≈1200`, summed in — see `pipeline/baseline.md` |
| Lineshape between Lorentzian and Gaussian | `lineshape_g=` per peak |
| An empty voxel / dead channel | `xr.zeros_like(fid)` for that slice |

Live exemplar of all of this at once: `fitting/pyamares.md` § 1.

- **Seed randomness** (`np.random.default_rng(42)`, or `target_snr=` with a fixed seed) whenever
  asserts depend on noisy data. Unseeded noise means flaky CI.
- Real data only from `tests/data/` (relative path `../../../tests/data/` from a notebook). The
  gitignore blocks all data extensions; new files need explicit whitelist entries and must stay
  small (<10 MB).

## Cell tags

| Tag | Rendered site | Tests (nbmake) | Use for |
|---|---|---|---|
| `remove-cell` | cell gone entirely | executed | matplotlib setup, STRICT TESTS |
| `remove-input` | output only | executed | `export_widget_static(...)` call |
| `remove-output` | input only | executed | live widget call the reader should type |
| `hide-input` | input collapsed | executed | data-generation boilerplate |
| `hide-output` | output collapsed | executed | verbose prints / long assertion logs |

## Widget pages only

Static docs have no kernel, so live widgets need the export pattern from
`docs/contribute/static_widgets.md`:

1. Show the reader the live call in a cell tagged `remove-output`.
2. Follow it with a hidden cell tagged `remove-input` that calls
   `export_widget_static(widget_factory, *args, **kwargs)` — that renders the interactive canvas
   on the site.

Mind the ~2.5 MB iframe payload limit (`debug=True` to inspect) and the `remove-me-close-btn`
CSS-class rule for kernel-dependent buttons. Screenshots live in `assets/notebook-assets/`.
