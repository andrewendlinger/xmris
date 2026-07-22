# Tutorial — `docs/notebooks/<area>/`

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

- Prefer `xmris.fitting.simulation.simulate_fid` for MRS-like signals; hand-rolled numpy is fine
  when the page is *teaching* raw construction.
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
`docs/contributing/static_widgets.md`:

1. Show the reader the live call in a cell tagged `remove-output`.
2. Follow it with a hidden cell tagged `remove-input` that calls
   `export_widget_static(widget_factory, *args, **kwargs)` — that renders the interactive canvas
   on the site.

Mind the ~2.5 MB iframe payload limit (`debug=True` to inspect) and the `remove-me-close-btn`
CSS-class rule for kernel-dependent buttons. Screenshots live in `assets/notebook-assets/`.
