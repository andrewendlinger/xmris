---
name: new-doc-notebook
description: Create or edit an xmris documentation notebook (MyST .md under docs/notebooks/) following the project's notebook conventions. Use when adding a tutorial/doc page, writing notebook tests for a function, documenting a new .xmr method, or restructuring existing doc notebooks.
---

# Write an xmris doc notebook

Doc notebooks are **simultaneously the documentation and the test suite**. Every `.md` under `docs/notebooks/` is executed twice in CI:

1. `uv run test` — jupytext converts it to `tests/autogen_notebooks/*.ipynb`, then pytest runs it via nbmake (parallel, with coverage).
2. `myst build --html --execute` — the docs deploy re-executes it and publishes the output.

So every cell must run headless, deterministically, and reasonably fast. A notebook that only renders but doesn't prove anything is half done; a notebook that asserts but reads like a test file is also half done.

## 1. Placement & naming

- Category dirs: `basics/`, `pipeline/`, `fitting/`, `vendor/`, `visualization/plot/`, `visualization/widget/`.
- Lowercase snake_case filenames. Numeric `NN_` prefixes **only** inside the two `visualization/` dirs (elsewhere the TOC controls order).
- `testonly_` prefix = executed by the test suite but never added to the TOC, never rendered. Use for internal validation against ground-truth data.

## 2. Tone & writing style

Conversational, sharp, concise. No filler — get to the point. But never assume too much prior knowledge: background and theory that experts would skip goes into a `:::{dropdown}` block (the "Under the Hood: No Magic Strings" / "Imports & a small plotting helper" pattern). Every page should be educational, precise, easy to read, and look modern and inviting.

Use the full mystmd palette *where it earns its place*: LaTeX for the math actually being demonstrated, comparison tables, admonitions (`{note}`, `{warning}`, `seealso`), mermaid diagrams, sparing emoji as visual anchors (✅ ❌ 📦 ⚡ 🚧). Style exemplars: `docs/index.md`, `docs/notebooks/basics/architecture.md`, `docs/notebooks/pipeline/domain_agnostic_autophase.md`.

## 3. The canonical skeleton

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

One-paragraph hook: the problem this page solves, in plain language.
Physics/math motivation next — LaTeX, a table, or a mermaid diagram if it
genuinely clarifies.

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

Hard rules baked into that skeleton:

- **Frontmatter is exactly this.** `display_name: Python 3 (xmris)` is the frozen project label — if your local Jupyter rewrites it (to `.venv`, `Python 3 (ipykernel)`, …), fix it back before committing. Execution is unaffected either way; `name: python3` resolves to the uv venv.
- **Nothing before `(target)=` + `# H1`.** mystmd lifts the first H1 as the page title *only* when it is the first content node. Any cell before it — even a hidden `remove-cell` one — breaks the lift and the built page renders the title **twice**. The setup cell always comes *after* the H1. Exactly one H1 per file; never restate the title as another header.
- **Explicit MyST target above every header** (H1 and all `##`/`###`), kebab-case, prefixed with the notebook topic. Never rely on auto-generated slugs.
- **Reader-facing cells use plain strings** (`"time"`, `"frequency"`) — the low entrance barrier is deliberate. `ATTRS`/`DIMS`/`COORDS` from `xmris.core.config` may appear **only inside hidden test cells**.
- Imports in a plain visible cell; wrap them in a `:::{dropdown}` only when bundled with a longer plotting-helper function.
- `+++` splits adjacent markdown into separate cells — use it when prose after a `:::` block would otherwise be swallowed into it.

## 4. Hidden assert cells

Recommended for any cell that demonstrates a computation (use judgment — pure-concept or plotting-gallery pages can skip them). Pattern:

- Tagged `:tags: [remove-cell]`, placed **immediately after** the demonstration it verifies. nbmake still executes it; the site hides it.
- Opens with a `# STRICT TESTS: <what>` comment. Underscore-prefix throwaway variables (`_target_points`).
- `np.testing.assert_allclose` / `assert_array_equal` with an `err_msg=`, plus plain `assert`s for metadata.
- Prove three things: (a) the **math** is right, (b) **coordinates** were built/extrapolated correctly, (c) original **attrs survived** and the new lineage attrs were stamped (quantitative parameters only, e.g. `phase_p0` — never boolean flags).

## 5. Data

- Prefer `xmris.fitting.simulation.simulate_fid` for MRS-like signals; hand-rolled numpy is fine when the notebook is *teaching* raw construction.
- Seed randomness (`np.random.default_rng(42)` / `target_snr=...` with fixed seed context) whenever asserts depend on noisy data — unseeded noise means flaky CI.
- Real data only from `tests/data/` (relative path `../../../tests/data/` from a notebook). The gitignore blocks all data extensions; new files need explicit whitelist entries in `.gitignore` and must stay small (<10 MB).

## 6. Mermaid diagrams

Escaping is where these usually break — the existing diagrams were expensive to develop, so follow the house rules:

- Always double-quote node and edge labels: `A["label"]`, `B{"decision?"}`, `-->|"edge label"|`.
- Line breaks inside labels: `<br>`, never `\n`.
- Code inside labels: `<span style='font-family:monospace;'>fid.xmr.to_spectrum()</span>` — markdown backticks do **not** render inside mermaid labels.
- Single quotes for HTML attributes and for literal quotes inside labels (`dims='time'`); never nest unescaped double quotes.
- Prefer the ```` ```{mermaid} ```` directive in notebooks (bare ```` ```mermaid ```` fences also render).
- For sophisticated styling (`classDef`, `subgraph`, styled spans) copy the reference diagram in `docs/index.md`; for simple decision flowcharts copy `basics/architecture.md`.

## 7. Cell tag reference

| Tag | Rendered site | Tests (nbmake) | Use for |
|---|---|---|---|
| `remove-cell` | cell gone entirely | executed | matplotlib setup, STRICT TESTS |
| `remove-input` | output only | executed | `export_widget_static(...)` call |
| `remove-output` | input only | executed | live widget call the reader should type |
| `hide-input` | input collapsed | executed | data-generation boilerplate |
| `hide-output` | output collapsed | executed | verbose prints / long assertion logs |

## 8. Widget notebooks only

Static docs have no kernel, so live widgets need the export pattern from `docs/contributing/static_widgets.md`:

1. Show the reader the live call in a cell tagged `remove-output`.
2. Follow it with a hidden cell tagged `remove-input` that calls `export_widget_static(widget_factory, *args, **kwargs)` — that renders the interactive canvas on the site.

Mind the ~2.5 MB iframe payload limit (`debug=True` to inspect) and the `xmris-close-btn` CSS-class rule for kernel-dependent buttons. Screenshots live in `assets/notebook-assets/`.

## 9. Register, link, commit

- **TOC entry is mandatory** (except `testonly_`): add the file under the right group in `docs/myst.yml` with a `title:`. The sidebar shows the TOC title — keep it consistent with the H1. A notebook missing from the TOC still passes tests but silently never renders.
- **Cross-links**: `[text](#explicit-target)` or relative `.md` paths only. Never link `.ipynb` — those are excluded from the build, so the link dies.
- **Commit only the `.md`.** `docs/**/*.ipynb` is gitignored; edits made in an `.ipynb` twin are invisible to both tests and docs until synced back (`uv run jupytext --sync <file>`).

## 10. Verify before finishing

```bash
uv run test-gen
uv run pytest "tests/autogen_notebooks/<category>/<name>.ipynb" -n0 --no-cov
```

(`--nbmake` is already in pytest addopts.) Optionally preview the rendered page: `cd docs && uv run myst start --execute`.

Final checklist:

- [ ] Frontmatter exact; `display_name: Python 3 (xmris)`
- [ ] H1 is the very first content; target above it and above every header; single H1
- [ ] Setup cell after the H1, tagged `remove-cell`, dpi 150
- [ ] Reader-facing cells use plain dim strings; config singletons only in hidden cells
- [ ] Assert cells hidden with `remove-cell` and placed right after what they prove
- [ ] TOC entry in `docs/myst.yml` (unless `testonly_`)
- [ ] No `.ipynb` links; no unquoted mermaid labels
- [ ] Single-notebook nbmake run is green
- [ ] Only the `.md` staged
