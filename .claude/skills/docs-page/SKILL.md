---
name: docs-page
description: Create or edit any hand-authored page in the xmris docs — tutorial notebooks (docs/notebooks/), concept explainers (docs/explanation/), or contributor guides (docs/contributing/). Use when adding or restructuring a doc page, writing notebook tests for a function, documenting a new .xmr method, thinning or consolidating existing pages, or fixing a page that renders wrong.
---

# Write an xmris docs page

All three hand-authored genres are MyST notebooks — jupytext frontmatter plus a kernelspec — so
any of them can reach for live `code-cell`s, real output and plots. What separates them is
**shape, reader, and where the cells are executed.**

## 1. Route by genre — read one template, not all four

| | **Tutorial** | **Explainer** | **Guide** |
|---|---|---|---|
| Lives in | `docs/notebooks/<area>/` | `docs/explanation/` | `docs/contributing/` |
| Reader wants | to *do* the task | to know *why it is like this* | to contribute |
| Shape | demonstrate → assert, step by step | motivated narrative, claim → live proof | numbered steps + commands |
| `uv run test` (nbmake + coverage) | ✅ | ✅ | ❌ |
| `myst build --execute` (deploy.yml, **on PRs**) | ✅ | ✅ | ✅ |
| Template | `templates/tutorial.md` | `templates/explainer.md` | `templates/guide.md` |

`templates/patterns.md` is the shared MyST pattern library — read it alongside whichever genre
template applies. It holds the devices mined from the pages that work, each pointing at its live
example so you copy rather than reinvent.

**Two paths never route here:**

- `docs/diary/` belongs to the **`dev-diary`** skill. Hand off; do not write an entry from here.
- `docs/api_reference/` is **generated and gitignored**. `docs_api()` clears the directory before
  regenerating, so a hand edit there is destroyed with no git history to recover it. Fix the
  docstring in `src/` instead.

House style — motivated narrative, one home per concept, every article stands alone, the MyST
palette carries the argument — lives in **`CLAUDE.md` § "Documentation style"** and is not
restated here. One carve-out: the *motivated narrative* rule binds explainers and tutorials, not
guides. A numbered list of commands is the correct shape for `setup.md`; novelizing an install
guide makes it worse.

## 2. Rules that survive genre

Every one of these is enforced by `check_docs.py`, and every one is currently violated somewhere
in the repo — the build is silent about all of them.

- **Frontmatter is exact**, and `display_name: Python 3 (xmris)` is the frozen project label. If
  your local Jupyter rewrites it (to `.venv`, `Python 3 (ipykernel)`, …), fix it back before
  committing. Execution is unaffected either way; `name: python3` resolves to the uv venv.
- **Nothing before `(target)=` + `# H1`.** mystmd lifts the first heading into the page title but
  only *removes* it from the body when it leads the page. Put anything first — even a hidden
  `remove-cell` — and the title renders **twice**. The setup cell always comes *after* the H1.
  Exactly one H1 per file; never restate the title as another header.
- **Explicit MyST target above every header** (Commandment 8), kebab-case, prefixed with the page
  topic. Auto-generated slugs are numbered by *document position* — `id-1-`, `id-2-` — so
  inserting one section silently renumbers every anchor below it and breaks deep links.
- **Never link `.ipynb`.** `myst.yml` excludes `notebooks/**/*.ipynb`, so the link resolves to
  `null` and dies — with no build warning. Use `[text](#explicit-target)` or a relative `.md` path.
- **TOC entry in `docs/myst.yml` is mandatory** (except `testonly_`). The sidebar shows the TOC
  title — keep it consistent with the H1. A page missing from the TOC never renders.
- **Reader-facing cells use plain strings** (`"time"`, `"frequency"`) — the low entrance barrier is
  deliberate. `ATTRS`/`DIMS`/`COORDS` appear only in hidden test cells or in passages explicitly
  addressed to contributors.
- **Commit only the `.md`.** `docs/**/*.ipynb` is gitignored; edits made in an `.ipynb` twin are
  invisible to both tests and docs until synced back (`uv run jupytext --sync <file>`).

Tone across all three: conversational, sharp, concise, no filler — but never assuming expertise.
Background an expert would skip goes in a `:::{dropdown}`, off the main line.

## 3. Editing an existing page

Most doc work is editing, and the house rules make thinning **expected work, not scope creep**.

- Run the checker on the page *before* you start. Pre-existing errors are not yours to fix
  silently, but say what you found — several pages carry them today.
- Adding a `(target)=` above a header **changes its slug**. Grep for inbound links first:
  `git grep -nF "#old-slug" -- docs`.
- Consolidating across pages is the *one home per concept* rule doing its job. Say what you moved
  in the PR body, and keep the orienting recap on the page you thinned.

## 4. Verify before finishing

```bash
uv run python .claude/skills/docs-page/check_docs.py docs/<path>/<page>.md
```

Errors are render-breaking and exit 1; warnings are drift and exit 0. Then, for a **tutorial or an
explainer** — `test-gen` walks `docs/notebooks/` *and* `docs/explanation/`, so both are in the
pytest suite:

```bash
uv run test-gen
uv run pytest "tests/autogen_notebooks/<category>/<name>.ipynb" -n0 --no-cov
# explainers land flat under explanation/:
uv run pytest "tests/autogen_notebooks/explanation/<name>.ipynb" -n0 --no-cov
```

(`--nbmake` is already in pytest addopts.) A page only joins that suite once it carries a jupytext
**kernelspec** — that is the exact condition `_is_executable_page` tests, so a frontmatter-less
page is skipped rather than failing with no kernel to start.

**Guide** cells are executed by the docs build only — `cd docs && uv run myst build --html
--execute` locally, the same command `deploy.yml` runs on every PR. (`uv run docs` and `myst start`
serve a live preview and never exit; they are for reading the site, not for checking it.)

Two pathspec traps, both of which produced wrong answers while auditing this repo:

- `git grep … -- 'docs/**/*.md'` **silently skips `docs/index.md`** — no intermediate directory.
  Use `-- docs`.
- `grep -r docs/` walks ~9,800 gitignored `_build/` files carrying stale copies of everything.

## Checklist

- [ ] Genre identified; the matching template and `patterns.md` read
- [ ] Frontmatter exact, `display_name: Python 3 (xmris)`
- [ ] `(target)=` + single H1 is the very first content; a target above every header
- [ ] Reader-facing cells use plain dim strings; config singletons only in hidden cells
- [ ] TOC entry in `docs/myst.yml` (unless `testonly_`); no `.ipynb` links
- [ ] `check_docs.py` passes on the page (0 errors)
- [ ] Tutorial only: single-notebook nbmake run is green
- [ ] Only the `.md` staged
