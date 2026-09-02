(contribute-docs)=
# Write a docs page

Every hand-authored page in these docs is a MyST notebook — jupytext frontmatter plus a kernelspec
— so any of them can run live `code-cell`s with real output and plots. There are three genres,
split by reader and by where their cells execute:

- **Tutorials** (the five hands-on chapters — `docs/basics/`, `pipeline/`, `fitting/`,
  `visualization/`, `vendor/`) demonstrate a task step by step, and *are* the test suite:
  `uv run test` runs their asserts.
- **Explainers** (`docs/concepts/`) are motivated narrative for *why* something is the way it is
  — executed too, so their live claims are asserted like any tutorial cell.
- **Guides** (`docs/contribute/`) are procedural pages like this one; executed on the PR build,
  and — once they carry a kernelspec — by `uv run test` as well.

The four house-style rules — motivated narrative, one home per concept, every article stands alone,
and the MyST palette carries the argument — are the single source of truth in [`CLAUDE.md` §
Documentation style](https://github.com/andrewendlinger/xmris/blob/main/CLAUDE.md).

(contribute-docs-skill)=
## Working with Claude Code

The **`docs-page`** skill owns the cell structure, the hidden-assert convention, and the TOC step,
and it ships a stdlib-only checker (`check_docs.py`) that catches what the build stays silent about
— a missing target, a dead `.ipynb` link, a drifted kernel name. Run it on the page you are editing:

```bash
uv run python .claude/skills/docs-page/check_docs.py docs/<path>/<page>.md
```

Its errors **gate CI** — the `Docs style` job in `ci-fast.yml` runs the same command over the whole
tree on every PR, so a page with errors is a red build. Its warnings deliberately do not: they are
real drift, but too judgment-dependent to block a merge on. The checklist:

```{literalinclude} ../../.claude/skills/docs-page/SKILL.md
:language: markdown
:start-after: <!-- excerpt:start -->
:end-before: <!-- excerpt:end -->
:caption: Quote from the [docs-page/SKILL.md](https://github.com/andrewendlinger/xmris/blob/main/.claude/skills/docs-page/SKILL.md)
:class: skill-quote
```

(contribute-docs-format)=
## The page is half of a notebook pair

Every page here is committed as Markdown, but jupytext keeps it in sync with an `.ipynb` twin
(gitignored — `docs/**/*.ipynb` never enters git). Edit whichever side you prefer: prose and diffs
are easier in the `.md`, live plotting easier in Jupyter. What matters is that the `.md` you commit
is the one jupytext itself would write — the cell markers, the blank line after a directive's
options, the metadata kept by `[tool.jupytext]` in `pyproject.toml`.

Hand-editing drifts from that form invisibly: the page renders identically, so nothing complains
until the next contributor opens it in Jupyter and their first save produces a diff they never made.
So the form is checked mechanically — each page is converted `md → ipynb → md` and compared byte for
byte:

```bash
uv run docs-format          # fails on any page that is not in canonical form
uv run docs-format --fix    # rewrites them; review the diff, it should be whitespace only
```

`uv run lint` runs the check as its third step, and the `Lint` job runs it on every PR, so drift
cannot reach `main`. To catch it one step earlier — before the commit rather than after the push —
install the repository's hooks once:

```bash
uv run pre-commit install
```

Nothing depends on you doing so; `.pre-commit-config.yaml` runs the same three commands CI does.
