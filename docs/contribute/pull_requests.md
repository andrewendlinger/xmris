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

(contribute-pr)=
# Open a pull request

Your change is written and green locally. This page covers everything between that and it being on
`main`: how to name the commit, what the six checks that gate the merge actually measure, where to
read your change rendered as a website before anyone reviews it, and who merges. It applies to every
kind of change — a processing method, a widget, a docs page, a diary entry.

`main` is protected: it takes no direct pushes, so a pull request is the only route in.

(contribute-pr-branch)=
## 1. Branch, and title it as a Conventional Commit

```bash
git checkout main && git pull
git checkout -b <type>/<short-slug>     # e.g. feat/apodize-gauss, docs/domains-thinning
```

The **pull request title** becomes the commit subject on `main`, because pull requests are
squash-merged — so it is the title, not your individual commits, that has to read well in
`git log`. Use a [Conventional Commit](https://www.conventionalcommits.org/) prefix: `feat:`,
`fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`. A parenthesised scope is encouraged where
it narrows usefully — `fix(packaging)`, `chore(ci)`, `feat(fitting)` — and a `!` before the colon
marks a breaking change.

The body is where you say what moved. If you consolidated documentation — thinned one page into
another, gave a concept a new home — name the pages, because that is the part a reviewer cannot see
from the diff alone. Write it for someone who was not in the discussion: at release time the
[changelog](#changelog) entry for your change is written from this body, not from the diff, and it
carries your issue and pull-request numbers forward as its trail.

(contribute-pr-open)=
## 2. Open it against `main`

```bash
git push -u origin <your-branch>
gh pr create --base main --title "docs: thin the domains page into the contract" --body "…"
```

Or push the branch and open it from the GitHub web interface — after a push, the repository page
offers a **Compare & pull request** button, and the title and body fields are the same ones the
command above fills in. Nothing downstream cares which route you took.

Every push to the branch re-runs the checks below and refreshes your preview.

(contribute-pr-checks)=
## 3. Six checks gate the merge

All six must pass before the merge button works. Each one is reproducible locally with a single
command — run it there first, since a local failure costs seconds and a CI failure costs minutes:

| Check | What it measures | Reproduce locally |
|---|---|---|
| **`Docs style`** | The docs rules `myst build` stays silent about: a header with no explicit target, a dead `.ipynb` link, a drifted kernel label, a page missing from the TOC | `uv run python .claude/skills/docs-page/check_docs.py` |
| **`Lint`** | Both halves of ruff: that every Python file is formatted, and that it is free of lint errors | `uv run lint` |
| **`build`** | Executes **every** notebook and explainer in the documentation, then fails on any mystmd error — a broken cross-reference, an unresolvable DOI, a directive that did not render | `cd docs && uv run myst build --html --execute --strict` |
| **`bare install`** | Installs **only** `[project].dependencies` into a clean venv — the set a real user receives — then imports xmris and runs the processing chain | `uv venv /tmp/bare && uv pip install --python /tmp/bare/bin/python . && /tmp/bare/bin/python .github/scripts/bare_install_smoke.py` |
| **`test (3.10)`** and **`test (3.13)`** | The full suite — including the tutorials, which *are* the maths tests — on both ends of the supported Python range | `uv run test` |

Three things worth knowing about that table. The `build` check is why a notebook that hangs can never
reach `main`: it runs under a 20-minute ceiling, so a stuck kernel fails visibly instead of burning
six hours. And `--strict` is load-bearing — without it mystmd reports its errors and *still* exits 0,
which is how a dead link once survived on the site for months.

`Lint` is the newest, and it stops somewhere deliberate. ruff formats Python inside Markdown code
blocks as readily as inside `.py` files, which would put every snippet on this site in its hands —
and it flattens them: the leading-dot `.xmr` chains collapse onto one line, the aligned comment
gutters lose their gutter. Since every page here is executed as a test, correctness is gated
already; what is left is layout, and layout is an authoring decision. So `pyproject.toml` excludes
`*.md` from ruff entirely, and `uv run ruff format .` is safe to run over the whole tree.

`bare install` is the odd one out, and deliberately so: every other job installs with `uv sync
--all-extras --dev`, which means the dependency set an actual `pip install xmris` produces is
exercised nowhere else. That gap once shipped a release whose `import xmris` raised
`ModuleNotFoundError` while all four other checks stayed green, so this job installs the way a user
does and refuses to trust anything the dev environment happens to provide.

Two further jobs report on the pull request without gating it. `publish` assembles and deploys the
site — that is where your preview comes from — and `links` checks external URLs **weekly on a
schedule** rather than on your branch, because a third-party URL rotting is not your fault and
should not redden your pull request.

(contribute-pr-preview)=
## 4. Read your change on its own website

Within a few minutes a bot comments a link like

```text
https://andrewendlinger.github.io/xmris/pr-preview/pr-123/
```

That is your branch, built and **fully executed** — the same build the live site gets, not a lighter
one. Plots are real plots, hidden asserts really ran. For anything reader-facing this is the review
surface: prose, admonitions, mermaid diagrams and cell output only reveal what they actually look
like once rendered. A [dev-diary](#contribute-dev-diary) entry is reviewed here like any
other page.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    P["Push to the PR"] --> B["Build: execute + strict"]
    B --> U["Upload preview-pr-N"]
    U --> A["Assemble: main + every open PR"]
    A --> D["Deploy the whole site"]
    D --> C["Bot comments the link"]
    M["Merge or close"] --> X["Dropped from the next assembly"]
```

Two caveats. The link arrives a couple of minutes after the green `build` check, because the deploy
runs after it — the comment is posted once the site is actually live, so the link never points at
nothing. And a pull request from a **fork** gets no preview: its token is read-only by design, so
the build still runs as a smoke test but cannot publish. That is not a failure — an external
contributor's first pull request should not open with a red X they have no way to fix.

(contribute-pr-green)=
## 5. Drive it green, then hand off

Push fixes until all six checks pass. You do **not** need to keep the branch up to date with `main`
— the checks are not configured strictly, so an unrelated merge landing meanwhile will not force you
to rebase. Rebase only for a real conflict:

```bash
git fetch origin && git rebase origin/main
```

Then hand off: a maintainer reviews and merges. No approving review is required by the tooling, so a
maintainer can merge their own work once it is green — the checks are the gate, not a rubber stamp.
Cutting a release is a separate, maintainer-run workflow ([Publishing](#contribute-release)).

(contribute-pr-deployment)=
## How the documentation reaches the web

Deployment is continuous and has nothing to do with releases, and the built site is never stored
anywhere: each deploy reassembles it from scratch — `main`'s most recent build, plus one artifact
per *currently open* pull request — and publishes that whole tree to GitHub Pages. Your preview is
on the site because your pull request is open, and it leaves on the first deploy after it closes.
Nothing removes it; it simply stops being an input. A weekly scheduled run performs the same
assembly, which keeps `main`'s build fresh and sweeps the preview of a pull request that closed
during a quiet week. ([The diary entry](#diary-docs-previews) tells why it works this way.)

The one subtlety worth internalising if you ever touch the workflow: mystmd bakes the site's base
path into every asset link **at build time**, so a preview has to be built for the subdirectory it
will be served from. That is what this step does, quoted from the workflow itself:

```{literalinclude} ../../.github/workflows/deploy.yml
:language: yaml
:start-at: - name: Build MyST Site
:end-at: uv run myst build --html --execute --strict
:caption: The build step — quoted from .github/workflows/deploy.yml at build time
```

To rebuild and republish the site without merging anything — after a failed deploy, say — run the
**Documentation** workflow manually from the Actions tab (`workflow_dispatch`). That is also the
recovery when `publish` fails with `No usable 'site-main' artifact`: a Pages deployment is atomic,
so the previous site keeps serving until a good one replaces it, and the failure is loud rather
than destructive.

```{code-cell} python
:tags: [remove-cell]
from pathlib import Path

# Pin the literalinclude anchors above. mystmd only *warns* on an unmatched
# start-at/end-at, even under --strict, so a renamed step or build command would
# silently truncate the quote instead of failing the build. This cell fails it.
#
# mystmd executes a notebook with the working directory set to the page's own
# folder (docs/contribute/ here), so walk up to the repo root rather than
# guessing a relative depth.
REL = Path(".github/workflows/deploy.yml")
start = Path.cwd()
workflow = next((d / REL for d in [start, *start.parents] if (d / REL).exists()), None)
assert workflow is not None, f"{REL} not found walking up from {start}"

text = workflow.read_text()
assert "- name: Build MyST Site" in text, "literalinclude start-at anchor no longer matches"
assert "uv run myst build --html --execute --strict" in text, "end-at anchor no longer matches"
```

(contribute-pr-troubleshooting)=
## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `Docs style` red, `header '…' has no explicit (target)= above it` | MyST auto-slugs are numbered by document position, so they break on insertion | Add `(page-topic-section)=` above the header; link it as `[text](#page-topic-section)` |
| `build` red, `Could not find DOI "…" from doi.org` | DOI metadata is resolved over the network unless it is frozen in `docs/myst.doi.bib` | `cd docs && uv run myst build --doi-bib`, then commit `myst.doi.bib` |
| `build` red, `Site has N error(s), stopping build` | A cross-reference, directive or link mystmd could not resolve | Reproduce with the `build` command above; the error names the file and line |
| `build` red only in CI, green locally | Stale `docs/_build` cache locally | `rm -rf docs/_build` and rebuild |
| Preview link 404s | The `publish` job has not finished deploying yet, or the pull request is from a fork — forks never publish | Wait for `publish` to go green; for forks, ask a maintainer to read the branch locally |
| Merge button blocked with everything green | The branch is out of date in a way GitHub cannot merge, or a check never reported | Check the merge box for which requirement is unmet; rebase only if there is a real conflict |

:::{seealso}
New to the project? [Contributing to `xmris`](#contribute-home) is the map — it routes you to the
right page for the *kind* of change you are making, and each of those pages carries its own
checklist.
:::
