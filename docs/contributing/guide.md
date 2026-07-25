(contribute-home)=
# Contributing to `xmris`

Welcome! `xmris` is built on a strict **"xarray in, xarray out"** philosophy: the pipeline is
functional and N-dimensional, and every function preserves the physics metadata xarray carries.
Contributing means adding to that pipeline without breaking its guarantees.

Rather than one long checklist that applies unevenly, the rules here are organised by **the kind of
change you are making**. Find your row and follow the page it points to:

| You are adding… | Start here | It defers to |
|---|---|---|
| A library function — transform, loader, or fit | [Add a processing method](#contribute-methods) | [The Architecture Contract](./contract.md), [The Two Domains](../explanation/domains.md) |
| An interactive widget — a UI over the maths | [Add a widget](#contribute-widget) | [The Architecture Contract](./contract.md) |
| A docs page — tutorial, explainer, or guide | [Write a docs page](#contribute-docs) | Documentation style |
| The record of a significant decision | [Write a dev-diary entry](#contribute-dev-diary) | [A dev diary for xmris](#diary-about) |

Each of those pages carries a **live checklist**, rendered straight from the Claude Code skill that
automates that kind of change — so whether you work by hand or with Claude, you follow the same,
always-current rules.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    C1["Library function"] --> M["xmr-method"]
    C2["Widget"] --> W["xmr-widget"]
    C3["Docs page"] --> P["docs-page"]
    C4["Decision record"] --> V["dev-diary"]
    M --> A["Architecture Contract — the Commandments"]
    W --> A
    P --> H["Documentation style"]
    V --> H
```

:::{note}
**For Claude Code users:** four of the skills under
[`.claude/skills/`](https://github.com/andrewendlinger/xmris/tree/main/.claude/skills) fire on the
matching change above; a fifth, user-triggered `release` skill drives [cutting a
release](#contribute-release). None carries rules of its own — each routes to the one canonical doc
that owns it, obeying the same "one home per concept" rule these docs preach.
:::

(contribute-home-first)=
## Before your first change

1. [**Set up your environment**](./setup.md) — clone the repo, run `uv sync --all-extras --dev`, and
   confirm `uv run test` is green.
2. **Make your change**, following the page for its kind above. A significant decision starts as a
   [dev-diary draft](#contribute-dev-diary) that gets reviewed before the code is written.
3. [**Open a pull request**](#contribute-pr) against `main` — `main` takes no direct pushes. Four
   checks gate the merge: the test suite on Python 3.10 and 3.13, the docs style checker, and an
   executed build of this documentation, so a broken notebook fails your branch rather than `main`.
   That build also publishes your change as a **live preview site**, which is where anything
   reader-facing gets reviewed.
4. **Drive it green, then hand off** — a maintainer reviews and merges. Cutting a release is a
   separate, maintainer-run workflow: [Publishing](#contribute-release).

Where any contributor page differs from [The Architecture Contract](./contract.md), that page
wins: it is the authoritative rule set, and the skills defer to it too.
