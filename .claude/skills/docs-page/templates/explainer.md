# Explainer — `docs/concepts/`

The genre that answers **"why is it like this?"** — the permanent home of a concept, written as a
motivated narrative rather than a reference dump. `domains.md` and `vocabulary.md` are the
exemplars; read one before writing a third.

An explainer is where a concept lands when it outgrows a diary entry. The diary keeps the *story
of arriving* at the decision; the explainer owns the concept from then on.

## Claims must be live cells, not static blocks

An explainer carries frontmatter and a kernelspec, so its code executes — and `deploy.yml` runs
`myst build --execute` on **every PR**. That makes the choice of fence a choice about rot:

- **`code-cell`** for anything asserting how the library *actually behaves*. If it stops being
  true, the PR fails. Prefer this.
- **Static ```` ```python ````** only for code that must *not* run: an API that does not exist yet,
  or a deliberate ❌ anti-pattern.

This is the fix for a real failure. `domains.md` was written mid-design (#76), never reconciled,
and sat wrong on `main` across five merges until two code reviews caught it — quoted error
strings had drifted, and a decision criterion in a table was simply false. Static prose cannot
notice that. A live cell can.

```{warning}
Executed is not the same as *tested*. `test-gen` picks an explainer up only once it carries a
jupytext kernelspec — without one it runs in the docs build with **no nbmake and no coverage**.
And a cell that merely runs proves the API exists and does not raise; proving a *value* still
needs an `assert` in a `remove-cell`.
```

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

(my-concept)=
# The Concept, Named Plainly

<Two or three sentences establishing the ground the reader already stands on. Then the
tension — ideally as the reader's own question, in a blockquote:>

> **My scanner doesn't call it `reference_frequency`. It's `spec_freq`. Now what?**

<One line committing to follow that question. It shapes the whole page.>

(my-concept-problem)=
## The problem

<The tension made concrete. Show the failure, do not describe it — a live cell that raises
is worth a paragraph of prose.>

(my-concept-goal)=
## The goal

:::{important}
<The whole design in one sentence. If it takes two, the page is at the wrong altitude.>
:::

(my-concept-shape)=
## How it works

<The body. A contract table or decision-tree mermaid wherever it beats prose.>

:::{dropdown} Why not <the alternative>?
<What it would have bought, what it cost, why the cost won. Link the issue for the full
deliberation.>
:::

(my-concept-contributors)=
## For contributors: <the mechanism>

<Where `ATTRS`/`DIMS`/`COORDS` and decorator names are allowed — this passage is explicitly
addressed to contributors. Everything above it stays in plain strings.>
````

## What makes these pages work

- **Follow one question the whole way down.** A cold "Why X?" heading makes a sound decision read
  as an assertion to accept. `vocabulary.md` opens on the reader's own objection and never leaves it.
- **Name the tempting wrong answer, then kill it.** `vocabulary.md`'s "The tempting answer (and why
  we didn't)" earns the real design by first making the alias table sound reasonable — then showing
  the same array behaving two ways.
- **Quote error messages verbatim.** Grep the actual string out of `src/`; do not paraphrase. This
  is the single most rot-prone thing on the page, which is the argument for a live cell that
  raises it instead.
- **Put the deep rationale in a `:::{dropdown}`.** Off the main line, where it informs without
  derailing. `vocabulary.md`'s `str`-vs-`Enum` aside is the model.
- **Every article stands alone.** Readers arrive from search and deep links. Declare a hard
  prerequisite in a `seealso` at the top; keep the orienting recap when you thin the page.
- **Guardrails get a `{warning}`,** including what fails loudly and why that is the desired
  behavior.

## Register

TOC group is **Basics** in `docs/myst.yml`, alongside the architecture guide — an explainer is
core reading, not an appendix. Cross-link the tutorial that demonstrates the concept, and the
diary entry that decided it, with relative `.md` paths.
