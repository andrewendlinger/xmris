# MyST pattern library

Devices mined from the xmris pages that work, each pointing at its live example. Copy the example
rather than hand-rolling — several of these were expensive to get right.

Reach for one only where it **carries the argument**. A decision tree drawn as a flowchart is
checkable at a glance; the same tree in prose is not. Nothing decorative.

## Structuring the argument

| Job | Device | Live example |
|---|---|---|
| The whole design in one sentence | `:::{important}` | `explanation/domains.md` § The goal |
| What you get, at a glance | contract table, input × output | `explanation/domains.md` § What you get |
| Tiers that differ in cost | table with a cost column | `pipeline/domain_agnostic_autophase.md` |
| A choice a contributor faces | `{mermaid}` decision tree | `explanation/domains.md` § For contributors |
| One call traced through every layer | `{mermaid}` flowchart TD | `basics/architecture.md` § Putting It All Together |
| Guardrail, one-way door, footgun | `:::{warning}` + verbatim error | `explanation/domains.md` § Guardrails |
| Hard prerequisite | `:::{seealso}` at the top | `basics/architecture.md` → `vocabulary.md` |

### Open on the reader's own question

`explanation/vocabulary.md` states the objection as a blockquote and then follows it the whole way
down, instead of announcing a topic:

```markdown
> **My scanner doesn't call it `reference_frequency`. It's `spec_freq`. Now what?**

That single question shapes the whole vocabulary design. Let's follow it.
```

### Name the tempting wrong answer, then kill it

Also `vocabulary.md`. The section is titled *"The tempting answer (and why we didn't)"*, makes the
alias table sound reasonable, shows the same array behaving two ways, and lands on:

```markdown
❌ **The road not taken:** bend the vocabulary to fit the data.
...
✅ **The rule:** move the data to the vocabulary, not the vocabulary to the data.
```

### ❌ anti-pattern → ✅ solution, on the main line

`basics/architecture.md` § "The Parameter Soup Problem" shows the bad pipeline first, explains the
three specific costs in a `{warning}` admonition, then shows the xarray version.

This stays **on the main line** — it is pedagogy, not an appendix. The split is by *who* rejected
the option, and it matches the rule `dev-diary` already uses:

- **The reader would naively try it** → paired ❌ / ✅ blocks, inline.
- **We considered and dropped it** → `:::{dropdown} Why not <X>?`, off the main line.

## Dropdowns — three distinct jobs

| Job | Example |
|---|---|
| Background an expert skips, a beginner needs | `architecture.md` — "What's a decorator?", "What is a singleton?" |
| Deliberation, with the issue linked for the full record | `vocabulary.md` — "why are these `str` objects and not an `Enum`?" (issue #65) |
| Boilerplate the reader should not have to scroll past | `domain_agnostic_autophase.md` — "Imports & a small plotting helper" |

The third is worth the habit: a long plotting helper inline pushes the actual point below the fold.

## Math

LaTeX for the math **actually being demonstrated**, not for decoration. `$$…$$` display for the
equation the page is about, `$…$` inline for symbols in prose:

```markdown
$$
S_\text{phased}(\nu) = S(\nu)\, \exp\!\left[i\left(p_0 + p_1\,\frac{\nu - \nu_\text{pivot}}{\Delta\nu}\right)\right].
$$
```

Complexity claims read well in a table cell: `$O(N \log N)$`.

## Mermaid

Escaping is where these break. The existing diagrams were expensive to develop — copy one.

- **Always double-quote** node and edge labels: `A["label"]`, `B{"decision?"}`, `-->|"edge label"|`.
- **Line breaks inside labels: `<br>`**, never `\n`.
- **Code inside labels:** `<span style='font-family:monospace;'>fid.xmr.to_spectrum()</span>` —
  markdown backticks do **not** render inside mermaid labels.
- **Single quotes** for HTML attributes and for literal quotes inside labels (`dims='time'`);
  never nest unescaped double quotes.
- Prefer the ```` ```{mermaid} ```` directive. Bare ```` ```mermaid ```` fences also render, but
  the repo is split 8/5 between them and the directive is the house style — the checker warns.
- For sophisticated styling (`classDef`, `subgraph`, styled spans) copy the reference diagram in
  `docs/index.md`; for simple decision flowcharts copy `explanation/domains.md`.

## Admonition syntax

Both `:::{note}` and ```` ```{note} ```` fence styles work; `:::` is preferred for anything
containing a code fence, since nesting backticks inside backticks is fragile. Write `:::{note}`
with no space after the colons.

A title goes on the opening line — `:::{note} Naming convention` — and `docs/index.md` uses this
well. `:::{admonition} Custom title` + `:class: warning` gives an arbitrary title on a styled box,
as in `architecture.md` § "Why is this bad?".
