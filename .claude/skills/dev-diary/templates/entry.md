# Diary entry skeleton

Copy the structure below into `docs/diary/YYYY-MM-DD-<topic-slug>.md`. Prose only — no jupytext
frontmatter, no kernel. Explicit MyST targets above every header (Commandment 8).

Order is deliberate: the reader gets the story, then what is still uncertain. Sections that carry
nothing for this change get dropped — an entry with an empty guardrail section is padding.

---

````markdown
(diary-<slug>)=
# <Title — a claim or a question, not a topic label>

:::{note} Status: planned — branch `<branch-name>`
:::

<One paragraph. The concrete problem, felt as a tension the reader already has. Not an
abstraction, not a summary of the plan. This is the driving question in prose form.>

:::{important}
<The decision, in one sentence. If it takes two, the entry is at the wrong altitude.>
:::

(diary-<slug>-<section>)=
## <How it works — the shape, not the steps>

<A mermaid diagram or a table wherever it beats prose. The plan file owns the implementation
steps; repeating them here produces a second plan. Budget: one screen, ≤500 words of prose,
at most one diagram and one table.>

:::{dropdown} Why not <the alternative we dropped>?
<Two or three sentences. What it would have bought, what it cost, why the cost won.>
:::

:::{attention} Assumptions to verify
- <Something the plan asserts that no code has demonstrated yet.>
- <Be honest — an entry with none usually means none were looked for.>
:::
````

---

## Pass 2 — what changes

- Banner → `:::{note} Status: built — #101, #104` (link the merged PRs).
- Drifted prose corrected **in place**, so the article never misleads.
- The `{attention}` block **deleted**: each item folded into prose (it held) or promoted to a
  bullet below (it broke).
- A closing section appended:

````markdown
(diary-<slug>-changed)=
## What changed from the plan

- <Where reality diverged, and what that revealed. This is the diary's payoff — a plan that
  survived unchanged is rare enough that pretending otherwise devalues every entry.>
````

## Blocks worth reaching for

Only where they carry the argument — nothing decorative.

| Job | Block |
|---|---|
| Guardrail, one-way door, footgun | `:::{warning}` |
| An approach *the reader* would naively try | paired ❌ / ✅ code blocks, on the main line |
| A concept that outgrew the diary | relative link to its `docs/explanation/` page |
