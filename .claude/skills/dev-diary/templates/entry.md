# Diary entry skeleton

Copy the structure below into `docs/diary/YYYY-MM-DD-<topic-slug>.md`. Prose only — no jupytext
frontmatter, no kernel. Explicit MyST targets above every header (Commandment 8).

Order is deliberate: the reader gets the story, then what is still uncertain. Sections that carry
nothing for this change get dropped — an entry with an empty guardrail section is padding.

---

````markdown
(diary-<slug>)=
# <Title — a claim or a question, not a topic label>

<span style="color: gray; font-size: 0.9em;">Last edited: YYYY-MM-DD</span>

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

```python
# Optional but encouraged: the call site you wish existed — a small user story, not a spec.
# Pass-1 code is illustrative (the API may not exist yet); drop the block if it earns nothing.
spectrum = fid.xmr.to_spectrum().xmr.new_thing(...)
```

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

The 500-word budget is a **pass-1** figure. Pass 2 legitimately exceeds it: the reconciled story
is what the entry is for.

- `Last edited` line → the reconcile date with the merged PR numbers appended
  (`Last edited: 2026-01-15 · #101, #104`).
- Drifted prose rewritten **in place** — the deliverable is the story of how it is now and why,
  not the draft plus patches.
- The `{attention}` block **deleted**: each item folded into the story, whether it held or broke.
- Rationale only the plan held — decision criteria, rejected options, discovered constraints —
  absorbed into the main line or a dropdown; the plan file does not survive the merge.
- A closing section **only when the divergence itself teaches** (an instructive failure, or a
  prior state someone actually saw — with no witnesses, fold the lesson into the main argument
  instead):

````markdown
(diary-<slug>-changed)=
## What changed from the plan

- <The plan assumed X; reality showed Y, which revealed Z. Self-contained — the reader has no
  plan to consult.>
````

## Blocks worth reaching for

Only where they carry the argument — nothing decorative.

| Job | Block |
|---|---|
| Guardrail, one-way door, footgun | `:::{warning}` |
| An approach *the reader* would naively try | paired ❌ / ✅ code blocks, on the main line |
| A concept that outgrew the diary | relative link to its `docs/explanation/` page |
