# Diary entry skeleton

Copy the structure below into `docs/diary/YYYY-MM-DD-<topic-slug>.md`. Prose only — no jupytext
frontmatter, no kernel. Explicit MyST targets above every header (Commandment 8).

Order is deliberate: the reader gets the story, then the finer points. Sections that carry
nothing for this change get dropped — an entry with an empty guardrail section is padding.

---

````markdown
(diary-<slug>)=
# <Title — a claim or a question, not a topic label>

<span style="color: gray; font-size: 0.9em;">Last edited: YYYY-MM-DD · #NN</span>

<One paragraph. The concrete problem, felt as a tension the reader already has. Not an
abstraction, not a summary of the plan. This is the driving question in prose form.>

:::{important}
<The decision, in one sentence. If it takes two, the entry is at the wrong altitude.>
:::

(diary-<slug>-<section>)=
## <How it works — the shape, not the steps>

<A mermaid diagram or a table wherever it beats prose. Commits and the diff own the
implementation steps; repeating them here produces a second plan. Budget: one screen, ≤500 words
of prose, at most one diagram and one table — flex it only when the story truly needs the room.>

```python
# Optional but encouraged: the call site as it now works — a small user story, not a spec.
# Verify names, signatures and defaults against src/; drop the block if it earns nothing.
spectrum = fid.xmr.to_spectrum().xmr.new_thing(...)
```

:::{dropdown} Why not <the alternative we dropped>?
<Two or three sentences. What it would have bought, what it cost, why the cost won.>
:::
````

An optional closing section — **only when the divergence from the plan itself teaches** (an
instructive failure, or a prior state someone actually saw; with no witnesses, fold the lesson
into the main argument instead):

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
| A concept that outgrew the diary | relative link to its `docs/concepts/` page |
