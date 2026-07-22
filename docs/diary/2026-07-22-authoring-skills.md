(diary-authoring-skills)=
# The skills remember the rules so you don't

:::{note} Status: built — #103
:::

Every change to xmris has to clear the same stack of contracts at once. A new function owes the
[eight Commandments](../contributing/ai_context.md), the right [domain
decorator](../explanation/domains.md), and the canonical [config
vocabulary](../explanation/vocabulary.md). A new docs page owes the four house-style rules. Hold all
of that in your head on every edit and something slips — a hardcoded `"time"`, a missing target, a
boolean flag bloating `.attrs`.

So the knowledge lives in four Claude Code skills under `.claude/skills/`, one per kind of change,
each firing on its own trigger:

| Skill | Fires on |
|---|---|
| `xmr-method` | a library function — a transform, a vendor loader, a fit |
| `xmr-widget` | an interactive widget: a UI over that math |
| `docs-page` | any hand-authored page — tutorial, explainer, or guide |
| `dev-diary` | the decision record for a significant change (this entry) |

:::{important}
None of the four skills *contains* the rules. Each routes to the one canonical doc that owns them —
the skills obey *one home per concept*, the very rule the docs preach.
:::

That routing is the whole design. A rule copied into a skill is a rule that drifts the moment the
doc changes; a rule *pointed at* stays singular.

```{mermaid}
flowchart LR
    M["xmr-method"] --> A["ai_context.md<br/>the 8 Commandments"]
    W["xmr-widget"] --> S["static_widgets.md"]
    W --> A
    P["docs-page"] --> H["CLAUDE.md<br/>§ Documentation style"]
    V["dev-diary"] --> H
```

(diary-authoring-skills-seam)=
## The what, and the why

The four split along one seam. `xmr-method` and `xmr-widget` produce the **what** — the maths and
the UI over it. A widget is never a home for new maths, so it hands any missing method back to
`xmr-method` first. `docs-page` and `dev-diary` produce the **how-to-use** and the **why** — the
tutorial a reader runs, and the dated record of the decision behind it. A change that touches every
layer fires all four in turn; a typo fix fires just `docs-page`.

:::{dropdown} Why not one CONTRIBUTING.md?
One flat doc is the obvious move, and it fails the same way every time: the rules genuinely differ
by change type, a single page gets skimmed rather than read, and the parts nobody is editing rot
silently. Splitting by *kind of change* means each skill carries only what its change needs — and
routes the rest home.
:::

New here? [`ai_context.md`](../contributing/ai_context.md) is the library contract; [the controlled
vocabulary](../explanation/vocabulary.md) is the naming discipline under all of it;
[`static_widgets.md`](../contributing/static_widgets.md) is the widget canon. A stdlib-only checker
(`check_docs.py`) enforces the docs-style half — the backlog it surfaced is [issue
#104](https://github.com/andrewendlinger/xmris/issues/104).
