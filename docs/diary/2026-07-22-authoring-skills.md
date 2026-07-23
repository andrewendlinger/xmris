(diary-authoring-skills)=
# The skills remember the rules so you don't

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-07-23</span>

Every change to xmris has to clear the same stack of contracts at once. A new function for examples has to honour

- [eight Commandments](../contributing/ai_context.md)
- the right [domain decorator](../explanation/domains.md)
- the canonical [config vocabulary](../explanation/vocabulary.md).

Hold all of that in your head on every edit and something slips — a hardcoded `"time"`, a missing target, a
boolean flag bloating `.attrs`.

So the knowledge lives in four author-time Claude Code skills under `.claude/skills/`, one per kind
of change, each firing on its own trigger. So that a contributor without Claude Code is not locked out, each skill
is also surfaced by a matching page in [Contribute](#contribute-home).

:::{important}
None of the skills *contains* the rules. Each routes to the one canonical doc that owns them.
:::

That routing is the whole design: a rule copied into a skill drifts the moment the doc changes; a
rule *pointed at* stays singular. The [Contributing overview](#contribute-home) draws the full map of
which skill defers to which doc.

(diary-authoring-skills-seam)=
## The what, and the why

The four split along one seam. `xmr-method` and `xmr-widget` produce the **what** — the maths and
the UI over it. A widget is never a home for new maths, so it hands any missing method back to
`xmr-method` first. `docs-page` and `dev-diary` produce the **how-to-use** and the **why** — the
tutorial a reader runs, and the dated record of the decision behind it. A change that touches every
layer fires all four in turn; a typo fix fires just `docs-page`.

A fifth skill sits off this seam — `release`, the user-triggered **ship** step — yet it routes to
[Publishing](#contribute-release) exactly as the four route to their docs. That operational axis is
why the group is now **Workflows**, not just the authoring four.

New here? The [Contribute](#contribute-home) section is the map — start there. A stdlib-only checker
(`check_docs.py`) enforces the docs-style half; the backlog it surfaced is [issue
#104](https://github.com/andrewendlinger/xmris/issues/104).

(diary-authoring-skills-changed)=
## What changed from the plan

- **The "why not one `CONTRIBUTING.md`?" argument got half-adopted.** An earlier draft of this entry
  argued that a single flat contributing doc was simply the wrong move — the rules differ by change
  type, one page gets skimmed, and the parts nobody edits rot. That still holds against a *flat* doc,
  but it quietly ignored discoverability: an external contributor — and a JOSS reviewer — looks for a
  `CONTRIBUTING.md` first. So there is now a thin root
  [`CONTRIBUTING.md`](https://github.com/andrewendlinger/xmris/blob/main/CONTRIBUTING.md) and a real
  [Contribute](#contribute-home) section, both routing *by kind of change* rather than flattening the
  rules into one page. The split survived; the "no landing at all" corollary did not.
- **The skills grew a human-facing face.** Each Claude skill now has a Contribute page that live-imports its
  checklist straight from the `SKILL.md`. The skills had been treated as Claude-only; rendering them
  on the site cost one small notebook cell per page and closed the drift gap for good.
- **A fifth skill joined the group.** The plan scoped four author-time skills; the operational
  `release` step lived apart as the [Publishing](#contribute-release) doc plus a `/release` checklist
  that had drifted from it — it bumped the version *before* CI, which the doc forbids. Consolidating
  the two (the doc stays the canonical home, the skill defers to it) folded release into the same
  routing rule and renamed the group *Skills → Workflows*.
