(diary-authoring-skills)=
# The skills remember the rules so you don't

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-21 · #103, #104, #114, #166</span>

Every change to xmris has to clear the same stack of contracts at once. A new function, for
example, has to honour

- the [Architecture Contract](#contract)'s Commandments
- the right [domain decorator](#domains)
- the canonical [config vocabulary](#vocabulary).

Hold all of that in your head on every edit and something slips — a hardcoded `"time"`, a missing
target, a boolean flag bloating `.attrs`.

So the knowledge lives in author-time Claude Code skills under `.claude/skills/`, one per kind of
change, each firing on its own trigger. So that a contributor without Claude Code is not locked
out, each skill is also surfaced by a matching page in [Contribute](#contribute-home) that quotes
its checklist live from the `SKILL.md` — one small cell per page, and the gap between what Claude
enforces and what the site says is closed for good.

:::{important}
None of the skills *contains* the rules. Each routes to the one canonical doc that owns them.
:::

That routing is the whole design: a rule copied into a skill drifts the moment the doc changes; a
rule *pointed at* stays singular. The [Contributing overview](#contribute-home) draws the full map
of which skill defers to which doc.

:::{dropdown} Why not one flat `CONTRIBUTING.md`?
The rules differ by kind of change, a single page gets skimmed, and the parts nobody edits rot.
But discoverability cuts the other way: an external contributor — and a JOSS reviewer — looks for
a `CONTRIBUTING.md` first. So a thin root
[`CONTRIBUTING.md`](https://github.com/andrewendlinger/xmris/blob/main/CONTRIBUTING.md) exists and
routes into the [Contribute](#contribute-home) section *by kind of change* — the split, without
the missing front door.
:::

(diary-authoring-skills-seam)=
## The what, and the why

The skills split along one seam. `xmr-method` and `xmr-widget` produce the **what** — the maths
and the UI over it. A widget is never a home for new maths, so it hands any missing method back to
`xmr-method` first. `docs-page` and `dev-diary` produce the **how-to-use** and the **why** — the
tutorial a reader runs, and the decision log you are reading now. `changelog` produces the **what
shipped** — the one record a reader reaches from PyPI rather than from the sidebar. A sixth skill
sits off this seam: `release`, the user-triggered **ship** step, routes to
[Publishing](#contribute-release) exactly as the five route to their docs — that operational axis
is why the group is **Workflows**, not just the authoring five.

(diary-authoring-skills-diary)=
## The diary keeps the why

The `dev-diary` skill is where this system meets how xmris is actually built. A significant repo
or architectural decision — one where a defensible alternative lost — is offered a diary entry,
and once the work has landed the decision is written up as a one-screen story, checked against
the code as built. Cycles of using the skill taught three refinements, now law in it:

- An entry is proposed for **decisions**, not categories — a new rule, decorator or namespace
  earns one; a vocabulary term that merely names what a new method did does not.
- The entry is written **against the code, not from memory**. The plan file does not survive
  the merge, so the entry absorbs its rationale; a "what changed from the plan" section appears
  only where the divergence itself teaches.
- **One entry per decision, rewritten in place as the decision evolves** — this very article
  absorbed its later turns rather than spawning sequels, and its *Last edited* line says the
  story is current.

The docs-style half has a second enforcer, and it is not a skill. A stdlib-only checker
(`check_docs.py`) measures the rules `myst build` stays silent about, and it first ran as an
advisory: 141 errors across the tree, which is too many to gate on and too many to read. Paying
them down ([#104](https://github.com/andrewendlinger/xmris/issues/104)) was worth doing only
because of what it bought — at zero, `exit 1` became trustworthy, and the checker moved into
`ci-fast.yml` as the **`Docs style`** job. The rules now hold the way the Commandments do: a
missing target is a red build, not a note in a review.

That job turned out to be the first of three layers rather than the whole answer. It measures what
`myst build` never mentions; `--strict` covers what the build *does* report and used to exit 0 on
anyway; and neither can tell you whether a page reads right, which is what a rendered per-pull-request
preview is for — [Every pull request publishes the page it changes](#diary-docs-previews) tells that
story.

New here? The [Contribute](#contribute-home) section is the map — start there.
