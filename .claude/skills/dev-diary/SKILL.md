---
name: dev-diary
description: Write an xmris dev-diary entry — a short, rendered article recording why a significant repo or architectural decision was made. Suggest one when a change picks between viable approaches, adds conceptual surface (a rule, decorator, or namespace), or spans multiple PRs; if the user accepts, the entry is written once the work has landed — the story of how it is now and why.
---

# Write an xmris dev-diary entry

A diary entry is a **decision record told as a story**, rendered on the docs site. It is not a
reference page, not a changelog — and not the plan retold: commits and the diff own the steps;
the entry owns the why.

**One entry per decision.** When a later change extends a decision an entry already tells, that
entry is rewritten ground-up into the current story — `Last edited` updated, the new PR number
appended — rather than a sibling entry spawned. A new dated entry is for a new decision; the
reader should never have to join two articles to get one answer.

The `Dev Diary` group also has **one evergreen page** — `docs/diary/index.md` ("A dev diary for
xmris") — that tells readers *what the diary is*. It is **not** an entry: no `Last edited` line,
and it stays pinned at the **top** of the group. This skill governs the dated entries **below**
it; touch `index.md` only when the diary's own workflow changes.

**House style lives in `CLAUDE.md` § "Documentation style"** and is not restated here — with one
carve-out: *one home per concept* binds an entry at the **decision** level, not the concept level.
Two entries may touch the same concept when their decisions differ; neither is the concept's home.
When a concept needs a permanent home it graduates into an explainer under `docs/concepts/`,
which is `docs-page`'s job, not this skill's.

## 1. Assess the triggers, then ask — always

**Never decide this autonomously.** Assess, then put it to the user with `AskUserQuestion` and wait.
The entry is published and costs real effort; whether a change earns one is the user's call.

Triggers worth proposing (any one):

- **≥2 viable approaches existed** — you had to pick and the rejected option was defensible. These
  get silently re-litigated six months later.
- **New conceptual surface** — a new rule, decorator, domain, or accessor namespace. A new
  `ATTRS`/`DIMS`/`COORDS`/`VARS` term only when it required a real choice of its own — not when it
  merely names what a new function did.
- **Multi-PR chain or cross-cutting refactor.**

The key is decision-weight, not category: an entry records a choice that could have gone another
way. Weak candidates: a bug fix, a processing function following existing patterns, a dependency
bump — and **a term that follows an existing pattern** (a lineage attr riding a new method is
vocabulary, not a decision). Still ask if invoked — recommend skipping and name the missing
trigger.

Before proposing a *new* entry, check whether an existing entry already tells this decision's
story. If one does, the proposal becomes **update that entry** — rewritten ground-up per §4 —
instead of writing a sibling.

The ask is one question, two options: **write an entry / skip** (or **update `<entry>` / skip**).
Name the concrete trigger in the question ("adds `@ensures_domain` — a new decorator contract"),
never ask abstractly. If the user skips, drop it — do not re-ask later in the same change.

Ask the moment the trigger is recognized — often during planning. An accepted entry is **noted,
then written once the work has landed**; it is never a precondition for starting, and nothing
waits on it.

## 2. Write the entry — once the change is built

Read `templates/entry.md` and follow its skeleton. File: `docs/diary/YYYY-MM-DD-<topic-slug>.md`.
Commit as `docs: diary entry for <topic>` — normally the last docs commit on the branch, written
when the change is in its final shape.

**Budget: one screen rendered.** ≤500 words of prose, at most one diagram and one table. The budget
is the feature — it forces the entry up to the altitude where the decision is visible.

Open with the **driving question** the change answers, felt as a tension rather than announced as a
cold "Why X?" heading. Write it in the PR body too. If you cannot name it, you have a topic, not an
article.

**Write against the code, not from memory.** The entry is written after the work lands precisely
so it is born accurate. Writing docs from the design instead cost PR #90: `domains.md` was written
mid-design (#76) and never brought back in line with the code, so it sat wrong on `main` across
five merges until two code reviews caught it. Every category below is a real defect from that
incident — verify each against `src/` before committing:

- **Quoted error messages** — wording drifts. Grep the actual string in `src/` and paste it verbatim.
- **Decision criteria in diagrams and tables** — walk every branch against real code (that draft
  gated on "length-preserving?", which is false: `zero_fill` changes length and is still
  `@computes_in`).
- **Over-general guardrails** — scope each rule to the paths it actually covers.
- **API surface and every snippet** — names, signatures, defaults, attr keys all drift.
- **Adopted rejections** — if something the entry argues against got built, the rationale reads
  backwards. Rewrite or delete it.

**Absorb rationale that only the plan held** — decision criteria, rejected options, constraints
discovered on the way — into the main line or a dropdown. The plan file lives outside the repo and
does not survive the merge, so the entry and the PR body are the only reasoning record. *Steps*
stay unrestated: commits and the diff own those.

A closing **`## What changed from the plan`** is *conditional*, not mandatory. Add it only when
the divergence itself teaches — an instructive failure mode, or a prior state real enough that
someone actually saw it (shipped code, a published page). Early in a package's life the
abandoned state usually had no witnesses, and a delta against a plan nobody read confuses more
than it helps — fold the lesson into the main argument instead. When the section does appear,
every bullet states inline what was previously assumed, so it reads without the plan.

### Which MyST feature carries which load

| Job in the argument | Feature |
|---|---|
| When the entry was last touched | a muted `Last edited:` line — **required**, directly under the H1 |
| The decision, in one sentence | `{important}` |
| The shape: states, or a choice a contributor faces | `{mermaid}` |
| The call site, as it now works — a small user story | a static `python` block |
| A contract surface | markdown table |
| Guardrail, one-way door, footgun | `{warning}` |
| An approach *we* rejected | `:::{dropdown} Why not <X>?` |
| An approach *the reader* would try | paired ❌ / ✅ blocks, on the main line |

The exact `Last edited` span — kept muted with an inline style, because MyST's `[text]{.class}`
shorthand does **not** parse here — lives in `templates/entry.md`.

Mermaid escaping rules live in `docs-page`'s `templates/patterns.md` — quote every label, `<br>`
not `\n`, monospace `<span>` for code inside labels. Copy an existing diagram rather than
hand-rolling syntax.

## 3. Rejected alternatives

Split by **who** rejected it:

- **We considered it and dropped it** → `:::{dropdown} Why not <X>?`, off the main line. This is the
  default for everything out of the planning session. Left on the main line it buries the actual
  implementation.
- **The reader would naively try it** → stays on the main line as paired ❌ / ✅ blocks. That is
  `architecture.md`'s "Parameter Soup", and it is pedagogy, not an appendix.

## 4. When a decision spans PRs or evolves

The entry lands with the PR that **completes** the decision. When a later PR changes behavior an
existing entry describes, that PR updates the entry in place — prose rewritten into the current
story, the `Last edited` date refreshed, the new PR number appended — so the article never sits
wrong on `main`.

## 5. Register and link

- **TOC entry in `docs/myst.yml`** under the `Dev Diary` group: **append it at the bottom** of
  `children`, below the pinned `about.md` intro and any earlier entries (chronological, oldest
  first — the TOC is hand-maintained). A page missing from the TOC never renders.
- Link any explainer the entry produced with a relative `.md` path; never link `.ipynb`.
- Link the entry from the PR body. It is the summary — do not restate it there.

## Checklist

<!-- excerpt:start -->
- [ ] Trigger named (decision-weight, not category) and the choice **put to the user** — including
      update-an-existing-entry when one already tells this decision's story
- [ ] Written once the change has landed — never a gate on starting the work
- [ ] One screen: ≤500 words, no restated plan steps, driving question named in the PR body
- [ ] `Last edited` line present with the PR numbers; rejections in dropdowns
- [ ] Written against the code, not from memory — error strings, diagram branches, guardrail
      scopes and snippets all verified against `src/`
- [ ] `## What changed from the plan` only where the divergence teaches — each bullet readable
      without the plan
- [ ] TOC entry appended at the bottom of the `Dev Diary` group (below the pinned intro)
<!-- excerpt:end -->
