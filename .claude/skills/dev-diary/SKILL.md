---
name: dev-diary
description: Write and reconcile an xmris dev-diary entry — a short, rendered article recording why a change was made and how it actually went. Use at the START of any change that picks between viable approaches, adds conceptual surface (a rule, decorator, or namespace), or spans multiple PRs — the entry is the review gate the user reads before implementation begins; and again at the END to reconcile it into the story of how it is now.
---

# Write an xmris dev-diary entry

A diary entry is a **decision record told as a story**, written twice and rendered on the docs
site. It is not a reference page and not a changelog. Its two passes serve two readers who never
meet:

| Pass | Written | Reader | Deliverable |
|---|---|---|---|
| **1** | first commit on the branch, straight from the approved plan | **the user, now** — reviewing on the rendered site *before implementation starts* | one screen: problem → decision → shape → what's assumed |
| **2** | last commit on the branch | whoever asks *"why is it like this?"* later | the same entry, rewritten into how it is now and why |

**One entry per decision.** When a later change extends a decision an entry already tells, that
entry is rewritten ground-up into the current story — `Last edited` updated, the new PR number
appended — rather than a sibling entry spawned. A new dated entry is for a new decision; the
reader should never have to join two articles to get one answer.

The `Dev Diary` group also has **one evergreen page** — `docs/diary/about.md` ("A dev diary for
xmris") — that tells readers *what the diary is*. It is **not** an entry: no `Last edited` line, no
assumptions block, no reconciliation, and it stays pinned at the **top** of the group. This skill
governs the dated entries **below** it; touch `about.md` only when the diary's own workflow changes.

Pass 1 exists because the plan file is precise but heavy — right for executing, wrong for approving.
**Never restate the plan's steps.** If the entry reads like a second plan, it has failed.

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

## 2. Pass 1 — from the plan, as the branch's first commit

Read `templates/entry.md` and follow its skeleton. File: `docs/diary/YYYY-MM-DD-<topic-slug>.md`.
Commit as `docs: diary entry for <topic>`.

**Budget: one screen rendered.** ≤500 words of prose, at most one diagram and one table. The budget
is the feature — it forces the entry up to the altitude where the decision is visible.

Open with the **driving question** the change answers, felt as a tension rather than announced as a
cold "Why X?" heading. Write it in the PR body too. If you cannot name it, you have a topic, not an
article.

Mark everything the plan asserts but code has not yet demonstrated in **one consolidated, visible
block** near the end:

```markdown
:::{attention} Assumptions to verify
- `zero_fill` is the only length-changing op that stays `@computes_in`.
:::
```

It must *render* — HTML comments are invisible on the page the user actually reads, which is the
only moment an assumption matters. Inline `:::{attention} Assumption` boxes only where one qualifies
a single specific passage. A pass-1 entry with no assumptions marked usually means none were looked
for.

**Then stop — the draft is the gate.** Commit, name the page, tell the user to run `uv run docs`,
and **end the turn**. Implementation, verification, further commits — everything waits until the
user responds; a bare "go" is approval. This binds in auto-accept mode too: the handoff is the
last thing in the turn, with nothing queued behind it. Rolling past an unreviewed draft defeats
the entry's purpose, which is catching a bad decision *before* it is built.

```{note}
**Pass-1 code is illustrative — do not chase executability.** The API does not exist yet; you are
sketching the call site you *wish* existed, which is design work in its own right. Static
` ```python ` blocks are correct there. What pass 2 owes you is **accuracy, not executability**.
```

### Which MyST feature carries which load

| Job in the argument | Feature |
|---|---|
| When the entry was last touched | a muted `Last edited:` line — **required**, directly under the H1 |
| The decision, in one sentence | `{important}` |
| The shape: states, or a choice a contributor faces | `{mermaid}` |
| The call site you wish existed (pass 1) | a static `python` block — a small user story |
| A contract surface | markdown table |
| Guardrail, one-way door, footgun | `{warning}` |
| An approach *we* rejected | `:::{dropdown} Why not <X>?` |
| An approach *the reader* would try | paired ❌ / ✅ blocks, on the main line |
| Unproven claim (pass 1 only) | `:::{attention} Assumptions to verify` |

There is no rendered "planned / built" banner. A cold re-invocation tells the passes apart
structurally: an open `:::{attention} Assumptions` block means pass 1 is still outstanding; its
absence means pass 2 has landed. The exact `Last edited` span — kept muted with an inline style,
because MyST's `[text]{.class}` shorthand does **not** parse here — lives in `templates/entry.md`.

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

## 4. Pass 2 — reconcile into the story of how it is now

Re-read the entry **against the merged code**, not from memory. Commit as
`docs: reconcile diary entry for <topic>`. Not re-asked — accepting pass 1 commits to it.

The deliverable is a coherent article about **how it is now and why** — not the draft plus
patches, and not a delta log. The plan file lives outside the repo and does not survive the merge,
so after this commit the entry and the PR body are the only reasoning record:

1. Update the `Last edited` line to the reconcile date, appending the merged PR numbers.
2. **Rewrite drifted prose in place** — real paths, real snippets, the argument as you would make
   it *having now built it* — so the article never misleads.
3. Empty and **delete** the assumptions block — each item folded into the story, whether it held
   or broke.
4. **Absorb rationale that only the plan held** — decision criteria, rejected options, constraints
   discovered on the way — into the main line or a dropdown. *Steps* stay unrestated: commits and
   the diff own those; the entry owns the why.
5. A closing **`## What changed from the plan`** is *conditional*, not mandatory. Add it only when
   the divergence itself teaches — an instructive failure mode, or a prior state real enough that
   someone actually saw it (shipped code, a published page). Early in a package's life the
   abandoned state usually had no witnesses, and a delta against a plan nobody read confuses more
   than it helps — fold the lesson into the main argument instead. When the section does appear,
   every bullet states inline what was previously assumed, so it reads without the plan.

Skipping this pass cost PR #90: `domains.md` was written mid-design (#76) and never reconciled, so
it sat wrong on `main` across five merges until two code reviews caught it. Every category below is
a real defect from that incident:

- **Quoted error messages** — wording drifts. Grep the actual string in `src/` and paste it verbatim.
- **Decision criteria in diagrams and tables** — the draft gated on "length-preserving?", which is
  false (`zero_fill` changes length and is still `@computes_in`). Walk every branch against real code.
- **Over-general guardrails** — "explicit dims pass through" held only for `@computes_in`. Scope each
  rule to the paths it actually covers.
- **API surface and every snippet** — names, signatures, defaults, attr keys all drift.
- **Adopted rejections** — if something the entry argued against got built, the rationale now reads
  backwards. Rewrite or delete it.

```bash
git grep -nF "{attention} Assumption" -- 'docs/diary/*.md'   # must be empty
```

Use `git grep` scoped to tracked files, **not** `grep -r docs/` — `docs/_build/` is ~9,800
gitignored files that retain stale markers from earlier previews.

## 5. Multi-PR chains

The entry lives in the **first** PR and is reconciled in the **last** — so the first PR merges with
pass 2 outstanding, by design. Each intermediate PR that changes described behavior carries its own
reconcile hunk; batching them to the end is exactly the #76→#90 failure. For a single-PR change the
opposite holds: do not merge with pass 2 outstanding.

**Invoked mid-work?** Do not rewrite history to fake a first commit. Commit the entry now, run
pass 2 as usual, and note the mid-flight start in the PR body.

## 6. Register and link

- **TOC entry in `docs/myst.yml`** under the `Dev Diary` group: **append it at the bottom** of
  `children`, below the pinned `about.md` intro and any earlier entries (chronological, oldest
  first — the TOC is hand-maintained). A page missing from the TOC never renders.
- Link any explainer the entry produced with a relative `.md` path; never link `.ipynb`.
- Link the entry from the PR body. It is the summary — do not restate it there.

## Checklist

<!-- excerpt:start -->
- [ ] Trigger named (decision-weight, not category) and the choice **put to the user** — including
      update-an-existing-entry when one already tells this decision's story
- [ ] Entry is the branch's first commit (or mid-flight start noted in the PR body)
- [ ] One screen: ≤500 words, no restated plan steps, driving question named in the PR body
- [ ] `Last edited` line present; assumptions in a **rendered** block; rejections in dropdowns
- [ ] **Pass 1 ends the turn**: page named, `uv run docs` handoff given, implementation not started
- [ ] Pass 2 committed last, rewritten into how it is now and read against the code — error
      strings, diagram branches, guardrail scopes and snippets all verified against `src/`
- [ ] `git grep -nF "{attention} Assumption" -- 'docs/diary/*.md'` is empty
- [ ] `## What changed from the plan` only where the divergence teaches — each bullet readable
      without the plan
- [ ] TOC entry appended at the bottom of the `Dev Diary` group (below the pinned intro)
<!-- excerpt:end -->
