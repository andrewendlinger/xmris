---
name: dev-diary
description: Write and reconcile an xmris dev-diary entry — a short, rendered article recording why a change was made and how it actually went. Use at the START of any change that adds vocabulary or a contract, picks between viable approaches, or spans multiple PRs (the entry doubles as the plan overview the user reviews); and again at the END to reconcile it against what was built.
---

# Write an xmris dev-diary entry

A diary entry is a **dated article about one change**, written twice and rendered on the docs site.
It is not a reference page. Its two passes serve two readers who never meet:

| Pass | Written | Reader | Deliverable |
|---|---|---|---|
| **1** | first commit on the branch, straight from the approved plan | **the user, now** — reviewing on the rendered site | one screen: problem → decision → shape → what's assumed |
| **2** | last commit on the branch | whoever asks *"why is it like this?"* later | same entry, corrected, plus what actually changed |

Pass 1 exists because the plan file is precise but heavy — right for executing, wrong for approving.
**Never restate the plan's steps.** If the entry reads like a second plan, it has failed.

**House style lives in `CLAUDE.md` § "Documentation style"** and is not restated here — with one
carve-out: a diary entry is a dated record, so *one home per concept* does **not** bind it. Two
entries months apart may touch the same concept; neither is its home. When a concept needs a
permanent home it graduates into an explainer under `docs/explanation/`, which is
`new-doc-notebook`'s job, not this skill's.

## 1. Assess the triggers, then ask — always

**Never decide this autonomously.** Assess, then put it to the user with `AskUserQuestion` and wait.
The entry is published and costs real effort; whether a change earns one is the user's call.

Triggers worth proposing (any one):

- **New vocabulary or contract** — a new `ATTRS`/`DIMS`/`COORDS`/`VARS` term, decorator, domain
  rule, or accessor namespace. Anything growing the package's conceptual surface.
- **≥2 viable approaches existed** — you had to pick and the rejected option was defensible. These
  get silently re-litigated six months later.
- **Multi-PR chain or cross-cutting refactor.**

Weak candidates: a bug fix, a processing function following existing patterns, a dependency bump.
Still ask if invoked — recommend skipping and name the missing trigger.

The ask is one question, two options: **write an entry / skip**. Name the concrete trigger in the
question ("adds `ATTRS.group_delay` plus a new decorator"), never ask abstractly. If the user skips,
drop it — do not re-ask later in the same change.

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

**Then tell the user to run `uv run docs` and name the page.** That handoff is what makes the entry
a review artifact instead of a file nobody opens.

```{note}
**Pass-1 code is illustrative — do not chase executability.** The API does not exist yet; you are
sketching the call site you *wish* existed, which is design work in its own right. Static
` ```python ` blocks are correct there. What pass 2 owes you is **accuracy, not executability**.
```

### Which MyST feature carries which load

| Job in the argument | Feature |
|---|---|
| Pass state (planned / built) | `{note}` status banner — **required**, first block on the page |
| The decision, in one sentence | `{important}` |
| The shape: states, or a choice a contributor faces | `{mermaid}` |
| A contract surface | markdown table |
| Guardrail, one-way door, footgun | `{warning}` |
| An approach *we* rejected | `:::{dropdown} Why not <X>?` |
| An approach *the reader* would try | paired ❌ / ✅ blocks, on the main line |
| Unproven claim (pass 1 only) | `:::{attention} Assumptions to verify` |

The status banner is not decoration: skills are re-read cold, so it is how a later invocation knows
which pass it is in.

Mermaid escaping rules live in the `new-doc-notebook` skill — quote every label, `<br>` not `\n`,
monospace `<span>` for code inside labels. Copy an existing diagram rather than hand-rolling syntax.

## 3. Rejected alternatives

Split by **who** rejected it:

- **We considered it and dropped it** → `:::{dropdown} Why not <X>?`, off the main line. This is the
  default for everything out of the planning session. Left on the main line it buries the actual
  implementation.
- **The reader would naively try it** → stays on the main line as paired ❌ / ✅ blocks. That is
  `architecture.md`'s "Parameter Soup", and it is pedagogy, not an appendix.

## 4. Pass 2 — reconcile, as the branch's last commit

Re-read the entry **against the merged code**, not from memory. Commit as
`docs: reconcile diary entry for <topic>`. Not re-asked — accepting pass 1 commits to it.

1. Flip the status banner to `built`, linking the merged PRs.
2. **Correct drifted prose in place** so the article never misleads.
3. Empty and **delete** the assumptions block — each item either folded into prose (it held) or
   promoted to a bullet in step 4 (it broke).
4. Add a short closing **`## What changed from the plan`**. This is the payoff: divergence is
   content, not embarrassment. A plan that survives contact with implementation unchanged is rare
   enough that pretending otherwise makes every entry less useful.

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

- **TOC entry in `docs/myst.yml`** under the `Dev Diary` group, at the **top** of `children`
  (newest first — the TOC is hand-maintained). A page missing from the TOC never renders.
- Link any explainer the entry produced with a relative `.md` path; never link `.ipynb`.
- Link the entry from the PR body. It is the summary — do not restate it there.

## Checklist

- [ ] Trigger named and the choice **put to the user**
- [ ] Entry is the branch's first commit (or mid-flight start noted in the PR body)
- [ ] One screen: ≤500 words, no restated plan steps, driving question named in the PR body
- [ ] Status banner present; assumptions in a **rendered** block; rejections in dropdowns
- [ ] User told to run `uv run docs`, with the page named
- [ ] Pass 2 committed last, read against the code — error strings, diagram branches, guardrail
      scopes and snippets all verified against `src/`
- [ ] `git grep -nF "{attention} Assumption" -- 'docs/diary/*.md'` is empty
- [ ] `## What changed from the plan` present and honest
- [ ] TOC entry at the top of the `Dev Diary` group
