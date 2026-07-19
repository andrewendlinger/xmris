---
name: design-doc
description: Write and reconcile the design document for a significant xmris change — the reader-facing explanation of why a decision was made. Use at the START of any change that adds vocabulary or a contract, picks between viable approaches, or spans multiple PRs; and again at the END of that work to reconcile the doc against what was actually built.
---

# Write an xmris design doc

A design doc is a **published explanation article that doubles as the decision record**. It is written twice: once from the plan, before code exists, and once at the end of the work, reconciled against what was actually built. The diff between those two passes is the highest-value review artifact the PR produces — it is where falsified assumptions become visible.

This is not a separate genre from the docs. It ships in `docs/`, in the house voice, useful to a new user who was never in the room. The decision record is a side effect of explaining the thing well.

## 0. Assess the triggers, then ask — always

**Never decide this autonomously.** Assess first, then put the choice to the user with `AskUserQuestion` and wait. Writing a design doc is a real cost, the doc is published, and whether a decision deserves the record is the user's call — not a judgment to infer from the diff.

Triggers that make it worth proposing (any one):

- **New vocabulary or contract** — a new `ATTRS`/`DIMS`/`COORDS`/`VARS` term, a new decorator, a new domain rule, a new accessor namespace. Anything that grows the package's conceptual surface.
- **≥2 viable approaches existed** — you had to pick, and the rejected option was defensible. These are the decisions that get silently re-litigated six months later.
- **Multi-PR chain or cross-cutting refactor** — work spanning more than one PR, where the design must stay coherent across the chain.

Weak candidates: a bug fix, a single processing function following existing patterns, a dependency bump, a docs-only edit. Those are served by `new-processing-method` + `new-doc-notebook`. Still ask if invoked — just recommend skipping, and say which trigger is missing.

Ask the artifact choice and the skip in **one** question, so the user sees the full menu:

```
header:   "Design doc"
question: "<Change X> hits <trigger>. Document the reasoning?"
options:
  - "Explainer + notebook"  → the rule and its executable proof, cross-linked
  - "Prose explainer only"  → docs/explanation/<topic>.md, no kernel
  - "Notebook only"         → docs/notebooks/<category>/<topic>.md
  - "Skip"                  → no design doc for this change
```

Name the concrete trigger in the question ("adds `ATTRS.group_delay` + a new decorator") — never ask abstractly. Mark the option you'd pick `(Recommended)` and lead with it. If the user skips, drop it entirely; do not re-ask later in the same change.

## 1. Which artifact fits

| Change | Artifact | Where |
|---|---|---|
| Cross-cutting concept, not demonstrable in a few cells | Prose explainer (no kernel) | `docs/explanation/<topic>.md` |
| Concept whose payoff is a runnable pipeline | Executable notebook | `docs/notebooks/<category>/<topic>.md` |
| Both — a rule *and* its proof | Explainer + companion notebook | both, cross-linked |

The domain-contract work is the reference for the "both" case: `docs/explanation/domains.md` states the rule, `docs/notebooks/pipeline/domain_contracts.md` proves it with hidden strict asserts. When you write a notebook, **the mechanics come from the `new-doc-notebook` skill** — frontmatter, targets, hidden assert cells, TOC. This skill only governs the *argument* the page makes.

## 2. Pass 1 — draft from the plan (first commit on the branch)

Write it before the implementation, straight off the approved plan, as the branch's first commit:

```
docs: design note for <topic>
```

Committing the draft is the point: the reviewer sees design → implementation → reconcile as three phases of one PR, and pass 2's diff is legible. Do not merge the branch with pass 2 outstanding.

### The skeleton

Follow `domains.md` — it is the shape that worked:

1. **The problem.** What breaks without this, in plain language. Concrete failure, not abstraction.
2. **The goal**, compressed to one sentence in an `{important}` admonition. If you cannot write that sentence, the design is not settled yet — stop and settle it.
3. **The design.** The taxonomy, mechanism, or rule. A mermaid diagram if the structure is branching.
4. **What you get, at a glance.** A table of the contract surface — input × operation → output. This is the part users actually return to.
5. **Guardrails.** One-way doors, failure modes, the exact error text a user hits, the escape hatch (`set_options`, an explicit call).
6. **For contributors.** How to declare or extend the thing — decorator, config term, decision flowchart.

### Rejected alternatives are pedagogy, not an appendix

Do not append a dry "Alternatives Considered" list. House style walks the reader *through* the rejected option and lets them feel why it fails — `architecture.md`'s "❌ The Anti-Pattern: Parameter Soup" → "✅ The xmris Way", `domains.md`'s "either you pepper pipelines with manual conversions … or the library converts silently". Same information, and it teaches instead of filing. The decision record falls out of the explanation.

### Mark unproven claims

Anything the draft asserts that code has not yet demonstrated is an assumption, and assumptions are what pass 2 exists to catch. Tag each one inline:

```markdown
<!-- ASSUMPTION: length-preserving is the criterion separating computes_in from ensures_domain -->
```

HTML comments do not render. Pass 2 must resolve every one and delete the marker; `grep -rn "ASSUMPTION:" docs/` returning nothing is a merge gate. Be honest here — a draft with no assumptions marked usually means they were not looked for.

## 3. Pass 2 — reconcile against the implementation (last commit on the branch)

Re-read the draft **against the merged code**, not from memory. Commit as:

```
docs: reconcile design note for <topic>
```

Pass 2 is **not** re-asked — accepting pass 1 commits to it, and a reconciled doc is the deliverable. The only ask at this stage is when the skill is invoked at the end of work that never had a pass 1: put the same question from §0, noting the doc will be written retroactively and so records the outcome rather than the deliberation.

This pass is not optional polish. In this repo it was skipped once and cost PR #90 — the `domains.md` draft (#76) landed before the rollout (#78/#79) and sat wrong on `main` for four PRs until two independent code reviews caught it. Every category below is a real defect from that incident:

- **Quoted error messages.** The doc quoted a `ValueError` whose wording had since changed. Grep the actual string in `src/` and paste it verbatim.
- **Decision criteria in diagrams and tables.** The draft's flowchart gated on "same physics, length-preserving?" — plausible at design time, and false: `zero_fill` changes length and is still `@computes_in`. Walk every branch of every diagram against the real code and confirm a function takes that path.
- **Over-general guardrails.** "Explicit dims pass through" was stated unconditionally; it holds only for `@computes_in`, not `@ensures_domain`. For each rule, ask which code paths it actually covers, and scope the sentence to those.
- **API surface.** Function names, signatures, defaults, attr keys, accessor spellings — all drift during implementation.
- **Adopted rejections.** If something the draft argued against got built anyway, the rationale now reads backwards. Rewrite it, or delete the passage.

Then verify the claims execute:

```bash
grep -rn "ASSUMPTION:" docs/                 # must be empty
uv run test-gen && uv run pytest "tests/autogen_notebooks/<category>/<topic>.ipynb" -n0 --no-cov
```

Every user-visible behavior the doc promises should be asserted somewhere — in the companion notebook's hidden test cells for demonstrable claims, in `tests/test_core.py` for architectural invariants. A promise in prose with no assert behind it is the next `#90`.

## 4. Multi-PR chains

For work spanning several PRs, the doc lives in the **first** PR of the chain and is reconciled in the **last**. Each intermediate PR that changes behavior the doc describes carries its own reconcile hunk — do not batch them to the end, that is exactly the #76→#90 failure mode. If the chain's design shifts mid-flight, amend the doc in the PR that shifted it, and say so in that PR body.

## 5. Register and link

- **TOC entry in `docs/myst.yml`** with a `title:`. Explanation articles currently sit in the group whose concepts they explain (`domains.md` → Basics). A page missing from the TOC never renders.
- **Cross-link the pair**: explainer → notebook via `[text](#explicit-target)`, notebook → explainer via a relative `.md` path. Never link `.ipynb`.
- Link the doc from the PR body. It is the summary — the PR description should not restate it.

## Final checklist

- [ ] Trigger assessed and the doc/artifact choice **put to the user**, naming the concrete trigger
- [ ] Pass 1 committed as the branch's first commit, before implementation
- [ ] Goal compressed to one sentence in an `{important}` admonition
- [ ] Contract surface stated as a table
- [ ] Rejected alternatives walked through as pedagogy, not appended as a list
- [ ] Pass 2 committed as the branch's last commit, read against the code
- [ ] Error messages, diagram branches, and guardrail scopes verified against `src/`
- [ ] `grep -rn "ASSUMPTION:" docs/` is empty
- [ ] Every promised behavior has an assert behind it
- [ ] TOC entry present; explainer and notebook cross-linked
