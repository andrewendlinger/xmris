---
name: design-doc
description: Write and reconcile the design document for a significant xmris change — the reader-facing explanation of why a decision was made. Use at the START of any change that adds vocabulary or a contract, picks between viable approaches, or spans multiple PRs; and again at the END of that work to reconcile the doc against what was actually built.
---

# Write an xmris design doc

A design doc is a **published explanation article that doubles as the decision record**. It is written twice: once from the plan, before code exists, and once at the end of the work, reconciled against what was actually built. The diff between those two passes is the highest-value review artifact the PR produces — it is where falsified assumptions become visible.

This is not a separate genre from the docs. It ships in `docs/`, useful to a new user who was never in the room. The decision record is a side effect of explaining the thing well.

**House style lives in `CLAUDE.md` § "Documentation style"** — motivated narrative, one home per concept, every article stands alone, the MyST palette carries the argument. Those four rules govern all docs and are not restated here. This skill covers what is specific to a *design* doc: when to write one, the two-pass workflow, and how the four rules apply when the subject is a decision.

## 0. Assess the triggers, then ask — always

**Never decide this autonomously.** Assess first, then put the choice to the user with `AskUserQuestion` and wait. Writing a design doc is a real cost, the doc is published, and whether a decision deserves the record is the user's call — not a judgment to infer from the diff.

Triggers that make it worth proposing (any one):

- **New vocabulary or contract** — a new `ATTRS`/`DIMS`/`COORDS`/`VARS` term, a new decorator, a new domain rule, a new accessor namespace. Anything that grows the package's conceptual surface.
- **≥2 viable approaches existed** — you had to pick, and the rejected option was defensible. These are the decisions that get silently re-litigated six months later.
- **Multi-PR chain or cross-cutting refactor** — work spanning more than one PR, where the design must stay coherent across the chain.

Weak candidates: a bug fix, a single processing function following existing patterns, a dependency bump, **a docs edit that records no new decision**. Those are served by `new-processing-method` + `new-doc-notebook`. Still ask if invoked — just recommend skipping, and say which trigger is missing. (Note the narrowing: a docs-only *PR* can absolutely be a design doc. `vocabulary.md` is one.)

### Composing the question

Ask the artifact choice and the skip in **one** question. `AskUserQuestion` caps at four options, so build the menu from what is actually plausible for this change rather than listing every artifact:

```
header:   "Design doc"
question: "<Change X> hits <trigger>. Document the reasoning?"

# No existing page covers the topic:
  "Explainer + notebook"      → the rule and its executable proof, cross-linked
  "Prose explainer only"      → docs/explanation/<topic>.md, no kernel
  "Notebook only"             → docs/notebooks/<category>/<topic>.md
  "Skip"

# An existing page already covers it — offer this instead of one of the above:
  "Extend <that page>"        → deepen what's there; no new file
```

**Always include the extend option, naming the page, whenever an existing page covers the topic.** Not every decision earns a new file, and extending is the option users forget exists — never let the menu make it unreachable. Name the concrete trigger in the question ("adds `ATTRS.group_delay` + a new decorator"), never ask abstractly. Mark the option you'd pick `(Recommended)` and lead with it. If the user skips, drop it entirely; do not re-ask later in the same change.

## 1. Which artifact fits

| Change | Artifact | Where |
|---|---|---|
| Cross-cutting concept, not demonstrable in a few cells | Prose explainer (no kernel) | `docs/explanation/<topic>.md` |
| Concept whose payoff is a runnable pipeline | Executable notebook | `docs/notebooks/<category>/<topic>.md` |
| Both — a rule *and* its proof | Explainer + companion notebook | both, cross-linked |
| Concept that deepens something already documented | **Edits to the existing page** | wherever it already lives |

The domain-contract work is the reference for the "both" case: `docs/explanation/domains.md` states the rule, `docs/notebooks/pipeline/domain_contracts.md` proves it with hidden strict asserts. `docs/explanation/vocabulary.md` is the reference for consolidation — it absorbed `architecture.md`'s lowercase-convention section, and `architecture.md` got shorter.

When you write a notebook, **the mechanics come from the `new-doc-notebook` skill** — frontmatter, targets, hidden assert cells, TOC.

## 2. Placing the concept

CLAUDE.md's *one home per concept* rule decides where each piece goes. Two things specific to design docs:

- **Survey by concept, not by TOC position.** Grep `docs/` for the terms your decision touches and read every page that already discusses them, wherever it sits in the tree. Adjacency in `myst.yml` is not a proxy for overlap — a `vendor/` page can duplicate a `basics/` concept.
- **Mechanics can pin content in place.** Anything that only renders in an *executed* notebook — `_repr_html_` tables, plots, live widget output — stays in a notebook page regardless of where it conceptually belongs. That is a rendering constraint, not an ownership claim.

For the prerequisite `seealso` that CLAUDE.md requires on dependent pages, note the link is relative to *your* page: from `docs/explanation/` it is `domains.md`; from `docs/notebooks/pipeline/` it is `../../explanation/domains.md`, as `domain_contracts.md` writes it.

## 3. Pass 1 — draft from the plan (first commit on the branch)

Write it before the implementation, straight off the approved plan, as the branch's first commit:

```
docs: design note for <topic>
```

Committing the draft is the point: the reviewer sees design → implementation → reconcile as three phases of one PR, and pass 2's diff is legible.

**Invoked mid-work?** If the branch already has implementation commits, do not rewrite history to fake a first commit. Commit the draft now, run pass 2 as usual before the PR merges, and note in the PR body that the draft was written mid-flight. You lose part of the draft-vs-final diff; you keep the reconcile, which is the part that catches defects.

### The driving question

CLAUDE.md requires a motivated narrative. For a design doc, the driving question is the one a reader brings *to this decision* — from the page's declared prerequisite where it has one, otherwise from the concept itself. Never from TOC adjacency.

- `vocabulary.md`: *"My scanner doesn't call it `reference_frequency`. It's `spec_freq`. Now what?"* Why not aliases, conform-your-data, why terms are frozen — each falls out of a tension that one question raises.
- `domains.md`: *"What happens when I call a spectral operation on a FID?"*

Write the driving question down in the PR body before drafting. If you cannot name it, you have a topic, not an article — and you will produce section-per-topic prose by default.

**Beats, not a template.** `domains.md` passes through: the problem (concrete failure, not abstraction) → the goal, compressed to one sentence in an `{important}` admonition → the design (taxonomy or rule) → the contract surface at a glance → guardrails (one-way doors, exact error text, the escape hatch) → how contributors declare or extend it. Those are waypoints, not headings to fill, and not all of them apply — a vocabulary explainer has no contract surface. The driving question determines the actual headings and their order.

### Which MyST feature carries which load

CLAUDE.md requires the palette to do real work. For a design doc specifically:

| Job in the argument | Feature | In the wild |
|---|---|---|
| The choice a contributor has to make | `{mermaid}` flowchart | `domains.md`'s `@ensures_domain` / `@computes_in` decision tree |
| Structural relation between states | `{mermaid}` graph | the time ↔ spectral converter diagram |
| The contract surface | markdown table | `domains.md`, "What you get, at a glance" |
| The goal, in one sentence | `{important}` | `domains.md` |
| Guardrail, one-way door, footgun | `{warning}` / `{note}` | `bruker_filter_removal.md` |
| Deep or tangential rationale | `:::{dropdown}` | `fft.md`'s "Under the Hood: No Magic Strings" |
| Rejected approach vs house way | paired ❌ / ✅ code blocks | `architecture.md`'s Parameter Soup |
| The math being demonstrated | `$…$` / `$$…$$` | `hz_and_ppm.md` |
| A **load-bearing** claim (pass 2 only) | `{code-cell}` + hidden assert | `domain_contracts.md` |

Mermaid escaping rules live in the `new-doc-notebook` skill — quote every label, `<br>` not `\n`, monospace `<span>` for code inside labels. Copy an existing diagram rather than hand-rolling syntax.

```{note}
**Pass-1 code is illustrative by definition — do not chase executability.** The API does not exist yet. You are sketching the call site you *wish* existed, which is design work in its own right and often the fastest way to feel whether an API is ergonomic. Static ` ```python ` blocks are correct there, and plenty of snippets stay static forever: ❌ anti-pattern blocks (which must *never* run), pseudo-code, quoted error text.

What pass 2 owes you is **accuracy, not executability** — the two are easy to conflate. Re-read every snippet against the built API and fix what drifted. Promote a snippet to a `{code-cell}` only where the claim is load-bearing and a reader would be genuinely misled if it were wrong. That is a handful of snippets, in the companion notebook — not all of them, and not the default.
```

### Rejected alternatives are pedagogy, not an appendix

Do not append a dry "Alternatives Considered" list. House style walks the reader *through* the rejected option and lets them feel why it fails — `architecture.md`'s "❌ The Anti-Pattern: Parameter Soup" → "✅ The xmris Way", `domains.md`'s "either you pepper pipelines with manual conversions … or the library converts silently". Same information, and it teaches instead of filing. The decision record falls out of the explanation.

### Mark unproven claims

Anything the draft asserts that code has not yet demonstrated is an assumption, and assumptions are what pass 2 exists to catch. Tag each one inline:

```markdown
<!-- ASSUMPTION: length-preserving is the criterion separating computes_in from ensures_domain -->
```

HTML comments do not render. Pass 2 must resolve every one and delete the marker. Be honest here — a draft with no assumptions marked usually means they were not looked for.

## 4. Pass 2 — reconcile against the implementation (last commit on the branch)

Re-read the draft **against the merged code**, not from memory. Commit as:

```
docs: reconcile design note for <topic>
```

Pass 2 is **not** re-asked — accepting pass 1 commits to it, and a reconciled doc is the deliverable. The only ask at this stage is when the skill is invoked at the end of work that never had a pass 1: put the same question from §0, noting the doc will be written retroactively and so records the outcome rather than the deliberation.

This pass is not optional polish. In this repo it was skipped once and cost PR #90 — and the real history is worth knowing, because it is messier than the tidy version. The domain-contract **engine landed first** (#77), the `domains.md` explanation followed (#76), then the rollout (#78) and strict mode (#79). So the doc was written against a design already in motion, and never reconciled after it settled: it sat wrong on `main` across five subsequent merges (#78, #79, #86, #51, #91) until two independent code reviews caught it in #90. Every category below is a real defect from that incident:

- **Quoted error messages.** The doc quoted a `ValueError` whose wording had since changed. Grep the actual string in `src/` and paste it verbatim.
- **Decision criteria in diagrams and tables.** The draft's flowchart gated on "same physics, length-preserving?" — plausible at design time, and false: `zero_fill` changes length and is still `@computes_in`. Walk every branch of every diagram against the real code and confirm a function takes that path.
- **Over-general guardrails.** "Explicit dims pass through" was stated unconditionally; it holds only for `@computes_in`, not `@ensures_domain`. For each rule, ask which code paths it actually covers, and scope the sentence to those.
- **API surface, and every code snippet.** Function names, signatures, defaults, attr keys, accessor spellings all drift during implementation, and pass-1 snippets were written against an API that did not exist yet. Read each one against the built code and correct it. Correctness is the bar — a snippet does not have to become executable to be right.
- **Adopted rejections.** If something the draft argued against got built anyway, the rationale now reads backwards. Rewrite it, or delete the passage.

Then verify:

```bash
git grep -n "ASSUMPTION:" -- 'docs/**/*.md'    # must be empty
```

Use `git grep` scoped to tracked `.md`, **not** `grep -r docs/` — `docs/` is ~9,800 files, 99% of them gitignored `_build/` artifacts that retain HTML comments from earlier previews, so a recursive grep keeps failing long after the source is clean.

Notebook-flavor docs only, to confirm the cells execute:

```bash
uv run test-gen && uv run pytest "tests/autogen_notebooks/<category>/<topic>.ipynb" -n0 --no-cov
```

Where a promised behavior is demonstrable, back it with an assert — hidden test cells in the companion notebook, or `tests/test_core.py` for architectural invariants. A demonstrable promise with nothing behind it is the next `#90`. Claims that are genuinely prose (a naming rationale, a rejected alternative) need no test.

## 5. Multi-PR chains

For work spanning several PRs, the doc lives in the **first** PR of the chain and is reconciled in the **last** — so the first PR *does* merge with pass 2 outstanding, by design. Each intermediate PR that changes behavior the doc describes carries its own reconcile hunk; do not batch them to the end, that is exactly the #76→#90 failure mode. If the chain's design shifts mid-flight, amend the doc in the PR that shifted it, and say so in that PR body.

For a single-PR change, the opposite holds: do not merge with pass 2 outstanding.

## 6. Register and link

- **TOC entry in `docs/myst.yml`** with a `title:`. Explanation articles sit in the group whose concepts they explain (`domains.md`, `vocabulary.md` → Basics). A page missing from the TOC never renders.
- **Cross-link the pair**: explainer → notebook via `[text](#explicit-target)`, notebook → explainer via a relative `.md` path. Never link `.ipynb`.
- Link the doc from the PR body. It is the summary — the PR description should not restate it.

## Final checklist

- [ ] Trigger assessed and the choice **put to the user**, naming the concrete trigger; extend-existing-page offered when a page covers the topic
- [ ] Pass 1 committed as the branch's first commit (or mid-flight status noted in the PR body)
- [ ] **Driving question named in the PR body** — and no decision announced as a cold "Why X?" heading
- [ ] MyST palette used where it carries the argument; nothing decorative
- [ ] Rejected alternatives walked through as pedagogy, not appended as a list
- [ ] Goal in an `{important}` admonition, and contract surface as a table — *where the page has them*
- [ ] Pass 2 committed as the branch's last commit, read against the code
- [ ] Error messages, diagram branches, guardrail scopes, and every snippet verified against `src/`
- [ ] `git grep -n "ASSUMPTION:" -- 'docs/**/*.md'` is empty
- [ ] Demonstrable promises have an assert behind them
- [ ] Concepts have a single home; pages you thinned still read whole; moves called out in the PR body
- [ ] TOC entry present; explainer and notebook cross-linked
