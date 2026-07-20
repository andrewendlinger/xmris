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
| Concept that deepens something already documented | **Edits to the existing page** | wherever it already lives |

That last row is real and easy to miss: not every design decision earns a new file. Extending and sharpening an existing explainer is often the better outcome, and it is always in scope — see §2.

The domain-contract work is the reference for the "both" case: `docs/explanation/domains.md` states the rule, `docs/notebooks/pipeline/domain_contracts.md` proves it with hidden strict asserts. When you write a notebook, **the mechanics come from the `new-doc-notebook` skill** — frontmatter, targets, hidden assert cells, TOC. This skill governs the *argument* the page makes and the shape it makes it in.

## 2. One home per concept — edit existing pages freely

Before writing, read the pages a reader would hit *before* yours. Then place each concept where it belongs, **consolidating in whichever direction fits best** — "where does this belong?" beats "who had it first."

- If a section of an existing page is really a decision your new page owns, **move it there and thin the original**, leaving a cross-link that resolves to the new anchor. `vocabulary.md` took `architecture.md`'s lowercase-convention subsection for exactly this reason, and `architecture.md` got shorter.
- If your new material deepens something already explained well, **extend that page instead of starting a new one**.
- Concise it here, extend it there. Editing other docs is expected work in a design-doc PR, not scope creep — say what you moved in the PR body.

Two limits on the knife:

- **Every article must still read on its own** — see below. Consolidation trims duplicated *explanation*, never the orienting sentence that keeps the origin page whole.
- **Mechanics can pin content in place.** Anything that only renders in an *executed* notebook — `_repr_html_` tables, plots, live widget output — must stay in a notebook page regardless of where it conceptually belongs. That is a rendering constraint, not an ownership claim.

### Standalone readability, and declared prerequisites

Readers arrive from search, from a deep link, from the API reference — not by walking the TOC in order. **Every article must be readable start to finish on its own**, referencing others where useful rather than depending on them silently.

That is a real constraint on §2's knife. When you move a section into a new page, leave behind whatever the origin page needs to still make sense — a sentence of recap plus the cross-link, not a hole where the explanation was. Redundancy of that kind is wanted; it is what lets both pages stand.

When an article genuinely **builds on** another — it cannot be made self-contained without restating the whole thing — say so at the top, before the first section, using the house `seealso`:

```markdown
::: {seealso}
Builds on [The Two Domains](domains.md) — the domain contracts and the
`@ensures_domain` / `@computes_in` split are assumed here.
:::
```

Placement follows existing usage: **top** for a hard prerequisite the whole page assumes; **inline, where the dependency bites**, for a lateral reference (`bruker_filter_removal.md` points at `phase.md` exactly where the phase twist starts mattering); **end** for onward reading (`domain_contracts.md`). Use the top note sparingly — needing one on most pages means concepts are split too thin, and the fix is consolidation, not more notes.

## 3. Pass 1 — draft from the plan (first commit on the branch)

Write it before the implementation, straight off the approved plan, as the branch's first commit:

```
docs: design note for <topic>
```

Committing the draft is the point: the reviewer sees design → implementation → reconcile as three phases of one PR, and pass 2's diff is legible. Do not merge the branch with pass 2 outstanding.

### Shape: one driving question, never a FAQ of "Why X?" sections

This is the part that decides whether the page works. **Find the single question a reader naturally arrives with, and let it drive the whole article.** Every decision then arrives as the answer to a tension the reader just felt — no decision is *announced*, each is *arrived at*.

That question comes from the page's declared prerequisite where it has one, and otherwise from the concept itself — not from TOC adjacency, since readers land here from search and deep links as often as from the previous page.

- `vocabulary.md` is driven by *"my data doesn't use xmris's names — now what?"* Why not aliases, conform-your-data, why terms are frozen — each falls out of a tension that one question raises.
- `domains.md` is driven by *"what happens when I call a spectral operation on a FID?"*

The failure mode is a correct decision presented cold:

❌ A `## Why no aliases` heading appearing out of nowhere. The reader disengages, and a sound decision reads as an assertion to accept.
✅ The reader runs into the aliasing problem while following the driving question, and the decision reads as the conclusion they were already reaching.

Write the driving question down before drafting. If you cannot name it, you have a topic, not an article — and you will produce section-per-topic prose by default.

**Beats, not a template.** `domains.md` passes through: the problem (concrete failure, not abstraction) → the goal, compressed to one sentence in an `{important}` admonition → the design (taxonomy or rule; mermaid if the structure branches) → the contract surface at a glance (a table; the part users return to) → guardrails (one-way doors, exact error text, the escape hatch) → how contributors declare or extend it. Those are waypoints the narrative passes through, not headings to fill. The driving question determines the actual headings, and their order.

**Concise and conversational.** Guide the reader through the thought process; do not write a long technical brief. Code examples are welcome. Deep or tangential rationale — a representation comparison like StrEnum vs dataclass vs plain constants — goes in a `:::{dropdown}`, off the main line of reasoning.

### Use the MyST palette — the richness *is* part of the argument

A design doc is a page in a docs pipeline that renders and executes, not a README. Plain prose under plain headings wastes the medium, and — more to the point — makes the reasoning harder to review: a decision tree drawn as a flowchart is checkable at a glance, the same tree in paragraphs is not. Reviewability and documentation quality come from the same richness.

Reach for the feature that carries the argumentative load:

| Job in the argument | Feature | In the wild |
|---|---|---|
| The choice a contributor has to make | `{mermaid}` flowchart | `domains.md`'s `@ensures_domain` / `@computes_in` decision tree |
| Structural relation between states | `{mermaid}` graph | the time ↔ spectral converter diagram |
| The contract surface | markdown table | `domains.md`, "What you get, at a glance" |
| The goal, in one sentence | `{important}` | `domains.md` |
| Guardrail, one-way door, footgun | `{warning}` / `{note}` | `bruker_filter_removal.md` |
| Deep or tangential rationale | `:::{dropdown}` | `architecture.md`'s "Under the Hood" |
| Rejected approach vs house way | paired ❌ / ✅ code blocks | `architecture.md`'s Parameter Soup |
| The math being demonstrated | `$…$` / `$$…$$` | `hz_and_ppm.md` |
| Prerequisite, lateral, onward links | `{seealso}` | §2 above |
| A **load-bearing** claim (pass 2 only, where it earns it) | `{code-cell}` + hidden assert | `domain_contracts.md` |

Three constraints on reaching:

- **Stay inside the palette the docs already use** — the table above, plus `{tip}` and `{admonition}` with a `:class:`. Nothing in `docs/` uses grids, cards, or tab-sets; a design doc is the wrong place to introduce a directive the rest of the site doesn't share.
- **Mermaid escaping rules live in the `new-doc-notebook` skill** — quote every label, `<br>` not `\n`, monospace `<span>` for code inside labels. Those diagrams were expensive to get right; copy an existing one rather than hand-rolling syntax.
- **Every element earns its place.** A diagram that restates the sentence above it is noise. If your flowchart has two nodes, it was a sentence.

```{note}
**Pass-1 code is illustrative by definition — do not chase executability.** The API does not exist yet. You are sketching the call site you *wish* existed, which is design work in its own right and often the fastest way to feel whether an API is ergonomic. Static ` ```python ` blocks are correct there, and plenty of snippets stay static forever: ❌ anti-pattern blocks (which must *never* run), pseudo-code, quoted error text, illustrative fragments.

What pass 2 owes you is **accuracy, not executability** — the two are easy to conflate. Re-read every snippet against the built API and fix what drifted; that is the #90 lesson, a quoted `ValueError` the implementation had moved on from. Promote a snippet to a `{code-cell}` only where the claim is load-bearing and a reader would be genuinely misled if it were wrong. That is a handful of snippets, in the companion notebook — not all of them, and not the default.
```

### Rejected alternatives are pedagogy, not an appendix

Do not append a dry "Alternatives Considered" list. House style walks the reader *through* the rejected option and lets them feel why it fails — `architecture.md`'s "❌ The Anti-Pattern: Parameter Soup" → "✅ The xmris Way", `domains.md`'s "either you pepper pipelines with manual conversions … or the library converts silently". Same information, and it teaches instead of filing. The decision record falls out of the explanation.

### Mark unproven claims

Anything the draft asserts that code has not yet demonstrated is an assumption, and assumptions are what pass 2 exists to catch. Tag each one inline:

```markdown
<!-- ASSUMPTION: length-preserving is the criterion separating computes_in from ensures_domain -->
```

HTML comments do not render. Pass 2 must resolve every one and delete the marker; `grep -rn "ASSUMPTION:" docs/` returning nothing is a merge gate. Be honest here — a draft with no assumptions marked usually means they were not looked for.

## 4. Pass 2 — reconcile against the implementation (last commit on the branch)

Re-read the draft **against the merged code**, not from memory. Commit as:

```
docs: reconcile design note for <topic>
```

Pass 2 is **not** re-asked — accepting pass 1 commits to it, and a reconciled doc is the deliverable. The only ask at this stage is when the skill is invoked at the end of work that never had a pass 1: put the same question from §0, noting the doc will be written retroactively and so records the outcome rather than the deliberation.

This pass is not optional polish. In this repo it was skipped once and cost PR #90 — the `domains.md` draft (#76) landed before the rollout (#78/#79) and sat wrong on `main` for four PRs until two independent code reviews caught it. Every category below is a real defect from that incident:

- **Quoted error messages.** The doc quoted a `ValueError` whose wording had since changed. Grep the actual string in `src/` and paste it verbatim.
- **Decision criteria in diagrams and tables.** The draft's flowchart gated on "same physics, length-preserving?" — plausible at design time, and false: `zero_fill` changes length and is still `@computes_in`. Walk every branch of every diagram against the real code and confirm a function takes that path.
- **Over-general guardrails.** "Explicit dims pass through" was stated unconditionally; it holds only for `@computes_in`, not `@ensures_domain`. For each rule, ask which code paths it actually covers, and scope the sentence to those.
- **API surface, and every code snippet.** Function names, signatures, defaults, attr keys, accessor spellings all drift during implementation, and pass-1 snippets were written against an API that did not exist yet. Read each one against the built code and correct it. Correctness is the bar — a snippet does not have to become executable to be right.
- **Adopted rejections.** If something the draft argued against got built anyway, the rationale now reads backwards. Rewrite it, or delete the passage.

Then verify the claims execute:

```bash
grep -rn "ASSUMPTION:" docs/                 # must be empty
uv run test-gen && uv run pytest "tests/autogen_notebooks/<category>/<topic>.ipynb" -n0 --no-cov
```

Every user-visible behavior the doc promises should be asserted somewhere — in the companion notebook's hidden test cells for demonstrable claims, in `tests/test_core.py` for architectural invariants. A promise in prose with no assert behind it is the next `#90`.

## 5. Multi-PR chains

For work spanning several PRs, the doc lives in the **first** PR of the chain and is reconciled in the **last**. Each intermediate PR that changes behavior the doc describes carries its own reconcile hunk — do not batch them to the end, that is exactly the #76→#90 failure mode. If the chain's design shifts mid-flight, amend the doc in the PR that shifted it, and say so in that PR body.

## 6. Register and link

- **TOC entry in `docs/myst.yml`** with a `title:`. Explanation articles currently sit in the group whose concepts they explain (`domains.md` → Basics). A page missing from the TOC never renders.
- **Cross-link the pair**: explainer → notebook via `[text](#explicit-target)`, notebook → explainer via a relative `.md` path. Never link `.ipynb`.
- Link the doc from the PR body. It is the summary — the PR description should not restate it.

## Final checklist

- [ ] Trigger assessed and the doc/artifact choice **put to the user**, naming the concrete trigger
- [ ] Pass 1 committed as the branch's first commit, before implementation
- [ ] **Driving question named** — and no decision announced as a cold "Why X?" heading
- [ ] Main line of reasoning concise and conversational; tangents in `:::{dropdown}`
- [ ] **MyST palette used where it carries the argument** — mermaid for decisions and structure, tables for the contract surface, admonitions for goal and guardrails; nothing decorative
- [ ] Pass 2: every snippet re-read against the built API and corrected — executable only where a claim is load-bearing
- [ ] Goal compressed to one sentence in an `{important}` admonition
- [ ] Contract surface stated as a table
- [ ] Rejected alternatives walked through as pedagogy, not appended as a list
- [ ] **Single home per concept** — overlapping sections in existing pages thinned and cross-linked, not duplicated; moves called out in the PR body
- [ ] **Reads start to finish on its own**, including any page you thinned; hard prerequisites declared in a top `seealso`
- [ ] Pass 2 committed as the branch's last commit, read against the code
- [ ] Error messages, diagram branches, and guardrail scopes verified against `src/`
- [ ] `grep -rn "ASSUMPTION:" docs/` is empty
- [ ] Every promised behavior has an assert behind it
- [ ] TOC entry present; explainer and notebook cross-linked
