(contribute-dev-diary)=
# Write a dev-diary entry

A [dev-diary entry](#diary-about) records *why* a significant change was made — the kind of decision
someone will want explained when they ask "why is it like this?" a year from now. You write it
twice: a short draft distilled from the approved plan (the thing actually reviewed *before* the work
starts), and a reconciled final version once the change has landed and reality has argued back.

Not every change earns one. Adding vocabulary or a contract, choosing between two viable approaches,
or a refactor that spans several PRs does; a bug fix or a routine dependency bump does not. The skill
always puts that call to you before writing anything.

Entries live under `docs/diary/`, below the pinned [intro](#diary-about), with the newest at the
bottom. The mechanics — the two passes, the one-screen budget, and how open assumptions are marked
so they *render* on the page you review — are exactly what the skill enforces:

(contribute-dev-diary-skill)=
## Working with Claude Code

The **`dev-diary`** skill's checklist:

```{literalinclude} ../../.claude/skills/dev-diary/SKILL.md
:language: markdown
:start-after: <!-- excerpt:start -->
:end-before: <!-- excerpt:end -->
:caption: Quote from the [dev-diary/SKILL.md](https://github.com/andrewendlinger/xmris/blob/main/.claude/skills/dev-diary/SKILL.md)
:class: skill-quote
```
