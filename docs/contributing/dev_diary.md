(contribute-dev-diary)=
# Write a dev-diary entry

A [dev-diary entry](#diary-about) records *why* a significant change was made — the kind of decision
someone will want explained when they ask "why is it like this?" a year from now. You write it
twice: a short draft distilled from the approved plan — the thing actually reviewed *before* the
work starts, and the work waits until it has been — and a final version reconciled once the change
has landed, rewritten into the story of how it is now.

Not every change earns one. Choosing between two viable approaches, adding conceptual surface (a
rule, a decorator, a namespace), or a refactor that spans several PRs does; a bug fix, a routine
dependency bump, or a vocabulary term that follows an existing pattern does not. The skill always
puts that call to you before writing anything — and when an existing entry already tells the
decision's story, it proposes updating that entry instead of adding a sibling.

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
