(contribute-dev-diary)=
# Write a dev-diary entry

A [dev-diary entry](#diary-about) records *why* a significant change was made — the kind of decision
someone will want explained when they ask "why is it like this?" a year from now. It is written
once, when the change has landed — checked against the real code rather than drafted from a plan,
so it tells the story of how it is now and why.

Not every change earns one. Choosing between two viable approaches, adding conceptual surface (a
rule, a decorator, a namespace), or a refactor that spans several PRs does; a bug fix, a routine
dependency bump, or a vocabulary term that follows an existing pattern does not. The skill always
puts that call to you before writing anything — and when an existing entry already tells the
decision's story, it proposes updating that entry instead of adding a sibling.

Entries live under `docs/diary/`, below the pinned [intro](#diary-about), with the newest at the
bottom. The mechanics — the one-screen budget, the `Last edited` line, and where an entry
registers — are exactly what the skill enforces:

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
