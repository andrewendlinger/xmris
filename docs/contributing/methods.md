(contribute-methods)=
# Add a processing method

A "method" is a library function wired into the `.xmr` accessor — a transform under
`processing/`, a vendor loader under `vendor/`, or a fit under `fitting/`. Adding one means
touching the free function, its accessor delegator, the package exports, and a notebook that
proves the math. The rules that hold all of that together are not restated here; they live where
they can be enforced:

- [**The Architecture Contract**](#contract) — the Commandments, `_check_dims`, `as_variable`,
  and the real exemplars quoted live from the source. All of it applies.
- [**The Two Domains**](#domains) — which decorator a function gets (funnel,
  domain-preserving, or none), with the decision tree.
- [**The Controlled Vocabulary**](#vocabulary) — why you never hardcode `"time"`,
  and how to grow the vocabulary when it is genuinely missing a term.

Document the maths itself in a notebook, not a `test_*.py` — see [Write a docs
page](#contribute-docs) for the tutorial genre and its hidden-assert convention.

(contribute-methods-skill)=
## Working with Claude Code

The **`xmr-method`** skill walks Claude through every step above, and doubles as a manual checklist
if you are not using Claude Code:

```{literalinclude} ../../.claude/skills/xmr-method/SKILL.md
:language: markdown
:start-after: <!-- excerpt:start -->
:end-before: <!-- excerpt:end -->
:caption: Quote from the [xmr-method/SKILL.md](https://github.com/andrewendlinger/xmris/blob/main/.claude/skills/xmr-method/SKILL.md)
:class: skill-quote
```
