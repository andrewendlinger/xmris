(contribute-changelog)=
# Write a changelog entry

The [changelog](#changelog) answers one question, for one reader: *I just ran
`pip install -U xmris` — what is different?* Nothing else on the site answers it. The
[dev diary](#diary-about) records **why** a decision was made and the [roadmap](#roadmap) records
**what is next**; both are about reasoning, and this page is about consequences.

It is written **at release time, not per pull request** — during the fifteen minutes the full
matrix runs, so the entry rides the version bump into `main` in a single pull request
([Publishing, step ②b](#release-changelog)).

There is no generator and there are no fragment files to collect. `git log` since the last tag is
raw material, not a draft: a squash-merge subject says what the *diff* did, and a bullet has to say
what the *user* got. Turning one into the other is the whole job. What replaces a generator's
guarantee that nothing was silently missed is an accounting rule — every commit in the range ends
up either a bullet or a deliberate, stated drop — and what earns a bullet is **user-visible
consequence, not commit type**. A `chore:` that changes what `pip install xmris` pulls in earns
one; a `feat:` that only adds an internal helper does not.

Every bullet then carries its trail, in a fixed order: issues → pull requests → the docs page it
produced → the diary entry that argued it, so a reader can always get from a one-line consequence
back to the reasoning. The last two are MyST targets rather than URLs, which makes that half of the
trail machine-checkable — with one trap: an unresolved target is a *warning*, and `--strict` only
promotes errors, so the build log has to be grepped rather than trusted to its exit code.

:::{note}
**The genre carve-out.** The changelog is the one page on this site that is a **reference genre**:
no driving question, no felt tension, no admonitions, no dropdowns. A reader scans it; they do not
read it. The [four house-style rules](#contribute-docs) bind everywhere else — so do not carry this
shape out of `docs/changelog.md`, and do not carry theirs in.
:::

(contribute-changelog-skill)=
## Working with Claude Code

The **`changelog`** skill's checklist:

```{literalinclude} ../../.claude/skills/changelog/SKILL.md
:language: markdown
:start-after: <!-- excerpt:start -->
:end-before: <!-- excerpt:end -->
:caption: Quote from the [changelog/SKILL.md](https://github.com/andrewendlinger/xmris/blob/main/.claude/skills/changelog/SKILL.md)
:class: skill-quote
```
