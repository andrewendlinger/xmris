# Changelog section skeleton

Copy the structure below into `docs/changelog.md`, **above** the existing version sections and
below the page intro. Prose only — no jupytext frontmatter, no kernel (the page is genre `other`,
like `roadmap.md`, so nothing executes it).

One H2 and one `(target)=` per version. No H3s: the group labels are bold runs, so a release owns
exactly one anchor and inserting a version churns no other.

---

````markdown
(changelog-v<X>-<Y>-<Z>)=
## v<X>.<Y>.<Z> — unreleased

<Optional: one line naming what shipped, if the release has a shape worth stating.
 What shipped — never what the changelog is, how it is formatted, or who it is for.>

**Breaking**
- <What no longer works, and the one line that fixes it. This group leads whenever it exists —
  it is the only one a reader must not scroll past.> — [#<issue>](…) · [#<pr>](…)

**Added**
- `<public.symbol>` <does what, for whom>. — [#<issue>](https://github.com/andrewendlinger/xmris/issues/<issue>) · [#<pr>](https://github.com/andrewendlinger/xmris/pull/<pr>) · [<Page title>](#<page-target>) · [diary](#diary-<slug>)

**Changed**
- <A default, a domain contract, an error message — anything a working pipeline would notice.> — [#<pr>](…)

**Fixed**
- <The wrong behaviour, stated as what the user saw, not as the root cause.> — [#<issue>](…) · [#<pr>](…)

**Documentation**
- <The new or rewritten page, linked by its title.> — [#<pr>](…) · [<Page title>](#<page-target>)

**Maintenance**
- <N> dependency updates, and <the packaging or CI change a user would actually notice>. — [#<pr>](…)
````

---

## The rules the skeleton encodes

| | |
|---|---|
| **Group order** | Breaking · Added · Changed · Fixed · Documentation · Maintenance. Drop any that is empty. |
| **Trail order** | issues → PRs → docs page → diary. Fixed, so a reader learns to skim it. |
| **Issues and PRs** | bare `#N` with the full `https://github.com/andrewendlinger/xmris/...` URL — readers arrive from PyPI, where a relative link is dead. |
| **Docs and diary** | MyST targets (`[The Two Domains](#domains)`), never URLs or file paths — `myst build --strict` then verifies them for you. |
| **Sentence** | names the public symbol, describes the consequence, needs no diff to understand. |
| **`unreleased`** | stays in the heading until the release; the `release` skill refuses to tag while it is there. Replace with `— YYYY-MM-DD`. |

## The bottom of the page

Pinned last, written once, never regenerated:

````markdown
(changelog-earlier)=
## Earlier releases

v0.1.0 – v0.6.1 predate this changelog. Their contents are the
[tag list](https://github.com/andrewendlinger/xmris/tags) and the commits between them.
````
