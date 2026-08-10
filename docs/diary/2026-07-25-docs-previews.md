(diary-docs-previews)=
# Every pull request publishes the page it changes

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-10 · #112, #114</span>

A preview link was the one thing a pull request never handed you, and getting one turned out to be
mostly a matter of not deleting a build CI already paid for: the reviewer's copy used to live two
minutes on a runner and then vanish. Three weeks later the bill arrived. The branch carrying the
site is 103 MB across 3,079 files, its history 674 MB of unique blobs that every clone pays;
GitHub's own build of that branch has crept from 28 s to 494 s and errored on three of its last ten
runs; the live site is serving a layout two merges old; and the preview for a pull request merged
four days ago is still published. Four bugs, one cause — **the published site is versioned state**.

:::{important}
Built output is derived, never stored. Every deploy recomputes the whole site from `main`'s build
plus one artifact per *currently open* pull request, and `gh-pages` goes away.
:::

(diary-docs-previews-derived)=
## Nothing to clean up, because nothing accumulates

Write the published site as a function of what is open *right now* and the entire class of failures
above stops being reachable:

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    E["Any deploy event"] --> Q["Ask GitHub what is open"]
    Q --> M["main's build artifact"]
    Q --> P["one artifact per open PR"]
    M --> A["Assemble the site"]
    P --> A
    A --> D["Deploy it whole"]
```

A closed pull request's preview is not removed; on the next deploy it is simply no longer an input.
That one sentence covers the preview still live four days after its merge, the `.nojekyll` files
stranded in closed pull requests' directories, and the branch that only ever grew — all three were
incremental-cleanup bugs, and there is no increment left to clean.

The store becomes GitHub's artifact retention rather than git: 90 days for `main`'s build, 14 for a
preview, with the weekly link-check run extended to rebuild and republish. That keeps the live
artifact under a week old and doubles as the sweep, bounding how long a closed pull request's
preview can linger when nothing else happens to deploy.

:::{warning}
Publishing now runs on pull-request events, so a branch's workflow writes to the live deployment.
`main`'s content always comes from `main`'s own artifact, so a broken branch can corrupt only its
own subdirectory — but the fork guard stops being a token limitation and becomes a security
boundary: a fork's HTML would otherwise be served from our origin on `github.io`. It is still
design rather than evidence; no fork pull request has arrived to exercise it.
:::

(diary-docs-previews-baseurl)=
## What survives the move, and what it costs

Two details of the build step are unchanged and still load-bearing. `BASE_URL` is baked into every
asset path at build time, so a preview must be built for the subdirectory it will be served from;
building for `/xmris` and serving from `/xmris/pr-preview/pr-N/` 404s every stylesheet. That is
also why the site refuses to compress — only 183 of 1,269 files across two builds are
byte-identical, so `plotly-5BWH43UK.js` at 4.77 MB is a fresh blob per preview.

Deleting the branch stops that from *accumulating*; it does not shrink a single copy, and the
assembled site now scales with the number of open pull requests instead. Per-copy weight is
therefore the only thing left that grows a deploy — today ~12 MB of it is executed-notebook figure
output, which the move to vector figures should largely dissolve.

:::{dropdown} The `.nojekyll` trap, now historical
A branch-served Pages site runs Jekyll, which silently drops every `_`-prefixed path — here
`build/_assets`, `build/_shared` and `build/routes/_index-*`, which is all of the site's CSS and
JS. Neither deploy action wrote a `.nojekyll`, so the build did — for the root only. Writing one
into each preview directory looked harmless and was not: the removal deploys an *empty* folder, and
the action adds `--exclude .nojekyll` to its delete-rsync whenever the source lacks one, so exactly
that file outlived every closed pull request. A site served from a workflow artifact is published
as-is, so both the file and its footgun are gone.
:::

(diary-docs-previews-layers)=
## Three enforcement layers, and how the third one bit

Docs rules have three enforcers, each because the one above it is blind to something:

| Layer | Catches | Blind to |
|---|---|---|
| `check_docs.py` — the `Docs style` job | what the build never mentions: a missing target, a dead `.ipynb` link, a drifted kernel name | anything it has no rule for |
| `myst build --strict` | what the build *reports* and then exits 0 on anyway | its own warnings — a missed `literalinclude` anchor still only warns |
| the preview | whether the page actually reads right | nothing a human declines to look at |

That middle row is narrower than it sounds: mystmd's `--strict` exits non-zero on **errors** only,
never on warnings. So the quoted-source pins in [the Architecture Contract](#contract) stay
load-bearing — a renamed decorator would still truncate a `literalinclude` with nothing but a
warning to show for it.

Turning that layer on had a consequence nobody predicted. The first strict build on CI went red on
a *valid* DOI:

```text
⛔️ 03_plotting_1dfid.md:198 Could not find DOI "https://doi.org/10.1002/mrm.25568" from doi.org
```

mystmd resolves every `doi.org` link against the network at build time, so `--strict` had quietly
made the documentation build — and the deploy behind it — depend on doi.org being reachable from a
GitHub runner. The fix is to stop asking: `myst build --doi-bib` freezes that metadata into
`docs/myst.doi.bib`, which only takes effect once *listed* in `myst.yml`'s `bibliography`, because
declaring that key at all makes mystmd load only the files named there. A `check_docs.py` rule now
errors on any `doi.org` link missing from the cache, so the next contributor to add one is told
before CI is. The lesson generalises past DOIs: **a strict gate converts every latent network
dependency in a build into a flaky failure**, and it fails on whichever pull request happens to be
open at the time.

(diary-docs-previews-gate)=
## What the gate cost, and what it was worth

A preview only helps if a red check can actually stop a merge, and none could: `main`'s ruleset
contained a single `deletion` rule, so nothing — not the tests, not `Docs style` — was required. It
now requires all four checks and a pull request, with zero approving reviews so a sole maintainer
can still merge their own work.

Requiring a pull request broke exactly one thing, and instructively: the release flow ended with
`git merge release/vX.Y.Z` and `git push origin main`. Routing that through a pull request exposes a
second problem, because this repository squash-merges — a tag cut on the release branch would sit
on a commit that never enters `main`'s history. So the order flipped: bump, merge, *then* tag the
merged commit ([Publishing](#release-tag-publish)). Publishing itself never moved; the `v*` tag
still triggers the matrix and the PyPI job.

:::{warning}
Changing where Pages gets its bytes is not self-starting, and the trap has now appeared twice in
the same shape. Moving *onto* the branch: a branch-served site builds on the next push to that
branch, and the last `gh-pages` push predated the flip, so the old artifact kept serving with
`pages/builds/latest` returning 404 and no workflow run to look at —
`gh api -X POST repos/OWNER/REPO/pages/builds` bootstraps it. Moving *off* it: the `github-pages`
environment restricts which refs may deploy, so a pull request is rejected — `Branch
"refs/pull/142/merge" is not allowed to deploy to github-pages` — before a byte moves. The rule that
admits it must be spelled `refs/pull/*/merge`, and a plain `*` will not do: policies are matched
against `GITHUB_REF` with `fnmatch` semantics, where a wildcard never crosses a `/`.
:::

:::{dropdown} Why not Cloudflare Pages or Netlify?
Both hand out preview URLs with no workflow YAML at all. The price is a vendor account and an API
token in the repository's secrets — for a package heading into JOSS review, one more dependency a
reviewer cannot see the far side of. GitHub Pages already hosts the site; a preview should not add
a second host. The same stance is why the assembly step uses nothing but `gh` and the two
first-party Pages actions, rather than the third-party deploy actions it replaces.
:::

:::{dropdown} Why not just download the build artifact?
That was the smallest possible change back when previews were added — delete one `if:` and the
artifact survives. Artifacts are now the transport, but the conclusion is unchanged: a 52 MB zip
you download, unpack and serve locally is not a link, and a review gate only works when reading the
page is the path of least resistance.
:::

:::{attention} Assumptions to verify
- No rebuild fallback is needed when `main`'s artifact is missing: a Pages deployment is atomic, so
  failing loudly leaves the previous site serving rather than an empty one.
- The weekly rebuild plus 90-day retention keeps that artifact warm enough that expiry is never
  reached in practice.
- The flip moves no URLs: every path, plus the search index, `objects.inv` and `sitemap.xml`,
  answers identically before and after — and `gh` alone replaces both deploy actions, sticky
  preview comment included.
:::
