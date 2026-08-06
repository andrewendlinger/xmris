(diary-docs-previews)=
# Every pull request publishes the page it changes

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-07-25 · #112, #114</span>

The [`dev-diary` workflow](#diary-about-how) stops the work until a draft entry has been read *on
the rendered site* — and the only way to read it there was to start a blocking local server and wait
for it to execute every notebook. Meanwhile CI already built the whole site on every pull request,
executed every cell, and threw it away: the reviewer's copy lived two minutes on a runner. For a
project whose docs are its tests and its review surface, the one thing a pull request should hand
you was the one thing it never had — a link.

:::{important}
Every pull request publishes its own executed build to the `gh-pages` branch under
`pr-preview/pr-N/`, and comments the link on the pull request.
:::

(diary-docs-previews-shape)=
## Two publishers, one branch

The expensive half was already paid for, so this was mostly a matter of not deleting it. One branch
carries both the live site and every preview: `main` publishes to the root
(`JamesIves/github-pages-deploy-action` with `clean-exclude: pr-preview/`), a pull request to its own
subdirectory (`rossjrw/pr-preview-action`), and closing it takes that subdirectory away.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    B["Executed build"] --> P{"Which event?"}
    P -->|"push to main"| R["gh-pages root"]
    P -->|"PR opened or updated"| V["gh-pages: pr-preview/pr-N"]
    P -->|"PR closed"| X["preview removed"]
    R --> S["the live site"]
    V --> C["link commented on the PR"]
```

A preview is not a lighter build: it is the same executed run the live site gets, so a notebook that
hangs or dies there would have done the same on `main`. What makes that affordable is that the
`.github/workflows/deploy.yml` build step is unchanged — only its destination is new. Two details in
that step are load-bearing, and both were guesses in the draft that turned out to hold:

`BASE_URL` is baked into every asset path at build time, so a preview must be built for the
subdirectory it will be served from; building for `/xmris` and serving from
`/xmris/pr-preview/pr-N/` 404s every stylesheet. And a branch-served Pages site runs Jekyll, which
silently drops every `_`-prefixed path — here `build/_assets`, `build/_shared` and
`build/routes/_index-*`, which is all of the site's CSS and JS. Neither deploy action writes a
`.nojekyll`, so the build does.

:::{warning}
`.nojekyll` is honoured at the **published root only**. Writing one into each preview directory
looked harmless and was not: the removal deploys an *empty* folder, and the action adds
`--exclude .nojekyll` to its delete-rsync whenever the source folder lacks one — so exactly that file
outlived every closed pull request. It is now written for root deploys only.
:::

(diary-docs-previews-layers)=
## Three enforcement layers, and how the third one bit

Docs rules now have three enforcers, each because the one above it is blind to something:

| Layer | Catches | Blind to |
|---|---|---|
| `check_docs.py` — the `Docs style` job | what the build never mentions: a missing target, a dead `.ipynb` link, a drifted kernel name | anything it has no rule for |
| `myst build --strict` | what the build *reports* and then exits 0 on anyway | its own warnings — a missed `literalinclude` anchor still only warns |
| the preview | whether the page actually reads right | nothing a human declines to look at |

That middle row is narrower than it sounds, and it matters: mystmd's `--strict` exits non-zero on
**errors** only, never on warnings. So the quoted-source pins in
[the Architecture Contract](#contract) stay load-bearing — a renamed decorator
would still truncate a `literalinclude` with nothing but a warning to show for it.

Turning that layer on also had a consequence nobody predicted. The first strict build on CI went red
on a *valid* DOI:

```text
⛔️ 03_plotting_1dfid.md:198 Could not find DOI "https://doi.org/10.1002/mrm.25568" from doi.org
```

mystmd resolves every `doi.org` link against the network at build time. `--strict` had therefore
quietly made the documentation build — and the deploy behind it — depend on doi.org being reachable
from a GitHub runner. The fix is to stop asking: `myst build --doi-bib` freezes that metadata into
`docs/myst.doi.bib`, which only takes effect once *listed* in `myst.yml`'s `bibliography`, because
declaring that key at all makes mystmd load only the files named there. A `check_docs.py` rule now
errors on any `doi.org` link missing from the cache, so the next contributor to add one is told
before CI is.

The lesson generalises past DOIs: **a strict gate converts every latent network dependency in a build
into a flaky failure**, and it fails on whichever pull request happens to be open at the time.

(diary-docs-previews-gate)=
## What the gate cost, and what it was worth

A preview only helps if a red check can actually stop a merge, and none could: `main`'s ruleset
contained a single `deletion` rule, so nothing — not the tests, not `Docs style` — was required.
It now requires all four checks and a pull request, with zero approving reviews so a sole maintainer
can still merge their own work.

Requiring a pull request broke exactly one thing, and instructively: the release flow ended with
`git merge release/vX.Y.Z` and `git push origin main`. Routing that through a pull request instead
exposes a second problem, because this repository squash-merges — a tag cut on the release branch
would sit on a commit that never enters `main`'s history. So the order flipped: bump, merge, *then*
tag the merged commit ([Publishing](#release-tag-publish)). Publishing itself never moved; the `v*`
tag still triggers the matrix and the PyPI job.

:::{dropdown} Why not Cloudflare Pages or Netlify?
Both hand out preview URLs with no workflow YAML at all. The price is a vendor account and an API
token in the repository's secrets — for a package heading into JOSS review, one more dependency a
reviewer cannot see the far side of. GitHub Pages already hosts the site; a preview should not add a
second host.
:::

:::{dropdown} Why not just download the build artifact?
It was the smallest possible change — delete one `if:` and the artifact survives. But a 52 MB zip you
download, unpack and serve locally is not a link, and a review gate only works when reading the page
is the path of least resistance.
:::

(diary-docs-previews-changed)=
## What changed from the plan

Two of the draft's assumptions were wrong in ways worth keeping, because both would cost an hour to
rediscover:

- **The draft assumed the fork guard and the merge-time write race were the risks worth naming.**
  Neither has bitten: the `closed` cleanup and the `main` deploy fired simultaneously on #112 and
  `force: false` rebased cleanly, and no fork pull request has arrived yet to exercise the read-only
  token path — that guard is still design, not evidence.
- **Flipping the Pages source was assumed to be the last step; it is not self-starting.** A
  branch-served site builds on the next *push* to that branch, and the last `gh-pages` push predated
  the flip — so the old artifact kept serving, with `pages/builds/latest` returning 404 and no
  workflow run to look at. `gh api -X POST repos/OWNER/REPO/pages/builds` bootstraps it. Anyone
  repeating this migration will otherwise conclude the deploy silently failed.
