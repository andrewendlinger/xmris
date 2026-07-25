(diary-docs-previews)=
# Every pull request publishes the page it changes

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-07-25</span>

The [`dev-diary` workflow](#diary-about-how) stops the work until a draft entry has been read *on
the rendered site* — and the only way to read it there is to start a blocking local server and wait
for it to execute every notebook. Meanwhile CI already builds the whole site on every pull request,
executes every cell, and throws it away: the reviewer's copy lives two minutes on a runner. For a
project whose docs are its tests and its review surface, the one thing a pull request should hand
you is the one thing it never had — a link.

:::{important}
Every pull request publishes its own executed build to the `gh-pages` branch under
`pr-preview/pr-N/`, and comments the link on the pull request.
:::

(diary-docs-previews-shape)=
## Two publishers, one branch

The expensive half was already paid for, so this is mostly a matter of not deleting it. One branch
carries both the live site and every preview: `main` publishes to the root, a pull request to its
own subdirectory, and closing it takes that subdirectory away.

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
hangs or dies there would have done the same on `main`.

(diary-docs-previews-layers)=
## The third enforcement layer

Docs rules now have three enforcers, and each exists because the one above it is blind to something:

| Layer | Catches | Blind to |
|---|---|---|
| `check_docs.py` — the `Docs style` job | what the build never mentions: a missing target, a dead `.ipynb` link, a drifted kernel name | anything it has no rule for |
| `myst build --strict` | what the build *reports* and then exits 0 on anyway | its own warnings — a missed `literalinclude` anchor still only warns |
| the preview | whether the page actually reads right | nothing a human declines to look at |

External link rot stays off the list on purpose — checked weekly on a schedule, since a dead
third-party URL is no reason to redden a pull request that never touched it.

:::{dropdown} Why not Cloudflare Pages or Netlify?
Both hand out preview URLs with no workflow YAML at all. The price is a vendor account and an API
token in the repository's secrets — for a package heading into JOSS review, one more dependency a
reviewer cannot see the far side of. GitHub Pages already hosts the site; a preview should not add a
second host.
:::

:::{dropdown} Why not just download the build artifact?
It is the smallest possible change — delete one `if:` and the artifact survives. But a 52 MB zip you
download, unpack and serve locally is not a link, and a review gate only works when reading the page
is the path of least resistance.
:::

:::{attention} Assumptions to verify
- A branch-served Pages site runs Jekyll, which drops every `_`-prefixed path — and the build
  contains `build/_assets`, `build/_shared` and `build/routes/_index-*`. Neither deploy action
  writes a `.nojekyll`, so the workflow must; unproven until the first real deploy.
- `BASE_URL` is baked in at build time, so a preview built for `/xmris` would 404 every asset. The
  expression that swaps it for `/xmris/pr-preview/pr-N` is unproven in a run.
- Read from mystmd's exit logic, `--strict` fails on *errors* only — if so, the quoted-source pins in
  [the Architecture Contract](../contributing/contract.md) stay load-bearing.
- Fork pull requests get a read-only token: they must skip publishing and stay green as a build-only
  smoke test, not fail on someone's first contribution.
- Merging fires the preview cleanup and the `main` deploy at once, on the same branch; rebasing
  rather than force-pushing is assumed enough to keep them from clobbering each other.
:::
