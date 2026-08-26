(diary-docs-previews)=
# Every pull request publishes the page it changes

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-26 · #112, #114, #142, #143, #144, #171</span>

A preview link was the one thing a pull request never handed you, and getting one turned out to be
mostly a matter of not deleting a build CI had already paid for. Two questions had to be answered
before that was safe, and each was answered wrong the first time: **what the published site is made
of**, and **who is allowed to write it**.

:::{important}
Built output is derived, never stored: every deploy recomputes the whole site from `main`'s build
plus one artifact per *currently open* pull request. Who may publish it is decided in
`publish.yml`, which GitHub runs from `main` whatever branch triggered it — never in the file the
branch being tested can edit.
:::

(diary-docs-previews-derived)=
## Nothing to clean up, because nothing accumulates

The first answer arrived as a bill: three weeks in, the branch carrying the site had reached 103 MB,
its build had crept from 28 s to 494 s, the live site was serving a layout two merges old, and a
preview whose pull request merged four days earlier was still published. Four bugs, one cause — the
published site was **versioned state**. Write it instead as a function of what is open *right now*
and that entire class of failure stops being reachable:

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    subgraph B["The branch · read-only token"]
        E["Build: execute + strict"] --> P["Upload preview-pr-N"]
    end
    subgraph M["main's workflow · write token"]
        Q["Ask GitHub what is open"] --> A["Assemble: main's build<br>+ each open preview"]
        A --> D["Deploy it whole"]
    end
    P -.->|"one artifact"| Q
```

A closed pull request's preview is never removed; it simply stops being an input. That one sentence
covers the stranded preview, the `.nojekyll` files orphaned in closed pull requests' directories,
and the branch that only ever grew — all three were incremental-cleanup bugs, and there is no
increment left to clean. The store is GitHub's artifact retention rather than git.

(diary-docs-previews-provenance)=
## The name on an artifact is not a claim about who wrote it

The second answer arrived as a contributor, and this page asserted the wrong one for a month. The
first fork pull request (#164) got no preview, and the guard that arranged that was doing nothing of
the sort. For a `pull_request` event GitHub runs the workflow file **from the merge commit** — for a
fork, a commit the contributor controls. The guard lived in that file. So did the line naming the
artifact. And assembly chose the site root by asking for the newest artifact called `site-main`.
Each link is defensible alone; chained, they meant a fork could have published the **root** of the
site, with nothing in the way but a run-approval prompt that stops appearing after a contributor's
first merged pull request.

So the guard was deleted rather than fixed, and forks now publish like anyone else. What makes that
safe is a rule that reads nothing the branch wrote — **provenance, not name**:

| Admitted | Only when |
|---|---|
| `site-main` | its run's `head_repository_id` equals `repository_id`, and its branch is `main` |
| `preview-pr-N` | its run's head SHA **is** pull request N's current head |
| the trigger itself | the completed run was `.github/workflows/deploy.yml` |

Every field there comes from the API, never from the artifact. The middle row is what lets a fork
publish at all: it cannot claim another pull request's directory without its branch tip *being* that
commit, at which point it has claimed nothing.

:::{warning}
This buys containment, not innocence. A fork's preview is still contributor-written HTML served from
`andrewendlinger.github.io`, an origin shared with every Pages site on the account — the trade every
hosted preview service makes. It is bounded, not removed: a fork writes only under its own
`pr-preview/pr-N/`, closing the pull request triggers the deploy that drops it, and the payload is
capped per preview, by file count, and against Pages' 1 GB ceiling. GitHub's approval prompt is a
**run** gate, not a review gate — it is granted before the content exists.
:::

(diary-docs-previews-changed)=
## What changed from the plan

- **This page called the fork guard a security boundary.** It was a token workaround wearing a
  boundary's clothes, and it sat here unchallenged until the first fork pull request arrived to test
  it. A guard that lives in a file the adversary edits is documentation, not enforcement.
- **The publishing job has no checkout, and that broke `gh`.** Moving artifacts needs no source, so
  omitting `actions/checkout` looked like free economy — but `gh run download`, `gh pr list` and
  `gh pr comment` all resolve the repository from a git remote, and the job died half a second in
  with `failed to run git: fatal: not a git repository`. Only `gh api` survived, because it carries
  an explicit `repos/OWNER/REPO/...` path. `GH_REPO` in the step environment fixes it.

What the plan got right is the part that mattered: the fail-loud path fired for real on the first
pull request, before any `site-main` artifact existed, printing `No usable 'site-main' artifact. The
live site is unchanged; recover with: gh workflow run deploy.yml` — and the previous site kept
serving throughout, exactly as an atomic deployment should.

:::{dropdown} Why not `pull_request_target` for the build?
It is the obvious way to hand a fork's build a write token, and the wrong one here: the build runs
`myst build --execute`, which executes every notebook in the branch. `pull_request_target` would
pair arbitrary contributor code with a token that can write to the repository. `workflow_run`
inverts it — the untrusted half keeps a read-only token and passes an artifact across, and the
privileged half never sees the branch at all.
:::

:::{dropdown} Why not Cloudflare Pages or Netlify?
Both hand out preview URLs with no workflow YAML at all. The price is a vendor account and an API
token in the repository's secrets — for a package heading into JOSS review, one more dependency a
reviewer cannot see the far side of. GitHub Pages already hosts the site; a preview should not add a
second host. The same stance is why assembly uses nothing but `gh`.
:::

:::{dropdown} Why not just download the build artifact?
That was the smallest possible change back when previews were added — delete one `if:` and the
artifact survives. Artifacts are now the transport, but the conclusion is unchanged: a 52 MB zip you
download, unpack and serve locally is not a link, and a review gate only works when reading the page
is the path of least resistance.
:::

:::{dropdown} How `--strict` broke the build on a valid DOI
Docs rules have three enforcers, each because the one above it is blind to something:
`check_docs.py` catches what the build never mentions (a missing target, a dead `.ipynb` link, a
drifted kernel name); `myst build --strict` catches what the build reports and then exits 0 on
anyway; the preview catches whether the page actually reads right. Turning the middle one on went
red on a *valid* DOI — `Could not find DOI "https://doi.org/10.1002/mrm.25568" from doi.org` —
because mystmd resolves every `doi.org` link over the network at build time. `myst build --doi-bib`
freezes that metadata into `docs/myst.doi.bib`, which only takes effect once *listed* in
`myst.yml`'s `bibliography`, since declaring that key makes mystmd load only the files named there.
The lesson generalises past DOIs: **a strict gate converts every latent network dependency into a
flaky failure**, on whichever pull request happens to be open at the time. Note `--strict` exits
non-zero on **errors** only, never warnings — so the quoted-source pins in
[the Architecture Contract](#contract) stay load-bearing.
:::

:::{dropdown} The `.nojekyll` trap, and the ref pattern, now historical
A branch-served Pages site runs Jekyll, which silently drops every `_`-prefixed path — all of the
site's CSS and JS. Neither deploy action wrote a `.nojekyll`, so the build did, for the root only.
Writing one into each preview directory looked harmless and was not: the removal deploys an *empty*
folder, and the action adds `--exclude .nojekyll` to its delete-rsync whenever the source lacks one,
so exactly that file outlived every closed pull request. A site served from a workflow artifact is
published as-is, so both the file and its footgun are gone.

Separately, a pull request was once refused with `Branch "refs/pull/142/merge" is not allowed to
deploy to github-pages`. The rule admitting it must be spelled `refs/pull/*/merge`, because policies
are matched against `GITHUB_REF` with `fnmatch` semantics, where a wildcard never crosses a `/`.
That entry is now removable: with publishing moved to `main`'s workflow, no deployment ever comes
from a merge ref again.
:::
