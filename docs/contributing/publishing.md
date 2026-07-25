(contribute-release)=
# Publishing and Deployment Workflow

`xmris` strictly separates **CI (testing)** from **CD (publishing)** to avoid the "bump version → push → CI fails → bump again" cycle. Never bump the version until all tests pass.

We use `uv` for dependency management and GitHub Actions for testing across Python 3.10–3.13 on Ubuntu, Windows, and macOS.

_Note_: [Workflow diagram](#release-diagram) at the end of the page.

:::{note}
**For Claude Code users:** the user-triggered [`release` skill](https://github.com/andrewendlinger/xmris/blob/main/.claude/skills/release/SKILL.md) is the operational checklist for this workflow. Like the other skills, it carries no rules of its own — it routes back to this page for the reasoning behind each step.
:::

---

(release-daily)=
## ① Daily Development

Work on a branch and land it through a pull request — `main` takes no direct pushes, and four checks
gate every merge. That path has its own page: [Open a pull request](#contribute-pr).

- **Do not** bump the version.
- Just code, open pull requests, and iterate.

Documentation is republished on every merge to `main`, independently of releases — see
[⑤](#release-docs-deploy).

---

(release-prep)=
## ② Release Preparation

When `main` is stable, create a release branch:

```bash
git checkout -b release/v0.2.0
git push origin release/v0.2.0
```

This triggers the **Full CD Pipeline** (`ci-publish.yml`) — a 12-job matrix covering all OS and Python combinations.

(release-tests-fail)=
### If tests fail

Do not bump the version. Fix directly on the release branch and push:

```bash
git commit -am "fix: windows path issue" && git push
```

The full matrix re-runs automatically. Repeat until green.

```{note}
**macOS exception:** Due to an upstream `pyAMARES` issue, macOS jobs use `continue-on-error: true`. macOS failures won't block the pipeline.
```

---

(release-tag-publish)=
## ③ Bump, Merge, Then Tag

Once the matrix is fully green, bump on the release branch and land it through a pull request:

```bash
uv version --bump minor                  # bump version in pyproject.toml
git commit -am "chore: bump version to 0.2.0"
git push origin release/v0.2.0
gh pr create --base main --title "chore: bump version to 0.2.0"
```

Merge that pull request, then tag the merged commit on `main` — the tag is what ships:

```bash
git checkout main && git pull
git tag v0.2.0
git push origin v0.2.0                   # triggers the publish job
```

**Tag after the merge, never before.** The tag is the permanent release marker, so it has to sit on a
commit that is actually in `main`'s history.

:::{dropdown} Why the order matters
Tagging the bump commit on the release branch and merging afterwards used to work because the
merge-back was a direct `git push origin main`, which preserved that commit. `main` now takes no
direct pushes, and pull requests here are **squash-merged** — so the merge would create a *new*
commit and the tagged one would never enter `main`'s history. Tagging after the merge keeps
`git describe` on `main` meaningful, and costs nothing: the tag triggers the full matrix again before
anything reaches PyPI.
:::

```{warning}
The `v*` tag triggers the `publish` job in `ci-publish.yml`. It uses `uv build --no-sources` to strip local Git forks (e.g. the `pyamares` patch) so PyPI users get standard dependencies. Upload uses PyPI Trusted Publishing (OIDC) — no passwords required.
```

---

(release-cleanup)=
## ④ After the Release

The merge in ③ already put the bump on `main`, so cleanup is only:

- delete the release branch (GitHub offers this on the merge, or `git push origin --delete release/v0.2.0`),
- confirm the new version is live on PyPI.

The `v*` tag stays as the permanent release marker.

---

(release-docs-deploy)=
## ⑤ Documentation Deployment

Nothing to do here — and nothing release-specific. The documentation is rebuilt and republished on
**every merge to `main`**, so the site tracks `main` rather than the latest release. The mechanics,
including the per-pull-request preview sites and the manual redeploy button, belong to the
pull-request lifecycle: [How the documentation reaches the web](#contribute-pr-deployment).

---

(release-diagram)=
## Workflow Diagram

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart TD
    subgraph dev ["① Daily development"]
        A["Branch + pull request"] -->|"four checks"| B["Fast CI: Py 3.10 & 3.13"]
        B -->|"merge"| MAIN["main"]
    end

    MAIN -.->|"main is stable"| C

    subgraph release ["② Release preparation"]
        C["Push release/v* branch"] --> D["Full matrix: 3 OS, Py 3.10-3.13"]
        D --> E{"All tests pass?"}
        E -->|"No"| F["Fix on the release branch"]
        F -->|"push fix"| D
        E -->|"Yes"| G
    end

    G -.-> H

    subgraph publish ["③ Bump, merge, then tag"]
        H["Bump on the release branch"] --> I["PR into main, merge"]
        I -->|"tag v* on main"| J["Publish job: uv build --no-sources"]
        J -->|"Trusted Publishing"| K[("PyPI")]
    end

    K -.-> L

    subgraph after ["④ After the release"]
        L["Delete the branch, verify on PyPI"]
    end

    style dev fill:#f0f7ff,stroke:#4a90d9,stroke-width:2px,color:#1a1a1a
    style release fill:#fff8f0,stroke:#e6820e,stroke-width:2px,color:#1a1a1a
    style publish fill:#f0fff0,stroke:#2d8a4e,stroke-width:2px,color:#1a1a1a
    style after fill:#f8f0ff,stroke:#7c4dff,stroke-width:2px,color:#1a1a1a

    style B fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    style D fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style F fill:#fce4ec,stroke:#c62828
    style J fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style K fill:#fff9c4,stroke:#f57f17,stroke-width:3px
```