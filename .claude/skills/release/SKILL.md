---
name: release
description: Cut an xmris release — run the full CI matrix via a release branch, land the bump on main through a PR, then tag the merged commit to trigger the PyPI publish. User-triggered only.
disable-model-invocation: true
---

# Release xmris

Guided checklist for cutting a release. The full workflow, rationale, and CI/CD diagram live in
`docs/contribute/publishing.md` — this skill is the operational checklist that runs it.
Publishing is automated via GitHub Actions (`ci-publish.yml`) using PyPI Trusted Publishing (OIDC) — **you never run `uv publish` by hand**; pushing a `vX.Y.Z` tag does it.

`$ARGUMENTS` is the bump level (`patch` | `minor` | `major`) or an explicit version (`0.7.0`). If empty, ask which.

Pushing branches and tags triggers CI and an irreversible PyPI publish. **Confirm with the user before any push**, and stop if a step fails.

**Never bump the version until the full matrix is green.** The release branch runs first, *unbumped*; the bump lands only once CI passes. This is the whole point of separating CI from CD — it avoids the "bump → push → CI fails → bump again" cycle.

## Checklist

1. **Preconditions.** On `main`, working tree clean, up to date with origin. Confirm the fast CI on `main` is green (`gh run list --branch main --limit 3`).

2. **Determine the target version.** Show the current version (`uv version`) and the target implied by `$ARGUMENTS` (e.g. `0.2.0`). Confirm with the user — but **do not bump yet**.

3. **Run the full matrix — and write the changelog while it runs.** Push an *unbumped* `release/vX.Y.Z` branch — this triggers the full Ubuntu/Windows/macOS × Py 3.10–3.13 matrix in `ci-publish.yml`. Wait for it and check results with `gh run watch` / `gh run list`. If a job fails, fix it on the release branch and push again until green. **Every leg blocks, macOS included** — it was `continue-on-error` only while official `pyamares` hard-required `hlsvdpro` (no arm64 wheel); the `fitting` extra now depends on `pyamares-xmris`, whose platform marker skips it, so the exemption was removed in #105.

   The matrix takes ~15 minutes and nothing depends on it here. Spend them on the changelog: **invoke the `changelog` skill** for `vX.Y.Z`. Its section in `docs/changelog.md` rides the bump commit in step 4, so the entry and the version land together in one pull request.

4. **Bump and merge.** Once the matrix is green, bump on the release branch and land it through a PR — `main` takes no direct pushes:
   ```
   uv version --bump <level>                 # or set the explicit version
   git commit -am "chore: bump version to X.Y.Z"
   git push origin release/vX.Y.Z
   gh pr create --base main --title "chore: bump version to X.Y.Z"
   ```
   The changelog section from step 3 belongs in this commit. Drive the PR's six checks green; the user merges it (squash). Expect **two** CI cycles here: pushing the bump re-triggers the full matrix on `release/**` (harmless, and it does test the bumped tree), while the PR's six fast checks are the actual merge gate — do not wait on the matrix to merge.

5. **Tag & publish.** Tag the *merged* commit on `main` — **never the release branch**. PRs are squash-merged, so a tag cut before the merge would sit on a commit that never enters `main`'s history, leaving `git describe` on `main` blind to the release:
   ```
   git checkout main && git pull
   git log --oneline -1                          # MUST be the bump commit -- see below
   uv version                                    # MUST print X.Y.Z
   grep -n "^## vX.Y.Z" docs/changelog.md        # MUST exist, and MUST NOT say "unreleased"
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
   The changelog grep is the guard against the one failure this workflow cannot undo cheaply: a tag pushed for a version the changelog never described. Catch it here, before the tag — `ci-publish.yml` deliberately knows nothing about the changelog, because a check that fires *after* the tag would leave you deleting and re-pushing one.

   Check that tip before tagging: anything merged into `main` between the bump and the tag ships in
   this release without ever having seen the pre-flight matrix. If something did land, decide
   deliberately — either ship it (the tag re-runs the full matrix anyway) or tag the bump commit
   explicitly by SHA.
   The tag re-runs the full matrix and then triggers `uv build --no-sources` + `uv publish` to PyPI via OIDC. Watch the publish run to confirm success.

6. **Announce it on GitHub.** Once the publish run is green and the version is live on PyPI, create the GitHub Release — this is what fills a body that used to be left empty:
   ```
   gh release create vX.Y.Z --title vX.Y.Z --notes-file <notes.md>
   ```
   Write `notes.md` to the scratchpad from the `vX.Y.Z` section you produced in step 3, converting the MyST target links (`[The Two Domains](#domains)`) to full docs URLs — GitHub cannot resolve them. Nothing in CI does this: creating the release by hand keeps `contents: write` off the job that publishes to PyPI, and means the announcement only ever exists for a version that actually shipped.

7. **Reconcile the roadmap.** `docs/roadmap.md` expresses release state by band membership — the conventions live on the page itself, under "How this page changes". Move the cards this release shipped into the Shipped band, point the Shipped chip at the new version, clear or add `roadmap-status--merged` chips to match `main`, and update the `Last edited` line. Land it as a small `docs(roadmap)` pull request — bookkeeping, not a design pass.

8. **Wrap up.** Confirm the new version is live on PyPI and `main` is in the expected state. Delete the release branch. Summarize what shipped.
