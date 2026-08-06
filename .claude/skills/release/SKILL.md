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

3. **Run the full matrix.** Push an *unbumped* `release/vX.Y.Z` branch — this triggers the full Ubuntu/Windows/macOS × Py 3.10–3.13 matrix in `ci-publish.yml`. Wait for it and check results with `gh run watch` / `gh run list`. If a job fails, fix it on the release branch and push again until green. **Every leg blocks, macOS included** — it was `continue-on-error` only while official `pyamares` hard-required `hlsvdpro` (no arm64 wheel); the `fitting` extra now depends on `pyamares-xmris`, whose platform marker skips it, so the exemption was removed in #105.

4. **Bump and merge.** Once the matrix is green, bump on the release branch and land it through a PR — `main` takes no direct pushes:
   ```
   uv version --bump <level>                 # or set the explicit version
   git commit -am "chore: bump version to X.Y.Z"
   git push origin release/vX.Y.Z
   gh pr create --base main --title "chore: bump version to X.Y.Z"
   ```
   Drive its four checks green; the user merges it (squash). Expect **two** CI cycles here: pushing
   the bump re-triggers the full matrix on `release/**` (harmless, and it does test the bumped tree),
   while the PR's four fast checks are the actual merge gate — do not wait on the matrix to merge.

5. **Tag & publish.** Tag the *merged* commit on `main` — **never the release branch**. PRs are squash-merged, so a tag cut before the merge would sit on a commit that never enters `main`'s history, leaving `git describe` on `main` blind to the release:
   ```
   git checkout main && git pull
   git log --oneline -1          # MUST be the bump commit -- see below
   uv version                    # MUST print X.Y.Z
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
   Check that tip before tagging: anything merged into `main` between the bump and the tag ships in
   this release without ever having seen the pre-flight matrix. If something did land, decide
   deliberately — either ship it (the tag re-runs the full matrix anyway) or tag the bump commit
   explicitly by SHA.
   The tag re-runs the full matrix and then triggers `uv build --no-sources` + `uv publish` to PyPI via OIDC. Watch the publish run to confirm success. Delete the release branch afterwards.

6. **Wrap up.** Confirm the new version is live on PyPI and `main` is in the expected state. Summarize what shipped.
