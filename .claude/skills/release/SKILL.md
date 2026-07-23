---
name: release
description: Cut an xmris release — bump the version, run the full CI matrix via a release branch, and tag to trigger the PyPI publish. User-triggered only.
disable-model-invocation: true
---

# Release xmris

Guided checklist for cutting a release. The full workflow, rationale, and CI/CD diagram live in
`docs/contributing/publishing.md` — this skill is the operational checklist that runs it.
Publishing is automated via GitHub Actions (`ci-publish.yml`) using PyPI Trusted Publishing (OIDC) — **you never run `uv publish` by hand**; pushing a `vX.Y.Z` tag does it.

`$ARGUMENTS` is the bump level (`patch` | `minor` | `major`) or an explicit version (`0.7.0`). If empty, ask which.

Pushing branches and tags triggers CI and an irreversible PyPI publish. **Confirm with the user before any push**, and stop if a step fails.

**Never bump the version until the full matrix is green.** The release branch runs first, *unbumped*; the bump lands only once CI passes. This is the whole point of separating CI from CD — it avoids the "bump → push → CI fails → bump again" cycle.

## Checklist

1. **Preconditions.** On `main`, working tree clean, up to date with origin. Confirm the fast CI on `main` is green (`gh run list --branch main --limit 3`).

2. **Determine the target version.** Show the current version (`uv version`) and the target implied by `$ARGUMENTS` (e.g. `0.2.0`). Confirm with the user — but **do not bump yet**.

3. **Run the full matrix.** Push an *unbumped* `release/vX.Y.Z` branch — this triggers the full Ubuntu/Windows/macOS × Py 3.10–3.13 matrix in `ci-publish.yml`. Wait for it and check results with `gh run watch` / `gh run list`. If a job fails, fix it on the release branch and push again until green. Note: **macOS is allowed to fail** (`continue-on-error`, upstream pyAMARES issue) — a red macOS leg alone does not block the release; everything else must be green.

4. **Bump, tag & publish.** Once the matrix is green, bump on the release branch and ship:
   ```
   uv version --bump <level>                 # or set the explicit version
   git commit -am "chore: bump version to X.Y.Z"
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
   The tag triggers `uv build --no-sources` + `uv publish` to PyPI via OIDC. Watch the publish run to confirm success.

5. **Cleanup.** Merge the release branch back into `main` and delete it; the `vX.Y.Z` tag remains as the permanent release marker.

6. **Wrap up.** Confirm the new version is live on PyPI and `main` is in the expected state. Summarize what shipped.
