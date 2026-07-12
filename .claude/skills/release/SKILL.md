---
name: release
description: Cut an xmris release — bump the version, run the full CI matrix via a release branch, and tag to trigger the PyPI publish. User-triggered only.
disable-model-invocation: true
---

# Release xmris

Guided checklist for cutting a release. Publishing is automated via GitHub Actions (`ci-publish.yml`) using PyPI Trusted Publishing (OIDC) — **you never run `uv publish` by hand**; pushing a `vX.Y.Z` tag does it.

`$ARGUMENTS` is the bump level (`patch` | `minor` | `major`) or an explicit version (`0.7.0`). If empty, ask which.

Pushing branches and tags triggers CI and an irreversible PyPI publish. **Confirm with the user before any push**, and stop if a step fails.

## Checklist

1. **Preconditions.** On `main`, working tree clean, up to date with origin. Confirm the fast CI on `main` is green (`gh run list --branch main --limit 3`). Never bump a version while CI is red.

2. **Determine the version.** Show the current version (`uv version`) and the target implied by `$ARGUMENTS`. Confirm with the user.

3. **Bump.** `uv version --bump <level>` (or set the explicit version). Commit: `chore: release vX.Y.Z`.

4. **Run the full matrix.** Push a `release/vX.Y.Z` branch — this triggers the full Ubuntu/Windows/macOS × Py 3.10–3.13 matrix in `ci-publish.yml`. Wait for it and check results with `gh run watch` / `gh run list`. Note: **macOS is allowed to fail** (`continue-on-error`, upstream pyAMARES issue) — a red macOS leg alone does not block the release; everything else must be green.

5. **Tag & publish.** Once the matrix is green, tag and push:
   ```
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
   The tag triggers `uv build --no-sources` + `uv publish` to PyPI via OIDC. Watch the publish run to confirm success.

6. **Wrap up.** Confirm the new version is live on PyPI and the `main`/release branches are in the expected state. Summarize what shipped.
