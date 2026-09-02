# Contributing to xmris

Thanks for considering a contribution to **xmris** — an xarray-based toolbox for MRI and MRS.

The full, always-current contribution guidelines live in the rendered documentation:

### 📖 https://andrewendlinger.github.io/xmris/guide

They are organised by the *kind of change* you are making — adding a processing method, an
interactive widget, a documentation page, or a dev-diary decision record — and each kind has its own
short page with a live checklist.

## Quick start

```bash
git clone https://github.com/andrewendlinger/xmris
cd xmris
uv sync --all-extras --dev   # uv replaces pip/virtualenv — see the setup guide
uv run lint                  # formatting, lint, docs notebook format — what the Lint check runs
uv run docs-format --fix     # repair docs pages that drifted from jupytext's canonical form
uv run test                  # regenerate the notebook tests and run the full suite
```

Run `lint` and `test` before you push: together they reproduce four of the six checks that gate a merge.

The package manager is [`uv`](https://docs.astral.sh/uv/); please never use `pip`. The [environment
setup guide](https://andrewendlinger.github.io/xmris/setup) covers the full toolchain (`uv`, `ruff`,
`mystmd`).

## Questions, bugs, and ideas

Please open an issue at https://github.com/andrewendlinger/xmris/issues. For a substantial change,
start with an issue or discussion before a large PR so we can agree on the approach first.

## Community guidelines

Be respectful and constructive in issues, reviews, and discussions. We follow the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/). Reports of unacceptable behaviour can
be raised privately with the maintainers via a GitHub issue or direct contact.
