# Tests
Currently, the primary test suite for `xmris` is integrated into our documentation notebooks 
to ensure that all tutorials and examples remain functional and scientifically accurate.

You can find the test cases here:
👉 the five hands-on chapters under [`xmris/docs/`](../docs) — `basics/`, `pipeline/`,
`fitting/`, `visualization/`, `vendor/` — plus the executable explainers in
[`docs/concepts/`](../docs/concepts)

These are executed via `uv run pytest` (using the `nbmake` plugin).
