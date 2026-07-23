(diary-architecture-contract)=
# The Commandments now run against the code they govern

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-07-23</span>

The [architecture rules](../contributing/contract.md) are the standing law for everything under
`src/xmris/` — and the law had quietly stopped describing the land. Commandment 3 bans descriptive
strings from `.attrs`, yet the vocabulary itself blesses `baseline_method="als"`. The module map
files `processing/` under `core/`, where it does not live. The page's teaching template stacks two
decorators no real function combines and records a lineage attr that does not exist in `config.py`.
Each drift was invisible for the same reason: the page is prose, hand-copied from code that kept
moving, and nothing could ever fail ([#72](https://github.com/andrewendlinger/xmris/issues/72)).

:::{important}
The "Context for AI" page is retired: the rules become a lean **Architecture Contract** whose
exemplars are quoted and executed from the real source on every PR build — drift now fails CI
instead of accumulating.
:::

(diary-architecture-contract-mechanism)=
## What makes it drift-proof

The page stops carrying hand-copies. Each numbered rule states its law in a few lines plus one line
naming its enforcer — the test class that guards it, or "reviewer checks" where none does. Three
rules the tests already enforce but no Commandment stated join as 9–11: thin accessor delegators,
errors that end with a copy-pasteable fix, and the `# xmris-diagnostic-dim` escape hatch. Ordinals
1–8 stay stable, because code and tooling cite them by number.

The exemplars are `{literalinclude}`s of the real `apodize_exp` and `to_ppm` — but an unmatched
anchor only *warns* in mystmd, so a hidden cell pins both slices with `inspect.getsource` asserts
and runs the pipeline the page preaches:

```python
fid = xmris.simulate_fid(...)          # reference_frequency and carrier_ppm set
spec = fid.xmr.apodize_exp(lb=5.0).xmr.to_spectrum().xmr.to_ppm()
assert "apodization_lb" in spec.attrs  # lineage appended — Commandment 3, live
```

Commandment 3 itself is rewritten to the law the code actually follows: preserve, then append the
parameters applied — scalars, config-blessed strings, lists — and never a state flag. The open
provenance question (`xmr_history`,
[#64](https://github.com/andrewendlinger/xmris/issues/64)) stays open; the page links it rather
than pre-empting it.

:::{dropdown} Why not keep patching the page in place?
It preserves the "System Instructions for the LLM" framing on a public site page, and the ~60–70%
of content duplicated from CLAUDE.md, the skills, and the explainers — the very sync burden that
produced this drift. Patching buys one accurate snapshot; the next drift starts immediately.
:::

:::{dropdown} Why not dissolve the page entirely?
The numbered Commandments are cited by ordinal from code comments and tooling, and the skills this
branch just shipped route to "the one canonical doc" per concept. Dissolving removes the citeable
home everything was just pointed at, for the price of re-homing every rule anyway.
:::

:::{attention} Assumptions to verify
- `inspect.getsource` on the decorated `apodize_exp` returns the *decorated* source — the pins
  assume it sees `@computes_in(TIME_DIMS)`, not the wrapper inside `validation.py`.
- A kernelspec'd page under `docs/contributing/` executes in the PR docs build while staying out of
  `uv run test`, as the docs-page skill claims.
- The anchor-miss-warns-only behavior observed in the local mystmd 1.10.1 bundle matches CI.
- Adding `fit_amares` to `TestAccessorDefaults` inspects its signature without triggering the lazy
  pyAMARES import.
:::
