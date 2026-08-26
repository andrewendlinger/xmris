(diary-architecture-contract)=
# The Commandments now run against the code they govern

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-21 · #103, #166</span>

The [architecture rules](#contract) are the standing law for everything under
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
(which do return the *decorated* source, decorators included) and runs the pipeline the page
preaches:

```python
fid = xmris.simulate_fid(...)  # reference_frequency and carrier_ppm set
spectrum = fid.xmr.apodize_exp(lb=5.0).xmr.to_spectrum().xmr.to_ppm()
assert spectrum.attrs["apodization_lb"] == 5.0  # lineage — Commandment 3, live
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

(diary-architecture-contract-changed)=
## What changed from the plan

- **One planned `_check_dims` adoption was impossible.** `build_fid` checks its `dims` argument —
  a plain list, before any `DataArray` exists — so the helper (which takes a `DataArray`) cannot
  apply. That check stays hand-rolled by design; the other two call sites converted cleanly.
- **The warn-only anchor rot fired on day one — from the inside.** mystmd parses directive options
  as raw strings, not YAML, so quoting the `start-at:` anchors made the build search for the quote
  characters themselves. The build stayed exit 0 with four warnings; the grep-for-warnings
  verification step caught it, which is precisely the failure mode the `inspect.getsource` pins
  exist for.
- **One Commandment 7 violation was left standing.** `build_fid`'s repetition coordinate still
  hand-builds its attrs dict: `COORDS` has no `repetition` term, and growing the vocabulary is a
  Commandment 4 event of its own — flagged as follow-up rather than smuggled in.
- All four assumptions marked in the draft held: `inspect.getsource` sees the decorators; the kernelspec'd page
  executes in the docs build while `test-gen` still walks exactly its 26 tutorial/explainer files;
  the pinned mystmd's anchor behavior is what the build showed; and the new `TestAccessorDefaults`
  row runs without importing pyAMARES.
