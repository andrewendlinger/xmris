# Template: the test lists a new method forces you to edit

`tests/test_core.py` is 1615 lines and 192 tests, and **exactly one of its 24 classes discovers
functions automatically**. Everything else is a hardcoded list. A new accessor method that isn't
added to these lists is not "lightly covered" — it is covered by nothing, and the suite stays green
to tell you so.

Three edits, always. A fourth only when you created a new module.

## 1. `TestAccessorDefaults` — `tests/test_core.py:681`

Add one row per `dim`-like parameter. Methods with two axis parameters get two rows (see
`to_spectrum` / `to_fid`).

```diff
             ("zero_fill", "dim", DIMS.time),
+            ("<your_method>", "dim", DIMS.<axis>),
             ("remove_digital_filter", "dim", DIMS.time),
```

**Skip this if your `dim` default is `None`** (multi-label domain) — the list is for constants.
Those methods get a dedicated assertion instead; `test_autophase_dim_is_none_by_design` (`:726`)
and `test_baseline_accessor_mirrors_dim_none` (`:1371`) are the two patterns to copy.

Known limitation, worth not being fooled by: this test compares with `==`, and `XmrisTerm`
subclasses `str`. A bare `"time"` default passes today and only breaks if the config value itself
changes. The list protects against config drift, **not** against magic strings.

## 2. `TestAttrsPreservation` — `tests/test_core.py:815`

Add one method. The shared helper `_assert_attrs_preserved` (`:828`) does the work; pick the
fixture matching your input domain — `valid_fid_da` or `valid_spectrum_da`.

```python
    def test_<your_method>_preserves_attrs(self, valid_fid_da):
        """``<your_method>`` must preserve all input attrs."""
        result = valid_fid_da.xmr.<your_method>(<minimal args>)
        self._assert_attrs_preserved(valid_fid_da, result)
```

Why it can't be skipped: xarray strips `.attrs` by default on arithmetic, `where()`, `concat()` and
more. Every xmris method has to put them back deliberately, and this is the only check that it did.

If your function is decorated with `@ensures_domain`/`@computes_in` and never touches `.attrs`
itself, the decorator machinery is separately covered — but a function that *writes* lineage attrs
is not, so add the test anyway.

:::{warning}
Copy the *structure*, not a neighbouring test body. `test_to_hz_preserves_attrs` (`:849`) is a
verbatim duplicate of the `to_ppm` test below it and calls the wrong method, so `to_hz` has no
coverage at all. Check that the method you call is the method you named.
:::

## 3. `TestDomainRollout` — `tests/test_core.py:1324`

Add your function to exactly one of three tuples, matching its contract:

| Contract | Test | Line | Assertion |
|---|---|---|---|
| Funnel | `test_funnel_ops` | `:1338` | `__xmris_domain__ == (SPECTRAL_DIMS, False)` |
| Domain-preserving | `test_domain_preserving_ops` | `:1345` | `__xmris_domain__ == (TIME_DIMS, True)` |
| Undecorated by design | `test_undecorated_by_design` | `:1356` | `not hasattr(func, "__xmris_domain__")` |

```diff
         for func in (apodize_exp, apodize_lg, zero_fill):
+        for func in (apodize_exp, apodize_lg, zero_fill, <your_func>):
```

(Import it at the top of the test method — these use local imports.)

The third tuple is the one people forget: **an undecorated function still needs pinning**, because
"no decorator" is a deliberate contract here, not an absence of one. Without the entry, someone
later adds a decorator and nothing objects.

## 4. Only if you created a new module — `tests/test_core.py:766`

`TestDomainDimRule` is the one auto-discovering test, but it walks a hardcoded module list:

```diff
     modules = [
         xmris.processing.baseline,
         xmris.processing.fid,
+        xmris.processing.<your_new_module>,
         ...
     ]
```

Miss this and every function in the new module escapes the `dim`-default rule silently — the
`assert checked >= 10` floor keeps passing on the other modules' functions. This is the most
consequential of the four edits and the easiest to skip, because it is the one that isn't triggered
by adding a *function*.

## Verify

```bash
uv run pytest tests/test_core.py -n0 --no-cov
```

Confirm the count went **up** by the number of tests you added — `TestAccessorDefaults` is
parametrized, so each row is its own test. A green run at an unchanged count means your edits
didn't land where you thought.
