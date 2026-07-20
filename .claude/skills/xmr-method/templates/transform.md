# Template: a transform, end to end

`docs/contributing/ai_context.md` owns the annotated `example_func` skeleton — the validation
order, `_check_dims`, `as_variable`, the lineage rule. **Read that first; it is not repeated here.**

This file carries what it doesn't: which decorator stack to copy, the accessor delegator, and the
export step.

## 1. Pick a decorator stack by copying a real one

Four stacks exist in the library. Find yours in `domains.md`, then copy the matching exemplar
rather than assembling one from the rules.

**Funnel** — only meaningful in one domain, result stays there. Multi-label domain, so `dim`
defaults to `None` (`src/xmris/processing/baseline.py:44`):

```python
@ensures_domain(SPECTRAL_DIMS)
def baseline_als(
    da: xr.DataArray,
    dim: str | None = None,
    lam: float = 1e5,
    ...
) -> xr.DataArray:
```

Inside the body, `dim` is guaranteed non-`None` — the decorator's merged resolution fills it. The
real function says so in a comment at `baseline.py:94`; keep that habit, it stops the next reader
from adding a redundant `if dim is None` guard.

**Domain-preserving** — same physics either side, representation restored. Single-label domain, so
`dim` defaults to the constant (`src/xmris/processing/fid.py:104`):

```python
@computes_in(TIME_DIMS)
def apodize_exp(da: xr.DataArray, dim: str = DIMS.time, lb: float = 1.0) -> xr.DataArray:
```

`zero_fill` (`fid.py:202`) is the same stack and the more instructive read: it *changes the array
length* and the contract still holds — the decorator restores the input representation, not the
input shape.

**Attribute gate, no domain tier** — physics constants must exist, but the function is a converter
so its transform stays explicit (`src/xmris/processing/referencing.py:18`):

```python
@requires_attrs(ATTRS.reference_frequency, ATTRS.carrier_ppm)
def to_ppm(da: xr.DataArray, dim: str = DIMS.frequency) -> xr.DataArray:
```

**Undecorated** — converters, FFT primitives, fitting, vendor loaders
(`src/xmris/processing/fid.py:10`):

```python
def to_spectrum(
    da: xr.DataArray, dim: str = DIMS.time, out_dim: str = DIMS.frequency
) -> xr.DataArray:
```

:::{note}
There is currently **no example of `@requires_attrs` stacked on a domain decorator**. If yours
needs both, you are writing the first one — put the gate on top (it should reject before any
conversion happens) and say so in your report.
:::

### The biconditional, in one line

`dim` defaults to `None` **iff** the domain is multi-label. Today `SPECTRAL_DIMS` (frequency +
chemical_shift) is the only multi-label domain; `TIME_DIMS` has one member. `TestDomainDimRule`
enforces this automatically — it is the one rule you cannot get wrong silently.

## 2. Write the accessor delegator

The method is a **thin forwarder** — no logic, ever. Two docstring styles are in use; both are
fine, so match the mixin you are landing in.

Terse, for self-evident signatures (`src/xmris/core/accessor.py:343`):

```python
    def to_ppm(self, dim: str = DIMS.frequency) -> xr.DataArray:
        """Convert relative frequency axis [Hz] to absolute chemical shift axis [ppm]."""
        return to_ppm(self._obj, dim=dim)
```

Full NumPy docstring, for anything with tunable parameters (`accessor.py:435`):

```python
    def apodize_exp(self, dim: str = DIMS.time, lb: float = 1.0) -> xr.DataArray:
        """
        Multiply the time-domain signal by a decreasing mono-exponential filter.

        Parameters
        ----------
        dim : str, optional
            The dimension corresponding to time, by default `DIMS.time`.
        lb : float, optional
            The desired line broadening factor in Hz, by default 1.0.

        Returns
        -------
        xr.DataArray
            A new apodized DataArray, preserving coordinates and attributes.
        """
        return apodize_exp(self._obj, dim=dim, lb=lb)
```

Three ways this goes wrong, none of which any test catches:

- **Defaults drift.** The delegator's defaults must equal the free function's. Copy them, don't
  retype them.
- **Parameters get dropped.** If the free function grows a keyword and the delegator doesn't, it is
  reachable only through `**kwargs` — and the docstring will lie about it. This has already
  happened in-tree.
- **A `da` parameter appears in the delegator docstring.** The method takes `self`, not `da`.
  Document only what the method actually accepts.

## 3. Export it

In `src/xmris/__init__.py`: import the function and add it to `__all__` under the matching labelled
section. Nothing tests this — a missing entry means the function works as
`.xmr.<name>()` but not as `from xmris import <name>`, and the split can survive for releases
before anyone notices.

## 4. Then

The three test-list edits: **`tests.md`**, in this directory. The notebook: the **`docs-page`**
skill, tutorial genre.
