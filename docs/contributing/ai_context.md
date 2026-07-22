# Context for AI

**System Instructions for the LLM:**
You are an expert Python developer assisting a human in building and maintaining `xmris`. 
Read this entire document carefully before writing any code. It dictates the strict architectural patterns, tech stack, and coding conventions of this project. Do not deviate from these rules.

### 1. Project Overview

* **Name:** `xmris`
* **Purpose:** An N-dimensional, xarray-based toolbox for Magnetic Resonance Imaging (MRI) and Spectroscopy (MRS).
* **Core Philosophy:** "Xarray in, xarray out." The pipeline is entirely functional. We leverage xarray's named dimensions, coordinates, and attributes to preserve physics metadata, track data lineage, and avoid alignment errors. 

### 2. Tech Stack & Tooling

* **Package Manager:** `uv` (fast, isolated virtual environments).
* **Data Structures:** `xarray`, `numpy`.
* **Testing:** `pytest` paired with the `nbmake` plugin (tests are executed directly inside Jupyter Notebooks).
* **Documentation:** Standalone `mystmd` CLI paired with `quartodoc`. Jupytext is used to manage notebook files as `py:percent` scripts.

### 3. Architecture & Namespacing

Functions are strictly segregated into domain-specific nested modules under the hood, but are exposed to the user via a unified `xarray` **Accessor**.

* **Internal Modules (`src/xmris/core/` and beyond):**
    * `config.py`: Contains the global singletons (`ATTRS`, `DIMS`, `COORDS`, `VARS`) which serve as the single source of truth for xarray string keys and metadata.
    * `validation.py`: Contains the validation decorators — `@requires_attrs` (gate) and the domain decorators `@ensures_domain` (funnel) / `@computes_in` (domain-preserving). See `docs/explanation/domains.md` for the contract design.
    * `options.py`: Contains `set_options` (e.g. `auto_convert` strict mode).
    * `processing/`: Core mathematical transforms (e.g., `fourier.py`, `fid.py`, `phase.py`).
    * `vendor/`: Hardware-specific sanitization (e.g., `bruker.py`).
    * `fitting/`: Mathematical modeling (e.g., `amares.py`).

* **User API (`src/xmris/core/accessor.py`):** Users interact via the `.xmr` namespace (e.g., `da.xmr.to_ppm()`). All user-facing functions must be registered to the `XmrisAccessor` class.

### 4. Strict Coding Rules (The "8 Commandments")

Whenever you generate a new function for `xmris`, you MUST follow these rules:

1. **Xarray First:** The pipeline relies on `xarray.DataArray` and `xarray.Dataset`.
2. **Functional Purity:** NEVER modify data in-place. Always return a *new* object.
3. **Data Lineage:** You MUST preserve coordinates and attributes. Append new processing parameters to `da.attrs` so the user has a permanent record of what was done to the data. Keep `.attrs` strictly to quantifiable mathematical parameters applied (e.g., `phase_p0=15.0`); do NOT use boolean flags or descriptive string flags (e.g., `phase_applied=True`) to avoid metadata bloat.
4. **No Magic Strings (The Config):** NEVER hardcode raw strings for dimensions (like `"time"`) or attributes (like `"reference_frequency"`). Import the singletons `ATTRS`, `DIMS`, `COORDS`, and `VARS` from `xmris.core.config`. These contain `XmrisTerm` objects that evaluate as strings but carry `.unit` and `.long_name` metadata. Note that this only applies for INSIDE the xmris package and must not affect user code and examples. The user is free to use 'time' etc. to keep the entrance barrier low.
    * **CRITICAL:** If a new function requires a dimension, coordinate, or lineage attribute that is not already in the vocabulary, **you must propose adding it to `config.py`**. Furthermore, you must **explicitly highlight and mention** to the user in your response any new `ATTRS`, `DIMS`, or `COORDS` you are introducing so they can be tracked.
5. **Dimension Defaults (the biconditional):** A `dim` argument defaults to the config constant (e.g., `def func(da, dim: str = DIMS.time):`) — EXCEPT it must default to `None` **iff** the function is domain-decorated with a *multi-label* domain (`@ensures_domain(SPECTRAL_DIMS)` / `@computes_in(SPECTRAL_DIMS)`), whose merged resolution fills it at call time (`frequency` [Hz] vs `chemical_shift` [ppm]). This rule is mechanically enforced by `TestDomainDimRule` in `tests/test_core.py`.
6. **Strict Validation & Domain Contracts:**
    * Validate hidden state (attributes) using the `@requires_attrs(...)` decorator (free-function style: the first argument is the DataArray).
    * Declare a domain-sensitive function's working domain with `@ensures_domain(DOMAIN)` (*funnel*: only meaningful in that domain, result stays there — e.g. `autophase`, `baseline_als`) or `@computes_in(DOMAIN)` (*domain-preserving*: same physics either side, representation restored — e.g. `apodize_exp`, `zero_fill`). Converters (`to_spectrum`/`to_fid`/`to_ppm`/`to_hz`), low-level primitives (`phase`, `fft` family), and fitting stay undecorated: their transforms are explicit by design. Never inline `fft`/`ifft` for domain handling — route through the converters. See `docs/explanation/domains.md` for the decision tree.
    * Validate dimensions explicitly inside the function using `_check_dims(da, dim, "func_name")`.
7. **Coordinate Building:** When creating new coordinates, do not manually mutate `.attrs`. Instead, use the internal `as_variable(TERM, dim, data)` helper to bundle data and metadata into a fully formed `xr.Variable` before assigning it via `.assign_coords()`.
8. **MyST Markdown Links:** When writing documentation, never rely on auto-generated header slugs for internal links. Always define explicit MyST targets (e.g., `(my-target)=`) immediately above the header, and link to it via `[text](#my-target)`.

### 5. Testing & Documentation Strategy

We do not use traditional hidden `test_*.py` files for mathematical processing. Our tests *are* our documentation. We use **Jupyter Notebooks** managed via Jupytext (`md:myst` format): tutorials under `docs/notebooks/`, and concept explainers under `docs/explanation/` that carry a kernelspec. `uv run test` walks both trees, so a live cell in an explainer is executed and asserted exactly like a tutorial cell. (Note: Architecture is tested in standard pytest files).

When asked to write notebook tests for a new function, generate a Jupytext script structure that includes:
1. Markdown cells explaining the math/physics.
2. Python cells generating synthetic, noisy `xarray` data.
3. Python cells applying the `xmris` function and plotting the result.
4. **CRITICAL:** Python cells containing strict `assert` or `np.testing.assert_allclose` statements to mathematically prove the output values AND prove that xarray dimensions, coordinates, and attributes were preserved.
5. **HIDE TESTS:** You MUST add the `# %% tags=["remove-cell"]` metadata to any cell containing pure `assert` statements so `mystmd` hides them from the final rendered website, while `nbmake` still executes them in CI.

In addition, reach for the rich mystmd palette (latex formulas, mermaid diagrams, dropdowns, highlights) wherever it carries the argument — nothing decorative.

### Example Accessor Function Template

(utils for reference)
```python
# src/xmris/core/utils.py
import numpy as np
import xarray as xr

from xmris.core.config import XmrisTerm


def _check_dims(da: xr.DataArray, dims: str | list[str], method_name: str) -> None:
    """Validate that required dimensions exist in the DataArray."""
    dims_to_check = [dims] if isinstance(dims, str) else dims
    missing = [d for d in dims_to_check if d not in da.dims]

    if missing:
        raise ValueError(
            f"Method '{method_name}' attempted to operate on missing "
            f"dimension(s): {missing}.\n"
            f"Available dimensions are: {list(da.dims)}.\n\n"
            f"To fix this, either pass the correct `dim` string argument to the function,"
            f" or rename your data's axes using xarray:\n"
            f"    >>> obj = obj.rename({{{repr(missing[0])}: 'correct_name'}})"
        )


def as_variable(term: XmrisTerm, dims: str | tuple, data: np.ndarray) -> xr.Variable:
    """Wrap a numpy array into an xarray Variable.

    Automatically apply the correct units and long_name from the provided XmrisTerm.
    """
    attrs = {"long_name": term.long_name}
    if term.unit:
        attrs["units"] = term.unit

    return xr.Variable(dims, data, attrs=attrs)
```

(example processing function — decorators live on the free function; the accessor method is a thin `return example_func(self._obj, ...)` delegator)
```python
import xarray as xr
import numpy as np

from xmris.core.config import ATTRS, COORDS, DIMS, TIME_DIMS
from xmris.core.utils import _check_dims, as_variable
from xmris.core.validation import computes_in, requires_attrs

# 1. Validate hidden state (attributes) at the door, and declare the domain
#    contract: computes_in = domain-preserving (representation restored);
#    use ensures_domain for funnel ops. Omit for converters/primitives/fitting.
@requires_attrs(ATTRS.reference_frequency)
@computes_in(TIME_DIMS)
def example_func(da: xr.DataArray, dim: str = DIMS.time, scale: float = 1.0) -> xr.DataArray:
    """
    NumPy docstring here.

    Parameters
    ----------
    da : xr.DataArray
        The input time-domain data.
    dim : str, optional
        Dimension to process. Defaults to DIMS.time.
        (Spectral ops decorated with a multi-label domain default to None instead
        — the domain decorator resolves it. See Commandment 5.)
    scale : float, optional
        Scaling factor applied, by default 1.0.
    """
    # 2. Validate the action space (dimensions)
    _check_dims(da, dim, "example_func")

    # 3. Extract physics constants safely (decorator guarantees they exist)
    mhz = da.attrs[ATTRS.reference_frequency]

    # 4. Perform pure mathematics
    new_vals = (da.data * scale) / mhz

    # 5. Build new coordinates safely using XmrisTerm metadata
    new_time_coords = da.coords[dim].values * 2.0
    time_var = as_variable(COORDS.time, dim, new_time_coords)

    # 6. Rebuild DataArray and assign variables
    da_new = da.copy(data=new_vals)
    da_new = da_new.assign_coords({COORDS.time: time_var})

    # 7. Preserve lineage by appending new processing parameters
    da_new.attrs[ATTRS.example_scale_applied] = scale

    return da_new
```

