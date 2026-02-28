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
    * `validation.py`: Contains the `@requires_attrs` decorator.
    * `processing/`: Core mathematical transforms (e.g., `fourier.py`, `fid.py`, `phase.py`).
    * `vendor/`: Hardware-specific sanitization (e.g., `bruker.py`).
    * `fitting/`: Mathematical modeling (e.g., `amares.py`).

* **User API (`src/xmris/core/accessor.py`):** Users interact via the `.xmr` namespace (e.g., `da.xmr.to_ppm()`). All user-facing functions must be registered to the `XmrisAccessor` class.

### 4. Strict Coding Rules (The "8 Commandments")

Whenever you generate a new function for `xmris`, you MUST follow these rules:

1. **Xarray First:** The pipeline relies on `xarray.DataArray` and `xarray.Dataset`.
2. **Functional Purity:** NEVER modify data in-place. Always return a *new* object.
3. **Data Lineage:** You MUST preserve coordinates and attributes. Append new processing parameters to `da.attrs` so the user has a permanent record of what was done to the data.
4. **No Magic Strings (The Config):** NEVER hardcode raw strings for dimensions (like `"time"`) or attributes (like `"reference_frequency"`). Import the singletons `ATTRS`, `DIMS`, `COORDS`, and `VARS` from `xmris.core.config`. These contain `XmrisTerm` objects that evaluate as strings but carry `.unit` and `.long_name` metadata. Note that this only applies for INSIDE the xmris package and must not affect user code and examples. The user is free to use 'time' etc. to keep the entrance barrier low.
5. **Accessor Defaults:** Method signatures that take a dimension must use the config constant directly as the default (e.g., `def func(self, dim: str = DIMS.time):`). Do NOT default to `None`.
6. **Strict Validation:** * Validate hidden state (attributes) using the `@requires_attrs(...)` decorator.
    * Validate dimensions explicitly inside the function using `_check_dims(self._obj, dim, "func_name")`.
7. **Coordinate Building:** When creating new coordinates, do not manually mutate `.attrs`. Instead, use the internal `as_variable(TERM, dim, data)` helper to bundle data and metadata into a fully formed `xr.Variable` before assigning it via `.assign_coords()`.
8. **MyST Markdown Links:** When writing documentation, never rely on auto-generated header slugs for internal links. Always define explicit MyST targets (e.g., `(my-target)=`) immediately above the header, and link to it via `[text](#my-target)`.

### 5. Testing & Documentation Strategy

We do not use traditional hidden `test_*.py` files for mathematical processing. Our tests *are* our documentation. We use **Jupyter Notebooks** in the `notebooks/` directory, managed via Jupytext (`py:percent` format). (Note: Architecture is tested in standard pytest files).

When asked to write notebook tests for a new function, generate a Jupytext script structure that includes:
1. Markdown cells explaining the math/physics.
2. Python cells generating synthetic, noisy `xarray` data.
3. Python cells applying the `xmris` function and plotting the result.
4. **CRITICAL:** Python cells containing strict `assert` or `np.testing.assert_allclose` statements to mathematically prove the output values AND prove that xarray dimensions, coordinates, and attributes were preserved.
5. **HIDE TESTS:** You MUST add the `# %% tags=["remove-cell"]` metadata to any cell containing pure `assert` statements so `mystmd` hides them from the final rendered website, while `nbmake` still executes them in CI.

### Example Accessor Function Template

```python
import xarray as xr
import numpy as np

from xmris.core.config import ATTRS, DIMS, COORDS
from xmris.core.validation import requires_attrs
from xmris.core.accessor import _check_dims, as_variable

# 1. Validate hidden state (attributes) at the door
@requires_attrs(ATTRS.reference_frequency)
def example_func(self, dim: str = DIMS.time, scale: float = 1.0) -> xr.DataArray:
    """
    NumPy docstring here.

    Parameters
    ----------
    dim : str, optional
        Dimension to process. Defaults to DIMS.time.
    scale : float, optional
        Scaling factor applied, by default 1.0.
    """
    # 2. Validate the action space (dimensions)
    _check_dims(self._obj, dim, "example_func")
    
    # 3. Extract physics constants safely (decorator guarantees they exist)
    mhz = self._obj.attrs[ATTRS.reference_frequency]
    
    # 4. Perform pure mathematics
    raw_data = self._obj.data
    new_vals = (raw_data * scale) / mhz
    
    # 5. Build new coordinates safely using XmrisTerm metadata
    new_time_coords = self._obj.coords[dim].values * 2.0
    time_var = as_variable(COORDS.time, dim, new_time_coords)
    
    # 6. Rebuild DataArray and assign variables
    da_new = self._obj.copy(data=new_vals)
    da_new = da_new.assign_coords({COORDS.time: time_var})
    
    # 7. Preserve lineage by appending new processing parameters
    da_new.attrs["example_scale_applied"] = scale
    
    return da_new

```