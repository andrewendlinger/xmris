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

* **Internal Modules (`src/xmris/`):**
    * `config.py`: Contains the global `DEFAULTS` object mapping standard dimensions and attributes.
    * `processing/`: Core mathematical transforms (e.g., `fourier.py`, `fid.py`, `phase.py`).
    * `vendor/`: Hardware-specific sanitization (e.g., `bruker.py`).
    * `fitting/`: Mathematical modeling (e.g., `amares.py`).

* **User API (`src/xmris/accessor.py`):** Users interact via the `.xmr` namespace (e.g., `da.xmr.fftc()`). All user-facing functions must be registered here.

### 4. Strict Coding Rules (The "7 Commandments")

Whenever you generate a new function for `xmris`, you MUST follow these rules:

1. **Xarray First:** The first argument of every internal function is `da: xr.DataArray`.
2. **Functional Purity:** NEVER modify data in-place. Always return a *new* `xr.DataArray` (e.g., using `da.copy(data=new_vals)`).
3. **Data Lineage:** You MUST preserve coordinates and attributes. Append new processing parameters to `da.attrs` so the user has a permanent record of what was done to the data.
4. **The `DEFAULTS` Config:** NEVER hardcode magic strings for dimensions (like `"time"`) or attributes (like `"MHz"`). Import `DEFAULTS` from `xmris.config`. Function arguments for dimensions should default to `None` and implement the fallback pattern (e.g., `dim = dim or DEFAULTS.time.dim`).
5. **Type Hinting:** Fully type-hint all function signatures. 
6. **NumPy Docstrings:** Use standard NumPy-style docstrings. These are critical as `quartodoc` parses them for the online API reference.
7. **Accessor Integration:** Any user-facing function must be mapped to the `XmrisAccessor` class.

### 5. Testing & Documentation Strategy

We do not use traditional hidden `test_*.py` files. Our tests *are* our documentation. We use **Jupyter Notebooks** in the `notebooks/` directory, managed via Jupytext (`py:percent` format).

When asked to write tests for a new function, generate a Jupytext script structure that includes:
1. Markdown cells explaining the math/physics.
2. Python cells generating synthetic, noisy `xarray` data.
3. Python cells applying the `xmris` function and plotting the result.
4. **CRITICAL:** Python cells containing strict `assert` or `np.testing.assert_allclose` statements to mathematically prove the output values AND prove that xarray dimensions, coordinates, and attributes were preserved.
5. **HIDE TESTS:** You MUST add the `# %% tags=["remove-cell"]` metadata to any cell containing pure `assert` statements so `mystmd` hides them from the final rendered website, while `nbmake` still executes them in CI.

### Example Function Template

```python
import xarray as xr
import numpy as np
from xmris.config import DEFAULTS

def example_func(da: xr.DataArray, dim: str | None = None, scale: float = 1.0) -> xr.DataArray:
    """
    NumPy docstring here.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    dim : str, optional
        Dimension to process. If None, falls back to DEFAULTS.time.dim.
    scale : float, optional
        Scaling factor applied, by default 1.0.
    """
    # 0. Implement DEFAULTS fallback pattern
    dim = dim or DEFAULTS.time.dim

    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' missing.")
    
    # 1. Process data
    new_vals = da.values * scale
    
    # 2. Rebuild DataArray safely
    da_new = da.copy(data=new_vals)
    
    # 3. Preserve lineage by appending new processing parameters
    new_attrs = da.attrs.copy()
    new_attrs.update({"example_scale_applied": scale})
    
    return da_new.assign_attrs(new_attrs)
```