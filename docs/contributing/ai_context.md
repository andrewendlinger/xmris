# Context for AI

**System Instructions for the LLM:**
You are an expert Python developer assisting a human in building and maintaining `xmris`.
Read this entire document carefully before writing any code. It dictates the strict architectural patterns, tech stack, and coding conventions of this project. Do not deviate from these rules.

### 1. Project Overview

* **Name:** `xmris`
* **Purpose:** An N-dimensional, xarray-based toolbox for Magnetic Resonance Imaging (MRI) and Spectroscopy (MRS).
* **Core Philosophy:** "Xarray in, xarray out." The pipeline is entirely functional. We leverage xarray's named dimensions, coordinates, and attributes to preserve physics metadata, track data lineage, and avoid the alignment errors common in pure NumPy arrays.

### 2. Tech Stack & Tooling

* **Package Manager:** `uv` (fast, isolated virtual environments).
* **Data Structures:** `xarray`, `numpy`.
* **Testing:** `pytest` paired with the `nbmake` plugin (tests are executed directly inside Jupyter Notebooks).
* **Documentation:** Standalone `mystmd` CLI (for rendering) paired with `quartodoc` (for scraping Python API docstrings). Jupytext is used to manage notebook files as `py:percent` scripts.

### 3. Architecture & Namespacing

Functions are strictly segregated into domain-specific modules under the hood based on their physical domain (time vs. frequency), but are exposed to the user via a unified `xarray` **Accessor**.

* **Internal Modules (`src/xmris/`):**
* `fourier.py`: Core mathematical transforms (e.g., `fft`, `fftc`, `fftshift`).
* `fid.py`: Strictly time-domain operations (e.g., `apodize_exp`, `zero_fill`, `to_spectrum`).
* `phase.py`: Strictly frequency-domain operations (e.g., `phase`, `autophase`).


* **User API (`src/xmris/accessor.py`):** Users interact via the `.xmr` namespace. Example: `da.xmr.fftc(dim="Time")`. All user-facing functions must be registered here.

### 4. Strict Coding Rules (The "7 Commandments")

Whenever you generate a new function for `xmris`, you MUST follow these rules:

1. **Xarray First:** The first argument of every internal function is `da: xr.DataArray`.
2. **Functional Purity:** NEVER modify data in-place. Always return a *new* `xr.DataArray`. The safest way to do this is using `da.copy(data=new_vals)`.
3. **Data Lineage:** You MUST preserve coordinates and attributes. If the function applies a processing parameter (like line broadening, or phase angles), append it to the dataset's attributes (`da.assign_attrs()`) so the user has a permanent record of what was done to the data.
4. **Named Dimensions:** Ban the word `axis` from API arguments. Always use `dim` (string) or `dims` (list of strings).
5. **Type Hinting:** Fully type-hint all function signatures. Use `xr.DataArray` consistently so they cross-reference correctly in the docs.
6. **NumPy Docstrings:** Use standard NumPy-style docstrings. These are critical as `quartodoc` parses them for the online API reference.
7. **Accessor Integration:** Any user-facing function must be mapped to the `XmrisAccessor` class.

### 5. Testing & Documentation Strategy

We do not use traditional hidden `test_*.py` files. Our tests *are* our documentation.
We use **Jupyter Notebooks** in the `notebooks/` directory, managed via Jupytext (`py:percent` format).

When asked to write tests for a new function, generate a Jupytext script structure that includes:

1. Markdown cells explaining the math/physics.
2. Python cells generating synthetic, noisy `xarray` data.
3. Python cells applying the `xmris` function and plotting the result.
4. **CRITICAL:** Python cells containing strict `assert` or `np.testing.assert_allclose` statements to mathematically prove the output values AND prove that xarray dimensions, coordinates, and attributes were preserved.
5. **HIDE TESTS:** You MUST add the `# %% tags=["remove-cell"]` metadata to any cell containing pure `assert` statements so `mystmd` hides them from the final rendered website, while `nbmake` still executes them in CI.

### 6. Common Developer Commands

(Provide these to the human if they ask how to run things):

* **Run Tests:** `uv run pytest` (automatically runs normal tests and `nbmake` on the notebooks).
* **Update API Docs:** `uv run docs-api` (runs quartodoc, cleans formatting, and links Xarray types).
* **Live-Edit Tutorials:** `uv run docs-notebooks` (spins up the fast MyST hot-reload server).
* **Full Docs Build:** `uv run docs-all` (generates API and starts the preview server).

### Example Function Template

```python
import xarray as xr
import numpy as np

def example_func(da: xr.DataArray, dim: str = "Time", scale: float = 1.0) -> xr.DataArray:
    """
    NumPy docstring here.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    dim : str, optional
        Dimension to process, by default "Time".
    scale : float, optional
        Scaling factor applied, by default 1.0.

    Returns
    -------
    xr.DataArray
        Processed data with preserved lineage.
    """
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