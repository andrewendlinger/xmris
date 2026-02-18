# 03 Context for AI

**System Instructions for the LLM:**
You are an expert Python developer assisting a human in building and maintaining `xmris`. 
Read this entire document carefully before writing any code. It dictates the strict architectural patterns, tech stack, and coding conventions of this project. Do not deviate from these rules.

## 1. Project Overview
* **Name:** `xmris`
* **Purpose:** An N-dimensional, xarray-based toolbox for Magnetic Resonance Imaging (MRI) and Spectroscopy (MRS).
* **Core Philosophy:** "Xarray in, xarray out." The pipeline is entirely functional. We leverage xarray's named dimensions and attributes to preserve physics metadata and avoid the alignment errors common in pure NumPy arrays.

## 2. Tech Stack & Tooling
* **Package Manager:** `uv` (fast, isolated virtual environments).
* **Data Structures:** `xarray`, `numpy`.
* **Testing:** `pytest` paired with `pytest-nbmake` (tests are executed inside Jupyter Notebooks).
* **Documentation:** `jupyter-book` (v1.0+, using the `myst` engine).

## 3. Architecture & Namespacing
Functions are grouped into domain-specific modules under the hood, but are exposed to the user via an `xarray` **Accessor**.

* **Internal Modules (`src/xmris/`):**
  * `core.py`: Low-level xarray manipulations (e.g., `to_complex`).
  * `signal.py`: Domain-independent DSP (e.g., `fftc`, `ifftc`).
  * `mrs.py`: Spectroscopy-specific (e.g., `apodize`, `phase_correct`).
  * `mri.py`: Imaging-specific.
  * `vendor.py`: Hardware-specific data parsing/correction.
* **User API (`src/xmris/accessor.py`):** * Users interact via the `.xmr` namespace. Example: `da.xmr.fftc(dim="Time")`.

## 4. Strict Coding Rules (The "7 Commandments")
Whenever you generate a new function for `xmris`, you MUST follow these rules:

1. **Xarray First:** The first argument of every internal function is `da: xr.DataArray`.
2. **Functional Purity:** NEVER modify data in-place. Always return a *new* `xr.DataArray` with the original coordinates and attributes preserved (`coords=da.coords, attrs=da.attrs`).
3. **Named Dimensions:** Ban the word `axis` from API arguments. Always use `dim` (string) or `dims` (list of strings).
4. **Type Hinting:** Fully type-hint all function signatures.
5. **NumPy Docstrings:** Use standard NumPy-style docstrings.
6. **Accessor Integration:** Any user-facing function must be mapped to the `XmrisAccessor` class.

## 5. Testing & Documentation Strategy
We do not use traditional hidden `test_*.py` files. Our tests *are* our documentation.
We use **Jupyter Notebooks** in the `docs/notebooks/` directory.

When asked to write tests for a new function, generate a Jupyter Notebook structure that includes:
1. Markdown cells explaining the math/physics.
2. Python cells generating synthetic xarray data.
3. Python cells applying the `xmris` function.
4. **CRITICAL:** Python cells containing strict `assert` statements to mathematically prove the output values AND prove that xarray dimensions/attributes were preserved.

## 6. Common Developer Commands
(Provide these to the human if they ask how to run things):
* **Run Tests:** `uv run pytest --nbmake docs/notebooks/`
* **Live-Edit Docs (MyST server):** `cd docs && uv run jupyter book start`
* **Build Static Docs:** `cd docs && uv run jupyter book build --html`

## Example Function Template
```python
import xarray as xr

def example_func(da: xr.DataArray, dim: str = "Time", scale: float = 1.0) -> xr.DataArray:
    """NumPy docstring here."""
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' missing.")
    
    # Extract, process, rebuild
    new_vals = da.values * scale
    return xr.DataArray(new_vals, dims=da.dims, coords=da.coords, attrs=da.attrs)
```