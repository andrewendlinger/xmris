# 01 Contributing to `xmris`

Welcome! `xmris` is built on a strict **"xarray in, xarray out"** philosophy. Our goal is to make MR imaging and spectroscopy processing functional, robust, and N-dimensional.

To keep the codebase clean and the documentation pristine, we have established a strict set of developer guidelines. Whenever you add a new function to `xmris`, please follow these 4 steps:

### Step 1: Write the Standalone Function

All functions should live in their appropriate domain module (e.g., `xmris.signal`, `xmris.mrs`, `xmris.mri`, `xmris.vendor`).

When writing your function, adhere to these core rules:

1. **Xarray First:** The first argument must always be `da: xr.DataArray`. Always return a *new* `xr.DataArray` (do not modify data in-place) to preserve functional purity.
2. **Standardized Dimensions:** Never use the word `axis`. Always use `dim` (for a single string) or `dims` (for a list of strings) to leverage xarray's named dimensions.
3. **Type Hinting:** Fully type-hint the function signature.
4. **NumPy Docstrings:** Include a standard NumPy-style docstring. Optionally, with an `Examples:` section showing basic usage.

**Example Template:**

```python
import xarray as xr

def my_new_function(da: xr.DataArray, dim: str = "Time", factor: float = 1.0) -> xr.DataArray:
    """
    Brief description of what the function does.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    dim : str, optional
        The dimension to operate along. Default is "Time".
    factor : float, optional
        A scaling factor.

    Returns
    -------
    xr.DataArray
        The processed DataArray with all attributes preserved.
    """
    # 1. Validate dimensions
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found.")
        
    # 2. Extract values, do math, rebuild DataArray
    new_values = da.values * factor
    
    return xr.DataArray(
        new_values, dims=da.dims, coords=da.coords, attrs=da.attrs
    )

```

### Step 2: Register it in the `.xmr` Accessor

Users should rarely call your function directly. Instead, expose it through the xarray accessor so users can chain methods.

Open `src/xmris/accessor.py`, import your new function, and add it to the `XmrisAccessor` class:

```python
from xmris.mrs import my_new_function

@xr.register_dataarray_accessor("xmr")
class XmrisAccessor:
    # ... existing methods ...

    def my_new_function(self, dim: str = "Time", factor: float = 1.0) -> xr.DataArray:
        """Applies my new function."""
        return my_new_function(self._obj, dim=dim, factor=factor)

```

### Step 3: Write the Combined Test & Tutorial Notebook

We use **Jupyter Book** for documentation and **pytest-nbmake** for testing. We do not write traditional, hidden test files. Instead, your tests *are* your documentation.

Create a new Jupyter Notebook in `docs/notebooks/` (e.g., `docs/notebooks/tutorial_my_function.ipynb`).

1. Write Markdown cells explaining the math or physics behind your function.
2. Write Python cells applying your function (`da.xmr.my_new_function()`) to synthetic or sample data and plot the results.
3. **The Crucial Step:** Include `assert` statements in your cells to mathematically prove your function worked and that xarray metadata was preserved.

When CI/CD runs `uv run pytest --nbmake docs/notebooks/`, it will execute your tutorial and fail if any `assert` statement fails.

### Step 4: Update the Architecture Diagram & Docs

If you added a major new module or changed how components interact:

1. Update the `mermaid.js` diagram (usually located in the developer docs or `README.md`) to reflect the new architecture.
2. Add your new notebook to the `docs/_toc.yml` file so it appears in the Jupyter Book sidebar.

---

### ✅ The Quick Contributor Checklist

* [ ] First argument is an `xarray.DataArray`.
* [ ] Returns a new `xarray.DataArray` (preserves coords/attrs).
* [ ] Uses `dim`/`dims` instead of `axis`.
* [ ] Function is fully type-hinted.
* [ ] Docstring follows standard NumPy format.
* [ ] Function is mapped to the `XmrisAccessor` in `accessor.py`.
* [ ] A tutorial Jupyter Notebook is created in `docs/notebooks/`.
* [ ] The notebook contains `assert` statements to test the math/metadata.
* [ ] Tests pass locally (`uv run pytest --nbmake docs/notebooks/`).
* [ ] `docs/_toc.yml` and Mermaid diagrams are updated.