# 01 Contributing to `xmris`

Welcome! `xmris` is built on a strict **"xarray in, xarray out"** philosophy. Our goal is to make MR imaging and spectroscopy processing functional, robust, and N-dimensional.

To keep the codebase clean and the documentation pristine, we follow a modern "docs-as-tests" pipeline using **MyST**, **quartodoc**, and **uv**.

---

### Step 1: Write the Standalone Function

Functions live in specific domain modules (`xmris.fid`, `xmris.phase`, `xmris.fourier`, etc.). **Respect the physical domain:** if an operation is strictly a frequency-domain manipulation (like phase correction), it belongs in a frequency-focused module, not mixed into time-domain FID scripts.

1. **Xarray First:** The first argument must be `da: xr.DataArray`. Always return a *new* `xr.DataArray`.
2. **Named Dimensions:** Use `dim` (string) or `dims` (list) instead of `axis`.
3. **Data Lineage (Crucial):** Always pass along existing coordinates and attributes. If your function calculates a new parameter (e.g., phase angles, line broadening factors), append it to the `.attrs` dictionary to preserve the processing history.
4. **Type Hinting:** Fully type-hint signatures. These are used by `quartodoc` to generate the API reference.
5. **NumPy Docstrings:** Use standard NumPy format. These are the source of truth for our online API docs.

```python
def my_func(da: xr.DataArray, dim: str = "Time", my_param: float = 1.0) -> xr.DataArray:
    """
    Description for the API docs.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    dim : str, optional
        Dimension to process, by default "Time".
    my_param : float, optional
        A parameter used for processing, by default 1.0.
    """
    # ... mathematical operations ...
    
    # 1. Update data
    da_new = da.copy(data=new_vals)
    
    # 2. Preserve lineage by logging new parameters
    new_attrs = da.attrs.copy()
    new_attrs.update({"my_param_used": my_param})
    
    return da_new.assign_attrs(new_attrs)

```

---

### Step 2: Register the Accessor

Expose your function through the `.xmr` namespace in `src/xmris/accessor.py`. This allows users to chain methods cleanly: `da.xmr.fft().xmr.my_func()`. Ensure the method signature matches your standalone function exactly.

---

### Step 3: Create the Tutorial-Test (Jupytext)

We don't use standard `test_*.py` files. Your tutorials *are* the test suite. Create a Python script in `01_notebooks/` using the Jupytext percent format (`# %%`).

1. **Explain:** Use Markdown cells to explain the math, physics, or purpose of the function.
2. **Demonstrate:** Show the function in action with `matplotlib` plots.
3. **Verify Math & Metadata:** Use `assert` statements to prove the math is correct, but **also assert that dimensions, coordinates, and attributes were not accidentally dropped.**
4. **Hide Tests:** Add `# %% tags=["remove-cell"]` to assertion cells. This ensures `pytest --nbmake` checks them during CI/CD, but the generated website stays clean for readers.

---

### Step 4: Update the Build Pipeline

1. **API Docs:** Run `uv run docs-api`. This triggers `quartodoc` to "scrape" your new function's docstring into the API Reference.
2. **Navigation:** Add your new tutorial to the `nav` section in `myst.yml`.
3. **Verify Build:** Run `uv run docs` to ensure the site renders correctly and the search index updates seamlessly.

---

### ✅ Contributor Checklist

* [ ] Placed in the correct domain module (e.g., time vs. frequency).
* [ ] First argument is `xr.DataArray`; returns a new `xr.DataArray`.
* [ ] Uses `dim`/`dims` for target axes, not `axis` integers.
* [ ] **Preserves `da.coords` and updates `da.attrs` with applied parameters for data lineage.**
* [ ] NumPy docstrings are complete and type-hinted (for `quartodoc`).
* [ ] Mapped appropriately to `XmrisAccessor` in `accessor.py`.
* [ ] Created a `01_notebooks/` Jupytext script for documentation.
* [ ] **Crucial:** Assertion cells test *both* math and metadata preservation, and are tagged with `remove-cell`.
* [ ] Verified locally via `uv run pytest` and `uv run docs`.