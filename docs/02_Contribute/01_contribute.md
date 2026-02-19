# 01 Contributing to `xmris`

Welcome! `xmris` is built on a strict **"xarray in, xarray out"** philosophy. Our goal is to make MR imaging and spectroscopy processing functional, robust, and N-dimensional.

To keep the codebase clean and the documentation pristine, we follow a modern "docs-as-tests" pipeline using **MyST**, **quartodoc**, and **uv**.

---

### Step 1: Write the Standalone Function

Functions live in domain modules: `xmris.signal`, `xmris.mrs`, `xmris.mri`, or `xmris.vendor`.

1. **Xarray First:** First argument must be `da: xr.DataArray`. Always return a *new* `xr.DataArray`.
2. **Named Dimensions:** Use `dim` (string) or `dims` (list) instead of `axis`.
3. **Type Hinting:** Fully type-hint signatures. These are used by `quartodoc` to generate the API reference.
4. **NumPy Docstrings:** Use standard NumPy format. These are the source of truth for our online API docs.

```python
def my_func(da: xr.DataArray, dim: str = "Time") -> xr.DataArray:
    """
    Description for the API docs.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    dim : str
        Dimension to process.
    """
    # math...
    return xr.DataArray(new_vals, dims=da.dims, coords=da.coords, attrs=da.attrs)

```

---

### Step 2: Register the Accessor

Expose your function through the `.xmr` namespace in `src/xmris/accessor.py`. This allows users to chain methods: `da.xmr.fft().xmr.my_func()`.

---

### Step 3: Create the Tutorial-Test (Jupytext)

We don't use `test_*.py` files. Your tutorials *are* the test suite. Create a Python script in `01_notebooks/` using the Jupytext percent format.

1. **Explain:** Use Markdown cells for the math/physics.
2. **Demonstrate:** Show the function in action with plots.
3. **Verify:** Use `assert` statements to prove correctness.
4. **Hide Tests:** Add `# %% tags=["remove-cell"]` to assertion cells. This ensures `pytest --nbmake` checks them, but the website stays clean.

---

### Step 4: Update the Build Pipeline

1. **API Docs:** Run `uv run docs-api`. This triggers `quartodoc` to "scrape" your new function's docstring into the API Reference.
2. **Navigation:** Add your new tutorial to the `nav` section in `myst.yml`.
3. **Verify Build:** Run `uv run docs` to ensure the site renders correctly and the search index updates.

---

### ✅ Contributor Checklist

* [ ] Argument is `xr.DataArray`; returns new `xr.DataArray`.
* [ ] Uses `dim`/`dims`, not `axis`.
* [ ] NumPy docstrings are complete (for `quartodoc`).
* [ ] Mapped to `XmrisAccessor` in `accessor.py`.
* [ ] Created a `01_notebooks/` Jupytext script.
* [ ] **Crucial:** Assertion cells are tagged with `remove-cell`.
* [ ] Verified locally via `uv run pytest` and `uv run docs`.
