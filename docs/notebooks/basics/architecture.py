# %% [markdown]
# # The xmris Architecture: Why We Built It This Way
#
# Welcome to the engine room of `xmris`! If you are wondering why we rely so heavily on `xarray`, why we don't just pass sequence parameters as function arguments, or what the deal is with our decorators, you are in the right place.
#
# This guide reads a bit like a story. We will walk through the exact problems we faced when designing this package, and the architectural decisions we made to solve them.
#
# Let's dive in.
#
# ---

# %% [markdown]
# ## 1. The Parameter Soup Problem
#
# Imagine you are writing Python functions to process an MRI Free Induction Decay (FID) signal. You need the raw data, but to do anything meaningful — converting frequencies to ppm, removing a digital filter, auto-phasing — you also need the scanner metadata: the spectrometer frequency, the B0 field, the dwell time, and so on.
#
# If we built `xmris` like a traditional library, a simple processing pipeline would look like this:

# %% [markdown]
# ❌ The Anti-Pattern: Parameter Soup
# ```python
# def apodize(data, dwell_time, lb): ...
#
#
# def fft_to_spectrum(data): ...
#
#
# def to_ppm(data, mhz): ...
#
#
# def autophase(data, mhz, dwell_time): ...
#
#
# User code — threading the same metadata through every step:
# data = apodize(data, dwell_time=0.0005, lb=5.0)
# data = fft_to_spectrum(data)
# data = to_ppm(data, mhz=300.15)
# data = autophase(data, mhz=300.15, dwell_time=0.0005)
# ```

# %% [markdown]
# ```{admonition} Why is this bad?
# :class: warning
# 1. **Cognitive Load:** The user has to manually thread `mhz` and `dwell_time` through every single step, even though those values never change within a single experiment.
# 2. **Boilerplate:** Every function signature becomes 80% parameter plumbing and 20% actual science.
# 3. **Unnamed axes:** Every operation implicitly acts on "the first axis" or "axis 0". There is no way to say *which* axis you mean by name — and with multidimensional data (e.g., spatial × spectral), positional indexing becomes a minefield.
# ```
#
# That last point deserves emphasis. In raw numpy, axes are just integers:
#
# ```python
# # Which axis is time? Which is space? Hope you remember.
# result = np.fft.fft(data, axis=0)
# result = np.fft.fft(data, axis=1)  # wrong axis, no error, wrong science
# ```
#
# ### The `xarray` Solution
#
# To avoid parameter soup, `xmris` is built natively on top of [xarray](https://docs.xarray.dev/en/stable/). An `xarray.DataArray` bundles together the raw data, named dimensions ("numpy" axes), coordinates (axis labels), and arbitrary metadata (`.attrs`) into a single, self-describing object.
#
# Here is what an `xmris` DataArray looks like in practice:

# %%
import numpy as np
import xarray as xr

# A typical xmris FID — data + metadata in one object:
n_points = 2048
dwell_time = 0.0005  # seconds

fid = xr.DataArray(
    data=np.random.randn(n_points) + 1j * np.random.randn(n_points),
    dims=["time"],
    coords={"time": np.arange(n_points) * dwell_time},
    attrs={
        "b0_field": 7.0,  # Tesla
        "reference_frequency": 300.15,  # MHz
    },
)

fid

# %% [markdown]
# The data now carries its own context. The pipeline from above collapses to this:

# %% [markdown]
# ```python
# # ✅ The xmris Way: Encapsulated, Chainable Processing
# spectrum = (
#     fid
#     .xmr.apodize_exp(lb=5.0)
#     .xmr.to_spectrum()
#     .xmr.to_ppm()
#     .xmr.autophase()
# )
# ```

# %% [markdown]
# Two things happened here:
#
# 1. **Metadata travels with the data.** `to_ppm()` and `autophase()` take *zero*
#    metadata arguments. They find the spectrometer frequency inside `fid.attrs`
#    automatically — and because `xarray` preserves attributes through operations,
#    that metadata is still there at step four without any effort from you.
#
# 2. **Operations act on named dimensions, not integer positions.** Instead of
#    remembering whether time is `axis=0` or `axis=1`, you refer to axes by
#    name. Each function has a sensible default — `to_spectrum()` defaults to
#    `dim="time"` — but you can always pass a different name if your data uses
#    different conventions:

# %%
# activate the 'xmr' accessor
import xmris

# Create a 2D complex MRSI dataset:
# 16 spatial voxels × 2048 temporal samples (FID signal)
mrsi_fid = xr.DataArray(
    data=np.random.randn(16, 2048) + 1j * np.random.randn(16, 2048),
    dims=["voxel", "time"],
    coords={
        "voxel": np.arange(16),
        "time": np.arange(2048) * dwell_time,
    },
    attrs={"b0_field": 7.0, "reference_frequency": 300.15},
)

# Transform from time domain (FID) to frequency domain (spectrum)
# by explicitly specifying the dimension
mrsi_spectrum = mrsi_fid.xmr.to_spectrum(dim="time")

# Even better: "time" is the default transform dimension for to_spectrum,
# so the same result can be obtained using
mrsi_spectrum = mrsi_fid.xmr.to_spectrum()

# %% [markdown]
# Compare this to the numpy equivalent, where you'd have to track that time is
# `axis=1` (and hope nobody transposes the array upstream):
#
# ```python
# # 🤞 numpy — is time axis 0 or 1? Better check.
# result = np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)
# ```
#
# The user still passes arguments that represent *choices* (`lb=5.0`),
# but never has to re-supply physical constants of the experiment or remember
# which integer axis is which. The metadata and the axis semantics travel
# *with* the data. You never carry them yourself.
#
# ---

# %% [markdown]
# ## 2. The Danger of "Hidden State"
#
# Encapsulation is beautiful, but it introduces a dangerous new problem: **magic strings and hidden state.**
#
# If `to_ppm()` implicitly reads the frequency from `data.attrs["MHz"]`, three things can go wrong:
#
# 1. The user's data doesn't have that attribute.
# 2. The user spelled it `"mhz"` or `"ref_freq"`.
# 3. The user has no way of knowing `"MHz"` was required in the first place.
#
# A naive implementation would look like this:
#
# ```python
# # 💥 Naive approach — no safeguards:
# def to_ppm(self, dim="Frequency"):
#     mhz = self._obj.attrs["MHz"]  # ← what if "MHz" doesn't exist?
#     ppm_coords = self._obj.coords[dim].values / mhz
#     return self._obj.assign_coords({"ppm": (dim, ppm_coords)})
# ```
#
#
# And there is a subtler problem: how does the user even *know* that `to_ppm()` requires `"MHz"`?
# If we document it by hand in a docstring, those docs will inevitably drift out of sync with the actual code.
#
# We needed a system that:
# 1. **Prevents** the crash before it happens.
# 2. **Tells the user** exactly what is wrong and how to fix it.
# 3. **Documents itself** automatically so documentation can never go stale.
#
# The solution has two parts: a **Data Dictionary** (see section 3) and a **Decorator Engine** (see section 4).
#
#
# :::{dropdown} What's a decorator?
# A decorator is a Python function that **wraps another function** to add
# behavior before or after it runs — without modifying the function's own code.
# You apply one with the `@` syntax:
#
# ```python
# @requires_attrs(ATTRS.reference_frequency)
# def to_ppm(self, dim="frequency"):
#     ...
# ```
#
# This is equivalent to writing:
#
# ```python
# def to_ppm(self, dim="frequency"):
#     ...
#
# to_ppm = requires_attrs(ATTRS.reference_frequency)(to_ppm)
# ```
#
# The decorator returns a new function that first checks whether
# `reference_frequency` exists in `.attrs`, and only then calls the
# original `to_ppm`. The original function never contains any validation
# code — the decorator handles it from the outside.
# :::
#
# ---

# %% [markdown]
# ## 3. Building the Data Dictionary
#
# To eliminate magic strings, we built a **single source of truth** for the entire vocabulary of `xmris` — the Data Dictionary in `xmris.core.config`.
#
# Instead of scattering raw strings like `"Time"`, `"MHz"`, or `"ppm"` throughout the codebase, every internal access goes through frozen `dataclass` singletons:
#
#
# :::{dropdown} What is a singleton?
# A singleton is a design pattern where only **one instance** of a class ever
# exists in the entire program. In xmris, the config objects are created once
# at the bottom of `config.py`:
#
# ```python
# ATTRS = XmrisAttributes()
# DIMS = XmrisDimensions()
# COORDS = XmrisCoordinates()
# ```
#
# Every module that does `from xmris.core import ATTRS` gets a reference to
# the **same object**. There is no way to accidentally create a second,
# conflicting vocabulary. Combined with the `frozen=True` dataclass decorator
# (which prevents modification after creation), this guarantees that the
# vocabulary is both **global** and **immutable** — a single source of truth
# that cannot drift.
# :::

# %%
from xmris.core import ATTRS, COORDS, DIMS

# These are typed Python objects, not bare strings.
# Your IDE will autocomplete them — typos become impossible.
print(f"{ATTRS.reference_frequency=}")
print(f"{ATTRS.b0_field=}")
print(f"{DIMS.time=}")
print(f"{DIMS.frequency=}")
print(f"{COORDS.chemical_shift=}")

# %% [markdown]
# Each entry carries rich metadata — a human-readable description, physical units, and the actual
# xarray string key it maps to. In Jupyter, simply type the name to render a formatted reference table:

# %%
print("This code cell ran and produced this ⬇️ overview.")
ATTRS

# %%
DIMS

# %%
COORDS

# %% [markdown]
# ```{tip}
# Because `ATTRS`, `DIMS`, and `COORDS` are Python `dataclass` instances (not TOML files or plain dicts),
# your IDE provides full autocomplete and type checking. A typo like `ATTRS.referece_frequency` raises
# an `AttributeError` at import time — not a silent bug three hours into a processing run.
# ```
#
# ### How the Dictionary Is Used Internally
#
# Throughout the `xmris` codebase, **no function uses a bare string to access metadata.** Every
# attribute access, dimension reference, and coordinate name goes through the config:
#
# ```python
# # ❌ Never this:
# mhz = self._obj.attrs["MHz"]
# ppm_coords = hz_coords / mhz
# self._obj.assign_coords({"ppm": (dim, ppm_coords)})
#
# # ✅ Always this:
# mhz = self._obj.attrs[ATTRS.reference_frequency]
# ppm_coords = hz_coords / mhz
# self._obj.assign_coords({COORDS.ppm: (dim, ppm_coords)})
# ```
#
# This means if the underlying key ever changes (e.g., `"MHz"` → `"spectrometer_frequency"`),
# we update it in *one place* — the dataclass field default — and the entire package updates automatically.
#
# ---

# %% [markdown]
# ## 4. The "Bouncer" Pattern (Decorators)
#
# With our vocabulary locked in, we needed a way to **enforce** it at runtime. We created a
# decorator engine, `@requires_attrs`, that acts as a bouncer at the door of every function
# that depends on hidden state.
#
# Here is the actual source code for `to_ppm`, straight from the `xmris` codebase:
#
# ```python
# # From xmris/core/accessor.py:
# @requires_attrs(ATTRS.b0_field, ATTRS.reference_frequency)
# def to_ppm(self, dim: str = DIMS.frequency) -> xr.DataArray:
#     """Convert the frequency axis coordinates from Hz to ppm."""
#     # Safe! The decorator already verified these exist before we got here.
#     mhz = self._obj.attrs[ATTRS.reference_frequency]
#     hz_coords = self._obj.coords[dim].values
#     ppm_coords = hz_coords / mhz
#     return self._obj.assign_coords({COORDS.ppm: (dim, ppm_coords)})
# ```
#
# ```{mermaid}
# flowchart LR
#     User("User calls<br>.xmr.to_ppm()") --> Bouncer{"@requires_attrs<br>checks .attrs"}
#     Bouncer -- "Missing 'MHz'" --> Error["❌<br>Raise clear ValueError<br>with fix instructions"]
#     Bouncer -- "All present" --> Math["✅<br>Execute function body"]
# ```
#
# The decorator does two things:
#
# ### 1. Fail-Fast with Helpful Errors
#
# If a required attribute is missing, the bouncer intercepts the call *before* any math runs
# and tells the user exactly what is wrong and how to fix it using standard `xarray` methods:
#
# ````{dropdown} 💡 Click to view the actual xmris error message
# ```python
# spectrum.xmr.to_ppm()
# ```
# ```
# ValueError: Method 'to_ppm' requires the following missing attributes
# in `obj.attrs`: ['b0_field', 'MHz'].
#
# To fix this, assign them using standard xarray methods:
#     >>> obj = obj.assign_attrs({'b0_field': value})
# ```
# No `KeyError`. No stack trace through numpy internals. Just a clear message with
# copy-pasteable fix code.
# ````
#
# ### 2. Self-Documenting Functions
#
# At import time, the decorator dynamically injects a **"Required Attributes"** section into
# each function's docstring by pulling descriptions and units directly from the Data Dictionary:
#
# ````{dropdown} 📖 Click to view the auto-generated docstring section
# ```python
# help(spectrum.xmr.to_ppm)
# ```
# ```
# Convert the frequency axis coordinates from Hz to ppm.
#
# ...
#
# Required Attributes
# --------------------
# * ``b0_field``: Static main magnetic field strength. [T]
# * ``MHz``: Spectrometer working/reference frequency. [MHz]
# ```
# Because the docstring is generated from the *same config* that powers the runtime
# validation, it is **physically impossible** for the documentation to drift out of
# sync with the code.
# ````
#
# ---

# %% [markdown]
# ## 5. Dimensions vs. Attributes: The Great Divide
#
# You might be wondering: *"If decorators are so great for attributes, why don't you use them for dimensions to enforce consistent use of e.g. `time` or `frequency`?"*
#
# This was the single most important architectural decision we made. We treat **Dimensions**
# and **Attributes** with different strategies, because they play fundamentally
# different roles.
#
# ### Attributes Are "Hidden State"
# A $B_0$ field strength is a physical constant of the experiment. You don't apply an operation
# *to* the $B_0$ field; the math just requires it to exist in the background. Because it is
# invisible, it needs strict guarding by our `@requires_attrs` decorator.
#
# ### Dimensions Are an "Action Space"
# When you apply an FFT or an apodization, you are actively choosing *which axis* to act upon.
# We want you to have the freedom to say, *"apply this to the `t` axis"* — even if your data
# doesn't follow `xmris` naming conventions.
#
# *If* we strictly forced you to rename your axes to `time` and `frequency` before doing *any*
# processing, the package would feel rigid and hostile toward quick-and-dirty datasets.
#
# Therefore, dimensions are passed as **explicit arguments with smart defaults**:

# %%
from xmris.core import DIMS

# Your data uses the xmris standard "time" dimension? Just use the defaults:
result = fid.xmr.apodize_exp(lb=5.0)

# Your data has a custom axis name? No problem — just pass it:
result = fid.xmr.apodize_exp(dim="t", lb=5.0)

# You can even pass xmris constants explicitly for maximum clarity:
result = fid.xmr.apodize_exp(dim=DIMS.time, lb=5.0)

# %% [markdown]
# And if you pass a dimension that doesn't exist at all, `xmris` gives you a clear,
# actionable error — just like the attribute bouncer:
#
# ````{dropdown} 💡 Click to view the dimension error message
# ```python
# fid.xmr.apodize_exp(dim="randomname")
# ```
# ```
# ValueError: Method 'apodize_exp' attempted to operate on missing
# dimension(s): ['randomname'].
# Available dimensions are: ['Time'].
#
# To fix this, either pass the correct `dim` string argument to the function,
# or rename your data's axes using xarray:
#     >>> obj = obj.rename({'randomname': DIMS.time})
# ```
# ````
#
# ### The Design Rule
#
# Here is the rule we follow throughout the entire codebase:
#
# | | **Attributes** (Hidden State) | **Dimensions** (Action Space) |
# |---|---|---|
# | **Nature** | Physical constants of the experiment | Axes the user chooses to act upon |
# | **Guarded by** | `@requires_attrs` decorator | `_check_dims` helper |
# | **User interface** | Implicit (read from `.attrs`) | Explicit argument with smart default |
# | **Example** | `ATTRS.reference_frequency` → `"MHz"` | `dim=DIMS.time` → `dim="time"` |
#
# ---

# %% [markdown]
# ## Putting It All Together
#
# Let's trace through a single function call — `spectrum.xmr.to_ppm()` — to see every
# architectural layer working in concert:
#
# ```{mermaid}
# flowchart TD
#     A["User calls spectrum.xmr.to_ppm()"] --> B["@requires_attrs decorator fires"]
#     B --> C{"'b0_field' in .attrs?<br>'MHz' in .attrs?"}
#     C -- "No" --> D["Friendly ValueError:<br>'assign them with obj.assign_attrs(...)'"]
#     C -- "Yes" --> E["_check_dims validates dim='Frequency'"]
#     E --> F{"'Frequency' in .dims?"}
#     F -- "No" --> G["Friendly ValueError:<br>'Available dimensions are: [...]'"]
#     F -- "Yes" --> H["Execute: ppm = hz / attrs[ATTRS.reference_frequency]"]
#     H --> I["Return DataArray with new COORDS.ppm coordinate"]
# ```
#
# Every layer serves a distinct purpose:
#
# 1. **Config constants** (`ATTRS.b0_field`, `DIMS.frequency`, `COORDS.ppm`) eliminate magic strings everywhere.
# 2. **`@requires_attrs`** catches missing metadata *before* the math runs and auto-generates the docstring.
# 3. **`_check_dims`** validates the dimension argument at call time, listing what's available.
# 4. **The function body** is pure science — no validation code, no defensive `try/except` blocks.

# %% [markdown]
# ## Summary
#
# By combining `xarray` encapsulation, a strongly-typed Data Dictionary, fail-fast decorators
# for hidden state, and explicit arguments for action spaces, `xmris` strives for three goals:
#
# * **Rigorously safe** — no silent math failures from swapped parameters or missing metadata.
# * **Highly transparent** — docstrings generate themselves from the config; documentation can never drift from code.
# * **Easy to use** — clean, chainable APIs with zero parameter soup.
#
#
# Happy processing!
