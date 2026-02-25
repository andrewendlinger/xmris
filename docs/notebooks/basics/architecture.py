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

# %%
# ❌ The Anti-Pattern: Parameter Soup
def remove_filter(data, group_delay, dwell_time):
    ...

def apodize(data, dwell_time, lb):
    ...

def fft_to_spectrum(data):
    ...

def to_ppm(data, mhz):
    ...

def autophase(data, mhz, dwell_time):
    ...


# User code — threading the same metadata through every step:
data = remove_filter(raw, group_delay=68.0, dwell_time=0.0005)
data = apodize(data, dwell_time=0.0005, lb=5.0)
data = fft_to_spectrum(data)
data = to_ppm(data, mhz=300.15)
data = autophase(data, mhz=300.15, dwell_time=0.0005)

# %% [markdown]
# ```{admonition} Why is this bad?
# :class: warning
# 1. **Cognitive Load:** The user has to manually thread `mhz` and `dwell_time` through every single step, even though those values never change within a single experiment.
# 2. **Fragility:** If you swap `phase_0` and `phase_1` in a function call, the code won't crash — it will silently give you the wrong scientific result.
# 3. **Boilerplate:** Every function signature becomes 80% parameter plumbing and 20% actual science.
# ```
#
# ### The `xarray` Solution
#
# To avoid parameter soup, `xmris` is built natively on top of [xarray](https://docs.xarray.dev/en/stable/). An `xarray.DataArray` bundles together the raw data, named dimensions (axes), coordinates (axis labels), and arbitrary metadata (`.attrs`) into a single, self-describing object.
#
# Here is what an `xmris` DataArray looks like in practice:

# %%
import numpy as np
import xarray as xr
from xmris.core import ATTRS, DIMS, COORDS

# A typical xmris FID — data + metadata in one object:
n_points = 2048
dwell_time = 0.0005  # seconds

fid = xr.DataArray(
    data=np.random.randn(n_points) + 1j * np.random.randn(n_points),
    dims=[DIMS.time],
    coords={DIMS.time: np.arange(n_points) * dwell_time},
    attrs={
        ATTRS.b0_field: 7.0,               # Tesla
        ATTRS.reference_frequency: 300.15,  # MHz
    },
)

fid

# %% [markdown]
# The data now carries its own context. The five-step pipeline from above collapses to this:

# %%
# ✅ The xmris Way: Encapsulated, Chainable Processing
spectrum = (
    fid
    .xmr.remove_digital_filter(group_delay=68)
    .xmr.apodize_exp(lb=5.0)
    .xmr.to_spectrum()
    .xmr.to_ppm()
    .xmr.autophase()
)

# %% [markdown]
# Notice that `to_ppm()` and `autophase()` take *zero* metadata arguments. They find
# the spectrometer frequency inside `fid.attrs` automatically — and because `xarray`
# preserves attributes through operations, that metadata is still there at step five
# without any effort from you.
#
# The user still passes arguments that represent *choices* (`lb=5.0`, `group_delay=68`),
# but never has to re-supply physical constants of the experiment. The metadata travels
# *with* the data. You never carry it yourself.
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
# ::: {dropdown} 💥 Click to see the dreaded KeyError
# ```python
# # Somewhere deep in xmris internals...
# mhz = self._obj.attrs["MHz"]
# KeyError: 'MHz'
# ```
# *The pipeline crashes deep inside the library with a cryptic error that tells the user nothing about what went wrong or how to fix it.*
# :::
#
# And there is a subtler problem: how does the user even *know* that `to_ppm()` requires `"MHz"`?
# If we document it by hand in a docstring, those docs will inevitably drift out of sync with the actual code.
#
# We needed a system that:
# 1. **Prevents** the crash before it happens.
# 2. **Tells the user** exactly what is wrong and how to fix it.
# 3. **Documents itself** automatically so documentation can never go stale.
#
# The solution has two parts: a **Data Dictionary** and a **Decorator Engine**.
#
# ---

# %% [markdown]
# ## 3. Building the Data Dictionary
#
# To eliminate magic strings, we built a **single source of truth** for the entire vocabulary of `xmris` — the Data Dictionary in `xmris.core.config`.
#
# Instead of scattering raw strings like `"Time"`, `"MHz"`, or `"ppm"` throughout the codebase, every internal access goes through frozen `dataclass` singletons:

# %%
from xmris.core import ATTRS, DIMS, COORDS

# These are typed Python objects, not bare strings.
# Your IDE will autocomplete them — typos become impossible.
print(ATTRS.reference_frequency)  # → "MHz"
print(ATTRS.b0_field)             # → "b0_field"
print(DIMS.time)                  # → "Time"
print(DIMS.frequency)             # → "Frequency"
print(COORDS.ppm)                 # → "ppm"

# %% [markdown]
# Each entry carries rich metadata — a human-readable description, physical units, and the actual
# xarray string key it maps to. In Jupyter, simply type the name to render a formatted reference table:

# %%
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
#     User("User calls\n.xmr.to_ppm()") --> Bouncer{"@requires_attrs\nchecks .attrs"}
#     Bouncer -- "Missing 'MHz'" --> Error["Raise clear ValueError\nwith fix instructions"]
#     Bouncer -- "All present" --> Math["Execute function body\n(pure science, no boilerplate)"]
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
# You might be wondering: *"If decorators are so great for attributes, why don't you use them for dimensions like `Time` or `Frequency`?"*
#
# This was the single most important architectural decision we made. We treat **Dimensions**
# and **Attributes** with fundamentally different strategies, because they play fundamentally
# different roles.
#
# ### Attributes Are "Hidden State"
# A B0 field strength is a physical constant of the experiment. You don't apply an operation
# *to* the B0 field; the math just requires it to exist in the background. Because it is
# invisible, it needs strict guarding by our `@requires_attrs` decorator.
#
# ### Dimensions Are an "Action Space"
# When you apply an FFT or an apodization, you are actively choosing *which axis* to act upon.
# We want you to have the freedom to say, *"apply this to the `t` axis"* — even if your data
# doesn't follow `xmris` naming conventions.
#
# If we strictly forced you to rename your axes to `Time` and `Frequency` before doing *any*
# processing, the package would feel rigid and hostile toward quick-and-dirty datasets.
#
# Therefore, dimensions are passed as **explicit arguments with smart defaults**:

# %%
from xmris.core import DIMS

# Your data uses the xmris standard "Time" dimension? Just use the defaults:
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
# fid.xmr.apodize_exp(dim="nonexistent")
# ```
# ```
# ValueError: Method 'apodize_exp' attempted to operate on missing
# dimension(s): ['nonexistent'].
# Available dimensions are: ['Time'].
#
# To fix this, either pass the correct `dim` string argument to the function,
# or rename your data's axes using xarray:
#     >>> obj = obj.rename({'nonexistent': DIMS.time})
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
# | **Example** | `ATTRS.reference_frequency` → `"MHz"` | `dim=DIMS.time` → `dim="Time"` |
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
#     B --> C{"'b0_field' in .attrs?\n'MHz' in .attrs?"}
#     C -- "No" --> D["Friendly ValueError:\n'assign them with obj.assign_attrs(...)'"]
#     C -- "Yes" --> E["_check_dims validates dim='Frequency'"]
#     E --> F{"'Frequency' in .dims?"}
#     F -- "No" --> G["Friendly ValueError:\n'Available dimensions are: [...]'"]
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
# for hidden state, and explicit arguments for action spaces, `xmris` achieves three goals simultaneously:
#
# * **Rigorously safe** — no silent math failures from swapped parameters or missing metadata.
# * **Highly transparent** — docstrings generate themselves from the config; documentation can never drift from code.
# * **Easy to use** — clean, chainable APIs with zero parameter soup.
#
# ```python
# # This is the entire user interface. The architecture handles the rest.
# spectrum = (
#     fid
#     .xmr.remove_digital_filter(group_delay=68)
#     .xmr.apodize_exp(lb=5.0)
#     .xmr.to_spectrum()
#     .xmr.to_ppm()
#     .xmr.autophase()
# )
# ```
#
# Happy processing!