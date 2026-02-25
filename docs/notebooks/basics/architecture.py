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
# Imagine you are writing a standard Python function to process an MRI Free Induction Decay (FID) signal. You need the raw data, but to do anything meaningful (like converting frequencies to ppm, or applying a digital filter), you also need the sequence metadata.
#
# If we built `xmris` like a traditional library, a simple processing pipeline would look like this:

# %%
# ❌ The Anti-Pattern: Parameter Soup
def process_mrs(data, b0_field, mhz, dwell_time, group_delay, phase_0, phase_1):
    # ... math happens here ...
    pass


# User code:
processed = process_mrs(raw_data, 3.0, 127.8, 0.0005, 68.0, 45.0, -10.0)

# %% [markdown]
# ```{admonition} Why is this bad?
# :class: warning
# 1. **Cognitive Load:** The user has to manually lug around a dozen variables through every step of their pipeline.
# 2. **Fragility:** If you swap the order of `phase_0` and `phase_1`, the code won't crash—it will just silently give you the wrong scientific result.
# 3. **Boilerplate:** Your functions become 90% argument definitions and 10% actual math.
# ```
#
# ### The `xarray` Solution
#
# To avoid parameter soup, `xmris` is built natively on top of `xarray`.
#
#
#
# By encapsulating the data, the dimensions (axes), the coordinates (labels), and the metadata (`attrs`) into a single object, the data carries its own context. Our pipeline now looks like this:

# %%
# ✅ The xmris Way: Encapsulated State
processed_data = raw_data.xmr.phase().xmr.to_ppm()

# %% [markdown]
# Notice how `phase()` and `to_ppm()` take *zero* arguments? The functions automatically look inside `raw_data.attrs` to find the phase angles and the spectrometer frequency.

# %% [markdown]
# ## 2. The Danger of "Hidden State"
#
# Encapsulation is beautiful, but it introduces a new, highly dangerous problem: **Magic Strings and Hidden State.**
#
# If `to_ppm()` implicitly hunts for the frequency in `data.attrs["MHz"]`, what happens if the user's data doesn't have that attribute? Or what if they spelled it `"mhz"`?
#
# ::: {dropdown} 💥 Click to see the dreaded KeyError
# ```python
# # Deep inside numpy...
# ppm_coords = hz_coords / self._obj.attrs["MHz"]
# KeyError: 'MHz'
# ```
# *There is nothing worse than a pipeline crashing deep inside a library with a cryptic error message.*
# :::
#
# Furthermore, how does the user even *know* that `to_ppm()` requires `"MHz"`? If we just type it in the docstring, the documentation will inevitably drift out of sync with the actual code.
#
# ---

# %% [markdown]
# ## 3. Building the Data Dictionary
#
# To solve the magic string problem, we realized `xmris` needed a **single source of truth** for its vocabulary.
#
# We built a centralized, immutable Data Dictionary (`xmris.core.config`). Instead of using floating strings like `"Time"` or `"MHz"`, the entire backend relies on typed, frozen configurations.

# %%
from xmris.core import ATTRS, COORDS, DIMS

# In Jupyter, simply typing 'ATTRS' renders a beautiful,
# publication-ready HTML table explaining exactly what
# attributes xmris expects and their physical units!
ATTRS

# %% [markdown]
# ```{tip} Try it out!
# Because these are Python `dataclasses` (and not TOML files), your IDE will autocomplete `ATTRS.b0_field` for you, completely eliminating typos.
# ```

# %% [markdown]
# ## 4. The "Bouncer" Pattern (Decorators)
#
# With our vocabulary locked in, we needed a way to enforce it. We created a decorator engine (`@requires_attrs`) that acts as a bouncer at the door of our processing functions.
#
# ```{mermaid}
# flowchart LR
#     User(User calls .to_ppm) --> Bouncer{Decorator checks data.attrs}
#     Bouncer -- Missing 'MHz' --> Error[Raise Frustration-Free ValueError]
#     Bouncer -- Has 'MHz' --> Math[Execute function logic]
# ```
#
# This decorator does two brilliant things:
#
# 1. **Fail-Fast Execution:** It intercepts the call *before* the math starts.
# 2. **DRY Documentation:** It dynamically injects the required attributes into the function's docstring at import time. The docs can never drift from the code!
#
# ### A Frustration-Free UX
# We hate unhelpful errors. If the bouncer kicks you out, it tells you exactly how to get back in using standard `xarray` code:
#
# ```{dropdown} 💡 View the xmris Error Message
# `ValueError: Method 'to_ppm' requires the following missing attributes in obj.attrs: ['MHz'].`
#
# `To fix this, assign them using standard xarray methods:`
# `    >>> obj = obj.assign_attrs({'MHz': value})`
# ```

# %% [markdown]
# ## 5. Dimensions vs. Attributes: The Great Divide
#
# You might be wondering: *"If decorators are so great for attributes, why don't you use them for dimensions like `Time` or `Frequency`?"*
#
# This was the most important architectural decision we made. We treat **Dimensions** and **Attributes** entirely differently.
#
# ### Attributes are "Hidden State"
# A B0 field strength is a physical constant. You don't apply an operation *to* the B0 field; the math just requires it to exist in the background. Because it is hidden, it needs strict guarding by our `@requires_attrs` decorator.
#
# ### Dimensions are an "Action Space"
# When you apply an FFT, you are actively choosing an axis to act upon.
# We want you to have the flexibility to say, *"Apply this math to the 't' axis."* #
# If we strictly forced you to rename your axes to `xmris` standards before doing *any* math, the package would feel incredibly rigid and annoying to use with quick-and-dirty datasets.
#
# Therefore, dimensions are passed as **explicit arguments with smart defaults**:


# %%
# 🧠 The implementation of an xmris processing function
def apodize_exp(self, dim: str = DIMS.time, lb: float = 1.0):
    # The default is our standard 'Time', but the user can pass 't', 'time_axis', etc.
    ...


# %% [markdown]
# ## Summary
#
# And that's it! By combining the power of `xarray` encapsulation, a strongly-typed Data Dictionary, fail-fast decorators for hidden state, and flexible arguments for action spaces, `xmris` achieves the holy grail of scientific Python packages:
#
# * **It is rigorously safe** (no silent math failures).
# * **It is highly transparent** (docstrings generate themselves).
# * **It is incredibly easy to use** (clean, chainable APIs with no parameter soup).
#
# Happy processing!
