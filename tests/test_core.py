"""
Core architecture tests for the xmris package.

This module validates the foundational safety and configuration layers that
the entire package depends on. These tests are intentionally strict; if any
of them fail, it signals a regression in a guarantee that downstream code
or user pipelines depend on.

### What This Module Tests
1. Configuration Singletons: Vocabulary instances are shared globally in memory.
2. Naming Conventions: All xarray keys strictly follow the lowercase convention.
3. Metadata Completeness: Every config field carries a description for auto-docs.
4. Decorator Engine: `@requires_attrs` validates at runtime and injects docstrings.
5. Dimension Validation: `_check_dims` produces actionable errors for missing dims.
6. Accessor Registration: The `.xmr` namespace is available on DataArrays/Datasets.
7. Accessor Defaults: Method signatures use config constants, not bare strings.
8. Attrs Preservation: Processing methods never silently drop `.attrs`.
9. Integration (to_ppm): End-to-end test of the most heavily guarded method.

### Maintenance Guide: When to Update This File

* **Adding a new dimension, coordinate, or attribute to `config.py`:**
  You usually do NOT need to update these tests. The metadata and naming
  convention tests dynamically scan your classes. However, if your new attribute
  becomes globally required by many methods, you must add it to the dummy
  DataArrays in the `Fixtures` section.

* **Adding a new accessor method to `accessor.py`:**
  1. Add the method to the parametrization list in `TestAccessorDefaults` to
     verify it uses configuration constants (e.g., `DIMS.time`) instead of
     bare strings.
  2. Add a basic pass-through test in `TestAttrsPreservation` to guarantee
     your new method does not accidentally strip xarray `.attrs`.

* **Modifying or adding a decorator in `validation.py`:**
  Update the `MockAccessor` class and add specific behavior checks to
  `TestRequiresAttrsRuntime` or `TestRequiresAttrsDocstring`.

* **Changing core mathematical logic:**
  Do not test complex scientific logic here. This file is for architecture.
  Test the math in a separate `test_processing.py` suite. The `TestToPpm`
  class here exists solely as a structural integration test of the pipeline.
"""

import numpy as np
import pytest
import xarray as xr

from xmris.core.accessor import _check_dims
from xmris.core.config import (
    ATTRS,
    COORDS,
    DIMS,
    VARS,
)
from xmris.core.validation import requires_attrs

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_da():
    """Create a minimal 1D real-valued DataArray with a non-standard dimension.

    This fixture represents the "worst case" input: no xmris-standard
    dimensions, no coordinates with physical meaning, and no ``.attrs``.
    It is used to verify that validation layers reject or guide the user
    appropriately.

    Returns
    -------
    xr.DataArray
        Shape (100,) with dim ``"x"`` and integer coordinates.
    """
    rng = np.random.default_rng()
    return xr.DataArray(
        rng.random(100),
        dims=["x"],
        coords={"x": np.arange(100)},
    )


@pytest.fixture
def valid_fid_da():
    """Create a well-formed time-domain FID DataArray.

    Simulates a single-voxel Free Induction Decay with:
    - Complex data (2048 points)
    - Standard ``DIMS.time`` dimension
    - Physical time coordinates derived from a 0.5 ms dwell time
    - Required attrs for downstream processing
        (-> `b0_field`, `reference_frequency`, `carrier_ppm`)

    Returns
    -------
    xr.DataArray
        Shape (2048,), complex128, with dim ``"time"`` and full attrs.
    """
    n = 2048
    dwell = 0.0005
    rng = np.random.default_rng()
    return xr.DataArray(
        rng.standard_normal(n) + 1j * rng.standard_normal(n),
        dims=[DIMS.time],
        coords={DIMS.time: np.arange(n) * dwell},
        attrs={
            ATTRS.b0_field: 7.0,
            ATTRS.reference_frequency: 300.15,
            ATTRS.carrier_ppm: 4.7,
        },
    )


@pytest.fixture
def valid_spectrum_da():
    """Create a well-formed frequency-domain spectrum DataArray.

    Simulates a single-voxel MR spectrum with:
    - Complex data (1024 points)
    - Standard ``DIMS.frequency`` dimension
    - Frequency coordinates spanning ±5000 Hz
    - Full attrs (3T field, 127.8 MHz reference, 4.7 ppm carrier)

    Returns
    -------
    xr.DataArray
        Shape (1024,), complex128, with dim ``"frequency"`` and full attrs.
    """
    n = 1024
    rng = np.random.default_rng()
    return xr.DataArray(
        rng.standard_normal(n) + 1j * rng.standard_normal(n),
        dims=[DIMS.frequency],
        coords={DIMS.frequency: np.linspace(-5000, 5000, n)},
        attrs={
            ATTRS.b0_field: 3.0,
            ATTRS.reference_frequency: 127.8,
            ATTRS.carrier_ppm: 4.7,
        },
    )


@pytest.fixture
def multi_dim_da():
    """Create a 2D DataArray simulating a multi-voxel MRSI FID dataset.

    Has a non-standard spatial dimension (``"voxel"``) alongside the standard
    time dimension. Used to verify that xmris correctly handles N-dimensional
    data and operates on the correct axis.

    Returns
    -------
    xr.DataArray
        Shape (16, 2048), complex128, with dims ``["voxel", "time"]`` and full attrs.
    """
    n_voxels, n_time = 16, 2048
    dwell = 0.0005
    rng = np.random.default_rng()
    return xr.DataArray(
        rng.standard_normal((n_voxels, n_time))
        + 1j * rng.standard_normal((n_voxels, n_time)),
        dims=["voxel", DIMS.time],
        coords={
            "voxel": np.arange(n_voxels),
            DIMS.time: np.arange(n_time) * dwell,
        },
        attrs={
            ATTRS.b0_field: 7.0,
            ATTRS.reference_frequency: 300.15,
            ATTRS.carrier_ppm: 4.7,
        },
    )


# =============================================================================
# 1. Configuration: Singletons
# =============================================================================


class TestConfigSingletons:
    """Vocabulary instances (ATTRS, DIMS, COORDS, VARS) must be singletons.

    The entire xmris architecture depends on these objects acting as a unified
    single source of truth. Multiple imports must resolve to the exact same
    memory address.
    """

    def test_singletons_are_same_object(self):
        """Multiple imports of ATTRS must return the exact same object in memory."""
        from xmris.core.config import ATTRS as ATTRS2

        assert ATTRS is ATTRS2


# =============================================================================
# 2. Configuration: Naming Conventions
# =============================================================================


class TestConfigNamingConventions:
    """All xarray string keys must be lowercase, per the pre-1.0 convention.

    This convention aligns with the xarray ecosystem (CF Conventions, cf-xarray,
    xarray tutorials) and avoids ambiguity with multi-word names
    (e.g., ``"chemical_shift"`` not ``"Chemical_Shift"``).
    """

    @pytest.mark.parametrize("prop_name, term_val", list(DIMS._get_terms().items()))
    def test_dims_are_lowercase(self, prop_name, term_val):
        """Every DIMS field value must be a lowercase string."""
        assert term_val == term_val.lower(), (
            f"DIMS.{prop_name} = {term_val!r} is not lowercase. "
            f"All dimension keys must be lowercase per project convention."
        )

    @pytest.mark.parametrize("prop_name, term_val", list(COORDS._get_terms().items()))
    def test_coords_are_lowercase(self, prop_name, term_val):
        """Every COORDS field value must be a lowercase string."""
        assert term_val == term_val.lower(), (
            f"COORDS.{prop_name} = {term_val!r} is not lowercase."
        )

    @pytest.mark.parametrize("prop_name, term_val", list(ATTRS._get_terms().items()))
    def test_attrs_are_lowercase(self, prop_name, term_val):
        """Every ATTRS field value must be a lowercase string."""
        assert term_val == term_val.lower(), (
            f"ATTRS.{prop_name} = {term_val!r} is not lowercase. "
            f"Consider aligning the xarray key with the Python field name."
        )


# =============================================================================
# 3. Configuration: Metadata Completeness
# =============================================================================


class TestConfigMetadata:
    """Every config field must carry rich metadata for the auto-documentation system.

    The ``@requires_attrs`` decorator and the Jupyter ``_repr_html_`` rendering
    both pull descriptions (and units) from field metadata. A field without a
    description will produce an empty entry in auto-generated docstrings and
    HTML tables.
    """

    @pytest.mark.parametrize("vocab", [ATTRS, DIMS, COORDS, VARS])
    def test_all_fields_have_descriptions(self, vocab):
        """Every field across all vocabularies must have a non-empty description."""
        for prop_name, term in vocab._get_terms().items():
            assert term.description, (
                f"{vocab.__class__.__name__}.{prop_name} is missing a 'description' "
                f"in its metadata."
            )

    def test_get_description_valid_key(self):
        """``get_description`` must return the correct description for a known key."""
        desc = ATTRS.get_description(ATTRS.b0_field)
        assert "magnetic field" in desc.lower()

    def test_get_description_invalid_key(self):
        """``get_description`` must return a fallback string for unknown keys."""
        assert ATTRS.get_description("nonexistent") == "Unknown xarray key."

    @pytest.mark.parametrize("vocab", [ATTRS, DIMS, COORDS, VARS])
    def test_html_repr_renders(self, vocab):
        """The Jupyter HTML table must render without errors and include all fields."""
        html = vocab._repr_html_()
        assert "<table" in html
        for term in vocab._get_terms().values():
            assert str(term) in html


# =============================================================================
# 4. Decorator Engine: @requires_attrs
# =============================================================================


class MockAccessor:
    """Minimal mock of the xmris accessor pattern for testing ``@requires_attrs``.

    Replicates the real accessor's ``self._obj`` convention so the decorator
    can inspect ``.attrs`` without depending on the full ``XmrisAccessor`` class.
    """

    def __init__(self, obj: xr.DataArray):
        """Store the DataArray, matching the real accessor's ``__init__`` signature."""
        self._obj = obj

    @requires_attrs(ATTRS.b0_field, ATTRS.reference_frequency)
    def needs_both(self):
        """Original docstring."""
        return self._obj.attrs[ATTRS.b0_field]

    @requires_attrs(ATTRS.b0_field)
    def needs_one(self):
        """Method requiring only a single attribute."""  # noqa: D401
        return self._obj.attrs[ATTRS.b0_field]

    @requires_attrs(ATTRS.b0_field)
    def no_docstring(self):  # noqa: D102
        pass


class TestRequiresAttrsRuntime:
    """Verify that ``@requires_attrs`` correctly validates at call time.

    The decorator must:
    - Raise ``ValueError`` if any required attr is missing.
    - Include the missing key names and fix instructions in the error message.
    - Pass through to the wrapped function if all attrs are present.
    - Not alter the function's return value.
    """

    def test_all_missing(self, empty_da):
        """All required attrs absent — must raise with a descriptive message."""
        accessor = MockAccessor(empty_da)
        with pytest.raises(ValueError, match="missing attributes"):
            accessor.needs_both()

    def test_partial_missing(self, empty_da):
        """One attr present, one missing — must still raise.

        This catches a potential bug where the decorator short-circuits
        after finding the first present attr instead of checking all of them.
        """
        da = empty_da.assign_attrs({ATTRS.b0_field: 3.0})
        accessor = MockAccessor(da)
        with pytest.raises(ValueError, match="missing attributes"):
            accessor.needs_both()

    def test_all_present(self, valid_spectrum_da):
        """All required attrs present — must execute the function body normally."""
        accessor = MockAccessor(valid_spectrum_da)
        result = accessor.needs_both()
        assert result == 3.0

    def test_error_message_lists_missing_keys(self, empty_da):
        """The error message must name the specific missing attr keys."""
        accessor = MockAccessor(empty_da)
        with pytest.raises(ValueError, match=ATTRS.b0_field):
            accessor.needs_both()

    def test_error_message_includes_fix(self, empty_da):
        """The error message must include copy-pasteable ``assign_attrs`` fix code."""
        accessor = MockAccessor(empty_da)
        with pytest.raises(ValueError, match="assign_attrs"):
            accessor.needs_both()

    def test_returns_original_value(self, valid_spectrum_da):
        """The decorator must be transparent — return value is unchanged."""
        accessor = MockAccessor(valid_spectrum_da)
        assert accessor.needs_one() == valid_spectrum_da.attrs[ATTRS.b0_field]


class TestRequiresAttrsDocstring:
    """Verify that ``@requires_attrs`` injects documentation at import time.

    The decorator appends a "Required Attributes" section to the wrapped
    function's docstring, pulling descriptions from the config vocabulary.
    This ensures docs and code can never drift apart.
    """

    def test_injects_section_header(self):
        """The auto-generated docstring must contain a 'Required Attributes' header."""
        assert "Required Attributes" in MockAccessor.needs_both.__doc__

    def test_injects_key_names(self):
        """Every required attr key must appear in the generated docstring."""
        doc = MockAccessor.needs_both.__doc__
        assert ATTRS.b0_field in doc
        assert ATTRS.reference_frequency in doc

    def test_preserves_original_docstring(self):
        """The original docstring text must not be overwritten by the injection."""
        assert "Original docstring." in MockAccessor.needs_both.__doc__

    def test_handles_no_docstring(self):
        """Decorating a function with ``None`` docstring must not crash.

        The decorator should gracefully create a new docstring containing
        only the 'Required Attributes' section.
        """
        doc = MockAccessor.no_docstring.__doc__
        assert doc is not None
        assert "Required Attributes" in doc

    def test_preserves_function_name(self):
        """``functools.wraps`` must preserve ``__name__`` for introspection and debugging."""  # noqa: E501
        assert MockAccessor.needs_both.__name__ == "needs_both"


# =============================================================================
# 5. Dimension Validation: _check_dims
# =============================================================================


class TestCheckDims:
    """Tests for ``_check_dims``, the internal dimension validation helper.

    This function is called at the top of accessor methods that take a ``dim``
    argument. It produces actionable error messages listing available dimensions
    and suggesting ``xr.DataArray.rename()`` as a fix.
    """

    def test_single_dim_present(self, empty_da):
        """A dimension that exists in the DataArray must pass silently."""
        _check_dims(empty_da, "x", "test_func")

    def test_single_dim_missing(self, empty_da):
        """A missing dimension must raise ``ValueError``."""
        with pytest.raises(ValueError, match="missing dimension"):
            _check_dims(empty_da, DIMS.time, "test_func")

    def test_list_of_dims_all_present(self, multi_dim_da):
        """A list of dimensions that all exist must pass silently."""
        _check_dims(multi_dim_da, ["voxel", DIMS.time], "test_func")

    def test_list_of_dims_partial_missing(self, multi_dim_da):
        """If any dimension in the list is missing, must raise ``ValueError``."""
        with pytest.raises(ValueError, match="missing dimension"):
            _check_dims(multi_dim_da, [DIMS.time, "nonexistent"], "test_func")

    def test_error_lists_available_dims(self, multi_dim_da):
        """The error message must list the dimensions that *do* exist,
        so the user can identify the correct name.
        """  # noqa: D205
        with pytest.raises(ValueError, match="voxel"):
            _check_dims(multi_dim_da, "nonexistent", "test_func")

    def test_error_includes_rename_fix(self, empty_da):
        """The error message must suggest ``xr.DataArray.rename()`` as a fix."""
        with pytest.raises(ValueError, match="rename"):
            _check_dims(empty_da, DIMS.time, "test_func")

    def test_error_includes_function_name(self, empty_da):
        """The error message must name the calling function for traceability."""
        with pytest.raises(ValueError, match="my_function"):
            _check_dims(empty_da, DIMS.time, "my_function")


# =============================================================================
# 6. Accessor Registration
# =============================================================================


class TestAccessorRegistration:
    """Verify that the ``.xmr`` namespace is correctly registered on xarray objects.

    xmris uses ``@xr.register_dataarray_accessor("xmr")`` and
    ``@xr.register_dataset_accessor("xmr")`` to attach processing methods.
    The plot sub-accessor uses lazy initialization to avoid import overhead.
    """

    def test_dataarray_accessor_exists(self, empty_da):
        """Every ``xr.DataArray`` must have the ``.xmr`` attribute after import."""
        assert hasattr(empty_da, "xmr")

    def test_dataset_accessor_exists(self):
        """Every ``xr.Dataset`` must have the ``.xmr`` attribute after import."""
        ds = xr.Dataset()
        assert hasattr(ds, "xmr")

    def test_plot_sub_accessor_exists(self, empty_da):
        """The ``.xmr.plot`` sub-accessor must be reachable."""
        assert hasattr(empty_da.xmr, "plot")

    def test_plot_sub_accessor_is_lazy(self, empty_da):
        """The plot sub-accessor must not be instantiated until first access.

        This keeps ``import xmris`` fast by deferring matplotlib imports
        until the user actually calls a plotting method.
        """
        accessor = empty_da.xmr
        assert accessor._plot is None
        _ = accessor.plot
        assert accessor._plot is not None

    def test_plot_sub_accessor_is_cached(self, empty_da):
        """Repeated ``.xmr.plot`` access must return the same cached instance.

        Without caching, each access would create a new ``XmrisPlotAccessor``
        object, wasting memory and breaking identity checks.
        """
        accessor = empty_da.xmr
        plot1 = accessor.plot
        plot2 = accessor.plot
        assert plot1 is plot2


# =============================================================================
# 7. Accessor Default Dimensions
# =============================================================================


class TestAccessorDefaults:
    """Guard against bare-string defaults drifting from the config constants.

    Every accessor method that takes a ``dim`` parameter should use a config
    constant (e.g., ``DIMS.time``) as its default value, not a bare string
    like ``"time"``. If the config value ever changes, bare-string defaults
    silently break.

    This test introspects method signatures via ``inspect.signature`` and
    compares actual defaults against the expected config values.
    """

    @pytest.mark.parametrize(
        "method_name, param_name, expected_default",
        [
            ("fft", "dim", DIMS.time),
            ("ifft", "dim", DIMS.frequency),
            ("fftc", "dim", DIMS.time),
            ("ifftc", "dim", DIMS.frequency),
            ("apodize_exp", "dim", DIMS.time),
            ("apodize_lg", "dim", DIMS.time),
            ("to_spectrum", "dim", DIMS.time),
            ("to_spectrum", "out_dim", DIMS.frequency),
            ("to_fid", "dim", DIMS.frequency),
            ("to_fid", "out_dim", DIMS.time),
            ("zero_fill", "dim", DIMS.time),
            ("autophase", "dim", DIMS.frequency),
            ("remove_digital_filter", "dim", DIMS.time),
            ("to_ppm", "dim", DIMS.frequency),
            ("to_hz", "dim", DIMS.chemical_shift),
            ("to_real_imag", "dim", DIMS.component),
            ("to_complex", "dim", DIMS.component),
        ],
    )
    def test_default_matches_config(self, method_name, param_name, expected_default):
        """Verify that the method's default for ``param_name`` equals the config constant.

        This test checks string equality, not object identity. It will still
        pass if the default is a bare string that happens to match today.
        The primary protection is that if a config value changes (e.g.,
        ``DIMS.time`` from ``"time"`` to ``"t"``), any method using a bare
        ``"time"`` default will fail this test.
        """
        import inspect

        from xmris.core.accessor import XmrisAccessor

        method = getattr(XmrisAccessor, method_name)
        sig = inspect.signature(method)
        actual_default = sig.parameters[param_name].default

        assert actual_default == expected_default, (
            f"XmrisAccessor.{method_name}(... {param_name}={actual_default!r} ...) "
            f"does not match the config value {expected_default!r}. "
            f"Use the config constant (e.g., DIMS.time) as the default."
        )


# =============================================================================
# 8. Attrs Preservation
# =============================================================================


class TestAttrsPreservation:
    """Verify that xmris processing methods never silently drop ``.attrs``.

    xarray's default behavior strips ``.attrs`` on most operations (arithmetic,
    ``where()``, ``concat()``, etc.). Since xmris's safety architecture depends
    on attrs surviving through processing chains, every method must explicitly
    preserve them.

    Each test runs a single processing method and verifies that all original
    attrs are present and unchanged in the output. The final test runs a
    multi-step chain to catch cumulative attr loss.
    """

    def _assert_attrs_preserved(self, original: xr.DataArray, result: xr.DataArray):
        """Assert that every attr from ``original`` exists unchanged in ``result``.

        Parameters
        ----------
        original : xr.DataArray
            The input DataArray before processing.
        result : xr.DataArray
            The output DataArray after processing.

        Raises
        ------
        AssertionError
            If any attr is missing or has a different value.
        """
        for key, value in original.attrs.items():
            assert key in result.attrs, (
                f"Attribute {key!r} was silently dropped during processing."
            )
            assert result.attrs[key] == value, (
                f"Attribute {key!r} was modified: {value!r} → {result.attrs[key]!r}"
            )

    def test_to_hz_preserves_attrs(self, valid_spectrum_da):
        """``to_hz`` must preserve all input attrs."""
        result = valid_spectrum_da.xmr.to_ppm()
        self._assert_attrs_preserved(valid_spectrum_da, result)

    def test_to_ppm_preserves_attrs(self, valid_spectrum_da):
        """``to_ppm`` must preserve all input attrs."""
        result = valid_spectrum_da.xmr.to_ppm()
        self._assert_attrs_preserved(valid_spectrum_da, result)

    def test_apodize_exp_preserves_attrs(self, valid_fid_da):
        """``apodize_exp`` must preserve all input attrs."""
        result = valid_fid_da.xmr.apodize_exp(lb=5.0)
        self._assert_attrs_preserved(valid_fid_da, result)

    def test_to_spectrum_preserves_attrs(self, valid_fid_da):
        """``to_spectrum`` must preserve all input attrs."""
        result = valid_fid_da.xmr.to_spectrum()
        self._assert_attrs_preserved(valid_fid_da, result)

    def test_phase_preserves_attrs(self, valid_spectrum_da):
        """``phase`` must preserve all input attrs (and may add ``p0``, ``p1``)."""
        result = valid_spectrum_da.xmr.phase(p0=10.0)
        self._assert_attrs_preserved(valid_spectrum_da, result)

    def test_zero_fill_preserves_attrs(self, valid_fid_da):
        """``zero_fill`` must preserve all input attrs."""
        result = valid_fid_da.xmr.zero_fill(target_points=4096)
        self._assert_attrs_preserved(valid_fid_da, result)

    def test_fft_preserves_attrs(self, valid_fid_da):
        """``fft`` must preserve all input attrs."""
        result = valid_fid_da.xmr.fft()
        self._assert_attrs_preserved(valid_fid_da, result)

    def test_multi_step_chain_preserves_attrs(self, valid_fid_da):
        """Attrs must survive a realistic multi-step processing chain."""
        result = valid_fid_da.xmr.apodize_exp(lb=5.0).xmr.to_spectrum().xmr.to_ppm()
        self._assert_attrs_preserved(valid_fid_da, result)


# =============================================================================
# 9. Integration: to_ppm end-to-end
# =============================================================================


class TestToPpm:
    """End-to-end integration tests for ``to_ppm``.

    This method exercises every architectural layer simultaneously:
    ``@requires_attrs`` for attr validation, ``_check_dims`` for dimension
    validation, config constants for coordinate naming, and the actual
    Hz-to-ppm math. It is the best single method for verifying that the
    architecture works as a whole.
    """

    def test_creates_chemical_shift_coord(self, valid_spectrum_da):
        """``to_ppm`` must add a new coordinate named ``COORDS.chemical_shift``."""
        result = valid_spectrum_da.xmr.to_ppm()
        assert COORDS.chemical_shift in result.coords

    def test_math_is_correct(self, valid_spectrum_da):
        """The ppm values must equal `carrier_ppm + (hz_coords / reference_frequency)`."""
        result = valid_spectrum_da.xmr.to_ppm()
        hz = valid_spectrum_da.coords[DIMS.frequency].values
        mhz = valid_spectrum_da.attrs[ATTRS.reference_frequency]
        carrier = valid_spectrum_da.attrs[ATTRS.carrier_ppm]

        expected_ppm = carrier + (hz / mhz)
        np.testing.assert_array_almost_equal(
            result.coords[COORDS.chemical_shift].values, expected_ppm
        )

    def test_preserves_original_frequency_coord(self, valid_spectrum_da):
        """``to_ppm`` adds a *new* coordinate — it must not destroy the original
        frequency coordinate, since users may need both Hz and ppm views.
        """  # noqa: D205
        result = valid_spectrum_da.xmr.to_ppm()
        assert DIMS.frequency in result.coords

    def test_fails_with_missing_attrs(self, empty_da):
        """Calling ``to_ppm`` without required attrs must trigger the bouncer."""
        with pytest.raises(ValueError, match="missing attributes"):
            empty_da.xmr.to_ppm()

    def test_fails_with_wrong_dim(self, valid_spectrum_da):
        """Passing a nonexistent dimension must trigger ``_check_dims``."""
        with pytest.raises(ValueError, match="missing dimension"):
            valid_spectrum_da.xmr.to_ppm(dim="nonexistent")

    def test_works_with_custom_dim_name(self):
        """Users with non-standard dim names must be able to pass them explicitly."""
        rng = np.random.default_rng()
        da = xr.DataArray(
            rng.standard_normal(100),
            dims=["freq"],
            coords={"freq": np.linspace(-1000, 1000, 100)},
            attrs={
                ATTRS.b0_field: 3.0,
                ATTRS.reference_frequency: 127.8,
                ATTRS.carrier_ppm: 4.7,
            },
        )
        result = da.xmr.to_ppm(dim="freq")
        assert COORDS.chemical_shift in result.coords

    def test_deleted_attr_fails(self, valid_spectrum_da):
        """Simulates a user who accidentally drops attrs mid-pipeline."""
        broken = valid_spectrum_da.copy()
        del broken.attrs[ATTRS.reference_frequency]
        with pytest.raises(ValueError, match="missing attributes"):
            broken.xmr.to_ppm()

    def test_multidim_input(self, multi_dim_da):
        """``to_ppm`` must work on N-dimensional data without flattening."""
        spectrum = multi_dim_da.xmr.to_spectrum()
        result = spectrum.xmr.to_ppm()
        assert COORDS.chemical_shift in result.coords
        assert "voxel" in result.dims
