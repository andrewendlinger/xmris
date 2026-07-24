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
  Update the module-level probe functions and add specific behavior checks to
  the decorator sections (`TestRequiresAttrs*`, `TestEnsuresDomain`,
  `TestComputesIn`, `TestDomainDimRule`).

* **Changing core mathematical logic:**
  Do not test complex scientific logic here. This file is for architecture.
  Test the math in a separate `test_processing.py` suite. The `TestToPpm`
  class here exists solely as a structural integration test of the pipeline.
"""

import copy
import logging
import pickle
import re
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
import xarray as xr

from xmris.core.accessor import _check_dims
from xmris.core.config import (
    ATTRS,
    COORDS,
    DIMS,
    SPECTRAL_DIMS,
    TIME_DIMS,
    VARS,
    BaseVocabulary,
    XmrisTerm,
)
from xmris.core.utils import _resolve_dim
from xmris.core.validation import computes_in, ensures_domain, requires_attrs
from xmris.fitting import build_prior_knowledge
from xmris.processing.fid import to_fid, to_spectrum

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _restore_xmris_options():
    """Snapshot and restore the global ``OPTIONS`` around every test.

    ``OPTIONS`` is a process-global with no other reset path. Without this, a
    leaked mutation — a permanent ``set_options(...)`` call, or a bug in it —
    would bleed into later tests and flake under ``-n auto``.
    """
    from xmris.core.options import OPTIONS

    snapshot = dict(OPTIONS)
    yield
    OPTIONS.clear()
    OPTIONS.update(snapshot)


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
        rng.standard_normal((n_voxels, n_time)) + 1j * rng.standard_normal((n_voxels, n_time)),
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
        assert term_val == term_val.lower(), f"COORDS.{prop_name} = {term_val!r} is not lowercase."

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
# 3b. Configuration: Term Hardening (immutability, uniqueness)
# =============================================================================


class TestXmrisTermHardening:
    """Vocabulary terms are frozen value objects whose metadata survives round-trips."""

    def test_mutation_raises(self):
        """Setting an attribute on a term must raise (frozen).

        Uses a local term, not the shared ATTRS singleton: were the freeze to
        regress, mutating the singleton here would corrupt it for every later test.
        """
        term = XmrisTerm("probe", description="d", unit="s")
        with pytest.raises(AttributeError, match="immutable"):
            term.unit = "Hz"

    def test_deletion_raises(self):
        """Deleting an attribute from a term must raise (frozen)."""
        term = XmrisTerm("probe", description="d", unit="s")
        with pytest.raises(AttributeError, match="immutable"):
            del term.unit

    @pytest.mark.parametrize("proto", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_roundtrip_preserves_metadata_and_freeze(self, proto):
        """Pickle round-trips keep value, metadata, and immutability on every protocol."""
        term = pickle.loads(pickle.dumps(ATTRS.reference_frequency, proto))
        assert term == ATTRS.reference_frequency
        assert term.description == ATTRS.reference_frequency.description
        assert term.unit == "MHz"
        with pytest.raises(AttributeError):
            term.unit = "Hz"

    def test_deepcopy_roundtrip(self):
        """``copy.deepcopy`` keeps value, metadata, and immutability."""
        term = copy.deepcopy(ATTRS.group_delay)
        assert term == ATTRS.group_delay
        assert term.unit == "samples"
        with pytest.raises(AttributeError):
            term.unit = "x"


class TestVocabularyUniqueness:
    """Duplicate canonical values inside one vocabulary fail at import time."""

    def test_duplicate_canonical_value_raises(self):
        """Two terms with the same canonical value must fail at class definition."""
        with pytest.raises(ValueError, match="duplicate vocabulary key 'twin'"):

            class _Broken(BaseVocabulary):
                a = XmrisTerm("twin", description="d")
                b = XmrisTerm("twin", description="d")

    def test_well_formed_vocabulary_constructs(self):
        """Distinct values construct without error."""

        class _Fine(BaseVocabulary):
            a = XmrisTerm("a", description="d")
            b = XmrisTerm("b", description="d")

        assert _Fine()._get_terms().keys() == {"a", "b"}


class TestParseInputDims:
    """Stack-dim auto-detect uses the singular vocabulary dims."""

    @staticmethod
    def _da(dims: tuple[str, ...]) -> xr.DataArray:
        return xr.DataArray(np.zeros((2,) * len(dims)), dims=list(dims))

    @pytest.mark.parametrize("stack", ["average", "repetition"])
    def test_detects_stack_dim(self, stack):
        """Auto-detect finds the acquisition axis (was AttributeError on DIMS.averages).

        ``coil`` is placed FIRST so it is ``remaining_dims[0]`` (the positional
        fallback). Only correct name-matching returns ``stack``; a broken match
        would fall back to ``coil`` and fail the assertion.
        """
        from xmris.visualization.plot._input_parsing import parse_input_dims_timeseries

        da = self._da((DIMS.chemical_shift, DIMS.coil, stack))
        x_dim, stack_dim = parse_input_dims_timeseries(da)
        assert x_dim == DIMS.chemical_shift
        assert stack_dim == stack

    def test_average_preferred_over_repetition(self):
        """Historical preference order (average before repetition) is preserved."""
        from xmris.visualization.plot._input_parsing import parse_input_dims_timeseries

        da = self._da((DIMS.chemical_shift, DIMS.repetition, DIMS.average))
        assert parse_input_dims_timeseries(da)[1] == DIMS.average

    def test_single_remaining_dim_is_stack(self):
        """With exactly one non-spectral dim, it becomes the stack axis."""
        from xmris.visualization.plot._input_parsing import parse_input_dims_timeseries

        da = self._da((DIMS.chemical_shift, DIMS.coil))
        assert parse_input_dims_timeseries(da) == (DIMS.chemical_shift, DIMS.coil)

    def test_frequency_used_as_x_axis(self):
        """When chemical_shift is absent, frequency is the x-axis."""
        from xmris.visualization.plot._input_parsing import parse_input_dims_timeseries

        da = self._da((DIMS.frequency, DIMS.coil))
        assert parse_input_dims_timeseries(da) == (DIMS.frequency, DIMS.coil)

    def test_one_dimensional_raises(self):
        """A 1-D spectral array has no stack axis and must raise."""
        from xmris.visualization.plot._input_parsing import parse_input_dims_timeseries

        da = self._da((DIMS.chemical_shift,))
        with pytest.raises(ValueError, match="at least two dimensions"):
            parse_input_dims_timeseries(da)

    def test_no_spectral_axis_raises(self):
        """Without chemical_shift or frequency, the x-axis cannot be resolved."""
        from xmris.visualization.plot._input_parsing import parse_input_dims_timeseries

        da = self._da((DIMS.coil, DIMS.echo))
        with pytest.raises(ValueError, match="resolve x-axis"):
            parse_input_dims_timeseries(da)


class TestBrukerVocabularyDims:
    """The Bruker loader emits singular vocabulary dim names (no plural magic strings)."""

    _PV_PARAMS = {
        "PVM_SpecMatrix": 8,
        "PVM_EncNReceivers": 2,
        "PVM_NAverages": 3,
        "PVM_NRepetitions": 4,
        "PVM_SpecSWH": 10000.0,
        "PVM_RepetitionTime": 1000.0,
        "PVM_FrqRef": 120.3,
        "PVM_FrqWorkPpm": 0.0,
        "groupDelay": 0.0,
    }

    def test_reshape_emits_singular_dims(self):
        """``reshape_bruker_raw`` labels axes with the singular vocabulary terms."""
        from xmris.vendor.bruker import reshape_bruker_raw

        raw = np.zeros(8 * 2 * 3 * 4, dtype=complex)
        _, dims = reshape_bruker_raw(raw, self._PV_PARAMS)
        assert dims == [DIMS.time, DIMS.coil, DIMS.average, DIMS.repetition]

    def test_build_fid_writes_tr_coordinate(self):
        """The TR coordinate is written for the canonical repetition dim."""
        from xmris.vendor.bruker import build_fid

        data = np.zeros((8, 4), dtype=complex)
        da = build_fid(data, [DIMS.time, DIMS.repetition], self._PV_PARAMS)
        assert DIMS.repetition in da.coords
        np.testing.assert_allclose(da.coords[DIMS.repetition].values, np.arange(1, 5) * 1.0)
        assert da.coords[DIMS.repetition].attrs["units"] == "s"


# =============================================================================
# 4. Decorator Engine: @requires_attrs
# =============================================================================


@requires_attrs(ATTRS.b0_field, ATTRS.reference_frequency)
def _needs_both(da: xr.DataArray):
    """Original docstring."""
    return da.attrs[ATTRS.b0_field]


@requires_attrs(ATTRS.b0_field)
def _needs_one(da: xr.DataArray):
    """Function requiring only a single attribute."""  # noqa: D401
    return da.attrs[ATTRS.b0_field]


@requires_attrs(ATTRS.b0_field)
def _no_docstring(da: xr.DataArray):  # noqa: D103
    pass


class TestRequiresAttrsRuntime:
    """Verify that ``@requires_attrs`` correctly validates at call time.

    The decorator wraps free functions whose first positional argument is the
    DataArray. It must:
    - Raise ``ValueError`` if any required attr is missing.
    - Include the missing key names and fix instructions in the error message.
    - Pass through to the wrapped function if all attrs are present.
    - Not alter the function's return value.
    """

    def test_all_missing(self, empty_da):
        """All required attrs absent — must raise with a descriptive message."""
        with pytest.raises(ValueError, match="missing attributes"):
            _needs_both(empty_da)

    def test_partial_missing(self, empty_da):
        """One attr present, one missing — must still raise.

        This catches a potential bug where the decorator short-circuits
        after finding the first present attr instead of checking all of them.
        """
        da = empty_da.assign_attrs({ATTRS.b0_field: 3.0})
        with pytest.raises(ValueError, match="missing attributes"):
            _needs_both(da)

    def test_all_present(self, valid_spectrum_da):
        """All required attrs present — must execute the function body normally."""
        result = _needs_both(valid_spectrum_da)
        assert result == 3.0

    def test_error_message_lists_missing_keys(self, empty_da):
        """The error message must name the specific missing attr keys."""
        with pytest.raises(ValueError, match=ATTRS.b0_field):
            _needs_both(empty_da)

    def test_error_message_includes_fix(self, empty_da):
        """The error message must include copy-pasteable ``assign_attrs`` fix code."""
        with pytest.raises(ValueError, match="assign_attrs"):
            _needs_both(empty_da)

    def test_returns_original_value(self, valid_spectrum_da):
        """The decorator must be transparent — return value is unchanged."""
        assert _needs_one(valid_spectrum_da) == valid_spectrum_da.attrs[ATTRS.b0_field]

    def test_works_as_keyword_argument(self, valid_spectrum_da):
        """The DataArray must also be accepted as a keyword argument."""
        assert _needs_one(da=valid_spectrum_da) == valid_spectrum_da.attrs[ATTRS.b0_field]


class TestRequiresAttrsDocstring:
    """Verify that ``@requires_attrs`` injects documentation at import time.

    The decorator appends a "Required Attributes" section to the wrapped
    function's docstring, pulling descriptions from the config vocabulary.
    This ensures docs and code can never drift apart.
    """

    def test_injects_section_header(self):
        """The auto-generated docstring must contain a 'Required Attributes' header."""
        assert "Required Attributes" in _needs_both.__doc__

    def test_injects_key_names(self):
        """Every required attr key must appear in the generated docstring."""
        doc = _needs_both.__doc__
        assert ATTRS.b0_field in doc
        assert ATTRS.reference_frequency in doc

    def test_preserves_original_docstring(self):
        """The original docstring text must not be overwritten by the injection."""
        assert "Original docstring." in _needs_both.__doc__

    def test_handles_no_docstring(self):
        """Decorating a function with ``None`` docstring must not crash.

        The decorator should gracefully create a new docstring containing
        only the 'Required Attributes' section.
        """
        doc = _no_docstring.__doc__
        assert doc is not None
        assert "Required Attributes" in doc

    def test_preserves_function_name(self):
        """``functools.wraps`` must preserve ``__name__`` for introspection and debugging."""  # noqa: E501
        assert _needs_both.__name__ == "_needs_both"


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
            ("remove_digital_filter", "dim", DIMS.time),
            ("estimate_group_delay", "dim", DIMS.time),
            ("to_ppm", "dim", DIMS.frequency),
            ("to_hz", "dim", DIMS.chemical_shift),
            ("to_real_imag", "dim", DIMS.component),
            ("to_complex", "dim", DIMS.component),
            ("fit_amares", "dim", DIMS.time),
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

    def test_autophase_dim_is_none_by_design(self):
        """`autophase` mirrors the domain-decorator rule: `dim` defaults to None.

        The free function carries ``@ensures_domain(SPECTRAL_DIMS)`` — a
        multi-label domain — whose merged resolution fills ``dim`` at call time
        by detecting ``frequency`` vs ``chemical_shift``. The accessor forwarder
        mirrors that signature. See ``TestDomainDimRule`` for the package-wide
        biconditional this instance follows.
        """  # noqa: D205
        import inspect

        from xmris.core.accessor import XmrisAccessor

        sig = inspect.signature(XmrisAccessor.autophase)
        assert sig.parameters["dim"].default is None


class TestDomainDimRule:
    """Enforce the package-wide `dim`-default rule (the "biconditional").

    ``dim`` defaults to ``None`` **iff** the function is domain-decorated with
    a *multi-label* domain (the decorator's merged resolution fills it at call
    time); every other ``dim`` defaults to a config constant. This replaces
    per-function carve-outs with one mechanically-enforced rule.
    """

    @staticmethod
    def _public_dim_functions():
        """Yield (qualname, function, dim_default) for the public processing API."""
        import inspect

        import xmris.fitting.amares
        import xmris.processing.baseline
        import xmris.processing.fid
        import xmris.processing.fourier
        import xmris.processing.phasing
        import xmris.processing.referencing
        import xmris.processing.utils
        import xmris.vendor.bruker

        modules = [
            xmris.processing.baseline,
            xmris.processing.fid,
            xmris.processing.fourier,
            xmris.processing.phasing,
            xmris.processing.referencing,
            xmris.processing.utils,
            xmris.fitting.amares,
            xmris.vendor.bruker,
        ]
        for mod in modules:
            for name, obj in inspect.getmembers(mod, inspect.isfunction):
                # Only functions *defined* in the module (functools.wraps keeps
                # __module__ pointing at the definition site), skip privates.
                if name.startswith("_") or obj.__module__ != mod.__name__:
                    continue
                sig = inspect.signature(obj)
                if "dim" not in sig.parameters:
                    continue
                yield f"{mod.__name__}.{name}", obj, sig.parameters["dim"].default

    def test_dim_defaults_follow_biconditional(self):
        """``dim=None`` ⟺ domain-decorated with a multi-label domain."""
        checked = 0
        for qualname, func, default in self._public_dim_functions():
            domain_info = getattr(func, "__xmris_domain__", None)
            multi_label = domain_info is not None and len(domain_info[0]) > 1
            if multi_label:
                assert default is None, (
                    f"{qualname} is domain-decorated with a multi-label domain "
                    f"but defaults dim={default!r}. Multi-label domain functions "
                    f"must default dim=None (the decorator resolves it)."
                )
            else:
                assert default is not None, (
                    f"{qualname} defaults dim=None but is not domain-decorated "
                    f"with a multi-label domain. Use the config constant "
                    f"(e.g., DIMS.time) as the default."
                )
            checked += 1
        # Sanity: the introspection must actually cover the processing API.
        assert checked >= 10, f"only {checked} functions found — introspection broke?"


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
            assert key in result.attrs, f"Attribute {key!r} was silently dropped during processing."
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


# =============================================================================
# 10. Domain Resolution: _resolve_dim
# =============================================================================


class TestResolveDim:
    """Verify ``_resolve_dim`` identifies the unique dimension of a domain group.

    The shared resolution helper behind the domain decorators' merged
    ``dim=None`` handling (and the visualization widgets' axis auto-detection):
    it must find the unique candidate dimension and raise actionable errors
    when resolution is impossible or ambiguous.
    """

    def test_finds_frequency(self, valid_spectrum_da):
        """A spectrum in Hz must resolve to ``DIMS.frequency``."""
        assert _resolve_dim(valid_spectrum_da, SPECTRAL_DIMS) == DIMS.frequency

    def test_finds_chemical_shift(self, valid_spectrum_da):
        """A spectrum in ppm must resolve to ``DIMS.chemical_shift``."""
        ppm = valid_spectrum_da.xmr.to_ppm()
        assert _resolve_dim(ppm, SPECTRAL_DIMS) == DIMS.chemical_shift

    def test_finds_time(self, valid_fid_da):
        """A FID must resolve ``TIME_DIMS`` to ``DIMS.time``."""
        assert _resolve_dim(valid_fid_da, TIME_DIMS) == DIMS.time

    def test_raises_when_no_candidate(self, valid_fid_da):
        """Time-domain input (no spectral dim) must raise with guidance."""
        with pytest.raises(ValueError, match="spectral dimension"):
            _resolve_dim(valid_fid_da, SPECTRAL_DIMS)

    def test_raises_when_ambiguous(self):
        """Multiple spectral dims present must raise and ask for an explicit dim."""
        da = xr.DataArray(
            np.zeros((4, 4)),
            dims=[DIMS.frequency, DIMS.chemical_shift],
            coords={DIMS.frequency: np.arange(4), DIMS.chemical_shift: np.arange(4)},
        )
        with pytest.raises(ValueError, match="[Aa]mbiguous"):
            _resolve_dim(da, SPECTRAL_DIMS)


# =============================================================================
# 11. Domain Decorators: @ensures_domain (funnel contract)
# =============================================================================


@ensures_domain(SPECTRAL_DIMS)
def _ensure_spectral_probe(da):
    """Test probe for ``@ensures_domain(SPECTRAL_DIMS)``: return the coerced da."""
    return da


@ensures_domain(TIME_DIMS)
def _ensure_time_probe(da):
    """Test probe for ``@ensures_domain(TIME_DIMS)``: return the coerced da."""
    return da


@ensures_domain(SPECTRAL_DIMS)
def _funnel_dim_probe(da, dim=None):
    """Funnel probe with a ``dim`` argument: return the dim the decorator settled on."""
    return dim


class TestEnsuresDomain:
    """Verify ``@ensures_domain`` implements the funnel contract.

    The funnel flavor must pass already-in-domain data through untouched
    (identity, no FFT), transform mismatched data into the target domain via
    the standard converters, leave the result there (no restore), resolve a
    ``dim=None`` argument after coercion, and preserve ``.attrs`` across the
    transform (the issue #21 coupling).
    """

    def test_noop_when_already_spectral(self, valid_spectrum_da):
        """A spectrum passed to a spectral-domain function is returned untouched."""
        result = _ensure_spectral_probe(valid_spectrum_da)
        assert result is valid_spectrum_da  # identity: no transform performed

    def test_transforms_fid_to_spectrum(self, valid_fid_da):
        """A FID is auto-FFT'd into the frequency domain and left there."""
        result = _ensure_spectral_probe(valid_fid_da)
        assert DIMS.frequency in result.dims
        assert DIMS.time not in result.dims

    def test_builds_coords_through_conversion(self, valid_fid_da):
        """The auto-FFT rebuilds physical coordinates on the new frequency dim."""
        result = _ensure_spectral_probe(valid_fid_da)
        assert DIMS.frequency in result.coords
        assert result.coords[DIMS.frequency].size == valid_fid_da.sizes[DIMS.time]

    def test_preserves_attrs_across_transform(self, valid_fid_da):
        """Auto-FFT must not silently drop ``.attrs`` (issue #21 guarantee)."""
        result = _ensure_spectral_probe(valid_fid_da)
        for key, value in valid_fid_da.attrs.items():
            assert result.attrs.get(key) == value

    def test_noop_when_already_time(self, valid_fid_da):
        """A FID passed to a time-domain function is returned untouched."""
        result = _ensure_time_probe(valid_fid_da)
        assert result is valid_fid_da

    def test_transforms_spectrum_to_fid(self, valid_spectrum_da):
        """A Hz spectrum is auto-IFFT'd into the time domain and left there."""
        result = _ensure_time_probe(valid_spectrum_da)
        assert DIMS.time in result.dims
        assert DIMS.frequency not in result.dims

    # --- merged dim resolution -------------------------------------------------

    def test_resolves_dim_on_spectrum(self, valid_spectrum_da):
        """``dim=None`` on an in-domain spectrum resolves to ``frequency``."""
        assert _funnel_dim_probe(valid_spectrum_da) == DIMS.frequency

    def test_resolves_dim_on_ppm(self, valid_spectrum_da):
        """``dim=None`` on a ppm spectrum resolves to ``chemical_shift``."""
        ppm = valid_spectrum_da.xmr.to_ppm()
        assert _funnel_dim_probe(ppm) == DIMS.chemical_shift

    def test_resolves_dim_after_coercion(self, valid_fid_da):
        """A FID is coerced first, then ``dim=None`` resolves on the *coerced* array.

        This is the ordering the old two-decorator stack made load-bearing;
        merged into one decorator it can no longer be applied wrongly.
        """
        assert _funnel_dim_probe(valid_fid_da) == DIMS.frequency

    def test_respects_explicit_dim(self, valid_spectrum_da):
        """An explicitly-passed ``dim`` must never be overridden."""
        assert _funnel_dim_probe(valid_spectrum_da, dim="custom") == "custom"

    # --- ppm leg & complexity gate --------------------------------------------

    def test_ppm_to_time_reconstructs_physical_dwell(self, valid_spectrum_da):
        """Coercing a ppm spectrum to time routes ppm→Hz→FID with a physical dwell time.

        Regression test for the latent dwell-time bug: computing the time axis
        from *ppm* coordinate spacing would be off by the reference frequency
        (a factor of ~1e8). The engine must reference back to Hz first.
        """
        ppm = valid_spectrum_da.xmr.to_ppm()
        result = _ensure_time_probe(ppm)

        assert DIMS.time in result.dims
        hz = valid_spectrum_da.coords[DIMS.frequency].values
        df = abs(hz[1] - hz[0])
        expected_dt = 1.0 / (len(hz) * df)
        actual_dt = float(result.coords[DIMS.time][1] - result.coords[DIMS.time][0])
        np.testing.assert_allclose(actual_dt, expected_dt, rtol=1e-9)

    def test_ppm_to_time_without_attrs_raises(self):
        """The ppm leg needs referencing attrs — missing ones raise the standard fix."""
        rng = np.random.default_rng()
        bare_ppm = xr.DataArray(
            rng.standard_normal(64) + 1j * rng.standard_normal(64),
            dims=[DIMS.chemical_shift],
            coords={DIMS.chemical_shift: np.linspace(0, 10, 64)},
        )
        with pytest.raises(ValueError, match="missing attributes"):
            _ensure_time_probe(bare_ppm)

    def test_real_spectrum_to_time_raises(self, valid_spectrum_da):
        """Real-valued spectral data cannot be transformed to time — loud error."""
        real_spec = valid_spectrum_da.copy(data=valid_spectrum_da.values.real)
        with pytest.raises(ValueError, match="real-valued"):
            _ensure_time_probe(real_spec)


# =============================================================================
# 12. Domain Decorators: @computes_in (domain-preserving contract)
# =============================================================================


@computes_in(TIME_DIMS)
def _time_scale_probe(da, dim=DIMS.time, factor=2.0):
    """Length-preserving time-domain probe: scale values, preserve metadata."""
    return da.copy(data=da.values * factor)


@computes_in(TIME_DIMS)
def _time_pad_probe(da, dim=DIMS.time):
    """Length-changing time-domain probe: zero-fill to double length."""
    from xmris.processing.fid import zero_fill

    return zero_fill(da, dim=dim, target_points=2 * da.sizes[dim])


class TestComputesIn:
    """Verify ``@computes_in`` implements the domain-preserving contract.

    The restore flavor must process in-domain input directly (no transform),
    round-trip out-of-domain input through the standard converters and hand it
    back in the input's representation with the original coordinates reassigned
    verbatim (length-preserving ops), recompute coordinates for length-changing
    ops, reject real-valued spectral input, and pass an explicitly-requested
    foreign dim through untouched.
    """

    def test_in_domain_input_is_not_transformed(self, valid_fid_da):
        """A FID handed to a time-domain op stays a FID — no round trip."""
        result = _time_scale_probe(valid_fid_da)
        assert list(result.dims) == list(valid_fid_da.dims)
        np.testing.assert_allclose(result.values, valid_fid_da.values * 2.0)
        np.testing.assert_array_equal(
            result.coords[DIMS.time].values, valid_fid_da.coords[DIMS.time].values
        )

    def test_spectrum_round_trips_to_spectrum(self, valid_spectrum_da):
        """A Hz spectrum comes back as a Hz spectrum (scaling commutes with the FFT)."""
        result = _time_scale_probe(valid_spectrum_da)
        assert DIMS.frequency in result.dims
        assert DIMS.time not in result.dims
        np.testing.assert_allclose(
            result.values, valid_spectrum_da.values * 2.0, rtol=1e-12, atol=1e-12
        )

    def test_round_trip_restores_coords_verbatim(self, valid_spectrum_da):
        """Length-preserving ops reassign the original coordinate exactly."""
        result = _time_scale_probe(valid_spectrum_da)
        np.testing.assert_array_equal(
            result.coords[DIMS.frequency].values,
            valid_spectrum_da.coords[DIMS.frequency].values,
        )

    def test_ppm_round_trips_to_ppm(self, valid_spectrum_da):
        """A ppm spectrum comes back as a ppm spectrum with its exact coordinates."""
        ppm = valid_spectrum_da.xmr.to_ppm()
        result = _time_scale_probe(ppm)
        assert DIMS.chemical_shift in result.dims
        assert DIMS.time not in result.dims
        np.testing.assert_array_equal(
            result.coords[DIMS.chemical_shift].values,
            ppm.coords[DIMS.chemical_shift].values,
        )
        np.testing.assert_allclose(result.values, ppm.values * 2.0, rtol=1e-12, atol=1e-12)

    def test_round_trip_preserves_attrs(self, valid_spectrum_da):
        """The full round trip must not drop ``.attrs`` (issue #21 guarantee)."""
        result = _time_scale_probe(valid_spectrum_da)
        for key, value in valid_spectrum_da.attrs.items():
            assert result.attrs.get(key) == value

    def test_length_change_recomputes_coords(self, valid_spectrum_da):
        """Length-changing ops keep converter-recomputed coordinates (finer grid)."""
        n = valid_spectrum_da.sizes[DIMS.frequency]
        result = _time_pad_probe(valid_spectrum_da)
        assert DIMS.frequency in result.dims
        assert result.sizes[DIMS.frequency] == 2 * n
        freqs = result.coords[DIMS.frequency].values
        assert np.all(np.diff(freqs) > 0)  # monotonic physical axis

    def test_real_spectrum_raises(self, valid_spectrum_da):
        """Real-valued spectral input has no valid FID behind it — loud error."""
        real_spec = valid_spectrum_da.copy(data=valid_spectrum_da.values.real)
        with pytest.raises(ValueError, match="real-valued"):
            _time_scale_probe(real_spec)

    def test_explicit_foreign_dim_passes_through(self):
        """An explicitly-requested non-domain dim disables coercion entirely."""
        rng = np.random.default_rng()
        kspace = xr.DataArray(
            rng.standard_normal(32) + 1j * rng.standard_normal(32),
            dims=["kx"],
            coords={"kx": np.arange(32)},
        )
        result = _time_scale_probe(kspace, dim="kx")
        assert list(result.dims) == ["kx"]
        np.testing.assert_allclose(result.values, kspace.values * 2.0)

    def test_decorator_stamps_introspection_metadata(self):
        """The wrapper must expose ``__xmris_domain__`` for the architecture tests."""
        assert _time_scale_probe.__xmris_domain__ == (TIME_DIMS, True)
        assert _ensure_spectral_probe.__xmris_domain__ == (SPECTRAL_DIMS, False)


# =============================================================================
# 13. Converter Round Trip (the guarantee the domain engine relies on)
# =============================================================================


class TestConverterRoundTrip:
    """``to_fid(to_spectrum(fid)) ≈ fid`` — unitary transforms, physical coords.

    The ``computes_in`` contract is only sound because the converter pair is
    numerically unitary (``norm="ortho"``) and reconstructs the physical time
    axis. Pin that guarantee down to tight tolerances.
    """

    def test_values_round_trip(self, valid_fid_da):
        """Data values must survive the round trip to ~1e-12."""
        rt = to_fid(to_spectrum(valid_fid_da))
        np.testing.assert_allclose(rt.values, valid_fid_da.values, rtol=0, atol=1e-12)

    def test_time_coords_round_trip(self, valid_fid_da):
        """The reconstructed time axis must match the original dwell spacing."""
        rt = to_fid(to_spectrum(valid_fid_da))
        np.testing.assert_allclose(
            rt.coords[DIMS.time].values,
            valid_fid_da.coords[DIMS.time].values,
            rtol=0,
            atol=1e-12,
        )


# =============================================================================
# 14. Integration: autophase domain-agnostic pilot
# =============================================================================


class TestAutophasePilot:
    """End-to-end: ``autophase`` wired to ``@ensures_domain`` (funnel + merged resolution).

    Proves the taxonomy works through the real accessor: a FID is auto-FFT'd,
    phased, and returned as a spectrum with metadata intact; a spectrum is
    phased in place; an explicit ``dim`` is honored.
    """

    def test_autophase_on_fid_returns_spectrum(self, valid_fid_da):
        """Calling ``autophase`` on a FID returns a phased spectrum (auto-FFT)."""
        result = valid_fid_da.xmr.autophase()
        assert DIMS.frequency in result.dims
        assert DIMS.time not in result.dims

    def test_autophase_on_fid_preserves_attrs(self, valid_fid_da):
        """The auto-FFT + phase chain preserves the original attrs."""
        result = valid_fid_da.xmr.autophase()
        for key, value in valid_fid_da.attrs.items():
            assert result.attrs.get(key) == value

    def test_autophase_on_spectrum_stays_spectral(self, valid_spectrum_da):
        """A spectrum input needs no conversion and stays in the frequency domain."""
        result = valid_spectrum_da.xmr.autophase()
        assert DIMS.frequency in result.dims

    def test_autophase_respects_explicit_dim(self, valid_spectrum_da):
        """An explicit ``dim`` is honored (resolution never overrides it)."""
        result = valid_spectrum_da.xmr.autophase(dim=DIMS.frequency)
        assert DIMS.frequency in result.dims

    def test_autophase_on_ppm_stays_ppm(self, valid_spectrum_da):
        """A ppm spectrum resolves to ``chemical_shift`` and stays in ppm."""
        ppm = valid_spectrum_da.xmr.to_ppm()
        result = ppm.xmr.autophase()
        assert DIMS.chemical_shift in result.dims
        assert DIMS.time not in result.dims


# =============================================================================
# 15. Domain Contract Rollout (which ops carry which contract)
# =============================================================================


class TestDomainRollout:
    """Pin every operation's domain contract (or deliberate absence of one).

    The op-class table from the domain-contracts design: funnel ops land in
    their home domain, domain-preserving physics ops restore the input
    representation, and converters/primitives stay undecorated with explicit
    domain handling. Fitting is domain-preserving too, but hand-rolls the round
    trip (it returns a Dataset, which the decorator's restore cannot handle) — see
    ``TestFittingDomain``.
    """

    def test_funnel_ops(self):
        """``autophase`` and ``baseline_als`` carry the funnel contract."""
        from xmris.processing.baseline import baseline_als
        from xmris.processing.phasing import autophase

        for func in (autophase, baseline_als):
            assert getattr(func, "__xmris_domain__", None) == (SPECTRAL_DIMS, False), func.__name__

    def test_domain_preserving_ops(self):
        """The apodizers and ``zero_fill`` carry the domain-preserving contract."""
        from xmris.processing.fid import apodize_exp, apodize_lg, zero_fill

        for func in (apodize_exp, apodize_lg, zero_fill):
            assert getattr(func, "__xmris_domain__", None) == (TIME_DIMS, True), func.__name__

    def test_undecorated_by_design(self):
        """Converters and primitives must NOT carry a domain decorator.

        Fitting is deliberately absent: it is domain-preserving but hand-rolls the
        round trip (Dataset return), so it carries no ``__xmris_domain__`` marker
        yet still auto-converts — its behavior is pinned in ``TestFittingDomain``.
        """
        from xmris.processing.fourier import fft, ifft
        from xmris.processing.phasing import phase
        from xmris.processing.referencing import to_hz, to_ppm
        from xmris.vendor.bruker import estimate_group_delay, remove_digital_filter

        undecorated = (
            to_spectrum,
            to_fid,
            to_ppm,
            to_hz,
            fft,
            ifft,
            phase,
            remove_digital_filter,
            estimate_group_delay,
        )
        for func in undecorated:
            assert not hasattr(func, "__xmris_domain__"), func.__name__

    def test_baseline_accessor_mirrors_dim_none(self):
        """The accessor forwarder mirrors ``baseline_als``'s ``dim=None`` signature."""
        import inspect

        from xmris.core.accessor import XmrisAccessor

        sig = inspect.signature(XmrisAccessor.baseline_als)
        assert sig.parameters["dim"].default is None

    # --- behavior smoke through the real accessor ------------------------------

    def test_spectrum_apodize_stays_spectrum(self, valid_spectrum_da):
        """Line-broadening a spectrum hands back a spectrum (round trip inside)."""
        result = valid_spectrum_da.xmr.apodize_exp(lb=5.0)
        assert DIMS.frequency in result.dims
        assert DIMS.time not in result.dims

    def test_fid_baseline_returns_real_spectrum(self, valid_fid_da):
        """Baseline on a FID funnels: lands as a real-valued spectrum."""
        result = valid_fid_da.xmr.baseline_als()
        assert DIMS.frequency in result.dims
        assert not np.iscomplexobj(result.values)

    def test_ppm_baseline_stays_ppm(self, valid_spectrum_da):
        """Baseline on a ppm spectrum resolves ``chemical_shift`` and stays ppm."""
        ppm = valid_spectrum_da.xmr.to_ppm()
        result = ppm.xmr.baseline_als()
        assert DIMS.chemical_shift in result.dims

    def test_baseline_then_apodize_raises(self, valid_spectrum_da):
        """One-way data downstream of baseline is caught by the complexity gate."""
        corrected = valid_spectrum_da.xmr.baseline_als()
        with pytest.raises(ValueError, match="real-valued"):
            corrected.xmr.apodize_exp(lb=2.0)

    def test_kspace_zero_fill_passes_through(self):
        """An explicit non-domain dim disables coercion — k-space stays k-space."""
        rng = np.random.default_rng()
        kspace = xr.DataArray(
            rng.standard_normal((8, 8)),
            dims=["kx", "ky"],
            coords={"kx": np.arange(8), "ky": np.arange(8)},
        )
        result = kspace.xmr.zero_fill(dim="kx", target_points=16, position="symmetric")
        assert result.sizes["kx"] == 16
        assert list(result.dims) == ["kx", "ky"]


class TestPriorKnowledgeBuilder:
    """Pin the in-memory prior-knowledge builder and the traps it defends against.

    The builder is dependency-light (no pyAMARES), so these run everywhere. It
    emits pyAMARES's positional CSV from a friendly spec while baking in the
    hard-won rules: always-explicit phase bounds, letters-only peak names, and
    anchor-first ordering for phase ties.
    """

    _SPEC = {
        "PCr": {"amplitude": 10, "chem_shift": 0.0, "linewidth": 15},
        "ATP": {
            "amplitude": 5,
            "chem_shift": -7.5,
            "linewidth": 20,
            "chem_shift_bounds": (-8.0, -7.0),
        },
    }

    def _sections(self, text):
        """Parse into (header, init_rows, bound_rows), handling quoted bound cells.

        Init and Bounds repeat the same five row labels, so a flat dict would
        collide them — this keeps the two sections apart.
        """
        import csv
        import io

        rows = list(csv.reader(io.StringIO(text)))
        header = rows[0][1:]
        init = {r[0]: r[1:] for r in rows[2:7]}  # after "Index" + "Initial Values"
        bounds = {r[0]: r[1:] for r in rows[8:13]}  # after the "Bounds" marker
        return header, init, bounds

    def test_builds_positional_layout(self):
        """The CSV has the exact section/row order pyAMARES reads positionally."""
        lines = build_prior_knowledge(self._SPEC).splitlines()
        labels = [ln.split(",")[0] for ln in lines]
        assert labels == [
            "Index",
            "Initial Values",
            "amplitude",
            "chemicalshift",
            "linewidth",
            "phase",
            "g",
            "Bounds",
            "amplitude",
            "chemicalshift",
            "linewidth",
            "phase",
            "g",
        ]
        assert lines[0] == "Index,PCr,ATP"

    def test_phase_bounds_always_explicit(self):
        """Every phase gets (-180, 180) bounds — a blank cell is pyAMARES's NaN trap."""
        text = build_prior_knowledge(self._SPEC)
        phase_bound_line = text.splitlines()[-2]  # phase is 4th of 5 bound rows
        assert "(-180.0, 180.0)" in phase_bound_line

    def test_amplitude_upper_bound_is_open(self):
        """Amplitude is non-negative with an open upper bound."""
        text = build_prior_knowledge(self._SPEC)
        amp_bounds = [ln for ln in text.splitlines() if ln.startswith("amplitude,")][1]
        assert "(0.0, " in amp_bounds and "(0.0, )" not in amp_bounds

    def test_chem_shift_default_window(self):
        """Absent explicit bounds, chem-shift is a symmetric window around the init."""
        text = build_prior_knowledge(self._SPEC, shift_window=0.5)
        cs_bounds = [ln for ln in text.splitlines() if ln.startswith("chemicalshift,")][1]
        assert "(-0.5, 0.5)" in cs_bounds  # PCr at 0.0 +/- 0.5
        assert "(-8.0, -7.0)" in cs_bounds  # ATP override respected

    def test_rejects_multiplet_digit_name(self):
        """A trailing digit would silently multiplet-sum in pyAMARES (BUG-008)."""
        with pytest.raises(ValueError, match="letters only"):
            build_prior_knowledge({"ATP2": {"amplitude": 1, "chem_shift": 0, "linewidth": 1}})

    def test_rejects_missing_required(self):
        """Every peak needs amplitude, chem_shift and linewidth."""
        with pytest.raises(ValueError, match="missing required"):
            build_prior_knowledge({"PCr": {"amplitude": 1, "chem_shift": 0}})

    def test_rejects_unknown_key(self):
        """Canonical-only: no aliases, unknown keys are refused (not silently dropped)."""
        with pytest.raises(ValueError, match="unknown key"):
            build_prior_knowledge(
                {"PCr": {"amplitude": 1, "chem_shift": 0, "linewidth": 1, "cs": 9}}
            )

    def test_rejects_empty(self):
        """An empty spec is a user error, caught early."""
        with pytest.raises(ValueError, match="empty"):
            build_prior_knowledge({})

    def test_rejects_inverted_bounds(self):
        """A (lower > upper) bound is refused before it reaches the solver."""
        with pytest.raises(ValueError, match="lower > upper"):
            build_prior_knowledge(
                {
                    "PCr": {
                        "amplitude": 1,
                        "chem_shift": 0,
                        "linewidth": 1,
                        "linewidth_bounds": (30, 5),
                    }
                }
            )

    def test_optional_phase_and_g_default_zero(self):
        """Phase and g are optional and default to 0."""
        _, init, _ = self._sections(build_prior_knowledge(self._SPEC))
        assert init["phase"] == ["0.0", "0.0"]
        assert init["g"] == ["0.0", "0.0"]

    def test_tie_phase_orders_anchor_first(self):
        """A phase tie moves the anchor to the first column and ties the rest to it."""
        header, init, _ = self._sections(build_prior_knowledge(self._SPEC, tie_phase_to="ATP"))
        assert header == ["ATP", "PCr"]  # anchor first
        assert init["phase"] == ["0.0", "ATP"]  # anchor's value, then the tie expr

    def test_tie_phase_unknown_anchor_raises(self):
        """A tie anchor that names no peak is a clear error, not a silent no-op."""
        with pytest.raises(ValueError, match="not one of the peaks"):
            build_prior_knowledge(self._SPEC, tie_phase_to="Xx")

    def test_none_bound_endpoint_opens_side(self):
        """A None endpoint in `<name>_bounds` opens that side — same as the other params."""
        text = build_prior_knowledge(
            {
                "PCr": {
                    "amplitude": 1,
                    "chem_shift": 0,
                    "linewidth": 1,
                    "chem_shift_bounds": (None, 1.0),
                }
            }
        )
        cs_bounds = [ln for ln in text.splitlines() if ln.startswith("chemicalshift,")][1]
        assert ", 1.0)" in cs_bounds and "None" not in cs_bounds

    def test_explicit_none_init_defaults(self):
        """An explicit None phase/g is treated like an absent key (0.0), not a TypeError."""
        _, init, _ = self._sections(
            build_prior_knowledge(
                {"PCr": {"amplitude": 1, "chem_shift": 0, "linewidth": 1, "phase": None, "g": None}}
            )
        )
        assert init["phase"] == ["0.0"]
        assert init["g"] == ["0.0"]


class TestSimulateFid:
    """Pin `simulate_fid`'s time axis to exactly `n_points` samples.

    Like the builder above, these are pyAMARES-free and run everywhere. The axis used
    to be `np.arange(0, dwelltime * n_points, dwelltime)`, whose length is float-rounding
    dependent — and because the signal and its coordinate were built from that same
    expression, an off-by-one agreed with itself and never raised. Callers that concat,
    add or slice two simulated FIDs are the ones that would have paid for it.
    """

    # (spectral_width, n_points) pairs whose float-step arange returned n + 1 samples.
    _HOSTILE = [(3001.2, 60), (2999.7, 1000), (1234.5, 255), (3333.3, 1000)]
    _BENIGN = [(8000.0, 512), (10000.0, 1024)]

    @staticmethod
    def _fid(sw, n, **kwargs):
        from xmris.fitting.simulation import simulate_fid

        return simulate_fid(
            amplitudes=[1.0], frequencies=[10.0], spectral_width=sw, n_points=n, **kwargs
        )

    @pytest.mark.parametrize(("sw", "n"), _HOSTILE + _BENIGN)
    def test_length_exact(self, sw, n):
        """`n_points=n` returns n samples — data and coordinate alike."""
        da = self._fid(sw, n)
        assert da.sizes[DIMS.time] == n
        # Pinned separately: the two were built by independent expressions, so a future
        # divergence between signal and coordinate would otherwise pass unnoticed.
        assert da.coords[COORDS.time].size == n

    @pytest.mark.parametrize(("sw", "n"), _HOSTILE)
    def test_axis_step_and_offset(self, sw, n):
        """Fixing the length leaves the sampling instants themselves untouched."""
        dead_time = 7.5e-5
        t = self._fid(sw, n, dead_time=dead_time).coords[COORDS.time].values
        np.testing.assert_allclose(t[0], dead_time)
        np.testing.assert_allclose(np.diff(t), 1.0 / sw)

    @pytest.mark.parametrize(("sw", "n"), _HOSTILE)
    def test_noise_matches_length(self, sw, n):
        """The noisy branch is length-exact too — it shapes noise from the signal."""
        da = self._fid(sw, n, target_snr=50.0, seed=0)
        assert da.sizes[DIMS.time] == n == da.coords[COORDS.time].size


# =============================================================================
# 16. Runtime Options: strict mode (auto_convert=False)
# =============================================================================


class TestFittingDomain:
    """Pin `fit_amares`'s domain-preserving behavior (the 2026 pivot).

    Fitting runs in the time domain, but a spectrum is accepted and the
    time-domain outputs (`data`/`fit`/`residuals`) are returned in the input's
    representation (ppm in -> ppm out). Uses tiny 1-D synthetic data and
    ``num_workers=1`` to stay fast. Requires the optional pyAMARES package.
    """

    _MHZ = 120.0
    _SW = 10000.0

    @pytest.fixture(autouse=True)
    def _require_pyamares(self):
        pytest.importorskip("pyAMARES")

    @pytest.fixture
    def pk_path(self, tmp_path):
        """A minimal 2-peak (PCr/ATP) pyAMARES prior-knowledge CSV."""
        content = (
            "Index,PCr,ATP\n"
            "Initial Values,,\n"
            "amplitude,10.0,5.0\n"
            "chemicalshift,0.0,-7.5\n"
            "linewidth,15.0,20.0\n"
            "phase,0,0\n"
            "g,0,0\n"
            "Bounds,,\n"
            'amplitude,"(0, ","(0, "\n'
            'chemicalshift,"(-0.5, 0.5)","(-8.0, -7.0)"\n'
            'linewidth,"(5.0, 30.0)","(10.0, 40.0)"\n'
            'phase,"(-180, 180)","(-180, 180)"\n'
            'g,"(0, 1)","(0, 1)"\n'
        )
        path = tmp_path / "pk.csv"
        path.write_text(content)
        return path

    @pytest.fixture
    def fid(self):
        """A clean 1-D 2-peak FID carrying reference_frequency + carrier_ppm."""
        n = 512
        dt = 1.0 / self._SW
        t = np.arange(n) * dt
        rng = np.random.default_rng(0)
        sig = 10.0 * np.exp(-15.0 * np.pi * t) * np.exp(2j * np.pi * 0.0 * self._MHZ * t)
        sig += 5.0 * np.exp(-20.0 * np.pi * t) * np.exp(2j * np.pi * -7.5 * self._MHZ * t)
        sig = sig + rng.normal(0, 0.2, n) + 1j * rng.normal(0, 0.2, n)
        return xr.DataArray(
            sig,
            dims=[DIMS.time],
            coords={DIMS.time: t},
            attrs={str(ATTRS.reference_frequency): self._MHZ, str(ATTRS.carrier_ppm): 0.0},
        )

    def _fit(self, da, pk_path):
        return da.xmr.fit_amares(prior_knowledge=pk_path, method="least_squares", num_workers=1)

    def test_fid_in_returns_time_domain(self, fid, pk_path):
        """A FID fits directly; outputs stay time-domain and carry the vocab."""
        ds = self._fit(fid, pk_path)
        assert set(ds.data_vars) >= {VARS.original_data, VARS.fit, VARS.residuals}
        assert DIMS.time in ds[VARS.fit].dims
        assert DIMS.metabolite in ds.dims
        assert np.all(np.isfinite(ds[VARS.amplitude].values))
        assert ATTRS.amares_amplitude_scale in ds.attrs

    def test_spectrum_in_returns_frequency(self, fid, pk_path):
        """A Hz spectrum funnels to a FID for the fit and comes back as a spectrum."""
        ds = self._fit(fid.xmr.to_spectrum(), pk_path)
        assert DIMS.frequency in ds[VARS.fit].dims
        assert DIMS.frequency in ds[VARS.original_data].dims
        assert DIMS.time not in ds[VARS.fit].dims

    def test_ppm_in_returns_ppm(self, fid, pk_path):
        """A ppm spectrum is restored to ppm (the whole round trip runs)."""
        ppm = fid.xmr.to_spectrum().xmr.to_ppm()
        ds = self._fit(ppm, pk_path)
        assert DIMS.chemical_shift in ds[VARS.fit].dims

    def test_fid_and_spectrum_agree(self, fid, pk_path):
        """Fitting a FID vs its spectrum yields the same parameters."""
        a = self._fit(fid, pk_path)
        b = self._fit(fid.xmr.to_spectrum(), pk_path)
        np.testing.assert_allclose(a[VARS.amplitude].values, b[VARS.amplitude].values, rtol=1e-3)

    def test_real_spectrum_refused(self, fid, pk_path):
        """A real-valued spectrum has no FID behind it — refused, not fitted."""
        real_spec = fid.xmr.to_spectrum().real
        with pytest.raises(ValueError, match="real-valued"):
            real_spec.xmr.fit_amares(prior_knowledge=pk_path, num_workers=1)

    def test_strict_mode_refuses_with_recipe(self, fid, pk_path):
        """Under auto_convert=False a spectrum fit raises the explicit to_fid recipe."""
        from xmris import set_options

        spec = fid.xmr.to_spectrum()
        with set_options(auto_convert=False):
            with pytest.raises(ValueError, match="to_fid"):
                spec.xmr.fit_amares(prior_knowledge=pk_path, num_workers=1)

    def test_scale_trap_defeated(self, fid, pk_path):
        """A Bruker-scale FID (x1e7) converges and rescales — it doesn't echo the prior."""
        base = self._fit(fid, pk_path)[VARS.amplitude].values
        big = (fid * 1e7).assign_attrs(fid.attrs)  # arithmetic drops attrs; real FIDs keep them
        ds_big = self._fit(big, pk_path)
        np.testing.assert_allclose(ds_big[VARS.amplitude].values, base * 1e7, rtol=0.05)
        assert ds_big.attrs[ATTRS.amares_amplitude_scale] > 1e6

    def test_zero_signal_is_nan(self, fid, pk_path):
        """A zero-signal spectrum yields NaN, distinguishable from a real fit."""
        stack = xr.concat([fid, xr.zeros_like(fid)], dim="voxel").assign_attrs(fid.attrs)
        ds = self._fit(stack, pk_path)
        assert np.all(np.isnan(ds[VARS.amplitude].isel(voxel=1).values))
        assert np.all(np.isfinite(ds[VARS.amplitude].isel(voxel=0).values))

    def test_parallel_path_matches_serial(self, fid, pk_path):
        """The loky parallel branch (num_workers>1) matches the serial fit exactly.

        Every other fitting test forces ``num_workers=1``, so this is the only test
        that drives ``_run_parallel_fitting_optimal`` (the joblib/loky pool) — it
        guards both the parallel dispatch and the generator's result-to-index
        ordering. The two voxels are made *distinct* (v1 at half scale) so a
        mis-ordered parallel assembly would diverge from the serial result. Kept in
        the arch suite under ``-n0``: loky inside nbmake's ``-n auto`` xdist would
        nest parallel pools.
        """
        stack = xr.concat([fid, 0.5 * fid], dim="voxel").assign_attrs(fid.attrs)
        serial = stack.xmr.fit_amares(
            prior_knowledge=pk_path, method="least_squares", num_workers=1
        )
        parallel = stack.xmr.fit_amares(
            prior_knowledge=pk_path, method="least_squares", num_workers=2
        )
        # Same shape and variable set whichever engine ran.
        assert dict(parallel.sizes) == dict(serial.sizes)
        assert set(parallel.data_vars) == set(serial.data_vars)
        # Same fit, voxel for voxel: leastsq is deterministic, only the dispatch differs.
        np.testing.assert_allclose(
            parallel[VARS.amplitude].values,
            serial[VARS.amplitude].values,
            rtol=1e-6,
            equal_nan=True,
        )
        # Ordering sanity: v0 carries twice v1's amplitude, so a swap would fail here.
        amps = serial[VARS.amplitude]
        np.testing.assert_allclose(
            amps.isel(voxel=0).values, 2.0 * amps.isel(voxel=1).values, rtol=0.05
        )

    # --- in-memory prior knowledge (workstream C) ---

    # A dict spec numerically equivalent to the `pk_path` hand-written fixture.
    _DICT_PK = {
        "PCr": {
            "amplitude": 10.0,
            "chem_shift": 0.0,
            "linewidth": 15.0,
            "chem_shift_bounds": (-0.5, 0.5),
            "linewidth_bounds": (5.0, 30.0),
        },
        "ATP": {
            "amplitude": 5.0,
            "chem_shift": -7.5,
            "linewidth": 20.0,
            "chem_shift_bounds": (-8.0, -7.0),
            "linewidth_bounds": (10.0, 40.0),
        },
    }

    def test_dict_prior_knowledge_fits(self, fid):
        """A fit runs straight from an in-memory dict — no CSV the user must write."""
        ds = fid.xmr.fit_amares(
            prior_knowledge=self._DICT_PK, method="least_squares", num_workers=1
        )
        assert list(ds[DIMS.metabolite].values) == ["PCr", "ATP"]
        amps = ds[VARS.amplitude].values
        assert np.all(np.isfinite(amps))
        np.testing.assert_allclose(amps[0] / amps[1], 2.0, rtol=0.05)  # true 10:5

    def test_dict_and_path_agree(self, fid, pk_path):
        """The builder reproduces the hand-written fixture: identical fits."""
        by_dict = fid.xmr.fit_amares(
            prior_knowledge=self._DICT_PK, method="least_squares", num_workers=1
        )
        by_path = self._fit(fid, pk_path)
        np.testing.assert_allclose(
            by_dict[VARS.amplitude].values, by_path[VARS.amplitude].values, rtol=1e-6
        )

    def test_missing_path_raises(self, fid):
        """A nonexistent prior-knowledge path fails clearly, not deep inside pyAMARES."""
        with pytest.raises(FileNotFoundError, match="not found"):
            fid.xmr.fit_amares(prior_knowledge="/no/such/pk.csv", num_workers=1)

    # --- per-parameter uncertainties: Shape B (workstream D) ---

    def test_uncertainties_span_parameter_dim(self, fid, pk_path):
        """crlb/sd carry a per-parameter axis; values stay named data vars."""
        ds = self._fit(fid, pk_path)
        assert ds[VARS.crlb].dims == (DIMS.metabolite, DIMS.parameter)
        assert ds[VARS.sd].dims == (DIMS.metabolite, DIMS.parameter)
        assert list(ds[DIMS.parameter].values) == [
            VARS.amplitude,
            VARS.chem_shift,
            VARS.linewidth,
            VARS.phase,
        ]
        assert ds[VARS.amplitude].dims == (DIMS.metabolite,)  # value stays named

    def test_amplitude_sd_scales_crlb_invariant(self, fid, pk_path):
        """Amplitude sd tracks the signal scale (absolute); CRLB% (relative) does not."""
        base = self._fit(fid, pk_path)
        scaled = self._fit((fid * 100.0).assign_attrs(fid.attrs), pk_path)
        amp = dict(parameter=VARS.amplitude)
        ratio = (scaled[VARS.sd].sel(**amp) / base[VARS.sd].sel(**amp)).values
        np.testing.assert_allclose(ratio, 100.0, rtol=1e-3)
        np.testing.assert_allclose(
            scaled[VARS.crlb].sel(**amp).values,
            base[VARS.crlb].sel(**amp).values,
            rtol=1e-3,
        )

    def test_g_global_forwarded(self, fid, pk_path):
        """g_global reaches pyAMARES: a Gaussian lineshape changes the fit."""
        lor = fid.xmr.fit_amares(prior_knowledge=pk_path, num_workers=1, g_global=0.0)
        gau = fid.xmr.fit_amares(prior_knowledge=pk_path, num_workers=1, g_global=1.0)
        assert not np.allclose(lor[VARS.amplitude].values, gau[VARS.amplitude].values, rtol=1e-3)
        # `False` (fit each peak's g) is accepted and yields a valid fit.
        free = fid.xmr.fit_amares(prior_knowledge=pk_path, num_workers=1, g_global=False)
        assert np.all(np.isfinite(free[VARS.amplitude].values))

    def test_carrier_enables_absolute_ppm(self, pk_path):
        """A nonzero carrier_ppm lets prior knowledge use absolute/literature ppm."""
        from xmris.fitting.simulation import simulate_fid

        # Carrier at 2.0 ppm; peaks given at ABSOLUTE 0.0 and -7.5 ppm. dampings are
        # linewidth * pi, so 15/20 Hz linewidths match the pk_path fixture bounds.
        fid = simulate_fid(
            amplitudes=[10.0, 5.0],
            chemical_shifts=[0.0, -7.5],
            reference_frequency=self._MHZ,
            carrier_ppm=2.0,
            spectral_width=self._SW,
            n_points=1024,
            dampings=[15.0 * np.pi, 20.0 * np.pi],
            target_snr=200,
            seed=0,
        )
        # carrier auto-read from carrier_ppm=2.0 -> absolute shifts recovered.
        ds = fid.xmr.fit_amares(prior_knowledge=pk_path, num_workers=1)
        np.testing.assert_allclose(ds[VARS.chem_shift].values, [0.0, -7.5], atol=0.1)
        np.testing.assert_allclose(ds[VARS.amplitude].values, [10.0, 5.0], rtol=0.05)

        # carrier=0 override reads carrier-relative -> the abs-ppm peaks miss (pinned).
        rel = fid.xmr.fit_amares(prior_knowledge=pk_path, num_workers=1, carrier=0.0)
        assert not np.allclose(rel[VARS.chem_shift].values, [0.0, -7.5], atol=0.1)

    def test_no_domain_marker(self):
        """Fitting hand-rolls the contract, so it carries no decorator marker."""
        from xmris.fitting.amares import fit_amares

        assert not hasattr(fit_amares, "__xmris_domain__")

    # --- hardening fixes (2026-07 review) ---

    def test_fractional_dwelltime_reconstruction(self, pk_path):
        """A (sw, n) whose `dwelltime * n` rounds up reconstructs length-exactly.

        `np.arange(0, dwelltime * n, dwelltime)` yields n+1 samples here (sw=3001.2,
        n=60), which pre-fix broadcast-crashed the model assignment.
        """
        sw, n = 3001.2, 60
        t = np.arange(n) / sw
        sig = 10.0 * np.exp(-15.0 * np.pi * t) * np.exp(2j * np.pi * 0.0 * self._MHZ * t)
        sig += 5.0 * np.exp(-20.0 * np.pi * t) * np.exp(2j * np.pi * -7.5 * self._MHZ * t)
        fid = xr.DataArray(
            sig,
            dims=[DIMS.time],
            coords={DIMS.time: t},
            attrs={str(ATTRS.reference_frequency): self._MHZ, str(ATTRS.carrier_ppm): 0.0},
        )
        ds = fid.xmr.fit_amares(prior_knowledge=pk_path, method="least_squares", num_workers=1)
        assert ds[VARS.fit].sizes[DIMS.time] == n  # length-exact; pre-fix raised ValueError

    def test_fit_keeps_referencing_attrs(self, fid, pk_path):
        """fit/residuals keep `reference_frequency`, so they convert like `data` (ppm-in)."""
        ds = self._fit(fid.xmr.to_spectrum().xmr.to_ppm(), pk_path)
        for var in (VARS.fit, VARS.residuals):
            assert ATTRS.reference_frequency in ds[var].attrs
        ds[VARS.fit].xmr.to_hz()  # ppm -> Hz needs the calibration; must not raise

    def test_stack_dim_name_collision(self, fid, pk_path):
        """An input dim literally named 'spectrum' fits without a stack-name collision."""
        stack = xr.concat([fid, fid], dim="spectrum").assign_attrs(fid.attrs)
        ds = stack.xmr.fit_amares(prior_knowledge=pk_path, method="least_squares", num_workers=1)
        assert ds[VARS.amplitude].sizes["spectrum"] == 2

    def test_coord_long_names(self, fid, pk_path):
        """The new metabolite/parameter coords carry their vocab long_name (Commandment 7)."""
        ds = self._fit(fid, pk_path)
        assert ds[DIMS.metabolite].attrs.get("long_name") == "Metabolite"
        assert ds[DIMS.parameter].attrs.get("long_name") == "Parameter"

    def test_dataframe_prior_knowledge(self, fid, pk_path):
        """A labels-as-index DataFrame fits; a RangeIndex frame is refused clearly."""
        import io

        import pandas as pd

        csv = build_prior_knowledge(self._DICT_PK)
        df_index = pd.read_csv(io.StringIO(csv), index_col=0)  # canonical: labels as index
        ds = fid.xmr.fit_amares(prior_knowledge=df_index, method="least_squares", num_workers=1)
        assert np.all(np.isfinite(ds[VARS.amplitude].values))
        df_range = pd.read_csv(io.StringIO(csv))  # RangeIndex — labels in a column
        with pytest.raises(ValueError, match="row labels as its index"):
            fid.xmr.fit_amares(prior_knowledge=df_range, num_workers=1)

    def test_all_nan_signal_no_crash(self, fid, pk_path):
        """An all-NaN (fully masked) input degrades to NaN, not an `nanargmax` crash."""
        nan_fid = xr.full_like(fid, np.nan)
        stack = xr.concat([nan_fid, nan_fid], dim="voxel").assign_attrs(fid.attrs)
        ds = stack.xmr.fit_amares(prior_knowledge=pk_path, num_workers=1)  # must not raise
        assert np.all(np.isnan(ds[VARS.amplitude].values))

    # --- empty voxels are never dispatched (PR #105 follow-up, item 2) ---

    @pytest.fixture
    def gapped_stack(self, fid):
        """Three voxels with an *empty* one in the middle, the live two distinguishable.

        The empty voxel sits at index 1 (not the end) and v0 carries twice v2's
        amplitude, so a result mis-mapped by the skip would show up as a swap.
        """
        return xr.concat([fid, xr.zeros_like(fid), 0.5 * fid], dim="voxel").assign_attrs(fid.attrs)

    def test_empty_voxel_not_dispatched(self, gapped_stack, pk_path, monkeypatch):
        """An all-zero voxel never reaches the optimizer — it is skipped up front.

        Fitting it was pure waste: the result is discarded by the `no_signal` flag
        either way, and the degenerate covariance leaks an lmfit RuntimeWarning.
        """
        from xmris.fitting import amares as amares_mod

        real = amares_mod._fit_dataset_safe
        dispatched = []

        def _counting(fid_current, *args, **kwargs):
            dispatched.append(float(np.abs(fid_current).max()))
            return real(fid_current, *args, **kwargs)

        monkeypatch.setattr(amares_mod, "_fit_dataset_safe", _counting)
        ds = gapped_stack.xmr.fit_amares(
            prior_knowledge=pk_path, method="least_squares", num_workers=1
        )
        assert len(dispatched) == 2, f"the empty voxel was fitted anyway: {dispatched}"
        assert all(m > 0 for m in dispatched)
        # The output is unchanged by the skip: still no_signal, still NaN.
        np.testing.assert_array_equal(ds[VARS.fit_status].values, [0, 1, 0])
        assert np.all(np.isnan(ds[VARS.amplitude].isel(voxel=1).values))
        assert np.all(np.isfinite(ds[VARS.amplitude].isel(voxel=[0, 2]).values))

    def test_parallel_empty_voxel_keeps_order(self, gapped_stack, pk_path):
        """Skipping empty voxels must not shift the parallel results off their index.

        The pool is now handed only the *active* rows, so the write-back is what
        maps results home. A mis-scatter would put voxel 2's fit at index 1. Same
        loky caveat as ``test_parallel_path_matches_serial``.
        """
        kw = dict(prior_knowledge=pk_path, method="least_squares")
        serial = gapped_stack.xmr.fit_amares(num_workers=1, **kw)
        parallel = gapped_stack.xmr.fit_amares(num_workers=2, **kw)
        np.testing.assert_array_equal(parallel[VARS.fit_status].values, [0, 1, 0])
        np.testing.assert_allclose(
            parallel[VARS.amplitude].values,
            serial[VARS.amplitude].values,
            rtol=1e-6,
            equal_nan=True,
        )
        # Ordering sanity: v0 carries twice v2's amplitude across the gap at v1.
        amps = parallel[VARS.amplitude]
        np.testing.assert_allclose(
            amps.isel(voxel=0).values, 2.0 * amps.isel(voxel=2).values, rtol=0.05
        )

    def test_empty_voxel_leaks_no_stderr(self):
        """A grid with an empty voxel fits without leaking a warning onto stderr.

        This is the user-visible half: lmfit's degenerate-covariance
        ``RuntimeWarning`` used to escape with an absolute ``.venv`` path, which
        then rendered into notebook output on the docs site. It runs in a
        subprocess because that leak is a file-descriptor write from a joblib
        worker — nothing in-process observes it.
        """
        script = """
            import numpy as np
            import xarray as xr
            import xmris

            fid = xmris.simulate_fid(
                amplitudes=[10.0, 5.0], chemical_shifts=[0.0, -7.5],
                reference_frequency=120.0, spectral_width=10000.0, n_points=256,
                dampings=[15.0 * np.pi, 20.0 * np.pi], target_snr=200.0, seed=0,
            )
            grid = xr.concat(
                [fid, xr.zeros_like(fid), 0.5 * fid], dim="voxel"
            ).assign_attrs(fid.attrs)
            pk = {
                "PCr": {"amplitude": 10.0, "chem_shift": 0.0, "linewidth": 15.0},
                "ATP": {"amplitude": 5.0, "chem_shift": -7.5, "linewidth": 20.0},
            }
            ds = grid.xmr.fit_amares(pk, method="least_squares", num_workers=4)
            assert ds["fit_status"].values.tolist() == [0, 1, 0]
            print("OK")
        """
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("OK"), result.stdout
        noisy = [ln for ln in result.stderr.splitlines() if "Warning" in ln]
        assert not noisy, f"the empty voxel leaked onto stderr: {noisy}"


class TestFittingVerbosity:
    """Pin BUG-010: `verbose=False` silences pyAMARES, and it holds in workers.

    pyAMARES emits on four channels (stdout prints, tqdm, warnings, its own
    loggers). Verbosity is (re)applied per fit call — inside `_fit_dataset_safe`,
    which runs in every joblib worker — so silence holds at any `num_workers`,
    not only in-process.
    """

    @pytest.fixture(autouse=True)
    def _require_pyamares(self):
        pytest.importorskip("pyAMARES")

    @pytest.fixture(autouse=True)
    def _restore_quiet(self):
        """Leave the loggers quiet after each test regardless of what it set."""
        yield
        from xmris.fitting.amares import _set_verbosity

        _set_verbosity(False)

    def test_set_verbosity_levels(self):
        """Log levels track `verbose` — including pyAMARES's lazily-made loggers."""
        from pyAMARES.libs import logger as pa_logger

        from xmris.fitting.amares import _set_verbosity, logger

        _set_verbosity(False)
        assert logger.level == logging.ERROR
        assert pa_logger.DEFAULT_LOG_LEVEL == "error"

        _set_verbosity(True)
        assert logger.level == logging.INFO
        assert pa_logger.DEFAULT_LOG_LEVEL == "info"

    def test_muted_warnings(self):
        """Only the known nuisances are muted; genuinely-new warnings still surface.

        The filters are message + module targeted, so a novel warning is NOT swallowed
        by the default quiet path (the old blanket ``simplefilter`` would have eaten
        it). Muting of the real scipy/pyAMARES messages is pinned end-to-end by
        ``test_fit_silent_when_not_verbose``.
        """
        from xmris.fitting.amares import _muted_warnings

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with _muted_warnings(False):
                warnings.warn("something genuinely new happened", RuntimeWarning)
                warnings.warn("a novel user warning", UserWarning)
            assert len(rec) == 2  # neither novel warning was blanket-muted
            with _muted_warnings(True):
                warnings.warn("shown when verbose", UserWarning)
            assert len(rec) == 3  # verbose passes everything through

    def test_fit_silent_when_not_verbose(self):
        """A fit that trips a real warning leaks nothing at verbose=False.

        Both directions are asserted. The `verbose=True` half is what keeps this
        test honest: it proves the trigger still fires, so the `verbose=False`
        half cannot quietly degrade into asserting silence about nothing. That is
        not hypothetical — the previous trigger was an exactly-zero voxel, which
        `fit_amares` no longer dispatches at all (empty voxels are skipped up
        front), which would have left this test passing vacuously.

        The trigger here is a vanishingly small voxel under `least_squares`: its
        magnitude-derived tolerance falls below machine epsilon, so scipy emits
        the `xtol`/`ftol` UserWarning that `_muted_warnings` targets.
        """
        mhz, sw, n = 120.0, 10000.0, 256
        t = np.arange(n) / sw
        sig = 10.0 * np.exp(-15.0 * np.pi * t) * np.exp(2j * np.pi * 0.0 * mhz * t)
        fid = xr.DataArray(
            sig,
            dims=["time"],
            coords={"time": t},
            attrs={"reference_frequency": mhz, "carrier_ppm": 0.0},
        )
        stack = xr.concat([fid, fid * 1e-30], dim="voxel").assign_attrs(fid.attrs)
        pk = {"PCr": {"amplitude": 10.0, "chem_shift": 0.0, "linewidth": 15.0}}

        def _fit(verbose):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                ds = stack.xmr.fit_amares(
                    prior_knowledge=pk, num_workers=1, method="least_squares", verbose=verbose
                )
            return ds, rec

        ds_loud, loud = _fit(True)
        assert loud, "the trigger stopped firing — this test would now be vacuous"

        ds, rec = _fit(False)
        assert rec == [], f"verbose=False leaked: {[str(w.message) for w in rec]}"
        # Same fit either way: muting changes what is reported, never what is computed.
        # Pinned on the real voxel — the 1e-30 one is degenerate, so its fitted value
        # is float noise and would be fragile to compare at any tolerance.
        assert np.all(np.isfinite(ds[VARS.amplitude].isel(voxel=0).values))
        np.testing.assert_allclose(
            ds[VARS.amplitude].isel(voxel=0).values,
            ds_loud[VARS.amplitude].isel(voxel=0).values,
            rtol=1e-9,
        )


class TestFittingPackaging:
    """Pin the optional-``fitting``-extra guard (pyAMARES).

    Fitting is an optional extra so a bare ``pip install xmris`` stays clean on
    every platform (no ``hlsvdpro``, no ``numpy<2``). These run in a subprocess
    so the result is independent of whatever the test session already imported,
    and they need pyAMARES *absent*, so they carry no ``importorskip``.
    """

    def _run(self, script: str) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("OK"), result.stdout
        return result

    def test_import_does_not_eagerly_load_pyamares(self):
        """``import xmris`` must not pull in pyAMARES — fitting is opt-in."""
        self._run(
            """
            import sys
            import xmris

            leaked = sorted(m for m in sys.modules if m.lower().startswith("pyamares"))
            assert not leaked, f"import xmris eagerly loaded pyAMARES: {leaked}"

            # The dependency-light constructor works without any fitting import.
            fid = xmris.simulate_fid([1.0], frequencies=[100.0], n_points=32)
            assert fid.sizes["time"] == 32
            assert not any(m.lower().startswith("pyamares") for m in sys.modules)
            print("OK")
            """
        )

    def test_fit_amares_absent_raises_friendly(self):
        """With pyAMARES absent, core imports but every fitting path errors clearly."""
        self._run(
            """
            import importlib.abc
            import sys

            class _Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name == "pyAMARES" or name.startswith("pyAMARES."):
                        raise ModuleNotFoundError(name)
                    return None

            sys.meta_path.insert(0, _Block())

            import xmris  # must succeed with pyAMARES "absent"
            fid = xmris.simulate_fid([1.0], frequencies=[100.0], n_points=32)

            def _expect_import_error(fn, label):
                try:
                    fn()
                except ImportError as exc:
                    assert "fitting" in str(exc).lower(), (label, str(exc))
                else:
                    raise AssertionError(f"{label}: expected ImportError")

            def _reach_subpackage():
                from xmris.fitting import fit_amares  # noqa: F401

            _expect_import_error(lambda: xmris.fit_amares, "top-level free function")
            _expect_import_error(_reach_subpackage, "xmris.fitting attribute")
            _expect_import_error(lambda: fid.xmr.fit_amares("x.csv"), "accessor method")
            print("OK")
            """
        )

    def test_star_import_succeeds_without_pyamares(self):
        """`from xmris import *` must not force the lazy resolver on a bare install.

        The star-import iterates `__all__`, so `fit_amares` is pruned when absent.
        """
        self._run(
            """
            import importlib.abc
            import sys

            class _Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name == "pyAMARES" or name.startswith("pyAMARES."):
                        raise ModuleNotFoundError(name)
                    return None

            sys.meta_path.insert(0, _Block())

            top = {}
            exec("from xmris import *", top)
            assert "fit_amares" not in top, "fit_amares leaked into a fitting-free star-import"
            assert "simulate_fid" in top, "simulate_fid missing from star-import"

            # The fitting subpackage star-import must degrade the same way.
            sub = {}
            exec("from xmris.fitting import *", sub)
            assert "fit_amares" not in sub, "fit_amares leaked from xmris.fitting star-import"
            print("OK")
            """
        )

    def test_broken_pyamares_surfaces_real_error(self):
        """A present-but-broken pyAMARES must surface its real ImportError.

        Not the 'install the extra' message, which points at an installed package.
        """
        self._run(
            """
            import sys
            import types

            # pyAMARES is present in sys.modules but missing the symbols amares.py
            # imports, so `from pyAMARES import initialize_FID, ...` raises a real
            # "cannot import name" ImportError.
            sys.modules["pyAMARES"] = types.ModuleType("pyAMARES")

            import xmris

            try:
                xmris.fit_amares
            except ImportError as exc:
                msg = str(exc)
                assert "fitting" not in msg.lower(), ("masked real error:", msg)
                assert "initialize_FID" in msg or "cannot import name" in msg, msg
            else:
                raise AssertionError("expected the real ImportError to surface")
            print("OK")
            """
        )


class TestSetOptions:
    """Verify ``xmris.set_options(auto_convert=False)`` turns coercion into errors.

    Strict mode never changes numbers — it only converts the automatic domain
    transforms into loud, actionable errors, so every Fourier transform in a
    quantitative pipeline is written explicitly. In-domain calls and explicit
    foreign-dim passthrough are unaffected.
    """

    def test_default_is_auto_convert(self):
        """Automatic conversion is on by default."""
        from xmris.core.options import OPTIONS

        assert OPTIONS["auto_convert"] is True

    def test_strict_funnel_raises_actionable(self, valid_fid_da):
        """A funnel coercion under strict mode raises with the explicit fix."""
        import xmris

        with xmris.set_options(auto_convert=False):
            with pytest.raises(ValueError, match="to_spectrum"):
                _ensure_spectral_probe(valid_fid_da)

    def test_strict_restore_raises_actionable(self, valid_spectrum_da):
        """A domain-preserving coercion under strict mode raises with the fix."""
        import xmris

        with xmris.set_options(auto_convert=False):
            with pytest.raises(ValueError, match="to_fid"):
                _time_scale_probe(valid_spectrum_da)

    def test_strict_error_names_the_switch(self, valid_fid_da):
        """The error must name ``auto_convert`` so users can find the switch."""
        import xmris

        with xmris.set_options(auto_convert=False):
            with pytest.raises(ValueError, match="auto_convert"):
                _ensure_spectral_probe(valid_fid_da)

    def test_strict_in_domain_unaffected(self, valid_spectrum_da):
        """In-domain calls need no conversion and work identically under strict."""
        import xmris

        with xmris.set_options(auto_convert=False):
            result = _ensure_spectral_probe(valid_spectrum_da)
        assert result is valid_spectrum_da

    def test_strict_foreign_dim_passthrough_unaffected(self):
        """Explicit foreign-dim passthrough involves no conversion — still works."""
        import xmris

        rng = np.random.default_rng()
        kspace = xr.DataArray(
            rng.standard_normal(16) + 1j * rng.standard_normal(16),
            dims=["kx"],
            coords={"kx": np.arange(16)},
        )
        with xmris.set_options(auto_convert=False):
            result = _time_scale_probe(kspace, dim="kx")
        np.testing.assert_allclose(result.values, kspace.values * 2.0)

    def test_context_manager_restores(self, valid_fid_da):
        """Leaving the context restores automatic conversion."""
        import xmris

        with xmris.set_options(auto_convert=False):
            pass
        result = _ensure_spectral_probe(valid_fid_da)  # converts again
        assert DIMS.frequency in result.dims

    def test_unknown_option_raises(self):
        """Misspelled options must fail loudly, listing the valid ones."""
        import xmris

        with pytest.raises(ValueError, match="Unknown xmris option"):
            xmris.set_options(auto_conver=False)

    def test_mixed_valid_invalid_does_not_leak(self):
        """A valid key alongside an invalid one must not be applied (atomicity)."""
        import xmris
        from xmris.core.options import OPTIONS

        assert OPTIONS["auto_convert"] is True
        with pytest.raises(ValueError, match="Unknown xmris option"):
            xmris.set_options(auto_convert=False, bogus=True)
        # The valid key must not have been applied before the raise; because
        # __init__ raised, __exit__ could never restore it.
        assert OPTIONS["auto_convert"] is True

    def test_non_bool_value_rejected(self):
        """Only real bools are accepted — truthy strings/ints must fail loudly."""
        import xmris
        from xmris.core.options import OPTIONS

        for bad in ("false", 0, 1, None):
            with pytest.raises(ValueError, match="Invalid value"):
                xmris.set_options(auto_convert=bad)
            # A rejected value must never mutate the global.
            assert OPTIONS["auto_convert"] is True

    def test_strict_ppm_hint_routes_through_to_hz(self, valid_spectrum_da):
        """Strict hint for ppm input must suggest to_hz().to_fid(), not bare to_fid()."""
        import xmris

        ppm = valid_spectrum_da.xmr.to_ppm()
        with xmris.set_options(auto_convert=False):
            with pytest.raises(ValueError, match="to_hz") as exc:
                _time_scale_probe(ppm)
        # Following the printed recipe must not dead-end on `to_fid`'s frequency default.
        assert "to_hz().xmr.to_fid()" in str(exc.value)

    def test_strict_real_spectrum_refuses_without_to_fid(self, valid_spectrum_da):
        """Strict mode must reuse the loud real-valued refusal, not suggest to_fid()."""
        import xmris

        real_spectrum = valid_spectrum_da.real  # imaginary part gone → no valid FID
        with xmris.set_options(auto_convert=False):
            with pytest.raises(ValueError, match="real-valued") as exc:
                _time_scale_probe(real_spectrum)
        # A `to_fid()` suggestion here would silently fabricate a bogus FID.
        assert "to_fid" not in str(exc.value)


class TestGroupDelayAttr:
    """``group_delay='header'`` reads the canonical ``group_delay`` attr."""

    @staticmethod
    def _fid_with_attr(key: str, value: float) -> xr.DataArray:
        n = 512
        t = np.arange(n) * 0.0002
        return xr.DataArray(
            np.exp(-t * 30.0) * np.exp(2j * np.pi * 100.0 * t),
            dims=[DIMS.time],
            coords={DIMS.time: t},
            attrs={key: value},
        )

    def test_header_reads_canonical_key(self):
        """``group_delay='header'`` reads the canonical ``group_delay`` attr."""
        from xmris.vendor.bruker import remove_digital_filter

        da = self._fid_with_attr(ATTRS.group_delay, 40.0)
        out = remove_digital_filter(da, group_delay="header")
        assert out.attrs[ATTRS.group_delay_removed] == 40.0

    def test_header_missing_attr_raises(self):
        """``group_delay='header'`` raises a guiding error when the attr is absent."""
        from xmris.vendor.bruker import remove_digital_filter

        da = self._fid_with_attr("unrelated", 1.0)
        with pytest.raises(ValueError, match=re.escape(ATTRS.group_delay)):
            remove_digital_filter(da, group_delay="header")


class TestEstimateGroupDelayRobustness:
    """Regression tests for the review's estimator bugs (crash / degenerate handling)."""

    @staticmethod
    def _fid(n: int, freqs=(50.0,), amps=None) -> xr.DataArray:
        from xmris.fitting.simulation import simulate_fid

        amps = list(amps) if amps is not None else [1.0] * len(freqs)
        return simulate_fid(
            amplitudes=amps, frequencies=list(freqs), spectral_width=5000.0, n_points=n
        )

    def test_short_fid_does_not_crash(self):
        """A FID shorter than the default search ceiling returns a float, not a ValueError."""
        d = self._fid(64).xmr.estimate_group_delay()
        assert isinstance(d, float)
        assert 0.0 <= d <= 63.0

    def test_overlong_explicit_delay_raises_clearly(self):
        """remove_digital_filter rejects a delay >= the FID length with a clear message."""
        from xmris.vendor.bruker import remove_digital_filter

        with pytest.raises(ValueError, match="removes .* points, but"):
            remove_digital_filter(self._fid(64), group_delay=65.0)

    def test_measure_sentinel_short_fid(self):
        """group_delay='measure' on a short FID completes without crashing."""
        from xmris.vendor.bruker import remove_digital_filter

        out = remove_digital_filter(self._fid(64), group_delay="measure")
        assert out.sizes[DIMS.time] == 64

    def test_wide_range_stays_finite_and_in_bounds(self):
        """A search_range reaching the FID length completes with a finite, in-range result."""
        fid = self._fid(256, freqs=(300.0, 1200.0), amps=(1.0, 0.7))
        d = fid.xmr.estimate_group_delay(search_range=(70, 255))
        assert np.isfinite(d)
        assert 70.0 <= d <= 255.0


class TestPlotQCGridDomain:
    """`plot_qc_grid` must handle a domain-preserving fit result in either domain.

    `fit_amares` returns data/fit/residuals in the caller's domain, so a
    spectrum-domain fit is already spectral and must not be FFT'd as if it were a
    time-domain FID (finding 02). These are pyAMARES-free: the Dataset is synthetic.
    """

    def _fit_ds(self, axis_dim: str, axis_units: str) -> xr.Dataset:
        """Build a minimal AMARES-shaped Dataset with the signals on ``axis_dim``."""
        rng = np.random.default_rng(0)
        n_rep, n_pts, n_metab = 3, 64, 2
        params = [VARS.amplitude, VARS.chem_shift, VARS.linewidth, VARS.phase]

        def _signal():
            return rng.standard_normal((n_rep, n_pts)) + 1j * rng.standard_normal((n_rep, n_pts))

        data_dims = ("repetition", axis_dim)
        return xr.Dataset(
            {
                VARS.original_data: (data_dims, _signal()),
                VARS.fit: (data_dims, _signal()),
                VARS.residuals: (data_dims, _signal()),
                VARS.crlb: (
                    ("repetition", DIMS.metabolite, DIMS.parameter),
                    rng.uniform(1.0, 10.0, (n_rep, n_metab, len(params))),
                ),
            },
            coords={
                "repetition": np.arange(n_rep),
                axis_dim: xr.DataArray(
                    np.linspace(0.0, 5.0, n_pts), dims=axis_dim, attrs={"units": axis_units}
                ),
                DIMS.metabolite: [f"m{i}" for i in range(n_metab)],
                DIMS.parameter: params,
            },
        )

    def _render(self, ds: xr.Dataset):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xmris.visualization.plot.plot_qc_grid import plot_qc_grid

        fig = plot_qc_grid(ds, dim="repetition")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_spectrum_domain_fit_renders(self):
        """A ppm-domain fit result renders without the old `dim='time'` crash."""
        self._render(self._fit_ds(DIMS.chemical_shift, "ppm"))

    def test_time_domain_fit_still_renders(self):
        """A time-domain (FID) fit result still renders — no regression."""
        self._render(self._fit_ds(DIMS.time, "s"))
