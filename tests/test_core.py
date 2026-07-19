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
import pickle

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
from xmris.core.utils import _resolve_dim, read_attr
from xmris.core.validation import computes_in, ensures_domain, requires_attrs
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
# 3b. Configuration: Term Hardening (immutability, aliases, uniqueness)
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

    def test_bare_string_alias_rejected(self):
        """A bare-str ``aliases`` (missing trailing comma) must raise, not explode per-char."""
        with pytest.raises(TypeError, match="not a bare str"):
            XmrisTerm("modern", description="d", aliases="legacy")

    def test_aliases_are_plain_string_tuples(self):
        """Aliases must be tuples of *plain* str (metadata-free by construction)."""
        for vocab in (ATTRS, DIMS, COORDS, VARS):
            for prop_name, term in vocab._get_terms().items():
                assert isinstance(term.aliases, tuple)
                assert all(type(a) is str for a in term.aliases), (
                    f"{vocab.__class__.__name__}.{prop_name} aliases must be plain str."
                )

    def test_alias_normalization(self):
        """Alias inputs are normalized to plain strings, even when given as terms."""
        term = XmrisTerm("modern", description="d", aliases=(XmrisTerm("legacy"), "older"))
        assert term.aliases == ("legacy", "older")
        assert all(type(a) is str for a in term.aliases)

    @pytest.mark.parametrize("proto", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_roundtrip_preserves_metadata_and_freeze(self, proto):
        """Pickle round-trips keep value, metadata, and immutability on every protocol."""
        term = pickle.loads(pickle.dumps(ATTRS.reference_frequency, proto))
        assert term == ATTRS.reference_frequency
        assert term.description == ATTRS.reference_frequency.description
        assert term.unit == "MHz"
        assert term.aliases == ("MHz",)
        with pytest.raises(AttributeError):
            term.unit = "Hz"

    def test_deepcopy_roundtrip(self):
        """``copy.deepcopy`` keeps value, metadata, and immutability."""
        term = copy.deepcopy(ATTRS.group_delay)
        assert term == ATTRS.group_delay
        assert term.unit == "samples"
        assert term.aliases == ("bruker_group_delay",)
        with pytest.raises(AttributeError):
            term.unit = "x"


class TestVocabularyUniqueness:
    """Duplicate keys (canonical or alias) inside one vocabulary fail at import time."""

    def test_duplicate_canonical_value_raises(self):
        """Two terms with the same canonical value must fail at class definition."""
        with pytest.raises(ValueError, match="duplicate vocabulary key 'twin'"):

            class _Broken(BaseVocabulary):
                a = XmrisTerm("twin", description="d")
                b = XmrisTerm("twin", description="d")

    def test_alias_colliding_with_canonical_raises(self):
        """An alias shadowing another term's canonical value must fail."""
        with pytest.raises(ValueError, match="duplicate vocabulary key 'taken'"):

            class _Broken(BaseVocabulary):
                a = XmrisTerm("taken", description="d")
                b = XmrisTerm("other", description="d", aliases=("taken",))

    def test_alias_colliding_with_alias_raises(self):
        """Two terms claiming the same alias must fail."""
        with pytest.raises(ValueError, match="duplicate vocabulary key 'legacy'"):

            class _Broken(BaseVocabulary):
                a = XmrisTerm("a", description="d", aliases=("legacy",))
                b = XmrisTerm("b", description="d", aliases=("legacy",))

    def test_well_formed_vocabulary_constructs(self):
        """Distinct values and aliases construct without error."""

        class _Fine(BaseVocabulary):
            a = XmrisTerm("a", description="d", aliases=("old_a",))
            b = XmrisTerm("b", description="d")

        assert _Fine()._get_terms().keys() == {"a", "b"}

    def test_get_description_resolves_alias(self):
        """``get_description`` on a legacy alias points at the canonical term."""
        desc = ATTRS.get_description("MHz")
        assert "Legacy alias of 'reference_frequency'" in desc


class TestReadAttr:
    """The shared alias-aware attrs resolver: canonical key wins, aliases fall back."""

    @staticmethod
    def _da(attrs: dict) -> xr.DataArray:
        return xr.DataArray(np.zeros(4), dims=[DIMS.time], attrs=attrs)

    def test_canonical_wins_over_alias(self):
        """The canonical key is preferred when both canonical and alias are present."""
        da = self._da({ATTRS.reference_frequency: 400.1, "MHz": 999.0})
        assert read_attr(da, ATTRS.reference_frequency) == 400.1

    def test_alias_fallback(self):
        """A legacy alias key is read when the canonical key is absent."""
        da = self._da({"MHz": 120.3})
        assert read_attr(da, ATTRS.reference_frequency) == 120.3

    def test_missing_returns_default(self):
        """Neither key present returns the default (None unless overridden)."""
        da = self._da({"unrelated": 1.0})
        assert read_attr(da, ATTRS.reference_frequency) is None
        assert read_attr(da, ATTRS.reference_frequency, default=1.5) == 1.5


class TestParseInputDims:
    """Stack-dim auto-detect uses singular vocabulary dims + legacy plural aliases."""

    @staticmethod
    def _da(dims: tuple[str, ...]) -> xr.DataArray:
        return xr.DataArray(np.zeros((4, 3, 2)), dims=list(dims))

    @pytest.mark.parametrize("stack", ["average", "averages", "repetition", "repetitions"])
    def test_detects_singular_and_legacy_plural(self, stack):
        """Auto-detect matches the name the data actually carries (was AttributeError).

        ``coil`` is placed FIRST so it is ``remaining_dims[0]`` (the positional
        fallback). Only correct alias-matching returns ``stack``; a broken match
        loop would fall back to ``coil`` and fail the assertion.
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

    @pytest.mark.parametrize("rep_dim", [str(DIMS.repetition), "repetitions"])
    def test_build_fid_writes_tr_coordinate(self, rep_dim):
        """The TR coordinate is written for canonical and legacy plural dim names."""
        from xmris.vendor.bruker import build_fid

        data = np.zeros((8, 4), dtype=complex)
        da = build_fid(data, [DIMS.time, rep_dim], self._PV_PARAMS)
        assert rep_dim in da.coords
        np.testing.assert_allclose(da.coords[rep_dim].values, np.arange(1, 5) * 1.0)
        assert da.coords[rep_dim].attrs["units"] == "s"


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
    representation, and converters/primitives/fitting stay undecorated with
    explicit domain handling.
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
        """Converters, primitives, and fitting must NOT auto-convert."""
        from xmris.fitting.amares import fit_amares
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
            fit_amares,
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


# =============================================================================
# 16. Runtime Options: strict mode (auto_convert=False)
# =============================================================================


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
    """The ``group_delay`` attr rename keeps reading legacy ``bruker_group_delay`` keys."""

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

    def test_header_reads_legacy_key(self):
        """``group_delay='header'`` falls back to the legacy ``bruker_group_delay`` attr."""
        from xmris.vendor.bruker import remove_digital_filter

        da = self._fid_with_attr("bruker_group_delay", 40.0)
        out = remove_digital_filter(da, group_delay="header")
        assert out.attrs[ATTRS.group_delay_removed] == 40.0

    def test_header_prefers_canonical_key(self):
        """The canonical ``group_delay`` key wins when both keys are present."""
        from xmris.vendor.bruker import remove_digital_filter

        da = self._fid_with_attr("bruker_group_delay", 40.0)
        da.attrs[ATTRS.group_delay] = 50.0
        out = remove_digital_filter(da, group_delay="header")
        assert out.attrs[ATTRS.group_delay_removed] == 50.0

    def test_read_attr_prefers_canonical_and_falls_back(self):
        """``read_attr(da, ATTRS.group_delay)`` prefers the new key, falls back, else None."""
        legacy = self._fid_with_attr("bruker_group_delay", 76.125)
        assert read_attr(legacy, ATTRS.group_delay) == 76.125
        assert read_attr(self._fid_with_attr(ATTRS.group_delay, 84.0), ATTRS.group_delay) == 84.0
        assert read_attr(self._fid_with_attr("unrelated", 1.0), ATTRS.group_delay) is None


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
