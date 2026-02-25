from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr
from xmris.core.accessor import _check_dims

from xmris.core.config import ATTRS, COORDS, DIMS
from xmris.core.validation import requires_attrs

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_da():
    """Returns a basic 1D DataArray with no special attributes or dimensions."""
    data = np.random.rand(100)
    return xr.DataArray(data, dims=["x"], coords={"x": np.arange(100)})


@pytest.fixture
def valid_mrs_da():
    """Returns a perfectly formatted MRS DataArray simulating a spectrum."""
    data = np.random.rand(1024) + 1j * np.random.rand(1024)
    da = xr.DataArray(
        data,
        dims=[DIMS.frequency],
        coords={DIMS.frequency: np.linspace(-5000, 5000, 1024)},
    )
    # Assign required attributes
    da = da.assign_attrs(
        {
            ATTRS.b0_field: 3.0,
            ATTRS.reference_frequency: 127.8,  # MHz
        }
    )
    return da


# =============================================================================
# 1. Configuration & Vocabulary Tests
# =============================================================================


def test_config_immutability():
    """Ensure that developers or users cannot accidentally overwrite the vocabularies at runtime."""
    with pytest.raises(FrozenInstanceError):
        ATTRS.b0_field = "new_b0_key"


def test_config_get_description():
    """Test that the base vocabulary correctly extracts metadata descriptions."""
    # Test valid key
    desc = ATTRS.get_description("MHz")
    assert "working/reference frequency" in desc

    # Test invalid key
    bad_desc = ATTRS.get_description("non_existent_key")
    assert bad_desc == "Unknown xarray key."


def test_config_html_repr():
    """Smoke test to ensure the Jupyter HTML representation renders without crashing."""
    html = ATTRS._repr_html_()
    assert "MHz" in html
    assert "<table" in html


# =============================================================================
# 2. Decorator Engine Tests
# =============================================================================


class DummyAccessor:
    """A minimal mock accessor to test the decorator in isolation."""

    def __init__(self, obj: xr.DataArray):
        self._obj = obj

    @requires_attrs(ATTRS.b0_field, ATTRS.reference_frequency)
    def compute_something(self):
        """Original docstring."""
        return self._obj.attrs[ATTRS.b0_field] * 2


def test_requires_attrs_missing(empty_da):
    """Test that the decorator correctly catches missing attributes and guides the user."""
    accessor = DummyAccessor(empty_da)

    with pytest.raises(ValueError, match="requires the following missing attributes"):
        accessor.compute_something()

    # Test that the error message contains the helpful xarray fix prompt
    try:
        accessor.compute_something()
    except ValueError as e:
        error_msg = str(e)
        assert "To fix this, assign them using standard xarray methods" in error_msg
        assert "assign_attrs" in error_msg


def test_requires_attrs_success(valid_mrs_da):
    """Test that the decorator gets out of the way when attributes are present."""
    accessor = DummyAccessor(valid_mrs_da)
    result = accessor.compute_something()
    assert result == 6.0  # 3.0 (from b0_field) * 2


def test_requires_attrs_docstring_generation():
    """Test that the decorator automatically injects the requirements into the docstring."""
    doc = DummyAccessor.compute_something.__doc__
    assert "Original docstring." in doc
    assert "Required Attributes" in doc
    assert "b0_field" in doc
    assert "MHz" in doc


# =============================================================================
# 3. Accessor & Dimensions Tests
# =============================================================================


def test_check_dims_missing(empty_da):
    """Test the internal dimension validation helper."""
    with pytest.raises(ValueError, match="attempted to operate on missing dimension"):
        _check_dims(empty_da, dims=DIMS.time, func_name="test_func")

    try:
        _check_dims(empty_da, dims=DIMS.time, func_name="test_func")
    except ValueError as e:
        error_msg = str(e)
        assert "rename your data's axes using xarray" in error_msg
        assert "rename(" in error_msg


def test_check_dims_success(empty_da):
    """Test that explicit custom dimensions bypass the defaults perfectly."""
    # empty_da has the dimension "x". If we explicitly pass "x", it should not raise an error.
    _check_dims(empty_da, dims="x", func_name="test_func")


# =============================================================================
# 4. Integration Test: The `.xmr.to_ppm()` method
# =============================================================================


def test_to_ppm_success(valid_mrs_da):
    """Test the complete pipeline: accessor -> check_dims -> requires_attrs -> math."""
    # 0 Hz / 127.8 MHz = 0 ppm

    da_ppm = valid_mrs_da.xmr.to_ppm()

    # 1. Check that the ppm coordinate was created
    assert COORDS.ppm in da_ppm.coords

    # 2. Verify the math is correct (Hz / MHz)
    ppm_values = da_ppm.coords[COORDS.ppm].values
    hz_values = valid_mrs_da.coords[DIMS.frequency].values
    mhz = valid_mrs_da.attrs[ATTRS.reference_frequency]

    np.testing.assert_array_almost_equal(ppm_values, hz_values / mhz)


def test_to_ppm_fails_cleanly(valid_mrs_da):
    """Ensure that integration fails safely if the user breaks the data mid-script."""
    # User drops the MHz attribute
    broken_da = valid_mrs_da.assign_attrs({ATTRS.reference_frequency: None})
    del broken_da.attrs[ATTRS.reference_frequency]

    with pytest.raises(ValueError, match="missing attributes"):
        broken_da.xmr.to_ppm()
