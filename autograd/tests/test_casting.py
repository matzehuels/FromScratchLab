"""Tests for type casting functionality."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from autograd.engine import Tensor, TensorLike, _cast_array, _cast_tensor
from tests.conftest import (
    arrays_strategy,
    default_floats_strategy,
    default_ints_strategy,
    default_lists_strategy,
    tensors_strategy,
)


@given(
    st.one_of(
        st.dictionaries(st.text(), st.integers()),  # dict
        st.sets(st.integers()),  # set
        st.text(),  # str
        st.binary(),  # bytes
        st.none(),  # None
    )
)
def test_cast_array_invalid_input(invalid_data) -> None:
    """Test that _cast_array raises ValueError for invalid input types."""
    with pytest.raises(ValueError):
        _cast_array(invalid_data)


@given(
    st.one_of(
        st.dictionaries(st.text(), st.integers()),
        st.sets(st.integers()),
        st.text(),
        st.binary(),
        st.none(),
    )
)
def test_cast_tensor_invalid_input(invalid_data) -> None:
    """Test that _cast_tensor raises ValueError for invalid input types."""
    with pytest.raises(ValueError):
        _cast_tensor(invalid_data)


@given(
    st.one_of(
        arrays_strategy(), default_floats_strategy, default_ints_strategy, default_lists_strategy
    )
)
def test_cast_array(data: TensorLike) -> None:
    """Test casting various types to numpy array."""
    result = _cast_array(data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32

    if isinstance(data, (float, int)):
        assert result.shape == (1,)
    elif isinstance(data, list):
        assert result.shape == (len(data),)
    elif isinstance(data, np.ndarray):
        assert result.shape == data.shape
    elif isinstance(data, Tensor):
        assert result.shape == data.shape


@given(
    st.one_of(
        arrays_strategy(),
        tensors_strategy(),
        default_floats_strategy,
        default_ints_strategy,
        default_lists_strategy,
    )
)
def test_cast_tensor(data: TensorLike) -> None:
    """Test casting various types to Tensor."""
    result = _cast_tensor(data)
    assert isinstance(result, Tensor)
    assert result.data.dtype == np.float32

    if isinstance(data, (float, int)):
        assert result.shape == (1,)
    elif isinstance(data, list):
        assert result.shape == (len(data),)
    elif isinstance(data, np.ndarray):
        assert result.shape == data.shape
    elif isinstance(data, Tensor):
        assert result.shape == data.shape
