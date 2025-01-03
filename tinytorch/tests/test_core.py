"""Tests for core Tensor functionality."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tinytorch.engine import Operation, Tensor
from tests.conftest import arrays_strategy, tensors_strategy


@given(arrays_strategy())
def test_constructor_and_core_properties(array: np.ndarray) -> None:
    """Test basic functionality of Tensor class."""
    tensor = Tensor(data=array)
    assert isinstance(tensor, Tensor)
    assert isinstance(tensor.data, np.ndarray)
    assert isinstance(tensor.shape, tuple)
    assert tensor.shape == array.shape
    assert repr(tensor) == f"Tensor(shape={array.shape})"


@given(tensors_strategy())
def test_negation_operation(tensor: Tensor) -> None:
    """Test tensor negation properties:
    1. Double negation returns original: --x = x
    2. Negation is same as multiplying by -1: -x = (-1 * x)
    3. Negating zero returns zero
    """
    # Test double negation
    double_neg = -(-tensor)
    np.testing.assert_array_equal(double_neg.data, tensor.data)

    # Test negation is same as multiplying by -1
    neg = -tensor
    mult_neg = tensor * -1
    np.testing.assert_array_equal(neg.data, mult_neg.data)
    assert neg._op == Operation.MULT  # Negation uses multiplication internally

    # Test negating zero returns zero
    zero_tensor = Tensor(np.zeros_like(tensor.data))
    np.testing.assert_array_equal((-zero_tensor).data, zero_tensor.data)


@given(arrays_strategy(), st.text(min_size=1))
def test_tensor_labeling(array: np.ndarray, label: str) -> None:
    """Test that tensors can be labeled."""
    tensor = Tensor(data=array, label=label)
    assert tensor.label == label

    # Test label can be None
    tensor_no_label = Tensor(data=array)
    assert tensor_no_label.label is None


@given(tensors_strategy())
def test_repr_equals_str(tensor):
    """Test that __repr__ and __str__ return the same value."""
    assert str(tensor) == repr(tensor)

    # Also test with a labeled tensor
    labeled = Tensor(tensor.data, label="test")
    assert str(labeled) == repr(labeled)
