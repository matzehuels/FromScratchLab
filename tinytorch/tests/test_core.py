"""Tests for core Tensor functionality."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tests.conftest import arrays_strategy, tensors_strategy
from tinytorch.engine import Operation, Tensor


@given(arrays_strategy())
def test_constructor_and_core_properties(array: np.ndarray) -> None:
    """Test tensor construction.

    Tests: data type, shape, repr
    """
    tensor = Tensor(data=array)
    assert isinstance(tensor, Tensor)
    assert isinstance(tensor.data, np.ndarray)
    assert isinstance(tensor.shape, tuple)
    assert tensor.shape == array.shape
    assert repr(tensor) == f"Tensor(shape={array.shape})"


@given(tensors_strategy())
def test_negation_operation(tensor: Tensor) -> None:
    """Test tensor negation.

    Tests: double negation, -1 multiplication, zero handling
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
    """Test tensor labeling.

    Tests: label storage, None handling
    """
    tensor = Tensor(data=array, label=label)
    assert tensor.label == label

    # Test label can be None
    tensor_no_label = Tensor(data=array)
    assert tensor_no_label.label is None


@given(tensors_strategy())
def test_repr_equals_str(tensor):
    """Test string representations.

    Tests: __repr__ == __str__, labeled tensors
    """
    assert str(tensor) == repr(tensor)

    # Also test with a labeled tensor
    labeled = Tensor(tensor.data, label="test")
    assert str(labeled) == repr(labeled)
