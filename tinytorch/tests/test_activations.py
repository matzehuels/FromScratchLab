"""Tests for neural network activation functions."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from tests.conftest import (
    ATOL,
    RTOL,
    default_floats_strategy,
    same_shape_tensors_strategy,
    tensors_strategy,
)
from tinytorch.engine import Operation, Tensor


@given(same_shape_tensors_strategy())
@settings(deadline=None)
def test_max_operation(tensors: tuple[Tensor, Tensor]) -> None:
    """Test element-wise maximum operation."""
    t1, t2 = tensors
    result = t1.max(t2)
    assert result._op == Operation.MAX
    np.testing.assert_array_equal(result.data, np.maximum(t1.data, t2.data))


@given(tensors_strategy())
def test_lin_operation(tensor):
    """Test linear (identity) activation function.

    Properties tested:
    1. Basic identity function: output equals input
    2. Operation type is correctly set
    3. Children are tracked properly
    4. Numerical correctness against numpy
    """
    result = tensor.lin()
    assert result._op == Operation.IDENT
    assert result._children == {tensor}
    np.testing.assert_array_equal(
        result.data,
        tensor.data,
        "Linear activation should match input exactly",
    )


@given(tensors_strategy(), default_floats_strategy)
def test_max_with_scalar(tensor: Tensor, scalar: float) -> None:
    """Test maximum operation with scalar."""
    result = tensor.max(scalar)
    assert result._op == Operation.MAX
    np.testing.assert_array_equal(result.data, np.maximum(tensor.data, scalar))


@given(tensors_strategy())
def test_relu_operation(tensor: Tensor) -> None:
    """Test ReLU activation function."""
    result = tensor.relu()
    np.testing.assert_array_equal(result.data, np.maximum(tensor.data, 0))


@given(tensors_strategy())
def test_exp_operation(t1: Tensor) -> None:
    """Test basic functionality of e^x."""
    result = t1.exp()
    assert result._op == Operation.EXP
    assert result._children == {t1}
    np.testing.assert_array_equal(result.data, np.exp(t1.data))


@given(tensors_strategy())
def test_sigmoid_operation(tensor: Tensor) -> None:
    """Test sigmoid activation function."""
    result = tensor.sigmoid()
    np.testing.assert_allclose(result.data, 1 / (1 + np.exp(-tensor.data)), rtol=RTOL, atol=ATOL)


@given(
    tensors_strategy(
        floats_strategy=st.floats(
            min_value=0.0010000000474974513,
            max_value=10,
            allow_infinity=False,
            allow_nan=False,
            allow_subnormal=False,
            width=32,
        )
    )
)
def test_tanh_operation(tensor: Tensor) -> None:
    """Test hyperbolic tangent function."""
    result = tensor.tanh()
    np.testing.assert_allclose(result.data, np.tanh(tensor.data), rtol=RTOL, atol=ATOL)
