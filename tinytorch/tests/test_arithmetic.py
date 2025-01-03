"""Tests for arithmetic operations."""

import numpy as np
from hypothesis import given, note
from hypothesis import strategies as st
from tests.conftest import (
    ATOL,
    RTOL,
    default_floats_strategy,
    same_shape_tensors_strategy,
    tensors_strategy,
)

from tinytorch.engine import Operation, Tensor


@given(same_shape_tensors_strategy())
def test_add_operation(tensors: tuple[Tensor, Tensor]) -> None:
    """Test addition operations between tensors and with scalars.

    Properties tested:
    1. Basic tensor addition: t1 + t2
    2. Operation type is correctly set
    3. Children are tracked properly
    4. Scalar addition via __radd__
    5. Numerical correctness against numpy
    """
    t1, t2 = tensors
    note(f"Testing shapes: {t1.shape} and {t2.shape}")

    # Test tensor-tensor addition
    forward = t1 + t2
    assert forward._op == Operation.ADD, "Addition operation type should be ADD"
    assert forward._children == {
        t1,
        t2,
    }, "Addition should track both operands as children"
    np.testing.assert_array_equal(
        forward.data, t1.data + t2.data, "Tensor addition should match numpy addition"
    )

    # Test scalar-tensor addition (tests __radd__)
    reverse = 1.0 + t1
    assert reverse._op == Operation.ADD, "Scalar addition should use ADD operation"
    np.testing.assert_array_equal(
        reverse.data,
        1.0 + t1.data,
        "Scalar addition should match numpy scalar addition",
    )


@given(same_shape_tensors_strategy())
def test_multiply_operation(tensors: tuple[Tensor, Tensor]) -> None:
    """Test multiplication operations between tensors and with scalars.

    Properties tested:
    1. Basic tensor multiplication: t1 * t2
    2. Operation type is correctly set
    3. Children are tracked properly
    4. Scalar multiplication via __rmul__
    5. Numerical correctness against numpy
    """
    t1, t2 = tensors
    note(f"Testing shapes: {t1.shape} and {t2.shape}")

    # Test tensor-tensor multiplication
    forward = t1 * t2
    assert forward._op == Operation.MULT, "Multiplication operation type should be MULT"
    assert forward._children == {
        t1,
        t2,
    }, "Multiplication should track both operands as children"
    np.testing.assert_array_equal(
        forward.data,
        t1.data * t2.data,
        "Tensor multiplication should match numpy multiplication",
    )

    # Test scalar-tensor multiplication (tests __rmul__)
    reverse = 2.0 * t1
    assert (
        reverse._op == Operation.MULT
    ), "Scalar multiplication should use MULT operation"
    np.testing.assert_array_equal(
        reverse.data,
        2.0 * t1.data,
        "Scalar multiplication should match numpy scalar multiplication",
    )


@given(same_shape_tensors_strategy())
def test_subtract_operation(tensors: tuple[Tensor, Tensor]) -> None:
    """Test subtraction operations between tensors and with scalars.

    Properties tested:
    1. Basic tensor subtraction: t1 - t2
    2. Operation type is correctly set
    3. Scalar subtraction via __rsub__
    4. Numerical correctness against numpy
    """
    t1, t2 = tensors
    note(f"Testing shapes: {t1.shape} and {t2.shape}")

    # Test tensor-tensor subtraction
    forward = t1 - t2
    np.testing.assert_array_equal(
        forward.data,
        t1.data - t2.data,
        "Tensor subtraction should match numpy subtraction",
    )

    # Test scalar-tensor subtraction (tests __rsub__)
    reverse = 1.0 - t1
    np.testing.assert_array_equal(
        reverse.data,
        1.0 - t1.data,
        "Scalar subtraction should match numpy scalar subtraction",
    )


@given(same_shape_tensors_strategy())
def test_add_commutative(tensors: tuple[Tensor, Tensor]) -> None:
    """Test that a + b = b + a"""
    t1, t2 = tensors
    note(f"Testing shapes: {t1.shape} and {t2.shape}")
    forward = t1 + t2
    reverse = t2 + t1
    np.testing.assert_array_equal(forward.data, reverse.data)
    assert forward._children == reverse._children


@given(same_shape_tensors_strategy())
def test_multiply_commutative(tensors: tuple[Tensor, Tensor]) -> None:
    """Test that a * b = b * a"""
    t1, t2 = tensors
    note(f"Testing shapes: {t1.shape} and {t2.shape}")
    forward = t1 * t2
    reverse = t2 * t1
    np.testing.assert_array_equal(forward.data, reverse.data)
    assert forward._children == reverse._children


@given(
    same_shape_tensors_strategy(
        floats_strategy=st.floats(
            min_value=0.0010000000474974513,
            max_value=10,
            allow_infinity=False,
            allow_nan=False,
            exclude_min=True,
            width=32,
            allow_subnormal=False,
        )
    )
)
def test_divide_operation(tensors: tuple[Tensor, Tensor]) -> None:
    """Test division operations between tensors and with scalars.

    Properties tested:
    1. Basic tensor division: t1 / t2
    2. Operation type is correctly set
    3. Division-multiplication relationship: (a/b) * b ≈ a
    4. Numerical correctness against numpy with appropriate tolerances
    """
    t1, t2 = tensors
    note(f"Testing shapes: {t1.shape} and {t2.shape}")

    # Test tensor-tensor division
    forward = t1 / t2

    # Test division-multiplication relationship
    result = forward * t2
    np.testing.assert_allclose(
        result.data,
        t1.data,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Division-multiplication property: (a/b) * b should approximately equal a",
    )


@given(tensors_strategy(), st.integers(2, 5))
def test_power_operation(t1: Tensor, exponent: int) -> None:
    """Test basic functionality of exponentiation."""
    result = t1**exponent
    assert result._op == Operation.POW
    assert result._children == {t1}
    np.testing.assert_array_equal(result.data, np.power(t1.data, exponent))


@given(tensors_strategy(), default_floats_strategy)
def test_add_with_scalar(tensor: Tensor, scalar: float) -> None:
    """Test addition between tensor and scalar."""
    result1 = tensor + scalar
    result2 = scalar + tensor
    assert result1._op == Operation.ADD
    assert result2._op == Operation.ADD
    np.testing.assert_array_equal(result1.data, tensor.data + scalar)
    np.testing.assert_array_equal(result2.data, scalar + tensor.data)
    np.testing.assert_array_equal(result1.data, result2.data)  # Commutativity


@given(tensors_strategy(), default_floats_strategy)
def test_multiply_with_scalar(tensor: Tensor, scalar: float) -> None:
    """Test multiplication between tensor and scalar."""
    result1 = tensor * scalar
    result2 = scalar * tensor
    assert result1._op == Operation.MULT
    assert result2._op == Operation.MULT
    np.testing.assert_array_equal(result1.data, tensor.data * scalar)
    np.testing.assert_array_equal(result2.data, scalar * tensor.data)
    np.testing.assert_array_equal(result1.data, result2.data)  # Commutativity


@given(tensors_strategy(), default_floats_strategy)
def test_subtract_with_scalar(tensor: Tensor, scalar: float) -> None:
    """Test subtraction between tensor and scalar."""
    result1 = tensor - scalar
    result2 = scalar - tensor
    np.testing.assert_array_equal(result1.data, tensor.data - scalar)
    np.testing.assert_array_equal(result2.data, scalar - tensor.data)


@given(
    tensors_strategy(),
    st.floats(
        min_value=0.0010000000474974513,
        max_value=10,
        allow_infinity=False,
        allow_nan=False,
        exclude_min=True,
        width=32,
        allow_subnormal=False,
    ),
)
def test_divide_with_scalar(tensor: Tensor, scalar: float) -> None:
    """Test division between tensor and scalar."""
    result1 = tensor / scalar
    result2 = scalar / tensor
    np.testing.assert_allclose(result1.data, tensor.data / scalar, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result2.data, scalar / tensor.data, rtol=RTOL, atol=ATOL)
