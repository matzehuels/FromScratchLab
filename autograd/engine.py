"""
This module implements the core concepts of automatic differentiation (autodiff),
a technique for efficiently computing derivatives of functions defined by a
computational graph. It provides forward and reverse mode differentiation,
fundamental operations, and utilities for building and differentiating custom
functions. The implementation leverages NumPy for efficient numerical operations.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np

TensorLike = Union[np.ndarray, List[float | int], float, int, "Tensor"]


class Operation(Enum):
    """Basic operations that are supported."""

    ADD = "+"
    MULT = "*"
    EXP = "exp"
    POW = "**"
    SUBTRACT = "-"
    DIVIDE = "/"
    MAX = "max"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    RELU = "relu"


def _cast_array(x: TensorLike, shape: Tuple[int, ...] = (1,)) -> np.ndarray:
    """Casts compatible datatypes to numpy array, raises if not compatible."""
    if isinstance(x, np.ndarray):
        out = x.astype(np.float32)
    elif isinstance(x, list):
        out = np.array(x, dtype=np.float32)
    elif isinstance(x, (float, int)):
        out = np.ones(shape, dtype=np.float32) * x
    else:
        raise ValueError()
    return out


def _cast_tensor(x: TensorLike, shape: Tuple[int, ...] = (1,)) -> Tensor:
    """Casts compatible datatypes to Tensor, raises if not compatible."""
    out = x if isinstance(x, Tensor) else Tensor(_cast_array(x, shape))
    return out


class Tensor(object):
    def __init__(
        self,
        data: TensorLike,
        _children: Optional[Tuple[Tensor, ...]] = None,
        _op: Optional[Operation] = None,
    ) -> None:
        """Initialize a new Tensor with data and optional children and operation."""
        self.data = _cast_array(data)
        self._op = _op
        self._children = set(_children) if _children is not None else set()

    def __repr__(self) -> str:
        """Return string representation of the tensor."""
        return f"Tensor(shape={self.shape})"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor's data."""
        return self.data.shape

    def __add__(self, other: TensorLike) -> Tensor:
        """Add other tensor-like object to this tensor."""
        other = _cast_tensor(other)
        out = Tensor(self.data + other.data, (self, other), Operation.ADD)
        return out

    def __radd__(self, other: TensorLike) -> Tensor:
        """Handle addition when tensor is the right operand."""
        other = _cast_tensor(other)
        return self + other

    def __mul__(self, other: TensorLike) -> Tensor:
        """Multiply other tensor-like object with this tensor."""
        other = _cast_tensor(other)
        out = Tensor(self.data * other.data, (self, other), Operation.MULT)
        return out

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Handle multiplication when tensor is the right operand."""
        other = _cast_tensor(other)
        return self * other

    def __neg__(self) -> Tensor:
        """Return the negation of this tensor."""
        return self * -1

    def __sub__(self, other: TensorLike) -> Tensor:
        """Subtract other tensor-like object from this tensor."""
        other = _cast_tensor(other)
        out = self + (-1 * other)
        out._op = Operation.SUBTRACT
        return out

    def __rsub__(self, other: TensorLike) -> Tensor:
        """Handle subtraction when tensor is the right operand."""
        other = _cast_tensor(other)
        return other - self

    def __pow__(self, exponent: float | int) -> Tensor:
        """Raise tensor to the power of exponent."""
        return Tensor(np.pow(self.data, exponent), (self,), Operation.POW)

    def __truediv__(self, other: TensorLike) -> Tensor:
        """Divide tensor by other tensor-like object."""
        other = _cast_tensor(other)
        out = self * (other**-1)
        out._op = Operation.DIVIDE
        return out

    def __rtruediv__(self, other: TensorLike) -> "Tensor":
        """Handle division when tensor is the denominator (other / self)."""
        other = _cast_tensor(other)
        out = other * self ** (-1)
        out._op = Operation.DIVIDE
        return out

    def exp(self) -> Tensor:
        """Compute element-wise exponential of tensor."""
        return Tensor(np.exp(self.data), (self,), Operation.EXP)

    def max(self, other: TensorLike) -> Tensor:
        """Return element-wise maximum between self and other."""
        other = _cast_tensor(other)
        out = Tensor(np.maximum(self.data, other.data), (self, other), Operation.MAX)
        return out

    def tanh(self) -> Tensor:
        """Compute hyperbolic tangent of tensor.

        Uses a numerically stable implementation:
            tanh(x) = (e^x - e^-x)/(e^x + e^-x)
        For large x, we clamp the input to avoid overflow in float32
        """
        clamped = self.max(-100).max(-100)
        exp_pos = clamped.exp()
        exp_neg = (-clamped).exp()
        out = (exp_pos - exp_neg) / (exp_pos + exp_neg)
        out._op = Operation.TANH
        return out

    def sigmoid(self) -> Tensor:
        """Compute sigmoid activation function of tensor."""
        exp = (-1 * self).exp()
        num = 1
        den = 1 + exp
        out = num / den
        out._op = Operation.SIGMOID
        return out

    def relu(self) -> Tensor:
        """Compute ReLU activation function: max(0, x)."""
        out = self.max(0)
        out._op = Operation.RELU
        return out
