# AutoGrad: A Minimal Automatic Differentiation Library

AutoGrad is a minimal implementation of automatic differentiation (autodiff) in Python, designed for educational purposes. It demonstrates the core concepts behind modern deep learning frameworks like [PyTorch](https://pytorch.org/) and [JAX](https://jax.readthedocs.io/en/latest/).

## Features

- **Tensor Operations**: Basic arithmetic operations (+, -, *, /) with automatic tracking of computational graphs
- **Neural Network Activations**: Common activation functions (ReLU, sigmoid, tanh)
- **Broadcasting**: Support for scalar operations and broadcasting
- **Type Safety**: Full type annotations and strict type checking
- **Property-Based Testing**: Comprehensive test suite using Hypothesis

## Core Concepts

The library implements forward-mode automatic differentiation through operator overloading. Key components:

- `Tensor`: Main class that wraps numpy arrays and tracks operations
- `Operation`: `Enum` of supported operations
- Computational Graph: Built automatically through operator overloading

## Example Usage

```python
from autograd import Tensor

# Create tensors
x = Tensor([1.0, 2.0, 3.0])
y = Tensor([4.0, 5.0, 6.0])

# Basic operations
z = x + y
w = z.tanh()

# Neural network operations
output = w.sigmoid()
```

## Testing

The library uses property-based testing with [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) to verify:
- Mathematical properties (commutativity, associativity)
- Numerical stability
- Type safety and error handling
- Compatibility with [NumPy](https://numpy.org/doc/) operations

## Development

- Python 3.11+
- [NumPy](https://numpy.org/doc/) for array operations
- [Ruff](https://docs.astral.sh/ruff/) for formatting and linting
- [MyPy](https://mypy.readthedocs.io/en/stable/) for static type checking
- [Pytest](https://docs.pytest.org/en/stable/) and [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) for testing
- [JAX](https://jax.readthedocs.io/en/latest/) for testing and benchmarking