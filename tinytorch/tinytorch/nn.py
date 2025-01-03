import copy
from enum import Enum
from functools import reduce
from typing import Optional, Tuple

import numpy as np

from tinytorch.engine import List, Tensor, TensorLike

ShapeType = int | Tuple[int, ...]


class Activation(Enum):
    """Options for activation functions."""

    TANH = "tanh"
    SIGMOID = "sigmoid"
    RELU = "relu"
    LIN = "lin"


class Neuron(object):
    """Single neuron implementation."""

    def __init__(
        self,
        n_input: int,
        activation: Activation,
        label: Optional[str] = None,
    ) -> None:
        """Initialize a neuron with Xavier/Glorot initialization.

        The weights are initialized from a normal distribution with:
        - mean = 0
        - std = sqrt(2 / (n_input + 1))  # +1 for the output

        This helps prevent vanishing/exploding gradients and maintains
        similar variance of activations and gradients across layers.
        """
        self.label = label
        self.n_input = n_input
        self.activation = activation

        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (n_input + 1))  # +1 for the output
        self.w = Tensor(np.random.normal(0, scale, size=(1, n_input)), label="w")
        self.b = Tensor(np.zeros((1,)), label="b")  # Initialize bias to zero

    @property
    def parameters(self) -> List[Tensor]:
        """Access all tunable parameters."""
        return [self.w, self.b]

    def __call__(self, x: TensorLike) -> Tensor:
        """Implements the forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size,)

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size,) after applying weights and activation.

        Notes
        -----
        Steps:
            1. Multiply input with weights element-wise
            2. Sum over all dimensions except batch
            3. Add bias and apply activation
        """
        # Element-wise multiply and sum over all dimensions except batch
        a = (self.w * x).sum(axis=1) + self.b
        a.label = self.label or "a"
        act_function = getattr(a, self.activation.value)
        out = act_function()
        return out

    def __repr__(self) -> str:
        """Return detailed string representation of the neuron."""
        return (
            f"Neuron("
            f"n_input={self.n_input}"
            f"activation={self.activation.value}"
            f"{f', label={self.label}' if self.label else ''}"
            f")"
        )

    def __str__(self) -> str:
        """Return concise string representation of the neuron."""
        return f"Neuron({self.activation.value})"


class Layer(object):
    """Collection of neurons."""

    def __init__(
        self,
        n_input: int,
        n_neurons: int = 1,
        activation: Activation = Activation.RELU,
        label: Optional[str] = None,
    ) -> None:
        self.label = label
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.activation = activation
        self.neurons = [
            Neuron(n_input, activation, label=f"neuron_{i}") for i in range(0, n_neurons)
        ]

    def __repr__(self) -> str:
        """Return detailed string representation of the layer."""
        return (
            f"Layer("
            f"n_input={self.n_input}, "
            f"n_neurons={self.n_neurons}, "
            f"activation={self.activation.value}"
            f"{f', label={self.label}' if self.label else ''}"
            f")"
        )

    def __str__(self) -> str:
        """Return concise string representation of the layer."""
        return f"Layer({self.n_neurons} neurons, {self.activation.value})"

    def __call__(self, x: TensorLike) -> Tensor:
        """Forward pass through the layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size,)

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, n_neurons)
        """
        neuron_outputs = [neuron(x) for neuron in self.neurons]
        return Tensor.stack(neuron_outputs, axis=1)


class MLP(object):
    """Collection of layers forming a multi-layer perceptron."""

    def __init__(
        self,
        n_input: int,
        layers: List[Tuple[int, Activation]],
        label: Optional[str] = None,
    ) -> None:
        """Initialize MLP.

        Parameters
        ----------
        n_input : int
            Number of input features
        layers : List[Tuple[int, Activation]]
            List of (n_neurons, activation) pairs for each layer
        label : Optional[str]
            Optional label for visualization
        """
        self.label = label

        # Build list of consecutive pairs of dimensions for each layer
        dims = [n_input] + [n for n, _ in layers]  # [input_dim, hidden_dim, output_dim]
        activations = [Activation.LIN] + [act for _, act in layers]  # [lin, hidden_act, output_act]

        # Create layers using zip to pair consecutive dimensions
        self.layers = [
            Layer(n_input=n_in, n_neurons=n_out, activation=act)
            for (n_in, n_out, act) in zip(dims[:-1], dims[1:], activations[1:])
        ]

    def __call__(self, x: TensorLike) -> Tensor:
        """Forward pass through MLP using function composition."""
        return reduce(lambda xi, layer: layer(xi), self.layers, x)

    def __repr__(self) -> str:
        """Return detailed string representation of the MLP."""
        layers_str = ",\n    ".join(
            f"Layer({layer.n_input}->{layer.n_neurons}, {layer.activation.value})"
            for layer in self.layers
        )
        label_str = f',  label="{self.label}"' if self.label else ""
        return (
            "MLP(\n"
            + f"  input_dim={self.layers[0].n_input},\n"
            + f"  layers=[\n    {layers_str}\n  ]"
            + label_str
            + "\n"
            + ")"
        )

    def __str__(self) -> str:
        """Return concise string representation showing layer dimensions."""
        layers_str = " -> ".join(
            f"{layer.n_neurons}/{layer.activation.value}" for layer in self.layers
        )
        return f"MLP({self.layers[0].n_input} -> [{layers_str}])"
