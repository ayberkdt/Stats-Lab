"""Small fully connected networks evaluated with automatic differentiation.

A multilayer perceptron maps an input vector through affine layers and
elementwise activations [EQ-NN-MLP-001]. Parameters live in one flat vector,
so the same network can be evaluated on floats, forward-mode duals, or
reverse-mode tape variables: every operation goes through :data:`admath`,
which dispatches on the operand type. Training code differentiates a loss
with :func:`probstat.numerics.reverse_value_and_gradient` and updates the flat
vector with :func:`probstat.optimization.adam_minimize`.

The networks are intended for the low-dimensional density, score, and drift
models of the library (normalizing flows, score matching, neural SDEs), not
for large-scale learning: the arithmetic is scalar Python.

References:
    [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Chapter 6] and
    [REF-GLOROT-BENGIO-2010].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..foundations.random_streams import RandomStream
from .autodiff import admath

ACTIVATIONS: tuple[str, ...] = ("tanh", "sigmoid", "softplus", "identity")


def _activate(name: str, value: Any) -> Any:
    """Apply the named activation to one scalar (float, dual, or tape variable).

    ``softplus`` uses ``max(x, 0) + log1p(exp(-|x|))`` so it never overflows;
    ``sigmoid`` uses ``(1 + tanh(x / 2)) / 2``.

    Args:
        name: One of :data:`ACTIVATIONS`.
        value: Pre-activation.

    Returns:
        The activation of ``value``, of the same numeric kind.

    References:
        [EQ-NN-MLP-001], [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Section 6.3].

    """
    if name == "tanh":
        return admath.tanh(value)
    if name == "sigmoid":
        return 0.5 * (1.0 + admath.tanh(0.5 * value))
    if name == "softplus":
        if value > 0:
            return value + admath.log1p(admath.exp(-value))
        return admath.log1p(admath.exp(value))
    return value


@dataclass(frozen=True)
class MLP:
    """Multilayer perceptron with a flat parameter vector.

    ``layer_sizes = (d_in, h_1, ..., h_k, d_out)`` defines ``k + 1`` affine
    maps. Hidden layers use ``activation``; the output layer uses
    ``output_activation`` (``"identity"`` by default). Parameters are stored
    layer by layer, weights row-major (``out x in``) followed by biases.

    Attributes:
        layer_sizes: Widths from input to output (at least two entries).
        activation: Hidden-layer activation, one of :data:`ACTIVATIONS`.
        output_activation: Output activation, one of :data:`ACTIVATIONS`.

    References:
        [EQ-NN-MLP-001], [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Chapter 6].

    """

    layer_sizes: tuple[int, ...]
    activation: str = "tanh"
    output_activation: str = "identity"

    def __post_init__(self) -> None:
        """Validate the architecture.

        Raises:
            ValueError: For fewer than two layers, a non-positive width, or
                an unknown activation.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        sizes = tuple(self.layer_sizes)
        if len(sizes) < 2:
            raise ValueError("layer_sizes needs an input and an output width")
        for size in sizes:
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError("layer widths must be positive integers")
        for name in (self.activation, self.output_activation):
            if name not in ACTIVATIONS:
                raise ValueError(f"activation must be one of {ACTIVATIONS}")
        object.__setattr__(self, "layer_sizes", sizes)

    @property
    def input_dimension(self) -> int:
        """Return the input width.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self.layer_sizes[0]

    @property
    def output_dimension(self) -> int:
        """Return the output width.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self.layer_sizes[-1]

    @property
    def parameter_count(self) -> int:
        """Return the number of weights and biases.

        References:
            [EQ-NN-MLP-001]. Counting only.

        """
        return sum(
            (fan_in + 1) * fan_out
            for fan_in, fan_out in zip(
                self.layer_sizes[:-1], self.layer_sizes[1:], strict=True
            )
        )

    def initial_parameters(
        self, seed: int | RandomStream, *, output_scale: float = 1.0
    ) -> tuple[float, ...]:
        """Draw Glorot-uniform weights and zero biases.

        Weights of a layer with fan-in ``a`` and fan-out ``b`` are uniform on
        ``[-r, r]`` with ``r = sqrt(6 / (a + b))`` [EQ-NN-GLOROT-001]; the
        output layer is multiplied by ``output_scale`` (``0`` starts a
        residual block at the identity).

        Args:
            seed: Integer seed or a :class:`RandomStream`.
            output_scale: Non-negative factor on the last layer's weights.

        Returns:
            Flat parameter vector of length :attr:`parameter_count`.

        Raises:
            ValueError: If ``output_scale`` is negative or not finite.

        References:
            [EQ-NN-GLOROT-001], [REF-GLOROT-BENGIO-2010, Equation 16].

        """
        if not math.isfinite(output_scale) or output_scale < 0.0:
            raise ValueError("output_scale must be finite and non-negative")
        stream = seed if isinstance(seed, RandomStream) else RandomStream(seed)
        generator = stream.child("mlp-init").generator()
        parameters: list[float] = []
        last = len(self.layer_sizes) - 2
        for layer, (fan_in, fan_out) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=True)
        ):
            radius = math.sqrt(6.0 / (fan_in + fan_out))
            scale = output_scale if layer == last else 1.0
            parameters.extend(
                scale * generator.uniform(-radius, radius)
                for _ in range(fan_in * fan_out)
            )
            parameters.extend(0.0 for _ in range(fan_out))
        return tuple(parameters)

    def forward(self, parameters: Sequence[Any], inputs: Sequence[Any]) -> list[Any]:
        """Evaluate the network at one input vector.

        Args:
            parameters: Flat vector of length :attr:`parameter_count` (floats,
                duals, or reverse-mode variables).
            inputs: Input vector of length :attr:`input_dimension`.

        Returns:
            Output vector of length :attr:`output_dimension`.

        Raises:
            ValueError: If a length does not match the architecture.

        References:
            [EQ-NN-MLP-001].

        """
        if len(parameters) != self.parameter_count:
            raise ValueError(
                f"expected {self.parameter_count} parameters, got {len(parameters)}"
            )
        if len(inputs) != self.input_dimension:
            raise ValueError(
                f"expected {self.input_dimension} inputs, got {len(inputs)}"
            )
        values: list[Any] = list(inputs)
        offset = 0
        last = len(self.layer_sizes) - 2
        for layer, (fan_in, fan_out) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=True)
        ):
            weights_end = offset + fan_in * fan_out
            biases = parameters[weights_end : weights_end + fan_out]
            name = self.output_activation if layer == last else self.activation
            outputs: list[Any] = []
            for row in range(fan_out):
                start = offset + row * fan_in
                total: Any = biases[row]
                for column in range(fan_in):
                    total = total + parameters[start + column] * values[column]
                outputs.append(_activate(name, total))
            values = outputs
            offset = weights_end + fan_out
        return values
