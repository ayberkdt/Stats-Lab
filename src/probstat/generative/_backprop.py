"""Float-only forward pass, backpropagation, and input divergence of an MLP.

:func:`probstat.numerics.reverse_value_and_gradient` records every scalar
operation of a network on a tape, which is general but slow. The squared
error and cross-entropy losses of flow matching and ratio estimation need
only the classical layer-by-layer recursion [EQ-GEN-FM-005], written here for
plain floats and a ``tanh`` hidden activation with an identity output. The
parameter layout is that of :class:`probstat.numerics.MLP` (per layer:
weights row-major, then biases), so the same flat vector can be checked
against the tape (the tests do).

References:
    [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Section 6.5] and
    [REF-GRIEWANK-WALTHER-2008, Chapter 3].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from operator import mul

Layer = tuple[list[Sequence[float]], Sequence[float]]
OutputLoss = Callable[[int, list[float]], tuple[float, Sequence[float]]]


def split_layers(sizes: Sequence[int], parameters: Sequence[float]) -> list[Layer]:
    """Split a flat parameter vector into per-layer weight rows and biases.

    Args:
        sizes: Layer widths ``(d_in, h_1, ..., d_out)``.
        parameters: Flat vector in the layout of
            :class:`probstat.numerics.MLP`.

    Returns:
        One ``(rows, biases)`` pair per affine layer.

    Raises:
        ValueError: If the vector length does not match the widths.

    References:
        [EQ-NN-MLP-001].

    """
    expected = sum((a + 1) * b for a, b in zip(sizes[:-1], sizes[1:], strict=True))
    if len(parameters) != expected:
        raise ValueError(f"expected {expected} parameters, got {len(parameters)}")
    layers: list[Layer] = []
    offset = 0
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
        rows = [
            parameters[offset + r * fan_in : offset + (r + 1) * fan_in]
            for r in range(fan_out)
        ]
        offset += fan_in * fan_out
        layers.append((rows, parameters[offset : offset + fan_out]))
        offset += fan_out
    return layers


def forward(layers: Sequence[Layer], inputs: Sequence[float]) -> list[float]:
    """Evaluate a ``tanh`` network with an identity output on floats.

    Args:
        layers: Output of :func:`split_layers`.
        inputs: Input vector.

    Returns:
        The output vector.

    References:
        [EQ-NN-MLP-001].

    """
    values: Sequence[float] = inputs
    last = len(layers) - 1
    for index, (rows, biases) in enumerate(layers):
        affine = [
            b + sum(map(mul, row, values)) for row, b in zip(rows, biases, strict=True)
        ]
        values = affine if index == last else list(map(math.tanh, affine))
    return list(values)


def forward_with_divergence(
    layers: Sequence[Layer], inputs: Sequence[float], count: int
) -> tuple[list[float], float]:
    """Return the output and ``sum_{k < count} d out_k / d in_k``.

    One tangent vector per differentiated input is pushed through the layers
    together with the values (forward-mode differentiation written out by
    hand), so the divergence is exact up to rounding.

    Args:
        layers: Output of :func:`split_layers`.
        inputs: Input vector.
        count: Number of leading inputs (and outputs) in the divergence.

    Returns:
        ``(output, divergence)``.

    References:
        [EQ-GEN-FM-004], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    values: Sequence[float] = inputs
    width = len(inputs)
    tangents: list[list[float]] = [
        [1.0 if j == k else 0.0 for j in range(width)] for k in range(count)
    ]
    last = len(layers) - 1
    for index, (rows, biases) in enumerate(layers):
        affine = [
            b + sum(map(mul, row, values)) for row, b in zip(rows, biases, strict=True)
        ]
        pushed = [[sum(map(mul, row, tangent)) for row in rows] for tangent in tangents]
        if index == last:
            values = affine
            tangents = pushed
        else:
            values = list(map(math.tanh, affine))
            slopes = [1.0 - a * a for a in values]
            tangents = [list(map(mul, slopes, tangent)) for tangent in pushed]
    return list(values), math.fsum(tangents[k][k] for k in range(count))


def batch_loss_and_gradient(
    sizes: Sequence[int],
    parameters: Sequence[float],
    batch: Sequence[Sequence[float]],
    output_loss: OutputLoss,
) -> tuple[float, list[float]]:
    """Return the mean loss over a batch and its parameter gradient.

    For every input the network is evaluated, ``output_loss(i, output)``
    returns the loss of case ``i`` and its gradient with respect to the
    output, and the recursion ``delta_{l-1} = (W_l' delta_l) * (1 - a^2)``
    carries that gradient back to every weight and bias [EQ-GEN-FM-005].

    Args:
        sizes: Layer widths.
        parameters: Flat parameter vector.
        batch: Input vectors.
        output_loss: ``(case index, output) -> (loss, d loss / d output)``.

    Returns:
        ``(mean loss, gradient)`` with the gradient in the flat layout.

    Raises:
        ValueError: If the parameter vector has the wrong length.

    References:
        [EQ-GEN-FM-005], [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Section 6.5].

    """
    layers = split_layers(sizes, parameters)
    transposed = [list(zip(*rows, strict=True)) for rows, _ in layers]
    weight_gradients = [[[0.0] * len(rows[0]) for _ in rows] for rows, _ in layers]
    bias_gradients = [[0.0] * len(biases) for _, biases in layers]
    last = len(layers) - 1
    total = 0.0
    for case, inputs in enumerate(batch):
        activations: list[Sequence[float]] = [inputs]
        values: Sequence[float] = inputs
        for index, (rows, biases) in enumerate(layers):
            affine = [
                b + sum(map(mul, row, values))
                for row, b in zip(rows, biases, strict=True)
            ]
            values = affine if index == last else list(map(math.tanh, affine))
            activations.append(values)
        loss, delta = output_loss(case, list(values))
        total += loss
        for index in range(last, -1, -1):
            below = activations[index]
            layer_gradient = weight_gradients[index]
            bias_gradient = bias_gradients[index]
            for r, d in enumerate(delta):
                if d != 0.0:
                    row_gradient = layer_gradient[r]
                    layer_gradient[r] = [
                        g + d * v for g, v in zip(row_gradient, below, strict=True)
                    ]
                    bias_gradient[r] += d
            if index > 0:
                delta = [
                    sum(map(mul, column, delta)) * (1.0 - a * a)
                    for column, a in zip(transposed[index], below, strict=True)
                ]
    scale = 1.0 / len(batch)
    gradient: list[float] = []
    for layer_gradient, bias_gradient in zip(
        weight_gradients, bias_gradients, strict=True
    ):
        for row_gradient in layer_gradient:
            gradient.extend(scale * g for g in row_gradient)
        gradient.extend(scale * g for g in bias_gradient)
    return scale * total, gradient
