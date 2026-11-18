"""Deep-ensemble neural-network surrogates with predictive uncertainty.

A deep ensemble trains ``M`` small multilayer perceptrons on the same data
from different random initializations (optionally on bootstrap resamples).
Each member outputs a Gaussian law ``N(mu_m(x), sigma_m^2(x))`` and is fitted
by minimizing the Gaussian negative log-likelihood, a proper scoring rule
[EQ-SUR-DE-001]. The ensemble predicts the uniformly weighted mixture of the
member laws: its mean is the average member mean and its variance splits
exactly into an aleatoric part (the average member variance, the noise the
members attribute to the data) and an epistemic part (the spread of the
member means, which grows where the data do not constrain the networks)
[EQ-SUR-DE-002].

Inputs are standardized per coordinate and the output is standardized
before training; predictions and variances are reported in original units.
Gradients come from a closed-form backward pass over a whole mini-batch in
plain floats (the same parameter layout as :class:`probstat.numerics.MLP`, so
the result can be checked against reverse-mode differentiation of
``MLP.forward``), and the parameters are updated by
:func:`probstat.optimization.adam_minimize`. The arithmetic is scalar Python:
the module targets designs of tens to a few hundred runs and inputs of up to
a few tens of dimensions, where a Gaussian process with ``O(n^3)``
likelihood evaluations per optimizer step becomes slow. Measured costs are
listed in ``docs/foundations/surrogate-models.md``.

References:
    [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017], [REF-KINGMA-BA-2015],
    [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Chapters 6 and 7].

"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from operator import mul
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.modeling import ProbabilisticModel
from probstat.numerics import ACTIVATIONS, MLP
from probstat.optimization import adam_minimize
from probstat.propagation import make_design

_LOG_TWO_PI = math.log(2.0 * math.pi)
_LOG_VARIANCE_BOUND = 20.0
_VARIANCE_FLOOR = 1e-6

Layers = list[tuple[list[list[float]], list[float]]]


def _dot_fallback(left: Iterable[float], right: Iterable[float]) -> float:
    """Return the dot product of two equal-length vectors (Python < 3.12).

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        ``sum_i left_i right_i``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. Elementary sum of products.

    """
    return float(sum(map(mul, left, right)))


# ``math.sumprod`` (Python 3.12+) is about three times faster than the
# generator form and accumulates in extended precision.
_dot: Callable[[Iterable[float], Iterable[float]], float] = getattr(
    math, "sumprod", _dot_fallback
)


def _activation_value(name: str, z: float) -> float:
    """Apply a hidden activation to one float, matching ``MLP.forward``.

    Args:
        name: One of :data:`probstat.numerics.ACTIVATIONS`.
        z: Pre-activation.

    Returns:
        The activation value.

    Equation:
        [EQ-NN-MLP-001].

    References:
        [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Section 6.3].

    """
    if name == "tanh":
        return math.tanh(z)
    if name == "sigmoid":
        return 0.5 * (1.0 + math.tanh(0.5 * z))
    if name == "softplus":
        if z > 0.0:
            return z + math.log1p(math.exp(-z))
        return math.log1p(math.exp(z))
    return z


def _activation_slope(name: str, z: float, a: float) -> float:
    """Return the derivative of a hidden activation at ``z`` (value ``a``).

    Args:
        name: Activation name.
        z: Pre-activation.
        a: Activation value at ``z``.

    Returns:
        ``d a / d z``.

    Equation:
        [EQ-NN-MLP-001].

    References:
        [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Section 6.5].

    """
    if name == "tanh":
        return 1.0 - a * a
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "softplus":
        return 0.5 * (1.0 + math.tanh(0.5 * z))
    return 1.0


def _unpack(network: MLP, parameters: Sequence[float]) -> Layers:
    """Split a flat MLP parameter vector into per-layer weights and biases.

    Args:
        network: Architecture (weights row-major ``out x in``, then biases).
        parameters: Flat vector of at least :attr:`MLP.parameter_count`.

    Returns:
        List of ``(weights, biases)`` per affine layer.

    References:
        [EQ-NN-MLP-001]. Bookkeeping only.

    """
    layers: Layers = []
    offset = 0
    sizes = network.layer_sizes
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
        weights = [
            list(parameters[offset + row * fan_in : offset + (row + 1) * fan_in])
            for row in range(fan_out)
        ]
        offset += fan_in * fan_out
        biases = list(parameters[offset : offset + fan_out])
        offset += fan_out
        layers.append((weights, biases))
    return layers


def _forward(
    layers: Layers, activation: str, inputs: Sequence[Sequence[float]]
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Propagate a batch of inputs through the network.

    Args:
        layers: Output of :func:`_unpack`.
        activation: Hidden activation name (the output layer is linear).
        inputs: Batch of standardized input vectors.

    Returns:
        ``(activations, pre_activations)``: ``activations[l]`` is the batch
        entering layer ``l`` (``activations[0]`` the inputs, the last entry
        the network outputs) and ``pre_activations[l]`` the hidden
        pre-activations of layer ``l``.

    Equation:
        [EQ-NN-MLP-001].

    References:
        [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Section 6.5].

    """
    current = [list(point) for point in inputs]
    activations = [current]
    pre_activations: list[list[list[float]]] = []
    last = len(layers) - 1
    for index, (weights, biases) in enumerate(layers):
        z_batch = [
            [bias + _dot(row, point) for row, bias in zip(weights, biases, strict=True)]
            for point in current
        ]
        if index == last:
            current = z_batch
        else:
            pre_activations.append(z_batch)
            if activation == "tanh":
                current = [list(map(math.tanh, zs)) for zs in z_batch]
            else:
                current = [
                    [_activation_value(activation, z) for z in zs] for zs in z_batch
                ]
        activations.append(current)
    return activations, pre_activations


def _gaussian_terms(
    target: float, mean: float, raw_log_variance: float
) -> tuple[float, float, float, float]:
    """Return one Gaussian NLL term, its two derivatives, and the variance.

    ``sigma^2 = exp(clip(s, -20, 20)) + 1e-6`` (standardized units) and
    ``l = (log(2 pi sigma^2) + (y - mu)^2 / sigma^2) / 2``; the derivative
    with respect to ``s`` is zero outside the clip interval.

    Args:
        target: Standardized observation ``y``.
        mean: Member mean ``mu``.
        raw_log_variance: Network log-variance output ``s``.

    Returns:
        ``(l, dl/dmu, dl/ds, sigma^2)``.

    Equation:
        [EQ-SUR-DE-001].

    References:
        [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Section 2.2.1].

    """
    clipped = min(max(raw_log_variance, -_LOG_VARIANCE_BOUND), _LOG_VARIANCE_BOUND)
    exponential = math.exp(clipped)
    variance = exponential + _VARIANCE_FLOOR
    residual = target - mean
    ratio = residual * residual / variance
    loss = 0.5 * (_LOG_TWO_PI + math.log(variance) + ratio)
    d_mean = -residual / variance
    if abs(raw_log_variance) < _LOG_VARIANCE_BOUND:
        d_log = 0.5 * exponential / variance * (1.0 - ratio)
    else:
        d_log = 0.0
    return loss, d_mean, d_log, variance


def _member_loss_and_gradient(
    network: MLP,
    heteroscedastic: bool,
    weight_decay: float,
    weight_mask: Sequence[bool],
    parameters: Sequence[float],
    inputs: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    fixed_variance: bool = False,
) -> tuple[float, list[float]]:
    """Return the regularized mean Gaussian NLL of a batch and its gradient.

    The loss is ``mean_b l_b + (lambda / 2) sum w^2`` over the weights (not
    the biases or the homoscedastic log-variance), with ``l_b`` from
    :func:`_gaussian_terms`. The gradient is the closed-form backward pass of
    the network.

    Args:
        network: Member architecture.
        heteroscedastic: Whether the second network output is the
            log-variance; otherwise the last parameter is a shared one.
        weight_decay: ``lambda >= 0``.
        weight_mask: Flags marking the weight entries of ``parameters``.
        parameters: Member parameters.
        inputs: Standardized batch inputs.
        targets: Standardized batch outputs.
        fixed_variance: Hold the log-variance at zero (unit variance, so the
            loss is half the mean squared error plus a constant) and give the
            variance head a zero gradient; used for the mean warm-up.

    Returns:
        ``(loss, gradient)``.

    Equation:
        [EQ-SUR-DE-001], [EQ-NN-MLP-001].

    References:
        [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Section 2.2.1],
        [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Sections 6.5 and 7.1.1].

    """
    count = len(targets)
    layers = _unpack(network, parameters)
    activations, pre_activations = _forward(layers, network.activation, inputs)
    outputs = activations[-1]
    shared_log = parameters[-1] if not heteroscedastic else 0.0
    total = 0.0
    shared_gradient = 0.0
    delta: list[list[float]] = []
    for output, target in zip(outputs, targets, strict=True):
        if fixed_variance:
            raw = 0.0
        else:
            raw = output[1] if heteroscedastic else shared_log
        loss, d_mean, d_log, _ = _gaussian_terms(target, output[0], raw)
        if fixed_variance:
            d_log = 0.0
        total += loss
        if heteroscedastic:
            delta.append([d_mean / count, d_log / count])
        else:
            delta.append([d_mean / count])
            shared_gradient += d_log / count
    pieces: list[list[float]] = []
    for index in range(len(layers) - 1, -1, -1):
        weights, _ = layers[index]
        incoming = activations[index]
        delta_columns = list(zip(*delta, strict=True))
        input_columns = list(zip(*incoming, strict=True))
        weight_gradient = [
            _dot(column, feature)
            for column in delta_columns
            for feature in input_columns
        ]
        bias_gradient = [math.fsum(column) for column in delta_columns]
        pieces.append(weight_gradient + bias_gradient)
        if index > 0:
            transposed = list(zip(*weights, strict=True))
            pre = pre_activations[index - 1]
            name = network.activation
            if name == "tanh":
                slopes = [[1.0 - a * a for a in a_row] for a_row in incoming]
            else:
                slopes = [
                    [
                        _activation_slope(name, z, a)
                        for z, a in zip(z_row, a_row, strict=True)
                    ]
                    for z_row, a_row in zip(pre, incoming, strict=True)
                ]
            delta = [
                [
                    _dot(row_delta, column) * slope
                    for column, slope in zip(transposed, slope_row, strict=True)
                ]
                for row_delta, slope_row in zip(delta, slopes, strict=True)
            ]
    gradient: list[float] = []
    for piece in reversed(pieces):
        gradient.extend(piece)
    if not heteroscedastic:
        gradient.append(shared_gradient)
    penalty = 0.0
    if weight_decay > 0.0:
        penalty = math.fsum(
            value * value
            for value, is_weight in zip(parameters, weight_mask, strict=True)
            if is_weight
        )
        gradient = [
            g + weight_decay * value if is_weight else g
            for g, value, is_weight in zip(
                gradient, parameters, weight_mask, strict=True
            )
        ]
    return total / count + 0.5 * weight_decay * penalty, gradient


def _weight_mask(network: MLP, heteroscedastic: bool) -> tuple[bool, ...]:
    """Flag the weight entries of a member parameter vector.

    Args:
        network: Member architecture.
        heteroscedastic: Whether a shared log-variance follows the MLP block.

    Returns:
        Tuple of flags, ``True`` for a weight and ``False`` for a bias or the
        shared log-variance.

    References:
        [EQ-NN-MLP-001]. Bookkeeping only.

    """
    mask: list[bool] = []
    sizes = network.layer_sizes
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
        mask.extend([True] * (fan_in * fan_out))
        mask.extend([False] * fan_out)
    if not heteroscedastic:
        mask.append(False)
    return tuple(mask)


def _positive_int(value: object, *, name: str) -> int:
    """Validate a positive integer that is not a bool.

    Args:
        value: Candidate.
        name: Argument name for the message.

    Returns:
        The integer.

    Raises:
        ValueError: If ``value`` is not a positive integer.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _standardization(
    columns: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    """Return per-column sample means and standard deviations (zero -> one).

    Args:
        columns: Data columns.

    Returns:
        ``(means, scales)``.

    References:
        [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Section 8.7.1].
        Sample moments only.

    """
    means: list[float] = []
    scales: list[float] = []
    for column in columns:
        mean = math.fsum(column) / len(column)
        variance = math.fsum((v - mean) ** 2 for v in column) / (len(column) - 1)
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale > 0.0 else 1.0)
    return means, scales


@dataclass(frozen=True)
class DeepEnsemblePrediction:
    """Mixture prediction of a deep ensemble at one input, in original units.

    Args:
        mean: Mixture mean, the average member mean.
        variance: Mixture variance, ``aleatoric_variance +
            epistemic_variance``.
        aleatoric_variance: Average member variance.
        epistemic_variance: Population variance of the member means.
        member_means: Member means.
        member_variances: Member variances.

    References:
        [EQ-SUR-DE-002], [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017].

    """

    mean: float
    variance: float
    aleatoric_variance: float
    epistemic_variance: float
    member_means: tuple[float, ...]
    member_variances: tuple[float, ...]


@dataclass(frozen=True)
class DeepEnsemble:
    """A fitted deep ensemble of Gaussian mean-variance networks.

    Args:
        network: Shared member architecture (input width ``d``; output width
            2 when heteroscedastic, otherwise 1).
        members: Trained parameter vectors, one per member; a homoscedastic
            member carries its shared log-variance as the last entry.
        heteroscedastic: Whether each member predicts an input-dependent
            variance.
        points: Training inputs (original units).
        values: Training outputs (original units).
        input_mean: Per-coordinate training input means.
        input_scale: Per-coordinate training input standard deviations
            (one for a constant coordinate).
        output_mean: Training output mean.
        output_scale: Training output standard deviation (one when
            constant).
        member_losses: Final Gaussian NLL of each member on its full
            training set, standardized units, without weight decay.
        loss_traces: Mini-batch loss (with weight decay) at every Adam step,
            one trace per member.
        training_rmse: Root-mean-square error of the ensemble mean on the
            training data, original units.
        iterations: Adam steps per member.
        batch_size: Mini-batch size used (capped at the training size).
        learning_rate: Adam step size.
        weight_decay: Weight-decay coefficient ``lambda``.
        tail_fraction: Fraction of final Adam iterates averaged.
        bootstrap: Whether each member saw a bootstrap resample.
        warmup_fraction: Fraction of the iterations spent on the mean with
            the variance held fixed.
        training_seconds: Wall-clock training time (not compared).

    References:
        [EQ-SUR-DE-001], [EQ-SUR-DE-002],
        [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017].

    """

    network: MLP
    members: tuple[tuple[float, ...], ...]
    heteroscedastic: bool
    points: tuple[tuple[float, ...], ...]
    values: tuple[float, ...]
    input_mean: tuple[float, ...]
    input_scale: tuple[float, ...]
    output_mean: float
    output_scale: float
    member_losses: tuple[float, ...]
    loss_traces: tuple[tuple[float, ...], ...]
    training_rmse: float
    iterations: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    tail_fraction: float
    bootstrap: bool
    warmup_fraction: float = 0.0
    training_seconds: float = field(default=0.0, compare=False)
    _layers: tuple[Layers, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Unpack the member parameters once for fast prediction.

        Raises:
            ValueError: If there are no members or a member has the wrong
                number of parameters.

        References:
            [EQ-NN-MLP-001]. Bookkeeping only.

        """
        if not self.members:
            raise ValueError("a deep ensemble needs at least one member")
        expected = self.network.parameter_count + (0 if self.heteroscedastic else 1)
        for parameters in self.members:
            if len(parameters) != expected:
                raise ValueError(f"each member needs {expected} parameters")
        object.__setattr__(
            self,
            "_layers",
            tuple(_unpack(self.network, parameters) for parameters in self.members),
        )

    @property
    def size(self) -> int:
        """Return the number of ensemble members.

        References:
            [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017]. Counting only.

        """
        return len(self.members)

    @property
    def dimension(self) -> int:
        """Return the input dimension.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self.network.input_dimension

    def member_predictions(self, x: Sequence[float]) -> tuple[tuple[float, float], ...]:
        """Return each member's mean and variance at one input (original units).

        Args:
            x: Input vector of length :attr:`dimension`.

        Returns:
            One ``(mu_m, sigma_m^2)`` pair per member.

        Raises:
            ValueError: If ``x`` has the wrong length or is not finite.

        Equation:
            [EQ-SUR-DE-001].

        References:
            [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Section 2.2.1].

        """
        point = [float(value) for value in x]
        if len(point) != self.dimension:
            raise ValueError("x must have the training input dimension")
        if not all(math.isfinite(value) for value in point):
            raise ValueError("x must be finite")
        scaled = [
            (value - mean) / scale
            for value, mean, scale in zip(
                point, self.input_mean, self.input_scale, strict=True
            )
        ]
        pairs: list[tuple[float, float]] = []
        for parameters, layers in zip(self.members, self._layers, strict=True):
            activations, _ = _forward(layers, self.network.activation, [scaled])
            output = activations[-1][0]
            raw = output[1] if self.heteroscedastic else parameters[-1]
            clipped = min(max(raw, -_LOG_VARIANCE_BOUND), _LOG_VARIANCE_BOUND)
            variance = math.exp(clipped) + _VARIANCE_FLOOR
            pairs.append(
                (
                    self.output_mean + self.output_scale * output[0],
                    self.output_scale**2 * variance,
                )
            )
        return tuple(pairs)

    def predict_decomposed(self, x: Sequence[float]) -> DeepEnsemblePrediction:
        """Return the mixture mean and the split of its variance.

        ``mu* = mean_m mu_m``; ``sigma*^2 = mean_m (sigma_m^2 + mu_m^2) -
        mu*^2 = mean_m sigma_m^2 + mean_m (mu_m - mu*)^2``.

        Args:
            x: Input vector.

        Returns:
            :class:`DeepEnsemblePrediction` in original units.

        Raises:
            ValueError: If ``x`` has the wrong length or is not finite.

        Equation:
            [EQ-SUR-DE-002].

        References:
            [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Section 2.4].

        """
        pairs = self.member_predictions(x)
        means = tuple(pair[0] for pair in pairs)
        variances = tuple(pair[1] for pair in pairs)
        count = len(pairs)
        mean = math.fsum(means) / count
        aleatoric = math.fsum(variances) / count
        epistemic = math.fsum((value - mean) ** 2 for value in means) / count
        return DeepEnsemblePrediction(
            mean=mean,
            variance=aleatoric + epistemic,
            aleatoric_variance=aleatoric,
            epistemic_variance=epistemic,
            member_means=means,
            member_variances=variances,
        )

    def predict_with_variance(
        self, x: Sequence[float], *, include_aleatoric: bool = True
    ) -> tuple[float, float]:
        """Return the mixture mean and predictive variance at one input.

        Args:
            x: Input vector.
            include_aleatoric: Return the full mixture variance (default);
                ``False`` returns only the epistemic part, the analogue of a
                GP latent-function variance for a deterministic simulator.

        Returns:
            ``(mean, variance)`` in original units.

        Raises:
            ValueError: If ``x`` has the wrong length or is not finite.

        Equation:
            [EQ-SUR-DE-002].

        References:
            [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Section 2.4].

        """
        prediction = self.predict_decomposed(x)
        if include_aleatoric:
            return prediction.mean, prediction.variance
        return prediction.mean, prediction.epistemic_variance

    def predict(self, x: Sequence[float]) -> float:
        """Return the mixture mean at one input.

        Args:
            x: Input vector.

        Returns:
            Average member mean in original units.

        Raises:
            ValueError: If ``x`` has the wrong length or is not finite.

        Equation:
            [EQ-SUR-DE-002].

        References:
            [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Section 2.4].

        """
        return self.predict_decomposed(x).mean

    def epistemic_variance(self, x: Sequence[float]) -> float:
        """Return the spread of the member means at one input.

        Args:
            x: Input vector.

        Returns:
            ``mean_m (mu_m - mu*)^2`` in original units.

        Raises:
            ValueError: If ``x`` has the wrong length or is not finite.

        Equation:
            [EQ-SUR-DE-002].

        References:
            [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Section 2.4].

        """
        return self.predict_decomposed(x).epistemic_variance

    def aleatoric_variance(self, x: Sequence[float]) -> float:
        """Return the average member variance at one input.

        Args:
            x: Input vector.

        Returns:
            ``mean_m sigma_m^2`` in original units.

        Raises:
            ValueError: If ``x`` has the wrong length or is not finite.

        Equation:
            [EQ-SUR-DE-002].

        References:
            [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Section 2.4].

        """
        return self.predict_decomposed(x).aleatoric_variance


def _validated_data(
    points: object, values: object
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Validate training inputs and outputs.

    Args:
        points: Sequence of equal-length finite input vectors.
        values: Sequence of finite outputs, one per point.

    Returns:
        ``(points, values)`` as float tuples.

    Raises:
        ValueError: For fewer than two points, ragged or empty vectors,
            non-finite entries, or mismatched lengths.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if not isinstance(points, Sequence) or isinstance(points, str):
        raise ValueError("points must be a sequence of input vectors")
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise ValueError("values must be a sequence of numbers")
    rows: list[tuple[float, ...]] = []
    for point in points:
        if not isinstance(point, Sequence) or isinstance(point, str):
            raise ValueError("each point must be a sequence of numbers")
        rows.append(tuple(float(value) for value in point))
    if len(rows) < 2:
        raise ValueError("at least two training points are required")
    dimension = len(rows[0])
    if dimension == 0 or any(len(row) != dimension for row in rows):
        raise ValueError("points must be non-empty vectors of equal length")
    outputs = tuple(float(value) for value in values)
    if len(outputs) != len(rows):
        raise ValueError("values must have one entry per point")
    if not all(math.isfinite(v) for row in rows for v in row) or not all(
        math.isfinite(v) for v in outputs
    ):
        raise ValueError("points and values must be finite")
    return tuple(rows), outputs


def fit_deep_ensemble(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    members: int = 5,
    hidden: Sequence[int] = (32, 32),
    activation: str = "tanh",
    heteroscedastic: bool = True,
    iterations: int = 2000,
    learning_rate: float = 0.005,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    tail_fraction: float = 0.2,
    warmup_fraction: float = 0.3,
    bootstrap: bool = False,
    seed: int | RandomStream | None = None,
) -> DeepEnsemble:
    """Fit a deep ensemble of Gaussian mean-variance networks.

    Every member is an MLP ``(d, *hidden, 2)`` whose outputs are the mean and
    the log-variance of a Gaussian (``(d, *hidden, 1)`` plus one shared
    log-variance parameter when ``heteroscedastic=False``). Members start
    from independent Glorot initializations, optionally see a bootstrap
    resample of the data, and take ``iterations`` Adam steps on mini-batches
    of the regularized Gaussian negative log-likelihood [EQ-SUR-DE-001] in
    standardized units. Prediction is the uniform mixture [EQ-SUR-DE-002].

    Args:
        points: Training inputs (at least two equal-length vectors).
        values: Training outputs, one per point.
        members: Number of networks ``M``.
        hidden: Hidden-layer widths (may be empty for a linear model).
        activation: Hidden activation, one of
            :data:`probstat.numerics.ACTIVATIONS`.
        heteroscedastic: Predict an input-dependent variance; otherwise one
            noise variance per member.
        iterations: Adam steps per member.
        learning_rate: Adam step size.
        batch_size: Mini-batch size, capped at the number of points.
        weight_decay: Non-negative coefficient ``lambda`` of the penalty
            ``(lambda / 2) sum w^2`` on the weights.
        tail_fraction: Fraction in ``[0, 1)`` of the final Adam iterates
            averaged into each member's parameters (Polyak-Ruppert); the
            average damps the mini-batch noise of the last iterate, which
            otherwise can leave a member overconfident at some training
            points. ``0`` keeps the last iterate.
        warmup_fraction: Fraction in ``[0, 1)`` of the iterations spent
            first on the mean alone, with the variance held at one
            (standardized units), before the full likelihood is minimized by
            a fresh Adam run. The likelihood alone lets the variance absorb
            the misfit early and leaves the mean underfitted; on the 1-D
            benchmark of the guide the warm-up cut the validation error
            about tenfold. ``0`` is the original procedure.
        bootstrap: Train each member on a bootstrap resample (the original
            method uses the full data; resampling usually hurts accuracy).
        seed: Root seed or stream; member ``m`` uses the children
            ``("member", m)``. ``None`` is not reproducible.

    Returns:
        Fitted :class:`DeepEnsemble`.

    Raises:
        ValueError: For invalid data or settings, or a non-finite loss
            during training.

    Equation:
        [EQ-SUR-DE-001], [EQ-SUR-DE-002], [EQ-OPT-ADAM-002].

    References:
        [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Algorithm 1],
        [REF-KINGMA-BA-2015, Algorithm 1].

    """
    data, outputs = _validated_data(points, values)
    count_members = _positive_int(members, name="members")
    widths = tuple(hidden)
    for width in widths:
        _positive_int(width, name="hidden widths")
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}")
    if not isinstance(heteroscedastic, bool):
        raise ValueError("heteroscedastic must be a bool")
    if not isinstance(bootstrap, bool):
        raise ValueError("bootstrap must be a bool")
    steps = _positive_int(iterations, name="iterations")
    size = min(_positive_int(batch_size, name="batch_size"), len(data))
    rate = float(learning_rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    decay = float(weight_decay)
    if not math.isfinite(decay) or decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    tail = float(tail_fraction)
    if not 0.0 <= tail < 1.0:
        raise ValueError("tail_fraction must lie in [0, 1)")
    warmup = float(warmup_fraction)
    if not 0.0 <= warmup < 1.0:
        raise ValueError("warmup_fraction must lie in [0, 1)")
    warmup_steps = min(int(warmup * steps), steps - 1)

    started = time.perf_counter()
    dimension = len(data[0])
    input_mean, input_scale = _standardization(list(zip(*data, strict=True)))
    (output_mean,), (output_scale,) = _standardization([outputs])
    scaled_inputs = tuple(
        tuple(
            (value - mean) / scale
            for value, mean, scale in zip(row, input_mean, input_scale, strict=True)
        )
        for row in data
    )
    scaled_targets = tuple((value - output_mean) / output_scale for value in outputs)
    network = MLP(
        (dimension, *widths, 2 if heteroscedastic else 1), activation=activation
    )
    mask = _weight_mask(network, heteroscedastic)
    root = resolve_stream(seed)

    trained: list[tuple[float, ...]] = []
    traces: list[tuple[float, ...]] = []
    losses: list[float] = []
    for member in range(count_members):
        stream = root.child("member", member)
        if bootstrap:
            generator = stream.child("bootstrap").generator()
            chosen = [generator.randrange(len(data)) for _ in range(len(data))]
        else:
            chosen = list(range(len(data)))
        member_inputs = [scaled_inputs[i] for i in chosen]
        member_targets = [scaled_targets[i] for i in chosen]
        start = network.initial_parameters(stream.child("initial"))
        if not heteroscedastic:
            start = (*start, 0.0)

        def objective(
            parameters: tuple[float, ...],
            batch_stream: RandomStream,
            member_inputs: list[tuple[float, ...]] = member_inputs,
            member_targets: list[float] = member_targets,
            fixed: bool = False,
        ) -> tuple[float, list[float]]:
            """Return the mini-batch regularized NLL and its gradient.

            ``fixed=True`` holds the variance at one (mean warm-up).

            References:
                [EQ-SUR-DE-001],
                [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Algorithm 1].

            """
            generator = batch_stream.child("batch").generator()
            batch = generator.sample(range(len(member_inputs)), size)
            return _member_loss_and_gradient(
                network,
                heteroscedastic,
                decay,
                mask,
                parameters,
                [member_inputs[i] for i in batch],
                [member_targets[i] for i in batch],
                fixed_variance=fixed,
            )

        warmup_losses: tuple[float, ...] = ()
        if warmup_steps > 0:
            warm = adam_minimize(
                partial(objective, fixed=True),
                start,
                iterations=warmup_steps,
                learning_rate=rate,
                seed=stream.child("warmup"),
            )
            start = warm.x
            warmup_losses = warm.losses
        result = adam_minimize(
            objective,
            start,
            iterations=steps - warmup_steps,
            learning_rate=rate,
            tail_fraction=tail,
            seed=stream.child("adam"),
        )
        final_loss, _ = _member_loss_and_gradient(
            network,
            heteroscedastic,
            0.0,
            mask,
            result.x,
            member_inputs,
            member_targets,
        )
        trained.append(result.x)
        traces.append(warmup_losses + result.losses)
        losses.append(final_loss)

    ensemble = DeepEnsemble(
        network=network,
        members=tuple(trained),
        heteroscedastic=heteroscedastic,
        points=data,
        values=outputs,
        input_mean=tuple(input_mean),
        input_scale=tuple(input_scale),
        output_mean=output_mean,
        output_scale=output_scale,
        member_losses=tuple(losses),
        loss_traces=tuple(traces),
        training_rmse=math.nan,
        iterations=steps,
        batch_size=size,
        learning_rate=rate,
        weight_decay=decay,
        tail_fraction=tail,
        bootstrap=bootstrap,
        warmup_fraction=warmup,
    )
    squared = math.fsum(
        (ensemble.predict(point) - value) ** 2
        for point, value in zip(data, outputs, strict=True)
    )
    object.__setattr__(ensemble, "training_rmse", math.sqrt(squared / len(data)))
    object.__setattr__(ensemble, "training_seconds", time.perf_counter() - started)
    return ensemble


def fit_deep_ensemble_to_model(
    model: ProbabilisticModel,
    quantity: str | None = None,
    *,
    size: int,
    design: str = "lhs",
    seed: int | RandomStream | None = None,
    members: int = 5,
    hidden: Sequence[int] = (32, 32),
    activation: str = "tanh",
    heteroscedastic: bool = True,
    iterations: int = 2000,
    learning_rate: float = 0.005,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    tail_fraction: float = 0.2,
    warmup_fraction: float = 0.3,
    bootstrap: bool = False,
) -> DeepEnsemble:
    """Train a deep-ensemble surrogate of one QoI on a design of the model.

    Args:
        model: Probabilistic model.
        quantity: QoI name; defaults to the first declared quantity.
        size: Number of training evaluations.
        design: ``"mc"``, ``"lhs"``, ``"halton"``, or ``"sobol"``.
        seed: Root seed or stream (child ``"ensemble-design"`` for the
            design, the root itself for the members).
        members: Number of networks.
        hidden: Hidden-layer widths.
        activation: Hidden activation.
        heteroscedastic: Predict an input-dependent variance.
        iterations: Adam steps per member.
        learning_rate: Adam step size.
        batch_size: Mini-batch size.
        weight_decay: Weight-decay coefficient.
        tail_fraction: Fraction of final Adam iterates averaged.
        warmup_fraction: Fraction of iterations spent on the mean alone.
        bootstrap: Train members on bootstrap resamples.

    Returns:
        Fitted :class:`DeepEnsemble` on physical inputs;
        ``model.evaluations`` grows by ``size``.

    Raises:
        KeyError: If ``quantity`` is not declared by the model.
        ValueError: For invalid settings (see :func:`fit_deep_ensemble`).

    Equation:
        [EQ-SUR-DE-001], [EQ-SUR-DE-002], [EQ-SAMP-INV-001].

    References:
        [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017, Algorithm 1].

    """
    name = model.quantity_names[0] if quantity is None else quantity
    model.quantity(name)
    root = resolve_stream(seed)
    unit = make_design(
        design, size, model.dimension, seed=root.child("ensemble-design")
    )
    sample: Any = model.inputs.sample_design(unit)
    evaluated = model.evaluate_many(sample)[name]
    return fit_deep_ensemble(
        sample,
        evaluated,
        members=members,
        hidden=hidden,
        activation=activation,
        heteroscedastic=heteroscedastic,
        iterations=iterations,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        tail_fraction=tail_fraction,
        warmup_fraction=warmup_fraction,
        bootstrap=bootstrap,
        seed=root,
    )
