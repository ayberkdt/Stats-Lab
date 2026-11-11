"""Stochastic first-order minimization with Adam.

Training a density, score, or drift network minimizes a loss whose gradient
is estimated from a random mini-batch, so line-search methods such as BFGS do
not apply. :func:`adam_minimize` implements the Adam update [EQ-OPT-ADAM-002]
with optional Polyak-Ruppert averaging of the last iterates, and reports the
loss trace so convergence can be judged instead of assumed.

References:
    [REF-KINGMA-BA-2015, Algorithm 1] and [REF-POLYAK-JUDITSKY-1992].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream

StochasticObjective = Callable[
    [tuple[float, ...], RandomStream], tuple[float, Sequence[float]]
]


@dataclass(frozen=True, slots=True)
class AdamResult:
    """Outcome of :func:`adam_minimize`.

    Args:
        x: Reported parameters: the average of the last ``tail_fraction`` of
            the iterates, or the final iterate when averaging is off.
        final_iterate: Parameters after the last update.
        losses: Loss estimate at every iteration (before its update).
        iterations: Completed iterations.
        gradient_evaluations: Calls of the objective (one per iteration).
        averaged_over: Number of iterates in the reported average.

    References:
        [EQ-OPT-ADAM-002].

    """

    x: tuple[float, ...]
    final_iterate: tuple[float, ...]
    losses: tuple[float, ...]
    iterations: int
    gradient_evaluations: int
    averaged_over: int

    def smoothed_loss(self, window: int = 50) -> float:
        """Return the mean loss over the last ``window`` iterations.

        Args:
            window: Positive number of trailing iterations.

        Returns:
            Mean of the trailing loss estimates.

        Raises:
            ValueError: If ``window`` is not a positive integer.

        References:
            Project reporting policy [REF-PROBSTAT-REPORTING].
            No equation is evaluated.

        """
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        tail = self.losses[-window:]
        return math.fsum(tail) / len(tail)


def adam_minimize(
    objective: StochasticObjective,
    x0: Sequence[float],
    *,
    iterations: int = 1000,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    tail_fraction: float = 0.0,
    seed: int | RandomStream = 0,
) -> AdamResult:
    """Minimize a stochastic objective with Adam.

    At iteration ``t`` the objective is called with the current parameters
    and the child stream ``t`` of ``seed`` (for mini-batches or noise), and
    must return ``(loss, gradient)``. The update is the descent form of
    [EQ-OPT-ADAM-002].

    Args:
        objective: ``(x, stream) -> (loss, gradient)``.
        x0: Starting parameters.
        iterations: Positive number of updates.
        learning_rate: Positive step size ``alpha``.
        beta1: First-moment decay in ``[0, 1)``.
        beta2: Second-moment decay in ``[0, 1)``.
        epsilon: Positive denominator offset.
        tail_fraction: Fraction in ``[0, 1)`` of the final iterates to
            average (``0`` reports the last iterate).
        seed: Integer seed or :class:`RandomStream`.

    Returns:
        :class:`AdamResult` with the loss trace.

    Raises:
        ValueError: For invalid settings, a gradient of the wrong length, or
            a non-finite loss or gradient.

    References:
        [EQ-OPT-ADAM-002], [REF-KINGMA-BA-2015, Algorithm 1],
        [REF-POLYAK-JUDITSKY-1992].

    """
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValueError("iterations must be a positive integer")
    if iterations < 1:
        raise ValueError("iterations must be a positive integer")
    for name, value in (("learning_rate", learning_rate), ("epsilon", epsilon)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (("beta1", beta1), ("beta2", beta2)):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must lie in [0, 1)")
    if not 0.0 <= tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie in [0, 1)")
    x = [float(value) for value in x0]
    if not x or not all(math.isfinite(value) for value in x):
        raise ValueError("x0 must be a non-empty finite vector")
    stream = seed if isinstance(seed, RandomStream) else RandomStream(seed)
    dimension = len(x)
    first = [0.0] * dimension
    second = [0.0] * dimension
    tail_start = iterations - max(1, int(round(tail_fraction * iterations)))
    averaged_over = iterations - tail_start if tail_fraction > 0.0 else 1
    total = [0.0] * dimension
    losses: list[float] = []
    for step in range(1, iterations + 1):
        loss, gradient = objective(tuple(x), stream.child("adam", step))
        gradient = [float(value) for value in gradient]
        if len(gradient) != dimension:
            raise ValueError("gradient length does not match the parameters")
        if not math.isfinite(loss) or not all(math.isfinite(g) for g in gradient):
            raise ValueError(f"non-finite loss or gradient at iteration {step}")
        losses.append(float(loss))
        correction1 = 1.0 - beta1**step
        correction2 = 1.0 - beta2**step
        for index, value in enumerate(gradient):
            first[index] = beta1 * first[index] + (1.0 - beta1) * value
            second[index] = beta2 * second[index] + (1.0 - beta2) * value * value
            step_size = (first[index] / correction1) / (
                math.sqrt(second[index] / correction2) + epsilon
            )
            x[index] -= learning_rate * step_size
        if tail_fraction > 0.0 and step > tail_start:
            for index in range(dimension):
                total[index] += x[index]
    final = tuple(x)
    reported = (
        tuple(value / averaged_over for value in total)
        if tail_fraction > 0.0
        else final
    )
    return AdamResult(
        x=reported,
        final_iterate=final,
        losses=tuple(losses),
        iterations=iterations,
        gradient_evaluations=iterations,
        averaged_over=averaged_over,
    )
