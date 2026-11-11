"""Unadjusted and Metropolis-adjusted Langevin sampling.

The Langevin diffusion ``dX = grad log pi(X) dt / 2 + dW`` has stationary
law ``pi``. Its Euler-Maruyama discretization with step ``h``,

``y = x + (h / 2) grad log pi(x) + sqrt(h) xi``,  ``xi ~ N(0, I)``,

is the unadjusted Langevin algorithm (ULA, [EQ-GEN-LANG-001]); it needs only
the score, so it samples an energy model without its normalizing constant,
but its stationary law is biased by ``O(h)``. Using the step as a
Metropolis-Hastings proposal removes the bias (MALA, [EQ-GEN-LANG-002]).
Roberts and Tweedie (1996) analyse both chains and show that ULA can be
transient for any ``h`` when the tails are light enough, which the
adjustment cures.

:func:`sample_langevin` serves the energy models and accepts any
log-density written with :class:`~probstat.numerics.admath`, differentiated
by reverse mode.

References:
    [REF-ROBERTS-TWEEDIE-1996], [EQ-GEN-LANG-001], [EQ-GEN-LANG-002].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..foundations.random_streams import RandomStream
from ..numerics import reverse_value_and_gradient
from ._common import Point, as_points, positive_integer, positive_real, resolve

LogDensity = Callable[[Sequence[Any]], Any]
ValueAndGradient = Callable[[Sequence[float]], tuple[float, Sequence[float]]]


@dataclass(frozen=True)
class LangevinResult:
    """Output of :func:`sample_langevin`.

    Args:
        samples: Retained states after burn-in and thinning.
        method: ``"ula"`` or ``"mala"``.
        step_size: Step ``h``.
        acceptance_rate: Fraction of accepted MALA proposals over all
            iterations, or ``None`` for ULA.
        burn_in: Discarded initial iterations.
        thin: Retained every ``thin``-th state.
        gradient_evaluations: Log-density value-and-gradient evaluations.

    References:
        [EQ-GEN-LANG-001], [EQ-GEN-LANG-002].

    """

    samples: tuple[Point, ...]
    method: str
    step_size: float
    acceptance_rate: float | None
    burn_in: int
    thin: int
    gradient_evaluations: int

    def mean(self) -> Point:
        """Return the coordinate-wise sample mean of the retained states.

        References:
            [EQ-GEN-DIAG-002]. Sample average.

        """
        n = len(self.samples)
        d = len(self.samples[0])
        return tuple(math.fsum(s[i] for s in self.samples) / n for i in range(d))


def _reverse_value_and_gradient(log_density: LogDensity) -> ValueAndGradient:
    """Return ``x -> (log pi(x), grad log pi(x))`` by reverse mode.

    References:
        [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """

    def evaluate(x: Sequence[float]) -> tuple[float, Sequence[float]]:
        """Evaluate the log-density and its gradient at ``x``.

        References:
            [EQ-AD-REVERSE-001].

        """
        return reverse_value_and_gradient(log_density, x)

    return evaluate


def _checked(
    oracle: ValueAndGradient, x: Sequence[float]
) -> tuple[float, tuple[float, ...]] | None:
    """Evaluate the oracle; return ``None`` outside the support.

    A ``ValueError`` (including a non-differentiable point) or a non-finite
    value or gradient is read as a point outside the support.

    References:
        [EQ-GEN-LANG-002], [REF-ROBERTS-TWEEDIE-1996].

    """
    try:
        value, gradient = oracle(x)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    value = float(value)
    values = tuple(float(g) for g in gradient)
    if not math.isfinite(value) or not all(math.isfinite(g) for g in values):
        return None
    return value, values


def sample_langevin(
    log_density: LogDensity,
    initial: object,
    *,
    n_samples: int,
    step_size: float,
    adjusted: bool = True,
    burn_in: int = 0,
    thin: int = 1,
    seed: int | RandomStream = 0,
    value_and_gradient: ValueAndGradient | None = None,
) -> LangevinResult:
    """Sample a density known up to a constant by (adjusted) Langevin steps.

    The proposal is ``y = x + (h/2) g(x) + sqrt(h) xi`` with
    ``g = grad log pi`` ([EQ-GEN-LANG-001]). With ``adjusted=True`` it is
    accepted with probability
    ``min(1, pi(y) q(x | y) / (pi(x) q(y | x)))``,
    ``q(y | x) = N(y; x + (h/2) g(x), h I)`` ([EQ-GEN-LANG-002]); otherwise
    every proposal is kept (ULA).

    Args:
        log_density: Log-density up to a constant, written with operators
            and :class:`~probstat.numerics.admath` (differentiated by
            reverse mode unless ``value_and_gradient`` is given).
        initial: Starting point (finite, inside the support).
        n_samples: Positive number of retained states.
        step_size: Positive step ``h``.
        adjusted: Apply the Metropolis-Hastings correction (MALA).
        burn_in: Non-negative number of discarded initial iterations.
        thin: Positive thinning interval.
        seed: Integer seed or :class:`RandomStream`.
        value_and_gradient: Optional ``x -> (log pi(x), grad log pi(x))``.

    Returns:
        :class:`LangevinResult`.

    Raises:
        ValueError: For invalid settings, a start outside the support, or an
            unadjusted step that leaves the support or diverges.

    References:
        [EQ-GEN-LANG-001], [EQ-GEN-LANG-002], [REF-ROBERTS-TWEEDIE-1996].

    """
    start = as_points([initial], name="initial")[0]
    count = positive_integer(n_samples, name="n_samples")
    h = positive_real(step_size, name="step_size")
    if isinstance(burn_in, bool) or not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError("burn_in must be a non-negative integer")
    interval = positive_integer(thin, name="thin")
    oracle = value_and_gradient or _reverse_value_and_gradient(log_density)
    generator = resolve(seed).child("langevin").generator()
    current = _checked(oracle, start)
    if current is None:
        raise ValueError("initial point is outside the support")
    x = list(start)
    value, gradient = current
    root = math.sqrt(h)
    half = 0.5 * h
    total = burn_in + count * interval
    accepted = 0
    evaluations = 1
    samples: list[Point] = []
    for iteration in range(1, total + 1):
        noise = [generator.gauss(0.0, 1.0) for _ in x]
        proposal = [
            xi + half * gi + root * e
            for xi, gi, e in zip(x, gradient, noise, strict=True)
        ]
        candidate = _checked(oracle, proposal)
        evaluations += 1
        if not adjusted:
            if candidate is None:
                raise ValueError(
                    f"unadjusted Langevin left the support at iteration {iteration}"
                )
            x, (value, gradient) = proposal, candidate
        elif candidate is not None:
            new_value, new_gradient = candidate
            forward = math.fsum(e * e for e in noise) / 2.0
            backward = math.fsum(
                (xi - yi - half * gi) ** 2
                for xi, yi, gi in zip(x, proposal, new_gradient, strict=True)
            ) / (2.0 * h)
            log_ratio = new_value - value + forward - backward
            if log_ratio >= 0.0 or generator.random() < math.exp(log_ratio):
                x, value, gradient = proposal, new_value, new_gradient
                accepted += 1
        if iteration > burn_in and (iteration - burn_in) % interval == 0:
            samples.append(tuple(x))
    return LangevinResult(
        samples=tuple(samples),
        method="mala" if adjusted else "ula",
        step_size=h,
        acceptance_rate=accepted / total if adjusted else None,
        burn_in=burn_in,
        thin=interval,
        gradient_evaluations=evaluations,
    )
