"""Theory of ideal multilevel splitting (conditioned sampling).

A small probability ``p = P(G <= 0)`` factors through nested levels
``b_0 = +inf > b_1 > ... > b_M = 0`` into conditional probabilities
``P_m = P(G <= b_m | G <= b_{m-1})``. When every ``P_m`` is estimated from
independent ideal draws, the relative variance of the product estimator has a
closed form. These functions evaluate it, find the number of levels that
minimizes it for a fixed total budget, and compare the result with crude Monte
Carlo and the moving-particles (last-particle) estimator under the same
idealization. They are design tools: no limit state is evaluated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.optimization import brent_root

_LEVEL_SCAN_FACTOR = 10.0
"""Integer levels up to ``max(50, 10 |ln p|)`` are scanned for the optimum."""


def _require_probability_open(value: object, name: str = "probability") -> float:
    """Validate a probability strictly inside ``(0, 1)``.

    Args:
        value: Candidate probability.
        name: Argument name used in error messages.

    Returns:
        The probability as a float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If it is not finite or not strictly inside ``(0, 1)``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    probability = require_finite_real(value, name=name)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return probability


def _require_count(value: object, name: str, minimum: int = 1) -> int:
    """Validate an integer count with a lower limit.

    Args:
        value: Candidate count.
        name: Argument name used in error messages.
        minimum: Smallest admissible value.

    Returns:
        The count.

    Raises:
        ValueError: If ``value`` is not an integer of at least ``minimum``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _relative_variance(probability: float, levels: int, budget: float) -> float:
    """Return ``delta^2 = (delta0^2 + 1)^M - 1`` for equal ideal levels.

    Args:
        probability: Target probability ``p`` in ``(0, 1)``.
        levels: Number of levels ``M``.
        budget: Total number of ideal draws ``N``, split as ``N / M``.

    Returns:
        Squared coefficient of variation of the product estimator.

    Equation:
        [EQ-REL-SPLIT-001].

    References:
        [REF-LAGNOUX-2006].

    """
    level_term = math.expm1(-math.log(probability) / levels)
    delta0_squared = level_term * levels / budget
    return math.expm1(levels * math.log1p(delta0_squared))


def splitting_coefficient_of_variation(
    probability: float, levels: int, budget: int
) -> float:
    """Return the c.o.v. of ideal splitting with equal conditional levels.

    The ``M`` thresholds are placed so that every conditional probability
    equals ``p^(1/M)`` and each one is estimated from ``N / M`` independent
    ideal draws (exact conditional sampling, no Markov-chain correlation).
    Each level estimate then has squared c.o.v.
    ``delta0^2 = (p^(-1/M) - 1) M / N``, and because the level estimates are
    independent and unbiased the product has
    ``delta^2 = (delta0^2 + 1)^M - 1``. ``M = 1`` is crude Monte Carlo,
    ``delta^2 = (1 - p) / (p N)``.

    Args:
        probability: Target probability ``p`` strictly inside ``(0, 1)``.
        levels: Number of levels ``M``, a positive integer.
        budget: Total number of ideal draws ``N``, an integer of at least
            ``M``; ``N / M`` need not be an integer.

    Returns:
        Coefficient of variation ``delta`` of the product estimator.

    Raises:
        TypeError: If ``probability`` is not real.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-REL-SPLIT-001].

    References:
        [REF-LAGNOUX-2006], [REF-AU-BECK-2001].

    """
    p = _require_probability_open(probability)
    m = _require_count(levels, "levels")
    n = _require_count(budget, "budget")
    if n < m:
        raise ValueError("budget must be at least the number of levels")
    return math.sqrt(_relative_variance(p, m, float(n)))


def _optimal_level_exponent() -> float:
    """Return ``x*``, the positive root of ``x e^x = 2 (e^x - 1)``.

    With ``x = -ln(p) / M`` the large-budget relative variance is
    ``(ln p)^2 (e^x - 1) / (x^2 N)``; setting its derivative in ``x`` to zero
    gives this equation. The root lies in ``(1, 2)``: the left side minus the
    right side is ``2 - e < 0`` at one and ``2 > 0`` at two.

    Returns:
        ``x* = 1.5936...``.

    Equation:
        [EQ-REL-SPLIT-002].

    References:
        [REF-LAGNOUX-2006], [REF-BRENT-1973, Chapter 4].

    """

    def stationarity(x: float) -> float:
        """Return ``x e^x - 2 (e^x - 1)``.

        Args:
            x: Level exponent.

        Returns:
            Stationarity residual.

        Equation:
            [EQ-REL-SPLIT-002].

        References:
            [REF-LAGNOUX-2006].

        """
        return x * math.exp(x) - 2.0 * math.expm1(x)

    return brent_root(stationarity, 1.0, 2.0, tolerance=1e-15).root


OPTIMAL_LEVEL_EXPONENT = _optimal_level_exponent()
"""``x* = 1.5936...``: root of ``x e^x = 2 (e^x - 1)`` (large-budget optimum)."""

OPTIMAL_LEVEL_PROBABILITY = math.exp(-OPTIMAL_LEVEL_EXPONENT)
"""``exp(-x*) = 0.2032...``: optimal conditional probability per level."""

OPTIMAL_LEVELS_PER_LOG = 1.0 / OPTIMAL_LEVEL_EXPONENT
"""``1 / x* = 0.6275...``: optimal levels per unit of ``-ln p``."""


def moving_particles_coefficient_of_variation(
    probability: float, particles: float
) -> float:
    """Return the c.o.v. of the ideal moving-particles estimator.

    With ``N`` particles and exact conditional sampling the number of moves
    is ``M ~ Poisson(-N ln p)`` and ``(1 - 1/N)^M`` is unbiased with variance
    ``p^2 (p^(-1/N) - 1)``, so ``delta = sqrt(p^(-1/N) - 1)``, close to
    ``sqrt(-ln(p) / N)`` for large ``N``.

    Args:
        probability: Target probability ``p`` strictly inside ``(0, 1)``.
        particles: Number of particles ``N``, a real number above one (a
            non-integer value is allowed for budget comparisons).

    Returns:
        Exact ideal coefficient of variation.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-REL-MP-001].

    References:
        [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011], [REF-WALTER-2015].

    """
    p = _require_probability_open(probability)
    n = require_finite_real(particles, name="particles")
    if n <= 1.0:
        raise ValueError("particles must exceed one")
    return math.sqrt(math.expm1(-math.log(p) / n))


@dataclass(frozen=True, slots=True)
class OptimalSplittingResult:
    """Optimal number of ideal splitting levels for a fixed budget.

    Args:
        probability: Target probability ``p``.
        budget: Total number of ideal draws ``N``.
        levels: Integer ``M`` minimizing the exact c.o.v. of
            [EQ-REL-SPLIT-001].
        level_probability: Conditional probability ``p^(1/M)`` per level at
            the integer optimum.
        coefficient_of_variation: Exact c.o.v. at the integer optimum.
        continuous_levels: Large-budget optimum ``-ln(p) / x*``
            (about ``-0.6275 ln p``).
        continuous_level_probability: ``exp(-x*)``, about ``0.2032``.
        asymptotic_coefficient_of_variation: Large-budget c.o.v. at the
            continuous optimum, ``|ln p| sqrt(e^x* / (2 x* N))``.
        monte_carlo_coefficient_of_variation: Crude Monte Carlo
            ``sqrt((1 - p) / (p N))`` with the same budget.
        moving_particles_coefficient_of_variation: Ideal moving-particles
            c.o.v. ``sqrt(p^(-1/K) - 1)`` with ``K = N / (1 - ln p)``
            particles, so that the expected number of ideal draws ``K + M``
            equals the budget.

    References:
        [REF-LAGNOUX-2006], [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011].

    """

    probability: float
    budget: int
    levels: int
    level_probability: float
    coefficient_of_variation: float
    continuous_levels: float
    continuous_level_probability: float
    asymptotic_coefficient_of_variation: float
    monte_carlo_coefficient_of_variation: float
    moving_particles_coefficient_of_variation: float


def optimal_splitting(probability: float, budget: int) -> OptimalSplittingResult:
    """Return the number of equal splitting levels that minimizes the c.o.v.

    The integer optimum minimizes ``delta^2(M) = (delta0^2 + 1)^M - 1`` of
    [EQ-REL-SPLIT-001] over ``M = 1, ..., min(N, max(50, 10 |ln p|))``. For a
    large budget ``delta^2 ~ M^2 (p^(-1/M) - 1) / N``; with
    ``x = -ln(p) / M`` this is ``(ln p)^2 (e^x - 1) / (x^2 N)``, minimized at
    the root ``x* = 1.5936`` of ``x e^x = 2 (e^x - 1)``. Hence
    ``M ~ -0.6275 ln p`` levels of conditional probability
    ``exp(-x*) = 0.2032`` each, independently of ``N``. The approximation
    drops terms of order ``1 / N`` and treats ``M`` as continuous; for a
    small budget the integer optimum can differ from it.

    Args:
        probability: Target probability ``p`` strictly inside ``(0, 1)``.
        budget: Total number of ideal draws ``N``, a positive integer.

    Returns:
        Integer and continuous optima with crude Monte Carlo and
        moving-particles coefficients of variation for the same budget.

    Raises:
        TypeError: If ``probability`` is not real.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-REL-SPLIT-001], [EQ-REL-SPLIT-002], [EQ-REL-MP-001].

    References:
        [REF-LAGNOUX-2006], [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011].

    """
    p = _require_probability_open(probability)
    n = _require_count(budget, "budget")
    log_inverse = -math.log(p)
    upper = min(n, max(50, math.ceil(_LEVEL_SCAN_FACTOR * log_inverse)))
    best_levels = 1
    best_variance = _relative_variance(p, 1, float(n))
    for levels in range(2, upper + 1):
        variance = _relative_variance(p, levels, float(n))
        if variance < best_variance:
            best_levels, best_variance = levels, variance
    x_star = OPTIMAL_LEVEL_EXPONENT
    asymptotic = log_inverse * math.sqrt(math.exp(x_star) / (2.0 * x_star * n))
    particles = n / (1.0 + log_inverse)
    moving = (
        math.sqrt(math.expm1(log_inverse / particles)) if particles > 1.0 else math.inf
    )
    return OptimalSplittingResult(
        probability=p,
        budget=n,
        levels=best_levels,
        level_probability=math.exp(-log_inverse / best_levels),
        coefficient_of_variation=math.sqrt(best_variance),
        continuous_levels=log_inverse / x_star,
        continuous_level_probability=OPTIMAL_LEVEL_PROBABILITY,
        asymptotic_coefficient_of_variation=asymptotic,
        monte_carlo_coefficient_of_variation=math.sqrt((1.0 - p) / (p * n)),
        moving_particles_coefficient_of_variation=moving,
    )
