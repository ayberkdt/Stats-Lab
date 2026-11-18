"""Model-based directional simulation with several roots per ray.

In independent standard-normal space ``U = R A`` with ``R^2 ~ chi-square_d``
independent of the uniform unit direction ``A``. For one direction ``a`` the
failure set along the ray ``{r a : r >= 0}`` is a union of radial segments,
and its probability ``P(R in F_a)`` is a sum of chi-distribution masses. The
failure probability is the mean of that quantity over random directions.
Unlike :func:`~probstat.reliability.estimate_directional_failure_probability`,
which needs a caller-supplied boundary radius and a star-shaped failure set,
:func:`directional_simulation` locates every sign change of the limit state
along each ray from a radial grid and refines each one with Brent's method.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from probstat.distributions import ChiSquare
from probstat.distributions._special import regularized_gamma_q
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector
from probstat.monte_carlo import estimate_mean
from probstat.optimization import brent_root

from .model_based import ModelLimitState, generalized_reliability_index

RADIUS_TAIL_PROBABILITY = 1e-15
"""Default ``radius_max`` leaves at most this chi-square mass beyond it."""

_ROOT_TOLERANCE = 1e-10
"""Absolute bracket-width tolerance of the radial root refinement."""


@dataclass(frozen=True, slots=True)
class DirectionalSimulationResult:
    """Directional-simulation estimate of a failure probability.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: Mean of the per-direction (or per-pair) probabilities.
        standard_error: Sample standard deviation of those values over
            ``sqrt(directions)``.
        confidence_interval: Normal (central-limit) interval at
            ``confidence``.
        confidence: Nominal level of ``confidence_interval``.
        coefficient_of_variation: ``standard_error / probability``; ``inf``
            when the estimate is zero.
        directions: Number of independent sampled directions (pairs when
            ``antithetic`` is true).
        rays: Number of rays scanned, ``directions`` or ``2 directions``.
        antithetic: Whether every direction was paired with its opposite.
        evaluations: Model evaluations, grid scans plus root refinement.
        radius_max: Outer radius of the radial scan.
        grid: Number of radial grid intervals on ``[0, radius_max]``.
        origin_fails: Whether ``G(0) <= 0``.
        roots: Radii of the located sign changes, one tuple per ray in
            sampling order (the opposite ray directly follows its direction
            when ``antithetic`` is true).
        ray_probabilities: Conditional failure probability ``P(R in F_a)`` of
            each ray, in the same order.

    References:
        [REF-BJERAGER-1988], [REF-DITLEVSEN-MADSEN-1996].

    """

    limit_state: str
    probability: float
    standard_error: float
    confidence_interval: tuple[float, float]
    confidence: float
    coefficient_of_variation: float
    directions: int
    rays: int
    antithetic: bool
    evaluations: int
    radius_max: float
    grid: int
    origin_fails: bool
    roots: tuple[Vector, ...]
    ray_probabilities: Vector

    @property
    def beta(self) -> float:
        """Return the generalized reliability index of the estimate.

        Equation:
            [EQ-REL-BETA-001].

        References:
            [REF-DITLEVSEN-MADSEN-1996].

        """
        return generalized_reliability_index(self.probability)


def default_radius_max(dimension: int) -> float:
    """Return the radius beyond which the chi(d) mass is below ``1e-15``.

    Args:
        dimension: Standard-normal dimension ``d``, a positive integer.

    Returns:
        ``sqrt(chi2_d^{-1}(1 - 1e-15))``, about 8.3 for ``d = 2``.

    Raises:
        ValueError: If ``dimension`` is not a positive integer.

    Equation:
        [EQ-REL-DIR-002].

    References:
        [REF-BJERAGER-1988].

    """
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    return math.sqrt(ChiSquare(float(dimension)).isf(RADIUS_TAIL_PROBABILITY))


def _chi_tail(dimension: int, radius: float) -> float:
    """Return ``P(R > r)`` for ``R^2 ~ chi-square_d``.

    Args:
        dimension: Degrees of freedom ``d``.
        radius: Non-negative radius or ``inf``.

    Returns:
        ``Q(d/2, r^2/2)``, one at the origin and zero at infinity.

    Equation:
        [EQ-REL-DIR-001].

    References:
        [REF-DITLEVSEN-MADSEN-1996].

    """
    if radius == math.inf:
        return 0.0
    if radius <= 0.0:
        return 1.0
    return regularized_gamma_q(0.5 * dimension, 0.5 * radius * radius)


def _ray_probability(
    values: list[float],
    radii: list[float],
    dimension: int,
    ray: Callable[[float], float],
) -> tuple[float, Vector]:
    """Return the chi mass of the failure segments along one ray.

    Every grid interval whose end values lie on opposite sides of failure
    (``G <= 0`` against ``G > 0``) holds one boundary radius, refined by
    Brent's method; the failure segments between consecutive boundaries are
    integrated exactly with the chi(d) law. A ray still failing at the last
    grid radius is taken to fail out to infinity.

    Args:
        values: Limit-state values on the radial grid, starting at the origin.
        radii: Grid radii, ``radii[0] = 0``.
        dimension: Standard-normal dimension ``d``.
        ray: ``r -> G(r a)`` for this ray; called only inside brackets.

    Returns:
        Conditional failure probability of the ray and its boundary radii.

    Equation:
        [EQ-REL-DIR-002], [EQ-OPT-BRENT-001].

    References:
        [REF-BJERAGER-1988], [REF-BRENT-1973, Chapter 4].

    """
    roots: list[float] = []
    failing = values[0] <= 0.0
    entry = 0.0 if failing else math.nan
    total = 0.0
    for index in range(1, len(radii)):
        now_failing = values[index] <= 0.0
        if now_failing == failing:
            continue
        lower, upper = radii[index - 1], radii[index]
        cached = {lower: values[index - 1], upper: values[index]}

        def bracketed(radius: float, cache: dict[float, float] = cached) -> float:
            """Return ``G(r a)``, reusing the two grid values at the ends.

            Args:
                radius: Radius inside the current bracket.
                cache: Known values at the bracket ends.

            Returns:
                Limit-state value.

            Equation:
                [EQ-REL-MLS-001].

            References:
                [REF-BRENT-1973, Chapter 4].

            """
            known = cache.get(radius)
            return ray(radius) if known is None else known

        root = brent_root(bracketed, lower, upper, tolerance=_ROOT_TOLERANCE).root
        roots.append(root)
        if now_failing:
            entry = root
        else:
            total += _chi_tail(dimension, entry) - _chi_tail(dimension, root)
        failing = now_failing
    if failing:
        total += _chi_tail(dimension, entry)
    return min(1.0, max(0.0, total)), tuple(roots)


def directional_simulation(
    limit_state: ModelLimitState,
    directions: int = 100,
    *,
    radius_max: float | None = None,
    grid: int = 20,
    antithetic: bool = False,
    seed: int | RandomStream | None = None,
    confidence: float = 0.95,
) -> DirectionalSimulationResult:
    """Estimate a failure probability by directional simulation.

    Writing ``U = R A`` with ``R^2 ~ chi-square_d`` independent of the
    uniform unit direction ``A``, ``P_f = E_A[P(R in F_A)]`` where
    ``F_a = {r >= 0 : G(r a) <= 0}``. For each sampled direction the limit
    state is evaluated on ``grid + 1`` equally spaced radii in
    ``[0, radius_max]``; every sign change is refined with Brent's method,
    and ``P(R in F_a)`` is the sum over failure segments ``[r_in, r_out]`` of
    ``Q(d/2, r_in^2/2) - Q(d/2, r_out^2/2)``, with ``r_in = 0`` when the
    origin fails and ``r_out = inf`` when the ray still fails at
    ``radius_max``. The value at the origin is computed once and shared. The
    estimate is the mean over directions with a central-limit interval. With
    ``antithetic=True`` each direction ``a`` is paired with ``-a`` and the
    pair average is one sample, which cancels the odd part of the integrand.

    Two boundary crossings inside one grid interval are not detected, so the
    grid must be finer than the narrowest failure (or safe) segment along a
    ray.

    Args:
        limit_state: Model limit state (failure ``G <= 0``).
        directions: Independent directions (pairs when ``antithetic``), at
            least two.
        radius_max: Outer scan radius; ``None`` uses
            :func:`default_radius_max`, beyond which the chi mass is below
            ``1e-15``.
        grid: Number of radial grid intervals, at least one.
        antithetic: Pair each direction with its opposite.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.
        confidence: Level of the normal interval, strictly inside ``(0, 1)``.

    Returns:
        Estimate, standard error, interval, evaluation count, and the roots
        and conditional probability of every ray.

    Raises:
        TypeError: If ``limit_state`` has the wrong type or an option is not
            real.
        ValueError: If an option is outside its domain or the limit state
            returns a non-finite value.

    Equation:
        [EQ-REL-DIR-001], [EQ-REL-DIR-002], [EQ-MC-SE-001].

    References:
        [REF-BJERAGER-1988], [REF-DITLEVSEN-MADSEN-1996].

    """
    if not isinstance(limit_state, ModelLimitState):
        raise TypeError("limit_state must be a ModelLimitState")
    state = limit_state
    if isinstance(directions, bool) or not isinstance(directions, int):
        raise ValueError("directions must be an integer of at least 2")
    if directions < 2:
        raise ValueError("directions must be an integer of at least 2")
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 1:
        raise ValueError("grid must be a positive integer")
    if not isinstance(antithetic, bool):
        raise TypeError("antithetic must be a bool")
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    dimension = state.dimension
    outer = (
        default_radius_max(dimension)
        if radius_max is None
        else require_finite_real(radius_max, name="radius_max")
    )
    if outer <= 0.0:
        raise ValueError("radius_max must be positive")
    generator = resolve_stream(seed).child("directional-simulation").generator()
    before = state.model.evaluations

    def evaluate(u: Vector) -> float:
        """Return ``G(u)`` and reject non-finite limit-state values.

        Args:
            u: Point in standard-normal space.

        Returns:
            Finite limit-state value.

        Raises:
            ValueError: If the value is not finite.

        Equation:
            [EQ-REL-MLS-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        value = state.standard(u)
        if not math.isfinite(value):
            raise ValueError("limit state returned a non-finite value")
        return value

    radii = [outer * index / grid for index in range(grid + 1)]
    radii[-1] = outer
    origin_value = evaluate(tuple(0.0 for _ in range(dimension)))

    def scan(direction: Vector) -> tuple[float, Vector]:
        """Return the conditional failure probability and roots of one ray.

        Args:
            direction: Unit direction ``a``.

        Returns:
            ``P(R in F_a)`` and the boundary radii along the ray.

        Equation:
            [EQ-REL-DIR-002].

        References:
            [REF-BJERAGER-1988].

        """

        def ray(radius: float) -> float:
            """Return ``G(r a)``.

            Args:
                radius: Non-negative radius.

            Returns:
                Limit-state value on the ray.

            Equation:
                [EQ-REL-MLS-001].

            References:
                [REF-BJERAGER-1988].

            """
            return evaluate(tuple(radius * component for component in direction))

        values = [origin_value] + [ray(radius) for radius in radii[1:]]
        return _ray_probability(values, radii, dimension, ray)

    samples: list[float] = []
    roots: list[Vector] = []
    ray_probabilities: list[float] = []
    for _ in range(directions):
        raw = [generator.gauss(0.0, 1.0) for _ in range(dimension)]
        norm = math.sqrt(math.fsum(value * value for value in raw))
        while norm == 0.0:
            raw = [generator.gauss(0.0, 1.0) for _ in range(dimension)]
            norm = math.sqrt(math.fsum(value * value for value in raw))
        direction = tuple(value / norm for value in raw)
        probability, ray_roots = scan(direction)
        roots.append(ray_roots)
        ray_probabilities.append(probability)
        if antithetic:
            opposite, opposite_roots = scan(tuple(-value for value in direction))
            roots.append(opposite_roots)
            ray_probabilities.append(opposite)
            samples.append(0.5 * (probability + opposite))
        else:
            samples.append(probability)
    estimate = estimate_mean(samples, confidence=level)
    mean = estimate.estimate
    return DirectionalSimulationResult(
        limit_state=state.label,
        probability=mean,
        standard_error=estimate.standard_error,
        confidence_interval=(
            estimate.confidence_interval.lower,
            estimate.confidence_interval.upper,
        ),
        confidence=level,
        coefficient_of_variation=(
            estimate.standard_error / mean if mean > 0.0 else math.inf
        ),
        directions=directions,
        rays=len(ray_probabilities),
        antithetic=antithetic,
        evaluations=state.model.evaluations - before,
        radius_max=outer,
        grid=grid,
        origin_fails=origin_value <= 0.0,
        roots=tuple(roots),
        ray_probabilities=tuple(ray_probabilities),
    )
