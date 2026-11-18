"""Moving particles: the last-particle multilevel splitting algorithm.

``N`` particles start as independent standard-normal vectors. At every move
the particle with the largest limit-state value ``L = max G`` is discarded and
replaced by a draw from the standard normal law conditioned on ``G < L``,
obtained by a short Markov chain started from one of the other particles.
After ``M`` moves every particle satisfies ``G <= 0`` and ``(1 - 1/N)^M``
estimates ``P(G <= 0)``. Under exact conditional sampling the successive
levels form a Poisson process in ``-ln P(G <= L)`` with intensity ``N``, so
``M ~ Poisson(-N ln p)``, the estimator is unbiased, and an exact Poisson
confidence interval is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inference import poisson_mean_interval
from probstat.linalg import Vector

from .model_based import ModelLimitState, generalized_reliability_index
from .splitting import moving_particles_coefficient_of_variation


@dataclass(frozen=True, slots=True)
class MovingParticlesResult:
    """Moving-particles (last-particle) estimate of a failure probability.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: Estimate ``(1 - 1/N)^M``.
        confidence_interval: Exact Poisson interval
            ``(exp(-lambda_hi / N), exp(-lambda_lo / N))`` built from the
            Garwood interval ``(lambda_lo, lambda_hi)`` of the mean of ``M``.
        confidence: Nominal level of ``confidence_interval``.
        coefficient_of_variation: Ideal c.o.v. ``sqrt(p^(-1/N) - 1)`` at the
            estimate (about ``sqrt(-ln(p) / N)``).
        particles: Number of particles ``N``.
        moves: Number of moves ``M``.
        steps: Markov-chain steps per move.
        correlation: Crank-Nicolson correlation ``rho``.
        evaluations: Model evaluations, ``N + M * steps``.
        levels: Successive levels ``L_1 > L_2 > ... > L_M`` (each the largest
            limit-state value at the start of a move, all positive).
        acceptance_rate: Fraction of accepted Markov-chain proposals over all
            moves (``nan`` when no move was needed).
        final_particles: Particle positions in standard-normal space at the
            end of the run.
        final_values: Limit-state values of ``final_particles``.
        converged: ``True`` when every particle satisfies ``G <= 0``;
            ``False`` when ``max_moves`` stopped the run, in which case the
            estimate is biased upward.

    The interval and the c.o.v. assume exact conditional sampling. A Markov
    chain with too few steps or a low acceptance rate correlates the new
    particle with its starting particle, and the real spread is then larger
    than reported.

    References:
        [REF-GUYADER-HENGARTNER-MATZNERLOBER-2011], [REF-WALTER-2015],
        [REF-GARWOOD-1936].

    """

    limit_state: str
    probability: float
    confidence_interval: tuple[float, float]
    confidence: float
    coefficient_of_variation: float
    particles: int
    moves: int
    steps: int
    correlation: float
    evaluations: int
    levels: Vector
    acceptance_rate: float
    final_particles: tuple[Vector, ...]
    final_values: Vector
    converged: bool

    @property
    def beta(self) -> float:
        """Return the generalized reliability index of the estimate.

        Equation:
            [EQ-REL-BETA-001].

        References:
            [REF-DITLEVSEN-MADSEN-1996].

        """
        return generalized_reliability_index(self.probability)

    @property
    def variance(self) -> float:
        """Return the ideal variance ``p^2 (p^(-1/N) - 1)`` at the estimate.

        Equation:
            [EQ-REL-MP-001].

        References:
            [REF-GUYADER-HENGARTNER-MATZNERLOBER-2011].

        """
        return (self.probability * self.coefficient_of_variation) ** 2


def moving_particles_interval(
    moves: int, particles: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Return the exact Poisson confidence interval of a moving-particles run.

    Under exact conditional sampling ``M ~ Poisson(lambda)`` with
    ``lambda = -N ln p``. Garwood's interval for ``lambda`` uses the
    chi-square quantiles ``lambda_lo = chi2_{alpha/2}(2M) / 2`` (zero when
    ``M = 0``) and ``lambda_hi = chi2_{1-alpha/2}(2M + 2) / 2``, with
    ``alpha = 1 - confidence``; equivalently they are the Poisson means whose
    upper tail ``P(X >= M)`` and lower tail ``P(X <= M)`` equal ``alpha/2``.
    Because ``p = exp(-lambda / N)`` decreases in ``lambda`` the endpoints
    swap: the interval for ``p`` is
    ``(exp(-lambda_hi / N), exp(-lambda_lo / N))``, and its coverage is at
    least the nominal level.

    Args:
        moves: Observed number of moves ``M``, a non-negative integer.
        particles: Number of particles ``N``, at least two.
        confidence: Two-sided level strictly inside ``(0, 1)``.

    Returns:
        Lower and upper confidence limits of ``p``.

    Raises:
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-REL-MP-002], [EQ-INF-COUNT-003].

    References:
        [REF-GUYADER-HENGARTNER-MATZNERLOBER-2011], [REF-GARWOOD-1936].

    """
    if isinstance(moves, bool) or not isinstance(moves, int) or moves < 0:
        raise ValueError("moves must be a non-negative integer")
    if isinstance(particles, bool) or not isinstance(particles, int) or particles < 2:
        raise ValueError("particles must be an integer of at least 2")
    interval = poisson_mean_interval([moves], confidence)
    return (
        math.exp(-interval.upper / particles),
        math.exp(-interval.lower / particles),
    )


def _require_count(value: object, name: str, minimum: int) -> int:
    """Validate an integer option with a lower limit.

    Args:
        value: Candidate value.
        name: Argument name used in error messages.
        minimum: Smallest admissible value.

    Returns:
        The validated integer.

    Raises:
        ValueError: If ``value`` is not an integer of at least ``minimum``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def moving_particles(
    limit_state: ModelLimitState,
    particles: int = 100,
    *,
    steps: int = 20,
    correlation: float = 0.8,
    seed: int | RandomStream | None = None,
    confidence: float = 0.95,
    max_moves: int | None = None,
) -> MovingParticlesResult:
    """Estimate a small failure probability by moving particles.

    ``N`` independent standard-normal particles are drawn and evaluated.
    While the largest value ``L = max_i G(u_i)`` is positive, that particle
    is removed, one of the other ``N - 1`` particles is chosen uniformly at
    random, and a Markov chain of ``steps`` steps is run from a copy of it.
    Each step proposes the conditional (preconditioned Crank-Nicolson) move
    ``v = rho u + sqrt(1 - rho^2) z`` with ``z`` standard normal, which leaves
    the standard normal law invariant, and accepts ``v`` exactly when
    ``G(v) < L``; the chain therefore targets the standard normal law
    restricted to ``G < L``. The final state replaces the removed particle.
    The run stops when every particle satisfies ``G <= 0``; with ``M`` moves
    the estimate is ``p = (1 - 1/N)^M``.

    Under exact conditional sampling ``M ~ Poisson(-N ln p)``, so the
    estimator is unbiased with variance ``p^2 (p^(-1/N) - 1)`` and c.o.v.
    close to ``sqrt(-ln(p) / N)``; the expected cost is about
    ``N (1 - steps ln p)`` model evaluations. The interval is the exact
    Poisson interval of :func:`moving_particles_interval`.

    Args:
        limit_state: Model limit state (failure ``G <= 0``).
        particles: Number of particles ``N``, at least two.
        steps: Markov-chain steps per move, at least one; each step costs one
            model evaluation.
        correlation: Crank-Nicolson correlation ``rho`` in ``[0, 1)``;
            ``rho = 0`` proposes independent standard-normal draws.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.
        confidence: Level of the Poisson interval, strictly inside ``(0, 1)``.
        max_moves: Largest number of moves before the run stops unconverged;
            ``None`` uses ``200 N`` (enough for ``p`` down to ``e^-200`` on
            average).

    Returns:
        Estimate, Poisson interval, moves, evaluations, level sequence, mean
        acceptance rate, final particles, and convergence flag.

    Raises:
        TypeError: If ``limit_state`` has the wrong type or an option is not
            real.
        ValueError: If an option is outside its domain or the limit state
            returns a non-finite value.

    Equation:
        [EQ-REL-MP-001], [EQ-REL-MP-002], [EQ-REL-PCN-001].

    References:
        [REF-GUYADER-HENGARTNER-MATZNERLOBER-2011], [REF-WALTER-2015],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    if not isinstance(limit_state, ModelLimitState):
        raise TypeError("limit_state must be a ModelLimitState")
    state = limit_state
    count = _require_count(particles, "particles", 2)
    chain_steps = _require_count(steps, "steps", 1)
    rho = require_finite_real(correlation, name="correlation")
    if not 0.0 <= rho < 1.0:
        raise ValueError("correlation must lie in [0, 1)")
    level_confidence = require_finite_real(confidence, name="confidence")
    if not 0.0 < level_confidence < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    move_limit = (
        200 * count if max_moves is None else _require_count(max_moves, "max_moves", 1)
    )
    innovation = math.sqrt(1.0 - rho * rho)
    generator = resolve_stream(seed).child("moving-particles").generator()
    dimension = state.dimension
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

    points: list[Vector] = [
        tuple(generator.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(count)
    ]
    values = [evaluate(u) for u in points]
    levels: list[float] = []
    accepted = 0
    proposed = 0
    while len(levels) < move_limit:
        worst = max(range(count), key=values.__getitem__)
        level = values[worst]
        if level <= 0.0:
            break
        levels.append(level)
        start = generator.randrange(count - 1)
        if start >= worst:
            start += 1
        current, current_value = points[start], values[start]
        for _ in range(chain_steps):
            candidate = tuple(
                rho * coordinate + innovation * generator.gauss(0.0, 1.0)
                for coordinate in current
            )
            candidate_value = evaluate(candidate)
            proposed += 1
            if candidate_value < level:
                current, current_value = candidate, candidate_value
                accepted += 1
        points[worst], values[worst] = current, current_value
    converged = max(values) <= 0.0
    moves = len(levels)
    probability = (1.0 - 1.0 / count) ** moves
    interval = moving_particles_interval(moves, count, level_confidence)
    cov = (
        moving_particles_coefficient_of_variation(probability, count)
        if probability < 1.0
        else 0.0
    )
    return MovingParticlesResult(
        limit_state=state.label,
        probability=probability,
        confidence_interval=interval,
        confidence=level_confidence,
        coefficient_of_variation=cov,
        particles=count,
        moves=moves,
        steps=chain_steps,
        correlation=rho,
        evaluations=state.model.evaluations - before,
        levels=tuple(levels),
        acceptance_rate=accepted / proposed if proposed else math.nan,
        final_particles=tuple(points),
        final_values=tuple(values),
        converged=converged,
    )
