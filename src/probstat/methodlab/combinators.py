"""Combinators that derive new estimators from existing ones.

Each function here is a *construction principle* from the literature turned
into a reusable operation on :class:`~probstat.methodlab.Estimator` objects
(or, for the data-based recipes, on statistics of a sample). The output is
again an estimator, with the derivation step appended to its lineage, so
principles compose: a randomized-truncation estimator can be combined with a
control variate and then averaged, and the result still states how it was
built.

Every construction is exact only under conditions, which each docstring
states. None of them is checked by the type system: a derived estimator must
be certified with :mod:`probstat.methodlab.certify` before its properties are
relied upon.

Construction principles:

- randomized truncation of a convergent sequence (single-term and
  independent-sum estimators of Rhee and Glynn; McLeish) removes the bias of a
  sequence of approximations at the price of a random cost;
- Richardson extrapolation cancels the leading terms of a known bias
  expansion;
- best linear unbiased combination merges several unbiased estimators of
  one quantity with weights from their covariance;
- control variates subtract correlated estimators with known means;
- antithetic pairing averages a draw with its reflection;
- jackknife and bootstrap bias corrections remove the ``O(1/n)`` bias term
  of a statistic of a sample;
- the empirical influence function (infinitesimal jackknife) derives a
  standard error for any smooth weighted functional, and the one-step
  correction adds the mean of an influence function to a plug-in estimate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, lu_solve, solve_spd

from .estimators import Draw, Estimator

T = TypeVar("T")
LevelSampler = Callable[[int, RandomStream], Draw]

_PROBABILITY_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Randomized truncation (Rhee-Glynn, McLeish)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LevelDistribution:
    """A distribution of the random truncation level ``N`` on ``0, 1, 2, ...``.

    Either a finite probability vector (``N <= len(probabilities) - 1``) or a
    geometric law ``P(N = n) = (1 - ratio) ratio^n`` with unbounded support.
    Only an unbounded law removes the bias completely: with a finite vector
    of length ``L + 1`` the randomized estimators are unbiased for the
    level-``L`` approximation, not for its limit.

    Args:
        probabilities: Finite probability vector, or ``()`` for a geometric
            law.
        ratio: Geometric ratio in ``(0, 1)``, or ``nan`` for a finite vector.

    References:
        [REF-RHEE-GLYNN-2015], [EQ-MLAB-RT-001].

    """

    probabilities: Vector
    ratio: float

    @property
    def is_unbounded(self) -> bool:
        """Return whether the level has unbounded (geometric) support.

        References:
            [REF-RHEE-GLYNN-2015]. No equation is evaluated.

        """
        return not self.probabilities

    @property
    def maximum_level(self) -> int | None:
        """Return the largest attainable level, or ``None`` if unbounded.

        References:
            [REF-RHEE-GLYNN-2015]. No equation is evaluated.

        """
        return None if self.is_unbounded else len(self.probabilities) - 1

    def probability(self, level: int) -> float:
        """Return ``P(N = level)``.

        Args:
            level: Non-negative level.

        Returns:
            The point probability (zero above the maximum level).

        References:
            [REF-RHEE-GLYNN-2015], [EQ-MLAB-RT-001].

        """
        index = require_non_negative_integer(level, name="level")
        if self.is_unbounded:
            return float((1.0 - self.ratio) * self.ratio**index)
        if index >= len(self.probabilities):
            return 0.0
        return self.probabilities[index]

    def tail(self, level: int) -> float:
        """Return ``P(N >= level)``.

        Args:
            level: Non-negative level.

        Returns:
            The upper tail probability.

        References:
            [REF-RHEE-GLYNN-2015], [EQ-MLAB-RT-002].

        """
        index = require_non_negative_integer(level, name="level")
        if self.is_unbounded:
            return float(self.ratio**index)
        return math.fsum(self.probabilities[index:])

    def sample(self, stream: RandomStream) -> int:
        """Draw a level by inversion of the distribution function.

        Args:
            stream: Stream that supplies one uniform variate.

        Returns:
            A level with the declared law.

        References:
            [REF-DEVROYE-1986] (inversion; the geometric case is
            ``floor(log U / log ratio)``).

        """
        uniform = stream.generator().random()
        if self.is_unbounded:
            # 1 - U is in (0, 1], so the logarithm is finite.
            return int(math.log(1.0 - uniform) / math.log(self.ratio))
        cumulative = 0.0
        for level, mass in enumerate(self.probabilities):
            cumulative += mass
            if uniform < cumulative:
                return level
        return len(self.probabilities) - 1


def geometric_levels(ratio: float) -> LevelDistribution:
    """Return the geometric level law ``P(N = n) = (1 - ratio) ratio^n``.

    Args:
        ratio: Geometric ratio strictly between zero and one. Smaller ratios
            visit high levels less often (cheaper, more variable).

    Returns:
        Unbounded level distribution.

    Raises:
        ValueError: If ``ratio`` is not in ``(0, 1)``.

    References:
        [REF-RHEE-GLYNN-2015], [EQ-MLAB-RT-001].

    """
    value = require_finite_real(ratio, name="ratio")
    if not 0.0 < value < 1.0:
        raise ValueError("ratio must lie strictly between zero and one")
    return LevelDistribution((), value)


def finite_levels(probabilities: Sequence[float]) -> LevelDistribution:
    """Return a level law with the given finite probability vector.

    Args:
        probabilities: Strictly positive probabilities of levels
            ``0, ..., L`` that sum to one within ``1e-9``.

    Returns:
        Bounded level distribution. Randomized estimators built on it are
        unbiased for the level-``L`` approximation only.

    Raises:
        ValueError: If a probability is not positive or the sum is not one.

    References:
        [REF-RHEE-GLYNN-2015], [EQ-MLAB-RT-001].

    """
    values = tuple(require_finite_real(p, name="probability") for p in probabilities)
    if not values:
        raise ValueError("at least one level probability is required")
    if any(p <= 0.0 for p in values):
        raise ValueError("every level probability must be positive")
    if abs(math.fsum(values) - 1.0) > _PROBABILITY_TOLERANCE:
        raise ValueError("level probabilities must sum to one")
    return LevelDistribution(values, math.nan)


def optimal_level_probabilities(
    second_moments: Sequence[float], costs: Sequence[float]
) -> Vector:
    """Return the level probabilities that minimize second moment times cost.

    For the single-term estimator ``Z = Delta_N / p_N`` the second moment is
    ``sum_n m_n / p_n`` with ``m_n = E[Delta_n^2]`` and the expected cost is
    ``sum_n p_n c_n``. By the Cauchy-Schwarz inequality their product is at
    least ``(sum_n sqrt(m_n c_n))^2`` with equality at ``p_n`` proportional
    to ``sqrt(m_n / c_n)`` ([EQ-MLAB-RT-003]). The product of the *variance*
    and the cost differs by the term ``mu^2 sum_n p_n c_n``; the returned
    vector is therefore the exact optimum only when the target ``mu`` is
    zero and a good choice when ``mu^2`` is small against ``sum m_n / p_n``.

    Args:
        second_moments: Positive ``E[Delta_n^2]`` for levels ``0, ..., L``.
        costs: Positive costs of one draw of ``Delta_n``.

    Returns:
        Probabilities proportional to ``sqrt(m_n / c_n)``.

    Raises:
        ValueError: If lengths differ, are zero, or an entry is not positive.

    References:
        [REF-RHEE-GLYNN-2015] (work-variance trade-off of the
        level law), [EQ-MLAB-RT-003].

    """
    moments = tuple(
        require_finite_real(m, name="second moment") for m in second_moments
    )
    work = tuple(require_finite_real(c, name="cost") for c in costs)
    if not moments or len(moments) != len(work):
        raise ValueError("second_moments and costs need one entry per level")
    if any(m <= 0.0 for m in moments) or any(c <= 0.0 for c in work):
        raise ValueError("second moments and costs must be positive")
    weights = [math.sqrt(m / c) for m, c in zip(moments, work, strict=True)]
    total = math.fsum(weights)
    return tuple(w / total for w in weights)


def single_term_estimator(
    name: str, difference: LevelSampler, levels: LevelDistribution
) -> Estimator:
    """Derive the single-term randomized-truncation estimator.

    With ``Delta_n = Y_n - Y_(n-1)`` (``Y_(-1) = 0``) for a sequence of
    approximations ``Y_n`` whose expectations converge to ``mu``, and an
    independent level ``N`` with ``P(N = n) = p_n > 0``,

    ``Z = Delta_N / p_N``

    has expectation ``sum_n E[Delta_n] = mu`` ([EQ-MLAB-RT-001]). It is
    unbiased with finite variance when ``sum_n E[Delta_n^2] / p_n`` is
    finite; its expected cost is ``sum_n p_n c_n``. If ``E[Delta_n^2]``
    decays like ``2^(-beta n)`` and the cost grows like ``2^(gamma n)``,
    both are finite for a geometric law only when ``beta > gamma``.

    Args:
        name: Label of the derived estimator.
        difference: ``difference(n, stream)`` returns one draw of
            ``Delta_n`` with its cost; ``Y_n`` and ``Y_(n-1)`` inside one
            draw must be coupled (computed from the same randomness) for the
            second moments to decay.
        levels: Law of the truncation level. Unbounded support is required
            for unbiasedness with respect to the limit.

    Returns:
        Estimator with lineage ``("single-term randomized truncation ...",)``.
        The level is drawn from child stream ``"level"`` and the difference
        from ``("difference", n)``.

    References:
        [REF-RHEE-GLYNN-2015] (single term
        estimator), [REF-MCLEISH-2011], [EQ-MLAB-RT-001].

    """
    if not callable(difference):
        raise TypeError("difference must be callable")

    def sampler(stream: RandomStream) -> Draw:
        """Draw the level, then one importance-weighted difference.

        References:
            [EQ-MLAB-RT-001], [REF-RHEE-GLYNN-2015].

        """
        level = levels.sample(stream.child("level"))
        draw = difference(level, stream.child("difference", level))
        return Draw(draw.value / levels.probability(level), draw.cost)

    support = (
        "unbounded" if levels.is_unbounded else f"max level {levels.maximum_level}"
    )
    return Estimator(
        name,
        sampler,
        (f"single-term randomized truncation of a coupled sequence ({support})",),
    )


def coupled_sum_estimator(
    name: str, difference: LevelSampler, levels: LevelDistribution
) -> Estimator:
    """Derive the sum-type randomized-truncation estimator (independent sum).

    ``Z = sum_(n=0)^N Delta_n / P(N >= n)`` with the level ``N`` independent
    of the differences ([EQ-MLAB-RT-002]). Every difference is drawn on its
    own child stream ``("difference", n)``, so the differences of one
    replicate are *independent across levels*: this is the **independent-sum
    estimator** of Rhee and Glynn, not their coupled-sum estimator, in which
    all ``Y_0, ..., Y_N`` of one replicate come from a single coupled path.
    The function keeps its historical name. (Within one difference ``Y_n``
    and ``Y_(n-1)`` must still be coupled, exactly as for
    :func:`single_term_estimator`.)

    With ``mu_n = E[Y_n]``, ``mu_(-1) = 0`` and ``mu = lim mu_n``, the
    expectation is ``sum_n E[Delta_n] = mu`` and the second moment is

    ``E[Z^2] = sum_n (Var[Delta_n] + (mu - mu_(n-1))^2 - (mu - mu_n)^2)
    / P(N >= n)``,

    finite when ``sum_n Var[Delta_n] / P(N >= n)`` and
    ``sum_n (mu - mu_(n-1))^2 / P(N >= n)`` converge. The condition
    ``sum_n E[(Y_(n-1) - Y)^2] / P(N >= n) < infinity`` of Rhee and Glynn's
    Theorem 1 belongs to the coupled-sum estimator, whose variance is
    different (in a seeded example with one shared normal variate the
    implemented estimator has variance 22.7 where the coupled sum would have
    11.6). The cost of one replicate is ``sum_(n <= N) c_n``.

    Args:
        name: Label of the derived estimator.
        difference: ``difference(n, stream)`` returns one draw of
            ``Delta_n = Y_n - Y_(n-1)`` with its cost.
        levels: Law of the truncation level.

    Returns:
        Derived estimator whose differences are independent across levels.

    References:
        [REF-RHEE-GLYNN-2015] (independent-sum estimator; the coupled-sum
        estimator is not implemented), [EQ-MLAB-RT-002].

    """
    if not callable(difference):
        raise TypeError("difference must be callable")

    def sampler(stream: RandomStream) -> Draw:
        """Sum tail-weighted differences up to a random level.

        References:
            [EQ-MLAB-RT-002], [REF-RHEE-GLYNN-2015].

        """
        top = levels.sample(stream.child("level"))
        value = 0.0
        cost = 0.0
        for level in range(top + 1):
            draw = difference(level, stream.child("difference", level))
            value += draw.value / levels.tail(level)
            cost += draw.cost
        return Draw(value, cost)

    support = (
        "unbounded" if levels.is_unbounded else f"max level {levels.maximum_level}"
    )
    return Estimator(
        name,
        sampler,
        (f"sum of tail-weighted differences up to a random level ({support})",),
    )


@dataclass(frozen=True, slots=True)
class TruncationDesign:
    """Second moment and expected cost of a single-term estimator.

    Args:
        probabilities: Level probabilities used.
        second_moment: ``sum_n m_n / p_n``, the second moment of ``Z``.
        expected_cost: ``sum_n p_n c_n``.
        lower_bound: ``(sum_n sqrt(m_n c_n))^2``, the smallest attainable
            product of second moment and expected cost.

    References:
        [REF-RHEE-GLYNN-2015], [EQ-MLAB-RT-003].

    """

    probabilities: Vector
    second_moment: float
    expected_cost: float
    lower_bound: float

    @property
    def product(self) -> float:
        """Return second moment times expected cost.

        References:
            [REF-RHEE-GLYNN-2015], [EQ-MLAB-RT-003].

        """
        return self.second_moment * self.expected_cost

    @property
    def efficiency(self) -> float:
        """Return ``lower_bound / product`` in ``(0, 1]``.

        References:
            [REF-RHEE-GLYNN-2015], [EQ-MLAB-RT-003].

        """
        return self.lower_bound / self.product


def truncation_design(
    second_moments: Sequence[float],
    costs: Sequence[float],
    probabilities: Sequence[float] | None = None,
) -> TruncationDesign:
    """Evaluate a level law for the single-term estimator on finite levels.

    Args:
        second_moments: Positive ``m_n = E[Delta_n^2]`` per level.
        costs: Positive cost ``c_n`` per level.
        probabilities: Level law to evaluate; the optimal law of
            :func:`optimal_level_probabilities` when omitted.

    Returns:
        Second moment, expected cost, and the Cauchy-Schwarz lower bound of
        their product.

    Raises:
        ValueError: If the inputs are inconsistent or not positive.

    References:
        [REF-RHEE-GLYNN-2015], [EQ-MLAB-RT-003].

    """
    optimal = optimal_level_probabilities(second_moments, costs)
    law = (
        optimal if probabilities is None else finite_levels(probabilities).probabilities
    )
    if len(law) != len(optimal):
        raise ValueError("probabilities need one entry per level")
    moments = [float(m) for m in second_moments]
    work = [float(c) for c in costs]
    second = math.fsum(m / p for m, p in zip(moments, law, strict=True))
    cost = math.fsum(p * c for p, c in zip(law, work, strict=True))
    bound = math.fsum(math.sqrt(m * c) for m, c in zip(moments, work, strict=True)) ** 2
    return TruncationDesign(tuple(law), second, cost, bound)


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------


def extrapolation_weights(steps: Sequence[float], orders: Sequence[float]) -> Vector:
    """Return weights that cancel the leading terms of a bias expansion.

    If ``E[Y(h)] = mu + sum_k a_k h^(q_k) + (higher terms)`` with known
    exponents ``q_1 < ... < q_K`` and unknown coefficients ``a_k``, then for
    ``K + 1`` distinct step sizes ``h_0, ..., h_K`` the weights solving

    ``sum_i w_i = 1`` and ``sum_i w_i h_i^(q_k) = 0`` for ``k = 1, ..., K``

    give ``E[sum_i w_i Y(h_i)] = mu + (higher terms)``
    ([EQ-MLAB-RICH-001]). For one exponent ``q`` and steps ``h, h/r`` this is
    the classical Richardson pair ``(-1, r^q) / (r^q - 1)``.

    Args:
        steps: ``K + 1`` distinct positive step sizes.
        orders: ``K`` distinct positive exponents of the bias expansion.

    Returns:
        The ``K + 1`` weights, in the order of ``steps``.

    Raises:
        ValueError: If counts do not match, steps or orders repeat or are
            not positive, or the system is singular.

    References:
        [REF-ROACHE-1998, Chapters 4-5],
        [EQ-MLAB-RICH-001].

    """
    grid = tuple(require_finite_real(h, name="step") for h in steps)
    exponents = tuple(require_finite_real(q, name="order") for q in orders)
    if len(grid) != len(exponents) + 1:
        raise ValueError("steps needs exactly one more entry than orders")
    if any(h <= 0.0 for h in grid) or any(q <= 0.0 for q in exponents):
        raise ValueError("steps and orders must be positive")
    if len(set(grid)) != len(grid) or len(set(exponents)) != len(exponents):
        raise ValueError("steps and orders must be distinct")
    matrix: Matrix = (
        tuple(1.0 for _ in grid),
        *(tuple(h**q for h in grid) for q in exponents),
    )
    right_hand_side = (1.0, *(0.0 for _ in exponents))
    return lu_solve(matrix, right_hand_side)


def extrapolated_estimator(
    name: str,
    family: Callable[[float], Estimator],
    steps: Sequence[float],
    orders: Sequence[float],
    *,
    common_stream: bool = True,
) -> Estimator:
    """Derive the Richardson-extrapolated combination of a family.

    Args:
        name: Label of the derived estimator.
        family: ``family(h)`` returns the estimator with discretization
            parameter ``h`` whose bias has the expansion of
            :func:`extrapolation_weights`.
        steps: Step sizes ``h_0, ..., h_K``.
        orders: Known exponents ``q_1, ..., q_K`` of the bias expansion.
        common_stream: Draw every member from the same stream (common random
            numbers, which usually reduces the variance when the members are
            positively correlated) instead of independent child streams.

    Returns:
        Estimator ``sum_i w_i Y(h_i)`` whose bias starts at the first term
        *not* listed in ``orders``. With independent members its variance is
        ``sum_i w_i^2 Var[Y(h_i)]``, which exceeds each member's variance
        when weights are large; extrapolation trades bias for variance.

    Raises:
        ValueError: If the weights cannot be computed.

    References:
        [REF-ROACHE-1998, Chapters 4-5],
        [EQ-MLAB-RICH-001].

    """
    weights = extrapolation_weights(steps, orders)
    members = tuple(family(float(h)) for h in steps)

    def sampler(stream: RandomStream) -> Draw:
        """Combine one draw of every member with the extrapolation weights.

        References:
            [EQ-MLAB-RICH-001], [REF-ROACHE-1998, Chapters 4-5].

        """
        value = 0.0
        cost = 0.0
        for index, (weight, member) in enumerate(zip(weights, members, strict=True)):
            source = stream if common_stream else stream.child("member", index)
            draw = member.draw(source)
            value += weight * draw.value
            cost += draw.cost
        return Draw(value, cost)

    coupling = "common random numbers" if common_stream else "independent members"
    return Estimator(
        name,
        sampler,
        (
            f"Richardson extrapolation over {len(members)} step sizes cancelling "
            f"orders {tuple(float(q) for q in orders)} ({coupling})",
        ),
    )


# ---------------------------------------------------------------------------
# Linear combinations: BLUE and control variates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearCombination:
    """Weights of a linear combination and the variance they predict.

    Args:
        weights: Coefficient of each component.
        variance: Variance of the combination under the covariance used.
        covariance: The covariance matrix the weights were computed from.

    References:
        [REF-SCHADEN-ULLMANN-2020], [EQ-MLAB-BLUE-001].

    """

    weights: Vector
    variance: float
    covariance: Matrix


def best_linear_unbiased_weights(covariance: Matrix) -> LinearCombination:
    """Return the minimum-variance weights for unbiased estimators of one mean.

    For estimators ``Z_1, ..., Z_k`` with common expectation ``mu`` and
    positive-definite covariance ``Sigma``, the unbiased linear combination
    ``sum_i w_i Z_i`` (``sum_i w_i = 1``) of smallest variance has

    ``w = Sigma^-1 1 / (1' Sigma^-1 1)`` and variance ``1 / (1' Sigma^-1 1)``

    ([EQ-MLAB-BLUE-001]), the generalized least-squares estimator of a
    constant mean. The variance never exceeds that of any single component.
    With *estimated* covariance the combination stays unbiased only if the
    weights are independent of the combined draws; use a separate pilot.

    Args:
        covariance: Symmetric positive-definite ``k x k`` covariance matrix.

    Returns:
        Weights summing to one and the variance of the combination.

    Raises:
        ValueError: If the matrix is not positive definite.

    References:
        [REF-SCHADEN-ULLMANN-2020] (best linear unbiased estimators),
        [REF-MONTGOMERY-PECK-VINING-2021] (generalized least squares),
        [EQ-MLAB-BLUE-001].

    """
    size = len(covariance)
    if size == 0 or any(len(row) != size for row in covariance):
        raise ValueError("covariance must be a non-empty square matrix")
    solution = solve_spd(covariance, tuple(1.0 for _ in range(size)))
    total = math.fsum(solution)
    if total <= 0.0:
        raise ValueError("covariance must be positive definite")
    return LinearCombination(
        tuple(s / total for s in solution), 1.0 / total, covariance
    )


def _joint_draws(
    estimators: Sequence[Estimator],
    count: int,
    root: RandomStream,
    label: str,
    shared: bool,
) -> list[list[Draw]]:
    """Return ``count`` joint draws of several estimators.

    References:
        [REF-GLASSERMAN-2003, Section 4.1] (common random numbers within a
        joint draw). No equation is evaluated.

    """
    rows: list[list[Draw]] = []
    for index in range(count):
        base = root.child(label, index)
        rows.append(
            [
                member.draw(base if shared else base.child("member", position))
                for position, member in enumerate(estimators)
            ]
        )
    return rows


def _sample_covariance(rows: Sequence[Sequence[float]]) -> Matrix:
    """Return the unbiased sample covariance matrix of row vectors.

    References:
        [EQ-STAT-006], [REF-CASELLA-BERGER-2002, Chapter 4].

    """
    count = len(rows)
    width = len(rows[0])
    means = [math.fsum(row[j] for row in rows) / count for j in range(width)]
    return tuple(
        tuple(
            math.fsum((row[i] - means[i]) * (row[j] - means[j]) for row in rows)
            / (count - 1)
            for j in range(width)
        )
        for i in range(width)
    )


def pilot_covariance(
    estimators: Sequence[Estimator],
    pilot_size: int,
    seed: int | RandomStream,
    *,
    shared_stream: bool = True,
) -> Matrix:
    """Estimate the covariance matrix of jointly drawn estimators.

    Args:
        estimators: At least one estimator.
        pilot_size: Number of joint pilot draws, larger than the number of
            estimators so that the matrix can be positive definite.
        seed: Root seed or stream; pilot draw ``i`` uses child
            ``("pilot", i)``.
        shared_stream: Give every estimator the same stream within a joint
            draw (common random numbers) instead of separate child streams.

    Returns:
        Unbiased sample covariance matrix.

    Raises:
        ValueError: If there are no estimators or too few pilot draws.

    References:
        [EQ-STAT-006], [REF-GLASSERMAN-2003, Section 4.1].

    """
    members = tuple(estimators)
    if not members:
        raise ValueError("at least one estimator is required")
    size = require_non_negative_integer(pilot_size, name="pilot_size")
    if size <= len(members):
        raise ValueError("pilot_size must exceed the number of estimators")
    rows = _joint_draws(members, size, resolve_stream(seed), "pilot", shared_stream)
    return _sample_covariance([[draw.value for draw in row] for row in rows])


def combine_unbiased(
    name: str,
    estimators: Sequence[Estimator],
    *,
    pilot_size: int,
    seed: int | RandomStream,
    shared_stream: bool = True,
) -> tuple[Estimator, LinearCombination]:
    """Derive the best linear unbiased combination from a pilot run.

    The covariance of the jointly drawn estimators is estimated from
    ``pilot_size`` pilot draws, the weights of
    :func:`best_linear_unbiased_weights` are frozen, and the derived
    estimator applies them to fresh joint draws. Because the weights do not
    depend on the draws they multiply, the combination is exactly unbiased
    whenever every component is; only its variance depends on the quality of
    the pilot.

    Args:
        name: Label of the derived estimator.
        estimators: Estimators of the same quantity (each assumed unbiased).
        pilot_size: Number of joint pilot draws.
        seed: Seed of the pilot. Use a different seed for later evaluation.
        shared_stream: Whether members of a joint draw share one stream.

    Returns:
        The derived estimator and the frozen weights with their predicted
        variance.

    Raises:
        ValueError: If the pilot covariance is not positive definite.

    References:
        [REF-SCHADEN-ULLMANN-2020] (best linear unbiased
        combination of correlated model evaluations), [EQ-MLAB-BLUE-001].

    """
    members = tuple(estimators)
    covariance = pilot_covariance(
        members, pilot_size, seed, shared_stream=shared_stream
    )
    combination = best_linear_unbiased_weights(covariance)

    def sampler(stream: RandomStream) -> Draw:
        """Apply the frozen weights to one fresh joint draw.

        References:
            [EQ-MLAB-BLUE-001], [REF-SCHADEN-ULLMANN-2020].

        """
        draws = [
            member.draw(stream if shared_stream else stream.child("member", i))
            for i, member in enumerate(members)
        ]
        return Draw(
            math.fsum(
                w * d.value for w, d in zip(combination.weights, draws, strict=True)
            ),
            math.fsum(d.cost for d in draws),
        )

    step = (
        f"best linear unbiased combination of {len(members)} estimators "
        f"(weights from {pilot_size} pilot draws)"
    )
    return Estimator(name, sampler, (step,)), combination


def control_variate_estimator(
    name: str,
    target: Estimator,
    controls: Sequence[Estimator],
    control_means: Sequence[float],
    *,
    pilot_size: int,
    seed: int | RandomStream,
) -> tuple[Estimator, Vector]:
    """Derive a control-variate estimator with pilot-estimated coefficients.

    With target ``Y``, controls ``C`` of known mean ``nu``, and
    ``beta = Var[C]^-1 Cov[C, Y]``,

    ``Z = Y - beta' (C - nu)``

    has expectation ``E[Y]`` for every fixed ``beta`` and variance
    ``(1 - R^2) Var[Y]`` at the optimal ``beta`` ([EQ-MLAB-CV-001]), where
    ``R`` is the multiple correlation of ``Y`` with ``C``. The coefficients
    are estimated once from pilot draws and then frozen, so the derived
    estimator is exactly unbiased (coefficients estimated from the same
    draws would add a bias of order ``1/n``).

    Args:
        name: Label of the derived estimator.
        target: Estimator of the quantity of interest.
        controls: Estimators with known expectations, drawn on the same
            stream as the target so that they are correlated with it.
        control_means: The known expectations ``nu``.
        pilot_size: Number of joint pilot draws for the coefficients.
        seed: Seed of the pilot.

    Returns:
        The derived estimator and the frozen coefficient vector ``beta``.

    Raises:
        ValueError: If lengths differ or the control covariance is singular.

    References:
        [REF-GLASSERMAN-2003, Section 4.1], [REF-OWEN-2013, Chapter 8],
        [EQ-MLAB-CV-001].

    """
    members = tuple(controls)
    means = tuple(require_finite_real(m, name="control mean") for m in control_means)
    if not members or len(members) != len(means):
        raise ValueError("controls and control_means need one entry per control")
    covariance = pilot_covariance(
        (target, *members), pilot_size, seed, shared_stream=True
    )
    control_block: Matrix = tuple(row[1:] for row in covariance[1:])
    cross: Vector = tuple(row[0] for row in covariance[1:])
    beta = solve_spd(control_block, cross)

    def sampler(stream: RandomStream) -> Draw:
        """Subtract the frozen linear correction from one target draw.

        References:
            [EQ-MLAB-CV-001], [REF-GLASSERMAN-2003, Section 4.1].

        """
        base = target.draw(stream)
        draws = [member.draw(stream) for member in members]
        correction = math.fsum(
            b * (d.value - m) for b, d, m in zip(beta, draws, means, strict=True)
        )
        return Draw(
            base.value - correction, base.cost + math.fsum(d.cost for d in draws)
        )

    step = (
        f"{len(members)} control variate(s) with coefficients frozen from "
        f"{pilot_size} pilot draws"
    )
    return target.derived(name, sampler, step), beta


def thinned_estimator(name: str, estimator: Estimator, probability: float) -> Estimator:
    """Derive an estimator that is evaluated only with a given probability.

    ``Z = B Z' / q`` with ``B`` a Bernoulli(``q``) indicator independent of
    ``Z'``; ``Z'`` is drawn (and paid for) only when ``B = 1``. Then
    ``E[Z] = E[Z']``, ``Var[Z] = E[Z'^2] / q - E[Z']^2``, and the expected
    cost is ``q E[cost(Z')]`` ([EQ-MLAB-THIN-001]). It is the single-term
    construction of [EQ-MLAB-RT-001] on a two-point level law whose first
    difference is zero. Thinning pays when ``Z'`` is an expensive correction
    with a small second moment: added to a base term of variance ``V`` and
    cost ``c``, the product of total variance and total cost is smallest
    near ``q = sqrt((E[Z'^2] / E[cost(Z')]) / (V / c))`` (capped at one).

    Args:
        name: Label of the derived estimator.
        estimator: The estimator ``Z'`` to thin.
        probability: Evaluation probability ``q`` in ``(0, 1]``.

    Returns:
        Derived estimator. The indicator uses child stream ``"thin"`` and
        the draw of ``Z'`` child stream ``"draw"``.

    Raises:
        ValueError: If ``probability`` is not in ``(0, 1]``.

    References:
        [REF-RHEE-GLYNN-2015] (randomization of a term with an inverse
        probability weight), [EQ-MLAB-THIN-001].

    """
    chance = require_finite_real(probability, name="probability")
    if not 0.0 < chance <= 1.0:
        raise ValueError("probability must lie in (0, 1]")

    def sampler(stream: RandomStream) -> Draw:
        """Draw the indicator, then the weighted estimator if selected.

        References:
            [EQ-MLAB-THIN-001], [REF-RHEE-GLYNN-2015].

        """
        if stream.child("thin").generator().random() >= chance:
            return Draw(0.0, 0.0)
        draw = estimator.draw(stream.child("draw"))
        return Draw(draw.value / chance, draw.cost)

    return estimator.derived(
        name, sampler, f"evaluated with probability {chance:.6g}, weighted 1/q"
    )


def sum_of_estimators(name: str, estimators: Sequence[Estimator]) -> Estimator:
    """Derive the sum of independently drawn estimators.

    ``Z = sum_i Z_i`` with member ``i`` drawn on the child stream
    ``("term", i)``. By linearity ``E[Z] = sum_i E[Z_i]``, and because the
    members are independent ``Var[Z] = sum_i Var[Z_i]``; the cost is the sum
    of the costs ([EQ-MLAB-SUM-001]). The typical use is a telescoping
    split: an inexpensive base approximation estimated directly plus a
    randomized-truncation estimator of the remaining correction, which keeps
    the large base term out of the importance-weighted part.

    Args:
        name: Label of the derived estimator.
        estimators: At least one estimator.

    Returns:
        Derived estimator of the sum of the members' expectations.

    Raises:
        ValueError: If no estimator is given.

    References:
        [REF-RHEE-GLYNN-2015] (independent-sum construction),
        [REF-GILES-2015] (telescoping sum of independent level estimators),
        [EQ-MLAB-SUM-001].

    """
    members = tuple(estimators)
    if not members:
        raise ValueError("at least one estimator is required")

    def sampler(stream: RandomStream) -> Draw:
        """Add one independent draw of every member.

        References:
            [EQ-MLAB-SUM-001], [REF-GILES-2015].

        """
        draws = [m.draw(stream.child("term", i)) for i, m in enumerate(members)]
        return Draw(math.fsum(d.value for d in draws), math.fsum(d.cost for d in draws))

    lineage: list[str] = []
    for member in members:
        lineage.extend(f"[{member.name}] {step}" for step in member.lineage)
    lineage.append("sum of independent terms: " + " + ".join(m.name for m in members))
    return Estimator(name, sampler, tuple(lineage))


def antithetic_estimator(
    name: str,
    function: Callable[[Sequence[float]], float],
    dimension: int,
    *,
    cost: float = 1.0,
) -> Estimator:
    """Derive the antithetic-pair estimator of an integral over the unit cube.

    ``Z = (f(U) + f(1 - U)) / 2`` with ``U`` uniform on ``[0, 1]^d`` is
    unbiased for ``integral f`` and has variance
    ``(1 + rho) Var[f(U)] / 2`` with ``rho = Corr[f(U), f(1 - U)]``
    ([EQ-MLAB-ANTI-001]). It beats two independent draws exactly when
    ``rho < 0``, which holds for every ``f`` monotone in each coordinate.

    Args:
        name: Label of the derived estimator.
        function: Integrand on the unit cube.
        dimension: Positive dimension ``d``.
        cost: Cost of one evaluation of ``function``; a draw costs twice it.

    Returns:
        Derived estimator.

    Raises:
        ValueError: If ``dimension`` is not positive.

    References:
        [REF-GLASSERMAN-2003, Section 4.2],
        [EQ-MLAB-ANTI-001].

    """
    size = require_non_negative_integer(dimension, name="dimension")
    if size == 0:
        raise ValueError("dimension must be positive")
    unit = require_finite_real(cost, name="cost")

    def sampler(stream: RandomStream) -> Draw:
        """Average the integrand at a uniform point and its reflection.

        References:
            [EQ-MLAB-ANTI-001], [REF-GLASSERMAN-2003, Section 4.2].

        """
        generator = stream.generator()
        point = [generator.random() for _ in range(size)]
        mirror = [1.0 - u for u in point]
        return Draw(0.5 * (float(function(point)) + float(function(mirror))), 2 * unit)

    return Estimator(name, sampler, ("antithetic pair f(U), f(1 - U)",))


# ---------------------------------------------------------------------------
# Data-based bias corrections
# ---------------------------------------------------------------------------


def jackknife_bias_corrected(
    statistic: Callable[[Sequence[T]], float],
) -> Callable[[Sequence[T]], float]:
    """Derive the jackknife bias-corrected version of a statistic.

    For a statistic ``theta_hat`` of ``n`` observations with leave-one-out
    values ``theta_(i)``,

    ``theta_J = n theta_hat - (n - 1) mean_i theta_(i)``

    ([EQ-MLAB-JACK-001]). If the bias has the expansion
    ``a_1 / n + a_2 / n^2 + ...`` the corrected statistic has bias
    ``O(1/n^2)``; it is *exactly* unbiased when the bias is exactly
    ``a_1 / n`` (the plug-in variance becomes the unbiased sample variance).
    The correction fails for non-smooth statistics such as the median and
    costs ``n + 1`` evaluations of the statistic.

    Args:
        statistic: Function of a sample (any sequence of observations).

    Returns:
        A function of a sample of at least two observations that evaluates
        the corrected statistic.

    References:
        [REF-QUENOUILLE-1956], [REF-EFRON-TIBSHIRANI-1993],
        [EQ-MLAB-JACK-001].

    """
    if not callable(statistic):
        raise TypeError("statistic must be callable")

    def corrected(data: Sequence[T]) -> float:
        """Evaluate the jackknife bias-corrected statistic.

        Raises:
            ValueError: If fewer than two observations are given.

        References:
            [EQ-MLAB-JACK-001], [REF-QUENOUILLE-1956].

        """
        sample = tuple(data)
        size = len(sample)
        if size < 2:
            raise ValueError("at least two observations are required")
        full = float(statistic(sample))
        leave_one_out = math.fsum(
            float(statistic(sample[:i] + sample[i + 1 :])) for i in range(size)
        )
        return size * full - (size - 1) * leave_one_out / size

    return corrected


def bootstrap_bias_corrected(
    statistic: Callable[[Sequence[T]], float],
    *,
    resamples: int,
    seed: int | RandomStream,
) -> Callable[[Sequence[T]], float]:
    """Derive the bootstrap bias-corrected version of a statistic.

    ``theta_B = 2 theta_hat - mean_b theta*_b`` where ``theta*_b`` is the
    statistic on the ``b``-th resample drawn with replacement
    ([EQ-MLAB-BOOT-001]). The bootstrap estimate of bias,
    ``mean_b theta*_b - theta_hat``, is itself random: the correction
    removes the ``O(1/n)`` bias term for smooth statistics but can increase
    the mean squared error when the bias is small against the standard
    error.

    Args:
        statistic: Function of a sample.
        resamples: Positive number of bootstrap resamples.
        seed: Seed of the resampling; the derived function is deterministic
            given the data.

    Returns:
        A function of a sample that evaluates the corrected statistic.

    Raises:
        ValueError: If ``resamples`` is not positive.

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapter 10], [EQ-MLAB-BOOT-001].

    """
    if not callable(statistic):
        raise TypeError("statistic must be callable")
    count = require_non_negative_integer(resamples, name="resamples")
    if count == 0:
        raise ValueError("resamples must be positive")
    root = resolve_stream(seed)

    def corrected(data: Sequence[T]) -> float:
        """Evaluate the bootstrap bias-corrected statistic.

        Raises:
            ValueError: If the sample is empty.

        References:
            [EQ-MLAB-BOOT-001], [REF-EFRON-TIBSHIRANI-1993, Chapter 10].

        """
        sample = tuple(data)
        size = len(sample)
        if size == 0:
            raise ValueError("at least one observation is required")
        full = float(statistic(sample))
        total = 0.0
        for index in range(count):
            generator = root.child("resample", index).generator()
            resample = tuple(sample[generator.randrange(size)] for _ in range(size))
            total += float(statistic(resample))
        return 2.0 * full - total / count

    return corrected


# ---------------------------------------------------------------------------
# Influence functions
# ---------------------------------------------------------------------------

WeightedFunctional = Callable[[Sequence[T], Sequence[float]], float]


@dataclass(frozen=True, slots=True)
class InfluenceAnalysis:
    """Empirical influence values of a functional and what follows from them.

    Args:
        estimate: The functional at the empirical distribution.
        influence: Empirical influence value of each observation.
        standard_error: Infinitesimal-jackknife standard error
            ``sqrt(sum_i I_i^2) / n``.
        epsilon: Contamination step used for the directional derivative.

    References:
        [REF-HAMPEL-1974], [REF-EFRON-TIBSHIRANI-1993], [EQ-MLAB-IF-001],
        [EQ-MLAB-IF-002].

    """

    estimate: float
    influence: Vector
    standard_error: float
    epsilon: float

    @property
    def mean_influence(self) -> float:
        """Return the mean of the influence values (zero in exact calculus).

        A value far from zero relative to ``standard_error`` signals that
        the functional is not smooth at the empirical distribution or that
        ``epsilon`` is too large. The converse does not hold: for a weighted
        median the mean influence came out as zero while the standard error
        was meaningless (see :func:`empirical_influence`).

        References:
            [REF-HAMPEL-1974], [EQ-MLAB-IF-001].

        """
        return math.fsum(self.influence) / len(self.influence)


def empirical_influence(
    functional: WeightedFunctional[T],
    data: Sequence[T],
    *,
    epsilon: float = 1e-4,
) -> InfluenceAnalysis:
    """Differentiate a weighted functional to get influence values.

    For a functional ``T(P)`` evaluated on weighted samples, the empirical
    influence value of observation ``x_i`` is the Gateaux derivative

    ``I_i = d/d eps T((1 - eps) P_n + eps delta_(x_i)) at eps = 0``

    ([EQ-MLAB-IF-001]), computed here by a central difference in ``eps``
    (negative ``eps`` is admissible because the perturbed weights stay
    positive for ``eps < 1/n``). The infinitesimal-jackknife standard error
    of ``T(P_n)`` is ``sqrt(sum_i I_i^2) / n`` ([EQ-MLAB-IF-002]). This
    derives a standard error for any new smooth estimator that can be
    written as a function of observation weights, without an analytical
    derivation; it is first-order only and fails for non-differentiable
    functionals such as sample quantiles. The failure is silent: for a
    weighted median the returned standard error is proportional to
    ``1 / epsilon`` (129 at the default step and 1.3e4 at ``1e-6`` for 20
    lognormal observations). For smooth functionals (standard deviation,
    ratio of moments, correlation) the result changed by less than ``1e-5``
    relative between ``epsilon = 1e-8`` and ``1e-4``, and by up to 10% when
    ``epsilon`` approaches ``1/n``; repeating the call with a second step
    size is a cheap check of smoothness. The default step requires
    ``n < 10 000``; pass a smaller ``epsilon`` for larger samples.

    Args:
        functional: ``functional(data, weights)`` with non-negative weights
            summing to one; equal weights give the usual estimate.
        data: At least two observations.
        epsilon: Contamination step, positive and below ``1/n``.

    Returns:
        Estimate, influence values, and standard error.

    Raises:
        ValueError: If the sample is too small or ``epsilon`` is out of
            range.

    References:
        [REF-HAMPEL-1974] (influence curve), [REF-EFRON-TIBSHIRANI-1993]
        (infinitesimal jackknife), [REF-CARONE-LUEDTKE-VANDERLAAN-2019]
        (numerical differentiation of statistical functionals),
        [EQ-MLAB-IF-001], [EQ-MLAB-IF-002].

    """
    sample = tuple(data)
    size = len(sample)
    if size < 2:
        raise ValueError("at least two observations are required")
    step = require_finite_real(epsilon, name="epsilon")
    if not 0.0 < step < 1.0 / size:
        raise ValueError("epsilon must be positive and below 1/n")
    base = tuple(1.0 / size for _ in range(size))
    estimate = float(functional(sample, base))
    influence: list[float] = []
    for index in range(size):
        plus = [(1.0 - step) / size] * size
        minus = [(1.0 + step) / size] * size
        plus[index] += step
        minus[index] -= step
        derivative = (
            float(functional(sample, plus)) - float(functional(sample, minus))
        ) / (2.0 * step)
        influence.append(derivative)
    standard_error = math.sqrt(math.fsum(v * v for v in influence)) / size
    return InfluenceAnalysis(estimate, tuple(influence), standard_error, step)


@dataclass(frozen=True, slots=True)
class OneStepEstimate:
    """A plug-in estimate corrected by the mean of an influence function.

    Args:
        plug_in: The initial plug-in estimate.
        correction: Sample mean of the influence-function values.
        estimate: ``plug_in + correction``.
        standard_error: ``sd(influence) / sqrt(n)``.

    References:
        [REF-KENNEDY-2022], [EQ-MLAB-ONESTEP-001].

    """

    plug_in: float
    correction: float
    estimate: float
    standard_error: float


def one_step_estimator(
    plug_in: float, influence_values: Sequence[float]
) -> OneStepEstimate:
    """Correct a plug-in estimate by the mean of its influence function.

    Given a plug-in estimate ``psi(P_hat)`` and the values
    ``phi(x_i; P_hat)`` of the (efficient) influence function of the target
    functional evaluated at the fitted nuisance ``P_hat``,

    ``psi_1 = psi(P_hat) + (1/n) sum_i phi(x_i; P_hat)``

    ([EQ-MLAB-ONESTEP-001]) removes the first-order bias of the plug-in.
    Under the conditions of the cited review (nuisance errors whose product
    is ``o(n^-1/2)``, and sample splitting or a Donsker condition) it is
    asymptotically normal with standard error ``sd(phi) / sqrt(n)``. Those
    conditions concern how ``P_hat`` was fitted and cannot be checked from
    the influence values alone.

    Args:
        plug_in: Initial estimate ``psi(P_hat)``.
        influence_values: ``phi(x_i; P_hat)`` for each observation, at least
            two.

    Returns:
        The corrected estimate with its plug-in, correction, and standard
        error.

    Raises:
        ValueError: If fewer than two influence values are given.

    References:
        [REF-KENNEDY-2022] (one-step bias-corrected estimator),
        [REF-HINES-ETAL-2022], [EQ-MLAB-ONESTEP-001].

    """
    initial = require_finite_real(plug_in, name="plug_in")
    values = tuple(
        require_finite_real(v, name="influence value") for v in influence_values
    )
    size = len(values)
    if size < 2:
        raise ValueError("at least two influence values are required")
    mean = math.fsum(values) / size
    variance = math.fsum((v - mean) ** 2 for v in values) / (size - 1)
    return OneStepEstimate(initial, mean, initial + mean, math.sqrt(variance / size))
