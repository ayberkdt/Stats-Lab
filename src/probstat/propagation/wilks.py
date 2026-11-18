"""Distribution-free guarantees on output quantiles from order statistics.

A computer code is run ``n`` times on independent draws of its inputs and the
outputs are sorted, ``Y_(1) <= ... <= Y_(n)``. Whatever the (continuous) law
of the output, the number of outputs below the ``a``-quantile ``q_a`` is
``Binomial(n, a)``, so the probability that a given order statistic lies
above (or below) ``q_a`` is a binomial sum that does not depend on the law.
Wilks (1941, 1942) used this to choose a rank, or a number of runs, that
guarantees a quantile with a stated confidence: ``Y_(k)`` is an upper bound of
``q_a`` with probability at least ``beta``.

The module answers four questions:

- ``wilks_rank``: which order statistic of ``n`` values bounds ``q_a`` with
  confidence ``beta``?
- ``wilks_sample_size``: how many runs are needed so that the ``m``-th
  largest (or smallest) value is such a bound? The classical 95 %/95 %
  answers are 59, 93, 124, and 153 runs for ``m = 1..4``.
- ``wilks_bound``: the bound itself for an observed sample.
- ``wilks_tolerance_interval``: a two-sided interval ``[Y_(m), Y_(n-m+1)]``
  that contains at least a fraction ``gamma`` of the output law with
  confidence ``beta``.

Relation to other tools: :func:`probstat.propagation.order_statistic_interval`
is a *two-sided, equal-tailed* confidence interval for ``q_a`` (each tail
carries ``(1 - confidence)/2``); a Wilks bound is *one-sided* and puts the
whole risk ``1 - beta`` in one tail, so its rank is closer to ``n a``. The
first-order sample size is the success-run theorem: ``wilks_sample_size(a,
beta, order=m)`` equals
:func:`probstat.reliability.success_run_sample_size` with reliability ``a``,
confidence ``beta``, and ``m - 1`` allowed failures.

Assumptions: the values are independent draws of one law. The confidence is
exact for a continuous law; with atoms the one-sided bounds remain
conservative (their confidence is at least the stated value). Latin
hypercube and quasi-random designs are not independent draws.

References:
    [REF-WILKS-1941], [REF-WILKS-1942], [REF-NUTT-WALLIS-2004],
    [REF-DAVID-NAGARAJA-2003, Sections 7.1-7.2].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.distributions import Binomial
from probstat.foundations.validation import require_finite_real
from probstat.statistics import quantile

SIDES = ("upper", "lower")
SAMPLE_SIZE_SIDES = ("upper", "lower", "two-sided")


@dataclass(frozen=True, slots=True)
class WilksRank:
    """The order statistic that bounds a quantile with a stated confidence.

    Args:
        size: Number of values ``n``.
        probability: Quantile level ``a``.
        confidence: Requested confidence ``beta``.
        sided: ``"upper"`` for an upper bound of ``q_a``, ``"lower"`` for a
            lower bound.
        rank: One-based rank ``k`` of the bounding order statistic
            ``Y_(k)`` in increasing order.
        order: Position counted from the bounded side: ``n - k + 1`` for an
            upper bound (1 is the maximum), ``k`` for a lower bound (1 is the
            minimum).
        achieved_confidence: Exact ``P(Y_(k) >= q_a)`` (upper) or
            ``P(Y_(k) <= q_a)`` (lower) for a continuous law; at least
            ``confidence``.

    References:
        [REF-WILKS-1941], [REF-DAVID-NAGARAJA-2003, Section 7.1].

    """

    size: int
    probability: float
    confidence: float
    sided: str
    rank: int
    order: int
    achieved_confidence: float


@dataclass(frozen=True, slots=True)
class WilksBound:
    """A one-sided distribution-free confidence bound for one quantile.

    Args:
        bound: The order statistic ``Y_(k)``.
        rank: One-based rank ``k`` in increasing order.
        order: Position counted from the bounded side (1 is the extreme).
        achieved_confidence: Exact confidence of the rank for a continuous
            law (at least ``confidence``).
        empirical_quantile: Type-7 sample quantile of level ``probability``,
            the point estimate the bound guards.
        size: Number of values ``n``.
        probability: Quantile level ``a``.
        confidence: Requested confidence ``beta``.
        sided: ``"upper"`` or ``"lower"``.

    References:
        [REF-WILKS-1941], [REF-HYNDMAN-FAN-1996].

    """

    bound: float
    rank: int
    order: int
    achieved_confidence: float
    empirical_quantile: float
    size: int
    probability: float
    confidence: float
    sided: str


@dataclass(frozen=True, slots=True)
class WilksToleranceInterval:
    """A two-sided distribution-free tolerance interval.

    The interval ``[Y_(m), Y_(n-m+1)]`` contains at least a fraction
    ``coverage`` of the sampled law with probability ``achieved_confidence``.

    Args:
        lower: Order statistic ``Y_(m)``.
        upper: Order statistic ``Y_(n-m+1)``.
        lower_rank: One-based rank ``m``.
        upper_rank: One-based rank ``n - m + 1``.
        order: Number ``m`` of values at or beyond each end.
        coverage: Required content ``gamma`` of the law.
        confidence: Requested confidence ``beta``.
        achieved_confidence: Exact ``P(F(Y_(n-m+1)) - F(Y_(m)) >= gamma)``
            for a continuous law ``F`` (at least ``confidence``).
        size: Number of values ``n``.

    References:
        [REF-WILKS-1942], [REF-DAVID-NAGARAJA-2003, Section 7.2].

    """

    lower: float
    upper: float
    lower_rank: int
    upper_rank: int
    order: int
    coverage: float
    confidence: float
    achieved_confidence: float
    size: int


def _level(value: float, *, name: str) -> float:
    """Return a finite level strictly inside ``(0, 1)``.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is outside ``(0, 1)``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    level = require_finite_real(value, name=name)
    if not 0.0 < level < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return level


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer, rejecting booleans.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is smaller than one.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least one")
    return value


def _upper_confidence(size: int, probability: float, rank: int) -> float:
    """Return ``P(Y_(k) >= q_a) = P(Binomial(n, a) <= k - 1)``.

    References:
        [REF-WILKS-1941]. Evaluates [EQ-PROP-WILKS-001].

    """
    return float(Binomial(size, probability).cdf(rank - 1))


def _lower_confidence(size: int, probability: float, rank: int) -> float:
    """Return ``P(Y_(k) <= q_a) = P(Binomial(n, a) >= k)``.

    References:
        [REF-WILKS-1941]. Evaluates [EQ-PROP-WILKS-001].

    """
    return float(Binomial(size, probability).sf(rank - 1))


def wilks_rank(
    size: int, probability: float, confidence: float, *, sided: str = "upper"
) -> WilksRank:
    """Return the order statistic of ``n`` values that bounds ``q_a``.

    For an upper bound ``k`` is the smallest rank with
    ``sum_(j=0)^(k-1) C(n, j) a^j (1-a)^(n-j) >= beta``, i.e.
    ``P(Binomial(n, a) <= k - 1) >= beta``, so that ``P(Y_(k) >= q_a) >= beta``
    for every continuous law. For a lower bound ``k`` is the largest rank
    with ``P(Binomial(n, a) >= k) >= beta``. The binomial tails are
    regularized incomplete beta functions and ``k`` is found by bisection, so
    ``n`` may be large.

    Unlike :func:`probstat.propagation.order_statistic_interval`, which
    splits the risk equally between two tails of a confidence interval, the
    whole risk ``1 - beta`` sits on one side here.

    Args:
        size: Number of values ``n`` (positive integer).
        probability: Quantile level ``a`` strictly inside ``(0, 1)``.
        confidence: Confidence ``beta`` strictly inside ``(0, 1)``.
        sided: ``"upper"`` or ``"lower"``.

    Returns:
        The rank, its order from the bounded side, and the exact achieved
        confidence.

    Raises:
        TypeError: If ``size`` is not an integer or a level is not real.
        ValueError: If a level is outside ``(0, 1)``, ``sided`` is unknown,
            or ``n`` values are too few for the requested guarantee (the
            message gives the minimal size).

    Equation:
        [EQ-PROP-WILKS-001].

    References:
        [REF-WILKS-1941], [REF-DAVID-NAGARAJA-2003, Section 7.1].

    """
    n = _positive_integer(size, name="size")
    a = _level(probability, name="probability")
    beta = _level(confidence, name="confidence")
    if sided not in SIDES:
        raise ValueError(f"sided must be one of {SIDES}")
    if sided == "upper":
        if _upper_confidence(n, a, n) < beta:
            needed = wilks_sample_size(a, beta, sided="upper")
            raise ValueError(
                f"{n} values are too few for a {beta:g} upper bound of the "
                f"{a:g} quantile; at least {needed} are needed"
            )
        low, high = 0, n  # confidence(low) < beta <= confidence(high)
        while high - low > 1:
            middle = (low + high) // 2
            if middle >= 1 and _upper_confidence(n, a, middle) >= beta:
                high = middle
            else:
                low = middle
        achieved = _upper_confidence(n, a, high)
        return WilksRank(n, a, beta, sided, high, n - high + 1, achieved)
    if _lower_confidence(n, a, 1) < beta:
        needed = wilks_sample_size(a, beta, sided="lower")
        raise ValueError(
            f"{n} values are too few for a {beta:g} lower bound of the "
            f"{a:g} quantile; at least {needed} are needed"
        )
    low, high = 1, n + 1  # confidence(low) >= beta > confidence(high)
    while high - low > 1:
        middle = (low + high) // 2
        if _lower_confidence(n, a, middle) >= beta:
            low = middle
        else:
            high = middle
    achieved = _lower_confidence(n, a, low)
    return WilksRank(n, a, beta, sided, low, low, achieved)


def wilks_sample_size(
    probability: float,
    confidence: float,
    *,
    order: int = 1,
    sided: str = "upper",
) -> int:
    """Return the smallest number of runs for an order-``m`` Wilks guarantee.

    - ``"upper"``: the smallest ``n`` with ``P(Binomial(n, a) <= n - m) >=
      beta``, so that the ``m``-th largest value bounds ``q_a`` from above
      with confidence ``beta``. For ``m = 1`` this is ``1 - a^n >= beta``,
      ``n = ceil(ln(1 - beta) / ln a)``.
    - ``"lower"``: the smallest ``n`` with ``P(Binomial(n, a) >= m) >=
      beta`` (the ``m``-th smallest value bounds ``q_a`` from below).
    - ``"two-sided"``: ``probability`` is the content ``gamma`` and the
      interval ``[Y_(m), Y_(n-m+1)]`` must contain a fraction ``gamma`` of
      the law with confidence ``beta``. Its content is
      ``Beta(n - 2m + 1, 2m)``, so the condition is
      ``P(Binomial(n, gamma) <= n - 2m) >= beta``: a two-sided order ``m``
      costs the same as a one-sided order ``2m``.

    Each confidence increases with ``n``, so the answer is found by doubling
    and bisection. The one-sided upper case equals the success-run sample
    size :func:`probstat.reliability.success_run_sample_size` with
    reliability ``a`` and ``m - 1`` allowed failures. With
    ``a = beta = 0.95`` the upper sizes are 59, 93, 124, and 153 for
    ``m = 1..4`` and the first-order two-sided size is 93.

    Args:
        probability: Quantile level ``a`` (or content ``gamma`` for
            ``"two-sided"``) strictly inside ``(0, 1)``.
        confidence: Confidence ``beta`` strictly inside ``(0, 1)``.
        order: Order ``m`` of the bounding statistic (positive integer).
        sided: ``"upper"``, ``"lower"``, or ``"two-sided"``.

    Returns:
        Minimal number of independent runs.

    Raises:
        TypeError: If ``order`` is not an integer or a level is not real.
        ValueError: If a level is outside ``(0, 1)`` or ``sided`` is unknown.

    Equation:
        [EQ-PROP-WILKS-002], [EQ-PROP-WILKS-003].

    References:
        [REF-WILKS-1941], [REF-WILKS-1942], [REF-NUTT-WALLIS-2004].

    """
    a = _level(probability, name="probability")
    beta = _level(confidence, name="confidence")
    m = _positive_integer(order, name="order")
    if sided not in SAMPLE_SIZE_SIDES:
        raise ValueError(f"sided must be one of {SAMPLE_SIZE_SIDES}")
    if sided == "lower":
        # P(Bin(n, a) >= m) = P(Bin(n, 1 - a) <= n - m): the upper case at 1 - a.
        a, sided = 1.0 - a, "upper"
    elif sided == "two-sided":
        m, sided = 2 * m, "upper"

    def passes(n: int) -> bool:
        """Return whether ``n`` runs give the order-``m`` guarantee.

        References:
            [EQ-PROP-WILKS-002]. No new equation.

        """
        return float(Binomial(n, a).cdf(n - m)) >= beta

    if m == 1:
        guess = max(1, math.ceil(math.log1p(-beta) / math.log(a)))
        while guess > 1 and passes(guess - 1):
            guess -= 1
        while not passes(guess):
            guess += 1
        return guess
    low, high = m - 1, m
    while not passes(high):
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if passes(middle):
            high = middle
        else:
            low = middle
    return high


def _finite_sorted(values: Sequence[float]) -> list[float]:
    """Return the finite values sorted increasingly.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is not finite or the sample is empty.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    ordered = sorted(require_finite_real(value, name="value") for value in values)
    if not ordered:
        raise ValueError("values must contain at least one value")
    return ordered


def wilks_bound(
    values: Sequence[float],
    probability: float,
    confidence: float,
    *,
    sided: str = "upper",
) -> WilksBound:
    """Return the Wilks one-sided confidence bound of a quantile.

    The bound is ``Y_(k)`` with ``k`` from :func:`wilks_rank`: for
    independent draws of a continuous law, ``P(Y_(k) >= q_a) >= beta``
    (upper) or ``P(Y_(k) <= q_a) >= beta`` (lower). The type-7 empirical
    quantile is returned beside it; for ``n`` large it underestimates
    ``q_a`` about half of the time, which the bound is designed to avoid.

    Args:
        values: Independent finite sample.
        probability: Quantile level ``a`` strictly inside ``(0, 1)``.
        confidence: Confidence ``beta`` strictly inside ``(0, 1)``.
        sided: ``"upper"`` or ``"lower"``.

    Returns:
        The bound with its rank and exact achieved confidence.

    Raises:
        TypeError: If a value or level is not real.
        ValueError: If a value is not finite, a level is outside ``(0, 1)``,
            ``sided`` is unknown, or the sample is too small.

    Equation:
        [EQ-PROP-WILKS-001], [EQ-STAT-QUANT-001].

    References:
        [REF-WILKS-1941], [REF-HYNDMAN-FAN-1996].

    """
    ordered = _finite_sorted(values)
    result = wilks_rank(len(ordered), probability, confidence, sided=sided)
    estimate = (
        ordered[0] if len(ordered) == 1 else quantile(ordered, result.probability)
    )
    return WilksBound(
        bound=ordered[result.rank - 1],
        rank=result.rank,
        order=result.order,
        achieved_confidence=result.achieved_confidence,
        empirical_quantile=estimate,
        size=result.size,
        probability=result.probability,
        confidence=result.confidence,
        sided=result.sided,
    )


def wilks_tolerance_interval(
    values: Sequence[float], coverage: float, confidence: float
) -> WilksToleranceInterval:
    """Return the narrowest symmetric two-sided distribution-free tolerance interval.

    The content ``F(Y_(n-m+1)) - F(Y_(m))`` of ``[Y_(m), Y_(n-m+1)]`` follows
    ``Beta(n - 2m + 1, 2m)`` for any continuous law ``F``, so the interval
    contains at least ``gamma`` of the law with probability
    ``P(Binomial(n, gamma) <= n - 2m)``. The largest ``m`` that keeps this
    probability at or above ``beta`` gives the narrowest such interval. This
    statement is about the *content* of the interval, not about a single
    quantile: it is the two-sided guarantee of Wilks (1942).

    Args:
        values: Independent finite sample of at least two values.
        coverage: Required content ``gamma`` strictly inside ``(0, 1)``.
        confidence: Confidence ``beta`` strictly inside ``(0, 1)``.

    Returns:
        Tolerance interval with its ranks and exact achieved confidence.

    Raises:
        TypeError: If a value or level is not real.
        ValueError: If a value is not finite, a level is outside ``(0, 1)``,
            or the sample is too small (the message gives the minimal size).

    Equation:
        [EQ-PROP-WILKS-003].

    References:
        [REF-WILKS-1942], [REF-DAVID-NAGARAJA-2003, Section 7.2].

    """
    gamma = _level(coverage, name="coverage")
    beta = _level(confidence, name="confidence")
    ordered = _finite_sorted(values)
    n = len(ordered)

    def achieved(m: int) -> float:
        """Return the confidence of the order-``m`` symmetric interval.

        References:
            [EQ-PROP-WILKS-003]. No new equation.

        """
        return float(Binomial(n, gamma).cdf(n - 2 * m))

    if n < 2 or achieved(1) < beta:
        needed = wilks_sample_size(gamma, beta, sided="two-sided")
        raise ValueError(
            f"{n} values are too few for a {gamma:g}-content, {beta:g}-confidence "
            f"tolerance interval; at least {needed} are needed"
        )
    low, high = 1, n // 2 + 1  # achieved(low) >= beta > achieved(high)
    while high - low > 1:
        middle = (low + high) // 2
        if achieved(middle) >= beta:
            low = middle
        else:
            high = middle
    return WilksToleranceInterval(
        lower=ordered[low - 1],
        upper=ordered[n - low],
        lower_rank=low,
        upper_rank=n - low + 1,
        order=low,
        coverage=gamma,
        confidence=beta,
        achieved_confidence=achieved(low),
        size=n,
    )
