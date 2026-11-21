"""E-values, e-BH, betting confidence sequences, and universal inference.

An e-value is the realization of a non-negative statistic ``E`` whose
expectation is at most one under every law of the null hypothesis. Markov's
inequality gives ``P(E >= 1/alpha) <= alpha``, so rejecting when
``E >= 1/alpha`` is a level-``alpha`` test, and ``min(1, 1/E)`` is a valid
(conservative) p-value. Unlike p-values, e-values may be averaged under
arbitrary dependence and multiplied when each one is an e-value given the
earlier ones, which is what makes them the natural currency of sequential
analysis.

A test (super)martingale is a non-negative process ``K_t`` with ``K_0 = 1``
whose conditional expectation given the past never increases under the
null. Ville's inequality bounds the probability that it *ever* reaches
``1/alpha`` by ``alpha``, so the data may be inspected after every
observation and sampling may stop at any data-dependent time. The betting
construction for the mean of bounded observations, its confidence sequence,
and the split likelihood-ratio test of universal inference are implemented
here.

Every guarantee in this module is non-asymptotic and holds for the stated
null class only: bounded observations with the stated conditional means for
the betting processes, and a correctly specified dominated model with an
exact null maximizer for the split likelihood ratio. The choice of bets
changes power, never validity, provided each bet is computed from earlier
observations only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

from probstat.foundations import coerce_finite_sample
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_probability,
)

from ._hypothesis import Alternative, check_alternative, check_level

BetRule = Literal["agrapa", "plug-in", "fixed"]
MergeMethod = Literal["mean", "product"]

ObservationT = TypeVar("ObservationT")
ParameterT = TypeVar("ParameterT")

_SERIES_SWITCH = 1.0e-4
_MAX_EXPONENT = 709.0


def _exp(value: float) -> float:
    """Return ``exp(value)`` with overflow mapped to infinity.

    Args:
        value: Log of a non-negative quantity; may be ``-inf``.

    Returns:
        ``exp(value)``, or ``inf`` above the double-precision range.

    References:
        Project numerical policy [REF-HIGHAM-2002]. No equation is evaluated.

    """
    if value > _MAX_EXPONENT:
        return math.inf
    return math.exp(value)


def _coerce_e_value(value: object, *, name: str) -> float:
    """Validate one e-value: a real number in ``[0, inf]``.

    Args:
        value: Candidate e-value.
        name: Parameter name used in error messages.

    Returns:
        The e-value as a float; infinity is allowed.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If ``value`` is negative or not a number.

    References:
        [EQ-INF-EVAL-001], [REF-VOVK-WANG-2021].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if math.isnan(result) or result < 0.0:
        raise ValueError(f"{name} must be a non-negative number")
    return result


def _coerce_e_values(values: Iterable[object], *, name: str) -> tuple[float, ...]:
    """Materialize a non-empty family of e-values.

    Args:
        values: Iterable of e-values in ``[0, inf]``.
        name: Parameter name used in error messages.

    Returns:
        Tuple of validated e-values in input order.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the family is empty or an entry is negative or NaN.

    References:
        [EQ-INF-EVAL-001], [REF-VOVK-WANG-2021].

    """
    result = tuple(
        _coerce_e_value(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def calibrate_p_value(p_value: float, *, kappa: float | None = None) -> float:
    """Convert a p-value into an e-value with an admissible calibrator.

    A calibrator is a decreasing function ``f`` on ``[0, 1]`` with
    ``integral_0^1 f(p) dp <= 1``; then ``f(P)`` is an e-value whenever ``P``
    is a p-value (``P(P <= u) <= u`` under the null). Two calibrators are
    provided:

    - ``kappa`` given: ``f_kappa(p) = kappa p^(kappa - 1)`` for a fixed
      ``kappa`` in ``(0, 1)`` chosen before the data are seen;
    - ``kappa=None``: the mixture
      ``F(p) = integral_0^1 kappa p^(kappa - 1) d kappa
      = (1 - p + p log p) / (p (log p)^2)``, which needs no tuning and has
      ``F(1) = 1/2`` and ``F(0) = inf``.

    Both integrate to exactly one. An e-value obtained by calibration is
    much smaller than ``1/p``: ``p = 0.01`` gives ``F = 4.45``, not 100.

    Args:
        p_value: A p-value in ``[0, 1]``.
        kappa: Exponent in ``(0, 1)`` of the power calibrator, or ``None``
            for the tuning-free mixture calibrator.

    Returns:
        The e-value ``f(p)``; infinity when ``p_value`` is zero.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``p_value`` is outside ``[0, 1]`` or ``kappa`` is
            outside ``(0, 1)``.

    Equation:
        [EQ-INF-EVAL-002].

    References:
        [REF-VOVK-WANG-2021].

    """
    p = require_probability(p_value, name="p_value")
    if kappa is not None:
        exponent = check_level(kappa, name="kappa")
        if p == 0.0:
            return math.inf
        return exponent * math.pow(p, exponent - 1.0)
    if p == 0.0:
        return math.inf
    log_p = math.log(p)
    if abs(log_p) < _SERIES_SWITCH:
        # (exp(-L) - 1 + L) / L^2 = 1/2 - L/6 + L^2/24 - ... near L = 0.
        return 0.5 - log_p / 6.0 + log_p * log_p / 24.0
    if -log_p > _MAX_EXPONENT:
        return math.inf
    return (math.expm1(-log_p) + log_p) / (log_p * log_p)


def e_to_p_value(e_value: float) -> float:
    """Convert an e-value into a p-value by Markov's inequality.

    If ``E >= 0`` has ``E[E] <= 1`` under the null, then
    ``P(1/E <= u) = P(E >= 1/u) <= u`` for ``u`` in ``(0, 1]``, so
    ``min(1, 1/E)`` is a p-value. It is the only admissible e-to-p
    calibrator, and it is conservative: the round trip p to e to p loses
    evidence.

    Args:
        e_value: A non-negative e-value; infinity is allowed.

    Returns:
        ``min(1, 1/e_value)``, with one for a zero e-value and zero for an
        infinite one.

    Raises:
        TypeError: If ``e_value`` is not real.
        ValueError: If ``e_value`` is negative or NaN.

    Equation:
        [EQ-INF-EVAL-001].

    References:
        [REF-VOVK-WANG-2021].

    """
    value = _coerce_e_value(e_value, name="e_value")
    if value <= 1.0:
        return 1.0
    if math.isinf(value):
        return 0.0
    return 1.0 / value


def merge_e_values(
    e_values: Iterable[object],
    *,
    method: MergeMethod = "mean",
    weights: Iterable[object] | None = None,
) -> float:
    """Merge several e-values for one null hypothesis into a single e-value.

    - ``"mean"``: the weighted arithmetic mean ``sum_k w_k e_k`` with fixed
      weights (equal by default). It is an e-value under *arbitrary*
      dependence because expectation is linear, and it essentially dominates
      every symmetric merging function.
    - ``"product"``: ``prod_k e_k``. It is an e-value when the e-values are
      independent or, more generally, sequential: each ``E_k`` satisfies
      ``E[E_k | E_1, ..., E_{k-1}] <= 1`` under the null. Under other
      dependence the product has no guarantee.

    Args:
        e_values: Non-empty iterable of e-values in ``[0, inf]``.
        method: ``"mean"`` or ``"product"``.
        weights: Optional non-negative weights for ``"mean"`` that sum to one
            and were fixed before the data were seen. Not allowed with
            ``"product"``.

    Returns:
        The merged e-value. A product containing a zero is zero.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the family is empty, an e-value is negative, the
            method is unknown, or the weights are invalid.

    Equation:
        [EQ-INF-EVAL-003].

    References:
        [REF-VOVK-WANG-2021].

    """
    values = _coerce_e_values(e_values, name="e_values")
    if method == "product":
        if weights is not None:
            raise ValueError("weights are not used by product merging")
        if any(value == 0.0 for value in values):
            return 0.0
        product = 1.0
        for value in values:
            product *= value
        return product
    if method != "mean":
        raise ValueError("method must be 'mean' or 'product'")
    if weights is None:
        share = 1.0 / len(values)
        checked: tuple[float, ...] = (share,) * len(values)
    else:
        checked = tuple(
            require_probability(weight, name=f"weights[{index}]")
            for index, weight in enumerate(weights)
        )
        if len(checked) != len(values):
            raise ValueError("weights must have one entry per e-value")
        if abs(math.fsum(checked) - 1.0) > 1.0e-9:
            raise ValueError("weights must sum to one")
    terms = [
        weight * value
        for weight, value in zip(checked, values, strict=True)
        if weight > 0.0
    ]
    if any(math.isinf(term) for term in terms):
        return math.inf
    return math.fsum(terms)


@dataclass(frozen=True, slots=True)
class EBenjaminiHochbergResult:
    """Rejection set of the e-BH procedure.

    Args:
        e_values: E-values in input order.
        rejected: Whether each hypothesis is rejected.
        alpha: Target false discovery rate level.
        rejection_count: Number ``k*`` of rejected hypotheses.
        threshold: Smallest e-value that is rejected, ``K / (alpha k*)``;
            infinity when nothing is rejected.

    References:
        [EQ-INF-EVAL-004], [REF-WANG-RAMDAS-2022].

    """

    e_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float
    rejection_count: int
    threshold: float


def e_benjamini_hochberg(
    e_values: Iterable[object], *, alpha: float = 0.05
) -> EBenjaminiHochbergResult:
    """Control the false discovery rate with e-values (the e-BH procedure).

    With ``K`` e-values sorted descending as ``e_(1) >= ... >= e_(K)``, let
    ``k* = max{k : k e_(k) / K >= 1/alpha}`` (zero if the set is empty) and
    reject the hypotheses with the ``k*`` largest e-values. Equivalently,
    apply Benjamini-Hochberg to the p-values ``1/e_k``.

    If every null e-value has expectation at most one, the false discovery
    rate is at most ``alpha K_0 / K`` (``K_0`` true nulls) under *arbitrary*
    dependence between the e-values; no correction factor of the
    Benjamini-Yekutieli kind is needed. The price is that honest e-values
    are smaller than reciprocal p-values.

    Args:
        e_values: Non-empty iterable of e-values in ``[0, inf]``, one per
            hypothesis.
        alpha: Target FDR level in ``(0, 1)``.

    Returns:
        The rejection set in input order with ``k*`` and the threshold.

    Raises:
        TypeError: If an e-value is not real.
        ValueError: If the family is empty, an e-value is negative, or
            ``alpha`` is outside ``(0, 1)``.

    Equation:
        [EQ-INF-EVAL-004].

    References:
        [REF-WANG-RAMDAS-2022].

    """
    values = _coerce_e_values(e_values, name="e_values")
    level = check_level(alpha, name="alpha")
    count = len(values)
    order = sorted(range(count), key=lambda index: -values[index])
    k_star = 0
    for rank, index in enumerate(order, start=1):
        if rank * values[index] >= count / level:
            k_star = rank
    rejected = [False] * count
    for index in order[:k_star]:
        rejected[index] = True
    threshold = count / (level * k_star) if k_star else math.inf
    return EBenjaminiHochbergResult(values, tuple(rejected), level, k_star, threshold)


def _unit_sample(
    observations: Iterable[object], lower: float, upper: float
) -> tuple[tuple[float, ...], float, float]:
    """Validate bounded observations and map them to the unit interval.

    Args:
        observations: Non-empty iterable of observations in
            ``[lower, upper]``.
        lower: Known lower bound of the observations.
        upper: Known upper bound, above ``lower``.

    Returns:
        ``(x, lower, width)`` with ``x_i = (X_i - lower) / width`` in
        ``[0, 1]``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If the bounds are not ordered or an observation lies
            outside them.

    References:
        [EQ-INF-CS-001], [REF-WAUDBY-SMITH-RAMDAS-2024].

    """
    low = require_finite_real(lower, name="lower")
    high = require_finite_real(upper, name="upper")
    if high <= low:
        raise ValueError("upper must be greater than lower")
    sample = coerce_finite_sample(observations, name="observations")
    width = high - low
    for index, value in enumerate(sample):
        if not low <= value <= high:
            raise ValueError(f"observations[{index}] lies outside [lower, upper]")
    return tuple((value - low) / width for value in sample), low, width


def _check_truncation(truncation: object) -> float:
    """Validate the truncation constant ``c`` of the bets.

    Args:
        truncation: Fraction of the admissible bet range that may be used.

    Returns:
        ``c`` strictly inside ``(0, 1)``, so that every factor of the
        capital is at least ``1 - c > 0``.

    Raises:
        TypeError: If ``truncation`` is not real.
        ValueError: If it is outside ``(0, 1)``.

    References:
        [EQ-INF-CS-003], [REF-WAUDBY-SMITH-RAMDAS-2024].

    """
    return check_level(truncation, name="truncation")


def _check_bet_rule(bet: object, fixed_bet: float | None) -> tuple[BetRule, float]:
    """Validate a bet rule and its optional fixed size.

    Args:
        bet: ``"agrapa"``, ``"plug-in"``, or ``"fixed"``.
        fixed_bet: Non-negative bet size on the unit scale, required for
            ``"fixed"`` and forbidden otherwise.

    Returns:
        The rule and the fixed size (zero when unused).

    Raises:
        ValueError: If the rule is unknown or ``fixed_bet`` is missing,
            negative, or supplied for another rule.

    References:
        [EQ-INF-CS-003], [REF-WAUDBY-SMITH-RAMDAS-2024].

    """
    if bet == "fixed":
        if fixed_bet is None:
            raise ValueError("bet='fixed' requires fixed_bet")
        size = require_finite_real(fixed_bet, name="fixed_bet")
        if size < 0.0:
            raise ValueError("fixed_bet must be non-negative")
        return "fixed", size
    if fixed_bet is not None:
        raise ValueError("fixed_bet is only used with bet='fixed'")
    if bet == "agrapa":
        return "agrapa", 0.0
    if bet == "plug-in":
        return "plug-in", 0.0
    raise ValueError("bet must be 'agrapa', 'plug-in', or 'fixed'")


def _predictable_moments(
    sample: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the regularized running mean and variance known before each bet.

    With ``mu_t = (1/2 + sum_{i<=t} x_i) / (t + 1)`` and
    ``s2_t = (1/4 + sum_{i<=t} (x_i - mu_i)^2) / (t + 1)``, entry ``t - 1``
    of each returned tuple holds ``mu_{t-1}`` and ``s2_{t-1}``, the values
    available when the bet on observation ``t`` is placed
    (``mu_0 = 1/2``, ``s2_0 = 1/4``).

    Args:
        sample: Observations on the unit interval.

    Returns:
        ``(means, variances)``, each of the length of ``sample``.

    References:
        [EQ-INF-CS-003], [REF-WAUDBY-SMITH-RAMDAS-2024].

    """
    means: list[float] = []
    variances: list[float] = []
    total = 0.5
    squares = 0.25
    for index, value in enumerate(sample):
        means.append(total / (index + 1))
        variances.append(squares / (index + 1))
        total += value
        squares += (value - total / (index + 2)) ** 2
    return tuple(means), tuple(variances)


def _bet_sizes(
    rule: BetRule, fixed: float, variances: Sequence[float], tail_level: float
) -> tuple[float, ...]:
    """Return the untruncated sizes of the fixed or plug-in bets.

    The plug-in size for observation ``t`` is
    ``sqrt(2 log(1/a) / (s2_{t-1} t log(t + 1)))`` with ``a`` the level
    spent on one side.

    Args:
        rule: ``"fixed"`` or ``"plug-in"``.
        fixed: Constant size used by ``"fixed"``.
        variances: Predictable variance estimates ``s2_{t-1}``.
        tail_level: Level ``a`` per side (``alpha/2`` when two-sided).

    Returns:
        One non-negative size per observation, independent of the candidate
        mean.

    References:
        [EQ-INF-CS-003], [REF-WAUDBY-SMITH-RAMDAS-2024].

    """
    if rule == "fixed":
        return (fixed,) * len(variances)
    numerator = 2.0 * math.log(1.0 / tail_level)
    return tuple(
        math.sqrt(numerator / (variance * time * math.log(time + 1.0)))
        for time, variance in enumerate(variances, start=1)
    )


def _caps(mean: float, truncation: float) -> tuple[float, float]:
    """Return the largest allowed bets on values above and below ``mean``.

    Args:
        mean: Candidate mean ``m`` in ``[0, 1]``.
        truncation: Constant ``c`` in ``(0, 1)``.

    Returns:
        ``(c / m, c / (1 - m))``, infinite where the denominator is zero
        (the corresponding factor is then at least one for every bet).

    References:
        [EQ-INF-CS-001], [REF-WAUDBY-SMITH-RAMDAS-2024].

    """
    up = truncation / mean if mean > 0.0 else math.inf
    down = truncation / (1.0 - mean) if mean < 1.0 else math.inf
    return up, down


def _agrapa_bet(
    mean: float, running_mean: float, running_variance: float, up: float, down: float
) -> float:
    """Return the signed approximate-GRAPA bet for one observation.

    Args:
        mean: Candidate mean ``m``.
        running_mean: Predictable estimate ``mu_{t-1}``.
        running_variance: Predictable estimate ``s2_{t-1} > 0``.
        up: Largest allowed positive bet ``c / m``.
        down: Largest allowed negative bet size ``c / (1 - m)``.

    Returns:
        ``(mu - m) / (s2 + (mu - m)^2)`` clipped to ``[-down, up]``.

    References:
        [EQ-INF-CS-003], [REF-WAUDBY-SMITH-RAMDAS-2024].

    """
    gap = running_mean - mean
    bet = gap / (running_variance + gap * gap)
    return max(-down, min(up, bet))


@dataclass(frozen=True, slots=True)
class BettingTestResult:
    """Capital path of a betting test for the mean of bounded observations.

    Args:
        null_mean: Boundary value ``m0`` of the null hypothesis, on the
            scale of the observations.
        alternative: ``"two-sided"`` (null: every conditional mean equals
            ``m0``), ``"greater"`` (null: conditional means at most ``m0``),
            or ``"less"``.
        bet: Bet rule that produced the path.
        alpha: Level of the sequential test.
        log_capital: ``log K_t`` after each observation.
        positive_bets: Bets ``lambda_t^+ >= 0`` on values above ``m0``, on
            the unit scale ``(X - lower) / (upper - lower)``.
        negative_bets: Bets ``lambda_t^- >= 0`` on values below ``m0``.
        e_value: Final capital ``K_n``; an e-value at the fixed sample size
            and at any stopping time.
        maximum_capital: ``max_t K_t``. It is not an e-value; its reciprocal
            is the anytime-valid p-value.
        p_value: ``min(1, 1 / max_t K_t)``, valid at every data-dependent
            stopping time by Ville's inequality.
        rejected: Whether the capital reached ``1/alpha`` at some time.
        stopping_time: First ``t`` (one-based) with ``K_t >= 1/alpha``, or
            ``None``.
        sample_size: Number of observations processed.

    References:
        [EQ-INF-CS-001], [EQ-INF-CS-004], [REF-WAUDBY-SMITH-RAMDAS-2024],
        [REF-RAMDAS-ETAL-2023].

    """

    null_mean: float
    alternative: str
    bet: str
    alpha: float
    log_capital: tuple[float, ...]
    positive_bets: tuple[float, ...]
    negative_bets: tuple[float, ...]
    e_value: float
    maximum_capital: float
    p_value: float
    rejected: bool
    stopping_time: int | None
    sample_size: int

    @property
    def capital(self) -> tuple[float, ...]:
        """Return the capital path ``K_1, ..., K_n``.

        Returns:
            ``exp(log_capital)`` entry by entry, infinite on overflow.

        References:
            [EQ-INF-CS-001], [REF-WAUDBY-SMITH-RAMDAS-2024].

        """
        return tuple(_exp(value) for value in self.log_capital)


def betting_mean_test(
    observations: Iterable[object],
    null_mean: float,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    bet: BetRule = "agrapa",
    fixed_bet: float | None = None,
    theta: float = 0.5,
    truncation: float = 0.5,
    lower: float = 0.0,
    upper: float = 1.0,
) -> BettingTestResult:
    """Test the mean of bounded observations sequentially by betting.

    After rescaling to ``x_t = (X_t - lower) / (upper - lower)`` and
    ``m = (m0 - lower) / (upper - lower)``, a gambler starts with capital
    one and multiplies it by ``1 + lambda_t (x_t - m)`` at each step, where
    the bet ``lambda_t`` in ``[-c/(1-m), c/m]`` is computed from
    ``x_1, ..., x_{t-1}`` only (``c = truncation``):

    - ``"greater"`` tests ``H0: E[X_t | past] <= m0`` for all ``t`` with
      bets ``lambda_t >= 0``: ``K_t^+ = prod (1 + lambda_i^+ (x_i - m))``;
    - ``"less"`` tests ``H0: E[X_t | past] >= m0`` with
      ``K_t^- = prod (1 - lambda_i^- (x_i - m))``;
    - ``"two-sided"`` tests ``H0: E[X_t | past] = m0``. With ``bet="agrapa"``
      the signed bet gives one martingale
      ``prod (1 + (lambda_i^+ - lambda_i^-)(x_i - m))``; with ``"plug-in"``
      or ``"fixed"`` the capital is hedged,
      ``max(theta K_t^+, (1 - theta) K_t^-)``.

    Under the null each process is a non-negative supermartingale (the
    hedged maximum is dominated by the martingale
    ``theta K^+ + (1 - theta) K^-``), so by Ville's inequality
    ``P(exists t: K_t >= 1/alpha) <= alpha``. The test may therefore be
    monitored continuously and stopped at ``stopping_time``. No
    distributional assumption beyond the bounds and the conditional means
    is used; in particular independence is not required.

    Validity holds for *every* predictable bet; the rules differ in power
    only. ``"agrapa"`` uses ``(mu - m) / (s2 + (mu - m)^2)`` with the
    regularized running mean ``mu`` and variance ``s2`` of the earlier
    observations (a second-order approximation of the growth-rate optimal
    bet); ``"plug-in"`` uses ``sqrt(2 log(1/a) / (s2 t log(t + 1)))`` with
    ``a = alpha/2`` (two-sided) or ``alpha`` (one-sided); ``"fixed"`` uses
    ``fixed_bet``. Sizes are truncated at ``c/m`` and ``c/(1 - m)``.

    Args:
        observations: Non-empty iterable of observations in
            ``[lower, upper]``, in the order in which they arrived.
        null_mean: Null value ``m0`` in ``[lower, upper]``.
        alpha: Level of the sequential test in ``(0, 1)``.
        alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
        bet: ``"agrapa"``, ``"plug-in"``, or ``"fixed"``.
        fixed_bet: Bet size on the unit scale for ``bet="fixed"``.
        theta: Weight of the upward bets in the hedged capital, in
            ``(0, 1)``; ignored unless the capital is hedged.
        truncation: Constant ``c`` in ``(0, 1)`` limiting the bets.
        lower: Known lower bound of the observations.
        upper: Known upper bound of the observations.

    Returns:
        The capital path, the bets, the final e-value, the anytime-valid
        p-value, and the first crossing time of ``1/alpha``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an observation or ``null_mean`` lies outside the
            bounds, or a level, weight, or bet specification is invalid.

    Equation:
        [EQ-INF-CS-001], [EQ-INF-CS-003], [EQ-INF-CS-004].

    References:
        [REF-WAUDBY-SMITH-RAMDAS-2024], [REF-RAMDAS-ETAL-2023],
        [REF-HOWARD-ETAL-2021].

    """
    sample, low, width = _unit_sample(observations, lower, upper)
    target = require_finite_real(null_mean, name="null_mean")
    if not low <= target <= low + width:
        raise ValueError("null_mean must lie in [lower, upper]")
    mean = (target - low) / width
    level = check_level(alpha, name="alpha")
    side = check_alternative(alternative)
    rule, fixed = _check_bet_rule(bet, fixed_bet)
    weight = check_level(theta, name="theta")
    cut = _check_truncation(truncation)
    up, down = _caps(mean, cut)
    means, variances = _predictable_moments(sample)
    positive: list[float] = []
    negative: list[float] = []
    if rule == "agrapa":
        for running_mean, running_variance in zip(means, variances, strict=True):
            signed = _agrapa_bet(mean, running_mean, running_variance, up, down)
            positive.append(max(signed, 0.0) if side != "less" else 0.0)
            negative.append(max(-signed, 0.0) if side != "greater" else 0.0)
    else:
        tail = 0.5 * level if side == "two-sided" else level
        for size in _bet_sizes(rule, fixed, variances, tail):
            positive.append(min(size, up) if side != "less" else 0.0)
            negative.append(min(size, down) if side != "greater" else 0.0)
    hedged = side == "two-sided" and rule != "agrapa"
    log_plus = 0.0
    log_minus = 0.0
    log_theta = math.log(weight)
    log_other = math.log1p(-weight)
    path: list[float] = []
    for value, bet_up, bet_down in zip(sample, positive, negative, strict=True):
        gap = value - mean
        if hedged:
            log_plus += math.log1p(bet_up * gap)
            log_minus += math.log1p(-bet_down * gap)
            path.append(max(log_theta + log_plus, log_other + log_minus))
        else:
            log_plus += math.log1p((bet_up - bet_down) * gap)
            path.append(log_plus)
    boundary = -math.log(level)
    stopping_time = next(
        (time for time, value in enumerate(path, start=1) if value >= boundary),
        None,
    )
    peak = max(path)
    return BettingTestResult(
        null_mean=target,
        alternative=side,
        bet=rule,
        alpha=level,
        log_capital=tuple(path),
        positive_bets=tuple(positive),
        negative_bets=tuple(negative),
        e_value=_exp(path[-1]),
        maximum_capital=_exp(peak),
        p_value=min(1.0, math.exp(-peak)),
        rejected=stopping_time is not None,
        stopping_time=stopping_time,
        sample_size=len(sample),
    )


def paired_betting_test(
    first: Iterable[object],
    second: Iterable[object],
    *,
    null_difference: float = 0.0,
    difference_bound: float = 1.0,
    alpha: float = 0.05,
    alternative: Alternative = "greater",
    bet: BetRule = "agrapa",
    fixed_bet: float | None = None,
    theta: float = 0.5,
    truncation: float = 0.5,
) -> BettingTestResult:
    """Compare two paired streams sequentially through their differences.

    The differences ``D_t = first_t - second_t`` lie in ``[-B, B]`` with
    ``B = difference_bound`` (``B = 1`` for two streams of losses or scores
    in ``[0, 1]``). The function applies :func:`betting_mean_test` to
    ``D_t`` with ``lower = -B`` and ``upper = B``; ``"greater"`` tests
    ``H0: E[D_t | past] <= null_difference``, so a rejection is anytime-valid
    evidence that the first stream has the larger mean.

    Args:
        first: Observations of the first stream, in order of arrival.
        second: Observations of the second stream, paired with ``first``.
        null_difference: Null value of the mean difference, in ``[-B, B]``.
        difference_bound: Known bound ``B > 0`` on ``|D_t|``.
        alpha: Level of the sequential test in ``(0, 1)``.
        alternative: ``"greater"``, ``"less"``, or ``"two-sided"``.
        bet: ``"agrapa"``, ``"plug-in"``, or ``"fixed"``.
        fixed_bet: Bet size on the unit scale ``(D + B) / (2B)`` for
            ``bet="fixed"``.
        theta: Weight of the upward bets in the hedged capital.
        truncation: Constant in ``(0, 1)`` limiting the bets.

    Returns:
        The betting test of the differences; ``null_mean`` holds
        ``null_difference``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If the streams differ in length, a difference exceeds
            the bound, or another argument is invalid.

    Equation:
        [EQ-INF-CS-004].

    References:
        [REF-WAUDBY-SMITH-RAMDAS-2024], [REF-RAMDAS-ETAL-2023].

    """
    left = coerce_finite_sample(first, name="first")
    right = coerce_finite_sample(second, name="second")
    if len(left) != len(right):
        raise ValueError("first and second must have the same length")
    bound = require_finite_real(difference_bound, name="difference_bound")
    if bound <= 0.0:
        raise ValueError("difference_bound must be positive")
    differences = tuple(a - b for a, b in zip(left, right, strict=True))
    if any(abs(value) > bound for value in differences):
        raise ValueError("a paired difference exceeds difference_bound")
    return betting_mean_test(
        differences,
        null_difference,
        alpha=alpha,
        alternative=alternative,
        bet=bet,
        fixed_bet=fixed_bet,
        theta=theta,
        truncation=truncation,
        lower=-bound,
        upper=bound,
    )


@dataclass(frozen=True, slots=True)
class ConfidenceSequence:
    """A sequence of intervals that covers the mean at all times at once.

    Args:
        lower: Lower endpoint after each observation.
        upper: Upper endpoint after each observation.
        alpha: Error level: the probability that the mean is outside the
            interval at *any* time is at most ``alpha``.
        bet: Bet rule that produced the sequence.
        running_intersection: Whether each interval was intersected with
            all earlier ones.
        grid_size: Number of grid cells on ``[lower bound, upper bound]``.
        resolution: Width of one grid cell on the scale of the
            observations; endpoints are rounded outwards by at most this
            amount.
        first_empty_time: First ``t`` (one-based) at which no candidate mean
            survives, or ``None``. From then on the endpoints at empty times
            are NaN. An empty set has probability at most ``alpha`` when the
            assumptions hold; otherwise it signals a mean that changed over
            time or data outside the stated model. With ``bet="agrapa"`` only
            an empty running intersection is reported as empty.

    References:
        [EQ-INF-CS-002], [REF-WAUDBY-SMITH-RAMDAS-2024],
        [REF-HOWARD-ETAL-2021].

    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    alpha: float
    bet: str
    running_intersection: bool
    grid_size: int
    resolution: float
    first_empty_time: int | None

    def interval(self, time: int) -> tuple[float, float]:
        """Return the interval after ``time`` observations.

        Args:
            time: One-based number of observations, from 1 to the sample
                size.

        Returns:
            ``(lower, upper)`` at that time (NaN when the set is empty).

        Raises:
            ValueError: If ``time`` is not a valid one-based index.

        References:
            [EQ-INF-CS-002], [REF-WAUDBY-SMITH-RAMDAS-2024].

        """
        if isinstance(time, bool) or not isinstance(time, int):
            raise ValueError("time must be an integer")
        if not 1 <= time <= len(self.lower):
            raise ValueError("time must lie between 1 and the sample size")
        return self.lower[time - 1], self.upper[time - 1]

    @property
    def final(self) -> tuple[float, float]:
        """Return the interval after the last observation.

        Returns:
            ``(lower, upper)`` at the final time.

        References:
            [EQ-INF-CS-002], [REF-WAUDBY-SMITH-RAMDAS-2024].

        """
        return self.lower[-1], self.upper[-1]

    @property
    def width(self) -> tuple[float, ...]:
        """Return the width of the interval after each observation.

        Returns:
            ``upper - lower`` entry by entry (NaN when the set is empty).

        References:
            [EQ-INF-CS-002], [REF-WAUDBY-SMITH-RAMDAS-2024].

        """
        return tuple(b - a for a, b in zip(self.lower, self.upper, strict=True))


def _last_true(flags: Sequence[bool]) -> int:
    """Return the last index of a true prefix by bisection.

    Args:
        flags: Booleans that are true on a (possibly empty) prefix.

    Returns:
        Index of the last true entry, or ``-1`` if the first is false.

    References:
        [EQ-INF-CS-002]. No further equation is evaluated.

    """
    low, high = -1, len(flags)
    while high - low > 1:
        middle = (low + high) // 2
        if flags[middle]:
            low = middle
        else:
            high = middle
    return low


def betting_confidence_sequence(
    observations: Iterable[object],
    *,
    alpha: float = 0.05,
    bet: BetRule = "plug-in",
    fixed_bet: float | None = None,
    theta: float = 0.5,
    truncation: float = 0.5,
    grid_size: int = 500,
    running_intersection: bool = True,
    lower: float = 0.0,
    upper: float = 1.0,
) -> ConfidenceSequence:
    """Compute a confidence sequence for a bounded mean by betting.

    For observations with ``E[X_t | past] = mu`` for all ``t`` and values in
    ``[lower, upper]``, the set ``C_t = {m : K_t(m) < 1/alpha}`` of candidate
    means whose betting capital (see :func:`betting_mean_test`) has not
    reached ``1/alpha`` satisfies
    ``P(mu in C_t for all t >= 1) >= 1 - alpha``. With
    ``running_intersection=True`` the reported set is
    ``intersection_{s <= t} C_s``, which has the same guarantee and never
    widens.

    The capital is evaluated on the grid ``m_j = j / grid_size`` of the unit
    scale:

    - ``"plug-in"`` and ``"fixed"`` use the hedged capital
      ``max(theta K_t^+(m), (1 - theta) K_t^-(m))`` with sizes that do not
      depend on ``m`` before truncation at ``c/m`` and ``c/(1 - m)``. Then
      ``K_t^+`` is non-increasing and ``K_t^-`` non-decreasing in ``m``, so
      ``C_t`` is an interval. The reported lower endpoint is the largest
      grid point rejected by ``K^+`` and the upper endpoint the smallest
      grid point rejected by ``K^-``; the reported interval therefore
      *contains* ``C_t`` and is at most one grid cell wider on each side,
      so the coverage guarantee is kept exactly.
    - ``"agrapa"`` bets depend on ``m`` and ``C_t`` need not be an interval.
      The reported endpoints are the smallest and largest surviving grid
      points, each moved outwards by one cell; when every grid point is
      rejected, the grid points of smallest capital are used in their place,
      and the running intersection is taken over these reported intervals.
      The reported interval contains ``C_t`` whenever ``m -> K_t(m)`` has a
      single valley (is quasi-convex). That property is observed, not
      proved: on simulated data it failed only through bumps of about 0.01
      in ``log K_t``. The coverage guarantee of this rule therefore rests
      on it; use ``"plug-in"`` when an unconditional guarantee is needed.

    The cost is ``O(n grid_size)`` capital updates.

    Args:
        observations: Non-empty iterable of observations in
            ``[lower, upper]``, in order of arrival.
        alpha: Error level in ``(0, 1)`` for the whole sequence.
        bet: ``"plug-in"``, ``"fixed"``, or ``"agrapa"``.
        fixed_bet: Bet size on the unit scale for ``bet="fixed"``.
        theta: Weight of the upward bets in the hedged capital, in
            ``(0, 1)``.
        truncation: Constant ``c`` in ``(0, 1)`` limiting the bets.
        grid_size: Number of grid cells, at least 2.
        running_intersection: Whether to intersect with earlier intervals.
        lower: Known lower bound of the observations.
        upper: Known upper bound of the observations.

    Returns:
        Interval endpoints after each observation, on the scale of the
        observations.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an observation lies outside the bounds or a level,
            weight, grid size, or bet specification is invalid.

    Equation:
        [EQ-INF-CS-002], [EQ-INF-CS-003].

    References:
        [REF-WAUDBY-SMITH-RAMDAS-2024], [REF-HOWARD-ETAL-2021],
        [REF-RAMDAS-ETAL-2023].

    """
    sample, low, width = _unit_sample(observations, lower, upper)
    level = check_level(alpha, name="alpha")
    rule, fixed = _check_bet_rule(bet, fixed_bet)
    weight = check_level(theta, name="theta")
    cut = _check_truncation(truncation)
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2")
    if not isinstance(running_intersection, bool):
        raise TypeError("running_intersection must be a boolean")
    grid = tuple(index / grid_size for index in range(grid_size + 1))
    caps = tuple(_caps(point, cut) for point in grid)
    means, variances = _predictable_moments(sample)
    boundary = -math.log(level)
    if rule == "agrapa":
        indices = _agrapa_sequence(
            sample, grid, caps, means, variances, boundary, running_intersection
        )
    else:
        sizes = _bet_sizes(rule, fixed, variances, 0.5 * level)
        indices = _hedged_sequence(
            sample,
            grid,
            caps,
            sizes,
            boundary - math.log(weight),
            boundary - math.log1p(-weight),
            running_intersection,
        )
    lowers: list[float] = []
    uppers: list[float] = []
    first_empty: int | None = None
    for time, (left, right) in enumerate(indices, start=1):
        if left > right:
            if first_empty is None:
                first_empty = time
            lowers.append(math.nan)
            uppers.append(math.nan)
        else:
            lowers.append(low + width * grid[left])
            uppers.append(low + width * grid[right])
    return ConfidenceSequence(
        lower=tuple(lowers),
        upper=tuple(uppers),
        alpha=level,
        bet=rule,
        running_intersection=running_intersection,
        grid_size=grid_size,
        resolution=width / grid_size,
        first_empty_time=first_empty,
    )


def _hedged_sequence(
    sample: Sequence[float],
    grid: Sequence[float],
    caps: Sequence[tuple[float, float]],
    sizes: Sequence[float],
    plus_boundary: float,
    minus_boundary: float,
    running_intersection: bool,
) -> list[tuple[int, int]]:
    """Return grid indices of the hedged-capital interval at each time.

    Args:
        sample: Observations on the unit interval.
        grid: Ascending candidate means.
        caps: Bet caps ``(c/m, c/(1-m))`` per grid point.
        sizes: Untruncated bet sizes per observation.
        plus_boundary: ``log(1/(alpha theta))``, the rejection level of
            ``log K^+``.
        minus_boundary: ``log(1/(alpha (1 - theta)))`` for ``log K^-``.
        running_intersection: Whether to intersect over time.

    Returns:
        Pairs ``(j, k)``: the largest grid index rejected by ``K^+`` (zero
        if none) and the smallest rejected by ``K^-`` (the last index if
        none); ``j > k`` marks an empty set.

    References:
        [EQ-INF-CS-002], [REF-WAUDBY-SMITH-RAMDAS-2024].

    """
    count = len(grid)
    log_plus = [0.0] * count
    log_minus = [0.0] * count
    left, right = 0, count - 1
    result: list[tuple[int, int]] = []
    for value, size in zip(sample, sizes, strict=True):
        for index in range(count):
            gap = value - grid[index]
            up, down = caps[index]
            log_plus[index] += math.log1p(min(size, up) * gap)
            log_minus[index] += math.log1p(-min(size, down) * gap)
        new_left = max(0, _last_true([entry >= plus_boundary for entry in log_plus]))
        reversed_flags = [entry >= minus_boundary for entry in reversed(log_minus)]
        new_right = min(count - 1, count - 1 - _last_true(reversed_flags))
        if running_intersection:
            left = max(left, new_left)
            right = min(right, new_right)
        else:
            left, right = new_left, new_right
        result.append((left, right))
    return result


def _agrapa_sequence(
    sample: Sequence[float],
    grid: Sequence[float],
    caps: Sequence[tuple[float, float]],
    means: Sequence[float],
    variances: Sequence[float],
    boundary: float,
    running_intersection: bool,
) -> list[tuple[int, int]]:
    """Return grid indices of the hull of the aGRAPA set at each time.

    Args:
        sample: Observations on the unit interval.
        grid: Ascending candidate means.
        caps: Bet caps ``(c/m, c/(1-m))`` per grid point.
        means: Predictable running means.
        variances: Predictable running variances.
        boundary: ``log(1/alpha)``.
        running_intersection: Whether to intersect the reported intervals
            over time.

    Returns:
        Pairs ``(j, k)`` of the smallest and largest surviving grid index,
        each moved outwards by one cell. When no grid point survives, the
        grid points of smallest capital take the place of the survivors: if
        ``m -> K_t(m)`` has a single valley, a surviving set that fits
        between two grid points lies in a cell next to them. ``j > k`` marks
        an empty running intersection.

    References:
        [EQ-INF-CS-002], [EQ-INF-CS-003], [REF-WAUDBY-SMITH-RAMDAS-2024].

    """
    count = len(grid)
    log_capital = [0.0] * count
    left, right = 0, count - 1
    result: list[tuple[int, int]] = []
    for value, running_mean, running_variance in zip(
        sample, means, variances, strict=True
    ):
        for index in range(count):
            point = grid[index]
            up, down = caps[index]
            bet = _agrapa_bet(point, running_mean, running_variance, up, down)
            log_capital[index] += math.log1p(bet * (value - point))
        survivors = [index for index in range(count) if log_capital[index] < boundary]
        if not survivors:
            # Every grid point is rejected. The surviving set is then empty or
            # lies strictly inside one cell; the grid cannot tell which, so
            # the cells next to the smallest grid capital are kept.
            smallest = min(log_capital)
            survivors = [
                index for index in range(count) if log_capital[index] == smallest
            ]
        new_left = max(0, survivors[0] - 1)
        new_right = min(count - 1, survivors[-1] + 1)
        if running_intersection:
            left = max(left, new_left)
            right = min(right, new_right)
        else:
            left, right = new_left, new_right
        result.append((left, right))
    return result


@dataclass(frozen=True, slots=True)
class SplitLikelihoodRatioResult:
    """Outcome of the split (or cross-fit) likelihood-ratio test.

    Args:
        e_value: The split statistic ``T``, or the average of the two
            cross-fit statistics; an e-value under the null.
        log_e_value: Natural logarithm of ``e_value``.
        p_value: ``min(1, 1 / e_value)``.
        rejected: Whether ``e_value >= 1/alpha``.
        alpha: Level of the test.
        log_statistics: ``log T`` of each split used (one entry, or two for
            the cross-fit variant).
        estimation_size: Number of observations in ``D1`` (used to choose
            the alternative parameter).
        inference_size: Number of observations in ``D0`` (on which the
            likelihood ratio is evaluated).
        cross_fit: Whether the roles of the halves were swapped and the two
            statistics averaged.
        likelihood_evaluations: Number of calls of ``log_likelihood``.
        fits: Number of calls of ``fit_alternative`` and ``fit_null``
            together.

    References:
        [EQ-INF-EVAL-005], [REF-WASSERMAN-RAMDAS-BALAKRISHNAN-2020].

    """

    e_value: float
    log_e_value: float
    p_value: float
    rejected: bool
    alpha: float
    log_statistics: tuple[float, ...]
    estimation_size: int
    inference_size: int
    cross_fit: bool
    likelihood_evaluations: int
    fits: int


def _log_split_ratio(
    estimation: Sequence[ObservationT],
    inference: Sequence[ObservationT],
    log_likelihood: Callable[[ParameterT, Sequence[ObservationT]], float],
    fit_alternative: Callable[[Sequence[ObservationT]], ParameterT],
    fit_null: Callable[[Sequence[ObservationT]], ParameterT],
) -> float:
    """Return ``log T`` for one assignment of the two parts of the data.

    Args:
        estimation: Part ``D1`` passed to ``fit_alternative``.
        inference: Part ``D0`` on which both likelihoods are evaluated.
        log_likelihood: Joint log-likelihood of a data part at a parameter.
        fit_alternative: Any estimator computed from ``D1``.
        fit_null: Maximizer of the likelihood of ``D0`` over the null.

    Returns:
        ``l_D0(theta_1) - l_D0(theta_0)``; ``-inf`` when the alternative
        parameter has zero likelihood on ``D0``.

    Raises:
        ValueError: If a log-likelihood is NaN or ``+inf``, or the null
            maximizer has zero likelihood.

    References:
        [EQ-INF-EVAL-005], [REF-WASSERMAN-RAMDAS-BALAKRISHNAN-2020].

    """
    alternative_value = float(log_likelihood(fit_alternative(estimation), inference))
    null_value = float(log_likelihood(fit_null(inference), inference))
    if math.isnan(alternative_value) or alternative_value == math.inf:
        raise ValueError("log_likelihood at the alternative fit must be finite or -inf")
    if not math.isfinite(null_value):
        raise ValueError("log_likelihood at the null fit must be finite")
    return alternative_value - null_value


def split_likelihood_ratio_test(
    data: Sequence[ObservationT],
    log_likelihood: Callable[[ParameterT, Sequence[ObservationT]], float],
    fit_alternative: Callable[[Sequence[ObservationT]], ParameterT],
    fit_null: Callable[[Sequence[ObservationT]], ParameterT],
    *,
    alpha: float = 0.05,
    estimation_fraction: float = 0.5,
    cross_fit: bool = False,
    shuffle: bool = True,
    seed: int | RandomStream | None = None,
) -> SplitLikelihoodRatioResult:
    """Test a composite null by the split likelihood ratio (universal inference).

    The data are divided into ``D1`` (estimation) and ``D0`` (inference).
    With ``theta_1 = fit_alternative(D1)`` any estimator and
    ``theta_0 = fit_null(D0)`` the maximum-likelihood estimator over the
    null computed on ``D0``,

    ``T = L_D0(theta_1) / L_D0(theta_0)``,

    where ``L_D0`` is the likelihood of ``D0``. For independent observations
    from a density ``p_theta*`` with ``theta*`` in the null,
    ``E[L_D0(theta_1) / L_D0(theta*) | D1] = integral p_theta_1 <= 1`` and
    ``L_D0(theta_0) >= L_D0(theta*)``, so ``E[T] <= 1``: ``T`` is an e-value
    in finite samples with *no* regularity conditions (no Wilks theorem is
    needed), and the test rejects when ``T >= 1/alpha``. With
    ``cross_fit=True`` the roles of ``D0`` and ``D1`` are swapped and the
    two statistics averaged, which is again an e-value and removes part of
    the randomness of the split.

    The guarantee requires that (i) the model is correctly specified and
    ``log_likelihood`` is the complete joint log-density of its data
    argument (the same dominating measure for every parameter, no dropped
    parameter-dependent constants); (ii) ``D0`` and ``D1`` are independent,
    which random splitting of independent observations ensures; (iii)
    ``fit_null`` returns a global maximizer over the null, or at least a
    parameter whose likelihood on ``D0`` is not below that of the truth. A
    numerical optimizer that stops early makes ``T`` too large and voids
    the guarantee. The test is conservative: its power is lower than that
    of the classical likelihood-ratio test where the latter is valid.

    Args:
        data: Sequence of at least two independent observations.
        log_likelihood: ``log_likelihood(theta, part)`` returns the joint
            log-likelihood of the observations in ``part``.
        fit_alternative: Maps a data part to a parameter of the full model.
        fit_null: Maps a data part to the null maximum-likelihood
            estimate (a constant function for a simple null).
        alpha: Level in ``(0, 1)``.
        estimation_fraction: Share of the data assigned to ``D1``, strictly
            between zero and one; both parts keep at least one observation.
        cross_fit: Whether to swap the parts and average the statistics.
        shuffle: Whether to permute the data before splitting. With
            ``False`` the first observations form ``D1``, which is valid
            when the order carries no information.
        seed: Integer seed or stream for the permutation; ``None`` draws
            fresh entropy and is not reproducible.

    Returns:
        The e-value, its logarithm, the p-value ``min(1, 1/T)``, the
        decision, and the cost.

    Raises:
        TypeError: If ``seed`` or a flag has an invalid type.
        ValueError: If fewer than two observations are given, a level or
            fraction is invalid, or a log-likelihood value is not usable.

    Equation:
        [EQ-INF-EVAL-005].

    References:
        [REF-WASSERMAN-RAMDAS-BALAKRISHNAN-2020].

    """
    items = list(data)
    if len(items) < 2:
        raise ValueError("data must contain at least two observations")
    level = check_level(alpha, name="alpha")
    fraction = check_level(estimation_fraction, name="estimation_fraction")
    if not isinstance(cross_fit, bool) or not isinstance(shuffle, bool):
        raise TypeError("cross_fit and shuffle must be booleans")
    if shuffle:
        resolve_stream(seed).child("split").generator().shuffle(items)
    cut = min(len(items) - 1, max(1, round(fraction * len(items))))
    estimation = tuple(items[:cut])
    inference = tuple(items[cut:])
    logs = [
        _log_split_ratio(
            estimation, inference, log_likelihood, fit_alternative, fit_null
        )
    ]
    if cross_fit:
        logs.append(
            _log_split_ratio(
                inference, estimation, log_likelihood, fit_alternative, fit_null
            )
        )
    peak = max(logs)
    if peak == -math.inf:
        log_e = -math.inf
    else:
        log_e = peak + math.log(
            math.fsum(math.exp(value - peak) for value in logs) / len(logs)
        )
    e_value = _exp(log_e)
    return SplitLikelihoodRatioResult(
        e_value=e_value,
        log_e_value=log_e,
        p_value=math.exp(-log_e) if log_e > 0.0 else 1.0,
        rejected=log_e >= -math.log(level),
        alpha=level,
        log_statistics=tuple(logs),
        estimation_size=len(estimation),
        inference_size=len(inference),
        cross_fit=cross_fit,
        likelihood_evaluations=2 * len(logs),
        fits=2 * len(logs),
    )
