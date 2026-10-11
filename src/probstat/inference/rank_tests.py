"""Rank-based tests: Mann-Whitney U and Wilcoxon signed-rank.

Exact null distributions are computed by integer generating-function
recursions when there are no ties; otherwise a normal approximation with tie
and continuity corrections is used. A p-value is the probability, under the
null hypothesis, of a statistic at least as extreme as observed; it is not
the probability that the null hypothesis is true.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import require_finite_real, require_probability
from probstat.statistics import ranks

from ._hypothesis import Alternative, check_alternative, clamp_probability
from .sampling_distributions import standard_normal_cdf, standard_normal_sf

RankMethod = Literal["auto", "exact", "normal"]
_EXACT_LIMIT = 50


@dataclass(frozen=True, slots=True)
class RankTestResult:
    """Result of a Mann-Whitney U or Wilcoxon signed-rank test.

    Args:
        statistic: ``U`` of the first sample or the positive-rank sum ``W+``.
        p_value: Probability under the null of a statistic at least as
            extreme as observed in the direction of ``alternative``; not the
            probability that the null hypothesis is true.
        alternative: Direction of the alternative hypothesis.
        method: Test label, including ``exact`` or ``normal approximation``.
        exact: Whether the exact null distribution was used.
        z_statistic: Standardized statistic for the normal approximation, or
            ``None`` when exact.
        effect_size: Rank-biserial correlation in ``[-1, 1]``.
        sample_sizes: Sample sizes after discarding zero differences.

    References:
        [REF-MANN-WHITNEY-1947], [REF-WILCOXON-1945],
        [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapters 3-4].

    """

    statistic: float
    p_value: float
    alternative: str
    method: str
    exact: bool
    z_statistic: float | None
    effect_size: float
    sample_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the p-value.

        Raises:
            ValueError: If the p-value lies outside ``[0, 1]``.

        References:
            [REF-HOLLANDER-WOLFE-CHICKEN-2014]. No equation is evaluated.

        """
        object.__setattr__(
            self, "p_value", require_probability(self.p_value, name="p_value")
        )


def _check_method(method: object) -> RankMethod:
    """Validate a rank-test method label.

    Args:
        method: ``"auto"``, ``"exact"``, or ``"normal"``.

    Returns:
        Validated label.

    Raises:
        ValueError: If the label is unknown.

    References:
        [REF-HOLLANDER-WOLFE-CHICKEN-2014]. No equation is evaluated.

    """
    if method == "auto":
        return "auto"
    if method == "exact":
        return "exact"
    if method == "normal":
        return "normal"
    raise ValueError("method must be 'auto', 'exact', or 'normal'")


def mann_whitney_u_distribution(first_size: int, second_size: int) -> tuple[int, ...]:
    """Return the exact null counts of the Mann-Whitney ``U`` statistic.

    Without ties every assignment of ranks to the first sample is equally
    likely; the number of assignments with ``U = u`` is the coefficient of
    ``q^u`` in the Gaussian binomial ``[m+n choose m]_q``, built by the
    product ``prod_{i=1}^{m} (1 - q^{n+i}) / (1 - q^i)``.

    Args:
        first_size: Positive size ``m`` of the first sample.
        second_size: Positive size ``n`` of the second sample.

    Returns:
        Integer counts for ``u = 0, ..., m n`` summing to ``C(m+n, m)``.

    Raises:
        ValueError: If a size is not positive.

    Equation:
        [EQ-INF-MWU-001].

    References:
        [REF-MANN-WHITNEY-1947], [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 4].

    """
    m, n = first_size, second_size
    if isinstance(m, bool) or isinstance(n, bool) or m < 1 or n < 1:
        raise ValueError("sample sizes must be positive integers")
    counts = [1]
    for i in range(1, m + 1):
        shift = n + i
        product = counts + [0] * shift
        for k in range(len(product) - 1, shift - 1, -1):
            product[k] -= product[k - shift]
        quotient = product[: len(product) - i]
        for k in range(i, len(quotient)):
            quotient[k] += quotient[k - i]
        counts = quotient
    return tuple(counts)


def signed_rank_distribution(size: int) -> tuple[int, ...]:
    """Return the exact null counts of the Wilcoxon signed-rank sum ``W+``.

    Under symmetry about zero each rank carries a positive sign with
    probability one half independently, so the counts are the coefficients
    of ``prod_{k=1}^{n} (1 + q^k)``.

    Args:
        size: Positive number of non-zero differences.

    Returns:
        Integer counts for ``w = 0, ..., n(n+1)/2`` summing to ``2^n``.

    Raises:
        ValueError: If ``size`` is not positive.

    Equation:
        [EQ-INF-WSR-001].

    References:
        [REF-WILCOXON-1945], [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 3].

    """
    if isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer")
    counts = [1]
    for k in range(1, size + 1):
        extended = counts + [0] * k
        for w in range(len(extended) - 1, k - 1, -1):
            extended[w] += extended[w - k]
        counts = extended
    return tuple(counts)


def _exact_p_value(
    counts: tuple[int, ...], observed: float, alternative: Alternative
) -> float:
    """Return an exact p-value from a symmetric integer null distribution.

    Args:
        counts: Null counts indexed by statistic value.
        observed: Observed integer-valued statistic.
        alternative: Direction of the alternative.

    Returns:
        Upper, lower, or doubled smaller tail probability (capped at one).

    References:
        [EQ-INF-MWU-001], [EQ-INF-WSR-001].

    """
    total = sum(counts)
    value = round(observed)
    upper = sum(counts[value:]) / total
    lower = sum(counts[: value + 1]) / total
    if alternative == "greater":
        return upper
    if alternative == "less":
        return lower
    return min(1.0, 2.0 * min(upper, lower))


def _normal_p_value(
    statistic: float,
    mean: float,
    variance: float,
    alternative: Alternative,
    continuity: bool,
) -> tuple[float, float]:
    """Return the continuity-corrected normal p-value and z statistic.

    Args:
        statistic: Observed statistic.
        mean: Null mean.
        variance: Positive tie-corrected null variance.
        alternative: Direction of the alternative.
        continuity: Apply the half-unit continuity correction.

    Returns:
        ``(p_value, z)``.

    References:
        [EQ-INF-MWU-001], [EQ-INF-WSR-001],
        [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapters 3-4].

    """
    sd = math.sqrt(variance)
    correction = 0.5 if continuity else 0.0
    deviation = statistic - mean
    if alternative == "two-sided":
        z = max(0.0, abs(deviation) - correction) / sd
        z_signed = math.copysign(z, deviation)
        return min(1.0, 2.0 * standard_normal_sf(z)), z_signed
    if alternative == "greater":
        z = (deviation - correction) / sd
        return standard_normal_sf(z), z
    z = (deviation + correction) / sd
    return standard_normal_cdf(z), z


def _tie_sum(values: Iterable[float]) -> float:
    """Return ``sum (t^3 - t)`` over tie groups of equal values.

    Args:
        values: Values (ranks or observations) to group.

    Returns:
        Tie-correction sum, zero without ties.

    References:
        [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 4]. No new equation.

    """
    return float(sum(t**3 - t for t in Counter(values).values()))


def mann_whitney_u_test(
    first: Iterable[object],
    second: Iterable[object],
    *,
    alternative: Alternative = "two-sided",
    method: RankMethod = "auto",
    continuity_correction: bool = True,
) -> RankTestResult:
    """Run the Mann-Whitney U (Wilcoxon rank-sum) test for two samples.

    ``U = #{(i, j): x_i > y_j} + #{x_i = y_j} / 2`` for the first sample.
    Under the null that both samples come from the same continuous law,
    ``"greater"`` means the first sample tends to be larger. ``"auto"`` uses
    the exact distribution when there are no ties and both sizes are at most
    50; otherwise the normal approximation with tie-corrected variance
    ``(mn/12)(N + 1 - sum(t^3 - t) / (N (N - 1)))``.

    Args:
        first: First sample.
        second: Second sample.
        alternative: Direction of the alternative.
        method: ``"auto"``, ``"exact"`` (no ties allowed), or ``"normal"``.
        continuity_correction: Apply the half-unit correction (normal only).

    Returns:
        ``U``, p-value, and rank-biserial correlation ``2U/(mn) - 1``.

    Raises:
        TypeError: If observations are not real.
        ValueError: If a sample is empty, labels are invalid, the exact method
            is requested with ties, or all values are tied.

    Equation:
        [EQ-INF-MWU-001].

    References:
        [REF-MANN-WHITNEY-1947],
        [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 4].

    """
    a = coerce_finite_sample(first, name="first")
    b = coerce_finite_sample(second, name="second")
    side = check_alternative(alternative)
    chosen = _check_method(method)
    m, n = len(a), len(b)
    total = m + n
    combined_ranks = ranks(a + b)
    rank_sum = math.fsum(combined_ranks[:m])
    u = rank_sum - m * (m + 1) / 2.0
    ties = _tie_sum(a + b)
    effect = 2.0 * u / (m * n) - 1.0
    if chosen == "exact" and ties > 0.0:
        raise ValueError("exact Mann-Whitney distribution requires no ties")
    use_exact = chosen == "exact" or (
        chosen == "auto" and ties == 0.0 and max(m, n) <= _EXACT_LIMIT
    )
    if use_exact:
        p_value = _exact_p_value(mann_whitney_u_distribution(m, n), u, side)
        return RankTestResult(
            u,
            clamp_probability(p_value),
            side,
            "Mann-Whitney U test (exact)",
            True,
            None,
            effect,
            (m, n),
        )
    variance = m * n / 12.0 * ((total + 1) - ties / (total * (total - 1)))
    if variance <= 0.0:
        raise ValueError("all observations are tied; the test is undefined")
    p_value, z = _normal_p_value(u, m * n / 2.0, variance, side, continuity_correction)
    return RankTestResult(
        u,
        clamp_probability(p_value),
        side,
        "Mann-Whitney U test (normal approximation, tie-corrected)",
        False,
        z,
        effect,
        (m, n),
    )


def wilcoxon_signed_rank_test(
    first: Iterable[object],
    second: Iterable[object] | None = None,
    *,
    null_value: float = 0.0,
    alternative: Alternative = "two-sided",
    method: RankMethod = "auto",
    continuity_correction: bool = True,
) -> RankTestResult:
    """Run the Wilcoxon signed-rank test for a location shift.

    Differences ``d = first - second - null_value`` (or ``first -
    null_value``) equal to zero are discarded (Wilcoxon's convention). The
    absolute differences are mid-ranked and ``W+`` sums the ranks of positive
    differences. Under the null the differences are symmetric about zero.
    ``"auto"`` uses the exact distribution when no absolute differences tie
    and at most 50 remain; otherwise the normal approximation with variance
    ``n(n+1)(2n+1)/24 - sum(t^3 - t)/48``.

    Args:
        first: First measurements, or the only sample.
        second: Optional paired second measurements of equal length.
        null_value: Hypothesized location of the differences.
        alternative: ``"greater"`` means differences tend to exceed the null.
        method: ``"auto"``, ``"exact"`` (no ties allowed), or ``"normal"``.
        continuity_correction: Apply the half-unit correction (normal only).

    Returns:
        ``W+``, p-value, and matched-pairs rank-biserial correlation
        ``(W+ - W-) / (W+ + W-)``.

    Raises:
        TypeError: If observations are not real.
        ValueError: If lengths differ, all differences are zero, or the exact
            method is requested with ties.

    Equation:
        [EQ-INF-WSR-001].

    References:
        [REF-WILCOXON-1945], [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 3].

    """
    a = coerce_finite_sample(first, name="first")
    if second is None:
        raw = list(a)
    else:
        b = coerce_finite_sample(second, name="second")
        if len(a) != len(b):
            raise ValueError("paired samples must have equal lengths")
        raw = [x - y for x, y in zip(a, b, strict=True)]
    shift = require_finite_real(null_value, name="null_value")
    differences = [d - shift for d in raw if d - shift != 0.0]
    side = check_alternative(alternative)
    chosen = _check_method(method)
    n = len(differences)
    if n == 0:
        raise ValueError("all differences are zero; the test is undefined")
    absolute_ranks = ranks([abs(d) for d in differences])
    positive = math.fsum(
        r for r, d in zip(absolute_ranks, differences, strict=True) if d > 0.0
    )
    total_rank = n * (n + 1) / 2.0
    effect = (2.0 * positive - total_rank) / total_rank
    ties = _tie_sum(abs(d) for d in differences)
    if chosen == "exact" and ties > 0.0:
        raise ValueError("exact signed-rank distribution requires no ties")
    use_exact = chosen == "exact" or (
        chosen == "auto" and ties == 0.0 and n <= _EXACT_LIMIT
    )
    if use_exact:
        p_value = _exact_p_value(signed_rank_distribution(n), positive, side)
        return RankTestResult(
            positive,
            clamp_probability(p_value),
            side,
            "Wilcoxon signed-rank test (exact)",
            True,
            None,
            effect,
            (n,),
        )
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - ties / 48.0
    if variance <= 0.0:
        raise ValueError("all absolute differences are tied; the test is undefined")
    p_value, z = _normal_p_value(
        positive, total_rank / 2.0, variance, side, continuity_correction
    )
    return RankTestResult(
        positive,
        clamp_probability(p_value),
        side,
        "Wilcoxon signed-rank test (normal approximation, tie-corrected)",
        False,
        z,
        effect,
        (n,),
    )
