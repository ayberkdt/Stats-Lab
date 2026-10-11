"""Order statistics, quantiles with explicit conventions, and shape summaries."""

from __future__ import annotations

import math
from collections.abc import Iterable

from probstat.foundations import coerce_finite_sample, require_probability


def _sorted_sample(values: Iterable[float], *, minimum: int = 1) -> tuple[float, ...]:
    """Materialize, validate, and sort a finite sample.

    Args:
        values: Finite real observations.
        minimum: Required minimum sample size.

    Returns:
        Ascending tuple.

    Raises:
        ValueError: If fewer than ``minimum`` observations are supplied.

    References:
        [REF-HYNDMAN-FAN-1996]. No equation is evaluated.

    """
    sample = coerce_finite_sample(values, name="values")
    if len(sample) < minimum:
        raise ValueError(f"at least {minimum} observations are required")
    return tuple(sorted(sample))


def quantile(values: Iterable[float], probability: float, *, method: int = 7) -> float:
    """Return a sample quantile under an explicit Hyndman-Fan definition.

    Args:
        values: Non-empty finite sample.
        probability: Level in ``[0, 1]``.
        method: ``7`` (linear interpolation of order statistics, the default
            in R and NumPy) or ``1`` (inverse of the empirical CDF).

    Returns:
        Sample quantile.

    Raises:
        ValueError: If the sample is empty or ``method`` is unsupported.

    Equation:
        [EQ-STAT-QUANT-001].

    References:
        [REF-HYNDMAN-FAN-1996].

    """
    ordered = _sorted_sample(values)
    p = require_probability(probability, name="probability")
    n = len(ordered)
    if method == 1:
        if p == 0.0:
            return ordered[0]
        return ordered[max(0, math.ceil(n * p) - 1)]
    if method != 7:
        raise ValueError("method must be 1 or 7")
    position = (n - 1) * p
    lower = math.floor(position)
    upper = min(lower + 1, n - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def quantiles(
    values: Iterable[float], probabilities: Iterable[float], *, method: int = 7
) -> tuple[float, ...]:
    """Return several sample quantiles from one sort.

    Args:
        values: Non-empty finite sample.
        probabilities: Levels in ``[0, 1]``.
        method: Hyndman-Fan definition, ``7`` or ``1``.

    Returns:
        Quantiles in the order of ``probabilities``.

    Equation:
        [EQ-STAT-QUANT-001].

    References:
        [REF-HYNDMAN-FAN-1996].

    """
    ordered = _sorted_sample(values)
    return tuple(quantile(ordered, p, method=method) for p in probabilities)


def median(values: Iterable[float]) -> float:
    """Return the sample median (type-7 quantile at one half).

    Args:
        values: Non-empty finite sample.

    Returns:
        Median.

    Equation:
        [EQ-STAT-QUANT-001].

    References:
        [REF-HYNDMAN-FAN-1996].

    """
    return quantile(values, 0.5)


def interquartile_range(values: Iterable[float]) -> float:
    """Return the type-7 interquartile range ``Q(0.75) - Q(0.25)``.

    Args:
        values: Non-empty finite sample.

    Returns:
        Interquartile range.

    Equation:
        [EQ-STAT-QUANT-001].

    References:
        [REF-HYNDMAN-FAN-1996], [REF-NIST-EDA].

    """
    lower, upper = quantiles(values, (0.25, 0.75))
    return upper - lower


def median_absolute_deviation(
    values: Iterable[float], *, normal_consistent: bool = True
) -> float:
    """Return the median absolute deviation from the median.

    Args:
        values: Non-empty finite sample.
        normal_consistent: Multiply by ``1/Phi^-1(3/4) ~= 1.4826`` so the
            statistic estimates a normal standard deviation.

    Returns:
        Robust scale estimate.

    Equation:
        [EQ-STAT-MAD-001].

    References:
        [REF-HAMPEL-1974], [REF-NIST-EDA].

    """
    sample = _sorted_sample(values)
    center = median(sample)
    raw = median(abs(value - center) for value in sample)
    return raw * 1.482602218505602 if normal_consistent else raw


def skewness(values: Iterable[float]) -> float:
    """Return the sample moment skewness ``g1 = m3 / m2^(3/2)``.

    Args:
        values: Finite sample with at least two distinct values.

    Returns:
        Moment-ratio skewness (biased, population-style).

    Raises:
        ValueError: If the sample has zero variance.

    Equation:
        [EQ-STAT-SHAPE-001].

    References:
        [REF-JOANES-GILL-1998].

    """
    sample = _sorted_sample(values, minimum=2)
    n = len(sample)
    center = math.fsum(sample) / n
    m2 = math.fsum((v - center) ** 2 for v in sample) / n
    if m2 == 0.0:
        raise ValueError("skewness is undefined for zero variance")
    m3 = math.fsum((v - center) ** 3 for v in sample) / n
    return float(m3 / m2**1.5)


def excess_kurtosis(values: Iterable[float]) -> float:
    """Return the sample moment excess kurtosis ``g2 = m4 / m2^2 - 3``.

    Args:
        values: Finite sample with at least two distinct values.

    Returns:
        Excess kurtosis (zero for a normal law in expectation, asymptotically).

    Raises:
        ValueError: If the sample has zero variance.

    Equation:
        [EQ-STAT-SHAPE-001].

    References:
        [REF-JOANES-GILL-1998].

    """
    sample = _sorted_sample(values, minimum=2)
    n = len(sample)
    center = math.fsum(sample) / n
    m2 = math.fsum((v - center) ** 2 for v in sample) / n
    if m2 == 0.0:
        raise ValueError("kurtosis is undefined for zero variance")
    m4 = math.fsum((v - center) ** 4 for v in sample) / n
    return m4 / (m2 * m2) - 3.0


def ranks(values: Iterable[float]) -> tuple[float, ...]:
    """Return one-based ranks with ties assigned their average rank.

    Args:
        values: Non-empty finite sample.

    Returns:
        Ranks in the original observation order.

    Equation:
        [EQ-STAT-RANK-001].

    References:
        [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 8].

    """
    sample = coerce_finite_sample(values, name="values")
    if not sample:
        raise ValueError("at least one observation is required")
    order = sorted(range(len(sample)), key=lambda index: sample[index])
    result = [0.0] * len(sample)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and sample[order[stop + 1]] == sample[order[start]]:
            stop += 1
        average = 0.5 * (start + stop) + 1.0
        for position in range(start, stop + 1):
            result[order[position]] = average
        start = stop + 1
    return tuple(result)


def spearman_correlation(first: Iterable[float], second: Iterable[float]) -> float:
    """Return Spearman's rank correlation (Pearson correlation of ranks).

    Args:
        first: Finite sample.
        second: Paired finite sample of equal length.

    Returns:
        Rank correlation in ``[-1, 1]``.

    Raises:
        ValueError: If lengths differ, fewer than two pairs exist, or a rank
            vector is constant.

    Equation:
        [EQ-STAT-RANK-001], [EQ-STAT-007].

    References:
        [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 8].

    """
    x = ranks(first)
    y = ranks(second)
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if len(x) < 2:
        raise ValueError("at least two pairs are required")
    mx = math.fsum(x) / len(x)
    my = math.fsum(y) / len(y)
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    if sxx == 0.0 or syy == 0.0:
        raise ValueError("rank correlation is undefined for constant ranks")
    return sxy / math.sqrt(sxx * syy)
