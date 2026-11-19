"""Order statistics, quantiles with explicit conventions, and shape summaries."""

from __future__ import annotations

import math
from collections.abc import Iterable

from probstat.foundations import coerce_finite_sample, require_probability

from ._hyndman_fan import hyndman_fan_quantile


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


def _smallest_index_reaching(size: int, probability: float) -> int:
    """Return the smallest ``j`` in ``1..n`` with ``j / n >= p``.

    ``ceil(n p)`` in floating point can overshoot by one when ``n p`` is an
    integer that the product rounds above (``100 * 0.07 = 7.000000000000001``).
    The result is corrected against the comparison ``j / n >= p`` evaluated
    on the same floats as the empirical CDF, so ``F_n(x_(j)) = j / n``
    reaching ``p`` selects ``x_(j)``.

    Args:
        size: Sample size ``n >= 1``.
        probability: Level ``p`` in ``(0, 1]``.

    Returns:
        Order-statistic index ``j``.

    References:
        [EQ-STAT-QUANT-001], [REF-HYNDMAN-FAN-1996].

    """
    index = min(max(math.ceil(size * probability), 1), size)
    while index > 1 and (index - 1) / size >= probability:
        index -= 1
    while index < size and index / size < probability:
        index += 1
    return index


def _largest_count_within(size: int, proportion: float) -> int:
    """Return the largest ``g`` in ``0..n`` with ``g / n <= proportion``.

    This is ``floor(proportion n)`` without the floating-point undershoot of
    the product (``0.29 * 100 = 28.999999999999996``).

    Args:
        size: Sample size ``n >= 1``.
        proportion: Non-negative fraction.

    Returns:
        Count ``g``.

    References:
        [EQ-STAT-TRIM-001], [REF-STIGLER-1977].

    """
    count = min(max(math.floor(proportion * size), 0), size)
    while count < size and (count + 1) / size <= proportion:
        count += 1
    while count > 0 and count / size > proportion:
        count -= 1
    return count


def quantile(values: Iterable[float], probability: float, *, method: int = 7) -> float:
    """Return a sample quantile under an explicit Hyndman-Fan definition.

    Every type is ``(1 - gamma) x_(j) + gamma x_(j+1)`` for adjacent order
    statistics; the types differ in the position and the weight ``gamma``.
    Types 1-3 are step functions of ``p``, types 4-9 interpolate linearly
    between the plotting positions ``p_k = (k - alpha)/(n + 1 - alpha -
    beta)``. When ``p`` is the floating-point value of a level at which the
    position is an integer, the position is that integer (``n = 100`` and
    ``p = 0.07`` select ``x_(7)`` exactly although ``100 * 0.07`` rounds to
    ``7.000000000000001``).

    Args:
        values: Non-empty finite sample.
        probability: Level in ``[0, 1]``.
        method: Hyndman-Fan type, numbered as R's ``quantile(type = )``:

            - ``1``: inverse of the empirical CDF, ``x_(j)`` with the
              smallest ``j`` such that ``j / n >= p``.
            - ``2``: as type 1, but the average of ``x_(j)`` and
              ``x_(j+1)`` where ``n p = j`` is an integer (the SAS
              default).
            - ``3``: the nearest even order statistic to ``n p``, a former
              SAS default.
            - ``4``: linear interpolation of the empirical CDF,
              ``p_k = k / n``.
            - ``5``: ``p_k = (k - 1/2) / n``, the midpoints of the CDF steps.
            - ``6``: ``p_k = k / (n + 1)``, so ``p_k = E[F(X_(k))]``
              (Minitab, SPSS).
            - ``7`` (default, as in R and NumPy): ``p_k = (k - 1)/(n - 1)``.
            - ``8``: ``p_k = (k - 1/3)/(n + 1/3)``, approximately
              median-unbiased for any continuous law (Hyndman and Fan's
              recommendation).
            - ``9``: ``p_k = (k - 3/8)/(n + 1/4)``, approximately unbiased
              for the expected order statistics of a normal law.

    Returns:
        Sample quantile.

    Raises:
        ValueError: If the sample is empty or ``method`` is not an integer
            from 1 to 9.

    Equation:
        [EQ-STAT-QUANT-001], [EQ-STAT-QUANT-002].

    References:
        [REF-HYNDMAN-FAN-1996].

    """
    ordered = _sorted_sample(values)
    p = require_probability(probability, name="probability")
    n = len(ordered)
    if isinstance(method, bool) or method not in range(1, 10):
        raise ValueError("method must be an integer from 1 to 9")
    if method == 1:
        if p == 0.0:
            return ordered[0]
        return ordered[_smallest_index_reaching(n, p) - 1]
    if method != 7:
        return hyndman_fan_quantile(ordered, p, int(method))
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
        method: Hyndman-Fan type from 1 to 9 (see ``quantile``); default 7.

    Returns:
        Quantiles in the order of ``probabilities``.

    Raises:
        ValueError: If the sample is empty or ``method`` is not an integer
            from 1 to 9.

    Equation:
        [EQ-STAT-QUANT-001], [EQ-STAT-QUANT-002].

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


def trimmed_mean(values: Iterable[float], proportion: float = 0.1) -> float:
    """Return the symmetric trimmed mean.

    With ``g = floor(proportion * n)``, the ``g`` smallest and ``g`` largest
    observations are removed and the remaining ``n - 2g`` are averaged.
    ``proportion = 0`` gives the arithmetic mean. The floor is taken exactly:
    ``proportion = 0.29`` with ``n = 100`` trims 29 values per end, although
    the floating-point product ``0.29 * 100`` is ``28.999999999999996``.

    Args:
        values: Non-empty finite sample.
        proportion: Fraction trimmed from each end, in ``[0, 0.5)``.

    Returns:
        Trimmed mean.

    Raises:
        TypeError: If ``proportion`` is not real.
        ValueError: If the sample is empty or ``proportion`` is outside
            ``[0, 0.5)``.

    Equation:
        [EQ-STAT-TRIM-001].

    References:
        [REF-STIGLER-1977].

    """
    ordered = _sorted_sample(values)
    fraction = require_probability(proportion, name="proportion")
    if fraction >= 0.5:
        raise ValueError("proportion must be below one half")
    trimmed = _largest_count_within(len(ordered), fraction)
    kept = ordered[trimmed : len(ordered) - trimmed]
    return math.fsum(kept) / len(kept)


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


def _shape_type(kind: int) -> int:
    """Validate a Joanes-Gill shape-statistic convention.

    Args:
        kind: ``1`` (``g``), ``2`` (``G``), or ``3`` (``b``).

    Returns:
        The convention as an integer.

    Raises:
        ValueError: If ``kind`` is not 1, 2, or 3.

    References:
        [EQ-STAT-SHAPE-002], [REF-JOANES-GILL-1998].

    """
    if isinstance(kind, bool) or kind not in (1, 2, 3):
        raise ValueError("type must be 1, 2, or 3")
    return int(kind)


def skewness(values: Iterable[float], *, type: int = 1) -> float:
    """Return the sample skewness under one of three Joanes-Gill conventions.

    With central moments ``m_k = n^-1 sum (x_i - xbar)^k`` and ``g1 = m3 /
    m2^(3/2)``:

    - ``type=1`` (default): ``g1``, the moment ratio (e1071 type 1, older
      textbooks).
    - ``type=2``: ``G1 = g1 sqrt(n (n - 1)) / (n - 2)``, the ratio of the
      unbiased cumulant estimators ``k3 / k2^(3/2)`` (SAS and SPSS; e1071
      type 2).
    - ``type=3``: ``b1 = m3 / s^3 = g1 ((n - 1) / n)^(3/2)`` with ``s`` the
      standard deviation with divisor ``n - 1`` (MINITAB, BMDP; e1071's
      default type 3).

    Args:
        values: Finite sample with at least two distinct values (three for
            ``type=2``).
        type: Convention ``1``, ``2``, or ``3``, numbered as in e1071 and
            Joanes and Gill.

    Returns:
        Sample skewness.

    Raises:
        ValueError: If the sample has zero variance, ``type`` is not 1, 2,
            or 3, or ``type=2`` has fewer than three observations.

    Equation:
        [EQ-STAT-SHAPE-001], [EQ-STAT-SHAPE-002].

    References:
        [REF-JOANES-GILL-1998].

    """
    kind = _shape_type(type)
    sample = _sorted_sample(values, minimum=2)
    n = len(sample)
    if kind == 2 and n < 3:
        raise ValueError("type 2 skewness needs at least three observations")
    center = math.fsum(sample) / n
    m2 = math.fsum((v - center) ** 2 for v in sample) / n
    if m2 == 0.0:
        raise ValueError("skewness is undefined for zero variance")
    m3 = math.fsum((v - center) ** 3 for v in sample) / n
    g1 = float(m3 / m2**1.5)
    if kind == 2:
        return g1 * math.sqrt(n * (n - 1)) / (n - 2)
    if kind == 3:
        return g1 * float(((n - 1) / n) ** 1.5)
    return g1


def excess_kurtosis(values: Iterable[float], *, type: int = 1) -> float:
    """Return the sample excess kurtosis under one of three Joanes-Gill conventions.

    With central moments ``m_k = n^-1 sum (x_i - xbar)^k`` and ``g2 = m4 /
    m2^2 - 3``:

    - ``type=1`` (default): ``g2``, the moment ratio (e1071 type 1).
    - ``type=2``: ``G2 = ((n + 1) g2 + 6)(n - 1) / ((n - 2)(n - 3))``, the
      ratio of the unbiased cumulant estimators ``k4 / k2^2`` (SAS and
      SPSS; e1071 type 2), the only one of the three that is unbiased for
      normal data.
    - ``type=3``: ``b2 = m4 / s^4 - 3 = (g2 + 3)(1 - 1/n)^2 - 3`` with ``s``
      the standard deviation with divisor ``n - 1`` (MINITAB, BMDP; e1071's
      default type 3).

    Args:
        values: Finite sample with at least two distinct values (four for
            ``type=2``).
        type: Convention ``1``, ``2``, or ``3``, numbered as in e1071 and
            Joanes and Gill.

    Returns:
        Excess kurtosis (zero for a normal law in expectation, asymptotically).

    Raises:
        ValueError: If the sample has zero variance, ``type`` is not 1, 2,
            or 3, or ``type=2`` has fewer than four observations.

    Equation:
        [EQ-STAT-SHAPE-001], [EQ-STAT-SHAPE-002].

    References:
        [REF-JOANES-GILL-1998].

    """
    kind = _shape_type(type)
    sample = _sorted_sample(values, minimum=2)
    n = len(sample)
    if kind == 2 and n < 4:
        raise ValueError("type 2 kurtosis needs at least four observations")
    center = math.fsum(sample) / n
    m2 = math.fsum((v - center) ** 2 for v in sample) / n
    if m2 == 0.0:
        raise ValueError("kurtosis is undefined for zero variance")
    m4 = math.fsum((v - center) ** 4 for v in sample) / n
    g2 = m4 / (m2 * m2) - 3.0
    if kind == 2:
        return ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3))
    if kind == 3:
        return (g2 + 3.0) * (1.0 - 1.0 / n) ** 2 - 3.0
    return g2


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
