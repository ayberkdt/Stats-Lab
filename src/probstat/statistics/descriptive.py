"""Auditable descriptive statistics implemented with the standard library."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real

from probstat.foundations.validation import coerce_finite_sample


def mean(values: Iterable[Real]) -> float:
    """Return the arithmetic mean of a non-empty finite sample.

    Args:
        values: Finite real observations.

    Returns:
        The arithmetic mean. ``math.fsum`` reduces avoidable summation error.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If the sample is empty or contains a non-finite value.

    Equation:
        [EQ-STAT-001].

    References:
        [REF-NIST-EDA, Section 1.3.5.1].

    """
    sample = coerce_finite_sample(values)
    return math.fsum(sample) / len(sample)


def variance(values: Iterable[Real], *, ddof: int = 0) -> float:
    """Return finite-sample variance with an explicit degrees-of-freedom offset.

    The two-pass centered computation is more stable than subtracting the
    squared mean from the mean square. Use ``ddof=0`` for population variance
    and ``ddof=1`` for the conventional unbiased sample-variance estimator.

    Args:
        values: Finite real observations.
        ddof: Delta degrees of freedom in the denominator ``n - ddof``.

    Returns:
        The non-negative variance.

    Raises:
        TypeError: If observations or ``ddof`` have invalid types.
        ValueError: If data are invalid or ``ddof`` is outside ``[0, n)``.

    Equation:
        [EQ-STAT-002].

    References:
        [REF-NIST-EDA, Sections 1.3.5.1 and 1.3.5.2].

    """
    sample = coerce_finite_sample(values)
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError("ddof must be an integer")
    if not 0 <= ddof < len(sample):
        raise ValueError("ddof must satisfy 0 <= ddof < number of observations")
    center = math.fsum(sample) / len(sample)
    squared_deviations = ((value - center) ** 2 for value in sample)
    return math.fsum(squared_deviations) / (len(sample) - ddof)


def standard_deviation(values: Iterable[Real], *, ddof: int = 0) -> float:
    """Return the non-negative square root of finite-sample variance.

    Args:
        values: Finite real observations.
        ddof: Delta degrees of freedom passed to :func:`variance`.

    Returns:
        Standard deviation in the observations' original units.

    Raises:
        TypeError: If observations or ``ddof`` have invalid types.
        ValueError: If data are invalid or ``ddof`` is outside ``[0, n)``.

    Equation:
        [EQ-STAT-003].

    References:
        [REF-NIST-EDA, Section 1.3.5.2].

    """
    return math.sqrt(variance(values, ddof=ddof))


def z_scores(values: Iterable[Real], *, ddof: int = 0) -> tuple[float, ...]:
    """Standardize observations relative to their sample center and spread.

    Args:
        values: Finite real observations.
        ddof: Delta degrees of freedom used for the standard deviation.

    Returns:
        One z-score per observation, in original order.

    Raises:
        TypeError: If observations or ``ddof`` have invalid types.
        ValueError: If data are invalid, ``ddof`` is invalid, or the sample has
            zero standard deviation.

    Equation:
        [EQ-STAT-004].

    References:
        [REF-NIST-EDA, Section 1.3.5.17].

    """
    sample = coerce_finite_sample(values)
    center = mean(sample)
    spread = standard_deviation(sample, ddof=ddof)
    if spread == 0.0:
        raise ValueError("z-scores are undefined for zero-variance data")
    return tuple((value - center) / spread for value in sample)


def empirical_cdf(values: Iterable[Real], x: Real) -> float:
    """Evaluate the empirical cumulative distribution function at ``x``.

    Args:
        values: Finite real observations.
        x: Finite evaluation point.

    Returns:
        The fraction of observations less than or equal to ``x``.

    Raises:
        TypeError: If an observation or ``x`` is not real.
        ValueError: If the sample is empty or any input is non-finite.

    Equation:
        [EQ-STAT-005].

    References:
        [REF-WASSERMAN-2004, Section 7.3].

    """
    sample = coerce_finite_sample(values)
    point = coerce_finite_sample((x,), name="x")[0]
    return math.fsum(value <= point for value in sample) / len(sample)


def _paired_samples(
    x_values: Iterable[Real], y_values: Iterable[Real]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate two finite samples and require equal lengths.

    Args:
        x_values: First sequence of paired observations.
        y_values: Second sequence of paired observations.

    Returns:
        Two validated tuples with equal, non-zero lengths.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If either sample is invalid or their lengths differ.

    References:
        [REF-NIST-EDA]. This helper validates domains and evaluates no equation.

    """
    x_sample = coerce_finite_sample(x_values, name="x_values")
    y_sample = coerce_finite_sample(y_values, name="y_values")
    if len(x_sample) != len(y_sample):
        raise ValueError("paired samples must have equal lengths")
    return x_sample, y_sample


def covariance(
    x_values: Iterable[Real], y_values: Iterable[Real], *, ddof: int = 0
) -> float:
    """Return covariance for two equally sized finite samples.

    Args:
        x_values: First sequence of paired observations.
        y_values: Second sequence of paired observations.
        ddof: Delta degrees of freedom in the denominator ``n - ddof``.

    Returns:
        Covariance in the product of the input units.

    Raises:
        TypeError: If observations or ``ddof`` have invalid types.
        ValueError: If data are invalid or ``ddof`` is outside ``[0, n)``.

    Equation:
        [EQ-STAT-006].

    References:
        [REF-NIST-EDA, Section 1.3.5.12].

    """
    x_sample, y_sample = _paired_samples(x_values, y_values)
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError("ddof must be an integer")
    if not 0 <= ddof < len(x_sample):
        raise ValueError("ddof must satisfy 0 <= ddof < number of pairs")
    x_mean = mean(x_sample)
    y_mean = mean(y_sample)
    cross_products = (
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_sample, y_sample, strict=True)
    )
    return math.fsum(cross_products) / (len(x_sample) - ddof)


def pearson_correlation(
    x_values: Iterable[Real], y_values: Iterable[Real]
) -> float:
    """Return the Pearson product-moment correlation coefficient.

    Args:
        x_values: First sequence of paired observations.
        y_values: Second sequence of paired observations.

    Returns:
        Dimensionless linear association in ``[-1, 1]``, up to floating-point
        roundoff.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If samples are invalid, lengths differ, or either sample
            has zero variance.

    Equation:
        [EQ-STAT-007].

    References:
        [REF-NIST-EDA, Section 1.3.5.12].

    """
    x_sample, y_sample = _paired_samples(x_values, y_values)
    x_spread = standard_deviation(x_sample)
    y_spread = standard_deviation(y_sample)
    if x_spread == 0.0 or y_spread == 0.0:
        raise ValueError("correlation is undefined for zero-variance data")
    result = covariance(x_sample, y_sample) / (x_spread * y_spread)
    return max(-1.0, min(1.0, result))

