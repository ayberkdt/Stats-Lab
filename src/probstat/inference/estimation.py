"""Maximum-likelihood estimators for foundational continuous families."""

from __future__ import annotations

import math
from collections.abc import Iterable

from probstat.distributions import Gamma, LogNormal, Normal, Weibull
from probstat.foundations import coerce_finite_sample
from probstat.numerics import log_minus_digamma, trigamma

_EPSILON = math.ulp(1.0)
_MAX_NEWTON_ITERATIONS = 200


def fit_normal_mle(values: Iterable[object]) -> Normal:
    """Fit a non-degenerate normal law by maximum likelihood.

    Args:
        values: Non-empty iterable of finite real observations.

    Returns:
        Normal law with the MLE mean and population-divisor scale.

    Raises:
        TypeError: If observations are not an iterable of real values.
        ValueError: If input is empty, non-finite, or has zero MLE variance.

    Equation:
        [EQ-INF-MLE-NORM-001].

    References:
        [REF-CASELLA-BERGER-2002, Sections 7.2-7.3].

    """
    sample = coerce_finite_sample(values)
    count = len(sample)
    location = math.fsum(sample) / count
    squared_deviations = math.fsum((value - location) ** 2 for value in sample)
    scale = math.sqrt(squared_deviations / count)
    if scale == 0.0:
        raise ValueError("normal MLE requires at least two distinct values")
    return Normal(location, scale)


def fit_lognormal_mle(values: Iterable[object]) -> LogNormal:
    """Fit a two-parameter lognormal law by transformed normal MLE.

    Args:
        values: Non-empty iterable of finite strictly positive observations.

    Returns:
        Lognormal law whose log-space parameters maximize the likelihood.

    Raises:
        TypeError: If observations are not an iterable of real values.
        ValueError: If input is empty, non-finite, non-positive, or degenerate
            after logarithmic transformation.

    Equation:
        [EQ-INF-MLE-LN-001].

    References:
        [REF-NIST-DISTRIBUTIONS, Lognormal Distribution],
        [REF-CASELLA-BERGER-2002, Section 7.2].

    """
    sample = coerce_finite_sample(values)
    if any(value <= 0.0 for value in sample):
        raise ValueError("lognormal MLE requires strictly positive observations")
    fitted_logarithms = fit_normal_mle(tuple(math.log(value) for value in sample))
    return LogNormal(fitted_logarithms.mu, fitted_logarithms.sigma)


def _safeguarded_step(
    shape: float, candidate: float, lower: float, upper: float
) -> float:
    """Keep a Newton candidate inside the current root bracket.

    Args:
        shape: Current positive iterate.
        candidate: Proposed Newton iterate (may be NaN).
        lower: Largest point known to lie below the root (``0`` if none).
        upper: Smallest point known to lie above the root (``inf`` if none).

    Returns:
        ``candidate`` when it lies strictly inside the bracket; otherwise a
        doubling, halving, or geometric-bisection step.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 11]. No new equation is evaluated.

    """
    if lower < candidate < upper:
        return candidate
    if upper == math.inf:
        return 2.0 * shape
    if lower == 0.0:
        return 0.5 * shape
    return math.sqrt(lower * upper)


def fit_gamma_mle(values: Iterable[object]) -> Gamma:
    """Fit a shape-rate gamma law by maximum likelihood.

    The rate is profiled out as ``lambda = k / mean(x)``, which leaves the
    shape equation ``log k - psi(k) = s`` with
    ``s = log(mean(x)) - mean(log x) > 0``. The equation is solved by Newton's
    method on its strictly decreasing left side, safeguarded by a bracket that
    falls back to bisection. The start ``k0 = (3 + sqrt(9 + 12 s)) / (12 s)``
    solves the two-term asymptotic approximation ``1/(2k) + 1/(12 k^2) = s``.

    Args:
        values: Iterable of finite strictly positive observations containing
            at least two distinct values.

    Returns:
        Gamma law with the maximum-likelihood shape and rate.

    Raises:
        TypeError: If observations are not an iterable of real values.
        ValueError: If input is empty, non-finite, non-positive, or constant.
        ArithmeticError: If the Newton iteration does not converge.

    Equation:
        [EQ-INF-MLE-GAM-001], [EQ-NUM-DIGAMMA-001].

    References:
        [REF-JKB-1994, Chapter 17], [REF-CASELLA-BERGER-2002, Section 7.2].

    """
    sample = coerce_finite_sample(values)
    if any(value <= 0.0 for value in sample):
        raise ValueError("gamma MLE requires strictly positive observations")
    count = len(sample)
    arithmetic_mean = math.fsum(sample) / count
    mean_log = math.fsum(math.log(value) for value in sample) / count
    statistic = math.log(arithmetic_mean) - mean_log
    if not statistic > 0.0:
        raise ValueError("gamma MLE requires at least two distinct values")
    shape = (3.0 + math.sqrt(9.0 + 12.0 * statistic)) / (12.0 * statistic)
    lower, upper = 0.0, math.inf
    for _ in range(_MAX_NEWTON_ITERATIONS):
        residual = log_minus_digamma(shape) - statistic
        if residual == 0.0:
            break
        if residual > 0.0:
            lower = shape
        else:
            upper = shape
        slope = 1.0 / shape - trigamma(shape)
        newton = shape - residual / slope if slope < 0.0 else math.nan
        candidate = _safeguarded_step(shape, newton, lower, upper)
        converged = abs(candidate - shape) <= 4.0 * _EPSILON * shape
        shape = candidate
        if converged:
            break
    else:
        raise ArithmeticError("gamma shape Newton iteration did not converge")
    return Gamma(shape, shape / arithmetic_mean)


def _weibull_profile(
    centered_logs: tuple[float, ...], shape: float
) -> tuple[float, float, float]:
    """Evaluate the Weibull shape profile equation and its derivative.

    With centred logarithms ``y_i`` and weights ``w_i = exp(k y_i - max)``,
    the profile residual is ``sum w_i y_i / sum w_i - 1/k`` (``mean(y) = 0``)
    and its derivative is the weighted variance of ``y`` plus ``1/k^2``.

    Args:
        centered_logs: Logarithms minus their mean.
        shape: Positive trial shape ``k``.

    Returns:
        Residual, derivative, and ``log mean(exp(k y))``.

    Equation:
        [EQ-INF-MLE-WEIB-001].

    References:
        [REF-JKB-1994, Chapter 21].

    """
    exponents = tuple(shape * value for value in centered_logs)
    largest = max(exponents)
    weights = tuple(math.exp(exponent - largest) for exponent in exponents)
    total = math.fsum(weights)
    first = math.fsum(w * y for w, y in zip(weights, centered_logs, strict=True))
    second = math.fsum(
        w * y * y for w, y in zip(weights, centered_logs, strict=True)
    )
    weighted_mean = first / total
    weighted_variance = max(second / total - weighted_mean * weighted_mean, 0.0)
    residual = weighted_mean - 1.0 / shape
    slope = weighted_variance + 1.0 / (shape * shape)
    log_power_mean = largest + math.log(total / len(centered_logs))
    return residual, slope, log_power_mean


def fit_weibull_mle(values: Iterable[object]) -> Weibull:
    """Fit a two-parameter Weibull law by maximum likelihood.

    The scale is profiled out as ``lambda = (mean(x^k))^(1/k)``, leaving the
    shape equation ``sum x^k log x / sum x^k - 1/k - mean(log x) = 0``, whose
    left side is strictly increasing in ``k``. Newton's method is safeguarded
    by a bracket with bisection fallback. Logarithms are centred and the power
    sums use a log-sum-exp shift, so the fit is scale invariant and free of
    overflow. The start ``k0 = pi / (sqrt 6 sd(log x))`` matches the standard
    deviation of ``log X``, which follows a Gumbel law for minima.

    Args:
        values: Iterable of finite strictly positive observations containing
            at least two distinct values.

    Returns:
        Weibull law with the maximum-likelihood shape and scale.

    Raises:
        TypeError: If observations are not an iterable of real values.
        ValueError: If input is empty, non-finite, non-positive, or constant.
        ArithmeticError: If the Newton iteration does not converge.

    Equation:
        [EQ-INF-MLE-WEIB-001].

    References:
        [REF-JKB-1994, Chapter 21], [REF-NIST-DISTRIBUTIONS, Weibull
        Distribution].

    """
    sample = coerce_finite_sample(values)
    if any(value <= 0.0 for value in sample):
        raise ValueError("Weibull MLE requires strictly positive observations")
    logarithms = tuple(math.log(value) for value in sample)
    center = math.fsum(logarithms) / len(logarithms)
    centered = tuple(value - center for value in logarithms)
    spread = math.sqrt(math.fsum(y * y for y in centered) / len(centered))
    if spread == 0.0 or max(centered) <= 0.0:
        raise ValueError("Weibull MLE requires at least two distinct values")
    shape = math.pi / (math.sqrt(6.0) * spread)
    lower, upper = 0.0, math.inf
    for _ in range(_MAX_NEWTON_ITERATIONS):
        residual, slope, _ = _weibull_profile(centered, shape)
        if residual == 0.0:
            break
        if residual < 0.0:
            lower = shape
        else:
            upper = shape
        candidate = _safeguarded_step(shape, shape - residual / slope, lower, upper)
        converged = abs(candidate - shape) <= 4.0 * _EPSILON * shape
        shape = candidate
        if converged:
            break
    else:
        raise ArithmeticError("Weibull shape Newton iteration did not converge")
    _, _, log_power_mean = _weibull_profile(centered, shape)
    return Weibull(shape, math.exp(center + log_power_mean / shape))
