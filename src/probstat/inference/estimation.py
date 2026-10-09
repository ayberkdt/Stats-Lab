"""Maximum-likelihood estimators for foundational continuous families."""

from __future__ import annotations

import math
from collections.abc import Iterable

from probstat.distributions import LogNormal, Normal
from probstat.foundations import coerce_finite_sample


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
