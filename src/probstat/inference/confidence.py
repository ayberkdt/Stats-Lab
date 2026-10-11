"""Explicit confidence-interval procedures with named assumptions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist

from probstat.distributions._special import regularized_beta
from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.optimization import brent_root

from ._hypothesis import t_critical


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A two-sided numerical confidence interval and its construction label.

    Args:
        lower: Finite lower endpoint.
        upper: Finite upper endpoint not below ``lower``.
        confidence: Nominal confidence level strictly between zero and one.
        method: Non-empty method label describing the interval construction.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 9].

    """

    lower: float
    upper: float
    confidence: float
    method: str

    def __post_init__(self) -> None:
        """Validate endpoints, confidence level, and method label.

        Raises:
            TypeError: If fields have invalid types.
            ValueError: If endpoints, confidence, or label have invalid domains.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 9]. No equation is evaluated.

        """
        lower = require_finite_real(self.lower, name="lower")
        upper = require_finite_real(self.upper, name="upper")
        confidence = _confidence_level(self.confidence)
        if upper < lower:
            raise ValueError("upper must not be below lower")
        if not isinstance(self.method, str):
            raise TypeError("method must be a string")
        if not self.method.strip():
            raise ValueError("method must not be empty")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "confidence", confidence)


def normal_mean_confidence_interval(
    values: Iterable[object],
    *,
    confidence: float = 0.95,
    known_sigma: float | None = None,
) -> ConfidenceInterval:
    """Construct a z interval for a mean under explicit normal assumptions.

    When ``known_sigma`` is omitted, the sample standard deviation is inserted
    and the result is a large-sample normal approximation, not an exact
    Student-t interval.

    Args:
        values: Non-empty iterable of finite real observations.
        confidence: Nominal two-sided confidence level in ``(0, 1)``.
        known_sigma: Optional finite positive population standard deviation.

    Returns:
        Symmetric confidence interval around the sample mean.

    Raises:
        TypeError: If inputs have invalid types.
        ValueError: If sample, confidence, or scale domains are invalid.

    Equation:
        [EQ-INF-CI-NMEAN-001].

    References:
        [REF-CASELLA-BERGER-2002, Sections 9.2-9.3].

    """
    sample = coerce_finite_sample(values)
    checked_confidence = _confidence_level(confidence)
    if known_sigma is None:
        if len(sample) < 2:
            raise ValueError("estimated-scale interval requires at least two values")
        scale = statistics.stdev(sample)
        if scale == 0.0:
            raise ValueError("estimated-scale interval requires positive variation")
        method = "normal approximation with estimated standard deviation"
    else:
        scale = require_finite_real(known_sigma, name="known_sigma")
        if scale <= 0.0:
            raise ValueError("known_sigma must be positive")
        method = "normal z interval with known standard deviation"
    center = math.fsum(sample) / len(sample)
    critical = NormalDist().inv_cdf(0.5 + 0.5 * checked_confidence)
    half_width = critical * scale / math.sqrt(len(sample))
    return ConfidenceInterval(
        center - half_width,
        center + half_width,
        checked_confidence,
        method,
    )


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Construct a Wilson score interval for a binomial probability.

    Args:
        successes: Observed successes from zero through ``trials``.
        trials: Positive number of Bernoulli trials.
        confidence: Nominal two-sided confidence level in ``(0, 1)``.

    Returns:
        Wilson interval clipped to the probability domain.

    Raises:
        TypeError: If counts or confidence have invalid types.
        ValueError: If count or confidence domains are invalid.

    Equation:
        [EQ-INF-CI-WILSON-001].

    References:
        [REF-WILSON-1927].

    """
    checked_successes = require_non_negative_integer(successes, name="successes")
    checked_trials = require_non_negative_integer(trials, name="trials")
    checked_confidence = _confidence_level(confidence)
    if checked_trials == 0:
        raise ValueError("trials must be positive")
    if checked_successes > checked_trials:
        raise ValueError("successes must not exceed trials")
    estimate = checked_successes / checked_trials
    critical = NormalDist().inv_cdf(0.5 + 0.5 * checked_confidence)
    squared = critical**2
    denominator = 1.0 + squared / checked_trials
    center = (estimate + squared / (2.0 * checked_trials)) / denominator
    half_width = (
        critical
        * math.sqrt(
            estimate * (1.0 - estimate) / checked_trials
            + squared / (4.0 * checked_trials**2)
        )
        / denominator
    )
    lower = 0.0 if checked_successes == 0 else max(0.0, center - half_width)
    upper = (
        1.0
        if checked_successes == checked_trials
        else min(1.0, center + half_width)
    )
    return ConfidenceInterval(
        lower,
        upper,
        checked_confidence,
        "Wilson score interval",
    )


def student_t_mean_confidence_interval(
    values: Iterable[object],
    *,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Construct the exact Student-t confidence interval for a normal mean.

    Unlike the estimated-scale branch of
    :func:`normal_mean_confidence_interval`, this interval uses the
    ``t_{n-1}`` critical value, so its coverage is exact for independent
    normal observations of any size ``n >= 2``. The confidence level is a
    property of the procedure over repeated samples; it is not the
    probability that this particular interval contains the mean.

    Args:
        values: Iterable of at least two finite real observations.
        confidence: Nominal two-sided confidence level in ``(0, 1)``.

    Returns:
        Interval ``x_bar +- t_{n-1, 1-alpha/2} s / sqrt(n)``.

    Raises:
        TypeError: If inputs have invalid types.
        ValueError: If fewer than two values, zero variation, or an invalid
            confidence level is supplied.

    Equation:
        [EQ-INF-CI-TMEAN-001], [EQ-DIST-T-001].

    References:
        [REF-STUDENT-1908], [REF-CASELLA-BERGER-2002, Chapter 9].

    """
    sample = coerce_finite_sample(values)
    checked_confidence = _confidence_level(confidence)
    if len(sample) < 2:
        raise ValueError("Student-t interval requires at least two values")
    scale = statistics.stdev(sample)
    if scale == 0.0:
        raise ValueError("Student-t interval requires positive variation")
    center = math.fsum(sample) / len(sample)
    critical = t_critical(0.5 + 0.5 * checked_confidence, len(sample) - 1)
    half_width = critical * scale / math.sqrt(len(sample))
    return ConfidenceInterval(
        center - half_width,
        center + half_width,
        checked_confidence,
        "Student-t interval for a normal mean",
    )


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Construct the Clopper-Pearson exact interval for a binomial probability.

    The interval inverts two one-sided exact binomial tests, so its coverage
    is at least the nominal level for every true probability (it is
    conservative because the binomial law is discrete).

    Args:
        successes: Observed successes from zero through ``trials``.
        trials: Positive number of Bernoulli trials.
        confidence: Nominal two-sided confidence level in ``(0, 1)``.

    Returns:
        ``[B(alpha/2; x, n-x+1), B(1-alpha/2; x+1, n-x)]`` with beta quantiles
        ``B``, using ``0`` for ``x = 0`` and ``1`` for ``x = n``.

    Raises:
        TypeError: If counts or confidence have invalid types.
        ValueError: If count or confidence domains are invalid.

    Equation:
        [EQ-INF-CI-CP-001], [EQ-NUM-BETA-001].

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-AGRESTI-2013, Chapter 1].

    """
    checked_successes = require_non_negative_integer(successes, name="successes")
    checked_trials = require_non_negative_integer(trials, name="trials")
    checked_confidence = _confidence_level(confidence)
    if checked_trials == 0:
        raise ValueError("trials must be positive")
    if checked_successes > checked_trials:
        raise ValueError("successes must not exceed trials")
    alpha = 1.0 - checked_confidence
    if checked_successes == 0:
        lower = 0.0
    else:
        lower = _beta_quantile(
            0.5 * alpha,
            float(checked_successes),
            float(checked_trials - checked_successes + 1),
        )
    if checked_successes == checked_trials:
        upper = 1.0
    else:
        upper = _beta_quantile(
            1.0 - 0.5 * alpha,
            float(checked_successes + 1),
            float(checked_trials - checked_successes),
        )
    return ConfidenceInterval(
        lower, max(lower, upper), checked_confidence, "Clopper-Pearson exact interval"
    )


def _beta_quantile(probability: float, a: float, b: float) -> float:
    """Invert the regularized incomplete beta function by Brent's method.

    Args:
        probability: Target level strictly inside ``(0, 1)``.
        a: Positive first shape.
        b: Positive second shape.

    Returns:
        ``x`` with ``I_x(a, b) = probability``.

    References:
        [EQ-NUM-BETA-001], [EQ-OPT-BRENT-001], [REF-BRENT-1973, Chapter 4].

    """

    def excess(x: float) -> float:
        """Return ``I_x(a, b) - probability`` for the root search.

        References:
            [EQ-NUM-BETA-001]. No new equation is introduced.

        """
        return regularized_beta(x, a, b) - probability

    return brent_root(excess, 0.0, 1.0, tolerance=1e-15).root


def _confidence_level(value: float) -> float:
    """Validate a confidence level strictly between zero and one.

    Args:
        value: Candidate confidence level.

    Returns:
        Validated float confidence level.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If ``value`` is non-finite or outside ``(0, 1)``.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 9]. No equation is evaluated.

    """
    confidence = require_probability(value, name="confidence")
    if confidence in (0.0, 1.0):
        raise ValueError("confidence must be strictly between zero and one")
    return confidence
