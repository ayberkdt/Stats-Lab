"""Restricted mean survival time and its two-sample contrasts.

The restricted mean survival time ``mu(tau) = E[min(T, tau)] = int_0^tau
S(u) du`` is the expected event-free time within the horizon ``tau``. It is
estimated by the area under the Kaplan-Meier curve with the Greenwood-type
variance of [EQ-SURV-RMST-001]. For two independent groups the difference
and the ratio of the restricted means summarize the effect on the time
scale without a proportional-hazards assumption [EQ-SURV-RMST-002].

The estimand depends on ``tau``: it must be chosen before looking at the
curves and lie inside the follow-up of both groups. Intervals use the
large-sample normal approximation (on the log scale for the ratio).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from .data import SurvivalData
from .nonparametric import RestrictedMeanSurvival, _z_critical, kaplan_meier


@dataclass(frozen=True, slots=True)
class RestrictedMeanComparison:
    """Difference and ratio of two restricted mean survival times.

    Args:
        tau: Common truncation time.
        first: Restricted mean of the first group.
        second: Restricted mean of the second (reference) group.
        difference: ``mu_1(tau) - mu_2(tau)``.
        difference_standard_error: ``sqrt(Var_1 + Var_2)``.
        difference_lower: Lower limit of ``difference -/+ z se``.
        difference_upper: Upper limit of ``difference -/+ z se``.
        ratio: ``mu_1(tau) / mu_2(tau)``.
        log_ratio_standard_error: Delta-method standard error of
            ``log ratio``, ``sqrt(Var_1 / mu_1^2 + Var_2 / mu_2^2)``.
        ratio_lower: ``ratio exp(-z se_log)``.
        ratio_upper: ``ratio exp(+z se_log)``.
        statistic: ``z = difference / difference_standard_error`` (``nan``
            when the standard error is zero).
        p_value: Two-sided normal p-value of ``mu_1 = mu_2`` (``nan`` when
            the standard error is zero).
        confidence: Nominal confidence level of both intervals.

    References:
        [EQ-SURV-RMST-002], [REF-KLEIN-MOESCHBERGER-2003].

    """

    tau: float
    first: RestrictedMeanSurvival
    second: RestrictedMeanSurvival
    difference: float
    difference_standard_error: float
    difference_lower: float
    difference_upper: float
    ratio: float
    log_ratio_standard_error: float
    ratio_lower: float
    ratio_upper: float
    statistic: float
    p_value: float
    confidence: float


def restricted_mean_survival(
    data: SurvivalData, tau: float, *, confidence: float = 0.95
) -> RestrictedMeanSurvival:
    """Estimate ``E[min(T, tau)]`` by the area under the Kaplan-Meier curve.

    ``mu(tau) = int_0^tau S(u) du`` with variance
    ``sum_{t_j <= tau} A_j^2 d_j / (n_j (n_j - d_j))``,
    ``A_j = int_{t_j}^tau S(u) du``, and the interval ``mu +/- z se``.
    Without censoring the estimate equals the sample mean of
    ``min(t_i, tau)``.

    Args:
        data: Right-censored sample.
        tau: Positive truncation time; it may exceed the last observed time
            only when the Kaplan-Meier curve has reached zero.
        confidence: Confidence level in ``(0, 1)``.

    Returns:
        Estimate, variance, standard error, and interval.

    Raises:
        ValueError: If ``tau`` is not positive, lies beyond the follow-up
            while the curve is positive, or ``confidence`` is not strictly
            inside ``(0, 1)``.

    Equation:
        [EQ-SURV-RMST-001].

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 4], [REF-GREENWOOD-1926].

    """
    return kaplan_meier(data, confidence=confidence).restricted_mean(tau)


def compare_restricted_means(
    first: SurvivalData,
    second: SurvivalData,
    tau: float,
    *,
    confidence: float = 0.95,
) -> RestrictedMeanComparison:
    """Compare the restricted mean survival times of two independent groups.

    The groups are independent, so the variance of the difference is the sum
    of the two variances; the ratio is analysed on the log scale, where the
    delta method gives ``Var(log mu) = Var(mu) / mu^2``:

    ``D = mu_1 - mu_2``, ``se(D) = sqrt(V_1 + V_2)``,
    ``R = mu_1 / mu_2``, ``se(log R) = sqrt(V_1 / mu_1^2 + V_2 / mu_2^2)``,
    with intervals ``D +/- z se(D)`` and ``R exp(+/- z se(log R))`` and the
    test statistic ``D / se(D)`` referred to the standard normal law.

    Args:
        first: Right-censored sample of the first group.
        second: Right-censored sample of the reference group.
        tau: Positive truncation time inside the follow-up of both groups.
        confidence: Confidence level in ``(0, 1)``.

    Returns:
        Both restricted means, their difference and ratio with standard
        errors and intervals, and the two-sided test of equality.

    Raises:
        ValueError: If ``tau`` is not positive or not identified in one of
            the groups, or ``confidence`` is not strictly inside ``(0, 1)``.

    Equation:
        [EQ-SURV-RMST-001], [EQ-SURV-RMST-002].

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 4], [REF-CASELLA-BERGER-2002]
        (delta method).

    """
    z = _z_critical(confidence)
    one = restricted_mean_survival(first, tau, confidence=confidence)
    two = restricted_mean_survival(second, tau, confidence=confidence)
    difference = one.estimate - two.estimate
    error = math.sqrt(one.variance + two.variance)
    # tau > 0 and S = 1 before the first event time, so both means are positive
    # unless a group has an event at time zero for every subject.
    if not (one.estimate > 0.0 and two.estimate > 0.0):
        raise ValueError(
            "a restricted mean is zero (every subject failed at time zero); "
            "the ratio is undefined"
        )
    ratio = one.estimate / two.estimate
    log_error = math.sqrt(
        one.variance / one.estimate**2 + two.variance / two.estimate**2
    )
    if error > 0.0:
        statistic = difference / error
        p_value = 2.0 * NormalDist().cdf(-abs(statistic))
    else:
        statistic = p_value = math.nan
    return RestrictedMeanComparison(
        one.tau,
        one,
        two,
        difference,
        error,
        difference - z * error,
        difference + z * error,
        ratio,
        log_error,
        ratio * math.exp(-z * log_error),
        ratio * math.exp(z * log_error),
        statistic,
        p_value,
        float(confidence),
    )
