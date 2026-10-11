"""Shared validation and tail-area helpers for classical hypothesis tests.

A p-value is the probability, computed under the null hypothesis and its
assumptions, of a test statistic at least as extreme as the observed one in
the direction of the alternative. It is never the probability that the null
hypothesis is true.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from probstat.foundations.validation import require_finite_real, require_probability

from .bayesian import student_t_quantile
from .sampling_distributions import (
    standard_normal_cdf,
    standard_normal_sf,
    student_t_sf,
    student_t_two_sided_p,
)

Alternative = Literal["two-sided", "greater", "less"]
ALTERNATIVES: tuple[str, ...] = ("two-sided", "greater", "less")
P_VALUE_MEANING = (
    "probability under the null hypothesis of a statistic at least as extreme "
    "as the observed one; not the probability that the null is true"
)


def check_alternative(alternative: object) -> Alternative:
    """Validate an alternative-hypothesis label.

    Args:
        alternative: One of ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns:
        The validated label.

    Raises:
        TypeError: If ``alternative`` is not a string.
        ValueError: If the label is unknown.

    References:
        [REF-LEHMANN-ROMANO-2005, Chapter 3]. No equation is evaluated.

    """
    if not isinstance(alternative, str):
        raise TypeError("alternative must be a string")
    if alternative == "two-sided":
        return "two-sided"
    if alternative == "greater":
        return "greater"
    if alternative == "less":
        return "less"
    raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


def check_level(value: object, *, name: str) -> float:
    """Validate a significance or confidence level strictly inside ``(0, 1)``.

    Args:
        value: Candidate level.
        name: Parameter name used in error messages.

    Returns:
        Validated float level.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If ``value`` is outside ``(0, 1)``.

    References:
        [REF-LEHMANN-ROMANO-2005, Chapter 3]. No equation is evaluated.

    """
    level = require_probability(value, name=name)
    if level in (0.0, 1.0):
        raise ValueError(f"{name} must be strictly between zero and one")
    return level


def t_p_value(statistic: float, df: float, alternative: Alternative) -> float:
    """Return the Student-t p-value for an observed statistic.

    Args:
        statistic: Observed t statistic.
        df: Positive degrees of freedom.
        alternative: Direction of the alternative hypothesis.

    Returns:
        Two-sided ``P(|T| >= |t|)``, upper ``P(T >= t)``, or lower
        ``P(T <= t)`` under ``T ~ t_df``.

    References:
        [EQ-DIST-T-001], [REF-CASELLA-BERGER-2002, Chapter 8].

    """
    if alternative == "two-sided":
        return student_t_two_sided_p(statistic, df)
    if alternative == "greater":
        return student_t_sf(statistic, df)
    return student_t_sf(-statistic, df)


def normal_p_value(statistic: float, alternative: Alternative) -> float:
    """Return the standard-normal p-value for an observed z statistic.

    Args:
        statistic: Observed z statistic.
        alternative: Direction of the alternative hypothesis.

    Returns:
        Two-sided ``2 (1 - Phi(|z|))``, upper ``1 - Phi(z)``, or lower
        ``Phi(z)``, capped at one.

    References:
        [EQ-INF-DIST-NORM-001], [REF-CASELLA-BERGER-2002, Chapter 8].

    """
    z = require_finite_real(statistic, name="statistic")
    if alternative == "two-sided":
        return min(1.0, 2.0 * standard_normal_sf(abs(z)))
    if alternative == "greater":
        return standard_normal_sf(z)
    return standard_normal_cdf(z)


@lru_cache(maxsize=512)
def t_critical(probability: float, df: float) -> float:
    """Return a cached Student-t quantile used as a critical value.

    Args:
        probability: Level strictly inside ``(0, 1)``.
        df: Positive degrees of freedom.

    Returns:
        ``t`` with ``F_df(t) = probability``; repeated designs reuse the
        Brent inversion.

    References:
        [EQ-DIST-T-001], [REF-BRENT-1973, Chapter 4].

    """
    return student_t_quantile(probability, df)


def clamp_probability(value: float) -> float:
    """Clamp a computed tail probability to ``[0, 1]`` against roundoff.

    Args:
        value: Nearly valid probability.

    Returns:
        ``value`` restricted to the unit interval.

    References:
        Project numerical policy. No equation is evaluated.

    """
    return max(0.0, min(1.0, value))
