"""One-sample empirical-distribution goodness-of-fit diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import cast

from probstat.foundations import (
    CumulativeDistribution,
    SurvivalFunction,
    coerce_finite_sample,
)
from probstat.foundations.validation import (
    require_non_negative_integer,
    require_probability,
)


@dataclass(frozen=True, slots=True)
class KolmogorovSmirnovResult:
    """One-sample Kolmogorov-Smirnov statistic and asymptotic p-value.

    Args:
        statistic: Supremum empirical-CDF distance in ``[0, 1]``.
        p_value: Approximate null-tail probability in ``[0, 1]``.
        sample_size: Positive number of observations.
        method: Label identifying the p-value approximation.

    References:
        [REF-NIST-GOF, Kolmogorov-Smirnov Goodness-of-Fit Test].

    """

    statistic: float
    p_value: float
    sample_size: int
    method: str = "Stephens-corrected asymptotic one-sample KS"

    def __post_init__(self) -> None:
        """Validate probability-valued fields, sample size, and method label.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If a field has an invalid domain.

        References:
            [EQ-INF-GOF-KS-001], [REF-NIST-GOF].

        """
        statistic = require_probability(self.statistic, name="statistic")
        p_value = require_probability(self.p_value, name="p_value")
        sample_size = require_non_negative_integer(
            self.sample_size, name="sample_size"
        )
        if sample_size == 0:
            raise ValueError("sample_size must be positive")
        if not isinstance(self.method, str):
            raise TypeError("method must be a string")
        if not self.method.strip():
            raise ValueError("method must not be empty")
        object.__setattr__(self, "statistic", statistic)
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "sample_size", sample_size)


@dataclass(frozen=True, slots=True)
class AndersonDarlingResult:
    """One-sample Anderson-Darling statistic for a fully specified CDF.

    Critical values and p-values depend on the null family and whether its
    parameters were estimated. This generic result intentionally reports no
    universal p-value.

    Args:
        statistic: Non-negative tail-weighted discrepancy, possibly infinite.
        sample_size: Positive number of observations.

    References:
        [REF-NIST-GOF, Anderson-Darling Test].

    """

    statistic: float
    sample_size: int

    def __post_init__(self) -> None:
        """Validate non-negative discrepancy and positive sample size.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If the statistic is NaN/negative or sample size is zero.

        References:
            [EQ-INF-GOF-AD-001], [REF-NIST-GOF].

        """
        if isinstance(self.statistic, bool) or not isinstance(self.statistic, Real):
            raise TypeError("statistic must be a real number")
        statistic = float(self.statistic)
        if math.isnan(statistic) or statistic < 0.0:
            raise ValueError("statistic must be non-negative and not NaN")
        sample_size = require_non_negative_integer(
            self.sample_size, name="sample_size"
        )
        if sample_size == 0:
            raise ValueError("sample_size must be positive")
        object.__setattr__(self, "statistic", statistic)
        object.__setattr__(self, "sample_size", sample_size)


def kolmogorov_smirnov_test(
    values: Iterable[object],
    distribution: CumulativeDistribution,
) -> KolmogorovSmirnovResult:
    """Run a one-sample KS test against a fully specified continuous CDF.

    The returned p-value is asymptotic. If distribution parameters were fitted
    from the same sample, its ordinary null calibration is invalid; use a
    parametric bootstrap or a family-specific correction instead.

    Args:
        values: Non-empty iterable of finite real observations.
        distribution: Object exposing a valid cumulative distribution method.

    Returns:
        KS statistic, approximate p-value, sample size, and method label.

    Raises:
        TypeError: If inputs violate the iterable or CDF protocol.
        ValueError: If observations or returned probabilities are invalid.

    Equation:
        [EQ-INF-GOF-KS-001], [EQ-INF-GOF-KS-002].

    References:
        [REF-NIST-GOF, Kolmogorov-Smirnov Goodness-of-Fit Test].

    """
    sample = tuple(sorted(coerce_finite_sample(values)))
    if not isinstance(distribution, CumulativeDistribution):
        raise TypeError("distribution must expose cdf(x)")
    count = len(sample)
    lower_gap = 0.0
    upper_gap = 0.0
    for rank, value in enumerate(sample, start=1):
        probability = _cdf_probability(distribution, value)
        lower_gap = max(lower_gap, rank / count - probability)
        upper_gap = max(upper_gap, probability - (rank - 1) / count)
    statistic = max(lower_gap, upper_gap)
    root_count = math.sqrt(count)
    corrected = (root_count + 0.12 + 0.11 / root_count) * statistic
    return KolmogorovSmirnovResult(
        statistic,
        _kolmogorov_survival(corrected),
        count,
    )


def anderson_darling_test(
    values: Iterable[object],
    distribution: CumulativeDistribution,
) -> AndersonDarlingResult:
    """Compute the one-sample Anderson-Darling statistic.

    Args:
        values: Non-empty iterable of finite real observations.
        distribution: Fully specified continuous null CDF. A direct survival
            method is used for upper tails when available.

    Returns:
        Tail-weighted discrepancy and sample size; no universal p-value.

    Raises:
        TypeError: If inputs violate the iterable or CDF protocol.
        ValueError: If observations or returned probabilities are invalid.

    Equation:
        [EQ-INF-GOF-AD-001].

    References:
        [REF-NIST-GOF, Anderson-Darling Test].

    """
    sample = tuple(sorted(coerce_finite_sample(values)))
    if not isinstance(distribution, CumulativeDistribution):
        raise TypeError("distribution must expose cdf(x)")
    count = len(sample)
    weighted_logs: list[float] = []
    has_survival = isinstance(distribution, SurvivalFunction)
    for rank, lower_value in enumerate(sample, start=1):
        upper_value = sample[-rank]
        lower_probability = _cdf_probability(distribution, lower_value)
        if has_survival:
            survival = cast(SurvivalFunction, distribution)
            upper_probability = require_probability(
                survival.sf(upper_value),
                name="distribution.sf(x)",
            )
        else:
            upper_probability = 1.0 - _cdf_probability(distribution, upper_value)
        if lower_probability == 0.0 or upper_probability == 0.0:
            return AndersonDarlingResult(math.inf, count)
        weighted_logs.append(
            (2 * rank - 1)
            * (math.log(lower_probability) + math.log(upper_probability))
        )
    statistic = -count - math.fsum(weighted_logs) / count
    return AndersonDarlingResult(max(0.0, statistic), count)


def _cdf_probability(distribution: CumulativeDistribution, value: float) -> float:
    """Evaluate and validate one null-CDF probability.

    Args:
        distribution: Cumulative distribution object.
        value: Finite sample observation.

    Returns:
        Validated probability in ``[0, 1]``.

    Raises:
        TypeError: If the result is not real.
        ValueError: If the result is non-finite or outside ``[0, 1]``.

    References:
        [EQ-INF-GOF-KS-001], [REF-NIST-GOF].

    """
    return require_probability(distribution.cdf(value), name="distribution.cdf(x)")


def _kolmogorov_survival(argument: float) -> float:
    """Evaluate the limiting Kolmogorov survival function stably.

    Args:
        argument: Non-negative corrected root-sample statistic.

    Returns:
        Limiting upper-tail probability in ``[0, 1]``.

    Equation:
        [EQ-INF-GOF-KS-002].

    References:
        [REF-NIST-GOF, Kolmogorov-Smirnov Goodness-of-Fit Test].

    """
    if argument <= 0.0:
        return 1.0
    if argument < 1.18:
        factor = math.sqrt(2.0 * math.pi) / argument
        terms = (
            math.exp(
                -((2 * index - 1) ** 2)
                * math.pi**2
                / (8.0 * argument**2)
            )
            for index in range(1, 1000)
        )
        cumulative = factor * math.fsum(terms)
        return min(1.0, max(0.0, 1.0 - cumulative))
    alternating: list[float] = []
    for index in range(1, 1000):
        term = 2.0 * math.exp(-2.0 * index**2 * argument**2)
        alternating.append(term if index % 2 else -term)
        if term < 1e-16:
            break
    return min(1.0, max(0.0, math.fsum(alternating)))
