"""Standardized effect sizes, power, and sample-size planning for mean tests.

Power is the probability that a test rejects its null hypothesis when a stated
alternative effect is true. It is a design-stage property of a procedure under
an assumed effect size and must not be computed from the observed effect of
the same data ("post-hoc power" is a monotone transform of the p-value and
adds no information).

The t-test power uses the exact noncentral t distribution evaluated by
numerical quadrature (see
:func:`probstat.inference.sampling_distributions.noncentral_t_cdf`); no
normal approximation and no simulation is involved.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

from ._hypothesis import Alternative, check_alternative, check_level, t_critical
from .sampling_distributions import noncentral_t_cdf, standard_normal_cdf

Design = Literal["one-sample", "two-sample", "paired"]
_DESIGNS: tuple[str, ...] = ("one-sample", "two-sample", "paired")


def _check_design(design: object) -> Design:
    """Validate a mean-test design label.

    Args:
        design: ``"one-sample"``, ``"two-sample"``, or ``"paired"``.

    Returns:
        The validated label.

    Raises:
        TypeError: If ``design`` is not a string.
        ValueError: If the label is unknown.

    References:
        [REF-COHEN-1988, Chapter 2]. No equation is evaluated.

    """
    if not isinstance(design, str):
        raise TypeError("design must be a string")
    if design == "one-sample":
        return "one-sample"
    if design == "two-sample":
        return "two-sample"
    if design == "paired":
        return "paired"
    raise ValueError("design must be 'one-sample', 'two-sample', or 'paired'")


def _sample_with_variation(values: Iterable[object], name: str) -> tuple[float, ...]:
    """Materialize a sample of at least two values with positive variance.

    Args:
        values: Finite real observations.
        name: Parameter name used in error messages.

    Returns:
        Validated tuple.

    Raises:
        ValueError: If fewer than two values or zero variation is supplied.

    References:
        [REF-COHEN-1988, Chapter 2]. No equation is evaluated.

    """
    sample = coerce_finite_sample(values, name=name)
    if len(sample) < 2:
        raise ValueError(f"{name} must contain at least two observations")
    if statistics.variance(sample) == 0.0:
        raise ValueError(f"{name} must have positive variation")
    return sample


def pooled_standard_deviation(
    first: Iterable[object], second: Iterable[object]
) -> float:
    """Return the pooled two-sample standard deviation.

    Args:
        first: First sample with at least two values.
        second: Second sample with at least two values.

    Returns:
        ``s_p = sqrt(((n1-1) s1^2 + (n2-1) s2^2) / (n1 + n2 - 2))``.

    Raises:
        TypeError: If observations are not real.
        ValueError: If a sample has fewer than two values or both samples are
            constant.

    Equation:
        [EQ-INF-T-002].

    References:
        [REF-CASELLA-BERGER-2002, Chapter 11], [REF-COHEN-1988, Chapter 2].

    """
    a = coerce_finite_sample(first, name="first")
    b = coerce_finite_sample(second, name="second")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each sample must contain at least two observations")
    pooled_variance = (
        (len(a) - 1) * statistics.variance(a) + (len(b) - 1) * statistics.variance(b)
    ) / (len(a) + len(b) - 2)
    if pooled_variance == 0.0:
        raise ValueError("pooled standard deviation is zero")
    return math.sqrt(pooled_variance)


def cohens_d(
    first: Iterable[object],
    second: Iterable[object] | None = None,
    *,
    null_value: float = 0.0,
    paired: bool = False,
) -> float:
    """Return Cohen's standardized mean difference.

    Args:
        first: First sample (or the only sample).
        second: Optional second sample. Omit for a one-sample effect.
        null_value: Reference mean for a one-sample effect.
        paired: Treat ``first`` and ``second`` as matched pairs and
            standardize the mean difference by the standard deviation of the
            differences (``d_z``).

    Returns:
        One-sample ``(x_bar - mu_0) / s``, two-sample
        ``(x_bar_1 - x_bar_2) / s_p``, or paired ``d_bar / s_d``.

    Raises:
        TypeError: If observations are not real.
        ValueError: If sample sizes or variation are insufficient, or paired
            samples have unequal lengths.

    Equation:
        [EQ-INF-ES-001].

    References:
        [REF-COHEN-1988, Chapter 2].

    """
    reference = require_finite_real(null_value, name="null_value")
    if second is None:
        if paired:
            raise ValueError("paired effect size requires a second sample")
        sample = _sample_with_variation(first, "first")
        return (statistics.fmean(sample) - reference) / statistics.stdev(sample)
    a = coerce_finite_sample(first, name="first")
    b = coerce_finite_sample(second, name="second")
    if paired:
        if len(a) != len(b):
            raise ValueError("paired samples must have equal lengths")
        differences = _sample_with_variation(
            [x - y for x, y in zip(a, b, strict=True)], "differences"
        )
        return statistics.fmean(differences) / statistics.stdev(differences)
    return (statistics.fmean(a) - statistics.fmean(b)) / pooled_standard_deviation(a, b)


def hedges_correction(degrees_of_freedom: float) -> float:
    """Return Hedges' exact small-sample bias-correction factor ``J(m)``.

    Args:
        degrees_of_freedom: Degrees of freedom ``m > 1`` of the variance
            estimate used to standardize.

    Returns:
        ``J(m) = Gamma(m/2) / (sqrt(m/2) Gamma((m-1)/2))``, approximately
        ``1 - 3 / (4 m - 1)``.

    Raises:
        TypeError: If the argument is not real.
        ValueError: If ``m <= 1``.

    Equation:
        [EQ-INF-ES-001].

    References:
        [REF-HEDGES-1981].

    """
    m = require_finite_real(degrees_of_freedom, name="degrees_of_freedom")
    if m <= 1.0:
        raise ValueError("degrees_of_freedom must exceed one")
    return math.exp(
        math.lgamma(0.5 * m) - 0.5 * math.log(0.5 * m) - math.lgamma(0.5 * (m - 1.0))
    )


def hedges_g(
    first: Iterable[object],
    second: Iterable[object] | None = None,
    *,
    null_value: float = 0.0,
    paired: bool = False,
) -> float:
    """Return Hedges' bias-corrected standardized mean difference.

    Args:
        first: First sample (or the only sample).
        second: Optional second sample.
        null_value: Reference mean for a one-sample effect.
        paired: Use the paired (``d_z``) standardization.

    Returns:
        ``g = J(m) d`` with ``m = n1 + n2 - 2`` for two independent samples
        and ``m = n - 1`` otherwise.

    Raises:
        TypeError: If observations are not real.
        ValueError: If sample sizes or variation are insufficient.

    Equation:
        [EQ-INF-ES-001].

    References:
        [REF-HEDGES-1981], [REF-COHEN-1988, Chapter 2].

    """
    first_sample = coerce_finite_sample(first, name="first")
    second_sample = (
        None if second is None else coerce_finite_sample(second, name="second")
    )
    d = cohens_d(first_sample, second_sample, null_value=null_value, paired=paired)
    if second_sample is None or paired:
        m = float(len(first_sample) - 1)
    else:
        m = float(len(first_sample) + len(second_sample) - 2)
    return hedges_correction(m) * d


def _noncentrality(effect_size: float, sample_size: int, design: Design) -> float:
    """Return the noncentrality ``delta`` of a standardized mean test.

    Args:
        effect_size: Standardized effect ``d``.
        sample_size: Observations (pairs, or per group for two samples).
        design: Test design.

    Returns:
        ``d sqrt(n)`` for one-sample or paired, ``d sqrt(n / 2)`` for two
        equal groups.

    References:
        [EQ-INF-POWER-001], [REF-COHEN-1988, Chapter 2].

    """
    if design == "two-sample":
        return effect_size * math.sqrt(0.5 * sample_size)
    return effect_size * math.sqrt(sample_size)


def _check_power_inputs(
    effect_size: float,
    sample_size: int,
    alpha: float,
    alternative: object,
    design: object,
    minimum: int,
) -> tuple[float, int, float, Alternative, Design]:
    """Validate shared power-function inputs.

    Args:
        effect_size: Standardized effect.
        sample_size: Observations (or per group).
        alpha: Significance level.
        alternative: Alternative label.
        design: Design label.
        minimum: Smallest admissible sample size.

    Returns:
        Validated tuple.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument has an invalid domain.

    References:
        [REF-COHEN-1988, Chapter 2]. No equation is evaluated.

    """
    d = require_finite_real(effect_size, name="effect_size")
    n = require_non_negative_integer(sample_size, name="sample_size")
    if n < minimum:
        raise ValueError(f"sample_size must be at least {minimum}")
    level = check_level(alpha, name="alpha")
    return d, n, level, check_alternative(alternative), _check_design(design)


def z_test_power(
    effect_size: float,
    sample_size: int,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    design: Design = "one-sample",
) -> float:
    """Return the power of a z test for a standardized mean effect.

    Args:
        effect_size: True standardized effect ``d`` (mean shift divided by
            the known standard deviation of one observation or difference).
        sample_size: Observations, pairs, or observations per group (two
            equal groups) for ``design="two-sample"``.
        alpha: Significance level in ``(0, 1)``.
        alternative: Direction of the alternative hypothesis.
        design: ``"one-sample"``, ``"paired"``, or ``"two-sample"``.

    Returns:
        Two-sided ``Phi(delta - z_{1-alpha/2}) + Phi(-delta - z_{1-alpha/2})``
        or one-sided ``Phi(+-delta - z_{1-alpha})``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument has an invalid domain.

    Equation:
        [EQ-INF-POWER-001].

    References:
        [REF-COHEN-1988, Chapter 2], [REF-LEHMANN-ROMANO-2005, Chapter 3].

    """
    d, n, level, side, kind = _check_power_inputs(
        effect_size, sample_size, alpha, alternative, design, 1
    )
    delta = _noncentrality(d, n, kind)
    normal = NormalDist()
    if side == "two-sided":
        critical = normal.inv_cdf(1.0 - 0.5 * level)
        return standard_normal_cdf(delta - critical) + standard_normal_cdf(
            -delta - critical
        )
    critical = normal.inv_cdf(1.0 - level)
    if side == "greater":
        return standard_normal_cdf(delta - critical)
    return standard_normal_cdf(-delta - critical)


def t_test_power(
    effect_size: float,
    sample_size: int,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    design: Design = "one-sample",
) -> float:
    """Return the exact power of a Student t test for a standardized effect.

    The statistic follows a noncentral t law with ``df = n - 1``
    (one-sample, paired) or ``2n - 2`` (two equal groups of ``n``) and
    noncentrality ``delta = d sqrt(n)`` or ``d sqrt(n/2)``. The noncentral CDF
    is evaluated by quadrature, not by a normal approximation or simulation.

    Args:
        effect_size: True standardized effect ``d``.
        sample_size: Observations, pairs, or per-group size (at least two).
        alpha: Significance level in ``(0, 1)``.
        alternative: Direction of the alternative hypothesis.
        design: ``"one-sample"``, ``"paired"``, or ``"two-sample"``.

    Returns:
        Two-sided ``1 - F'(c) + F'(-c)`` with ``c = t_{df, 1-alpha/2}``, or
        the matching one-sided tail with ``c = t_{df, 1-alpha}``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument has an invalid domain.

    Equation:
        [EQ-INF-POWER-002], [EQ-INF-DIST-NCT-001].

    References:
        [REF-COHEN-1988, Chapter 2],
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995, noncentral t chapter].

    """
    d, n, level, side, kind = _check_power_inputs(
        effect_size, sample_size, alpha, alternative, design, 2
    )
    delta = _noncentrality(d, n, kind)
    df = float(2 * n - 2 if kind == "two-sample" else n - 1)
    if side == "two-sided":
        critical = t_critical(1.0 - 0.5 * level, df)
        return min(
            1.0,
            1.0
            - noncentral_t_cdf(critical, df, delta)
            + noncentral_t_cdf(-critical, df, delta),
        )
    critical = t_critical(1.0 - level, df)
    if side == "greater":
        return 1.0 - noncentral_t_cdf(critical, df, delta)
    return noncentral_t_cdf(-critical, df, delta)


@dataclass(frozen=True, slots=True)
class SampleSizeResult:
    """Smallest sample size achieving a target power for a mean test.

    Args:
        sample_size: Required observations, pairs, or per-group size.
        achieved_power: Power at ``sample_size`` (at least the target).
        target_power: Requested power.
        alpha: Significance level.
        effect_size: Assumed standardized effect.
        alternative: Direction of the alternative hypothesis.
        design: Test design.
        test: ``"t"`` or ``"z"``.
        method: Description of the power computation.

    References:
        [REF-COHEN-1988, Chapter 2].

    """

    sample_size: int
    achieved_power: float
    target_power: float
    alpha: float
    effect_size: float
    alternative: str
    design: str
    test: str
    method: str


def required_sample_size(
    effect_size: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    design: Design = "one-sample",
    test: Literal["t", "z"] = "t",
    max_sample_size: int = 1_000_000,
) -> SampleSizeResult:
    """Return the smallest sample size whose power reaches a target.

    The search starts from the closed-form z-test size
    ``k (z_{1-alpha'} + z_{power})^2 / d^2`` (``k = 2`` for two groups) and
    moves by unit steps until the exact power function crosses the target.

    Args:
        effect_size: Assumed nonzero standardized effect ``d``; its sign must
            agree with a one-sided alternative.
        power: Target power strictly between ``alpha`` and one.
        alpha: Significance level in ``(0, 1)``.
        alternative: Direction of the alternative hypothesis.
        design: ``"one-sample"``, ``"paired"``, or ``"two-sample"``.
        test: ``"t"`` (estimated variance) or ``"z"`` (known variance).
        max_sample_size: Upper limit of the search.

    Returns:
        Required size and the power it achieves.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If the effect is zero or has the wrong sign, the target is
            unattainable, or the limit is exceeded.

    Equation:
        [EQ-INF-POWER-003], [EQ-INF-POWER-001], [EQ-INF-POWER-002].

    References:
        [REF-COHEN-1988, Chapter 2].

    """
    d = require_finite_real(effect_size, name="effect_size")
    target = check_level(power, name="power")
    level = check_level(alpha, name="alpha")
    side = check_alternative(alternative)
    kind = _check_design(design)
    if test not in ("t", "z"):
        raise ValueError("test must be 't' or 'z'")
    limit = require_non_negative_integer(max_sample_size, name="max_sample_size")
    if d == 0.0:
        raise ValueError("effect_size must be nonzero")
    if (side == "greater" and d < 0.0) or (side == "less" and d > 0.0):
        raise ValueError("effect_size sign contradicts the one-sided alternative")
    if target <= level:
        raise ValueError("power must exceed alpha")
    minimum = 2 if test == "t" else 1

    def power_at(n: int) -> float:
        """Return the exact power at sample size ``n``.

        References:
            [EQ-INF-POWER-001], [EQ-INF-POWER-002].

        """
        if test == "t":
            return t_test_power(d, n, alpha=level, alternative=side, design=kind)
        return z_test_power(d, n, alpha=level, alternative=side, design=kind)

    normal = NormalDist()
    tail = 0.5 * level if side == "two-sided" else level
    factor = 2.0 if kind == "two-sample" else 1.0
    start = (
        factor * (normal.inv_cdf(1.0 - tail) + normal.inv_cdf(target)) ** 2 / (d * d)
    )
    n = max(minimum, math.ceil(start) - 1)
    if n > limit:
        raise ValueError("required sample size exceeds max_sample_size")
    while power_at(n) < target:
        n += 1
        if n > limit:
            raise ValueError("required sample size exceeds max_sample_size")
    while n > minimum and power_at(n - 1) >= target:
        n -= 1
    method = (
        "exact noncentral-t power (quadrature)"
        if test == "t"
        else "normal (known-variance) power"
    )
    return SampleSizeResult(n, power_at(n), target, level, d, side, kind, test, method)
