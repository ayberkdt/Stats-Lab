"""Classical parametric and contingency-table hypothesis tests.

Every result reports the statistic, its reference distribution's degrees of
freedom where relevant, the alternative, a method label, and a p-value. A
p-value is the probability, computed under the null hypothesis and the test's
assumptions, of a statistic at least as extreme as the observed one; it is not
the probability that the null hypothesis is true, and a large p-value is not
evidence of equivalence (use :func:`tost_equivalence_test` for that).
"""

from __future__ import annotations

import math
import statistics
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._hypothesis import (
    Alternative,
    check_alternative,
    check_level,
    clamp_probability,
    t_critical,
    t_p_value,
)
from .confidence import ConfidenceInterval, clopper_pearson_interval
from .power import cohens_d
from .sampling_distributions import chi_square_sf, f_sf

_RELATIVE_TIE = 1.0 + 1e-7


def _check_p_value(value: float) -> float:
    """Validate a p-value field.

    Args:
        value: Candidate p-value.

    Returns:
        Validated probability.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value lies outside ``[0, 1]``.

    References:
        [REF-LEHMANN-ROMANO-2005, Chapter 3]. No equation is evaluated.

    """
    return require_probability(value, name="p_value")


@dataclass(frozen=True, slots=True)
class TTestResult:
    """Result of a one-sample, paired, or two-sample t test.

    Args:
        statistic: Observed t statistic.
        p_value: Probability under the null of a t statistic at least as
            extreme as the observed one in the direction of ``alternative``;
            not the probability that the null hypothesis is true.
        df: Degrees of freedom (fractional for Welch).
        alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
        method: Test label.
        estimate: Sample mean, mean difference, or difference of means.
        standard_error: Estimated standard error of ``estimate``.
        null_value: Hypothesized value of the estimated quantity.
        confidence_interval: Two-sided Student-t interval for the estimated
            quantity at the requested confidence, whatever the alternative.
        effect_size: Cohen's ``d`` (pooled for two samples, ``d_z`` paired).
        sample_sizes: Sample sizes used.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 8], [REF-STUDENT-1908].

    """

    statistic: float
    p_value: float
    df: float
    alternative: str
    method: str
    estimate: float
    standard_error: float
    null_value: float
    confidence_interval: ConfidenceInterval
    effect_size: float
    sample_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the p-value and degrees of freedom.

        Raises:
            ValueError: If the p-value or degrees of freedom are invalid.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 8]. No equation is evaluated.

        """
        object.__setattr__(self, "p_value", _check_p_value(self.p_value))
        if not self.df > 0.0:
            raise ValueError("df must be positive")


def welch_satterthwaite_df(
    first_variance: float, first_size: int, second_variance: float, second_size: int
) -> float:
    """Return the Welch-Satterthwaite effective degrees of freedom.

    Args:
        first_variance: Unbiased sample variance of the first sample.
        first_size: First sample size (at least two).
        second_variance: Unbiased sample variance of the second sample.
        second_size: Second sample size (at least two).

    Returns:
        ``(v1/n1 + v2/n2)^2 / ((v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1))``, which
        lies between ``min(n1, n2) - 1`` and ``n1 + n2 - 2``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If a size is below two, a variance is negative, or both
            variances are zero.

    Equation:
        [EQ-INF-T-003].

    References:
        [REF-WELCH-1947], [REF-SATTERTHWAITE-1946].

    """
    v1 = require_finite_real(first_variance, name="first_variance")
    v2 = require_finite_real(second_variance, name="second_variance")
    n1 = require_non_negative_integer(first_size, name="first_size")
    n2 = require_non_negative_integer(second_size, name="second_size")
    if n1 < 2 or n2 < 2:
        raise ValueError("sample sizes must be at least two")
    if v1 < 0.0 or v2 < 0.0:
        raise ValueError("variances must be non-negative")
    a = v1 / n1
    b = v2 / n2
    if a + b == 0.0:
        raise ValueError("at least one variance must be positive")
    return (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n2 - 1))


def _t_result(
    *,
    estimate: float,
    standard_error: float,
    df: float,
    null_value: float,
    alternative: Alternative,
    confidence: float,
    method: str,
    effect_size: float,
    sample_sizes: tuple[int, ...],
) -> TTestResult:
    """Assemble a t-test result from an estimate and its standard error.

    Args:
        estimate: Point estimate.
        standard_error: Positive standard error.
        df: Positive degrees of freedom.
        null_value: Hypothesized value.
        alternative: Alternative label.
        confidence: Confidence level of the reported interval.
        method: Test label.
        effect_size: Standardized effect.
        sample_sizes: Sample sizes.

    Returns:
        Complete result.

    References:
        [EQ-INF-T-001], [EQ-INF-CI-TMEAN-001].

    """
    statistic = (estimate - null_value) / standard_error
    critical = t_critical(0.5 + 0.5 * confidence, df)
    interval = ConfidenceInterval(
        estimate - critical * standard_error,
        estimate + critical * standard_error,
        confidence,
        f"Student-t interval ({method})",
    )
    return TTestResult(
        statistic,
        clamp_probability(t_p_value(statistic, df, alternative)),
        df,
        alternative,
        method,
        estimate,
        standard_error,
        null_value,
        interval,
        effect_size,
        sample_sizes,
    )


def _varying_sample(values: Iterable[object], name: str) -> tuple[float, ...]:
    """Materialize a sample of at least two values with positive variance.

    Args:
        values: Observations.
        name: Parameter name used in errors.

    Returns:
        Validated tuple.

    Raises:
        ValueError: If fewer than two values or zero variation is supplied.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 8]. No equation is evaluated.

    """
    sample = coerce_finite_sample(values, name=name)
    if len(sample) < 2:
        raise ValueError(f"{name} must contain at least two observations")
    if statistics.variance(sample) == 0.0:
        raise ValueError(f"{name} has zero variance; the t statistic is undefined")
    return sample


def one_sample_t_test(
    values: Iterable[object],
    *,
    null_mean: float = 0.0,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> TTestResult:
    """Test a normal mean against a hypothesized value with Student's t.

    Assumes independent observations from a normal law with unknown
    variance; the test is robust to moderate non-normality for larger ``n``.

    Args:
        values: At least two finite observations with positive variance.
        null_mean: Hypothesized mean ``mu_0``.
        alternative: ``"two-sided"``, ``"greater"`` (mean above ``mu_0``),
            or ``"less"``.
        confidence: Level of the reported two-sided interval for the mean.

    Returns:
        ``t = (x_bar - mu_0) / (s / sqrt(n))`` with ``n - 1`` degrees of
        freedom, its p-value, interval, and Cohen's ``d``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument has an invalid domain.

    Equation:
        [EQ-INF-T-001], [EQ-INF-CI-TMEAN-001].

    References:
        [REF-STUDENT-1908], [REF-CASELLA-BERGER-2002, Chapter 8].

    """
    sample = _varying_sample(values, "values")
    reference = require_finite_real(null_mean, name="null_mean")
    side = check_alternative(alternative)
    level = check_level(confidence, name="confidence")
    n = len(sample)
    return _t_result(
        estimate=statistics.fmean(sample),
        standard_error=statistics.stdev(sample) / math.sqrt(n),
        df=float(n - 1),
        null_value=reference,
        alternative=side,
        confidence=level,
        method="one-sample Student t test",
        effect_size=cohens_d(sample, null_value=reference),
        sample_sizes=(n,),
    )


def paired_t_test(
    first: Iterable[object],
    second: Iterable[object],
    *,
    null_difference: float = 0.0,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> TTestResult:
    """Test the mean of paired differences ``first - second`` with Student's t.

    Assumes the differences are independent normal observations.

    Args:
        first: First measurement of each pair.
        second: Second measurement of each pair, same length.
        null_difference: Hypothesized mean difference.
        alternative: Direction of the alternative for ``mean(first - second)``.
        confidence: Level of the reported interval for the mean difference.

    Returns:
        One-sample t test on the differences with ``n - 1`` degrees of
        freedom and the paired effect size ``d_z``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If lengths differ or the differences are constant.

    Equation:
        [EQ-INF-T-001].

    References:
        [REF-STUDENT-1908], [REF-CASELLA-BERGER-2002, Chapter 8].

    """
    a = coerce_finite_sample(first, name="first")
    b = coerce_finite_sample(second, name="second")
    if len(a) != len(b):
        raise ValueError("paired samples must have equal lengths")
    differences = _varying_sample(
        [x - y for x, y in zip(a, b, strict=True)], "differences"
    )
    reference = require_finite_real(null_difference, name="null_difference")
    side = check_alternative(alternative)
    level = check_level(confidence, name="confidence")
    n = len(differences)
    return _t_result(
        estimate=statistics.fmean(differences),
        standard_error=statistics.stdev(differences) / math.sqrt(n),
        df=float(n - 1),
        null_value=reference,
        alternative=side,
        confidence=level,
        method="paired Student t test",
        effect_size=cohens_d(differences, null_value=reference),
        sample_sizes=(n,),
    )


def _two_sample_parts(
    first: Iterable[object], second: Iterable[object], equal_variances: bool
) -> tuple[float, float, float, str, float, tuple[int, int]]:
    """Return estimate, standard error, df, label, effect, and sizes.

    Args:
        first: First sample.
        second: Second sample.
        equal_variances: Use the pooled Student statistic instead of Welch's.

    Returns:
        Components of a two-sample t statistic.

    Raises:
        ValueError: If a sample has fewer than two values or both are constant.

    References:
        [EQ-INF-T-002], [EQ-INF-T-003], [REF-WELCH-1947].

    """
    a = coerce_finite_sample(first, name="first")
    b = coerce_finite_sample(second, name="second")
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("each sample must contain at least two observations")
    v1, v2 = statistics.variance(a), statistics.variance(b)
    if v1 == 0.0 and v2 == 0.0:
        raise ValueError("both samples are constant; the t statistic is undefined")
    estimate = statistics.fmean(a) - statistics.fmean(b)
    if equal_variances:
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        standard_error = math.sqrt(pooled * (1.0 / n1 + 1.0 / n2))
        df = float(n1 + n2 - 2)
        label = "two-sample Student t test (pooled variance)"
    else:
        standard_error = math.sqrt(v1 / n1 + v2 / n2)
        df = welch_satterthwaite_df(v1, n1, v2, n2)
        label = "Welch two-sample t test"
    return estimate, standard_error, df, label, cohens_d(a, b), (n1, n2)


def two_sample_t_test(
    first: Iterable[object],
    second: Iterable[object],
    *,
    equal_variances: bool = False,
    null_difference: float = 0.0,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> TTestResult:
    """Compare two independent means with Welch's or Student's pooled t test.

    Welch's test (default) does not assume equal variances and uses the
    Welch-Satterthwaite degrees of freedom; Student's pooled test assumes a
    common variance. Both assume independent normal samples.

    Args:
        first: First sample (at least two values).
        second: Second sample (at least two values).
        equal_variances: Use the pooled-variance Student test.
        null_difference: Hypothesized ``mu_1 - mu_2``.
        alternative: Direction of the alternative for ``mu_1 - mu_2``.
        confidence: Level of the reported interval for ``mu_1 - mu_2``.

    Returns:
        t statistic, degrees of freedom, p-value, interval, and pooled
        Cohen's ``d``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument has an invalid domain.

    Equation:
        [EQ-INF-T-002], [EQ-INF-T-003].

    References:
        [REF-WELCH-1947], [REF-SATTERTHWAITE-1946],
        [REF-CASELLA-BERGER-2002, Chapter 8].

    """
    estimate, standard_error, df, label, effect, sizes = _two_sample_parts(
        first, second, equal_variances
    )
    return _t_result(
        estimate=estimate,
        standard_error=standard_error,
        df=df,
        null_value=require_finite_real(null_difference, name="null_difference"),
        alternative=check_alternative(alternative),
        confidence=check_level(confidence, name="confidence"),
        method=label,
        effect_size=effect,
        sample_sizes=sizes,
    )


@dataclass(frozen=True, slots=True)
class EquivalenceTestResult:
    """Two one-sided tests (TOST) for equivalence of a mean or mean difference.

    Args:
        lower_bound: Lower equivalence margin.
        upper_bound: Upper equivalence margin.
        estimate: Estimated mean or mean difference.
        standard_error: Its standard error.
        df: Degrees of freedom.
        lower_statistic: ``(estimate - lower_bound) / se``.
        upper_statistic: ``(estimate - upper_bound) / se``.
        lower_p_value: Upper-tail p-value of the lower one-sided test.
        upper_p_value: Lower-tail p-value of the upper one-sided test.
        p_value: ``max(lower_p_value, upper_p_value)``, the probability bound
            under the non-equivalence null; not the probability that the
            means are equivalent or not.
        alpha: Significance level.
        equivalent: Whether non-equivalence is rejected at ``alpha``.
        confidence_interval: The ``1 - 2 alpha`` interval; equivalence is
            declared exactly when it lies inside the margins.
        method: Test label.

    References:
        [REF-SCHUIRMANN-1987].

    """

    lower_bound: float
    upper_bound: float
    estimate: float
    standard_error: float
    df: float
    lower_statistic: float
    upper_statistic: float
    lower_p_value: float
    upper_p_value: float
    p_value: float
    alpha: float
    equivalent: bool
    confidence_interval: ConfidenceInterval
    method: str


def tost_equivalence_test(
    first: Iterable[object],
    second: Iterable[object] | None = None,
    *,
    lower_bound: float,
    upper_bound: float,
    alpha: float = 0.05,
    paired: bool = False,
    equal_variances: bool = False,
) -> EquivalenceTestResult:
    """Test equivalence of a mean (difference) within margins by TOST.

    The null hypothesis is non-equivalence, ``theta <= lower_bound`` or
    ``theta >= upper_bound``; it is rejected at level ``alpha`` when both
    one-sided t tests reject. Failing to reject means equivalence was not
    demonstrated, not that the effect lies outside the margins.

    Args:
        first: First sample, or the only sample for a one-sample mean.
        second: Optional second sample.
        lower_bound: Lower equivalence margin.
        upper_bound: Upper margin, strictly above ``lower_bound``.
        alpha: Significance level of each one-sided test.
        paired: Analyse ``first - second`` as paired differences.
        equal_variances: Use the pooled two-sample statistic.

    Returns:
        Both one-sided statistics and p-values, the decision, and the
        ``1 - 2 alpha`` confidence interval.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If margins or samples are invalid.

    Equation:
        [EQ-INF-TOST-001], [EQ-INF-T-001], [EQ-INF-T-003].

    References:
        [REF-SCHUIRMANN-1987], [REF-LEHMANN-ROMANO-2005, Chapter 3].

    """
    low = require_finite_real(lower_bound, name="lower_bound")
    high = require_finite_real(upper_bound, name="upper_bound")
    if not low < high:
        raise ValueError("lower_bound must be below upper_bound")
    level = check_level(alpha, name="alpha")
    if level >= 0.5:
        raise ValueError("alpha must be below one half")
    if second is None:
        if paired:
            raise ValueError("paired equivalence requires a second sample")
        base = one_sample_t_test(first)
        method = "one-sample TOST"
    elif paired:
        base = paired_t_test(first, second)
        method = "paired TOST"
    else:
        base = two_sample_t_test(first, second, equal_variances=equal_variances)
        method = f"TOST ({base.method})"
    se = base.standard_error
    lower_statistic = (base.estimate - low) / se
    upper_statistic = (base.estimate - high) / se
    lower_p = clamp_probability(t_p_value(lower_statistic, base.df, "greater"))
    upper_p = clamp_probability(t_p_value(upper_statistic, base.df, "less"))
    p_value = max(lower_p, upper_p)
    critical = t_critical(1.0 - level, base.df)
    interval = ConfidenceInterval(
        base.estimate - critical * se,
        base.estimate + critical * se,
        1.0 - 2.0 * level,
        "Student-t interval at 1 - 2 alpha",
    )
    return EquivalenceTestResult(
        low,
        high,
        base.estimate,
        se,
        base.df,
        lower_statistic,
        upper_statistic,
        lower_p,
        upper_p,
        p_value,
        level,
        p_value < level,
        interval,
        method,
    )


class SmallExpectedCountWarning(UserWarning):
    """Warn that chi-square approximations rest on small expected counts.

    Cochran's rule of thumb asks that expected counts be at least five (in
    most cells); below that the chi-square reference distribution can be a
    poor approximation and an exact test should be preferred.

    References:
        [REF-COCHRAN-1954], [REF-AGRESTI-2013, Chapter 3].

    """


@dataclass(frozen=True, slots=True)
class ChiSquareTestResult:
    """Pearson chi-square goodness-of-fit or independence test result.

    Args:
        statistic: Pearson ``X^2`` (Yates-corrected when requested).
        p_value: Upper-tail chi-square probability under the null of a
            statistic at least as large as observed; not the probability
            that the null hypothesis is true.
        df: Degrees of freedom.
        method: Test label.
        observed: Observed counts, flattened row-major for tables.
        expected: Expected counts under the null, flattened likewise.
        min_expected: Smallest expected count.
        small_expected_counts: Whether any expected count is below five.
        effect_size: Cohen's ``w`` (goodness of fit) or Cramer's ``V``
            (independence), both from the uncorrected statistic.

    References:
        [REF-AGRESTI-2013, Chapters 1 and 3], [REF-COCHRAN-1954].

    """

    statistic: float
    p_value: float
    df: int
    method: str
    observed: tuple[float, ...]
    expected: tuple[float, ...]
    min_expected: float
    small_expected_counts: bool
    effect_size: float

    def __post_init__(self) -> None:
        """Validate the p-value.

        Raises:
            ValueError: If the p-value lies outside ``[0, 1]``.

        References:
            [REF-AGRESTI-2013, Chapter 1]. No equation is evaluated.

        """
        object.__setattr__(self, "p_value", _check_p_value(self.p_value))


def _count(value: object, name: str) -> float:
    """Validate one non-negative finite count (integers or real weights).

    Args:
        value: Candidate count.
        name: Name used in error messages.

    Returns:
        Count as a float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is negative or not finite.

    References:
        [REF-AGRESTI-2013, Chapter 1]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real count")
    count = float(value)
    if not math.isfinite(count) or count < 0.0:
        raise ValueError(f"{name} must be a non-negative finite count")
    return count


def _warn_small(expected: Sequence[float]) -> bool:
    """Emit :class:`SmallExpectedCountWarning` when any expected count is small.

    Args:
        expected: Expected counts.

    Returns:
        Whether a warning was emitted.

    References:
        [REF-COCHRAN-1954]. No equation is evaluated.

    """
    smallest = min(expected)
    if smallest < 5.0:
        warnings.warn(
            f"smallest expected count is {smallest:.3g} (< 5); the chi-square "
            "approximation may be inaccurate, consider an exact test",
            SmallExpectedCountWarning,
            stacklevel=3,
        )
        return True
    return False


def chi_square_goodness_of_fit(
    observed: Iterable[object],
    probabilities: Iterable[object] | None = None,
    *,
    estimated_parameters: int = 0,
) -> ChiSquareTestResult:
    """Run Pearson's chi-square goodness-of-fit test for category counts.

    Args:
        observed: Counts in ``k >= 2`` categories with positive total.
        probabilities: Null category probabilities (positive, summing to one
            within ``1e-9``); uniform when omitted.
        estimated_parameters: Number of parameters estimated from the same
            counts, subtracted from the degrees of freedom.

    Returns:
        ``X^2 = sum (O - E)^2 / E`` with ``k - 1 - estimated_parameters``
        degrees of freedom and Cohen's ``w = sqrt(X^2 / n)``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If counts, probabilities, or degrees of freedom are invalid.

    Warns:
        SmallExpectedCountWarning: If an expected count is below five.

    Equation:
        [EQ-INF-CHI2-GOF-001], [EQ-INF-DIST-CHI2-001].

    References:
        [REF-AGRESTI-2013, Chapter 1], [REF-COCHRAN-1954].

    """
    counts = tuple(
        _count(value, f"observed[{index}]") for index, value in enumerate(observed)
    )
    k = len(counts)
    if k < 2:
        raise ValueError("at least two categories are required")
    total = math.fsum(counts)
    if total <= 0.0:
        raise ValueError("observed counts must have a positive total")
    if probabilities is None:
        probs = tuple(1.0 / k for _ in range(k))
    else:
        probs = tuple(
            require_probability(value, name=f"probabilities[{index}]")
            for index, value in enumerate(probabilities)
        )
        if len(probs) != k:
            raise ValueError("probabilities must match the number of categories")
        if any(p == 0.0 for p in probs):
            raise ValueError("null probabilities must be positive")
        if abs(math.fsum(probs) - 1.0) > 1e-9:
            raise ValueError("probabilities must sum to one")
    fitted = require_non_negative_integer(
        estimated_parameters, name="estimated_parameters"
    )
    df = k - 1 - fitted
    if df < 1:
        raise ValueError("degrees of freedom must be positive")
    expected = tuple(total * p for p in probs)
    statistic = math.fsum(
        (o - e) ** 2 / e for o, e in zip(counts, expected, strict=True)
    )
    small = _warn_small(expected)
    return ChiSquareTestResult(
        statistic,
        chi_square_sf(statistic, df),
        df,
        "Pearson chi-square goodness-of-fit test",
        counts,
        expected,
        min(expected),
        small,
        math.sqrt(statistic / total),
    )


def _table(table: Iterable[Iterable[object]]) -> tuple[tuple[float, ...], ...]:
    """Validate a rectangular contingency table of non-negative counts.

    Args:
        table: Row-wise counts.

    Returns:
        Validated rows.

    Raises:
        ValueError: If the table is not rectangular, smaller than 2 x 2, or has
            an empty row or column.

    References:
        [REF-AGRESTI-2013, Chapter 2]. No equation is evaluated.

    """
    rows = tuple(
        tuple(_count(v, f"table[{i}][{j}]") for j, v in enumerate(row))
        for i, row in enumerate(table)
    )
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("table must be rectangular with at least two rows")
    if len(rows[0]) < 2:
        raise ValueError("table must have at least two columns")
    if any(math.fsum(row) == 0.0 for row in rows):
        raise ValueError("table rows must have positive totals")
    if any(math.fsum(column) == 0.0 for column in zip(*rows, strict=True)):
        raise ValueError("table columns must have positive totals")
    return rows


def chi_square_independence_test(
    table: Iterable[Iterable[object]],
    *,
    yates_correction: bool = False,
) -> ChiSquareTestResult:
    """Test independence of rows and columns in an ``r x c`` contingency table.

    Args:
        table: Row-wise counts with at least two rows and two columns and no
            empty row or column.
        yates_correction: Apply Yates' continuity correction (2 x 2 only).

    Returns:
        Pearson ``X^2 = sum (O - E)^2 / E`` with ``E = R_i C_j / n``,
        ``(r - 1)(c - 1)`` degrees of freedom, and Cramer's ``V``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If the table is invalid or Yates is requested for a
            larger table.

    Warns:
        SmallExpectedCountWarning: If an expected count is below five.

    Equation:
        [EQ-INF-CHI2-IND-001], [EQ-INF-DIST-CHI2-001].

    References:
        [REF-AGRESTI-2013, Chapter 3], [REF-COCHRAN-1954].

    """
    rows = _table(table)
    r, c = len(rows), len(rows[0])
    if yates_correction and (r, c) != (2, 2):
        raise ValueError("Yates correction applies only to 2 x 2 tables")
    row_totals = [math.fsum(row) for row in rows]
    column_totals = [math.fsum(column) for column in zip(*rows, strict=True)]
    total = math.fsum(row_totals)
    observed: list[float] = []
    expected: list[float] = []
    for i in range(r):
        for j in range(c):
            observed.append(rows[i][j])
            expected.append(row_totals[i] * column_totals[j] / total)
    pearson = math.fsum(
        (o - e) ** 2 / e for o, e in zip(observed, expected, strict=True)
    )
    if yates_correction:
        statistic = math.fsum(
            max(0.0, abs(o - e) - 0.5) ** 2 / e
            for o, e in zip(observed, expected, strict=True)
        )
        method = "Pearson chi-square independence test with Yates correction"
    else:
        statistic = pearson
        method = "Pearson chi-square independence test"
    df = (r - 1) * (c - 1)
    small = _warn_small(expected)
    return ChiSquareTestResult(
        statistic,
        chi_square_sf(statistic, df),
        df,
        method,
        tuple(observed),
        tuple(expected),
        min(expected),
        small,
        math.sqrt(pearson / (total * (min(r, c) - 1))),
    )


@dataclass(frozen=True, slots=True)
class FisherExactResult:
    """Fisher's exact test for a 2 x 2 table.

    Args:
        odds_ratio: Sample odds ratio ``ad / (bc)``; ``inf`` when ``bc = 0 <
            ad`` and ``nan`` when both products are zero.
        p_value: Conditional probability, given the margins and the null of
            independence, of a table at least as extreme as observed; not the
            probability that the null hypothesis is true.
        alternative: Direction relative to an odds ratio of one.
        method: Test label.

    References:
        [REF-AGRESTI-2013, Chapter 3].

    """

    odds_ratio: float
    p_value: float
    alternative: str
    method: str

    def __post_init__(self) -> None:
        """Validate the p-value.

        Raises:
            ValueError: If the p-value lies outside ``[0, 1]``.

        References:
            [REF-AGRESTI-2013, Chapter 3]. No equation is evaluated.

        """
        object.__setattr__(self, "p_value", _check_p_value(self.p_value))


def fisher_exact_test(
    table: Iterable[Iterable[object]],
    *,
    alternative: Alternative = "two-sided",
) -> FisherExactResult:
    """Run Fisher's exact test on a 2 x 2 table of integer counts.

    Conditioning on both margins, the top-left count is hypergeometric under
    independence. The two-sided p-value sums the probabilities of all tables
    no more probable than the observed one (relative tolerance ``1e-7``).

    Args:
        table: ``[[a, b], [c, d]]`` with non-negative integers.
        alternative: ``"greater"`` tests odds ratio above one.

    Returns:
        Sample odds ratio and exact p-value.

    Raises:
        TypeError: If counts are not integers.
        ValueError: If the table is not 2 x 2 or has an empty margin.

    Equation:
        [EQ-INF-FISHER-001].

    References:
        [REF-AGRESTI-2013, Chapter 3].

    """
    raw = [list(row) for row in table]
    if len(raw) != 2 or any(len(row) != 2 for row in raw):
        raise ValueError("Fisher's exact test requires a 2 x 2 table")
    a, b = (require_non_negative_integer(v, name="table entry") for v in raw[0])
    c, d = (require_non_negative_integer(v, name="table entry") for v in raw[1])
    side = check_alternative(alternative)
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    if min(row1, row2, col1, n - col1) == 0:
        raise ValueError("table margins must be positive")
    low, high = max(0, col1 - row2), min(row1, col1)
    denominator = math.comb(n, col1)
    probabilities = {
        k: math.comb(row1, k) * math.comb(row2, col1 - k) / denominator
        for k in range(low, high + 1)
    }
    if side == "greater":
        p_value = math.fsum(p for k, p in probabilities.items() if k >= a)
    elif side == "less":
        p_value = math.fsum(p for k, p in probabilities.items() if k <= a)
    else:
        threshold = probabilities[a] * _RELATIVE_TIE
        p_value = math.fsum(p for p in probabilities.values() if p <= threshold)
    if b * c == 0:
        odds_ratio = math.nan if a * d == 0 else math.inf
    else:
        odds_ratio = a * d / (b * c)
    return FisherExactResult(
        odds_ratio, clamp_probability(p_value), side, "Fisher exact test (2 x 2)"
    )


@dataclass(frozen=True, slots=True)
class BinomialTestResult:
    """Exact binomial test of a success probability.

    Args:
        successes: Observed successes.
        trials: Number of trials.
        null_probability: Hypothesized success probability.
        p_value: Exact probability under the null of a count at least as
            extreme as observed; not the probability that the null is true.
        alternative: Direction of the alternative hypothesis.
        estimate: Observed proportion.
        confidence_interval: Clopper-Pearson interval at the requested level.
        method: Test label.

    References:
        [REF-AGRESTI-2013, Chapter 1], [REF-CLOPPER-PEARSON-1934].

    """

    successes: int
    trials: int
    null_probability: float
    p_value: float
    alternative: str
    estimate: float
    confidence_interval: ConfidenceInterval
    method: str

    def __post_init__(self) -> None:
        """Validate the p-value.

        Raises:
            ValueError: If the p-value lies outside ``[0, 1]``.

        References:
            [REF-AGRESTI-2013, Chapter 1]. No equation is evaluated.

        """
        object.__setattr__(self, "p_value", _check_p_value(self.p_value))


def _binomial_pmf(k: int, n: int, p: float) -> float:
    """Return the binomial mass ``C(n,k) p^k (1-p)^(n-k)`` in log space.

    Args:
        k: Count between zero and ``n``.
        n: Trials.
        p: Probability strictly inside ``(0, 1)``.

    Returns:
        Probability mass.

    References:
        [EQ-INF-BINOM-001], [REF-AGRESTI-2013, Chapter 1].

    """
    log_mass = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    return math.exp(log_mass)


def binomial_test(
    successes: int,
    trials: int,
    *,
    null_probability: float = 0.5,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> BinomialTestResult:
    """Run the exact binomial test by direct summation of the null mass.

    The two-sided p-value sums ``P(X = k)`` over all ``k`` whose null mass is
    at most the observed mass (relative tolerance ``1e-7``), the
    minimum-likelihood convention of Clopper-Pearson-type exact inference.

    Args:
        successes: Observed successes from zero through ``trials``.
        trials: Positive number of trials.
        null_probability: Hypothesized probability strictly inside ``(0, 1)``.
        alternative: ``"greater"`` tests a probability above the null.
        confidence: Level of the reported Clopper-Pearson interval.

    Returns:
        Exact p-value, proportion, and Clopper-Pearson interval.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument has an invalid domain.

    Equation:
        [EQ-INF-BINOM-001], [EQ-INF-CI-CP-001].

    References:
        [REF-AGRESTI-2013, Chapter 1], [REF-CLOPPER-PEARSON-1934].

    """
    x = require_non_negative_integer(successes, name="successes")
    n = require_non_negative_integer(trials, name="trials")
    if n == 0:
        raise ValueError("trials must be positive")
    if x > n:
        raise ValueError("successes must not exceed trials")
    p0 = check_level(null_probability, name="null_probability")
    side = check_alternative(alternative)
    level = check_level(confidence, name="confidence")
    masses = [_binomial_pmf(k, n, p0) for k in range(n + 1)]
    if side == "greater":
        p_value = math.fsum(masses[x:])
    elif side == "less":
        p_value = math.fsum(masses[: x + 1])
    else:
        threshold = masses[x] * _RELATIVE_TIE
        p_value = math.fsum(m for m in masses if m <= threshold)
    return BinomialTestResult(
        x,
        n,
        p0,
        clamp_probability(p_value),
        side,
        x / n,
        clopper_pearson_interval(x, n, confidence=level),
        "exact binomial test",
    )


@dataclass(frozen=True, slots=True)
class AnovaResult:
    """One-way analysis-of-variance F test.

    Args:
        statistic: ``F = MS_between / MS_within``.
        p_value: Upper-tail F probability under the null of equal means of a
            statistic at least as large as observed; not the probability that
            the means are equal.
        df_between: ``k - 1``.
        df_within: ``N - k``.
        ss_between: Between-group sum of squares.
        ss_within: Within-group sum of squares.
        eta_squared: ``SS_between / SS_total``.
        group_sizes: Sizes of the groups.
        method: Test label.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 11].

    """

    statistic: float
    p_value: float
    df_between: int
    df_within: int
    ss_between: float
    ss_within: float
    eta_squared: float
    group_sizes: tuple[int, ...]
    method: str

    def __post_init__(self) -> None:
        """Validate the p-value.

        Raises:
            ValueError: If the p-value lies outside ``[0, 1]``.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 11]. No equation is evaluated.

        """
        object.__setattr__(self, "p_value", _check_p_value(self.p_value))


def one_way_anova(*groups: Iterable[object]) -> AnovaResult:
    """Test equality of several normal means with a one-way ANOVA F test.

    Assumes independent normal groups with a common variance.

    Args:
        *groups: At least two non-empty samples with ``N > k`` in total.

    Returns:
        F statistic with ``(k - 1, N - k)`` degrees of freedom, p-value from
        the regularized incomplete beta function, and ``eta^2``.

    Raises:
        TypeError: If observations are not real.
        ValueError: If fewer than two groups, an empty group, too few
            observations, or zero within-group variation is supplied.

    Equation:
        [EQ-INF-ANOVA-001], [EQ-INF-DIST-F-001].

    References:
        [REF-CASELLA-BERGER-2002, Chapter 11].

    """
    samples = tuple(
        coerce_finite_sample(group, name=f"groups[{index}]")
        for index, group in enumerate(groups)
    )
    k = len(samples)
    if k < 2:
        raise ValueError("at least two groups are required")
    total_n = sum(len(sample) for sample in samples)
    if total_n <= k:
        raise ValueError("total observations must exceed the number of groups")
    grand_mean = math.fsum(math.fsum(sample) for sample in samples) / total_n
    group_means = [statistics.fmean(sample) for sample in samples]
    ss_between = math.fsum(
        len(sample) * (mean - grand_mean) ** 2
        for sample, mean in zip(samples, group_means, strict=True)
    )
    ss_within = math.fsum(
        (value - mean) ** 2
        for sample, mean in zip(samples, group_means, strict=True)
        for value in sample
    )
    if ss_within == 0.0:
        raise ValueError("within-group variation is zero; F is undefined")
    df_between, df_within = k - 1, total_n - k
    statistic = (ss_between / df_between) / (ss_within / df_within)
    return AnovaResult(
        statistic,
        f_sf(statistic, df_between, df_within),
        df_between,
        df_within,
        ss_between,
        ss_within,
        ss_between / (ss_between + ss_within),
        tuple(len(sample) for sample in samples),
        "one-way ANOVA F test",
    )
