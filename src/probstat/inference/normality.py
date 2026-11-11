"""EDF tests of normality with estimated mean and variance (fitted-null case).

When the mean and standard deviation are estimated from the tested sample,
the ordinary Kolmogorov-Smirnov, Cramer-von Mises, and Anderson-Darling null
laws no longer apply: the fitted CDF is closer to the data than the true CDF,
so the classical p-values are far too large. For the normal family the null
laws of the fitted statistics do not depend on the unknown parameters
(location-scale invariance), so they were tabulated once and summarized by
closed-form p-value approximations:

* Kolmogorov-Smirnov (Lilliefors test): the Dallal-Wilkinson analytic
  approximation for ``p <= 0.1`` and, above that, a polynomial in the
  Stephens-modified statistic ``D (sqrt(n) - 0.01 + 0.85 / sqrt(n))``.
* Cramer-von Mises: ``W^2 (1 + 0.5 / n)`` with the piecewise exponential
  p-value formulas of Stephens (1986, Table 4.9).
* Anderson-Darling: ``A^2 (1 + 0.75 / n + 2.25 / n^2)`` with the same table.

These are family-specific calibrations: they are valid only for the normal
family with both parameters estimated (the sample mean and the ``n - 1``
standard deviation). Other families use the parametric bootstrap of
:func:`probstat.inference.bootstrap_goodness_of_fit`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.foundations import coerce_finite_sample
from probstat.foundations.protocols import CumulativeDistribution
from probstat.foundations.validation import require_probability

from .sampling_distributions import standard_normal_cdf

MINIMUM_NORMALITY_SAMPLE = 8
"""Smallest sample size for which the tabulated approximations are used."""


@dataclass(frozen=True, slots=True)
class EDFNormalityTest:
    """One EDF statistic against the fitted normal law and its p-value.

    Args:
        name: ``"lilliefors"``, ``"cramer-von-mises"``, or
            ``"anderson-darling"``.
        statistic: Unmodified statistic against ``N(x_bar, s^2)``.
        modified_statistic: Sample-size-modified statistic that enters the
            tabulated p-value formula.
        p_value: Approximate upper-tail p-value in ``[0, 1]``.
        method: Source of the calibration.

    References:
        [EQ-INF-GOF-NORM-001], [EQ-INF-GOF-NORM-002], [EQ-INF-GOF-NORM-003],
        [REF-STEPHENS-1986].

    """

    name: str
    statistic: float
    modified_statistic: float
    p_value: float
    method: str

    def rejected(self, alpha: float = 0.05) -> bool:
        """Return whether the p-value falls below ``alpha``.

        Args:
            alpha: Significance level strictly inside ``(0, 1)``.

        Returns:
            ``True`` when ``p_value < alpha``.

        Raises:
            ValueError: If ``alpha`` is not strictly inside ``(0, 1)``.

        References:
            [REF-STEPHENS-1986].

        """
        level = require_probability(alpha, name="alpha")
        if not 0.0 < level < 1.0:
            raise ValueError("alpha must lie strictly inside (0, 1)")
        return self.p_value < level


@dataclass(frozen=True, slots=True)
class NormalityTests:
    """Lilliefors, Cramer-von Mises, and Anderson-Darling normality tests.

    Args:
        mean: Sample mean used for the fitted null.
        standard_deviation: Sample standard deviation (divisor ``n - 1``).
        sample_size: Number of observations.
        lilliefors: Kolmogorov-Smirnov distance with estimated parameters.
        cramer_von_mises: Cramer-von Mises statistic with estimated parameters.
        anderson_darling: Anderson-Darling statistic with estimated parameters.

    References:
        [REF-LILLIEFORS-1967], [REF-STEPHENS-1974], [REF-STEPHENS-1986].

    """

    mean: float
    standard_deviation: float
    sample_size: int
    lilliefors: EDFNormalityTest
    cramer_von_mises: EDFNormalityTest
    anderson_darling: EDFNormalityTest

    @property
    def tests(self) -> tuple[EDFNormalityTest, ...]:
        """Return the three tests in a fixed order.

        Returns:
            ``(lilliefors, cramer_von_mises, anderson_darling)``.

        References:
            [REF-STEPHENS-1986].

        """
        return (self.lilliefors, self.cramer_von_mises, self.anderson_darling)


def cramer_von_mises_statistic(
    values: Iterable[object], distribution: CumulativeDistribution
) -> float:
    """Return the Cramer-von Mises statistic of a sample against a CDF.

    ``W^2 = 1 / (12 n) + sum_i (F(x_(i)) - (2 i - 1) / (2 n))^2``.

    Args:
        values: Non-empty iterable of finite observations.
        distribution: Continuous null law exposing ``cdf``.

    Returns:
        The non-negative statistic ``W^2``. It carries no p-value: the null
        law depends on whether parameters were estimated (see
        :func:`normality_tests` and
        :func:`probstat.inference.bootstrap_goodness_of_fit`).

    Raises:
        TypeError: If ``distribution`` does not expose ``cdf``.
        ValueError: If the sample is empty or non-finite, or a CDF value lies
            outside ``[0, 1]``.

    Equation:
        [EQ-INF-GOF-CVM-001].

    References:
        [REF-STEPHENS-1974], [REF-STEPHENS-1986].

    """
    sample = tuple(sorted(coerce_finite_sample(values)))
    if not isinstance(distribution, CumulativeDistribution):
        raise TypeError("distribution must expose cdf(x)")
    probabilities = tuple(
        require_probability(distribution.cdf(value), name="distribution.cdf(x)")
        for value in sample
    )
    return _cramer_von_mises(probabilities)


def normality_tests(values: Iterable[object]) -> NormalityTests:
    """Test normality with estimated mean and variance by three EDF statistics.

    The sample is standardized by its mean and ``n - 1`` standard deviation,
    the statistics are computed against ``N(0, 1)``, and each p-value comes
    from the tabulated null law of the fitted statistic (not from the
    fully specified KS/AD laws, which would be badly conservative).

    Args:
        values: At least eight finite observations, not all equal.

    Returns:
        The fitted parameters and the three tests.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If fewer than eight observations are given or all are
            equal.

    Equation:
        [EQ-INF-GOF-KS-001], [EQ-INF-GOF-CVM-001], [EQ-INF-GOF-AD-001],
        [EQ-INF-GOF-NORM-001], [EQ-INF-GOF-NORM-002], [EQ-INF-GOF-NORM-003].

    References:
        [REF-LILLIEFORS-1967], [REF-DALLAL-WILKINSON-1986],
        [REF-STEPHENS-1974], [REF-STEPHENS-1986, Table 4.9],
        [REF-GROSS-LIGGES-2015].

    """
    sample = tuple(sorted(coerce_finite_sample(values)))
    n = len(sample)
    if n < MINIMUM_NORMALITY_SAMPLE:
        raise ValueError(
            f"normality_tests needs at least {MINIMUM_NORMALITY_SAMPLE} observations"
        )
    mean = math.fsum(sample) / n
    variance = math.fsum((value - mean) ** 2 for value in sample) / (n - 1)
    if variance <= 0.0:
        raise ValueError("normality_tests needs observations that are not all equal")
    sd = math.sqrt(variance)
    scores = tuple((value - mean) / sd for value in sample)
    probabilities = tuple(standard_normal_cdf(z) for z in scores)
    upper_tails = tuple(standard_normal_cdf(-z) for z in scores)
    ks = max(
        max((i + 1) / n - p for i, p in enumerate(probabilities)),
        max(p - i / n for i, p in enumerate(probabilities)),
    )
    cvm = _cramer_von_mises(probabilities)
    ad = _anderson_darling(probabilities, upper_tails)
    cvm_modified = cvm * (1.0 + 0.5 / n)
    ad_modified = ad * (1.0 + 0.75 / n + 2.25 / n**2)
    ks_modified = ks * (math.sqrt(n) - 0.01 + 0.85 / math.sqrt(n))
    return NormalityTests(
        mean=mean,
        standard_deviation=sd,
        sample_size=n,
        lilliefors=EDFNormalityTest(
            "lilliefors",
            ks,
            ks_modified,
            lilliefors_p_value(ks, n),
            "Dallal-Wilkinson approximation; Stephens-modified D above p = 0.1",
        ),
        cramer_von_mises=EDFNormalityTest(
            "cramer-von-mises",
            cvm,
            cvm_modified,
            _cvm_normal_p_value(cvm_modified),
            "Stephens (1986) Table 4.9, W^2 (1 + 0.5/n)",
        ),
        anderson_darling=EDFNormalityTest(
            "anderson-darling",
            ad,
            ad_modified,
            _ad_normal_p_value(ad_modified),
            "Stephens (1986) Table 4.9, A^2 (1 + 0.75/n + 2.25/n^2)",
        ),
    )


def lilliefors_p_value(statistic: float, sample_size: int) -> float:
    """Return the Lilliefors p-value of a fitted-normal KS distance.

    For ``n > 100`` the statistic is rescaled by ``(n / 100)^0.49`` and the
    formula is evaluated at ``n = 100`` (Dallal and Wilkinson). When the
    analytic approximation exceeds ``0.1`` the p-value is replaced by a
    quartic polynomial in the Stephens-modified statistic
    ``Z = D (sqrt(n) - 0.01 + 0.85 / sqrt(n))``.

    Args:
        statistic: Kolmogorov-Smirnov distance ``D`` in ``[0, 1]`` computed
            against the normal law with estimated mean and variance.
        sample_size: Sample size, at least ``5``.

    Returns:
        Approximate p-value clipped to ``[0, 1]``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``statistic`` is outside ``[0, 1]`` or
            ``sample_size < 5``.

    Equation:
        [EQ-INF-GOF-NORM-001].

    References:
        [REF-DALLAL-WILKINSON-1986], [REF-STEPHENS-1974],
        [REF-GROSS-LIGGES-2015].

    """
    d = require_probability(statistic, name="statistic")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if sample_size < 5:
        raise ValueError("sample_size must be at least 5")
    n = sample_size
    if n <= 100:
        kd, nd = d, float(n)
    else:
        kd, nd = d * (n / 100.0) ** 0.49, 100.0
    shifted = nd + 2.78019
    p = math.exp(
        -7.01256 * kd**2 * shifted
        + 2.99587 * kd * math.sqrt(shifted)
        - 0.122119
        + 0.974598 / math.sqrt(nd)
        + 1.67997 / nd
    )
    if p > 0.1:
        z = d * (math.sqrt(n) - 0.01 + 0.85 / math.sqrt(n))
        if z <= 0.302:
            p = 1.0
        elif z <= 0.5:
            p = _quartic(z, (2.76773, -19.828315, 80.709644, -138.55152, 81.218052))
        elif z <= 0.9:
            p = _quartic(z, (-4.901232, 40.662806, -97.490286, 94.029866, -32.355711))
        elif z <= 1.31:
            p = _quartic(z, (6.198765, -19.558097, 23.186922, -12.234627, 2.423045))
        else:
            p = 0.0
    return min(1.0, max(0.0, p))


def _quartic(z: float, coefficients: tuple[float, float, float, float, float]) -> float:
    """Evaluate ``c0 + c1 z + c2 z^2 + c3 z^3 + c4 z^4`` by Horner's rule.

    References:
        [EQ-INF-GOF-NORM-001], [REF-GROSS-LIGGES-2015].

    """
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * z + coefficient
    return result


def _cvm_normal_p_value(modified: float) -> float:
    """Return the fitted-normal Cramer-von Mises p-value of ``W^2 (1 + 0.5/n)``.

    References:
        [EQ-INF-GOF-NORM-002], [REF-STEPHENS-1986, Table 4.9].

    """
    w = modified
    if w < 0.0275:
        p = 1.0 - math.exp(-13.953 + 775.5 * w - 12542.61 * w**2)
    elif w < 0.051:
        p = 1.0 - math.exp(-5.903 + 179.546 * w - 1515.29 * w**2)
    elif w < 0.092:
        p = math.exp(0.886 - 31.62 * w + 10.897 * w**2)
    elif w < 1.1:
        p = math.exp(1.111 - 34.242 * w + 12.832 * w**2)
    else:
        p = 7.37e-10
    return min(1.0, max(0.0, p))


def _ad_normal_p_value(modified: float) -> float:
    """Return the fitted-normal Anderson-Darling p-value of the modified ``A^2``.

    References:
        [EQ-INF-GOF-NORM-003], [REF-STEPHENS-1986, Table 4.9].

    """
    a = modified
    if a < 0.2:
        p = 1.0 - math.exp(-13.436 + 101.14 * a - 223.73 * a**2)
    elif a < 0.34:
        p = 1.0 - math.exp(-8.318 + 42.796 * a - 59.938 * a**2)
    elif a < 0.6:
        p = math.exp(0.9177 - 4.279 * a - 1.38 * a**2)
    elif a < 10.0:
        p = math.exp(1.2937 - 5.709 * a + 0.0186 * a**2)
    else:
        p = 3.7e-24
    return min(1.0, max(0.0, p))


def _cramer_von_mises(probabilities: tuple[float, ...]) -> float:
    """Return ``W^2`` from ordered null-CDF values ``F(x_(1)) <= ... <= F(x_(n))``.

    References:
        [EQ-INF-GOF-CVM-001], [REF-STEPHENS-1974].

    """
    n = len(probabilities)
    return 1.0 / (12.0 * n) + math.fsum(
        (p - (2 * i + 1) / (2.0 * n)) ** 2 for i, p in enumerate(probabilities)
    )


def _anderson_darling(
    probabilities: tuple[float, ...], upper_tails: tuple[float, ...]
) -> float:
    """Return ``A^2`` from ordered CDF values and their upper tails.

    ``upper_tails[i] = 1 - probabilities[i]`` evaluated directly, so extreme
    observations keep full relative accuracy. Returns ``inf`` when a
    probability sits on the boundary.

    References:
        [EQ-INF-GOF-AD-001], [REF-STEPHENS-1974].

    """
    n = len(probabilities)
    terms: list[float] = []
    for i, p in enumerate(probabilities):
        upper = upper_tails[n - 1 - i]
        if p <= 0.0 or upper <= 0.0:
            return math.inf
        terms.append((2 * i + 1) * (math.log(p) + math.log(upper)))
    return max(0.0, -n - math.fsum(terms) / n)
