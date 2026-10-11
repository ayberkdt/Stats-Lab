"""Check the values quoted in ``docs/prerequisites/r-bridge.md``.

Each quoted R value is compared with the documented ``probstat`` equivalent
and, where a short standard-library computation exists, with an independent
second computation (closed forms, exact rational arithmetic, enumeration, or
Simpson integration of a density written from scratch).
"""

from __future__ import annotations

import itertools
import math
import statistics
import unittest
from fractions import Fraction

from probstat.distributions import (
    Beta,
    Binomial,
    ChiSquare,
    Exponential,
    FisherF,
    Gamma,
    Geometric,
    Logistic,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
    Weibull,
)
from probstat.inference import (
    adjust_p_values,
    binomial_test,
    chi_square_independence_test,
    fisher_exact_test,
    fit_simple_linear_regression,
    mann_whitney_u_test,
    one_sample_t_test,
    two_sample_t_test,
    wilcoxon_signed_rank_test,
)
from probstat.multivariate import MultivariateNormal
from probstat.optimization import brent_root
from probstat.reliability import bivariate_normal_cdf
from probstat.statistics import (
    empirical_cdf,
    interquartile_range,
    median_absolute_deviation,
    quantiles,
    variance,
)

STANDARD_NORMAL = statistics.NormalDist()


def significant(value: float, digits: int = 7) -> float:
    """Round to a number of significant digits, as R prints by default.

    Args:
        value: Finite non-zero number.
        digits: Significant digits.

    Returns:
        Rounded value.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return float(f"{value:.{digits - 1}e}")


def student_t_density(x: float, nu: float) -> float:
    """Return the Student-t density written directly from its formula.

    Args:
        x: Evaluation point.
        nu: Degrees of freedom.

    Returns:
        Density value.

    References:
        [REF-CASELLA-BERGER-2002, Section 5.3]. Test helper only.

    """
    log_constant = (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
    )
    return math.exp(log_constant - (nu + 1.0) / 2.0 * math.log1p(x * x / nu))


def simpson(function: object, lower: float, upper: float, intervals: int) -> float:
    """Integrate a scalar function by the composite Simpson rule.

    Args:
        function: Callable of one float.
        lower: Lower limit.
        upper: Upper limit.
        intervals: Even number of subintervals.

    Returns:
        Integral approximation.

    References:
        [REF-DAVIS-RABINOWITZ-1984]. Test helper only.

    """
    assert callable(function)
    width = (upper - lower) / intervals
    total = function(lower) + function(upper)
    for index in range(1, intervals):
        total += (4.0 if index % 2 else 2.0) * function(lower + index * width)
    return float(total * width / 3.0)


class DistributionValueTests(unittest.TestCase):
    """d/p/q values quoted in the R bridge."""

    def test_quoted_values_match_probstat(self) -> None:
        """Every quoted distribution value is reproduced to 7 digits."""
        cases = (
            (Normal().cdf(1.96), 0.9750021),
            (Normal().quantile(0.975), 1.959964),
            (Normal().cdf(-3.0), 0.001349898),
            (StudentT(10).quantile(0.975), 2.228139),
            (StudentT(5).cdf(2.0), 0.9490303),
            (ChiSquare(1).quantile(0.95), 3.841459),
            (ChiSquare(4).cdf(3.0), 0.4421746),
            (FisherF(2, 10).cdf(3.0), 0.9046326),
            (FisherF(2, 10).quantile(0.95), 4.102821),
            (Gamma(2.0, rate=0.5).quantile(0.5), 3.356694),
            (Exponential(rate=2.0).cdf(1.0), 0.8646647),
            (Weibull(1.5, 2.0).quantile(0.5), 1.566440),
            (LogNormal(0.0, 1.0).cdf(2.0), 0.7558914),
            (Beta(2.0, 5.0).cdf(0.3), 0.579825),
            (Logistic(0.0, 1.0).cdf(1.0), 0.7310586),
            (Binomial(10, 0.5).pmf(3), 0.1171875),
            (Binomial(10, 0.5).cdf(3), 0.171875),
            (Poisson(3.0).cdf(2), 0.4231901),
            (Geometric(0.3).pmf(2), 0.147),
            (NegativeBinomial(2, 0.4).pmf(3), 0.13824),
        )
        for computed, quoted in cases:
            with self.subTest(quoted=quoted):
                self.assertEqual(significant(computed), significant(quoted))

    def test_independent_closed_forms(self) -> None:
        """Quoted values agree with formulas that bypass ``probstat``."""
        self.assertEqual(significant(STANDARD_NORMAL.cdf(1.96)), 0.9750021)
        self.assertEqual(significant(STANDARD_NORMAL.inv_cdf(0.975)), 1.959964)
        self.assertEqual(significant(STANDARD_NORMAL.inv_cdf(0.975) ** 2), 3.841459)
        self.assertEqual(significant(1.0 - math.exp(-1.5) * 2.5), 0.4421746)
        self.assertEqual(significant(1.0 - 1.6**-5), 0.9046326)
        self.assertEqual(significant(5.0 * (0.05 ** (-0.2) - 1.0)), 4.102821)
        self.assertEqual(significant(1.0 - math.exp(-2.0)), 0.8646647)
        self.assertEqual(significant(2.0 * math.log(2.0) ** (2.0 / 3.0)), 1.566440)
        self.assertEqual(significant(STANDARD_NORMAL.cdf(math.log(2.0))), 0.7558914)
        self.assertEqual(significant(1.0 / (1.0 + math.exp(-1.0))), 0.7310586)
        self.assertEqual(significant(math.exp(-3.0) * 8.5), 0.4231901)
        beta_cdf = sum(
            math.comb(6, j) * Fraction(3, 10) ** j * Fraction(7, 10) ** (6 - j)
            for j in range(2, 7)
        )
        self.assertEqual(beta_cdf, Fraction("0.579825"))
        self.assertEqual(Fraction(math.comb(10, 3), 2**10), Fraction("0.1171875"))
        self.assertEqual(
            sum(Fraction(math.comb(10, k), 2**10) for k in range(4)),
            Fraction("0.171875"),
        )
        self.assertEqual(
            Fraction(7, 10) ** 2 * Fraction(3, 10), Fraction("0.147")
        )
        self.assertEqual(
            4 * Fraction(2, 5) ** 2 * Fraction(3, 5) ** 3, Fraction("0.13824")
        )
        median = 3.356694
        self.assertLess(
            abs(1.0 - math.exp(-median / 2.0) * (1.0 + median / 2.0) - 0.5), 1e-6
        )

    def test_student_t_by_simpson_integration(self) -> None:
        """``qt(0.975, 10)`` and ``pt(2, 5)`` from a from-scratch density."""
        mass = simpson(lambda x: student_t_density(x, 10.0), 0.0, 2.228139, 2000)
        self.assertLess(abs(mass - 0.475), 1e-6)
        upper = simpson(lambda x: student_t_density(x, 5.0), 0.0, 2.0, 2000)
        self.assertEqual(significant(0.5 + upper), 0.9490303)


class DescriptiveValueTests(unittest.TestCase):
    """``var``, ``quantile(type = 7)``, ``IQR``, ``mad``, and ``ecdf``."""

    def test_descriptive_values(self) -> None:
        """Sample variance, type-7 quantiles, and R's MAD constant."""
        x = (2.1, 3.4, 1.9, 5.6, 4.4)
        self.assertAlmostEqual(variance(x, ddof=1), 2.437, places=12)
        self.assertAlmostEqual(statistics.variance(x), 2.437, places=12)
        expected = (1.98, 2.1, 3.4, 5.12)
        for computed, quoted in zip(
            quantiles(x, (0.1, 0.25, 0.5, 0.9)), expected, strict=True
        ):
            self.assertAlmostEqual(computed, quoted, places=12)
        ordered = sorted(x)
        for p, quoted in zip((0.1, 0.25, 0.5, 0.9), expected, strict=True):
            h = (len(x) - 1) * p
            low = math.floor(h)
            by_hand = ordered[low] + (h - low) * (ordered[low + 1] - ordered[low])
            self.assertAlmostEqual(by_hand, quoted, places=12)
        self.assertAlmostEqual(interquartile_range(x), 2.3, places=12)
        self.assertAlmostEqual(1.4826 * 1.3, 1.92738, places=12)
        self.assertLess(abs(median_absolute_deviation(x) - 1.92738), 5e-6)
        self.assertEqual(empirical_cdf(x, 3.4), 0.6)


class ClassicalTestValueTests(unittest.TestCase):
    """Quoted test statistics and p-values."""

    def test_t_tests(self) -> None:
        """Welch, pooled, and one-sample t tests."""
        x = (5.1, 4.8, 6.0, 5.7, 5.3, 6.2, 5.9, 5.5)
        y = (4.6, 5.0, 4.4, 5.2, 4.9, 4.3, 4.7)
        welch = two_sample_t_test(x, y)
        self.assertEqual(round(welch.statistic, 4), 3.9915)
        self.assertEqual(round(welch.df, 3), 12.337)
        self.assertEqual(significant(welch.p_value, 4), 0.001697)
        pooled = two_sample_t_test(x, y, equal_variances=True)
        self.assertEqual(round(pooled.statistic, 4), 3.8878)
        self.assertEqual(pooled.df, 13.0)
        self.assertEqual(significant(pooled.p_value, 4), 0.001869)
        self.assertEqual(
            significant(one_sample_t_test(x, null_mean=5.0).p_value, 4), 0.01261
        )
        mx, my = statistics.fmean(x), statistics.fmean(y)
        vx, vy = statistics.variance(x), statistics.variance(y)
        t = (mx - my) / math.sqrt(vx / 8 + vy / 7)
        df = (vx / 8 + vy / 7) ** 2 / ((vx / 8) ** 2 / 7 + (vy / 7) ** 2 / 6)
        self.assertEqual(round(t, 4), 3.9915)
        self.assertEqual(round(df, 3), 12.337)

    def test_chi_square_with_and_without_yates(self) -> None:
        """R's default 2 x 2 Yates correction versus ``correct = FALSE``."""
        table = ((12, 5), (7, 15))
        yates = chi_square_independence_test(table, yates_correction=True)
        plain = chi_square_independence_test(table)
        self.assertEqual(round(yates.statistic, 4), 4.3221)
        self.assertEqual(significant(yates.p_value, 4), 0.03762)
        self.assertEqual(round(plain.statistic, 4), 5.7696)
        self.assertEqual(significant(plain.p_value, 4), 0.01631)
        self.assertAlmostEqual(
            plain.p_value, math.erfc(math.sqrt(plain.statistic / 2.0)), places=14
        )

    def test_fisher_and_binomial(self) -> None:
        """Exact two-sided rules and the sample odds ratio."""
        tea = ((3, 1), (1, 3))
        result = fisher_exact_test(tea)
        self.assertAlmostEqual(result.p_value, 34 / 70, places=14)
        self.assertEqual(result.odds_ratio, 9.0)
        self.assertAlmostEqual(
            fisher_exact_test(tea, alternative="greater").p_value, 17 / 70, places=14
        )
        masses = [
            math.comb(20, k) * Fraction(3, 10) ** k * Fraction(7, 10) ** (20 - k)
            for k in range(21)
        ]
        threshold = masses[10] * Fraction(10_000_001, 10_000_000)
        exact = float(sum(m for m in masses if m <= threshold))
        computed = binomial_test(10, 20, null_probability=0.3).p_value
        self.assertAlmostEqual(computed, exact, places=13)
        self.assertEqual(significant(computed, 4), 0.08345)
        symmetric = binomial_test(7, 20).p_value
        lower_tail = sum(Fraction(math.comb(20, k), 2**20) for k in range(8))
        self.assertAlmostEqual(symmetric, float(2 * lower_tail), places=14)
        self.assertEqual(significant(symmetric, 4), 0.2632)

    def test_rank_tests(self) -> None:
        """Signed-rank ``V`` by enumeration and rank-sum ``W``."""
        before = (1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30)
        after = (0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29)
        signed = wilcoxon_signed_rank_test(before, after)
        self.assertEqual(signed.statistic, 40.0)
        self.assertTrue(signed.exact)
        at_least = sum(
            1
            for signs in itertools.product((0, 1), repeat=9)
            if sum(rank * s for rank, s in zip(range(1, 10), signs, strict=True)) >= 40
        )
        self.assertEqual(at_least, 10)
        self.assertAlmostEqual(signed.p_value, 2 * at_least / 512, places=15)
        a = (0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
        b = (1.15, 0.88, 0.90, 0.74, 1.21)
        rank_sum = mann_whitney_u_test(a, b)
        u = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)
        self.assertEqual(rank_sum.statistic, u)
        self.assertEqual(u, 35.0)
        self.assertEqual(significant(rank_sum.p_value, 4), 0.2544)

    def test_p_adjust(self) -> None:
        """Holm, BH, and BY follow R's ``cummax``/``cummin`` definitions."""
        p = (0.01, 0.04, 0.03, 0.005, 0.2)
        n = len(p)
        order = sorted(range(n), key=lambda i: p[i])
        holm = [0.0] * n
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (n - rank) * p[index])
            holm[index] = min(1.0, running)
        bh = [0.0] * n
        running = 1.0
        for rank in range(n - 1, -1, -1):
            index = order[rank]
            running = min(running, n / (rank + 1) * p[index])
            bh[index] = running
        harmonic = sum(1.0 / i for i in range(1, n + 1))
        expected = {
            "holm": (0.04, 0.09, 0.09, 0.025, 0.2),
            "bh": (0.025, 0.05, 0.05, 0.025, 0.2),
            "by": (0.05708333, 0.11416667, 0.11416667, 0.05708333, 0.45666667),
        }
        references = {
            "holm": holm,
            "bh": bh,
            "by": [min(1.0, harmonic * value) for value in bh],
        }
        for method, quoted in expected.items():
            computed = adjust_p_values(p, method=method).adjusted_p_values  # type: ignore[arg-type]
            for c, r, q in zip(computed, references[method], quoted, strict=True):
                with self.subTest(method=method, quoted=q):
                    self.assertAlmostEqual(c, r, places=14)
                    self.assertAlmostEqual(c, q, places=8)


class OptimizationAndModelTests(unittest.TestCase):
    """``uniroot``, ``lm``, and ``mvtnorm`` equivalents."""

    def test_root_regression_and_mvn(self) -> None:
        """Plastic-number root, closed-form OLS, and bivariate normal values."""
        root = brent_root(lambda t: t**3 - t - 1.0, 1.0, 2.0).root
        self.assertAlmostEqual(root, 1.324717957244746, places=12)
        fit = fit_simple_linear_regression((1, 2, 3, 4, 5), (2.1, 3.9, 6.2, 7.8, 10.1))
        self.assertAlmostEqual(fit.slope, 19.9 / 10.0, places=12)
        self.assertAlmostEqual(fit.intercept, 0.05, places=12)
        self.assertEqual(round(fit.r_squared, 4), 0.9973)
        self.assertEqual(round(math.sqrt(fit.residual_variance), 4), 0.1889)
        law = MultivariateNormal((0.0, 0.0), ((1.0, 0.5), (0.5, 1.0)))
        self.assertEqual(significant(law.pdf((0.5, -0.2))), 0.141701)
        direct = math.exp(
            -(0.25 + 0.1 + 0.04) / (2.0 * 0.75)
        ) / (2.0 * math.pi * math.sqrt(0.75))
        self.assertAlmostEqual(law.pdf((0.5, -0.2)), direct, places=14)
        cdf = bivariate_normal_cdf(1.0, 1.0, 0.5)
        self.assertEqual(significant(cdf), 0.7452036)

        def plackett(r: float) -> float:
            """Return the bivariate normal density at ``(1, 1)`` with correlation r.

            References:
                [REF-GENZ-2004]. Test helper only.

            """
            return math.exp(-(2.0 - 2.0 * r) / (2.0 * (1.0 - r * r))) / (
                2.0 * math.pi * math.sqrt(1.0 - r * r)
            )

        independent = STANDARD_NORMAL.cdf(1.0) ** 2 + simpson(plackett, 0.0, 0.5, 200)
        self.assertAlmostEqual(independent, cdf, places=10)


if __name__ == "__main__":
    unittest.main()
