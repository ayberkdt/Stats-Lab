"""Regression tests for the second round of independently confirmed defects.

Every oracle here is computed independently of the code under test: closed
forms, exact rational arithmetic (``fractions``), or 50-100 digit
``decimal`` evaluations of series written out in this file.
"""

from __future__ import annotations

import itertools
import math
import statistics
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

from probstat.distributions import (
    Beta,
    ChiSquare,
    FisherF,
    Gamma,
    StudentT,
    TruncatedNormal,
    Uniform,
)
from probstat.distributions._special import regularized_beta, regularized_gamma_p
from probstat.foundations.random_streams import resolve_stream
from probstat.inference import (
    bootstrap_confidence_interval,
    kolmogorov_smirnov_test,
    noncentral_t_cdf,
    one_way_anova,
    permutation_test,
    two_sample_t_test,
)
from probstat.inference.goodness_of_fit import kolmogorov_exact_sf
from probstat.inference.resampling import _resamples
from probstat.numerics import trigamma
from probstat.processes import MarkovChain
from probstat.validation import ValidationDomain, u_pooling

_PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640"
)
# B_(2k) for k = 1..8.
_BERNOULLI = (
    Fraction(1, 6),
    Fraction(-1, 30),
    Fraction(1, 42),
    Fraction(-1, 30),
    Fraction(5, 66),
    Fraction(-691, 2730),
    Fraction(7, 6),
    Fraction(-3617, 510),
)


def decimal_lgamma(a: float | Decimal) -> Decimal:
    """Return ``log Gamma(a)`` for ``a >= 1000`` by the Stirling series.

    With eight Bernoulli terms the truncation error is below ``1e-50``.

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    x = Decimal(a)
    total = (x - Decimal("0.5")) * x.ln() - x + (2 * _PI).ln() / 2
    for k, bernoulli in enumerate(_BERNOULLI, start=1):
        coefficient = Decimal(bernoulli.numerator) / Decimal(bernoulli.denominator)
        total += coefficient / (2 * k * (2 * k - 1) * x ** (2 * k - 1))
    return total


def decimal_gamma_p(a: float, x: float) -> Decimal:
    """Return ``P(a, x)`` by its power series in 50-digit decimal arithmetic.

    References:
        [REF-NIST-DLMF-GAMMA, Section 8.7].

    """
    shape, point = Decimal(a), Decimal(x)
    term = total = Decimal(1)
    k = 0
    while term > total * Decimal("1e-45"):
        k += 1
        term = term * point / (shape + k)
        total += term
    log_front = shape * point.ln() - point - decimal_lgamma(a) - shape.ln()
    return total * log_front.exp()


def decimal_beta_cdf(a: float, b: float, x: float | Decimal) -> Decimal:
    """Return ``I_x(a, b)`` from the Lentz continued fraction in decimal.

    References:
        [REF-NIST-DLMF-BETA, Section 8.17].

    """
    first, second, point = Decimal(a), Decimal(b), Decimal(x)
    one = Decimal(1)
    if point > (first + 1) / (first + second + 2):
        return one - decimal_beta_cdf(b, a, 1 - point)
    c = one
    d = one / (one - (first + second) * point / (first + 1))
    fraction = d
    m = 0
    while True:
        m += 1
        coefficient = m * (second - m) * point / ((first + 2 * m - 1) * (first + 2 * m))
        d = one / (one + coefficient * d)
        c = one + coefficient / c
        fraction *= d * c
        coefficient = -(first + m) * (first + second + m) * point / (
            (first + 2 * m) * (first + 2 * m + 1)
        )
        d = one / (one + coefficient * d)
        c = one + coefficient / c
        fraction *= d * c
        if abs(d * c - one) < Decimal("1e-45"):
            break
    log_front = (
        first * point.ln()
        + second * (one - point).ln()
        - decimal_lgamma(a)
        - decimal_lgamma(b)
        + decimal_lgamma(first + second)
    )
    return log_front.exp() * fraction / first


def marsaglia_tsang_wang_cdf(n: int, d: Fraction) -> Fraction:
    """Return ``P(D_n < d)`` exactly from the Marsaglia-Tsang-Wang matrix formula.

    References:
        [REF-MARSAGLIA-TSANG-WANG-2003].

    """
    k = math.floor(n * d) + 1
    m = 2 * k - 1
    h = k - n * d
    matrix = [
        [
            Fraction(1, math.factorial(i - j + 1)) if i - j + 1 >= 0 else Fraction(0)
            for j in range(m)
        ]
        for i in range(m)
    ]
    for i in range(m):
        matrix[i][0] -= h ** (i + 1) / math.factorial(i + 1)
        matrix[m - 1][i] -= h ** (m - i) / math.factorial(m - i)
    if 2 * h - 1 > 0:
        matrix[m - 1][0] += (2 * h - 1) ** m / math.factorial(m)

    def product(left: list[list[Fraction]], right: list[list[Fraction]]) -> list:
        """Multiply two square rational matrices.

        References:
            [REF-MARSAGLIA-TSANG-WANG-2003].

        """
        return [
            [sum(left[i][t] * right[t][j] for t in range(m)) for j in range(m)]
            for i in range(m)
        ]

    power = [[Fraction(int(i == j)) for j in range(m)] for i in range(m)]
    for _ in range(n):
        power = product(power, matrix)
    return power[k - 1][k - 1] * Fraction(math.factorial(n), n**n)


class KolmogorovSmirnovExactTests(unittest.TestCase):
    """Small-sample KS p-values are exact, so u-pooling can reject."""

    def test_single_and_paired_observation_oracles(self) -> None:
        """For n = 1 the p-value is 2(1 - d); the (0.9, 0.97) pair gives 0.02."""
        uniform = Uniform(0.0, 1.0)
        for values, expected in (
            ((1.0,), 0.0),
            ((0.995,), 0.01),
            ((0.98,), 0.04),
            ((0.9, 0.97), 0.02),
        ):
            with self.subTest(values=values):
                result = kolmogorov_smirnov_test(values, uniform)
                self.assertAlmostEqual(result.p_value, expected, delta=1e-14)
                self.assertIn("exact", result.method)

    def test_matches_exact_rational_matrix_formula(self) -> None:
        """The recursion agrees with Marsaglia-Tsang-Wang in exact arithmetic."""
        for n in range(1, 7):
            for d in (Fraction(3, 20), Fraction(1, 4), Fraction(2, 5), Fraction(7, 10)):
                if d <= Fraction(1, 2 * n):
                    continue
                exact = 1 - marsaglia_tsang_wang_cdf(n, d)
                with self.subTest(n=n, d=d):
                    self.assertAlmostEqual(
                        kolmogorov_exact_sf(n, float(d)), float(exact), delta=1e-14
                    )

    def test_three_quartiles_exact_value(self) -> None:
        """For (1/4, 1/2, 3/4) the exact p-value is 35/36."""
        result = kolmogorov_smirnov_test((0.25, 0.5, 0.75), Uniform(0.0, 1.0))
        self.assertAlmostEqual(result.p_value, 35.0 / 36.0, delta=1e-14)

    def test_u_pooling_rejects_an_absurd_single_prediction(self) -> None:
        """One observation far above every prediction gives a pooled p of 0."""
        pooled = u_pooling([((0.0, 0.1, 0.2), (50.0,))])
        self.assertEqual(pooled.u_values, (1.0,))
        self.assertEqual(pooled.ks.p_value, 0.0)


class LargeShapeGammaTests(unittest.TestCase):
    """The incomplete gamma ratio converges and stays accurate for huge shapes."""

    def test_series_regime_below_the_mean(self) -> None:
        """Gamma(3e6).cdf(a - sqrt(a)/2) matches a 50-digit series."""
        a = 3e6
        x = a - 0.5 * math.sqrt(a)
        with localcontext() as context:
            context.prec = 50
            exact = float(decimal_gamma_p(a, x))
        self.assertAlmostEqual(Gamma(a).cdf(x) / exact, 1.0, delta=1e-13)

    def test_chi_square_at_its_mean(self) -> None:
        """ChiSquare(2e7).cdf(2e7) = P(1e7, 1e7) to about 1e-15."""
        with localcontext() as context:
            context.prec = 50
            exact = float(decimal_gamma_p(1e7, 1e7))
        self.assertAlmostEqual(ChiSquare(2e7).cdf(2e7) / exact, 1.0, delta=1e-13)
        self.assertAlmostEqual(regularized_gamma_p(1e7, 1e7) / exact, 1.0, delta=1e-13)


class LargeParameterDensityTests(unittest.TestCase):
    """Densities keep ~1e-12 relative accuracy at parameters up to 1e12."""

    def test_student_t_log_density(self) -> None:
        """StudentT(nu).log_pdf(1.3) for nu = 1e10, 1e12."""
        for nu in (1e10, 1e12):
            with localcontext() as context:
                context.prec = 60
                v, z = Decimal(nu), Decimal(1.3)
                exact = (
                    decimal_lgamma((v + 1) / 2)
                    - decimal_lgamma(v / 2)
                    - (v * _PI).ln() / 2
                    - (v + 1) / 2 * (1 + z * z / v).ln()
                )
            with self.subTest(nu=nu):
                self.assertAlmostEqual(
                    StudentT(nu).log_pdf(1.3), float(exact), delta=1e-12
                )

    def test_gamma_log_density(self) -> None:
        """Gamma(k).log_pdf near the mode for k = 1e10, 1e12."""
        for k in (1e10, 1e12):
            x = k + 0.7 * math.sqrt(k)
            with localcontext() as context:
                context.prec = 60
                shape, point = Decimal(k), Decimal(x)
                exact = (shape - 1) * point.ln() - point - decimal_lgamma(k)
            with self.subTest(k=k):
                self.assertAlmostEqual(Gamma(k).log_pdf(x), float(exact), delta=1e-12)

    def test_beta_and_f_log_densities(self) -> None:
        """Beta(a, a) and F(nu, nu) log densities at a, nu = 1e10."""
        a = 1e10
        x = 0.5 + 0.3 / math.sqrt(a)
        nu = 1e10
        f = 1.0 + 1e-5
        with localcontext() as context:
            context.prec = 60
            shape, point = Decimal(a), Decimal(x)
            log_beta = 2 * decimal_lgamma(a) - decimal_lgamma(2 * shape)
            beta_exact = (shape - 1) * (point.ln() + (1 - point).ln()) - log_beta
            half, value = Decimal(nu) / 2, Decimal(f)
            y = half * value / (half * value + half)
            f_log_beta = 2 * decimal_lgamma(half) - decimal_lgamma(2 * half)
            f_exact = half * y.ln() + half * (1 - y).ln() - value.ln() - f_log_beta
        self.assertAlmostEqual(Beta(a, a).log_pdf(x), float(beta_exact), delta=1e-12)
        self.assertAlmostEqual(FisherF(nu, nu).log_pdf(f), float(f_exact), delta=1e-12)

    def test_beta_cdf_at_large_shapes(self) -> None:
        """Beta(1e8, 1e8).cdf keeps 1e-11 relative accuracy (was 6e-8)."""
        a = 1e8
        x = 0.5 + 0.3 / math.sqrt(a)
        with localcontext() as context:
            context.prec = 60
            exact = float(decimal_beta_cdf(a, a, x))
        self.assertAlmostEqual(Beta(a, a).cdf(x) / exact, 1.0, delta=1e-11)
        self.assertAlmostEqual(regularized_beta(x, a, a) / exact, 1.0, delta=1e-11)


def _decimal_erf_integral(x: Decimal) -> Decimal:
    """Return ``sum (-1)^n x^(2n+1) / (n! (2n+1))`` (``sqrt(pi)/2 erf(x)``).

    References:
        [REF-NIST-DLMF, Section 7.6].

    """
    total = term = x
    n = 0
    square = x * x
    while True:
        n += 1
        term = -term * square / n
        piece = term / (2 * n + 1)
        total += piece
        if abs(piece) < Decimal("1e-90"):
            return total


def _decimal_truncated_moments(low: float, high: float) -> tuple[float, float]:
    """Return the standard-normal mean and variance on ``[low, high]``.

    ``Z = sqrt(2) (E(high/sqrt 2) - E(low/sqrt 2))`` with ``E`` the scaled
    error-function series; the closed-form moments are then evaluated with
    100 digits, so the cancellation that defeats float arithmetic is
    harmless.

    References:
        [REF-JKB-1994, Chapter 13].

    """
    with localcontext() as context:
        context.prec = 100
        a, b = Decimal(low), Decimal(high)
        root = Decimal(2).sqrt()
        upper_part = _decimal_erf_integral(b / root)
        mass = root * (upper_part - _decimal_erf_integral(a / root))
        da, db = (-a * a / 2).exp(), (-b * b / 2).exp()
        mean = (da - db) / mass
        variance = 1 + (a * da - b * db) / mass - mean * mean
        return float(mean), float(variance)


class TruncatedNormalNarrowIntervalTests(unittest.TestCase):
    """Narrow two-sided truncations no longer cancel in mean and variance."""

    def test_symmetric_and_tail_intervals(self) -> None:
        """(-1e-8, 1e-8) variance 3.33e-17; (10, 10.0001) to 1e-12."""
        for low, high in ((-1e-8, 1e-8), (10.0, 10.0001), (3.0, 3.001)):
            mean, variance = _decimal_truncated_moments(low, high)
            law = TruncatedNormal(0.0, 1.0, low, high)
            with self.subTest(interval=(low, high)):
                self.assertAlmostEqual(law.mean, mean, delta=1e-14 * max(1, abs(mean)))
                self.assertAlmostEqual(law.variance / variance, 1.0, delta=1e-12)


def _noncentral_t_series(t: float, nu: float, delta: float) -> float:
    """Return the noncentral t CDF for ``t >= 0`` by its Poisson mixture series.

    ``F = Phi(-delta) + 1/2 sum_j [p_j I_x(j+1/2, nu/2) + q_j I_x(j+1, nu/2)]``
    with ``x = t^2/(t^2+nu)``, ``p_j = e^-l l^j/j!``,
    ``q_j = delta e^-l l^j / (sqrt 2 Gamma(j + 3/2))``, ``l = delta^2/2``.

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995].

    """
    x = t * t / (t * t + nu)
    lam = 0.5 * delta * delta
    total = 0.5 * math.erfc(delta / math.sqrt(2.0))
    terms = []
    for j in range(200):
        p = math.exp(-lam + j * math.log(lam) - math.lgamma(j + 1.0))
        q = delta * math.exp(-lam + j * math.log(lam) - math.lgamma(j + 1.5))
        q /= math.sqrt(2.0)
        terms.append(p * regularized_beta(x, j + 0.5, 0.5 * nu))
        terms.append(q * regularized_beta(x, j + 1.0, 0.5 * nu))
    return total + 0.5 * math.fsum(terms)


def _noncentral_t_brute_force(t: float, nu: float, delta: float) -> float:
    """Return the mixture integral by 16000 fixed 10-point Gauss panels.

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

    """
    nodes = (
        -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
        -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
        0.4333953941292472, 0.6794095682990244, 0.8650633666889845,
        0.9739065285171717,
    )
    weights = (
        0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
        0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
        0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
        0.0666713443086881,
    )
    half = 0.5 * nu
    log_constant = math.log(2.0) + half * math.log(half) - math.lgamma(half)
    upper = 1.0 + 12.0 / math.sqrt(nu)
    panels = 16000
    width = upper / panels
    pieces = []
    for index in range(panels):
        middle = (index + 0.5) * width
        for node, weight in zip(nodes, weights, strict=True):
            s = middle + 0.5 * width * node
            density = math.exp(log_constant + (nu - 1.0) * math.log(s) - half * s * s)
            normal = 0.5 * math.erfc(-(t * s - delta) / math.sqrt(2.0))
            pieces.append(0.5 * width * weight * normal * density)
    return math.fsum(pieces)


class NoncentralTTests(unittest.TestCase):
    """The noncentral t CDF is accurate near nu = 1 and for sharp steps."""

    def test_small_degrees_of_freedom(self) -> None:
        """nct(1; 1.01, 0.2) = 0.691306 (Simpson gave 0.690627)."""
        exact = _noncentral_t_series(1.0, 1.01, 0.2)
        self.assertAlmostEqual(noncentral_t_cdf(1.0, 1.01, 0.2), exact, delta=1e-12)
        self.assertAlmostEqual(exact, 0.6913060633647, delta=1e-12)

    def test_sharp_normal_step(self) -> None:
        """nct(2000; 30, 1950) = 0.542967 (Simpson gave 0.542586)."""
        exact = _noncentral_t_brute_force(2000.0, 30.0, 1950.0)
        self.assertAlmostEqual(
            noncentral_t_cdf(2000.0, 30.0, 1950.0), exact, delta=1e-11
        )
        self.assertAlmostEqual(exact, 0.5429673869, delta=1e-9)

    def test_moderate_cases_against_series(self) -> None:
        """Ordinary arguments agree with the series to 1e-12."""
        for t, nu, delta in ((2.0, 5.0, 1.0), (0.4, 3.0, -0.5), (3.0, 1.5, 2.0)):
            with self.subTest(t=t, nu=nu, delta=delta):
                self.assertAlmostEqual(
                    noncentral_t_cdf(t, nu, delta),
                    _noncentral_t_series(t, nu, delta),
                    delta=1e-12,
                )


class ResamplingTests(unittest.TestCase):
    """Studentized intervals survive degenerate resamples; ties survive offsets."""

    def test_studentized_interval_skips_constant_resamples(self) -> None:
        """Zero-SE resamples are skipped and counted, not fatal (n = 3)."""
        data = (1.0, 2.0, 4.0)
        replicates = 500
        result = bootstrap_confidence_interval(
            statistics.fmean,
            data,
            method="studentized",
            replicates=replicates,
            seed=7,
        )
        constant = sum(
            1
            for sample in _resamples(data, replicates, resolve_stream(7))
            if len(set(sample)) == 1
        )
        self.assertGreater(constant, 0)
        self.assertEqual(result.degenerate_resamples, constant)
        self.assertEqual(len(result.bootstrap.replicates), replicates)
        self.assertLess(result.lower, 7.0 / 3.0)
        self.assertGreater(result.upper, 7.0 / 3.0)

    def test_studentized_median_of_five_no_longer_raises(self) -> None:
        """The median of five values gives many zero jackknife SEs."""
        result = bootstrap_confidence_interval(
            statistics.median,
            (1.0, 2.0, 3.0, 4.0, 5.0),
            method="studentized",
            replicates=200,
            seed=1,
        )
        self.assertGreater(result.degenerate_resamples, 0)
        self.assertTrue(math.isfinite(result.lower) and math.isfinite(result.upper))

    def test_permutation_ties_survive_a_large_offset(self) -> None:
        """Offset 1e9: p = 0.4 exactly (previously 0.2857)."""
        first = (0.3, 0.3, 0.0)
        second = (1.2, 0.3, 0.6, 0.1)
        offset = 1e9
        result = permutation_test(
            [offset + v for v in first], [offset + v for v in second]
        )
        exact_first = [Fraction(repr(v)) for v in first]
        exact_second = [Fraction(repr(v)) for v in second]
        pooled = exact_first + exact_second
        total = sum(pooled)
        observed = sum(exact_first) / 3 - (total - sum(exact_first)) / 4
        hits = 0
        count = 0
        for chosen in itertools.combinations(range(7), 3):
            subtotal = sum(pooled[i] for i in chosen)
            count += 1
            hits += abs(subtotal / 3 - (total - subtotal) / 4) >= abs(observed)
        self.assertEqual(Fraction(hits, count), Fraction(2, 5))
        self.assertAlmostEqual(result.p_value, 0.4, delta=1e-15)


class StudentTExtremeTailTests(unittest.TestCase):
    """Tails stay finite beyond the overflow point of t^2."""

    def test_cauchy_tail_and_quantile(self) -> None:
        """Cauchy: cdf(-z) = 1/(pi z) far out; quantile(p) = -1/(pi p) as p -> 0."""
        law = StudentT(1.0)
        for z in (1e150, 1e160, 1e200, 1e300):
            with self.subTest(z=z):
                self.assertAlmostEqual(law.cdf(-z) * z * math.pi, 1.0, delta=1e-13)
                self.assertAlmostEqual(law.sf(z) * z * math.pi, 1.0, delta=1e-13)
        self.assertAlmostEqual(
            law.quantile(1e-300) * 1e-300 * math.pi, -1.0, delta=1e-12
        )


class MarkovAbsorptionTests(unittest.TestCase):
    """Absorbing states follow the row tolerance, not P_ii == 1.0."""

    def test_rounded_diagonal_is_still_absorbing(self) -> None:
        """P_22 = 1 - 1e-11 is absorbing; absorption is 3/7 and 4/7."""
        chain = MarkovChain([[1.0, 0.0, 0.0], [0.3, 0.3, 0.4], [0.0, 0.0, 1.0 - 1e-11]])
        self.assertEqual(chain.absorbing_states(), (0, 2))
        analysis = chain.absorption()
        self.assertAlmostEqual(analysis.probability(1, 0), 3.0 / 7.0, delta=1e-15)
        self.assertAlmostEqual(analysis.probability(1, 2), 4.0 / 7.0, delta=1e-15)


class TrigammaRangeTests(unittest.TestCase):
    """Trigamma near zero is exact or raises OverflowError deliberately."""

    def test_tiny_arguments(self) -> None:
        """1/x^2 + pi^2/6 when representable; OverflowError beyond."""
        for x in (1e-100, 1e-154):
            with self.subTest(x=x):
                expected = 1.0 / x**2 if x > 1e-120 else (1.0 / x) ** 2
                self.assertAlmostEqual(trigamma(x) / expected, 1.0, delta=1e-15)
        for x in (1e-160, 1e-300, 5e-324):
            with self.subTest(x=x), self.assertRaises(OverflowError):
                trigamma(x)


class QuantileSaturationTests(unittest.TestCase):
    """Quantiles beyond the float range return the true limits."""

    def test_limits_instead_of_finite_boundaries(self) -> None:
        """Underflowing quantiles return 0; overflowing ones return -inf/inf."""
        self.assertEqual(Gamma(0.01).quantile(1e-300), 0.0)
        self.assertEqual(StudentT(0.5).quantile(1e-300), -math.inf)
        self.assertEqual(StudentT(0.5).isf(1e-300), math.inf)
        law = Gamma(0.5)
        self.assertAlmostEqual(law.cdf(law.quantile(1e-10)) / 1e-10, 1.0, delta=1e-10)


class ValidationDomainSlackTests(unittest.TestCase):
    """The bounding-box slack is relative to the range width."""

    def test_narrow_range_at_large_offset(self) -> None:
        """A 1e-4 range at 1e6 no longer admits points 1e-4 outside it."""
        domain = ValidationDomain(("x",), ((1e6,), (1e6 + 1e-4,)), ("a", "b"))
        self.assertTrue(domain.check({"x": 1e6 + 5e-5}).inside_box)
        self.assertFalse(domain.check({"x": 1e6 + 2e-4}).inside_box)

    def test_zero_width_range_rule(self) -> None:
        """Zero width: relative 1e-9 agreement, exact equality at zero."""
        domain = ValidationDomain(("x",), ((5.0,), (5.0,)), ("a", "b"))
        self.assertTrue(domain.check({"x": 5.0 * (1.0 + 1e-12)}).inside_box)
        self.assertFalse(domain.check({"x": 5.0 + 1e-6}).inside_box)
        zero = ValidationDomain(("x",), ((0.0,), (0.0,)), ("a", "b"))
        self.assertTrue(zero.check({"x": 0.0}).inside_box)
        self.assertFalse(zero.check({"x": 1e-300}).inside_box)


class OffsetCentringTests(unittest.TestCase):
    """ANOVA and Welch statistics are exact for offset dyadic data."""

    def test_anova_and_welch_at_offset_1e9(self) -> None:
        """F = 8/5 and t from exact rationals (previously 6e-5 off)."""
        offset = 1e9
        first = [offset + k / 1024 for k in (1, 2, 3)]
        second = [offset + k / 1024 for k in (2, 3, 5)]
        a = [Fraction(v) for v in first]
        b = [Fraction(v) for v in second]
        mean_a, mean_b = sum(a) / 3, sum(b) / 3
        grand = (sum(a) + sum(b)) / 6
        between = 3 * (mean_a - grand) ** 2 + 3 * (mean_b - grand) ** 2
        within_a = sum((v - mean_a) ** 2 for v in a)
        within_b = sum((v - mean_b) ** 2 for v in b)
        exact_f = between / ((within_a + within_b) / 4)
        self.assertEqual(exact_f, Fraction(8, 5))
        self.assertAlmostEqual(one_way_anova(first, second).statistic, 1.6, delta=1e-14)
        exact_t = float(mean_a - mean_b) / math.sqrt(
            float((within_a / 2) / 3 + (within_b / 2) / 3)
        )
        self.assertAlmostEqual(
            two_sample_t_test(first, second).statistic / exact_t, 1.0, delta=1e-14
        )


if __name__ == "__main__":
    unittest.main()
