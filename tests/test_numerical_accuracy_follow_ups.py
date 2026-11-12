"""Regression tests for the numerical-accuracy follow-ups of ADR-0089.

Every oracle is independent of the float code under test: exact rational
arithmetic (``fractions``), 40-digit decimal references from
:mod:`probstat.numerics.reference` (ADR-0077), and an exact-rational
re-derivation of Temme's coefficients. ULP budgets are those recorded in
ADR-0089; each test also names the defect it guards against.
"""

from __future__ import annotations

import math
import random
import unittest
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction

from probstat.distributions import Normal
from probstat.numerics import _double_double as dd
from probstat.numerics import (
    compare_to_reference,
    reference_log,
    reference_log_gamma,
    reference_normal_cdf,
    reference_pi,
)


def grid(lower: float, upper: float, count: int) -> list[float]:
    """Return ``count`` equally spaced floats from ``lower`` to ``upper``.

    References:
        [EQ-NUM-REFERENCE-008].

    """
    return [lower + (upper - lower) * i / (count - 1) for i in range(count)]


def exact_z(x: float, mu: float, sigma: float) -> Fraction:
    """Return the exact standardized point ``(x - mu)/sigma`` of float inputs.

    References:
        [REF-PYTHON-FRACTIONS].

    """
    return (Fraction(x) - Fraction(mu)) / Fraction(sigma)


def reference_log_cdf(z: Fraction) -> Decimal:
    """Return ``log Phi(z)``; above zero ``-sum q^k/k`` with ``q = Phi(-z)``.

    ``log(1 - q)`` of a rounded ``1 - q`` would lose ``q`` entirely once it
    is below the working precision, so the series is summed directly.

    References:
        [EQ-NUM-REFERENCE-002], [REF-NIST-DLMF, Section 7.2].

    """
    if z <= 0:
        return reference_normal_cdf(z).ln()
    q = reference_normal_cdf(-z)
    total, power, k = Decimal(0), q, 1
    bound = Decimal(10) ** (-getcontext().prec - 5)
    while True:
        term = power / k
        total += term
        if term <= total * bound:
            return -total
        power *= q
        k += 1


class DoubleDoubleTests(unittest.TestCase):
    """The error-free transformations and the double-double kernels."""

    def test_error_free_transformations_are_exact(self) -> None:
        """two_sum and two_product reproduce the exact rational result."""
        generator = random.Random(89)
        for _ in range(500):
            a = generator.uniform(-1, 1) * 10.0 ** generator.randint(-150, 150)
            b = generator.uniform(-1, 1) * 10.0 ** generator.randint(-150, 150)
            total, error = dd.two_sum(a, b)
            exact_sum = Fraction(a) + Fraction(b)
            self.assertEqual(Fraction(total) + Fraction(error), exact_sum)
            product, error = dd.two_product(a, b)
            self.assertEqual(
                Fraction(product) + Fraction(error), Fraction(a) * Fraction(b)
            )

    def test_constants_against_references(self) -> None:
        """The split constants equal 40-digit references to 1e-32 relative."""
        with localcontext() as context:
            context.prec = 45
            pi = reference_pi(digits=45)
            expected = {
                "LOG_SQRT_TWO_PI": reference_log(2 * pi, digits=45) / 2,
                "INV_SQRT_TWO": Decimal(2).sqrt() / 2,
            }
            for name, value in expected.items():
                high, low = getattr(dd, name)
                gap = abs(Decimal(high) + Decimal(low) - value) / value
                self.assertLess(gap, Decimal("1e-32"), name)

    def test_logarithm_and_log1pmx(self) -> None:
        """dd_log within 1e-22 absolute, dd_log1pmx within 1e-20 relative."""
        generator = random.Random(1)
        with localcontext() as context:
            context.prec = 50
            for _ in range(400):
                x = math.exp(generator.uniform(-700.0, 700.0))
                high, low = dd.dd_log((x, 0.0))
                exact = Decimal(x).ln()
                self.assertLess(abs(Decimal(high) + Decimal(low) - exact), 1e-22)
                t = generator.uniform(-0.5, 0.5) * 10.0 ** generator.uniform(-8, 0)
                high, low = dd.dd_log1pmx((t, 0.0))
                exact = (1 + Decimal(t)).ln() - Decimal(t)
                gap = abs(Decimal(high) + Decimal(low) - exact) / abs(exact)
                self.assertLess(gap, 1e-20, t)

    def test_stirling_remainder_against_log_gamma(self) -> None:
        """sigma(a) within 2e-18 absolute, including shapes lifted below 15."""
        with localcontext() as context:
            context.prec = 50
            pi = reference_pi(digits=50)
            for a in (1e-300, 1e-5, 0.05, 0.5, 2.5, 7.5, 14.9, 15.0, 200.0, 1e6):
                shape = Decimal(a)
                exact = (
                    reference_log_gamma(a, digits=50)
                    - (shape - Decimal("0.5")) * shape.ln()
                    + shape
                    - (2 * pi).ln() / 2
                )
                high, low = dd.dd_stirling_error((a, 0.0))
                self.assertLess(abs(Decimal(high) + Decimal(low) - exact), 2e-18, a)


class NormalTailTests(unittest.TestCase):
    """Normal cdf, sf, log_cdf, and log_sf keep a few ulp in every tail."""

    def test_cdf_and_sf_over_the_whole_range(self) -> None:
        """At most 8 ulp for z in [-37.5, 37.5]; the old code had 1228 ulp at -37."""
        for mu, sigma in ((0.0, 1.0), (3.0, 0.7), (-1e3, 13.3)):
            law = Normal(mu, sigma)
            lower = [mu + sigma * z for z in grid(-37.5, 8.0, 47)]
            upper = [mu + sigma * z for z in grid(-8.0, 37.5, 47)]
            def lower_tail(x: float, m: float = mu, s: float = sigma) -> Decimal:
                """Return the exact-argument reference ``Phi((x - m)/s)``.

                References:
                    [EQ-NUM-REFERENCE-002].

                """
                return reference_normal_cdf(exact_z(x, m, s))

            def upper_tail(x: float, m: float = mu, s: float = sigma) -> Decimal:
                """Return the exact-argument reference ``1 - Phi((x - m)/s)``.

                References:
                    [EQ-NUM-REFERENCE-002].

                """
                return reference_normal_cdf(-exact_z(x, m, s))

            cdf = compare_to_reference(law.cdf, lower_tail, lower)
            sf = compare_to_reference(law.sf, upper_tail, upper)
            self.assertLessEqual(cdf.max_ulp, 8.0, (mu, sigma, cdf.worst_point))
            self.assertLessEqual(sf.max_ulp, 8.0, (mu, sigma, sf.worst_point))

    def test_log_cdf_near_zero_keeps_relative_accuracy(self) -> None:
        """Log Phi(z) for 0 < z <= 5 was log of a rounded Phi: 1.26e6 ulp at z = 5."""
        law = Normal()
        points = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.5, 19.0, 37.0]
        result = compare_to_reference(
            law.log_cdf, lambda x: reference_log_cdf(Fraction(x)), points
        )
        self.assertLessEqual(result.max_ulp, 4.0, result.worst_point)
        survival = compare_to_reference(
            law.log_sf, lambda x: reference_log_cdf(-Fraction(x)), [-z for z in points]
        )
        self.assertLessEqual(survival.max_ulp, 4.0, survival.worst_point)

    def test_log_tails_far_beyond_underflow(self) -> None:
        """log_cdf and log_sf stay within 4 ulp down to z = -1000."""
        law = Normal(3.0, 0.7)
        points = [3.0 + 0.7 * z for z in grid(-40.0, 0.0, 21) + [-1e3, -200.0, -50.0]]
        result = compare_to_reference(
            law.log_cdf,
            lambda x: reference_log_cdf(exact_z(x, 3.0, 0.7)),
            points,
        )
        self.assertLessEqual(result.max_ulp, 4.0, result.worst_point)


if __name__ == "__main__":
    unittest.main()
