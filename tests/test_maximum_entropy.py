"""Tests for maximum-entropy laws under moment and expectation constraints.

Oracles: the closed-form maximum-entropy laws (uniform, exponential, normal,
truncated exponential through its one-dimensional mean equation, gamma
through ``E[log X]``, Laplace through ``E[|X|]``), the entropy formulas of
:mod:`probstat.information`, the moment-space inequalities (Pearson,
Bhatia-Davis, Hausdorff), and seeded sampling judged by its standard error.
"""

from __future__ import annotations

import math
import unittest

from probstat.distributions import (
    Beta,
    Exponential,
    MaximumEntropy,
    Normal,
    maximum_entropy,
)
from probstat.information import (
    beta_entropy,
    exponential_entropy,
    gamma_entropy,
    normal_entropy,
    uniform_entropy,
)
from probstat.numerics import digamma

COURSE = {
    "mean": 1.0,
    "variance": 1.0,
    "skewness": 2.0,
    "excess_kurtosis": 7.0,
}


def truncated_exponential_rate(mean: float) -> float:
    """Solve ``mean = 1/l - 1/(e^l - 1)`` on ``[0, 1]`` by bisection.

    The mean of ``f(x) ∝ exp(-l x)`` on ``[0, 1]`` decreases from ``1/2``
    (``l = 0``) to ``0``; 200 bisection steps reach the float resolution.

    References:
        [EQ-DIST-MAXENT-001], [REF-COVER-THOMAS-2006, Chapter 12].

    """

    def mean_of(rate: float) -> float:
        """Return the mean of the truncated exponential law.

        References:
            [EQ-DIST-MAXENT-001]. No new equation.

        """
        return 1.0 / rate - 1.0 / math.expm1(rate)

    low, high = 1e-9, 100.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if mean_of(middle) > mean:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


class ClosedFormTests(unittest.TestCase):
    """The solver reproduces the classical maximum-entropy laws."""

    def assert_converged(self, law: MaximumEntropy) -> None:
        """Check the convergence flag and the relative residual."""
        self.assertTrue(law.diagnostics.converged)
        self.assertLessEqual(law.diagnostics.max_relative_residual, 1e-10)

    def test_bounded_support_without_constraints_is_uniform(self) -> None:
        """No constraint on ``[-1, 3]`` gives ``U(-1, 3)`` and ``log 4`` nats."""
        law = maximum_entropy(-1.0, 3.0)
        self.assert_converged(law)
        self.assertEqual(law.multipliers, ())
        for x in (-1.0, 0.0, 2.5, 3.0):
            self.assertAlmostEqual(law.pdf(x), 0.25, places=14)
        self.assertEqual(law.pdf(3.5), 0.0)
        self.assertAlmostEqual(law.cdf(1.0), 0.5, places=14)
        self.assertAlmostEqual(law.quantile(0.3), 0.2, places=12)
        self.assertAlmostEqual(law.entropy, uniform_entropy(-1.0, 3.0), places=14)
        self.assertAlmostEqual(law.variance, 16.0 / 12.0, places=13)

    def test_half_line_with_a_mean_is_exponential(self) -> None:
        """``E[X] = 3`` on ``[1, inf)`` gives the shifted exponential, rate 1/2."""
        law = maximum_entropy(1.0, mean=3.0)
        self.assert_converged(law)
        self.assertAlmostEqual(law.multipliers[0], 0.5, places=12)
        reference = Exponential(0.5)
        for x in (1.1, 2.0, 5.0, 31.0):
            self.assertAlmostEqual(law.pdf(x) / reference.pdf(x - 1.0), 1.0, places=12)
            self.assertAlmostEqual(law.sf(x) / reference.sf(x - 1.0), 1.0, places=11)
        self.assertAlmostEqual(law.entropy, exponential_entropy(0.5), places=13)
        self.assertAlmostEqual(law.mean, 3.0, places=12)

    def test_real_line_with_mean_and_variance_is_normal(self) -> None:
        """Mean 1, variance 4 on the real line give ``N(1, 2^2)``.

        The exponent ``(x - 1)^2 / 8`` has ``lambda_1 = -1/4``,
        ``lambda_2 = 1/8`` and ``lambda_0 = log(2 sqrt(2 pi)) + 1/8``.
        """
        law = maximum_entropy(mean=1.0, variance=4.0)
        self.assert_converged(law)
        self.assertAlmostEqual(law.multipliers[0], -0.25, places=12)
        self.assertAlmostEqual(law.multipliers[1], 0.125, places=12)
        self.assertAlmostEqual(
            law.log_normalizer,
            math.log(2.0 * math.sqrt(2.0 * math.pi)) + 0.125,
            places=12,
        )
        reference = Normal(1.0, 2.0)
        for x in (-7.0, -1.0, 1.0, 2.5, 9.0):
            self.assertAlmostEqual(law.cdf(x), reference.cdf(x), places=13)
            self.assertAlmostEqual(law.pdf(x) / reference.pdf(x), 1.0, places=12)
        for p in (1e-10, 0.01, 0.5, 0.9, 1.0 - 1e-12):
            self.assertAlmostEqual(law.quantile(p), reference.quantile(p), delta=1e-9)
        self.assertAlmostEqual(law.entropy, normal_entropy(2.0), places=13)
        self.assertEqual(law.support.lower, None)

    def test_bounded_support_with_a_mean_is_truncated_exponential(self) -> None:
        """``E[X] = 0.3`` on ``[0, 1]`` gives ``exp(-l x)`` with the 1-D rate.

        Entropy ``log Z + l mu`` with ``Z = (1 - e^-l) / l``.
        """
        rate = truncated_exponential_rate(0.3)
        law = maximum_entropy(0.0, 1.0, mean=0.3)
        self.assert_converged(law)
        self.assertAlmostEqual(law.multipliers[0], rate, places=10)
        normalizer = -math.expm1(-rate) / rate
        self.assertAlmostEqual(
            law.entropy, math.log(normalizer) + rate * 0.3, places=11
        )
        self.assertAlmostEqual(law.log_normalizer, math.log(normalizer), places=10)

    def test_log_constraint_on_the_half_line_is_gamma(self) -> None:
        """``E[X] = k theta`` and ``E[log X] = psi(k) + log theta`` give Gamma.

        The multipliers are ``1/theta`` and ``1 - k``.
        """
        shape, scale = 2.5, 1.5
        law = maximum_entropy(
            0.0,
            mean=shape * scale,
            constraints=((math.log, digamma(shape) + math.log(scale)),),
            constraint_names=("E[log X]",),
        )
        self.assert_converged(law)
        self.assertEqual(law.constraint_names, ("E[X]", "E[log X]"))
        self.assertAlmostEqual(law.multipliers[0], 1.0 / scale, places=9)
        self.assertAlmostEqual(law.multipliers[1], 1.0 - shape, places=8)
        self.assertAlmostEqual(law.entropy, gamma_entropy(shape, 1.0 / scale), places=9)

    def test_absolute_moment_on_the_real_line_is_laplace(self) -> None:
        """``E[|X|] = 2`` gives Laplace(0, 2), entropy ``1 + log 4``."""
        law = maximum_entropy(constraints=((abs, 2.0),), initial=(1.0,))
        self.assert_converged(law)
        self.assertAlmostEqual(law.multipliers[0], 0.5, places=12)
        self.assertAlmostEqual(law.entropy, 1.0 + math.log(4.0), places=12)
        self.assertAlmostEqual(law.variance, 8.0, places=10)

    def test_entropy_exceeds_another_law_with_the_same_mean(self) -> None:
        """The maximum on ``[0, 1]`` with mean 0.3 beats ``Beta(1.5, 3.5)``."""
        law = maximum_entropy(0.0, 1.0, mean=0.3)
        rival = Beta(1.5, 3.5)
        self.assertAlmostEqual(rival.mean, 0.3, places=14)
        self.assertGreater(law.entropy, beta_entropy(1.5, 3.5))


class CourseExampleTests(unittest.TestCase):
    """The ATHENS example: support ``[-5, 5]``, moments ``(1, 1, 2, 10)``."""

    def test_course_moments_are_feasible_and_reproduced(self) -> None:
        """Pearson ``10 > 2^2 + 1``; the solution reproduces all four moments.

        The quartic multiplier is negative, so the density rises again
        towards both ends of the support: the kurtosis of 10 forces mass
        into the tails, which a bounded support can only hold near its ends.
        """
        law = maximum_entropy(-5.0, 5.0, **COURSE)
        self.assertTrue(law.diagnostics.converged)
        self.assertLessEqual(law.diagnostics.max_relative_residual, 1e-10)
        for residual in law.residuals:
            self.assertLess(abs(residual), 1e-10)
        self.assertAlmostEqual(law.mean, 1.0, places=10)
        self.assertAlmostEqual(law.variance, 1.0, places=10)
        self.assertAlmostEqual(law.skewness, 2.0, places=9)
        self.assertAlmostEqual(law.kurtosis, 7.0, places=8)
        self.assertLess(law.multipliers[3], 0.0)
        self.assertGreater(law.pdf(5.0), law.pdf(4.0))
        self.assertGreater(law.pdf(-5.0), law.pdf(-4.0))
        self.assertGreater(law.pdf(5.0), law.pdf(-5.0))
        self.assertAlmostEqual(law.cdf(5.0), 1.0, places=15)
        self.assertAlmostEqual(law.entropy, law.diagnostics.dual_value, places=10)

    def test_bfgs_reaches_the_newton_solution(self) -> None:
        """BFGS on the same dual gives the same multipliers."""
        newton = maximum_entropy(-5.0, 5.0, **COURSE)
        bfgs = maximum_entropy(-5.0, 5.0, method="bfgs", **COURSE)
        self.assertEqual(bfgs.diagnostics.method, "bfgs")
        for left, right in zip(newton.multipliers, bfgs.multipliers, strict=True):
            self.assertAlmostEqual(left, right, delta=1e-8 * max(1.0, abs(left)))

    def test_course_moments_on_the_real_line_are_attained(self) -> None:
        """On the real line the same moments give a decaying quartic exponent."""
        law = maximum_entropy(**COURSE)
        self.assertTrue(law.diagnostics.converged)
        self.assertGreater(law.multipliers[3], 0.0)
        self.assertAlmostEqual(law.skewness, 2.0, places=8)
        self.assertAlmostEqual(law.kurtosis, 7.0, places=7)


class FeasibilityTests(unittest.TestCase):
    """Infeasible or unattainable constraints raise with a clear reason."""

    def test_pearson_inequality(self) -> None:
        """A kurtosis of 4.5 with skewness 2 violates ``kurtosis > skew^2 + 1``."""
        with self.assertRaisesRegex(ValueError, "Pearson"):
            maximum_entropy(
                -5.0, 5.0, mean=1.0, variance=1.0, skewness=2.0, excess_kurtosis=1.5
            )

    def test_bhatia_davis_bound(self) -> None:
        """A variance of 0.26 on ``[0, 1]`` with mean 0.5 exceeds ``1/4``."""
        with self.assertRaisesRegex(ValueError, "Bhatia-Davis"):
            maximum_entropy(0.0, 1.0, mean=0.5, variance=0.26)

    def test_hausdorff_third_moment(self) -> None:
        """``m = (1, 0.5, 0.3, 0.25)`` is no density's moments on ``[0, 1]``.

        The localizing matrix ``[m_(i+j) - m_(i+j+1)]`` is
        ``[[0.5, 0.2], [0.2, 0.05]]``, whose determinant is ``-0.015``.
        """
        with self.assertRaisesRegex(ValueError, "moment space"):
            maximum_entropy(0.0, 1.0, raw_moments={1: 0.5, 2: 0.3, 3: 0.25})

    def test_invalid_inputs(self) -> None:
        """Means outside the support and incomplete specifications raise."""
        with self.assertRaises(ValueError):
            maximum_entropy(0.0, 1.0, mean=1.5)
        with self.assertRaises(ValueError):
            maximum_entropy(0.0, 1.0, variance=0.1)
        with self.assertRaises(ValueError):
            maximum_entropy(2.0, 1.0)
        with self.assertRaises(ValueError):
            maximum_entropy(0.0, 1.0, mean=0.5, raw_moments={1: 0.4})
        with self.assertRaises(ValueError):
            maximum_entropy(0.0, 1.0, method="simplex")

    def test_unbounded_support_without_constraints(self) -> None:
        """The entropy is unbounded on a half-line without constraints."""
        with self.assertRaisesRegex(ValueError, "unbounded"):
            maximum_entropy(0.0)

    def test_odd_top_moment_on_the_real_line(self) -> None:
        """A third moment the normal law misses cannot be attained.

        Skewness 0 is satisfied by the normal law, whose cubic multiplier is
        zero; skewness 0.5 is not attainable on the real line.
        """
        law = maximum_entropy(mean=1.0, variance=4.0, skewness=0.0)
        self.assertEqual(law.multipliers[2], 0.0)
        self.assertAlmostEqual(law.entropy, normal_entropy(2.0), places=13)
        with self.assertRaisesRegex(ValueError, "not attained"):
            maximum_entropy(mean=1.0, variance=4.0, skewness=0.5)
        with self.assertRaisesRegex(ValueError, "even"):
            maximum_entropy(mean=1.0)

    def test_super_normal_kurtosis_without_skewness_is_not_attained(self) -> None:
        """Mean 0, variance 1, skewness 0, kurtosis 4 on the real line.

        A symmetric exponent ``l_2 x^2 + l_4 x^4`` with ``l_4 > 0`` has a
        kurtosis below 3, so kurtosis 4 needs ``l_4 < 0``, which does not
        decay (Junk 2000).
        """
        with self.assertRaisesRegex(ValueError, "not attained"):
            maximum_entropy(mean=0.0, variance=1.0, skewness=0.0, excess_kurtosis=1.0)

    def test_infeasible_expectation_constraint(self) -> None:
        """``E[X^2] = 2`` on ``[0, 1]`` drives the density onto a point."""
        with self.assertRaisesRegex(ValueError, "did not converge"):
            maximum_entropy(0.0, 1.0, constraints=((lambda x: x * x, 2.0),))

    def test_unbounded_callable_needs_initial_multipliers(self) -> None:
        """A callable constraint alone on the real line needs a start."""
        with self.assertRaisesRegex(ValueError, "initial"):
            maximum_entropy(constraints=((abs, 1.0),))

    def test_linearly_dependent_constraints(self) -> None:
        """Dependent constraints: consistent ones give the same density.

        ``E[X] = 0.3`` with ``E[2X] = 0.6`` determines only
        ``lambda_1 + 2 lambda_2``, which equals the truncated-exponential
        rate; ``E[2X] = 0.7`` contradicts the mean and raises.
        """
        law = maximum_entropy(
            0.0, 1.0, mean=0.3, constraints=((lambda x: 2.0 * x, 0.6),)
        )
        combined = law.multipliers[0] + 2.0 * law.multipliers[1]
        self.assertAlmostEqual(combined, truncated_exponential_rate(0.3), places=10)
        with self.assertRaises(ValueError):
            maximum_entropy(0.0, 1.0, mean=0.3, constraints=((lambda x: 2.0 * x, 0.7),))


class DistributionInterfaceTests(unittest.TestCase):
    """Distribution functions, quantiles, sampling, and conditioning."""

    def test_tail_accuracy_of_sf_and_isf(self) -> None:
        """Upper tails come from the upper-tail sums, not ``1 - F``."""
        law = maximum_entropy(0.0, mean=2.0)
        reference = Exponential(0.5)
        self.assertAlmostEqual(law.sf(30.0) / reference.sf(30.0), 1.0, places=11)
        self.assertAlmostEqual(law.isf(1e-12), reference.isf(1e-12), delta=1e-8)
        for p in (0.05, 0.5, 0.95):
            self.assertAlmostEqual(law.cdf(law.quantile(p)), p, places=12)

    def test_seeded_draws_match_the_moments(self) -> None:
        """400 draws: sample mean within 4 standard errors of the law mean."""
        law = maximum_entropy(-5.0, 5.0, **COURSE)
        draws = law.sample(400, seed=20261116)
        self.assertTrue(all(-5.0 <= x <= 5.0 for x in draws))
        sample_mean = math.fsum(draws) / len(draws)
        self.assertLess(
            abs(sample_mean - law.mean), 4.0 * math.sqrt(law.variance / 400)
        )

    def test_support_far_from_zero_is_well_conditioned(self) -> None:
        """Four moments on ``[100, 110]`` are reproduced to full accuracy."""
        law = maximum_entropy(
            100.0, 110.0, mean=103.0, variance=2.0, skewness=0.5, excess_kurtosis=0.0
        )
        self.assertLessEqual(law.diagnostics.max_relative_residual, 1e-10)
        self.assertAlmostEqual(law.mean, 103.0, places=10)
        self.assertAlmostEqual(law.variance, 2.0, places=10)
        self.assertAlmostEqual(law.skewness, 0.5, places=8)
        self.assertAlmostEqual(law.kurtosis, 0.0, places=8)
        self.assertAlmostEqual(law.residuals[3] / law.targets[3], 0.0, places=13)

    def test_kurtosis_without_skewness_leaves_the_third_moment_free(self) -> None:
        """The fourth central moment is constrained; the mean stays centred."""
        law = maximum_entropy(-3.0, 3.0, mean=0.0, variance=1.0, excess_kurtosis=0.5)
        self.assertEqual(law.constraint_names[-1], "E[(X - mean)^4]")
        self.assertAlmostEqual(law.kurtosis, 0.5, places=10)
        self.assertAlmostEqual(law.skewness, 0.0, places=10)

    def test_expectation_and_log_density(self) -> None:
        """``expect`` and ``log_pdf`` agree with the closed-form normal law."""
        law = maximum_entropy(mean=0.0, variance=1.0)
        self.assertAlmostEqual(law.expect(lambda x: x**4), 3.0, places=11)
        self.assertAlmostEqual(law.log_pdf(1.0), Normal().log_pdf(1.0), places=12)


if __name__ == "__main__":
    unittest.main()
