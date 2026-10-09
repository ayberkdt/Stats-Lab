"""Behavioral and numerical tests for foundational discrete distributions."""

from __future__ import annotations

import math
import statistics
import unittest

from probstat.distributions import Binomial, Categorical, Poisson
from probstat.distributions._special import (
    regularized_gamma_p,
    regularized_gamma_q,
)
from probstat.foundations import (
    DiscreteProbabilityLaw,
    LogProbabilityMassFunction,
    ProbabilityMassFunction,
)


class BinomialTests(unittest.TestCase):
    """Verify binomial mass, moments, boundaries, and sampling."""

    def test_reference_mass_normalization_and_moments(self) -> None:
        """The symmetric ten-trial law matches its closed-form identities."""
        law = Binomial(10, 0.5)
        self.assertAlmostEqual(law.pmf(5), 252.0 / 1024.0)
        self.assertAlmostEqual(math.fsum(law.pmf(k) for k in range(11)), 1.0)
        self.assertAlmostEqual(law.cdf(4), 0.376953125)
        self.assertEqual(law.mean, 5.0)
        self.assertEqual(law.variance, 2.5)

    def test_boundary_parameters_and_log_mass(self) -> None:
        """Degenerate parameters and outside-support mass remain well defined."""
        self.assertEqual(Binomial(4, 0.0).pmf(0), 1.0)
        self.assertEqual(Binomial(4, 1.0).pmf(4), 1.0)
        self.assertEqual(Binomial(4, 0.5).log_pmf(5), -math.inf)
        self.assertAlmostEqual(
            math.exp(Binomial(4, 0.3).log_pmf(2)), Binomial(4, 0.3).pmf(2)
        )
        self.assertEqual(Binomial(0, 0.2).support.finite_values(), (0,))

    def test_sampling_is_reproducible(self) -> None:
        """Call-local seeds reproduce binomial samples."""
        law = Binomial(8, 0.3)
        self.assertEqual(law.sample(30, seed=7), law.sample(30, seed=7))

    def test_cdf_selects_small_tail_by_location_not_support_midpoint(self) -> None:
        """Highly skewed laws retain small lower and upper tail probabilities."""
        lower_tail = Binomial(100, 0.99).cdf(80)
        upper_complement = Binomial(100, 0.01).cdf(20)
        self.assertGreater(lower_tail, 0.0)
        self.assertLess(lower_tail, 1e-5)
        self.assertGreater(upper_complement, 1.0 - 1e-10)
        self.assertLessEqual(upper_complement, 1.0)


class CategoricalTests(unittest.TestCase):
    """Verify unordered categorical laws preserve label semantics."""

    def test_mass_events_modes_and_entropy(self) -> None:
        """Categorical summaries follow event and Shannon definitions."""
        law = Categorical({"red": 0.5, "green": 0.25, "blue": 0.25})
        self.assertEqual(law.pmf("absent"), 0.0)
        self.assertAlmostEqual(law.probability_of(("red", "red", "blue")), 0.75)
        self.assertEqual(law.mode, ("red",))
        self.assertAlmostEqual(law.entropy, 1.0397207708399179)
        with self.assertRaises(TypeError):
            law.probabilities["red"] = 0.1  # type: ignore[index]

    def test_zero_mass_ties_and_sampling(self) -> None:
        """Zero masses use log-zero convention and tied modes keep order."""
        law = Categorical({"a": 0.5, "b": 0.5, "c": 0.0})
        self.assertEqual(law.log_pmf("c"), -math.inf)
        self.assertEqual(law.mode, ("a", "b"))
        self.assertEqual(law.sample(20, seed=17), law.sample(20, seed=17))

    def test_protocols_do_not_invent_order_or_numeric_moments(self) -> None:
        """Categorical laws expose mass but not ordered numeric-law structure."""
        law = Categorical({"yes": 0.6, "no": 0.4})
        self.assertIsInstance(law, ProbabilityMassFunction)
        self.assertIsInstance(law, LogProbabilityMassFunction)
        self.assertNotIsInstance(law, DiscreteProbabilityLaw)


class PoissonTests(unittest.TestCase):
    """Verify Poisson mass, gamma tails, moments, and exact samplers."""

    def test_reference_values_tails_and_degenerate_rate(self) -> None:
        """Known masses and incomplete-gamma tails agree with finite sums."""
        law = Poisson(3.0)
        expected_cdf = math.exp(-3.0) * (1.0 + 3.0 + 4.5)
        self.assertAlmostEqual(law.pmf(2), 4.5 * math.exp(-3.0))
        self.assertAlmostEqual(law.cdf(2), expected_cdf)
        self.assertAlmostEqual(law.cdf(2) + law.sf(2), 1.0)
        self.assertEqual(law.mean, 3.0)
        self.assertEqual(law.variance, 3.0)
        self.assertEqual(Poisson(0.0).sample(5, seed=2), (0, 0, 0, 0, 0))

    def test_large_rate_cdf_and_ptrs_sampling(self) -> None:
        """Large-rate evaluation is monotone and transformed rejection is sound."""
        law = Poisson(50.0)
        cdf_values = tuple(law.cdf(k) for k in (35, 50, 65))
        self.assertEqual(cdf_values, tuple(sorted(cdf_values)))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in cdf_values))
        sample = law.sample(5_000, seed=2026)
        self.assertAlmostEqual(statistics.fmean(sample), 50.0, delta=0.6)
        self.assertEqual(sample, law.sample(5_000, seed=2026))

    def test_invalid_parameters_and_arguments_are_rejected(self) -> None:
        """Negative rates and non-integral mass arguments fail explicitly."""
        with self.assertRaises(ValueError):
            Poisson(-1.0)
        with self.assertRaises(TypeError):
            Poisson(2.0).pmf(1.5)  # type: ignore[arg-type]


class RegularizedGammaTests(unittest.TestCase):
    """Verify special-function branches and complement identities."""

    def test_exponential_identity_and_complements(self) -> None:
        """Shape one recovers exponential tails and P plus Q equals one."""
        for point in (0.0, 0.1, 1.0, 5.0, 20.0):
            lower = regularized_gamma_p(1.0, point)
            upper = regularized_gamma_q(1.0, point)
            self.assertAlmostEqual(lower, 1.0 - math.exp(-point), places=14)
            self.assertAlmostEqual(upper, math.exp(-point), places=14)
            self.assertAlmostEqual(lower + upper, 1.0, places=14)
        for shape, point in ((0.5, 0.2), (2.5, 3.0), (20.0, 8.0)):
            self.assertAlmostEqual(
                regularized_gamma_p(shape, point)
                + regularized_gamma_q(shape, point),
                1.0,
                places=13,
            )


if __name__ == "__main__":
    unittest.main()
