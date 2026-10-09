"""Tests for generating functions and shared probability-law protocols."""

from __future__ import annotations

import cmath
import math
import unittest

from probstat.distributions import Bernoulli, Normal
from probstat.foundations import (
    ContinuousProbabilityLaw,
    CumulativeDistribution,
    DiscreteProbabilityLaw,
    HasMeanAndVariance,
)
from probstat.probability import (
    FiniteProbabilityMeasure,
    FiniteRandomVariable,
    FiniteSampleSpace,
)


class ProbabilityLawProtocolTests(unittest.TestCase):
    """Verify structural conformance without cross-domain imports."""

    def test_analytic_distributions_conform_to_expected_protocols(self) -> None:
        """Bernoulli is discrete and normal is continuous and cumulative."""
        bernoulli = Bernoulli(0.3)
        normal = Normal()
        self.assertIsInstance(bernoulli, DiscreteProbabilityLaw)
        self.assertIsInstance(bernoulli, CumulativeDistribution)
        self.assertIsInstance(bernoulli, HasMeanAndVariance)
        self.assertIsInstance(normal, ContinuousProbabilityLaw)
        self.assertIsInstance(normal, CumulativeDistribution)
        self.assertIsInstance(normal, HasMeanAndVariance)
        self.assertNotIsInstance(normal, DiscreteProbabilityLaw)

    def test_finite_numeric_variable_conforms_as_discrete_law(self) -> None:
        """An induced finite numeric law shares PMF/CDF/moment interfaces."""
        space = FiniteSampleSpace((0, 1))
        measure = FiniteProbabilityMeasure(space, {0: 0.7, 1: 0.3})
        variable = FiniteRandomVariable.from_function(measure, int)
        self.assertIsInstance(variable, DiscreteProbabilityLaw)
        self.assertIsInstance(variable, HasMeanAndVariance)
        self.assertAlmostEqual(variable.mean, variable.expectation)


class GeneratingFunctionTests(unittest.TestCase):
    """Verify PGF, MGF, characteristic-function identities, and domains."""

    def setUp(self) -> None:
        """Create a finite Bernoulli variable with success mass 0.3."""
        self.space = FiniteSampleSpace((0, 1))
        self.measure = FiniteProbabilityMeasure(
            self.space, {0: 0.7, 1: 0.3}
        )
        self.variable = FiniteRandomVariable.from_function(
            self.measure, int, name="Bernoulli"
        )

    def test_pgf_matches_bernoulli_polynomial(self) -> None:
        """The Bernoulli PGF equals ``(1-p) + pz`` at real and complex points."""
        for point in (0, 1, 2, 1 + 2j):
            expected = 0.7 + 0.3 * point
            self.assertAlmostEqual(
                self.variable.probability_generating_function(point), expected
            )

    def test_mgf_matches_bernoulli_exponential_transform(self) -> None:
        """The Bernoulli MGF equals ``(1-p) + p exp(t)``."""
        for point in (-1.0, 0.0, 0.5, 2.0):
            expected = 0.7 + 0.3 * math.exp(point)
            self.assertAlmostEqual(
                self.variable.moment_generating_function(point), expected
            )

    def test_characteristic_function_identity_and_symmetry(self) -> None:
        """The CF matches its analytic form and conjugate symmetry."""
        point = 0.75
        expected = 0.7 + 0.3 * cmath.exp(1j * point)
        observed = self.variable.characteristic_function(point)
        self.assertAlmostEqual(observed.real, expected.real)
        self.assertAlmostEqual(observed.imag, expected.imag)
        self.assertAlmostEqual(
            self.variable.characteristic_function(-point), observed.conjugate()
        )
        self.assertLessEqual(abs(observed), 1.0 + 1e-15)

    def test_pgf_requires_non_negative_integer_values(self) -> None:
        """Negative, fractional, and categorical values violate the PGF domain."""
        negative = FiniteRandomVariable(
            self.measure, {0: -1, 1: 1}, name="Negative"
        )
        fractional = FiniteRandomVariable(
            self.measure, {0: 0.0, 1: 1.0}, name="Fractional"
        )
        categorical = FiniteRandomVariable(
            self.measure, {0: "no", 1: "yes"}, name="Categorical"
        )
        with self.assertRaises(ValueError):
            negative.probability_generating_function(1)
        with self.assertRaises(TypeError):
            fractional.probability_generating_function(1)
        with self.assertRaises(TypeError):
            categorical.probability_generating_function(1)

    def test_non_finite_arguments_and_machine_overflow_are_explicit(self) -> None:
        """Invalid arguments and unrepresentable transforms never return infinity."""
        with self.assertRaises(ValueError):
            self.variable.probability_generating_function(complex(math.inf, 0.0))
        with self.assertRaises(ValueError):
            self.variable.moment_generating_function(math.nan)
        with self.assertRaises(ValueError):
            self.variable.characteristic_function(math.inf)

        large = FiniteRandomVariable(
            self.measure, {0: 1_000.0, 1: 2_000.0}, name="Large"
        )
        with self.assertRaises(OverflowError):
            large.moment_generating_function(1_000.0)
        with self.assertRaises(OverflowError):
            large.characteristic_function(1e308)


if __name__ == "__main__":
    unittest.main()
