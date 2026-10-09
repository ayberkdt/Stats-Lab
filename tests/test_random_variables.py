"""Behavioral tests for finite random variables and joint laws."""

from __future__ import annotations

import math
import unittest

from probstat.probability import (
    FiniteProbabilityMeasure,
    FiniteRandomVariable,
    FiniteSampleSpace,
)


class FiniteRandomVariableTests(unittest.TestCase):
    """Verify induced laws, transformations, numeric domains, and moments."""

    def setUp(self) -> None:
        """Create a fair die measure shared by random-variable tests."""
        self.space = FiniteSampleSpace((1, 2, 3, 4, 5, 6))
        self.measure = FiniteProbabilityMeasure.uniform(self.space)
        self.die = FiniteRandomVariable.from_function(
            self.measure, float, name="Die"
        )

    def test_constructor_contract_and_immutability(self) -> None:
        """Mappings must be complete, hashable-valued, named, and read-only."""
        with self.assertRaises(ValueError):
            FiniteRandomVariable(self.measure, {1: "one"})
        with self.assertRaises(TypeError):
            FiniteRandomVariable.from_function(
                self.measure, lambda outcome: [outcome]
            )
        with self.assertRaises(ValueError):
            FiniteRandomVariable.from_function(self.measure, str, name="  ")
        with self.assertRaises(TypeError):
            self.die.values[1] = 99.0  # type: ignore[index]

    def test_induced_pmf_preimages_and_support(self) -> None:
        """A many-to-one parity mapping aggregates source mass correctly."""
        parity = FiniteRandomVariable.from_function(
            self.measure,
            lambda outcome: "even" if outcome % 2 == 0 else "odd",
            name="Parity",
        )
        self.assertEqual(parity.preimage("even"), self.space.event((2, 4, 6)))
        self.assertAlmostEqual(parity.pmf("even"), 0.5)
        self.assertAlmostEqual(math.fsum(parity.distribution.values()), 1.0)
        self.assertEqual(set(parity.support), {"odd", "even"})

    def test_structural_and_positive_support_are_distinct(self) -> None:
        """Values reached only through zero-mass outcomes remain structural."""
        space = FiniteSampleSpace(("possible", "null"))
        measure = FiniteProbabilityMeasure(
            space, {"possible": 1.0, "null": 0.0}
        )
        variable = FiniteRandomVariable(
            measure, {"possible": "seen", "null": "unseen"}
        )
        self.assertEqual(variable.support, ("seen", "unseen"))
        self.assertEqual(variable.positive_support, ("seen",))

    def test_cdf_expectation_and_moments(self) -> None:
        """A fair die matches analytic CDF, mean, variance, and raw moments."""
        self.assertAlmostEqual(self.die.cdf(3.0), 0.5)
        self.assertAlmostEqual(self.die.expectation, 3.5)
        self.assertAlmostEqual(self.die.raw_moment(2), 91.0 / 6.0)
        self.assertAlmostEqual(self.die.variance, 35.0 / 12.0)
        self.assertAlmostEqual(self.die.central_moment(3), 0.0)
        self.assertAlmostEqual(
            self.die.standard_deviation, math.sqrt(35.0 / 12.0)
        )

    def test_transformation_reuses_measure_and_pushes_mass_forward(self) -> None:
        """Squaring a symmetric variable preserves ownership and merges values."""
        space = FiniteSampleSpace((-1, 0, 1))
        measure = FiniteProbabilityMeasure.uniform(space)
        variable = FiniteRandomVariable.from_function(measure, float, name="X")
        squared = variable.transform(lambda value: value**2, name="X squared")
        self.assertIs(squared.measure, measure)
        self.assertAlmostEqual(squared.pmf(0.0), 1.0 / 3.0)
        self.assertAlmostEqual(squared.pmf(1.0), 2.0 / 3.0)

    def test_numeric_methods_reject_categorical_or_non_finite_values(self) -> None:
        """Numeric operations fail instead of inventing categorical encodings."""
        categorical = FiniteRandomVariable.from_function(
            self.measure, lambda outcome: "even" if outcome % 2 == 0 else "odd"
        )
        with self.assertRaises(TypeError):
            _ = categorical.expectation
        with self.assertRaises(ValueError):
            self.die.cdf(math.inf)
        with self.assertRaises(ValueError):
            self.die.raw_moment(-1)

    def test_event_where_requires_boolean_predicate(self) -> None:
        """Predicate events accept explicit truth values rather than coercion."""
        high = self.die.event_where(lambda value: value >= 4.0)
        self.assertEqual(high, self.space.event((4, 5, 6)))
        with self.assertRaises(TypeError):
            self.die.event_where(lambda _: 1)  # type: ignore[arg-type,return-value]


class FiniteJointDistributionTests(unittest.TestCase):
    """Verify joint PMFs, marginals, conditioning, and dependence summaries."""

    def test_independent_coin_coordinates_factorize(self) -> None:
        """Coordinates of a uniform two-coin product space are independent."""
        space = FiniteSampleSpace(("HH", "HT", "TH", "TT"))
        measure = FiniteProbabilityMeasure.uniform(space)
        first = FiniteRandomVariable.from_function(
            measure, lambda outcome: outcome[0], name="First"
        )
        second = FiniteRandomVariable.from_function(
            measure, lambda outcome: outcome[1], name="Second"
        )
        joint = first.joint_with(second)

        self.assertAlmostEqual(joint.joint_pmf("H", "H"), 0.25)
        self.assertEqual(dict(joint.first_marginal), {"H": 0.5, "T": 0.5})
        self.assertEqual(dict(joint.second_marginal), {"H": 0.5, "T": 0.5})
        self.assertAlmostEqual(
            joint.conditional_pmf_first_given_second("H", given_second="T"),
            0.5,
        )
        self.assertTrue(joint.are_independent())
        self.assertAlmostEqual(math.fsum(joint.distribution.values()), 1.0)

    def test_covariance_correlation_and_nonlinear_dependence(self) -> None:
        """Zero covariance does not imply independence for nonlinear mappings."""
        space = FiniteSampleSpace((-1, 0, 1))
        measure = FiniteProbabilityMeasure.uniform(space)
        x = FiniteRandomVariable.from_function(measure, float, name="X")
        y = x.transform(lambda value: value**2, name="Y")
        nonlinear = x.joint_with(y)

        self.assertAlmostEqual(nonlinear.covariance, 0.0)
        self.assertAlmostEqual(nonlinear.correlation, 0.0)
        self.assertFalse(nonlinear.are_independent())

        affine = x.joint_with(x.transform(lambda value: 2.0 * value + 3.0))
        self.assertAlmostEqual(affine.correlation, 1.0)

    def test_joint_law_requires_exact_shared_measure(self) -> None:
        """Equal-looking measures remain distinct probabilistic experiments."""
        space = FiniteSampleSpace((0, 1))
        first_measure = FiniteProbabilityMeasure.uniform(space)
        second_measure = FiniteProbabilityMeasure.uniform(space)
        first = FiniteRandomVariable.from_function(first_measure, int)
        second = FiniteRandomVariable.from_function(second_measure, int)
        with self.assertRaises(ValueError):
            first.joint_with(second)

    def test_zero_mass_conditioning_and_zero_variance_correlation_fail(self) -> None:
        """Undefined conditional and correlation domains raise clear errors."""
        space = FiniteSampleSpace(("a", "b"))
        measure = FiniteProbabilityMeasure(space, {"a": 1.0, "b": 0.0})
        first = FiniteRandomVariable(measure, {"a": 1.0, "b": 2.0})
        second = FiniteRandomVariable(measure, {"a": "seen", "b": "null"})
        joint = first.joint_with(second)
        with self.assertRaises(ValueError):
            joint.conditional_pmf_first_given_second(2.0, given_second="null")

        constant = FiniteRandomVariable.from_function(measure, lambda _: 1.0)
        with self.assertRaises(ValueError):
            _ = first.joint_with(constant).correlation


if __name__ == "__main__":
    unittest.main()
