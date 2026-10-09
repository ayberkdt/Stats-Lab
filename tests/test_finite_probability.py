"""Behavioral and algebraic tests for finite probability spaces."""

from __future__ import annotations

import math
import unittest

from probstat.probability import FiniteProbabilityMeasure, FiniteSampleSpace


class FiniteSampleSpaceTests(unittest.TestCase):
    """Verify explicit ownership and immutable event-algebra behavior."""

    def test_sample_space_validation(self) -> None:
        """Empty, repeated, and unhashable outcome declarations fail."""
        with self.assertRaises(ValueError):
            FiniteSampleSpace(())
        with self.assertRaises(ValueError):
            FiniteSampleSpace((1, 1))
        with self.assertRaises(TypeError):
            FiniteSampleSpace(([1],))

    def test_event_algebra_and_de_morgan_identity(self) -> None:
        """Finite event operations satisfy complements and De Morgan's law."""
        space = FiniteSampleSpace((1, 2, 3, 4, 5, 6))
        even = space.event((2, 4, 6))
        high = space.event((4, 5, 6))

        self.assertEqual(even.intersection(high).members, frozenset((4, 6)))
        self.assertEqual(even.union(high).members, frozenset((2, 4, 5, 6)))
        self.assertEqual(even.difference(high).members, frozenset((2,)))
        self.assertEqual(even.complement().complement(), even)
        self.assertEqual(
            even.union(high).complement(),
            even.complement().intersection(high.complement()),
        )
        self.assertTrue(space.empty_event.is_subset_of(even))

    def test_distinct_spaces_do_not_share_event_ownership(self) -> None:
        """Equal labels in different experiments do not make events compatible."""
        first_space = FiniteSampleSpace((0, 1))
        second_space = FiniteSampleSpace((0, 1))
        with self.assertRaises(ValueError):
            first_space.singleton(1).union(second_space.singleton(1))


class FiniteProbabilityMeasureTests(unittest.TestCase):
    """Verify finite measures and derived probability equations."""

    def test_uniform_die_axioms_and_derived_rules(self) -> None:
        """A fair die satisfies normalization, complements, and inclusion-exclusion."""
        space = FiniteSampleSpace((1, 2, 3, 4, 5, 6))
        measure = FiniteProbabilityMeasure.uniform(space)
        even = space.event((2, 4, 6))
        high = space.event((4, 5, 6))

        self.assertAlmostEqual(measure.probability(space.empty_event), 0.0)
        self.assertAlmostEqual(measure.probability(space.certain_event), 1.0)
        self.assertAlmostEqual(
            measure.probability(even.complement()),
            1.0 - measure.probability(even),
        )
        self.assertAlmostEqual(
            measure.probability(even.union(high)),
            measure.probability(even)
            + measure.probability(high)
            - measure.probability(even.intersection(high)),
        )

    def test_weight_contract_and_read_only_view(self) -> None:
        """Measures require complete normalized weights and expose no mutation."""
        space = FiniteSampleSpace(("a", "b"))
        measure = FiniteProbabilityMeasure(space, {"a": 0.25, "b": 0.75})
        self.assertAlmostEqual(measure.probability(space.singleton("a")), 0.25)
        with self.assertRaises(TypeError):
            measure.weights["a"] = 0.5  # type: ignore[index]
        with self.assertRaises(ValueError):
            FiniteProbabilityMeasure(space, {"a": 1.0})
        with self.assertRaises(ValueError):
            FiniteProbabilityMeasure(space, {"a": 0.2, "b": 0.2})
        with self.assertRaises(ValueError):
            FiniteProbabilityMeasure(space, {"a": -0.1, "b": 1.1})

    def test_conditional_probability_and_independence(self) -> None:
        """Two coordinates of a uniform binary product space are independent."""
        space = FiniteSampleSpace(("HH", "HT", "TH", "TT"))
        measure = FiniteProbabilityMeasure.uniform(space)
        first_head = space.event(("HH", "HT"))
        second_head = space.event(("HH", "TH"))
        disjoint_positive = space.event(("TT",))

        self.assertAlmostEqual(
            measure.conditional_probability(first_head, second_head), 0.5
        )
        self.assertTrue(measure.are_independent(first_head, second_head))
        self.assertFalse(measure.are_independent(first_head, disjoint_positive))
        with self.assertRaises(ValueError):
            measure.conditional_probability(first_head, space.empty_event)

    def test_total_probability_and_bayes_rule(self) -> None:
        """A diagnostic-test model reconstructs evidence and posterior mass."""
        outcomes = ("D+", "D-", "H+", "H-")
        space = FiniteSampleSpace(outcomes)
        measure = FiniteProbabilityMeasure(
            space,
            {"D+": 0.0099, "D-": 0.0001, "H+": 0.0495, "H-": 0.9405},
        )
        disease = space.event(("D+", "D-"))
        healthy = space.event(("H+", "H-"))
        positive = space.event(("D+", "H+"))

        reconstructed = measure.total_probability(positive, (disease, healthy))
        self.assertAlmostEqual(reconstructed, measure.probability(positive))
        self.assertAlmostEqual(measure.bayes_probability(disease, positive), 1.0 / 6.0)

    def test_invalid_partitions_are_rejected(self) -> None:
        """Partitions must be non-empty, disjoint, exhaustive, and positive."""
        space = FiniteSampleSpace((1, 2, 3))
        measure = FiniteProbabilityMeasure(space, {1: 0.5, 2: 0.5, 3: 0.0})
        target = space.singleton(1)
        with self.assertRaises(ValueError):
            measure.total_probability(target, ())
        with self.assertRaises(ValueError):
            measure.total_probability(
                target, (space.event((1, 2)), space.event((2, 3)))
            )
        with self.assertRaises(ValueError):
            measure.total_probability(target, (space.event((1, 2)),))
        with self.assertRaises(ValueError):
            measure.total_probability(
                target, (space.event((1, 2)), space.singleton(3))
            )

    def test_finite_expectation(self) -> None:
        """Expected value of a fair die equals its probability-weighted center."""
        space = FiniteSampleSpace((1, 2, 3, 4, 5, 6))
        measure = FiniteProbabilityMeasure.uniform(space)
        self.assertAlmostEqual(measure.expectation(float), 3.5)
        with self.assertRaises(ValueError):
            measure.expectation(lambda _: math.inf)


if __name__ == "__main__":
    unittest.main()

