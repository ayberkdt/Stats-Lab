"""Tests for reusable distribution constraints and support objects."""

from __future__ import annotations

import math
import unittest

from probstat.distributions import FiniteSupport, IntegerSupport, RealSupport
from probstat.distributions.constraints import (
    IntegerIntervalConstraint,
    ProbabilitySimplexConstraint,
    RealIntervalConstraint,
)


class ConstraintTests(unittest.TestCase):
    """Verify parameter constraints reject invalid mathematical domains."""

    def test_real_interval_endpoints_and_validation(self) -> None:
        """Open and closed endpoints have explicit membership semantics."""
        interval = RealIntervalConstraint(0.0, 1.0, lower_closed=False)
        self.assertFalse(interval.contains(0.0))
        self.assertTrue(interval.contains(0.5))
        self.assertEqual(interval.validate(1, name="x"), 1.0)
        with self.assertRaises(ValueError):
            interval.validate(0.0, name="x")
        with self.assertRaises(ValueError):
            RealIntervalConstraint(1.0, 0.0)
        with self.assertRaises(ValueError):
            RealIntervalConstraint(math.nan, 1.0)

    def test_integer_interval_rejects_booleans_and_reversed_bounds(self) -> None:
        """Integer constraints never silently treat booleans as counts."""
        interval = IntegerIntervalConstraint(0, 3)
        self.assertTrue(interval.contains(2))
        self.assertFalse(interval.contains(True))
        with self.assertRaises(TypeError):
            interval.validate(1.0, name="n")
        with self.assertRaises(ValueError):
            IntegerIntervalConstraint(3, 2)

    def test_simplex_normalizes_only_near_unit_totals(self) -> None:
        """Probability vectors must be non-empty, valid, and sum near one."""
        simplex = ProbabilitySimplexConstraint(tolerance=1e-12)
        values = simplex.validate((0.1, 0.2, 0.7), name="p")
        self.assertAlmostEqual(math.fsum(values), 1.0)
        with self.assertRaises(ValueError):
            simplex.validate((), name="p")
        with self.assertRaises(ValueError):
            simplex.validate((0.2, 0.2), name="p")


class SupportTests(unittest.TestCase):
    """Verify finite, integer, and real support descriptors."""

    def test_integer_support_membership_and_enumeration(self) -> None:
        """Bounded support enumerates; unbounded support does not pretend to."""
        bounded = IntegerSupport(0, 3)
        self.assertEqual(bounded.finite_values(), (0, 1, 2, 3))
        self.assertTrue(bounded.contains(3))
        self.assertFalse(bounded.contains(4))
        unbounded = IntegerSupport(0, None)
        self.assertTrue(unbounded.contains(10_000))
        with self.assertRaises(ValueError):
            unbounded.finite_values()

    def test_finite_support_preserves_order_and_uniqueness(self) -> None:
        """Finite support keeps declared display order and rejects duplicates."""
        support = FiniteSupport(("red", "green", "blue"))
        self.assertEqual(tuple(support), ("red", "green", "blue"))
        self.assertTrue(support.contains("green"))
        self.assertFalse(support.contains([]))
        with self.assertRaises(ValueError):
            FiniteSupport(("red", "red"))

    def test_real_support_validates_bounds_and_openness(self) -> None:
        """Real support validates its interval before serving membership tests."""
        support = RealSupport(0.0, 1.0, lower_closed=True)
        self.assertTrue(support.contains(0.0))
        self.assertFalse(support.contains(1.0))
        self.assertFalse(support.contains(math.inf))
        with self.assertRaises(ValueError):
            RealSupport(2.0, 1.0)
        with self.assertRaises(ValueError):
            RealSupport(1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
