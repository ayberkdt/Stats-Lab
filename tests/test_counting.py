"""Behavioral tests for exact counting primitives."""

from __future__ import annotations

import unittest

from probstat.probability import (
    combinations,
    factorial,
    multiplication_rule,
    permutations,
)


class CountingTests(unittest.TestCase):
    """Verify nominal values, identities, and input boundaries."""

    def test_factorial_boundaries(self) -> None:
        """Factorial follows its zero convention and a known exact value."""
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(6), 720)

    def test_permutation_and_combination_values(self) -> None:
        """Ordered and unordered counts match known finite cases."""
        self.assertEqual(permutations(5, 2), 20)
        self.assertEqual(combinations(5, 2), 10)

    def test_combination_symmetry(self) -> None:
        """Binomial coefficients satisfy C(n,r) = C(n,n-r)."""
        for n in range(10):
            for r in range(n + 1):
                self.assertEqual(combinations(n, r), combinations(n, n - r))

    def test_multiplication_rule(self) -> None:
        """Sequential choice counts multiply exactly."""
        self.assertEqual(multiplication_rule((2, 3, 4)), 24)
        self.assertEqual(multiplication_rule((2, 0, 4)), 0)

    def test_invalid_counts_are_rejected(self) -> None:
        """Negative, reversed, Boolean, and empty domains fail clearly."""
        with self.assertRaises(ValueError):
            factorial(-1)
        with self.assertRaises(ValueError):
            combinations(2, 3)
        with self.assertRaises(TypeError):
            permutations(True, 1)
        with self.assertRaises(ValueError):
            multiplication_rule(())


if __name__ == "__main__":
    unittest.main()

