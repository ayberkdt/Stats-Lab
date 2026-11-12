"""Regression tests for the systems, processes, and risk defect review (ADR-0093).

Every test compares the library with an independent oracle: the closed-form
Riccati fixed point of a scalar random walk, and exact enumeration of a
two-point epistemic law.
"""

from __future__ import annotations

import math
import unittest

from probstat.filtering import LinearGaussianStateSpace, steady_state_kalman
from probstat.systems import SystemStructure, parallel


def _random_walk(scale: float) -> LinearGaussianStateSpace:
    """Return ``x_k = x_(k-1) + w``, ``y = x + v`` with ``q = r = scale``."""
    return LinearGaussianStateSpace(
        ((1.0,),), ((scale,),), ((1.0,),), ((scale,),), (0.0,), ((0.0,),)
    )


class SteadyStateKalmanScaleTest(unittest.TestCase):
    """The Riccati convergence test must not depend on the state's units."""

    def test_fixed_point_matches_closed_form_at_every_scale(self) -> None:
        """The fixed point and gain match the closed form at any scale."""
        # Scalar random walk: P^- = (q + sqrt(q^2 + 4 q r)) / 2, K = P^-/(P^- + r).
        for scale in (1.0, 1e-6, 1e-12, 1e-14, 1e6):
            with self.subTest(scale=scale):
                result = steady_state_kalman(_random_walk(scale))
                exact = 0.5 * (scale + math.sqrt(scale * scale + 4.0 * scale * scale))
                self.assertTrue(result.converged)
                self.assertAlmostEqual(
                    result.predicted_covariance[0][0] / exact, 1.0, delta=1e-10
                )
                self.assertAlmostEqual(
                    result.gain[0][0], exact / (exact + scale), delta=1e-10
                )

    def test_iteration_count_is_unit_invariant(self) -> None:
        """Rescaling every covariance leaves the iteration count unchanged."""
        counts = {
            steady_state_kalman(_random_walk(scale)).iterations
            for scale in (1.0, 1e-12, 1e8)
        }
        self.assertEqual(len(counts), 1)

    def test_zero_covariance_fixed_point_converges(self) -> None:
        """A zero fixed point is reported as converged."""
        model = LinearGaussianStateSpace(
            ((0.5,),), ((0.0,),), ((1.0,),), ((1.0,),), (0.0,), ((0.0,),)
        )
        result = steady_state_kalman(model)
        self.assertTrue(result.converged)
        self.assertEqual(result.predicted_covariance[0][0], 0.0)


class SharedMomentsConventionTest(unittest.TestCase):
    """``shared_moments`` holds raw moments of the shared reliability ``p``."""

    def setUp(self) -> None:
        """Build the structure and the enumerated exact moments."""
        # Two redundant units share one reliability p in {0.7, 0.95}, each
        # with probability 1/2 (an exact two-point epistemic law).
        self.values = (0.7, 0.95)
        self.mean = math.fsum(self.values) / 2.0
        self.variance = math.fsum((p - self.mean) ** 2 for p in self.values) / 2.0
        self.structure = SystemStructure(parallel("x", "y"))
        system = [1.0 - (1.0 - p) ** 2 for p in self.values]
        self.exact_mean = math.fsum(system) / 2.0
        self.exact_variance = (
            math.fsum((r - self.exact_mean) ** 2 for r in system) / 2.0
        )

    def _moments(self, raw: list[float]):  # type: ignore[no-untyped-def]
        """Return the shared-group moments for given raw moments."""
        return self.structure.reliability_moments(
            {"x": self.mean, "y": self.mean},
            {"x": self.variance, "y": self.variance},
            shared={"x": "g", "y": "g"},
            shared_moments={"g": raw},
        )

    def test_reliability_raw_moments_reproduce_enumeration(self) -> None:
        """Raw moments of p give the enumerated mean and variance."""
        raw = [math.fsum(p**k for p in self.values) / 2.0 for k in range(1, 5)]
        result = self._moments(raw)
        self.assertAlmostEqual(result.mean, self.exact_mean, delta=1e-14)
        self.assertAlmostEqual(result.variance, self.exact_variance, delta=1e-14)

    def test_unreliability_raw_moments_are_rejected_with_a_hint(self) -> None:
        """Raw moments of 1 - p are rejected with a message naming p."""
        raw = [
            math.fsum((1.0 - p) ** k for p in self.values) / 2.0 for k in (1, 2, 3, 4)
        ]
        with self.assertRaisesRegex(ValueError, "reliability p"):
            self._moments(raw)

    def test_docstring_names_the_reliability(self) -> None:
        """The docstring asks for moments of the reliability."""
        doc = SystemStructure.reliability_moments.__doc__ or ""
        self.assertIn("E[p^k]", doc)
        self.assertNotIn("E[q^k]", doc)


if __name__ == "__main__":
    unittest.main()
