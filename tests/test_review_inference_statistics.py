"""Regression tests for the inference and statistics defect review (ADR-0091).

Each test pins one defect found by the review and names its independent
oracle in the docstring: exact rational arithmetic (``fractions.Fraction``),
the library's own empirical CDF, the split conformal order statistic, the
union bound ``P(A or B) <= P(A) + P(B)`` on the two one-sided Kolmogorov
events, or a hand count of an exact design.
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

from probstat.conformal import AdaptiveConformalPredictor, conformal_quantile
from probstat.design import d_optimal_design, efficient_rounding, polynomial_basis
from probstat.information import conditional_entropy, entropy, joint_entropy
from probstat.statistics import empirical_cdf, quantile, trimmed_mean


class SampleQuantileTypeOneTests(unittest.TestCase):
    """Type-1 quantile ``x_(j)``, smallest ``j`` with ``j / n >= p``."""

    def test_integer_np_selects_the_exact_order_statistic(self) -> None:
        """``100 * 0.07`` rounds to ``7.000000000000001``; ``Q(0.07) = x_(7)``.

        Oracle: the definition ``Q(p) = inf {x : F_n(x) >= p}`` with the
        library's own ``empirical_cdf``: ``F_n(7) = 0.07 >= p``.
        """
        values = list(range(1, 101))
        self.assertEqual(empirical_cdf(values, 7), 0.07)
        self.assertEqual(quantile(values, 0.07, method=1), 7.0)
        self.assertEqual(quantile(list(range(1, 26)), 0.28, method=1), 7.0)

    def test_matches_exact_rational_ceiling(self) -> None:
        """Every ``p = k/100`` and ``n <= 200`` agrees with ``ceil(n Fraction(p))``.

        Oracle: exact rational arithmetic on the decimal level.
        """
        for n in range(1, 201):
            values = [float(i * i) for i in range(1, n + 1)]
            for k in range(1, 101):
                index = math.ceil(n * Fraction(k, 100))
                self.assertEqual(
                    quantile(values, k / 100, method=1), values[index - 1], (n, k)
                )


class TrimmedMeanTests(unittest.TestCase):
    """Trim count ``g = floor(proportion n)`` taken exactly."""

    def test_floor_does_not_undershoot(self) -> None:
        """``0.29 * 100 = 28.999999999999996``; 29 values are trimmed per end.

        Oracle: ``floor(100 * Fraction(29, 100)) = 29`` and the mean of the
        remaining 42 squares computed directly.
        """
        values = [float(i * i) for i in range(100)]
        kept = values[29:71]
        self.assertAlmostEqual(trimmed_mean(values, 0.29), math.fsum(kept) / 42)

    def test_matches_exact_rational_floor(self) -> None:
        """Every ``proportion = k/100 < 1/2`` and ``n <= 300`` agrees exactly.

        Oracle: exact rational arithmetic on the decimal proportion.
        """
        for n in (100, 180, 200, 300):
            values = [float(i * i) for i in range(n)]
            for k in range(50):
                g = math.floor(n * Fraction(k, 100))
                kept = values[g : n - g]
                self.assertAlmostEqual(
                    trimmed_mean(values, k / 100),
                    math.fsum(kept) / len(kept),
                    msg=str((n, k)),
                )


class AdaptiveConformalRankTests(unittest.TestCase):
    """A frozen ACI level reproduces the split conformal quantile."""

    def test_frozen_level_matches_split_conformal(self) -> None:
        """``alpha = 0.7``, nine scores: rank ``ceil(0.3 * 10) = 3``, not 4.

        Oracle: ``conformal_quantile`` (exact rational rank) and the hand
        value ``k = 3``; the binary product ``(1 - 0.7) * 10`` is
        ``3.0000000000000004``.
        """
        scores = [float(i) for i in range(1, 10)]
        predictor = AdaptiveConformalPredictor(
            alpha=0.7, gamma=0.0, calibration_scores=scores
        )
        self.assertEqual(predictor.radius(), 3.0)
        self.assertEqual(predictor.radius(), conformal_quantile(scores, 0.7))

    def test_frozen_level_agrees_over_a_grid(self) -> None:
        """Agreement for every ``alpha = k/20`` and history size up to 120.

        Oracle: ``conformal_quantile`` on the same scores.
        """
        for k in range(1, 20):
            alpha = k / 20
            for size in range(1, 121):
                scores = [float(i) for i in range(size)]
                predictor = AdaptiveConformalPredictor(
                    alpha=alpha, gamma=0.0, calibration_scores=scores
                )
                self.assertEqual(
                    predictor.radius(), conformal_quantile(scores, alpha), (k, size)
                )


class DiscreteEntropySignTests(unittest.TestCase):
    """A point mass has entropy ``+0.0``, not ``-0.0``."""

    def test_point_mass_entropies_are_positive_zero(self) -> None:
        """``-(1 log 1)`` is ``-0.0`` in floating point; the result is ``+0.0``.

        Oracle: ``H = 0`` for a degenerate law [Cover and Thomas, Section
        2.1]; the sign bit is checked with ``math.copysign``.
        """
        for value in (
            entropy([1.0]),
            entropy([0.0, 1.0], unit="bits"),
            joint_entropy([[1.0]]),
            conditional_entropy([[0.5, 0.0], [0.0, 0.5]]),
        ):
            self.assertEqual(value, 0.0)
            self.assertEqual(math.copysign(1.0, value), 1.0)


class EfficientRoundingGuidanceTests(unittest.TestCase):
    """The documented condense-then-round workflow gives the textbook design."""

    def test_condensed_design_rounds_onto_the_optimal_support(self) -> None:
        """Seven runs of the quadratic D-optimal design land on ``{-1, 0, 1}``.

        Oracle: the D-optimal quadratic design on ``[-1, 1]`` puts weight
        1/3 on ``-1, 0, 1`` [Atkinson, Donev and Tobias 2007, Chapter 11];
        efficient rounding of thirds to seven runs gives counts ``3, 2, 2``
        up to the order of the ties. Without condensing, residual grid
        weights each take a run.
        """
        candidates = [i / 20 - 1 for i in range(41)]
        basis = polynomial_basis(2)
        result = d_optimal_design(candidates, basis)
        raw = efficient_rounding(result.design, 7)
        self.assertGreater(len(raw.points), 3)
        exact = efficient_rounding(result.design.condensed(min_weight=1e-3), 7)
        self.assertEqual(
            tuple(round(point[0], 12) for point in exact.points), (-1.0, 0.0, 1.0)
        )
        self.assertEqual(sorted(exact.counts), [2, 2, 3])


if __name__ == "__main__":
    unittest.main()
