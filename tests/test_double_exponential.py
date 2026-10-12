"""Tests for the tanh-sinh rule's convergence reporting."""

from __future__ import annotations

import math
import unittest
import warnings
from fractions import Fraction

from probstat.distributions.normal import Normal
from probstat.information.wasserstein import quantile_wasserstein
from probstat.numerics import tanh_sinh
from probstat.numerics.double_exponential import (
    sign_change_points,
    tanh_sinh_pieces,
    warn_if_unreliable,
)
from probstat.validation.shift import distribution_wasserstein_distance


class TanhSinhReportingTests(unittest.TestCase):
    """Converged estimates are flagged; hard cases are not silently trusted."""

    def test_end_singularity_converges(self) -> None:
        """int_0^1 u^-0.9 du = 10 to rounding, flagged converged."""
        result = tanh_sinh(lambda u, uc: u**-0.9, 0.0, 1.0)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.value, 10.0, places=10)
        self.assertEqual(result.skipped_nodes, 0)

    def test_failing_tail_evaluations_are_counted(self) -> None:
        """Evaluations that fail very close to 0 are counted, not hidden."""

        def fragile(u: float, uc: float) -> float:
            """Return u^-0.95 but fail below 1e-16."""
            if u < 1e-16:
                raise ValueError("underflow")
            return u**-0.95

        result = tanh_sinh(fragile, 0.0, 1.0)
        self.assertGreater(result.skipped_nodes, 0)

    def test_nonconvergence_warns(self) -> None:
        """u^-0.999 carries mass below 1e-300: not converged, and it warns."""
        result = tanh_sinh(lambda u, uc: u**-0.999, 0.0, 1.0)
        self.assertFalse(result.converged)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = warn_if_unreliable(result, "test integral", stacklevel=2)
        self.assertEqual(value, result.value)
        self.assertTrue(any(w.category is RuntimeWarning for w in caught))
        self.assertTrue(math.isfinite(value))

    def test_truncation_keeps_slow_end_singularity_unconverged(self) -> None:
        """``int_0^1 u^-0.965 = 1/0.035``: true relative error 3.2e-11.

        That exceeds the default tolerance 1e-11, yet the level difference
        alone reported convergence; the truncation term now refuses it.
        """
        exact = 1.0 / 0.035
        result = tanh_sinh(lambda u, uc: u**-0.965, 0.0, 1.0)
        self.assertFalse(result.converged)
        self.assertGreaterEqual(result.error_estimate, abs(result.value - exact))


class TanhSinhPiecesTests(unittest.TestCase):
    """Piecewise integration judged on the sum, with exact complements."""

    def test_tiny_piece_does_not_spoil_converged_sum(self) -> None:
        """``|u - (u + 0.01 (u - .501)(u - .509))|`` over ``(0, 1)``.

        Exact value from the rational antiderivative (computed in the test).
        The piece ``(0.5, 0.501)`` has value ``4e-11`` and cannot reach its
        own relative tolerance under cancellation; the old per-piece rule
        warned although the total was accurate to ``2e-15``.
        """
        a, b, c = Fraction("0.501"), Fraction("0.509"), Fraction("0.01")

        def antiderivative(u: Fraction) -> Fraction:
            """Return ``int_0^u c (s - a)(s - b) ds`` exactly."""
            return c * (u**3 / 3 - (a + b) * u**2 / 2 + a * b * u)

        exact = float(
            antiderivative(a)
            - (antiderivative(b) - antiderivative(a))
            + (antiderivative(Fraction(1)) - antiderivative(b))
        )

        def gap(u: float, uc: float) -> float:
            """Return the signed difference of the two quantile functions."""
            return u - (u + 0.01 * (u - 0.501) * (u - 0.509))

        breaks = (*sign_change_points(gap), 0.5)
        result = tanh_sinh_pieces(
            lambda u, uc: abs(gap(u, uc)), breaks, absolute_tolerance=1e-30
        )
        self.assertTrue(result.converged)
        self.assertLess(abs(result.value - exact), 1e-14 * exact)

    def test_upper_tail_kink_sits_on_a_piece_boundary(self) -> None:
        """``N(0, 1)`` vs ``N(-8, 2)``: quantiles cross at ``1 - Phi(-8)``.

        ``Phi(-8) = 6.220960574271784e-16`` (mpmath); as a float ``u`` the
        crossing rounds to ``1 - 6.66e-16``, inside a piece, and ``W_1``
        warned twice although exact (``W_1 = 8`` to double precision).
        """
        first, second = Normal(0.0, 1.0), Normal(-8.0, 2.0)

        def signed(u: float, uc: float) -> float:
            """Return ``Q_0 - Q_1`` using ``isf`` in the upper half."""
            if u > 0.5:
                return first.isf(uc) - second.isf(uc)
            return first.quantile(u) - second.quantile(u)

        pairs = sign_change_points(signed, complements=True)
        self.assertEqual(len(pairs), 1)
        complement = pairs[0][1]  # type: ignore[index]
        self.assertLess(abs(complement / 6.220960574271784e-16 - 1.0), 1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w1 = quantile_wasserstein(first, second, p=1.0)
            estimate = distribution_wasserstein_distance(first, second, order=1.0)
        self.assertLess(abs(w1 - 8.0), 1e-14 * 8.0)
        self.assertLess(abs(estimate.distance - 8.0), 1e-14 * 8.0)

    def test_step_quantiles_leaving_zero_are_breakpoints(self) -> None:
        """Bernoulli(0.3) vs Bernoulli(0.6): ``W_1 = 0.3``, ``W_2 = sqrt(0.3)``.

        ``Q_0 - Q_1`` is zero, then ``-1`` on ``(0.4, 0.7]``, then zero; it
        never changes sign, so the jumps were missed (relative error 3.6e-5
        with two warnings).
        """

        class Step:
            """Quantile function of a Bernoulli law."""

            def __init__(self, p: float) -> None:
                """Store the success probability."""
                self.p = p

            def quantile(self, u: float) -> float:
                """Return the Bernoulli quantile."""
                return 0.0 if u <= 1.0 - self.p else 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w1 = quantile_wasserstein(Step(0.3), Step(0.6), p=1.0)
            w2 = quantile_wasserstein(Step(0.3), Step(0.6), p=2.0)
        self.assertLess(abs(w1 - 0.3), 1e-14)
        self.assertLess(abs(w2 - math.sqrt(0.3)), 1e-14)
        points = sign_change_points(
            lambda u, uc: Step(0.3).quantile(u) - Step(0.6).quantile(u)
        )
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0], 0.4, delta=1e-15)  # type: ignore[arg-type]
        self.assertAlmostEqual(points[1], 0.7, delta=1e-15)  # type: ignore[arg-type]

    def test_complement_limits_are_validated(self) -> None:
        """Inconsistent complements and reversed pieces are rejected."""
        with self.assertRaises(ValueError):
            tanh_sinh(lambda u, uc: 1.0, 0.2, 0.4, upper_complement=0.1)
        with self.assertRaises(ValueError):
            tanh_sinh(
                lambda u, uc: 1.0,
                1.0,
                1.0,
                lower_complement=1e-17,
                upper_complement=2e-17,
            )
        tiny = tanh_sinh(
            lambda u, uc: 1.0, 1.0, 1.0, lower_complement=2e-17, upper_complement=1e-17
        )
        self.assertAlmostEqual(tiny.value, 1e-17, delta=1e-30)
        with self.assertRaises(ValueError):
            tanh_sinh_pieces(lambda u, uc: 1.0, (1.5,))


if __name__ == "__main__":
    unittest.main()


class EleventhReviewRegressionTests(unittest.TestCase):
    """Contradictory or blind quadrature results are never called converged."""

    def test_auto_does_not_trust_a_blind_kronrod_zero(self) -> None:
        """exp(-|x|) on +/-1e10: exact 2; neither rule resolves it, so flag it."""
        from probstat.numerics import integrate

        result = integrate(lambda x: math.exp(-abs(x)), -1e10, 1e10)
        self.assertFalse(result.converged)

    def test_discrete_law_without_isf(self) -> None:
        """Poisson laws have no isf; W1 must stay finite (exact 0.5)."""
        from probstat.distributions import Poisson
        from probstat.information import quantile_wasserstein

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = quantile_wasserstein(Poisson(2.0), Poisson(2.5), p=1.0)
        self.assertAlmostEqual(value, 0.5, delta=1e-3)
