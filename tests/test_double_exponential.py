"""Tests for the tanh-sinh rule's convergence reporting."""

from __future__ import annotations

import math
import unittest
import warnings

from probstat.numerics import tanh_sinh
from probstat.numerics.double_exponential import warn_if_unreliable


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


if __name__ == "__main__":
    unittest.main()
