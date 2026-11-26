"""Regression test for ``examples/large_scale_inverse_lab.py``.

The lab walks through the matrix-free inverse layer: early-stopped LSQR with
the discrepancy principle (``krylov_regularization``), a regularization path
with GCV, UPRE and discrepancy, total variation, and the exact posterior of a
correlated Gaussian benchmark. Only invariants are asserted here (finite
results, stop reasons, interior optima, agreement of two exact solvers); the
ranking of the methods on one seed is not a universal statement.

References:
    [EQ-INV-ITER-002], [EQ-INV-SEL-001], [EQ-INV-LG-002].

"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType

LAB_PATH = Path(__file__).parents[1] / "examples" / "large_scale_inverse_lab.py"


def load_lab() -> ModuleType:
    """Import the lab script as a module without running it.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    name = "_probstat_large_scale_inverse_lab"
    spec = importlib.util.spec_from_file_location(name, LAB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class LargeScaleInverseLabTests(unittest.TestCase):
    """The lab's numbers satisfy the invariants of the methods it calls."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the lab computation once on a small grid."""
        cls.lab = load_lab()
        cls.result = cls.lab.compute(size=32)

    def test_lsqr_stops_by_the_discrepancy_rule(self) -> None:
        """``krylov_regularization`` fires its rule before the iteration cap."""
        result = self.result
        self.assertEqual(result.lsqr_stop, "discrepancy")
        self.assertGreater(result.lsqr_iterations, 0)
        self.assertLess(result.lsqr_iterations, 200)

    def test_table_is_finite_and_costs_are_counted(self) -> None:
        """Every row has a finite error and positive operator counts."""
        self.assertEqual(len(self.result.rows), 5)
        for row in self.result.rows:
            with self.subTest(method=row.method):
                self.assertTrue(math.isfinite(row.error))
                self.assertGreater(row.error, 0.0)
                self.assertGreater(row.forward, 0)
                self.assertGreater(row.transpose, 0)
        lsqr, *_, total_variation = self.result.rows
        self.assertLess(lsqr.forward, total_variation.forward)

    def test_path_choices_are_interior_to_the_grid(self) -> None:
        """GCV, UPRE and the discrepancy rule pick interior grid points."""
        for criterion in ("gcv", "upre", "discrepancy"):
            with self.subTest(criterion=criterion):
                self.assertEqual(self.result.path_statuses[criterion], "interior")
                value = self.result.path_lambdas[criterion]
                assert value is not None
                grid = self.lab.LAMBDA_GRID
                self.assertGreater(value, grid[0])
                self.assertLess(value, grid[-1])

    def test_total_variation_scan_is_reported(self) -> None:
        """The alpha scan is finite, ordered, and the chosen alpha is on it."""
        scan = self.result.tv_scan
        self.assertEqual(tuple(entry[0] for entry in scan), self.lab.ALPHA_GRID)
        for _alpha, misfit, _converged, reason, error in scan:
            self.assertTrue(math.isfinite(misfit) and math.isfinite(error))
            self.assertIsInstance(reason, str)
        misfits = [entry[1] for entry in scan]
        self.assertEqual(misfits, sorted(misfits))
        self.assertIn(self.result.tv_alpha, (*self.lab.ALPHA_GRID, None))

    def test_gaussian_posterior_is_exact_and_consistent(self) -> None:
        """Both solver paths agree and the evidence and spreads are finite."""
        gaussian = self.result.gaussian
        self.assertTrue(math.isfinite(gaussian.log_evidence))
        self.assertLess(gaussian.solver_mean_difference, 1e-9)
        self.assertLess(gaussian.solver_evidence_difference, 1e-9)
        self.assertTrue(all(s > 0.0 and math.isfinite(s) for s in gaussian.sd))
        self.assertLess(max(gaussian.sd), 1.0 + 1e-12)  # prior variance is 1
        self.assertAlmostEqual(gaussian.prior_error, 1.0, places=12)
        self.assertTrue(0 <= gaussian.inside <= len(gaussian.truth))


if __name__ == "__main__":
    unittest.main()
