"""Tests of the flexible Golub-Kahan hybrid method."""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.inverse import (
    LinearInverseProblem,
    hybrid_lsqr,
    irls_regularized_least_squares,
)
from probstat.inverse.flexible_hybrid import flexible_hybrid_lsqr
from probstat.linalg import (
    Convolution1DOperator,
    FirstDifferenceOperator,
    LinearOperator,
    cholesky,
)

# Measured on these seeded cases (see the report): l1 gap 0.253e-1 at k = n and
# 0.053 at k = 3n/4; TV gap 0.144 at k = n. Bands are about 1.7-2x the measurement.
L1_FULL_BAND = 0.05
L1_PARTIAL_BAND = 0.10
TV_FULL_BAND = 0.25


def _norm(v: tuple[float, ...] | list[float]) -> float:
    """Euclidean norm."""
    return math.sqrt(math.fsum(a * a for a in v))


def _gaussian_kernel(width: float, half: int) -> list[float]:
    """Normalised Gaussian kernel."""
    taps = [math.exp(-0.5 * (i / width) ** 2) for i in range(-half, half + 1)]
    total = sum(taps)
    return [t / total for t in taps]


def _sparse_case(
    seed: int = 5, n: int = 48, noise: float = 0.01
) -> tuple[LinearInverseProblem, list[float]]:
    """Sparse spikes blurred by a Gaussian kernel with seeded noise."""
    rng = random.Random(seed)
    op = Convolution1DOperator(_gaussian_kernel(2.0, 6), n)
    truth = [0.0] * n
    for index, value in ((8, 1.0), (19, -0.7), (30, 1.5), (40, 0.8)):
        if index < n:
            truth[index] = value
    y = [a + noise * rng.gauss(0.0, 1.0) for a in op.matvec(truth)]
    return LinearInverseProblem.create(op, y), truth


def _tv_case(
    seed: int = 7, n: int = 40, noise: float = 0.01
) -> tuple[LinearInverseProblem, list[float]]:
    """Piecewise-constant signal blurred by a short kernel."""
    rng = random.Random(seed)
    op = Convolution1DOperator([0.25, 0.5, 0.25], n, boundary="reflect")
    truth = [1.0 if 10 <= i < 25 else (0.4 if i >= 25 else 0.0) for i in range(n)]
    y = [a + noise * rng.gauss(0.0, 1.0) for a in op.matvec(truth)]
    return LinearInverseProblem.create(op, y), truth


def _dense(op: LinearOperator, n: int) -> list[list[float]]:
    """Materialise a square operator column by column."""
    columns = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        columns.append(list(op.matvec(e)))
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def _chol_solve(lower: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve ``L L' x = rhs`` for a lower-triangular ``L``."""
    n = len(rhs)
    y = [0.0] * n
    for i in range(n):
        y[i] = (rhs[i] - math.fsum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        s = math.fsum(lower[k][i] * x[k] for k in range(i + 1, n))
        x[i] = (y[i] - s) / lower[i][i]
    return x


class IdentityPreconditionerOracleTest(unittest.TestCase):
    """With M = I and unit weights the method is hybrid_lsqr."""

    def _problem(self) -> LinearInverseProblem:
        rng = random.Random(1)
        matrix = [[1.0 / (1 + i + j) for j in range(8)] for i in range(10)]
        truth = [math.sin(0.7 * j) for j in range(8)]
        y = [
            math.fsum(r * t for r, t in zip(row, truth, strict=True))
            + 1e-3 * rng.gauss(0, 1)
            for row in matrix
        ]
        return LinearInverseProblem.create(matrix, y)

    def _compare(self, problem: LinearInverseProblem, **kwargs: object) -> None:
        steps = 5
        flex = flexible_hybrid_lsqr(
            problem,
            penalty="none",
            max_iterations=steps,
            stabilization_tolerance=0.0,
            **kwargs,  # type: ignore[arg-type]
        )
        ref = hybrid_lsqr(
            problem,
            max_iterations=steps,
            stabilization_tolerance=0.0,
            **kwargs,  # type: ignore[arg-type]
        )
        self.assertEqual(flex.iterations, ref.iterations)
        for a, b in zip(flex.estimate, ref.estimate, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-10)
        lam_ref = ref.regularization_history or ()
        for a, b in zip(flex.lambda_history, lam_ref, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9 * max(1.0, abs(b)))
        for a, b in zip(flex.misfit_history, ref.misfit_history, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-10)
        self.assertLess(flex.orthogonality_loss, 1e-12)

    def test_fixed_lambda(self) -> None:
        """Fixed-lambda iterates equal hybrid_lsqr iterates."""
        self._compare(self._problem(), rule="fixed", regularization=0.05)

    def test_gcv_and_discrepancy(self) -> None:
        """Projected GCV and discrepancy choose the same lambda."""
        problem = self._problem()
        self._compare(problem, rule="gcv")
        self._compare(problem, rule="discrepancy", noise_norm=0.01)

    def test_general_penalty(self) -> None:
        """A general L of the problem is handled like in hybrid_lsqr."""
        base = self._problem()
        problem = LinearInverseProblem.create(
            base.operator, base.data, penalty=FirstDifferenceOperator(8)
        )
        self._compare(problem, rule="fixed", regularization=0.1)

    def test_first_step_uses_unit_weights(self) -> None:
        """The first step of the l1 mode uses unit weights, so it equals hybrid."""
        problem, _ = _sparse_case()
        flex = flexible_hybrid_lsqr(
            problem, penalty="l1", rule="fixed", regularization=0.1, max_iterations=1
        )
        ref = hybrid_lsqr(problem, rule="fixed", regularization=0.1, max_iterations=1)
        for a, b in zip(flex.estimate, ref.estimate, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-10)


class ClosedFormTest(unittest.TestCase):
    """Full-dimension subspace: the last step solves the weighted problem."""

    def test_full_space_matches_weighted_normal_equations(self) -> None:
        """With k = n the final iterate solves (A'A + lam^2 S^2) x = A'y."""
        n = 12
        problem, _ = _sparse_case(n=n, noise=0.02)
        lam = 0.15
        result = flexible_hybrid_lsqr(
            problem,
            penalty="l1",
            preconditioner="identity",
            rule="fixed",
            regularization=lam,
            max_iterations=n,
            stabilization_tolerance=0.0,
        )
        self.assertEqual(result.iterations, n)
        weights = result.weights or ()
        self.assertEqual(len(weights), n)
        a = _dense(problem.operator, n)
        gram = [
            [
                math.fsum(a[r][i] * a[r][j] for r in range(n))
                + (lam * lam * weights[i] if i == j else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]
        rhs = [math.fsum(a[r][i] * problem.data[r] for r in range(n)) for i in range(n)]
        factor = cholesky(tuple(tuple(r) for r in gram))
        solution = _chol_solve([list(r) for r in factor], rhs)
        # The weights of the last step come from the previous iterate, so the
        # last iterate is the exact minimiser of that weighted functional.
        for got, want in zip(result.estimate, solution, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-6)

    def test_projected_residual_is_data_misfit(self) -> None:
        """misfit_history equals ||A x_k - y|| for the final iterate."""
        problem, _ = _sparse_case()
        for pre in ("weights", "identity"):
            result = flexible_hybrid_lsqr(
                problem,
                penalty="l1",
                preconditioner=pre,
                rule="fixed",
                regularization=0.1,
                max_iterations=12,
            )
            residual = [
                a - y
                for a, y in zip(
                    problem.operator.matvec(result.estimate), problem.data, strict=True
                )
            ]
            self.assertAlmostEqual(
                _norm(residual), result.misfit_history[-1], delta=1e-10
            )
            self.assertLess(result.orthogonality_loss, 1e-12)


class SparseDeconvolutionTest(unittest.TestCase):
    """Comparison with the IRLS objective (measured, not presumed)."""

    def test_gap_to_irls_is_small(self) -> None:
        """The flexible objective approaches the IRLS objective as k grows."""
        problem, truth = _sparse_case(n=32)
        lam = 0.1
        irls = irls_regularized_least_squares(
            problem.operator, problem.data, lam * lam, penalty="l1", max_iterations=100
        )
        gaps = []
        for steps, band in ((24, L1_PARTIAL_BAND), (32, L1_FULL_BAND)):
            flex = flexible_hybrid_lsqr(
                problem,
                penalty="l1",
                rule="fixed",
                regularization=lam,
                max_iterations=steps,
                stabilization_tolerance=0.0,
                truth=truth,
            )
            gap = (flex.objective_history[-1] - irls.objective) / irls.objective
            gaps.append(gap)
            self.assertGreaterEqual(gap, -1e-6)  # IRLS objective is a minimum
            self.assertLess(gap, band)
            self.assertEqual(len(flex.lambda_history), flex.iterations)
            self.assertEqual(len(flex.objective_history), flex.iterations + 1)
            # One extra forward product forms the residual y - A x_ref.
            self.assertEqual(flex.counts.forward, flex.iterations + 1)
            self.assertEqual(flex.counts.transpose, flex.iterations)
            self.assertEqual(flex.penalty_counts.forward, flex.iterations)
        self.assertLess(gaps[1], gaps[0])

    def test_adaptive_rules_run(self) -> None:
        """GCV, WGCV and discrepancy rules run and beat the zero estimate."""
        problem, truth = _sparse_case(n=32)
        for kwargs in (
            {"rule": "gcv"},
            {"rule": "wgcv", "gcv_weight": 0.5},
            {"rule": "discrepancy", "noise_norm": 0.01 * math.sqrt(32)},
        ):
            result = flexible_hybrid_lsqr(
                problem,
                penalty="l1",
                max_iterations=5,
                truth=truth,
                **kwargs,  # type: ignore[arg-type]
            )
            self.assertTrue(all(math.isfinite(v) for v in result.estimate))
            assert result.error_history is not None
            self.assertLess(min(result.error_history), _norm(truth))


class TotalVariationTest(unittest.TestCase):
    """Piecewise-constant signal against the existing TV solver."""

    def test_gap_to_irls_tv(self) -> None:
        """The anisotropic TV objective is within the measured band of IRLS."""
        problem, truth = _tv_case(n=24)
        lam = 0.1
        irls = irls_regularized_least_squares(
            problem.operator, problem.data, lam * lam, penalty="tv", max_iterations=100
        )
        flex = flexible_hybrid_lsqr(
            problem,
            penalty="tv",
            rule="fixed",
            regularization=lam,
            max_iterations=24,
            stabilization_tolerance=0.0,
            truth=truth,
        )
        gap = (flex.objective_history[-1] - irls.objective) / irls.objective
        self.assertGreaterEqual(gap, -1e-6)
        self.assertLess(gap, TV_FULL_BAND)


class FailureTest(unittest.TestCase):
    """Invalid input and numerical failure."""

    def setUp(self) -> None:
        """Build a small sparse problem."""
        self.problem, _ = _sparse_case(n=16)

    def test_invalid_arguments(self) -> None:
        """Invalid rules, penalties and parameters raise."""
        p = self.problem
        bad_calls: list[dict[str, object]] = [
            {"rule": "nope"},
            {"penalty": "nope"},
            {"rule": "discrepancy"},
            {"rule": "fixed"},
            {"rule": "fixed", "regularization": -1.0},
            {"rule": "wgcv", "gcv_weight": 2.0},
            {"epsilon": 0.0},
            {"epsilon": float("nan")},
            {"preconditioner": "weights", "penalty": "tv"},
            {"preconditioner": "nope"},
            {"penalty": "custom"},
            {"penalty": "none", "preconditioner": "weights"},
            {"stabilization_tolerance": -1.0},
            {"max_iterations": 0},
        ]
        for kwargs in bad_calls:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                flexible_hybrid_lsqr(p, **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            flexible_hybrid_lsqr(p, max_iterations=2.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            flexible_hybrid_lsqr("problem")  # type: ignore[arg-type]

    def test_problem_penalty_conflict(self) -> None:
        """A non-identity problem penalty with a sparse penalty is rejected."""
        problem = LinearInverseProblem.create(
            self.problem.operator,
            self.problem.data,
            penalty=FirstDifferenceOperator(16),
        )
        with self.assertRaises(ValueError):
            flexible_hybrid_lsqr(problem, penalty="l1")

    def test_bad_preconditioner_callback(self) -> None:
        """A preconditioner with non-finite or non-positive entries raises."""
        for bad in (float("nan"), 0.0, -1.0, float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                flexible_hybrid_lsqr(
                    self.problem, preconditioner=lambda w, b=bad: (b,) * 16
                )
        with self.assertRaises(ValueError):
            flexible_hybrid_lsqr(self.problem, preconditioner=lambda w: (1.0,) * 3)

    def test_callback_equal_to_identity(self) -> None:
        """A callback returning ones reproduces the identity preconditioner."""
        common: dict[str, object] = {
            "rule": "fixed",
            "regularization": 0.1,
            "max_iterations": 6,
        }
        a = flexible_hybrid_lsqr(
            self.problem,
            preconditioner="identity",
            **common,  # type: ignore[arg-type]
        )
        b = flexible_hybrid_lsqr(
            self.problem,
            preconditioner=lambda w: (1.0,) * 16,
            **common,  # type: ignore[arg-type]
        )
        self.assertEqual(a.estimate, b.estimate)
        self.assertEqual(b.preconditioner, "callback")

    def test_zero_data_and_breakdown(self) -> None:
        """Zero data is exact; a rank-one operator breaks down after one step."""
        zero = LinearInverseProblem.create([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
        res = flexible_hybrid_lsqr(zero, rule="fixed", regularization=0.1)
        self.assertEqual(res.stop_reason, "breakdown_exact")
        self.assertEqual(res.iterations, 0)
        rank_one = LinearInverseProblem.create(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], [1.0, 2.0, 3.0]
        )
        res = flexible_hybrid_lsqr(
            rank_one,
            penalty="none",
            rule="fixed",
            regularization=0.1,
            max_iterations=3,
            stabilization_tolerance=0.0,
        )
        self.assertTrue(res.converged)
        self.assertEqual(res.iterations, 1)
        self.assertEqual(res.stop_reason, "breakdown_exact")

    def test_non_finite_iterate_is_reported(self) -> None:
        """A callback that overflows the iterate stops with a failure reason."""
        res = flexible_hybrid_lsqr(
            self.problem,
            preconditioner=lambda w: (1e308,) * 16,
            rule="fixed",
            regularization=1e-3,
            max_iterations=4,
        )
        self.assertIn(res.stop_reason, ("non_finite", "projected_solve_failed"))
        self.assertFalse(res.converged)

    def test_iteration_cap(self) -> None:
        """The cap is reported as max_iterations."""
        res = flexible_hybrid_lsqr(
            self.problem,
            rule="fixed",
            regularization=0.1,
            max_iterations=3,
            stabilization_tolerance=0.0,
        )
        self.assertEqual(res.iterations, 3)
        self.assertEqual(res.stop_reason, "max_iterations")
        self.assertFalse(res.converged)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudyTest(unittest.TestCase):
    """Gap to IRLS over lambda and seeds (slow tier)."""

    def test_gap_sweep(self) -> None:
        """Print the relative objective gap for several lambda and seeds."""
        worst = 0.0
        for seed in range(3):
            problem, _ = _sparse_case(seed=seed, n=64)
            for lam in (0.1, 0.2):
                irls = irls_regularized_least_squares(
                    problem.operator,
                    problem.data,
                    lam * lam,
                    penalty="l1",
                    max_iterations=100,
                )
                flex = flexible_hybrid_lsqr(
                    problem,
                    penalty="l1",
                    rule="fixed",
                    regularization=lam,
                    max_iterations=40,
                )
                gap = (flex.objective_history[-1] - irls.objective) / irls.objective
                print(f"seed={seed} lam={lam} gap={gap:.4g} iters={flex.iterations}")
                worst = max(worst, gap)
        self.assertLess(worst, 0.2)


if __name__ == "__main__":
    unittest.main()
