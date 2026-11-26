"""Tests for the generalized linear inverse problem (roadmap task 05).

Oracles are independent of the implementation under test: closed-form
scalar solutions, the legacy ``tikhonov`` (stacked QR with ``W = I`` and
``x_ref = 0``), the normal-equation formula solved by ``lu_solve`` on small
well-conditioned cases, the KKT system of equality-constrained least squares,
and the enumeration of all active sets for tiny NNLS problems.
"""

from __future__ import annotations

import itertools
import math
import random
import unittest

from probstat.inverse import tikhonov
from probstat.inverse.linear_problem import (
    LinearInverseProblem,
    bounded_least_squares,
    equality_constrained_least_squares,
    linear_estimator_analysis,
    nonnegative_least_squares,
    regularized_factorization,
    solve_regularized,
    total_least_squares,
)
from probstat.linalg import (
    CountingOperator,
    DenseCovariance,
    DenseOperator,
    DiagonalCovariance,
    householder_qr,
    lu_solve,
    matmul,
    matvec,
    solve_spd,
    transpose,
)

SEED = 5150


def gaussian_matrix(
    rows: int, columns: int, generator: random.Random
) -> list[list[float]]:
    """Return a Gaussian matrix.

    References:
        [EQ-INV-LIN-001].

    """
    return [[generator.gauss(0.0, 1.0) for _ in range(columns)] for _ in range(rows)]


def max_diff(left: object, right: object) -> float:
    """Return the largest absolute entrywise difference.

    References:
        [EQ-INV-LIN-001].

    """
    return max(abs(a - b) for a, b in zip(left, right, strict=True))  # type: ignore[call-overload]


def normal_equation_solution(
    a: list[list[float]],
    y: list[float],
    covariance: list[list[float]],
    penalty: list[list[float]],
    reference: list[float],
    lam: float,
) -> tuple[float, ...]:
    """Solve ``(A'G^-1 A + lam^2 L'L) x = A'G^-1 y + lam^2 L'L x_ref`` (small oracle).

    References:
        [EQ-INV-LIN-001].

    """
    n = len(a[0])
    ginv_a = [solve_spd(covariance, [a[i][j] for i in range(len(a))]) for j in range(n)]
    ginv_a_rows = transpose(ginv_a)
    normal = matmul(transpose(a), ginv_a_rows)
    ltl = matmul(transpose(penalty), penalty)
    system = [
        [normal[i][j] + lam * lam * ltl[i][j] for j in range(n)] for i in range(n)
    ]
    rhs_data = matvec(transpose(a), solve_spd(covariance, y))
    rhs_pen = matvec(ltl, reference)
    rhs = [d + lam * lam * p for d, p in zip(rhs_data, rhs_pen, strict=True)]
    return lu_solve(system, rhs)


class AffineTikhonovTests(unittest.TestCase):
    """Weighted, affine Tikhonov against independent formulas [EQ-INV-LIN-001]."""

    def setUp(self) -> None:
        """Build a correlated-noise problem with a first-difference penalty.

        References:
            [EQ-INV-LIN-001].

        """
        generator = random.Random(SEED)
        self.a = gaussian_matrix(9, 5, generator)
        self.y = [generator.gauss(0.0, 1.0) for _ in range(9)]
        self.covariance = [
            [(1.0 if i == j else 0.0) + 0.4 * 0.6 ** abs(i - j) for j in range(9)]
            for i in range(9)
        ]
        self.penalty = [
            [(-1.0 if j == i else 1.0 if j == i + 1 else 0.0) for j in range(5)]
            for i in range(4)
        ]
        self.reference = [generator.gauss(0.0, 1.0) for _ in range(5)]
        self.problem = LinearInverseProblem.create(
            self.a,
            self.y,
            noise=DenseCovariance(self.covariance),
            penalty=self.penalty,
            reference=self.reference,
        )

    def test_scalar_closed_form(self) -> None:
        """``x = (a y / s^2 + lam^2 x_ref) / (a^2 / s^2 + lam^2)`` for a scalar."""
        a, y, s2, lam, ref = 2.0, 3.0, 0.25, 1.5, -0.4
        problem = LinearInverseProblem.create(
            [[a]], [y], noise=DiagonalCovariance([s2]), reference=[ref]
        )
        expected = (a * y / s2 + lam * lam * ref) / (a * a / s2 + lam * lam)
        for method in ("qr", "svd", "lsqr", "lsmr", "cgls"):
            solution = solve_regularized(problem, lam, method=method)
            self.assertAlmostEqual(solution.estimate[0], expected, places=12)
        solution = solve_regularized(problem, lam)
        objective = (
            0.5 * (a * expected - y) ** 2 / s2 + 0.5 * lam**2 * (expected - ref) ** 2
        )
        self.assertAlmostEqual(solution.objective, objective, places=12)
        self.assertLess(solution.stationarity, 1e-12)

    def test_all_routes_match_normal_equations(self) -> None:
        """QR, SVD and the Krylov routes agree with the small normal-equation oracle."""
        lam = 0.7
        expected = normal_equation_solution(
            self.a, self.y, self.covariance, self.penalty, self.reference, lam
        )
        for method in ("qr", "svd", "lsqr", "lsmr", "cgls"):
            with self.subTest(method=method):
                solution = solve_regularized(
                    self.problem, lam, method=method, tolerance=1e-13
                )
                self.assertLess(max_diff(solution.estimate, expected), 1e-9)
                self.assertLess(solution.stationarity, 1e-8)

    def test_legacy_tikhonov_equivalence(self) -> None:
        """``W = I`` and ``x_ref = 0`` reproduce ``tikhonov`` exactly."""
        for lam in (0.0, 0.05, 1.3):
            legacy = tikhonov(self.a, self.y, lam)
            new = solve_regularized(LinearInverseProblem.create(self.a, self.y), lam)
            self.assertLess(max_diff(legacy.solution, new.estimate), 1e-13)
            self.assertAlmostEqual(legacy.residual_norm, new.residual_norm, places=12)
        legacy = tikhonov(self.a, self.y, 0.4, penalty=self.penalty)
        new = solve_regularized(
            LinearInverseProblem.create(self.a, self.y, penalty=self.penalty), 0.4
        )
        self.assertLess(max_diff(legacy.solution, new.estimate), 1e-13)

    def test_row_permutation_invariance(self) -> None:
        """Permuting observations with the covariance leaves ``x`` unchanged."""
        perm = [3, 0, 8, 1, 7, 2, 6, 4, 5]
        a = [self.a[i] for i in perm]
        y = [self.y[i] for i in perm]
        cov = [[self.covariance[i][j] for j in perm] for i in perm]
        permuted = LinearInverseProblem.create(
            a,
            y,
            noise=DenseCovariance(cov),
            penalty=self.penalty,
            reference=self.reference,
        )
        base = solve_regularized(self.problem, 0.7)
        other = solve_regularized(permuted, 0.7)
        self.assertLess(max_diff(base.estimate, other.estimate), 1e-12)
        self.assertAlmostEqual(base.misfit, other.misfit, places=12)
        operator_permuted = LinearInverseProblem.create(
            matmul(
                [[1.0 if j == perm[i] else 0.0 for j in range(9)] for i in range(9)],
                self.a,
            ),
            y,
            noise=DenseCovariance(cov),
            penalty=self.penalty,
            reference=self.reference,
        )
        self.assertLess(
            max_diff(solve_regularized(operator_permuted, 0.7).estimate, base.estimate),
            1e-12,
        )

    def test_limits_of_lambda(self) -> None:
        """``lambda -> inf`` projects on ``x_ref + null(L)``; ``0`` is GLS."""
        heavy = solve_regularized(self.problem, 1e7)
        increments = [
            a - b for a, b in zip(heavy.estimate, self.reference, strict=True)
        ]
        spread = max(increments) - min(increments)
        self.assertLess(spread, 1e-6)
        gls = solve_regularized(self.problem, 0.0)
        expected = normal_equation_solution(
            self.a, self.y, self.covariance, self.penalty, self.reference, 0.0
        )
        self.assertLess(max_diff(gls.estimate, expected), 1e-10)

    def test_nonunique_problems_are_explicit(self) -> None:
        """Rank-deficient stack: QR raises; SVD gives min ``||x - x_ref||``."""
        wide = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
        problem = LinearInverseProblem.create(
            wide, [2.0, 3.0], reference=[1.0, 0.0, 0.0]
        )
        with self.assertRaises(ValueError):
            solve_regularized(problem, 0.0, method="qr")
        solution = solve_regularized(problem, 0.0, method="svd")
        self.assertEqual(solution.status, "minimum_norm")
        self.assertEqual(solution.rank, 2)
        self.assertLess(solution.misfit, 1e-12)
        # Minimum-norm increment z = A^+ (y - A x_ref) lies in range(A').
        z = [a - b for a, b in zip(solution.estimate, (1.0, 0.0, 0.0), strict=True)]
        null_direction = (1.0, -1.0, 1.0)
        self.assertAlmostEqual(
            math.fsum(a * b for a, b in zip(z, null_direction, strict=True)),
            0.0,
            places=12,
        )
        constant = LinearInverseProblem.create(
            [[1.0, -1.0], [2.0, -2.0]], [0.0, 0.0], penalty=[[1.0, -1.0]]
        )
        with self.assertRaises(ValueError):
            solve_regularized(constant, 1.0, method="qr")

    def test_zero_operator_and_invalid_input(self) -> None:
        """A zero operator returns the reference; invalid input raises."""
        problem = LinearInverseProblem.create(
            [[0.0, 0.0], [0.0, 0.0]], [1.0, 2.0], reference=[0.5, -1.0]
        )
        solution = solve_regularized(problem, 0.3)
        self.assertLess(max_diff(solution.estimate, (0.5, -1.0)), 1e-14)
        with self.assertRaises(ValueError):
            LinearInverseProblem.create([[1.0]], [1.0, 2.0])
        with self.assertRaises(ValueError):
            LinearInverseProblem.create([[1.0]], [math.nan])
        with self.assertRaises(ValueError):
            solve_regularized(problem, -1.0)
        with self.assertRaises(ValueError):
            solve_regularized(problem, 1.0, method="normal_equations")
        with self.assertRaises(ValueError):
            LinearInverseProblem.create(
                [[1.0, 2.0]], [1.0], noise=DiagonalCovariance([1.0, 1.0])
            )


class EstimatorAnalysisTests(unittest.TestCase):
    """Resolution, influence and covariance [EQ-INV-LIN-002]."""

    def test_against_explicit_gain(self) -> None:
        """Compare with ``M`` from the normal equations on a small problem."""
        generator = random.Random(SEED + 1)
        a = gaussian_matrix(6, 3, generator)
        variances = [0.5, 1.0, 2.0, 0.25, 1.5, 0.8]
        lam = 0.6
        problem = LinearInverseProblem.create(
            a, [0.0] * 6, noise=DiagonalCovariance(variances)
        )
        analysis = linear_estimator_analysis(problem, lam)
        weighted = [[a[i][j] / variances[i] for j in range(3)] for i in range(6)]
        normal = matmul(transpose(a), weighted)
        system = [
            [normal[i][j] + (lam * lam if i == j else 0.0) for j in range(3)]
            for i in range(3)
        ]
        for r in range(6):
            column = lu_solve(system, [weighted[r][j] for j in range(3)])
            for i in range(3):
                self.assertAlmostEqual(analysis.gain[i][r], column[i], places=12)
        resolution = matmul(analysis.gain, a)
        self.assertLess(
            max_diff(
                [v for row in resolution for v in row],
                [v for row in analysis.resolution for v in row],
            ),
            1e-12,
        )
        trace_r = math.fsum(analysis.resolution[i][i] for i in range(3))
        self.assertAlmostEqual(analysis.degrees_of_freedom, trace_r, places=12)
        cov = [[variances[i] if i == j else 0.0 for j in range(6)] for i in range(6)]
        expected_cov = matmul(matmul(analysis.gain, cov), transpose(analysis.gain))
        self.assertLess(
            max_diff(
                [v for row in expected_cov for v in row],
                [v for row in analysis.estimator_covariance for v in row],
            ),
            1e-12,
        )
        # Estimator covariance and Bayesian covariance differ (unless lam = 0).
        self.assertGreater(
            max_diff(
                [v for row in analysis.bayesian_covariance for v in row],
                [v for row in analysis.estimator_covariance for v in row],
            ),
            1e-3,
        )

    def test_bias_and_risk(self) -> None:
        """``bias = (R - I)(x - x_ref)`` and ``risk = trace + ||bias||^2``."""
        problem = LinearInverseProblem.create([[1.0, 0.0], [0.0, 0.01]], [0.0, 0.0])
        analysis = linear_estimator_analysis(problem, 0.1)
        bias = analysis.bias([1.0, 1.0], [0.0, 0.0])
        self.assertAlmostEqual(bias[0], 1.0 / 1.01 - 1.0, places=12)
        self.assertAlmostEqual(bias[1], 1e-4 / (1e-4 + 1e-2) - 1.0, places=12)
        risk = analysis.risk([1.0, 1.0], [0.0, 0.0])
        trace = (1.0 / 1.01) ** 2 + (0.01 / 0.0101) ** 2
        self.assertAlmostEqual(risk, trace + bias[0] ** 2 + bias[1] ** 2, places=12)
        self.assertEqual(len(analysis.leverages), 2)


class ConstrainedLeastSquaresTests(unittest.TestCase):
    """Equality constraints, NNLS, box constraints and TLS."""

    def test_equality_constraints_against_kkt(self) -> None:
        """The nullspace method matches the KKT system [EQ-INV-LIN-003]."""
        generator = random.Random(SEED + 2)
        a = gaussian_matrix(7, 4, generator)
        y = [generator.gauss(0.0, 1.0) for _ in range(7)]
        c = gaussian_matrix(2, 4, generator)
        d = [0.5, -1.0]
        solution = equality_constrained_least_squares(a, y, c, d)
        ata = matmul(transpose(a), a)
        kkt = [list(ata[i]) + [c[0][i], c[1][i]] for i in range(4)] + [
            c[0] + [0.0, 0.0],
            c[1] + [0.0, 0.0],
        ]
        rhs = list(matvec(transpose(a), y)) + d
        exact = lu_solve(kkt, rhs)
        self.assertLess(max_diff(solution.estimate, exact[:4]), 1e-11)
        self.assertLess(solution.constraint_residual, 1e-12)
        self.assertLess(max_diff(solution.multipliers, [-exact[4], -exact[5]]), 1e-10)
        self.assertLess(solution.optimality, 1e-10)

    def test_redundant_and_inconsistent_constraints(self) -> None:
        """A duplicated constraint is accepted; contradictory ones raise."""
        a = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        y = [1.0, 2.0, 2.0]
        redundant = equality_constrained_least_squares(
            a, y, [[1.0, 1.0], [2.0, 2.0]], [1.0, 2.0]
        )
        self.assertAlmostEqual(
            redundant.estimate[0] + redundant.estimate[1], 1.0, places=12
        )
        with self.assertRaises(ValueError):
            equality_constrained_least_squares(
                a, y, [[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0]
            )
        boundary = equality_constrained_least_squares(
            a, y, [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]
        )
        self.assertLess(max_diff(boundary.estimate, (0.0, 0.0)), 1e-14)

    def test_nnls_against_active_set_enumeration(self) -> None:
        """Lawson-Hanson finds the best feasible active set [EQ-INV-LIN-004]."""
        for trial in range(6):
            generator = random.Random(SEED + 10 + trial)
            a = gaussian_matrix(6, 3, generator)
            y = [generator.gauss(0.0, 1.0) for _ in range(6)]
            best_value = math.inf
            best: tuple[float, ...] | None = None
            for size in range(4):
                for subset in itertools.combinations(range(3), size):
                    x = [0.0, 0.0, 0.0]
                    if subset:
                        sub = [[a[i][j] for j in subset] for i in range(6)]
                        values = householder_qr(sub).solve(tuple(y))
                        if any(v < 0.0 for v in values):
                            continue
                        for j, v in zip(subset, values, strict=True):
                            x[j] = v
                    residual = math.dist(matvec(a, x), y)
                    if residual < best_value:
                        best_value, best = residual, tuple(x)
            solution = nonnegative_least_squares(a, y)
            self.assertTrue(solution.converged, solution.status)
            assert best is not None
            self.assertLess(max_diff(solution.estimate, best), 1e-10)
            for j in range(3):
                if j in solution.active:
                    self.assertLessEqual(solution.multipliers[j], 1e-10)
                    self.assertEqual(solution.estimate[j], 0.0)
                else:
                    self.assertAlmostEqual(solution.multipliers[j], 0.0, places=10)

    def test_clipping_is_not_the_constrained_optimum(self) -> None:
        """Clipping the unconstrained solution is suboptimal on a correlated design."""
        a = [[1.0, 0.9], [0.9, 1.0], [0.1, 0.2]]
        y = [1.0, -0.2, 0.3]
        unconstrained = householder_qr(a).solve(tuple(y))
        clipped = [max(v, 0.0) for v in unconstrained]
        solution = nonnegative_least_squares(a, y)
        self.assertLess(solution.misfit, math.dist(matvec(a, clipped), y) - 1e-6)

    def test_box_constraints(self) -> None:
        """Projected gradients satisfy the box KKT signs [EQ-INV-LIN-005]."""
        generator = random.Random(SEED + 3)
        a = gaussian_matrix(8, 4, generator)
        y = [3.0 * generator.gauss(0.0, 1.0) for _ in range(8)]
        problem = LinearInverseProblem.create(a, y)
        lower, upper = [-0.2] * 4, [0.3] * 4
        solution = bounded_least_squares(
            problem, lower, upper, tolerance=1e-10, max_iterations=20000
        )
        self.assertTrue(solution.converged, solution.status)
        for j, (x, g) in enumerate(
            zip(solution.estimate, solution.multipliers, strict=True)
        ):
            self.assertGreaterEqual(x, lower[j])
            self.assertLessEqual(x, upper[j])
            if lower[j] < x < upper[j]:
                self.assertAlmostEqual(g, 0.0, places=6)
            elif x == lower[j]:
                self.assertLessEqual(g, 1e-6)
            else:
                self.assertGreaterEqual(g, -1e-6)
        wide_open = bounded_least_squares(
            problem, [-1e9] * 4, [1e9] * 4, tolerance=1e-12, max_iterations=20000
        )
        self.assertLess(
            max_diff(wide_open.estimate, householder_qr(a).solve(tuple(y))), 1e-6
        )
        with self.assertRaises(ValueError):
            bounded_least_squares(problem, [1.0] * 4, [0.0] * 4)
        short = bounded_least_squares(problem, lower, upper, max_iterations=1)
        self.assertFalse(short.converged)

    def test_total_least_squares(self) -> None:
        """Consistent data: exact solution; no gap: raises [EQ-INV-LIN-006]."""
        generator = random.Random(SEED + 4)
        a = gaussian_matrix(20, 2, generator)
        x = (1.0, -2.0)
        y = matvec(a, x)
        solution = total_least_squares(a, y)
        self.assertLess(max_diff(solution.estimate, x), 1e-10)
        self.assertLess(solution.smallest_singular_value, 1e-10)
        self.assertGreater(solution.gap, 0.0)
        with self.assertRaises(ValueError):
            total_least_squares([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
        with self.assertRaises(ValueError):
            total_least_squares([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [0.0, 1.0, 0.0])


def replicate_estimator_moments(
    replicates: int, seed: int
) -> tuple[list[list[float]], list[list[float]], list[float], list[float], int]:
    """Return empirical and analytic estimator moments over ``replicates`` datasets.

    The data ``y_k = A x + e_k``, ``e_k ~ N(0, diag(variances))``, are solved
    with one shared factorization. Returns the empirical covariance, the
    analytic ``M Gamma M'``, the empirical mean, the analytic mean
    ``x_ref + R (x - x_ref)`` and the number of forward applications spent
    on materializing the factor.

    References:
        [EQ-INV-LIN-002], [EQ-INV-LIN-007].

    """
    generator = random.Random(seed)
    a = gaussian_matrix(7, 3, generator)
    variances = [0.5, 1.0, 2.0, 0.25, 1.5, 0.8, 1.2]
    truth = [1.0, -0.5, 2.0]
    reference = [0.2, 0.0, -0.1]
    lam = 0.7
    problem = LinearInverseProblem.create(
        a, [0.0] * 7, noise=DiagonalCovariance(variances), reference=reference
    )
    analysis = linear_estimator_analysis(problem, lam)
    factor = regularized_factorization(problem, lam)
    clean = matvec(a, truth)
    datasets = [
        [
            c + math.sqrt(v) * generator.gauss(0.0, 1.0)
            for c, v in zip(clean, variances, strict=True)
        ]
        for _ in range(replicates)
    ]
    estimates = factor.solve_many(datasets, reference)
    mean = [math.fsum(e[i] for e in estimates) / replicates for i in range(3)]
    covariance = [
        [
            math.fsum((e[i] - mean[i]) * (e[j] - mean[j]) for e in estimates)
            / (replicates - 1)
            for j in range(3)
        ]
        for i in range(3)
    ]
    shift = [x - r for x, r in zip(truth, reference, strict=True)]
    expected_mean = [
        reference[i] + math.fsum(analysis.resolution[i][j] * shift[j] for j in range(3))
        for i in range(3)
    ]
    return (
        covariance,
        [list(row) for row in analysis.estimator_covariance],
        mean,
        expected_mean,
        factor.factor_counts.forward,
    )


class FactorizationReuseTests(unittest.TestCase):
    """One QR of the stacked operator for many data vectors [EQ-INV-LIN-007]."""

    def test_matches_separate_solves_and_counts(self) -> None:
        """Every reused solve equals ``solve_regularized``; the factor is built once."""
        generator = random.Random(SEED + 7)
        a = gaussian_matrix(6, 4, generator)
        covariance = [
            [1.0 if i == j else 0.3 if abs(i - j) == 1 else 0.0 for j in range(6)]
            for i in range(6)
        ]
        penalty = [
            [-1.0 if j == i else 1.0 if j == i + 1 else 0.0 for j in range(4)]
            for i in range(3)
        ]
        reference = [0.5, -0.2, 0.1, 0.0]
        counted = CountingOperator(DenseOperator(a))
        base = LinearInverseProblem.create(
            counted, [0.0] * 6, noise=DenseCovariance(covariance), penalty=penalty
        )
        factor = regularized_factorization(base, 0.4)
        self.assertGreater(factor.factor_counts.forward, 0)
        before = (counted.counts.forward, counted.counts.transpose)
        for k in range(5):
            y = [generator.gauss(0.0, 1.0) for _ in range(6)]
            problem = LinearInverseProblem.create(
                a,
                y,
                noise=DenseCovariance(covariance),
                penalty=penalty,
                reference=reference,
            )
            expected = solve_regularized(problem, 0.4).estimate
            self.assertLess(max_diff(factor.solve(y, reference), expected), 1e-12, k)
        # One forward action (A x_ref) per solve, no transpose, no refactorization.
        self.assertEqual(counted.counts.forward, before[0] + 5)
        self.assertEqual(counted.counts.transpose, before[1])
        many = factor.solve_many([[1.0] * 6, [0.0] * 6])
        self.assertEqual(len(many), 2)
        self.assertLess(max(abs(v) for v in many[1]), 1e-15)

    def test_refusals(self) -> None:
        """Rank-deficient, underdetermined and invalid inputs raise."""
        singular = LinearInverseProblem.create([[1.0, 1.0], [2.0, 2.0]], [0.0, 0.0])
        with self.assertRaises(ValueError):
            regularized_factorization(singular, 0.0)
        wide = LinearInverseProblem.create([[1.0, 2.0, 3.0]], [0.0])
        with self.assertRaises(ValueError):
            regularized_factorization(wide, 0.0)
        factor = regularized_factorization(singular, 0.5)
        with self.assertRaises(ValueError):
            factor.solve([1.0])
        with self.assertRaises(ValueError):
            factor.solve([1.0, math.nan])
        with self.assertRaises(TypeError):
            regularized_factorization("problem", 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            regularized_factorization(singular, -1.0)

    def _check_moments(self, replicates: int, seed: int) -> None:
        """Compare replicate moments with ``M Gamma M'`` within 4.5 MCSE.

        For Gaussian estimates the sample covariance entry ``s_ij`` has
        variance ``(S_ii S_jj + S_ij^2) / (R - 1)`` and the sample mean
        ``S_ii / R``.

        References:
            [EQ-INV-LIN-002].

        """
        empirical, analytic, mean, expected_mean, _ = replicate_estimator_moments(
            replicates, seed
        )
        for i in range(3):
            se_mean = math.sqrt(analytic[i][i] / replicates)
            self.assertLess(abs(mean[i] - expected_mean[i]), 4.5 * se_mean, i)
            for j in range(3):
                se = math.sqrt(
                    (analytic[i][i] * analytic[j][j] + analytic[i][j] ** 2)
                    / (replicates - 1)
                )
                self.assertLess(abs(empirical[i][j] - analytic[i][j]), 4.5 * se, (i, j))

    def test_monte_carlo_estimator_covariance_fast(self) -> None:
        """200 replicates agree with the sampling covariance ``M Gamma M'``."""
        self._check_moments(200, SEED + 11)

    def test_monte_carlo_estimator_covariance_1000(self) -> None:
        """1000 replicates agree with ``M Gamma M'`` (0.15 s with the shared factor)."""
        self._check_moments(1000, SEED + 12)


if __name__ == "__main__":
    unittest.main()
