"""Tests for matrix-free operators, covariance factors and Krylov solvers.

Every operator is compared with its explicitly materialized matrix, and every
transpose with the dot-product identity [EQ-LINALG-OP-001]. The least-squares
solvers are compared with an independent dense oracle: Householder QR of the
stacked matrix ``[A; damp I]`` (``probstat.linalg.householder_qr``) or, for a
wide matrix, the minimum-norm solution from the pseudo-inverse.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.linalg import (
    BlockDiagonalCovariance,
    BlockDiagonalOperator,
    CallbackOperator,
    ComposedOperator,
    Convolution1DOperator,
    CountingOperator,
    DenseCovariance,
    DenseOperator,
    DiagonalCovariance,
    DiagonalOperator,
    FirstDifferenceOperator,
    GridGradientOperator,
    HStackOperator,
    IdentityOperator,
    PermutationOperator,
    ScaledIdentityCovariance,
    ScaledOperator,
    SelectionOperator,
    SumOperator,
    TransposedOperator,
    VStackOperator,
    ZeroOperator,
    adjoint_test,
    cgls,
    cholesky,
    conjugate_gradient,
    householder_qr,
    hutchinson_trace,
    lanczos_eigenpairs,
    log_det_spd,
    lsmr,
    lsqr,
    materialize,
    matmul,
    matvec,
    power_iteration,
    pseudo_inverse,
    randomized_svd,
    singular_value_decomposition,
    solve_spd,
    transpose,
    weighted_adjoint,
)

SEED = 20261126


def random_matrix(
    rows: int, columns: int, generator: random.Random
) -> list[list[float]]:
    """Return a Gaussian matrix.

    References:
        [EQ-LINALG-OP-001].

    """
    return [[generator.gauss(0.0, 1.0) for _ in range(columns)] for _ in range(rows)]


def random_vector(size: int, generator: random.Random) -> list[float]:
    """Return a Gaussian vector.

    References:
        [EQ-LINALG-OP-001].

    """
    return [generator.gauss(0.0, 1.0) for _ in range(size)]


def max_abs_difference(left: object, right: object) -> float:
    """Return the largest absolute entrywise difference of two vectors.

    References:
        [EQ-LINALG-OP-001].

    """
    pairs = zip(left, right, strict=True)  # type: ignore[call-overload]
    return max((abs(a - b) for a, b in pairs), default=0.0)


def dense_least_squares(
    matrix: list[list[float]], data: list[float], damp: float
) -> tuple[float, ...]:
    """Return the damped least-squares solution by QR of ``[A; damp I]``.

    For a wide undamped matrix the minimum-norm solution ``A^+ b`` is returned.

    References:
        [EQ-LINALG-009], [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    rows, columns = len(matrix), len(matrix[0])
    if damp == 0.0 and rows < columns:
        return matvec(pseudo_inverse(matrix), data)
    stacked = [list(row) for row in matrix]
    rhs = list(data)
    if damp > 0.0:
        for j in range(columns):
            stacked.append([damp if i == j else 0.0 for i in range(columns)])
            rhs.append(0.0)
    return householder_qr(stacked).solve(tuple(rhs))


class OperatorAlgebraTests(unittest.TestCase):
    """Operators match their materialized matrices and pass the adjoint test."""

    def setUp(self) -> None:
        """Create a seeded generator.

        References:
            [EQ-LINALG-OP-001].

        """
        self.generator = random.Random(SEED)

    def check_operator(self, operator: object, dense: object) -> None:
        """Compare ``matvec``/``rmatvec`` with a dense matrix and run the adjoint test.

        References:
            [EQ-LINALG-OP-001].

        """
        rows, columns = operator.shape  # type: ignore[attr-defined]
        x = random_vector(columns, self.generator)
        y = random_vector(rows, self.generator)
        self.assertLess(max_abs_difference(operator.matvec(x), matvec(dense, x)), 1e-12)  # type: ignore[attr-defined, arg-type]
        self.assertLess(
            max_abs_difference(operator.rmatvec(y), matvec(transpose(dense), y)),  # type: ignore[attr-defined, arg-type]
            1e-12,
        )
        self.assertTrue(adjoint_test(operator, x, y).passed)  # type: ignore[arg-type]

    def test_dense_rectangular_shapes(self) -> None:
        """Tall 7x4 and wide 3x8 dense operators satisfy the adjoint identity."""
        for rows, columns in ((7, 4), (3, 8)):
            matrix = random_matrix(rows, columns, self.generator)
            operator = DenseOperator(matrix)
            self.assertEqual(operator.shape, (rows, columns))
            self.check_operator(operator, matrix)
            result = adjoint_test(
                operator,
                random_vector(columns, self.generator),
                random_vector(rows, self.generator),
            )
            self.assertLessEqual(result.relative_error, 1e-12)

    def test_stacked_tikhonov_transpose(self) -> None:
        """``[A; lambda L]'(u, v) = A'u + lambda L'v`` [EQ-LINALG-OP-002]."""
        matrix = random_matrix(6, 5, self.generator)
        lam = 0.7
        stacked = VStackOperator(
            [DenseOperator(matrix), ScaledOperator(FirstDifferenceOperator(5), lam)]
        )
        u = random_vector(6, self.generator)
        v = random_vector(4, self.generator)
        expected = [
            a + lam * b
            for a, b in zip(
                matvec(transpose(matrix), u),
                FirstDifferenceOperator(5).rmatvec(v),
                strict=True,
            )
        ]
        self.assertLess(max_abs_difference(stacked.rmatvec(u + v), expected), 1e-12)
        self.check_operator(stacked, materialize(stacked))

    def test_composition_sum_and_stacks_match_dense(self) -> None:
        """Every combinator agrees with the dense algebra."""
        a = random_matrix(4, 3, self.generator)
        b = random_matrix(3, 5, self.generator)
        c = random_matrix(4, 3, self.generator)
        da, db, dc = DenseOperator(a), DenseOperator(b), DenseOperator(c)
        self.check_operator(ComposedOperator([da, db]), matmul(a, b))
        self.check_operator(
            SumOperator([da, dc]),
            [
                [x + y for x, y in zip(r, s, strict=True)]
                for r, s in zip(a, c, strict=True)
            ],
        )
        self.check_operator(
            ScaledOperator(da, -2.5), [[-2.5 * x for x in row] for row in a]
        )
        self.check_operator(TransposedOperator(da), transpose(a))
        self.check_operator(da.transpose(), transpose(a))
        self.check_operator(
            HStackOperator([da, dc]), [r + s for r, s in zip(a, c, strict=True)]
        )
        block = materialize(BlockDiagonalOperator([da, db]))
        self.assertEqual(len(block), 7)
        self.assertEqual(len(block[0]), 8)
        self.check_operator(BlockDiagonalOperator([da, db]), block)
        self.assertEqual(block[0][3:], (0.0,) * 5)

    def test_shape_mismatches_raise(self) -> None:
        """Wrong shapes are rejected early and no broadcasting happens."""
        da = DenseOperator(random_matrix(4, 3, self.generator))
        db = DenseOperator(random_matrix(4, 3, self.generator))
        with self.assertRaises(ValueError):
            ComposedOperator([da, db])
        with self.assertRaises(ValueError):
            SumOperator([da, DenseOperator(random_matrix(3, 3, self.generator))])
        with self.assertRaises(ValueError):
            VStackOperator([da, DenseOperator(random_matrix(2, 2, self.generator))])
        with self.assertRaises(ValueError):
            da.matvec([1.0, 2.0])
        with self.assertRaises(ValueError):
            da.rmatvec([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            da.matvec([1.0, math.nan, 2.0])
        with self.assertRaises(ValueError):
            DenseOperator([[1.0, 2.0], [3.0]])

    def test_callback_output_is_validated(self) -> None:
        """A callback of the wrong length or with NaN output raises."""
        short = CallbackOperator((3, 2), lambda x: (1.0, 2.0), lambda y: (0.0, 0.0))
        with self.assertRaises(ValueError):
            short.matvec([1.0, 1.0])
        bad = CallbackOperator((2, 2), lambda x: (math.nan, 0.0), lambda y: y)
        with self.assertRaises(ValueError):
            bad.matvec([1.0, 1.0])
        with self.assertRaises(TypeError):
            CallbackOperator((2, 2), "not callable", lambda y: y)  # type: ignore[arg-type]

    def test_primitive_operators(self) -> None:
        """Identity, zero, diagonal, selection and permutation match dense forms."""
        self.check_operator(
            IdentityOperator(3), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        self.check_operator(ZeroOperator(2, 3), ((0.0,) * 3, (0.0,) * 3))
        self.check_operator(
            DiagonalOperator([2.0, -1.0, 0.5]), ((2.0, 0, 0), (0, -1.0, 0), (0, 0, 0.5))
        )
        selection = SelectionOperator(4, [2, 0, 2])
        self.check_operator(selection, ((0, 0, 1, 0), (1, 0, 0, 0), (0, 0, 1, 0)))
        self.assertEqual(selection.rmatvec([1.0, 2.0, 3.0]), (2.0, 0.0, 4.0, 0.0))
        self.check_operator(
            PermutationOperator([2, 0, 1]), ((0, 0, 1), (1, 0, 0), (0, 1, 0))
        )
        with self.assertRaises(ValueError):
            SelectionOperator(3, [3])
        with self.assertRaises(ValueError):
            PermutationOperator([0, 0, 1])

    def test_difference_operators_boundaries_and_nullspace(self) -> None:
        """First differences: shapes, boundary rows and the constant nullspace."""
        h = 0.25
        none = FirstDifferenceOperator(5, spacing=h)
        self.assertEqual(none.shape, (4, 5))
        self.assertEqual(none.matvec([3.0] * 5), (0.0,) * 4)
        periodic = FirstDifferenceOperator(5, spacing=h, boundary="periodic")
        self.assertEqual(periodic.matvec([3.0] * 5), (0.0,) * 5)
        dirichlet = FirstDifferenceOperator(4, spacing=h, boundary="dirichlet")
        self.assertEqual(dirichlet.shape, (5, 4))
        self.assertEqual(
            dirichlet.matvec([1.0, 1.0, 1.0, 1.0]), (4.0, 0.0, 0.0, 0.0, -4.0)
        )
        for operator in (none, periodic, dirichlet):
            self.check_operator(operator, materialize(operator))
        self.assertEqual(
            none.matvec([0.0, 1.0, 3.0, 6.0, 10.0]), (4.0, 8.0, 12.0, 16.0)
        )
        with self.assertRaises(ValueError):
            FirstDifferenceOperator(1)
        with self.assertRaises(ValueError):
            FirstDifferenceOperator(4, spacing=0.0)
        with self.assertRaises(ValueError):
            FirstDifferenceOperator(4, boundary="neumann")

    def test_grid_gradient_and_convolution(self) -> None:
        """The image gradient and the 1D convolution are exact transposes."""
        gradient = GridGradientOperator(3, 4, spacing=(0.5, 2.0))
        self.check_operator(gradient, materialize(gradient))
        self.assertEqual(gradient.matvec([7.0] * 12), (0.0,) * 24)
        ramp = [float(j) for _ in range(3) for j in range(4)]
        values = gradient.matvec(ramp)
        self.assertEqual(values[:4], (0.5, 0.5, 0.5, 0.0))
        self.assertEqual(values[12:], (0.0,) * 12)
        kernel = [0.1, 0.25, 0.5, 0.15]
        for boundary in ("zero", "periodic", "reflect"):
            operator = Convolution1DOperator(kernel, 7, boundary=boundary)
            self.check_operator(operator, materialize(operator))
        periodic = Convolution1DOperator(kernel, 7, boundary="periodic")
        self.assertAlmostEqual(math.fsum(periodic.matvec([1.0] * 7)), 7.0, places=12)
        reflect = Convolution1DOperator([1.0, 0.0, 0.0], 3, boundary="reflect")
        self.assertEqual(reflect.matvec([1.0, 2.0, 3.0]), (1.0, 1.0, 2.0))

    def test_large_convolution_is_matrix_free(self) -> None:
        """A 10000-point convolution transpose costs one application, not n."""
        size = 10_000
        counted = CountingOperator(Convolution1DOperator([0.25, 0.5, 0.25], size))
        y = [1.0] * size
        result = counted.rmatvec(y)
        self.assertEqual(len(result), size)
        self.assertEqual(counted.counts.forward, 0)
        self.assertEqual(counted.counts.transpose, 1)
        with self.assertRaises(ValueError):
            materialize(counted, max_entries=1_000_000)

    def test_weighted_adjoint(self) -> None:
        """``<A x, y>_Y = <x, A* y>_X`` for SPD metrics [EQ-LINALG-OP-005]."""
        matrix = random_matrix(3, 4, self.generator)
        mx = DenseCovariance(
            [
                [2.0, 0.3, 0.0, 0.1],
                [0.3, 1.5, 0.2, 0.0],
                [0.0, 0.2, 1.0, 0.0],
                [0.1, 0.0, 0.0, 0.8],
            ]
        )
        my = DiagonalCovariance([0.5, 2.0, 1.5])
        adjoint = weighted_adjoint(
            DenseOperator(matrix), domain_metric=mx, range_metric=my
        )
        x = random_vector(4, self.generator)
        y = random_vector(3, self.generator)
        left = math.fsum(
            a * b for a, b in zip(my.apply(matvec(matrix, x)), y, strict=True)
        )
        right = math.fsum(
            a * b for a, b in zip(mx.apply(x), adjoint.matvec(y), strict=True)
        )
        self.assertAlmostEqual(left, right, places=12)
        euclidean = matvec(transpose(matrix), y)
        self.assertGreater(max_abs_difference(adjoint.matvec(y), euclidean), 1e-3)
        self.assertTrue(adjoint_test(adjoint, y, x).passed)


class CovarianceFactorTests(unittest.TestCase):
    """Whitening, solves and log-determinants against dense Cholesky formulas."""

    covariance = (
        (2.0, 0.9, 0.1),
        (0.9, 1.5, 0.3),
        (0.1, 0.3, 1.0),
    )

    def test_dense_covariance_actions(self) -> None:
        """``W'W = Gamma^-1``, ``C C' = Gamma`` and ``log det`` [EQ-LINALG-COV-001]."""
        factor = DenseCovariance(self.covariance)
        v = (0.3, -1.0, 2.0)
        self.assertLess(
            max_abs_difference(factor.solve(v), solve_spd(self.covariance, v)), 1e-13
        )
        self.assertLess(
            max_abs_difference(factor.apply(v), matvec(self.covariance, v)), 1e-14
        )
        self.assertAlmostEqual(
            factor.log_det(), log_det_spd(self.covariance), places=13
        )
        whitened = factor.whiten(v)
        self.assertAlmostEqual(
            math.fsum(w * w for w in whitened),
            math.fsum(
                a * b for a, b in zip(v, solve_spd(self.covariance, v), strict=True)
            ),
            places=13,
        )
        lower = cholesky(self.covariance)
        self.assertLess(max_abs_difference(factor.unwhiten(v), matvec(lower, v)), 1e-14)
        self.assertLess(
            max_abs_difference(
                factor.unwhiten_transpose(v), matvec(transpose(lower), v)
            ),
            1e-14,
        )
        # W' is not W for a correlated covariance.
        self.assertGreater(
            max_abs_difference(factor.whiten_transpose(v), factor.whiten(v)), 1e-3
        )
        self.assertTrue(
            adjoint_test(factor.whitening_operator(), v, (1.0, 2.0, -1.0)).passed
        )
        self.assertTrue(
            adjoint_test(factor.factor_operator(), v, (1.0, 2.0, -1.0)).passed
        )
        self.assertTrue(
            adjoint_test(factor.precision_operator(), v, (1.0, 2.0, -1.0)).passed
        )

    def test_block_diagonal_equals_full(self) -> None:
        """Independent blocks give the same actions as the full matrix."""
        full = (
            (2.0, 0.5, 0.0, 0.0),
            (0.5, 1.0, 0.0, 0.0),
            (0.0, 0.0, 3.0, 0.0),
            (0.0, 0.0, 0.0, 0.25),
        )
        blocks = BlockDiagonalCovariance(
            [DenseCovariance(((2.0, 0.5), (0.5, 1.0))), DiagonalCovariance([3.0, 0.25])]
        )
        dense = DenseCovariance(full)
        v = (1.0, -2.0, 0.5, 4.0)
        for method in ("whiten", "whiten_transpose", "unwhiten", "apply", "solve"):
            self.assertLess(
                max_abs_difference(
                    getattr(blocks, method)(v), getattr(dense, method)(v)
                ),
                1e-13,
            )
        self.assertAlmostEqual(blocks.log_det(), dense.log_det(), places=13)

    def test_diagonal_and_scaled_identity(self) -> None:
        """Diagonal factors whiten componentwise."""
        factor = DiagonalCovariance([4.0, 0.25])
        self.assertEqual(factor.whiten([2.0, 1.0]), (1.0, 2.0))
        self.assertAlmostEqual(factor.log_det(), 0.0, places=15)
        iso = ScaledIdentityCovariance(3, 9.0)
        self.assertEqual(iso.whiten([3.0, 6.0, 9.0]), (1.0, 2.0, 3.0))
        self.assertEqual(
            DiagonalCovariance.from_standard_deviations([2.0]).standard_deviations,
            (2.0,),
        )

    def test_invalid_covariances_raise(self) -> None:
        """Non-SPD, asymmetric or non-positive input raises; no jitter is added."""
        with self.assertRaises(ValueError):
            DenseCovariance(((1.0, 2.0), (2.0, 1.0)))
        with self.assertRaises(ValueError):
            DenseCovariance(((1.0, 0.5), (0.4, 1.0)))
        with self.assertRaises(ValueError):
            DiagonalCovariance([1.0, 0.0])
        with self.assertRaises(ValueError):
            DiagonalCovariance([])
        with self.assertRaises(ValueError):
            DenseCovariance(((1.0, 0.0), (0.0, 1.0))).whiten([1.0])


class KrylovSolverTests(unittest.TestCase):
    """CGLS, LSQR and LSMR agree with the dense QR oracle."""

    def setUp(self) -> None:
        """Create a seeded generator.

        References:
            [EQ-LINALG-KRY-005].

        """
        self.generator = random.Random(SEED + 1)

    def test_least_squares_against_dense_oracle(self) -> None:
        """Tall, wide (minimum norm) and damped problems [EQ-LINALG-KRY-007]."""
        for rows, columns in ((7, 4), (3, 8), (25, 10)):
            matrix = random_matrix(rows, columns, self.generator)
            data = random_vector(rows, self.generator)
            for damp in (0.0, 0.4):
                expected = dense_least_squares(matrix, data, damp)
                for solver in (cgls, lsqr, lsmr):
                    if solver is cgls:
                        result = solver(
                            matrix,
                            data,
                            damping=damp,
                            residual_tolerance=1e-14,
                            normal_tolerance=1e-14,
                            max_iterations=100,
                        )
                    else:
                        result = solver(
                            matrix,
                            data,
                            damping=damp,
                            atol=1e-14,
                            btol=1e-14,
                            max_iterations=100,
                        )
                    with self.subTest(
                        shape=(rows, columns), damp=damp, solver=solver.__name__
                    ):
                        self.assertTrue(result.converged, result.stop_reason)
                        self.assertLess(
                            max_abs_difference(result.solution, expected), 1e-9
                        )
                        residual = math.dist(matvec(matrix, result.solution), data)
                        self.assertAlmostEqual(
                            result.true_residual_norm, residual, places=12
                        )
                        self.assertLess(
                            abs(result.residual_norm - residual), 1e-8 * (1 + residual)
                        )
                        self.assertEqual(
                            result.counts.forward, result.counts.transpose + 1
                        )

    def test_lsmr_normal_residual_is_monotone(self) -> None:
        """LSMR's ``||A'r||`` history decreases monotonically [EQ-LINALG-KRY-008]."""
        matrix = random_matrix(30, 12, self.generator)
        for i, row in enumerate(matrix):
            row[0] *= 1e-3 * (i + 1)
        data = random_vector(30, self.generator)
        result = lsmr(matrix, data, atol=1e-14, btol=1e-14, max_iterations=60)
        history = result.normal_residual_history
        for earlier, later in zip(history, history[1:], strict=False):
            self.assertLessEqual(later, earlier * (1 + 1e-10))

    def test_exact_and_zero_cases(self) -> None:
        """Zero data, consistent start and zero operator stop at once."""
        matrix = random_matrix(5, 3, self.generator)
        for solver in (cgls, lsqr, lsmr):
            zero = solver(matrix, [0.0] * 5)
            self.assertEqual(zero.stop_reason, "zero_rhs")
            self.assertEqual(zero.solution, (0.0, 0.0, 0.0))
            x = (1.0, -2.0, 0.5)
            exact = solver(matrix, matvec(matrix, x), initial=x)
            self.assertEqual(exact.stop_reason, "exact")
            nul = solver(ZeroOperator(4, 3), [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(nul.stop_reason, "exact")
            self.assertEqual(nul.solution, (0.0, 0.0, 0.0))

    def test_budget_exhaustion_is_not_convergence(self) -> None:
        """Hitting ``max_iterations`` reports ``converged = False``."""
        matrix = random_matrix(40, 20, self.generator)
        data = random_vector(40, self.generator)
        for solver in (cgls, lsqr, lsmr):
            result = solver(matrix, data, max_iterations=2)
            self.assertEqual(result.stop_reason, "max_iterations")
            self.assertFalse(result.converged)
            self.assertEqual(result.iterations, 2)

    def test_callback_stops(self) -> None:
        """A user rule (e.g. a discrepancy test) stops the iteration."""
        matrix = random_matrix(20, 10, self.generator)
        data = random_vector(20, self.generator)
        seen: list[int] = []

        def rule(k: int, x: tuple[float, ...], residual: float) -> bool:
            """Stop after three iterations.

            References:
                [EQ-LINALG-KRY-006].

            """
            seen.append(k)
            return k >= 3

        result = lsqr(matrix, data, callback=rule)
        self.assertEqual(result.stop_reason, "callback")
        self.assertEqual(result.iterations, 3)
        self.assertEqual(seen, [1, 2, 3])

    def test_rank_deficient_minimum_norm(self) -> None:
        """From zero, CGLS and LSQR return the minimum-norm solution."""
        base = random_matrix(8, 2, self.generator)
        matrix = [[row[0], row[1], row[0] + row[1]] for row in base]
        data = random_vector(8, self.generator)
        expected = matvec(pseudo_inverse(matrix), data)
        for solver in (cgls, lsqr):
            result = solver(matrix, data, max_iterations=20)
            self.assertLess(max_abs_difference(result.solution, expected), 1e-8)

    def test_conjugate_gradient(self) -> None:
        """CG solves an SPD system; an indefinite one breaks down honestly."""
        spd = ((4.0, 1.0, 0.0), (1.0, 3.0, 0.5), (0.0, 0.5, 2.0))
        b = (1.0, 2.0, 3.0)
        result = conjugate_gradient(spd, b)
        self.assertTrue(result.converged)
        self.assertLess(max_abs_difference(result.solution, solve_spd(spd, b)), 1e-12)
        indefinite = conjugate_gradient(((1.0, 0.0), (0.0, -1.0)), (0.0, 1.0))
        self.assertEqual(indefinite.stop_reason, "breakdown")
        self.assertFalse(indefinite.converged)

    def test_large_stencil_least_squares_is_matrix_free(self) -> None:
        """A 4000-unknown damped deconvolution runs without a dense matrix."""
        size = 4000
        operator = Convolution1DOperator([0.2, 0.6, 0.2], size)
        truth = [math.sin(6.0 * math.pi * i / size) for i in range(size)]
        data = operator.matvec(truth)
        result = lsqr(
            operator, data, damping=1e-3, atol=1e-10, btol=1e-10, max_iterations=300
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(
            result.true_residual_norm,
            1e-6 * math.hypot(*data) + 1e-3 * math.hypot(*result.solution),
        )
        self.assertLessEqual(result.counts.forward, 302)


class SpectralEstimatorTests(unittest.TestCase):
    """Power iteration, Lanczos, Hutchinson and randomized SVD."""

    def test_power_iteration_known_spectrum(self) -> None:
        """Known, repeated and zero spectra [EQ-LINALG-KRY-001]."""
        estimate = power_iteration(
            DiagonalOperator([3.0, -5.0, 1.0, 0.5]), tolerance=1e-12, max_iterations=200
        )
        self.assertAlmostEqual(estimate.estimate, 5.0, places=8)
        self.assertLessEqual(estimate.estimate, 5.0 * (1 + 1e-14))
        repeated = power_iteration(
            DiagonalOperator([2.0, 2.0, 1.0]), tolerance=1e-12, max_iterations=200
        )
        self.assertAlmostEqual(repeated.estimate, 2.0, places=8)
        zero = power_iteration(ZeroOperator(3, 4))
        self.assertEqual(zero.estimate, 0.0)

    def test_lanczos_against_dense_eigenvalues(self) -> None:
        """Full Lanczos Ritz values equal the eigenvalues [EQ-LINALG-KRY-002]."""
        generator = random.Random(SEED + 2)
        b = random_matrix(12, 10, generator)
        gram = matmul(transpose(b), b)
        result = lanczos_eigenpairs(DenseOperator(gram), 10)
        exact = sorted(
            (s * s for s in singular_value_decomposition(b).singular_values),
            reverse=True,
        )
        self.assertLess(max_abs_difference(result.eigenvalues, exact), 1e-9 * exact[0])
        for value, vector in zip(
            result.eigenvalues[:3], result.eigenvectors[:3], strict=True
        ):
            residual = [
                a - value * c for a, c in zip(matvec(gram, vector), vector, strict=True)
            ]
            self.assertLess(math.hypot(*residual), 1e-8 * exact[0])

    def test_lanczos_repeated_eigenvalue_breakdown(self) -> None:
        """A repeated eigenvalue is seen once; the invariant subspace is flagged."""
        result = lanczos_eigenpairs(DiagonalOperator([5.0, 4.0, 4.0, 3.0, 1.0]), 5)
        self.assertTrue(result.breakdown)
        self.assertEqual(result.steps, 4)
        self.assertLess(
            max_abs_difference(result.eigenvalues, (5.0, 4.0, 3.0, 1.0)), 1e-10
        )

    def test_hutchinson_trace(self) -> None:
        """Exact on a diagonal; unbiased within MCSE [EQ-LINALG-KRY-003]."""
        self.assertEqual(
            hutchinson_trace(DiagonalOperator([1.0, 2.0, 3.0, 4.0]), probes=3).estimate,
            10.0,
        )
        matrix = [
            [(0.3 if i != j else 1.0 + j / 10.0) for j in range(20)] for i in range(20)
        ]
        exact = math.fsum(matrix[i][i] for i in range(20))
        estimate = hutchinson_trace(DenseOperator(matrix), probes=200, seed=5)
        self.assertLess(abs(estimate.estimate - exact), 4.0 * estimate.standard_error)
        again = hutchinson_trace(DenseOperator(matrix), probes=200, seed=5)
        self.assertEqual(estimate.samples, again.samples)

    def test_randomized_svd(self) -> None:
        """Leading singular triplets with small residuals [EQ-LINALG-KRY-009]."""
        generator = random.Random(SEED + 3)
        u = random_matrix(30, 4, generator)
        v = random_matrix(4, 20, generator)
        low_rank = matmul(u, v)
        exact = singular_value_decomposition(low_rank).singular_values
        result = randomized_svd(low_rank, 4, oversampling=4, power_iterations=1, seed=1)
        self.assertLess(
            max_abs_difference(result.singular_values, exact[:4]), 1e-9 * exact[0]
        )
        self.assertLess(max(result.right_residuals), 1e-8 * exact[0])
        self.assertLess(max(result.left_residuals), 1e-8 * exact[0])
        with self.assertRaises(ValueError):
            randomized_svd(low_rank, 21)


if __name__ == "__main__":
    unittest.main()
