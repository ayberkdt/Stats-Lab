"""Tests for proximal operators, ISTA/FISTA, PDHG, ADMM and sparse inversion."""

from __future__ import annotations

import itertools
import math
import random
import unittest

from probstat.inverse.sparse import (
    frobenius_norm_bound,
    l1_regularized_least_squares,
    lasso_zero_threshold,
    total_variation_regularized_least_squares,
    total_variation_value,
)
from probstat.linalg import (
    Convolution1DOperator,
    DenseOperator,
    DiagonalCovariance,
    FirstDifferenceOperator,
    GridGradientOperator,
    IdentityOperator,
    adjoint_test,
    lu_solve,
)
from probstat.optimization.proximal import (
    admm,
    fista,
    ista,
    primal_dual_hybrid_gradient,
    project_box,
    project_l2_ball,
    project_linf_ball,
    project_pointwise_l2_ball,
    prox_elastic_net,
    prox_group_l2,
    prox_l1,
    prox_nonnegative_l1,
    soft_threshold,
)


def _random_problem(m: int, n: int, seed: int):
    """Check:  random problem."""
    rng = random.Random(seed)
    matrix = tuple(tuple(rng.gauss(0, 1) for _ in range(n)) for _ in range(m))
    y = tuple(rng.gauss(0, 1) for _ in range(m))
    return matrix, y


def _lasso_objective(matrix, y, alpha, x):
    """Check:  lasso objective."""
    r = [
        sum(a * b for a, b in zip(row, x, strict=True)) - yi
        for row, yi in zip(matrix, y, strict=False)
    ]
    return 0.5 * sum(v * v for v in r) + alpha * sum(abs(v) for v in x)


def _lasso_enumerate(matrix, y, alpha):
    """Check:  lasso enumerate."""
    """Exact lasso by enumerating supports and signs with KKT checks."""
    m, n = len(matrix), len(matrix[0])
    gram = [
        [sum(matrix[k][i] * matrix[k][j] for k in range(m)) for j in range(n)]
        for i in range(n)
    ]
    aty = [sum(matrix[k][i] * y[k] for k in range(m)) for i in range(n)]
    best = None
    for size in range(n + 1):
        for support in itertools.combinations(range(n), size):
            for signs in itertools.product((-1.0, 1.0), repeat=size):
                x = [0.0] * n
                if size:
                    g = tuple(tuple(gram[i][j] for j in support) for i in support)
                    rhs = tuple(
                        aty[i] - alpha * s for i, s in zip(support, signs, strict=True)
                    )
                    sol = lu_solve(g, rhs)
                    if any(s * v <= 0 for s, v in zip(signs, sol, strict=True)):
                        continue
                    for i, v in zip(support, sol, strict=True):
                        x[i] = v
                grad = [
                    sum(gram[i][j] * x[j] for j in range(n)) - aty[i] for i in range(n)
                ]
                if all(
                    abs(grad[i]) <= alpha + 1e-9 for i in range(n) if i not in support
                ):
                    value = _lasso_objective(matrix, y, alpha, x)
                    if best is None or value < best[0]:
                        best = (value, tuple(x))
    return best


class ProxTests(unittest.TestCase):
    """Tests for ProxTests."""

    def test_soft_threshold_exact(self):
        """Check: soft threshold exact."""
        self.assertEqual(
            soft_threshold((3.0, -3.0, 0.5, -0.5), 1.0), (2.0, -2.0, 0.0, 0.0)
        )
        self.assertEqual(soft_threshold((3.0, -3.0), (1.0, 2.0)), (2.0, -1.0))
        self.assertEqual(
            prox_l1((3.0, -3.0, 1.0), 0.5, weights=(1.0, 2.0, 4.0)), (2.5, -2.0, 0.0)
        )

    def test_l1_center_and_box(self):
        """Check: l1 center and box."""
        out = prox_l1((5.0, 0.2), 1.0, center=(1.0, 1.0), lower=0.0, upper=3.0)
        self.assertEqual(out, (3.0, 1.0))  # |0.2-1| < 1 -> centre

    def test_elastic_net_exact(self):
        """Check: elastic net exact."""
        out = prox_elastic_net((3.0, -0.4), 2.0, l1=1.0, l2=0.5)
        # soft(3, 2) = 1, soft(-0.4, 2) = 0, divide by 1 + 2*0.5 = 2
        self.assertEqual(out, (0.5, 0.0))

    def test_nonnegative_and_box(self):
        """Check: nonnegative and box."""
        self.assertEqual(prox_nonnegative_l1((2.0, -1.0, 0.3), 0.5), (1.5, 0.0, 0.0))
        self.assertEqual(
            project_box((-2.0, 0.5, 9.0), 0.0, (1.0, 1.0, 1.0)), (0.0, 0.5, 1.0)
        )
        with self.assertRaises(ValueError):
            project_box((0.0,), 1.0, 0.0)

    def test_group_prox_and_overlap_rejected(self):
        """Check: group prox and overlap rejected."""
        out = prox_group_l2((3.0, 4.0, 0.1), 2.0, [(0, 1), (2,)])
        self.assertAlmostEqual(out[0], 3.0 * 0.6)
        self.assertAlmostEqual(out[1], 4.0 * 0.6)
        self.assertEqual(out[2], 0.0)
        with self.assertRaises(ValueError):
            prox_group_l2((1.0, 2.0, 3.0), 1.0, [(0, 1), (1, 2)])

    def test_ball_projections(self):
        """Check: ball projections."""
        self.assertEqual(project_l2_ball((3.0, 4.0), 1.0), (0.6, 0.8))
        self.assertEqual(project_linf_ball((3.0, -4.0, 0.2), 1.0), (1.0, -1.0, 0.2))

    def test_pointwise_isotropic_projection(self):
        """Check: pointwise isotropic projection."""
        # two points, stacked [p1; p2]: point 0 = (3, 4), point 1 = (0.3, 0.4)
        out = project_pointwise_l2_ball((3.0, 0.3, 4.0, 0.4), 1.0, components=2)
        self.assertAlmostEqual(out[0], 0.6)
        self.assertAlmostEqual(out[2], 0.8)
        self.assertEqual((out[1], out[3]), (0.3, 0.4))


class ProximalGradientTests(unittest.TestCase):
    """Tests for ProximalGradientTests."""

    def _solve(self, solver, matrix, y, alpha, **kw):
        """Check:  solve."""
        n = len(matrix[0])

        def value(x):
            r = [
                sum(a * b for a, b in zip(row, x, strict=True)) - yi
                for row, yi in zip(matrix, y, strict=False)
            ]
            return 0.5 * sum(v * v for v in r)

        def grad(x):
            r = [
                sum(a * b for a, b in zip(row, x, strict=True)) - yi
                for row, yi in zip(matrix, y, strict=False)
            ]
            return tuple(
                sum(matrix[k][i] * r[k] for k in range(len(y))) for i in range(n)
            )

        return solver(
            value,
            grad,
            lambda x: alpha * sum(map(abs, x)),
            lambda v, t: soft_threshold(v, t * alpha),
            (0.0,) * n,
            **kw,
        )

    def test_ista_fista_match_enumeration(self):
        """Check: ista fista match enumeration."""
        matrix, y = _random_problem(6, 3, 5)
        alpha = 0.7
        exact = _lasso_enumerate(matrix, y, alpha)
        for solver in (ista, fista):
            res = self._solve(
                solver, matrix, y, alpha, tolerance=1e-10, max_iterations=20000
            )
            self.assertTrue(res.converged, res.stop_reason)
            self.assertAlmostEqual(res.objective, exact[0], places=9)
            for a, b in zip(res.solution, exact[1], strict=True):
                self.assertAlmostEqual(a, b, places=6)
            self.assertLessEqual(res.gradient_mapping_norm, 1e-10)

    def test_ista_monotone_and_max_iterations_not_converged(self):
        """Check: ista monotone and max iterations not converged."""
        matrix, y = _random_problem(6, 3, 6)
        res = self._solve(ista, matrix, y, 0.1, max_iterations=3, tolerance=1e-14)
        self.assertFalse(res.converged)
        self.assertEqual(res.stop_reason, "max_iterations")
        hist = res.objective_history
        self.assertTrue(
            all(b <= a + 1e-12 for a, b in zip(hist, hist[1:], strict=False))
        )

    def test_fixed_lipschitz(self):
        """Check: fixed lipschitz."""
        matrix, y = _random_problem(5, 3, 7)
        big = sum(v * v for row in matrix for v in row)  # Frobenius^2 >= ||A||^2
        res = self._solve(
            fista, matrix, y, 0.5, lipschitz=big, tolerance=1e-9, max_iterations=20000
        )
        self.assertTrue(res.converged)
        self.assertEqual(res.backtracks, 0)

    def test_nan_callback_reported(self):
        """Check: nan callback reported."""
        res = ista(
            lambda x: math.nan, lambda x: x, lambda x: 0.0, lambda v, t: v, (1.0,)
        )
        self.assertEqual(res.stop_reason, "non_finite")
        self.assertFalse(res.converged)

    def test_invalid_inputs(self):
        """Check: invalid inputs."""
        with self.assertRaises(ValueError):
            ista(lambda x: 0.0, lambda x: x, lambda x: 0.0, lambda v, t: v, (math.nan,))
        with self.assertRaises(ValueError):
            ista(
                lambda x: 0.0,
                lambda x: x,
                lambda x: 0.0,
                lambda v, t: v,
                (1.0,),
                backtrack_factor=1.5,
            )


class LassoTests(unittest.TestCase):
    """Tests for LassoTests."""

    def test_scalar_soft_threshold_solution(self):
        """Check: scalar soft threshold solution."""
        # A = I: x = soft(y, alpha)
        res = l1_regularized_least_squares(IdentityOperator(3), (3.0, -0.5, 1.5), 1.0)
        for a, b in zip(res.solution, (2.0, 0.0, 0.5), strict=True):
            self.assertAlmostEqual(a, b, places=7)
        self.assertEqual(res.support, (0, 2))

    def test_zero_threshold(self):
        """Check: zero threshold."""
        matrix, y = _random_problem(8, 4, 11)
        op = DenseOperator(matrix)
        amax = lasso_zero_threshold(op, y)
        expected = max(
            abs(sum(matrix[k][i] * y[k] for k in range(8))) for i in range(4)
        )
        self.assertAlmostEqual(amax, expected)
        res = l1_regularized_least_squares(op, y, amax * 1.0000001)
        self.assertTrue(all(v == 0.0 for v in res.solution))
        res2 = l1_regularized_least_squares(op, y, amax * 0.9)
        self.assertTrue(any(v != 0.0 for v in res2.solution))

    def test_whitening_changes_solution_consistently(self):
        """Check: whitening changes solution consistently."""
        op = IdentityOperator(2)
        noise = DiagonalCovariance((4.0, 1.0))  # W = diag(1/2, 1)
        res = l1_regularized_least_squares(op, (4.0, 4.0), 0.5, noise=noise)
        # separable: x1 = soft(4, 0.5/0.25)=2, x2 = soft(4, 0.5)=3.5
        self.assertAlmostEqual(res.solution[0], 2.0, places=6)
        self.assertAlmostEqual(res.solution[1], 3.5, places=6)

    def test_nonnegative_and_box_kkt(self):
        """Check: nonnegative and box kkt."""
        matrix, y = _random_problem(8, 4, 3)
        op = DenseOperator(matrix)
        alpha = 0.3
        res = l1_regularized_least_squares(
            op, y, alpha, nonnegative=True, tolerance=1e-11, max_iterations=20000
        )
        self.assertTrue(res.converged)
        grad = [sum(matrix[k][i] * res.residual[k] for k in range(8)) for i in range(4)]
        for x, g in zip(res.solution, grad, strict=True):
            self.assertGreaterEqual(x, 0.0)
            if x > 0:
                self.assertAlmostEqual(g + alpha, 0.0, places=6)
            else:
                self.assertGreaterEqual(g + alpha, -1e-6)
        box = l1_regularized_least_squares(
            op, y, alpha, bounds=(-0.1, 0.1), tolerance=1e-11, max_iterations=20000
        )
        self.assertTrue(box.converged)
        for x, g in zip(
            box.solution,
            [sum(matrix[k][i] * box.residual[k] for k in range(8)) for i in range(4)],
            strict=True,
        ):
            self.assertTrue(-0.1 - 1e-12 <= x <= 0.1 + 1e-12)
            if x >= 0.1 - 1e-12:
                self.assertLessEqual(g + alpha, 1e-6)  # pushes upward
            elif x <= -0.1 + 1e-12:
                self.assertGreaterEqual(g - alpha, -1e-6)

    def test_ista_equals_fista_objective(self):
        """Check: ista equals fista objective."""
        matrix, y = _random_problem(10, 6, 21)
        a = l1_regularized_least_squares(
            DenseOperator(matrix),
            y,
            0.5,
            method="ista",
            tolerance=1e-10,
            max_iterations=50000,
        )
        b = l1_regularized_least_squares(
            DenseOperator(matrix),
            y,
            0.5,
            method="fista",
            tolerance=1e-10,
            max_iterations=50000,
        )
        self.assertTrue(a.converged and b.converged)
        self.assertAlmostEqual(a.objective, b.objective, places=9)

    def test_debias_is_separate(self):
        """Check: debias is separate."""
        res = l1_regularized_least_squares(
            IdentityOperator(3), (3.0, -0.5, 1.5), 1.0, debias=True
        )
        for a, b in zip(res.debiased_solution, (3.0, 0.0, 1.5), strict=True):
            self.assertAlmostEqual(a, b, places=7)
        self.assertAlmostEqual(res.solution[0], 2.0, places=6)
        with self.assertRaises(ValueError):
            l1_regularized_least_squares(
                IdentityOperator(2), (1.0, 1.0), 0.1, nonnegative=True, debias=True
            )

    def test_invalid(self):
        """Check: invalid."""
        op = IdentityOperator(2)
        with self.assertRaises(ValueError):
            l1_regularized_least_squares(op, (1.0, 1.0), -0.1)
        with self.assertRaises(ValueError):
            l1_regularized_least_squares(op, (1.0, math.nan), 0.1)
        with self.assertRaises(ValueError):
            l1_regularized_least_squares(op, (1.0, 1.0), 0.1, bounds=(1.0, 0.0))
        with self.assertRaises(ValueError):
            l1_regularized_least_squares(op, (1.0, 1.0), 0.1, method="newton")
        res = l1_regularized_least_squares(op, (1.0, 1.0), 0.0)
        self.assertAlmostEqual(res.solution[0], 1.0, places=6)


def _two_point_tv(a, b, alpha):
    """Check:  two point tv."""
    if abs(b - a) <= 2 * alpha:
        mean = 0.5 * (a + b)
        return (mean, mean)
    s = math.copysign(1.0, b - a)
    return (a + alpha * s, b - alpha * s)


class TotalVariationTests(unittest.TestCase):
    """Tests for TotalVariationTests."""

    def test_constant_has_zero_tv_and_boundary(self):
        """Check: constant has zero tv and boundary."""
        self.assertEqual(total_variation_value((2.0, 2.0, 2.0)), 0.0)
        self.assertEqual(total_variation_value((0.0, 1.0, 3.0)), 3.0)
        per = FirstDifferenceOperator(3, boundary="periodic")
        self.assertEqual(total_variation_value((0.0, 1.0, 3.0), per), 6.0)
        grid = GridGradientOperator(2, 2)
        const = total_variation_value((5.0,) * 4, grid)
        self.assertEqual(const, 0.0)
        # isotropic vs anisotropic on a corner: u = [0,3;4,0]
        iso = total_variation_value((0.0, 3.0, 4.0, 0.0), grid, isotropic=True)
        ani = total_variation_value((0.0, 3.0, 4.0, 0.0), grid, isotropic=False)
        self.assertAlmostEqual(iso, 12.0)
        self.assertAlmostEqual(ani, 14.0)
        self.assertGreater(ani, iso)

    def test_adjoint_consistency(self):
        """Check: adjoint consistency."""
        for op in (FirstDifferenceOperator(7), GridGradientOperator(3, 4)):
            n = op.shape[1]
            x = tuple(math.sin(i + 1) for i in range(n))
            y = tuple(math.cos(i + 2) for i in range(op.shape[0]))
            self.assertLess(adjoint_test(op, x, y).relative_error, 1e-13)

    def test_two_point_closed_form_both_methods(self):
        """Check: two point closed form both methods."""
        for y, alpha in (((1.0, 4.0), 0.5), ((1.0, 1.8), 0.5), ((3.0, -1.0), 1.0)):
            exact = _two_point_tv(*y, alpha)
            op = IdentityOperator(2)
            for method in ("pdhg", "admm"):
                res = total_variation_regularized_least_squares(
                    op,
                    y,
                    alpha,
                    gradient=FirstDifferenceOperator(2),
                    method=method,
                    tolerance_absolute=1e-10,
                    tolerance_relative=1e-9,
                    max_iterations=50000,
                )
                self.assertTrue(res.converged, (method, res.stop_reason))
                for a, b in zip(res.solution, exact, strict=True):
                    self.assertAlmostEqual(a, b, places=5)

    def test_pdhg_admm_agree_on_1d_deblur(self):
        """Check: pdhg admm agree on 1d deblur."""
        n = 20
        truth = [0.0] * 7 + [1.0] * 7 + [0.3] * 6
        conv = Convolution1DOperator((0.25, 0.5, 0.25), n, boundary="reflect")
        rng = random.Random(4)
        y = tuple(a + 0.02 * rng.gauss(0, 1) for a in conv.matvec(truth))
        alpha = 0.05
        p = total_variation_regularized_least_squares(
            conv,
            y,
            alpha,
            method="pdhg",
            tolerance_absolute=1e-9,
            tolerance_relative=1e-8,
            max_iterations=200000,
        )
        a = total_variation_regularized_least_squares(
            conv,
            y,
            alpha,
            method="admm",
            tolerance_absolute=1e-10,
            tolerance_relative=1e-9,
            max_iterations=5000,
        )
        self.assertTrue(p.converged and a.converged, (p.stop_reason, a.stop_reason))
        self.assertAlmostEqual(p.objective, a.objective, delta=1e-6 * (1 + a.objective))
        self.assertGreaterEqual(a.inner_iterations, 1)
        self.assertEqual(a.inner_failures, 0)
        # recomputed objective equals the solver's own at the solution
        self.assertAlmostEqual(p.objective, p.optimization.objective, places=9)
        # TV beats the unregularised solution's TV
        self.assertLess(p.total_variation, total_variation_value(y) + 1e-9)

    def test_bounds_pdhg_and_admm_refusal(self):
        """Check: bounds pdhg and admm refusal."""
        res = total_variation_regularized_least_squares(
            IdentityOperator(2),
            (1.0, 4.0),
            0.5,
            bounds=(0.0, 3.0),
            tolerance_absolute=1e-10,
            tolerance_relative=1e-9,
            max_iterations=50000,
        )
        self.assertTrue(res.converged)
        self.assertLessEqual(max(res.solution), 3.0 + 1e-12)
        with self.assertRaises(ValueError):
            total_variation_regularized_least_squares(
                IdentityOperator(2), (1.0, 4.0), 0.5, bounds=(0.0, 3.0), method="admm"
            )

    def test_2d_isotropic_small(self):
        """Check: 2d isotropic small."""
        grid = GridGradientOperator(3, 3)
        y = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        res = total_variation_regularized_least_squares(
            IdentityOperator(9),
            y,
            0.2,
            gradient=grid,
            tolerance_absolute=1e-9,
            tolerance_relative=1e-8,
            max_iterations=100000,
        )
        self.assertTrue(res.converged)
        adm = total_variation_regularized_least_squares(
            IdentityOperator(9),
            y,
            0.2,
            gradient=grid,
            method="admm",
            tolerance_absolute=1e-10,
            tolerance_relative=1e-9,
        )
        self.assertTrue(adm.converged)
        self.assertAlmostEqual(res.objective, adm.objective, places=6)

    def test_step_and_input_validation(self):
        """Check: step and input validation."""
        op = IdentityOperator(2)
        with self.assertRaises(ValueError):
            total_variation_regularized_least_squares(op, (1.0, 2.0), -1.0)
        with self.assertRaises(ValueError):
            total_variation_regularized_least_squares(op, (1.0, math.nan), 1.0)
        with self.assertRaises(ValueError):
            total_variation_regularized_least_squares(
                op, (1.0, 2.0), 1.0, method="gradient"
            )
        self.assertAlmostEqual(frobenius_norm_bound(op), math.sqrt(2.0))

    def test_max_iterations_not_converged(self):
        """Check: max iterations not converged."""
        res = total_variation_regularized_least_squares(
            IdentityOperator(2), (1.0, 4.0), 0.5, max_iterations=2
        )
        self.assertFalse(res.converged)
        self.assertEqual(res.stop_reason, "max_iterations")


class PDHGAdmmGenericTests(unittest.TestCase):
    """Tests for PDHGAdmmGenericTests."""

    def test_step_condition_violation_rejected(self):
        """Check: step condition violation rejected."""
        op = DenseOperator(((2.0, 0.0), (0.0, 1.0)))
        ident = lambda v, t: tuple(v)  # noqa: E731
        with self.assertRaises(ValueError):
            primal_dual_hybrid_gradient(
                op,
                ident,
                ident,
                (0.0, 0.0),
                operator_norm_bound=2.0,
                tau=0.5,
                sigma=0.5,
            )
        with self.assertRaises(ValueError):
            primal_dual_hybrid_gradient(
                op,
                ident,
                ident,
                (0.0, 0.0),
                operator_norm_bound=2.0,
                tau=0.5,
                sigma=0.5,
                theta=2.0,
            )
        with self.assertRaises(ValueError):
            primal_dual_hybrid_gradient(
                op, ident, ident, (0.0, 0.0), operator_norm_bound=-1.0
            )
        ok = primal_dual_hybrid_gradient(
            op,
            ident,
            ident,
            (1.0, 1.0),
            operator_norm_bound=2.0,
            tau=0.2,
            sigma=0.2,
            max_iterations=2,
        )
        self.assertLess(ok.step_product, 1.0)
        self.assertFalse(ok.converged)

    def test_pdhg_scalar_l1(self):
        """Check: pdhg scalar l1."""
        # min 1/2 (x-3)^2 + |x| via K = I, g = |.|; prox_f(v) = (v + 3 t)/(1 + t)
        op = IdentityOperator(1)
        res = primal_dual_hybrid_gradient(
            op,
            lambda v, t: ((v[0] + 3.0 * t) / (1.0 + t),),
            lambda q, s: project_linf_ball(q, 1.0),
            (0.0,),
            operator_norm_bound=1.0,
            tolerance_absolute=1e-12,
            tolerance_relative=1e-10,
        )
        self.assertTrue(res.converged)
        self.assertAlmostEqual(res.solution[0], 2.0, places=7)

    def test_admm_scalar_l1(self):
        """Check: admm scalar l1."""
        # min 1/2 (x-3)^2 + |z|, x = z. x-update: (3 + rho v)/(1 + rho)
        res = admm(
            IdentityOperator(1),
            lambda v, rho, old: ((3.0 + rho * v[0]) / (1.0 + rho),),
            lambda w, t: soft_threshold(w, t),
            (0.0,),
            tolerance_absolute=1e-12,
            tolerance_relative=1e-10,
            adapt_rho=True,
        )
        self.assertTrue(res.converged)
        self.assertAlmostEqual(res.solution[0], 2.0, places=7)
        with self.assertRaises(ValueError):
            admm(
                IdentityOperator(1), lambda v, r, o: v, lambda w, t: w, (0.0,), rho=-1.0
            )


if __name__ == "__main__":
    unittest.main()
