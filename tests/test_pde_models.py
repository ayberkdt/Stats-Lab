"""Tests for the 1D finite-volume diffusion model with tangent and adjoint.

Oracles: the dense pivoted ``lu_solve`` of ``probstat.linalg`` for the Thomas
solver, closed-form manufactured solutions for the discretization (observed
second-order convergence), central finite differences for the derivatives, and
the dot-product identity ``w'(J v) = (J' w)' v`` for adjoint consistency.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.inverse.pde_models import (
    EllipticDiffusion1D,
    adjoint_gradient,
    solve_tridiagonal,
    solve_tridiagonal_transpose,
)
from probstat.linalg import lu_solve


def dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Return the Euclidean inner product."""
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def random_vector(rng: random.Random, n: int, scale: float = 1.0) -> tuple[float, ...]:
    """Return a uniform random vector."""
    return tuple(rng.uniform(-scale, scale) for _ in range(n))


class ThomasTests(unittest.TestCase):
    """Tridiagonal solver against the dense pivoted solver."""

    def test_matches_lu_solve_and_transpose(self) -> None:
        """Random diagonally dominant system, solve and transpose solve."""
        rng = random.Random(7)
        n = 30
        lower = [rng.uniform(-1, 1) for _ in range(n - 1)]
        upper = [rng.uniform(-1, 1) for _ in range(n - 1)]
        diag = [3.0 + rng.uniform(0, 1) for _ in range(n)]
        rhs = [rng.uniform(-1, 1) for _ in range(n)]
        dense = [[0.0] * n for _ in range(n)]
        for i in range(n):
            dense[i][i] = diag[i]
            if i < n - 1:
                dense[i][i + 1] = upper[i]
                dense[i + 1][i] = lower[i]
        matrix = tuple(tuple(r) for r in dense)
        transposed = tuple(tuple(dense[j][i] for j in range(n)) for i in range(n))
        got = solve_tridiagonal(lower, diag, upper, rhs)
        ref = lu_solve(matrix, tuple(rhs))
        got_t = solve_tridiagonal_transpose(lower, diag, upper, rhs)
        ref_t = lu_solve(transposed, tuple(rhs))
        for a, b in zip(got, ref, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-13)
        for a, b in zip(got_t, ref_t, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-13)

    def test_zero_pivot_and_bad_input_raise(self) -> None:
        """Singular leading minor, wrong lengths, and non-finite entries fail."""
        with self.assertRaises(ValueError):
            solve_tridiagonal([1.0], [0.0, 1.0], [1.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            solve_tridiagonal([1.0], [1.0, 1.0], [1.0], [1.0, 1.0])  # pivot 1-1=0
        with self.assertRaises(ValueError):
            solve_tridiagonal([], [1.0, 1.0], [], [1.0, 1.0])
        with self.assertRaises(ValueError):
            solve_tridiagonal([0.0], [1.0, math.nan], [0.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            solve_tridiagonal([0.0], [1.0, 1.0], [0.0], [1.0])


def smooth_problem(
    n: int,
) -> tuple[EllipticDiffusion1D, tuple[float, ...], list[float]]:
    """Manufactured problem u = sin(pi s), k = exp(s / 2) on (0, 1)."""

    def q(s: float) -> float:
        k = math.exp(0.5 * s)
        return -0.5 * k * math.pi * math.cos(math.pi * s) + k * math.pi**2 * math.sin(
            math.pi * s
        )

    model = EllipticDiffusion1D(n, sensors=[0.3, 0.5], source=q)
    centres = [(i + 0.5) / n for i in range(n)]
    x = tuple(0.5 * s for s in centres)
    exact = [math.sin(math.pi * s) for s in centres]
    return model, x, exact


def error_of(
    model: EllipticDiffusion1D, x: tuple[float, ...], exact: list[float]
) -> float:
    """Return the maximum cell-centre error."""
    u = model.state(x)
    return max(abs(a - b) for a, b in zip(u, exact, strict=True))


class DiscretizationTests(unittest.TestCase):
    """Convergence and conservation of the finite-volume scheme."""

    def test_constant_k_quadratic_converges_second_order(self) -> None:
        """-k u'' = q constant: quadratic solution, observed O(h^2) error."""
        k, q0 = 2.0, 3.0

        def exact(s: float) -> float:
            return q0 / (2 * k) * s * (1.0 - s)  # u(0)=u(1)=0

        errors = []
        for n in (16, 32, 64):
            model = EllipticDiffusion1D(n, sensors=[0.5], source=q0)
            u = model.state([math.log(k)] * n)
            errors.append(max(abs(u[i] - exact((i + 0.5) / n)) for i in range(n)))
        orders = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
        for order in orders:
            self.assertGreater(order, 1.8)
            self.assertLess(order, 2.2)

    def test_smooth_variable_k_converges_second_order(self) -> None:
        """Manufactured sin solution with k = exp(s/2)."""
        errors = [error_of(*smooth_problem(n)) for n in (16, 32, 64)]
        for i in range(2):
            order = math.log2(errors[i] / errors[i + 1])
            self.assertGreater(order, 1.8)
            self.assertLess(order, 2.2)

    def test_flux_conservation(self) -> None:
        """Cell balances and the global balance hold to rounding."""
        model, x, _ = smooth_problem(48)
        balance = model.flux_balance(x)
        scale = max(abs(f) for f in model.face_fluxes(x))
        self.assertLess(balance.cell_residual, 1e-12 * scale)
        self.assertLess(balance.global_imbalance, 1e-12 * scale)
        self.assertAlmostEqual(
            balance.right_flux - balance.left_flux, balance.total_source, delta=1e-12
        )

    def test_linear_solve_residual_and_positivity(self) -> None:
        """K u = f is satisfied by the stored state; K is SPD (positive diag)."""
        rng = random.Random(3)
        model, _, _ = smooth_problem(40)
        x = random_vector(rng, 40, 1.0)
        lo, di, up, rhs = model.system(x)
        u = model.state(x)
        for i in range(40):
            row = di[i] * u[i]
            if i > 0:
                row += lo[i - 1] * u[i - 1]
            if i < 39:
                row += up[i] * u[i + 1]
            self.assertAlmostEqual(row, rhs[i], delta=1e-12 * (1 + abs(rhs[i])))
        self.assertTrue(all(d > 0 for d in di))
        self.assertTrue(all(abs(di[i]) > abs(lo[i - 1]) for i in range(1, 40)))

    def test_nonzero_boundary_values_are_reproduced(self) -> None:
        """Linear profile for constant k, q = 0, boundary (1, 3), exact at sensors."""
        model = EllipticDiffusion1D(20, sensors=[0.01, 0.5, 0.99], boundary=(1.0, 3.0))
        got = model.value([0.7] * 20)
        for s, g in zip(model.sensors, got, strict=True):
            self.assertAlmostEqual(g, 1.0 + 2.0 * s, delta=1e-12)


class DerivativeTests(unittest.TestCase):
    """Tangent, adjoint and gradient consistency."""

    def setUp(self) -> None:
        """Build a 24-cell model with nonzero boundary values."""
        self.n = 24
        self.model = EllipticDiffusion1D(
            self.n,
            sensors=[0.02, 0.2, 0.45, 0.7, 0.97],
            source=lambda s: 1.0 + math.sin(3 * s),
            boundary=(0.5, -0.25),
        )
        self.rng = random.Random(11)
        self.x = random_vector(self.rng, self.n, 0.8)

    def finite_difference(self, v: tuple[float, ...], eps: float) -> tuple[float, ...]:
        """Central difference of value() along v."""
        plus = self.model.value([a + eps * b for a, b in zip(self.x, v, strict=True)])
        minus = self.model.value([a - eps * b for a, b in zip(self.x, v, strict=True)])
        return tuple((a - b) / (2 * eps) for a, b in zip(plus, minus, strict=True))

    def test_jvp_matches_central_differences(self) -> None:
        """Error of the central difference is O(eps^2) around the exact JVP."""
        v = random_vector(self.rng, self.n)
        exact = self.model.jvp(self.x, v)
        scale = max(abs(a) for a in exact)
        errs = []
        for eps in (1e-2, 5e-3):
            fd = self.finite_difference(v, eps)
            errs.append(max(abs(a - b) for a, b in zip(fd, exact, strict=True)) / scale)
        self.assertLess(errs[1], 1e-4)
        self.assertGreater(errs[0] / errs[1], 3.0)  # second-order decay (ratio 4)
        fd = self.finite_difference(v, 1e-5)
        err = max(abs(a - b) for a, b in zip(fd, exact, strict=True)) / scale
        self.assertLess(err, 1e-8)

    def test_dot_test(self) -> None:
        """w'(J v) == (J' w)' v to rounding."""
        for _ in range(5):
            v = random_vector(self.rng, self.n)
            w = random_vector(self.rng, 5)
            lhs = dot(w, self.model.jvp(self.x, v))
            rhs = dot(self.model.vjp(self.x, w), v)
            self.assertLess(abs(lhs - rhs), 1e-12 * (abs(lhs) + abs(rhs) + 1e-3))

    def test_taylor_remainder_second_order(self) -> None:
        """|G(x+tv) - G(x) - t J v| decays like t^2."""
        v = random_vector(self.rng, self.n)
        g0 = self.model.value(self.x)
        jv = self.model.jvp(self.x, v)
        rems = []
        for t in (4e-2, 2e-2, 1e-2):
            gt = self.model.value([a + t * b for a, b in zip(self.x, v, strict=True)])
            rems.append(
                max(abs(g - h - t * j) for g, h, j in zip(gt, g0, jv, strict=True))
            )
        for i in range(2):
            self.assertGreater(math.log2(rems[i] / rems[i + 1]), 1.9)
            self.assertLess(math.log2(rems[i] / rems[i + 1]), 2.1)

    def test_adjoint_gradient_equals_finite_differences(self) -> None:
        """Gradient of the weighted misfit against central differences."""
        data = tuple(
            g + 0.01 for g in self.model.value(random_vector(self.rng, self.n))
        )
        noise = [0.05, 0.1, 0.05, 0.2, 0.1]

        def objective(x: tuple[float, ...]) -> float:
            g = self.model.value(x)
            return 0.5 * sum(
                ((a - b) / s) ** 2 for a, b, s in zip(g, data, noise, strict=True)
            )

        result = adjoint_gradient(self.model, self.x, data, noise)
        self.assertAlmostEqual(result.objective, objective(self.x), delta=1e-10)
        eps = 1e-5
        scale = max(abs(g) for g in result.gradient)
        for i in range(self.n):
            xp = list(self.x)
            xm = list(self.x)
            xp[i] += eps
            xm[i] -= eps
            fd = (objective(tuple(xp)) - objective(tuple(xm))) / (2 * eps)
            self.assertAlmostEqual(result.gradient[i], fd, delta=1e-6 * scale)

    def test_gradient_cost_is_independent_of_n(self) -> None:
        """One forward and one adjoint solve; the Jacobian needs n tangent solves."""
        n = 200
        model = EllipticDiffusion1D(n, sensors=[0.25, 0.5, 0.75], source=1.0)
        x = random_vector(random.Random(5), n, 0.5)
        adjoint_gradient(model, x, (0.0, 0.0, 0.0), 1.0)
        counts = model.counts
        self.assertEqual((counts.forward, counts.tangent, counts.adjoint), (1, 0, 1))
        model.reset_counts()
        columns = []
        for i in range(n):
            e = [0.0] * n
            e[i] = 1.0
            columns.append(model.jvp(x, e))
        counts = model.counts
        self.assertEqual((counts.forward, counts.tangent, counts.adjoint), (0, n, 0))
        # the full Jacobian agrees with the adjoint rows
        w = (1.0, -2.0, 0.5)
        via_adjoint = model.vjp(x, w)
        via_jacobian = [dot(w, col) for col in columns]
        for a, b in zip(via_adjoint, via_jacobian, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-11 * (1 + abs(b)))


class ValidationTests(unittest.TestCase):
    """Invalid inputs fail loudly."""

    def test_invalid_construction(self) -> None:
        """Too few cells, bad sensors, bad lengths and sources."""
        with self.assertRaises(ValueError):
            EllipticDiffusion1D(1, sensors=[0.5])
        with self.assertRaises(TypeError):
            EllipticDiffusion1D(4.0, sensors=[0.5])  # type: ignore[arg-type]
        for bad in (0.0, 1.0, -0.1, 1.5, math.nan):
            with self.assertRaises(ValueError):
                EllipticDiffusion1D(8, sensors=[bad])
        with self.assertRaises(ValueError):
            EllipticDiffusion1D(8, sensors=[])
        with self.assertRaises(ValueError):
            EllipticDiffusion1D(8, sensors=[0.5], length=0.0)
        with self.assertRaises(ValueError):
            EllipticDiffusion1D(8, sensors=[0.5], source=[1.0] * 7)
        with self.assertRaises(ValueError):
            EllipticDiffusion1D(8, sensors=[0.5], boundary=(0.0, math.inf))

    def test_invalid_parameters(self) -> None:
        """Non-finite x, overflowing exp(x), wrong lengths."""
        model = EllipticDiffusion1D(8, sensors=[0.5], source=1.0)
        for bad in (math.nan, math.inf, 800.0, -800.0, 400.0):
            with self.assertRaisesRegex(ValueError, "x|finite"):
                model.state([bad] + [0.0] * 7)
        with self.assertRaises(ValueError):
            model.state([0.0] * 7)
        with self.assertRaises(ValueError):
            model.jvp([0.0] * 8, [0.0] * 7)
        with self.assertRaises(ValueError):
            model.vjp([0.0] * 8, [0.0, 0.0])
        with self.assertRaises(ValueError):
            adjoint_gradient(model, [0.0] * 8, (0.0,), -1.0)


if __name__ == "__main__":
    unittest.main()
