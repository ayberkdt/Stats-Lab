"""Tests of the 2D elliptic model and the explicit-Euler stability case."""

from __future__ import annotations

import math
import os
import unittest

from probstat.foundations.random_streams import resolve_stream
from probstat.inverse.gauss_newton import gauss_newton_krylov
from probstat.inverse.pde_models import adjoint_gradient
from probstat.inverse.pde_models_2d import (
    EllipticDiffusion2D,
    HeatExplicitEuler1D,
    LinearSolverError,
    UnstableTimeStepError,
    explicit_amplification_factors,
    explicit_heat_inverse_problem,
    explicit_mode_growth,
    explicit_stability,
    explicit_stability_limit,
    synthetic_fine_grid_data,
)
from probstat.linalg import (
    ScaledIdentityCovariance,
    adjoint_test,
    materialize,
    symmetric_eigen,
)

PI = math.pi


def _random_vector(size: int, seed: int, scale: float = 1.0) -> tuple[float, ...]:
    """Return a reproducible uniform vector in ``[-scale, scale]``."""
    rng = resolve_stream(seed).generator()
    return tuple(rng.uniform(-scale, scale) for _ in range(size))


def _u_exact(x: float, y: float) -> float:
    """Manufactured solution ``sin(pi x) sin(pi y)``."""
    return math.sin(PI * x) * math.sin(PI * y)


def _k_smooth(x: float, y: float) -> float:
    """Smooth positive conductivity ``1 + x y``."""
    return 1.0 + x * y


def _q_const(x: float, y: float) -> float:
    """Source of the manufactured solution for ``k = 1``."""
    return 2.0 * PI**2 * _u_exact(x, y)


def _q_smooth(x: float, y: float) -> float:
    """Source of the manufactured solution for ``k = 1 + x y``."""
    return (
        -y * PI * math.cos(PI * x) * math.sin(PI * y)
        - x * PI * math.sin(PI * x) * math.cos(PI * y)
        + 2.0 * PI**2 * _k_smooth(x, y) * _u_exact(x, y)
    )


def _observed_error(
    n: int,
    k: object,
    q: object,
    solver: str,
) -> float:
    """Return the max cell-centre error of the manufactured problem."""
    model = EllipticDiffusion2D(
        n,
        n,
        sensors=[(0.5, 0.5)],
        source=q,
        solver=solver,  # type: ignore[arg-type]
    )
    x = [math.log(k(a, b)) for a, b in model.cell_centres]  # type: ignore[operator]
    u = model.state(x)
    return max(
        abs(v - _u_exact(a, b)) for v, (a, b) in zip(u, model.cell_centres, strict=True)
    )


def _sensor_grid(m: int) -> list[tuple[float, float]]:
    """Return an ``m x m`` interior sensor grid on the unit square."""
    return [((a + 1) / (m + 1), (b + 1) / (m + 1)) for a in range(m) for b in range(m)]


class ClosedFormTests(unittest.TestCase):
    """Exact discrete solution and manufactured-solution convergence."""

    def test_two_by_two_exact(self) -> None:
        """Exact 2x2 case, k = 1, q = 1: K = [[6,-1,-1,0],...], u = 1/16."""
        for solver in ("pcg", "banded_cholesky"):
            model = EllipticDiffusion2D(
                2, 2, sensors=[(0.25, 0.25)], source=1.0, solver=solver
            )
            u = model.state([0.0] * 4)
            for value in u:
                self.assertAlmostEqual(value, 0.0625, places=12)
            self.assertAlmostEqual(model.value([0.0] * 4)[0], 0.0625, places=12)

    def test_solvers_agree(self) -> None:
        """PCG and banded Cholesky give the same state for a rough field."""
        x = _random_vector(36, 1, 1.5)
        states = []
        for solver in ("pcg", "banded_cholesky"):
            model = EllipticDiffusion2D(
                6, 6, sensors=[(0.3, 0.6)], source=_q_const, solver=solver
            )
            states.append(model.state(x))
            self.assertTrue(model.last_solve is not None and model.last_solve.converged)
        scale = max(abs(v) for v in states[1])
        for a, b in zip(*states, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9 * scale)

    def test_second_order_convergence_constant_k(self) -> None:
        """Measured rate of the cell-centre error is 2 (k = 1)."""
        errors = [
            _observed_error(n, lambda a, b: 1.0, _q_const, "pcg") for n in (4, 8, 16)
        ]
        rates = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
        for rate in rates:
            self.assertGreater(rate, 1.85)
            self.assertLess(rate, 2.15)

    def test_second_order_convergence_variable_k(self) -> None:
        """Measured rate is 2 for k = 1 + x y (harmonic face means)."""
        errors = [
            _observed_error(n, _k_smooth, _q_smooth, "banded_cholesky")
            for n in (4, 8, 16)
        ]
        rates = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
        for rate in rates:
            self.assertGreater(rate, 1.85)
            self.assertLess(rate, 2.15)

    def test_flux_balance_and_residual(self) -> None:
        """Global flux balance holds to roundoff; the TRUE residual is recorded."""
        x = _random_vector(64, 2, 1.0)
        model = EllipticDiffusion2D(
            8, 8, sensors=[(0.5, 0.5)], source=_q_const, solver="pcg"
        )
        balance = model.flux_balance(x)
        self.assertLess(balance.imbalance, 1e-10 * abs(balance.total_source))
        record = model.last_solve
        assert record is not None
        self.assertTrue(record.converged)
        self.assertLessEqual(record.relative_residual, 1e-11)
        self.assertGreater(record.iterations, 0)

    def test_average_observation(self) -> None:
        """A window covering all cells returns the mean state; a tiny one a cell."""
        wide = EllipticDiffusion2D(
            2, 2, sensors=[(0.5, 0.5)], source=1.0, average_radius=0.5
        )
        self.assertAlmostEqual(wide.value([0.0] * 4)[0], 0.0625, places=12)
        x = _random_vector(16, 3, 1.0)
        tiny = EllipticDiffusion2D(
            4, 4, sensors=[(0.125, 0.375)], source=1.0, average_radius=0.01
        )
        self.assertAlmostEqual(tiny.value(x)[0], tiny.state(x)[4], places=14)


class DerivativeTests(unittest.TestCase):
    """Tangent, adjoint, finite differences and solve counts."""

    def _model(self, solver: str, average: float | None = None) -> EllipticDiffusion2D:
        """Return a 5x4 model with 4 sensors."""
        return EllipticDiffusion2D(
            5,
            4,
            sensors=[(0.3, 0.4), (0.7, 0.2), (0.5, 0.8), (0.15, 0.55)],
            source=_q_const,
            solver=solver,
            average_radius=average,
        )

    def test_dot_test(self) -> None:
        """w'(J v) = (J'w)'v for both solvers and observation types."""
        cases = (
            ("banded_cholesky", None, 1e-12),
            ("pcg", None, 1e-9),
            ("banded_cholesky", 0.2, 1e-12),
        )
        for solver, average, tol in cases:
            model = self._model(solver, average)
            x = _random_vector(20, 4, 1.0)
            v = _random_vector(20, 5)
            w = _random_vector(4, 6)
            jv = model.jvp(x, v)
            jtw = model.vjp(x, w)
            left = sum(a * b for a, b in zip(w, jv, strict=True))
            right = sum(a * b for a, b in zip(jtw, v, strict=True))
            scale = math.sqrt(sum(a * a for a in jv)) * math.sqrt(sum(a * a for a in w))
            self.assertLess(abs(left - right), tol * scale, (solver, average))

    def test_jvp_matches_finite_difference_and_taylor(self) -> None:
        """Central differences agree; the second-order remainder scales as eps^2."""
        model = self._model("banded_cholesky")
        x = _random_vector(20, 7, 1.0)
        v = _random_vector(20, 8)
        jv = model.jvp(x, v)
        errors = []
        for eps in (1e-2, 5e-3):
            plus = model.value([a + eps * b for a, b in zip(x, v, strict=True)])
            minus = model.value([a - eps * b for a, b in zip(x, v, strict=True)])
            fd = [(p - m) / (2 * eps) for p, m in zip(plus, minus, strict=True)]
            errors.append(max(abs(a - b) for a, b in zip(fd, jv, strict=True)))
        self.assertLess(errors[1], 1e-4 * max(abs(a) for a in jv))
        self.assertAlmostEqual(errors[0] / errors[1], 4.0, delta=0.4)

    def test_gradient_matches_finite_difference(self) -> None:
        """The adjoint misfit gradient equals central differences (all entries)."""
        model = self._model("banded_cholesky")
        x = _random_vector(20, 9, 0.8)
        truth = _random_vector(20, 10, 0.8)
        data = model.value(truth)
        sigma = 1e-3
        result = adjoint_gradient(model, x, data, sigma)
        eps = 1e-6
        for i in range(20):
            shifted = []
            for sign in (1.0, -1.0):
                y = list(x)
                y[i] += sign * eps
                r = [a - b for a, b in zip(model.value(y), data, strict=True)]
                shifted.append(0.5 * sum((c / sigma) ** 2 for c in r))
            fd = (shifted[0] - shifted[1]) / (2 * eps)
            self.assertAlmostEqual(result.gradient[i], fd, delta=1e-6 * abs(fd) + 1e-3)

    def test_one_gradient_costs_two_solves(self) -> None:
        """One value plus one vjp is 1 forward + 1 adjoint solve."""
        for n in (3, 6):
            model = EllipticDiffusion2D(
                n, n, sensors=[(0.5, 0.5), (0.3, 0.3)], source=1.0
            )
            x = _random_vector(n * n, 11, 0.5)
            model.reset_counts()
            adjoint_gradient(model, x, [0.0, 0.0], 1.0)
            counts = model.counts
            self.assertEqual(
                (counts.forward, counts.tangent, counts.adjoint), (1, 0, 1)
            )
        # The full Jacobian by tangents needs one solve per parameter.
        model.reset_counts()
        for i in range(36):
            e = [0.0] * 36
            e[i] = 1.0
            model.jvp(x, e)
        self.assertEqual(model.counts.tangent, 36)
        self.assertEqual(model.counts.forward, 0)  # cached from the loop above

    def test_cache_reused(self) -> None:
        """Repeated evaluation at the same x does not re-solve."""
        model = self._model("pcg")
        x = _random_vector(20, 12, 0.5)
        model.value(x)
        model.value(x)
        model.jvp(x, [0.1] * 20)
        self.assertEqual(model.counts.forward, 1)


class SolverFailureTests(unittest.TestCase):
    """Linear-solver failures are reported, never hidden."""

    def test_pcg_nonconvergence_is_reported(self) -> None:
        """A 2-iteration cap yields nan values, counted failures and raises."""
        model = EllipticDiffusion2D(
            8, 8, sensors=[(0.5, 0.5)], source=1.0, max_iterations=2
        )
        x = _random_vector(64, 13, 1.0)
        value = model.value(x)
        self.assertTrue(math.isnan(value[0]))
        self.assertEqual(model.linear_failures, 1)
        record = model.last_solve
        assert record is not None
        self.assertFalse(record.converged)
        self.assertGreater(record.relative_residual, 1e-6)
        with self.assertRaises(LinearSolverError):
            model.state(x)
        with self.assertRaises(LinearSolverError):
            model.jvp(x, [0.0] * 64)
        with self.assertRaises(LinearSolverError):
            model.vjp(x, [1.0])

    def test_gauss_newton_refuses_failed_start(self) -> None:
        """A failed forward evaluation at the start is a ValueError downstream."""
        model = EllipticDiffusion2D(
            8, 8, sensors=[(0.5, 0.5)], source=1.0, max_iterations=2
        )
        with self.assertRaises(ValueError):
            gauss_newton_krylov(model, [0.05], regularization=1.0)

    def test_cholesky_reports_residual(self) -> None:
        """The Cholesky solve records a roundoff-level true residual."""
        model = EllipticDiffusion2D(
            6, 6, sensors=[(0.5, 0.5)], source=1.0, solver="banded_cholesky"
        )
        model.state(_random_vector(36, 14, 2.0))
        record = model.last_solve
        assert record is not None
        self.assertTrue(record.converged)
        self.assertLess(record.relative_residual, 1e-12)
        self.assertEqual(record.iterations, 0)

    def test_validation(self) -> None:
        """Invalid input is rejected with TypeError or ValueError."""
        with self.assertRaises(ValueError):
            EllipticDiffusion2D(1, 4, sensors=[(0.5, 0.5)])
        with self.assertRaises(TypeError):
            EllipticDiffusion2D(4.0, 4, sensors=[(0.5, 0.5)])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            EllipticDiffusion2D(4, 4, sensors=[(1.0, 0.5)])
        with self.assertRaises(ValueError):
            EllipticDiffusion2D(4, 4, sensors=[])
        with self.assertRaises(ValueError):
            EllipticDiffusion2D(4, 4, sensors=[(0.5, 0.5)], solver="lu")
        with self.assertRaises(ValueError):
            EllipticDiffusion2D(4, 4, sensors=[(0.5, 0.5)], tolerance=0.0)
        with self.assertRaises(ValueError):
            EllipticDiffusion2D(4, 4, sensors=[(0.5, 0.5)], source=[1.0] * 3)
        with self.assertRaises(ValueError):
            EllipticDiffusion2D(
                4, 4, sensors=[(0.5, 0.5)], average_radius=0.01
            )  # window holds no cell centre
        model = EllipticDiffusion2D(3, 3, sensors=[(0.5, 0.5)], source=1.0)
        with self.assertRaises(ValueError):
            model.value([0.0] * 8)
        with self.assertRaises(ValueError):
            model.value([0.0] * 8 + [500.0])
        with self.assertRaises(ValueError):
            model.value([0.0] * 8 + [math.nan])
        vector_source = EllipticDiffusion2D(
            3, 3, sensors=[(0.5, 0.5)], source=[1.0] * 9
        )
        with self.assertRaises(ValueError):
            vector_source.refined(2)


class InverseRecoveryTests(unittest.TestCase):
    """Gauss-Newton recovery from data generated on a finer grid."""

    @staticmethod
    def _truth(x: float, y: float) -> float:
        """Smooth log-conductivity truth."""
        return 0.6 * math.cos(PI * x) * math.sin(PI * y)

    def test_recovery_without_inverse_crime(self) -> None:
        """4x4 inversion of 8x8 data: misfit at noise level, error well below prior."""
        model = EllipticDiffusion2D(
            4, 4, sensors=_sensor_grid(3), source=1.0, solver="banded_cholesky"
        )
        fine = synthetic_fine_grid_data(
            model, self._truth, noise_sd=5e-4, seed=3, refinement=2
        )
        self.assertFalse(fine.inverse_crime)
        self.assertEqual(fine.refinement, 2)
        # Discretization error is visible: coarse model on the coarse truth does
        # not reproduce the fine-grid clean data to noise level.
        coarse_clean = model.value(fine.truth_coarse)
        gap = max(abs(a - b) for a, b in zip(coarse_clean, fine.clean, strict=True))
        self.assertGreater(gap, 0.1 * fine.noise_sd)
        result = gauss_newton_krylov(
            model,
            fine.data,
            noise=ScaledIdentityCovariance(9, fine.noise_sd**2),
            regularization=10.0,
            max_iterations=30,
        )
        self.assertEqual(model.linear_failures, 0)
        self.assertTrue(result.converged, result.stop_reason)
        zero_error = math.sqrt(sum(v * v for v in fine.truth_coarse))
        error = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(result.estimate, fine.truth_coarse, strict=True)
            )
        )
        chi2 = (result.whitened_residual_norm**2) / 9
        self.assertLess(error / zero_error, 0.5)
        self.assertLess(chi2, 5.0)
        self.assertGreater(chi2, 0.2)

    def test_inverse_crime_is_flagged(self) -> None:
        """Data with refinement = 1 share the discretization and is flagged."""
        model = EllipticDiffusion2D(4, 4, sensors=_sensor_grid(3), source=1.0)
        data = synthetic_fine_grid_data(
            model, self._truth, noise_sd=1e-4, seed=1, refinement=1
        )
        self.assertTrue(data.inverse_crime)
        exact = model.value(data.truth_coarse)
        for a, b in zip(exact, data.clean, strict=True):
            self.assertAlmostEqual(a, b, places=12)

    def test_fine_data_validation_and_determinism(self) -> None:
        """Bad noise or refinement is rejected; the same seed gives the same data."""
        model = EllipticDiffusion2D(3, 3, sensors=[(0.5, 0.5)], source=1.0)
        with self.assertRaises(ValueError):
            synthetic_fine_grid_data(model, self._truth, noise_sd=0.0, seed=1)
        with self.assertRaises(ValueError):
            synthetic_fine_grid_data(
                model, self._truth, noise_sd=1e-3, seed=1, refinement=0
            )
        first = synthetic_fine_grid_data(model, self._truth, noise_sd=1e-3, seed=5)
        second = synthetic_fine_grid_data(model, self._truth, noise_sd=1e-3, seed=5)
        self.assertEqual(first.data, second.data)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowTests(unittest.TestCase):
    """Finer convergence study (opt in with PROBSTAT_SLOW=1)."""

    def test_rates_to_32(self) -> None:
        """Rates stay at 2 up to 32x32 for both problems and both solvers."""
        for k, q in ((lambda a, b: 1.0, _q_const), (_k_smooth, _q_smooth)):
            for solver in ("pcg", "banded_cholesky"):
                errors = [_observed_error(n, k, q, solver) for n in (8, 16, 32)]
                for i in range(2):
                    rate = math.log2(errors[i] / errors[i + 1])
                    self.assertGreater(rate, 1.9)
                    self.assertLess(rate, 2.1)


class ExplicitStabilityTests(unittest.TestCase):
    """Explicit Euler for the heat equation: limits, growth and the adjoint."""

    def test_limits(self) -> None:
        """h^2/(2 kappa) in 1D, h^2/(4 kappa) in 2D."""
        self.assertAlmostEqual(explicit_stability_limit(2.0, 0.1), 0.0025)
        self.assertAlmostEqual(explicit_stability_limit(2.0, 0.1, dimension=2), 0.00125)
        with self.assertRaises(ValueError):
            explicit_stability_limit(0.0, 0.1)
        with self.assertRaises(ValueError):
            explicit_stability_limit(1.0, 0.1, dimension=4)

    def test_amplification_factors_closed_form_and_eigenvalues(self) -> None:
        """For n = 1 the factor is 1 - 2 r; for n = 6 it is the spectrum of I - dt D."""
        self.assertAlmostEqual(
            explicit_amplification_factors(1.0, 0.5, 0.1, 1)[0], 1.0 - 2.0 * 0.4
        )
        n, kappa, h, dt = 6, 1.3, 0.2, 0.01
        r = kappa * dt / h**2
        matrix = [
            [
                (1.0 - 2.0 * r) if i == j else (r if abs(i - j) == 1 else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]
        eigenvalues, _ = symmetric_eigen(matrix)
        factors = sorted(explicit_amplification_factors(kappa, h, dt, n))
        for a, b in zip(eigenvalues, factors, strict=True):
            self.assertAlmostEqual(a, b, places=12)

    def test_two_dimensional_limit_is_stricter(self) -> None:
        """A step between h^2/(4k) and h^2/(2k): stable in 1D, unstable in 2D."""
        h, kappa, n = 0.05, 1.0, 19
        dt = 0.7 * explicit_stability_limit(kappa, h)
        one = explicit_stability(kappa, h, dt, n, dimension=1)
        two = explicit_stability(kappa, h, dt, n, dimension=2)
        self.assertTrue(one.stable)
        self.assertLessEqual(one.max_amplification, 1.0)
        self.assertFalse(two.stable)
        self.assertGreater(two.max_amplification, 1.0)
        self.assertEqual(len(two.factors), n * n)

    def _operator(
        self, ratio: float, *, steps: int = 60, **kwargs: object
    ) -> HeatExplicitEuler1D:
        """Return an operator whose step is ``ratio`` times the limit."""
        n, kappa = 15, 1.0
        h = 1.0 / (n + 1)
        dt = ratio * explicit_stability_limit(kappa, h)
        return HeatExplicitEuler1D(
            n,
            diffusivity=kappa,
            final_time=dt * steps,
            steps=steps,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_stable_step_damps_every_mode(self) -> None:
        """At 0.9 of the limit no mode grows; measured growth equals |g|^N."""
        op = self._operator(0.9)
        self.assertTrue(op.stable)
        for mode in range(1, 16):
            measured = explicit_mode_growth(op, mode)
            predicted = abs(op.stability.factors[mode - 1]) ** op.steps
            self.assertLessEqual(measured, 1.0 + 1e-12)
            self.assertAlmostEqual(measured, predicted, delta=1e-10)

    def test_unstable_step_amplifies_high_frequencies(self) -> None:
        """At 1.2 x limit the top mode grows by |g_n|^N; the lowest mode decays."""
        op = self._operator(1.2, allow_unstable=True)
        self.assertFalse(op.stable)
        top = explicit_mode_growth(op, 15)
        low = explicit_mode_growth(op, 1)
        predicted = abs(op.stability.factors[14]) ** op.steps
        self.assertGreater(op.stability.max_amplification, 1.3)
        self.assertGreater(top, 1e6)
        self.assertAlmostEqual(math.log(top), math.log(predicted), delta=1e-8)
        self.assertLess(low, 1.0)
        self.assertEqual(op.stability.worst_mode, 14)
        self.assertAlmostEqual(op.growth_bound, predicted, delta=1e-6 * predicted)

    def test_smooth_mode_matches_exponential_decay(self) -> None:
        """The stable scheme reproduces exp(-kappa pi^2 T) for the lowest mode."""
        op = self._operator(0.4, steps=400)
        horizon = op.time_step * op.steps
        measured = explicit_mode_growth(op, 1)
        self.assertAlmostEqual(
            measured,
            math.exp(-(PI**2) * op.diffusivity * horizon),
            delta=2e-3 * measured,
        )

    def test_adjoint_is_exact_transpose(self) -> None:
        """A' equals the transpose of the materialized A (stable and unstable)."""
        for ratio, allow in ((0.8, False), (1.1, True)):
            op = self._operator(
                ratio,
                steps=12,
                observation_steps=[3, 7, 12],
                observation_nodes=[0, 4, 9, 14],
                allow_unstable=allow,
            )
            matrix = materialize(op)
            rows, cols = op.shape
            self.assertEqual((rows, cols), (12, 15))
            scale = max(abs(v) for row in matrix for v in row)
            for j in range(cols):
                e = [0.0] * rows
                for i in range(rows):
                    e[i] = 1.0
                    column = op.rmatvec(e)[j]
                    e[i] = 0.0
                    self.assertAlmostEqual(column, matrix[i][j], delta=1e-13 * scale)
            x = _random_vector(cols, 15)
            y = _random_vector(rows, 16)
            result = adjoint_test(op, x, y, tolerance=1e-13)
            self.assertTrue(result.passed, result.relative_error)

    def test_unstable_operator_is_refused_by_default(self) -> None:
        """A step above the limit raises unless allow_unstable=True."""
        with self.assertRaises(UnstableTimeStepError):
            self._operator(1.05)
        op = self._operator(1.05, allow_unstable=True)
        self.assertFalse(op.stable)
        self.assertGreater(op.stability.ratio, 1.0)

    def test_inverse_problem_refused_on_unstable_step(self) -> None:
        """The inverse-problem builder refuses an unstable operator twice over."""
        unstable = self._operator(1.5, steps=10, allow_unstable=True)
        data = [0.0] * unstable.shape[0]
        with self.assertRaises(UnstableTimeStepError):
            explicit_heat_inverse_problem(unstable, data)
        problem = explicit_heat_inverse_problem(unstable, data, allow_unstable=True)
        self.assertEqual(problem.operator.shape, unstable.shape)
        stable = self._operator(0.5, steps=10)
        explicit_heat_inverse_problem(stable, [0.0] * stable.shape[0])

    def test_overflow_is_not_hidden(self) -> None:
        """A flagged unstable run that overflows raises instead of returning inf."""
        op = self._operator(1.5, steps=2500, allow_unstable=True)
        with self.assertRaises(ValueError):
            op.matvec([1.0] * 15)

    def test_validation(self) -> None:
        """Invalid configurations are rejected."""
        with self.assertRaises(ValueError):
            HeatExplicitEuler1D(1, final_time=1.0, steps=10)
        with self.assertRaises(ValueError):
            HeatExplicitEuler1D(5, final_time=0.0, steps=10)
        with self.assertRaises(ValueError):
            HeatExplicitEuler1D(5, final_time=0.01, steps=10, observation_steps=[11])
        with self.assertRaises(ValueError):
            HeatExplicitEuler1D(5, final_time=0.01, steps=10, observation_nodes=[5])
        with self.assertRaises(TypeError):
            HeatExplicitEuler1D(5, final_time=0.01, steps=10, allow_unstable=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            explicit_heat_inverse_problem("op", [0.0])  # type: ignore[arg-type]
        op = self._operator(0.5)
        with self.assertRaises(ValueError):
            explicit_mode_growth(op, 16)


if __name__ == "__main__":
    unittest.main()
