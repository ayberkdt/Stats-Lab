"""Tests of the one-dimensional heat-equation operators (pure Python)."""

from __future__ import annotations

import math
import unittest

from probstat.foundations.random_streams import resolve_stream
from probstat.inverse.heat_models import (
    HeatEquation1D,
    HeatSource1D,
    heat_amplification,
    heat_semigroup_singular_values,
)
from probstat.inverse.iterative import expected_noise_norm, krylov_regularization
from probstat.inverse.linear_problem import LinearInverseProblem
from probstat.inverse.test_problems import backward_heat_problem
from probstat.linalg import (
    ScaledIdentityCovariance,
    adjoint_test,
    materialize,
    singular_value_decomposition,
)

SCHEMES = ("implicit_euler", "crank_nicolson")


def _random_vector(size: int, seed: int) -> tuple[float, ...]:
    """Return a reproducible standard-normal vector."""
    rng = resolve_stream(seed).generator()
    return tuple(rng.gauss(0.0, 1.0) for _ in range(size))


def _sine(n: int, length: float = 1.0) -> list[float]:
    """Return the lowest sine mode on ``n`` interior nodes."""
    h = length / (n + 1)
    return [math.sin(math.pi * (j + 1) * h / length) for j in range(n)]


class ClosedFormTests(unittest.TestCase):
    """Closed-form singular values and amplification factors."""

    def test_amplification_factors(self) -> None:
        """Implicit Euler and CN factors, and unconditional stability."""
        self.assertAlmostEqual(heat_amplification("implicit_euler", 3.0), 0.25)
        self.assertAlmostEqual(heat_amplification("crank_nicolson", 2.0), 0.0)
        for scheme in SCHEMES:
            for x in (0.0, 1e-3, 1.0, 50.0, 1e8):
                self.assertLessEqual(abs(heat_amplification(scheme, x)), 1.0)
        self.assertLess(heat_amplification("crank_nicolson", 1e8), -0.999)
        with self.assertRaises(ValueError):
            heat_amplification("explicit_euler", 1.0)
        with self.assertRaises(ValueError):
            heat_amplification("implicit_euler", -1.0)

    def test_singular_values_match_materialized_operator(self) -> None:
        """Dense SVD of the materialized n = 12 map equals the closed form."""
        for scheme in SCHEMES:
            for length, kappa in ((1.0, 1.0), (2.5, 0.3)):
                op = HeatEquation1D(
                    12,
                    length=length,
                    diffusivity=kappa,
                    final_time=0.2,
                    steps=5,
                    scheme=scheme,
                )
                dense = singular_value_decomposition(materialize(op)).singular_values
                closed = heat_semigroup_singular_values(
                    12,
                    length=length,
                    diffusivity=kappa,
                    final_time=0.2,
                    steps=5,
                    scheme=scheme,
                )
                self.assertEqual(len(closed), 12)
                for a, b in zip(dense, closed, strict=True):
                    self.assertAlmostEqual(a, b, delta=1e-13)

    def test_contractive_for_huge_time_step(self) -> None:
        """Unconditional stability: ``||A x|| <= ||x||`` even for ``mu dt >> 1``."""
        x = _random_vector(30, 3)
        for scheme in SCHEMES:
            op = HeatEquation1D(30, final_time=100.0, steps=2, scheme=scheme)
            self.assertLessEqual(
                math.hypot(*op.matvec(x)), math.hypot(*x) * (1 + 1e-12)
            )
            sigma = heat_semigroup_singular_values(
                30, final_time=100.0, steps=2, scheme=scheme
            )
            self.assertLessEqual(sigma[0], 1.0)

    def test_crank_nicolson_oscillates_on_nonsmooth_data(self) -> None:
        """A one-node spike: implicit Euler stays non-negative, CN does not."""
        n = 40
        spike = [0.0] * n
        spike[n // 2] = 1.0
        euler = HeatEquation1D(n, final_time=0.01, steps=1).matvec(spike)
        cn = HeatEquation1D(
            n, final_time=0.01, steps=1, scheme="crank_nicolson"
        ).matvec(spike)
        self.assertGreaterEqual(min(euler), 0.0)
        self.assertLess(min(cn), -1e-3)


class AdjointTests(unittest.TestCase):
    """The rmatvec is the exact transpose of the discrete map."""

    def test_final_time_adjoint(self) -> None:
        """Dot-product test for final-time observations, both schemes."""
        for scheme in SCHEMES:
            op = HeatEquation1D(23, final_time=0.07, steps=9, scheme=scheme)
            for seed in (1, 2):
                result = adjoint_test(
                    op, _random_vector(23, seed), _random_vector(23, seed + 10)
                )
                self.assertLess(result.relative_error, 1e-12)

    def test_sensor_adjoint_with_fractional_times(self) -> None:
        """Dot-product test with sensors and times between steps."""
        sensors = [0.13, 0.5, 0.77, 0.9]
        times = [0.0, 0.013, 0.05, 0.0999, 0.1]
        for scheme in SCHEMES:
            for cls in (HeatEquation1D, HeatSource1D):
                op = cls(
                    19,
                    final_time=0.1,
                    steps=10,
                    scheme=scheme,
                    observation="sensors",
                    sensors=sensors,
                    times=times,
                )
                self.assertEqual(op.shape, (len(sensors) * len(times), 19))
                result = adjoint_test(
                    op, _random_vector(19, 5), _random_vector(op.shape[0], 6)
                )
                self.assertLess(result.relative_error, 1e-12)

    def test_source_adjoint_final_time(self) -> None:
        """Dot-product test of the source operator, final-time observation."""
        for scheme in SCHEMES:
            op = HeatSource1D(17, final_time=0.3, steps=6, scheme=scheme)
            result = adjoint_test(op, _random_vector(17, 7), _random_vector(17, 8))
            self.assertLess(result.relative_error, 1e-12)

    def test_transpose_matches_materialized_transpose(self) -> None:
        """``rmatvec`` equals the transposed dense matrix, entry by entry."""
        op = HeatEquation1D(
            7,
            final_time=0.05,
            steps=4,
            scheme="crank_nicolson",
            observation="sensors",
            sensors=[0.3, 0.6],
            times=[0.02, 0.05],
        )
        dense = materialize(op)
        y = _random_vector(4, 9)
        direct = op.rmatvec(y)
        for j in range(7):
            expected = math.fsum(dense[i][j] * y[i] for i in range(4))
            self.assertAlmostEqual(direct[j], expected, delta=1e-14)


class ObservationTests(unittest.TestCase):
    """Sensor interpolation in space and time."""

    def test_node_sensor_equals_final_state_row(self) -> None:
        """A sensor on a node at ``t = T`` reproduces the final-time map."""
        n, steps = 9, 6
        h = 1.0 / (n + 1)
        full = HeatEquation1D(n, final_time=0.06, steps=steps)
        node = HeatEquation1D(
            n,
            final_time=0.06,
            steps=steps,
            observation="sensors",
            sensors=[3 * h, 7 * h],
            times=[0.06],
        )
        x = _random_vector(n, 11)
        target = full.matvec(x)
        got = node.matvec(x)
        self.assertAlmostEqual(got[0], target[2], delta=1e-13)
        self.assertAlmostEqual(got[1], target[6], delta=1e-13)

    def test_linear_interpolation_in_time_and_space(self) -> None:
        """Value at ``(k + 1/4) dt`` and a mid-cell sensor, by hand."""
        n, dt, k = 9, 0.01, 3
        h = 1.0 / (n + 1)
        x = _random_vector(n, 12)
        state = [
            HeatEquation1D(n, final_time=m * dt, steps=m).matvec(x)
            if m > 0
            else list(x)
            for m in (k, k + 1)
        ]
        op = HeatEquation1D(
            n,
            final_time=0.1,
            steps=10,
            observation="sensors",
            sensors=[4.5 * h],
            times=[(k + 0.25) * dt],
        )
        space = [0.5 * (s[3] + s[4]) for s in state]
        expected = 0.75 * space[0] + 0.25 * space[1]
        self.assertAlmostEqual(op.matvec(x)[0], expected, delta=1e-14)

    def test_boundary_cell_sensor_uses_zero_boundary(self) -> None:
        """A sensor between the left boundary and node 1 sees ``u(0) = 0``."""
        n = 5
        h = 1.0 / (n + 1)
        x = _random_vector(n, 13)
        op = HeatEquation1D(
            n,
            final_time=0.1,
            steps=2,
            observation="sensors",
            sensors=[0.5 * h],
            times=[0.0],
        )
        self.assertAlmostEqual(op.matvec(x)[0], 0.5 * x[0], delta=1e-15)
        self.assertEqual(op.observation_points, ((0.0, 0.5 * h),))


class ConvergenceTests(unittest.TestCase):
    """Observed orders in dt and agreement with the continuous solution."""

    @staticmethod
    def _orders(scheme: str) -> list[float]:
        """Observed time orders against the exact semi-discrete decay."""
        n = 20
        h = 1.0 / (n + 1)
        mu = 4.0 / h**2 * math.sin(math.pi * h / 2.0) ** 2
        x = _sine(n)
        final_time = 0.1
        errors = []
        for steps in (20, 40, 80, 160):
            op = HeatEquation1D(n, final_time=final_time, steps=steps, scheme=scheme)
            y = op.matvec(x)
            decay = math.exp(-mu * final_time)
            errors.append(max(abs(a - decay * b) for a, b in zip(y, x, strict=True)))
        return [math.log2(errors[i] / errors[i + 1]) for i in range(3)]

    def test_implicit_euler_first_order_in_dt(self) -> None:
        """Observed order of implicit Euler lies in [0.8, 1.2]."""
        for order in self._orders("implicit_euler"):
            self.assertGreater(order, 0.8)
            self.assertLess(order, 1.2)

    def test_crank_nicolson_second_order_in_dt(self) -> None:
        """Observed order of Crank-Nicolson lies in [1.7, 2.3]."""
        for order in self._orders("crank_nicolson"):
            self.assertGreater(order, 1.7)
            self.assertLess(order, 2.3)

    def test_sine_mode_approaches_continuous_decay(self) -> None:
        """Joint refinement in ``n`` and ``N`` approaches ``exp(-kappa pi^2 T)``."""
        kappa, final_time, length = 0.7, 0.1, 1.0
        decay = math.exp(-kappa * (math.pi / length) ** 2 * final_time)
        errors = []
        for n, steps in ((7, 5), (15, 10), (31, 20), (63, 40)):
            op = HeatEquation1D(
                n,
                diffusivity=kappa,
                final_time=final_time,
                steps=steps,
                scheme="crank_nicolson",
            )
            x = _sine(n)
            y = op.matvec(x)
            errors.append(max(abs(a - decay * b) for a, b in zip(y, x, strict=True)))
        for coarse, fine in zip(errors, errors[1:], strict=False):
            self.assertLess(fine, coarse)
        self.assertLess(errors[-1], 1e-3)

    def test_matches_backward_heat_benchmark(self) -> None:
        """The existing closed-form backward-heat operator is the oracle.

        ``backward_heat_problem(size, time)`` uses ``L = 1``, ``kappa = 1``,
        ``h = 1 / (size + 1)`` and the exact semigroup ``exp(-time D)`` of the
        same second-difference matrix, so the conventions agree exactly with
        ``HeatEquation1D(size, final_time=time)`` in the limit ``dt -> 0``.
        """
        problem = backward_heat_problem(32, time=0.005)
        scale = math.hypot(*problem.exact_data)
        previous = {scheme: math.inf for scheme in SCHEMES}
        for steps in (50, 200, 800):
            for scheme in SCHEMES:
                op = HeatEquation1D(32, final_time=0.005, steps=steps, scheme=scheme)
                y = op.matvec(problem.exact_solution)
                error = math.dist(y, problem.exact_data) / scale
                self.assertLess(error, previous[scheme])
                previous[scheme] = error
        self.assertLess(previous["implicit_euler"], 5e-5)
        self.assertLess(previous["crank_nicolson"], 1e-7)
        # singular values: the discrete amplification converges to exp(-mu T)
        exact = problem.singular_values
        sigma = heat_semigroup_singular_values(
            32, final_time=0.005, steps=4000, scheme="crank_nicolson"
        )
        for a, b in zip(sorted(exact, reverse=True)[:12], sigma[:12], strict=True):
            self.assertAlmostEqual(a, b, delta=1e-6)


class SourceTests(unittest.TestCase):
    """The linear source-inversion operator."""

    def test_sine_source_closed_form(self) -> None:
        """For a sine source ``u^N = (1 - g^N) / mu`` times the sine mode."""
        n, steps, final_time = 15, 8, 0.2
        h = 1.0 / (n + 1)
        mu = 4.0 / h**2 * math.sin(math.pi * h / 2.0) ** 2
        x = _sine(n)
        for scheme in SCHEMES:
            op = HeatSource1D(n, final_time=final_time, steps=steps, scheme=scheme)
            g = heat_amplification(scheme, mu * final_time / steps)
            factor = (1.0 - g**steps) / mu
            y = op.matvec(x)
            for value, mode in zip(y, x, strict=True):
                self.assertAlmostEqual(value, factor * mode, delta=1e-13)

    def test_steady_state_limit(self) -> None:
        """For a long horizon the response tends to ``D^-1 q``."""
        n = 11
        op = HeatSource1D(n, final_time=50.0, steps=5)
        q = [1.0] * n
        u = op.matvec(q)
        h = 1.0 / (n + 1)
        for j in range(n):
            s = (j + 1) * h
            self.assertAlmostEqual(u[j], 0.5 * s * (1.0 - s), delta=1e-9)


class CountTests(unittest.TestCase):
    """Work counting."""

    def test_solves_per_application(self) -> None:
        """One application of the final-time maps is ``steps`` solves."""
        for cls in (HeatEquation1D, HeatSource1D):
            for scheme in SCHEMES:
                op = cls(10, final_time=0.1, steps=7, scheme=scheme)
                op.matvec([1.0] * 10)
                counts = op.counts
                self.assertEqual(counts.forward_applications, 1)
                self.assertEqual(counts.forward_solves, 7)
                self.assertEqual(counts.transpose_solves, 0)
                op.rmatvec([1.0] * 10)
                op.rmatvec([2.0] * 10)
                counts = op.counts
                self.assertEqual(counts.transpose_applications, 2)
                self.assertEqual(counts.transpose_solves, 14)
                op.reset_counts()
                self.assertEqual(op.counts.forward_solves, 0)

    def test_sensor_operator_stops_at_last_needed_step(self) -> None:
        """Observations before ``T`` need fewer solves."""
        op = HeatEquation1D(
            10,
            final_time=0.1,
            steps=10,
            observation="sensors",
            sensors=[0.5],
            times=[0.045],
        )
        op.matvec([1.0] * 10)
        self.assertEqual(op.counts.forward_solves, 5)
        op.rmatvec([1.0])
        self.assertEqual(op.counts.transpose_solves, 5)


class InvalidInputTests(unittest.TestCase):
    """Invalid configurations raise."""

    def test_invalid_parameters(self) -> None:
        """Non-positive kappa, T, steps; wrong types; unknown scheme."""
        for kwargs in (
            {"diffusivity": 0.0},
            {"diffusivity": -1.0},
            {"final_time": 0.0},
            {"final_time": -0.1},
            {"final_time": math.inf},
            {"steps": 0},
            {"length": 0.0},
            {"scheme": "explicit_euler"},
            {"observation": "everything"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HeatEquation1D(8, **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            HeatEquation1D(1)
        with self.assertRaises(TypeError):
            HeatEquation1D(8, steps=2.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            HeatEquation1D(8.0)  # type: ignore[arg-type]

    def test_invalid_sensors_and_times(self) -> None:
        """Sensors outside the domain, times beyond ``T``, bad combinations."""
        base = {"final_time": 0.1, "steps": 4, "observation": "sensors"}
        for sensors, times in (
            ([0.0], [0.05]),
            ([1.0], [0.05]),
            ([1.5], [0.05]),
            ([-0.1], [0.05]),
            ([0.5], [0.11]),
            ([0.5], [-0.01]),
            ([0.5], [math.nan]),
            ([], [0.05]),
            ([0.5], []),
        ):
            with self.subTest(sensors=sensors, times=times):
                with self.assertRaises(ValueError):
                    HeatEquation1D(8, sensors=sensors, times=times, **base)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            HeatEquation1D(8, final_time=0.1, steps=4, observation="sensors")
        with self.assertRaises(ValueError):
            HeatEquation1D(8, observation="final", sensors=[0.5], times=[0.1])
        with self.assertRaises(TypeError):
            HeatEquation1D(8, sensors="abc", times=[0.1], **base)  # type: ignore[arg-type]

    def test_invalid_vectors(self) -> None:
        """Wrong-length and non-finite arguments are rejected by the operator."""
        op = HeatEquation1D(6)
        with self.assertRaises(ValueError):
            op.matvec([1.0] * 5)
        with self.assertRaises(ValueError):
            op.rmatvec([math.nan] * 6)
        with self.assertRaises(ValueError):
            heat_semigroup_singular_values(6, steps=0)


class InverseDemoTests(unittest.TestCase):
    """Initial-condition inversion with an independent data generator."""

    n = 32
    final_time = 0.01
    steps = 20

    def test_noise_floor_counts_few_singular_values(self) -> None:
        """Severe ill-posedness: only a handful of modes exceed the noise."""
        sigma = heat_semigroup_singular_values(
            self.n, final_time=self.final_time, steps=self.steps
        )
        above = sum(1 for s in sigma if s > 1e-2)
        self.assertGreaterEqual(above, 3)
        self.assertLessEqual(above, 10)
        self.assertLess(sigma[-1], 1e-9)

    def test_krylov_inversion_beats_zero_estimate(self) -> None:
        """Data from ``4N`` steps, 1 % noise, inversion with ``N`` steps."""
        n = self.n
        h = 1.0 / (n + 1)
        truth = tuple(6.75 * (j + 1) * h * (1.0 - (j + 1) * h) ** 2 for j in range(n))
        generator = HeatEquation1D(
            n, final_time=self.final_time, steps=4 * self.steps, scheme="crank_nicolson"
        )
        clean = generator.matvec(truth)
        noise_std = 0.01 * math.hypot(*clean) / math.sqrt(n)
        rng = resolve_stream(2024).generator()
        data = tuple(c + rng.gauss(0.0, noise_std) for c in clean)
        model = HeatEquation1D(
            n, final_time=self.final_time, steps=self.steps, scheme="crank_nicolson"
        )
        problem = LinearInverseProblem.create(
            model, data, noise=ScaledIdentityCovariance(n, noise_std**2)
        )
        for method in ("cgls", "lsqr"):
            result = krylov_regularization(
                problem,
                method=method,
                noise_norm=expected_noise_norm(n),
                tau=1.1,
                max_iterations=30,
                truth=truth,
            )
            self.assertEqual(result.stop_reason, "discrepancy")
            self.assertTrue(result.converged)
            error = math.dist(result.estimate, truth)
            self.assertLess(error, 0.6 * math.hypot(*truth))
            self.assertLess(result.iterations, 20)
            # work: one forward and one transpose application per iteration,
            # each costing `steps` tridiagonal solves.
            self.assertGreaterEqual(model.counts.forward_solves, self.steps)


if __name__ == "__main__":
    unittest.main()
