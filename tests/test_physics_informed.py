"""Tests for hybrid physics and machine-learning surrogates (layer 5).

Oracles: closed forms (tanh jets, antiderivative, exponential decay, sine
BVP), the independent nested forward-mode Hessian of ``MLP.forward``, a
log-linear least-squares estimate of the decay rate, exact rational conformal
ranks and ``probstat.conformal.conformal_quantile``. Training tests use fixed
seeds and small networks; tolerances come from the measured error with a
safety factor (stated at each assertion). The larger studies run only with
``PROBSTAT_SLOW=1``.
"""

from __future__ import annotations

import math
import os
import random
import unittest
from typing import Any

from probstat.conformal import conformal_quantile
from probstat.numerics import (
    MLP,
    admath,
    integrate_fixed_step,
    reverse_gradient,
    value_gradient_and_hessian,
)
from probstat.surrogates.physics_informed import (
    DeepONet,
    PINNCondition,
    PINNLossWeights,
    antiderivative_dataset,
    antiderivative_exact,
    compare_physics_ml_hybrid,
    deeponet_relative_error,
    distill_sparse_regression,
    fit_deeponet,
    fit_hybrid_ode,
    fit_pinn_1d,
    fit_residual_hybrid,
    network_jet,
    split_conformal_halfwidth,
)


class NetworkJetTests(unittest.TestCase):
    """Exact input derivatives of tanh networks."""

    def test_single_neuron_closed_form(self) -> None:
        """Match ``u = w2 tanh(w1 t + b1) + b2`` and its derivatives."""
        network = MLP((1, 1, 1))
        w1, b1, w2, b2 = 0.7, -0.2, 1.3, 0.4
        t = 0.55
        a = math.tanh(w1 * t + b1)
        s = 1.0 - a * a
        u, du, d2u = network_jet(network, (w1, b1, w2, b2), t)
        self.assertAlmostEqual(u, w2 * a + b2, places=14)
        self.assertAlmostEqual(du, w2 * w1 * s, places=14)
        self.assertAlmostEqual(d2u, -2.0 * w2 * w1 * w1 * a * s, places=14)

    def test_deep_network_matches_nested_forward_mode(self) -> None:
        """Agree with the library's forward-over-forward Hessian to 1e-12."""
        network = MLP((1, 5, 4, 1))
        parameters = network.initial_parameters(7)
        for t in (-0.8, 0.0, 0.37, 1.4):

            def f(x: Any) -> Any:
                """Return the network output on duals."""
                return network.forward(parameters, [x[0]])[0]

            value, gradient, hessian = value_gradient_and_hessian(f, [t])
            u, du, d2u = network_jet(network, parameters, t)
            self.assertAlmostEqual(u, value, delta=1e-13)
            self.assertAlmostEqual(du, gradient[0], delta=1e-12)
            self.assertAlmostEqual(d2u, hessian[0][0], delta=1e-12)

    def test_weight_gradient_of_second_derivative(self) -> None:
        """Reverse mode through the jets agrees with central differences."""
        network = MLP((1, 4, 1))
        base = network.initial_parameters(3)

        def d2(parameters: Any) -> Any:
            """Return u'' at a fixed point as a function of the weights."""
            return network_jet(network, parameters, 0.3)[2]

        gradient = reverse_gradient(d2, base)
        for index in (0, 3, 6, 11):
            h = 1e-6
            up = list(base)
            down = list(base)
            up[index] += h
            down[index] -= h
            fd = (d2(up) - d2(down)) / (2.0 * h)
            # Central-difference error is O(h^2) ~ 1e-12 times the third
            # derivative scale; 1e-7 leaves a wide margin.
            self.assertAlmostEqual(gradient[index], fd, delta=1e-7)

    def test_rejects_unsupported_networks(self) -> None:
        """Reject non-tanh, vector, and wrong-length-parameter cases."""
        with self.assertRaises(ValueError):
            network_jet(MLP((1, 3, 1), activation="softplus"), [0.0] * 10, 0.1)
        with self.assertRaises(ValueError):
            network_jet(MLP((2, 3, 1)), [0.0] * 13, 0.1)
        with self.assertRaises(ValueError):
            network_jet(MLP((1, 3, 1)), [0.0] * 9, 0.1)
        with self.assertRaises(TypeError):
            network_jet("not a network", [0.0], 0.1)  # type: ignore[arg-type]


def _decay_residual(t: Any, u: Any, du: Any, d2u: Any, theta: Any) -> Any:
    """Return the residual of ``u' = -1.5 u``."""
    return du + 1.5 * u


def _poisson_residual(x: Any, u: Any, du: Any, d2u: Any, theta: Any) -> Any:
    """Return the residual of ``u'' = -pi^2 sin(pi x)``."""
    return d2u + math.pi**2 * admath.sin(math.pi * x)


class PINNTests(unittest.TestCase):
    """Forward and inverse PINNs against analytic solutions."""

    def test_forward_decay(self) -> None:
        """Solve ``u' = -1.5 u, u(0) = 1``; measured max error 2.1e-5."""
        result = fit_pinn_1d(
            _decay_residual,
            (0.0, 1.0),
            conditions=[PINNCondition(0.0, 1.0)],
            hidden=(8,),
            collocation_points=11,
            max_iterations=100,
            gradient_tolerance=1e-7,
            seed=1,
        )
        error = max(
            abs(result.predict(i / 50) - math.exp(-1.5 * i / 50)) for i in range(51)
        )
        self.assertLess(error, 1e-3)
        self.assertLess(abs(result.derivative(0.5) + 1.5 * result.predict(0.5)), 1e-3)
        self.assertLess(abs(result.residual_at(0.37)), 1e-3)
        # Cost accounting is reported, not hidden.
        self.assertGreater(result.evaluations, 0)
        self.assertLessEqual(result.iterations, 100)
        self.assertGreaterEqual(result.seconds, 0.0)
        self.assertEqual(result.physical, ())

    def test_poisson_bvp(self) -> None:
        """Solve ``u'' = -pi^2 sin(pi x)``, zero ends; measured error 5.1e-5."""
        result = fit_pinn_1d(
            _poisson_residual,
            (0.0, 1.0),
            conditions=[PINNCondition(0.0, 0.0), PINNCondition(1.0, 0.0)],
            hidden=(8,),
            collocation_points=11,
            max_iterations=150,
            gradient_tolerance=1e-7,
            seed=1,
        )
        error = max(
            abs(result.predict(i / 50) - math.sin(math.pi * i / 50)) for i in range(51)
        )
        self.assertLess(error, 5e-3)
        self.assertLess(abs(result.predict(0.0)), 5e-3)

    def test_inverse_decay_rate(self) -> None:
        """Estimate k from 7 noisy points (sigma 0.01), true k = 1.5.

        The oracle is the closed-form log-linear least-squares rate through
        the origin (1.4714 for this draw); the PINN, which also enforces the
        ODE, measured 1.4782. The difference is bounded by 0.03 and the
        distance to the truth by 0.1 (about 3 standard errors).
        """
        generator = random.Random(3)
        times = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
        values = [math.exp(-1.5 * t) + 0.01 * generator.gauss(0.0, 1.0) for t in times]
        oracle = -sum(t * math.log(y) for t, y in zip(times, values, strict=True))
        oracle /= sum(t * t for t in times)

        def residual(t: Any, u: Any, du: Any, d2u: Any, theta: Any) -> Any:
            """Return ``u' + k u`` with the unknown rate ``k = theta[0]``."""
            return du + theta[0] * u

        result = fit_pinn_1d(
            residual,
            (0.0, 1.0),
            conditions=[PINNCondition(0.0, 1.0)],
            data_times=times,
            data_values=values,
            physical_initial=[0.5],
            hidden=(6,),
            collocation_points=11,
            max_iterations=80,
            gradient_tolerance=1e-7,
            seed=1,
        )
        self.assertEqual(len(result.physical), 1)
        self.assertLess(abs(result.physical[0] - oracle), 0.03)
        self.assertLess(abs(result.physical[0] - 1.5), 0.1)

    def test_loss_is_the_explicit_weighted_sum(self) -> None:
        """The reported loss equals ``a_r L_r + a_c L_c + a_d L_d``."""
        weights = PINNLossWeights(residual=2.0, condition=5.0, data=0.5)
        result = fit_pinn_1d(
            _decay_residual,
            (0.0, 1.0),
            conditions=[PINNCondition(0.0, 1.0)],
            data_times=[0.5],
            data_values=[0.5],
            hidden=(3,),
            collocation_points=5,
            weights=weights,
            max_iterations=5,
            seed=2,
        )
        expected = (
            2.0 * result.residual_loss
            + 5.0 * result.condition_loss
            + 0.5 * result.data_loss
        )
        self.assertAlmostEqual(result.loss, expected, delta=1e-12 * (1 + expected))

    def test_rejects_invalid_input(self) -> None:
        """Invalid problems raise instead of being repaired."""
        good: dict[str, Any] = {"conditions": [PINNCondition(0.0, 1.0)]}
        with self.assertRaises(TypeError):
            fit_pinn_1d("not callable", (0.0, 1.0), **good)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_pinn_1d(_decay_residual, (1.0, 0.0), **good)
        with self.assertRaises(ValueError):
            fit_pinn_1d(_decay_residual, (0.0, 1.0), collocation_points=1, **good)
        with self.assertRaises(ValueError):
            fit_pinn_1d(
                _decay_residual,
                (0.0, 1.0),
                conditions=[PINNCondition(0.0, 1.0, order=3)],
            )
        with self.assertRaises(ValueError):
            fit_pinn_1d(
                _decay_residual, (0.0, 1.0), data_times=[0.1], data_values=[], **good
            )
        with self.assertRaises(ValueError):
            PINNLossWeights(residual=-1.0)
        with self.assertRaises(ValueError):
            fit_pinn_1d(
                _decay_residual,
                (0.0, 1.0),
                weights=PINNLossWeights(residual=0.0),
            )
        with self.assertRaises(ValueError):
            fit_pinn_1d(
                _decay_residual,
                (0.0, 1.0),
                conditions=[PINNCondition(0.0, math.nan)],
            )


# Damped oscillator in SI-like units (t [s], x [m], v [m/s]): unit angular
# frequency squared and linear damping are known; the cubic damping term is
# the "missing physics". Synthetic data, not a measured system.
OMEGA2, DAMPING, CUBIC = 1.0, 0.1, 0.4


def _truth_rhs(t: Any, y: Any) -> list[Any]:
    """Return the true oscillator with cubic damping."""
    return [y[1], -OMEGA2 * y[0] - DAMPING * y[1] - CUBIC * y[1] ** 3]


def _physics_rhs(t: Any, y: Any, theta: Any) -> list[Any]:
    """Return the known physics (no cubic damping)."""
    return [y[1], -OMEGA2 * y[0] - DAMPING * y[1]]


def _trajectory(y0: tuple[float, float], intervals: int, dt: float) -> Any:
    """Integrate the truth with a 20-fold finer RK4 step; sample every dt."""
    path = integrate_fixed_step(
        _truth_rhs, 0.0, y0, step=dt / 20.0, steps=intervals * 20
    )
    return path.states[::20]


def _rms(values: list[float]) -> float:
    """Return the root-mean-square of a list."""
    return math.sqrt(math.fsum(v * v for v in values) / len(values))


class HybridODETests(unittest.TestCase):
    """Universal differential equation with an unknown damping term."""

    def _fit(
        self,
        intervals: int,
        hidden: tuple[int, ...],
        iterations: int,
        starts: tuple[tuple[float, float], ...] = ((1.0, 0.0),),
    ) -> Any:
        """Fit on the trajectories of the given initial states."""
        dt = 0.2
        data = [_trajectory(y0, intervals, dt) for y0 in starts]
        times = [i * dt for i in range(intervals + 1)]
        result = fit_hybrid_ode(
            _physics_rhs,
            times,
            data,
            correction_components=(1,),
            hidden=hidden,
            max_iterations=iterations,
            gradient_tolerance=1e-8,
            seed=2,
        )
        return result, data

    def _score(self, result: Any, intervals: int) -> tuple[float, float, float]:
        """Return correction RMS error, reference RMS, and forecast ratio."""
        dt = 0.2
        held = _trajectory((0.7, 0.6), intervals, dt)
        errors = [result.correction(s)[0] - (-CUBIC * s[1] ** 3) for s in held]
        reference = _rms([CUBIC * s[1] ** 3 for s in held])
        hybrid = result.simulate((0.7, 0.6), step=dt, steps=intervals)
        physics = integrate_fixed_step(
            lambda t, y: _physics_rhs(t, y, ()),
            0.0,
            (0.7, 0.6),
            step=dt,
            steps=intervals,
        ).states

        def forecast(states: Any) -> float:
            """Return the RMS state error against the held-out truth."""
            return _rms(
                [
                    a - b
                    for model, truth in zip(states, held, strict=True)
                    for a, b in zip(model, truth, strict=True)
                ]
            )

        return _rms(errors), reference, forecast(hybrid) / forecast(physics)

    def test_recovers_missing_damping_fast(self) -> None:
        """One trajectory, 4 hidden units, 100 BFGS steps.

        Measured: correction RMS error 0.023 against a reference of 0.082
        (28 %), hybrid forecast error 0.25 of the physics-only error. The
        bounds (0.5 and 0.5) leave a factor of about 2.
        """
        result, _ = self._fit(20, (4,), 100)
        self.assertGreater(result.evaluations, 0)
        error, reference, ratio = self._score(result, 20)
        self.assertLess(error, 0.5 * reference)
        self.assertLess(ratio, 0.5)

    def test_pure_physics_data_gives_no_correction(self) -> None:
        """Data generated by the known physics leave a near-zero correction."""
        dt = 0.2
        data = [
            integrate_fixed_step(
                lambda t, y: _physics_rhs(t, y, ()),
                0.0,
                (1.0, 0.0),
                step=dt / 20.0,
                steps=10 * 20,
            ).states[::20]
        ]
        result = fit_hybrid_ode(
            _physics_rhs,
            [i * dt for i in range(11)],
            data,
            correction_components=(1,),
            hidden=(3,),
            max_iterations=40,
            gradient_tolerance=1e-8,
            seed=4,
        )
        worst = max(abs(result.correction(s)[0]) for s in data[0])
        self.assertLess(worst, 5e-3)
        self.assertLess(result.loss, 1e-8)

    def test_rejects_invalid_input(self) -> None:
        """Validation of times, components and shapes."""
        data = [[(1.0, 0.0), (0.9, -0.1), (0.7, -0.2)]]
        common: dict[str, Any] = {"correction_components": (1,), "hidden": (2,)}
        with self.assertRaises(ValueError):
            fit_hybrid_ode(_physics_rhs, [0.0, 0.2, 0.5], data, **common)
        with self.assertRaises(ValueError):
            fit_hybrid_ode(
                _physics_rhs,
                [0.0, 0.2, 0.4],
                data,
                correction_components=(2,),
                hidden=(2,),
            )
        with self.assertRaises(ValueError):
            fit_hybrid_ode(
                _physics_rhs,
                [0.0, 0.2, 0.4],
                data,
                correction_components=(1, 1),
                hidden=(2,),
            )
        with self.assertRaises(ValueError):
            fit_hybrid_ode(_physics_rhs, [0.0, 0.2], data, **common)
        with self.assertRaises(TypeError):
            fit_hybrid_ode(None, [0.0, 0.2, 0.4], data, **common)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_hybrid_ode(lambda t, y, th: [y[0]], [0.0, 0.2, 0.4], data, **common)


class SparseDistillationTests(unittest.TestCase):
    """Sequentially thresholded least squares."""

    def test_recovers_exact_sparse_polynomial(self) -> None:
        """Noise-free ``0.7 v - 0.4 v^3`` is recovered to 1e-10."""
        points = [(x / 4.0, v / 4.0) for x in range(-4, 5) for v in range(-4, 5)]
        targets = [0.7 * v - 0.4 * v**3 for _, v in points]
        fit = distill_sparse_regression(
            points, targets, degree=3, threshold=0.05, variable_names=("x", "v")
        )
        terms = fit.terms()
        self.assertEqual(sorted(terms), ["v", "v^3"])
        self.assertAlmostEqual(terms["v"], 0.7, delta=1e-10)
        self.assertAlmostEqual(terms["v^3"], -0.4, delta=1e-10)
        self.assertTrue(fit.converged)
        self.assertLess(fit.rms_residual, 1e-10)
        self.assertAlmostEqual(fit.evaluate((0.3, 0.5)), 0.35 - 0.05, places=10)

    def test_noisy_data_keeps_the_active_set(self) -> None:
        """Noise 0.01 on 81 points: same active set, coefficients within 0.03."""
        generator = random.Random(11)
        points = [(x / 4.0, v / 4.0) for x in range(-4, 5) for v in range(-4, 5)]
        targets = [
            0.7 * v - 0.4 * v**3 + 0.01 * generator.gauss(0.0, 1.0) for _, v in points
        ]
        fit = distill_sparse_regression(points, targets, threshold=0.1)
        terms = fit.terms()
        self.assertEqual(sorted(terms), ["x1", "x1^3"])
        self.assertAlmostEqual(terms["x1"], 0.7, delta=0.03)
        self.assertAlmostEqual(terms["x1^3"], -0.4, delta=0.03)

    def test_rejects_invalid_input(self) -> None:
        """Too few samples, bad threshold, dependent library, empty result."""
        points = [(float(i),) for i in range(3)]
        with self.assertRaises(ValueError):
            distill_sparse_regression(points, [0.0, 1.0, 2.0], degree=3)
        grid = [(i / 5.0,) for i in range(12)]
        with self.assertRaises(ValueError):
            distill_sparse_regression(grid, [0.0] * 12, threshold=0.0)
        with self.assertRaises(ValueError):
            distill_sparse_regression([(1.0,)] * 12, [1.0] * 12, degree=2)
        with self.assertRaises(ValueError):
            distill_sparse_regression(grid, [0.0] * 12, threshold=0.5)
        with self.assertRaises(ValueError):
            distill_sparse_regression(grid, [0.0] * 11)


class ConformalTests(unittest.TestCase):
    """Local split-conformal quantile."""

    def test_exact_rank_where_float_arithmetic_fails(self) -> None:
        """``alpha = 0.7, n = 9``: rank is exactly 3 (float gives 4)."""
        scores = [float(i) for i in range(1, 10)]
        self.assertEqual(split_conformal_halfwidth(scores, 0.7), 3.0)
        self.assertEqual(math.ceil((1 - 0.7) * 10), 4)  # the float pitfall

    def test_matches_conformal_package_and_infinite_case(self) -> None:
        """Agree with ``probstat.conformal`` and return inf when ``k = n+1``."""
        generator = random.Random(1)
        scores = [abs(generator.gauss(0.0, 1.0)) for _ in range(23)]
        for alpha in (0.05, 0.1, 0.2, 0.5):
            self.assertEqual(
                split_conformal_halfwidth(scores, alpha),
                conformal_quantile(scores, alpha),
            )
        self.assertEqual(split_conformal_halfwidth([1.0] * 9, 0.05), math.inf)

    def test_rejects_invalid_input(self) -> None:
        """Alpha outside (0, 1) and empty scores raise."""
        for alpha in (0.0, 1.0, -0.1, math.nan):
            with self.assertRaises(ValueError):
                split_conformal_halfwidth([1.0, 2.0], alpha)
        with self.assertRaises(ValueError):
            split_conformal_halfwidth([], 0.1)
        with self.assertRaises(TypeError):
            split_conformal_halfwidth([1.0], "0.1")  # type: ignore[arg-type]


def _truth(x: float) -> float:
    """Return the synthetic truth: decay plus an unmodelled oscillation."""
    return 2.0 * math.exp(-0.4 * x) + 0.5 * math.sin(2.0 * x)


def _physics(x: Any) -> float:
    """Return the physics model that misses the oscillation."""
    return 2.0 * math.exp(-0.4 * x[0])


class ResidualHybridTests(unittest.TestCase):
    """Physics-only versus ML-only versus hybrid, measured."""

    @staticmethod
    def _draw(
        generator: random.Random, n: int, low: float, high: float
    ) -> tuple[list[list[float]], list[float]]:
        """Draw noisy observations (sigma 0.02) of the truth."""
        xs = [[generator.uniform(low, high)] for _ in range(n)]
        return xs, [_truth(x[0]) + 0.02 * generator.gauss(0.0, 1.0) for x in xs]

    def _data(self) -> Any:
        """Return train, calibration, in-range and extrapolation sets."""
        generator = random.Random(5)
        train = self._draw(generator, 30, 0.0, 3.0)
        calibration = self._draw(generator, 40, 0.0, 3.0)
        inside = self._draw(generator, 300, 0.0, 3.0)
        outside = self._draw(generator, 100, 3.0, 4.5)
        return train, calibration, inside, outside

    def test_gp_comparison(self) -> None:
        """Measured (seed 1): in-range RMSE physics 0.360, ML 0.0244, hybrid 0.0245.

        Extrapolation RMSE: physics 0.361, ML 0.216, hybrid 0.202; hybrid
        extrapolation coverage 0.07 (nominal 0.9): the conformal guarantee is
        marginal for exchangeable data and does not survive extrapolation.
        """
        train, calibration, inside, outside = self._data()
        report = compare_physics_ml_hybrid(
            _physics,
            *train,
            *calibration,
            {"in": inside, "extrapolation": outside},
            learner="gp",
            alpha=0.1,
            seed=1,
        )
        self.assertAlmostEqual(report.nominal_coverage, 0.9)
        physics_in = report.get("physics", "in")
        hybrid_in = report.get("hybrid", "in")
        ml_in = report.get("ml", "in")
        self.assertLess(hybrid_in.rmse, 0.2 * physics_in.rmse)
        self.assertLess(ml_in.rmse, 0.2 * physics_in.rmse)
        # Marginal coverage (calibration-conditional sd about 0.045): 0.80
        # leaves two standard deviations below the nominal 0.9.
        for name in ("physics", "ml", "hybrid"):
            self.assertGreaterEqual(report.get(name, "in").coverage, 0.8)
        hybrid_out = report.get("hybrid", "extrapolation")
        self.assertLess(hybrid_out.rmse, report.get("physics", "extrapolation").rmse)
        self.assertGreater(report.get("ml", "extrapolation").rmse, 3 * ml_in.rmse)
        self.assertLess(hybrid_out.coverage, 0.5)  # honest failure outside range
        self.assertEqual(len(report.scores), 6)

    def test_ensemble_learner_runs_and_beats_physics_in_range(self) -> None:
        """Ensemble hybrid in-range RMSE (measured 0.068) beats physics (0.360)."""
        train, calibration, inside, _ = self._data()
        report = compare_physics_ml_hybrid(
            _physics,
            *train,
            *calibration,
            {"in": inside},
            learner="ensemble",
            alpha=0.1,
            seed=1,
        )
        self.assertLess(
            report.get("hybrid", "in").rmse, 0.5 * report.get("physics", "in").rmse
        )

    def test_interval_geometry(self) -> None:
        """Interval is symmetric about the prediction with the calibrated width."""
        train, calibration, _, _ = self._data()
        model = fit_residual_hybrid(_physics, *train, *calibration, seed=1)
        lower, upper = model.interval([1.0])
        centre = model.predict([1.0])
        self.assertAlmostEqual(0.5 * (lower + upper), centre, places=12)
        self.assertAlmostEqual(0.5 * (upper - lower), model.quantile, places=12)
        self.assertEqual(
            model.quantile, split_conformal_halfwidth(model.calibration_scores, 0.1)
        )

    def test_physics_only_reproduces_the_physics(self) -> None:
        """With ``learner='none'`` the prediction is the physics model."""
        train, calibration, _, _ = self._data()
        model = fit_residual_hybrid(
            _physics, *train, *calibration, learner="none", seed=1
        )
        self.assertEqual(model.predict([2.0]), _physics([2.0]))

    def test_rejects_invalid_input(self) -> None:
        """Unknown learner, missing physics, mismatched lengths."""
        train, calibration, _, _ = self._data()
        with self.assertRaises(ValueError):
            fit_residual_hybrid(_physics, *train, *calibration, learner="svm")
        with self.assertRaises(ValueError):
            fit_residual_hybrid(None, *train, *calibration, learner="none")
        with self.assertRaises(ValueError):
            fit_residual_hybrid(_physics, train[0], train[1][:-1], *calibration)
        with self.assertRaises(TypeError):
            fit_residual_hybrid("physics", *train, *calibration)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            compare_physics_ml_hybrid(_physics, *train, *calibration, {}, learner="gp")


class DeepONetTests(unittest.TestCase):
    """Antiderivative operator learned from generated data."""

    def test_exact_antiderivative_matches_quadrature(self) -> None:
        """Closed form versus a 2000-panel Simpson rule (error < 1e-10)."""
        coefficients = (0.3, -0.8, 0.5)
        panels = 2000
        for y in (0.25, 0.6, 1.0):
            h = y / panels
            total = 0.0
            for i in range(panels + 1):
                s = i * h
                value = (
                    coefficients[0]
                    + coefficients[1] * math.cos(math.pi * s)
                    + coefficients[2] * math.sin(math.pi * s)
                )
                weight = 1 if i in (0, panels) else (4 if i % 2 else 2)
                total += weight * value
            self.assertAlmostEqual(
                antiderivative_exact(coefficients, y), total * h / 3.0, delta=1e-10
            )

    def test_dataset_is_reproducible_and_consistent(self) -> None:
        """Same seed, same data; targets satisfy ``G(0+) -> 0`` structure."""
        first = antiderivative_dataset(4, seed=9)
        second = antiderivative_dataset(4, seed=9)
        self.assertEqual(first.branch_inputs, second.branch_inputs)
        self.assertEqual(first.targets, second.targets)
        self.assertEqual(len(first.sensors), 6)
        self.assertEqual(len(first.queries), 8)
        c = first.coefficients[0]
        self.assertAlmostEqual(
            first.branch_inputs[0][0], c[0] + c[1], places=14
        )  # u at s = 0
        self.assertAlmostEqual(
            first.targets[0][-1], antiderivative_exact(c, 1.0), places=14
        )

    def test_prediction_is_the_branch_trunk_inner_product(self) -> None:
        """``G(u)(y) = sum_k b_k t_k + b0`` evaluated by hand."""
        data = antiderivative_dataset(1, n_sensors=3, seed=2)
        fit = fit_deeponet(
            data.sensors,
            data.branch_inputs,
            data.queries,
            data.targets,
            latent=3,
            branch_hidden=(3,),
            trunk_hidden=(3,),
            iterations=3,
            seed=1,
        )
        model = fit.model
        nb = model.branch.parameter_count
        nt = model.trunk.parameter_count
        b = model.branch.forward(model.parameters[:nb], list(data.branch_inputs[0]))
        for y, predicted in zip(
            data.queries,
            model.predict(data.branch_inputs[0], data.queries),
            strict=True,
        ):
            t = model.trunk.forward(model.parameters[nb : nb + nt], [y])
            manual = model.parameters[nb + nt] + sum(
                bk * tk for bk, tk in zip(b, t, strict=True)
            )
            self.assertAlmostEqual(predicted, manual, places=13)

    def test_generalizes_to_held_out_functions(self) -> None:
        """10 training functions, 250 Adam steps (about 4 s).

        Measured relative L2 error: 0.030 on the training functions and
        0.205 on 50 held-out functions (seed 99). The trivial zero predictor
        has relative error 1; the bound 0.4 leaves a factor of about 2.
        """
        train = antiderivative_dataset(10, n_sensors=5, n_queries=6, seed=1)
        test = antiderivative_dataset(50, n_sensors=5, n_queries=6, seed=99)
        fit = fit_deeponet(
            train.sensors,
            train.branch_inputs,
            train.queries,
            train.targets,
            latent=4,
            branch_hidden=(6,),
            trunk_hidden=(6,),
            iterations=250,
            learning_rate=0.03,
            seed=2,
        )
        self.assertIsInstance(fit.model, DeepONet)
        self.assertEqual(fit.evaluations, 251)
        self.assertLess(deeponet_relative_error(fit.model, train), 0.15)
        self.assertLess(deeponet_relative_error(fit.model, test), 0.4)

    def test_rejects_invalid_input(self) -> None:
        """Shape and value validation."""
        data = antiderivative_dataset(2, seed=1)
        with self.assertRaises(ValueError):
            fit_deeponet(
                data.sensors, data.branch_inputs[:1], data.queries, data.targets
            )
        with self.assertRaises(ValueError):
            fit_deeponet(
                data.sensors[:-1], data.branch_inputs, data.queries, data.targets
            )
        with self.assertRaises(ValueError):
            antiderivative_dataset(3, n_sensors=1)
        other = antiderivative_dataset(2, n_sensors=4, seed=1)
        fit = fit_deeponet(
            data.sensors,
            data.branch_inputs,
            data.queries,
            data.targets,
            latent=2,
            branch_hidden=(2,),
            trunk_hidden=(2,),
            iterations=2,
        )
        with self.assertRaises(ValueError):
            deeponet_relative_error(fit.model, other)
        with self.assertRaises(ValueError):
            fit.model.predict([0.0], data.queries)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudies(unittest.TestCase):
    """Larger runs; execute with ``PROBSTAT_SLOW=1``."""

    def test_hybrid_ode_two_trajectories_and_distillation(self) -> None:
        """Two initial states, 6 hidden units, 300 steps, then distillation.

        Measured: correction RMS error 0.0013 against a reference of 0.071,
        forecast error ratio 0.007, distilled term ``-0.398 v^3`` (true -0.4).
        """
        helper = HybridODETests()
        result, data = helper._fit(30, (6,), 300, ((1.0, 0.0), (0.0, 1.0)))
        error, reference, ratio = helper._score(result, 30)
        self.assertLess(error, 0.05 * reference)
        self.assertLess(ratio, 0.05)
        states = [s for trajectory in data for s in trajectory]
        fit = distill_sparse_regression(
            states,
            [result.correction(s)[0] for s in states],
            degree=3,
            threshold=0.05,
            variable_names=("x", "v"),
        )
        terms = fit.terms()
        self.assertEqual(list(terms), ["v^3"])
        self.assertAlmostEqual(terms["v^3"], -CUBIC, delta=0.02)

    def test_deeponet_more_functions_and_steps(self) -> None:
        """20 training functions, latent 6, 1500 Adam steps (about 1 min).

        Measured held-out relative L2 error 0.042 (training 0.031).
        """
        train = antiderivative_dataset(20, seed=1)
        test = antiderivative_dataset(50, seed=99)
        fit = fit_deeponet(
            train.sensors,
            train.branch_inputs,
            train.queries,
            train.targets,
            latent=6,
            branch_hidden=(10,),
            trunk_hidden=(10,),
            iterations=1500,
            learning_rate=0.01,
            seed=2,
        )
        self.assertLess(deeponet_relative_error(fit.model, test), 0.1)


if __name__ == "__main__":
    unittest.main()
