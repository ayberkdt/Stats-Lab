"""Tests for flow matching, flow-matching posteriors, and ratio estimation.

Oracles:

* Gaussian data ``N(m, sd^2)`` have the closed-form marginal velocity
  ``v(t, x) = m + c(t) (x - t m)`` [EQ-GEN-FM-003]. Integrating it must map
  ``z`` to ``m + sqrt(sd^2 + s^2) z`` and the instantaneous change of
  variables must return the exact normal log-density, which tests the ODE
  sampler and the density code without any training. The field itself is
  checked against a Monte Carlo regression of ``u`` on ``x_t``.
* Backpropagation on floats equals reverse-mode autodiff of the same loss.
* Trained networks: a 1-D Gaussian (mean, sd), a 1-D two-component mixture
  (mode masses), the linear-Gaussian simulator ``x = 2 theta + e`` with
  exact posterior ``N(2 x / 5, 1 / 5)`` and exact log-ratio, and
  ``x = theta^2 + e`` whose posterior is bimodal, where the Gaussian
  amortized estimator of ``probstat.inference`` must fail.

Tolerances for trained networks are about twice the largest error seen over
six seeds at the same settings (ADR-0129 lists the spread).
"""

from __future__ import annotations

import math
import random
import statistics
import unittest
from collections.abc import Sequence

from probstat.foundations.random_streams import RandomStream
from probstat.generative import (
    FlowMatchingFit,
    FlowMatchingPosterior,
    GaussianMarginalVelocity,
    GridPosterior,
    RatioEstimator,
    VelocityNetwork,
    conditional_flow_path,
    fit_flow_matching,
    fit_flow_matching_posterior,
    fit_ratio_estimator,
    flow_log_density,
    flow_matching_loss_and_gradient,
    integrate_flow,
    ratio_loss_and_gradient,
)
from probstat.inference import (
    fit_amortized_gaussian_posterior,
    simulation_based_calibration,
)
from probstat.numerics import MLP, admath, reverse_value_and_gradient

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
POSTERIOR_SD = math.sqrt(0.2)


def _normal_log_density(x: float, mean: float, sd: float) -> float:
    """Return the log-density of N(mean, sd^2) at x."""
    return -0.5 * ((x - mean) / sd) ** 2 - math.log(sd) - LOG_SQRT_TWO_PI


def _normal_prior(rng: random.Random) -> list[float]:
    """Draw theta from N(0, 1)."""
    return [rng.gauss(0.0, 1.0)]


def _linear_simulator(theta: Sequence[float], rng: random.Random) -> list[float]:
    """Return x = 2 theta + N(0, 1)."""
    return [2.0 * theta[0] + rng.gauss(0.0, 1.0)]


def _uniform_prior(rng: random.Random) -> list[float]:
    """Draw theta from U(-2, 2)."""
    return [rng.uniform(-2.0, 2.0)]


def _square_simulator(theta: Sequence[float], rng: random.Random) -> list[float]:
    """Return x = theta^2 + N(0, 0.2^2), which hides the sign of theta."""
    return [theta[0] ** 2 + rng.gauss(0.0, 0.2)]


def _normal_log_prior(theta: Sequence[float]) -> float:
    """Return the log-density of the N(0, 1) prior."""
    return -0.5 * theta[0] ** 2 - LOG_SQRT_TWO_PI


def _exact_log_ratio(theta: float, x: float) -> float:
    """Return log p(x | theta) / p(x) for x = 2 theta + N(0, 1), theta ~ N(0, 1)."""
    return -0.5 * (x - 2.0 * theta) ** 2 + x * x / 10.0 + 0.5 * math.log(5.0)


class ConditionalPathTests(unittest.TestCase):
    """The path and its velocity follow their definitions."""

    def test_endpoints_and_velocity(self) -> None:
        """x_0 at t=0, x_1 + s x_0 at t=1, and u = x_1 - (1 - s) x_0."""
        x0, x1, s = (0.3, -1.2), (2.0, 0.5), 0.1
        start, velocity = conditional_flow_path(x0, x1, 0.0, sigma_min=s)
        end, _ = conditional_flow_path(x0, x1, 1.0, sigma_min=s)
        for k in range(2):
            self.assertAlmostEqual(start[k], x0[k], places=14)
            self.assertAlmostEqual(end[k], x1[k] + s * x0[k], places=14)
            self.assertAlmostEqual(velocity[k], x1[k] - 0.9 * x0[k], places=14)

    def test_velocity_is_the_time_derivative_of_the_path(self) -> None:
        """A central difference of x_t in t reproduces u, and the field form."""
        x0, x1, s, t, h = 0.7, -1.5, 0.05, 0.4, 1e-6
        point, velocity = conditional_flow_path(x0, x1, t, sigma_min=s)
        upper, _ = conditional_flow_path(x0, x1, t + h, sigma_min=s)
        lower, _ = conditional_flow_path(x0, x1, t - h, sigma_min=s)
        self.assertAlmostEqual((upper[0] - lower[0]) / (2 * h), velocity[0], places=8)
        field = (x1 - (1 - s) * point[0]) / (1 - (1 - s) * t)
        self.assertAlmostEqual(field, velocity[0], places=12)

    def test_invalid_inputs(self) -> None:
        """Bad times, widths, and dimensions are rejected."""
        with self.assertRaises(ValueError):
            conditional_flow_path(0.0, 1.0, 1.5)
        with self.assertRaises(ValueError):
            conditional_flow_path(0.0, 1.0, 0.5, sigma_min=1.0)
        with self.assertRaises(ValueError):
            conditional_flow_path((0.0, 1.0), (1.0,), 0.5)
        with self.assertRaises(ValueError):
            conditional_flow_path(math.nan, 1.0, 0.5)


class GaussianMarginalFieldTests(unittest.TestCase):
    """The exact Gaussian field tests the sampler and the density code."""

    def test_field_is_the_regression_of_u_on_x_t(self) -> None:
        """Least squares of u on x_t over 40000 draws gives m - c t m and c."""
        mean, sd, s, t = 1.5, 0.7, 0.1, 0.6
        field = GaussianMarginalVelocity(mean, sd, s)
        rng = random.Random(3)
        xs, us = [], []
        for _ in range(40000):
            point, velocity = conditional_flow_path(
                rng.gauss(0.0, 1.0), rng.gauss(mean, sd), t, sigma_min=s
            )
            xs.append(point[0])
            us.append(velocity[0])
        slope, intercept = statistics.linear_regression(xs, us)
        exact_slope = field.slope(t)[0]
        self.assertAlmostEqual(slope, exact_slope, delta=0.02)
        self.assertAlmostEqual(intercept, mean - exact_slope * t * mean, delta=0.02)
        variance = (1 - 0.9 * t) ** 2 + (t * sd) ** 2
        self.assertAlmostEqual(
            exact_slope, (2 * t * sd**2 - 1.8 * (1 - 0.9 * t)) / (2 * variance)
        )

    def test_flow_map_is_the_affine_transport(self) -> None:
        """Integration maps z to m + sqrt(sd^2 + s^2) z to 1e-6 (RK4, 20 steps)."""
        for s in (0.0, 0.2):
            field = GaussianMarginalVelocity(1.5, 0.7, s)
            scale = math.sqrt(field.terminal_variance[0])
            for z in (-2.0, -0.5, 0.0, 1.0, 2.5):
                self.assertAlmostEqual(
                    integrate_flow(field, z)[0], 1.5 + scale * z, delta=1e-6
                )

    def test_transported_sample_moments(self) -> None:
        """2000 N(0,1) draws arrive with the mean and variance of N(m, sd^2)."""
        field = GaussianMarginalVelocity(-0.4, 1.8)
        rng = random.Random(8)
        base = [rng.gauss(0.0, 1.0) for _ in range(2000)]
        moved = [integrate_flow(field, z, steps=10)[0] for z in base]
        self.assertAlmostEqual(
            statistics.fmean(moved), -0.4 + 1.8 * statistics.fmean(base), delta=1e-4
        )
        self.assertAlmostEqual(
            statistics.pstdev(moved), 1.8 * statistics.pstdev(base), delta=1e-4
        )
        self.assertAlmostEqual(statistics.fmean(moved), -0.4, delta=4 * 1.8 / 44.7)
        self.assertAlmostEqual(statistics.pstdev(moved), 1.8, delta=0.12)

    def test_log_density_matches_the_exact_normal(self) -> None:
        """All three divergence modes agree with log N(x; m, sd^2 + s^2)."""
        for s in (0.0, 0.3):
            field = GaussianMarginalVelocity(1.5, 0.7, s)
            sd = math.sqrt(field.terminal_variance[0])
            for x in (0.2, 1.5, 2.4, 3.5):
                exact = _normal_log_density(x, 1.5, sd)
                for mode in ("autodiff", "finite-difference", field.divergence):
                    self.assertAlmostEqual(
                        flow_log_density(field, x, divergence=mode), exact, delta=2e-6
                    )

    def test_two_dimensional_log_density(self) -> None:
        """Independent coordinates give the product density."""
        field = GaussianMarginalVelocity((1.0, -2.0), (0.5, 2.0))
        x = (1.3, -0.5)
        exact = _normal_log_density(x[0], 1.0, 0.5) + _normal_log_density(
            x[1], -2.0, 2.0
        )
        self.assertAlmostEqual(flow_log_density(field, x, steps=40), exact, delta=1e-6)
        moved = integrate_flow(field, (0.5, -1.0), steps=40)
        self.assertAlmostEqual(moved[0], 1.25, delta=1e-6)
        self.assertAlmostEqual(moved[1], -4.0, delta=1e-6)

    def test_solver_order(self) -> None:
        """Euler is first order: halving the step roughly halves the error."""
        field = GaussianMarginalVelocity(1.5, 0.7)
        coarse = abs(integrate_flow(field, 1.0, steps=20, method="euler")[0] - 2.2)
        fine = abs(integrate_flow(field, 1.0, steps=40, method="euler")[0] - 2.2)
        self.assertGreater(coarse, 1e-4)
        self.assertAlmostEqual(coarse / fine, 2.0, delta=0.3)

    def test_invalid_inputs(self) -> None:
        """Bad moments, settings, and callables are rejected."""
        with self.assertRaises(ValueError):
            GaussianMarginalVelocity(0.0, 0.0)
        with self.assertRaises(ValueError):
            GaussianMarginalVelocity((0.0, 1.0), (1.0,))
        field = GaussianMarginalVelocity(0.0, 1.0)
        with self.assertRaises(ValueError):
            field(0.5, [0.0, 1.0])
        with self.assertRaises(ValueError):
            integrate_flow(field, 0.0, steps=0)
        with self.assertRaises(ValueError):
            integrate_flow(field, 0.0, method="midpoint")
        with self.assertRaises(TypeError):
            integrate_flow(None, 0.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            flow_log_density(field, 0.0, divergence="exact")
        with self.assertRaises(ValueError):
            flow_log_density(
                field, 0.0, divergence="finite-difference", difference_step=0.0
            )


class BackpropagationTests(unittest.TestCase):
    """Float backpropagation equals reverse-mode autodiff."""

    def test_flow_matching_gradient(self) -> None:
        """Loss and gradient agree with the tape to 1e-12 (2-D, conditioned)."""
        network = VelocityNetwork(2, 1, (5, 4))
        parameters = network.initial_parameters(4)
        rng = random.Random(5)
        cases = [
            (
                rng.random(),
                [rng.gauss(0, 1), rng.gauss(0, 1)],
                [rng.gauss(0, 1), rng.gauss(0, 1)],
                [rng.gauss(0, 1)],
            )
            for _ in range(6)
        ]

        def loss(phi: Sequence[object]) -> object:
            """Return the mean squared velocity error on the tape."""
            total: object = 0.0
            for t, x, u, c in cases:
                out = network.forward(phi, t, x, c)
                for value, target in zip(out, u, strict=True):
                    total = total + (value - target) * (value - target)
            return total / len(cases)  # type: ignore[operator]

        value, gradient = reverse_value_and_gradient(loss, parameters)
        fast_value, fast_gradient = flow_matching_loss_and_gradient(
            network, parameters, cases
        )
        self.assertAlmostEqual(fast_value, value, places=12)
        self.assertEqual(len(fast_gradient), network.parameter_count)
        for a, b in zip(fast_gradient, gradient, strict=True):
            self.assertAlmostEqual(a, b, places=12)

    def test_ratio_gradient(self) -> None:
        """The cross-entropy gradient agrees with the tape to 1e-12."""
        network = MLP((2, 6, 3, 1))
        parameters = network.initial_parameters(9)
        rng = random.Random(6)
        joint = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(4)]
        product = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(4)]

        def loss(phi: Sequence[object]) -> object:
            """Return the balanced cross-entropy on the tape."""
            total: object = 0.0
            for sign, rows in ((-1.0, joint), (1.0, product)):
                for row in rows:
                    logit = network.forward(phi, row)[0]
                    total = total + admath.log1p(admath.exp(sign * logit))
            return total / 8.0  # type: ignore[operator]

        value, gradient = reverse_value_and_gradient(loss, parameters)
        fast_value, fast_gradient = ratio_loss_and_gradient(
            network, parameters, joint, product
        )
        self.assertAlmostEqual(fast_value, value, places=12)
        for a, b in zip(fast_gradient, gradient, strict=True):
            self.assertAlmostEqual(a, b, places=12)

    def test_network_divergence_and_fast_forward(self) -> None:
        """Exact divergence equals dual-number and central-difference values."""
        network = VelocityNetwork(2, 1, (6, 5))
        parameters = network.initial_parameters(2)
        t, x, c = 0.3, [0.4, -0.9], [1.1]
        slow = network.forward(parameters, t, x, c)
        fast = network.velocity(parameters, t, x, c)
        for a, b in zip(fast, slow, strict=True):
            self.assertAlmostEqual(a, float(b), places=13)
        exact = network.divergence(parameters, t, x, c)

        def field(time: float, point: Sequence[object]) -> list[object]:
            """Return the network velocity for any numeric kind."""
            return network.forward(parameters, time, point, c)

        # flow_log_density with zero steps is not available, so compare the
        # divergence through one Euler step: the accumulated term is div v.
        dual = -flow_log_density(field, x, steps=1, method="euler") + (
            flow_log_density(
                field, x, steps=1, method="euler", divergence=lambda *_: 0.0
            )
        )
        self.assertAlmostEqual(
            dual, network.divergence(parameters, 1.0, x, c), places=12
        )
        h = 1e-5
        numeric = sum(
            (
                network.velocity(
                    parameters, t, [v + h * (j == k) for j, v in enumerate(x)], c
                )[k]
                - network.velocity(
                    parameters, t, [v - h * (j == k) for j, v in enumerate(x)], c
                )[k]
            )
            / (2 * h)
            for k in range(2)
        )
        self.assertAlmostEqual(exact, numeric, places=8)

    def test_invalid_inputs(self) -> None:
        """Empty batches and wrong sizes are rejected."""
        network = VelocityNetwork(1)
        parameters = network.initial_parameters(0)
        with self.assertRaises(ValueError):
            flow_matching_loss_and_gradient(network, parameters, [])
        with self.assertRaises(ValueError):
            flow_matching_loss_and_gradient(
                network, parameters[:-1], [(0.5, [0.0], [1.0], [])]
            )
        with self.assertRaises(ValueError):
            network.velocity(parameters, 0.5, [0.0, 1.0])
        with self.assertRaises(ValueError):
            VelocityNetwork(0)
        with self.assertRaises(ValueError):
            VelocityNetwork(1, -1)
        with self.assertRaises(ValueError):
            VelocityNetwork(1, 0, ())
        with self.assertRaises(ValueError):
            ratio_loss_and_gradient(MLP((2, 3, 1)), (0.0,) * 13, [], [[0.0, 0.0]])


class TrainedGaussianFlowTests(unittest.TestCase):
    """A small network reproduces a 1-D Gaussian."""

    fit: FlowMatchingFit
    data: list[float]

    @classmethod
    def setUpClass(cls) -> None:
        """Fit 400 draws of N(1.5, 0.7^2) with one hidden layer of 16."""
        rng = random.Random(100)
        cls.data = [rng.gauss(1.5, 0.7) for _ in range(400)]
        cls.fit = fit_flow_matching(
            cls.data, hidden=(16,), iterations=600, batch_size=32, seed=0
        )

    def test_sample_mean_and_standard_deviation(self) -> None:
        """400 draws: mean within 0.15 and sd within 0.1 of the truth."""
        draws = [p[0] for p in self.fit.sample(400, steps=10, seed=0)]
        self.assertAlmostEqual(statistics.fmean(draws), 1.5, delta=0.15)
        self.assertAlmostEqual(statistics.pstdev(draws), 0.7, delta=0.1)

    def test_log_density(self) -> None:
        """The density integrates to one and is within 0.25 nats in the bulk."""
        grid = [-2.5 + 8.0 * i / 160 for i in range(161)]
        density = [math.exp(self.fit.log_density(x, steps=10)) for x in grid]
        self.assertAlmostEqual(sum(density) * 0.05, 1.0, delta=0.01)
        for x in (0.8, 1.5, 2.2):
            self.assertAlmostEqual(
                self.fit.log_density(x), _normal_log_density(x, 1.5, 0.7), delta=0.25
            )

    def test_reporting_and_reproducibility(self) -> None:
        """Loss trace, flag, counts, and seeded reproducibility."""
        self.assertEqual(len(self.fit.losses), 600)
        self.assertEqual(self.fit.iterations, 600)
        self.assertEqual(self.fit.training_points, 400)
        self.assertTrue(self.fit.loss_decreased)
        self.assertGreater(self.fit.final_loss, 0.5)
        again = fit_flow_matching(
            self.data, hidden=(16,), iterations=30, batch_size=32, seed=3
        )
        twin = fit_flow_matching(
            self.data, hidden=(16,), iterations=30, batch_size=32, seed=3
        )
        self.assertEqual(again.parameters, twin.parameters)
        self.assertEqual(again.sample(3, seed=1), twin.sample(3, seed=1))
        self.assertNotEqual(again.sample(3, seed=1), again.sample(3, seed=2))

    def test_invalid_inputs(self) -> None:
        """Too few points, bad settings, and bad queries are rejected."""
        with self.assertRaises(ValueError):
            fit_flow_matching([1.0])
        with self.assertRaises(ValueError):
            fit_flow_matching([1.0, math.inf])
        with self.assertRaises(ValueError):
            fit_flow_matching(self.data, iterations=0)
        with self.assertRaises(ValueError):
            fit_flow_matching(self.data, sigma_min=-0.1)
        with self.assertRaises(ValueError):
            fit_flow_matching(self.data, learning_rate=0.0)
        with self.assertRaises(ValueError):
            self.fit.sample(0)
        with self.assertRaises(ValueError):
            self.fit.log_density((0.0, 1.0))


class TrainedMixtureFlowTests(unittest.TestCase):
    """A two-layer network captures both modes of a 1-D mixture."""

    def test_mode_masses_and_locations(self) -> None:
        """0.3 N(-2, 0.5^2) + 0.7 N(2, 0.5^2): masses within 0.08, means 0.25."""
        rng = random.Random(200)
        data = [
            rng.gauss(-2.0, 0.5) if rng.random() < 0.3 else rng.gauss(2.0, 0.5)
            for _ in range(500)
        ]
        fit = fit_flow_matching(
            data, hidden=(16, 16), iterations=1200, batch_size=32, seed=0
        )
        draws = [p[0] for p in fit.sample(500, steps=10, seed=0)]
        left = [v for v in draws if v < 0.0]
        right = [v for v in draws if v >= 0.0]
        data_mass = sum(v < 0.0 for v in data) / len(data)
        self.assertAlmostEqual(len(left) / 500, data_mass, delta=0.08)
        self.assertAlmostEqual(statistics.fmean(left), -2.0, delta=0.25)
        self.assertAlmostEqual(statistics.fmean(right), 2.0, delta=0.25)
        self.assertLess(statistics.pstdev(right), 0.75)
        self.assertLess(sum(abs(v) < 1.0 for v in draws) / 500, 0.12)


class LinearGaussianPosteriorTests(unittest.TestCase):
    """Flow-matching posterior against the exact conjugate posterior."""

    posterior: FlowMatchingPosterior

    @classmethod
    def setUpClass(cls) -> None:
        """Train once on 1000 simulations of x = 2 theta + e."""
        cls.posterior = fit_flow_matching_posterior(
            _normal_prior,
            _linear_simulator,
            1000,
            hidden_layers=(16, 16),
            iterations=1200,
            batch_size=32,
            learning_rate=0.01,
            seed=1,
        )

    def test_moments_at_several_observations(self) -> None:
        """Mean 2x/5 within 0.12 and sd sqrt(1/5) within 0.08 at four x."""
        for x in (-3.0, 0.0, 2.0, 4.0):
            draws = [p[0] for p in self.posterior.sample([x], 300, steps=10, seed=1)]
            self.assertAlmostEqual(statistics.fmean(draws), 0.4 * x, delta=0.12)
            self.assertAlmostEqual(statistics.pstdev(draws), POSTERIOR_SD, delta=0.08)

    def test_log_density_is_normalized(self) -> None:
        """exp(log q(theta | x)) integrates to one over theta at x = 2."""
        grid = [-1.7 + 5.0 * i / 100 for i in range(101)]
        density = [
            math.exp(self.posterior.log_density([v], [2.0], steps=10)) for v in grid
        ]
        self.assertAlmostEqual(sum(density) * 0.05, 1.0, delta=0.02)
        peak = grid[max(range(101), key=density.__getitem__)]
        self.assertAlmostEqual(peak, 0.8, delta=0.25)

    def test_simulation_based_calibration(self) -> None:
        """Ranks are not rejected at 1%; an overconfident posterior is."""
        posterior = self.posterior

        def sampler(
            data: object, stream: RandomStream
        ) -> tuple[tuple[float, ...], ...]:
            """Return 19 posterior draws for one simulated data set."""
            return posterior.sample(data, 19, steps=8, seed=stream)  # type: ignore[arg-type]

        def narrow(data: object, stream: RandomStream) -> list[tuple[float, ...]]:
            """Return draws shrunk to half their spread (overconfident)."""
            draws = sampler(data, stream)
            center = statistics.fmean(d[0] for d in draws)
            return [(center + 0.5 * (d[0] - center),) for d in draws]

        good = simulation_based_calibration(
            _normal_prior, _linear_simulator, sampler, replicates=200, seed=1
        )
        self.assertEqual(good.draws, 19)
        self.assertGreater(good.p_values[0], 0.01)
        bad = simulation_based_calibration(
            _normal_prior, _linear_simulator, narrow, replicates=100, seed=1
        )
        self.assertLess(bad.p_values[0], 1e-4)

    def test_reporting_and_validation(self) -> None:
        """Counts, loss flag, reproducibility, and input checks."""
        self.assertEqual(self.posterior.simulations, 1000)
        self.assertEqual(self.posterior.dimension, 1)
        self.assertEqual(len(self.posterior.losses), 1200)
        self.assertTrue(self.posterior.loss_decreased)
        self.assertEqual(
            self.posterior.sample([1.0], 2, seed=4),
            self.posterior.sample([1.0], 2, seed=4),
        )
        with self.assertRaises(ValueError):
            self.posterior.sample([1.0, 2.0], 2)
        with self.assertRaises(ValueError):
            self.posterior.sample([1.0], 0)
        with self.assertRaises(ValueError):
            self.posterior.log_density([0.0, 0.0], [1.0])
        with self.assertRaises(TypeError):
            fit_flow_matching_posterior(None, _linear_simulator, 10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_flow_matching_posterior(_normal_prior, _linear_simulator, 1)
        with self.assertRaises(ValueError):
            fit_flow_matching_posterior(
                _normal_prior, lambda theta, rng: [math.nan], 10
            )
        with self.assertRaises(ValueError):
            fit_flow_matching_posterior(
                _normal_prior, _linear_simulator, 10, iterations=0
            )


class BimodalPosteriorTests(unittest.TestCase):
    """x = theta^2 + e: the flow finds two modes, the Gaussian cannot."""

    def test_two_modes_where_the_gaussian_estimator_fails(self) -> None:
        """At x = 1: masses within 0.1 of 1/2 at +-1; Gaussian centred near 0."""
        posterior = fit_flow_matching_posterior(
            _uniform_prior,
            _square_simulator,
            1000,
            hidden_layers=(16, 16),
            iterations=1200,
            batch_size=32,
            learning_rate=0.01,
            seed=0,
        )
        draws = [p[0] for p in posterior.sample([1.0], 400, steps=10, seed=1)]
        left = [v for v in draws if v < 0.0]
        right = [v for v in draws if v >= 0.0]
        self.assertAlmostEqual(len(left) / 400, 0.5, delta=0.1)
        self.assertAlmostEqual(statistics.fmean(left), -1.0, delta=0.15)
        self.assertAlmostEqual(statistics.fmean(right), 1.0, delta=0.15)
        self.assertLess(statistics.pstdev(right), 0.35)
        self.assertLess(sum(abs(v) < 0.5 for v in draws) / 400, 0.12)
        gaussian = fit_amortized_gaussian_posterior(
            _uniform_prior,
            _square_simulator,
            1000,
            hidden_layers=(8,),
            iterations=400,
            batch_size=32,
            seed=0,
        )
        means, sds = gaussian.moments([1.0])
        self.assertLess(abs(means[0]), 0.3)
        self.assertGreater(sds[0], 0.8)
        # The Gaussian puts its highest density where the posterior has none.
        self.assertGreater(
            gaussian.log_density([0.0], [1.0]), gaussian.log_density([1.0], [1.0])
        )
        self.assertGreater(
            posterior.log_density([1.0], [1.0]),
            posterior.log_density([0.0], [1.0]) + 1.0,
        )


class RatioEstimatorTests(unittest.TestCase):
    """Neural ratio estimation against the exact linear-Gaussian ratio."""

    estimator: RatioEstimator

    @classmethod
    def setUpClass(cls) -> None:
        """Train once on 1000 simulations of x = 2 theta + e."""
        cls.estimator = fit_ratio_estimator(
            _normal_prior,
            _linear_simulator,
            1000,
            hidden_layers=(16, 16),
            iterations=1000,
            batch_size=32,
            learning_rate=0.01,
            seed=1,
        )

    def test_logit_matches_the_exact_log_ratio(self) -> None:
        """Within +-2 posterior sd: correlation > 0.9, mean |error| < 0.4."""
        for x in (-2.0, 0.0, 1.0, 3.0):
            grid = [0.4 * x + POSTERIOR_SD * (-2.0 + 0.2 * i) for i in range(21)]
            learned = [self.estimator.log_ratio([v], [x]) for v in grid]
            exact = [_exact_log_ratio(v, x) for v in grid]
            self.assertGreater(statistics.correlation(learned, exact), 0.9)
            error = statistics.fmean(
                abs(a - b) for a, b in zip(learned, exact, strict=True)
            )
            self.assertLess(error, 0.4)

    def test_grid_posterior_moments(self) -> None:
        """Mean within 0.12, sd within 0.06, normalization within 0.3 of one."""
        grid = [-4.0 + 0.02 * i for i in range(401)]
        for x in (-2.0, 0.0, 1.0, 3.0):
            result = self.estimator.posterior_on_grid([x], _normal_log_prior, grid)
            self.assertIsInstance(result, GridPosterior)
            self.assertAlmostEqual(result.mean, 0.4 * x, delta=0.12)
            self.assertAlmostEqual(result.standard_deviation, POSTERIOR_SD, delta=0.06)
            self.assertAlmostEqual(result.ratio_normalization, 1.0, delta=0.3)
            self.assertAlmostEqual(sum(result.density) * 0.02, 1.0, delta=1e-6)

    def test_mcmc_agrees_with_the_grid(self) -> None:
        """Random-walk Metropolis on prior + logit matches the grid moments."""
        grid = [-4.0 + 0.02 * i for i in range(401)]
        reference = self.estimator.posterior_on_grid([1.0], _normal_log_prior, grid)
        result = self.estimator.sample_posterior(
            [1.0],
            _normal_log_prior,
            [0.4],
            proposal_scale=0.8,
            draws=3000,
            warmup=200,
            chains=2,
            seed=2,
        )
        draws = [s[0] for chain in result.chains for s in chain.samples]
        self.assertAlmostEqual(statistics.fmean(draws), reference.mean, delta=0.06)
        self.assertAlmostEqual(
            statistics.pstdev(draws), reference.standard_deviation, delta=0.05
        )

    def test_reporting_and_validation(self) -> None:
        """Counts, the loss below log 2, and input checks."""
        self.assertEqual(self.estimator.simulations, 1000)
        self.assertTrue(self.estimator.loss_decreased)
        self.assertLess(self.estimator.final_loss, math.log(2.0))
        grid = [-1.0, 0.0, 1.0]
        with self.assertRaises(ValueError):
            self.estimator.posterior_on_grid([0.0], _normal_log_prior, [0.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            self.estimator.posterior_on_grid([0.0], lambda theta: -math.inf, grid)
        with self.assertRaises(ValueError):
            self.estimator.posterior_on_grid([0.0], lambda theta: math.nan, grid)
        with self.assertRaises(TypeError):
            self.estimator.posterior_on_grid([0.0], None, grid)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.estimator.log_ratio([0.0, 1.0], [0.0])
        with self.assertRaises(ValueError):
            self.estimator.sample_posterior(
                [0.0], _normal_log_prior, [0.0], proposal_scale=-1.0, draws=10
            )
        with self.assertRaises(ValueError):
            fit_ratio_estimator(_normal_prior, _linear_simulator, 10, batch_size=1)
        with self.assertRaises(ValueError):
            fit_ratio_estimator(_normal_prior, _linear_simulator, 10, hidden_layers=())


if __name__ == "__main__":
    unittest.main()
