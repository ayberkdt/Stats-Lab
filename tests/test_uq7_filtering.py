"""Tests for UQ-7 state-space filtering, smoothing, and data assimilation."""

from __future__ import annotations

import math
import unittest

from probstat.data_assimilation import (
    Ensemble,
    ensemble_analysis,
    ensemble_kalman_filter,
    inflate,
    three_dimensional_variational,
)
from probstat.filtering import (
    GaussianState,
    LinearGaussianStateSpace,
    NonlinearStateSpace,
    bootstrap_particle_filter,
    effective_sample_size,
    extended_kalman_filter,
    kalman_filter,
    kalman_predict,
    kalman_update,
    rts_smoother,
    simulate,
    steady_state_kalman,
    systematic_resample,
    unscented_kalman_filter,
    unscented_transform,
)
from probstat.inference import condition_linear_gaussian
from probstat.linalg import add, identity, matmul, sandwich, transpose
from probstat.multivariate import MultivariateNormal


def _random_walk(q: float = 0.5, r: float = 2.0) -> LinearGaussianStateSpace:
    """Return a scalar random walk observed with additive noise."""
    return LinearGaussianStateSpace(
        ((1.0,),), ((q,),), ((1.0,),), ((r,),), (0.0,), ((10.0,),)
    )


def _constant_velocity(
    dt: float = 1.0, q: float = 0.05, r: float = 1.0
) -> LinearGaussianStateSpace:
    """Return the white-noise-acceleration constant-velocity tracking model."""
    transition = ((1.0, dt), (0.0, 1.0))
    process = (
        (q * dt**3 / 3.0, q * dt**2 / 2.0),
        (q * dt**2 / 2.0, q * dt),
    )
    return LinearGaussianStateSpace(
        transition,
        process,
        ((1.0, 0.0),),
        ((r,),),
        (0.0, 1.0),
        ((1.0, 0.0), (0.0, 0.5)),
    )


def _power(matrix, exponent):
    """Return an integer matrix power."""
    result = identity(len(matrix))
    for _ in range(exponent):
        result = matmul(result, matrix)
    return result


def _batch_prior(model: LinearGaussianStateSpace, steps: int) -> MultivariateNormal:
    """Return the joint Gaussian law of the stacked states ``x_0..x_T``."""
    n = model.state_dimension
    f = model.transition_matrix
    blocks = {}
    for i in range(steps + 1):
        for k in range(steps + 1):
            block = matmul(
                matmul(_power(f, i), model.initial_covariance),
                transpose(_power(f, k)),
            )
            for j in range(1, min(i, k) + 1):
                block = add(
                    block,
                    matmul(
                        matmul(_power(f, i - j), model.process_covariance),
                        transpose(_power(f, k - j)),
                    ),
                )
            blocks[i, k] = block
    covariance = tuple(
        tuple(blocks[i // n, k // n][i % n][k % n] for k in range(n * (steps + 1)))
        for i in range(n * (steps + 1))
    )
    mean = []
    state = model.initial_mean
    for _ in range(steps + 1):
        mean.extend(state)
        state = model.transition(state)
    return MultivariateNormal(tuple(mean), covariance)


class KalmanFilterTests(unittest.TestCase):
    """Exact linear-Gaussian filtering and smoothing identities."""

    def test_scalar_random_walk_reaches_closed_form_steady_state(self) -> None:
        """Gain and variances converge to the scalar Riccati fixed point."""
        q, r = 0.5, 2.0
        model = _random_walk(q, r)
        data = simulate(model, 80, seed=1).observations
        result = kalman_filter(model, data)
        predicted = (q + math.sqrt(q * q + 4.0 * q * r)) / 2.0
        gain = predicted / (predicted + r)
        self.assertAlmostEqual(result.predicted[-1].covariance[0][0], predicted, 12)
        self.assertAlmostEqual(result.gains[-1][0][0], gain, 12)
        self.assertAlmostEqual(result.filtered[-1].covariance[0][0], gain * r, 12)
        steady = steady_state_kalman(model)
        self.assertTrue(steady.converged)
        self.assertAlmostEqual(steady.predicted_covariance[0][0], predicted, 10)
        self.assertAlmostEqual(steady.gain[0][0], gain, 10)
        self.assertAlmostEqual(steady.filtered_covariance[0][0], gain * r, 10)

    def test_noise_roles_move_the_gain_in_opposite_directions(self) -> None:
        """More process noise raises the gain; more measurement noise lowers it."""
        base = steady_state_kalman(_random_walk(0.5, 2.0)).gain[0][0]
        noisier_dynamics = steady_state_kalman(_random_walk(2.0, 2.0)).gain[0][0]
        noisier_sensor = steady_state_kalman(_random_walk(0.5, 8.0)).gain[0][0]
        self.assertGreater(noisier_dynamics, base)
        self.assertLess(noisier_sensor, base)

    def test_filter_and_smoother_equal_batch_gaussian_conditioning(self) -> None:
        """KF last state, RTS marginals, and likelihood match batch conditioning."""
        model = _constant_velocity()
        steps = 4
        data = simulate(model, steps, seed=5).observations
        prior = _batch_prior(model, steps)
        n = model.state_dimension
        design = tuple(
            tuple(
                model.observation_matrix[0][c % n] if c // n == k else 0.0
                for c in range(n * (steps + 1))
            )
            for k in range(1, steps + 1)
        )
        y = tuple(observation[0] for observation in data)
        variance = model.observation_covariance[0][0]
        posterior = condition_linear_gaussian(
            prior, design, y, observation_variance=variance
        )
        filtered = kalman_filter(model, data)
        smoothed = rts_smoother(filtered)
        last = filtered.filtered[-1]
        for i in range(n):
            self.assertAlmostEqual(last.mean[i], posterior.mean[steps * n + i], 9)
            for j in range(n):
                self.assertAlmostEqual(
                    last.covariance[i][j],
                    posterior.covariance[steps * n + i][steps * n + j],
                    9,
                )
        for k, state in enumerate(smoothed.smoothed):
            for i in range(n):
                self.assertAlmostEqual(state.mean[i], posterior.mean[k * n + i], 9)
                self.assertAlmostEqual(
                    state.covariance[i][i],
                    posterior.covariance[k * n + i][k * n + i],
                    9,
                )
        marginal = MultivariateNormal(
            tuple(
                sum(a * b for a, b in zip(row, prior.mean, strict=True))
                for row in design
            ),
            add(sandwich(design, prior.covariance), identity(steps), scale=variance),
        )
        self.assertAlmostEqual(filtered.log_likelihood, marginal.log_pdf(y), 9)

    def test_smoother_variance_never_exceeds_filter_variance(self) -> None:
        """Smoothing uses future data, so its marginal variances are smaller."""
        model = _constant_velocity()
        data = simulate(model, 30, seed=2).observations
        filtered = kalman_filter(model, data)
        smoothed = rts_smoother(filtered)
        for f_state, s_state in zip(filtered.filtered, smoothed.smoothed, strict=True):
            for f_var, s_var in zip(f_state.variances, s_state.variances, strict=True):
                self.assertLessEqual(s_var, f_var + 1e-12)
        self.assertEqual(smoothed.smoothed[-1], filtered.filtered[-1])
        self.assertLess(
            smoothed.smoothed[10].variances[0], filtered.filtered[10].variances[0]
        )

    def test_constant_velocity_tracking_recovers_unmeasured_velocity(self) -> None:
        """Filtering beats raw measurements and estimates the hidden velocity."""
        model = _constant_velocity(q=0.01, r=4.0)
        simulation = simulate(model, 150, seed=11)
        filtered = kalman_filter(model, simulation.observations)
        smoothed = rts_smoother(filtered)
        truth = simulation.states[1:]

        def rmse(estimates, index):
            """Return the RMSE of one state component."""
            errors = [
                (e[index] - t[index]) ** 2
                for e, t in zip(estimates, truth, strict=True)
            ]
            return math.sqrt(sum(errors) / len(errors))

        raw = math.sqrt(
            sum(
                (y[0] - t[0]) ** 2
                for y, t in zip(simulation.observations, truth, strict=True)
            )
            / len(truth)
        )
        filter_position = rmse(filtered.means[1:], 0)
        smoother_position = rmse(smoothed.means[1:], 0)
        self.assertLess(filter_position, 0.7 * raw)
        self.assertLess(smoother_position, filter_position)
        self.assertLess(rmse(filtered.means[1:], 1), 0.5)
        inside = sum(
            abs(state.mean[0] - t[0]) <= 2.0 * state.standard_deviations[0]
            for state, t in zip(filtered.filtered[1:], truth, strict=True)
        )
        self.assertGreater(inside / len(truth), 0.85)

    def test_missing_observation_skips_the_update(self) -> None:
        """A missing observation leaves the prediction and likelihood untouched."""
        model = _random_walk()
        result = kalman_filter(model, [(1.0,), None, (0.5,)])
        self.assertEqual(result.filtered[2], result.predicted[1])
        self.assertEqual(result.log_likelihood_terms[1], 0.0)
        self.assertIsNone(result.gains[1])
        self.assertGreater(
            result.predicted[2].covariance[0][0], result.predicted[1].covariance[0][0]
        )

    def test_controls_shift_the_prediction(self) -> None:
        """A control input enters the predicted mean through ``B u``."""
        model = LinearGaussianStateSpace(
            ((1.0,),),
            ((0.1,),),
            ((1.0,),),
            ((1.0,),),
            (0.0,),
            ((1.0,),),
            control_matrix=((2.0,),),
        )
        state = GaussianState((1.0,), ((1.0,),))
        predicted = kalman_predict(model, state, (0.5,))
        self.assertEqual(predicted.mean, (2.0,))
        self.assertAlmostEqual(predicted.covariance[0][0], 1.1)
        update = kalman_update(model, predicted, (2.5,))
        self.assertAlmostEqual(update.gain[0][0], 1.1 / 2.1)

    def test_invalid_models_and_observations_raise(self) -> None:
        """Misshapen or indefinite inputs fail explicitly."""
        with self.assertRaises(ValueError):
            LinearGaussianStateSpace(
                ((1.0,),), ((0.1,),), ((1.0,),), ((0.0,),), (0.0,), ((1.0,),)
            )
        with self.assertRaises(ValueError):
            LinearGaussianStateSpace(
                ((1.0,),), ((-0.1,),), ((1.0,),), ((1.0,),), (0.0,), ((1.0,),)
            )
        with self.assertRaises(ValueError):
            kalman_filter(_random_walk(), [(1.0, 2.0)])
        with self.assertRaises(ValueError):
            kalman_filter(_random_walk(), [])
        with self.assertRaises(ValueError):
            _random_walk().as_nonlinear().transition((0.0,), (1.0,))


class NonlinearFilterTests(unittest.TestCase):
    """Extended and unscented filters."""

    def test_extended_filter_equals_kalman_filter_on_linear_model(self) -> None:
        """Exact Jacobians make the EKF identical to the KF."""
        model = _constant_velocity()
        data = simulate(model, 20, seed=3).observations
        exact = kalman_filter(model, data)
        extended = extended_kalman_filter(model.as_nonlinear(), data)
        for a, b in zip(exact.filtered, extended.filtered, strict=True):
            for i in range(2):
                self.assertAlmostEqual(a.mean[i], b.mean[i], 12)
                self.assertAlmostEqual(a.covariance[i][i], b.covariance[i][i], 12)
        self.assertAlmostEqual(exact.log_likelihood, extended.log_likelihood, 10)

    def test_unscented_filter_and_smoother_equal_kalman_on_linear_model(self) -> None:
        """Sigma points propagate linear maps exactly."""
        model = _constant_velocity()
        data = simulate(model, 15, seed=4).observations
        exact = kalman_filter(model, data)
        unscented = unscented_kalman_filter(model.as_nonlinear(), data)
        self.assertAlmostEqual(exact.log_likelihood, unscented.log_likelihood, 9)
        smoothed_exact = rts_smoother(exact)
        smoothed_unscented = rts_smoother(unscented)
        for a, b in zip(
            smoothed_exact.smoothed, smoothed_unscented.smoothed, strict=True
        ):
            for i in range(2):
                self.assertAlmostEqual(a.mean[i], b.mean[i], 9)
                self.assertAlmostEqual(a.covariance[i][i], b.covariance[i][i], 9)

    def test_unscented_transform_is_exact_for_a_quadratic_mean(self) -> None:
        """``E[X^2] = m^2 + P`` is reproduced exactly by sigma points."""
        result = unscented_transform(lambda x: (x[0] ** 2,), (1.5,), ((0.4,),))
        self.assertAlmostEqual(result.mean[0], 1.5**2 + 0.4, 12)

    def test_numerical_jacobians_match_analytic_jacobians(self) -> None:
        """Central-difference EKF agrees with the analytic-Jacobian EKF."""

        def transition(x):
            """Pendulum-like dynamics."""
            return (x[0] + 0.1 * x[1], x[1] - 0.1 * math.sin(x[0]))

        def observation(x):
            """Horizontal bob position."""
            return (math.sin(x[0]),)

        common = (
            transition,
            observation,
            ((1e-3, 0.0), (0.0, 1e-3)),
            ((0.05,),),
            (0.5, 0.0),
            ((0.1, 0.0), (0.0, 0.1)),
        )
        analytic = NonlinearStateSpace(
            *common,
            state_dimension=2,
            observation_dimension=1,
            transition_jacobian_function=lambda x: (
                (1.0, 0.1),
                (-0.1 * math.cos(x[0]), 1.0),
            ),
            observation_jacobian_function=lambda x: ((math.cos(x[0]), 0.0),),
        )
        numeric = NonlinearStateSpace(
            *common, state_dimension=2, observation_dimension=1
        )
        data = simulate(analytic, 25, seed=8).observations
        a = extended_kalman_filter(analytic, data)
        b = extended_kalman_filter(numeric, data)
        for left, right in zip(a.filtered, b.filtered, strict=True):
            self.assertAlmostEqual(left.mean[0], right.mean[0], 6)
            self.assertAlmostEqual(left.mean[1], right.mean[1], 6)
        u = unscented_kalman_filter(analytic, data)
        self.assertLess(abs(u.filtered[-1].mean[0] - a.filtered[-1].mean[0]), 0.1)


class ParticleFilterTests(unittest.TestCase):
    """Bootstrap particle filtering and resampling."""

    def test_systematic_resampling_and_effective_sample_size(self) -> None:
        """Each particle receives floor or ceil of ``N w`` copies."""
        weights = (0.1, 0.2, 0.3, 0.4)
        for offset in (0.0, 0.3, 0.99):
            ancestors = systematic_resample(weights, offset)
            counts = [ancestors.count(i) for i in range(4)]
            self.assertEqual(sum(counts), 4)
            for count, weight in zip(counts, weights, strict=True):
                self.assertIn(count, (math.floor(4 * weight), math.ceil(4 * weight)))
        self.assertAlmostEqual(effective_sample_size(weights), 1.0 / 0.3)
        self.assertAlmostEqual(effective_sample_size((1.0, 1.0, 1.0)), 3.0)
        with self.assertRaises(ValueError):
            systematic_resample(weights, 1.0)

    def test_particle_log_likelihood_approaches_kalman_value(self) -> None:
        """With many particles the likelihood estimate matches the exact one."""
        model = _random_walk()
        data = simulate(model, 10, seed=1).observations
        exact = kalman_filter(model, data)
        particle = bootstrap_particle_filter(model, data, particles=2000, seed=3)
        self.assertLess(abs(particle.log_likelihood - exact.log_likelihood), 0.25)
        self.assertLess(
            abs(particle.means[-1][0] - exact.filtered[-1].mean[0]),
            3.0 * math.sqrt(exact.filtered[-1].covariance[0][0] / 200.0),
        )
        self.assertTrue(any(particle.resampled))
        self.assertTrue(
            all(1.0 <= ess <= 2000.0 for ess in particle.effective_sample_sizes)
        )
        repeat = bootstrap_particle_filter(model, data, particles=2000, seed=3)
        self.assertEqual(repeat.log_likelihood, particle.log_likelihood)

    def test_particle_filter_rejects_invalid_arguments(self) -> None:
        """Particle count and resampling threshold are validated."""
        with self.assertRaises(ValueError):
            bootstrap_particle_filter(_random_walk(), [(0.0,)], particles=1)
        with self.assertRaises(ValueError):
            bootstrap_particle_filter(
                _random_walk(), [(0.0,)], particles=10, resample_threshold=1.5
            )


class DataAssimilationTests(unittest.TestCase):
    """Ensemble Kalman filtering and 3D-Var."""

    def test_enkf_converges_to_kalman_filter_with_large_ensemble(self) -> None:
        """Large-ensemble EnKF moments approach the exact Kalman moments."""
        model = _constant_velocity()
        data = simulate(model, 5, seed=7).observations
        exact = kalman_filter(model, data).filtered[-1]
        ensemble = ensemble_kalman_filter(model, data, ensemble_size=1000, seed=0)
        mean = ensemble.analysis_means[-1]
        covariance = ensemble.analysis_covariances[-1]
        for i in range(2):
            self.assertLess(abs(mean[i] - exact.mean[i]), 0.1)
            for j in range(2):
                self.assertLess(
                    abs(covariance[i][j] / exact.covariance[i][j] - 1.0), 0.15
                )
        self.assertEqual(ensemble.ensemble_size, 1000)

    def test_inflation_scales_covariance_and_keeps_mean(self) -> None:
        """Multiplicative inflation multiplies the sample covariance exactly."""
        ensemble = Ensemble(((0.0, 1.0), (1.0, 3.0), (2.0, 2.0), (5.0, 0.0)))
        inflated = inflate(ensemble, 1.44)
        for a, b in zip(ensemble.mean, inflated.mean, strict=True):
            self.assertAlmostEqual(a, b, 12)
        for row_a, row_b in zip(ensemble.covariance, inflated.covariance, strict=True):
            for a, b in zip(row_a, row_b, strict=True):
                self.assertAlmostEqual(1.44 * a, b, 12)
        with self.assertRaises(ValueError):
            inflate(ensemble, 0.9)

    def test_perturbed_observation_analysis_matches_kalman_in_expectation(self) -> None:
        """One analysis step reproduces the Kalman mean within sampling error."""
        model = _random_walk(0.5, 2.0)
        prior = GaussianState((1.0,), ((3.0,),))
        exact = kalman_update(model, prior, (4.0,))
        members = tuple((1.0 + math.sqrt(3.0) * z,) for z in _normal_scores(2000))
        analysis = ensemble_analysis(
            Ensemble(members), (4.0,), model.observe, ((2.0,),), seed=9
        )
        self.assertLess(abs(analysis.ensemble.mean[0] - exact.state.mean[0]), 0.08)
        self.assertLess(
            abs(analysis.ensemble.covariance[0][0] / exact.state.covariance[0][0] - 1),
            0.1,
        )

    def test_three_dimensional_variational_matches_kalman_update(self) -> None:
        """For a linear observation operator 3D-Var gives the BLUE analysis."""
        model = _constant_velocity()
        data = simulate(model, 1, seed=7).observations
        predicted = kalman_filter(model, data).predicted[0]
        update = kalman_update(model, predicted, data[0])
        analysis = three_dimensional_variational(
            predicted.mean,
            predicted.covariance,
            data[0],
            model.observe,
            model.observation_covariance,
        )
        self.assertTrue(analysis.converged)
        for i in range(2):
            self.assertAlmostEqual(analysis.analysis[i], update.state.mean[i], 7)
            for j in range(2):
                self.assertAlmostEqual(
                    analysis.analysis_covariance[i][j], update.state.covariance[i][j], 7
                )

    def test_three_dimensional_variational_handles_nonlinear_operator(self) -> None:
        """A nonlinear operator yields a stationary point below the background cost."""
        analysis = three_dimensional_variational(
            (1.0,), ((0.5,),), (2.0,), lambda x: (x[0] ** 2,), ((0.1,),)
        )
        self.assertTrue(analysis.converged)
        x = analysis.analysis[0]
        gradient = (x - 1.0) / 0.5 - 2.0 * x * (2.0 - x * x) / 0.1
        self.assertLess(abs(gradient), 1e-6)
        self.assertLess(analysis.cost, 0.5 * (2.0 - 1.0) ** 2 / 0.1)


def _normal_scores(count: int) -> tuple[float, ...]:
    """Return deterministic standard-normal quantiles at midpoint ranks."""
    from statistics import NormalDist

    normal = NormalDist()
    return tuple(normal.inv_cdf((k + 0.5) / count) for k in range(count))


if __name__ == "__main__":
    unittest.main()
