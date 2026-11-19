"""Tests for the auxiliary and optimal-proposal particle filters and FFBS.

The Kalman filter and the RTS smoother are the exact references on
linear-Gaussian models. Seeded Monte Carlo checks compare an average over
independent replications with the exact value in units of its standard
error; nothing pins the digits of a seeded run.
"""

from __future__ import annotations

import math
import statistics
import unittest

from probstat.filtering import (
    LinearGaussianStateSpace,
    NonlinearStateSpace,
    auxiliary_particle_filter,
    bootstrap_particle_filter,
    ffbs_particle_smoother,
    kalman_filter,
    optimal_proposal_particle_filter,
    rts_smoother,
    simulate,
)

SIGMAS = 4.0


def _ar1(
    phi: float = 0.9, q: float = 0.3, r: float = 2.0, p0: float = 1.0
) -> LinearGaussianStateSpace:
    """Return a scalar AR(1) state observed with additive noise."""
    return LinearGaussianStateSpace(
        ((phi,),), ((q,),), ((1.0,),), ((r,),), (0.0,), ((p0,),)
    )


def _ratio_band(estimates: list[float], exact: float) -> tuple[float, float]:
    """Return the mean of ``exp(estimate - exact)`` and its standard error."""
    ratios = [math.exp(value - exact) for value in estimates]
    return statistics.fmean(ratios), statistics.stdev(ratios) / math.sqrt(len(ratios))


class HistoryFlagTests(unittest.TestCase):
    """The opt-in particle history does not change the bootstrap filter."""

    def test_history_is_opt_in_and_leaves_every_number_unchanged(self) -> None:
        """History is opt in and leaves every number unchanged."""
        model = _ar1()
        data = simulate(model, 6, seed=1).observations
        plain = bootstrap_particle_filter(model, data, particles=50, seed=3)
        stored = bootstrap_particle_filter(
            model, data, particles=50, seed=3, store_history=True
        )
        self.assertIsNone(plain.particle_history)
        self.assertIsNone(plain.weight_history)
        self.assertEqual(plain.log_likelihood_terms, stored.log_likelihood_terms)
        self.assertEqual(plain.means, stored.means)
        self.assertEqual(plain.particles, stored.particles)
        assert stored.particle_history is not None
        assert stored.weight_history is not None
        self.assertEqual(len(stored.particle_history), len(data) + 1)
        self.assertEqual(len(stored.weight_history), len(data) + 1)
        for k in (0, 3, 6):
            cloud = stored.particle_history[k]
            weights = stored.weight_history[k]
            self.assertAlmostEqual(math.fsum(weights), 1.0, places=12)
            mean = math.fsum(w * x[0] for w, x in zip(weights, cloud, strict=True))
            self.assertAlmostEqual(mean, stored.means[k][0], places=12)


class ExactCaseTests(unittest.TestCase):
    """Degenerate noise makes the particle estimates deterministic and exact."""

    def test_noise_free_dynamics_reproduce_the_kalman_likelihood(self) -> None:
        """Noise free dynamics reproduce the kalman likelihood."""
        # With Q = 0 and P_0 = 0 every particle follows the same deterministic
        # path, so the first-stage and second-stage weights, the predictive
        # density of the optimal proposal, and the bootstrap weights are all
        # exactly N(y_k; H F^k m_0, R).
        model = LinearGaussianStateSpace(
            ((1.0, 0.5), (0.0, 0.9)),
            ((0.0, 0.0), (0.0, 0.0)),
            ((1.0, -0.4),),
            ((0.7,),),
            (0.3, 1.0),
            ((0.0, 0.0), (0.0, 0.0)),
        )
        data = [(0.2,), (1.4,), None, (0.9,)]
        exact = kalman_filter(model, data).log_likelihood
        for run in (
            auxiliary_particle_filter(model, data, particles=5, seed=1),
            optimal_proposal_particle_filter(model, data, particles=5, seed=1),
            bootstrap_particle_filter(model, data, particles=5, seed=1),
        ):
            self.assertTrue(math.isclose(run.log_likelihood, exact, rel_tol=1e-12))

    def test_optimal_proposal_is_exact_for_one_step_from_a_point_prior(self) -> None:
        """Optimal proposal is exact for one step from a point prior."""
        # From x_0 = m_0 the optimal-proposal increment is p(y_1 | m_0), which
        # is the exact Kalman likelihood of the first observation.
        model = _ar1(p0=0.0)
        run = optimal_proposal_particle_filter(model, [(1.3,)], particles=4, seed=9)
        exact = kalman_filter(model, [(1.3,)]).log_likelihood
        self.assertTrue(math.isclose(run.log_likelihood, exact, rel_tol=1e-12))
        # Every particle is drawn from N(m_0 + K (y - H m_0), Sigma).
        gain = 0.3 / (0.3 + 2.0)
        mean = gain * 1.3
        variance = (1.0 - gain) * 0.3
        self.assertLess(abs(run.means[1][0] - mean), 8.0 * math.sqrt(variance))


class UnbiasednessTests(unittest.TestCase):
    """``E[p_hat(y_1:T)] = p(y_1:T)`` for the new estimators."""

    @classmethod
    def setUpClass(cls) -> None:
        """Simulate one data set and its exact Kalman log-likelihood."""
        cls.model = _ar1(q=0.3, r=2.0)
        cls.data = simulate(cls.model, 6, seed=4).observations
        cls.exact = kalman_filter(cls.model, cls.data).log_likelihood

    def test_auxiliary_filter_likelihood_is_unbiased(self) -> None:
        """Auxiliary filter likelihood is unbiased."""
        estimates = [
            auxiliary_particle_filter(
                self.model, self.data, particles=40, seed=seed
            ).log_likelihood
            for seed in range(200)
        ]
        mean, error = _ratio_band(estimates, self.exact)
        self.assertLess(abs(mean - 1.0), SIGMAS * error)
        self.assertLess(error, 0.05)

    def test_optimal_proposal_likelihood_is_unbiased(self) -> None:
        """Optimal proposal likelihood is unbiased."""
        estimates = [
            optimal_proposal_particle_filter(
                self.model, self.data, particles=40, seed=seed
            ).log_likelihood
            for seed in range(200)
        ]
        mean, error = _ratio_band(estimates, self.exact)
        self.assertLess(abs(mean - 1.0), SIGMAS * error)
        self.assertLess(error, 0.05)

    def test_optimal_proposal_beats_the_bootstrap_on_informative_data(self) -> None:
        """Optimal proposal beats the bootstrap on informative data."""
        # Precise observations (r = 0.5 against q = 1) make the transition
        # prior a poor proposal; the optimal proposal's weights do not depend
        # on the new particle.
        model = _ar1(q=1.0, r=0.5)
        data = simulate(model, 10, seed=4).observations
        exact = kalman_filter(model, data).log_likelihood
        optimal = [
            optimal_proposal_particle_filter(
                model, data, particles=30, seed=seed
            ).log_likelihood
            for seed in range(60)
        ]
        bootstrap = [
            bootstrap_particle_filter(
                model, data, particles=30, seed=seed
            ).log_likelihood
            for seed in range(60)
        ]
        self.assertLess(
            statistics.variance(optimal), 0.5 * statistics.variance(bootstrap)
        )
        mean, error = _ratio_band(optimal, exact)
        self.assertLess(abs(mean - 1.0), SIGMAS * error)


class OptimalProposalInterfaceTests(unittest.TestCase):
    """The closed-form proposal needs a linear observation operator."""

    def test_nonlinear_model_needs_an_observation_matrix(self) -> None:
        """Nonlinear model needs an observation matrix."""
        linear = _ar1()
        nonlinear = linear.as_nonlinear()
        with self.assertRaisesRegex(ValueError, "linear observation operator"):
            optimal_proposal_particle_filter(nonlinear, [(1.0,)], particles=4)

    def test_a_nonlinear_observation_map_is_rejected(self) -> None:
        """A nonlinear observation map is rejected."""
        squared = NonlinearStateSpace(
            lambda x: (0.9 * x[0],),
            lambda x: (x[0] ** 2,),
            ((0.3,),),
            ((2.0,),),
            (0.5,),
            ((1.0,),),
            state_dimension=1,
            observation_dimension=1,
        )
        with self.assertRaisesRegex(ValueError, "not the linear map"):
            optimal_proposal_particle_filter(
                squared, [(1.0,)], particles=4, observation_matrix=((1.0,),)
            )

    def test_nonlinear_dynamics_with_a_linear_sensor_match_the_linear_model(
        self,
    ) -> None:
        """Nonlinear dynamics with a linear sensor match the linear model."""
        linear = _ar1()
        data = simulate(linear, 5, seed=2).observations
        reference = optimal_proposal_particle_filter(linear, data, particles=20, seed=6)
        run = optimal_proposal_particle_filter(
            linear.as_nonlinear(),
            data,
            particles=20,
            seed=6,
            observation_matrix=((1.0,),),
        )
        for a, b in zip(
            reference.log_likelihood_terms, run.log_likelihood_terms, strict=True
        ):
            self.assertTrue(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))

    def test_invalid_arguments_raise(self) -> None:
        """Invalid arguments raise."""
        model = _ar1()
        with self.assertRaises(ValueError):
            auxiliary_particle_filter(model, [(1.0,)], particles=1)
        with self.assertRaises(ValueError):
            auxiliary_particle_filter(model, [], particles=4)
        with self.assertRaises(ValueError):
            optimal_proposal_particle_filter(
                model, [(1.0,)], particles=4, resample_threshold=1.5
            )
        with self.assertRaises(ValueError):
            optimal_proposal_particle_filter(
                model, [(1.0,)], particles=4, observation_matrix=((1.0, 0.0),)
            )

    def test_missing_observations_contribute_nothing(self) -> None:
        """Missing observations contribute nothing."""
        model = _ar1()
        data = [(0.4,), None, (1.0,)]
        for run in (
            auxiliary_particle_filter(model, data, particles=10, seed=2),
            optimal_proposal_particle_filter(model, data, particles=10, seed=2),
        ):
            self.assertEqual(run.log_likelihood_terms[1], 0.0)
            self.assertEqual(len(run.means), 4)
        apf = auxiliary_particle_filter(model, data, particles=10, seed=2)
        self.assertEqual(apf.resampled, (True, False, True))


class ParticleSmootherTests(unittest.TestCase):
    """FFBS trajectories approach the RTS smoothing marginals."""

    def test_ffbs_moments_approach_the_rts_smoother(self) -> None:
        """Ffbs moments approach the rts smoother."""
        model = _ar1(phi=0.9, q=0.5, r=1.0)
        data = simulate(model, 6, seed=8).observations
        exact = rts_smoother(kalman_filter(model, data)).smoothed
        replications = 30
        trajectories = 20
        means: list[list[float]] = [[] for _ in range(len(data) + 1)]
        variances: list[list[float]] = [[] for _ in range(len(data) + 1)]
        for seed in range(replications):
            forward = bootstrap_particle_filter(
                model, data, particles=200, seed=seed, store_history=True
            )
            smooth = ffbs_particle_smoother(
                model, forward, trajectories=trajectories, seed=1000 + seed
            )
            self.assertEqual(len(smooth.trajectories), trajectories)
            self.assertEqual(len(smooth.trajectories[0]), len(data) + 1)
            for k in range(len(data) + 1):
                means[k].append(smooth.means[k][0])
                # Undo the plug-in factor (M - 1) / M of the trajectory variance.
                variances[k].append(
                    smooth.variances[k][0] * trajectories / (trajectories - 1)
                )
        for k, state in enumerate(exact):
            with self.subTest(k=k):
                mean_error = statistics.stdev(means[k]) / math.sqrt(replications)
                self.assertLess(
                    abs(statistics.fmean(means[k]) - state.mean[0]),
                    SIGMAS * mean_error,
                )
                variance_error = statistics.stdev(variances[k]) / math.sqrt(
                    replications
                )
                self.assertLess(
                    abs(statistics.fmean(variances[k]) - state.covariance[0][0]),
                    SIGMAS * variance_error,
                )

    def test_ffbs_runs_on_the_auxiliary_filter(self) -> None:
        """Ffbs runs on the auxiliary filter."""
        model = _ar1()
        data = simulate(model, 4, seed=3).observations
        forward = auxiliary_particle_filter(
            model, data, particles=30, seed=1, store_history=True
        )
        smooth = ffbs_particle_smoother(model, forward, trajectories=5, seed=2)
        self.assertEqual(smooth.trajectory_count, 5)
        self.assertEqual(len(smooth.means), 5)
        # Every trajectory point is a particle of the matching stored cloud.
        assert forward.particle_history is not None
        for path in smooth.trajectories:
            for k, point in enumerate(path):
                self.assertIn(point, forward.particle_history[k])

    def test_invalid_inputs_raise(self) -> None:
        """Invalid inputs raise."""
        model = _ar1()
        data = simulate(model, 3, seed=3).observations
        without = bootstrap_particle_filter(model, data, particles=10, seed=1)
        with self.assertRaisesRegex(ValueError, "store_history=True"):
            ffbs_particle_smoother(model, without)
        stored = bootstrap_particle_filter(
            model, data, particles=10, seed=1, store_history=True
        )
        with self.assertRaises(ValueError):
            ffbs_particle_smoother(model, stored, trajectories=1)
        singular = LinearGaussianStateSpace(
            ((1.0, 1.0), (0.0, 1.0)),
            ((0.0, 0.0), (0.0, 0.1)),
            ((1.0, 0.0),),
            ((1.0,),),
            (0.0, 0.0),
            ((1.0, 0.0), (0.0, 1.0)),
        )
        run = bootstrap_particle_filter(
            singular, [(0.1,), (0.2,)], particles=10, seed=1, store_history=True
        )
        with self.assertRaisesRegex(ValueError, "positive-definite"):
            ffbs_particle_smoother(singular, run)


if __name__ == "__main__":
    unittest.main()
