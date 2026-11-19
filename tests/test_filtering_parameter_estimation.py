"""Tests for state-space parameter estimation: MLE, EM, and PMMH.

References are exact wherever possible: the Kalman log-likelihood, batch
Gaussian conditioning, R's ``StructTS`` fit (hard-coded values from an offline
run), and the posterior computed by quadrature of the exact likelihood.
Seeded Monte Carlo results are judged in units of their Monte Carlo standard
error, never by their digits.
"""

from __future__ import annotations

import math
import statistics
import unittest

from probstat.filtering import (
    LinearGaussianStateSpace,
    fit_state_space_mle,
    kalman_filter,
    linear_gaussian_em,
    particle_marginal_metropolis_hastings,
    rts_smoother,
    simulate,
    smoothed_lag_one_covariances,
)
from probstat.inference import condition_linear_gaussian, monte_carlo_standard_error
from probstat.multivariate import MultivariateNormal
from probstat.numerics import composite_simpson

SIGMAS = 4.0

# Annual Nile flow at Aswan, 1871-1970 (R dataset ``Nile``).
NILE = (
    1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140, 995, 935, 1110,
    994, 1020, 960, 1180, 799, 958, 1140, 1100, 1210, 1150, 1250, 1260, 1220,
    1030, 1100, 774, 840, 874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969,
    831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821, 768, 845, 864, 862,
    698, 845, 744, 796, 1040, 759, 781, 865, 845, 944, 984, 897, 822, 1010,
    771, 676, 649, 846, 812, 742, 801, 1040, 860, 874, 848, 890, 744, 749,
    838, 1050, 918, 986, 797, 923, 975, 815, 1020, 906, 901, 1170, 912, 746,
    919, 718, 714, 740,
)  # fmt: skip

# R 4.6.2, run offline (tests never call R):
#   f <- StructTS(Nile, type = "level")
#   options(digits = 17); f$coef; f$loglik; f$code
# gives level = 1469.1466192376836, epsilon = 15098.5771535974000,
# loglik = -643.20098751576131, code = 0. StructTS uses x_0 ~ N(y_1, P_0)
# with P_0 = 1e6 * var(y) / 100 and predicts before the first update
# (KalmanLike with nit = -1), so the first observation enters with the
# diffuse variance P_0 + q + r; its loglik is the full Gaussian
# log-likelihood of all 100 observations. Those are exactly the conventions
# of kalman_filter below, so the log-likelihood at R's estimate is matched to
# rounding. The estimates themselves are matched only to the optimizer
# tolerance: R stops L-BFGS-B (factr = 1e7) on a scaled problem, and the
# likelihood is flat in q.
R_LEVEL = 1469.1466192376836
R_EPSILON = 15098.5771535974
R_LOGLIK = -643.20098751576131


def _nile_model(q: float, r: float) -> LinearGaussianStateSpace:
    """Return the StructTS local-level model of the Nile series."""
    variance = statistics.variance([float(value) for value in NILE])
    return LinearGaussianStateSpace(
        ((1.0,),),
        ((q,),),
        ((1.0,),),
        ((r,),),
        (float(NILE[0]),),
        ((1e6 * variance / 100.0,),),
    )


def _ar1(phi: float, q: float, r: float) -> LinearGaussianStateSpace:
    """Return a scalar AR(1) state observed with noise, ``x_0 ~ N(0, 1)``."""
    return LinearGaussianStateSpace(
        ((phi,),), ((q,),), ((1.0,),), ((r,),), (0.0,), ((1.0,),)
    )


class MaximumLikelihoodTests(unittest.TestCase):
    """``fit_state_space_mle`` against R and against the EKF route."""

    def test_kalman_log_likelihood_matches_structts(self) -> None:
        """The filter likelihood at R's estimate equals R's loglik."""
        data = [(float(value),) for value in NILE]
        value = kalman_filter(_nile_model(R_LEVEL, R_EPSILON), data).log_likelihood
        self.assertTrue(math.isclose(value, R_LOGLIK, rel_tol=1e-11))

    def test_nile_fit_matches_structts(self) -> None:
        """The MLE of the Nile local-level model matches StructTS."""
        data = [(float(value),) for value in NILE]
        start = math.log(statistics.variance([float(v) for v in NILE]) / 100.0)

        def build(theta: tuple[float, ...]) -> LinearGaussianStateSpace:
            return _nile_model(math.exp(theta[0]), math.exp(theta[1]))

        fit = fit_state_space_mle(
            build, (start, start), data, names=("log_level", "log_epsilon")
        )
        self.assertTrue(fit.converged)
        self.assertEqual(fit.message, "ok")
        self.assertEqual(fit.names, ("log_level", "log_epsilon"))
        level, epsilon = (math.exp(value) for value in fit.parameters)
        self.assertTrue(math.isclose(level, R_LEVEL, rel_tol=2e-3))
        self.assertTrue(math.isclose(epsilon, R_EPSILON, rel_tol=2e-4))
        # A maximizer cannot be worse than R's point by more than rounding.
        self.assertGreater(fit.log_likelihood, R_LOGLIK - 1e-8)
        self.assertLess(fit.log_likelihood, R_LOGLIK + 1e-6)
        assert fit.covariance is not None and fit.standard_errors is not None
        self.assertEqual(len(fit.covariance), 2)
        # The level variance is poorly determined (Durbin and Koopman).
        self.assertGreater(fit.standard_errors[0], fit.standard_errors[1])

    def test_extended_filter_route_equals_the_kalman_route(self) -> None:
        """On a linear model the EKF likelihood gives the same estimate."""
        truth = _ar1(0.8, 1.0, 0.5)
        data = simulate(truth, 30, seed=12).observations

        def linear(theta: tuple[float, ...]) -> LinearGaussianStateSpace:
            return _ar1(theta[0], math.exp(theta[1]), 0.5)

        exact = fit_state_space_mle(linear, (0.5, 0.0), data)
        extended = fit_state_space_mle(
            lambda theta: linear(theta).as_nonlinear(), (0.5, 0.0), data, filter="ekf"
        )
        for a, b in zip(exact.parameters, extended.parameters, strict=True):
            self.assertTrue(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6))
        self.assertEqual(extended.filter, "ekf")

    def test_bounds_keep_variances_positive(self) -> None:
        """Box bounds on the variances give an interior optimum."""
        truth = _ar1(1.0, 0.5, 2.0)
        data = simulate(truth, 40, seed=3).observations
        fit = fit_state_space_mle(
            lambda theta: _ar1(1.0, theta[0], theta[1]),
            (1.0, 1.0),
            data,
            bounds=((0.0, None), (0.0, None)),
        )
        self.assertTrue(fit.converged)
        self.assertTrue(all(value > 0.0 for value in fit.parameters))

    def test_invalid_arguments_raise(self) -> None:
        """Unknown filters, nonlinear models with the KF, and bad names raise."""
        data = [(1.0,), (0.5,)]
        with self.assertRaises(ValueError):
            fit_state_space_mle(
                lambda t: _ar1(t[0], 1.0, 1.0), (0.5,), data, filter="x"
            )
        with self.assertRaises(TypeError):
            fit_state_space_mle(
                lambda t: _ar1(t[0], 1.0, 1.0).as_nonlinear(), (0.5,), data
            )
        with self.assertRaises(ValueError):
            fit_state_space_mle(
                lambda t: _ar1(t[0], 1.0, 1.0), (0.5,), data, names=("a", "b")
            )


class ExpectationMaximizationTests(unittest.TestCase):
    """Shumway-Stoffer EM: monotone, consistent, and equal to the MLE."""

    def test_lag_one_covariances_match_batch_conditioning(self) -> None:
        """``P^s_k G_(k-1)'`` equals the joint Gaussian posterior covariance."""
        phi, q, r, p0 = 0.7, 0.4, 0.9, 1.5
        model = LinearGaussianStateSpace(
            ((phi,),), ((q,),), ((1.0,),), ((r,),), (0.2,), ((p0,),)
        )
        data = [(0.3,), (-0.4,), (1.1,)]
        variances = [p0]
        for _ in data:
            variances.append(phi * phi * variances[-1] + q)
        size = len(data) + 1
        covariance = tuple(
            tuple(phi ** abs(i - j) * variances[min(i, j)] for j in range(size))
            for i in range(size)
        )
        mean = tuple(0.2 * phi**k for k in range(size))
        design = tuple(
            tuple(1.0 if column == k else 0.0 for column in range(size))
            for k in range(1, size)
        )
        posterior = condition_linear_gaussian(
            MultivariateNormal(mean, covariance),
            design,
            tuple(value[0] for value in data),
            observation_variance=r,
        )
        lag_one = smoothed_lag_one_covariances(rts_smoother(kalman_filter(model, data)))
        self.assertEqual(len(lag_one), len(data))
        for k in range(1, size):
            self.assertAlmostEqual(
                lag_one[k - 1][0][0], posterior.covariance[k][k - 1], places=12
            )

    def test_em_agrees_with_the_mle_and_recovers_the_truth(self) -> None:
        """AR(1) plus noise: EM and BFGS reach the same maximum."""
        truth = _ar1(0.8, 1.0, 0.5)
        data = simulate(truth, 100, seed=2026).observations
        em = linear_gaussian_em(
            _ar1(0.5, 0.5, 1.0),
            data,
            estimate=("transition", "process_covariance", "observation_covariance"),
            iterations=1000,
            tolerance=1e-10,
        )
        self.assertTrue(em.converged)
        steps = [
            b - a
            for a, b in zip(
                em.log_likelihood_trace, em.log_likelihood_trace[1:], strict=False
            )
        ]
        self.assertGreaterEqual(min(steps), -1e-9 * abs(em.log_likelihood))
        fit = fit_state_space_mle(
            lambda theta: _ar1(theta[0], math.exp(theta[1]), math.exp(theta[2])),
            (0.5, math.log(0.5), 0.0),
            data,
        )
        self.assertTrue(fit.converged)
        self.assertLess(abs(em.log_likelihood - fit.log_likelihood), 1e-5)
        estimates = (
            em.model.transition_matrix[0][0],
            math.log(em.model.process_covariance[0][0]),
            math.log(em.model.observation_covariance[0][0]),
        )
        for a, b in zip(estimates, fit.parameters, strict=True):
            self.assertLess(abs(a - b), 2e-3)
        assert fit.standard_errors is not None
        for value, true, error in zip(
            fit.parameters,
            (0.8, 0.0, math.log(0.5)),
            fit.standard_errors,
            strict=True,
        ):
            self.assertLess(abs(value - true), SIGMAS * error)

    def test_all_blocks_never_decrease_the_likelihood(self) -> None:
        """A 2-D model with missing data and every block estimated."""
        truth = LinearGaussianStateSpace(
            ((0.9, 0.2), (-0.1, 0.7)),
            ((0.5, 0.1), (0.1, 0.3)),
            ((1.0, 0.0), (0.5, 1.0)),
            ((0.4, 0.0), (0.0, 0.6)),
            (0.0, 0.0),
            ((1.0, 0.0), (0.0, 1.0)),
        )
        data: list[tuple[float, ...] | None] = list(
            simulate(truth, 40, seed=5).observations
        )
        data[7] = None
        data[21] = None
        start = LinearGaussianStateSpace(
            ((0.5, 0.0), (0.0, 0.5)),
            ((1.0, 0.0), (0.0, 1.0)),
            ((1.0, 0.1), (0.1, 1.0)),
            ((1.0, 0.0), (0.0, 1.0)),
            (0.5, -0.5),
            ((2.0, 0.0), (0.0, 2.0)),
        )
        em = linear_gaussian_em(start, data, iterations=25, tolerance=1e-14)
        self.assertEqual(em.iterations, 25)
        self.assertFalse(em.converged)
        self.assertEqual(len(em.log_likelihood_trace), 26)
        trace = em.log_likelihood_trace
        for a, b in zip(trace, trace[1:], strict=False):
            self.assertGreaterEqual(b - a, -1e-9 * abs(b))
        self.assertGreater(trace[-1], trace[0] + 1.0)
        self.assertEqual(
            em.estimated,
            (
                "transition",
                "observation",
                "process_covariance",
                "observation_covariance",
                "initial",
            ),
        )
        self.assertAlmostEqual(
            em.log_likelihood, kalman_filter(em.model, data).log_likelihood, 12
        )

    def test_fixed_blocks_stay_fixed(self) -> None:
        """Blocks outside ``estimate`` are not touched."""
        model = _ar1(0.6, 1.0, 1.0)
        data = simulate(model, 20, seed=1).observations
        em = linear_gaussian_em(
            model, data, iterations=3, estimate=("observation_covariance",)
        )
        self.assertEqual(em.model.transition_matrix, model.transition_matrix)
        self.assertEqual(em.model.process_covariance, model.process_covariance)
        self.assertEqual(em.model.initial_mean, model.initial_mean)
        self.assertNotEqual(
            em.model.observation_covariance, model.observation_covariance
        )

    def test_invalid_arguments_raise(self) -> None:
        """Unknown blocks, controlled models, and bad tolerances raise."""
        model = _ar1(0.6, 1.0, 1.0)
        with self.assertRaises(ValueError):
            linear_gaussian_em(model, [(1.0,)], estimate=("gain",))
        with self.assertRaises(ValueError):
            linear_gaussian_em(model, [(1.0,)], tolerance=0.0)
        with self.assertRaises(TypeError):
            linear_gaussian_em(model.as_nonlinear(), [(1.0,)])  # type: ignore[arg-type]
        controlled = LinearGaussianStateSpace(
            ((1.0,),), ((1.0,),), ((1.0,),), ((1.0,),), (0.0,), ((1.0,),),
            control_matrix=((1.0,),),
        )  # fmt: skip
        with self.assertRaises(ValueError):
            linear_gaussian_em(controlled, [(1.0,)])
        with self.assertRaises(ValueError):
            linear_gaussian_em(model, [None], estimate=("observation_covariance",))


def _pmmh_problem() -> tuple[list[tuple[float, ...]], float, float]:
    """Return AR(1) data and the exact posterior mean and variance of ``phi``.

    The prior is uniform on ``(-1, 1)``; the posterior moments come from
    Simpson quadrature (400 intervals) of the exact Kalman likelihood.
    """
    data = [tuple(y) for y in simulate(_ar1(0.7, 1.0, 0.5), 12, seed=31).observations]
    peak = kalman_filter(_ar1(0.5, 1.0, 0.5), data).log_likelihood

    def density(phi: float) -> float:
        return math.exp(kalman_filter(_ar1(phi, 1.0, 0.5), data).log_likelihood - peak)

    grid = 400
    mass = composite_simpson(density, -1.0, 1.0, intervals=grid).value
    first = composite_simpson(
        lambda phi: phi * density(phi), -1.0, 1.0, intervals=grid
    ).value
    second = composite_simpson(
        lambda phi: phi * phi * density(phi), -1.0, 1.0, intervals=grid
    ).value
    mean = first / mass
    return data, mean, second / mass - mean * mean


class ParticleMarginalMetropolisHastingsTests(unittest.TestCase):
    """PMMH targets the exact posterior and reuses the current estimate."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run one seeded PMMH sampler shared by the tests."""
        cls.data, cls.mean, cls.variance = _pmmh_problem()

        def build(theta: tuple[float, ...]) -> LinearGaussianStateSpace:
            return _ar1(theta[0], 1.0, 0.5)

        def log_prior(theta: tuple[float, ...]) -> float:
            return 0.0 if -1.0 < theta[0] < 1.0 else -math.inf

        cls.result = particle_marginal_metropolis_hastings(
            build,
            log_prior,
            (0.5,),
            cls.data,
            particles=20,
            draws=1500,
            warmup=100,
            chains=2,
            proposal_covariance=((0.3,),),
            seed=8,
            particle_filter="optimal",
            check_convergence=False,
        )

    def test_posterior_mean_and_variance_match_the_exact_posterior(self) -> None:
        """Mean and variance agree with quadrature within k MCSE."""
        columns = self.result.chain_columns(0)
        mcse = monte_carlo_standard_error(columns)
        self.assertLess(abs(self.result.mean()[0] - self.mean), SIGMAS * mcse)
        squared = tuple(
            tuple((value - self.mean) ** 2 for value in chain) for chain in columns
        )
        second = statistics.fmean(v for chain in squared for v in chain)
        self.assertLess(
            abs(second - self.variance),
            SIGMAS * monte_carlo_standard_error(squared),
        )

    def test_current_estimate_is_stored_and_reused(self) -> None:
        """A rejected move keeps the stored estimate; no state is re-estimated."""
        result = self.result
        for chain in result.chains:
            for (a, la), (b, lb) in zip(
                zip(chain.samples, chain.log_density, strict=True),
                zip(chain.samples[1:], chain.log_density[1:], strict=True),
                strict=False,
            ):
                if a == b:
                    self.assertEqual(la, lb)
            # One filter run for the start plus at most one per iteration.
            self.assertLessEqual(chain.evaluations, 1 + 100 + 1500)
            self.assertGreater(chain.acceptance_rate, 0.1)
        self.assertEqual(result.algorithm, "particle marginal Metropolis-Hastings")
        self.assertEqual(result.draws_per_chain, 1500)

    def test_invalid_arguments_raise(self) -> None:
        """Bad filters, starts, and covariances raise."""

        def build(theta: tuple[float, ...]) -> LinearGaussianStateSpace:
            return _ar1(theta[0], 1.0, 0.5)

        def flat(theta: tuple[float, ...]) -> float:
            return 0.0 if -1.0 < theta[0] < 1.0 else -math.inf

        common = {"particles": 5, "draws": 2, "proposal_covariance": ((0.1,),)}
        with self.assertRaises(ValueError):
            particle_marginal_metropolis_hastings(
                build, flat, (0.5,), self.data, particle_filter="x", **common
            )
        with self.assertRaises(ValueError):
            particle_marginal_metropolis_hastings(
                build, flat, (1.5,), self.data, **common
            )
        with self.assertRaises(ValueError):
            particle_marginal_metropolis_hastings(
                build,
                flat,
                (0.5,),
                self.data,
                particles=5,
                draws=2,
                proposal_covariance=((-1.0,),),
            )
        with self.assertRaises(TypeError):
            particle_marginal_metropolis_hastings(
                build,
                None,  # type: ignore[arg-type]
                (0.5,),
                self.data,
                **common,
            )


if __name__ == "__main__":
    unittest.main()
