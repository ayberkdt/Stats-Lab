"""Tests for MCMC, Laplace, Normal-inverse-gamma, and nonlinear calibration."""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.foundations import RandomStream
from probstat.inference import (
    NormalInverseGamma,
    adaptive_metropolis,
    condition_linear_gaussian,
    condition_linear_normal_inverse_gamma,
    effective_sample_size,
    gaussian_log_likelihood,
    laplace_approximation,
    monte_carlo_standard_error,
    random_walk_metropolis,
    rank_normalized_rhat,
    split_rhat,
    student_t_cdf,
    student_t_quantile,
    summarize_draws,
)
from probstat.inverse import (
    CalibrationProblem,
    Experiment,
    NoiseModel,
    fisher_identifiability,
    predict,
)
from probstat.linalg import inverse_spd
from probstat.modeling import (
    JointInput,
    ProbabilisticModel,
    RandomInput,
    standard_normal_quantile,
)
from probstat.multivariate import MultivariateNormal
from probstat.uq import QuantityOfInterest

TARGET_MEAN = (1.0, -2.0)
TARGET_COVARIANCE = ((1.0, 0.8), (0.8, 2.0))
TARGET_PRECISION = inverse_spd(TARGET_COVARIANCE)
DECAY_TIMES = tuple(0.5 * i for i in range(21))
TRUE_AMPLITUDE = 5.0
TRUE_RATE = 0.3
TRUE_SIGMA = 0.1


def correlated_gaussian(x: tuple[float, ...]) -> float:
    """Return the unnormalized log-density of a correlated 2-D Gaussian."""
    d0, d1 = x[0] - TARGET_MEAN[0], x[1] - TARGET_MEAN[1]
    p = TARGET_PRECISION
    return -0.5 * (
        d0 * (p[0][0] * d0 + p[0][1] * d1) + d1 * (p[1][0] * d0 + p[1][1] * d1)
    )


def bimodal(x: tuple[float, ...]) -> float:
    """Return a well-separated two-component normal mixture log-density."""
    a = -0.5 * (x[0] - 5.0) ** 2
    b = -0.5 * (x[0] + 5.0) ** 2
    top = max(a, b)
    return top + math.log(math.exp(a - top) + math.exp(b - top))


def ar1_chains(phi: float, n: int, m: int, seed: int) -> list[list[float]]:
    """Simulate stationary unit-variance AR(1) chains."""
    chains = []
    for stream in RandomStream(seed).spawn(m):
        generator = stream.generator()
        x = [generator.gauss(0.0, 1.0)]
        for _ in range(n - 1):
            x.append(phi * x[-1] + generator.gauss(0.0, math.sqrt(1.0 - phi**2)))
        chains.append(x)
    return chains


def line_model(parameters):
    """Return a straight-line response."""
    return {"y": parameters["a"] + parameters["b"] * parameters["x"]}


def decay_model(parameters):
    """Return an exponential-decay signal."""
    return {"y": parameters["A"] * math.exp(-parameters["k"] * parameters["t"])}


def product_model(parameters):
    """Return a response that only identifies the product ``a b``."""
    return {"y": parameters["a"] * parameters["b"] * parameters["x"]}


def decay_problem(noise: NoiseModel) -> CalibrationProblem:
    """Build the seeded synthetic exponential-decay calibration problem."""
    prior = JointInput.independent(
        RandomInput("A", Uniform(1.0, 10.0)),
        RandomInput("k", LogNormal(math.log(0.5), 1.0)),
    )
    model = ProbabilisticModel(prior, decay_model, (QuantityOfInterest("y", "signal"),))
    generator = random.Random(7)
    experiments = tuple(
        Experiment(
            {
                "y": TRUE_AMPLITUDE * math.exp(-TRUE_RATE * t)
                + generator.gauss(0.0, TRUE_SIGMA)
            },
            {"t": t},
        )
        for t in DECAY_TIMES
    )
    return CalibrationProblem(model, experiments, (noise,))


def line_problem() -> tuple[CalibrationProblem, tuple[float, ...], list[float]]:
    """Build a linear-Gaussian calibration with independent Gaussian priors."""
    prior = JointInput.independent(
        RandomInput("a", Normal(0.0, 2.0)), RandomInput("b", Normal(1.0, 1.0))
    )
    model = ProbabilisticModel(prior, line_model, (QuantityOfInterest("y", "line"),))
    xs = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    generator = random.Random(5)
    ys = [1.0 + 0.7 * x + generator.gauss(0.0, 0.3) for x in xs]
    experiments = tuple(
        Experiment({"y": y}, {"x": x}) for x, y in zip(xs, ys, strict=False)
    )
    problem = CalibrationProblem(
        model, experiments, (NoiseModel("y", standard_deviation=0.3),)
    )
    return problem, xs, ys


class MarkovChainMonteCarloTests(unittest.TestCase):
    """Verify samplers and convergence diagnostics on known targets."""

    def test_adaptive_metropolis_recovers_correlated_gaussian(self) -> None:
        """Mean is within MC error and covariance within 15 percent."""
        result = adaptive_metropolis(
            correlated_gaussian, (0.0, 0.0), draws=2000, warmup=600, chains=4, seed=7
        )
        summaries = result.summary()
        for index, summary in enumerate(summaries):
            with self.subTest(coordinate=index):
                self.assertLess(
                    abs(summary.mean - TARGET_MEAN[index]), 4.0 * summary.mcse_mean
                )
                self.assertLess(summary.split_rhat, 1.02)
                self.assertLess(summary.rank_normalized_rhat, 1.02)
                self.assertGreater(summary.effective_sample_size, 500.0)
        covariance = result.covariance()
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    covariance[i][j], TARGET_COVARIANCE[i][j], delta=0.15 * 2.0
                )
        self.assertTrue(0.15 < result.acceptance_rate < 0.6)
        self.assertEqual(result.evaluations, 4 * (1 + 600 + 2000))
        self.assertEqual(result.draws_per_chain, 2000)
        frozen = result.chains[0].proposal_covariance
        self.assertGreater(frozen[0][1], 0.0)

    def test_random_walk_is_reproducible_and_counts_work(self) -> None:
        """Same seed reproduces chains; thinning and counts are explicit."""
        kwargs = {
            "proposal_covariance": ((1.0, 0.0), (0.0, 1.0)),
            "draws": 200,
            "warmup": 50,
            "chains": 2,
            "thin": 2,
            "seed": RandomStream(3),
        }
        first = random_walk_metropolis(correlated_gaussian, (0.0, 0.0), **kwargs)
        second = random_walk_metropolis(correlated_gaussian, (0.0, 0.0), **kwargs)
        self.assertEqual(first.chains[0].samples, second.chains[0].samples)
        self.assertNotEqual(first.chains[0].samples, first.chains[1].samples)
        self.assertEqual(first.evaluations, 2 * (1 + 50 + 400))
        self.assertEqual(len(first.chains[0].log_density), 200)
        self.assertEqual(first.names, ("x0", "x1"))
        self.assertEqual(len(first.chain_columns("x1")), 2)

    def test_rhat_flags_non_mixing_chains(self) -> None:
        """Chains trapped in different modes give R-hat far above 1.1."""
        result = random_walk_metropolis(
            bimodal,
            ((-5.0,), (5.0,)),
            proposal_covariance=((0.25,),),
            draws=500,
            chains=2,
            seed=1,
        )
        columns = result.chain_columns(0)
        self.assertGreater(split_rhat(columns), 1.1)
        self.assertGreater(rank_normalized_rhat(columns), 1.1)

    def test_rank_normalized_rhat_detects_scale_differences(self) -> None:
        """Folded R-hat catches equal-mean chains with different spreads."""
        generator = random.Random(2)
        narrow = [generator.gauss(0.0, 1.0) for _ in range(1000)]
        wide = [generator.gauss(0.0, 3.0) for _ in range(1000)]
        self.assertLess(split_rhat([narrow, wide]), 1.05)
        self.assertGreater(rank_normalized_rhat([narrow, wide]), 1.1)

    def test_converged_iid_chains_have_rhat_near_one(self) -> None:
        """Independent draws from one law give R-hat near one."""
        generator = random.Random(4)
        chains = [[generator.gauss(0.0, 1.0) for _ in range(1000)] for _ in range(4)]
        self.assertLess(abs(split_rhat(chains) - 1.0), 0.01)
        self.assertLess(abs(rank_normalized_rhat(chains) - 1.0), 0.01)
        ess = effective_sample_size(chains)
        self.assertGreater(ess, 3000.0)

    def test_ess_matches_ar1_theory(self) -> None:
        """ESS of AR(1) chains matches ``n (1-phi)/(1+phi)`` within 15 percent."""
        phi = 0.5
        chains = ar1_chains(phi, 5000, 4, seed=0)
        theory = 4 * 5000 * (1.0 - phi) / (1.0 + phi)
        self.assertAlmostEqual(effective_sample_size(chains) / theory, 1.0, delta=0.15)
        single = ar1_chains(0.8, 20000, 1, seed=2)
        theory_single = 20000 * 0.2 / 1.8
        self.assertAlmostEqual(
            effective_sample_size(single) / theory_single, 1.0, delta=0.15
        )
        mcse = monte_carlo_standard_error(chains)
        self.assertAlmostEqual(mcse, math.sqrt(1.0 / theory), delta=0.1 * mcse)

    def test_ess_of_antithetic_chains_is_positive_and_capped(self) -> None:
        """Strongly antithetic chains give a positive ESS, not a negative one.

        Regression (benchmark suite, ADR-0049): a static-HMC calibration chain
        had lag-one autocorrelation near -1, the truncated Geyer sum gave
        ``tau = -0.16`` and ``ESS = -2471``, and ``summarize_draws`` crashed in
        ``sqrt``. An alternating chain has ``rho_1 ~ -1``, so the first pair
        sum is non-positive and the uncapped estimate was ``-M n``.

        With stan-dev/posterior's improved estimate (review fix) a stopping
        lag of zero gives ``tau = -1 + 2 rho_0 + rho_0 = 2`` (the R code's
        ``sum(rho_hat_t[1:0])`` is ``rho_0``), so the alternating chain has
        ``ESS = 100/2``; it used to hit the cap ``100 log10(100)``. AR(1)
        chains with ``phi = -0.9`` still reach the cap ``200 log10(200)``.
        """
        alternating = [1.0 if t % 2 == 0 else -1.0 for t in range(100)]
        ess = effective_sample_size([alternating])
        self.assertAlmostEqual(ess, 50.0)
        capped = effective_sample_size(ar1_chains(-0.9, 200, 1, seed=0))
        self.assertAlmostEqual(capped, 200.0 * math.log10(200.0))
        jittered = [
            value + 0.01 * math.sin(3.0 * t) for t, value in enumerate(alternating)
        ]
        summary = summarize_draws([alternating, jittered], name="anti")
        self.assertGreater(summary.effective_sample_size, 0.0)
        self.assertTrue(math.isfinite(summary.mcse_mean))
        self.assertLessEqual(summary.effective_sample_size, 200.0 * math.log10(200.0))

    def test_ess_matches_posterior_improved_estimate(self) -> None:
        """ESS equals stan-dev/posterior ``.ess`` on mildly antithetic chains.

        Four AR(1) chains (``phi = -0.3``, ``n = 100``, ``random.Random(3)``).
        Oracle ``825.6296706351059`` from a numpy port of ``.ess`` in
        posterior's R/convergence.R (FFT autocovariances); without the
        improved-estimate term ``rho_T`` the old code gave ``943.10``.
        """
        generator = random.Random(3)
        chains = []
        for _ in range(4):
            x = [generator.gauss(0.0, 1.0)]
            for _ in range(99):
                x.append(-0.3 * x[-1] + generator.gauss(0.0, math.sqrt(1.0 - 0.09)))
            chains.append(x)
        ess = effective_sample_size(chains, split=False)
        self.assertLess(abs(ess / 825.6296706351059 - 1.0), 1e-12)

    def test_summaries_and_degenerate_chains(self) -> None:
        """Summaries report quantiles; constant chains give undefined diagnostics."""
        summary = summarize_draws([[0.0, 1.0, 2.0, 3.0, 4.0]], name="z")
        self.assertEqual(summary.name, "z")
        self.assertAlmostEqual(summary.quantile(0.5), 2.0)
        self.assertAlmostEqual(summary.mean, 2.0)
        with self.assertRaises(KeyError):
            summary.quantile(0.25)
        self.assertTrue(math.isnan(split_rhat([[1.0] * 6, [1.0] * 6])))
        self.assertTrue(math.isnan(effective_sample_size([[1.0] * 6])))

    def test_sampler_and_diagnostic_errors(self) -> None:
        """Invalid arguments and pathological targets raise explicit errors."""
        cov = ((1.0,),)
        with self.assertRaises(ValueError):
            random_walk_metropolis(bimodal, (0.0,), proposal_covariance=cov, draws=0)
        with self.assertRaises(ValueError):
            random_walk_metropolis(
                bimodal, (0.0,), proposal_covariance=((-1.0,),), draws=5
            )
        with self.assertRaises(ValueError):
            random_walk_metropolis(
                lambda x: -math.inf, (0.0,), proposal_covariance=cov, draws=5
            )
        with self.assertRaises(ValueError):
            random_walk_metropolis(
                lambda x: math.nan, (0.0,), proposal_covariance=cov, draws=5
            )
        with self.assertRaises(ValueError):
            random_walk_metropolis(
                bimodal, ((0.0,),), proposal_covariance=cov, draws=5, chains=2
            )
        with self.assertRaises(TypeError):
            random_walk_metropolis(1.0, (0.0,), proposal_covariance=cov, draws=5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            adaptive_metropolis(bimodal, (0.0,), draws=5, warmup=0)
        with self.assertRaises(ValueError):
            split_rhat([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            effective_sample_size([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            split_rhat([])


class LaplaceApproximationTests(unittest.TestCase):
    """Verify the Gaussian mode approximation."""

    def test_laplace_is_exact_for_gaussian_target(self) -> None:
        """Mode, covariance, and log normalizer match the Gaussian exactly."""
        laplace = laplace_approximation(correlated_gaussian, (0.0, 0.0))
        self.assertTrue(laplace.converged)
        for i in range(2):
            self.assertAlmostEqual(laplace.mode[i], TARGET_MEAN[i], places=6)
            for j in range(2):
                self.assertAlmostEqual(
                    laplace.covariance[i][j], TARGET_COVARIANCE[i][j], places=6
                )
        determinant = 1.0 * 2.0 - 0.8 * 0.8
        expected = 0.5 * math.log((2.0 * math.pi) ** 2 * determinant)
        self.assertAlmostEqual(laplace.log_normalizer, expected, places=6)
        self.assertAlmostEqual(laplace.log_pdf(TARGET_MEAN), -expected, places=6)
        draws = laplace.sample(4000, seed=2)
        mean0 = math.fsum(d[0] for d in draws) / len(draws)
        self.assertAlmostEqual(mean0, 1.0, delta=0.1)

    def test_laplace_rejects_flat_direction(self) -> None:
        """A direction without curvature has no Gaussian approximation."""
        with self.assertRaises(ValueError):
            laplace_approximation(lambda x: -(x[0] ** 2), (0.3, 0.2))
        with self.assertRaises(ValueError):
            laplace_approximation(lambda x: -math.inf, (0.0,))


class NormalInverseGammaTests(unittest.TestCase):
    """Verify the unknown-variance conjugate linear-regression update."""

    def test_student_t_functions_match_closed_forms(self) -> None:
        """Cauchy and two-degree-of-freedom laws have closed forms."""
        self.assertAlmostEqual(
            student_t_quantile(0.975, 1.0), math.tan(0.475 * math.pi), places=9
        )
        x = 1.3
        self.assertAlmostEqual(
            student_t_cdf(x, 2.0), 0.5 + x / (2.0 * math.sqrt(2.0 + x * x)), places=12
        )
        self.assertAlmostEqual(student_t_cdf(-x, 2.0), 1.0 - student_t_cdf(x, 2.0))
        self.assertAlmostEqual(student_t_quantile(0.975, 10.0), 2.228138852, places=8)
        self.assertAlmostEqual(
            student_t_quantile(0.9, 1e8), standard_normal_quantile(0.9), places=6
        )
        with self.assertRaises(ValueError):
            student_t_quantile(1.0, 3.0)

    def test_update_matches_hand_computed_posterior(self) -> None:
        """Mean model, prior (0, 1, 2, 2), data (1, 2, 3) gives (1.5, 1/4, 3.5, 4.5)."""
        prior = NormalInverseGamma((0.0,), ((1.0,),), 2.0, 2.0)
        posterior = condition_linear_normal_inverse_gamma(
            prior, ((1.0,), (1.0,), (1.0,)), (1.0, 2.0, 3.0)
        )
        self.assertAlmostEqual(posterior.mean[0], 1.5)
        self.assertAlmostEqual(posterior.scale_matrix[0][0], 0.25)
        self.assertAlmostEqual(posterior.shape, 3.5)
        self.assertAlmostEqual(posterior.rate, 4.5)
        self.assertAlmostEqual(posterior.noise_variance_mean, 1.8)
        self.assertAlmostEqual(posterior.noise_variance_mode, 1.0)
        self.assertEqual(posterior.degrees_of_freedom, 7.0)
        low, high = posterior.coefficient_interval(0, 0.95)
        half = student_t_quantile(0.975, 7.0) * math.sqrt(4.5 / 3.5 * 0.25)
        self.assertAlmostEqual(low, 1.5 - half)
        self.assertAlmostEqual(high, 1.5 + half)
        predictive = posterior.predictive(((1.0,),))
        self.assertAlmostEqual(predictive.scale_matrix[0][0], 4.5 / 3.5 * 1.25)
        self.assertAlmostEqual(predictive.variances[0], 7.0 / 5.0 * 4.5 / 3.5 * 1.25)
        low_s2, high_s2 = posterior.noise_variance_interval(0.95)
        self.assertLess(low_s2, 1.8)
        self.assertGreater(high_s2, 1.8)
        draws = posterior.sample(4000, seed=5)
        inside = sum(low_s2 <= variance <= high_s2 for _, variance in draws) / 4000
        self.assertAlmostEqual(inside, 0.95, delta=0.015)

    def test_conditional_on_variance_matches_known_variance_update(self) -> None:
        """Given sigma^2, beta's law equals the Gaussian conjugate update."""
        design = ((1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 4.0))
        y = (0.9, 2.1, 2.8, 5.2)
        prior = NormalInverseGamma((0.0, 1.0), ((4.0, 0.5), (0.5, 1.0)), 3.0, 1.0)
        posterior = condition_linear_normal_inverse_gamma(prior, design, y)
        variance = 0.37
        gaussian = condition_linear_gaussian(
            MultivariateNormal(
                (0.0, 1.0),
                ((4.0 * variance, 0.5 * variance), (0.5 * variance, variance)),
            ),
            design,
            y,
            observation_variance=variance,
        )
        for i in range(2):
            self.assertAlmostEqual(posterior.mean[i], gaussian.mean[i], places=10)
            for j in range(2):
                self.assertAlmostEqual(
                    variance * posterior.scale_matrix[i][j],
                    gaussian.covariance[i][j],
                    places=10,
                )

    def test_invalid_normal_inverse_gamma_inputs(self) -> None:
        """Invalid parameters and shapes raise errors."""
        with self.assertRaises(ValueError):
            NormalInverseGamma((0.0,), ((1.0,),), 0.0, 1.0)
        with self.assertRaises(ValueError):
            NormalInverseGamma((0.0,), ((-1.0,),), 1.0, 1.0)
        prior = NormalInverseGamma((0.0,), ((1.0,),), 1.0, 1.0)
        with self.assertRaises(ValueError):
            prior.noise_variance_mean  # noqa: B018
        with self.assertRaises(ValueError):
            condition_linear_normal_inverse_gamma(prior, ((1.0,),), (1.0, 2.0))
        with self.assertRaises(TypeError):
            condition_linear_normal_inverse_gamma(None, ((1.0,),), (1.0,))  # type: ignore[arg-type]


class NonlinearCalibrationTests(unittest.TestCase):
    """Verify calibration of general models against exact and synthetic truth."""

    @classmethod
    def setUpClass(cls) -> None:
        """Sample the known-noise decay posterior once for several tests."""
        cls.decay = decay_problem(NoiseModel("y", standard_deviation=TRUE_SIGMA))
        cls.decay_posterior = cls.decay.sample_posterior(
            draws=800, warmup=400, chains=4, seed=3
        )

    def test_linear_case_matches_conjugate_posterior(self) -> None:
        """Laplace is exact and MCMC agrees within MC error with the closed form."""
        problem, xs, ys = line_problem()
        exact = condition_linear_gaussian(
            MultivariateNormal((0.0, 1.0), ((4.0, 0.0), (0.0, 1.0))),
            [(1.0, x) for x in xs],
            ys,
            observation_variance=0.09,
        )
        laplace = problem.laplace_posterior()
        for i in range(2):
            self.assertAlmostEqual(laplace.mode[i], exact.mean[i], places=6)
            for j in range(2):
                self.assertAlmostEqual(
                    laplace.covariance[i][j], exact.covariance[i][j], places=6
                )
        posterior = problem.sample_posterior(draws=1000, warmup=500, chains=4, seed=11)
        for index, summary in enumerate(posterior.summary()):
            with self.subTest(parameter=summary.name):
                self.assertLess(
                    abs(summary.mean - exact.mean[index]), 4.0 * summary.mcse_mean
                )
                self.assertAlmostEqual(
                    summary.standard_deviation,
                    math.sqrt(exact.covariance[index][index]),
                    delta=0.15 * math.sqrt(exact.covariance[index][index]),
                )
                self.assertLess(summary.rank_normalized_rhat, 1.05)

    def test_standard_space_density_absorbs_prior_jacobian(self) -> None:
        """Physical and standard-space log posteriors differ by a constant."""
        problem = decay_problem(
            NoiseModel("y", standard_deviation_prior=LogNormal(math.log(0.2), 1.0))
        )
        inputs = problem.model.inputs
        differences = []
        for u0, u1, eta in ((0.1, -0.3, -2.0), (1.2, 0.4, -1.5), (-0.7, 0.9, -2.5)):
            theta = inputs.from_standard_normal((u0, u1))
            physical = problem.log_posterior(theta + (eta,))
            log_jacobian = sum(
                -0.5 * u * u - 0.5 * math.log(2.0 * math.pi) - item.log_pdf(value)
                for u, item, value in zip((u0, u1), inputs.inputs, theta, strict=False)
            )
            differences.append(
                physical + log_jacobian - problem.standard_log_posterior((u0, u1, eta))
            )
        for value in differences:
            self.assertAlmostEqual(value, -math.log(2.0 * math.pi), places=8)

    def test_decay_least_squares_and_identifiability(self) -> None:
        """LM recovers the truth within three standard errors."""
        fit = self.decay.fit_least_squares()
        self.assertTrue(fit.converged)
        truth = (TRUE_AMPLITUDE, TRUE_RATE)
        for value, error, target in zip(
            fit.estimate, fit.standard_errors, truth, strict=False
        ):
            self.assertLess(abs(value - target), 3.0 * error)
        self.assertGreater(fit.evaluations, 0)
        self.assertTrue(fit.identifiability.identifiable)
        self.assertIsNone(fit.identifiability.warning)
        self.assertEqual(fit.as_mapping()["A"], fit.estimate[0])

    def test_decay_posterior_recovers_parameters(self) -> None:
        """Posterior means lie within three posterior SDs of the truth."""
        posterior = self.decay_posterior
        summaries = {s.name: s for s in posterior.summary()}
        for name, truth in (("A", TRUE_AMPLITUDE), ("k", TRUE_RATE)):
            with self.subTest(parameter=name):
                summary = summaries[name]
                self.assertLess(
                    abs(summary.mean - truth), 3.0 * summary.standard_deviation
                )
                self.assertLess(summary.rank_normalized_rhat, 1.05)
        self.assertEqual(posterior.noise_names, ())
        self.assertGreater(posterior.evaluations, 0)
        self.assertIsNotNone(posterior.laplace)
        self.assertEqual(len(posterior.draws()), 3200)

    def test_unknown_noise_scale_is_recovered(self) -> None:
        """The log-sigma posterior covers the true noise scale."""
        problem = decay_problem(
            NoiseModel("y", standard_deviation_prior=LogNormal(math.log(0.3), 1.0))
        )
        fit = problem.fit_least_squares()
        self.assertAlmostEqual(
            fit.noise_standard_deviations["y"], TRUE_SIGMA, delta=0.03
        )
        posterior = problem.sample_posterior(draws=800, warmup=500, chains=4, seed=3)
        summaries = {s.name: s for s in posterior.summary()}
        sigma = summaries["sigma[y]"]
        self.assertLess(sigma.quantile(0.05), TRUE_SIGMA)
        self.assertGreater(sigma.quantile(0.95), TRUE_SIGMA)
        self.assertAlmostEqual(sigma.mean, TRUE_SIGMA, delta=0.3 * TRUE_SIGMA)
        self.assertLess(sigma.rank_normalized_rhat, 1.05)

    def test_push_forward_prediction_covers_truth(self) -> None:
        """Latent 95% interval covers the true response; noise widens it."""
        prediction = predict(
            self.decay,
            self.decay_posterior,
            ["y"],
            [{"t": 12.0}, {"t": 2.0}],
            include_noise=True,
            max_draws=1600,
            seed=1,
        )
        self.assertEqual(prediction.draws, 1600)
        self.assertEqual(prediction.evaluations, 3200)
        for index, t in enumerate((12.0, 2.0)):
            with self.subTest(t=t):
                item = prediction.get("y", index)
                truth = TRUE_AMPLITUDE * math.exp(-TRUE_RATE * t)
                low, high = item.interval(0.95)
                self.assertLessEqual(low, truth)
                self.assertGreaterEqual(high, truth)
                noisy_low, noisy_high = item.interval(0.95, observed=True)
                self.assertLess(noisy_low, low)
                self.assertGreater(noisy_high, high)
                expected = item.latent_variance + TRUE_SIGMA**2
                self.assertAlmostEqual(
                    item.observed_variance, expected, delta=0.15 * expected
                )
        latent_only = predict(
            self.decay.model, self.decay_posterior.draws()[:50], conditions=[{"t": 1.0}]
        )
        self.assertIsNone(latent_only.get("y").observed)
        with self.assertRaises(ValueError):
            latent_only.get("y").observed_variance  # noqa: B018

    def test_student_t_likelihood_resists_an_outlier(self) -> None:
        """A robust likelihood moves the MAP estimate less than a Gaussian one."""
        prior = JointInput.independent(
            RandomInput("a", Normal(0.0, 10.0)), RandomInput("b", Normal(0.0, 10.0))
        )
        model = ProbabilisticModel(prior, line_model, (QuantityOfInterest("y", "y"),))
        xs = [0.1 * i for i in range(20)]
        generator = random.Random(9)
        ys = [1.0 + 2.0 * x + generator.gauss(0.0, 0.1) for x in xs]
        ys[10] += 5.0
        experiments = tuple(
            Experiment({"y": y}, {"x": x}) for x, y in zip(xs, ys, strict=False)
        )
        gaussian = CalibrationProblem(
            model, experiments, (NoiseModel("y", standard_deviation=0.1),)
        )
        robust = CalibrationProblem(
            model,
            experiments,
            (NoiseModel("y", standard_deviation=0.1, degrees_of_freedom=3.0),),
        )
        gaussian_map = gaussian.laplace_posterior().mode
        robust_map = robust.laplace_posterior().mode
        self.assertLess(abs(robust_map[0] - 1.0), abs(gaussian_map[0] - 1.0))
        self.assertLess(abs(robust_map[0] - 1.0), 0.1)

    def test_discrepancy_variance_adds_to_noise(self) -> None:
        """Gaussian likelihood uses sigma^2 plus discrepancy variance."""
        problem, xs, ys = line_problem()
        inflated = CalibrationProblem(
            problem.model,
            problem.experiments,
            (NoiseModel("y", standard_deviation=0.3, discrepancy_variance=0.16),),
        )
        theta = (1.0, 0.7)
        predictions = [theta[0] + theta[1] * x for x in xs]
        expected = gaussian_log_likelihood(ys, predictions, variance=0.25)
        self.assertAlmostEqual(inflated.log_likelihood(theta), expected, places=10)

    def test_product_model_is_flagged_non_identifiable(self) -> None:
        """Only ``a b`` is informed, so ``J'J`` is rank deficient."""
        prior = JointInput.independent(
            RandomInput("a", Normal(1.0, 1.0)), RandomInput("b", Normal(1.0, 1.0))
        )
        model = ProbabilisticModel(
            prior, product_model, (QuantityOfInterest("y", "y"),)
        )
        experiments = tuple(
            Experiment({"y": 2.0 * x}, {"x": x}) for x in (1.0, 2.0, 3.0)
        )
        problem = CalibrationProblem(
            model, experiments, (NoiseModel("y", standard_deviation=0.1),)
        )
        diagnostic = problem.identifiability()
        self.assertFalse(diagnostic.identifiable)
        self.assertEqual(diagnostic.rank, 1)
        self.assertIsNotNone(diagnostic.warning)
        weak_value, weak_direction = diagnostic.weak_directions[0]
        self.assertAlmostEqual(
            abs(weak_direction["a"]), abs(weak_direction["b"]), places=5
        )
        self.assertLess(weak_direction["a"] * weak_direction["b"], 0.0)
        conditioned = fisher_identifiability(
            ((1.0, 0.0), (0.0, 1e-5)), ("p", "q"), condition_warning=1e8
        )
        self.assertTrue(conditioned.identifiable)
        self.assertIsNotNone(conditioned.warning)
        self.assertAlmostEqual(conditioned.condition_number, 1e10, delta=1e4)

    def test_calibration_input_errors(self) -> None:
        """Invalid declarations and prediction requests raise errors."""
        problem, _, _ = line_problem()
        with self.assertRaises(ValueError):
            NoiseModel("y")
        with self.assertRaises(ValueError):
            NoiseModel("y", standard_deviation=1.0, standard_deviation_prior=Normal())
        with self.assertRaises(ValueError):
            NoiseModel("y", standard_deviation=-1.0)
        with self.assertRaises(ValueError):
            NoiseModel("y", standard_deviation=1.0, degrees_of_freedom=0.0)
        with self.assertRaises(ValueError):
            Experiment({})
        with self.assertRaises(ValueError):
            CalibrationProblem(
                problem.model,
                (Experiment({"z": 1.0}),),
                (NoiseModel("y", standard_deviation=1.0),),
            )
        with self.assertRaises(ValueError):
            CalibrationProblem(
                problem.model,
                (Experiment({"y": 1.0}, {"a": 1.0}),),
                (NoiseModel("y", standard_deviation=1.0),),
            )
        with self.assertRaises(ValueError):
            CalibrationProblem(
                problem.model,
                (Experiment({"y": 1.0}),),
                (
                    NoiseModel("y", standard_deviation=1.0),
                    NoiseModel("y", standard_deviation=2.0),
                ),
            )
        with self.assertRaises(ValueError):
            problem.log_posterior((1.0,))
        with self.assertRaises(ValueError):
            predict(problem, [{"a": 1.0}], conditions=[{"x": 1.0}])
        with self.assertRaises(KeyError):
            predict(problem, [{"a": 1.0, "b": 1.0}], ["nope"], [{"x": 1.0}])
        with self.assertRaises(ValueError):
            predict(problem.model, [{"a": 1.0, "b": 1.0}], include_noise=True)
        with self.assertRaises(ValueError):
            problem.sample_posterior(draws=10, warmup=10, chains=0)


if __name__ == "__main__":
    unittest.main()
