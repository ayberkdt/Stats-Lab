"""Tests for pCN, tempered SMC with evidence, ABC, and their calibration wiring.

Exact answers come from conjugate normal models: the posterior and the
marginal likelihood in closed form, and the ABC posterior of a sufficient
summary under a uniform kernel by one-dimensional quadrature. Seeded Monte
Carlo results are judged by bands of a few standard errors against those
answers, never by pinned digits.
"""

from __future__ import annotations

import math
import random
import unittest
import warnings
from statistics import NormalDist

from probstat.distributions import LogNormal, Normal
from probstat.inference import (
    ABCResult,
    ConvergenceWarning,
    SMCResult,
    abc_rejection,
    abc_smc,
    euclidean_distance,
    preconditioned_crank_nicolson,
    random_walk_metropolis,
    sequential_monte_carlo,
)
from probstat.inference.smc import _next_temperature
from probstat.inverse import (
    CalibrationProblem,
    Experiment,
    NoiseModel,
    calibrate_linear_gaussian,
)
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.multivariate import MultivariateNormal
from probstat.uq import QuantityOfInterest

# Conjugate model: x_k ~ N(0, 1) a priori, y_kj ~ N(x_k, NOISE^2) independently.
DATA = ((0.8, 1.1, 0.5, 1.3), (-0.4, -0.9, -0.2, -0.6))
NOISE = 0.5
STANDARD_NORMAL = NormalDist()


def conjugate_log_likelihood(x: tuple[float, ...]) -> float:
    """Return the fully normalized Gaussian log-likelihood of ``DATA``."""
    return math.fsum(
        -0.5 * math.log(2.0 * math.pi * NOISE * NOISE)
        - 0.5 * ((y - center) / NOISE) ** 2
        for center, ys in zip(x, DATA, strict=True)
        for y in ys
    )


def standard_normal_log_prior(x: tuple[float, ...]) -> float:
    """Return ``-|x|^2 / 2``."""
    return -0.5 * math.fsum(value * value for value in x)


def standard_normal_draw(generator: random.Random) -> tuple[float, float]:
    """Draw one point of the two-dimensional standard normal prior."""
    return (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))


def conjugate_exact() -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Return the exact log evidence, posterior means, and posterior variances."""
    log_evidence = 0.0
    means = []
    variances = []
    for ys in DATA:
        n = len(ys)
        precision = 1.0 + n / NOISE**2
        means.append(math.fsum(ys) / NOISE**2 / precision)
        variances.append(1.0 / precision)
        # y ~ N(0, NOISE^2 I + 1 1'): determinant lemma and Sherman-Morrison.
        log_det = n * math.log(NOISE**2) + math.log(precision)
        quadratic = (
            math.fsum(y * y for y in ys) / NOISE**2
            - (math.fsum(ys) / NOISE**2) ** 2 / precision
        )
        log_evidence += -0.5 * (n * math.log(2.0 * math.pi) + log_det + quadratic)
    return log_evidence, tuple(means), tuple(variances)


# Discretized random field observed at three points (pCN dimension test).
FIELD_POINTS = (0.25, 0.5, 0.75)
FIELD_DATA = (0.6, 0.9, 0.4)
FIELD_NOISE = 0.2


def field_log_likelihood(dimension: int):
    """Return the log-likelihood of a sine-series field with decaying modes."""
    basis = [
        [
            math.sqrt(2.0) * math.sin((k + 1) * math.pi * s) / (k + 1)
            for k in range(dimension)
        ]
        for s in FIELD_POINTS
    ]

    def log_likelihood(u: tuple[float, ...]) -> float:
        total = 0.0
        for row, y in zip(basis, FIELD_DATA, strict=True):
            value = math.fsum(a * b for a, b in zip(row, u, strict=True))
            total += -0.5 * ((y - value) / FIELD_NOISE) ** 2
        return total

    return log_likelihood


# ABC: theta ~ N(0, 1), summary ybar ~ N(theta, 1 / ABC_N) (sufficient).
ABC_N = 10
ABC_OBSERVED = 0.7


def abc_simulator(theta: tuple[float, ...], generator: random.Random):
    """Simulate the sample mean of ``ABC_N`` unit-variance observations."""
    return (generator.gauss(theta[0], 1.0 / math.sqrt(ABC_N)),)


def abc_prior(generator: random.Random) -> tuple[float]:
    """Draw from the standard normal prior."""
    return (generator.gauss(0.0, 1.0),)


def abc_log_prior(theta: tuple[float, ...]) -> float:
    """Return the standard normal log-density up to a constant."""
    return -0.5 * theta[0] ** 2


def abc_exact_moments(tolerance: float) -> tuple[float, float]:
    """Return the mean and variance of the uniform-kernel ABC posterior.

    ``pi_eps(theta) ∝ phi(theta) [Phi((y + eps - theta) sqrt n) -
    Phi((y - eps - theta) sqrt n)]``, integrated by the trapezoid rule.
    """
    root = math.sqrt(ABC_N)
    step = 0.002
    grid = [-6.0 + step * i for i in range(6001)]
    weights = [
        math.exp(-0.5 * x * x)
        * (
            STANDARD_NORMAL.cdf((ABC_OBSERVED + tolerance - x) * root)
            - STANDARD_NORMAL.cdf((ABC_OBSERVED - tolerance - x) * root)
        )
        for x in grid
    ]
    total = math.fsum(weights)
    mean = math.fsum(w * x for w, x in zip(weights, grid, strict=True)) / total
    variance = (
        math.fsum(w * (x - mean) ** 2 for w, x in zip(weights, grid, strict=True))
        / total
    )
    return mean, variance


ABC_POSTERIOR_MEAN = ABC_N * ABC_OBSERVED / (ABC_N + 1)
ABC_POSTERIOR_VARIANCE = 1.0 / (ABC_N + 1)


def line_model(parameters):
    """Return a straight-line response."""
    return {"y": parameters["a"] + parameters["b"] * parameters["x"]}


LINE_X = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def line_data() -> list[float]:
    """Return the seeded straight-line observations."""
    generator = random.Random(5)
    return [1.0 + 0.7 * x + generator.gauss(0.0, 0.3) for x in LINE_X]


def line_problem(noise: NoiseModel) -> CalibrationProblem:
    """Build the linear-Gaussian calibration with independent normal priors."""
    prior = JointInput.independent(
        RandomInput("a", Normal(0.0, 2.0)), RandomInput("b", Normal(1.0, 1.0))
    )
    model = ProbabilisticModel(prior, line_model, (QuantityOfInterest("y", "line"),))
    experiments = tuple(
        Experiment({"y": y}, {"x": x}) for x, y in zip(LINE_X, line_data(), strict=True)
    )
    return CalibrationProblem(model, experiments, (noise,))


LINE_PRIOR = MultivariateNormal((0.0, 1.0), ((4.0, 0.0), (0.0, 1.0)))


def line_marginal_log_likelihood(sigma: float) -> float:
    """Return ``log N(y; X m, X S X' + sigma^2 I)`` for the line problem."""
    mean = tuple(0.0 + 1.0 * x for x in LINE_X)
    covariance = tuple(
        tuple(
            4.0 + xi * xj + (sigma * sigma if i == j else 0.0)
            for j, xj in enumerate(LINE_X)
        )
        for i, xi in enumerate(LINE_X)
    )
    return MultivariateNormal(mean, covariance).log_pdf(line_data())


class PreconditionedCrankNicolsonTests(unittest.TestCase):
    """pCN on a conjugate target and on a refined random field."""

    def test_recovers_the_conjugate_posterior(self) -> None:
        """Means within 4 MCSE, variances within 15 %, acceptance near target."""
        _, means, variances = conjugate_exact()
        result = preconditioned_crank_nicolson(
            conjugate_log_likelihood,
            (0.0, 0.0),
            draws=1500,
            warmup=500,
            chains=4,
            seed=1,
        )
        self.assertEqual(result.algorithm, "preconditioned Crank-Nicolson")
        self.assertEqual(result.draws_per_chain, 1500)
        for summary, mean, variance in zip(
            result.summary(), means, variances, strict=False
        ):
            with self.subTest(parameter=summary.name):
                self.assertLess(abs(summary.mean - mean), 4.0 * summary.mcse_mean)
                self.assertAlmostEqual(
                    summary.standard_deviation**2, variance, delta=0.15 * variance
                )
                self.assertLess(summary.rank_normalized_rhat, 1.05)
        self.assertGreater(result.acceptance_rate, 0.15)
        self.assertLess(result.acceptance_rate, 0.40)
        for chain in result.chains:
            beta = math.sqrt(chain.proposal_covariance[0][0])
            self.assertGreater(beta, 0.0)
            self.assertLessEqual(beta, 1.0)
            self.assertEqual(chain.proposal_covariance[0][1], 0.0)
            # The trace is log L(u) - |u|^2 / 2.
            sample, value = chain.samples[-1], chain.log_density[-1]
            self.assertAlmostEqual(
                value,
                conjugate_log_likelihood(sample) + standard_normal_log_prior(sample),
                delta=1e-9 * max(1.0, abs(value)),
            )

    def test_acceptance_is_dimension_robust_while_random_walk_collapses(self) -> None:
        """Fixed beta keeps acceptance in a band as d grows; a fixed RWM fails."""
        pcn_rates = []
        rwm_rates = []
        for dimension in (10, 50, 200):
            log_likelihood = field_log_likelihood(dimension)
            generator = random.Random(1)
            start = [generator.gauss(0.0, 1.0) for _ in range(dimension)]
            pcn = preconditioned_crank_nicolson(
                log_likelihood,
                start,
                draws=1000,
                warmup=0,
                chains=1,
                step=0.3,
                adapt=False,
                seed=7,
                check_convergence=False,
            )
            self.assertTrue(math.isnan(pcn.chains[0].warmup_acceptance_rate))
            pcn_rates.append(pcn.acceptance_rate)

            def target(u, log_likelihood=log_likelihood):
                return log_likelihood(u) + standard_normal_log_prior(u)

            rwm = random_walk_metropolis(
                target,
                start,
                proposal_covariance=[
                    [0.09 if i == j else 0.0 for j in range(dimension)]
                    for i in range(dimension)
                ],
                draws=200,
                chains=1,
                seed=7,
                check_convergence=False,
            )
            rwm_rates.append(rwm.acceptance_rate)
        for rate in pcn_rates:
            self.assertGreater(rate, 0.10)
            self.assertLess(rate, 0.32)
        self.assertLess(max(pcn_rates) - min(pcn_rates), 0.08)
        self.assertGreater(rwm_rates[0], 0.15)
        self.assertLess(rwm_rates[2], 0.12)
        self.assertLess(rwm_rates[2], 0.5 * rwm_rates[0])

    def test_independence_sampler_and_reproducibility(self) -> None:
        """A unit step proposes prior draws; integer seeds reproduce the chains."""
        first = preconditioned_crank_nicolson(
            conjugate_log_likelihood,
            (0.0, 0.0),
            draws=50,
            warmup=0,
            chains=2,
            step=1.0,
            adapt=False,
            seed=3,
            check_convergence=False,
        )
        second = preconditioned_crank_nicolson(
            conjugate_log_likelihood,
            (0.0, 0.0),
            draws=50,
            warmup=0,
            chains=2,
            step=1.0,
            adapt=False,
            seed=3,
            check_convergence=False,
        )
        self.assertEqual(first.chains[0].samples, second.chains[0].samples)
        self.assertNotEqual(first.chains[0].samples, first.chains[1].samples)
        self.assertEqual(first.chains[0].proposal_covariance[0][0], 1.0)
        self.assertEqual(first.evaluations, 2 * 51)

    def test_convergence_warning_and_input_validation(self) -> None:
        """Short runs warn; invalid steps, targets, and starts raise."""
        with self.assertWarns(ConvergenceWarning):
            preconditioned_crank_nicolson(
                conjugate_log_likelihood, (0.0, 0.0), draws=20, warmup=0, seed=1
            )
        cases = (
            {"step": 0.0},
            {"step": 1.5},
            {"target_acceptance": 1.0},
            {"draws": 0},
        )
        for case in cases:
            options = {"draws": 10, "warmup": 0, "seed": 1, **case}
            with self.subTest(case=case), self.assertRaises(ValueError):
                preconditioned_crank_nicolson(
                    conjugate_log_likelihood, (0.0, 0.0), **options
                )
        with self.assertRaises(ValueError):
            preconditioned_crank_nicolson(
                lambda u: -math.inf, (0.0,), draws=5, warmup=0, seed=1
            )
        with self.assertRaises(ValueError):
            preconditioned_crank_nicolson(
                lambda u: math.nan, (0.0,), draws=5, warmup=0, seed=1
            )
        with self.assertRaises(TypeError):
            preconditioned_crank_nicolson(None, (0.0,), draws=5, warmup=0)
        with self.assertRaises(TypeError):
            preconditioned_crank_nicolson(
                conjugate_log_likelihood, (0.0, 0.0), draws=5, warmup=0, adapt=1
            )


class SequentialMonteCarloTests(unittest.TestCase):
    """Tempered SMC: posterior, evidence, schedule, and error estimate."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the random-walk SMC on the conjugate model once."""
        cls.exact = conjugate_exact()
        cls.result = sequential_monte_carlo(
            standard_normal_draw,
            standard_normal_log_prior,
            conjugate_log_likelihood,
            particles=1000,
            seed=2,
        )

    def test_evidence_matches_the_exact_marginal_likelihood(self) -> None:
        """The log evidence is within 4 Lee-Whiteley SEs and 0.25 of exact."""
        result = self.result
        error = result.log_evidence - self.exact[0]
        self.assertTrue(math.isfinite(result.log_evidence_standard_error))
        self.assertGreater(result.log_evidence_standard_error, 0.0)
        self.assertLess(abs(error), 4.0 * result.log_evidence_standard_error)
        self.assertLess(abs(error), 0.25)
        self.assertAlmostEqual(
            result.log_evidence,
            math.fsum(result.log_evidence_increments),
            delta=1e-12 * abs(result.log_evidence),
        )

    def test_posterior_moments_match_the_conjugate_posterior(self) -> None:
        """Particle means within a quarter posterior SD; variances within 20 %."""
        _, means, variances = self.exact
        mean = self.result.mean()
        covariance = self.result.covariance()
        for k in range(2):
            with self.subTest(coordinate=k):
                tolerance = 0.25 * math.sqrt(variances[k])
                self.assertLess(abs(mean[k] - means[k]), tolerance)
                self.assertAlmostEqual(
                    covariance[k][k], variances[k], delta=0.2 * variances[k]
                )

    def test_schedule_ess_and_bookkeeping(self) -> None:
        """Temperatures rise to one; each stage keeps half the particles."""
        result = self.result
        self.assertIsInstance(result, SMCResult)
        self.assertEqual(result.temperatures[0], 0.0)
        self.assertEqual(result.temperatures[-1], 1.0)
        self.assertTrue(
            all(
                a < b
                for a, b in zip(
                    result.temperatures, result.temperatures[1:], strict=False
                )
            )
        )
        self.assertEqual(result.stages, len(result.effective_sample_sizes))
        self.assertEqual(result.stages, len(result.acceptance_rates))
        for ess in result.effective_sample_sizes:
            self.assertGreaterEqual(ess, 0.5 * 1000 * (1.0 - 1e-9))
        for rate in result.acceptance_rates:
            self.assertGreater(rate, 0.05)
            self.assertLess(rate, 0.95)
        self.assertEqual(len(result.particles), 1000)
        self.assertAlmostEqual(math.fsum(result.weights), 1.0, places=12)
        self.assertEqual(result.dimension, 2)
        self.assertEqual(result.names, ("x0", "x1"))
        self.assertGreater(result.evaluations, 1000 * result.stages)
        for point, value in zip(
            result.particles[:20], result.log_likelihoods[:20], strict=False
        ):
            self.assertAlmostEqual(
                value, conjugate_log_likelihood(point), delta=1e-9 * abs(value)
            )

    def test_pcn_mutation_and_systematic_resampling(self) -> None:
        """A pCN mutation is accurate; systematic resampling reports no SE."""
        log_evidence, means, variances = self.exact
        pcn = sequential_monte_carlo(
            standard_normal_draw,
            None,
            conjugate_log_likelihood,
            particles=800,
            mutation="pcn",
            seed=4,
        )
        self.assertEqual(pcn.mutation, "pcn")
        self.assertLess(
            abs(pcn.log_evidence - log_evidence),
            max(4.0 * pcn.log_evidence_standard_error, 0.1),
        )
        for k in range(2):
            tolerance = 0.3 * math.sqrt(variances[k])
            self.assertLess(abs(pcn.mean()[k] - means[k]), tolerance)
        for scale in pcn.proposal_scales:
            self.assertGreater(scale, 0.0)
            self.assertLessEqual(scale, 1.0)
        systematic = sequential_monte_carlo(
            standard_normal_draw,
            standard_normal_log_prior,
            conjugate_log_likelihood,
            particles=500,
            resampling="systematic",
            seed=4,
        )
        self.assertTrue(math.isnan(systematic.log_evidence_relative_variance))
        self.assertTrue(math.isnan(systematic.log_evidence_standard_error))
        self.assertLess(abs(systematic.log_evidence - log_evidence), 0.4)

    def test_truncated_likelihood_has_exact_one_stage_evidence(self) -> None:
        """L = 1{x > 0}: Z = 1/2 in one stage; the error estimate is 1/sqrt(N)."""
        count = 2000

        def half_line(x):
            return 0.0 if x[0] > 0.0 else -math.inf

        result = sequential_monte_carlo(
            lambda generator: (generator.gauss(0.0, 1.0),),
            standard_normal_log_prior,
            half_line,
            particles=count,
            seed=6,
        )
        self.assertEqual(result.temperatures, (0.0, 1.0))
        standard_error = math.sqrt(1.0 / count)
        self.assertLess(abs(result.log_evidence - math.log(0.5)), 4.0 * standard_error)
        self.assertAlmostEqual(
            result.log_evidence_standard_error,
            standard_error,
            delta=0.1 * standard_error,
        )
        self.assertTrue(all(point[0] > 0.0 for point in result.particles))

    def test_reproducibility_and_prior_draws(self) -> None:
        """Integer seeds reproduce the run; explicit prior draws are accepted."""
        draws = [standard_normal_draw(random.Random(i)) for i in range(200)]
        first = sequential_monte_carlo(
            draws, standard_normal_log_prior, conjugate_log_likelihood, seed=8
        )
        second = sequential_monte_carlo(
            draws, standard_normal_log_prior, conjugate_log_likelihood, seed=8
        )
        self.assertEqual(first.particles, second.particles)
        self.assertEqual(first.log_evidence, second.log_evidence)
        self.assertEqual(len(first.particles), 200)

    def test_next_temperature_bisection(self) -> None:
        """The chosen step attains the ESS target; a flat likelihood ends at one."""
        values = [0.1 * i - 30.0 for i in range(100)]
        beta = _next_temperature(values, 0.0, 50.0)
        self.assertGreater(beta, 0.0)
        self.assertLess(beta, 1.0)
        weights = [math.exp(beta * (value - max(values))) for value in values]
        ess = math.fsum(weights) ** 2 / math.fsum(w * w for w in weights)
        self.assertAlmostEqual(ess, 50.0, delta=1e-6)
        self.assertEqual(_next_temperature([1.0] * 10, 0.3, 10.0), 1.0)
        with self.assertRaises(ValueError):
            _next_temperature([1.0, 2.0], 1.0, 1.0)
        with self.assertRaises(ValueError):
            _next_temperature([1.0, -math.inf], 0.0, 2.0)
        with self.assertRaises(ValueError):
            _next_temperature([1.0, math.nan], 0.0, 1.0)

    def test_input_validation(self) -> None:
        """Invalid mutation, prior, and tempering options raise."""
        sampler, prior, likelihood = (
            standard_normal_draw,
            standard_normal_log_prior,
            conjugate_log_likelihood,
        )
        with self.assertRaises(ValueError):
            sequential_monte_carlo(sampler, prior, likelihood, mutation="hmc")
        with self.assertRaises(ValueError):
            sequential_monte_carlo(sampler, prior, likelihood, mutation="pcn")
        with self.assertRaises(TypeError):
            sequential_monte_carlo(sampler, None, likelihood, mutation="rwm")
        with self.assertRaises(TypeError):
            sequential_monte_carlo(sampler, prior, None)
        for options in (
            {"resampling": "residual"},
            {"ess_fraction": 1.0},
            {"mutation_steps": 0},
            {"target_acceptance": 0.0},
            {"regularization": -1.0},
            {"max_stages": 0},
            {"proposal_scale": -1.0},
            {"particles": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                sequential_monte_carlo(sampler, prior, likelihood, seed=1, **options)
        with self.assertRaises(ValueError):
            sequential_monte_carlo(
                [(0.0, 0.0), (1.0, 1.0)], prior, likelihood, particles=3
            )
        with self.assertRaises(ValueError):
            sequential_monte_carlo(
                sampler, None, likelihood, mutation="pcn", proposal_scale=1.5
            )
        with self.assertRaises(ValueError):
            sequential_monte_carlo(
                sampler, prior, likelihood, particles=50, max_stages=1, seed=1
            )
        with self.assertRaises(ValueError):
            sequential_monte_carlo(
                sampler, prior, lambda x: -math.inf, particles=10, seed=1
            )
        with self.assertRaises(ValueError):
            sequential_monte_carlo(sampler, prior, likelihood, names=("a",), seed=1)


class ApproximateBayesianComputationTests(unittest.TestCase):
    """ABC rejection and ABC-PMC against the exact ABC and true posteriors."""

    def test_rejection_matches_the_exact_abc_posterior_and_converges(self) -> None:
        """Moments match pi_eps; pi_eps approaches the posterior as eps shrinks."""
        exact = [abc_exact_moments(eps) for eps in (1.0, 0.5, 0.2, 0.05)]
        gaps = [abs(variance - ABC_POSTERIOR_VARIANCE) for _, variance in exact]
        self.assertTrue(all(a > b for a, b in zip(gaps, gaps[1:], strict=False)))
        draws = 2000
        for tolerance in (0.5, 0.05):
            mean, variance = abc_exact_moments(tolerance)
            result = abc_rejection(
                abc_simulator,
                abc_prior,
                (ABC_OBSERVED,),
                draws=draws,
                tolerance=tolerance,
                seed=3,
            )
            with self.subTest(tolerance=tolerance):
                self.assertEqual(result.method, "rejection")
                self.assertEqual(result.tolerances, (tolerance,))
                self.assertTrue(all(rho <= tolerance for rho in result.distances))
                self.assertLess(
                    abs(result.mean()[0] - mean), 4.0 * math.sqrt(variance / draws)
                )
                # SE of a normal sample variance: sigma^2 sqrt(2 / n).
                self.assertLess(
                    abs(result.covariance()[0][0] - variance),
                    4.0 * variance * math.sqrt(2.0 / draws),
                )
                self.assertGreater(result.simulations, draws)
                self.assertAlmostEqual(
                    result.acceptance_rates[0], draws / result.simulations
                )
        # At the smallest tolerance the sample is close to the true posterior.
        self.assertLess(
            abs(result.mean()[0] - ABC_POSTERIOR_MEAN),
            4.0 * math.sqrt(ABC_POSTERIOR_VARIANCE / draws) + 0.01,
        )

    def test_quantile_mode(self) -> None:
        """ceil(draws / q) simulations; the tolerance is the largest kept distance."""
        result = abc_rejection(
            abc_simulator,
            abc_prior,
            (ABC_OBSERVED,),
            draws=300,
            quantile=0.1,
            seed=4,
        )
        self.assertEqual(result.simulations, 3000)
        self.assertEqual(len(result.parameters), 300)
        self.assertEqual(result.tolerance, max(result.distances))
        self.assertEqual(result.names, ("theta0",))
        self.assertAlmostEqual(result.effective_sample_size, 300.0)

    def test_population_monte_carlo(self) -> None:
        """Tolerances shrink; the final population matches pi_eps at the end."""
        draws = 300
        result = abc_smc(
            abc_simulator,
            abc_prior,
            abc_log_prior,
            (ABC_OBSERVED,),
            draws=draws,
            generations=5,
            seed=5,
            names=("theta",),
        )
        self.assertIsInstance(result, ABCResult)
        self.assertEqual(result.method, "population Monte Carlo")
        self.assertEqual(result.stopped_by, "generations")
        self.assertEqual(len(result.tolerances), 5)
        self.assertEqual(result.tolerances[0], math.inf)
        self.assertTrue(
            all(
                a > b
                for a, b in zip(result.tolerances, result.tolerances[1:], strict=False)
            )
        )
        self.assertEqual(result.simulations, sum(result.simulations_per_generation))
        self.assertAlmostEqual(math.fsum(result.weights), 1.0, places=12)
        self.assertTrue(all(rho <= result.tolerance for rho in result.distances))
        mean, variance = abc_exact_moments(result.tolerance)
        ess = result.effective_sample_size
        self.assertGreater(ess, 0.5 * draws)
        self.assertLess(abs(result.mean()[0] - mean), 4.0 * math.sqrt(variance / ess))
        self.assertLess(
            abs(result.covariance()[0][0] - variance),
            4.0 * variance * math.sqrt(2.0 / ess),
        )

    def test_population_stopping_rules(self) -> None:
        """final_tolerance and the budget end the run as documented."""
        reached = abc_smc(
            abc_simulator,
            abc_prior,
            abc_log_prior,
            (ABC_OBSERVED,),
            draws=100,
            generations=20,
            final_tolerance=0.3,
            seed=6,
        )
        self.assertEqual(reached.stopped_by, "tolerance")
        self.assertEqual(reached.tolerance, 0.3)
        budget = abc_smc(
            abc_simulator,
            abc_prior,
            abc_log_prior,
            (ABC_OBSERVED,),
            draws=100,
            generations=20,
            max_simulations=400,
            seed=6,
        )
        self.assertEqual(budget.stopped_by, "budget")
        self.assertLessEqual(budget.simulations, 400)
        self.assertEqual(len(budget.parameters), 100)
        self.assertLess(sum(budget.simulations_per_generation), 400 + 1)

    def test_reproducibility_distance_and_validation(self) -> None:
        """Seeds reproduce draws; invalid options and outputs raise."""
        first = abc_rejection(
            abc_simulator, abc_prior, (ABC_OBSERVED,), draws=20, quantile=0.5, seed=9
        )
        second = abc_rejection(
            abc_simulator, abc_prior, (ABC_OBSERVED,), draws=20, quantile=0.5, seed=9
        )
        self.assertEqual(first.parameters, second.parameters)
        self.assertEqual(euclidean_distance((3.0, 0.0), (0.0, 4.0)), 5.0)
        with self.assertRaises(ValueError):
            euclidean_distance((1.0,), (1.0, 2.0))
        observed = (ABC_OBSERVED,)
        with self.assertRaises(ValueError):
            abc_rejection(abc_simulator, abc_prior, observed, draws=10)
        with self.assertRaises(ValueError):
            abc_rejection(
                abc_simulator, abc_prior, observed, draws=10, tolerance=1, quantile=0.5
            )
        with self.assertRaises(ValueError):
            abc_rejection(abc_simulator, abc_prior, observed, draws=10, quantile=1.5)
        with self.assertRaises(ValueError):
            abc_rejection(abc_simulator, abc_prior, observed, draws=10, tolerance=-1)
        with self.assertRaises(ValueError):
            abc_rejection(
                abc_simulator,
                abc_prior,
                observed,
                draws=10,
                tolerance=1e-9,
                max_simulations=50,
                seed=1,
            )
        with self.assertRaises(ValueError):
            abc_rejection(
                lambda theta, g: (1.0, 2.0), abc_prior, observed, draws=2, quantile=1.0
            )
        with self.assertRaises(ValueError):
            abc_rejection(
                abc_simulator,
                abc_prior,
                observed,
                lambda a, b: -1.0,
                draws=2,
                quantile=1.0,
            )
        with self.assertRaises(TypeError):
            abc_rejection(None, abc_prior, observed, draws=2, quantile=1.0)
        with self.assertRaises(ValueError):
            abc_smc(abc_simulator, abc_prior, abc_log_prior, observed, draws=1)
        with self.assertRaises(ValueError):
            abc_smc(
                abc_simulator, abc_prior, abc_log_prior, observed, draws=5, quantile=1.0
            )
        with self.assertRaises(ValueError):
            abc_smc(
                abc_simulator,
                abc_prior,
                lambda theta: -math.inf,
                observed,
                draws=5,
                seed=1,
            )
        with self.assertRaises(TypeError):
            abc_smc(abc_simulator, abc_prior, None, observed, draws=5)


class CalibrationWiringTests(unittest.TestCase):
    """pcn and smc methods and the log evidence of a CalibrationProblem."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the known-noise line problem and its exact posterior."""
        cls.problem = line_problem(NoiseModel("y", standard_deviation=0.3))
        cls.exact = calibrate_linear_gaussian(
            LINE_X, line_data(), observation_variance=0.09, prior=LINE_PRIOR
        ).posterior

    def test_pcn_matches_the_conjugate_posterior(self) -> None:
        """Means within 4 MCSE and SDs within 20 % of the closed form."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            posterior = self.problem.sample_posterior(
                method="pcn", draws=1500, warmup=500, chains=4, seed=11
            )
        self.assertEqual(posterior.mcmc.algorithm, "preconditioned Crank-Nicolson")
        self.assertEqual(posterior.standard_space.names, ("u[a]", "u[b]"))
        self.assertIsNotNone(posterior.laplace)
        self.assertIsNone(posterior.smc)
        for index, summary in enumerate(posterior.summary()):
            scale = math.sqrt(self.exact.covariance[index][index])
            with self.subTest(parameter=summary.name):
                self.assertLess(
                    abs(summary.mean - self.exact.mean[index]), 4.0 * summary.mcse_mean
                )
                self.assertAlmostEqual(
                    summary.standard_deviation, scale, delta=0.2 * scale
                )

    def test_smc_matches_the_conjugate_posterior(self) -> None:
        """SMC particles reproduce the closed-form mean and covariance."""
        posterior = self.problem.sample_posterior(
            method="smc", particles=400, mutation_steps=5, seed=12
        )
        self.assertIsNotNone(posterior.smc)
        self.assertIsNone(posterior.laplace)
        self.assertEqual(posterior.mcmc.draws_per_chain, 400)
        self.assertEqual(posterior.mcmc.names, ("a", "b"))
        mean = posterior.mcmc.mean()
        covariance = posterior.mcmc.covariance()
        for index in range(2):
            scale = math.sqrt(self.exact.covariance[index][index])
            with self.subTest(index=index):
                self.assertLess(abs(mean[index] - self.exact.mean[index]), 0.3 * scale)
                self.assertAlmostEqual(
                    math.sqrt(covariance[index][index]), scale, delta=0.2 * scale
                )

    def test_log_evidence_matches_the_exact_marginal_likelihood(self) -> None:
        """The log evidence is within 4 SEs and 0.4 of the Gaussian marginal."""
        exact = line_marginal_log_likelihood(0.3)
        estimate = self.problem.log_evidence(particles=400, mutation_steps=5, seed=5)
        self.assertEqual(estimate.names, ("u[a]", "u[b]"))
        self.assertLess(
            abs(estimate.log_evidence - exact),
            4.0 * estimate.log_evidence_standard_error,
        )
        self.assertLess(abs(estimate.log_evidence - exact), 0.4)

    def test_unknown_noise_uses_standardized_noise_coordinates(self) -> None:
        """Noise enters through w = Phi^-1(F(sigma)); the evidence integrates sigma."""
        prior = LogNormal(math.log(0.3), 0.5)
        problem = line_problem(NoiseModel("y", standard_deviation_prior=prior))
        # Likelihood consistency between (u, w) and (theta, log sigma).
        for u0, u1, w in ((0.2, -0.4, 0.3), (-1.0, 0.5, -1.2)):
            theta = problem.model.inputs.from_standard_normal((u0, u1))
            sigma = prior.quantile(STANDARD_NORMAL.cdf(w))
            value = problem.standard_log_likelihood((u0, u1, w))
            reference = problem.log_likelihood(theta + (math.log(sigma),))
            self.assertAlmostEqual(value, reference, delta=1e-9 * abs(reference))
            sampling = problem._whitened_to_sampling((u0, u1, w))
            back = problem._sampling_to_whitened(sampling)
            self.assertAlmostEqual(back[2], w, delta=1e-9)
        with self.assertRaises(ValueError):
            problem.standard_log_likelihood((0.0, 0.0))
        # Exact evidence by quadrature over log sigma of the Gaussian marginal.
        step = 0.01
        grid = [math.log(0.3) + 0.5 * (-8.0 + step * i) for i in range(1601)]
        terms = [
            line_marginal_log_likelihood(math.exp(eta))
            + prior.log_pdf(math.exp(eta))
            + eta
            for eta in grid
        ]
        top = max(terms)
        exact = top + math.log(0.5 * step * math.fsum(math.exp(t - top) for t in terms))
        estimate = problem.log_evidence(particles=500, mutation_steps=5, seed=7)
        self.assertEqual(estimate.names, ("u[a]", "u[b]", "w[y]"))
        self.assertLess(abs(estimate.log_evidence - exact), 0.4)
        self.assertLess(
            abs(estimate.log_evidence - exact),
            4.0 * estimate.log_evidence_standard_error,
        )
        posterior = problem.sample_posterior(
            method="smc", particles=300, mutation_steps=5, seed=8
        )
        self.assertEqual(posterior.mcmc.names, ("a", "b", "sigma[y]"))
        self.assertEqual(
            posterior.standard_space.names, ("u[a]", "u[b]", "log_sigma[y]")
        )
        sample = posterior.standard_space.chains[0].samples[0]
        physical = posterior.mcmc.chains[0].samples[0]
        self.assertAlmostEqual(math.exp(sample[2]), physical[2], delta=1e-12)

    def test_unknown_method_and_default_unchanged(self) -> None:
        """An unknown method raises; the default path still runs."""
        with self.assertRaises(ValueError):
            self.problem.sample_posterior(method="gibbs")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            default = self.problem.sample_posterior(
                draws=30, warmup=30, chains=2, seed=1
            )
        self.assertEqual(default.mcmc.algorithm, "adaptive Metropolis")
        self.assertIsNone(default.smc)


if __name__ == "__main__":
    unittest.main()
