"""Tests for Hamiltonian Monte Carlo, NUTS, and Gaussian ADVI."""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import Normal
from probstat.inference import (
    HamiltonianResult,
    adaptation_windows,
    advi,
    condition_linear_gaussian,
    effective_sample_size,
    hamiltonian_energy,
    hamiltonian_monte_carlo,
    leapfrog,
    no_u_turn_sampler,
    random_walk_metropolis,
)
from probstat.inverse import CalibrationProblem, Experiment, NoiseModel
from probstat.linalg import inverse_spd, log_det_spd
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.multivariate import MultivariateNormal
from probstat.uq import QuantityOfInterest

TARGET_MEAN = (1.0, -2.0)
TARGET_COVARIANCE = ((1.0, 0.8), (0.8, 2.0))
TARGET_PRECISION = inverse_spd(TARGET_COVARIANCE)
LOG_NORMALIZER = math.log(2.0 * math.pi) + 0.5 * log_det_spd(TARGET_COVARIANCE)


def correlated_gaussian(x: tuple[float, ...]) -> float:
    """Return the unnormalized log-density of a correlated 2-D Gaussian."""
    d0, d1 = x[0] - TARGET_MEAN[0], x[1] - TARGET_MEAN[1]
    p = TARGET_PRECISION
    return -0.5 * (
        d0 * (p[0][0] * d0 + p[0][1] * d1) + d1 * (p[1][0] * d0 + p[1][1] * d1)
    )


def correlated_gradient(x: tuple[float, ...]) -> tuple[float, float]:
    """Return the gradient of :func:`correlated_gaussian`."""
    d0, d1 = x[0] - TARGET_MEAN[0], x[1] - TARGET_MEAN[1]
    p = TARGET_PRECISION
    return (-(p[0][0] * d0 + p[0][1] * d1), -(p[1][0] * d0 + p[1][1] * d1))


def standard_normal(x: tuple[float, ...]) -> float:
    """Return the unnormalized standard-normal log-density in any dimension."""
    return -0.5 * math.fsum(v * v for v in x)


def standard_gradient(x: tuple[float, ...]) -> tuple[float, ...]:
    """Return the gradient of :func:`standard_normal`."""
    return tuple(-v for v in x)


def funnel(x: tuple[float, ...]) -> float:
    """Return Neal's funnel log-density with ``v ~ N(0, 9)``, ``z | v ~ N(0, e^v)``."""
    v = x[0]
    squares = math.fsum(z * z for z in x[1:])
    return -v * v / 18.0 - 0.5 * squares * math.exp(-v) - 0.5 * (len(x) - 1) * v


def funnel_gradient(x: tuple[float, ...]) -> tuple[float, ...]:
    """Return the gradient of :func:`funnel`."""
    v = x[0]
    squares = math.fsum(z * z for z in x[1:])
    head = -v / 9.0 + 0.5 * squares * math.exp(-v) - 0.5 * (len(x) - 1)
    return (head,) + tuple(-z * math.exp(-v) for z in x[1:])


def line_model(values: dict[str, float]) -> dict[str, float]:
    """Evaluate a straight line ``a + b x``."""
    return {"y": values["a"] + values["b"] * values["x"]}


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
        Experiment({"y": y}, {"x": x}) for x, y in zip(xs, ys, strict=True)
    )
    problem = CalibrationProblem(
        model, experiments, (NoiseModel("y", standard_deviation=0.3),)
    )
    return problem, xs, ys


def max_energy_error(step: float, steps: int, inverse_metric: object) -> float:
    """Return ``max_t |H(z_t) - H(z_0)|`` along a leapfrog trajectory."""
    metric = inverse_metric  # type: ignore[assignment]
    position: tuple[float, ...] = (1.5, -0.5)
    momentum: tuple[float, ...] = (0.3, -1.0)
    start = hamiltonian_energy(
        correlated_gaussian, position, momentum, inverse_metric=metric
    )
    worst = 0.0
    for _ in range(steps):
        position, momentum = leapfrog(
            correlated_gradient,
            position,
            momentum,
            step_size=step,
            steps=1,
            inverse_metric=metric,
        )
        energy = hamiltonian_energy(
            correlated_gaussian, position, momentum, inverse_metric=metric
        )
        worst = max(worst, abs(energy - start))
    return worst


class LeapfrogTests(unittest.TestCase):
    """Verify the integrator's order, reversibility, and the energy function."""

    def test_energy_error_is_second_order_in_step_size(self) -> None:
        """Halving eps over a fixed time divides the energy error by about 4."""
        for metric in (None, ((1.0, 0.3), (0.3, 0.5))):
            with self.subTest(metric=metric):
                coarse = max_energy_error(0.1, 30, metric)
                fine = max_energy_error(0.05, 60, metric)
                finer = max_energy_error(0.025, 120, metric)
                self.assertLess(coarse, 0.05)
                self.assertAlmostEqual(coarse / fine, 4.0, delta=0.6)
                self.assertAlmostEqual(fine / finer, 4.0, delta=0.6)

    def test_leapfrog_is_time_reversible(self) -> None:
        """Integrating back with negated momentum returns to the start."""
        x0, p0 = (0.4, 1.1), (-0.7, 0.2)
        x1, p1 = leapfrog(correlated_gradient, x0, p0, step_size=0.2, steps=25)
        x2, p2 = leapfrog(
            correlated_gradient, x1, tuple(-v for v in p1), step_size=0.2, steps=25
        )
        for a, b in zip(x2, x0, strict=True):
            self.assertAlmostEqual(a, b, places=10)
        for a, b in zip(p2, p0, strict=True):
            self.assertAlmostEqual(-a, b, places=10)

    def test_hamiltonian_energy_value(self) -> None:
        """H equals -log p plus the metric kinetic energy."""
        value = hamiltonian_energy(
            standard_normal,
            (1.0, 2.0),
            (2.0, 1.0),
            inverse_metric=((0.5, 0.0), (0.0, 2.0)),
        )
        self.assertAlmostEqual(value, 2.5 + 0.5 * (0.5 * 4.0 + 2.0 * 1.0))
        self.assertEqual(
            hamiltonian_energy(lambda x: -math.inf, (0.0,), (0.0,)), math.inf
        )
        with self.assertRaises(ValueError):
            hamiltonian_energy(standard_normal, (0.0, 1.0), (0.0,))
        with self.assertRaises(ValueError):
            leapfrog(
                standard_gradient,
                (0.0, 1.0),
                (0.0, 1.0),
                step_size=0.1,
                steps=1,
                inverse_metric=((1.0, 2.0), (2.0, 1.0)),
            )

    def test_adaptation_windows_follow_documented_schedule(self) -> None:
        """Windows double between a 75-step and a 50-step fast buffer."""
        self.assertEqual(adaptation_windows(1000), (75, 50, (100, 150, 250, 450, 950)))
        self.assertEqual(adaptation_windows(150), (75, 50, (100,)))
        self.assertEqual(adaptation_windows(100), (15, 10, (90,)))
        self.assertEqual(adaptation_windows(10), (10, 0, ()))
        self.assertEqual(adaptation_windows(0), (0, 0, ()))


class HamiltonianSamplerTests(unittest.TestCase):
    """Verify HMC and NUTS on targets with known answers."""

    def assert_recovers_target(self, result: HamiltonianResult) -> None:
        """Assert mean within 4 MCSE, covariance within MC error, and R-hat."""
        for index, summary in enumerate(result.summary()):
            with self.subTest(algorithm=result.algorithm, coordinate=index):
                self.assertLess(
                    abs(summary.mean - TARGET_MEAN[index]), 4.0 * summary.mcse_mean
                )
                self.assertLess(summary.rank_normalized_rhat, 1.05)
                self.assertGreater(summary.effective_sample_size, 300.0)
        covariance = result.covariance()
        for i in range(2):
            for j in range(2):
                scale = math.sqrt(TARGET_COVARIANCE[i][i] * TARGET_COVARIANCE[j][j])
                self.assertAlmostEqual(
                    covariance[i][j], TARGET_COVARIANCE[i][j], delta=0.2 * scale
                )

    def test_nuts_recovers_correlated_gaussian(self) -> None:
        """Diagonal and dense metrics both recover mean and covariance."""
        for metric in ("diag", "dense"):
            result = no_u_turn_sampler(
                correlated_gaussian,
                (0.0, 0.0),
                gradient=correlated_gradient,
                draws=400,
                warmup=300,
                chains=4,
                metric=metric,
                seed=1,
            )
            self.assert_recovers_target(result)
            self.assertEqual(result.divergences, 0)
            for record in result.statistics:
                self.assertGreater(record.mean_acceptance, 0.7)
                self.assertEqual(sum(record.tree_depth_counts().values()), 400)
                self.assertEqual(record.max_tree_depth_hits, 0)
                self.assertGreater(record.e_bfmi, 0.3)
            if metric == "dense":
                adapted = result.statistics[0].inverse_metric
                self.assertAlmostEqual(adapted[0][1] / adapted[1][1], 0.4, delta=0.2)

    def test_hmc_recovers_correlated_gaussian(self) -> None:
        """Static HMC with step-size jitter recovers mean and covariance."""
        result = hamiltonian_monte_carlo(
            correlated_gaussian,
            (0.0, 0.0),
            gradient=correlated_gradient,
            draws=400,
            warmup=300,
            chains=4,
            leapfrog_steps=6,
            step_size_jitter=0.3,
            seed=2,
        )
        self.assert_recovers_target(result)
        self.assertEqual(result.divergences, 0)
        self.assertTrue(math.isnan(result.statistics[0].mean_tree_depth))
        self.assertEqual(result.statistics[0].mean_leapfrog_steps, 6.0)

    def test_tiny_step_has_acceptance_near_one(self) -> None:
        """Leapfrog with tiny eps nearly conserves H, so almost all is accepted."""
        common = {
            "gradient": standard_gradient,
            "warmup": 0,
            "chains": 2,
            "step_size": 0.01,
            "adapt_step_size": False,
            "seed": 3,
        }
        static = hamiltonian_monte_carlo(
            standard_normal, (0.5, -0.5), draws=200, **common  # type: ignore[arg-type]
        )
        nuts = no_u_turn_sampler(
            standard_normal,
            (0.5, -0.5),
            draws=100,
            max_tree_depth=6,
            **common,  # type: ignore[arg-type]
        )
        for result in (static, nuts):
            with self.subTest(algorithm=result.algorithm):
                self.assertGreater(result.mcmc.acceptance_rate, 0.999)
                self.assertEqual(result.divergences, 0)

    def test_nuts_beats_random_walk_in_ess_per_gradient(self) -> None:
        """On a 10-D iid Gaussian NUTS gives more ESS per evaluation than RWM."""
        dimension = 10
        start = (0.5,) * dimension
        nuts = no_u_turn_sampler(
            standard_normal,
            start,
            gradient=standard_gradient,
            draws=300,
            warmup=200,
            chains=2,
            seed=5,
        )
        nuts_ess = min(
            effective_sample_size(nuts.mcmc.chain_columns(i)) for i in range(dimension)
        )
        budget = nuts.gradient_evaluations
        scale = 2.38**2 / dimension
        rwm = random_walk_metropolis(
            standard_normal,
            start,
            proposal_covariance=tuple(
                tuple(scale if i == j else 0.0 for j in range(dimension))
                for i in range(dimension)
            ),
            draws=budget // 2 - 1,
            chains=2,
            seed=5,
        )
        self.assertLessEqual(rwm.evaluations, budget)
        rwm_ess = max(
            effective_sample_size(rwm.chain_columns(i)) for i in range(dimension)
        )
        self.assertGreater(nuts_ess / budget, 2.0 * rwm_ess / rwm.evaluations)

    def test_divergences_are_flagged_on_funnel(self) -> None:
        """A step too large for the funnel neck diverges; a Gaussian does not."""
        for sampler in (hamiltonian_monte_carlo, no_u_turn_sampler):
            result = sampler(
                funnel,
                (0.0, 1.0, 1.0),
                gradient=funnel_gradient,
                draws=200,
                warmup=0,
                chains=2,
                step_size=1.0,
                adapt_step_size=False,
                adapt_metric=False,
                seed=1,
            )
            with self.subTest(algorithm=result.algorithm):
                per_chain = result.divergences_per_chain
                self.assertTrue(all(count > 0 for count in per_chain))
                self.assertEqual(result.divergences, sum(per_chain))
                for record in result.statistics:
                    indices = record.divergent_iterations
                    self.assertTrue(all(0 <= i < 200 for i in indices))
        calm = no_u_turn_sampler(
            standard_normal,
            (0.0, 1.0, 1.0),
            gradient=standard_gradient,
            draws=200,
            warmup=100,
            chains=2,
            seed=1,
        )
        self.assertEqual(calm.divergences, 0)

    def test_finite_difference_gradient_cost_and_accuracy(self) -> None:
        """Without a gradient each gradient costs 2d + 1 density evaluations."""
        result = no_u_turn_sampler(
            correlated_gaussian, (0.0, 0.0), draws=200, warmup=150, chains=2, seed=4
        )
        for record in result.statistics:
            self.assertEqual(
                record.log_density_evaluations, 5 * record.gradient_evaluations
            )
        mean = result.mean()
        summaries = result.summary()
        for index in range(2):
            self.assertLess(
                abs(mean[index] - TARGET_MEAN[index]), 4.0 * summaries[index].mcse_mean
            )

    def test_seeded_runs_are_reproducible(self) -> None:
        """The same seed gives identical draws; a different seed does not."""
        first = no_u_turn_sampler(
            correlated_gaussian, (0.0, 0.0), gradient=correlated_gradient,
            draws=20, warmup=30, chains=2, seed=9,
        )
        second = no_u_turn_sampler(
            correlated_gaussian, (0.0, 0.0), gradient=correlated_gradient,
            draws=20, warmup=30, chains=2, seed=9,
        )
        other = no_u_turn_sampler(
            correlated_gaussian, (0.0, 0.0), gradient=correlated_gradient,
            draws=20, warmup=30, chains=2, seed=10,
        )
        self.assertEqual(first.mcmc.pooled(), second.mcmc.pooled())
        self.assertNotEqual(first.mcmc.pooled(), other.mcmc.pooled())

    def test_invalid_arguments_raise(self) -> None:
        """Bad settings and unusable starts are rejected."""
        cases = (
            {"draws": 0},
            {"draws": 5, "chains": 0},
            {"draws": 5, "metric": "full"},
            {"draws": 5, "step_size": -1.0},
            {"draws": 5, "adapt_step_size": False},
            {"draws": 5, "target_acceptance": 1.0},
            {"draws": 5, "max_tree_depth": 0},
            {"draws": 5, "max_energy_error": 0.0},
            {"draws": 5, "initial_inverse_metric": ((1.0, 2.0), (2.0, 1.0))},
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                no_u_turn_sampler(standard_normal, (0.0, 0.0), **options)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            hamiltonian_monte_carlo(standard_normal, (0.0,), draws=5, leapfrog_steps=0)
        with self.assertRaises(ValueError):
            hamiltonian_monte_carlo(
                standard_normal, (0.0,), draws=5, step_size_jitter=1.0
            )
        with self.assertRaises(ValueError):
            no_u_turn_sampler(lambda x: -math.inf, (0.0,), draws=5)
        with self.assertRaises(TypeError):
            no_u_turn_sampler(None, (0.0,), draws=5)  # type: ignore[arg-type]


class VariationalInferenceTests(unittest.TestCase):
    """Verify ADVI against Gaussian targets with closed-form optima."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fit the full-rank and mean-field families once."""
        options = {
            "gradient": correlated_gradient,
            "gradient_samples": 4,
            "learning_rate": 0.02,
            "tolerance": 1e-4,
            "draws": 2000,
            "seed": 2,
        }
        cls.full = advi(correlated_gaussian, (0.0, 0.0), family="full-rank", **options)  # type: ignore[arg-type]
        cls.mean_field = advi(
            correlated_gaussian, (0.0, 0.0), family="mean-field", **options  # type: ignore[arg-type]
        )

    def test_full_rank_recovers_mean_and_covariance(self) -> None:
        """The full-rank optimum is the target itself."""
        fit = self.full
        self.assertTrue(fit.converged)
        for i in range(2):
            self.assertAlmostEqual(fit.mean[i], TARGET_MEAN[i], delta=0.05)
            for j in range(2):
                self.assertAlmostEqual(
                    fit.covariance[i][j], TARGET_COVARIANCE[i][j], delta=0.08
                )

    def test_mean_field_matches_inverse_precision_diagonal(self) -> None:
        """Mean-field recovers the mean and variances 1 / Lambda_ii (too small)."""
        fit = self.mean_field
        self.assertTrue(fit.converged)
        for i in range(2):
            self.assertAlmostEqual(fit.mean[i], TARGET_MEAN[i], delta=0.05)
            expected = 1.0 / TARGET_PRECISION[i][i]
            self.assertAlmostEqual(fit.covariance[i][i], expected, delta=0.06)
            self.assertLess(fit.covariance[i][i], TARGET_COVARIANCE[i][i])
        self.assertEqual(fit.covariance[0][1], 0.0)

    def test_slow_ridge_climb_is_not_reported_as_converged(self) -> None:
        """A run still climbing a flat ridge must not claim convergence."""
        mean = (10.0, -5.0)
        precision = inverse_spd(((25.0, 4.9), (4.9, 1.0)))

        def gradient(x: tuple[float, ...]) -> tuple[float, ...]:
            """Return the gradient of the ridge-shaped Gaussian."""
            d = (x[0] - mean[0], x[1] - mean[1])
            return tuple(
                -(precision[i][0] * d[0] + precision[i][1] * d[1]) for i in range(2)
            )

        def log_density(x: tuple[float, ...]) -> float:
            """Return the log-density of the ridge-shaped Gaussian."""
            g = gradient(x)
            return 0.5 * ((x[0] - mean[0]) * g[0] + (x[1] - mean[1]) * g[1])

        fit = advi(log_density, (0.0, 0.0), gradient=gradient, seed=0, draws=0)
        if fit.converged:
            self.assertAlmostEqual(fit.mean[0], mean[0], delta=0.5)
        else:
            self.assertEqual(fit.iterations, 10000)

    def test_elbo_increases_toward_log_normalizer(self) -> None:
        """The ELBO trace rises and ends near log Z, the bound's maximum."""
        for fit in (self.full, self.mean_field):
            values = [value for _, value in fit.elbo_trace]
            self.assertGreater(values[-1], values[0] + 0.5)
            self.assertLess(max(values[:3]), max(values))
        self.assertAlmostEqual(self.full.elbo, LOG_NORMALIZER, delta=0.1)
        gap = LOG_NORMALIZER - self.mean_field.elbo
        kl_mean_field = 0.5 * (
            math.log(TARGET_PRECISION[0][0] * TARGET_PRECISION[1][1])
            - log_det_spd(TARGET_PRECISION)
        )
        self.assertAlmostEqual(gap, kl_mean_field, delta=0.1)

    def test_draws_density_and_mcmc_wrapper(self) -> None:
        """Draws follow q, log_pdf is the Gaussian density, and summaries work."""
        fit = self.full
        reference = MultivariateNormal(fit.mean, fit.covariance)
        for point in ((0.0, 0.0), (1.5, -3.0)):
            self.assertAlmostEqual(
                fit.log_pdf(point), reference.log_pdf(point), places=10
            )
        wrapped = fit.as_mcmc_result()
        self.assertEqual(wrapped.draws_per_chain, 2000)
        for index, summary in enumerate(wrapped.summary()):
            error = abs(summary.mean - fit.mean[index])
            self.assertLess(error, 4.0 * summary.mcse_mean)
        self.assertEqual(len(fit.sample(5, seed=1)), 5)
        self.assertEqual(len(fit.log_density), 2000)

    def test_finite_difference_gradient_path(self) -> None:
        """ADVI also works without an analytic gradient."""
        fit = advi(
            correlated_gaussian, (0.0, 0.0), family="full-rank", seed=4, draws=0
        )
        for i in range(2):
            self.assertAlmostEqual(fit.mean[i], TARGET_MEAN[i], delta=0.2)
        self.assertGreater(fit.log_density_evaluations, 4 * fit.gradient_evaluations)

    def test_invalid_arguments_raise(self) -> None:
        """Bad settings are rejected."""
        for options in (
            {"family": "diagonal"},
            {"learning_rate": 0.0},
            {"gradient_samples": 0},
            {"beta1": 1.0},
            {"tolerance": 0.0},
            {"names": ("a",)},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                advi(correlated_gaussian, (0.0, 0.0), **options)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            advi(correlated_gaussian, ())
        with self.assertRaises(ValueError):
            advi(lambda x: -math.inf, (0.0,), gradient=lambda x: (0.0,), seed=1)


class CalibrationMethodTests(unittest.TestCase):
    """Verify the optional HMC/NUTS path of nonlinear calibration."""

    def test_default_method_is_unchanged(self) -> None:
        """The default reproduces adaptive Metropolis exactly."""
        problem, _, _ = line_problem()
        default = problem.sample_posterior(draws=50, warmup=50, chains=2, seed=3)
        explicit = problem.sample_posterior(
            draws=50, warmup=50, chains=2, seed=3, method="adaptive-metropolis"
        )
        self.assertEqual(default.mcmc.pooled(), explicit.mcmc.pooled())
        self.assertIsNone(default.sampler)
        self.assertEqual(default.mcmc.algorithm, "adaptive Metropolis")

    def test_nuts_matches_conjugate_posterior(self) -> None:
        """NUTS in standard space agrees with the closed-form posterior."""
        problem, xs, ys = line_problem()
        exact = condition_linear_gaussian(
            MultivariateNormal((0.0, 1.0), ((4.0, 0.0), (0.0, 1.0))),
            [(1.0, x) for x in xs],
            ys,
            observation_variance=0.09,
        )
        posterior = problem.sample_posterior(
            draws=300, warmup=200, chains=2, seed=11, method="nuts"
        )
        self.assertIsNotNone(posterior.sampler)
        assert posterior.sampler is not None
        self.assertEqual(posterior.sampler.divergences, 0)
        self.assertEqual(posterior.mcmc.algorithm, "NUTS")
        for index, summary in enumerate(posterior.summary()):
            with self.subTest(parameter=summary.name):
                self.assertLess(
                    abs(summary.mean - exact.mean[index]), 4.0 * summary.mcse_mean
                )
                self.assertAlmostEqual(
                    summary.standard_deviation,
                    math.sqrt(exact.covariance[index][index]),
                    delta=0.2 * math.sqrt(exact.covariance[index][index]),
                )

    def test_hmc_method_and_invalid_method(self) -> None:
        """Static HMC runs through the same path; unknown methods raise."""
        problem, _, _ = line_problem()
        posterior = problem.sample_posterior(
            draws=30, warmup=30, chains=2, seed=1, method="hmc"
        )
        self.assertEqual(posterior.mcmc.algorithm, "Hamiltonian Monte Carlo")
        with self.assertRaises(ValueError):
            problem.sample_posterior(draws=10, warmup=10, method="gibbs")


if __name__ == "__main__":
    unittest.main()
