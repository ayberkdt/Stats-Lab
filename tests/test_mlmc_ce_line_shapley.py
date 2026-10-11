"""Tests for multilevel Monte Carlo, CE-IS, line sampling, and Shapley effects."""

from __future__ import annotations

import functools
import math
import unittest

from probstat.distributions import LogNormal, Normal
from probstat.foundations.random_streams import RandomStream
from probstat.modeling import (
    JointInput,
    ProbabilisticModel,
    RandomInput,
    gauss_hermite,
    standard_normal_cdf,
)
from probstat.monte_carlo import (
    adaptive_multilevel_monte_carlo,
    convergence_rates,
    multilevel_monte_carlo,
    optimal_level_allocation,
)
from probstat.reliability import (
    AVAILABLE_METHODS,
    METHODS,
    ModelLimitState,
    compare_reliability_methods,
    cross_entropy_importance_sampling,
    form_hlrf,
    line_sampling,
)
from probstat.sensitivity import (
    exact_shapley_values,
    linear_gaussian_shapley,
    shapley_effects,
)
from probstat.uq import QuantityOfInterest

Y = (QuantityOfInterest("y", "response"),)


class GeometricBrownianMotion:
    """Coupled Euler-Maruyama sampler of ``S_T`` for ``dS = mu S dt + sigma S dW``.

    Level ``l`` uses ``2^l`` steps; the coarse path reuses the summed Brownian
    increments of each fine step pair. ``steps`` counts every Euler step taken,
    which must equal the MLMC cost with ``path_cost(l) = 2^l``.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        """Store the drift and volatility (``S_0 = 1``, ``T = 1``)."""
        self.mu = mu
        self.sigma = sigma
        self.steps = 0

    def __call__(
        self, level: int, count: int, stream: RandomStream
    ) -> tuple[list[float], list[float]]:
        """Return ``count`` coupled fine and coarse terminal values."""
        generator = stream.generator()
        fine_steps = 2**level
        h = 1.0 / fine_steps
        root_h = math.sqrt(h)
        fine: list[float] = []
        coarse: list[float] = []
        for _ in range(count):
            if level == 0:
                value = 1.0 + self.mu + self.sigma * generator.gauss(0.0, 1.0)
                fine.append(value)
                coarse.append(0.0)
                self.steps += 1
                continue
            s_fine = s_coarse = 1.0
            for _pair in range(fine_steps // 2):
                d1 = root_h * generator.gauss(0.0, 1.0)
                d2 = root_h * generator.gauss(0.0, 1.0)
                s_fine *= 1.0 + self.mu * h + self.sigma * d1
                s_fine *= 1.0 + self.mu * h + self.sigma * d2
                s_coarse *= 1.0 + 2.0 * self.mu * h + self.sigma * (d1 + d2)
            fine.append(s_fine)
            coarse.append(s_coarse)
            self.steps += fine_steps + fine_steps // 2
        return fine, coarse


def standard_pair() -> JointInput:
    """Return two independent standard normal inputs u1 and u2."""
    return JointInput.independent(
        RandomInput("u1", Normal(0.0, 1.0)), RandomInput("u2", Normal(0.0, 1.0))
    )


def linear_state(beta: float = 4.0) -> ModelLimitState:
    """Return ``G = beta - (u1 + u2)/sqrt 2`` with ``P_f = Phi(-beta)``."""
    model = ProbabilisticModel(
        standard_pair(),
        lambda p: {"y": beta - (p["u1"] + p["u2"]) / math.sqrt(2.0)},
        Y,
    )
    return ModelLimitState(model, "y", 0.0, "below")


def parabolic_state(kappa: float, beta: float = 3.0) -> ModelLimitState:
    """Return ``G = beta - u2 + kappa u1^2 / 2``."""
    model = ProbabilisticModel(
        standard_pair(),
        lambda p: {"y": beta - p["u2"] + 0.5 * kappa * p["u1"] ** 2},
        Y,
    )
    return ModelLimitState(model, "y", 0.0, "below")


@functools.cache
def parabolic_exact(kappa: float, beta: float = 3.0) -> float:
    """Return ``E[Phi(-(beta + kappa U^2 / 2))]`` by Gauss-Hermite quadrature."""
    nodes, weights = gauss_hermite(60)
    return math.fsum(
        w * standard_normal_cdf(-(beta + 0.5 * kappa * z * z))
        for z, w in zip(nodes, weights, strict=True)
    )


def lognormal_state() -> tuple[ModelLimitState, float]:
    """Return the correlated lognormal ``R - S`` limit state and exact ``P_f``."""
    s_r, s_s, rho = 0.15, 0.25, 0.4
    inputs = JointInput(
        (
            RandomInput("r", LogNormal(1.2, s_r)),
            RandomInput("s", LogNormal(0.3, s_s)),
        ),
        ((1.0, rho), (rho, 1.0)),
    )
    model = ProbabilisticModel(inputs, lambda p: {"y": p["r"] - p["s"]}, Y)
    beta = 0.9 / math.sqrt(s_r**2 + s_s**2 - 2.0 * rho * s_r * s_s)
    return ModelLimitState(model, "y", 0.0, "below"), standard_normal_cdf(-beta)


class MultilevelMonteCarloTests(unittest.TestCase):
    """Allocation, rates, cost accounting, and accuracy of MLMC."""

    def test_optimal_allocation_meets_target_variance(self) -> None:
        """``N_l = sqrt(V_l/C_l) sum sqrt(V_k C_k) / v`` meets the constraint."""
        sizes = optimal_level_allocation((4.0, 1.0), (1.0, 4.0), 0.01)
        self.assertEqual(sizes, (800, 200))
        self.assertLessEqual(4.0 / sizes[0] + 1.0 / sizes[1], 0.01 + 1e-15)
        self.assertEqual(optimal_level_allocation((0.0, 1.0), (1.0, 1.0), 1.0)[0], 0)
        with self.assertRaises(ValueError):
            optimal_level_allocation((1.0,), (1.0, 2.0), 0.1)
        with self.assertRaises(ValueError):
            optimal_level_allocation((-1.0,), (1.0,), 0.1)
        with self.assertRaises(ValueError):
            optimal_level_allocation((1.0,), (0.0,), 0.1)

    def test_convergence_rates_recover_exact_powers(self) -> None:
        """Regression returns alpha, beta, gamma of exact power laws."""
        levels = range(6)
        alpha, beta, gamma = convergence_rates(
            [3.0 * 2.0**-level for level in levels],
            [5.0 * 4.0**-level for level in levels],
            [2.0**level for level in levels],
        )
        self.assertAlmostEqual(alpha, 1.0, places=12)
        self.assertAlmostEqual(beta, 2.0, places=12)
        self.assertAlmostEqual(gamma, 1.0, places=12)
        self.assertTrue(
            math.isnan(convergence_rates([1.0, 0.5], [1.0, 1.0], [1, 2])[0])
        )

    def test_fixed_allocation_telescopes_and_counts_cost(self) -> None:
        """The estimate is the sum of level means; cost equals Euler steps."""
        sampler = GeometricBrownianMotion(0.05, 0.2)
        result = multilevel_monte_carlo(sampler, (400, 100, 50), seed=5)
        self.assertAlmostEqual(
            result.estimate, math.fsum(level.mean for level in result.levels), 14
        )
        self.assertAlmostEqual(
            result.variance,
            math.fsum(level.variance / level.samples for level in result.levels),
            14,
        )
        self.assertEqual(result.samples, (400, 100, 50))
        self.assertEqual(result.evaluations, 550)
        self.assertEqual(result.total_cost, sampler.steps)
        self.assertEqual(result.levels[2].cost_per_sample, 6.0)
        self.assertTrue(result.converged)
        self.assertLess(
            abs(result.estimate - math.exp(0.05)), 4 * result.standard_error
        )

    def test_adaptive_mlmc_gbm_beats_single_level_cost(self) -> None:
        """Adaptive MLMC meets the RMSE target far cheaper than plain MC."""
        mu, sigma, eps = 1.0, 0.5, 0.02
        sampler = GeometricBrownianMotion(mu, sigma)
        result = adaptive_multilevel_monte_carlo(sampler, eps, seed=1)
        self.assertTrue(result.converged)
        self.assertLess(abs(result.estimate - math.exp(mu)), 3.0 * eps)
        self.assertLessEqual(result.variance, 0.5 * eps * eps * 1.0001)
        self.assertLessEqual(result.rmse_estimate, eps * 1.0001)
        self.assertEqual(result.total_cost, sampler.steps)
        self.assertGreaterEqual(result.finest_level, 5)
        self.assertAlmostEqual(result.gamma, 1.0, places=9)
        self.assertGreater(result.alpha, 0.6)
        self.assertLess(result.alpha, 1.5)
        self.assertGreater(result.beta, 0.6)
        self.assertLess(result.beta, 1.5)
        self.assertGreater(result.cost_ratio, 3.0)
        # Plain MC on the finest level with the exact Var(S_T) and the same
        # sampling-variance budget eps^2/2 needs N = 2 Var / eps^2 paths.
        exact_variance = math.exp(2.0 * mu) * (math.exp(sigma**2) - 1.0)
        single = 2.0 * exact_variance / eps**2 * 2.0**result.finest_level
        self.assertGreater(single, 5.0 * result.total_cost)
        decreasing = [level.samples for level in result.levels]
        self.assertEqual(decreasing, sorted(decreasing, reverse=True))

    def test_mlmc_rejects_invalid_inputs(self) -> None:
        """Malformed samplers and options raise explicit errors."""
        sampler = GeometricBrownianMotion(0.05, 0.2)
        with self.assertRaises(ValueError):
            multilevel_monte_carlo(sampler, (1, 10), seed=1)
        with self.assertRaises(ValueError):
            adaptive_multilevel_monte_carlo(sampler, 0.0, seed=1)
        with self.assertRaises(ValueError):
            adaptive_multilevel_monte_carlo(sampler, 0.1, max_level=1, seed=1)

        def short(level: int, count: int, stream: RandomStream):
            """Return one sample too few."""
            return [0.0] * (count - 1), [0.0] * (count - 1)

        with self.assertRaises(ValueError):
            multilevel_monte_carlo(short, (5,), seed=1)


class CrossEntropyTests(unittest.TestCase):
    """Cross-entropy adaptive importance sampling in standard-normal space."""

    def test_linear_limit_state(self) -> None:
        """Mean-shift CE recovers Phi(-4) and adapts toward the design point."""
        exact = standard_normal_cdf(-4.0)
        result = cross_entropy_importance_sampling(
            linear_state(), samples_per_level=500, seed=1
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.thresholds[-1], 0.0)
        self.assertEqual(
            list(result.thresholds), sorted(result.thresholds, reverse=True)
        )
        self.assertEqual(result.evaluations, 500 * result.levels + 500)
        self.assertLess(abs(result.probability - exact), 3.0 * result.standard_error)
        self.assertLess(result.coefficient_of_variation, 0.15)
        cosine = (result.mean[0] + result.mean[1]) / (
            math.sqrt(2.0) * math.hypot(*result.mean)
        )
        self.assertGreater(cosine, 0.99)

    def test_correlated_lognormal_limit_state(self) -> None:
        """CE works through the Nataf map of correlated non-normal inputs."""
        state, exact = lognormal_state()
        result = cross_entropy_importance_sampling(
            state, samples_per_level=1000, seed=1
        )
        self.assertLess(abs(result.probability / exact - 1.0), 3.0 * 0.065 + 0.01)
        self.assertLess(abs(result.probability - exact), 3.0 * result.standard_error)

    def test_covariance_adaptation_on_curved_limit_state(self) -> None:
        """Covariance adaptation returns a symmetric positive proposal."""
        exact = parabolic_exact(0.2)
        result = cross_entropy_importance_sampling(
            parabolic_state(0.2), samples_per_level=500, adapt_covariance=True, seed=1
        )
        cov = result.covariance
        self.assertAlmostEqual(cov[0][1], cov[1][0], places=14)
        self.assertGreater(cov[0][0] * cov[1][1] - cov[0][1] ** 2, 0.0)
        self.assertLess(abs(result.probability - exact), 4.0 * result.standard_error)

    def test_cross_entropy_rejects_invalid_options(self) -> None:
        """Invalid options raise explicit errors."""
        state = linear_state()
        with self.assertRaises(ValueError):
            cross_entropy_importance_sampling(state, level_quantile=0.7)
        with self.assertRaises(ValueError):
            cross_entropy_importance_sampling(
                state, samples_per_level=10, adapt_covariance=True
            )
        with self.assertRaises(ValueError):
            cross_entropy_importance_sampling(state, initial_mean=(0.0,))
        with self.assertRaises(TypeError):
            cross_entropy_importance_sampling("state")  # type: ignore[arg-type]


class LineSamplingTests(unittest.TestCase):
    """Line sampling with Brent root finding on each line."""

    def test_planar_limit_state_is_exact(self) -> None:
        """For a planar surface every line gives Phi(-beta): zero variance."""
        state, exact = lognormal_state()
        result = line_sampling(state, 40, seed=3)
        self.assertAlmostEqual(result.probability / exact, 1.0, places=6)
        self.assertLess(result.coefficient_of_variation, 1e-6)
        self.assertLessEqual(result.evaluations, 4 * 40)
        self.assertGreater(result.form_evaluations, 0)

    def test_parabolic_limit_state(self) -> None:
        """Curved limit states agree with quadrature within three errors."""
        for kappa in (0.2, -0.2):
            exact = parabolic_exact(kappa)
            state = parabolic_state(kappa)
            result = line_sampling(state, 100, seed=1)
            self.assertAlmostEqual(result.direction[1], 1.0, places=6)
            self.assertLess(
                abs(result.probability - exact), 3.0 * result.standard_error
            )
            explicit = line_sampling(state, 100, direction=(0.0, 2.0), seed=1)
            self.assertEqual(explicit.form_evaluations, 0)
            self.assertLess(
                abs(explicit.probability - exact), 3.0 * explicit.standard_error
            )

    def test_lines_without_crossing(self) -> None:
        """Always-safe and always-failed lines contribute 0 and 1."""
        safe = ProbabilisticModel(
            standard_pair(), lambda p: {"y": 5.0 + p["u1"] ** 2}, Y
        )
        result = line_sampling(
            ModelLimitState(safe, "y", 0.0, "below"), 5, direction=(1.0, 0.0), seed=1
        )
        self.assertEqual(result.probability, 0.0)
        self.assertTrue(all(d == math.inf for d in result.distances))
        failed = ProbabilisticModel(standard_pair(), lambda p: {"y": -1.0}, Y)
        result = line_sampling(
            ModelLimitState(failed, "y", 0.0, "below"), 5, direction=(1.0, 0.0), seed=1
        )
        self.assertEqual(result.probability, 1.0)

    def test_line_sampling_rejects_invalid_options(self) -> None:
        """Zero or mis-sized directions raise explicit errors."""
        state = linear_state()
        with self.assertRaises(ValueError):
            line_sampling(state, 10, direction=(0.0, 0.0))
        with self.assertRaises(ValueError):
            line_sampling(state, 10, direction=(1.0,))
        with self.assertRaises(ValueError):
            line_sampling(state, 1, direction=(1.0, 1.0))


class ComparisonIntegrationTests(unittest.TestCase):
    """The new estimators are opt-in rows of compare_reliability_methods."""

    def test_default_methods_unchanged_and_opt_in_rows(self) -> None:
        """Defaults keep five methods; 'ce' and 'line' add CE-IS and Line rows."""
        self.assertEqual(METHODS, ("form", "sorm", "is", "subset", "mc"))
        self.assertEqual(set(AVAILABLE_METHODS) - set(METHODS), {"ce", "line"})
        state, exact = lognormal_state()
        comparison = compare_reliability_methods(
            state,
            methods=("form", "ce", "line"),
            cross_entropy_samples=500,
            line_samples=20,
            seed=2026,
        )
        labels = [row.method for row in comparison.rows]
        self.assertEqual(labels, ["FORM", "CE-IS", "Line"])
        form = comparison.row("FORM")
        line = comparison.row("Line")
        self.assertGreater(line.evaluations, form.evaluations)
        self.assertAlmostEqual(line.probability / exact, 1.0, places=6)
        self.assertLess(abs(comparison.row("CE-IS").probability / exact - 1.0), 0.4)
        lone = form_hlrf(state)
        self.assertAlmostEqual(lone.beta, form.beta, places=9)


class ShapleyTests(unittest.TestCase):
    """Shapley effects with independent and Gaussian-copula inputs."""

    def test_exact_shapley_values_of_simple_games(self) -> None:
        """Additive games return their weights; symmetric games split evenly."""
        weights = (1.0, 2.0, 4.0)
        additive = exact_shapley_values(3, lambda s: math.fsum(weights[i] for i in s))
        for value, weight in zip(additive, weights, strict=True):
            self.assertAlmostEqual(value, weight, places=12)
        symmetric = exact_shapley_values(4, lambda s: float(len(s) ** 2))
        for value in symmetric:
            self.assertAlmostEqual(value, 4.0, places=12)
        with self.assertRaises(ValueError):
            exact_shapley_values(0, lambda s: 0.0)

    def test_linear_gaussian_closed_form(self) -> None:
        """The 2-D closed form agrees with the subset enumeration."""
        cases = (
            (1.0, 1.0, 1.0, 2.0, 0.6),
            (2.0, -1.0, 0.5, 3.0, -0.3),
            (0.3, 1.7, 1.2, 0.4, -0.8),
        )
        for b1, b2, s1, s2, rho in cases:
            covariance = ((s1 * s1, rho * s1 * s2), (rho * s1 * s2, s2 * s2))
            variance = (
                b1 * b1 * s1 * s1 + b2 * b2 * s2 * s2 + 2.0 * rho * b1 * b2 * s1 * s2
            )
            closed = (
                b1**2 * s1**2 * (1.0 - rho**2 / 2.0)
                + rho * b1 * b2 * s1 * s2
                + b2**2 * s2**2 * rho**2 / 2.0
            ) / variance
            effects = linear_gaussian_shapley((b1, b2), covariance)
            self.assertAlmostEqual(effects[0], closed, places=12)
            self.assertAlmostEqual(sum(effects), 1.0, places=12)
        independent = linear_gaussian_shapley(
            (1.0, 2.0, 3.0), ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
        )
        for value, expected in zip(independent, (1 / 14, 4 / 14, 9 / 14), strict=True):
            self.assertAlmostEqual(value, expected, places=12)

    def additive_model(self) -> ProbabilisticModel:
        """Return ``y = x0 + 2 x1 + 3 x2`` with independent standard normals."""
        inputs = JointInput.independent(
            *(RandomInput(f"x{i}", Normal(0.0, 1.0)) for i in range(3))
        )
        return ProbabilisticModel(
            inputs, lambda p: {"y": p["x0"] + 2.0 * p["x1"] + 3.0 * p["x2"]}, Y
        )

    def test_independent_additive_shapley_equals_sobol(self) -> None:
        """Without interactions or dependence Shapley = first = total Sobol."""
        model = self.additive_model()
        expected = (1 / 14, 4 / 14, 9 / 14)
        exact = shapley_effects(model, "y", outer_samples=400, seed=4)
        self.assertEqual(exact.method, "exact")
        self.assertAlmostEqual(sum(exact.effects), 1.0, places=12)
        self.assertEqual(exact.evaluations, 1000 + 6 * 400 * 3)
        for value, error, target in zip(
            exact.effects, exact.standard_errors, expected, strict=True
        ):
            self.assertLess(abs(value - target), max(4.0 * error, 0.03))
        self.assertEqual(exact.ranking(), ("x2", "x1", "x0"))
        sampled = shapley_effects(model, "y", method="random", permutations=300, seed=4)
        self.assertAlmostEqual(sum(sampled.effects), 1.0, places=12)
        self.assertEqual(sampled.evaluations, 1000 + 300 * 2 * 1 * 3)
        for value, error, target in zip(
            sampled.effects, sampled.standard_errors, expected, strict=True
        ):
            self.assertLess(abs(value - target), 4.0 * error)

    def test_correlated_gaussian_linear_model(self) -> None:
        """Conditional sampling reproduces analytic dependent Shapley effects."""
        rho, s1, s2 = 0.6, 1.0, 2.0
        inputs = JointInput(
            (RandomInput("x1", Normal(0.0, s1)), RandomInput("x2", Normal(0.0, s2))),
            ((1.0, rho), (rho, 1.0)),
        )
        model = ProbabilisticModel(inputs, lambda p: {"y": p["x1"] + p["x2"]}, Y)
        covariance = ((s1 * s1, rho * s1 * s2), (rho * s1 * s2, s2 * s2))
        analytic = linear_gaussian_shapley((1.0, 1.0), covariance)
        result = shapley_effects(
            model, "y", outer_samples=500, variance_samples=2000, seed=1
        )
        self.assertAlmostEqual(sum(result.effects), 1.0, places=12)
        for value, error, target in zip(
            result.effects, result.standard_errors, analytic, strict=True
        ):
            self.assertLess(abs(value - target), 4.0 * error)
        variance = 1.0 + 4.0 + 2.0 * rho * 2.0
        first_x1 = (s1 + rho * s2) ** 2 / variance
        total_x1 = 1.0 - (s2 + rho * s1) ** 2 / variance
        self.assertLess(total_x1, analytic[0])
        self.assertLess(analytic[0], first_x1)

    def test_shapley_rejects_invalid_options(self) -> None:
        """Invalid options and a constant output raise explicit errors."""
        model = self.additive_model()
        with self.assertRaises(ValueError):
            shapley_effects(model, "y", method="sobol")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            shapley_effects(model, "y", inner_samples=1)
        constant = ProbabilisticModel(model.inputs, lambda p: {"y": 1.0}, Y)
        with self.assertRaises(ValueError):
            shapley_effects(constant, "y", variance_samples=10, seed=1)
        with self.assertRaises(KeyError):
            shapley_effects(model, "missing")


if __name__ == "__main__":
    unittest.main()
