"""Tests for the acquisition functions and the Bayesian optimization loop.

The improvement function ``h`` and its logarithm are compared with the
arbitrary-precision reference backend; expected improvement with adaptive
quadrature of ``max(f - f*, 0)`` against the normal density; the knowledge
gradient with a Monte Carlo estimate of ``E[max mu'] - max mu``; constrained
expected improvement with its limits; the joint GP posterior with the
marginal predictions; and the loop with known global optima and with random
search of the same budget over several seeds.
"""

from __future__ import annotations

import math
import random
import unittest
from decimal import Decimal

from probstat.design import (
    ACQUISITIONS,
    BayesianOptimizationResult,
    ThompsonDraw,
    bayesian_optimization,
    confidence_bound,
    constrained_expected_improvement,
    expected_improvement,
    gaussian_process_joint_posterior,
    gp_ucb_beta,
    improvement_function,
    knowledge_gradient,
    log_expected_improvement,
    log_improvement_function,
    probability_of_feasibility,
    thompson_sample,
)
from probstat.modeling.benchmarks import rastrigin
from probstat.numerics import (
    integrate,
    reference_exp,
    reference_normal_cdf,
    reference_pi,
)
from probstat.surrogates import gaussian_process_from_hyperparameters


def wavy(x: tuple[float, ...]) -> float:
    """Return a one-dimensional test function on ``[0, 1]``.

    Its global minimum is ``-6.020740`` at ``x = 0.757249`` and it has a
    second local minimum ``-0.986325`` at ``x = 0.142590``.
    """
    return (6.0 * x[0] - 2.0) ** 2 * math.sin(12.0 * x[0] - 4.0)


WAVY_MINIMUM = -6.020740055767


def reference_log_h(z: float) -> float:
    """Return ``log(phi(z) + z Phi(z))`` from 50-digit reference arithmetic."""
    point = Decimal(z)
    density = (
        reference_exp(-point * point / 2, digits=50)
        / (2 * reference_pi(digits=50)).sqrt()
    )
    value = density + point * reference_normal_cdf(point, digits=50)
    return float(value.ln())


def random_search(
    function: object,
    box: list[tuple[float, float]],
    budget: int,
    seed: int,
    *,
    maximize: bool = False,
) -> float:
    """Return the best of ``budget`` uniform random evaluations."""
    generator = random.Random(seed)
    values = [
        function(tuple(generator.uniform(a, b) for a, b in box))  # type: ignore[operator]
        for _ in range(budget)
    ]
    return max(values) if maximize else min(values)


class ImprovementFunctionTests(unittest.TestCase):
    """``h(z) = phi(z) + z Phi(z)`` and its stable logarithm."""

    def test_quoted_values(self) -> None:
        """``h(0)`` and ``h(-1)`` match ten published digits."""
        self.assertAlmostEqual(improvement_function(0.0), 0.3989422804, delta=5e-11)
        self.assertAlmostEqual(improvement_function(-1.0), 0.0833154706, delta=5e-11)
        self.assertAlmostEqual(
            improvement_function(0.0), 1.0 / math.sqrt(2.0 * math.pi), delta=1e-16
        )

    def test_log_matches_high_precision_reference(self) -> None:
        """``log h`` agrees with 50-digit arithmetic to 2e-14 relative."""
        worst = 0.0
        points = (
            8.0,
            3.0,
            0.5,
            0.0,
            -0.5,
            -0.999,
            -1.0,
            -1.001,
            -1.5,
            -2.0,
            -3.5,
            -6.0,
            -10.0,
            -20.0,
            -37.0,
            -40.0,
            -100.0,
        )
        for z in points:
            got = log_improvement_function(z)
            want = reference_log_h(z)
            worst = max(worst, abs(got - want) / max(1.0, abs(want)))
            self.assertAlmostEqual(got, want, delta=2e-14 * max(1.0, abs(want)))
        self.assertLess(worst, 2e-14)

    def test_tail_is_finite_where_the_naive_formula_underflows(self) -> None:
        """``log h(-40)`` is about -808.3 while ``phi + z Phi`` is zero."""
        z = -40.0
        naive = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) + z * 0.5 * math.erfc(
            -z / math.sqrt(2.0)
        )
        self.assertEqual(naive, 0.0)
        value = log_improvement_function(z)
        self.assertTrue(math.isfinite(value))
        self.assertAlmostEqual(value, -808.2985683566, delta=1e-9)
        self.assertEqual(improvement_function(z), 0.0)
        huge = log_improvement_function(-1e6)
        self.assertAlmostEqual(
            huge,
            -0.5e12 - 0.5 * math.log(2.0 * math.pi) - 2.0 * math.log(1e6),
            delta=1e-3,
        )

    def test_monotone_continuous_and_limits(self) -> None:
        """``log h`` increases, is continuous at the branch point, and has limits."""
        grid = [-60.0 + 0.25 * i for i in range(281)]
        values = [log_improvement_function(z) for z in grid]
        self.assertTrue(all(a < b for a, b in zip(values, values[1:], strict=False)))
        below = log_improvement_function(math.nextafter(-1.0, -2.0))
        above = log_improvement_function(math.nextafter(-1.0, 0.0))
        self.assertAlmostEqual(below, above, delta=1e-14)
        self.assertEqual(log_improvement_function(-math.inf), -math.inf)
        self.assertEqual(log_improvement_function(math.inf), math.inf)
        self.assertEqual(log_improvement_function(-1e200), -math.inf)
        self.assertAlmostEqual(
            log_improvement_function(1e200), math.log(1e200), delta=1e-9
        )
        self.assertEqual(improvement_function(math.inf), math.inf)
        self.assertEqual(improvement_function(-math.inf), 0.0)
        for function in (log_improvement_function, improvement_function):
            with self.assertRaises(ValueError):
                function(math.nan)
        for z in (-0.3, -4.0, 2.0):
            self.assertAlmostEqual(
                math.log(improvement_function(z)),
                log_improvement_function(z),
                delta=1e-13,
            )


class ExpectedImprovementTests(unittest.TestCase):
    """Expected improvement against quadrature, and its logarithm."""

    def test_matches_quadrature_of_the_improvement(self) -> None:
        """``E[max(F - f*, 0)]`` by quadrature equals ``s h(z)`` both ways."""
        cases = ((0.3, 0.8, 1.1), (2.0, 0.5, 0.4), (-1.0, 2.5, 3.0), (0.0, 1.0, 0.0))
        for mean, sd, best in cases:

            def density(f: float, mean: float = mean, sd: float = sd) -> float:
                """Return the normal predictive density."""
                z = (f - mean) / sd
                return math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))

            above = integrate(
                lambda f, best=best: (f - best) * density(f), best, math.inf
            ).value
            below = integrate(
                lambda f, best=best: (best - f) * density(f), -math.inf, best
            ).value
            self.assertAlmostEqual(
                expected_improvement(mean, sd, best, maximize=True), above, delta=1e-9
            )
            self.assertAlmostEqual(
                expected_improvement(mean, sd, best), below, delta=1e-9
            )
            self.assertAlmostEqual(above - below, mean - best, delta=1e-9)
            for maximize in (False, True):
                self.assertAlmostEqual(
                    log_expected_improvement(mean, sd, best, maximize=maximize),
                    math.log(expected_improvement(mean, sd, best, maximize=maximize)),
                    delta=1e-12,
                )

    def test_zero_deviation_and_far_tail(self) -> None:
        """``s = 0`` gives the deterministic improvement; the log stays finite."""
        self.assertEqual(expected_improvement(1.0, 0.0, 3.0), 2.0)
        self.assertEqual(expected_improvement(1.0, 0.0, 3.0, maximize=True), 0.0)
        self.assertEqual(log_expected_improvement(1.0, 0.0, 3.0), math.log(2.0))
        self.assertEqual(
            log_expected_improvement(1.0, 0.0, 3.0, maximize=True), -math.inf
        )
        self.assertEqual(expected_improvement(50.0, 1.0, 0.0), 0.0)
        far = log_expected_improvement(50.0, 1.0, 0.0)
        self.assertTrue(math.isfinite(far))
        self.assertAlmostEqual(far, log_improvement_function(-50.0), delta=1e-12)
        self.assertLess(
            log_expected_improvement(51.0, 1.0, 0.0),
            log_expected_improvement(50.0, 1.0, 0.0),
        )

    def test_validation(self) -> None:
        """Negative deviations and non-finite arguments raise."""
        for function in (expected_improvement, log_expected_improvement):
            with self.assertRaises(ValueError):
                function(0.0, -1.0, 0.0)
            with self.assertRaises(ValueError):
                function(math.nan, 1.0, 0.0)
            with self.assertRaises(ValueError):
                function(0.0, 1.0, math.inf)


class ConfidenceBoundTests(unittest.TestCase):
    """Confidence bounds and the GP-UCB schedule."""

    def test_bounds_and_direction(self) -> None:
        """The bound is ``mu + sqrt(beta) s`` (max) or ``mu - sqrt(beta) s``."""
        self.assertAlmostEqual(
            confidence_bound(1.0, 0.5, 4.0, maximize=True), 2.0, delta=1e-15
        )
        self.assertAlmostEqual(confidence_bound(1.0, 0.5, 4.0), 0.0, delta=1e-15)
        self.assertEqual(confidence_bound(1.0, 0.5, 0.0), 1.0)
        with self.assertRaises(ValueError):
            confidence_bound(1.0, 0.5, -1.0)
        with self.assertRaises(ValueError):
            confidence_bound(1.0, -0.5, 1.0)

    def test_gp_ucb_schedule(self) -> None:
        """``beta_t = 2 log(|D| t^2 pi^2 / (6 delta))`` and it grows with ``t``."""
        self.assertAlmostEqual(
            gp_ucb_beta(1, 1, 0.5), 2.0 * math.log(math.pi**2 / 3.0), delta=1e-14
        )
        self.assertAlmostEqual(
            gp_ucb_beta(7, 100, 0.1) - gp_ucb_beta(1, 100, 0.1),
            4.0 * math.log(7.0),
            delta=1e-12,
        )
        self.assertAlmostEqual(
            gp_ucb_beta(3, 500, 0.1) - gp_ucb_beta(3, 5, 0.1),
            2.0 * math.log(100.0),
            delta=1e-12,
        )
        self.assertGreater(gp_ucb_beta(2, 10, 0.01), gp_ucb_beta(2, 10, 0.1))
        for arguments in ((0, 10, 0.1), (1, 0, 0.1), (1, 10, 0.0), (1, 10, 1.0)):
            with self.assertRaises(ValueError):
                gp_ucb_beta(*arguments)
        with self.assertRaises(ValueError):
            gp_ucb_beta(1.5, 10, 0.1)  # type: ignore[arg-type]


class ConstrainedImprovementTests(unittest.TestCase):
    """Expected improvement weighted by the probability of feasibility."""

    def test_certain_constraints_reduce_to_plain_improvement(self) -> None:
        """Certainly satisfied constraints give EI; a violated one gives zero."""
        plain = expected_improvement(0.2, 0.7, 0.5)
        satisfied = constrained_expected_improvement(
            0.2, 0.7, 0.5, [-50.0, -3.0], [1.0, 0.0]
        )
        self.assertAlmostEqual(satisfied, plain, delta=1e-15)
        self.assertEqual(constrained_expected_improvement(0.2, 0.7, 0.5, [], []), plain)
        self.assertEqual(
            constrained_expected_improvement(0.2, 0.7, 0.5, [-50.0, 2.0], [1.0, 0.0]),
            0.0,
        )

    def test_product_of_feasibility_probabilities(self) -> None:
        """The weight is ``prod Phi((threshold - mu) / s)``."""
        value = constrained_expected_improvement(
            1.0,
            0.4,
            0.9,
            [0.0, 1.0],
            [1.0, 2.0],
            thresholds=[0.0, 2.0],
            maximize=True,
        )
        phi_half = 0.5 * math.erfc(-0.5 / math.sqrt(2.0))
        self.assertAlmostEqual(
            value,
            expected_improvement(1.0, 0.4, 0.9, maximize=True) * 0.5 * phi_half,
            delta=1e-15,
        )
        self.assertAlmostEqual(probability_of_feasibility(0.0, 1.0), 0.5, delta=1e-16)
        self.assertAlmostEqual(
            probability_of_feasibility(1.0, 2.0, 2.0), phi_half, delta=1e-16
        )
        self.assertEqual(probability_of_feasibility(0.0, 0.0), 1.0)
        self.assertEqual(probability_of_feasibility(1e-9, 0.0), 0.0)
        with self.assertRaises(ValueError):
            constrained_expected_improvement(1.0, 0.4, 0.9, [0.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            constrained_expected_improvement(
                1.0, 0.4, 0.9, [0.0], [1.0], thresholds=[0.0, 1.0]
            )
        with self.assertRaises(ValueError):
            probability_of_feasibility(0.0, -1.0)


class KnowledgeGradientTests(unittest.TestCase):
    """Knowledge gradient for independent normal beliefs."""

    means = (0.3, 1.1, 0.9, -0.5, 1.05)
    deviations = (1.2, 0.4, 0.8, 2.0, 0.0)
    noise = (0.5, 0.1, 1.0, 0.3, 0.2)

    def test_matches_monte_carlo_of_the_one_step_gain(self) -> None:
        """``E[max mu'] - max mu`` by Monte Carlo agrees within 4 standard errors."""
        exact = knowledge_gradient(self.means, self.deviations, self.noise)
        generator = random.Random(123)
        draws = [generator.gauss(0.0, 1.0) for _ in range(40000)]
        best = max(self.means)
        for index, (mean, deviation, noise) in enumerate(
            zip(self.means, self.deviations, self.noise, strict=True)
        ):
            if deviation == 0.0:
                self.assertEqual(exact[index], 0.0)
                continue
            change = deviation**2 / math.sqrt(deviation**2 + noise)
            rival = max(m for j, m in enumerate(self.means) if j != index)
            gains = [max(mean + change * z, rival) - best for z in draws]
            estimate = sum(gains) / len(gains)
            spread = math.sqrt(
                sum((g - estimate) ** 2 for g in gains) / (len(gains) - 1) / len(gains)
            )
            self.assertGreater(exact[index], 0.0)
            self.assertAlmostEqual(exact[index], estimate, delta=4.0 * spread)

    def test_posterior_mean_change_has_the_stated_variance(self) -> None:
        """Conjugate updating gives ``Var(mu') = s^4 / (s^2 + lambda)``."""
        generator = random.Random(5)
        mean, deviation, noise = 0.3, 1.2, 0.5
        updated = []
        for _ in range(20000):
            truth = generator.gauss(mean, deviation)
            observation = generator.gauss(truth, math.sqrt(noise))
            precision = 1.0 / deviation**2 + 1.0 / noise
            updated.append((mean / deviation**2 + observation / noise) / precision)
        centre = sum(updated) / len(updated)
        variance = sum((u - centre) ** 2 for u in updated) / (len(updated) - 1)
        expected = deviation**4 / (deviation**2 + noise)
        self.assertAlmostEqual(variance, expected, delta=0.05 * expected)

    def test_direction_scalar_noise_and_validation(self) -> None:
        """Minimization mirrors maximization; a scalar noise is broadcast."""
        mirrored = knowledge_gradient(
            [-m for m in self.means], self.deviations, self.noise, maximize=False
        )
        direct = knowledge_gradient(self.means, self.deviations, self.noise)
        for a, b in zip(mirrored, direct, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-15)
        scalar = knowledge_gradient(self.means, self.deviations, 0.25)
        vector = knowledge_gradient(self.means, self.deviations, [0.25] * 5)
        self.assertEqual(scalar, vector)
        noiseless = knowledge_gradient((0.0, 0.0), (1.0, 1.0), 0.0)
        self.assertAlmostEqual(
            noiseless[0], 1.0 / math.sqrt(2.0 * math.pi), delta=1e-15
        )
        with self.assertRaises(ValueError):
            knowledge_gradient((1.0,), (1.0,), 0.1)
        with self.assertRaises(ValueError):
            knowledge_gradient((1.0, 2.0), (1.0, -1.0), 0.1)
        with self.assertRaises(ValueError):
            knowledge_gradient((1.0, 2.0), (1.0, 1.0), -0.1)
        with self.assertRaises(ValueError):
            knowledge_gradient((1.0, 2.0), (1.0, 1.0), (0.1,))


class JointPosteriorAndThompsonTests(unittest.TestCase):
    """Joint GP posterior on a finite set and Thompson draws from it."""

    def setUp(self) -> None:
        """Condition a GP with fixed hyperparameters on five points."""
        self.points = [(0.05,), (0.3,), (0.55,), (0.8,), (0.95,)]
        self.model = gaussian_process_from_hyperparameters(
            self.points,
            [wavy(point) for point in self.points],
            signal_variance=16.0,
            length_scales=(0.2,),
            noise_variance=1e-8,
            mean=1.0,
        )
        self.candidates = [(0.1,), (0.42,), (0.7,), (0.76,), (0.3,)]

    def test_joint_posterior_matches_marginals(self) -> None:
        """Means and the covariance diagonal equal the pointwise predictions."""
        means, covariance = gaussian_process_joint_posterior(
            self.model, self.candidates
        )
        for i, candidate in enumerate(self.candidates):
            mean, variance = self.model.predict_with_variance(candidate)
            self.assertAlmostEqual(means[i], mean, delta=1e-10)
            self.assertAlmostEqual(covariance[i][i], variance, delta=1e-9)
            for j in range(len(self.candidates)):
                self.assertEqual(covariance[i][j], covariance[j][i])
                bound = math.sqrt(abs(covariance[i][i] * covariance[j][j])) + 1e-6
                self.assertLessEqual(abs(covariance[i][j]), bound)
        self.assertLess(covariance[4][4], 1e-6)
        with self.assertRaises(ValueError):
            gaussian_process_joint_posterior(self.model, [])
        with self.assertRaises(ValueError):
            gaussian_process_joint_posterior(self.model, [(0.1, 0.2)])

    def test_joint_posterior_of_a_fitted_gp_uses_original_units(self) -> None:
        """A standardized, fitted GP gives the same agreement."""
        from probstat.surrogates import fit_gaussian_process

        fitted = fit_gaussian_process(
            self.points, [100.0 + 5.0 * wavy(p) for p in self.points], seed=2
        )
        means, covariance = gaussian_process_joint_posterior(fitted, self.candidates)
        for i, candidate in enumerate(self.candidates):
            mean, variance = fitted.predict_with_variance(candidate)
            self.assertAlmostEqual(means[i], mean, delta=1e-9)
            self.assertAlmostEqual(
                covariance[i][i], variance, delta=1e-6 * (1.0 + variance)
            )

    def test_thompson_draws_have_the_posterior_moments(self) -> None:
        """Sample means and variances of draws match the joint posterior."""
        means, covariance = gaussian_process_joint_posterior(
            self.model, self.candidates
        )
        draws = [
            thompson_sample(self.model, self.candidates, seed=seed)
            for seed in range(400)
        ]
        for draw in draws[:20]:
            self.assertIsInstance(draw, ThompsonDraw)
            self.assertEqual(draw.values[draw.index], min(draw.values))
            self.assertEqual(draw.point, self.candidates[draw.index])
        for i in range(len(self.candidates)):
            sample = [draw.values[i] for draw in draws]
            centre = sum(sample) / len(sample)
            variance = sum((v - centre) ** 2 for v in sample) / (len(sample) - 1)
            deviation = math.sqrt(max(covariance[i][i], 0.0))
            self.assertAlmostEqual(
                centre, means[i], delta=4.0 * deviation / math.sqrt(400) + 1e-4
            )
            self.assertAlmostEqual(variance, covariance[i][i], delta=0.35 * variance)
        upper = thompson_sample(self.model, self.candidates, maximize=True, seed=1)
        self.assertEqual(upper.values[upper.index], max(upper.values))
        again = thompson_sample(self.model, self.candidates, maximize=True, seed=1)
        self.assertEqual(upper, again)


class BayesianOptimizationLoopTests(unittest.TestCase):
    """The sequential loop on functions with known optima."""

    def test_one_dimensional_minimum_and_random_search(self) -> None:
        """Fifteen evaluations reach the minimum to 1e-3 and beat random search."""
        box = [(0.0, 1.0)]
        regrets = []
        baseline = []
        for seed in range(5):
            result = bayesian_optimization(wavy, box, budget=15, seed=seed)
            self.assertIsInstance(result, BayesianOptimizationResult)
            self.assertEqual(result.evaluations, 15)
            assert result.best_point is not None
            self.assertAlmostEqual(result.best_point[0], 0.757249, delta=5e-3)
            regrets.append(result.best_value - WAVY_MINIMUM)
            baseline.append(random_search(wavy, box, 15, seed) - WAVY_MINIMUM)
        self.assertLess(max(regrets), 1e-3)
        self.assertGreaterEqual(min(regrets), -1e-9)
        self.assertLess(sum(regrets), 0.01 * sum(baseline))
        self.assertGreaterEqual(
            sum(r <= b for r, b in zip(regrets, baseline, strict=True)), 4
        )

    def test_two_dimensional_maximum_of_the_library_benchmark(self) -> None:
        """Twenty-five evaluations reach the Rastrigin maximum 20 within 0.05."""

        def objective(x: tuple[float, ...]) -> float:
            """Return the library's modified Rastrigin function."""
            return rastrigin(x[0], x[1])

        box = [(-0.75, 0.75), (-0.6, 0.9)]
        regrets = []
        baseline = []
        for seed in range(3):
            result = bayesian_optimization(
                objective, box, budget=25, maximize=True, seed=seed
            )
            assert result.best_point is not None
            self.assertTrue(result.maximize)
            self.assertLess(math.hypot(*result.best_point), 0.03)
            regrets.append(20.0 - result.best_value)
            baseline.append(
                20.0 - random_search(objective, box, 25, seed, maximize=True)
            )
        self.assertLess(max(regrets), 0.05)
        self.assertGreaterEqual(min(regrets), 0.0)
        self.assertLess(sum(regrets), 0.1 * sum(baseline))
        self.assertTrue(all(r < b for r, b in zip(regrets, baseline, strict=True)))

    def test_every_acquisition_finds_the_basin(self) -> None:
        """EI, UCB, and Thompson sampling all reach the global basin."""
        for acquisition, tolerance in (("ei", 1e-3), ("ucb", 1e-2), ("thompson", 0.05)):
            result = bayesian_optimization(
                wavy, [(0.0, 1.0)], budget=15, acquisition=acquisition, seed=1
            )
            self.assertEqual(result.acquisition, acquisition)
            self.assertLess(result.best_value - WAVY_MINIMUM, tolerance)
            self.assertEqual(len(result.acquisition_values), 15 - result.initial_size)
        fixed = bayesian_optimization(
            wavy, [(0.0, 1.0)], budget=12, acquisition="ucb", beta=4.0, seed=1
        )
        self.assertLess(fixed.best_value - WAVY_MINIMUM, 0.05)
        self.assertEqual(ACQUISITIONS, ("log_ei", "ei", "ucb", "thompson"))

    def test_constrained_minimum(self) -> None:
        """With ``x <= 0.6`` the loop finds the constrained minimum -0.986325."""

        def constraint(x: tuple[float, ...]) -> float:
            """Return ``x - 0.6`` (feasible when non-positive)."""
            return x[0] - 0.6

        for seed in range(2):
            result = bayesian_optimization(
                wavy, [(0.0, 1.0)], budget=18, constraints=[constraint], seed=seed
            )
            assert result.best_point is not None
            self.assertTrue(result.feasible_found)
            self.assertLessEqual(result.best_point[0], 0.6)
            self.assertAlmostEqual(result.best_value, -0.986325, delta=1e-3)
            self.assertEqual(result.constraint_evaluations, 18)
            for point, limits, allowed in zip(
                result.points, result.constraint_values, result.feasible, strict=True
            ):
                self.assertEqual(limits, (point[0] - 0.6,))
                self.assertEqual(allowed, point[0] <= 0.6)
            feasible_values = [
                v for v, ok in zip(result.values, result.feasible, strict=True) if ok
            ]
            self.assertEqual(result.best_value, min(feasible_values))

    def test_infeasible_start_and_never_feasible(self) -> None:
        """Feasibility is searched first; an empty feasible set is reported."""
        never = bayesian_optimization(
            wavy, [(0.0, 1.0)], budget=7, constraints=[lambda x: 1.0 + x[0]], seed=0
        )
        self.assertFalse(never.feasible_found)
        self.assertIsNone(never.best_point)
        self.assertTrue(math.isnan(never.best_value))
        self.assertTrue(all(math.isnan(v) for v in never.best_history))
        self.assertIn("No feasible point", never.summary())
        narrow = bayesian_optimization(
            wavy,
            [(0.0, 1.0)],
            budget=14,
            initial_size=4,
            constraints=[lambda x: 0.93 - x[0]],
            seed=5,
        )
        self.assertTrue(narrow.feasible_found)
        assert narrow.best_point is not None
        self.assertGreaterEqual(narrow.best_point[0], 0.93)

    def test_bookkeeping_and_reproducibility(self) -> None:
        """Counts, history, and seeds behave as documented."""
        calls = []

        def counted(x: tuple[float, ...]) -> float:
            """Record the call and return the test function."""
            calls.append(x)
            return wavy(x)

        result = bayesian_optimization(
            counted, [(0.0, 1.0)], budget=10, initial_size=5, seed=11
        )
        self.assertEqual(len(calls), 10)
        self.assertEqual(result.points, tuple(calls))
        self.assertEqual(result.initial_size, 5)
        self.assertEqual(result.surrogate_fits + result.fallback_iterations, 5)
        self.assertGreater(result.likelihood_evaluations, 0)
        self.assertGreater(result.acquisition_evaluations, 5 * 100)
        self.assertEqual(result.constraint_evaluations, 0)
        self.assertEqual(len(set(result.points)), 10)
        self.assertTrue(all(0.0 <= p[0] <= 1.0 for p in result.points))
        self.assertEqual(result.best_value, min(result.values))
        history = result.best_history
        self.assertEqual(len(history), 10)
        self.assertTrue(all(a >= b for a, b in zip(history, history[1:], strict=False)))
        self.assertEqual(history[-1], result.best_value)
        self.assertIn("not a certified global optimum", result.summary())
        again = bayesian_optimization(
            wavy, [(0.0, 1.0)], budget=10, initial_size=5, seed=11
        )
        self.assertEqual(result.points, again.points)
        other = bayesian_optimization(
            wavy, [(0.0, 1.0)], budget=10, initial_size=5, seed=12
        )
        self.assertNotEqual(result.points, other.points)
        maximin = bayesian_optimization(
            wavy,
            [(0.0, 1.0)],
            budget=6,
            initial_size=6,
            initial_design="maximin",
            seed=1,
        )
        self.assertEqual(maximin.evaluations, 6)
        self.assertEqual(maximin.surrogate_fits, 0)

    def test_constant_objective_falls_back_without_failing(self) -> None:
        """A surrogate that cannot be fitted is reported, not hidden."""
        result = bayesian_optimization(lambda x: 2.5, [(0.0, 1.0)], budget=7, seed=3)
        self.assertEqual(result.fallback_iterations, 7 - result.initial_size)
        self.assertEqual(result.surrogate_fits, 0)
        self.assertEqual(result.best_value, 2.5)
        self.assertTrue(all(math.isnan(v) for v in result.acquisition_values))
        self.assertEqual(len(set(result.points)), 7)

    def test_input_validation(self) -> None:
        """Invalid options raise ``ValueError`` before any evaluation."""
        box = [(0.0, 1.0)]
        bad = [
            {"budget": 0},
            {"budget": 10, "acquisition": "pi"},
            {"budget": 10, "initial_size": 1},
            {"budget": 10, "initial_size": 11},
            {"budget": 10, "initial_design": "grid"},
            {"budget": 10, "kernel": "cubic"},
            {"budget": 10, "candidates": 0},
            {"budget": 10, "local_starts": 0},
            {"budget": 10, "beta": -1.0},
            {"budget": 10, "delta": 1.5},
            {"budget": 10, "acquisition": "ucb", "constraints": [lambda x: -1.0]},
            {"budget": 10, "acquisition": "thompson", "constraints": [lambda x: -1.0]},
        ]
        for options in bad:
            with self.assertRaises(ValueError):
                bayesian_optimization(wavy, box, **options)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bayesian_optimization(wavy, [(1.0, 1.0)], budget=5)
        with self.assertRaises(ValueError):
            bayesian_optimization(wavy, [(0.0, math.inf)], budget=5)
        with self.assertRaises(ValueError):
            bayesian_optimization(wavy, [], budget=5)
        with self.assertRaises(ValueError):
            bayesian_optimization(lambda x: math.nan, box, budget=5)


if __name__ == "__main__":
    unittest.main()
