"""Tests for reliability sensitivities: FORM, score-function, failure Sobol."""

from __future__ import annotations

import math
import random
import statistics
import unittest

from probstat.modeling import gauss_hermite
from probstat.reliability.reliability_sensitivity import (
    failure_indicator_sobol,
    form_sensitivities,
    form_sensitivities_finite_difference,
    normal_input_limit_state,
    normal_parameter_names,
    normal_scores,
    score_function_estimator,
    score_function_monte_carlo,
    score_function_subset,
)


def pdf(x: float) -> float:
    """Return the standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def cdf(x: float) -> float:
    """Return the standard normal CDF through erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


A = (1.0, 0.5)


def linear_truth(
    r: float, mean: tuple[float, float], std: tuple[float, float]
) -> tuple[float, tuple[float, ...]]:
    """Return ``P_f`` and ``dP_f/d(mu, sigma)`` of ``g = r - a.x <= 0``.

    ``a.X ~ N(m, s^2)`` with ``m = a.mu`` and ``s^2 = sum a_i^2 sigma_i^2``;
    ``P_f = Phi((m - r)/s)`` so ``dP_f/dmu_i = phi(z) a_i / s`` and
    ``dP_f/dsigma_i = -phi(z) z a_i^2 sigma_i / s^2`` with ``z = (m - r)/s``.
    """
    m = sum(a * mu for a, mu in zip(A, mean, strict=True))
    s = math.sqrt(sum((a * sg) ** 2 for a, sg in zip(A, std, strict=True)))
    z = (m - r) / s
    d_mu = tuple(pdf(z) * a / s for a in A)
    d_sigma = tuple(
        -pdf(z) * z * a * a * sg / (s * s) for a, sg in zip(A, std, strict=True)
    )
    return cdf(z), d_mu + d_sigma


def linear_g(r: float):  # type: ignore[no-untyped-def]
    """Return the physical limit state ``r - a.x``."""
    return lambda x: r - A[0] * x[0] - A[1] * x[1]


class FORMSensitivityTests(unittest.TestCase):
    """Design-point sensitivities against closed forms and FORM differences."""

    def test_linear_limit_state_closed_form(self) -> None:
        """Linear case: dbeta/dmu_i = -a_i/s, dbeta/dsigma_i = -beta a_i^2 s_i/s^2."""
        r, mean, std = 6.0, (1.0, 2.0), (0.5, 0.8)
        problem = normal_input_limit_state(linear_g(r), 2)
        theta = mean + std
        result = form_sensitivities(problem, theta, 2, start=(0.1, 0.1))
        m = A[0] * mean[0] + A[1] * mean[1]
        s = math.sqrt((A[0] * std[0]) ** 2 + (A[1] * std[1]) ** 2)
        beta = (r - m) / s
        expected = tuple(-a / s for a in A) + tuple(
            -beta * a * a * sg / (s * s) for a, sg in zip(A, std, strict=True)
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.beta, beta, 8)
        for got, want in zip(result.dbeta, expected, strict=True):
            self.assertAlmostEqual(got, want, 7)
        # dP/dtheta = -phi(beta) dbeta/dtheta; the first-order probability is
        # exact for a linear limit state of normals.
        _, exact = linear_truth(r, mean, std)
        for got, want in zip(result.dprobability, exact, strict=True):
            self.assertAlmostEqual(got, want, 7)
        self.assertAlmostEqual(result.probability, cdf(-beta), 12)
        self.assertAlmostEqual(math.fsum(a * a for a in result.alpha), 1.0, 12)

    def test_design_parameter_closed_form(self) -> None:
        """G = d R - S: beta = (d mu_R - mu_S)/sqrt(d^2 s_R^2 + s_S^2)."""
        mu_r, mu_s, s_r, s_s, d = 3.0, 1.0, 0.5, 0.4, 1.2
        problem = normal_input_limit_state(
            lambda x, dd: dd[0] * x[0] - x[1], 2, design=1
        )
        theta = (mu_r, mu_s, s_r, s_s, d)
        result = form_sensitivities(problem, theta, 2, start=(0.1, 0.1))
        s = math.sqrt(d * d * s_r * s_r + s_s * s_s)
        m = d * mu_r - mu_s
        self.assertAlmostEqual(result.beta, m / s, 8)
        self.assertAlmostEqual(result.dbeta[4], mu_r / s - m * d * s_r**2 / s**3, 7)
        self.assertAlmostEqual(result.dbeta[0], d / s, 7)
        self.assertAlmostEqual(result.dbeta[1], -1.0 / s, 7)
        self.assertEqual(
            normal_parameter_names(2, 1),
            ("mu_1", "mu_2", "sigma_1", "sigma_2", "d_1"),
        )

    def test_nonlinear_matches_finite_differences_of_form(self) -> None:
        """Envelope-theorem derivatives equal central differences of FORM runs."""

        def g(x: tuple[float, ...], d: tuple[float, ...]) -> float:
            return d[0] - x[0] - 0.4 * x[1] - 0.05 * x[0] * x[1] + 0.02 * x[1] ** 2

        problem = normal_input_limit_state(g, 2, design=1)
        theta = (1.0, 0.5, 0.6, 0.9, 6.0)
        analytic = form_sensitivities(problem, theta, 2, start=(0.3, 0.3))
        numeric = form_sensitivities_finite_difference(
            problem, theta, 2, start=(0.3, 0.3)
        )
        self.assertTrue(analytic.converged)
        self.assertGreater(analytic.beta, 2.0)
        for a, b in zip(analytic.dbeta, numeric, strict=True):
            self.assertAlmostEqual(a, b, delta=2e-6 * max(1.0, abs(b)))

    def test_analytic_gradient_gives_same_result(self) -> None:
        """A supplied grad_u G reproduces the finite-difference search."""
        r, mean, std = 6.0, (1.0, 2.0), (0.5, 0.8)
        problem = normal_input_limit_state(linear_g(r), 2)
        theta = mean + std

        def grad(u: tuple[float, ...], th: tuple[float, ...]) -> tuple[float, ...]:
            return (-A[0] * th[2], -A[1] * th[3])

        a = form_sensitivities(problem, theta, 2, start=(0.1, 0.1), gradient_u=grad)
        b = form_sensitivities(problem, theta, 2, start=(0.1, 0.1))
        for x, y in zip(a.dbeta, b.dbeta, strict=True):
            self.assertAlmostEqual(x, y, 7)
        self.assertLessEqual(a.evaluations, b.evaluations)

    def test_failure_cases(self) -> None:
        """Invalid arguments, vanishing gradient, non-convergence."""
        problem = normal_input_limit_state(linear_g(6.0), 2)
        with self.assertRaises(TypeError):
            form_sensitivities("g", (0.0,), 1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            form_sensitivities(problem, (1.0, 2.0, 0.5, 0.8), 2, start=(0.0,))
        with self.assertRaises(ValueError):
            form_sensitivities(problem, (1.0, 2.0, 0.5, -0.8), 2, start=(0.1, 0.1))
        with self.assertRaises(ValueError):
            form_sensitivities(problem, (1.0, 2.0, 0.5), 2, start=(0.1, 0.1))
        with self.assertRaises(ValueError):
            form_sensitivities(lambda u, th: 1.0 + 0.0 * th[0], (1.0,), 2)
        with self.assertRaises(ValueError):
            form_sensitivities(problem, (1.0, 2.0, 0.5, 0.8), 2, tolerance=0.0)
        with self.assertRaises(TypeError):
            normal_input_limit_state(3.0, 2)  # type: ignore[arg-type]
        hard = form_sensitivities(
            lambda u, th: th[0] - u[0] - 0.5 * u[1] ** 2,
            (3.0,),
            2,
            start=(0.5, 0.5),
            max_iterations=1,
        )
        self.assertFalse(hard.converged)
        with self.assertRaises(ValueError):
            form_sensitivities_finite_difference(
                lambda u, th: th[0] - u[0] - 0.5 * u[1] ** 2,
                (3.0,),
                2,
                start=(0.5, 0.5),
                max_iterations=1,
            )


class ScoreFunctionTests(unittest.TestCase):
    """Score-function estimators against closed-form derivatives."""

    def test_normal_scores_values_and_mean_zero(self) -> None:
        """Hand values; the score of a regular family has mean zero."""
        score = normal_scores((3.0,), (1.0,), (2.0,))
        self.assertAlmostEqual(score[0], 2.0 / 4.0, 14)
        self.assertAlmostEqual(score[1], (1.0 - 1.0) / 2.0, 14)
        # exact E[score] = 0 by Gauss-Hermite quadrature
        nodes, weights = gauss_hermite(20)
        for k in range(2):
            value = sum(
                w * normal_scores((1.5 + 0.7 * z,), (1.5,), (0.7,))[k]
                for z, w in zip(nodes, weights, strict=True)
            )
            self.assertAlmostEqual(value, 0.0, 10)

    def test_estimator_on_hand_case(self) -> None:
        """Mean of 1_F s over four samples, by hand."""
        est, se = score_function_estimator(
            [True, False, True, False], [(2.0,), (9.0,), (4.0,), (9.0,)]
        )
        self.assertAlmostEqual(est[0], 1.5, 14)
        # values 2,0,4,0: sum of squares 20, so (20/4 - 2.25)/(4 - 1) = 2.75/3
        self.assertAlmostEqual(se[0], math.sqrt(2.75 / 3.0), 14)

    def test_monte_carlo_is_unbiased_within_mcse(self) -> None:
        """All four derivatives within 4 standard errors of the closed form."""
        r, mean, std = 1.6, (0.0, 0.0), (1.0, 1.0)
        probability, truth = linear_truth(r, mean, std)
        result = score_function_monte_carlo(
            linear_g(r), mean, std, samples=60000, seed=3
        )
        self.assertTrue(result.converged)
        self.assertLess(
            abs(result.probability - probability),
            4.0 * result.probability_standard_error,
        )
        self.assertEqual(result.parameter_names, normal_parameter_names(2))
        for est, se, want in zip(
            result.derivatives, result.standard_errors, truth, strict=True
        ):
            self.assertLess(abs(est - want), 4.0 * se)
            self.assertLess(se / abs(want), 0.1)  # useful precision at this N

    def test_monte_carlo_unbiasedness_over_replicates(self) -> None:
        """The replicate mean of dP/dmu_1 is unbiased and the reported SE honest."""
        r, mean, std = 1.2, (0.5, -0.3), (0.8, 1.1)
        _, truth = linear_truth(r, mean, std)
        runs = [
            score_function_monte_carlo(linear_g(r), mean, std, samples=3000, seed=s)
            for s in range(40)
        ]
        estimates = [run.derivatives[0] for run in runs]
        spread = statistics.stdev(estimates)
        reported = statistics.fmean(run.standard_errors[0] for run in runs)
        self.assertLess(
            abs(statistics.fmean(estimates) - truth[0]), 4.0 * spread / math.sqrt(40)
        )
        self.assertGreater(spread / reported, 0.7)
        self.assertLess(spread / reported, 1.4)

    def test_subset_simulation_rare_event_replicates(self) -> None:
        """P_f = 1e-4 (z = 3.72): replicate means within 4 SE; SE honest."""
        scale = math.sqrt(A[0] ** 2 + A[1] ** 2)
        mean, std = (0.0, 0.0), (1.0, 1.0)
        r = 3.72 * scale
        probability, truth = linear_truth(r, mean, std)
        runs = [
            score_function_subset(
                linear_g(r), mean, std, samples_per_level=1000, seed=s
            )
            for s in range(24)
        ]
        self.assertTrue(all(run.converged for run in runs))
        self.assertGreaterEqual(min(run.levels for run in runs), 4)
        for k in (0, 1, 2, 3):
            estimates = [run.derivatives[k] for run in runs]
            spread = statistics.stdev(estimates)
            mean_est = statistics.fmean(estimates)
            self.assertLess(
                abs(mean_est - truth[k]), 4.0 * spread / math.sqrt(len(runs))
            )
            reported = statistics.fmean(run.standard_errors[k] for run in runs)
            self.assertGreater(spread / reported, 0.4)
            self.assertLess(spread / reported, 2.5)
        pf = statistics.fmean(run.probability for run in runs)
        self.assertAlmostEqual(math.log(pf / probability), 0.0, delta=0.25)

    def test_subset_reduces_to_monte_carlo_when_event_is_common(self) -> None:
        """A failure fraction above p0 stops at level 0 (one level, N evaluations)."""
        result = score_function_subset(
            linear_g(0.2), (0.0, 0.0), (1.0, 1.0), samples_per_level=2000, seed=2
        )
        self.assertEqual(result.levels, 1)
        self.assertEqual(result.evaluations, 2000)
        _, truth = linear_truth(0.2, (0.0, 0.0), (1.0, 1.0))
        for est, se, want in zip(
            result.derivatives, result.standard_errors, truth, strict=True
        ):
            self.assertLess(abs(est - want), 4.5 * se)

    def test_reproducible_with_seed(self) -> None:
        """Equal seeds give identical results."""
        a = score_function_subset(
            linear_g(4.0), (0.0, 0.0), (1.0, 1.0), samples_per_level=200, seed=7
        )
        b = score_function_subset(
            linear_g(4.0), (0.0, 0.0), (1.0, 1.0), samples_per_level=200, seed=7
        )
        self.assertEqual(a.derivatives, b.derivatives)

    def test_failure_cases(self) -> None:
        """No failures, invalid options, and inconsistent inputs are reported."""
        none = score_function_monte_carlo(
            linear_g(50.0), (0.0, 0.0), (1.0, 1.0), samples=100, seed=1
        )
        self.assertFalse(none.converged)
        self.assertEqual(none.derivatives, (0.0,) * 4)
        with self.assertRaises(ValueError):
            score_function_monte_carlo(linear_g(1.0), (0.0,), (1.0, 1.0), samples=10)
        with self.assertRaises(ValueError):
            score_function_monte_carlo(
                linear_g(1.0), (0.0, 0.0), (1.0, 0.0), samples=10
            )
        with self.assertRaises(ValueError):
            score_function_monte_carlo(linear_g(1.0), (0.0, 0.0), (1.0, 1.0), samples=1)
        with self.assertRaises(TypeError):
            score_function_monte_carlo(
                linear_g(1.0),
                (0.0, 0.0),
                (1.0, 1.0),
                samples=10.0,  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            score_function_monte_carlo(5.0, (0.0,), (1.0,), samples=10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            score_function_subset(
                linear_g(1.0), (0.0, 0.0), (1.0, 1.0), samples_per_level=105
            )
        with self.assertRaises(ValueError):
            score_function_subset(
                linear_g(1.0), (0.0, 0.0), (1.0, 1.0), correlation=1.0
            )
        with self.assertRaises(ValueError):
            score_function_subset(
                lambda x: 1.0, (0.0,), (1.0,), samples_per_level=100, seed=1
            )  # ties at the first threshold
        with self.assertRaises(ValueError):
            score_function_estimator([True], [(1.0,)])
        with self.assertRaises(ValueError):
            normal_scores((1.0,), (0.0, 0.0), (1.0, 1.0))

    def test_unreached_failure_is_flagged(self) -> None:
        """Stopping at max_levels before p0 N failures sets converged False."""
        result = score_function_subset(
            linear_g(8.0),
            (0.0, 0.0),
            (1.0, 1.0),
            samples_per_level=200,
            max_levels=2,
            seed=1,
        )
        self.assertFalse(result.converged)


def sobol_oracle(r: float, a: tuple[float, ...], index: int) -> float:
    """Return ``Var(E[1_F | X_i]) / Var(1_F)`` for ``F = {a.X >= r}``.

    Inputs are i.i.d. standard normal. Given ``X_i = x`` the failure
    probability is ``Phi((a_i x - r) / s_rest)``, integrated by Gauss-Hermite.
    """
    total = math.sqrt(sum(v * v for v in a))
    rest = math.sqrt(total * total - a[index] ** 2)
    p = cdf(-r / total)
    if rest == 0.0:
        raise ValueError("single-input problem")
    nodes, weights = gauss_hermite(40)
    second = sum(
        w * cdf((a[index] * z - r) / rest) ** 2
        for z, w in zip(nodes, weights, strict=True)
    )
    return (second - p * p) / (p * (1.0 - p))


class FailureSobolTests(unittest.TestCase):
    """First-order indices of the failure indicator."""

    def test_linear_gaussian_matches_quadrature_oracle(self) -> None:
        """Three inputs, the third inactive; indices within bootstrap bands."""
        a = (1.0, 0.5, 0.0)
        r = 0.4
        rng = random.Random(8)
        n = 10000
        samples = [tuple(rng.gauss(0.0, 1.0) for _ in range(3)) for _ in range(n)]
        failed = [
            sum(ai * xi for ai, xi in zip(a, x, strict=True)) >= r for x in samples
        ]
        result = failure_indicator_sobol(
            samples, failed, bins=8, n_bootstrap=25, seed=1
        )
        for i in (0, 1):
            want = sobol_oracle(r, a[:2], i)
            # 4 bootstrap SE plus 0.02 for the within-bin smoothing bias
            self.assertLess(
                abs(result.first_order[i] - want),
                4.0 * result.standard_errors[i] + 0.02,
            )
        self.assertLess(
            abs(result.first_order[2]), 4.0 * result.standard_errors[2] + 0.01
        )
        self.assertGreater(result.first_order[0], 2.0 * result.first_order[1])
        self.assertAlmostEqual(result.failure_probability, sum(failed) / n, 14)
        self.assertEqual(result.bins, 8)

    def test_oracle_hand_check_for_symmetric_case(self) -> None:
        """Equal weights give equal indices; a_i x_i sum is symmetric."""
        value_a = sobol_oracle(0.5, (1.0, 1.0), 0)
        value_b = sobol_oracle(0.5, (1.0, 1.0), 1)
        self.assertAlmostEqual(value_a, value_b, 12)
        self.assertGreater(value_a, 0.0)
        self.assertLess(value_a, 1.0)

    def test_failure_cases(self) -> None:
        """Constant indicators, small samples, and shape errors are rejected."""
        rng = random.Random(1)
        samples = [(rng.random(), rng.random()) for _ in range(100)]
        with self.assertRaises(ValueError):
            failure_indicator_sobol(samples, [False] * 100)
        with self.assertRaises(ValueError):
            failure_indicator_sobol(samples[:10], [True, False] * 5)
        with self.assertRaises(ValueError):
            failure_indicator_sobol(samples, [True, False] * 49)
        with self.assertRaises(ValueError):
            failure_indicator_sobol([(1.0,), (1.0, 2.0)] * 20, [True, False] * 20)
        with self.assertRaises(ValueError):
            failure_indicator_sobol(samples, [True, False] * 50, bins=1)
        with self.assertRaises(TypeError):
            failure_indicator_sobol(samples, [True, False] * 50, bins=2.5)  # type: ignore[arg-type]
        result = failure_indicator_sobol(
            samples, [x[0] > 0.7 for x in samples], bins=4, n_bootstrap=0
        )
        self.assertTrue(math.isnan(result.standard_errors[0]))


if __name__ == "__main__":
    unittest.main()
