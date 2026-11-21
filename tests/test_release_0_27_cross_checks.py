"""Integrator's cross-checks of the 0.27 work packages (ADR-0117 to ADR-0130).

Each work package was written and tested by its own author. The checks here
were written afterwards by the integrator, without reading those tests, from
oracles that can be stated in one line: a textbook closed form, an algebraic
identity, or exact enumeration. They are deliberately few and independent;
a failure here means a package and its own tests agree on something wrong.
"""

from __future__ import annotations

import itertools
import math
import random
import unittest

from probstat.causal import omitted_variable_bias, robustness_value
from probstat.conformal import conformal_quantile, weighted_conformal_quantile
from probstat.design import improvement_function, log_improvement_function
from probstat.distributions import Normal
from probstat.generative import GaussianMarginalVelocity, integrate_flow
from probstat.inference import (
    adjust_p_values,
    betting_mean_test,
    e_benjamini_hochberg,
    pareto_smooth,
)
from probstat.inference.survival import cumulative_incidence
from probstat.information import sliced_wasserstein
from probstat.inverse import (
    CalibrationProblem,
    Experiment,
    NoiseModel,
    unscented_kalman_inversion,
)
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.monte_carlo import mfmc_variance, mlblue_variance
from probstat.numerics import integrate
from probstat.reliability import ModelLimitState, improved_cross_entropy
from probstat.risk import FiniteMDP, value_iteration
from probstat.sampling import lattice_squared_error, rqmc_mean
from probstat.sensitivity import rank_sobol_indices
from probstat.statistics import (
    chatterjee_correlation,
    distance_correlation,
    ledoit_wolf_covariance,
)
from probstat.surrogates import GaussianMeasure, kernel_mean
from probstat.timeseries import hierarchy_summing_matrix, reconcile_forecasts
from probstat.uq import QuantityOfInterest


class MonteCarloCrossChecks(unittest.TestCase):
    """Multifidelity variances against the textbook control variate."""

    def test_two_model_mfmc_is_the_control_variate_with_estimated_mean(self) -> None:
        """``Var = s^2 (1 - rho^2 (1 - n/m)) / n`` at the optimal coefficient."""
        rho, n, m = 0.9, 20.0, 400.0
        covariance = ((1.0, rho), (rho, 1.0))
        exact = (1.0 - rho * rho * (1.0 - n / m)) / n
        self.assertAlmostEqual(mfmc_variance(covariance, (n, m)), exact, places=12)

    def test_mlblue_with_one_group_is_the_sample_mean(self) -> None:
        """A single group holding only the target model gives ``s^2 / n``."""
        covariance = ((2.0, 0.5), (0.5, 1.0))
        self.assertAlmostEqual(
            mlblue_variance(covariance, [(0,)], (25,)), 2.0 / 25, places=12
        )

    def test_mlblue_two_groups_matches_generalized_least_squares(self) -> None:
        """Groups {0,1} and {1}: variance from the 2x2 normal equations."""
        rho, n, extra = 0.8, 10.0, 90.0
        covariance = ((1.0, rho), (rho, 1.0))
        # Psi = n C^-1 + extra e_1 e_1'; invert the 2x2 matrix by hand.
        det = 1.0 - rho * rho
        a, b, d = n / det, -n * rho / det, n / det + extra
        exact = d / (a * d - b * b)
        got = mlblue_variance(covariance, [(0, 1), (1,)], (n, extra))
        self.assertAlmostEqual(got, exact, places=12)
        # The same design is the control variate with n + extra control draws.
        total = n + extra
        self.assertAlmostEqual(
            exact, (1.0 - rho * rho * (1.0 - n / total)) / n, places=12
        )


class InferenceCrossChecks(unittest.TestCase):
    """E-values, betting capital, and Pareto smoothing."""

    def test_e_bh_on_reciprocal_p_values_is_bh(self) -> None:
        """With ``e = 1/p`` the two procedures reject the same hypotheses."""
        generator = random.Random(3)
        for _ in range(50):
            p_values = [generator.random() ** 3 for _ in range(12)]
            bh = adjust_p_values(p_values, method="bh", alpha=0.1)
            ebh = e_benjamini_hochberg([1.0 / p for p in p_values], alpha=0.1)
            self.assertEqual(tuple(bh.rejected), tuple(ebh.rejected))

    def test_fixed_bet_capital_is_a_martingale_by_enumeration(self) -> None:
        """Over all 2^8 Bernoulli(0.3) paths the mean capital is one."""
        mean = 0.0
        for path in itertools.product((0.0, 1.0), repeat=8):
            ones = sum(path)
            probability = 0.3**ones * 0.7 ** (8 - ones)
            result = betting_mean_test(
                path, 0.3, bet="fixed", fixed_bet=0.6, alternative="greater"
            )
            direct = math.prod(1.0 + 0.6 * (x - 0.3) for x in path)
            self.assertAlmostEqual(result.e_value, direct, places=10)
            mean += probability * result.e_value
        self.assertAlmostEqual(mean, 1.0, places=12)

    def test_pareto_k_matches_the_gaussian_tail_exponent(self) -> None:
        """Proposal N(0,1), target N(0, 1.5^2): tail shape ``1 - 1/1.5^2``."""
        generator = random.Random(11)
        scale = 1.5
        draws = [generator.gauss(0.0, 1.0) for _ in range(40000)]
        log_ratios = [0.5 * x * x * (1.0 - 1.0 / scale**2) for x in draws]
        self.assertAlmostEqual(
            pareto_smooth(log_ratios).pareto_k, 1.0 - 1.0 / scale**2, delta=0.1
        )


class ConformalAndCausalCrossChecks(unittest.TestCase):
    """Reductions and closed forms."""

    def test_unit_weights_reduce_to_split_conformal(self) -> None:
        """Equal weights give the ordinary conformal quantile."""
        generator = random.Random(5)
        scores = [generator.random() for _ in range(37)]
        for alpha in (0.05, 0.1, 0.3):
            self.assertEqual(
                weighted_conformal_quantile(scores, [1.0] * 37, 1.0, alpha),
                conformal_quantile(scores, alpha),
            )

    def test_robustness_value_brings_the_estimate_to_zero(self) -> None:
        """A confounder of strength RV has bias ``|t| se`` exactly."""
        t_value, df, se = 3.2, 80, 0.25
        rv = robustness_value(t_value, df)
        f2 = t_value**2 / df
        self.assertAlmostEqual(rv * rv + f2 * rv - f2, 0.0, places=12)
        self.assertAlmostEqual(
            omitted_variable_bias(se, df, rv, rv), abs(t_value) * se, places=12
        )


class SensitivityAndDependenceCrossChecks(unittest.TestCase):
    """Closed-form indices and coefficients."""

    def test_rank_sobol_on_a_linear_gaussian_model(self) -> None:
        """``Y = 3 X1 + X2``: indices 0.9 and 0.1."""
        generator = random.Random(8)
        inputs = [
            (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)) for _ in range(6000)
        ]
        outputs = [3.0 * a + b for a, b in inputs]
        values = rank_sobol_indices(inputs, outputs).first_order
        self.assertAlmostEqual(values[0], 0.9, delta=0.04)
        self.assertAlmostEqual(values[1], 0.1, delta=0.04)

    def test_chatterjee_and_distance_correlation_limits(self) -> None:
        """Monotone data: ``xi = (n-2)/(n+1)``; linear data: dCor one."""
        x = [float(i) for i in range(1, 41)]
        y = [math.exp(0.1 * value) for value in x]
        self.assertAlmostEqual(
            chatterjee_correlation(x, y).statistic,
            38.0 / 41.0,
            places=12,
        )
        linear = [2.0 - 3.0 * value for value in x]
        result = distance_correlation(x, linear)
        self.assertAlmostEqual(result.distance_correlation, 1.0, places=10)

    def test_ledoit_wolf_is_positive_definite_when_p_exceeds_n(self) -> None:
        """Five observations in eight dimensions: every pivot is positive."""
        generator = random.Random(2)
        data = [[generator.gauss(0.0, 1.0) for _ in range(8)] for _ in range(5)]
        matrix = [list(row) for row in ledoit_wolf_covariance(data).covariance]
        for k in range(8):
            self.assertGreater(matrix[k][k], 0.0)
            for i in range(k + 1, 8):
                factor = matrix[i][k] / matrix[k][k]
                for j in range(k, 8):
                    matrix[i][j] -= factor * matrix[k][j]


class DecisionAndOptimizationCrossChecks(unittest.TestCase):
    """Geometric series and the improvement function."""

    def test_two_state_chain_value_by_hand(self) -> None:
        """Stay (reward 1) or leave to an absorbing zero state."""
        mdp = FiniteMDP(
            transitions=(
                ((1.0, 0.0), (0.0, 1.0)),
                ((0.0, 1.0), (0.0, 1.0)),
            ),
            rewards=((1.0, 0.0), (3.0, 0.0)),
            discount=0.8,
        )
        solution = value_iteration(mdp, tolerance=1e-12)
        # Staying is worth 1/(1-0.8) = 5; leaving is worth 3.
        self.assertAlmostEqual(solution.values[0], 5.0, places=8)
        self.assertEqual(solution.policy[0], 0)

    def test_improvement_function_against_quadrature(self) -> None:
        """``h(z) = E[max(z + Z, 0)]`` by a Riemann sum; the log is finite."""
        normal = Normal(0.0, 1.0)
        for z in (-2.0, -0.5, 0.0, 1.3):
            step = 1e-3
            total = sum(
                max(z + (-9.0 + k * step), 0.0) * normal.pdf(-9.0 + k * step) * step
                for k in range(18001)
            )
            self.assertAlmostEqual(improvement_function(z), total, places=5)
            self.assertAlmostEqual(
                log_improvement_function(z), math.log(total), places=4
            )
        self.assertTrue(math.isfinite(log_improvement_function(-45.0)))
        self.assertLess(log_improvement_function(-45.0), -1000.0)


class TransportFlowAndQuadratureCrossChecks(unittest.TestCase):
    """Translations, exact flows, lattices, and coherent forecasts."""

    def test_sliced_wasserstein_of_a_translation(self) -> None:
        """Shift by ``t`` in two dimensions: ``SW_2^2`` near ``|t|^2 / 2``."""
        generator = random.Random(4)
        cloud = [
            (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)) for _ in range(60)
        ]
        moved = [(a + 3.0, b - 1.0) for a, b in cloud]
        result = sliced_wasserstein(cloud, moved, directions=400, seed=1)
        self.assertAlmostEqual(result.power, 5.0, delta=0.5)

    def test_exact_flow_transports_the_standard_normal(self) -> None:
        """The closed-form field maps ``z`` to ``m + s z``."""
        field = GaussianMarginalVelocity((2.0,), (0.5,))
        for z in (-1.5, 0.0, 0.7):
            end = integrate_flow(field, (z,), steps=40)
            self.assertAlmostEqual(end[0], 2.0 + 0.5 * z, places=5)

    def test_one_dimensional_lattice_error_and_exact_integration(self) -> None:
        """``e^2 = 1/(6 n^2)`` for unit weight; a linear integrand is exact."""
        self.assertAlmostEqual(
            lattice_squared_error(31, (1,)), 1.0 / (6.0 * 31**2), places=14
        )
        result = rqmc_mean(
            lambda u: (1.0 + 0.8 * (u[0] - 0.5)) * (1.0 - 0.4 * (u[1] - 0.5)),
            2,
            127,
            replicates=8,
            seed=3,
        )
        self.assertAlmostEqual(result.estimate, 1.0, delta=5e-3)
        self.assertEqual(result.evaluations, 8 * 127)

    def test_reconciled_forecasts_add_up(self) -> None:
        """OLS reconciliation of ``(total, a, b)`` by hand: project on S."""
        structure = hierarchy_summing_matrix({"total": ("a", "b")})
        result = reconcile_forecasts((10.0, 3.0, 4.0), structure, method="ols")
        values = list(result.reconciled)
        self.assertAlmostEqual(values[0], values[1] + values[2], places=12)
        # Least squares: bottom = base_bottom + (total - a - b)/3 each.
        self.assertAlmostEqual(values[1], 4.0, places=10)
        self.assertAlmostEqual(values[2], 5.0, places=10)


class ReliabilityInverseSurvivalSurrogateCrossChecks(unittest.TestCase):
    """One closed form for each remaining package."""

    def test_improved_cross_entropy_on_a_linear_limit_state(self) -> None:
        """``g = 3.5 - (u1 + u2)/sqrt(2)``: ``P_f = Phi(-3.5)``."""
        inputs = JointInput(
            (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0)))
        )
        model = ProbabilisticModel(
            inputs,
            lambda p: {"y": 3.5 - (p["a"] + p["b"]) / math.sqrt(2.0)},
            (QuantityOfInterest("y", "margin"),),
        )
        state = ModelLimitState(model, "y", 0.0, "below", "linear")
        exact = Normal(0.0, 1.0).cdf(-3.5)
        for seed in (1, 2, 3):
            result = improved_cross_entropy(state, seed=seed)
            self.assertTrue(result.converged)
            self.assertLess(
                abs(result.probability - exact), 5.0 * result.standard_error
            )
            self.assertLess(result.coefficient_of_variation, 0.15)

    def test_unscented_inversion_solves_a_conjugate_problem(self) -> None:
        """One unknown mean, prior N(0, 2^2), four readings with sd 0.5."""
        readings = (1.2, 0.7, 1.5, 1.0)
        prior = JointInput.independent(RandomInput("m", Normal(0.0, 2.0)))
        model = ProbabilisticModel(
            prior, lambda p: {"y": p["m"]}, (QuantityOfInterest("y", "reading"),)
        )
        problem = CalibrationProblem(
            model,
            tuple(Experiment({"y": value}, {}) for value in readings),
            (NoiseModel("y", standard_deviation=0.5),),
        )
        result = unscented_kalman_inversion(problem, iterations=200, tolerance=1e-12)
        precision = 1.0 / 4.0 + len(readings) / 0.25
        mean = (sum(readings) / 0.25) / precision
        self.assertAlmostEqual(result.posterior_mean[0], mean, places=6)
        self.assertAlmostEqual(
            result.posterior_covariance[0][0], 1.0 / precision, places=6
        )

    def test_cumulative_incidence_without_censoring_is_a_proportion(self) -> None:
        """Six subjects, two causes: the incidence is the empirical share."""
        times = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        causes = (1, 2, 1, 1, 2, 1)
        first = cumulative_incidence(times, causes, cause=1)
        second = cumulative_incidence(times, causes, cause=2)
        self.assertAlmostEqual(first.incidence[-1], 4.0 / 6.0, places=12)
        self.assertAlmostEqual(second.incidence[-1], 2.0 / 6.0, places=12)
        for a, b, survival in zip(
            first.incidence, second.incidence, first.overall_survival, strict=True
        ):
            self.assertAlmostEqual(a + b, 1.0 - survival, places=12)

    def test_kernel_mean_against_quadrature(self) -> None:
        """Integral of a squared-exponential kernel against N(0.3, 0.8^2)."""
        measure = GaussianMeasure(0.3, 0.8)
        density = Normal(0.3, 0.8)
        for x, length in ((0.0, 0.5), (1.1, 1.3)):
            reference = integrate(
                lambda t, x=x, length=length: (
                    math.exp(-0.5 * ((t - x) / length) ** 2) * density.pdf(t)
                ),
                -8.0,
                8.0,
            ).value
            self.assertAlmostEqual(kernel_mean(measure, x, length), reference, places=8)


if __name__ == "__main__":
    unittest.main()
