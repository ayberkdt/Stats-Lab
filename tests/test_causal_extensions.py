"""Tests for the quasi-experimental and debiased causal estimators.

Oracles are independent of the code under test:

- omitted-variable bias: a simulated confounder is actually omitted and the
  long and short regressions are fitted with the library's linear model;
  the partial-R-squared formulas must reproduce the observed change
  exactly (they are algebraic identities);
- double ML: the Frisch-Waugh-Lovell theorem (no cross-fitting, linear
  nuisances) and seeded coverage simulations;
- difference-in-differences: the four-means formula, noise-free panels in
  which every ``ATT(g, t)`` is known by construction, and Monte Carlo
  checks of the influence-function standard errors;
- synthetic control: duplicated donors, exact convex combinations, the
  Karush-Kuhn-Tucker conditions and a brute-force grid;
- instrumental variables: the Wald ratio and the closed form
  ``(Z'X)^-1 Z'y``.
"""

from __future__ import annotations

import math
import random
import statistics
import unittest
import warnings

from probstat.causal import (
    PositivityWarning,
    adjust_for_confounder,
    did_two_by_two,
    dml_average_treatment_effect,
    dml_partially_linear,
    estimate_aipw,
    group_time_effects,
    knn_learner,
    linear_learner,
    logistic_learner,
    omitted_variable_bias,
    panel_from_long,
    partial_r2,
    project_to_simplex,
    ridge_learner,
    robustness_value,
    sensitivity_statistics,
    synthetic_control,
    synthetic_control_placebo,
    synthetic_control_weights,
    two_stage_least_squares,
)
from probstat.distributions import StudentT
from probstat.inference import RankDeficientDesignError, fit_linear_model
from probstat.linalg import lu_solve, matmul, transpose


def confounded_regression(
    n: int, seed: int
) -> tuple[list[tuple[float, float]], list[float], list[float], list[float]]:
    """Simulate covariates ``X``, a confounder ``Z``, treatment and outcome."""
    rng = random.Random(seed)
    x = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n)]
    z = [0.5 * row[0] + rng.gauss(0, 1) for row in x]
    d = [
        0.7 * zi + 0.3 * row[1] + rng.gauss(0, 1) for zi, row in zip(z, x, strict=False)
    ]
    y = [
        1.0 * di + 0.8 * zi + 0.4 * row[0] + rng.gauss(0, 1)
        for di, zi, row in zip(d, z, x, strict=False)
    ]
    return x, z, d, y


class OmittedVariableBiasTests(unittest.TestCase):
    """Check the partial-R-squared algebra against real omitted regressors."""

    def setUp(self) -> None:
        """Fit the long, short and treatment regressions of one data set."""
        x, z, d, y = confounded_regression(120, 1)
        self.short = fit_linear_model(
            [(1, di, *row) for di, row in zip(d, x, strict=False)], y
        )
        self.long = fit_linear_model(
            [(1, di, *row, zi) for di, row, zi in zip(d, x, z, strict=False)], y
        )
        treatment = fit_linear_model(
            [(1, *row, zi) for row, zi in zip(x, z, strict=False)], d
        )
        self.r2_outcome = partial_r2(self.long.t_values[-1], self.long.df_residual)
        self.r2_treatment = partial_r2(treatment.t_values[-1], treatment.df_residual)
        self.without_d = fit_linear_model([(1, *row) for row in x], y)

    def test_partial_r2_matches_the_drop_in_residual_sum_of_squares(self) -> None:
        """``t^2 / (t^2 + df)`` equals ``(RSS_0 - RSS_1) / RSS_0``."""
        expected = (
            self.without_d.residual_sum_squares - self.short.residual_sum_squares
        ) / self.without_d.residual_sum_squares
        value = partial_r2(self.short.t_values[1], self.short.df_residual)
        self.assertAlmostEqual(value, expected, places=12)

    def test_bias_formula_reproduces_the_observed_coefficient_change(self) -> None:
        """The bias bound is an identity for an actually omitted regressor."""
        observed = self.short.coefficients[1] - self.long.coefficients[1]
        bias = omitted_variable_bias(
            self.short.standard_errors[1],
            self.short.df_residual,
            self.r2_outcome,
            self.r2_treatment,
        )
        self.assertGreater(abs(observed), 0.2)
        self.assertAlmostEqual(bias, abs(observed), places=12)

    def test_adjustment_reproduces_the_long_regression(self) -> None:
        """Adjusted estimate, standard error and t equal the long fit."""
        adjusted = adjust_for_confounder(
            self.short.coefficients[1],
            self.short.standard_errors[1],
            self.short.df_residual,
            r2_outcome=self.r2_outcome,
            r2_treatment=self.r2_treatment,
        )
        self.assertAlmostEqual(
            adjusted.adjusted_estimate, self.long.coefficients[1], places=12
        )
        self.assertAlmostEqual(
            adjusted.adjusted_standard_error, self.long.standard_errors[1], places=12
        )
        self.assertAlmostEqual(adjusted.adjusted_t, self.long.t_values[1], places=10)
        self.assertEqual(adjusted.adjusted_degrees_of_freedom, self.long.df_residual)

    def test_identity_holds_when_the_confounder_masks_the_effect(self) -> None:
        """With ``reduce=False`` the adjustment moves away from zero."""
        rng = random.Random(5)
        z = [rng.gauss(0, 1) for _ in range(80)]
        d = [zi + rng.gauss(0, 1) for zi in z]
        y = [di - 1.5 * zi + rng.gauss(0, 1) for di, zi in zip(d, z, strict=False)]
        short = fit_linear_model([(1, di) for di in d], y)
        long = fit_linear_model([(1, di, zi) for di, zi in zip(d, z, strict=False)], y)
        treatment = fit_linear_model([(1, zi) for zi in z], d)
        self.assertGreater(short.coefficients[1], 0.0)
        self.assertGreater(long.coefficients[1], short.coefficients[1])
        adjusted = adjust_for_confounder(
            short.coefficients[1],
            short.standard_errors[1],
            short.df_residual,
            r2_outcome=partial_r2(long.t_values[2], long.df_residual),
            r2_treatment=partial_r2(treatment.t_values[1], treatment.df_residual),
            reduce=False,
        )
        self.assertAlmostEqual(
            adjusted.adjusted_estimate, long.coefficients[1], places=12
        )

    def test_robustness_value_moves_the_estimate_to_the_target(self) -> None:
        """Equal strengths ``RV_q`` remove exactly the fraction ``q``."""
        estimate, error = self.short.coefficients[1], self.short.standard_errors[1]
        df = self.short.df_residual
        for q in (1.0, 0.5, 0.25):
            rv = robustness_value(estimate / error, df, q=q)
            adjusted = adjust_for_confounder(
                estimate, error, df, r2_outcome=rv, r2_treatment=rv
            )
            self.assertAlmostEqual(
                adjusted.adjusted_estimate, (1.0 - q) * estimate, places=12
            )
        f = abs(estimate / error) / math.sqrt(df)
        closed = 0.5 * (math.sqrt(f**4 + 4.0 * f**2) - f**2)
        self.assertAlmostEqual(robustness_value(estimate / error, df), closed, 12)

    def test_robustness_value_for_significance_hits_the_critical_value(self) -> None:
        """Equal strengths ``RV_{1,alpha}`` leave ``t`` at the critical value."""
        summary = sensitivity_statistics(
            self.short.coefficients[1],
            self.short.standard_errors[1],
            self.short.df_residual,
            alpha=0.05,
        )
        rv = summary.robustness_value_alpha
        self.assertLess(rv, summary.robustness_value)
        adjusted = adjust_for_confounder(
            summary.estimate,
            summary.standard_error,
            summary.degrees_of_freedom,
            r2_outcome=rv,
            r2_treatment=rv,
        )
        critical = StudentT(float(summary.degrees_of_freedom - 1)).quantile(0.975)
        self.assertAlmostEqual(adjusted.adjusted_t, critical, places=9)
        self.assertAlmostEqual(
            summary.partial_r2,
            partial_r2(summary.t_value, summary.degrees_of_freedom),
            places=15,
        )

    def test_insignificant_estimate_has_zero_significance_robustness(self) -> None:
        """A t statistic below the critical value gives ``RV_alpha = 0``."""
        self.assertEqual(robustness_value(1.0, 50, alpha=0.05), 0.0)
        self.assertGreater(robustness_value(1.0, 50), 0.0)
        self.assertEqual(robustness_value(0.0, 50), 0.0)

    def test_limits_of_the_bias(self) -> None:
        """No association with treatment or outcome means no bias."""
        self.assertEqual(omitted_variable_bias(0.2, 30, 0.0, 0.5), 0.0)
        self.assertEqual(omitted_variable_bias(0.2, 30, 0.5, 0.0), 0.0)
        extreme = adjust_for_confounder(1.0, 0.2, 30, r2_outcome=1.0, r2_treatment=0.1)
        self.assertEqual(extreme.adjusted_standard_error, 0.0)
        self.assertEqual(extreme.adjusted_t, math.inf)

    def test_invalid_inputs_are_rejected(self) -> None:
        """Out-of-range arguments raise ``ValueError``."""
        with self.assertRaises(ValueError):
            partial_r2(2.0, 0)
        with self.assertRaises(ValueError):
            omitted_variable_bias(0.1, 10, 0.5, 1.0)
        with self.assertRaises(ValueError):
            omitted_variable_bias(0.1, 10, 1.5, 0.2)
        with self.assertRaises(ValueError):
            omitted_variable_bias(-0.1, 10, 0.5, 0.2)
        with self.assertRaises(ValueError):
            robustness_value(2.0, 10, q=0.0)
        with self.assertRaises(ValueError):
            robustness_value(2.0, 10, alpha=1.0)
        with self.assertRaises(ValueError):
            robustness_value(2.0, 1, alpha=0.05)
        with self.assertRaises(ValueError):
            sensitivity_statistics(1.0, 0.0, 10)
        with self.assertRaises(ValueError):
            adjust_for_confounder(1.0, 0.1, 1, r2_outcome=0.1, r2_treatment=0.1)


def partially_linear_sample(
    n: int, seed: int
) -> tuple[list[tuple[float]], list[float], list[float]]:
    """Simulate ``Y = 0.5 D + g(X) + e`` with nonlinear ``g`` and ``m``."""
    rng = random.Random(seed)
    x = [(rng.uniform(-2, 2),) for _ in range(n)]
    d = [math.sin(2 * row[0]) + 0.5 * row[0] ** 2 + rng.gauss(0, 1) for row in x]
    y = [
        0.5 * di + math.cos(2 * row[0]) + row[0] ** 2 + rng.gauss(0, 1)
        for di, row in zip(d, x, strict=False)
    ]
    return x, d, y


class DoubleMachineLearningTests(unittest.TestCase):
    """Check the partially linear estimator and its learners."""

    def test_no_cross_fitting_with_linear_learners_is_least_squares(self) -> None:
        """Frisch-Waugh-Lovell: residual-on-residual equals the OLS slope."""
        x, _, d, y = confounded_regression(90, 3)
        ols = fit_linear_model(
            [(1, di, *row) for di, row in zip(d, x, strict=False)], y
        )
        result = dml_partially_linear(x, d, y, folds=1)
        self.assertFalse(result.cross_fitted)
        self.assertAlmostEqual(result.estimate, ols.coefficients[1], places=12)
        hc0 = math.sqrt(ols.robust_covariance("HC0")[1][1])
        self.assertAlmostEqual(result.standard_error, hc0, places=12)
        self.assertEqual(result.nuisance_fits, 2)

    def test_folds_are_balanced_seeded_and_reproducible(self) -> None:
        """Fold sizes differ by at most one and depend only on the seed."""
        x, _, d, y = confounded_regression(53, 4)
        first = dml_partially_linear(x, d, y, folds=5, seed=9)
        again = dml_partially_linear(x, d, y, folds=5, seed=9)
        other = dml_partially_linear(x, d, y, folds=5, seed=10)
        sizes = [first.fold_assignment.count(k) for k in range(5)]
        self.assertEqual(sorted(sizes), [10, 10, 11, 11, 11])
        self.assertEqual(first, again)
        self.assertNotEqual(first.fold_assignment, other.fold_assignment)
        self.assertTrue(first.cross_fitted)
        self.assertEqual(first.nuisance_fits, 10)
        self.assertEqual(len(first.outcome_residuals), 53)

    def test_built_in_learners_agree_with_direct_computation(self) -> None:
        """KNN averages the nearest targets; ridge(0) equals least squares."""
        rows = ((0.0,), (1.0,), (2.0,), (10.0,))
        targets = (1.0, 3.0, 5.0, 100.0)
        predict = knn_learner(2, standardize=False)(rows, targets)
        self.assertEqual(list(predict(((0.4,), (9.0,)))), [2.0, 52.5])
        everything = knn_learner(4)(rows, targets)
        self.assertAlmostEqual(everything(((3.0,),))[0], 27.25, places=12)
        wide = ((0.0, 0.0), (1.0, 500.0), (2.0, -500.0))
        scaled = knn_learner(1)(wide, (1.0, 2.0, 3.0))
        self.assertEqual(list(scaled(((0.9, 30.0),))), [1.0])
        unscaled = knn_learner(1, standardize=False)(wide, (1.0, 2.0, 3.0))
        self.assertEqual(list(unscaled(((1.1, 480.0),))), [2.0])
        x, _, d, _ = confounded_regression(40, 6)
        matrix = tuple(tuple(row) for row in x)
        linear = linear_learner()(matrix, tuple(d))(matrix[:5])
        ridge = ridge_learner(0.0)(matrix, tuple(d))(matrix[:5])
        for a, b in zip(linear, ridge, strict=False):
            self.assertAlmostEqual(a, b, places=10)

    def test_dml_covers_the_truth_while_the_naive_plug_in_is_biased(self) -> None:
        """Cross-fitted kNN double ML covers 0.5 at about the nominal rate.

        The naive estimator removes ``E[Y | X]`` from the outcome but does
        not residualize the treatment: it converges to
        ``theta var(D - m(X)) / E[D^2]``, not to ``theta``.
        """
        replicates, covered = 120, 0
        estimates: list[float] = []
        naive: list[float] = []
        errors: list[float] = []
        for rep in range(replicates):
            x, d, y = partially_linear_sample(300, rep)
            result = dml_partially_linear(
                x,
                d,
                y,
                outcome_learner=knn_learner(15),
                treatment_learner=knn_learner(15),
                folds=4,
                seed=rep,
            )
            low, high = result.confidence_interval()
            covered += low <= 0.5 <= high
            estimates.append(result.estimate)
            errors.append(result.standard_error)
            naive.append(
                math.fsum(
                    a * b for a, b in zip(d, result.outcome_residuals, strict=False)
                )
                / math.fsum(a * a for a in d)
            )
        coverage = covered / replicates
        self.assertGreaterEqual(coverage, 0.88)
        monte_carlo_error = statistics.stdev(estimates) / math.sqrt(replicates)
        self.assertLess(abs(statistics.fmean(estimates) - 0.5), 4 * monte_carlo_error)
        ratio = statistics.fmean(errors) / statistics.stdev(estimates)
        self.assertGreater(ratio, 0.8)
        self.assertLess(ratio, 1.25)
        self.assertLess(statistics.fmean(naive), 0.35)

    def test_in_sample_nearest_neighbour_fit_is_degenerate(self) -> None:
        """Without cross-fitting a 1-NN learner interpolates the treatment."""
        x, d, y = partially_linear_sample(40, 0)
        with self.assertRaisesRegex(ValueError, "perfectly"):
            dml_partially_linear(
                x,
                d,
                y,
                outcome_learner=knn_learner(1),
                treatment_learner=knn_learner(1),
                folds=1,
            )

    def test_invalid_inputs_and_learners_are_rejected(self) -> None:
        """Bad folds, shapes and learner outputs raise ``ValueError``."""
        x, _, d, y = confounded_regression(20, 7)
        with self.assertRaises(ValueError):
            dml_partially_linear(x, d, y, folds=0)
        with self.assertRaises(ValueError):
            dml_partially_linear(x, d, y, folds=21)
        with self.assertRaises(ValueError):
            dml_partially_linear(x, d[:-1], y)
        with self.assertRaises(ValueError):
            knn_learner(0)
        with self.assertRaises(ValueError):
            ridge_learner(-1.0)
        with self.assertRaisesRegex(ValueError, "predictions"):
            dml_partially_linear(
                x, d, y, outcome_learner=lambda rows, t: lambda new: [0.0], seed=1
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            dml_partially_linear(
                x,
                d,
                y,
                outcome_learner=lambda rows, t: lambda new: [math.nan] * len(new),
                seed=1,
            )
        with self.assertRaisesRegex(ValueError, "training rows"):
            dml_partially_linear(x, d, y, outcome_learner=knn_learner(50), seed=1)
        with self.assertRaises(ValueError):
            dml_partially_linear(x, d, y, seed=1).confidence_interval(1.5)


def binary_treatment_sample(
    n: int, seed: int
) -> tuple[list[tuple[float, float]], list[int], list[float]]:
    """Simulate a confounded binary treatment with a true ATE of 2."""
    rng = random.Random(seed)
    x = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n)]
    t = [int(rng.random() < 1.0 / (1.0 + math.exp(-(0.6 * a - 0.4 * b)))) for a, b in x]
    y = [
        2.0 * ti + a + 0.5 * b + rng.gauss(0, 1)
        for ti, (a, b) in zip(t, x, strict=False)
    ]
    return x, t, y


class CrossFittedAipwTests(unittest.TestCase):
    """Check the interactive-model (AIPW) estimator."""

    def test_single_fold_with_default_learners_is_estimate_aipw(self) -> None:
        """Without cross-fitting the result equals the parametric AIPW."""
        x, t, y = binary_treatment_sample(200, 2)
        reference = estimate_aipw(x, t, y)
        result = dml_average_treatment_effect(x, t, y, folds=1)
        self.assertAlmostEqual(result.estimate, reference.estimate, places=10)
        self.assertAlmostEqual(
            result.standard_error, reference.standard_error, places=10
        )
        self.assertEqual(result.method, "aipw-learners")

    def test_cross_fitted_estimate_is_close_to_the_truth(self) -> None:
        """The cross-fitted ATE is within four standard errors of 2."""
        x, t, y = binary_treatment_sample(400, 3)
        result = dml_average_treatment_effect(x, t, y, folds=5, seed=4)
        self.assertEqual(result.method, "aipw-crossfit")
        self.assertLess(abs(result.estimate - 2.0), 4 * result.standard_error)
        self.assertLess(result.standard_error, 0.2)
        again = dml_average_treatment_effect(
            x, t, y, folds=5, seed=4, propensity_learner=logistic_learner()
        )
        self.assertEqual(result, again)
        self.assertAlmostEqual(
            result.estimate, result.treated_mean - result.control_mean, places=12
        )

    def test_clipping_warns_and_bad_arguments_raise(self) -> None:
        """A regression learner outside ``(0, 1)`` is clipped with a warning."""
        x, t, y = binary_treatment_sample(120, 5)
        with self.assertWarns(PositivityWarning):
            result = dml_average_treatment_effect(
                x,
                t,
                y,
                propensity_learner=lambda rows, target: lambda new: [1.5] * len(new),
                folds=3,
                seed=1,
            )
        assert result.propensity_scores is not None
        self.assertEqual(set(result.propensity_scores), {0.99})
        with self.assertRaises(ValueError):
            dml_average_treatment_effect(x, t, y, clip=0.5)
        with self.assertRaises(ValueError):
            dml_average_treatment_effect(x, [2] * 120, y)
        with self.assertRaises(ValueError):
            dml_average_treatment_effect(x, t, y[:-1])


def staggered_effect(group: int, period: int) -> float:
    """Return the constructed effect of group ``group`` at ``period``."""
    exposure = period - group
    return (1.0 + 0.5 * (group == 2)) * (1.0 + exposure)


def staggered_panel(
    seed: int, per_group: int = 30, noise: float = 1.0
) -> tuple[list[list[float]], list[int | None]]:
    """Simulate six periods with adoption at 2 and 4 and a never-treated group.

    ``Y_it = a_i + 0.3 t + effect(g_i, t) + noise`` with the dynamic,
    group-specific effect of :func:`staggered_effect`.
    """
    rng = random.Random(seed)
    outcomes: list[list[float]] = []
    groups: list[int | None] = []
    for group in (2, 4, None):
        for _ in range(per_group):
            level = rng.gauss(0, 1) + (0.0 if group is None else float(group))
            row = []
            for period in range(6):
                effect = (
                    staggered_effect(group, period)
                    if group is not None and period >= group
                    else 0.0
                )
                row.append(level + 0.3 * period + effect + noise * rng.gauss(0, 1))
            outcomes.append(row)
            groups.append(group)
    return outcomes, groups


def two_way_fixed_effects(
    outcomes: list[list[float]], groups: list[int | None]
) -> float:
    """Return the static TWFE coefficient of the treatment dummy.

    Regression of ``Y_it`` on unit dummies, period dummies and
    ``1{t >= g_i}`` with the library's linear model.
    """
    n, width = len(outcomes), len(outcomes[0])
    design: list[list[float]] = []
    response: list[float] = []
    for i in range(n):
        for t in range(width):
            group = groups[i]
            row = [1.0 if group is not None and t >= group else 0.0]
            row += [1.0 if i == j else 0.0 for j in range(n)]
            row += [1.0 if t == s else 0.0 for s in range(1, width)]
            design.append(row)
            response.append(outcomes[i][t])
    return fit_linear_model(design, response).coefficients[0]


class DifferenceInDifferencesTests(unittest.TestCase):
    """Check the 2x2 estimator and the group-time effects."""

    def test_two_by_two_equals_the_four_means_formula(self) -> None:
        """The estimate is the textbook difference of mean changes."""
        pre = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 5.0]
        post = [2.0, 2.5, 5.0, 6.0, 9.0, 9.0, 11.0]
        treated = [0, 0, 0, 1, 1, 1, 1]
        result = did_two_by_two(pre, post, treated)
        four_means = (statistics.fmean(post[3:]) - statistics.fmean(pre[3:])) - (
            statistics.fmean(post[:3]) - statistics.fmean(pre[:3])
        )
        self.assertAlmostEqual(result.estimate, four_means, places=14)
        self.assertAlmostEqual(result.estimate, 4.5 - 3.5 / 3, places=14)
        d1 = [b - a for a, b in zip(pre[3:], post[3:], strict=False)]
        d0 = [b - a for a, b in zip(pre[:3], post[:3], strict=False)]
        error = math.sqrt(statistics.variance(d1) / 4 + statistics.variance(d0) / 3)
        self.assertAlmostEqual(result.standard_error, error, places=14)
        self.assertEqual((result.n_treated, result.n_control), (4, 3))
        low, high = result.confidence_interval(0.9)
        self.assertAlmostEqual(0.5 * (low + high), result.estimate, places=12)

    def test_two_period_group_time_effect_is_the_two_by_two(self) -> None:
        """One group and two periods reproduce ``did_two_by_two``."""
        rng = random.Random(8)
        pre = [rng.gauss(0, 1) for _ in range(40)]
        treated = [int(i < 15) for i in range(40)]
        post = [
            a + 0.4 + 1.5 * t + rng.gauss(0, 1)
            for a, t in zip(pre, treated, strict=False)
        ]
        reference = did_two_by_two(pre, post, treated)
        for comparison in ("never", "notyet"):
            result = group_time_effects(
                list(zip(pre, post, strict=False)),
                [1 if t else None for t in treated],
                comparison=comparison,
            )
            (cell,) = result.effects
            self.assertAlmostEqual(cell.estimate, reference.estimate, places=13)
            d1 = [b - a for a, b, t in zip(pre, post, treated, strict=False) if t]
            d0 = [b - a for a, b, t in zip(pre, post, treated, strict=False) if not t]
            error = math.sqrt(
                statistics.pvariance(d1) / 15 + statistics.pvariance(d0) / 25
            )
            self.assertAlmostEqual(cell.standard_error, error, places=13)
            self.assertAlmostEqual(result.overall().estimate, cell.estimate, 13)
            self.assertAlmostEqual(
                result.overall().standard_error, cell.standard_error, places=13
            )

    def test_noise_free_panel_recovers_every_effect_exactly(self) -> None:
        """Without noise each ``ATT(g, t)`` equals its constructed value."""
        outcomes, groups = staggered_panel(1, per_group=5, noise=0.0)
        for comparison in ("never", "notyet"):
            result = group_time_effects(outcomes, groups, comparison=comparison)
            self.assertEqual(len(result.effects), 6)
            self.assertEqual(result.skipped_cells, ())
            for cell in result.effects:
                truth = staggered_effect(int(cell.group), int(cell.period))
                self.assertAlmostEqual(cell.estimate, truth, places=12)
                self.assertAlmostEqual(cell.standard_error, 0.0, places=12)
                self.assertEqual(cell.exposure, int(cell.period - cell.group))
            self.assertEqual(result.effect(2, 5).n_treated, 5)
            self.assertEqual(
                result.effect(2, 3).n_comparison, 5 if comparison == "never" else 10
            )
            self.assertEqual(
                result.effect(2, 4).n_comparison, 5, "group 4 is treated at t = 4"
            )

    def test_aggregations_use_the_documented_weights(self) -> None:
        """Event-study and overall averages equal hand-computed sums."""
        outcomes, groups = staggered_panel(2, per_group=4, noise=0.0)
        outcomes, groups = outcomes[2:], groups[2:]  # group sizes 2, 4, 4
        result = group_time_effects(outcomes, groups)
        self.assertEqual(result.group_sizes, ((2.0, 2), (4.0, 4)))
        self.assertEqual(result.n_never_treated, 4)
        study = result.event_study()
        self.assertEqual([a.exposure for a in study], [0, 1, 2, 3])
        # exposures 0 and 1 mix both groups with weights 2/6 and 4/6
        for a in study[:2]:
            e = a.exposure
            assert e is not None
            expected = (
                2 * staggered_effect(2, 2 + e) + 4 * staggered_effect(4, 4 + e)
            ) / 6
            self.assertAlmostEqual(a.estimate, expected, places=12)
            self.assertAlmostEqual(sum(w for _, _, w in a.weights), 1.0, places=14)
        self.assertAlmostEqual(study[3].estimate, staggered_effect(2, 5), places=12)
        self.assertEqual(study[3].weights, ((2.0, 5.0, 1.0),))
        cells = [(2, t) for t in range(2, 6)] + [(4, 4), (4, 5)]
        sizes = {2: 2, 4: 4}
        total = sum(sizes[g] for g, _ in cells)
        expected = sum(sizes[g] * staggered_effect(g, t) for g, t in cells) / total
        overall = result.overall()
        self.assertAlmostEqual(overall.estimate, expected, places=12)
        self.assertEqual(overall.kind, "overall")
        self.assertIsNone(overall.exposure)

    def test_static_two_way_fixed_effects_misses_the_average_effect(self) -> None:
        """TWFE is far from the true average effect; ATT(g, t) is not."""
        outcomes, groups = staggered_panel(3, per_group=8, noise=0.0)
        truth = [staggered_effect(2, t) for t in range(2, 6)] + [
            staggered_effect(4, t) for t in (4, 5)
        ]
        true_average = statistics.fmean(truth)  # equal group sizes: 3.0
        self.assertAlmostEqual(true_average, 3.0, places=12)
        overall = group_time_effects(outcomes, groups).overall().estimate
        self.assertAlmostEqual(overall, true_average, places=12)
        twfe = two_way_fixed_effects(outcomes, groups)
        self.assertLess(twfe, true_average - 0.5)

    def test_noisy_effects_are_recovered_within_sampling_error(self) -> None:
        """Every cell and aggregate lies within four standard errors."""
        outcomes, groups = staggered_panel(11, per_group=60)
        for comparison in ("never", "notyet"):
            result = group_time_effects(outcomes, groups, comparison=comparison)
            for cell in result.effects:
                truth = staggered_effect(int(cell.group), int(cell.period))
                self.assertLess(abs(cell.estimate - truth), 4 * cell.standard_error)
                low, high = cell.confidence_interval()
                self.assertLess(low, cell.estimate)
                self.assertLess(cell.estimate, high)
            overall = result.overall()
            self.assertLess(abs(overall.estimate - 3.0), 4 * overall.standard_error)
            self.assertLess(overall.standard_error, 0.3)

    def test_influence_function_errors_match_the_sampling_spread(self) -> None:
        """Mean reported SE agrees with the Monte Carlo SD of each estimate.

        Group membership is redrawn in every replicate so that the
        group-share term of the aggregated influence function matters; the
        target of the overall average then varies with the realized
        shares, which the comparison accounts for by using its limit.
        """
        replicates = 150
        cell_estimates: list[float] = []
        cell_errors: list[float] = []
        overall_estimates: list[float] = []
        overall_errors: list[float] = []
        for rep in range(replicates):
            rng = random.Random(1000 + rep)
            outcomes: list[list[float]] = []
            groups: list[int | None] = []
            for _ in range(90):
                group = rng.choice((2, 4, None))
                level = rng.gauss(0, 1)
                outcomes.append(
                    [
                        level
                        + 0.3 * t
                        + (
                            staggered_effect(group, t)
                            if group is not None and t >= group
                            else 0.0
                        )
                        + rng.gauss(0, 1)
                        for t in range(6)
                    ]
                )
                groups.append(group)
            result = group_time_effects(outcomes, groups, comparison="notyet")
            cell = result.effect(2, 3)
            cell_estimates.append(cell.estimate)
            cell_errors.append(cell.standard_error)
            overall = result.overall()
            overall_estimates.append(overall.estimate)
            overall_errors.append(overall.standard_error)
        for estimates, errors in (
            (cell_estimates, cell_errors),
            (overall_estimates, overall_errors),
        ):
            ratio = statistics.fmean(errors) / statistics.stdev(estimates)
            self.assertGreater(ratio, 0.85)
            self.assertLess(ratio, 1.15)
        centre = statistics.fmean(overall_estimates)
        spread = statistics.stdev(overall_estimates) / math.sqrt(replicates)
        self.assertLess(abs(centre - 3.0), 4 * spread)

    def test_not_yet_treated_without_never_treated_units_skips_late_cells(
        self,
    ) -> None:
        """Cells without comparison units are reported, not estimated."""
        outcomes, groups = staggered_panel(4, per_group=6, noise=0.0)
        outcomes, groups = outcomes[:12], groups[:12]
        with self.assertRaisesRegex(ValueError, "never-treated"):
            group_time_effects(outcomes, groups)
        result = group_time_effects(outcomes, groups, comparison="notyet")
        self.assertEqual(
            [(c.group, c.period) for c in result.effects], [(2.0, 2.0), (2.0, 3.0)]
        )
        self.assertEqual(
            result.skipped_cells, ((2.0, 4.0), (2.0, 5.0), (4.0, 4.0), (4.0, 5.0))
        )
        self.assertAlmostEqual(result.effect(2, 3).estimate, 3.0, places=12)
        with self.assertRaises(KeyError):
            result.effect(4, 4)

    def test_period_labels_and_long_format(self) -> None:
        """Calendar labels and long records give the same estimates."""
        outcomes, groups = staggered_panel(6, per_group=5)
        years = [2010 + 2 * t for t in range(6)]
        labelled = group_time_effects(
            outcomes,
            [None if g is None else 2010 + 2 * g for g in groups],
            periods=years,
        )
        plain = group_time_effects(outcomes, groups)
        self.assertEqual(
            [c.estimate for c in labelled.effects], [c.estimate for c in plain.effects]
        )
        self.assertEqual(labelled.effects[1].period, 2016.0)
        self.assertEqual(labelled.effects[1].exposure, 1)
        records = [
            (f"u{i}", years[t], outcomes[i][t])
            for t in reversed(range(6))
            for i in range(len(outcomes))
        ]
        units, grid, matrix = panel_from_long(
            [r[0] for r in records], [r[1] for r in records], [r[2] for r in records]
        )
        self.assertEqual(grid, tuple(float(v) for v in years))
        self.assertEqual(units[0], "u0")
        self.assertEqual(matrix, tuple(tuple(row) for row in outcomes))
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            panel_from_long(
                [r[0] for r in records[:-1]],
                [r[1] for r in records[:-1]],
                [r[2] for r in records[:-1]],
            )
        with self.assertRaisesRegex(ValueError, "more than once"):
            panel_from_long(["a", "a"], [1, 1], [0.0, 1.0])

    def test_invalid_designs_are_rejected(self) -> None:
        """Unbalanced panels and impossible adoption periods raise."""
        panel = [[0.0, 1.0, 2.0], [0.0, 1.0, 2.5], [0.0, 1.0, 2.0]]
        with self.assertRaisesRegex(ValueError, "balanced"):
            group_time_effects([[0.0, 1.0], [0.0]], [1, None])
        with self.assertRaisesRegex(ValueError, "first period"):
            group_time_effects(panel, [0, 1, None])
        with self.assertRaisesRegex(ValueError, "not one of the periods"):
            group_time_effects(panel, [7, 1, None])
        with self.assertRaisesRegex(ValueError, "ever treated"):
            group_time_effects(panel, [None, None, None])
        with self.assertRaises(ValueError):
            group_time_effects(panel, [1, None])
        with self.assertRaises(ValueError):
            group_time_effects(panel, [1, 1, None], comparison="always")
        with self.assertRaises(ValueError):
            group_time_effects(panel, [1, 1, None], periods=[3, 2, 1])
        with self.assertRaises(ValueError):
            did_two_by_two([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1, 0, 0])
        with self.assertRaises(ValueError):
            did_two_by_two([1.0, 2.0], [1.0], [1, 0])


def simplex_objective(
    weights: tuple[float, ...],
    target: tuple[float, ...],
    donors: tuple[tuple[float, ...], ...],
    importance: tuple[float, ...],
) -> float:
    """Evaluate ``sum_m v_m (x1_m - sum_j w_j X0_jm)^2`` directly."""
    return sum(
        v * (x - sum(w * row[m] for w, row in zip(weights, donors, strict=False))) ** 2
        for m, (x, v) in enumerate(zip(target, importance, strict=False))
    )


class SyntheticControlTests(unittest.TestCase):
    """Check the simplex-constrained weights and the placebo ranks."""

    def test_simplex_projection(self) -> None:
        """Known projections and the KKT characterization hold."""
        self.assertEqual(project_to_simplex([0.2, 0.3, 0.5]), (0.2, 0.3, 0.5))
        self.assertEqual(project_to_simplex([5.0, 0.0, 0.0]), (1.0, 0.0, 0.0))
        self.assertEqual(project_to_simplex([7.0]), (1.0,))
        for a, b in zip(
            project_to_simplex([1.0, 1.0, 1.0, 1.0]), [0.25] * 4, strict=False
        ):
            self.assertAlmostEqual(a, b, places=15)
        rng = random.Random(3)
        for _ in range(50):
            v = [rng.gauss(0, 2) for _ in range(6)]
            w = project_to_simplex(v)
            self.assertAlmostEqual(sum(w), 1.0, places=12)
            self.assertGreaterEqual(min(w), 0.0)
            shifts = [vi - wi for vi, wi in zip(v, w, strict=False) if wi > 0.0]
            theta = shifts[0]
            for shift in shifts:
                self.assertAlmostEqual(shift, theta, places=12)
            for vi, wi in zip(v, w, strict=False):
                if wi == 0.0:
                    self.assertLessEqual(vi, theta + 1e-12)

    def test_treated_unit_that_duplicates_a_donor_gets_that_donor(self) -> None:
        """A duplicated donor receives weight one."""
        rng = random.Random(4)
        donors = tuple(tuple(rng.gauss(0, 1) for _ in range(7)) for _ in range(5))
        fit = synthetic_control_weights(donors[3], donors)
        self.assertTrue(fit.converged)
        for j, weight in enumerate(fit.weights):
            self.assertAlmostEqual(weight, 1.0 if j == 3 else 0.0, places=7)
        self.assertAlmostEqual(fit.objective, 0.0, places=12)

    def test_exact_convex_combination_is_recovered(self) -> None:
        """A treated unit inside the donor hull has zero pre-period error."""
        rng = random.Random(5)
        donors = tuple(tuple(rng.gauss(0, 1) for _ in range(8)) for _ in range(4))
        truth = (0.5, 0.0, 0.2, 0.3)
        target = tuple(
            sum(w * row[m] for w, row in zip(truth, donors, strict=False))
            for m in range(8)
        )
        fit = synthetic_control_weights(target, donors)
        self.assertTrue(fit.converged)
        for weight, expected in zip(fit.weights, truth, strict=False):
            self.assertAlmostEqual(weight, expected, places=7)
        post = tuple(tuple(rng.gauss(0, 1) for _ in range(3)) for _ in range(4))
        treated_post = tuple(
            sum(w * row[m] for w, row in zip(truth, post, strict=False)) + 2.0
            for m in range(3)
        )
        result = synthetic_control(
            target + treated_post,
            [a + b for a, b in zip(donors, post, strict=False)],
            8,
        )
        self.assertAlmostEqual(result.pre_rmspe, 0.0, places=7)
        for effect in result.effects[8:]:
            self.assertAlmostEqual(effect, 2.0, places=6)
        self.assertAlmostEqual(result.average_effect, 2.0, places=6)
        self.assertAlmostEqual(result.post_rmspe, 2.0, places=6)
        self.assertEqual(len(result.synthetic), 11)

    def test_karush_kuhn_tucker_conditions_hold_at_the_solution(self) -> None:
        """Gradient entries are equal on the support and larger off it."""
        rng = random.Random(6)
        for trial in range(5):
            donors = tuple(tuple(rng.gauss(0, 1) for _ in range(6)) for _ in range(8))
            target = tuple(rng.gauss(0.3, 1) for _ in range(6))
            importance = tuple(rng.uniform(0.5, 2.0) for _ in range(6))
            fit = synthetic_control_weights(target, donors, importance=importance)
            self.assertTrue(fit.converged, trial)
            self.assertAlmostEqual(sum(fit.weights), 1.0, places=12)
            self.assertGreaterEqual(min(fit.weights), 0.0)
            residual = [
                target[m]
                - sum(w * row[m] for w, row in zip(fit.weights, donors, strict=False))
                for m in range(6)
            ]
            gradient = [
                -2.0 * sum(importance[m] * residual[m] * row[m] for m in range(6))
                for row in donors
            ]
            multiplier = min(gradient)
            for weight, g in zip(fit.weights, gradient, strict=False):
                if weight > 1e-8:
                    self.assertAlmostEqual(g, multiplier, places=6)
            self.assertAlmostEqual(
                fit.objective,
                simplex_objective(fit.weights, target, donors, importance),
                places=12,
            )
            self.assertLessEqual(fit.duality_gap, 1e-9 * 10)

    def test_three_donor_solution_beats_a_brute_force_grid(self) -> None:
        """No grid point of the simplex has a smaller objective."""
        donors = ((1.0, 0.0, 2.0, 1.0), (0.0, 2.0, 1.0, -1.0), (3.0, 1.0, 0.0, 0.5))
        target = (1.2, 1.1, 0.4, 2.0)
        importance = (1.0, 2.0, 0.5, 1.0)
        fit = synthetic_control_weights(target, donors, importance=importance)
        self.assertTrue(fit.converged)
        steps = 200
        best, best_weights = math.inf, (0.0, 0.0, 0.0)
        for a in range(steps + 1):
            for b in range(steps + 1 - a):
                candidate = (a / steps, b / steps, (steps - a - b) / steps)
                value = simplex_objective(candidate, target, donors, importance)
                if value < best:
                    best, best_weights = value, candidate
        self.assertLessEqual(fit.objective, best + 1e-12)
        self.assertGreater(fit.objective, best - 1e-3)
        for weight, grid_weight in zip(fit.weights, best_weights, strict=False):
            self.assertLess(abs(weight - grid_weight), 0.01)

    def test_zero_importance_predictors_are_ignored(self) -> None:
        """A predictor with importance zero does not influence the weights."""
        donors = ((0.0, 5.0), (1.0, -3.0))
        fit = synthetic_control_weights((0.25, 100.0), donors, importance=(1.0, 0.0))
        self.assertAlmostEqual(fit.weights[0], 0.75, places=8)
        self.assertAlmostEqual(fit.objective, 0.0, places=12)
        custom = synthetic_control(
            [0.0, 0.0, 1.0],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            2,
            treated_predictors=[0.25, 100.0],
            donor_predictors=donors,
            importance=(1.0, 0.0),
        )
        self.assertAlmostEqual(custom.effects[2], 0.75, places=8)

    def test_unconverged_weights_are_flagged(self) -> None:
        """Hitting the iteration limit reports ``converged = False``."""
        rng = random.Random(9)
        donors = tuple(tuple(rng.gauss(0, 1) for _ in range(12)) for _ in range(10))
        target = tuple(rng.gauss(0, 1) for _ in range(12))
        loose = synthetic_control_weights(
            target, donors, max_iterations=1, tolerance=1e-14
        )
        tight = synthetic_control_weights(target, donors)
        self.assertTrue(tight.converged)
        self.assertFalse(loose.converged)
        self.assertEqual(loose.iterations, 1)
        self.assertGreater(loose.duality_gap, 1.0)
        self.assertGreater(loose.objective, tight.objective)
        self.assertLessEqual(
            loose.objective - tight.objective, loose.duality_gap, "gap is a bound"
        )
        partial = synthetic_control(
            target + (0.0,),
            [r + (0.0,) for r in donors],
            12,
            max_iterations=1,
            tolerance=1e-14,
        )
        self.assertFalse(partial.converged)

    def test_placebo_rank_of_a_large_effect(self) -> None:
        """A unit with a clear effect has the largest RMSPE ratio."""
        rng = random.Random(10)
        periods, n_pre = 14, 10
        factor = [math.sin(0.5 * t) + 0.1 * t for t in range(periods)]
        donors = []
        for _ in range(9):
            loading = rng.uniform(0.5, 1.5)
            donors.append([loading * f + 0.05 * rng.gauss(0, 1) for f in factor])
        treated = [
            1.0 * f + 0.05 * rng.gauss(0, 1) + (3.0 if t >= n_pre else 0.0)
            for t, f in enumerate(factor)
        ]
        inference = synthetic_control_placebo(treated, donors, n_pre)
        self.assertTrue(inference.all_converged)
        self.assertEqual(inference.rank, 1)
        self.assertAlmostEqual(inference.p_value, 0.1, places=15)
        self.assertEqual(inference.n_units, 10)
        self.assertEqual(len(inference.placebo_ratios), 9)
        self.assertLess(abs(inference.treated.average_effect - 3.0), 0.3)
        manual = synthetic_control(donors[2], donors[:2] + donors[3:] + [treated], 10)
        self.assertAlmostEqual(
            inference.placebo_ratios[2], manual.rmspe_ratio, places=10
        )
        excluded = synthetic_control_placebo(
            treated, donors, n_pre, treated_in_placebo_pools=False
        )
        manual = synthetic_control(donors[2], donors[:2] + donors[3:], 10)
        self.assertAlmostEqual(
            excluded.placebo_ratios[2], manual.rmspe_ratio, places=10
        )
        null = synthetic_control_placebo(donors[0], donors[1:], n_pre)
        self.assertGreater(null.rank, 1)

    def test_invalid_inputs_are_rejected(self) -> None:
        """Shape and range errors raise ``ValueError``."""
        donors = [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]]
        with self.assertRaises(ValueError):
            synthetic_control_weights([0.0, 1.0], donors)
        with self.assertRaises(ValueError):
            synthetic_control_weights([0.0, 1.0, 2.0], donors, importance=[0, 0, 0])
        with self.assertRaises(ValueError):
            synthetic_control_weights([0.0, 1.0, 2.0], donors, importance=[1, -1, 1])
        with self.assertRaises(ValueError):
            synthetic_control_weights([0.0, 1.0, 2.0], donors, tolerance=0.0)
        with self.assertRaises(ValueError):
            synthetic_control_weights([0.0, 1.0, 2.0], donors, max_iterations=0)
        with self.assertRaises(ValueError):
            synthetic_control([0.0, 1.0, 2.0], donors, 3)
        with self.assertRaises(ValueError):
            synthetic_control([0.0, 1.0, 2.0], donors, 0)
        with self.assertRaises(ValueError):
            synthetic_control([0.0, 1.0], donors, 1)
        with self.assertRaises(ValueError):
            synthetic_control([0.0, 1.0, 2.0], donors, 2, treated_predictors=[1.0])
        with self.assertRaises(ValueError):
            synthetic_control_placebo([0.0, 1.0, 2.0], donors[:1], 2)


def solve_matrix(
    left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...]
) -> tuple[tuple[float, ...], ...]:
    """Return ``left^-1 right`` column by column with an LU solve."""
    columns = [lu_solve(left, column) for column in transpose(right)]
    return transpose(tuple(columns))


def instrument_sample(
    n: int, seed: int
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Simulate two instruments, a covariate, an endogenous ``d`` and ``y``.

    The structural coefficient of ``d`` is 1.5; an unobserved ``u`` drives
    both ``d`` and ``y``, and the error variance grows with ``|z1|``.
    """
    rng = random.Random(seed)
    z1 = [rng.gauss(0, 1) for _ in range(n)]
    z2 = [rng.gauss(0, 1) for _ in range(n)]
    x = [rng.gauss(0, 1) for _ in range(n)]
    u = [rng.gauss(0, 1) for _ in range(n)]
    d = [
        0.8 * a + 0.5 * b + 0.3 * c + e + rng.gauss(0, 1)
        for a, b, c, e in zip(z1, z2, x, u, strict=False)
    ]
    y = [
        1.5 * di + 0.7 * c + 2.0 * e + (0.5 + abs(a)) * rng.gauss(0, 1)
        for di, c, e, a in zip(d, x, u, z1, strict=False)
    ]
    return z1, z2, x, d, y


class InstrumentalVariablesTests(unittest.TestCase):
    """Check two-stage least squares against closed forms."""

    def test_binary_instrument_gives_the_wald_ratio(self) -> None:
        """2SLS equals the ratio of mean differences by instrument arm."""
        rng = random.Random(12)
        z = [int(rng.random() < 0.5) for _ in range(200)]
        u = [rng.gauss(0, 1) for _ in range(200)]
        d = [
            int(rng.random() < 0.2 + 0.5 * zi + 0.1 * (ui > 0))
            for zi, ui in zip(z, u, strict=False)
        ]
        y = [2.0 * di + ui + rng.gauss(0, 1) for di, ui in zip(d, u, strict=False)]

        def arm_mean(values: list[float] | list[int], arm: int) -> float:
            """Return the mean of ``values`` among units with ``z == arm``."""
            return statistics.fmean(
                v for v, zi in zip(values, z, strict=False) if zi == arm
            )

        wald = (arm_mean(y, 1) - arm_mean(y, 0)) / (arm_mean(d, 1) - arm_mean(d, 0))
        result = two_stage_least_squares(y, d, z)
        self.assertAlmostEqual(result.coefficient("d0"), wald, places=10)
        self.assertEqual(result.column_names, ("(Intercept)", "d0"))
        self.assertEqual((result.n_instruments, result.n_endogenous), (1, 1))
        first_stage = fit_linear_model([(1, zi) for zi in z], d)
        self.assertAlmostEqual(
            result.first_stage_f[0], first_stage.t_values[1] ** 2, places=8
        )
        self.assertAlmostEqual(
            result.first_stage_f[0], first_stage.f_statistic, places=8
        )
        self.assertEqual(result.first_stage_df, (1, 198))
        self.assertFalse(result.weak_instruments)

    def test_exactly_identified_model_equals_the_closed_form(self) -> None:
        """Coefficients and robust covariance match ``(Z'X)^-1`` formulas."""
        z1, _, x, d, y = instrument_sample(150, 13)
        result = two_stage_least_squares(
            y, d, z1, exogenous=[(v,) for v in x], endogenous_names=["dose"]
        )
        design = tuple((1.0, di, xi) for di, xi in zip(d, x, strict=False))
        instruments = tuple((1.0, zi, xi) for zi, xi in zip(z1, x, strict=False))
        zx = matmul(transpose(instruments), design)
        zy = tuple(
            sum(row[a] * yi for row, yi in zip(instruments, y, strict=False))
            for a in range(3)
        )
        beta = lu_solve(zx, zy)
        self.assertEqual(result.column_names, ("(Intercept)", "dose", "x0"))
        for a, b in zip(result.coefficients, beta, strict=False):
            self.assertAlmostEqual(a, b, places=9)
        residuals = [
            yi - sum(b * v for b, v in zip(beta, row, strict=False))
            for yi, row in zip(y, design, strict=False)
        ]
        for a, b in zip(result.residuals, residuals, strict=False):
            self.assertAlmostEqual(a, b, places=8)
        meat = tuple(
            tuple(
                sum(
                    e * e * row[a] * row[b]
                    for e, row in zip(residuals, instruments, strict=False)
                )
                for b in range(3)
            )
            for a in range(3)
        )
        left = solve_matrix(zx, meat)  # (Z'X)^-1 M
        covariance = transpose(solve_matrix(zx, transpose(left)))  # ... (X'Z)^-1
        for a in range(3):
            for b in range(3):
                self.assertAlmostEqual(
                    result.covariance[a][b], covariance[a][b], places=9
                )
        hc1 = two_stage_least_squares(
            y, d, z1, exogenous=[(v,) for v in x], robust="HC1"
        )
        self.assertAlmostEqual(
            hc1.standard_errors[1] / result.standard_errors[1],
            math.sqrt(150 / 147),
            places=12,
        )
        low, high = result.confidence_interval("dose")
        self.assertAlmostEqual(
            high - low, 2 * 1.959963984540054 * result.standard_errors[1], places=9
        )

    def test_overidentified_two_stage_estimate_is_the_projection_formula(self) -> None:
        """Second-stage OLS on fitted values equals ``(Xhat'X)^-1 Xhat'y``."""
        z1, z2, x, d, y = instrument_sample(120, 14)
        result = two_stage_least_squares(
            y, d, list(zip(z1, z2, strict=False)), exogenous=[(v,) for v in x]
        )
        full = [(1.0, a, b, c) for a, b, c in zip(z1, z2, x, strict=False)]
        fitted = fit_linear_model(full, d).fitted_values
        hat_design = tuple((1.0, f, xi) for f, xi in zip(fitted, x, strict=False))
        design = tuple((1.0, di, xi) for di, xi in zip(d, x, strict=False))
        left = matmul(transpose(hat_design), design)
        right = tuple(
            sum(row[a] * yi for row, yi in zip(hat_design, y, strict=False))
            for a in range(3)
        )
        for a, b in zip(result.coefficients, lu_solve(left, right), strict=False):
            self.assertAlmostEqual(a, b, places=9)
        restricted = fit_linear_model([(1.0, c) for c in x], d).residual_sum_squares
        unrestricted = fit_linear_model(full, d).residual_sum_squares
        f_value = ((restricted - unrestricted) / 2) / (unrestricted / 116)
        self.assertAlmostEqual(result.first_stage_f[0], f_value, places=9)
        sigma2 = sum(e * e for e in result.residuals) / 117
        hat_fit = fit_linear_model(hat_design, y)
        self.assertAlmostEqual(
            result.classical_standard_errors[1],
            math.sqrt(sigma2 * hat_fit.unscaled_covariance[1][1]),
            places=12,
        )

    def test_instruments_remove_the_confounding_bias(self) -> None:
        """OLS is biased upward; 2SLS covers 1.5 at about the nominal rate."""
        replicates, covered = 100, 0
        estimates: list[float] = []
        errors: list[float] = []
        ols: list[float] = []
        for rep in range(replicates):
            z1, z2, x, d, y = instrument_sample(200, 500 + rep)
            result = two_stage_least_squares(
                y, d, list(zip(z1, z2, strict=False)), exogenous=[(v,) for v in x]
            )
            low, high = result.confidence_interval("d0")
            covered += low <= 1.5 <= high
            estimates.append(result.coefficient("d0"))
            errors.append(result.standard_errors[1])
            if rep < 10:
                ols.append(
                    fit_linear_model(
                        [(1.0, di, xi) for di, xi in zip(d, x, strict=False)], y
                    ).coefficients[1]
                )
        self.assertGreater(min(ols), 2.0)
        self.assertGreaterEqual(covered / replicates, 0.88)
        ratio = statistics.fmean(errors) / statistics.stdev(estimates)
        self.assertGreater(ratio, 0.8)
        self.assertLess(ratio, 1.25)
        monte_carlo_error = statistics.stdev(estimates) / math.sqrt(replicates)
        self.assertLess(abs(statistics.fmean(estimates) - 1.5), 4 * monte_carlo_error)

    def test_weak_and_invalid_specifications(self) -> None:
        """Weak instruments are flagged; bad specifications raise."""
        rng = random.Random(15)
        z = [rng.gauss(0, 1) for _ in range(100)]
        d = [0.02 * zi + rng.gauss(0, 1) for zi in z]
        y = [di + rng.gauss(0, 1) for di in d]
        weak = two_stage_least_squares(y, d, z)
        self.assertTrue(weak.weak_instruments)
        self.assertLess(weak.first_stage_f[0], 10.0)
        no_intercept = two_stage_least_squares(y, d, z, intercept=False)
        self.assertEqual(no_intercept.column_names, ("d0",))
        ratio = sum(a * b for a, b in zip(z, y, strict=False)) / sum(
            a * b for a, b in zip(z, d, strict=False)
        )
        self.assertAlmostEqual(no_intercept.coefficients[0], ratio, places=9)
        self.assertEqual(no_intercept.first_stage_df, (1, 99))
        with self.assertRaisesRegex(ValueError, "under-identified"):
            two_stage_least_squares(y, list(zip(d, z, strict=False)), z)
        with self.assertRaises(ValueError):
            two_stage_least_squares(y, d, z[:-1])
        with self.assertRaises(ValueError):
            two_stage_least_squares(y, d, z, robust="HC9")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            two_stage_least_squares(y, d, z, endogenous_names=["a", "b"])
        with self.assertRaises(RankDeficientDesignError):
            two_stage_least_squares(y, d, list(zip(z, z, strict=False)))
        with self.assertRaises(KeyError):
            weak.coefficient("missing")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            two_stage_least_squares(y, d, z)


class AuditRegressionTests(unittest.TestCase):
    """Regressions found by the independent audit of the 0.27 causal modules."""

    def test_significance_robustness_is_a_minimum_over_both_strengths(self) -> None:
        """With a large t on few degrees of freedom the worst confounder is unequal.

        For ``f_q > 1 / f*`` the equal-strength root overstates the value; the
        bound ``(f_q^2 - f*^2) / (1 + f_q^2)`` is checked here by a grid search
        over ``R2_Y <= R`` at ``R2_D = R`` (the adjusted t decreases in
        ``R2_D``).
        """
        t, df = 20.0, 5
        critical = StudentT(float(df - 1)).quantile(0.975)
        f, f_star = t / math.sqrt(df), critical / math.sqrt(df - 1)
        self.assertGreater(f * f_star, 1.0)
        value = robustness_value(t, df, alpha=0.05)
        self.assertAlmostEqual(value, (f * f - f_star**2) / (1 + f * f), places=12)

        def smallest_t(bound: float) -> float:
            """Return the smallest adjusted t with both strengths <= bound."""
            return min(
                adjust_for_confounder(
                    t, 1.0, df, r2_outcome=bound * i / 2000, r2_treatment=bound
                ).adjusted_t
                for i in range(2001)
            )

        self.assertGreater(smallest_t(value - 1e-3), critical)
        self.assertLess(smallest_t(value + 1e-3), critical)
        worst = adjust_for_confounder(
            t,
            1.0,
            df,
            r2_outcome=value / (f * f * (1 - value)),
            r2_treatment=value,
        )
        self.assertAlmostEqual(worst.adjusted_t, critical, places=9)
        # Outside that regime the equal-strength root is the minimum.
        usual = robustness_value(3.0, 100, alpha=0.05)
        critical = StudentT(99.0).quantile(0.975)
        at_root = adjust_for_confounder(
            3.0, 1.0, 100, r2_outcome=usual, r2_treatment=usual
        )
        self.assertAlmostEqual(at_root.adjusted_t, critical, places=9)

    def test_synthetic_control_converges_on_outcomes_in_levels(self) -> None:
        """A common level of 10^4 in all predictors does not stall the solver."""
        rng = random.Random(3)
        base = [1e4 + 100 * math.sin(0.4 * t) + 30 * t for t in range(12)]
        donors = [
            [b * (1 + 0.01 * rng.gauss(0, 1)) + rng.gauss(0, 1) for b in base]
            for _ in range(30)
        ]
        treated = [b + rng.gauss(0, 1) for b in base]
        fit = synthetic_control_weights(treated, donors)
        self.assertTrue(fit.converged)
        self.assertLess(fit.iterations, 2000)
        self.assertLess(fit.objective, 1e-3)
        # The same problem shifted by the treated unit has the same solution.
        shifted = synthetic_control_weights(
            [0.0] * 12,
            [[v - x for v, x in zip(row, treated, strict=True)] for row in donors],
        )
        self.assertTrue(shifted.converged)
        self.assertLess(shifted.objective, 1e-3)

    def test_single_unit_arm_has_no_standard_error(self) -> None:
        """One comparison (or treated) unit gives ``nan``, not a zero error."""
        rng = random.Random(1)
        groups = [2] * 10 + [3] + [None]
        panel = [
            [rng.gauss(0, 1) + (1.0 if g and t >= g else 0.0) for t in range(5)]
            for g in groups
        ]
        result = group_time_effects(panel, groups)
        for cell in result.effects:
            self.assertTrue(math.isnan(cell.standard_error))
        self.assertTrue(math.isnan(result.overall().standard_error))
        with self.assertRaises(ValueError):
            result.overall().confidence_interval()
        notyet = group_time_effects(panel, groups, comparison="notyet")
        self.assertTrue(math.isfinite(notyet.effect(2, 2).standard_error))
        self.assertTrue(math.isnan(notyet.effect(3, 3).standard_error))


if __name__ == "__main__":
    unittest.main()
