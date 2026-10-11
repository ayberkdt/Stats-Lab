"""Tests for missing-data EM, Little's test, multiple imputation, and pooling."""

from __future__ import annotations

import math
import random
import unittest
from fractions import Fraction

from probstat.foundations.random_streams import RandomStream
from probstat.inference import (
    NormalInverseWishartPrior,
    complete_cases,
    fit_incomplete_multivariate_normal,
    fit_multivariate_normal_mle,
    impute_multivariate_normal,
    incomplete_normal_log_likelihood,
    little_mcar_test,
    mean_impute,
    missing_pattern,
    pool_estimates,
    sample_inverse_wishart,
    student_t_quantile,
)
from probstat.multivariate import MultivariateNormal

TRUE_MEAN = (1.0, 2.0, -1.0)
TRUE_COVARIANCE = (
    (1.0, 0.6, 0.3),
    (0.6, 1.5, 0.4),
    (0.3, 0.4, 0.8),
)


def logistic(value: float) -> float:
    """Return the standard logistic function."""
    return 1.0 / (1.0 + math.exp(-value))


def simulate_rows(
    n: int, seed: int, mechanism: str
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float | None, ...], ...]]:
    """Return full and incomplete trivariate normal rows.

    Variable 0 is always observed. Under ``"mar"`` variable 1 is missing with
    probability ``logistic(-0.5 + 2 (x0 - 1))`` (depends on the observed x0);
    under ``"mcar"`` with probability 0.35. Variable 2 is missing completely at
    random with probability 0.2 in both cases.
    """
    law = MultivariateNormal(TRUE_MEAN, TRUE_COVARIANCE)
    full = law.sample(n, seed=seed)
    rng = random.Random(seed + 10_000)
    rows: list[tuple[float | None, ...]] = []
    for x in full:
        if mechanism == "mar":
            drop_1 = rng.random() < logistic(-0.5 + 2.0 * (x[0] - 1.0))
        else:
            drop_1 = rng.random() < 0.35
        drop_2 = rng.random() < 0.2
        rows.append((x[0], None if drop_1 else x[1], math.nan if drop_2 else x[2]))
    return full, tuple(rows)


def column_mean(rows: tuple[tuple[float, ...], ...], index: int) -> float:
    """Return the mean of one column of complete rows."""
    return math.fsum(row[index] for row in rows) / len(rows)


class RepresentationTests(unittest.TestCase):
    """Missing entries, pattern summaries, and baselines."""

    def test_pattern_summary_counts(self) -> None:
        """None and nan are both missing; patterns are counted and sorted."""
        rows = [
            (1.0, 2.0, 3.0),
            (1.5, None, 2.0),
            (2.0, math.nan, None),
            (0.5, 1.0, 4.0),
            (None, None, None),
        ]
        summary = missing_pattern(rows)
        self.assertEqual(summary.n_observations, 5)
        self.assertEqual(summary.complete_cases, 2)
        self.assertEqual(summary.missing_fraction, (0.2, 0.6, 0.4))
        self.assertEqual(
            summary.patterns,
            (
                (True, True, True),
                (True, False, True),
                (True, False, False),
                (False, False, False),
            ),
        )
        self.assertEqual(summary.pattern_counts, (2, 1, 1, 1))
        self.assertAlmostEqual(summary.overall_missing_fraction, 1.2 / 3)
        self.assertEqual(len(complete_cases(rows)), 2)

    def test_invalid_entries_are_rejected(self) -> None:
        """Infinite, boolean, non-numeric, and ragged inputs raise."""
        with self.assertRaises(ValueError):
            missing_pattern([(1.0, math.inf)])
        with self.assertRaises(TypeError):
            missing_pattern([(1.0, True)])
        with self.assertRaises(TypeError):
            missing_pattern([(1.0, "a")])
        with self.assertRaises(ValueError):
            missing_pattern([(1.0, 2.0), (1.0,)])
        with self.assertRaises(ValueError):
            missing_pattern([])

    def test_mean_imputation_shrinks_variance(self) -> None:
        """Mean imputation keeps the observed mean but underestimates variance."""
        _, rows = simulate_rows(600, 21, "mar")
        imputed = mean_impute(rows)
        observed = [row[1] for row in rows if row[1] is not None]
        observed_mean = math.fsum(observed) / len(observed)
        self.assertAlmostEqual(column_mean(imputed, 1), observed_mean, places=12)
        n = len(imputed)
        imputed_variance = (
            math.fsum((row[1] - observed_mean) ** 2 for row in imputed) / n
        )
        fit = fit_incomplete_multivariate_normal(rows, standard_errors=False)
        fraction_observed = len(observed) / n
        # Seed 21: 42.2% of variable 1 is missing; the mean-imputed variance is
        # 0.778 against the EM estimate 1.446 (true 1.5).
        self.assertLess(imputed_variance, 0.75 * fit.covariance[1][1])
        self.assertLess(imputed_variance, fraction_observed * TRUE_COVARIANCE[1][1])


class LikelihoodAndEMTests(unittest.TestCase):
    """Observed-data likelihood and EM for the multivariate normal."""

    def test_log_likelihood_matches_marginal_densities(self) -> None:
        """Complete rows use the full density, partial rows their marginal."""
        law = MultivariateNormal(TRUE_MEAN, TRUE_COVARIANCE)
        rows = [(1.2, 2.5, -0.7), (0.3, None, None), (None, 1.0, -1.5)]
        marginal_23 = law.marginal_subset((1, 2))
        expected = (
            law.log_pdf(rows[0])
            + law.marginal(0).log_pdf(0.3)
            + marginal_23.log_pdf((1.0, -1.5))
        )
        value = incomplete_normal_log_likelihood(rows, TRUE_MEAN, TRUE_COVARIANCE)
        self.assertAlmostEqual(value, expected, places=12)

    def test_em_log_likelihood_is_monotone(self) -> None:
        """Every EM step increases the observed-data log-likelihood."""
        _, rows = simulate_rows(300, 3, "mar")
        fit = fit_incomplete_multivariate_normal(rows, standard_errors=False)
        self.assertTrue(fit.em.monotone)
        self.assertTrue(fit.em.converged)
        trace = fit.em.trace
        self.assertGreater(len(trace), 5)
        for old, new in zip(trace, trace[1:], strict=False):
            self.assertGreaterEqual(new, old - 1e-9 * (1.0 + abs(old)))
        self.assertAlmostEqual(
            fit.log_likelihood,
            incomplete_normal_log_likelihood(rows, fit.mean, fit.covariance),
            places=9,
        )

    def test_em_recovers_parameters_where_complete_cases_are_biased(self) -> None:
        """Under MAR on x0, EM is close to the truth; complete cases are not."""
        _, rows = simulate_rows(600, 5, "mar")
        fit = fit_incomplete_multivariate_normal(rows)
        assert fit.mean_standard_errors is not None
        # Seed 5, n = 600 (285 complete cases): EM mean (0.970, 1.985, -1.020)
        # with standard errors (0.040, 0.067, 0.039); complete-case means
        # (0.515, 1.755, -1.143), i.e. biases (-0.485, -0.245, -0.143).
        for j in range(3):
            error = abs(fit.mean[j] - TRUE_MEAN[j])
            self.assertLess(error, 3.0 * fit.mean_standard_errors[j])
            self.assertLess(error, 0.1)
            for k in range(3):
                self.assertLess(abs(fit.covariance[j][k] - TRUE_COVARIANCE[j][k]), 0.2)
        cases = complete_cases(rows)
        self.assertLess(column_mean(cases, 1) - TRUE_MEAN[1], -0.2)
        self.assertLess(column_mean(cases, 0) - TRUE_MEAN[0], -0.4)
        self.assertGreater(fit.mean_standard_errors[1], math.sqrt(1.5 / 600))

    def test_no_missing_values_reproduce_the_mle_exactly(self) -> None:
        """Without missing values EM returns the MLE bit for bit."""
        full = MultivariateNormal(TRUE_MEAN, TRUE_COVARIANCE).sample(200, seed=3)
        fit = fit_incomplete_multivariate_normal(full)
        mle = fit_multivariate_normal_mle(full)
        self.assertEqual(fit.mean, mle.mean)
        self.assertEqual(fit.covariance, mle.covariance)
        assert fit.mean_standard_errors is not None
        for j in range(3):
            expected = math.sqrt(mle.covariance[j][j] / 200)
            self.assertAlmostEqual(
                fit.mean_standard_errors[j] / expected, 1.0, delta=1e-6
            )
        self.assertEqual(fit.pattern.complete_cases, 200)


class LittleTestTests(unittest.TestCase):
    """Little's MCAR test under MCAR and under MAR."""

    def test_p_values_under_mcar_are_not_concentrated_near_zero(self) -> None:
        """Twenty MCAR replications give roughly uniform p-values."""
        p_values = []
        for rep in range(20):
            _, rows = simulate_rows(150, 500 + rep, "mcar")
            result = little_mcar_test(rows)
            self.assertEqual(result.degrees_of_freedom, 5)
            p_values.append(result.p_value)
        # Seeds 500..519: mean p-value 0.552, 2 of 20 below 0.05.
        self.assertLessEqual(sum(p < 0.05 for p in p_values), 4)
        self.assertGreater(sum(p_values) / 20, 0.3)
        self.assertLess(sum(p_values) / 20, 0.7)

    def test_mar_shift_gives_a_small_p_value(self) -> None:
        """Missingness driven by x0 shifts pattern means and is detected."""
        _, rows = simulate_rows(300, 9, "mar")
        result = little_mcar_test(rows)
        # Seed 9, n = 300: d^2 = 120.74 on 5 df, p = 2.2e-24.
        self.assertLess(result.p_value, 1e-6)
        self.assertEqual(result.n_patterns, 4)

    def test_single_pattern_is_rejected(self) -> None:
        """With complete data the test has no degrees of freedom."""
        full = MultivariateNormal(TRUE_MEAN, TRUE_COVARIANCE).sample(50, seed=1)
        with self.assertRaises(ValueError):
            little_mcar_test(full)


class InverseWishartTests(unittest.TestCase):
    """Bartlett inverse-Wishart sampler."""

    def test_mean_matches_scale_over_shifted_df(self) -> None:
        """The sample mean matches ``Psi / (nu - p - 1)`` within MC error."""
        psi = ((2.0, 0.5), (0.5, 1.0))
        nu = 8.0
        draws = sample_inverse_wishart(psi, nu, 8000, seed=2)
        for i in range(2):
            for j in range(2):
                values = [draw[i][j] for draw in draws]
                mean = math.fsum(values) / len(values)
                spread = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
                target = psi[i][j] / (nu - 2 - 1)
                self.assertLess(
                    abs(mean - target), 4.0 * math.sqrt(spread / len(values))
                )
        for draw in draws[:50]:
            self.assertEqual(draw[0][1], draw[1][0])
            MultivariateNormal((0.0, 0.0), draw)

    def test_seed_reproducibility_and_validation(self) -> None:
        """Equal seeds give equal draws; invalid df raises."""
        psi = ((1.0, 0.2), (0.2, 1.0))
        first = sample_inverse_wishart(psi, 5.0, 3, seed=7)
        second = sample_inverse_wishart(psi, 5.0, 3, seed=RandomStream(7))
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            sample_inverse_wishart(psi, 1.0)


class RubinRulesTests(unittest.TestCase):
    """Rubin's combining rules and the Barnard-Rubin degrees of freedom."""

    def test_hand_example(self) -> None:
        """Fixed numbers reproduce T, r, lambda, gamma, and both df values."""
        estimates = (10.0, 12.0, 11.0, 13.0, 9.0)
        variances = (3.5, 4.5, 4.0, 3.0, 5.0)
        # Qbar = 11, Ubar = 4, B = (1 + 1 + 0 + 4 + 4) / 4 = 2.5,
        # T = 4 + 1.2 * 2.5 = 7, r = 3/4, lambda = 3/7,
        # nu_old = 4 / (3/7)^2 = 196/9.
        pooled = pool_estimates(estimates, variances)
        self.assertEqual(pooled.estimate, 11.0)
        self.assertEqual(pooled.within_variance, 4.0)
        self.assertAlmostEqual(pooled.between_variance, 2.5, places=14)
        self.assertAlmostEqual(pooled.total_variance, 7.0, places=14)
        self.assertAlmostEqual(pooled.relative_increase, 0.75, places=14)
        self.assertAlmostEqual(pooled.missing_variance_fraction, 3.0 / 7.0, places=14)
        self.assertAlmostEqual(pooled.degrees_of_freedom, 196.0 / 9.0, places=10)
        # gamma = ((nu + 1) lambda + 2) / (nu + 3) = 741 / 1561.
        self.assertAlmostEqual(
            pooled.fraction_missing_information, 741.0 / 1561.0, places=12
        )
        critical = student_t_quantile(0.975, 196.0 / 9.0)
        self.assertAlmostEqual(
            pooled.lower, 11.0 - critical * math.sqrt(7.0), places=10
        )
        self.assertAlmostEqual(
            pooled.upper, 11.0 + critical * math.sqrt(7.0), places=10
        )
        # Barnard-Rubin with nu_com = 20:
        # nu_obs = (21/23) * 20 * (4/7) = 240/23,
        # nu = 1 / (9/196 + 23/240) = 11760/1667.
        small = pool_estimates(estimates, variances, complete_data_df=20)
        lam = Fraction(3, 7)
        nu_obs = Fraction(21, 23) * 20 * (1 - lam)
        self.assertEqual(nu_obs, Fraction(240, 23))
        nu_br = 1 / (Fraction(9, 196) + 1 / nu_obs)
        self.assertAlmostEqual(small.degrees_of_freedom, float(nu_br), places=10)
        self.assertEqual(nu_br, Fraction(11760, 1667))  # 7.05459...
        self.assertLess(small.degrees_of_freedom, 20.0)

    def test_zero_between_variance_and_validation(self) -> None:
        """Identical estimates give r = 0 and an infinite large-sample df."""
        pooled = pool_estimates((1.0, 1.0, 1.0), (0.04, 0.04, 0.04))
        self.assertEqual(pooled.relative_increase, 0.0)
        self.assertTrue(math.isinf(pooled.degrees_of_freedom))
        self.assertAlmostEqual(pooled.upper - pooled.estimate, 1.959964 * 0.2, places=5)
        with self.assertRaises(ValueError):
            pool_estimates((1.0,), (0.1,))
        with self.assertRaises(ValueError):
            pool_estimates((1.0, 2.0), (0.1, -0.1))


class MultipleImputationTests(unittest.TestCase):
    """Data augmentation, determinism, and coverage of pooled intervals."""

    @staticmethod
    def _bivariate(n: int, seed: int) -> tuple[tuple[float, float | None], ...]:
        """Return (x, y) rows with y missing at random given x."""
        law = MultivariateNormal((0.0, 1.0), ((1.0, 0.6), (0.6, 1.0)))
        rng = random.Random(seed + 77)
        rows: list[tuple[float, float | None]] = []
        for x, y in law.sample(n, seed=seed):
            missing = rng.random() < logistic(-0.3 + 1.5 * x)
            rows.append((x, None if missing else y))
        return tuple(rows)

    def test_observed_values_kept_and_seed_determinism(self) -> None:
        """Imputations keep observed entries and are reproducible by seed."""
        rows = self._bivariate(80, 4)
        first = impute_multivariate_normal(rows, 3, burn_in=5, seed=11)
        second = impute_multivariate_normal(rows, 3, burn_in=5, seed=RandomStream(11))
        other = impute_multivariate_normal(rows, 3, burn_in=5, seed=12)
        self.assertEqual(first.datasets, second.datasets)
        self.assertNotEqual(first.datasets, other.datasets)
        self.assertNotEqual(first.datasets[0], first.datasets[1])
        for dataset in first.datasets:
            for original, completed in zip(rows, dataset, strict=True):
                self.assertEqual(original[0], completed[0])
                if original[1] is not None:
                    self.assertEqual(original[1], completed[1])
                self.assertTrue(math.isfinite(completed[1]))
        with self.assertRaises(ValueError):
            impute_multivariate_normal(rows, 2, burn_in=0, seed=1)

    def test_informative_prior_is_accepted(self) -> None:
        """A proper normal-inverse-Wishart prior runs and keeps dimensions."""
        rows = self._bivariate(40, 8)
        prior = NormalInverseWishartPrior(
            mean=(0.0, 0.0),
            kappa=1.0,
            scale=((1.0, 0.0), (0.0, 1.0)),
            degrees_of_freedom=4.0,
        )
        result = impute_multivariate_normal(rows, 2, burn_in=3, prior=prior, seed=5)
        self.assertEqual(len(result.datasets), 2)
        self.assertEqual(len(result.covariances[0]), 2)
        with self.assertRaises(ValueError):
            NormalInverseWishartPrior(kappa=1.0)

    def test_pooled_interval_coverage_near_nominal(self) -> None:
        """MI and EM intervals for E[y] cover near 95%; complete cases do not.

        100 replications, n = 60, m = 10, burn-in 10 (about 45% of y missing,
        MAR on x; average fraction of missing information 0.48). Seeds
        2000..2099 give MI coverage 0.92, EM Wald coverage 0.95, and
        complete-case t-interval coverage 0.59.
        """
        reps, n, m = 100, 60, 10
        mi_hits = em_hits = cc_hits = 0
        for rep in range(reps):
            rows = self._bivariate(n, 2000 + rep)
            fit = fit_incomplete_multivariate_normal(rows)
            imputed = impute_multivariate_normal(
                rows, m, burn_in=10, start=fit, seed=RandomStream(rep)
            )
            estimates = []
            variances = []
            for dataset in imputed.datasets:
                ys = [row[1] for row in dataset]
                mean = math.fsum(ys) / n
                estimates.append(mean)
                variances.append(math.fsum((y - mean) ** 2 for y in ys) / (n - 1) / n)
            pooled = pool_estimates(estimates, variances, complete_data_df=n - 1)
            mi_hits += pooled.lower <= 1.0 <= pooled.upper
            assert fit.mean_standard_errors is not None
            half = 1.959964 * fit.mean_standard_errors[1]
            em_hits += abs(fit.mean[1] - 1.0) <= half
            observed = [row[1] for row in rows if row[1] is not None]
            k = len(observed)
            cc_mean = math.fsum(observed) / k
            cc_se = math.sqrt(
                math.fsum((y - cc_mean) ** 2 for y in observed) / (k - 1) / k
            )
            cc_hits += abs(cc_mean - 1.0) <= student_t_quantile(0.975, k - 1) * cc_se
        self.assertGreaterEqual(mi_hits / reps, 0.89)
        self.assertLessEqual(mi_hits / reps, 0.995)
        self.assertGreaterEqual(em_hits / reps, 0.87)
        self.assertLess(cc_hits / reps, 0.85)


if __name__ == "__main__":
    unittest.main()
