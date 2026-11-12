"""Tests for count and categorical regression models (ADR-0103).

Every reference value is computed independently inside the test:

* closed forms: the intercept-only negative binomial model reproduces the
  existing count-sample fit and its standard error
  ``sqrt((1 + mu/theta)/(n mu))``; saturated factor designs give group
  means, robust variances ``sum (y - y_bar)^2/(n y_bar)^2``, log ratios of
  cell counts with ``sqrt(sum 1/n)`` standard errors, and differences of
  group proportions; thresholds-only ordinal fits give empirical cumulative
  logits with ``Var(theta_k) = 1/(n F_k (1 - F_k))``; intercept-only
  zero-inflated and hurdle models solve ``lambda/(1 - exp(-lambda)) =`` the
  mean positive count; the Brant statistic for one binary covariate and
  three categories has the explicit log-odds-ratio form;
* identities: a two-category ordinal or multinomial fit is a logistic
  regression; exposures are offsets ``log t``;
* independent likelihoods written in this file, whose finite-difference
  scores vanish at the reported maxima;
* seeded simulations, judged by the reported standard errors.

No R or statsmodels output is quoted: neither tool was available, and no
public-domain data set with verified published output was used.

References:
    [REF-CAMERON-TRIVEDI-2013], [REF-CAMERON-TRIVEDI-1990],
    [REF-LAMBERT-1992], [REF-MULLAHY-1986], [REF-VUONG-1989],
    [REF-WILSON-2015], [REF-MCCULLAGH-1980], [REF-BRANT-1990],
    [REF-AGRESTI-2013].

"""

from __future__ import annotations

import math
import random
import unittest

from probstat.inference import (
    CountRegressionResult,
    SeparationError,
    brant_test,
    fit_glm,
    fit_hurdle,
    fit_multinomial_logit,
    fit_negative_binomial_counts,
    fit_negative_binomial_regression,
    fit_ordinal_logistic,
    fit_poisson_regression,
    fit_zero_inflated,
    poisson_overdispersion_test,
    vuong_test,
)
from probstat.inference.sampling_distributions import (
    chi_square_sf,
    standard_normal_sf,
)


def poisson_draw(rng: random.Random, mean: float) -> int:
    """Draw a Poisson count by multiplying uniforms (Knuth).

    References:
        [EQ-DIST-POIS-001]. Simulation helper for the tests.

    """
    limit = math.exp(-mean)
    count, product = 0, rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def nb_draw(rng: random.Random, mean: float, theta: float) -> int:
    """Draw an NB2 count as a gamma-mixed Poisson.

    References:
        [EQ-CNT-REG-003]. Simulation helper for the tests.

    """
    return poisson_draw(rng, rng.gammavariate(theta, mean / theta))


def nb_log_likelihood(beta: list[float], log_theta: float, rows, y) -> float:
    """Return the NB2 log-likelihood written independently with ``lgamma``.

    References:
        [EQ-CNT-REG-003].

    """
    theta = math.exp(log_theta)
    total = 0.0
    for row, yi in zip(rows, y, strict=True):
        mu = math.exp(sum(b * x for b, x in zip(beta, row, strict=True)))
        total += (
            math.lgamma(yi + theta)
            - math.lgamma(theta)
            - math.lgamma(yi + 1)
            + theta * math.log(theta / (theta + mu))
            + yi * math.log(mu / (theta + mu))
        )
    return total


def finite_score(function, point: list[float], step: float = 1e-6) -> list[float]:
    """Return a central-difference gradient.

    References:
        [EQ-OPT-FD-001].

    """
    score = []
    for j in range(len(point)):
        up = list(point)
        down = list(point)
        up[j] += step
        down[j] -= step
        score.append((function(up) - function(down)) / (2.0 * step))
    return score


def truncated_poisson_rate(mean_positive: float) -> float:
    """Solve ``lambda / (1 - exp(-lambda)) = m`` for ``m > 1`` by bisection.

    References:
        [EQ-CNT-HURDLE-001].

    """
    low, high = 1e-12, mean_positive
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle / -math.expm1(-middle) < mean_positive:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


COUNTS = (0, 0, 0, 1, 2, 0, 3, 1, 0, 4, 2, 0, 0, 5, 1, 7, 0, 2, 9, 0)


class NegativeBinomialRegressionTests(unittest.TestCase):
    """NB2 regression against closed forms and simulation.

    References:
        [EQ-CNT-REG-003], [REF-CAMERON-TRIVEDI-2013].

    """

    def test_intercept_only_matches_count_fit_and_closed_form_error(self) -> None:
        """Intercept-only NB2 equals the iid count fit; se = sqrt((1+mu/theta)/(n mu)).

        References:
            [EQ-CNT-REG-003], [EQ-INF-COUNT-002].

        """
        fit = fit_negative_binomial_regression([(1.0,)] * len(COUNTS), COUNTS)
        iid = fit_negative_binomial_counts(COUNTS)
        n = len(COUNTS)
        mean = sum(COUNTS) / n
        self.assertFalse(fit.at_boundary)
        self.assertAlmostEqual(math.exp(fit.coefficients[0]), mean, places=10)
        self.assertAlmostEqual(fit.theta / iid.estimates[0], 1.0, places=8)
        self.assertAlmostEqual(
            fit.theta_standard_error / iid.standard_errors[0], 1.0, places=6
        )
        self.assertAlmostEqual(fit.log_likelihood, iid.log_likelihood, places=8)
        expected = math.sqrt((1.0 + mean / fit.theta) / (n * mean))
        self.assertAlmostEqual(fit.standard_errors[0], expected, places=10)
        self.assertEqual(fit.parameter_count, 2)

    def test_score_vanishes_at_the_maximum(self) -> None:
        """An independently coded NB2 likelihood is stationary at the fit.

        References:
            [EQ-CNT-REG-003].

        """
        rng = random.Random(11)
        rows = [(1.0, rng.uniform(-1.0, 1.0)) for _ in range(150)]
        y = [nb_draw(rng, math.exp(0.7 + 0.5 * r[1]), 1.5) for r in rows]
        fit = fit_negative_binomial_regression(rows, y)
        point = [*fit.coefficients, math.log(fit.theta)]
        score = finite_score(lambda q: nb_log_likelihood(q[:2], q[2], rows, y), point)
        for value in score:
            self.assertLess(abs(value), 1e-4)

    def test_simulation_recovers_parameters(self) -> None:
        """Seeded NB2 data: estimates within four standard errors of the truth.

        References:
            [EQ-CNT-REG-003].

        """
        rng = random.Random(2026)
        beta, theta = (0.4, 0.6), 2.0
        rows = [(1.0, rng.gauss(0.0, 1.0)) for _ in range(1500)]
        y = [nb_draw(rng, math.exp(beta[0] + beta[1] * r[1]), theta) for r in rows]
        fit = fit_negative_binomial_regression(rows, y)
        self.assertTrue(fit.converged)
        for estimate, error, truth in zip(
            fit.coefficients, fit.standard_errors, beta, strict=True
        ):
            self.assertLess(abs(estimate - truth), 4.0 * error)
        self.assertLess(abs(fit.alpha - 1.0 / theta), 4.0 * fit.alpha_standard_error)

    def test_underdispersed_counts_sit_on_the_poisson_boundary(self) -> None:
        """No overdispersion: theta = inf and the Poisson coefficients.

        References:
            [EQ-CNT-REG-003].

        """
        rows = [(1.0, float(i % 2)) for i in range(12)]
        y = [2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 3]
        fit = fit_negative_binomial_regression(rows, y)
        poisson = fit_poisson_regression(rows, y)
        self.assertTrue(fit.at_boundary)
        self.assertTrue(math.isinf(fit.theta))
        self.assertEqual(fit.alpha, 0.0)
        for a, b in zip(fit.coefficients, poisson.coefficients, strict=True):
            self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(fit.parameter_count, 3)

    def test_rejects_non_integer_counts(self) -> None:
        """Responses must be non-negative integers.

        References:
            [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaises(ValueError):
            fit_negative_binomial_regression([(1.0,)] * 3, [1.5, 2.0, 3.0])
        with self.assertRaises(ValueError):
            fit_poisson_regression([(1.0,)] * 3, [-1, 2, 3])

    def test_average_marginal_effect_and_delta_method_error(self) -> None:
        """AME = beta_j mean(mu) with the analytic delta-method gradient.

        References:
            [EQ-REG-AME-001].

        """
        rng = random.Random(5)
        rows = [(1.0, rng.uniform(0.0, 2.0)) for _ in range(300)]
        y = [nb_draw(rng, math.exp(0.2 + 0.4 * r[1]), 3.0) for r in rows]
        fit = fit_negative_binomial_regression(rows, y)
        effect = fit.average_marginal_effect(1)
        mus = fit.fitted_values
        mean_mu = sum(mus) / len(mus)
        b1 = fit.coefficients[1]
        self.assertAlmostEqual(effect.estimates[0], b1 * mean_mu, places=10)
        gradient = [
            b1 * mean_mu,
            mean_mu
            + b1 * sum(m * r[1] for m, r in zip(mus, rows, strict=True)) / len(mus),
        ]
        variance = sum(
            gradient[a] * fit.covariance[a][b] * gradient[b]
            for a in range(2)
            for b in range(2)
        )
        self.assertAlmostEqual(
            effect.standard_errors[0] / math.sqrt(variance), 1.0, places=5
        )


class PoissonRegressionTests(unittest.TestCase):
    """Poisson regression with exposures, sandwich errors, and the CT test.

    References:
        [EQ-CNT-REG-001], [EQ-CNT-REG-002], [EQ-CNT-REG-004].

    """

    def test_exposure_is_a_log_offset_and_factor_rates_are_closed_form(self) -> None:
        """A saturated factor model gives rates sum(y)/sum(t) per group.

        References:
            [EQ-CNT-REG-001].

        """
        rows = [(1.0, float(i >= 6)) for i in range(12)]
        y = [3, 5, 2, 4, 6, 1, 9, 7, 12, 8, 10, 11]
        t = [1.0, 2.0, 1.5, 1.0, 2.5, 0.5, 2.0, 1.0, 3.0, 1.5, 2.0, 2.5]
        by_exposure = fit_poisson_regression(rows, y, exposure=t)
        by_offset = fit_poisson_regression(rows, y, offset=[math.log(v) for v in t])
        for a, b in zip(by_exposure.coefficients, by_offset.coefficients, strict=True):
            self.assertAlmostEqual(a, b, places=12)
        rate0 = sum(y[:6]) / sum(t[:6])
        rate1 = sum(y[6:]) / sum(t[6:])
        ratio, _, _ = by_exposure.incidence_rate_ratios()[1]
        self.assertAlmostEqual(math.exp(by_exposure.coefficients[0]), rate0, places=9)
        self.assertAlmostEqual(ratio, rate1 / rate0, places=9)
        self.assertAlmostEqual(
            by_exposure.predict([(1.0, 1.0)], exposure=[2.0])[0], 2.0 * rate1, places=8
        )

    def test_robust_errors_match_the_group_formula(self) -> None:
        """Sandwich variance of log y_bar_g is sum (y - y_bar)^2/(n y_bar)^2.

        References:
            [EQ-CNT-REG-002].

        """
        y = [0, 1, 5, 2, 8, 1, 3, 9, 0, 14, 2, 6, 11, 4]
        rows = [(1.0, float(i >= 7)) for i in range(len(y))]
        fit = fit_poisson_regression(rows, y, covariance="robust")
        groups = (y[:7], y[7:])
        variances = []
        for group in groups:
            mean = sum(group) / len(group)
            variances.append(
                sum((v - mean) ** 2 for v in group) / (len(group) * mean) ** 2
            )
        self.assertEqual(fit.covariance_type, "robust")
        self.assertAlmostEqual(fit.standard_errors[0] ** 2, variances[0], places=10)
        self.assertAlmostEqual(
            fit.standard_errors[1] ** 2, variances[0] + variances[1], places=10
        )
        model = fit_poisson_regression(rows, y)
        mean0 = sum(groups[0]) / 7
        self.assertAlmostEqual(
            model.standard_errors[0] ** 2, 1.0 / (7 * mean0), places=10
        )

    def test_overdispersion_statistic_matches_hand_regression(self) -> None:
        """The CT statistic equals the no-intercept OLS t ratio.

        References:
            [EQ-CNT-REG-004].

        """
        rng = random.Random(8)
        rows = [(1.0, rng.uniform(-1.0, 1.0)) for _ in range(120)]
        y = [nb_draw(rng, math.exp(1.0 + 0.5 * r[1]), 1.0) for r in rows]
        fit = fit_poisson_regression(rows, y)
        for form in ("nb2", "nb1"):
            test = poisson_overdispersion_test(fit, form=form)
            mu = fit.fitted_values
            aux = [((a - m) ** 2 - a) / m for a, m in zip(y, mu, strict=True)]
            z = list(mu) if form == "nb2" else [1.0] * len(mu)
            alpha = sum(a * b for a, b in zip(aux, z, strict=True)) / sum(
                b * b for b in z
            )
            rss = sum((a - alpha * b) ** 2 for a, b in zip(aux, z, strict=True))
            se = math.sqrt(rss / (len(y) - 1) / sum(b * b for b in z))
            self.assertAlmostEqual(test.alpha, alpha, places=10)
            self.assertAlmostEqual(test.statistic, alpha / se, places=8)
            self.assertAlmostEqual(test.p_value, standard_normal_sf(alpha / se))
            self.assertLess(test.p_value, 0.01)

    def test_overdispersion_test_size_under_poisson_null(self) -> None:
        """Rejections at 5% under Poisson data stay within binomial error.

        References:
            [EQ-CNT-REG-004].

        """
        rng = random.Random(77)
        replicates, rejections = 200, 0
        for _ in range(replicates):
            rows = [(1.0, rng.uniform(-1.0, 1.0)) for _ in range(150)]
            y = [poisson_draw(rng, math.exp(1.0 + 0.4 * r[1])) for r in rows]
            fit = fit_poisson_regression(rows, y)
            rejections += poisson_overdispersion_test(fit).p_value < 0.05
        rate = rejections / replicates
        tolerance = 4.0 * math.sqrt(0.05 * 0.95 / replicates)
        self.assertLess(abs(rate - 0.05), tolerance)

    def test_overdispersion_test_requires_a_poisson_fit(self) -> None:
        """An NB fit is rejected.

        References:
            [EQ-CNT-REG-004].

        """
        fit = fit_negative_binomial_regression([(1.0,)] * len(COUNTS), COUNTS)
        self.assertIsInstance(fit, CountRegressionResult)
        with self.assertRaises(ValueError):
            poisson_overdispersion_test(fit)


class ZeroCountModelTests(unittest.TestCase):
    """Zero-inflated and hurdle models.

    References:
        [EQ-CNT-ZI-001], [EQ-CNT-HURDLE-001].

    """

    def test_intercept_only_models_have_closed_forms(self) -> None:
        """ZIP and Poisson hurdle: lambda from the truncated mean; equal maxima.

        References:
            [EQ-CNT-ZI-001], [EQ-CNT-HURDLE-001].

        """
        n = len(COUNTS)
        rows = [(1.0,)] * n
        positives = [v for v in COUNTS if v > 0]
        rate = truncated_poisson_rate(sum(positives) / len(positives))
        zip_fit = fit_zero_inflated(rows, COUNTS)
        hurdle = fit_hurdle(rows, COUNTS)
        self.assertAlmostEqual(math.exp(zip_fit.count_coefficients[0]), rate, places=7)
        self.assertAlmostEqual(math.exp(hurdle.count_coefficients[0]), rate, places=7)
        inflation = 1.0 - (sum(COUNTS) / n) / rate
        self.assertAlmostEqual(
            1.0 / (1.0 + math.exp(-zip_fit.zero_coefficients[0])), inflation, places=7
        )
        share = len(positives) / n
        self.assertAlmostEqual(
            hurdle.zero_coefficients[0], math.log(share / (1.0 - share)), places=9
        )
        self.assertAlmostEqual(
            hurdle.zero_standard_errors[0],
            math.sqrt(1.0 / (n * share * (1.0 - share))),
            places=9,
        )
        self.assertAlmostEqual(zip_fit.log_likelihood, hurdle.log_likelihood, places=7)
        zero_share = 1.0 - share
        self.assertAlmostEqual(zip_fit.zero_probabilities[0], zero_share, places=7)
        self.assertAlmostEqual(hurdle.zero_probabilities[0], zero_share, places=12)
        mean = sum(COUNTS) / n
        self.assertAlmostEqual(zip_fit.fitted_means[0], mean, places=7)
        self.assertAlmostEqual(hurdle.fitted_means[0], mean, places=7)

    def test_intercept_only_negative_binomial_versions_agree(self) -> None:
        """ZINB and the NB hurdle reach the same intercept-only maximum.

        References:
            [EQ-CNT-ZI-001], [EQ-CNT-HURDLE-001].

        """
        rows = [(1.0,)] * len(COUNTS)
        a = fit_zero_inflated(rows, COUNTS, family="negative-binomial")
        b = fit_hurdle(rows, COUNTS, family="negative-binomial")
        self.assertAlmostEqual(a.log_likelihood, b.log_likelihood, places=6)
        self.assertAlmostEqual(a.theta / b.theta, 1.0, places=4)

    def test_zip_standard_errors_equal_independent_observed_information(self) -> None:
        """Intercept-only ZIP: covariance equals an independently coded inverse Hessian.

        References:
            [EQ-CNT-ZI-001].

        """
        rows = [(1.0,)] * len(COUNTS)
        fit = fit_zero_inflated(rows, COUNTS)

        def log_likelihood(q: list[float]) -> float:
            """Independent ZIP log-likelihood in (log lambda, logit pi).

            References:
                [EQ-CNT-ZI-001].

            """
            lam = math.exp(q[0])
            pi = 1.0 / (1.0 + math.exp(-q[1]))
            total = 0.0
            for v in COUNTS:
                if v == 0:
                    total += math.log(pi + (1.0 - pi) * math.exp(-lam))
                else:
                    total += (
                        math.log(1.0 - pi)
                        + v * math.log(lam)
                        - lam
                        - math.lgamma(v + 1)
                    )
            return total

        point = [fit.count_coefficients[0], fit.zero_coefficients[0]]
        h = 1e-4
        hessian = [[0.0, 0.0], [0.0, 0.0]]
        for a in range(2):
            for b in range(2):
                values = []
                for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    q = list(point)
                    q[a] += sa * h
                    q[b] += sb * h
                    values.append(sa * sb * log_likelihood(q))
                hessian[a][b] = sum(values) / (4.0 * h * h)
        det = hessian[0][0] * hessian[1][1] - hessian[0][1] ** 2
        inverse = [
            [-hessian[1][1] / det, hessian[0][1] / det],
            [hessian[1][0] / det, -hessian[0][0] / det],
        ]
        self.assertAlmostEqual(fit.covariance[0][0] / inverse[0][0], 1.0, places=4)
        self.assertAlmostEqual(fit.covariance[1][1] / inverse[1][1], 1.0, places=4)
        self.assertAlmostEqual(fit.covariance[0][1] / inverse[0][1], 1.0, places=4)

    def test_zip_simulation_recovers_both_parts(self) -> None:
        """Seeded ZIP data with covariates: estimates within four SEs.

        References:
            [EQ-CNT-ZI-001], [REF-LAMBERT-1992].

        """
        rng = random.Random(1992)
        beta, gamma = (1.0, 0.3), (-0.5, 0.8)
        rows, y = [], []
        for _ in range(2000):
            x = rng.gauss(0.0, 1.0)
            pi = 1.0 / (1.0 + math.exp(-(gamma[0] + gamma[1] * x)))
            mu = math.exp(beta[0] + beta[1] * x)
            y.append(0 if rng.random() < pi else poisson_draw(rng, mu))
            rows.append((1.0, x))
        fit = fit_zero_inflated(rows, y)
        self.assertTrue(fit.converged)
        estimates = (*fit.count_coefficients, *fit.zero_coefficients)
        errors = (*fit.count_standard_errors, *fit.zero_standard_errors)
        for estimate, error, truth in zip(
            estimates, errors, (*beta, *gamma), strict=True
        ):
            self.assertLess(abs(estimate - truth), 4.0 * error)
        self.assertEqual(fit.parameter_count, 4)

    def test_zinb_and_nb_hurdle_simulation(self) -> None:
        """Seeded ZINB data: count, inflation, and log theta within four SEs.

        References:
            [EQ-CNT-ZI-001], [EQ-CNT-HURDLE-001].

        """
        rng = random.Random(31)
        beta, gamma, theta = (1.2, -0.4), (-1.0, 0.5), 2.5
        rows, y = [], []
        for _ in range(1500):
            x = rng.uniform(-1.5, 1.5)
            pi = 1.0 / (1.0 + math.exp(-(gamma[0] + gamma[1] * x)))
            mu = math.exp(beta[0] + beta[1] * x)
            y.append(0 if rng.random() < pi else nb_draw(rng, mu, theta))
            rows.append((1.0, x))
        fit = fit_zero_inflated(rows, y, family="negative-binomial")
        estimates = (*fit.count_coefficients, *fit.zero_coefficients)
        errors = (*fit.count_standard_errors, *fit.zero_standard_errors)
        for estimate, error, truth in zip(
            estimates, errors, (*beta, *gamma), strict=True
        ):
            self.assertLess(abs(estimate - truth), 4.0 * error)
        self.assertLess(
            abs(math.log(fit.theta) - math.log(theta)),
            4.0 * fit.log_theta_standard_error,
        )
        hurdle = fit_hurdle(rows, y, family="negative-binomial")
        self.assertTrue(hurdle.converged)
        self.assertEqual(hurdle.parameter_count, 5)

    def test_hurdle_count_part_is_stationary(self) -> None:
        """The truncated Poisson part solves its own score equation.

        References:
            [EQ-CNT-HURDLE-001], [REF-MULLAHY-1986].

        """
        rng = random.Random(4)
        rows = [(1.0, rng.uniform(-1.0, 1.0)) for _ in range(200)]
        y = [poisson_draw(rng, math.exp(0.3 + 0.7 * r[1])) for r in rows]
        fit = fit_hurdle(rows, y)
        pairs = [(r, v) for r, v in zip(rows, y, strict=True) if v > 0]

        def truncated(q: list[float]) -> float:
            """Independent zero-truncated Poisson log-likelihood.

            References:
                [EQ-CNT-HURDLE-001].

            """
            total = 0.0
            for r, v in pairs:
                mu = math.exp(q[0] + q[1] * r[1])
                total += (
                    v * math.log(mu)
                    - mu
                    - math.lgamma(v + 1)
                    - math.log(-math.expm1(-mu))
                )
            return total

        for value in finite_score(truncated, list(fit.count_coefficients)):
            self.assertLess(abs(value), 1e-5)
        binary = fit_glm(rows, [float(v > 0) for v in y], family="binomial")
        for a, b in zip(fit.zero_coefficients, binary.coefficients, strict=True):
            self.assertAlmostEqual(a, b, places=12)

    def test_zero_models_need_zeros(self) -> None:
        """Data without zeros are rejected.

        References:
            [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaises(ValueError):
            fit_zero_inflated([(1.0,)] * 4, [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            fit_hurdle([(1.0,)] * 4, [1, 2, 3, 4])


class VuongTests(unittest.TestCase):
    """The Vuong statistic against its defining formula.

    References:
        [EQ-CNT-VUONG-001], [REF-VUONG-1989].

    """

    def test_statistic_and_corrections(self) -> None:
        """V = (sum m - c)/(sqrt(n) omega) with omega on divisor n.

        References:
            [EQ-CNT-VUONG-001].

        """
        a = [-1.2, -0.5, -2.0, -0.7, -1.1, -0.3, -1.9]
        b = [-1.0, -0.9, -2.2, -1.5, -1.0, -0.8, -2.4]
        m = [x - y for x, y in zip(a, b, strict=True)]
        n = len(m)
        mean = sum(m) / n
        omega = math.sqrt(sum((v - mean) ** 2 for v in m) / n)
        for correction, shift in (
            ("none", 0.0),
            ("aic", 2.0),
            ("bic", 2.0 * math.log(n) / 2.0),
        ):
            test = vuong_test(
                a, b, parameters_a=4, parameters_b=2, correction=correction
            )
            expected = (sum(m) - shift) / (math.sqrt(n) * omega)
            self.assertAlmostEqual(test.statistic, expected, places=12)
            self.assertAlmostEqual(
                test.p_value, 2.0 * standard_normal_sf(abs(expected)), places=12
            )
        with self.assertRaises(ValueError):
            vuong_test(a, b[:-1], parameters_a=1, parameters_b=1)

    def test_accepts_pointwise_log_likelihoods_of_fits(self) -> None:
        """Pointwise terms of a ZIP and a Poisson fit sum to their log-likelihoods.

        References:
            [EQ-CNT-VUONG-001], [REF-WILSON-2015].

        """
        rows = [(1.0,)] * len(COUNTS)
        zip_fit = fit_zero_inflated(rows, COUNTS)
        poisson = fit_poisson_regression(rows, COUNTS)
        self.assertAlmostEqual(
            sum(zip_fit.pointwise_log_likelihood), zip_fit.log_likelihood, places=10
        )
        test = vuong_test(
            zip_fit.pointwise_log_likelihood,
            poisson.pointwise_log_likelihood,
            parameters_a=zip_fit.parameter_count,
            parameters_b=poisson.parameter_count,
        )
        self.assertGreater(test.statistic, 0.0)


class OrdinalLogisticTests(unittest.TestCase):
    """Proportional-odds model and the Brant test.

    References:
        [EQ-CAT-ORD-001], [EQ-CAT-BRANT-001].

    """

    def test_thresholds_only_fit_is_closed_form(self) -> None:
        """theta_k = logit(F_k), Cov(theta_k, theta_l) = 1/(n (1 - F_k) F_l) for k < l.

        References:
            [EQ-CAT-ORD-001].

        """
        y = ["low"] * 7 + ["mid"] * 5 + ["high"] * 8
        fit = fit_ordinal_logistic(None, y, levels=["low", "mid", "high"])
        n = len(y)
        shares = (7 / n, 12 / n)
        for theta, share in zip(fit.thresholds, shares, strict=True):
            self.assertAlmostEqual(theta, math.log(share / (1.0 - share)), places=9)
        self.assertAlmostEqual(
            fit.covariance[0][0], 1.0 / (n * shares[0] * (1.0 - shares[0])), places=8
        )
        self.assertAlmostEqual(
            fit.covariance[0][1], 1.0 / (n * (1.0 - shares[0]) * shares[1]), places=8
        )
        probabilities = fit.predict_proba()[0]
        for p, count in zip(probabilities, (7, 5, 8), strict=True):
            self.assertAlmostEqual(p, count / n, places=9)

    def test_two_categories_reproduce_logistic_regression(self) -> None:
        """J = 2: slopes equal the logit slopes and theta = -intercept.

        References:
            [EQ-CAT-ORD-001].

        """
        rng = random.Random(3)
        x = [rng.gauss(0.0, 1.0) for _ in range(120)]
        y = [int(rng.random() < 1.0 / (1.0 + math.exp(-(0.3 + 1.1 * v)))) for v in x]
        ordinal = fit_ordinal_logistic([(v,) for v in x], y)
        logit = fit_glm([(1.0, v) for v in x], [float(v) for v in y], family="binomial")
        self.assertAlmostEqual(ordinal.coefficients[0], logit.coefficients[1], places=8)
        self.assertAlmostEqual(ordinal.thresholds[0], -logit.coefficients[0], places=8)
        self.assertAlmostEqual(
            ordinal.standard_errors[0], logit.standard_errors[1], places=7
        )
        self.assertAlmostEqual(
            ordinal.standard_errors[1], logit.standard_errors[0], places=7
        )

    def test_simulation_and_marginal_effects(self) -> None:
        """Seeded latent-logistic data: recovery and AMEs that sum to zero.

        References:
            [EQ-CAT-ORD-001], [EQ-REG-AME-001].

        """
        rng = random.Random(1980)
        beta, cuts = (0.8, -0.6), (-0.5, 1.0)
        rows, y = [], []
        for _ in range(1200):
            x1, x2 = rng.gauss(0.0, 1.0), float(rng.random() < 0.4)
            u = rng.random()
            latent = beta[0] * x1 + beta[1] * x2 + math.log(u / (1.0 - u))
            y.append(0 if latent < cuts[0] else 1 if latent < cuts[1] else 2)
            rows.append((x1, x2))
        fit = fit_ordinal_logistic(rows, y, column_names=["x1", "x2"])
        truth = (*beta, *cuts)
        estimates = (*fit.coefficients, *fit.thresholds)
        for estimate, error, value in zip(
            estimates, fit.standard_errors, truth, strict=True
        ):
            self.assertLess(abs(estimate - value), 4.0 * error)
        effect = fit.average_marginal_effects("x1")
        self.assertAlmostEqual(sum(effect.estimates), 0.0, places=10)
        h = 1e-5
        up = fit.predict_proba([(a + h, b) for a, b in rows])
        down = fit.predict_proba([(a - h, b) for a, b in rows])
        for k in range(3):
            numeric = sum(u[k] - d[k] for u, d in zip(up, down, strict=True)) / (
                2 * h * len(rows)
            )
            self.assertAlmostEqual(effect.estimates[k], numeric, places=7)
        discrete = fit.average_marginal_effects("x2", discrete=True)
        high = fit.predict_proba([(a, 1.0) for a, _ in rows])
        low = fit.predict_proba([(a, 0.0) for a, _ in rows])
        for k in range(3):
            expected = sum(u[k] - d[k] for u, d in zip(high, low, strict=True)) / len(
                rows
            )
            self.assertAlmostEqual(discrete.estimates[k], expected, places=12)
        self.assertTrue(all(s > 0.0 for s in effect.standard_errors))

    def test_brant_statistic_matches_log_odds_ratio_form(self) -> None:
        """One binary covariate, three categories: the explicit Wald form.

        References:
            [EQ-CAT-BRANT-001], [REF-BRANT-1990].

        """
        table = {0.0: (14, 9, 7), 1.0: (6, 8, 15)}
        rows, y = [], []
        for group, counts in table.items():
            for category, count in enumerate(counts):
                rows.extend([(group,)] * count)
                y.extend([category] * count)
        fit = fit_ordinal_logistic(rows, y)
        test = brant_test(fit)
        slopes, variances, covariance = [], [0.0, 0.0], 0.0
        exceed = {}
        for group, counts in table.items():
            n = sum(counts)
            exceed[group] = ((counts[1] + counts[2]) / n, counts[2] / n, n)
        for cut in range(2):
            logits = [
                math.log(exceed[g][cut] / (1.0 - exceed[g][cut])) for g in (0.0, 1.0)
            ]
            slopes.append(logits[1] - logits[0])
            variances[cut] = sum(
                1.0 / (exceed[g][2] * exceed[g][cut] * (1.0 - exceed[g][cut]))
                for g in (0.0, 1.0)
            )
        covariance = sum(
            1.0 / (exceed[g][2] * exceed[g][0] * (1.0 - exceed[g][1]))
            for g in (0.0, 1.0)
        )
        expected = (slopes[0] - slopes[1]) ** 2 / (
            variances[0] + variances[1] - 2.0 * covariance
        )
        self.assertAlmostEqual(test.statistic, expected, places=8)
        self.assertEqual(test.degrees_of_freedom, 1)
        self.assertAlmostEqual(test.p_value, chi_square_sf(expected, 1), places=10)
        self.assertAlmostEqual(test.binary_slopes[0][0], slopes[0], places=8)

    def test_brant_detects_non_proportional_odds(self) -> None:
        """A slope that changes sign between the cuts gives a tiny p-value.

        References:
            [EQ-CAT-BRANT-001].

        """
        rng = random.Random(9)
        rows, y = [], []
        for _ in range(1000):
            x = rng.gauss(0.0, 1.0)
            low = 1.0 / (1.0 + math.exp(-(0.5 + 1.2 * x)))
            high = 1.0 / (1.0 + math.exp(-(-0.5 - 1.5 * x)))
            u = rng.random()
            above_first = u < low
            y.append(0 if not above_first else 2 if rng.random() < high else 1)
            rows.append((x,))
        test = brant_test(fit_ordinal_logistic(rows, y))
        self.assertLess(test.p_value, 1e-4)

    def test_design_validation(self) -> None:
        """An intercept column or an unobserved level is rejected.

        References:
            [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaises(ValueError):
            fit_ordinal_logistic([(1.0, 0.2), (1.0, 0.4), (1.0, 0.1)], [0, 1, 1])
        with self.assertRaises(ValueError):
            fit_ordinal_logistic(None, [0, 0, 2], levels=[0, 1, 2])
        with self.assertRaises(SeparationError):
            fit_ordinal_logistic([(float(i),) for i in range(10)], [0] * 5 + [1] * 5)


class MultinomialLogitTests(unittest.TestCase):
    """Baseline-category logit against saturated closed forms.

    References:
        [EQ-CAT-MNL-001].

    """

    def test_saturated_factor_model_is_closed_form(self) -> None:
        """Log ratios of counts, sqrt(sum 1/n) errors, proportion AMEs.

        References:
            [EQ-CAT-MNL-001], [EQ-REG-AME-001].

        """
        table = {0.0: (12, 7, 5), 1.0: (6, 9, 14)}
        rows, y = [], []
        for group, counts in table.items():
            for category, count in enumerate(counts):
                rows.extend([(1.0, group)] * count)
                y.extend([("a", "b", "c")[category]] * count)
        fit = fit_multinomial_logit(rows, y, column_names=["(Intercept)", "g"])
        self.assertEqual(fit.levels, ("a", "b", "c"))
        c0, c1 = table[0.0], table[1.0]
        for k in (1, 2):
            intercept = math.log(c0[k] / c0[0])
            slope = math.log(c1[k] * c0[0] / (c1[0] * c0[k]))
            self.assertAlmostEqual(fit.coefficients[k - 1][0], intercept, places=8)
            self.assertAlmostEqual(fit.coefficients[k - 1][1], slope, places=8)
            self.assertAlmostEqual(
                fit.standard_errors[k - 1][0],
                math.sqrt(1 / c0[k] + 1 / c0[0]),
                places=7,
            )
            self.assertAlmostEqual(
                fit.standard_errors[k - 1][1],
                math.sqrt(1 / c0[k] + 1 / c0[0] + 1 / c1[k] + 1 / c1[0]),
                places=7,
            )
        effect = fit.average_marginal_effects("g", discrete=True)
        n0, n1 = sum(c0), sum(c1)
        for k in range(3):
            p0, p1 = c0[k] / n0, c1[k] / n1
            self.assertAlmostEqual(effect.estimates[k], p1 - p0, places=8)
            self.assertAlmostEqual(
                effect.standard_errors[k],
                math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0),
                places=6,
            )

    def test_reference_level_and_two_category_identity(self) -> None:
        """K = 2 equals logistic regression; changing the reference flips signs.

        References:
            [EQ-CAT-MNL-001].

        """
        rng = random.Random(12)
        x = [rng.uniform(-2.0, 2.0) for _ in range(150)]
        y = [
            "yes" if rng.random() < 1 / (1 + math.exp(-(0.2 - 0.9 * v))) else "no"
            for v in x
        ]
        rows = [(1.0, v) for v in x]
        fit = fit_multinomial_logit(rows, y)
        logit = fit_glm(rows, [float(v == "yes") for v in y], family="binomial")
        for a, b in zip(fit.coefficients[0], logit.coefficients, strict=True):
            self.assertAlmostEqual(a, b, places=8)
        flipped = fit_multinomial_logit(rows, y, reference="yes")
        self.assertEqual(flipped.levels, ("yes", "no"))
        for a, b in zip(flipped.coefficients[0], logit.coefficients, strict=True):
            self.assertAlmostEqual(a, -b, places=8)

    def test_derivative_effects_match_numeric_derivatives(self) -> None:
        """Analytic AMEs equal central differences of the mean probabilities.

        References:
            [EQ-REG-AME-001].

        """
        rng = random.Random(21)
        rows, y = [], []
        for _ in range(400):
            x = rng.gauss(0.0, 1.0)
            etas = (0.0, 0.3 + 0.8 * x, -0.2 - 0.5 * x)
            weights = [math.exp(e) for e in etas]
            u = rng.random() * sum(weights)
            category = 0 if u < weights[0] else 1 if u < weights[0] + weights[1] else 2
            rows.append((1.0, x))
            y.append(category)
        fit = fit_multinomial_logit(rows, y)
        effect = fit.average_marginal_effects(1)
        h = 1e-5
        up = fit.predict_proba([(1.0, x + h) for _, x in rows])
        down = fit.predict_proba([(1.0, x - h) for _, x in rows])
        for k in range(3):
            numeric = sum(a[k] - b[k] for a, b in zip(up, down, strict=True)) / (
                2 * h * len(rows)
            )
            self.assertAlmostEqual(effect.estimates[k], numeric, places=7)
        with self.assertRaises(ValueError):
            fit.average_marginal_effects(0)
        with self.assertRaises(SeparationError):
            fit_multinomial_logit(
                [(1.0, float(i)) for i in range(10)], [0] * 5 + [1] * 5
            )


if __name__ == "__main__":
    unittest.main()
