"""Tests for right-censored survival analysis in ``probstat.inference.survival``.

Reference values are recomputed inside each test from the defining formulas:
the product-limit and Greenwood sums by hand, the log-rank hypergeometric
moments by explicit loops, Cox estimates by an independent scalar Newton
iteration on the partial likelihood, and Efron's tie correction term by term.
Stochastic checks use fixed seeds and tolerances of a few standard errors.
"""

from __future__ import annotations

import math
import unittest
from statistics import NormalDist

from probstat.foundations import RandomStream
from probstat.inference import fit_weibull_mle
from probstat.inference.survival import (
    SurvivalData,
    cox_partial_likelihood,
    fit_cox_model,
    fit_parametric_survival,
    kaplan_meier,
    log_rank_test,
    nelson_aalen,
    survival_data,
)

# Small hand example: (time, event) with a censoring tied to an event at t=2.
HAND_TIMES = (1, 2, 2, 3, 4, 5, 5, 6)
HAND_EVENTS = (1, 1, 0, 1, 0, 1, 1, 0)


def _weibull_sample(
    size: int, shape: float, scale: float, censor_max: float, seed: int
) -> SurvivalData:
    """Draw Weibull lifetimes with independent uniform censoring times."""
    generator = RandomStream(seed).generator()
    times: list[float] = []
    events: list[bool] = []
    for _ in range(size):
        lifetime = scale * (-math.log(1.0 - generator.random())) ** (1.0 / shape)
        censor = censor_max * generator.random()
        times.append(min(lifetime, censor))
        events.append(lifetime <= censor)
    return survival_data(times, events)


class SurvivalDataTests(unittest.TestCase):
    """The censored sample record and its risk-set table."""

    def test_risk_table_counts_censored_subjects_at_their_time(self) -> None:
        """A subject censored at ``t`` is still at risk for events at ``t``."""
        data = survival_data(HAND_TIMES, HAND_EVENTS)
        table = data.risk_table()
        self.assertEqual([row.time for row in table], [1, 2, 3, 4, 5, 6])
        self.assertEqual([row.at_risk for row in table], [8, 7, 5, 4, 3, 1])
        self.assertEqual([row.events for row in table], [1, 1, 1, 0, 2, 0])
        self.assertEqual([row.censored for row in table], [0, 1, 0, 1, 0, 1])
        self.assertEqual((data.size, data.event_count, data.censored_count), (8, 5, 3))

    def test_validation(self) -> None:
        """Bad times, indicators, and lengths are rejected."""
        with self.assertRaises(ValueError):
            survival_data([1.0, -2.0], [1, 1])
        with self.assertRaises(ValueError):
            survival_data([1.0, 2.0], [1])
        with self.assertRaises(ValueError):
            survival_data([1.0], [2])
        with self.assertRaises(TypeError):
            survival_data([1.0], ["yes"])
        with self.assertRaises(ValueError):
            survival_data([], [])


class KaplanMeierTests(unittest.TestCase):
    """Product-limit values, Greenwood variance, log-log intervals, summaries."""

    def setUp(self) -> None:
        """Fit the hand example once."""
        self.data = survival_data(HAND_TIMES, HAND_EVENTS)
        self.km = kaplan_meier(self.data)

    def test_product_limit_values_by_hand(self) -> None:
        """``S = prod (1 - d/n)`` at the event times 1, 2, 3, 5."""
        s1 = 1 - 1 / 8
        s2 = s1 * (1 - 1 / 7)
        s3 = s2 * (1 - 1 / 5)
        s5 = s3 * (1 - 2 / 3)
        self.assertEqual(self.km.times, (1, 2, 3, 5))
        for got, want in zip(self.km.survival, (s1, s2, s3, s5), strict=True):
            self.assertAlmostEqual(got, want, places=14)
        self.assertAlmostEqual(s5, 0.2, places=14)
        self.assertEqual(self.km.survival_at(0.5), 1.0)
        self.assertAlmostEqual(self.km.survival_at(4.9), 0.6, places=14)
        self.assertAlmostEqual(self.km.survival_at(10.0), 0.2, places=14)

    def test_greenwood_variance_formula(self) -> None:
        """``Var S(t) = S^2 sum d / (n (n - d))``."""
        sums = (
            1 / (8 * 7),
            1 / (8 * 7) + 1 / (7 * 6),
            1 / (8 * 7) + 1 / (7 * 6) + 1 / (5 * 4),
            1 / (8 * 7) + 1 / (7 * 6) + 1 / (5 * 4) + 2 / (3 * 1),
        )
        for index, total in enumerate(sums):
            s = self.km.survival[index]
            self.assertAlmostEqual(self.km.greenwood_sums[index], total, places=14)
            self.assertAlmostEqual(self.km.variance[index], s * s * total, places=14)
            self.assertAlmostEqual(
                self.km.standard_errors[index], s * math.sqrt(total), places=14
            )

    def test_log_log_interval(self) -> None:
        """The interval is ``S^{exp(+/- z sqrt(sum)/|log S|)}`` and brackets S."""
        z = NormalDist().inv_cdf(0.975)
        s = 0.6
        total = 1 / 56 + 1 / 42 + 1 / 20
        width = math.sqrt(total) / abs(math.log(s))
        lower, upper = self.km.interval_at(3.0)
        self.assertAlmostEqual(lower, s ** math.exp(z * width), places=12)
        self.assertAlmostEqual(upper, s ** math.exp(-z * width), places=12)
        for low, s_j, high in zip(
            self.km.lower, self.km.survival, self.km.upper, strict=True
        ):
            self.assertTrue(0.0 < low < s_j < high < 1.0)
        self.assertEqual(self.km.interval_at(0.0), (1.0, 1.0))

    def test_curve_reaching_zero(self) -> None:
        """When the last subject fails, ``S = 0`` with zero variance."""
        km = kaplan_meier(survival_data([1, 2, 3], [1, 0, 1]))
        self.assertEqual(km.survival[-1], 0.0)
        self.assertEqual(km.variance[-1], 0.0)
        self.assertEqual((km.lower[-1], km.upper[-1]), (0.0, 0.0))
        self.assertTrue(math.isinf(km.greenwood_sums[-1]))

    def test_median_and_quantiles(self) -> None:
        """The median is the first time with ``S <= 0.5``; CI inverts the bands."""
        median = self.km.median()
        self.assertEqual(median.estimate, 5)
        expected_lower = next(
            t for t, low in zip(self.km.times, self.km.lower, strict=True) if low <= 0.5
        )
        self.assertEqual(median.lower, expected_lower)
        self.assertTrue(median.reached)
        self.assertLessEqual(median.lower, median.estimate)
        self.assertLessEqual(median.estimate, median.upper)
        self.assertEqual(self.km.quantile(0.2).estimate, 2)
        never = kaplan_meier(survival_data([1, 2, 3, 4], [1, 0, 0, 0])).median()
        self.assertFalse(never.reached)
        self.assertTrue(math.isinf(never.upper))
        with self.assertRaises(ValueError):
            self.km.quantile(1.0)

    def test_restricted_mean_survival_time(self) -> None:
        """RMST is the area under the step curve with the Greenwood-type variance."""
        rmst = self.km.restricted_mean(6.0)
        s = (1.0, 7 / 8, 0.75, 0.6, 0.2)
        area = s[0] * 1 + s[1] * 1 + s[2] * 1 + s[3] * 2 + s[4] * 1
        self.assertAlmostEqual(rmst.estimate, area, places=14)
        self.assertAlmostEqual(area, 4.025, places=14)
        tails = (
            s[1] * 1 + s[2] * 1 + s[3] * 2 + s[4] * 1,  # from t=1
            s[2] * 1 + s[3] * 2 + s[4] * 1,  # from t=2
            s[3] * 2 + s[4] * 1,  # from t=3
            s[4] * 1,  # from t=5
        )
        terms = (1 / (8 * 7), 1 / (7 * 6), 1 / (5 * 4), 2 / (3 * 1))
        variance = sum(a * a * w for a, w in zip(tails, terms, strict=True))
        self.assertAlmostEqual(rmst.variance, variance, places=12)
        z = NormalDist().inv_cdf(0.975)
        self.assertAlmostEqual(rmst.upper - rmst.estimate, z * math.sqrt(variance))
        self.assertAlmostEqual(self.km.restricted_mean(0.5).estimate, 0.5)
        with self.assertRaises(ValueError):
            self.km.restricted_mean(7.0)


class NelsonAalenTests(unittest.TestCase):
    """Cumulative hazard increments ``d/n`` and variance ``d/n^2``."""

    def test_hand_values(self) -> None:
        """Nelson-Aalen sums on the hand example."""
        na = nelson_aalen(survival_data(HAND_TIMES, HAND_EVENTS))
        increments = (1 / 8, 1 / 7, 1 / 5, 2 / 3)
        spreads = (1 / 64, 1 / 49, 1 / 25, 2 / 9)
        for j in range(4):
            self.assertAlmostEqual(
                na.cumulative_hazard[j], sum(increments[: j + 1]), places=14
            )
            self.assertAlmostEqual(na.variance[j], sum(spreads[: j + 1]), places=14)
            self.assertLess(na.lower[j], na.cumulative_hazard[j])
            self.assertGreater(na.upper[j], na.cumulative_hazard[j])
        self.assertEqual(na.cumulative_hazard_at(0.9), 0.0)
        self.assertAlmostEqual(na.survival_at(5.0), math.exp(-sum(increments)))
        km = kaplan_meier(survival_data(HAND_TIMES, HAND_EVENTS))
        self.assertGreater(na.survival_at(5.0), km.survival_at(5.0))


class LogRankTests(unittest.TestCase):
    """Two- and k-sample log-rank statistics against explicit formulas."""

    A = survival_data([3, 5, 5, 8, 10, 12], [1, 1, 1, 1, 0, 1])
    B = survival_data([1, 2, 4, 4, 6, 7, 9], [1, 1, 1, 0, 1, 1, 0])
    C = survival_data([2, 6, 11, 13, 14], [1, 0, 1, 1, 1])

    @staticmethod
    def _moments(groups: list[SurvivalData]) -> tuple[list[float], list[list[float]]]:
        """Return ``O - E`` and the hypergeometric covariance by direct loops."""
        k = len(groups)
        times = sorted(
            {t for g in groups for t, e in zip(g.times, g.events, strict=True) if e}
        )
        diff = [0.0] * k
        cov = [[0.0] * k for _ in range(k)]
        for t in times:
            n_g = [sum(1 for u in g.times if u >= t) for g in groups]
            d_g = [
                sum(1 for u, e in zip(g.times, g.events, strict=True) if e and u == t)
                for g in groups
            ]
            n, d = sum(n_g), sum(d_g)
            for a in range(k):
                diff[a] += d_g[a] - d * n_g[a] / n
                for b in range(k):
                    if n > 1:
                        delta = 1.0 if a == b else 0.0
                        cov[a][b] += (
                            d * (n - d) / (n - 1) * n_g[a] / n * (delta - n_g[b] / n)
                        )
        return diff, cov

    def test_two_group_statistic(self) -> None:
        """``(O_1 - E_1)^2 / V_11`` with ``V = n1 n2 d (n-d) / (n^2 (n-1))``."""
        diff, cov = self._moments([self.A, self.B])
        result = log_rank_test(self.A, self.B)
        self.assertAlmostEqual(result.statistic, diff[0] ** 2 / cov[0][0], places=12)
        self.assertEqual(result.degrees_of_freedom, 1)
        self.assertEqual(result.observed, (5, 5))
        self.assertAlmostEqual(sum(result.expected), 10.0, places=12)
        self.assertAlmostEqual(
            result.p_value, math.erfc(math.sqrt(result.statistic / 2.0)), places=12
        )
        swapped = log_rank_test(self.B, self.A)
        self.assertAlmostEqual(swapped.statistic, result.statistic, places=12)

    def test_three_group_statistic(self) -> None:
        """``U' V^-1 U`` on the first two groups with an explicit 2x2 inverse."""
        diff, cov = self._moments([self.A, self.B, self.C])
        det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0]
        quad = (
            cov[1][1] * diff[0] ** 2
            - 2.0 * cov[0][1] * diff[0] * diff[1]
            + cov[0][0] * diff[1] ** 2
        ) / det
        result = log_rank_test(self.A, self.B, self.C)
        self.assertAlmostEqual(result.statistic, quad, places=10)
        self.assertEqual(result.degrees_of_freedom, 2)
        self.assertAlmostEqual(result.p_value, math.exp(-quad / 2.0), places=10)

    def test_errors(self) -> None:
        """One group or no event is rejected."""
        with self.assertRaises(ValueError):
            log_rank_test(self.A)
        with self.assertRaises(ValueError):
            log_rank_test(survival_data([1], [0]), survival_data([2], [0]))


class ParametricSurvivalTests(unittest.TestCase):
    """Censored AFT maximum likelihood for exponential, Weibull, lognormal."""

    def test_exponential_closed_form(self) -> None:
        """Censored exponential MLE: rate ``d / sum t`` and ``se = 1/sqrt(d)``."""
        data = survival_data(HAND_TIMES, HAND_EVENTS)
        fit = fit_parametric_survival(data, "exponential")
        total = sum(HAND_TIMES)
        self.assertAlmostEqual(fit.estimates[0], math.log(total / 5), places=7)
        self.assertAlmostEqual(fit.standard_errors[0], 1 / math.sqrt(5), places=5)
        law = fit.distribution()
        self.assertAlmostEqual(law.rate, 5 / total, places=7)
        loglik = 5 * math.log(5 / total) - 5
        self.assertAlmostEqual(fit.log_likelihood, loglik, places=8)

    def test_uncensored_fits_match_closed_forms(self) -> None:
        """Without censoring, AFT fits equal the ordinary lognormal/Weibull MLEs."""
        generator = RandomStream(11).generator()
        times = [2.0 * (-math.log(1.0 - generator.random())) ** 0.7 for _ in range(60)]
        data = survival_data(times, [1] * len(times))
        logs = [math.log(t) for t in times]
        mu = sum(logs) / len(logs)
        sigma = math.sqrt(sum((v - mu) ** 2 for v in logs) / len(logs))
        lognormal = fit_parametric_survival(data, "lognormal")
        self.assertAlmostEqual(lognormal.estimates[0], mu, places=6)
        self.assertAlmostEqual(lognormal.sigma, sigma, places=6)
        self.assertAlmostEqual(lognormal.standard_errors[0], sigma / math.sqrt(60), 4)
        weibull = fit_parametric_survival(data, "weibull")
        reference = fit_weibull_mle(times)
        shape, _, scale, _ = weibull.weibull_parameters()
        self.assertAlmostEqual(shape, reference.shape, places=5)
        self.assertAlmostEqual(scale, reference.scale, places=5)

    def test_weibull_censored_mle_recovers_seeded_parameters(self) -> None:
        """Seeded Weibull(1.7, 5) with ~30% censoring is recovered within 3 SE."""
        data = _weibull_sample(500, 1.7, 5.0, 14.0, seed=2026)
        censored = data.censored_count / data.size
        self.assertTrue(0.15 < censored < 0.45, censored)
        fit = fit_parametric_survival(data, "weibull")
        self.assertTrue(fit.converged)
        shape, shape_se, scale, scale_se = fit.weibull_parameters()
        self.assertLess(abs(shape - 1.7), 3.0 * shape_se)
        self.assertLess(abs(scale - 5.0), 3.0 * scale_se)
        self.assertLess(shape_se, 0.15)
        law = fit.distribution()
        self.assertAlmostEqual(law.shape, shape)
        self.assertAlmostEqual(fit.survival(5.0), math.exp(-((5.0 / scale) ** shape)))
        weibull_aic = fit.aic
        exponential = fit_parametric_survival(data, "exponential")
        self.assertLess(weibull_aic, exponential.aic)

    def test_covariate_time_ratio(self) -> None:
        """A seeded AFT covariate effect is recovered within 3 standard errors."""
        generator = RandomStream(7).generator()
        times: list[float] = []
        events: list[bool] = []
        rows: list[list[float]] = []
        for i in range(400):
            x = float(i % 2)
            scale = math.exp(1.0 + 0.6 * x)
            lifetime = scale * (-math.log(1.0 - generator.random())) ** (1.0 / 1.5)
            censor = 12.0 * generator.random()
            times.append(min(lifetime, censor))
            events.append(lifetime <= censor)
            rows.append([x])
        data = survival_data(times, events)
        fit = fit_parametric_survival(
            data, "weibull", covariates=rows, covariate_names=["treated"]
        )
        self.assertEqual(fit.parameter_names, ("intercept", "treated", "log_sigma"))
        self.assertLess(abs(fit.estimates[1] - 0.6), 3.0 * fit.standard_errors[1])
        self.assertLess(abs(fit.sigma - 1 / 1.5), 3.0 * fit.sigma_standard_error)
        self.assertGreater(fit.survival(3.0, [1.0]), fit.survival(3.0, [0.0]))

    def test_errors(self) -> None:
        """Zero times, unknown families, and event-free data are rejected."""
        with self.assertRaises(ValueError):
            fit_parametric_survival(survival_data([0.0, 1.0], [1, 1]))
        with self.assertRaises(ValueError):
            fit_parametric_survival(survival_data([1.0], [1]), "gamma")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_parametric_survival(survival_data([1.0, 2.0], [0, 0]))


COX_TIMES = (2, 3, 5, 7, 8, 11, 13, 17, 19, 23)
COX_EVENTS = (1, 1, 0, 1, 1, 0, 1, 1, 0, 1)
COX_X = (0.5, 1.2, -0.3, 0.8, -1.0, 0.1, -0.6, 0.0, -1.4, -0.9)


def _scalar_cox_derivatives(beta: float) -> tuple[float, float, float]:
    """Return ``l, U, I`` of the untied one-covariate partial likelihood."""
    value = score = information = 0.0
    for t_i, e_i, x_i in zip(COX_TIMES, COX_EVENTS, COX_X, strict=True):
        if not e_i:
            continue
        risk = [x for t, x in zip(COX_TIMES, COX_X, strict=True) if t >= t_i]
        s0 = sum(math.exp(beta * x) for x in risk)
        s1 = sum(x * math.exp(beta * x) for x in risk)
        s2 = sum(x * x * math.exp(beta * x) for x in risk)
        value += beta * x_i - math.log(s0)
        score += x_i - s1 / s0
        information += s2 / s0 - (s1 / s0) ** 2
    return value, score, information


class CoxModelTests(unittest.TestCase):
    """Newton-Raphson Cox regression, ties, and Wald inference."""

    def test_matches_independent_scalar_newton(self) -> None:
        """Library estimate and SE equal an in-test Newton iteration."""
        beta = 0.0
        for _ in range(50):
            _, score, information = _scalar_cox_derivatives(beta)
            beta += score / information
        value, score, information = _scalar_cox_derivatives(beta)
        self.assertLess(abs(score), 1e-12)
        data = survival_data(COX_TIMES, COX_EVENTS)
        rows = [[x] for x in COX_X]
        for ties in ("breslow", "efron"):
            fit = fit_cox_model(data, rows, ties=ties)  # type: ignore[arg-type]
            self.assertTrue(fit.converged)
            self.assertAlmostEqual(fit.coefficients[0], beta, places=8)
            self.assertAlmostEqual(
                fit.standard_errors[0], 1.0 / math.sqrt(information), places=8
            )
            self.assertAlmostEqual(fit.log_partial_likelihood, value, places=10)
        z = NormalDist().inv_cdf(0.975)
        se = 1.0 / math.sqrt(information)
        self.assertAlmostEqual(fit.hazard_ratios[0], math.exp(beta), places=8)
        self.assertAlmostEqual(fit.hazard_ratio_lower[0], math.exp(beta - z * se), 8)
        self.assertAlmostEqual(fit.hazard_ratio_upper[0], math.exp(beta + z * se), 8)
        null_value, null_score, null_info = _scalar_cox_derivatives(0.0)
        self.assertAlmostEqual(fit.likelihood_ratio, 2 * (value - null_value), 8)
        self.assertAlmostEqual(fit.score_statistic, null_score**2 / null_info, 10)

    def test_score_test_equals_log_rank_without_ties(self) -> None:
        """With untied times and a group indicator, the Cox score test is log-rank."""
        group_a = survival_data([1, 4, 6, 9, 12], [1, 1, 0, 1, 1])
        group_b = survival_data([2, 3, 5, 7, 8, 10], [1, 1, 1, 0, 1, 1])
        pooled = survival_data(
            group_a.times + group_b.times, group_a.events + group_b.events
        )
        rows = [[1.0]] * 5 + [[0.0]] * 6
        fit = fit_cox_model(pooled, rows, ties="breslow")
        self.assertAlmostEqual(
            fit.score_statistic, log_rank_test(group_a, group_b).statistic, places=10
        )

    def test_partial_likelihood_derivatives(self) -> None:
        """Score and information agree with central differences (two covariates)."""
        data = survival_data([1, 1, 2, 3, 3, 3, 5, 6], [1, 1, 0, 1, 1, 1, 1, 0])
        rows = [
            [0.2, 1],
            [1.0, 0],
            [-0.5, 1],
            [0.3, 0],
            [0.9, 1],
            [-1.1, 0],
            [0.4, 1],
            [0.0, 0],
        ]
        beta = (0.3, -0.4)
        h = 1e-6
        for ties in ("breslow", "efron"):
            base = cox_partial_likelihood(data, rows, beta, ties=ties)  # type: ignore[arg-type]
            for k in range(2):
                up = list(beta)
                down = list(beta)
                up[k] += h
                down[k] -= h
                plus = cox_partial_likelihood(data, rows, up, ties=ties)  # type: ignore[arg-type]
                minus = cox_partial_likelihood(data, rows, down, ties=ties)  # type: ignore[arg-type]
                self.assertAlmostEqual(
                    base.score[k], (plus.value - minus.value) / (2 * h), places=6
                )
                for j in range(2):
                    numeric = -(plus.score[j] - minus.score[j]) / (2 * h)
                    self.assertAlmostEqual(base.information[k][j], numeric, places=5)

    def test_efron_and_breslow_tie_handling(self) -> None:
        """Efron's ``sum_r log(S0(R) - r/d S0(D))`` term by term; Breslow differs."""
        times = (1, 1, 2, 2, 2, 3, 4, 4)
        events = (1, 1, 1, 1, 0, 1, 1, 0)
        xs = (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0)
        beta = 0.4
        data = survival_data(times, events)
        rows = [[x] for x in xs]
        efron = breslow = 0.0
        for t in sorted({t for t, e in zip(times, events, strict=True) if e}):
            risk = [
                math.exp(beta * x) for u, x in zip(times, xs, strict=True) if u >= t
            ]
            dead = [
                (x, math.exp(beta * x))
                for u, e, x in zip(times, events, xs, strict=True)
                if e and u == t
            ]
            d = len(dead)
            linear = sum(beta * x for x, _ in dead)
            s0_r = sum(risk)
            s0_d = sum(w for _, w in dead)
            efron += linear - sum(math.log(s0_r - r / d * s0_d) for r in range(d))
            breslow += linear - d * math.log(s0_r)
        got_e = cox_partial_likelihood(data, rows, [beta], ties="efron")
        got_b = cox_partial_likelihood(data, rows, [beta], ties="breslow")
        self.assertAlmostEqual(got_e.value, efron, places=12)
        self.assertAlmostEqual(got_b.value, breslow, places=12)
        self.assertGreater(got_e.value, got_b.value)
        fit_e = fit_cox_model(data, rows, ties="efron")
        fit_b = fit_cox_model(data, rows, ties="breslow")
        self.assertNotAlmostEqual(fit_e.coefficients[0], fit_b.coefficients[0], 3)
        # Without ties the two approximations are the same likelihood.
        untied = survival_data(COX_TIMES, COX_EVENTS)
        untied_rows = [[x] for x in COX_X]
        self.assertAlmostEqual(
            cox_partial_likelihood(untied, untied_rows, [0.7], ties="efron").value,
            cox_partial_likelihood(untied, untied_rows, [0.7], ties="breslow").value,
            places=13,
        )

    def test_breslow_baseline_and_prediction(self) -> None:
        """``H_0(t) = sum d_j / sum_{R_j} exp(x' beta)``; predictions use it."""
        data = survival_data(HAND_TIMES, HAND_EVENTS)
        xs = [float(i % 2) for i in range(8)]
        fit = fit_cox_model(data, [[x] for x in xs])
        beta = fit.coefficients[0]
        self.assertEqual(fit.baseline_times, (1, 2, 3, 5))
        h0 = []
        running = 0.0
        for row in data.event_table():
            risk = sum(
                math.exp(beta * x)
                for u, x in zip(HAND_TIMES, xs, strict=True)
                if u >= row.time
            )
            running += row.events / risk
            h0.append(running)
        for got, want in zip(fit.baseline_cumulative_hazard, h0, strict=True):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(fit.survival(0.5, [1.0]), 1.0)
        self.assertAlmostEqual(
            fit.survival(4.0, [1.0]), math.exp(-h0[2] * math.exp(beta)), places=12
        )

    def test_median_on_flat_half_segment_is_midpoint(self) -> None:
        """Uncensored n = 20 gives the sample median (t10 + t11) / 2."""
        even = kaplan_meier(
            survival_data([float(i) for i in range(1, 21)], [True] * 20)
        )
        self.assertAlmostEqual(even.median().estimate, 10.5, places=12)
        odd = kaplan_meier(survival_data([float(i) for i in range(1, 22)], [True] * 21))
        self.assertAlmostEqual(odd.median().estimate, 11.0, places=12)

    def test_cox_fit_is_invariant_to_covariate_shift(self) -> None:
        """Uncentred covariates (|x' beta| > 709) neither overflow nor shift S."""
        rng = RandomStream(7).generator()
        xs = [rng.gauss(0.0, 1.0) for _ in range(60)]
        times = [-math.log(1.0 - rng.random()) / math.exp(x) for x in xs]
        data = survival_data(times, [True] * 60)
        base = fit_cox_model(data, [[x] for x in xs])
        self.assertEqual(base.baseline_reference, 0.0)
        for shift in (800.0, -800.0):
            fit = fit_cox_model(data, [[x + shift] for x in xs])
            self.assertAlmostEqual(fit.coefficients[0], base.coefficients[0], places=6)
            self.assertAlmostEqual(
                fit.survival(0.5, [0.3 + shift]),
                base.survival(0.5, [0.3]),
                places=9,
            )

    def test_errors(self) -> None:
        """Constant covariates, bad ties labels, and shapes are rejected."""
        data = survival_data(COX_TIMES, COX_EVENTS)
        with self.assertRaises(ValueError):
            fit_cox_model(data, [[1.0]] * 10)
        with self.assertRaises(ValueError):
            fit_cox_model(data, [[x] for x in COX_X], ties="exact")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_cox_model(data, [[1.0]] * 3)


if __name__ == "__main__":
    unittest.main()
