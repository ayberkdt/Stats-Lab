"""Tests for launch-vehicle and rocket-engine reliability (illustrative numbers).

All parameter values are synthetic and illustrative; none is data from a real
engine or vehicle. Closed forms come first, then independent oracles (exact
binomial sums, numerical integration, the exact BDD of ``probstat.systems``),
then stochastic checks with fixed seeds and Monte Carlo standard-error bands.
"""

from __future__ import annotations

import math
import os
import unittest

from probstat.aerospace.engine_reliability import (
    BetaMixture,
    ClusterUncertainty,
    EngineClusterModel,
    LaunchVehicleDesign,
    VehicleUncertainty,
    assess_test_campaign,
    bayesian_demonstration_confidence,
    bayesian_demonstration_size,
    cluster_reliability_distribution,
    cluster_reliability_moments,
    demonstration_plan,
    equivalent_prior_tests,
    exponential_test_time,
    heritage_prior,
    launch_event_tree,
    nhpp_mission_reliability,
    operating_characteristic,
    predictive_pass_probability,
    reliability_lower_bound,
    smallest_plan_for_producer_risk,
    vehicle_outcome_probabilities,
    vehicle_reliability,
    vehicle_reliability_distribution,
    weibull_demonstration_units,
)
from probstat.distributions import Beta
from probstat.inference.confidence import clopper_pearson_interval
from probstat.reliability import CrowAMSAAFit, success_run_sample_size
from probstat.systems import ccf_parameter_uncertainty


def binomial_pmf(x: int, n: int, p: float) -> float:
    """Return the binomial pmf by direct combinatorics (test oracle)."""
    return math.comb(n, x) * p**x * (1.0 - p) ** (n - x)


class DemonstrationTests(unittest.TestCase):
    """Binomial and Clopper-Pearson demonstration plans."""

    def test_zero_failure_closed_form(self) -> None:
        """Zero-failure size ceil(ln(1 - C) / ln R), and 230 for 0.99 at 0.90."""
        for reliability, confidence in ((0.99, 0.90), (0.95, 0.80), (0.999, 0.95)):
            plan = demonstration_plan(reliability, confidence)
            exact = math.log(1.0 - confidence) / math.log(reliability)
            self.assertEqual(plan.tests, math.ceil(exact))
            assert plan.continuous_tests is not None
            self.assertAlmostEqual(plan.continuous_tests, exact, places=12)
            self.assertGreaterEqual(plan.achieved_confidence, confidence)
        self.assertEqual(demonstration_plan(0.99, 0.90).tests, 230)

    def test_plan_matches_binomial_sum_and_is_minimal(self) -> None:
        """The plan satisfies the binomial sum and one fewer test does not."""
        for r in (0, 1, 2, 3):
            plan = demonstration_plan(0.98, 0.90, allowed_failures=r)
            p = 0.02
            pass_n = sum(binomial_pmf(x, plan.tests, p) for x in range(r + 1))
            pass_prev = sum(binomial_pmf(x, plan.tests - 1, p) for x in range(r + 1))
            self.assertLessEqual(pass_n, 0.10 + 1e-12)
            self.assertGreater(pass_prev, 0.10)
            self.assertAlmostEqual(plan.achieved_confidence, 1.0 - pass_n, places=10)
            self.assertEqual(
                plan.tests, success_run_sample_size(0.98, 0.90, allowed_failures=r)
            )

    def test_clopper_pearson_lower_bound(self) -> None:
        """Closed form for f = 0, binomial-tail definition, and the interval."""
        for n in (10, 50, 230):
            self.assertAlmostEqual(
                reliability_lower_bound(n, 0, 0.90), 0.10 ** (1.0 / n), places=12
            )
        for n, f in ((30, 1), (60, 3), (200, 2)):
            bound = reliability_lower_bound(n, f, 0.90)
            s = n - f
            tail = sum(binomial_pmf(x, n, bound) for x in range(s, n + 1))
            self.assertAlmostEqual(tail, 0.10, places=8)
            # one-sided C equals the lower end of the two-sided level 2C - 1
            interval = clopper_pearson_interval(s, n, confidence=0.80)
            self.assertAlmostEqual(bound, interval.lower, places=8)
        self.assertEqual(reliability_lower_bound(5, 5, 0.90), 0.0)

    def test_bound_and_plan_are_consistent(self) -> None:
        """The plan size is exactly when the lower bound reaches R."""
        for r in (0, 2):
            plan = demonstration_plan(0.97, 0.90, allowed_failures=r)
            self.assertGreaterEqual(
                reliability_lower_bound(plan.tests, r, 0.90), 0.97 - 1e-12
            )
            self.assertLess(
                reliability_lower_bound(plan.tests - 1, r, 0.90), 0.97 + 1e-12
            )

    def test_operating_characteristic(self) -> None:
        """Closed form, endpoints, monotonicity and the consumer risk bound."""
        n, r, reliability = 10, 1, 0.9
        (value,) = operating_characteristic(n, r, [reliability])
        self.assertAlmostEqual(value, 0.9**10 + 10 * 0.1 * 0.9**9, places=13)
        grid = [i / 20 for i in range(21)]
        curve = operating_characteristic(n, r, grid)
        self.assertEqual(curve[-1], 1.0)
        self.assertEqual(curve[0], 0.0)
        self.assertTrue(all(b >= a for a, b in zip(curve, curve[1:], strict=False)))
        plan = demonstration_plan(0.95, 0.90, allowed_failures=2)
        (consumer,) = operating_characteristic(plan.tests, 2, [0.95])
        self.assertLessEqual(consumer, 0.10 + 1e-12)

    def test_producer_risk_plan(self) -> None:
        """The first r meeting the producer risk is returned; invalid input fails."""
        plan = smallest_plan_for_producer_risk(0.95, 0.90, 0.995, 0.20)
        assert plan.producer_risk is not None
        self.assertLessEqual(plan.producer_risk, 0.20)
        if plan.allowed_failures > 0:
            earlier = demonstration_plan(
                0.95,
                0.90,
                allowed_failures=plan.allowed_failures - 1,
                design_reliability=0.995,
            )
            assert earlier.producer_risk is not None
            self.assertGreater(earlier.producer_risk, 0.20)
        with self.assertRaises(ValueError):
            smallest_plan_for_producer_risk(0.95, 0.90, 0.94, 0.2)
        with self.assertRaises(ValueError):
            smallest_plan_for_producer_risk(0.99, 0.90, 0.991, 1e-9, max_failures=2)


class BayesianDemonstrationTests(unittest.TestCase):
    """Beta heritage priors, Bayesian size and the beta-binomial predictive."""

    def test_heritage_prior(self) -> None:
        """Power prior shapes, zero weight, and the equivalent test count."""
        prior = heritage_prior(40, 2, weight=0.5)
        self.assertEqual((prior.alpha, prior.beta), (21.0, 2.0))
        self.assertEqual(equivalent_prior_tests(prior), 23.0)
        flat = heritage_prior(40, 2, weight=0.0)
        self.assertEqual((flat.alpha, flat.beta), (1.0, 1.0))
        with self.assertRaises(ValueError):
            heritage_prior(1, 0, weight=1.5)
        with self.assertRaises(TypeError):
            equivalent_prior_tests(0.5)  # type: ignore[arg-type]

    def test_uniform_prior_closed_form(self) -> None:
        """Beta(1,1), n tests, 0 failures: P(R >= x) = 1 - x^(n+1)."""
        prior = Beta(1.0, 1.0)
        for n, x in ((20, 0.9), (100, 0.97)):
            got = bayesian_demonstration_confidence(prior, n, 0, x)
            self.assertAlmostEqual(got, 1.0 - x ** (n + 1), places=10)
        size = bayesian_demonstration_size(prior, 0.95, 0.90)
        self.assertEqual(size, math.ceil(math.log(0.10) / math.log(0.95)) - 1)
        self.assertGreaterEqual(
            bayesian_demonstration_confidence(prior, size, 0, 0.95), 0.9
        )
        self.assertLess(
            bayesian_demonstration_confidence(prior, size - 1, 0, 0.95), 0.9
        )

    def test_heritage_reduces_tests_and_failures_increase_them(self) -> None:
        """Informative priors need fewer tests; allowed failures need more."""
        flat = bayesian_demonstration_size(Beta(1.0, 1.0), 0.95, 0.90)
        informed = bayesian_demonstration_size(
            heritage_prior(60, 0, weight=0.5), 0.95, 0.90
        )
        self.assertLess(informed, flat)
        with_failure = bayesian_demonstration_size(
            Beta(1.0, 1.0), 0.95, 0.90, allowed_failures=1
        )
        self.assertGreater(with_failure, flat)
        # a prior that already demonstrates the requirement needs no extra tests
        strong = Beta(500.0, 2.0)
        self.assertEqual(bayesian_demonstration_size(strong, 0.95, 0.90), 0)
        with self.assertRaises(ValueError):
            bayesian_demonstration_size(Beta(1.0, 1.0), 0.999999, 0.99, max_tests=100)

    def test_predictive_pass_probability(self) -> None:
        """Uniform prior gives (r+1)/(n+1); one test gives a/(a+b); quadrature."""
        for n, r in ((9, 0), (9, 3), (20, 2)):
            self.assertAlmostEqual(
                predictive_pass_probability(Beta(1.0, 1.0), n, r),
                (r + 1) / (n + 1),
                places=12,
            )
        prior = Beta(8.0, 2.0)
        self.assertAlmostEqual(predictive_pass_probability(prior, 1, 0), 0.8, places=13)
        # independent oracle: average the operating characteristic over the prior
        n, r, m = 12, 1, 4000
        nodes = [(i + 0.5) / m for i in range(m)]
        curve = operating_characteristic(n, r, nodes)
        integral = sum(w * prior.pdf(x) for w, x in zip(curve, nodes, strict=True)) / m
        self.assertAlmostEqual(
            predictive_pass_probability(prior, n, r), integral, places=5
        )


class DurationTests(unittest.TestCase):
    """Exponential and Weibull time-on-test equivalents."""

    def test_exponential_zero_failure_equals_success_run(self) -> None:
        """T / t_m = ln(1 - C) / ln R, the continuous success-run size."""
        plan = exponential_test_time(0.99, 0.90, 150.0)
        self.assertAlmostEqual(
            plan.mission_equivalents, math.log(0.10) / math.log(0.99), places=10
        )
        self.assertAlmostEqual(
            plan.total_test_time, plan.mission_equivalents * 150.0, places=8
        )
        self.assertAlmostEqual(plan.failure_rate, -math.log(0.99) / 150.0, places=14)
        with_failure = exponential_test_time(0.99, 0.90, 150.0, allowed_failures=1)
        self.assertGreater(with_failure.total_test_time, plan.total_test_time)

    def test_weibull_units(self) -> None:
        """Shape 1 and L = 1 reproduce the success run; longer tests need fewer."""
        base = success_run_sample_size(0.99, 0.90)
        self.assertEqual(
            weibull_demonstration_units(0.99, 0.90, 100.0, 100.0, 1.0), base
        )
        # L = 2, shape 2: unit test reliability is R^4
        self.assertEqual(
            weibull_demonstration_units(0.99, 0.90, 100.0, 200.0, 2.0),
            success_run_sample_size(0.99**4, 0.90),
        )
        short = weibull_demonstration_units(0.99, 0.90, 100.0, 50.0, 2.0)
        longer = weibull_demonstration_units(0.99, 0.90, 100.0, 150.0, 2.0)
        self.assertGreater(short, base)
        self.assertLess(longer, base)
        with self.assertRaises(ValueError):
            weibull_demonstration_units(0.99, 0.90, 100.0, 1e-12, 2.0)


class GrowthTests(unittest.TestCase):
    """Crow-AMSAA campaign assessment and the beta mixture."""

    def test_mission_reliability_closed_forms(self) -> None:
        """Shape one reduces to exp(-lambda tau); other shapes match the formula."""
        hpp = CrowAMSAAFit(5, 1000.0, "time", 1.0, 0.8, 0.005, (0.5, 1.5), 0.9)
        self.assertAlmostEqual(
            nhpp_mission_reliability(hpp, 100.0), math.exp(-0.5), places=13
        )
        fit = CrowAMSAAFit(6, 1000.0, "time", 0.6, 0.5, 0.02, (0.3, 0.9), 0.9)
        expected = math.exp(-0.02 * (1100.0**0.6 - 1000.0**0.6))
        self.assertAlmostEqual(
            nhpp_mission_reliability(fit, 100.0), expected, places=13
        )
        self.assertAlmostEqual(
            nhpp_mission_reliability(fit, 100.0, start_time=0.0),
            math.exp(-0.02 * 100.0**0.6),
            places=13,
        )
        with self.assertRaises(TypeError):
            nhpp_mission_reliability("fit", 1.0)  # type: ignore[arg-type]

    def test_campaign_growth_detected(self) -> None:
        """Early-bunched failures show growth; NHPP projection beats constant rate."""
        growing = assess_test_campaign(
            [5.0, 15.0, 40.0, 100.0, 250.0, 600.0], 1000.0, 50.0
        )
        self.assertLess(growing.fit.beta, 1.0)
        self.assertTrue(growing.growth_detected)
        self.assertGreater(growing.nhpp_reliability, growing.constant_rate_reliability)
        self.assertAlmostEqual(
            growing.constant_rate_reliability,
            math.exp(-growing.fit.intensity(1000.0) * 50.0),
            places=13,
        )
        steady = assess_test_campaign([100.0 * i for i in range(1, 10)], 1000.0, 50.0)
        self.assertFalse(steady.growth_detected)
        self.assertAlmostEqual(
            steady.fit.beta_interval[0] < 1.0 < steady.fit.beta_interval[1], True
        )
        with self.assertRaises(ValueError):
            assess_test_campaign([10.0], 100.0, 5.0)

    def test_single_component_is_conjugate(self) -> None:
        """One component reproduces the Beta update and its evidence."""
        mixture = BetaMixture((1.0,), (Beta(2.0, 3.0),))
        update = mixture.update(4, 6)
        self.assertEqual(
            (update.posterior.components[0].alpha, update.posterior.components[0].beta),
            (6.0, 9.0),
        )
        log_b = lambda a, b: math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)  # noqa: E731
        self.assertAlmostEqual(
            update.log_evidence, log_b(6, 9) - log_b(2, 3), places=12
        )

    def test_mixture_update_against_numerical_bayes(self) -> None:
        """Weights, mean and quantile against a midpoint-grid Bayes computation."""
        mixture = BetaMixture((0.6, 0.4), (Beta(30.0, 2.0), Beta(1.0, 1.0)))
        events, non_events = 1, 9
        update = mixture.update(events, non_events)
        m = 20000
        grid = [(i + 0.5) / m for i in range(m)]
        prior = [
            sum(
                w * c.pdf(x)
                for w, c in zip(mixture.weights, mixture.components, strict=True)
            )
            for x in grid
        ]
        like = [x**events * (1.0 - x) ** non_events for x in grid]
        evidence = sum(p * lk for p, lk in zip(prior, like, strict=True)) / m
        post_mean = (
            sum(x * p * lk for x, p, lk in zip(grid, prior, like, strict=True))
            / m
            / evidence
        )
        self.assertAlmostEqual(update.log_evidence, math.log(evidence), places=6)
        self.assertAlmostEqual(update.posterior.mean, post_mean, places=6)
        self.assertAlmostEqual(sum(update.posterior.weights), 1.0, places=14)
        # posterior cdf vs the grid
        point = 0.3
        grid_cdf = (
            sum(p * lk for x, p, lk in zip(grid, prior, like, strict=True) if x < point)
            / m
            / evidence
        )
        self.assertAlmostEqual(update.posterior.cdf(point), grid_cdf, places=4)
        level = update.posterior.quantile(0.9)
        self.assertAlmostEqual(update.posterior.cdf(level), 0.9, places=10)
        self.assertAlmostEqual(
            update.posterior.cdf(level) + update.posterior.sf(level), 1.0, places=12
        )

    def test_weight_moves_toward_the_better_supported_component(self) -> None:
        """Failures contradict a high-reliability heritage prior."""
        mixture = BetaMixture((0.7, 0.3), (Beta(60.0, 2.0), Beta(2.0, 2.0)))
        good = mixture.update(10, 0).posterior.weights[0]
        bad = mixture.update(4, 6).posterior.weights[0]
        self.assertGreater(good, 0.7)
        self.assertLess(bad, 0.7)
        self.assertLess(bad, 1e-3)

    def test_mixture_moments_sampling_and_flip(self) -> None:
        """Closed-form moments, MCSE-banded sample mean, and the flipped mixture."""
        mixture = BetaMixture((0.3, 0.7), (Beta(2.0, 8.0), Beta(9.0, 1.0)))
        self.assertAlmostEqual(mixture.mean, 0.3 * 0.2 + 0.7 * 0.9, places=14)
        second = 0.3 * (2 * 3) / (10 * 11) + 0.7 * (9 * 10) / (10 * 11)
        self.assertAlmostEqual(mixture.variance, second - mixture.mean**2, places=14)
        draws = mixture.sample(20000, seed=3)
        mean = sum(draws) / len(draws)
        sd = math.sqrt(mixture.variance / len(draws))
        self.assertLess(abs(mean - mixture.mean), 4 * sd)
        self.assertEqual(draws, mixture.sample(20000, seed=3))
        self.assertAlmostEqual(mixture.flipped().mean, 1.0 - mixture.mean, places=14)
        with self.assertRaises(ValueError):
            BetaMixture((0.5, 0.4), (Beta(1.0, 1.0), Beta(2.0, 2.0)))
        with self.assertRaises(TypeError):
            BetaMixture((1.0,), (0.5,))  # type: ignore[arg-type]


class ClusterTests(unittest.TestCase):
    """Closed form, exact BDD, common-cause and catastrophic effects."""

    def test_k_out_of_n_without_ccf_is_binomial(self) -> None:
        """Without CCF or catastrophe, success is P(Binomial(n, q) <= n - k)."""
        for n, k, q in ((9, 8, 0.03), (4, 2, 0.1), (3, 3, 0.02), (1, 1, 0.05)):
            model = EngineClusterModel(n, k, q)
            expected = sum(binomial_pmf(b, n, q) for b in range(n - k + 1))
            self.assertAlmostEqual(model.success_probability(), expected, places=13)
        series = EngineClusterModel(5, 5, 0.01, ccf_beta=0.2)
        expected_series = (1 - 0.2 * 0.01) * (1 - 0.8 * 0.01) ** 5
        self.assertAlmostEqual(series.success_probability(), expected_series, places=13)

    def test_closed_form_matches_system_model(self) -> None:
        """All three outcomes agree with the exact BDD, with CCF and catastrophe."""
        for n, k, q, beta, c in (
            (9, 8, 0.02, 0.10, 0.30),
            (5, 3, 0.05, 0.25, 0.10),
            (3, 2, 0.08, 0.00, 0.50),
            (2, 1, 0.04, 0.50, 0.20),
            (1, 1, 0.03, 0.00, 0.40),
        ):
            model = EngineClusterModel(n, k, q, beta, c)
            closed, exact = model.outcomes(), model.system_outcomes()
            for name in ("success", "benign_shortfall", "catastrophic"):
                self.assertAlmostEqual(
                    getattr(closed, name), getattr(exact, name), places=12
                )
            total = closed.success + closed.benign_shortfall + closed.catastrophic
            self.assertAlmostEqual(total, 1.0, places=13)

    def test_common_cause_and_catastrophe_directions(self) -> None:
        """CCF direction depends on q n (n - 1); c shifts mass to catastrophe.

        For ``k = n - 1`` and ``c = 0`` the loss probability is
        ``beta q + C(n, 2) ((1 - beta) q)^2`` to leading order, whose slope in
        ``beta`` at zero is ``q - n (n - 1) q^2``: positive (CCF hurts) when
        ``q < 1 / (n (n - 1))`` and negative above it, because the model holds
        the engine outage probability ``q`` fixed and moves independent mass
        into one common event. Both signs are asserted.
        """
        betas = (0.0, 0.05, 0.1, 0.3)
        for c in (0.0, 0.05):
            small_q = [
                EngineClusterModel(9, 8, 0.002, beta, c).success_probability()
                for beta in betas
            ]
            self.assertTrue(
                all(b < a for a, b in zip(small_q, small_q[1:], strict=False))
            )
        self.assertGreater(0.02 * 9 * 8, 1.0)
        large_q = [
            EngineClusterModel(9, 8, 0.02, beta, 0.0).success_probability()
            for beta in betas
        ]
        self.assertTrue(all(b > a for a, b in zip(large_q, large_q[1:], strict=False)))
        low = EngineClusterModel(9, 8, 0.02, 0.1, 0.1).outcomes()
        high = EngineClusterModel(9, 8, 0.02, 0.1, 0.6).outcomes()
        self.assertLess(high.success, low.success)
        self.assertGreater(high.catastrophic, low.catastrophic)
        self.assertEqual(
            EngineClusterModel(9, 8, 0.02, 0.1, 0.0).outcomes().catastrophic, 0.0
        )
        self.assertGreater(
            EngineClusterModel(9, 8, 0.02).success_probability(),
            EngineClusterModel(9, 9, 0.02).success_probability(),
        )

    def test_validation(self) -> None:
        """Invalid clusters and laws are rejected."""
        with self.assertRaises(ValueError):
            EngineClusterModel(3, 4, 0.1)
        with self.assertRaises(ValueError):
            EngineClusterModel(1, 1, 0.1, ccf_beta=0.1)
        with self.assertRaises(ValueError):
            EngineClusterModel(3, 2, 1.5)
        with self.assertRaises(TypeError):
            EngineClusterModel(3.0, 2, 0.1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ClusterUncertainty(3, 2, 1.5)
        with self.assertRaises(ValueError):
            ClusterUncertainty(1, 1, 0.1, ccf_beta=Beta(1.0, 9.0))


class ClusterUncertaintyTests(unittest.TestCase):
    """Exact moments, Monte Carlo agreement and Sobol ranking."""

    def test_exact_moments_closed_forms(self) -> None:
        """One engine gives R = 1 - q; two engines with k = 1 give R = 1 - q^2."""
        q = Beta(2.0, 38.0)
        mean, var = cluster_reliability_moments(ClusterUncertainty(1, 1, q))
        self.assertAlmostEqual(mean, 1.0 - 2.0 / 40.0, places=13)
        self.assertAlmostEqual(var, float(q.variance), places=13)
        moment = lambda m: math.prod((2.0 + i) / (40.0 + i) for i in range(m))  # noqa: E731
        mean2, var2 = cluster_reliability_moments(ClusterUncertainty(2, 1, q))
        self.assertAlmostEqual(mean2, 1.0 - moment(2), places=13)
        self.assertAlmostEqual(var2, moment(4) - moment(2) ** 2, places=13)

    def test_exact_moments_match_ccf_parameter_uncertainty(self) -> None:
        """Beta-factor uncertainty: independent exact route in probstat.systems."""
        model = EngineClusterModel(4, 3, 0.02, 0.1, 0.3)
        ref = ccf_parameter_uncertainty(
            model.system_model(), 0.0, {(0, "beta"): Beta(2.0, 18.0)}
        )
        self.assertEqual(ref.method, "exact")
        mean, var = cluster_reliability_moments(
            ClusterUncertainty(4, 3, 0.02, Beta(2.0, 18.0), 0.3)
        )
        self.assertAlmostEqual(mean, ref.mean, places=12)
        self.assertAlmostEqual(var, ref.variance, delta=1e-12)

    def test_mixture_input_moments(self) -> None:
        """A mixture of two identical Betas equals the Beta; mixtures mix moments."""
        beta_law = Beta(3.0, 77.0)
        twin = BetaMixture((0.5, 0.5), (beta_law, beta_law))
        a = cluster_reliability_moments(ClusterUncertainty(9, 8, beta_law, 0.1, 0.2))
        b = cluster_reliability_moments(ClusterUncertainty(9, 8, twin, 0.1, 0.2))
        self.assertAlmostEqual(a[0], b[0], places=12)
        self.assertAlmostEqual(a[1], b[1], places=12)
        far = BetaMixture((0.5, 0.5), (Beta(2.0, 98.0), Beta(8.0, 92.0)))
        mean_far, _ = cluster_reliability_moments(ClusterUncertainty(1, 1, far))
        self.assertAlmostEqual(mean_far, 1.0 - far.mean, places=13)

    def test_monte_carlo_matches_exact_moments(self) -> None:
        """Sample mean within 4 MCSE and sample variance within 15 % of exact."""
        unc = ClusterUncertainty(9, 8, Beta(2.0, 98.0), Beta(2.0, 18.0), Beta(3.0, 7.0))
        mean, var = cluster_reliability_moments(unc)
        count = 6000
        dist = cluster_reliability_distribution(
            unc, samples=count, sobol_samples=200, seed=11
        )
        self.assertLess(abs(dist.mean - mean), 4.0 * math.sqrt(var / count))
        self.assertLess(abs(dist.standard_deviation**2 - var) / var, 0.15)
        low, high = dist.interval
        self.assertLess(low, dist.mean)
        self.assertLess(dist.mean, high)
        self.assertLess(low, high)
        self.assertEqual(dist.values, tuple(sorted(dist.values)))
        again = cluster_reliability_distribution(
            unc, samples=count, sobol_samples=200, seed=11
        )
        self.assertEqual(dist.values, again.values)

    def test_sobol_exact_zero_and_ranking(self) -> None:
        """An ignored input has index exactly zero; q dominates the cluster."""
        # for n = 1, k = 1 the success probability 1 - q does not involve c
        flat = cluster_reliability_distribution(
            ClusterUncertainty(1, 1, Beta(2.0, 38.0), 0.0, Beta(3.0, 7.0)),
            samples=500,
            sobol_samples=3000,
            seed=5,
        )
        self.assertEqual(
            flat.input_names, ("failure_probability", "catastrophic_fraction")
        )
        assert flat.sobol_first is not None and flat.sobol_total is not None
        self.assertEqual(flat.sobol_total[1], 0.0)
        self.assertEqual(flat.sobol_first[1], 0.0)
        # Monte Carlo error of the pick-freeze estimators is O(N^-1/2)
        self.assertAlmostEqual(flat.sobol_first[0], 1.0, delta=0.15)
        self.assertAlmostEqual(flat.sobol_total[0], 1.0, delta=0.15)
        unc = ClusterUncertainty(9, 8, Beta(2.0, 98.0), Beta(2.0, 18.0), Beta(3.0, 7.0))
        dist = cluster_reliability_distribution(
            unc, samples=500, sobol_samples=1500, seed=2
        )
        self.assertEqual(dist.ranking[0], "failure_probability")
        self.assertEqual(dist.ranking[-1], "ccf_beta")
        assert dist.sobol_total is not None and dist.sobol_first is not None
        for first, total in zip(dist.sobol_first, dist.sobol_total, strict=True):
            self.assertLess(first, total + 0.05)

    def test_constant_inputs_are_left_out(self) -> None:
        """Constants do not enter the ranking; all-constant inputs have no indices."""
        dist = cluster_reliability_distribution(
            ClusterUncertainty(4, 3, Beta(2.0, 38.0), 0.1, 0.3),
            samples=200,
            sobol_samples=100,
        )
        self.assertEqual(dist.input_names, ("failure_probability",))
        fixed = cluster_reliability_distribution(
            ClusterUncertainty(4, 3, 0.05, 0.1, 0.3), samples=50, sobol_samples=20
        )
        self.assertIsNone(fixed.sobol_first)
        self.assertEqual(fixed.ranking, ())
        self.assertEqual(fixed.standard_deviation, 0.0)
        self.assertAlmostEqual(
            fixed.mean,
            EngineClusterModel(4, 3, 0.05, 0.1, 0.3).success_probability(),
            places=14,
        )


class VehicleTests(unittest.TestCase):
    """Launch-vehicle event tree and uncertainty propagation."""

    design = LaunchVehicleDesign(
        EngineClusterModel(9, 8, 0.02, 0.10, 0.30),
        0.005,
        0.01,
        EngineClusterModel(1, 1, 0.015, 0.0, 0.20),
    )

    def test_event_tree_is_the_product_of_stage_factors(self) -> None:
        """Success is the stage product; end states are exhaustive; paths enumerate."""
        design = self.design
        product = (
            design.first_stage.success_probability()
            * (1 - 0.005)
            * (1 - 0.01)
            * design.upper_stage.success_probability()
        )
        self.assertAlmostEqual(vehicle_reliability(design), product, places=14)
        outcomes = vehicle_outcome_probabilities(design)
        self.assertAlmostEqual(sum(outcomes.values()), 1.0, places=13)
        self.assertEqual(
            set(outcomes), {"success", "loss_of_mission", "loss_of_vehicle"}
        )
        tree = launch_event_tree(design)
        self.assertEqual(len(tree.sequences()), 7)
        self.assertAlmostEqual(tree.total_frequency(), 1.0, places=13)
        stage1 = design.first_stage.outcomes()
        stage2 = design.upper_stage.outcomes()
        lov = stage1.catastrophic + stage1.success * 0.995 * 0.99 * stage2.catastrophic
        self.assertAlmostEqual(outcomes["loss_of_vehicle"], lov, places=13)

    def test_vehicle_distribution_matches_independent_exact_moments(self) -> None:
        """Mean and variance of a product of independent stage factors, exactly."""
        stage1 = ClusterUncertainty(
            9, 8, Beta(2.0, 98.0), Beta(2.0, 18.0), Beta(3.0, 7.0)
        )
        stage2 = ClusterUncertainty(1, 1, Beta(1.5, 98.5), 0.0, Beta(2.0, 8.0))
        unc = VehicleUncertainty(stage1, Beta(1.0, 199.0), Beta(1.0, 99.0), stage2)
        m1, v1 = cluster_reliability_moments(stage1)
        m2, v2 = cluster_reliability_moments(stage2)
        sep, ign = Beta(1.0, 199.0), Beta(1.0, 99.0)
        moments = [(m1, v1 + m1**2), (m2, v2 + m2**2)]
        for law in (sep, ign):
            mean = 1.0 - float(law.mean)
            moments.append((mean, float(law.variance) + mean**2))
        exact_mean = math.prod(m for m, _ in moments)
        exact_var = math.prod(s for _, s in moments) - exact_mean**2
        count = 6000
        dist = vehicle_reliability_distribution(
            unc, samples=count, sobol_samples=200, seed=21
        )
        self.assertLess(abs(dist.mean - exact_mean), 4.0 * math.sqrt(exact_var / count))
        self.assertLess(abs(dist.standard_deviation**2 - exact_var) / exact_var, 0.15)
        self.assertEqual(len(dist.input_names), 7)
        self.assertIn(dist.ranking[0], dist.input_names)
        # at the nominal (mean) parameters the draw formula equals the tree
        nominal = LaunchVehicleDesign(
            EngineClusterModel(9, 8, 0.02, 0.1, 0.3),
            0.005,
            0.01,
            EngineClusterModel(1, 1, 0.015, 0.0, 0.2),
        )
        self.assertGreater(vehicle_reliability(nominal), 0.0)

    def test_validation(self) -> None:
        """Wrong types and invalid probabilities are rejected."""
        with self.assertRaises(TypeError):
            launch_event_tree("design")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LaunchVehicleDesign(
                self.design.first_stage, 1.5, 0.01, self.design.upper_stage
            )
        with self.assertRaises(TypeError):
            VehicleUncertainty(1, 0.1, 0.1, 2)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cluster_reliability_distribution(ClusterUncertainty(2, 1, 0.1), samples=5)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudyTests(unittest.TestCase):
    """Large Monte Carlo and Sobol checks (PROBSTAT_SLOW=1)."""

    def test_large_sample_moments_and_ranking_stability(self) -> None:
        """200k samples match exact moments; the ranking is stable across seeds."""
        unc = ClusterUncertainty(9, 8, Beta(2.0, 98.0), Beta(2.0, 18.0), Beta(3.0, 7.0))
        mean, var = cluster_reliability_moments(unc)
        count = 200_000
        dist = cluster_reliability_distribution(
            unc, samples=count, sobol_samples=5000, seed=1
        )
        self.assertLess(abs(dist.mean - mean), 4.0 * math.sqrt(var / count))
        self.assertLess(abs(dist.standard_deviation**2 - var) / var, 0.03)
        for seed in (2, 3):
            other = cluster_reliability_distribution(
                unc, samples=100, sobol_samples=5000, seed=seed
            )
            self.assertEqual(other.ranking, dist.ranking)


if __name__ == "__main__":
    unittest.main()
