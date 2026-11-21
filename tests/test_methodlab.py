"""Tests for the method laboratory: combinators, certification, studies, search.

Every construction principle is checked against an independent oracle:
exact enumeration over the level law for the randomized-truncation
estimators, closed-form variances, the classical Richardson pair, a
brute-force minimization for the best linear unbiased weights, the
identity "jackknife of the plug-in variance is the unbiased variance", and
analytical influence functions. The certification checks are tested in both
directions: they must refute a false claim and must not refute a true one.
"""

from __future__ import annotations

import math
import random
import statistics
import unittest

from probstat.distributions import StudentT
from probstat.foundations import RandomStream
from probstat.methodlab import (
    MEASURES,
    DataGeneratingMechanism,
    Draw,
    Estimator,
    Method,
    MethodOutput,
    StudyProtocol,
    antithetic_estimator,
    best_linear_unbiased_weights,
    bootstrap_bias_corrected,
    check_convergence_rate,
    check_coverage,
    check_efficiency_gain,
    check_invariance,
    check_unbiasedness,
    check_variance,
    check_variance_stability,
    combine_unbiased,
    compare_sequentially,
    control_variate_estimator,
    coupled_sum_estimator,
    empirical_influence,
    estimator_from_function,
    extrapolated_estimator,
    extrapolation_weights,
    finite_levels,
    geometric_levels,
    interval_verdict,
    jackknife_bias_corrected,
    method_card,
    one_step_estimator,
    optimal_level_probabilities,
    performance_measures,
    pilot_covariance,
    rank_by_efficiency,
    repetitions_for_standard_error,
    run_simulation_study,
    sample_mean,
    sequential_halving,
    single_term_estimator,
    sum_of_estimators,
    summarize,
    thinned_estimator,
    truncation_design,
)

SIGMA = 0.5


def noisy_difference(level: int, stream: RandomStream) -> Draw:
    """Return ``2^-(n+1) (1 + SIGMA xi)``; the differences sum to one."""
    noise = stream.generator().gauss(0.0, 1.0)
    return Draw(2.0 ** -(level + 1) * (1.0 + SIGMA * noise), 2.0**level)


def exact_difference(level: int, stream: RandomStream) -> Draw:
    """Return the deterministic difference ``2^-(n+1)`` with unit cost."""
    return Draw(2.0 ** -(level + 1), 1.0)


def uniform_estimator() -> Estimator:
    """Return one uniform variate as an estimator of one half."""
    return estimator_from_function("uniform", lambda g: g.random())


class EstimatorContractTests(unittest.TestCase):
    """The draw contract, replication, and summaries."""

    def test_draws_are_reproducible_and_validated(self) -> None:
        """Equal streams give equal draws; a bad sampler is rejected."""
        estimator = uniform_estimator()
        self.assertEqual(estimator.draw(3), estimator.draw(RandomStream(3)))
        self.assertNotEqual(estimator.draw(3), estimator.draw(4))
        bad = Estimator("bad", lambda stream: 1.0)  # type: ignore[arg-type,return-value]
        with self.assertRaises(TypeError):
            bad.draw(1)
        with self.assertRaises(ValueError):
            Estimator(" ", lambda stream: Draw(0.0, 1.0))
        with self.assertRaises(ValueError):
            Draw(math.inf, 1.0)
        with self.assertRaises(ValueError):
            Draw(0.0, -1.0)

    def test_summary_matches_the_statistics_module(self) -> None:
        """Mean and variance agree with independent formulas."""
        estimator = uniform_estimator()
        draws = estimator.replicates(200, 11)
        values = [d.value for d in draws]
        summary = summarize(estimator, 200, 11)
        self.assertAlmostEqual(summary.mean, statistics.fmean(values), places=14)
        self.assertAlmostEqual(summary.variance, statistics.variance(values), places=14)
        self.assertAlmostEqual(
            summary.mean_standard_error,
            math.sqrt(statistics.variance(values) / 200),
            places=14,
        )
        self.assertEqual(summary.mean_cost, 1.0)
        with self.assertRaises(ValueError):
            estimator.replicates(0, 1)

    def test_variance_standard_error_matches_the_uniform_closed_form(self) -> None:
        """For a uniform variate ``sd(s^2) ~ sqrt((mu4 - sigma^4)/R)``."""
        summary = summarize(uniform_estimator(), 20000, 5)
        mu4, sigma2 = 1.0 / 80.0, 1.0 / 12.0
        exact = math.sqrt((mu4 - sigma2**2) / 20000)
        self.assertAlmostEqual(summary.variance_standard_error / exact, 1.0, delta=0.05)

    def test_sample_mean_keeps_the_work_normalized_variance(self) -> None:
        """Averaging divides the variance and multiplies the cost."""
        base = uniform_estimator()
        mean = sample_mean(base, 10)
        summary = summarize(mean, 4000, 2)
        self.assertEqual(summary.mean_cost, 10.0)
        self.assertAlmostEqual(summary.work_normalized_variance, 1 / 12, delta=0.008)
        self.assertEqual(mean.lineage, ("sample mean of 10 independent draws",))
        with self.assertRaises(ValueError):
            sample_mean(base, 0)


class RandomizedTruncationTests(unittest.TestCase):
    """Single-term and coupled-sum estimators against exact enumeration."""

    def test_finite_law_expectation_by_enumeration(self) -> None:
        """Both estimators have expectation ``Y_L`` under any finite law."""
        law = finite_levels((0.4, 0.3, 0.2, 0.1))
        target = 1.0 - 2.0**-4
        single = sum(
            law.probability(n) * (2.0 ** -(n + 1) / law.probability(n))
            for n in range(4)
        )
        coupled = sum(
            law.probability(top)
            * sum(2.0 ** -(n + 1) / law.tail(n) for n in range(top + 1))
            for top in range(4)
        )
        self.assertAlmostEqual(single, target, places=14)
        self.assertAlmostEqual(coupled, target, places=14)
        for build in (single_term_estimator, coupled_sum_estimator):
            estimator = build("z", exact_difference, law)
            summary = summarize(estimator, 20000, 9)
            self.assertAlmostEqual(
                summary.mean, target, delta=5 * summary.mean_standard_error
            )

    def test_level_laws(self) -> None:
        """Probabilities, tails, and sampling frequencies agree."""
        law = geometric_levels(0.5)
        self.assertTrue(law.is_unbounded)
        self.assertIsNone(law.maximum_level)
        self.assertAlmostEqual(law.probability(3), 0.5**4)
        self.assertAlmostEqual(law.tail(3), 0.5**3)
        root = RandomStream(1)
        counts = [0] * 4
        for index in range(20000):
            level = law.sample(root.child(index))
            if level < 4:
                counts[level] += 1
        for level, count in enumerate(counts):
            self.assertAlmostEqual(count / 20000, law.probability(level), delta=0.012)
        finite = finite_levels((0.25, 0.75))
        self.assertEqual(finite.maximum_level, 1)
        self.assertEqual(finite.probability(5), 0.0)
        self.assertAlmostEqual(finite.tail(1), 0.75)
        for bad in ((), (0.5, 0.6), (1.0, 0.0)):
            with self.assertRaises(ValueError):
                finite_levels(bad)
        for ratio in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                geometric_levels(ratio)

    def test_single_term_is_unbiased_with_the_closed_form_variance(self) -> None:
        """Geometric law ``r = 1/2``: mean 1 and variance ``(1+s^2)/2 - 1``."""
        estimator = single_term_estimator("st", noisy_difference, geometric_levels(0.5))
        bias = check_unbiasedness(
            estimator, 1.0, replicates=40000, seed=3, tolerance=0.02
        )
        self.assertEqual(bias.verdict, "supported")
        # sum_n 4^-(n+1) (1 + s^2) / (0.5 * 0.5^n) - 1 = (1 + s^2) - 1
        variance = check_variance(
            estimator,
            SIGMA**2,
            replicates=40000,
            seed=4,
            relative_tolerance=0.15,
        )
        self.assertEqual(variance.verdict, "supported")
        self.assertIn("single-term", estimator.lineage[0])

    def test_coupled_sum_is_unbiased(self) -> None:
        """The coupled sum has expectation one and a random cost."""
        estimator = coupled_sum_estimator("cs", noisy_difference, geometric_levels(0.4))
        result = check_unbiasedness(estimator, 1.0, replicates=30000, seed=6)
        self.assertEqual(result.verdict, "not refuted")
        unit = coupled_sum_estimator("unit", exact_difference, geometric_levels(0.4))
        summary = summarize(unit, 5000, 6)
        # Unit-cost differences: E[cost] = sum_n P(N >= n) = 1 / (1 - 0.4).
        self.assertAlmostEqual(summary.mean_cost, 1.0 / 0.6, delta=0.08)

    def test_too_light_a_level_law_shows_unstable_variance(self) -> None:
        """With ``r < 1/4`` the variance is infinite and one draw dominates."""
        heavy = single_term_estimator("bad", noisy_difference, geometric_levels(0.1))
        sound = single_term_estimator("ok", noisy_difference, geometric_levels(0.5))
        unstable = check_variance_stability(heavy, replicates=20000, seed=8)
        stable = check_variance_stability(sound, replicates=20000, seed=8)
        self.assertGreater(unstable.largest_share, 10 * stable.largest_share)
        self.assertFalse(stable.suspicious)
        self.assertEqual(len(stable.block_variances), 8)
        with self.assertRaises(ValueError):
            check_variance_stability(sound, replicates=10, seed=1)

    def test_thinning_keeps_the_mean_and_has_the_closed_form_variance(self) -> None:
        """``B Z'/q``: mean kept, variance ``E[Z'^2]/q - mu^2``, cost ``q c``."""
        base = estimator_from_function("g", lambda g: g.gauss(0.5, 1.0), cost=4.0)
        thin = thinned_estimator("thin", base, 0.25)
        bias = check_unbiasedness(thin, 0.5, replicates=40000, seed=5)
        self.assertEqual(bias.verdict, "not refuted")
        claim = check_variance(
            thin, 1.25 / 0.25 - 0.25, replicates=40000, seed=6, relative_tolerance=0.1
        )
        self.assertEqual(claim.verdict, "supported")
        self.assertAlmostEqual(summarize(thin, 20000, 7).mean_cost, 1.0, delta=0.05)
        self.assertEqual(thinned_estimator("all", base, 1.0).draw(3).cost, 4.0)
        for bad in (0.0, 1.5):
            with self.assertRaises(ValueError):
                thinned_estimator("bad", base, bad)

    def test_sum_adds_means_variances_costs_and_lineages(self) -> None:
        """Independent terms: variances and costs add; lineage is merged."""
        first = estimator_from_function("a", lambda g: g.gauss(1.0, 1.0), cost=2.0)
        second = thinned_estimator(
            "b", estimator_from_function("c", lambda g: g.gauss(2.0, 1.0)), 0.5
        )
        total = sum_of_estimators("a+b", (first, second))
        bias = check_unbiasedness(total, 3.0, replicates=40000, seed=8)
        self.assertEqual(bias.verdict, "not refuted")
        claim = check_variance(
            total,
            1.0 + (5.0 / 0.5 - 4.0),
            replicates=40000,
            seed=9,
            relative_tolerance=0.1,
        )
        self.assertEqual(claim.verdict, "supported")
        self.assertAlmostEqual(summarize(total, 20000, 1).mean_cost, 2.5, delta=0.02)
        self.assertEqual(len(total.lineage), 2)
        self.assertIn("[b] evaluated with probability 0.5", total.lineage[0])
        with self.assertRaises(ValueError):
            sum_of_estimators("empty", ())

    def test_optimal_level_law_attains_the_cauchy_schwarz_bound(self) -> None:
        """``p ~ sqrt(m/c)`` attains the bound; other laws exceed it."""
        moments = [4.0**-n for n in range(6)]
        costs = [2.0**n for n in range(6)]
        optimal = truncation_design(moments, costs)
        self.assertAlmostEqual(optimal.product, optimal.lower_bound, places=12)
        self.assertAlmostEqual(optimal.efficiency, 1.0, places=12)
        self.assertAlmostEqual(sum(optimal_level_probabilities(moments, costs)), 1.0)
        generator = random.Random(0)
        for _ in range(50):
            raw = [generator.random() + 0.01 for _ in range(6)]
            law = [r / sum(raw) for r in raw]
            other = truncation_design(moments, costs, law)
            self.assertGreaterEqual(other.product, optimal.lower_bound - 1e-12)
            self.assertLessEqual(other.efficiency, 1.0 + 1e-12)
        with self.assertRaises(ValueError):
            optimal_level_probabilities([1.0, 0.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            truncation_design(moments, costs, [0.5, 0.5])


class ExtrapolationTests(unittest.TestCase):
    """Richardson weights and the extrapolated estimator."""

    def test_weights_reduce_to_the_classical_pair(self) -> None:
        """Steps ``h, h/2`` and order ``q`` give ``(-1, 2^q)/(2^q - 1)``."""
        for order in (1.0, 2.0, 0.5):
            weights = extrapolation_weights((0.2, 0.1), (order,))
            scale = 2.0**order - 1.0
            self.assertAlmostEqual(weights[0], -1.0 / scale, places=12)
            self.assertAlmostEqual(weights[1], 2.0**order / scale, places=12)

    def test_two_orders_are_cancelled_exactly(self) -> None:
        """``mu + a h + b h^2`` is extrapolated to ``mu``; ``h^3`` remains."""

        def family(step: float) -> Estimator:
            """Return a deterministic biased 'estimator' at step ``h``."""
            value = 3.0 + 2.0 * step - 5.0 * step**2
            return Estimator(f"h={step}", lambda stream: Draw(value, 1.0 / step))

        estimator = extrapolated_estimator("rich", family, (0.4, 0.2, 0.1), (1.0, 2.0))
        draw = estimator.draw(1)
        self.assertAlmostEqual(draw.value, 3.0, places=12)
        self.assertAlmostEqual(draw.cost, 2.5 + 5.0 + 10.0)
        cubic = extrapolated_estimator(
            "cubic",
            lambda h: Estimator("c", lambda stream: Draw(3.0 + h**3, 1.0)),
            (0.4, 0.2, 0.1),
            (1.0, 2.0),
            common_stream=False,
        )
        self.assertGreater(abs(cubic.draw(1).value - 3.0), 1e-4)

    def test_validation(self) -> None:
        """Counts, positivity, and distinctness are enforced."""
        for steps, orders in (
            ((0.1, 0.2), (1.0, 2.0)),
            ((0.1, 0.1), (1.0,)),
            ((0.1, -0.2), (1.0,)),
            ((0.1, 0.2, 0.3), (1.0, 1.0)),
        ):
            with self.assertRaises(ValueError):
                extrapolation_weights(steps, orders)


class LinearCombinationTests(unittest.TestCase):
    """Best linear unbiased weights and control variates."""

    def test_independent_estimators_get_inverse_variance_weights(self) -> None:
        """Two independent estimators: ``w_1 = v_2/(v_1+v_2)``."""
        result = best_linear_unbiased_weights(((1.0, 0.0), (0.0, 3.0)))
        self.assertAlmostEqual(result.weights[0], 0.75, places=12)
        self.assertAlmostEqual(result.variance, 0.75, places=12)

    def test_weights_minimize_the_variance_by_brute_force(self) -> None:
        """A grid search over ``w`` finds no smaller variance."""
        covariance = ((2.0, 0.9, 0.2), (0.9, 1.0, 0.3), (0.2, 0.3, 1.5))
        result = best_linear_unbiased_weights(covariance)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=12)

        def variance(weights: tuple[float, float, float]) -> float:
            """Return ``w' Sigma w``."""
            return sum(
                weights[i] * covariance[i][j] * weights[j]
                for i in range(3)
                for j in range(3)
            )

        self.assertAlmostEqual(variance(result.weights), result.variance, places=12)
        best = min(
            variance((a / 50, b / 50, 1.0 - a / 50 - b / 50))
            for a in range(-25, 76)
            for b in range(-25, 76)
        )
        self.assertLessEqual(result.variance, best + 1e-12)
        self.assertLessEqual(result.variance, min(covariance[i][i] for i in range(3)))
        with self.assertRaises(ValueError):
            best_linear_unbiased_weights(((1.0, 2.0), (2.0, 1.0)))
        with self.assertRaises(ValueError):
            best_linear_unbiased_weights(())

    def test_combination_is_unbiased_and_beats_its_members(self) -> None:
        """Pilot weights: unbiased, variance near the predicted one."""
        first = estimator_from_function("a", lambda g: 1.0 + g.gauss(0.0, 1.0))
        second = estimator_from_function("b", lambda g: 1.0 + g.gauss(0.0, 2.0))
        combined, weights = combine_unbiased(
            "blue", (first, second), pilot_size=4000, seed=1, shared_stream=False
        )
        self.assertAlmostEqual(weights.weights[0], 0.8, delta=0.03)
        bias = check_unbiasedness(combined, 1.0, replicates=20000, seed=12)
        self.assertEqual(bias.verdict, "not refuted")
        summary = summarize(combined, 20000, 12)
        self.assertAlmostEqual(summary.variance, 0.8, delta=0.05)
        self.assertEqual(summary.mean_cost, 2.0)
        with self.assertRaises(ValueError):
            pilot_covariance((first, second), 2, 1)
        with self.assertRaises(ValueError):
            pilot_covariance((), 10, 1)

    def test_control_variate_reaches_the_correlation_bound(self) -> None:
        """``exp(U)`` with control ``U``: variance ``(1 - rho^2) Var``."""
        target = estimator_from_function("exp", lambda g: math.exp(g.random()))
        control = estimator_from_function("u", lambda g: g.random(), cost=0.0)
        derived, beta = control_variate_estimator(
            "cv", target, (control,), (0.5,), pilot_size=5000, seed=1
        )
        variance = (math.e**2 - 1.0) / 2.0 - (math.e - 1.0) ** 2
        covariance = 1.0 - (math.e - 1.0) / 2.0
        self.assertAlmostEqual(beta[0], covariance * 12.0, delta=0.05)
        residual = variance - covariance**2 * 12.0
        bias = check_unbiasedness(derived, math.e - 1.0, replicates=20000, seed=2)
        self.assertEqual(bias.verdict, "not refuted")
        claim = check_variance(
            derived, residual, replicates=20000, seed=3, relative_tolerance=0.1
        )
        self.assertEqual(claim.verdict, "supported")
        gain = check_efficiency_gain(derived, target, replicates=5000, seed=4)
        self.assertEqual(gain.verdict, "supported")
        self.assertGreater(gain.estimate, 30.0)
        self.assertEqual(len(derived.lineage), 1)
        with self.assertRaises(ValueError):
            control_variate_estimator(
                "cv", target, (control,), (0.5, 0.1), pilot_size=10, seed=1
            )

    def test_antithetic_pair(self) -> None:
        """Linear integrands are integrated exactly; ``exp`` gains."""
        linear = antithetic_estimator("lin", lambda u: u[0] + 2.0 * u[1], 2)
        for seed in range(5):
            self.assertAlmostEqual(linear.draw(seed).value, 1.5, places=14)
        exponential = antithetic_estimator("anti", lambda u: math.exp(u[0]), 1)
        bias = check_unbiasedness(exponential, math.e - 1.0, replicates=20000, seed=1)
        self.assertEqual(bias.verdict, "not refuted")
        # Var = (Var f + Cov(f(U), f(1-U))) / 2 with Cov = e - (e-1)^2.
        variance = (math.e**2 - 1.0) / 2.0 - (math.e - 1.0) ** 2
        exact = 0.5 * (variance + math.e - (math.e - 1.0) ** 2)
        claim = check_variance(
            exponential, exact, replicates=20000, seed=2, relative_tolerance=0.1
        )
        self.assertEqual(claim.verdict, "supported")
        self.assertEqual(exponential.draw(1).cost, 2.0)
        with self.assertRaises(ValueError):
            antithetic_estimator("x", lambda u: 0.0, 0)


def plug_in_variance(data: tuple[float, ...] | list[float]) -> float:
    """Return the variance with divisor ``n``."""
    mean = math.fsum(data) / len(data)
    return math.fsum((x - mean) ** 2 for x in data) / len(data)


class BiasCorrectionTests(unittest.TestCase):
    """Jackknife, bootstrap, influence functions, one-step."""

    def setUp(self) -> None:
        """Draw one seeded sample."""
        generator = random.Random(4)
        self.data = [generator.gauss(2.0, 1.5) for _ in range(25)]

    def test_jackknife_turns_the_plug_in_variance_into_the_unbiased_one(self) -> None:
        """The bias ``-sigma^2/n`` is removed exactly."""
        corrected = jackknife_bias_corrected(plug_in_variance)
        self.assertAlmostEqual(
            corrected(self.data), statistics.variance(self.data), places=12
        )
        mean = jackknife_bias_corrected(statistics.fmean)
        self.assertAlmostEqual(mean(self.data), statistics.fmean(self.data), places=12)
        with self.assertRaises(ValueError):
            corrected([1.0])

    def test_bootstrap_correction_of_the_plug_in_variance(self) -> None:
        """``E*[theta*] = (n-1)/n theta_hat`` so ``theta_B ~ (1+1/n) theta_hat``."""
        corrected = bootstrap_bias_corrected(plug_in_variance, resamples=4000, seed=1)
        plain = plug_in_variance(self.data)
        self.assertAlmostEqual(corrected(self.data) / plain, 1.0 + 1.0 / 25, delta=0.01)
        self.assertEqual(corrected(self.data), corrected(self.data))
        with self.assertRaises(ValueError):
            bootstrap_bias_corrected(plug_in_variance, resamples=0, seed=1)

    def test_influence_of_the_mean_and_the_variance(self) -> None:
        """Numerical influence values equal the analytical ones."""

        def weighted_mean(data, weights) -> float:  # type: ignore[no-untyped-def]
            """Return the weighted mean."""
            return math.fsum(w * x for w, x in zip(weights, data, strict=True))

        def weighted_variance(data, weights) -> float:  # type: ignore[no-untyped-def]
            """Return the weighted plug-in variance."""
            mean = weighted_mean(data, weights)
            return math.fsum(
                w * (x - mean) ** 2 for w, x in zip(weights, data, strict=True)
            )

        mean = statistics.fmean(self.data)
        size = len(self.data)
        analysis = empirical_influence(weighted_mean, self.data)
        for value, x in zip(analysis.influence, self.data, strict=True):
            self.assertAlmostEqual(value, x - mean, places=8)
        self.assertAlmostEqual(
            analysis.standard_error,
            math.sqrt(plug_in_variance(self.data) / size),
            places=8,
        )
        self.assertAlmostEqual(analysis.mean_influence, 0.0, places=8)
        second = empirical_influence(weighted_variance, self.data)
        plain = plug_in_variance(self.data)
        for value, x in zip(second.influence, self.data, strict=True):
            self.assertAlmostEqual(value, (x - mean) ** 2 - plain, places=6)
        for epsilon in (0.0, 0.5):
            with self.assertRaises(ValueError):
                empirical_influence(weighted_mean, self.data, epsilon=epsilon)

    def test_influence_standard_error_matches_the_delta_method(self) -> None:
        """Ratio of means: the delta-method variance is reproduced."""
        generator = random.Random(8)
        pairs = [
            (generator.gauss(5.0, 1.0), generator.gauss(10.0, 2.0)) for _ in range(60)
        ]

        def ratio(data, weights) -> float:  # type: ignore[no-untyped-def]
            """Return the ratio of weighted means."""
            top = math.fsum(w * y for w, (_, y) in zip(weights, data, strict=True))
            bottom = math.fsum(w * x for w, (x, _) in zip(weights, data, strict=True))
            return top / bottom

        analysis = empirical_influence(ratio, pairs)
        size = len(pairs)
        mean_x = statistics.fmean(x for x, _ in pairs)
        mean_y = statistics.fmean(y for _, y in pairs)
        estimate = mean_y / mean_x
        residuals = [(y - estimate * x) / mean_x for x, y in pairs]
        delta = math.sqrt(math.fsum(r * r for r in residuals)) / size
        self.assertAlmostEqual(analysis.estimate, estimate, places=12)
        self.assertAlmostEqual(analysis.standard_error, delta, places=7)

    def test_one_step_repairs_a_wrong_plug_in(self) -> None:
        """For the mean with initial guess ``m`` the one-step is ``x_bar``."""
        guess = 0.7
        result = one_step_estimator(guess, [x - guess for x in self.data])
        self.assertAlmostEqual(result.estimate, statistics.fmean(self.data), places=12)
        self.assertAlmostEqual(
            result.standard_error,
            math.sqrt(statistics.variance(self.data) / len(self.data)),
            places=12,
        )
        self.assertAlmostEqual(result.plug_in + result.correction, result.estimate)
        with self.assertRaises(ValueError):
            one_step_estimator(0.0, [1.0])


class CertificationTests(unittest.TestCase):
    """Checks must refute false claims and must not refute true ones."""

    def test_interval_verdict_truth_table(self) -> None:
        """All four verdicts arise from the documented rules."""
        self.assertEqual(interval_verdict(-0.1, 0.1, 0.0, None), "not refuted")
        self.assertEqual(interval_verdict(0.1, 0.2, 0.0, None), "refuted")
        self.assertEqual(interval_verdict(-0.1, 0.1, 0.0, 0.2), "supported")
        self.assertEqual(interval_verdict(0.3, 0.4, 0.0, 0.2), "refuted")
        self.assertEqual(interval_verdict(0.1, 0.3, 0.0, 0.2), "inconclusive")
        with self.assertRaises(ValueError):
            interval_verdict(1.0, 0.0, 0.0, None)
        with self.assertRaises(ValueError):
            interval_verdict(0.0, 1.0, 0.0, -1.0)

    def test_unbiasedness_check_in_both_directions(self) -> None:
        """A bias of 0.05 is refuted; the unbiased estimator is supported."""
        fair = uniform_estimator()
        biased = estimator_from_function("biased", lambda g: g.random() + 0.05)
        good = check_unbiasedness(fair, 0.5, replicates=20000, seed=1, tolerance=0.02)
        self.assertEqual(good.verdict, "supported")
        self.assertLess(abs(good.estimate), 0.01)
        bad = check_unbiasedness(biased, 0.5, replicates=20000, seed=1, tolerance=0.02)
        self.assertEqual(bad.verdict, "refuted")
        vague = check_unbiasedness(fair, 0.5, replicates=50, seed=1, tolerance=0.02)
        self.assertEqual(vague.verdict, "inconclusive")
        self.assertIn("unbiasedness", good.sentence())
        with self.assertRaises(ValueError):
            check_unbiasedness(fair, 0.5, replicates=1, seed=1)
        with self.assertRaises(ValueError):
            check_unbiasedness(fair, 0.5, replicates=10, seed=1, confidence=1.0)

    def test_false_refutation_rate_is_near_the_nominal_level(self) -> None:
        """Over 400 seeds a true claim is refuted about 5% of the time."""
        fair = uniform_estimator()
        refuted = sum(
            check_unbiasedness(
                fair, 0.5, replicates=200, seed=seed, confidence=0.95
            ).verdict
            == "refuted"
            for seed in range(400)
        )
        self.assertLessEqual(refuted, 36)  # Binomial(400, 0.05) upper 0.1% point
        self.assertGreaterEqual(refuted, 6)

    def test_variance_check(self) -> None:
        """The uniform variance 1/12 is supported and 1/6 refuted."""
        fair = uniform_estimator()
        good = check_variance(
            fair, 1 / 12, replicates=20000, seed=1, relative_tolerance=0.1
        )
        self.assertEqual(good.verdict, "supported")
        bad = check_variance(fair, 1 / 6, replicates=20000, seed=1)
        self.assertEqual(bad.verdict, "refuted")
        with self.assertRaises(ValueError):
            check_variance(fair, 0.0, replicates=10, seed=1)

    def test_coverage_check(self) -> None:
        """A t interval covers; a too-narrow interval is refuted."""
        quantile = StudentT(9).quantile(0.975)

        def interval(width: float):  # type: ignore[no-untyped-def]
            """Return an interval procedure with the given critical value."""

            def procedure(stream: RandomStream) -> tuple[float, float]:
                """Return mean plus or minus ``width`` standard errors."""
                generator = stream.generator()
                data = [generator.gauss(1.0, 2.0) for _ in range(10)]
                centre = statistics.fmean(data)
                half = width * statistics.stdev(data) / math.sqrt(10)
                return centre - half, centre + half

            return procedure

        good = check_coverage(
            "t",
            interval(quantile),
            1.0,
            nominal=0.95,
            replicates=6000,
            seed=1,
            tolerance=0.02,
        )
        self.assertEqual(good.verdict, "supported")
        bad = check_coverage(
            "narrow", interval(1.0), 1.0, nominal=0.95, replicates=2000, seed=1
        )
        self.assertEqual(bad.verdict, "refuted")
        wide = check_coverage(
            "wide",
            interval(5.0),
            1.0,
            nominal=0.95,
            replicates=2000,
            seed=1,
            conservative_ok=True,
        )
        self.assertEqual(wide.verdict, "supported")
        strict = check_coverage(
            "wide", interval(5.0), 1.0, nominal=0.95, replicates=2000, seed=1
        )
        self.assertEqual(strict.verdict, "refuted")

    def test_convergence_rate_of_the_sample_mean(self) -> None:
        """The mean squared error of a sample mean decays at rate one."""
        base = uniform_estimator()
        result = check_convergence_rate(
            "mean",
            lambda n: sample_mean(base, n),
            (4, 16, 64, 256),
            0.5,
            expected_rate=1.0,
            replicates=1500,
            seed=1,
            tolerance=0.15,
        )
        self.assertEqual(result.verdict, "supported")
        wrong = check_convergence_rate(
            "mean",
            lambda n: sample_mean(base, n),
            (4, 16, 64, 256),
            0.5,
            expected_rate=2.0,
            replicates=1500,
            seed=1,
        )
        self.assertEqual(wrong.verdict, "refuted")
        with self.assertRaises(ValueError):
            check_convergence_rate(
                "mean",
                lambda n: base,
                (4, 4, 8),
                0.5,
                expected_rate=1.0,
                replicates=10,
                seed=1,
            )

    def test_efficiency_gain_is_inconclusive_for_equal_estimators(self) -> None:
        """A mean of ten has the same work-normalized variance as one draw."""
        base = uniform_estimator()
        result = check_efficiency_gain(
            sample_mean(base, 10), base, replicates=2000, seed=1
        )
        self.assertEqual(result.verdict, "inconclusive")
        self.assertAlmostEqual(result.estimate, 1.0, delta=0.15)
        worse = check_efficiency_gain(
            estimator_from_function("costly", lambda g: g.random(), cost=5.0),
            base,
            replicates=2000,
            seed=1,
        )
        self.assertEqual(worse.verdict, "refuted")

    def test_metamorphic_relations(self) -> None:
        """Permutation invariance holds for the mean, not for 'first'."""

        def generate(stream: RandomStream) -> list[float]:
            """Return a random sample."""
            generator = stream.generator()
            return [generator.gauss(0.0, 1.0) for _ in range(12)]

        def reverse(data: list[float]) -> list[float]:
            """Return the reversed sample."""
            return data[::-1]

        holds = check_invariance(
            "mean is permutation invariant",
            statistics.fmean,
            generate,
            reverse,
            lambda value: value,
            cases=50,
            seed=1,
        )
        self.assertTrue(holds.holds)
        broken = check_invariance(
            "first element is permutation invariant",
            lambda d: d[0],
            generate,
            reverse,
            lambda value: value,
            cases=50,
            seed=1,
        )
        self.assertFalse(broken.holds)
        self.assertEqual(broken.violations, 50)
        shift = check_invariance(
            "mean is translation equivariant",
            statistics.fmean,
            generate,
            lambda d: [x + 3.0 for x in d],
            lambda value: value + 3.0,
            cases=50,
            seed=2,
        )
        self.assertTrue(shift.holds)

    def test_method_card(self) -> None:
        """The card lists lineage, checks, scope, and a scoped verdict."""
        estimator = single_term_estimator("st", noisy_difference, geometric_levels(0.5))
        check = check_unbiasedness(
            estimator, 1.0, replicates=20000, seed=3, tolerance=0.03
        )
        card = method_card(estimator, "geometric differences, seed 3", [check])
        self.assertEqual(card.verdict, "supported within scope")
        text = card.render()
        self.assertIn("single-term randomized truncation", text)
        self.assertIn("Scope of the evidence", text)
        self.assertNotIn("validated", text)
        biased = estimator_from_function("b", lambda g: g.random() + 0.2)
        failing = method_card(
            biased, "uniform", [check_unbiasedness(biased, 0.5, replicates=500, seed=1)]
        )
        self.assertEqual(failing.verdict, "refuted")
        self.assertEqual(failing.refuted, ("unbiasedness",))
        open_card = method_card(
            biased, "uniform", [check_unbiasedness(biased, 0.7, replicates=500, seed=1)]
        )
        self.assertEqual(open_card.verdict, "not refuted within scope")
        with self.assertRaises(ValueError):
            method_card(biased, " ", [])


def normal_sample(generator: random.Random) -> list[float]:
    """Return fifteen normal observations with mean one."""
    return [generator.gauss(1.0, 2.0) for _ in range(15)]


def mean_method(data: list[float]) -> MethodOutput:
    """Return the mean with its t interval."""
    centre = statistics.fmean(data)
    error = statistics.stdev(data) / math.sqrt(len(data))
    half = StudentT(len(data) - 1).quantile(0.975) * error
    return MethodOutput(centre, error, centre - half, centre + half)


def median_method(data: list[float]) -> MethodOutput:
    """Return the median without uncertainty."""
    return MethodOutput(statistics.median(data))


def protocol(**changes: object) -> StudyProtocol:
    """Return the study protocol, optionally modified."""
    fields: dict[str, object] = {
        "aims": "compare mean and median for a normal location",
        "mechanisms": ("normal",),
        "estimand": "population mean",
        "methods": ("mean", "median"),
        "measures": MEASURES,
        "repetitions": 4000,
        "seed": 7,
        "primary_comparison": "relative precision of median against mean",
    }
    fields.update(changes)
    return StudyProtocol(**fields)  # type: ignore[arg-type]


class SimulationStudyTests(unittest.TestCase):
    """ADEMP studies, performance measures, and the protocol fingerprint."""

    def test_performance_measures_by_hand(self) -> None:
        """A four-point example reproduces the formulas of the source."""
        estimates = [1.0, 2.0, 3.0, 6.0]
        errors = [1.0, 1.0, 2.0, 2.0]
        intervals = [(0.0, 2.0), (1.0, 3.5), (2.5, 4.0), (5.0, 7.0)]
        result = performance_measures(
            estimates, 2.5, standard_errors=errors, intervals=intervals
        )
        spread = statistics.stdev(estimates)
        self.assertAlmostEqual(result["bias"][0], 0.5)
        self.assertAlmostEqual(result["bias"][1], spread / 2.0)
        self.assertAlmostEqual(result["empirical_se"][0], spread)
        self.assertAlmostEqual(result["empirical_se"][1], spread / math.sqrt(6.0))
        squared = [(e - 2.5) ** 2 for e in estimates]
        self.assertAlmostEqual(result["mse"][0], statistics.fmean(squared))
        self.assertAlmostEqual(
            result["mse"][1], math.sqrt(statistics.variance(squared) / 4.0)
        )
        model = math.sqrt(2.5)
        self.assertAlmostEqual(result["model_se"][0], model)
        self.assertAlmostEqual(result["model_se"][1], math.sqrt(3.0 / (4 * 4 * 2.5)))
        self.assertAlmostEqual(
            result["relative_error_model_se"][0], model / spread - 1.0
        )
        self.assertAlmostEqual(result["coverage"][0], 0.5)
        self.assertAlmostEqual(result["coverage"][1], 0.25)
        self.assertAlmostEqual(result["bias_eliminated_coverage"][0], 0.5)
        bare = performance_measures(estimates, 2.5)
        self.assertTrue(math.isnan(bare["coverage"][0]))
        with self.assertRaises(ValueError):
            performance_measures([1.0], 0.0)
        with self.assertRaises(ValueError):
            performance_measures(estimates, 0.0, standard_errors=[1.0])

    def test_normal_location_study_reproduces_known_results(self) -> None:
        """Unbiased mean, 95% coverage, median efficiency below one."""
        plan = protocol()
        result = run_simulation_study(
            plan,
            [DataGeneratingMechanism("normal", normal_sample, 1.0)],
            [Method("mean", mean_method), Method("median", median_method)],
            expected_fingerprint=plan.fingerprint(),
        )
        mean = result.performance("normal", "mean")
        self.assertLess(abs(mean.value("bias")), 4 * mean.standard_error("bias"))
        self.assertAlmostEqual(
            mean.value("empirical_se"), 2.0 / math.sqrt(15), delta=0.03
        )
        self.assertAlmostEqual(mean.value("coverage"), 0.95, delta=0.012)
        self.assertLess(abs(mean.value("relative_error_model_se")), 0.05)
        self.assertEqual(mean.failures, 0)
        median = result.performance("normal", "median")
        self.assertTrue(math.isnan(median.value("coverage")))
        comparison = result.compare_precision("normal", "median", "mean")
        # Exact variance ratio for n = 15 lies between 2/pi and 1.
        self.assertLess(comparison.relative_gain + 4 * comparison.standard_error, 0.0)
        self.assertGreater(comparison.relative_gain, 2.0 / math.pi - 1.0 - 0.05)
        self.assertGreater(comparison.correlation, 0.5)
        text = result.render()
        self.assertIn(plan.fingerprint(), text)
        self.assertIn("| mean | 0 |", text)
        with self.assertRaises(KeyError):
            result.performance("normal", "mode")

    def test_fingerprint_detects_a_changed_design(self) -> None:
        """Any change alters the hash; a mismatch stops the run."""
        plan = protocol()
        self.assertEqual(plan.fingerprint(), protocol().fingerprint())
        self.assertEqual(len(plan.fingerprint()), 64)
        changed = protocol(repetitions=4001)
        self.assertNotEqual(plan.fingerprint(), changed.fingerprint())
        with self.assertRaises(ValueError):
            run_simulation_study(
                changed,
                [DataGeneratingMechanism("normal", normal_sample, 1.0)],
                [Method("mean", mean_method), Method("median", median_method)],
                expected_fingerprint=plan.fingerprint(),
            )
        for bad in (
            {"aims": " "},
            {"methods": ("mean", "mean")},
            {"repetitions": 1},
            {"mechanisms": ()},
        ):
            with self.assertRaises(ValueError):
                protocol(**bad)

    def test_failures_are_counted_and_names_checked(self) -> None:
        """A method failing on some data sets is reported, not dropped."""

        def fragile(data: list[float]) -> MethodOutput:
            """Fail when the first observation is negative."""
            if data[0] < 0.0:
                raise RuntimeError("no convergence")
            return MethodOutput(statistics.fmean(data))

        plan = protocol(
            methods=("mean", "fragile"), repetitions=400, measures=("bias",)
        )
        result = run_simulation_study(
            plan,
            [DataGeneratingMechanism("normal", normal_sample, 1.0)],
            [Method("mean", mean_method), Method("fragile", fragile)],
        )
        fragile_record = result.performance("normal", "fragile")
        self.assertGreater(fragile_record.failures, 80)
        self.assertEqual(
            sum(math.isnan(e) for e in fragile_record.estimates),
            fragile_record.failures,
        )
        # Conditioning on success biases the fragile method upwards.
        self.assertGreater(fragile_record.value("bias"), 0.0)
        with self.assertRaises(ValueError):
            run_simulation_study(
                plan,
                [DataGeneratingMechanism("other", normal_sample, 1.0)],
                [Method("mean", mean_method), Method("fragile", fragile)],
            )
        with self.assertRaises(ValueError):
            run_simulation_study(
                protocol(measures=("power",)),
                [DataGeneratingMechanism("normal", normal_sample, 1.0)],
                [Method("mean", mean_method), Method("median", median_method)],
            )

    def test_repetition_planning(self) -> None:
        """The planning formulas invert the standard errors."""
        self.assertEqual(
            repetitions_for_standard_error("coverage", 0.005, pilot=0.9), 3600
        )
        self.assertEqual(repetitions_for_standard_error("bias", 0.01, pilot=0.04), 400)
        for arguments in (
            ("power", 0.1, 0.5),
            ("bias", 0.0, 1.0),
            ("coverage", 0.1, 1.0),
        ):
            with self.assertRaises(ValueError):
                repetitions_for_standard_error(
                    arguments[0], arguments[1], pilot=arguments[2]
                )


class SearchTests(unittest.TestCase):
    """Efficiency ranking and sequential halving."""

    def test_ranking_orders_by_work_normalized_variance(self) -> None:
        """A costly copy ranks below the plain estimator."""
        plain = uniform_estimator()
        costly = estimator_from_function("costly", lambda g: g.random(), cost=4.0)
        anti = antithetic_estimator("anti", lambda u: u[0] ** 2, 1)
        square = estimator_from_function("square", lambda g: g.random() ** 2)
        ranking = rank_by_efficiency((costly, plain), replicates=2000, seed=1)
        self.assertEqual(ranking.best, "uniform")
        self.assertTrue(ranking.separated())
        second = rank_by_efficiency((square, anti), replicates=2000, seed=1)
        self.assertEqual(second.best, "anti")
        tie = rank_by_efficiency(
            (plain, estimator_from_function("twin", lambda g: 1.0 - g.random())),
            replicates=2000,
            seed=1,
        )
        self.assertFalse(tie.separated())
        with self.assertRaises(ValueError):
            rank_by_efficiency((plain, plain), replicates=10, seed=1)
        with self.assertRaises(ValueError):
            rank_by_efficiency((), replicates=10, seed=1)

    def test_halving_returns_the_best_deterministic_candidate(self) -> None:
        """Without noise the minimizer always survives within budget."""
        losses = [0.9, 0.3, 0.7, 0.1, 0.5, 0.8, 0.2]
        result = sequential_halving(
            len(losses), lambda index, stream: losses[index], budget=100, seed=1
        )
        self.assertEqual(result.best, 3)
        self.assertLessEqual(result.total_pulls, 100)
        self.assertEqual(len(result.rounds), 3)
        self.assertEqual(result.rounds[0].pulls, 100 // (7 * 3))
        self.assertEqual(len(result.rounds[1].survivors), 4)
        best_reward = sequential_halving(
            len(losses),
            lambda index, stream: losses[index],
            budget=100,
            seed=1,
            minimize=False,
        )
        self.assertEqual(best_reward.best, 0)
        single = sequential_halving(1, lambda index, stream: 0.0, budget=0, seed=1)
        self.assertEqual(single.best, 0)
        with self.assertRaises(ValueError):
            sequential_halving(8, lambda index, stream: 0.0, budget=10, seed=1)
        with self.assertRaises(ValueError):
            sequential_halving(0, lambda index, stream: 0.0, budget=10, seed=1)

    def test_halving_finds_the_best_noisy_candidate(self) -> None:
        """With gaps of 0.2 and unit noise the best arm wins most runs."""
        means = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

        def pull(index: int, stream: RandomStream) -> float:
            """Return a noisy loss of one candidate."""
            return means[index] + stream.generator().gauss(0.0, 1.0)

        wins = sum(
            sequential_halving(8, pull, budget=2400, seed=seed).best == 0
            for seed in range(40)
        )
        self.assertGreaterEqual(wins, 30)

    def test_sequential_comparison_decides_for_a_better_method(self) -> None:
        """A method with half the error scale is declared better early."""

        def loss(scale: float):  # type: ignore[no-untyped-def]
            """Return a bounded absolute-error loss with the given scale."""

            def pull(index: int, stream: RandomStream) -> float:
                """Return ``min(1, |scale * z|)``."""
                return min(1.0, abs(stream.generator().gauss(0.0, scale)))

            return pull

        result = compare_sequentially(
            loss(0.5), loss(1.0), bound=1.0, max_pairs=2000, seed=1
        )
        self.assertTrue(result.decided)
        self.assertLess(result.pairs, 200)
        self.assertGreater(result.mean_difference, 0.0)
        self.assertLessEqual(result.p_value, 0.05)
        same = compare_sequentially(
            loss(1.0), loss(1.0), bound=1.0, max_pairs=300, seed=1
        )
        self.assertFalse(same.decided)
        self.assertEqual(same.pairs, 300)
        with self.assertRaises(ValueError):
            compare_sequentially(loss(1.0), loss(1.0), bound=0.0, max_pairs=10, seed=1)
        with self.assertRaises(ValueError):
            compare_sequentially(loss(1.0), loss(1.0), bound=1.0, max_pairs=0, seed=1)

    def test_sequential_comparison_keeps_its_level_under_stopping(self) -> None:
        """Equal methods are wrongly preferred in at most about 5% of runs."""

        def arm(label: str):  # type: ignore[no-untyped-def]
            """Return an independent uniform loss on its own child stream."""

            def pull(index: int, stream: RandomStream) -> float:
                """Return one uniform loss."""
                return stream.child(label).generator().random()

            return pull

        wrong = sum(
            compare_sequentially(
                arm("a"), arm("b"), bound=1.0, max_pairs=300, seed=seed
            ).decided
            for seed in range(200)
        )
        self.assertLessEqual(wrong, 21)  # Binomial(200, 0.05) upper 0.1% point


if __name__ == "__main__":
    unittest.main()
