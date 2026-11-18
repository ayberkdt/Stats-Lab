"""Wilks order-statistic bounds and Monte Carlo certification of a risk.

Exact values are pinned from base R 4.6.2 (``pbinom``, ``qbinom``, ``ppois``,
``pbeta``); each pin carries the R expression that produced it. The tests
never call R. Monte Carlo checks compare an observed frequency with the exact
probability on the scale of its binomial standard error.

References:
    [REF-WILKS-1941], [REF-WILKS-1942], [REF-NUTT-WALLIS-2004],
    [REF-CLOPPER-PEARSON-1934], [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011].
    Checks [EQ-PROP-WILKS-001], [EQ-PROP-WILKS-002], [EQ-PROP-WILKS-003],
    [EQ-REL-CERT-001], [EQ-REL-CERT-002], [EQ-REL-CERT-003].

"""

from __future__ import annotations

import math
import random
import unittest
from statistics import NormalDist

from probstat.propagation import (
    WilksBound,
    order_statistic_interval,
    wilks_bound,
    wilks_rank,
    wilks_sample_size,
    wilks_tolerance_interval,
)
from probstat.reliability import (
    CertificationProbability,
    certification_margin,
    certification_probability,
    certification_sample_size,
    moving_particles_certification_probability,
    success_run_sample_size,
)


class WilksRankTests(unittest.TestCase):
    """Ranks, achieved confidences, and sample sizes against R.

    References:
        [REF-WILKS-1941], [REF-NUTT-WALLIS-2004]. Checks [EQ-PROP-WILKS-001]
        and [EQ-PROP-WILKS-002].

    """

    def test_rank_for_thousand_values(self) -> None:
        """``n = 1000``, ``a = beta = 0.95`` gives rank 962.

        References:
            [REF-WILKS-1941]. Checks [EQ-PROP-WILKS-001].

        """
        result = wilks_rank(1000, 0.95, 0.95)
        # R: which(pbinom(0:999, 1000, 0.95) >= 0.95)[1] -> 962
        self.assertEqual(result.rank, 962)
        self.assertEqual(result.order, 39)
        # R: pbinom(961, 1000, 0.95) -> 0.95665180939309091
        self.assertAlmostEqual(result.achieved_confidence, 0.95665180939309091, 12)
        # R: pbinom(960, 1000, 0.95) -> 0.94018512094809747 (< 0.95: rank 961 fails)
        self.assertLess(0.94018512094809747, 0.95)

    def test_lower_rank_is_symmetric(self) -> None:
        """The lower bound of ``q_0.05`` mirrors the upper bound of ``q_0.95``.

        References:
            [REF-WILKS-1941]. Checks [EQ-PROP-WILKS-001].

        """
        result = wilks_rank(1000, 0.05, 0.95, sided="lower")
        # R: max(k[1 - pbinom(k - 1, 1000, 0.05) >= 0.95]) -> 39
        self.assertEqual(result.rank, 39)
        self.assertEqual(result.order, 39)
        # R: 1 - pbinom(38, 1000, 0.05) -> 0.95665180939309036
        self.assertAlmostEqual(result.achieved_confidence, 0.95665180939309036, 12)

    def test_small_and_large_sizes(self) -> None:
        """Ranks at ``n = 100`` and ``n = 1e6`` match R.

        References:
            [REF-WILKS-1941]. Checks [EQ-PROP-WILKS-001].

        """
        small = wilks_rank(100, 0.95, 0.95)
        # R: which(pbinom(0:99, 100, 0.95) >= 0.95)[1] -> 99
        self.assertEqual(small.rank, 99)
        # R: pbinom(98, 100, 0.95) -> 0.96291879067264496
        self.assertAlmostEqual(small.achieved_confidence, 0.96291879067264496, 12)
        large = wilks_rank(10**6, 0.95, 0.95)
        # R: qbinom(0.95, 1e6, 0.95) + 1 -> 950359
        self.assertEqual(large.rank, 950359)
        # R: pbinom(950358, 1e6, 0.95) -> 0.95012733511764569
        self.assertAlmostEqual(large.achieved_confidence, 0.95012733511764569, 9)

    def test_classical_sample_sizes(self) -> None:
        """95 %/95 % sizes 59, 93, 124, 153 and the related classics.

        References:
            [REF-NUTT-WALLIS-2004]. Checks [EQ-PROP-WILKS-002].

        """
        # R: smallest n with pbinom(n - m, n, 0.95) >= 0.95 for m = 1..4
        self.assertEqual(
            [wilks_sample_size(0.95, 0.95, order=m) for m in range(1, 5)],
            [59, 93, 124, 153],
        )
        # R: 1 - pbeta(0.95, n - 1, 2) >= 0.95 first holds at n = 93
        self.assertEqual(wilks_sample_size(0.95, 0.95, sided="two-sided"), 93)
        # R: 1 - pbeta(0.95, n - 3, 4) >= 0.95 first holds at n = 153
        self.assertEqual(wilks_sample_size(0.95, 0.95, order=2, sided="two-sided"), 153)
        # R: 1 - pbeta(0.99, n - 1, 2) >= 0.95 first holds at n = 473
        self.assertEqual(wilks_sample_size(0.99, 0.95, sided="two-sided"), 473)
        # R: 1 - 0.5^n >= 0.95 first holds at n = 5 (median)
        self.assertEqual(wilks_sample_size(0.5, 0.95), 5)
        # R: same search -> 299 (99/95), 90 (95/99), 459 (99/99)
        self.assertEqual(wilks_sample_size(0.99, 0.95), 299)
        self.assertEqual(wilks_sample_size(0.95, 0.99), 90)
        self.assertEqual(wilks_sample_size(0.99, 0.99), 459)
        self.assertEqual(wilks_sample_size(0.05, 0.95, sided="lower"), 59)

    def test_sample_size_is_minimal(self) -> None:
        """The size reaches the confidence and one run fewer does not.

        References:
            [REF-WILKS-1941]. Checks [EQ-PROP-WILKS-002].

        """
        for order in (1, 2, 3, 4):
            size = wilks_sample_size(0.95, 0.95, order=order)
            self.assertEqual(wilks_rank(size, 0.95, 0.95).order, order)
            if order == 1:
                with self.assertRaises(ValueError):
                    wilks_rank(size - 1, 0.95, 0.95)
            else:
                self.assertLess(wilks_rank(size - 1, 0.95, 0.95).order, order)

    def test_sample_size_equals_success_run(self) -> None:
        """Order ``m`` equals the success-run size with ``m - 1`` failures.

        References:
            [REF-WILKS-1941]. Checks [EQ-PROP-WILKS-002], [EQ-REL-DATA-004].

        """
        for level, confidence in ((0.95, 0.95), (0.9, 0.99), (0.999, 0.9)):
            for order in (1, 2, 5):
                self.assertEqual(
                    wilks_sample_size(level, confidence, order=order),
                    success_run_sample_size(
                        level, confidence, allowed_failures=order - 1
                    ),
                )

    def test_one_sided_rank_is_tighter_than_equal_tailed_interval(self) -> None:
        """The Wilks rank lies below the upper rank of the equal-tailed interval.

        References:
            [REF-DAVID-NAGARAJA-2003, Section 7.1]. Checks [EQ-PROP-WILKS-001],
            [EQ-PROP-QCI-001].

        """
        values = [float(i) for i in range(1000)]
        interval = order_statistic_interval(values, 0.95, confidence=0.90)
        rank = wilks_rank(1000, 0.95, 0.95).rank
        self.assertEqual(interval.upper_rank, rank)
        interval95 = order_statistic_interval(values, 0.95, confidence=0.95)
        self.assertGreater(interval95.upper_rank, rank)


class WilksBoundTests(unittest.TestCase):
    """Bounds and tolerance intervals on samples.

    References:
        [REF-WILKS-1941], [REF-WILKS-1942]. Checks [EQ-PROP-WILKS-001] and
        [EQ-PROP-WILKS-003].

    """

    def test_bound_picks_the_order_statistic(self) -> None:
        """The bound is the 962-th smallest of 1000 values.

        References:
            [REF-WILKS-1941]. Checks [EQ-PROP-WILKS-001].

        """
        rng = random.Random(7)
        values = [rng.gauss(0.0, 1.0) for _ in range(1000)]
        result = wilks_bound(values, 0.95, 0.95)
        self.assertIsInstance(result, WilksBound)
        self.assertEqual(result.bound, sorted(values)[961])
        self.assertEqual(result.size, 1000)
        self.assertGreaterEqual(result.bound, result.empirical_quantile)
        lower = wilks_bound(values, 0.05, 0.95, sided="lower")
        self.assertEqual(lower.bound, sorted(values)[38])

    def test_first_order_bound_is_the_maximum(self) -> None:
        """With 59 values the 95/95 upper bound is the maximum.

        References:
            [REF-WILKS-1941]. Checks [EQ-PROP-WILKS-001].

        """
        values = [float(i) for i in range(59)]
        result = wilks_bound(values, 0.95, 0.95)
        self.assertEqual(result.bound, 58.0)
        self.assertEqual(result.order, 1)
        # R: 1 - 0.95^59 -> 0.9515054747505769
        self.assertAlmostEqual(result.achieved_confidence, 0.9515054747505769, 12)

    def test_upper_bound_coverage_by_simulation(self) -> None:
        """The upper bound covers ``q_0.95`` in about 95.15 % of samples.

        References:
            [REF-WILKS-1941]. Checks [EQ-PROP-WILKS-001].

        """
        rng = random.Random(20261118)
        truth = NormalDist().inv_cdf(0.95)
        replications = 2000
        covered = 0
        for _ in range(replications):
            sample = [rng.gauss(0.0, 1.0) for _ in range(59)]
            covered += wilks_bound(sample, 0.95, 0.95).bound >= truth
        expected = 1.0 - 0.95**59
        error = math.sqrt(expected * (1.0 - expected) / replications)
        self.assertLess(abs(covered / replications - expected), 4.0 * error)

    def test_tolerance_interval(self) -> None:
        """For ``n = 1000`` the 95/95 interval drops 18 values at each end.

        References:
            [REF-WILKS-1942]. Checks [EQ-PROP-WILKS-003].

        """
        values = [float(i) for i in range(1000)]
        result = wilks_tolerance_interval(values, 0.95, 0.95)
        # R: largest m with 1 - pbeta(0.95, 1000 - 2m + 1, 2m) >= 0.95 -> 19
        self.assertEqual(result.order, 19)
        self.assertEqual((result.lower_rank, result.upper_rank), (19, 982))
        self.assertEqual((result.lower, result.upper), (18.0, 981.0))
        # R: 1 - pbeta(0.95, 963, 38) -> 0.96933560995207224
        self.assertAlmostEqual(result.achieved_confidence, 0.96933560995207224, 11)
        minimal = wilks_tolerance_interval([float(i) for i in range(93)], 0.95, 0.95)
        self.assertEqual((minimal.lower, minimal.upper), (0.0, 92.0))
        # R: 1 - pbeta(0.95, 92, 2) -> 0.95002420475738358
        self.assertAlmostEqual(minimal.achieved_confidence, 0.95002420475738358, 12)

    def test_tolerance_content_by_simulation(self) -> None:
        """The first-order interval of 93 uniforms holds 95 % content as stated.

        References:
            [REF-WILKS-1942]. Checks [EQ-PROP-WILKS-003].

        """
        rng = random.Random(4242)
        replications = 2000
        hits = 0
        for _ in range(replications):
            sample = [rng.random() for _ in range(93)]
            interval = wilks_tolerance_interval(sample, 0.95, 0.95)
            hits += interval.upper - interval.lower >= 0.95
        expected = 0.95002420475738358
        error = math.sqrt(expected * (1.0 - expected) / replications)
        self.assertLess(abs(hits / replications - expected), 4.0 * error)

    def test_edge_cases_and_errors(self) -> None:
        """Too-small samples, bad levels, and bad arguments raise.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaisesRegex(ValueError, "at least 59"):
            wilks_bound([float(i) for i in range(58)], 0.95, 0.95)
        with self.assertRaisesRegex(ValueError, "at least 93"):
            wilks_tolerance_interval([float(i) for i in range(92)], 0.95, 0.95)
        with self.assertRaisesRegex(ValueError, "at least 59"):
            wilks_rank(10, 0.05, 0.95, sided="lower")
        for bad in (0.0, 1.0, -0.1, math.nan):
            with self.assertRaises(ValueError):
                wilks_rank(100, bad, 0.95)
            with self.assertRaises(ValueError):
                wilks_sample_size(0.95, bad)
        with self.assertRaises(ValueError):
            wilks_rank(100, 0.5, 0.95, sided="both")
        with self.assertRaises(ValueError):
            wilks_sample_size(0.5, 0.95, sided="upper-ish")
        with self.assertRaises(TypeError):
            wilks_rank(True, 0.5, 0.95)
        with self.assertRaises(ValueError):
            wilks_rank(0, 0.5, 0.95)
        with self.assertRaises(ValueError):
            wilks_sample_size(0.5, 0.95, order=0)
        with self.assertRaises(ValueError):
            wilks_bound([], 0.5, 0.5)
        with self.assertRaises(ValueError):
            wilks_bound([1.0, math.inf], 0.5, 0.5)
        single = wilks_bound([3.0], 0.5, 0.5)
        self.assertEqual((single.bound, single.empirical_quantile), (3.0, 3.0))


class CertificationTests(unittest.TestCase):
    """Normal and Clopper-Pearson certification against R.

    References:
        [REF-CLOPPER-PEARSON-1934]. Checks [EQ-REL-CERT-001] and
        [EQ-REL-CERT-002].

    """

    def test_margin(self) -> None:
        """``c(1000, 0.01, 0.05) = qnorm(0.95) sqrt(0.01 * 0.99 / 1000)``.

        References:
            [REF-ROSS-2019, Section 8.3]. Checks [EQ-REL-CERT-001].

        """
        # R: qnorm(0.95) * sqrt(0.01 * 0.99 / 1000) -> 0.0051754111136744071
        self.assertAlmostEqual(
            certification_margin(1000, 0.01), 0.0051754111136744071, 14
        )

    def test_exact_probabilities(self) -> None:
        """Certifying counts and probabilities match R, up to ``n = 1e10``.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-REL-CERT-002].

        """
        # (n, p_f, alpha, k_normal, P_normal, k_cp, P_cp) with
        # R: k = floor(n * (a - qnorm(0.95) * sqrt(a * (1 - a) / n))),
        #    k_cp = largest k with pbinom(k, n, a) <= 0.05 (from qbinom),
        #    P = pbinom(k, n, pf).
        cases = (
            (1000, 5e-3, 1e-2, 4, 0.44005340580644114, 4, 0.44005340580644114),
            (10**4, 5e-3, 1e-2, 83, 0.999993254619297, 83, 0.999993254619297),
            (10**6, 9e-5, 1e-4, 83, 0.24955625389733277, 83, 0.24955625389733277),
            (3000, 1e-2, 1e-2, 21, 0.05358095419196323, 20, 0.034614897403148798),
            (
                10**10,
                1e-6,
                1e-6,
                9835,
                0.049689596177329813,
                9835,
                0.049689596177329813,
            ),
            (10**10, 5e-7, 1e-6, 9835, 1.0, 9835, 1.0),
        )
        for n, p_f, alpha, k_normal, p_normal, k_cp, p_cp in cases:
            with self.subTest(n=n, p_f=p_f, alpha=alpha):
                normal = certification_probability(n, p_f, alpha)
                self.assertIsInstance(normal, CertificationProbability)
                self.assertEqual(normal.max_failures, k_normal)
                self.assertTrue(
                    math.isclose(normal.probability, p_normal, rel_tol=1e-9)
                )
                exact = certification_probability(n, p_f, alpha, rule="clopper-pearson")
                self.assertEqual(exact.max_failures, k_cp)
                self.assertTrue(math.isclose(exact.probability, p_cp, rel_tol=1e-9))

    def test_no_certifying_outcome(self) -> None:
        """Fewer than about ``3/alpha`` runs never certify.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-REL-CERT-002].

        """
        for rule in ("normal", "clopper-pearson"):
            # R: floor(59 * (0.01 - qnorm(0.95) * sqrt(0.0099 / 59))) -> -1
            result = certification_probability(59, 1e-3, 1e-2, rule=rule)
            self.assertEqual((result.max_failures, result.probability), (-1, 0.0))
            # R: floor(299 * (0.01 - qnorm(0.95) * sqrt(0.0099 / 299))) -> 0 and
            # R: pbinom(0, 299, 0.01) -> 0.04953626 <= 0.05; p_f = 0 then certifies.
            certain = certification_probability(299, 0.0, 1e-2, rule=rule)
            self.assertEqual(certain.probability, 1.0)

    def test_clopper_pearson_bounds_false_certification(self) -> None:
        """At ``p_f = alpha`` the exact rule never exceeds ``beta``; normal can.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-REL-CERT-002].

        """
        normal_peak = 0.0
        for n in range(1, 4001, 7):
            exact = certification_probability(n, 0.01, 0.01, rule="clopper-pearson")
            self.assertLessEqual(exact.probability, 0.05)
            normal_peak = max(
                normal_peak, certification_probability(n, 0.01, 0.01).probability
            )
        self.assertGreater(normal_peak, 0.05)
        worse = certification_probability(2000, 0.02, 0.01, rule="clopper-pearson")
        self.assertLess(worse.probability, 0.05)

    def test_frequency_by_simulation(self) -> None:
        """Simulated certification frequency matches the exact probability.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-REL-CERT-002].

        """
        rng = random.Random(99)
        n, p_f, alpha = 800, 0.006, 0.01
        exact = certification_probability(n, p_f, alpha)
        margin = certification_margin(n, alpha)
        replications = 1500
        certified = 0
        for _ in range(replications):
            failures = sum(rng.random() < p_f for _ in range(n))
            certified += failures / n + margin <= alpha
        error = math.sqrt(exact.probability * (1.0 - exact.probability) / replications)
        self.assertLess(abs(certified / replications - exact.probability), 4.0 * error)

    def test_sample_sizes_reproduce_r_brute_force(self) -> None:
        """Stable and first sizes match an R scan over every ``n``.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-REL-CERT-002].

        """
        # R: p = pbinom(k_n, n, pf) for n = 1..N; stable = max(which(p < 0.95)) + 1,
        #    first = which(p >= 0.95)[1]; k_n as in test_exact_probabilities.
        cases = (
            (1e-3, 1e-2, "normal", 0.95, 602, 602),
            (1e-3, 1e-2, "clopper-pearson", 0.95, 628, 628),
            (5e-3, 1e-2, "normal", 0.95, 3347, 3114),
            (5e-3, 1e-2, "clopper-pearson", 0.95, 3487, 3137),
            (9e-3, 1e-2, "normal", 0.95, 103260, 101412),
            (9e-3, 1e-2, "clopper-pearson", 0.95, 103692, 101845),
            (1e-5, 1e-4, "normal", 0.95, 60434, 60434),
            (5e-5, 1e-4, "normal", 0.95, 335231, 311850),
            (5e-3, 1e-2, "normal", 0.5, 1157, 888),
        )
        for p_f, alpha, rule, power, stable, first in cases:
            with self.subTest(p_f=p_f, alpha=alpha, rule=rule, power=power):
                result = certification_sample_size(p_f, alpha, power=power, rule=rule)
                self.assertEqual((result.size, result.first_size), (stable, first))
                self.assertGreaterEqual(result.probability, power)

    def test_sample_size_brute_force_in_python(self) -> None:
        """For ``alpha = 0.1`` the stable size agrees with a full scan.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-REL-CERT-002].

        """
        for p_f in (0.0, 0.02, 0.05):
            for rule in ("normal", "clopper-pearson"):
                result = certification_sample_size(p_f, 0.1, power=0.9, rule=rule)
                values = [
                    certification_probability(n, p_f, 0.1, rule=rule).probability
                    for n in range(1, 3 * result.size + 100)
                ]
                failing = [n for n, v in enumerate(values, start=1) if v < 0.9]
                self.assertEqual(result.size, failing[-1] + 1)
                self.assertEqual(
                    result.first_size,
                    next(n for n, v in enumerate(values, start=1) if v >= 0.9),
                )

    def test_rare_event_sizes_scale_with_inverse_risk(self) -> None:
        """``n alpha`` at 95 % power is near 6, 34, and 1040 for the three ratios.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-REL-CERT-002].

        """
        expected = {0.1: 6043694, 0.5: 33523632, 0.9: 1042098397}
        for ratio, size in expected.items():
            result = certification_sample_size(ratio * 1e-6, 1e-6)
            self.assertEqual(result.size, size)

    def test_errors(self) -> None:
        """Invalid arguments raise.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaises(ValueError):
            certification_sample_size(0.01, 0.01)
        with self.assertRaises(ValueError):
            certification_probability(10, 1.5, 0.01)
        with self.assertRaises(ValueError):
            certification_probability(10, 0.001, 0.01, rule="wald")
        with self.assertRaises(ValueError):
            certification_margin(10, 0.0)
        with self.assertRaises(TypeError):
            certification_margin(10.0, 0.01)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            certification_sample_size(0.001, 0.01, power=1.0)


class MovingParticlesTests(unittest.TestCase):
    """Poisson certification of the moving-particles estimate.

    References:
        [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011], [REF-WALTER-2015].
        Checks [EQ-REL-CERT-003].

    """

    def test_values_against_r(self) -> None:
        """Minimal move counts and probabilities match R.

        References:
            [REF-WALTER-2015]. Checks [EQ-REL-CERT-003].

        """
        # R: la = -N log(a); m = smallest m with ppois(m - 1, la, lower = FALSE)
        #    <= 0.05; P = ppois(m - 1, -N log(pf), lower = FALSE).
        cases = (
            (100, 1e-5, 1e-4, 972, 0.99999997358574522),
            (10, 1e-3, 1e-2, 58, 0.9213484366304856),
            (1000, 5e-7, 1e-6, 14010, 0.99998455202273062),
            (50, 1e-4, 1e-4, 497, 0.048127282056286844),
            (1, 1e-3, 1e-2, 9, 0.25896377281338467),
        )
        for particles, p_f, alpha, moves, probability in cases:
            with self.subTest(particles=particles, p_f=p_f, alpha=alpha):
                result = moving_particles_certification_probability(
                    particles, p_f, alpha
                )
                self.assertEqual(result.min_moves, moves)
                self.assertTrue(
                    math.isclose(result.probability, probability, rel_tol=1e-9)
                )
                self.assertAlmostEqual(
                    result.expected_calls,
                    particles * (1.0 - 10.0 * math.log(p_f)),
                    places=6,
                )

    def test_false_certification_and_limits(self) -> None:
        """``p_f >= alpha`` certifies with probability at most ``beta``.

        References:
            [REF-WALTER-2015]. Checks [EQ-REL-CERT-003].

        """
        for particles in (1, 7, 30, 200, 5000):
            at_risk = moving_particles_certification_probability(particles, 1e-3, 1e-3)
            self.assertLessEqual(at_risk.probability, 0.05)
        zero = moving_particles_certification_probability(10, 0.0, 1e-3)
        self.assertEqual(zero.probability, 1.0)
        one = moving_particles_certification_probability(10, 1.0, 1e-3)
        self.assertEqual(one.probability, 0.0)
        with self.assertRaises(ValueError):
            moving_particles_certification_probability(0, 1e-4, 1e-3)
        with self.assertRaises(ValueError):
            moving_particles_certification_probability(10, 1e-4, 1e-3, steps=0)

    def test_fewer_calls_than_crude_monte_carlo_for_rare_events(self) -> None:
        """At ``alpha = 1e-6``, 32 particles certify ``alpha/10`` with ~5e3 calls.

        References:
            [REF-WALTER-2015]. Checks [EQ-REL-CERT-002], [EQ-REL-CERT-003].

        """
        particles = moving_particles_certification_probability(32, 1e-7, 1e-6)
        self.assertGreaterEqual(particles.probability, 0.95)
        self.assertLess(
            moving_particles_certification_probability(31, 1e-7, 1e-6).probability,
            0.95,
        )
        crude = certification_sample_size(1e-7, 1e-6)
        self.assertGreater(crude.size / particles.expected_calls, 1000.0)

    def test_frequency_by_simulation(self) -> None:
        """Simulated Poisson move counts certify at the exact rate.

        References:
            [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011]. Checks [EQ-REL-CERT-003].

        """
        rng = random.Random(5)
        result = moving_particles_certification_probability(10, 2e-3, 1e-2)
        mean = result.expected_moves
        replications = 3000
        certified = 0
        for _ in range(replications):
            # Poisson draw by counting unit-rate exponential arrivals before ``mean``.
            total, count = rng.expovariate(1.0), 0
            while total < mean:
                count += 1
                total += rng.expovariate(1.0)
            certified += count >= result.min_moves
        error = math.sqrt(
            result.probability * (1.0 - result.probability) / replications
        )
        self.assertLess(abs(certified / replications - result.probability), 4.0 * error)


if __name__ == "__main__":
    unittest.main()
