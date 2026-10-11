"""Tests for scoring rules, forecast calibration, and distribution-shift tests.

Closed forms are checked against an independent numerical integration of
the defining integral ``int (F(x) - 1{x >= y})^2 dx`` (composite Simpson,
split at ``y`` and at support kinks), and the ensemble CRPS against the
exact integral of the step CDF in rational arithmetic. Random checks are
seeded and their tolerances are stated in Monte Carlo standard errors.
"""

from __future__ import annotations

import itertools
import math
import random
import unittest
import warnings
from collections.abc import Callable, Sequence
from fractions import Fraction

from probstat.distributions import (
    Exponential,
    Gamma,
    Logistic,
    LogNormal,
    Normal,
    Poisson,
    Uniform,
    Weibull,
)
from probstat.inference import (
    SmallExpectedCountWarning,
    kolmogorov_smirnov_test,
    kolmogorov_smirnov_two_sample_test,
)
from probstat.numerics.quadrature import composite_simpson
from probstat.validation import (
    ValidationExperiment,
    area_metric,
    assess_predictions,
    brier_decomposition,
    brier_score,
    crps_distribution,
    crps_ensemble,
    crps_exponential,
    crps_lognormal,
    crps_normal,
    crps_quadrature,
    crps_uniform,
    dawid_sebastiani_score,
    distribution_wasserstein_distance,
    energy_distance,
    energy_distance_test,
    energy_score,
    ensemble_pit,
    interval_coverage,
    interval_score,
    log_score,
    mmd_squared,
    mmd_test,
    normal_wasserstein_distance,
    pinball_loss,
    pit_coverage,
    pit_histogram,
    pit_values,
    reliability_diagram,
    sharpness_summary,
    variogram_score,
    wasserstein_distance,
    weighted_interval_score,
)


def _exact_ensemble_crps(members: Sequence[int], y: int) -> Fraction:
    """Integrate ``(F_m(x) - 1{x >= y})^2`` exactly over the merged breakpoints."""
    m = len(members)
    points = sorted(set(members) | {y})
    total = Fraction(0)
    for left, right in zip(points, points[1:], strict=False):
        cdf = Fraction(sum(1 for x in members if x <= left), m)
        step = 1 if left >= y else 0
        total += (cdf - step) ** 2 * (right - left)
    return total


def _definition_crps(
    cdf: Callable[[float], float],
    y: float,
    lower: float,
    upper: float,
    kinks: Sequence[float] = (),
) -> float:
    """Integrate the CRPS definition by Simpson's rule between breakpoints."""
    points = sorted({lower, upper, y, *[k for k in kinks if lower < k < upper]})
    total = 0.0
    for left, right in zip(points, points[1:], strict=False):
        if right <= left:
            continue
        step = 1.0 if left >= y else 0.0

        def integrand(x: float, step: float = step) -> float:
            """Return the squared CDF-step difference."""
            return (cdf(x) - step) ** 2

        total += composite_simpson(integrand, left, right, intervals=4000).value
    return total


class EnsembleCrpsTests(unittest.TestCase):
    """The sorted ensemble CRPS is the CRPS of the empirical CDF."""

    def test_matches_exact_integral_of_step_cdf(self) -> None:
        """Small integer ensembles, with ties and outside observations, agree."""
        cases = [
            ([1, 2, 3], 2),
            ([0, 0, 5], 1),
            ([3, 3, 3, 3], 3),
            ([-4, 7, 1, 1, 9], 20),
            ([-4, 7, 1, 1, 9], -10),
            ([2], 5),
        ]
        for members, y in cases:
            with self.subTest(members=members, y=y):
                exact = _exact_ensemble_crps(members, y)
                self.assertAlmostEqual(
                    crps_ensemble(members, y), float(exact), places=13
                )

    def test_sorted_formula_matches_double_sum_and_fair_version(self) -> None:
        """Both denominators reproduce the naive O(m^2) kernel expression."""
        generator = random.Random(11)
        members = [generator.gauss(3.0, 2.0) for _ in range(37)]
        y = 1.3
        m = len(members)
        accuracy = math.fsum(abs(x - y) for x in members) / m
        pairs = math.fsum(abs(a - b) for a in members for b in members)
        self.assertAlmostEqual(
            crps_ensemble(members, y), accuracy - pairs / (2 * m * m), places=12
        )
        self.assertAlmostEqual(
            crps_ensemble(members, y, fair=True),
            accuracy - pairs / (2 * m * (m - 1)),
            places=12,
        )
        shifted = [x + 1.0e9 for x in members]
        self.assertAlmostEqual(
            crps_ensemble(shifted, y + 1.0e9), crps_ensemble(members, y), places=6
        )
        with self.assertRaises(ValueError):
            crps_ensemble([1.0], 0.0, fair=True)

    def test_fair_crps_is_unbiased_for_the_normal_closed_form(self) -> None:
        """Replicate ensembles from N(1, 2): fair mean within 4 SE of the exact."""
        generator = random.Random(2026)
        y, mu, sigma = 0.4, 1.0, 2.0
        exact = crps_normal(y, mu, sigma)
        fair: list[float] = []
        plain: list[float] = []
        for _ in range(40):
            members = [generator.gauss(mu, sigma) for _ in range(400)]
            fair.append(crps_ensemble(members, y, fair=True))
            plain.append(crps_ensemble(members, y))
        mean = math.fsum(fair) / len(fair)
        se = math.sqrt(
            math.fsum((v - mean) ** 2 for v in fair) / (len(fair) - 1) / len(fair)
        )
        self.assertLess(abs(mean - exact), 4.0 * se)
        expected_bias = (2.0 * sigma / math.sqrt(math.pi)) / (2 * 400)
        bias = math.fsum(plain) / len(plain) - mean
        self.assertAlmostEqual(bias, expected_bias, delta=0.1 * expected_bias)


class ClosedFormCrpsTests(unittest.TestCase):
    """Closed forms agree with the defining integral and with quadrature."""

    def test_closed_forms_match_numerical_definition(self) -> None:
        """Each family and several observations, to 1e-8 absolute."""
        cases = [
            (Normal(1.0, 2.0), (-3.0, 0.4, 1.0, 6.5), -25.0, 25.0, ()),
            (LogNormal(0.2, 0.5), (-1.0, 0.3, 1.7, 5.0), -1.0, 60.0, (0.0,)),
            (Logistic(1.0, 0.7), (-2.0, 1.0, 3.3), -45.0, 45.0, ()),
            (Exponential(1.5), (-0.5, 0.0, 0.4, 3.0), -0.5, 40.0, (0.0,)),
            (Uniform(-1.0, 2.0), (-2.0, -1.0, 0.5, 2.0, 3.0), -2.0, 3.0, (-1.0, 2.0)),
        ]
        for law, observations, lower, upper, kinks in cases:
            for y in observations:
                with self.subTest(law=law, y=y):
                    low = min(lower, y)
                    numeric = _definition_crps(law.cdf, y, low, upper, kinks)
                    closed = crps_distribution(law, y, method="closed-form")
                    self.assertAlmostEqual(closed, numeric, delta=1e-8)
                    quadrature = crps_distribution(law, y, method="quadrature")
                    self.assertAlmostEqual(closed, quadrature, delta=1e-9)

    def test_quadrature_for_families_without_closed_form(self) -> None:
        """Weibull and gamma CRPS by quadrature match the definition."""
        for law, y in ((Weibull(1.7, 2.0), 1.2), (Gamma(2.5, 1.5), 0.9)):
            with self.subTest(law=law):
                numeric = _definition_crps(law.cdf, y, 0.0, 80.0)
                self.assertAlmostEqual(crps_quadrature(law, y), numeric, delta=1e-8)
                self.assertAlmostEqual(crps_distribution(law, y), numeric, delta=1e-8)
                with self.assertRaises(ValueError):
                    crps_distribution(law, y, method="closed-form")

    def test_special_values(self) -> None:
        """Standard normal at its mean, exponential and uniform hand values."""
        self.assertAlmostEqual(
            crps_normal(0.0), (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi), places=15
        )
        self.assertAlmostEqual(crps_exponential(-2.0, 2.0), 2.25, places=15)
        self.assertAlmostEqual(crps_uniform(0.5), 0.25 - 1.0 / 6.0, places=15)
        self.assertAlmostEqual(crps_uniform(3.0, 0.0, 2.0), 2.0 - 2.0 / 6.0, places=15)
        self.assertGreater(crps_lognormal(-1.0, 0.0, 1.0), 1.0)


class OtherScoreTests(unittest.TestCase):
    """Hand examples for the remaining scores."""

    def test_interval_score_hand_examples(self) -> None:
        """Width plus 2/alpha times the miss distance."""
        self.assertAlmostEqual(interval_score(1.0, 3.0, 2.0, 0.2), 2.0)
        self.assertAlmostEqual(interval_score(1.0, 3.0, 0.0, 0.2), 12.0)
        self.assertAlmostEqual(interval_score(1.0, 3.0, 4.5, 0.2), 17.0)
        for y in (0.0, 2.0, 4.5):
            with self.subTest(y=y):
                pinball = pinball_loss(1.0, y, 0.1) + pinball_loss(3.0, y, 0.9)
                self.assertAlmostEqual(0.1 * interval_score(1.0, 3.0, y, 0.2), pinball)
        with self.assertRaises(ValueError):
            interval_score(3.0, 1.0, 2.0, 0.2)

    def test_weighted_interval_score_hand_example(self) -> None:
        """WIS = 1.42 by hand and equals twice the mean pinball loss."""
        result = weighted_interval_score(10.0, 8.0, [(0.5, 6.0, 9.0), (0.2, 4.0, 12.0)])
        self.assertAlmostEqual(result.score, 1.42, places=14)
        self.assertAlmostEqual(result.dispersion, 0.62, places=14)
        self.assertAlmostEqual(result.underprediction, 0.8, places=14)
        self.assertEqual(result.overprediction, 0.0)
        quantiles = {0.1: 4.0, 0.25: 6.0, 0.5: 8.0, 0.75: 9.0, 0.9: 12.0}
        mean_pinball = math.fsum(
            pinball_loss(q, 10.0, tau) for tau, q in quantiles.items()
        ) / len(quantiles)
        self.assertAlmostEqual(result.score, 2.0 * mean_pinball, places=14)

    def test_log_dawid_sebastiani_energy_and_variogram(self) -> None:
        """Gaussian log score, DSS, 1-D energy score, and a variogram example."""
        self.assertAlmostEqual(
            log_score(Normal(0.0, 1.0), 1.0), 0.5 * math.log(2 * math.pi) + 0.5
        )
        self.assertAlmostEqual(log_score(Poisson(2.0), 0.0), 2.0)
        self.assertAlmostEqual(dawid_sebastiani_score(3.0, 1.0, 4.0), 1.0 + math.log(4))
        members = [0.3, -1.2, 2.5, 0.9]
        self.assertAlmostEqual(
            energy_score([[x] for x in members], [0.7]), crps_ensemble(members, 0.7)
        )
        self.assertAlmostEqual(
            energy_score([[x] for x in members], [0.7], fair=True),
            crps_ensemble(members, 0.7, fair=True),
        )
        # d = 2, p = 1: (|0 - 1| - mean(|0 - 2|, |1 - 1|))^2 = (1 - 1)^2 = 0.
        self.assertEqual(
            variogram_score([[0.0, 2.0], [1.0, 1.0]], [0.0, 1.0], order=1.0), 0.0
        )
        self.assertAlmostEqual(
            variogram_score([[0.0, 2.0], [1.0, 1.0]], [0.0, 3.0], order=1.0), 4.0
        )


class BrierTests(unittest.TestCase):
    """Brier score and its decompositions."""

    def test_hand_example(self) -> None:
        """BS = 0.19 = 0.065 - 0.0625 + 0.1875 for four forecasts."""
        result = brier_decomposition([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 1])
        self.assertAlmostEqual(result.score, 0.19, places=15)
        self.assertAlmostEqual(result.reliability, 0.065, places=15)
        self.assertAlmostEqual(result.resolution, 0.0625, places=15)
        self.assertAlmostEqual(result.uncertainty, 0.1875, places=15)

    def test_identity_exact_on_binned_and_extended_on_raw_forecasts(self) -> None:
        """REL - RES + UNC is exact for binned values; WBV - WBC closes raw."""
        generator = random.Random(5)
        raw = [generator.random() for _ in range(2000)]
        outcomes = [generator.random() < p**1.3 for p in raw]
        binned = [(math.floor(p * 10) + 0.5) / 10 for p in raw]
        grouped = brier_decomposition(binned, outcomes)
        self.assertEqual(grouped.within_bin_variance, 0.0)
        self.assertLess(
            abs(
                grouped.score
                - (grouped.reliability - grouped.resolution + grouped.uncertainty)
            ),
            1e-15,
        )
        self.assertAlmostEqual(grouped.score, brier_score(binned, outcomes), places=15)
        ten = brier_decomposition(raw, outcomes, bins=10)
        self.assertLess(abs(ten.identity_residual), 1e-15)
        three_term = ten.reliability - ten.resolution + ten.uncertainty
        self.assertGreater(abs(ten.score - three_term), 1e-4)
        self.assertAlmostEqual(
            ten.score - three_term,
            ten.within_bin_variance - ten.within_bin_covariance,
            places=15,
        )
        self.assertEqual(sum(ten.counts), 2000)


class CalibrationTests(unittest.TestCase):
    """PIT, coverage, reliability diagrams, and sharpness."""

    @classmethod
    def setUpClass(cls) -> None:
        """Draw one seeded set of outcomes from N(m_i, 1) with varying m_i."""
        generator = random.Random(42)
        cls.means = [generator.uniform(-3.0, 3.0) for _ in range(1000)]
        cls.outcomes = [generator.gauss(m, 1.0) for m in cls.means]

    def test_calibrated_pit_is_uniform_and_overconfident_is_u_shaped(self) -> None:
        """Chi-square p not tiny when calibrated; U shape when sigma is halved."""
        calibrated = pit_values([Normal(m, 1.0) for m in self.means], self.outcomes)
        histogram = pit_histogram(calibrated, bins=10)
        self.assertGreater(histogram.chi_square.p_value, 0.01)
        self.assertLess(abs(histogram.edge_ratio - 1.0), 0.25)
        narrow = pit_values([Normal(m, 0.5) for m in self.means], self.outcomes)
        u_shape = pit_histogram(narrow, bins=10)
        self.assertLess(u_shape.chi_square.p_value, 1e-12)
        self.assertGreater(u_shape.edge_ratio, 2.0)
        self.assertEqual(u_shape.sample_size, 1000)

    def test_ensemble_pit_with_ties_is_uniform_when_exchangeable(self) -> None:
        """Integer ensembles with ties: randomized PIT is uniform; narrow is U."""
        generator = random.Random(9)
        ensembles = [[generator.randint(0, 4) for _ in range(9)] for _ in range(800)]
        outcomes = [generator.randint(0, 4) for _ in range(800)]
        values = ensemble_pit(ensembles, outcomes, seed=3)
        self.assertEqual(values, ensemble_pit(ensembles, outcomes, seed=3))
        self.assertGreater(pit_histogram(values).chi_square.p_value, 0.01)
        self.assertGreater(
            kolmogorov_smirnov_test(values, Uniform(0.0, 1.0)).p_value, 0.01
        )
        narrow = [[generator.gauss(0.0, 0.3) for _ in range(19)] for _ in range(800)]
        wide_truth = [generator.gauss(0.0, 1.0) for _ in range(800)]
        histogram = pit_histogram(ensemble_pit(narrow, wide_truth, seed=4))
        self.assertGreater(histogram.edge_ratio, 3.0)

    def test_coverage_levels_and_intervals(self) -> None:
        """Calibrated coverage is consistent at every level; narrow is not."""
        calibrated = pit_values([Normal(m, 1.0) for m in self.means], self.outcomes)
        for method in ("clopper-pearson", "wilson"):
            reports = pit_coverage(calibrated, method=method)
            self.assertEqual([r.nominal for r in reports], [0.5, 0.8, 0.9, 0.95])
            self.assertTrue(all(r.consistent for r in reports), reports)
        narrow = pit_values([Normal(m, 0.5) for m in self.means], self.outcomes)
        for report in pit_coverage(narrow):
            self.assertFalse(report.consistent)
            self.assertLess(report.rate, report.nominal)
        explicit = interval_coverage(
            [0.0, 0.0, 0.0, 0.0], [1.0] * 4, [0.5, 1.0, 2.0, -1.0], nominal=0.5
        )
        self.assertEqual((explicit.covered, explicit.size), (2, 4))
        self.assertTrue(explicit.consistent)

    def test_reliability_diagram_and_bands(self) -> None:
        """Reliable forecasts stay inside the bands; squared ones do not."""
        generator = random.Random(17)
        forecasts = [generator.random() for _ in range(3000)]
        reliable = [generator.random() < p for p in forecasts]
        diagram = reliability_diagram(
            forecasts, reliable, bins=10, consistency_level=0.99, resamples=300, seed=1
        )
        self.assertEqual(sum(b.count for b in diagram.bins), 3000)
        self.assertLessEqual(len(diagram.outside_bands), 1)
        skewed = [generator.random() < p * p for p in forecasts]
        biased = reliability_diagram(
            forecasts, skewed, bins=10, consistency_level=0.99, resamples=300, seed=1
        )
        self.assertGreaterEqual(len(biased.outside_bands), 5)
        plain = reliability_diagram(forecasts, reliable, bins=[0.0, 0.5, 1.0])
        self.assertTrue(math.isnan(plain.bins[0].consistency_lower))
        self.assertEqual(len(plain.bins), 2)

    def test_sharpness_for_laws_and_ensembles(self) -> None:
        """Normal widths are 2 z sigma; ensemble widths use type-7 quantiles."""
        summary = sharpness_summary([Normal(0.0, 1.0), Normal(5.0, 3.0)], levels=(0.9,))
        z = Normal(0.0, 1.0).quantile(0.95)
        self.assertAlmostEqual(summary.mean_widths[0], 2 * z * 2.0, places=12)
        self.assertAlmostEqual(summary.mean_standard_deviation, 2.0, places=12)
        ensemble = sharpness_summary([[0.0, 1.0, 2.0, 3.0, 4.0]], levels=(0.5,))
        self.assertAlmostEqual(ensemble.mean_widths[0], 2.0)
        self.assertAlmostEqual(ensemble.mean_standard_deviation, math.sqrt(2.5))


class ShiftTests(unittest.TestCase):
    """Two-sample tests and Wasserstein distances."""

    def test_permutation_p_values_are_uniform_under_the_null(self) -> None:
        """Energy and MMD null p-values pass a KS uniformity check."""
        generator = random.Random(7)
        energy: list[float] = []
        mmd: list[float] = []
        for replicate in range(60):
            x = [generator.gauss(0.0, 1.0) for _ in range(20)]
            y = [generator.gauss(0.0, 1.0) for _ in range(20)]
            energy.append(
                energy_distance_test(x, y, permutations=99, seed=replicate).p_value
            )
            if replicate < 40:
                mmd.append(mmd_test(x, y, permutations=99, seed=replicate).p_value)
        for values in (energy, mmd):
            mean = math.fsum(values) / len(values)
            self.assertLess(abs(mean - 0.5), 0.12)
            ks = kolmogorov_smirnov_test(values, Uniform(0.0, 1.0))
            self.assertGreater(ks.p_value, 0.001)

    def test_shift_gives_small_p_values(self) -> None:
        """A unit mean shift (n = 50) and a 2-D scale change are detected."""
        generator = random.Random(8)
        x = [generator.gauss(0.0, 1.0) for _ in range(50)]
        y = [generator.gauss(1.0, 1.0) for _ in range(50)]
        self.assertLessEqual(
            energy_distance_test(x, y, permutations=199, seed=1).p_value, 0.01
        )
        self.assertLessEqual(mmd_test(x, y, permutations=199, seed=1).p_value, 0.01)
        self.assertLess(kolmogorov_smirnov_two_sample_test(x, y).p_value, 0.01)
        a = [(generator.gauss(0, 1), generator.gauss(0, 1)) for _ in range(40)]
        b = [(generator.gauss(0, 3), generator.gauss(0, 3)) for _ in range(40)]
        self.assertLessEqual(
            energy_distance_test(a, b, permutations=199, seed=2).p_value, 0.01
        )
        self.assertGreater(mmd_squared(a, b), 0.0)

    def test_energy_distance_paths_agree(self) -> None:
        """The sorted 1-D path equals the general Euclidean path."""
        generator = random.Random(1)
        x = [generator.gauss(0.0, 1.0) for _ in range(30)]
        y = [generator.gauss(0.5, 2.0) for _ in range(25)]
        self.assertAlmostEqual(
            energy_distance(x, y),
            energy_distance([(v, 0.0) for v in x], [(v, 0.0) for v in y]),
            places=12,
        )
        self.assertLess(energy_distance(x, x), 1e-14)

    def test_wasserstein_hand_formulas(self) -> None:
        """Exact sample W_p, the area metric, and closed forms for laws."""
        self.assertAlmostEqual(wasserstein_distance([0.0, 1.0], [2.0, 3.0]), 2.0)
        self.assertAlmostEqual(wasserstein_distance([0.0, 2.0], [1.0]), 1.0)
        self.assertAlmostEqual(wasserstein_distance([0.0, 2.0], [1.0], order=2), 1.0)
        self.assertAlmostEqual(wasserstein_distance([0.0, 1.0, 2.0], [0.0, 3.0]), 5 / 6)
        generator = random.Random(3)
        x = [generator.gauss(0, 1) for _ in range(17)]
        y = [generator.expovariate(1.0) for _ in range(23)]
        self.assertAlmostEqual(wasserstein_distance(x, y), area_metric(x, y), places=12)
        w2 = distribution_wasserstein_distance(Normal(0, 1), Normal(1, 2), order=2)
        self.assertAlmostEqual(w2.distance, math.sqrt(2.0), places=10)
        self.assertAlmostEqual(normal_wasserstein_distance(0, 1, 1, 2), math.sqrt(2.0))
        # E|1 + Z| = 2 phi(1) + (2 Phi(1) - 1).
        phi = math.exp(-0.5) / math.sqrt(2 * math.pi)
        w1_exact = 2 * phi + math.erf(1 / math.sqrt(2))
        w1 = distribution_wasserstein_distance(Normal(0, 1), Normal(1, 2))
        self.assertAlmostEqual(w1.distance, w1_exact, delta=1e-6)
        # int_0^1 (u + ln(1 - u))^2 du = 1/3 - 3/2 + 2 = 5/6.
        w2_ue = distribution_wasserstein_distance(
            Uniform(0, 1), Exponential(1), order=2
        )
        self.assertAlmostEqual(w2_ue.distance**2, 5 / 6, places=10)

    def test_two_sample_ks_exact_values(self) -> None:
        """Separated samples, brute-force enumeration with ties, asymptotics."""
        separated = kolmogorov_smirnov_two_sample_test([1, 2, 3], [4, 5, 6])
        self.assertEqual(separated.statistic, 1.0)
        self.assertAlmostEqual(separated.p_value, 0.1, places=15)
        a, b = [0, 1, 1, 2, 3], [1, 2, 2, 4, 4, 5]
        observed = kolmogorov_smirnov_two_sample_test(a, b)
        pooled = a + b

        def statistic(first: Sequence[float], second: Sequence[float]) -> float:
            """Return the brute-force two-sample KS distance."""
            return max(
                abs(
                    sum(v <= z for v in first) / len(first)
                    - sum(v <= z for v in second) / len(second)
                )
                for z in pooled
            )

        hits = total = 0
        for chosen in itertools.combinations(range(len(pooled)), len(a)):
            first = [pooled[i] for i in chosen]
            second = [pooled[i] for i in range(len(pooled)) if i not in chosen]
            total += 1
            hits += statistic(first, second) >= observed.statistic - 1e-12
        self.assertAlmostEqual(observed.p_value, hits / total, places=15)
        generator = random.Random(4)
        x = [generator.gauss(0, 1) for _ in range(300)]
        y = [generator.gauss(0.2, 1) for _ in range(400)]
        exact = kolmogorov_smirnov_two_sample_test(x, y)
        asymptotic = kolmogorov_smirnov_two_sample_test(x, y, exact=False)
        self.assertTrue(exact.exact)
        self.assertFalse(asymptotic.exact)
        self.assertAlmostEqual(
            exact.p_value, asymptotic.p_value, delta=0.2 * exact.p_value
        )


class AssessmentIntegrationTests(unittest.TestCase):
    """The validation assessment reports the mean ensemble CRPS."""

    def test_experiment_assessment_exposes_crps(self) -> None:
        """``ExperimentAssessment.crps`` averages the per-observation CRPS."""
        experiments = [
            ValidationExperiment(f"e{k}", [float(k), k + 0.5], provenance="test")
            for k in range(4)
        ]
        samples = [[k + 0.1 * j for j in range(-10, 11)] for k in range(4)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SmallExpectedCountWarning)
            result = assess_predictions(
                "y",
                experiments,
                samples,
                tolerance=1.0,
                seed=1,
                include_measurement_noise=False,
            )
        item = result.experiments[2]
        expected = (crps_ensemble(samples[2], 2.0) + crps_ensemble(samples[2], 2.5)) / 2
        self.assertAlmostEqual(item.crps, expected, places=14)
        self.assertIn("CRPS=", result.summary())


if __name__ == "__main__":
    unittest.main()
