"""Tests for cross-validated and multivariate Gaussian kernel density estimates.

Oracles: R 4.6.2 base ``stats`` (``bw.nrd0``, ``bw.nrd``, ``bw.ucv``,
``bw.bcv``, and the compiled score routines ``C_bw_ucv`` / ``C_bw_bcv``), a
direct R implementation of the leave-one-out likelihood maximized by
``optimize``, closed forms for two and three points, the divergence of the
in-sample likelihood, exact integrated squared errors of Gaussian mixtures in
a seeded simulation, and a tensor Simpson integral of the multivariate
density. Tests do not call R; the constants below were computed offline.
"""

from __future__ import annotations

import math
import unittest
import warnings

from probstat.distributions import Mixture, Normal
from probstat.foundations import RandomStream
from probstat.numerics import composite_simpson
from probstat.statistics import (
    LIEBLEIN_ZELEN_BALL_BEARINGS,
    NEWCOMB_PASSAGE_TIMES,
    gaussian_kde,
    gaussian_kde_multivariate,
    kde_bandwidth,
    kde_bcv,
    kde_loo_log_likelihood,
    kde_ucv,
    multivariate_kde_factor,
    multivariate_kde_loo_log_likelihood,
    oversmoothed_bandwidth,
    select_kde_bandwidth,
)
from probstat.statistics.bandwidth_selection import (
    _bcv_terms,
    _loo_terms,
    _pair_squares,
    _row_squares,
    _ucv_terms,
)

BIMODAL = (
    -2.734, -1.773, -1.403, -2.308, -2.797, -2.039, -1.713, -1.341, -2.73,
    -2.776, -1.576, -1.716, -1.08, -1.411, -2.061, -2.165, -0.458, -2.173,
    -2.471, -2.683, -1.943, -1.913, -2.207, -2.037, -1.836, -1.585, -1.342,
    -1.874, -3.074, -1.579, 0.199, 1.71, 1.25, 1.068, 2.37, 1.559, 1.253,
    3.122, 1.055, 0.363, 0.263, 1.026, 2.093, 2.267, 2.525, 3.167, 3.051,
    2.101, 2.724, 2.12, 1.024, 2.283, 1.596, 1.354, 3.355, 2.245, 1.108,
    1.478, 3.672, 1.324,
)  # fmt: skip
SLEEP_DRUG_2 = (1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4)
DATASETS = {
    "bimodal": BIMODAL,
    "bearings": LIEBLEIN_ZELEN_BALL_BEARINGS.columns[0].values,
    "sleep2": SLEEP_DRUG_2,
    "newcomb": NEWCOMB_PASSAGE_TIMES.columns[0].values,
}
# R 4.6.2 (2026-10-29 ucrt), packageVersion("stats") == "4.6.2":
#   nrd0, nrd: bw.nrd0(x), bw.nrd(x)
#   ucv, bcv:  bw.ucv(x), bw.bcv(x)                     (defaults: nb = 1000)
#   ucv_fine, bcv_fine: bw.ucv(x, nb = 10000000L, tol = 1e-10 * hmax), same
#              for bw.bcv, with hmax = 1.144 * sqrt(var(x)) * length(x)^(-1/5)
#   loo, loo_value: optimize(function(h) loo(x, h), c(0.1, 2) * hmax,
#              maximum = TRUE, tol = 1e-12 * hmax) with
#              loo <- function(x, h) {k <- dnorm(outer(x, x, "-"), sd = h);
#              diag(k) <- 0; sum(log(rowSums(k) / (length(x) - 1)))}
R_VALUES = {
    "bimodal": {
        "nrd0": 0.81590599404106268,
        "nrd": 0.96095594853725175,
        "ucv": 0.3024693994624072,
        "bcv": 0.53898394089653767,
        "ucv_fine": 0.31054102616190477,
        "bcv_fine": 0.53691699983080021,
        "loo": 0.3656172897553267,
        "loo_value": -112.67213594256175,
    },
    "bearings": {
        "nrd0": 17.521359445794392,
        "nrd": 20.636267791713394,
        "ucv": 21.672881317965249,
        "bcv": 22.827513889660679,
        "ucv_fine": 21.754610487348852,
        "bcv_fine": 22.907663610554575,
        "loo": 25.18330654838044,
        "loo_value": -117.32757573742927,
    },
    "sleep2": {
        "nrd0": 1.1370001907944722,
        "nrd": 1.3391335580468229,
        "ucv": 1.4363287406442846,
        "bcv": 1.4394953839045534,
        "ucv_fine": 1.4452535445298862,
        "bcv_fine": 1.4452535368931796,
        "loo": 1.0419263263612868,
        "loo_value": -21.792382244464495,
    },
    "newcomb": {
        "nrd0": 1.96124882676766,
        "nrd": 2.3099152848596884,
        "ucv": 2.7307033541976127,
        "bcv": 2.5962244826147587,
        "ucv_fine": 2.7182490665742547,
        "bcv_fine": 2.6015770926406274,
        "loo": 8.0835860407721931,
        "loo_value": -243.1662689072227,
    },
}


def normal_density(x: float, sd: float) -> float:
    """Return the centred normal density ``phi(x / sd) / sd``.

    References:
        [EQ-STAT-KDE-001]. No new equation.

    """
    return math.exp(-0.5 * (x / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))


class RuleOfThumbTests(unittest.TestCase):
    """Rules of thumb against R's ``bw.nrd0`` and ``bw.nrd``."""

    def test_silverman_and_nrd_match_r(self) -> None:
        """``silverman`` is ``bw.nrd0`` and ``nrd`` is ``bw.nrd`` to rounding."""
        for name, values in DATASETS.items():
            with self.subTest(dataset=name):
                expected = R_VALUES[name]
                self.assertAlmostEqual(
                    kde_bandwidth(values) / expected["nrd0"], 1.0, places=14
                )
                self.assertAlmostEqual(
                    kde_bandwidth(values, rule="nrd") / expected["nrd"], 1.0, places=14
                )


class CrossValidationScoreTests(unittest.TestCase):
    """UCV and BCV scores and their minimizers against R."""

    def test_scores_match_the_compiled_r_routines(self) -> None:
        """Points ``(0, 1, 3)``: distances 1, 2, 3 as unit-width counts.

        R: ``.Call(stats:::C_bw_ucv, 3L, 1.0, c(0, 1, 1, 1), h)`` and the
        same with ``C_bw_bcv``, R 4.6.2.
        """
        expected = {
            0.7: (0.10503737471903921, 0.12565343776953253),
            1.3: (0.00060257159771464177, 0.066554777082481642),
        }
        for h, (ucv, bcv) in expected.items():
            self.assertAlmostEqual(kde_ucv((0.0, 1.0, 3.0), h) / ucv, 1.0, places=13)
            self.assertAlmostEqual(kde_bcv((0.0, 1.0, 3.0), h) / bcv, 1.0, places=13)

    def test_minimizers_match_r_with_fine_binning(self) -> None:
        """Exact-distance minimizers agree with R at ``nb = 1e7`` to ``1e-5``.

        R bins each distance down to a multiple of ``1.01 range / nb``, which
        leaves a relative difference of about ``1e-6``. On ``sleep2`` both
        scores and on ``bearings`` the BCV score have their minimum on the
        upper end ``h_os``, as R finds within its tolerance.
        """
        boundary = {("sleep2", "ucv"), ("sleep2", "bcv"), ("bearings", "bcv")}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for name, values in DATASETS.items():
                for method in ("ucv", "bcv"):
                    with self.subTest(dataset=name, method=method):
                        result = select_kde_bandwidth(values, method=method)
                        fine = R_VALUES[name][f"{method}_fine"]
                        self.assertLess(abs(result.bandwidth / fine - 1.0), 1e-5)
                        self.assertEqual(result.at_boundary, (name, method) in boundary)

    def test_minimizers_lie_within_the_tolerance_of_r_defaults(self) -> None:
        """R's default search stops within ``tol = 0.1 lower = 0.01 h_os``.

        The binning with ``nb = 1000`` adds at most ``0.001 h_os``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for name, values in DATASETS.items():
                h_os = oversmoothed_bandwidth(values)
                for method in ("ucv", "bcv"):
                    with self.subTest(dataset=name, method=method):
                        result = select_kde_bandwidth(values, method=method)
                        self.assertLess(
                            abs(result.bandwidth - R_VALUES[name][method]),
                            0.011 * h_os,
                        )

    def test_analytic_derivatives_match_finite_differences(self) -> None:
        """The ``log h`` derivatives used by Brent's method match differences."""
        values = (0.1, 0.5, 1.7, 2.2, 3.9, 4.0, 6.5)
        n = len(values)
        pairs = _pair_squares(values)
        rows = _row_squares(values)
        criteria = {
            "ucv": lambda h: _ucv_terms(pairs, n, h),
            "bcv": lambda h: _bcv_terms(pairs, n, h),
            "loo": lambda h: _loo_terms(rows, h, 1, leave_one_out=True),
        }
        step = 1e-5
        for h in (0.3, 1.1):
            for name, criterion in criteria.items():
                with self.subTest(h=h, criterion=name):
                    numeric = (
                        criterion(h * math.exp(step))[0]
                        - criterion(h * math.exp(-step))[0]
                    ) / (2.0 * step)
                    analytic = criterion(h)[1]
                    self.assertAlmostEqual(
                        analytic, numeric, delta=1e-7 * max(1.0, abs(analytic))
                    )
        self.assertEqual(criteria["ucv"](0.3)[0], kde_ucv(values, 0.3))


class LikelihoodCrossValidationTests(unittest.TestCase):
    """Leave-one-out likelihood: closed forms, divergence, and R."""

    def test_two_points_select_their_distance(self) -> None:
        """``L(h) = 2 log phi_h(d)`` is maximal at ``h = |d|``."""
        d = 1.5
        selection = select_kde_bandwidth((0.0, d), lower=0.1, upper=10.0)
        self.assertAlmostEqual(selection.bandwidth, d, places=11)
        self.assertAlmostEqual(
            selection.criterion, 2.0 * math.log(normal_density(d, d)), places=12
        )

    def test_three_points_by_hand(self) -> None:
        """``sum_i log((1/2) sum_(j != i) phi_h(x_i - x_j))`` for three points."""
        points = (-1.2, 0.3, 2.0)
        for h in (0.5, 1.0):
            expected = math.fsum(
                math.log(
                    0.5
                    * math.fsum(normal_density(x - y, h) for y in points if y is not x)
                )
                for x in points
            )
            self.assertAlmostEqual(
                kde_loo_log_likelihood(points, h), expected, places=12
            )

    def test_in_sample_likelihood_diverges_as_h_vanishes(self) -> None:
        """Without leaving out, ``L(h) >= n log(1/(n h sqrt(2 pi)))`` grows to ``inf``.

        The leave-one-out likelihood of distinct points instead falls to
        ``-inf`` as ``h -> 0``.
        """
        values = BIMODAL[:12]
        n = len(values)
        previous = -math.inf
        for h in (1e-1, 1e-2, 1e-4, 1e-8):
            in_sample = kde_loo_log_likelihood(values, h, leave_one_out=False)
            bound = n * math.log(1.0 / (n * h * math.sqrt(2.0 * math.pi)))
            self.assertGreaterEqual(in_sample, bound - 1e-9)
            self.assertGreater(in_sample, previous)
            previous = in_sample
        self.assertLess(kde_loo_log_likelihood(values, 1e-4), -1e6)

    def test_maximizers_match_r(self) -> None:
        """Argmax to ``1e-6`` (golden section in R) and maximum to ``1e-12``."""
        for name, values in DATASETS.items():
            with self.subTest(dataset=name):
                h_os = oversmoothed_bandwidth(values)
                result = select_kde_bandwidth(
                    values, method="likelihood", lower=0.1 * h_os, upper=2.0 * h_os
                )
                expected = R_VALUES[name]
                self.assertLess(abs(result.bandwidth / expected["loo"] - 1.0), 1e-6)
                self.assertAlmostEqual(
                    result.criterion / expected["loo_value"], 1.0, places=12
                )
                self.assertFalse(result.at_boundary)

    def test_value_where_r_underflows(self) -> None:
        """At ``h = 1`` R gives ``-Inf`` for the bearings; the shifted sum does not.

        R: ``loo(sleep2, 1)`` is ``-21.797763867277972``; for the bearings the
        nearest-neighbour term alone is ``log(phi(0.24) / 22) = -4.0372...``.
        """
        self.assertAlmostEqual(
            kde_loo_log_likelihood(SLEEP_DRUG_2, 1.0), -21.797763867277972, places=12
        )
        value = kde_loo_log_likelihood(
            LIEBLEIN_ZELEN_BALL_BEARINGS.columns[0].values, 1.0
        )
        self.assertTrue(math.isfinite(value))

    def test_repeated_observations_push_the_optimum_to_the_lower_end(self) -> None:
        """Every value repeated: each term contains ``phi_h(0)``, so ``L -> inf``.

        A single tie among distinct values does not suffice: the terms of the
        unrepeated points fall like ``-d^2 / (2 h^2)``.
        """
        with self.assertWarns(RuntimeWarning):
            result = select_kde_bandwidth(
                (0.0, 0.0, 1.0, 1.0, 2.5, 2.5), lower=1e-6, upper=10.0
            )
        self.assertTrue(result.at_boundary)
        self.assertAlmostEqual(result.bandwidth, 1e-6, delta=1e-20)
        single = select_kde_bandwidth((0.0, 0.0, 1.0, 2.5, 3.0), lower=1e-6, upper=10.0)
        self.assertFalse(single.at_boundary)

    def test_rules_reach_gaussian_kde(self) -> None:
        """``kde_bandwidth`` and ``gaussian_kde`` accept the selection rules."""
        for method in ("likelihood", "ucv", "bcv"):
            with self.subTest(method=method):
                expected = select_kde_bandwidth(BIMODAL, method=method).bandwidth
                self.assertEqual(kde_bandwidth(BIMODAL, rule=method), expected)
                self.assertEqual(
                    gaussian_kde(BIMODAL, bandwidth=method).bandwidth, expected
                )
        with self.assertRaises(ValueError):
            select_kde_bandwidth(BIMODAL, method="plugin")
        with self.assertRaises(ValueError):
            select_kde_bandwidth((1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            select_kde_bandwidth(BIMODAL, lower=2.0, upper=1.0)


class SimulationTests(unittest.TestCase):
    """Seeded mean integrated squared error on a bimodal law."""

    def test_cross_validation_beats_silverman_on_a_bimodal_law(self) -> None:
        """``0.5 N(-2, 0.5^2) + 0.5 N(2, 0.5^2)``, ``n = 100``, 12 replications.

        The integrated squared error of a Gaussian KDE against a Gaussian
        mixture is exact: ``int f_h^2 - 2 int f_h f + int f^2`` with every
        term a sum of normal densities. The mean ISE reduction relative to
        Silverman's rule must exceed four standard errors of the paired
        differences.
        """
        weights, means, sds = (0.5, 0.5), (-2.0, 2.0), (0.5, 0.5)

        def ise(sample: tuple[float, ...], h: float) -> float:
            """Return the exact integrated squared error.

            References:
                [EQ-STAT-KDE-001]. No new equation.

            """
            n = len(sample)
            kernel = math.fsum(
                normal_density(x - y, math.sqrt(2.0) * h)
                for x in sample
                for y in sample
            ) / (n * n)
            cross = (
                math.fsum(
                    w * normal_density(x - m, math.hypot(h, s))
                    for x in sample
                    for w, m, s in zip(weights, means, sds, strict=True)
                )
                / n
            )
            law = math.fsum(
                w1 * w2 * normal_density(m1 - m2, math.hypot(s1, s2))
                for w1, m1, s1 in zip(weights, means, sds, strict=True)
                for w2, m2, s2 in zip(weights, means, sds, strict=True)
            )
            return kernel - 2.0 * cross + law

        mixture = Mixture((Normal(-2.0, 0.5), Normal(2.0, 0.5)), (0.5, 0.5))
        differences: dict[str, list[float]] = {"likelihood": [], "ucv": []}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for stream in RandomStream(20261116).spawn(12):
                sample = mixture.sample(100, seed=stream)
                silverman = ise(sample, kde_bandwidth(sample))
                for method, store in differences.items():
                    selected = select_kde_bandwidth(sample, method=method).bandwidth
                    store.append(silverman - ise(sample, selected))
        for method, values in differences.items():
            with self.subTest(method=method):
                count = len(values)
                centre = math.fsum(values) / count
                spread = math.sqrt(
                    math.fsum((v - centre) ** 2 for v in values) / (count - 1)
                )
                self.assertGreater(centre, 4.0 * spread / math.sqrt(count))


class MultivariateTests(unittest.TestCase):
    """Covariance-shaped multivariate KDE."""

    POINTS = (
        (0.0, 0.1),
        (1.0, 0.7),
        (2.0, 2.4),
        (0.5, -0.3),
        (1.7, 1.1),
        (3.0, 2.0),
    )

    def test_rule_of_thumb_factors(self) -> None:
        """Silverman ``(4/(N(d+2)))^(1/(d+4))`` and Scott ``N^(-1/(d+4))``."""
        self.assertAlmostEqual(
            multivariate_kde_factor(100, 3), (4.0 / 500.0) ** (1.0 / 7.0), places=15
        )
        self.assertAlmostEqual(
            multivariate_kde_factor(100, 3, rule="scott"),
            100.0 ** (-1.0 / 7.0),
            places=15,
        )
        self.assertAlmostEqual(
            multivariate_kde_factor(50, 1), (4.0 / 3.0) ** 0.2 * 50.0**-0.2, places=15
        )

    def test_one_dimension_reduces_to_the_univariate_estimate(self) -> None:
        """For ``d = 1`` factor ``h`` is the univariate bandwidth ``h s``."""
        values = (0.1, 0.5, 1.7, 2.2, 3.9, 4.0, 6.5)
        centre = math.fsum(values) / len(values)
        sd = math.sqrt(math.fsum((v - centre) ** 2 for v in values) / (len(values) - 1))
        factor = 0.7
        estimate = gaussian_kde_multivariate([(v,) for v in values], bandwidth=factor)
        reference = gaussian_kde(values, bandwidth=factor * sd)
        for x in (-1.0, 2.0, 5.5):
            self.assertAlmostEqual(
                estimate.pdf((x,)) / reference.pdf(x), 1.0, places=13
            )
        self.assertAlmostEqual(
            multivariate_kde_loo_log_likelihood([(v,) for v in values], factor),
            kde_loo_log_likelihood(values, factor * sd),
            places=12,
        )

    def test_three_points_by_hand(self) -> None:
        """Explicit 2x2 inverse and determinant of ``h^2 R`` for three points."""
        points = ((0.0, 0.0), (1.0, 0.5), (0.4, 2.0))
        n = 3
        mx = math.fsum(p[0] for p in points) / n
        my = math.fsum(p[1] for p in points) / n
        sxx = math.fsum((p[0] - mx) ** 2 for p in points) / (n - 1)
        syy = math.fsum((p[1] - my) ** 2 for p in points) / (n - 1)
        sxy = math.fsum((p[0] - mx) * (p[1] - my) for p in points) / (n - 1)
        h = 0.8
        a, b, c = h * h * sxx, h * h * sxy, h * h * syy
        det = a * c - b * b
        x = (0.5, 0.9)
        expected = (
            math.fsum(
                math.exp(
                    -0.5
                    * (
                        c * (x[0] - p[0]) ** 2
                        - 2.0 * b * (x[0] - p[0]) * (x[1] - p[1])
                        + a * (x[1] - p[1]) ** 2
                    )
                    / det
                )
                / (2.0 * math.pi * math.sqrt(det))
                for p in points
            )
            / n
        )
        estimate = gaussian_kde_multivariate(points, bandwidth=h)
        self.assertAlmostEqual(estimate.pdf(x) / expected, 1.0, places=13)
        self.assertAlmostEqual(estimate.kernel_covariance[0][1], b, places=15)

    def test_density_integrates_to_one(self) -> None:
        """Tensor Simpson (160 x 160 intervals) gives 1 to ``1e-7``."""
        estimate = gaussian_kde_multivariate(self.POINTS)
        widths = [
            math.sqrt(estimate.covariance[i][i]) * (1.0 + estimate.factor)
            for i in range(2)
        ]
        low = [min(p[i] for p in self.POINTS) - 9.0 * widths[i] for i in range(2)]
        high = [max(p[i] for p in self.POINTS) + 9.0 * widths[i] for i in range(2)]

        def inner(y: float) -> float:
            """Integrate over the first coordinate at fixed ``y``.

            References:
                [EQ-STAT-KDE-006]. No new equation.

            """
            return composite_simpson(
                lambda x: estimate.pdf((x, y)), low[0], high[0], intervals=160
            ).value

        total = composite_simpson(inner, low[1], high[1], intervals=160).value
        self.assertAlmostEqual(total, 1.0, places=7)

    def test_draws_match_the_mixture_moments(self) -> None:
        """4000 draws: mean within 4 standard errors, variances within 10%."""
        estimate = gaussian_kde_multivariate(self.POINTS)
        draws = estimate.draw(4000, seed=20261116)
        covariance = estimate.mixture_covariance
        for i in range(2):
            centre = math.fsum(d[i] for d in draws) / len(draws)
            spread = math.fsum((d[i] - centre) ** 2 for d in draws) / (len(draws) - 1)
            self.assertLess(
                abs(centre - estimate.mean[i]),
                4.0 * math.sqrt(covariance[i][i] / len(draws)),
            )
            self.assertLess(abs(spread / covariance[i][i] - 1.0), 0.10)

    def test_likelihood_factor_maximizes_the_criterion(self) -> None:
        """The selected factor beats Silverman's and reports the maximum."""
        normal_a, normal_b = Normal(-2.0, 0.4), Normal(2.0, 0.4)
        stream = RandomStream(7)
        xs = normal_a.sample(40, seed=stream.child("a")) + normal_b.sample(
            40, seed=stream.child("b")
        )
        ys = Normal(0.0, 1.0).sample(80, seed=stream.child("y"))
        points = list(zip(xs, ys, strict=True))
        estimate = gaussian_kde_multivariate(points, bandwidth="likelihood")
        assert estimate.selection is not None
        self.assertFalse(estimate.selection.at_boundary)
        silverman = multivariate_kde_factor(len(points), 2)
        self.assertLess(estimate.factor, silverman)
        self.assertAlmostEqual(
            estimate.selection.criterion,
            multivariate_kde_loo_log_likelihood(points, estimate.factor),
            places=10,
        )
        self.assertGreater(
            estimate.selection.criterion,
            multivariate_kde_loo_log_likelihood(points, silverman),
        )

    def test_invalid_samples(self) -> None:
        """Ragged, collinear, or too few points raise."""
        with self.assertRaises(ValueError):
            gaussian_kde_multivariate([(0.0, 1.0), (1.0,), (2.0, 0.0)])
        with self.assertRaisesRegex(ValueError, "singular"):
            gaussian_kde_multivariate([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
        with self.assertRaises(ValueError):
            gaussian_kde_multivariate([(0.0, 0.0), (1.0, 1.0)])
        with self.assertRaises(ValueError):
            gaussian_kde_multivariate(self.POINTS, bandwidth="plugin")


if __name__ == "__main__":
    unittest.main()
