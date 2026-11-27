"""Tests for probstat.surrogates.gp_bayes (Bayesian GP emulator extensions).

Oracles: the existing exact-gradient marginal likelihood of
``probstat.surrogates.gaussian_process``, dense ``probstat.linalg`` solves, the
univariate Student-t density, the analytic Ishigami indices, and scalar root
finding for the Laplace mode. Stochastic tests use fixed seeds.
"""

from __future__ import annotations

import math
import os
import unittest

from probstat.distributions import ChiSquare, FisherF
from probstat.foundations.random_streams import resolve_stream
from probstat.linalg import solve_spd
from probstat.modeling.benchmarks import ishigami
from probstat.surrogates import (
    fit_gaussian_process,
    gaussian_process_log_marginal_likelihood,
    kernel_value,
)
from probstat.surrogates.gp_bayes import (
    GPHyperPrior,
    _cholesky,
    _LogPosterior,
    box_cox,
    default_hyper_prior,
    fit_bayesian_gp,
    fit_laplace_gp_classifier,
    fit_student_t_process,
    fit_warped_gp,
    gp_predictive_moments,
    gp_sobol_posterior,
    inverse_box_cox,
    sobol_estimates,
    student_t_process_log_marginal_likelihood,
    validate_emulator_moments,
    validate_gp_emulator,
)

SLOW = os.environ.get("PROBSTAT_SLOW") == "1"


def _grid_1d(count: int, low: float = 0.0, high: float = 1.0) -> list[tuple[float]]:
    """Return equally spaced one-dimensional points."""
    return [(low + (high - low) * i / (count - 1),) for i in range(count)]


def _smooth(x: float) -> float:
    """Return a smooth test function on [0, 1]."""
    return math.sin(6.0 * x) + 0.5 * x


# ---------------------------------------------------------------------------
# 1. Fully Bayesian hyperparameters
# ---------------------------------------------------------------------------


class BayesianGPTests(unittest.TestCase):
    """Posterior sampling, mixture predictive and the small-n comparison."""

    def setUp(self) -> None:
        """Build a small deterministic data set."""
        self.points = _grid_1d(7)
        self.values = [_smooth(p[0]) for p in self.points]

    def test_log_posterior_matches_exact_marginal_likelihood(self) -> None:
        """Closed form first: likelihood plus prior equals the library value."""
        prior = default_hyper_prior(self.points)
        mean = sum(self.values) / 7
        sd = math.sqrt(sum((v - mean) ** 2 for v in self.values) / 6)
        targets = tuple((v - mean) / sd for v in self.values)
        target = _LogPosterior("matern52", tuple(self.points), targets, prior, 1e-8)
        for theta in ((0.2, -1.0, -5.0), (-0.5, -0.3, -3.0), (1.0, -2.0, -7.0)):
            exact, _ = gaussian_process_log_marginal_likelihood(
                self.points,
                targets,
                theta,
                kernel="matern52",
                nugget=1e-8,
                estimate_noise=True,
            )
            self.assertAlmostEqual(
                target(theta), exact + prior.log_density(theta), places=8
            )
        self.assertEqual(target((30.0, 0.0, 0.0)), -math.inf)

    def test_sharp_prior_recovers_plain_gp(self) -> None:
        """A near-degenerate prior makes the mixture the fixed-theta GP."""
        theta = (0.1, -1.2, -6.0)
        prior = GPHyperPrior(theta[0], 1e-5, (theta[1],), (1e-5,), theta[2], 1e-5)
        bayes = fit_bayesian_gp(
            self.points, self.values, prior=prior, draws=20, warmup=5, seed=3
        )
        gp = fit_gaussian_process(
            self.points,
            self.values,
            optimize=False,
            estimate_noise=True,
            initial_log_parameters=theta,
            nugget=1e-8,
            restarts=1,
        )
        for x in (0.13, 0.5, 0.91):
            mean, variance = gp.predict_with_variance((x,))
            pred = bayes.predict((x,))
            self.assertAlmostEqual(pred.mean, mean, delta=1e-3 * (1 + abs(mean)))
            self.assertAlmostEqual(
                pred.variance, variance, delta=1e-2 * variance + 1e-9
            )
            low, high = bayes.predictive_interval((x,), 0.95)
            half = 1.959963984540054 * math.sqrt(variance)
            self.assertAlmostEqual(0.5 * (low + high), mean, delta=1e-3)
            self.assertAlmostEqual(0.5 * (high - low), half, delta=0.02 * half)

    def test_total_variance_decomposition(self) -> None:
        """Mixture variance = mean of variances + variance of means."""
        bayes = fit_bayesian_gp(self.points, self.values, draws=40, warmup=20, seed=11)
        pred = bayes.predict((0.37,), include_noise=True)
        means = pred.component_means
        variances = pred.component_variances
        count = len(means)
        self.assertEqual(count, 40)
        grand = sum(means) / count
        second = sum(v + m * m for m, v in zip(means, variances, strict=True)) / count
        self.assertAlmostEqual(pred.mean, grand, places=12)
        self.assertAlmostEqual(pred.variance, second - grand**2, places=9)
        self.assertGreater(pred.variance_of_means, 0.0)
        self.assertAlmostEqual(
            pred.variance, pred.mean_of_variances + pred.variance_of_means, places=12
        )
        # Noise-free prediction has strictly smaller variance.
        self.assertLess(bayes.predict((0.37,)).variance, pred.variance)

    def test_quantile_inverts_mixture_cdf(self) -> None:
        """The bisection quantile solves cdf(q) = p."""
        bayes = fit_bayesian_gp(self.points, self.values, draws=30, warmup=20, seed=5)
        for p in (0.05, 0.5, 0.93):
            q = bayes.predictive_quantile((0.4,), p)
            self.assertAlmostEqual(bayes.predictive_cdf((0.4,), q), p, places=9)
        low, high = bayes.predictive_interval((0.4,), 0.9)
        self.assertLess(low, bayes.predict((0.4,)).mean)
        self.assertGreater(high, bayes.predict((0.4,)).mean)

    def test_reproducible_and_validated(self) -> None:
        """Same seed gives the same draws; invalid input is rejected."""
        a = fit_bayesian_gp(self.points, self.values, draws=10, warmup=5, seed=9)
        b = fit_bayesian_gp(self.points, self.values, draws=10, warmup=5, seed=9)
        self.assertEqual(a.draws, b.draws)
        self.assertEqual(a.failed_factorizations, b.failed_factorizations)
        with self.assertRaises(ValueError):
            fit_bayesian_gp(self.points, [1.0] * 7, draws=5, warmup=2, seed=1)
        with self.assertRaises(ValueError):
            fit_bayesian_gp(self.points, self.values, kernel="rbf", seed=1)
        with self.assertRaises(ValueError):
            fit_bayesian_gp(
                self.points,
                self.values,
                prior=default_hyper_prior([(0.0, 0.0), (1.0, 1.0)]),
                seed=1,
            )
        with self.assertRaises(ValueError):
            fit_bayesian_gp(self.points, self.values, nugget=0.0, seed=1)
        with self.assertRaises(ValueError):
            GPHyperPrior(0.0, -1.0, (0.0,), (1.0,), 0.0, 1.0)
        with self.assertRaises(ValueError):
            default_hyper_prior([(0.0,), (0.0,)])

    def _coverage_study(self, replicates: int, draws: int, warmup: int) -> dict:
        """Return coverage of nominal 80 % intervals for ML and Bayes.

        Each replicate draws hyperparameters from the prior, simulates six noisy
        training points and five held-out noisy points from the GP, and counts
        how often each method's interval covers the held-out outputs.
        """
        stream = resolve_stream(2024)
        hits_bayes = hits_ml = total = 0
        for rep in range(replicates):
            child = stream.child("rep", rep)
            generator = child.generator()
            xs = [generator.random() for _ in range(11)]
            pts = tuple((x,) for x in xs)
            theta = default_hyper_prior(pts[:6]).sample(child.child("theta"))
            signal, scale, noise = (math.exp(v) for v in theta)
            cov = [
                [
                    kernel_value("matern52", pts[i], pts[j], (scale,), signal)
                    + (noise + 1e-8 if i == j else 0.0)
                    for j in range(i + 1)
                ]
                for i in range(11)
            ]
            factor = _cholesky(cov)
            z = [generator.gauss(0.0, 1.0) for _ in range(11)]
            y = [sum(factor[i][k] * z[k] for k in range(i + 1)) for i in range(11)]
            train, held = pts[:6], pts[6:]
            try:
                bayes = fit_bayesian_gp(
                    train,
                    y[:6],
                    prior=default_hyper_prior(train),
                    draws=draws,
                    warmup=warmup,
                    seed=child.child("mcmc"),
                )
                ml = fit_gaussian_process(train, y[:6], estimate_noise=True, seed=1)
            except ValueError:
                continue
            for point, truth in zip(held, y[6:], strict=True):
                low, high = bayes.predictive_interval(point, 0.8, include_noise=True)
                hits_bayes += low <= truth <= high
                mean, variance = ml.predict_with_variance(point, include_noise=True)
                hits_ml += abs(truth - mean) <= 1.2815515655446004 * math.sqrt(variance)
                total += 1
        return {
            "bayes": hits_bayes / total,
            "ml": hits_ml / total,
            "count": total,
        }

    def test_small_n_bayes_covers_better_than_ml(self) -> None:
        """With six points ML plug-in intervals are too narrow; Bayes is wider."""
        result = self._coverage_study(replicates=10, draws=60, warmup=30)
        self.assertGreaterEqual(result["count"], 40)
        # Nominal 0.80. ML undercovers; the mixture is closer to nominal and
        # clearly above ML (measured gap on this seed is far larger than 0.05).
        self.assertLess(result["ml"], 0.70)
        self.assertGreater(result["bayes"], result["ml"] + 0.05)
        self.assertGreater(result["bayes"], 0.60)

    @unittest.skipUnless(SLOW, "slow tier")
    def test_small_n_coverage_large_study(self) -> None:
        """Forty replicates; prints the measured coverages."""
        result = self._coverage_study(replicates=40, draws=100, warmup=50)
        print(f"\ncoverage study (slow tier): {result}")
        self.assertLess(result["ml"], 0.72)
        self.assertGreater(result["bayes"], result["ml"] + 0.05)


# ---------------------------------------------------------------------------
# 2. Emulator validation
# ---------------------------------------------------------------------------


class ValidationTests(unittest.TestCase):
    """Bastos-O'Hagan diagnostics."""

    def test_predictive_moments_match_pointwise_prediction(self) -> None:
        """Joint moments agree with the library's pointwise predictor."""
        pts = _grid_1d(8)
        gp = fit_gaussian_process(
            pts, [_smooth(p[0]) for p in pts], estimate_noise=True, seed=2
        )
        queries = [(0.05,), (0.33,), (0.71,), (1.4,)]
        mean, covariance = gp_predictive_moments(gp, queries)
        for i, q in enumerate(queries):
            m, v = gp.predict_with_variance(q)
            self.assertAlmostEqual(mean[i], m, places=9)
            self.assertAlmostEqual(covariance[i][i], v, delta=1e-9 + 1e-7 * v)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(covariance[i][j], covariance[j][i], places=12)
        _, noisy = gp_predictive_moments(gp, queries, include_noise=True)
        self.assertAlmostEqual(
            noisy[0][0] - covariance[0][0],
            gp.output_scale**2 * gp.standardized_noise_variance,
            delta=1e-9,
        )
        with self.assertRaises(TypeError):
            gp_predictive_moments("gp", queries)  # type: ignore[arg-type]

    def test_chi_square_reference_closed_form(self) -> None:
        """Identity covariance: MD = r'r and the p-value is the chi-square tail."""
        residual = [0.5, -1.0, 2.0]
        result = validate_emulator_moments(
            residual,
            [0.0, 0.0, 0.0],
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
        )
        self.assertAlmostEqual(result.mahalanobis, 5.25, places=12)
        cdf = ChiSquare(3.0).cdf(5.25)
        self.assertAlmostEqual(
            result.mahalanobis_p_value, 2.0 * min(cdf, 1.0 - cdf), places=12
        )
        self.assertEqual(result.mahalanobis_reference, "chi_square(3)")
        self.assertEqual(result.standardized_errors, (0.5, -1.0, 2.0))
        # Diagonal covariance: pivot order by decreasing variance, errors r/sd.
        spread = validate_emulator_moments(
            [0.4, 3.0, -1.0],
            [0.0, 0.0, 0.0],
            [[1.0, 0, 0], [0, 9.0, 0], [0, 0, 4.0]],
        )
        self.assertEqual(spread.pivot_order, (1, 2, 0))
        for got, want in zip(spread.pivoted_errors, (1.0, -0.5, 0.4), strict=True):
            self.assertAlmostEqual(got, want, places=12)

    def test_scaled_f_reference(self) -> None:
        """With estimated variance the statistic (nu-2)MD/(m nu) is F(m, nu)."""
        nu = 12.0
        size = 3
        residual = [1.0, 0.5, -0.8]
        identity = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        result = validate_emulator_moments(
            residual, [0.0] * 3, identity, degrees_of_freedom=nu
        )
        distance = sum(r * r for r in residual)
        cdf = FisherF(3.0, nu).cdf((nu - 2.0) * distance / (size * nu))
        self.assertAlmostEqual(
            result.mahalanobis_p_value, 2.0 * min(cdf, 1.0 - cdf), places=12
        )
        self.assertEqual(result.mahalanobis_reference, "scaled_F(3, 12)")
        with self.assertRaises(ValueError):
            validate_emulator_moments(
                residual, [0.0] * 3, identity, degrees_of_freedom=2.0
            )

    def test_pivoted_errors_reproduce_mahalanobis(self) -> None:
        """For a full-rank covariance the squared pivoted errors sum to MD."""
        pts = _grid_1d(9)
        gp = fit_gaussian_process(
            pts, [_smooth(p[0]) for p in pts], estimate_noise=True, seed=4
        )
        held = [(0.07,), (0.21,), (0.44,), (0.58,), (0.93,)]
        truth = [_smooth(p[0]) + 0.01 * (-1) ** i for i, p in enumerate(held)]
        result = validate_gp_emulator(gp, held, truth, include_noise=True)
        self.assertTrue(result.full_rank)
        self.assertEqual(sorted(result.pivot_order), [0, 1, 2, 3, 4])
        self.assertAlmostEqual(
            sum(e * e for e in result.pivoted_errors),
            result.mahalanobis,
            delta=1e-8 * (1 + result.mahalanobis),
        )

    def test_calibrated_emulator_passes_and_overconfident_fails(self) -> None:
        """Draws from the stated law pass; variances 400x too small are flagged."""
        generator = resolve_stream(77).generator()
        size = 12
        pts = [(i / (size - 1),) for i in range(size)]
        covariance = [
            [
                kernel_value("squared_exponential", pts[i], pts[j], (0.3,), 1.0)
                + (0.05 if i == j else 0.0)
                for j in range(size)
            ]
            for i in range(size)
        ]
        factor = _cholesky(covariance)
        z = [generator.gauss(0.0, 1.0) for _ in range(size)]
        y = [sum(factor[i][k] * z[k] for k in range(i + 1)) for i in range(size)]
        good = validate_emulator_moments(y, [0.0] * size, covariance)
        self.assertTrue(good.passed, good.flags)
        self.assertTrue(good.full_rank)
        scaled = [[v / 400.0 for v in row] for row in covariance]
        bad = validate_emulator_moments(y, [0.0] * size, scaled)
        self.assertFalse(bad.passed)
        self.assertTrue(any("Mahalanobis" in flag for flag in bad.flags))
        self.assertTrue(any("individual" in flag for flag in bad.flags))
        self.assertLess(bad.coverage, 0.95)

    def test_biased_simulator_is_flagged_on_fitted_gp(self) -> None:
        """A GP validated against a drifted simulator fails the checks."""
        pts = _grid_1d(8)
        gp = fit_gaussian_process(
            pts, [_smooth(p[0]) for p in pts], estimate_noise=True, seed=2
        )
        inside = [(0.1,), (0.35,), (0.6,), (0.85,)]
        ok = validate_gp_emulator(gp, inside, [_smooth(p[0]) for p in inside])
        # Interpolating a noise-free smooth function the errors are far below the
        # predicted spread: coverage is full and the only flag is the (legitimate)
        # "Mahalanobis distance too small" conservatism warning.
        self.assertEqual(ok.coverage, 1.0)
        self.assertTrue(all("too small" in flag for flag in ok.flags), ok.flags)
        self.assertLess(ok.mahalanobis, ok.mahalanobis_expected)
        # A simulator that drifted by +0.5 since the emulator was trained: the
        # interpolating GP is confident, so the errors are many sd's large.
        bad = validate_gp_emulator(gp, inside, [_smooth(p[0]) + 0.5 for p in inside])
        self.assertFalse(bad.passed)

    def test_rank_deficiency_and_invalid_input(self) -> None:
        """Near-singular covariance is flagged; exact singular raises."""
        eps = 1e-13
        near = [[1.0, 1.0 - eps], [1.0 - eps, 1.0]]
        result = validate_emulator_moments([0.1, 0.1], [0.0, 0.0], near)
        self.assertFalse(result.full_rank)
        self.assertTrue(any("rank" in flag for flag in result.flags))
        with self.assertRaises(ValueError):
            validate_emulator_moments([0.1, 0.1], [0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            validate_emulator_moments([0.1], [0.0], [[1.0]])
        with self.assertRaises(ValueError):
            validate_emulator_moments(
                [0.1, 0.2], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], level=1.0
            )


# ---------------------------------------------------------------------------
# 3. GP Sobol indices with emulator uncertainty
# ---------------------------------------------------------------------------

ISHIGAMI_THETA = (1.0176, 0.6302, 0.2551, 0.6272)  # ML fit at n=100, LHS seed 11


def _lhs(count: int, seed: int) -> list[tuple[float, float, float]]:
    """Return a Latin hypercube design on [-pi, pi]^3."""
    generator = resolve_stream(seed).generator()
    columns = []
    for _ in range(3):
        strata = list(range(count))
        generator.shuffle(strata)
        columns.append(
            [
                -math.pi + 2.0 * math.pi * (k + generator.random()) / count
                for k in strata
            ]
        )
    return [(columns[0][i], columns[1][i], columns[2][i]) for i in range(count)]


def _ishigami_truth(a: float = 7.0, b: float = 0.1) -> tuple[list[float], list[float]]:
    """Return analytic Ishigami first-order and total indices."""
    pi = math.pi
    variance = a * a / 8.0 + b * pi**4 / 5.0 + b * b * pi**8 / 18.0 + 0.5
    v1 = b * pi**4 / 5.0 + b * b * pi**8 / 50.0 + 0.5
    v2 = a * a / 8.0
    v13 = 8.0 * b * b * pi**8 / 225.0
    return [v1 / variance, v2 / variance, 0.0], [
        (v1 + v13) / variance,
        v2 / variance,
        v13 / variance,
    ]


class SobolTests(unittest.TestCase):
    """GP posterior Sobol indices on the Ishigami function."""

    @staticmethod
    def _emulator(count: int):
        """Fit-free GP on the Ishigami function with fixed hyperparameters."""
        pts = _lhs(count, 11)
        return fit_gaussian_process(
            pts,
            [ishigami(*p) for p in pts],
            kernel="squared_exponential",
            optimize=False,
            initial_log_parameters=ISHIGAMI_THETA,
            restarts=1,
        )

    def test_analytic_indices_reference(self) -> None:
        """The closed-form Ishigami indices sum as expected."""
        first, total = _ishigami_truth()
        for got, want in zip(first, (0.3139, 0.4424, 0.0), strict=True):
            self.assertAlmostEqual(got, want, places=3)
        for got, want in zip(total, (0.5576, 0.4424, 0.2437), strict=True):
            self.assertAlmostEqual(got, want, places=3)

    def test_estimators_on_additive_function(self) -> None:
        """Saltelli/Jansen estimators are exact in structure for f = x1 + 2 x2."""
        count = 4000
        generator = resolve_stream(1).generator()
        a = [[generator.random() for _ in range(2)] for _ in range(count)]
        b = [[generator.random() for _ in range(2)] for _ in range(count)]
        rows = [*a, *b]
        for index in range(2):
            for ra, rb in zip(a, b, strict=True):
                hybrid = list(ra)
                hybrid[index] = rb[index]
                rows.append(hybrid)
        values = [r[0] + 2.0 * r[1] for r in rows]
        first, total = sobol_estimates(values, count, 2)
        # Variances 1/12 and 4/12: S = (0.2, 0.8); no interactions so ST = S.
        for got, want in zip(first, (0.2, 0.8), strict=True):
            self.assertAlmostEqual(got, want, delta=0.05)
        for got, want in zip(total, (0.2, 0.8), strict=True):
            self.assertAlmostEqual(got, want, delta=0.05)
        with self.assertRaises(ValueError):
            sobol_estimates([1.0] * (4 * 8), 8, 2)

    def test_posterior_concentrates_with_design_size(self) -> None:
        """More training runs: smaller spread and error against the true function.

        Truth is the estimator applied to the true Ishigami function on the
        same Saltelli design, so Monte Carlo design error cancels.
        """
        bounds = ((-math.pi,) * 3, (math.pi,) * 3)
        errors = {}
        spreads = {}
        for count in (40, 120):
            result = gp_sobol_posterior(
                self._emulator(count),
                *bounds,
                base_samples=48,
                paths=30,
                seed=3,
            )
            truth_first, truth_total = sobol_estimates(
                [ishigami(*p) for p in result.design], 48, 3
            )
            self.assertEqual(result.evaluation_points, 48 * 5)
            self.assertEqual(len(result.first_order_samples), 30)
            errors[count] = max(
                max(
                    abs(m - t)
                    for m, t in zip(result.first_order_mean, truth_first, strict=True)
                ),
                max(
                    abs(m - t)
                    for m, t in zip(result.total_mean, truth_total, strict=True)
                ),
            )
            spreads[count] = sum(result.first_order_std) + sum(result.total_std)
            for index in range(3):
                low, high = result.total_interval[index]
                self.assertLess(low, result.total_mean[index])
                self.assertGreater(high, result.total_mean[index])
        self.assertLess(errors[120], errors[40])
        self.assertLess(errors[120], 0.12)
        self.assertLess(spreads[120], 0.6 * spreads[40])

    @unittest.skipUnless(SLOW, "slow tier")
    def test_large_design_near_analytic_indices(self) -> None:
        """N=120 base samples (600 points) and 120 training runs; prints numbers."""
        result = gp_sobol_posterior(
            self._emulator(120),
            (-math.pi,) * 3,
            (math.pi,) * 3,
            base_samples=120,
            paths=100,
            seed=8,
            max_points=700,
        )
        first, total = _ishigami_truth()
        truth_first, truth_total = sobol_estimates(
            [ishigami(*p) for p in result.design], 120, 3
        )
        print(
            "Sobol slow tier: S1",
            [round(v, 3) for v in result.first_order_mean],
            "ST",
            [round(v, 3) for v in result.total_mean],
            "truth on same design S1",
            [round(v, 3) for v in truth_first],
            "ST",
            [round(v, 3) for v in truth_total],
            "analytic S1",
            [round(v, 3) for v in first],
            "ST",
            [round(v, 3) for v in total],
        )
        # Emulator error alone (same design): tight. Analytic values also carry the
        # Monte Carlo error of the N=120 design: loose.
        for got, want in zip(result.first_order_mean, truth_first, strict=True):
            self.assertAlmostEqual(got, want, delta=0.1)
        for got, want in zip(result.total_mean, truth_total, strict=True):
            self.assertAlmostEqual(got, want, delta=0.1)
        for got, want in zip(result.first_order_mean, first, strict=True):
            self.assertAlmostEqual(got, want, delta=0.25)
        for got, want in zip(result.total_mean, total, strict=True):
            self.assertAlmostEqual(got, want, delta=0.25)

    def test_validation(self) -> None:
        """Invalid arguments are rejected."""
        gp = self._emulator(20)
        low, high = (-math.pi,) * 3, (math.pi,) * 3
        with self.assertRaises(ValueError):
            gp_sobol_posterior(gp, low, (math.pi,) * 2)
        with self.assertRaises(ValueError):
            gp_sobol_posterior(gp, high, low)
        with self.assertRaises(ValueError):
            gp_sobol_posterior(gp, low, high, base_samples=200, max_points=100)
        with self.assertRaises(TypeError):
            gp_sobol_posterior("gp", low, high)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. GP classification (Laplace)
# ---------------------------------------------------------------------------


def _logistic(f: float) -> float:
    """Return the logistic function."""
    return 1.0 / (1.0 + math.exp(-f))


class ClassifierTests(unittest.TestCase):
    """Laplace GP classification."""

    def test_single_point_mode_and_evidence(self) -> None:
        """One observation: the mode solves f = k (t - sigmoid(f))."""
        signal = 2.0
        nugget = 1e-8
        # Two points are needed for both classes; use a decoupled pair so the
        # one-dimensional closed form applies to each coordinate.
        clf = fit_laplace_gp_classifier(
            [(0.0,), (1e3,)],
            [1, 0],
            length_scales=(1.0,),
            signal_variance=signal,
            nugget=nugget,
        )
        self.assertTrue(clf.converged)
        k = signal + nugget
        # Scalar bisection for f = k (t - sigmoid(f)), t = 1 then t = 0.
        for index, target in ((0, 1.0), (1, 0.0)):
            low, high = -50.0, 50.0
            for _ in range(200):
                mid = 0.5 * (low + high)
                if mid - k * (target - _logistic(mid)) > 0.0:
                    high = mid
                else:
                    low = mid
            self.assertAlmostEqual(clf.latent_mode[index], 0.5 * (low + high), places=8)
        # Evidence of the decoupled pair is the sum of 1-D Laplace evidences.
        total = 0.0
        for f, sign in zip(clf.latent_mode, (1.0, -1.0), strict=True):
            weight = _logistic(f) * (1.0 - _logistic(f))
            total += (
                -0.5 * f * f / k
                + math.log(_logistic(sign * f))
                - 0.5 * math.log(1.0 + weight * k)
            )
        self.assertAlmostEqual(clf.log_marginal_likelihood, total, places=7)

    def test_mode_satisfies_stationarity(self) -> None:
        """At the mode f = K (t - pi) holds (independent dense check)."""
        pts = [(x,) for x in (-2.0, -1.2, -0.4, 0.3, 1.1, 1.9)]
        labels = [0, 0, 1, 0, 1, 1]
        clf = fit_laplace_gp_classifier(
            pts, labels, length_scales=(0.9,), signal_variance=4.0, nugget=1e-8
        )
        self.assertTrue(clf.converged)
        for i, p in enumerate(pts):
            rhs = sum(
                kernel_value("squared_exponential", p, q, (0.9,), 4.0)
                * (t - _logistic(f))
                for q, t, f in zip(pts, labels, clf.latent_mode, strict=True)
            )
            self.assertAlmostEqual(
                clf.latent_mode[i],
                rhs + 1e-8 * (labels[i] - _logistic(clf.latent_mode[i])),
                places=6,
            )

    def test_separable_one_dimensional(self) -> None:
        """Separable 1-D data: probability crosses 1/2 between the classes."""
        pts = [(x,) for x in (-3.0, -2.2, -1.5, -0.8, 0.8, 1.5, 2.2, 3.0)]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        clf = fit_laplace_gp_classifier(
            pts, labels, length_scales=(1.5,), signal_variance=9.0
        )
        self.assertLess(clf.predict_proba((-2.5,)), 0.25)
        self.assertGreater(clf.predict_proba((2.5,)), 0.75)
        self.assertAlmostEqual(clf.predict_proba((0.0,)), 0.5, delta=0.02)
        self.assertEqual(clf.predict_class((-1.0,)), 0)
        self.assertEqual(clf.predict_class((1.0,)), 1)
        # Probability is monotone through the gap between the classes (it reverts
        # towards 1/2 near the outer edge of the data, so the ends are excluded).
        grid = [clf.predict_proba((x / 4.0,)) for x in range(-6, 7)]
        self.assertTrue(
            all(b >= a - 1e-9 for a, b in zip(grid, grid[1:], strict=False))
        )
        # Quadrature agrees with the probit-type closed approximation.
        mean, variance = clf.predict_latent((2.0,))
        approx = _logistic(mean / math.sqrt(1.0 + math.pi * variance / 8.0))
        self.assertAlmostEqual(clf.predict_proba((2.0,)), approx, delta=0.01)
        self.assertGreater(variance, 0.0)

    def test_nonseparable_two_dimensional(self) -> None:
        """Class = inside the unit circle cannot be split by a line."""
        generator = resolve_stream(31).generator()
        pts: list[tuple[float, float]] = []
        labels: list[int] = []
        while len(pts) < 30:
            p = (generator.uniform(-2, 2), generator.uniform(-2, 2))
            if abs(math.hypot(*p) - 1.0) < 0.2:
                continue
            pts.append(p)
            labels.append(1 if math.hypot(*p) < 1.0 else 0)
        clf = fit_laplace_gp_classifier(
            pts, labels, length_scales=(0.8, 0.8), signal_variance=9.0
        )
        self.assertTrue(clf.converged)
        self.assertGreater(clf.predict_proba((0.0, 0.0)), 0.8)
        for outside in ((1.2, 1.2), (-1.4, -1.0), (0.0, -1.5), (-1.5, 1.2)):
            self.assertLess(clf.predict_proba(outside), 0.25)
        # Training accuracy.
        correct = sum(
            clf.predict_class(p) == t for p, t in zip(pts, labels, strict=True)
        )
        self.assertGreaterEqual(correct, 28)

    def test_optimization_increases_evidence(self) -> None:
        """Maximizing the Laplace evidence cannot lower it."""
        pts = [(x,) for x in (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0)]
        labels = [0, 0, 1, 0, 1, 1, 1, 0]
        start = fit_laplace_gp_classifier(
            pts, labels, length_scales=(0.2,), signal_variance=1.0
        )
        tuned = fit_laplace_gp_classifier(
            pts, labels, length_scales=(0.2,), signal_variance=1.0, optimize=True
        )
        self.assertGreater(
            tuned.log_marginal_likelihood, start.log_marginal_likelihood + 1e-3
        )

    def test_validation(self) -> None:
        """Invalid labels and parameters are rejected."""
        pts = [(0.0,), (1.0,), (2.0,)]
        with self.assertRaises(ValueError):
            fit_laplace_gp_classifier(pts, [1, 1, 1])
        with self.assertRaises(ValueError):
            fit_laplace_gp_classifier(pts, [0, 2, 1])
        with self.assertRaises(ValueError):
            fit_laplace_gp_classifier(pts, [0, 1, 1], length_scales=(-1.0,))
        with self.assertRaises(ValueError):
            fit_laplace_gp_classifier(pts, [0, 1])
        clf = fit_laplace_gp_classifier(pts, [0, 1, 1])
        with self.assertRaises(ValueError):
            clf.predict_class((0.5,), threshold=1.0)
        with self.assertRaises(ValueError):
            clf.predict_proba((0.5, 0.5))


# ---------------------------------------------------------------------------
# 5. Student-t process
# ---------------------------------------------------------------------------


class StudentTProcessTests(unittest.TestCase):
    """Student-t process regression."""

    def test_univariate_density(self) -> None:
        """A single observation gives the univariate Student-t density."""
        nu = 5.0
        kvar = 1.7
        y = 0.9
        got = student_t_process_log_marginal_likelihood(
            [(0.0,)],
            [y],
            nu=nu,
            signal_variance=kvar - 0.2,
            length_scales=(1.0,),
            noise_variance=0.2,
        )
        scale = math.sqrt(kvar * (nu - 2.0) / nu)
        z = y / scale
        want = (
            math.lgamma(0.5 * (nu + 1.0))
            - math.lgamma(0.5 * nu)
            - 0.5 * math.log(nu * math.pi)
            - math.log(scale)
            - 0.5 * (nu + 1.0) * math.log(1.0 + z * z / nu)
        )
        self.assertAlmostEqual(got, want, places=12)

    def test_tends_to_gaussian_likelihood(self) -> None:
        """The gap to the Gaussian log density shrinks like 1/nu."""
        pts = _grid_1d(6)
        y = [_smooth(p[0]) for p in pts]
        theta = (0.2, -1.0, -4.0)
        gaussian, _ = gaussian_process_log_marginal_likelihood(
            pts, y, theta, kernel="matern52", nugget=1e-10, estimate_noise=True
        )
        gaps = []
        for nu in (50.0, 5e3, 5e5):
            value = student_t_process_log_marginal_likelihood(
                pts,
                y,
                nu=nu,
                signal_variance=math.exp(theta[0]),
                length_scales=(math.exp(theta[1]),),
                noise_variance=1e-10 + math.exp(theta[2]),
            )
            gaps.append(abs(value - gaussian))
        self.assertGreater(gaps[0], 10.0 * gaps[1])
        self.assertGreater(gaps[1], 10.0 * gaps[2])
        self.assertLess(gaps[2], 1e-3)

    def test_predictive_variance_scaling_against_dense_solve(self) -> None:
        """Variance = (nu + beta - 2)/(nu + n - 2) times the GP variance."""
        pts = _grid_1d(7)
        y = [_smooth(p[0]) for p in pts]
        theta = (0.1, -1.1, -5.0)
        nu = 6.0
        fit = fit_student_t_process(pts, y, nu=nu, log_parameters=theta, nugget=1e-8)
        mean = sum(y) / 7
        sd = math.sqrt(sum((v - mean) ** 2 for v in y) / 6)
        z = [(v - mean) / sd for v in y]
        signal, scale, noise = (math.exp(t) for t in theta)
        matrix = [
            [
                kernel_value("matern52", a, b, (scale,), signal)
                + (noise + 1e-8 if a == b else 0.0)
                for b in pts
            ]
            for a in pts
        ]
        alpha = solve_spd(matrix, z)
        beta = sum(a * b for a, b in zip(z, alpha, strict=True))
        self.assertAlmostEqual(fit.beta, beta, delta=1e-9 * (1 + beta))
        query = (0.43,)
        cross = [kernel_value("matern52", query, p, (scale,), signal) for p in pts]
        v = solve_spd(matrix, cross)
        gp_var = signal - sum(a * b for a, b in zip(cross, v, strict=True))
        gp_mean = sum(a * b for a, b in zip(cross, alpha, strict=True))
        want_var = sd**2 * (nu + beta - 2.0) / (nu + 7 - 2.0) * gp_var
        got_mean, got_var = fit.predict_with_variance(query)
        self.assertAlmostEqual(got_mean, mean + sd * gp_mean, places=9)
        self.assertAlmostEqual(got_var, want_var, delta=1e-9 * want_var)
        low, high = fit.predictive_interval(query, 0.9)
        self.assertAlmostEqual(0.5 * (low + high), got_mean, places=9)

    def test_variance_depends_on_data_unlike_gp(self) -> None:
        """An outlier inflates the t-process variance; the GP variance cannot."""
        pts = _grid_1d(8)
        smooth = [_smooth(p[0]) for p in pts]
        outlier = list(smooth)
        outlier[4] += 6.0
        theta = (0.0, -1.0, -4.0)
        plain = fit_student_t_process(pts, smooth, nu=4.0, log_parameters=theta)
        spiky = fit_student_t_process(pts, outlier, nu=4.0, log_parameters=theta)
        self.assertGreater(spiky.beta, plain.beta)
        # Compare scale-free ratios: standardized targets make GP variance equal.
        ratio = (
            spiky.predict_with_variance((0.5,))[1]
            / spiky.output_scale**2
            / (plain.predict_with_variance((0.5,))[1] / plain.output_scale**2)
        )
        self.assertAlmostEqual(
            ratio, spiky.variance_scaling() / plain.variance_scaling(), places=9
        )
        self.assertGreater(ratio, 1.0)

    def test_large_nu_matches_gaussian_process(self) -> None:
        """A huge nu (1e6) reproduces the GP predictive mean and variance."""
        pts = _grid_1d(8)
        y = [_smooth(p[0]) for p in pts]
        theta = (0.1, -1.0, -5.0)
        gp = fit_gaussian_process(
            pts,
            y,
            optimize=False,
            estimate_noise=True,
            initial_log_parameters=theta,
            nugget=1e-8,
            restarts=1,
        )
        fit = fit_student_t_process(pts, y, nu=1e6, log_parameters=theta, nugget=1e-8)
        for x in (0.1, 0.52, 0.97):
            gm, gv = gp.predict_with_variance((x,))
            tm, tv = fit.predict_with_variance((x,))
            self.assertAlmostEqual(tm, gm, places=9)
            self.assertAlmostEqual(tv, gv, delta=2e-5 * gv + 1e-12)
        self.assertAlmostEqual(fit.variance_scaling(), 1.0, delta=2e-5)

    def test_nu_selection_and_validation(self) -> None:
        """Profile selection runs on the grid; invalid nu is rejected."""
        pts = _grid_1d(9)
        y = [_smooth(p[0]) for p in pts]
        fit = fit_student_t_process(pts, y, seed=1)
        self.assertGreater(fit.nu, 2.0)
        self.assertTrue(math.isfinite(fit.log_likelihood))
        with self.assertRaises(ValueError):
            fit_student_t_process(pts, y, nu=2.0)
        with self.assertRaises(ValueError):
            fit_student_t_process(pts, y, nu=5.0, log_parameters=(0.0, 0.0))
        with self.assertRaises(ValueError):
            student_t_process_log_marginal_likelihood(
                pts,
                y,
                nu=1.5,
                signal_variance=1.0,
                length_scales=(0.3,),
                noise_variance=0.1,
            )


# ---------------------------------------------------------------------------
# 6. Output-warped GP
# ---------------------------------------------------------------------------


class WarpedGPTests(unittest.TestCase):
    """Box-Cox warped GP on a positive skewed response."""

    def setUp(self) -> None:
        """Build a positive, right-skewed response."""
        self.points = _grid_1d(12)
        self.values = [
            math.exp(1.2 * math.sin(2.0 * math.pi * p[0])) for p in self.points
        ]

    def test_box_cox_round_trip_and_limits(self) -> None:
        """Inverse recovers y; lambda -> 0 tends to log."""
        for lam in (-1.5, -0.3, 0.0, 0.4, 1.0, 2.0):
            for y in (0.2, 1.0, 3.7):
                self.assertAlmostEqual(
                    inverse_box_cox(box_cox(y, lam), lam), y, delta=1e-12 * y
                )
        self.assertAlmostEqual(box_cox(2.0, 1e-9), math.log(2.0), places=8)
        self.assertAlmostEqual(box_cox(2.0, 1.0), 1.0, places=14)
        with self.assertRaises(ValueError):
            box_cox(-1.0, 0.5)
        with self.assertRaises(ValueError):
            inverse_box_cox(-3.0, 0.5)  # lambda z + 1 = -0.5

    def test_log_warp_equals_gp_on_log_values(self) -> None:
        """With lambda = 0 the warp is the log: median and Jacobian are exact."""
        warped = fit_warped_gp(self.points, self.values, warp_lambda=0.0, seed=2)
        plain = fit_gaussian_process(
            self.points,
            [math.log(v) for v in self.values],
            estimate_noise=True,
            restarts=2,
            seed=2,
            nugget=1e-8,
        )
        self.assertFalse(warped.lambda_searched)
        for x in (0.05, 0.33, 0.8):
            self.assertAlmostEqual(
                warped.median((x,)), math.exp(plain.predict((x,))), places=9
            )
            mean, variance = plain.predict_with_variance((x,))
            want = math.exp(mean + 1.6448536269514722 * math.sqrt(variance))
            self.assertAlmostEqual(warped.quantile((x,), 0.95), want, delta=1e-9 * want)
        jacobian = -sum(math.log(v) for v in self.values)
        self.assertAlmostEqual(
            warped.log_likelihood, plain.log_marginal_likelihood + jacobian, places=8
        )

    def test_positive_skewed_response(self) -> None:
        """The searched warp is near log, beats the plain GP and stays positive."""
        warped = fit_warped_gp(self.points, self.values, seed=2, search_iterations=5)
        self.assertTrue(warped.lambda_searched)
        self.assertLess(abs(warped.warp_lambda), 0.5)
        plain = fit_gaussian_process(
            self.points,
            self.values,
            estimate_noise=True,
            restarts=2,
            seed=2,
            nugget=1e-8,
        )
        # Both are densities of y: the warped one includes the Jacobian.
        self.assertGreater(warped.log_likelihood, plain.log_marginal_likelihood)
        queries = [(i / 40.0,) for i in range(41)]
        lows = [warped.quantile(q, 0.05) for q in queries]
        self.assertTrue(all(v > 0.0 for v in lows))
        # Right skew in y: at the peak the upper tail is longer than the lower.
        peak = (0.25,)
        median = warped.median(peak)
        self.assertGreater(
            warped.quantile(peak, 0.95) - median, median - warped.quantile(peak, 0.05)
        )
        # Predictive median tracks the truth on the held-out midpoints.
        for q in [(0.025,), (0.275,), (0.525,), (0.775,)]:
            truth = math.exp(1.2 * math.sin(2.0 * math.pi * q[0]))
            self.assertAlmostEqual(warped.median(q), truth, delta=0.25 * truth)

    def test_validation(self) -> None:
        """Non-positive outputs and invalid arguments are rejected."""
        with self.assertRaises(ValueError):
            fit_warped_gp(self.points, [*self.values[:-1], -0.1])
        with self.assertRaises(ValueError):
            fit_warped_gp(self.points, self.values, lambda_bounds=(1.0, 0.0))
        warped = fit_warped_gp(self.points, self.values, warp_lambda=1.0, seed=1)
        with self.assertRaises(ValueError):
            warped.quantile((0.5,), 1.0)


if __name__ == "__main__":
    unittest.main()
