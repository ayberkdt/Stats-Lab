"""Tests for the generalized Pareto law and ``probstat.risk.extremes``.

Closed forms are re-derived inside each test from the textbook formulas
(Coles 2001, Hosking-Wallis 1987); stochastic checks use fixed seeds and
tolerances of a few standard errors.
"""

from __future__ import annotations

import math
import unittest

from probstat.distributions import (
    Exponential,
    GeneralizedExtremeValue,
    GeneralizedPareto,
)
from probstat.foundations import RandomStream
from probstat.modeling import standard_normal_quantile
from probstat.risk import (
    ExceedanceRate,
    block_maxima,
    decluster_runs,
    exceedance_rate,
    fit_gev,
    fit_gpd,
    fit_pot,
    gev_return_level,
    gev_return_level_gradient,
    mean_residual_life,
    pot_return_level,
    sample_l_moments,
    threshold_excesses,
    threshold_stability,
)


class GeneralizedParetoTests(unittest.TestCase):
    """Distribution-level contracts of :class:`GeneralizedPareto`."""

    SHAPES = (0.4, 0.05, 1e-9, 0.0, -1e-9, -0.3, -1.0, -1.7)

    def test_cdf_quantile_isf_round_trip(self) -> None:
        """``F(Q(p)) = p`` and ``S(isf(q)) = q`` across shapes and tails.

        Tiny ``q`` is checked only for ``xi >= 0``: with a finite upper
        endpoint the level ``isf(q)`` is within an ulp of the endpoint, so ``q``
        cannot be recovered to relative precision from a rounded ``x``.
        """
        for xi in self.SHAPES:
            law = GeneralizedPareto(2.0, 1.5, xi)
            for p in (1e-12, 1e-4, 0.1, 0.5, 0.9, 0.999):
                with self.subTest(xi=xi, p=p):
                    x = law.quantile(p)
                    self.assertAlmostEqual(law.cdf(x), p, delta=1e-12 * max(1.0, p))
                    self.assertAlmostEqual(law.cdf(x) + law.sf(x), 1.0, places=14)
            tails = (1e-15, 1e-8, 0.01, 0.5) if xi >= 0.0 else (0.01, 0.5)
            for q in tails:
                with self.subTest(xi=xi, q=q):
                    x = law.isf(q)
                    self.assertLess(abs(law.sf(x) / q - 1.0), 1e-9)

    def test_exponential_limit_is_continuous(self) -> None:
        """At ``xi = 0`` and ``|xi| = 1e-12`` the law matches ``Exp(1/sigma)``.

        The log-density tolerance is ``1e-8``: at ``z = 16`` the exact GPD with
        ``xi = 1e-12`` differs from the exponential by ``xi z^2/2 ~ 1e-10``.
        """
        reference = Exponential(1.0 / 2.5)
        for xi in (0.0, 1e-12, -1e-12):
            law = GeneralizedPareto(0.0, 2.5, xi)
            for x in (0.0, 0.3, 4.0, 40.0):
                with self.subTest(xi=xi, x=x):
                    self.assertAlmostEqual(law.cdf(x), reference.cdf(x), places=12)
                    self.assertAlmostEqual(
                        law.log_pdf(x), reference.log_pdf(x), places=8
                    )
            for q in (0.5, 1e-10):
                with self.subTest(xi=xi, q=q):
                    self.assertAlmostEqual(
                        law.isf(q) / reference.isf(q), 1.0, places=10
                    )
        tiny = GeneralizedPareto(0.0, 1.0, 1e-7)
        self.assertAlmostEqual(tiny.mean, 1.0 / (1.0 - 1e-7), places=15)

    def test_density_is_derivative_of_cdf(self) -> None:
        """A central difference of the CDF reproduces the density."""
        for xi in (0.3, 0.0, -0.4):
            law = GeneralizedPareto(1.0, 2.0, xi)
            for x in (1.2, 2.0, 4.5):
                h = 1e-5
                numeric = (law.cdf(x + h) - law.cdf(x - h)) / (2.0 * h)
                with self.subTest(xi=xi, x=x):
                    self.assertAlmostEqual(law.pdf(x), numeric, places=8)
                    self.assertAlmostEqual(
                        law.hazard(x), law.pdf(x) / law.sf(x), places=12
                    )

    def test_support_endpoints_and_uniform_case(self) -> None:
        """Support ends, the uniform case ``xi = -1``, and off-support values."""
        law = GeneralizedPareto(1.0, 2.0, -0.5)
        self.assertEqual(law.upper_endpoint, 5.0)
        self.assertEqual(law.quantile(1.0), 5.0)
        self.assertEqual(law.cdf(6.0), 1.0)
        self.assertEqual(law.log_pdf(0.5), -math.inf)
        self.assertEqual(law.log_pdf(5.0), -math.inf)
        self.assertEqual(law.quantile(0.0), 1.0)
        uniform = GeneralizedPareto(0.0, 3.0, -1.0)
        self.assertAlmostEqual(uniform.pdf(1.0), 1.0 / 3.0, places=15)
        self.assertAlmostEqual(uniform.cdf(1.5), 0.5, places=15)
        self.assertEqual(GeneralizedPareto(0.0, 1.0, 0.2).upper_endpoint, None)
        self.assertEqual(GeneralizedPareto(0.0, 1.0, 0.2).quantile(1.0), math.inf)
        with self.assertRaises(ValueError):
            GeneralizedPareto(0.0, 0.0, 0.1)

    def test_moments_and_sampling(self) -> None:
        """Closed-form mean and variance agree with a seeded sample."""
        law = GeneralizedPareto(0.0, 1.0, 0.2)
        self.assertAlmostEqual(law.mean, 1.25, places=15)
        self.assertAlmostEqual(law.variance, 1.0 / (0.64 * 0.6), places=14)
        self.assertEqual(GeneralizedPareto(0.0, 1.0, 0.6).variance, math.inf)
        draws = law.sample(20_000, seed=7)
        self.assertEqual(draws, law.sample(20_000, seed=7))
        mean = math.fsum(draws) / len(draws)
        self.assertLess(abs(mean - law.mean), 4.0 * math.sqrt(law.variance / 20_000))
        self.assertTrue(all(x >= 0.0 for x in draws))


class BlockMaximaTests(unittest.TestCase):
    """GEV return levels, delta-method and profile intervals."""

    def test_block_maxima_discards_partial_block(self) -> None:
        """Seven values in blocks of three give two maxima."""
        self.assertEqual(block_maxima([1, 5, 2, 0, 3, 4, 9], 3), (5.0, 4.0))
        with self.assertRaises(ValueError):
            block_maxima([1, 2], 3)
        with self.assertRaises(ValueError):
            block_maxima([1, 2], 0)

    def test_return_level_matches_closed_form(self) -> None:
        """``z_T = mu - sigma/xi (1 - y^-xi)`` and ``mu - sigma log y`` at 0."""
        for xi in (0.25, -0.2, 0.0, 1e-10):
            for period in (2.0, 10.0, 100.0, 1e4):
                y = -math.log(1.0 - 1.0 / period)
                if xi == 0.0:
                    expected = 3.0 - 2.0 * math.log(y)
                else:
                    expected = 3.0 - 2.0 / xi * (1.0 - y ** (-xi))
                with self.subTest(xi=xi, period=period):
                    level = gev_return_level(3.0, 2.0, xi, period)
                    self.assertAlmostEqual(level, expected, delta=1e-6 * abs(expected))
                    isf = GeneralizedExtremeValue(3.0, 2.0, xi).isf(1.0 / period)
                    self.assertAlmostEqual(level, isf, delta=1e-12 * abs(isf))
        with self.assertRaises(ValueError):
            gev_return_level(0.0, 1.0, 0.1, 1.0)

    def test_gradient_matches_finite_differences(self) -> None:
        """The analytic gradient, including its ``xi -> 0`` series branch."""
        for xi in (0.3, 0.01, 0.0, -0.25):
            params = [1.0, 2.0, xi]
            gradient = gev_return_level_gradient(*params, 50.0)
            for index in range(3):
                h = 1e-6
                up, down = list(params), list(params)
                up[index] += h
                down[index] -= h
                numeric = (
                    gev_return_level(*up, 50.0) - gev_return_level(*down, 50.0)
                ) / (2.0 * h)
                with self.subTest(xi=xi, index=index):
                    self.assertAlmostEqual(gradient[index], numeric, places=5)

    def test_fit_and_intervals(self) -> None:
        """MLE is near the truth; the profile interval is right-skewed."""
        maxima = GeneralizedExtremeValue(10.0, 2.0, 0.1).sample(80, seed=3)
        fit = fit_gev(maxima)
        for estimate, truth, error in zip(
            fit.fit.estimates, (10.0, 2.0, 0.1), fit.fit.standard_errors, strict=True
        ):
            self.assertLess(abs(estimate - truth), 3.0 * error)
        delta = fit.return_level(100.0)
        self.assertAlmostEqual(
            delta.level, fit.distribution.isf(0.01), delta=1e-9 * delta.level
        )
        z = standard_normal_quantile(0.975)
        self.assertAlmostEqual(
            delta.upper - delta.level, z * delta.standard_error, places=10
        )
        profile = fit.profile_return_level(100.0)
        self.assertEqual(profile.method, "profile")
        self.assertLess(profile.lower, profile.level)
        self.assertGreater(profile.upper - profile.level, profile.level - profile.lower)
        critical = z * z
        for end in (profile.lower, profile.upper):
            deviance = 2.0 * (
                fit.fit.log_likelihood - fit.profile_log_likelihood(end, 100.0)
            )
            self.assertAlmostEqual(deviance, critical, delta=2e-3)

    def test_delta_interval_coverage_sanity(self) -> None:
        """Nominal 95 % delta intervals for ``z_10`` cover in 82-99 % of 100 fits.

        The Wald interval is known to undercover slightly for ``n = 50``.
        """
        truth = gev_return_level(0.0, 1.0, 0.1, 10.0)
        stream = RandomStream(2026)
        replicates = 100
        hits = 0
        for index in range(replicates):
            seed = stream.child("gev", index).integer_seed()
            data = GeneralizedExtremeValue(0.0, 1.0, 0.1).sample(50, seed=seed)
            interval = fit_gev(data).return_level(10.0)
            hits += interval.lower <= truth <= interval.upper
        self.assertGreaterEqual(hits / replicates, 0.82)
        self.assertLessEqual(hits / replicates, 0.99)


class GeneralizedParetoFittingTests(unittest.TestCase):
    """PWM/L-moment and maximum-likelihood GPD estimators."""

    def test_l_moments_and_pwm_on_hand_data(self) -> None:
        """For ``1..5``: ``b0 = 3``, ``b1 = 2``, so ``sigma = 6`` and ``xi = -1``."""
        self.assertEqual(sample_l_moments([5, 3, 1, 4, 2]), (3.0, 1.0, 0.0, 0.0))
        fit = fit_gpd([1, 2, 3, 4, 5], method="pwm")
        self.assertAlmostEqual(fit.sigma, 6.0, places=14)
        self.assertAlmostEqual(fit.xi, -1.0, places=14)
        self.assertIsNone(fit.covariance)
        data = (0.2, 0.5, 0.9, 1.4, 3.1, 7.0)
        n = len(data)
        b0 = sum(data) / n
        b1 = sum(j * x for j, x in enumerate(sorted(data))) / (n * (n - 1))
        alpha0, alpha1 = b0, b0 - b1
        xi_hw = 2.0 - alpha0 / (alpha0 - 2.0 * alpha1)
        sigma_hw = 2.0 * alpha0 * alpha1 / (alpha0 - 2.0 * alpha1)
        fit = fit_gpd(data, method="pwm")
        self.assertAlmostEqual(fit.xi, xi_hw, places=12)
        self.assertAlmostEqual(fit.sigma, sigma_hw, places=12)

    def test_mle_recovers_seeded_parameters(self) -> None:
        """2000 excesses from ``GPD(1.5, 0.2)`` give estimates within 3 SE."""
        excesses = GeneralizedPareto(0.0, 1.5, 0.2).sample(2000, seed=11)
        fit = fit_gpd(excesses)
        se_sigma, se_xi = fit.standard_errors
        self.assertTrue(fit.converged)
        self.assertLess(abs(fit.sigma - 1.5), 3.0 * se_sigma)
        self.assertLess(abs(fit.xi - 0.2), 3.0 * se_xi)
        self.assertLess(se_xi, 0.05)
        pwm = fit_gpd(excesses, method="pwm")
        self.assertLess(abs(pwm.xi - 0.2), 0.1)
        self.assertLessEqual(pwm.log_likelihood, fit.log_likelihood + 1e-9)

    def test_invalid_inputs(self) -> None:
        """Negative excesses, constant data and unknown methods are rejected."""
        with self.assertRaises(ValueError):
            fit_gpd([1.0, -0.5, 2.0])
        with self.assertRaises(ValueError):
            fit_gpd([1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            fit_gpd([1.0, 2.0, 3.0], method="moments")  # type: ignore[arg-type]


class PeaksOverThresholdTests(unittest.TestCase):
    """Threshold diagnostics, rates, declustering and POT return levels."""

    def test_excesses_and_rate(self) -> None:
        """Excesses are strict, and the binomial rate algebra is exact."""
        values = [1.0, 3.0, 2.0, 5.0, 2.0, 0.5, 4.0, 2.0]
        self.assertEqual(threshold_excesses(values, 2.0), (1.0, 3.0, 2.0))
        rate = exceedance_rate(values, 2.0, observations_per_year=4.0)
        self.assertEqual((rate.exceedances, rate.observations), (3, 8))
        self.assertAlmostEqual(rate.probability, 0.375)
        self.assertAlmostEqual(rate.annual_rate, 1.5)
        self.assertAlmostEqual(rate.probability_variance, 0.375 * 0.625 / 8)
        self.assertAlmostEqual(
            rate.probability_of_exceedance(2.0), 1.0 - math.exp(-3.0)
        )
        with self.assertRaises(ValueError):
            ExceedanceRate(0, 5, 1.0)

    def test_runs_declustering_hand_example(self) -> None:
        """Run length 2 splits ``5 6 | 7 | 8 9`` into three clusters."""
        series = [0, 5, 6, 0, 0, 7, 0, 0, 0, 8, 9, 0]
        result = decluster_runs(series, 4.0, 2)
        self.assertEqual(result.peaks, (6.0, 7.0, 9.0))
        self.assertEqual(result.peak_indices, (2, 5, 10))
        self.assertEqual(result.cluster_sizes, (2, 1, 2))
        self.assertAlmostEqual(result.extremal_index, 0.6)
        merged = decluster_runs(series, 4.0, 3)
        self.assertEqual(merged.peaks, (7.0, 9.0))
        self.assertEqual(merged.cluster_sizes, (3, 2))

    def test_pot_return_level_formula_and_limit(self) -> None:
        """``x_N = u + sigma/xi ((m zeta)^xi - 1)``; ``u + sigma log(m zeta)`` at 0."""
        m_zeta = 50.0 * 365.0 * 0.02
        expected = 3.0 + 1.2 / 0.15 * (m_zeta**0.15 - 1.0)
        self.assertAlmostEqual(
            pot_return_level(3.0, 1.2, 0.15, 0.02, 365.0, 50.0), expected, places=10
        )
        self.assertAlmostEqual(
            pot_return_level(3.0, 1.2, 0.0, 0.02, 365.0, 50.0),
            3.0 + 1.2 * math.log(m_zeta),
            places=12,
        )
        self.assertAlmostEqual(
            pot_return_level(3.0, 1.2, 1e-12, 0.02, 365.0, 50.0),
            3.0 + 1.2 * math.log(m_zeta),
            places=9,
        )
        with self.assertRaises(ValueError):
            pot_return_level(3.0, 1.2, 0.1, 0.001, 1.0, 10.0)

    def test_fit_pot_return_level_and_rate_consistency(self) -> None:
        """The fitted level has annual exceedance rate ``1/N`` and a sane CI."""
        series = GeneralizedPareto(0.0, 1.0, 0.1).sample(3650, seed=5)
        pot = fit_pot(series, 2.0, observations_per_year=365.0)
        level = pot.return_level(50.0)
        self.assertAlmostEqual(
            pot.annual_exceedance_rate(level.level), 1.0 / 50.0, places=12
        )
        truth = GeneralizedPareto(0.0, 1.0, 0.1).isf(1.0 / (50.0 * 365.0))
        self.assertLess(level.lower, truth)
        self.assertGreater(level.upper, truth)
        params = (pot.rate.probability, pot.gpd.sigma, pot.gpd.xi)
        h = 1e-6
        gradient = []
        for index in range(3):
            up, down = list(params), list(params)
            up[index] += h
            down[index] -= h
            gradient.append(
                (
                    pot_return_level(2.0, up[1], up[2], up[0], 365.0, 50.0)
                    - pot_return_level(2.0, down[1], down[2], down[0], 365.0, 50.0)
                )
                / (2.0 * h)
            )
        v = pot.gpd.covariance
        assert v is not None
        variance = (
            gradient[0] ** 2 * pot.rate.probability_variance
            + gradient[1] ** 2 * v[0][0]
            + 2.0 * gradient[1] * gradient[2] * v[0][1]
            + gradient[2] ** 2 * v[1][1]
        )
        self.assertAlmostEqual(level.standard_error, math.sqrt(variance), places=5)
        pwm = fit_pot(series, 2.0, observations_per_year=365.0, method="pwm")
        self.assertEqual(pwm.return_level(50.0).method, "point")
        declustered = fit_pot(series, 2.0, observations_per_year=365.0, run_length=3)
        assert declustered.declustered is not None
        self.assertLessEqual(declustered.rate.exceedances, pot.rate.exceedances)

    def test_threshold_diagnostics_on_exponential_data(self) -> None:
        """Exponential data: flat mean excess near 1, shape near 0, stable scale."""
        data = Exponential(1.0).sample(4000, seed=21)
        points = mean_residual_life(data, [0.0, 0.5, 1.0, 2.0])
        for point in points:
            excess = [x - point.threshold for x in data if x > point.threshold]
            self.assertEqual(point.exceedances, len(excess))
            self.assertAlmostEqual(point.mean_excess, sum(excess) / len(excess))
            self.assertLess(point.lower, 1.0 + 0.15)
            self.assertGreater(point.upper, 1.0 - 0.15)
        self.assertEqual(len(mean_residual_life(data)), 40)
        stability = threshold_stability(data, [0.5, 1.0, 1.5])
        for point in stability:
            self.assertLess(point.xi_lower, 0.0)
            self.assertGreater(point.xi_upper, 0.0)
            self.assertLess(point.modified_scale_lower, 1.0)
            self.assertGreater(point.modified_scale_upper, 1.0)


if __name__ == "__main__":
    unittest.main()
