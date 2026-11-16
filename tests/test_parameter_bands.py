"""Tests for parameter-uncertainty CDF bands (``inference.parameter_uncertainty_band``).

Oracles:

* Closed form of the delta method for a normal fit with
  ``Cov = diag(sigma^2/n, sigma^2/(2n))``:
  ``Var F(x) = phi(u)^2 (1 + u^2/2)/n`` and
  ``Var x_p = sigma^2 (1 + z_p^2/2)/n``.
* Exact quantiles of the normal approximation for the normal family: with
  ``mu* ~ N(mu_hat, a^2)`` and ``sigma* ~ N(sigma_hat, b^2)`` independent,
  ``P(F(x; theta*) <= Phi(t)) = Phi((mu_hat + t sigma_hat - x)/sqrt(a^2 + t^2 b^2))``
  (up to the negligible mass of ``sigma* <= 0``); the simulated quantile
  band must agree with these quantiles within its Monte Carlo standard
  error ``sqrt(p (1 - p)/R)/g(q_p)``.
* Seeded coverage simulations judged by the binomial standard error, and
  structural properties (nested envelopes, rejection counts).

References:
    [EQ-INF-BAND-001], [EQ-INF-BAND-002], [EQ-INF-MLE-GEN-002].

"""

from __future__ import annotations

import dataclasses
import math
import unittest

from probstat.distributions import Normal
from probstat.foundations import RandomStream
from probstat.inference import (
    EXPONENTIAL_FAMILY,
    GEV_FAMILY,
    NORMAL_FAMILY,
    UNIFORM_FAMILY,
    WEIBULL_FAMILY,
    DistributionFamily,
    FitResult,
    fit_distribution,
    fit_distribution_moments,
    parameter_uncertainty_band,
)
from probstat.optimization import brent_root

STANDARD = Normal(0.0, 1.0)


def exact_normal_fit(mu: float, sigma: float, n: int) -> FitResult:
    """Return a normal ``FitResult`` with the exact asymptotic covariance.

    References:
        [EQ-INF-MLE-NORM-001]. Test helper.

    """
    return FitResult(
        family="Normal",
        parameter_names=("mu", "sigma"),
        estimates=(mu, sigma),
        standard_errors=(sigma / math.sqrt(n), sigma / math.sqrt(2 * n)),
        covariance=((sigma**2 / n, 0.0), (0.0, sigma**2 / (2 * n))),
        log_likelihood=0.0,
        sample_size=n,
        aic=0.0,
        aicc=0.0,
        bic=0.0,
        converged=True,
        method="closed-form",
        message="exact covariance",
        distribution=Normal(mu, sigma),
    )


class DeltaMethodBand(unittest.TestCase):
    """The delta-method band against closed forms."""

    def test_normal_cdf_standard_error(self) -> None:
        """The standard error of F(x) is phi(u) sqrt((1 + u^2/2)/n) to 1e-8."""
        mu, sigma, n = 20.0, 3.0, 400
        fit = exact_normal_fit(mu, sigma, n)
        grid = tuple(mu + sigma * u for u in (-2.5, -1.0, 0.0, 0.7, 2.0))
        band = parameter_uncertainty_band(fit, grid, draws=10, seed=1)
        for x, se, f in zip(
            grid, band.delta_standard_errors, band.cdf_estimate, strict=True
        ):
            u = (x - mu) / sigma
            density = math.exp(-u * u / 2) / math.sqrt(2 * math.pi)
            expected = density * math.sqrt((1 + u * u / 2) / n)
            self.assertLess(abs(se - expected) / expected, 1e-8)
            self.assertAlmostEqual(f, STANDARD.cdf(u), places=15)
        critical = STANDARD.quantile(0.975)
        for f, se, lo, hi in zip(
            band.cdf_estimate,
            band.delta_standard_errors,
            band.delta_lower,
            band.delta_upper,
            strict=True,
        ):
            self.assertAlmostEqual(lo, max(0.0, f - critical * se), places=15)
            self.assertAlmostEqual(hi, min(1.0, f + critical * se), places=15)

    def test_normal_quantile_standard_error(self) -> None:
        """The standard error of x_p is sigma sqrt((1 + z_p^2/2)/n) to 1e-8."""
        mu, sigma, n = 5.0, 2.0, 50
        fit = exact_normal_fit(mu, sigma, n)
        probabilities = (0.01, 0.5, 0.9)
        band = parameter_uncertainty_band(
            fit, (mu,), draws=10, seed=1, quantile_probabilities=probabilities
        )
        critical = STANDARD.quantile(0.975)
        for p, estimate, lo, hi in zip(
            probabilities,
            band.quantile_estimate,
            band.quantile_delta_lower,
            band.quantile_delta_upper,
            strict=True,
        ):
            z = STANDARD.quantile(p)
            se = sigma * math.sqrt((1 + z * z / 2) / n)
            self.assertAlmostEqual(estimate, mu + sigma * z, places=12)
            self.assertLess(abs((hi - lo) / (2 * critical) - se) / se, 1e-8)

    def test_observed_information_fit_matches_closed_form(self) -> None:
        """With the numerical-Hessian covariance the closed form holds to 1e-5."""
        values = Normal(20.0, 3.0).sample(400, seed=11)
        fit = fit_distribution(values, NORMAL_FAMILY)
        mu, sigma = fit.estimates
        grid = tuple(mu + sigma * u for u in (-2.0, 0.0, 1.5))
        band = parameter_uncertainty_band(fit, grid, draws=10, seed=1)
        for x, se in zip(grid, band.delta_standard_errors, strict=True):
            u = (x - mu) / sigma
            expected = math.exp(-u * u / 2) / math.sqrt(2 * math.pi)
            expected *= math.sqrt((1 + u * u / 2) / len(values))
            self.assertLess(abs(se - expected) / expected, 1e-5)


class SimulationBand(unittest.TestCase):
    """The draw-based band against the exact normal-approximation quantiles."""

    def setUp(self) -> None:
        """Fit a normal law to 400 seeded draws.

        References:
            [EQ-INF-MLE-NORM-001]. Test fixture.

        """
        values = Normal(20.0, 3.0).sample(400, seed=11)
        self.fit = fit_distribution(values, NORMAL_FAMILY)
        mu, sigma = self.fit.estimates
        self.grid = tuple(mu + sigma * u for u in (-2.0, -1.0, 0.0, 0.5, 1.5, 2.5))

    def exact_probability(self, value: float, x: float) -> float:
        """Return ``P(F(x; theta*) <= value)`` under the normal approximation.

        References:
            [EQ-INF-BAND-001]. Test helper.

        """
        mu, sigma = self.fit.estimates
        assert self.fit.covariance is not None
        a2, b2 = self.fit.covariance[0][0], self.fit.covariance[1][1]
        t = STANDARD.quantile(value)
        return STANDARD.cdf((mu + t * sigma - x) / math.sqrt(a2 + t * t * b2))

    def test_quantile_band_matches_exact_quantiles(self) -> None:
        """With R = 20000 every band end lies within 4 Monte Carlo SE."""
        draws = 20000
        band = parameter_uncertainty_band(self.fit, self.grid, draws=draws, seed=1)
        self.assertEqual(band.rejected_draws, 0)
        for index, x in enumerate(self.grid):
            for p, simulated in (
                (0.025, band.lower[index]),
                (0.975, band.upper[index]),
            ):
                exact = brent_root(
                    lambda u, x=x, p=p: self.exact_probability(u, x) - p,
                    1e-12,
                    1.0 - 1e-12,
                    tolerance=1e-15,
                ).root
                step = min(1e-6 * math.sqrt(exact * (1 - exact)), exact / 2)
                density = (
                    self.exact_probability(exact + step, x)
                    - self.exact_probability(exact - step, x)
                ) / (2 * step)
                mcse = math.sqrt(p * (1 - p) / draws) / density
                with self.subTest(x=x, p=p):
                    self.assertLess(abs(simulated - exact), 4.0 * mcse)

    def test_delta_and_simulation_bands_agree_where_f_is_near_linear(self) -> None:
        """For |u| <= 1 the two bands differ by less than 0.1 standard error."""
        band = parameter_uncertainty_band(self.fit, self.grid[1:4], draws=20000, seed=2)
        for lo, hi, dlo, dhi, se in zip(
            band.lower,
            band.upper,
            band.delta_lower,
            band.delta_upper,
            band.delta_standard_errors,
            strict=True,
        ):
            self.assertLess(abs(lo - dlo), 0.1 * se)
            self.assertLess(abs(hi - dhi), 0.1 * se)

    def test_envelope_widens_and_quantile_band_converges(self) -> None:
        """Envelopes of nested draw sets are nested; quantile bands settle."""
        small = parameter_uncertainty_band(self.fit, self.grid, draws=200, seed=3)
        large = parameter_uncertainty_band(self.fit, self.grid, draws=20000, seed=3)
        self.assertEqual(large.parameter_draws[:200], small.parameter_draws)
        for index in range(len(self.grid)):
            self.assertLessEqual(
                large.envelope_lower[index], small.envelope_lower[index]
            )
            self.assertGreaterEqual(
                large.envelope_upper[index], small.envelope_upper[index]
            )
        center = 2
        small_width = small.envelope_upper[center] - small.envelope_lower[center]
        large_width = large.envelope_upper[center] - large.envelope_lower[center]
        self.assertGreater(large_width, 1.2 * small_width)
        other = parameter_uncertainty_band(self.fit, self.grid, draws=20000, seed=4)
        width = large.upper[center] - large.lower[center]
        self.assertLess(
            abs(other.upper[center] - other.lower[center] - width), 0.03 * width
        )

    def test_pointwise_coverage(self) -> None:
        """Both bands cover the true F(x) at 95 % within 4 binomial SE."""
        truth = Normal(10.0, 2.0)
        root = RandomStream(5)
        points = tuple(10.0 + 2.0 * u for u in (-1.5, 0.0, 1.0))
        replications = 300
        delta_hits = [0, 0, 0]
        simulation_hits = [0, 0, 0]
        for r in range(replications):
            values = truth.sample(200, seed=root.child("data", r))
            fit = fit_distribution(values, NORMAL_FAMILY)
            band = parameter_uncertainty_band(
                fit, points, draws=300, seed=root.child("band", r)
            )
            for i, x in enumerate(points):
                target = truth.cdf(x)
                delta_hits[i] += band.delta_lower[i] <= target <= band.delta_upper[i]
                simulation_hits[i] += band.lower[i] <= target <= band.upper[i]
        tolerance = 4.0 * math.sqrt(0.95 * 0.05 / replications)
        for hits in (*delta_hits, *simulation_hits):
            self.assertLess(abs(hits / replications - 0.95), tolerance)


class DrawHandling(unittest.TestCase):
    """Parameter-space rejection, reproducibility, other fits, and errors."""

    def test_rejected_draws_are_counted(self) -> None:
        """A rate with se equal to its value loses about Phi(-1) of the draws."""
        fit = dataclasses.replace(
            exact_normal_fit(0.0, 1.0, 10),
            family="Exponential",
            parameter_names=("rate",),
            estimates=(1.0,),
            standard_errors=(1.0,),
            covariance=((1.0,),),
        )
        band = parameter_uncertainty_band(fit, (0.5, 1.0, 2.0), draws=4000, seed=9)
        self.assertEqual(band.accepted_draws + band.rejected_draws, 4000)
        self.assertTrue(all(theta[0] > 0.0 for theta in band.parameter_draws))
        expected = STANDARD.cdf(-1.0)
        se = math.sqrt(expected * (1 - expected) / 4000)
        self.assertLess(abs(band.rejected_fraction - expected), 4.0 * se)

    def test_reproducible_with_a_seed(self) -> None:
        """The same seed gives the same band."""
        values = Normal(3.0, 0.5).sample(80, seed=3)
        fit = fit_distribution(values, WEIBULL_FAMILY)
        first = parameter_uncertainty_band(
            fit, (2.5, 3.0), draws=100, seed=RandomStream(8)
        )
        second = parameter_uncertainty_band(
            fit, (2.5, 3.0), draws=100, seed=RandomStream(8)
        )
        self.assertEqual(first, second)

    def test_moment_fit_and_gev_quantiles(self) -> None:
        """A GEV moment fit gives ordered CDF and quantile bands."""
        values = Normal(20.0, 3.0).sample(300, seed=21)
        fit = fit_distribution_moments(values, GEV_FAMILY)
        band = parameter_uncertainty_band(
            fit,
            (16.0, 20.0, 24.0),
            draws=500,
            seed=4,
            quantile_probabilities=(0.5, 0.99),
        )
        for lo, f, hi in zip(band.lower, band.cdf_estimate, band.upper, strict=True):
            self.assertLess(lo, f)
            self.assertLess(f, hi)
        for lo, q, hi in zip(
            band.quantile_lower,
            band.quantile_estimate,
            band.quantile_upper,
            strict=True,
        ):
            self.assertLess(lo, q)
            self.assertLess(q, hi)
        self.assertEqual(band.family, "GEV")

    def test_errors(self) -> None:
        """Missing covariance, bad arguments, and family mismatches raise."""
        values = Normal(1.0, 0.5).sample(30, seed=3)
        uniform = fit_distribution(values, UNIFORM_FAMILY)
        with self.assertRaisesRegex(ValueError, "no covariance"):
            parameter_uncertainty_band(uniform, (1.0,), seed=1)
        fit = fit_distribution(values, NORMAL_FAMILY)
        with self.assertRaisesRegex(ValueError, "draws"):
            parameter_uncertainty_band(fit, (1.0,), draws=1, seed=1)
        with self.assertRaisesRegex(ValueError, "level"):
            parameter_uncertainty_band(fit, (1.0,), level=1.0, seed=1)
        with self.assertRaisesRegex(ValueError, "quantile probabilities"):
            parameter_uncertainty_band(
                fit, (1.0,), quantile_probabilities=(0.0,), seed=1
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            parameter_uncertainty_band(fit, (1.0,), family=EXPONENTIAL_FAMILY, seed=1)
        custom = DistributionFamily(
            "MyNormal",
            ("mu", "sigma"),
            ("location", "positive"),
            lambda p: Normal(p[0], p[1]),
            lambda s: (0.0, 1.0),
        )
        renamed = dataclasses.replace(fit, family="MyNormal")
        with self.assertRaisesRegex(ValueError, "pass family="):
            parameter_uncertainty_band(renamed, (1.0,), seed=1)
        band = parameter_uncertainty_band(
            renamed, (1.0,), family=custom, draws=50, seed=1
        )
        self.assertEqual(band.accepted_draws, 50)
        with self.assertRaises(TypeError):
            parameter_uncertainty_band("fit", (1.0,))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
