"""Tests for correlograms, ARIMA state-space likelihood, and exponential smoothing."""

from __future__ import annotations

import cmath
import math
import unittest

from probstat.filtering import KalmanFilterResult, kalman_filter
from probstat.timeseries import (
    ARIMAFit,
    acf,
    ar_root_check,
    arma_autocovariance,
    arma_log_likelihood,
    arma_state_space,
    autocovariance,
    bartlett_band,
    classical_decomposition,
    coefficients_to_pacf,
    difference,
    durbin_levinson,
    fit_arima,
    forecast_arima,
    holt_linear,
    is_invertible,
    is_stationary,
    ljung_box,
    ma_root_check,
    pacf,
    pacf_to_coefficients,
    polynomial_roots,
    psi_weights,
    seasonal_difference,
    simple_exponential_smoothing,
    simulate_arima,
    simulate_arma,
    undifference,
    white_noise_band,
    yule_walker,
)


def ar1_closed_form_log_likelihood(
    series: tuple[float, ...], phi: float, variance: float, mean: float
) -> float:
    """Return the exact AR(1) Gaussian log-likelihood in closed form.

    The first observation is ``N(mu, sigma^2 / (1 - phi^2))``; later ones are
    conditionally ``N(mu + phi (y_(t-1) - mu), sigma^2)``.

    Args:
        series: Observations.
        phi: Stationary AR coefficient.
        variance: Innovation variance.
        mean: Process mean.

    Returns:
        Exact log-likelihood.

    References:
        [REF-HAMILTON-1994, Chapter 5]; equation [EQ-TS-LIK-002].

    """
    first = series[0] - mean
    stationary = variance / (1.0 - phi * phi)
    value = -0.5 * (math.log(2.0 * math.pi * stationary) + first * first / stationary)
    for previous, current in zip(series, series[1:], strict=False):
        residual = current - mean - phi * (previous - mean)
        value -= 0.5 * (math.log(2.0 * math.pi * variance) + residual**2 / variance)
    return value


class DescriptiveTests(unittest.TestCase):
    """Sample ACF/PACF, bands, Ljung-Box, differencing, and decomposition."""

    def test_ar1_sample_acf_matches_phi_power(self) -> None:
        """The ACF of a long AR(1) simulation is close to ``phi^k``."""
        phi = 0.7
        series = simulate_arma((phi,), (), 4000, seed=101)
        rho = acf(series, 6)
        self.assertEqual(rho[0], 1.0)
        for k in range(1, 7):
            self.assertAlmostEqual(rho[k], phi**k, delta=0.06)

    def test_theoretical_ar1_autocovariance(self) -> None:
        """The state-space autocovariance equals ``sigma^2 phi^k / (1 - phi^2)``."""
        gamma = arma_autocovariance((0.6,), (), 5, variance=2.0)
        for k, value in enumerate(gamma):
            self.assertAlmostEqual(value, 2.0 * 0.6**k / (1.0 - 0.36), places=12)

    def test_theoretical_ma1_acf_cuts_off(self) -> None:
        """An MA(1) has ``rho(1) = theta / (1 + theta^2)`` and zero beyond."""
        gamma = arma_autocovariance((), (0.6,), 4)
        self.assertAlmostEqual(gamma[0], 1.36, places=12)
        self.assertAlmostEqual(gamma[1] / gamma[0], 0.6 / 1.36, places=12)
        for value in gamma[2:]:
            self.assertAlmostEqual(value, 0.0, places=14)

    def test_durbin_levinson_pacf_cuts_off_for_ar2(self) -> None:
        """The theoretical AR(2) PACF is ``phi_2`` at lag 2 and zero beyond."""
        gamma = arma_autocovariance((0.5, -0.3), (), 8)
        result = durbin_levinson([value / gamma[0] for value in gamma])
        self.assertAlmostEqual(result.partial_autocorrelations[0], 0.5 / 1.3, places=12)
        self.assertAlmostEqual(result.partial_autocorrelations[1], -0.3, places=12)
        for value in result.partial_autocorrelations[2:]:
            self.assertAlmostEqual(value, 0.0, places=12)
        self.assertEqual(len(result.coefficients[1]), 2)
        self.assertAlmostEqual(result.coefficients[1][0], 0.5, places=12)

    def test_sample_pacf_cut_off(self) -> None:
        """Sample AR(2) PACF: lags 1-2 exceed the band, lags 3-10 stay small."""
        n = 2000
        series = simulate_arma((0.5, -0.3), (), n, seed=102)
        partial = pacf(series, 10)
        band = white_noise_band(n)
        self.assertGreater(abs(partial[0]), band)
        self.assertGreater(abs(partial[1]), band)
        for value in partial[2:]:
            self.assertLess(abs(value), 3.0 / math.sqrt(n))

    def test_bands(self) -> None:
        """White-noise and Bartlett half-widths follow their formulas."""
        self.assertAlmostEqual(white_noise_band(100), 0.1959963984540054, places=12)
        widths = bartlett_band((1.0, 0.5, 0.2), 100)
        self.assertAlmostEqual(widths[0], 0.1959963984540054, places=12)
        self.assertAlmostEqual(widths[1], 0.1959963984540054 * math.sqrt(1.5))

    def test_ljung_box_white_noise_versus_ar(self) -> None:
        """Ljung-Box accepts white noise and strongly rejects an AR(1)."""
        noise = simulate_arma((), (), 300, seed=103)
        dependent = simulate_arma((0.6,), (), 300, seed=104)
        accepted = ljung_box(noise, 10)
        rejected = ljung_box(dependent, 10)
        self.assertGreater(accepted.p_value, 0.05)
        self.assertLess(rejected.p_value, 1e-6)
        rho = acf(noise, 10)
        manual = 300 * 302 * math.fsum(rho[k] ** 2 / (300 - k) for k in range(1, 11))
        self.assertAlmostEqual(accepted.statistic, manual, places=10)
        adjusted = ljung_box(noise, 10, fitted_parameters=2)
        self.assertEqual(adjusted.degrees_of_freedom, 8)

    def test_differencing_round_trip(self) -> None:
        """Differencing then undifferencing restores the series."""
        series = (1.0, 4.0, 9.0, 16.0, 25.0, 36.0)
        self.assertEqual(difference(series), (3.0, 5.0, 7.0, 9.0, 11.0))
        self.assertEqual(difference(series, 2), (2.0, 2.0, 2.0, 2.0))
        restored = undifference(difference(series, 2), series[:2], 2)
        for a, b in zip(restored, series, strict=True):
            self.assertAlmostEqual(a, b, places=12)
        seasonal = seasonal_difference((1.0, 2.0, 3.0, 11.0, 12.0, 13.0), 3)
        self.assertEqual(seasonal, (10.0, 10.0, 10.0))

    def test_additive_decomposition_recovers_components(self) -> None:
        """A linear trend plus a zero-sum season is recovered exactly."""
        season = (2.0, -1.0, 0.5, -1.5)
        series = [10.0 + 0.5 * t + season[t % 4] for t in range(24)]
        result = classical_decomposition(series, 4)
        for got, expected in zip(result.seasonal_indices, season, strict=True):
            self.assertAlmostEqual(got, expected, places=12)
        self.assertIsNone(result.trend[0])
        self.assertAlmostEqual(result.trend[2] or 0.0, 11.0, places=12)
        for value in result.remainder:
            if value is not None:
                self.assertAlmostEqual(value, 0.0, places=12)

    def test_multiplicative_decomposition_indices_average_one(self) -> None:
        """Multiplicative indices average one and track the true factors."""
        factors = (1.2, 0.9, 1.1, 0.8)
        series = [(10.0 + 0.5 * t) * factors[t % 4] for t in range(40)]
        result = classical_decomposition(series, 4, model="multiplicative")
        self.assertAlmostEqual(sum(result.seasonal_indices) / 4, 1.0, places=12)
        for got, expected in zip(result.seasonal_indices, factors, strict=True):
            self.assertAlmostEqual(got, expected, delta=0.01)

    def test_descriptive_errors(self) -> None:
        """Invalid descriptive inputs fail explicitly."""
        with self.assertRaises(ValueError):
            autocovariance((1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            acf((1.0, 2.0, 3.0), 3)
        with self.assertRaises(ValueError):
            ljung_box(simulate_arma((), (), 50, seed=1), 3, fitted_parameters=3)
        with self.assertRaises(ValueError):
            classical_decomposition((1.0, 2.0, 3.0), 4)
        with self.assertRaises(ValueError):
            classical_decomposition([1.0, -1.0] * 8, 2, model="multiplicative")
        with self.assertRaises(ValueError):
            durbin_levinson((1.0, 1.0, 0.5))
        with self.assertRaises(TypeError):
            difference((1.0, 2.0, 3.0), 1.5)  # type: ignore[arg-type]


class PolynomialTests(unittest.TestCase):
    """Root finding, stationarity, invertibility, and the PACF transform."""

    def test_durand_kerner_finds_complex_roots(self) -> None:
        """Roots of ``(z - 2)(z^2 + 1)`` are found to high accuracy."""
        roots = polynomial_roots((-2.0, 1.0, -2.0, 1.0))
        expected = (1j, -1j, 2.0 + 0j)
        for target in expected:
            self.assertLess(min(abs(root - target) for root in roots), 1e-12)

    def test_ar2_complex_roots_and_step_down_agree(self) -> None:
        """``1 - 0.5 z + 0.3 z^2`` has conjugate roots of modulus ``sqrt(1/0.3)``."""
        check = ar_root_check((0.5, -0.3))
        self.assertTrue(check.satisfied)
        self.assertEqual(len(check.roots), 2)
        self.assertAlmostEqual(check.roots[0], check.roots[1].conjugate(), places=12)
        self.assertAlmostEqual(check.minimum_modulus, math.sqrt(1 / 0.3), places=12)
        self.assertNotEqual(cmath.phase(check.roots[0]), 0.0)

    def test_stationarity_rejection(self) -> None:
        """Explosive and unit-root AR polynomials are rejected everywhere."""
        for ar in ((1.2,), (1.0,), (0.5, 0.5), (0.2, 0.9)):
            self.assertFalse(is_stationary(ar))
            self.assertLessEqual(ar_root_check(ar).minimum_modulus, 1.0 + 1e-9)
            with self.assertRaises(ValueError):
                arma_state_space(ar)
            with self.assertRaises(ValueError):
                simulate_arma(ar, (), 10, seed=1)
            with self.assertRaises(ValueError):
                arma_log_likelihood((0.1, 0.2, 0.3), ar)
        self.assertTrue(is_stationary((1.2, -0.5)))

    def test_invertibility(self) -> None:
        """An MA root inside the unit circle is detected."""
        self.assertTrue(is_invertible((0.6,)))
        self.assertFalse(is_invertible((1.5,)))
        self.assertFalse(ma_root_check((1.5,)).satisfied)
        self.assertAlmostEqual(ma_root_check((1.5,)).roots[0].real, -1 / 1.5)

    def test_pacf_transform_round_trip(self) -> None:
        """The step-up and step-down maps are mutual inverses."""
        reflections = (0.7, -0.4, 0.2)
        coefficients = pacf_to_coefficients(reflections)
        self.assertTrue(is_stationary(coefficients))
        back = coefficients_to_pacf(coefficients)
        assert back is not None
        for a, b in zip(back, reflections, strict=True):
            self.assertAlmostEqual(a, b, places=12)
        with self.assertRaises(ValueError):
            pacf_to_coefficients((1.0,))

    def test_psi_weights_of_arma11(self) -> None:
        """ARMA(1,1) weights are ``(phi + theta) phi^(j-1)``."""
        weights = psi_weights((0.5,), (0.3,), 6)
        self.assertEqual(weights[0], 1.0)
        for j in range(1, 6):
            self.assertAlmostEqual(weights[j], 0.8 * 0.5 ** (j - 1), places=12)


class LikelihoodTests(unittest.TestCase):
    """Exact Kalman-filter likelihood and its agreement with closed forms."""

    def test_ar1_kalman_likelihood_equals_closed_form(self) -> None:
        """Kalman exact likelihood of an AR(1) equals the closed form."""
        series = simulate_arma((0.7,), (), 200, variance=0.8, mean=0.3, seed=105)
        exact = ar1_closed_form_log_likelihood(series, 0.7, 0.8, 0.3)
        fast = arma_log_likelihood(series, (0.7,), variance=0.8, mean=0.3)
        model = arma_state_space((0.7,), variance=0.8, mean=0.3)
        record = model.kalman(series)
        self.assertIsInstance(record, KalmanFilterResult)
        self.assertAlmostEqual(fast, exact, places=9)
        self.assertAlmostEqual(record.log_likelihood, exact, places=9)

    def test_scalar_and_general_filters_agree_for_arma(self) -> None:
        """The O(r^2) estimation filter equals the general-matrix filter."""
        series = simulate_arma((0.2, 0.1, -0.3), (0.3, -0.2), 150, seed=106)
        model = arma_state_space((0.2, 0.1, -0.3), (0.3, -0.2), variance=1.3)
        fast = arma_log_likelihood(series, (0.2, 0.1, -0.3), (0.3, -0.2), variance=1.3)
        self.assertAlmostEqual(fast, model.kalman(series).log_likelihood, places=8)

    def test_noisy_arma_matches_filtering_kalman_filter(self) -> None:
        """With measurement noise the path equals ``filtering.kalman_filter``."""
        model = arma_state_space((0.5, -0.3), (0.4,), variance=1.3, mean=0.7)
        series = simulate_arma((0.5, -0.3), (0.4,), 80, variance=1.3, mean=0.7, seed=9)
        ours = model.kalman(series, observation_variance=0.25)
        reference = kalman_filter(
            model.to_linear_gaussian(0.25), [(value - 0.7,) for value in series]
        )
        self.assertAlmostEqual(ours.log_likelihood, reference.log_likelihood, places=10)
        for a, b in zip(ours.means, reference.means, strict=True):
            self.assertAlmostEqual(a[0], b[0], places=10)
        with self.assertRaises(ValueError):
            model.to_linear_gaussian(0.0)


class EstimationTests(unittest.TestCase):
    """Yule-Walker, maximum likelihood, diagnostics, and forecasting."""

    def test_yule_walker_ar1(self) -> None:
        """Yule-Walker recovers an AR(1) coefficient and is stationary."""
        series = simulate_arma((0.6,), (), 1500, variance=2.0, seed=107)
        result = yule_walker(series, 1)
        self.assertAlmostEqual(result.coefficients[0], 0.6, delta=4 * 0.0207)
        self.assertAlmostEqual(result.variance, 2.0, delta=0.2)
        self.assertTrue(is_stationary(result.coefficients))

    def test_mle_recovers_ar2(self) -> None:
        """Exact MLE recovers AR(2) coefficients within three standard errors."""
        truth = (0.5, -0.3)
        series = simulate_arma(truth, (), 800, mean=1.0, seed=108)
        fit = fit_arima(series, (2, 0, 0))
        self.assertIsInstance(fit, ARIMAFit)
        self.assertTrue(fit.converged)
        assert fit.standard_errors is not None
        for estimate, error, target in zip(
            fit.ar, fit.standard_errors, truth, strict=False
        ):
            self.assertLess(abs(estimate - target), 3.0 * error)
            self.assertLess(error, 0.06)
        self.assertEqual(fit.parameter_names, ("ar1", "ar2", "mean", "variance"))
        self.assertAlmostEqual(fit.aic, 2 * 4 - 2 * fit.log_likelihood, places=9)
        exact = arma_log_likelihood(
            series, fit.ar, variance=fit.variance, mean=fit.mean or 0.0
        )
        self.assertAlmostEqual(fit.log_likelihood, exact, places=8)
        self.assertGreater(fit.diagnostics(10).ljung_box.p_value, 0.01)

    def test_mle_recovers_ma1(self) -> None:
        """Exact MLE recovers an MA(1) coefficient within three standard errors."""
        series = simulate_arma((), (0.6,), 800, seed=109)
        fit = fit_arima(series, (0, 0, 1), include_mean=False)
        assert fit.standard_errors is not None
        self.assertLess(abs(fit.ma[0] - 0.6), 3.0 * fit.standard_errors[0])
        self.assertTrue(is_invertible(fit.ma))
        self.assertAlmostEqual(fit.variance, 1.0, delta=0.15)

    def test_random_walk_intervals_widen_as_sqrt_h(self) -> None:
        """ARIMA(0,1,0) forecasts stay at the last value; widths grow as sqrt(h)."""
        series = simulate_arima((), 1, (), 300, seed=110)
        fit = fit_arima(series, (0, 1, 0))
        forecast = forecast_arima(fit, 8)
        base = forecast.standard_errors[0]
        self.assertAlmostEqual(base, math.sqrt(fit.variance), places=12)
        for h in range(1, 9):
            self.assertAlmostEqual(forecast.mean[h - 1], series[-1], places=12)
            self.assertAlmostEqual(
                forecast.standard_errors[h - 1], base * math.sqrt(h), places=12
            )
            half = forecast.upper[h - 1] - forecast.mean[h - 1]
            self.assertAlmostEqual(half, 1.959963984540054 * base * math.sqrt(h))

    def test_ar1_forecast_matches_closed_form(self) -> None:
        """AR(1) h-step mean ``mu + phi^h (y_n - mu)``, variance ``sum phi^2j``."""
        series = simulate_arma((0.6,), (), 400, mean=2.0, seed=111)
        fit = fit_arima(series, (1, 0, 0))
        phi, mu, s2 = fit.ar[0], fit.mean or 0.0, fit.variance
        forecast = fit.forecast(4)
        for h in range(1, 5):
            mean = mu + phi**h * (series[-1] - mu)
            variance = s2 * sum(phi ** (2 * j) for j in range(h))
            self.assertAlmostEqual(forecast.mean[h - 1], mean, places=10)
            self.assertAlmostEqual(
                forecast.standard_errors[h - 1], math.sqrt(variance), places=10
            )

    def test_estimation_errors(self) -> None:
        """Invalid estimation inputs fail explicitly."""
        with self.assertRaises(ValueError):
            fit_arima((1.0, 2.0, 3.0), (2, 0, 2))
        with self.assertRaises(ValueError):
            yule_walker((1.0, 2.0, 3.0), 3)
        series = simulate_arima((), 1, (), 50, seed=2)
        with self.assertRaises(ValueError):
            fit_arima(series, (0, 1, 0)).forecast(0)
        with self.assertRaises(TypeError):
            forecast_arima("fit", 2)  # type: ignore[arg-type]


class SimulationAndSmoothingTests(unittest.TestCase):
    """Seeded simulation and exponential smoothing."""

    def test_simulation_is_seeded(self) -> None:
        """The same seed reproduces a path; different seeds differ."""
        first = simulate_arma((0.5,), (0.2,), 20, seed=7)
        self.assertEqual(first, simulate_arma((0.5,), (0.2,), 20, seed=7))
        self.assertNotEqual(first, simulate_arma((0.5,), (0.2,), 20, seed=8))
        integrated = simulate_arima((0.5,), 1, (0.2,), 20, seed=7)
        self.assertAlmostEqual(integrated[-1], math.fsum(first), places=10)

    def test_simple_smoothing_on_constant_plus_noise(self) -> None:
        """For a constant plus noise, alpha is small and forecasts hit the level."""
        series = simulate_arma((), (), 200, mean=5.0, seed=112)
        fit = simple_exponential_smoothing(series)
        self.assertLess(fit.alpha, 0.1)
        self.assertAlmostEqual(fit.forecast(3)[2], 5.0, delta=0.25)
        self.assertIsNone(fit.beta)
        fixed = simple_exponential_smoothing(series, alpha=0.5)
        self.assertEqual(fixed.alpha, 0.5)
        self.assertGreaterEqual(fixed.sse, fit.sse)

    def test_holt_recovers_linear_trend(self) -> None:
        """Holt's method recovers the slope of a noisy linear trend."""
        noise = simulate_arma((), (), 150, variance=0.25, seed=113)
        series = [3.0 + 0.7 * t + e for t, e in enumerate(noise)]
        fit = holt_linear(series)
        assert fit.trend is not None
        self.assertAlmostEqual(fit.trend, 0.7, delta=0.02)
        forecast = fit.forecast(2)
        self.assertAlmostEqual(forecast[1] - forecast[0], fit.trend, places=12)

    def test_smoothing_errors(self) -> None:
        """Invalid smoothing inputs fail explicitly."""
        with self.assertRaises(ValueError):
            simple_exponential_smoothing((1.0, 2.0))
        with self.assertRaises(ValueError):
            simple_exponential_smoothing((1.0, 2.0, 3.0), alpha=1.0)
        with self.assertRaises(ValueError):
            holt_linear((1.0, 2.0, 3.0, 4.0), beta=0.0)


if __name__ == "__main__":
    unittest.main()
