"""Univariate time-series analysis: correlograms, ARIMA, and exponential smoothing.

``descriptive`` provides the sample ACF/PACF (Durbin-Levinson), white-noise and
Bartlett bands, the Ljung-Box test, differencing, and classical decomposition;
``polynomials`` provides lag-polynomial roots (Durand-Kerner), the Schur-Cohn
stationarity / invertibility test, and the partial-autocorrelation
reparameterization; ``arma`` provides the Harvey state-space form, the exact
Kalman-filter likelihood, Yule-Walker and maximum-likelihood ARIMA estimation,
forecasting with exact prediction intervals, residual diagnostics, and
simulation; ``smoothing`` provides simple and Holt exponential smoothing.

Main entry points:
    ``acf``: sample autocorrelation function (``1/n`` divisor).
    ``pacf``: sample partial autocorrelations by Durbin-Levinson.
    ``ljung_box``: portmanteau test for residual autocorrelation.
    ``fit_arima``: exact Gaussian maximum-likelihood ARIMA(p, d, q) fit.
    ``forecast_arima``: forecasts with exact Gaussian prediction intervals
        given the parameters.
    ``simulate_arma``: seeded ARMA simulation with a burn-in.
    ``simple_exponential_smoothing``: level-only exponential smoothing.
    ``holt_linear``: exponential smoothing with a linear trend.

Assumptions:
    Observations are equally spaced and complete. After ``d`` differences
    the series is a stationary, invertible ARMA process with Gaussian
    innovations; the likelihood, the standard errors, and the prediction
    intervals rely on that model.

Common pitfalls:
    - The AR polynomial is ``1 - sum phi_j z^j`` and the MA polynomial
      ``1 + sum theta_j z^j``; MA coefficients from a source that writes
      the MA part with a minus sign change sign.
    - ``ljung_box`` defaults to ``fitted_parameters=0``; on ARMA residuals
      pass ``p + q`` (``ARIMAFit.diagnostics`` does it).
    - ``fit_arima`` estimates a mean only for ``d = 0`` unless
      ``include_mean`` is given, so a differenced model has no drift by
      default.
    - Forecast intervals omit parameter-estimation uncertainty, and the
      smoothing functions return point forecasts only.

Guide: docs/foundations/time-series.md

References:
    [REF-BOX-JENKINS-2015], [REF-BROCKWELL-DAVIS-2016],
    [REF-DURBIN-KOOPMAN-2012], [REF-LJUNG-BOX-1978].

"""

from .arma import (
    ARIMAFit,
    ARIMAForecast,
    ARMAStateSpace,
    ResidualDiagnostics,
    YuleWalkerResult,
    arma_autocovariance,
    arma_log_likelihood,
    arma_state_space,
    fit_arima,
    forecast_arima,
    simulate_arima,
    simulate_arma,
    yule_walker,
)
from .descriptive import (
    ClassicalDecomposition,
    DurbinLevinsonResult,
    LjungBoxResult,
    acf,
    autocovariance,
    bartlett_band,
    centered_moving_average,
    classical_decomposition,
    difference,
    durbin_levinson,
    ljung_box,
    pacf,
    seasonal_difference,
    undifference,
    white_noise_band,
)
from .polynomials import (
    RootCheck,
    ar_root_check,
    coefficients_to_pacf,
    is_invertible,
    is_stationary,
    ma_root_check,
    pacf_to_coefficients,
    polynomial_roots,
    psi_weights,
)
from .smoothing import (
    ExponentialSmoothingFit,
    holt_linear,
    simple_exponential_smoothing,
)

__all__ = [
    "ARIMAFit",
    "ARIMAForecast",
    "ARMAStateSpace",
    "ClassicalDecomposition",
    "DurbinLevinsonResult",
    "ExponentialSmoothingFit",
    "LjungBoxResult",
    "ResidualDiagnostics",
    "RootCheck",
    "YuleWalkerResult",
    "acf",
    "ar_root_check",
    "arma_autocovariance",
    "arma_log_likelihood",
    "arma_state_space",
    "autocovariance",
    "bartlett_band",
    "centered_moving_average",
    "classical_decomposition",
    "coefficients_to_pacf",
    "difference",
    "durbin_levinson",
    "fit_arima",
    "forecast_arima",
    "holt_linear",
    "is_invertible",
    "is_stationary",
    "ljung_box",
    "ma_root_check",
    "pacf",
    "pacf_to_coefficients",
    "polynomial_roots",
    "psi_weights",
    "seasonal_difference",
    "simple_exponential_smoothing",
    "simulate_arima",
    "simulate_arma",
    "undifference",
    "white_noise_band",
    "yule_walker",
]
