"""Univariate time-series analysis: correlograms, ARIMA, and exponential smoothing.

``descriptive`` provides the sample ACF/PACF (Durbin-Levinson), white-noise and
Bartlett bands, the Ljung-Box test, differencing, and classical decomposition;
``polynomials`` provides lag-polynomial roots (Durand-Kerner), the Schur-Cohn
stationarity / invertibility test, and the partial-autocorrelation
reparameterization; ``arma`` provides the Harvey state-space form, the exact
Kalman-filter likelihood, Yule-Walker and maximum-likelihood ARIMA estimation,
forecasting with exact prediction intervals, residual diagnostics, and
simulation; ``smoothing`` provides simple and Holt exponential smoothing.
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
