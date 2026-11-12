"""Time series: correlograms, ARIMA, smoothing, GARCH, VAR, unit roots, change points.

``descriptive`` provides the sample ACF/PACF (Durbin-Levinson), white-noise and
Bartlett bands, the Ljung-Box test, differencing, and classical decomposition;
``polynomials`` provides lag-polynomial roots (Durand-Kerner), the Schur-Cohn
stationarity / invertibility test, and the partial-autocorrelation
reparameterization; ``arma`` provides the Harvey state-space form, the exact
Kalman-filter likelihood, Yule-Walker and maximum-likelihood ARIMA estimation,
forecasting with exact prediction intervals, residual diagnostics, and
simulation; ``smoothing`` provides simple and Holt exponential smoothing;
``garch`` provides GARCH(p, q) volatility models with normal or Student-t
innovations; ``var`` provides vector autoregressions with order selection,
stability, impulse responses, variance decompositions, and Granger tests;
``unit_root`` provides the augmented Dickey-Fuller and KPSS tests; and
``changepoint`` provides the CUSUM mean-shift test and PELT segmentation.

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
    ``fit_garch``: GARCH(p, q) by maximum likelihood with stationarity
        constraints; ``GARCHFit.forecast`` and ``GARCHFit.diagnostics``.
    ``fit_var`` / ``select_var_order``: VAR(p) least squares and AIC/BIC/HQ/
        FPE order selection; ``VARFit.impulse_response`` (with bootstrap
        bands from ``VARFit.impulse_response_bands``), ``VARFit.fevd``,
        ``VARFit.granger_causality``, ``VARFit.is_stable``.
    ``adf_test``: augmented Dickey-Fuller unit-root test (MacKinnon 2010
        critical values, simulated p-value).
    ``kpss_test``: KPSS stationarity test with an exact asymptotic p-value.
    ``cusum_test``: CUSUM test for one mean shift.
    ``pelt``: multiple change points in mean and/or variance (BIC/MBIC).

Assumptions:
    Observations are equally spaced and complete. After ``d`` differences
    the series is a stationary, invertible ARMA process with Gaussian
    innovations; the likelihood, the standard errors, and the prediction
    intervals rely on that model. GARCH, VAR, and the unit-root tests
    condition on the first observations (pre-sample values are backcast or
    dropped); change-point costs assume independent Gaussian segments.

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
    - ``fit_garch(series, p, q)`` follows Bollerslev: ``q`` is the ARCH
      order and ``p`` the GARCH order.
    - Orthogonalized impulse responses depend on the order of the
      variables (Cholesky factor).
    - ADF tests the null of a unit root, KPSS the null of stationarity;
      regressing one random walk on another gives spurious significance.
    - A level shift can look like volatility clustering or a unit root;
      check for change points before fitting GARCH or differencing.

Guide: docs/foundations/time-series.md

References:
    [REF-BOX-JENKINS-2015], [REF-BROCKWELL-DAVIS-2016],
    [REF-DURBIN-KOOPMAN-2012], [REF-LJUNG-BOX-1978],
    [REF-BOLLERSLEV-1986], [REF-LUTKEPOHL-2005], [REF-MACKINNON-2010],
    [REF-KPSS-1992], [REF-KILLICK-2012].

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
from .changepoint import (
    ChangePointResult,
    CUSUMResult,
    Segment,
    cusum_test,
    kolmogorov_sf,
    pelt,
    robust_noise_variance,
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
from .garch import (
    ARCHTestResult,
    GARCHDiagnostics,
    GARCHFit,
    GARCHForecast,
    GARCHSimulation,
    arch_lm_test,
    fit_garch,
    garch_log_likelihood,
    garch_variances,
    simulate_garch,
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
from .unit_root import (
    ADFResult,
    KPSSResult,
    adf_critical_values,
    adf_simulated_p_value,
    adf_test,
    kpss_asymptotic_sf,
    kpss_critical_values,
    kpss_test,
    newey_west_variance,
)
from .var import (
    GrangerCausalityResult,
    ImpulseResponseBands,
    VARFit,
    VARForecast,
    VAROrderSelection,
    fit_var,
    select_var_order,
    simulate_var,
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
    "ADFResult",
    "ARCHTestResult",
    "CUSUMResult",
    "ChangePointResult",
    "GARCHDiagnostics",
    "GARCHFit",
    "GARCHForecast",
    "GARCHSimulation",
    "GrangerCausalityResult",
    "KPSSResult",
    "Segment",
    "VARFit",
    "ImpulseResponseBands",
    "VARForecast",
    "VAROrderSelection",
    "adf_critical_values",
    "adf_simulated_p_value",
    "adf_test",
    "arch_lm_test",
    "cusum_test",
    "fit_garch",
    "fit_var",
    "garch_log_likelihood",
    "garch_variances",
    "kolmogorov_sf",
    "kpss_asymptotic_sf",
    "kpss_critical_values",
    "kpss_test",
    "newey_west_variance",
    "pelt",
    "robust_noise_variance",
    "select_var_order",
    "simulate_garch",
    "simulate_var",
]
