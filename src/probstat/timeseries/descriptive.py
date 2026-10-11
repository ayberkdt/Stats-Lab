"""Descriptive time-series tools: correlograms, portmanteau tests, decomposition.

The sample autocorrelation function (ACF) and partial autocorrelation function
(PACF) are the identification tools of the Box-Jenkins approach: an MA(q)
process has an ACF that cuts off after lag ``q``, and an AR(p) process has a
PACF that cuts off after lag ``p``. The PACF is computed from the ACF by the
Durbin-Levinson recursion, which also solves the Yule-Walker equations for
every order at once.

All estimators use the ``1/n`` (biased) autocovariance, which keeps the sample
autocovariance matrix positive semidefinite, so Durbin-Levinson never produces
a partial autocorrelation outside ``[-1, 1]``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inference import chi_square_sf
from probstat.modeling import standard_normal_quantile

Vector = tuple[float, ...]


def _series(
    values: Iterable[object], *, minimum: int = 2, name: str = "series"
) -> Vector:
    """Materialize a finite series with a minimum length.

    Args:
        values: Candidate observations in time order.
        minimum: Minimum number of observations.
        name: Name used in error messages.

    Returns:
        Tuple of finite floats.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If the series is too short or non-finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    data = coerce_finite_sample(values, name=name)
    if len(data) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} observations")
    return data


def _max_lag(max_lag: int | None, n: int) -> int:
    """Validate a maximum lag against a series length.

    Args:
        max_lag: Requested maximum lag or ``None`` for ``min(n - 1, 10 log10 n)``.
        n: Series length.

    Returns:
        Validated maximum lag in ``[1, n - 1]``.

    Raises:
        TypeError: If ``max_lag`` is not an integer.
        ValueError: If ``max_lag`` is zero or not below ``n``.

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 1]. No equation is evaluated.

    """
    if max_lag is None:
        return max(1, min(n - 1, int(10.0 * math.log10(n))))
    lag = require_non_negative_integer(max_lag, name="max_lag")
    if lag < 1 or lag >= n:
        raise ValueError("max_lag must satisfy 1 <= max_lag < len(series)")
    return lag


def _level(level: float) -> float:
    """Validate a two-sided confidence level.

    Args:
        level: Candidate level strictly inside ``(0, 1)``.

    Returns:
        The validated level.

    Raises:
        ValueError: If the level is not strictly inside ``(0, 1)``.

    References:
        Project validation policy. No equation is evaluated.

    """
    value = require_finite_real(level, name="level")
    if not 0.0 < value < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    return value


def autocovariance(series: Iterable[float], max_lag: int | None = None) -> Vector:
    """Return the sample autocovariances ``gamma_hat(0..max_lag)``.

    ``gamma_hat(h) = (1/n) sum_{t=1}^{n-h} (x_t - x_bar)(x_{t+h} - x_bar)``.

    Args:
        series: At least two finite observations in time order.
        max_lag: Largest lag (default ``min(n - 1, 10 log10 n)``).

    Returns:
        Autocovariances for lags ``0..max_lag``.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If the series is too short, constant, or ``max_lag`` is
            outside ``[1, n - 1]``.

    Equation:
        [EQ-TS-ACF-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 1], [REF-BOX-JENKINS-2015, Chapter 2].

    """
    data = _series(series)
    n = len(data)
    lag = _max_lag(max_lag, n)
    centre = math.fsum(data) / n
    centred = [value - centre for value in data]
    values = tuple(
        math.fsum(centred[t] * centred[t + h] for t in range(n - h)) / n
        for h in range(lag + 1)
    )
    if values[0] <= 0.0:
        raise ValueError("series must not be constant")
    return values


def acf(series: Iterable[float], max_lag: int | None = None) -> Vector:
    """Return the sample autocorrelations ``rho_hat(0..max_lag)``.

    Args:
        series: At least two finite, non-constant observations.
        max_lag: Largest lag (default ``min(n - 1, 10 log10 n)``).

    Returns:
        Autocorrelations with ``rho_hat(0) = 1``.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If the series is too short or constant, or ``max_lag`` is
            invalid.

    Equation:
        [EQ-TS-ACF-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 1], [REF-BOX-JENKINS-2015, Chapter 2].

    """
    gamma = autocovariance(series, max_lag)
    return tuple(value / gamma[0] for value in gamma)


@dataclass(frozen=True, slots=True)
class DurbinLevinsonResult:
    """All Yule-Walker solutions of orders ``1..K`` from one autocorrelation run.

    Args:
        partial_autocorrelations: ``phi_kk`` for ``k = 1..K``.
        coefficients: ``coefficients[k-1]`` is ``(phi_k1, ..., phi_kk)``.
        prediction_variances: Normalized one-step prediction variances
            ``v_k / gamma(0)`` for ``k = 0..K``.

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 2].

    """

    partial_autocorrelations: Vector
    coefficients: tuple[Vector, ...]
    prediction_variances: Vector


def durbin_levinson(autocorrelations: Sequence[float]) -> DurbinLevinsonResult:
    """Run the Durbin-Levinson recursion on an autocorrelation sequence.

    ``phi_kk = (rho(k) - sum_j phi_(k-1),j rho(k-j)) / v_(k-1)``,
    ``phi_kj = phi_(k-1),j - phi_kk phi_(k-1),(k-j)``, ``v_k = v_(k-1)(1 - phi_kk^2)``.

    Args:
        autocorrelations: ``rho(0..K)`` with ``rho(0) = 1`` and ``K >= 1``.

    Returns:
        Partial autocorrelations, AR coefficients of every order, and the
        normalized prediction variances.

    Raises:
        ValueError: If ``rho(0) != 1``, fewer than two values are given, or the
            sequence is not positive definite (a prediction variance reaches 0).

    Equation:
        [EQ-TS-PACF-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 2], [REF-BOX-JENKINS-2015, Chapter 3].

    """
    rho = coerce_finite_sample(autocorrelations, name="autocorrelations")
    if len(rho) < 2:
        raise ValueError("autocorrelations must include lags 0 and 1")
    if abs(rho[0] - 1.0) > 1e-12:
        raise ValueError("autocorrelations[0] must equal 1")
    previous: list[float] = []
    variance = 1.0
    partial: list[float] = []
    coefficients: list[Vector] = []
    variances = [1.0]
    for k in range(1, len(rho)):
        numerator = rho[k] - math.fsum(
            previous[j] * rho[k - 1 - j] for j in range(k - 1)
        )
        if variance <= 1e-14:
            raise ValueError("autocorrelation sequence is not positive definite")
        reflection = numerator / variance
        if abs(reflection) >= 1.0:
            raise ValueError("autocorrelation sequence is not positive definite")
        current = [previous[j] - reflection * previous[k - 2 - j] for j in range(k - 1)]
        current.append(reflection)
        variance *= 1.0 - reflection * reflection
        previous = current
        partial.append(reflection)
        coefficients.append(tuple(current))
        variances.append(variance)
    return DurbinLevinsonResult(tuple(partial), tuple(coefficients), tuple(variances))


def pacf(series: Iterable[float], max_lag: int | None = None) -> Vector:
    """Return the sample partial autocorrelations ``phi_hat_kk`` for ``k = 1..K``.

    Args:
        series: At least two finite, non-constant observations.
        max_lag: Largest lag (default ``min(n - 1, 10 log10 n)``).

    Returns:
        Sample partial autocorrelations for lags ``1..max_lag``.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If the series is too short or constant, or ``max_lag`` is
            invalid.

    Equation:
        [EQ-TS-ACF-001], [EQ-TS-PACF-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 3], [REF-BOX-JENKINS-2015, Chapter 3].

    """
    return durbin_levinson(acf(series, max_lag)).partial_autocorrelations


def white_noise_band(sample_size: int, level: float = 0.95) -> float:
    """Return the half-width ``z_(1-alpha/2) / sqrt(n)`` of the white-noise band.

    Under i.i.d. noise with finite variance, each sample autocorrelation (and
    partial autocorrelation) at a nonzero lag is approximately ``N(0, 1/n)``.

    Args:
        sample_size: Series length ``n >= 1``.
        level: Two-sided pointwise confidence level.

    Returns:
        Half-width of the pointwise band around zero.

    Raises:
        TypeError: If ``sample_size`` is not an integer.
        ValueError: If ``sample_size`` is zero or ``level`` is invalid.

    Equation:
        [EQ-TS-BAND-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 1], [REF-BOX-JENKINS-2015, Chapter 2].

    """
    n = require_non_negative_integer(sample_size, name="sample_size")
    if n == 0:
        raise ValueError("sample_size must be positive")
    z = standard_normal_quantile(0.5 + 0.5 * _level(level))
    return z / math.sqrt(n)


def bartlett_band(
    autocorrelations: Sequence[float], sample_size: int, level: float = 0.95
) -> Vector:
    """Return Bartlett half-widths for testing an MA(k-1) null at each lag ``k``.

    ``Var(rho_hat(k)) ~ (1/n) (1 + 2 sum_{j=1}^{k-1} rho_hat(j)^2)``: the band at
    lag ``k`` assumes the autocorrelations beyond lag ``k - 1`` are zero.

    Args:
        autocorrelations: ``rho_hat(0..K)`` with ``rho_hat(0) = 1``.
        sample_size: Series length ``n``.
        level: Two-sided pointwise confidence level.

    Returns:
        Half-widths for lags ``1..K``.

    Raises:
        TypeError: If ``sample_size`` is not an integer.
        ValueError: If fewer than two autocorrelations are given, ``n`` is zero,
            or ``level`` is invalid.

    Equation:
        [EQ-TS-BAND-002].

    References:
        [REF-BOX-JENKINS-2015, Chapter 2], [REF-BROCKWELL-DAVIS-2016, Chapter 2].

    """
    rho = coerce_finite_sample(autocorrelations, name="autocorrelations")
    if len(rho) < 2:
        raise ValueError("autocorrelations must include lags 0 and 1")
    base = white_noise_band(sample_size, level)
    widths: list[float] = []
    accumulated = 0.0
    for k in range(1, len(rho)):
        widths.append(base * math.sqrt(1.0 + 2.0 * accumulated))
        accumulated += rho[k] * rho[k]
    return tuple(widths)


@dataclass(frozen=True, slots=True)
class LjungBoxResult:
    """Ljung-Box portmanteau test of no autocorrelation up to lag ``h``.

    Args:
        statistic: ``Q = n (n + 2) sum_{k=1}^{h} rho_hat(k)^2 / (n - k)``.
        degrees_of_freedom: ``h - fitted_parameters``.
        p_value: Upper chi-square tail probability of ``Q``.
        lags: Number of lags ``h``.

    References:
        [REF-LJUNG-BOX-1978], [REF-BOX-JENKINS-2015, Chapter 8].

    """

    statistic: float
    degrees_of_freedom: int
    p_value: float
    lags: int


def ljung_box(
    series: Iterable[float], lags: int = 10, *, fitted_parameters: int = 0
) -> LjungBoxResult:
    """Return the Ljung-Box statistic and its asymptotic chi-square p-value.

    Args:
        series: Observations or model residuals.
        lags: Number of autocorrelations ``h`` combined.
        fitted_parameters: ARMA parameters ``p + q`` estimated before computing
            residuals; subtracted from the degrees of freedom.

    Returns:
        Statistic, degrees of freedom, p-value, and lag count.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``lags`` is not in ``[1, n - 1]`` or the degrees of
            freedom are not positive.

    Equation:
        [EQ-TS-LB-001], [EQ-INF-DIST-CHI2-001].

    References:
        [REF-LJUNG-BOX-1978], [REF-BOX-JENKINS-2015, Chapter 8].

    """
    data = _series(series, minimum=3)
    n = len(data)
    h = _max_lag(lags, n)
    fitted = require_non_negative_integer(fitted_parameters, name="fitted_parameters")
    df = h - fitted
    if df < 1:
        raise ValueError("lags must exceed fitted_parameters")
    rho = acf(data, h)
    statistic = (
        n * (n + 2.0) * math.fsum(rho[k] ** 2 / (n - k) for k in range(1, h + 1))
    )
    return LjungBoxResult(statistic, df, chi_square_sf(statistic, df), h)


def difference(series: Iterable[float], order: int = 1, *, lag: int = 1) -> Vector:
    """Apply ``(1 - B^lag)^order`` to a series.

    ``lag = 1`` is ordinary differencing; ``lag = s`` is seasonal differencing
    with period ``s``. Each pass shortens the series by ``lag``.

    Args:
        series: Finite observations.
        order: Number of passes ``d >= 0``.
        lag: Differencing lag ``>= 1``.

    Returns:
        Differenced series of length ``n - order * lag``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``lag`` is zero or the series is too short.

    Equation:
        [EQ-TS-DIFF-001].

    References:
        [REF-BOX-JENKINS-2015, Chapter 4], [REF-BROCKWELL-DAVIS-2016, Chapter 6].

    """
    data = list(_series(series, minimum=1))
    d = require_non_negative_integer(order, name="order")
    step = require_non_negative_integer(lag, name="lag")
    if step < 1:
        raise ValueError("lag must be at least one")
    if len(data) <= d * step:
        raise ValueError("series is too short for the requested differencing")
    for _ in range(d):
        data = [data[t] - data[t - step] for t in range(step, len(data))]
    return tuple(data)


def seasonal_difference(series: Iterable[float], period: int, order: int = 1) -> Vector:
    """Apply the seasonal difference ``(1 - B^s)^D``.

    Args:
        series: Finite observations.
        period: Season length ``s >= 2``.
        order: Number of seasonal passes ``D``.

    Returns:
        Seasonally differenced series of length ``n - D s``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``period < 2`` or the series is too short.

    Equation:
        [EQ-TS-DIFF-001].

    References:
        [REF-BOX-JENKINS-2015, Chapter 9], [REF-BROCKWELL-DAVIS-2016, Chapter 6].

    """
    s = require_non_negative_integer(period, name="period")
    if s < 2:
        raise ValueError("period must be at least two")
    return difference(series, order, lag=s)


def undifference(
    differenced: Iterable[float], initial_values: Sequence[float], order: int = 1
) -> Vector:
    """Invert ordinary differencing given the first ``d`` original observations.

    Args:
        differenced: Values of ``(1 - B)^d y_t`` for ``t = d+1..n``.
        initial_values: Original observations ``y_1..y_d``.
        order: Differencing order ``d``.

    Returns:
        Reconstructed ``y_1..y_n`` (initial values included).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``len(initial_values) != order``.

    Equation:
        [EQ-TS-DIFF-001].

    References:
        [REF-BOX-JENKINS-2015, Chapter 4].

    """
    d = require_non_negative_integer(order, name="order")
    start = tuple(float(value) for value in initial_values)
    if len(start) != d:
        raise ValueError("initial_values must contain exactly `order` values")
    if d == 0:
        return coerce_finite_sample(differenced, name="differenced")
    increments = coerce_finite_sample(differenced, name="differenced")
    binomials = [math.comb(d, k) * (-1) ** k for k in range(d + 1)]
    values = list(start)
    for x in increments:
        values.append(x - math.fsum(binomials[k] * values[-k] for k in range(1, d + 1)))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class ClassicalDecomposition:
    """Moving-average decomposition into trend, seasonal, and remainder parts.

    Args:
        observed: Original series.
        trend: Centred moving average; ``None`` where the window is incomplete.
        seasonal: Seasonal component, one value per observation.
        seasonal_indices: One index per season position ``0..period-1``.
        remainder: ``y - T - S`` (additive) or ``y / (T S)`` (multiplicative);
            ``None`` where the trend is undefined.
        period: Season length.
        model: ``"additive"`` or ``"multiplicative"``.

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 1].

    """

    observed: Vector
    trend: tuple[float | None, ...]
    seasonal: Vector
    seasonal_indices: Vector
    remainder: tuple[float | None, ...]
    period: int
    model: str


def centered_moving_average(
    series: Iterable[float], period: int
) -> tuple[float | None, ...]:
    """Return the centred moving average that removes a period-``s`` cycle.

    Odd ``s`` uses an equal-weight window of ``s`` points; even ``s`` uses the
    ``2 x s`` window with half weights on the two end points.

    Args:
        series: Finite observations.
        period: Window period ``s >= 2``.

    Returns:
        Smoothed values, ``None`` for the first and last ``floor(s/2)`` points.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``period < 2`` or the series is shorter than ``s + 1``.

    Equation:
        [EQ-TS-DECOMP-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 1].

    """
    data = _series(series, minimum=1)
    s = require_non_negative_integer(period, name="period")
    if s < 2:
        raise ValueError("period must be at least two")
    if len(data) < s + 1:
        raise ValueError("series must contain more than one period")
    half = s // 2
    if s % 2:
        weights = [1.0 / s] * s
    else:
        weights = [0.5 / s] + [1.0 / s] * (s - 1) + [0.5 / s]
    result: list[float | None] = [None] * len(data)
    for t in range(half, len(data) - half):
        window = data[t - half : t + half + 1]
        result[t] = math.fsum(w * v for w, v in zip(weights, window, strict=True))
    return tuple(result)


def classical_decomposition(
    series: Iterable[float], period: int, *, model: str = "additive"
) -> ClassicalDecomposition:
    """Decompose a series by the classical moving-average method.

    The trend is the centred moving average; seasonal indices average the
    detrended values per season position and are normalized to sum to zero
    (additive) or average one (multiplicative); the remainder is what is left.

    Args:
        series: Finite observations (strictly positive for multiplicative).
        period: Season length ``s >= 2``; at least two full periods required.
        model: ``"additive"`` (``y = T + S + R``) or ``"multiplicative"``
            (``y = T S R``).

    Returns:
        Trend, seasonal, and remainder components.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``model`` is unknown, the series is shorter than two
            periods, or a multiplicative series is not positive.

    Equation:
        [EQ-TS-DECOMP-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 1], [REF-BOX-JENKINS-2015, Chapter 9].

    """
    if model not in ("additive", "multiplicative"):
        raise ValueError("model must be 'additive' or 'multiplicative'")
    data = _series(series, minimum=1)
    s = require_non_negative_integer(period, name="period")
    if s < 2:
        raise ValueError("period must be at least two")
    if len(data) < 2 * s:
        raise ValueError("series must contain at least two full periods")
    multiplicative = model == "multiplicative"
    if multiplicative and min(data) <= 0.0:
        raise ValueError("multiplicative decomposition requires positive data")
    trend = centered_moving_average(data, s)
    buckets: list[list[float]] = [[] for _ in range(s)]
    for t, (value, level) in enumerate(zip(data, trend, strict=True)):
        if level is not None:
            buckets[t % s].append(value / level if multiplicative else value - level)
    raw = [math.fsum(bucket) / len(bucket) for bucket in buckets]
    if multiplicative:
        scale = math.fsum(raw) / s
        indices = tuple(value / scale for value in raw)
    else:
        shift = math.fsum(raw) / s
        indices = tuple(value - shift for value in raw)
    seasonal = tuple(indices[t % s] for t in range(len(data)))
    remainder: list[float | None] = []
    for value, level, season in zip(data, trend, seasonal, strict=True):
        if level is None:
            remainder.append(None)
        elif multiplicative:
            remainder.append(value / (level * season))
        else:
            remainder.append(value - level - season)
    return ClassicalDecomposition(
        data, trend, seasonal, indices, tuple(remainder), s, model
    )
