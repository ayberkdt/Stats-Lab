"""Unit-root and stationarity tests: augmented Dickey-Fuller and KPSS.

**ADF.** The regression

    Delta y_t = D_t' delta + gamma y_(t-1) + sum_{j=1}^{k} phi_j Delta y_(t-j) + e_t

with deterministic terms ``D_t`` (none, constant, constant and trend, or
constant, trend and squared trend) is fitted by OLS; the statistic is the
t-ratio of ``gamma`` and the null of a unit root (``gamma = 0``) is rejected
for large negative values [EQ-TS-ADF-001]. Critical values come from the
response surfaces ``beta_inf + beta_1/T + beta_2/T^2 + beta_3/T^3`` of
MacKinnon (2010), Tables 2-4, row ``N = 1``, transcribed from the published
working paper (``T`` is the number of observations in the test regression)
[EQ-TS-ADF-002]. P-values are obtained by simulation: the Dickey-Fuller
statistic is computed on seeded Gaussian random walks of the same regression
length, and the p-value is the Monte Carlo proportion ``(1 + #{tau* <=
tau}) / (1 + R)`` with standard error ``sqrt(p (1 - p) / R)``
[EQ-TS-ADF-003]. The lag terms do not change the limiting distribution
(Said and Dickey 1984), so the same reference distribution serves the
augmented test.

**KPSS.** With residuals ``e_t`` from a regression on a constant (level
stationarity) or a constant and trend, partial sums ``S_t`` and the
Newey-West (Bartlett) long-run variance ``s^2(l)``, the statistic is
``eta = sum S_t^2 / (T^2 s^2(l))`` and the null of stationarity is rejected
for large values [EQ-TS-KPSS-001]. Its limit is ``sum_k lambda_k chi^2_1``
with the eigenvalues of the demeaned (Brownian bridge) or detrended
covariance kernel: ``lambda_k = 1/(k pi)^2`` for the level case (the
Cramer-von Mises limit), and ``1/(2 pi j)^2`` together with ``1/(4 x_j^2)``,
``tan x_j = x_j``, for the trend case. P-values are computed from this
limit by Imhof's numerical inversion of the characteristic function
[EQ-TS-KPSS-002], not by interpolation in a table.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import inverse_spd
from probstat.numerics import gauss_kronrod

Vector = tuple[float, ...]
_ADF_REGRESSIONS = ("n", "c", "ct", "ctt")
_KPSS_REGRESSIONS = ("c", "ct")

# MacKinnon (2010), "Critical Values for Cointegration Tests", Queen's
# Economics Department Working Paper No. 1227, Tables 2 (tau_nc, tau_c),
# 3 (tau_ct) and 4 (tau_ctt), rows N = 1: (beta_inf, beta_1, beta_2, beta_3)
# for the 1%, 5% and 10% lower-tail critical values; beta_3 = 0 where the
# table leaves it blank.
_MACKINNON_2010: dict[str, dict[float, tuple[float, float, float, float]]] = {
    "n": {
        0.01: (-2.56574, -2.2358, -3.627, 0.0),
        0.05: (-1.94100, -0.2686, -3.365, 31.223),
        0.10: (-1.61682, 0.2656, -2.714, 25.364),
    },
    "c": {
        0.01: (-3.43035, -6.5393, -16.786, -79.433),
        0.05: (-2.86154, -2.8903, -4.234, -40.040),
        0.10: (-2.56677, -1.5384, -2.809, 0.0),
    },
    "ct": {
        0.01: (-3.95877, -9.0531, -28.428, -134.155),
        0.05: (-3.41049, -4.3904, -9.036, -45.374),
        0.10: (-3.12705, -2.5856, -3.925, -22.380),
    },
    "ctt": {
        0.01: (-4.37113, -11.5882, -35.819, -334.047),
        0.05: (-3.83239, -5.9057, -12.490, -118.284),
        0.10: (-3.55326, -3.6596, -5.293, -63.559),
    },
}

# Upper quantiles of the KPSS limiting laws, computed by inverting
# kpss_asymptotic_sf (the regression test recomputes every entry).
_KPSS_CRITICAL: dict[str, dict[float, float]] = {
    "c": {0.10: 0.347305, 0.05: 0.461361, 0.025: 0.580615, 0.01: 0.743459},
    "ct": {0.10: 0.119220, 0.05: 0.147890, 0.025: 0.177468, 0.01: 0.217747},
}


def _regression_name(value: str, allowed: tuple[str, ...]) -> str:
    """Validate a deterministic-term specification.

    Args:
        value: Requested specification.
        allowed: Permitted names.

    Returns:
        The validated name.

    Raises:
        ValueError: If the name is not permitted.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if value not in allowed:
        raise ValueError(f"regression must be one of {allowed}")
    return value


def _deterministic(regression: str, t: int) -> list[float]:
    """Return the deterministic regressors at time index ``t``.

    Args:
        regression: ``"n"``, ``"c"``, ``"ct"``, or ``"ctt"``.
        t: Time index (1-based within the regression sample).

    Returns:
        ``[]``, ``[1]``, ``[1, t]``, or ``[1, t, t^2]``.

    References:
        [REF-MACKINNON-2010]. No equation is evaluated.

    """
    if regression == "n":
        return []
    if regression == "c":
        return [1.0]
    if regression == "ct":
        return [1.0, float(t)]
    return [1.0, float(t), float(t) * t]


def adf_critical_values(nobs: int, regression: str = "c") -> dict[float, float]:
    """Return MacKinnon (2010) Dickey-Fuller critical values for ``T`` observations.

    Evaluates ``beta_inf + beta_1/T + beta_2/T^2 + beta_3/T^3`` with the
    ``N = 1`` coefficients of MacKinnon (2010), Table 2 (``"n"``, ``"c"``),
    Table 3 (``"ct"``), and Table 4 (``"ctt"``). ``T`` is the number of
    observations in the test regression; the surfaces are strictly valid for
    i.i.d. normal errors and the Dickey-Fuller regression without lags.

    Args:
        nobs: Regression sample size ``T >= 1``.
        regression: Deterministic terms ``"n"``, ``"c"``, ``"ct"``, ``"ctt"``.

    Returns:
        Lower-tail critical values keyed by level ``0.01``, ``0.05``, ``0.10``.

    Raises:
        ValueError: If ``nobs < 1`` or the regression is unknown.

    Equation:
        [EQ-TS-ADF-002].

    References:
        [REF-MACKINNON-2010, Tables 2-4].

    """
    t = require_non_negative_integer(nobs, name="nobs")
    if t < 1:
        raise ValueError("nobs must be at least 1")
    name = _regression_name(regression, _ADF_REGRESSIONS)
    return {
        level: b0 + b1 / t + b2 / t**2 + b3 / t**3
        for level, (b0, b1, b2, b3) in _MACKINNON_2010[name].items()
    }


@functools.lru_cache(maxsize=32)
def _simulated_statistics(
    nobs: int, regression: str, simulations: int, seed: int
) -> tuple[float, ...]:
    """Return sorted Dickey-Fuller statistics of seeded Gaussian random walks.

    Each replication draws ``y_0 = 0`` and ``y_t = y_(t-1) + z_t`` for
    ``t = 1..T`` and computes the t-ratio of ``gamma`` in the regression of
    ``Delta y_t`` on ``D_t`` and ``y_(t-1)`` by the Frisch-Waugh-Lovell
    partialling of the deterministic terms (one pass of sufficient sums).

    Args:
        nobs: Regression length ``T``.
        regression: Deterministic terms.
        simulations: Replications ``R``.
        seed: Root seed; replication ``r`` uses child ``("adf", r)``.

    Returns:
        Sorted statistics.

    Equation:
        [EQ-TS-ADF-001], [EQ-TS-ADF-003].

    References:
        [REF-DICKEY-FULLER-1979], [REF-MACKINNON-2010].

    """
    root = RandomStream(seed)
    rows = [_deterministic(regression, t) for t in range(1, nobs + 1)]
    m = len(rows[0])
    inverse = (
        inverse_spd(
            tuple(
                tuple(math.fsum(r[i] * r[j] for r in rows) for j in range(m))
                for i in range(m)
            )
        )
        if m
        else ()
    )
    values: list[float] = []
    for replication in range(simulations):
        gauss = root.child("adf", replication).generator().gauss
        level = 0.0
        sxx = sxd = sdd = 0.0
        dx = [0.0] * m
        dd = [0.0] * m
        for t in range(nobs):
            step = gauss(0.0, 1.0)
            row = rows[t]
            sxx += level * level
            sxd += level * step
            sdd += step * step
            for i in range(m):
                dx[i] += row[i] * level
                dd[i] += row[i] * step
            level += step
        if m:
            qxx = qxd = qdd = 0.0
            for i in range(m):
                for j in range(m):
                    weight = inverse[i][j]
                    qxx += dx[i] * weight * dx[j]
                    qxd += dx[i] * weight * dd[j]
                    qdd += dd[i] * weight * dd[j]
            sxx -= qxx
            sxd -= qxd
            sdd -= qdd
        gamma = sxd / sxx
        residual = max(sdd - sxd * gamma, 0.0)
        scale = residual / (nobs - m - 1)
        values.append(gamma / math.sqrt(scale / sxx))
    values.sort()
    return tuple(values)


def adf_simulated_p_value(
    statistic: float,
    nobs: int,
    regression: str = "c",
    *,
    simulations: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return the simulated Dickey-Fuller p-value and its Monte Carlo error.

    The reference distribution is the finite-sample null law of the
    Dickey-Fuller t-ratio for a Gaussian random walk with a regression of
    ``T`` observations; the p-value is ``(1 + #{tau*_r <= tau}) / (1 + R)``
    and its standard error ``sqrt(p (1 - p) / R)``. Results are cached per
    ``(T, regression, R, seed)``.

    Args:
        statistic: Observed t-ratio ``tau``.
        nobs: Regression sample size ``T`` (more than the number of
            regressors plus one).
        regression: Deterministic terms.
        simulations: Replications ``R >= 100``.
        seed: Non-negative root seed.

    Returns:
        ``(p_value, standard_error)``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-TS-ADF-003].

    References:
        [REF-DICKEY-FULLER-1979], [REF-SAID-DICKEY-1984],
        [REF-DAVISON-HINKLEY-1997, Chapter 4].

    """
    tau = require_finite_real(statistic, name="statistic")
    name = _regression_name(regression, _ADF_REGRESSIONS)
    t = require_non_negative_integer(nobs, name="nobs")
    count = require_non_negative_integer(simulations, name="simulations")
    root = require_non_negative_integer(seed, name="seed")
    if count < 100:
        raise ValueError("simulations must be at least 100")
    if t < len(_deterministic(name, 1)) + 3:
        raise ValueError("nobs is too small for the regression")
    reference = _simulated_statistics(t, name, count, root)
    below = _count_at_most(reference, tau)
    p_value = (1.0 + below) / (1.0 + count)
    return p_value, math.sqrt(p_value * (1.0 - p_value) / count)


def _count_at_most(values: Sequence[float], target: float) -> int:
    """Return how many sorted values are at most ``target`` (binary search).

    Args:
        values: Sorted values.
        target: Threshold.

    Returns:
        Count of ``values <= target``.

    References:
        Project numerical policy [REF-HIGHAM-2002]. No equation is evaluated.

    """
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] <= target:
            low = middle + 1
        else:
            high = middle
    return low


def _ols(
    rows: Sequence[Sequence[float]], target: Sequence[float]
) -> tuple[list[float], list[float], float]:
    """Return OLS coefficients, their standard errors, and the residual sum of squares.

    Args:
        rows: Regressor rows.
        target: Response.

    Returns:
        Coefficients, standard errors (``s^2 = RSS / (n - m)``), and RSS.

    Raises:
        ValueError: If the regressors are collinear.

    Equation:
        [EQ-TS-ADF-001].

    References:
        [REF-HAMILTON-1994, Chapter 8].

    """
    m = len(rows[0])
    n = len(rows)
    gram = tuple(
        tuple(math.fsum(r[i] * r[j] for r in rows) for j in range(m)) for i in range(m)
    )
    inverse = inverse_spd(gram)
    moment = [
        math.fsum(r[i] * y for r, y in zip(rows, target, strict=True)) for i in range(m)
    ]
    beta = [math.fsum(inverse[i][j] * moment[j] for j in range(m)) for i in range(m)]
    rss = math.fsum(
        (y - math.fsum(b * x for b, x in zip(beta, r, strict=True))) ** 2
        for r, y in zip(rows, target, strict=True)
    )
    scale = rss / (n - m)
    errors = [math.sqrt(scale * inverse[i][i]) for i in range(m)]
    return beta, errors, rss


def _adf_design(
    data: Sequence[float], lags: int, start: int, regression: str
) -> tuple[list[list[float]], list[float]]:
    """Return ADF regressor rows ``[D_t, y_(t-1), Delta y_(t-1..t-k)]`` and responses.

    Args:
        data: Observations ``y_0..y_(n-1)``.
        lags: Number of lagged differences ``k``.
        start: First 0-based index ``t`` of ``Delta y_t`` (at least ``k+1``).
        regression: Deterministic terms.

    Returns:
        Regressor rows and ``Delta y_t`` responses.

    Equation:
        [EQ-TS-ADF-001].

    References:
        [REF-SAID-DICKEY-1984].

    """
    differences = [0.0] + [data[t] - data[t - 1] for t in range(1, len(data))]
    rows: list[list[float]] = []
    target: list[float] = []
    for position, t in enumerate(range(start, len(data)), start=1):
        row = _deterministic(regression, position)
        row.append(data[t - 1])
        row.extend(differences[t - j] for j in range(1, lags + 1))
        rows.append(row)
        target.append(differences[t])
    return rows, target


@dataclass(frozen=True, slots=True)
class ADFResult:
    """Augmented Dickey-Fuller test result.

    Args:
        statistic: t-ratio ``tau`` of ``gamma``.
        p_value: Simulated p-value (lower tail).
        p_value_standard_error: Monte Carlo standard error of ``p_value``.
        lags: Number of lagged differences ``k`` used.
        nobs: Observations in the test regression ``T``.
        gamma: Estimated coefficient of ``y_(t-1)``.
        critical_values: MacKinnon (2010) values for ``T`` at 1%, 5%, 10%.
        regression: Deterministic terms.
        criterion: Lag-selection criterion, or ``None`` for a fixed ``k``.
        simulations: Replications behind the p-value.

    References:
        [REF-DICKEY-FULLER-1979], [REF-SAID-DICKEY-1984],
        [REF-MACKINNON-2010].

    """

    statistic: float
    p_value: float
    p_value_standard_error: float
    lags: int
    nobs: int
    gamma: float
    critical_values: dict[float, float]
    regression: str
    criterion: str | None
    simulations: int


def adf_test(
    series: Iterable[float],
    *,
    regression: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    criterion: str = "aic",
    simulations: int = 2000,
    seed: int = 0,
) -> ADFResult:
    """Run the augmented Dickey-Fuller test for a unit root.

    With ``lags=None`` the number of lagged differences is chosen by
    ``criterion`` (``"aic"`` or ``"bic"``, ``log(RSS/T') + c m / T'``) over
    ``k = 0..max_lags`` on the common sample that drops the first
    ``max_lags + 1`` observations; the chosen ``k`` is then refitted on the
    largest sample it allows. The default ``max_lags`` is Schwert's
    ``floor(12 (n/100)^(1/4))``, capped so the regression keeps at least ten
    residual degrees of freedom.

    Args:
        series: Observations ``y_1..y_n`` (levels, not differences).
        regression: ``"n"`` (no deterministic terms), ``"c"`` (constant),
            ``"ct"`` (constant and trend), or ``"ctt"`` (adds ``t^2``).
        lags: Fixed number of lagged differences, or ``None`` to select.
        max_lags: Largest ``k`` considered when selecting.
        criterion: ``"aic"`` or ``"bic"``.
        simulations: Replications for the simulated p-value.
        seed: Root seed of the p-value simulation.

    Returns:
        Statistic, simulated p-value with its Monte Carlo error, lag order,
        and MacKinnon critical values.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is invalid or the series is too short.

    Equation:
        [EQ-TS-ADF-001], [EQ-TS-ADF-002], [EQ-TS-ADF-003].

    References:
        [REF-DICKEY-FULLER-1979], [REF-SAID-DICKEY-1984],
        [REF-MACKINNON-2010], [REF-SCHWERT-1989].

    """
    data = coerce_finite_sample(series, name="series")
    name = _regression_name(regression, _ADF_REGRESSIONS)
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'")
    n = len(data)
    deterministic = len(_deterministic(name, 1))
    if n < deterministic + 12:
        raise ValueError("series is too short for the test")
    if lags is not None:
        chosen = require_non_negative_integer(lags, name="lags")
        selection: str | None = None
    else:
        if max_lags is None:
            largest = int(12.0 * (n / 100.0) ** 0.25)
        else:
            largest = require_non_negative_integer(max_lags, name="max_lags")
        largest = min(largest, max(0, (n - deterministic - 12) // 2))
        start = largest + 1
        sample = n - start
        best_value = math.inf
        chosen = 0
        for k in range(largest + 1):
            rows, target = _adf_design(data, k, start, name)
            _, _, rss = _ols(rows, target)
            parameters = len(rows[0])
            penalty = 2.0 if criterion == "aic" else math.log(sample)
            value = math.log(rss / sample) + penalty * parameters / sample
            if value < best_value - 1e-12:
                best_value, chosen = value, k
        selection = criterion
    start = chosen + 1
    rows, target = _adf_design(data, chosen, start, name)
    nobs = len(rows)
    if nobs - len(rows[0]) < 3:
        raise ValueError("too few observations for the requested lags")
    beta, errors, _ = _ols(rows, target)
    gamma = beta[deterministic]
    statistic = gamma / errors[deterministic]
    p_value, error = adf_simulated_p_value(
        statistic, nobs, name, simulations=simulations, seed=seed
    )
    return ADFResult(
        statistic=statistic,
        p_value=p_value,
        p_value_standard_error=error,
        lags=chosen,
        nobs=nobs,
        gamma=gamma,
        critical_values=adf_critical_values(nobs, name),
        regression=name,
        criterion=selection,
        simulations=simulations,
    )


def _tan_fixed_points(count: int) -> list[float]:
    """Return the first ``count`` positive roots of ``tan x = x``.

    Newton's method on ``sin x - x cos x`` from ``q - 1/q`` with
    ``q = (j + 1/2) pi``; root ``j`` lies in ``(j pi, (j + 1/2) pi)``.

    Args:
        count: Number of roots.

    Returns:
        Increasing roots.

    Equation:
        [EQ-TS-KPSS-002].

    References:
        [REF-KPSS-1992]. No other equation is evaluated.

    """
    roots: list[float] = []
    for j in range(1, count + 1):
        q = (j + 0.5) * math.pi
        x = q - 1.0 / q
        for _ in range(60):
            step = (math.sin(x) - x * math.cos(x)) / (x * math.sin(x))
            x -= step
            if abs(step) <= 1e-16 * x:
                break
        roots.append(x)
    return roots


_EXPLICIT_TERMS = 50
_TAIL_TERMS = 20000


@functools.lru_cache(maxsize=2)
def _kpss_eigenvalues(regression: str) -> tuple[Vector, float, float, float]:
    """Return leading eigenvalues and tail moments of the KPSS limit law.

    Args:
        regression: ``"c"`` or ``"ct"``.

    Returns:
        The ``50`` largest eigenvalues and the tail sums ``sum lambda``,
        ``sum lambda^2``, ``sum lambda^3`` over the remaining ones (the
        first from the exact trace ``1/6`` or ``1/15``).

    Equation:
        [EQ-TS-KPSS-002].

    References:
        [REF-KPSS-1992], [REF-ANDERSON-DARLING-1952].

    """
    if regression == "c":
        values = [1.0 / (k * math.pi) ** 2 for k in range(1, _TAIL_TERMS + 1)]
        trace = 1.0 / 6.0
    else:
        half = _TAIL_TERMS // 2
        values = [1.0 / (2.0 * math.pi * j) ** 2 for j in range(1, half + 1)]
        values += [1.0 / (4.0 * x * x) for x in _tan_fixed_points(half)]
        values.sort(reverse=True)
        trace = 1.0 / 15.0
    head = tuple(values[:_EXPLICIT_TERMS])
    tail = values[_EXPLICIT_TERMS:]
    return (
        head,
        trace - math.fsum(head),
        math.fsum(v * v for v in tail),
        math.fsum(v**3 for v in tail),
    )


def kpss_asymptotic_sf(statistic: float, regression: str = "c") -> float:
    """Return ``P(eta > x)`` under the limiting null law of the KPSS statistic.

    The limit is ``Q = sum_k lambda_k Z_k^2`` with ``Z_k`` i.i.d. standard
    normal. Imhof's formula ``P(Q > x) = 1/2 + (1/pi) int_0^inf sin
    theta(u) / (u rho(u)) du`` with ``theta(u) = (1/2) sum arctan(lambda_k
    u) - x u / 2`` and ``rho(u) = prod (1 + lambda_k^2 u^2)^(1/4)`` is
    integrated by adaptive Gauss-Kronrod quadrature with the 50 largest
    eigenvalues explicit and the rest entering through their first three
    power sums (``arctan`` and ``log(1 + .)`` expanded to third and second
    order). Agreement with the closed-form Cramer-von Mises limit is within
    ``1e-9`` for the level case.

    Args:
        statistic: Observed ``eta >= 0``.
        regression: ``"c"`` (level) or ``"ct"`` (trend).

    Returns:
        Upper-tail probability in ``[0, 1]``.

    Raises:
        ValueError: If the statistic is negative or the regression unknown.

    Equation:
        [EQ-TS-KPSS-002].

    References:
        [REF-KPSS-1992], [REF-IMHOF-1961], [REF-ANDERSON-DARLING-1952].

    """
    x = require_finite_real(statistic, name="statistic")
    if x < 0.0:
        raise ValueError("statistic must be non-negative")
    head, m1, m2, m3 = _kpss_eigenvalues(
        _regression_name(regression, _KPSS_REGRESSIONS)
    )
    if x == 0.0:
        return 1.0

    def integrand(u: float) -> float:
        """Return Imhof's integrand ``sin theta(u) / (u rho(u))``.

        References:
            [EQ-TS-KPSS-002], [REF-IMHOF-1961].

        """
        if u == 0.0:
            return 0.5 * (math.fsum(head) + m1 - x)
        theta = 0.5 * (
            math.fsum(math.atan(v * u) for v in head) + u * m1 - u**3 * m3 / 3.0 - x * u
        )
        log_rho = 0.25 * (
            math.fsum(math.log1p((v * u) ** 2) for v in head) + u * u * m2
        )
        return math.sin(theta) / (u * math.exp(log_rho))

    edges = (0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
    integral = math.fsum(
        gauss_kronrod(integrand, a, b, abs_tol=1e-12, rel_tol=1e-10).value
        for a, b in zip(edges, edges[1:], strict=False)
    )
    return min(1.0, max(0.0, 0.5 + integral / math.pi))


def kpss_critical_values(regression: str = "c") -> dict[float, float]:
    """Return upper-tail critical values of the KPSS limiting null law.

    The values are the ``0.90``, ``0.95``, ``0.975``, and ``0.99`` quantiles
    of the law in :func:`kpss_asymptotic_sf`, to six decimals.

    Args:
        regression: ``"c"`` (level) or ``"ct"`` (trend).

    Returns:
        Critical values keyed by level ``0.10``, ``0.05``, ``0.025``, ``0.01``.

    Raises:
        ValueError: If the regression is unknown.

    Equation:
        [EQ-TS-KPSS-002].

    References:
        [REF-KPSS-1992], [REF-IMHOF-1961].

    """
    return dict(_KPSS_CRITICAL[_regression_name(regression, _KPSS_REGRESSIONS)])


def newey_west_variance(residuals: Sequence[float], lags: int) -> float:
    """Return the Bartlett-kernel (Newey-West) long-run variance.

    ``s^2(l) = gamma_0 + 2 sum_{j=1}^{l} (1 - j/(l+1)) gamma_j`` with
    ``gamma_j = (1/T) sum_{t>j} e_t e_(t-j)``; non-negative by construction.

    Args:
        residuals: Mean-zero residuals ``e_1..e_T``.
        lags: Truncation lag ``l`` in ``[0, T - 1]``.

    Returns:
        Long-run variance estimate.

    Raises:
        ValueError: If ``lags`` is out of range or the series is empty.

    Equation:
        [EQ-TS-KPSS-001].

    References:
        [REF-NEWEY-WEST-1987].

    """
    data = coerce_finite_sample(tuple(residuals), name="residuals")
    n = len(data)
    truncation = require_non_negative_integer(lags, name="lags")
    if truncation >= n:
        raise ValueError("lags must be smaller than the number of residuals")
    total = math.fsum(e * e for e in data) / n
    for j in range(1, truncation + 1):
        weight = 1.0 - j / (truncation + 1.0)
        total += (
            2.0 * weight * math.fsum(data[t] * data[t - j] for t in range(j, n)) / n
        )
    return total


@dataclass(frozen=True, slots=True)
class KPSSResult:
    """KPSS stationarity test result.

    Args:
        statistic: ``eta = sum S_t^2 / (T^2 s^2(l))``.
        p_value: Upper-tail probability under the limiting null law.
        lags: Newey-West truncation lag ``l``.
        long_run_variance: ``s^2(l)``.
        critical_values: Limiting-law critical values at 10%, 5%, 2.5%, 1%.
        regression: ``"c"`` or ``"ct"``.

    References:
        [REF-KPSS-1992], [REF-NEWEY-WEST-1987].

    """

    statistic: float
    p_value: float
    lags: int
    long_run_variance: float
    critical_values: dict[float, float]
    regression: str


def kpss_test(
    series: Iterable[float],
    *,
    regression: str = "c",
    lags: int | str = "short",
) -> KPSSResult:
    """Run the KPSS test of the null of level or trend stationarity.

    Args:
        series: Observations ``y_1..y_T``.
        regression: ``"c"`` (level stationarity) or ``"ct"`` (trend
            stationarity).
        lags: Newey-West truncation lag: an integer, ``"short"``
            (``floor(4 (T/100)^(1/4))``) or ``"long"``
            (``floor(12 (T/100)^(1/4))``), the two rules of R's
            ``tseries::kpss.test``.

    Returns:
        Statistic, asymptotic p-value, lag, long-run variance, and critical
        values.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is invalid, the series is too short, or
            the residuals are identically zero.

    Equation:
        [EQ-TS-KPSS-001], [EQ-TS-KPSS-002].

    References:
        [REF-KPSS-1992], [REF-NEWEY-WEST-1987].

    """
    data = coerce_finite_sample(series, name="series")
    name = _regression_name(regression, _KPSS_REGRESSIONS)
    n = len(data)
    if n < 10:
        raise ValueError("series must contain at least 10 observations")
    if lags == "short":
        truncation = int(4.0 * (n / 100.0) ** 0.25)
    elif lags == "long":
        truncation = int(12.0 * (n / 100.0) ** 0.25)
    elif isinstance(lags, str):
        raise ValueError("lags must be an integer, 'short', or 'long'")
    else:
        truncation = require_non_negative_integer(lags, name="lags")
    if truncation >= n:
        raise ValueError("lags must be smaller than the sample size")
    if name == "c":
        centre = math.fsum(data) / n
        residuals = [y - centre for y in data]
    else:
        time_mean = (n + 1) / 2.0
        sxx = math.fsum((t - time_mean) ** 2 for t in range(1, n + 1))
        y_mean = math.fsum(data) / n
        slope = (
            math.fsum(
                (t - time_mean) * (y - y_mean)
                for t, y in zip(range(1, n + 1), data, strict=True)
            )
            / sxx
        )
        residuals = [
            y - y_mean - slope * (t - time_mean)
            for t, y in zip(range(1, n + 1), data, strict=True)
        ]
    variance = newey_west_variance(residuals, truncation)
    if variance <= 0.0:
        raise ValueError("the residuals are identically zero")
    partial = 0.0
    squares: list[float] = []
    for e in residuals:
        partial += e
        squares.append(partial * partial)
    statistic = math.fsum(squares) / (n * n * variance)
    return KPSSResult(
        statistic=statistic,
        p_value=kpss_asymptotic_sf(statistic, name),
        lags=truncation,
        long_run_variance=variance,
        critical_values=kpss_critical_values(name),
        regression=name,
    )
