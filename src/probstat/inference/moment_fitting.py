"""Method-of-moments fitting of the ready-made distribution families.

:func:`fit_distribution_moments` matches the first ``k`` sample moments of
the data to the moments of a ``k``-parameter family: the mean, the
population variance (divisor ``n``) and, for the three-parameter GEV, the
population skewness. These are the conventions of ``fitdistrplus::mmedist``
(``method = "mme"`` in ``fitdist``), which uses ``v = (n - 1)/n var(x)`` for
its closed formulas and raw moments ``mean(x^j)`` for numerical matching;
both define the same estimator. Standard errors come from the delta method
applied to the sample moments, with the moment covariance taken either from
the fitted law (``"model"``, the ``calcvcov`` convention of fitdistrplus) or
from the data (``"empirical"``, a sandwich estimate that does not assume the
family). A moment estimate does not maximize the likelihood, so the result
carries no information criteria; the log-likelihood is reported only as the
value at the moment estimate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from probstat.distributions import GeneralizedExtremeValue
from probstat.linalg import Matrix, Vector, lu_solve, sandwich
from probstat.numerics.special import EULER_GAMMA
from probstat.optimization import brent_root, central_jacobian

from .fitting import (
    BETA_FAMILY,
    EXPONENTIAL_FAMILY,
    GAMMA_FAMILY,
    GEV_FAMILY,
    GUMBEL_FAMILY,
    LOGISTIC_FAMILY,
    LOGNORMAL_FAMILY,
    NORMAL_FAMILY,
    UNIFORM_FAMILY,
    WEIBULL_FAMILY,
    DistributionFamily,
    FittableDistribution,
    _validated_fit_sample,
)

MomentCovariance = Literal["model", "empirical"]

_COVARIANCES = ("model", "empirical")
_ZETA_3 = 1.2020569031595942
# zeta(k) for k = 2..30, used by the small-shape GEV skewness series; the
# values for k >= 13 are partial sums over n < 40 (tail below 1e-19).
_ZETA = (
    math.pi**2 / 6.0,
    _ZETA_3,
    math.pi**4 / 90.0,
    1.0369277551433699,
    math.pi**6 / 945.0,
    1.0083492773819228,
    math.pi**8 / 9450.0,
    1.0020083928260822,
    math.pi**10 / 93555.0,
    1.0004941886041195,
    691.0 * math.pi**12 / 638512875.0,
) + tuple(math.fsum(n ** (-k) for n in range(1, 40)) for k in range(13, 31))
_GEV_SERIES_THRESHOLD = 0.05
_GEV_QUADRATURE_THRESHOLD = 0.1
_GEV_SHAPE_FLOOR = -50.0
_GUMBEL_SKEWNESS = 2.0 * _ZETA_3 / math.pow(math.pi**2 / 6.0, 1.5)
_MOMENT_NAMES = ("mean", "variance", "third central moment")


@dataclass(frozen=True)
class MomentFitResult:
    """Method-of-moments fit of one family with delta-method standard errors.

    Args:
        family: Family label.
        parameter_names: Parameter names in constructor order.
        estimates: Moment estimates (natural parameters).
        standard_errors: Delta-method standard errors (``nan`` when the
            moments the covariance needs do not exist).
        covariance: Delta-method covariance of the estimates, or ``None``.
        moment_names: Names of the matched sample moments.
        sample_moments: Matched sample moments: mean, population variance
            (divisor ``n``) and, for three parameters, the third central
            moment (divisor ``n``).
        log_likelihood: Log-likelihood evaluated at the moment estimates
            (``-inf`` when an observation lies outside the fitted support).
            It is not maximized, so AIC and BIC do not apply.
        sample_size: Number of observations ``n``.
        covariance_method: ``"model"`` or ``"empirical"``.
        message: Note on the standard errors.
        distribution: Fitted law.
        method: Always ``"moments"``.

    References:
        [EQ-INF-MOM-001], [EQ-INF-MOM-004],
        [REF-CASELLA-BERGER-2002, Section 7.2.1],
        [REF-DELIGNETTE-MULLER-DUTANG-2015].

    """

    family: str
    parameter_names: tuple[str, ...]
    estimates: Vector
    standard_errors: Vector
    covariance: Matrix | None
    moment_names: tuple[str, ...]
    sample_moments: Vector
    log_likelihood: float
    sample_size: int
    covariance_method: str
    message: str
    distribution: FittableDistribution
    method: str = "moments"

    @property
    def parameter_count(self) -> int:
        """Return the number of estimated parameters ``k``.

        References:
            [REF-CASELLA-BERGER-2002, Section 7.2.1]. No equation is evaluated.

        """
        return len(self.estimates)

    @property
    def params(self) -> dict[str, float]:
        """Return estimates keyed by parameter name.

        References:
            [REF-CASELLA-BERGER-2002, Section 7.2.1]. No equation is evaluated.

        """
        return dict(zip(self.parameter_names, self.estimates, strict=True))

    @property
    def standard_error_map(self) -> dict[str, float]:
        """Return standard errors keyed by parameter name.

        References:
            [EQ-INF-MOM-004]. No equation is evaluated.

        """
        return dict(zip(self.parameter_names, self.standard_errors, strict=True))


@dataclass(frozen=True)
class _MomentRule:
    """Forward map, inverse map, and central moments of one family.

    ``forward`` maps parameters to the matched moments (mean, variance, and
    third central moment for three parameters); ``inverse`` solves the
    moment equations; ``central`` returns the central moment of order
    ``r >= 2`` of the law (``inf`` when it does not exist).

    References:
        [EQ-INF-MOM-002], [EQ-INF-MOM-003].

    """

    forward: Callable[[Vector], Vector]
    inverse: Callable[[Vector], Vector]
    central: Callable[[Vector, int], float]


# ---------------------------------------------------------------------------
# Central moments from cumulants and the GEV skewness
# ---------------------------------------------------------------------------


def _central_from_cumulants(variance: float, k3: float, k4: float, order: int) -> float:
    """Return a central moment of order 2, 3, or 4 from cumulants.

    ``mu_2 = k_2``, ``mu_3 = k_3``, ``mu_4 = k_4 + 3 k_2^2``.

    References:
        [REF-CASELLA-BERGER-2002, Section 2.3]. No new equation is evaluated.

    """
    if order == 2:
        return variance
    if order == 3:
        return k3
    if order == 4:
        return k4 + 3.0 * variance * variance
    raise ValueError(f"central moment of order {order} is not provided")


def _phi(x: float) -> float:
    """Return ``exp(x) - 1 - x`` without cancellation for small ``|x|``.

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.7]. No new equation is evaluated.

    """
    if abs(x) >= 0.1:
        return math.expm1(x) - x
    term = x * x / 2.0
    total = 0.0
    for order in range(3, 24):
        total += term
        term *= x / order
    return total


def _gev_shape_differences(xi: float) -> tuple[float, float, float]:
    """Return ``a = L2 - 2 L1``, ``b = L3 - 3 L1``, and ``d = b - 3a`` over ``xi^p``.

    ``L_k = log Gamma(1 - k xi)``. The values returned are ``a / xi^2``,
    ``b / xi^2``, and ``d / xi^3``; for ``|xi| < 0.05`` they come from the
    series ``log Gamma(1 - z) = gamma_E z + sum_(k>=2) zeta(k) z^k / k``, in
    which the leading terms cancel exactly.

    References:
        [EQ-INF-MOM-003], [REF-NIST-DLMF-GAMMA, Section 5.7].

    """
    if abs(xi) < _GEV_SERIES_THRESHOLD:
        a_scaled = b_scaled = d_scaled = 0.0
        for index in range(len(_ZETA) - 1, -1, -1):
            order = index + 2
            zeta = _ZETA[index]
            a_scaled = a_scaled * xi + zeta * (2.0**order - 2.0) / order
            b_scaled = b_scaled * xi + zeta * (3.0**order - 3.0) / order
            if order >= 3:
                d_scaled = (
                    d_scaled * xi + zeta * (3.0**order - 3.0 * 2.0**order + 3.0) / order
                )
        return a_scaled, b_scaled, d_scaled
    l1 = math.lgamma(1.0 - xi)
    l2 = math.lgamma(1.0 - 2.0 * xi)
    l3 = math.lgamma(1.0 - 3.0 * xi)
    a = l2 - 2.0 * l1
    b = l3 - 3.0 * l1
    d = (l3 - 3.0 * l2) + 3.0 * l1
    return a / (xi * xi), b / (xi * xi), d / (xi * xi * xi)


def _gev_skewness(xi: float) -> float:
    """Return the skewness of the GEV law with shape ``xi``.

    With ``g_k = Gamma(1 - k xi)``, the skewness is
    ``sign(xi) (g3 - 3 g1 g2 + 2 g1^3) / (g2 - g1^2)^(3/2)``. It is evaluated
    as ``(d + phi(b) - 3 phi(a)) / xi^3`` over ``(expm1(a) / xi^2)^(3/2)``
    with ``a = L2 - 2 L1``, ``b = L3 - 3 L1``, ``d = b - 3a``,
    ``L_k = log g_k``, and ``phi(x) = e^x - 1 - x``, which is free of the
    ``O(xi^3)`` cancellation of the textbook form; the limit at ``xi = 0`` is
    the Gumbel value ``12 sqrt(6) zeta(3) / pi^3 = 1.1395...``.

    Args:
        xi: Finite shape (climate convention: ``xi > 0`` is heavy-tailed).

    Returns:
        Skewness, increasing in ``xi`` from ``-inf`` to ``+inf`` as ``xi``
        approaches ``1/3``; ``inf`` for ``xi >= 1/3``.

    Raises:
        ValueError: If ``xi`` is not finite.

    Equation:
        [EQ-INF-MOM-003].

    References:
        [REF-COLES-2001, Section 3.1], [REF-NIST-DLMF-GAMMA, Section 5.7].

    """
    shape = float(xi)
    if not math.isfinite(shape):
        raise ValueError("xi must be finite")
    if shape >= 1.0 / 3.0:
        return math.inf
    if shape == 0.0:
        return _GUMBEL_SKEWNESS
    a_scaled, b_scaled, d_scaled = _gev_shape_differences(shape)
    square = shape * shape
    a, b = a_scaled * square, b_scaled * square
    third = d_scaled + (_phi(b) - 3.0 * _phi(a)) / (square * shape)
    spread = math.expm1(a) / square
    return third / math.pow(spread, 1.5)


def _gev_central_moments_quadrature(xi: float, orders: int) -> tuple[float, ...]:
    """Return central moments ``2..orders`` of the standard GEV by quadrature.

    ``Z = expm1(xi G)/xi`` with ``G`` standard Gumbel; the trapezoid rule on
    ``G`` over ``[-6, 200]`` with step ``0.05`` converges geometrically for
    this analytic, doubly-exponentially decaying integrand. Used for
    ``|xi| < 0.1``, where the gamma-function form cancels.

    References:
        [EQ-NUM-TRAP-001], [REF-COLES-2001, Section 3.1].

    """
    step = 0.05
    nodes = [-6.0 + step * index for index in range(int(206.0 / step) + 1)]
    weights: list[float] = []
    values: list[float] = []
    for g in nodes:
        weights.append(math.exp(-g - math.exp(-g)))
        values.append(math.expm1(xi * g) / xi if xi != 0.0 else g)
    total = math.fsum(weights)
    center = math.fsum(w * z for w, z in zip(weights, values, strict=True)) / total
    return tuple(
        math.fsum(w * (z - center) ** r for w, z in zip(weights, values, strict=True))
        / total
        for r in range(2, orders + 1)
    )


def _gev_central_moments(xi: float, orders: int) -> tuple[float, ...]:
    """Return central moments ``2..orders`` of the standard GEV law.

    ``E[(Z - EZ)^r] = xi^-r sum_i C(r, i) g_i (-g_1)^(r - i)`` with
    ``g_i = Gamma(1 - i xi)`` for ``|xi| >= 0.1``; quadrature otherwise.
    Moments of order ``r`` exist only for ``xi < 1/r``.

    References:
        [EQ-INF-MOM-004], [REF-COLES-2001, Section 3.1].

    """
    if xi * orders >= 1.0:
        return tuple(
            math.inf if xi * r >= 1.0 else math.nan for r in range(2, orders + 1)
        )
    if abs(xi) < _GEV_QUADRATURE_THRESHOLD:
        return _gev_central_moments_quadrature(xi, orders)
    gammas = [1.0] + [math.gamma(1.0 - i * xi) for i in range(1, orders + 1)]
    first = gammas[1]
    moments = []
    for r in range(2, orders + 1):
        total = math.fsum(
            math.comb(r, i) * gammas[i] * (-first) ** (r - i) for i in range(r + 1)
        )
        moments.append(total / xi**r)
    return tuple(moments)


# ---------------------------------------------------------------------------
# Family rules
# ---------------------------------------------------------------------------


def _require_positive(value: float, message: str) -> float:
    """Return ``value`` when it is finite and positive.

    Raises:
        ValueError: With ``message`` otherwise.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(message)
    return value


def _normal_inverse(moments: Vector) -> Vector:
    """Return ``(m, sqrt(v))``.

    References:
        [EQ-INF-MOM-002]. No new equation is evaluated.

    """
    mean, variance = moments
    return (mean, math.sqrt(_require_positive(variance, "variance must be positive")))


def _normal_central(theta: Vector, order: int) -> float:
    """Return a central moment of ``Normal(mu, sigma)``.

    References:
        [REF-JKB-1994, Chapter 13]. No new equation is evaluated.

    """
    variance = theta[1] ** 2
    return _central_from_cumulants(variance, 0.0, 0.0, order)


def _lognormal_forward(theta: Vector) -> Vector:
    """Return ``(exp(mu + s^2/2), mean^2 expm1(s^2))``.

    References:
        [EQ-DIST-LN-004]. No new equation is evaluated.

    """
    mu, sigma = theta
    mean = math.exp(mu + 0.5 * sigma * sigma)
    return (mean, mean * mean * math.expm1(sigma * sigma))


def _lognormal_inverse(moments: Vector) -> Vector:
    """Return ``(log m - s2/2, sqrt(s2))`` with ``s2 = log(1 + v/m^2)``.

    References:
        [EQ-INF-MOM-002]. No new equation is evaluated.

    """
    mean, variance = moments
    _require_positive(mean, "lognormal moment matching requires a positive mean")
    _require_positive(variance, "variance must be positive")
    log_variance = math.log1p(variance / (mean * mean))
    return (math.log(mean) - 0.5 * log_variance, math.sqrt(log_variance))


def _lognormal_central(theta: Vector, order: int) -> float:
    """Return a central moment of ``LogNormal(mu, sigma)``.

    With ``w = exp(s^2)`` and mean ``m``: ``mu_2 = m^2 (w - 1)``,
    ``mu_3 = m^3 (w - 1)^2 (w + 2)``,
    ``mu_4 = m^4 (w - 1)^2 (w^4 + 2 w^3 + 3 w^2 - 3)``.

    References:
        [REF-JKB-1994, Chapter 14]. No new equation is evaluated.

    """
    mean, _ = _lognormal_forward(theta)
    excess = math.expm1(theta[1] ** 2)
    w = excess + 1.0
    if order == 2:
        return mean**2 * excess
    if order == 3:
        return mean**3 * excess**2 * (w + 2.0)
    if order == 4:
        return mean**4 * excess**2 * (w**4 + 2.0 * w**3 + 3.0 * w**2 - 3.0)
    raise ValueError(f"central moment of order {order} is not provided")


def _exponential_inverse(moments: Vector) -> Vector:
    """Return the rate ``1/m``.

    References:
        [EQ-INF-MOM-002]. No new equation is evaluated.

    """
    mean = _require_positive(
        moments[0], "exponential matching requires a positive mean"
    )
    return (1.0 / mean,)


def _exponential_central(theta: Vector, order: int) -> float:
    """Return a central moment of ``Exponential(rate)`` (``k2 = 1/l^2``).

    References:
        [REF-JKB-1994, Chapter 19]. No new equation is evaluated.

    """
    scale = 1.0 / theta[0]
    return _central_from_cumulants(scale**2, 2.0 * scale**3, 6.0 * scale**4, order)


def _gamma_inverse(moments: Vector) -> Vector:
    """Return ``(m^2/v, m/v)``.

    References:
        [EQ-INF-MOM-002]. No new equation is evaluated.

    """
    mean, variance = moments
    _require_positive(mean, "gamma moment matching requires a positive mean")
    _require_positive(variance, "variance must be positive")
    return (mean * mean / variance, mean / variance)


def _gamma_central(theta: Vector, order: int) -> float:
    """Return a central moment of ``Gamma(shape, rate)`` (``k_r = a (r-1)!/b^r``).

    References:
        [REF-JKB-1994, Chapter 17]. No new equation is evaluated.

    """
    shape, rate = theta
    return _central_from_cumulants(
        shape / rate**2, 2.0 * shape / rate**3, 6.0 * shape / rate**4, order
    )


def _weibull_raw(theta: Vector, order: int) -> float:
    """Return ``E[X^r] = scale^r Gamma(1 + r/shape)``.

    References:
        [EQ-DIST-WEIB-004], [REF-JKB-1994, Chapter 21].

    """
    shape, scale = theta
    return scale**order * math.gamma(1.0 + order / shape)


def _weibull_forward(theta: Vector) -> Vector:
    """Return the Weibull mean and variance.

    References:
        [EQ-DIST-WEIB-004], [EQ-DIST-WEIB-005].

    """
    shape, scale = theta
    log_first = math.lgamma(1.0 + 1.0 / shape)
    ratio = math.expm1(math.lgamma(1.0 + 2.0 / shape) - 2.0 * log_first)
    mean = scale * math.exp(log_first)
    return (mean, mean * mean * ratio)


def _weibull_inverse(moments: Vector) -> Vector:
    """Solve ``Gamma(1+2/k)/Gamma(1+1/k)^2 - 1 = v/m^2`` for ``k``, then ``scale``.

    The squared coefficient of variation decreases strictly in ``k``; the
    root is bracketed in ``log k`` on ``[log 0.01, log 10^6]``.

    References:
        [EQ-INF-MOM-003], [REF-JKB-1994, Chapter 21].

    """
    mean, variance = moments
    _require_positive(mean, "Weibull moment matching requires a positive mean")
    _require_positive(variance, "variance must be positive")
    target = math.log(variance / (mean * mean))

    def equation(log_shape: float) -> float:
        """Return ``log CV^2(k) - log(v/m^2)`` at ``k = exp(log_shape)``.

        References:
            [EQ-INF-MOM-003]. No new equation is evaluated.

        """
        shape = math.exp(log_shape)
        excess = math.lgamma(1.0 + 2.0 / shape) - 2.0 * math.lgamma(1.0 + 1.0 / shape)
        return math.log(math.expm1(excess)) - target

    lower, upper = math.log(0.01), math.log(1e6)
    if equation(lower) < 0.0 or equation(upper) > 0.0:
        raise ValueError(
            "the sample coefficient of variation lies outside the Weibull range "
            "for shapes in [0.01, 1e6]"
        )
    root = brent_root(equation, lower, upper, tolerance=1e-15)
    shape = math.exp(root.root)
    return (shape, mean / math.exp(math.lgamma(1.0 + 1.0 / shape)))


def _weibull_central(theta: Vector, order: int) -> float:
    """Return a central moment of the Weibull law from its raw moments.

    References:
        [EQ-DIST-WEIB-004], [REF-JKB-1994, Chapter 21].

    """
    mean = _weibull_raw(theta, 1)
    return math.fsum(
        math.comb(order, i) * _weibull_raw(theta, i) * (-mean) ** (order - i)
        for i in range(order + 1)
    )


def _gumbel_inverse(moments: Vector) -> Vector:
    """Return ``(m - gamma_E beta, beta)`` with ``beta = sqrt(6 v)/pi``.

    References:
        [EQ-INF-MOM-002]. No new equation is evaluated.

    """
    mean, variance = moments
    beta = math.sqrt(6.0 * _require_positive(variance, "variance must be positive"))
    beta /= math.pi
    return (mean - EULER_GAMMA * beta, beta)


def _gumbel_central(theta: Vector, order: int) -> float:
    """Return a central moment of ``Gumbel(mu, beta)``.

    Cumulants ``k2 = pi^2 beta^2/6``, ``k3 = 2 zeta(3) beta^3``,
    ``k4 = pi^4 beta^4/15``.

    References:
        [REF-JKB-1995, Chapter 22]. No new equation is evaluated.

    """
    beta = theta[1]
    return _central_from_cumulants(
        math.pi**2 * beta**2 / 6.0,
        2.0 * _ZETA_3 * beta**3,
        math.pi**4 * beta**4 / 15.0,
        order,
    )


def _logistic_inverse(moments: Vector) -> Vector:
    """Return ``(m, sqrt(3 v)/pi)``.

    References:
        [EQ-INF-MOM-002]. No new equation is evaluated.

    """
    mean, variance = moments
    scale = math.sqrt(3.0 * _require_positive(variance, "variance must be positive"))
    return (mean, scale / math.pi)


def _logistic_central(theta: Vector, order: int) -> float:
    """Return a central moment of ``Logistic(mu, s)``.

    Cumulants ``k2 = pi^2 s^2/3``, ``k3 = 0``, ``k4 = 2 pi^4 s^4/15``.

    References:
        [REF-JKB-1995, Chapter 23]. No new equation is evaluated.

    """
    s = theta[1]
    return _central_from_cumulants(
        math.pi**2 * s**2 / 3.0, 0.0, 2.0 * math.pi**4 * s**4 / 15.0, order
    )


def _beta_forward(theta: Vector) -> Vector:
    """Return the beta mean ``a/(a+b)`` and variance ``ab/((a+b)^2 (a+b+1))``.

    References:
        [REF-JKB-1995, Chapter 25]. No new equation is evaluated.

    """
    a, b = theta
    total = a + b
    return (a / total, a * b / (total * total * (total + 1.0)))


def _beta_inverse(moments: Vector) -> Vector:
    """Return ``(m c, (1 - m) c)`` with ``c = m (1 - m)/v - 1``.

    Raises:
        ValueError: If ``v >= m (1 - m)``, which no beta law attains.

    References:
        [EQ-INF-MOM-002]. No new equation is evaluated.

    """
    mean, variance = moments
    _require_positive(variance, "variance must be positive")
    common = mean * (1.0 - mean) / variance - 1.0
    if not (0.0 < mean < 1.0 and common > 0.0):
        raise ValueError(
            "beta moment matching requires 0 < m < 1 and v < m (1 - m); "
            "the sample variance is too large for a beta law"
        )
    return (mean * common, (1.0 - mean) * common)


def _beta_central(theta: Vector, order: int) -> float:
    """Return a central moment of ``Beta(a, b)`` from its raw moments.

    ``E[X^r] = prod_(j<r) (a + j)/(a + b + j)``.

    References:
        [REF-JKB-1995, Chapter 25]. No new equation is evaluated.

    """
    a, b = theta
    raw = [1.0]
    for j in range(order):
        raw.append(raw[-1] * (a + j) / (a + b + j))
    mean = raw[1]
    return math.fsum(
        math.comb(order, i) * raw[i] * (-mean) ** (order - i) for i in range(order + 1)
    )


def _uniform_forward(theta: Vector) -> Vector:
    """Return ``((a + b)/2, (b - a)^2/12)``.

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3]. No new equation is evaluated.

    """
    lower, upper = theta
    return (0.5 * (lower + upper), (upper - lower) ** 2 / 12.0)


def _uniform_inverse(moments: Vector) -> Vector:
    """Return ``(m - sqrt(3 v), m + sqrt(3 v))``.

    References:
        [EQ-INF-MOM-002]. No new equation is evaluated.

    """
    mean, variance = moments
    half = math.sqrt(3.0 * _require_positive(variance, "variance must be positive"))
    return (mean - half, mean + half)


def _uniform_central(theta: Vector, order: int) -> float:
    """Return the central moment ``w^r / (2^r (r + 1))`` (even ``r``) of a uniform law.

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3]. No new equation is evaluated.

    """
    width = theta[1] - theta[0]
    if order % 2:
        return 0.0
    return width**order / (2.0**order * (order + 1))


def _gev_forward(theta: Vector) -> Vector:
    """Return the GEV mean, variance, and third central moment.

    References:
        [EQ-DIST-GEV-003], [EQ-INF-MOM-003].

    """
    mu, sigma, xi = theta
    law = GeneralizedExtremeValue(mu, sigma, xi)
    variance = law.variance
    return (law.mean, variance, _gev_skewness(xi) * variance**1.5)


def _gev_inverse(moments: Vector) -> Vector:
    """Solve the GEV skewness equation for ``xi``, then scale and location.

    The skewness is strictly increasing in ``xi`` on ``(-inf, 1/3)`` and
    covers the real line, so every sample skewness has one root; it is
    bracketed on ``[-50, 1/3 - delta]`` with ``delta`` shrunk until the
    upper skewness exceeds the target.

    References:
        [EQ-INF-MOM-003], [REF-COLES-2001, Section 3.1].

    """
    mean, variance, third = moments
    _require_positive(variance, "variance must be positive")
    target = third / variance**1.5
    if target <= _gev_skewness(_GEV_SHAPE_FLOOR):
        raise ValueError("sample skewness is below the GEV range for xi >= -50")
    gap = 0.01
    while _gev_skewness(1.0 / 3.0 - gap) <= target:
        gap /= 10.0
        if gap < 1e-12:
            raise ValueError("sample skewness is too large for a GEV law")
    root = brent_root(
        lambda xi: _gev_skewness(xi) - target,
        _GEV_SHAPE_FLOOR,
        1.0 / 3.0 - gap,
        tolerance=1e-15,
    )
    xi = root.root
    standard = GeneralizedExtremeValue(0.0, 1.0, xi)
    sigma = math.sqrt(variance / standard.variance)
    return (mean - sigma * standard.mean, sigma, xi)


def _gev_central(theta: Vector, order: int) -> float:
    """Return a central moment of ``GEV(mu, sigma, xi)`` (``inf`` if absent).

    References:
        [EQ-INF-MOM-004], [REF-COLES-2001, Section 3.1].

    """
    _, sigma, xi = theta
    if xi * order >= 1.0:
        return math.inf
    return sigma**order * _gev_central_moments(xi, order)[order - 2]


_RULES: dict[str, tuple[DistributionFamily, _MomentRule]] = {
    "Normal": (
        NORMAL_FAMILY,
        _MomentRule(lambda t: (t[0], t[1] ** 2), _normal_inverse, _normal_central),
    ),
    "LogNormal": (
        LOGNORMAL_FAMILY,
        _MomentRule(_lognormal_forward, _lognormal_inverse, _lognormal_central),
    ),
    "Exponential": (
        EXPONENTIAL_FAMILY,
        _MomentRule(
            lambda t: (1.0 / t[0],), _exponential_inverse, _exponential_central
        ),
    ),
    "Gamma": (
        GAMMA_FAMILY,
        _MomentRule(
            lambda t: (t[0] / t[1], t[0] / t[1] ** 2), _gamma_inverse, _gamma_central
        ),
    ),
    "Weibull": (
        WEIBULL_FAMILY,
        _MomentRule(_weibull_forward, _weibull_inverse, _weibull_central),
    ),
    "Gumbel": (
        GUMBEL_FAMILY,
        _MomentRule(
            lambda t: (t[0] + EULER_GAMMA * t[1], math.pi**2 * t[1] ** 2 / 6.0),
            _gumbel_inverse,
            _gumbel_central,
        ),
    ),
    "GEV": (GEV_FAMILY, _MomentRule(_gev_forward, _gev_inverse, _gev_central)),
    "Logistic": (
        LOGISTIC_FAMILY,
        _MomentRule(
            lambda t: (t[0], math.pi**2 * t[1] ** 2 / 3.0),
            _logistic_inverse,
            _logistic_central,
        ),
    ),
    "Beta": (BETA_FAMILY, _MomentRule(_beta_forward, _beta_inverse, _beta_central)),
    "Uniform": (
        UNIFORM_FAMILY,
        _MomentRule(_uniform_forward, _uniform_inverse, _uniform_central),
    ),
}

MOMENT_FAMILIES: tuple[DistributionFamily, ...] = tuple(
    family for family, _ in _RULES.values()
)
"""Ready-made families supported by :func:`fit_distribution_moments`."""


def _rule_for(family: DistributionFamily) -> _MomentRule:
    """Return the moment rule of a ready-made family.

    Raises:
        TypeError: If ``family`` is not a :class:`DistributionFamily`.
        ValueError: If the family has no moment rule.

    References:
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    if not isinstance(family, DistributionFamily):
        raise TypeError("family must be a DistributionFamily")
    entry = _RULES.get(family.name)
    if entry is not None and entry[0] is family:
        return entry[1]
    if family.name == "StudentT":
        raise ValueError(
            "moment matching is not provided for StudentT: the location-scale t "
            "needs the fourth moment (nu > 4) and its kurtosis estimator is "
            "unstable; use fit_distribution (maximum likelihood)"
        )
    known = ", ".join(item.name for item in MOMENT_FAMILIES)
    raise ValueError(
        f"no moment rule for family {family.name!r}; supported ready-made "
        f"families: {known}"
    )


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------


def _sample_moments(values: Iterable[float], count: int) -> Vector:
    """Return the first ``count`` matched sample moments with divisor ``n``.

    ``m = (1/n) sum x_i``, ``v = (1/n) sum (x_i - m)^2``, and
    ``c3 = (1/n) sum (x_i - m)^3``, the fitdistrplus ``mme`` convention.

    Args:
        values: Finite observations (at least two).
        count: Number of moments, ``1``, ``2``, or ``3``.

    Returns:
        ``(m,)``, ``(m, v)``, or ``(m, v, c3)``.

    Raises:
        ValueError: If ``count`` is not 1, 2, or 3, or fewer than two values
            are given.

    Equation:
        [EQ-INF-MOM-001].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.2.1],
        [REF-DELIGNETTE-MULLER-DUTANG-2015].

    """
    sample = tuple(float(value) for value in values)
    if count not in (1, 2, 3):
        raise ValueError("count must be 1, 2, or 3")
    if len(sample) < 2:
        raise ValueError("at least two observations are required")
    n = len(sample)
    mean = math.fsum(sample) / n
    moments = [mean]
    for order in range(2, count + 1):
        moments.append(math.fsum((x - mean) ** order for x in sample) / n)
    return tuple(moments)


def _shifted_raw(rule: _MomentRule, theta: Vector, center: float) -> Vector:
    """Return the raw moments about ``center`` implied by ``theta``.

    ``v1 = m - c``, ``v2 = var + v1^2``, ``v3 = c3 + 3 v1 var + v1^3``.

    References:
        [EQ-INF-MOM-004]. No new equation is evaluated.

    """
    moments = rule.forward(theta)
    v1 = moments[0] - center
    raw = [v1]
    if len(moments) >= 2:
        raw.append(moments[1] + v1 * v1)
    if len(moments) >= 3:
        raw.append(moments[2] + 3.0 * v1 * moments[1] + v1**3)
    return tuple(raw)


def _moment_covariance(
    rule: _MomentRule,
    sample: Vector,
    estimates: Vector,
    center: float,
    covariance: str,
) -> tuple[Matrix, str]:
    """Return the covariance ``A`` of ``((x - c), ..., (x - c)^k)`` per observation.

    ``"model"``: ``A_ij = mu_(i+j) - mu_i mu_j`` from the central moments of
    the fitted law (whose mean equals the sample mean ``c``); ``"empirical"``:
    the divisor-``n`` sample covariance of the powers.

    Raises:
        ValueError: If a model moment of order ``2k`` does not exist.

    References:
        [EQ-INF-MOM-004], [REF-DELIGNETTE-MULLER-DUTANG-2015].

    """
    k = len(estimates)
    if covariance == "empirical":
        n = len(sample)
        powers = [[(x - center) ** order for x in sample] for order in range(1, k + 1)]
        means = [math.fsum(row) / n for row in powers]
        rows = [
            tuple(
                math.fsum(a * b for a, b in zip(powers[i], powers[j], strict=True)) / n
                - means[i] * means[j]
                for j in range(k)
            )
            for i in range(k)
        ]
        return tuple(rows), "empirical (sandwich) moment covariance"
    central = [0.0, 0.0] + [rule.central(estimates, r) for r in range(2, 2 * k + 1)]
    if not all(math.isfinite(value) for value in central):
        raise ValueError(
            f"the fitted law has no finite moment of order {2 * k}, so the "
            "model-based covariance of the sample moments does not exist"
        )
    rows = [
        tuple(central[i + j] - central[i] * central[j] for j in range(1, k + 1))
        for i in range(1, k + 1)
    ]
    return tuple(rows), "model-based moment covariance"


def _delta_covariance(
    rule: _MomentRule,
    kinds: tuple[str, ...],
    sample: Vector,
    estimates: Vector,
    center: float,
    spread: float,
    covariance: str,
) -> tuple[Vector, Matrix | None, str]:
    """Return delta-method standard errors and covariance of moment estimates.

    ``Cov(theta_hat) = J^-1 A J^-T / n`` with ``J = d v(theta)/d theta`` the
    central-difference Jacobian of the raw moments about ``c``. Steps scale
    with the data spread for locations, with the value for positive
    parameters, and with ``max(|theta|, 0.1)`` for real shapes.

    References:
        [EQ-INF-MOM-004], [REF-VAN-DER-VAART-1998, Chapter 4],
        [REF-CASELLA-BERGER-2002, Section 5.5.4].

    """
    k = len(estimates)
    undefined = (math.nan,) * k
    try:
        moment_cov, label = _moment_covariance(
            rule, sample, estimates, center, covariance
        )
    except (ValueError, ArithmeticError) as error:
        return undefined, None, f"standard errors unavailable: {error}"
    scales = tuple(
        max(abs(value), spread)
        if kind == "location"
        else (abs(value) if kind == "positive" else max(abs(value), 0.1))
        for kind, value in zip(kinds, estimates, strict=True)
    )
    try:
        jacobian, _ = central_jacobian(
            lambda theta: _shifted_raw(rule, theta, center),
            estimates,
            typical_scale=scales,
        )
        columns = [
            lu_solve(jacobian, tuple(1.0 if i == j else 0.0 for i in range(k)))
            for j in range(k)
        ]
    except (ValueError, ArithmeticError) as error:
        return undefined, None, f"standard errors unavailable: {error}"
    inverse = tuple(tuple(columns[j][i] for j in range(k)) for i in range(k))
    n = len(sample)
    matrix = tuple(
        tuple(value / n for value in row) for row in sandwich(inverse, moment_cov)
    )
    errors = tuple(math.sqrt(max(matrix[i][i], 0.0)) for i in range(k))
    return errors, matrix, f"delta-method standard errors, {label}"


def fit_distribution_moments(
    values: Iterable[object],
    family: DistributionFamily,
    *,
    covariance: MomentCovariance = "model",
) -> MomentFitResult:
    """Fit a ready-made family by the method of moments.

    The first ``k`` moments of the family are equated to the sample mean,
    the population variance ``v = (1/n) sum (x_i - m)^2``, and (for the GEV)
    the population third central moment; this is exactly
    ``fitdistrplus::fitdist(x, distr, method = "mme")``. Closed forms:
    Normal ``(m, sqrt v)``; LogNormal ``s2 = log(1 + v/m^2)``,
    ``(log m - s2/2, sqrt s2)`` (moments of ``x``, not of ``log x``);
    Exponential ``1/m``; Gamma ``(m^2/v, m/v)``; Uniform
    ``m -+ sqrt(3v)``; Logistic ``(m, sqrt(3v)/pi)``; Gumbel
    ``beta = sqrt(6v)/pi``, ``mu = m - gamma_E beta``; Beta
    ``c = m(1-m)/v - 1``, ``(m c, (1-m) c)``. The Weibull shape solves the
    coefficient-of-variation equation and the GEV shape the skewness
    equation, both by Brent's method.

    Standard errors use the delta method: with ``v(theta)`` the raw moments
    about the sample mean and ``J`` its Jacobian,
    ``Cov(theta_hat) = J^-1 A J^-T / n``. ``covariance="model"`` takes
    ``A`` from the fitted law (fitdistrplus ``calcvcov``, valid when the
    family is correct and has ``2k`` finite moments); ``"empirical"`` takes
    the sample covariance of the powers (valid under misspecification when
    the data law has ``2k`` finite moments). Moment estimates are generally
    less efficient than maximum likelihood, and a Uniform or GEV moment fit
    may exclude observations from its support (log-likelihood ``-inf``).

    Args:
        values: Finite observations inside the family's sample domain with at
            least two distinct values (three for the GEV).
        family: One of :data:`MOMENT_FAMILIES` (the ready-made Normal,
            LogNormal, Exponential, Gamma, Weibull, Gumbel, GEV, Logistic,
            Beta, and Uniform families).
        covariance: ``"model"`` or ``"empirical"`` moment covariance.

    Returns:
        Estimates, delta-method standard errors and covariance, the matched
        sample moments, the log-likelihood at the estimate, and the law.

    Raises:
        TypeError: If ``family`` is not a :class:`DistributionFamily`.
        ValueError: If the family has no moment rule (Student t, custom
            families), the sample is outside the family domain, the moment
            equations have no solution in the parameter space (e.g. a
            coefficient of variation outside the Weibull range), or
            ``covariance`` is unknown.

    Equation:
        [EQ-INF-MOM-001], [EQ-INF-MOM-002], [EQ-INF-MOM-003],
        [EQ-INF-MOM-004].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.2.1],
        [REF-DELIGNETTE-MULLER-DUTANG-2015],
        [REF-VAN-DER-VAART-1998, Chapter 4].

    """
    rule = _rule_for(family)
    if covariance not in _COVARIANCES:
        raise ValueError(f"covariance must be one of {_COVARIANCES}")
    sample = _validated_fit_sample(values, family, "auto")
    k = len(family.parameter_names)
    if k == 3 and len(set(sample)) < 3:
        raise ValueError(
            f"{family.name} moment matching needs at least {k} distinct values"
        )
    moments = _sample_moments(sample, k)
    estimates = tuple(float(value) for value in rule.inverse(moments))
    law = family.build(estimates)
    try:
        log_likelihood = math.fsum(law.log_pdf(x) for x in sample)
    except (ValueError, ArithmeticError):
        log_likelihood = -math.inf
    if math.isnan(log_likelihood):
        log_likelihood = -math.inf
    spread = math.sqrt(moments[1]) if k >= 2 else abs(moments[0])
    errors, matrix, note = _delta_covariance(
        rule, family.parameter_kinds, sample, estimates, moments[0], spread, covariance
    )
    if not math.isfinite(log_likelihood):
        note += "; an observation lies outside the fitted support"
    return MomentFitResult(
        family=family.name,
        parameter_names=family.parameter_names,
        estimates=estimates,
        standard_errors=errors,
        covariance=matrix,
        moment_names=_MOMENT_NAMES[:k],
        sample_moments=moments,
        log_likelihood=log_likelihood,
        sample_size=len(sample),
        covariance_method=covariance,
        message=note,
        distribution=law,
    )
