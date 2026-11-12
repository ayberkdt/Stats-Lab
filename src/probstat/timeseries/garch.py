"""GARCH(p, q) volatility models: maximum likelihood, forecasts, diagnostics.

Model (Bollerslev 1986, with Bollerslev's order convention)::

    y_t = mu + e_t,   e_t = sigma_t z_t,   z_t i.i.d. with mean 0, variance 1,
    sigma_t^2 = omega + sum_{i=1}^{q} alpha_i e_(t-i)^2
                      + sum_{j=1}^{p} beta_j sigma_(t-j)^2,

so ``q`` counts the ARCH terms (lagged squared innovations) and ``p`` the
GARCH terms (lagged conditional variances); GARCH(0, q) is Engle's ARCH(q).
With ``omega > 0`` and non-negative ``alpha`` and ``beta`` the process is
covariance stationary exactly when the persistence
``sum alpha_i + sum beta_j < 1``, and its unconditional variance is then
``omega / (1 - sum alpha_i - sum beta_j)`` [EQ-TS-GARCH-001].

**Estimation.** The innovations ``z_t`` are standard normal or standardized
Student-t with ``nu > 2`` degrees of freedom (unit variance; Bollerslev 1987).
The conditional log-likelihood [EQ-TS-GARCH-002] is maximized by BFGS over
unconstrained coordinates that map onto the stationary region: ``omega`` is
the exponential of a coordinate, the persistence is a logistic image in
``(0, 1)``, and its split among the ``p + q`` coefficients is a softmax, so
every iterate satisfies ``omega > 0``, ``alpha_i, beta_j > 0`` and
``sum alpha + sum beta < 1``. The recursion starts from the backcast
``e_s^2 = sigma_s^2 = (1/n) sum_t (y_t - mu)^2`` for ``s <= 0``, the sample
mean of the squared residuals at the current ``mu``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inference import chi_square_sf, information_criteria
from probstat.linalg import Matrix, inverse_spd, solve_spd
from probstat.optimization import central_hessian, minimize_bfgs

from .descriptive import LjungBoxResult, ljung_box

Vector = tuple[float, ...]
_LOG_TWO_PI = math.log(2.0 * math.pi)
_DISTRIBUTIONS = ("normal", "t")


def _logistic(value: float) -> float:
    """Return ``1 / (1 + exp(-value))`` without overflow.

    Args:
        value: Real number.

    Returns:
        Value in ``[0, 1]``.

    References:
        Project numerical policy [REF-HIGHAM-2002]. No equation is evaluated.

    """
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _orders(p: int, q: int) -> tuple[int, int]:
    """Validate the GARCH orders.

    Args:
        p: Number of lagged conditional variances (GARCH terms).
        q: Number of lagged squared innovations (ARCH terms).

    Returns:
        The validated ``(p, q)``.

    Raises:
        TypeError: If an order is not an integer.
        ValueError: If ``q < 1``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    p_value = require_non_negative_integer(p, name="p")
    q_value = require_non_negative_integer(q, name="q")
    if q_value < 1:
        raise ValueError("q (the ARCH order) must be at least 1")
    return p_value, q_value


def _distribution(name: str) -> str:
    """Validate the innovation distribution name.

    Args:
        name: ``"normal"`` or ``"t"``.

    Returns:
        The validated name.

    Raises:
        ValueError: If the name is unknown.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if name not in _DISTRIBUTIONS:
        raise ValueError("distribution must be 'normal' or 't'")
    return name


def garch_variances(
    residuals: Sequence[float],
    omega: float,
    alpha: Sequence[float],
    beta: Sequence[float],
    *,
    backcast: float | None = None,
) -> Vector:
    """Return the GARCH conditional variances ``sigma_1^2..sigma_n^2``.

    Args:
        residuals: Innovations ``e_1..e_n`` (observations minus the mean).
        omega: Constant ``omega``.
        alpha: ARCH coefficients ``alpha_1..alpha_q``.
        beta: GARCH coefficients ``beta_1..beta_p``.
        backcast: Pre-sample value of ``e_s^2`` and ``sigma_s^2`` for
            ``s <= 0``; default ``(1/n) sum e_t^2``.

    Returns:
        Conditional variances; an entry is not positive only when the
        parameters allow it (negative coefficients), which callers reject.

    Raises:
        ValueError: If ``residuals`` is empty.

    Equation:
        [EQ-TS-GARCH-001].

    References:
        [REF-BOLLERSLEV-1986].

    """
    n = len(residuals)
    if n == 0:
        raise ValueError("residuals must not be empty")
    squares = [e * e for e in residuals]
    start = math.fsum(squares) / n if backcast is None else backcast
    q = len(alpha)
    p = len(beta)
    variances: list[float] = []
    for t in range(n):
        value = omega
        for i in range(q):
            index = t - 1 - i
            value += alpha[i] * (squares[index] if index >= 0 else start)
        for j in range(p):
            index = t - 1 - j
            value += beta[j] * (variances[index] if index >= 0 else start)
        variances.append(value)
    return tuple(variances)


def _t_constant(shape: float) -> float:
    """Return the log normalizing constant of the standardized Student-t law.

    Args:
        shape: Degrees of freedom ``nu > 2``.

    Returns:
        ``log Gamma((nu+1)/2) - log Gamma(nu/2) - (1/2) log(pi (nu - 2))``.

    Equation:
        [EQ-TS-GARCH-002].

    References:
        [REF-BOLLERSLEV-1987].

    """
    return (
        math.lgamma(0.5 * (shape + 1.0))
        - math.lgamma(0.5 * shape)
        - 0.5 * math.log(math.pi * (shape - 2.0))
    )


def garch_log_likelihood(
    series: Iterable[float],
    mean: float,
    omega: float,
    alpha: Sequence[float],
    beta: Sequence[float],
    *,
    distribution: str = "normal",
    shape: float | None = None,
) -> float:
    """Return the conditional GARCH log-likelihood of a series.

    Normal innovations give ``-(1/2) sum [log 2 pi + log sigma_t^2 +
    e_t^2 / sigma_t^2]``; standardized Student-t innovations give
    ``sum [c(nu) - (1/2) log sigma_t^2 - ((nu+1)/2) log(1 + e_t^2 /
    ((nu - 2) sigma_t^2))]`` with ``c(nu) = log Gamma((nu+1)/2) -
    log Gamma(nu/2) - (1/2) log(pi (nu - 2))``. The variances start from the
    backcast described in the module docstring.

    Args:
        series: Observations ``y_1..y_n``.
        mean: Constant mean ``mu``.
        omega: Positive constant ``omega``.
        alpha: ARCH coefficients (``q >= 1``, non-negative).
        beta: GARCH coefficients (``p >= 0``, non-negative).
        distribution: ``"normal"`` or ``"t"``.
        shape: Degrees of freedom ``nu > 2`` for ``distribution="t"``.

    Returns:
        Log-likelihood; ``-inf`` if a conditional variance is not positive.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a parameter is outside its domain or the series has
            fewer than two observations.

    Equation:
        [EQ-TS-GARCH-001], [EQ-TS-GARCH-002].

    References:
        [REF-BOLLERSLEV-1986], [REF-BOLLERSLEV-1987].

    """
    data = coerce_finite_sample(series, name="series")
    if len(data) < 2:
        raise ValueError("series must contain at least two observations")
    family = _distribution(distribution)
    mu = require_finite_real(mean, name="mean")
    w = require_finite_real(omega, name="omega")
    if w <= 0.0:
        raise ValueError("omega must be positive")
    a = coerce_finite_sample(tuple(alpha), name="alpha") if alpha else ()
    b = coerce_finite_sample(tuple(beta), name="beta") if beta else ()
    if not a:
        raise ValueError("alpha must contain at least one coefficient")
    if min(a + b) < 0.0:
        raise ValueError("alpha and beta must be non-negative")
    nu: float | None = None
    if family == "t":
        if shape is None:
            raise ValueError("shape is required for distribution='t'")
        nu = require_finite_real(shape, name="shape")
        if nu <= 2.0:
            raise ValueError("shape must exceed 2")
    return _log_likelihood(data, mu, w, a, b, nu)


def _log_likelihood(
    data: Sequence[float],
    mu: float,
    omega: float,
    alpha: Sequence[float],
    beta: Sequence[float],
    shape: float | None,
) -> float:
    """Evaluate the conditional log-likelihood without validation.

    Args:
        data: Observations.
        mu: Mean.
        omega: Constant.
        alpha: ARCH coefficients.
        beta: GARCH coefficients.
        shape: Student-t degrees of freedom, or ``None`` for normal.

    Returns:
        Log-likelihood or ``-inf`` for a non-positive variance.

    Equation:
        [EQ-TS-GARCH-002].

    References:
        [REF-BOLLERSLEV-1986], [REF-BOLLERSLEV-1987].

    """
    residuals = [y - mu for y in data]
    variances = garch_variances(residuals, omega, alpha, beta)
    if min(variances) <= 0.0:
        return -math.inf
    if shape is None:
        terms = (
            _LOG_TWO_PI + math.log(v) + e * e / v
            for e, v in zip(residuals, variances, strict=True)
        )
        return -0.5 * math.fsum(terms)
    constant = _t_constant(shape)
    scale = shape - 2.0
    half = 0.5 * (shape + 1.0)
    terms = (
        constant - 0.5 * math.log(v) - half * math.log1p(e * e / (scale * v))
        for e, v in zip(residuals, variances, strict=True)
    )
    return math.fsum(terms)


@dataclass(frozen=True, slots=True)
class GARCHForecast:
    """Multi-step GARCH variance forecasts.

    Args:
        mean: Forecast of ``y_(n+h)`` (the constant mean) for every step.
        variance: ``E[sigma_(n+h)^2 | y_1..y_n]`` for ``h = 1..H``.
        volatility: Square roots of ``variance``.

    References:
        [REF-BOLLERSLEV-1986].

    """

    mean: float
    variance: Vector
    volatility: Vector


@dataclass(frozen=True, slots=True)
class ARCHTestResult:
    """Engle's Lagrange-multiplier test for ARCH effects.

    Args:
        statistic: ``n' R^2`` of the regression of ``x_t^2`` on a constant
            and ``x_(t-1)^2..x_(t-m)^2`` (``n'`` usable observations).
        degrees_of_freedom: Number of lags ``m``.
        p_value: Upper chi-square tail probability.

    References:
        [REF-ENGLE-1982].

    """

    statistic: float
    degrees_of_freedom: int
    p_value: float


@dataclass(frozen=True, slots=True)
class GARCHDiagnostics:
    """Checks on the standardized residuals ``z_t = e_t / sigma_t``.

    Args:
        standardized_residuals: ``z_1..z_n``.
        mean: Sample mean of ``z`` (near 0 for a correct model).
        variance: Sample variance of ``z`` with divisor ``n`` (near 1).
        skewness: Sample skewness of ``z``.
        excess_kurtosis: Sample excess kurtosis of ``z`` (near 0 for normal
            innovations, ``6 / (nu - 4)`` for Student-t with ``nu > 4``).
        ljung_box: Ljung-Box test on ``z`` (remaining mean dependence).
        ljung_box_squared: Ljung-Box test on ``z^2`` (remaining volatility
            clustering).
        arch_test: Engle's LM test on ``z`` (remaining ARCH effects).

    References:
        [REF-LJUNG-BOX-1978], [REF-ENGLE-1982], [REF-BOLLERSLEV-1986].

    """

    standardized_residuals: Vector
    mean: float
    variance: float
    skewness: float
    excess_kurtosis: float
    ljung_box: LjungBoxResult
    ljung_box_squared: LjungBoxResult
    arch_test: ARCHTestResult


def arch_lm_test(series: Iterable[float], lags: int = 5) -> ARCHTestResult:
    """Return Engle's LM test for autoregressive conditional heteroskedasticity.

    Regresses ``x_t^2`` on ``1, x_(t-1)^2, ..., x_(t-m)^2`` by least squares
    over the ``n - m`` usable observations; under the null of no ARCH effects
    ``(n - m) R^2`` is asymptotically ``chi^2(m)``. The series is used as
    given (pass demeaned returns or standardized residuals).

    Args:
        series: Observations ``x_1..x_n``.
        lags: Number of lags ``m >= 1``.

    Returns:
        Statistic, degrees of freedom, and p-value.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``lags < 1``, too few observations remain, or the
            squared series is constant.

    Equation:
        [EQ-TS-ARCHLM-001], [EQ-INF-DIST-CHI2-001].

    References:
        [REF-ENGLE-1982].

    """
    data = coerce_finite_sample(series, name="series")
    m = require_non_negative_integer(lags, name="lags")
    if m < 1:
        raise ValueError("lags must be at least 1")
    n = len(data)
    if n - m < m + 3:
        raise ValueError("too few observations for the requested lags")
    squares = [x * x for x in data]
    rows = [[1.0] + [squares[t - j] for j in range(1, m + 1)] for t in range(m, n)]
    target = squares[m:]
    size = m + 1
    gram = tuple(
        tuple(math.fsum(row[i] * row[j] for row in rows) for j in range(size))
        for i in range(size)
    )
    moment = tuple(
        math.fsum(row[i] * y for row, y in zip(rows, target, strict=True))
        for i in range(size)
    )
    coefficients = solve_spd(gram, moment)
    fitted = [
        math.fsum(c * x for c, x in zip(coefficients, row, strict=True)) for row in rows
    ]
    centre = math.fsum(target) / len(target)
    total = math.fsum((y - centre) ** 2 for y in target)
    if total == 0.0:
        raise ValueError("the squared series must not be constant")
    residual = math.fsum((y - f) ** 2 for y, f in zip(target, fitted, strict=True))
    r_squared = max(0.0, 1.0 - residual / total)
    statistic = len(target) * r_squared
    return ARCHTestResult(statistic, m, chi_square_sf(statistic, m))


@dataclass(frozen=True)
class GARCHFit:
    """Maximum-likelihood GARCH(p, q) fit.

    Args:
        order: ``(p, q)``: GARCH (lagged variance) and ARCH (lagged squared
            innovation) orders.
        distribution: ``"normal"`` or ``"t"``.
        mean: Estimated constant mean, or ``0.0`` when it was not estimated.
        omega: Estimated ``omega``.
        alpha: Estimated ARCH coefficients ``alpha_1..alpha_q``.
        beta: Estimated GARCH coefficients ``beta_1..beta_p``.
        shape: Estimated Student-t degrees of freedom, or ``None``.
        log_likelihood: Maximized conditional log-likelihood.
        aic: Akaike information criterion.
        bic: Bayesian information criterion.
        parameter_names: Names aligned with ``estimates``.
        estimates: Estimates aligned with ``parameter_names``.
        standard_errors: Observed-information standard errors or ``None``.
        covariance: Inverse observed information or ``None``.
        converged: Optimizer convergence flag.
        message: Optimizer termination message.
        evaluations: Objective evaluations used by the optimizer.
        series: Observations.
        residuals: ``e_t = y_t - mu``.
        conditional_variances: ``sigma_t^2``.

    References:
        [REF-BOLLERSLEV-1986], [REF-BOLLERSLEV-1987].

    """

    order: tuple[int, int]
    distribution: str
    mean: float
    omega: float
    alpha: Vector
    beta: Vector
    shape: float | None
    log_likelihood: float
    aic: float
    bic: float
    parameter_names: tuple[str, ...]
    estimates: Vector
    standard_errors: Vector | None
    covariance: Matrix | None
    converged: bool
    message: str
    evaluations: int
    series: Vector
    residuals: Vector
    conditional_variances: Vector

    @property
    def persistence(self) -> float:
        """Return ``sum alpha_i + sum beta_j`` (below one by construction).

        Equation:
            [EQ-TS-GARCH-001].

        References:
            [REF-BOLLERSLEV-1986].

        """
        return math.fsum(self.alpha + self.beta)

    @property
    def unconditional_variance(self) -> float:
        """Return ``omega / (1 - sum alpha_i - sum beta_j)``.

        Equation:
            [EQ-TS-GARCH-001].

        References:
            [REF-BOLLERSLEV-1986].

        """
        return self.omega / (1.0 - self.persistence)

    @property
    def standardized_residuals(self) -> Vector:
        """Return ``z_t = e_t / sigma_t``.

        Equation:
            [EQ-TS-GARCH-001].

        References:
            [REF-BOLLERSLEV-1986].

        """
        return tuple(
            e / math.sqrt(v)
            for e, v in zip(self.residuals, self.conditional_variances, strict=True)
        )

    def diagnostics(self, lags: int = 10, arch_lags: int = 5) -> GARCHDiagnostics:
        """Return moment, Ljung-Box, and ARCH-LM checks on ``z_t``.

        Args:
            lags: Lags of the two Ljung-Box tests.
            arch_lags: Lags of the ARCH-LM test.

        Returns:
            Standardized-residual diagnostics. The Ljung-Box tests use no
            degrees-of-freedom correction, so their p-values are approximate
            for residuals of a fitted model.

        Raises:
            ValueError: If a lag count is invalid for the sample size.

        Equation:
            [EQ-TS-LB-001], [EQ-TS-ARCHLM-001].

        References:
            [REF-LJUNG-BOX-1978], [REF-ENGLE-1982].

        """
        z = self.standardized_residuals
        n = len(z)
        centre = math.fsum(z) / n
        spread = math.fsum((x - centre) ** 2 for x in z) / n
        third = math.fsum((x - centre) ** 3 for x in z) / n
        fourth = math.fsum((x - centre) ** 4 for x in z) / n
        return GARCHDiagnostics(
            standardized_residuals=z,
            mean=centre,
            variance=spread,
            skewness=third / spread**1.5,
            excess_kurtosis=fourth / spread**2 - 3.0,
            ljung_box=ljung_box(z, lags),
            ljung_box_squared=ljung_box([x * x for x in z], lags),
            arch_test=arch_lm_test(z, arch_lags),
        )

    def forecast(self, steps: int) -> GARCHForecast:
        """Forecast the conditional variance ``steps`` periods ahead.

        Uses ``sigma_(n+h)^2 = omega + sum_i alpha_i E[e_(n+h-i)^2] +
        sum_j beta_j E[sigma_(n+h-j)^2]`` with ``E[e_s^2] = E[sigma_s^2]``
        for ``s > n`` and the observed values otherwise. For GARCH(1, 1) this
        equals ``sigma_bar^2 + (alpha + beta)^(h-1) (sigma_(n+1)^2 -
        sigma_bar^2)``, which decays geometrically to the unconditional
        variance ``sigma_bar^2``.

        Args:
            steps: Horizon ``H >= 1``.

        Returns:
            Mean, variance, and volatility forecasts.

        Raises:
            ValueError: If ``steps < 1``.

        Equation:
            [EQ-TS-GARCH-003].

        References:
            [REF-BOLLERSLEV-1986].

        """
        horizon = require_non_negative_integer(steps, name="steps")
        if horizon < 1:
            raise ValueError("steps must be at least 1")
        squares = [e * e for e in self.residuals]
        variances = list(self.conditional_variances)
        n = len(squares)
        q = len(self.alpha)
        p = len(self.beta)
        start = math.fsum(squares) / n
        forecasts: list[float] = []
        for h in range(1, horizon + 1):
            index = n - 1 + h
            value = self.omega
            for i in range(q):
                s = index - 1 - i
                if s >= n:
                    value += self.alpha[i] * forecasts[s - n]
                elif s >= 0:
                    value += self.alpha[i] * squares[s]
                else:
                    value += self.alpha[i] * start
            for j in range(p):
                s = index - 1 - j
                if s >= n:
                    value += self.beta[j] * forecasts[s - n]
                elif s >= 0:
                    value += self.beta[j] * variances[s]
                else:
                    value += self.beta[j] * start
            forecasts.append(value)
        return GARCHForecast(
            self.mean, tuple(forecasts), tuple(math.sqrt(v) for v in forecasts)
        )


def _unpack(
    u: Sequence[float],
    p: int,
    q: int,
    include_mean: bool,
    student: bool,
    centre: float,
    scale: float,
) -> tuple[float, float, Vector, Vector, float | None]:
    """Map unconstrained coordinates to ``(mu, omega, alpha, beta, nu)``.

    Layout: ``[mu?] log(omega / s^2), logit(persistence), softmax logits of
    the first p + q - 1 shares, [log(nu - 2)?]``.

    Args:
        u: Unconstrained coordinates.
        p: GARCH order.
        q: ARCH order.
        include_mean: Whether the first coordinate is the mean.
        student: Whether the last coordinate is the Student-t shape.
        centre: Sample mean (location of the mean coordinate).
        scale: Sample variance ``s^2`` (scale of ``omega``).

    Returns:
        Mean, omega, ARCH and GARCH coefficients, and shape.

    Equation:
        [EQ-TS-GARCH-001].

    References:
        [REF-BOLLERSLEV-1986].

    """
    position = 0
    mu = centre
    if include_mean:
        mu = centre + math.sqrt(scale) * u[0]
        position = 1
    omega = scale * math.exp(max(-50.0, min(50.0, u[position])))
    persistence = _logistic(u[position + 1])
    count = p + q
    logits = list(u[position + 2 : position + 1 + count]) + [0.0]
    top = max(logits)
    weights = [math.exp(value - top) for value in logits]
    total = math.fsum(weights)
    shares = [persistence * w / total for w in weights]
    alpha = tuple(shares[:q])
    beta = tuple(shares[q:])
    nu: float | None = None
    if student:
        nu = 2.0 + math.exp(max(-30.0, min(30.0, u[-1])))
    return mu, omega, alpha, beta, nu


def fit_garch(
    series: Iterable[float],
    p: int = 1,
    q: int = 1,
    *,
    distribution: str = "normal",
    include_mean: bool = True,
    max_iterations: int = 300,
) -> GARCHFit:
    """Fit a GARCH(p, q) model by conditional maximum likelihood.

    ``q`` is the number of ARCH terms ``alpha_i e_(t-i)^2`` and ``p`` the
    number of GARCH terms ``beta_j sigma_(t-j)^2`` (Bollerslev 1986; the same
    convention as ``order = c(p, q)`` in R's ``tseries::garch``). The
    optimizer works in coordinates that keep every iterate inside the
    stationary region (module docstring), starting from ``alpha`` totalling
    0.05 and ``beta`` totalling 0.90 (``alpha = 0.95`` when ``p = 0``) and
    ``nu = 8``. Standard errors come from the inverse central-difference
    Hessian of the negative log-likelihood in ``(mu, omega, alpha, beta,
    nu)`` at the estimate; they are ``None`` when the Hessian is not
    positive definite, which happens when an estimate sits on the boundary
    (for example a coefficient near zero).

    Args:
        series: Observations ``y_1..y_n`` (returns, not prices).
        p: GARCH order ``p >= 0``.
        q: ARCH order ``q >= 1``.
        distribution: ``"normal"`` or ``"t"`` (standardized Student-t).
        include_mean: Estimate a constant mean; otherwise ``mu = 0``.
        max_iterations: BFGS iteration limit.

    Returns:
        Fitted model with information criteria, standard errors, residuals,
        and conditional variances.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an order or the distribution is invalid, the series is
            too short, or it is constant.

    Equation:
        [EQ-TS-GARCH-001], [EQ-TS-GARCH-002], [EQ-INF-IC-001],
        [EQ-INF-IC-003], [EQ-OPT-BFGS-001], [EQ-OPT-FD-002].

    References:
        [REF-BOLLERSLEV-1986], [REF-BOLLERSLEV-1987], [REF-HAMILTON-1994,
        Chapter 21].

    """
    data = coerce_finite_sample(series, name="series")
    p, q = _orders(p, q)
    family = _distribution(distribution)
    student = family == "t"
    mean_flag = bool(include_mean)
    parameter_count = int(mean_flag) + 1 + p + q + int(student)
    n = len(data)
    if n < max(30, 5 * parameter_count):
        raise ValueError("too few observations for the requested model")
    centre = math.fsum(data) / n if mean_flag else 0.0
    scale = math.fsum((y - centre) ** 2 for y in data) / n
    if scale == 0.0:
        raise ValueError("series must not be constant")
    alpha_total, beta_total = (0.05, 0.90) if p else (0.95, 0.0)
    shares = [alpha_total / q] * q + ([beta_total / p] * p if p else [])
    persistence = alpha_total + beta_total
    start: list[float] = [0.0] if mean_flag else []
    start.append(math.log(1.0 - persistence))
    start.append(math.log(persistence / (1.0 - persistence)))
    last = shares[-1]
    start.extend(math.log(share / last) for share in shares[:-1])
    if student:
        start.append(math.log(6.0))

    def objective(u: Sequence[float]) -> float:
        """Return the negative log-likelihood per observation.

        References:
            [EQ-TS-GARCH-002], [REF-BOLLERSLEV-1986].

        """
        mu, omega, alpha, beta, nu = _unpack(u, p, q, mean_flag, student, centre, scale)
        value = _log_likelihood(data, mu, omega, alpha, beta, nu)
        if not math.isfinite(value):
            return 1e300
        return -value / n

    result = minimize_bfgs(
        objective, start, gradient_tolerance=1e-7, max_iterations=max_iterations
    )
    mu, omega, alpha, beta, nu = _unpack(
        result.x, p, q, mean_flag, student, centre, scale
    )
    log_likelihood = _log_likelihood(data, mu, omega, alpha, beta, nu)
    names = (
        (("mean",) if mean_flag else ())
        + ("omega",)
        + tuple(f"alpha{i + 1}" for i in range(q))
        + tuple(f"beta{j + 1}" for j in range(p))
        + (("shape",) if student else ())
    )
    estimates = (
        ((mu,) if mean_flag else ())
        + (omega,)
        + alpha
        + beta
        + ((nu,) if nu is not None else ())
    )

    def negative_log_likelihood(theta: Sequence[float]) -> float:
        """Return ``-log L`` in natural coordinates (``inf`` outside the domain).

        References:
            [EQ-TS-GARCH-002], [REF-BOLLERSLEV-1986].

        """
        position = 0
        level = 0.0
        if mean_flag:
            level = theta[0]
            position = 1
        w = theta[position]
        a = tuple(theta[position + 1 : position + 1 + q])
        b = tuple(theta[position + 1 + q : position + 1 + q + p])
        shape = theta[-1] if student else None
        if w <= 0.0 or min(a + b) < 0.0 or (shape is not None and shape <= 2.0):
            return math.inf
        return -_log_likelihood(data, level, w, a, b, shape)

    errors: Vector | None = None
    covariance: Matrix | None = None
    typical = [max(abs(value), 1e-12) for value in estimates]
    if mean_flag:
        typical[0] = max(abs(estimates[0]), math.sqrt(scale))
    try:
        hessian, _ = central_hessian(
            negative_log_likelihood, estimates, typical_scale=typical
        )
        if all(math.isfinite(value) for row in hessian for value in row):
            covariance = inverse_spd(hessian)
            errors = tuple(math.sqrt(covariance[i][i]) for i in range(len(estimates)))
    except ValueError:
        errors, covariance = None, None
    aic, _, bic = information_criteria(log_likelihood, parameter_count, n)
    residuals = tuple(y - mu for y in data)
    return GARCHFit(
        order=(p, q),
        distribution=family,
        mean=mu,
        omega=omega,
        alpha=alpha,
        beta=beta,
        shape=nu,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        parameter_names=names,
        estimates=estimates,
        standard_errors=errors,
        covariance=covariance,
        converged=result.converged,
        message=result.message,
        evaluations=result.evaluations,
        series=data,
        residuals=residuals,
        conditional_variances=garch_variances(residuals, omega, alpha, beta),
    )


@dataclass(frozen=True, slots=True)
class GARCHSimulation:
    """A simulated GARCH path.

    Args:
        series: Observations ``y_1..y_n``.
        conditional_variances: True ``sigma_t^2``.

    References:
        [REF-BOLLERSLEV-1986].

    """

    series: Vector
    conditional_variances: Vector


def simulate_garch(
    omega: float,
    alpha: Sequence[float],
    beta: Sequence[float],
    n: int,
    *,
    mean: float = 0.0,
    distribution: str = "normal",
    shape: float | None = None,
    seed: int | RandomStream | None = None,
    burn_in: int = 500,
) -> GARCHSimulation:
    """Simulate a stationary GARCH(p, q) path with a burn-in.

    The recursion [EQ-TS-GARCH-001] starts with every pre-sample ``e^2`` and
    ``sigma^2`` at the unconditional variance; the first ``burn_in`` values
    are discarded. Innovations come from the child stream ``"garch"``:
    standard normal, or ``N / sqrt(G / nu) * sqrt((nu - 2) / nu)`` with
    ``G ~ chi^2(nu)`` for the standardized Student-t.

    Args:
        omega: Positive constant.
        alpha: Non-negative ARCH coefficients (at least one).
        beta: Non-negative GARCH coefficients.
        n: Number of returned observations ``>= 1``.
        mean: Constant mean ``mu``.
        distribution: ``"normal"`` or ``"t"``.
        shape: Degrees of freedom ``nu > 2`` for ``"t"``.
        seed: Seed or stream; ``None`` is not reproducible.
        burn_in: Discarded initial values.

    Returns:
        Observations and their conditional variances.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a parameter is outside its domain or the persistence
            is not below one.

    Equation:
        [EQ-TS-GARCH-001].

    References:
        [REF-BOLLERSLEV-1986], [REF-BOLLERSLEV-1987].

    """
    w = require_finite_real(omega, name="omega")
    a = coerce_finite_sample(tuple(alpha), name="alpha") if alpha else ()
    b = coerce_finite_sample(tuple(beta), name="beta") if beta else ()
    if w <= 0.0 or not a or min(a + b) < 0.0:
        raise ValueError("need omega > 0, at least one alpha, and alpha, beta >= 0")
    persistence = math.fsum(a + b)
    if persistence >= 1.0:
        raise ValueError("sum(alpha) + sum(beta) must be below 1 (stationarity)")
    count = require_non_negative_integer(n, name="n")
    burn = require_non_negative_integer(burn_in, name="burn_in")
    if count < 1:
        raise ValueError("n must be at least 1")
    mu = require_finite_real(mean, name="mean")
    family = _distribution(distribution)
    nu = 0.0
    if family == "t":
        if shape is None:
            raise ValueError("shape is required for distribution='t'")
        nu = require_finite_real(shape, name="shape")
        if nu <= 2.0:
            raise ValueError("shape must exceed 2")
    generator = resolve_stream(seed).child("garch").generator()
    level = w / (1.0 - persistence)
    q = len(a)
    p = len(b)
    squares = [level] * q
    variances = [level] * max(p, 1)
    series: list[float] = []
    history: list[float] = []
    factor = math.sqrt((nu - 2.0) / nu) if family == "t" else 1.0
    for _ in range(burn + count):
        variance = w
        for i in range(q):
            variance += a[i] * squares[-1 - i]
        for j in range(p):
            variance += b[j] * variances[-1 - j]
        z = generator.gauss(0.0, 1.0)
        if family == "t":
            chi_square = generator.gammavariate(0.5 * nu, 2.0)
            z = z / math.sqrt(chi_square / nu) * factor
        e = math.sqrt(variance) * z
        squares.append(e * e)
        variances.append(variance)
        if len(squares) > q:
            del squares[0]
        if len(variances) > max(p, 1):
            del variances[0]
        series.append(mu + e)
        history.append(variance)
    return GARCHSimulation(tuple(series[burn:]), tuple(history[burn:]))
