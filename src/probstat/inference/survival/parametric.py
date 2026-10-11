"""Parametric survival regression in accelerated-failure-time form.

The model is ``log T_i = x_i' beta + sigma W`` with intercept ``x_i0 = 1`` and
a standard error law ``W``: the minimum extreme-value law for the Weibull
(``sigma = 1`` gives the exponential) and the standard normal law for the
lognormal. Covariates therefore act multiplicatively on time:
``exp(beta_k)`` is the time ratio per unit of covariate ``k``.

With ``z_i = (log t_i - x_i' beta) / sigma`` the censored log-likelihood is
``sum_i delta_i [log f_W(z_i) - log sigma - log t_i] + (1 - delta_i)
log S_W(z_i)``. It is maximized over ``(beta, log sigma)`` by BFGS with the
analytic gradient; standard errors come from the inverse observed
information (a central-difference Jacobian of the analytic gradient).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.distributions import Exponential, LogNormal, Weibull
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, inverse_spd, solve_spd, symmetrize
from probstat.optimization import central_jacobian, minimize_bfgs

from .data import SurvivalData, covariate_rows, require_positive_times

AftFamily = Literal["exponential", "weibull", "lognormal"]
_FAMILIES = ("exponential", "weibull", "lognormal")
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def _log_normal_sf(z: float) -> float:
    """Return ``log(1 - Phi(z))`` accurately in the upper tail.

    For ``z < 30`` the complementary error function is used directly; beyond,
    the asymptotic Mills-ratio series ``log phi(z) - log z +
    log(1 - z^-2 + 3 z^-4)`` avoids underflow.

    References:
        [REF-NIST-DLMF, Section 7.12]. No new equation is evaluated.

    """
    if z < 30.0:
        return math.log(0.5 * math.erfc(z / math.sqrt(2.0)))
    inverse = 1.0 / (z * z)
    return (
        -0.5 * z * z
        - _HALF_LOG_TWO_PI
        - math.log(z)
        + math.log1p(-inverse + 3.0 * inverse * inverse)
    )


def _contribution(family: AftFamily, z: float, event: bool) -> tuple[float, float]:
    """Return ``log f_W(z)`` (event) or ``log S_W(z)`` (censored) and its slope.

    Args:
        family: AFT family label.
        z: Standardized log time.
        event: Whether the observation is an event.

    Returns:
        Value and derivative with respect to ``z``.

    References:
        [EQ-SURV-AFT-001], [REF-KLEIN-MOESCHBERGER-2003, Chapter 12].

    """
    if family == "lognormal":
        if event:
            return -0.5 * z * z - _HALF_LOG_TWO_PI, -z
        log_sf = _log_normal_sf(z)
        mills = math.exp(-0.5 * z * z - _HALF_LOG_TWO_PI - log_sf)
        return log_sf, -mills
    if z > 700.0:
        raise ValueError("extreme-value contribution overflowed")
    exp_z = math.exp(z)
    if event:
        return z - exp_z, 1.0 - exp_z
    return -exp_z, -exp_z


@dataclass(frozen=True, slots=True)
class ParametricSurvivalFit:
    """Maximum-likelihood AFT fit with observed-information standard errors.

    Args:
        family: ``"exponential"``, ``"weibull"``, or ``"lognormal"``.
        parameter_names: ``("intercept", <covariates>..., ["log_sigma"])``.
        estimates: Estimates of ``(beta, log sigma)`` in that order.
        standard_errors: Observed-information standard errors (``nan`` when
            the information is not positive definite).
        covariance: Inverse observed information, or ``None``.
        log_likelihood: Maximized censored log-likelihood (time scale ``t``).
        size: Number of subjects.
        events: Number of observed events.
        converged: Whether BFGS met its tolerance.
        iterations: BFGS iterations.
        message: Optimizer termination message.

    References:
        [EQ-SURV-AFT-001], [REF-KLEIN-MOESCHBERGER-2003, Chapter 12],
        [REF-CASELLA-BERGER-2002, Section 10.1].

    """

    family: str
    parameter_names: tuple[str, ...]
    estimates: Vector
    standard_errors: Vector
    covariance: Matrix | None
    log_likelihood: float
    size: int
    events: int
    converged: bool
    iterations: int
    message: str

    @property
    def coefficient_count(self) -> int:
        """Return the number of regression coefficients (with intercept).

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 12]. No equation is evaluated.

        """
        return len(self.estimates) - (0 if self.family == "exponential" else 1)

    @property
    def coefficients(self) -> Vector:
        """Return ``beta`` (intercept first).

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 12]. No equation is evaluated.

        """
        return self.estimates[: self.coefficient_count]

    @property
    def sigma(self) -> float:
        """Return the AFT scale ``sigma`` (``1`` for the exponential).

        References:
            [EQ-SURV-AFT-001], [REF-KLEIN-MOESCHBERGER-2003, Chapter 12].

        """
        if self.family == "exponential":
            return 1.0
        return math.exp(self.estimates[-1])

    @property
    def sigma_standard_error(self) -> float:
        """Return the delta-method standard error ``sigma * se(log sigma)``.

        References:
            [REF-CASELLA-BERGER-2002, Section 5.5]. No new equation.

        """
        if self.family == "exponential":
            return 0.0
        return self.sigma * self.standard_errors[-1]

    @property
    def aic(self) -> float:
        """Return Akaike's criterion ``2k - 2 log L``.

        References:
            [REF-AKAIKE-1974]. No new equation is evaluated.

        """
        return 2.0 * len(self.estimates) - 2.0 * self.log_likelihood

    def linear_predictor(self, covariates: Sequence[float] = ()) -> float:
        """Return ``eta = x' beta`` for one covariate row (without intercept).

        Args:
            covariates: Covariate values in fit order.

        Returns:
            Location of ``log T`` for that subject.

        Raises:
            ValueError: If the covariate count is wrong.

        Equation:
            [EQ-SURV-AFT-001].

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 12].

        """
        row = tuple(require_finite_real(v, name="covariates") for v in covariates)
        beta = self.coefficients
        if len(row) != len(beta) - 1:
            raise ValueError(f"expected {len(beta) - 1} covariate values")
        return beta[0] + math.fsum(b * v for b, v in zip(beta[1:], row, strict=True))

    def distribution(
        self, covariates: Sequence[float] = ()
    ) -> Exponential | Weibull | LogNormal:
        """Return the fitted lifetime law for one covariate row.

        Args:
            covariates: Covariate values in fit order.

        Returns:
            ``Exponential(rate=exp(-eta))``,
            ``Weibull(shape=1/sigma, scale=exp(eta))``, or
            ``LogNormal(mu=eta, sigma=sigma)``.

        Equation:
            [EQ-SURV-AFT-001].

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 12].

        """
        eta = self.linear_predictor(covariates)
        if self.family == "exponential":
            return Exponential(rate=math.exp(-eta))
        if self.family == "weibull":
            return Weibull(shape=1.0 / self.sigma, scale=math.exp(eta))
        return LogNormal(mu=eta, sigma=self.sigma)

    def survival(self, time: float, covariates: Sequence[float] = ()) -> float:
        """Return the fitted survival probability ``S(t | x)``.

        Args:
            time: Finite non-negative time.
            covariates: Covariate values in fit order.

        Returns:
            ``P(T > t | x)``.

        Equation:
            [EQ-SURV-AFT-001].

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 12].

        """
        return self.distribution(covariates).sf(time)

    def weibull_parameters(self) -> tuple[float, float, float, float]:
        """Return ``(shape, se, scale, se)`` of a covariate-free Weibull fit.

        ``shape = 1/sigma`` with ``se = se(log sigma)/sigma`` and
        ``scale = exp(beta_0)`` with ``se = scale * se(beta_0)`` (delta method).

        Returns:
            Shape, its standard error, scale, its standard error.

        Raises:
            ValueError: If the fit is not a covariate-free Weibull model.

        Equation:
            [EQ-SURV-AFT-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 5.5].

        """
        if self.family != "weibull" or self.coefficient_count != 1:
            raise ValueError("weibull_parameters needs a covariate-free Weibull fit")
        shape = 1.0 / self.sigma
        scale = math.exp(self.estimates[0])
        return (
            shape,
            shape * self.standard_errors[1],
            scale,
            scale * self.standard_errors[0],
        )


def _negative_log_likelihood(
    family: AftFamily,
    theta: Vector,
    log_times: Vector,
    events: tuple[bool, ...],
    design: tuple[tuple[float, ...], ...],
) -> tuple[float, Vector]:
    """Return the negative censored log-likelihood and its gradient.

    References:
        [EQ-SURV-AFT-001], [REF-KLEIN-MOESCHBERGER-2003, Chapter 12].

    """
    width = len(design[0])
    beta = theta[:width]
    log_sigma = 0.0 if family == "exponential" else theta[width]
    if abs(log_sigma) > 50.0:
        raise ValueError("log sigma left the representable range")
    sigma = math.exp(log_sigma)
    value_terms: list[float] = []
    gradient = [0.0] * len(theta)
    for y, event, row in zip(log_times, events, design, strict=True):
        eta = math.fsum(b * x for b, x in zip(beta, row, strict=True))
        z = (y - eta) / sigma
        value, slope = _contribution(family, z, event)
        if event:
            value -= log_sigma + y
        value_terms.append(value)
        for k in range(width):
            gradient[k] -= slope * (-row[k] / sigma)
        if family != "exponential":
            gradient[width] -= -z * slope - (1.0 if event else 0.0)
    total = -math.fsum(value_terms)
    if not math.isfinite(total):
        raise ValueError("non-finite log-likelihood")
    return total, tuple(gradient)


def _start(
    family: AftFamily, log_times: Vector, design: tuple[tuple[float, ...], ...]
) -> Vector:
    """Return least-squares starting values on the log-time scale.

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 12]. No new equation.

    """
    width = len(design[0])
    gram = tuple(
        tuple(
            math.fsum(row[i] * row[j] for row in design) + (1e-8 if i == j else 0.0)
            for j in range(width)
        )
        for i in range(width)
    )
    moments = tuple(
        math.fsum(row[i] * y for row, y in zip(design, log_times, strict=True))
        for i in range(width)
    )
    beta = solve_spd(gram, moments)
    residuals = [
        y - math.fsum(b * x for b, x in zip(beta, row, strict=True))
        for row, y in zip(design, log_times, strict=True)
    ]
    spread = math.sqrt(math.fsum(r * r for r in residuals) / len(residuals))
    if family == "exponential":
        return tuple(beta)
    return (*beta, math.log(spread) if spread > 1e-8 else 0.0)


def fit_parametric_survival(
    data: SurvivalData,
    family: AftFamily = "weibull",
    *,
    covariates: Sequence[Sequence[float]] | None = None,
    covariate_names: Sequence[str] | None = None,
) -> ParametricSurvivalFit:
    """Fit an exponential, Weibull, or lognormal AFT model to censored data.

    Args:
        data: Right-censored sample with strictly positive times.
        family: ``"exponential"``, ``"weibull"``, or ``"lognormal"``.
        covariates: Optional subject-by-covariate rows (no intercept column).
        covariate_names: Optional names; default ``x1, x2, ...``.

    Returns:
        Maximum-likelihood fit with standard errors.

    Raises:
        ValueError: If the family is unknown, a time is zero, there is no
            event, covariates are malformed, or the optimizer cannot evaluate
            the likelihood at its start.

    Equation:
        [EQ-SURV-AFT-001].

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 12],
        [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    if family not in _FAMILIES:
        raise ValueError(f"unknown AFT family {family!r}")
    require_positive_times(data)
    if data.event_count == 0:
        raise ValueError("at least one observed event is required")
    rows = covariate_rows(covariates, data.size, name="covariates")
    width = len(rows[0])
    if covariate_names is None:
        names = tuple(f"x{k + 1}" for k in range(width))
    else:
        names = tuple(covariate_names)
        if len(names) != width:
            raise ValueError("covariate_names must match the covariate count")
    design = tuple((1.0, *row) for row in rows)
    log_times = tuple(math.log(t) for t in data.times)
    events = data.events

    def objective(theta: Vector) -> float:
        """Return the negative log-likelihood.

        References:
            [EQ-SURV-AFT-001]. No new equation is evaluated.

        """
        return _negative_log_likelihood(family, theta, log_times, events, design)[0]

    def gradient(theta: Vector) -> Vector:
        """Return the analytic gradient of the negative log-likelihood.

        References:
            [EQ-SURV-AFT-001]. No new equation is evaluated.

        """
        return _negative_log_likelihood(family, theta, log_times, events, design)[1]

    start = _start(family, log_times, design)
    result = minimize_bfgs(
        objective, start, gradient=gradient, gradient_tolerance=1e-10
    )
    estimates = result.x
    size = len(estimates)
    covariance: Matrix | None
    try:
        jacobian, _ = central_jacobian(gradient, estimates, relative_step=1e-5)
        inverse = inverse_spd(symmetrize(jacobian, relative_tolerance=1e-5))
        covariance = inverse
        errors = tuple(math.sqrt(max(inverse[i][i], 0.0)) for i in range(size))
    except ValueError:
        covariance = None
        errors = (math.nan,) * size
    parameter_names = ("intercept", *names)
    if family != "exponential":
        parameter_names = (*parameter_names, "log_sigma")
    return ParametricSurvivalFit(
        family,
        parameter_names,
        estimates,
        errors,
        covariance,
        -result.value,
        data.size,
        data.event_count,
        result.converged,
        result.iterations,
        result.message,
    )
