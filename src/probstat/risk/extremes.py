"""Extreme value analysis: block maxima, peaks over threshold, declustering.

Two classical routes to the far tail of a series are provided.

* **Block maxima.** :func:`block_maxima` reduces a series to per-block maxima
  and :func:`fit_gev` fits a GEV law by maximum likelihood through
  :func:`probstat.inference.fit_distribution`. :class:`GEVFit` reports return
  levels ``z_T`` with delta-method intervals and profile-likelihood intervals.
* **Peaks over threshold (POT).** Excesses over a high threshold follow a
  generalized Pareto law (Pickands-Balkema-de Haan). :func:`fit_gpd` fits it
  by maximum likelihood or by probability-weighted moments (equivalently
  sample L-moments); :func:`mean_residual_life` and
  :func:`threshold_stability` support the threshold choice;
  :func:`exceedance_rate` models the exceedance count as binomial/Poisson;
  :func:`fit_pot` combines both into ``N``-year return levels with
  delta-method intervals; :func:`decluster_runs` keeps one peak per cluster
  of dependent exceedances.

Return periods are counted in blocks (GEV) or in years via
``observations_per_year`` (POT). Maximum-likelihood estimates are regular only
for ``xi > -1/2``; for ``xi <= -1`` the likelihood is unbounded, and the
intervals reported here are then not meaningful.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.distributions import GeneralizedExtremeValue, GeneralizedPareto
from probstat.foundations import coerce_finite_sample, require_finite_real
from probstat.inference import (
    GEV_FAMILY,
    DistributionFamily,
    FitResult,
    fit_distribution,
)
from probstat.linalg import Matrix, Vector
from probstat.modeling import standard_normal_quantile
from probstat.optimization import brent_root, minimize_nelder_mead

GPDMethod = Literal["mle", "pwm"]
_GPD_METHODS = ("mle", "pwm")
_SERIES_CUTOFF = 0.05
_SERIES_TERMS = 30
_MAX_EXPANSIONS = 40


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def _expm1_ratio(a: float, t: float) -> float:
    """Return ``r(a, t) = expm1(a t)/a`` with the limit ``t`` at ``a = 0``.

    Args:
        a: Real shape-like coefficient.
        t: Real log-scale argument.

    Returns:
        ``expm1(a t)/a``.

    Equation:
        [EQ-EVA-RL-001].

    References:
        [REF-COLES-2001, Section 3.3]. Floating-point form of the Box-Cox map.

    """
    product = a * t
    if product == 0.0:
        return t
    return math.expm1(product) / a


def _expm1_ratio_derivative(a: float, t: float) -> float:
    """Return ``d r(a, t)/da`` without cancellation near ``a = 0``.

    For ``|a t| < 0.05`` the series ``sum_(k>=2) (k-1) t^k a^(k-2)/k!`` is
    used (limit ``t^2/2``); otherwise ``(a t e^(a t) - expm1(a t))/a^2``.

    Args:
        a: Real coefficient.
        t: Real argument.

    Returns:
        Partial derivative with respect to ``a``.

    Equation:
        [EQ-EVA-RL-002].

    References:
        [REF-COLES-2001, Section 3.3].

    """
    product = a * t
    if abs(product) < _SERIES_CUTOFF:
        total = 0.0
        term = t * t / 2.0  # t^k a^(k-2) / k! at k = 2
        for order in range(2, _SERIES_TERMS):
            total += (order - 1) * term
            term *= product / (order + 1)
        return total
    return (product * math.exp(product) - math.expm1(product)) / (a * a)


def _require_confidence(confidence: float) -> float:
    """Validate a two-sided confidence level strictly inside ``(0, 1)``.

    Args:
        confidence: Candidate confidence level.

    Returns:
        The level as a float.

    Raises:
        ValueError: If the level is not strictly inside ``(0, 1)``.

    References:
        Project validation policy. No equation is evaluated.

    """
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    return level


def _require_return_period(return_period: float) -> float:
    """Validate a return period ``T > 1``.

    Args:
        return_period: Candidate return period (blocks or years).

    Returns:
        The return period as a float.

    Raises:
        ValueError: If ``T <= 1`` or non-finite.

    References:
        [REF-COLES-2001, Section 3.3]. No equation is evaluated.

    """
    period = require_finite_real(return_period, name="return_period")
    if not period > 1.0:
        raise ValueError("return_period must exceed one")
    return period


def _quadratic(gradient: Sequence[float], covariance: Matrix) -> float:
    """Return ``g' V g`` for a gradient and a covariance matrix.

    Args:
        gradient: Gradient vector.
        covariance: Square covariance matrix of matching size.

    Returns:
        Delta-method variance (clipped below at zero).

    Raises:
        ArithmeticError: If the covariance contains non-finite entries.

    Equation:
        [EQ-EVA-RL-002].

    References:
        [REF-COLES-2001, Section 2.6].

    """
    total = math.fsum(
        gradient[i] * covariance[i][j] * gradient[j]
        for i in range(len(gradient))
        for j in range(len(gradient))
    )
    if not math.isfinite(total):
        raise ArithmeticError("covariance is unavailable for the delta method")
    return max(total, 0.0)


@dataclass(frozen=True, slots=True)
class ReturnLevel:
    """A return level with a two-sided confidence interval.

    Args:
        return_period: Return period ``T`` (blocks for GEV, years for POT).
        level: Point estimate of the level exceeded on average once per ``T``.
        standard_error: Delta-method standard error (``nan`` for profile
            intervals or when no covariance is available).
        lower: Lower interval endpoint (``-inf``/``nan`` when unavailable).
        upper: Upper interval endpoint (``inf``/``nan`` when unavailable).
        confidence: Nominal two-sided confidence.
        method: ``"delta"``, ``"profile"``, or ``"point"``.

    References:
        [REF-COLES-2001, Sections 3.3 and 4.3].

    """

    return_period: float
    level: float
    standard_error: float
    lower: float
    upper: float
    confidence: float
    method: str


def _delta_return_level(
    period: float,
    level: float,
    gradient: Sequence[float],
    covariance: Matrix,
    confidence: float,
) -> ReturnLevel:
    """Build a symmetric Wald interval ``level +/- z sqrt(g' V g)``.

    Args:
        period: Return period.
        level: Point estimate.
        gradient: Gradient of the level in the parameters.
        covariance: Parameter covariance.
        confidence: Two-sided confidence.

    Returns:
        Delta-method return level.

    Equation:
        [EQ-EVA-RL-002].

    References:
        [REF-COLES-2001, Section 2.6].

    """
    error = math.sqrt(_quadratic(gradient, covariance))
    z = standard_normal_quantile(0.5 * (1.0 + confidence))
    return ReturnLevel(
        period, level, error, level - z * error, level + z * error, confidence, "delta"
    )


# ---------------------------------------------------------------------------
# Block maxima and the GEV
# ---------------------------------------------------------------------------


def block_maxima(values: Iterable[float], block_size: int) -> tuple[float, ...]:
    """Return maxima of consecutive, non-overlapping blocks.

    An incomplete final block is discarded so that every maximum is taken
    over the same number of observations.

    Args:
        values: Finite series in time order.
        block_size: Positive number of observations per block.

    Returns:
        One maximum per complete block.

    Raises:
        TypeError: If ``block_size`` is not an integer.
        ValueError: If ``block_size`` is not positive or no block is complete.

    Equation:
        [EQ-EVA-BM-001].

    References:
        [REF-COLES-2001, Section 3.1].

    """
    series = coerce_finite_sample(values)
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block_size must be an integer")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    count = len(series) // block_size
    if count == 0:
        raise ValueError("the series is shorter than one block")
    return tuple(
        max(series[k * block_size : (k + 1) * block_size]) for k in range(count)
    )


def gev_return_level(mu: float, sigma: float, xi: float, return_period: float) -> float:
    """Return the GEV return level ``z_T`` exceeded with probability ``1/T``.

    ``z_T = mu - sigma r(-xi, log y)`` with ``y = -log(1 - 1/T)`` and
    ``r(a, t) = expm1(a t)/a``; at ``xi = 0`` this is ``mu - sigma log y``.

    Args:
        mu: Finite location.
        sigma: Positive scale.
        xi: Finite shape.
        return_period: Return period ``T > 1`` in blocks.

    Returns:
        Return level.

    Raises:
        ValueError: If a parameter is invalid.

    Equation:
        [EQ-EVA-RL-001].

    References:
        [REF-COLES-2001, Section 3.3].

    """
    law = GeneralizedExtremeValue(mu, sigma, xi)
    period = _require_return_period(return_period)
    log_y = math.log(-math.log1p(-1.0 / period))
    return law.mu - law.sigma * _expm1_ratio(-law.xi, log_y)


def gev_return_level_gradient(
    mu: float, sigma: float, xi: float, return_period: float
) -> Vector:
    """Return ``grad z_T`` with respect to ``(mu, sigma, xi)``.

    ``(1, -r(-xi, l), sigma r_a(-xi, l))`` with ``l = log y``, ``r_a`` the
    derivative of ``r`` in its first argument; the ``xi -> 0`` limit of the
    last entry is ``sigma l^2/2``.

    Args:
        mu: Finite location.
        sigma: Positive scale.
        xi: Finite shape.
        return_period: Return period ``T > 1``.

    Returns:
        Gradient vector of length three.

    Raises:
        ValueError: If a parameter is invalid.

    Equation:
        [EQ-EVA-RL-002].

    References:
        [REF-COLES-2001, Section 3.3].

    """
    law = GeneralizedExtremeValue(mu, sigma, xi)
    period = _require_return_period(return_period)
    log_y = math.log(-math.log1p(-1.0 / period))
    return (
        1.0,
        -_expm1_ratio(-law.xi, log_y),
        law.sigma * _expm1_ratio_derivative(-law.xi, log_y),
    )


def _gev_negative_log_likelihood(
    sample: Vector, mu: float, sigma: float, xi: float
) -> float:
    """Return ``-sum log f_GEV(x_i)``, ``inf`` for invalid parameters.

    Args:
        sample: Block maxima.
        mu: Location.
        sigma: Scale.
        xi: Shape.

    Returns:
        Negative log-likelihood.

    Equation:
        [EQ-DIST-GEV-001], [EQ-EVA-BM-001].

    References:
        [REF-COLES-2001, Section 3.3].

    """
    try:
        law = GeneralizedExtremeValue(mu, sigma, xi)
        total = math.fsum(law.log_pdf(x) for x in sample)
    except (ValueError, ArithmeticError):
        return math.inf
    if math.isnan(total) or total == math.inf:
        return math.inf
    return -total


@dataclass(frozen=True)
class GEVFit:
    """Maximum-likelihood GEV fit to block maxima with return-level tools.

    Args:
        maxima: The fitted block maxima.
        fit: Generic fit record from :func:`probstat.inference.fit_distribution`
            with estimates ``(mu, sigma, xi)`` and observed-information
            covariance.

    References:
        [REF-COLES-2001, Chapter 3].

    """

    maxima: Vector
    fit: FitResult

    @property
    def mu(self) -> float:
        """Return the fitted location.

        References:
            [REF-COLES-2001, Section 3.3]. No equation is evaluated.

        """
        return self.fit.estimates[0]

    @property
    def sigma(self) -> float:
        """Return the fitted scale.

        References:
            [REF-COLES-2001, Section 3.3]. No equation is evaluated.

        """
        return self.fit.estimates[1]

    @property
    def xi(self) -> float:
        """Return the fitted shape.

        References:
            [REF-COLES-2001, Section 3.3]. No equation is evaluated.

        """
        return self.fit.estimates[2]

    @property
    def distribution(self) -> GeneralizedExtremeValue:
        """Return the fitted GEV law.

        References:
            [EQ-DIST-GEV-001], [REF-COLES-2001, Section 3.3].

        """
        return GeneralizedExtremeValue(self.mu, self.sigma, self.xi)

    def return_level(
        self, return_period: float, *, confidence: float = 0.95
    ) -> ReturnLevel:
        """Return ``z_T`` with a delta-method (Wald) interval.

        Args:
            return_period: Return period ``T > 1`` in blocks.
            confidence: Two-sided confidence strictly inside ``(0, 1)``.

        Returns:
            Return level with standard error ``sqrt(grad' V grad)``.

        Raises:
            ValueError: If an argument is invalid.
            ArithmeticError: If the fit has no usable covariance.

        Equation:
            [EQ-EVA-RL-001], [EQ-EVA-RL-002].

        References:
            [REF-COLES-2001, Section 3.3].

        """
        level_conf = _require_confidence(confidence)
        period = _require_return_period(return_period)
        if self.fit.covariance is None:
            raise ArithmeticError("the GEV fit has no covariance matrix")
        level = gev_return_level(self.mu, self.sigma, self.xi, period)
        gradient = gev_return_level_gradient(self.mu, self.sigma, self.xi, period)
        return _delta_return_level(
            period, level, gradient, self.fit.covariance, level_conf
        )

    def profile_log_likelihood(self, level: float, return_period: float) -> float:
        """Return the profile log-likelihood of the return level ``z_T``.

        The GEV is reparameterized by ``(z_T, sigma, xi)`` with
        ``mu = z_T + sigma r(-xi, log y)`` and the log-likelihood is
        maximized over ``(log sigma, xi)`` by Nelder-Mead from several
        starts (the MLE and the Gumbel point).

        Args:
            level: Candidate return level ``z_T``.
            return_period: Return period ``T > 1``.

        Returns:
            Maximized log-likelihood (``-inf`` if no start is feasible).

        Raises:
            ValueError: If an argument is invalid.

        Equation:
            [EQ-EVA-PROF-001].

        References:
            [REF-COLES-2001, Sections 2.6 and 3.3].

        """
        period = _require_return_period(return_period)
        target = require_finite_real(level, name="level")
        value, _ = _profile_maximum(
            self.maxima, target, period, ((math.log(self.sigma), self.xi),)
        )
        return -value

    def profile_return_level(
        self, return_period: float, *, confidence: float = 0.95
    ) -> ReturnLevel:
        """Return ``z_T`` with a profile-likelihood interval.

        The interval is ``{z : 2 (l_max - l_p(z)) <= chi2_1(confidence)}``.
        Each endpoint is bracketed by expanding steps of the delta-method
        standard error and refined by Brent's method. An endpoint that is
        never reached within 40 doublings is reported as ``-inf``/``inf``
        (typical for the upper end with heavy tails and few maxima).

        Args:
            return_period: Return period ``T > 1`` in blocks.
            confidence: Two-sided confidence strictly inside ``(0, 1)``.

        Returns:
            Return level with profile endpoints and ``standard_error = nan``.

        Raises:
            ValueError: If an argument is invalid.

        Equation:
            [EQ-EVA-PROF-001].

        References:
            [REF-COLES-2001, Sections 2.6 and 3.3].

        """
        level_conf = _require_confidence(confidence)
        period = _require_return_period(return_period)
        estimate = gev_return_level(self.mu, self.sigma, self.xi, period)
        z = standard_normal_quantile(0.5 * (1.0 + level_conf))
        critical = z * z
        best = -self.fit.log_likelihood
        starts: list[tuple[float, tuple[float, float]]] = [
            (estimate, (math.log(self.sigma), self.xi))
        ]
        try:
            step = self.return_level(period, confidence=level_conf).standard_error
        except ArithmeticError:
            step = 0.0
        if not (math.isfinite(step) and step > 0.0):
            step = 0.1 * max(self.sigma, abs(estimate) * 1e-3)

        def excess(candidate: float) -> float:
            """Return ``2 (l_max - l_p(z)) - chi2`` with warm starts.

            References:
                [EQ-EVA-PROF-001], [REF-COLES-2001, Section 2.6].

            """
            ordered = sorted(starts, key=lambda item: abs(item[0] - candidate))
            initial = tuple(point for _, point in ordered[:2])
            value, point = _profile_maximum(self.maxima, candidate, period, initial)
            if math.isfinite(value):
                starts.append((candidate, point))
            deviance = 2.0 * (value - best) if math.isfinite(value) else math.inf
            return deviance - critical

        lower = _profile_endpoint(excess, estimate, -step)
        upper = _profile_endpoint(excess, estimate, step)
        return ReturnLevel(
            period, estimate, math.nan, lower, upper, level_conf, "profile"
        )


def _profile_maximum(
    sample: Vector,
    level: float,
    period: float,
    starts: Sequence[tuple[float, float]],
) -> tuple[float, tuple[float, float]]:
    """Minimize the GEV negative log-likelihood at a fixed return level.

    Args:
        sample: Block maxima.
        level: Fixed return level ``z_T``.
        period: Return period ``T``.
        starts: Candidate starts ``(log sigma, xi)``; a Gumbel start is added.

    Returns:
        Minimum negative log-likelihood and its ``(log sigma, xi)``.

    Equation:
        [EQ-EVA-PROF-001].

    References:
        [REF-COLES-2001, Section 3.3], [REF-NELDER-MEAD-1965].

    """
    log_y = math.log(-math.log1p(-1.0 / period))

    def objective(u: Vector) -> float:
        """Return the negative log-likelihood in ``(log sigma, xi)``.

        References:
            [EQ-EVA-PROF-001], [REF-COLES-2001, Section 3.3].

        """
        if abs(u[0]) > 700.0:
            return math.inf
        sigma = math.exp(u[0])
        mu = level + sigma * _expm1_ratio(-u[1], log_y)
        return _gev_negative_log_likelihood(sample, mu, sigma, u[1])

    candidates = list(starts)
    if candidates:
        candidates.append((candidates[0][0], 0.0))
    best_value, best_point = math.inf, candidates[0] if candidates else (0.0, 0.0)
    for start in candidates:
        if not math.isfinite(objective(start)):
            continue
        result = minimize_nelder_mead(
            objective, start, initial_step=0.2, tolerance=1e-11
        )
        if result.value < best_value:
            best_value = result.value
            best_point = (result.x[0], result.x[1])
        if math.isfinite(best_value):
            break
    return best_value, best_point


def _profile_endpoint(
    excess: Callable[[float], float], estimate: float, step: float
) -> float:
    """Find where the profile deviance crosses the chi-square critical value.

    Args:
        excess: Deviance minus critical value; negative at the estimate.
        estimate: Point estimate.
        step: Signed initial step.

    Returns:
        Endpoint, or a signed infinity if no crossing is bracketed.

    Equation:
        [EQ-EVA-PROF-001].

    References:
        [REF-BRENT-1973, Chapter 4].

    """
    inner = estimate
    for _ in range(_MAX_EXPANSIONS):
        outer = inner + step
        if excess(outer) > 0.0:
            root = brent_root(excess, inner, outer, tolerance=1e-9 * (1.0 + abs(outer)))
            return root.root
        inner = outer
        step *= 2.0
    return math.copysign(math.inf, step)


def fit_gev(maxima: Iterable[float]) -> GEVFit:
    """Fit a GEV law to block maxima by maximum likelihood.

    Delegates to :func:`probstat.inference.fit_distribution` with
    :data:`probstat.inference.GEV_FAMILY` (BFGS from a Gumbel-moment start,
    observed-information covariance).

    Args:
        maxima: At least three finite block maxima with two distinct values.

    Returns:
        GEV fit with return-level methods.

    Raises:
        ValueError: If fewer than three maxima are supplied or fitting fails.

    Equation:
        [EQ-EVA-BM-001], [EQ-INF-MLE-GEN-001], [EQ-INF-MLE-GEN-002].

    References:
        [REF-COLES-2001, Section 3.3].

    """
    sample = coerce_finite_sample(maxima, name="maxima")
    if len(sample) < 3:
        raise ValueError("at least three block maxima are required")
    return GEVFit(sample, fit_distribution(sample, GEV_FAMILY))


# ---------------------------------------------------------------------------
# Generalized Pareto: L-moments, PWM, and maximum likelihood
# ---------------------------------------------------------------------------


def sample_l_moments(values: Iterable[float]) -> tuple[float, float, float, float]:
    """Return the unbiased sample L-moments ``(l1, l2, l3, l4)``.

    ``b_r = n^-1 sum_j [C(j-1, r)/C(n-1, r)] x_(j)`` (unbiased probability-
    weighted moments) and ``l1 = b0``, ``l2 = 2b1 - b0``,
    ``l3 = 6b2 - 6b1 + b0``, ``l4 = 20b3 - 30b2 + 12b1 - b0``. The
    L-skewness and L-kurtosis are ``l3/l2`` and ``l4/l2``.

    Args:
        values: At least four finite observations.

    Returns:
        First four sample L-moments.

    Raises:
        ValueError: If fewer than four observations are supplied.

    Equation:
        [EQ-EVA-LMOM-001].

    References:
        [REF-HOSKING-1990].

    """
    ordered = sorted(coerce_finite_sample(values))
    n = len(ordered)
    if n < 4:
        raise ValueError("at least four observations are required")
    b = [0.0, 0.0, 0.0, 0.0]
    for order in range(4):
        weights = (
            math.comb(j, order) / math.comb(n - 1, order) for j in range(n)
        )
        b[order] = math.fsum(w * x for w, x in zip(weights, ordered, strict=True)) / n
    return (
        b[0],
        2.0 * b[1] - b[0],
        6.0 * b[2] - 6.0 * b[1] + b[0],
        20.0 * b[3] - 30.0 * b[2] + 12.0 * b[1] - b[0],
    )


def _gpd_pwm_parameters(sample: Vector) -> tuple[float, float]:
    """Return the PWM/L-moment GPD estimate ``(sigma, xi)`` for excesses.

    Args:
        sample: Non-negative excesses with at least two distinct values.

    Returns:
        ``sigma = l1 (l1/l2 - 1)`` and ``xi = 2 - l1/l2``.

    Raises:
        ValueError: If ``l2`` is not positive.

    Equation:
        [EQ-EVA-PWM-001].

    References:
        [REF-HOSKING-WALLIS-1987], [REF-HOSKING-1990].

    """
    ordered = sorted(sample)
    n = len(ordered)
    b0 = math.fsum(ordered) / n
    b1 = math.fsum(j * x for j, x in enumerate(ordered)) / (n * (n - 1))
    l1, l2 = b0, 2.0 * b1 - b0
    if not l2 > 0.0:
        raise ValueError("PWM estimate needs a positive second L-moment")
    ratio = l1 / l2
    sigma = l1 * (ratio - 1.0)
    if not sigma > 0.0:
        raise ValueError("PWM estimate produced a non-positive scale")
    return sigma, 2.0 - ratio


def _gpd_start(sample: Vector) -> Vector:
    """Return a feasible MLE start from the PWM estimate.

    The shape is clipped to ``[-0.5, 0.9]`` and the scale raised, if needed,
    so the largest excess lies strictly inside the support.

    Args:
        sample: Non-negative excesses.

    Returns:
        Starting values ``(sigma, xi)``.

    References:
        [REF-HOSKING-WALLIS-1987], [REF-COLES-2001, Section 4.3].

    """
    try:
        sigma, xi = _gpd_pwm_parameters(sample)
    except ValueError:
        sigma, xi = max(math.fsum(sample) / len(sample), 1e-12), 0.0
    xi = min(max(xi, -0.5), 0.9)
    largest = max(sample)
    if xi < 0.0 and sigma <= -xi * largest * 1.01:
        sigma = -xi * largest * 1.5
    return (sigma, xi)


GPD_FAMILY = DistributionFamily(
    "GPD",
    ("sigma", "xi"),
    ("positive", "real"),
    lambda p: GeneralizedPareto(0.0, p[0], p[1]),
    _gpd_start,
    "nonnegative",
)
"""Two-parameter GPD for threshold excesses (location fixed at zero)."""


@dataclass(frozen=True)
class GPDFit:
    """A generalized Pareto fit to threshold excesses.

    Args:
        threshold: Threshold ``u`` (the fitted law has location ``u``).
        sigma: Scale estimate.
        xi: Shape estimate.
        covariance: ``2 x 2`` covariance of ``(sigma, xi)`` from the observed
            information, or ``None`` (PWM, or a singular Hessian).
        log_likelihood: GPD log-likelihood of the excesses at the estimate
            (``-inf`` if a PWM estimate excludes an excess).
        sample_size: Number of excesses.
        method: ``"mle"`` or ``"pwm"``.
        converged: Whether the optimizer converged (always ``True`` for PWM).

    References:
        [REF-COLES-2001, Section 4.3], [REF-HOSKING-WALLIS-1987].

    """

    threshold: float
    sigma: float
    xi: float
    covariance: Matrix | None
    log_likelihood: float
    sample_size: int
    method: str
    converged: bool

    @property
    def distribution(self) -> GeneralizedPareto:
        """Return the fitted law of exceedances ``X | X > u``.

        References:
            [EQ-DIST-GPD-001], [REF-COLES-2001, Section 4.2].

        """
        return GeneralizedPareto(self.threshold, self.sigma, self.xi)

    @property
    def standard_errors(self) -> tuple[float, float]:
        """Return standard errors of ``(sigma, xi)`` (``nan`` if unavailable).

        References:
            [EQ-INF-MLE-GEN-002], [REF-COLES-2001, Section 4.3].

        """
        if self.covariance is None:
            return (math.nan, math.nan)
        return (
            math.sqrt(max(self.covariance[0][0], 0.0)),
            math.sqrt(max(self.covariance[1][1], 0.0)),
        )


def threshold_excesses(values: Iterable[float], threshold: float) -> tuple[float, ...]:
    """Return the excesses ``x - u`` of the observations strictly above ``u``.

    Args:
        values: Finite observations.
        threshold: Finite threshold ``u``.

    Returns:
        Positive excesses in input order.

    Raises:
        ValueError: If an input is non-finite.

    Equation:
        [EQ-EVA-GPD-001].

    References:
        [REF-COLES-2001, Section 4.2].

    """
    level = require_finite_real(threshold, name="threshold")
    return tuple(x - level for x in coerce_finite_sample(values) if x > level)


def fit_gpd(
    excesses: Iterable[float],
    *,
    method: GPDMethod = "mle",
    threshold: float = 0.0,
) -> GPDFit:
    """Fit a GPD to non-negative excesses by MLE or by PWM/L-moments.

    ``"mle"`` maximizes the GPD likelihood through
    :func:`probstat.inference.fit_distribution` (BFGS from a feasible PWM
    start) and reports the observed-information covariance; it is regular
    for ``xi > -1/2``. ``"pwm"`` is the Hosking-Wallis probability-weighted
    moment estimator with unbiased ``b_r``, identical to the two-L-moment
    estimator; it exists for ``xi < 1`` and has no covariance here.

    Args:
        excesses: Non-negative excesses with at least two distinct values
            (``>= 3`` for MLE).
        method: ``"mle"`` or ``"pwm"``.
        threshold: Threshold recorded as the location of the fitted law.

    Returns:
        GPD fit record.

    Raises:
        ValueError: If the excesses or method are invalid.

    Equation:
        [EQ-EVA-GPD-001], [EQ-EVA-PWM-001], [EQ-INF-MLE-GEN-002].

    References:
        [REF-COLES-2001, Section 4.3], [REF-HOSKING-WALLIS-1987],
        [REF-HOSKING-1990], [REF-SMITH-1985].

    """
    if method not in _GPD_METHODS:
        raise ValueError(f"method must be one of {_GPD_METHODS}")
    sample = coerce_finite_sample(excesses, name="excesses")
    location = require_finite_real(threshold, name="threshold")
    if any(x < 0.0 for x in sample):
        raise ValueError("excesses must be non-negative")
    if len(set(sample)) < 2:
        raise ValueError("at least two distinct excesses are required")
    if method == "pwm":
        sigma, xi = _gpd_pwm_parameters(sample)
        law = GeneralizedPareto(0.0, sigma, xi)
        log_likelihood = math.fsum(law.log_pdf(x) for x in sample)
        return GPDFit(
            location, sigma, xi, None, log_likelihood, len(sample), "pwm", True
        )
    if len(sample) < 3:
        raise ValueError("maximum likelihood needs at least three excesses")
    fit = fit_distribution(sample, GPD_FAMILY)
    return GPDFit(
        location,
        fit.estimates[0],
        fit.estimates[1],
        fit.covariance,
        fit.log_likelihood,
        len(sample),
        "mle",
        fit.converged,
    )


# ---------------------------------------------------------------------------
# Threshold diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeanResidualLifePoint:
    """One point of the mean residual life (mean excess) plot.

    Args:
        threshold: Threshold ``u``.
        mean_excess: Sample mean of ``x - u`` over ``x > u``.
        lower: Lower normal-theory confidence limit.
        upper: Upper normal-theory confidence limit.
        exceedances: Number of observations above ``u``.

    References:
        [REF-COLES-2001, Section 4.3].

    """

    threshold: float
    mean_excess: float
    lower: float
    upper: float
    exceedances: int


def _default_thresholds(sample: Vector, minimum: int, count: int) -> Vector:
    """Return an even threshold grid leaving at least ``minimum`` exceedances.

    Args:
        sample: Finite observations.
        minimum: Minimum number of exceedances at the top threshold.
        count: Number of grid points.

    Returns:
        Ascending thresholds from the sample minimum.

    Raises:
        ValueError: If the sample is too small.

    References:
        [REF-COLES-2001, Section 4.3]. No equation is evaluated.

    """
    ordered = sorted(sample)
    if len(ordered) <= minimum:
        raise ValueError("too few observations for the requested exceedances")
    low, high = ordered[0], ordered[len(ordered) - minimum - 1]
    if not high > low:
        return (low,)
    return tuple(low + (high - low) * k / (count - 1) for k in range(count))


def mean_residual_life(
    values: Iterable[float],
    thresholds: Iterable[float] | None = None,
    *,
    confidence: float = 0.95,
    min_exceedances: int = 5,
) -> tuple[MeanResidualLifePoint, ...]:
    """Return mean-residual-life data ``(u, e(u))`` with confidence limits.

    Above a threshold where the GPD holds, ``e(u) = (sigma_u0 + xi u)/(1 - xi)``
    is linear in ``u``; the plot is read for the lowest ``u`` beyond which it
    is approximately linear. Limits are ``e(u) +/- z s_u / sqrt(n_u)``.

    Args:
        values: Finite observations.
        thresholds: Thresholds to evaluate; by default 40 even points from the
            minimum to the order statistic leaving ``min_exceedances``.
        confidence: Two-sided confidence strictly inside ``(0, 1)``.
        min_exceedances: Thresholds with fewer exceedances are skipped (``>= 2``).

    Returns:
        Points in threshold order.

    Raises:
        ValueError: If an argument is invalid.

    Equation:
        [EQ-EVA-MRL-001].

    References:
        [REF-COLES-2001, Section 4.3].

    """
    sample = coerce_finite_sample(values)
    level = _require_confidence(confidence)
    if isinstance(min_exceedances, bool) or not isinstance(min_exceedances, int):
        raise TypeError("min_exceedances must be an integer")
    if min_exceedances < 2:
        raise ValueError("min_exceedances must be at least two")
    grid = (
        _default_thresholds(sample, min_exceedances, 40)
        if thresholds is None
        else tuple(sorted(coerce_finite_sample(thresholds, name="thresholds")))
    )
    z = standard_normal_quantile(0.5 * (1.0 + level))
    points: list[MeanResidualLifePoint] = []
    for u in grid:
        excess = [x - u for x in sample if x > u]
        k = len(excess)
        if k < min_exceedances:
            continue
        mean = math.fsum(excess) / k
        spread = math.sqrt(math.fsum((e - mean) ** 2 for e in excess) / (k - 1))
        half = z * spread / math.sqrt(k)
        points.append(MeanResidualLifePoint(u, mean, mean - half, mean + half, k))
    return tuple(points)


@dataclass(frozen=True, slots=True)
class ThresholdStabilityPoint:
    """GPD estimates at one threshold for the parameter-stability plot.

    Args:
        threshold: Threshold ``u``.
        exceedances: Number of excesses fitted.
        xi: Shape estimate.
        xi_lower: Lower Wald limit for ``xi``.
        xi_upper: Upper Wald limit for ``xi``.
        modified_scale: ``sigma* = sigma_u - xi u``.
        modified_scale_lower: Lower delta-method limit for ``sigma*``.
        modified_scale_upper: Upper delta-method limit for ``sigma*``.

    References:
        [REF-COLES-2001, Section 4.3].

    """

    threshold: float
    exceedances: int
    xi: float
    xi_lower: float
    xi_upper: float
    modified_scale: float
    modified_scale_lower: float
    modified_scale_upper: float


def threshold_stability(
    values: Iterable[float],
    thresholds: Iterable[float],
    *,
    confidence: float = 0.95,
    min_exceedances: int = 10,
) -> tuple[ThresholdStabilityPoint, ...]:
    """Return GPD shape and modified-scale estimates across thresholds.

    If the GPD holds above ``u0``, the shape ``xi`` and the modified scale
    ``sigma* = sigma_u - xi u`` are constant for ``u > u0``. Limits are Wald
    intervals; ``Var sigma* = V_ss - 2u V_sx + u^2 V_xx``.

    Args:
        values: Finite observations.
        thresholds: Thresholds to fit.
        confidence: Two-sided confidence strictly inside ``(0, 1)``.
        min_exceedances: Thresholds with fewer excesses are skipped (``>= 3``).

    Returns:
        Stability points in threshold order (limits ``nan`` when the
        covariance is unavailable).

    Raises:
        ValueError: If an argument is invalid.

    Equation:
        [EQ-EVA-STAB-001].

    References:
        [REF-COLES-2001, Section 4.3].

    """
    sample = coerce_finite_sample(values)
    level = _require_confidence(confidence)
    if isinstance(min_exceedances, bool) or not isinstance(min_exceedances, int):
        raise TypeError("min_exceedances must be an integer")
    if min_exceedances < 3:
        raise ValueError("min_exceedances must be at least three")
    z = standard_normal_quantile(0.5 * (1.0 + level))
    points: list[ThresholdStabilityPoint] = []
    for u in sorted(coerce_finite_sample(thresholds, name="thresholds")):
        excess = threshold_excesses(sample, u)
        if len(excess) < min_exceedances or len(set(excess)) < 2:
            continue
        fit = fit_gpd(excess, threshold=u)
        modified = fit.sigma - fit.xi * u
        if fit.covariance is None:
            xi_se = scale_se = math.nan
        else:
            v = fit.covariance
            xi_se = math.sqrt(max(v[1][1], 0.0))
            scale_se = math.sqrt(
                max(v[0][0] - 2.0 * u * v[0][1] + u * u * v[1][1], 0.0)
            )
        points.append(
            ThresholdStabilityPoint(
                u,
                len(excess),
                fit.xi,
                fit.xi - z * xi_se,
                fit.xi + z * xi_se,
                modified,
                modified - z * scale_se,
                modified + z * scale_se,
            )
        )
    return tuple(points)


# ---------------------------------------------------------------------------
# Exceedance rate, declustering, and POT return levels
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExceedanceRate:
    """Binomial/Poisson model for the number of threshold exceedances.

    Args:
        exceedances: Number of exceedances (or cluster peaks) ``k``.
        observations: Number of observations ``n``.
        observations_per_year: Observations per year ``n_y``.

    References:
        [REF-COLES-2001, Section 4.3].

    """

    exceedances: int
    observations: int
    observations_per_year: float

    def __post_init__(self) -> None:
        """Validate counts and the sampling frequency.

        Raises:
            TypeError: If a count is not an integer.
            ValueError: If ``0 < k <= n`` or ``n_y > 0`` fails.

        References:
            Project validation policy. No equation is evaluated.

        """
        for name in ("exceedances", "observations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 0 < self.exceedances <= self.observations:
            raise ValueError("need 0 < exceedances <= observations")
        frequency = require_finite_real(
            self.observations_per_year, name="observations_per_year"
        )
        if not frequency > 0.0:
            raise ValueError("observations_per_year must be positive")
        object.__setattr__(self, "observations_per_year", frequency)

    @property
    def probability(self) -> float:
        """Return ``zeta_u = k/n``, the per-observation exceedance probability.

        Equation:
            [EQ-EVA-RATE-001].

        References:
            [REF-COLES-2001, Section 4.3].

        """
        return self.exceedances / self.observations

    @property
    def probability_variance(self) -> float:
        """Return the binomial variance ``zeta_u (1 - zeta_u)/n``.

        Equation:
            [EQ-EVA-RATE-001].

        References:
            [REF-COLES-2001, Section 4.3].

        """
        zeta = self.probability
        return zeta * (1.0 - zeta) / self.observations

    @property
    def annual_rate(self) -> float:
        """Return the mean number of exceedances per year ``lambda = n_y zeta_u``.

        Equation:
            [EQ-EVA-RATE-001].

        References:
            [REF-COLES-2001, Section 4.3].

        """
        return self.observations_per_year * self.probability

    @property
    def annual_rate_standard_error(self) -> float:
        """Return ``n_y sqrt(zeta_u (1 - zeta_u)/n)``.

        Equation:
            [EQ-EVA-RATE-001].

        References:
            [REF-COLES-2001, Section 4.3].

        """
        return self.observations_per_year * math.sqrt(self.probability_variance)

    def probability_of_exceedance(self, years: float = 1.0) -> float:
        """Return ``1 - exp(-lambda t)``, the Poisson chance of any exceedance.

        Args:
            years: Non-negative exposure ``t`` in years.

        Returns:
            Probability of at least one exceedance in ``t`` years.

        Raises:
            ValueError: If ``years`` is negative or non-finite.

        Equation:
            [EQ-EVA-RATE-001], [EQ-RISK-RP-001].

        References:
            [REF-COLES-2001, Section 4.3].

        """
        exposure = require_finite_real(years, name="years")
        if exposure < 0.0:
            raise ValueError("years must be non-negative")
        return -math.expm1(-self.annual_rate * exposure)


def exceedance_rate(
    values: Iterable[float], threshold: float, *, observations_per_year: float = 1.0
) -> ExceedanceRate:
    """Return the exceedance-rate model for observations above ``u``.

    Args:
        values: Finite observations.
        threshold: Finite threshold ``u``.
        observations_per_year: Positive sampling frequency ``n_y``.

    Returns:
        Exceedance-rate record.

    Raises:
        ValueError: If no observation exceeds ``u`` or an input is invalid.

    Equation:
        [EQ-EVA-RATE-001].

    References:
        [REF-COLES-2001, Section 4.3].

    """
    sample = coerce_finite_sample(values)
    count = len(threshold_excesses(sample, threshold))
    if count == 0:
        raise ValueError("no observation exceeds the threshold")
    return ExceedanceRate(count, len(sample), observations_per_year)


@dataclass(frozen=True, slots=True)
class DeclusteredExceedances:
    """Cluster peaks of threshold exceedances found by the runs method.

    Args:
        threshold: Threshold ``u``.
        run_length: Run length ``r``.
        peak_indices: Index of the (first) largest value of each cluster.
        peaks: Cluster maxima.
        cluster_sizes: Number of exceedances in each cluster.
        observation_count: Length of the series.

    References:
        [REF-COLES-2001, Section 5.3].

    """

    threshold: float
    run_length: int
    peak_indices: tuple[int, ...]
    peaks: Vector
    cluster_sizes: tuple[int, ...]
    observation_count: int

    @property
    def cluster_count(self) -> int:
        """Return the number of clusters.

        References:
            [REF-COLES-2001, Section 5.3]. No equation is evaluated.

        """
        return len(self.peaks)

    @property
    def exceedance_count(self) -> int:
        """Return the total number of exceedances.

        References:
            [REF-COLES-2001, Section 5.3]. No equation is evaluated.

        """
        return sum(self.cluster_sizes)

    @property
    def extremal_index(self) -> float:
        """Return the runs estimate ``theta = clusters / exceedances``.

        Equation:
            [EQ-EVA-DECL-001].

        References:
            [REF-COLES-2001, Section 5.3].

        """
        return self.cluster_count / self.exceedance_count

    @property
    def excesses(self) -> Vector:
        """Return the cluster-peak excesses ``peak - u``.

        References:
            [EQ-EVA-GPD-001], [REF-COLES-2001, Section 5.3].

        """
        return tuple(peak - self.threshold for peak in self.peaks)


def decluster_runs(
    values: Iterable[float], threshold: float, run_length: int
) -> DeclusteredExceedances:
    """Group exceedances into clusters by the runs method and keep the peaks.

    A cluster starts at an exceedance of ``u`` and ends once ``r``
    consecutive observations do not exceed ``u``.

    Args:
        values: Finite series in time order.
        threshold: Finite threshold ``u``.
        run_length: Positive run length ``r``.

    Returns:
        Cluster peaks, positions, and sizes.

    Raises:
        TypeError: If ``run_length`` is not an integer.
        ValueError: If ``run_length`` is not positive or nothing exceeds ``u``.

    Equation:
        [EQ-EVA-DECL-001].

    References:
        [REF-COLES-2001, Section 5.3].

    """
    series = coerce_finite_sample(values)
    level = require_finite_real(threshold, name="threshold")
    if isinstance(run_length, bool) or not isinstance(run_length, int):
        raise TypeError("run_length must be an integer")
    if run_length < 1:
        raise ValueError("run_length must be positive")
    indices: list[int] = []
    peaks: list[float] = []
    sizes: list[int] = []
    current_index = -1
    current_size = 0
    gap = 0
    for index, value in enumerate(series):
        if value > level:
            if current_size == 0:
                current_index, current_size = index, 1
            else:
                current_size += 1
                if value > series[current_index]:
                    current_index = index
            gap = 0
        elif current_size:
            gap += 1
            if gap >= run_length:
                indices.append(current_index)
                peaks.append(series[current_index])
                sizes.append(current_size)
                current_size = 0
    if current_size:
        indices.append(current_index)
        peaks.append(series[current_index])
        sizes.append(current_size)
    if not peaks:
        raise ValueError("no observation exceeds the threshold")
    return DeclusteredExceedances(
        level, run_length, tuple(indices), tuple(peaks), tuple(sizes), len(series)
    )


def pot_return_level(
    threshold: float,
    sigma: float,
    xi: float,
    exceedance_probability: float,
    observations_per_year: float,
    return_period: float,
) -> float:
    """Return the ``N``-year POT return level.

    ``x_N = u + sigma r(xi, log(m zeta_u))`` with ``m = N n_y`` observations
    and ``r(a, t) = expm1(a t)/a`` (``u + sigma log(m zeta_u)`` at
    ``xi = 0``); defined when ``m zeta_u > 1``.

    Args:
        threshold: Threshold ``u``.
        sigma: Positive GPD scale.
        xi: GPD shape.
        exceedance_probability: ``zeta_u`` in ``(0, 1]``.
        observations_per_year: Positive ``n_y``.
        return_period: Return period ``N`` in years.

    Returns:
        Return level.

    Raises:
        ValueError: If an argument is invalid or ``m zeta_u <= 1``.

    Equation:
        [EQ-EVA-POT-001].

    References:
        [REF-COLES-2001, Section 4.3].

    """
    law = GeneralizedPareto(threshold, sigma, xi)
    log_m_zeta = _log_m_zeta(
        exceedance_probability, observations_per_year, return_period
    )
    return law.mu + law.sigma * _expm1_ratio(law.xi, log_m_zeta)


def _log_m_zeta(zeta: float, frequency: float, period: float) -> float:
    """Return ``log(N n_y zeta_u)`` after validation.

    Args:
        zeta: Exceedance probability per observation.
        frequency: Observations per year.
        period: Return period in years.

    Returns:
        ``log(m zeta_u)``, positive.

    Raises:
        ValueError: If an input is invalid or ``m zeta_u <= 1``.

    Equation:
        [EQ-EVA-POT-001].

    References:
        [REF-COLES-2001, Section 4.3].

    """
    probability = require_finite_real(zeta, name="exceedance_probability")
    per_year = require_finite_real(frequency, name="observations_per_year")
    years = require_finite_real(period, name="return_period")
    if not 0.0 < probability <= 1.0:
        raise ValueError("exceedance_probability must lie in (0, 1]")
    if not (per_year > 0.0 and years > 0.0):
        raise ValueError("observations_per_year and return_period must be positive")
    product = years * per_year * probability
    if not product > 1.0:
        raise ValueError(
            "the return period is too short: fewer than one expected "
            "exceedance, so the level lies below the threshold"
        )
    return math.log(product)


@dataclass(frozen=True)
class PeaksOverThresholdFit:
    """A POT model: GPD excesses plus a binomial/Poisson exceedance rate.

    Args:
        gpd: GPD fit to the (possibly declustered) excesses.
        rate: Exceedance-rate model (clusters per observation if declustered).
        declustered: Declustering record, or ``None``.

    References:
        [REF-COLES-2001, Sections 4.3 and 5.3].

    """

    gpd: GPDFit
    rate: ExceedanceRate
    declustered: DeclusteredExceedances | None = None

    @property
    def threshold(self) -> float:
        """Return the threshold ``u``.

        References:
            [REF-COLES-2001, Section 4.3]. No equation is evaluated.

        """
        return self.gpd.threshold

    def annual_exceedance_rate(self, level: float) -> float:
        """Return ``lambda S_GPD(level)``, the yearly rate of exceeding a level.

        Args:
            level: Level at or above the threshold.

        Returns:
            Mean number of exceedances per year (its inverse is the return
            period of ``level``).

        Raises:
            ValueError: If ``level`` is below the threshold.

        Equation:
            [EQ-EVA-POT-001], [EQ-RISK-RP-001].

        References:
            [REF-COLES-2001, Section 4.3].

        """
        point = require_finite_real(level, name="level")
        if point < self.threshold:
            raise ValueError("level must not lie below the threshold")
        return self.rate.annual_rate * self.gpd.distribution.sf(point)

    def return_level(
        self, return_period: float, *, confidence: float = 0.95
    ) -> ReturnLevel:
        """Return the ``N``-year level with a delta-method interval.

        The gradient in ``(zeta_u, sigma, xi)`` is
        ``(sigma e^(xi l)/zeta_u, r(xi, l), sigma r_a(xi, l))`` with
        ``l = log(m zeta_u)``; ``zeta_u`` is independent of ``(sigma, xi)``
        with binomial variance. A PWM fit has no covariance, so its record
        has ``method = "point"`` and ``nan`` limits.

        Args:
            return_period: Return period ``N`` in years.
            confidence: Two-sided confidence strictly inside ``(0, 1)``.

        Returns:
            Return level.

        Raises:
            ValueError: If an argument is invalid or the level would lie
                below the threshold.

        Equation:
            [EQ-EVA-POT-001], [EQ-EVA-POT-002], [EQ-EVA-RL-002].

        References:
            [REF-COLES-2001, Section 4.3].

        """
        level_conf = _require_confidence(confidence)
        zeta = self.rate.probability
        frequency = self.rate.observations_per_year
        period = require_finite_real(return_period, name="return_period")
        level = pot_return_level(
            self.threshold, self.gpd.sigma, self.gpd.xi, zeta, frequency, period
        )
        if self.gpd.covariance is None:
            return ReturnLevel(
                period, level, math.nan, math.nan, math.nan, level_conf, "point"
            )
        log_m_zeta = _log_m_zeta(zeta, frequency, period)
        sigma, xi = self.gpd.sigma, self.gpd.xi
        gradient = (
            sigma * math.exp(xi * log_m_zeta) / zeta,
            _expm1_ratio(xi, log_m_zeta),
            sigma * _expm1_ratio_derivative(xi, log_m_zeta),
        )
        v = self.gpd.covariance
        covariance = (
            (self.rate.probability_variance, 0.0, 0.0),
            (0.0, v[0][0], v[0][1]),
            (0.0, v[1][0], v[1][1]),
        )
        return _delta_return_level(period, level, gradient, covariance, level_conf)


def fit_pot(
    values: Iterable[float],
    threshold: float,
    *,
    observations_per_year: float = 1.0,
    method: GPDMethod = "mle",
    run_length: int | None = None,
) -> PeaksOverThresholdFit:
    """Fit a peaks-over-threshold model to a series.

    Without ``run_length`` every exceedance enters the GPD fit and the rate.
    With ``run_length = r`` the series is declustered by runs first; the GPD
    is fitted to cluster-peak excesses and ``zeta_u`` counts clusters, so
    return levels refer to independent cluster peaks.

    Args:
        values: Finite series in time order.
        threshold: Finite threshold ``u``.
        observations_per_year: Positive sampling frequency ``n_y``.
        method: GPD estimator, ``"mle"`` or ``"pwm"``.
        run_length: Optional positive run length for declustering.

    Returns:
        POT fit with return-level methods.

    Raises:
        ValueError: If too few excesses exist or an argument is invalid.

    Equation:
        [EQ-EVA-GPD-001], [EQ-EVA-RATE-001], [EQ-EVA-DECL-001].

    References:
        [REF-COLES-2001, Sections 4.3 and 5.3].

    """
    series = coerce_finite_sample(values)
    level = require_finite_real(threshold, name="threshold")
    declustered: DeclusteredExceedances | None = None
    if run_length is None:
        excess = threshold_excesses(series, level)
    else:
        declustered = decluster_runs(series, level, run_length)
        excess = declustered.excesses
    if not excess:
        raise ValueError("no observation exceeds the threshold")
    gpd = fit_gpd(excess, method=method, threshold=level)
    rate = ExceedanceRate(len(excess), len(series), observations_per_year)
    return PeaksOverThresholdFit(gpd, rate, declustered)
