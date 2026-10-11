"""Hierarchical normal (eight-schools) and beta-binomial models.

Hierarchical normal model
    ``y_j | theta_j ~ N(theta_j, s_j^2)`` with known ``s_j``,
    ``theta_j | mu, tau ~ N(mu, tau^2)``, a flat (or normal) prior on ``mu``
    and a prior ``p(tau)`` on ``tau >= 0`` (uniform by default).

    *Exact route* (:func:`hierarchical_normal_posterior`). Integrating out
    ``theta`` and then ``mu`` in closed form leaves the one-dimensional
    marginal posterior ``p(tau | y)`` ([EQ-HIER-002]), which is evaluated on a
    grid and integrated by composite Simpson quadrature. Posterior moments of
    ``mu``, ``tau``, ``theta_j`` and of the shrinkage factors
    ``B_j = s_j^2 / (s_j^2 + tau^2)`` ([EQ-HIER-004]) follow by quadrature,
    and joint draws are obtained by sampling ``tau`` from the gridded
    marginal and then ``mu`` and ``theta`` from their normal conditionals
    ([EQ-HIER-001]). This is the computation of Gelman et al. (BDA3,
    Chapter 5).

    *Sampling route* (:func:`sample_hierarchical_normal`). The existing NUTS
    sampler explores the non-centered parameterization
    ``theta_j = mu + tau z_j``, ``z_j ~ N(0, 1)``, ``tau = exp(eta)``
    ([EQ-HIER-003]), which removes the funnel-shaped dependence between
    ``tau`` and ``theta`` that hampers the centered form when the data are
    weakly informative about ``tau`` (Betancourt and Girolami).

Beta-binomial empirical Bayes
    :func:`fit_beta_binomial` maximizes the beta-binomial marginal likelihood
    of ``y_j ~ Binomial(n_j, p_j)``, ``p_j ~ Beta(alpha, beta)`` and reports
    the shrunken posterior means ``(y_j + alpha) / (n_j + alpha + beta)``
    ([EQ-HIER-BB-001]). Plugging in the point estimate ignores the
    uncertainty in ``(alpha, beta)``.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations import coerce_finite_sample
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.linalg import Vector
from probstat.optimization import OptimizationResult, minimize_bfgs

from .hamiltonian import HamiltonianResult, no_u_turn_sampler
from .mcmc import MCMCResult

LogTauPrior = Callable[[float], float]

_TAIL_RATIO = 1e-14
_MAX_DOUBLINGS = 200
_PROPER_SLOPE = -1.2


def half_cauchy_log_prior(scale: float) -> LogTauPrior:
    """Return the unnormalized log-density of a half-Cauchy prior on ``tau``.

    Args:
        scale: Positive scale ``A``; ``log p(tau) = -log(1 + (tau / A)^2)``.

    Returns:
        Callable ``tau -> log p(tau)`` for ``tau >= 0``.

    Raises:
        ValueError: If ``scale`` is not positive.

    References:
        [REF-GELMAN-2006].

    """
    checked = require_finite_real(scale, name="scale")
    if checked <= 0.0:
        raise ValueError("scale must be positive")

    def log_prior(tau: float) -> float:
        """Evaluate ``-log(1 + (tau/A)^2)``.

        References:
            [REF-GELMAN-2006]. No new equation is introduced.

        """
        return -math.log1p((tau / checked) ** 2)

    return log_prior


def _uniform_log_prior(tau: float) -> float:
    """Return the log-density of the improper uniform prior on ``tau >= 0``.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5]. No equation is evaluated.

    """
    return 0.0


def _checked_data(
    estimates: Iterable[object], standard_errors: Iterable[object]
) -> tuple[Vector, Vector]:
    """Validate group estimates and their known standard errors.

    References:
        Project validation policy. No equation is evaluated.

    """
    y = coerce_finite_sample(estimates, name="estimates")
    s = coerce_finite_sample(standard_errors, name="standard_errors")
    if len(y) != len(s):
        raise ValueError("estimates and standard_errors must have equal length")
    if len(y) < 2:
        raise ValueError("at least two groups are required")
    if any(value <= 0.0 for value in s):
        raise ValueError("standard errors must be positive")
    return tuple(y), tuple(s)


def _checked_mu_prior(
    mu_prior: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Validate an optional ``(mean, sd)`` normal prior on ``mu``.

    References:
        Project validation policy. No equation is evaluated.

    """
    if mu_prior is None:
        return None
    mean = require_finite_real(mu_prior[0], name="mu_prior mean")
    sd = require_finite_real(mu_prior[1], name="mu_prior sd")
    if sd <= 0.0:
        raise ValueError("mu_prior sd must be positive")
    return mean, sd


def shrinkage_factors(
    standard_errors: Iterable[object], tau: float
) -> tuple[float, ...]:
    """Return partial-pooling factors ``B_j = s_j^2 / (s_j^2 + tau^2)``.

    ``B_j = 1`` is complete pooling (``theta_j = mu``), ``B_j = 0`` no pooling.

    Args:
        standard_errors: Positive known standard errors ``s_j``.
        tau: Non-negative population standard deviation.

    Returns:
        One factor per group.

    Equation:
        [EQ-HIER-004].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5], [REF-MORRIS-1983].

    """
    s = coerce_finite_sample(standard_errors, name="standard_errors")
    t = require_finite_real(tau, name="tau")
    if t < 0.0 or any(value <= 0.0 for value in s):
        raise ValueError("tau must be non-negative and standard errors positive")
    return tuple(v * v / (v * v + t * t) for v in s)


# ---------------------------------------------------------------------------
# Exact marginal posterior of tau
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HierarchicalNormalPosterior:
    """Gridded exact posterior of the hierarchical normal model.

    The grid is uniform in ``u = log(1 + tau / c)`` (``c`` is the smallest
    standard error), so the bulk near ``tau = 0`` and a long right tail are
    both resolved. ``weights`` are Simpson weights in ``u`` multiplied by the
    Jacobian ``tau + c`` and the normalized density, so
    ``E[g(tau) | y] = sum_i weights_i g(tau_i)``.

    Args:
        estimates: Group estimates ``y_j``.
        standard_errors: Known standard errors ``s_j``.
        tau_grid: Grid points ``tau_i`` (increasing, starting at zero).
        log_density: Unnormalized ``log p(tau_i | y)``.
        weights: Normalized quadrature weights (sum to one).
        cdf: Trapezoid CDF in ``u`` at each grid point, used for sampling.
        scale: Grid scale ``c``.
        mu_prior: Optional ``(mean, sd)`` normal prior on ``mu``.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5], [REF-RUBIN-1981].

    """

    estimates: Vector
    standard_errors: Vector
    tau_grid: Vector
    log_density: Vector
    weights: Vector
    cdf: Vector
    scale: float
    mu_prior: tuple[float, float] | None

    @property
    def groups(self) -> int:
        """Return the number of groups ``J``.

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5]. No equation is evaluated.

        """
        return len(self.estimates)

    @property
    def names(self) -> tuple[str, ...]:
        """Return draw coordinate names ``mu, tau, theta_1..theta_J``.

        References:
            Project API policy. No equation is evaluated.

        """
        return ("mu", "tau") + tuple(f"theta_{j + 1}" for j in range(self.groups))

    def conditional_mu(self, tau: float) -> tuple[float, float]:
        """Return the mean and variance of ``mu | tau, y``.

        Equation:
            [EQ-HIER-001].

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5].

        """
        return _conditional_mu(self.estimates, self.standard_errors, tau, self.mu_prior)

    def conditional_theta(self, tau: float) -> tuple[Vector, Vector]:
        """Return means and variances of ``theta_j | tau, y`` (``mu`` integrated).

        ``E = (1 - B_j) y_j + B_j mu_hat`` and
        ``Var = (1 - B_j) s_j^2 + B_j^2 V_mu``.

        Equation:
            [EQ-HIER-001], [EQ-HIER-004].

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5].

        """
        mu_hat, v_mu = self.conditional_mu(tau)
        means: list[float] = []
        variances: list[float] = []
        for y, s in zip(self.estimates, self.standard_errors, strict=True):
            b = s * s / (s * s + tau * tau)
            means.append((1.0 - b) * y + b * mu_hat)
            variances.append((1.0 - b) * s * s + b * b * v_mu)
        return tuple(means), tuple(variances)

    def expectation(self, function: Callable[[float], float]) -> float:
        """Return ``E[g(tau) | y]`` by the stored quadrature rule.

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5]. No new equation is introduced.

        """
        return math.fsum(
            w * function(t) for w, t in zip(self.weights, self.tau_grid, strict=True)
            if w > 0.0
        )

    def tau_mean(self) -> float:
        """Return ``E[tau | y]``.

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5]. Equation [EQ-HIER-002].

        """
        return self.expectation(lambda t: t)

    def tau_sd(self) -> float:
        """Return ``sd(tau | y)``.

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5]. Equation [EQ-HIER-002].

        """
        mean = self.tau_mean()
        return math.sqrt(max(self.expectation(lambda t: t * t) - mean * mean, 0.0))

    def tau_quantile(self, probability: float) -> float:
        """Return a quantile of ``tau | y`` from the piecewise-linear CDF in ``u``.

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5]. Equation [EQ-HIER-002].

        """
        p = require_probability(probability, name="probability")
        return self._tau_from_cdf(p)

    def mu_moments(self) -> tuple[float, float]:
        """Return ``(E[mu | y], sd(mu | y))`` by total expectation and variance.

        Equation:
            [EQ-HIER-001], [EQ-HIER-002].

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5].

        """
        mean = self.expectation(lambda t: self.conditional_mu(t)[0])
        second = self.expectation(
            lambda t: self.conditional_mu(t)[1] + self.conditional_mu(t)[0] ** 2
        )
        return mean, math.sqrt(max(second - mean * mean, 0.0))

    def theta_moments(self) -> tuple[Vector, Vector]:
        """Return posterior means and standard deviations of every ``theta_j``.

        Equation:
            [EQ-HIER-001], [EQ-HIER-002].

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5].

        """
        first = [0.0] * self.groups
        second = [0.0] * self.groups
        for w, t in zip(self.weights, self.tau_grid, strict=True):
            if w <= 0.0:
                continue
            means, variances = self.conditional_theta(t)
            for j in range(self.groups):
                first[j] += w * means[j]
                second[j] += w * (variances[j] + means[j] ** 2)
        sds = tuple(
            math.sqrt(max(sq - m * m, 0.0)) for m, sq in zip(first, second, strict=True)
        )
        return tuple(first), sds

    def expected_shrinkage(self) -> tuple[float, ...]:
        """Return ``E[B_j | y]`` for every group.

        Equation:
            [EQ-HIER-004].

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5].

        """
        def factor(error: float) -> Callable[[float], float]:
            """Return ``tau -> s^2 / (s^2 + tau^2)`` for one group.

            References:
                [REF-GELMAN-ETAL-2013, Chapter 5]. Equation [EQ-HIER-004].

            """
            return lambda t: error * error / (error * error + t * t)

        return tuple(self.expectation(factor(s)) for s in self.standard_errors)

    def _tau_from_cdf(self, p: float) -> float:
        """Invert the piecewise-linear CDF in ``u`` and map back to ``tau``.

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5]. No new equation is introduced.

        """
        cdf = self.cdf
        index = bisect.bisect_left(cdf, p)
        if index <= 0:
            return self.tau_grid[0]
        if index >= len(cdf):
            return self.tau_grid[-1]
        lower_u = math.log1p(self.tau_grid[index - 1] / self.scale)
        upper_u = math.log1p(self.tau_grid[index] / self.scale)
        span = cdf[index] - cdf[index - 1]
        fraction = 0.0 if span <= 0.0 else (p - cdf[index - 1]) / span
        u = lower_u + fraction * (upper_u - lower_u)
        return self.scale * math.expm1(u)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw ``(mu, tau, theta_1..theta_J)`` from the exact joint posterior.

        ``tau`` is drawn by inverse CDF from the gridded marginal (linear
        interpolation in ``u``), then ``mu | tau, y`` and
        ``theta_j | mu, tau, y ~ N((1-B_j) y_j + B_j mu, (1-B_j) s_j^2)``.

        Args:
            size: Non-negative number of draws.
            seed: Integer seed or stream.

        Returns:
            Independent joint draws.

        Equation:
            [EQ-HIER-001], [EQ-HIER-002].

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        draws: list[Vector] = []
        for _ in range(count):
            tau = self._tau_from_cdf(generator.random())
            mu_hat, v_mu = self.conditional_mu(tau)
            mu = generator.gauss(mu_hat, math.sqrt(v_mu))
            thetas: list[float] = []
            for y, s in zip(self.estimates, self.standard_errors, strict=True):
                b = s * s / (s * s + tau * tau)
                center = (1.0 - b) * y + b * mu
                thetas.append(generator.gauss(center, math.sqrt((1.0 - b) * s * s)))
            draws.append((mu, tau, *thetas))
        return tuple(draws)


def _conditional_mu(
    y: Sequence[float],
    s: Sequence[float],
    tau: float,
    mu_prior: tuple[float, float] | None,
) -> tuple[float, float]:
    """Return ``(mu_hat, V_mu)`` of ``mu | tau, y``.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5]. Equation [EQ-HIER-001].

    """
    precision = 0.0
    weighted = 0.0
    if mu_prior is not None:
        precision = 1.0 / mu_prior[1] ** 2
        weighted = mu_prior[0] * precision
    for value, error in zip(y, s, strict=True):
        inverse = 1.0 / (error * error + tau * tau)
        precision += inverse
        weighted += value * inverse
    return weighted / precision, 1.0 / precision


def _log_marginal_tau(
    y: Sequence[float],
    s: Sequence[float],
    tau: float,
    mu_prior: tuple[float, float] | None,
    log_tau_prior: LogTauPrior,
) -> float:
    """Return the unnormalized ``log p(tau | y)``.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5]. Equation [EQ-HIER-002].

    """
    mu_hat, v_mu = _conditional_mu(y, s, tau, mu_prior)
    total = float(log_tau_prior(tau)) + 0.5 * math.log(v_mu)
    for value, error in zip(y, s, strict=True):
        variance = error * error + tau * tau
        total -= 0.5 * (math.log(variance) + (value - mu_hat) ** 2 / variance)
    if mu_prior is not None:
        total -= 0.5 * ((mu_hat - mu_prior[0]) / mu_prior[1]) ** 2
    return total


def hierarchical_normal_posterior(
    estimates: Iterable[object],
    standard_errors: Iterable[object],
    *,
    mu_prior: tuple[float, float] | None = None,
    log_tau_prior: LogTauPrior | None = None,
    grid_size: int = 2001,
    tau_max: float | None = None,
) -> HierarchicalNormalPosterior:
    """Compute the exact marginal posterior of ``tau`` on an adaptive grid.

    ``p(tau | y) ∝ p(tau) V_mu^(1/2) prod_j (s_j^2 + tau^2)^(-1/2)
    exp(-(y_j - mu_hat)^2 / (2 (s_j^2 + tau^2)))`` for a flat ``mu`` prior
    (an extra ``N(mu_hat | m_0, s_0^2)`` factor for a normal prior). The upper
    end is doubled until the density times the Jacobian falls below
    ``1e-14`` of its maximum, after checking that the tail decays fast enough
    for the posterior to be proper.

    Args:
        estimates: Group estimates ``y_j`` (at least two).
        standard_errors: Known positive standard errors ``s_j``.
        mu_prior: Optional ``(mean, sd)``; ``None`` is the flat prior.
        log_tau_prior: Unnormalized ``log p(tau)`` on ``tau >= 0``; ``None``
            is the improper uniform prior of BDA3 Chapter 5.
        grid_size: Odd number of grid points (at least 101).
        tau_max: Optional fixed upper grid end (skips the automatic search).

    Returns:
        Gridded posterior with moment, quantile, and sampling methods.

    Raises:
        ValueError: For invalid inputs or an apparently improper posterior.

    Equation:
        [EQ-HIER-001], [EQ-HIER-002].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5], [REF-RUBIN-1981].

    """
    y, s = _checked_data(estimates, standard_errors)
    prior_mu = _checked_mu_prior(mu_prior)
    log_prior = _uniform_log_prior if log_tau_prior is None else log_tau_prior
    if not callable(log_prior):
        raise TypeError("log_tau_prior must be callable")
    size = require_non_negative_integer(grid_size, name="grid_size")
    if size < 101 or size % 2 == 0:
        raise ValueError("grid_size must be odd and at least 101")
    scale = min(s)

    def log_f(tau: float) -> float:
        """Log density in ``u`` (includes the Jacobian ``tau + c``).

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5]. Equation [EQ-HIER-002].

        """
        return _log_marginal_tau(y, s, tau, prior_mu, log_prior) + math.log(
            tau + scale
        )

    if tau_max is None:
        upper = 10.0 * max(s) + (max(y) - min(y))
        for _ in range(_MAX_DOUBLINGS):
            coarse = [
                scale * math.expm1(i / 200.0 * math.log1p(upper / scale))
                for i in range(201)
            ]
            peak = max(log_f(t) for t in coarse)
            if log_f(upper) - peak < math.log(_TAIL_RATIO):
                break
            upper *= 2.0
        else:
            raise ValueError("posterior of tau does not decay; it may be improper")
        slope = (
            _log_marginal_tau(y, s, 2.0 * upper, prior_mu, log_prior)
            - _log_marginal_tau(y, s, upper, prior_mu, log_prior)
        ) / math.log(2.0)
        if not slope < _PROPER_SLOPE:
            raise ValueError(
                "tail of p(tau | y) decays too slowly; the posterior may be "
                "improper (e.g. uniform tau prior with fewer than three groups)"
            )
    else:
        upper = require_finite_real(tau_max, name="tau_max")
        if upper <= 0.0:
            raise ValueError("tau_max must be positive")
    u_max = math.log1p(upper / scale)
    step = u_max / (size - 1)
    grid = tuple(scale * math.expm1(i * step) for i in range(size))
    log_density = tuple(
        _log_marginal_tau(y, s, t, prior_mu, log_prior) for t in grid
    )
    if not all(math.isfinite(v) for v in log_density):
        raise ValueError("log p(tau | y) is not finite on the grid")
    peak = max(log_density)
    in_u = [
        math.exp(v - peak) * (t + scale)
        for v, t in zip(log_density, grid, strict=True)
    ]
    simpson = [
        (1.0 if i in (0, size - 1) else (4.0 if i % 2 else 2.0)) * value
        for i, value in enumerate(in_u)
    ]
    total = math.fsum(simpson)
    weights = tuple(value / total for value in simpson)
    cumulative = [0.0]
    for left, right in zip(in_u[:-1], in_u[1:], strict=True):
        cumulative.append(cumulative[-1] + 0.5 * (left + right))
    cdf = tuple(value / cumulative[-1] for value in cumulative)
    return HierarchicalNormalPosterior(
        y, s, grid, log_density, weights, cdf, scale, prior_mu
    )


# ---------------------------------------------------------------------------
# NUTS on the non-centered parameterization
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HierarchicalNormalSamples:
    """NUTS output for the non-centered hierarchical normal model.

    Args:
        hamiltonian: Raw sampler output on ``(mu, log tau, z_1..z_J)``.
        draws: The same draws mapped to ``(mu, tau, theta_1..theta_J)``.

    References:
        [REF-HOFFMAN-GELMAN-2014], [REF-BETANCOURT-GIROLAMI-2013].

    """

    hamiltonian: HamiltonianResult
    draws: MCMCResult


def non_centered_log_density(
    estimates: Iterable[object],
    standard_errors: Iterable[object],
    *,
    mu_prior: tuple[float, float] | None = None,
    log_tau_prior: LogTauPrior | None = None,
) -> tuple[Callable[[Vector], float], Callable[[Vector], tuple[float, ...]]]:
    """Return the non-centered log-posterior and its gradient.

    State ``x = (mu, eta, z_1..z_J)`` with ``tau = exp(eta)`` and
    ``theta_j = mu + tau z_j``; the log-density includes the Jacobian ``eta``.
    The gradient is analytic except for ``d log p(tau) / d tau`` of a custom
    prior, which uses a central difference.

    Args:
        estimates: Group estimates ``y_j``.
        standard_errors: Known standard errors ``s_j``.
        mu_prior: Optional ``(mean, sd)`` normal prior on ``mu``.
        log_tau_prior: Optional ``log p(tau)``; ``None`` is uniform.

    Returns:
        ``(log_density, gradient)`` callables on ``R^(J+2)``.

    Equation:
        [EQ-HIER-003].

    References:
        [REF-BETANCOURT-GIROLAMI-2013], [REF-GELMAN-ETAL-2013, Chapter 5].

    """
    y, s = _checked_data(estimates, standard_errors)
    prior_mu = _checked_mu_prior(mu_prior)
    custom = log_tau_prior
    precisions = tuple(1.0 / (v * v) for v in s)

    def log_density(x: Vector) -> float:
        """Evaluate the non-centered log-posterior.

        References:
            [REF-BETANCOURT-GIROLAMI-2013]. Equation [EQ-HIER-003].

        """
        mu, eta = x[0], x[1]
        tau = math.exp(eta)
        total = eta
        if custom is not None:
            total += float(custom(tau))
        if prior_mu is not None:
            total -= 0.5 * ((mu - prior_mu[0]) / prior_mu[1]) ** 2
        for z, value, precision in zip(x[2:], y, precisions, strict=True):
            residual = value - mu - tau * z
            total -= 0.5 * (z * z + residual * residual * precision)
        return total

    def gradient(x: Vector) -> tuple[float, ...]:
        """Evaluate the gradient of the non-centered log-posterior.

        References:
            [REF-BETANCOURT-GIROLAMI-2013]. Equation [EQ-HIER-003].

        """
        mu, eta = x[0], x[1]
        tau = math.exp(eta)
        d_mu = 0.0
        d_eta = 1.0
        if custom is not None:
            h = 1e-6 * max(1.0, tau)
            low = max(tau - h, 0.0)
            d_eta += tau * (custom(tau + h) - custom(low)) / (tau + h - low)
        if prior_mu is not None:
            d_mu -= (mu - prior_mu[0]) / prior_mu[1] ** 2
        d_z: list[float] = []
        for z, value, precision in zip(x[2:], y, precisions, strict=True):
            scaled = (value - mu - tau * z) * precision
            d_mu += scaled
            d_eta += scaled * tau * z
            d_z.append(-z + scaled * tau)
        return (d_mu, d_eta, *d_z)

    return log_density, gradient


def sample_hierarchical_normal(
    estimates: Iterable[object],
    standard_errors: Iterable[object],
    *,
    draws: int = 1_000,
    warmup: int = 1_000,
    chains: int = 4,
    mu_prior: tuple[float, float] | None = None,
    log_tau_prior: LogTauPrior | None = None,
    target_acceptance: float = 0.9,
    max_tree_depth: int = 10,
    seed: int | RandomStream | None = None,
) -> HierarchicalNormalSamples:
    """Sample the hierarchical normal posterior by NUTS (non-centered).

    Args:
        estimates: Group estimates ``y_j``.
        standard_errors: Known standard errors ``s_j``.
        draws: Kept draws per chain.
        warmup: Warm-up iterations per chain.
        chains: Number of chains.
        mu_prior: Optional ``(mean, sd)`` normal prior on ``mu``.
        log_tau_prior: Optional ``log p(tau)``; ``None`` is uniform.
        target_acceptance: Dual-averaging target (default ``0.9``).
        max_tree_depth: NUTS tree-depth limit.
        seed: Integer seed or stream.

    Returns:
        Raw sampler output and draws of ``(mu, tau, theta_1..theta_J)``.

    Equation:
        [EQ-HIER-003], [EQ-NUTS-UTURN-001].

    References:
        [REF-HOFFMAN-GELMAN-2014], [REF-BETANCOURT-GIROLAMI-2013],
        [REF-GELMAN-ETAL-2013, Chapter 5].

    """
    y, s = _checked_data(estimates, standard_errors)
    log_density, gradient = non_centered_log_density(
        y, s, mu_prior=mu_prior, log_tau_prior=log_tau_prior
    )
    groups = len(y)
    center = math.fsum(y) / groups
    spread = math.sqrt(math.fsum((v - center) ** 2 for v in y) / (groups - 1))
    start = (center, math.log(max(spread, 1e-3 * min(s))), *([0.0] * groups))
    names = ("mu", "log_tau") + tuple(f"z_{j + 1}" for j in range(groups))
    result = no_u_turn_sampler(
        log_density,
        start,
        gradient=gradient,
        draws=draws,
        warmup=warmup,
        chains=chains,
        target_acceptance=target_acceptance,
        max_tree_depth=max_tree_depth,
        seed=seed,
        names=names,
    )

    def natural(x: Vector) -> tuple[float, ...]:
        """Map ``(mu, eta, z)`` to ``(mu, tau, theta)``.

        References:
            [REF-BETANCOURT-GIROLAMI-2013]. Equation [EQ-HIER-003].

        """
        tau = math.exp(x[1])
        return (x[0], tau, *(x[0] + tau * z for z in x[2:]))

    natural_names = ("mu", "tau") + tuple(f"theta_{j + 1}" for j in range(groups))
    return HierarchicalNormalSamples(
        result, result.mcmc.transformed(natural, natural_names)
    )


# ---------------------------------------------------------------------------
# Beta-binomial empirical Bayes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BetaBinomialEmpiricalBayes:
    """Type-II maximum-likelihood fit of a beta-binomial hierarchy.

    Args:
        alpha: Estimated ``alpha``.
        beta: Estimated ``beta``.
        log_marginal_likelihood: Beta-binomial log-likelihood at the estimate.
        posterior_means: ``(y_j + alpha) / (n_j + alpha + beta)``.
        shrinkage: ``(alpha + beta) / (n_j + alpha + beta)`` (weight on the
            prior mean ``alpha / (alpha + beta)``).
        optimization: Optimizer record on ``(logit m, log kappa)``.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5], [REF-MORRIS-1983].

    """

    alpha: float
    beta: float
    log_marginal_likelihood: float
    posterior_means: Vector
    shrinkage: Vector
    optimization: OptimizationResult

    @property
    def prior_mean(self) -> float:
        """Return ``alpha / (alpha + beta)``.

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5]. No equation is evaluated.

        """
        return self.alpha / (self.alpha + self.beta)


def beta_binomial_log_likelihood(
    successes: Sequence[int], trials: Sequence[int], alpha: float, beta: float
) -> float:
    """Return the beta-binomial marginal log-likelihood.

    ``sum_j [log C(n_j, y_j) + log B(y_j + alpha, n_j - y_j + beta)
    - log B(alpha, beta)]``.

    Args:
        successes: Counts ``0 <= y_j <= n_j``.
        trials: Positive trial counts ``n_j``.
        alpha: Positive ``alpha``.
        beta: Positive ``beta``.

    Returns:
        Log marginal likelihood.

    Equation:
        [EQ-HIER-BB-001].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5].

    """
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    log_beta_prior = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    total: list[float] = []
    for y, n in zip(successes, trials, strict=True):
        log_choose = math.lgamma(n + 1) - math.lgamma(y + 1) - math.lgamma(n - y + 1)
        log_beta_post = (
            math.lgamma(y + alpha)
            + math.lgamma(n - y + beta)
            - math.lgamma(n + alpha + beta)
        )
        total.append(log_choose + log_beta_post - log_beta_prior)
    return math.fsum(total)


def fit_beta_binomial(
    successes: Iterable[object], trials: Iterable[object]
) -> BetaBinomialEmpiricalBayes:
    """Fit ``Beta(alpha, beta)`` by maximizing the beta-binomial likelihood.

    The optimizer works on ``(logit m, log kappa)`` with ``m = alpha/(alpha +
    beta)`` and ``kappa = alpha + beta``. When the counts are not
    over-dispersed relative to a common binomial, the maximum lies at
    ``kappa -> inf`` (complete pooling) and ``optimization.converged`` may be
    false with a very large ``kappa``.

    Args:
        successes: Integer counts ``0 <= y_j <= n_j``.
        trials: Positive integer trial counts.

    Returns:
        Estimates, log marginal likelihood, shrunken means, shrinkage weights.

    Raises:
        ValueError: For invalid counts or fewer than two groups.

    Equation:
        [EQ-HIER-BB-001].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 5], [REF-MORRIS-1983],
        [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    y = tuple(require_non_negative_integer(v, name="successes") for v in successes)
    n = tuple(require_non_negative_integer(v, name="trials") for v in trials)
    if len(y) != len(n) or len(y) < 2:
        raise ValueError("successes and trials must align with at least two groups")
    if any(count == 0 or k > count for k, count in zip(y, n, strict=True)):
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    pooled = (sum(y) + 0.5) / (sum(n) + 1.0)

    def negative(x: Vector) -> float:
        """Return the negative log marginal likelihood on the unconstrained scale.

        References:
            [REF-GELMAN-ETAL-2013, Chapter 5]. Equation [EQ-HIER-BB-001].

        """
        if abs(x[0]) > 700.0 or abs(x[1]) > 700.0:
            return math.inf
        m = 1.0 / (1.0 + math.exp(-x[0]))
        kappa = math.exp(x[1])
        alpha, beta = m * kappa, (1.0 - m) * kappa
        if alpha <= 0.0 or beta <= 0.0:
            return math.inf
        return -beta_binomial_log_likelihood(y, n, alpha, beta)

    start = (math.log(pooled / (1.0 - pooled)), math.log(10.0))
    result = minimize_bfgs(negative, start, max_iterations=500)
    m = 1.0 / (1.0 + math.exp(-result.x[0]))
    kappa = math.exp(result.x[1])
    alpha, beta = m * kappa, (1.0 - m) * kappa
    means = tuple((k + alpha) / (count + kappa) for k, count in zip(y, n, strict=True))
    shrink = tuple(kappa / (count + kappa) for count in n)
    return BetaBinomialEmpiricalBayes(
        alpha, beta, -result.value, means, shrink, result
    )
