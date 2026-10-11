"""Generic expectation-maximization driver and EM for censored normal data.

The EM algorithm of Dempster, Laird, and Rubin (1977) maximizes an
observed-data log-likelihood ``l(theta)`` by alternating

* **E-step** -- compute the expected complete-data sufficient statistics
  ``S_t = E[s(Z) | y, theta_t]`` (equivalently the function
  ``Q(theta | theta_t)``), and
* **M-step** -- set ``theta_{t+1} = argmax_theta Q(theta | theta_t)``.

Because ``Q(theta | theta_t) - l(theta)`` is maximized at ``theta = theta_t``
(Jensen's inequality), every exact EM step satisfies
``l(theta_{t+1}) >= l(theta_t)`` ([EQ-EM-001]). :func:`expectation_maximization`
checks that inequality at every iteration and reports violations, which in
practice reveal a coding error in a user E- or M-step. Convergence to a
*global* maximum is not guaranteed; limit points are stationary points under
the regularity conditions of Wu (1983).

Censored normal data
    :func:`fit_censored_normal` handles a univariate normal sample in which
    each observation is either exact or known only to lie in an interval
    (left-, right-, or interval-censored, e.g. detection limits or a type-I
    censored life test). The E-step uses the closed-form truncated-normal
    moments already implemented by
    :class:`~probstat.distributions.TruncatedNormal`, and the observed-data
    log-likelihood is available exactly, so the monotonicity check is a
    genuine test of the implementation.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from probstat.distributions import TruncatedNormal
from probstat.distributions._continuous import log_normal_interval_mass
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

ParameterT = TypeVar("ParameterT")
StatisticsT = TypeVar("StatisticsT")
DecreasePolicy = Literal["warn", "raise", "ignore"]

_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


class EMMonotonicityWarning(RuntimeWarning):
    """Raised (as a warning) when an EM step decreases the log-likelihood.

    An exact EM step cannot decrease the observed-data log-likelihood, so a
    decrease beyond round-off indicates an incorrect E- or M-step, an M-step
    that does not maximize ``Q``, or an objective that is not the one the
    steps optimize.

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-WU-1983].

    """


@dataclass(frozen=True, slots=True)
class EMResult(Generic[ParameterT]):
    """Outcome of an EM run.

    Args:
        parameters: Final parameter value ``theta_T``.
        log_likelihood: Objective ``l(theta_T)`` reported by the E-step.
        trace: Objective after every E-step, ``trace[0] = l(theta_0)``.
        iterations: Completed M-steps.
        converged: Whether the relative-change tolerance was met.
        monotone: Whether no decrease beyond the slack was observed.
        max_decrease: Largest observed decrease ``l(theta_t) - l(theta_{t+1})``
            (``0`` if the trace never decreased).

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-MCLACHLAN-KRISHNAN-2008].

    """

    parameters: ParameterT
    log_likelihood: float
    trace: tuple[float, ...]
    iterations: int
    converged: bool
    monotone: bool
    max_decrease: float


def expectation_maximization(
    initial: ParameterT,
    e_step: Callable[[ParameterT], tuple[StatisticsT, float]],
    m_step: Callable[[StatisticsT], ParameterT],
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 1_000,
    decrease_slack: float = 1e-9,
    on_decrease: DecreasePolicy = "warn",
) -> EMResult[ParameterT]:
    """Run the EM iteration until the objective stops increasing.

    The E-step receives ``theta_t`` and returns the expected complete-data
    statistics together with the observed-data objective ``l(theta_t)``
    (a log-likelihood or a penalized log-likelihood / log-posterior). The
    M-step maps the statistics to ``theta_{t+1}``. Iteration stops when
    ``l(theta_{t+1}) - l(theta_t) <= tolerance * (1 + |l(theta_{t+1})|)``.

    Args:
        initial: Starting parameter ``theta_0``.
        e_step: ``theta -> (statistics, objective)``.
        m_step: ``statistics -> theta``.
        tolerance: Relative objective-change tolerance (non-negative).
        max_iterations: Maximum number of M-steps (positive).
        decrease_slack: Relative round-off allowance; a step is a violation
            when ``l_new < l_old - slack * (1 + |l_old|)``.
        on_decrease: ``"warn"`` (default) emits :class:`EMMonotonicityWarning`,
            ``"raise"`` raises ``ArithmeticError``, ``"ignore"`` only records.

    Returns:
        Final parameters, objective trace, and monotonicity evidence.

    Raises:
        ValueError: For invalid settings or a non-finite objective.
        ArithmeticError: On a decrease when ``on_decrease="raise"``.

    Equation:
        [EQ-EM-001].

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-WU-1983],
        [REF-MCLACHLAN-KRISHNAN-2008].

    """
    if not callable(e_step) or not callable(m_step):
        raise TypeError("e_step and m_step must be callable")
    checked_tolerance = require_finite_real(tolerance, name="tolerance")
    if checked_tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    slack = require_finite_real(decrease_slack, name="decrease_slack")
    if slack < 0.0:
        raise ValueError("decrease_slack must be non-negative")
    limit = require_non_negative_integer(max_iterations, name="max_iterations")
    if limit == 0:
        raise ValueError("max_iterations must be positive")
    if on_decrease not in ("warn", "raise", "ignore"):
        raise ValueError("on_decrease must be 'warn', 'raise', or 'ignore'")

    parameters = initial
    statistics, objective = e_step(parameters)
    objective = _finite_objective(objective)
    trace = [objective]
    monotone = True
    max_decrease = 0.0
    converged = False
    iterations = 0
    for _ in range(limit):
        candidate = m_step(statistics)
        new_statistics, new_objective = e_step(candidate)
        new_objective = _finite_objective(new_objective)
        iterations += 1
        trace.append(new_objective)
        change = new_objective - objective
        if change < 0.0:
            max_decrease = max(max_decrease, -change)
            if -change > slack * (1.0 + abs(objective)):
                monotone = False
                message = (
                    f"EM objective decreased by {-change:.3e} at iteration {iterations}"
                )
                if on_decrease == "raise":
                    raise ArithmeticError(message)
                if on_decrease == "warn":
                    warnings.warn(message, EMMonotonicityWarning, stacklevel=2)
        parameters, statistics, objective = candidate, new_statistics, new_objective
        if abs(change) <= checked_tolerance * (1.0 + abs(new_objective)):
            converged = True
            break
    return EMResult(
        parameters,
        objective,
        tuple(trace),
        iterations,
        converged,
        monotone,
        max_decrease,
    )


def _finite_objective(value: float) -> float:
    """Validate the E-step objective.

    References:
        Project validation policy. No equation is evaluated.

    """
    objective = float(value)
    if not math.isfinite(objective):
        raise ValueError("E-step objective must be finite")
    return objective


# ---------------------------------------------------------------------------
# Censored normal data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CensoredNormalFit:
    """Maximum-likelihood normal fit to exact and interval-censored data.

    Args:
        mean: Estimated ``mu``.
        standard_deviation: Estimated ``sigma``.
        log_likelihood: Observed-data log-likelihood at the estimate.
        exact_count: Number of exactly observed values.
        censored_count: Number of censored values.
        em: Underlying EM trace (parameters are ``(mu, sigma)``).

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-MCLACHLAN-KRISHNAN-2008].

    """

    mean: float
    standard_deviation: float
    log_likelihood: float
    exact_count: int
    censored_count: int
    em: EMResult[tuple[float, float]]


def censored_normal_log_likelihood(
    intervals: Sequence[tuple[float, float]], mean: float, standard_deviation: float
) -> float:
    """Return the observed-data log-likelihood of interval-censored normal data.

    Args:
        intervals: ``(lower, upper)`` pairs; ``lower == upper`` is an exact
            value, ``-inf``/``inf`` mark left/right censoring.
        mean: Normal location ``mu``.
        standard_deviation: Positive normal scale ``sigma``.

    Returns:
        ``sum_exact log phi((y-mu)/sigma)/sigma
        + sum_cens log(Phi(b) - Phi(a))`` with standardized endpoints.

    Raises:
        ValueError: For an invalid interval or scale.

    Equation:
        [EQ-EM-CENS-001].

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977],
        [REF-NIST-DLMF, Section 7.2].

    """
    checked = _checked_intervals(intervals)
    mu = require_finite_real(mean, name="mean")
    sigma = require_finite_real(standard_deviation, name="standard_deviation")
    if sigma <= 0.0:
        raise ValueError("standard_deviation must be positive")
    return _censored_log_likelihood(checked, mu, sigma)


def _censored_log_likelihood(
    intervals: Sequence[tuple[float, float]], mu: float, sigma: float
) -> float:
    """Evaluate [EQ-EM-CENS-001] for validated inputs.

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977]. Equation [EQ-EM-CENS-001].

    """
    terms: list[float] = []
    log_sigma = math.log(sigma)
    for lower, upper in intervals:
        if lower == upper:
            z = (lower - mu) / sigma
            terms.append(-0.5 * z * z - _LOG_SQRT_TWO_PI - log_sigma)
        else:
            terms.append(
                log_normal_interval_mass((lower - mu) / sigma, (upper - mu) / sigma)
            )
    return math.fsum(terms)


def _checked_intervals(
    intervals: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Validate censoring intervals.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked: list[tuple[float, float]] = []
    for item in intervals:
        if len(item) != 2:
            raise ValueError("each interval must be a (lower, upper) pair")
        lower, upper = float(item[0]), float(item[1])
        if math.isnan(lower) or math.isnan(upper):
            raise ValueError("interval endpoints must not be NaN")
        if lower > upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")
        if lower == upper and not math.isfinite(lower):
            raise ValueError("exact observations must be finite")
        if lower == -math.inf and upper == math.inf:
            raise ValueError("an interval (-inf, inf) carries no information")
        checked.append((lower, upper))
    if not checked:
        raise ValueError("at least one observation is required")
    return tuple(checked)


def fit_censored_normal(
    intervals: Iterable[tuple[float, float]],
    *,
    initial: tuple[float, float] | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 5_000,
) -> CensoredNormalFit:
    """Fit ``Normal(mu, sigma)`` to exact and censored data by EM.

    E-step: for a censored value in ``[a, b]`` the conditional law is the
    normal truncated to ``[a, b]``, so ``E[Y]`` and ``E[Y^2]`` are the
    truncated-normal mean and second moment. M-step: ``mu`` and ``sigma^2``
    are the mean and variance of the completed sufficient statistics
    ([EQ-EM-CENS-002]).

    Args:
        intervals: ``(lower, upper)`` pairs; equal endpoints are exact values,
            ``-inf`` / ``inf`` give left / right censoring.
        initial: Optional ``(mu, sigma)`` start; by default the mean and
            standard deviation of the exact values, or of the finite
            endpoints when fewer than two values are exact.
        tolerance: Relative log-likelihood change tolerance.
        max_iterations: Maximum EM iterations.

    Returns:
        Estimate, log-likelihood, counts, and the EM trace.

    Raises:
        ValueError: For invalid intervals or when the data cannot identify a
            positive scale.

    Equation:
        [EQ-EM-001], [EQ-EM-CENS-001], [EQ-EM-CENS-002], [EQ-DIST-TN-003].

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977],
        [REF-MCLACHLAN-KRISHNAN-2008], [REF-JKB-1994, Chapter 13].

    """
    data = _checked_intervals(intervals)
    exact = [lower for lower, upper in data if lower == upper]
    if initial is None:
        pool = (
            exact
            if len(exact) >= 2
            else [value for pair in data for value in pair if math.isfinite(value)]
        )
        if len(pool) < 2:
            raise ValueError("at least two finite values are needed to start EM")
        start_mu = math.fsum(pool) / len(pool)
        spread = math.fsum((v - start_mu) ** 2 for v in pool) / (len(pool) - 1)
        start_sigma = math.sqrt(spread) if spread > 0.0 else 1.0
    else:
        start_mu = require_finite_real(initial[0], name="initial mean")
        start_sigma = require_finite_real(initial[1], name="initial sigma")
        if start_sigma <= 0.0:
            raise ValueError("initial sigma must be positive")
    count = len(data)

    def e_step(
        theta: tuple[float, float],
    ) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], float]:
        """Return per-observation conditional means and variances, and log L.

        Keeping the moments per observation lets the M-step centre them on
        the new mean; summing raw second moments and subtracting ``mu^2``
        cancels catastrophically when the location dwarfs the scale.

        References:
            [REF-DEMPSTER-LAIRD-RUBIN-1977]. Equation [EQ-EM-CENS-002].

        """
        mu, sigma = theta
        first: list[float] = []
        second: list[float] = []
        for lower, upper in data:
            if lower == upper:
                first.append(lower)
                second.append(0.0)
                continue
            law = TruncatedNormal(
                mu,
                sigma,
                None if lower == -math.inf else lower,
                None if upper == math.inf else upper,
            )
            first.append(law.mean)
            second.append(law.variance)
        return (
            (tuple(first), tuple(second)),
            _censored_log_likelihood(data, mu, sigma),
        )

    def m_step(
        statistics: tuple[tuple[float, ...], tuple[float, ...]],
    ) -> tuple[float, float]:
        """Return the complete-data normal MLE from completed moments.

        ``mu = mean(E[Y_i])`` and
        ``sigma^2 = mean(Var[Y_i] + (E[Y_i] - mu)^2)``, the centred form of
        ``E[Y^2] - mu^2``.

        References:
            [REF-DEMPSTER-LAIRD-RUBIN-1977]. Equation [EQ-EM-CENS-002].

        """
        means, variances = statistics
        mu = math.fsum(means) / count
        variance = (
            math.fsum(v + (m - mu) ** 2 for m, v in zip(means, variances, strict=True))
            / count
        )
        if not variance > 0.0:
            raise ValueError("censored data do not identify a positive scale")
        return mu, math.sqrt(variance)

    result = expectation_maximization(
        (start_mu, start_sigma),
        e_step,
        m_step,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    mu, sigma = result.parameters
    return CensoredNormalFit(
        mu,
        sigma,
        result.log_likelihood,
        len(exact),
        count - len(exact),
        result,
    )
