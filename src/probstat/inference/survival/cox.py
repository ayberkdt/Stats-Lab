"""Cox proportional-hazards regression by Newton-Raphson on the partial likelihood.

The hazard of subject ``i`` is ``h_0(t) exp(x_i' beta)`` with an unspecified
baseline ``h_0``. ``beta`` is estimated from the partial likelihood, which
compares each event with its risk set ``R_j = {i : t_i >= t_j}``. Tied event
times are handled by Breslow's approximation (every tied event sees the full
risk set) or Efron's (the tied events are removed from the risk set in equal
fractions). Without ties both coincide with Cox's partial likelihood.

Risk-set sums are accumulated from the largest time downward, so one
likelihood evaluation costs ``O(n p^2)`` after sorting.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, inverse_spd, solve_spd

from ..sampling_distributions import chi_square_sf, standard_normal_sf
from .data import SurvivalData, covariate_rows

TieMethod = Literal["breslow", "efron"]
_TIES = ("breslow", "efron")


@dataclass(frozen=True, slots=True)
class PartialLikelihood:
    """Value, score, and observed information of a Cox partial likelihood.

    Args:
        value: ``log PL(beta)``.
        score: Gradient ``U(beta)``.
        information: ``I(beta) = -d^2 log PL / d beta d beta'``.

    References:
        [EQ-SURV-COX-001], [EQ-SURV-COX-002], [REF-COX-1972].

    """

    value: float
    score: Vector
    information: Matrix


def _groups(
    data: SurvivalData,
) -> list[tuple[float, list[int], list[int]]]:
    """Return ``(time, all subjects at time, event subjects at time)`` descending.

    References:
        [REF-COX-1972]. No equation is evaluated.

    """
    order = sorted(range(data.size), key=lambda i: data.times[i], reverse=True)
    groups: list[tuple[float, list[int], list[int]]] = []
    for index in order:
        time = data.times[index]
        if not groups or groups[-1][0] != time:
            groups.append((time, [], []))
        groups[-1][1].append(index)
        if data.events[index]:
            groups[-1][2].append(index)
    return groups


def _check_ties(ties: object) -> TieMethod:
    """Validate the tie-handling label.

    References:
        [REF-BRESLOW-1974], [REF-EFRON-1977]. No equation is evaluated.

    """
    if ties == "breslow":
        return "breslow"
    if ties == "efron":
        return "efron"
    raise ValueError(f"ties must be one of {_TIES}")


def _partial_likelihood(
    data: SurvivalData,
    rows: tuple[tuple[float, ...], ...],
    beta: Vector,
    ties: TieMethod,
) -> PartialLikelihood:
    """Evaluate the partial likelihood, score, and information.

    References:
        [EQ-SURV-COX-001], [EQ-SURV-COX-002], [REF-COX-1972],
        [REF-BRESLOW-1974], [REF-EFRON-1977].

    """
    p = len(beta)
    etas = [math.fsum(b * x for b, x in zip(beta, row, strict=True)) for row in rows]
    shift = max(etas)
    weights = [math.exp(eta - shift) for eta in etas]
    s0 = 0.0
    s1 = [0.0] * p
    s2 = [[0.0] * p for _ in range(p)]
    value_terms: list[float] = []
    score = [0.0] * p
    information = [[0.0] * p for _ in range(p)]
    for _, members, deaths in _groups(data):
        for i in members:
            w, x = weights[i], rows[i]
            s0 += w
            for a in range(p):
                s1[a] += w * x[a]
                for b in range(p):
                    s2[a][b] += w * x[a] * x[b]
        d = len(deaths)
        if d == 0:
            continue
        d0 = math.fsum(weights[i] for i in deaths)
        d1 = [math.fsum(weights[i] * rows[i][a] for i in deaths) for a in range(p)]
        d2 = [
            [
                math.fsum(weights[i] * rows[i][a] * rows[i][b] for i in deaths)
                for b in range(p)
            ]
            for a in range(p)
        ]
        for i in deaths:
            value_terms.append(etas[i])
            for a in range(p):
                score[a] += rows[i][a]
        for r in range(d):
            fraction = r / d if ties == "efron" else 0.0
            a0 = s0 - fraction * d0
            a1 = [s1[a] - fraction * d1[a] for a in range(p)]
            value_terms.append(-(math.log(a0) + shift))
            for a in range(p):
                mean_a = a1[a] / a0
                score[a] -= mean_a
                for b in range(p):
                    a2 = s2[a][b] - fraction * d2[a][b]
                    information[a][b] += a2 / a0 - mean_a * a1[b] / a0
    return PartialLikelihood(
        math.fsum(value_terms),
        tuple(score),
        tuple(tuple(row) for row in information),
    )


def cox_partial_likelihood(
    data: SurvivalData,
    covariates: Sequence[Sequence[float]],
    beta: Sequence[float],
    *,
    ties: TieMethod = "efron",
) -> PartialLikelihood:
    """Evaluate the Cox partial log-likelihood with its derivatives.

    Breslow: ``l = sum_j [sum_{i in D_j} x_i' beta - d_j log S0(R_j)]``.
    Efron: ``l = sum_j [sum_{i in D_j} x_i' beta - sum_{r=0}^{d_j-1}
    log(S0(R_j) - (r/d_j) S0(D_j))]`` with ``S0(A) = sum_{i in A}
    exp(x_i' beta)``.

    Args:
        data: Right-censored sample.
        covariates: Subject-by-covariate rows (no intercept).
        beta: Coefficient vector.
        ties: ``"breslow"`` or ``"efron"``.

    Returns:
        Value, score vector, and observed information matrix.

    Raises:
        ValueError: If shapes disagree, ``ties`` is unknown, or there are no
            covariates.

    Equation:
        [EQ-SURV-COX-001], [EQ-SURV-COX-002].

    References:
        [REF-COX-1972], [REF-BRESLOW-1974], [REF-EFRON-1977],
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 8].

    """
    method = _check_ties(ties)
    rows = covariate_rows(covariates, data.size, name="covariates")
    coefficients = tuple(require_finite_real(b, name="beta") for b in beta)
    if not coefficients or len(coefficients) != len(rows[0]):
        raise ValueError("beta must have one entry per covariate (at least one)")
    return _partial_likelihood(data, rows, coefficients, method)


@dataclass(frozen=True, slots=True)
class CoxModelResult:
    """Fitted Cox proportional-hazards model with Wald inference.

    Args:
        covariate_names: Covariate labels.
        coefficients: Maximum partial-likelihood estimates ``beta``.
        standard_errors: ``sqrt(diag I(beta)^-1)``.
        covariance: Inverse observed information.
        z_statistics: Wald statistics ``beta / se``.
        p_values: Two-sided normal p-values of the Wald statistics.
        hazard_ratios: ``exp(beta)``.
        hazard_ratio_lower: ``exp(beta - z se)``.
        hazard_ratio_upper: ``exp(beta + z se)``.
        confidence: Nominal level of the Wald intervals.
        log_partial_likelihood: ``l(beta_hat)``.
        null_log_partial_likelihood: ``l(0)``.
        likelihood_ratio: ``2 [l(beta_hat) - l(0)]``.
        likelihood_ratio_p_value: Chi-square (``p`` df) upper tail.
        score_statistic: ``U(0)' I(0)^-1 U(0)``.
        score_p_value: Chi-square (``p`` df) upper tail.
        ties: Tie-handling method.
        size: Number of subjects.
        events: Number of events.
        iterations: Newton iterations used.
        converged: Whether the convergence criterion was met.
        baseline_times: Distinct event times.
        baseline_cumulative_hazard: Breslow estimate of the cumulative
            hazard at the event times for a subject whose linear predictor
            equals ``baseline_reference``.
        baseline_reference: Linear predictor ``c`` at which the baseline is
            stored, so ``H(t | x) = H_c(t) exp(x' beta - c)``. It is ``0``
            (the usual ``x = 0`` baseline) unless the fitted linear
            predictors are so large that ``exp(x' beta)`` would overflow or
            underflow, as with uncentred covariates.

    References:
        [EQ-SURV-COX-001], [EQ-SURV-COX-002], [EQ-SURV-COX-003],
        [EQ-SURV-COX-004], [REF-COX-1972],
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 8].

    """

    covariate_names: tuple[str, ...]
    coefficients: Vector
    standard_errors: Vector
    covariance: Matrix
    z_statistics: Vector
    p_values: Vector
    hazard_ratios: Vector
    hazard_ratio_lower: Vector
    hazard_ratio_upper: Vector
    confidence: float
    log_partial_likelihood: float
    null_log_partial_likelihood: float
    likelihood_ratio: float
    likelihood_ratio_p_value: float
    score_statistic: float
    score_p_value: float
    ties: str
    size: int
    events: int
    iterations: int
    converged: bool
    baseline_times: Vector
    baseline_cumulative_hazard: Vector
    baseline_reference: float = 0.0

    def survival(self, time: float, covariates: Sequence[float]) -> float:
        """Return ``S(t | x) = exp(-H_0(t) exp(x' beta))``.

        Args:
            time: Finite time.
            covariates: Covariate row in fit order.

        Returns:
            Predicted survival probability (``1`` before the first event).

        Raises:
            ValueError: If the covariate count is wrong.

        Equation:
            [EQ-SURV-COX-004].

        References:
            [REF-BRESLOW-1974], [REF-KLEIN-MOESCHBERGER-2003, Chapter 8].

        """
        point = require_finite_real(time, name="time")
        row = tuple(require_finite_real(v, name="covariates") for v in covariates)
        if len(row) != len(self.coefficients):
            raise ValueError("wrong number of covariate values")
        index = bisect.bisect_right(self.baseline_times, point)
        if index == 0:
            return 1.0
        eta = math.fsum(b * x for b, x in zip(self.coefficients, row, strict=True))
        exponent = eta - self.baseline_reference
        if exponent > 700.0:
            return 0.0
        return math.exp(
            -self.baseline_cumulative_hazard[index - 1] * math.exp(exponent)
        )


def _solve_step(information: Matrix, score: Vector) -> Vector:
    """Return the Newton step ``I^-1 U``.

    Raises:
        ValueError: If the information matrix is not positive definite.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 3.3]. No new equation is evaluated.

    """
    try:
        return solve_spd(information, score)
    except ValueError as error:
        raise ValueError(
            "partial-likelihood information is not positive definite; "
            "a covariate may be constant or collinear"
        ) from error


def fit_cox_model(
    data: SurvivalData,
    covariates: Sequence[Sequence[float]],
    *,
    ties: TieMethod = "efron",
    covariate_names: Sequence[str] | None = None,
    confidence: float = 0.95,
    max_iterations: int = 50,
    tolerance: float = 1e-10,
) -> CoxModelResult:
    """Fit a Cox proportional-hazards model by Newton-Raphson.

    Starting from ``beta = 0``, each iteration takes the Newton step
    ``I(beta)^-1 U(beta)``, halving it until the partial likelihood does not
    decrease. Iteration stops when the likelihood change is at most
    ``tolerance * max(1, |l|)``. Monotone likelihood (complete separation)
    shows up as ``converged=False`` with a very large coefficient.

    Args:
        data: Right-censored sample with at least one event.
        covariates: Subject-by-covariate rows (no intercept).
        ties: ``"efron"`` (default) or ``"breslow"``.
        covariate_names: Optional labels; default ``x1, x2, ...``.
        confidence: Level of the Wald hazard-ratio intervals.
        max_iterations: Maximum Newton iterations.
        tolerance: Relative likelihood-change tolerance.

    Returns:
        Coefficients, Wald inference, global tests, and the Breslow baseline.

    Raises:
        ValueError: If there is no event or covariate, shapes disagree, the
            information is singular, or a setting is out of range.

    Equation:
        [EQ-SURV-COX-001], [EQ-SURV-COX-002], [EQ-SURV-COX-003],
        [EQ-SURV-COX-004].

    References:
        [REF-COX-1972], [REF-BRESLOW-1974], [REF-EFRON-1977],
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 8].

    """
    method = _check_ties(ties)
    rows = covariate_rows(covariates, data.size, name="covariates")
    p = len(rows[0])
    if p == 0:
        raise ValueError("at least one covariate is required")
    if data.event_count == 0:
        raise ValueError("at least one observed event is required")
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not tolerance > 0.0:
        raise ValueError("tolerance must be positive")
    names = (
        tuple(f"x{k + 1}" for k in range(p))
        if covariate_names is None
        else tuple(covariate_names)
    )
    if len(names) != p:
        raise ValueError("covariate_names must match the covariate count")

    beta: Vector = (0.0,) * p
    null = _partial_likelihood(data, rows, beta, method)
    current = null
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):  # noqa: B007
        step = _solve_step(current.information, current.score)
        scale = 1.0
        while True:
            trial = tuple(b + scale * s for b, s in zip(beta, step, strict=True))
            try:
                candidate = _partial_likelihood(data, rows, trial, method)
                accepted = candidate.value >= current.value - 1e-12 * max(
                    1.0, abs(current.value)
                )
            except (OverflowError, ValueError):
                accepted = False
            if accepted:
                break
            scale *= 0.5
            if scale < 1e-10:
                raise ValueError("Newton step halving failed to improve l(beta)")
        change = candidate.value - current.value
        beta, current = trial, candidate
        if abs(change) <= tolerance * max(1.0, abs(current.value)):
            converged = True
            break
    covariance = inverse_spd(current.information)
    errors = tuple(math.sqrt(max(covariance[k][k], 0.0)) for k in range(p))
    z = NormalDist().inv_cdf(0.5 + 0.5 * level)
    z_stats = tuple(
        b / e if e > 0.0 else math.nan for b, e in zip(beta, errors, strict=True)
    )
    p_values = tuple(
        2.0 * standard_normal_sf(abs(value)) if math.isfinite(value) else math.nan
        for value in z_stats
    )
    null_step = _solve_step(null.information, null.score)
    score_statistic = math.fsum(
        u * s for u, s in zip(null.score, null_step, strict=True)
    )
    likelihood_ratio = max(2.0 * (current.value - null.value), 0.0)
    times, hazard, reference = _breslow_baseline(data, rows, beta)
    return CoxModelResult(
        names,
        beta,
        errors,
        covariance,
        z_stats,
        p_values,
        tuple(math.exp(b) for b in beta),
        tuple(math.exp(b - z * e) for b, e in zip(beta, errors, strict=True)),
        tuple(math.exp(b + z * e) for b, e in zip(beta, errors, strict=True)),
        level,
        current.value,
        null.value,
        likelihood_ratio,
        chi_square_sf(likelihood_ratio, p),
        score_statistic,
        chi_square_sf(max(score_statistic, 0.0), p),
        method,
        data.size,
        data.event_count,
        iterations,
        converged,
        times,
        hazard,
        reference,
    )


def _breslow_baseline(
    data: SurvivalData, rows: tuple[tuple[float, ...], ...], beta: Vector
) -> tuple[Vector, Vector, float]:
    """Return the Breslow baseline cumulative hazard at the event times.

    ``H_c(t) = sum_{t_j <= t} d_j / sum_{i in R_j} exp(x_i' beta - c)``, the
    cumulative hazard of a subject with linear predictor ``c``. ``c = 0``
    gives the usual baseline; when some ``|x_i' beta|`` exceeds 500 the
    largest linear predictor is used instead so that the weights stay
    finite and the risk-set sums stay positive.

    References:
        [EQ-SURV-COX-004], [REF-BRESLOW-1974].

    """
    etas = [math.fsum(b * x for b, x in zip(beta, row, strict=True)) for row in rows]
    reference = max(etas) if max(abs(e) for e in etas) > 500.0 else 0.0
    risk = [math.exp(e - reference) for e in etas]
    increments: list[tuple[float, float]] = []
    total = 0.0
    for time, members, deaths in _groups(data):
        total += math.fsum(risk[i] for i in members)
        if deaths:
            increments.append((time, len(deaths) / total))
    increments.reverse()
    cumulative: list[float] = []
    running = 0.0
    for _, increment in increments:
        running += increment
        cumulative.append(running)
    return tuple(time for time, _ in increments), tuple(cumulative), reference
