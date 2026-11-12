"""Zero-inflated and hurdle count regressions, and the Vuong statistic.

A zero-inflated model mixes a point mass at zero (probability ``pi_i``,
logit link) with a Poisson or NB2 count law; zeros arise from both parts
[EQ-CNT-ZI-001]. A hurdle model separates the zeros completely: a binary
logit for ``P(y_i > 0)`` and a zero-truncated count law for the positive
counts [EQ-CNT-HURDLE-001]. Both are fitted by direct maximum likelihood
with analytic gradients; the hurdle likelihood factorizes, so its two parts
are fitted separately.

The Vuong statistic [EQ-CNT-VUONG-001] compares two non-nested models by
the per-observation log-likelihood ratio. Its normal reference requires
models that are strictly non-nested; a Poisson (or NB2) model is the
zero-inflated model with ``pi = 0``, a point on the boundary of the
inflation parameter space reached as the logit intercept tends to
``-infinity``, so the statistic is not a valid test of zero inflation
[REF-WILSON-2015].
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.linalg import Matrix, Vector, as_matrix

from ._ml_tools import (
    inverse_information,
    log1pexp,
    logistic,
    newton_maximize,
    normal_intervals,
    wald_table,
)
from .count_regression import (
    _digamma_difference,
    _linear,
    _offsets,
    _prepare,
    fit_negative_binomial_regression,
    fit_poisson_regression,
    nb2_log_pmf,
)
from .fitting import information_criteria
from .glm import fit_glm
from .linear_models import DesignMatrix, _coerce_design
from .sampling_distributions import standard_normal_sf

CountFamily = Literal["poisson", "negative-binomial"]
Correction = Literal["none", "aic", "bic"]


@dataclass(frozen=True, slots=True)
class ZeroCountResult:
    """Maximum-likelihood zero-inflated or hurdle count regression.

    Attributes:
        model: ``"zero-inflated"`` or ``"hurdle"``.
        family: ``"poisson"`` or ``"negative-binomial"``.
        count_coefficients: ``beta`` of the count part (log mean of the
            untruncated count law).
        zero_coefficients: ``gamma`` of the logit part: the log odds of a
            structural zero (zero-inflated) or of a positive count (hurdle).
        count_standard_errors: Standard errors of ``beta``.
        zero_standard_errors: Standard errors of ``gamma``.
        count_column_names: Count-design names.
        zero_column_names: Zero-design names.
        theta: NB2 size (``inf`` for Poisson).
        log_theta_standard_error: Standard error of ``log theta`` (``nan``
            for Poisson).
        parameters: ``(beta, gamma[, log theta])`` stacked.
        covariance: Inverse observed information of ``parameters``.
        p_values: Two-sided Wald p-values of ``parameters``.
        log_likelihood: Maximized log-likelihood.
        pointwise_log_likelihood: Per-observation terms.
        aic: ``-2 log L + 2k``.
        bic: ``-2 log L + k log n``.
        parameter_count: ``k``.
        fitted_means: ``E[y_i]``.
        zero_probabilities: Fitted ``P(y_i = 0)``.
        iterations: Newton iterations (count part for hurdle models).
        converged: Whether every part converged.
        n_observations: ``n``.
        count_rows: Count design rows.
        zero_rows: Zero design rows.
        offset: Count-part offsets.

    References:
        [EQ-CNT-ZI-001], [EQ-CNT-HURDLE-001], [REF-LAMBERT-1992],
        [REF-MULLAHY-1986].

    """

    model: str
    family: str
    count_coefficients: Vector
    zero_coefficients: Vector
    count_standard_errors: Vector
    zero_standard_errors: Vector
    count_column_names: tuple[str, ...]
    zero_column_names: tuple[str, ...]
    theta: float
    log_theta_standard_error: float
    parameters: Vector
    covariance: Matrix
    p_values: Vector
    log_likelihood: float
    pointwise_log_likelihood: Vector
    aic: float
    bic: float
    parameter_count: int
    fitted_means: Vector
    zero_probabilities: Vector
    iterations: int
    converged: bool
    n_observations: int
    count_rows: Matrix
    zero_rows: Matrix
    offset: Vector

    def count_rate_ratios(
        self, level: float = 0.95
    ) -> tuple[tuple[float, float, float], ...]:
        """Return ``exp(beta_j)`` with exponentiated Wald limits.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            ``(ratio, lower, upper)`` per count coefficient; ratios of the
            mean of the count part, not of ``E[y]``.

        Raises:
            ValueError: If ``level`` is invalid.

        Equation:
            [EQ-CNT-REG-001].

        References:
            [REF-CAMERON-TRIVEDI-2013, chapter "Generalized Count Regression"].

        """
        intervals = normal_intervals(
            self.count_coefficients, self.count_standard_errors, level
        )
        return tuple(
            (math.exp(b), math.exp(lo), math.exp(hi))
            for b, (lo, hi) in zip(self.count_coefficients, intervals, strict=True)
        )

    def predict(
        self,
        count_rows: Iterable[Iterable[object]],
        zero_rows: Iterable[Iterable[object]] | None = None,
        *,
        offset: Iterable[object] | None = None,
        exposure: Iterable[object] | None = None,
    ) -> tuple[Vector, Vector]:
        """Return ``(E[y], P(y = 0))`` at new rows.

        Args:
            count_rows: Count-part design rows.
            zero_rows: Zero-part rows; defaults to ``count_rows``.
            offset: Optional count-part offsets.
            exposure: Optional positive exposures.

        Returns:
            Expected counts and zero probabilities.

        Raises:
            ValueError: If shapes are invalid.

        Equation:
            [EQ-CNT-ZI-001], [EQ-CNT-HURDLE-001].

        References:
            [REF-LAMBERT-1992], [REF-MULLAHY-1986].

        """
        counts = as_matrix(count_rows, name="count_rows")
        zeros = (
            as_matrix(zero_rows, name="zero_rows") if zero_rows is not None else counts
        )
        if len(counts[0]) != len(self.count_coefficients):
            raise ValueError("count rows must have one entry per count column")
        if len(zeros[0]) != len(self.zero_coefficients) or len(zeros) != len(counts):
            raise ValueError("zero rows must match the zero design and row count")
        offsets = _offsets(offset, exposure, len(counts))
        eta = _linear(counts, self.count_coefficients, offsets)
        zeta = _linear(zeros, self.zero_coefficients, (0.0,) * len(zeros))
        return _means_and_zeros(self.model, eta, zeta, self.theta)


def _log_zero_mass(mu: float, theta: float) -> float:
    """Return ``log f(0)`` of the Poisson (``theta = inf``) or NB2 law.

    References:
        [EQ-CNT-REG-003].

    """
    return -mu if math.isinf(theta) else -theta * math.log1p(mu / theta)


def _means_and_zeros(
    model: str, eta: Vector, zeta: Vector, theta: float
) -> tuple[Vector, Vector]:
    """Return ``E[y]`` and ``P(y = 0)`` from the two linear predictors.

    References:
        [EQ-CNT-ZI-001], [EQ-CNT-HURDLE-001].

    """
    means: list[float] = []
    zeros: list[float] = []
    for e, z in zip(eta, zeta, strict=True):
        mu = math.exp(e)
        p0 = math.exp(_log_zero_mass(mu, theta))
        share = logistic(z)
        if model == "zero-inflated":
            means.append((1.0 - share) * mu)
            zeros.append(share + (1.0 - share) * p0)
        else:
            means.append(share * mu / (-math.expm1(_log_zero_mass(mu, theta))))
            zeros.append(1.0 - share)
    return tuple(means), tuple(zeros)


def _count_derivatives(
    y: float, mu: float, theta: float
) -> tuple[float, float, float, float, float]:
    """Return ``log f(y)``, ``log f(0)``, their ``eta`` scores, and the ``phi`` score.

    The tuple is ``(log f(y), log f(0), d log f(y)/d eta, d log f(0)/d eta,
    d log f(y)/d log theta)``.

    References:
        [EQ-CNT-REG-003], [REF-CAMERON-TRIVEDI-2013, chapter "Basic Count Regression"].

    """
    log_f = nb2_log_pmf(y, mu, theta)
    if math.isinf(theta):
        return log_f, -mu, y - mu, -mu, 0.0
    total = theta + mu
    d_eta = (y - mu) * theta / total
    d_phi = theta * (
        _digamma_difference(y, theta)
        - math.log1p(mu / theta)
        + 1.0
        - (y + theta) / total
    )
    zero_eta = -theta * mu / total
    return log_f, _log_zero_mass(mu, theta), d_eta, zero_eta, d_phi


def _zero_phi_score(mu: float, theta: float) -> float:
    """Return ``d log f(0) / d log theta = theta [mu/(theta+mu) - log(1 + mu/theta)]``.

    References:
        [EQ-CNT-REG-003].

    """
    return theta * (mu / (theta + mu) - math.log1p(mu / theta))


def _zi_terms(
    counts: Matrix,
    zeros: Matrix,
    y: Vector,
    offsets: Vector,
    parameters: Vector,
    negative_binomial: bool,
) -> tuple[Vector, Vector]:
    """Return per-observation log-likelihoods and the total gradient.

    ``y = 0``: ``l = log(pi + (1 - pi) f(0))``; ``y > 0``:
    ``l = log(1 - pi) + log f(y)``, with ``pi = logistic(z' gamma)``.

    References:
        [EQ-CNT-ZI-001], [REF-LAMBERT-1992].

    """
    p, q = len(counts[0]), len(zeros[0])
    beta, gamma = parameters[:p], parameters[p : p + q]
    theta = math.exp(parameters[p + q]) if negative_binomial else math.inf
    gradient = [0.0] * len(parameters)
    terms: list[float] = []
    for row, zrow, yi, oi in zip(counts, zeros, y, offsets, strict=True):
        eta = math.fsum(b * x for b, x in zip(beta, row, strict=True)) + oi
        if eta > 700.0:
            return (-math.inf,), tuple(gradient)
        zeta = math.fsum(g * z for g, z in zip(gamma, zrow, strict=True))
        mu = math.exp(eta)
        log_f, log_f0, d_eta, zero_eta, d_phi = _count_derivatives(yi, mu, theta)
        log_pi = -log1pexp(-zeta)
        log_one_minus = -log1pexp(zeta)
        pi = logistic(zeta)
        if yi == 0.0:
            a, b = log_pi, log_one_minus + log_f0
            top = max(a, b)
            value = top + math.log(math.exp(a - top) + math.exp(b - top))
            share = math.exp(b - value)
            g_eta = share * zero_eta
            g_zeta = (1.0 - share) - pi
            g_phi = share * _zero_phi_score(mu, theta) if negative_binomial else 0.0
        else:
            value = log_one_minus + log_f
            g_eta, g_zeta, g_phi = d_eta, -pi, d_phi
        terms.append(value)
        for a_index, x in enumerate(row):
            gradient[a_index] += g_eta * x
        for b_index, z in enumerate(zrow):
            gradient[p + b_index] += g_zeta * z
        if negative_binomial:
            gradient[p + q] += g_phi
    return tuple(terms), tuple(gradient)


def _zero_start(zeros: Matrix, intercept_index: int | None, share: float) -> Vector:
    """Return a logit starting vector with intercept ``logit(share)``.

    References:
        [REF-LAMBERT-1992]. No equation is evaluated.

    """
    start = [0.0] * len(zeros[0])
    if intercept_index is not None:
        clipped = min(0.9, max(0.05, share))
        start[intercept_index] = math.log(clipped / (1.0 - clipped))
    return tuple(start)


def _intercept_index(rows: Matrix) -> int | None:
    """Return the index of an all-ones column, if any.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    for j in range(len(rows[0])):
        if all(row[j] == 1.0 for row in rows):
            return j
    return None


def _zero_design(
    zero_design: DesignMatrix | Iterable[Iterable[object]] | None,
    zero_column_names: Sequence[str] | None,
    counts: Matrix,
    count_names: tuple[str, ...],
) -> tuple[Matrix, tuple[str, ...]]:
    """Return the zero-part design (the count design by default).

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if zero_design is None:
        return counts, count_names
    rows, names, _ = _coerce_design(zero_design, zero_column_names)
    if len(rows) != len(counts):
        raise ValueError("the zero design must have one row per observation")
    return rows, names


def _check_family(family: str) -> bool:
    """Validate the count family and return whether it is NB2.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if family not in ("poisson", "negative-binomial"):
        raise ValueError("family must be 'poisson' or 'negative-binomial'")
    return family == "negative-binomial"


def fit_zero_inflated(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    family: CountFamily = "poisson",
    zero_design: DesignMatrix | Iterable[Iterable[object]] | None = None,
    offset: Iterable[object] | None = None,
    exposure: Iterable[object] | None = None,
    column_names: Sequence[str] | None = None,
    zero_column_names: Sequence[str] | None = None,
    max_iterations: int = 200,
) -> ZeroCountResult:
    """Fit a zero-inflated Poisson or NB2 regression by maximum likelihood.

    ``P(y = 0) = pi + (1 - pi) f(0; mu)`` and
    ``P(y = k) = (1 - pi) f(k; mu)`` for ``k > 0``, with
    ``log mu = x' beta + o`` and ``logit pi = z' gamma``. Newton steps on
    the analytic gradient with a central-difference Hessian start from the
    Poisson (or NB2) regression, a logit intercept set to the excess-zero
    share, and the NB2 size of the plain fit. The covariance is the inverse
    observed information.

    Args:
        design: Count-part design (include an intercept column).
        response: Non-negative integer counts.
        family: ``"poisson"`` or ``"negative-binomial"``.
        zero_design: Inflation-part design; defaults to ``design``.
        offset: Count-part offsets.
        exposure: Positive exposures (``log t`` added to the offset).
        column_names: Names for a raw count design.
        zero_column_names: Names for a raw zero design.
        max_iterations: Newton iteration limit.

    Returns:
        :class:`ZeroCountResult` with ``model == "zero-inflated"``.

    Raises:
        ValueError: If inputs are invalid or there are no zeros.
        ArithmeticError: If the information matrix is singular (for example
            an inflation probability driven to zero).

    Equation:
        [EQ-CNT-ZI-001], [EQ-CNT-REG-003].

    References:
        [REF-LAMBERT-1992],
        [REF-CAMERON-TRIVEDI-2013, chapter "Generalized Count Regression"].

    """
    negative_binomial = _check_family(family)
    counts, names, y, offsets = _prepare(
        design, response, offset, exposure, column_names
    )
    zeros, zero_names = _zero_design(zero_design, zero_column_names, counts, names)
    n = len(y)
    observed_zero = sum(1 for v in y if v == 0.0) / n
    if observed_zero == 0.0:
        raise ValueError("a zero-inflated model needs at least one zero count")
    base = (
        fit_negative_binomial_regression(counts, y, offset=offsets, column_names=names)
        if negative_binomial
        else fit_poisson_regression(counts, y, offset=offsets, column_names=names)
    )
    theta0 = base.theta if math.isfinite(base.theta) else 10.0
    predicted_zero = (
        math.fsum(
            math.exp(_log_zero_mass(m, theta0 if negative_binomial else math.inf))
            for m in base.fitted_values
        )
        / n
    )
    excess = (observed_zero - predicted_zero) / max(1e-12, 1.0 - predicted_zero)
    start = (
        *base.coefficients,
        *_zero_start(zeros, _intercept_index(zeros), excess),
        *((math.log(theta0),) if negative_binomial else ()),
    )

    def objective(parameters: Vector) -> float:
        """Return the zero-inflated log-likelihood.

        References:
            [EQ-CNT-ZI-001].

        """
        terms, _ = _zi_terms(counts, zeros, y, offsets, parameters, negative_binomial)
        return math.fsum(terms)

    def gradient(parameters: Vector) -> Vector:
        """Return the analytic gradient.

        References:
            [EQ-CNT-ZI-001].

        """
        return _zi_terms(counts, zeros, y, offsets, parameters, negative_binomial)[1]

    outcome = newton_maximize(objective, gradient, start, max_iterations=max_iterations)
    covariance = inverse_information(outcome.hessian)
    pointwise, _ = _zi_terms(counts, zeros, y, offsets, outcome.x, negative_binomial)
    return _assemble(
        "zero-inflated",
        family,
        counts,
        zeros,
        names,
        zero_names,
        offsets,
        outcome.x,
        covariance,
        pointwise,
        outcome.iterations,
        outcome.converged,
    )


def _assemble(
    model: str,
    family: str,
    counts: Matrix,
    zeros: Matrix,
    names: tuple[str, ...],
    zero_names: tuple[str, ...],
    offsets: Vector,
    parameters: Vector,
    covariance: Matrix,
    pointwise: Vector,
    iterations: int,
    converged: bool,
) -> ZeroCountResult:
    """Assemble a :class:`ZeroCountResult`; ``parameters`` is ``(beta, gamma[, phi])``.

    References:
        [EQ-CNT-ZI-001], [EQ-CNT-HURDLE-001].

    """
    p, q = len(counts[0]), len(zeros[0])
    negative_binomial = family == "negative-binomial"
    errors, _, p_values = wald_table(parameters, covariance)
    theta = math.exp(parameters[p + q]) if negative_binomial else math.inf
    eta = _linear(counts, parameters[:p], offsets)
    zeta = _linear(zeros, parameters[p : p + q], (0.0,) * len(zeros))
    means, zero_probabilities = _means_and_zeros(model, eta, zeta, theta)
    log_likelihood = math.fsum(pointwise)
    k = len(parameters)
    aic, _, bic = information_criteria(log_likelihood, k, len(counts))
    return ZeroCountResult(
        model=model,
        family=family,
        count_coefficients=parameters[:p],
        zero_coefficients=parameters[p : p + q],
        count_standard_errors=errors[:p],
        zero_standard_errors=errors[p : p + q],
        count_column_names=names,
        zero_column_names=zero_names,
        theta=theta,
        log_theta_standard_error=errors[p + q] if negative_binomial else math.nan,
        parameters=parameters,
        covariance=covariance,
        p_values=p_values,
        log_likelihood=log_likelihood,
        pointwise_log_likelihood=pointwise,
        aic=aic,
        bic=bic,
        parameter_count=k,
        fitted_means=means,
        zero_probabilities=zero_probabilities,
        iterations=iterations,
        converged=converged,
        n_observations=len(counts),
        count_rows=counts,
        zero_rows=zeros,
        offset=offsets,
    )


def _truncated_terms(
    rows: Matrix,
    y: Vector,
    offsets: Vector,
    parameters: Vector,
    negative_binomial: bool,
) -> tuple[Vector, Vector]:
    """Return zero-truncated log-likelihood terms and gradient for ``y > 0``.

    ``l = log f(y) - log(1 - f(0))``; the truncation adds
    ``f(0)/(1 - f(0))`` times the score of ``log f(0)``.

    References:
        [EQ-CNT-HURDLE-001], [REF-MULLAHY-1986].

    """
    p = len(rows[0])
    beta = parameters[:p]
    theta = math.exp(parameters[p]) if negative_binomial else math.inf
    gradient = [0.0] * len(parameters)
    terms: list[float] = []
    for row, yi, oi in zip(rows, y, offsets, strict=True):
        eta = math.fsum(b * x for b, x in zip(beta, row, strict=True)) + oi
        if eta > 700.0:
            return (-math.inf,), tuple(gradient)
        mu = math.exp(eta)
        log_f, log_f0, d_eta, zero_eta, d_phi = _count_derivatives(yi, mu, theta)
        positive = -math.expm1(log_f0)
        if positive <= 0.0:
            return (-math.inf,), tuple(gradient)
        ratio = math.exp(log_f0) / positive
        terms.append(log_f - math.log(positive))
        g_eta = d_eta + ratio * zero_eta
        for a, x in enumerate(row):
            gradient[a] += g_eta * x
        if negative_binomial:
            gradient[p] += d_phi + ratio * _zero_phi_score(mu, theta)
    return tuple(terms), tuple(gradient)


def fit_hurdle(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    family: CountFamily = "poisson",
    zero_design: DesignMatrix | Iterable[Iterable[object]] | None = None,
    offset: Iterable[object] | None = None,
    exposure: Iterable[object] | None = None,
    column_names: Sequence[str] | None = None,
    zero_column_names: Sequence[str] | None = None,
    max_iterations: int = 200,
) -> ZeroCountResult:
    """Fit a logit hurdle model with a zero-truncated Poisson or NB2 count part.

    ``P(y = 0) = 1 - p`` and ``P(y = k) = p f(k; mu) / (1 - f(0; mu))`` for
    ``k > 0``, with ``logit p = z' gamma`` (the probability of crossing the
    hurdle) and ``log mu = x' beta + o``. The likelihood factorizes: the
    binary part is a logistic regression of ``1{y > 0}`` on ``z`` (by
    :func:`fit_glm`), and the truncated count part is fitted on the positive
    counts by Newton steps with a central-difference Hessian. The
    covariance is block diagonal.

    Args:
        design: Count-part design (include an intercept column).
        response: Non-negative integer counts.
        family: ``"poisson"`` or ``"negative-binomial"``.
        zero_design: Hurdle-part design; defaults to ``design``.
        offset: Count-part offsets.
        exposure: Positive exposures.
        column_names: Names for a raw count design.
        zero_column_names: Names for a raw zero design.
        max_iterations: Newton iteration limit.

    Returns:
        :class:`ZeroCountResult` with ``model == "hurdle"``; the zero
        coefficients model ``P(y > 0)``.

    Raises:
        SeparationError: If the binary part is separated.
        ValueError: If inputs are invalid, there are no zeros, or fewer
            positive counts than count coefficients.
        ArithmeticError: If the information matrix is singular.

    Equation:
        [EQ-CNT-HURDLE-001], [EQ-CNT-REG-003].

    References:
        [REF-MULLAHY-1986],
        [REF-CAMERON-TRIVEDI-2013, chapter "Generalized Count Regression"].

    """
    negative_binomial = _check_family(family)
    counts, names, y, offsets = _prepare(
        design, response, offset, exposure, column_names
    )
    zeros, zero_names = _zero_design(zero_design, zero_column_names, counts, names)
    if not any(v == 0.0 for v in y):
        raise ValueError("a hurdle model needs at least one zero count")
    crossed = tuple(1.0 if v > 0.0 else 0.0 for v in y)
    binary = fit_glm(zeros, crossed, family="binomial", column_names=zero_names)
    positive = [i for i, v in enumerate(y) if v > 0.0]
    p = len(counts[0])
    if len(positive) <= p:
        raise ValueError("too few positive counts for the count part")
    rows_pos = tuple(counts[i] for i in positive)
    y_pos = tuple(y[i] for i in positive)
    off_pos = tuple(offsets[i] for i in positive)
    start_fit = fit_poisson_regression(rows_pos, y_pos, offset=off_pos)
    start = (
        *start_fit.coefficients,
        *((0.0,) if negative_binomial else ()),
    )

    def objective(parameters: Vector) -> float:
        """Return the truncated count log-likelihood.

        References:
            [EQ-CNT-HURDLE-001].

        """
        terms, _ = _truncated_terms(
            rows_pos, y_pos, off_pos, parameters, negative_binomial
        )
        return math.fsum(terms)

    def gradient(parameters: Vector) -> Vector:
        """Return the analytic gradient of the truncated part.

        References:
            [EQ-CNT-HURDLE-001].

        """
        return _truncated_terms(
            rows_pos, y_pos, off_pos, parameters, negative_binomial
        )[1]

    outcome = newton_maximize(objective, gradient, start, max_iterations=max_iterations)
    count_cov = inverse_information(outcome.hessian)
    q = len(zeros[0])
    count_size = len(outcome.x)
    size = count_size + q
    covariance = [[0.0] * size for _ in range(size)]
    order = list(range(p)) + list(range(p + q, size))
    for a, ia in enumerate(range(count_size)):
        for b, ib in enumerate(range(count_size)):
            covariance[order[ia]][order[ib]] = count_cov[a][b]
    for a in range(q):
        for b in range(q):
            covariance[p + a][p + b] = binary.covariance[a][b]
    parameters = (
        *outcome.x[:p],
        *binary.coefficients,
        *outcome.x[p:],
    )
    truncated, _ = _truncated_terms(
        rows_pos, y_pos, off_pos, outcome.x, negative_binomial
    )
    pointwise = [0.0] * len(y)
    lookup = dict(zip(positive, truncated, strict=True))
    for i, (ci, mi) in enumerate(zip(crossed, binary.fitted_values, strict=True)):
        if ci == 1.0:
            pointwise[i] = math.log(mi) + lookup[i]
        else:
            pointwise[i] = math.log1p(-mi)
    return _assemble(
        "hurdle",
        family,
        counts,
        zeros,
        names,
        zero_names,
        offsets,
        tuple(parameters),
        tuple(tuple(row) for row in covariance),
        tuple(pointwise),
        outcome.iterations,
        outcome.converged and binary.converged,
    )


@dataclass(frozen=True, slots=True)
class VuongTest:
    """Vuong's statistic for two non-nested models.

    Attributes:
        statistic: ``(sum m_i - c) / (sqrt(n) omega)``, ``m_i`` the
            per-observation log-likelihood ratio of model A over model B.
        p_value: Two-sided standard normal p-value.
        mean_difference: ``mean(m_i)``.
        omega: ``sqrt((1/n) sum (m_i - mean m)^2)``.
        correction: ``"none"``, ``"aic"`` (``c = k_A - k_B``), or ``"bic"``
            (``c = (k_A - k_B) log(n) / 2``).
        preferred: ``"a"`` or ``"b"`` when ``|statistic|`` exceeds the 0.975
            normal quantile, otherwise ``"neither"``.
        n_observations: ``n``.

    References:
        [EQ-CNT-VUONG-001], [REF-VUONG-1989], [REF-WILSON-2015].

    """

    statistic: float
    p_value: float
    mean_difference: float
    omega: float
    correction: str
    preferred: str
    n_observations: int


def vuong_test(
    log_likelihood_a: Iterable[object],
    log_likelihood_b: Iterable[object],
    *,
    parameters_a: int,
    parameters_b: int,
    correction: Correction = "none",
) -> VuongTest:
    """Return Vuong's non-nested likelihood-ratio statistic.

    Under the null that both strictly non-nested models are equally close to
    the truth (in Kullback-Leibler divergence), the statistic is
    asymptotically ``N(0, 1)``. A zero-inflated model and its count model
    without inflation are *not* strictly non-nested (the latter is the
    boundary ``pi = 0``), so this statistic is not a valid test for zero
    inflation; compare such models by AIC/BIC or a boundary-corrected
    likelihood-ratio test instead [REF-WILSON-2015].

    Args:
        log_likelihood_a: Per-observation log-likelihoods of model A (for
            example ``result.pointwise_log_likelihood``).
        log_likelihood_b: The same for model B, in the same order.
        parameters_a: Parameter count of model A.
        parameters_b: Parameter count of model B.
        correction: ``"none"``, ``"aic"``, or ``"bic"``.

    Returns:
        :class:`VuongTest`.

    Raises:
        ValueError: If the inputs are invalid or every ``m_i`` is equal.

    Equation:
        [EQ-CNT-VUONG-001].

    References:
        [REF-VUONG-1989], [REF-WILSON-2015].

    """
    from probstat.linalg import as_vector

    first = as_vector(log_likelihood_a, name="log_likelihood_a")
    second = as_vector(log_likelihood_b, name="log_likelihood_b")
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("both models need the same two or more observations")
    for name, value in (("parameters_a", parameters_a), ("parameters_b", parameters_b)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if correction not in ("none", "aic", "bic"):
        raise ValueError("correction must be 'none', 'aic', or 'bic'")
    n = len(first)
    m = tuple(a - b for a, b in zip(first, second, strict=True))
    mean = math.fsum(m) / n
    omega = math.sqrt(math.fsum((v - mean) ** 2 for v in m) / n)
    if omega == 0.0:
        raise ValueError("the log-likelihood ratios have zero variance")
    shift = 0.0
    if correction == "aic":
        shift = float(parameters_a - parameters_b)
    elif correction == "bic":
        shift = (parameters_a - parameters_b) * math.log(n) / 2.0
    statistic = (math.fsum(m) - shift) / (math.sqrt(n) * omega)
    p_value = 2.0 * standard_normal_sf(abs(statistic))
    critical = 1.959963984540054
    preferred = (
        "a" if statistic > critical else "b" if statistic < -critical else "neither"
    )
    return VuongTest(
        statistic=statistic,
        p_value=p_value,
        mean_difference=mean,
        omega=omega,
        correction=correction,
        preferred=preferred,
        n_observations=n,
    )
