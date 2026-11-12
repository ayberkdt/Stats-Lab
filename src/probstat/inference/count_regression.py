"""Poisson and negative binomial (NB2) regression for counts with exposures.

The mean model is ``log mu_i = x_i' beta + o_i`` with ``o_i = log t_i`` for
an exposure ``t_i`` [EQ-CNT-REG-001]. :func:`fit_poisson_regression` fits the
Poisson model by the IRLS of :func:`probstat.inference.fit_glm` and reports
either the model-based or the robust (sandwich) covariance, which stays
consistent for ``beta`` whenever the mean is correctly specified, whatever
the variance [EQ-CNT-REG-002]. :func:`fit_negative_binomial_regression`
fits the NB2 model ``Var(y) = mu + alpha mu^2`` by joint maximum likelihood
in ``beta`` and ``log theta`` (``theta = 1/alpha``) [EQ-CNT-REG-003], and
:func:`poisson_overdispersion_test` is the auxiliary-regression test of
Cameron and Trivedi (1990) of ``alpha = 0`` against ``alpha > 0``
[EQ-CNT-REG-004].
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.linalg import Matrix, Vector, as_matrix, as_vector

from ._ml_tools import (
    MarginalEffects,
    delta_method_effects,
    inverse_information,
    newton_maximize,
    normal_intervals,
    resolve_column,
    set_column,
    wald_table,
)
from .fitting import information_criteria
from .glm import fit_glm
from .linear_models import DesignMatrix, _coerce_design
from .sampling_distributions import standard_normal_sf

CovarianceKind = Literal["model", "robust"]
DispersionForm = Literal["nb2", "nb1"]
_THETA_CEILING = 1e12


def _count_response(response: Iterable[object], n: int) -> Vector:
    """Validate non-negative integer-valued counts, one per design row.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    y = as_vector(response, name="response")
    if len(y) != n:
        raise ValueError("response length must equal the number of design rows")
    for index, value in enumerate(y):
        if value < 0.0 or value != math.floor(value):
            raise ValueError(f"response[{index}] must be a non-negative integer count")
    return y


def _offsets(
    offset: Iterable[object] | None, exposure: Iterable[object] | None, n: int
) -> Vector:
    """Return ``o_i + log t_i`` from optional offsets and exposures.

    References:
        [EQ-CNT-REG-001], [REF-CAMERON-TRIVEDI-2013, chapter "Basic Count Regression"].

    """
    values = [0.0] * n
    if offset is not None:
        checked = as_vector(offset, name="offset")
        if len(checked) != n:
            raise ValueError("offset must have one entry per observation")
        values = [a + b for a, b in zip(values, checked, strict=True)]
    if exposure is not None:
        times = as_vector(exposure, name="exposure")
        if len(times) != n:
            raise ValueError("exposure must have one entry per observation")
        if any(t <= 0.0 for t in times):
            raise ValueError("exposure must be strictly positive")
        values = [a + math.log(t) for a, t in zip(values, times, strict=True)]
    return tuple(values)


def _linear(rows: Matrix, beta: Sequence[float], offset: Vector) -> Vector:
    """Return ``X beta + o``.

    References:
        [EQ-CNT-REG-001]. No other equation is evaluated.

    """
    return tuple(
        math.fsum(b * x for b, x in zip(beta, row, strict=True)) + o
        for row, o in zip(rows, offset, strict=True)
    )


def _digamma_difference(y: float, theta: float) -> float:
    """Return ``psi(y + theta) - psi(theta) = sum_(j<y) 1/(theta + j)``.

    References:
        [EQ-CNT-REG-003], [REF-NIST-DLMF-GAMMA, Section 5.5].

    """
    return math.fsum(1.0 / (theta + j) for j in range(int(y)))


def _trigamma_difference(y: float, theta: float) -> float:
    """Return ``psi'(theta) - psi'(y + theta) = sum_(j<y) 1/(theta + j)^2``.

    References:
        [EQ-CNT-REG-003], [REF-NIST-DLMF-GAMMA, Section 5.15].

    """
    return math.fsum(1.0 / (theta + j) ** 2 for j in range(int(y)))


def nb2_log_pmf(y: float, mu: float, theta: float) -> float:
    """Return the NB2 log mass with mean ``mu`` and size ``theta``.

    Args:
        y: Non-negative integer count.
        mu: Mean ``mu > 0``.
        theta: Size ``theta > 0``; ``inf`` gives the Poisson law.

    Returns:
        ``log Gamma(y + theta) - log Gamma(theta) - log y! +
        theta log(theta / (theta + mu)) + y log(mu / (theta + mu))``.

    Equation:
        [EQ-CNT-REG-003], [EQ-DIST-POIS-001].

    References:
        [REF-CAMERON-TRIVEDI-2013, chapter "Basic Count Regression"].

    """
    if math.isinf(theta):
        return (y * math.log(mu) if y > 0.0 else 0.0) - mu - math.lgamma(y + 1.0)
    value = (
        math.lgamma(y + theta)
        - math.lgamma(theta)
        - math.lgamma(y + 1.0)
        - theta * math.log1p(mu / theta)
    )
    if y > 0.0:
        value += y * (math.log(mu) - math.log(theta + mu))
    return value


@dataclass(frozen=True, slots=True)
class CountRegressionResult:
    """Maximum-likelihood Poisson or NB2 regression.

    Attributes:
        model: ``"poisson"`` or ``"negative-binomial"``.
        coefficients: ``beta`` on the log-mean scale.
        standard_errors: Square roots of the diagonal of ``covariance``.
        statistics: Wald ratios ``beta_j / se_j``.
        p_values: Two-sided normal p-values.
        column_names: Design-column names.
        covariance: Coefficient covariance of kind ``covariance_type``.
        model_covariance: Model-based ``(X'WX)^-1`` with
            ``W = mu / (1 + alpha mu)`` (expected information).
        covariance_type: ``"model"`` or ``"robust"`` (sandwich, HC0).
        theta: NB2 size ``theta = 1/alpha`` (``inf`` for Poisson or at the
            Poisson boundary).
        theta_standard_error: ``1 / sqrt(observed information of theta)``
            at the fitted ``beta`` (``nan`` when ``theta`` is infinite).
        alpha: Dispersion ``alpha = 1/theta`` (``0`` for Poisson).
        alpha_standard_error: Delta-method ``se(theta) / theta^2``.
        at_boundary: Whether the NB2 maximum is the Poisson limit.
        fitted_values: ``mu_i``.
        linear_predictor: ``eta_i = x_i' beta + o_i``.
        offset: ``o_i`` including ``log`` exposures.
        response: Observed counts.
        design_rows: Design rows (for marginal effects).
        log_likelihood: Maximized log-likelihood.
        pointwise_log_likelihood: Per-observation log-likelihood terms.
        aic: ``-2 log L + 2k``.
        bic: ``-2 log L + k log n``.
        parameter_count: ``k = p`` (+1 for the NB2 size).
        pearson_chi2: ``sum (y - mu)^2 / V(mu)``.
        deviance: ``2 sum [log f(y_i; y_i) - log f(y_i; mu_i)]`` at the
            fitted size.
        iterations: Newton or IRLS iterations used.
        converged: Whether the fit converged.
        n_observations: ``n``.

    References:
        [EQ-CNT-REG-001], [EQ-CNT-REG-002], [EQ-CNT-REG-003],
        [REF-CAMERON-TRIVEDI-2013, chapter "Basic Count Regression"].

    """

    model: str
    coefficients: Vector
    standard_errors: Vector
    statistics: Vector
    p_values: Vector
    column_names: tuple[str, ...]
    covariance: Matrix
    model_covariance: Matrix
    covariance_type: str
    theta: float
    theta_standard_error: float
    alpha: float
    alpha_standard_error: float
    at_boundary: bool
    fitted_values: Vector
    linear_predictor: Vector
    offset: Vector
    response: Vector
    design_rows: Matrix
    log_likelihood: float
    pointwise_log_likelihood: Vector
    aic: float
    bic: float
    parameter_count: int
    pearson_chi2: float
    deviance: float
    iterations: int
    converged: bool
    n_observations: int

    def coefficient_intervals(
        self, level: float = 0.95
    ) -> tuple[tuple[float, float], ...]:
        """Return normal Wald intervals for ``beta``.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            One ``(lower, upper)`` pair per coefficient.

        Raises:
            ValueError: If ``level`` is invalid.

        Equation:
            [EQ-GLM-WALD-001].

        References:
            [REF-CAMERON-TRIVEDI-2013, chapter "Basic Count Regression"].

        """
        return normal_intervals(self.coefficients, self.standard_errors, level)

    def incidence_rate_ratios(
        self, level: float = 0.95
    ) -> tuple[tuple[float, float, float], ...]:
        """Return ``exp(beta_j)`` with the exponentiated Wald interval.

        A one-unit increase of column ``j`` multiplies the expected count (per
        unit of exposure) by ``exp(beta_j)``, other columns held fixed.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            ``(ratio, lower, upper)`` per coefficient.

        Raises:
            ValueError: If ``level`` is invalid.

        Equation:
            [EQ-CNT-REG-001].

        References:
            [REF-CAMERON-TRIVEDI-2013, chapter "Basic Count Regression"].

        """
        return tuple(
            (math.exp(b), math.exp(lower), math.exp(upper))
            for b, (lower, upper) in zip(
                self.coefficients, self.coefficient_intervals(level), strict=True
            )
        )

    def predict(
        self,
        rows: Iterable[Iterable[object]],
        *,
        offset: Iterable[object] | None = None,
        exposure: Iterable[object] | None = None,
    ) -> Vector:
        """Return fitted means ``exp(x' beta + o)`` at new rows.

        Args:
            rows: Design rows encoded like the fitting design.
            offset: Optional offsets.
            exposure: Optional positive exposures (``o += log t``).

        Returns:
            Expected counts.

        Raises:
            ValueError: If shapes are invalid.

        Equation:
            [EQ-CNT-REG-001].

        References:
            [REF-CAMERON-TRIVEDI-2013, chapter "Basic Count Regression"].

        """
        checked = as_matrix(rows, name="rows")
        if len(checked[0]) != len(self.coefficients):
            raise ValueError("new rows must have one entry per design column")
        offsets = _offsets(offset, exposure, len(checked))
        return tuple(math.exp(e) for e in _linear(checked, self.coefficients, offsets))

    def average_marginal_effect(
        self, column: str | int, *, discrete: bool = False
    ) -> MarginalEffects:
        """Return the average marginal effect of a column on the mean count.

        The derivative effect is ``(1/n) sum_i beta_j mu_i``; the discrete
        effect is ``(1/n) sum_i [mu_i(x_ij = 1) - mu_i(x_ij = 0)]``, the
        right summary for an indicator column. Standard errors use the delta
        method with ``covariance``.

        Args:
            column: Column name or index (not the intercept).
            discrete: Use the discrete change from 0 to 1.

        Returns:
            :class:`MarginalEffects` with the single outcome ``"mean"``.

        Raises:
            KeyError: If the column is unknown.
            ValueError: If the column is constant.

        Equation:
            [EQ-REG-AME-001].

        References:
            [REF-CAMERON-TRIVEDI-2013], [REF-CASELLA-BERGER-2002, Section 5.5].

        """
        index = resolve_column(self.column_names, column)
        rows = self.design_rows
        if len({row[index] for row in rows}) == 1:
            raise ValueError("marginal effects of a constant column are undefined")
        offsets = self.offset

        def effect(beta: Vector) -> Vector:
            """Return the AME as a function of the coefficients.

            References:
                [EQ-REG-AME-001].

            """
            if discrete:
                high = _linear(set_column(rows, index, 1.0), beta, offsets)
                low = _linear(set_column(rows, index, 0.0), beta, offsets)
                return (
                    math.fsum(
                        math.exp(a) - math.exp(b)
                        for a, b in zip(high, low, strict=True)
                    )
                    / len(rows),
                )
            means = _linear(rows, beta, offsets)
            return (beta[index] * math.fsum(math.exp(e) for e in means) / len(rows),)

        return delta_method_effects(
            effect,
            self.coefficients,
            self.covariance,
            column=self.column_names[index],
            kind="discrete" if discrete else "derivative",
            outcomes=("mean",),
        )


def _prepare(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    offset: Iterable[object] | None,
    exposure: Iterable[object] | None,
    column_names: Sequence[str] | None,
) -> tuple[Matrix, tuple[str, ...], Vector, Vector]:
    """Validate the design, counts, and offsets of a count regression.

    References:
        [EQ-CNT-REG-001], [REF-PROBSTAT-VALIDATION].

    """
    rows, names, _ = _coerce_design(design, column_names)
    n = len(rows)
    if n <= len(rows[0]):
        raise ValueError("count regression needs more observations than coefficients")
    y = _count_response(response, n)
    if not any(value > 0.0 for value in y):
        raise ValueError("at least one count must be positive")
    return rows, names, y, _offsets(offset, exposure, n)


def _sandwich(rows: Matrix, bread: Matrix, scores: Vector) -> Matrix:
    """Return ``A^-1 (sum s_i^2 x_i x_i') A^-1`` with ``A^-1 = bread``.

    References:
        [EQ-CNT-REG-002], [REF-WHITE-1982].

    """
    p = len(bread)
    meat = [[0.0] * p for _ in range(p)]
    for row, s in zip(rows, scores, strict=True):
        s2 = s * s
        for a in range(p):
            for b in range(a + 1):
                meat[a][b] += s2 * row[a] * row[b]
    for a in range(p):
        for b in range(a):
            meat[b][a] = meat[a][b]
    left = tuple(
        tuple(math.fsum(bread[i][k] * meat[k][j] for k in range(p)) for j in range(p))
        for i in range(p)
    )
    return tuple(
        tuple(math.fsum(left[i][k] * bread[k][j] for k in range(p)) for j in range(p))
        for i in range(p)
    )


def _build(
    model: str,
    rows: Matrix,
    names: tuple[str, ...],
    y: Vector,
    offsets: Vector,
    beta: Vector,
    theta: float,
    theta_se: float,
    model_cov: Matrix,
    covariance_type: str,
    at_boundary: bool,
    iterations: int,
    converged: bool,
) -> CountRegressionResult:
    """Assemble a :class:`CountRegressionResult` from fitted parameters.

    References:
        [EQ-CNT-REG-001], [EQ-CNT-REG-002], [EQ-CNT-REG-003].

    """
    eta = _linear(rows, beta, offsets)
    mu = tuple(math.exp(e) for e in eta)
    alpha = 0.0 if math.isinf(theta) else 1.0 / theta
    if covariance_type == "robust":
        scores = tuple(
            (yi - mi) / (1.0 + alpha * mi) for yi, mi in zip(y, mu, strict=True)
        )
        covariance = _sandwich(rows, model_cov, scores)
    else:
        covariance = model_cov
    errors, ratios, p_values = wald_table(beta, covariance)
    pointwise = tuple(nb2_log_pmf(yi, mi, theta) for yi, mi in zip(y, mu, strict=True))
    log_likelihood = math.fsum(pointwise)
    k = len(beta) + (0 if model == "poisson" else 1)
    aic, _, bic = information_criteria(log_likelihood, k, len(y))
    pearson = math.fsum(
        (yi - mi) ** 2 / (mi + alpha * mi * mi) for yi, mi in zip(y, mu, strict=True)
    )
    deviance = 2.0 * math.fsum(
        (nb2_log_pmf(yi, yi, theta) if yi > 0.0 else 0.0) - li
        for yi, li in zip(y, pointwise, strict=True)
    )
    return CountRegressionResult(
        model=model,
        coefficients=beta,
        standard_errors=errors,
        statistics=ratios,
        p_values=p_values,
        column_names=names,
        covariance=covariance,
        model_covariance=model_cov,
        covariance_type=covariance_type,
        theta=theta,
        theta_standard_error=theta_se,
        alpha=alpha,
        alpha_standard_error=(
            theta_se / (theta * theta) if math.isfinite(theta) else math.nan
        ),
        at_boundary=at_boundary,
        fitted_values=mu,
        linear_predictor=eta,
        offset=offsets,
        response=y,
        design_rows=rows,
        log_likelihood=log_likelihood,
        pointwise_log_likelihood=pointwise,
        aic=aic,
        bic=bic,
        parameter_count=k,
        pearson_chi2=pearson,
        deviance=deviance,
        iterations=iterations,
        converged=converged,
        n_observations=len(y),
    )


def _check_covariance(covariance: str) -> None:
    """Validate the covariance kind.

    References:
        [EQ-CNT-REG-002], [REF-PROBSTAT-VALIDATION].

    """
    if covariance not in ("model", "robust"):
        raise ValueError("covariance must be 'model' or 'robust'")


def fit_poisson_regression(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    offset: Iterable[object] | None = None,
    exposure: Iterable[object] | None = None,
    column_names: Sequence[str] | None = None,
    covariance: CovarianceKind = "model",
) -> CountRegressionResult:
    """Fit Poisson regression ``log mu = x' beta + log t`` by IRLS.

    The coefficients are those of :func:`fit_glm` with the Poisson family.
    ``covariance="robust"`` replaces ``(X' diag(mu) X)^-1`` by the sandwich
    ``A^-1 B A^-1`` with ``B = sum (y_i - mu_i)^2 x_i x_i'`` (HC0, no
    small-sample factor), which is consistent under any variance function as
    long as the mean model is right.

    Args:
        design: :class:`DesignMatrix` or ``n x p`` rows (include an intercept
            column explicitly).
        response: Non-negative integer counts.
        offset: Known additive terms of the linear predictor.
        exposure: Positive exposures; ``log t`` is added to the offset, so
            ``exp(beta)`` are rate ratios per unit of exposure.
        column_names: Names for a raw design.
        covariance: ``"model"`` or ``"robust"``.

    Returns:
        :class:`CountRegressionResult` with ``model == "poisson"``.

    Raises:
        RankDeficientDesignError: If the design is rank deficient.
        ValueError: If inputs are invalid.

    Equation:
        [EQ-CNT-REG-001], [EQ-CNT-REG-002], [EQ-GLM-IRLS-001].

    References:
        [REF-CAMERON-TRIVEDI-2013, chapter "Basic Count Regression"], [REF-WHITE-1982],
        [REF-MCCULLAGH-NELDER-1989, Chapter 6].

    """
    _check_covariance(covariance)
    rows, names, y, offsets = _prepare(design, response, offset, exposure, column_names)
    glm = fit_glm(rows, y, family="poisson", offset=offsets, column_names=names)
    return _build(
        "poisson",
        rows,
        names,
        y,
        offsets,
        glm.coefficients,
        math.inf,
        math.nan,
        glm.unscaled_covariance,
        covariance,
        False,
        glm.iterations,
        glm.converged,
    )


def _nb_terms(
    rows: Matrix, y: Vector, offsets: Vector, parameters: Vector
) -> tuple[float, Vector, Matrix]:
    """Return the NB2 log-likelihood, gradient, and Hessian in ``(beta, log theta)``.

    With ``theta = exp(phi)``: ``dl/d eta = (y - mu) theta / (theta + mu)``,
    ``dl/d theta = psi(y + theta) - psi(theta) + log(theta / (theta + mu))
    + 1 - (y + theta)/(theta + mu)``; the second derivatives are those of
    [EQ-CNT-REG-003].

    References:
        [EQ-CNT-REG-003], [REF-CAMERON-TRIVEDI-2013, chapter "Basic Count Regression"].

    """
    p = len(rows[0])
    beta, phi = parameters[:p], parameters[p]
    theta = math.exp(phi)
    eta = _linear(rows, beta, offsets)
    gradient = [0.0] * (p + 1)
    hessian = [[0.0] * (p + 1) for _ in range(p + 1)]
    terms: list[float] = []
    for row, yi, ei in zip(rows, y, eta, strict=True):
        if ei > 700.0:
            return -math.inf, tuple(gradient), tuple(tuple(r) for r in hessian)
        mu = math.exp(ei)
        terms.append(nb2_log_pmf(yi, mu, theta))
        total = theta + mu
        d_eta = (yi - mu) * theta / total
        d_theta = (
            _digamma_difference(yi, theta)
            - math.log1p(mu / theta)
            + 1.0
            - (yi + theta) / total
        )
        h_eta = -mu * theta * (theta + yi) / (total * total)
        h_eta_theta = (yi - mu) * mu / (total * total)
        h_theta = (
            -_trigamma_difference(yi, theta)
            + 1.0 / theta
            - 1.0 / total
            + (yi - mu) / (total * total)
        )
        d_phi = theta * d_theta
        h_phi = theta * theta * h_theta + theta * d_theta
        h_eta_phi = theta * h_eta_theta
        for a in range(p):
            gradient[a] += d_eta * row[a]
            hessian[a][p] += h_eta_phi * row[a]
            for b in range(a + 1):
                hessian[a][b] += h_eta * row[a] * row[b]
        gradient[p] += d_phi
        hessian[p][p] += h_phi
    for a in range(p):
        hessian[p][a] = hessian[a][p]
        for b in range(a):
            hessian[b][a] = hessian[a][b]
    return math.fsum(terms), tuple(gradient), tuple(tuple(r) for r in hessian)


def fit_negative_binomial_regression(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    offset: Iterable[object] | None = None,
    exposure: Iterable[object] | None = None,
    column_names: Sequence[str] | None = None,
    covariance: CovarianceKind = "model",
    max_iterations: int = 200,
) -> CountRegressionResult:
    """Fit NB2 regression ``Var(y) = mu + mu^2 / theta`` by maximum likelihood.

    Newton-Raphson on ``(beta, log theta)`` with the analytic Hessian starts
    from the Poisson fit and a moment estimate of ``alpha``. When
    ``sum_i [(y_i - mu_i)^2 - y_i] <= 0`` at the Poisson fit (the leading
    term of the ``theta`` score as ``theta -> infinity``), or Newton drives
    ``theta`` beyond ``1e12``, the maximum is the Poisson boundary and the
    Poisson fit is returned with ``theta = inf`` and ``at_boundary``. The
    coefficient covariance is the inverse expected information
    ``(X'WX)^-1``, ``W = mu / (1 + alpha mu)`` (block-diagonal against
    ``theta``); ``theta`` has the observed-information standard error.

    Args:
        design: :class:`DesignMatrix` or ``n x p`` rows.
        response: Non-negative integer counts.
        offset: Known additive terms of the linear predictor.
        exposure: Positive exposures (``log t`` is added to the offset).
        column_names: Names for a raw design.
        covariance: ``"model"`` or ``"robust"`` (sandwich with the NB2 score
            ``(y - mu)/(1 + alpha mu)``).
        max_iterations: Newton iteration limit.

    Returns:
        :class:`CountRegressionResult` with ``model == "negative-binomial"``;
        the size counts as a parameter in the information criteria even at
        the boundary.

    Raises:
        RankDeficientDesignError: If the design is rank deficient.
        ValueError: If inputs are invalid.
        ArithmeticError: If the information matrix is singular.

    Equation:
        [EQ-CNT-REG-001], [EQ-CNT-REG-003], [EQ-CNT-REG-002].

    References:
        [REF-CAMERON-TRIVEDI-2013, chapter "Basic Count Regression"].

    """
    _check_covariance(covariance)
    rows, names, y, offsets = _prepare(design, response, offset, exposure, column_names)
    poisson = fit_glm(rows, y, family="poisson", offset=offsets, column_names=names)
    mu0 = poisson.fitted_values
    excess = math.fsum((yi - mi) ** 2 - yi for yi, mi in zip(y, mu0, strict=True))
    p = len(rows[0])
    if excess > 0.0:
        alpha0 = excess / math.fsum(mi * mi for mi in mu0)
        start = (*poisson.coefficients, -math.log(alpha0))

        def objective(parameters: Vector) -> float:
            """Return the NB2 log-likelihood.

            References:
                [EQ-CNT-REG-003].

            """
            if parameters[p] > math.log(_THETA_CEILING) + 5.0:
                return -math.inf
            return _nb_terms(rows, y, offsets, parameters)[0]

        outcome = newton_maximize(
            objective,
            lambda q: _nb_terms(rows, y, offsets, q)[1],
            start,
            hessian=lambda q: _nb_terms(rows, y, offsets, q)[2],
            max_iterations=max_iterations,
        )
        theta = math.exp(outcome.x[p])
        if theta < _THETA_CEILING:
            beta = outcome.x[:p]
            alpha = 1.0 / theta
            eta = _linear(rows, beta, offsets)
            weights = tuple(math.exp(e) / (1.0 + alpha * math.exp(e)) for e in eta)
            information = tuple(
                tuple(
                    -math.fsum(
                        w * row[a] * row[b]
                        for w, row in zip(weights, rows, strict=True)
                    )
                    for b in range(p)
                )
                for a in range(p)
            )
            model_cov = inverse_information(information)
            h_theta = math.fsum(
                -_trigamma_difference(yi, theta)
                + 1.0 / theta
                - 1.0 / (theta + math.exp(ei))
                + (yi - math.exp(ei)) / (theta + math.exp(ei)) ** 2
                for yi, ei in zip(y, eta, strict=True)
            )
            theta_se = math.sqrt(-1.0 / h_theta) if h_theta < 0.0 else math.nan
            return _build(
                "negative-binomial",
                rows,
                names,
                y,
                offsets,
                beta,
                theta,
                theta_se,
                model_cov,
                covariance,
                False,
                outcome.iterations,
                outcome.converged,
            )
    return _build(
        "negative-binomial",
        rows,
        names,
        y,
        offsets,
        poisson.coefficients,
        math.inf,
        math.nan,
        poisson.unscaled_covariance,
        covariance,
        True,
        poisson.iterations,
        poisson.converged,
    )


@dataclass(frozen=True, slots=True)
class OverdispersionTest:
    """Cameron-Trivedi regression test of Poisson equidispersion.

    Attributes:
        form: ``"nb2"`` (``Var = mu + alpha mu^2``) or ``"nb1"``
            (``Var = (1 + alpha) mu``).
        alpha: Least-squares estimate of ``alpha`` from the auxiliary
            regression.
        standard_error: Its ordinary least-squares standard error.
        statistic: ``alpha / se``, asymptotically ``N(0, 1)`` under the
            Poisson null.
        p_value: Normal p-value for ``alternative``.
        alternative: ``"greater"`` (overdispersion) or ``"two-sided"``.
        n_observations: ``n``.

    References:
        [EQ-CNT-REG-004], [REF-CAMERON-TRIVEDI-1990].

    """

    form: str
    alpha: float
    standard_error: float
    statistic: float
    p_value: float
    alternative: str
    n_observations: int


def poisson_overdispersion_test(
    result: CountRegressionResult,
    *,
    form: DispersionForm = "nb2",
    alternative: Literal["greater", "two-sided"] = "greater",
) -> OverdispersionTest:
    """Test ``alpha = 0`` in ``Var(y) = mu + alpha g(mu)`` after a Poisson fit.

    The auxiliary regression without intercept is
    ``[(y_i - mu_i)^2 - y_i] / mu_i = alpha g(mu_i) / mu_i + e_i`` with
    ``g(mu) = mu^2`` (NB2, regressor ``mu_i``) or ``g(mu) = mu`` (NB1,
    regressor ``1``). The statistic is the least-squares t ratio
    ``alpha_hat / se`` with ``se^2 = s^2 / sum z_i^2`` and
    ``s^2 = RSS / (n - 1)``, referred to the standard normal law.

    Args:
        result: A :class:`CountRegressionResult` with ``model == "poisson"``.
        form: ``"nb2"`` or ``"nb1"``.
        alternative: ``"greater"`` or ``"two-sided"``.

    Returns:
        :class:`OverdispersionTest`.

    Raises:
        TypeError: If ``result`` is not a count regression result.
        ValueError: If the fit is not Poisson or an option is invalid.

    Equation:
        [EQ-CNT-REG-004].

    References:
        [REF-CAMERON-TRIVEDI-1990], [REF-CAMERON-TRIVEDI-2013].

    """
    if not isinstance(result, CountRegressionResult):
        raise TypeError("result must be a CountRegressionResult")
    if result.model != "poisson":
        raise ValueError("the overdispersion test needs a Poisson fit")
    if form not in ("nb2", "nb1"):
        raise ValueError("form must be 'nb2' or 'nb1'")
    if alternative not in ("greater", "two-sided"):
        raise ValueError("alternative must be 'greater' or 'two-sided'")
    y, mu = result.response, result.fitted_values
    auxiliary = tuple(((yi - mi) ** 2 - yi) / mi for yi, mi in zip(y, mu, strict=True))
    regressor = mu if form == "nb2" else tuple(1.0 for _ in mu)
    sxx = math.fsum(z * z for z in regressor)
    alpha = math.fsum(z * a for z, a in zip(regressor, auxiliary, strict=True)) / sxx
    n = len(y)
    rss = math.fsum(
        (a - alpha * z) ** 2 for z, a in zip(regressor, auxiliary, strict=True)
    )
    se = math.sqrt(rss / (n - 1) / sxx)
    statistic = alpha / se if se > 0.0 else math.copysign(math.inf, alpha)
    if alternative == "greater":
        p_value = standard_normal_sf(statistic)
    else:
        p_value = 2.0 * standard_normal_sf(abs(statistic))
    return OverdispersionTest(
        form=form,
        alpha=alpha,
        standard_error=se,
        statistic=statistic,
        p_value=p_value,
        alternative=alternative,
        n_observations=n,
    )
