"""Ordinal (proportional-odds) and multinomial (baseline-category) logit models.

The proportional-odds model ``logit P(Y <= j | x) = theta_j - x' beta`` has
one threshold per cut and one slope vector shared by every cut
[EQ-CAT-ORD-001]; :func:`brant_test` checks the shared-slope assumption by
comparing separate binary logits of ``1{Y > j}`` [EQ-CAT-BRANT-001]. The
baseline-category logit ``log(P(Y = k)/P(Y = r)) = x' beta_k`` has one
coefficient vector per non-reference category [EQ-CAT-MNL-001]. Both are
fitted by Newton-Raphson with analytic Hessians; predicted probabilities and
average marginal effects with delta-method standard errors are available on
the results [EQ-REG-AME-001].
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.linalg import Matrix, Vector, as_matrix, inverse_spd

from ._ml_tools import (
    MarginalEffects,
    delta_method_effects,
    inverse_information,
    logistic,
    newton_maximize,
    normal_intervals,
    resolve_column,
    set_column,
    wald_table,
)
from .fitting import information_criteria
from .glm import GLMConvergenceWarning, SeparationError, fit_glm
from .linear_models import DesignMatrix, _coerce_design, _has_ones_column
from .sampling_distributions import chi_square_sf


def _encode_categories(
    response: Iterable[object], levels: Sequence[object] | None
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Return integer codes ``0..J-1`` and the level labels.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    values = tuple(response)
    if levels is None:
        try:
            ordered = sorted(set(values))  # type: ignore[type-var]
        except TypeError as error:
            raise TypeError("pass levels explicitly for unorderable labels") from error
    else:
        ordered = list(levels)
        if len(set(ordered)) != len(ordered):
            raise ValueError("levels must be distinct")
    lookup = {value: index for index, value in enumerate(ordered)}
    codes: list[int] = []
    for index, value in enumerate(values):
        if value not in lookup:
            raise ValueError(f"response[{index}] = {value!r} is not a listed level")
        codes.append(lookup[value])
    if len(ordered) < 2:
        raise ValueError("at least two response categories are required")
    missing = [str(ordered[k]) for k in range(len(ordered)) if k not in set(codes)]
    if missing:
        raise ValueError(f"levels without observations: {missing}")
    return tuple(codes), tuple(str(level) for level in ordered)


_CERTAIN = 1.0 - 1e-10


def _check_separation(
    probabilities: tuple[Vector, ...], codes: tuple[int, ...]
) -> None:
    """Raise when a fitted probability of an observed category is numerically one.

    An existing maximum-likelihood estimate keeps every fitted probability
    of an observed category below one; a value above ``1 - 1e-10`` (a logit
    beyond about 23) arises when Newton follows a direction of complete or
    quasi-complete separation, along which the likelihood keeps increasing.

    References:
        [REF-ALBERT-ANDERSON-1984], [EQ-GLM-SEP-001].

    """
    certain = [
        i
        for i, (row, code) in enumerate(zip(probabilities, codes, strict=True))
        if row[code] > _CERTAIN
    ]
    if certain:
        raise SeparationError(
            "maximum-likelihood estimates do not exist: complete or "
            "quasi-complete separation (fitted probabilities numerically one "
            f"for {len(certain)} observation(s): {certain[:10]})"
        )


def _logistic_density(value: float) -> float:
    """Return ``F(t)(1 - F(t))`` for the logistic law (``0`` at ``+-inf``).

    References:
        [REF-AGRESTI-2013].

    """
    if math.isinf(value):
        return 0.0
    exponential = math.exp(-abs(value))
    return exponential / (1.0 + exponential) ** 2


def _interval_probability(upper: float, lower: float) -> float:
    """Return ``F(upper) - F(lower)`` without cancellation in either tail.

    References:
        [EQ-CAT-ORD-001].

    """
    if upper + lower > 0.0:
        return logistic(-lower) - logistic(-upper)
    return logistic(upper) - logistic(lower)


def _ordinal_cuts(
    thresholds: Sequence[float], category: int, eta: float
) -> tuple[float, float]:
    """Return ``(theta_k - eta, theta_(k-1) - eta)`` with infinite end cuts.

    References:
        [EQ-CAT-ORD-001].

    """
    upper = thresholds[category] - eta if category < len(thresholds) else math.inf
    lower = thresholds[category - 1] - eta if category > 0 else -math.inf
    return upper, lower


def _ordinal_terms(
    rows: Matrix, codes: tuple[int, ...], parameters: Vector, p: int
) -> tuple[float, Vector, Matrix]:
    """Return log-likelihood, gradient, and Hessian of the proportional-odds model.

    With ``a = theta_k - eta``, ``b = theta_(k-1) - eta``, ``P = F(a) - F(b)``:
    ``l_a = f(a)/P``, ``l_b = -f(b)/P``, ``l_aa = f'(a)/P - l_a^2``,
    ``l_bb = -f'(b)/P - l_b^2``, ``l_ab = -l_a l_b``, ``f' = f(1 - 2F)``;
    the parameter Jacobians are ``u_a = e_(theta_k) - x`` and
    ``u_b = e_(theta_(k-1)) - x``.

    References:
        [EQ-CAT-ORD-001], [REF-MCCULLAGH-1980], [REF-AGRESTI-2013].

    """
    beta, thresholds = parameters[:p], parameters[p:]
    size = len(parameters)
    if any(b <= a for a, b in zip(thresholds, thresholds[1:], strict=False)):
        return -math.inf, (0.0,) * size, tuple((0.0,) * size for _ in range(size))
    gradient = [0.0] * size
    hessian = [[0.0] * size for _ in range(size)]
    terms: list[float] = []
    for row, k in zip(rows, codes, strict=True):
        eta = math.fsum(b * x for b, x in zip(beta, row, strict=True))
        upper, lower = _ordinal_cuts(thresholds, k, eta)
        probability = _interval_probability(upper, lower)
        if probability <= 0.0:
            return -math.inf, tuple(gradient), tuple(tuple(r) for r in hessian)
        terms.append(math.log(probability))
        fa, fb = _logistic_density(upper), _logistic_density(lower)
        da = 0.0 if math.isinf(upper) else fa * (1.0 - 2.0 * logistic(upper))
        db = 0.0 if math.isinf(lower) else fb * (1.0 - 2.0 * logistic(lower))
        la, lb = fa / probability, -fb / probability
        laa = da / probability - la * la
        lbb = -db / probability - lb * lb
        lab = -la * lb
        ua = [0.0] * size
        ub = [0.0] * size
        for j in range(p):
            ua[j] = -row[j]
            ub[j] = -row[j]
        if k < len(thresholds):
            ua[p + k] = 1.0
        if k > 0:
            ub[p + k - 1] = 1.0
        for i in range(size):
            gradient[i] += la * ua[i] + lb * ub[i]
            for j in range(i + 1):
                hessian[i][j] += (
                    laa * ua[i] * ua[j]
                    + lbb * ub[i] * ub[j]
                    + lab * (ua[i] * ub[j] + ub[i] * ua[j])
                )
    for i in range(size):
        for j in range(i):
            hessian[j][i] = hessian[i][j]
    return math.fsum(terms), tuple(gradient), tuple(tuple(r) for r in hessian)


def _ordinal_probabilities(
    rows: Matrix, beta: Sequence[float], thresholds: Sequence[float]
) -> tuple[Vector, ...]:
    """Return ``P(Y = k | x_i)`` for every row and category.

    References:
        [EQ-CAT-ORD-001].

    """
    result: list[Vector] = []
    for row in rows:
        eta = math.fsum(b * x for b, x in zip(beta, row, strict=True))
        result.append(
            tuple(
                _interval_probability(*_ordinal_cuts(thresholds, k, eta))
                for k in range(len(thresholds) + 1)
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class OrdinalRegressionResult:
    """Proportional-odds (cumulative logit) fit.

    Attributes:
        coefficients: Slopes ``beta``; a positive slope shifts the response
            towards higher categories.
        thresholds: Cut points ``theta_1 < ... < theta_(J-1)``.
        standard_errors: Standard errors of ``(beta, theta)``.
        statistics: Wald ratios of ``(beta, theta)``.
        p_values: Two-sided normal p-values of ``(beta, theta)``.
        column_names: Design-column names.
        levels: Ordered category labels.
        covariance: Inverse observed information of ``(beta, theta)``.
        log_likelihood: Maximized log-likelihood.
        aic: ``-2 log L + 2k``.
        bic: ``-2 log L + k log n``.
        parameter_count: ``p + J - 1``.
        iterations: Newton iterations.
        converged: Whether Newton converged.
        n_observations: ``n``.
        design_rows: Design rows (no intercept).
        codes: Category codes ``0..J-1`` of the responses.

    References:
        [EQ-CAT-ORD-001], [REF-MCCULLAGH-1980].

    """

    coefficients: Vector
    thresholds: Vector
    standard_errors: Vector
    statistics: Vector
    p_values: Vector
    column_names: tuple[str, ...]
    levels: tuple[str, ...]
    covariance: Matrix
    log_likelihood: float
    aic: float
    bic: float
    parameter_count: int
    iterations: int
    converged: bool
    n_observations: int
    design_rows: Matrix
    codes: tuple[int, ...]

    def odds_ratios(
        self, level: float = 0.95
    ) -> tuple[tuple[float, float, float], ...]:
        """Return cumulative odds ratios ``exp(beta_j)`` with Wald limits.

        ``exp(beta_j)`` multiplies the odds of ``Y > k`` for every cut ``k``
        per unit increase of column ``j``.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            ``(ratio, lower, upper)`` per slope.

        Raises:
            ValueError: If ``level`` is invalid.

        Equation:
            [EQ-CAT-ORD-001].

        References:
            [REF-AGRESTI-2013].

        """
        p = len(self.coefficients)
        intervals = normal_intervals(self.coefficients, self.standard_errors[:p], level)
        return tuple(
            (math.exp(b), math.exp(lo), math.exp(hi))
            for b, (lo, hi) in zip(self.coefficients, intervals, strict=True)
        )

    def predict_proba(
        self, rows: Iterable[Iterable[object]] | None = None
    ) -> tuple[Vector, ...]:
        """Return category probabilities at new rows (the fitting rows by default).

        Args:
            rows: Design rows without intercept; ``None`` uses the data.

        Returns:
            One probability vector per row, in level order.

        Raises:
            ValueError: If the rows have the wrong width.

        Equation:
            [EQ-CAT-ORD-001].

        References:
            [REF-MCCULLAGH-1980].

        """
        checked = self.design_rows if rows is None else as_matrix(rows, name="rows")
        if len(checked[0]) != len(self.coefficients):
            raise ValueError("rows must have one entry per design column")
        return _ordinal_probabilities(checked, self.coefficients, self.thresholds)

    def average_marginal_effects(
        self, column: str | int, *, discrete: bool = False
    ) -> MarginalEffects:
        """Return average marginal effects of a column on every category probability.

        The derivative effect of column ``j`` on ``P(Y = k)`` is
        ``-beta_j [f(theta_k - eta) - f(theta_(k-1) - eta)]`` averaged over
        the rows; the discrete effect averages ``P_k(x_j = 1) - P_k(x_j = 0)``.
        The effects sum to zero over categories.

        Args:
            column: Column name or index.
            discrete: Use the discrete change from 0 to 1.

        Returns:
            :class:`MarginalEffects`, one effect per category.

        Raises:
            KeyError: If the column is unknown.
            ValueError: If the column is constant.

        Equation:
            [EQ-REG-AME-001], [EQ-CAT-ORD-001].

        References:
            [REF-AGRESTI-2013].

        """
        index = resolve_column(self.column_names, column)
        rows = self.design_rows
        if len({row[index] for row in rows}) == 1:
            raise ValueError("marginal effects of a constant column are undefined")
        p, levels = len(self.coefficients), len(self.levels)

        def effect(parameters: Vector) -> Vector:
            """Return the AMEs as a function of ``(beta, theta)``.

            References:
                [EQ-REG-AME-001].

            """
            beta, thresholds = parameters[:p], parameters[p:]
            if discrete:
                high = _ordinal_probabilities(
                    set_column(rows, index, 1.0), beta, thresholds
                )
                low = _ordinal_probabilities(
                    set_column(rows, index, 0.0), beta, thresholds
                )
                return tuple(
                    math.fsum(h[k] - lo[k] for h, lo in zip(high, low, strict=True))
                    / len(rows)
                    for k in range(levels)
                )
            sums = [0.0] * levels
            for row in rows:
                eta = math.fsum(b * x for b, x in zip(beta, row, strict=True))
                for k in range(levels):
                    upper, lower = _ordinal_cuts(thresholds, k, eta)
                    sums[k] += -beta[index] * (
                        _logistic_density(upper) - _logistic_density(lower)
                    )
            return tuple(value / len(rows) for value in sums)

        return delta_method_effects(
            effect,
            (*self.coefficients, *self.thresholds),
            self.covariance,
            column=self.column_names[index],
            kind="discrete" if discrete else "derivative",
            outcomes=self.levels,
        )


def fit_ordinal_logistic(
    design: DesignMatrix | Iterable[Iterable[object]] | None,
    response: Iterable[object],
    *,
    levels: Sequence[object] | None = None,
    column_names: Sequence[str] | None = None,
    max_iterations: int = 100,
) -> OrdinalRegressionResult:
    """Fit the proportional-odds model ``logit P(Y <= k) = theta_k - x' beta``.

    The thresholds absorb the intercept, so the design must not contain a
    constant column. Newton-Raphson with the analytic Hessian starts from
    ``beta = 0`` and the empirical cumulative logits, and keeps the
    thresholds strictly increasing by step halving.

    Args:
        design: Covariate rows without intercept, a :class:`DesignMatrix`
            built with ``intercept=False``, or ``None`` for a
            thresholds-only model.
        response: Ordered category labels.
        levels: Category order; defaults to sorted unique labels. Every
            level must be observed.
        column_names: Names for a raw design.
        max_iterations: Newton iteration limit.

    Returns:
        :class:`OrdinalRegressionResult`.

    Raises:
        SeparationError: If a fitted probability of an observed category is
            numerically one (the estimates do not exist).
        ValueError: If the design has a constant column, the shapes differ,
            or a level is unobserved.
        ArithmeticError: If the information matrix is singular.

    Warns:
        GLMConvergenceWarning: If Newton does not converge (for example under
            separation).

    Equation:
        [EQ-CAT-ORD-001].

    References:
        [REF-MCCULLAGH-1980], [REF-AGRESTI-2013], [REF-ALBERT-ANDERSON-1984].

    """
    codes, labels = _encode_categories(response, levels)
    n = len(codes)
    if design is None:
        rows: Matrix = tuple(() for _ in range(n))
        names: tuple[str, ...] = ()
    else:
        rows, names, _ = _coerce_design(design, column_names)
        if len(rows) != n:
            raise ValueError("response length must equal the number of design rows")
        if _has_ones_column(rows) or any(
            len({row[j] for row in rows}) == 1 for j in range(len(rows[0]))
        ):
            raise ValueError(
                "the thresholds absorb the intercept; remove constant design columns"
            )
    p = len(names)
    cumulative = 0
    start_thresholds: list[float] = []
    for k in range(len(labels) - 1):
        cumulative += sum(1 for c in codes if c == k)
        share = cumulative / n
        start_thresholds.append(math.log(share / (1.0 - share)))
    start = (*(0.0,) * p, *start_thresholds)
    outcome = newton_maximize(
        lambda q: _ordinal_terms(rows, codes, q, p)[0],
        lambda q: _ordinal_terms(rows, codes, q, p)[1],
        start,
        hessian=lambda q: _ordinal_terms(rows, codes, q, p)[2],
        max_iterations=max_iterations,
    )
    if not outcome.converged:
        warnings.warn(
            "proportional-odds Newton iterations did not converge "
            "(check for separation)",
            GLMConvergenceWarning,
            stacklevel=2,
        )
    _check_separation(_ordinal_probabilities(rows, outcome.x[:p], outcome.x[p:]), codes)
    covariance = inverse_information(outcome.hessian)
    errors, ratios, p_values = wald_table(outcome.x, covariance)
    k = len(outcome.x)
    aic, _, bic = information_criteria(outcome.value, k, n)
    return OrdinalRegressionResult(
        coefficients=outcome.x[:p],
        thresholds=outcome.x[p:],
        standard_errors=errors,
        statistics=ratios,
        p_values=p_values,
        column_names=names,
        levels=labels,
        covariance=covariance,
        log_likelihood=outcome.value,
        aic=aic,
        bic=bic,
        parameter_count=k,
        iterations=outcome.iterations,
        converged=outcome.converged,
        n_observations=n,
        design_rows=rows,
        codes=codes,
    )


@dataclass(frozen=True, slots=True)
class BrantTest:
    """Brant's Wald test of the proportional-odds assumption.

    Attributes:
        statistic: Omnibus ``X^2`` for equal slopes across the ``J - 1``
            binary logits.
        degrees_of_freedom: ``(J - 2) p``.
        p_value: Chi-square upper tail.
        variables: ``(name, X^2, df, p)`` per design column (``df = J - 2``).
        binary_slopes: Slopes of each binary logit ``1{Y > k}``, one tuple
            per cut.

    References:
        [EQ-CAT-BRANT-001], [REF-BRANT-1990].

    """

    statistic: float
    degrees_of_freedom: int
    p_value: float
    variables: tuple[tuple[str, float, int, float], ...]
    binary_slopes: tuple[Vector, ...]


def _quadratic_test(estimate: Vector, covariance: Matrix) -> float:
    """Return ``d' V^-1 d``.

    References:
        [EQ-CAT-BRANT-001].

    """
    inverse = inverse_spd(covariance)
    size = len(estimate)
    return math.fsum(
        estimate[i] * inverse[i][j] * estimate[j]
        for i in range(size)
        for j in range(size)
    )


def brant_test(result: OrdinalRegressionResult) -> BrantTest:
    """Test the shared-slope assumption of a proportional-odds fit.

    For each cut ``k`` a binary logit of ``1{Y > k}`` on ``(1, x)`` gives
    ``beta_k``. Their joint covariance is
    ``Cov(beta_k, beta_l) = (X'W_kk X)^-1 X'W_kl X (X'W_ll X)^-1`` with
    ``W_kl = diag(pi_(i, max(k,l)) - pi_ik pi_il)`` and ``pi_ik`` the fitted
    ``P(Y > k | x_i)``. The statistic is the Wald form of
    ``beta_1 = beta_k`` for ``k = 2..J-1`` on the slopes (intercepts
    excluded), with ``(J - 2) p`` degrees of freedom; one test per column
    uses that column's slopes alone.

    Args:
        result: A fit from :func:`fit_ordinal_logistic` with at least three
            categories and one covariate.

    Returns:
        :class:`BrantTest`.

    Raises:
        TypeError: If ``result`` is not an ordinal fit.
        ValueError: If there are fewer than three categories or no
            covariates.
        SeparationError: If a binary logit is separated.

    Equation:
        [EQ-CAT-BRANT-001].

    References:
        [REF-BRANT-1990], [REF-AGRESTI-2013].

    """
    if not isinstance(result, OrdinalRegressionResult):
        raise TypeError("result must be an OrdinalRegressionResult")
    cuts = len(result.thresholds)
    p = len(result.coefficients)
    if cuts < 2 or p == 0:
        raise ValueError(
            "the Brant test needs three or more categories and a covariate"
        )
    design = tuple((1.0, *row) for row in result.design_rows)
    width = p + 1
    fits = [
        fit_glm(
            design,
            tuple(1.0 if c > k else 0.0 for c in result.codes),
            family="binomial",
        )
        for k in range(cuts)
    ]
    probabilities = [fit.fitted_values for fit in fits]

    def cross(k: int, m: int) -> Matrix:
        """Return ``X' W_km X``.

        References:
            [EQ-CAT-BRANT-001].

        """
        top = max(k, m)
        weights = tuple(
            pt - pk * pm
            for pt, pk, pm in zip(
                probabilities[top], probabilities[k], probabilities[m], strict=True
            )
        )
        return tuple(
            tuple(
                math.fsum(w * x[a] * x[b] for w, x in zip(weights, design, strict=True))
                for b in range(width)
            )
            for a in range(width)
        )

    bread = [fit.unscaled_covariance for fit in fits]

    def block(k: int, m: int) -> Matrix:
        """Return the slope block of ``Cov(beta_k, beta_m)``.

        References:
            [EQ-CAT-BRANT-001].

        """
        middle = cross(k, m)
        left = tuple(
            tuple(
                math.fsum(bread[k][a][c] * middle[c][b] for c in range(width))
                for b in range(width)
            )
            for a in range(width)
        )
        full = tuple(
            tuple(
                math.fsum(left[a][c] * bread[m][c][b] for c in range(width))
                for b in range(width)
            )
            for a in range(width)
        )
        return tuple(
            tuple(full[a][b] for b in range(1, width)) for a in range(1, width)
        )

    blocks = {(k, m): block(k, m) for k in range(cuts) for m in range(cuts)}
    slopes = tuple(fit.coefficients[1:] for fit in fits)

    def statistic(columns: Sequence[int]) -> float:
        """Return the Wald statistic for equal slopes of the given columns.

        References:
            [EQ-CAT-BRANT-001].

        """
        pairs = [(k, j) for k in range(1, cuts) for j in columns]
        contrast = tuple(slopes[0][j] - slopes[k][j] for k, j in pairs)

        def cov(k: int, a: int, m: int, b: int) -> float:
            """Return ``Cov(beta_k[a], beta_m[b])``.

            References:
                [EQ-CAT-BRANT-001].

            """
            return blocks[(k, m)][a][b]

        matrix = tuple(
            tuple(
                cov(0, a, 0, b) - cov(0, a, m, b) - cov(k, a, 0, b) + cov(k, a, m, b)
                for m, b in pairs
            )
            for k, a in pairs
        )
        return _quadratic_test(contrast, matrix)

    omnibus = statistic(range(p))
    df = (cuts - 1) * p
    variables = tuple(
        (
            result.column_names[j],
            value := statistic((j,)),
            cuts - 1,
            chi_square_sf(value, cuts - 1),
        )
        for j in range(p)
    )
    return BrantTest(
        statistic=omnibus,
        degrees_of_freedom=df,
        p_value=chi_square_sf(omnibus, df),
        variables=variables,
        binary_slopes=slopes,
    )


def _multinomial_terms(
    rows: Matrix, codes: tuple[int, ...], parameters: Vector, classes: int
) -> tuple[float, Vector, Matrix]:
    """Return log-likelihood, gradient, and Hessian of the baseline-category logit.

    Parameters are ``beta_1, ..., beta_(K-1)`` stacked (code ``0`` is the
    reference). ``dl/d beta_k = sum_i (y_ik - p_ik) x_i`` and
    ``d^2 l / d beta_k d beta_l' = -sum_i p_ik (delta_kl - p_il) x_i x_i'``.

    References:
        [EQ-CAT-MNL-001], [REF-AGRESTI-2013].

    """
    p = len(rows[0])
    size = len(parameters)
    gradient = [0.0] * size
    hessian = [[0.0] * size for _ in range(size)]
    terms: list[float] = []
    for row, code in zip(rows, codes, strict=True):
        etas = [0.0] + [
            math.fsum(parameters[(k - 1) * p + j] * row[j] for j in range(p))
            for k in range(1, classes)
        ]
        top = max(etas)
        normalizer = top + math.log(math.fsum(math.exp(e - top) for e in etas))
        terms.append(etas[code] - normalizer)
        probabilities = [math.exp(e - normalizer) for e in etas]
        for k in range(1, classes):
            residual = (1.0 if code == k else 0.0) - probabilities[k]
            for j in range(p):
                gradient[(k - 1) * p + j] += residual * row[j]
            for m in range(1, k + 1):
                weight = probabilities[k] * (
                    (1.0 if k == m else 0.0) - probabilities[m]
                )
                for a in range(p):
                    for b in range(p):
                        hessian[(k - 1) * p + a][(m - 1) * p + b] -= (
                            weight * row[a] * row[b]
                        )
    for i in range(size):
        for j in range(i + 1, size):
            hessian[i][j] = hessian[j][i]
    return math.fsum(terms), tuple(gradient), tuple(tuple(r) for r in hessian)


def _multinomial_probabilities(
    rows: Matrix, parameters: Sequence[float], classes: int
) -> tuple[Vector, ...]:
    """Return ``P(Y = k | x_i)`` for every row (reference code ``0`` first).

    References:
        [EQ-CAT-MNL-001].

    """
    p = len(rows[0])
    result: list[Vector] = []
    for row in rows:
        etas = [0.0] + [
            math.fsum(parameters[(k - 1) * p + j] * row[j] for j in range(p))
            for k in range(1, classes)
        ]
        top = max(etas)
        exps = [math.exp(e - top) for e in etas]
        total = math.fsum(exps)
        result.append(tuple(e / total for e in exps))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class MultinomialLogitResult:
    """Baseline-category multinomial logit fit.

    Attributes:
        coefficients: One coefficient tuple per non-reference level, in
            ``levels[1:]`` order (log relative risks against the reference).
        standard_errors: Standard errors with the same shape.
        p_values: Two-sided normal p-values with the same shape.
        column_names: Design-column names.
        levels: Category labels; ``levels[0]`` is the reference.
        covariance: Inverse information of the stacked coefficients.
        log_likelihood: Maximized log-likelihood.
        deviance: ``-2 log L``.
        aic: ``-2 log L + 2k``.
        bic: ``-2 log L + k log n``.
        parameter_count: ``(K - 1) p``.
        iterations: Newton iterations.
        converged: Whether Newton converged.
        n_observations: ``n``.
        design_rows: Design rows.
        codes: Category codes (``0`` is the reference).

    References:
        [EQ-CAT-MNL-001], [REF-AGRESTI-2013].

    """

    coefficients: tuple[Vector, ...]
    standard_errors: tuple[Vector, ...]
    p_values: tuple[Vector, ...]
    column_names: tuple[str, ...]
    levels: tuple[str, ...]
    covariance: Matrix
    log_likelihood: float
    deviance: float
    aic: float
    bic: float
    parameter_count: int
    iterations: int
    converged: bool
    n_observations: int
    design_rows: Matrix
    codes: tuple[int, ...]

    def _stacked(self) -> Vector:
        """Return the coefficients stacked by category.

        References:
            [EQ-CAT-MNL-001]. No equation is evaluated.

        """
        return tuple(value for block in self.coefficients for value in block)

    def relative_risk_ratios(
        self, level: float = 0.95
    ) -> tuple[tuple[tuple[float, float, float], ...], ...]:
        """Return ``exp(beta_kj)`` with exponentiated Wald limits.

        ``exp(beta_kj)`` multiplies ``P(Y = k)/P(Y = reference)`` per unit
        increase of column ``j``.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            ``(ratio, lower, upper)`` per category and column.

        Raises:
            ValueError: If ``level`` is invalid.

        Equation:
            [EQ-CAT-MNL-001].

        References:
            [REF-AGRESTI-2013].

        """
        return tuple(
            tuple(
                (math.exp(b), math.exp(lo), math.exp(hi))
                for b, (lo, hi) in zip(
                    block, normal_intervals(block, errors, level), strict=True
                )
            )
            for block, errors in zip(
                self.coefficients, self.standard_errors, strict=True
            )
        )

    def predict_proba(
        self, rows: Iterable[Iterable[object]] | None = None
    ) -> tuple[Vector, ...]:
        """Return category probabilities at new rows (the fitting rows by default).

        Args:
            rows: Design rows; ``None`` uses the data.

        Returns:
            One probability vector per row, in level order.

        Raises:
            ValueError: If the rows have the wrong width.

        Equation:
            [EQ-CAT-MNL-001].

        References:
            [REF-AGRESTI-2013].

        """
        checked = self.design_rows if rows is None else as_matrix(rows, name="rows")
        if len(checked[0]) != len(self.column_names):
            raise ValueError("rows must have one entry per design column")
        return _multinomial_probabilities(checked, self._stacked(), len(self.levels))

    def average_marginal_effects(
        self, column: str | int, *, discrete: bool = False
    ) -> MarginalEffects:
        """Return average marginal effects of a column on every category probability.

        The derivative effect on ``P(Y = k)`` is
        ``p_k (beta_kj - sum_l p_l beta_lj)`` averaged over the rows
        (``beta_0 = 0``); the discrete effect averages
        ``P_k(x_j = 1) - P_k(x_j = 0)``.

        Args:
            column: Column name or index (not a constant column).
            discrete: Use the discrete change from 0 to 1.

        Returns:
            :class:`MarginalEffects`, one effect per category.

        Raises:
            KeyError: If the column is unknown.
            ValueError: If the column is constant.

        Equation:
            [EQ-REG-AME-001], [EQ-CAT-MNL-001].

        References:
            [REF-AGRESTI-2013].

        """
        index = resolve_column(self.column_names, column)
        rows = self.design_rows
        if len({row[index] for row in rows}) == 1:
            raise ValueError("marginal effects of a constant column are undefined")
        classes, p = len(self.levels), len(self.column_names)

        def effect(parameters: Vector) -> Vector:
            """Return the AMEs as a function of the stacked coefficients.

            References:
                [EQ-REG-AME-001].

            """
            if discrete:
                high = _multinomial_probabilities(
                    set_column(rows, index, 1.0), parameters, classes
                )
                low = _multinomial_probabilities(
                    set_column(rows, index, 0.0), parameters, classes
                )
                return tuple(
                    math.fsum(h[k] - lo[k] for h, lo in zip(high, low, strict=True))
                    / len(rows)
                    for k in range(classes)
                )
            slopes = [0.0] + [
                parameters[(k - 1) * p + index] for k in range(1, classes)
            ]
            sums = [0.0] * classes
            for probs in _multinomial_probabilities(rows, parameters, classes):
                average = math.fsum(
                    pk * bk for pk, bk in zip(probs, slopes, strict=True)
                )
                for k in range(classes):
                    sums[k] += probs[k] * (slopes[k] - average)
            return tuple(value / len(rows) for value in sums)

        return delta_method_effects(
            effect,
            self._stacked(),
            self.covariance,
            column=self.column_names[index],
            kind="discrete" if discrete else "derivative",
            outcomes=self.levels,
        )


def fit_multinomial_logit(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    levels: Sequence[object] | None = None,
    reference: object | None = None,
    column_names: Sequence[str] | None = None,
    max_iterations: int = 100,
) -> MultinomialLogitResult:
    """Fit the baseline-category logit ``log(p_k / p_ref) = x' beta_k`` by Newton.

    Args:
        design: :class:`DesignMatrix` or rows (include an intercept column).
        response: Category labels.
        levels: Category order; defaults to sorted unique labels.
        reference: Reference category; defaults to the first level.
        column_names: Names for a raw design.
        max_iterations: Newton iteration limit.

    Returns:
        :class:`MultinomialLogitResult`; ``levels[0]`` is the reference.

    Raises:
        SeparationError: If a fitted probability of an observed category is
            numerically one (the estimates do not exist).
        ValueError: If the inputs are invalid or a level is unobserved.
        ArithmeticError: If the information matrix is singular (for example
            a rank-deficient design).

    Warns:
        GLMConvergenceWarning: If Newton does not converge (for example under
            separation, where the estimates do not exist).

    Equation:
        [EQ-CAT-MNL-001].

    References:
        [REF-AGRESTI-2013].

    """
    codes, labels = _encode_categories(response, levels)
    if reference is not None:
        if str(reference) not in labels:
            raise ValueError(f"reference {reference!r} is not a level")
        position = labels.index(str(reference))
        order = [position] + [k for k in range(len(labels)) if k != position]
        remap = {old: new for new, old in enumerate(order)}
        codes = tuple(remap[c] for c in codes)
        labels = tuple(labels[k] for k in order)
    rows, names, _ = _coerce_design(design, column_names)
    n, p = len(rows), len(rows[0])
    if len(codes) != n:
        raise ValueError("response length must equal the number of design rows")
    classes = len(labels)
    if n <= (classes - 1) * p:
        raise ValueError("more observations than coefficients are required")
    start = (0.0,) * ((classes - 1) * p)
    outcome = newton_maximize(
        lambda q: _multinomial_terms(rows, codes, q, classes)[0],
        lambda q: _multinomial_terms(rows, codes, q, classes)[1],
        start,
        hessian=lambda q: _multinomial_terms(rows, codes, q, classes)[2],
        max_iterations=max_iterations,
    )
    if not outcome.converged:
        warnings.warn(
            "multinomial Newton iterations did not converge (check for separation)",
            GLMConvergenceWarning,
            stacklevel=2,
        )
    _check_separation(_multinomial_probabilities(rows, outcome.x, classes), codes)
    covariance = inverse_information(outcome.hessian)
    errors, _, p_values = wald_table(outcome.x, covariance)

    def shaped(values: Vector) -> tuple[Vector, ...]:
        """Return values reshaped to one tuple per non-reference category.

        References:
            [EQ-CAT-MNL-001]. No equation is evaluated.

        """
        return tuple(values[k * p : (k + 1) * p] for k in range(classes - 1))

    k = len(outcome.x)
    aic, _, bic = information_criteria(outcome.value, k, n)
    return MultinomialLogitResult(
        coefficients=shaped(outcome.x),
        standard_errors=shaped(errors),
        p_values=shaped(p_values),
        column_names=names,
        levels=labels,
        covariance=covariance,
        log_likelihood=outcome.value,
        deviance=-2.0 * outcome.value,
        aic=aic,
        bic=bic,
        parameter_count=k,
        iterations=outcome.iterations,
        converged=outcome.converged,
        n_observations=n,
        design_rows=rows,
        codes=codes,
    )
