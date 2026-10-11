"""Missing-data methods for multivariate normal data.

Rows are equal-length sequences in which a missing entry is ``None`` or
``math.nan``. The module offers

* a **missingness summary** (:func:`missing_pattern`) and **Little's MCAR
  test** (:func:`little_mcar_test`);
* **EM for the multivariate normal** under missingness at random
  (:func:`fit_incomplete_multivariate_normal`): the E-step replaces each
  missing block by its conditional mean given the observed block (Schur
  complement), adds the conditional covariance to the cross-products, and the
  M-step is the complete-data MLE. The exact observed-data log-likelihood
  (:func:`incomplete_normal_log_likelihood`) drives the generic
  :func:`~probstat.inference.expectation_maximization` monotonicity check, and
  standard errors of the mean come from the observed information (a central
  difference Hessian of that log-likelihood in all parameters);
* **multiple imputation by data augmentation**
  (:func:`impute_multivariate_normal`): the I-step draws missing values from
  their conditional normal law, the P-step draws ``(mu, Sigma)`` from the
  normal-inverse-Wishart posterior given the completed data, using a Bartlett
  inverse-Wishart sampler (:func:`sample_inverse_wishart`);
* **Rubin's rules** (:func:`pool_estimates`) with the Barnard-Rubin
  small-sample degrees of freedom; and
* two **baselines that are biased under MAR**, kept for comparison only:
  listwise deletion (:func:`complete_cases`) and unconditional mean
  imputation (:func:`mean_impute`).

Ignorability: EM and data augmentation are valid when values are missing at
random and the parameters of the missingness mechanism are distinct from
``(mu, Sigma)``. Nothing in the data can confirm MAR against a missing-not-
at-random alternative; Little's test only probes the stronger MCAR condition.
"""

from __future__ import annotations

import math
import numbers
import operator
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.distributions import Normal
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    cholesky_semidefinite,
    cholesky_solve,
    inverse_spd,
    solve_lower,
)
from probstat.multivariate import MultivariateNormal
from probstat.optimization import central_hessian

from .bayesian import student_t_quantile
from .em import EMResult, expectation_maximization
from .sampling_distributions import chi_square_sf

_LOG_TWO_PI = math.log(2.0 * math.pi)

_IncompleteRows = tuple[tuple[float, ...], ...]


# ---------------------------------------------------------------------------
# Representation and pattern summary
# ---------------------------------------------------------------------------


def _coerce_incomplete(rows: Iterable[Iterable[object]]) -> _IncompleteRows:
    """Validate rows and encode missing entries as ``nan``.

    References:
        Project validation policy. No equation is evaluated.

    """
    try:
        raw = [tuple(row) for row in rows]
    except TypeError as error:
        raise TypeError("rows must be an iterable of iterables") from error
    if not raw:
        raise ValueError("at least one row is required")
    width = len(raw[0])
    if width == 0:
        raise ValueError("rows must contain at least one variable")
    coerced: list[tuple[float, ...]] = []
    for index, row in enumerate(raw):
        if len(row) != width:
            raise ValueError(f"row {index} has {len(row)} entries, expected {width}")
        values: list[float] = []
        for value in row:
            if value is None:
                values.append(math.nan)
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError("entries must be real numbers, None, or nan")
            number = float(value)
            if math.isinf(number):
                raise ValueError("entries must be finite (use None or nan if missing)")
            values.append(number)
        coerced.append(tuple(values))
    return tuple(coerced)


@dataclass(frozen=True, slots=True)
class _PatternGroup:
    """Rows sharing one missingness pattern, with their observed sub-vectors.

    References:
        [REF-LITTLE-RUBIN-2019]. No equation is evaluated.

    """

    observed: tuple[int, ...]
    missing: tuple[int, ...]
    rows: tuple[int, ...]
    values: tuple[Vector, ...]


def _group_patterns(data: _IncompleteRows) -> tuple[_PatternGroup, ...]:
    """Group rows by missingness pattern in a deterministic order.

    Patterns are sorted by decreasing number of observed variables, then
    lexicographically by the observed indicator (observed first).

    References:
        [REF-LITTLE-RUBIN-2019]. No equation is evaluated.

    """
    groups: dict[tuple[bool, ...], list[int]] = {}
    for index, row in enumerate(data):
        key = tuple(not math.isnan(value) for value in row)
        groups.setdefault(key, []).append(index)
    ordered = sorted(groups, key=lambda key: (-sum(key), tuple(not k for k in key)))
    result: list[_PatternGroup] = []
    for key in ordered:
        observed = tuple(j for j, flag in enumerate(key) if flag)
        missing = tuple(j for j, flag in enumerate(key) if not flag)
        members = tuple(groups[key])
        values = tuple(tuple(data[i][j] for j in observed) for i in members)
        result.append(_PatternGroup(observed, missing, members, values))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class MissingPattern:
    """Summary of where values are missing.

    Args:
        n_observations: Number of rows ``n``.
        n_variables: Number of variables ``p``.
        missing_fraction: Fraction of rows missing each variable.
        patterns: Distinct observed-indicator patterns (``True`` = observed),
            sorted by decreasing number of observed variables.
        pattern_counts: Number of rows with each pattern.
        complete_cases: Number of fully observed rows.

    References:
        [REF-LITTLE-RUBIN-2019], [REF-SCHAFER-1997].

    """

    n_observations: int
    n_variables: int
    missing_fraction: tuple[float, ...]
    patterns: tuple[tuple[bool, ...], ...]
    pattern_counts: tuple[int, ...]
    complete_cases: int

    @property
    def overall_missing_fraction(self) -> float:
        """Return the fraction of all ``n p`` entries that are missing.

        References:
            [REF-LITTLE-RUBIN-2019]. No equation is evaluated.

        """
        return math.fsum(self.missing_fraction) / self.n_variables


def _pattern_summary(
    data: _IncompleteRows, groups: Sequence[_PatternGroup]
) -> MissingPattern:
    """Build a :class:`MissingPattern` from validated data.

    References:
        [REF-LITTLE-RUBIN-2019]. No equation is evaluated.

    """
    n, p = len(data), len(data[0])
    fractions = tuple(
        sum(1 for row in data if math.isnan(row[j])) / n for j in range(p)
    )
    patterns = tuple(tuple(j in group.observed for j in range(p)) for group in groups)
    counts = tuple(len(group.rows) for group in groups)
    complete = sum(len(group.rows) for group in groups if not group.missing)
    return MissingPattern(n, p, fractions, patterns, counts, complete)


def missing_pattern(rows: Iterable[Iterable[object]]) -> MissingPattern:
    """Summarize the missingness of a data set.

    Args:
        rows: Equal-length rows; ``None`` or ``nan`` marks a missing entry.

    Returns:
        Per-variable missing fractions, distinct patterns with counts, and
        the complete-case count.

    Raises:
        TypeError: If an entry is neither real nor missing.
        ValueError: If rows are empty, ragged, or contain infinities.

    References:
        [REF-LITTLE-RUBIN-2019], [REF-SCHAFER-1997].

    """
    data = _coerce_incomplete(rows)
    return _pattern_summary(data, _group_patterns(data))


def complete_cases(rows: Iterable[Iterable[object]]) -> tuple[Vector, ...]:
    """Return the fully observed rows (listwise deletion).

    **Biased baseline.** Complete-case analysis is unbiased under MCAR but
    generally biased under MAR: when missingness depends on an observed
    variable correlated with the analysis variables, the complete cases are a
    selected subsample. It also discards the observed parts of incomplete
    rows. Provided for comparison with EM and multiple imputation.

    Args:
        rows: Equal-length rows; ``None`` or ``nan`` marks a missing entry.

    Returns:
        The rows without missing entries, in their original order.

    Raises:
        TypeError: If an entry is neither real nor missing.
        ValueError: If rows are empty, ragged, or contain infinities.

    Equation:
        [EQ-MISS-009].

    References:
        [REF-LITTLE-RUBIN-2019].

    """
    data = _coerce_incomplete(rows)
    return tuple(row for row in data if not any(math.isnan(v) for v in row))


def mean_impute(rows: Iterable[Iterable[object]]) -> tuple[Vector, ...]:
    """Fill each missing entry with its variable's observed mean.

    **Biased baseline.** Unconditional mean imputation keeps the observed
    mean of each variable (itself biased under MAR), but shrinks variances by
    roughly the observed fraction, attenuates covariances, and makes any
    downstream standard error too small because imputed values are treated as
    data. Provided for comparison with EM and multiple imputation.

    Args:
        rows: Equal-length rows; ``None`` or ``nan`` marks a missing entry.

    Returns:
        Completed rows.

    Raises:
        TypeError: If an entry is neither real nor missing.
        ValueError: If the data are invalid or a variable is never observed.

    Equation:
        [EQ-MISS-009].

    References:
        [REF-LITTLE-RUBIN-2019].

    """
    data = _coerce_incomplete(rows)
    means = _observed_means(data)
    return tuple(
        tuple(means[j] if math.isnan(v) else v for j, v in enumerate(row))
        for row in data
    )


def _observed_means(data: _IncompleteRows) -> Vector:
    """Return available-case means of each variable.

    References:
        [REF-LITTLE-RUBIN-2019]. No equation is evaluated.

    """
    means: list[float] = []
    for j in range(len(data[0])):
        column = [row[j] for row in data if not math.isnan(row[j])]
        if not column:
            raise ValueError(f"variable {j} is never observed")
        means.append(math.fsum(column) / len(column))
    return tuple(means)


# ---------------------------------------------------------------------------
# Observed-data likelihood and EM
# ---------------------------------------------------------------------------


def _submatrix(matrix: Matrix, rows: Sequence[int], columns: Sequence[int]) -> Matrix:
    """Return ``matrix[rows][columns]``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    return tuple(tuple(matrix[i][j] for j in columns) for i in rows)


def _observed_log_likelihood(
    groups: Sequence[_PatternGroup], mean: Vector, covariance: Matrix
) -> float:
    """Evaluate [EQ-MISS-001] for validated, grouped data.

    Raises:
        ValueError: If an observed covariance block is not positive definite.

    References:
        [REF-LITTLE-RUBIN-2019]. Equation [EQ-MISS-001].

    """
    terms: list[float] = []
    for group in groups:
        observed = group.observed
        if not observed:
            continue
        factor = cholesky(_submatrix(covariance, observed, observed))
        log_det = 2.0 * math.fsum(math.log(factor[i][i]) for i in range(len(observed)))
        constant = len(observed) * _LOG_TWO_PI + log_det
        location = tuple(mean[j] for j in observed)
        for value in group.values:
            residual = tuple(v - m for v, m in zip(value, location, strict=True))
            z = solve_lower(factor, residual)
            terms.append(-0.5 * (constant + math.fsum(c * c for c in z)))
    return math.fsum(terms)


def _checked_parameters(
    mean: Iterable[object], covariance: Iterable[Iterable[object]], width: int
) -> tuple[Vector, Matrix]:
    """Validate a mean vector and a positive-definite covariance.

    References:
        Project validation policy. No equation is evaluated.

    """
    law = MultivariateNormal(
        as_vector(mean, name="mean"),
        tuple(as_vector(row, name="covariance row") for row in covariance),
    )
    if law.dimension != width:
        raise ValueError("parameter dimension must equal the number of variables")
    return law.mean, law.covariance


def incomplete_normal_log_likelihood(
    rows: Iterable[Iterable[object]],
    mean: Iterable[object],
    covariance: Iterable[Iterable[object]],
) -> float:
    """Return the observed-data multivariate normal log-likelihood.

    Each row contributes the marginal normal density of its observed block,
    ``N(x_o | mu_o, Sigma_oo)``; rows with no observed entry contribute zero.
    Under MAR with distinct parameters this is the likelihood that ignorable
    inference maximizes.

    Args:
        rows: Equal-length rows; ``None`` or ``nan`` marks a missing entry.
        mean: Mean vector ``mu``.
        covariance: Symmetric positive-definite ``Sigma``.

    Returns:
        ``sum_i log N(x_(o,i) | mu_(o,i), Sigma_(oo,i))``.

    Raises:
        TypeError: If an entry is neither real nor missing.
        ValueError: For invalid data or parameters.

    Equation:
        [EQ-MISS-001].

    References:
        [REF-LITTLE-RUBIN-2019], [REF-SCHAFER-1997].

    """
    data = _coerce_incomplete(rows)
    mu, sigma = _checked_parameters(mean, covariance, len(data[0]))
    return _observed_log_likelihood(_group_patterns(data), mu, sigma)


def _conditional_law(
    group: _PatternGroup, mean: Vector, covariance: Matrix
) -> tuple[tuple[Vector, ...], Matrix]:
    """Return regression rows ``B = Sigma_mo Sigma_oo^-1`` and ``Sigma_mm.o``.

    Returns:
        ``B`` as one row per missing index and the Schur complement
        ``Sigma_mm - Sigma_mo Sigma_oo^-1 Sigma_om``.

    References:
        [REF-LITTLE-RUBIN-2019], [REF-MARDIA-KENT-BIBBY-1979, Chapter 3].
        Equation [EQ-MISS-002].

    """
    observed, missing = group.observed, group.missing
    if not observed:
        return tuple(() for _ in missing), _submatrix(covariance, missing, missing)
    factor = cholesky(_submatrix(covariance, observed, observed))
    cross = [tuple(covariance[o][j] for o in observed) for j in missing]
    coefficients = tuple(cholesky_solve(factor, column) for column in cross)
    schur = tuple(
        tuple(
            covariance[j][k]
            - math.fsum(
                c * b for c, b in zip(cross[a], coefficients[b_index], strict=True)
            )
            for b_index, k in enumerate(missing)
        )
        for a, j in enumerate(missing)
    )
    symmetric = tuple(
        tuple(0.5 * (schur[a][b] + schur[b][a]) for b in range(len(missing)))
        for a in range(len(missing))
    )
    return coefficients, symmetric


def _conditional_mean(
    group: _PatternGroup,
    value: Vector,
    coefficients: tuple[Vector, ...],
    mean: Vector,
) -> Vector:
    """Return ``mu_m + B (x_o - mu_o)`` for one row.

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapter 3]. Equation [EQ-MISS-002].

    """
    residual = tuple(v - mean[j] for v, j in zip(value, group.observed, strict=True))
    return tuple(
        mean[j] + math.fsum(b * r for b, r in zip(row, residual, strict=True))
        for j, row in zip(group.missing, coefficients, strict=True)
    )


def _completed_row(
    group: _PatternGroup, value: Vector, filled: Vector, width: int
) -> Vector:
    """Merge observed values and filled-in missing values into one row.

    References:
        [REF-LITTLE-RUBIN-2019]. No equation is evaluated.

    """
    row = [0.0] * width
    for j, v in zip(group.observed, value, strict=True):
        row[j] = v
    for j, v in zip(group.missing, filled, strict=True):
        row[j] = v
    return tuple(row)


_Statistics = tuple[tuple[Vector, ...], Matrix]


def _e_step(
    groups: Sequence[_PatternGroup],
    n: int,
    width: int,
    mean: Vector,
    covariance: Matrix,
) -> _Statistics:
    """Return conditionally expected rows and summed conditional covariances.

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-LITTLE-RUBIN-2019].
        Equation [EQ-MISS-002].

    """
    completed: list[Vector] = [()] * n
    correction = [[0.0] * width for _ in range(width)]
    for group in groups:
        if not group.missing:
            for index, value in zip(group.rows, group.values, strict=True):
                completed[index] = value
            continue
        coefficients, schur = _conditional_law(group, mean, covariance)
        count = len(group.rows)
        for a, j in enumerate(group.missing):
            for b, k in enumerate(group.missing):
                correction[j][k] += count * schur[a][b]
        for index, value in zip(group.rows, group.values, strict=True):
            filled = _conditional_mean(group, value, coefficients, mean)
            completed[index] = _completed_row(group, value, filled, width)
    return tuple(completed), tuple(tuple(row) for row in correction)


def _m_step(statistics: _Statistics) -> tuple[Vector, Matrix]:
    """Return the complete-data MLE from completed rows and corrections.

    With no missing values this is exactly the sample mean and the divisor-``n``
    sample covariance computed on centred values.

    Raises:
        ValueError: If the completed covariance is not positive definite.

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-LITTLE-RUBIN-2019].
        Equation [EQ-MISS-002].

    """
    rows, correction = statistics
    n, width = len(rows), len(rows[0])
    mean = tuple(math.fsum(row[j] for row in rows) / n for j in range(width))
    centered = tuple(tuple(row[j] - mean[j] for j in range(width)) for row in rows)
    covariance = tuple(
        tuple(
            (math.fsum(row[i] * row[j] for row in centered) + correction[i][j]) / n
            for j in range(width)
        )
        for i in range(width)
    )
    try:
        cholesky(covariance)
    except ValueError as error:
        raise ValueError(
            "EM covariance is not positive definite; the data do not identify "
            "Sigma (too few observed pairs or collinear variables)"
        ) from error
    return mean, covariance


@dataclass(frozen=True, slots=True)
class IncompleteNormalFit:
    """Maximum-likelihood multivariate normal fit to incomplete data.

    Args:
        mean: EM estimate of ``mu``.
        covariance: EM estimate of ``Sigma`` (divisor ``n``).
        log_likelihood: Observed-data log-likelihood at the estimate.
        pattern: Missingness summary of the data.
        mean_covariance: Inverse observed information restricted to ``mu``
            (accounts for the ``mu``-``Sigma`` cross information), or ``None``
            if not requested or not positive definite.
        mean_standard_errors: Square roots of its diagonal, or ``None``.
        em: Underlying EM trace with parameters ``(mu, Sigma)``.

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-LITTLE-RUBIN-2019].

    """

    mean: Vector
    covariance: Matrix
    log_likelihood: float
    pattern: MissingPattern
    mean_covariance: Matrix | None
    mean_standard_errors: Vector | None
    em: EMResult[tuple[Vector, Matrix]]

    @property
    def distribution(self) -> MultivariateNormal:
        """Return the fitted law as a :class:`MultivariateNormal`.

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 3]. No equation is evaluated.

        """
        return MultivariateNormal(self.mean, self.covariance)


def _initial_parameters(data: _IncompleteRows) -> tuple[Vector, Matrix]:
    """Return available-case means and a diagonal available-case covariance.

    References:
        [REF-SCHAFER-1997]. No equation is evaluated.

    """
    means = _observed_means(data)
    variances: list[float] = []
    for j, location in enumerate(means):
        column = [row[j] for row in data if not math.isnan(row[j])]
        if len(column) < 2:
            raise ValueError(f"variable {j} needs at least two observed values")
        spread = math.fsum((v - location) ** 2 for v in column) / len(column)
        if not spread > 0.0:
            raise ValueError(f"observed values of variable {j} are constant")
        variances.append(spread)
    width = len(means)
    covariance = tuple(
        tuple(variances[i] if i == j else 0.0 for j in range(width))
        for i in range(width)
    )
    return means, covariance


def _mean_information_covariance(
    groups: Sequence[_PatternGroup],
    mean: Vector,
    covariance: Matrix,
    relative_step: float,
) -> Matrix | None:
    """Invert the numerical observed information and return its ``mu`` block.

    Parameters are standardized, ``mu_j = m_j + s_j t_j`` and
    ``Sigma_jk = S_jk + s_j s_k u_jk`` (``j >= k``) with ``s_j = sqrt(S_jj)``,
    so one finite-difference step suits every scale. The Hessian of the
    observed-data log-likelihood is taken at ``t = u = 0`` and mapped back.

    References:
        [REF-LITTLE-RUBIN-2019], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].
        Equation [EQ-MISS-003].

    """
    width = len(mean)
    scales = tuple(math.sqrt(covariance[j][j]) for j in range(width))
    pairs = tuple((j, k) for j in range(width) for k in range(j + 1))

    def objective(theta: Sequence[float]) -> float:
        """Return the log-likelihood at standardized perturbations.

        References:
            [REF-LITTLE-RUBIN-2019]. Equation [EQ-MISS-001].

        """
        mu = tuple(mean[j] + scales[j] * theta[j] for j in range(width))
        sigma = [list(row) for row in covariance]
        for offset, (j, k) in enumerate(pairs):
            delta = scales[j] * scales[k] * theta[width + offset]
            sigma[j][k] += delta
            if j != k:
                sigma[k][j] += delta
        frozen = tuple(tuple(row) for row in sigma)
        return _observed_log_likelihood(groups, mu, frozen)

    hessian, _ = central_hessian(
        objective, (0.0,) * (width + len(pairs)), relative_step=relative_step
    )
    information = tuple(tuple(-value for value in row) for row in hessian)
    try:
        inverse = inverse_spd(information)
    except ValueError:
        return None
    return tuple(
        tuple(scales[i] * scales[j] * inverse[i][j] for j in range(width))
        for i in range(width)
    )


def fit_incomplete_multivariate_normal(
    rows: Iterable[Iterable[object]],
    *,
    initial: tuple[Sequence[float], Sequence[Sequence[float]]] | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 10_000,
    standard_errors: bool = True,
    hessian_step: float = 1e-3,
) -> IncompleteNormalFit:
    """Fit ``N_p(mu, Sigma)`` to data with values missing at random by EM.

    E-step (per missingness pattern, observed block ``o``, missing block
    ``m``): ``E[x_m | x_o] = mu_m + Sigma_mo Sigma_oo^-1 (x_o - mu_o)`` and
    ``Cov[x_m | x_o] = Sigma_mm - Sigma_mo Sigma_oo^-1 Sigma_om`` (Schur
    complement). M-step: ``mu`` is the mean of the completed rows and
    ``Sigma`` their divisor-``n`` covariance plus the summed conditional
    covariances ([EQ-MISS-002]). The exact observed-data log-likelihood
    ([EQ-MISS-001]) is checked for monotonicity at every iteration by
    :func:`~probstat.inference.expectation_maximization`.

    With no missing values the first M-step returns exactly
    :func:`~probstat.inference.fit_multivariate_normal_mle`'s estimate.

    Args:
        rows: Equal-length rows; ``None`` or ``nan`` marks a missing entry.
        initial: Optional ``(mu, Sigma)`` start; by default the available-case
            means and a diagonal matrix of available-case variances.
        tolerance: Relative log-likelihood change tolerance.
        max_iterations: Maximum EM iterations.
        standard_errors: Whether to compute the observed-information
            covariance of the mean ([EQ-MISS-003]).
        hessian_step: Finite-difference step in standardized parameters.

    Returns:
        :class:`IncompleteNormalFit`.

    Raises:
        TypeError: If an entry is neither real nor missing.
        ValueError: If a variable has fewer than two observed values, is
            constant, or the data do not identify a positive-definite
            ``Sigma``.

    Equation:
        [EQ-EM-001], [EQ-MISS-001], [EQ-MISS-002], [EQ-MISS-003].

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-LITTLE-RUBIN-2019],
        [REF-SCHAFER-1997].

    """
    data = _coerce_incomplete(rows)
    groups = _group_patterns(data)
    n, width = len(data), len(data[0])
    if initial is None:
        start = _initial_parameters(data)
    else:
        start = _checked_parameters(initial[0], initial[1], width)
    step = require_finite_real(hessian_step, name="hessian_step")
    if step <= 0.0:
        raise ValueError("hessian_step must be positive")

    def e_step(theta: tuple[Vector, Matrix]) -> tuple[_Statistics, float]:
        """Return completed statistics and the observed log-likelihood.

        References:
            [REF-DEMPSTER-LAIRD-RUBIN-1977]. Equation [EQ-MISS-002].

        """
        mu, sigma = theta
        return (
            _e_step(groups, n, width, mu, sigma),
            _observed_log_likelihood(groups, mu, sigma),
        )

    result = expectation_maximization(
        start, e_step, _m_step, tolerance=tolerance, max_iterations=max_iterations
    )
    mean, covariance = result.parameters
    mean_covariance: Matrix | None = None
    errors: Vector | None = None
    if standard_errors:
        mean_covariance = _mean_information_covariance(groups, mean, covariance, step)
        if mean_covariance is not None:
            errors = tuple(math.sqrt(mean_covariance[j][j]) for j in range(width))
    return IncompleteNormalFit(
        mean,
        covariance,
        result.log_likelihood,
        _pattern_summary(data, groups),
        mean_covariance,
        errors,
        result,
    )


# ---------------------------------------------------------------------------
# Little's MCAR test
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LittleMCARResult:
    """Little's chi-square test of missing completely at random.

    Args:
        statistic: ``d^2`` of [EQ-MISS-004].
        degrees_of_freedom: ``sum_j p_j - p``.
        p_value: Upper chi-square tail probability.
        n_patterns: Number of patterns with at least one observed variable.

    References:
        [REF-LITTLE-1988].

    """

    statistic: float
    degrees_of_freedom: int
    p_value: float
    n_patterns: int


def little_mcar_test(
    rows: Iterable[Iterable[object]],
    *,
    fit: IncompleteNormalFit | None = None,
) -> LittleMCARResult:
    """Test MCAR by comparing pattern-wise means with the EM estimate.

    ``d^2 = sum_j n_j (ybar_j - mu_(o_j))' Sigma_(o_j o_j)^-1 (ybar_j - mu_(o_j))``
    over patterns ``j`` with ``n_j`` rows and observed set ``o_j``, using the
    EM (maximum-likelihood) ``mu`` and ``Sigma``. Under MCAR and normality it
    is asymptotically chi-square with ``sum_j |o_j| - p`` degrees of freedom.
    A small p-value is evidence against MCAR; a large one does not establish
    MAR or MCAR, and the test is sensitive to non-normality.

    Args:
        rows: Equal-length rows; ``None`` or ``nan`` marks a missing entry.
        fit: Optional EM fit of the same rows (computed when omitted).

    Returns:
        :class:`LittleMCARResult`.

    Raises:
        ValueError: If there is only one pattern with observed values (the
            test has zero degrees of freedom) or the data are invalid.

    Equation:
        [EQ-MISS-004].

    References:
        [REF-LITTLE-1988], [REF-LITTLE-RUBIN-2019].

    """
    data = _coerce_incomplete(rows)
    groups = _group_patterns(data)
    width = len(data[0])
    if fit is None:
        fit = fit_incomplete_multivariate_normal(data, standard_errors=False)
    elif len(fit.mean) != width:
        raise ValueError("fit dimension must equal the number of variables")
    terms: list[float] = []
    observed_total = 0
    used = 0
    for group in groups:
        observed = group.observed
        if not observed:
            continue
        used += 1
        observed_total += len(observed)
        count = len(group.values)
        means = tuple(
            math.fsum(value[a] for value in group.values) / count
            for a in range(len(observed))
        )
        residual = tuple(m - fit.mean[j] for m, j in zip(means, observed, strict=True))
        factor = cholesky(_submatrix(fit.covariance, observed, observed))
        z = solve_lower(factor, residual)
        terms.append(count * math.fsum(c * c for c in z))
    degrees = observed_total - width
    if degrees <= 0:
        raise ValueError("Little's test needs at least two missingness patterns")
    statistic = math.fsum(terms)
    return LittleMCARResult(statistic, degrees, chi_square_sf(statistic, degrees), used)


# ---------------------------------------------------------------------------
# Inverse-Wishart sampling and data augmentation
# ---------------------------------------------------------------------------


def _inverse_wishart_draw(
    generator: random.Random, scale_factor: Matrix, degrees: float
) -> Matrix:
    """Draw ``Sigma ~ IW(Psi, nu)`` from the lower Cholesky factor of ``Psi``.

    Bartlett: ``A`` lower triangular with ``A_ii^2 ~ chi^2(nu - i)``
    (``i = 0..p-1``) and standard normal ``A_ij`` below the diagonal, so
    ``A A' ~ W_p(nu, I)``. With ``Psi = C C'``,
    ``Sigma = C A^-T A^-1 C'`` has ``Sigma^-1 ~ W_p(nu, Psi^-1)``.

    References:
        [REF-SMITH-HOCKING-1972], [REF-MARDIA-KENT-BIBBY-1979, Chapter 3].
        Equation [EQ-MISS-005].

    """
    p = len(scale_factor)
    bartlett = [[0.0] * p for _ in range(p)]
    for i in range(p):
        bartlett[i][i] = math.sqrt(
            2.0 * generator.gammavariate(0.5 * (degrees - i), 1.0)
        )
        for j in range(i):
            bartlett[i][j] = generator.gauss(0.0, 1.0)
    lower = tuple(tuple(row) for row in bartlett)
    inverse_columns = [
        solve_lower(lower, tuple(1.0 if r == c else 0.0 for r in range(p)))
        for c in range(p)
    ]
    # G = C A^-T, G[i][k] = sum_l C[i][l] (A^-1)[k][l]; both factors are lower
    # triangular, so only l <= min(i, k) contributes. (A^-1)[k][l] is entry k
    # of column l, i.e. inverse_columns[l][k].
    factor = tuple(
        tuple(
            math.fsum(
                scale_factor[i][ell] * inverse_columns[ell][k]
                for ell in range(min(i, k) + 1)
            )
            for k in range(p)
        )
        for i in range(p)
    )
    result = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1):
            value = math.fsum(factor[i][k] * factor[j][k] for k in range(p))
            result[i][j] = result[j][i] = value
    return tuple(tuple(row) for row in result)


def _checked_scale(scale: Iterable[Iterable[object]]) -> Matrix:
    """Validate a symmetric positive-definite scale matrix.

    References:
        Project validation policy. No equation is evaluated.

    """
    rows = tuple(as_vector(row, name="scale row") for row in scale)
    width = len(rows)
    if width == 0:
        raise ValueError("scale must be a non-empty square matrix")
    return MultivariateNormal((0.0,) * width, rows).covariance


def sample_inverse_wishart(
    scale: Iterable[Iterable[object]],
    degrees_of_freedom: float,
    size: int = 1,
    *,
    seed: int | RandomStream | None = None,
) -> tuple[Matrix, ...]:
    """Draw inverse-Wishart matrices by the Bartlett decomposition.

    ``Sigma ~ IW_p(Psi, nu)`` means ``Sigma^-1 ~ W_p(nu, Psi^-1)``; then
    ``E[Sigma] = Psi / (nu - p - 1)`` for ``nu > p + 1``.

    Args:
        scale: Symmetric positive-definite ``p x p`` scale ``Psi``.
        degrees_of_freedom: ``nu > p - 1`` (real values allowed).
        size: Number of draws (non-negative).
        seed: Integer seed, :class:`RandomStream`, or ``None``.

    Returns:
        Tuple of symmetric positive-definite matrices.

    Raises:
        ValueError: For an invalid scale, degrees of freedom, or size.

    Equation:
        [EQ-MISS-005].

    References:
        [REF-SMITH-HOCKING-1972], [REF-MARDIA-KENT-BIBBY-1979, Chapter 3],
        [REF-GELMAN-ETAL-2013, Section 3.6].

    """
    psi = _checked_scale(scale)
    p = len(psi)
    nu = require_finite_real(degrees_of_freedom, name="degrees_of_freedom")
    if nu <= p - 1:
        raise ValueError("degrees_of_freedom must exceed p - 1")
    count = require_non_negative_integer(size, name="size")
    generator = resolve_stream(seed).generator()
    factor = cholesky(psi)
    return tuple(_inverse_wishart_draw(generator, factor, nu) for _ in range(count))


@dataclass(frozen=True, slots=True)
class NormalInverseWishartPrior:
    """Normal-inverse-Wishart prior for ``(mu, Sigma)``.

    ``Sigma ~ IW(Psi_0, nu_0)`` and ``mu | Sigma ~ N(mu_0, Sigma / kappa_0)``.
    The default (``kappa_0 = 0``, ``Psi_0 = 0``, ``nu_0 = -1``) is the
    improper Jeffreys-type prior ``p(mu, Sigma) propto |Sigma|^-(p+1)/2``, whose
    complete-data posterior is ``Sigma | Y ~ IW(A, n - 1)``,
    ``mu | Sigma, Y ~ N(ybar, Sigma / n)`` with ``A`` the centred
    cross-product matrix.

    Args:
        mean: Prior location ``mu_0`` (required when ``kappa > 0``).
        kappa: Prior sample size ``kappa_0 >= 0`` for the mean.
        scale: Prior scale ``Psi_0`` (``None`` means the zero matrix).
        degrees_of_freedom: ``nu_0``.

    References:
        [REF-GELMAN-ETAL-2013, Section 3.6], [REF-SCHAFER-1997].

    """

    mean: Vector | None = None
    kappa: float = 0.0
    scale: Matrix | None = None
    degrees_of_freedom: float = -1.0

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: For a negative ``kappa`` or a missing ``mean``.

        References:
            Project validation policy. No equation is evaluated.

        """
        kappa = require_finite_real(self.kappa, name="kappa")
        if kappa < 0.0:
            raise ValueError("kappa must be non-negative")
        if kappa > 0.0 and self.mean is None:
            raise ValueError("a prior mean is required when kappa > 0")
        require_finite_real(self.degrees_of_freedom, name="degrees_of_freedom")
        if self.mean is not None:
            object.__setattr__(
                self,
                "mean",
                tuple(require_finite_real(v, name="mean") for v in self.mean),
            )
        if self.scale is not None:
            rows = tuple(
                tuple(require_finite_real(v, name="scale") for v in row)
                for row in self.scale
            )
            object.__setattr__(self, "scale", rows)


def _posterior_step(
    completed: Sequence[Vector],
    prior: NormalInverseWishartPrior,
    generator: random.Random,
) -> tuple[Vector, Matrix]:
    """Draw ``(mu, Sigma)`` from the normal-inverse-Wishart posterior (P-step).

    References:
        [REF-GELMAN-ETAL-2013, Section 3.6], [REF-SCHAFER-1997].
        Equation [EQ-MISS-006].

    """
    n, width = len(completed), len(completed[0])
    columns = list(zip(*completed, strict=True))
    ybar = tuple(math.fsum(column) / n for column in columns)
    centered = [
        [value - location for value in column]
        for column, location in zip(columns, ybar, strict=True)
    ]
    psi = [[0.0] * width for _ in range(width)]
    for i in range(width):
        for j in range(i + 1):
            psi[i][j] = psi[j][i] = math.fsum(
                map(operator.mul, centered[i], centered[j])
            )
    kappa_n = prior.kappa + n
    location = ybar
    if prior.kappa > 0.0 and prior.mean is not None:
        mu0 = prior.mean
        shrink = prior.kappa * n / kappa_n
        for i in range(width):
            for j in range(width):
                psi[i][j] += shrink * (ybar[i] - mu0[i]) * (ybar[j] - mu0[j])
        location = tuple(
            (prior.kappa * mu0[j] + n * ybar[j]) / kappa_n for j in range(width)
        )
    if prior.scale is not None:
        for i in range(width):
            for j in range(width):
                psi[i][j] += prior.scale[i][j]
    nu_n = prior.degrees_of_freedom + n
    if nu_n <= width - 1:
        raise ValueError("posterior degrees of freedom must exceed p - 1")
    factor = cholesky(tuple(tuple(row) for row in psi))
    sigma = _inverse_wishart_draw(generator, factor, nu_n)
    sigma_factor = cholesky(sigma)
    z = tuple(generator.gauss(0.0, 1.0) for _ in range(width))
    root = 1.0 / math.sqrt(kappa_n)
    mu = tuple(
        location[i] + root * math.fsum(sigma_factor[i][k] * z[k] for k in range(i + 1))
        for i in range(width)
    )
    return mu, sigma


def _imputation_step(
    groups: Sequence[_PatternGroup],
    n: int,
    width: int,
    mean: Vector,
    covariance: Matrix,
    generator: random.Random,
) -> tuple[Vector, ...]:
    """Draw every missing block from its conditional normal law (I-step).

    References:
        [REF-TANNER-WONG-1987], [REF-SCHAFER-1997]. Equation [EQ-MISS-006].

    """
    completed: list[Vector] = [()] * n
    for group in groups:
        if not group.missing:
            for index, value in zip(group.rows, group.values, strict=True):
                completed[index] = value
            continue
        coefficients, schur = _conditional_law(group, mean, covariance)
        factor = cholesky_semidefinite(schur)
        gauss = generator.gauss
        observed_mean = tuple(mean[j] for j in group.observed)
        missing_mean = tuple(mean[j] for j in group.missing)
        rows_of_factor = tuple((a, row[: a + 1]) for a, row in enumerate(factor))
        for index, value in zip(group.rows, group.values, strict=True):
            residual = tuple(map(operator.sub, value, observed_mean))
            z = [gauss(0.0, 1.0) for _ in missing_mean]
            draw = tuple(
                missing_mean[a]
                + sum(map(operator.mul, coefficients[a], residual))
                + sum(map(operator.mul, lower, z))
                for a, lower in rows_of_factor
            )
            completed[index] = _completed_row(group, value, draw, width)
    return tuple(completed)


@dataclass(frozen=True, slots=True)
class MultipleImputation:
    """Completed data sets from multivariate normal data augmentation.

    Args:
        datasets: ``m`` completed copies of the data; observed entries are
            unchanged.
        means: Final ``mu`` draw of each chain (the one used for its I-step).
        covariances: Final ``Sigma`` draw of each chain.
        burn_in: Data-augmentation iterations per chain before imputing.
        start: EM fit used as the starting value of every chain.
        pattern: Missingness summary.

    References:
        [REF-RUBIN-1987], [REF-SCHAFER-1997].

    """

    datasets: tuple[tuple[Vector, ...], ...]
    means: tuple[Vector, ...]
    covariances: tuple[Matrix, ...]
    burn_in: int
    start: IncompleteNormalFit
    pattern: MissingPattern


def impute_multivariate_normal(
    rows: Iterable[Iterable[object]],
    imputations: int = 5,
    *,
    burn_in: int = 50,
    prior: NormalInverseWishartPrior | None = None,
    start: IncompleteNormalFit | None = None,
    seed: int | RandomStream | None = None,
) -> MultipleImputation:
    """Create proper multiple imputations under ``N_p(mu, Sigma)`` and MAR.

    Each imputation ``k`` runs its own data-augmentation chain on the child
    stream ``seed.child("imputation", k)``, started at the EM estimate:
    ``burn_in`` cycles of an I-step (draw ``x_m ~ N(mu_m + B(x_o - mu_o),
    Sigma_mm.o)``) followed by a P-step (draw ``(mu, Sigma)`` from the
    normal-inverse-Wishart posterior of the completed data), then a final
    I-step whose completed data set is returned. Because parameters are drawn
    from their posterior, between-imputation variance reflects parameter
    uncertainty, as Rubin's rules require.

    Args:
        rows: Equal-length rows; ``None`` or ``nan`` marks a missing entry.
        imputations: Number ``m >= 1`` of completed data sets.
        burn_in: Data-augmentation cycles per chain (``>= 1``).
        prior: Normal-inverse-Wishart prior; default the Jeffreys-type prior.
        start: Optional EM fit of the same rows (computed when omitted).
        seed: Integer seed, :class:`RandomStream`, or ``None``.

    Returns:
        :class:`MultipleImputation`.

    Raises:
        ValueError: For invalid settings or data EM cannot fit.

    Equation:
        [EQ-MISS-002], [EQ-MISS-005], [EQ-MISS-006].

    References:
        [REF-TANNER-WONG-1987], [REF-SCHAFER-1997], [REF-RUBIN-1987],
        [REF-LITTLE-RUBIN-2019].

    """
    data = _coerce_incomplete(rows)
    groups = _group_patterns(data)
    n, width = len(data), len(data[0])
    count = require_non_negative_integer(imputations, name="imputations")
    if count == 0:
        raise ValueError("imputations must be positive")
    cycles = require_non_negative_integer(burn_in, name="burn_in")
    if cycles == 0:
        raise ValueError(
            "burn_in must be positive (imputing at the EM estimate is improper)"
        )
    chosen_prior = NormalInverseWishartPrior() if prior is None else prior
    if chosen_prior.mean is not None and len(chosen_prior.mean) != width:
        raise ValueError("prior mean dimension must equal the number of variables")
    if chosen_prior.scale is not None and len(chosen_prior.scale) != width:
        raise ValueError("prior scale dimension must equal the number of variables")
    fit = (
        fit_incomplete_multivariate_normal(data, standard_errors=False)
        if start is None
        else start
    )
    if len(fit.mean) != width:
        raise ValueError("start dimension must equal the number of variables")
    stream = resolve_stream(seed)
    datasets: list[tuple[Vector, ...]] = []
    means: list[Vector] = []
    covariances: list[Matrix] = []
    for k in range(count):
        generator = stream.child("imputation", k).generator()
        mu, sigma = fit.mean, fit.covariance
        for _ in range(cycles):
            completed = _imputation_step(groups, n, width, mu, sigma, generator)
            mu, sigma = _posterior_step(completed, chosen_prior, generator)
        datasets.append(_imputation_step(groups, n, width, mu, sigma, generator))
        means.append(mu)
        covariances.append(sigma)
    return MultipleImputation(
        tuple(datasets),
        tuple(means),
        tuple(covariances),
        cycles,
        fit,
        _pattern_summary(data, groups),
    )


# ---------------------------------------------------------------------------
# Rubin's rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PooledEstimate:
    """Scalar estimate combined across imputations by Rubin's rules.

    Args:
        estimate: ``Qbar``, the mean of the completed-data estimates.
        within_variance: ``Ubar``, the mean completed-data variance.
        between_variance: ``B``, the between-imputation variance.
        total_variance: ``T = Ubar + (1 + 1/m) B``.
        standard_error: ``sqrt(T)``.
        relative_increase: ``r = (1 + 1/m) B / Ubar``.
        missing_variance_fraction: ``lambda = (1 + 1/m) B / T``.
        fraction_missing_information: ``gamma = ((nu + 1) lambda + 2)/(nu + 3)``
            with ``nu`` the reported degrees of freedom.
        degrees_of_freedom: Rubin's ``nu_old`` or, given the complete-data
            degrees of freedom, the Barnard-Rubin ``nu`` (``inf`` when
            ``B = 0`` and no complete-data value is given).
        lower: Lower limit of the t-based interval.
        upper: Upper limit of the t-based interval.
        confidence: Interval level.
        imputations: ``m``.

    References:
        [REF-RUBIN-1987], [REF-BARNARD-RUBIN-1999].

    """

    estimate: float
    within_variance: float
    between_variance: float
    total_variance: float
    standard_error: float
    relative_increase: float
    missing_variance_fraction: float
    fraction_missing_information: float
    degrees_of_freedom: float
    lower: float
    upper: float
    confidence: float
    imputations: int


def pool_estimates(
    estimates: Iterable[float],
    variances: Iterable[float],
    *,
    complete_data_df: float | None = None,
    confidence: float = 0.95,
) -> PooledEstimate:
    """Combine completed-data estimates of a scalar by Rubin's rules.

    ``Qbar = mean(Q_k)``, ``Ubar = mean(U_k)``,
    ``B = sum (Q_k - Qbar)^2 / (m - 1)``, ``T = Ubar + (1 + 1/m) B``,
    ``nu_old = (m - 1) / lambda^2``. With complete-data degrees of freedom
    ``nu_com`` the Barnard-Rubin value ``(1/nu_old + 1/nu_obs)^-1`` with
    ``nu_obs = (nu_com + 1)/(nu_com + 3) nu_com (1 - lambda)`` is used; it never
    exceeds ``nu_com``. The interval is ``Qbar +/- t_nu sqrt(T)``.

    Args:
        estimates: ``m >= 2`` completed-data point estimates ``Q_k``.
        variances: Their completed-data variances ``U_k >= 0`` (squared
            standard errors).
        complete_data_df: Optional complete-data degrees of freedom (for a
            regression coefficient, ``n - p``).
        confidence: Interval level in ``(0, 1)``.

    Returns:
        :class:`PooledEstimate`.

    Raises:
        ValueError: For mismatched lengths, ``m < 2``, negative variances,
            ``Ubar = 0``, or invalid ``complete_data_df`` / ``confidence``.

    Equation:
        [EQ-MISS-007], [EQ-MISS-008].

    References:
        [REF-RUBIN-1987], [REF-BARNARD-RUBIN-1999], [REF-SCHAFER-1997].

    """
    q = tuple(require_finite_real(v, name="estimate") for v in estimates)
    u = tuple(require_finite_real(v, name="variance") for v in variances)
    if len(q) != len(u):
        raise ValueError("estimates and variances must have the same length")
    m = len(q)
    if m < 2:
        raise ValueError("at least two imputations are required")
    if any(v < 0.0 for v in u):
        raise ValueError("variances must be non-negative")
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    q_bar = math.fsum(q) / m
    u_bar = math.fsum(u) / m
    if not u_bar > 0.0:
        raise ValueError("the mean within-imputation variance must be positive")
    between = math.fsum((v - q_bar) ** 2 for v in q) / (m - 1)
    inflated = (1.0 + 1.0 / m) * between
    total = u_bar + inflated
    r = inflated / u_bar
    lam = inflated / total
    nu_old = math.inf if lam == 0.0 else (m - 1) / (lam * lam)
    if complete_data_df is None:
        nu = nu_old
    else:
        nu_com = require_finite_real(complete_data_df, name="complete_data_df")
        if nu_com <= 0.0:
            raise ValueError("complete_data_df must be positive")
        nu_obs = (nu_com + 1.0) / (nu_com + 3.0) * nu_com * (1.0 - lam)
        nu = nu_obs if math.isinf(nu_old) else 1.0 / (1.0 / nu_old + 1.0 / nu_obs)
    gamma = lam if math.isinf(nu) else ((nu + 1.0) * lam + 2.0) / (nu + 3.0)
    upper_prob = 0.5 * (1.0 + level)
    if math.isinf(nu):
        critical = Normal(0.0, 1.0).quantile(upper_prob)
    else:
        critical = student_t_quantile(upper_prob, nu)
    se = math.sqrt(total)
    return PooledEstimate(
        q_bar,
        u_bar,
        between,
        total,
        se,
        r,
        lam,
        gamma,
        nu,
        q_bar - critical * se,
        q_bar + critical * se,
        level,
        m,
    )
