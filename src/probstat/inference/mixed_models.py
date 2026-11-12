"""Linear and generalized linear mixed models with grouped random effects.

Linear mixed model
    ``y = X beta + Z b + epsilon`` with ``epsilon ~ N(0, sigma^2 I)`` and
    ``b ~ N(0, Sigma_theta)``. Each random-effect term has a grouping factor
    with ``l`` levels and ``r`` columns (an intercept and slopes); its ``r x r``
    covariance is shared by all levels and terms are independent. Following
    lme4 (Bates et al. 2015), ``b = Lambda_theta u`` with
    ``u ~ N(0, sigma^2 I_q)``, where the relative covariance factor
    ``Lambda_theta`` is block diagonal with one lower-triangular ``r x r``
    block ``T`` per level [EQ-LMM-001]. For fixed ``theta`` the penalized
    least-squares problem ``min ||y - X beta - Z Lambda u||^2 + ||u||^2`` is
    solved through the Cholesky factor ``L L' = Lambda' Z' Z Lambda + I`` and
    its Schur complement ``R_X' R_X`` [EQ-LMM-002], which gives ``beta``,
    ``u``, ``sigma`` and the profiled ML and REML deviances in closed form
    [EQ-LMM-003, EQ-LMM-004]; only ``theta`` is optimized numerically. The
    factor is dense but computed inside the envelope of the matrix, so a
    single grouping factor costs ``O(l r^3)`` rather than ``O(q^3)``.

    Reported quantities: fixed effects and their covariance
    ``sigma^2 (R_X' R_X)^-1``, variance components, conditional modes
    (BLUPs) ``b = Lambda u`` with conditional covariances
    ``sigma^2 Lambda (L L')^-1 Lambda'`` [EQ-LMM-005], marginal and
    conditional fitted values and residuals, log-likelihood, AIC and BIC,
    Satterthwaite denominator degrees of freedom for t tests (Kuznetsova et
    al. 2017) [EQ-LMM-006], and likelihood-ratio tests between nested models
    with chi-square, boundary-mixture, and parametric-bootstrap p-values
    [EQ-LMM-007].

Generalized linear mixed model
    Binomial (logit) and Poisson (log) responses with
    ``g(mu) = X beta + offset + Z Lambda u``, ``u ~ N(0, I)``. The marginal
    likelihood ``int p(y | u) phi(u) du`` is approximated by the Laplace
    method around the conditional mode found by penalized IRLS
    [EQ-GLMM-001] or, for a single scalar random effect per group, by
    adaptive Gauss-Hermite quadrature centred and scaled at each group's mode
    [EQ-GLMM-002] (Liu and Pierce 1994; Pinheiro and Bates 1995); one node
    reproduces the Laplace value. ``theta`` and ``beta`` are optimized jointly
    and the fixed-effect covariance is twice the inverse Hessian of the
    deviance.

Nested factors are written as separate terms whose inner factor is labelled
by the pair (outer level, inner level), the expansion of lme4's
``(1 | school/class)`` into ``(1 | school) + (1 | school:class)``.
Uncorrelated intercepts and slopes are two terms on the same factor.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import (
    DEFAULT_RANK_TOLERANCE,
    Matrix,
    Vector,
    as_vector,
    cholesky,
    inverse_spd,
    solve_lower,
    solve_upper,
    symmetric_eigen,
    transpose,
)
from probstat.optimization import (
    OptimizationResult,
    central_hessian,
    central_jacobian,
    minimize_bfgs,
    minimize_nelder_mead,
)

from .bayesian import student_t_quantile
from .fitting import information_criteria
from .glm import _validate_response, fit_glm, glm_family
from .linear_models import (
    DesignMatrix,
    _coerce_design,
    _factorize,
    _positive_weights,
    _validate_level,
)
from .sampling_distributions import (
    chi_square_sf,
    standard_normal_sf,
    student_t_two_sided_p,
)

MixedMethod = Literal["REML", "ML"]
DenominatorDF = Literal["satterthwaite", "asymptotic"]
OptimizerName = Literal["bfgs", "nelder-mead"]
GLMMMethod = Literal["laplace", "adaptive_quadrature"]

_STANDARD_NORMAL = NormalDist()
_SINGULAR_TOLERANCE = 1e-4
_MAX_QUADRATURE_ORDER = 64
_LOG_SQRT_PI = 0.5 * math.log(math.pi)
_LOG_TWO_PI = math.log(2.0 * math.pi)


class MixedModelConvergenceWarning(UserWarning):
    """Warns that the variance-parameter optimizer stopped without converging.

    References:
        [REF-BATES-ETAL-2015, Section 3.4].

    """


# ---------------------------------------------------------------------------
# Random-effect terms
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RandomEffectTerm:
    """One random-effect term: a grouping factor and its effect columns.

    Attributes:
        name: Name of the grouping factor (used in reports).
        groups: Level of every observation (hashable labels).
        columns: ``n x r`` values of the effect columns (the first is all
            ones when the term has an intercept).
        column_names: Names of the ``r`` columns.
        levels: Distinct levels in order of first appearance.
        level_index: Position in ``levels`` of every observation's level.

    References:
        [REF-BATES-ETAL-2015, Section 2.1], [REF-LAIRD-WARE-1982].

    """

    name: str
    groups: tuple[Hashable, ...]
    columns: Matrix
    column_names: tuple[str, ...]
    levels: tuple[Hashable, ...]
    level_index: tuple[int, ...]

    @property
    def size(self) -> int:
        """Return the number of effect columns ``r``.

        Returns:
            Columns per level.

        References:
            [REF-BATES-ETAL-2015, Section 2.1]. No equation is evaluated.

        """
        return len(self.column_names)

    @property
    def n_levels(self) -> int:
        """Return the number of levels ``l`` of the grouping factor.

        Returns:
            Distinct levels.

        References:
            [REF-BATES-ETAL-2015, Section 2.1]. No equation is evaluated.

        """
        return len(self.levels)


def random_effect(
    groups: Iterable[Hashable],
    *,
    slopes: Mapping[str, Iterable[object]] | None = None,
    intercept: bool = True,
    name: str = "group",
) -> RandomEffectTerm:
    """Build a random-effect term with correlated intercept and slopes.

    ``random_effect(subject, slopes={"days": days})`` is lme4's
    ``(days | subject)``; ``intercept=False`` gives ``(0 + days | subject)``.
    A nested factor is labelled by pairs, for example
    ``random_effect(list(zip(school, klass)), name="school:class")``.

    Args:
        groups: Level of each observation; labels must be hashable.
        slopes: Optional mapping from covariate name to its values; each
            covariate gets a random slope correlated with the other columns.
        intercept: Whether the term includes a random intercept.
        name: Name of the grouping factor.

    Returns:
        :class:`RandomEffectTerm`.

    Raises:
        ValueError: If no column is requested, lengths differ, a covariate is
            not finite, or a column name repeats.

    Equation:
        [EQ-LMM-001].

    References:
        [REF-BATES-ETAL-2015, Section 2.1], [REF-LAIRD-WARE-1982].

    """
    labels = tuple(groups)
    n = len(labels)
    if n == 0:
        raise ValueError("groups must not be empty")
    names: list[str] = ["(Intercept)"] if intercept else []
    values: list[Vector] = [tuple(1.0 for _ in labels)] if intercept else []
    for covariate, raw in (slopes or {}).items():
        column = as_vector(raw, name=f"slope {covariate!r}")
        if len(column) != n:
            raise ValueError(f"slope {covariate!r} must have one value per observation")
        names.append(str(covariate))
        values.append(column)
    if not names:
        raise ValueError("a random-effect term needs an intercept or a slope")
    if len(set(names)) != len(names):
        raise ValueError("random-effect column names must be unique")
    index: dict[Hashable, int] = {}
    positions: list[int] = []
    for label in labels:
        positions.append(index.setdefault(label, len(index)))
    rows = tuple(tuple(column[i] for column in values) for i in range(n))
    return RandomEffectTerm(
        name=str(name),
        groups=labels,
        columns=rows,
        column_names=tuple(names),
        levels=tuple(index),
        level_index=tuple(positions),
    )


# ---------------------------------------------------------------------------
# Shared structure: sparse Z, block layout, envelope Cholesky
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Structure:
    """Index layout of ``Z``, ``Lambda`` and ``theta`` for a list of terms.

    ``blocks`` lists ``(start, size, term)`` for every (term, level) pair in
    the order of ``u``; ``pairs`` the block pairs ``(a, c)`` with ``a <= c``
    whose cross-product block is not identically zero; ``first[i]`` is the
    first column of row ``i``'s envelope.

    References:
        [REF-BATES-ETAL-2015, Section 3.1].

    """

    n: int
    q: int
    terms: tuple[RandomEffectTerm, ...]
    theta_offsets: tuple[int, ...]
    theta_size: int
    diagonal_theta: tuple[int, ...]
    z_rows: tuple[tuple[tuple[int, float], ...], ...]
    blocks: tuple[tuple[int, int, int], ...]
    pairs: tuple[tuple[int, int], ...]
    first: tuple[int, ...]


def _structure(terms: Sequence[RandomEffectTerm], n: int) -> _Structure:
    """Lay out ``u``, ``theta`` and the sparse rows of ``Z``.

    References:
        [EQ-LMM-001], [REF-BATES-ETAL-2015, Section 2.3].

    """
    blocks: list[tuple[int, int, int]] = []
    theta_offsets: list[int] = []
    diagonal: list[int] = []
    term_block_start: list[int] = []
    start = 0
    theta_count = 0
    for k, term in enumerate(terms):
        if len(term.groups) != n:
            raise ValueError(
                f"random-effect term {term.name!r} must have one level per observation"
            )
        r = term.size
        term_block_start.append(len(blocks))
        for _ in range(term.n_levels):
            blocks.append((start, r, k))
            start += r
        theta_offsets.append(theta_count)
        index = theta_count
        for c in range(r):
            diagonal.append(index)
            index += r - c
        theta_count += r * (r + 1) // 2
    q = start
    z_rows: list[tuple[tuple[int, float], ...]] = []
    observation_blocks: list[tuple[int, ...]] = []
    for i in range(n):
        entries: list[tuple[int, float]] = []
        owned: list[int] = []
        for k, term in enumerate(terms):
            block = term_block_start[k] + term.level_index[i]
            owned.append(block)
            offset = blocks[block][0]
            for c, value in enumerate(term.columns[i]):
                if value != 0.0:
                    entries.append((offset + c, value))
        z_rows.append(tuple(entries))
        observation_blocks.append(tuple(owned))
    pair_set: set[tuple[int, int]] = set()
    for blocks_of_row in observation_blocks:
        for a in blocks_of_row:
            for c in blocks_of_row:
                pair_set.add((min(a, c), max(a, c)))
    pairs = tuple(sorted(pair_set))
    first = [0] * q
    for block_start, size, _ in blocks:
        for i in range(block_start, block_start + size):
            first[i] = block_start
    for a, c in pairs:
        a_start = blocks[a][0]
        c_start, c_size, _ = blocks[c]
        for i in range(c_start, c_start + c_size):
            first[i] = min(first[i], a_start)
    return _Structure(
        n=n,
        q=q,
        terms=tuple(terms),
        theta_offsets=tuple(theta_offsets),
        theta_size=theta_count,
        diagonal_theta=tuple(diagonal),
        z_rows=tuple(z_rows),
        blocks=tuple(blocks),
        pairs=pairs,
        first=tuple(first),
    )


def _relative_factors(
    structure: _Structure, theta: Sequence[float]
) -> list[list[list[float]]]:
    """Return the lower-triangular ``T_k`` of every term from ``theta``.

    ``theta`` holds each ``T_k`` column by column (lower triangle), the
    lme4 order.

    References:
        [EQ-LMM-001], [REF-BATES-ETAL-2015, Section 3.1].

    """
    factors: list[list[list[float]]] = []
    for k, term in enumerate(structure.terms):
        r = term.size
        index = structure.theta_offsets[k]
        matrix = [[0.0] * r for _ in range(r)]
        for c in range(r):
            for row in range(c, r):
                matrix[row][c] = float(theta[index])
                index += 1
        factors.append(matrix)
    return factors


def _initial_theta(structure: _Structure) -> list[float]:
    """Return lme4's starting value: ones on the diagonal, zeros elsewhere.

    References:
        [REF-BATES-ETAL-2015, Section 3.4]. No equation is evaluated.

    """
    theta = [0.0] * structure.theta_size
    for index in structure.diagonal_theta:
        theta[index] = 1.0
    return theta


def _normalize_theta(structure: _Structure, theta: Sequence[float]) -> list[float]:
    """Flip columns of each ``T_k`` so that its diagonal is non-negative.

    ``T T'`` is unchanged by negating a column, so the optimizer may work in
    unconstrained coordinates and the result is mapped to lme4's convention.

    References:
        [EQ-LMM-001], [REF-BATES-ETAL-2015, Section 3.1].

    """
    normalized = [float(value) for value in theta]
    for k, term in enumerate(structure.terms):
        r = term.size
        index = structure.theta_offsets[k]
        for c in range(r):
            if normalized[index] < 0.0:
                for offset in range(r - c):
                    normalized[index + offset] = -normalized[index + offset]
            index += r - c
    return normalized


def _penalized_cross(
    structure: _Structure,
    cross: list[list[float]],
    factors: list[list[list[float]]],
) -> list[list[float]]:
    """Return the lower triangle of ``Lambda' C Lambda + I`` for symmetric ``C``.

    References:
        [EQ-LMM-002], [REF-BATES-ETAL-2015, Section 3.2].

    """
    q = structure.q
    result = [[0.0] * q for _ in range(q)]
    for a, c in structure.pairs:
        a_start, a_size, a_term = structure.blocks[a]
        c_start, c_size, c_term = structure.blocks[c]
        ta = factors[a_term]
        tc = factors[c_term]
        product = [
            [
                math.fsum(
                    cross[c_start + row][a_start + i] * ta[i][col]
                    for i in range(col, a_size)
                )
                for col in range(a_size)
            ]
            for row in range(c_size)
        ]
        for d in range(c_size):
            for col in range(a_size):
                if a == c and col > d:
                    continue
                result[c_start + d][a_start + col] = math.fsum(
                    tc[row][d] * product[row][col] for row in range(d, c_size)
                )
    for i in range(q):
        result[i][i] += 1.0
    return result


def _envelope_cholesky(matrix: list[list[float]], first: Sequence[int]) -> None:
    """Overwrite the lower triangle of an SPD matrix with its Cholesky factor.

    Only entries inside the envelope ``first[i] <= j <= i`` are read or
    written; the factor of a matrix has no fill outside its envelope.

    Raises:
        ValueError: If the matrix is not numerically positive definite.

    References:
        [EQ-LINALG-001], [REF-GOLUB-VANLOAN-2013, Section 4.3].

    """
    for i, row in enumerate(matrix):
        fi = first[i]
        for j in range(fi, i + 1):
            other = matrix[j]
            start = max(fi, first[j])
            total = row[j] - sum(row[k] * other[k] for k in range(start, j))
            if j == i:
                if not total > 0.0:
                    raise ValueError("penalized cross-product is not positive definite")
                row[i] = math.sqrt(total)
            else:
                row[j] = total / other[j]


def _forward(
    lower: list[list[float]], first: Sequence[int], rhs: Sequence[float]
) -> list[float]:
    """Solve ``L x = b`` for an envelope lower factor.

    References:
        [EQ-LINALG-003], [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    x = [float(value) for value in rhs]
    for i, row in enumerate(lower):
        x[i] = (x[i] - sum(row[k] * x[k] for k in range(first[i], i))) / row[i]
    return x


def _backward(
    lower: list[list[float]], first: Sequence[int], rhs: Sequence[float]
) -> list[float]:
    """Solve ``L' x = b`` for an envelope lower factor (column oriented).

    References:
        [EQ-LINALG-003], [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    x = [float(value) for value in rhs]
    for i in range(len(lower) - 1, -1, -1):
        row = lower[i]
        x[i] /= row[i]
        value = x[i]
        for k in range(first[i], i):
            x[k] -= row[k] * value
    return x


def _lambda_transpose(
    structure: _Structure, factors: list[list[list[float]]], vector: Sequence[float]
) -> list[float]:
    """Return ``Lambda' v`` block by block.

    References:
        [EQ-LMM-001], [REF-BATES-ETAL-2015, Section 3.1].

    """
    out = [0.0] * structure.q
    for start, size, term in structure.blocks:
        t = factors[term]
        for col in range(size):
            out[start + col] = math.fsum(
                t[row][col] * vector[start + row] for row in range(col, size)
            )
    return out


def _lambda(
    structure: _Structure, factors: list[list[list[float]]], vector: Sequence[float]
) -> list[float]:
    """Return ``Lambda v`` block by block.

    References:
        [EQ-LMM-001], [REF-BATES-ETAL-2015, Section 3.1].

    """
    out = [0.0] * structure.q
    for start, size, term in structure.blocks:
        t = factors[term]
        for row in range(size):
            out[start + row] = math.fsum(
                t[row][col] * vector[start + col] for col in range(row + 1)
            )
    return out


def _z_times(structure: _Structure, vector: Sequence[float]) -> list[float]:
    """Return ``Z v`` from the sparse rows of ``Z``.

    References:
        [EQ-LMM-001]. No other equation is evaluated.

    """
    return [sum(value * vector[col] for col, value in row) for row in structure.z_rows]


def _z_cross(
    structure: _Structure, weights: Sequence[float] | None
) -> list[list[float]]:
    """Return the dense symmetric ``Z' W Z`` (``W = I`` when ``weights`` is None).

    References:
        [EQ-LMM-002]. No other equation is evaluated.

    """
    q = structure.q
    cross = [[0.0] * q for _ in range(q)]
    for i, row in enumerate(structure.z_rows):
        w = 1.0 if weights is None else weights[i]
        for a, va in row:
            target = cross[a]
            for c, vc in row:
                target[c] += w * va * vc
    return cross


def _block_inverse_diagonals(
    structure: _Structure, lower: list[list[float]]
) -> list[list[list[float]]]:
    """Return the diagonal blocks of ``(L L')^-1``, one per (term, level).

    References:
        [EQ-LMM-005], [REF-BATES-ETAL-2015, Section 3.6].

    """
    q = structure.q
    out: list[list[list[float]]] = []
    for start, size, _ in structure.blocks:
        columns = []
        for col in range(size):
            unit = [0.0] * q
            unit[start + col] = 1.0
            solution = _backward(
                lower, structure.first, _forward(lower, structure.first, unit)
            )
            columns.append(solution[start : start + size])
        out.append([[columns[c][r] for c in range(size)] for r in range(size)])
    return out


def _sandwich_block(
    t: list[list[float]], middle: list[list[float]], scale: float
) -> Matrix:
    """Return ``scale T M T'`` for a small block.

    References:
        [EQ-LMM-005]. No other equation is evaluated.

    """
    size = len(t)
    left = [
        [math.fsum(t[i][k] * middle[k][j] for k in range(size)) for j in range(size)]
        for i in range(size)
    ]
    return tuple(
        tuple(
            scale * math.fsum(left[i][k] * t[j][k] for k in range(size))
            for j in range(size)
        )
        for i in range(size)
    )


# ---------------------------------------------------------------------------
# Result records shared by LMM and GLMM fits
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VarianceComponent:
    """Estimated covariance of one random-effect term.

    Attributes:
        group: Grouping-factor name.
        column_names: Effect columns of the term.
        covariance: ``r x r`` covariance ``sigma^2 T T'`` (``T T'`` for a
            GLMM).
        standard_deviations: Square roots of the diagonal.
        correlations: Correlation matrix (``nan`` where a standard deviation
            is zero).

    References:
        [REF-BATES-ETAL-2015, Section 3.1], [EQ-LMM-001].

    """

    group: str
    column_names: tuple[str, ...]
    covariance: Matrix
    standard_deviations: Vector
    correlations: Matrix


@dataclass(frozen=True, slots=True)
class RandomEffectEstimates:
    """Conditional modes and conditional covariances of one term.

    Attributes:
        group: Grouping-factor name.
        column_names: Effect columns of the term.
        levels: Levels in order of first appearance.
        values: One row of conditional modes (BLUPs for an LMM) per level.
        conditional_covariances: One ``r x r`` conditional covariance per
            level, given the data and the estimated parameters.

    References:
        [REF-HENDERSON-1975], [REF-BATES-ETAL-2015, Section 3.6].

    """

    group: str
    column_names: tuple[str, ...]
    levels: tuple[Hashable, ...]
    values: Matrix
    conditional_covariances: tuple[Matrix, ...]

    def for_level(self, level: Hashable) -> Vector:
        """Return the conditional modes of one level.

        Args:
            level: A level of the grouping factor.

        Returns:
            One value per effect column.

        Raises:
            KeyError: If the level is unknown.

        References:
            [REF-HENDERSON-1975]. No equation is evaluated.

        """
        for candidate, row in zip(self.levels, self.values, strict=True):
            if candidate == level:
                return row
        raise KeyError(f"unknown level {level!r} of {self.group!r}")


def _variance_components(
    structure: _Structure, factors: list[list[list[float]]], scale: float
) -> tuple[VarianceComponent, ...]:
    """Return ``scale T_k T_k'`` with standard deviations and correlations.

    References:
        [EQ-LMM-001], [REF-BATES-ETAL-2015, Section 3.1].

    """
    components: list[VarianceComponent] = []
    for term, t in zip(structure.terms, factors, strict=True):
        r = term.size
        identity = [[1.0 if i == j else 0.0 for j in range(r)] for i in range(r)]
        covariance = _sandwich_block(t, identity, scale)
        sds = tuple(math.sqrt(max(covariance[i][i], 0.0)) for i in range(r))
        correlations = tuple(
            tuple(
                (1.0 if i == j else covariance[i][j] / (sds[i] * sds[j]))
                if sds[i] > 0.0 and sds[j] > 0.0
                else (1.0 if i == j else math.nan)
                for j in range(r)
            )
            for i in range(r)
        )
        components.append(
            VarianceComponent(
                term.name, term.column_names, covariance, sds, correlations
            )
        )
    return tuple(components)


def _random_effect_estimates(
    structure: _Structure,
    factors: list[list[list[float]]],
    u: Sequence[float],
    lower: list[list[float]],
    scale: float,
) -> tuple[RandomEffectEstimates, ...]:
    """Return ``b = Lambda u`` and ``scale Lambda (L L')^-1 Lambda'`` per level.

    References:
        [EQ-LMM-005], [REF-BATES-ETAL-2015, Section 3.6].

    """
    b = _lambda(structure, factors, u)
    inverse_blocks = _block_inverse_diagonals(structure, lower)
    estimates: list[RandomEffectEstimates] = []
    block = 0
    for k, term in enumerate(structure.terms):
        values: list[Vector] = []
        covariances: list[Matrix] = []
        for _ in range(term.n_levels):
            start, size, _ = structure.blocks[block]
            values.append(tuple(b[start : start + size]))
            covariances.append(
                _sandwich_block(factors[k], inverse_blocks[block], scale)
            )
            block += 1
        estimates.append(
            RandomEffectEstimates(
                term.name,
                term.column_names,
                term.levels,
                tuple(values),
                tuple(covariances),
            )
        )
    return tuple(estimates)


def _is_singular(structure: _Structure, theta: Sequence[float]) -> bool:
    """Return whether some diagonal of a ``T_k`` is below ``1e-4`` (lme4 rule).

    References:
        [REF-BATES-ETAL-2015, Section 3.1]. No equation is evaluated.

    """
    return any(abs(theta[i]) < _SINGULAR_TOLERANCE for i in structure.diagonal_theta)


def _run_optimizer(
    objective: Callable[[Vector], float],
    start: Sequence[float],
    optimizer: str,
    tolerance: float,
    max_iterations: int,
    typical_scale: Sequence[float] | None,
) -> OptimizationResult:
    """Run one BFGS or Nelder-Mead minimization.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 6], [REF-BATES-ETAL-2015, Section 3.4].

    """
    if optimizer == "bfgs":
        return minimize_bfgs(
            objective,
            start,
            gradient_tolerance=tolerance,
            max_iterations=max_iterations,
            typical_scale=1.0 if typical_scale is None else typical_scale,
        )
    return minimize_nelder_mead(
        objective,
        start,
        initial_step=0.25,
        tolerance=tolerance,
        max_evaluations=200 * max_iterations,
    )


def _saddle_escape(
    objective: Callable[[Vector], float],
    x: Sequence[float],
    diagonal: Sequence[int],
) -> list[float] | None:
    """Return a lower point next to a zero diagonal ``theta``, if one exists.

    The deviance is even in each column of ``T_k``, so ``theta_ii = 0`` is
    always a stationary point of the unconstrained objective. It is a
    boundary minimum or a local maximum between the two mirror-image optima;
    a gradient-based optimizer that lands on it stops there. Trial points
    ``theta_ii = 0.01, 0.1, 1`` distinguish the two cases.

    References:
        [REF-BATES-ETAL-2015, Section 3.4], [EQ-LMM-003].

    """
    base = objective(tuple(x))
    for index in diagonal:
        if abs(x[index]) >= _SINGULAR_TOLERANCE:
            continue
        for trial in (0.01, 0.1, 1.0):
            candidate = [float(v) for v in x]
            candidate[index] = trial
            try:
                value = objective(tuple(candidate))
            except ValueError:
                continue
            if value < base - 1e-9 * (1.0 + abs(base)):
                return candidate
    return None


def _minimize(
    objective: Callable[[Vector], float],
    start: Sequence[float],
    optimizer: str,
    tolerance: float,
    max_iterations: int,
    diagonal: Sequence[int],
    typical_scale: Sequence[float] | None = None,
) -> OptimizationResult:
    """Minimize the deviance, restarting away from false boundary stationary points.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 6], [REF-BATES-ETAL-2015, Section 3.4].

    """
    result = _run_optimizer(
        objective, start, optimizer, tolerance, max_iterations, typical_scale
    )
    iterations, evaluations = result.iterations, result.evaluations
    for _ in range(3):
        restart = _saddle_escape(objective, result.x, diagonal)
        evaluations += 1 + 3 * len(diagonal)
        if restart is None:
            break
        result = _run_optimizer(
            objective, restart, optimizer, tolerance, max_iterations, typical_scale
        )
        iterations += result.iterations
        evaluations += result.evaluations
    return OptimizationResult(
        x=result.x,
        value=result.value,
        gradient_norm=result.gradient_norm,
        iterations=iterations,
        evaluations=evaluations,
        converged=result.converged,
        message=result.message,
    )


# ---------------------------------------------------------------------------
# Linear mixed models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LMMData:
    """Cross products that do not depend on ``theta``.

    References:
        [EQ-LMM-002], [REF-BATES-ETAL-2015, Section 3.2].

    """

    structure: _Structure
    x: Matrix
    y: Vector
    ztz: list[list[float]]
    ztx: list[list[float]]
    zty: list[float]
    xtx: list[list[float]]
    xty: list[float]


@dataclass(frozen=True, slots=True)
class _LMMState:
    """Solution of the penalized least-squares problem at one ``theta``.

    References:
        [EQ-LMM-002], [REF-BATES-ETAL-2015, Section 3.2].

    """

    factors: list[list[list[float]]]
    lower: list[list[float]]
    rx: Matrix
    beta: Vector
    u: list[float]
    penalized_rss: float
    log_det_l: float
    log_det_x: float


def _lmm_data(structure: _Structure, x: Matrix, y: Vector) -> _LMMData:
    """Precompute ``Z'Z``, ``Z'X``, ``Z'y``, ``X'X`` and ``X'y``.

    References:
        [EQ-LMM-002]. No other equation is evaluated.

    """
    q, p = structure.q, len(x[0])
    ztx = [[0.0] * p for _ in range(q)]
    zty = [0.0] * q
    for row, xi, yi in zip(structure.z_rows, x, y, strict=True):
        for a, value in row:
            target = ztx[a]
            for j in range(p):
                target[j] += value * xi[j]
            zty[a] += value * yi
    xtx = [
        [math.fsum(row[j] * row[k] for row in x) for k in range(p)] for j in range(p)
    ]
    xty = [
        math.fsum(row[j] * yi for row, yi in zip(x, y, strict=True)) for j in range(p)
    ]
    return _LMMData(structure, x, y, _z_cross(structure, None), ztx, zty, xtx, xty)


def _lmm_state(data: _LMMData, theta: Sequence[float]) -> _LMMState:
    """Solve the penalized least-squares problem by block Cholesky.

    References:
        [EQ-LMM-002], [REF-BATES-ETAL-2015, Section 3.2].

    """
    structure = data.structure
    first = structure.first
    p = len(data.xty)
    factors = _relative_factors(structure, theta)
    lower = _penalized_cross(structure, data.ztz, factors)
    _envelope_cholesky(lower, first)
    cu = _forward(lower, first, _lambda_transpose(structure, factors, data.zty))
    rzx = [
        _forward(
            lower,
            first,
            _lambda_transpose(structure, factors, [row[j] for row in data.ztx]),
        )
        for j in range(p)
    ]
    schur = tuple(
        tuple(
            data.xtx[j][k]
            - math.fsum(a * b for a, b in zip(rzx[j], rzx[k], strict=True))
            for k in range(p)
        )
        for j in range(p)
    )
    rx = cholesky(schur)
    rhs = tuple(
        data.xty[j] - math.fsum(a * b for a, b in zip(rzx[j], cu, strict=True))
        for j in range(p)
    )
    beta = solve_upper(transpose(rx), solve_lower(rx, rhs))
    target = [
        cu[i] - math.fsum(rzx[j][i] * beta[j] for j in range(p))
        for i in range(structure.q)
    ]
    u = _backward(lower, first, target)
    zb = _z_times(structure, _lambda(structure, factors, u))
    residuals = (
        yi - math.fsum(b * x for b, x in zip(beta, row, strict=True)) - zi
        for yi, row, zi in zip(data.y, data.x, zb, strict=True)
    )
    penalized = math.fsum(r * r for r in residuals) + math.fsum(v * v for v in u)
    log_det_l = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(structure.q))
    log_det_x = 2.0 * math.fsum(math.log(rx[j][j]) for j in range(p))
    return _LMMState(factors, lower, rx, beta, u, penalized, log_det_l, log_det_x)


def _profiled_deviance(state: _LMMState, n: int, p: int, reml: bool) -> float:
    """Return the profiled ML or REML deviance of a penalized LS solution.

    References:
        [EQ-LMM-003], [EQ-LMM-004], [REF-BATES-ETAL-2015, Section 3.4].

    """
    if not state.penalized_rss > 0.0:
        raise ValueError("penalized residual sum of squares is zero")
    if reml:
        degrees = n - p
        return (
            state.log_det_l
            + state.log_det_x
            + degrees * (1.0 + math.log(2.0 * math.pi * state.penalized_rss / degrees))
        )
    return state.log_det_l + n * (
        1.0 + math.log(2.0 * math.pi * state.penalized_rss / n)
    )


def _full_deviance(state: _LMMState, sigma: float, n: int, p: int, reml: bool) -> float:
    """Return the deviance as a function of ``(theta, sigma)`` (lmerTest form).

    References:
        [EQ-LMM-006], [REF-KUZNETSOVA-ETAL-2017, Section 3].

    """
    variance = sigma * sigma
    value = (
        state.log_det_l
        + state.penalized_rss / variance
        + n * math.log(2.0 * math.pi * variance)
    )
    if reml:
        value += state.log_det_x - p * math.log(2.0 * math.pi * variance)
    return value


def _beta_covariance(state: _LMMState, variance: float) -> Matrix:
    """Return ``sigma^2 (R_X' R_X)^-1``.

    References:
        [EQ-LMM-005], [REF-BATES-ETAL-2015, Section 3.5].

    """
    p = len(state.rx)
    upper = transpose(state.rx)
    columns = []
    for j in range(p):
        unit = tuple(1.0 if i == j else 0.0 for i in range(p))
        columns.append(solve_upper(upper, solve_lower(state.rx, unit)))
    return tuple(tuple(variance * columns[k][j] for k in range(p)) for j in range(p))


@dataclass(frozen=True, slots=True)
class ContrastTest:
    """t test of a linear combination ``c' beta`` of fixed effects.

    Attributes:
        estimate: ``c' beta``.
        standard_error: ``sqrt(c' Cov(beta) c)``.
        statistic: ``estimate / standard_error``.
        degrees_of_freedom: Satterthwaite denominator df (``inf`` for the
            asymptotic normal reference).
        p_value: Two-sided p-value.

    References:
        [REF-KUZNETSOVA-ETAL-2017, Section 3.2], [EQ-LMM-006].

    """

    estimate: float
    standard_error: float
    statistic: float
    degrees_of_freedom: float
    p_value: float


@dataclass(frozen=True, slots=True)
class LinearMixedModelResult:
    """Fit of a linear mixed model by ML or REML.

    Attributes:
        method: ``"REML"`` or ``"ML"``.
        coefficients: Fixed effects ``beta``.
        standard_errors: ``sqrt(diag(Cov(beta)))``.
        statistics: t ratios ``beta_j / se_j``.
        degrees_of_freedom: Satterthwaite denominator df per coefficient
            (``inf`` with ``denominator_df="asymptotic"``; ``nan`` when the
            variance-parameter Hessian is not positive definite).
        p_values: Two-sided p-values (Student t on the Satterthwaite df,
            normal when the df is infinite or unavailable).
        column_names: Fixed-effect column names.
        covariance: ``sigma^2 (R_X' R_X)^-1``.
        residual_variance: ``sigma^2`` (``r^2/n`` for ML, ``r^2/(n-p)`` for
            REML).
        theta: Relative covariance factor parameters (lme4 order, diagonal
            entries non-negative).
        variance_components: One :class:`VarianceComponent` per term.
        random_effects: One :class:`RandomEffectEstimates` per term.
        fitted_values: Conditional fitted values ``X beta + Z b``.
        marginal_fitted_values: ``X beta``.
        residuals: Conditional residuals ``y - X beta - Z b``.
        marginal_residuals: ``y - X beta``.
        log_likelihood: Maximized log-likelihood (the REML criterion
            ``-d_R/2`` for REML fits).
        deviance: ``-2 log_likelihood``.
        aic: ``-2 log L + 2k``.
        bic: ``-2 log L + k log n``.
        parameter_count: ``k = p + len(theta) + 1``.
        variance_parameters: ``(theta, sigma)`` at the optimum.
        variance_parameter_covariance: ``2 H^-1`` of the deviance in
            ``(theta, sigma)`` (empty unless Satterthwaite df were computed).
        covariance_gradients: ``d Cov(beta) / d psi_m`` for each variance
            parameter (empty unless Satterthwaite df were computed).
        singular: Whether some ``T_k`` has a diagonal below ``1e-4`` (a
            boundary estimate).
        converged: Whether the optimizer met its tolerance.
        iterations: Optimizer iterations.
        evaluations: Deviance evaluations.
        message: Optimizer termination message.
        design: Fixed-effect design rows.
        response: Responses.
        terms: Random-effect terms.
        n_observations: ``n``.

    References:
        [REF-BATES-ETAL-2015], [REF-KUZNETSOVA-ETAL-2017],
        [REF-PATTERSON-THOMPSON-1971].

    """

    method: str
    coefficients: Vector
    standard_errors: Vector
    statistics: Vector
    degrees_of_freedom: Vector
    p_values: Vector
    column_names: tuple[str, ...]
    covariance: Matrix
    residual_variance: float
    theta: Vector
    variance_components: tuple[VarianceComponent, ...]
    random_effects: tuple[RandomEffectEstimates, ...]
    fitted_values: Vector
    marginal_fitted_values: Vector
    residuals: Vector
    marginal_residuals: Vector
    log_likelihood: float
    deviance: float
    aic: float
    bic: float
    parameter_count: int
    variance_parameters: Vector
    variance_parameter_covariance: Matrix
    covariance_gradients: tuple[Matrix, ...]
    singular: bool
    converged: bool
    iterations: int
    evaluations: int
    message: str
    design: Matrix = field(repr=False)
    response: Vector = field(repr=False)
    terms: tuple[RandomEffectTerm, ...] = field(repr=False)
    n_observations: int = 0

    @property
    def residual_sd(self) -> float:
        """Return the residual standard deviation ``sigma``.

        Returns:
            ``sqrt(residual_variance)``.

        References:
            [EQ-LMM-003]. No other equation is evaluated.

        """
        return math.sqrt(self.residual_variance)

    def coefficient(self, name: str) -> float:
        """Return one fixed effect by column name.

        Args:
            name: Fixed-effect column name.

        Returns:
            The estimate.

        Raises:
            KeyError: If the name is unknown.

        References:
            [EQ-LMM-002]. No other equation is evaluated.

        """
        if name not in self.column_names:
            raise KeyError(f"unknown coefficient {name!r}")
        return self.coefficients[self.column_names.index(name)]

    def coefficient_intervals(
        self, level: float = 0.95
    ) -> tuple[tuple[float, float], ...]:
        """Return t intervals on the Satterthwaite df (normal when infinite).

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            One ``(lower, upper)`` pair per coefficient.

        Raises:
            ValueError: If ``level`` is invalid.

        Equation:
            [EQ-LMM-006].

        References:
            [REF-KUZNETSOVA-ETAL-2017, Section 3].

        """
        checked = _validate_level(level)
        upper = 0.5 * (1.0 + checked)
        out = []
        for b, s, df in zip(
            self.coefficients,
            self.standard_errors,
            self.degrees_of_freedom,
            strict=True,
        ):
            quantile = (
                student_t_quantile(upper, df)
                if math.isfinite(df)
                else _STANDARD_NORMAL.inv_cdf(upper)
            )
            out.append((b - quantile * s, b + quantile * s))
        return tuple(out)

    def test_contrast(self, contrast: Iterable[object]) -> ContrastTest:
        """Test ``c' beta = 0`` with a Satterthwaite t reference.

        Args:
            contrast: One weight per fixed effect.

        Returns:
            :class:`ContrastTest`; the df is ``inf`` when the fit did not
            compute Satterthwaite quantities and ``nan`` when they failed.

        Raises:
            ValueError: If the contrast has the wrong length or is zero.

        Equation:
            [EQ-LMM-006].

        References:
            [REF-KUZNETSOVA-ETAL-2017, Section 3.2], [REF-SATTERTHWAITE-1946].

        """
        c = as_vector(contrast, name="contrast")
        p = len(self.coefficients)
        if len(c) != p:
            raise ValueError("contrast must have one weight per coefficient")
        if all(value == 0.0 for value in c):
            raise ValueError("contrast must not be zero")
        estimate = math.fsum(a * b for a, b in zip(c, self.coefficients, strict=True))
        variance = _quadratic(self.covariance, c)
        error = math.sqrt(max(variance, 0.0))
        statistic = (
            estimate / error if error > 0.0 else math.copysign(math.inf, estimate)
        )
        if self.covariance_gradients:
            gradient = [_quadratic(g, c) for g in self.covariance_gradients]
            df = _satterthwaite(variance, gradient, self.variance_parameter_covariance)
        else:
            finite = all(math.isinf(d) for d in self.degrees_of_freedom)
            df = math.inf if finite else math.nan
        return ContrastTest(estimate, error, statistic, df, _two_sided_p(statistic, df))


def _quadratic(matrix: Matrix, vector: Sequence[float]) -> float:
    """Return ``v' M v``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return math.fsum(
        vector[i] * matrix[i][j] * vector[j]
        for i in range(len(vector))
        for j in range(len(vector))
    )


def _satterthwaite(
    variance: float, gradient: Sequence[float], covariance: Matrix
) -> float:
    """Return ``2 V^2 / (g' A g)`` or ``nan`` when it is undefined.

    References:
        [EQ-LMM-006], [REF-SATTERTHWAITE-1946], [REF-KUZNETSOVA-ETAL-2017].

    """
    if not covariance:
        return math.nan
    denominator = _quadratic(covariance, gradient)
    if not (denominator > 0.0 and math.isfinite(denominator)):
        return math.nan
    return 2.0 * variance * variance / denominator


def _two_sided_p(statistic: float, df: float) -> float:
    """Return a two-sided p-value from Student t (finite df) or the normal law.

    References:
        [EQ-LMM-006], [REF-KUZNETSOVA-ETAL-2017]. No other equation is evaluated.

    """
    if not math.isfinite(statistic):
        return 0.0
    if math.isfinite(df) and df > 0.0:
        return student_t_two_sided_p(statistic, df)
    return 2.0 * standard_normal_sf(abs(statistic))


def _check_terms(
    random_effects: Sequence[RandomEffectTerm] | RandomEffectTerm, n: int
) -> tuple[RandomEffectTerm, ...]:
    """Validate the random-effect terms against the number of observations.

    References:
        [REF-BATES-ETAL-2015, Section 2.3]. No equation is evaluated.

    """
    terms = (
        (random_effects,)
        if isinstance(random_effects, RandomEffectTerm)
        else tuple(random_effects)
    )
    for term in terms:
        if not isinstance(term, RandomEffectTerm):
            raise TypeError("random_effects must be RandomEffectTerm objects")
        if len(term.groups) != n:
            raise ValueError(
                f"random-effect term {term.name!r} must have one level per observation"
            )
        if term.n_levels < 2:
            raise ValueError(f"grouping factor {term.name!r} needs at least two levels")
        if term.n_levels * term.size >= n:
            raise ValueError(
                f"term {term.name!r} has {term.n_levels * term.size} random "
                f"effects for "
                f"{n} observations; the model is not identifiable"
            )
    return terms


def fit_linear_mixed_model(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    random_effects: Sequence[RandomEffectTerm] | RandomEffectTerm,
    *,
    method: MixedMethod | str = "REML",
    column_names: Sequence[str] | None = None,
    denominator_df: DenominatorDF | str = "satterthwaite",
    optimizer: OptimizerName | str = "bfgs",
    tolerance: float = 1e-7,
    max_iterations: int = 500,
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> LinearMixedModelResult:
    """Fit ``y = X beta + Z b + epsilon`` by profiled ML or REML.

    The variance parameters ``theta`` are optimized on the profiled deviance
    of [EQ-LMM-003] (ML) or [EQ-LMM-004] (REML); ``beta``, ``sigma`` and the
    conditional modes follow from one penalized least-squares solve
    [EQ-LMM-002]. An empty ``random_effects`` fits the fixed-effects model,
    which is useful as the null model of a likelihood-ratio test.

    Args:
        design: :class:`DesignMatrix` or ``n x p`` fixed-effect rows.
        response: ``n`` finite responses.
        random_effects: One term or a sequence of terms from
            :func:`random_effect`.
        method: ``"REML"`` (default) or ``"ML"``.
        column_names: Names for a raw design.
        denominator_df: ``"satterthwaite"`` (default) for t tests on
            Satterthwaite df, or ``"asymptotic"`` for normal z tests.
        optimizer: ``"bfgs"`` (default) or ``"nelder-mead"``.
        tolerance: Gradient-norm (BFGS) or simplex (Nelder-Mead) tolerance.
        max_iterations: Positive optimizer iteration limit.
        rank_tolerance: Relative threshold for a dependent design column.

    Returns:
        :class:`LinearMixedModelResult`.

    Raises:
        RankDeficientDesignError: If the fixed-effect design is rank
            deficient.
        TypeError: If a term is not a :class:`RandomEffectTerm`.
        ValueError: For invalid options, mismatched lengths, a grouping
            factor with one level, or more random effects than observations.

    Warns:
        MixedModelConvergenceWarning: If the optimizer did not converge.

    Equation:
        [EQ-LMM-001], [EQ-LMM-002], [EQ-LMM-003], [EQ-LMM-004],
        [EQ-LMM-005], [EQ-LMM-006].

    References:
        [REF-BATES-ETAL-2015, Sections 2-3], [REF-KUZNETSOVA-ETAL-2017],
        [REF-PATTERSON-THOMPSON-1971], [REF-HENDERSON-1975].

    """
    if method not in ("REML", "ML"):
        raise ValueError("method must be 'REML' or 'ML'")
    if denominator_df not in ("satterthwaite", "asymptotic"):
        raise ValueError("denominator_df must be 'satterthwaite' or 'asymptotic'")
    if optimizer not in ("bfgs", "nelder-mead"):
        raise ValueError("optimizer must be 'bfgs' or 'nelder-mead'")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    rows, names, _ = _coerce_design(design, column_names)
    y = as_vector(response, name="response")
    n, p = len(rows), len(rows[0])
    if len(y) != n:
        raise ValueError("response length must equal the number of design rows")
    if n <= p:
        raise ValueError("a mixed model needs more observations than fixed effects")
    _factorize(rows, names, rank_tolerance)
    terms = _check_terms(random_effects, n)
    reml = method == "REML"
    structure = _structure(terms, n)
    data = _lmm_data(structure, rows, y)

    def objective(theta: Sequence[float]) -> float:
        """Profiled deviance at ``theta``.

        References:
            [EQ-LMM-003], [EQ-LMM-004].

        """
        return _profiled_deviance(_lmm_state(data, theta), n, p, reml)

    if structure.theta_size:
        optimum = _minimize(
            objective,
            _initial_theta(structure),
            optimizer,
            tolerance,
            max_iterations,
            structure.diagonal_theta,
        )
        theta = _normalize_theta(structure, optimum.x)
        converged, iterations = optimum.converged, optimum.iterations
        evaluations, message = optimum.evaluations, optimum.message
    else:
        theta = []
        converged, iterations, evaluations, message = (
            True,
            0,
            1,
            "no variance parameters",
        )
    if not converged:
        warnings.warn(
            f"mixed-model optimizer did not converge: {message}",
            MixedModelConvergenceWarning,
            stacklevel=2,
        )
    state = _lmm_state(data, theta)
    deviance = _profiled_deviance(state, n, p, reml)
    variance = state.penalized_rss / (n - p if reml else n)
    sigma = math.sqrt(variance)
    covariance = _beta_covariance(state, variance)
    errors = tuple(math.sqrt(max(covariance[j][j], 0.0)) for j in range(p))
    psi = (*theta, sigma)
    parameter_covariance: Matrix = ()
    gradients: tuple[Matrix, ...] = ()
    if denominator_df == "satterthwaite":
        parameter_covariance, gradients = _satterthwaite_inputs(data, psi, n, p, reml)
        dfs = tuple(
            _satterthwaite(
                covariance[j][j],
                [g[j][j] for g in gradients],
                parameter_covariance,
            )
            for j in range(p)
        )
    else:
        dfs = tuple(math.inf for _ in range(p))
    statistics = tuple(
        b / s if s > 0.0 else math.copysign(math.inf, b)
        for b, s in zip(state.beta, errors, strict=True)
    )
    p_values = tuple(_two_sided_p(t, df) for t, df in zip(statistics, dfs, strict=True))
    marginal = tuple(
        math.fsum(b * x for b, x in zip(state.beta, row, strict=True)) for row in rows
    )
    zb = _z_times(structure, _lambda(structure, state.factors, state.u))
    conditional = tuple(m + z for m, z in zip(marginal, zb, strict=True))
    log_likelihood = -0.5 * deviance
    k = p + len(theta) + 1
    aic, _, bic = information_criteria(log_likelihood, k, n)
    return LinearMixedModelResult(
        method=method,
        coefficients=state.beta,
        standard_errors=errors,
        statistics=statistics,
        degrees_of_freedom=dfs,
        p_values=p_values,
        column_names=names,
        covariance=covariance,
        residual_variance=variance,
        theta=tuple(theta),
        variance_components=_variance_components(structure, state.factors, variance),
        random_effects=_random_effect_estimates(
            structure, state.factors, state.u, state.lower, variance
        ),
        fitted_values=conditional,
        marginal_fitted_values=marginal,
        residuals=tuple(yi - c for yi, c in zip(y, conditional, strict=True)),
        marginal_residuals=tuple(yi - m for yi, m in zip(y, marginal, strict=True)),
        log_likelihood=log_likelihood,
        deviance=deviance,
        aic=aic,
        bic=bic,
        parameter_count=k,
        variance_parameters=psi,
        variance_parameter_covariance=parameter_covariance,
        covariance_gradients=gradients,
        singular=_is_singular(structure, theta),
        converged=converged,
        iterations=iterations,
        evaluations=evaluations,
        message=message,
        design=rows,
        response=y,
        terms=terms,
        n_observations=n,
    )


def _satterthwaite_inputs(
    data: _LMMData, psi: Sequence[float], n: int, p: int, reml: bool
) -> tuple[Matrix, tuple[Matrix, ...]]:
    """Return ``A = 2 H^-1`` and ``d Cov(beta)/d psi`` at ``psi = (theta, sigma)``.

    The Hessian ``H`` of the deviance in ``(theta, sigma)`` and the Jacobian
    of ``Cov(beta) = sigma^2 (R_X' R_X)^-1(theta)`` are central differences,
    as in lmerTest. An empty ``A`` signals a Hessian that is not positive
    definite.

    References:
        [EQ-LMM-006], [REF-KUZNETSOVA-ETAL-2017, Section 3].

    """
    m = len(psi)

    def deviance(values: Sequence[float]) -> float:
        """Deviance in ``(theta, sigma)``.

        References:
            [EQ-LMM-006].

        """
        state = _lmm_state(data, values[:-1])
        return _full_deviance(state, values[-1], n, p, reml)

    def flattened(values: Sequence[float]) -> Vector:
        """Row-major ``Cov(beta)`` at ``(theta, sigma)``.

        References:
            [EQ-LMM-005].

        """
        state = _lmm_state(data, values[:-1])
        covariance = _beta_covariance(state, values[-1] * values[-1])
        return tuple(value for row in covariance for value in row)

    scale = [max(abs(value), 0.1) for value in psi]
    hessian, _ = central_hessian(deviance, psi, typical_scale=scale)
    jacobian, _ = central_jacobian(flattened, psi, typical_scale=scale)
    gradients = tuple(
        tuple(tuple(jacobian[j * p + k][index] for k in range(p)) for j in range(p))
        for index in range(m)
    )
    try:
        inverse = inverse_spd(tuple(tuple(v for v in row) for row in hessian))
    except ValueError:
        return (), gradients
    return tuple(tuple(2.0 * v for v in row) for row in inverse), gradients


# ---------------------------------------------------------------------------
# Likelihood-ratio tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MixedModelLikelihoodRatioTest:
    """Likelihood-ratio comparison of two nested linear mixed models.

    Attributes:
        statistic: ``2 (log L_full - log L_reduced)``, floored at zero.
        degrees_of_freedom: Difference in parameter counts.
        p_value: ``P(chi^2_df >= statistic)``; conservative when the null
            places a variance on its boundary.
        boundary_mixture_p_value: ``0.5 P(chi^2_(df-1) >= T) + 0.5
            P(chi^2_df >= T)``, the Self-Liang / Stram-Lee reference when the
            full model adds one random effect (with its covariances) whose
            variance is zero under the null.
        bootstrap_p_value: ``(1 + #{T* >= T}) / (B + 1)`` from a parametric
            bootstrap under the reduced fit (``nan`` when ``B = 0``).
        bootstrap_statistics: The ``B`` simulated statistics ``T*``.
        method: ``"REML"`` or ``"ML"``.

    References:
        [REF-SELF-LIANG-1987], [REF-STRAM-LEE-1994],
        [REF-EFRON-TIBSHIRANI-1993, Chapter 6], [EQ-LMM-007].

    """

    statistic: float
    degrees_of_freedom: int
    p_value: float
    boundary_mixture_p_value: float
    bootstrap_p_value: float
    bootstrap_statistics: Vector
    method: str


def _same_design(reduced: LinearMixedModelResult, full: LinearMixedModelResult) -> bool:
    """Return whether two fits share their fixed-effect design (names ignored).

    References:
        [REF-BATES-ETAL-2015, Section 5.2]. No equation is evaluated.

    """
    return reduced.design == full.design


def _simulate_response(
    fit: LinearMixedModelResult, generator_seed: RandomStream
) -> Vector:
    """Draw ``y* = X beta + Z b* + epsilon*`` from a fitted model.

    References:
        [EQ-LMM-001], [REF-EFRON-TIBSHIRANI-1993, Chapter 6].

    """
    rng = generator_seed.generator()
    sigma = fit.residual_sd
    structure = _structure(fit.terms, fit.n_observations)
    factors = _relative_factors(structure, fit.theta)
    draws = [sigma * rng.gauss(0.0, 1.0) for _ in range(structure.q)]
    zb = _z_times(structure, _lambda(structure, factors, draws))
    return tuple(
        m + z + sigma * rng.gauss(0.0, 1.0)
        for m, z in zip(fit.marginal_fitted_values, zb, strict=True)
    )


def _refit(fit: LinearMixedModelResult, response: Vector) -> LinearMixedModelResult:
    """Refit the specification of ``fit`` to a new response.

    References:
        [REF-BATES-ETAL-2015, Section 3]. No new equation is evaluated.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MixedModelConvergenceWarning)
        return fit_linear_mixed_model(
            fit.design,
            response,
            fit.terms,
            method=fit.method,
            column_names=fit.column_names,
            denominator_df="asymptotic",
        )


def mixed_model_likelihood_ratio_test(
    reduced: LinearMixedModelResult,
    full: LinearMixedModelResult,
    *,
    bootstrap_samples: int = 0,
    seed: int | RandomStream | None = None,
) -> MixedModelLikelihoodRatioTest:
    """Compare nested linear mixed models by their likelihood ratio.

    REML log-likelihoods are comparable only between models with the same
    fixed effects; to compare fixed effects, fit both models with
    ``method="ML"``. A variance component tested at zero lies on the
    boundary of the parameter space, so the chi-square reference with the
    parameter-count difference is conservative; the boundary-mixture p-value
    and the parametric bootstrap address this.

    Args:
        reduced: Fit of the null (smaller) model.
        full: Fit of the alternative (larger) model on the same response.
        bootstrap_samples: Number ``B`` of parametric-bootstrap replicates
            simulated from ``reduced`` and refitted under both models.
        seed: Seed or stream for the bootstrap (``None`` is not
            reproducible).

    Returns:
        :class:`MixedModelLikelihoodRatioTest`.

    Raises:
        ValueError: If the fits use different methods or responses, REML
            fits differ in fixed effects, the full model has no extra
            parameters, or ``bootstrap_samples`` is negative.

    Equation:
        [EQ-LMM-007].

    References:
        [REF-SELF-LIANG-1987], [REF-STRAM-LEE-1994],
        [REF-BATES-ETAL-2015, Section 5.2], [REF-EFRON-TIBSHIRANI-1993].

    """
    if reduced.method != full.method:
        raise ValueError("both fits must use the same method")
    if reduced.response != full.response:
        raise ValueError("both fits must use the same response")
    if full.method == "REML" and not _same_design(reduced, full):
        raise ValueError(
            "REML likelihoods are not comparable when the fixed effects differ; "
            "refit both models with method='ML'"
        )
    df = full.parameter_count - reduced.parameter_count
    if df <= 0:
        raise ValueError(
            "the full model must have more parameters than the reduced one"
        )
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise TypeError("bootstrap_samples must be an integer")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    statistic = max(2.0 * (full.log_likelihood - reduced.log_likelihood), 0.0)
    p_value = chi_square_sf(statistic, df)
    lower_tail = (
        (1.0 if statistic <= 0.0 else 0.0)
        if df == 1
        else chi_square_sf(statistic, df - 1)
    )
    mixture = 0.5 * lower_tail + 0.5 * p_value
    simulated: list[float] = []
    bootstrap_p = math.nan
    if bootstrap_samples:
        stream = resolve_stream(seed)
        tolerance = 1e-8 * (1.0 + statistic)
        for child in stream.spawn(bootstrap_samples, label="lrt-bootstrap"):
            response = _simulate_response(reduced, child)
            null_fit = _refit(reduced, response)
            alternative = _refit(full, response)
            simulated.append(
                max(2.0 * (alternative.log_likelihood - null_fit.log_likelihood), 0.0)
            )
        exceed = sum(1 for value in simulated if value >= statistic - tolerance)
        bootstrap_p = (1.0 + exceed) / (bootstrap_samples + 1.0)
    return MixedModelLikelihoodRatioTest(
        statistic=statistic,
        degrees_of_freedom=df,
        p_value=p_value,
        boundary_mixture_p_value=mixture,
        bootstrap_p_value=bootstrap_p,
        bootstrap_statistics=tuple(simulated),
        method=full.method,
    )


# ---------------------------------------------------------------------------
# Generalized linear mixed models
# ---------------------------------------------------------------------------


@functools.cache
def _gauss_hermite(order: int) -> tuple[Vector, Vector]:
    """Return physicists' Gauss-Hermite nodes and weights for ``exp(-x^2)``.

    Golub-Welsch eigenvalues of the Jacobi matrix (off-diagonal
    ``sqrt(k/2)``) seed two Newton steps on the orthonormal Hermite
    polynomial ``h_n`` (``h_n' = sqrt(2n) h_(n-1)``); the weights are the
    Christoffel numbers ``1 / sum_(j<n) h_j(x)^2``.

    References:
        [EQ-GLMM-002], [REF-GOLUB-WELSCH-1969],
        [REF-NIST-DLMF-ORTHOGONAL, Sections 3.5 and 18.3].

    """
    if order == 1:
        return (0.0,), (math.sqrt(math.pi),)
    jacobi = tuple(
        tuple(
            math.sqrt(0.5 * max(i, j)) if abs(i - j) == 1 else 0.0 for j in range(order)
        )
        for i in range(order)
    )
    eigenvalues, _ = symmetric_eigen(jacobi)

    def orthonormal(x: float) -> list[float]:
        """Return ``h_0(x) .. h_n(x)``.

        References:
            [REF-NIST-DLMF-ORTHOGONAL, Section 18.9].

        """
        values = [math.pi**-0.25, math.sqrt(2.0) * x * math.pi**-0.25]
        for j in range(1, order):
            values.append(
                math.sqrt(2.0 / (j + 1)) * x * values[j]
                - math.sqrt(j / (j + 1.0)) * values[j - 1]
            )
        return values

    nodes = []
    weights = []
    for guess in sorted(eigenvalues):
        x = guess
        for _ in range(2):
            values = orthonormal(x)
            x -= values[order] / (math.sqrt(2.0 * order) * values[order - 1])
        values = orthonormal(x)
        nodes.append(x)
        weights.append(1.0 / math.fsum(v * v for v in values[:order]))
    half = order // 2
    for i in range(half):
        mirror = order - 1 - i
        magnitude = 0.5 * (nodes[mirror] - nodes[i])
        weight = 0.5 * (weights[i] + weights[mirror])
        nodes[i], nodes[mirror] = -magnitude, magnitude
        weights[i] = weights[mirror] = weight
    if order % 2:
        nodes[half] = 0.0
    return tuple(nodes), tuple(weights)


def _softplus(eta: float) -> float:
    """Return ``log(1 + exp(eta))`` without overflow.

    References:
        [REF-MCCULLAGH-NELDER-1989, Section 4.3]. No named equation is needed.

    """
    return eta + math.log1p(math.exp(-eta)) if eta > 0.0 else math.log1p(math.exp(eta))


@dataclass(frozen=True, slots=True)
class _GLMMData:
    """Data of a GLMM fit with per-observation constants precomputed.

    References:
        [EQ-GLMM-001], [REF-MCCULLAGH-NELDER-1989, Chapter 4].

    """

    structure: _Structure
    family: str
    x: Matrix
    y: Vector
    weights: Vector
    offset: Vector
    constants: Vector
    group_members: tuple[tuple[int, ...], ...]


def _glmm_data(
    structure: _Structure, family: str, x: Matrix, y: Vector, w: Vector, offset: Vector
) -> _GLMMData:
    """Precompute the normalizing constants of the binomial or Poisson law.

    References:
        [EQ-GLMM-001], [REF-MCCULLAGH-NELDER-1989, Chapter 4].

    """
    constants: list[float] = []
    for yi, wi in zip(y, w, strict=True):
        if family == "binomial":
            successes = wi * yi
            constants.append(
                math.lgamma(wi + 1.0)
                - math.lgamma(successes + 1.0)
                - math.lgamma(wi - successes + 1.0)
            )
        else:
            constants.append(-wi * math.lgamma(yi + 1.0))
    members: tuple[tuple[int, ...], ...] = ()
    if len(structure.terms) == 1 and structure.terms[0].size == 1:
        term = structure.terms[0]
        groups: list[list[int]] = [[] for _ in range(term.n_levels)]
        for i, level in enumerate(term.level_index):
            groups[level].append(i)
        members = tuple(tuple(g) for g in groups)
    return _GLMMData(structure, family, x, y, w, offset, tuple(constants), members)


def _unit(data: _GLMMData, i: int, eta: float) -> tuple[float, float, float]:
    """Return ``(log p(y_i | eta), d/d eta, -d^2/d eta^2)`` for one observation.

    References:
        [EQ-GLMM-001], [REF-MCCULLAGH-NELDER-1989, Chapter 4].

    """
    y, w = data.y[i], data.weights[i]
    if data.family == "binomial":
        mu = (
            1.0 / (1.0 + math.exp(-eta))
            if eta >= 0.0
            else math.exp(eta) / (1.0 + math.exp(eta))
        )
        value = w * (y * eta - _softplus(eta)) + data.constants[i]
        return value, w * (y - mu), w * mu * (1.0 - mu)
    if eta > 700.0:
        raise ValueError("Poisson linear predictor overflows")
    mu = math.exp(eta)
    return w * (y * eta - mu) + data.constants[i], w * (y - mu), w * mu


def _fixed_predictor(data: _GLMMData, beta: Sequence[float]) -> list[float]:
    """Return ``X beta + offset``.

    References:
        [EQ-GLMM-001]. No other equation is evaluated.

    """
    return [
        math.fsum(b * x for b, x in zip(beta, row, strict=True)) + o
        for row, o in zip(data.x, data.offset, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class _LaplaceState:
    """Conditional mode and Laplace deviance at one ``(theta, beta)``.

    References:
        [EQ-GLMM-001], [REF-BATES-ETAL-2015, Section 3].

    """

    u: list[float]
    lower: list[list[float]]
    factors: list[list[list[float]]]
    deviance: float
    eta: list[float]


def _laplace_state(
    data: _GLMMData,
    theta: Sequence[float],
    beta: Sequence[float],
    start: Sequence[float] | None = None,
    max_iterations: int = 200,
) -> _LaplaceState:
    """Find the conditional mode by penalized IRLS and the Laplace deviance.

    Newton steps on ``h(u) = sum log p(y_i | eta_i) - ||u||^2 / 2`` with the
    canonical-link Hessian ``-(Lambda' Z' D Z Lambda + I)`` and step halving.

    Raises:
        ValueError: If the mode search fails to increase ``h``.

    References:
        [EQ-GLMM-001], [REF-BATES-ETAL-2015, Section 3],
        [REF-PINHEIRO-BATES-1995].

    """
    structure = data.structure
    first = structure.first
    factors = _relative_factors(structure, theta)
    base = _fixed_predictor(data, beta)
    u = list(start) if start is not None else [0.0] * structure.q

    def evaluate(
        values: Sequence[float],
    ) -> tuple[float, list[float], list[float], list[float]]:
        """Return ``h``, the score, the curvature weights and ``eta``.

        References:
            [EQ-GLMM-001].

        """
        zb = _z_times(structure, _lambda(structure, factors, values))
        eta = [a + b for a, b in zip(base, zb, strict=True)]
        logs, scores, curvature = [], [], []
        for i, value in enumerate(eta):
            log_p, score, weight = _unit(data, i, value)
            logs.append(log_p)
            scores.append(score)
            curvature.append(weight)
        h = math.fsum(logs) - 0.5 * math.fsum(v * v for v in values)
        return h, scores, curvature, eta

    h, scores, curvature, eta = evaluate(u)
    for _ in range(max_iterations):
        lower = _penalized_cross(structure, _z_cross(structure, curvature), factors)
        _envelope_cholesky(lower, first)
        zs = [0.0] * structure.q
        for row, score in zip(structure.z_rows, scores, strict=True):
            for a, value in row:
                zs[a] += value * score
        gradient = [
            g - v
            for g, v in zip(_lambda_transpose(structure, factors, zs), u, strict=True)
        ]
        step = _backward(lower, first, _forward(lower, first, gradient))
        length = 1.0
        while True:
            candidate = [v + length * s for v, s in zip(u, step, strict=True)]
            try:
                new = evaluate(candidate)
            except (ValueError, OverflowError):
                new = None
            if new is not None and new[0] >= h - 1e-12 * (abs(h) + 1.0):
                break
            length *= 0.5
            if length < 1e-10:
                raise ValueError("penalized IRLS failed to increase the objective")
        assert new is not None
        largest = max((abs(length * s) for s in step), default=0.0)
        u = candidate
        h, scores, curvature, eta = new
        if largest <= 1e-10 * (1.0 + max((abs(v) for v in u), default=0.0)):
            break
    lower = _penalized_cross(structure, _z_cross(structure, curvature), factors)
    _envelope_cholesky(lower, first)
    log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(structure.q))
    return _LaplaceState(u, lower, factors, -2.0 * h + log_det, eta)


def _quadrature_deviance(
    data: _GLMMData,
    theta: float,
    beta: Sequence[float],
    order: int,
    start: list[float] | None = None,
) -> tuple[float, list[float], list[float]]:
    """Return the AGHQ deviance, the group modes and the curvature scales.

    For group ``j``, ``h_j(u) = sum_(i in j) log p(y_i | eta_i + theta u)
    - u^2/2``; with mode ``u_j`` and ``s_j = (-h_j''(u_j))^(-1/2)``,
    ``L_j = s_j sum_k (w_k e^(x_k^2) / sqrt(pi)) exp(h_j(u_j + sqrt(2) s_j
    x_k))``. One node gives the Laplace value ``s_j exp(h_j(u_j))``.

    Raises:
        ValueError: If a group's mode search fails.

    References:
        [EQ-GLMM-002], [REF-LIU-PIERCE-1994], [REF-PINHEIRO-BATES-1995].

    """
    nodes, weights = _gauss_hermite(order)
    log_weights = [
        math.log(w) + x * x - _LOG_SQRT_PI for x, w in zip(nodes, weights, strict=True)
    ]
    base = _fixed_predictor(data, beta)
    modes: list[float] = []
    scales: list[float] = []
    total: list[float] = []

    def group_h(members: tuple[int, ...], u: float) -> tuple[float, float, float]:
        """Return ``h_j(u)``, ``h_j'(u)`` and ``-h_j''(u)``.

        References:
            [EQ-GLMM-002].

        """
        value, slope, curvature = -0.5 * u * u, -u, 1.0
        for i in members:
            log_p, score, weight = _unit(data, i, base[i] + theta * u)
            value += log_p
            slope += theta * score
            curvature += theta * theta * weight
        return value, slope, curvature

    for j, members in enumerate(data.group_members):
        u = start[j] if start is not None else 0.0
        value, slope, curvature = group_h(members, u)
        for _ in range(200):
            step = slope / curvature
            length = 1.0
            while True:
                candidate = u + length * step
                try:
                    trial = group_h(members, candidate)
                except (ValueError, OverflowError):
                    trial = None
                if trial is not None and trial[0] >= value - 1e-13 * (abs(value) + 1.0):
                    break
                length *= 0.5
                if length < 1e-12:
                    raise ValueError("group mode search failed")
            assert trial is not None
            u = candidate
            value, slope, curvature = trial
            if abs(length * step) <= 1e-12 * (1.0 + abs(u)):
                break
        scale = 1.0 / math.sqrt(curvature)
        if order == 1:
            log_lj = math.log(scale) + value
        else:
            terms = [
                lw + group_h(members, u + math.sqrt(2.0) * scale * x)[0]
                for x, lw in zip(nodes, log_weights, strict=True)
            ]
            peak = max(terms)
            log_lj = (
                math.log(scale)
                + peak
                + math.log(math.fsum(math.exp(t - peak) for t in terms))
            )
        modes.append(u)
        scales.append(scale)
        total.append(log_lj)
    return -2.0 * math.fsum(total), modes, scales


@dataclass(frozen=True, slots=True)
class GeneralizedLinearMixedModelResult:
    """Fit of a binomial or Poisson GLMM by Laplace or adaptive quadrature.

    Attributes:
        family: ``"binomial"`` or ``"poisson"``.
        link: ``"logit"`` or ``"log"``.
        method: ``"laplace"`` or ``"adaptive_quadrature"``.
        quadrature_points: Gauss-Hermite nodes per group (one for Laplace).
        coefficients: Fixed effects ``beta``.
        standard_errors: Square roots of the diagonal of ``covariance``.
        statistics: Wald ratios ``beta_j / se_j``.
        p_values: Two-sided normal p-values.
        column_names: Fixed-effect column names.
        covariance: ``beta`` block of ``2 H^-1``, ``H`` the Hessian of the
            deviance in ``(theta, beta)``.
        theta: Relative covariance factor parameters (lme4 order); the
            random-effect covariance is ``T T'``.
        variance_components: One :class:`VarianceComponent` per term.
        random_effects: Conditional modes ``Lambda u`` with conditional
            covariances ``Lambda (L L')^-1 Lambda'``.
        linear_predictor: ``X beta + offset + Z b`` at the modes.
        fitted_values: Conditional means ``g^-1(linear_predictor)``.
        log_likelihood: Approximate maximized log-likelihood.
        deviance: ``-2 log_likelihood``.
        aic: ``-2 log L + 2k``.
        bic: ``-2 log L + k log n``.
        parameter_count: ``k = p + len(theta)``.
        singular: Whether some ``T_k`` has a diagonal below ``1e-4``.
        converged: Whether the optimizer met its tolerance.
        iterations: Optimizer iterations.
        evaluations: Deviance evaluations.
        message: Optimizer termination message.
        n_observations: ``n``.

    References:
        [REF-BATES-ETAL-2015], [REF-PINHEIRO-BATES-1995], [REF-LIU-PIERCE-1994].

    """

    family: str
    link: str
    method: str
    quadrature_points: int
    coefficients: Vector
    standard_errors: Vector
    statistics: Vector
    p_values: Vector
    column_names: tuple[str, ...]
    covariance: Matrix
    theta: Vector
    variance_components: tuple[VarianceComponent, ...]
    random_effects: tuple[RandomEffectEstimates, ...]
    linear_predictor: Vector
    fitted_values: Vector
    log_likelihood: float
    deviance: float
    aic: float
    bic: float
    parameter_count: int
    singular: bool
    converged: bool
    iterations: int
    evaluations: int
    message: str
    n_observations: int

    def coefficient_intervals(
        self, level: float = 0.95
    ) -> tuple[tuple[float, float], ...]:
        """Return Wald intervals ``beta_j +- z se_j``.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            One ``(lower, upper)`` pair per coefficient.

        Raises:
            ValueError: If ``level`` is invalid.

        Equation:
            [EQ-GLM-WALD-001].

        References:
            [REF-BATES-ETAL-2015, Section 5].

        """
        checked = _validate_level(level)
        quantile = _STANDARD_NORMAL.inv_cdf(0.5 * (1.0 + checked))
        return tuple(
            (b - quantile * s, b + quantile * s)
            for b, s in zip(self.coefficients, self.standard_errors, strict=True)
        )


def glmm_deviance(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    random_effects: Sequence[RandomEffectTerm] | RandomEffectTerm,
    theta: Iterable[object],
    beta: Iterable[object],
    *,
    family: str = "binomial",
    weights: Iterable[object] | None = None,
    offset: Iterable[object] | None = None,
    quadrature_points: int = 1,
) -> float:
    """Evaluate the approximate GLMM deviance at given parameters.

    With one node (or more than one scalar random effect per observation)
    this is the Laplace deviance of [EQ-GLMM-001]; with several nodes and a
    single random intercept it is the adaptive Gauss-Hermite deviance of
    [EQ-GLMM-002]. Useful for profiling and for checking quadrature
    convergence at fixed parameters.

    Args:
        design: Fixed-effect design rows.
        response: Proportions (binomial, with trials as ``weights``) or
            counts (Poisson).
        random_effects: One term or a sequence of terms.
        theta: Relative covariance factor parameters (lme4 order).
        beta: Fixed effects.
        family: ``"binomial"`` (logit) or ``"poisson"`` (log).
        weights: Binomial trials or Poisson prior weights.
        offset: Known additive terms of the linear predictor.
        quadrature_points: Gauss-Hermite nodes (``1..64``).

    Returns:
        ``-2 log L`` including the normalizing constants of the law.

    Raises:
        ValueError: For invalid inputs, or several nodes with a model that
            is not a single random intercept.

    Equation:
        [EQ-GLMM-001], [EQ-GLMM-002].

    References:
        [REF-PINHEIRO-BATES-1995], [REF-LIU-PIERCE-1994].

    """
    data, _, _ = _prepare_glmm(
        design, response, random_effects, family, weights, offset, None
    )
    order = _check_order(quadrature_points)
    theta_values = as_vector(theta, name="theta")
    beta_values = as_vector(beta, name="beta")
    if len(theta_values) != data.structure.theta_size:
        raise ValueError("theta has the wrong length")
    if len(beta_values) != len(data.x[0]):
        raise ValueError("beta must have one value per design column")
    if order > 1 and not data.group_members:
        raise ValueError(
            "adaptive quadrature needs a single scalar random effect per group"
        )
    if data.group_members:
        return _quadrature_deviance(data, theta_values[0], beta_values, order)[0]
    return _laplace_state(data, theta_values, beta_values).deviance


def _check_order(quadrature_points: object) -> int:
    """Validate the number of Gauss-Hermite nodes.

    References:
        [REF-LIU-PIERCE-1994]. No equation is evaluated.

    """
    if isinstance(quadrature_points, bool) or not isinstance(quadrature_points, int):
        raise TypeError("quadrature_points must be an integer")
    if not 1 <= quadrature_points <= _MAX_QUADRATURE_ORDER:
        raise ValueError(f"quadrature_points must lie in 1..{_MAX_QUADRATURE_ORDER}")
    return quadrature_points


def _prepare_glmm(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    random_effects: Sequence[RandomEffectTerm] | RandomEffectTerm,
    family: str,
    weights: Iterable[object] | None,
    offset: Iterable[object] | None,
    column_names: Sequence[str] | None,
) -> tuple[_GLMMData, tuple[str, ...], Matrix]:
    """Validate GLMM inputs and build its data record.

    References:
        [REF-MCCULLAGH-NELDER-1989, Chapter 2]. No equation is evaluated.

    """
    if family not in ("binomial", "poisson"):
        raise ValueError("family must be 'binomial' or 'poisson'")
    glm = glm_family(family)
    rows, names, _ = _coerce_design(design, column_names)
    y = as_vector(response, name="response")
    n = len(rows)
    if len(y) != n:
        raise ValueError("response length must equal the number of design rows")
    if n <= len(rows[0]):
        raise ValueError("a mixed model needs more observations than fixed effects")
    _validate_response(glm, y)
    w = _positive_weights(weights, n)
    offsets = as_vector(offset, name="offset") if offset is not None else (0.0,) * n
    if len(offsets) != n:
        raise ValueError("offset must have one entry per observation")
    terms = _check_terms(random_effects, n) if random_effects else ()
    if not terms:
        raise ValueError("a GLMM needs at least one random-effect term")
    structure = _structure(terms, n)
    return _glmm_data(structure, family, rows, y, w, offsets), names, rows


def fit_generalized_linear_mixed_model(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    random_effects: Sequence[RandomEffectTerm] | RandomEffectTerm,
    *,
    family: str = "binomial",
    method: GLMMMethod | str = "laplace",
    quadrature_points: int = 9,
    weights: Iterable[object] | None = None,
    offset: Iterable[object] | None = None,
    column_names: Sequence[str] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 500,
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> GeneralizedLinearMixedModelResult:
    """Fit a binomial (logit) or Poisson (log) GLMM.

    ``method="laplace"`` maximizes the Laplace approximation
    [EQ-GLMM-001] and supports any random-effect terms (intercepts and
    slopes); ``method="adaptive_quadrature"`` uses ``quadrature_points``
    adaptive Gauss-Hermite nodes per group [EQ-GLMM-002] and requires a
    single random intercept. ``theta`` and ``beta`` are optimized jointly by
    BFGS from the GLM fit without random effects.

    Args:
        design: :class:`DesignMatrix` or ``n x p`` fixed-effect rows.
        response: Proportions in ``[0, 1]`` with trials as ``weights``
            (binomial) or non-negative counts (Poisson).
        random_effects: One term or a sequence of terms.
        family: ``"binomial"`` or ``"poisson"`` (canonical links only).
        method: ``"laplace"`` or ``"adaptive_quadrature"``.
        quadrature_points: Nodes per group for adaptive quadrature
            (``1..64``; one node is the Laplace approximation).
        weights: Binomial trials (or Poisson prior weights).
        offset: Known additive terms of the linear predictor.
        column_names: Names for a raw design.
        tolerance: BFGS gradient-norm tolerance.
        max_iterations: Positive optimizer iteration limit.
        rank_tolerance: Relative threshold for a dependent design column.

    Returns:
        :class:`GeneralizedLinearMixedModelResult`.

    Raises:
        RankDeficientDesignError: If the design is rank deficient.
        SeparationError: If the starting GLM shows separation.
        ValueError: For invalid inputs or adaptive quadrature with a model
            that is not a single random intercept.

    Warns:
        MixedModelConvergenceWarning: If the optimizer did not converge.

    Equation:
        [EQ-GLMM-001], [EQ-GLMM-002].

    References:
        [REF-BATES-ETAL-2015], [REF-PINHEIRO-BATES-1995],
        [REF-LIU-PIERCE-1994], [REF-MCCULLAGH-NELDER-1989].

    """
    if method not in ("laplace", "adaptive_quadrature"):
        raise ValueError("method must be 'laplace' or 'adaptive_quadrature'")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    order = _check_order(quadrature_points) if method == "adaptive_quadrature" else 1
    data, names, rows = _prepare_glmm(
        design, response, random_effects, family, weights, offset, column_names
    )
    _factorize(rows, names, rank_tolerance)
    if method == "adaptive_quadrature" and not data.group_members:
        raise ValueError(
            "adaptive quadrature needs a single scalar random effect per group"
        )
    structure = data.structure
    m = structure.theta_size
    start_glm = fit_glm(
        rows,
        data.y,
        family=family,
        weights=data.weights,
        offset=data.offset,
        column_names=names,
    )
    scalar = bool(data.group_members)
    cache: dict[str, list[float]] = {}

    def deviance(values: Sequence[float]) -> float:
        """Approximate deviance at ``(theta, beta)`` with warm-started modes.

        References:
            [EQ-GLMM-001], [EQ-GLMM-002].

        """
        theta, beta = values[:m], values[m:]
        if scalar:
            value, modes, _ = _quadrature_deviance(
                data, theta[0], beta, order, cache.get("u")
            )
            cache["u"] = modes
            return value
        state = _laplace_state(data, theta, beta, cache.get("u"))
        cache["u"] = state.u
        return state.deviance

    start = [*_initial_theta(structure), *start_glm.coefficients]
    optimum = _minimize(
        deviance,
        start,
        "bfgs",
        tolerance,
        max_iterations,
        structure.diagonal_theta,
        [1.0] * m + [max(abs(b), 1.0) for b in start_glm.coefficients],
    )
    if not optimum.converged:
        warnings.warn(
            f"GLMM optimizer did not converge: {optimum.message}",
            MixedModelConvergenceWarning,
            stacklevel=2,
        )
    theta = _normalize_theta(structure, optimum.x[:m])
    beta = tuple(optimum.x[m:])
    x_hat = (*theta, *beta)
    cache.pop("u", None)
    value = deviance(x_hat)
    hessian, _ = central_hessian(
        deviance, x_hat, typical_scale=[max(abs(v), 0.1) for v in x_hat]
    )
    p = len(beta)
    try:
        inverse = inverse_spd(hessian)
        covariance = tuple(
            tuple(2.0 * inverse[m + i][m + j] for j in range(p)) for i in range(p)
        )
    except ValueError:
        covariance = tuple(tuple(math.nan for _ in range(p)) for _ in range(p))
    errors = tuple(
        math.sqrt(covariance[j][j]) if covariance[j][j] >= 0.0 else math.nan
        for j in range(p)
    )
    statistics = tuple(
        b / s if s > 0.0 else math.nan for b, s in zip(beta, errors, strict=True)
    )
    p_values = tuple(
        2.0 * standard_normal_sf(abs(t)) if math.isfinite(t) else math.nan
        for t in statistics
    )
    state = _laplace_state(data, theta, beta)
    eta = tuple(state.eta)
    if family == "binomial":
        fitted = tuple(
            1.0 / (1.0 + math.exp(-e))
            if e >= 0.0
            else math.exp(e) / (1.0 + math.exp(e))
            for e in eta
        )
    else:
        fitted = tuple(math.exp(e) for e in eta)
    log_likelihood = -0.5 * value
    k = p + m
    aic, _, bic = information_criteria(log_likelihood, k, len(rows))
    return GeneralizedLinearMixedModelResult(
        family=family,
        link="logit" if family == "binomial" else "log",
        method=method,
        quadrature_points=order,
        coefficients=beta,
        standard_errors=errors,
        statistics=statistics,
        p_values=p_values,
        column_names=names,
        covariance=covariance,
        theta=tuple(theta),
        variance_components=_variance_components(structure, state.factors, 1.0),
        random_effects=_random_effect_estimates(
            structure, state.factors, state.u, state.lower, 1.0
        ),
        linear_predictor=eta,
        fitted_values=fitted,
        log_likelihood=log_likelihood,
        deviance=value,
        aic=aic,
        bic=bic,
        parameter_count=k,
        singular=_is_singular(structure, theta),
        converged=optimum.converged,
        iterations=optimum.iterations,
        evaluations=optimum.evaluations,
        message=optimum.message,
        n_observations=len(rows),
    )
