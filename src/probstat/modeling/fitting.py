"""Fit a :class:`JointInput` (marginals plus Gaussian copula) to tabular data.

Each named column receives a marginal law, either from a user-specified
family or by information-criterion model selection over candidate families.
Dependence is a Gaussian copula whose latent correlation is estimated from
ranks, so it is invariant to the marginal fits:

- ``"spearman"``: pairwise Spearman correlations mapped to the latent normal
  correlation by ``rho = 2 sin(pi r_s / 6)``;
- ``"normal-scores"``: Pearson correlation of the van der Waerden normal
  scores ``Phi^-1(rank / (n + 1))`` built from the rank pseudo-observations;
- ``"independent"``: identity correlation.

A pairwise estimate that is not positive definite is repaired by eigenvalue
clipping and re-normalization, and the repair is reported.

:func:`fit_copula` fits any copula family of :mod:`probstat.modeling.copulas`
from ranks: by inversion of Kendall's tau (:func:`kendall_tau`) or by maximum
pseudo-likelihood for the one-parameter families (Clayton, Gumbel, Frank, and
the t copula's degrees of freedom).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from probstat.foundations import coerce_finite_sample
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inference.fitting import (
    DistributionFamily,
    FitResult,
    ModelSelectionResult,
    family_by_name,
    fit_distribution,
    select_distribution,
)
from probstat.linalg import Matrix, cholesky, symmetric_eigen
from probstat.optimization import brent_root
from probstat.statistics import ranks, spearman_correlation

from .copulas import (
    ClaytonCopula,
    Copula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependenceCopula,
    StudentTCopula,
    frank_tau,
    nearest_correlation_matrix,
)
from .inputs import (
    ContinuousMarginal,
    JointInput,
    RandomInput,
    standard_normal_quantile,
)

Dependence = Literal["spearman", "normal-scores", "independent"]
_DEPENDENCE = ("spearman", "normal-scores", "independent")
_EIGEN_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class GaussianCopulaFit:
    """Rank-based Gaussian-copula correlation estimate.

    Args:
        names: Column names in order.
        rank_correlation: Spearman matrix (``"spearman"``) or normal-score
            correlation matrix (``"normal-scores"``).
        latent_correlation: Positive-definite latent normal correlation ``R``.
        method: Estimation method label.
        repaired: Whether eigenvalue clipping was needed for definiteness.
        sample_size: Number of rows.

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995], [REF-NELSEN-2006, Chapter 5].

    """

    names: tuple[str, ...]
    rank_correlation: Matrix
    latent_correlation: Matrix
    method: str
    repaired: bool
    sample_size: int


@dataclass(frozen=True, slots=True)
class JointInputFit:
    """Fitted joint input law with per-column fits and the copula estimate.

    Args:
        joint_input: Fitted :class:`JointInput` (marginals plus copula).
        marginal_fits: Maximum-likelihood fits in column order.
        selections: Model-selection results in column order (``None`` for
            columns whose family was specified).
        copula: Gaussian-copula estimate (identity for ``"independent"``).

    References:
        [REF-LIU-DERKIUREGHIAN-1986], [REF-GENEST-GHOUDI-RIVEST-1995].

    """

    joint_input: JointInput
    marginal_fits: tuple[FitResult, ...]
    selections: tuple[ModelSelectionResult | None, ...]
    copula: GaussianCopulaFit

    @property
    def names(self) -> tuple[str, ...]:
        """Return column names in model order.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return self.joint_input.names


def spearman_to_latent_correlation(spearman: float) -> float:
    """Map a Spearman correlation to the Gaussian-copula latent correlation.

    For a bivariate normal pair (and hence a Gaussian copula) Spearman's rho
    is ``r_s = (6 / pi) arcsin(rho / 2)``, inverted as
    ``rho = 2 sin(pi r_s / 6)``.

    Args:
        spearman: Spearman rank correlation in ``[-1, 1]``.

    Returns:
        Latent normal correlation in ``[-1, 1]``.

    Raises:
        TypeError: If the input is not real.
        ValueError: If the input is outside ``[-1, 1]`` or non-finite.

    Equation:
        [EQ-MOD-COPULA-FIT-001].

    References:
        [REF-KRUSKAL-1958], [REF-MCNEIL-FREY-EMBRECHTS-2015].

    """
    if isinstance(spearman, bool) or not isinstance(spearman, int | float):
        raise TypeError("spearman must be a real number")
    value = float(spearman)
    if not math.isfinite(value) or abs(value) > 1.0:
        raise ValueError("spearman correlation must lie in [-1, 1]")
    return max(-1.0, min(1.0, 2.0 * math.sin(math.pi * value / 6.0)))


def normal_scores(values: Iterable[object]) -> tuple[float, ...]:
    """Return van der Waerden normal scores ``Phi^-1(rank_i / (n + 1))``.

    Ties receive their average rank. The ranks scaled by ``n + 1`` are the
    pseudo-observations of the copula and stay strictly inside ``(0, 1)``.

    Args:
        values: At least two finite observations.

    Returns:
        Normal scores in the original order.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If fewer than two finite observations are given.

    Equation:
        [EQ-MOD-COPULA-FIT-002].

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995], [REF-HOLLANDER-WOLFE-CHICKEN-2014,
        Chapter 8].

    """
    sample = coerce_finite_sample(values)
    if len(sample) < 2:
        raise ValueError("normal scores need at least two observations")
    scale = len(sample) + 1.0
    return tuple(standard_normal_quantile(rank / scale) for rank in ranks(sample))


def _pearson(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the Pearson correlation of two equally long sequences.

    Raises:
        ValueError: If either sequence is constant.

    References:
        [EQ-STAT-007], [REF-NIST-EDA, Section 1.3.5]. No new equation.

    """
    count = len(first)
    mean_a = math.fsum(first) / count
    mean_b = math.fsum(second) / count
    sab = math.fsum(
        (a - mean_a) * (b - mean_b) for a, b in zip(first, second, strict=True)
    )
    saa = math.fsum((a - mean_a) ** 2 for a in first)
    sbb = math.fsum((b - mean_b) ** 2 for b in second)
    if saa == 0.0 or sbb == 0.0:
        raise ValueError("correlation is undefined for a constant column")
    return max(-1.0, min(1.0, sab / math.sqrt(saa * sbb)))


def _repair_correlation(matrix: Matrix) -> tuple[Matrix, bool]:
    """Return a positive-definite unit-diagonal version of a correlation matrix.

    If every eigenvalue is at least ``1e-6`` the matrix is returned
    unchanged. Otherwise the eigenvalues are clipped at ``1e-6`` and the
    result is rescaled to a unit diagonal (the eigenvalue method). Testing
    the eigenvalues rather than Cholesky success matters: a latent
    correlation of ``0.9999999999999999`` from perfectly monotone columns
    factorizes, but its pivot of order ``1e-8`` maps observed points to
    standard-normal coordinates of order ``1e6``.

    References:
        [REF-MCNEIL-FREY-EMBRECHTS-2015], [REF-GOLUB-VANLOAN-2013,
        Section 8.5].

    """
    values, vectors = symmetric_eigen(matrix)
    if min(values) >= _EIGEN_FLOOR:
        return matrix, False
    size = len(matrix)
    clipped = tuple(max(value, _EIGEN_FLOOR) for value in values)
    rebuilt = [
        [
            math.fsum(
                vectors[i][k] * clipped[k] * vectors[j][k] for k in range(size)
            )
            for j in range(size)
        ]
        for i in range(size)
    ]
    diagonal = [math.sqrt(rebuilt[i][i]) for i in range(size)]
    repaired = tuple(
        tuple(
            1.0 if i == j else rebuilt[i][j] / (diagonal[i] * diagonal[j])
            for j in range(size)
        )
        for i in range(size)
    )
    cholesky(repaired)
    return repaired, True


def _columns(
    data: Mapping[str, Iterable[object]],
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Validate named columns of equal length.

    Raises:
        TypeError: If ``data`` is not a mapping with string keys.
        ValueError: If there are no columns, lengths differ, or a column has
            fewer than three rows.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping from column name to values")
    names = tuple(data)
    if not names:
        raise ValueError("data must contain at least one column")
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise TypeError("column names must be non-empty strings")
    columns = tuple(coerce_finite_sample(data[name], name=name) for name in names)
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError("all columns must have the same number of rows")
    if lengths.pop() < 3:
        raise ValueError("joint fitting needs at least three rows")
    return names, columns


def fit_gaussian_copula(
    data: Mapping[str, Iterable[object]],
    *,
    method: Dependence = "spearman",
) -> GaussianCopulaFit:
    """Estimate a Gaussian-copula correlation matrix from ranks.

    Args:
        data: Named columns of equal length (at least three rows).
        method: ``"spearman"``, ``"normal-scores"``, or ``"independent"``.

    Returns:
        Rank correlation, positive-definite latent correlation, and a repair
        flag (a :class:`UserWarning` is also emitted when repairing).

    Raises:
        TypeError: If ``data`` is malformed.
        ValueError: If the method is unknown, columns are inconsistent, or a
            column is constant.

    Equation:
        [EQ-MOD-COPULA-FIT-001], [EQ-MOD-COPULA-FIT-002], [EQ-STAT-RANK-001].

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995], [REF-KRUSKAL-1958],
        [REF-MCNEIL-FREY-EMBRECHTS-2015].

    """
    if method not in _DEPENDENCE:
        raise ValueError(f"method must be one of {_DEPENDENCE}")
    names, columns = _columns(data)
    size = len(names)
    count = len(columns[0])
    identity_rows = tuple(
        tuple(1.0 if i == j else 0.0 for j in range(size)) for i in range(size)
    )
    if method == "independent":
        return GaussianCopulaFit(
            names, identity_rows, identity_rows, method, False, count
        )
    rank_matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    latent = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    scores = (
        tuple(normal_scores(column) for column in columns)
        if method == "normal-scores"
        else ()
    )
    for i in range(size):
        for j in range(i + 1, size):
            if method == "spearman":
                if len(set(columns[i])) < 2 or len(set(columns[j])) < 2:
                    raise ValueError("correlation is undefined for a constant column")
                rank_value = spearman_correlation(columns[i], columns[j])
                latent_value = spearman_to_latent_correlation(rank_value)
            else:
                rank_value = _pearson(scores[i], scores[j])
                latent_value = rank_value
            rank_matrix[i][j] = rank_matrix[j][i] = rank_value
            latent[i][j] = latent[j][i] = latent_value
    repaired_matrix, repaired = _repair_correlation(
        tuple(tuple(row) for row in latent)
    )
    if repaired:
        warnings.warn(
            "pairwise latent correlation was not safely positive definite and was "
            "repaired by eigenvalue clipping",
            UserWarning,
            stacklevel=2,
        )
    return GaussianCopulaFit(
        names,
        tuple(tuple(row) for row in rank_matrix),
        repaired_matrix,
        method,
        repaired,
        count,
    )


def fit_joint_input(
    data: Mapping[str, Iterable[object]],
    *,
    marginals: Mapping[str, DistributionFamily | str] | None = None,
    candidates: Sequence[DistributionFamily | str] | None = None,
    dependence: Dependence = "spearman",
    criterion: Literal["aic", "aicc", "bic"] = "aic",
    n_bootstrap: int = 0,
    alpha: float = 0.05,
    seed: int | RandomStream | None = None,
) -> JointInputFit:
    """Fit marginals per column and a Gaussian copula; return a ``JointInput``.

    Columns named in ``marginals`` are fitted with that family by maximum
    likelihood; the others are chosen by :func:`select_distribution` over
    ``candidates`` (``None`` means every admissible ready-made family). The
    copula is estimated from ranks, so marginal misfit does not bias it.

    Args:
        data: Named columns of equal length (at least three rows).
        marginals: Optional family (or ready-made name) per column.
        candidates: Candidate families for unspecified columns.
        dependence: ``"spearman"``, ``"normal-scores"``, or ``"independent"``.
        criterion: Selection criterion for unspecified columns.
        n_bootstrap: Bootstrap GOF replicates per candidate (``0`` skips).
        alpha: GOF rejection level for selection warnings.
        seed: Seed or stream for bootstrap calibration.

    Returns:
        Fitted joint input, marginal fits, selection results, and copula.

    Raises:
        TypeError: If inputs are malformed.
        KeyError: If ``marginals`` names an unknown column or family.
        ValueError: If data, settings, or fits are invalid.

    Equation:
        [EQ-INF-MLE-GEN-001], [EQ-INF-IC-001], [EQ-MOD-COPULA-FIT-001],
        [EQ-MOD-COPULA-FIT-002], [EQ-MOD-COPULA-001].

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995], [REF-LIU-DERKIUREGHIAN-1986],
        [REF-BURNHAM-ANDERSON-2002, Chapter 2].

    """
    names, columns = _columns(data)
    chosen: dict[str, DistributionFamily] = {}
    if marginals is not None:
        if not isinstance(marginals, Mapping):
            raise TypeError("marginals must be a mapping from column to family")
        for key, family in marginals.items():
            if key not in names:
                raise KeyError(f"marginals names unknown column {key!r}")
            if isinstance(family, str):
                chosen[key] = family_by_name(family)
            elif isinstance(family, DistributionFamily):
                chosen[key] = family
            else:
                raise TypeError("each marginal must be a DistributionFamily or name")
    stream = resolve_stream(seed) if n_bootstrap else RandomStream(0)
    fits: list[FitResult] = []
    selections: list[ModelSelectionResult | None] = []
    for name, column in zip(names, columns, strict=True):
        if name in chosen:
            fits.append(fit_distribution(column, chosen[name]))
            selections.append(None)
            continue
        selection = select_distribution(
            column,
            candidates,
            criterion=criterion,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=stream.child("column", name),
        )
        fits.append(selection.best.fit)
        selections.append(selection)
    copula = fit_gaussian_copula(
        dict(zip(names, columns, strict=True)), method=dependence
    )
    inputs = tuple(
        RandomInput(name, cast(ContinuousMarginal, fitted.distribution))
        for name, fitted in zip(names, fits, strict=True)
    )
    correlation = None if dependence == "independent" else copula.latent_correlation
    return JointInputFit(
        JointInput(inputs, correlation),
        tuple(fits),
        tuple(selections),
        copula,
    )


CopulaFamily = Literal[
    "independence", "gaussian", "student-t", "clayton", "gumbel", "frank"
]
CopulaMethod = Literal["kendall", "pseudo-likelihood"]
_COPULA_FAMILIES = (
    "independence",
    "gaussian",
    "student-t",
    "clayton",
    "gumbel",
    "frank",
)
_COPULA_METHODS = ("kendall", "pseudo-likelihood")
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def kendall_tau(first: Iterable[object], second: Iterable[object]) -> float:
    """Return Kendall's tau-b of two paired samples.

    ``tau_b = (n_c - n_d) / sqrt((n_0 - n_1)(n_0 - n_2))`` over all
    ``n_0 = n(n-1)/2`` pairs, with ``n_c``/``n_d`` concordant/discordant
    pairs and ``n_1``/``n_2`` pairs tied in the first/second sample. Without
    ties it equals tau-a. The direct pair count costs ``O(n^2)``.

    Args:
        first: At least two finite observations.
        second: Paired finite observations of equal length.

    Returns:
        Sample Kendall's tau in ``[-1, 1]``.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If lengths differ, fewer than two pairs are given, or a
            sample is constant.

    Equation:
        [EQ-MOD-COPULA-FIT-003].

    References:
        [REF-KENDALL-1938], [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 8].

    """
    a = coerce_finite_sample(first, name="first")
    b = coerce_finite_sample(second, name="second")
    if len(a) != len(b):
        raise ValueError("samples must have the same length")
    count = len(a)
    if count < 2:
        raise ValueError("Kendall's tau needs at least two pairs")
    concordant = discordant = tied_first = tied_second = 0
    for i in range(count):
        ai, bi = a[i], b[i]
        for j in range(i + 1, count):
            da = a[j] - ai
            db = b[j] - bi
            if da == 0.0:
                tied_first += 1
            if db == 0.0:
                tied_second += 1
            product = da * db
            if product > 0.0:
                concordant += 1
            elif product < 0.0:
                discordant += 1
    pairs = count * (count - 1) // 2
    denominator = math.sqrt((pairs - tied_first) * (pairs - tied_second))
    if denominator == 0.0:
        raise ValueError("Kendall's tau is undefined for a constant sample")
    return max(-1.0, min(1.0, (concordant - discordant) / denominator))


def kendall_to_latent_correlation(tau: float) -> float:
    """Map Kendall's tau to the latent correlation of an elliptical copula.

    For every elliptical copula (Gaussian, Student-t)
    ``tau = (2 / pi) arcsin(rho)``, inverted as ``rho = sin(pi tau / 2)``.

    Args:
        tau: Kendall's tau in ``[-1, 1]``.

    Returns:
        Latent correlation in ``[-1, 1]``.

    Raises:
        TypeError: If ``tau`` is not real.
        ValueError: If ``tau`` is outside ``[-1, 1]``.

    Equation:
        [EQ-MOD-COPULA-FIT-005].

    References:
        [REF-MCNEIL-FREY-EMBRECHTS-2015], [REF-DEMARTA-MCNEIL-2005].

    """
    if isinstance(tau, bool) or not isinstance(tau, int | float):
        raise TypeError("tau must be a real number")
    value = float(tau)
    if not math.isfinite(value) or abs(value) > 1.0:
        raise ValueError("tau must lie in [-1, 1]")
    return max(-1.0, min(1.0, math.sin(math.pi * value / 2.0)))


def pseudo_observations(
    data: Mapping[str, Iterable[object]],
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Return rank pseudo-observations ``rank_ij / (n + 1)`` row by row.

    Args:
        data: Named columns of equal length (at least three rows).

    Returns:
        Column names and rows of pseudo-observations in ``(0, 1)``.

    Raises:
        TypeError: If ``data`` is malformed.
        ValueError: If columns are inconsistent.

    Equation:
        [EQ-MOD-COPULA-FIT-002], [EQ-STAT-RANK-001].

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995].

    """
    names, columns = _columns(data)
    scale = len(columns[0]) + 1.0
    ranked = tuple(tuple(rank / scale for rank in ranks(column)) for column in columns)
    return names, tuple(zip(*ranked, strict=True))


@dataclass(frozen=True, slots=True)
class CopulaFit:
    """A fitted copula with the empirical dependence it was fitted to.

    Args:
        names: Column names in order.
        family: Copula family name.
        copula: Fitted copula object.
        parameter: Fitted scalar parameter (``theta`` for Archimedean
            families, ``nu`` for the t copula, ``None`` otherwise).
        method: ``"kendall"`` (tau inversion) or ``"pseudo-likelihood"``.
        kendall_tau: Pairwise sample Kendall's tau matrix.
        log_pseudo_likelihood: ``sum_i log c(u_i)`` at the fit over the
            pseudo-observations.
        sample_size: Number of rows.
        repaired: Whether a latent correlation needed the nearest-correlation
            repair (elliptical families).
        at_boundary: Whether the pseudo-likelihood maximum lies at an end of
            the search range (the estimate is then a bound, not an optimum).

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995], [REF-JOE-2014].

    """

    names: tuple[str, ...]
    family: str
    copula: Copula
    parameter: float | None
    method: str
    kendall_tau: Matrix
    log_pseudo_likelihood: float
    sample_size: int
    repaired: bool
    at_boundary: bool


def _kendall_matrix(columns: tuple[tuple[float, ...], ...]) -> Matrix:
    """Return the pairwise sample Kendall's tau matrix.

    References:
        [REF-KENDALL-1938]. No new equation.

    """
    size = len(columns)
    matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            matrix[i][j] = matrix[j][i] = kendall_tau(columns[i], columns[j])
    return tuple(tuple(row) for row in matrix)


def _mean_off_diagonal(matrix: Matrix) -> float:
    """Return the average off-diagonal entry.

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995]. No new equation.

    """
    size = len(matrix)
    values = [matrix[i][j] for i in range(size) for j in range(i + 1, size)]
    return math.fsum(values) / len(values)


def archimedean_theta_from_tau(family: str, tau: float, dimension: int = 2) -> float:
    """Invert Kendall's tau for a one-parameter Archimedean family.

    Clayton ``theta = 2 tau / (1 - tau)``; Gumbel ``theta = 1 / (1 - tau)``;
    Frank solves ``1 + 4 (D_1(theta) - 1) / theta = tau`` by Brent's method.

    Args:
        family: ``"clayton"``, ``"gumbel"``, or ``"frank"``.
        tau: Kendall's tau.
        dimension: Copula dimension (negative Frank dependence needs ``d = 2``).

    Returns:
        Dependence parameter.

    Raises:
        ValueError: If the family is unknown or ``tau`` is outside the range
            the family can attain.

    Equation:
        [EQ-MOD-COPULA-FIT-004], [EQ-MOD-COP-FRANK-001].

    References:
        [REF-GENEST-RIVEST-1993], [REF-NELSEN-2006, Chapter 5].

    """
    if isinstance(tau, bool) or not isinstance(tau, int | float):
        raise TypeError("tau must be a real number")
    value = float(tau)
    if not math.isfinite(value) or not -1.0 < value < 1.0:
        raise ValueError("tau must lie strictly inside (-1, 1)")
    if family == "clayton":
        if value <= 0.0:
            raise ValueError("the Clayton copula needs a positive Kendall's tau")
        return 2.0 * value / (1.0 - value)
    if family == "gumbel":
        if value < 0.0:
            raise ValueError("the Gumbel copula needs a non-negative Kendall's tau")
        return 1.0 / (1.0 - value)
    if family == "frank":
        if value == 0.0:
            raise ValueError("Kendall's tau of zero is the independence copula")
        if value < 0.0 and dimension > 2:
            raise ValueError("negative Frank dependence needs dimension two")
        limit = 700.0
        if not frank_tau(-limit) < value < frank_tau(limit):
            raise ValueError("Kendall's tau is outside the attainable Frank range")
        low, high = (1e-12, limit) if value > 0.0 else (-limit, -1e-12)
        return brent_root(
            lambda theta: frank_tau(theta) - value, low, high, tolerance=1e-13
        ).root
    raise ValueError("family must be 'clayton', 'gumbel', or 'frank'")


def _build_copula(
    family: str, parameter: float, dimension: int, extra: Matrix | None
) -> Copula:
    """Construct a copula of a family from a scalar parameter.

    References:
        [REF-NELSEN-2006, Chapter 4]. No equation is evaluated.

    """
    if family == "clayton":
        return ClaytonCopula(parameter, dimension)
    if family == "gumbel":
        return GumbelCopula(parameter, dimension)
    if family == "frank":
        return FrankCopula(parameter, dimension)
    assert extra is not None
    return StudentTCopula(extra, parameter)


def _log_pseudo_likelihood(copula: Copula, rows: Sequence[Sequence[float]]) -> float:
    """Return ``sum_i log c(u_i)``.

    Equation:
        [EQ-MOD-COPULA-FIT-006].

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995].

    """
    return math.fsum(copula.log_density(row) for row in rows)


def _golden_maximize(
    function: Callable[[float], float], low: float, high: float, points: int = 41
) -> tuple[float, bool]:
    """Maximize a scalar function on ``[low, high]``: grid scan, then golden section.

    Returns:
        Arg-max and whether it lies in a boundary cell of the grid.

    Equation:
        [EQ-MOD-COPULA-FIT-006].

    References:
        [REF-BRENT-1973, Chapter 5], [REF-KIEFER-1953].

    """
    grid = [low + (high - low) * k / (points - 1) for k in range(points)]
    values = []
    for point in grid:
        try:
            values.append(function(point))
        except (ValueError, ArithmeticError, OverflowError):
            values.append(-math.inf)
    best = max(range(points), key=lambda k: values[k])
    if values[best] == -math.inf:
        raise ValueError("the pseudo-likelihood is not finite anywhere on the grid")
    at_boundary = best in (0, points - 1)
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, points - 1)]

    def safe(point: float) -> float:
        """Return the objective with failures mapped to ``-inf``.

        References:
            [REF-BRENT-1973, Chapter 5].

        """
        try:
            return function(point)
        except (ValueError, ArithmeticError, OverflowError):
            return -math.inf

    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = safe(c), safe(d)
    for _ in range(200):
        if abs(b - a) <= 1e-10 * max(1.0, abs(a) + abs(b)):
            break
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = safe(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = safe(d)
    return 0.5 * (a + b), at_boundary


def fit_copula(
    data: Mapping[str, Iterable[object]],
    family: CopulaFamily,
    *,
    method: CopulaMethod = "kendall",
    nu: float | None = None,
) -> CopulaFit:
    """Fit a copula to tabular data from ranks (marginal-free).

    ``"kendall"``: inversion of Kendall's tau. Elliptical families use
    ``R_ij = sin(pi tau_ij / 2)`` (repaired to the nearest correlation matrix
    with a :class:`UserWarning` if needed; the t copula then needs ``nu``);
    exchangeable Archimedean families invert the average pairwise tau.

    ``"pseudo-likelihood"``: maximum pseudo-likelihood
    ``max_theta sum_i log c(u_i; theta)`` over rank pseudo-observations
    ``u_ij = rank_ij / (n + 1)`` for the one-parameter families (Clayton,
    Gumbel, Frank), and over ``nu`` for the t copula with ``R`` from tau
    inversion. The maximum is located by a grid scan followed by
    golden-section search; ``at_boundary`` flags a maximum at the range end.
    Standard errors are not reported: the pseudo-likelihood information
    understates the variance because the margins are estimated by ranks.

    Args:
        data: Named columns of equal length (at least three rows, at least
            two columns).
        family: ``"independence"``, ``"gaussian"``, ``"student-t"``,
            ``"clayton"``, ``"gumbel"``, or ``"frank"``.
        method: ``"kendall"`` or ``"pseudo-likelihood"``.
        nu: Degrees of freedom for the t copula (required with
            ``"kendall"``, fixes ``nu`` with ``"pseudo-likelihood"``).

    Returns:
        Fitted copula with the empirical Kendall matrix and pseudo-likelihood.

    Raises:
        TypeError: If ``data`` is malformed.
        ValueError: If the family or method is unknown, the data have one
            column, or the dependence is outside the family's range.

    Equation:
        [EQ-MOD-COPULA-FIT-003], [EQ-MOD-COPULA-FIT-004],
        [EQ-MOD-COPULA-FIT-005], [EQ-MOD-COPULA-FIT-006].

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995], [REF-GENEST-RIVEST-1993],
        [REF-JOE-2014].

    """
    if family not in _COPULA_FAMILIES:
        raise ValueError(f"family must be one of {_COPULA_FAMILIES}")
    if method not in _COPULA_METHODS:
        raise ValueError(f"method must be one of {_COPULA_METHODS}")
    names, columns = _columns(data)
    dimension = len(names)
    if dimension < 2:
        raise ValueError("copula fitting needs at least two columns")
    _, rows = pseudo_observations(dict(zip(names, columns, strict=True)))
    taus = _kendall_matrix(columns)
    count = len(rows)
    if family == "independence":
        copula: Copula = IndependenceCopula(dimension)
        return CopulaFit(
            names, family, copula, None, method, taus, 0.0, count, False, False
        )
    if family in ("gaussian", "student-t"):
        latent = tuple(
            tuple(
                1.0 if i == j else kendall_to_latent_correlation(taus[i][j])
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        repair = nearest_correlation_matrix(latent)
        if repair.repaired:
            warnings.warn(
                "latent correlation from Kendall's tau was not positive definite "
                f"and was repaired (Frobenius distance {repair.distance:.3g})",
                UserWarning,
                stacklevel=2,
            )
        matrix = repair.matrix
        if family == "gaussian":
            if method == "pseudo-likelihood":
                raise ValueError(
                    "pseudo-likelihood is implemented for one-parameter families; "
                    "use method='kendall' for the Gaussian copula"
                )
            copula = GaussianCopula(matrix)
            return CopulaFit(
                names,
                family,
                copula,
                None,
                method,
                taus,
                _log_pseudo_likelihood(copula, rows),
                count,
                repair.repaired,
                False,
            )
        if nu is not None or method == "kendall":
            if nu is None:
                raise ValueError("the t copula fitted by Kendall's tau needs nu")
            copula = StudentTCopula(matrix, nu)
            return CopulaFit(
                names,
                family,
                copula,
                copula.nu,
                method,
                taus,
                _log_pseudo_likelihood(copula, rows),
                count,
                repair.repaired,
                False,
            )
        log_nu, at_boundary = _golden_maximize(
            lambda value: _log_pseudo_likelihood(
                StudentTCopula(matrix, math.exp(value)), rows
            ),
            math.log(0.5),
            math.log(500.0),
            points=25,
        )
        copula = StudentTCopula(matrix, math.exp(log_nu))
        return CopulaFit(
            names,
            family,
            copula,
            copula.nu,
            method,
            taus,
            _log_pseudo_likelihood(copula, rows),
            count,
            repair.repaired,
            at_boundary,
        )
    average = _mean_off_diagonal(taus)
    if method == "kendall":
        theta = archimedean_theta_from_tau(family, average, dimension)
        copula = _build_copula(family, theta, dimension, None)
        return CopulaFit(
            names,
            family,
            copula,
            theta,
            method,
            taus,
            _log_pseudo_likelihood(copula, rows),
            count,
            False,
            False,
        )
    transform: Callable[[float], float]
    if family == "clayton":
        low, high = math.log(1e-4), math.log(100.0)
        transform = math.exp
    elif family == "gumbel":
        low, high = math.log(1e-6), math.log(100.0)

        def transform(value: float) -> float:
            """Map the search variable to ``theta = 1 + e^value``.

            References:
                [REF-NELSEN-2006, Chapter 4]. No equation is evaluated.

            """
            return 1.0 + math.exp(value)

    elif dimension == 2:
        low, high = -60.0, 60.0

        def transform(value: float) -> float:
            """Map the search variable to a non-zero Frank ``theta``.

            References:
                [REF-NELSEN-2006, Chapter 4]. No equation is evaluated.

            """
            return value if value != 0.0 else 1e-12

    else:
        low, high = math.log(1e-4), math.log(60.0)
        transform = math.exp
    best, at_boundary = _golden_maximize(
        lambda value: _log_pseudo_likelihood(
            _build_copula(family, transform(value), dimension, None), rows
        ),
        low,
        high,
        points=61 if family == "frank" and dimension == 2 else 41,
    )
    theta = transform(best)
    copula = _build_copula(family, theta, dimension, None)
    return CopulaFit(
        names,
        family,
        copula,
        theta,
        method,
        taus,
        _log_pseudo_likelihood(copula, rows),
        count,
        False,
        at_boundary,
    )
