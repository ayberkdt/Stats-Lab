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
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
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
from probstat.statistics import ranks, spearman_correlation

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
