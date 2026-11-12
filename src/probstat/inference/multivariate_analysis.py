"""Principal components, maximum-likelihood factor analysis, and varimax.

- :func:`principal_components` diagonalizes the sample covariance (or
  correlation) matrix with the active dense backend's symmetric
  eigendecomposition (EQ-MVA-001).
- :func:`factor_analysis` fits ``Sigma = Lambda Lambda' + Psi`` to a
  correlation matrix by maximum likelihood: for fixed uniquenesses ``Psi``
  the loadings come from the eigendecomposition of
  ``Psi^-1/2 R Psi^-1/2``, and ``Psi`` minimizes the concentrated
  discrepancy (EQ-MVA-002); Bartlett's corrected likelihood-ratio statistic
  tests the number of factors (EQ-MVA-004).
- :func:`varimax` rotates loadings to Kaiser's varimax criterion by planar
  rotations (EQ-MVA-003).

References:
    [REF-PEARSON-1901], [REF-HOTELLING-1933], [REF-JOLLIFFE-2002],
    [REF-LAWLEY-MAXWELL-1971], [REF-JORESKOG-1967], [REF-KAISER-1958],
    [REF-MARDIA-KENT-BIBBY-1979].

"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.distributions import ChiSquare
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, get_backend, symmetrize
from probstat.optimization import minimize_bfgs

__all__ = [
    "FactorAnalysisResult",
    "PrincipalComponents",
    "VarimaxResult",
    "factor_analysis",
    "principal_components",
    "varimax",
]

_MIN_UNIQUENESS = 0.005


def _rows(data: Iterable[Iterable[object]], *, minimum: int = 2) -> tuple[Vector, ...]:
    """Validate a data matrix given as rows of equal length.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    rows: list[Vector] = []
    for row in data:
        values = tuple(float(require_finite_real(v, name="data")) for v in row)
        rows.append(values)
    if len(rows) < minimum:
        raise ValueError(f"data must contain at least {minimum} rows")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("rows must be non-empty and of equal length")
    return tuple(rows)


def _mean_and_covariance(rows: Sequence[Vector]) -> tuple[Vector, Matrix]:
    """Return the column means and the unbiased covariance matrix.

    References:
        [REF-MARDIA-KENT-BIBBY-1979]. No named equation is evaluated.

    """
    n, p = len(rows), len(rows[0])
    means = tuple(math.fsum(row[j] for row in rows) / n for j in range(p))
    centred = [[row[j] - means[j] for j in range(p)] for row in rows]
    covariance = tuple(
        tuple(math.fsum(r[i] * r[j] for r in centred) / (n - 1) for j in range(p))
        for i in range(p)
    )
    return means, covariance


def _sign_fix(vector: Sequence[float]) -> Vector:
    """Flip a vector so that its entry of largest magnitude is positive.

    References:
        [REF-JOLLIFFE-2002]. No equation is evaluated.

    """
    pivot = max(range(len(vector)), key=lambda i: abs(vector[i]))
    sign = -1.0 if vector[pivot] < 0.0 else 1.0
    return tuple(sign * v for v in vector)


def _descending_eigen(matrix: Matrix) -> tuple[Vector, tuple[Vector, ...]]:
    """Return eigenvalues in descending order and the eigenvectors as rows.

    References:
        [EQ-LINALG-005], [REF-GOLUB-VANLOAN-2013, Section 8.5].

    """
    values, vectors = get_backend().symmetric_eigen(matrix)
    size = len(values)
    order = list(range(size - 1, -1, -1))
    return (
        tuple(values[i] for i in order),
        tuple(tuple(vectors[r][i] for r in range(size)) for i in order),
    )


# ---------------------------------------------------------------------------
# Principal components
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrincipalComponents:
    """Principal component analysis of a data matrix.

    Args:
        means: Column means.
        scales: Column scales divided out before the analysis (the sample
            standard deviations when standardized, otherwise ones).
        standardized: Whether the correlation matrix was analysed.
        variances: All eigenvalues ``l_1 >= ... >= l_p`` (component
            variances).
        axes: Unit eigenvectors of the retained components, one per row,
            each signed so that its largest-magnitude entry is positive.
        scores: Component scores ``z_ik = a_k' (x_i - m)/s`` of the data.

    References:
        [EQ-MVA-001], [REF-JOLLIFFE-2002], [REF-HOTELLING-1933].

    """

    means: Vector
    scales: Vector
    standardized: bool
    variances: Vector
    axes: tuple[Vector, ...]
    scores: Matrix

    @property
    def explained_variance_ratio(self) -> Vector:
        """Return ``l_k / sum_j l_j`` for the retained components.

        References:
            [EQ-MVA-001], [REF-JOLLIFFE-2002].

        """
        total = math.fsum(max(v, 0.0) for v in self.variances)
        return tuple(max(v, 0.0) / total for v in self.variances[: len(self.axes)])

    @property
    def cumulative_variance_ratio(self) -> Vector:
        """Return the running sum of :attr:`explained_variance_ratio`.

        References:
            [EQ-MVA-001], [REF-JOLLIFFE-2002].

        """
        out: list[float] = []
        total = 0.0
        for value in self.explained_variance_ratio:
            total += value
            out.append(total)
        return tuple(out)

    @property
    def loadings(self) -> Matrix:
        """Return ``a_jk sqrt(l_k)`` with variables as rows.

        For a standardized analysis these are the correlations between the
        variables and the components.

        References:
            [EQ-MVA-001], [REF-JOLLIFFE-2002].

        """
        k = len(self.axes)
        return tuple(
            tuple(
                self.axes[c][j] * math.sqrt(max(self.variances[c], 0.0))
                for c in range(k)
            )
            for j in range(len(self.means))
        )

    def transform(self, data: Iterable[Iterable[object]]) -> Matrix:
        """Return the component scores of new rows.

        Args:
            data: Rows with the same columns as the fitted data.

        Returns:
            Scores, one row per input row.

        Raises:
            ValueError: For rows of the wrong length.

        References:
            [EQ-MVA-001], [REF-JOLLIFFE-2002].

        """
        rows = _rows(data, minimum=1)
        if len(rows[0]) != len(self.means):
            raise ValueError("rows must have one value per fitted column")
        return tuple(
            tuple(
                math.fsum(
                    a[j] * (row[j] - self.means[j]) / self.scales[j]
                    for j in range(len(row))
                )
                for a in self.axes
            )
            for row in rows
        )

    def reconstruct(self, scores: Iterable[Iterable[object]]) -> Matrix:
        """Map scores back to the data scale, ``m + s * sum_k z_k a_k``.

        Args:
            scores: Rows of retained-component scores.

        Returns:
            Reconstructed rows (exact with every component retained).

        References:
            [EQ-MVA-001], [REF-JOLLIFFE-2002].

        """
        rows = _rows(scores, minimum=1)
        if len(rows[0]) != len(self.axes):
            raise ValueError("score rows must have one value per retained component")
        p = len(self.means)
        return tuple(
            tuple(
                self.means[j]
                + self.scales[j]
                * math.fsum(z[c] * self.axes[c][j] for c in range(len(z)))
                for j in range(p)
            )
            for z in rows
        )


def principal_components(
    data: Iterable[Iterable[object]],
    *,
    standardize: bool = False,
    components: int | None = None,
) -> PrincipalComponents:
    """Compute the principal components of a data matrix.

    The sample covariance ``S`` (divisor ``n - 1``), or the correlation
    matrix with ``standardize=True``, is diagonalized as ``S = A L A'``
    through ``linalg.get_backend().symmetric_eigen`` (EQ-MVA-001): the
    ``k``-th axis ``a_k`` maximizes the variance ``a'Sa`` subject to unit
    length and orthogonality to the earlier axes, and that variance is the
    eigenvalue ``l_k``.

    Args:
        data: Rows of observations (``n >= 2``).
        standardize: Analyse the correlation matrix (variables on different
            scales) instead of the covariance matrix.
        components: Number of components to retain (default all).

    Returns:
        The principal components.

    Raises:
        ValueError: For ragged rows, too few rows, a constant column with
            ``standardize=True``, or an invalid ``components``.

    Equation:
        [EQ-MVA-001], [EQ-LINALG-005].

    References:
        [REF-PEARSON-1901], [REF-HOTELLING-1933], [REF-JOLLIFFE-2002].

    """
    rows = _rows(data)
    p = len(rows[0])
    k = (
        p
        if components is None
        else require_non_negative_integer(components, name="components")
    )
    if not 1 <= k <= p:
        raise ValueError("components must lie in [1, number of columns]")
    means, covariance = _mean_and_covariance(rows)
    if standardize:
        scales = tuple(math.sqrt(covariance[j][j]) for j in range(p))
        if any(not s > 0.0 for s in scales):
            raise ValueError("a constant column cannot be standardized")
        matrix = tuple(
            tuple(covariance[i][j] / (scales[i] * scales[j]) for j in range(p))
            for i in range(p)
        )
    else:
        scales = (1.0,) * p
        matrix = covariance
    values, vectors = _descending_eigen(matrix)
    axes = tuple(_sign_fix(v) for v in vectors[:k])
    scores = tuple(
        tuple(
            math.fsum(a[j] * (row[j] - means[j]) / scales[j] for j in range(p))
            for a in axes
        )
        for row in rows
    )
    return PrincipalComponents(
        means=means,
        scales=scales,
        standardized=bool(standardize),
        variances=values,
        axes=axes,
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Varimax
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VarimaxResult:
    """Varimax-rotated loadings.

    Args:
        loadings: Rotated loadings ``Lambda T`` (variables as rows).
        rotation: Orthogonal rotation matrix ``T``.
        criterion: Raw varimax criterion of the rotated loadings.
        sweeps: Number of sweeps over all factor pairs.

    References:
        [EQ-MVA-003], [REF-KAISER-1958].

    """

    loadings: Matrix
    rotation: Matrix
    criterion: float
    sweeps: int


def _varimax_criterion(loadings: Sequence[Sequence[float]]) -> float:
    """Return ``sum_j [p sum_i l_ij^4 - (sum_i l_ij^2)^2] / p^2``.

    References:
        [REF-KAISER-1958]. Equation [EQ-MVA-003].

    """
    p = len(loadings)
    k = len(loadings[0])
    total = 0.0
    for j in range(k):
        squares = [row[j] ** 2 for row in loadings]
        total += (p * math.fsum(s * s for s in squares) - math.fsum(squares) ** 2) / (
            p * p
        )
    return total


def varimax(
    loadings: Iterable[Iterable[object]],
    *,
    normalize: bool = True,
    tolerance: float = 1e-10,
    max_sweeps: int = 1000,
) -> VarimaxResult:
    """Rotate loadings to maximize Kaiser's varimax criterion.

    Each planar rotation of columns ``a`` and ``b`` by the angle
    ``phi = atan2(D - 2AB/p, C - (A^2 - B^2)/p) / 4``, with
    ``u = x^2 - y^2``, ``v = 2xy``, ``A = sum u``, ``B = sum v``,
    ``C = sum (u^2 - v^2)``, and ``D = 2 sum uv``, maximizes the criterion
    over that pair (EQ-MVA-003); sweeps over all pairs stop when every
    angle is below ``tolerance``. With ``normalize`` (Kaiser
    normalization) rows are scaled to unit length first and back after.

    Args:
        loadings: Matrix with variables as rows and at least one factor.
        normalize: Apply Kaiser normalization.
        tolerance: Angle threshold in radians.
        max_sweeps: Maximum number of sweeps.

    Returns:
        The rotated loadings and the rotation.

    Raises:
        ValueError: For ragged input or no convergence.

    Equation:
        [EQ-MVA-003].

    References:
        [REF-KAISER-1958].

    """
    rows = _rows(loadings, minimum=1)
    p, k = len(rows), len(rows[0])
    norms = [
        math.sqrt(math.fsum(v * v for v in row)) if normalize else 1.0 for row in rows
    ]
    work = [
        [v / n if n > 0.0 else v for v in row]
        for row, n in zip(rows, norms, strict=True)
    ]
    rotation = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    limit = require_non_negative_integer(max_sweeps, name="max_sweeps")
    tol = require_finite_real(tolerance, name="tolerance")
    sweeps = 0
    if k > 1:
        for sweep in range(1, limit + 1):
            sweeps = sweep
            largest = 0.0
            for a in range(k - 1):
                for b in range(a + 1, k):
                    u = [row[a] ** 2 - row[b] ** 2 for row in work]
                    v = [2.0 * row[a] * row[b] for row in work]
                    big_a, big_b = math.fsum(u), math.fsum(v)
                    big_c = math.fsum(x * x - y * y for x, y in zip(u, v, strict=True))
                    big_d = 2.0 * math.fsum(x * y for x, y in zip(u, v, strict=True))
                    numerator = big_d - 2.0 * big_a * big_b / p
                    denominator = big_c - (big_a * big_a - big_b * big_b) / p
                    phi = 0.25 * math.atan2(numerator, denominator)
                    largest = max(largest, abs(phi))
                    if phi == 0.0:
                        continue
                    c, s = math.cos(phi), math.sin(phi)
                    for row in work:
                        x, y = row[a], row[b]
                        row[a], row[b] = c * x + s * y, -s * x + c * y
                    for row in rotation:
                        x, y = row[a], row[b]
                        row[a], row[b] = c * x + s * y, -s * x + c * y
            if largest <= tol:
                break
        else:
            raise ValueError("varimax did not converge")
    rotated = tuple(
        tuple(v * n for v in row) for row, n in zip(work, norms, strict=True)
    )
    return VarimaxResult(
        loadings=rotated,
        rotation=tuple(tuple(row) for row in rotation),
        criterion=_varimax_criterion(rotated),
        sweeps=sweeps,
    )


# ---------------------------------------------------------------------------
# Maximum-likelihood factor analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactorAnalysisResult:
    """Maximum-likelihood factor analysis of a correlation matrix.

    Args:
        loadings: Loadings ``Lambda`` (variables as rows, factors as
            columns), rotated if requested, columns ordered by decreasing
            sum of squares and signed to a positive column sum.
        uniquenesses: Specific variances ``psi_j``.
        rotation: Rotation applied to the unrotated ML loadings (identity
            without rotation).
        objective: Minimized discrepancy ``F = sum_(j>q)(theta_j - log
            theta_j - 1)``.
        statistic: Bartlett-corrected likelihood-ratio statistic (``nan``
            without ``observations``).
        degrees_of_freedom: ``((p - q)^2 - p - q)/2``.
        p_value: Upper chi-square tail of ``statistic`` (``nan`` when it is
            not defined).
        converged: Whether the optimizer met its tolerance.

    References:
        [EQ-MVA-002], [EQ-MVA-004], [REF-LAWLEY-MAXWELL-1971], [REF-JORESKOG-1967].

    """

    loadings: Matrix
    uniquenesses: Vector
    rotation: Matrix
    objective: float
    statistic: float
    degrees_of_freedom: float
    p_value: float
    converged: bool

    @property
    def communalities(self) -> Vector:
        """Return ``h_j^2 = sum_k lambda_jk^2 = 1 - psi_j``.

        References:
            [EQ-MVA-002], [REF-LAWLEY-MAXWELL-1971].

        """
        return tuple(math.fsum(v * v for v in row) for row in self.loadings)


def _to_correlation(matrix: Matrix) -> Matrix:
    """Scale a covariance matrix to a correlation matrix.

    References:
        [REF-MARDIA-KENT-BIBBY-1979]. No named equation is evaluated.

    """
    p = len(matrix)
    sd = [math.sqrt(matrix[i][i]) for i in range(p)]
    if any(not s > 0.0 for s in sd):
        raise ValueError("every variable must have positive variance")
    return tuple(
        tuple(matrix[i][j] / (sd[i] * sd[j]) for j in range(p)) for i in range(p)
    )


def _scaled_eigen(
    correlation: Matrix, psi: Sequence[float]
) -> tuple[Vector, tuple[Vector, ...]]:
    """Return the descending eigensystem of ``Psi^-1/2 R Psi^-1/2``.

    References:
        [REF-JORESKOG-1967]. Equation [EQ-MVA-002].

    """
    p = len(psi)
    root = [1.0 / math.sqrt(v) for v in psi]
    scaled = tuple(
        tuple(correlation[i][j] * root[i] * root[j] for j in range(p)) for i in range(p)
    )
    return _descending_eigen(symmetrize(scaled, relative_tolerance=1e-9))


def _ml_loadings(correlation: Matrix, psi: Sequence[float], factors: int) -> Matrix:
    """Return ``Psi^1/2 Omega_q (Theta_q - I)^1/2`` (variables as rows).

    References:
        [REF-JORESKOG-1967], [REF-LAWLEY-MAXWELL-1971]. Equation [EQ-MVA-002].

    """
    values, vectors = _scaled_eigen(correlation, psi)
    p = len(psi)
    return tuple(
        tuple(
            math.sqrt(psi[i]) * vectors[c][i] * math.sqrt(max(values[c] - 1.0, 0.0))
            for c in range(factors)
        )
        for i in range(p)
    )


def factor_analysis(
    data: Iterable[Iterable[object]] | None = None,
    *,
    factors: int,
    covariance: Iterable[Iterable[object]] | None = None,
    observations: int | None = None,
    rotation: str | None = "varimax",
) -> FactorAnalysisResult:
    """Fit the maximum-likelihood factor model ``R = Lambda Lambda' + Psi``.

    For fixed ``Psi`` the likelihood is maximized by
    ``Lambda = Psi^1/2 Omega_q (Theta_q - I)^1/2`` from the ``q`` largest
    eigenvalues ``theta`` of ``Psi^-1/2 R Psi^-1/2``; the concentrated
    discrepancy ``F(Psi) = sum_(j>q)(theta_j - log theta_j - 1)`` is then
    minimized over ``0.005 <= psi_j <= 1`` by BFGS with its analytic
    gradient ``diag(Lambda Lambda' + Psi - R) / psi^2`` (EQ-MVA-002), from
    the start ``psi_j = (1 - q/(2p)) / (R^-1)_jj``. With ``observations``,
    Bartlett's statistic ``(n - 1 - (2p + 5)/6 - 2q/3) F`` is referred to
    chi-square with ``((p - q)^2 - p - q)/2`` degrees of freedom
    (EQ-MVA-004).

    Args:
        data: Rows of observations; the sample size defaults to their count.
        factors: Number of factors ``q >= 1``.
        covariance: A covariance or correlation matrix instead of ``data``.
        observations: Sample size for the test (required with
            ``covariance`` for a p-value).
        rotation: ``"varimax"`` (default) or ``None``.

    Returns:
        The factor-analysis result on the correlation scale.

    Raises:
        ValueError: If neither or both inputs are given, the model has
            negative degrees of freedom, or the options are invalid.

    Equation:
        [EQ-MVA-002], [EQ-MVA-003], [EQ-MVA-004].

    References:
        [REF-LAWLEY-MAXWELL-1971], [REF-JORESKOG-1967], [REF-KAISER-1958].

    """
    if (data is None) == (covariance is None):
        raise ValueError("give exactly one of data and covariance")
    if data is not None:
        rows = _rows(data, minimum=3)
        _, matrix = _mean_and_covariance(rows)
        n_obs: int | None = len(rows) if observations is None else int(observations)
    else:
        matrix = tuple(
            tuple(require_finite_real(v, name="covariance") for v in row)
            for row in covariance or ()
        )
        n_obs = None if observations is None else int(observations)
    correlation = _to_correlation(symmetrize(matrix, relative_tolerance=1e-9))
    p = len(correlation)
    q = require_non_negative_integer(factors, name="factors")
    dof = 0.5 * ((p - q) ** 2 - p - q)
    if q < 1 or dof < 0:
        raise ValueError(
            "factors must be at least 1 and leave non-negative degrees of freedom"
        )
    if rotation not in ("varimax", None):
        raise ValueError("rotation must be 'varimax' or None")
    inverse = get_backend().inverse_spd(correlation)
    start = [
        min(1.0, max(_MIN_UNIQUENESS, (1.0 - 0.5 * q / p) / inverse[j][j]))
        for j in range(p)
    ]

    def objective(psi: Sequence[float]) -> float:
        """Return ``F(Psi)``.

        References:
            [REF-JORESKOG-1967]. Equation [EQ-MVA-002].

        """
        values, _ = _scaled_eigen(correlation, psi)
        tail = values[q:]
        if any(not v > 0.0 for v in tail):
            return math.inf
        # theta - log(theta) - 1 = d - log1p(d) with d = theta - 1, without
        # the cancellation of the direct form near the optimum theta = 1.
        return math.fsum((v - 1.0) - math.log1p(v - 1.0) for v in tail)

    def gradient(psi: Sequence[float]) -> Vector:
        """Return ``dF/dpsi_j = (Lambda Lambda' + Psi - R)_jj / psi_j^2``.

        References:
            [REF-JORESKOG-1967]. Equation [EQ-MVA-002].

        """
        lam = _ml_loadings(correlation, psi, q)
        return tuple(
            (math.fsum(v * v for v in lam[j]) + psi[j] - correlation[j][j])
            / (psi[j] ** 2)
            for j in range(p)
        )

    result = minimize_bfgs(
        objective,
        start,
        gradient=gradient,
        gradient_tolerance=1e-10,
        bounds=[(_MIN_UNIQUENESS, 1.0)] * p,
    )
    psi = tuple(result.x)
    unrotated = _ml_loadings(correlation, psi, q)
    if rotation == "varimax" and q > 1:
        rotated = varimax(unrotated)
        loadings, turn = rotated.loadings, rotated.rotation
    else:
        loadings = unrotated
        turn = tuple(tuple(1.0 if i == j else 0.0 for j in range(q)) for i in range(q))
    order = sorted(range(q), key=lambda c: -math.fsum(row[c] ** 2 for row in loadings))
    signs = [
        1.0 if math.fsum(row[c] for row in loadings) >= 0.0 else -1.0 for c in range(q)
    ]
    loadings = tuple(tuple(signs[c] * row[c] for c in order) for row in loadings)
    turn = tuple(tuple(signs[c] * row[c] for c in order) for row in turn)
    value = objective(psi)
    if n_obs is not None:
        statistic = (n_obs - 1 - (2 * p + 5) / 6.0 - 2 * q / 3.0) * value
        p_value = ChiSquare(dof).sf(statistic) if dof > 0 else math.nan
    else:
        statistic = p_value = math.nan
    return FactorAnalysisResult(
        loadings=loadings,
        uniquenesses=psi,
        rotation=turn,
        objective=value,
        statistic=statistic,
        degrees_of_freedom=dof,
        p_value=p_value,
        converged=result.converged,
    )
