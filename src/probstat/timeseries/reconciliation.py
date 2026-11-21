"""Forecast reconciliation for hierarchical and grouped time series.

A collection of ``n`` series is *coherent* when it satisfies its aggregation
constraints, ``y = S b``: ``b`` holds the ``m`` bottom-level series and the
``n x m`` summing matrix ``S`` has one row per series whose entries say which
bottom series it adds up. Base forecasts ``y_hat`` produced series by series
are usually not coherent. A linear reconciler maps them to
``y_tilde = S G y_hat`` with an ``m x n`` matrix ``G``; it keeps unbiased
base forecasts unbiased exactly when ``S G S = S``, that is ``G S = I``
[EQ-TS-RECON-001].

For base-forecast errors with covariance ``W`` the reconciled errors have
covariance ``S G W G' S'``, and Wickramasuriya, Athanasopoulos and Hyndman
(2019) show that its trace is smallest, among all ``G`` with ``G S = I``,
for the generalized-least-squares choice [EQ-TS-RECON-002]::

    G = (S' W^-1 S)^-1 S' W^-1

The methods here differ only in the ``W`` they plug in:

* ``"ols"``: ``W = I``, the orthogonal projection onto the column space of
  ``S``;
* ``"wls"``: ``W`` diagonal with the base-error variances;
* ``"mint"``: a full covariance, typically the shrunk estimate of
  :func:`shrink_covariance` [EQ-TS-RECON-003];
* ``"bottom_up"``: ``G`` selects the bottom-level base forecasts and
  ignores the others (the baseline; it also satisfies ``G S = I``).

The optimality of ``"mint"`` holds for the *true* ``W``. With an estimated
``W`` the reconciler is still a projection onto the coherent subspace
(coherent, idempotent, unbiasedness-preserving), but it need not beat OLS.
Only point forecasts are reconciled; no prediction intervals are produced.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    cholesky_solve,
    inverse_spd,
    matmul,
    matvec,
    transpose,
)

_METHODS = ("ols", "wls", "mint", "bottom_up")


@dataclass(frozen=True, slots=True)
class HierarchyStructure:
    """Series names and the summing matrix of an aggregation structure.

    Args:
        names: All series, aggregates first (in the order given) and then the
            bottom-level series.
        bottom_names: Bottom-level series in column order of ``matrix``.
        matrix: ``n x m`` summing matrix ``S`` of zeros and ones; row ``i``
            marks the bottom series that add up to ``names[i]``.

    References:
        [EQ-TS-RECON-001],
        [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

    """

    names: tuple[str, ...]
    bottom_names: tuple[str, ...]
    matrix: Matrix

    def aggregate(self, bottom: Sequence[float]) -> Vector:
        """Return the coherent vector ``S b`` of all series.

        Args:
            bottom: Bottom-level values in the order of ``bottom_names``.

        Returns:
            Values of every series in the order of ``names``.

        Raises:
            TypeError: If a value is not real.
            ValueError: If the length differs from the number of bottom
                series or a value is not finite.

        Equation:
            [EQ-TS-RECON-001].

        References:
            [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

        """
        values = as_vector(bottom, name="bottom")
        if len(values) != len(self.bottom_names):
            raise ValueError("bottom must have one value per bottom-level series")
        return matvec(self.matrix, values)


def _leaves(
    node: str,
    children: Mapping[str, tuple[str, ...]],
    cache: dict[str, tuple[str, ...]],
    path: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the bottom-level descendants of a node, depth first.

    Args:
        node: Series name.
        children: Parent-to-children specification.
        cache: Memo of already resolved nodes.
        path: Ancestors on the current recursion path (cycle detection).

    Returns:
        Bottom-level names in order of first visit.

    Raises:
        ValueError: If the specification has a cycle or a bottom series is
            reached twice below one aggregate.

    References:
        [EQ-TS-RECON-001]. No equation is evaluated.

    """
    if node in cache:
        return cache[node]
    if node in path:
        raise ValueError(f"the hierarchy has a cycle through {node!r}")
    if node not in children:
        return (node,)
    found: list[str] = []
    for child in children[node]:
        for leaf in _leaves(child, children, cache, (*path, node)):
            if leaf in found:
                raise ValueError(
                    f"bottom series {leaf!r} is counted twice below {node!r}"
                )
            found.append(leaf)
    cache[node] = tuple(found)
    return cache[node]


def hierarchy_summing_matrix(
    hierarchy: Mapping[str, Sequence[str]],
) -> HierarchyStructure:
    """Build the summing matrix from a parent-to-children specification.

    Each key is an aggregate and its value lists the series that add up to
    it; names that never occur as a key are bottom-level series. A strict
    hierarchy is a tree. A grouped structure is given by listing every
    aggregate of every grouping with its own children, for example
    ``{"total": ["a", "b", "c", "d"], "north": ["a", "b"], "south":
    ["c", "d"], "new": ["a", "c"], "old": ["b", "d"]}``. Row ``i`` of the
    matrix has a one for every bottom-level descendant of series ``i``.

    Args:
        hierarchy: Mapping from aggregate name to its non-empty children.

    Returns:
        Names (aggregates in key order, then bottom series in order of first
        appearance) and the summing matrix.

    Raises:
        TypeError: If a name is not a string.
        ValueError: If the specification is empty, an aggregate has no
            children or repeats a child, a cycle exists, or a bottom series
            would be counted twice in one aggregate.

    Equation:
        [EQ-TS-RECON-001].

    References:
        [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

    """
    if not hierarchy:
        raise ValueError("hierarchy must name at least one aggregate")
    children: dict[str, tuple[str, ...]] = {}
    for parent, members in hierarchy.items():
        listed = tuple(members)
        if not isinstance(parent, str) or any(
            not isinstance(member, str) for member in listed
        ):
            raise TypeError("series names must be strings")
        if not listed:
            raise ValueError(f"aggregate {parent!r} has no children")
        if len(set(listed)) != len(listed):
            raise ValueError(f"aggregate {parent!r} lists a child twice")
        children[parent] = listed
    bottom: list[str] = []
    for members in children.values():
        for member in members:
            if member not in children and member not in bottom:
                bottom.append(member)
    cache: dict[str, tuple[str, ...]] = {}
    position = {name: index for index, name in enumerate(bottom)}
    rows: list[Vector] = []
    for parent in children:
        row = [0.0] * len(bottom)
        for leaf in _leaves(parent, children, cache, ()):
            row[position[leaf]] = 1.0
        rows.append(tuple(row))
    for index in range(len(bottom)):
        rows.append(tuple(1.0 if j == index else 0.0 for j in range(len(bottom))))
    return HierarchyStructure((*children, *bottom), tuple(bottom), tuple(rows))


@dataclass(frozen=True, slots=True)
class ShrunkCovariance:
    """Sample covariance shrunk towards its diagonal.

    Args:
        covariance: ``lambda diag(C) + (1 - lambda) C`` for the sample
            covariance ``C`` (divisor ``N - 1``).
        intensity: Shrinkage intensity ``lambda`` in ``[0, 1]``.
        estimated: Whether ``lambda`` was estimated from the residuals
            (``False`` when supplied by the caller).
        sample_size: Number ``N`` of residual vectors.

    References:
        [EQ-TS-RECON-003],
        [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

    """

    covariance: Matrix
    intensity: float
    estimated: bool
    sample_size: int


def shrink_covariance(
    residuals: Sequence[Sequence[float]], *, intensity: float | None = None
) -> ShrunkCovariance:
    """Shrink the sample covariance of forecast errors towards its diagonal.

    The estimate is ``W = lambda diag(C) + (1 - lambda) C``: variances are
    kept and every covariance is multiplied by ``1 - lambda``. For any
    ``lambda > 0`` and positive variances ``W`` is positive definite even
    when there are fewer residual vectors than series.

    When ``intensity`` is omitted it is estimated on the correlation scale.
    Among fixed ``lambda`` the expected squared error
    ``sum_(i != j) E[((1 - lambda) r_ij - rho_ij)^2]`` of the shrunk
    correlations is smallest at
    ``lambda* = sum_(i != j) Var(r_ij) / sum_(i != j) E[r_ij^2]`` when
    ``r_ij`` is unbiased for ``rho_ij``. The plug-in estimate replaces
    ``E[r_ij^2]`` by ``r_ij^2`` and ``Var(r_ij)`` by the sampling variance of
    the mean of the products ``w_kij = z_ki z_kj`` of standardized residuals,
    ``N / (N - 1)^3 sum_k (w_kij - w_bar_ij)^2``, and clips the ratio to
    ``[0, 1]``. It is a plug-in estimate: it ignores the error of the
    standardization and is not itself the optimum.

    This plug-in is the shrinkage intensity of Schäfer and Strimmer for a
    correlation matrix shrunk towards the identity (variances unshrunk),
    which is the estimator used with MinT; the tests compare it with a
    literal evaluation of that formula
    (``sum_(i != j) Var_hat(r_ij) / sum_(i != j) r_ij^2``) and agree to
    rounding. One convention differs from common MinT software: the
    residuals are centred here (column means subtracted, divisor
    ``N - 1``), whereas implementations that treat forecast errors as having
    mean zero use the uncentred cross products with divisor ``N``. The two
    intensities differ noticeably on short samples (0.65 against 0.61 with
    12 residual vectors of 8 series, 0.75 against 0.89 with 20 vectors of 6
    series in one simulated example); compute the uncentred value separately
    and pass it as ``intensity`` if it is required.

    Args:
        residuals: ``N >= 2`` vectors of base-forecast errors, one row per
            time point and one column per series.
        intensity: Optional fixed ``lambda`` in ``[0, 1]``; estimated when
            omitted.

    Returns:
        Shrunk covariance and the intensity used.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If fewer than two rows are given, rows are ragged or not
            finite, a series has zero variance, or ``intensity`` is outside
            ``[0, 1]``.

    Equation:
        [EQ-TS-RECON-003].

    References:
        [REF-SCHAFER-STRIMMER-2005],
        [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

    """
    rows = as_matrix(residuals, name="residuals")
    count, width = len(rows), len(rows[0])
    if count < 2:
        raise ValueError("residuals need at least two rows")
    means = [math.fsum(row[j] for row in rows) / count for j in range(width)]
    centered = [[row[j] - means[j] for j in range(width)] for row in rows]
    variances = [
        math.fsum(row[j] * row[j] for row in centered) / (count - 1)
        for j in range(width)
    ]
    if any(not value > 0.0 for value in variances):
        raise ValueError("every residual series needs a positive variance")
    deviations = [math.sqrt(value) for value in variances]
    standardized = [[row[j] / deviations[j] for j in range(width)] for row in centered]
    correlation = [[1.0] * width for _ in range(width)]
    numerator: list[float] = []
    denominator: list[float] = []
    for i in range(width):
        for j in range(i + 1, width):
            products = [row[i] * row[j] for row in standardized]
            average = math.fsum(products) / count
            value = count / (count - 1) * average
            correlation[i][j] = correlation[j][i] = value
            numerator.append(
                count
                / (count - 1) ** 3
                * math.fsum((product - average) ** 2 for product in products)
            )
            denominator.append(value * value)
    if intensity is None:
        total = math.fsum(denominator)
        weight = 1.0 if total == 0.0 else math.fsum(numerator) / total
        weight = min(1.0, max(0.0, weight))
    else:
        weight = require_finite_real(intensity, name="intensity")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("intensity must lie in [0, 1]")
    covariance = tuple(
        tuple(
            variances[i]
            if i == j
            else (1.0 - weight) * correlation[i][j] * deviations[i] * deviations[j]
            for j in range(width)
        )
        for i in range(width)
    )
    return ShrunkCovariance(covariance, weight, intensity is None, count)


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Reconciled forecasts and the linear map that produced them.

    Args:
        method: ``"ols"``, ``"wls"``, ``"mint"`` or ``"bottom_up"``.
        base: Base forecasts ``y_hat`` (all series).
        reconciled: Coherent forecasts ``y_tilde = S G y_hat``.
        bottom: Reconciled bottom-level forecasts ``G y_hat``.
        mapping: The ``m x n`` matrix ``G``.
        summing: The ``n x m`` summing matrix ``S``.
        base_incoherence: Euclidean distance from ``y_hat`` to the coherent
            subspace, ``|y_hat - S (S'S)^-1 S' y_hat|``; zero exactly when
            the base forecasts already add up.

    References:
        [EQ-TS-RECON-001], [EQ-TS-RECON-002],
        [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

    """

    method: str
    base: Vector
    reconciled: Vector
    bottom: Vector
    mapping: Matrix
    summing: Matrix
    base_incoherence: float

    @property
    def adjustment(self) -> Vector:
        """Return ``y_tilde - y_hat``, the change made to each series.

        References:
            [EQ-TS-RECON-001]. No new equation is evaluated.

        """
        return tuple(
            new - old for new, old in zip(self.reconciled, self.base, strict=True)
        )

    def projection(self) -> Matrix:
        """Return the ``n x n`` matrix ``S G`` applied to the base forecasts.

        Returns:
            The reconciliation matrix; it is idempotent because ``G S = I``.

        Equation:
            [EQ-TS-RECON-001].

        References:
            [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

        """
        return matmul(self.summing, self.mapping)

    def error_covariance(self, covariance: Sequence[Sequence[float]]) -> Matrix:
        """Return ``S G W G' S'``, the covariance of the reconciled errors.

        Args:
            covariance: Covariance ``W`` of the base-forecast errors
                (``n x n``, symmetric; symmetry is not checked).

        Returns:
            Covariance of ``y - y_tilde`` for unbiased base forecasts with
            error covariance ``W``.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If the matrix is not ``n x n`` or not finite.

        Equation:
            [EQ-TS-RECON-002].

        References:
            [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

        """
        weights = as_matrix(covariance, name="covariance")
        size = len(self.base)
        if len(weights) != size or len(weights[0]) != size:
            raise ValueError("covariance must be square with one row per series")
        projector = self.projection()
        return matmul(matmul(projector, weights), transpose(projector))

    def total_error_variance(self, covariance: Sequence[Sequence[float]]) -> float:
        """Return ``tr(S G W G' S')``, the summed reconciled error variance.

        Args:
            covariance: Covariance ``W`` of the base-forecast errors.

        Returns:
            The quantity minimized by ``"mint"`` when ``W`` is the true
            covariance.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If the matrix is not ``n x n`` or not finite.

        Equation:
            [EQ-TS-RECON-002].

        References:
            [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

        """
        matrix = self.error_covariance(covariance)
        return math.fsum(matrix[i][i] for i in range(len(matrix)))


def _generalized_mapping(summing: Matrix, weights: Matrix | None) -> Matrix:
    """Return ``G = (S' W^-1 S)^-1 S' W^-1`` (``W = I`` when omitted).

    Args:
        summing: ``n x m`` summing matrix of full column rank.
        weights: Symmetric positive-definite ``W`` or ``None``.

    Returns:
        The ``m x n`` mapping matrix.

    Raises:
        ValueError: If ``W`` or ``S' W^-1 S`` is not positive definite.

    References:
        [EQ-TS-RECON-002],
        [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

    """
    columns = transpose(summing)
    if weights is None:
        whitened = columns
    else:
        try:
            factor = cholesky(weights)
        except ValueError as error:
            raise ValueError(
                "the base-error covariance must be symmetric positive definite"
            ) from error
        whitened = tuple(cholesky_solve(factor, column) for column in columns)
    try:
        inverse = inverse_spd(matmul(whitened, summing))
    except ValueError as error:
        raise ValueError("the summing matrix must have full column rank") from error
    return matmul(inverse, whitened)


def _bottom_rows(summing: Matrix) -> tuple[int, ...]:
    """Return, for each bottom series, the row of ``S`` that equals its unit row.

    Args:
        summing: ``n x m`` summing matrix.

    Returns:
        ``m`` row indices; row ``rows[j]`` of ``S`` is the unit vector ``e_j``.

    Raises:
        ValueError: If some bottom series has no row of its own.

    References:
        [EQ-TS-RECON-001]. No equation is evaluated.

    """
    width = len(summing[0])
    found: list[int] = []
    for j in range(width):
        unit = tuple(1.0 if k == j else 0.0 for k in range(width))
        matches = [index for index, row in enumerate(summing) if row == unit]
        if not matches:
            raise ValueError(
                "bottom-up needs every bottom-level series as a row of the "
                "summing matrix"
            )
        found.append(matches[0])
    return tuple(found)


def reconcile_forecasts(
    base: Sequence[float],
    summing: Sequence[Sequence[float]] | HierarchyStructure,
    *,
    method: str = "ols",
    variances: Sequence[float] | None = None,
    covariance: Sequence[Sequence[float]] | None = None,
) -> ReconciliationResult:
    """Reconcile base forecasts so that they satisfy the aggregation constraints.

    Computes ``y_tilde = S G y_hat`` with
    ``G = (S' W^-1 S)^-1 S' W^-1`` for ``W = I`` (``"ols"``), ``W =
    diag(variances)`` (``"wls"``) or ``W = covariance`` (``"mint"``), or
    with the selection matrix of the bottom-level rows (``"bottom_up"``).
    Every choice satisfies ``G S = I``: coherent base forecasts are returned
    unchanged and unbiased base forecasts stay unbiased. ``"mint"`` minimizes
    ``tr(S G W G' S')`` over such ``G`` when ``W`` is the covariance of the
    base-forecast errors; with an estimated ``W`` that guarantee is lost.

    Args:
        base: Base forecasts of all ``n`` series for one horizon, in row
            order of ``summing``.
        summing: ``n x m`` summing matrix ``S`` of full column rank, or a
            :class:`HierarchyStructure`.
        method: ``"ols"``, ``"wls"``, ``"mint"`` or ``"bottom_up"``.
        variances: Positive base-error variances, required for ``"wls"``.
        covariance: Symmetric positive-definite base-error covariance,
            required for ``"mint"`` (for example
            ``shrink_covariance(residuals).covariance``).

    Returns:
        Reconciled forecasts, the mapping ``G``, and the distance of the
        base forecasts from coherence.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, the method is unknown, a required
            weight is missing or an unused one is given, a variance is not
            positive, ``W`` is not positive definite, ``S`` lacks full
            column rank, or ``"bottom_up"`` finds no row for a bottom series.

    Equation:
        [EQ-TS-RECON-001], [EQ-TS-RECON-002].

    References:
        [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].

    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    matrix = (
        summing.matrix
        if isinstance(summing, HierarchyStructure)
        else as_matrix(summing, name="summing")
    )
    forecasts = as_vector(base, name="base")
    size, width = len(matrix), len(matrix[0])
    if len(forecasts) != size:
        raise ValueError("base must have one forecast per row of the summing matrix")
    if width > size:
        raise ValueError(
            "the summing matrix must have at least as many rows as columns"
        )
    if (variances is not None) != (method == "wls"):
        raise ValueError("variances are required for, and only used by, method 'wls'")
    if (covariance is not None) != (method == "mint"):
        raise ValueError("covariance is required for, and only used by, method 'mint'")
    ordinary = _generalized_mapping(matrix, None)
    if method == "ols":
        mapping = ordinary
    elif method == "bottom_up":
        chosen = _bottom_rows(matrix)
        mapping = tuple(
            tuple(1.0 if column == chosen[j] else 0.0 for column in range(size))
            for j in range(width)
        )
    else:
        if variances is not None:
            diagonal = as_vector(variances, name="variances")
            if len(diagonal) != size:
                raise ValueError("variances must have one entry per series")
            if any(not value > 0.0 for value in diagonal):
                raise ValueError("variances must be positive")
            weights: Matrix = tuple(
                tuple(diagonal[i] if i == j else 0.0 for j in range(size))
                for i in range(size)
            )
        else:
            weights = as_matrix(covariance or (), name="covariance")
            if len(weights) != size or len(weights[0]) != size:
                raise ValueError("covariance must be square with one row per series")
        mapping = _generalized_mapping(matrix, weights)
    bottom = matvec(mapping, forecasts)
    reconciled = matvec(matrix, bottom)
    projected = matvec(matrix, matvec(ordinary, forecasts))
    incoherence = math.sqrt(
        math.fsum((a - b) ** 2 for a, b in zip(forecasts, projected, strict=True))
    )
    return ReconciliationResult(
        method, forecasts, reconciled, bottom, mapping, matrix, incoherence
    )
