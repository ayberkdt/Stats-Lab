"""Gaussian mixture models fitted by EM, with k-means++ seeding and BIC/AIC.

A ``K``-component Gaussian mixture has density
``p(x) = sum_k w_k N(x | m_k, S_k)``. :func:`fit_gaussian_mixture` maximizes a
*penalized* log-likelihood

``l_pen(theta) = sum_i log p(x_i) - (1/2) sum_k tr(Lambda S_k^-1)``

by the generic driver :func:`~probstat.inference.em.expectation_maximization`.
``Lambda = diag(lambda_j)`` with ``lambda_j = regularization * var_j`` (the
sample variance of coordinate ``j``) is a fixed covariance floor: the exact
M-step becomes ``S_k = (W_k + Lambda) / N_k`` ([EQ-GMM-002]), which keeps every
covariance positive definite, bounds the objective (the unpenalized mixture
likelihood is unbounded when a component collapses onto one point), and --
because ``Lambda`` does not depend on the iteration -- keeps EM monotone for
``l_pen``. AIC and BIC are computed from the unpenalized log-likelihood.

Initialization uses k-means++ ``D^2`` seeding (Arthur and Vassilvitskii, 2007)
driven by a :class:`~probstat.foundations.random_streams.RandomStream`,
followed by a few Lloyd iterations. Components of the returned mixture are in
a canonical order (lexicographic in the mean vector, then weight), so two fits
of the same model are comparable without label switching.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, cholesky

from .em import expectation_maximization
from .fitting import information_criteria

CovarianceType = Literal["full", "diag"]
_LOG_TWO_PI = math.log(2.0 * math.pi)
_LLOYD_ITERATIONS = 20
_DEAD_COMPONENT = 1e-10


# ---------------------------------------------------------------------------
# Data handling and Gaussian helpers
# ---------------------------------------------------------------------------


def _as_points(data: Iterable[object]) -> tuple[Vector, ...]:
    """Convert univariate or multivariate data to a tuple of finite vectors.

    References:
        Project validation policy. No equation is evaluated.

    """
    points: list[Vector] = []
    for item in data:
        if isinstance(item, int | float) and not isinstance(item, bool):
            vector: Vector = (float(item),)
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes):
            vector = tuple(float(value) for value in item)
        else:
            raise TypeError("data must contain real numbers or real sequences")
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("data points must be non-empty and finite")
        points.append(vector)
    if not points:
        raise ValueError("data must be non-empty")
    dimension = len(points[0])
    if any(len(point) != dimension for point in points):
        raise ValueError("all data points must have the same dimension")
    return tuple(points)


@dataclass(frozen=True, slots=True)
class _Component:
    """Cached Cholesky factor and log-determinant of one component.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """

    mean: Vector
    lower: Matrix
    log_det: float


def _component(mean: Vector, covariance: Matrix) -> _Component:
    """Factor one component covariance.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. Equation [EQ-LINALG-001].

    """
    lower = cholesky(covariance)
    log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(len(lower)))
    return _Component(mean, lower, log_det)


def _whitened_norm(component: _Component, x: Vector) -> float:
    """Return ``(x - m)' S^-1 (x - m)`` by forward substitution.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1]. No new equation is introduced.

    """
    lower = component.lower
    solved: list[float] = []
    total = 0.0
    for row, (value, center) in enumerate(zip(x, component.mean, strict=True)):
        residual = value - center
        for column in range(row):
            residual -= lower[row][column] * solved[column]
        z = residual / lower[row][row]
        solved.append(z)
        total += z * z
    return total


def _log_gaussian(component: _Component, x: Vector) -> float:
    """Return ``log N(x | m, S)`` from a cached factorization.

    References:
        [REF-BISHOP-2006, Section 2.3]. Equation [EQ-GMM-001].

    """
    dimension = len(x)
    return -0.5 * (
        dimension * _LOG_TWO_PI + component.log_det + _whitened_norm(component, x)
    )


def _inverse_diagonal(component: _Component) -> Vector:
    """Return ``diag(S^-1)`` from the Cholesky factor.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No new equation is introduced.

    """
    lower = component.lower
    size = len(lower)
    diagonal: list[float] = []
    for j in range(size):
        solved = [0.0] * size
        total = 0.0
        for row in range(j, size):
            residual = (1.0 if row == j else 0.0) - math.fsum(
                lower[row][column] * solved[column] for column in range(j, row)
            )
            solved[row] = residual / lower[row][row]
            total += solved[row] * solved[row]
        diagonal.append(total)
    return tuple(diagonal)


def _log_sum_exp(values: Sequence[float]) -> float:
    """Return ``log sum exp(values)`` without overflow.

    References:
        [REF-BISHOP-2006, Section 9.2]. No new equation is introduced.

    """
    peak = max(values)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(math.fsum(math.exp(value - peak) for value in values))


# ---------------------------------------------------------------------------
# Mixture model and fit containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GaussianMixture:
    """A finite Gaussian mixture ``sum_k w_k N(m_k, S_k)``.

    Args:
        weights: Mixing proportions summing to one.
        means: One mean vector per component.
        covariances: One symmetric positive-definite covariance per component.
        covariance_type: ``"full"`` or ``"diag"`` (informational, and used by
            :attr:`parameter_count`).

    Raises:
        ValueError: For inconsistent shapes, invalid weights, or a covariance
            that is not positive definite.

    References:
        [REF-MCLACHLAN-PEEL-2000], [REF-BISHOP-2006, Section 9.2].

    """

    weights: Vector
    means: tuple[Vector, ...]
    covariances: tuple[Matrix, ...]
    covariance_type: str = "full"

    def __post_init__(self) -> None:
        """Validate shapes and normalize containers.

        References:
            Project validation policy. No equation is evaluated.

        """
        weights = tuple(float(w) for w in self.weights)
        means = tuple(tuple(float(v) for v in mean) for mean in self.means)
        covariances = tuple(
            tuple(tuple(float(v) for v in row) for row in matrix)
            for matrix in self.covariances
        )
        if not weights or len(weights) != len(means):
            raise ValueError("weights and means must be non-empty and aligned")
        if len(covariances) != len(means):
            raise ValueError("one covariance per component is required")
        if any(not math.isfinite(w) or w < 0.0 for w in weights):
            raise ValueError("weights must be finite and non-negative")
        if abs(math.fsum(weights) - 1.0) > 1e-8:
            raise ValueError("weights must sum to one")
        dimension = len(means[0])
        if dimension == 0 or any(len(mean) != dimension for mean in means):
            raise ValueError("means must share a positive dimension")
        for matrix in covariances:
            if len(matrix) != dimension or any(len(r) != dimension for r in matrix):
                raise ValueError("covariances must be square with mean dimension")
            cholesky(matrix)
        if self.covariance_type not in ("full", "diag"):
            raise ValueError("covariance_type must be 'full' or 'diag'")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "covariances", covariances)

    @property
    def components(self) -> int:
        """Return the number of components ``K``.

        References:
            [REF-MCLACHLAN-PEEL-2000]. No equation is evaluated.

        """
        return len(self.weights)

    @property
    def dimension(self) -> int:
        """Return the data dimension ``d``.

        References:
            [REF-MCLACHLAN-PEEL-2000]. No equation is evaluated.

        """
        return len(self.means[0])

    @property
    def parameter_count(self) -> int:
        """Return the free-parameter count used by AIC and BIC.

        ``(K - 1) + K d + K d (d + 1) / 2`` for full covariances and
        ``(K - 1) + 2 K d`` for diagonal ones.

        Equation:
            [EQ-GMM-004].

        References:
            [REF-MCLACHLAN-PEEL-2000], [REF-SCHWARZ-1978].

        """
        k, d = self.components, self.dimension
        covariance = d * (d + 1) // 2 if self.covariance_type == "full" else d
        return (k - 1) + k * d + k * covariance

    def _factors(self) -> tuple[_Component, ...]:
        """Return cached factorizations of every component.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

        """
        return tuple(
            _component(mean, covariance)
            for mean, covariance in zip(self.means, self.covariances, strict=True)
        )

    def _point(self, x: Sequence[float] | float) -> Vector:
        """Validate one evaluation point.

        References:
            Project validation policy. No equation is evaluated.

        """
        point = _as_points([x])[0]
        if len(point) != self.dimension:
            raise ValueError("point dimension does not match the mixture")
        return point

    def log_pdf(self, x: Sequence[float] | float) -> float:
        """Return ``log p(x) = log sum_k w_k N(x | m_k, S_k)``.

        Args:
            x: Scalar (univariate mixture) or vector of length ``d``.

        Returns:
            Log mixture density.

        Equation:
            [EQ-GMM-001].

        References:
            [REF-MCLACHLAN-PEEL-2000], [REF-BISHOP-2006, Section 9.2].

        """
        point = self._point(x)
        return _log_sum_exp(
            [
                math.log(w) + _log_gaussian(c, point) if w > 0.0 else -math.inf
                for w, c in zip(self.weights, self._factors(), strict=True)
            ]
        )

    def responsibilities(self, data: Iterable[object]) -> tuple[Vector, ...]:
        """Return posterior component probabilities ``r_ik`` for each point.

        Args:
            data: Points (scalars for a univariate mixture).

        Returns:
            One probability vector of length ``K`` per point.

        Equation:
            [EQ-GMM-001].

        References:
            [REF-BISHOP-2006, Section 9.2].

        """
        points = _as_points(data)
        if len(points[0]) != self.dimension:
            raise ValueError("data dimension does not match the mixture")
        rows, _ = _e_step_rows(points, self.weights, self._factors())
        return rows

    def predict(self, data: Iterable[object]) -> tuple[int, ...]:
        """Return the most probable component index of each point.

        References:
            [REF-BISHOP-2006, Section 9.2]. Equation [EQ-GMM-001].

        """
        return tuple(
            max(range(len(row)), key=row.__getitem__)
            for row in self.responsibilities(data)
        )

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw ``size`` points by ancestral sampling.

        Args:
            size: Non-negative number of draws.
            seed: Integer seed or stream.

        Returns:
            Draws as vectors (length one for univariate mixtures).

        References:
            [REF-BISHOP-2006, Section 9.2]. No new equation is introduced.

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        factors = self._factors()
        cumulative: list[float] = []
        running = 0.0
        for weight in self.weights:
            running += weight
            cumulative.append(running)
        draws: list[Vector] = []
        for _ in range(count):
            u = generator.random() * running
            index = next(
                (k for k, edge in enumerate(cumulative) if u < edge),
                len(cumulative) - 1,
            )
            factor = factors[index]
            z = [generator.gauss(0.0, 1.0) for _ in range(self.dimension)]
            draws.append(
                tuple(
                    factor.mean[row]
                    + math.fsum(factor.lower[row][c] * z[c] for c in range(row + 1))
                    for row in range(self.dimension)
                )
            )
        return tuple(draws)


@dataclass(frozen=True, slots=True)
class GaussianMixtureFit:
    """Result of fitting a Gaussian mixture by penalized EM.

    Args:
        mixture: Fitted mixture in canonical component order.
        log_likelihood: Unpenalized log-likelihood at the estimate.
        penalized_log_likelihood: EM objective ``l_pen`` at the estimate.
        sample_size: Number of data points ``n``.
        iterations: EM iterations of the selected restart.
        converged: Whether the selected restart met the tolerance.
        monotone: Whether ``l_pen`` never decreased beyond round-off.
        trace: ``l_pen`` after every E-step of the selected restart.
        responsibilities: ``r_ik`` at the estimate (canonical order).
        restart_objectives: Final ``l_pen`` of every restart.

    References:
        [REF-MCLACHLAN-PEEL-2000], [REF-DEMPSTER-LAIRD-RUBIN-1977].

    """

    mixture: GaussianMixture
    log_likelihood: float
    penalized_log_likelihood: float
    sample_size: int
    iterations: int
    converged: bool
    monotone: bool
    trace: tuple[float, ...]
    responsibilities: tuple[Vector, ...]
    restart_objectives: tuple[float, ...]

    @property
    def parameter_count(self) -> int:
        """Return the free-parameter count of the fitted mixture.

        References:
            [REF-MCLACHLAN-PEEL-2000]. Equation [EQ-GMM-004].

        """
        return self.mixture.parameter_count

    @property
    def aic(self) -> float:
        """Return ``AIC = 2 p - 2 log L`` (unpenalized ``log L``).

        References:
            [REF-AKAIKE-1974]. Equation [EQ-INF-IC-001].

        """
        return information_criteria(
            self.log_likelihood, self.parameter_count, self.sample_size
        )[0]

    @property
    def bic(self) -> float:
        """Return ``BIC = p log n - 2 log L`` (unpenalized ``log L``).

        References:
            [REF-SCHWARZ-1978]. Equation [EQ-INF-IC-003].

        """
        return information_criteria(
            self.log_likelihood, self.parameter_count, self.sample_size
        )[2]

    def labels(self) -> tuple[int, ...]:
        """Return the maximum-responsibility component of each data point.

        References:
            [REF-BISHOP-2006, Section 9.2]. Equation [EQ-GMM-001].

        """
        return tuple(
            max(range(len(row)), key=row.__getitem__) for row in self.responsibilities
        )


@dataclass(frozen=True, slots=True)
class MixtureSelection:
    """Gaussian mixtures fitted for several ``K`` and ranked by a criterion.

    Args:
        fits: One fit per candidate ``K`` (in candidate order).
        criterion: ``"bic"`` or ``"aic"``.
        scores: Criterion value of each fit.
        best_index: Index of the minimum-criterion fit.

    References:
        [REF-MCLACHLAN-PEEL-2000], [REF-SCHWARZ-1978], [REF-AKAIKE-1974].

    """

    fits: tuple[GaussianMixtureFit, ...]
    criterion: str
    scores: tuple[float, ...]
    best_index: int

    @property
    def best(self) -> GaussianMixtureFit:
        """Return the selected fit.

        References:
            [REF-SCHWARZ-1978]. No equation is evaluated.

        """
        return self.fits[self.best_index]

    @property
    def best_components(self) -> int:
        """Return the selected number of components.

        References:
            [REF-SCHWARZ-1978]. No equation is evaluated.

        """
        return self.best.mixture.components


# ---------------------------------------------------------------------------
# k-means++ seeding
# ---------------------------------------------------------------------------


def _squared_distance(a: Vector, b: Vector) -> float:
    """Return the squared Euclidean distance.

    References:
        [REF-ARTHUR-VASSILVITSKII-2007]. No new equation is introduced.

    """
    return math.fsum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def _kmeans_plus_plus(
    points: Sequence[Vector], components: int, generator: random.Random
) -> list[Vector]:
    """Choose initial centers by ``D^2`` sampling.

    References:
        [REF-ARTHUR-VASSILVITSKII-2007]. Equation [EQ-GMM-003].

    """
    centers = [points[generator.randrange(len(points))]]
    nearest = [_squared_distance(point, centers[0]) for point in points]
    while len(centers) < components:
        total = math.fsum(nearest)
        if total <= 0.0:
            index = generator.randrange(len(points))
        else:
            u = generator.random() * total
            running = 0.0
            index = len(points) - 1
            for candidate, distance in enumerate(nearest):
                running += distance
                if u < running:
                    index = candidate
                    break
        center = points[index]
        centers.append(center)
        nearest = [
            min(old, _squared_distance(point, center))
            for old, point in zip(nearest, points, strict=True)
        ]
    return centers


def kmeans_plus_plus(
    data: Iterable[object],
    components: int,
    *,
    seed: int | RandomStream | None = None,
) -> tuple[Vector, ...]:
    """Return ``K`` initial centers chosen by k-means++ ``D^2`` seeding.

    The first center is a uniformly chosen data point; each further center is
    a data point chosen with probability proportional to its squared distance
    to the nearest center already chosen.

    Args:
        data: Points (scalars or equal-length sequences).
        components: Number of centers ``1 <= K <= n``.
        seed: Integer seed or stream.

    Returns:
        Chosen centers in selection order.

    Raises:
        ValueError: If ``K`` is not in ``[1, n]``.

    Equation:
        [EQ-GMM-003].

    References:
        [REF-ARTHUR-VASSILVITSKII-2007].

    """
    points = _as_points(data)
    k = require_non_negative_integer(components, name="components")
    if not 1 <= k <= len(points):
        raise ValueError("components must lie in [1, number of points]")
    return tuple(_kmeans_plus_plus(points, k, resolve_stream(seed).generator()))


def _lloyd(points: Sequence[Vector], centers: list[Vector]) -> list[int]:
    """Refine centers by a few Lloyd iterations and return hard labels.

    References:
        [REF-ARTHUR-VASSILVITSKII-2007]. No new equation is introduced.

    """
    dimension = len(points[0])
    labels = [0] * len(points)
    for _ in range(_LLOYD_ITERATIONS):
        new_labels = [
            min(range(len(centers)), key=lambda k: _squared_distance(p, centers[k]))
            for p in points
        ]
        for k in range(len(centers)):
            members = [p for p, label in zip(points, new_labels, strict=True)
                       if label == k]
            if members:
                centers[k] = tuple(
                    math.fsum(p[j] for p in members) / len(members)
                    for j in range(dimension)
                )
        if new_labels == labels:
            break
        labels = new_labels
    return labels


# ---------------------------------------------------------------------------
# EM steps
# ---------------------------------------------------------------------------


def _e_step_rows(
    points: Sequence[Vector],
    weights: Sequence[float],
    factors: Sequence[_Component],
) -> tuple[tuple[Vector, ...], float]:
    """Return responsibilities and the unpenalized log-likelihood.

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-BISHOP-2006, Section 9.2].
        Equation [EQ-GMM-001].

    """
    log_weights = [math.log(w) if w > 0.0 else -math.inf for w in weights]
    rows: list[Vector] = []
    total: list[float] = []
    for point in points:
        terms = [
            lw + _log_gaussian(factor, point) if lw > -math.inf else -math.inf
            for lw, factor in zip(log_weights, factors, strict=True)
        ]
        normalizer = _log_sum_exp(terms)
        total.append(normalizer)
        rows.append(tuple(math.exp(term - normalizer) for term in terms))
    return tuple(rows), math.fsum(total)


def _penalty(factors: Sequence[_Component], floor: Vector) -> float:
    """Return ``(1/2) sum_k tr(Lambda S_k^-1)``.

    References:
        [REF-MCLACHLAN-PEEL-2000]. Equation [EQ-GMM-002].

    """
    return 0.5 * math.fsum(
        math.fsum(
            lam * inv for lam, inv in zip(floor, _inverse_diagonal(f), strict=True)
        )
        for f in factors
    )


def _m_step(
    points: Sequence[Vector],
    rows: Sequence[Vector],
    floor: Vector,
    covariance_type: str,
    previous: GaussianMixture,
) -> GaussianMixture:
    """Return the exact penalized M-step update.

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-MCLACHLAN-PEEL-2000].
        Equation [EQ-GMM-002].

    """
    n = len(points)
    dimension = len(points[0])
    components = len(rows[0])
    weights: list[float] = []
    means: list[Vector] = []
    covariances: list[Matrix] = []
    for k in range(components):
        column = [row[k] for row in rows]
        mass = math.fsum(column)
        if mass <= _DEAD_COMPONENT * n:
            weights.append(mass / n)
            means.append(previous.means[k])
            covariances.append(previous.covariances[k])
            continue
        mean = tuple(
            math.fsum(r * p[j] for r, p in zip(column, points, strict=True)) / mass
            for j in range(dimension)
        )
        scatter = [[0.0] * dimension for _ in range(dimension)]
        for i in range(dimension):
            for j in range(i + 1):
                if covariance_type == "diag" and i != j:
                    continue
                value = math.fsum(
                    r * (p[i] - mean[i]) * (p[j] - mean[j])
                    for r, p in zip(column, points, strict=True)
                )
                scatter[i][j] = value
                scatter[j][i] = value
        covariance = tuple(
            tuple(
                (scatter[i][j] + (floor[i] if i == j else 0.0)) / mass
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        weights.append(mass / n)
        means.append(mean)
        covariances.append(covariance)
    total = math.fsum(weights)
    return GaussianMixture(
        tuple(w / total for w in weights),
        tuple(means),
        tuple(covariances),
        covariance_type,
    )


def _canonical_order(mixture: GaussianMixture) -> tuple[int, ...]:
    """Return the permutation sorting components by mean, then weight.

    References:
        [REF-MCLACHLAN-PEEL-2000]. No equation is evaluated.

    """
    return tuple(
        sorted(
            range(mixture.components),
            key=lambda k: (mixture.means[k], mixture.weights[k]),
        )
    )


def _initial_mixture(
    points: Sequence[Vector],
    components: int,
    floor: Vector,
    covariance_type: str,
    generator: random.Random,
) -> GaussianMixture:
    """Seed by k-means++, refine by Lloyd, and form moment estimates.

    References:
        [REF-ARTHUR-VASSILVITSKII-2007], [REF-MCLACHLAN-PEEL-2000].
        Equation [EQ-GMM-003].

    """
    n = len(points)
    dimension = len(points[0])
    centers = _kmeans_plus_plus(points, components, generator)
    labels = _lloyd(points, centers)
    overall = tuple(math.fsum(p[j] for p in points) / n for j in range(dimension))

    def covariance_of(members: Sequence[Vector], center: Vector) -> Matrix:
        """Return the floored (diagonal or full) scatter of a member set.

        References:
            [REF-MCLACHLAN-PEEL-2000]. Equation [EQ-GMM-002].

        """
        count = len(members)
        return tuple(
            tuple(
                (
                    0.0
                    if covariance_type == "diag" and i != j
                    else math.fsum(
                        (p[i] - center[i]) * (p[j] - center[j]) for p in members
                    )
                )
                / count
                + (floor[i] / count if i == j else 0.0)
                for j in range(dimension)
            )
            for i in range(dimension)
        )

    global_covariance = covariance_of(points, overall)
    weights: list[float] = []
    means: list[Vector] = []
    covariances: list[Matrix] = []
    for k in range(components):
        members = [p for p, label in zip(points, labels, strict=True) if label == k]
        weights.append(max(len(members), 1) / n)
        means.append(centers[k])
        covariances.append(
            covariance_of(members, centers[k])
            if len(members) > dimension
            else global_covariance
        )
    total = math.fsum(weights)
    return GaussianMixture(
        tuple(w / total for w in weights),
        tuple(means),
        tuple(covariances),
        covariance_type,
    )


def fit_gaussian_mixture(
    data: Iterable[object],
    components: int,
    *,
    covariance_type: CovarianceType = "full",
    regularization: float = 1e-6,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
    restarts: int = 1,
    seed: int | RandomStream | None = None,
) -> GaussianMixtureFit:
    """Fit a ``K``-component Gaussian mixture by penalized EM.

    E-step: responsibilities ``r_ik = w_k N(x_i|m_k,S_k) / p(x_i)``.
    M-step: ``N_k = sum_i r_ik``, ``w_k = N_k / n``,
    ``m_k = sum_i r_ik x_i / N_k``, ``S_k = (W_k + Lambda) / N_k`` with
    ``W_k = sum_i r_ik (x_i - m_k)(x_i - m_k)'`` (only its diagonal for
    ``"diag"``). Each restart uses an independent child stream for k-means++
    seeding; the restart with the largest penalized objective is returned.

    Args:
        data: Scalars (univariate) or equal-length sequences.
        components: Number of components ``1 <= K <= n``.
        covariance_type: ``"full"`` or ``"diag"`` (identical when ``d = 1``).
        regularization: Non-negative relative floor; ``lambda_j`` equals this
            value times the sample variance of coordinate ``j`` (or times one
            when that variance is zero).
        tolerance: Relative objective-change tolerance of the EM driver.
        max_iterations: Maximum EM iterations per restart.
        restarts: Number of independently seeded EM runs (positive).
        seed: Integer seed or stream.

    Returns:
        Fit in canonical component order with likelihoods, trace, and
        responsibilities.

    Raises:
        ValueError: For invalid settings, or ``regularization = 0`` with a
            degenerate component.

    Equation:
        [EQ-EM-001], [EQ-GMM-001], [EQ-GMM-002], [EQ-GMM-003].

    References:
        [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-MCLACHLAN-PEEL-2000],
        [REF-ARTHUR-VASSILVITSKII-2007], [REF-BISHOP-2006, Section 9.2].

    """
    points = _as_points(data)
    n = len(points)
    dimension = len(points[0])
    k = require_non_negative_integer(components, name="components")
    if not 1 <= k <= n:
        raise ValueError("components must lie in [1, number of points]")
    if covariance_type not in ("full", "diag"):
        raise ValueError("covariance_type must be 'full' or 'diag'")
    ridge = require_finite_real(regularization, name="regularization")
    if ridge < 0.0:
        raise ValueError("regularization must be non-negative")
    runs = require_non_negative_integer(restarts, name="restarts")
    if runs == 0:
        raise ValueError("restarts must be positive")
    floor_values: list[float] = []
    for j in range(dimension):
        center = math.fsum(p[j] for p in points) / n
        spread = math.fsum((p[j] - center) ** 2 for p in points) / max(n - 1, 1)
        floor_values.append(ridge * (spread if spread > 0.0 else 1.0))
    floor = tuple(floor_values)
    stream = resolve_stream(seed)

    def e_step(
        mixture: GaussianMixture,
    ) -> tuple[tuple[tuple[Vector, ...], GaussianMixture], float]:
        """Return responsibilities and the penalized log-likelihood.

        References:
            [REF-DEMPSTER-LAIRD-RUBIN-1977]. Equations [EQ-GMM-001],
            [EQ-GMM-002].

        """
        factors = mixture._factors()
        rows, log_likelihood = _e_step_rows(points, mixture.weights, factors)
        return (rows, mixture), log_likelihood - _penalty(factors, floor)

    def m_step(
        statistics: tuple[tuple[Vector, ...], GaussianMixture],
    ) -> GaussianMixture:
        """Apply the penalized M-step to the current responsibilities.

        References:
            [REF-MCLACHLAN-PEEL-2000]. Equation [EQ-GMM-002].

        """
        rows, previous = statistics
        return _m_step(points, rows, floor, covariance_type, previous)

    best = None
    objectives: list[float] = []
    for run in range(runs):
        generator = stream.child("gmm-restart", run).generator()
        initial = _initial_mixture(points, k, floor, covariance_type, generator)
        result = expectation_maximization(
            initial,
            e_step,
            m_step,
            tolerance=tolerance,
            max_iterations=max_iterations,
            on_decrease="ignore",
        )
        objectives.append(result.log_likelihood)
        if best is None or result.log_likelihood > best.log_likelihood:
            best = result
    assert best is not None
    fitted = best.parameters
    order = _canonical_order(fitted)
    mixture = GaussianMixture(
        tuple(fitted.weights[i] for i in order),
        tuple(fitted.means[i] for i in order),
        tuple(fitted.covariances[i] for i in order),
        covariance_type,
    )
    factors = mixture._factors()
    rows, log_likelihood = _e_step_rows(points, mixture.weights, factors)
    return GaussianMixtureFit(
        mixture,
        log_likelihood,
        log_likelihood - _penalty(factors, floor),
        n,
        best.iterations,
        best.converged,
        best.monotone,
        best.trace,
        rows,
        tuple(objectives),
    )


def select_gaussian_mixture(
    data: Iterable[object],
    candidates: Iterable[int] = (1, 2, 3, 4, 5),
    *,
    criterion: Literal["bic", "aic"] = "bic",
    covariance_type: CovarianceType = "full",
    regularization: float = 1e-6,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
    restarts: int = 1,
    seed: int | RandomStream | None = None,
) -> MixtureSelection:
    """Fit mixtures for several ``K`` and select the minimum BIC (or AIC).

    Args:
        data: Scalars or equal-length sequences.
        candidates: Distinct candidate component counts.
        criterion: ``"bic"`` (default) or ``"aic"``.
        covariance_type: ``"full"`` or ``"diag"``.
        regularization: Relative covariance floor (see
            :func:`fit_gaussian_mixture`).
        tolerance: EM tolerance.
        max_iterations: EM iteration limit.
        restarts: Seeded restarts per candidate.
        seed: Integer seed or stream; candidate ``K`` uses child ``("K", K)``.

    Returns:
        All fits, their criterion values, and the selected index.

    Raises:
        ValueError: For an empty or repeated candidate list or bad criterion.

    Equation:
        [EQ-INF-IC-001], [EQ-INF-IC-003], [EQ-GMM-004].

    References:
        [REF-SCHWARZ-1978], [REF-AKAIKE-1974], [REF-MCLACHLAN-PEEL-2000].

    """
    points = _as_points(data)
    counts = tuple(candidates)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("candidates must be non-empty and distinct")
    if criterion not in ("bic", "aic"):
        raise ValueError("criterion must be 'bic' or 'aic'")
    stream = resolve_stream(seed)
    fits = tuple(
        fit_gaussian_mixture(
            points,
            count,
            covariance_type=covariance_type,
            regularization=regularization,
            tolerance=tolerance,
            max_iterations=max_iterations,
            restarts=restarts,
            seed=stream.child("K", count),
        )
        for count in counts
    )
    scores = tuple(fit.bic if criterion == "bic" else fit.aic for fit in fits)
    best_index = min(range(len(scores)), key=scores.__getitem__)
    return MixtureSelection(fits, criterion, scores, best_index)
