"""Gaussian random fields: covariance kernels and Karhunen-Loeve expansions.

A second-order random field ``H(x)`` with mean ``m`` and covariance kernel
``C(x, y)`` has the Karhunen-Loeve (KL) expansion

    H(x) = m + sum_k sqrt(lambda_k) xi_k phi_k(x),

where ``(lambda_k, phi_k)`` are the eigenpairs of the covariance integral
operator and, for a Gaussian field, ``xi_k`` are independent standard
normals. Truncating after ``M`` terms keeps the modes that carry the largest
share of the integrated variance ``sum_k lambda_k``; the coefficients
``xi_1..xi_M`` then become ordinary uncertain inputs of a forward model.

The integral eigenproblem is discretized by the Nystrom method: with
quadrature nodes ``x_i`` and positive weights ``w_i`` it becomes the
symmetric matrix problem ``W^(1/2) C W^(1/2) psi = lambda psi`` with
``phi = W^(-1/2) psi``. Unit weights reduce it to the plain eigenvalue
decomposition of the covariance matrix on a grid. Eigenpairs are computed by
the dependency-free Jacobi method, whose pure-Python cost grows as ``n^3``;
it is practical for grids of up to roughly 150 nodes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, symmetric_eigen
from probstat.optimization import brent_root

from .continuous import _positive, _positive_integer

KERNEL_FAMILIES = ("squared_exponential", "exponential", "matern32", "matern52")
QUADRATURE_RULES = ("gauss_legendre", "midpoint", "trapezoid")
_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)
_NEGATIVE_EIGENVALUE_TOLERANCE = 1e-8

Point = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CovarianceKernel:
    """Stationary isotropic covariance kernel ``C(x, y) = sigma^2 rho(r)``.

    ``r = ||x - y||`` is the Euclidean distance and ``ell`` the length scale:

    * ``squared_exponential``: ``rho = exp(-r^2 / (2 ell^2))``;
    * ``exponential``: ``rho = exp(-r / ell)`` (Matern 1/2);
    * ``matern32``: ``rho = (1 + sqrt(3) r / ell) exp(-sqrt(3) r / ell)``;
    * ``matern52``: ``rho = (1 + sqrt(5) r / ell + 5 r^2 / (3 ell^2))
      exp(-sqrt(5) r / ell)``.

    Args:
        family: One of :data:`KERNEL_FAMILIES`.
        variance: Positive point variance ``sigma^2``.
        length_scale: Positive length scale ``ell``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """

    family: str
    variance: float = 1.0
    length_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the family and parameters.

        Raises:
            ValueError: If the family is unknown or a parameter is not
                positive.

        References:
            Project validation policy. No equation is evaluated.

        """
        if self.family not in KERNEL_FAMILIES:
            raise ValueError(f"family must be one of {KERNEL_FAMILIES}")
        object.__setattr__(self, "variance", _positive(self.variance, name="variance"))
        object.__setattr__(
            self, "length_scale", _positive(self.length_scale, name="length_scale")
        )

    def correlation(self, distance: float) -> float:
        """Return the correlation ``rho(r)`` at a non-negative distance.

        Args:
            distance: Euclidean distance ``r >= 0``.

        Returns:
            Correlation in ``(0, 1]``.

        Raises:
            ValueError: If the distance is negative.

        Equation:
            [EQ-PROC-KER-001], [EQ-PROC-KER-002], [EQ-PROC-KER-003],
            [EQ-PROC-KER-004].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        r = require_finite_real(distance, name="distance")
        if r < 0.0:
            raise ValueError("distance must be non-negative")
        s = r / self.length_scale
        if self.family == "squared_exponential":
            return math.exp(-0.5 * s * s)
        if self.family == "exponential":
            return math.exp(-s)
        if self.family == "matern32":
            return (1.0 + _SQRT3 * s) * math.exp(-_SQRT3 * s)
        return (1.0 + _SQRT5 * s + 5.0 * s * s / 3.0) * math.exp(-_SQRT5 * s)

    def __call__(self, left: Sequence[float], right: Sequence[float]) -> float:
        """Return ``C(x, y)`` for two points of equal dimension.

        Args:
            left: First point.
            right: Second point.

        Returns:
            Covariance.

        Raises:
            ValueError: If the dimensions differ.

        Equation:
            [EQ-PROC-KER-001], [EQ-PROC-KER-002], [EQ-PROC-KER-003],
            [EQ-PROC-KER-004].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        if len(left) != len(right):
            raise ValueError("points must have equal dimension")
        distance = math.sqrt(
            math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True))
        )
        return self.variance * self.correlation(distance)


def _as_point(item: object, *, name: str) -> Point:
    """Normalize one scalar or coordinate sequence into a point.

    Args:
        item: Scalar (1-D location) or iterable of coordinates.
        name: Field name for error messages.

    Returns:
        Finite coordinate tuple.

    Raises:
        TypeError: If the item is neither real nor iterable.
        ValueError: If a coordinate is non-finite or the point is empty.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(item, Iterable):
        coords = tuple(require_finite_real(c, name=name) for c in item)
        if not coords:
            raise ValueError(f"{name} must have at least one coordinate")
        return coords
    return (require_finite_real(item, name=name),)


def as_points(points: Sequence[float] | Sequence[Sequence[float]]) -> tuple[Point, ...]:
    """Normalize scalar 1-D locations or coordinate tuples into points.

    Args:
        points: Non-empty sequence of scalars (1-D) or of equal-length
            coordinate sequences.

    Returns:
        Tuple of finite coordinate tuples.

    Raises:
        ValueError: If the set is empty or dimensions differ.

    References:
        Project validation policy. No equation is evaluated.

    """
    materialized: list[object] = list(points)
    if not materialized:
        raise ValueError("at least one point is required")
    result = [
        _as_point(item, name=f"points[{i}]") for i, item in enumerate(materialized)
    ]
    if len({len(p) for p in result}) != 1:
        raise ValueError("all points must have the same dimension")
    return tuple(result)


def covariance_matrix(
    kernel: CovarianceKernel, points: Sequence[float] | Sequence[Sequence[float]]
) -> Matrix:
    """Return the covariance matrix ``C_ij = C(x_i, x_j)`` on a point set.

    Args:
        kernel: Covariance kernel.
        points: 1-D scalars or coordinate tuples.

    Returns:
        Symmetric covariance matrix.

    Equation:
        [EQ-PROC-KER-001], [EQ-PROC-KER-002], [EQ-PROC-KER-003],
        [EQ-PROC-KER-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """
    nodes = as_points(points)
    size = len(nodes)
    rows = [[0.0] * size for _ in range(size)]
    for i in range(size):
        rows[i][i] = kernel.variance
        for j in range(i):
            rows[i][j] = rows[j][i] = kernel(nodes[i], nodes[j])
    return tuple(tuple(row) for row in rows)


def gauss_legendre(
    order: int, lower: float = -1.0, upper: float = 1.0
) -> tuple[Vector, Vector]:
    """Return Gauss-Legendre nodes and weights on ``[lower, upper]``.

    Args:
        order: Number of nodes, between 1 and 200.
        lower: Finite lower bound.
        upper: Finite upper bound greater than ``lower``.

    Returns:
        Ascending nodes and positive weights summing to ``upper - lower``.

    Raises:
        ValueError: If the order or the bounds are invalid.

    Equation:
        [EQ-PROC-QUAD-001], [EQ-LINALG-005].

    References:
        [REF-GOLUB-WELSCH-1969], [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

    """
    count = _positive_integer(order, name="order")
    if count > 200:
        raise ValueError("order must not exceed 200")
    a = require_finite_real(lower, name="lower")
    b = require_finite_real(upper, name="upper")
    if not b > a:
        raise ValueError("upper must exceed lower")
    jacobi = tuple(
        tuple(
            (max(i, j) / math.sqrt(4.0 * max(i, j) ** 2 - 1.0))
            if abs(i - j) == 1
            else 0.0
            for j in range(count)
        )
        for i in range(count)
    )
    values, vectors = symmetric_eigen(jacobi)
    half = 0.5 * (b - a)
    centre = 0.5 * (a + b)
    nodes = tuple(centre + half * t for t in values)
    weights = tuple(2.0 * half * vectors[0][k] ** 2 for k in range(count))
    return nodes, weights


def interval_rule(
    lower: float, upper: float, nodes: int, *, rule: str = "gauss_legendre"
) -> tuple[Vector, Vector]:
    """Return quadrature nodes and weights for a 1-D interval.

    Args:
        lower: Finite lower bound.
        upper: Finite upper bound greater than ``lower``.
        nodes: Positive number of nodes (at least two for ``trapezoid``).
        rule: One of :data:`QUADRATURE_RULES`.

    Returns:
        Nodes and positive weights summing to the interval length.

    Raises:
        ValueError: If the rule is unknown or the inputs are invalid.

    Equation:
        [EQ-PROC-QUAD-001].

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

    """
    a = require_finite_real(lower, name="lower")
    b = require_finite_real(upper, name="upper")
    count = _positive_integer(nodes, name="nodes")
    if not b > a:
        raise ValueError("upper must exceed lower")
    if rule == "gauss_legendre":
        return gauss_legendre(count, a, b)
    if rule == "midpoint":
        h = (b - a) / count
        return tuple(a + (k + 0.5) * h for k in range(count)), (h,) * count
    if rule == "trapezoid":
        if count < 2:
            raise ValueError("trapezoid rule needs at least two nodes")
        h = (b - a) / (count - 1)
        points = tuple(a + k * h for k in range(count - 1)) + (b,)
        weights = (0.5 * h,) + (h,) * (count - 2) + (0.5 * h,)
        return points, weights
    raise ValueError(f"rule must be one of {QUADRATURE_RULES}")


def rectangle_rule(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    counts: tuple[int, int],
    *,
    rule: str = "midpoint",
) -> tuple[tuple[Point, ...], Vector]:
    """Return a tensor-product quadrature on a rectangle.

    Args:
        x_bounds: ``(lower, upper)`` in the first coordinate.
        y_bounds: ``(lower, upper)`` in the second coordinate.
        counts: Node counts per coordinate.
        rule: One-dimensional rule applied in each coordinate.

    Returns:
        2-D points (first coordinate varying slowest) and product weights.

    Equation:
        [EQ-PROC-QUAD-001].

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapter 5].

    """
    xs, wx = interval_rule(x_bounds[0], x_bounds[1], counts[0], rule=rule)
    ys, wy = interval_rule(y_bounds[0], y_bounds[1], counts[1], rule=rule)
    points = tuple((x, y) for x in xs for y in ys)
    weights = tuple(a * b for a in wx for b in wy)
    return points, weights


@dataclass(frozen=True)
class KarhunenLoeveExpansion:
    """Truncated discrete Karhunen-Loeve expansion of a Gaussian field.

    Args:
        kernel: Covariance kernel of the field.
        points: Quadrature (or grid) nodes.
        weights: Positive quadrature weights (all one for a plain grid).
        mean: Constant field mean ``m``.
        eigenvalues: Retained eigenvalues in decreasing order.
        eigenvectors: Retained eigenfunction values at the nodes, one tuple
            per mode, normalized so that ``sum_i w_i phi_k(x_i)^2 = 1``.
        spectrum: All non-negative discrete eigenvalues (diagnostics).

    References:
        [REF-GHANEM-SPANOS-1991, Chapter 2], [REF-LEMAITRE-KNIO-2010, Chapter 2].

    """

    kernel: CovarianceKernel
    points: tuple[Point, ...]
    weights: Vector
    mean: float
    eigenvalues: Vector
    eigenvectors: tuple[Vector, ...]
    spectrum: Vector

    @property
    def terms(self) -> int:
        """Return the number of retained modes ``M``.

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        return len(self.eigenvalues)

    @property
    def total_variance(self) -> float:
        """Return the discrete integrated variance ``sum_i w_i C(x_i, x_i)``.

        Equation:
            [EQ-PROC-KL-003].

        References:
            [REF-SUDRET-DERKIUREGHIAN-2000, Chapter 2].

        """
        return self.kernel.variance * math.fsum(self.weights)

    @property
    def explained_variance_fraction(self) -> float:
        """Return ``sum_(k <= M) lambda_k / sum_k lambda_k``.

        Equation:
            [EQ-PROC-KL-003].

        References:
            [REF-SUDRET-DERKIUREGHIAN-2000, Chapter 2].

        """
        return math.fsum(self.eigenvalues) / math.fsum(self.spectrum)

    def eigenfunction(self, mode: int) -> Vector:
        """Return the values of eigenfunction ``mode`` at the nodes.

        Args:
            mode: Zero-based mode index below :attr:`terms`.

        Returns:
            Eigenfunction values.

        Raises:
            IndexError: If the mode is not retained.

        Equation:
            [EQ-PROC-KL-002].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 4.3].

        """
        if not 0 <= mode < self.terms:
            raise IndexError("mode is not retained")
        return self.eigenvectors[mode]

    def eigenfunction_at(self, mode: int, location: float | Sequence[float]) -> float:
        """Evaluate eigenfunction ``mode`` anywhere by Nystrom interpolation.

        ``phi_k(x) = (1 / lambda_k) sum_i w_i C(x, x_i) phi_k(x_i)``, which
        reproduces the nodal values exactly.

        Args:
            mode: Zero-based retained mode index.
            location: Scalar (1-D) or coordinate tuple.

        Returns:
            Eigenfunction value.

        Equation:
            [EQ-PROC-KL-005].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 4.3].

        """
        values = self.eigenfunction(mode)
        point = _as_point(location, name="location")
        return (
            math.fsum(
                w * self.kernel(point, node) * v
                for w, node, v in zip(self.weights, self.points, values, strict=True)
            )
            / self.eigenvalues[mode]
        )

    def _coefficients(self, coefficients: Sequence[float]) -> Vector:
        """Validate a coefficient vector of length :attr:`terms`.

        Args:
            coefficients: Candidate ``xi_1..xi_M``.

        Returns:
            Validated coefficients.

        Raises:
            ValueError: If the length is wrong.

        References:
            Project validation policy. No equation is evaluated.

        """
        values = tuple(
            require_finite_real(c, name=f"coefficients[{i}]")
            for i, c in enumerate(coefficients)
        )
        if len(values) != self.terms:
            raise ValueError(f"expected {self.terms} coefficients")
        return values

    def realize(self, coefficients: Sequence[float]) -> Vector:
        """Return field values at the nodes for given KL coefficients.

        Args:
            coefficients: ``xi_1..xi_M`` (standard normal for a Gaussian
                field).

        Returns:
            ``m + sum_k sqrt(lambda_k) xi_k phi_k(x_i)`` at each node.

        Raises:
            ValueError: If the number of coefficients is wrong.

        Equation:
            [EQ-PROC-KL-001].

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        xi = self._coefficients(coefficients)
        scaled = tuple(
            math.sqrt(lam) * c for lam, c in zip(self.eigenvalues, xi, strict=True)
        )
        return tuple(
            self.mean
            + math.fsum(s * self.eigenvectors[k][i] for k, s in enumerate(scaled))
            for i in range(len(self.points))
        )

    def realize_at(
        self, coefficients: Sequence[float], location: float | Sequence[float]
    ) -> float:
        """Return the field value at an arbitrary location.

        Args:
            coefficients: ``xi_1..xi_M``.
            location: Scalar (1-D) or coordinate tuple.

        Returns:
            Field value from Nystrom-interpolated eigenfunctions.

        Equation:
            [EQ-PROC-KL-001], [EQ-PROC-KL-005].

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        xi = self._coefficients(coefficients)
        return self.mean + math.fsum(
            math.sqrt(self.eigenvalues[k]) * xi[k] * self.eigenfunction_at(k, location)
            for k in range(self.terms)
        )

    def pointwise_variance(self) -> Vector:
        """Return ``sum_(k <= M) lambda_k phi_k(x_i)^2`` at each node.

        The truncated variance never exceeds the field variance; with all
        modes retained it reproduces ``C(x_i, x_i)``.

        Returns:
            Truncated variance per node.

        Equation:
            [EQ-PROC-KL-003].

        References:
            [REF-SUDRET-DERKIUREGHIAN-2000, Chapter 2].

        """
        return tuple(
            math.fsum(
                lam * vector[i] ** 2
                for lam, vector in zip(self.eigenvalues, self.eigenvectors, strict=True)
            )
            for i in range(len(self.points))
        )

    def covariance(self, first: int, second: int) -> float:
        """Return the truncated covariance between two nodes.

        Args:
            first: Node index.
            second: Node index.

        Returns:
            ``sum_(k <= M) lambda_k phi_k(x_i) phi_k(x_j)``.

        Equation:
            [EQ-PROC-KL-001].

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        return math.fsum(
            lam * vector[first] * vector[second]
            for lam, vector in zip(self.eigenvalues, self.eigenvectors, strict=True)
        )

    def sample(
        self, size: int, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw independent Gaussian field realizations at the nodes.

        Args:
            size: Positive number of realizations.
            seed: Integer seed or stream.

        Returns:
            Realizations, each a tuple of nodal values.

        Equation:
            [EQ-PROC-KL-001].

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        count = _positive_integer(size, name="size")
        generator = resolve_stream(seed).child("karhunen_loeve").generator()
        return tuple(
            self.realize([generator.gauss(0.0, 1.0) for _ in range(self.terms)])
            for _ in range(count)
        )


def karhunen_loeve(
    kernel: CovarianceKernel,
    points: Sequence[float] | Sequence[Sequence[float]],
    weights: Sequence[float] | None = None,
    *,
    mean: float = 0.0,
    variance_fraction: float = 0.95,
    max_terms: int | None = None,
) -> KarhunenLoeveExpansion:
    """Compute a truncated KL expansion by the Nystrom method.

    Solves ``W^(1/2) C W^(1/2) psi = lambda psi`` and keeps the smallest
    number of leading modes whose eigenvalues sum to at least
    ``variance_fraction`` of the discrete integrated variance, capped by
    ``max_terms``. Unit weights give the discrete KL (PCA) of the covariance
    matrix on a grid.

    Args:
        kernel: Covariance kernel.
        points: Nodes as 1-D scalars or coordinate tuples.
        weights: Positive quadrature weights; ``None`` means all ones.
        mean: Constant field mean.
        variance_fraction: Target explained fraction in ``(0, 1]``.
        max_terms: Optional positive cap on the number of modes.

    Returns:
        Truncated expansion.

    Raises:
        ValueError: If weights are invalid, the fraction is outside
            ``(0, 1]``, or the discrete operator has a materially negative
            eigenvalue.

    Equation:
        [EQ-PROC-KL-001], [EQ-PROC-KL-002], [EQ-PROC-KL-003], [EQ-LINALG-005].

    References:
        [REF-GHANEM-SPANOS-1991, Chapter 2],
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.3],
        [REF-SUDRET-DERKIUREGHIAN-2000, Chapter 2].

    """
    nodes = as_points(points)
    size = len(nodes)
    if weights is None:
        w = (1.0,) * size
    else:
        w = tuple(
            _positive(value, name=f"weights[{i}]") for i, value in enumerate(weights)
        )
        if len(w) != size:
            raise ValueError("weights must match the number of points")
    fraction = require_finite_real(variance_fraction, name="variance_fraction")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("variance_fraction must lie in (0, 1]")
    cap = size if max_terms is None else _positive_integer(max_terms, name="max_terms")
    root = tuple(math.sqrt(value) for value in w)
    covariance = covariance_matrix(kernel, nodes)
    operator = tuple(
        tuple(root[i] * covariance[i][j] * root[j] for j in range(size))
        for i in range(size)
    )
    values, vectors = symmetric_eigen(operator)
    largest = max(abs(v) for v in values)
    if min(values) < -_NEGATIVE_EIGENVALUE_TOLERANCE * largest:
        raise ValueError("discretized covariance operator is not positive semidefinite")
    order = sorted(range(size), key=lambda k: values[k], reverse=True)
    spectrum = tuple(max(0.0, values[k]) for k in order)
    total = math.fsum(spectrum)
    threshold = fraction * total * (1.0 - 1e-12)
    retained = 0
    running = 0.0
    while retained < size and spectrum[retained] > 0.0:
        running += spectrum[retained]
        retained += 1
        if running >= threshold:
            break
    retained = min(retained, cap)
    modes: list[Vector] = []
    for k in order[:retained]:
        phi = [vectors[i][k] / root[i] for i in range(size)]
        pivot = max(range(size), key=lambda i: abs(phi[i]))
        if phi[pivot] < 0.0:
            phi = [-value for value in phi]
        modes.append(tuple(phi))
    return KarhunenLoeveExpansion(
        kernel,
        nodes,
        w,
        require_finite_real(mean, name="mean"),
        spectrum[:retained],
        tuple(modes),
        spectrum,
    )


def karhunen_loeve_interval(
    kernel: CovarianceKernel,
    lower: float,
    upper: float,
    nodes: int,
    *,
    rule: str = "gauss_legendre",
    mean: float = 0.0,
    variance_fraction: float = 0.95,
    max_terms: int | None = None,
) -> KarhunenLoeveExpansion:
    """Compute a Nystrom KL expansion on a 1-D interval with a quadrature rule.

    Args:
        kernel: Covariance kernel.
        lower: Finite lower bound.
        upper: Finite upper bound greater than ``lower``.
        nodes: Number of quadrature nodes.
        rule: One of :data:`QUADRATURE_RULES`.
        mean: Constant field mean.
        variance_fraction: Target explained fraction in ``(0, 1]``.
        max_terms: Optional cap on the number of modes.

    Returns:
        Truncated expansion whose eigenvalues approximate those of the
        integral operator on ``[lower, upper]``.

    Equation:
        [EQ-PROC-KL-002], [EQ-PROC-QUAD-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.3],
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

    """
    points, weights = interval_rule(lower, upper, nodes, rule=rule)
    return karhunen_loeve(
        kernel,
        points,
        weights,
        mean=mean,
        variance_fraction=variance_fraction,
        max_terms=max_terms,
    )


@dataclass(frozen=True, slots=True)
class ExponentialKLMode:
    """One analytic KL eigenpair of the exponential kernel on ``[-a, a]``.

    Args:
        eigenvalue: ``lambda = 2 c sigma^2 / (omega^2 + c^2)``.
        frequency: Root ``omega`` of the mode's transcendental equation.
        parity: ``"even"`` (cosine) or ``"odd"`` (sine).
        half_width: Half-width ``a`` of the domain.

    References:
        [REF-GHANEM-SPANOS-1991, Chapter 2].

    """

    eigenvalue: float
    frequency: float
    parity: str
    half_width: float

    def __call__(self, location: float) -> float:
        """Evaluate the normalized eigenfunction at ``location``.

        Args:
            location: Point in ``[-a, a]``.

        Returns:
            ``cos(omega x) / sqrt(a + sin(2 omega a) / (2 omega))`` for even
            modes or ``sin(omega x) / sqrt(a - sin(2 omega a) / (2 omega))``
            for odd modes.

        Equation:
            [EQ-PROC-KL-004].

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        x = require_finite_real(location, name="location")
        omega, a = self.frequency, self.half_width
        correction = math.sin(2.0 * omega * a) / (2.0 * omega)
        if self.parity == "even":
            return math.cos(omega * x) / math.sqrt(a + correction)
        return math.sin(omega * x) / math.sqrt(a - correction)


def exponential_kernel_kl_modes(
    variance: float, correlation_length: float, half_width: float, count: int
) -> tuple[ExponentialKLMode, ...]:
    """Return the leading exact KL eigenpairs of an exponential kernel.

    For ``C(x, y) = sigma^2 exp(-|x - y| / b)`` on ``[-a, a]`` with
    ``c = 1 / b``, even modes solve ``c - omega tan(omega a) = 0`` with
    ``omega a`` in ``(k pi, k pi + pi/2)`` and odd modes solve
    ``omega + c tan(omega a) = 0`` with ``omega a`` in ``(k pi - pi/2, k pi)``.
    Roots are bracketed in the tangent-free forms and found by Brent's
    method; modes alternate even, odd, even, ... in decreasing eigenvalue.

    Args:
        variance: Positive ``sigma^2``.
        correlation_length: Positive ``b``.
        half_width: Positive ``a``.
        count: Positive number of modes, at most 500.

    Returns:
        Modes in decreasing eigenvalue order.

    Raises:
        ValueError: If a parameter is invalid.

    Equation:
        [EQ-PROC-KL-004], [EQ-PROC-KER-002].

    References:
        [REF-GHANEM-SPANOS-1991, Chapter 2], [REF-BRENT-1973, Chapter 4].

    """
    sigma2 = _positive(variance, name="variance")
    c = 1.0 / _positive(correlation_length, name="correlation_length")
    a = _positive(half_width, name="half_width")
    total = _positive_integer(count, name="count")
    if total > 500:
        raise ValueError("count must not exceed 500")

    def even(omega: float) -> float:
        """Return the tangent-free even-mode residual.

        References:
            [EQ-PROC-KL-004], [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        return c * math.cos(omega * a) - omega * math.sin(omega * a)

    def odd(omega: float) -> float:
        """Return the tangent-free odd-mode residual.

        References:
            [EQ-PROC-KL-004], [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        return omega * math.cos(omega * a) + c * math.sin(omega * a)

    modes: list[ExponentialKLMode] = []
    for index in range(total):
        if index % 2 == 0:
            k = index // 2
            low, high = k * math.pi / a, (k + 0.5) * math.pi / a
            omega = brent_root(even, low, high, tolerance=1e-14).root
            parity = "even"
        else:
            k = (index + 1) // 2
            low, high = (k - 0.5) * math.pi / a, k * math.pi / a
            omega = brent_root(odd, low, high, tolerance=1e-14).root
            parity = "odd"
        eigenvalue = 2.0 * c * sigma2 / (omega * omega + c * c)
        modes.append(ExponentialKLMode(eigenvalue, omega, parity, a))
    return tuple(modes)
