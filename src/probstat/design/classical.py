"""Classical optimal design for linear and linearized regression models.

A regression model ``E[y | x] = f(x)' beta`` with homoscedastic errors and an
**approximate design** ``xi = {(x_i, w_i)}`` (support points ``x_i`` with
weights ``w_i >= 0`` summing to one) has the normalized information matrix
``M(xi) = sum_i w_i f(x_i) f(x_i)'`` (EQ-OED-INFO-001). An exact design with
``n`` runs and ``n_i`` replicates at ``x_i`` is the special case
``w_i = n_i / n``; its Fisher information is ``n M / sigma^2``.

Optimality criteria are scalar functions of ``M`` (EQ-OED-CRIT-001):
D (maximize ``log det M``), A (minimize ``tr M^-1``), E (maximize the smallest
eigenvalue), and I (minimize the average prediction variance over a region).
The Kiefer-Wolfowitz equivalence theorem (EQ-OED-KW-001) certifies D-optimality
through the standardized prediction variance ``d(x, xi) = f(x)' M^-1 f(x)``:
``xi`` is D-optimal iff ``max_x d(x, xi) = p``.

Approximate D-optimal designs on a finite candidate grid are computed by the
multiplicative algorithm (EQ-OED-MULT-001) or the Fedorov-Wynn
vertex-direction algorithm (EQ-OED-FW-001). Exact ``n``-run designs come from
efficient rounding (EQ-OED-ROUND-001) followed by a Fedorov point exchange
(EQ-OED-EXCH-001). Nonlinear models are handled locally: the basis is the
parameter gradient of the mean response at a nominal parameter
(EQ-OED-LOCAL-001).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Real

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    inverse_spd,
    log_det_spd,
    quadratic_form,
    symmetric_eigen,
)
from probstat.modeling import EvaluationCounter
from probstat.optimization import central_gradient

Basis = Callable[[Vector], Sequence[float]]
"""Regression basis: maps a design point to the vector ``f(x)``."""

_CRITERIA = ("D", "A", "E", "I")


def _as_point(value: object) -> Vector:
    """Normalize a scalar or a sequence into a design-point vector.

    References:
        Project API policy. No equation is evaluated.

    """
    if isinstance(value, Real) and not isinstance(value, bool):
        return (require_finite_real(value, name="design point"),)
    if isinstance(value, Sequence):
        point = as_vector(value, name="design point")
        if not point:
            raise ValueError("a design point needs at least one coordinate")
        return point
    raise TypeError("a design point must be a real number or a sequence of reals")


def _as_points(values: Sequence[object], *, name: str) -> tuple[Vector, ...]:
    """Normalize a non-empty collection of design points of equal dimension.

    References:
        Project API policy. No equation is evaluated.

    """
    points = tuple(_as_point(value) for value in values)
    if not points:
        raise ValueError(f"{name} must contain at least one point")
    if len({len(point) for point in points}) != 1:
        raise ValueError(f"all {name} must have the same dimension")
    return points


def _basis_vector(basis: Basis, point: Vector, size: int | None = None) -> Vector:
    """Evaluate a regression basis and validate its length.

    References:
        [EQ-OED-INFO-001]. No new equation is introduced.

    """
    values = as_vector(basis(point), name="basis value")
    if not values:
        raise ValueError("the basis must return at least one function value")
    if size is not None and len(values) != size:
        raise ValueError("the basis returned vectors of different lengths")
    return values


def _moment_matrix(rows: Sequence[Vector], weights: Sequence[float]) -> Matrix:
    """Return ``sum_i w_i f_i f_i'`` for evaluated basis rows.

    References:
        [EQ-OED-INFO-001], [REF-ATKINSON-DONEV-TOBIAS-2007].

    """
    size = len(rows[0])
    return tuple(
        tuple(
            math.fsum(w * row[a] * row[b] for row, w in zip(rows, weights, strict=True))
            for b in range(size)
        )
        for a in range(size)
    )


def _safe_log_det(matrix: Matrix) -> float:
    """Return ``log det M`` or ``-inf`` when ``M`` is not positive definite.

    References:
        [EQ-LINALG-006]. No new equation is introduced.

    """
    try:
        return log_det_spd(matrix)
    except ValueError:
        return -math.inf


def _safe_inverse(matrix: Matrix) -> Matrix | None:
    """Return ``M^-1`` or ``None`` when ``M`` is not positive definite.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No new equation is introduced.

    """
    try:
        return inverse_spd(matrix)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Designs and bases
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ApproximateDesign:
    """Probability measure on design points (an approximate design).

    Args:
        points: Support points; a real number is read as a one-dimensional
            point.
        weights: Non-negative weights; they must sum to one within ``1e-8``
            and are renormalized to sum to one exactly.

    Raises:
        ValueError: If the points are empty or of mixed dimension, the weights
            do not match the points, are negative, or do not sum to one.

    References:
        [REF-KIEFER-WOLFOWITZ-1960], [REF-ATKINSON-DONEV-TOBIAS-2007],
        [REF-FEDOROV-1972].

    """

    points: tuple[Vector, ...]
    weights: Vector

    def __post_init__(self) -> None:
        """Validate and normalize points and weights.

        References:
            [EQ-OED-INFO-001]. No equation is evaluated.

        """
        points = _as_points(tuple(self.points), name="points")
        weights = as_vector(self.weights, name="weights")
        if len(weights) != len(points):
            raise ValueError("weights must have one entry per point")
        if any(w < 0.0 for w in weights):
            raise ValueError("weights must be non-negative")
        total = math.fsum(weights)
        if abs(total - 1.0) > 1e-8:
            raise ValueError("weights must sum to one")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", tuple(w / total for w in weights))

    @classmethod
    def uniform(cls, points: Sequence[object]) -> ApproximateDesign:
        """Return the design with equal weight on every point.

        Args:
            points: Support points.

        Returns:
            Design with weights ``1/len(points)``.

        References:
            [REF-ATKINSON-DONEV-TOBIAS-2007]. No equation is evaluated.

        """
        normalized = _as_points(points, name="points")
        return cls(normalized, tuple(1.0 / len(normalized) for _ in normalized))

    @property
    def dimension(self) -> int:
        """Return the number of design variables per point.

        References:
            Project API policy. No equation is evaluated.

        """
        return len(self.points[0])

    @property
    def support_size(self) -> int:
        """Return the number of points with positive weight.

        References:
            [REF-ATKINSON-DONEV-TOBIAS-2007]. No equation is evaluated.

        """
        return sum(w > 0.0 for w in self.weights)

    def condensed(
        self, *, merge_distance: float = 0.0, min_weight: float = 0.0
    ) -> ApproximateDesign:
        """Drop negligible weights and pool clustered support points.

        Grid algorithms leave small weights on neighbours of the true support
        points. Points with weight below ``min_weight`` are dropped. Then, in
        decreasing weight order, every point within Euclidean distance
        ``merge_distance`` of a kept point is pooled into it; the pooled point
        is the weight-averaged location. Weights are renormalized.

        Args:
            merge_distance: Non-negative pooling radius.
            min_weight: Weight threshold in ``[0, 1)``.

        Returns:
            The condensed design.

        Raises:
            ValueError: If the thresholds are invalid or nothing remains.

        References:
            [REF-ATKINSON-DONEV-TOBIAS-2007]. No equation is evaluated.

        """
        radius = require_finite_real(merge_distance, name="merge_distance")
        threshold = require_finite_real(min_weight, name="min_weight")
        if radius < 0.0 or not 0.0 <= threshold < 1.0:
            raise ValueError("merge_distance >= 0 and 0 <= min_weight < 1 required")
        order = sorted(
            (i for i, w in enumerate(self.weights) if w > threshold and w > 0.0),
            key=lambda i: -self.weights[i],
        )
        if not order:
            raise ValueError("no support point survives min_weight")
        clusters: list[list[int]] = []
        anchors: list[Vector] = []
        for index in order:
            point = self.points[index]
            for cluster, anchor in zip(clusters, anchors, strict=True):
                if math.dist(point, anchor) <= radius:
                    cluster.append(index)
                    break
            else:
                clusters.append([index])
                anchors.append(point)
        pooled_points: list[Vector] = []
        pooled_weights: list[float] = []
        for cluster in clusters:
            mass = math.fsum(self.weights[i] for i in cluster)
            location = tuple(
                math.fsum(self.weights[i] * self.points[i][k] for i in cluster) / mass
                for k in range(self.dimension)
            )
            pooled_points.append(location)
            pooled_weights.append(mass)
        total = math.fsum(pooled_weights)
        pairs = sorted(zip(pooled_points, pooled_weights, strict=True))
        return ApproximateDesign(
            tuple(p for p, _ in pairs), tuple(w / total for _, w in pairs)
        )


@dataclass(frozen=True, slots=True)
class ExactDesign:
    """Design with an integer number of runs at each support point.

    Args:
        points: Distinct support points.
        counts: Positive replicate counts, one per point.

    Raises:
        ValueError: If shapes disagree or a count is not a positive integer.

    References:
        [REF-ATKINSON-DONEV-TOBIAS-2007], [REF-PUKELSHEIM-RIEDER-1992].

    """

    points: tuple[Vector, ...]
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate points and replicate counts.

        References:
            [EQ-OED-ROUND-001]. No equation is evaluated.

        """
        points = _as_points(tuple(self.points), name="points")
        counts = tuple(
            require_non_negative_integer(c, name="counts") for c in self.counts
        )
        if len(counts) != len(points):
            raise ValueError("counts must have one entry per point")
        if any(c == 0 for c in counts):
            raise ValueError("counts must be positive; drop unused points")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "counts", counts)

    @property
    def size(self) -> int:
        """Return the total number of runs ``n``.

        References:
            [REF-PUKELSHEIM-RIEDER-1992]. No equation is evaluated.

        """
        return sum(self.counts)

    def as_approximate(self) -> ApproximateDesign:
        """Return the approximate design with weights ``n_i / n``.

        References:
            [EQ-OED-INFO-001], [REF-ATKINSON-DONEV-TOBIAS-2007].

        """
        total = self.size
        return ApproximateDesign(self.points, tuple(c / total for c in self.counts))

    def runs(self) -> tuple[Vector, ...]:
        """Return every run, with replicates listed consecutively.

        References:
            Project API policy. No equation is evaluated.

        """
        return tuple(
            point
            for point, count in zip(self.points, self.counts, strict=True)
            for _ in range(count)
        )


def polynomial_basis(degree: int) -> Basis:
    """Return the univariate polynomial basis ``f(x) = (1, x, ..., x^degree)``.

    Args:
        degree: Non-negative polynomial degree.

    Returns:
        Basis callable for one-dimensional design points.

    Raises:
        ValueError: If ``degree`` is negative.

    References:
        [REF-ATKINSON-DONEV-TOBIAS-2007]. No equation is evaluated.

    """
    order = require_non_negative_integer(degree, name="degree")

    def basis(point: Vector) -> Vector:
        """Evaluate the monomials at a one-dimensional point.

        References:
            [REF-ATKINSON-DONEV-TOBIAS-2007]. No equation is evaluated.

        """
        if len(point) != 1:
            raise ValueError("polynomial_basis needs one-dimensional points")
        x = point[0]
        return tuple(x**k for k in range(order + 1))

    return basis


@dataclass(frozen=True)
class LocalLinearization:
    """Local design basis of a nonlinear model at a nominal parameter.

    For ``E[y | x] = eta(x, theta)`` the asymptotic information of a design at
    ``theta_0`` is that of the linear model with basis
    ``f(x) = d eta(x, theta) / d theta`` at ``theta_0`` (EQ-OED-LOCAL-001). The
    gradient is a central difference; every call adds ``2 d`` response
    evaluations to :attr:`counter`. A *local* design is optimal only if the
    nominal value is right.

    Args:
        response: Mean response ``eta(x, theta)``.
        nominal: Nominal parameter vector ``theta_0``.
        relative_step: Central-difference step relative to
            ``max(1, |theta_j|)``.
        counter: Mutable evaluation counter.

    Raises:
        TypeError: If ``response`` is not callable.
        ValueError: If ``nominal`` is empty or the step is not positive.

    References:
        [REF-BOX-LUCAS-1959], [REF-ATKINSON-DONEV-TOBIAS-2007].

    """

    response: Callable[[Vector, Vector], float]
    nominal: Vector
    relative_step: float = 1e-6
    counter: EvaluationCounter = field(
        default_factory=EvaluationCounter, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate the response, nominal parameter, and step.

        References:
            [EQ-OED-LOCAL-001]. No equation is evaluated.

        """
        if not callable(self.response):
            raise TypeError("response must be callable")
        nominal = as_vector(self.nominal, name="nominal")
        if not nominal:
            raise ValueError("nominal must be non-empty")
        step = require_finite_real(self.relative_step, name="relative_step")
        if step <= 0.0:
            raise ValueError("relative_step must be positive")
        object.__setattr__(self, "nominal", nominal)
        object.__setattr__(self, "relative_step", step)

    def __call__(self, point: Vector) -> Vector:
        """Return ``d eta(point, theta) / d theta`` at the nominal parameter.

        Args:
            point: Design point.

        Returns:
            Parameter gradient of the mean response.

        Equation:
            [EQ-OED-LOCAL-001], [EQ-OPT-FD-001].

        References:
            [REF-BOX-LUCAS-1959], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        x = _as_point(point)

        def at(theta: Vector) -> float:
            """Evaluate the response at the fixed design point.

            References:
                [EQ-OED-LOCAL-001]. No new equation is introduced.

            """
            return float(self.response(x, theta))

        gradient, evaluations = central_gradient(
            at, self.nominal, relative_step=self.relative_step
        )
        self.counter.add(evaluations)
        return gradient

    @property
    def evaluations(self) -> int:
        """Return response evaluations used so far.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return self.counter.count


# ---------------------------------------------------------------------------
# Information matrix and criteria
# ---------------------------------------------------------------------------


def information_matrix(design: ApproximateDesign, basis: Basis) -> Matrix:
    """Return the normalized information matrix ``M(xi)`` of a design.

    Args:
        design: Approximate design.
        basis: Regression basis ``f``.

    Returns:
        ``M = sum_i w_i f(x_i) f(x_i)'`` (``p`` by ``p``).

    Raises:
        ValueError: If the basis returns vectors of different lengths.

    Equation:
        [EQ-OED-INFO-001].

    References:
        [REF-ATKINSON-DONEV-TOBIAS-2007], [REF-FEDOROV-1972].

    """
    rows = _rows(basis, design.points)
    return _moment_matrix(rows, design.weights)


def _rows(basis: Basis, points: Sequence[Vector]) -> tuple[Vector, ...]:
    """Evaluate a basis on points with a common length check.

    References:
        [EQ-OED-INFO-001]. No new equation is introduced.

    """
    first = _basis_vector(basis, points[0])
    return (first,) + tuple(
        _basis_vector(basis, point, len(first)) for point in points[1:]
    )


def standardized_variances(
    design: ApproximateDesign, basis: Basis, points: Sequence[object]
) -> Vector:
    """Return ``d(x, xi) = f(x)' M(xi)^-1 f(x)`` at each point.

    ``sigma^2 d(x, xi) / n`` is the variance of the least-squares prediction at
    ``x`` for an ``n``-run design with weights ``xi``.

    Args:
        design: Approximate design with nonsingular ``M``.
        basis: Regression basis.
        points: Evaluation points.

    Returns:
        Standardized prediction variances.

    Raises:
        ValueError: If ``M(xi)`` is singular.

    Equation:
        [EQ-OED-KW-001].

    References:
        [REF-KIEFER-WOLFOWITZ-1960], [REF-ATKINSON-DONEV-TOBIAS-2007].

    """
    inverse = _safe_inverse(information_matrix(design, basis))
    if inverse is None:
        raise ValueError("the information matrix of the design is singular")
    targets = _as_points(points, name="points")
    return tuple(
        quadratic_form(inverse, _basis_vector(basis, x, len(inverse))) for x in targets
    )


def design_criterion(
    design: ApproximateDesign,
    basis: Basis,
    criterion: str = "D",
    *,
    region: Sequence[object] | None = None,
) -> float:
    """Evaluate a classical optimality criterion of a design.

    * ``"D"``: ``log det M`` (maximize; ``-inf`` if singular).
    * ``"A"``: ``tr M^-1`` (minimize; ``inf`` if singular).
    * ``"E"``: smallest eigenvalue of ``M`` (maximize).
    * ``"I"``: average of ``d(x, xi)`` over the ``region`` points
      (minimize; ``inf`` if singular), i.e. ``tr(M^-1 M_region)``.

    Args:
        design: Approximate design.
        basis: Regression basis.
        criterion: One of ``"D"``, ``"A"``, ``"E"``, ``"I"``.
        region: Equally weighted integration points (required for ``"I"``).

    Returns:
        Criterion value.

    Raises:
        ValueError: For an unknown criterion or a missing region.

    Equation:
        [EQ-OED-CRIT-001].

    References:
        [REF-ATKINSON-DONEV-TOBIAS-2007], [REF-FEDOROV-1972].

    """
    name = criterion.upper()
    if name not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}")
    matrix = information_matrix(design, basis)
    if name == "D":
        return _safe_log_det(matrix)
    if name == "E":
        values, _ = symmetric_eigen(matrix)
        return values[0]
    inverse = _safe_inverse(matrix)
    if name == "A":
        if inverse is None:
            return math.inf
        return math.fsum(inverse[i][i] for i in range(len(inverse)))
    if region is None:
        raise ValueError("the I criterion needs region points")
    targets = _as_points(region, name="region")
    if inverse is None:
        return math.inf
    return math.fsum(
        quadratic_form(inverse, _basis_vector(basis, x, len(inverse)))
        for x in targets
    ) / len(targets)


def d_efficiency(
    design: ApproximateDesign, reference: ApproximateDesign, basis: Basis
) -> float:
    """Return the D-efficiency ``(det M(xi) / det M(xi_ref))^(1/p)``.

    An efficiency ``e`` means ``xi`` needs ``1/e`` times as many runs as the
    reference for the same generalized variance.

    Args:
        design: Design to rate.
        reference: Reference design with nonsingular ``M``.
        basis: Regression basis.

    Returns:
        Non-negative efficiency (0 when ``M(xi)`` is singular).

    Raises:
        ValueError: If the reference information matrix is singular.

    Equation:
        [EQ-OED-CRIT-001].

    References:
        [REF-ATKINSON-DONEV-TOBIAS-2007].

    """
    matrix = information_matrix(reference, basis)
    reference_value = _safe_log_det(matrix)
    if reference_value == -math.inf:
        raise ValueError("the reference information matrix is singular")
    value = _safe_log_det(information_matrix(design, basis))
    if value == -math.inf:
        return 0.0
    return math.exp((value - reference_value) / len(matrix))


# ---------------------------------------------------------------------------
# Equivalence theorem
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquivalenceCheck:
    """Kiefer-Wolfowitz certificate of D-optimality on a candidate set.

    Args:
        parameters: Number of basis functions ``p``.
        max_variance: ``max_x d(x, xi)`` over the candidates.
        argmax: Candidate attaining the maximum.
        support_variances: ``d(x_i, xi)`` at the design's support points
            (equal to ``p`` at a D-optimum wherever ``w_i > 0``).
        efficiency_lower_bound: ``exp(1 - max_variance / p)``, a guaranteed
            lower bound on the D-efficiency relative to the optimum on the
            candidate set.
        tolerance: Relative tolerance of the check.
        satisfied: Whether ``max_variance <= p (1 + tolerance)``.

    References:
        [REF-KIEFER-WOLFOWITZ-1960], [REF-ATKINSON-DONEV-TOBIAS-2007].

    """

    parameters: int
    max_variance: float
    argmax: Vector
    support_variances: Vector
    efficiency_lower_bound: float
    tolerance: float
    satisfied: bool


def equivalence_check(
    design: ApproximateDesign,
    basis: Basis,
    candidates: Sequence[object],
    *,
    tolerance: float = 1e-3,
) -> EquivalenceCheck:
    """Check the Kiefer-Wolfowitz equivalence theorem on a candidate set.

    For any design, ``sum_i w_i d(x_i, xi) = tr(M^-1 M) = p``, so
    ``max_x d(x, xi) >= p``; equality holds iff ``xi`` is D-optimal (and then
    also G-optimal). Concavity of ``log det`` gives
    ``log det M* - log det M(xi) <= max_x d(x, xi) - p`` for the optimum
    ``M*`` on the candidates, hence the efficiency bound
    ``(det M / det M*)^(1/p) >= exp(1 - max_x d / p)``.

    Args:
        design: Design with nonsingular ``M``.
        basis: Regression basis.
        candidates: Candidate points (the design space).
        tolerance: Non-negative relative tolerance.

    Returns:
        The certificate.

    Raises:
        ValueError: If ``M`` is singular or the tolerance is negative.

    Equation:
        [EQ-OED-KW-001].

    References:
        [REF-KIEFER-WOLFOWITZ-1960], [REF-ATKINSON-DONEV-TOBIAS-2007],
        [REF-FEDOROV-1972].

    """
    relative = require_finite_real(tolerance, name="tolerance")
    if relative < 0.0:
        raise ValueError("tolerance must be non-negative")
    points = _as_points(candidates, name="candidates")
    variances = standardized_variances(design, basis, points)
    index = max(range(len(points)), key=lambda i: variances[i])
    p = len(_basis_vector(basis, points[0]))
    support = standardized_variances(design, basis, design.points)
    largest = variances[index]
    return EquivalenceCheck(
        p,
        largest,
        points[index],
        support,
        math.exp(1.0 - largest / p),
        relative,
        largest <= p * (1.0 + relative),
    )


# ---------------------------------------------------------------------------
# Approximate D-optimal designs on a grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptimalDesignResult:
    """Approximate D-optimal design computed on a candidate grid.

    Args:
        design: Weights on the candidates (weights below ``1e-12`` dropped).
        information: Normalized information matrix of the full weight vector.
        log_det: ``log det M``.
        equivalence: Equivalence-theorem certificate on the candidates.
        iterations: Algorithm iterations performed.
        converged: Whether ``max d <= p (1 + tolerance)`` was reached.
        method: ``"multiplicative"`` or ``"fedorov-wynn"``.
        basis_evaluations: Basis evaluations (one per candidate; cached).

    References:
        [REF-SILVEY-TITTERINGTON-TORSNEY-1978], [REF-WYNN-1970],
        [REF-FEDOROV-1972].

    """

    design: ApproximateDesign
    information: Matrix
    log_det: float
    equivalence: EquivalenceCheck
    iterations: int
    converged: bool
    method: str
    basis_evaluations: int


def d_optimal_design(
    candidates: Sequence[object],
    basis: Basis,
    *,
    method: str = "multiplicative",
    tolerance: float = 1e-6,
    max_iterations: int = 10000,
    initial_weights: Sequence[float] | None = None,
) -> OptimalDesignResult:
    """Compute an approximate D-optimal design on a finite candidate set.

    ``"multiplicative"`` iterates ``w_i <- w_i d(x_i, w) / p``
    (EQ-OED-MULT-001); weights stay positive and sum to one, and
    ``log det M`` increases monotonically. ``"fedorov-wynn"`` moves mass
    ``alpha = (d_max - p) / ((d_max - 1) p)`` to the candidate of largest
    variance, the exact line-search step for ``log det`` (EQ-OED-FW-001).
    Both stop when ``max_x d(x, w) <= p (1 + tolerance)``; the equivalence
    theorem then bounds the D-efficiency below by ``exp(-tolerance)``.

    Args:
        candidates: Finite design space.
        basis: Regression basis.
        method: ``"multiplicative"`` or ``"fedorov-wynn"``.
        tolerance: Positive relative stopping tolerance.
        max_iterations: Positive iteration limit.
        initial_weights: Optional positive starting weights (default
            uniform).

    Returns:
        Design, information matrix, certificate, and iteration counts.

    Raises:
        ValueError: For an unknown method, invalid settings, or a singular
            starting information matrix.

    Equation:
        [EQ-OED-MULT-001], [EQ-OED-FW-001], [EQ-OED-KW-001].

    References:
        [REF-SILVEY-TITTERINGTON-TORSNEY-1978], [REF-WYNN-1970],
        [REF-FEDOROV-1972], [REF-ATKINSON-DONEV-TOBIAS-2007].

    """
    if method not in ("multiplicative", "fedorov-wynn"):
        raise ValueError("method must be 'multiplicative' or 'fedorov-wynn'")
    relative = require_finite_real(tolerance, name="tolerance")
    if relative <= 0.0:
        raise ValueError("tolerance must be positive")
    limit = require_non_negative_integer(max_iterations, name="max_iterations")
    if limit == 0:
        raise ValueError("max_iterations must be positive")
    points = _as_points(candidates, name="candidates")
    rows = _rows(basis, points)
    p = len(rows[0])
    if initial_weights is None:
        weights = [1.0 / len(points)] * len(points)
    else:
        start = as_vector(initial_weights, name="initial_weights")
        if len(start) != len(points) or any(w <= 0.0 for w in start):
            raise ValueError("initial_weights must be positive, one per candidate")
        total = math.fsum(start)
        weights = [w / total for w in start]
    matrix = _moment_matrix(rows, weights)
    if _safe_inverse(matrix) is None:
        raise ValueError("the starting design has a singular information matrix")
    iterations = 0
    converged = False
    while True:
        inverse = _safe_inverse(matrix)
        if inverse is None:
            raise ValueError("the information matrix became singular")
        variances = [quadratic_form(inverse, row) for row in rows]
        largest = max(variances)
        if largest <= p * (1.0 + relative):
            converged = True
            break
        if iterations >= limit:
            break
        iterations += 1
        if method == "multiplicative":
            weights = [w * v / p for w, v in zip(weights, variances, strict=True)]
            total = math.fsum(weights)
            weights = [w / total for w in weights]
            matrix = _moment_matrix(rows, weights)
        else:
            index = variances.index(largest)
            alpha = (largest - p) / ((largest - 1.0) * p)
            weights = [(1.0 - alpha) * w for w in weights]
            weights[index] += alpha
            row = rows[index]
            matrix = tuple(
                tuple(
                    (1.0 - alpha) * matrix[a][b] + alpha * row[a] * row[b]
                    for b in range(p)
                )
                for a in range(p)
            )
    kept = [i for i, w in enumerate(weights) if w > 1e-12]
    mass = math.fsum(weights[i] for i in kept)
    design = ApproximateDesign(
        tuple(points[i] for i in kept), tuple(weights[i] / mass for i in kept)
    )
    final_inverse = _safe_inverse(matrix)
    if final_inverse is None:
        raise ValueError("the information matrix became singular")
    variances = [quadratic_form(final_inverse, row) for row in rows]
    largest = max(variances)
    certificate = EquivalenceCheck(
        p,
        largest,
        points[variances.index(largest)],
        tuple(variances[i] for i in kept),
        math.exp(1.0 - largest / p),
        relative,
        largest <= p * (1.0 + relative),
    )
    return OptimalDesignResult(
        design,
        matrix,
        _safe_log_det(matrix),
        certificate,
        iterations,
        converged,
        method,
        len(points),
    )


# ---------------------------------------------------------------------------
# Exact designs
# ---------------------------------------------------------------------------


def efficient_rounding(design: ApproximateDesign, runs: int) -> ExactDesign:
    """Round an approximate design to an exact ``n``-run design.

    Efficient rounding (Pukelsheim-Rieder): with ``l`` support points start
    from ``n_i = ceil((n - l/2) w_i)``; while ``sum n_i < n`` increment a
    count with the smallest ``n_j / w_j``; while ``sum n_i > n`` decrement a
    count with the largest ``(n_j - 1) / w_j``. Points rounded to zero runs
    are dropped.

    Args:
        design: Approximate design (zero-weight points are ignored).
        runs: Positive total number of runs ``n``.

    Returns:
        Exact design with ``n`` runs.

    Raises:
        ValueError: If ``runs`` is not positive.

    Equation:
        [EQ-OED-ROUND-001].

    References:
        [REF-PUKELSHEIM-RIEDER-1992].

    """
    n = require_non_negative_integer(runs, name="runs")
    if n == 0:
        raise ValueError("runs must be positive")
    support = [(p, w) for p, w in zip(design.points, design.weights, strict=True)]
    support = [(p, w) for p, w in support if w > 0.0]
    count = len(support)
    weights = [w for _, w in support]
    counts = [math.ceil((n - 0.5 * count) * w) for w in weights]
    counts = [max(c, 0) for c in counts]
    while sum(counts) < n:
        j = min(range(count), key=lambda i: counts[i] / weights[i])
        counts[j] += 1
    while sum(counts) > n:
        j = max(
            (i for i in range(count) if counts[i] > 0),
            key=lambda i: (counts[i] - 1) / weights[i],
        )
        counts[j] -= 1
    kept = [(p, c) for (p, _), c in zip(support, counts, strict=True) if c > 0]
    return ExactDesign(tuple(p for p, _ in kept), tuple(c for _, c in kept))


@dataclass(frozen=True, slots=True)
class ExchangeResult:
    """Exact design improved by Fedorov point exchange.

    Args:
        design: Final exact design.
        log_det: ``log det M`` of its normalized information matrix.
        initial_log_det: ``log det M`` of the starting design.
        passes: Exchange passes performed.
        swaps: Accepted exchanges.
        converged: Whether a pass found no improving exchange.

    References:
        [REF-FEDOROV-1972], [REF-ATKINSON-DONEV-TOBIAS-2007].

    """

    design: ExactDesign
    log_det: float
    initial_log_det: float
    passes: int
    swaps: int
    converged: bool


def exchange_exact_design(
    design: ExactDesign,
    candidates: Sequence[object],
    basis: Basis,
    *,
    max_passes: int = 100,
) -> ExchangeResult:
    """Improve an exact design by best-improvement Fedorov exchanges.

    Each pass evaluates every exchange of one run at ``x_i`` for one run at a
    candidate ``x`` and applies the one that increases ``log det M`` most
    (EQ-OED-EXCH-001). The search stops when no exchange improves by more
    than ``1e-12``. The result is a local optimum among exact designs; it is
    never worse than the start.

    Args:
        design: Starting exact design with nonsingular ``M``.
        candidates: Candidate points.
        basis: Regression basis.
        max_passes: Positive pass limit.

    Returns:
        Improved design and exchange statistics.

    Raises:
        ValueError: If the starting design is singular or settings invalid.

    Equation:
        [EQ-OED-EXCH-001], [EQ-OED-INFO-001].

    References:
        [REF-FEDOROV-1972], [REF-ATKINSON-DONEV-TOBIAS-2007].

    """
    limit = require_non_negative_integer(max_passes, name="max_passes")
    if limit == 0:
        raise ValueError("max_passes must be positive")
    pool = list(_as_points(candidates, name="candidates"))
    cache: dict[Vector, Vector] = {}

    def row(point: Vector) -> Vector:
        """Return the cached basis vector of a point.

        References:
            [EQ-OED-INFO-001]. No new equation is introduced.

        """
        if point not in cache:
            cache[point] = _basis_vector(basis, point)
        return cache[point]

    runs = list(design.runs())
    n = len(runs)

    def score(points: Sequence[Vector]) -> float:
        """Return ``log det`` of the normalized information of the runs.

        References:
            [EQ-OED-INFO-001]. No new equation is introduced.

        """
        rows = [row(x) for x in points]
        return _safe_log_det(_moment_matrix(rows, [1.0 / n] * n))

    current = score(runs)
    if current == -math.inf:
        raise ValueError("the starting exact design is singular")
    initial = current
    passes = 0
    swaps = 0
    converged = False
    distinct_positions: dict[Vector, int] = {}
    while passes < limit:
        passes += 1
        distinct_positions = {}
        for position, point in enumerate(runs):
            distinct_positions.setdefault(point, position)
        best = (current, -1, runs[0])
        for position in distinct_positions.values():
            for candidate in pool:
                if candidate == runs[position]:
                    continue
                trial = list(runs)
                trial[position] = candidate
                value = score(trial)
                if value > best[0] + 1e-12:
                    best = (value, position, candidate)
        if best[1] < 0:
            converged = True
            break
        runs[best[1]] = best[2]
        current = best[0]
        swaps += 1
    tally: dict[Vector, int] = {}
    for point in runs:
        tally[point] = tally.get(point, 0) + 1
    ordered = sorted(tally.items())
    return ExchangeResult(
        ExactDesign(tuple(p for p, _ in ordered), tuple(c for _, c in ordered)),
        current,
        initial,
        passes,
        swaps,
        converged,
    )
