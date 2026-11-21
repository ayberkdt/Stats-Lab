"""Active subspaces: directions along which a function changes on average.

For a differentiable ``f`` of inputs with density ``rho``, Constantine, Dow
and Wang (2014) study the matrix

``C = E[grad f(X) grad f(X)']``,   ``C = W Lambda W'``

([EQ-SENS-ACTIVE-001]). The eigenvalue ``lambda_j = E[(w_j' grad f)^2]`` is
the mean squared directional derivative along ``w_j``. When the first ``k``
eigenvalues carry almost all of the trace, ``f(x)`` is nearly a function of
the ``k`` *active variables* ``y = W_1' x`` only, and a ``d``-dimensional
study can be replaced by one in ``k`` dimensions (a response surface in
``y``, a plot of ``f`` against ``y``). ``C`` is estimated by Monte Carlo,
``C_M = (1/M) sum_m grad f(x_m) grad f(x_m)'``.

Scaling. The eigenvectors depend on the coordinates: rescaling one input
rescales its partial derivative. For a :class:`ProbabilisticModel` this
module therefore works in the standard-normal space ``u`` of the inputs
(``x = T^-1(u)``, ``u ~ N(0, I)``), where every coordinate has unit variance
and the weight ``rho`` is the standard normal density. Gradients with
respect to ``u`` come from forward-mode automatic differentiation of the
model combined with the derivative of the marginal map
(``dx_i / du_i = phi(u_i) / f_i(x_i)``, [EQ-SENS-ACTIVE-004]) for
independent inputs, or from central differences in ``u`` otherwise.

Diagnostics. A gap between ``lambda_k`` and ``lambda_{k+1}`` is what makes
the ``k``-dimensional subspace well determined by a finite sample; the
default dimension is the position of the largest gap, which is a heuristic.
A bootstrap over the gradient sample gives percentile intervals of the
eigenvalues and, for each candidate ``k``, the mean distance
``|W_1 W_1' - W_1* W_1*'|_2`` between the estimated and the resampled
subspace ([EQ-SENS-ACTIVE-003]); it measures sampling variability only and
is itself a heuristic (eigenvalues of a sample moment matrix are biased:
the largest upwards, the smallest downwards). Activity scores
``alpha_i = sum_{j <= k} lambda_j w_ij^2`` ([EQ-SENS-ACTIVE-002]) rank the
original coordinates; with ``k = d`` they equal ``E[(df/dx_i)^2]``, the
derivative-based measure of [EQ-SENS-DGSM-001].
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, symmetric_eigen
from probstat.modeling import ProbabilisticModel
from probstat.numerics import AutodiffUnsupportedError
from probstat.optimization import central_gradient
from probstat.statistics import quantile

from .derivative_based import _dual_value_and_gradient

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class ActiveSubspaceResult:
    """Eigen-decomposition of the gradient moment matrix with diagnostics.

    Args:
        input_names: Coordinate names (inputs; in standard-normal space the
            coordinate ``u_i`` of input ``i``).
        eigenvalues: ``lambda_1 >= ... >= lambda_d >= 0``.
        eigenvectors: ``eigenvectors[j]`` is the unit vector ``w_j``; its
            sign is fixed so that its largest component is positive.
        matrix: The estimate ``C_M``.
        gaps: ``lambda_k - lambda_{k+1}`` for ``k = 1, ..., d - 1``.
        dimension: Active dimension ``k`` used by the scores and methods.
        dimension_rule: ``"largest-gap"`` (heuristic) or ``"user"``.
        activity_scores: ``alpha_i = sum_{j <= k} lambda_j w_ij^2``.
        eigenvalue_intervals: Bootstrap percentile intervals of the
            eigenvalues (``None`` without bootstrap).
        subspace_errors: Mean bootstrap distance between the estimated and
            the resampled ``k``-dimensional subspace for ``k = 1..d - 1``
            (``None`` without bootstrap); values in ``[0, 1]``.
        bootstrap: Number of bootstrap replicates.
        confidence: Level of the eigenvalue intervals.
        sample_size: Number ``M`` of gradients.
        evaluations: Model evaluations spent (0 for given gradients).
        gradient_method: ``"given"``, ``"autodiff"``, or
            ``"finite-difference"``.
        space: ``"given"`` or ``"standard-normal"``.
        points: Gradient locations (``None`` if not supplied).
        values: Function values at ``points`` (``None`` if not supplied).
        gradient_fallback_reason: Why autodiff was not used (``None`` if it
            was, or if finite differences were requested).

    References:
        [REF-CONSTANTINE-DOW-WANG-2014].

    """

    input_names: tuple[str, ...]
    eigenvalues: Vector
    eigenvectors: Matrix
    matrix: Matrix
    gaps: Vector
    dimension: int
    dimension_rule: str
    activity_scores: Vector
    eigenvalue_intervals: tuple[tuple[float, float], ...] | None
    subspace_errors: Vector | None
    bootstrap: int
    confidence: float
    sample_size: int
    evaluations: int
    gradient_method: str
    space: str
    points: Matrix | None = None
    values: Vector | None = None
    gradient_fallback_reason: str | None = None

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> activity score``.

        References:
            [EQ-SENS-ACTIVE-002].

        """
        return dict(zip(self.input_names, self.activity_scores, strict=True))

    def active_directions(self) -> Matrix:
        """Return the ``k`` leading eigenvectors ``w_1, ..., w_k`` as rows.

        References:
            [EQ-SENS-ACTIVE-001], [REF-CONSTANTINE-DOW-WANG-2014].

        """
        return self.eigenvectors[: self.dimension]

    def explained_fraction(self) -> float:
        """Return ``sum_{j <= k} lambda_j / trace(C)``.

        The trace is ``E[|grad f|^2]``; the returned share is the part of
        the mean squared gradient that lies in the active subspace (1 for a
        zero matrix by convention is not used: a zero trace returns ``nan``).

        References:
            [EQ-SENS-ACTIVE-001], [REF-CONSTANTINE-DOW-WANG-2014].

        """
        trace = math.fsum(self.eigenvalues)
        if not trace > 0.0:
            return math.nan
        return math.fsum(self.eigenvalues[: self.dimension]) / trace

    def active_variables(self, point: Sequence[float]) -> Vector:
        """Return the active variables ``y = W_1' x`` of one point.

        Args:
            point: Point in the coordinates of the analysis (standard-normal
                ``u`` for a model analysis).

        Returns:
            ``(w_1' x, ..., w_k' x)``.

        Raises:
            ValueError: If the point has the wrong length or is not finite.

        Equation:
            [EQ-SENS-ACTIVE-001].

        References:
            [REF-CONSTANTINE-DOW-WANG-2014].

        """
        x = tuple(
            require_finite_real(v, name=f"point[{i}]") for i, v in enumerate(point)
        )
        if len(x) != len(self.input_names):
            raise ValueError("point length must equal the number of inputs")
        return tuple(
            math.fsum(w * v for w, v in zip(direction, x, strict=True))
            for direction in self.active_directions()
        )

    def sufficient_summary(self) -> tuple[tuple[float, float], ...]:
        """Return the pairs ``(w_1' x_m, f(x_m))`` of the gradient sample.

        Plotted, the pairs show whether ``f`` is (nearly) a function of the
        first active variable: a thin curve supports a one-dimensional
        active subspace, a diffuse cloud does not.

        Returns:
            One ``(y, f)`` pair per sample point, in sample order.

        Raises:
            ValueError: If the result holds no points and values.

        Equation:
            [EQ-SENS-ACTIVE-001].

        References:
            [REF-CONSTANTINE-DOW-WANG-2014].

        """
        if self.points is None or self.values is None:
            raise ValueError("the result holds no sample points and function values")
        first = self.eigenvectors[0]
        return tuple(
            (math.fsum(w * v for w, v in zip(first, point, strict=True)), value)
            for point, value in zip(self.points, self.values, strict=True)
        )


def _moment_matrix(
    outer: Sequence[Sequence[float]], rows: Sequence[int], size: int
) -> Matrix:
    """Return ``(1/M) sum_m g_m g_m'`` from stored upper-triangle products.

    References:
        [EQ-SENS-ACTIVE-001], [REF-CONSTANTINE-DOW-WANG-2014].

    """
    count = len(rows)
    upper = [math.fsum(outer[m][t] for m in rows) / count for t in range(len(outer[0]))]
    matrix = [[0.0] * size for _ in range(size)]
    position = 0
    for i in range(size):
        for j in range(i, size):
            matrix[i][j] = matrix[j][i] = upper[position]
            position += 1
    return tuple(tuple(row) for row in matrix)


def _descending_eigen(matrix: Matrix) -> tuple[Vector, Matrix]:
    """Return eigenvalues in descending order and eigenvectors as rows.

    Eigenvalues are clipped at zero (the matrix is positive semidefinite, so
    negative values are rounding) and each eigenvector's largest component
    is made positive.

    References:
        [EQ-LINALG-005], [REF-GOLUB-VANLOAN-2013, Section 8.5].

    """
    values, columns = symmetric_eigen(matrix)
    size = len(values)
    vectors: list[Vector] = []
    for j in reversed(range(size)):
        vector = tuple(columns[i][j] for i in range(size))
        pivot = max(vector, key=abs)
        vectors.append(tuple(-v for v in vector) if pivot < 0.0 else vector)
    return tuple(max(v, 0.0) for v in reversed(values)), tuple(vectors)


def _subspace_distance(first: Matrix, second: Matrix) -> float:
    """Return ``|P_1 - P_2|_2`` for two ``k``-dimensional subspaces.

    ``first`` and ``second`` hold orthonormal bases as rows. With
    ``G = W_1' W_2`` the distance between the orthogonal projectors is
    ``sqrt(1 - sigma_min(G)^2)``, the sine of the largest principal angle.

    References:
        [EQ-SENS-ACTIVE-003], [REF-GOLUB-VANLOAN-2013, Section 2.5.3].

    """
    gram = tuple(
        tuple(math.fsum(a * b for a, b in zip(u, v, strict=True)) for v in second)
        for u in first
    )
    k = len(gram)
    if k == 1:
        smallest = gram[0][0] ** 2
    else:
        product = tuple(
            tuple(
                math.fsum(gram[i][t] * gram[j][t] for t in range(k)) for j in range(k)
            )
            for i in range(k)
        )
        smallest = symmetric_eigen(product)[0][0]
    return math.sqrt(min(1.0, max(0.0, 1.0 - smallest)))


def active_subspace_from_gradients(
    gradients: Sequence[Sequence[float]],
    *,
    names: Sequence[str] | None = None,
    dimension: int | None = None,
    bootstrap: int = 0,
    confidence: float = 0.95,
    seed: int | RandomStream | None = None,
    points: Sequence[Sequence[float]] | None = None,
    values: Sequence[float] | None = None,
) -> ActiveSubspaceResult:
    """Estimate an active subspace from a sample of gradients.

    ``C_M = (1/M) sum_m g_m g_m'`` is diagonalized by Jacobi rotations. The
    gradients must be evaluated at independent draws ``x_m`` of the input
    density and expressed in coordinates of comparable scale (see the module
    docstring). With ``bootstrap = B > 0`` the gradients are resampled with
    replacement ``B`` times; the eigenvalue intervals are the percentile
    intervals of the resampled eigenvalues and ``subspace_errors[k - 1]`` is
    the mean of ``|W_1 W_1' - W_1* W_1*'|_2`` over the replicates.

    Args:
        gradients: ``M >= 2`` gradient vectors of length ``d >= 1``.
        names: Coordinate names (default ``x0, x1, ...``).
        dimension: Active dimension ``k`` in ``[1, d]``; ``None`` selects
            the largest eigenvalue gap (``k = 1`` when ``d = 1``), a
            heuristic.
        bootstrap: Number of bootstrap replicates (0: none).
        confidence: Level of the eigenvalue intervals in ``(0, 1)``.
        seed: Seed or stream of the bootstrap (child
            ``"active-subspace-bootstrap"``).
        points: Optional sample points ``x_m`` (kept for
            :meth:`ActiveSubspaceResult.sufficient_summary`).
        values: Optional function values ``f(x_m)``.

    Returns:
        Eigenpairs, gaps, activity scores, and bootstrap diagnostics.

    Raises:
        ValueError: For ragged or non-finite data, fewer than two gradients,
            an invalid dimension, bootstrap count or confidence, or points
            and values that do not match the gradients.

    Equation:
        [EQ-SENS-ACTIVE-001], [EQ-SENS-ACTIVE-002], [EQ-SENS-ACTIVE-003].

    References:
        [REF-CONSTANTINE-DOW-WANG-2014], [REF-GOLUB-VANLOAN-2013, Section
        2.5.3], [REF-EFRON-TIBSHIRANI-1993].

    """
    rows = tuple(
        tuple(
            require_finite_real(v, name=f"gradients[{m}][{i}]") for i, v in enumerate(g)
        )
        for m, g in enumerate(gradients)
    )
    count = len(rows)
    if count < 2:
        raise ValueError("at least two gradients are required")
    size = len(rows[0])
    if size == 0 or any(len(g) != size for g in rows):
        raise ValueError("gradients must be a rectangular matrix with columns")
    labels = tuple(f"x{i}" for i in range(size)) if names is None else tuple(names)
    if len(labels) != size:
        raise ValueError("names must match the gradient length")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a non-negative integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if dimension is not None and (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not 1 <= dimension <= size
    ):
        raise ValueError("dimension must be an integer in [1, d]")
    kept_points: Matrix | None = None
    kept_values: Vector | None = None
    if (points is None) != (values is None):
        raise ValueError("points and values must be given together")
    if points is not None and values is not None:
        kept_points = tuple(
            tuple(
                require_finite_real(v, name=f"points[{m}][{i}]")
                for i, v in enumerate(p)
            )
            for m, p in enumerate(points)
        )
        kept_values = tuple(
            require_finite_real(v, name=f"values[{m}]") for m, v in enumerate(values)
        )
        if (
            len(kept_points) != count
            or len(kept_values) != count
            or any(len(p) != size for p in kept_points)
        ):
            raise ValueError("points and values must match the gradients")
    outer = [[g[i] * g[j] for i in range(size) for j in range(i, size)] for g in rows]
    matrix = _moment_matrix(outer, range(count), size)
    eigenvalues, eigenvectors = _descending_eigen(matrix)
    gaps = tuple(eigenvalues[k] - eigenvalues[k + 1] for k in range(size - 1))
    if dimension is None:
        active = 1 + max(range(size - 1), key=gaps.__getitem__) if size > 1 else 1
        rule = "largest-gap"
    else:
        active, rule = dimension, "user"
    scores = tuple(
        math.fsum(eigenvalues[j] * eigenvectors[j][i] ** 2 for j in range(active))
        for i in range(size)
    )
    intervals: tuple[tuple[float, float], ...] | None = None
    errors: Vector | None = None
    if bootstrap:
        generator = resolve_stream(seed).child("active-subspace-bootstrap").generator()
        resampled: list[list[float]] = [[] for _ in range(size)]
        distances: list[list[float]] = [[] for _ in range(size - 1)]
        for _ in range(bootstrap):
            draw = [generator.randrange(count) for _ in range(count)]
            values_b, vectors_b = _descending_eigen(_moment_matrix(outer, draw, size))
            for j, value in enumerate(values_b):
                resampled[j].append(value)
            for k in range(1, size):
                distances[k - 1].append(
                    _subspace_distance(eigenvectors[:k], vectors_b[:k])
                )
        tail = 0.5 * (1.0 - confidence)
        intervals = tuple(
            (quantile(column, tail), quantile(column, 1.0 - tail))
            for column in resampled
        )
        errors = tuple(math.fsum(column) / bootstrap for column in distances)
    return ActiveSubspaceResult(
        labels,
        eigenvalues,
        eigenvectors,
        matrix,
        gaps,
        active,
        rule,
        scores,
        intervals,
        errors,
        bootstrap,
        confidence,
        count,
        0,
        "given",
        "given",
        kept_points,
        kept_values,
    )


def active_subspace(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    seed: int | RandomStream | None,
    gradient: str = "auto",
    dimension: int | None = None,
    bootstrap: int = 0,
    confidence: float = 0.95,
    relative_step: float | None = None,
) -> ActiveSubspaceResult:
    """Estimate the active subspace of a model QoI in standard-normal space.

    ``size`` points ``u_m ~ N(0, I)`` are drawn; ``g(u) = f(T^-1(u))`` and its
    gradient are evaluated at each and passed to
    :func:`active_subspace_from_gradients`. For independent inputs the
    gradient is ``dg/du_i = (df/dx_i) phi(u_i) / f_i(x_i)`` with ``df/dx``
    from one dual-number model call; when the model does not accept dual
    numbers, or the inputs are dependent, central differences of ``g`` in
    ``u`` are used (``2 d + 1`` model calls per point). The eigenvectors,
    the active variables and the sufficient summary refer to ``u``; map a
    physical point with ``model.inputs.to_standard_normal`` first.

    Args:
        model: Probabilistic model with continuous inputs.
        quantity: QoI name.
        size: Number ``M >= 2`` of gradient points.
        seed: Root seed or stream (children ``"active-subspace"`` for the
            points and ``"active-subspace-bootstrap"``).
        gradient: ``"auto"`` (autodiff with finite-difference fallback),
            ``"autodiff"``, or ``"finite-difference"``.
        dimension: Active dimension, or ``None`` for the largest gap.
        bootstrap: Number of bootstrap replicates.
        confidence: Level of the eigenvalue intervals.
        relative_step: Central-difference step relative to
            ``max(1, |u_i|)``; ``None`` uses ``eps^(1/3)``.

    Returns:
        :class:`ActiveSubspaceResult` with ``space == "standard-normal"``,
        the points ``u_m``, the values ``g(u_m)``, and the evaluation count.

    Raises:
        ValueError: For invalid settings, discrete inputs, a marginal
            density of zero at a sample point, or non-finite model output.
        KeyError: For an unknown QoI.
        AutodiffUnsupportedError: With ``gradient="autodiff"`` for a model
            that does not accept dual inputs or has dependent inputs.

    Equation:
        [EQ-SENS-ACTIVE-001], [EQ-SENS-ACTIVE-002], [EQ-SENS-ACTIVE-003],
        [EQ-SENS-ACTIVE-004], [EQ-AD-DUAL-001], [EQ-OPT-FD-001].

    References:
        [REF-CONSTANTINE-DOW-WANG-2014], [REF-GRIEWANK-WALTHER-2008, Chapter
        3], [REF-LIU-DERKIUREGHIAN-1986].

    """
    model.quantity(quantity)
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer of at least two")
    if gradient not in ("auto", "autodiff", "finite-difference"):
        raise ValueError("gradient must be 'auto', 'autodiff', or 'finite-difference'")
    if relative_step is not None and not (
        math.isfinite(relative_step) and relative_step > 0.0
    ):
        raise ValueError("relative_step must be positive")
    if model.inputs.has_discrete:
        raise ValueError(
            "active subspaces need continuous inputs; discrete inputs "
            f"{list(model.inputs.discrete_names)} have no derivative"
        )
    stream = resolve_stream(seed)
    dimension_d = model.dimension
    method = "finite-difference" if gradient == "finite-difference" else "autodiff"
    fallback: str | None = None
    if method == "autodiff" and not model.inputs.is_independent:
        if gradient == "autodiff":
            raise AutodiffUnsupportedError(
                "autodiff gradients in standard-normal space need independent inputs"
            )
        method = "finite-difference"
        fallback = "dependent inputs: the map from u to x is differenced with the model"
    generator = stream.child("active-subspace").generator()
    standard = model.standard_space_function(quantity)
    laws = tuple(item.distribution for item in model.inputs.inputs)
    before = model.evaluations
    points: list[Vector] = []
    values: list[float] = []
    gradients: list[Vector] = []
    for _ in range(size):
        u = tuple(generator.gauss(0.0, 1.0) for _ in range(dimension_d))
        if method == "autodiff":
            x = model.inputs.from_standard_normal(u)
            try:
                value, physical = _dual_value_and_gradient(model, quantity, x)
            except TypeError as error:
                if gradient == "autodiff":
                    if isinstance(error, AutodiffUnsupportedError):
                        raise
                    raise AutodiffUnsupportedError(
                        f"model does not support Dual numbers: {error}"
                    ) from error
                method = "finite-difference"
                fallback = f"{type(error).__name__}: {error}"
            else:
                scaled: list[float] = []
                for law, u_i, x_i, slope in zip(laws, u, x, physical, strict=True):
                    density = float(law.pdf(x_i))  # type: ignore[union-attr]
                    if not density > 0.0:
                        raise ValueError(
                            "a marginal density is zero at a sample point; use "
                            "gradient='finite-difference'"
                        )
                    jacobian = _INV_SQRT_2PI * math.exp(-0.5 * u_i * u_i) / density
                    scaled.append(slope * jacobian)
                grad = tuple(scaled)
        if method == "finite-difference":
            value = standard(u)
            raw, _ = central_gradient(standard, u, relative_step=relative_step)
            grad = tuple(raw)
        if not math.isfinite(value) or not all(math.isfinite(g) for g in grad):
            raise ValueError("model value or gradient is not finite")
        points.append(u)
        values.append(value)
        gradients.append(grad)
    result = active_subspace_from_gradients(
        gradients,
        names=model.inputs.names,
        dimension=dimension,
        bootstrap=bootstrap,
        confidence=confidence,
        seed=stream,
        points=points,
        values=values,
    )
    return ActiveSubspaceResult(
        result.input_names,
        result.eigenvalues,
        result.eigenvectors,
        result.matrix,
        result.gaps,
        result.dimension,
        result.dimension_rule,
        result.activity_scores,
        result.eigenvalue_intervals,
        result.subspace_errors,
        result.bootstrap,
        result.confidence,
        result.sample_size,
        model.evaluations - before,
        method,
        "standard-normal",
        result.points,
        result.values,
        fallback,
    )
