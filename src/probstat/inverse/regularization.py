"""Classical deterministic regularization of linear and nonlinear inverse problems.

Tikhonov regularization replaces ``A x = b`` by the minimization of
``||A x - b||^2 + lambda^2 ||L x||^2``; it is solved here as the least-squares
problem of the stacked matrix ``[A; lambda L]`` by Householder QR, never
through the normal equations. Truncated SVD keeps the ``k`` largest singular
triplets. Three rules choose ``lambda``: the L-curve corner (maximum
curvature of ``(log ||A x - b||, log ||L x||)``), generalized cross-validation,
and Morozov's discrepancy principle. For ``L = I`` the curves are evaluated
from one SVD of ``A`` in ``O(k)`` work per ``lambda``; a general ``L`` costs
one QR factorization of the stacked matrix per evaluation. Non-smooth
penalties (total variation, ``l1``) are out of scope.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    HouseholderQR,
    Matrix,
    SingularValueDecomposition,
    Vector,
    as_matrix,
    as_vector,
    householder_qr,
    matvec,
    singular_value_decomposition,
)
from probstat.optimization import LeastSquaresResult, brent_root, least_squares

_STACKED_RANK_TOLERANCE = 1e-14
_GRID_FLOOR = 16.0 * 2.0**-52
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def _norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm with a correctly rounded sum of squares.

    Args:
        vector: Finite vector.

    Returns:
        ``sqrt(sum_i v_i^2)``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.2]. No named equation is needed.

    """
    return math.sqrt(math.fsum(value * value for value in vector))


@dataclass(frozen=True, slots=True)
class TikhonovSolution:
    """Tikhonov-regularized solution of ``A x = b`` for one ``lambda``.

    Attributes:
        solution: Minimizer ``x_lambda`` of ``||A x - b||^2 + lambda^2 ||L x||^2``.
        residual_norm: ``||A x_lambda - b||``.
        solution_norm: ``||L x_lambda||`` (``||x_lambda||`` when ``L = I``).
        regularization: The parameter ``lambda``.
        filter_factors: ``sigma_i^2 / (sigma_i^2 + lambda^2)`` in the order of
            the descending singular values of ``A`` when ``L = I``; ``None``
            for a general ``L``.

    References:
        [REF-TIKHONOV-1963], [REF-HANSEN-2008, Sections 2.2, 2.4, and 2.7.1].

    """

    solution: Vector
    residual_norm: float
    solution_norm: float
    regularization: float
    filter_factors: Vector | None


@dataclass(frozen=True, slots=True)
class TruncatedSVDSolution:
    """Truncated-SVD solution of ``A x = b`` that keeps ``k`` singular triplets.

    Attributes:
        solution: ``x_k = sum_(i <= k) (u_i'b / sigma_i) v_i``.
        residual_norm: ``||A x_k - b||``.
        solution_norm: ``||x_k||``.
        rank: The truncation index ``k``.
        filter_factors: ``1`` for the kept and ``0`` for the discarded
            singular values.

    References:
        [REF-HANSEN-2008, Section 2.7.3].

    """

    solution: Vector
    residual_norm: float
    solution_norm: float
    rank: int
    filter_factors: Vector


@dataclass(frozen=True, slots=True)
class PicardCoefficients:
    """Quantities of the discrete Picard plot in descending singular-value order.

    Attributes:
        singular_values: ``sigma_i``.
        data_coefficients: ``|u_i'b|``.
        solution_coefficients: ``|u_i'b| / sigma_i``; ``inf`` where
            ``sigma_i = 0 < |u_i'b|`` and ``nan`` where both vanish.

    References:
        [REF-HANSEN-2008, Section 2.4].

    """

    singular_values: Vector
    data_coefficients: Vector
    solution_coefficients: Vector


@dataclass(frozen=True, slots=True)
class RegularizationParameterChoice:
    """A chosen Tikhonov parameter with the curve that selected it.

    Attributes:
        method: ``"l_curve"``, ``"generalized_cross_validation"``, or
            ``"discrepancy_principle"``.
        regularization: The chosen ``lambda``.
        grid: Increasing ``lambda`` values at which the curve was evaluated.
        criterion: The criterion on ``grid``: the L-curve curvature, the GCV
            function, or the residual norm minus the discrepancy target.
        residual_norms: ``||A x_lambda - b||`` on ``grid``.
        solution_norms: ``||L x_lambda||`` on ``grid``.
        solution: Tikhonov solution at the chosen ``lambda``.
        converged: Whether the refinement (golden-section search or Brent
            root) met its tolerance.
        at_boundary: Whether the grid optimum lies at an end of the grid, in
            which case the true optimum may lie outside it; always ``False``
            for the discrepancy principle.

    References:
        [REF-HANSEN-2008, Section 2.9].

    """

    method: str
    regularization: float
    grid: Vector
    criterion: Vector
    residual_norms: Vector
    solution_norms: Vector
    solution: TikhonovSolution
    converged: bool
    at_boundary: bool


@dataclass(frozen=True, slots=True)
class NonlinearTikhonovResult:
    """Nonlinear Tikhonov solution of ``min ||r(x)||^2 + lambda^2 ||L (x - x_bar)||^2``.

    Attributes:
        solution: The minimizer found by Levenberg-Marquardt.
        residual_norm: ``||r(x)||`` at the solution.
        penalty_norm: ``||L (x - x_bar)||`` at the solution.
        regularization: The parameter ``lambda``.
        prior_mean: The penalty center ``x_bar``.
        least_squares: The Levenberg-Marquardt record of the stacked problem.

    References:
        [REF-HANSEN-2008, Section 2.2], [REF-MORE-1978].

    """

    solution: Vector
    residual_norm: float
    penalty_norm: float
    regularization: float
    prior_mean: Vector
    least_squares: LeastSquaresResult

    @property
    def converged(self) -> bool:
        """Return whether Levenberg-Marquardt met a stopping tolerance.

        Returns:
            ``least_squares.converged``.

        References:
            [REF-MORE-1978]. No equation is evaluated.

        """
        return self.least_squares.converged


def _validate_problem(
    operator: Matrix, data: Sequence[float], penalty: Matrix | None
) -> tuple[Matrix, Vector, Matrix | None]:
    """Validate the operator, the data, and an optional penalty matrix.

    Args:
        operator: Candidate ``m x n`` matrix ``A``.
        data: Candidate data vector of length ``m``.
        penalty: Optional ``p x n`` matrix ``L``.

    Returns:
        The validated triple.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree or values are non-finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    matrix = as_matrix(operator, name="operator")
    vector = as_vector(data, name="data")
    if len(vector) != len(matrix):
        raise ValueError("data length must equal the number of operator rows")
    checked_penalty: Matrix | None = None
    if penalty is not None:
        checked_penalty = as_matrix(penalty, name="penalty")
        if len(checked_penalty[0]) != len(matrix[0]):
            raise ValueError("penalty must have as many columns as the operator")
    return matrix, vector, checked_penalty


def _require_regularization(value: float, *, positive: bool) -> float:
    """Validate a regularization parameter.

    Args:
        value: Candidate ``lambda``.
        positive: Require ``lambda > 0`` instead of ``lambda >= 0``.

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is non-finite or out of range.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    checked = require_finite_real(value, name="regularization")
    if checked < 0.0 or (positive and checked == 0.0):
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"regularization must be {bound}")
    return checked


def _decomposition_for(
    matrix: Matrix, decomposition: SingularValueDecomposition | None
) -> SingularValueDecomposition:
    """Return a supplied SVD after a shape check, or compute one.

    Args:
        matrix: The operator ``A``.
        decomposition: Optional precomputed SVD of ``A``.

    Returns:
        The SVD of ``A``.

    Raises:
        ValueError: If the supplied SVD has the wrong shape.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 8.6]. No equation is evaluated.

    """
    if decomposition is None:
        return singular_value_decomposition(matrix)
    if decomposition.shape != (len(matrix), len(matrix[0])):
        raise ValueError("decomposition shape does not match the operator")
    return decomposition


def _filter_factors(singular_values: Vector, regularization: float) -> Vector:
    """Return the Tikhonov filter factors ``sigma^2 / (sigma^2 + lambda^2)``.

    Args:
        singular_values: Non-negative singular values.
        regularization: ``lambda >= 0``.

    Returns:
        Filter factors in ``[0, 1]``; ``0`` for a zero singular value.

    Equation:
        [EQ-INV-REG-002].

    References:
        [REF-HANSEN-2008, Section 2.4].

    """
    squared = regularization * regularization
    return tuple(
        0.0 if sigma == 0.0 else sigma * sigma / (sigma * sigma + squared)
        for sigma in singular_values
    )


def _stack(matrix: Matrix, penalty: Matrix, regularization: float) -> Matrix:
    """Return the stacked matrix ``[A; lambda L]``.

    Args:
        matrix: Operator ``A``.
        penalty: Penalty matrix ``L``.
        regularization: ``lambda``.

    Returns:
        The ``(m + p) x n`` stacked matrix.

    Equation:
        [EQ-INV-REG-001].

    References:
        [REF-HANSEN-2008, Section 2.7.1].

    """
    return matrix + tuple(
        tuple(regularization * value for value in row) for row in penalty
    )


def _identity(size: int) -> Matrix:
    """Return the ``size x size`` identity matrix.

    Args:
        size: Dimension.

    Returns:
        Identity matrix.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    return tuple(
        tuple(1.0 if i == j else 0.0 for j in range(size)) for i in range(size)
    )


def _stacked_qr(
    matrix: Matrix, penalty: Matrix, regularization: float
) -> HouseholderQR:
    """Factor ``[A; lambda L]`` and refuse a numerically rank-deficient stack.

    Args:
        matrix: Operator ``A``.
        penalty: Penalty matrix ``L``.
        regularization: ``lambda``.

    Returns:
        The Householder factorization of the stacked matrix.

    Raises:
        ValueError: If the stacked matrix has fewer rows than columns or is
            rank deficient (the null spaces of ``A`` and ``L`` intersect, or
            ``lambda`` is too small to regularize).

    Equation:
        [EQ-INV-REG-001], [EQ-LINALG-009].

    References:
        [REF-HANSEN-2008, Section 2.7.1], [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    stacked = _stack(matrix, penalty, regularization)
    if len(stacked) < len(stacked[0]):
        raise ValueError("the stacked matrix [A; lambda L] has fewer rows than columns")
    factorization = householder_qr(stacked, rank_tolerance=_STACKED_RANK_TOLERANCE)
    if not factorization.full_rank:
        raise ValueError(
            "the stacked matrix [A; lambda L] is numerically rank deficient: "
            "the null spaces of A and L intersect or lambda is too small"
        )
    return factorization


def tikhonov(
    operator: Matrix,
    data: Sequence[float],
    regularization: float,
    *,
    penalty: Matrix | None = None,
    decomposition: SingularValueDecomposition | None = None,
) -> TikhonovSolution:
    """Return the Tikhonov solution by QR of the stacked matrix ``[A; lambda L]``.

    The minimizer of ``||A x - b||^2 + lambda^2 ||L x||^2`` is the
    least-squares solution of ``[A; lambda L] x = [b; 0]``, computed by
    Householder QR; the normal equations ``(A'A + lambda^2 L'L) x = A'b``
    are never formed. For ``L = I`` the filter factors need the singular
    values of ``A``: pass ``decomposition`` to reuse an SVD, otherwise one is
    computed (``O(m n^2)`` Jacobi work).

    Args:
        operator: Finite ``m x n`` matrix ``A``.
        data: Data vector ``b`` of length ``m``.
        regularization: ``lambda >= 0``; ``0`` gives ordinary least squares
            and then requires ``A`` to have full column rank.
        penalty: Optional ``p x n`` matrix ``L`` (default ``I``); the null
            spaces of ``A`` and ``L`` must intersect only in zero.
        decomposition: Optional SVD of ``A``, used for the filter factors
            when ``L = I``.

    Returns:
        :class:`TikhonovSolution`.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, ``lambda`` is invalid, or the stacked
            matrix is numerically rank deficient.

    Equation:
        [EQ-INV-REG-001], [EQ-INV-REG-002].

    References:
        [REF-TIKHONOV-1963], [REF-HANSEN-2008, Sections 2.2 and 2.7.1],
        [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    matrix, vector, checked_penalty = _validate_problem(operator, data, penalty)
    lam = _require_regularization(regularization, positive=False)
    n = len(matrix[0])
    factors: Vector | None = None
    if checked_penalty is None:
        svd = _decomposition_for(matrix, decomposition)
        factors = _filter_factors(svd.singular_values, lam)
        effective_penalty = _identity(n)
    else:
        effective_penalty = checked_penalty
    factorization = _stacked_qr(matrix, effective_penalty, lam)
    right = vector + (0.0,) * len(effective_penalty)
    solution = factorization.solve(right)
    fitted = matvec(matrix, solution)
    residual = tuple(f - b for f, b in zip(fitted, vector, strict=True))
    penalized = (
        solution if checked_penalty is None else matvec(checked_penalty, solution)
    )
    return TikhonovSolution(
        solution=solution,
        residual_norm=_norm(residual),
        solution_norm=_norm(penalized),
        regularization=lam,
        filter_factors=factors,
    )


def _projections(svd: SingularValueDecomposition, data: Vector) -> tuple[Vector, float]:
    """Return ``beta = U'b`` and ``||b - U U'b||^2``.

    Args:
        svd: SVD of ``A``.
        data: Data vector ``b``.

    Returns:
        The coefficients and the squared norm of the part of ``b`` outside
        the range of ``U``.

    References:
        [REF-HANSEN-2008, Section 2.4]. No named equation is needed.

    """
    k = len(svd.singular_values)
    beta = tuple(
        math.fsum(svd.u[i][j] * data[i] for i in range(len(data))) for j in range(k)
    )
    outside = tuple(
        data[i] - math.fsum(svd.u[i][j] * beta[j] for j in range(k))
        for i in range(len(data))
    )
    return beta, math.fsum(value * value for value in outside)


def truncated_svd(
    operator: Matrix,
    data: Sequence[float],
    rank: int,
    *,
    decomposition: SingularValueDecomposition | None = None,
) -> TruncatedSVDSolution:
    """Return the truncated-SVD solution keeping the ``rank`` largest triplets.

    Args:
        operator: Finite ``m x n`` matrix ``A``.
        data: Data vector ``b`` of length ``m``.
        rank: Truncation index ``1 <= k <= min(m, n)`` with ``sigma_k > 0``.
        decomposition: Optional precomputed SVD of ``A``.

    Returns:
        :class:`TruncatedSVDSolution`.

    Raises:
        TypeError: If ``rank`` is not an integer or an entry is not real.
        ValueError: If shapes disagree or ``rank`` is out of range or reaches
            a zero singular value.

    Equation:
        [EQ-INV-REG-006].

    References:
        [REF-HANSEN-2008, Section 2.7.3].

    """
    matrix, vector, _ = _validate_problem(operator, data, None)
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an integer")
    svd = _decomposition_for(matrix, decomposition)
    values = svd.singular_values
    if not 1 <= rank <= len(values):
        raise ValueError("rank must lie between one and min(m, n)")
    if values[rank - 1] == 0.0:
        raise ValueError("rank must not include a zero singular value")
    beta, outside = _projections(svd, vector)
    coefficients = [beta[i] / values[i] for i in range(rank)]
    n = len(matrix[0])
    solution = tuple(
        math.fsum(svd.v[row][i] * coefficients[i] for i in range(rank))
        for row in range(n)
    )
    residual_squared = math.fsum(
        [beta[i] * beta[i] for i in range(rank, len(values))] + [outside]
    )
    return TruncatedSVDSolution(
        solution=solution,
        residual_norm=math.sqrt(residual_squared),
        solution_norm=_norm(coefficients),
        rank=rank,
        filter_factors=tuple(1.0 if i < rank else 0.0 for i in range(len(values))),
    )


def picard_coefficients(
    operator: Matrix,
    data: Sequence[float],
    *,
    decomposition: SingularValueDecomposition | None = None,
) -> PicardCoefficients:
    """Return ``sigma_i``, ``|u_i'b|``, and ``|u_i'b| / sigma_i`` for a Picard plot.

    The discrete Picard condition holds when ``|u_i'b|`` decays on average
    faster than ``sigma_i``; with noisy data the coefficients level off at the
    noise level, and the index where ``|u_i'b| / sigma_i`` starts to grow marks
    where regularization must filter.

    Args:
        operator: Finite ``m x n`` matrix ``A``.
        data: Data vector ``b`` of length ``m``.
        decomposition: Optional precomputed SVD of ``A``.

    Returns:
        :class:`PicardCoefficients` in descending singular-value order.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree.

    Equation:
        [EQ-INV-REG-007].

    References:
        [REF-HANSEN-2008, Section 2.4].

    """
    matrix, vector, _ = _validate_problem(operator, data, None)
    svd = _decomposition_for(matrix, decomposition)
    beta, _ = _projections(svd, vector)
    magnitudes = tuple(abs(value) for value in beta)
    ratios: list[float] = []
    for sigma, magnitude in zip(svd.singular_values, magnitudes, strict=True):
        if sigma > 0.0:
            ratios.append(magnitude / sigma)
        else:
            ratios.append(math.inf if magnitude > 0.0 else math.nan)
    return PicardCoefficients(
        singular_values=svd.singular_values,
        data_coefficients=magnitudes,
        solution_coefficients=tuple(ratios),
    )


@dataclass(frozen=True, slots=True)
class _CurvePoint:
    """Squared norms, the derivative of ``eta``, and the GCV trace at one lambda.

    References:
        [REF-HANSEN-OLEARY-1993, Section 5.1], [REF-HANSEN-2008, Section 2.9].

    """

    rho: float
    eta: float
    eta_derivative: float
    trace: float


class _Curve:
    """Evaluate the Tikhonov curve for ``L = I`` (SVD) or a general ``L`` (QR).

    References:
        [REF-HANSEN-2008, Sections 2.4 and 2.9], [REF-HANSEN-OLEARY-1993].

    """

    def __init__(
        self,
        matrix: Matrix,
        data: Vector,
        penalty: Matrix | None,
        decomposition: SingularValueDecomposition | None,
    ) -> None:
        """Store the problem and precompute the SVD quantities.

        Args:
            matrix: Operator ``A``.
            data: Data ``b``.
            penalty: ``None`` for ``L = I``, otherwise ``L``.
            decomposition: Optional SVD of ``A``.

        References:
            [REF-HANSEN-2008, Section 2.4]. No equation is evaluated.

        """
        self.matrix = matrix
        self.data = data
        self.penalty = penalty
        self.rows = len(matrix)
        self.svd = _decomposition_for(matrix, decomposition)
        self.beta, self.outside = _projections(self.svd, data)
        self.evaluations = 0

    def default_grid(self, points: int) -> Vector:
        """Return a geometric grid spanning the relevant singular-value range.

        For ``L = I`` it runs from ``max(sigma_k, 16 eps sigma_1)`` to
        ``sigma_1``; for a general ``L`` both ends are divided by the largest
        and the smallest nonzero singular value of ``L``.

        Args:
            points: Number of grid points (at least three).

        Returns:
            Increasing grid of ``lambda`` values.

        Raises:
            ValueError: If ``A`` is the zero matrix.

        References:
            [REF-HANSEN-2008, Section 2.4]. No equation is evaluated.

        """
        values = self.svd.singular_values
        if values[0] == 0.0:
            raise ValueError("the operator is the zero matrix")
        lower = max(values[-1], _GRID_FLOOR * values[0])
        upper = values[0]
        if self.penalty is not None:
            penalty_values = singular_value_decomposition(self.penalty).singular_values
            positive = [value for value in penalty_values if value > 0.0]
            if not positive:
                raise ValueError("the penalty is the zero matrix")
            lower = max(lower, 1e-12 * values[0]) / positive[0]
            upper = upper / positive[-1]
        ratio = math.log(upper / lower) / (points - 1)
        return tuple(lower * math.exp(ratio * index) for index in range(points))

    def point(self, regularization: float) -> _CurvePoint:
        """Return ``rho = ||r||^2``, ``eta = ||L x||^2``, ``eta'``, and the trace.

        Args:
            regularization: ``lambda > 0``.

        Returns:
            The curve quantities; ``trace`` is ``trace(I - A A^#)``.

        Equation:
            [EQ-INV-REG-003], [EQ-INV-REG-004].

        References:
            [REF-HANSEN-2008, Section 2.9], [REF-HANSEN-OLEARY-1993, Section 5.1].

        """
        self.evaluations += 1
        lam = regularization
        if self.penalty is None:
            squared = lam * lam
            rho_terms = [self.outside]
            eta_terms: list[float] = []
            derivative_terms: list[float] = []
            factor_sum: list[float] = []
            for sigma, beta in zip(self.svd.singular_values, self.beta, strict=True):
                if sigma == 0.0:
                    rho_terms.append(beta * beta)
                    continue
                denominator = sigma * sigma + squared
                f = sigma * sigma / denominator
                complement = squared / denominator
                rho_terms.append((complement * beta) ** 2)
                coefficient = beta / sigma
                eta_terms.append((f * coefficient) ** 2)
                derivative_terms.append(f * f * complement * coefficient * coefficient)
                factor_sum.append(f)
            return _CurvePoint(
                rho=math.fsum(rho_terms),
                eta=math.fsum(eta_terms),
                eta_derivative=-4.0 / lam * math.fsum(derivative_terms),
                trace=self.rows - math.fsum(factor_sum),
            )
        factorization = _stacked_qr(self.matrix, self.penalty, lam)
        padding = (0.0,) * len(self.penalty)
        solution = factorization.solve(self.data + padding)
        residual = tuple(
            f - b for f, b in zip(matvec(self.matrix, solution), self.data, strict=True)
        )
        z = factorization.solve(residual + padding)
        lx = matvec(self.penalty, solution)
        lz = matvec(self.penalty, z)
        q_top = math.fsum(
            value * value for row in factorization.q[: self.rows] for value in row
        )
        return _CurvePoint(
            rho=math.fsum(value * value for value in residual),
            eta=math.fsum(value * value for value in lx),
            eta_derivative=4.0
            / lam
            * math.fsum(a * b for a, b in zip(lx, lz, strict=True)),
            trace=self.rows - q_top,
        )

    def limits(self) -> tuple[float, float] | None:
        """Return the residual norms in the limits ``lambda -> 0`` and ``-> inf``.

        Returns:
            ``(||b - U U'b|| over the nonzero sigma, ||b||)`` for ``L = I``;
            ``None`` for a general ``L``.

        References:
            [REF-HANSEN-2008, Section 2.9]. No named equation is needed.

        """
        if self.penalty is not None:
            return None
        lower = math.fsum(
            [self.outside]
            + [
                beta * beta
                for sigma, beta in zip(self.svd.singular_values, self.beta, strict=True)
                if sigma == 0.0
            ]
        )
        return math.sqrt(lower), _norm(self.data)


def _curvature(point: _CurvePoint, regularization: float) -> float:
    """Return the curvature of ``(log ||r||, log ||L x||)`` at one ``lambda``.

    With ``rho = ||r||^2``, ``eta = ||L x||^2``, and ``rho' = -lambda^2 eta'``,
    the curvature formula of Hansen and O'Leary for the log norms reduces to
    ``kappa = 2 rho eta (lambda^2 eta' rho + 2 lambda rho eta
    + lambda^4 eta eta') / (|eta'| (lambda^4 eta^2 + rho^2)^(3/2))``.

    Args:
        point: Curve quantities at ``lambda``.
        regularization: ``lambda > 0``.

    Returns:
        The signed curvature; positive at a corner turning from the vertical
        to the horizontal branch. ``nan`` when ``eta'``, ``rho``, or ``eta``
        vanishes.

    Equation:
        [EQ-INV-REG-004].

    References:
        [REF-HANSEN-OLEARY-1993, Section 5.1, eq. (8)].

    """
    rho, eta, derivative = point.rho, point.eta, point.eta_derivative
    if derivative == 0.0 or rho == 0.0 or eta == 0.0:
        return math.nan
    lam = regularization
    lam2 = lam * lam
    numerator = 2.0 * rho * eta * (lam2 * derivative * rho + 2.0 * lam * rho * eta)
    numerator += 2.0 * rho * eta * (lam2 * lam2 * eta * derivative)
    base = lam2 * lam2 * eta * eta + rho * rho
    denominator = abs(derivative) * base * math.sqrt(base)
    return numerator / denominator


def _gcv(point: _CurvePoint) -> float:
    """Return the GCV function ``||r||^2 / trace(I - A A^#)^2``.

    Args:
        point: Curve quantities at ``lambda``.

    Returns:
        The GCV value; ``inf`` when the trace vanishes.

    Equation:
        [EQ-INV-REG-003].

    References:
        [REF-CRAVEN-WAHBA-1979], [REF-GOLUB-HEATH-WAHBA-1979],
        [REF-HANSEN-2008, Section 2.9].

    """
    if point.trace <= 0.0:
        return math.inf
    return point.rho / (point.trace * point.trace)


def _golden_minimize(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> tuple[float, bool]:
    """Minimize a unimodal function on ``[lower, upper]`` by golden sections.

    Args:
        function: Scalar function of the log parameter.
        lower: Left end of the bracket.
        upper: Right end of the bracket.
        tolerance: Absolute bracket-width tolerance.
        max_iterations: Maximum number of section steps.

    Returns:
        The best point found and whether the tolerance was met.

    References:
        [REF-BRENT-1973, Chapter 5]. No named equation is needed.

    """
    a, b = lower, upper
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = function(c), function(d)
    for _ in range(max_iterations):
        if b - a <= tolerance:
            return (c if fc <= fd else d), True
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = function(d)
    return (c if fc <= fd else d), False


def _require_grid(grid: Sequence[float] | None, points: int, curve: _Curve) -> Vector:
    """Validate a user grid or build the default one.

    Args:
        grid: Optional increasing positive ``lambda`` values.
        points: Size of the default grid.
        curve: Curve whose singular values set the default range.

    Returns:
        Increasing grid with at least three points.

    Raises:
        TypeError: If ``points`` is not an integer.
        ValueError: If the grid is too short, not positive, or not
            increasing.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if grid is None:
        if isinstance(points, bool) or not isinstance(points, int):
            raise TypeError("points must be an integer")
        if points < 3:
            raise ValueError("points must be at least three")
        return curve.default_grid(points)
    values = as_vector(grid, name="grid")
    if len(values) < 3:
        raise ValueError("grid must contain at least three values")
    if values[0] <= 0.0 or any(
        b <= a for a, b in zip(values, values[1:], strict=False)
    ):
        raise ValueError("grid must be positive and strictly increasing")
    return values


def _choose_by_extremum(
    method: str,
    curve: _Curve,
    grid: Vector,
    score: Callable[[_CurvePoint, float], float],
) -> RegularizationParameterChoice:
    """Minimize ``score`` on the grid, refine in ``log lambda``, and solve.

    Args:
        method: Name recorded in the result.
        curve: Curve evaluator.
        grid: Increasing ``lambda`` values.
        score: Quantity to minimize (negated curvature or GCV).

    Returns:
        The parameter choice.

    Raises:
        ValueError: If the score is undefined on the whole grid.

    References:
        [REF-HANSEN-OLEARY-1993, Section 5.1], [REF-HANSEN-2008, Section 2.9].

    """
    points = [curve.point(lam) for lam in grid]
    scores = [score(point, lam) for point, lam in zip(points, grid, strict=True)]
    finite = [index for index, value in enumerate(scores) if math.isfinite(value)]
    if not finite:
        raise ValueError(f"the {method} criterion is undefined on the whole grid")
    best = min(finite, key=lambda index: scores[index])
    at_boundary = best in (0, len(grid) - 1)
    left = math.log(grid[max(best - 1, 0)])
    right = math.log(grid[min(best + 1, len(grid) - 1)])

    def objective(log_lambda: float) -> float:
        """Return the score at ``exp(log_lambda)`` with ``inf`` for ``nan``.

        References:
            [REF-HANSEN-OLEARY-1993, Section 5.1]. No other equation is used.

        """
        lam = math.exp(log_lambda)
        value = score(curve.point(lam), lam)
        return value if math.isfinite(value) else math.inf

    log_choice, converged = _golden_minimize(objective, left, right)
    choice = math.exp(log_choice)
    if objective(log_choice) > scores[best]:
        choice = grid[best]
    criterion = tuple(-value if method == "l_curve" else value for value in scores)
    return RegularizationParameterChoice(
        method=method,
        regularization=choice,
        grid=grid,
        criterion=criterion,
        residual_norms=tuple(math.sqrt(point.rho) for point in points),
        solution_norms=tuple(math.sqrt(point.eta) for point in points),
        solution=tikhonov(
            curve.matrix,
            curve.data,
            choice,
            penalty=curve.penalty,
            decomposition=curve.svd,
        ),
        converged=converged,
        at_boundary=at_boundary,
    )


def l_curve(
    operator: Matrix,
    data: Sequence[float],
    *,
    penalty: Matrix | None = None,
    grid: Sequence[float] | None = None,
    points: int = 100,
    decomposition: SingularValueDecomposition | None = None,
) -> RegularizationParameterChoice:
    """Choose ``lambda`` at the L-curve corner (maximum log-log curvature).

    The L-curve is ``(log ||A x_lambda - b||, log ||L x_lambda||)``. Its
    curvature is evaluated in closed form from ``rho = ||r||^2``,
    ``eta = ||L x||^2`` and ``eta' = (4 / lambda) (L x)'(L z)`` with
    ``z = (A'A + lambda^2 L'L)^-1 A'(A x - b)`` (from the SVD when
    ``L = I``, from the same QR factorization otherwise). The grid maximum is
    refined by a golden-section search in ``log lambda`` between its grid
    neighbors.

    Args:
        operator: Finite ``m x n`` matrix ``A``.
        data: Data vector ``b`` of length ``m``.
        penalty: Optional ``p x n`` matrix ``L`` (default ``I``).
        grid: Optional increasing positive ``lambda`` grid (at least three
            values); the default spans the singular values of ``A``.
        points: Size of the default grid.
        decomposition: Optional precomputed SVD of ``A``.

    Returns:
        :class:`RegularizationParameterChoice` whose ``criterion`` is the
        curvature on ``grid``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, the grid is invalid, or the curvature
            is undefined on the whole grid.

    Equation:
        [EQ-INV-REG-004].

    References:
        [REF-HANSEN-OLEARY-1993, Sections 4.2 and 5.1],
        [REF-HANSEN-2008, Sections 2.5 and 2.9].

    """
    matrix, vector, checked_penalty = _validate_problem(operator, data, penalty)
    curve = _Curve(matrix, vector, checked_penalty, decomposition)
    lambdas = _require_grid(grid, points, curve)
    return _choose_by_extremum(
        "l_curve", curve, lambdas, lambda point, lam: -_curvature(point, lam)
    )


def generalized_cross_validation(
    operator: Matrix,
    data: Sequence[float],
    *,
    penalty: Matrix | None = None,
    grid: Sequence[float] | None = None,
    points: int = 100,
    decomposition: SingularValueDecomposition | None = None,
) -> RegularizationParameterChoice:
    """Choose ``lambda`` by minimizing the generalized cross-validation function.

    ``G(lambda) = ||A x_lambda - b||^2 / trace(I - A A^#)^2`` with
    ``trace(I - A A^#) = m - sum_i f_i`` for ``L = I`` and
    ``m - ||Q_1||_F^2`` for a general ``L``, where ``Q_1`` holds the first
    ``m`` rows of the orthogonal factor of ``[A; lambda L]``. The grid
    minimum is refined by a golden-section search in ``log lambda``.

    Args:
        operator: Finite ``m x n`` matrix ``A``.
        data: Data vector ``b`` of length ``m``.
        penalty: Optional ``p x n`` matrix ``L`` (default ``I``).
        grid: Optional increasing positive ``lambda`` grid (at least three
            values); the default spans the singular values of ``A``.
        points: Size of the default grid.
        decomposition: Optional precomputed SVD of ``A``.

    Returns:
        :class:`RegularizationParameterChoice` whose ``criterion`` is ``G`` on
        ``grid``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree or the grid is invalid.

    Equation:
        [EQ-INV-REG-003].

    References:
        [REF-CRAVEN-WAHBA-1979], [REF-GOLUB-HEATH-WAHBA-1979],
        [REF-HANSEN-2008, Section 2.9].

    """
    matrix, vector, checked_penalty = _validate_problem(operator, data, penalty)
    curve = _Curve(matrix, vector, checked_penalty, decomposition)
    lambdas = _require_grid(grid, points, curve)
    return _choose_by_extremum(
        "generalized_cross_validation", curve, lambdas, lambda point, _: _gcv(point)
    )


def discrepancy_principle(
    operator: Matrix,
    data: Sequence[float],
    noise_norm: float,
    *,
    safety_factor: float = 1.0,
    penalty: Matrix | None = None,
    grid: Sequence[float] | None = None,
    points: int = 100,
    decomposition: SingularValueDecomposition | None = None,
) -> RegularizationParameterChoice:
    """Choose ``lambda`` with ``||A x_lambda - b|| = tau delta`` (Morozov).

    The residual norm increases with ``lambda``; the root of
    ``||r(lambda)|| - tau delta`` is bracketed on the grid (extended by
    decades when needed) and found by Brent's method in ``log lambda``.

    Args:
        operator: Finite ``m x n`` matrix ``A``.
        data: Data vector ``b`` of length ``m``.
        noise_norm: ``delta > 0``, an estimate of the noise norm ``||e||``
            (for independent noise of standard deviation ``s``,
            ``delta ~ s sqrt(m)``).
        safety_factor: ``tau >= 1``; values slightly above one guard against
            an underestimated ``delta``.
        penalty: Optional ``p x n`` matrix ``L`` (default ``I``).
        grid: Optional increasing positive ``lambda`` grid (at least three
            values); the default spans the singular values of ``A``.
        points: Size of the default grid.
        decomposition: Optional precomputed SVD of ``A``.

    Returns:
        :class:`RegularizationParameterChoice` whose ``criterion`` is
        ``||r(lambda)|| - tau delta`` on ``grid``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, ``delta`` or ``tau`` is invalid, or
            ``tau delta`` lies outside the range of attainable residual
            norms (no ``lambda`` satisfies the principle).

    Equation:
        [EQ-INV-REG-005].

    References:
        [REF-MOROZOV-1966], [REF-HANSEN-2008, Section 2.9],
        [REF-BRENT-1973, Chapter 4].

    """
    matrix, vector, checked_penalty = _validate_problem(operator, data, penalty)
    delta = require_finite_real(noise_norm, name="noise_norm")
    if delta <= 0.0:
        raise ValueError("noise_norm must be positive")
    tau = require_finite_real(safety_factor, name="safety_factor")
    if tau < 1.0:
        raise ValueError("safety_factor must be at least one")
    target = tau * delta
    curve = _Curve(matrix, vector, checked_penalty, decomposition)
    limits = curve.limits()
    if limits is not None:
        low, high = limits
        if not low < target < high:
            raise ValueError(
                "tau * noise_norm must lie strictly between the least-squares "
                f"residual norm {low:.6g} and the data norm {high:.6g}"
            )
    lambdas = list(_require_grid(grid, points, curve))

    def gap(lam: float) -> float:
        """Return ``||r(lambda)|| - tau delta``.

        References:
            [REF-MOROZOV-1966]. Equation [EQ-INV-REG-005].

        """
        return math.sqrt(curve.point(lam).rho) - target

    for _ in range(40):
        if gap(lambdas[0]) < 0.0:
            break
        lambdas.insert(0, lambdas[0] / 10.0)
    for _ in range(40):
        if gap(lambdas[-1]) > 0.0:
            break
        lambdas.append(lambdas[-1] * 10.0)
    points_on_grid = [curve.point(lam) for lam in lambdas]
    residuals = [math.sqrt(point.rho) for point in points_on_grid]
    crossing = next(
        (
            index
            for index in range(len(lambdas) - 1)
            if residuals[index] - target <= 0.0 <= residuals[index + 1] - target
        ),
        None,
    )
    if crossing is None:
        raise ValueError("no lambda on the extended grid attains tau * noise_norm")
    root = brent_root(
        lambda log_lambda: gap(math.exp(log_lambda)),
        math.log(lambdas[crossing]),
        math.log(lambdas[crossing + 1]),
        tolerance=1e-12,
    )
    choice = math.exp(root.root)
    return RegularizationParameterChoice(
        method="discrepancy_principle",
        regularization=choice,
        grid=tuple(lambdas),
        criterion=tuple(value - target for value in residuals),
        residual_norms=tuple(residuals),
        solution_norms=tuple(math.sqrt(point.eta) for point in points_on_grid),
        solution=tikhonov(
            matrix, vector, choice, penalty=checked_penalty, decomposition=curve.svd
        ),
        converged=root.converged,
        at_boundary=False,
    )


def nonlinear_tikhonov(
    residual: Callable[[Vector], Sequence[float]],
    x0: Sequence[float],
    regularization: float,
    *,
    prior_mean: Sequence[float] | None = None,
    penalty: Matrix | None = None,
    jacobian: Callable[[Vector], Sequence[Sequence[float]]] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> NonlinearTikhonovResult:
    """Minimize ``||r(x)||^2 + lambda^2 ||L (x - x_bar)||^2`` by Levenberg-Marquardt.

    The stacked residual ``[r(x); lambda L (x - x_bar)]`` is passed to
    :func:`probstat.optimization.least_squares`; with an analytic Jacobian
    ``J`` of ``r`` the stacked Jacobian is ``[J(x); lambda L]``. The result is
    a local minimizer from ``x0``.

    Args:
        residual: Vector residual ``r(x)`` with a fixed length.
        x0: Finite starting point of length ``n``.
        regularization: ``lambda >= 0``.
        prior_mean: Penalty center ``x_bar`` of length ``n``; defaults to
            ``x0``.
        penalty: Optional ``p x n`` matrix ``L`` (default ``I``).
        jacobian: Optional analytic Jacobian of ``r``; central differences
            otherwise.
        tolerance: Levenberg-Marquardt tolerance.
        max_iterations: Levenberg-Marquardt iteration limit.

    Returns:
        :class:`NonlinearTikhonovResult`; check ``converged``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, ``lambda`` is invalid, or the stacked
            problem has fewer residuals than parameters.

    Equation:
        [EQ-INV-REG-001], [EQ-OPT-LM-001].

    References:
        [REF-HANSEN-2008, Section 2.2], [REF-TIKHONOV-1963], [REF-MORE-1978].

    """
    start = as_vector(x0, name="x0")
    if not start:
        raise ValueError("x0 must be non-empty")
    n = len(start)
    lam = _require_regularization(regularization, positive=False)
    center = start if prior_mean is None else as_vector(prior_mean, name="prior_mean")
    if len(center) != n:
        raise ValueError("prior_mean length must equal the length of x0")
    if penalty is None:
        matrix = _identity(n)
    else:
        matrix = as_matrix(penalty, name="penalty")
        if len(matrix[0]) != n:
            raise ValueError("penalty must have as many columns as x0 has entries")

    def stacked(point: Vector) -> Vector:
        """Return ``[r(x); lambda L (x - x_bar)]``.

        References:
            [REF-HANSEN-2008, Section 2.2]. Equation [EQ-INV-REG-001].

        """
        shift = tuple(a - b for a, b in zip(point, center, strict=True))
        return tuple(residual(point)) + tuple(
            lam * value for value in matvec(matrix, shift)
        )

    stacked_jacobian: Callable[[Vector], Sequence[Sequence[float]]] | None = None
    if jacobian is not None:
        scaled = tuple(tuple(lam * value for value in row) for row in matrix)
        supplied = jacobian

        def stacked_jacobian(point: Vector) -> Sequence[Sequence[float]]:
            """Return ``[J(x); lambda L]``.

            References:
                [REF-NOCEDAL-WRIGHT-2006, Section 10.3]. Equation [EQ-INV-REG-001].

            """
            return tuple(tuple(row) for row in supplied(point)) + scaled

    fit = least_squares(
        stacked,
        start,
        jacobian=stacked_jacobian,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    data_part = as_vector(residual(fit.x), name="residual")
    shift = tuple(a - b for a, b in zip(fit.x, center, strict=True))
    return NonlinearTikhonovResult(
        solution=fit.x,
        residual_norm=_norm(data_part),
        penalty_norm=_norm(matvec(matrix, shift)),
        regularization=lam,
        prior_mean=center,
        least_squares=fit,
    )
