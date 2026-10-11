"""Dependency-free dense linear algebra on immutable tuple matrices.

Every routine favors explicit failure over silent regularization: a matrix that
is not positive definite, singular, or misshapen raises instead of receiving
hidden jitter. Accumulations use :func:`math.fsum` to limit cancellation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from probstat.foundations.validation import require_finite_real

Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]


def as_vector(values: Iterable[object], *, name: str = "vector") -> Vector:
    """Materialize and validate a finite real vector.

    Args:
        values: Candidate components.
        name: Field name used in indexed error messages.

    Returns:
        Tuple of finite floats; may be empty.

    Raises:
        TypeError: If ``values`` is not iterable or a component is not real.
        ValueError: If a component is non-finite.

    References:
        Project validation policy. No mathematical equation is evaluated.

    """
    try:
        materialized = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error
    return tuple(
        require_finite_real(value, name=f"{name}[{index}]")
        for index, value in enumerate(materialized)
    )


def as_matrix(
    values: Iterable[Iterable[object]],
    *,
    name: str = "matrix",
    square: bool = False,
) -> Matrix:
    """Materialize and validate a non-empty rectangular finite matrix.

    Args:
        values: Iterable of rows.
        name: Field name used in error messages.
        square: Require equal row and column counts when true.

    Returns:
        Tuple of equally long finite float rows.

    Raises:
        TypeError: If rows or entries have invalid types.
        ValueError: If the matrix is empty, ragged, non-finite, or not square
            when ``square`` is requested.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation is evaluated.

    """
    try:
        raw_rows = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of rows") from error
    rows = tuple(
        as_vector(row, name=f"{name}[{index}]") for index, row in enumerate(raw_rows)
    )
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be non-empty")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be rectangular")
    if square and len(rows) != len(rows[0]):
        raise ValueError(f"{name} must be square")
    return rows


def symmetrize(
    values: Iterable[Iterable[object]],
    *,
    name: str = "matrix",
    relative_tolerance: float = 1e-12,
) -> Matrix:
    """Validate approximate symmetry and return the averaged symmetric matrix.

    Args:
        values: Square finite matrix.
        name: Field name used in error messages.
        relative_tolerance: Allowed relative mirror mismatch.

    Returns:
        ``(A + A') / 2`` after confirming ``A`` is symmetric to tolerance.

    Raises:
        ValueError: If the matrix is not square or not symmetric.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    matrix = as_matrix(values, name=name, square=True)
    rows: list[Vector] = []
    for row_index, row in enumerate(matrix):
        result: list[float] = []
        for column_index, value in enumerate(row):
            mirror = matrix[column_index][row_index]
            tolerance = relative_tolerance * max(1.0, abs(value), abs(mirror))
            if abs(value - mirror) > tolerance:
                raise ValueError(f"{name} must be symmetric")
            result.append(0.5 * (value + mirror))
        rows.append(tuple(result))
    return tuple(rows)


def identity(size: int) -> Matrix:
    """Return the ``size`` by ``size`` identity matrix.

    Args:
        size: Positive dimension.

    Returns:
        Identity matrix.

    Raises:
        ValueError: If ``size`` is not positive.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation is evaluated.

    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size))
        for row in range(size)
    )


def transpose(matrix: Matrix) -> Matrix:
    """Return the transpose of a rectangular matrix.

    Args:
        matrix: Rectangular matrix.

    Returns:
        Matrix with rows and columns exchanged.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation is evaluated.

    """
    return tuple(zip(*matrix, strict=True))


def dot(left: Vector, right: Vector) -> float:
    """Return the compensated Euclidean inner product.

    Args:
        left: Finite vector.
        right: Finite vector of equal length.

    Returns:
        ``sum_i left_i right_i``.

    Raises:
        ValueError: If lengths differ.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def norm(vector: Vector) -> float:
    """Return the Euclidean norm without intermediate overflow.

    Args:
        vector: Finite vector.

    Returns:
        ``||vector||_2``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.2]. No named equation is needed.

    """
    return math.hypot(*vector)


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    """Multiply a matrix by a compatible vector.

    Args:
        matrix: Rectangular matrix with ``n`` columns.
        vector: Vector of length ``n``.

    Returns:
        Product ``A x``.

    Raises:
        ValueError: If dimensions disagree.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return tuple(dot(row, vector) for row in matrix)


def matmul(left: Matrix, right: Matrix) -> Matrix:
    """Multiply two compatible matrices.

    Args:
        left: Matrix with ``k`` columns.
        right: Matrix with ``k`` rows.

    Returns:
        Product ``A B``.

    Raises:
        ValueError: If inner dimensions disagree.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    if len(left[0]) != len(right):
        raise ValueError("inner matrix dimensions must agree")
    columns = transpose(right)
    return tuple(tuple(dot(row, column) for column in columns) for row in left)


def outer(left: Vector, right: Vector) -> Matrix:
    """Return the outer product ``x y'``.

    Args:
        left: Column factor.
        right: Row factor.

    Returns:
        Rank-one matrix.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return tuple(tuple(a * b for b in right) for a in left)


def add(left: Matrix, right: Matrix, *, scale: float = 1.0) -> Matrix:
    """Return ``left + scale * right`` for equally shaped matrices.

    Args:
        left: Base matrix.
        right: Matrix with the same shape.
        scale: Finite multiplier applied to ``right``.

    Returns:
        Entrywise combination.

    Raises:
        ValueError: If shapes disagree.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    if len(left) != len(right) or any(
        len(a) != len(b) for a, b in zip(left, right, strict=True)
    ):
        raise ValueError("matrix shapes must agree")
    return tuple(
        tuple(a + scale * b for a, b in zip(row_a, row_b, strict=True))
        for row_a, row_b in zip(left, right, strict=True)
    )


def scale_matrix(matrix: Matrix, factor: float) -> Matrix:
    """Multiply every matrix entry by a finite scalar.

    Args:
        matrix: Rectangular matrix.
        factor: Finite multiplier.

    Returns:
        Scaled matrix.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return tuple(tuple(factor * value for value in row) for row in matrix)


def sandwich(design: Matrix, covariance: Matrix) -> Matrix:
    """Return the congruence ``H S H'`` used for linear covariance propagation.

    Args:
        design: ``m`` by ``n`` linear map.
        covariance: ``n`` by ``n`` symmetric matrix.

    Returns:
        Symmetrized ``m`` by ``m`` matrix ``H S H'``.

    Equation:
        [EQ-LINALG-007].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1], [REF-BISHOP-2006, Section 3.3].

    """
    product = matmul(matmul(design, covariance), transpose(design))
    size = len(product)
    return tuple(
        tuple(0.5 * (product[i][j] + product[j][i]) for j in range(size))
        for i in range(size)
    )


def cholesky(matrix: Matrix) -> Matrix:
    """Return the lower Cholesky factor of a positive-definite matrix.

    Args:
        matrix: Finite symmetric square matrix.

    Returns:
        Lower triangular ``L`` with positive diagonal and ``A = L L'``.

    Raises:
        ValueError: If the matrix is not positive definite.

    Equation:
        [EQ-LINALG-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            remainder = matrix[row][column] - math.fsum(
                lower[row][inner] * lower[column][inner] for inner in range(column)
            )
            if row == column:
                if remainder <= 0.0 or not math.isfinite(remainder):
                    raise ValueError("matrix must be positive definite")
                lower[row][column] = math.sqrt(remainder)
            else:
                lower[row][column] = remainder / lower[column][column]
    return tuple(tuple(row) for row in lower)


def cholesky_semidefinite(matrix: Matrix, *, tolerance: float = 1e-12) -> Matrix:
    """Return a lower square root of a positive-semidefinite matrix.

    Pivots smaller than ``tolerance`` times the largest entry are treated as
    exact zeros; a materially negative pivot is rejected.

    Args:
        matrix: Finite symmetric square matrix.
        tolerance: Relative pivot tolerance.

    Returns:
        Lower triangular ``L`` with ``A = L L'`` to rounding.

    Raises:
        ValueError: If the matrix is not positive semidefinite.

    Equation:
        [EQ-LINALG-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    size = len(matrix)
    largest = max(max(abs(value) for value in row) for row in matrix)
    threshold = tolerance * max(1.0, largest)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            remainder = matrix[row][column] - math.fsum(
                lower[row][inner] * lower[column][inner] for inner in range(column)
            )
            if row == column:
                if remainder < -threshold:
                    raise ValueError("matrix must be positive semidefinite")
                lower[row][column] = math.sqrt(max(0.0, remainder))
            elif lower[column][column] > threshold:
                lower[row][column] = remainder / lower[column][column]
            elif abs(remainder) > threshold:
                raise ValueError("matrix must be positive semidefinite")
    return tuple(tuple(row) for row in lower)


def solve_lower(lower: Matrix, right_hand_side: Vector) -> Vector:
    """Solve ``L x = b`` by forward substitution.

    Args:
        lower: Non-singular lower-triangular matrix.
        right_hand_side: Vector of matching length.

    Returns:
        Solution vector.

    Raises:
        ValueError: If a diagonal entry is zero.

    Equation:
        [EQ-LINALG-002].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    solution: list[float] = []
    for row, target in enumerate(right_hand_side):
        pivot = lower[row][row]
        if pivot == 0.0:
            raise ValueError("triangular matrix is singular")
        residual = target - math.fsum(
            lower[row][column] * solution[column] for column in range(row)
        )
        solution.append(residual / pivot)
    return tuple(solution)


def solve_upper(upper: Matrix, right_hand_side: Vector) -> Vector:
    """Solve ``U x = b`` by back substitution.

    Args:
        upper: Non-singular upper-triangular matrix.
        right_hand_side: Vector of matching length.

    Returns:
        Solution vector.

    Raises:
        ValueError: If a diagonal entry is zero.

    Equation:
        [EQ-LINALG-003].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    size = len(right_hand_side)
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        pivot = upper[row][row]
        if pivot == 0.0:
            raise ValueError("triangular matrix is singular")
        residual = right_hand_side[row] - math.fsum(
            upper[row][column] * solution[column] for column in range(row + 1, size)
        )
        solution[row] = residual / pivot
    return tuple(solution)


def cholesky_solve(lower: Matrix, right_hand_side: Vector) -> Vector:
    """Solve ``L L' x = b`` from a precomputed Cholesky factor.

    Args:
        lower: Lower Cholesky factor.
        right_hand_side: Vector of matching length.

    Returns:
        Solution vector.

    Equation:
        [EQ-LINALG-002], [EQ-LINALG-003].

    References:
        [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

    """
    return solve_upper(transpose(lower), solve_lower(lower, right_hand_side))


def solve_spd(matrix: Matrix, right_hand_side: Vector) -> Vector:
    """Solve a symmetric positive-definite system through Cholesky factors.

    Args:
        matrix: Symmetric positive-definite matrix.
        right_hand_side: Vector of matching length.

    Returns:
        Solution of ``A x = b``.

    Raises:
        ValueError: If the matrix is not positive definite.

    Equation:
        [EQ-LINALG-001], [EQ-LINALG-002], [EQ-LINALG-003].

    References:
        [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

    """
    return cholesky_solve(cholesky(matrix), right_hand_side)


def inverse_spd(matrix: Matrix) -> Matrix:
    """Invert a symmetric positive-definite matrix by Cholesky solves.

    Args:
        matrix: Symmetric positive-definite matrix.

    Returns:
        Symmetrized inverse matrix.

    Raises:
        ValueError: If the matrix is not positive definite.

    Equation:
        [EQ-LINALG-001], [EQ-LINALG-002], [EQ-LINALG-003].

    References:
        [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

    """
    lower = cholesky(matrix)
    size = len(matrix)
    columns = tuple(
        cholesky_solve(lower, tuple(1.0 if row == j else 0.0 for row in range(size)))
        for j in range(size)
    )
    return tuple(
        tuple(0.5 * (columns[j][i] + columns[i][j]) for j in range(size))
        for i in range(size)
    )


def log_det_spd(matrix: Matrix) -> float:
    """Return ``log det A`` for a symmetric positive-definite matrix.

    Args:
        matrix: Symmetric positive-definite matrix.

    Returns:
        Natural log-determinant ``2 sum log L_ii``.

    Raises:
        ValueError: If the matrix is not positive definite.

    Equation:
        [EQ-LINALG-006].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    lower = cholesky(matrix)
    return 2.0 * math.fsum(math.log(lower[i][i]) for i in range(len(lower)))


def lu_solve(matrix: Matrix, right_hand_side: Vector) -> Vector:
    """Solve a general square system by Gaussian elimination with pivoting.

    Args:
        matrix: Square finite matrix.
        right_hand_side: Vector of matching length.

    Returns:
        Solution of ``A x = b``.

    Raises:
        ValueError: If shapes disagree or the matrix is numerically singular.

    Equation:
        [EQ-LINALG-004].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.4].

    """
    size = len(matrix)
    if len(right_hand_side) != size or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be square and match the right-hand side")
    work = [
        list(row) + [value] for row, value in zip(matrix, right_hand_side, strict=True)
    ]
    largest = max(max(abs(v) for v in row) for row in matrix)
    threshold = 1e-14 * max(1.0, largest)
    for column in range(size):
        pivot_row = max(range(column, size), key=lambda r: abs(work[r][column]))
        if abs(work[pivot_row][column]) <= threshold:
            raise ValueError("matrix is singular to working precision")
        work[column], work[pivot_row] = work[pivot_row], work[column]
        pivot = work[column][column]
        for row in range(column + 1, size):
            factor = work[row][column] / pivot
            if factor != 0.0:
                for inner in range(column, size + 1):
                    work[row][inner] -= factor * work[column][inner]
    upper = tuple(tuple(row[:size]) for row in work)
    return solve_upper(upper, tuple(row[size] for row in work))


def symmetric_eigen(
    matrix: Matrix, *, tolerance: float = 1e-14, max_sweeps: int = 100
) -> tuple[Vector, Matrix]:
    """Diagonalize a real symmetric matrix by cyclic Jacobi rotations.

    Args:
        matrix: Finite symmetric square matrix.
        tolerance: Relative off-diagonal Frobenius stopping tolerance.
        max_sweeps: Maximum number of full rotation sweeps.

    Returns:
        Eigenvalues in ascending order and a matrix whose COLUMNS are the
        corresponding orthonormal eigenvectors.

    Raises:
        ValueError: If convergence is not reached within ``max_sweeps``.

    Equation:
        [EQ-LINALG-005].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 8.5].

    """
    size = len(matrix)
    a = [list(row) for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    scale = math.sqrt(math.fsum(value * value for row in a for value in row))
    for _ in range(max_sweeps):
        off = math.sqrt(
            math.fsum(a[i][j] ** 2 for i in range(size) for j in range(size) if i != j)
        )
        if off <= tolerance * max(scale, 1e-300):
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                if a[p][q] == 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0)
                )
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(size):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(size):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(size):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    else:
        raise ValueError("Jacobi eigenvalue iteration did not converge")
    order = sorted(range(size), key=lambda i: a[i][i])
    values = tuple(a[i][i] for i in order)
    vectors = tuple(tuple(v[row][i] for i in order) for row in range(size))
    return values, vectors


def quadratic_form(matrix: Matrix, vector: Vector) -> float:
    """Return ``x' A x``.

    Args:
        matrix: Square matrix.
        vector: Vector of matching length.

    Returns:
        Scalar quadratic form.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return dot(vector, matvec(matrix, vector))
