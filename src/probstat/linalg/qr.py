"""Householder QR factorization and QR-based least squares.

The thin factorization ``A = Q R`` of a tall matrix is the numerically
preferred route to linear least squares: it never forms ``A'A``, so the
condition number entering the solve is ``kappa(A)`` rather than
``kappa(A)^2``. Column pivoting is deliberately absent; a column that is
(numerically) a combination of earlier columns produces a tiny diagonal entry
of ``R``, which :func:`householder_qr` reports through its ``rank_deficient``
columns instead of silently regularizing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from operator import mul

from .dense import Matrix, Vector, as_matrix, as_vector, solve_upper

DEFAULT_RANK_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class HouseholderQR:
    """Immutable thin Householder factorization ``A = Q R`` of an ``n x p`` matrix.

    Attributes:
        q: ``n x p`` matrix with orthonormal columns.
        r: ``p x p`` upper-triangular factor.
        column_norms: Euclidean norms of the original columns.
        rank_deficient: Indices ``j`` with ``|R_jj| <= tol * ||a_j||``, i.e.
            columns numerically inside the span of the preceding columns.
        tolerance: Relative tolerance used for the rank decision.

    References:
        [REF-GOLUB-VANLOAN-2013, Sections 5.1-5.2], [REF-HOUSEHOLDER-1958].

    """

    q: Matrix
    r: Matrix
    column_norms: Vector
    rank_deficient: tuple[int, ...]
    tolerance: float

    @property
    def full_rank(self) -> bool:
        """Return whether every diagonal entry of ``R`` passed the rank test.

        Returns:
            ``True`` when ``rank_deficient`` is empty.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 5.4]. No equation is evaluated.

        """
        return not self.rank_deficient

    def solve(self, right_hand_side: Vector) -> Vector:
        """Return the least-squares solution ``x = R^-1 Q' b``.

        Args:
            right_hand_side: Vector ``b`` of length ``n``.

        Returns:
            Minimizer of ``||A x - b||_2``.

        Raises:
            ValueError: If the factorization is rank deficient or ``b`` has the
                wrong length.

        Equation:
            [EQ-LINALG-009], [EQ-LINALG-003].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 5.3].

        """
        if self.rank_deficient:
            raise ValueError("least-squares solve needs a full-rank factorization")
        target = as_vector(right_hand_side, name="right_hand_side")
        if len(target) != len(self.q):
            raise ValueError("right_hand_side length must equal the row count")
        width = len(self.r)
        try:
            q_columns = tuple(zip(*self.q, strict=True))
        except ValueError:
            q_columns = ()
        if len(q_columns) == width:
            projected = tuple(
                math.fsum(map(mul, column, target)) for column in q_columns
            )
        else:  # malformed hand-built factor: keep the reference behaviour
            projected = tuple(
                math.fsum(self.q[i][j] * target[i] for i in range(len(target)))
                for j in range(width)
            )
        return solve_upper(self.r, projected)

    def r_inverse(self) -> Matrix:
        """Return ``R^-1`` by back substitution against unit vectors.

        Returns:
            Upper-triangular inverse of ``R``; ``R^-1 R^-T = (A'A)^-1``.

        Raises:
            ValueError: If the factorization is rank deficient.

        Equation:
            [EQ-LINALG-003], [EQ-LINALG-009].

        References:
            [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 5.3].

        """
        if self.rank_deficient:
            raise ValueError("R is singular for a rank-deficient factorization")
        size = len(self.r)
        columns = [
            solve_upper(self.r, tuple(1.0 if i == j else 0.0 for i in range(size)))
            for j in range(size)
        ]
        return tuple(tuple(columns[j][i] for j in range(size)) for i in range(size))

    def gram_inverse(self) -> Matrix:
        """Return ``(A'A)^-1 = R^-1 R^-T`` without forming ``A'A``.

        Returns:
            Symmetric ``p x p`` inverse Gram matrix.

        Raises:
            ValueError: If the factorization is rank deficient.

        Equation:
            [EQ-LINALG-009].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 5.3].

        """
        inverse = self.r_inverse()
        size = len(inverse)
        # Entry (i, j) is a correctly rounded sum of the products
        # ``inverse[i][k] * inverse[j][k]``; the products commute exactly, so
        # entry (j, i) is the same float and only ``j >= i`` is formed.
        rows = [[0.0] * size for _ in range(size)]
        for i in range(size):
            row_i = inverse[i]
            for j in range(i, size):
                total = math.fsum(map(mul, row_i[j:], inverse[j][j:]))
                rows[i][j] = total
                rows[j][i] = total
        return tuple(tuple(row) for row in rows)

    def hat_diagonal(self) -> Vector:
        """Return the diagonal of the orthogonal projector ``Q Q'``.

        Returns:
            ``h_i = sum_j Q_ij^2``, the leverages of the rows of ``A``.

        Equation:
            [EQ-LINALG-009], [EQ-LM-DIAG-001].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 5.3],
            [REF-MONTGOMERY-PECK-VINING-2021, Chapter 6].

        """
        return tuple(math.fsum(map(mul, row, row)) for row in self.q)


def householder_qr(
    matrix: Matrix, *, rank_tolerance: float = DEFAULT_RANK_TOLERANCE
) -> HouseholderQR:
    """Return the thin Householder QR factorization of a tall matrix.

    Each step ``k`` reflects the trailing part of column ``k`` onto a multiple
    of the first unit vector with ``H_k = I - 2 v v'``, choosing the sign of
    the target ``-sign(x_1)||x||`` that avoids cancellation. The thin ``Q`` is
    ``H_0 H_1 ... H_(p-1)`` applied to the first ``p`` unit columns.

    Args:
        matrix: Finite ``n x p`` matrix with ``n >= p``.
        rank_tolerance: Relative threshold ``tol`` in ``|R_jj| <= tol ||a_j||``
            used to flag numerically dependent columns; must lie in ``(0, 1)``.

    Returns:
        :class:`HouseholderQR` with ``Q``, ``R``, and rank information.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix is empty, ragged, non-finite, wider than it is
            tall, or the tolerance is invalid.

    Equation:
        [EQ-LINALG-009].

    References:
        [REF-GOLUB-VANLOAN-2013, Sections 5.1-5.2], [REF-HOUSEHOLDER-1958].

    """
    checked = as_matrix(matrix, name="matrix")
    rows, columns = len(checked), len(checked[0])
    if rows < columns:
        raise ValueError("householder_qr needs at least as many rows as columns")
    if not 0.0 < rank_tolerance < 1.0:
        raise ValueError("rank_tolerance must lie strictly between zero and one")
    work = [list(column) for column in zip(*checked, strict=True)]
    column_norms = tuple(math.hypot(*column) for column in work)
    reflectors: list[list[float]] = []
    for k in range(columns):
        segment = work[k][k:]
        length = math.hypot(*segment)
        if length == 0.0:
            reflectors.append([0.0] * (rows - k))
            continue
        alpha = -length if segment[0] >= 0.0 else length
        vector = list(segment)
        vector[0] -= alpha
        vector_norm = math.hypot(*vector)
        if vector_norm == 0.0:
            reflectors.append([0.0] * (rows - k))
            continue
        vector = [value / vector_norm for value in vector]
        reflectors.append(vector)
        for j in range(k, columns):
            column = work[j]
            tail = column[k:]
            projection = 2.0 * math.fsum(map(mul, vector, tail))
            if projection != 0.0:
                column[k:] = [
                    c - projection * v for c, v in zip(tail, vector, strict=True)
                ]
    upper = tuple(
        tuple(work[j][i] if j >= i else 0.0 for j in range(columns))
        for i in range(columns)
    )
    finite_reflectors = all(all(map(math.isfinite, v)) for v in reflectors)
    q_columns: list[list[float]] = []
    for j in range(columns):
        column = [1.0 if i == j else 0.0 for i in range(rows)]
        # Reflectors ``k > j`` act on the all-zero tail ``column[k:]`` and
        # leave it unchanged (their projection is exactly zero), so they are
        # skipped; this is the structure of ``H_0 ... H_(p-1) e_j``.
        for k in range(j if finite_reflectors else columns - 1, -1, -1):
            vector = reflectors[k]
            tail = column[k:]
            projection = 2.0 * math.fsum(map(mul, vector, tail))
            if projection != 0.0:
                column[k:] = [
                    c - projection * v for c, v in zip(tail, vector, strict=True)
                ]
        q_columns.append(column)
    q = tuple(tuple(q_columns[j][i] for j in range(columns)) for i in range(rows))
    deficient = tuple(
        j
        for j in range(columns)
        if abs(upper[j][j]) <= rank_tolerance * column_norms[j]
    )
    return HouseholderQR(
        q=q,
        r=upper,
        column_norms=column_norms,
        rank_deficient=deficient,
        tolerance=rank_tolerance,
    )
