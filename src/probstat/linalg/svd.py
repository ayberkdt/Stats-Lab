"""Singular value decomposition by one-sided Jacobi rotations.

The thin decomposition ``A = U diag(sigma) V'`` of an ``m x n`` matrix is
computed by Hestenes' one-sided Jacobi method: plane rotations are applied to
pairs of columns of a working copy of ``A`` (accumulated in ``V``) until every
pair of columns is orthogonal to a relative tolerance. The column norms are
then the singular values and the normalized columns the left singular
vectors. A tall matrix is first reduced to its triangular QR factor, so the
rotations act on an ``n x n`` matrix; a wide matrix is handled through its
transpose. Like :func:`~probstat.linalg.symmetric_eigen`, the routine is a
direct pure-Python kernel and is not part of the
:class:`~probstat.linalg.DenseBackend` protocol (ADR-0110).

The decomposition never regularizes: exactly zero singular values are kept
as zeros (their left singular vectors complete an orthonormal set), and
rank decisions are made only by :func:`numerical_rank` and
:func:`pseudo_inverse` against an explicit tolerance.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from operator import mul, truediv

from probstat.foundations.validation import require_finite_real

from .dense import Matrix, Vector, as_matrix
from .qr import householder_qr

DEFAULT_JACOBI_TOLERANCE = 1e-15
_MACHINE_EPSILON = 2.0**-52


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a correctly rounded inner product of two equal-length sequences.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        ``sum_i left_i right_i`` accumulated by :func:`math.fsum`.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return math.fsum(map(operator.mul, left, right))


@dataclass(frozen=True, slots=True)
class SingularValueDecomposition:
    """Immutable thin singular value decomposition ``A = U diag(sigma) V'``.

    Attributes:
        u: ``m x k`` matrix with orthonormal columns (left singular vectors),
            ``k = min(m, n)``.
        singular_values: ``k`` non-negative singular values in descending
            order.
        v: ``n x k`` matrix with orthonormal columns (right singular vectors).
        sweeps: Number of Jacobi sweeps performed.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 8.6], [REF-HESTENES-1958].

    """

    u: Matrix
    singular_values: Vector
    v: Matrix
    sweeps: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape ``(m, n)`` of the decomposed matrix.

        Returns:
            Row and column counts of ``A``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 8.6]. No equation is evaluated.

        """
        return len(self.u), len(self.v)

    def default_tolerance(self) -> float:
        """Return the default relative rank tolerance ``max(m, n) eps``.

        Returns:
            ``max(m, n)`` times the double-precision unit roundoff ``2^-52``.

        Equation:
            [EQ-LINALG-SVD-002].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 5.5].

        """
        return max(self.shape) * _MACHINE_EPSILON

    def rank(self, *, tolerance: float | None = None) -> int:
        """Return the number of singular values above ``tolerance * sigma_1``.

        Args:
            tolerance: Relative threshold in ``[0, 1)``; ``None`` uses
                :meth:`default_tolerance`.

        Returns:
            Numerical rank; ``0`` for the zero matrix.

        Raises:
            ValueError: If the tolerance is negative, non-finite, or not
                below one.

        Equation:
            [EQ-LINALG-SVD-002].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 5.5].

        """
        threshold = self._threshold(tolerance)
        return sum(1 for value in self.singular_values if value > threshold)

    @property
    def condition_number(self) -> float:
        """Return the 2-norm condition number ``sigma_1 / sigma_k``.

        Returns:
            ``sigma_1 / sigma_k``; ``inf`` when ``sigma_k = 0`` (including the
            zero matrix).

        Equation:
            [EQ-LINALG-SVD-001].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 2.6].

        """
        smallest = self.singular_values[-1]
        if smallest == 0.0:
            return math.inf
        return self.singular_values[0] / smallest

    def pseudo_inverse(self, *, rcond: float | None = None) -> Matrix:
        """Return the truncated Moore-Penrose pseudo-inverse ``V S^+ U'``.

        Args:
            rcond: Relative cut-off in ``[0, 1)``: singular values not above
                ``rcond * sigma_1`` are treated as zero. ``None`` uses
                :meth:`default_tolerance`.

        Returns:
            ``n x m`` matrix ``sum_(sigma_i > rcond sigma_1) v_i u_i' / sigma_i``.

        Raises:
            ValueError: If ``rcond`` is negative, non-finite, or not below
                one.

        Equation:
            [EQ-LINALG-SVD-002].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 5.5], [REF-PENROSE-1955].

        """
        threshold = self._threshold(rcond)
        kept = [
            index
            for index, value in enumerate(self.singular_values)
            if value > threshold
        ]
        rows, columns = self.shape
        kept_values = [self.singular_values[index] for index in kept]
        u_kept = [[self.u[j][index] for index in kept] for j in range(rows)]
        result: list[Vector] = []
        for i in range(columns):
            v_kept = [self.v[i][index] for index in kept]
            result.append(
                tuple(
                    math.fsum(map(truediv, map(mul, v_kept, u_row), kept_values))
                    for u_row in u_kept
                )
            )
        return tuple(result)

    def reconstruct(self) -> Matrix:
        """Return ``U diag(sigma) V'``.

        Returns:
            The ``m x n`` product of the factors.

        Equation:
            [EQ-LINALG-SVD-001].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 8.6].

        """
        rows, columns = self.shape
        values = self.singular_values
        k = len(values)
        u_rows = [self.u[i][:k] for i in range(rows)]
        v_rows = [self.v[j][:k] for j in range(columns)]
        if any(len(row) != k for row in u_rows) or any(len(row) != k for row in v_rows):
            # Malformed hand-built factors: keep the reference IndexError.
            return tuple(
                tuple(
                    math.fsum(
                        self.u[i][index] * values[index] * self.v[j][index]
                        for index in range(k)
                    )
                    for j in range(columns)
                )
                for i in range(rows)
            )
        # ``(u_ik s_k) v_jk``: the left-to-right product of the reference.
        return tuple(
            tuple(math.fsum(map(mul, scaled, v_row)) for v_row in v_rows)
            for scaled in (list(map(mul, u_row, values)) for u_row in u_rows)
        )

    def _threshold(self, tolerance: float | None) -> float:
        """Return the absolute singular-value threshold for a relative tolerance.

        Args:
            tolerance: Relative tolerance or ``None`` for the default.

        Returns:
            ``tolerance * sigma_1``.

        Raises:
            ValueError: If the tolerance is outside ``[0, 1)``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 5.5]. No equation is evaluated.

        """
        if tolerance is None:
            relative = self.default_tolerance()
        else:
            relative = require_finite_real(tolerance, name="tolerance")
            if not 0.0 <= relative < 1.0:
                raise ValueError("tolerance must lie in [0, 1)")
        return relative * self.singular_values[0]


def _complete_orthonormal(
    columns: list[list[float]], missing: list[int], size: int
) -> None:
    """Replace the listed columns by unit vectors orthogonal to the others.

    Candidates are the coordinate vectors, orthogonalized twice by modified
    Gram-Schmidt against the columns already in the basis; the candidate with
    the largest remaining norm is taken.

    Args:
        columns: Column vectors of length ``size``, modified in place.
        missing: Indices of the columns to replace.
        size: Vector length.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 5.2]. No named equation is needed.

    """
    basis = [columns[j] for j in range(len(columns)) if j not in set(missing)]
    for index in missing:
        best: list[float] | None = None
        best_norm = -1.0
        for unit in range(size):
            candidate = [1.0 if i == unit else 0.0 for i in range(size)]
            for _ in range(2):
                for vector in basis:
                    projection = _dot(vector, candidate)
                    candidate = [
                        c - projection * b
                        for c, b in zip(candidate, vector, strict=True)
                    ]
            length = math.sqrt(_dot(candidate, candidate))
            if length > best_norm:
                best, best_norm = candidate, length
        assert best is not None
        normalized = [value / best_norm for value in best]
        columns[index] = normalized
        basis.append(normalized)


def _one_sided_jacobi(
    columns: list[list[float]], tolerance: float, max_sweeps: int
) -> tuple[list[list[float]], int]:
    """Orthogonalize the columns of a square-or-tall matrix by Jacobi rotations.

    Args:
        columns: The ``n`` columns (length ``m >= n``), modified in place.
        tolerance: Relative orthogonality tolerance.
        max_sweeps: Maximum number of cyclic sweeps.

    Returns:
        The accumulated rotation matrix ``V`` as a list of its columns, and the
        number of sweeps performed.

    Raises:
        ValueError: If the sweeps do not converge.

    Equation:
        [EQ-LINALG-SVD-001].

    References:
        [REF-HESTENES-1958], [REF-GOLUB-VANLOAN-2013, Sections 8.5-8.6].

    """
    # Every column has the same length (built so by ``_tall_svd``), so the
    # rotations below zip without the ``strict`` length check.
    size = len(columns)
    v = [[1.0 if i == j else 0.0 for i in range(size)] for j in range(size)]
    fsum = math.fsum
    norms = [fsum(map(mul, column, column)) for column in columns]
    for sweep in range(1, max_sweeps + 1):
        rotated = False
        for p in range(size - 1):
            for q in range(p + 1, size):
                alpha, beta = norms[p], norms[q]
                if alpha == 0.0 or beta == 0.0:
                    continue
                gamma = fsum(map(mul, columns[p], columns[q]))
                if abs(gamma) <= tolerance * math.sqrt(alpha) * math.sqrt(beta):
                    continue
                rotated = True
                zeta = (beta - alpha) / (2.0 * gamma)
                t = math.copysign(1.0, zeta) / (abs(zeta) + math.hypot(1.0, zeta))
                c = 1.0 / math.hypot(1.0, t)
                s = c * t
                cp, cq = columns[p], columns[q]
                columns[p] = [c * a - s * b for a, b in zip(cp, cq, strict=False)]
                columns[q] = [s * a + c * b for a, b in zip(cp, cq, strict=False)]
                vp, vq = v[p], v[q]
                v[p] = [c * a - s * b for a, b in zip(vp, vq, strict=False)]
                v[q] = [s * a + c * b for a, b in zip(vp, vq, strict=False)]
                columns_p, columns_q = columns[p], columns[q]
                norms[p] = fsum(map(mul, columns_p, columns_p))
                norms[q] = fsum(map(mul, columns_q, columns_q))
        if not rotated:
            return v, sweep
    raise ValueError("one-sided Jacobi SVD did not converge")


def _tall_svd(
    matrix: Matrix, tolerance: float, max_sweeps: int
) -> SingularValueDecomposition:
    """Return the thin SVD of an ``m x n`` matrix with ``m >= n``.

    Args:
        matrix: Validated matrix with at least as many rows as columns.
        tolerance: Relative Jacobi orthogonality tolerance.
        max_sweeps: Maximum number of sweeps.

    Returns:
        The decomposition with ``k = n``.

    Equation:
        [EQ-LINALG-SVD-001].

    References:
        [REF-HESTENES-1958], [REF-GOLUB-VANLOAN-2013, Sections 5.2 and 8.6].

    """
    rows, size = len(matrix), len(matrix[0])
    q: Matrix | None = None
    if rows > size:
        factorization = householder_qr(matrix)
        q = factorization.q
        work = [[factorization.r[i][j] for i in range(size)] for j in range(size)]
    else:
        work = [[matrix[i][j] for i in range(rows)] for j in range(size)]
    v_columns, sweeps = _one_sided_jacobi(work, tolerance, max_sweeps)
    values = [math.sqrt(_dot(column, column)) for column in work]
    order = sorted(range(size), key=lambda j: -values[j])
    values = [values[j] for j in order]
    work = [work[j] for j in order]
    v_columns = [v_columns[j] for j in order]
    zero = [j for j in range(size) if values[j] == 0.0]
    for j in range(size):
        if values[j] > 0.0:
            work[j] = [value / values[j] for value in work[j]]
    if zero:
        _complete_orthonormal(work, zero, len(work[0]))
    if q is not None:
        u_columns = [[_dot(q[i], column) for i in range(rows)] for column in work]
    else:
        u_columns = work
    u = tuple(tuple(u_columns[j][i] for j in range(size)) for i in range(rows))
    v = tuple(tuple(v_columns[j][i] for j in range(size)) for i in range(size))
    return SingularValueDecomposition(
        u=u, singular_values=tuple(values), v=v, sweeps=sweeps
    )


def singular_value_decomposition(
    matrix: Matrix,
    *,
    tolerance: float = DEFAULT_JACOBI_TOLERANCE,
    max_sweeps: int = 60,
) -> SingularValueDecomposition:
    """Return the thin SVD ``A = U diag(sigma) V'`` by one-sided Jacobi.

    A tall matrix is reduced to ``R`` of its Householder QR factorization,
    and the rotations ``[a_p, a_q] <- [a_p, a_q] [[c, s], [-s, c]]`` with
    ``zeta = (||a_q||^2 - ||a_p||^2) / (2 a_p'a_q)``,
    ``t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2))``, ``c = 1/sqrt(1 + t^2)``,
    ``s = c t`` are applied cyclically until
    ``|a_p'a_q| <= tolerance ||a_p|| ||a_q||`` for every pair. A wide matrix is
    decomposed through its transpose.

    Args:
        matrix: Finite ``m x n`` matrix.
        tolerance: Relative orthogonality tolerance in ``(0, 1)``.
        max_sweeps: Maximum number of cyclic sweeps (positive).

    Returns:
        :class:`SingularValueDecomposition` with ``k = min(m, n)`` singular
        values in descending order; zero singular values are reported as
        exact zeros with orthonormal completing singular vectors.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix is empty, ragged, or non-finite, an
            argument is invalid, or the sweeps do not converge.

    Equation:
        [EQ-LINALG-SVD-001].

    References:
        [REF-HESTENES-1958], [REF-GOLUB-VANLOAN-2013, Sections 8.5-8.6].

    """
    checked = as_matrix(matrix, name="matrix")
    tol = require_finite_real(tolerance, name="tolerance")
    if not 0.0 < tol < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int):
        raise TypeError("max_sweeps must be an integer")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be positive")
    rows, columns = len(checked), len(checked[0])
    if rows >= columns:
        return _tall_svd(checked, tol, max_sweeps)
    transposed = tuple(zip(*checked, strict=True))
    result = _tall_svd(transposed, tol, max_sweeps)
    return SingularValueDecomposition(
        u=result.v,
        singular_values=result.singular_values,
        v=result.u,
        sweeps=result.sweeps,
    )


def pseudo_inverse(matrix: Matrix, *, rcond: float | None = None) -> Matrix:
    """Return the Moore-Penrose pseudo-inverse with a relative cut-off.

    Args:
        matrix: Finite ``m x n`` matrix.
        rcond: Relative cut-off in ``[0, 1)``; singular values not above
            ``rcond * sigma_1`` are treated as zero. ``None`` uses
            ``max(m, n) * 2^-52``.

    Returns:
        ``n x m`` matrix ``A^+ = V S^+ U'``; the zero ``n x m`` matrix for a
        zero ``A``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix or ``rcond`` is invalid.

    Equation:
        [EQ-LINALG-SVD-002].

    References:
        [REF-PENROSE-1955], [REF-GOLUB-VANLOAN-2013, Section 5.5].

    """
    return singular_value_decomposition(matrix).pseudo_inverse(rcond=rcond)


def numerical_rank(matrix: Matrix, *, tolerance: float | None = None) -> int:
    """Return the number of singular values above ``tolerance * sigma_1``.

    Args:
        matrix: Finite ``m x n`` matrix.
        tolerance: Relative threshold in ``[0, 1)``; ``None`` uses
            ``max(m, n) * 2^-52``.

    Returns:
        Numerical rank between ``0`` and ``min(m, n)``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix or the tolerance is invalid.

    Equation:
        [EQ-LINALG-SVD-002].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 5.5].

    """
    return singular_value_decomposition(matrix).rank(tolerance=tolerance)
