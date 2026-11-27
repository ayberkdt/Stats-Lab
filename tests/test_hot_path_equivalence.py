"""Bit-for-bit equivalence of the optimized numerical kernels with the originals.

The linear-algebra kernels (``probstat.linalg``) and the Gaussian-process
likelihood were rewritten for speed with the promise that no result changes:
``math.fsum`` is exactly rounded, so ``fsum(map(mul, a, b))`` equals the former
``fsum(x * y for ...)`` whatever the order, and every other rewrite keeps the
order of the rounded operations. This file pins the promise.

The ``_ref_*`` code below is the implementation of git revision ``315d501``
(the last commit before the rewrite), extracted by syntax tree: identical code
apart from removed docstrings and the ``_ref_`` / ``_Ref`` name prefixes. Each
test feeds fixed-seed random inputs, including adversarial ones, to the current
and to the reference routine and requires equal float bit patterns
(``float.hex``) and identical exception types and messages.

Golden values at the end of the file were produced by the same revision for
the end-to-end Gaussian-process fit, which exercises most of the rewritten
code at once.

References:
    [REF-GOLUB-VANLOAN-2013, Chapters 3-5], [REF-RASMUSSEN-WILLIAMS-2006].

"""

from __future__ import annotations

import dataclasses
import math
import operator
import random
import unittest
from collections.abc import Callable
from typing import Any

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    cholesky,
    cholesky_semidefinite,
    cholesky_solve,
    determinant,
    dot,
    householder_qr,
    inverse_spd,
    log_abs_determinant,
    lu_solve,
    matmul,
    matvec,
    quadratic_form,
    singular_value_decomposition,
    solve_lower,
    solve_spd,
    solve_upper,
)
from probstat.linalg import dense as dense_module
from probstat.linalg.covariance_operators import (
    DenseCovariance,
    DiagonalCovariance,
    _CovarianceBase,
)
from probstat.linalg.dense import as_matrix, as_vector
from probstat.linalg.krylov import (
    _axpy,
    _finish,
    _norm,
    _require_positive_int,
    _require_tolerance,
    _scale,
    _setup,
    _sym_ortho,
)
from probstat.linalg.krylov import lsqr as new_lsqr
from probstat.linalg.operators import (
    _CONVOLUTION_BOUNDARIES,
    AdjointTestResult,
    Convolution1DOperator,
    DenseOperator,
    OperatorBase,
    _apply,
    _apply_transpose,
    _real,
    _require_dimension,
    _vector,
    adjoint_test,
    require_operator,
)
from probstat.linalg.qr import DEFAULT_RANK_TOLERANCE, HouseholderQR
from probstat.linalg.svd import DEFAULT_JACOBI_TOLERANCE, SingularValueDecomposition
from probstat.surrogates import (
    fit_gaussian_process,
    gaussian_process_log_marginal_likelihood,
)
from probstat.surrogates import gaussian_process as gp
from probstat.surrogates.gaussian_process import (
    _LOG_TWO_PI,
    _SQRT3,
    _SQRT5,
    KERNELS,
    _accelerated_backend,
    _dense_likelihood,
    _factorize_dense,
    _fast_cholesky,
    _gradient,
    _inverse_factor_columns,
    _split_parameters,
    _sumprod,
    _triangular_solve,
)


def _ref_transpose(matrix):
    return tuple(zip(*matrix, strict=True))


def _ref_dot(left, right):
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    return math.fsum((a * b for a, b in zip(left, right, strict=True)))


def _ref_matvec(matrix, vector):
    return tuple(_ref_dot(row, vector) for row in matrix)


def _ref_matmul(left, right):
    if len(left[0]) != len(right):
        raise ValueError("inner matrix dimensions must agree")
    columns = _ref_transpose(right)
    return tuple(tuple(_ref_dot(row, column) for column in columns) for row in left)


def _ref_cholesky(matrix):
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


def _ref_cholesky_semidefinite(matrix, *, tolerance=1e-12):
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


def _ref_solve_lower(lower, right_hand_side):
    solution = []
    for row, target in enumerate(right_hand_side):
        pivot = lower[row][row]
        if pivot == 0.0:
            raise ValueError("triangular matrix is singular")
        residual = target - math.fsum(
            lower[row][column] * solution[column] for column in range(row)
        )
        solution.append(residual / pivot)
    return tuple(solution)


def _ref_solve_upper(upper, right_hand_side):
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


def _ref_cholesky_solve(lower, right_hand_side):
    return _ref_solve_upper(
        _ref_transpose(lower), _ref_solve_lower(lower, right_hand_side)
    )


def _ref_inverse_from_cholesky(lower):
    size = len(lower)
    columns = tuple(
        _ref_cholesky_solve(
            lower, tuple(1.0 if row == j else 0.0 for row in range(size))
        )
        for j in range(size)
    )
    return tuple(
        tuple(0.5 * (columns[j][i] + columns[i][j]) for j in range(size))
        for i in range(size)
    )


def _ref_lu_solve(matrix, right_hand_side):
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
        work[column], work[pivot_row] = (work[pivot_row], work[column])
        pivot = work[column][column]
        for row in range(column + 1, size):
            factor = work[row][column] / pivot
            if factor != 0.0:
                for inner in range(column, size + 1):
                    work[row][inner] -= factor * work[column][inner]
    upper = tuple(tuple(row[:size]) for row in work)
    return _ref_solve_upper(upper, tuple(row[size] for row in work))


def _ref_lu_diagonal(matrix):
    checked = as_matrix(matrix, name="matrix", square=True)
    size = len(checked)
    work = [list(row) for row in checked]
    swaps = 0
    diagonal = []
    for column in range(size):
        pivot_row = max(range(column, size), key=lambda r: abs(work[r][column]))
        if work[pivot_row][column] == 0.0:
            return None
        if pivot_row != column:
            work[column], work[pivot_row] = (work[pivot_row], work[column])
            swaps += 1
        pivot = work[column][column]
        diagonal.append(pivot)
        for row in range(column + 1, size):
            factor = work[row][column] / pivot
            if factor != 0.0:
                for inner in range(column, size):
                    work[row][inner] -= factor * work[column][inner]
    return (tuple(diagonal), swaps)


def _ref_determinant(matrix):
    factors = _ref_lu_diagonal(matrix)
    if factors is None:
        return 0.0
    diagonal, swaps = factors
    return (-1.0 if swaps % 2 else 1.0) * math.prod(diagonal)


def _ref_log_abs_determinant(matrix):
    factors = _ref_lu_diagonal(matrix)
    if factors is None:
        raise ValueError("matrix is singular; log|det| is -infinity")
    diagonal, _ = factors
    return math.fsum(math.log(abs(value)) for value in diagonal)


def _ref_quadratic_form(matrix, vector):
    return _ref_dot(vector, _ref_matvec(matrix, vector))


def _ref_householder_qr(matrix, *, rank_tolerance=DEFAULT_RANK_TOLERANCE):
    checked = as_matrix(matrix, name="matrix")
    rows, columns = (len(checked), len(checked[0]))
    if rows < columns:
        raise ValueError("householder_qr needs at least as many rows as columns")
    if not 0.0 < rank_tolerance < 1.0:
        raise ValueError("rank_tolerance must lie strictly between zero and one")
    work = [list(column) for column in zip(*checked, strict=True)]
    column_norms = tuple(math.hypot(*column) for column in work)
    reflectors = []
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
            projection = 2.0 * math.fsum(
                vector[i] * column[k + i] for i in range(rows - k)
            )
            if projection != 0.0:
                for i in range(rows - k):
                    column[k + i] -= projection * vector[i]
    upper = tuple(
        tuple(work[j][i] if j >= i else 0.0 for j in range(columns))
        for i in range(columns)
    )
    q_columns = []
    for j in range(columns):
        column = [1.0 if i == j else 0.0 for i in range(rows)]
        for k in range(columns - 1, -1, -1):
            vector = reflectors[k]
            projection = 2.0 * math.fsum(
                vector[i] * column[k + i] for i in range(rows - k)
            )
            if projection != 0.0:
                for i in range(rows - k):
                    column[k + i] -= projection * vector[i]
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


def _ref_qr_solve(self, right_hand_side):
    if self.rank_deficient:
        raise ValueError("least-squares solve needs a full-rank factorization")
    target = as_vector(right_hand_side, name="right_hand_side")
    if len(target) != len(self.q):
        raise ValueError("right_hand_side length must equal the row count")
    projected = tuple(
        math.fsum(self.q[i][j] * target[i] for i in range(len(target)))
        for j in range(len(self.r))
    )
    return _ref_solve_upper(self.r, projected)


def _ref_qr_r_inverse(self):
    if self.rank_deficient:
        raise ValueError("R is singular for a rank-deficient factorization")
    size = len(self.r)
    columns = [
        _ref_solve_upper(self.r, tuple(1.0 if i == j else 0.0 for i in range(size)))
        for j in range(size)
    ]
    return tuple(tuple(columns[j][i] for j in range(size)) for i in range(size))


def _ref_qr_gram_inverse(self):
    inverse = self.r_inverse()
    size = len(inverse)
    rows = []
    for i in range(size):
        rows.append(
            tuple(
                math.fsum(inverse[i][k] * inverse[j][k] for k in range(max(i, j), size))
                for j in range(size)
            )
        )
    return tuple(rows)


def _ref_qr_hat_diagonal(self):
    return tuple(math.fsum(value * value for value in row) for row in self.q)


def _ref_svd_dot(left, right):
    return math.fsum(map(operator.mul, left, right))


def _ref_svd_complete_orthonormal(columns, missing, size):
    basis = [columns[j] for j in range(len(columns)) if j not in set(missing)]
    for index in missing:
        best = None
        best_norm = -1.0
        for unit in range(size):
            candidate = [1.0 if i == unit else 0.0 for i in range(size)]
            for _ in range(2):
                for vector in basis:
                    projection = _ref_svd_dot(vector, candidate)
                    candidate = [
                        c - projection * b
                        for c, b in zip(candidate, vector, strict=True)
                    ]
            length = math.sqrt(_ref_svd_dot(candidate, candidate))
            if length > best_norm:
                best, best_norm = (candidate, length)
        assert best is not None
        normalized = [value / best_norm for value in best]
        columns[index] = normalized
        basis.append(normalized)


def _ref_svd_one_sided_jacobi(columns, tolerance, max_sweeps):
    size = len(columns)
    v = [[1.0 if i == j else 0.0 for i in range(size)] for j in range(size)]
    norms = [_ref_svd_dot(column, column) for column in columns]
    for sweep in range(1, max_sweeps + 1):
        rotated = False
        for p in range(size - 1):
            for q in range(p + 1, size):
                alpha, beta = (norms[p], norms[q])
                if alpha == 0.0 or beta == 0.0:
                    continue
                gamma = _ref_svd_dot(columns[p], columns[q])
                if abs(gamma) <= tolerance * math.sqrt(alpha) * math.sqrt(beta):
                    continue
                rotated = True
                zeta = (beta - alpha) / (2.0 * gamma)
                t = math.copysign(1.0, zeta) / (abs(zeta) + math.hypot(1.0, zeta))
                c = 1.0 / math.hypot(1.0, t)
                s = c * t
                cp, cq = (columns[p], columns[q])
                columns[p] = [c * a - s * b for a, b in zip(cp, cq, strict=True)]
                columns[q] = [s * a + c * b for a, b in zip(cp, cq, strict=True)]
                vp, vq = (v[p], v[q])
                v[p] = [c * a - s * b for a, b in zip(vp, vq, strict=True)]
                v[q] = [s * a + c * b for a, b in zip(vp, vq, strict=True)]
                norms[p] = _ref_svd_dot(columns[p], columns[p])
                norms[q] = _ref_svd_dot(columns[q], columns[q])
        if not rotated:
            return (v, sweep)
    raise ValueError("one-sided Jacobi SVD did not converge")


def _ref_svd_tall_svd(matrix, tolerance, max_sweeps):
    rows, size = (len(matrix), len(matrix[0]))
    q = None
    if rows > size:
        factorization = _ref_householder_qr(matrix)
        q = factorization.q
        work = [[factorization.r[i][j] for i in range(size)] for j in range(size)]
    else:
        work = [[matrix[i][j] for i in range(rows)] for j in range(size)]
    v_columns, sweeps = _ref_svd_one_sided_jacobi(work, tolerance, max_sweeps)
    values = [math.sqrt(_ref_svd_dot(column, column)) for column in work]
    order = sorted(range(size), key=lambda j: -values[j])
    values = [values[j] for j in order]
    work = [work[j] for j in order]
    v_columns = [v_columns[j] for j in order]
    zero = [j for j in range(size) if values[j] == 0.0]
    for j in range(size):
        if values[j] > 0.0:
            work[j] = [value / values[j] for value in work[j]]
    if zero:
        _ref_svd_complete_orthonormal(work, zero, len(work[0]))
    if q is not None:
        u_columns = [
            [_ref_svd_dot(list(q[i]), column) for i in range(rows)] for column in work
        ]
    else:
        u_columns = work
    u = tuple(tuple(u_columns[j][i] for j in range(size)) for i in range(rows))
    v = tuple(tuple(v_columns[j][i] for j in range(size)) for i in range(size))
    return SingularValueDecomposition(
        u=u, singular_values=tuple(values), v=v, sweeps=sweeps
    )


def _ref_svdsingular_value_decomposition(
    matrix, *, tolerance=DEFAULT_JACOBI_TOLERANCE, max_sweeps=60
):
    checked = as_matrix(matrix, name="matrix")
    tol = require_finite_real(tolerance, name="tolerance")
    if not 0.0 < tol < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int):
        raise TypeError("max_sweeps must be an integer")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be positive")
    rows, columns = (len(checked), len(checked[0]))
    if rows >= columns:
        return _ref_svd_tall_svd(checked, tol, max_sweeps)
    transposed = tuple(zip(*checked, strict=True))
    result = _ref_svd_tall_svd(transposed, tol, max_sweeps)
    return SingularValueDecomposition(
        u=result.v,
        singular_values=result.singular_values,
        v=result.u,
        sweeps=result.sweeps,
    )


def _ref_svd_pseudo_inverse(self, *, rcond=None):
    threshold = self._threshold(rcond)
    kept = [
        index for index, value in enumerate(self.singular_values) if value > threshold
    ]
    rows, columns = self.shape
    return tuple(
        tuple(
            math.fsum(
                self.v[i][index] * self.u[j][index] / self.singular_values[index]
                for index in kept
            )
            for j in range(rows)
        )
        for i in range(columns)
    )


def _ref_svd_reconstruct(self):
    rows, columns = self.shape
    k = len(self.singular_values)
    return tuple(
        tuple(
            math.fsum(
                self.u[i][index] * self.singular_values[index] * self.v[j][index]
                for index in range(k)
            )
            for j in range(columns)
        )
        for i in range(rows)
    )


def _ref_vector(values, size, name):
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a sequence of real numbers") from error
    if len(vector) != size:
        raise ValueError(f"{name} has length {len(vector)}, expected {size}")
    if not all(map(math.isfinite, vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


class _RefDenseOperator(OperatorBase):
    __slots__ = ("_columns", "_rows")

    def __init__(self, matrix, *, name="dense"):
        rows = tuple(tuple(row) for row in matrix)
        if not rows or not rows[0]:
            raise ValueError("matrix must be non-empty")
        width = len(rows[0])
        checked = tuple(
            (
                _ref_vector(row, width, f"{name} row {index}")
                for index, row in enumerate(rows)
            )
        )
        super().__init__(len(checked), width, name=name)
        self._rows = checked
        self._columns = tuple(zip(*checked, strict=True))

    @property
    def matrix(self):
        return self._rows

    def _forward(self, vector):
        return tuple(
            math.fsum((a * b for a, b in zip(row, vector, strict=True)))
            for row in self._rows
        )

    def _transpose(self, vector):
        return tuple(
            math.fsum((a * b for a, b in zip(column, vector, strict=True)))
            for column in self._columns
        )


class _RefConvolution1DOperator(OperatorBase):
    __slots__ = ("boundary", "kernel", "_taps")

    def __init__(self, kernel, size, *, boundary="zero"):
        _require_dimension(size, "size")
        weights = tuple(float(value) for value in kernel)
        if not weights:
            raise ValueError("kernel must be non-empty")
        self.kernel = _ref_vector(weights, len(weights), "kernel")
        if boundary not in _CONVOLUTION_BOUNDARIES:
            raise ValueError(f"boundary must be one of {_CONVOLUTION_BOUNDARIES}")
        self.boundary = boundary
        centre = len(weights) // 2
        taps = []
        for i in range(size):
            row = []
            for offset, weight in enumerate(self.kernel):
                j = i + offset - centre
                mapped = self._map(j, size)
                if mapped is not None and weight != 0.0:
                    row.append((i, mapped, weight))
            taps.append(tuple(row))
        self._taps = tuple(taps)
        super().__init__(size, size, name=f"convolution ({boundary})")

    def _map(self, index, size):
        if 0 <= index < size:
            return index
        if self.boundary == "zero":
            return None
        if self.boundary == "periodic":
            return index % size
        period = 2 * size
        folded = index % period
        return folded if folded < size else period - 1 - folded

    def _forward(self, vector):
        return tuple(
            math.fsum((weight * vector[j] for _, j, weight in row))
            for row in self._taps
        )

    def _transpose(self, vector):
        result = [0.0] * self._shape[1]
        for row in self._taps:
            for i, j, weight in row:
                result[j] += weight * vector[i]
        return result


def _ref_adjoint_test(operator, x, y, *, tolerance=1e-12):
    checked = require_operator(operator)
    rows, columns = checked.shape
    xv = _ref_vector(x, columns, "x")
    yv = _ref_vector(y, rows, "y")
    ax = _apply(checked, xv)
    aty = _apply_transpose(checked, yv)
    forward = math.fsum((a * b for a, b in zip(yv, ax, strict=True)))
    backward = math.fsum((a * b for a, b in zip(aty, xv, strict=True)))
    scale = math.hypot(*ax) * math.hypot(*yv) + math.hypot(*xv) * math.hypot(*aty)
    error = abs(forward - backward) / scale if scale > 0.0 else 0.0
    tol = _real(tolerance, "tolerance")
    return AdjointTestResult(forward, backward, error, error <= tol, tol)


class _RefDiagonalCovariance(_CovarianceBase):
    __slots__ = ("standard_deviations",)

    def __init__(self, variances):
        values = tuple(float(value) for value in variances)
        if not values:
            raise ValueError("variances must be non-empty")
        values = _vector(values, len(values), "variances")
        if any(value <= 0.0 for value in values):
            raise ValueError("variances must be strictly positive")
        self.standard_deviations = tuple(math.sqrt(value) for value in values)

    @classmethod
    def from_standard_deviations(cls, deviations):
        values = tuple(float(value) for value in deviations)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("standard deviations must be finite and positive")
        return cls(tuple(value * value for value in values))

    @property
    def dimension(self):
        return len(self.standard_deviations)

    def whiten(self, vector):
        checked = _vector(vector, self.dimension, "vector")
        return tuple(
            (r / s for r, s in zip(checked, self.standard_deviations, strict=True))
        )

    def whiten_transpose(self, vector):
        return self.whiten(vector)

    def unwhiten(self, vector):
        checked = _vector(vector, self.dimension, "vector")
        return tuple(
            (z * s for z, s in zip(checked, self.standard_deviations, strict=True))
        )

    def unwhiten_transpose(self, vector):
        return self.unwhiten(vector)

    def log_det(self):
        return 2.0 * math.fsum(math.log(s) for s in self.standard_deviations)


class _RefDenseCovariance(_CovarianceBase):
    __slots__ = ("lower", "_upper")

    def __init__(self, covariance, *, symmetry_tolerance=1e-12):
        matrix = as_matrix(covariance, name="covariance")
        size = len(matrix)
        if any(len(row) != size for row in matrix):
            raise ValueError("covariance must be square")
        scale = max(abs(value) for row in matrix for value in row)
        for i in range(size):
            for j in range(i):
                if abs(matrix[i][j] - matrix[j][i]) > symmetry_tolerance * max(
                    scale, 1e-300
                ):
                    raise ValueError("covariance must be symmetric")
        try:
            self.lower = _ref_cholesky(matrix)
        except ValueError as error:
            raise ValueError("covariance must be positive definite") from error
        self._upper = tuple(zip(*self.lower, strict=True))

    @classmethod
    def from_factor(cls, lower):
        factor = as_matrix(lower, name="lower")
        size = len(factor)
        if any(len(row) != size for row in factor):
            raise ValueError("lower must be square")
        for i in range(size):
            if factor[i][i] <= 0.0:
                raise ValueError("lower must have a positive diagonal")
            if any(factor[i][j] != 0.0 for j in range(i + 1, size)):
                raise ValueError("lower must be lower triangular")
        instance = cls.__new__(cls)
        instance.lower = factor
        instance._upper = tuple(zip(*factor, strict=True))
        return instance

    @property
    def dimension(self):
        return len(self.lower)

    def whiten(self, vector):
        return _ref_solve_lower(self.lower, _vector(vector, self.dimension, "vector"))

    def whiten_transpose(self, vector):
        values = list(_vector(vector, self.dimension, "vector"))
        size = self.dimension
        for i in range(size - 1, -1, -1):
            total = values[i] - math.fsum(
                self.lower[j][i] * values[j] for j in range(i + 1, size)
            )
            values[i] = total / self.lower[i][i]
        return tuple(values)

    def unwhiten(self, vector):
        checked = _vector(vector, self.dimension, "vector")
        return tuple(
            (
                math.fsum((row[j] * checked[j] for j in range(i + 1)))
                for i, row in enumerate(self.lower)
            )
        )

    def unwhiten_transpose(self, vector):
        checked = _vector(vector, self.dimension, "vector")
        size = self.dimension
        return tuple(
            math.fsum(self._upper[i][j] * checked[j] for j in range(i, size))
            for i in range(size)
        )

    def log_det(self):
        return 2.0 * math.fsum(
            math.log(self.lower[i][i]) for i in range(self.dimension)
        )


def _ref_krylov_dot(left, right):
    return math.fsum((a * b for a, b in zip(left, right, strict=True)))


def _ref_lsqr(
    operator,
    data,
    *,
    damping=0.0,
    initial=None,
    atol=1e-10,
    btol=1e-10,
    condition_limit=100000000.0,
    max_iterations=None,
    callback=None,
):
    counted, b, x0 = _setup(operator, data, initial)
    damp = _require_tolerance(damping, "damping")
    a_tol = _require_tolerance(atol, "atol")
    b_tol = _require_tolerance(btol, "btol")
    conlim = _require_tolerance(condition_limit, "condition_limit")
    if conlim <= 1.0:
        raise ValueError("condition_limit must exceed one")
    n = counted.shape[1]
    limit = (
        2 * n
        if max_iterations is None
        else _require_positive_int(max_iterations, "max_iterations")
    )
    nan = math.nan
    ax0 = counted.matvec(x0)
    u = [bi - ai for bi, ai in zip(b, ax0, strict=True)]
    beta = _norm(u)
    bnorm = _norm(b)
    x = list(x0)
    if beta == 0.0:
        reason = "zero_rhs" if bnorm == 0.0 and _norm(x0) == 0.0 else "exact"
        return _finish(
            counted,
            b,
            x,
            iterations=0,
            reason=reason,
            residual=0.0,
            normal=0.0,
            damping=damp,
            norm_estimate=nan,
            condition=nan,
            residuals=[0.0],
            normals=[0.0],
            method="lsqr",
        )
    u = _scale(1.0 / beta, u)
    v = list(counted.rmatvec(u))
    alpha = _norm(v)
    if alpha == 0.0:
        return _finish(
            counted,
            b,
            x,
            iterations=0,
            reason="exact",
            residual=beta,
            normal=0.0,
            damping=damp,
            norm_estimate=nan,
            condition=nan,
            residuals=[beta],
            normals=[0.0],
            method="lsqr",
        )
    v = _scale(1.0 / alpha, v)
    w = list(v)
    z = [0.0] * n
    phibar, rhobar = (beta, alpha)
    anorm2 = 0.0
    ddnorm = 0.0
    res2 = 0.0
    residuals, normals = ([beta], [alpha * beta])
    reason = "max_iterations"
    iterations = 0
    rnorm = beta
    arnorm = alpha * beta
    anorm = acond = nan
    damp2 = damp * damp
    for k in range(1, limit + 1):
        av = counted.matvec(v)
        u = _axpy(-alpha, u, av)
        beta = _norm(u)
        if beta > 0.0:
            u = _scale(1.0 / beta, u)
        anorm2 += alpha * alpha + beta * beta + damp2
        anorm = math.sqrt(anorm2)
        if beta > 0.0:
            atu = counted.rmatvec(u)
            v = _axpy(-beta, v, atu)
            alpha = _norm(v)
            if alpha > 0.0:
                v = _scale(1.0 / alpha, v)
        else:
            alpha = 0.0
        cs1, sn1, rhobar1 = _sym_ortho(rhobar, damp)
        psi = sn1 * phibar
        phibar = cs1 * phibar
        cs, sn, rho = _sym_ortho(rhobar1, beta)
        if rho == 0.0:
            reason = "breakdown"
            break
        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi
        t1 = phi / rho
        t2 = -theta / rho
        dk = _scale(1.0 / rho, w)
        z = _axpy(t1, w, z)
        w = _axpy(t2, w, v)
        ddnorm += _ref_krylov_dot(dk, dk)
        x = [a + c for a, c in zip(x0, z, strict=True)]
        res2 += psi * psi
        augmented = math.sqrt(phibar * phibar + res2)
        znorm = _norm(z)
        rnorm = math.sqrt(max(augmented * augmented - damp2 * znorm * znorm, 0.0))
        arnorm = alpha * abs(tau)
        acond = anorm * math.sqrt(ddnorm)
        residuals.append(rnorm)
        normals.append(arnorm)
        iterations = k
        if callback is not None and callback(k, tuple(x), rnorm):
            reason = "callback"
            break
        if augmented <= b_tol * bnorm + a_tol * anorm * znorm:
            reason = "residual_tolerance"
            break
        if augmented > 0.0 and arnorm <= a_tol * anorm * augmented:
            reason = "normal_tolerance"
            break
        if augmented == 0.0 or arnorm == 0.0:
            reason = "normal_tolerance"
            break
        if acond >= conlim:
            reason = "condition_limit"
            break
        if alpha == 0.0:
            reason = "normal_tolerance"
            break
    return _finish(
        counted,
        b,
        x,
        iterations=iterations,
        reason=reason,
        residual=rnorm,
        normal=arnorm,
        damping=damp,
        norm_estimate=anorm,
        condition=acond,
        residuals=residuals,
        normals=normals,
        method="lsqr",
    )


def _ref_assemble(points, scales, signal, noise, kernel):
    count = len(points)
    parts = []
    correlation = []
    factors = []
    matrix = []
    squared_exponential = kernel == "squared_exponential"
    matern32 = kernel == "matern32"
    for i in range(count):
        row_parts = []
        row_corr = []
        row_factor = []
        row_matrix = []
        xi = points[i]
        for j in range(i):
            difference = tuple(
                (
                    ((a - b) / ell) ** 2
                    for a, b, ell in zip(xi, points[j], scales, strict=True)
                )
            )
            squared = math.fsum(difference)
            if squared_exponential:
                value = math.exp(-0.5 * squared)
                factor = value
            elif matern32:
                r = math.sqrt(squared)
                decay = math.exp(-_SQRT3 * r)
                value = (1.0 + _SQRT3 * r) * decay
                factor = 3.0 * decay
            else:
                r = math.sqrt(squared)
                decay = math.exp(-_SQRT5 * r)
                value = (1.0 + _SQRT5 * r + 5.0 * squared / 3.0) * decay
                factor = 5.0 / 3.0 * (1.0 + _SQRT5 * r) * decay
            row_parts.append(difference)
            row_corr.append(value)
            row_factor.append(factor)
            row_matrix.append(signal * value)
        row_matrix.append(signal + noise)
        parts.append(row_parts)
        correlation.append(row_corr)
        factors.append(row_factor)
        matrix.append(row_matrix)
    return (matrix, parts, correlation, factors)


def _ref_gp_loglik(
    points,
    targets,
    log_parameters,
    *,
    kernel="matern52",
    nugget=1e-10,
    estimate_noise=False,
    relative_nugget=0.0,
):
    x = tuple(as_vector(point, name="point") for point in points)
    y = as_vector(targets, name="targets")
    dimension = len(x[0])
    signal, scales, extra = _split_parameters(log_parameters, dimension, estimate_noise)
    tau = require_finite_real(relative_nugget, name="relative_nugget")
    if tau < 0.0:
        raise ValueError("relative_nugget must be non-negative")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    count = len(x)
    noise = nugget + extra
    if tau > 0.0:
        noise += tau * signal
    matrix, parts, correlation, factors = _ref_assemble(
        x, scales, signal, noise, kernel
    )
    backend = _accelerated_backend()
    if backend is not None:
        try:
            factor = _factorize_dense(backend, matrix)
        except ValueError as error:
            raise ValueError(
                "kernel matrix is not positive definite; increase the nugget"
            ) from error
        value, trace_weight, signal_sum, dense_sums = _dense_likelihood(
            factor, y, parts, correlation, factors, dimension
        )
        return (
            value,
            _gradient(
                signal, trace_weight, signal_sum, dense_sums, tau, extra, estimate_noise
            ),
        )
    try:
        lower = _fast_cholesky(matrix)
    except ValueError as error:
        raise ValueError(
            "kernel matrix is not positive definite; increase the nugget"
        ) from error
    alpha = _triangular_solve(lower, y)
    log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(count))
    value = -0.5 * _sumprod(y, alpha)
    value += -0.5 * log_det - 0.5 * count * _LOG_TWO_PI
    columns = _inverse_factor_columns(lower)
    diagonal_weight = [
        alpha[a] * alpha[a] - _sumprod(columns[a], columns[a]) for a in range(count)
    ]
    trace_weight = math.fsum(diagonal_weight)
    signal_sum = 0.0
    axis_sums = [0.0] * dimension
    for b in range(count):
        alpha_b = alpha[b]
        column_b = columns[b]
        row_corr = correlation[b]
        row_factor = factors[b]
        row_parts = parts[b]
        for a in range(b):
            weight = alpha[a] * alpha_b - _sumprod(columns[a][b - a :], column_b)
            signal_sum += weight * row_corr[a]
            scaled_weight = weight * row_factor[a]
            difference = row_parts[a]
            for axis in range(dimension):
                axis_sums[axis] += scaled_weight * difference[axis]
    return (
        value,
        _gradient(
            signal, trace_weight, signal_sum, axis_sums, tau, extra, estimate_noise
        ),
    )


def _bits(value: Any) -> Any:
    """Return a comparable image of ``value`` in which floats are bit patterns."""
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (tuple, list)):
        return tuple(_bits(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            tuple(
                (field.name, _bits(getattr(value, field.name)))
                for field in dataclasses.fields(value)
            ),
        )
    return value


def _outcome(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Return the bit image of the result, or the exception type and message."""
    try:
        return ("value", _bits(function(*args, **kwargs)))
    except Exception as error:  # noqa: BLE001 - the exception is the result
        return ("error", type(error).__name__, str(error))


def _spd(
    rng: random.Random, size: int, ridge: float = 1.0
) -> tuple[tuple[float, ...], ...]:
    """Return ``B B' + ridge I`` with a Gaussian ``B``."""
    b = [[rng.gauss(0.0, 1.0) for _ in range(size)] for _ in range(size)]
    return tuple(
        tuple(
            math.fsum(b[i][k] * b[j][k] for k in range(size))
            + (ridge if i == j else 0.0)
            for j in range(size)
        )
        for i in range(size)
    )


def _dense(rng: random.Random, rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
    """Return a Gaussian ``rows x cols`` matrix."""
    return tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(cols)) for _ in range(rows))


def _vec(rng: random.Random, size: int) -> tuple[float, ...]:
    """Return a Gaussian vector."""
    return tuple(rng.gauss(0.0, 1.0) for _ in range(size))


class EquivalenceCase(unittest.TestCase):
    """Base class with the bit-exact comparison helper."""

    def assertSame(  # noqa: N802 - unittest naming
        self,
        new: Callable[..., Any],
        ref: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Require equal bit patterns, or the same exception type and message."""
        got = _outcome(new, *args, **kwargs)
        want = _outcome(ref, *args, **kwargs)
        self.assertEqual(got, want)


class DenseKernelEquivalenceTests(EquivalenceCase):
    """Dense Cholesky, triangular solves, inverses and LU against the originals."""

    def test_dot_matvec_matmul(self) -> None:
        """Compensated products agree for every length, mixed magnitudes included."""
        rng = random.Random(1)
        for size in (0, 1, 2, 3, 7, 33):
            left = tuple(
                rng.gauss(0.0, 1.0) * 10.0 ** rng.randint(-8, 8) for _ in range(size)
            )
            right = tuple(
                rng.gauss(0.0, 1.0) * 10.0 ** rng.randint(-8, 8) for _ in range(size)
            )
            self.assertSame(dot, _ref_dot, left, right)
        self.assertSame(dot, _ref_dot, (1.0, 2.0), (1.0,))
        matrix = _dense(rng, 9, 6)
        other = _dense(rng, 6, 4)
        self.assertSame(matvec, _ref_matvec, matrix, _vec(rng, 6))
        self.assertSame(matvec, _ref_matvec, matrix, _vec(rng, 5))
        self.assertSame(matmul, _ref_matmul, matrix, other)
        square = _spd(rng, 8)
        self.assertSame(quadratic_form, _ref_quadratic_form, square, _vec(rng, 8))

    def test_cholesky_on_random_spd_matrices(self) -> None:
        """The factor is identical at several sizes and conditionings."""
        rng = random.Random(2)
        for size in (1, 2, 3, 5, 17, 40):
            for ridge in (1.0, 1e-3, 1e-9):
                matrix = _spd(rng, size, ridge)
                self.assertSame(cholesky, _ref_cholesky, matrix)

    def test_cholesky_failures_and_malformed_input(self) -> None:
        """Rejections, NaN, ragged rows and extra columns behave as before."""
        rng = random.Random(3)
        indefinite = ((1.0, 2.0), (2.0, 1.0))
        self.assertSame(cholesky, _ref_cholesky, indefinite)
        singular = ((1.0, 1.0), (1.0, 1.0))
        self.assertSame(cholesky, _ref_cholesky, singular)
        self.assertSame(cholesky, _ref_cholesky, ((math.nan, 0.0), (0.0, 1.0)))
        self.assertSame(cholesky, _ref_cholesky, ((1.0, 0.0), (0.0, math.inf)))
        self.assertSame(cholesky, _ref_cholesky, ((1.0, 0.0), (0.0,)))
        self.assertSame(cholesky, _ref_cholesky, ((4.0, 1.0, 9.0), (1.0, 3.0, 9.0)))
        self.assertSame(cholesky, _ref_cholesky, ())
        self.assertSame(cholesky, _ref_cholesky, ((-1.0,),))
        self.assertSame(cholesky, _ref_cholesky, ((0.0,),))
        listed = [list(row) for row in _spd(rng, 6)]
        self.assertSame(cholesky, _ref_cholesky, listed)

    def test_cholesky_semidefinite(self) -> None:
        """Rank-deficient, zero and indefinite inputs give identical factors."""
        rng = random.Random(4)
        for size, rank in ((4, 2), (7, 3), (12, 12), (9, 1)):
            factor = _dense(rng, size, rank)
            matrix = tuple(
                tuple(
                    math.fsum(a * b for a, b in zip(r1, r2, strict=True))
                    for r2 in factor
                )
                for r1 in factor
            )
            self.assertSame(cholesky_semidefinite, _ref_cholesky_semidefinite, matrix)
            self.assertSame(
                cholesky_semidefinite,
                _ref_cholesky_semidefinite,
                matrix,
                tolerance=1e-6,
            )
        self.assertSame(
            cholesky_semidefinite, _ref_cholesky_semidefinite, ((0.0, 0.0), (0.0, 0.0))
        )
        self.assertSame(
            cholesky_semidefinite, _ref_cholesky_semidefinite, ((1.0, 2.0), (2.0, 1.0))
        )
        self.assertSame(
            cholesky_semidefinite,
            _ref_cholesky_semidefinite,
            ((0.0, 1.0), (1.0, 0.0)),
        )

    def test_triangular_solves(self) -> None:
        """Forward and back substitution agree, including degenerate inputs."""
        rng = random.Random(5)
        for size in (1, 2, 4, 9, 31):
            lower_factor = _ref_cholesky(_spd(rng, size))
            rhs = _vec(rng, size)
            self.assertSame(solve_lower, _ref_solve_lower, lower_factor, rhs)
            upper_factor = _ref_transpose(lower_factor)
            self.assertSame(solve_upper, _ref_solve_upper, upper_factor, rhs)
            self.assertSame(cholesky_solve, _ref_cholesky_solve, lower_factor, rhs)
            self.assertSame(
                solve_spd,
                lambda m, r: _ref_cholesky_solve(_ref_cholesky(m), r),
                _spd(rng, size),
                rhs,
            )
        zero_pivot = ((1.0, 0.0), (1.0, 0.0))
        self.assertSame(solve_lower, _ref_solve_lower, zero_pivot, (1.0, 1.0))
        self.assertSame(
            solve_upper, _ref_solve_upper, ((0.0, 1.0), (0.0, 1.0)), (1.0, 1.0)
        )
        short_upper = ((1.0, 2.0, 3.0), (0.0, 1.0), (0.0, 0.0, 1.0))
        self.assertSame(solve_upper, _ref_solve_upper, short_upper, (1.0, 1.0, 1.0))
        self.assertSame(
            solve_lower, _ref_solve_lower, ((1.0,), (1.0, 2.0)), (1.0, 2.0, 3.0)
        )
        self.assertSame(solve_lower, _ref_solve_lower, ((1.0, 0.0), (1.0, 2.0)), (1.0,))
        self.assertSame(solve_lower, _ref_solve_lower, ((2.0,), (1.0,)), (1.0, 2.0))
        negative = ((-2.0, 0.0), (1.0, -3.0))
        self.assertSame(solve_lower, _ref_solve_lower, negative, (1.0, 1.0))
        self.assertSame(solve_upper, _ref_solve_upper, ((1.0, 2.0), (0.0, 1.0)), (1.0,))

    def test_inverse_spd_and_its_guarded_fast_path(self) -> None:
        """The structure-exploiting inverse equals the unit-vector solves exactly."""
        rng = random.Random(6)
        for size in (1, 2, 3, 8, 25, 50):
            for ridge in (1.0, 1e-6):
                matrix = _spd(rng, size, ridge)
                self.assertSame(
                    inverse_spd,
                    lambda m: _ref_inverse_from_cholesky(_ref_cholesky(m)),
                    matrix,
                )
        inverse_from_factor = dense_module._inverse_from_cholesky
        factors = {
            "negative diagonal": ((-2.0, 0.0), (1.0, -3.0)),
            "zero diagonal": ((1.0, 0.0), (1.0, 0.0)),
            "infinite entry": ((1.0, 0.0), (math.inf, 1.0)),
            "nan entry": ((1.0, 0.0), (math.nan, 1.0)),
            "ragged": ((1.0, 0.0), (1.0,)),
            "upper entries ignored": ((2.0, 5.0), (1.0, 3.0)),
            "wide": ((2.0, 0.0, 0.0), (1.0, 3.0, 0.0)),
            "empty": (),
        }
        for label, factor in factors.items():
            with self.subTest(label):
                self.assertSame(inverse_from_factor, _ref_inverse_from_cholesky, factor)

    def test_lu_family(self) -> None:
        """LU solves and determinants agree, singular and ill-sized cases included."""
        rng = random.Random(7)
        for size in (1, 2, 5, 12, 30):
            matrix = _dense(rng, size, size)
            rhs = _vec(rng, size)
            self.assertSame(lu_solve, _ref_lu_solve, matrix, rhs)
            self.assertSame(determinant, _ref_determinant, matrix)
            self.assertSame(log_abs_determinant, _ref_log_abs_determinant, matrix)
        singular = ((1.0, 2.0), (2.0, 4.0))
        self.assertSame(lu_solve, _ref_lu_solve, singular, (1.0, 1.0))
        self.assertSame(determinant, _ref_determinant, singular)
        self.assertSame(log_abs_determinant, _ref_log_abs_determinant, singular)
        self.assertSame(lu_solve, _ref_lu_solve, ((1.0, 2.0), (3.0, 4.0)), (1.0,))
        self.assertSame(
            lu_solve, _ref_lu_solve, ((1.0, 2.0, 3.0), (3.0, 4.0, 5.0)), (1.0, 1.0)
        )
        pivoting = ((0.0, 1.0, 2.0), (1.0, 0.0, 3.0), (4.0, 5.0, 0.0))
        self.assertSame(lu_solve, _ref_lu_solve, pivoting, (1.0, 2.0, 3.0))
        self.assertSame(determinant, _ref_determinant, pivoting)


class FactorizationEquivalenceTests(EquivalenceCase):
    """Householder QR and Jacobi SVD against the originals."""

    def _qr_cases(self) -> list[tuple[str, tuple[tuple[float, ...], ...]]]:
        """Return named test matrices, including rank-deficient ones."""
        rng = random.Random(8)
        cases: list[tuple[str, tuple[tuple[float, ...], ...]]] = []
        for rows, cols in ((1, 1), (3, 2), (6, 3), (12, 12), (20, 7), (40, 15)):
            cases.append((f"gauss {rows}x{cols}", _dense(rng, rows, cols)))
        base = _dense(rng, 9, 3)
        duplicated = tuple(row + (row[0],) for row in base)
        cases.append(("duplicate column", duplicated))
        zero_column = tuple((row[0], 0.0, row[1]) for row in base)
        cases.append(("zero column", zero_column))
        cases.append(("zero matrix", ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))))
        cases.append(("wide", _dense(rng, 2, 4)))
        cases.append(
            (
                "graded",
                tuple(
                    tuple(v * 10.0 ** (-k) for k, v in enumerate(r))
                    for r in _dense(rng, 10, 6)
                ),
            )
        )
        cases.append(("ragged", ((1.0, 2.0), (3.0,))))
        return cases

    def test_householder_qr_factors(self) -> None:
        """Q, R, norms and the rank flags are identical."""
        for label, matrix in self._qr_cases():
            with self.subTest(label):
                self.assertSame(householder_qr, _ref_householder_qr, matrix)

    def test_householder_qr_methods(self) -> None:
        """``solve``, ``r_inverse``, ``gram_inverse`` and ``hat_diagonal`` agree."""
        rng = random.Random(9)
        for label, matrix in self._qr_cases():
            try:
                factorization = _ref_householder_qr(matrix)
            except ValueError:
                continue
            rows = len(matrix)
            with self.subTest(label):
                for rhs in (_vec(rng, rows), _vec(rng, rows + 1)):
                    self.assertSame(
                        HouseholderQR.solve, _ref_qr_solve, factorization, rhs
                    )
                self.assertSame(
                    HouseholderQR.r_inverse, _ref_qr_r_inverse, factorization
                )
                self.assertSame(
                    HouseholderQR.gram_inverse, _ref_qr_gram_inverse, factorization
                )
                self.assertSame(
                    HouseholderQR.hat_diagonal, _ref_qr_hat_diagonal, factorization
                )

    def test_qr_solve_with_malformed_hand_built_factor(self) -> None:
        """A ragged ``q`` keeps the original behaviour."""
        factor = HouseholderQR(
            q=((1.0, 0.0), (0.0,)),
            r=((1.0, 0.0), (0.0, 1.0)),
            column_norms=(1.0, 1.0),
            rank_deficient=(),
            tolerance=1e-10,
        )
        self.assertSame(HouseholderQR.solve, _ref_qr_solve, factor, (1.0, 2.0))

    def test_svd_decompositions(self) -> None:
        """U, singular values, V and the sweep count are identical."""
        rng = random.Random(10)
        shapes = ((1, 1), (2, 2), (3, 5), (5, 3), (8, 8), (14, 6), (6, 14), (20, 10))
        for rows, cols in shapes:
            matrix = _dense(rng, rows, cols)
            with self.subTest((rows, cols)):
                self.assertSame(
                    singular_value_decomposition,
                    _ref_svdsingular_value_decomposition,
                    matrix,
                )
        low_rank = tuple(
            tuple(
                a * b + c * d for b, d in zip(_vec(rng, 5), _vec(rng, 5), strict=True)
            )
            for a, c in zip(_vec(rng, 9), _vec(rng, 9), strict=True)
        )
        zero = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        rank_one = ((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (3.0, 6.0, 9.0))
        for matrix in (low_rank, zero, rank_one):
            self.assertSame(
                singular_value_decomposition,
                _ref_svdsingular_value_decomposition,
                matrix,
            )
        self.assertSame(
            singular_value_decomposition,
            _ref_svdsingular_value_decomposition,
            _dense(rng, 6, 6),
            max_sweeps=1,
        )
        self.assertSame(
            singular_value_decomposition, _ref_svdsingular_value_decomposition, ()
        )

    def test_svd_derived_matrices(self) -> None:
        """``pseudo_inverse`` and ``reconstruct`` agree with their originals."""
        rng = random.Random(11)
        for rows, cols in ((4, 4), (7, 3), (3, 7)):
            decomposition = singular_value_decomposition(_dense(rng, rows, cols))
            self.assertSame(
                SingularValueDecomposition.pseudo_inverse,
                _ref_svd_pseudo_inverse,
                decomposition,
            )
            self.assertSame(
                SingularValueDecomposition.pseudo_inverse,
                _ref_svd_pseudo_inverse,
                decomposition,
                rcond=0.3,
            )
            self.assertSame(
                SingularValueDecomposition.reconstruct,
                _ref_svd_reconstruct,
                decomposition,
            )
        deficient = singular_value_decomposition(((1.0, 2.0), (2.0, 4.0), (3.0, 6.0)))
        self.assertSame(
            SingularValueDecomposition.pseudo_inverse,
            _ref_svd_pseudo_inverse,
            deficient,
        )
        self.assertSame(
            SingularValueDecomposition.reconstruct, _ref_svd_reconstruct, deficient
        )
        self.assertSame(
            SingularValueDecomposition.pseudo_inverse,
            _ref_svd_pseudo_inverse,
            deficient,
            rcond=2.0,
        )
        broken = SingularValueDecomposition(
            u=((1.0,), (0.0, 1.0)),
            singular_values=(2.0, 1.0),
            v=((1.0, 0.0), (0.0, 1.0)),
            sweeps=1,
        )
        self.assertSame(
            SingularValueDecomposition.reconstruct, _ref_svd_reconstruct, broken
        )


class OperatorEquivalenceTests(EquivalenceCase):
    """Operator validation, dense and convolution actions, covariance objects."""

    def test_vector_validation(self) -> None:
        """``_vector`` converts and rejects exactly as before."""
        cases: list[Any] = [
            (1, 2.5, True),
            [1.0, 2.0],
            (1.0, math.nan),
            (1.0, math.inf),
            ("a", 1.0),
            (None,),
            (1.0, 2.0, 3.0),
            iter((1.0, 2.0)),
            5,
            (),
        ]
        for values in cases:
            for size in (0, 2, 3):
                if hasattr(values, "__next__"):
                    values_new = iter((1.0, 2.0))
                    values_ref = iter((1.0, 2.0))
                else:
                    values_new = values_ref = values
                got = _outcome(_vector, values_new, size, "v")
                want = _outcome(_ref_vector, values_ref, size, "v")
                self.assertEqual(got, want)

    def test_dense_operator(self) -> None:
        """Forward and transpose products of ``DenseOperator`` are identical."""
        rng = random.Random(12)
        for rows, cols in ((1, 1), (3, 5), (8, 4), (20, 20)):
            matrix = _dense(rng, rows, cols)
            new = DenseOperator(matrix)
            ref = _RefDenseOperator(matrix)
            x, y = _vec(rng, cols), _vec(rng, rows)
            self.assertEqual(_bits(new.matvec(x)), _bits(ref.matvec(x)))
            self.assertEqual(_bits(new.rmatvec(y)), _bits(ref.rmatvec(y)))
            self.assertSame(new.matvec, ref.matvec, _vec(rng, cols + 1))
            self.assertSame(new.rmatvec, ref.rmatvec, (math.nan,) * rows)
            self.assertSame(adjoint_test, _ref_adjoint_test, new, x, y)

    def test_convolution_operator(self) -> None:
        """Convolution forward and scatter-add transpose keep their rounding."""
        rng = random.Random(13)
        kernels = (
            (1.0,),
            (0.25, 0.5, 0.25),
            (0.1, 0.0, 0.7, 0.2),
            tuple(rng.gauss(0.0, 1.0) for _ in range(5)),
            (0.0, 0.0),
            tuple(rng.gauss(0.0, 1.0) for _ in range(9)),
        )
        for boundary in ("zero", "periodic", "reflect"):
            for size in (1, 2, 4, 30, 65):
                for kernel in kernels:
                    new = Convolution1DOperator(kernel, size, boundary=boundary)
                    ref = _RefConvolution1DOperator(kernel, size, boundary=boundary)
                    x = _vec(rng, size)
                    with self.subTest((boundary, size, len(kernel))):
                        self.assertEqual(_bits(new.matvec(x)), _bits(ref.matvec(x)))
                        self.assertEqual(_bits(new.rmatvec(x)), _bits(ref.rmatvec(x)))
        got = _outcome(lambda: Convolution1DOperator((1.0,), 4, boundary="bad"))
        want = _outcome(lambda: _RefConvolution1DOperator((1.0,), 4, boundary="bad"))
        self.assertEqual(got, want)

    def test_covariance_objects(self) -> None:
        """Diagonal and dense covariance actions agree with their originals."""
        rng = random.Random(14)
        variances = tuple(0.1 + rng.random() for _ in range(9))
        new_diag = DiagonalCovariance(variances)
        ref_diag = _RefDiagonalCovariance(variances)
        matrix = _spd(rng, 9)
        new_dense = DenseCovariance(matrix)
        ref_dense = _RefDenseCovariance(matrix)
        for new, ref in ((new_diag, ref_diag), (new_dense, ref_dense)):
            for _ in range(3):
                v = _vec(rng, 9)
                for name in (
                    "whiten",
                    "whiten_transpose",
                    "unwhiten",
                    "unwhiten_transpose",
                    "quadratic_form",
                    "solve",
                    "apply",
                    "log_det",
                ):
                    method_new = getattr(new, name)
                    method_ref = getattr(ref, name)
                    if name == "log_det":
                        self.assertEqual(_bits(method_new()), _bits(method_ref()))
                    else:
                        self.assertEqual(
                            _outcome(method_new, v), _outcome(method_ref, v), name
                        )
            self.assertSame(new.whiten, ref.whiten, (1.0, 2.0))
        factor = _ref_cholesky(_spd(rng, 6))
        vector = _vec(rng, 6)
        new_factor = DenseCovariance.from_factor(factor)
        ref_factor = _RefDenseCovariance.from_factor(factor)
        for name in ("whiten", "whiten_transpose", "unwhiten", "unwhiten_transpose"):
            self.assertEqual(
                _outcome(getattr(new_factor, name), vector),
                _outcome(getattr(ref_factor, name), vector),
            )


def _krylov_problem(rng: random.Random, kind: str) -> tuple[Any, tuple[float, ...]]:
    """Return an operator and a data vector for the LSQR comparison."""
    if kind == "dense":
        matrix = _dense(rng, 30, 12)
        operator: Any = DenseOperator(matrix)
        data = _vec(rng, 30)
    elif kind == "convolution":
        operator = Convolution1DOperator((0.1, 0.4, 0.4, 0.1), 80, boundary="reflect")
        truth = tuple(math.sin(0.2 * i) for i in range(80))
        data = tuple(v + 1e-3 * rng.gauss(0.0, 1.0) for v in operator.matvec(truth))
    else:
        operator = DenseOperator(_dense(rng, 12, 12))
        data = _vec(rng, 12)
    return operator, data


class KrylovEquivalenceTests(EquivalenceCase):
    """LSQR with its lazily formed iterate against the original loop."""

    def test_lsqr_results(self) -> None:
        """Every field of the result is bit-identical, with and without extras."""
        rng = random.Random(15)
        for kind in ("dense", "convolution", "square"):
            operator, data = _krylov_problem(rng, kind)
            columns = operator.shape[1]
            options: list[dict[str, Any]] = [
                {},
                {"damping": 0.3},
                {"initial": _vec(rng, columns)},
                {"max_iterations": 3},
                {"atol": 1e-14, "btol": 1e-14, "condition_limit": 1e12},
                {"damping": 0.1, "initial": _vec(rng, columns), "max_iterations": 7},
            ]
            for option in options:
                with self.subTest((kind, tuple(option))):
                    self.assertSame(new_lsqr, _ref_lsqr, operator, data, **option)

    def test_lsqr_callback_and_special_cases(self) -> None:
        """Callbacks see identical iterates; zero data and exact starts agree."""
        rng = random.Random(16)
        operator, data = _krylov_problem(rng, "dense")

        def run(solver: Callable[..., Any], stop: int) -> tuple[Any, list[Any]]:
            seen: list[Any] = []

            def callback(k: int, x: Any, rnorm: float) -> bool:
                seen.append((k, _bits(x), rnorm.hex()))
                return k >= stop

            return solver(operator, data, callback=callback), seen

        for stop in (1, 4, 1000):
            got, got_seen = run(new_lsqr, stop)
            want, want_seen = run(_ref_lsqr, stop)
            self.assertEqual(_bits(got), _bits(want))
            self.assertEqual(got_seen, want_seen)
        zeros = (0.0,) * 30
        self.assertSame(new_lsqr, _ref_lsqr, operator, zeros)
        exact = tuple(operator.matvec((1.0,) * 12))
        self.assertSame(new_lsqr, _ref_lsqr, operator, exact, initial=(1.0,) * 12)
        self.assertSame(new_lsqr, _ref_lsqr, operator, data, damping=-1.0)
        self.assertSame(new_lsqr, _ref_lsqr, operator, data[:5])


class GaussianProcessEquivalenceTests(EquivalenceCase):
    """GP assembly, likelihood and gradient against the original code."""

    def _design(self, seed: int, count: int, dimension: int) -> tuple[Any, Any]:
        """Return random inputs and smooth targets."""
        rng = random.Random(seed)
        points = [[rng.random() for _ in range(dimension)] for _ in range(count)]
        values = [
            math.sin(3.0 * p[0]) + sum(p) + 0.01 * rng.gauss(0.0, 1.0) for p in points
        ]
        return points, values

    def test_assembled_rows_and_derivative_pieces(self) -> None:
        """Rows, squared differences, correlations and factors are identical."""
        for kernel in ("matern52", "matern32", "squared_exponential"):
            for count, dimension in ((1, 2), (2, 1), (9, 3), (30, 4)):
                points, _ = self._design(count, count, dimension)
                x = tuple(tuple(p) for p in points)
                scales = tuple(0.3 + 0.2 * k for k in range(dimension))
                want = _ref_assemble(x, scales, 1.7, 1e-6, kernel)
                got = gp._assemble(x, scales, 1.7, 1e-6, kernel)
                self.assertEqual(_bits(got), _bits(want))
                rows_only = gp._assemble(
                    x, scales, 1.7, 1e-6, kernel, derivatives=False
                )
                self.assertEqual(_bits(rows_only[0]), _bits(want[0]))

    def test_assembly_rejects_mismatched_dimensions_like_before(self) -> None:
        """Ragged points and wrong scale counts raise the same error."""
        x = ((0.1, 0.2), (0.3,), (0.5, 0.6))
        self.assertSame(
            gp._assemble, _ref_assemble, x, (1.0, 1.0), 1.0, 0.0, "matern52"
        )
        y = ((0.1, 0.2), (0.3, 0.4))
        self.assertSame(gp._assemble, _ref_assemble, y, (1.0,), 1.0, 0.0, "matern52")
        self.assertSame(
            gp._assemble, _ref_assemble, y, (1.0, 1.0, 1.0), 1.0, 0.0, "matern32"
        )

    def test_log_marginal_likelihood_and_gradient(self) -> None:
        """Value and exact gradient are bit-identical for every option."""
        for kernel in ("matern52", "matern32", "squared_exponential"):
            for count, dimension in ((2, 1), (10, 2), (35, 3)):
                points, values = self._design(count + dimension, count, dimension)
                theta = (0.1,) + (math.log(0.4),) * dimension
                options: list[dict[str, Any]] = [
                    {},
                    {"relative_nugget": 1e-6},
                    {"nugget": 1e-4},
                ]
                for option in options:
                    with self.subTest((kernel, count, dimension, tuple(option))):
                        self.assertSame(
                            gaussian_process_log_marginal_likelihood,
                            _ref_gp_loglik,
                            points,
                            values,
                            theta,
                            kernel=kernel,
                            **option,
                        )
                noisy = theta + (math.log(1e-3),)
                self.assertSame(
                    gaussian_process_log_marginal_likelihood,
                    _ref_gp_loglik,
                    points,
                    values,
                    noisy,
                    kernel=kernel,
                    estimate_noise=True,
                )

    def test_likelihood_failures(self) -> None:
        """Non-positive-definite kernels and bad arguments fail identically."""
        points = [[0.5, 0.5]] * 4
        values = [1.0, 2.0, 3.0, 4.0]
        theta = (0.0, 0.0, 0.0)
        self.assertSame(
            gaussian_process_log_marginal_likelihood,
            _ref_gp_loglik,
            points,
            values,
            theta,
            nugget=0.0,
        )
        self.assertSame(
            gaussian_process_log_marginal_likelihood,
            _ref_gp_loglik,
            points,
            values,
            theta,
            kernel="nope",
        )
        self.assertSame(
            gaussian_process_log_marginal_likelihood,
            _ref_gp_loglik,
            [[0.1, 0.2], [0.3], [0.5, 0.6]],
            values[:3],
            theta,
        )

    def test_fit_matches_golden_values_of_the_original(self) -> None:
        """A full fit reproduces predictions recorded from revision 315d501."""
        points, values = self._design(7, 24, 3)
        model = fit_gaussian_process(
            points, values, seed=3, restarts=1, estimate_noise=True
        )
        probe = [[0.2, 0.4, 0.6], [0.9, 0.1, 0.5]]
        observed = {
            "log_parameters": _bits(tuple(model.log_parameters)),
            "mean": _bits(tuple(model.predict(p) for p in probe)),
            "variance": _bits(tuple(model.predict_with_variance(p)[1] for p in probe)),
            "condition": _bits(model.condition_number),
        }
        self.assertEqual(observed, GOLDEN_FIT)


#: Recorded from git revision 315d501 (before the rewrite).
GOLDEN_FIT = {
    "log_parameters": (
        "0x1.fea05c6f195a4p+2",
        "0x1.db0cc838407e3p+0",
        "0x1.1a3f49ddf7675p+2",
        "0x1.17840458dc4eap+2",
        "-0x1.0971e279a5b5fp+3",
    ),
    "mean": ("0x1.c29fceaa46997p+0", "0x1.eefc623490815p+0"),
    "variance": ("0x1.7339dc421ddb2p-15", "0x1.2befd15d157f1p-12"),
    "condition": "0x1.207953d9e722dp+29",
}


if __name__ == "__main__":
    unittest.main()
