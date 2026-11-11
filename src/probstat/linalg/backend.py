"""Selectable dense-kernel backends behind one tuple-in, tuple-out boundary.

The dense consumers whose cost grows as ``n^3`` (Gaussian-process fitting
and prediction, multivariate-normal densities and sampling) reach their
factorizations and solves through a :class:`DenseBackend`. Two backends are
provided:

* :class:`PythonBackend`, the default reference backend, wraps the
  dependency-free routines of :mod:`probstat.linalg.dense` unchanged, so
  results are bit-identical to calling those functions directly;
* :class:`NumPyBackend` delegates to NumPy's LAPACK-backed ``cholesky``,
  ``eigh``, and BLAS matrix products. NumPy is imported only when this
  backend is constructed, which happens only on explicit selection
  (``set_backend("numpy")`` or ``use_backend("numpy")``); it is an optional
  extra (``pip install probstat-library[numerics]``).

Both backends accept and return immutable tuples (``Vector`` and
``Matrix``); arrays never cross the boundary, so the public API is the same
whichever backend is active. Both apply the same jitter policy: the
diagonal shift ``delta`` of a factorization is an explicit argument, zero
by default, never chosen or enlarged by the backend [EQ-LINALG-010]. A
matrix that is not positive definite after the requested shift raises
``ValueError`` in both backends.

The active backend is process-wide state. :func:`use_backend` restores the
previous backend on exit, also when the body raises; it does not isolate
threads that run concurrently.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from probstat.foundations.validation import require_finite_real

from . import dense
from .dense import Matrix, Vector

PYTHON_BACKEND = "python"
"""Name of the dependency-free reference backend (the default)."""

NUMPY_BACKEND = "numpy"
"""Name of the optional NumPy backend."""

_NUMPY_HINT = (
    "the 'numpy' dense backend requires NumPy; install it with "
    "'pip install probstat-library[numerics]' or keep the default 'python' "
    "backend"
)


def _check_jitter(jitter: float) -> float:
    """Validate an explicit diagonal shift ``delta >= 0``.

    Args:
        jitter: Requested shift.

    Returns:
        The shift as a finite float.

    Raises:
        TypeError: If ``jitter`` is not real.
        ValueError: If it is negative or non-finite.

    References:
        [EQ-LINALG-010], project validation policy [REF-PROBSTAT-VALIDATION].

    """
    shift = require_finite_real(jitter, name="jitter")
    if shift < 0.0:
        raise ValueError("jitter must be non-negative")
    return shift


def _shifted(matrix: Matrix, shift: float) -> Matrix:
    """Return ``A + delta I`` for a square tuple matrix.

    Args:
        matrix: Square matrix.
        shift: Diagonal shift ``delta``.

    Returns:
        The shifted matrix.

    Equation:
        [EQ-LINALG-010].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    return tuple(
        tuple(
            value + shift if row == column else value
            for column, value in enumerate(values)
        )
        for row, values in enumerate(matrix)
    )


def _lower_times(lower: Matrix, vector: Vector) -> Vector:
    """Return ``L z`` reading only the lower triangle, with ``math.fsum`` sums.

    Args:
        lower: Lower-triangular matrix.
        vector: Vector of matching length.

    Returns:
        Product ``L z``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1], [REF-PYTHON-MATH].
        No named equation is needed.

    """
    return tuple(
        math.fsum(lower[row][column] * vector[column] for column in range(row + 1))
        for row in range(len(lower))
    )


@runtime_checkable
class CholeskyFactorization(Protocol):
    """A retained factorization ``A + delta I = L L'`` and its solves.

    A factorization keeps the backend's native storage, so repeated solves
    against one matrix (predictions, density evaluations, samples) do not
    convert the factor again. Every method returns tuples.

    References:
        [EQ-LINALG-001], [EQ-LINALG-010], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """

    @property
    def lower(self) -> Matrix:
        """Lower factor ``L`` as a square tuple matrix (zeros above the diagonal).

        References:
            [EQ-LINALG-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        ...

    @property
    def jitter(self) -> float:
        """Diagonal shift ``delta`` that was added before factorizing.

        References:
            [EQ-LINALG-010], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        ...

    def solve_lower(self, right_hand_side: Vector) -> Vector:
        """Return ``L^-1 b`` by forward substitution.

        References:
            [EQ-LINALG-002], [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        ...

    def solve(self, right_hand_side: Vector) -> Vector:
        """Return ``(L L')^-1 b`` by forward and back substitution.

        References:
            [EQ-LINALG-002], [EQ-LINALG-003], [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        ...

    def multiply_lower(self, vector: Vector) -> Vector:
        """Return ``L z`` (the Gaussian sampling transform).

        References:
            [EQ-DIST-MVN-005], [REF-GOLUB-VANLOAN-2013, Section 1.1].

        """
        ...

    def log_det(self) -> float:
        """Return ``log det (L L') = 2 sum_i log L_ii``.

        References:
            [EQ-LINALG-006], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        ...

    def inverse(self) -> Matrix:
        """Return the symmetrized inverse ``L^-T L^-1``.

        References:
            [EQ-LINALG-011], [REF-HIGHAM-2002, Chapter 14].

        """
        ...


@runtime_checkable
class DenseBackend(Protocol):
    """The dense kernels that the ``n^3`` consumers of the library need.

    An implementation accepts and returns tuple vectors and matrices only,
    raises ``ValueError`` for a matrix that is not positive definite (after
    the explicit ``jitter`` shift) or a singular triangular matrix, and never
    regularizes silently.

    References:
        [EQ-LINALG-001], [EQ-LINALG-010], [REF-GOLUB-VANLOAN-2013, Chapters 1-4],
        [REF-LAPACK-1999, Chapter 2].

    """

    @property
    def name(self) -> str:
        """Registry name of the backend.

        References:
            [REF-LAPACK-1999, Chapter 2]. No equation is evaluated.

        """
        ...

    def cholesky(self, matrix: Matrix, *, jitter: float = 0.0) -> Matrix:
        """Return the lower factor of ``A + delta I``.

        References:
            [EQ-LINALG-001], [EQ-LINALG-010], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        ...

    def factorize(
        self, matrix: Matrix, *, jitter: float = 0.0
    ) -> CholeskyFactorization:
        """Return a retained factorization of ``A + delta I``.

        References:
            [EQ-LINALG-001], [EQ-LINALG-010], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        ...

    def solve_lower(self, lower: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``L x = b`` by forward substitution.

        References:
            [EQ-LINALG-002], [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        ...

    def solve_upper(self, upper: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``U x = b`` by back substitution.

        References:
            [EQ-LINALG-003], [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        ...

    def cholesky_solve(self, lower: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``L L' x = b`` from a lower factor.

        References:
            [EQ-LINALG-002], [EQ-LINALG-003], [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        ...

    def solve_spd(self, matrix: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``A x = b`` for symmetric positive-definite ``A``.

        References:
            [EQ-LINALG-001], [EQ-LINALG-002], [EQ-LINALG-003],
            [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

        """
        ...

    def log_det_spd(self, matrix: Matrix) -> float:
        """Return ``log det A`` for symmetric positive-definite ``A``.

        References:
            [EQ-LINALG-006], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        ...

    def inverse_spd(self, matrix: Matrix) -> Matrix:
        """Return the symmetrized inverse of symmetric positive-definite ``A``.

        References:
            [EQ-LINALG-011], [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

        """
        ...

    def matmul(self, left: Matrix, right: Matrix) -> Matrix:
        """Return the product ``A B``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

        """
        ...

    def matvec(self, matrix: Matrix, vector: Vector) -> Vector:
        """Return the product ``A x``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

        """
        ...

    def symmetric_eigen(self, matrix: Matrix) -> tuple[Vector, Matrix]:
        """Return ascending eigenvalues and eigenvector columns of symmetric ``A``.

        References:
            [EQ-LINALG-005], [REF-GOLUB-VANLOAN-2013, Section 8.5].

        """
        ...


class _PythonCholesky:
    """Reference factorization holding a tuple factor.

    References:
        [EQ-LINALG-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """

    __slots__ = ("_jitter", "_lower")

    def __init__(self, lower: Matrix, jitter: float) -> None:
        """Store the factor and the shift that produced it.

        Args:
            lower: Lower Cholesky factor.
            jitter: Diagonal shift ``delta``.

        References:
            [EQ-LINALG-010], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        self._lower = lower
        self._jitter = jitter

    @property
    def lower(self) -> Matrix:
        """Lower factor ``L``.

        References:
            [EQ-LINALG-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        return self._lower

    @property
    def jitter(self) -> float:
        """Diagonal shift ``delta``.

        References:
            [EQ-LINALG-010], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        return self._jitter

    def solve_lower(self, right_hand_side: Vector) -> Vector:
        """Return ``L^-1 b`` through :func:`probstat.linalg.solve_lower`.

        References:
            [EQ-LINALG-002], [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        return dense.solve_lower(self._lower, right_hand_side)

    def solve(self, right_hand_side: Vector) -> Vector:
        """Return ``(L L')^-1 b`` through :func:`probstat.linalg.cholesky_solve`.

        References:
            [EQ-LINALG-002], [EQ-LINALG-003], [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        return dense.cholesky_solve(self._lower, right_hand_side)

    def multiply_lower(self, vector: Vector) -> Vector:
        """Return ``L z`` over the lower triangle with ``math.fsum`` sums.

        References:
            [EQ-DIST-MVN-005], [REF-GOLUB-VANLOAN-2013, Section 1.1].

        """
        return _lower_times(self._lower, vector)

    def log_det(self) -> float:
        """Return ``2 sum_i log L_ii`` with a ``math.fsum`` sum.

        References:
            [EQ-LINALG-006], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        return 2.0 * math.fsum(
            math.log(self._lower[i][i]) for i in range(len(self._lower))
        )

    def inverse(self) -> Matrix:
        """Return ``(L L')^-1`` by unit-vector Cholesky solves.

        References:
            [EQ-LINALG-011], [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

        """
        return dense._inverse_from_cholesky(self._lower)


class PythonBackend:
    """Dependency-free reference backend (the default).

    Every method calls the routine of the same name in
    :mod:`probstat.linalg.dense`, so results are bit-identical to calling
    that routine directly; a zero ``jitter`` passes the matrix through
    unchanged.

    References:
        [EQ-LINALG-001], [REF-GOLUB-VANLOAN-2013, Chapters 1-4 and 8].

    """

    __slots__ = ()

    @property
    def name(self) -> str:
        """Registry name ``"python"``.

        References:
            [REF-PYTHON-MATH]. No equation is evaluated.

        """
        return PYTHON_BACKEND

    def __repr__(self) -> str:
        """Return ``PythonBackend()``.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return "PythonBackend()"

    def cholesky(self, matrix: Matrix, *, jitter: float = 0.0) -> Matrix:
        """Return the lower factor of ``A + delta I`` via :func:`linalg.cholesky`.

        Args:
            matrix: Symmetric positive-definite matrix (lower triangle read).
            jitter: Explicit diagonal shift ``delta >= 0``.

        Returns:
            Lower factor as a tuple matrix.

        Raises:
            ValueError: If ``A + delta I`` is not positive definite or the
                shift is negative.

        Equation:
            [EQ-LINALG-001], [EQ-LINALG-010].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        shift = _check_jitter(jitter)
        return dense.cholesky(_shifted(matrix, shift) if shift else matrix)

    def factorize(
        self, matrix: Matrix, *, jitter: float = 0.0
    ) -> CholeskyFactorization:
        """Return a retained reference factorization of ``A + delta I``.

        Args:
            matrix: Symmetric positive-definite matrix.
            jitter: Explicit diagonal shift ``delta >= 0``.

        Returns:
            Factorization with tuple results.

        Raises:
            ValueError: If ``A + delta I`` is not positive definite.

        Equation:
            [EQ-LINALG-001], [EQ-LINALG-010].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        shift = _check_jitter(jitter)
        return _PythonCholesky(self.cholesky(matrix, jitter=shift), shift)

    def solve_lower(self, lower: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``L x = b`` via :func:`probstat.linalg.solve_lower`.

        Equation:
            [EQ-LINALG-002].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        return dense.solve_lower(lower, right_hand_side)

    def solve_upper(self, upper: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``U x = b`` via :func:`probstat.linalg.solve_upper`.

        Equation:
            [EQ-LINALG-003].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        return dense.solve_upper(upper, right_hand_side)

    def cholesky_solve(self, lower: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``L L' x = b`` via :func:`probstat.linalg.cholesky_solve`.

        Equation:
            [EQ-LINALG-002], [EQ-LINALG-003].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        return dense.cholesky_solve(lower, right_hand_side)

    def solve_spd(self, matrix: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``A x = b`` via :func:`probstat.linalg.solve_spd`.

        Equation:
            [EQ-LINALG-001], [EQ-LINALG-002], [EQ-LINALG-003].

        References:
            [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

        """
        return dense.solve_spd(matrix, right_hand_side)

    def log_det_spd(self, matrix: Matrix) -> float:
        """Return ``log det A`` via :func:`probstat.linalg.log_det_spd`.

        Equation:
            [EQ-LINALG-006].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        return dense.log_det_spd(matrix)

    def inverse_spd(self, matrix: Matrix) -> Matrix:
        """Return ``A^-1`` via :func:`probstat.linalg.inverse_spd`.

        Equation:
            [EQ-LINALG-011].

        References:
            [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

        """
        return dense.inverse_spd(matrix)

    def matmul(self, left: Matrix, right: Matrix) -> Matrix:
        """Return ``A B`` via :func:`probstat.linalg.matmul`.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

        """
        return dense.matmul(left, right)

    def matvec(self, matrix: Matrix, vector: Vector) -> Vector:
        """Return ``A x`` via :func:`probstat.linalg.matvec`.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

        """
        return dense.matvec(matrix, vector)

    def symmetric_eigen(self, matrix: Matrix) -> tuple[Vector, Matrix]:
        """Diagonalize via :func:`probstat.linalg.symmetric_eigen` (Jacobi).

        Equation:
            [EQ-LINALG-005].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 8.5].

        """
        return dense.symmetric_eigen(matrix)


def _import_numpy() -> Any:
    """Import NumPy on demand and return the module.

    Returns:
        The ``numpy`` module, typed ``Any`` so that type checking does not
        depend on whether NumPy is installed.

    Raises:
        ImportError: If NumPy cannot be imported, with installation advice.

    References:
        [REF-HARRIS-ETAL-2020]. No equation is evaluated.

    """
    try:
        return importlib.import_module("numpy")
    except ImportError as error:
        raise ImportError(_NUMPY_HINT) from error


def _matrix_from(array: Any) -> Matrix:
    """Convert a two-dimensional float array to a tuple matrix.

    References:
        [REF-HARRIS-ETAL-2020]. No equation is evaluated.

    """
    rows: list[list[float]] = array.tolist()
    return tuple(tuple(row) for row in rows)


def _vector_from(array: Any) -> Vector:
    """Convert a one-dimensional float array to a tuple vector.

    References:
        [REF-HARRIS-ETAL-2020]. No equation is evaluated.

    """
    values: list[float] = array.tolist()
    return tuple(values)


class _NumPyKernels:
    """Array-level kernels shared by the NumPy backend and its factorization.

    References:
        [REF-HARRIS-ETAL-2020], [REF-LAPACK-1999, Chapter 2].

    """

    __slots__ = ()

    @property
    def np(self) -> Any:
        """The NumPy module, looked up in ``sys.modules`` on each access.

        Holding no module reference keeps factorizations picklable and
        copyable.

        References:
            [REF-HARRIS-ETAL-2020]. No equation is evaluated.

        """
        return _import_numpy()

    def square(self, matrix: Matrix | Sequence[Sequence[float]]) -> Any:
        """Return a square ``float64`` array or raise ``ValueError``.

        References:
            [REF-HARRIS-ETAL-2020]. No equation is evaluated.

        """
        array = self.np.asarray(matrix, dtype=self.np.float64)
        if array.ndim != 2 or array.shape[0] != array.shape[1] or array.size == 0:
            raise ValueError("matrix must be non-empty and square")
        return array

    def vector(self, values: Vector, size: int) -> Any:
        """Return a ``float64`` right-hand side of length ``size``.

        References:
            [REF-HARRIS-ETAL-2020]. No equation is evaluated.

        """
        array = self.np.asarray(values, dtype=self.np.float64)
        if array.ndim != 1 or array.shape[0] != size:
            raise ValueError("right-hand side length must match the matrix")
        return array

    def cholesky(self, matrix: Matrix, shift: float) -> Any:
        """Return the LAPACK lower factor of ``A + delta I`` as an array.

        Raises:
            ValueError: If the shifted matrix is not positive definite or the
                factor is not finite.

        Equation:
            [EQ-LINALG-001], [EQ-LINALG-010].

        References:
            [REF-LAPACK-1999, Chapter 2], [REF-HIGHAM-2002, Chapter 10].

        """
        np = self.np
        array = self.square(matrix)
        if shift:
            array = array + shift * np.eye(array.shape[0])
        try:
            lower = np.linalg.cholesky(array)
        except np.linalg.LinAlgError as error:
            raise ValueError("matrix must be positive definite") from error
        if not np.isfinite(lower).all():
            raise ValueError("matrix must be positive definite")
        return np.tril(lower)

    def forward(self, lower: Any, right_hand_side: Any) -> Any:
        """Solve ``L X = B`` row by row (one vector product per row).

        ``B`` may be a vector or a matrix of right-hand sides; the result has
        its shape.

        Raises:
            ValueError: If a diagonal entry is zero.

        Equation:
            [EQ-LINALG-002].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 3.1], [REF-HIGHAM-2002, Chapter 8].

        """
        np = self.np
        diagonal = np.diagonal(lower)
        if (diagonal == 0.0).any():
            raise ValueError("triangular matrix is singular")
        solution = np.zeros(right_hand_side.shape, dtype=np.float64)
        for row in range(lower.shape[0]):
            solution[row] = (
                right_hand_side[row] - lower[row, :row] @ solution[:row]
            ) / diagonal[row]
        return solution

    def backward(self, upper: Any, right_hand_side: Any) -> Any:
        """Solve ``U X = B`` row by row from the last row upward.

        Raises:
            ValueError: If a diagonal entry is zero.

        Equation:
            [EQ-LINALG-003].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 3.1], [REF-HIGHAM-2002, Chapter 8].

        """
        np = self.np
        diagonal = np.diagonal(upper)
        if (diagonal == 0.0).any():
            raise ValueError("triangular matrix is singular")
        size = upper.shape[0]
        solution = np.zeros(right_hand_side.shape, dtype=np.float64)
        for row in range(size - 1, -1, -1):
            solution[row] = (
                right_hand_side[row] - upper[row, row + 1 :] @ solution[row + 1 :]
            ) / diagonal[row]
        return solution

    def inverse(self, lower: Any) -> Any:
        """Return ``L^-T L^-1``, symmetrized, from a lower factor array.

        Equation:
            [EQ-LINALG-011].

        References:
            [REF-HIGHAM-2002, Chapter 14], [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        np = self.np
        inverse_lower = self.forward(lower, np.eye(lower.shape[0]))
        product = inverse_lower.T @ inverse_lower
        return 0.5 * (product + product.T)

    def log_det(self, lower: Any) -> float:
        """Return ``2 sum_i log L_ii`` with a ``math.fsum`` sum.

        Equation:
            [EQ-LINALG-006].

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2], [REF-PYTHON-MATH].

        """
        diagonal: list[float] = self.np.log(self.np.diagonal(lower)).tolist()
        return 2.0 * math.fsum(diagonal)


class _NumPyCholesky:
    """NumPy factorization keeping the factor as an array.

    References:
        [EQ-LINALG-001], [REF-LAPACK-1999, Chapter 2].

    """

    __slots__ = ("_array", "_jitter", "_kernels", "_lower")

    def __init__(self, kernels: _NumPyKernels, array: Any, jitter: float) -> None:
        """Store the array factor and its tuple view.

        References:
            [EQ-LINALG-010], [REF-HARRIS-ETAL-2020].

        """
        self._kernels = kernels
        self._array = array
        self._jitter = jitter
        self._lower = _matrix_from(array)

    @property
    def lower(self) -> Matrix:
        """Lower factor ``L`` as a tuple matrix.

        References:
            [EQ-LINALG-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        return self._lower

    @property
    def jitter(self) -> float:
        """Diagonal shift ``delta``.

        References:
            [EQ-LINALG-010], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        return self._jitter

    def solve_lower(self, right_hand_side: Vector) -> Vector:
        """Return ``L^-1 b`` by row-wise forward substitution.

        References:
            [EQ-LINALG-002], [REF-HIGHAM-2002, Chapter 8].

        """
        kernels = self._kernels
        target = kernels.vector(right_hand_side, self._array.shape[0])
        return _vector_from(kernels.forward(self._array, target))

    def solve(self, right_hand_side: Vector) -> Vector:
        """Return ``(L L')^-1 b`` by forward and back substitution.

        References:
            [EQ-LINALG-002], [EQ-LINALG-003], [REF-HIGHAM-2002, Chapter 8].

        """
        kernels = self._kernels
        target = kernels.vector(right_hand_side, self._array.shape[0])
        forward = kernels.forward(self._array, target)
        return _vector_from(kernels.backward(self._array.T, forward))

    def multiply_lower(self, vector: Vector) -> Vector:
        """Return ``L z`` by one BLAS matrix-vector product.

        References:
            [EQ-DIST-MVN-005], [REF-HIGHAM-2002, Chapter 3].

        """
        kernels = self._kernels
        return _vector_from(
            self._array @ kernels.vector(vector, self._array.shape[0])
        )

    def log_det(self) -> float:
        """Return ``2 sum_i log L_ii``.

        References:
            [EQ-LINALG-006], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        return self._kernels.log_det(self._array)

    def inverse(self) -> Matrix:
        """Return ``L^-T L^-1`` from one triangular inversion.

        References:
            [EQ-LINALG-011], [REF-HIGHAM-2002, Chapter 14].

        """
        return _matrix_from(self._kernels.inverse(self._array))


class NumPyBackend:
    """Optional backend on NumPy's LAPACK and BLAS kernels.

    Cholesky factorization calls ``numpy.linalg.cholesky`` (LAPACK
    ``potrf`` reading the lower triangle) and symmetric eigendecomposition
    ``numpy.linalg.eigh``; matrix products use BLAS. NumPy has no
    triangular solver, so forward and back substitution run one row at a
    time with a vector product per row: the same backward-stable algorithm
    as the reference backend [REF-HIGHAM-2002, Chapter 8], with ``n``
    interpreter steps instead of ``n^2``. Inputs and outputs are tuples;
    eigenvectors may differ from the reference backend in sign.

    Raises:
        ImportError: On construction, if NumPy is not installed.

    References:
        [REF-HARRIS-ETAL-2020], [REF-LAPACK-1999, Chapter 2],
        [REF-HIGHAM-2002, Chapters 3, 8, and 10].

    """

    __slots__ = ("_kernels",)

    def __init__(self) -> None:
        """Import NumPy now; fail with installation advice if it is absent.

        Raises:
            ImportError: If NumPy cannot be imported.

        References:
            [REF-HARRIS-ETAL-2020]. No equation is evaluated.

        """
        _import_numpy()
        self._kernels = _NumPyKernels()

    @property
    def name(self) -> str:
        """Registry name ``"numpy"``.

        References:
            [REF-HARRIS-ETAL-2020]. No equation is evaluated.

        """
        return NUMPY_BACKEND

    def __repr__(self) -> str:
        """Return ``NumPyBackend()``.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return "NumPyBackend()"

    def cholesky(self, matrix: Matrix, *, jitter: float = 0.0) -> Matrix:
        """Return the lower factor of ``A + delta I`` via LAPACK.

        Args:
            matrix: Symmetric positive-definite matrix (lower triangle read).
            jitter: Explicit diagonal shift ``delta >= 0``.

        Returns:
            Lower factor as a tuple matrix.

        Raises:
            ValueError: If ``A + delta I`` is not positive definite, the
                matrix is not square, or the shift is negative.

        Equation:
            [EQ-LINALG-001], [EQ-LINALG-010].

        References:
            [REF-LAPACK-1999, Chapter 2].

        """
        shift = _check_jitter(jitter)
        return _matrix_from(self._kernels.cholesky(matrix, shift))

    def factorize(
        self, matrix: Matrix, *, jitter: float = 0.0
    ) -> CholeskyFactorization:
        """Return a retained LAPACK factorization of ``A + delta I``.

        Args:
            matrix: Symmetric positive-definite matrix.
            jitter: Explicit diagonal shift ``delta >= 0``.

        Returns:
            Factorization that keeps the array factor.

        Raises:
            ValueError: If ``A + delta I`` is not positive definite.

        Equation:
            [EQ-LINALG-001], [EQ-LINALG-010].

        References:
            [REF-LAPACK-1999, Chapter 2].

        """
        shift = _check_jitter(jitter)
        kernels = self._kernels
        return _NumPyCholesky(kernels, kernels.cholesky(matrix, shift), shift)

    def solve_lower(self, lower: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``L x = b`` by row-wise forward substitution.

        Equation:
            [EQ-LINALG-002].

        References:
            [REF-HIGHAM-2002, Chapter 8].

        """
        kernels = self._kernels
        array = kernels.square(lower)
        target = kernels.vector(right_hand_side, array.shape[0])
        return _vector_from(kernels.forward(kernels.np.tril(array), target))

    def solve_upper(self, upper: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``U x = b`` by row-wise back substitution.

        Equation:
            [EQ-LINALG-003].

        References:
            [REF-HIGHAM-2002, Chapter 8].

        """
        kernels = self._kernels
        array = kernels.square(upper)
        target = kernels.vector(right_hand_side, array.shape[0])
        return _vector_from(kernels.backward(kernels.np.triu(array), target))

    def cholesky_solve(self, lower: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``L L' x = b`` by forward and back substitution.

        Equation:
            [EQ-LINALG-002], [EQ-LINALG-003].

        References:
            [REF-HIGHAM-2002, Chapter 8].

        """
        kernels = self._kernels
        array = kernels.np.tril(kernels.square(lower))
        target = kernels.vector(right_hand_side, array.shape[0])
        return _vector_from(
            kernels.backward(array.T, kernels.forward(array, target))
        )

    def solve_spd(self, matrix: Matrix, right_hand_side: Vector) -> Vector:
        """Solve ``A x = b`` through a LAPACK Cholesky factor.

        Raises:
            ValueError: If the matrix is not positive definite.

        Equation:
            [EQ-LINALG-001], [EQ-LINALG-002], [EQ-LINALG-003].

        References:
            [REF-LAPACK-1999, Chapter 2], [REF-HIGHAM-2002, Chapter 10].

        """
        return self.factorize(matrix).solve(right_hand_side)

    def log_det_spd(self, matrix: Matrix) -> float:
        """Return ``log det A`` from a LAPACK Cholesky factor.

        Raises:
            ValueError: If the matrix is not positive definite.

        Equation:
            [EQ-LINALG-006].

        References:
            [REF-LAPACK-1999, Chapter 2].

        """
        kernels = self._kernels
        return kernels.log_det(kernels.cholesky(matrix, 0.0))

    def inverse_spd(self, matrix: Matrix) -> Matrix:
        """Return ``A^-1 = L^-T L^-1`` from a LAPACK Cholesky factor.

        Raises:
            ValueError: If the matrix is not positive definite.

        Equation:
            [EQ-LINALG-011].

        References:
            [REF-HIGHAM-2002, Chapter 14].

        """
        kernels = self._kernels
        return _matrix_from(kernels.inverse(kernels.cholesky(matrix, 0.0)))

    def matmul(self, left: Matrix, right: Matrix) -> Matrix:
        """Return ``A B`` by one BLAS matrix product.

        Raises:
            ValueError: If the inner dimensions disagree.

        References:
            [REF-HIGHAM-2002, Chapter 3]. No named equation is needed.

        """
        np = self._kernels.np
        first = np.asarray(left, dtype=np.float64)
        second = np.asarray(right, dtype=np.float64)
        if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[0]:
            raise ValueError("inner matrix dimensions must agree")
        return _matrix_from(first @ second)

    def matvec(self, matrix: Matrix, vector: Vector) -> Vector:
        """Return ``A x`` by one BLAS matrix-vector product.

        Raises:
            ValueError: If the dimensions disagree.

        References:
            [REF-HIGHAM-2002, Chapter 3]. No named equation is needed.

        """
        np = self._kernels.np
        array = np.asarray(matrix, dtype=np.float64)
        values = np.asarray(vector, dtype=np.float64)
        if array.ndim != 2 or values.ndim != 1 or array.shape[1] != values.shape[0]:
            raise ValueError("vectors must have equal length")
        return _vector_from(array @ values)

    def symmetric_eigen(self, matrix: Matrix) -> tuple[Vector, Matrix]:
        """Diagonalize a symmetric matrix by LAPACK through ``numpy.linalg.eigh``.

        Returns:
            Ascending eigenvalues and a matrix whose columns are the
            orthonormal eigenvectors (signs may differ from Jacobi).

        Equation:
            [EQ-LINALG-005].

        References:
            [REF-LAPACK-1999, Chapter 2], [REF-GOLUB-VANLOAN-2013, Chapter 8].

        """
        kernels = self._kernels
        values, vectors = kernels.np.linalg.eigh(kernels.square(matrix))
        return _vector_from(values), _matrix_from(vectors)


_FACTORIES: dict[str, type[PythonBackend] | type[NumPyBackend]] = {
    PYTHON_BACKEND: PythonBackend,
    NUMPY_BACKEND: NumPyBackend,
}

_active: list[DenseBackend] = [PythonBackend()]


def available_backends() -> tuple[str, ...]:
    """Return the registered backend names (not a check that NumPy imports).

    Returns:
        ``("python", "numpy")``.

    References:
        [REF-HARRIS-ETAL-2020]. No equation is evaluated.

    """
    return tuple(_FACTORIES)


def get_backend() -> DenseBackend:
    """Return the active dense backend (initially :class:`PythonBackend`).

    Returns:
        The backend that dense consumers use now.

    References:
        [REF-PROBSTAT-API]. No equation is evaluated.

    """
    return _active[0]


def _resolve(backend: str | DenseBackend) -> DenseBackend:
    """Return a backend instance for a registry name or a backend object.

    Raises:
        ImportError: If ``"numpy"`` is requested and NumPy is absent.
        TypeError: If ``backend`` is neither a name nor a backend.
        ValueError: If the name is not registered.

    References:
        [REF-PROBSTAT-API]. No equation is evaluated.

    """
    if isinstance(backend, str):
        factory = _FACTORIES.get(backend.strip().lower())
        if factory is None:
            raise ValueError(
                f"unknown dense backend {backend!r}; choose one of "
                f"{available_backends()}"
            )
        return factory()
    if isinstance(backend, DenseBackend):
        return backend
    raise TypeError("backend must be a backend name or a DenseBackend")


def set_backend(backend: str | DenseBackend) -> DenseBackend:
    """Select the dense backend for the whole process.

    Args:
        backend: ``"python"``, ``"numpy"``, or an object implementing
            :class:`DenseBackend`.

    Returns:
        The previously active backend, so that a caller can restore it.

    Raises:
        ImportError: If ``"numpy"`` is requested and NumPy is not installed;
            the active backend is then unchanged.
        TypeError: If ``backend`` is neither a name nor a backend.
        ValueError: If the name is not registered.

    References:
        [REF-PROBSTAT-API], [REF-HARRIS-ETAL-2020]. No equation is evaluated.

    """
    resolved = _resolve(backend)
    previous = _active[0]
    _active[0] = resolved
    return previous


@contextmanager
def use_backend(backend: str | DenseBackend) -> Iterator[DenseBackend]:
    """Activate a dense backend for the duration of a ``with`` block.

    The previous backend is restored on exit, also when the block raises.

    Args:
        backend: ``"python"``, ``"numpy"``, or a :class:`DenseBackend`.

    Yields:
        The backend active inside the block.

    Raises:
        ImportError: If ``"numpy"`` is requested and NumPy is not installed
            (raised on entry; the active backend is unchanged).
        TypeError: If ``backend`` is neither a name nor a backend.
        ValueError: If the name is not registered.

    References:
        [REF-PROBSTAT-API]. No equation is evaluated.

    """
    previous = set_backend(backend)
    try:
        yield _active[0]
    finally:
        _active[0] = previous
