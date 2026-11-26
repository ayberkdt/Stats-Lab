"""Matrix-free linear operators, their transposes, and operator algebra.

A :class:`LinearOperator` represents a linear map ``A: R^n -> R^m`` by its two
actions ``x -> A x`` (``matvec``) and ``y -> A'y`` (``rmatvec``, the
EUCLIDEAN transpose). Nothing here allocates an ``m x n`` array unless
:func:`materialize` is called explicitly, so a field with ``10^5`` unknowns
and a stencil or convolution operator costs ``O(n)`` memory per application.

The algebra (:class:`ComposedOperator`, :class:`SumOperator`,
:class:`ScaledOperator`, :class:`VStackOperator`, :class:`HStackOperator`,
:class:`BlockDiagonalOperator`, :class:`TransposedOperator`) builds the
operators of an inverse problem, for example the stacked Tikhonov operator
``[W A; lambda L]`` whose transpose is ``u, v -> A'W'u + lambda L'v``
[EQ-LINALG-OP-002]. A weighted (non-Euclidean) adjoint
``A* = M_X^-1 A' M_Y`` is never the same as ``rmatvec``; it is built
explicitly by :func:`weighted_adjoint` from SPD metric objects.

Every public action validates the input length and finiteness and rejects a
non-finite or wrongly sized callback output; no broadcasting is performed.
:class:`CountingOperator` counts forward and transpose applications separately
so that solvers can report their work exactly.

Experimental (added in 0.28): the classes may change before they are declared
stable.

References:
    [REF-GOLUB-VANLOAN-2013, Sections 1.1 and 10.4],
    [REF-GAZZOLA-HANSEN-NAGY-2019].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .dense import Matrix, Vector

_DEFAULT_MATERIALIZE_LIMIT = 1_000_000


def _vector(values: Sequence[float], size: int, name: str) -> Vector:
    """Return ``values`` as a finite float tuple of the given length.

    Args:
        values: Candidate vector.
        size: Required length.
        name: Name used in error messages.

    Returns:
        Tuple of finite floats.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If the length differs or an entry is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a sequence of real numbers") from error
    if len(vector) != size:
        raise ValueError(f"{name} has length {len(vector)}, expected {size}")
    if not all(map(math.isfinite, vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _require_dimension(value: object, name: str, *, allow_zero: bool = False) -> int:
    """Validate an operator dimension.

    Args:
        value: Candidate integer.
        name: Name used in error messages.
        allow_zero: Whether ``0`` is accepted.

    Returns:
        The validated integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If it is negative, or zero when not allowed.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be positive")
    return value


def _real(value: object, name: str) -> float:
    """Return a finite real scalar.

    Args:
        value: Candidate scalar.
        name: Name used in error messages.

    Returns:
        The scalar as a float.

    Raises:
        TypeError: If it is not real.
        ValueError: If it is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@runtime_checkable
class LinearOperator(Protocol):
    """Structural protocol of a real linear map ``A: R^n -> R^m``.

    ``shape`` is ``(m, n)``; ``matvec`` applies ``A`` and ``rmatvec`` the
    Euclidean transpose ``A'``. Implementations must satisfy the adjoint
    identity ``y'(A x) = (A'y)'x`` [EQ-LINALG-OP-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1], [REF-GAZZOLA-HANSEN-NAGY-2019].

    """

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(m, n)``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

        """
        ...

    def matvec(self, vector: Sequence[float]) -> Vector:
        """Return ``A x`` for ``x`` of length ``n``.

        References:
            [EQ-LINALG-OP-001].

        """
        ...

    def rmatvec(self, vector: Sequence[float]) -> Vector:
        """Return ``A'y`` for ``y`` of length ``m``.

        References:
            [EQ-LINALG-OP-001].

        """
        ...


class OperatorBase:
    """Validated base class of the concrete operators in this module.

    Subclasses implement ``_forward`` and ``_transpose`` on validated tuples;
    the public ``matvec`` and ``rmatvec`` check the input length and
    finiteness and reject a wrongly sized or non-finite result.

    Attributes:
        name: Short description used in ``repr`` and error messages.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1], [EQ-LINALG-OP-001].

    """

    __slots__ = ("_shape", "name")

    def __init__(self, rows: int, columns: int, *, name: str) -> None:
        """Store the validated shape.

        Args:
            rows: ``m``; zero is allowed (an empty range).
            columns: ``n``; zero is allowed (an empty domain).
            name: Description used in messages.

        Raises:
            TypeError: If a dimension is not an integer.
            ValueError: If a dimension is negative.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        self._shape = (
            _require_dimension(rows, "rows", allow_zero=True),
            _require_dimension(columns, "columns", allow_zero=True),
        )
        self.name = name

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(m, n)``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

        """
        return self._shape

    def matvec(self, vector: Sequence[float]) -> Vector:
        """Return ``A x``.

        Args:
            vector: Finite vector of length ``n``.

        Returns:
            Finite vector of length ``m``.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If the length is wrong or a value (input or output)
                is not finite.

        References:
            [EQ-LINALG-OP-001].

        """
        rows, columns = self._shape
        checked = _vector(vector, columns, f"{self.name} input")
        return _vector(self._forward(checked), rows, f"{self.name} output")

    def rmatvec(self, vector: Sequence[float]) -> Vector:
        """Return ``A'y``.

        Args:
            vector: Finite vector of length ``m``.

        Returns:
            Finite vector of length ``n``.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If the length is wrong or a value (input or output)
                is not finite.

        References:
            [EQ-LINALG-OP-001].

        """
        rows, columns = self._shape
        checked = _vector(vector, rows, f"{self.name} transpose input")
        return _vector(
            self._transpose(checked), columns, f"{self.name} transpose output"
        )

    def transpose(self) -> TransposedOperator:
        """Return the operator ``A'`` (swapping ``matvec`` and ``rmatvec``).

        Returns:
            A :class:`TransposedOperator`.

        References:
            [EQ-LINALG-OP-001].

        """
        return TransposedOperator(self)

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Apply ``A`` to a validated vector (implemented by subclasses).

        Raises:
            NotImplementedError: Always, in the base class.

        References:
            [EQ-LINALG-OP-001].

        """
        raise NotImplementedError

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Apply ``A'`` to a validated vector (implemented by subclasses).

        Raises:
            NotImplementedError: Always, in the base class.

        References:
            [EQ-LINALG-OP-001].

        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a short description with the shape.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        return f"{type(self).__name__}({self.name!r}, shape={self._shape})"


def _apply(operator: LinearOperator, vector: Sequence[float]) -> Vector:
    """Apply any :class:`LinearOperator`, validating a foreign result.

    Args:
        operator: Operator to apply.
        vector: Input of length ``n``.

    Returns:
        ``A x`` as a validated tuple.

    References:
        [EQ-LINALG-OP-001].

    """
    if isinstance(operator, OperatorBase):
        return operator.matvec(vector)
    return _vector(operator.matvec(vector), operator.shape[0], "operator output")


def _apply_transpose(operator: LinearOperator, vector: Sequence[float]) -> Vector:
    """Apply the transpose of any :class:`LinearOperator`, validating the result.

    Args:
        operator: Operator whose transpose is applied.
        vector: Input of length ``m``.

    Returns:
        ``A'y`` as a validated tuple.

    References:
        [EQ-LINALG-OP-001].

    """
    if isinstance(operator, OperatorBase):
        return operator.rmatvec(vector)
    return _vector(
        operator.rmatvec(vector), operator.shape[1], "operator transpose output"
    )


def require_operator(operator: object, *, name: str = "operator") -> LinearOperator:
    """Return ``operator`` after checking that it is a linear operator.

    A dense row-sequence matrix is accepted and wrapped in
    :class:`DenseOperator`, so callers may pass either form.

    Args:
        operator: A :class:`LinearOperator` or a dense row-sequence matrix.
        name: Name used in error messages.

    Returns:
        A :class:`LinearOperator`.

    Raises:
        TypeError: If ``operator`` is neither.

    References:
        [EQ-LINALG-OP-001].

    """
    if isinstance(operator, LinearOperator) and not isinstance(operator, (tuple, list)):
        rows, columns = operator.shape
        _require_dimension(rows, f"{name} rows", allow_zero=True)
        _require_dimension(columns, f"{name} columns", allow_zero=True)
        return operator
    if isinstance(operator, (tuple, list)):
        return DenseOperator(operator, name=name)
    raise TypeError(f"{name} must be a LinearOperator or a dense matrix")


class DenseOperator(OperatorBase):
    """Operator of an explicit dense ``m x n`` matrix.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1], [EQ-LINALG-OP-001].

    """

    __slots__ = ("_columns", "_rows")

    def __init__(
        self, matrix: Sequence[Sequence[float]], *, name: str = "dense"
    ) -> None:
        """Validate and store the matrix by rows and by columns.

        Args:
            matrix: Non-empty rectangular finite matrix as rows.
            name: Description used in messages.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If the matrix is empty, ragged or non-finite.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        rows = tuple(tuple(row) for row in matrix)
        if not rows or not rows[0]:
            raise ValueError("matrix must be non-empty")
        width = len(rows[0])
        checked = tuple(
            _vector(row, width, f"{name} row {index}") for index, row in enumerate(rows)
        )
        super().__init__(len(checked), width, name=name)
        self._rows: Matrix = checked
        self._columns: Matrix = tuple(zip(*checked, strict=True))

    @property
    def matrix(self) -> Matrix:
        """Return the stored matrix as rows.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

        """
        return self._rows

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``A x`` with compensated row sums.

        References:
            [EQ-LINALG-OP-001].

        """
        return tuple(
            math.fsum(a * b for a, b in zip(row, vector, strict=True))
            for row in self._rows
        )

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return ``A'y`` with compensated column sums.

        References:
            [EQ-LINALG-OP-001].

        """
        return tuple(
            math.fsum(a * b for a, b in zip(column, vector, strict=True))
            for column in self._columns
        )


class IdentityOperator(OperatorBase):
    """The identity ``I_n``.

    References:
        [EQ-LINALG-OP-001].

    """

    __slots__ = ()

    def __init__(self, size: int) -> None:
        """Store ``n``.

        Args:
            size: Positive dimension.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        _require_dimension(size, "size")
        super().__init__(size, size, name="identity")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``x``.

        References:
            [EQ-LINALG-OP-001].

        """
        return vector

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return ``y``.

        References:
            [EQ-LINALG-OP-001].

        """
        return vector


class ZeroOperator(OperatorBase):
    """The zero map ``R^n -> R^m``.

    References:
        [EQ-LINALG-OP-001].

    """

    __slots__ = ()

    def __init__(self, rows: int, columns: int) -> None:
        """Store the shape.

        Args:
            rows: ``m`` (non-negative).
            columns: ``n`` (non-negative).

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        super().__init__(rows, columns, name="zero")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return the zero vector of length ``m``.

        References:
            [EQ-LINALG-OP-001].

        """
        return (0.0,) * self._shape[0]

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return the zero vector of length ``n``.

        References:
            [EQ-LINALG-OP-001].

        """
        return (0.0,) * self._shape[1]


class DiagonalOperator(OperatorBase):
    """Square diagonal operator ``diag(d)``.

    References:
        [EQ-LINALG-OP-001].

    """

    __slots__ = ("diagonal",)

    def __init__(self, diagonal: Sequence[float], *, name: str = "diagonal") -> None:
        """Store the diagonal.

        Args:
            diagonal: Non-empty finite diagonal entries.
            name: Description used in messages.

        Raises:
            ValueError: If the diagonal is empty or non-finite.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        values = tuple(float(value) for value in diagonal)
        if not values:
            raise ValueError("diagonal must be non-empty")
        self.diagonal = _vector(values, len(values), "diagonal")
        super().__init__(len(values), len(values), name=name)

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``d * x`` elementwise.

        References:
            [EQ-LINALG-OP-001].

        """
        return tuple(d * x for d, x in zip(self.diagonal, vector, strict=True))

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return ``d * y`` elementwise (a diagonal map is symmetric).

        References:
            [EQ-LINALG-OP-001].

        """
        return self._forward(vector)


class CallbackOperator(OperatorBase):
    """Operator defined by user callbacks for ``A x`` and ``A'y``.

    The callbacks receive validated tuples; their results are validated for
    length and finiteness. The adjoint identity is NOT checked automatically;
    run :func:`adjoint_test` on a new operator.

    References:
        [EQ-LINALG-OP-001].

    """

    __slots__ = ("_forward_callback", "_transpose_callback")

    def __init__(
        self,
        shape: tuple[int, int],
        matvec: Callable[[Vector], Sequence[float]],
        rmatvec: Callable[[Vector], Sequence[float]],
        *,
        name: str = "callback",
    ) -> None:
        """Store the shape and the two actions.

        Args:
            shape: ``(m, n)``.
            matvec: Callable returning ``A x``.
            rmatvec: Callable returning ``A'y``.
            name: Description used in messages.

        Raises:
            TypeError: If a callback is not callable or ``shape`` is malformed.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        if not callable(matvec) or not callable(rmatvec):
            raise TypeError("matvec and rmatvec must be callable")
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise TypeError("shape must be a pair (rows, columns)")
        super().__init__(shape[0], shape[1], name=name)
        self._forward_callback = matvec
        self._transpose_callback = rmatvec

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Call the forward callback.

        References:
            [EQ-LINALG-OP-001].

        """
        return self._forward_callback(vector)

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Call the transpose callback.

        References:
            [EQ-LINALG-OP-001].

        """
        return self._transpose_callback(vector)


class SelectionOperator(OperatorBase):
    """Restriction ``(S x)_i = x_(k_i)`` to selected indices.

    Repeated indices are allowed; the transpose then scatter-ADDS, so
    ``(S'y)_j = sum_(i: k_i = j) y_i``, which is the exact transpose.

    References:
        [EQ-LINALG-OP-001].

    """

    __slots__ = ("indices",)

    def __init__(self, size: int, indices: Sequence[int]) -> None:
        """Store the domain size and the indices.

        Args:
            size: Domain dimension ``n``.
            indices: Indices in ``[0, n)``; may be empty.

        Raises:
            TypeError: If an index is not an integer.
            ValueError: If an index is out of range.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        _require_dimension(size, "size")
        checked: list[int] = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("indices must be integers")
            if not 0 <= index < size:
                raise ValueError(f"index {index} outside [0, {size})")
            checked.append(index)
        self.indices = tuple(checked)
        super().__init__(len(checked), size, name="selection")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``x[k]``.

        References:
            [EQ-LINALG-OP-001].

        """
        return tuple(vector[index] for index in self.indices)

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Scatter-add ``y`` into a zero vector of length ``n``.

        References:
            [EQ-LINALG-OP-001].

        """
        result = [0.0] * self._shape[1]
        for value, index in zip(vector, self.indices, strict=True):
            result[index] += value
        return result


class PermutationOperator(OperatorBase):
    """Permutation ``(P x)_i = x_(p_i)`` with ``P' = P^-1``.

    References:
        [EQ-LINALG-OP-001].

    """

    __slots__ = ("permutation",)

    def __init__(self, permutation: Sequence[int]) -> None:
        """Store a permutation of ``0, ..., n - 1``.

        Args:
            permutation: Each index exactly once.

        Raises:
            ValueError: If it is not a permutation.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        values = tuple(permutation)
        if sorted(values) != list(range(len(values))) or not values:
            raise ValueError("permutation must contain each of 0..n-1 exactly once")
        self.permutation = values
        super().__init__(len(values), len(values), name="permutation")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``x[p]``.

        References:
            [EQ-LINALG-OP-001].

        """
        return tuple(vector[index] for index in self.permutation)

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return the inverse permutation applied to ``y``.

        References:
            [EQ-LINALG-OP-001].

        """
        result = [0.0] * len(vector)
        for value, index in zip(vector, self.permutation, strict=True):
            result[index] = value
        return result


_BOUNDARIES = ("none", "periodic", "dirichlet")


class FirstDifferenceOperator(OperatorBase):
    """Scaled first difference ``(D x)_i = (x_(i+1) - x_i) / h`` in one dimension.

    Boundary policies (all explicit, no hidden padding):

    - ``"none"``: ``(n - 1) x n``; the constants span the nullspace.
    - ``"periodic"``: ``n x n`` with ``x_n = x_0``; constants span the
      nullspace.
    - ``"dirichlet"``: ``(n + 1) x n`` with zero values outside, i.e.
      ``x_(-1) = x_n = 0``; full column rank.

    References:
        [REF-HANSEN-2008, Section 2.1], [EQ-LINALG-OP-003].

    """

    __slots__ = ("boundary", "spacing")

    def __init__(
        self, size: int, *, spacing: float = 1.0, boundary: str = "none"
    ) -> None:
        """Store the grid size, spacing and boundary policy.

        Args:
            size: Number of grid values ``n`` (at least 2 for ``"none"``).
            spacing: Positive grid spacing ``h``.
            boundary: ``"none"``, ``"periodic"`` or ``"dirichlet"``.

        Raises:
            ValueError: If an argument is invalid.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        _require_dimension(size, "size")
        if boundary not in _BOUNDARIES:
            raise ValueError(f"boundary must be one of {_BOUNDARIES}")
        if boundary == "none" and size < 2:
            raise ValueError("boundary 'none' needs at least two grid values")
        step = _real(spacing, "spacing")
        if step <= 0.0:
            raise ValueError("spacing must be positive")
        self.spacing = step
        self.boundary = boundary
        rows = {"none": size - 1, "periodic": size, "dirichlet": size + 1}[boundary]
        super().__init__(rows, size, name=f"first difference ({boundary})")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Apply the difference stencil.

        References:
            [EQ-LINALG-OP-003].

        """
        h = self.spacing
        n = len(vector)
        if self.boundary == "none":
            return tuple((vector[i + 1] - vector[i]) / h for i in range(n - 1))
        if self.boundary == "periodic":
            return tuple((vector[(i + 1) % n] - vector[i]) / h for i in range(n))
        padded = (0.0, *vector, 0.0)
        return tuple((padded[i + 1] - padded[i]) / h for i in range(n + 1))

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Apply the transposed stencil (a negative backward difference).

        References:
            [EQ-LINALG-OP-003].

        """
        h = self.spacing
        n = self._shape[1]
        result = [0.0] * n
        if self.boundary == "none":
            for i, value in enumerate(vector):
                result[i] -= value / h
                result[i + 1] += value / h
        elif self.boundary == "periodic":
            for i, value in enumerate(vector):
                result[i] -= value / h
                result[(i + 1) % n] += value / h
        else:
            for i, value in enumerate(vector):
                if i < n:
                    result[i] += value / h
                if i >= 1:
                    result[i - 1] -= value / h
        return result


class GridGradientOperator(OperatorBase):
    """Forward-difference gradient of a 2D grid function stored row-major.

    For an ``r x c`` image ``u`` (``u[i * c + j]``), the output is
    ``(D_x u, D_y u)`` stacked: first the ``r c`` horizontal differences
    ``(u[i, j+1] - u[i, j]) / h_x``, then the ``r c`` vertical differences
    ``(u[i+1, j] - u[i, j]) / h_y``, each set to zero on the last column or
    row (the Neumann convention of total-variation imaging). Constants span
    the nullspace.

    References:
        [REF-CHAMBOLLE-POCK-2011, Section 6.1], [EQ-LINALG-OP-003].

    """

    __slots__ = ("columns_count", "rows_count", "spacing")

    def __init__(
        self, rows: int, columns: int, *, spacing: tuple[float, float] = (1.0, 1.0)
    ) -> None:
        """Store the grid shape and the spacings ``(h_y, h_x)``.

        Args:
            rows: Number of grid rows ``r``.
            columns: Number of grid columns ``c``.
            spacing: Positive ``(h_y, h_x)``.

        Raises:
            ValueError: If an argument is invalid.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        _require_dimension(rows, "rows")
        _require_dimension(columns, "columns")
        hy, hx = (_real(value, "spacing") for value in spacing)
        if hy <= 0.0 or hx <= 0.0:
            raise ValueError("spacing must be positive")
        self.rows_count = rows
        self.columns_count = columns
        self.spacing = (hy, hx)
        size = rows * columns
        super().__init__(2 * size, size, name="grid gradient")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``(D_x u, D_y u)``.

        References:
            [EQ-LINALG-OP-003].

        """
        r, c = self.rows_count, self.columns_count
        hy, hx = self.spacing
        dx = [0.0] * (r * c)
        dy = [0.0] * (r * c)
        for i in range(r):
            base = i * c
            for j in range(c):
                k = base + j
                if j + 1 < c:
                    dx[k] = (vector[k + 1] - vector[k]) / hx
                if i + 1 < r:
                    dy[k] = (vector[k + c] - vector[k]) / hy
        return dx + dy

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return ``D_x' p + D_y' q`` (minus the discrete divergence).

        References:
            [EQ-LINALG-OP-003].

        """
        r, c = self.rows_count, self.columns_count
        hy, hx = self.spacing
        size = r * c
        result = [0.0] * size
        for i in range(r):
            base = i * c
            for j in range(c):
                k = base + j
                if j + 1 < c:
                    value = vector[k] / hx
                    result[k] -= value
                    result[k + 1] += value
                if i + 1 < r:
                    value = vector[size + k] / hy
                    result[k] -= value
                    result[k + c] += value
        return result


_CONVOLUTION_BOUNDARIES = ("zero", "periodic", "reflect")


class Convolution1DOperator(OperatorBase):
    """Discrete 1D convolution ``(K x)_i = sum_l k_l x_(i + l - c)`` on ``n`` points.

    ``c`` is the kernel centre index ``len(kernel) // 2``. Out-of-range
    indices follow the declared boundary: ``"zero"`` (dropped),
    ``"periodic"`` (wrapped) or ``"reflect"`` (mirrored about the end
    points, ``x_(-1) = x_0``). The transpose scatter-adds through the same
    index map, so it is exact for every boundary.

    References:
        [REF-HANSEN-2008, Section 2.1], [EQ-LINALG-OP-004].

    """

    __slots__ = ("boundary", "kernel", "_taps")

    def __init__(
        self, kernel: Sequence[float], size: int, *, boundary: str = "zero"
    ) -> None:
        """Store the kernel, the signal length and the boundary.

        Args:
            kernel: Non-empty finite kernel weights.
            size: Signal length ``n``.
            boundary: ``"zero"``, ``"periodic"`` or ``"reflect"``.

        Raises:
            ValueError: If an argument is invalid.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        _require_dimension(size, "size")
        weights = tuple(float(value) for value in kernel)
        if not weights:
            raise ValueError("kernel must be non-empty")
        self.kernel = _vector(weights, len(weights), "kernel")
        if boundary not in _CONVOLUTION_BOUNDARIES:
            raise ValueError(f"boundary must be one of {_CONVOLUTION_BOUNDARIES}")
        self.boundary = boundary
        centre = len(weights) // 2
        taps: list[tuple[tuple[int, int, float], ...]] = []
        for i in range(size):
            row: list[tuple[int, int, float]] = []
            for offset, weight in enumerate(self.kernel):
                j = i + offset - centre
                mapped = self._map(j, size)
                if mapped is not None and weight != 0.0:
                    row.append((i, mapped, weight))
            taps.append(tuple(row))
        self._taps = tuple(taps)
        super().__init__(size, size, name=f"convolution ({boundary})")

    def _map(self, index: int, size: int) -> int | None:
        """Map a possibly out-of-range index by the boundary rule.

        References:
            [EQ-LINALG-OP-004].

        """
        if 0 <= index < size:
            return index
        if self.boundary == "zero":
            return None
        if self.boundary == "periodic":
            return index % size
        period = 2 * size
        folded = index % period
        return folded if folded < size else period - 1 - folded

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``K x``.

        References:
            [EQ-LINALG-OP-004].

        """
        return tuple(
            math.fsum(weight * vector[j] for _, j, weight in row) for row in self._taps
        )

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return ``K'y`` by scatter-adding through the index map.

        References:
            [EQ-LINALG-OP-004].

        """
        result = [0.0] * self._shape[1]
        for row in self._taps:
            for i, j, weight in row:
                result[j] += weight * vector[i]
        return result


class TransposedOperator(OperatorBase):
    """The transpose ``A'`` of an operator.

    References:
        [EQ-LINALG-OP-001].

    """

    __slots__ = ("base",)

    def __init__(self, base: LinearOperator) -> None:
        """Wrap ``base``.

        Args:
            base: Operator to transpose.

        References:
            [EQ-LINALG-OP-001].

        """
        self.base = require_operator(base, name="base")
        rows, columns = self.base.shape
        super().__init__(
            columns, rows, name=f"transpose of {getattr(base, 'name', 'operator')}"
        )

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``A'x``.

        References:
            [EQ-LINALG-OP-001].

        """
        return _apply_transpose(self.base, vector)

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return ``A y``.

        References:
            [EQ-LINALG-OP-001].

        """
        return _apply(self.base, vector)


class ScaledOperator(OperatorBase):
    """The operator ``c A`` for a finite scalar ``c``.

    References:
        [EQ-LINALG-OP-002].

    """

    __slots__ = ("base", "factor")

    def __init__(self, base: LinearOperator, factor: float) -> None:
        """Store ``A`` and ``c``.

        Args:
            base: Operator ``A``.
            factor: Finite scalar ``c``.

        References:
            [EQ-LINALG-OP-002].

        """
        self.base = require_operator(base, name="base")
        self.factor = _real(factor, "factor")
        super().__init__(*self.base.shape, name=f"{self.factor} * operator")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``c A x``.

        References:
            [EQ-LINALG-OP-002].

        """
        return tuple(self.factor * value for value in _apply(self.base, vector))

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return ``c A'y``.

        References:
            [EQ-LINALG-OP-002].

        """
        return tuple(
            self.factor * value for value in _apply_transpose(self.base, vector)
        )


class SumOperator(OperatorBase):
    """The sum ``A_1 + ... + A_k`` of operators of equal shape.

    References:
        [EQ-LINALG-OP-002].

    """

    __slots__ = ("terms",)

    def __init__(self, terms: Sequence[LinearOperator]) -> None:
        """Store the terms.

        Args:
            terms: Non-empty operators of identical shape.

        Raises:
            ValueError: If empty or the shapes differ.

        References:
            [EQ-LINALG-OP-002].

        """
        checked = tuple(require_operator(term, name="term") for term in terms)
        if not checked:
            raise ValueError("terms must be non-empty")
        shape = checked[0].shape
        if any(term.shape != shape for term in checked):
            raise ValueError("all terms must have the same shape")
        self.terms = checked
        super().__init__(*shape, name="sum")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``sum_k A_k x`` with compensated sums.

        References:
            [EQ-LINALG-OP-002].

        """
        parts = [_apply(term, vector) for term in self.terms]
        return tuple(math.fsum(values) for values in zip(*parts, strict=True))

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return ``sum_k A_k'y`` with compensated sums.

        References:
            [EQ-LINALG-OP-002].

        """
        parts = [_apply_transpose(term, vector) for term in self.terms]
        return tuple(math.fsum(values) for values in zip(*parts, strict=True))


class ComposedOperator(OperatorBase):
    """The product ``A_1 A_2 ... A_k`` (``A_k`` applied first).

    References:
        [EQ-LINALG-OP-002].

    """

    __slots__ = ("factors",)

    def __init__(self, factors: Sequence[LinearOperator]) -> None:
        """Store the factors from left to right.

        Args:
            factors: Non-empty operators with matching inner dimensions.

        Raises:
            ValueError: If empty or the inner dimensions do not match.

        References:
            [EQ-LINALG-OP-002].

        """
        checked = tuple(require_operator(factor, name="factor") for factor in factors)
        if not checked:
            raise ValueError("factors must be non-empty")
        for left, right in zip(checked, checked[1:], strict=False):
            if left.shape[1] != right.shape[0]:
                raise ValueError(
                    f"cannot compose shapes {left.shape} and {right.shape}"
                )
        self.factors = checked
        super().__init__(checked[0].shape[0], checked[-1].shape[1], name="product")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Apply the factors right to left.

        References:
            [EQ-LINALG-OP-002].

        """
        result: Vector = vector
        for factor in reversed(self.factors):
            result = _apply(factor, result)
        return result

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Apply the transposed factors left to right.

        References:
            [EQ-LINALG-OP-002].

        """
        result: Vector = vector
        for factor in self.factors:
            result = _apply_transpose(factor, result)
        return result


class VStackOperator(OperatorBase):
    """Row stacking ``[A_1; ...; A_k]`` of operators with equal domain.

    The transpose of ``[A; B]`` maps ``(u, v)`` to ``A'u + B'v``
    [EQ-LINALG-OP-002].

    References:
        [EQ-LINALG-OP-002].

    """

    __slots__ = ("blocks", "offsets")

    def __init__(self, blocks: Sequence[LinearOperator]) -> None:
        """Store the blocks.

        Args:
            blocks: Non-empty operators with the same number of columns.

        Raises:
            ValueError: If empty or the column counts differ.

        References:
            [EQ-LINALG-OP-002].

        """
        checked = tuple(require_operator(block, name="block") for block in blocks)
        if not checked:
            raise ValueError("blocks must be non-empty")
        columns = checked[0].shape[1]
        if any(block.shape[1] != columns for block in checked):
            raise ValueError("all blocks must have the same number of columns")
        offsets = [0]
        for block in checked:
            offsets.append(offsets[-1] + block.shape[0])
        self.blocks = checked
        self.offsets = tuple(offsets)
        super().__init__(offsets[-1], columns, name="vertical stack")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Concatenate ``A_k x``.

        References:
            [EQ-LINALG-OP-002].

        """
        result: list[float] = []
        for block in self.blocks:
            result.extend(_apply(block, vector))
        return result

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Return ``sum_k A_k' y_k``.

        References:
            [EQ-LINALG-OP-002].

        """
        parts = [
            _apply_transpose(block, vector[start:stop])
            for block, start, stop in zip(
                self.blocks, self.offsets, self.offsets[1:], strict=False
            )
        ]
        return tuple(math.fsum(values) for values in zip(*parts, strict=True))


class HStackOperator(OperatorBase):
    """Column stacking ``[A_1, ..., A_k]`` of operators with equal range.

    References:
        [EQ-LINALG-OP-002].

    """

    __slots__ = ("blocks", "offsets")

    def __init__(self, blocks: Sequence[LinearOperator]) -> None:
        """Store the blocks.

        Args:
            blocks: Non-empty operators with the same number of rows.

        Raises:
            ValueError: If empty or the row counts differ.

        References:
            [EQ-LINALG-OP-002].

        """
        checked = tuple(require_operator(block, name="block") for block in blocks)
        if not checked:
            raise ValueError("blocks must be non-empty")
        rows = checked[0].shape[0]
        if any(block.shape[0] != rows for block in checked):
            raise ValueError("all blocks must have the same number of rows")
        offsets = [0]
        for block in checked:
            offsets.append(offsets[-1] + block.shape[1])
        self.blocks = checked
        self.offsets = tuple(offsets)
        super().__init__(rows, offsets[-1], name="horizontal stack")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Return ``sum_k A_k x_k``.

        References:
            [EQ-LINALG-OP-002].

        """
        parts = [
            _apply(block, vector[start:stop])
            for block, start, stop in zip(
                self.blocks, self.offsets, self.offsets[1:], strict=False
            )
        ]
        return tuple(math.fsum(values) for values in zip(*parts, strict=True))

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Concatenate ``A_k'y``.

        References:
            [EQ-LINALG-OP-002].

        """
        result: list[float] = []
        for block in self.blocks:
            result.extend(_apply_transpose(block, vector))
        return result


class BlockDiagonalOperator(OperatorBase):
    """Block-diagonal operator ``diag(A_1, ..., A_k)``.

    References:
        [EQ-LINALG-OP-002].

    """

    __slots__ = ("blocks", "column_offsets", "row_offsets")

    def __init__(self, blocks: Sequence[LinearOperator]) -> None:
        """Store the diagonal blocks.

        Args:
            blocks: Non-empty operators.

        Raises:
            ValueError: If empty.

        References:
            [EQ-LINALG-OP-002].

        """
        checked = tuple(require_operator(block, name="block") for block in blocks)
        if not checked:
            raise ValueError("blocks must be non-empty")
        rows, columns = [0], [0]
        for block in checked:
            rows.append(rows[-1] + block.shape[0])
            columns.append(columns[-1] + block.shape[1])
        self.blocks = checked
        self.row_offsets = tuple(rows)
        self.column_offsets = tuple(columns)
        super().__init__(rows[-1], columns[-1], name="block diagonal")

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Concatenate ``A_k x_k``.

        References:
            [EQ-LINALG-OP-002].

        """
        result: list[float] = []
        for block, start, stop in zip(
            self.blocks, self.column_offsets, self.column_offsets[1:], strict=False
        ):
            result.extend(_apply(block, vector[start:stop]))
        return result

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Concatenate ``A_k'y_k``.

        References:
            [EQ-LINALG-OP-002].

        """
        result: list[float] = []
        for block, start, stop in zip(
            self.blocks, self.row_offsets, self.row_offsets[1:], strict=False
        ):
            result.extend(_apply_transpose(block, vector[start:stop]))
        return result


@dataclass(frozen=True, slots=True)
class OperatorCounts:
    """Numbers of forward and transpose applications of a counted operator.

    Attributes:
        forward: Calls of ``matvec``.
        transpose: Calls of ``rmatvec``.

    References:
        Project work-accounting policy [REF-PROBSTAT-API]. No equation is
        evaluated.

    """

    forward: int
    transpose: int


class CountingOperator(OperatorBase):
    """Wrapper that counts forward and transpose applications separately.

    Every call is counted, including calls made by diagnostics, so a solver
    that wraps its operator reports its exact work. Create a new wrapper per
    run; counts are never shared between independent runs.

    References:
        Project work-accounting policy [REF-PROBSTAT-API], [EQ-LINALG-OP-001].

    """

    __slots__ = ("base", "_forward_calls", "_transpose_calls")

    def __init__(self, base: LinearOperator) -> None:
        """Wrap ``base`` with zero counts.

        Args:
            base: Operator to count.

        References:
            [EQ-LINALG-OP-001].

        """
        self.base = require_operator(base, name="base")
        self._forward_calls = 0
        self._transpose_calls = 0
        super().__init__(
            *self.base.shape, name=f"counted {getattr(base, 'name', 'operator')}"
        )

    @property
    def counts(self) -> OperatorCounts:
        """Return the current counts.

        References:
            Project work-accounting policy [REF-PROBSTAT-API].

        """
        return OperatorCounts(self._forward_calls, self._transpose_calls)

    def _forward(self, vector: Vector) -> Sequence[float]:
        """Count and apply ``A``.

        References:
            [EQ-LINALG-OP-001].

        """
        self._forward_calls += 1
        return _apply(self.base, vector)

    def _transpose(self, vector: Vector) -> Sequence[float]:
        """Count and apply ``A'``.

        References:
            [EQ-LINALG-OP-001].

        """
        self._transpose_calls += 1
        return _apply_transpose(self.base, vector)


def materialize(
    operator: LinearOperator, *, max_entries: int = _DEFAULT_MATERIALIZE_LIMIT
) -> Matrix:
    """Return the dense matrix of an operator by applying it to unit vectors.

    This is an explicit debugging and oracle tool: it costs ``n`` forward
    applications and ``m n`` memory, and refuses when ``m n`` exceeds
    ``max_entries``.

    Args:
        operator: Operator to materialize.
        max_entries: Upper limit on ``m n``.

    Returns:
        The ``m x n`` matrix as rows.

    Raises:
        ValueError: If ``m n > max_entries`` or the operator is empty.

    References:
        [EQ-LINALG-OP-001].

    """
    checked = require_operator(operator)
    rows, columns = checked.shape
    if rows == 0 or columns == 0:
        raise ValueError("cannot materialize an operator with an empty dimension")
    if rows * columns > max_entries:
        raise ValueError(
            f"materializing {rows} x {columns} exceeds max_entries={max_entries}"
        )
    columns_list = []
    for j in range(columns):
        unit = [0.0] * columns
        unit[j] = 1.0
        columns_list.append(_apply(checked, unit))
    return tuple(tuple(column[i] for column in columns_list) for i in range(rows))


@dataclass(frozen=True, slots=True)
class AdjointTestResult:
    """Outcome of the dot-product test ``y'(A x) = (A'y)'x``.

    Attributes:
        forward_inner: ``y'(A x)``.
        transpose_inner: ``(A'y)'x``.
        relative_error: ``|forward - transpose| / (||A x|| ||y|| + ||x|| ||A'y||)``
            (zero when the denominator vanishes).
        passed: Whether ``relative_error <= tolerance``.
        tolerance: Tolerance used.

    References:
        [EQ-LINALG-OP-001].

    """

    forward_inner: float
    transpose_inner: float
    relative_error: float
    passed: bool
    tolerance: float


def adjoint_test(
    operator: LinearOperator,
    x: Sequence[float],
    y: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> AdjointTestResult:
    """Run the adjoint (dot-product) test of an operator.

    The relative error is scaled by ``||A x|| ||y|| + ||x|| ||A'y||``, the
    natural size of the rounding error of the two inner products
    [EQ-LINALG-OP-001].

    Args:
        operator: Operator to test.
        x: Domain vector of length ``n``.
        y: Range vector of length ``m``.
        tolerance: Pass threshold on the relative error.

    Returns:
        :class:`AdjointTestResult`.

    References:
        [EQ-LINALG-OP-001], [REF-GAZZOLA-HANSEN-NAGY-2019].

    """
    checked = require_operator(operator)
    rows, columns = checked.shape
    xv = _vector(x, columns, "x")
    yv = _vector(y, rows, "y")
    ax = _apply(checked, xv)
    aty = _apply_transpose(checked, yv)
    forward = math.fsum(a * b for a, b in zip(yv, ax, strict=True))
    backward = math.fsum(a * b for a, b in zip(aty, xv, strict=True))
    scale = math.hypot(*ax) * math.hypot(*yv) + math.hypot(*xv) * math.hypot(*aty)
    error = abs(forward - backward) / scale if scale > 0.0 else 0.0
    tol = _real(tolerance, "tolerance")
    return AdjointTestResult(forward, backward, error, error <= tol, tol)


class SymmetricPositiveDefinite(Protocol):
    """Structural protocol of an SPD metric ``M`` with apply and solve actions.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """

    @property
    def dimension(self) -> int:
        """Return the dimension of ``M``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

        """
        ...

    def apply(self, vector: Sequence[float]) -> Vector:
        """Return ``M v``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

        """
        ...

    def solve(self, vector: Sequence[float]) -> Vector:
        """Return ``M^-1 v``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

        """
        ...


def weighted_adjoint(
    operator: LinearOperator,
    *,
    domain_metric: SymmetricPositiveDefinite,
    range_metric: SymmetricPositiveDefinite,
) -> CallbackOperator:
    """Return the adjoint ``A* = M_X^-1 A' M_Y`` in weighted inner products.

    With ``<x, x'>_X = x'M_X x'`` and ``<y, y'>_Y = y'M_Y y'``, the adjoint
    satisfies ``<A x, y>_Y = <x, A* y>_X`` [EQ-LINALG-OP-005]. The returned
    operator's own ``rmatvec`` is the Euclidean transpose of ``A*``, namely
    ``M_Y A M_X^-1``.

    Args:
        operator: ``A: R^n -> R^m``.
        domain_metric: SPD ``M_X`` of dimension ``n`` (apply and solve).
        range_metric: SPD ``M_Y`` of dimension ``m``.

    Returns:
        An ``n x m`` :class:`CallbackOperator` applying ``A*``.

    Raises:
        ValueError: If the metric dimensions do not match the operator.

    References:
        [EQ-LINALG-OP-005], [REF-STUART-2010, Section 2].

    """
    checked = require_operator(operator)
    rows, columns = checked.shape
    if domain_metric.dimension != columns or range_metric.dimension != rows:
        raise ValueError("metric dimensions must match the operator shape")

    def forward(vector: Vector) -> Vector:
        """Return ``M_X^-1 A' M_Y y``.

        References:
            [EQ-LINALG-OP-005].

        """
        return domain_metric.solve(
            _apply_transpose(checked, range_metric.apply(vector))
        )

    def backward(vector: Vector) -> Vector:
        """Return ``M_Y A M_X^-1 x``.

        References:
            [EQ-LINALG-OP-005].

        """
        return range_metric.apply(_apply(checked, domain_metric.solve(vector)))

    return CallbackOperator((columns, rows), forward, backward, name="weighted adjoint")


def apply(operator: LinearOperator, vector: Sequence[float]) -> Vector:
    """Return ``A x`` for any :class:`LinearOperator`, validating the result.

    Args:
        operator: Operator.
        vector: Input of length ``n``.

    Returns:
        ``A x``.

    References:
        [EQ-LINALG-OP-001].

    """
    return _apply(require_operator(operator), vector)


def apply_transpose(operator: LinearOperator, vector: Sequence[float]) -> Vector:
    """Return ``A'y`` for any :class:`LinearOperator`, validating the result.

    Args:
        operator: Operator.
        vector: Input of length ``m``.

    Returns:
        ``A'y``.

    References:
        [EQ-LINALG-OP-001].

    """
    return _apply_transpose(require_operator(operator), vector)
