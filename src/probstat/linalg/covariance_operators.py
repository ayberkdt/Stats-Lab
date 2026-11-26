"""Covariance factors: whitening, covariance and precision actions, log-determinants.

A Gaussian covariance ``Gamma`` (symmetric positive definite) is held through
a factor ``C`` with ``Gamma = C C'``. The whitening map is ``W = C^-1``, so
``W'W = Gamma^-1`` and ``||W r||^2 = r'Gamma^-1 r`` [EQ-LINALG-COV-001]. Each
class exposes

- ``apply(v) = Gamma v`` and ``solve(v) = Gamma^-1 v`` (so a covariance is
  also an SPD metric for :func:`probstat.linalg.weighted_adjoint`);
- ``whiten(r) = C^-1 r``, ``whiten_transpose(v) = C^-T v`` and
  ``unwhiten(z) = C z`` (``C z`` with ``z ~ N(0, I)`` draws from
  ``N(0, Gamma)``);
- ``log_det() = log det Gamma = 2 sum_i log C_ii``;
- ``whitening_operator()`` and ``factor_operator()`` as
  :class:`~probstat.linalg.operators.LinearOperator` objects whose
  ``rmatvec`` is the matching transpose.

Validation is strict: a non-positive variance or a matrix whose Cholesky
factorization fails raises ``ValueError``; no jitter is ever added.

Experimental (added in 0.28).

References:
    [REF-GOLUB-VANLOAN-2013, Section 4.2], [REF-TARANTOLA-2005,
    Chapter 3].

"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .dense import Matrix, Vector, as_matrix, cholesky, solve_lower
from .operators import CallbackOperator, _vector


class _CovarianceBase:
    """Shared operator constructors of the covariance factors.

    References:
        [EQ-LINALG-COV-001].

    """

    __slots__ = ()

    @property
    def dimension(self) -> int:
        """Return the dimension (implemented by subclasses).

        Raises:
            NotImplementedError: Always, in the base class.

        References:
            [EQ-LINALG-COV-001].

        """
        raise NotImplementedError

    def whiten(self, vector: Sequence[float]) -> Vector:
        """Return ``C^-1 r`` (implemented by subclasses).

        Raises:
            NotImplementedError: Always, in the base class.

        References:
            [EQ-LINALG-COV-001].

        """
        raise NotImplementedError

    def whiten_transpose(self, vector: Sequence[float]) -> Vector:
        """Return ``C^-T v`` (implemented by subclasses).

        Raises:
            NotImplementedError: Always, in the base class.

        References:
            [EQ-LINALG-COV-001].

        """
        raise NotImplementedError

    def unwhiten(self, vector: Sequence[float]) -> Vector:
        """Return ``C z`` (implemented by subclasses).

        Raises:
            NotImplementedError: Always, in the base class.

        References:
            [EQ-LINALG-COV-001].

        """
        raise NotImplementedError

    def unwhiten_transpose(self, vector: Sequence[float]) -> Vector:
        """Return ``C'v`` (implemented by subclasses).

        Raises:
            NotImplementedError: Always, in the base class.

        References:
            [EQ-LINALG-COV-001].

        """
        raise NotImplementedError

    def log_det(self) -> float:
        """Return ``log det Gamma`` (implemented by subclasses).

        Raises:
            NotImplementedError: Always, in the base class.

        References:
            [EQ-LINALG-COV-001].

        """
        raise NotImplementedError

    def apply(self, vector: Sequence[float]) -> Vector:
        """Return ``Gamma v = C (C'v)``.

        Args:
            vector: Finite vector of the covariance dimension.

        Returns:
            ``Gamma v``.

        References:
            [EQ-LINALG-COV-001].

        """
        return self.unwhiten(self.unwhiten_transpose(vector))

    def solve(self, vector: Sequence[float]) -> Vector:
        """Return ``Gamma^-1 v = C^-T (C^-1 v)``.

        Args:
            vector: Finite vector of the covariance dimension.

        Returns:
            ``Gamma^-1 v``.

        References:
            [EQ-LINALG-COV-001].

        """
        return self.whiten_transpose(self.whiten(vector))

    def quadratic_form(self, vector: Sequence[float]) -> float:
        """Return the squared Mahalanobis norm ``r'Gamma^-1 r = ||C^-1 r||^2``.

        Args:
            vector: Residual ``r``.

        Returns:
            ``||W r||^2``.

        References:
            [EQ-LINALG-COV-001].

        """
        return math.fsum(value * value for value in self.whiten(vector))

    def whitening_operator(self) -> CallbackOperator:
        """Return ``W = C^-1`` as an operator with ``rmatvec = W'``.

        References:
            [EQ-LINALG-COV-001].

        """
        n = self.dimension
        return CallbackOperator(
            (n, n), self.whiten, self.whiten_transpose, name="whitening"
        )

    def factor_operator(self) -> CallbackOperator:
        """Return ``C`` as an operator with ``rmatvec = C'``.

        References:
            [EQ-LINALG-COV-001].

        """
        n = self.dimension
        return CallbackOperator(
            (n, n), self.unwhiten, self.unwhiten_transpose, name="covariance factor"
        )

    def covariance_operator(self) -> CallbackOperator:
        """Return ``Gamma`` as a symmetric operator.

        References:
            [EQ-LINALG-COV-001].

        """
        n = self.dimension
        return CallbackOperator((n, n), self.apply, self.apply, name="covariance")

    def precision_operator(self) -> CallbackOperator:
        """Return ``Gamma^-1`` as a symmetric operator (by solves, never inverted).

        References:
            [EQ-LINALG-COV-001].

        """
        n = self.dimension
        return CallbackOperator((n, n), self.solve, self.solve, name="precision")


class DiagonalCovariance(_CovarianceBase):
    """Diagonal covariance ``diag(s_1^2, ..., s_m^2)`` with factor ``diag(s)``.

    References:
        [EQ-LINALG-COV-001].

    """

    __slots__ = ("standard_deviations",)

    def __init__(self, variances: Sequence[float]) -> None:
        """Store positive variances.

        Args:
            variances: Non-empty strictly positive finite variances.

        Raises:
            ValueError: If empty, non-finite or not strictly positive.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        values = tuple(float(value) for value in variances)
        if not values:
            raise ValueError("variances must be non-empty")
        values = _vector(values, len(values), "variances")
        if any(value <= 0.0 for value in values):
            raise ValueError("variances must be strictly positive")
        self.standard_deviations = tuple(math.sqrt(value) for value in values)

    @classmethod
    def from_standard_deviations(
        cls, deviations: Sequence[float]
    ) -> DiagonalCovariance:
        """Build the covariance from standard deviations ``s_i > 0``.

        Args:
            deviations: Strictly positive finite standard deviations.

        Returns:
            :class:`DiagonalCovariance` with variances ``s_i^2``.

        References:
            [EQ-LINALG-COV-001].

        """
        values = tuple(float(value) for value in deviations)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("standard deviations must be finite and positive")
        return cls(tuple(value * value for value in values))

    @property
    def dimension(self) -> int:
        """Return ``m``.

        References:
            [EQ-LINALG-COV-001].

        """
        return len(self.standard_deviations)

    def whiten(self, vector: Sequence[float]) -> Vector:
        """Return ``r_i / s_i``.

        References:
            [EQ-LINALG-COV-001].

        """
        checked = _vector(vector, self.dimension, "vector")
        return tuple(
            r / s for r, s in zip(checked, self.standard_deviations, strict=True)
        )

    def whiten_transpose(self, vector: Sequence[float]) -> Vector:
        """Return ``v_i / s_i`` (the factor is symmetric).

        References:
            [EQ-LINALG-COV-001].

        """
        return self.whiten(vector)

    def unwhiten(self, vector: Sequence[float]) -> Vector:
        """Return ``s_i z_i``.

        References:
            [EQ-LINALG-COV-001].

        """
        checked = _vector(vector, self.dimension, "vector")
        return tuple(
            z * s for z, s in zip(checked, self.standard_deviations, strict=True)
        )

    def unwhiten_transpose(self, vector: Sequence[float]) -> Vector:
        """Return ``s_i v_i``.

        References:
            [EQ-LINALG-COV-001].

        """
        return self.unwhiten(vector)

    def log_det(self) -> float:
        """Return ``sum_i log s_i^2``.

        References:
            [EQ-LINALG-COV-001].

        """
        return 2.0 * math.fsum(math.log(s) for s in self.standard_deviations)


class ScaledIdentityCovariance(DiagonalCovariance):
    """Isotropic covariance ``s^2 I_m``.

    References:
        [EQ-LINALG-COV-001].

    """

    __slots__ = ()

    def __init__(self, dimension: int, variance: float = 1.0) -> None:
        """Store the dimension and the common variance.

        Args:
            dimension: Positive ``m``.
            variance: Strictly positive common variance ``s^2``.

        Raises:
            ValueError: If an argument is invalid.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("dimension must be a positive integer")
        super().__init__((float(variance),) * dimension)


class DenseCovariance(_CovarianceBase):
    """Dense SPD covariance held by its lower Cholesky factor ``Gamma = L L'``.

    References:
        [EQ-LINALG-COV-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """

    __slots__ = ("lower", "_upper")

    def __init__(
        self,
        covariance: Sequence[Sequence[float]],
        *,
        symmetry_tolerance: float = 1e-12,
    ) -> None:
        """Factor an SPD matrix.

        Args:
            covariance: Square finite matrix, symmetric to
                ``symmetry_tolerance`` relative to its largest entry.
            symmetry_tolerance: Relative symmetry tolerance.

        Raises:
            ValueError: If the matrix is not square, not symmetric, or not
                positive definite (the Cholesky factorization fails).

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
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
            self.lower: Matrix = cholesky(matrix)
        except ValueError as error:
            raise ValueError("covariance must be positive definite") from error
        self._upper: Matrix = tuple(zip(*self.lower, strict=True))

    @classmethod
    def from_factor(cls, lower: Sequence[Sequence[float]]) -> DenseCovariance:
        """Build the covariance ``L L'`` from a lower-triangular factor.

        Args:
            lower: Square lower-triangular factor with positive diagonal.

        Returns:
            :class:`DenseCovariance`.

        Raises:
            ValueError: If the factor is not lower triangular with a positive
                diagonal.

        References:
            [EQ-LINALG-COV-001].

        """
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
    def dimension(self) -> int:
        """Return ``m``.

        References:
            [EQ-LINALG-COV-001].

        """
        return len(self.lower)

    def whiten(self, vector: Sequence[float]) -> Vector:
        """Return ``L^-1 r`` by forward substitution.

        References:
            [EQ-LINALG-COV-001].

        """
        return solve_lower(self.lower, _vector(vector, self.dimension, "vector"))

    def whiten_transpose(self, vector: Sequence[float]) -> Vector:
        """Return ``L^-T v`` by back substitution.

        References:
            [EQ-LINALG-COV-001].

        """
        values = list(_vector(vector, self.dimension, "vector"))
        size = self.dimension
        for i in range(size - 1, -1, -1):
            total = values[i] - math.fsum(
                self.lower[j][i] * values[j] for j in range(i + 1, size)
            )
            values[i] = total / self.lower[i][i]
        return tuple(values)

    def unwhiten(self, vector: Sequence[float]) -> Vector:
        """Return ``L z``.

        References:
            [EQ-LINALG-COV-001].

        """
        checked = _vector(vector, self.dimension, "vector")
        return tuple(
            math.fsum(row[j] * checked[j] for j in range(i + 1))
            for i, row in enumerate(self.lower)
        )

    def unwhiten_transpose(self, vector: Sequence[float]) -> Vector:
        """Return ``L'v``.

        References:
            [EQ-LINALG-COV-001].

        """
        checked = _vector(vector, self.dimension, "vector")
        size = self.dimension
        return tuple(
            math.fsum(self._upper[i][j] * checked[j] for j in range(i, size))
            for i in range(size)
        )

    def log_det(self) -> float:
        """Return ``2 sum_i log L_ii``.

        References:
            [EQ-LINALG-COV-001].

        """
        return 2.0 * math.fsum(
            math.log(self.lower[i][i]) for i in range(self.dimension)
        )


class BlockDiagonalCovariance(_CovarianceBase):
    """Block-diagonal covariance ``diag(Gamma_1, ..., Gamma_k)`` of independent blocks.

    References:
        [EQ-LINALG-COV-001].

    """

    __slots__ = ("blocks", "offsets")

    def __init__(self, blocks: Sequence[_CovarianceBase]) -> None:
        """Store the blocks.

        Args:
            blocks: Non-empty covariance factors of this module.

        Raises:
            TypeError: If a block is not a covariance factor.
            ValueError: If empty.

        References:
            [EQ-LINALG-COV-001].

        """
        checked = tuple(blocks)
        if not checked:
            raise ValueError("blocks must be non-empty")
        if not all(isinstance(block, _CovarianceBase) for block in checked):
            raise TypeError("blocks must be covariance factors")
        offsets = [0]
        for block in checked:
            offsets.append(offsets[-1] + block.dimension)
        self.blocks = checked
        self.offsets = tuple(offsets)

    @property
    def dimension(self) -> int:
        """Return the total dimension.

        References:
            [EQ-LINALG-COV-001].

        """
        return self.offsets[-1]

    def _blockwise(self, vector: Sequence[float], method: str) -> Vector:
        """Apply a block method to the matching segments.

        References:
            [EQ-LINALG-COV-001].

        """
        checked = _vector(vector, self.dimension, "vector")
        result: list[float] = []
        for block, start, stop in zip(
            self.blocks, self.offsets, self.offsets[1:], strict=False
        ):
            result.extend(getattr(block, method)(checked[start:stop]))
        return tuple(result)

    def whiten(self, vector: Sequence[float]) -> Vector:
        """Return the blockwise ``C_k^-1 r_k``.

        References:
            [EQ-LINALG-COV-001].

        """
        return self._blockwise(vector, "whiten")

    def whiten_transpose(self, vector: Sequence[float]) -> Vector:
        """Return the blockwise ``C_k^-T v_k``.

        References:
            [EQ-LINALG-COV-001].

        """
        return self._blockwise(vector, "whiten_transpose")

    def unwhiten(self, vector: Sequence[float]) -> Vector:
        """Return the blockwise ``C_k z_k``.

        References:
            [EQ-LINALG-COV-001].

        """
        return self._blockwise(vector, "unwhiten")

    def unwhiten_transpose(self, vector: Sequence[float]) -> Vector:
        """Return the blockwise ``C_k'v_k``.

        References:
            [EQ-LINALG-COV-001].

        """
        return self._blockwise(vector, "unwhiten_transpose")

    def log_det(self) -> float:
        """Return ``sum_k log det Gamma_k``.

        References:
            [EQ-LINALG-COV-001].

        """
        return math.fsum(block.log_det() for block in self.blocks)


CovarianceFactor = DiagonalCovariance | DenseCovariance | BlockDiagonalCovariance
"""Union of the concrete covariance factors (scaled identity is diagonal)."""


def is_covariance_factor(value: object) -> bool:
    """Return whether ``value`` is one of this module's covariance factors.

    Args:
        value: Candidate object.

    Returns:
        ``True`` for a :class:`DiagonalCovariance`, :class:`DenseCovariance`
        or :class:`BlockDiagonalCovariance` (including subclasses).

    References:
        [EQ-LINALG-COV-001].

    """
    return isinstance(value, _CovarianceBase)
