"""Multivariate normal distribution using dependency-free Cholesky algebra."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from probstat.distributions import Normal
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class MultivariateNormal:
    """A non-degenerate multivariate normal distribution.

    The covariance must be finite, symmetric, and positive definite. Singular
    Gaussian measures require lower-dimensional support machinery and are
    intentionally rejected rather than assigned a misleading ambient density.

    Args:
        mean: Non-empty finite mean vector.
        covariance: Square symmetric positive-definite covariance matrix.

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3].

    """

    mean: Vector
    covariance: Matrix
    _cholesky: Matrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize vectors and validate covariance through Cholesky factorization.

        Raises:
            TypeError: If inputs are not iterable finite real values.
            ValueError: If dimensions disagree or covariance is not symmetric
                positive definite.

        References:
            [EQ-DIST-MVN-001], [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

        """
        mean = _finite_vector(self.mean, name="mean")
        if not mean:
            raise ValueError("mean must contain at least one component")
        try:
            raw_rows = tuple(self.covariance)
        except TypeError as error:
            raise TypeError("covariance must be an iterable of rows") from error
        if len(raw_rows) != len(mean):
            raise ValueError("covariance row count must equal mean dimension")
        rows = tuple(
            _finite_vector(row, name=f"covariance[{index}]")
            for index, row in enumerate(raw_rows)
        )
        if any(len(row) != len(mean) for row in rows):
            raise ValueError("covariance must be square with mean dimension")
        symmetric_rows: list[tuple[float, ...]] = []
        for row_index, row in enumerate(rows):
            symmetric_row: list[float] = []
            for column_index, value in enumerate(row):
                mirror = rows[column_index][row_index]
                tolerance = 1e-12 * max(1.0, abs(value), abs(mirror))
                if not math.isclose(value, mirror, rel_tol=0.0, abs_tol=tolerance):
                    raise ValueError("covariance must be symmetric")
                symmetric_row.append(0.5 * (value + mirror))
            symmetric_rows.append(tuple(symmetric_row))
        covariance = tuple(symmetric_rows)
        cholesky = _cholesky_factor(covariance)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "_cholesky", cholesky)

    @property
    def dimension(self) -> int:
        """Return vector dimension.

        References:
            [EQ-DIST-MVN-001], [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

        """
        return len(self.mean)

    def mahalanobis_squared(self, value: Iterable[float]) -> float:
        """Return squared Mahalanobis distance from the mean.

        Args:
            value: Finite vector with the distribution dimension.

        Returns:
            Non-negative quadratic form ``(x-mu)' Sigma^-1 (x-mu)``.

        Raises:
            TypeError: If components are not iterable finite reals.
            ValueError: If vector dimension differs.

        Equation:
            [EQ-DIST-MVN-002].

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

        """
        point = _finite_vector(value, name="value")
        if len(point) != self.dimension:
            raise ValueError("value dimension must equal distribution dimension")
        centered = tuple(
            coordinate - location
            for coordinate, location in zip(point, self.mean, strict=True)
        )
        solved = _solve_lower(self._cholesky, centered)
        return math.fsum(component * component for component in solved)

    def log_pdf(self, value: Iterable[float]) -> float:
        """Evaluate the multivariate normal log-density.

        Args:
            value: Finite vector with the distribution dimension.

        Returns:
            Natural logarithm of density with respect to Lebesgue measure.

        Raises:
            TypeError: If components are not iterable finite reals.
            ValueError: If vector dimension differs.

        Equation:
            [EQ-DIST-MVN-001], [EQ-DIST-MVN-002].

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

        """
        quadratic = self.mahalanobis_squared(value)
        log_determinant = 2.0 * math.fsum(
            math.log(self._cholesky[index][index])
            for index in range(self.dimension)
        )
        return -0.5 * (
            self.dimension * math.log(2.0 * math.pi)
            + log_determinant
            + quadratic
        )

    def pdf(self, value: Iterable[float]) -> float:
        """Evaluate the multivariate normal probability density.

        Args:
            value: Finite vector with the distribution dimension.

        Returns:
            Density at ``value``; a density is not a point probability.

        Raises:
            TypeError: If components are not iterable finite reals.
            ValueError: If vector dimension differs.

        Equation:
            [EQ-DIST-MVN-001].

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

        """
        return math.exp(self.log_pdf(value))

    def marginal(self, index: int) -> Normal:
        """Return one univariate normal marginal.

        Args:
            index: Zero-based component index.

        Returns:
            Marginal normal distribution for the selected component.

        Raises:
            TypeError: If ``index`` is not an integer.
            IndexError: If ``index`` is outside the vector dimension.

        Equation:
            [EQ-DIST-MVN-003].

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 3].

        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < self.dimension:
            raise IndexError("index outside distribution dimension")
        return Normal(self.mean[index], math.sqrt(self.covariance[index][index]))

    def correlation(self, first: int, second: int) -> float:
        """Return the Pearson correlation between two components.

        Args:
            first: Zero-based first component index.
            second: Zero-based second component index.

        Returns:
            Covariance standardized by marginal standard deviations.

        Raises:
            TypeError: If an index is not an integer.
            IndexError: If an index is outside the vector dimension.

        Equation:
            [EQ-DIST-MVN-004].

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

        """
        first_law = self.marginal(first)
        second_law = self.marginal(second)
        return self.covariance[first][second] / (
            first_law.sigma * second_law.sigma
        )

    def sample(
        self, size: int = 1, *, seed: int | None = None
    ) -> tuple[Vector, ...]:
        """Draw reproducible pseudo-random multivariate normal observations.

        Args:
            size: Number of vector observations, with ``size >= 0``.
            seed: Optional seed for a call-local random-number generator.

        Returns:
            Tuple of finite vector observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.
            OverflowError: If a generated component exceeds float range.

        Equation:
            [EQ-DIST-MVN-005].

        References:
            [REF-DEVROYE-1986, Chapter XI],
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 3].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        observations: list[Vector] = []
        for _ in range(checked_size):
            independent = tuple(generator.gauss(0.0, 1.0) for _ in self.mean)
            observation = tuple(
                self.mean[row]
                + math.fsum(
                    self._cholesky[row][column] * independent[column]
                    for column in range(row + 1)
                )
                for row in range(self.dimension)
            )
            if not all(math.isfinite(component) for component in observation):
                raise OverflowError("multivariate normal sample is not finite")
            observations.append(observation)
        return tuple(observations)


def _finite_vector(values: Iterable[float], *, name: str) -> Vector:
    """Materialize and validate a finite real vector.

    Args:
        values: Candidate vector components.
        name: Field name for indexed error messages.

    Returns:
        Tuple of validated float components.

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


def _cholesky_factor(matrix: Matrix) -> Matrix:
    """Return a lower Cholesky factor or reject non-positive-definite input.

    Args:
        matrix: Finite symmetric square matrix.

    Returns:
        Lower triangular ``L`` satisfying ``matrix = L L'``.

    Raises:
        ValueError: If the matrix is not positive definite.

    Equation:
        [EQ-LINALG-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    dimension = len(matrix)
    factor = [[0.0] * dimension for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row + 1):
            remainder = matrix[row][column] - math.fsum(
                factor[row][inner] * factor[column][inner]
                for inner in range(column)
            )
            if row == column:
                if remainder <= 0.0 or not math.isfinite(remainder):
                    raise ValueError("covariance must be positive definite")
                factor[row][column] = math.sqrt(remainder)
            else:
                factor[row][column] = remainder / factor[column][column]
    return tuple(tuple(row) for row in factor)


def _solve_lower(matrix: Matrix, right_hand_side: Vector) -> Vector:
    """Solve one lower-triangular linear system by forward substitution.

    Args:
        matrix: Non-singular lower-triangular coefficient matrix.
        right_hand_side: Right-hand-side vector of matching dimension.

    Returns:
        Solution vector.

    Equation:
        [EQ-LINALG-002].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    solution: list[float] = []
    for row, target in enumerate(right_hand_side):
        residual = target - math.fsum(
            matrix[row][column] * solution[column] for column in range(row)
        )
        solution.append(residual / matrix[row][row])
    return tuple(solution)
