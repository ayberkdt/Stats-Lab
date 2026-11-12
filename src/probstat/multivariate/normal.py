"""Multivariate normal distribution using the shared Cholesky algebra.

The covariance factorization, whitening solves, and sampling transform go
through the active dense backend (:func:`probstat.linalg.get_backend`,
ADR-0081). A law keeps the factorization of the backend that was active when
it was constructed; under the default ``"python"`` backend every result is
bit-identical to the direct :mod:`probstat.linalg` routines.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from probstat.distributions import Normal
from probstat.distributions._autodiff import admath, any_variable
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import (
    CholeskyFactorization,
    Matrix,
    Vector,
    as_vector,
    dot,
    get_backend,
)

from ._support import real_vector


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
    _factor: CholeskyFactorization = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize vectors and validate covariance through Cholesky factorization.

        Raises:
            TypeError: If inputs are not iterable finite real values.
            ValueError: If dimensions disagree or covariance is not symmetric
                positive definite.

        References:
            [EQ-DIST-MVN-001], [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

        """
        mean = real_vector(self.mean, name="mean")
        if not mean:
            raise ValueError("mean must contain at least one component")
        try:
            raw_rows = tuple(self.covariance)
        except TypeError as error:
            raise TypeError("covariance must be an iterable of rows") from error
        if len(raw_rows) != len(mean):
            raise ValueError("covariance row count must equal mean dimension")
        rows = tuple(
            as_vector(row, name=f"covariance[{index}]")
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
        try:
            factor = get_backend().factorize(covariance)
        except ValueError as error:
            raise ValueError("covariance must be positive definite") from error
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "_cholesky", factor.lower)
        object.__setattr__(self, "_factor", factor)

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
        point = as_vector(value, name="value")
        if len(point) != self.dimension:
            raise ValueError("value dimension must equal distribution dimension")
        centered = tuple(
            coordinate - location
            for coordinate, location in zip(point, self.mean, strict=True)
        )
        solved = self._factor.solve_lower(centered)
        return math.fsum(component * component for component in solved)

    def _variable_mahalanobis_squared(self, point: Vector) -> float:
        """Return the squared Mahalanobis distance for variable entries.

        Forward substitution with the float Cholesky factor ``L`` written with
        operators, ``s_i = (c_i - sum_(j<i) L_ij s_j) / L_ii`` with
        ``c = x - mu``, so derivatives with respect to the point and the mean
        flow (ADR-0090).

        Args:
            point: Validated point (floats and variables).

        Returns:
            ``|L^-1 (x - mu)|^2``.

        Raises:
            ValueError: If the dimension differs.

        Equation:
            [EQ-DIST-MVN-002], [EQ-LINALG-002].

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 1],
            [REF-GOLUB-VANLOAN-2013, Section 3.1].

        """
        if len(point) != self.dimension:
            raise ValueError("value dimension must equal distribution dimension")
        lower = self._cholesky
        solved: list[float] = []
        for row, (coordinate, location) in enumerate(
            zip(point, self.mean, strict=True)
        ):
            residual = coordinate - location
            if row:
                residual = residual - admath.fsum(
                    lower[row][column] * solved[column] for column in range(row)
                )
            solved.append(residual / lower[row][row])
        return admath.fsum(component * component for component in solved)

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
        point = real_vector(value, name="value")
        if any_variable(*point, *self.mean):
            quadratic = self._variable_mahalanobis_squared(point)
        else:
            quadratic = self.mahalanobis_squared(point)
        log_determinant = self._factor.log_det()
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

    def _checked_indices(self, indices: Iterable[int]) -> tuple[int, ...]:
        """Validate a non-empty list of distinct in-range component indices.

        Args:
            indices: Zero-based component indices.

        Returns:
            Indices as a tuple in the order supplied.

        Raises:
            TypeError: If ``indices`` is not iterable or holds a non-integer.
            ValueError: If it is empty or repeats an index.
            IndexError: If an index is outside the vector dimension.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        try:
            selected = tuple(indices)
        except TypeError as error:
            raise TypeError("indices must be an iterable of integers") from error
        if not selected:
            raise ValueError("indices must be non-empty")
        for index in selected:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("indices must be integers")
            if not 0 <= index < self.dimension:
                raise IndexError("index outside distribution dimension")
        if len(set(selected)) != len(selected):
            raise ValueError("indices must be distinct")
        return selected

    def marginal_subset(self, indices: Iterable[int]) -> MultivariateNormal:
        """Return the joint marginal law of a subset of components.

        Args:
            indices: Distinct zero-based indices; their order sets the order of
                the returned components.

        Returns:
            ``Normal(mu_S, Sigma_SS)`` for the selected index list ``S``.

        Raises:
            TypeError: If indices are not integers.
            ValueError: If indices are empty or repeated.
            IndexError: If an index is outside the vector dimension.

        Equation:
            [EQ-DIST-MVN-006].

        References:
            [REF-BISHOP-2006, Section 2.3.2],
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 3].

        """
        selected = self._checked_indices(indices)
        return MultivariateNormal(
            tuple(self.mean[i] for i in selected),
            tuple(tuple(self.covariance[i][j] for j in selected) for i in selected),
        )

    def conditional(
        self, indices: Iterable[int], values: Iterable[float]
    ) -> MultivariateNormal:
        """Condition on observed components and return the remaining law.

        With ``b`` the observed indices and ``a`` the remaining indices in
        ascending order, the conditional law is Gaussian with mean
        ``mu_a + Sigma_ab Sigma_bb^-1 (x_b - mu_b)`` and Schur-complement
        covariance ``Sigma_aa - Sigma_ab Sigma_bb^-1 Sigma_ba``. Both are
        evaluated through the Cholesky factor ``Sigma_bb = L L'`` as
        ``W = L^-1 Sigma_ba``, so the covariance ``Sigma_aa - W'W`` is
        symmetric by construction and no inverse is formed.

        Args:
            indices: Distinct zero-based indices of the observed components.
            values: Observed values, one per index and in the same order.

        Returns:
            Conditional multivariate normal of the unobserved components,
            ordered by ascending original index.

        Raises:
            TypeError: If indices or values have invalid types.
            ValueError: If indices are empty, repeated, or cover every
                component; if ``values`` has the wrong length; or if the
                Schur complement is not numerically positive definite.
            IndexError: If an index is outside the vector dimension.

        Equation:
            [EQ-DIST-MVN-007].

        References:
            [REF-BISHOP-2006, Section 2.3.1],
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 3].

        """
        observed = self._checked_indices(indices)
        point = as_vector(values, name="values")
        if len(point) != len(observed):
            raise ValueError("values must have one entry per observed index")
        observed_set = set(observed)
        free = tuple(i for i in range(self.dimension) if i not in observed_set)
        if not free:
            raise ValueError("conditioning on every component leaves no free law")
        factor = get_backend().factorize(
            tuple(tuple(self.covariance[i][j] for j in observed) for i in observed)
        )
        whitened = factor.solve_lower(
            tuple(
                value - self.mean[index]
                for index, value in zip(observed, point, strict=True)
            ),
        )
        weights = tuple(
            factor.solve_lower(tuple(self.covariance[b][a] for b in observed))
            for a in free
        )
        mean = tuple(
            self.mean[a] + dot(weight, whitened)
            for a, weight in zip(free, weights, strict=True)
        )
        covariance = tuple(
            tuple(
                self.covariance[a][c] - dot(weights[row], weights[column])
                for column, c in enumerate(free)
            )
            for row, a in enumerate(free)
        )
        return MultivariateNormal(mean, covariance)

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
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw reproducible pseudo-random multivariate normal observations.

        Args:
            size: Number of vector observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy. Negative, boolean,
                float, and text seeds are rejected.

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
        generator = resolve_stream(seed).generator()
        observations: list[Vector] = []
        for _ in range(checked_size):
            independent = tuple(generator.gauss(0.0, 1.0) for _ in self.mean)
            observation = tuple(
                location + offset
                for location, offset in zip(
                    self.mean, self._factor.multiply_lower(independent), strict=True
                )
            )
            if not all(math.isfinite(component) for component in observation):
                raise OverflowError("multivariate normal sample is not finite")
            observations.append(observation)
        return tuple(observations)
