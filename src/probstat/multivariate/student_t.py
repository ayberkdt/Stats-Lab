"""Multivariate Student-t distribution with location, scale matrix and dof."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from probstat.distributions import StudentT
from probstat.distributions._special import log_gamma_ratio
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Matrix, Vector, as_vector, solve_lower

from ._support import (
    chi_square_variate,
    divergent_matrix,
    log_det_from_factor,
    nan_matrix,
    positive_real,
    scaled,
    spd_matrix,
)


@dataclass(frozen=True, slots=True)
class MultivariateStudentT:
    """The ``p``-variate Student-t law ``t_nu(mu, Sigma)``.

    ``X = mu + Z / sqrt(W / nu)`` with ``Z ~ N(0, Sigma)`` and an independent
    ``W ~ chi^2(nu)``. ``Sigma`` is the scale matrix, not the covariance.

    Args:
        location: Non-empty finite location vector ``mu``.
        scale_matrix: Symmetric positive-definite ``p x p`` scale ``Sigma``.
        dof: Finite positive degrees of freedom ``nu`` (real values allowed).

    References:
        [REF-KOTZ-NADARAJAH-2004, Chapter 1],
        [REF-GELMAN-ETAL-2013, Appendix A].

    """

    location: Vector
    scale_matrix: Matrix
    dof: float
    _cholesky: Matrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate parameters and factor the scale matrix.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If dimensions disagree, the scale matrix is not
                symmetric positive definite, or ``dof`` is not positive.

        References:
            [EQ-DIST-MVT-001], [REF-KOTZ-NADARAJAH-2004, Chapter 1].

        """
        location = as_vector(self.location, name="location")
        if not location:
            raise ValueError("location must contain at least one component")
        matrix, lower = spd_matrix(self.scale_matrix, name="scale_matrix")
        if len(matrix) != len(location):
            raise ValueError("scale_matrix order must equal the location dimension")
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale_matrix", matrix)
        object.__setattr__(self, "dof", positive_real(self.dof, name="dof"))
        object.__setattr__(self, "_cholesky", lower)

    @property
    def dimension(self) -> int:
        """Return the vector dimension ``p``.

        References:
            [EQ-DIST-MVT-001], [REF-KOTZ-NADARAJAH-2004, Chapter 1].

        """
        return len(self.location)

    def mahalanobis_squared(self, value: Iterable[float]) -> float:
        """Return ``delta = (x - mu)' Sigma^-1 (x - mu)`` through the factor.

        Args:
            value: Finite vector with the distribution dimension.

        Returns:
            Non-negative squared Mahalanobis distance in the scale metric.

        Raises:
            TypeError: If components are not finite reals.
            ValueError: If the dimension differs.

        Equation:
            [EQ-DIST-MVN-002].

        References:
            [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

        """
        point = as_vector(value, name="value")
        if len(point) != self.dimension:
            raise ValueError("value dimension must equal distribution dimension")
        whitened = solve_lower(
            self._cholesky,
            tuple(x - m for x, m in zip(point, self.location, strict=True)),
        )
        return math.fsum(component * component for component in whitened)

    def log_pdf(self, value: Iterable[float]) -> float:
        """Evaluate the multivariate Student-t log-density.

        ``log f = log Gamma((nu + p)/2) - log Gamma(nu/2) - (p/2) log(nu pi)
        - (1/2) log|Sigma| - ((nu + p)/2) log(1 + delta / nu)``.

        Args:
            value: Finite vector with the distribution dimension.

        Returns:
            Natural logarithm of the density.

        Raises:
            TypeError: If components are not finite reals.
            ValueError: If the dimension differs.

        Equation:
            [EQ-DIST-MVT-001].

        The gamma ratio is evaluated with the Stirling-series form of
        ``distributions._special.log_gamma_ratio``, so large ``nu`` does not
        lose digits to the cancellation of two ``nu log nu`` terms.

        References:
            [REF-KOTZ-NADARAJAH-2004, Chapter 1],
            [REF-GELMAN-ETAL-2013, Appendix A], [REF-NIST-DLMF-GAMMA, Section 5.11].

        """
        delta = self.mahalanobis_squared(value)
        p = self.dimension
        nu = self.dof
        return math.fsum(
            (
                log_gamma_ratio(0.5 * nu, 0.5 * p),
                -0.5 * p * math.log(nu * math.pi),
                -0.5 * log_det_from_factor(self._cholesky),
                -0.5 * (nu + p) * math.log1p(delta / nu),
            )
        )

    def pdf(self, value: Iterable[float]) -> float:
        """Evaluate the multivariate Student-t density.

        Args:
            value: Finite vector with the distribution dimension.

        Returns:
            Density with respect to Lebesgue measure.

        Raises:
            TypeError: If components are not finite reals.
            ValueError: If the dimension differs.

        Equation:
            [EQ-DIST-MVT-001].

        References:
            [REF-KOTZ-NADARAJAH-2004, Chapter 1].

        """
        return math.exp(self.log_pdf(value))

    @property
    def mean(self) -> Vector:
        """Return ``mu`` for ``nu > 1`` and a vector of ``nan`` otherwise.

        References:
            [EQ-DIST-MVT-002], [REF-KOTZ-NADARAJAH-2004, Chapter 1].

        """
        if self.dof > 1.0:
            return self.location
        return tuple(math.nan for _ in self.location)

    @property
    def covariance(self) -> Matrix:
        """Return ``nu / (nu - 2) Sigma`` for ``nu > 2``.

        For ``1 < nu <= 2`` the variances diverge (``inf`` diagonal) and the
        cross moments are not absolutely integrable (``nan``); for
        ``nu <= 1`` the mean is undefined and every entry is ``nan``.

        References:
            [EQ-DIST-MVT-002], [REF-KOTZ-NADARAJAH-2004, Chapter 1].

        """
        if self.dof > 2.0:
            return scaled(self.scale_matrix, self.dof / (self.dof - 2.0))
        if self.dof > 1.0:
            return divergent_matrix(self.dimension)
        return nan_matrix(self.dimension)

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
            Project validation policy. No equation is evaluated.

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

    def marginal_subset(self, indices: Iterable[int]) -> MultivariateStudentT:
        """Return the joint marginal law of a subset of components.

        The marginal of ``X_S`` is ``t_nu(mu_S, Sigma_SS)`` with the same
        degrees of freedom.

        Args:
            indices: Distinct zero-based indices; their order sets the order
                of the returned components.

        Returns:
            The multivariate Student-t marginal of the selected components.

        Raises:
            TypeError: If indices are not integers.
            ValueError: If indices are empty or repeated.
            IndexError: If an index is outside the vector dimension.

        Equation:
            [EQ-DIST-MVT-003].

        References:
            [REF-KOTZ-NADARAJAH-2004, Chapter 1].

        """
        selected = self._checked_indices(indices)
        return MultivariateStudentT(
            tuple(self.location[i] for i in selected),
            tuple(tuple(self.scale_matrix[i][j] for j in selected) for i in selected),
            self.dof,
        )

    def marginal(self, index: int) -> StudentT:
        """Return the univariate marginal ``t_nu(mu_i, sqrt(Sigma_ii))``.

        Args:
            index: Zero-based component index.

        Returns:
            Location-scale Student-t law of one component.

        Raises:
            TypeError: If ``index`` is not an integer.
            IndexError: If ``index`` is outside the vector dimension.

        Equation:
            [EQ-DIST-MVT-003].

        References:
            [REF-KOTZ-NADARAJAH-2004, Chapter 1].

        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < self.dimension:
            raise IndexError("index outside distribution dimension")
        return StudentT(
            self.dof, self.location[index], math.sqrt(self.scale_matrix[index][index])
        )

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw reproducible vectors as ``mu + L z / sqrt(W / nu)``.

        ``z`` has independent standard-normal components, ``L L' = Sigma``,
        and ``W ~ chi^2(nu)`` is independent of ``z``.

        Args:
            size: Number of vectors, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of vectors.

        Raises:
            TypeError: If ``size`` or ``seed`` has an invalid type.
            ValueError: If ``size`` or an integer seed is negative.
            OverflowError: If a component exceeds the float range.

        Equation:
            [EQ-DIST-MVT-004], [EQ-DIST-MVN-005].

        References:
            [REF-KOTZ-NADARAJAH-2004, Chapter 1], [REF-DEVROYE-1986, Chapter XI].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        p = self.dimension
        observations: list[Vector] = []
        for _ in range(checked_size):
            normals = tuple(generator.gauss(0.0, 1.0) for _ in range(p))
            chi_square = chi_square_variate(self.dof, generator)
            factor = math.sqrt(self.dof / chi_square) if chi_square > 0.0 else math.inf
            observation = tuple(
                self.location[row]
                + factor
                * math.fsum(
                    self._cholesky[row][column] * normals[column]
                    for column in range(row + 1)
                )
                for row in range(p)
            )
            if not all(math.isfinite(component) for component in observation):
                raise OverflowError("multivariate Student-t sample is not finite")
            observations.append(observation)
        return tuple(observations)
