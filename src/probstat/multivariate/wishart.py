"""Wishart and inverse-Wishart laws on symmetric positive-definite matrices.

Parameterization: ``W ~ Wishart_p(S, nu)`` has ``E[W] = nu S`` and is the law
of ``sum_(k=1..nu) z_k z_k'`` with ``z_k ~ N(0, S)`` when ``nu`` is an
integer. ``Sigma ~ InverseWishart_p(Psi, nu)`` means
``Sigma^-1 ~ Wishart_p(Psi^-1, nu)``. Both require real ``nu > p - 1``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, cholesky, symmetrize

from ._support import (
    chi_square_variate,
    divergent_matrix,
    gram_lower,
    log_det_from_factor,
    lower_inverse,
    lower_product,
    nan_matrix,
    scaled,
    spd_matrix,
    trace_of_solve,
)


def multivariate_log_gamma(a: float, dimension: int) -> float:
    """Return the multivariate log-gamma function ``log Gamma_p(a)``.

    ``Gamma_p(a) = pi^(p(p-1)/4) prod_(j=1..p) Gamma(a - (j-1)/2)``, the
    integral of ``etr(-X) |X|^(a-(p+1)/2)`` over positive-definite ``X``.

    Args:
        a: Real argument with ``a > (p - 1)/2``.
        dimension: Positive integer order ``p``.

    Returns:
        ``log Gamma_p(a)``; ``p = 1`` gives ``log Gamma(a)``.

    Raises:
        TypeError: If ``a`` is not real or ``dimension`` is not an integer.
        ValueError: If ``dimension < 1`` or ``a <= (p - 1)/2``.

    Equation:
        [EQ-NUM-MVLGAMMA-001].

    References:
        [REF-NIST-DLMF-MATRIX-ARGUMENT, Section 35.3],
        [REF-ANDERSON-2003, Section 7.2].

    """
    value = require_finite_real(a, name="a")
    order = require_non_negative_integer(dimension, name="dimension")
    if order < 1:
        raise ValueError("dimension must be at least 1")
    if value <= 0.5 * (order - 1):
        raise ValueError("a must exceed (dimension - 1) / 2")
    return 0.25 * order * (order - 1) * math.log(math.pi) + math.fsum(
        math.lgamma(value - 0.5 * j) for j in range(order)
    )


def _checked_dof(dof: object, dimension: int) -> float:
    """Validate real degrees of freedom ``nu > p - 1``.

    Args:
        dof: Candidate degrees of freedom.
        dimension: Matrix order ``p``.

    Returns:
        The degrees of freedom as a float.

    Raises:
        TypeError: If ``dof`` is not real.
        ValueError: If ``dof`` is non-finite or ``dof <= p - 1``.

    References:
        [REF-ANDERSON-2003, Section 7.2]. No equation is evaluated.

    """
    nu = require_finite_real(dof, name="dof")
    if nu <= dimension - 1:
        raise ValueError("dof must exceed dimension - 1")
    return nu


def _checked_argument(
    value: Iterable[Iterable[object]], dimension: int
) -> tuple[Matrix, Matrix] | None:
    """Return a density argument and its Cholesky factor, or ``None`` if not PD.

    Args:
        value: Candidate symmetric matrix.
        dimension: Required order ``p``.

    Returns:
        ``(X, L)`` with ``X`` symmetrized and ``L`` its lower Cholesky factor,
        or ``None`` for a symmetric matrix that is not positive definite
        (outside the support).

    Raises:
        TypeError: If entries are not real.
        ValueError: If the matrix is not square of order ``p`` or not
            symmetric.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    matrix = symmetrize(value, name="value")
    if len(matrix) != dimension:
        raise ValueError("value order must equal the distribution dimension")
    try:
        return matrix, cholesky(matrix)
    except ValueError:
        return None


def _bartlett_factor(dimension: int, dof: float, generator: random.Random) -> Matrix:
    """Draw the lower Bartlett factor ``A`` with ``A A' ~ Wishart_p(I, nu)``.

    ``A_ii = sqrt(chi^2(nu - i))`` for ``i = 0..p-1`` and ``A_ij ~ N(0, 1)``
    for ``i > j``, all independent.

    Args:
        dimension: Matrix order ``p``.
        dof: Degrees of freedom ``nu > p - 1``.
        generator: Call-local pseudo-random generator.

    Returns:
        Lower-triangular ``A``.

    Equation:
        [EQ-DIST-WISH-003].

    References:
        [REF-SMITH-HOCKING-1972], [REF-ANDERSON-2003, Section 7.2].

    """
    rows: list[tuple[float, ...]] = []
    for i in range(dimension):
        row = [generator.gauss(0.0, 1.0) for _ in range(i)]
        row.append(math.sqrt(chi_square_variate(dof - i, generator)))
        row.extend(0.0 for _ in range(i + 1, dimension))
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class Wishart:
    """The Wishart law ``Wishart_p(S, nu)`` on positive-definite matrices.

    Args:
        scale: Symmetric positive-definite ``p x p`` scale ``S``.
        dof: Real degrees of freedom ``nu > p - 1``.

    References:
        [REF-ANDERSON-2003, Section 7.2], [REF-MARDIA-KENT-BIBBY-1979, Chapter 3],
        [REF-GELMAN-ETAL-2013, Appendix A].

    """

    scale: Matrix
    dof: float
    _cholesky: Matrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the scale matrix and the degrees of freedom.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If the scale is not symmetric positive definite or
                ``dof <= p - 1``.

        References:
            [EQ-DIST-WISH-001], [REF-ANDERSON-2003, Section 7.2].

        """
        matrix, lower = spd_matrix(self.scale, name="scale")
        object.__setattr__(self, "scale", matrix)
        object.__setattr__(self, "dof", _checked_dof(self.dof, len(matrix)))
        object.__setattr__(self, "_cholesky", lower)

    @property
    def dimension(self) -> int:
        """Return the matrix order ``p``.

        References:
            [EQ-DIST-WISH-001], [REF-ANDERSON-2003, Section 7.2].

        """
        return len(self.scale)

    def log_pdf(self, value: Iterable[Iterable[object]]) -> float:
        """Evaluate the Wishart log-density at a symmetric matrix.

        ``log f(X) = ((nu - p - 1)/2) log|X| - tr(S^-1 X)/2 - (nu p/2) log 2
        - (nu/2) log|S| - log Gamma_p(nu/2)``, with respect to Lebesgue
        measure on the ``p(p+1)/2`` free entries. A symmetric matrix that is
        not positive definite has log-density ``-inf``.

        Args:
            value: Symmetric ``p x p`` matrix.

        Returns:
            ``log f(X)``.

        Raises:
            TypeError: If entries are not real.
            ValueError: If the matrix has the wrong order or is not symmetric.

        Equation:
            [EQ-DIST-WISH-001], [EQ-NUM-MVLGAMMA-001].

        References:
            [REF-ANDERSON-2003, Section 7.2], [REF-GELMAN-ETAL-2013, Appendix A].

        """
        checked = _checked_argument(value, self.dimension)
        if checked is None:
            return -math.inf
        argument, lower = checked
        p = self.dimension
        nu = self.dof
        return math.fsum(
            (
                0.5 * (nu - p - 1.0) * log_det_from_factor(lower),
                -0.5 * trace_of_solve(self._cholesky, argument),
                -0.5 * nu * p * math.log(2.0),
                -0.5 * nu * log_det_from_factor(self._cholesky),
                -multivariate_log_gamma(0.5 * nu, p),
            )
        )

    def pdf(self, value: Iterable[Iterable[object]]) -> float:
        """Evaluate the Wishart density at a symmetric matrix.

        Args:
            value: Symmetric ``p x p`` matrix.

        Returns:
            ``f(X)``; zero off the positive-definite cone.

        Raises:
            TypeError: If entries are not real.
            ValueError: If the matrix has the wrong order or is not symmetric.

        Equation:
            [EQ-DIST-WISH-001].

        References:
            [REF-ANDERSON-2003, Section 7.2].

        """
        return math.exp(self.log_pdf(value))

    @property
    def mean(self) -> Matrix:
        """Return ``E[W] = nu S``.

        References:
            [EQ-DIST-WISH-002], [REF-GELMAN-ETAL-2013, Appendix A].

        """
        return scaled(self.scale, self.dof)

    @property
    def mode(self) -> Matrix:
        """Return ``(nu - p - 1) S`` for ``nu >= p + 1`` and ``nan`` otherwise.

        For ``p - 1 < nu < p + 1`` the density is unbounded near the boundary
        of the cone and has no interior mode.

        References:
            [EQ-DIST-WISH-002], [REF-BISHOP-2006, Appendix B].

        """
        excess = self.dof - self.dimension - 1.0
        if excess < 0.0:
            return nan_matrix(self.dimension)
        return scaled(self.scale, excess)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[Matrix, ...]:
        """Draw reproducible matrices by the Bartlett decomposition.

        ``W = (L A)(L A)'`` with ``L L' = S`` and the Bartlett factor ``A``.

        Args:
            size: Number of matrices, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of symmetric positive-definite matrices.

        Raises:
            TypeError: If ``size`` or ``seed`` has an invalid type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-WISH-003].

        References:
            [REF-SMITH-HOCKING-1972], [REF-ANDERSON-2003, Section 7.2].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(
            gram_lower(
                lower_product(
                    self._cholesky,
                    _bartlett_factor(self.dimension, self.dof, generator),
                )
            )
            for _ in range(checked_size)
        )


@dataclass(frozen=True, slots=True)
class InverseWishart:
    """The inverse-Wishart law ``InverseWishart_p(Psi, nu)``.

    Args:
        scale: Symmetric positive-definite ``p x p`` scale ``Psi``.
        dof: Real degrees of freedom ``nu > p - 1``.

    References:
        [REF-ANDERSON-2003, Section 7.7], [REF-GELMAN-ETAL-2013, Appendix A].

    """

    scale: Matrix
    dof: float
    _cholesky: Matrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the scale matrix and the degrees of freedom.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If the scale is not symmetric positive definite or
                ``dof <= p - 1``.

        References:
            [EQ-DIST-IWISH-001], [REF-ANDERSON-2003, Section 7.7].

        """
        matrix, lower = spd_matrix(self.scale, name="scale")
        object.__setattr__(self, "scale", matrix)
        object.__setattr__(self, "dof", _checked_dof(self.dof, len(matrix)))
        object.__setattr__(self, "_cholesky", lower)

    @property
    def dimension(self) -> int:
        """Return the matrix order ``p``.

        References:
            [EQ-DIST-IWISH-001], [REF-ANDERSON-2003, Section 7.7].

        """
        return len(self.scale)

    def log_pdf(self, value: Iterable[Iterable[object]]) -> float:
        """Evaluate the inverse-Wishart log-density at a symmetric matrix.

        ``log f(X) = (nu/2) log|Psi| - (nu p/2) log 2 - log Gamma_p(nu/2)
        - ((nu + p + 1)/2) log|X| - tr(Psi X^-1)/2``. A symmetric matrix that
        is not positive definite has log-density ``-inf``.

        Args:
            value: Symmetric ``p x p`` matrix.

        Returns:
            ``log f(X)``.

        Raises:
            TypeError: If entries are not real.
            ValueError: If the matrix has the wrong order or is not symmetric.

        Equation:
            [EQ-DIST-IWISH-001], [EQ-NUM-MVLGAMMA-001].

        References:
            [REF-ANDERSON-2003, Section 7.7], [REF-GELMAN-ETAL-2013, Appendix A].

        """
        checked = _checked_argument(value, self.dimension)
        if checked is None:
            return -math.inf
        lower = checked[1]
        p = self.dimension
        nu = self.dof
        return math.fsum(
            (
                0.5 * nu * log_det_from_factor(self._cholesky),
                -0.5 * nu * p * math.log(2.0),
                -multivariate_log_gamma(0.5 * nu, p),
                -0.5 * (nu + p + 1.0) * log_det_from_factor(lower),
                -0.5 * trace_of_solve(lower, self.scale),
            )
        )

    def pdf(self, value: Iterable[Iterable[object]]) -> float:
        """Evaluate the inverse-Wishart density at a symmetric matrix.

        Args:
            value: Symmetric ``p x p`` matrix.

        Returns:
            ``f(X)``; zero off the positive-definite cone.

        Raises:
            TypeError: If entries are not real.
            ValueError: If the matrix has the wrong order or is not symmetric.

        Equation:
            [EQ-DIST-IWISH-001].

        References:
            [REF-ANDERSON-2003, Section 7.7].

        """
        return math.exp(self.log_pdf(value))

    @property
    def mean(self) -> Matrix:
        """Return ``Psi / (nu - p - 1)`` for ``nu > p + 1``.

        For ``p - 1 < nu <= p + 1`` the diagonal moments diverge (``inf``)
        and the off-diagonal entries are reported as ``nan``.

        References:
            [EQ-DIST-IWISH-002], [REF-GELMAN-ETAL-2013, Appendix A].

        """
        excess = self.dof - self.dimension - 1.0
        if excess > 0.0:
            return scaled(self.scale, 1.0 / excess)
        return divergent_matrix(self.dimension)

    @property
    def mode(self) -> Matrix:
        """Return the mode ``Psi / (nu + p + 1)``.

        References:
            [EQ-DIST-IWISH-002], [REF-ANDERSON-2003, Section 7.7].

        """
        return scaled(self.scale, 1.0 / (self.dof + self.dimension + 1.0))

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[Matrix, ...]:
        """Draw reproducible matrices by an inverted Bartlett decomposition.

        With ``Psi = C C'`` and the Bartlett factor ``A`` of
        ``Wishart_p(I, nu)``, ``Sigma = (C A^-T)(C A^-T)'`` satisfies
        ``Sigma^-1 = C^-T A A' C^-1 ~ Wishart_p(Psi^-1, nu)``. Only a
        triangular inverse of ``A`` is formed.

        Args:
            size: Number of matrices, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of symmetric positive-definite matrices.

        Raises:
            TypeError: If ``size`` or ``seed`` has an invalid type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-IWISH-003], [EQ-DIST-WISH-003].

        References:
            [REF-SMITH-HOCKING-1972], [REF-ANDERSON-2003, Section 7.7].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        p = self.dimension
        factor = self._cholesky
        draws: list[Matrix] = []
        for _ in range(checked_size):
            inverse = lower_inverse(_bartlett_factor(p, self.dof, generator))
            # G = C A^-T: G[i][k] = sum_l C[i][l] (A^-1)[k][l], l <= min(i, k).
            product = tuple(
                tuple(
                    math.fsum(
                        factor[i][ell] * inverse[k][ell] for ell in range(min(i, k) + 1)
                    )
                    for k in range(p)
                )
                for i in range(p)
            )
            result = [[0.0] * p for _ in range(p)]
            for i in range(p):
                for j in range(i + 1):
                    entry = math.fsum(product[i][k] * product[j][k] for k in range(p))
                    result[i][j] = entry
                    result[j][i] = entry
            draws.append(tuple(tuple(row) for row in result))
        return tuple(draws)
