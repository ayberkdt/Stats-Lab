"""Dirichlet distribution on the probability simplex.

``X ~ Dirichlet(alpha)`` with ``alpha = (alpha_1, ..., alpha_K)``, ``K >= 2``,
takes values in the simplex ``{x : x_i >= 0, sum_i x_i = 1}``. Its density is
taken with respect to Lebesgue measure on the first ``K - 1`` coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.distributions import Beta
from probstat.distributions._special import stirling_error
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Matrix, Vector, as_vector
from probstat.numerics import log_minus_digamma

from ._support import SIMPLEX_TOLERANCE, log_gamma_variate, positive_vector

_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def log_multivariate_beta(alpha: Vector) -> float:
    """Return ``log B(alpha) = sum_i log Gamma(alpha_i) - log Gamma(alpha_0)``.

    Each ``log Gamma`` is written in Stirling form ``(a - 1/2) log a - a +
    log sqrt(2 pi) + sigma(a)`` with the remainder ``sigma`` of
    :func:`~probstat.distributions._special.stirling_error`; the linear terms
    cancel exactly (``sum alpha_i = alpha_0``), leaving

    ``log B = sum_i (alpha_i - 1/2) log(alpha_i/alpha_0) - ((K-1)/2) log
    alpha_0 + (K-1) log sqrt(2 pi) + sum_i sigma(alpha_i) - sigma(alpha_0)``.

    No term of size ``alpha log alpha`` is formed, so the absolute error is
    a few ulp of the result instead of a few ulp of ``lgamma(alpha_0)``
    (``4e-6`` at ``alpha_0 = 3e9``).

    Args:
        alpha: Positive concentration parameters.

    Returns:
        Logarithm of the Dirichlet normalizing constant.

    Equation:
        [EQ-DIST-DIR-001], [EQ-DIST-DIR-007].

    References:
        [REF-BISHOP-2006, Appendix B], [REF-NIST-DLMF-GAMMA, Sections 5.2
        and 5.11].

    """
    total = math.fsum(alpha)
    count = len(alpha)
    terms = [(value - 0.5) * math.log(value / total) for value in alpha]
    terms.extend(stirling_error(value) for value in alpha)
    terms.append(-stirling_error(total))
    terms.append(-0.5 * (count - 1) * math.log(total))
    terms.append((count - 1) * _LOG_SQRT_TWO_PI)
    return math.fsum(terms)


@dataclass(frozen=True, slots=True)
class Dirichlet:
    """The Dirichlet law ``Dir(alpha)`` on the ``(K-1)``-simplex.

    Args:
        alpha: At least two finite, strictly positive concentration
            parameters.

    References:
        [REF-KOTZ-BALAKRISHNAN-JOHNSON-2000, Chapter 49],
        [REF-BISHOP-2006, Appendix B].

    """

    alpha: Vector

    def __post_init__(self) -> None:
        """Validate the concentration vector.

        Raises:
            TypeError: If ``alpha`` is not an iterable of reals.
            ValueError: If it has fewer than two entries or a non-positive
                or non-finite entry.

        References:
            [EQ-DIST-DIR-001], [REF-BISHOP-2006, Appendix B].

        """
        object.__setattr__(
            self, "alpha", positive_vector(self.alpha, name="alpha", minimum_length=2)
        )

    @property
    def dimension(self) -> int:
        """Return the number of categories ``K``.

        References:
            [EQ-DIST-DIR-001], [REF-BISHOP-2006, Appendix B].

        """
        return len(self.alpha)

    @property
    def concentration(self) -> float:
        """Return the total concentration ``alpha_0 = sum_i alpha_i``.

        References:
            [EQ-DIST-DIR-001], [REF-BISHOP-2006, Appendix B].

        """
        return math.fsum(self.alpha)

    def _checked_point(self, value: Iterable[float]) -> Vector:
        """Validate that a point lies on the hyperplane ``sum x_i = 1``.

        Args:
            value: Candidate point with ``K`` components.

        Returns:
            The point as a tuple of floats.

        Raises:
            TypeError: If components are not finite reals.
            ValueError: If the dimension differs or the components do not
                sum to one within :data:`SIMPLEX_TOLERANCE`.

        References:
            [REF-BISHOP-2006, Appendix B]. No equation is evaluated.

        """
        point = as_vector(value, name="value")
        if len(point) != self.dimension:
            raise ValueError("value dimension must equal the number of categories")
        if abs(math.fsum(point) - 1.0) > SIMPLEX_TOLERANCE:
            raise ValueError("value must sum to one (a point of the simplex)")
        return point

    def log_pdf(self, value: Iterable[float]) -> float:
        """Evaluate the Dirichlet log-density at a point of the simplex.

        A point with a negative component lies on the hyperplane but outside
        the support and has log-density ``-inf``. On a face of the simplex
        (some ``x_i = 0``) the value is the limit of the density: the factor
        ``x_i^(alpha_i - 1)`` is ``+inf`` for ``alpha_i < 1``, ``1`` for
        ``alpha_i = 1`` and ``0`` for ``alpha_i > 1``. If zero and infinite
        factors meet, the limit depends on the path and ``nan`` is returned.

        Args:
            value: Point with ``K`` components summing to one.

        Returns:
            ``log f(x)``.

        Raises:
            TypeError: If components are not finite reals.
            ValueError: If the dimension is wrong or the point is not on the
                hyperplane ``sum x_i = 1``.

        Equation:
            [EQ-DIST-DIR-001].

        References:
            [REF-BISHOP-2006, Appendix B],
            [REF-KOTZ-BALAKRISHNAN-JOHNSON-2000, Chapter 49].

        """
        point = self._checked_point(value)
        if any(coordinate < 0.0 for coordinate in point):
            return -math.inf
        terms: list[float] = []
        diverges = vanishes = False
        for coordinate, shape in zip(point, self.alpha, strict=True):
            if coordinate == 0.0:
                if shape < 1.0:
                    diverges = True
                elif shape > 1.0:
                    vanishes = True
                continue
            terms.append((shape - 1.0) * math.log(coordinate))
        if diverges and vanishes:
            return math.nan
        if vanishes:
            return -math.inf
        if diverges:
            return math.inf
        return math.fsum(terms) - log_multivariate_beta(self.alpha)

    def pdf(self, value: Iterable[float]) -> float:
        """Evaluate the Dirichlet density at a point of the simplex.

        Args:
            value: Point with ``K`` components summing to one.

        Returns:
            ``f(x)``; see :meth:`log_pdf` for the boundary convention.

        Raises:
            TypeError: If components are not finite reals.
            ValueError: If the dimension is wrong or the point is not on the
                hyperplane ``sum x_i = 1``.

        Equation:
            [EQ-DIST-DIR-001].

        References:
            [REF-BISHOP-2006, Appendix B].

        """
        return math.exp(self.log_pdf(value))

    @property
    def mean(self) -> Vector:
        """Return the mean vector ``alpha / alpha_0``.

        References:
            [EQ-DIST-DIR-002], [REF-BISHOP-2006, Appendix B].

        """
        total = self.concentration
        return tuple(value / total for value in self.alpha)

    @property
    def covariance(self) -> Matrix:
        """Return ``(diag(m) - m m') / (alpha_0 + 1)`` with ``m`` the mean.

        References:
            [EQ-DIST-DIR-002], [REF-BISHOP-2006, Appendix B].

        """
        mean = self.mean
        denominator = self.concentration + 1.0
        return tuple(
            tuple(
                ((mean[i] if i == j else 0.0) - mean[i] * mean[j]) / denominator
                for j in range(self.dimension)
            )
            for i in range(self.dimension)
        )

    @property
    def mode(self) -> Vector:
        """Return the mode ``(alpha_i - 1) / (alpha_0 - K)``.

        The interior mode exists only when every ``alpha_i > 1``; otherwise
        the density is unbounded or maximized on a face, and a vector of
        ``nan`` is returned.

        References:
            [EQ-DIST-DIR-003], [REF-BISHOP-2006, Appendix B].

        """
        if any(value <= 1.0 for value in self.alpha):
            return tuple(math.nan for _ in self.alpha)
        denominator = self.concentration - self.dimension
        return tuple((value - 1.0) / denominator for value in self.alpha)

    @property
    def entropy(self) -> float:
        """Return the differential entropy in nats.

        ``H = log B(alpha) + (alpha_0 - K) psi(alpha_0)
        - sum_i (alpha_i - 1) psi(alpha_i)``. With ``psi(a) = log a - L(a)``
        (``L`` from :func:`~probstat.numerics.log_minus_digamma`) and the
        Stirling form of ``log B`` the terms of size ``alpha log alpha``
        cancel in closed form:

        ``H = (1/2) sum_i log m_i - ((K-1)/2) log alpha_0 + (K-1) log
        sqrt(2 pi) + sum_i sigma(alpha_i) - sigma(alpha_0)
        - (alpha_0 - K) L(alpha_0) + sum_i (alpha_i - 1) L(alpha_i)``,
        ``m_i = alpha_i / alpha_0``; every term is of order ``log alpha``.

        References:
            [EQ-DIST-DIR-004], [EQ-DIST-DIR-007], [REF-BISHOP-2006, Appendix B],
            [REF-NIST-DLMF-GAMMA, Section 5.11].

        """
        total = self.concentration
        count = self.dimension
        terms = [0.5 * math.log(value / total) for value in self.alpha]
        terms.extend(stirling_error(value) for value in self.alpha)
        terms.extend(
            (value - 1.0) * log_minus_digamma(value) for value in self.alpha
        )
        terms.append(-stirling_error(total))
        terms.append(-(total - count) * log_minus_digamma(total))
        terms.append(-0.5 * (count - 1) * math.log(total))
        terms.append((count - 1) * _LOG_SQRT_TWO_PI)
        return math.fsum(terms)

    def marginal(self, index: int) -> Beta:
        """Return the marginal law ``X_i ~ Beta(alpha_i, alpha_0 - alpha_i)``.

        Args:
            index: Zero-based category index.

        Returns:
            The beta marginal of the selected coordinate.

        Raises:
            TypeError: If ``index`` is not an integer.
            IndexError: If ``index`` is outside the category range.

        Equation:
            [EQ-DIST-DIR-005].

        References:
            [REF-KOTZ-BALAKRISHNAN-JOHNSON-2000, Chapter 49],
            [REF-BISHOP-2006, Appendix B].

        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < self.dimension:
            raise IndexError("index outside the category range")
        shape = self.alpha[index]
        rest = math.fsum(value for i, value in enumerate(self.alpha) if i != index)
        return Beta(shape, rest)

    def aggregate(self, groups: Iterable[Iterable[int]]) -> Dirichlet:
        """Merge categories: sums over a partition are again Dirichlet.

        If ``G_1, ..., G_m`` partition ``{0, ..., K-1}``, then
        ``(sum_(i in G_1) X_i, ..., sum_(i in G_m) X_i) ~
        Dirichlet(sum_(i in G_1) alpha_i, ..., sum_(i in G_m) alpha_i)``.

        Args:
            groups: At least two non-empty groups of zero-based indices that
                together contain every index exactly once. The group order
                sets the order of the aggregated categories.

        Returns:
            The Dirichlet law of the aggregated coordinates.

        Raises:
            TypeError: If ``groups`` or an index has an invalid type.
            ValueError: If there are fewer than two groups, a group is empty,
                or the groups do not partition the categories.
            IndexError: If an index is outside the category range.

        Equation:
            [EQ-DIST-DIR-005].

        References:
            [REF-KOTZ-BALAKRISHNAN-JOHNSON-2000, Chapter 49].

        """
        try:
            materialized = tuple(tuple(group) for group in groups)
        except TypeError as error:
            raise TypeError("groups must be an iterable of index iterables") from error
        if len(materialized) < 2:
            raise ValueError("aggregation needs at least two groups")
        seen: set[int] = set()
        for group in materialized:
            if not group:
                raise ValueError("every group must be non-empty")
            for index in group:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError("group indices must be integers")
                if not 0 <= index < self.dimension:
                    raise IndexError("index outside the category range")
                if index in seen:
                    raise ValueError("groups must not repeat an index")
                seen.add(index)
        if len(seen) != self.dimension:
            raise ValueError("groups must cover every category")
        return Dirichlet(
            tuple(math.fsum(self.alpha[i] for i in group) for group in materialized)
        )

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw reproducible Dirichlet vectors by normalizing gamma variates.

        With independent ``G_i ~ Gamma(alpha_i, 1)``, ``X = G / sum_j G_j``.
        The gamma variates are formed in log space and normalized by a
        log-sum-exp, so small concentrations do not produce ``0 / 0``; a
        coordinate whose share is below the float range is returned as
        ``0.0``.

        Args:
            size: Number of vectors, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of points of the simplex.

        Raises:
            TypeError: If ``size`` or ``seed`` has an invalid type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-DIR-006].

        References:
            [REF-DEVROYE-1986, Chapter XI], [REF-MARSAGLIA-TSANG-2000].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        observations: list[Vector] = []
        for _ in range(checked_size):
            logs = tuple(log_gamma_variate(shape, generator) for shape in self.alpha)
            peak = max(logs)
            weights = tuple(math.exp(value - peak) for value in logs)
            total = math.fsum(weights)
            observations.append(tuple(weight / total for weight in weights))
        return tuple(observations)
