"""Multinomial and Dirichlet-multinomial laws of category counts.

Both laws live on the compositions ``{x in Z^K : x_i >= 0, sum_i x_i = n}``.
The Dirichlet-multinomial (Polya) law is the multinomial with Dirichlet
distributed probabilities integrated out; it is the posterior predictive law
of counts under a Dirichlet prior.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.distributions import Binomial
from probstat.distributions._autodiff import admath
from probstat.distributions._special import log_gamma_ratio
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Matrix, Vector

from ._support import (
    count_vector,
    log_gamma_variate,
    positive_vector,
    probability_vector,
)

EXACT_COEFFICIENT_LIMIT = 1000
"""Largest ``n`` for which the multinomial coefficient is formed as an integer."""


def log_multinomial_coefficient(counts: tuple[int, ...]) -> float:
    """Return ``log(n! / prod_i x_i!)`` for non-negative counts.

    For ``n <= EXACT_COEFFICIENT_LIMIT`` the coefficient is the exact integer
    ``prod_i C(n - x_1 - ... - x_(i-1), x_i)``, so only the final logarithm
    is rounded. Larger ``n`` use the log-gamma form
    ``log Gamma(n + 1) - sum_i log Gamma(x_i + 1)``.

    Args:
        counts: Non-negative integer counts.

    Returns:
        Natural logarithm of the multinomial coefficient.

    Equation:
        [EQ-DIST-MULT-001].

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35],
        [REF-NIST-DLMF-GAMMA, Section 5.2].

    """
    total = sum(counts)
    if total <= EXACT_COEFFICIENT_LIMIT:
        coefficient = 1
        remaining = total
        for count in counts:
            coefficient *= math.comb(remaining, count)
            remaining -= count
        return math.log(coefficient)
    return math.lgamma(total + 1.0) - math.fsum(
        math.lgamma(count + 1.0) for count in counts
    )


def _binomial_draw(trials: int, probability: float, generator: random.Random) -> int:
    """Draw one binomial count as a sum of Bernoulli trials.

    Args:
        trials: Non-negative number of trials.
        probability: Success probability in ``[0, 1]``.
        generator: Call-local pseudo-random generator.

    Returns:
        Number of successes.

    References:
        [REF-DEVROYE-1986, Chapter X]. The defining Bernoulli-sum construction
        is used, as in ``distributions.Binomial.sample``; no new equation.

    """
    if probability <= 0.0:
        return 0
    if probability >= 1.0:
        return trials
    return sum(generator.random() < probability for _ in range(trials))


def _multinomial_draw(
    trials: int, probabilities: Vector, generator: random.Random
) -> tuple[int, ...]:
    """Draw one count vector by sequential conditional binomials.

    ``X_1 ~ Bin(n, p_1)`` and, given the earlier counts,
    ``X_i ~ Bin(n - sum_(j<i) X_j, p_i / (1 - sum_(j<i) p_j))``; the last
    category takes the remainder.

    Args:
        trials: Total count ``n``.
        probabilities: Probability vector.
        generator: Call-local pseudo-random generator.

    Returns:
        Counts summing to ``trials``.

    Equation:
        [EQ-DIST-MULT-004].

    References:
        [REF-DEVROYE-1986, Chapter XI].

    """
    remaining = trials
    mass = 1.0
    counts: list[int] = []
    for probability in probabilities[:-1]:
        if remaining == 0 or mass <= 0.0:
            counts.append(0)
            continue
        count = _binomial_draw(remaining, min(1.0, probability / mass), generator)
        counts.append(count)
        remaining -= count
        mass -= probability
    counts.append(remaining)
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class Multinomial:
    """Counts of ``n`` independent categorical trials, ``Mult(n, p)``.

    Args:
        n: Non-negative number of trials.
        probabilities: At least two category probabilities in ``[0, 1]``
            summing to one within ``1e-10``; they are normalized.

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35],
        [REF-BISHOP-2006, Appendix B].

    """

    n: int
    probabilities: Vector

    def __post_init__(self) -> None:
        """Validate the trial count and the probability vector.

        Raises:
            TypeError: If ``n`` is not an integer or a probability is not real.
            ValueError: If ``n < 0`` or the probabilities are not a
                probability vector with at least two categories.

        References:
            [EQ-DIST-MULT-001], [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35].

        """
        object.__setattr__(self, "n", require_non_negative_integer(self.n, name="n"))
        object.__setattr__(
            self,
            "probabilities",
            probability_vector(self.probabilities, name="probabilities"),
        )

    @property
    def dimension(self) -> int:
        """Return the number of categories ``K``.

        References:
            [EQ-DIST-MULT-001], [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35].

        """
        return len(self.probabilities)

    def log_pmf(self, counts: Iterable[int]) -> float:
        """Evaluate the multinomial log-mass of a count vector.

        ``log P(x) = log(n! / prod x_i!) + sum_i x_i log p_i`` with
        ``0 log 0 = 0``. Count vectors with a negative entry or a total other
        than ``n``, and positive counts in a zero-probability category, have
        log-mass ``-inf``.

        Args:
            counts: ``K`` integer counts (integral floats are accepted).

        Returns:
            ``log P(X = x)``.

        Raises:
            TypeError: If a count is not integral.
            ValueError: If the number of counts differs from ``K``.

        Equation:
            [EQ-DIST-MULT-001].

        References:
            [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35].

        """
        checked = count_vector(counts, name="counts", length=self.dimension)
        if any(count < 0 for count in checked) or sum(checked) != self.n:
            return -math.inf
        terms: list[float] = [log_multinomial_coefficient(checked)]
        for count, probability in zip(checked, self.probabilities, strict=True):
            if count == 0:
                continue
            if probability == 0.0:
                return -math.inf
            terms.append(count * admath.log(probability))
        return admath.fsum(terms)

    def pmf(self, counts: Iterable[int]) -> float:
        """Evaluate the multinomial probability of a count vector.

        Args:
            counts: ``K`` integer counts (integral floats are accepted).

        Returns:
            ``P(X = x)``.

        Raises:
            TypeError: If a count is not integral.
            ValueError: If the number of counts differs from ``K``.

        Equation:
            [EQ-DIST-MULT-001].

        References:
            [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35].

        """
        return math.exp(self.log_pmf(counts))

    @property
    def mean(self) -> Vector:
        """Return ``E[X] = n p``.

        References:
            [EQ-DIST-MULT-002], [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35].

        """
        return tuple(self.n * probability for probability in self.probabilities)

    @property
    def covariance(self) -> Matrix:
        """Return ``Cov[X] = n (diag(p) - p p')``.

        References:
            [EQ-DIST-MULT-002], [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35].

        """
        p = self.probabilities
        return tuple(
            tuple(
                self.n * ((p[i] if i == j else 0.0) - p[i] * p[j])
                for j in range(self.dimension)
            )
            for i in range(self.dimension)
        )

    def marginal(self, index: int) -> Binomial:
        """Return the marginal law ``X_i ~ Binomial(n, p_i)``.

        Args:
            index: Zero-based category index.

        Returns:
            The binomial marginal of one category count.

        Raises:
            TypeError: If ``index`` is not an integer.
            IndexError: If ``index`` is outside the category range.

        Equation:
            [EQ-DIST-MULT-003].

        References:
            [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35].

        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < self.dimension:
            raise IndexError("index outside the category range")
        return Binomial(self.n, self.probabilities[index])

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[tuple[int, ...], ...]:
        """Draw reproducible count vectors by sequential conditional binomials.

        The cost is ``O(n)`` uniform variates per vector, because each
        conditional binomial is drawn as a Bernoulli sum.

        Args:
            size: Number of count vectors, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of count vectors, each summing to ``n``.

        Raises:
            TypeError: If ``size`` or ``seed`` has an invalid type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-MULT-004].

        References:
            [REF-DEVROYE-1986, Chapter XI].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(
            _multinomial_draw(self.n, self.probabilities, generator)
            for _ in range(checked_size)
        )


@dataclass(frozen=True, slots=True)
class DirichletMultinomial:
    """The compound (Polya) law ``DirMult(n, alpha)`` of category counts.

    ``X | p ~ Mult(n, p)`` with ``p ~ Dirichlet(alpha)``.

    Args:
        n: Non-negative number of trials.
        alpha: At least two finite, strictly positive concentrations.

    References:
        [REF-MOSIMANN-1962], [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35].

    """

    n: int
    alpha: Vector

    def __post_init__(self) -> None:
        """Validate the trial count and the concentrations.

        Raises:
            TypeError: If ``n`` is not an integer or ``alpha`` is not real.
            ValueError: If ``n < 0`` or ``alpha`` has fewer than two entries or
                a non-positive entry.

        References:
            [EQ-DIST-DIRMULT-001], [REF-MOSIMANN-1962].

        """
        object.__setattr__(self, "n", require_non_negative_integer(self.n, name="n"))
        object.__setattr__(
            self, "alpha", positive_vector(self.alpha, name="alpha", minimum_length=2)
        )

    @property
    def dimension(self) -> int:
        """Return the number of categories ``K``.

        References:
            [EQ-DIST-DIRMULT-001], [REF-MOSIMANN-1962].

        """
        return len(self.alpha)

    @property
    def concentration(self) -> float:
        """Return ``A = sum_i alpha_i``.

        References:
            [EQ-DIST-DIRMULT-001], [REF-MOSIMANN-1962].

        """
        return math.fsum(self.alpha)

    def log_pmf(self, counts: Iterable[int]) -> float:
        """Evaluate the Dirichlet-multinomial log-mass of a count vector.

        ``log P(x) = log(n! / prod x_i!) + log Gamma(A) - log Gamma(n + A)
        + sum_i [log Gamma(x_i + alpha_i) - log Gamma(alpha_i)]``.

        Args:
            counts: ``K`` integer counts (integral floats are accepted).

        Returns:
            ``log P(X = x)``; ``-inf`` for a negative count or a total other
            than ``n``.

        Raises:
            TypeError: If a count is not integral.
            ValueError: If the number of counts differs from ``K``.

        Each gamma ratio is evaluated by ``distributions._special.log_gamma_ratio``,
        which uses the Stirling-series form for large arguments, so the mass
        keeps its relative accuracy as the concentrations grow.

        Equation:
            [EQ-DIST-DIRMULT-001], [EQ-NUM-STIRLING-001].

        References:
            [REF-MOSIMANN-1962], [REF-NIST-DLMF-GAMMA, Sections 5.2 and 5.11].

        """
        checked = count_vector(counts, name="counts", length=self.dimension)
        if any(count < 0 for count in checked) or sum(checked) != self.n:
            return -math.inf
        terms = [log_multinomial_coefficient(checked)]
        if self.n:
            terms.append(-log_gamma_ratio(self.concentration, self.n))
        for count, shape in zip(checked, self.alpha, strict=True):
            if count:
                terms.append(log_gamma_ratio(shape, count))
        return math.fsum(terms)

    def pmf(self, counts: Iterable[int]) -> float:
        """Evaluate the Dirichlet-multinomial probability of a count vector.

        Args:
            counts: ``K`` integer counts (integral floats are accepted).

        Returns:
            ``P(X = x)``.

        Raises:
            TypeError: If a count is not integral.
            ValueError: If the number of counts differs from ``K``.

        Equation:
            [EQ-DIST-DIRMULT-001].

        References:
            [REF-MOSIMANN-1962].

        """
        return math.exp(self.log_pmf(counts))

    @property
    def overdispersion(self) -> float:
        """Return the covariance inflation ``(n + A) / (1 + A)`` over the multinomial.

        References:
            [EQ-DIST-DIRMULT-002], [REF-MOSIMANN-1962].

        """
        total = self.concentration
        return (self.n + total) / (1.0 + total)

    @property
    def mean(self) -> Vector:
        """Return ``E[X] = n alpha / A``.

        References:
            [EQ-DIST-DIRMULT-002], [REF-MOSIMANN-1962].

        """
        total = self.concentration
        return tuple(self.n * shape / total for shape in self.alpha)

    @property
    def covariance(self) -> Matrix:
        """Return ``n (n + A) / (1 + A) (diag(q) - q q')`` with ``q = alpha / A``.

        References:
            [EQ-DIST-DIRMULT-002], [REF-MOSIMANN-1962].

        """
        total = self.concentration
        q = tuple(shape / total for shape in self.alpha)
        factor = self.n * self.overdispersion
        return tuple(
            tuple(
                factor * ((q[i] if i == j else 0.0) - q[i] * q[j])
                for j in range(self.dimension)
            )
            for i in range(self.dimension)
        )

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[tuple[int, ...], ...]:
        """Draw reproducible count vectors by composition.

        Each draw takes ``p ~ Dirichlet(alpha)`` (normalized log-space gamma
        variates) and then ``X | p ~ Mult(n, p)`` by sequential conditional
        binomials.

        Args:
            size: Number of count vectors, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of count vectors, each summing to ``n``.

        Raises:
            TypeError: If ``size`` or ``seed`` has an invalid type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-DIRMULT-003], [EQ-DIST-DIR-006], [EQ-DIST-MULT-004].

        References:
            [REF-MOSIMANN-1962], [REF-DEVROYE-1986, Chapter XI].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        observations: list[tuple[int, ...]] = []
        for _ in range(checked_size):
            logs = tuple(log_gamma_variate(shape, generator) for shape in self.alpha)
            peak = max(logs)
            weights = tuple(math.exp(value - peak) for value in logs)
            total = math.fsum(weights)
            probabilities = tuple(weight / total for weight in weights)
            observations.append(_multinomial_draw(self.n, probabilities, generator))
        return tuple(observations)
