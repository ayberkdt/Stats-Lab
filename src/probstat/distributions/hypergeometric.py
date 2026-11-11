"""Hypergeometric law of successes drawn without replacement.

The support uses exact integer arithmetic; masses are exact rationals for
small populations and use ``lgamma`` otherwise.

References:
    [REF-JKB-2005, Chapter 6].

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._base import DiscreteDistribution
from .support import IntegerSupport

_EXACT_POPULATION = 5000


def _log_comb(n: int, k: int) -> float:
    """Return ``log C(n, k)`` through ``lgamma``.

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.2]. No new equation.

    """
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


@dataclass(frozen=True, slots=True)
class Hypergeometric(DiscreteDistribution):
    """Successes in ``draws`` draws without replacement from a finite population.

    ``P(X = k) = C(K, k) C(N - K, n - k) / C(N, n)`` on
    ``max(0, n + K - N) <= k <= min(n, K)``. For ``N <= 5000`` the masses
    are exact rationals rounded once; larger populations use ``lgamma``.

    Args:
        population: Population size ``N >= 0``.
        successes: Success states ``0 <= K <= N``.
        draws: Draws ``0 <= n <= N``.

    References:
        [REF-JKB-2005, Chapter 6].

    """

    population: int
    successes: int
    draws: int

    def __post_init__(self) -> None:
        """Validate the integer parameters.

        Raises:
            TypeError: If a parameter is not an integer.
            ValueError: If ``K`` or ``n`` lies outside ``[0, N]``.

        References:
            [EQ-DIST-HYPERGEOM-001], [REF-JKB-2005, Chapter 6].

        """
        total = require_non_negative_integer(self.population, name="population")
        good = require_non_negative_integer(self.successes, name="successes")
        draws = require_non_negative_integer(self.draws, name="draws")
        if good > total or draws > total:
            raise ValueError("successes and draws must not exceed population")

    @property
    def support(self) -> IntegerSupport:
        """Return ``[max(0, n + K - N), min(n, K)]``.

        References:
            [EQ-DIST-HYPERGEOM-001]. No new equation.

        """
        low = max(0, self.draws + self.successes - self.population)
        return IntegerSupport(low, min(self.draws, self.successes))

    def log_pmf(self, k: int) -> float:
        """Return ``log C(K, k) + log C(N - K, n - k) - log C(N, n)``.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-HYPERGEOM-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        support = self.support
        if not support.contains(k):
            return -math.inf
        n_total, good, n = self.population, self.successes, self.draws
        if n_total <= _EXACT_POPULATION:
            numerator = math.comb(good, k) * math.comb(n_total - good, n - k)
            return math.log(numerator) - math.log(math.comb(n_total, n))
        return (
            _log_comb(good, k)
            + _log_comb(n_total - good, n - k)
            - _log_comb(n_total, n)
        )

    def pmf(self, k: int) -> float:
        """Return the hypergeometric mass.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-HYPERGEOM-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if self.population <= _EXACT_POPULATION and self.support.contains(k):
            numerator = math.comb(self.successes, k) * math.comb(
                self.population - self.successes, self.draws - k
            )
            return float(Fraction(numerator, math.comb(self.population, self.draws)))
        value = self.log_pmf(k)
        return math.exp(value) if value > -745.2 else 0.0

    def _tail(self, first: int, last: int) -> float:
        """Return ``sum_(k=first..last) p_k`` (empty sums are zero).

        References:
            [EQ-DIST-HYPERGEOM-001]. No new equation.

        """
        if last < first:
            return 0.0
        return math.fsum(self.pmf(k) for k in range(first, last + 1))

    def _mode(self) -> int:
        """Return ``floor((n + 1)(K + 1) / (N + 2))``, the mode.

        References:
            [REF-JKB-2005, Chapter 6].

        """
        return (self.draws + 1) * (self.successes + 1) // (self.population + 2)

    def cdf(self, x: float) -> float:
        """Return ``P(X <= x)`` by summing the smaller side of the mode.

        Equation:
            [EQ-DIST-HYPERGEOM-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        point = math.floor(require_finite_real(x, name="x"))
        support = self.support
        upper = support.upper if support.upper is not None else support.lower
        if point < support.lower:
            return 0.0
        if point >= upper:
            return 1.0
        if point < self._mode():
            return min(1.0, self._tail(support.lower, point))
        return max(0.0, 1.0 - self._tail(point + 1, upper))

    def sf(self, x: float) -> float:
        """Return ``P(X > x)`` by summing the smaller side of the mode.

        Equation:
            [EQ-DIST-HYPERGEOM-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        point = math.floor(require_finite_real(x, name="x"))
        support = self.support
        upper = support.upper if support.upper is not None else support.lower
        if point < support.lower:
            return 1.0
        if point >= upper:
            return 0.0
        if point >= self._mode():
            return min(1.0, self._tail(point + 1, upper))
        return max(0.0, 1.0 - self._tail(support.lower, point))

    def quantile(self, probability: float) -> float:
        """Return the smallest ``k`` with ``F(k) >= p`` by bisection.

        Equation:
            [EQ-DIST-HYPERGEOM-001].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        p = require_probability(probability, name="probability")
        support = self.support
        low = support.lower
        high = support.upper if support.upper is not None else low
        if self.cdf(low) >= p:
            return float(low)
        while high - low > 1:
            middle = (low + high) // 2
            if self.cdf(middle) >= p:
                high = middle
            else:
                low = middle
        return float(high)

    @property
    def mean(self) -> float:
        """Return ``n K / N`` (``0`` for an empty population).

        Equation:
            [EQ-DIST-HYPERGEOM-002].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        if self.population == 0:
            return 0.0
        return self.draws * self.successes / self.population

    @property
    def variance(self) -> float:
        """Return ``n K (N - K)(N - n) / (N^2 (N - 1))``.

        Equation:
            [EQ-DIST-HYPERGEOM-002].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        total = self.population
        if total <= 1:
            return 0.0
        good, n = self.successes, self.draws
        return n * good * (total - good) * (total - n) / (total * total * (total - 1))

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[int, ...]:
        """Draw variates by sequential sampling without replacement.

        Each of the ``n`` draws succeeds with probability (remaining
        successes)/(remaining population); the work is ``O(n)`` per variate.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of success counts.

        Raises:
            TypeError: If ``size`` or ``seed`` has the wrong type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-HYPERGEOM-001].

        References:
            [REF-DEVROYE-1986, Chapter X].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        results: list[int] = []
        for _ in range(count):
            remaining, good, hits = self.population, self.successes, 0
            for _ in range(self.draws):
                if generator.random() * remaining < good:
                    good -= 1
                    hits += 1
                remaining -= 1
            results.append(hits)
        return tuple(results)
