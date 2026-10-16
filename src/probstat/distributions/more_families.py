"""Beta-PERT, discrete uniform, and hypergeometric distributions.

The beta-PERT law is a rescaled :class:`~probstat.distributions.Beta` and
inherits its tail-accurate distribution functions. The two discrete laws use
exact integer arithmetic for their supports and ``lgamma`` for masses.
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
from probstat.numerics.special import digamma

from ._base import ContinuousDistribution, DiscreteDistribution
from ._special import log_beta_function
from .beta import Beta
from .constraints import FINITE_REAL, POSITIVE_FINITE
from .support import IntegerSupport, RealSupport


@dataclass(frozen=True, slots=True)
class PERT(ContinuousDistribution):
    """The beta-PERT law on ``[low, high]`` with most likely value ``mode``.

    ``X = low + (high - low) B`` with ``B ~ Beta(alpha, beta)``,
    ``alpha = 1 + lam (mode - low)/(high - low)`` and
    ``beta = 1 + lam (high - mode)/(high - low)``. The standard PERT choice
    ``lam = 4`` gives the mean ``(low + 4 mode + high)/6`` of the classical
    PERT three-point estimate; other ``lam > 0`` give the "modified PERT"
    family (larger ``lam`` concentrates the law around the mode).

    Args:
        low: Finite minimum.
        mode: Most likely value with ``low <= mode <= high``.
        high: Finite maximum with ``low < high``.
        lam: Positive shape weight ``lam`` (default 4).

    References:
        [REF-MALCOLM-1959], [REF-VOSE-2008, Chapter 10].

    """

    low: float
    mode: float
    high: float
    lam: float = 4.0

    def __post_init__(self) -> None:
        """Validate the three points and the shape weight.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If ``low < high`` and ``low <= mode <= high`` fail or
                ``lam <= 0``.

        References:
            [EQ-DIST-PERT-001], [REF-MALCOLM-1959].

        """
        low = FINITE_REAL.validate(self.low, name="low")
        mode = FINITE_REAL.validate(self.mode, name="mode")
        high = FINITE_REAL.validate(self.high, name="high")
        lam = POSITIVE_FINITE.validate(self.lam, name="lam")
        if not low < high:
            raise ValueError("low must be below high")
        if not low <= mode <= high:
            raise ValueError("mode must lie in [low, high]")
        for name, value in (("low", low), ("mode", mode), ("high", high)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "lam", lam)

    @property
    def width(self) -> float:
        """Return ``high - low``.

        References:
            [EQ-DIST-PERT-001]. No new equation.

        """
        return self.high - self.low

    @property
    def alpha(self) -> float:
        """Return ``1 + lam (mode - low)/(high - low)``.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        return 1.0 + self.lam * (self.mode - self.low) / self.width

    @property
    def beta(self) -> float:
        """Return ``1 + lam (high - mode)/(high - low)``.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        return 1.0 + self.lam * (self.high - self.mode) / self.width

    @property
    def standard(self) -> Beta:
        """Return the underlying ``Beta(alpha, beta)`` law on ``[0, 1]``.

        References:
            [EQ-DIST-PERT-001]. No new equation.

        """
        return Beta(self.alpha, self.beta)

    @property
    def support(self) -> RealSupport:
        """Return ``[low, high]``.

        References:
            [EQ-DIST-PERT-001]. No new equation.

        """
        return RealSupport(self.low, self.high, lower_closed=True, upper_closed=True)

    def _unit(self, x: float) -> float:
        """Map a validated point to ``(x - low)/(high - low)`` clipped to ``[0, 1]``.

        References:
            [EQ-DIST-PERT-001]. No new equation.

        """
        point = require_finite_real(x, name="x")
        return min(max((point - self.low) / self.width, 0.0), 1.0)

    def log_pdf(self, x: float) -> float:
        """Return ``log f_B(u) - log(high - low)`` (``-inf`` outside).

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        point = require_finite_real(x, name="x")
        if point < self.low or point > self.high:
            return -math.inf
        return self.standard.log_pdf(self._unit(point)) - math.log(self.width)

    def pdf(self, x: float) -> float:
        """Return the PERT density.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        value = self.log_pdf(x)
        return math.exp(value) if value > -745.2 else 0.0

    def cdf(self, x: float) -> float:
        """Return ``F_B((x - low)/(high - low))``.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        return self.standard.cdf(self._unit(x))

    def sf(self, x: float) -> float:
        """Return ``S_B((x - low)/(high - low))``.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        return self.standard.sf(self._unit(x))

    def quantile(self, probability: float) -> float:
        """Return ``low + (high - low) Q_B(p)``.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        return self.low + self.width * self.standard.quantile(probability)

    def isf(self, probability: float) -> float:
        """Return ``low + (high - low) isf_B(q)``.

        Equation:
            [EQ-DIST-PERT-001], [EQ-DIST-ISF-001].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        return self.low + self.width * self.standard.isf(probability)

    @property
    def mean(self) -> float:
        """Return ``(low + lam mode + high) / (lam + 2)``.

        Equation:
            [EQ-DIST-PERT-002].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        return (self.low + self.lam * self.mode + self.high) / (self.lam + 2.0)

    @property
    def variance(self) -> float:
        """Return ``(mean - low)(high - mean) / (lam + 3)``.

        Equation:
            [EQ-DIST-PERT-002].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        mean = self.mean
        return (mean - self.low) * (self.high - mean) / (self.lam + 3.0)

    @property
    def skewness(self) -> float:
        """Return the beta skewness of ``Beta(a, b)``.

        ``2 (b - a) sqrt(a + b + 1) / ((a + b + 2) sqrt(a b))``.

        Equation:
            [EQ-DIST-PERT-002].

        References:
            [REF-JKB-1995, Chapter 25].

        """
        a, b = self.alpha, self.beta
        return (
            2.0 * (b - a) * math.sqrt(a + b + 1.0) / ((a + b + 2.0) * math.sqrt(a * b))
        )

    @property
    def kurtosis(self) -> float:
        """Return the beta excess kurtosis.

        ``6 [(a - b)^2 (a + b + 1) - a b (a + b + 2)] / (a b (a + b + 2)
        (a + b + 3))``.

        Equation:
            [EQ-DIST-PERT-002].

        References:
            [REF-JKB-1995, Chapter 25].

        """
        a, b = self.alpha, self.beta
        total = a + b
        numerator = (a - b) ** 2 * (total + 1.0) - a * b * (total + 2.0)
        return 6.0 * numerator / (a * b * (total + 2.0) * (total + 3.0))

    @property
    def entropy(self) -> float:
        """Return the differential entropy of the rescaled beta law.

        ``log B(a, b) - (a-1) psi(a) - (b-1) psi(b) + (a+b-2) psi(a+b) +
        log(high - low)``.

        Equation:
            [EQ-DIST-PERT-002].

        References:
            [REF-COVER-THOMAS-2006, Chapter 8].

        """
        a, b = self.alpha, self.beta
        return (
            log_beta_function(a, b)
            - (a - 1.0) * digamma(a)
            - (b - 1.0) * digamma(b)
            + (a + b - 2.0) * digamma(a + b)
            + math.log(self.width)
        )

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw PERT variates by rescaling beta variates.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` or ``seed`` has the wrong type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008, Chapter 10].

        """
        draws = self.standard.sample(size, seed=resolve_stream(seed))
        return tuple(self.low + self.width * value for value in draws)


@dataclass(frozen=True, slots=True)
class DiscreteUniform(DiscreteDistribution):
    """The discrete uniform law on the integers ``low, low + 1, ..., high``.

    Args:
        low: Integer minimum.
        high: Integer maximum with ``high >= low``.

    References:
        [REF-JKB-2005, Chapter 6].

    """

    low: int
    high: int

    def __post_init__(self) -> None:
        """Validate the integer bounds.

        Raises:
            TypeError: If a bound is not an integer.
            ValueError: If ``high < low``.

        References:
            [EQ-DIST-DUNIF-001], [REF-JKB-2005, Chapter 6].

        """
        for name in ("low", "high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.high < self.low:
            raise ValueError("high must not be below low")

    @property
    def count(self) -> int:
        """Return the number of support points ``n = high - low + 1``.

        References:
            [EQ-DIST-DUNIF-001]. No new equation.

        """
        return self.high - self.low + 1

    @property
    def support(self) -> IntegerSupport:
        """Return the integer support.

        References:
            [EQ-DIST-DUNIF-001]. No new equation.

        """
        return IntegerSupport(self.low, self.high)

    def log_pmf(self, k: int) -> float:
        """Return ``-log n`` on the support and ``-inf`` elsewhere.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if self.low <= k <= self.high:
            return -math.log(self.count)
        return -math.inf

    def pmf(self, k: int) -> float:
        """Return ``1/n`` on the support and ``0`` elsewhere.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        return 0.0 if self.log_pmf(k) == -math.inf else 1.0 / self.count

    def cdf(self, x: float) -> float:
        """Return ``(floor(x) - low + 1)/n`` clipped to ``[0, 1]``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        point = math.floor(require_finite_real(x, name="x"))
        if point < self.low:
            return 0.0
        if point >= self.high:
            return 1.0
        return (point - self.low + 1) / self.count

    def sf(self, x: float) -> float:
        """Return ``(high - floor(x))/n`` clipped to ``[0, 1]``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        point = math.floor(require_finite_real(x, name="x"))
        if point < self.low:
            return 1.0
        if point >= self.high:
            return 0.0
        return (self.high - point) / self.count

    def quantile(self, probability: float) -> float:
        """Return the smallest ``k`` with ``F(k) >= p`` (integer-valued float).

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        p = require_probability(probability, name="probability")
        k = self.low + max(math.ceil(p * self.count) - 1, 0)
        while k > self.low and self.cdf(k - 1) >= p:
            k -= 1
        while self.cdf(k) < p:
            k += 1
        return float(k)

    @property
    def mean(self) -> float:
        """Return ``(low + high) / 2``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        return 0.5 * (self.low + self.high)

    @property
    def variance(self) -> float:
        """Return ``(n^2 - 1) / 12``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        return (self.count * self.count - 1) / 12.0

    @property
    def skewness(self) -> float:
        """Return ``0`` (``nan`` for a single point).

        References:
            [REF-JKB-2005, Chapter 6].

        """
        return 0.0 if self.count > 1 else math.nan

    @property
    def kurtosis(self) -> float:
        """Return the excess kurtosis ``-6 (n^2 + 1) / (5 (n^2 - 1))``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        n2 = self.count * self.count
        return -6.0 * (n2 + 1) / (5.0 * (n2 - 1)) if n2 > 1 else math.nan

    @property
    def entropy(self) -> float:
        """Return ``log n``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-COVER-THOMAS-2006, Chapter 2].

        """
        return math.log(self.count)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[int, ...]:
        """Draw uniform integers with ``randrange``.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of integers in ``[low, high]``.

        Raises:
            TypeError: If ``size`` or ``seed`` has the wrong type.
            ValueError: If ``size`` or an integer seed is negative.

        References:
            [REF-MATSUMOTO-NISHIMURA-1998]. No equation is evaluated.

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(generator.randrange(self.low, self.high + 1) for _ in range(count))


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
