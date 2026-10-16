"""Triangular distribution on a finite interval with an interior or edge mode."""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._base import ContinuousDistribution
from .constraints import FINITE_REAL
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class Triangular(ContinuousDistribution):
    """A triangular law with lower end ``a``, mode ``c``, and upper end ``b``.

    The density rises linearly from ``a`` to ``2/(b-a)`` at ``c`` and falls
    linearly to ``b``. ``c = a`` or ``c = b`` gives a right or left triangle.

    Args:
        lower: Finite lower endpoint ``a``.
        mode: Finite mode ``c`` with ``a <= c <= b``.
        upper: Finite upper endpoint ``b > a``.

    References:
        [REF-KOTZ-VANDORP-2004, Chapter 1].

    """

    lower: float
    mode: float
    upper: float

    def __post_init__(self) -> None:
        """Validate the ordering ``a <= c <= b`` with ``a < b``.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or the ordering fails.

        References:
            [EQ-DIST-TRI-001], [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        lower = FINITE_REAL.validate(self.lower, name="lower")
        mode = FINITE_REAL.validate(self.mode, name="mode")
        upper = FINITE_REAL.validate(self.upper, name="upper")
        if not lower < upper:
            raise ValueError("lower must be strictly less than upper")
        if not lower <= mode <= upper:
            raise ValueError("mode must lie in [lower, upper]")
        if not math.isfinite(upper - lower):
            raise ValueError("upper - lower must be finite")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "upper", upper)

    @property
    def support(self) -> RealSupport:
        """Return the closed interval ``[a, b]``.

        References:
            [EQ-DIST-TRI-001], [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        return RealSupport(
            self.lower, self.upper, lower_closed=True, upper_closed=True
        )

    @property
    def _width(self) -> float:
        """Return ``b - a``.

        References:
            [EQ-DIST-TRI-001], [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        return self.upper - self.lower

    def pdf(self, x: float) -> float:
        """Evaluate the piecewise-linear triangular density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density, zero outside ``[a, b]`` and ``2/(b-a)`` at the mode.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-TRI-001].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        point = require_finite_real(x, name="x")
        a, c, b = self.lower, self.mode, self.upper
        if point < a or point > b:
            return 0.0
        if point == c:
            return 2.0 / self._width
        if point < c:
            return 2.0 * (point - a) / (self._width * (c - a))
        return 2.0 * (b - point) / (self._width * (b - c))

    def log_pdf(self, x: float) -> float:
        """Evaluate the triangular log-density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density, ``-inf`` where the density is zero.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-TRI-001].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        density = self.pdf(x)
        return math.log(density) if density > 0.0 else -math.inf

    def cdf(self, x: float) -> float:
        """Evaluate the triangular CDF from the nearer endpoint.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``(x-a)^2/((b-a)(c-a))`` left of the mode and
            ``1 - (b-x)^2/((b-a)(b-c))`` right of it.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-TRI-002].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        point = require_finite_real(x, name="x")
        a, c, b = self.lower, self.mode, self.upper
        if point <= a:
            return 0.0
        if point >= b:
            return 1.0
        if point <= c:
            return (point - a) ** 2 / (self._width * (c - a))
        return 1.0 - (b - point) ** 2 / (self._width * (b - c))

    def sf(self, x: float) -> float:
        """Evaluate the direct triangular survival function.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``(b-x)^2/((b-a)(b-c))`` right of the mode, without subtraction.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-TRI-002].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        point = require_finite_real(x, name="x")
        a, c, b = self.lower, self.mode, self.upper
        if point <= a:
            return 1.0
        if point >= b:
            return 0.0
        if point >= c:
            return (b - point) ** 2 / (self._width * (b - c))
        return 1.0 - (point - a) ** 2 / (self._width * (c - a))

    def _from_tails(self, lower: float, upper: float) -> float:
        """Invert using whichever of ``p`` and ``q = 1 - p`` lies in its branch.

        Args:
            lower: Lower-tail probability ``p`` in ``[0, 1]``.
            upper: Complementary upper-tail probability ``q``.

        Returns:
            ``a + sqrt(p (b-a)(c-a))`` when ``p <= (c-a)/(b-a)``, otherwise
            ``b - sqrt(q (b-a)(b-c))``.

        Equation:
            [EQ-DIST-TRI-003].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        a, c, b = self.lower, self.mode, self.upper
        split = (c - a) / self._width
        if lower <= split:
            return min(a + math.sqrt(lower * self._width * (c - a)), b)
        return max(b - math.sqrt(upper * self._width * (b - c)), a)

    def quantile(self, probability: float) -> float:
        """Return the closed-form triangular quantile.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Quantile in ``[a, b]``.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-TRI-003].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        p = require_probability(probability, name="probability")
        return self._from_tails(p, 1.0 - p)

    def isf(self, probability: float) -> float:
        """Return the inverse survival function without forming ``1 - q``.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Quantile in ``[a, b]``.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-ISF-001], [EQ-DIST-TRI-003].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        q = require_probability(probability, name="probability")
        return self._from_tails(1.0 - q, q)

    @property
    def mean(self) -> float:
        """Return ``E[X] = (a + b + c) / 3``.

        Equation:
            [EQ-DIST-TRI-004].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        return (self.lower + self.mode + self.upper) / 3.0

    @property
    def variance(self) -> float:
        """Return ``(a^2 + b^2 + c^2 - ab - ac - bc) / 18``.

        Evaluated as ``((b-a)^2 - (c-a)(b-c)) / 18`` to avoid cancellation for
        intervals far from the origin.

        Equation:
            [EQ-DIST-TRI-004].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        a, c, b = self.lower, self.mode, self.upper
        return (self._width**2 - (c - a) * (b - c)) / 18.0

    @property
    def skewness(self) -> float:
        """Return ``sqrt 2 (a+b-2c)(2a-b-c)(a-2b+c) / (5 (18 Var)^(3/2))``.

        Here ``a`` is the lower bound, ``b`` the upper bound and ``c`` the
        mode.

        Equation:
            [EQ-DIST-TRI-005].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        a, c, b = self.lower, self.mode, self.upper
        spread = 18.0 * self.variance
        product = (a + b - 2.0 * c) * (2.0 * a - b - c) * (a - 2.0 * b + c)
        return math.sqrt(2.0) * product / (5.0 * math.pow(spread, 1.5))

    @property
    def kurtosis(self) -> float:
        """Return the excess kurtosis ``-3/5``.

        Equation:
            [EQ-DIST-TRI-005].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        return -0.6

    @property
    def entropy(self) -> float:
        """Return ``1/2 + log((b - a)/2)``.

        Equation:
            [EQ-DIST-TRI-005].

        References:
            [REF-KOTZ-VANDORP-2004, Chapter 1].

        """
        return 0.5 + math.log(0.5 * self._width)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw reproducible triangular observations by inverse transform.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of observations in ``[a, b]``.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        Equation:
            [EQ-DIST-TRI-003].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        observations: list[float] = []
        for _ in range(checked_size):
            u = generator.random()
            observations.append(self._from_tails(u, 1.0 - u))
        return tuple(observations)
