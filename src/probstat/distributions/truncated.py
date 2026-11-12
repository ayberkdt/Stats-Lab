"""Truncation of a continuous law to an interval (ADR-0059).

The base law may be any object with the structural marginal interface; see
:mod:`probstat.distributions._wrapped` for the optional methods it may add.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_non_negative_integer,
    require_probability,
)

from ._base import ContinuousDistribution, _quantile_integral
from ._continuous import open_unit
from ._wrapped import (
    _check_law,
    _finite_point,
    _isf,
    _log_pdf,
    _sf,
)


@dataclass(frozen=True, slots=True)
class Truncated(ContinuousDistribution):
    """A continuous law conditioned on ``lower <= X <= upper`` (one side may be open).

    With ``Z = F(upper) - F(lower)`` (formed from survival probabilities
    when ``F(lower) > 1/2``, so upper-tail truncation keeps its accuracy):
    ``f_T = f / Z`` on the interval, ``F_T(x) = (F(x) - F(lower)) / Z`` and
    ``Q_T(p) = Q(F(lower) + p Z)`` (or ``isf(S(lower) - p Z)`` in the upper
    half). The mean and variance are computed by tanh-sinh quadrature of the
    truncated quantile function and are ``nan`` if it does not converge.

    Args:
        base: Continuous base law with the marginal interface.
        lower: Lower bound (``-inf`` for none).
        upper: Upper bound (``inf`` for none), with ``lower < upper``.

    References:
        [REF-JKB-1994, Chapter 13], [REF-CASELLA-BERGER-2002, Section 1.3].

    """

    base: Any
    lower: float = -math.inf
    upper: float = math.inf
    _tails: tuple[float, float, float, float] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the bounds and the retained probability mass.

        Raises:
            TypeError: If the base lacks the marginal interface.
            ValueError: If ``lower >= upper``, a bound is NaN, or the interval
                carries zero probability.

        References:
            [EQ-DIST-TRUNC-001]. No new equation.

        """
        _check_law(self.base, "base")
        low, high = float(self.lower), float(self.upper)
        if math.isnan(low) or math.isnan(high) or not low < high:
            raise ValueError("bounds must satisfy lower < upper")
        object.__setattr__(self, "lower", low)
        object.__setattr__(self, "upper", high)
        base = self.base
        tails = (
            0.0 if low == -math.inf else float(base.cdf(low)),
            1.0 if low == -math.inf else _sf(base, low),
            1.0 if high == math.inf else float(base.cdf(high)),
            0.0 if high == math.inf else _sf(base, high),
        )
        object.__setattr__(self, "_tails", tails)
        if not self._mass > 0.0:
            raise ValueError("the truncation interval has zero probability")

    @property
    def _lower_cdf(self) -> float:
        """Return ``F(lower)``.

        References:
            [EQ-DIST-TRUNC-001]. No new equation.

        """
        return self._tails[0]

    @property
    def _lower_sf(self) -> float:
        """Return ``S(lower)``.

        References:
            [EQ-DIST-TRUNC-001]. No new equation.

        """
        return self._tails[1]

    @property
    def _upper_side(self) -> bool:
        """Return whether the interval lies in the upper half (``F(lower) > 1/2``).

        References:
            [EQ-DIST-TRUNC-001]. No new equation.

        """
        return self._lower_cdf > 0.5

    @property
    def _mass(self) -> float:
        """Return ``Z = P(lower <= X <= upper)``.

        Equation:
            [EQ-DIST-TRUNC-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        if self._upper_side:
            return self._tails[1] - self._tails[3]
        return self._tails[2] - self._tails[0]

    def _inside(self, x: float) -> float:
        """Validate ``x`` and return it.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        return _finite_point(x)

    def log_pdf(self, x: float) -> float:
        """Return ``log f(x) - log Z`` inside the bounds, ``-inf`` outside.

        Equation:
            [EQ-DIST-TRUNC-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        point = self._inside(x)
        if point < self.lower or point > self.upper:
            return -math.inf
        return _log_pdf(self.base, point) - math.log(self._mass)

    def pdf(self, x: float) -> float:
        """Return ``f(x) / Z`` inside the bounds, ``0`` outside.

        Equation:
            [EQ-DIST-TRUNC-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        point = self._inside(x)
        if point < self.lower or point > self.upper:
            return 0.0
        return float(self.base.pdf(point)) / self._mass

    def cdf(self, x: float) -> float:
        """Return ``(F(x) - F(lower)) / Z`` clipped to ``[0, 1]``.

        Equation:
            [EQ-DIST-TRUNC-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        point = self._inside(x)
        if point <= self.lower:
            return 0.0
        if point >= self.upper:
            return 1.0
        if self._upper_side:
            value = (self._lower_sf - _sf(self.base, point)) / self._mass
        else:
            value = (float(self.base.cdf(point)) - self._lower_cdf) / self._mass
        return min(max(value, 0.0), 1.0)

    def sf(self, x: float) -> float:
        """Return ``(F(upper) - F(x)) / Z`` clipped to ``[0, 1]``.

        Equation:
            [EQ-DIST-TRUNC-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        point = self._inside(x)
        if point <= self.lower:
            return 1.0
        if point >= self.upper:
            return 0.0
        if self._upper_side:
            value = (_sf(self.base, point) - self._tails[3]) / self._mass
        else:
            value = (self._tails[2] - float(self.base.cdf(point))) / self._mass
        return min(max(value, 0.0), 1.0)

    def quantile(self, probability: float) -> float:
        """Return ``Q(F(lower) + p Z)`` (or ``isf(S(lower) - p Z)``), clamped.

        Equation:
            [EQ-DIST-TRUNC-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        p = require_probability(probability, name="probability")
        if p == 0.0:
            return (
                self.lower if self.lower > -math.inf else float(self.base.quantile(0.0))
            )
        if p == 1.0:
            return (
                self.upper if self.upper < math.inf else float(self.base.quantile(1.0))
            )
        if self._upper_side:
            target = self._lower_sf - p * self._mass
            value = _isf(self.base, max(target, 0.0))
        else:
            target = self._lower_cdf + p * self._mass
            value = float(self.base.quantile(min(target, 1.0)))
        return min(max(value, self.lower), self.upper)

    def isf(self, probability: float) -> float:
        """Return the truncated inverse survival function ``Q_T(1 - q)``.

        Equation:
            [EQ-DIST-TRUNC-001], [EQ-DIST-ISF-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        q = require_probability(probability, name="probability")
        if q == 1.0:
            # ``isf(S(lower))`` of the base need not round back to ``lower``.
            return self.quantile(0.0)
        if self._upper_side:
            if q == 0.0:
                return self.quantile(1.0)
            value = _isf(self.base, self._tails[3] + q * self._mass)
            return min(max(value, self.lower), self.upper)
        return self.quantile(1.0 - q)

    @property
    def mean(self) -> float:
        """Return ``E[X | lower <= X <= upper]`` by quantile-space quadrature.

        Equation:
            [EQ-DIST-BASE-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        return _quantile_integral(self, lambda x: x, 1.0)

    @property
    def variance(self) -> float:
        """Return the truncated variance by quantile-space quadrature.

        Equation:
            [EQ-DIST-BASE-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        mean = self.mean
        if not math.isfinite(mean):
            return math.nan
        return _quantile_integral(self, lambda x: (x - mean) ** 2, 1.0)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw truncated variates by inversion (no rejection).

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations inside the bounds.

        Equation:
            [EQ-DIST-TRUNC-001].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(self.quantile(open_unit(generator)) for _ in range(count))
