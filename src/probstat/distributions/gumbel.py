"""Gumbel (extreme-value type I) laws for maxima and minima."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._continuous import open_unit
from .constraints import FINITE_REAL, POSITIVE_FINITE
from .support import RealSupport

EULER_GAMMA = 0.57721566490153286060651209008240243
_EXP_LIMIT = 709.0


def _exp_capped(value: float) -> float:
    """Return ``exp(value)`` with overflow mapped to infinity.

    Args:
        value: Real exponent.

    Returns:
        ``exp(value)`` or ``inf`` beyond the float range.

    References:
        [REF-PYTHON-MATH]. No probability equation is evaluated.

    """
    return math.inf if value > _EXP_LIMIT else math.exp(value)


@dataclass(frozen=True, slots=True)
class Gumbel:
    """A Gumbel law for maxima (``kind="max"``) or minima (``kind="min"``).

    Maximum: ``F(x) = exp(-exp(-z))``. Minimum: ``F(x) = 1 - exp(-exp(z))``,
    with ``z = (x - mu)/beta``. The minimum law is the reflection ``-Y`` of a
    maximum law ``Y`` with location ``-mu``.

    Args:
        mu: Finite location (mode).
        beta: Finite positive scale.
        kind: ``"max"`` (right-skewed, largest-extreme) or ``"min"``
            (left-skewed, smallest-extreme).

    References:
        [REF-JKB-1995, Chapter 22], [REF-NIST-DISTRIBUTIONS, Gumbel
        Distribution], [REF-COLES-2001, Chapter 3].

    """

    mu: float = 0.0
    beta: float = 1.0
    kind: Literal["max", "min"] = "max"

    def __post_init__(self) -> None:
        """Validate location, scale, and extreme kind.

        Raises:
            TypeError: If a numeric parameter is not real.
            ValueError: If a parameter is non-finite, ``beta <= 0``, or
                ``kind`` is not ``"max"`` or ``"min"``.

        References:
            [EQ-DIST-GUM-001], [REF-JKB-1995, Chapter 22].

        """
        object.__setattr__(self, "mu", FINITE_REAL.validate(self.mu, name="mu"))
        object.__setattr__(
            self, "beta", POSITIVE_FINITE.validate(self.beta, name="beta")
        )
        if self.kind not in ("max", "min"):
            raise ValueError("kind must be 'max' or 'min'")

    @property
    def support(self) -> RealSupport:
        """Return the entire real line.

        References:
            [EQ-DIST-GUM-001], [REF-JKB-1995, Chapter 22].

        """
        return RealSupport()

    def _reflected(self, x: float) -> float:
        """Return the maximum-form standardized point ``w``.

        ``w = z`` for maxima and ``w = -z`` for minima, so every evaluation
        reduces to ``G(w) = exp(-exp(-w))`` and its complement.

        Args:
            x: Finite real evaluation point.

        Returns:
            Reflected standardized point.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        References:
            [EQ-DIST-GUM-001], [REF-JKB-1995, Chapter 22].

        """
        z = (require_finite_real(x, name="x") - self.mu) / self.beta
        return z if self.kind == "max" else -z

    def log_pdf(self, x: float) -> float:
        """Evaluate ``-log beta - w - exp(-w)`` in the reflected coordinate.

        Args:
            x: Finite real evaluation point.

        Returns:
            Natural log density, ``-inf`` where it underflows completely.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GUM-001].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        w = self._reflected(x)
        return -math.log(self.beta) - w - _exp_capped(-w)

    def pdf(self, x: float) -> float:
        """Evaluate the Gumbel density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density value.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GUM-001].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        log_density = self.log_pdf(x)
        return math.exp(log_density) if log_density > -745.2 else 0.0

    def _maximum_lower(self, w: float) -> float:
        """Return ``exp(-exp(-w))``.

        Args:
            w: Reflected standardized point.

        Returns:
            Lower tail of the standard maximum law.

        Equation:
            [EQ-DIST-GUM-001].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        return math.exp(-_exp_capped(-w))

    def _maximum_upper(self, w: float) -> float:
        """Return ``1 - exp(-exp(-w))`` through ``expm1``.

        Args:
            w: Reflected standardized point.

        Returns:
            Upper tail of the standard maximum law.

        Equation:
            [EQ-DIST-GUM-001].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        return -math.expm1(-_exp_capped(-w))

    def cdf(self, x: float) -> float:
        """Evaluate the Gumbel CDF for the configured extreme kind.

        Args:
            x: Finite real evaluation point.

        Returns:
            Lower-tail probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GUM-001].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        w = self._reflected(x)
        if self.kind == "max":
            return self._maximum_lower(w)
        return self._maximum_upper(w)

    def sf(self, x: float) -> float:
        """Evaluate the direct Gumbel survival function.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without subtracting the CDF from one.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GUM-001].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        w = self._reflected(x)
        if self.kind == "max":
            return self._maximum_upper(w)
        return self._maximum_lower(w)

    def _from_maximum_tails(self, lower: float, upper: float) -> float:
        """Return ``x`` whose maximum-form tails are ``lower`` and ``upper``.

        Uses ``w = -log(-log G)`` with ``-log G`` computed as ``-log(lower)``
        or ``-log1p(-upper)``, whichever argument is smaller.

        Args:
            lower: Maximum-form lower-tail probability in ``(0, 1)``.
            upper: Complementary maximum-form upper-tail probability.

        Returns:
            Physical quantile.

        Equation:
            [EQ-DIST-GUM-002].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        exponent = -math.log(lower) if lower <= upper else -math.log1p(-upper)
        w = -math.log(exponent)
        return self.mu + self.beta * (w if self.kind == "max" else -w)

    def quantile(self, probability: float) -> float:
        """Return the closed-form Gumbel quantile.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Real quantile, with infinite endpoints at zero and one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-GUM-002].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        p = require_probability(probability, name="probability")
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        if self.kind == "max":
            return self._from_maximum_tails(p, 1.0 - p)
        return self._from_maximum_tails(1.0 - p, p)

    def isf(self, probability: float) -> float:
        """Return the inverse survival function without forming ``1 - q``.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Real quantile, with infinite endpoints at zero and one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-ISF-001], [EQ-DIST-GUM-002].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf
        if q == 1.0:
            return -math.inf
        if self.kind == "max":
            return self._from_maximum_tails(1.0 - q, q)
        return self._from_maximum_tails(q, 1.0 - q)

    @property
    def mean(self) -> float:
        """Return ``mu + beta gamma_E`` (max) or ``mu - beta gamma_E`` (min).

        Equation:
            [EQ-DIST-GUM-003].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        shift = self.beta * EULER_GAMMA
        return self.mu + shift if self.kind == "max" else self.mu - shift

    @property
    def variance(self) -> float:
        """Return ``Var(X) = pi^2 beta^2 / 6``.

        Equation:
            [EQ-DIST-GUM-003].

        References:
            [REF-JKB-1995, Chapter 22].

        """
        return math.pi * math.pi * self.beta * self.beta / 6.0

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw reproducible Gumbel observations by inverse transform.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local random-number generator.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        Equation:
            [EQ-DIST-GUM-002].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        observations: list[float] = []
        for _ in range(checked_size):
            u = open_unit(generator)
            observations.append(self._from_maximum_tails(u, 1.0 - u))
        return tuple(observations)
