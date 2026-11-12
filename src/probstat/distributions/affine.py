"""Affine transformation ``Y = loc + scale X`` of a continuous law (ADR-0059).

The base law may be any object with the structural marginal interface; see
:mod:`probstat.distributions._wrapped` for the optional methods it may add.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from probstat.foundations.random_streams import RandomStream
from probstat.foundations.validation import (
    require_finite_real,
    require_probability,
)

from ._base import ContinuousDistribution
from ._wrapped import (
    _base_sample,
    _check_law,
    _default_property,
    _finite_point,
    _isf,
    _log_pdf,
    _sf,
)


@dataclass(frozen=True, slots=True)
class Affine(ContinuousDistribution):
    """The law of ``X = loc + scale Y`` for a continuous base law of ``Y``.

    A negative ``scale`` reflects the base law: ``F_X(x) = S_Y((x - loc)/
    scale)``. Moments transform as ``E[X] = loc + scale E[Y]``,
    ``Var X = scale^2 Var Y``, skewness by ``sign(scale)`` and excess
    kurtosis unchanged; ``H(X) = H(Y) + log|scale|``.

    Args:
        base: Base law with the marginal interface.
        loc: Finite shift.
        scale: Finite non-zero scale.

    References:
        [REF-CASELLA-BERGER-2002, Section 2.1].

    """

    base: Any
    loc: float = 0.0
    scale: float = 1.0
    _ends: tuple[float, float] = field(
        init=False, repr=False, compare=False, default=(-math.inf, math.inf)
    )

    def __post_init__(self) -> None:
        """Validate the base law, shift, and scale.

        Raises:
            TypeError: If the base lacks the marginal interface.
            ValueError: If ``scale`` is zero or a parameter is non-finite.

        References:
            [EQ-DIST-AFFINE-001]. No new equation.

        """
        _check_law(self.base, "base")
        object.__setattr__(self, "loc", require_finite_real(self.loc, name="loc"))
        scale = require_finite_real(self.scale, name="scale")
        if scale == 0.0:
            raise ValueError("scale must be non-zero")
        object.__setattr__(self, "scale", scale)
        # Support ends in x. ``(end - loc) / scale`` need not round back to
        # the base end, so cdf and sf compare with these ends directly.
        try:
            ends = (float(self.quantile(0.0)), float(self.quantile(1.0)))
        except (ArithmeticError, ValueError):
            ends = (-math.inf, math.inf)
        object.__setattr__(self, "_ends", ends)

    def _y(self, x: float) -> float:
        """Return ``(x - loc) / scale``.

        References:
            [EQ-DIST-AFFINE-001]. No new equation.

        """
        return (_finite_point(x) - self.loc) / self.scale

    def log_pdf(self, x: float) -> float:
        """Return ``log f_Y((x - loc)/scale) - log|scale|``.

        Equation:
            [EQ-DIST-AFFINE-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 2.1].

        """
        return _log_pdf(self.base, self._y(x)) - math.log(abs(self.scale))

    def pdf(self, x: float) -> float:
        """Return ``f_Y((x - loc)/scale) / |scale|``.

        Equation:
            [EQ-DIST-AFFINE-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 2.1].

        """
        return float(self.base.pdf(self._y(x))) / abs(self.scale)

    def cdf(self, x: float) -> float:
        """Return ``F_Y(y)`` (``S_Y(y)`` for a negative scale).

        Equation:
            [EQ-DIST-AFFINE-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 2.1].

        """
        point = _finite_point(x)
        if point <= self._ends[0]:
            return 0.0
        if point >= self._ends[1]:
            return 1.0
        y = (point - self.loc) / self.scale
        return float(self.base.cdf(y)) if self.scale > 0.0 else _sf(self.base, y)

    def sf(self, x: float) -> float:
        """Return ``S_Y(y)`` (``F_Y(y)`` for a negative scale).

        Equation:
            [EQ-DIST-AFFINE-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 2.1].

        """
        point = _finite_point(x)
        if point <= self._ends[0]:
            return 1.0
        if point >= self._ends[1]:
            return 0.0
        y = (point - self.loc) / self.scale
        return _sf(self.base, y) if self.scale > 0.0 else float(self.base.cdf(y))

    def quantile(self, probability: float) -> float:
        """Return ``loc + scale Q_Y(p)`` (``loc + scale isf_Y(p)`` if ``scale < 0``).

        Equation:
            [EQ-DIST-AFFINE-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 2.1].

        """
        p = require_probability(probability, name="probability")
        inner = float(self.base.quantile(p)) if self.scale > 0.0 else _isf(self.base, p)
        return self.loc + self.scale * inner

    def isf(self, probability: float) -> float:
        """Return ``loc + scale isf_Y(q)`` (``loc + scale Q_Y(q)`` if ``scale < 0``).

        Equation:
            [EQ-DIST-AFFINE-001], [EQ-DIST-ISF-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 2.1].

        """
        q = require_probability(probability, name="probability")
        inner = _isf(self.base, q) if self.scale > 0.0 else float(self.base.quantile(q))
        return self.loc + self.scale * inner

    @property
    def mean(self) -> float:
        """Return ``loc + scale E[Y]``.

        Equation:
            [EQ-DIST-AFFINE-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 2.2].

        """
        return self.loc + self.scale * float(self.base.mean)

    @property
    def variance(self) -> float:
        """Return ``scale^2 Var Y``.

        Equation:
            [EQ-DIST-AFFINE-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 2.3].

        """
        return self.scale * self.scale * float(self.base.variance)

    @property
    def skewness(self) -> float:
        """Return ``sign(scale)`` times the base skewness.

        References:
            [REF-JKB-1994, Chapter 1].

        """
        base = getattr(self.base, "skewness", None)
        if base is None:
            return _default_property(self, "skewness")
        return math.copysign(1.0, self.scale) * float(base)

    @property
    def kurtosis(self) -> float:
        """Return the base excess kurtosis (location-scale invariant).

        References:
            [REF-JKB-1994, Chapter 1].

        """
        base = getattr(self.base, "kurtosis", None)
        if base is None:
            return _default_property(self, "kurtosis")
        return float(base)

    @property
    def entropy(self) -> float:
        """Return ``H(Y) + log|scale|``.

        Equation:
            [EQ-DIST-AFFINE-001].

        References:
            [REF-COVER-THOMAS-2006, Chapter 8].

        """
        base = getattr(self.base, "entropy", None)
        if base is None:
            return _default_property(self, "entropy")
        return float(base) + math.log(abs(self.scale))

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw ``loc + scale Y`` with ``Y`` from the base sampler (or inversion).

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Equation:
            [EQ-DIST-AFFINE-001].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        return tuple(
            self.loc + self.scale * y for y in _base_sample(self.base, size, seed)
        )
