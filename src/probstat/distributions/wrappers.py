"""Generic transformations of continuous laws: Affine, Truncated, and Mixture.

Each wrapper accepts any object with the structural marginal interface
``pdf``, ``cdf``, ``quantile``, ``mean`` and ``variance``
(``probstat.modeling.ContinuousMarginal``) and satisfies that interface
itself, so wrapped laws can be model inputs. Optional ``sf``, ``isf``,
``log_pdf`` and ``sample`` methods of the base law are used when present
(for tail accuracy) and replaced by ``1 - cdf``, ``quantile(1 - q)`` and
``log pdf`` otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._base import ContinuousDistribution, _quantile_integral
from ._continuous import brent_increasing, open_unit, safe_log

_REQUIRED = ("pdf", "cdf", "quantile", "mean", "variance")


def _check_law(law: object, name: str) -> None:
    """Require the structural marginal interface.

    Raises:
        TypeError: If a required attribute is missing.

    References:
        Project validation policy. No equation is evaluated.

    """
    for attribute in _REQUIRED:
        if not hasattr(law, attribute):
            raise TypeError(f"{name} lacks required {attribute!r}")


def _sf(law: Any, x: float) -> float:
    """Return ``S(x)`` from ``law.sf`` or ``1 - law.cdf``.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 1]. No new equation.

    """
    method = getattr(law, "sf", None)
    if method is not None:
        return float(method(x))
    return 1.0 - float(law.cdf(x))


def _isf(law: Any, q: float) -> float:
    """Return ``S^-1(q)`` from ``law.isf`` or ``law.quantile(1 - q)``.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 2]. No new equation.

    """
    method = getattr(law, "isf", None)
    if method is not None:
        return float(method(q))
    return float(law.quantile(1.0 - q))


def _log_pdf(law: Any, x: float) -> float:
    """Return ``log f(x)`` from ``law.log_pdf`` or ``log law.pdf``.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 3]. No new equation.

    """
    method = getattr(law, "log_pdf", None)
    if method is not None:
        try:
            return float(method(x))
        except ValueError:
            return -math.inf
    return safe_log(float(law.pdf(x)))


def _default_property(instance: object, name: str) -> float:
    """Return the numerical default of a :class:`ContinuousDistribution` property.

    References:
        [EQ-DIST-BASE-001]. No new equation.

    """
    descriptor = getattr(ContinuousDistribution, name)
    return float(cast(property, descriptor).__get__(instance))


def _finite_point(x: float) -> float:
    """Validate a real point, allowing infinities (handled by the guards).

    References:
        Project validation policy. No equation is evaluated.

    """
    return require_finite_real(x, name="x")


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
        y = self._y(x)
        return float(self.base.cdf(y)) if self.scale > 0.0 else _sf(self.base, y)

    def sf(self, x: float) -> float:
        """Return ``S_Y(y)`` (``F_Y(y)`` for a negative scale).

        Equation:
            [EQ-DIST-AFFINE-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 2.1].

        """
        y = self._y(x)
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


def _base_sample(
    law: Any, size: int, seed: int | RandomStream | None
) -> tuple[float, ...]:
    """Sample a base law by its own sampler when present, else by inversion.

    References:
        [REF-DEVROYE-1986, Chapter II]. No new equation.

    """
    count = require_non_negative_integer(size, name="size")
    stream = resolve_stream(seed)
    sampler = getattr(law, "sample", None)
    if sampler is not None:
        return tuple(float(v) for v in sampler(count, seed=stream))
    generator = stream.generator()
    return tuple(float(law.quantile(open_unit(generator))) for _ in range(count))


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
            Project validation policy. No equation is evaluated.

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


@dataclass(frozen=True, slots=True)
class Mixture(ContinuousDistribution):
    """A finite mixture ``f = sum_i w_i f_i`` of continuous laws.

    ``F = sum w_i F_i`` and ``S = sum w_i S_i``. The quantile solves
    ``F(x) = p`` by Brent's method in the bracket
    ``[min_i Q_i(p), max_i Q_i(p)]``, which always contains the root
    because each ``F_i`` is non-decreasing (in the upper half the residual
    ``q - S(x)`` is used). ``E[X] = sum w_i mu_i`` and
    ``Var X = sum w_i (sigma_i^2 + (mu_i - mu)^2)``.

    Args:
        components: Component laws with the marginal interface.
        weights: Positive weights summing to one within ``1e-9`` (they are
            renormalized exactly).

    References:
        [REF-MCLACHLAN-PEEL-2000, Chapter 1], [REF-BRENT-1973, Chapter 4].

    """

    components: tuple[Any, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the components and weights.

        Raises:
            TypeError: If a component lacks the marginal interface.
            ValueError: If the lengths differ, a weight is not positive, or
                the weights do not sum to one.

        References:
            [EQ-DIST-MIXTURE-001]. No new equation.

        """
        components = tuple(self.components)
        weights = tuple(require_finite_real(w, name="weight") for w in self.weights)
        if not components or len(components) != len(weights):
            raise ValueError("need one positive weight per component")
        for index, component in enumerate(components):
            _check_law(component, f"component {index}")
        if any(w <= 0.0 for w in weights):
            raise ValueError("weights must be positive")
        total = math.fsum(weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("weights must sum to one")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "weights", tuple(w / total for w in weights))

    def _pairs(self) -> Sequence[tuple[float, Any]]:
        """Return ``(weight, component)`` pairs.

        References:
            [EQ-DIST-MIXTURE-001]. No new equation.

        """
        return tuple(zip(self.weights, self.components, strict=True))

    def pdf(self, x: float) -> float:
        """Return ``sum w_i f_i(x)``.

        Equation:
            [EQ-DIST-MIXTURE-001].

        References:
            [REF-MCLACHLAN-PEEL-2000, Chapter 1].

        """
        point = _finite_point(x)
        return math.fsum(w * float(c.pdf(point)) for w, c in self._pairs())

    def log_pdf(self, x: float) -> float:
        """Return ``log sum w_i f_i(x)`` by log-sum-exp of component log densities.

        Equation:
            [EQ-DIST-MIXTURE-001].

        References:
            [REF-MCLACHLAN-PEEL-2000, Chapter 1].

        """
        point = _finite_point(x)
        logs = [math.log(w) + _log_pdf(c, point) for w, c in self._pairs()]
        top = max(logs)
        if top == -math.inf:
            return -math.inf
        return top + math.log(math.fsum(math.exp(v - top) for v in logs))

    def cdf(self, x: float) -> float:
        """Return ``sum w_i F_i(x)``.

        Equation:
            [EQ-DIST-MIXTURE-001].

        References:
            [REF-MCLACHLAN-PEEL-2000, Chapter 1].

        """
        point = _finite_point(x)
        value = math.fsum(w * float(c.cdf(point)) for w, c in self._pairs())
        return min(max(value, 0.0), 1.0)

    def sf(self, x: float) -> float:
        """Return ``sum w_i S_i(x)``.

        Equation:
            [EQ-DIST-MIXTURE-001].

        References:
            [REF-MCLACHLAN-PEEL-2000, Chapter 1].

        """
        point = _finite_point(x)
        value = math.fsum(w * _sf(c, point) for w, c in self._pairs())
        return min(max(value, 0.0), 1.0)

    def _solve(self, p: float, q: float) -> float:
        """Solve ``F(x) = p`` (``S(x) = q`` in the upper half) in the bracket.

        References:
            [EQ-DIST-MIXTURE-001], [REF-BRENT-1973, Chapter 4].

        """
        if p <= 0.5:
            ends = [float(c.quantile(p)) for c in self.components]

            def residual(t: float) -> float:
                """Return ``F(t) - p``.

                References:
                    [EQ-DIST-MIXTURE-001]. No new equation.

                """
                return self.cdf(t) - p

        else:
            ends = [_isf(c, q) for c in self.components]

            def residual(t: float) -> float:
                """Return ``q - S(t)``.

                References:
                    [EQ-DIST-MIXTURE-001]. No new equation.

                """
                return q - self.sf(t)

        low, high = min(ends), max(ends)
        if low == high:
            return low
        if not (math.isfinite(low) and math.isfinite(high)):
            return _bracket_and_solve(residual, low, high)
        return brent_increasing(residual, low, high, residual(low), residual(high))

    def quantile(self, probability: float) -> float:
        """Return the mixture quantile by bracketed Brent iteration.

        Equation:
            [EQ-DIST-MIXTURE-001].

        References:
            [REF-BRENT-1973, Chapter 4].

        """
        p = require_probability(probability, name="probability")
        if p in (0.0, 1.0):
            values = [float(c.quantile(p)) for c in self.components]
            return min(values) if p == 0.0 else max(values)
        return self._solve(p, 1.0 - p)

    def isf(self, probability: float) -> float:
        """Return the mixture inverse survival function ``S(x) = q``.

        Equation:
            [EQ-DIST-MIXTURE-001], [EQ-DIST-ISF-001].

        References:
            [REF-BRENT-1973, Chapter 4].

        """
        q = require_probability(probability, name="probability")
        if q in (0.0, 1.0):
            return self.quantile(1.0 - q)
        return self._solve(1.0 - q, q)

    @property
    def mean(self) -> float:
        """Return ``sum w_i mu_i`` (``nan``/``inf`` propagate).

        Equation:
            [EQ-DIST-MIXTURE-001].

        References:
            [REF-MCLACHLAN-PEEL-2000, Chapter 1].

        """
        return math.fsum(w * float(c.mean) for w, c in self._pairs())

    @property
    def variance(self) -> float:
        """Return ``sum w_i (sigma_i^2 + (mu_i - mu)^2)``.

        Equation:
            [EQ-DIST-MIXTURE-001].

        References:
            [REF-MCLACHLAN-PEEL-2000, Chapter 1].

        """
        mean = self.mean
        if not math.isfinite(mean):
            return math.nan
        return math.fsum(
            w * (float(c.variance) + (float(c.mean) - mean) ** 2)
            for w, c in self._pairs()
        )

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw a component index by weight, then a variate by inversion.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Equation:
            [EQ-DIST-MIXTURE-001].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        cumulative: list[float] = []
        running = 0.0
        for weight in self.weights:
            running += weight
            cumulative.append(running)
        draws: list[float] = []
        for _ in range(count):
            u = generator.random() * running
            index = next(
                (i for i, c in enumerate(cumulative) if u < c), len(cumulative) - 1
            )
            component = self.components[index]
            draws.append(float(component.quantile(open_unit(generator))))
        return tuple(draws)


def _bracket_and_solve(
    residual: Callable[[float], float], low: float, high: float
) -> float:
    """Solve a non-decreasing residual when a bracket end is infinite.

    The infinite end is replaced by the finite end moved outward by doubling
    steps until the residual changes sign.

    References:
        [REF-BRENT-1973, Chapter 4]. No new equation.

    """
    finite = [v for v in (low, high) if math.isfinite(v)]
    centre = finite[0] if finite else 0.0
    step = max(1.0, abs(centre))
    a = low if math.isfinite(low) else centre - step
    b = high if math.isfinite(high) else centre + step
    for _ in range(2000):
        if residual(a) <= 0.0:
            break
        step *= 2.0
        a = centre - step
    for _ in range(2000):
        if residual(b) >= 0.0:
            break
        step *= 2.0
        b = centre + step
    return brent_increasing(residual, a, b, residual(a), residual(b))
