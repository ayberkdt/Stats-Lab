"""Common behaviour shared by every univariate distribution family.

:class:`ContinuousDistribution` and :class:`DiscreteDistribution` are
slot-free mixins. A family defines its own densities, distribution
functions, quantiles, and moments; the mixin supplies consistent defaults
for everything else and normalizes a few edge cases for all families:

* ``cdf``, ``sf``, ``pdf``, ``log_pdf`` (and ``log_cdf``, ``log_sf``,
  ``pmf``, ``log_pmf``) accept ``-inf`` and ``inf`` and return the limits
  ``cdf(-inf) = 0``, ``cdf(inf) = 1``, ``pdf(+-inf) = 0``; ``nan`` is still
  rejected by the family's validation;
* discrete ``pmf``/``log_pmf`` accept integral floats such as ``3.0`` (the
  values that ``quantile`` returns) and give zero mass at ``+-inf``; a
  non-integral float is still a ``TypeError``;
* undefined moments never raise from a property: a moment whose defining
  integral diverges to ``+inf`` is ``inf`` and an undefined one (an
  ``inf - inf`` form, such as the Cauchy mean, or a central moment about an
  undefined mean) is ``nan``.

Defaults are numerical where no closed form is supplied: the entropy is
``int_0^1 -log f(Q(u)) du`` and the central moments are
``int_0^1 (Q(u) - mu)^k du``, both by tanh-sinh quadrature in probability
space; a result that does not converge is reported as ``nan`` rather than
as a number of unknown accuracy.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any, cast

from probstat.foundations.validation import require_probability
from probstat.numerics.double_exponential import tanh_sinh

from ._continuous import safe_log

_MOMENT_TOLERANCE = 1e-10
_CONTINUOUS_LIMITS: dict[str, tuple[float, float]] = {
    "pdf": (0.0, 0.0),
    "log_pdf": (-math.inf, -math.inf),
    "cdf": (0.0, 1.0),
    "sf": (1.0, 0.0),
    "log_cdf": (-math.inf, 0.0),
    "log_sf": (0.0, -math.inf),
}
_DISCRETE_LIMITS: dict[str, tuple[float, float]] = {
    "cdf": (0.0, 1.0),
    "sf": (1.0, 0.0),
    "log_cdf": (-math.inf, 0.0),
    "log_sf": (0.0, -math.inf),
}
_MASS_METHODS: dict[str, float] = {"pmf": 0.0, "log_pmf": -math.inf}


def _member(instance: object, name: str) -> Any:
    """Return an attribute that the concrete family is required to define.

    References:
        Project validation policy. No equation is evaluated.

    """
    return getattr(instance, name)


def _number(instance: object, name: str) -> float:
    """Return a numeric property (such as ``mean``) of the concrete family.

    References:
        Project validation policy. No equation is evaluated.

    """
    return float(_member(instance, name))


def _method(instance: object, name: str) -> Callable[[float], float]:
    """Return a one-argument method (such as ``quantile``) of the family.

    References:
        Project validation policy. No equation is evaluated.

    """
    return cast(Callable[[float], float], _member(instance, name))


def _guard_infinite(
    method: Callable[[Any, Any], float], at_minus: float, at_plus: float
) -> Callable[[Any, Any], float]:
    """Wrap a one-argument method so ``+-inf`` returns its limit.

    Args:
        method: Family method of ``(self, x)``.
        at_minus: Value returned at ``x = -inf``.
        at_plus: Value returned at ``x = inf``.

    Returns:
        Wrapped method; other arguments are passed through unchanged.

    References:
        [REF-KOLMOGOROV-1956]. No equation is evaluated.

    """

    @functools.wraps(method)
    def guarded(self: Any, x: Any) -> float:
        """Return the limit at an infinite argument, else delegate.

        References:
            [REF-KOLMOGOROV-1956]. No equation is evaluated.

        """
        if isinstance(x, float):
            if x == math.inf:
                return at_plus
            if x == -math.inf:
                return at_minus
        return method(self, x)

    guarded.__dict__["_probstat_guarded"] = True
    return guarded


def _guard_mass(
    method: Callable[[Any, Any], float], zero: float
) -> Callable[[Any, Any], float]:
    """Wrap ``pmf``/``log_pmf`` to accept integral floats.

    An integral float (``3.0``) is converted to ``int`` and an infinite one
    has zero mass (``zero`` is ``0`` or ``-inf``). Non-integral floats,
    booleans and other types are passed through, so the family's type check
    still rejects them.

    Args:
        method: Family mass method of ``(self, k)``.
        zero: Value representing zero mass.

    Returns:
        Wrapped method.

    References:
        [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

    """

    @functools.wraps(method)
    def guarded(self: Any, k: Any) -> float:
        """Convert an integral float to ``int`` before delegating.

        References:
            [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

        """
        if isinstance(k, float):
            if math.isnan(k):
                raise ValueError("k must not be NaN")
            if math.isinf(k):
                return zero
            if k.is_integer():
                return method(self, int(k))
        return method(self, k)

    guarded.__dict__["_probstat_guarded"] = True
    return guarded


def _wrap_methods(
    cls: type,
    limits: dict[str, tuple[float, float]],
    masses: dict[str, float] | None = None,
) -> None:
    """Install the infinity and integral-float guards on a family class.

    Only methods defined in the class body are wrapped, and each at most
    once (``dataclass(slots=True)`` re-creates the class from its namespace).

    References:
        [REF-KOLMOGOROV-1956]. No equation is evaluated.

    """
    for name, (at_minus, at_plus) in limits.items():
        method = cls.__dict__.get(name)
        if callable(method) and not getattr(method, "_probstat_guarded", False):
            setattr(cls, name, _guard_infinite(method, at_minus, at_plus))
    for name, zero in (masses or {}).items():
        method = cls.__dict__.get(name)
        if callable(method) and not getattr(method, "_probstat_guarded", False):
            setattr(cls, name, _guard_mass(method, zero))


def _quantile_integral(
    quantile: Callable[[float], float],
    integrand: Callable[[float], float],
    scale: float = 1.0,
) -> float:
    """Return ``int_0^1 h(Q(u)) du`` by tanh-sinh quadrature, ``nan`` if unreliable.

    Args:
        quantile: Quantile function ``Q``.
        integrand: Function ``h`` of a support point.
        scale: Magnitude of typical integrand values; the absolute stopping
            floor is ``1e-10 scale`` so integrals near zero (odd central
            moments of symmetric laws, an entropy of zero) can converge.

    Returns:
        Integral estimate, or ``nan`` when the quadrature reports
        non-convergence or meets a non-finite value (a divergent moment).

    Equation:
        [EQ-DIST-BASE-001].

    References:
        [REF-TAKAHASI-MORI-1974], [REF-CASELLA-BERGER-2002, Chapter 2].

    """

    def wrapped(u: float, complement: float) -> float:
        """Evaluate ``h(Q(u))`` from the smaller of ``u`` and ``1 - u``.

        References:
            [REF-TAKAHASI-MORI-1974]. No equation is evaluated.

        """
        point = quantile(u)
        value = integrand(point)
        if not math.isfinite(value):
            raise ArithmeticError("non-finite integrand")
        return value

    try:
        result = tanh_sinh(
            wrapped,
            0.0,
            1.0,
            tolerance=_MOMENT_TOLERANCE,
            absolute_tolerance=_MOMENT_TOLERANCE * scale,
        )
    except (ArithmeticError, ValueError):
        return math.nan
    return result.value if result.converged else math.nan


class ContinuousDistribution:
    """Mixin with consistent defaults for continuous univariate laws.

    Subclasses provide ``pdf``, ``cdf``, ``sf``, ``quantile``, ``mean`` and
    ``variance``; the mixin supplies ``log_cdf``, ``log_sf``, ``isf``,
    ``median``, ``std``, ``entropy``, ``skewness`` and ``kurtosis`` (excess
    kurtosis) unless the subclass overrides them, and lets the distribution
    functions and densities accept infinite arguments.

    References:
        [REF-CASELLA-BERGER-2002, Chapters 2-3], [REF-JKB-1994, Chapter 1].

    """

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Install the infinite-argument guards on the subclass methods.

        References:
            [REF-KOLMOGOROV-1956]. No equation is evaluated.

        """
        super().__init_subclass__(**kwargs)
        _wrap_methods(cls, _CONTINUOUS_LIMITS)

    def _call(self, name: str, x: float) -> float:
        """Call a family method by name (used by the generic defaults).

        References:
            Project validation policy. No equation is evaluated.

        """
        return float(_method(self, name)(x))

    def log_cdf(self, x: float) -> float:
        """Return ``log F(x)``, through ``log1p(-S(x))`` when ``F(x) > 1/2``.

        Args:
            x: Real point (infinite values give the limits).

        Returns:
            Natural logarithm of the CDF.

        Equation:
            [EQ-DIST-BASE-002].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 1].

        """
        value = self._call("cdf", x)
        if value > 0.5:
            return math.log1p(-self._call("sf", x))
        return safe_log(value)

    def log_sf(self, x: float) -> float:
        """Return ``log S(x)``, through ``log1p(-F(x))`` when ``S(x) > 1/2``.

        Args:
            x: Real point (infinite values give the limits).

        Returns:
            Natural logarithm of the survival function.

        Equation:
            [EQ-DIST-BASE-002].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 1].

        """
        value = self._call("sf", x)
        if value > 0.5:
            return math.log1p(-self._call("cdf", x))
        return safe_log(value)

    def isf(self, probability: float) -> float:
        """Return the inverse survival function ``Q(1 - q)`` (generic fallback).

        Families with tail-accurate inversions override this; the fallback
        loses relative accuracy for ``q`` below about ``1e-16``, where
        ``1 - q`` rounds to one.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Point ``x`` with ``S(x) = q``.

        Equation:
            [EQ-DIST-ISF-001].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        q = require_probability(probability, name="probability")
        return self._call("quantile", 1.0 - q)

    @property
    def median(self) -> float:
        """Return the median ``Q(1/2)``.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        return self._call("quantile", 0.5)

    @property
    def std(self) -> float:
        """Return the standard deviation (``inf`` or ``nan`` with the variance).

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        variance = _number(self, "variance")
        return math.sqrt(variance) if variance >= 0.0 else math.nan

    @property
    def entropy(self) -> float:
        """Return the differential entropy ``-E[log f(X)]`` in nats.

        The default integrates ``-log f(Q(u))`` over ``(0, 1)``; families with
        a closed form override it. ``nan`` signals that the quadrature did not
        converge.

        Equation:
            [EQ-DIST-BASE-001].

        References:
            [REF-COVER-THOMAS-2006, Chapter 8].

        """
        quantile = _method(self, "quantile")
        return _quantile_integral(quantile, lambda x: -self._call("log_pdf", x))

    def _central_moment(self, order: int) -> float:
        """Return ``E[(X - mu)^order]`` numerically, ``nan`` when undefined.

        References:
            [EQ-DIST-BASE-001], [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        mean = _number(self, "mean")
        if not math.isfinite(mean):
            return math.nan
        quantile = _method(self, "quantile")
        variance = _number(self, "variance")
        scale = variance ** (0.5 * order) if math.isfinite(variance) else 1.0
        return _quantile_integral(quantile, lambda x: (x - mean) ** order, scale)

    @property
    def skewness(self) -> float:
        """Return the skewness ``E[(X - mu)^3] / sigma^3``.

        ``nan`` when the variance is infinite or undefined, or the third
        moment cannot be determined. Families with closed forms override it.

        Equation:
            [EQ-DIST-BASE-001].

        References:
            [REF-JKB-1994, Chapter 1].

        """
        variance = _number(self, "variance")
        if not (math.isfinite(variance) and variance > 0.0):
            return math.nan
        return self._central_moment(3) / math.pow(variance, 1.5)

    @property
    def kurtosis(self) -> float:
        """Return the excess kurtosis ``E[(X - mu)^4] / sigma^4 - 3``.

        ``nan`` when the variance is infinite or undefined or the fourth
        moment cannot be determined numerically. Families with closed forms
        override it (returning ``inf`` where the fourth moment diverges).

        Equation:
            [EQ-DIST-BASE-001].

        References:
            [REF-JKB-1994, Chapter 1].

        """
        variance = _number(self, "variance")
        if not (math.isfinite(variance) and variance > 0.0):
            return math.nan
        return self._central_moment(4) / (variance * variance) - 3.0


class DiscreteDistribution:
    """Mixin with consistent defaults for integer-valued laws.

    Subclasses provide ``pmf``, ``log_pmf``, ``cdf``, ``sf``, ``quantile``,
    ``support``, ``mean`` and ``variance``. The mixin adds ``isf``,
    ``log_cdf``, ``log_sf``, ``median``, ``std``, ``entropy``, ``skewness``
    and ``kurtosis`` (excess) computed from the mass function, lets
    ``pmf``/``log_pmf`` accept integral floats, and lets ``cdf``/``sf``
    accept infinite arguments.

    References:
        [REF-ROSS-2019, Chapter 4], [REF-JKB-2005, Chapter 1].

    """

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Install the argument guards on the subclass methods.

        References:
            [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

        """
        super().__init_subclass__(**kwargs)
        _wrap_methods(cls, _DISCRETE_LIMITS, _MASS_METHODS)

    def _call(self, name: str, x: float) -> float:
        """Call a family method by name (used by the generic defaults).

        References:
            Project validation policy. No equation is evaluated.

        """
        return float(_method(self, name)(x))

    def _bounds(self) -> tuple[int, int | None]:
        """Return the integer support bounds ``(lower, upper or None)``.

        References:
            [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

        """
        support = _member(self, "support")
        return int(support.lower), (
            None if support.upper is None else int(support.upper)
        )

    def log_cdf(self, x: float) -> float:
        """Return ``log F(x)``, through ``log1p(-S(x))`` when ``F(x) > 1/2``.

        Equation:
            [EQ-DIST-BASE-002].

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        value = self._call("cdf", x)
        if value > 0.5:
            return math.log1p(-self._call("sf", x))
        return safe_log(value)

    def log_sf(self, x: float) -> float:
        """Return ``log S(x)``, through ``log1p(-F(x))`` when ``S(x) > 1/2``.

        Equation:
            [EQ-DIST-BASE-002].

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        value = self._call("sf", x)
        if value > 0.5:
            return math.log1p(-self._call("cdf", x))
        return safe_log(value)

    def isf(self, probability: float) -> float:
        """Return the generalized inverse survival function ``min{k: S(k) <= q}``.

        The search uses ``S`` directly (doubling then bisection on the
        integers), so small upper-tail probabilities keep their accuracy.
        Mathematically ``isf(q) = quantile(1 - q)``.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Integer-valued float; ``inf`` at ``q = 0`` for unbounded support.

        Equation:
            [EQ-DIST-BASE-003].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        q = require_probability(probability, name="probability")
        lower, upper = self._bounds()
        if q == 0.0:
            return math.inf if upper is None else float(upper)
        if self._call("sf", float(lower)) <= q:
            return float(lower)
        low = lower
        step = 1
        while True:
            high = low + step if upper is None else min(low + step, upper)
            if self._call("sf", float(high)) <= q:
                break
            if upper is not None and high >= upper:
                return float(upper)
            low = high
            step *= 2
        while high - low > 1:
            middle = (low + high) // 2
            if self._call("sf", float(middle)) <= q:
                high = middle
            else:
                low = middle
        return float(high)

    @property
    def median(self) -> float:
        """Return the median ``quantile(1/2)``.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        return self._call("quantile", 0.5)

    @property
    def std(self) -> float:
        """Return the standard deviation.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        variance = _number(self, "variance")
        return math.sqrt(variance) if variance >= 0.0 else math.nan

    def _mass_points(self) -> list[tuple[int, float]]:
        """Return ``(k, p_k)`` over a range holding all but ``1e-17`` of the mass.

        References:
            [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

        """
        lower, upper = self._bounds()
        first = max(int(self._call("quantile", 1e-17)), lower)
        last_value = self._call("isf", 1e-17)
        if math.isfinite(last_value):
            last = int(last_value)
        elif upper is not None:
            last = upper
        else:
            return []
        points: list[tuple[int, float]] = []
        for k in range(first, last + 1):
            mass = self._call("pmf", float(k))
            if mass > 0.0:
                points.append((k, mass))
        return points

    @property
    def entropy(self) -> float:
        """Return the Shannon entropy ``-sum p_k log p_k`` in nats.

        Summed over the support range that carries all but ``2e-17`` of the
        probability. Families with closed forms override it.

        Equation:
            [EQ-DIST-BASE-004].

        References:
            [REF-COVER-THOMAS-2006, Chapter 2].

        """
        return -math.fsum(p * math.log(p) for _, p in self._mass_points())

    def _central_moment(self, order: int) -> float:
        """Return ``sum (k - mu)^order p_k`` over the effective support.

        References:
            [EQ-DIST-BASE-004], [REF-ROSS-2019, Chapter 4].

        """
        mean = _number(self, "mean")
        return math.fsum((k - mean) ** order * p for k, p in self._mass_points())

    @property
    def skewness(self) -> float:
        """Return the skewness ``mu_3 / sigma^3`` (``nan`` for zero variance).

        Equation:
            [EQ-DIST-BASE-004].

        References:
            [REF-JKB-2005, Chapter 1].

        """
        variance = _number(self, "variance")
        if not variance > 0.0:
            return math.nan
        return self._central_moment(3) / math.pow(variance, 1.5)

    @property
    def kurtosis(self) -> float:
        """Return the excess kurtosis ``mu_4 / sigma^4 - 3`` (``nan`` if degenerate).

        Equation:
            [EQ-DIST-BASE-004].

        References:
            [REF-JKB-2005, Chapter 1].

        """
        variance = _number(self, "variance")
        if not variance > 0.0:
            return math.nan
        return self._central_moment(4) / (variance * variance) - 3.0
