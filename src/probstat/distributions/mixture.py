"""Finite mixtures of continuous laws (ADR-0059).

The components may be any objects with the structural marginal interface;
see :mod:`probstat.distributions._wrapped` for the optional methods they may
add.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.numerics.integration import integrate

from ._base import ContinuousDistribution, _quantile_integral
from ._continuous import brent_increasing, open_unit
from ._wrapped import (
    _check_law,
    _finite_point,
    _isf,
    _log_pdf,
    _sf,
)


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

    def _central_moment(self, order: int) -> float:
        """Return ``E[(X - mu)^order]`` from the component moments (orders 3, 4).

        With ``d_i = mu_i - mu`` and component central moments ``m_ik``,
        ``E[(X - mu)^3] = sum w_i (m_i3 + 3 d_i m_i2 + d_i^3)`` and
        ``E[(X - mu)^4] = sum w_i (m_i4 + 4 d_i m_i3 + 6 d_i^2 m_i2 + d_i^4)``.
        Components without ``skewness``/``kurtosis`` fall back to the generic
        quadrature of the mixin; a component with an infinite moment of the
        requested order makes the mixture moment infinite.

        Equation:
            [EQ-DIST-MIXTURE-002].

        References:
            [REF-MCLACHLAN-PEEL-2000, Chapter 1].

        """
        mean = self.mean
        if not math.isfinite(mean) or order not in (3, 4):
            return ContinuousDistribution._central_moment(self, order)
        terms: list[float] = []
        for weight, component in self._pairs():
            if not (hasattr(component, "skewness") and hasattr(component, "kurtosis")):
                return ContinuousDistribution._central_moment(self, order)
            variance = float(component.variance)
            skewness = float(component.skewness)
            d = float(component.mean) - mean
            if variance == 0.0:
                m3 = m4 = 0.0
            else:
                m3 = skewness * variance**1.5
                m4 = (
                    (float(component.kurtosis) + 3.0) * variance * variance
                    if order == 4
                    else 0.0
                )
            if order == 3:
                value = m3 + 3.0 * d * variance + d**3
            else:
                value = m4 + 4.0 * d * m3 + 6.0 * d * d * variance + d**4
            if math.isnan(value):
                return ContinuousDistribution._central_moment(self, order)
            terms.append(weight * value)
        return math.fsum(terms)

    @property
    def entropy(self) -> float:
        """Return ``-int f log f`` by adaptive quadrature in ``x``.

        The support is split at every finite support end and median of the
        components (where a component density may be singular or kinked) and
        each piece is integrated by :func:`probstat.numerics.integrate` to a
        relative ``1e-10`` (an error estimate below ``1e-8`` is accepted); a
        node that rounds onto a singular point (density ``inf``) is a
        measure-zero event and contributes nothing. If a piece fails, the
        generic probability-space quadrature of the mixin is used, and
        ``nan`` signals that neither converged.

        Equation:
            [EQ-DIST-MIXTURE-002], [EQ-DIST-BASE-001].

        References:
            [REF-COVER-THOMAS-2006, Chapter 8], [REF-PIESSENS-QUADPACK-1983].

        """
        lows = [float(c.quantile(0.0)) for c in self.components]
        highs = [float(c.quantile(1.0)) for c in self.components]
        lower, upper = min(lows), max(highs)
        inner = {v for v in lows + highs if math.isfinite(v)}
        inner.update(float(c.quantile(0.5)) for c in self.components)
        edges = [lower, *sorted(v for v in inner if lower < v < upper), upper]

        def integrand(x: float) -> float:
            """Return ``-f log f`` (zero where ``f`` is zero or infinite).

            References:
                [REF-COVER-THOMAS-2006, Chapter 8].

            """
            density = self.pdf(x)
            if not (density > 0.0 and math.isfinite(density)):
                return 0.0
            return -density * math.log(density)

        pieces: list[float] = []
        for a, b in zip(edges, edges[1:], strict=False):
            try:
                result = integrate(integrand, a, b, rel_tol=1e-10, abs_tol=1e-13)
            except (ArithmeticError, ValueError):
                return _quantile_integral(self, lambda x: -self.log_pdf(x))
            # Next to a singular end away from zero the nodes round onto the
            # end, so a piece may stop just short of its 1e-10 target; an
            # error estimate below 1e-8 is still accepted.
            if not (
                result.converged
                or result.error_estimate <= 1e-8 * max(1.0, abs(result.value))
            ):
                return _quantile_integral(self, lambda x: -self.log_pdf(x))
            pieces.append(result.value)
        return math.fsum(pieces)

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
