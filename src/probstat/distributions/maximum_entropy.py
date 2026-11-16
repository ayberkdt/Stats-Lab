"""Maximum-entropy laws on an interval under moment and expectation constraints.

Given a support ``K`` (a bounded interval, a half-line, or the real line) and
constraints ``E[g_m(X)] = h_m``, the density of largest differential entropy
has the exponential form ``f(x) = 1_K(x) exp(-lambda_0 - sum_m lambda_m
g_m(x))`` (Jaynes 1957). The multipliers minimize the convex dual

    ``Gamma(lambda) = log int_K exp(-sum_m lambda_m (g_m(x) - h_m)) dx``

(Mead and Papanicolaou 1984), whose gradient is ``h - E_lambda[g]`` and whose
Hessian is the covariance matrix ``Cov_lambda[g]``. :func:`maximum_entropy`
minimizes it by Newton's method with a backtracking (Armijo) line search, or
by :func:`probstat.optimization.minimize_bfgs` on request.

Quadrature. The support is mapped to a finite interval ``t`` (identity for a
bounded support, ``x = a + s t/(1 - t)`` for a half-line, ``x = c + s t/(1 -
t^2)`` for the real line) and split into equal panels in ``t``; the Newton
iterations use the fixed 15-point Kronrod rule of
:mod:`probstat.numerics.gauss_kronrod` on every panel, so each feature
``g_m`` is evaluated once per node. After convergence the constraint values,
the normalizing constant, and the moments are recomputed by adaptive G7-K15
integration on every panel; when that independent check disagrees with the
fixed rule, or an unbounded panel at the end of the support still carries
mass, the number of panels is doubled and the solve restarts from the current
multipliers.

Feasibility. Raw moments ``1..K`` given contiguously are checked against the
interior of the moment space before solving: the Hankel matrix
``[m_(i+j)]`` and the localizing matrices of the support must be positive
definite (for standardized moments this contains Pearson's inequality
``kurtosis > skewness^2 + 1`` and, on ``[a, b]``, the Bhatia-Davis bound
``variance < (mean - a)(b - mean)``). On an unbounded support a maximum need
not be attained even for feasible moments (Junk 2000): the dual minimizer
then drives mass to infinity, which the panel test detects and reports.
"""

from __future__ import annotations

import bisect
import math
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.linalg import cholesky, cholesky_solve, symmetric_eigen
from probstat.numerics import gauss_kronrod
from probstat.numerics.gauss_kronrod import K15_NODES, K15_WEIGHTS, kronrod_panel
from probstat.optimization import brent_root, minimize_bfgs

from ._base import ContinuousDistribution
from .support import RealSupport

Feature = Callable[[float], float]

_METHODS = ("newton", "bfgs")
_ARMIJO = 1.0e-4
_MAX_BACKTRACKS = 60
_INITIAL_PANELS = 64
_MAX_PANELS = 2048
_CHECK_TOLERANCE = 1.0e-10
_DECAY_MARGIN = 40.0
_FEASIBILITY_TOLERANCE = 1.0e-12
_ROUNDING_DECREMENT = 1.0e-13
_EPS = sys.float_info.epsilon


@dataclass(frozen=True, slots=True)
class MaximumEntropyDiagnostics:
    """Convergence record of a maximum-entropy solve.

    Args:
        converged: Whether the relative residual test was met on the final
            quadrature grid and confirmed by adaptive quadrature.
        method: ``"newton"`` or ``"bfgs"``.
        iterations: Optimizer iterations summed over all grid refinements.
        dual_value: Final value of the dual ``Gamma(lambda)``; at the optimum
            it equals the entropy.
        max_relative_residual: ``max_m |E[g_m] - h_m| / sd(g_m)`` from the
            adaptive check.
        panels: Number of quadrature panels of the final grid.
        nodes: Number of fixed-rule nodes of the final grid.
        quadrature_error: Summed adaptive error estimate of the normalizing
            integral, relative to its value.
        message: Termination reason.

    References:
        [REF-MEAD-PAPANICOLAOU-1984], [REF-NOCEDAL-WRIGHT-2006, Chapter 3].

    """

    converged: bool
    method: str
    iterations: int
    dual_value: float
    max_relative_residual: float
    panels: int
    nodes: int
    quadrature_error: float
    message: str


@dataclass(frozen=True, slots=True)
class _Map:
    """Increasing map ``x = phi(t)`` from a finite ``t`` range onto the support.

    References:
        [REF-PIESSENS-QUADPACK-1983, Section 2.3] (interval substitutions).
        No new equation.

    """

    kind: str
    lower: float
    upper: float
    location: float
    scale: float

    @property
    def t_range(self) -> tuple[float, float]:
        """Return the ``t`` interval of the map.

        References:
            [REF-PIESSENS-QUADPACK-1983, Section 2.3]. No equation is
            evaluated.

        """
        return {
            "bounded": (0.0, 1.0),
            "lower": (0.0, 1.0),
            "upper": (-1.0, 0.0),
            "real": (-1.0, 1.0),
        }[self.kind]

    def x_and_jacobian(self, t: float) -> tuple[float, float]:
        """Return ``(phi(t), phi'(t))``.

        References:
            [REF-PIESSENS-QUADPACK-1983, Section 2.3]. No new equation.

        """
        kind = self.kind
        if kind == "bounded":
            width = self.upper - self.lower
            return self.lower + width * t, width
        if kind == "lower":
            gap = 1.0 - t
            return self.lower + self.scale * t / gap, self.scale / (gap * gap)
        if kind == "upper":
            gap = 1.0 + t
            return self.upper + self.scale * t / gap, self.scale / (gap * gap)
        gap = 1.0 - t * t
        return (
            self.location + self.scale * t / gap,
            self.scale * (1.0 + t * t) / (gap * gap),
        )

    def inverse(self, x: float) -> float:
        """Return ``t = phi^-1(x)`` for a point of the support.

        References:
            [REF-PIESSENS-QUADPACK-1983, Section 2.3]. No new equation.

        """
        kind = self.kind
        if kind == "bounded":
            return (x - self.lower) / (self.upper - self.lower)
        if kind == "lower":
            y = (x - self.lower) / self.scale
            return y / (1.0 + y)
        if kind == "upper":
            y = (x - self.upper) / self.scale
            return y / (1.0 - y)
        y = (x - self.location) / self.scale
        return 2.0 * y / (1.0 + math.sqrt(1.0 + 4.0 * y * y))


@dataclass(frozen=True, slots=True)
class _Basis:
    """Centred monomials ``((x - c)/s)^k, k = 1..K`` standing for ``x^1..x^K``.

    Both bases span the same polynomials, so they define the same family of
    densities; the centred one keeps the exponent free of cancellation when
    the support lies far from zero. Features after the first ``K`` are passed
    through unchanged.

    References:
        [EQ-DIST-MAXENT-002]. Binomial expansion; no new equation.

    """

    order: int
    centre: float
    scale: float

    def to_reported(self, internal: Sequence[float]) -> tuple[float, ...]:
        """Map centred-basis coefficients to raw-monomial coefficients.

        ``lambda_j = sum_(k >= j) beta_k C(k, j) (-c)^(k-j) / s^k``.

        References:
            [EQ-DIST-MAXENT-002]. No new equation.

        """
        order, c, s = self.order, self.centre, self.scale
        head = tuple(
            math.fsum(
                internal[k - 1] * math.comb(k, j) * (-c) ** (k - j) / s**k
                for k in range(j, order + 1)
            )
            for j in range(1, order + 1)
        )
        return head + tuple(internal[order:])

    def to_internal(self, reported: Sequence[float]) -> tuple[float, ...]:
        """Map raw-monomial coefficients to centred-basis coefficients.

        ``beta_k = sum_(j >= k) lambda_j C(j, k) c^(j-k) s^k``.

        References:
            [EQ-DIST-MAXENT-002]. No new equation.

        """
        order, c, s = self.order, self.centre, self.scale
        head = tuple(
            math.fsum(
                reported[j - 1] * math.comb(j, k) * c ** (j - k) * s**k
                for j in range(k, order + 1)
            )
            for k in range(1, order + 1)
        )
        return head + tuple(reported[order:])

    def residuals_to_reported(self, internal: Sequence[float]) -> tuple[float, ...]:
        """Map ``E[z^k] - mu_k`` to ``E[X^j] - m_j``.

        Uses ``x^j = sum_k C(j, k) c^(j-k) s^k z^k``.

        References:
            [EQ-DIST-MAXENT-002]. No new equation.

        """
        order, c, s = self.order, self.centre, self.scale
        head = tuple(
            math.fsum(
                internal[k - 1] * math.comb(j, k) * c ** (j - k) * s**k
                for k in range(1, j + 1)
            )
            for j in range(1, order + 1)
        )
        return head + tuple(internal[order:])


@dataclass(frozen=True, slots=True, eq=False)
class MaximumEntropy(ContinuousDistribution):
    """Maximum-entropy law ``f(x) = 1_K(x) exp(-lambda_0 - sum lambda_m g_m(x))``.

    Built by :func:`maximum_entropy`; the constructor is not meant to be
    called directly.

    Args:
        lower: Lower end of the support (``-inf`` allowed).
        upper: Upper end of the support (``inf`` allowed).
        multipliers: Lagrange multipliers ``lambda_m``, one per constraint.
        log_normalizer: ``lambda_0 = log int_K exp(-sum lambda_m g_m)``.
        constraint_names: Name of each constraint, such as ``"E[X^2]"``.
        targets: Prescribed values ``h_m``.
        residuals: Achieved ``E_f[g_m] - h_m`` by adaptive quadrature.
        diagnostics: Convergence record.

    References:
        [REF-JAYNES-1957], [REF-MEAD-PAPANICOLAOU-1984],
        [REF-COVER-THOMAS-2006, Chapter 12].

    """

    lower: float
    upper: float
    multipliers: tuple[float, ...]
    log_normalizer: float
    constraint_names: tuple[str, ...]
    targets: tuple[float, ...]
    residuals: tuple[float, ...]
    diagnostics: MaximumEntropyDiagnostics
    _features: tuple[Feature, ...] = field(repr=False)
    _lambdas: tuple[float, ...] = field(repr=False)
    _internal_targets: tuple[float, ...] = field(repr=False)
    _internal_residuals: tuple[float, ...] = field(repr=False)
    _map: _Map = field(repr=False)
    _edges: tuple[float, ...] = field(repr=False)
    _cumulative: tuple[float, ...] = field(repr=False)
    _upper_tail: tuple[float, ...] = field(repr=False)
    _pieces: tuple[float, ...] = field(repr=False)
    _shift: float = field(repr=False)
    _log_mass: float = field(repr=False)
    _mean: float = field(repr=False)
    _variance: float = field(repr=False)

    @property
    def support(self) -> RealSupport:
        """Return the support interval ``K``.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 3]. No equation is evaluated.

        """
        return RealSupport(
            None if math.isinf(self.lower) else self.lower,
            None if math.isinf(self.upper) else self.upper,
            lower_closed=math.isfinite(self.lower),
            upper_closed=math.isfinite(self.upper),
        )

    def _exponent(self, x: float) -> float:
        """Return ``sum_m lambda_m (g_m(x) - h_m)`` at a support point.

        References:
            [EQ-DIST-MAXENT-001]. No new equation.

        """
        return math.fsum(
            lam * (feature(x) - target)
            for lam, feature, target in zip(
                self._lambdas, self._features, self._internal_targets, strict=True
            )
        )

    def log_pdf(self, x: float) -> float:
        """Return ``log f(x)``, ``-inf`` outside the support.

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-JAYNES-1957].

        """
        point = require_finite_real(x, name="x")
        if point < self.lower or point > self.upper:
            return -math.inf
        return self._shift - self._exponent(point) - self._log_mass

    def pdf(self, x: float) -> float:
        """Return the density ``f(x)``.

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-JAYNES-1957].

        """
        value = self.log_pdf(x)
        return math.exp(value) if value > -math.inf else 0.0

    def _t_density(self, t: float) -> float:
        """Return the density in the ``t`` coordinate, ``f(phi(t)) phi'(t)``.

        References:
            [EQ-DIST-MAXENT-001]. No new equation.

        """
        x, jacobian = self._map.x_and_jacobian(t)
        if not math.isfinite(x):
            return 0.0
        log_value = self._shift - self._exponent(x) - self._log_mass
        return math.exp(log_value) * jacobian if log_value > -745.0 else 0.0

    def _partial(self, panel: int, t: float) -> float:
        """Return the probability of ``[t_panel, t]`` by adaptive quadrature.

        References:
            [EQ-NUM-GK15-001], [REF-PIESSENS-QUADPACK-1983]. No new equation.

        """
        start = self._edges[panel]
        if t <= start:
            return 0.0
        return gauss_kronrod(
            self._t_density, start, t, abs_tol=self._floor(panel), rel_tol=1e-13
        ).value

    def _locate(self, x: float) -> tuple[int, float]:
        """Return the panel index and ``t`` coordinate of a support point.

        References:
            [EQ-DIST-MAXENT-001]. No equation is evaluated.

        """
        t = self._map.inverse(x)
        panel = bisect.bisect_right(self._edges, t) - 1
        return min(max(panel, 0), len(self._edges) - 2), t

    def cdf(self, x: float) -> float:
        """Return ``F(x)`` by panel sums and adaptive quadrature in one panel.

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-PIESSENS-QUADPACK-1983, Section 2.2].

        """
        point = require_finite_real(x, name="x")
        if point <= self.lower:
            return 0.0
        if point >= self.upper:
            return 1.0
        panel, t = self._locate(point)
        value = self._cumulative[panel] + self._split(panel, t)[0]
        return min(max(value, 0.0), 1.0)

    def _floor(self, panel: int) -> float:
        """Return the absolute quadrature target ``64 eps`` times the panel mass.

        It sits above the rule's rounding floor, so an integrand known only
        to a few rounding errors still converges, while tail probabilities
        keep their relative accuracy.

        References:
            [EQ-NUM-GK15-001], [REF-PIESSENS-QUADPACK-1983]. No new equation.

        """
        return max(64.0 * _EPS * self._pieces[panel], 1e-300)

    def _split(self, panel: int, t: float) -> tuple[float, float]:
        """Return the probabilities of the panel left and right of ``t``.

        The smaller part is integrated directly and the larger one is the
        panel probability minus it, so ``F(x) + S(x) = 1`` holds to rounding
        and each tail keeps its relative accuracy.

        References:
            [EQ-NUM-GK15-001], [REF-PIESSENS-QUADPACK-1983]. No new equation.

        """
        piece = self._pieces[panel]
        left = self._partial(panel, t)
        if left <= 0.5 * piece:
            return left, max(piece - left, 0.0)
        end = self._edges[panel + 1]
        right = (
            gauss_kronrod(
                self._t_density, t, end, abs_tol=self._floor(panel), rel_tol=1e-13
            ).value
            if t < end
            else 0.0
        )
        return max(piece - right, 0.0), right

    def sf(self, x: float) -> float:
        """Return ``1 - F(x)`` summed from the upper end (no cancellation).

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-PIESSENS-QUADPACK-1983, Section 2.2].

        """
        point = require_finite_real(x, name="x")
        if point <= self.lower:
            return 1.0
        if point >= self.upper:
            return 0.0
        panel, t = self._locate(point)
        value = self._upper_tail[panel + 1] + self._split(panel, t)[1]
        return min(max(value, 0.0), 1.0)

    def quantile(self, probability: float) -> float:
        """Return ``x`` with ``F(x) = p`` by Brent's method inside one panel.

        Args:
            probability: Probability in ``[0, 1]``.

        Returns:
            Quantile; the support ends at ``p = 0`` and ``p = 1``.

        Raises:
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-BRENT-1973, Chapter 4].

        """
        p = require_probability(probability, name="probability")
        if p == 0.0:
            return self.lower
        if p == 1.0:
            return self.upper
        if p > 0.5:
            return self.isf(1.0 - p)
        panel = bisect.bisect_right(self._cumulative, p) - 1
        panel = min(max(panel, 0), len(self._edges) - 2)
        base = self._cumulative[panel]

        def residual(t: float) -> float:
            """Return ``F(phi(t)) - p`` inside the bracketing panel.

            References:
                [EQ-DIST-MAXENT-001]. No new equation.

            """
            return base + self._partial(panel, t) - p

        return self._root_in_panel(residual, panel)

    def isf(self, probability: float) -> float:
        """Return ``x`` with ``1 - F(x) = q``, solved on the upper-tail sums.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Inverse survival function; the support ends at ``q = 1`` and
            ``q = 0``.

        Raises:
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-BRENT-1973, Chapter 4].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return self.upper
        if q == 1.0:
            return self.lower
        if q > 0.5:
            return self.quantile(1.0 - q)
        tails = self._upper_tail
        # tails is non-increasing; find the panel with tails[j] >= q > tails[j+1].
        low, high = 0, len(tails) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if tails[middle] >= q:
                low = middle
            else:
                high = middle
        panel = min(low, len(self._edges) - 2)
        end = self._edges[panel + 1]
        above = tails[panel + 1]

        def residual(t: float) -> float:
            """Return ``q - S(phi(t))``, increasing in ``t``.

            References:
                [EQ-DIST-MAXENT-001]. No new equation.

            """
            inside = (
                gauss_kronrod(
                    self._t_density, t, end, abs_tol=self._floor(panel), rel_tol=1e-13
                ).value
                if t < end
                else 0.0
            )
            return q - (above + inside)

        return self._root_in_panel(residual, panel)

    def _root_in_panel(self, residual: Callable[[float], float], panel: int) -> float:
        """Solve an increasing ``residual(t) = 0`` inside one panel; return ``x``.

        References:
            [REF-BRENT-1973, Chapter 4]. No new equation.

        """
        start, end = self._edges[panel], self._edges[panel + 1]
        if residual(end) <= 0.0:
            t = end
        elif residual(start) >= 0.0:
            t = start
        else:
            t = brent_root(residual, start, end, tolerance=1e-15).root
        t_low, t_high = self._map.t_range
        if t <= t_low:
            return self.lower
        if t >= t_high:
            return self.upper
        x, _ = self._map.x_and_jacobian(t)
        return min(max(x, self.lower), self.upper)

    def expect(self, function: Callable[[float], float]) -> float:
        """Return ``E[function(X)]`` by adaptive G7-K15 quadrature on the panels.

        Args:
            function: Real function of a support point.

        Returns:
            The expectation.

        Raises:
            ValueError: If the function returns a non-finite value where the
                density is positive.

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-PIESSENS-QUADPACK-1983, Section 2.3].

        """

        def integrand(t: float) -> float:
            """Return ``function(phi(t)) f(phi(t)) phi'(t)``.

            References:
                [EQ-DIST-MAXENT-001]. No new equation.

            """
            weight = self._t_density(t)
            if weight == 0.0:
                return 0.0
            x, _ = self._map.x_and_jacobian(t)
            return function(x) * weight

        return _panel_integral(integrand, self._edges)[0]

    @property
    def mean(self) -> float:
        """Return ``E[X]``.

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-COVER-THOMAS-2006, Chapter 12].

        """
        return self._mean

    @property
    def variance(self) -> float:
        """Return ``Var[X]``.

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-COVER-THOMAS-2006, Chapter 12].

        """
        return self._variance

    @property
    def skewness(self) -> float:
        """Return ``E[(X - mu)^3] / sigma^3`` by adaptive quadrature.

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-JKB-1994, Chapter 1].

        """
        mu = self._mean
        return self.expect(lambda x: (x - mu) ** 3) / math.pow(self._variance, 1.5)

    @property
    def kurtosis(self) -> float:
        """Return the excess kurtosis ``E[(X - mu)^4] / sigma^4 - 3``.

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-JKB-1994, Chapter 1].

        """
        mu = self._mean
        return self.expect(lambda x: (x - mu) ** 4) / self._variance**2 - 3.0

    @property
    def entropy(self) -> float:
        """Return ``-E[log f(X)] = lambda_0 + sum_m lambda_m E[g_m(X)]`` in nats.

        At the optimum this equals the minimum of the dual (strong duality).

        Equation:
            [EQ-DIST-MAXENT-001].

        References:
            [REF-COVER-THOMAS-2006, Chapter 12].

        """
        return (
            self._log_mass
            - self._shift
            + math.fsum(
                lam * r
                for lam, r in zip(self._lambdas, self._internal_residuals, strict=True)
            )
        )

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw observations by inverting the distribution function.

        Each draw solves ``F(x) = U`` (about a millisecond per draw).

        Args:
            size: Number of draws, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of observations in the support.

        References:
            [REF-DEVROYE-1986, Chapter II]. Sampling inverts
            [EQ-DIST-MAXENT-001].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        draws: list[float] = []
        for _ in range(count):
            u = generator.random()
            while u == 0.0:
                u = generator.random()
            draws.append(self.quantile(u))
        return tuple(draws)


def _panel_integral(
    function: Callable[[float], float], edges: Sequence[float]
) -> tuple[float, float, tuple[float, ...]]:
    """Integrate over each panel by adaptive G7-K15 and sum.

    A first K15 pass estimates ``int |f|`` on every panel; each panel then
    runs adaptive G7-K15 to the relative target ``1e-13`` or the absolute
    target ``64 eps int_panel |f| + 1e-14 int |f| / panels``, whichever is
    looser, so the target never falls below the rule's rounding floor
    ``50 eps int_panel |f|`` when the integrand changes sign.

    Returns:
        ``(total, summed error estimate, per-panel values)``.

    References:
        [EQ-NUM-GK15-001], [EQ-NUM-QAG-001],
        [REF-PIESSENS-QUADPACK-1983, Section 2.3].

    """
    pairs = tuple(zip(edges[:-1], edges[1:], strict=True))
    magnitudes = [kronrod_panel(function, start, end)[2] for start, end in pairs]
    floor = 1e-14 * math.fsum(magnitudes) / len(pairs)
    values: list[float] = []
    errors: list[float] = []
    for (start, end), magnitude in zip(pairs, magnitudes, strict=True):
        absolute = max(64.0 * _EPS * magnitude + floor, 1e-300)
        result = gauss_kronrod(function, start, end, abs_tol=absolute, rel_tol=1e-13)
        values.append(result.value)
        errors.append(result.error_estimate)
    return math.fsum(values), math.fsum(errors), tuple(values)


@dataclass(frozen=True, slots=True)
class _Grid:
    """Fixed composite Kronrod nodes, weights, and feature values.

    References:
        [EQ-NUM-GK15-001]. No new equation.

    """

    edges: tuple[float, ...]
    points: tuple[float, ...]
    weights: tuple[float, ...]
    centred: tuple[tuple[float, ...], ...]
    outer: tuple[tuple[int, ...], ...]


def _build_grid(
    mapping: _Map,
    panels: int,
    features: Sequence[Feature],
    targets: Sequence[float],
) -> _Grid:
    """Place 15 Kronrod nodes in each equal ``t`` panel and evaluate the features.

    Raises:
        ValueError: If a feature returns a non-finite value at a node.

    References:
        [EQ-NUM-GK15-001], [REF-PIESSENS-QUADPACK-1983, Section 2.2].

    """
    t_low, t_high = mapping.t_range
    step = (t_high - t_low) / panels
    edges = tuple(t_low + index * step for index in range(panels)) + (t_high,)
    half = 0.5 * step
    offsets = [
        (-node, weight) for node, weight in zip(K15_NODES, K15_WEIGHTS, strict=True)
    ]
    offsets += [
        (node, weight)
        for node, weight in zip(K15_NODES[:-1], K15_WEIGHTS[:-1], strict=True)
    ]
    points: list[float] = []
    weights: list[float] = []
    centred: list[tuple[float, ...]] = []
    panel_nodes: list[list[int]] = [[] for _ in range(panels)]
    for panel in range(panels):
        centre = edges[panel] + half
        for node, weight in offsets:
            x, jacobian = mapping.x_and_jacobian(centre + half * node)
            if not (math.isfinite(x) and math.isfinite(jacobian)):
                continue
            row: list[float] = []
            for feature, target in zip(features, targets, strict=True):
                value = float(feature(x))
                if not math.isfinite(value):
                    raise ValueError(
                        f"a constraint function is not finite at x = {x!r}"
                    )
                row.append(value - target)
            panel_nodes[panel].append(len(points))
            points.append(x)
            weights.append(weight * half * jacobian)
            centred.append(tuple(row))
    outer: list[tuple[int, ...]] = []
    if mapping.kind in ("upper", "real"):
        outer.append(tuple(panel_nodes[0]))
    if mapping.kind in ("lower", "real"):
        outer.append(tuple(panel_nodes[-1]))
    return _Grid(edges, tuple(points), tuple(weights), tuple(centred), tuple(outer))


@dataclass(frozen=True, slots=True)
class _DualState:
    """Dual value, node probabilities, residual vector, and Hessian at ``lambda``.

    References:
        [EQ-DIST-MAXENT-001]. No new equation.

    """

    value: float
    shift: float
    probabilities: tuple[float, ...]
    residual: tuple[float, ...]
    hessian: tuple[tuple[float, ...], ...]


def _dual(grid: _Grid, multipliers: Sequence[float], *, second: bool) -> _DualState:
    """Evaluate ``Gamma(lambda)``, its gradient ``-r`` and Hessian ``Cov[g]``.

    The exponent is shifted by its minimum over the nodes, so the sum of
    exponentials never overflows.

    References:
        [EQ-DIST-MAXENT-001], [REF-MEAD-PAPANICOLAOU-1984].

    """
    exponents = [
        math.fsum(lam * value for lam, value in zip(multipliers, row, strict=True))
        for row in grid.centred
    ]
    shift = min(exponents)
    raw = [
        weight * math.exp(shift - exponent)
        for weight, exponent in zip(grid.weights, exponents, strict=True)
    ]
    total = math.fsum(raw)
    if not (total > 0.0 and math.isfinite(total)):
        return _DualState(math.inf, shift, (), (), ())
    probabilities = tuple(value / total for value in raw)
    size = len(multipliers)
    residual = tuple(
        math.fsum(
            p * row[m] for p, row in zip(probabilities, grid.centred, strict=True)
        )
        for m in range(size)
    )
    hessian: tuple[tuple[float, ...], ...] = ()
    if second:
        hessian = tuple(
            tuple(
                math.fsum(
                    p * row[i] * row[j]
                    for p, row in zip(probabilities, grid.centred, strict=True)
                )
                - residual[i] * residual[j]
                for j in range(size)
            )
            for i in range(size)
        )
    return _DualState(math.log(total) - shift, shift, probabilities, residual, hessian)


def _relative_residual(
    residual: Sequence[float], hessian: Sequence[Sequence[float]]
) -> float:
    """Return ``max_m |r_m| / sd(g_m)`` (``inf`` for a degenerate feature).

    References:
        [EQ-DIST-MAXENT-001]. No equation is evaluated.

    """
    worst = 0.0
    for index, value in enumerate(residual):
        spread = math.sqrt(max(hessian[index][index], 0.0))
        if spread == 0.0:
            return math.inf if value != 0.0 else worst
        worst = max(worst, abs(value) / spread)
    return worst


class _DegenerateDensity(ArithmeticError):
    """A constraint function has zero variance under the current density.

    References:
        [EQ-DIST-MAXENT-001]. No equation is evaluated.

    """


def _newton_step(state: _DualState) -> tuple[float, ...]:
    """Solve ``Cov[g] delta = r`` with Jacobi scaling and a ridge if needed.

    Raises:
        _DegenerateDensity: If a constraint function has zero variance.
        ValueError: If the constraint functions are linearly dependent on the
            support (the Hessian is singular).

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 3.4] (modified Newton).

    """
    hessian = state.hessian
    size = len(hessian)
    scales = [
        math.sqrt(hessian[i][i]) if hessian[i][i] > 0.0 else 0.0 for i in range(size)
    ]
    if any(scale == 0.0 for scale in scales):
        raise _DegenerateDensity
    scaled = [
        [hessian[i][j] / (scales[i] * scales[j]) for j in range(size)]
        for i in range(size)
    ]
    rhs = tuple(state.residual[i] / scales[i] for i in range(size))
    ridge = 0.0
    for _ in range(12):
        matrix = tuple(
            tuple(scaled[i][j] + (ridge if i == j else 0.0) for j in range(size))
            for i in range(size)
        )
        try:
            solution = cholesky_solve(cholesky(matrix), rhs)
        except ValueError:
            ridge = 1e-12 if ridge == 0.0 else 100.0 * ridge
            continue
        return tuple(solution[i] / scales[i] for i in range(size))
    raise ValueError(
        "the constraint functions are linearly dependent on the support "
        "(singular covariance matrix)"
    )


def _newton(
    grid: _Grid, start: Sequence[float], *, tolerance: float, max_iterations: int
) -> tuple[tuple[float, ...], _DualState, int, bool, str]:
    """Minimize the dual by Newton's method with Armijo backtracking.

    References:
        [EQ-DIST-MAXENT-001], [REF-MEAD-PAPANICOLAOU-1984],
        [REF-NOCEDAL-WRIGHT-2006, Section 3.1].

    """
    multipliers = tuple(start)
    state = _dual(grid, multipliers, second=True)
    if not math.isfinite(state.value):
        raise ValueError(
            "the starting multipliers give a non-integrable exponent; pass "
            "decaying initial multipliers"
        )
    if not multipliers:
        return multipliers, state, 0, True, "no constraints"
    for iteration in range(1, max_iterations + 1):
        if _relative_residual(state.residual, state.hessian) <= tolerance:
            return multipliers, state, iteration - 1, True, "residual tolerance met"
        try:
            step = _newton_step(state)
        except _DegenerateDensity:
            if iteration == 1:
                raise ValueError(
                    "a constraint function is constant on the support"
                ) from None
            return (
                multipliers,
                state,
                iteration,
                False,
                "the density collapsed onto a point",
            )
        slope = -math.fsum(r * d for r, d in zip(state.residual, step, strict=True))
        if -slope <= _ROUNDING_DECREMENT * max(1.0, abs(state.value)):
            # The predicted decrease is below the resolution of the dual value:
            # take the full step (quadratic convergence region).
            multipliers = tuple(
                lam + d for lam, d in zip(multipliers, step, strict=True)
            )
            state = _dual(grid, multipliers, second=True)
            continue
        alpha = 1.0
        for _ in range(_MAX_BACKTRACKS):
            trial = tuple(
                lam + alpha * d for lam, d in zip(multipliers, step, strict=True)
            )
            trial_value = _dual(grid, trial, second=False).value
            if trial_value <= state.value + _ARMIJO * alpha * slope:
                break
            alpha *= 0.5
        else:
            return multipliers, state, iteration, False, "line search found no decrease"
        multipliers = trial
        state = _dual(grid, multipliers, second=True)
    converged = _relative_residual(state.residual, state.hessian) <= tolerance
    message = "residual tolerance met" if converged else "iteration limit reached"
    return multipliers, state, max_iterations, converged, message


def _bfgs(
    grid: _Grid, start: Sequence[float], *, tolerance: float, max_iterations: int
) -> tuple[tuple[float, ...], _DualState, int, bool, str]:
    """Minimize the dual by BFGS with its analytic gradient ``-r``.

    References:
        [EQ-DIST-MAXENT-001], [REF-NOCEDAL-WRIGHT-2006, Chapter 6].

    """
    if not start:
        state = _dual(grid, (), second=True)
        return (), state, 0, True, "no constraints"

    def value(point: Sequence[float]) -> float:
        """Return the dual value.

        References:
            [EQ-DIST-MAXENT-001]. No new equation.

        """
        return _dual(grid, point, second=False).value

    def gradient(point: Sequence[float]) -> tuple[float, ...]:
        """Return the dual gradient ``h - E[g]``.

        References:
            [EQ-DIST-MAXENT-001]. No new equation.

        """
        return tuple(-r for r in _dual(grid, point, second=False).residual)

    result = minimize_bfgs(
        value,
        tuple(start),
        gradient=gradient,
        gradient_tolerance=tolerance * tolerance,
        max_iterations=max_iterations,
    )
    multipliers = tuple(result.x)
    state = _dual(grid, multipliers, second=True)
    converged = _relative_residual(state.residual, state.hessian) <= max(
        tolerance, 1e-8
    )
    return multipliers, state, result.iterations, converged, result.message


def _standardized_moments(
    raw: Sequence[float], centre: float, scale: float
) -> list[float]:
    """Return ``E[((X - c)/s)^k]`` for ``k = 0..K`` from raw moments ``m_0..m_K``.

    References:
        [EQ-DIST-MAXENT-002]. Binomial expansion; no new equation.

    """
    moments: list[float] = []
    for order in range(len(raw)):
        total = math.fsum(
            math.comb(order, j) * raw[j] * (-centre) ** (order - j)
            for j in range(order + 1)
        )
        moments.append(total / scale**order)
    return moments


def _positive_definite(matrix: Sequence[Sequence[float]]) -> bool:
    """Return whether the smallest eigenvalue is clearly positive.

    References:
        [EQ-DIST-MAXENT-002], [REF-SCHMUDGEN-2017].

    """
    if not matrix:
        return True
    eigenvalues, _ = symmetric_eigen(tuple(tuple(row) for row in matrix))
    largest = max(abs(value) for value in eigenvalues)
    return eigenvalues[0] > _FEASIBILITY_TOLERANCE * max(largest, 1.0)


def _check_moment_space(raw: Mapping[int, float], lower: float, upper: float) -> None:
    """Reject raw moments ``1..K`` outside the interior of the moment space.

    Raises:
        ValueError: If a Hankel or localizing matrix is not positive definite.

    References:
        [EQ-DIST-MAXENT-002], [REF-SCHMUDGEN-2017],
        [REF-KREIN-NUDELMAN-1977].

    """
    order = 0
    while order + 1 in raw:
        order += 1
    if order == 0:
        return
    moments = [1.0] + [raw[k] for k in range(1, order + 1)]
    centre = moments[1]
    spread = moments[2] - centre * centre if order >= 2 else 0.0
    if order >= 2 and not spread > 0.0:
        raise ValueError("the variance implied by the moments must be positive")
    if spread > 0.0:
        scale = math.sqrt(spread)
    elif math.isfinite(lower) and math.isfinite(upper):
        scale = upper - lower
    else:
        scale = max(1.0, abs(centre))
    mu = _standardized_moments(moments, centre, scale)
    a = (lower - centre) / scale
    b = (upper - centre) / scale
    half = order // 2
    matrices = [[[mu[i + j] for j in range(half + 1)] for i in range(half + 1)]]
    if math.isfinite(lower) and math.isfinite(upper):
        if order % 2 == 0:
            size = half
            matrices.append(
                [
                    [
                        (a + b) * mu[i + j + 1] - mu[i + j + 2] - a * b * mu[i + j]
                        for j in range(size)
                    ]
                    for i in range(size)
                ]
            )
        else:
            size = half + 1
            matrices.append(
                [
                    [mu[i + j + 1] - a * mu[i + j] for j in range(size)]
                    for i in range(size)
                ]
            )
            matrices.append(
                [
                    [b * mu[i + j] - mu[i + j + 1] for j in range(size)]
                    for i in range(size)
                ]
            )
    elif math.isfinite(lower):
        size = (order + 1) // 2
        matrices.append(
            [[mu[i + j + 1] - a * mu[i + j] for j in range(size)] for i in range(size)]
        )
    elif math.isfinite(upper):
        size = (order + 1) // 2
        matrices.append(
            [[b * mu[i + j] - mu[i + j + 1] for j in range(size)] for i in range(size)]
        )
    for matrix in matrices:
        if not _positive_definite(matrix):
            raise ValueError(
                f"the raw moments of orders 1..{order} lie outside the interior "
                "of the moment space of densities on the support (a Hankel or "
                "localizing matrix is not positive definite); no density "
                "satisfies them"
            )


def _raw_moment_targets(
    lower: float,
    upper: float,
    mean: float | None,
    variance: float | None,
    skewness: float | None,
    excess_kurtosis: float | None,
    raw_moments: Mapping[int, float] | None,
) -> tuple[dict[int, float], float | None]:
    """Merge standardized and raw moment inputs into raw moments ``E[X^k]``.

    Returns:
        ``(raw, central4)``: raw moments by order, and the target of
        ``E[(X - mean)^4]`` when a kurtosis is given without a skewness
        (the third moment is then free), else ``None``.

    Raises:
        ValueError: For an incomplete or contradictory specification, a mean
            outside the support, a non-positive variance, a variance at or
            above the Bhatia-Davis bound, or a kurtosis at or below Pearson's
            bound.

    References:
        [EQ-DIST-MAXENT-002], [REF-WILKINS-1944], [REF-BHATIA-DAVIS-2000].

    """
    raw: dict[int, float] = {}
    central4: float | None = None
    if mean is not None:
        mu = require_finite_real(mean, name="mean")
        if not lower < mu < upper:
            raise ValueError("mean must lie strictly inside the support")
        raw[1] = mu
    if variance is not None:
        if mean is None:
            raise ValueError("variance requires mean")
        sigma2 = require_finite_real(variance, name="variance")
        if not sigma2 > 0.0:
            raise ValueError("variance must be positive")
        if math.isfinite(lower) and math.isfinite(upper):
            bound = (raw[1] - lower) * (upper - raw[1])
            if not sigma2 < bound:
                raise ValueError(
                    f"variance must be below (mean - lower)(upper - mean) = {bound!r} "
                    "(Bhatia-Davis bound)"
                )
        raw[2] = sigma2 + raw[1] ** 2
    if (skewness is not None or excess_kurtosis is not None) and variance is None:
        raise ValueError("skewness and excess_kurtosis require mean and variance")
    gamma = 0.0
    if skewness is not None:
        gamma = require_finite_real(skewness, name="skewness")
        mu, sigma = raw[1], math.sqrt(raw[2] - raw[1] ** 2)
        raw[3] = gamma * sigma**3 + 3.0 * mu * sigma**2 + mu**3
    if excess_kurtosis is not None:
        kappa = require_finite_real(excess_kurtosis, name="excess_kurtosis") + 3.0
        if not kappa > gamma * gamma + 1.0:
            raise ValueError(
                f"kurtosis E[Z^4] = {kappa!r} must exceed skewness^2 + 1 = "
                f"{gamma * gamma + 1.0!r} (Pearson's inequality)"
            )
        mu, sigma = raw[1], math.sqrt(raw[2] - raw[1] ** 2)
        if skewness is None:
            central4 = kappa * sigma**4
        else:
            third = gamma * sigma**3
            raw[4] = (
                kappa * sigma**4 + 4.0 * mu * third + 6.0 * mu**2 * sigma**2 + mu**4
            )
    for order, value in (raw_moments or {}).items():
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("raw moment orders must be positive integers")
        checked = require_finite_real(value, name=f"raw_moments[{order}]")
        if order in raw and abs(raw[order] - checked) > 1e-12 * max(1.0, abs(checked)):
            raise ValueError(f"the moment of order {order} is specified twice")
        raw[order] = checked
    if central4 is not None and 4 in raw:
        raise ValueError(
            "excess_kurtosis without skewness conflicts with raw_moments[4]"
        )
    return raw, central4


def _power(order: int) -> Feature:
    """Return the monomial ``x -> x^order``.

    References:
        [EQ-DIST-MAXENT-001]. No equation is evaluated.

    """
    return lambda x: x**order


def _centred_feature(feature: Feature, target: float) -> Feature:
    """Return ``x -> g(x) - h``.

    References:
        [EQ-DIST-MAXENT-001]. No equation is evaluated.

    """
    return lambda x: feature(x) - target


def _centred_power(order: int, centre: float, scale: float) -> Feature:
    """Return the centred monomial ``x -> ((x - centre) / scale)^order``.

    References:
        [EQ-DIST-MAXENT-002]. No equation is evaluated.

    """
    return lambda x: ((x - centre) / scale) ** order


def _initial_multipliers(
    kind: str,
    lower: float,
    upper: float,
    raw: Mapping[int, float],
    basis: _Basis,
    size: int,
) -> list[float] | None:
    """Return decaying starting multipliers, or ``None`` when none is known.

    Bounded supports start from the uniform law (or a normal shape when the
    variance is below that of the uniform law); a half-line with a mean
    starts from the exponential law; the real line with a mean and a
    variance starts from the normal law. Multipliers are in the internal
    basis (centred monomials when ``basis.order > 0``).

    References:
        [EQ-DIST-MAXENT-001], [REF-COVER-THOMAS-2006, Chapter 12].

    """
    start = [0.0] * size
    if basis.order == 0:
        return start if kind == "bounded" else None
    has_normal = basis.order >= 2
    sigma2 = raw[2] - raw[1] ** 2 if has_normal else 0.0
    if kind == "bounded":
        if has_normal and sigma2 < (upper - lower) ** 2 / 12.0:
            start[1] = 0.5 * basis.scale**2 / sigma2
        return start
    if kind == "real":
        if not has_normal:
            return None
        start[1] = 0.5 * basis.scale**2 / sigma2
        return start
    gap = raw[1] - lower if kind == "lower" else upper - raw[1]
    start[0] = (1.0 if kind == "lower" else -1.0) * basis.scale / gap
    return start


@dataclass(frozen=True, slots=True)
class _Problem:
    """Features, targets, and names of one maximum-entropy problem.

    References:
        [EQ-DIST-MAXENT-001]. No equation is evaluated.

    """

    features: tuple[Feature, ...]
    targets: tuple[float, ...]
    names: tuple[str, ...]
    reported_targets: tuple[float, ...]
    basis: _Basis


def maximum_entropy(
    lower: float = -math.inf,
    upper: float = math.inf,
    *,
    mean: float | None = None,
    variance: float | None = None,
    skewness: float | None = None,
    excess_kurtosis: float | None = None,
    raw_moments: Mapping[int, float] | None = None,
    constraints: Sequence[tuple[Callable[[float], float], float]] = (),
    constraint_names: Sequence[str] | None = None,
    initial: Sequence[float] | None = None,
    location: float | None = None,
    scale: float | None = None,
    method: str = "newton",
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> MaximumEntropy:
    """Return the maximum-entropy law on ``[lower, upper]`` under the constraints.

    The density is ``f(x) = 1_K(x) exp(-lambda_0 - sum_m lambda_m g_m(x))``;
    the multipliers minimize the convex dual
    ``Gamma(lambda) = log int_K exp(-sum_m lambda_m (g_m - h_m)) dx``, which
    is the course form ``<Lambda, H> + int_K exp(-<Lambda, G>)`` after the
    normalizing multiplier ``lambda_0`` is minimized out.

    Moment constraints become monomial features ``g_k(x) = x^k`` with raw
    targets: ``mean``, ``variance``, ``skewness``, and ``excess_kurtosis``
    (standardized, so the course's ``E[((X - mu)/sigma)^4] = 10`` is
    ``excess_kurtosis=7``) are converted to ``E[X^k]``; ``raw_moments`` gives
    ``E[X^k]`` directly. An excess kurtosis without a skewness constrains
    ``E[(X - mean)^4]`` and leaves the third moment free. Further constraints
    ``E[g(X)] = h`` are ``(g, h)`` pairs; their multipliers follow the moment
    multipliers in the order given.

    Closed-form special cases: no constraint on ``[a, b]`` gives the uniform
    law, a mean on ``[a, inf)`` the shifted exponential law, a mean and a
    variance on the real line the normal law, and a mean on ``[a, b]`` a
    truncated exponential law.

    Args:
        lower: Lower end of the support (``-inf`` allowed).
        upper: Upper end of the support (``inf`` allowed), above ``lower``.
        mean: Prescribed ``E[X]``.
        variance: Prescribed ``Var[X]`` (requires ``mean``).
        skewness: Prescribed ``E[(X - mu)^3] / sigma^3`` (requires
            ``variance``).
        excess_kurtosis: Prescribed ``E[(X - mu)^4] / sigma^4 - 3``
            (requires ``variance``).
        raw_moments: Mapping ``k -> E[X^k]`` for positive integers ``k``.
        constraints: Pairs ``(g, h)`` meaning ``E[g(X)] = h``.
        constraint_names: Names of the ``constraints`` pairs (default
            ``"E[g_1(X)]"``, ...).
        initial: Starting multipliers, one per constraint in the reported
            order. Required on an unbounded support unless a mean (half-line)
            or a mean and a variance (real line) give a decaying start.
        location: Centre of the real-line map (default: the mean, else 0).
        scale: Scale of the unbounded-support map (default: the standard
            deviation, the distance of the mean from the finite end, or 1).
        method: ``"newton"`` (default) or ``"bfgs"``
            (:func:`probstat.optimization.minimize_bfgs`).
        tolerance: Newton stops when ``max_m |E[g_m] - h_m| / sd(g_m)`` is
            at most this value.
        max_iterations: Iteration limit per quadrature grid.

    Returns:
        :class:`MaximumEntropy` with ``pdf``, ``cdf``, ``quantile``, moments,
        ``entropy``, the multipliers, the constraint residuals, and the
        convergence record ``diagnostics``.

    Raises:
        ValueError: For an invalid support or input, moments outside the
            interior of the moment space (including ``kurtosis <= skewness^2
            + 1``), an unbounded support whose problem has no maximizer (no
            constraint, a top odd moment on the real line that the
            lower-order solution does not satisfy, or an exponent that does
            not decay), or a solve that does not converge.

    Equation:
        [EQ-DIST-MAXENT-001], [EQ-DIST-MAXENT-002].

    References:
        [REF-JAYNES-1957], [REF-MEAD-PAPANICOLAOU-1984],
        [REF-COVER-THOMAS-2006, Chapter 12], [REF-JUNK-2000].

    """
    a = float(lower)
    b = float(upper)
    if math.isnan(a) or math.isnan(b) or not a < b or a == math.inf or b == -math.inf:
        raise ValueError("the support must satisfy lower < upper")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    if not (tolerance > 0.0 and math.isfinite(tolerance)):
        raise ValueError("tolerance must be positive")
    iterations_limit = require_non_negative_integer(
        max_iterations, name="max_iterations"
    )
    raw, central4 = _raw_moment_targets(
        a, b, mean, variance, skewness, excess_kurtosis, raw_moments
    )
    _check_moment_space(raw, a, b)
    kind = (
        "bounded"
        if math.isfinite(a) and math.isfinite(b)
        else "lower"
        if math.isfinite(a)
        else "upper"
        if math.isfinite(b)
        else "real"
    )
    orders = sorted(raw)
    contiguous = bool(orders) and orders == list(range(1, len(orders) + 1))
    custom = tuple(constraints)
    if constraint_names is not None and len(constraint_names) != len(custom):
        raise ValueError("constraint_names must name every constraint pair")
    if kind != "bounded" and not orders and central4 is None and not custom:
        raise ValueError(
            "the entropy is unbounded on an unbounded support without constraints"
        )
    if kind == "real" and orders and central4 is None and not custom and orders[-1] % 2:
        return _odd_top_moment(
            a, b, orders, raw, location, scale, method, tolerance, iterations_limit
        )
    if contiguous:
        centre = raw[1]
        if len(orders) >= 2:
            spread = math.sqrt(raw[2] - raw[1] ** 2)
        elif kind == "bounded":
            spread = b - a
        else:
            spread = abs(raw[1] - (a if kind == "lower" else b))
        basis = _Basis(len(orders), centre, spread)
        moments = _standardized_moments(
            [1.0] + [raw[k] for k in orders], centre, spread
        )
        features: list[Feature] = [_centred_power(k, centre, spread) for k in orders]
        targets: list[float] = moments[1:]
    else:
        basis = _Basis(0, 0.0, 1.0)
        features = [_power(order) for order in orders]
        targets = [raw[order] for order in orders]
    reported = [raw[order] for order in orders]
    names = ["E[X]" if order == 1 else f"E[X^{order}]" for order in orders]
    if central4 is not None:
        centre_value = raw[1]
        features.append(lambda x: (x - centre_value) ** 4)
        targets.append(central4)
        reported.append(central4)
        names.append("E[(X - mean)^4]")
    for index, pair in enumerate(custom):
        function, target = pair
        if not callable(function):
            raise TypeError("each constraint must be a (callable, value) pair")
        value = require_finite_real(target, name=f"constraints[{index}] value")
        features.append(function)
        targets.append(value)
        reported.append(value)
        names.append(
            constraint_names[index]
            if constraint_names is not None
            else f"E[g_{index + 1}(X)]"
        )
    if initial is not None:
        given = tuple(require_finite_real(value, name="initial") for value in initial)
        if len(given) != len(features):
            raise ValueError("initial must give one multiplier per constraint")
        start = list(basis.to_internal(given))
    else:
        guess = _initial_multipliers(kind, a, b, raw, basis, len(features))
        if guess is None:
            raise ValueError(
                "an unbounded support needs decaying initial multipliers: pass "
                "initial=..., or a mean (half-line) or a mean and a variance "
                "(real line)"
            )
        start = guess
    sigma = math.sqrt(raw[2] - raw[1] ** 2) if 1 in raw and 2 in raw else None
    if scale is not None:
        map_scale = require_finite_real(scale, name="scale")
        if not map_scale > 0.0:
            raise ValueError("scale must be positive")
    elif sigma is not None:
        map_scale = sigma
    elif 1 in raw and kind == "lower":
        map_scale = raw[1] - a
    elif 1 in raw and kind == "upper":
        map_scale = b - raw[1]
    else:
        map_scale = 1.0
    map_centre = (
        require_finite_real(location, name="location")
        if location is not None
        else raw.get(1, 0.0)
    )
    problem = _Problem(
        tuple(features), tuple(targets), tuple(names), tuple(reported), basis
    )
    mapping = _Map(kind, a, b, map_centre, map_scale)
    return _solve(mapping, problem, start, method, tolerance, iterations_limit)


def _odd_top_moment(
    a: float,
    b: float,
    orders: Sequence[int],
    raw: Mapping[int, float],
    location: float | None,
    scale: float | None,
    method: str,
    tolerance: float,
    max_iterations: int,
) -> MaximumEntropy:
    """Handle a top odd moment on the real line.

    A non-zero multiplier of an odd top power cannot decay at both ends, so a
    maximizer exists only with that multiplier zero, that is, when the
    lower-order maximizer already has the prescribed top moment.

    Raises:
        ValueError: If no lower-order problem exists or its solution misses
            the top moment.

    References:
        [EQ-DIST-MAXENT-001], [REF-JUNK-2000].

    """
    top = orders[-1]
    reduced = {order: raw[order] for order in orders[:-1]}
    if not reduced:
        raise ValueError(
            "on the real line the highest constrained moment must have even "
            "order; no maximum-entropy density exists"
        )
    base = maximum_entropy(
        a,
        b,
        raw_moments=reduced,
        location=location,
        scale=scale,
        method=method,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    centre = base.mean
    spread = math.sqrt(base.variance)
    achieved = base.expect(lambda x: x**top)
    standardized = base.expect(lambda x: ((x - centre) / spread) ** top)
    if all(k in raw for k in range(1, top + 1)):
        wanted = _standardized_moments(
            [1.0] + [raw[k] for k in range(1, top + 1)], centre, spread
        )[top]
        gap = abs(standardized - wanted)
        limit = math.sqrt(base.expect(lambda x: ((x - centre) / spread) ** (2 * top)))
    else:
        gap = abs(achieved - raw[top])
        limit = math.sqrt(max(base.expect(lambda x: x ** (2 * top)), 1e-300))
    if gap > _CHECK_TOLERANCE * limit:
        raise ValueError(
            f"on the real line a non-zero multiplier of x^{top} cannot decay at "
            f"both ends; the maximum entropy is not attained (the lower-order "
            f"maximizer has E[X^{top}] = {achieved!r}, not {raw[top]!r}); "
            "use a bounded support or an even-order constraint"
        )
    name = "E[X]" if top == 1 else f"E[X^{top}]"
    return replace(
        base,
        multipliers=base.multipliers + (0.0,),
        constraint_names=base.constraint_names + (name,),
        targets=base.targets + (raw[top],),
        residuals=base.residuals + (achieved - raw[top],),
    )


def _decays_beyond_grid(
    mapping: _Map, grid: _Grid, problem: _Problem, multipliers: Sequence[float]
) -> bool:
    """Check that the exponent keeps growing beyond the outermost nodes.

    On each unbounded side the exponent ``sum lambda (g - h)`` is evaluated at
    ``x_k = x_n + (x_n - c)(2^k - 1)``, ``k = 0, 1, ...``, from the outermost
    node ``x_n`` out to ``|x| = 1e60``; it must be non-decreasing and at least
    ``40`` above its minimum over the nodes, so the density beyond the grid
    is below ``e^-40`` of its mode and decreasing. A multiplier that makes a
    leading power grow (for example a negative coefficient of ``x^4`` on the
    real line) fails this test however small it is.

    References:
        [EQ-DIST-MAXENT-001], [REF-JUNK-2000].

    """
    if mapping.kind == "bounded" or not multipliers:
        return True
    lambdas = tuple(multipliers)

    def exponent(x: float) -> float:
        """Return ``sum lambda (g(x) - h)``.

        References:
            [EQ-DIST-MAXENT-001]. No new equation.

        """
        return math.fsum(
            lam * (feature(x) - target)
            for lam, feature, target in zip(
                lambdas, problem.features, problem.targets, strict=True
            )
        )

    minimum = min(
        math.fsum(lam * value for lam, value in zip(lambdas, row, strict=True))
        for row in grid.centred
    )
    ends: list[float] = []
    if mapping.kind in ("upper", "real"):
        ends.append(min(grid.points))
    if mapping.kind in ("lower", "real"):
        ends.append(max(grid.points))
    centre = (
        mapping.location
        if mapping.kind == "real"
        else (mapping.lower if mapping.kind == "lower" else mapping.upper)
    )
    for end in ends:
        offset = end - centre
        previous = exponent(end)
        if previous < minimum + _DECAY_MARGIN:
            return False
        factor = 2.0
        while abs(centre + offset * factor) < 1e60:
            point = centre + offset * factor
            try:
                value = exponent(point)
            except (OverflowError, ValueError, ZeroDivisionError):
                break
            if math.isnan(value) or value < previous:
                return False
            if value == math.inf:
                break
            previous = value
            factor *= 2.0
    return True


def _solve(
    mapping: _Map,
    problem: _Problem,
    start: Sequence[float],
    method: str,
    tolerance: float,
    max_iterations: int,
) -> MaximumEntropy:
    """Run the dual solve with grid refinement and the adaptive check.

    Raises:
        ValueError: If the exponent does not decay on an unbounded support or
            the solve does not converge.

    References:
        [EQ-DIST-MAXENT-001], [REF-MEAD-PAPANICOLAOU-1984],
        [REF-PIESSENS-QUADPACK-1983, Section 2.3].

    """
    solver = _newton if method == "newton" else _bfgs
    panels = _INITIAL_PANELS
    multipliers = tuple(start)
    total_iterations = 0
    previous_tail = math.inf
    while True:
        grid = _build_grid(mapping, panels, problem.features, problem.targets)
        multipliers, state, iterations, converged, message = solver(
            grid, multipliers, tolerance=tolerance, max_iterations=max_iterations
        )
        total_iterations += iterations
        tail = max(
            (math.fsum(state.probabilities[k] for k in nodes) for nodes in grid.outer),
            default=0.0,
        )
        if tail > _CHECK_TOLERANCE:
            if panels >= _MAX_PANELS or tail > 0.1 * previous_tail:
                raise ValueError(
                    "the exponent does not decay on the unbounded support: a "
                    f"fraction {tail:.3g} of the mass sits in the outermost "
                    "quadrature panel and does not shrink under refinement, so "
                    "the maximum entropy is not attained for these constraints"
                )
            previous_tail = tail
            panels *= 2
            continue
        if converged and not _decays_beyond_grid(mapping, grid, problem, multipliers):
            raise ValueError(
                "the exponent does not decay on the unbounded support beyond the "
                "quadrature grid (the density would not be integrable), so the "
                "maximum entropy is not attained for these constraints"
            )
        if not converged:
            relative = (
                _relative_residual(state.residual, state.hessian)
                if state.hessian
                else math.inf
            )
            raise ValueError(
                f"the maximum-entropy solve did not converge ({message}; relative "
                f"residual {relative:.3g}); the constraints may be infeasible or "
                "on the boundary of the moment space"
            )
        law = _assemble(
            mapping,
            grid,
            problem,
            multipliers,
            state,
            (method, total_iterations, message, panels),
        )
        check = law.diagnostics.max_relative_residual
        if check <= max(_CHECK_TOLERANCE, 10.0 * tolerance):
            return law
        if panels >= _MAX_PANELS:
            warnings.warn(
                "the adaptive quadrature check did not confirm the fixed rule "
                f"(relative residual {check:.3g})",
                RuntimeWarning,
                stacklevel=3,
            )
            return replace(law, diagnostics=replace(law.diagnostics, converged=False))
        previous_tail = tail
        panels *= 2


def _assemble(
    mapping: _Map,
    grid: _Grid,
    problem: _Problem,
    multipliers: Sequence[float],
    state: _DualState,
    run: tuple[str, int, str, int],
) -> MaximumEntropy:
    """Normalize adaptively, compute residuals and moments, build the law.

    References:
        [EQ-DIST-MAXENT-001], [REF-PIESSENS-QUADPACK-1983, Section 2.3].

    """
    method, iterations, message, panels = run
    shift = state.shift
    lambdas = tuple(multipliers)
    features, targets = problem.features, problem.targets

    def unnormalized(t: float) -> float:
        """Return ``exp(shift - sum lambda (g - h)) phi'(t)``.

        References:
            [EQ-DIST-MAXENT-001]. No new equation.

        """
        x, jacobian = mapping.x_and_jacobian(t)
        if not (math.isfinite(x) and math.isfinite(jacobian)):
            return 0.0
        exponent = math.fsum(
            lam * (feature(x) - target)
            for lam, feature, target in zip(lambdas, features, targets, strict=True)
        )
        log_value = shift - exponent
        if log_value < -745.0:
            return 0.0
        return math.exp(min(log_value, 700.0)) * jacobian

    mass, mass_error, pieces = _panel_integral(unnormalized, grid.edges)
    if not (mass > 0.0 and math.isfinite(mass)):
        raise ValueError("the maximum-entropy density cannot be normalized")
    log_mass = math.log(mass)
    cumulative = [0.0]
    running: list[float] = []
    for piece in pieces:
        running.append(piece)
        cumulative.append(math.fsum(running) / mass)
    cumulative[-1] = 1.0
    upper_tail = [0.0]
    running = []
    for piece in reversed(pieces):
        running.append(piece)
        upper_tail.append(math.fsum(running) / mass)
    upper_tail.reverse()
    upper_tail[0] = 1.0

    def moment(function: Feature) -> float:
        """Return ``E[function(X)]`` under the normalized density.

        References:
            [EQ-DIST-MAXENT-001]. No new equation.

        """

        def integrand(t: float) -> float:
            """Return ``function(phi(t))`` times the normalized ``t`` density.

            References:
                [EQ-DIST-MAXENT-001]. No new equation.

            """
            weight = unnormalized(t)
            if weight == 0.0:
                return 0.0
            x, _ = mapping.x_and_jacobian(t)
            return function(x) * weight / mass

        return _panel_integral(integrand, grid.edges)[0]

    internal = tuple(
        moment(_centred_feature(feature, target))
        for feature, target in zip(features, targets, strict=True)
    )
    relative = 0.0
    for index, value in enumerate(internal):
        spread = (
            math.sqrt(max(state.hessian[index][index], 0.0)) if state.hessian else 0.0
        )
        relative = max(relative, abs(value) / spread if spread > 0.0 else abs(value))
    mean_value = moment(lambda x: x)
    variance_value = moment(lambda x: (x - mean_value) ** 2)
    reported = problem.basis.to_reported(lambdas)
    # lambda_0 of f = exp(-lambda_0 - sum lambda_j g_j) in the reported basis:
    # both exponents minus their target values agree, so the constants match.
    log_normalizer = (
        log_mass
        - shift
        - math.fsum(
            lam * target
            for lam, target in zip(reported, problem.reported_targets, strict=True)
        )
    )
    diagnostics = MaximumEntropyDiagnostics(
        converged=True,
        method=method,
        iterations=iterations,
        dual_value=log_mass - shift,
        max_relative_residual=relative,
        panels=panels,
        nodes=len(grid.points),
        quadrature_error=mass_error / mass,
        message=message,
    )
    return MaximumEntropy(
        mapping.lower,
        mapping.upper,
        reported,
        log_normalizer,
        problem.names,
        problem.reported_targets,
        problem.basis.residuals_to_reported(internal),
        diagnostics,
        features,
        lambdas,
        targets,
        internal,
        mapping,
        grid.edges,
        tuple(cumulative),
        tuple(upper_tail),
        tuple(piece / mass for piece in pieces),
        shift,
        log_mass,
        mean_value,
        variance_value,
    )
