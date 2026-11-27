"""Triangular transport maps and SVGD for inverse-problem posteriors.

Experimental (added in 0.28).

A triangular (Knothe-Rosenblatt) map ``S: R^d -> R^d`` has components
``S_k(u_1, ..., u_k)`` that depend only on the first ``k`` inputs and are
strictly increasing in ``u_k``. It then pushes any density to the standard
normal reference and is inverted component by component (a sequence of
scalar monotone root finds). This module provides

* the monotone parametrization (:class:`TriangularTransportMap`),
* a fit from target **samples**, one independent optimization per component
  (:func:`fit_triangular_map_from_samples`), which yields a density estimate
  and conditional sampling (useful for likelihood-free posteriors, with the
  data components ordered first),
* a fit from an **unnormalized density**, by minimizing the Kullback-Leibler
  divergence on a reference quadrature
  (:func:`fit_triangular_map_from_density`), which yields approximate
  posterior samples and a variance diagnostic, and
* Stein variational gradient descent for prior-whitened inverse posteriors
  with a stated bandwidth rule (:func:`svgd_posterior`), plus the comparison
  of samples with a quadrature reference
  (:func:`compare_samples_to_reference`).

Monotone parametrization ([EQ-INV-TMAP-001])
    ``S_k(u) = a_k(u_<k) + c u_k + int_0^{u_k} exp(b_k(u_<k, t)) dt`` with
    polynomial ``a_k`` (total degree ``degree``) and ``b_k`` (total degree
    ``slope_degree``) in the inputs standardized by an affine pre-map. The
    derivative is ``dS_k/du_k = c + exp(b_k) >= c = min_slope > 0`` for every
    real coefficient vector, so each ``S_k`` is strictly increasing, the map
    is a bijection of ``R^d`` (the slope floor makes the range all of ``R``)
    and the Jacobian is triangular with a positive diagonal. Monotonicity is
    therefore a **guarantee by construction**, not a constraint checked on
    data. The integral is evaluated by Gauss-Legendre quadrature; the
    density formulas use the exact derivative ``c + exp(b_k)``, so the
    density and the map agree to the quadrature error (about machine
    precision for ``slope_degree = 0``, where the integrand is constant, and
    small for moderate slopes; raise ``quadrature_order`` for strongly
    varying ``b_k``). The linear case ``degree = 1``, ``slope_degree = 0`` is
    the exact family of Gaussian triangular (Cholesky) maps.

Tail extrapolation
    A polynomial ``b_k`` extrapolates badly outside the data. The sample fit
    therefore clamps the last argument of ``b_k`` to the range of the
    training coordinate (``slope_lower``, ``slope_upper`` of the map): beyond
    the data the log slope is frozen, i.e. the map continues linearly with
    the boundary slope, which is Gaussian-like tail behavior. Without it a
    quartic ``b_k`` whose slope decays would send reference draws in the tails
    to very large ``x`` through the slope floor. The clamp makes the slope
    continuous but not smooth at the boundary and adds a small quadrature
    error there. The density fit does not clamp (its reference rule already
    spans the tails it is trained on).

Convexity, honestly
    Each component of the sample fit is a separate optimization. In ``a_k``
    the objective is a strictly convex quadratic; in the slope coefficients
    the integrated-exponential parametrization is **not** guaranteed to be
    jointly convex (that convexity belongs to linear-in-coefficients
    parametrizations that do not guarantee monotonicity). The solver is a
    damped Newton (Levenberg-Marquardt, exact Hessian) iteration that decreases the
    objective and reports ``converged`` and ``stop_reason`` per component;
    a stationary point, not a certified global minimum, is returned.

Sample fit ([EQ-INV-TMAP-002], [EQ-INV-TMAP-003])
    For samples ``x_n`` and the reference ``N(0, I)`` the maximum-likelihood
    objective splits into ``d`` terms
    ``J_k = mean_n[ S_k(x_n)^2 / 2 - log dS_k/dx_k(x_n) ]``. The fitted
    density is ``pi_hat(x) = N(S(x); 0, I) |det grad S(x)|``.

Density fit ([EQ-INV-TMAP-004], [EQ-INV-TMAP-005])
    For an unnormalized target ``pi~ = Z pi`` the map ``T`` from the
    reference minimizes ``KL(T#rho || pi) = E_rho[-log pi~(T(z)) -
    log |det grad T(z)|] + log Z`` over the coefficients, with the
    expectation over a fixed Gauss-Hermite tensor rule or fixed Monte Carlo
    reference points. With ``w(z) = log pi~(T(z)) + log |det grad T(z)| -
    log rho(z)`` one has ``E_rho[w] = log Z - KL`` and ``w`` is constant
    exactly when the map is exact: the variance of ``w`` is the diagnostic,
    ``KL ~ Var(w) / 2`` for small variance, and the self-normalized weights
    ``exp(w)`` correct the samples by importance sampling.

SVGD ([EQ-INV-TMAP-006])
    The particle update of :mod:`probstat.inference.particle_vi` applied to
    the prior-whitened log posterior; the kernel is the radial basis
    function with the median bandwidth rule ``h = med^2 / log n`` (or a fixed
    ``h``). SVGD particles are not independent draws and typically
    underestimate the variance in more than a few dimensions.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
)
from probstat.inference import stein_variational_gradient_descent
from probstat.linalg import Matrix, Vector, as_vector, solve_spd
from probstat.modeling import gauss_hermite
from probstat.optimization import minimize_bfgs
from probstat.processes import gauss_legendre

from .posterior_sampling import QuadratureReference

LogDensity = Callable[[Vector], float]
GradientFunction = Callable[[Vector], Sequence[float]]

_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_MAX_REFERENCE_POINTS = 50_000


# ---------------------------------------------------------------------------
# Basis helpers
# ---------------------------------------------------------------------------


@cache
def _exponents(variables: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """Return monomial exponents of total degree at most ``degree``.

    The zero exponent (constant) comes first, then by total degree.

    Args:
        variables: Number of variables.
        degree: Maximum total degree.

    Returns:
        Exponent tuples.

    References:
        [EQ-INV-TMAP-001]. Combinatorial enumeration only.

    """
    found = [
        e
        for e in itertools.product(range(degree + 1), repeat=variables)
        if sum(e) <= degree
    ]
    found.sort(key=lambda e: (sum(e), tuple(-v for v in e)))
    return tuple(found)


def _basis(
    exponents: tuple[tuple[int, ...], ...], point: Sequence[float], degree: int
) -> list[float]:
    """Evaluate monomials at a point.

    Args:
        exponents: Exponent tuples.
        point: Variable values.
        degree: Largest exponent used.

    Returns:
        Monomial values in the order of ``exponents``.

    References:
        [EQ-INV-TMAP-001]. Polynomial evaluation only.

    """
    powers = []
    for value in point:
        row = [1.0]
        for _ in range(degree):
            row.append(row[-1] * value)
        powers.append(row)
    out = []
    for exponent in exponents:
        term = 1.0
        for index, power in enumerate(exponent):
            if power:
                term *= powers[index][power]
        out.append(term)
    return out


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the Euclidean inner product.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        ``sum_i left_i right_i``.

    References:
        [EQ-INV-TMAP-001]. Plain arithmetic.

    """
    total = 0.0
    for a, b in zip(left, right, strict=True):
        total += a * b
    return total


@cache
def _legendre(order: int) -> tuple[Vector, Vector]:
    """Return cached Gauss-Legendre nodes and weights on ``[-1, 1]``.

    Args:
        order: Number of nodes.

    Returns:
        Nodes and weights.

    References:
        [EQ-INV-TMAP-001], [REF-DAVIS-RABINOWITZ-1984].

    """
    return gauss_legendre(order)


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    """Validate an integer at least ``minimum``.

    Args:
        value: Candidate.
        name: Argument name for messages.
        minimum: Smallest allowed value.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If it is below ``minimum``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    count = require_non_negative_integer(value, name=name)
    if count < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return count


# ---------------------------------------------------------------------------
# The monotone triangular map
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriangularTransportMap:
    """Monotone triangular map with integrated-positive components.

    Input ``p`` is standardized, ``u_i = (p_i - input_center_i) /
    input_scale_i``; component ``k`` is
    ``s_k = a_k(u_<k) + min_slope u_k + int_0^{u_k} exp(b_k(u_<k, t)) dt``
    with polynomial ``a_k`` and ``b_k`` and the output is
    ``v_k = output_center_k + output_scale_k s_k``
    ([EQ-INV-TMAP-001]). The map is strictly increasing in ``p_k`` for every
    coefficient vector, so it is a bijection of ``R^d``.

    Attributes:
        dimension: ``d``.
        degrees: Per component, total degree of the polynomial ``a_k``.
        slope_degrees: Per component, total degree of the polynomial ``b_k``.
        quadrature_order: Gauss-Legendre nodes for the integral.
        min_slope: Slope floor ``c > 0`` in standardized units.
        coefficients: Per component ``(alpha_k, beta_k)``: coefficients of
            ``a_k`` and ``b_k`` in the monomial order of the module
            (constant first).
        input_center: Centers of the input affine pre-map.
        input_scale: Positive scales of the input pre-map.
        output_center: Centers of the output affine post-map.
        output_scale: Positive scales of the output post-map.
        slope_lower: Optional per-component lower clamp (standardized units)
            of the last argument of ``b_k``; with ``slope_upper`` it freezes
            the log slope outside the range, so extrapolation is linear with
            the boundary slope instead of following the polynomial.
        slope_upper: Optional per-component upper clamp.

    References:
        [EQ-INV-TMAP-001], [REF-MARZOUK-ETAL-2017], [REF-ROSENBLATT-1952].

    """

    dimension: int
    degrees: tuple[int, ...]
    slope_degrees: tuple[int, ...]
    quadrature_order: int
    min_slope: float
    coefficients: tuple[tuple[Vector, Vector], ...]
    input_center: Vector
    input_scale: Vector
    output_center: Vector
    output_scale: Vector
    slope_lower: Vector | None = None
    slope_upper: Vector | None = None

    def __post_init__(self) -> None:
        """Validate shapes and positivity.

        Raises:
            ValueError: If sizes or signs are inconsistent.

        References:
            [EQ-INV-TMAP-001]. No equation is evaluated.

        """
        d = self.dimension
        if len(self.coefficients) != d:
            raise ValueError("one coefficient pair per component is required")
        for k, (alpha, beta) in enumerate(self.coefficients):
            if len(alpha) != len(_exponents(k, self.degrees[k])):
                raise ValueError(f"alpha[{k}] has the wrong length")
            if len(beta) != len(_exponents(k + 1, self.slope_degrees[k])):
                raise ValueError(f"beta[{k}] has the wrong length")
        for name in ("input_center", "input_scale", "output_center", "output_scale"):
            if len(getattr(self, name)) != d:
                raise ValueError(f"{name} must have length {d}")
        if not self.min_slope > 0.0:
            raise ValueError("min_slope must be positive")
        if (self.slope_lower is None) != (self.slope_upper is None):
            raise ValueError("slope_lower and slope_upper go together")
        if self.slope_lower is not None and self.slope_upper is not None:
            if len(self.slope_lower) != d or len(self.slope_upper) != d:
                raise ValueError(f"slope bounds must have length {d}")
            if any(
                lo >= hi
                for lo, hi in zip(self.slope_lower, self.slope_upper, strict=True)
            ):
                raise ValueError("slope_lower must be below slope_upper")
        if min(self.input_scale) <= 0.0 or min(self.output_scale) <= 0.0:
            raise ValueError("scales must be positive")

    def _bounds(self, k: int) -> tuple[float, float]:
        """Return the clamp interval of the slope polynomial argument.

        Args:
            k: Component index.

        Returns:
            ``(lower, upper)``, infinite when no bounds are stored.

        References:
            [EQ-INV-TMAP-001].

        """
        if self.slope_lower is None or self.slope_upper is None:
            return -math.inf, math.inf
        return self.slope_lower[k], self.slope_upper[k]

    def _component(
        self, k: int, prefix: Sequence[float], t: float
    ) -> tuple[float, float]:
        """Evaluate standardized component ``k`` and its slope.

        Args:
            k: Component index.
            prefix: Standardized inputs ``u_<k``.
            t: Standardized ``u_k``.

        Returns:
            ``(s_k, ds_k/du_k)``.

        Raises:
            ValueError: If an exponential overflows.

        References:
            [EQ-INV-TMAP-001].

        """
        alpha, beta = self.coefficients[k]
        phi = _basis(_exponents(k, self.degrees[k]), prefix, self.degrees[k])
        value = _dot(alpha, phi) + self.min_slope * t
        sdeg = self.slope_degrees[k]
        exps = _exponents(k + 1, sdeg)
        lo, hi = self._bounds(k)
        nodes, weights = _legendre(self.quadrature_order)
        try:
            for xi, w in zip(nodes, weights, strict=True):
                tq = min(max(0.5 * t * (1.0 + xi), lo), hi)
                psi = _basis(exps, (*prefix, tq), sdeg)
                value += 0.5 * t * w * math.exp(_dot(beta, psi))
            psi_end = _basis(exps, (*prefix, min(max(t, lo), hi)), sdeg)
            slope = self.min_slope + math.exp(_dot(beta, psi_end))
        except OverflowError as error:
            raise ValueError("transport map component overflowed") from error
        return value, slope

    def _standardize(self, point: Sequence[float]) -> Vector:
        """Apply the input affine pre-map.

        Args:
            point: Physical input.

        Returns:
            Standardized input.

        References:
            [EQ-INV-TMAP-001].

        """
        return tuple(
            (p - c) / s
            for p, c, s in zip(point, self.input_center, self.input_scale, strict=True)
        )

    def forward(self, point: Sequence[float]) -> Vector:
        """Evaluate the map.

        Args:
            point: Input of length ``dimension``.

        Returns:
            Output ``v`` of length ``dimension``.

        Raises:
            ValueError: For a wrong length, non-finite value or overflow.

        Equation:
            [EQ-INV-TMAP-001].

        References:
            [REF-MARZOUK-ETAL-2017].

        """
        p = as_vector(point, name="point")
        if len(p) != self.dimension:
            raise ValueError(f"point must have length {self.dimension}")
        u = self._standardize(p)
        out = []
        for k in range(self.dimension):
            s, _ = self._component(k, u[:k], u[k])
            out.append(self.output_center[k] + self.output_scale[k] * s)
        return tuple(out)

    def log_det_jacobian(self, point: Sequence[float]) -> float:
        """Return ``log |det grad map|`` at an input.

        Args:
            point: Input of length ``dimension``.

        Returns:
            ``sum_k [log output_scale_k - log input_scale_k +
            log(min_slope + exp b_k)]``.

        Raises:
            ValueError: For a wrong length, non-finite value or overflow.

        Equation:
            [EQ-INV-TMAP-003].

        References:
            [REF-MARZOUK-ETAL-2017].

        """
        p = as_vector(point, name="point")
        if len(p) != self.dimension:
            raise ValueError(f"point must have length {self.dimension}")
        u = self._standardize(p)
        total = 0.0
        for k in range(self.dimension):
            _, slope = self._component(k, u[:k], u[k])
            total += (
                math.log(self.output_scale[k])
                - math.log(self.input_scale[k])
                + math.log(slope)
            )
        return total

    def inverse(self, output: Sequence[float], *, head: Sequence[float] = ()) -> Vector:
        """Invert the map component by component.

        With ``head`` empty this is the full inverse. With a ``head`` of
        ``j`` leading *input* coordinates, ``output`` holds the output
        values of components ``j .. d-1`` and only the remaining input
        coordinates are solved (conditional sampling, [EQ-INV-TMAP-006]).

        Each scalar equation ``s_k(prefix, t) = target`` is solved by a
        safeguarded Newton iteration on a bracket; the bracket always
        exists because the slope is at least ``min_slope``.

        Args:
            output: Output values of the trailing components.
            head: Fixed leading input coordinates.

        Returns:
            Full input point (``head`` followed by the solved coordinates).

        Raises:
            ValueError: For wrong lengths, non-finite values, overflow or a
                root find that does not converge.

        Equation:
            [EQ-INV-TMAP-006].

        References:
            [REF-MARZOUK-ETAL-2017], [REF-ROSENBLATT-1952].

        """
        known = as_vector(head, name="head") if len(head) else ()
        target = as_vector(output, name="output") if len(output) else ()
        if len(known) + len(target) != self.dimension:
            raise ValueError("head and output lengths must add up to the dimension")
        j = len(known)
        u: list[float] = [
            (p - c) / s
            for p, c, s in zip(
                known, self.input_center[:j], self.input_scale[:j], strict=True
            )
        ]
        for k in range(j, self.dimension):
            goal = (target[k - j] - self.output_center[k]) / self.output_scale[k]
            u.append(self._solve(k, u, goal))
        return tuple(
            c + s * v
            for c, s, v in zip(self.input_center, self.input_scale, u, strict=True)
        )

    def _solve(self, k: int, prefix: Sequence[float], goal: float) -> float:
        """Solve ``s_k(prefix, t) = goal`` for ``t``.

        Args:
            k: Component index.
            prefix: Standardized inputs ``u_<k``.
            goal: Standardized target value.

        Returns:
            The root ``t``.

        Raises:
            ValueError: If no bracket or root is found.

        References:
            [EQ-INV-TMAP-006].

        """
        f0, _ = self._component(k, prefix, 0.0)
        f0 -= goal
        if f0 == 0.0:
            return 0.0
        sign = 1.0 if f0 < 0.0 else -1.0
        low, high = 0.0, sign
        for _ in range(200):
            f_high = self._component(k, prefix, high)[0] - goal
            if (f_high >= 0.0) == (sign > 0.0):
                break
            low, high = high, 2.0 * high
        else:
            raise ValueError("no bracket found for the monotone inverse")
        low, high = min(low, high), max(low, high)
        t = 0.5 * (low + high)
        for _ in range(300):
            value, slope = self._component(k, prefix, t)
            residual = value - goal
            if abs(residual) <= 1e-13 * (1.0 + abs(goal)):
                return t
            if residual > 0.0:
                high = t
            else:
                low = t
            candidate = t - residual / slope
            if not low < candidate < high:
                candidate = 0.5 * (low + high)
            if abs(candidate - t) <= 1e-15 * (1.0 + abs(t)):
                return candidate
            t = candidate
        raise ValueError("monotone inverse did not converge")


# ---------------------------------------------------------------------------
# Per-component tables and the shared objective pieces
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Table:
    """Precomputed basis values of one component at fixed standardized inputs.

    References:
        [EQ-INV-TMAP-002]. Internal container.

    """

    phi: list[list[float]]
    psi: list[list[list[float]]]
    weights: list[list[float]]
    psi_end: list[list[float]]
    t_end: list[float]


def _build_table(
    points: Sequence[Sequence[float]],
    k: int,
    degree: int,
    slope_degree: int,
    order: int,
    lo: float = -math.inf,
    hi: float = math.inf,
) -> _Table:
    """Precompute everything of component ``k`` that does not depend on coefficients.

    Args:
        points: Standardized input points.
        k: Component index.
        degree: Degree of ``a_k``.
        slope_degree: Degree of ``b_k``.
        order: Gauss-Legendre order.
        lo: Lower clamp of the slope argument.
        hi: Upper clamp of the slope argument.

    Returns:
        The table.

    References:
        [EQ-INV-TMAP-002].

    """
    exps_a = _exponents(k, degree)
    exps_b = _exponents(k + 1, slope_degree)
    nodes, gl_weights = _legendre(order)
    table = _Table([], [], [], [], [])
    for point in points:
        prefix = tuple(point[:k])
        t = point[k]
        table.phi.append(_basis(exps_a, prefix, degree))
        table.t_end.append(t)
        table.psi.append(
            [
                _basis(
                    exps_b,
                    (*prefix, min(max(0.5 * t * (1.0 + xi), lo), hi)),
                    slope_degree,
                )
                for xi in nodes
            ]
        )
        table.weights.append([0.5 * t * w for w in gl_weights])
        table.psi_end.append(
            _basis(exps_b, (*prefix, min(max(t, lo), hi)), slope_degree)
        )
    return table


def _row_terms(
    table: _Table, row: int, alpha: Vector, beta: Vector, floor: float
) -> tuple[float, Vector, float, float, list[float]]:
    """Evaluate one table row for given coefficients.

    Args:
        table: Component table.
        row: Row index.
        alpha: Coefficients of ``a_k``.
        beta: Coefficients of ``b_k``.
        floor: ``min_slope``.

    Returns:
        ``(s, dS/dbeta, slope, e_end, node_terms)`` with ``e_end = exp(b_k)``
        at the end and ``node_terms[q] = w_q exp(b_k(t_q))``.

    Raises:
        OverflowError: If an exponential overflows.

    References:
        [EQ-INV-TMAP-001].

    """
    s = _dot(alpha, table.phi[row]) + floor * table.t_end[row]
    grad_beta = [0.0] * len(beta)
    node_terms = []
    for w, psi in zip(table.weights[row], table.psi[row], strict=True):
        e = w * math.exp(_dot(beta, psi))
        node_terms.append(e)
        s += e
        for index, value in enumerate(psi):
            grad_beta[index] += e * value
    e_end = math.exp(_dot(beta, table.psi_end[row]))
    return s, tuple(grad_beta), floor + e_end, e_end, node_terms


def _initial_coefficients(
    k: int, degree: int, slope_degree: int, floor: float
) -> tuple[Vector, Vector]:
    """Return the identity-slope starting coefficients of component ``k``.

    Args:
        k: Component index.
        degree: Degree of ``a_k``.
        slope_degree: Degree of ``b_k``.
        floor: ``min_slope`` (must be below one).

    Returns:
        ``(alpha, beta)`` with unit slope and zero offset.

    References:
        [EQ-INV-TMAP-001].

    """
    alpha = (0.0,) * len(_exponents(k, degree))
    beta = [0.0] * len(_exponents(k + 1, slope_degree))
    beta[0] = math.log(1.0 - floor)
    return alpha, tuple(beta)


def _per_component(value: object, d: int, name: str) -> tuple[int, ...]:
    """Expand an integer or a sequence into one degree per component.

    Args:
        value: Integer or sequence of ``d`` integers.
        d: Dimension.
        name: Argument name for messages.

    Returns:
        Tuple of ``d`` degrees in ``[0, 8]``.

    Raises:
        TypeError: For non-integer entries.
        ValueError: For a wrong length or a degree above 8.

    References:
        [EQ-INV-TMAP-001]. Validation only.

    """
    if isinstance(value, Sequence) and not isinstance(value, str):
        items = tuple(value)
        if len(items) != d:
            raise ValueError(f"{name} must have one entry per component")
    else:
        items = (value,) * d
    out = tuple(require_non_negative_integer(v, name=name) for v in items)
    if max(out) > 8:
        raise ValueError("degrees above 8 are not supported")
    return out


def _check_options(
    degree: object,
    slope_degree: object,
    quadrature_order: object,
    min_slope: object,
    d: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int, float]:
    """Validate the parametrization options.

    Args:
        degree: Degree of ``a_k`` (integer or one per component).
        slope_degree: Degree of ``b_k`` (integer or one per component).
        quadrature_order: Gauss-Legendre order.
        min_slope: Slope floor in ``(0, 1)``.
        d: Dimension.

    Returns:
        Validated tuple.

    Raises:
        TypeError: For non-integer options.
        ValueError: For out-of-range options.

    References:
        [EQ-INV-TMAP-001]. Validation only.

    """
    a = _per_component(degree, d, "degree")
    b = _per_component(slope_degree, d, "slope_degree")
    q = _positive_int(quadrature_order, "quadrature_order", minimum=2)
    if q > 40:
        raise ValueError("quadrature_order must not exceed 40")
    c = require_finite_real(min_slope, name="min_slope")
    if not 0.0 < c < 1.0:
        raise ValueError("min_slope must lie in (0, 1)")
    return a, b, q, c


# ---------------------------------------------------------------------------
# Fit from samples
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SampleTransportFit:
    """Triangular map fitted to target samples, with density and sampling.

    ``transport_map`` maps target points ``x`` to the reference ``N(0, I)``
    (output center 0, scale 1). Component ``k`` depends on ``x_1..x_k``, so
    ordering the observed variables first gives conditional sampling
    ([EQ-INV-TMAP-006]).

    Attributes:
        transport_map: The fitted map ``S``.
        converged: Per component, whether the gradient tolerance was met.
        stop_reasons: Per component termination text.
        iterations: Per component iterations.
        objectives: Per component final ``J_k`` of [EQ-INV-TMAP-002]
            (reference constants omitted).
        gradient_norms: Per component final gradient norm.
        sample_count: Number of training samples.
        mean_log_density: Average of the fitted log density over the
            training samples (an in-sample figure; it favors overfitting).

    References:
        [EQ-INV-TMAP-002], [EQ-INV-TMAP-003], [REF-MARZOUK-ETAL-2017].

    """

    transport_map: TriangularTransportMap
    converged: tuple[bool, ...]
    stop_reasons: tuple[str, ...]
    iterations: tuple[int, ...]
    objectives: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    sample_count: int
    mean_log_density: float

    def log_density(self, point: Sequence[float]) -> float:
        """Return the fitted log density ``log N(S(x)) + log |det grad S|``.

        Args:
            point: Target point.

        Returns:
            Natural logarithm of the density estimate.

        Raises:
            ValueError: For invalid input or overflow.

        Equation:
            [EQ-INV-TMAP-003].

        References:
            [REF-MARZOUK-ETAL-2017].

        """
        z = self.transport_map.forward(point)
        return (
            -0.5 * _dot(z, z)
            - self.transport_map.dimension * _HALF_LOG_TWO_PI
            + self.transport_map.log_det_jacobian(point)
        )

    def sample(
        self, count: int, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw samples by inverting the map on reference draws.

        Args:
            count: Number of draws.
            seed: Integer seed or stream.

        Returns:
            Samples of the fitted density.

        Raises:
            ValueError: If ``count`` is invalid or an inversion fails.

        Equation:
            [EQ-INV-TMAP-003].

        References:
            [REF-MARZOUK-ETAL-2017].

        """
        n = _positive_int(count, "count")
        rng = resolve_stream(seed).generator()
        d = self.transport_map.dimension
        return tuple(
            self.transport_map.inverse(tuple(rng.gauss(0.0, 1.0) for _ in range(d)))
            for _ in range(n)
        )

    def conditional_sample(
        self,
        head: Sequence[float],
        count: int,
        *,
        seed: int | RandomStream | None = None,
    ) -> tuple[Vector, ...]:
        """Sample the trailing coordinates given the leading ones.

        For a triangular map the conditional of ``x_(j+1..d)`` given
        ``x_(1..j)`` is obtained by fixing the head and inverting the
        remaining components on fresh reference draws ([EQ-INV-TMAP-006]).
        Put the observations first and the parameters last to sample the
        posterior of a likelihood-free model from joint simulations.

        Args:
            head: Observed leading coordinates (fewer than ``d``).
            count: Number of draws.
            seed: Integer seed or stream.

        Returns:
            Draws of the trailing coordinates only.

        Raises:
            ValueError: If ``head`` is empty of meaning (length not in
                ``[0, d-1]``) or an inversion fails.

        Equation:
            [EQ-INV-TMAP-006].

        References:
            [REF-MARZOUK-ETAL-2017], [REF-ROSENBLATT-1952].

        """
        n = _positive_int(count, "count")
        d = self.transport_map.dimension
        known = as_vector(head, name="head")
        j = len(known)
        if not 0 <= j < d:
            raise ValueError("head must have between 0 and d - 1 entries")
        rng = resolve_stream(seed).generator()
        out = []
        for _ in range(n):
            z = tuple(rng.gauss(0.0, 1.0) for _ in range(d - j))
            out.append(self.transport_map.inverse(z, head=known)[j:])
        return tuple(out)


def _validated_samples(samples: Sequence[Sequence[float]]) -> tuple[Vector, ...]:
    """Validate a sample matrix.

    Args:
        samples: Rows of equal length.

    Returns:
        Tuple of vectors.

    Raises:
        ValueError: For fewer than two rows, ragged rows or non-finite values.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    rows = tuple(as_vector(row, name="sample") for row in samples)
    if len(rows) < 2:
        raise ValueError("at least two samples are required")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("samples must be non-empty rows of equal length")
    return rows


def _fit_component(
    table: _Table,
    alpha0: Vector,
    beta0: Vector,
    floor: float,
    ridge: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[Vector, Vector, float, float, int, bool, str]:
    """Minimize ``J_k`` by damped Newton (Levenberg-Marquardt).

    Args:
        table: Component table.
        alpha0: Starting ``alpha``.
        beta0: Starting ``beta``.
        floor: ``min_slope``.
        ridge: Penalty weight ``0.5 ridge |theta|^2`` added to ``J_k``.
        max_iterations: Iteration limit.
        tolerance: Gradient-norm tolerance.

    Returns:
        ``(alpha, beta, objective, gradient_norm, iterations, converged,
        reason)``.

    References:
        [EQ-INV-TMAP-002], [REF-NOCEDAL-WRIGHT-2006].

    """
    n_alpha = len(alpha0)
    size = n_alpha + len(beta0)
    rows = len(table.t_end)

    def evaluate(theta: Vector) -> tuple[float, Vector, Matrix]:
        """Return objective, gradient and exact Hessian at ``theta``.

        References:
            [EQ-INV-TMAP-002].

        """
        alpha, beta = theta[:n_alpha], theta[n_alpha:]
        value = 0.0
        grad = [0.0] * size
        hess = [[0.0] * size for _ in range(size)]
        for r in range(rows):
            s, g_beta, slope, e_end, nodes = _row_terms(table, r, alpha, beta, floor)
            value += 0.5 * s * s - math.log(slope)
            ratio = e_end / slope
            jac = [*table.phi[r], *g_beta]
            tail = [0.0] * n_alpha + [ratio * v for v in table.psi_end[r]]
            for i in range(size):
                grad[i] += s * jac[i] - tail[i]
                row_i = hess[i]
                ji = jac[i]
                for j in range(i + 1):
                    row_i[j] += ji * jac[j]
            # exact second-order terms of the beta block
            curvature = -floor * e_end / (slope * slope)
            psi_end = table.psi_end[r]
            for i in range(len(beta)):
                row_i = hess[n_alpha + i]
                for j in range(i + 1):
                    extra = curvature * psi_end[i] * psi_end[j]
                    for e, psi in zip(nodes, table.psi[r], strict=True):
                        extra += s * e * psi[i] * psi[j]
                    row_i[n_alpha + j] += extra
        scale = 1.0 / rows
        value *= scale
        grad_list = [g * scale for g in grad]
        for i in range(size):
            for j in range(i + 1):
                hess[i][j] *= scale
                hess[j][i] = hess[i][j]
        if ridge > 0.0:
            value += 0.5 * ridge * _dot(theta, theta)
            for i in range(size):
                grad_list[i] += ridge * theta[i]
                hess[i][i] += ridge
        return value, tuple(grad_list), tuple(tuple(row) for row in hess)

    theta = (*alpha0, *beta0)
    value, grad, hess = evaluate(theta)
    damping = 1e-6
    reason = "iteration limit"
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        norm = math.sqrt(_dot(grad, grad))
        if norm <= tolerance:
            reason = "gradient tolerance"
            iterations -= 1
            break
        accepted = False
        while damping <= 1e12:
            damped = tuple(
                tuple(
                    hess[i][j] + (damping * max(hess[i][i], 1e-8) if i == j else 0.0)
                    for j in range(size)
                )
                for i in range(size)
            )
            try:
                step = solve_spd(damped, tuple(-g for g in grad))
                trial = tuple(t + s for t, s in zip(theta, step, strict=True))
                new_value, new_grad, new_hess = evaluate(trial)
            except (ValueError, OverflowError):
                damping *= 8.0
                continue
            if new_value < value:
                theta, value, grad, hess = trial, new_value, new_grad, new_hess
                damping = max(damping / 4.0, 1e-12)
                accepted = True
                break
            damping *= 8.0
        if not accepted:
            reason = "damping exhausted"
            break
    norm = math.sqrt(_dot(grad, grad))
    return (
        theta[:n_alpha],
        theta[n_alpha:],
        value,
        norm,
        iterations,
        norm <= tolerance,
        reason,
    )


def fit_triangular_map_from_samples(
    samples: Sequence[Sequence[float]],
    *,
    degree: int | Sequence[int] = 3,
    slope_degree: int | Sequence[int] = 2,
    quadrature_order: int = 10,
    min_slope: float = 1e-3,
    ridge: float = 0.0,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-7,
) -> SampleTransportFit:
    """Fit a monotone triangular map from target samples to ``N(0, I)``.

    The samples are standardized per coordinate (an exactly invertible
    affine pre-map stored in the result) and each component ``k`` minimizes
    ``J_k = mean[S_k^2 / 2 - log(dS_k/dx_k)]`` ([EQ-INV-TMAP-002]) with the
    damped Newton iteration described in the module docstring.
    Components are independent problems, solved in order. Monotonicity holds
    by construction for any coefficients ([EQ-INV-TMAP-001]).

    Args:
        samples: Target samples, one row each (at least two).
        degree: Total degree of ``a_k`` (``1`` gives linear in the prefix);
            an integer or one entry per component.
        slope_degree: Total degree of ``b_k`` (``0`` gives a constant slope);
            an integer or one entry per component.
        quadrature_order: Gauss-Legendre nodes for the integral.
        min_slope: Slope floor in ``(0, 1)`` (standardized units).
        ridge: Non-negative ``L2`` penalty weight on the coefficients
            (default no penalty); documented regularization, not a hidden
            jitter.
        max_iterations: Iteration limit per component.
        gradient_tolerance: Stopping tolerance on the gradient norm.

    Returns:
        :class:`SampleTransportFit` with per-component status.

    Raises:
        TypeError: For non-integer options.
        ValueError: For invalid samples or options, or a coordinate with
            zero spread.

    Equation:
        [EQ-INV-TMAP-001], [EQ-INV-TMAP-002], [EQ-INV-TMAP-003].

    References:
        [REF-MARZOUK-ETAL-2017], [REF-EL-MOSELHY-MARZOUK-2012],
        [REF-NOCEDAL-WRIGHT-2006].

    """
    rows = _validated_samples(samples)
    d = len(rows[0])
    deg, sdeg, order, floor = _check_options(
        degree, slope_degree, quadrature_order, min_slope, d
    )
    reg = require_non_negative_finite_real(ridge, name="ridge")
    iters = _positive_int(max_iterations, "max_iterations")
    tol = require_finite_real(gradient_tolerance, name="gradient_tolerance")
    if tol <= 0.0:
        raise ValueError("gradient_tolerance must be positive")
    n = len(rows)
    center = tuple(math.fsum(r[i] for r in rows) / n for i in range(d))
    scale = tuple(
        math.sqrt(math.fsum((r[i] - center[i]) ** 2 for r in rows) / (n - 1))
        for i in range(d)
    )
    if min(scale) <= 0.0:
        raise ValueError("every coordinate must have positive sample spread")
    standardized = [
        tuple((r[i] - center[i]) / scale[i] for i in range(d)) for r in rows
    ]
    bounds = [
        (min(row[k] for row in standardized), max(row[k] for row in standardized))
        for k in range(d)
    ]
    coefficients = []
    flags, reasons, counts, objectives, norms = [], [], [], [], []
    for k in range(d):
        table = _build_table(
            standardized, k, deg[k], sdeg[k], order, bounds[k][0], bounds[k][1]
        )
        a0, b0 = _initial_coefficients(k, deg[k], sdeg[k], floor)
        a, b, value, norm, used, ok, why = _fit_component(
            table, a0, b0, floor, reg, iters, tol
        )
        coefficients.append((a, b))
        flags.append(ok)
        reasons.append(why)
        counts.append(used)
        objectives.append(value)
        norms.append(norm)
    transport = TriangularTransportMap(
        d,
        deg,
        sdeg,
        order,
        floor,
        tuple(coefficients),
        center,
        scale,
        (0.0,) * d,
        (1.0,) * d,
        tuple(b[0] for b in bounds),
        tuple(b[1] for b in bounds),
    )
    fit = SampleTransportFit(
        transport,
        tuple(flags),
        tuple(reasons),
        tuple(counts),
        tuple(objectives),
        tuple(norms),
        n,
        0.0,
    )
    mean_log = math.fsum(fit.log_density(r) for r in rows) / n
    return SampleTransportFit(
        transport,
        fit.converged,
        fit.stop_reasons,
        fit.iterations,
        fit.objectives,
        fit.gradient_norms,
        n,
        mean_log,
    )


# ---------------------------------------------------------------------------
# Fit from an unnormalized density
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DensityTransportFit:
    """Triangular map from the reference to an unnormalized target.

    ``transport_map`` is ``T`` with ``T#N(0, I) ~ pi``. The diagnostics use
    ``w(z) = log pi~(T(z)) + log |det grad T(z)| - log rho(z)``
    ([EQ-INV-TMAP-005]) on independent reference draws.

    Attributes:
        transport_map: The fitted map ``T`` (input ``z``, output ``x``).
        converged: Whether the optimizer met its tolerance.
        stop_reason: Optimizer message.
        iterations: Optimizer iterations.
        objective: Final training objective ``E_ref[-log pi~(T) - log
            |det grad T|]`` on the reference rule.
        gradient_norm: Final gradient norm of the training objective.
        target_evaluations: Evaluations of ``log pi~`` during training.
        reference_points: Number of training reference points.
        diagnostic_samples: Independent reference draws used for ``w``.
        log_weight_mean: ``E[w]``, a lower bound of ``log Z`` in
            expectation (``log Z - KL``).
        log_weight_variance: ``Var[w]``; zero for an exact map.
        kl_estimate: ``Var[w] / 2``, a small-variance estimate of
            ``KL(T#rho || pi)``; not a bound.
        log_normalizer_estimate: Same as ``log_weight_mean`` (biased low
            by the KL).
        importance_ess_fraction: Effective sample size of the self-
            normalized weights ``exp(w)`` divided by the draw count.

    References:
        [EQ-INV-TMAP-004], [EQ-INV-TMAP-005], [REF-EL-MOSELHY-MARZOUK-2012],
        [REF-MARZOUK-ETAL-2017].

    """

    transport_map: TriangularTransportMap
    converged: bool
    stop_reason: str
    iterations: int
    objective: float
    gradient_norm: float
    target_evaluations: int
    reference_points: int
    diagnostic_samples: int
    log_weight_mean: float
    log_weight_variance: float
    kl_estimate: float
    log_normalizer_estimate: float
    importance_ess_fraction: float

    def sample(
        self, count: int, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw approximate target samples ``T(z)``, ``z ~ N(0, I)``.

        Args:
            count: Number of draws.
            seed: Integer seed or stream.

        Returns:
            Approximate samples of the target.

        Raises:
            ValueError: If ``count`` is invalid.

        Equation:
            [EQ-INV-TMAP-004].

        References:
            [REF-EL-MOSELHY-MARZOUK-2012].

        """
        n = _positive_int(count, "count")
        rng = resolve_stream(seed).generator()
        d = self.transport_map.dimension
        return tuple(
            self.transport_map.forward(tuple(rng.gauss(0.0, 1.0) for _ in range(d)))
            for _ in range(n)
        )

    def log_density(self, point: Sequence[float]) -> float:
        """Return the normalized log density of ``T#rho`` at a point.

        Computed as ``log rho(T^-1(x)) - log |det grad T(T^-1(x))|``.

        Args:
            point: Target-space point.

        Returns:
            Log density of the fitted pushforward.

        Raises:
            ValueError: For invalid input or a failed inversion.

        Equation:
            [EQ-INV-TMAP-003].

        References:
            [REF-MARZOUK-ETAL-2017].

        """
        z = self.transport_map.inverse(point)
        return (
            -0.5 * _dot(z, z)
            - self.transport_map.dimension * _HALF_LOG_TWO_PI
            - self.transport_map.log_det_jacobian(z)
        )


def _reference_rule(
    dimension: int, kind: str, order: int, count: int, stream: RandomStream
) -> tuple[list[Vector], list[float]]:
    """Return reference points and weights for the training expectation.

    Args:
        dimension: Dimension.
        kind: ``"gauss-hermite"`` or ``"monte-carlo"``.
        order: Gauss-Hermite nodes per dimension.
        count: Monte Carlo points.
        stream: Random stream (Monte Carlo only).

    Returns:
        Points and weights summing to one.

    Raises:
        ValueError: For an unknown kind or too many tensor points.

    References:
        [EQ-INV-TMAP-004], [REF-DAVIS-RABINOWITZ-1984].

    """
    if kind == "gauss-hermite":
        nodes, weights = gauss_hermite(order)
        if len(nodes) ** dimension > _MAX_REFERENCE_POINTS:
            raise ValueError("tensor Gauss-Hermite rule is too large; use monte-carlo")
        points = [tuple(p) for p in itertools.product(nodes, repeat=dimension)]
        w = [math.prod(c) for c in itertools.product(weights, repeat=dimension)]
        return points, w
    if kind == "monte-carlo":
        rng = stream.generator()
        pts = [
            tuple(rng.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(count)
        ]
        return pts, [1.0 / count] * count
    raise ValueError("reference must be 'gauss-hermite' or 'monte-carlo'")


def fit_triangular_map_from_density(
    log_density: LogDensity,
    dimension: int,
    *,
    center: Sequence[float] | None = None,
    scale: Sequence[float] | None = None,
    degree: int | Sequence[int] = 3,
    slope_degree: int | Sequence[int] = 1,
    quadrature_order: int = 10,
    min_slope: float = 1e-3,
    gradient: GradientFunction | None = None,
    reference: str = "gauss-hermite",
    reference_order: int = 12,
    reference_samples: int = 1000,
    max_iterations: int = 300,
    gradient_tolerance: float = 1e-7,
    diagnostic_samples: int = 1000,
    seed: int | RandomStream | None = 0,
) -> DensityTransportFit:
    """Fit a monotone triangular map from ``N(0, I)`` to an unnormalized density.

    Minimizes ``E_rho[-log pi~(T(z)) - log |det grad T(z)|]`` over the
    coefficients of ``T`` with BFGS ([EQ-INV-TMAP-004]); ``T`` has the
    monotone parametrization of the module with output
    ``center + scale * s``. The expectation uses a fixed tensor Gauss-Hermite
    rule (exact for polynomial integrands, deterministic) or fixed Monte
    Carlo points, so the training problem is a deterministic smooth
    function of the coefficients. The objective is not convex in general
    (it is mode-seeking, ``KL(T#rho || pi)``) and BFGS returns a stationary
    point; the diagnostic of :class:`DensityTransportFit` reveals a poor
    one. Choose ``center`` and ``scale`` near the target mean and standard
    deviation to start close.

    Args:
        log_density: ``x -> log pi~(x)`` up to an additive constant;
            ``ValueError`` or a non-finite value makes the line search
            shorten the step.
        dimension: Target dimension ``d``.
        center: Output center (default zeros).
        scale: Positive output scales (default ones).
        degree: Total degree of ``a_k`` (integer or one per component).
        slope_degree: Total degree of ``b_k`` (integer or one per component).
        quadrature_order: Gauss-Legendre nodes for the integral.
        min_slope: Slope floor in ``(0, 1)``.
        gradient: Optional ``grad log pi~``; central differences with step
            ``1e-6 max(1, |x_i|)`` otherwise.
        reference: ``"gauss-hermite"`` or ``"monte-carlo"``.
        reference_order: Gauss-Hermite nodes per dimension.
        reference_samples: Monte Carlo reference points.
        max_iterations: BFGS iteration limit.
        gradient_tolerance: BFGS gradient tolerance.
        diagnostic_samples: Independent draws for the diagnostic ``w``.
        seed: Integer seed or stream (Monte Carlo reference and diagnostic).

    Returns:
        :class:`DensityTransportFit`.

    Raises:
        TypeError: If ``log_density`` is not callable or options are not
            integers.
        ValueError: For invalid dimension, scales or options, or a
            non-finite target at the starting map.

    Equation:
        [EQ-INV-TMAP-001], [EQ-INV-TMAP-004], [EQ-INV-TMAP-005].

    References:
        [REF-EL-MOSELHY-MARZOUK-2012], [REF-MARZOUK-ETAL-2017],
        [REF-NOCEDAL-WRIGHT-2006].

    """
    if not callable(log_density):
        raise TypeError("log_density must be callable")
    d = _positive_int(dimension, "dimension")
    deg, sdeg, order, floor = _check_options(
        degree, slope_degree, quadrature_order, min_slope, d
    )
    shift = as_vector(center, name="center") if center is not None else (0.0,) * d
    spread = as_vector(scale, name="scale") if scale is not None else (1.0,) * d
    if len(shift) != d or len(spread) != d:
        raise ValueError("center and scale must have length dimension")
    if min(spread) <= 0.0:
        raise ValueError("scale entries must be positive")
    stream = resolve_stream(seed)
    points, weights = _reference_rule(
        d,
        reference,
        _positive_int(reference_order, "reference_order"),
        _positive_int(reference_samples, "reference_samples"),
        stream.child("reference"),
    )
    tables = [_build_table(points, k, deg[k], sdeg[k], order) for k in range(d)]
    sizes = []
    start: list[float] = []
    for k in range(d):
        a0, b0 = _initial_coefficients(k, deg[k], sdeg[k], floor)
        sizes.append((len(a0), len(b0)))
        start.extend((*a0, *b0))
    evaluations = [0]
    log_scale = math.fsum(math.log(s) for s in spread)

    def unpack(theta: Sequence[float]) -> list[tuple[Vector, Vector]]:
        """Split the flat vector into per-component coefficient pairs.

        References:
            [EQ-INV-TMAP-004].

        """
        out = []
        offset = 0
        for na, nb in sizes:
            out.append(
                (
                    tuple(theta[offset : offset + na]),
                    tuple(theta[offset + na : offset + na + nb]),
                )
            )
            offset += na + nb
        return out

    def score(x: Vector) -> Vector:
        """Return ``grad log pi~`` at ``x``.

        References:
            [EQ-INV-TMAP-004].

        """
        if gradient is not None:
            g = as_vector(gradient(x), name="gradient")
            if len(g) != d:
                raise ValueError("gradient must have length dimension")
            return g
        out = []
        for i in range(d):
            h = 1e-6 * max(1.0, abs(x[i]))
            hi = (*x[:i], x[i] + h, *x[i + 1 :])
            lo = (*x[:i], x[i] - h, *x[i + 1 :])
            out.append((log_density(hi) - log_density(lo)) / (2.0 * h))
        return tuple(out)

    def objective(theta: Vector) -> tuple[float, Vector]:
        """Return the training objective and its gradient.

        References:
            [EQ-INV-TMAP-004].

        """
        pairs = unpack(theta)
        value = log_scale
        grad = [0.0] * len(theta)
        offsets = []
        offset = 0
        for na, nb in sizes:
            offsets.append(offset)
            offset += na + nb
        try:
            for m, w in enumerate(weights):
                x = []
                parts = []
                for k in range(d):
                    s, g_beta, slope, e_end, _ = _row_terms(
                        tables[k], m, pairs[k][0], pairs[k][1], floor
                    )
                    x.append(shift[k] + spread[k] * s)
                    parts.append((g_beta, slope, e_end))
                point = tuple(x)
                logp = log_density(point)
                if not math.isfinite(logp):
                    raise ValueError("log density is not finite")
                g = score(point)
                evaluations[0] += 1
                value += w * (-logp - math.fsum(math.log(p[1]) for p in parts))
                for k in range(d):
                    g_beta, slope, e_end = parts[k]
                    base = offsets[k]
                    na, nb = sizes[k]
                    gk = g[k] * spread[k]
                    for i, v in enumerate(tables[k].phi[m]):
                        grad[base + i] -= w * gk * v
                    ratio = e_end / slope
                    for i in range(nb):
                        grad[base + na + i] -= w * (
                            gk * g_beta[i] + ratio * tables[k].psi_end[m][i]
                        )
        except OverflowError as error:
            raise ValueError("transport map overflowed") from error
        return value, tuple(grad)

    def value_only(theta: Vector) -> float:
        """Objective value for the optimizer.

        References:
            [EQ-INV-TMAP-004].

        """
        return objective(theta)[0]

    def gradient_only(theta: Vector) -> Vector:
        """Objective gradient for the optimizer.

        References:
            [EQ-INV-TMAP-004].

        """
        return objective(theta)[1]

    result = minimize_bfgs(
        value_only,
        tuple(start),
        gradient=gradient_only,
        gradient_tolerance=require_finite_real(
            gradient_tolerance, name="gradient_tolerance"
        ),
        max_iterations=_positive_int(max_iterations, "max_iterations"),
        typical_scale=1.0,
    )
    transport = TriangularTransportMap(
        d,
        deg,
        sdeg,
        order,
        floor,
        tuple(unpack(result.x)),
        (0.0,) * d,
        (1.0,) * d,
        shift,
        spread,
    )
    count = _positive_int(diagnostic_samples, "diagnostic_samples", minimum=2)
    rng = stream.child("diagnostic").generator()
    logw = []
    for _ in range(count):
        z = tuple(rng.gauss(0.0, 1.0) for _ in range(d))
        x = transport.forward(z)
        lp = log_density(x)
        if not math.isfinite(lp):
            raise ValueError("log density is not finite on the diagnostic draws")
        logw.append(
            lp + transport.log_det_jacobian(z) + 0.5 * _dot(z, z) + d * _HALF_LOG_TWO_PI
        )
    mean_w = math.fsum(logw) / count
    var_w = math.fsum((v - mean_w) ** 2 for v in logw) / (count - 1)
    top = max(logw)
    expw = [math.exp(v - top) for v in logw]
    ess = math.fsum(expw) ** 2 / math.fsum(e * e for e in expw) / count
    return DensityTransportFit(
        transport,
        result.converged,
        result.message,
        result.iterations,
        result.value,
        result.gradient_norm,
        evaluations[0],
        len(points),
        count,
        mean_w,
        var_w,
        0.5 * var_w,
        mean_w,
        ess,
    )


# ---------------------------------------------------------------------------
# SVGD for inverse posteriors and the comparison with a reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SVGDPosteriorResult:
    """SVGD particles of a prior-whitened inverse posterior.

    Attributes:
        particles: Final particles in the coordinates the target was given
            in (whitened ``z``).
        physical_particles: ``to_physical(z)`` of every particle (equal to
            ``particles`` when no map was given).
        mean: Mean of the physical particles.
        covariance: Sample covariance of the physical particles.
        bandwidth: Kernel bandwidth ``h`` of the last iteration; the rule is
            ``h = med^2 / log n`` (median heuristic) unless fixed.
        bandwidth_rule: ``"median"`` or ``"fixed"``.
        iterations: Iterations performed.
        update_norm: Final largest particle update norm.
        converged: Whether the update tolerance was met.
        gradient_evaluations: Score evaluations.

    References:
        [EQ-INV-TMAP-006], [REF-LIU-WANG-2016].

    """

    particles: tuple[Vector, ...]
    physical_particles: tuple[Vector, ...]
    mean: Vector
    covariance: Matrix
    bandwidth: float
    bandwidth_rule: str
    iterations: int
    update_norm: float
    converged: bool
    gradient_evaluations: int


def svgd_posterior(
    log_density: LogDensity,
    gradient: GradientFunction,
    dimension: int,
    *,
    particles: int = 60,
    iterations: int = 300,
    step_size: float = 0.1,
    bandwidth: float | None = None,
    initial_scale: float = 1.0,
    tolerance: float = 0.0,
    to_physical: Callable[[Vector], Vector] | None = None,
    seed: int | RandomStream | None = 0,
) -> SVGDPosteriorResult:
    """Run SVGD on a prior-whitened inverse posterior.

    The target is the log posterior in whitened coordinates ``z`` (prior
    ``N(0, I)``), for instance ``WhitenedGaussianPriorTarget.log_density``
    and ``.gradient`` ([EQ-INV-TMAP-006]). Particles start as
    ``N(0, initial_scale^2 I)`` draws, which is the prior for
    ``initial_scale = 1``. The update is that of
    :func:`probstat.inference.stein_variational_gradient_descent` with the
    radial basis kernel and the median bandwidth ``h = med^2 / log n`` unless
    ``bandwidth`` is given; the step is fixed.

    Args:
        log_density: ``z -> log p(z)`` up to a constant.
        gradient: ``z -> grad log p(z)`` (analytic; required).
        dimension: Dimension of ``z``.
        particles: Particle count (at least 2).
        iterations: Iteration count.
        step_size: Fixed step ``eps > 0``.
        bandwidth: Fixed ``h`` or ``None`` for the median rule.
        initial_scale: Standard deviation of the starting draws.
        tolerance: Stop when the largest update is at most this.
        to_physical: Optional map from ``z`` to the physical parameters.
        seed: Integer seed or stream for the starting particles.

    Returns:
        :class:`SVGDPosteriorResult`.

    Raises:
        TypeError: If a callable argument is not callable.
        ValueError: For invalid sizes or scales.

    Equation:
        [EQ-INV-TMAP-006].

    References:
        [REF-LIU-WANG-2016], [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    if not callable(log_density) or not callable(gradient):
        raise TypeError("log_density and gradient must be callable")
    if to_physical is not None and not callable(to_physical):
        raise TypeError("to_physical must be callable")
    d = _positive_int(dimension, "dimension")
    n = _positive_int(particles, "particles", minimum=2)
    sigma = require_finite_real(initial_scale, name="initial_scale")
    if sigma <= 0.0:
        raise ValueError("initial_scale must be positive")
    rng = resolve_stream(seed).generator()
    start = [tuple(sigma * rng.gauss(0.0, 1.0) for _ in range(d)) for _ in range(n)]
    run = stein_variational_gradient_descent(
        log_density,
        start,
        step_size=step_size,
        iterations=iterations,
        gradient=gradient,
        bandwidth=bandwidth,
        tolerance=tolerance,
    )
    final = tuple(run.particles)
    physical = (
        tuple(as_vector(to_physical(p), name="physical") for p in final)
        if to_physical
        else final
    )
    m = len(physical)
    width = len(physical[0])
    mean = tuple(math.fsum(p[i] for p in physical) / m for i in range(width))
    cov = tuple(
        tuple(
            math.fsum((p[i] - mean[i]) * (p[j] - mean[j]) for p in physical) / (m - 1)
            for j in range(width)
        )
        for i in range(width)
    )
    return SVGDPosteriorResult(
        final,
        physical,
        mean,
        cov,
        run.bandwidth,
        "fixed" if bandwidth is not None else "median",
        run.iterations,
        run.update_norm,
        run.converged,
        run.gradient_evaluations,
    )


@dataclass(frozen=True, slots=True)
class ReferenceComparison:
    """Moment comparison of a sample with a quadrature reference.

    Attributes:
        mean_error_in_sd: ``(sample mean - reference mean) / reference sd``
            per coordinate.
        variance_ratio: ``sample variance / reference variance`` per
            coordinate (``< 1`` signals variance underestimation).
        covariance_relative_error: Frobenius norm of the covariance
            difference over that of the reference covariance.
        sample_count: Number of samples compared.

    References:
        [EQ-INV-TMAP-007], [EQ-INV-SMP-005].

    """

    mean_error_in_sd: Vector
    variance_ratio: Vector
    covariance_relative_error: float
    sample_count: int


def compare_samples_to_reference(
    samples: Sequence[Sequence[float]], reference: QuadratureReference
) -> ReferenceComparison:
    """Compare sample moments with a deterministic quadrature reference.

    Args:
        samples: Samples (rows) in the reference's coordinates.
        reference: Output of
            :func:`~probstat.inverse.posterior_sampling.quadrature_reference`.

    Returns:
        :class:`ReferenceComparison`.

    Raises:
        TypeError: If ``reference`` is not a quadrature reference.
        ValueError: For too few samples or a dimension mismatch.

    Equation:
        [EQ-INV-TMAP-007].

    References:
        [EQ-INV-SMP-005], [REF-DAVIS-RABINOWITZ-1984].

    """
    if not isinstance(reference, QuadratureReference):
        raise TypeError("reference must be a QuadratureReference")
    rows = _validated_samples(samples)
    d = reference.dimension
    if len(rows[0]) != d:
        raise ValueError("sample dimension must match the reference")
    n = len(rows)
    mean = tuple(math.fsum(r[i] for r in rows) / n for i in range(d))
    cov = [
        [
            math.fsum((r[i] - mean[i]) * (r[j] - mean[j]) for r in rows) / (n - 1)
            for j in range(d)
        ]
        for i in range(d)
    ]
    ref = reference.covariance
    sd = [math.sqrt(ref[i][i]) for i in range(d)]
    diff = math.sqrt(
        math.fsum((cov[i][j] - ref[i][j]) ** 2 for i in range(d) for j in range(d))
    )
    norm = math.sqrt(math.fsum(ref[i][j] ** 2 for i in range(d) for j in range(d)))
    return ReferenceComparison(
        tuple((mean[i] - reference.mean[i]) / sd[i] for i in range(d)),
        tuple(cov[i][i] / ref[i][i] for i in range(d)),
        diff / norm,
        n,
    )
