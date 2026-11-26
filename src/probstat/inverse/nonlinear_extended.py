"""Bound-constrained, robust and separable nonlinear least-squares inversion.

Three extensions of :mod:`probstat.inverse.gauss_newton` for a differentiable
forward map ``G`` (:class:`~probstat.inverse.gauss_newton.DifferentiableForward`,
Jacobian never formed) or, for variable projection, a small dense basis:

* :func:`bounded_gauss_newton` -- ``min 1/2 ||W (G(x) - y)||^2`` subject to
  ``l <= x <= u``. The method is a *projected Gauss-Newton with an
  epsilon-active set* in the style of Bertsekas' projected Newton method
  [EQ-INV-NLX-002]: variables that sit within ``eps_k`` of a bound with the
  gradient pointing outward are *active* and move only along a scaled
  negative gradient; the remaining *free* variables take a Gauss-Newton step
  from LSQR on the reduced linearized problem. The step is searched on the
  *projected arc* ``x(t) = P(x + t p)`` with the Armijo rule
  ``Phi(x(t)) <= Phi(x) + c g'(x(t) - x)`` [EQ-INV-NLX-003]. Optimality is
  measured by the projected gradient ``||x - P(x - g)||`` and the KKT
  violation with sign-correct bound multipliers [EQ-INV-NLX-001]; the active
  set is reported. There is no clamp of an unconstrained iterate: the start
  must be feasible and every trial is a projection followed by a sufficient
  decrease test. (This is not a scaled reflective method.)
* :func:`robust_gauss_newton` -- minimizes ``sum_i rho(e_i)``, ``e = W (G(x) -
  y)``, for a Huber or Cauchy loss [EQ-INV-NLX-004] by iteratively reweighted
  Gauss-Newton: the step minimizes the linearized *weighted* residual with
  weights ``rho'(e_i)/e_i`` and the line search tests the actual robust
  objective [EQ-INV-NLX-005]. The weights are ``ρ'(e)/e`` (IRLS), not
  ``rho''(e)``, which keeps the step matrix positive semidefinite for the
  non-convex Cauchy loss. A Gaussian weighted least-squares fit with Student-t
  weights is not robust and is not offered under this name.
* :func:`variable_projection` -- Golub-Pereyra variable projection for the
  separable model ``y ~ Phi(alpha) c``: the linear coefficients are eliminated
  by a thin SVD at every nonlinear point and Gauss-Newton runs on the reduced
  residual ``r(alpha) = (I - Phi Phi^+) y`` [EQ-INV-NLX-006]. The reduced
  Jacobian is either the *full* Golub-Pereyra one or Kaufman's approximation
  (``jacobian="full"`` or ``"kaufman"``, no default switching); the full one is
  the default. A rank-deficient ``Phi`` is a failure, never replaced by a
  pseudo-inverse.

Assumptions:
    ``G`` (resp. ``Phi``) is continuously differentiable near the iterates and
    the supplied derivatives are exact (use
    :func:`probstat.inverse.gauss_newton.check_derivatives`). A non-finite
    forward value at a *trial* point is a rejected trial, never clipped;
    at the start it raises ``ValueError``.

Common pitfalls:
    - All three methods are local: a stationary point is not a global minimum
      (the Cauchy loss and the separable fit are non-convex).
    - Robust thresholds are in *whitened* residual units (``W`` from
      ``noise``), so ``threshold ~ 1.3`` assumes unit-variance whitened noise.
    - ``converged`` is ``True`` only for the gradient-type stop reasons;
      ``"step"`` (stagnation) is reported as not converged.
    - Variable projection is not valid across rank changes of ``Phi(alpha)``;
      the solver stops with ``"rank_deficient"`` instead of differentiating
      through one.

Experimental (added in 0.28).

References:
    [REF-NOCEDAL-WRIGHT-2006], [REF-BERTSEKAS-1982], [REF-HUBER-1981],
    [REF-GOLUB-PEREYRA-1973], [REF-KAUFMAN-1975], [REF-OLEARY-RUST-2013],
    [REF-PAIGE-SAUNDERS-1982].

"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from probstat.inverse.gauss_newton import DifferentiableForward, ModelCounts
from probstat.linalg import (
    CallbackOperator,
    Matrix,
    Vector,
    dot,
    is_covariance_factor,
    lsqr,
    norm,
    singular_value_decomposition,
)

_EPS = 2.0**-52
_DEFAULT_FD_STEP: float = float(_EPS ** (1.0 / 3.0))
_DEFAULT_THRESHOLD = {"huber": 1.345, "cauchy": 2.385}


def _vec(values: object, size: int, name: str, *, infinite: bool = False) -> Vector:
    """Validate a real vector of a given length.

    Args:
        values: Candidate sequence of reals.
        size: Required length.
        name: Name used in messages.
        infinite: Whether ``+-inf`` entries are allowed (``nan`` never is).

    Returns:
        The entries as a tuple of floats.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If the length is wrong or an entry is not admissible.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    out: list[float] = []
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError(f"{name} must contain real numbers")
        out.append(float(entry))
    if len(out) != size:
        raise ValueError(f"{name} must have length {size}, got {len(out)}")
    for value in out:
        if math.isnan(value) or (not infinite and not math.isfinite(value)):
            raise ValueError(
                f"{name} must be finite" if not infinite else f"{name}: nan"
            )
    return tuple(out)


def _real(
    value: object, name: str, *, minimum: float = 0.0, strict: bool = False
) -> float:
    """Validate a finite real bounded below.

    Args:
        value: Candidate.
        name: Name used in messages.
        minimum: Lower bound.
        strict: Whether the bound is strict.

    Returns:
        The float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If non-finite or below the bound.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < minimum or (strict and result == minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be {relation} {minimum}")
    return result


def _posint(value: object, name: str) -> int:
    """Validate a positive integer.

    Args:
        value: Candidate.
        name: Name used in messages.

    Returns:
        The integer.

    Raises:
        TypeError: If not an integer.
        ValueError: If not positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _line_search_options(
    armijo: float, factor: float, tries: int
) -> tuple[float, float, int]:
    """Validate the Armijo constants.

    Args:
        armijo: Sufficient-decrease constant in ``(0, 1)``.
        factor: Backtracking factor in ``(0, 1)``.
        tries: Maximum trial steps per iteration.

    Returns:
        The validated triple.

    Raises:
        ValueError: If ``armijo`` or ``factor`` is outside ``(0, 1)``.

    References:
        [EQ-INV-NLX-003].

    """
    c1 = _real(armijo, "armijo", strict=True)
    f = _real(factor, "backtracking_factor", strict=True)
    if c1 >= 1.0 or f >= 1.0:
        raise ValueError("armijo and backtracking_factor must lie in (0, 1)")
    return c1, f, _posint(tries, "max_backtracks")


class _Evaluator:
    """Counted whitened residual, ``jvp`` and ``vjp`` of a validated problem.

    References:
        [EQ-INV-NLX-001].

    """

    def __init__(
        self, model: DifferentiableForward, data: Sequence[float], noise: Any
    ) -> None:
        """Validate the model, the data and the optional noise factor.

        Args:
            model: Differentiable forward map.
            data: Observations ``y``.
            noise: Covariance factor of ``probstat.linalg`` or ``None``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a dimension is inconsistent.

        References:
            [EQ-INV-NLX-001].

        """
        if not isinstance(model, DifferentiableForward):
            raise TypeError("model must implement DifferentiableForward")
        self.model = model
        self.n = _posint(model.input_dimension, "input_dimension")
        self.m = _posint(model.output_dimension, "output_dimension")
        self.data = _vec(data, self.m, "data")
        if noise is not None:
            if not is_covariance_factor(noise):
                raise TypeError("noise must be a probstat.linalg covariance factor")
            if noise.dimension != self.m:
                raise ValueError(
                    f"noise has dimension {noise.dimension}, expected {self.m}"
                )
        self.noise = noise
        self.calls = [0, 0, 0]
        self._start = getattr(model, "counts", None)

    def whiten(self, vector: Vector) -> Vector:
        """Return ``W v`` (identity without a noise model).

        References:
            [EQ-INV-NLX-001].

        """
        return vector if self.noise is None else tuple(self.noise.whiten(vector))

    def whiten_transpose(self, vector: Vector) -> Vector:
        """Return ``W' v`` (identity without a noise model).

        References:
            [EQ-INV-NLX-001].

        """
        if self.noise is None:
            return vector
        return tuple(self.noise.whiten_transpose(vector))

    def residual(self, x: Vector) -> tuple[Vector, bool]:
        """Return ``(G(x) - y, finite)`` (one counted forward evaluation).

        References:
            [EQ-INV-NLX-001].

        """
        self.calls[0] += 1
        value = tuple(float(v) for v in self.model.value(x))
        if len(value) != self.m:
            raise ValueError(f"value(x) must have length {self.m}, got {len(value)}")
        r = tuple(a - b for a, b in zip(value, self.data, strict=True))
        return r, all(math.isfinite(v) for v in r)

    def jvp(self, x: Vector, v: Vector) -> Vector:
        """Return ``J(x) v`` (counted).

        References:
            [EQ-INV-NLX-001].

        """
        self.calls[1] += 1
        out = tuple(float(a) for a in self.model.jvp(x, v))
        if len(out) != self.m:
            raise ValueError(f"jvp must have length {self.m}, got {len(out)}")
        return out

    def vjp(self, x: Vector, w: Vector) -> Vector:
        """Return ``J(x)' w`` (counted).

        References:
            [EQ-INV-NLX-001].

        """
        self.calls[2] += 1
        out = tuple(float(a) for a in self.model.vjp(x, w))
        if len(out) != self.n:
            raise ValueError(f"vjp must have length {self.n}, got {len(out)}")
        return out

    def counts(self) -> ModelCounts:
        """Return the evaluation counts of the solver.

        References:
            [EQ-INV-NLX-001].

        """
        inner = 0
        now = getattr(self.model, "counts", None)
        if isinstance(now, ModelCounts) and isinstance(self._start, ModelCounts):
            inner = now.derivative_value - self._start.derivative_value
        return ModelCounts(self.calls[0], self.calls[1], self.calls[2], inner)


def _all_finite(values: Sequence[float]) -> bool:
    """Return whether every entry is finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    return all(math.isfinite(v) for v in values)


# --------------------------------------------------------------------------
# Bound-constrained Gauss-Newton
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundedGaussNewtonIteration:
    """One iterate of :func:`bounded_gauss_newton`.

    Attributes:
        iteration: Index ``k`` (from ``0``).
        objective: ``Phi(x_k) = 1/2 ||W (G(x_k) - y)||^2``.
        projected_gradient_norm: ``||x_k - P(x_k - g_k)||``.
        active_count: Number of epsilon-active variables.
        step_length: Accepted arc parameter ``t`` (``0.0`` if no step).
        step_norm: ``||x_{k+1} - x_k||`` (``0.0`` if no step).
        rejected_objectives: ``Phi`` at each rejected arc trial (``nan`` where
            the forward value was not finite), so rejected attempts are kept.
        inner_iterations: LSQR iterations of the free-variable step.

    References:
        [EQ-INV-NLX-002], [EQ-INV-NLX-003].

    """

    iteration: int
    objective: float
    projected_gradient_norm: float
    active_count: int
    step_length: float
    step_norm: float
    rejected_objectives: tuple[float, ...]
    inner_iterations: int


@dataclass(frozen=True, slots=True)
class BoundedGaussNewtonResult:
    """Result of :func:`bounded_gauss_newton`.

    Attributes:
        estimate: Final iterate (always feasible).
        objective: ``1/2 ||W (G(x) - y)||^2``.
        whitened_residual_norm: ``||W (G(x) - y)||``.
        gradient: ``J' W' W (G(x) - y)`` at the estimate.
        projected_gradient_norm: ``||x - P(x - g)||``; zero iff ``x`` is a KKT
            point.
        kkt_violation: ``max_i`` of ``|g_i|`` for a free variable,
            ``max(0, -g_i)`` at a lower bound, ``max(0, g_i)`` at an upper
            bound.
        active_lower: Indices with ``x_i == l_i``.
        active_upper: Indices with ``x_i == u_i``.
        free: Indices strictly inside their bounds.
        multipliers: ``g_i`` at a lower bound, ``-g_i`` at an upper bound and
            ``0`` for free variables; nonnegative at a KKT point.
        iterations: Accepted steps.
        converged: ``True`` only for ``"projected_gradient"``.
        stop_reason: ``"projected_gradient"``, ``"step"`` (stagnation),
            ``"max_iterations"``, ``"line_search_failure"`` (no admissible
            decrease on the projected arc, including a non-descent
            direction) or ``"nonfinite"`` (non-finite derivative output).
        history: One :class:`BoundedGaussNewtonIteration` per iterate, the
            stopping iterate last.
        counts: Forward, ``jvp`` and ``vjp`` calls made by the solver.
        total_inner_iterations: Sum of LSQR iterations.
        wall_time: Seconds (benchmark information, not deterministic).

    References:
        [EQ-INV-NLX-001].

    """

    estimate: Vector
    objective: float
    whitened_residual_norm: float
    gradient: Vector
    projected_gradient_norm: float
    kkt_violation: float
    active_lower: tuple[int, ...]
    active_upper: tuple[int, ...]
    free: tuple[int, ...]
    multipliers: Vector
    iterations: int
    converged: bool
    stop_reason: str
    history: tuple[BoundedGaussNewtonIteration, ...]
    counts: ModelCounts
    total_inner_iterations: int
    wall_time: float


def _project(x: Vector, lower: Vector, upper: Vector) -> Vector:
    """Return the Euclidean projection ``P(x)`` onto the box.

    References:
        [EQ-INV-NLX-001].

    """
    return tuple(min(max(a, lo), hi) for a, lo, hi in zip(x, lower, upper, strict=True))


def _kkt_summary(
    x: Vector, g: Vector, lower: Vector, upper: Vector
) -> tuple[float, Vector, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return the KKT violation, multipliers and the active/free partition.

    Args:
        x: Feasible point.
        g: Gradient of the objective at ``x``.
        lower: Lower bounds.
        upper: Upper bounds.

    Returns:
        ``(violation, multipliers, at_lower, at_upper, free)``.

    References:
        [EQ-INV-NLX-001].

    """
    at_lower: list[int] = []
    at_upper: list[int] = []
    free: list[int] = []
    multipliers: list[float] = []
    violation = 0.0
    for i, (a, gi, lo, hi) in enumerate(zip(x, g, lower, upper, strict=True)):
        if a == lo:
            at_lower.append(i)
            multipliers.append(gi)
            violation = max(violation, -gi)
        elif a == hi:
            at_upper.append(i)
            multipliers.append(-gi)
            violation = max(violation, gi)
        else:
            free.append(i)
            multipliers.append(0.0)
            violation = max(violation, abs(gi))
    return violation, tuple(multipliers), tuple(at_lower), tuple(at_upper), tuple(free)


def bounded_gauss_newton(
    model: DifferentiableForward,
    data: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    initial: Sequence[float],
    *,
    noise: Any = None,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-8,
    step_tolerance: float = 1e-12,
    active_margin: float = 1e-3,
    inner_tolerance: float = 1e-10,
    inner_max_iterations: int | None = None,
    armijo: float = 1e-4,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 40,
) -> BoundedGaussNewtonResult:
    """Minimize ``1/2 ||W (G(x) - y)||^2`` over the box ``l <= x <= u``.

    Projected Gauss-Newton with an epsilon-active set [EQ-INV-NLX-002]. With
    ``g = J' W' W r`` and ``P`` the projection onto the box, a variable is
    *active* when it lies within ``eps = min(active_margin, ||x - P(x - g)||)``
    of a bound and ``g`` points out of the box there. Free variables take the
    LSQR solution of ``min_q ||W (r + J_F q)||`` (reduced Jacobian, never
    formed); active ones take ``-s g_i`` with the exact linearized line
    minimizer ``s = ||g_A||^2 / ||W J g_A||^2``. The step is searched on the
    projected arc ``P(x + t p)``, ``t = 1, f, f^2, ...`` with the Armijo rule
    ``Phi(trial) <= Phi(x) + c g'(trial - x)`` (plus a roundoff slack
    ``4 eps_mach |Phi|``) [EQ-INV-NLX-003]; a trial with a non-finite forward
    value is a rejected trial, never clipped. The projected-gradient test
    ``||x - P(x - g)|| <= gradient_tolerance max(1, ||pg_0||)`` is the only
    convergence criterion; it is zero exactly at a KKT point
    [EQ-INV-NLX-001].

    Args:
        model: Differentiable forward map.
        data: Observations ``y`` (length ``m``).
        lower: Lower bounds ``l`` (length ``n``; ``-inf`` allowed).
        upper: Upper bounds ``u`` (``+inf`` allowed), ``l < u`` entrywise.
        initial: Feasible start ``l <= x_0 <= u`` (never projected for you).
        noise: Covariance factor of ``probstat.linalg`` or ``None``.
        max_iterations: Outer iteration cap (``>= 1``).
        gradient_tolerance: Relative projected-gradient tolerance (``>= 0``).
        step_tolerance: Relative step tolerance (``>= 0``).
        active_margin: Upper bound ``> 0`` of the active-set margin.
        inner_tolerance: LSQR ``atol = btol`` (``> 0``).
        inner_max_iterations: LSQR cap per step (default ``2 n``).
        armijo: Sufficient-decrease constant in ``(0, 1)``.
        backtracking_factor: Factor in ``(0, 1)``.
        max_backtracks: Maximum trial steps per iteration.

    Returns:
        :class:`BoundedGaussNewtonResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a dimension is inconsistent, the bounds are invalid
            (``nan``, ``l >= u``), the start is infeasible, an option is out
            of range, or the forward map is not finite at ``initial``.

    References:
        [EQ-INV-NLX-001], [EQ-INV-NLX-002], [EQ-INV-NLX-003],
        [REF-BERTSEKAS-1982], [REF-NOCEDAL-WRIGHT-2006].

    """
    started = time.perf_counter()
    ev = _Evaluator(model, data, noise)
    n = ev.n
    lo = _vec(lower, n, "lower", infinite=True)
    hi = _vec(upper, n, "upper", infinite=True)
    if any(a >= b for a, b in zip(lo, hi, strict=True)):
        raise ValueError("lower must be strictly smaller than upper entrywise")
    x = _vec(initial, n, "initial")
    if any(a < l_ or a > h_ for a, l_, h_ in zip(x, lo, hi, strict=True)):
        raise ValueError("initial must be feasible (no silent projection)")
    cap = _posint(max_iterations, "max_iterations")
    gtol = _real(gradient_tolerance, "gradient_tolerance")
    stol = _real(step_tolerance, "step_tolerance")
    margin = _real(active_margin, "active_margin", strict=True)
    tol = _real(inner_tolerance, "inner_tolerance", strict=True)
    inner_cap = (
        2 * n
        if inner_max_iterations is None
        else _posint(inner_max_iterations, "inner_max_iterations")
    )
    c1, factor, tries = _line_search_options(
        armijo, backtracking_factor, max_backtracks
    )

    r, finite = ev.residual(x)
    if not finite:
        raise ValueError("forward map is not finite at the initial point")
    history: list[BoundedGaussNewtonIteration] = []
    inner_total = 0
    accepted = 0
    reason = "max_iterations"
    pg0 = math.nan
    stalled = False
    while True:
        e = ev.whiten(r)
        phi = 0.5 * dot(e, e)
        g = ev.vjp(x, ev.whiten_transpose(e))
        if not _all_finite(g):
            reason = "nonfinite"
            history.append(
                BoundedGaussNewtonIteration(accepted, phi, math.nan, 0, 0.0, 0.0, (), 0)
            )
            g = (math.nan,) * n
            break
        shifted = _project(tuple(a - b for a, b in zip(x, g, strict=True)), lo, hi)
        pgn = norm(tuple(a - b for a, b in zip(x, shifted, strict=True)))
        if math.isnan(pg0):
            pg0 = pgn
        eps = min(margin, pgn)
        active = [
            i
            for i in range(n)
            if (x[i] - lo[i] <= eps and g[i] > 0.0)
            or (hi[i] - x[i] <= eps and g[i] < 0.0)
        ]
        is_active = set(active)
        free_idx = [i for i in range(n) if i not in is_active]
        stop: str | None = None
        if pgn <= gtol * max(1.0, pg0):
            stop = "projected_gradient"
        elif stalled:
            stop = "step"
        elif accepted >= cap:
            stop = "max_iterations"
        if stop is not None:
            reason = stop
            history.append(
                BoundedGaussNewtonIteration(
                    accepted, phi, pgn, len(active), 0.0, 0.0, (), 0
                )
            )
            break

        p = [0.0] * n
        inner = 0
        if free_idx:

            def forward(
                q: Vector, _x: Vector = x, _free: list[int] = free_idx
            ) -> Vector:
                """Apply ``W J`` restricted to the free variables.

                References:
                    [EQ-INV-NLX-002].

                """
                full = [0.0] * n
                for j, i in enumerate(_free):
                    full[i] = q[j]
                return ev.whiten(ev.jvp(_x, tuple(full)))

            def adjoint(
                u: Vector, _x: Vector = x, _free: list[int] = free_idx
            ) -> Vector:
                """Apply the transpose of the restricted ``W J``.

                References:
                    [EQ-INV-NLX-002].

                """
                full = ev.vjp(_x, ev.whiten_transpose(u))
                return tuple(full[i] for i in _free)

            operator = CallbackOperator((ev.m, len(free_idx)), forward, adjoint)
            sol = lsqr(
                operator,
                tuple(-a for a in e),
                atol=tol,
                btol=tol,
                condition_limit=1e12,
                max_iterations=inner_cap,
            )
            inner = sol.iterations
            inner_total += inner
            for j, i in enumerate(free_idx):
                p[i] = sol.solution[j]
        if active:
            ga = [0.0] * n
            for i in active:
                ga[i] = g[i]
            gav = tuple(ga)
            bga = ev.whiten(ev.jvp(x, gav))
            denom = dot(bga, bga)
            scale = dot(gav, gav) / denom if denom > 0.0 else 1.0
            for i in active:
                p[i] = -scale * g[i]
        direction = tuple(p)
        slope = dot(g, direction)
        rejected: list[float] = []
        taken: tuple[Vector, Vector, float, float, float] | None = None
        if slope < 0.0 and _all_finite(direction):
            t = 1.0
            for _ in range(tries):
                trial = _project(
                    tuple(a + t * b for a, b in zip(x, direction, strict=True)), lo, hi
                )
                move = tuple(a - b for a, b in zip(trial, x, strict=True))
                decrease = dot(g, move)
                if decrease < 0.0:
                    r_t, fin = ev.residual(trial)
                    if fin:
                        e_t = ev.whiten(r_t)
                        phi_t = 0.5 * dot(e_t, e_t)
                        if phi_t <= phi + c1 * decrease + 4.0 * _EPS * abs(phi):
                            taken = (trial, r_t, t, norm(move), phi_t)
                            break
                        rejected.append(phi_t)
                    else:
                        rejected.append(math.nan)
                t *= factor
        if taken is None:
            reason = "line_search_failure"
            history.append(
                BoundedGaussNewtonIteration(
                    accepted, phi, pgn, len(active), 0.0, 0.0, tuple(rejected), inner
                )
            )
            break
        trial, r_t, t, step_norm, _phi_t = taken
        history.append(
            BoundedGaussNewtonIteration(
                accepted, phi, pgn, len(active), t, step_norm, tuple(rejected), inner
            )
        )
        stalled = step_norm <= stol * (1.0 + norm(x))
        x, r = trial, r_t
        accepted += 1

    e = ev.whiten(r)
    violation, multipliers, at_lower, at_upper, free = _kkt_summary(x, g, lo, hi)
    final_shifted = _project(tuple(a - b for a, b in zip(x, g, strict=True)), lo, hi)
    final_pg = (
        math.nan
        if reason == "nonfinite"
        else norm(tuple(a - b for a, b in zip(x, final_shifted, strict=True)))
    )
    return BoundedGaussNewtonResult(
        estimate=x,
        objective=0.5 * dot(e, e),
        whitened_residual_norm=norm(e),
        gradient=g,
        projected_gradient_norm=final_pg,
        kkt_violation=violation if reason != "nonfinite" else math.nan,
        active_lower=at_lower,
        active_upper=at_upper,
        free=free,
        multipliers=multipliers,
        iterations=accepted,
        converged=reason == "projected_gradient",
        stop_reason=reason,
        history=tuple(history),
        counts=ev.counts(),
        total_inner_iterations=inner_total,
        wall_time=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------
# Robust losses and iteratively reweighted Gauss-Newton
# --------------------------------------------------------------------------


def robust_loss(
    loss: Literal["huber", "cauchy"], threshold: float, residual: float
) -> float:
    """Return the robust loss ``rho(e)`` of a whitened residual.

    Huber: ``e^2/2`` for ``|e| <= c`` and ``c |e| - c^2/2`` otherwise
    (convex, linear tails). Cauchy: ``(c^2/2) log(1 + (e/c)^2)`` (non-convex,
    logarithmic tails) [EQ-INV-NLX-004]. Both are ``e^2/2 + O(e^4)`` near zero.

    Args:
        loss: ``"huber"`` or ``"cauchy"``.
        threshold: Scale ``c > 0`` in whitened residual units.
        residual: Whitened residual ``e``.

    Returns:
        ``rho(e)``.

    Raises:
        ValueError: If ``loss`` is unknown or ``threshold`` is not positive.

    References:
        [EQ-INV-NLX-004], [REF-HUBER-1981].

    """
    c = _real(threshold, "threshold", strict=True)
    a = abs(residual)
    if loss == "huber":
        return 0.5 * a * a if a <= c else c * a - 0.5 * c * c
    if loss == "cauchy":
        return 0.5 * c * c * math.log1p((a / c) ** 2)
    raise ValueError("loss must be 'huber' or 'cauchy'")


def robust_weight(
    loss: Literal["huber", "cauchy"], threshold: float, residual: float
) -> float:
    """Return the IRLS weight ``w(e) = rho'(e) / e`` (``1`` at ``e = 0``).

    Huber: ``min(1, c/|e|)``; Cauchy: ``1/(1 + (e/c)^2)`` [EQ-INV-NLX-004].
    The weight is in ``(0, 1]``; the gradient of ``sum rho(e_i)`` is
    ``J' W' (w * e)``.

    Args:
        loss: ``"huber"`` or ``"cauchy"``.
        threshold: Scale ``c > 0`` in whitened residual units.
        residual: Whitened residual ``e``.

    Returns:
        ``w(e)``.

    Raises:
        ValueError: If ``loss`` is unknown or ``threshold`` is not positive.

    References:
        [EQ-INV-NLX-004], [REF-HUBER-1981].

    """
    c = _real(threshold, "threshold", strict=True)
    a = abs(residual)
    if loss == "huber":
        return 1.0 if a <= c else c / a
    if loss == "cauchy":
        return 1.0 / (1.0 + (a / c) ** 2)
    raise ValueError("loss must be 'huber' or 'cauchy'")


@dataclass(frozen=True, slots=True)
class RobustGaussNewtonIteration:
    """One iterate of :func:`robust_gauss_newton`.

    Attributes:
        iteration: Index ``k`` (from ``0``).
        objective: ``sum_i rho(e_i)`` at ``x_k``.
        gradient_norm: ``||J' W' (w * e)||``.
        downweighted: Number of residuals with weight below ``1``.
        step_length: Accepted line-search factor (``0.0`` if no step).
        step_norm: ``||t p||``.
        rejected_objectives: Robust objective at rejected trials (``nan`` for a
            non-finite forward value).
        inner_iterations: LSQR iterations of the weighted step.

    References:
        [EQ-INV-NLX-005].

    """

    iteration: int
    objective: float
    gradient_norm: float
    downweighted: int
    step_length: float
    step_norm: float
    rejected_objectives: tuple[float, ...]
    inner_iterations: int


@dataclass(frozen=True, slots=True)
class RobustGaussNewtonResult:
    """Result of :func:`robust_gauss_newton`.

    Attributes:
        estimate: Final iterate.
        objective: ``sum_i rho(e_i)``.
        whitened_residual_norm: ``||W (G(x) - y)||`` (plain, not robust).
        gradient_norm: Norm of the robust gradient at the estimate.
        weights: Final IRLS weights ``w_i`` (outliers have small weights).
        downweighted: Number of weights below ``1``.
        loss: ``"huber"`` or ``"cauchy"``.
        threshold: Scale ``c`` actually used.
        iterations: Accepted steps.
        converged: ``True`` only for ``"gradient"``.
        stop_reason: ``"gradient"``, ``"step"`` (stagnation),
            ``"max_iterations"``, ``"line_search_failure"`` or ``"nonfinite"``.
        history: One :class:`RobustGaussNewtonIteration` per iterate.
        counts: Forward, ``jvp`` and ``vjp`` calls.
        total_inner_iterations: Sum of LSQR iterations.
        wall_time: Seconds (benchmark information, not deterministic).

    References:
        [EQ-INV-NLX-005].

    """

    estimate: Vector
    objective: float
    whitened_residual_norm: float
    gradient_norm: float
    weights: Vector
    downweighted: int
    loss: str
    threshold: float
    iterations: int
    converged: bool
    stop_reason: str
    history: tuple[RobustGaussNewtonIteration, ...]
    counts: ModelCounts
    total_inner_iterations: int
    wall_time: float


def robust_gauss_newton(
    model: DifferentiableForward,
    data: Sequence[float],
    *,
    loss: Literal["huber", "cauchy"] = "huber",
    threshold: float | None = None,
    noise: Any = None,
    initial: Sequence[float] | None = None,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-8,
    step_tolerance: float = 1e-12,
    inner_tolerance: float = 1e-10,
    inner_max_iterations: int | None = None,
    armijo: float = 1e-4,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 40,
) -> RobustGaussNewtonResult:
    """Minimize ``sum_i rho(e_i)``, ``e = W (G(x) - y)``, by IRLS Gauss-Newton.

    At ``x`` the weights are ``w_i = rho'(e_i)/e_i`` and the step minimizes
    the linearized weighted residual
    ``min_p ||diag(sqrt w) W (r + J p)||`` by LSQR [EQ-INV-NLX-005]. The
    gradient of the robust objective is ``J' W' (w * e)``. A backtracking
    Armijo search tests the *actual* robust objective (not the frozen-weight
    quadratic), with a roundoff slack ``4 eps |Phi|``. With ``loss="huber"``
    and a threshold above every residual all weights are ``1`` and the
    iteration is plain Gauss-Newton. The Cauchy loss is non-convex: the result
    is a local minimizer that depends on the start (for gross outliers start
    from a Huber or plain fit).

    Args:
        model: Differentiable forward map.
        data: Observations ``y`` (length ``m``).
        loss: ``"huber"`` or ``"cauchy"``.
        threshold: Scale ``c > 0`` in whitened units (default ``1.345`` for
            Huber, ``2.385`` for Cauchy).
        noise: Covariance factor of ``probstat.linalg`` or ``None``.
        initial: Start (default zeros).
        max_iterations: Outer iteration cap (``>= 1``).
        gradient_tolerance: Relative gradient tolerance (``>= 0``).
        step_tolerance: Relative step tolerance (``>= 0``).
        inner_tolerance: LSQR ``atol = btol`` (``> 0``).
        inner_max_iterations: LSQR cap per step (default ``2 n``).
        armijo: Sufficient-decrease constant in ``(0, 1)``.
        backtracking_factor: Factor in ``(0, 1)``.
        max_backtracks: Maximum trial steps per iteration.

    Returns:
        :class:`RobustGaussNewtonResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a dimension is inconsistent, ``loss`` is unknown, an
            option is out of range, or the forward map is not finite at the
            start.

    References:
        [EQ-INV-NLX-004], [EQ-INV-NLX-005], [REF-HUBER-1981],
        [REF-NOCEDAL-WRIGHT-2006].

    """
    started = time.perf_counter()
    ev = _Evaluator(model, data, noise)
    n, m = ev.n, ev.m
    if loss not in ("huber", "cauchy"):
        raise ValueError("loss must be 'huber' or 'cauchy'")
    c = (
        _DEFAULT_THRESHOLD[loss]
        if threshold is None
        else _real(threshold, "threshold", strict=True)
    )
    x = (0.0,) * n if initial is None else _vec(initial, n, "initial")
    cap = _posint(max_iterations, "max_iterations")
    gtol = _real(gradient_tolerance, "gradient_tolerance")
    stol = _real(step_tolerance, "step_tolerance")
    tol = _real(inner_tolerance, "inner_tolerance", strict=True)
    inner_cap = (
        2 * n
        if inner_max_iterations is None
        else _posint(inner_max_iterations, "inner_max_iterations")
    )
    c1, factor, tries = _line_search_options(
        armijo, backtracking_factor, max_backtracks
    )

    def objective_of(res: Vector) -> float:
        """Return ``sum_i rho(e_i)`` for a raw residual.

        References:
            [EQ-INV-NLX-004].

        """
        return math.fsum(robust_loss(loss, c, v) for v in ev.whiten(res))

    r, finite = ev.residual(x)
    if not finite:
        raise ValueError("forward map is not finite at the initial point")
    history: list[RobustGaussNewtonIteration] = []
    inner_total = 0
    accepted = 0
    reason = "max_iterations"
    g0 = math.nan
    stalled = False
    while True:
        e = ev.whiten(r)
        weights = tuple(robust_weight(loss, c, v) for v in e)
        down = sum(1 for w in weights if w < 1.0)
        phi = objective_of(r)
        g = ev.vjp(
            x,
            ev.whiten_transpose(tuple(w * v for w, v in zip(weights, e, strict=True))),
        )
        if not _all_finite(g):
            reason = "nonfinite"
            history.append(
                RobustGaussNewtonIteration(accepted, phi, math.nan, down, 0, 0, (), 0)
            )
            break
        gn = norm(g)
        if math.isnan(g0):
            g0 = gn
        stop: str | None = None
        if gn <= gtol * max(1.0, g0):
            stop = "gradient"
        elif stalled:
            stop = "step"
        elif accepted >= cap:
            stop = "max_iterations"
        if stop is not None:
            reason = stop
            history.append(
                RobustGaussNewtonIteration(accepted, phi, gn, down, 0.0, 0.0, (), 0)
            )
            break
        root = tuple(math.sqrt(w) for w in weights)

        def forward(p: Vector, _x: Vector = x, _root: Vector = root) -> Vector:
            """Apply ``diag(sqrt w) W J`` to ``p``.

            References:
                [EQ-INV-NLX-005].

            """
            jp = ev.whiten(ev.jvp(_x, p))
            return tuple(a * b for a, b in zip(_root, jp, strict=True))

        def adjoint(u: Vector, _x: Vector = x, _root: Vector = root) -> Vector:
            """Apply the transpose ``J' W' diag(sqrt w)`` to ``u``.

            References:
                [EQ-INV-NLX-005].

            """
            scaled = tuple(a * b for a, b in zip(_root, u, strict=True))
            return ev.vjp(_x, ev.whiten_transpose(scaled))

        sol = lsqr(
            CallbackOperator((m, n), forward, adjoint),
            tuple(-a * b for a, b in zip(root, e, strict=True)),
            atol=tol,
            btol=tol,
            condition_limit=1e12,
            max_iterations=inner_cap,
        )
        inner_total += sol.iterations
        direction = sol.solution
        slope = dot(g, direction)
        rejected: list[float] = []
        taken: tuple[Vector, Vector, float, float] | None = None
        if slope < 0.0 and _all_finite(direction):
            t = 1.0
            for _ in range(tries):
                trial = tuple(a + t * b for a, b in zip(x, direction, strict=True))
                r_t, fin = ev.residual(trial)
                if fin:
                    phi_t = objective_of(r_t)
                    if phi_t <= phi + c1 * t * slope + 4.0 * _EPS * abs(phi):
                        taken = (trial, r_t, t, t * norm(direction))
                        break
                    rejected.append(phi_t)
                else:
                    rejected.append(math.nan)
                t *= factor
        if taken is None:
            reason = "line_search_failure"
            history.append(
                RobustGaussNewtonIteration(
                    accepted, phi, gn, down, 0.0, 0.0, tuple(rejected), sol.iterations
                )
            )
            break
        trial, r_t, t, step_norm = taken
        history.append(
            RobustGaussNewtonIteration(
                accepted, phi, gn, down, t, step_norm, tuple(rejected), sol.iterations
            )
        )
        stalled = step_norm <= stol * (1.0 + norm(x))
        x, r = trial, r_t
        accepted += 1

    e = ev.whiten(r)
    final_weights = tuple(robust_weight(loss, c, v) for v in e)
    last = history[-1]
    return RobustGaussNewtonResult(
        estimate=x,
        objective=objective_of(r),
        whitened_residual_norm=norm(e),
        gradient_norm=last.gradient_norm,
        weights=final_weights,
        downweighted=sum(1 for w in final_weights if w < 1.0),
        loss=loss,
        threshold=c,
        iterations=accepted,
        converged=reason == "gradient",
        stop_reason=reason,
        history=tuple(history),
        counts=ev.counts(),
        total_inner_iterations=inner_total,
        wall_time=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------
# Variable projection
# --------------------------------------------------------------------------

BasisCallback = Callable[[Vector], Sequence[Sequence[float]]]
BasisDerivativeCallback = Callable[[Vector], Sequence[Sequence[Sequence[float]]]]


@dataclass(frozen=True, slots=True)
class VariableProjectionIteration:
    """One iterate of :func:`variable_projection`.

    Attributes:
        iteration: Index ``k`` (from ``0``).
        objective: ``1/2 ||r(alpha_k)||^2``.
        gradient_norm: ``||J_r' r||`` of the reduced problem.
        singular_ratio: ``sigma_min / sigma_max`` of ``Phi(alpha_k)``.
        step_length: Accepted line-search factor (``0.0`` if no step).
        step_norm: ``||t p||``.
        rejected_objectives: Reduced objective at rejected trials (``nan`` for
            a non-finite or rank-deficient ``Phi``).

    References:
        [EQ-INV-NLX-006].

    """

    iteration: int
    objective: float
    gradient_norm: float
    singular_ratio: float
    step_length: float
    step_norm: float
    rejected_objectives: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VariableProjectionResult:
    """Result of :func:`variable_projection`.

    Attributes:
        alpha: Final nonlinear parameters.
        coefficients: Linear coefficients ``c = Phi(alpha)^+ y``.
        objective: ``1/2 ||y - Phi c||^2``.
        residual_norm: ``||y - Phi c||``.
        gradient_norm: Norm of the reduced gradient at ``alpha``.
        iterations: Accepted steps.
        converged: ``True`` only for ``"gradient"``.
        stop_reason: ``"gradient"``, ``"step"`` (stagnation),
            ``"max_iterations"``, ``"line_search_failure"``,
            ``"rank_deficient"`` (the reduced Jacobian is numerically rank
            deficient at an accepted iterate).
        jacobian: ``"full"`` or ``"kaufman"``.
        derivative: ``"analytic"`` or ``"central_difference"``.
        history: One :class:`VariableProjectionIteration` per iterate.
        basis_evaluations: Calls of ``basis`` made by the solver (finite
            difference evaluations included).
        derivative_evaluations: Calls of ``basis_derivative``.
        rank_rejections: Trial points rejected for a rank-deficient ``Phi``.
        min_singular_ratio: Smallest ``sigma_min/sigma_max`` seen at an
            accepted iterate (a rank-change warning level).
        wall_time: Seconds (benchmark information, not deterministic).

    References:
        [EQ-INV-NLX-006].

    """

    alpha: Vector
    coefficients: Vector
    objective: float
    residual_norm: float
    gradient_norm: float
    iterations: int
    converged: bool
    stop_reason: str
    jacobian: str
    derivative: str
    history: tuple[VariableProjectionIteration, ...]
    basis_evaluations: int
    derivative_evaluations: int
    rank_rejections: int
    min_singular_ratio: float
    wall_time: float


@dataclass(frozen=True, slots=True)
class ReducedSystem:
    """Reduced residual, coefficients and Jacobian at one ``alpha``.

    Attributes:
        residual: ``r = (I - Phi Phi^+) y``.
        coefficients: ``c = Phi^+ y``.
        jacobian: ``m x p`` matrix ``dr/d alpha``.
        singular_ratio: ``sigma_min / sigma_max`` of ``Phi``.

    References:
        [EQ-INV-NLX-006].

    """

    residual: Vector
    coefficients: Vector
    jacobian: Matrix
    singular_ratio: float


class _Separable:
    """Counted evaluation of ``Phi(alpha)``, its derivatives and the reduction.

    References:
        [EQ-INV-NLX-006].

    """

    def __init__(
        self,
        basis: BasisCallback,
        data: Sequence[float],
        basis_derivative: BasisDerivativeCallback | None,
        derivative: str,
        step: float | None,
        mode: str,
        rank_tolerance: float,
    ) -> None:
        """Validate the options and store the callbacks.

        Args:
            basis: ``alpha -> Phi(alpha)`` as ``m`` rows of length ``k``.
            data: Observations ``y`` (length ``m``).
            basis_derivative: ``alpha -> [dPhi/dalpha_j]`` (analytic mode).
            derivative: ``"analytic"`` or ``"central_difference"``.
            step: Relative central-difference step.
            mode: ``"full"`` or ``"kaufman"``.
            rank_tolerance: Relative singular-value floor in ``(0, 1)``.

        Raises:
            TypeError: If a callback is not callable.
            ValueError: If the derivative options contradict each other.

        References:
            [EQ-INV-NLX-006].

        """
        if not callable(basis):
            raise TypeError("basis must be callable")
        if derivative not in ("analytic", "central_difference"):
            raise ValueError("derivative must be 'analytic' or 'central_difference'")
        if mode not in ("full", "kaufman"):
            raise ValueError("jacobian must be 'full' or 'kaufman'")
        if derivative == "analytic":
            if basis_derivative is None or not callable(basis_derivative):
                raise ValueError(
                    "analytic mode needs basis_derivative; there is no "
                    "finite-difference fallback, choose "
                    "derivative='central_difference' explicitly"
                )
            if step is not None:
                raise ValueError("step applies only to derivative='central_difference'")
            self.step = _DEFAULT_FD_STEP
        else:
            if basis_derivative is not None:
                raise ValueError(
                    "central_difference mode does not take basis_derivative"
                )
            self.step = (
                _DEFAULT_FD_STEP if step is None else _real(step, "step", strict=True)
            )
        self.basis = basis
        self.basis_derivative = basis_derivative
        self.derivative = derivative
        self.mode = mode
        self.y = _vec(data, len(data), "data")
        self.m = len(self.y)
        self.rank_tolerance = _real(rank_tolerance, "rank_tolerance", strict=True)
        if self.rank_tolerance >= 1.0:
            raise ValueError("rank_tolerance must be < 1")
        self.k = 0
        self.basis_calls = 0
        self.derivative_calls = 0

    def matrix(self, alpha: Vector) -> Matrix | None:
        """Return ``Phi(alpha)`` as a float matrix, or ``None`` if not finite.

        Raises:
            ValueError: If the shape is not ``m x k`` with ``m >= k >= 1`` or
                changes between calls.

        References:
            [EQ-INV-NLX-006].

        """
        self.basis_calls += 1
        rows = tuple(tuple(float(v) for v in row) for row in self.basis(alpha))
        if len(rows) != self.m:
            raise ValueError(f"basis must have {self.m} rows, got {len(rows)}")
        k = len(rows[0]) if rows else 0
        if k < 1 or k > self.m or any(len(row) != k for row in rows):
            raise ValueError("basis must be a rectangular m x k matrix with k <= m")
        if self.k not in (0, k):
            raise ValueError("the number of basis columns changed between calls")
        self.k = k
        if not all(_all_finite(row) for row in rows):
            return None
        return rows

    def derivatives(self, alpha: Vector) -> list[Matrix]:
        """Return ``dPhi/dalpha_j`` for every nonlinear parameter.

        Raises:
            ValueError: If a derivative has the wrong shape or the basis is not
                finite at a finite-difference point.

        References:
            [EQ-INV-NLX-006].

        """
        p = len(alpha)
        if self.derivative == "analytic":
            assert self.basis_derivative is not None
            self.derivative_calls += 1
            out = [
                tuple(tuple(float(v) for v in row) for row in mat)
                for mat in self.basis_derivative(alpha)
            ]
            if len(out) != p:
                raise ValueError(f"basis_derivative must return {p} matrices")
            for mat in out:
                if len(mat) != self.m or any(len(row) != self.k for row in mat):
                    raise ValueError("basis_derivative matrices must be m x k")
            return out
        result: list[Matrix] = []
        for j in range(p):
            h = self.step * (1.0 + abs(alpha[j]))
            up = list(alpha)
            down = list(alpha)
            up[j] += h
            down[j] -= h
            plus = self.matrix(tuple(up))
            minus = self.matrix(tuple(down))
            if plus is None or minus is None:
                raise ValueError("basis is not finite at a finite-difference point")
            result.append(
                tuple(
                    tuple((a - b) / (2.0 * h) for a, b in zip(ra, rb, strict=True))
                    for ra, rb in zip(plus, minus, strict=True)
                )
            )
        return result

    def reduce(self, alpha: Vector) -> tuple[Any, float] | None:
        """Return ``(state, ratio)`` for ``alpha`` or ``None`` if unusable.

        ``state = (Phi, U, sigma, V, c, r)``; ``None`` is returned when ``Phi``
        is not finite or ``sigma_min/sigma_max <= rank_tolerance``.

        References:
            [EQ-INV-NLX-006].

        """
        phi = self.matrix(alpha)
        if phi is None:
            return None
        svd = singular_value_decomposition(phi)
        s = svd.singular_values
        ratio = s[-1] / s[0] if s[0] > 0.0 else 0.0
        if ratio <= self.rank_tolerance:
            return None
        u, v = svd.u, svd.v
        uty = [dot(tuple(row[j] for row in u), self.y) for j in range(len(s))]
        c = tuple(
            math.fsum(v[i][j] * uty[j] / s[j] for j in range(len(s)))
            for i in range(self.k)
        )
        fit = tuple(dot(row, c) for row in phi)
        r = tuple(a - b for a, b in zip(self.y, fit, strict=True))
        return (phi, u, s, v, c, r), ratio

    def jacobian(self, state: Any, alpha: Vector) -> Matrix:
        """Return the ``m x p`` reduced Jacobian (full or Kaufman).

        Column ``j`` is ``-(P D_j c + (Phi^+)' D_j' r)`` with ``P = I - U U'``;
        Kaufman keeps only the first term [EQ-INV-NLX-006].

        References:
            [EQ-INV-NLX-006].

        """
        _phi, u, s, v, c, r = state
        mats = self.derivatives(alpha)
        r_cols = len(s)
        u_cols = [tuple(row[j] for row in u) for j in range(r_cols)]
        columns: list[Vector] = []
        for d in mats:
            dc = tuple(dot(row, c) for row in d)
            coef = [dot(col, dc) for col in u_cols]
            col = [
                dc[i] - math.fsum(coef[j] * u_cols[j][i] for j in range(r_cols))
                for i in range(self.m)
            ]
            if self.mode == "full":
                dtr = tuple(
                    math.fsum(d[i][a] * r[i] for i in range(self.m))
                    for a in range(self.k)
                )
                vt = [
                    math.fsum(v[a][j] * dtr[a] for a in range(self.k))
                    for j in range(r_cols)
                ]
                extra = [
                    math.fsum(u_cols[j][i] * vt[j] / s[j] for j in range(r_cols))
                    for i in range(self.m)
                ]
                col = [a + b for a, b in zip(col, extra, strict=True)]
            columns.append(tuple(-a for a in col))
        return tuple(
            tuple(columns[j][i] for j in range(len(columns))) for i in range(self.m)
        )


def variable_projection_reduced_system(
    basis: BasisCallback,
    data: Sequence[float],
    alpha: Sequence[float],
    *,
    basis_derivative: BasisDerivativeCallback | None = None,
    derivative: Literal["analytic", "central_difference"] = "analytic",
    step: float | None = None,
    jacobian: Literal["full", "kaufman"] = "full",
    rank_tolerance: float = 1e-10,
) -> ReducedSystem:
    """Return the reduced residual, coefficients and Jacobian at ``alpha``.

    ``r(alpha) = (I - Phi Phi^+) y``, ``c = Phi^+ y`` and
    ``dr/dalpha_j = -(P D_j c + (Phi^+)' D_j' r)`` with ``D_j = dPhi/dalpha_j``
    and ``P = I - Phi Phi^+``; ``jacobian="kaufman"`` drops the second term
    [EQ-INV-NLX-006]. Exposed so a derivative can be checked against finite
    differences of ``r`` (the second term matters when ``r`` is not small).

    Args:
        basis: ``alpha -> Phi(alpha)`` as ``m`` rows of length ``k``.
        data: Observations ``y`` (length ``m``).
        alpha: Nonlinear parameters.
        basis_derivative: ``alpha -> [dPhi/dalpha_j]`` for analytic mode.
        derivative: ``"analytic"`` or ``"central_difference"``.
        step: Relative central-difference step.
        jacobian: ``"full"`` (Golub-Pereyra) or ``"kaufman"``.
        rank_tolerance: Relative singular-value floor in ``(0, 1)``.

    Returns:
        :class:`ReducedSystem`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If options contradict each other, a shape is wrong, or
            ``Phi(alpha)`` is not finite or numerically rank deficient.

    References:
        [EQ-INV-NLX-006], [REF-GOLUB-PEREYRA-1973], [REF-KAUFMAN-1975].

    """
    sep = _Separable(
        basis, data, basis_derivative, derivative, step, jacobian, rank_tolerance
    )
    a = _vec(alpha, len(alpha), "alpha")
    reduced = sep.reduce(a)
    if reduced is None:
        raise ValueError("Phi(alpha) is not finite or is numerically rank deficient")
    state, ratio = reduced
    return ReducedSystem(state[5], state[4], sep.jacobian(state, a), ratio)


def variable_projection(
    basis: BasisCallback,
    data: Sequence[float],
    initial: Sequence[float],
    *,
    basis_derivative: BasisDerivativeCallback | None = None,
    derivative: Literal["analytic", "central_difference"] = "analytic",
    step: float | None = None,
    jacobian: Literal["full", "kaufman"] = "full",
    rank_tolerance: float = 1e-10,
    jacobian_rank_tolerance: float = 1e-10,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-8,
    step_tolerance: float = 1e-12,
    armijo: float = 1e-4,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 40,
) -> VariableProjectionResult:
    """Fit ``y ~ Phi(alpha) c`` by Golub-Pereyra variable projection.

    Each nonlinear point reduces the problem by a thin SVD of ``Phi(alpha)``
    (``c = Phi^+ y``, ``r = y - Phi c``) and Gauss-Newton with a backtracking
    Armijo search runs on ``1/2 ||r(alpha)||^2`` with the reduced Jacobian of
    :func:`variable_projection_reduced_system` [EQ-INV-NLX-006]. The Gauss-
    Newton step solves ``min_p ||r + J_r p||`` through the SVD of ``J_r``.
    ``jacobian="full"`` is the exact Golub-Pereyra Jacobian (quadratic-type
    local rate for zero residual and the correct reduced gradient for any
    residual); ``"kaufman"`` is the cheaper approximation whose gradient is
    inexact, so the gradient test is then only approximate.

    Failure is reported, never repaired: a trial point with rank-deficient or
    non-finite ``Phi`` is a rejected trial (counted in ``rank_rejections``),
    and a rank-deficient reduced Jacobian (a nonlinear parameter the data
    cannot identify) ends the run with ``"rank_deficient"``.

    Args:
        basis: ``alpha -> Phi(alpha)`` as ``m`` rows of length ``k <= m``.
        data: Observations ``y`` (length ``m``).
        initial: Start ``alpha_0`` (length ``p``).
        basis_derivative: ``alpha -> [dPhi/dalpha_j]`` (analytic mode).
        derivative: ``"analytic"`` or ``"central_difference"`` (explicit).
        step: Relative central-difference step (default ``eps^(1/3)``).
        jacobian: ``"full"`` or ``"kaufman"``.
        rank_tolerance: Relative singular-value floor in ``(0, 1)`` of
            ``Phi`` (a trial below it is rejected).
        jacobian_rank_tolerance: Relative singular-value floor in ``(0, 1)``
            of the reduced Jacobian ``J_r`` (below it the run stops with
            ``"rank_deficient"``).
        max_iterations: Outer iteration cap (``>= 1``).
        gradient_tolerance: Relative reduced-gradient tolerance (``>= 0``).
        step_tolerance: Relative step tolerance (``>= 0``).
        armijo: Sufficient-decrease constant in ``(0, 1)``.
        backtracking_factor: Factor in ``(0, 1)``.
        max_backtracks: Maximum trial steps per iteration.

    Returns:
        :class:`VariableProjectionResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If options contradict each other, a shape is wrong, or
            ``Phi(alpha_0)`` is not finite or is rank deficient.

    References:
        [EQ-INV-NLX-006], [REF-GOLUB-PEREYRA-1973], [REF-KAUFMAN-1975],
        [REF-OLEARY-RUST-2013].

    """
    started = time.perf_counter()
    sep = _Separable(
        basis, data, basis_derivative, derivative, step, jacobian, rank_tolerance
    )
    if isinstance(initial, (str, bytes)) or not isinstance(initial, Sequence):
        raise TypeError("initial must be a sequence of real numbers")
    alpha = _vec(initial, len(initial), "initial")
    if not alpha:
        raise ValueError("initial must not be empty")
    cap = _posint(max_iterations, "max_iterations")
    gtol = _real(gradient_tolerance, "gradient_tolerance")
    stol = _real(step_tolerance, "step_tolerance")
    jac_floor = _real(jacobian_rank_tolerance, "jacobian_rank_tolerance", strict=True)
    if jac_floor >= 1.0:
        raise ValueError("jacobian_rank_tolerance must be < 1")
    c1, factor, tries = _line_search_options(
        armijo, backtracking_factor, max_backtracks
    )
    reduced = sep.reduce(alpha)
    if reduced is None:
        raise ValueError("Phi(alpha_0) is not finite or is numerically rank deficient")
    state, ratio = reduced
    history: list[VariableProjectionIteration] = []
    rank_rejections = 0
    min_ratio = ratio
    accepted = 0
    reason = "max_iterations"
    g0 = math.nan
    stalled = False
    gn = math.nan
    while True:
        r = state[5]
        phi_val = 0.5 * dot(r, r)
        jac = sep.jacobian(state, alpha)
        g = tuple(
            math.fsum(jac[i][j] * r[i] for i in range(sep.m)) for j in range(len(alpha))
        )
        gn = norm(g)
        if math.isnan(g0):
            g0 = gn
        stop: str | None = None
        if gn <= gtol * max(1.0, g0):
            stop = "gradient"
        elif stalled:
            stop = "step"
        elif accepted >= cap:
            stop = "max_iterations"
        if stop is not None:
            reason = stop
            history.append(
                VariableProjectionIteration(accepted, phi_val, gn, ratio, 0.0, 0.0, ())
            )
            break
        jsvd = singular_value_decomposition(jac)
        js = jsvd.singular_values
        if js[0] <= 0.0 or js[-1] / js[0] <= jac_floor:
            reason = "rank_deficient"
            history.append(
                VariableProjectionIteration(accepted, phi_val, gn, ratio, 0.0, 0.0, ())
            )
            break
        utr = [dot(tuple(row[j] for row in jsvd.u), r) for j in range(len(js))]
        direction = tuple(
            -math.fsum(jsvd.v[i][j] * utr[j] / js[j] for j in range(len(js)))
            for i in range(len(alpha))
        )
        slope = dot(g, direction)
        rejected: list[float] = []
        taken: tuple[Vector, Any, float, float, float] | None = None
        if slope < 0.0 and _all_finite(direction):
            t = 1.0
            for _ in range(tries):
                trial = tuple(a + t * b for a, b in zip(alpha, direction, strict=True))
                out = sep.reduce(trial)
                if out is None:
                    rejected.append(math.nan)
                    rank_rejections += 1
                else:
                    st, rt = out
                    phi_t = 0.5 * dot(st[5], st[5])
                    if phi_t <= phi_val + c1 * t * slope + 4.0 * _EPS * abs(phi_val):
                        taken = (trial, st, rt, t, t * norm(direction))
                        break
                    rejected.append(phi_t)
                t *= factor
        if taken is None:
            reason = "line_search_failure"
            history.append(
                VariableProjectionIteration(
                    accepted, phi_val, gn, ratio, 0.0, 0.0, tuple(rejected)
                )
            )
            break
        trial, state_new, ratio_new, t, step_norm = taken
        history.append(
            VariableProjectionIteration(
                accepted, phi_val, gn, ratio, t, step_norm, tuple(rejected)
            )
        )
        stalled = step_norm <= stol * (1.0 + norm(alpha))
        alpha, state, ratio = trial, state_new, ratio_new
        min_ratio = min(min_ratio, ratio)
        accepted += 1

    r = state[5]
    return VariableProjectionResult(
        alpha=alpha,
        coefficients=state[4],
        objective=0.5 * dot(r, r),
        residual_norm=norm(r),
        gradient_norm=gn,
        iterations=accepted,
        converged=reason == "gradient",
        stop_reason=reason,
        jacobian=jacobian,
        derivative=derivative,
        history=tuple(history),
        basis_evaluations=sep.basis_calls,
        derivative_evaluations=sep.derivative_calls,
        rank_rejections=rank_rejections,
        min_singular_ratio=min_ratio,
        wall_time=time.perf_counter() - started,
    )
