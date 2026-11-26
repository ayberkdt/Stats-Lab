"""Steihaug truncated CG and a trust-region Newton-CG minimizer.

:func:`steihaug_cg` approximately solves the trust-region subproblem
``min_p g'p + 1/2 p'Hp  s.t.  ||p|| <= Delta`` [EQ-OPT-TR-001] by conjugate
gradients started at zero, using only Hessian *actions* ``v -> H v``. The
iteration stops on negative (or zero) curvature, on reaching the boundary
``||p|| = Delta`` [EQ-OPT-TR-002], or on a residual tolerance; the iterate is
inside the region at every step, so the result is always feasible and
(CG started at zero) its model value decreases monotonically [REF-STEIHAUG-1983].

:func:`trust_region_newton_cg` is the globalized Newton-CG method of Nocedal
and Wright (Algorithm 4.1 with Steihaug subproblem solves): the ratio
``rho = actual / predicted reduction`` accepts or rejects the trial step and
updates the radius [EQ-OPT-TR-003]; the inner CG tolerance follows an
inexact-Newton forcing term [EQ-OPT-TR-004]. A trial whose objective is not
finite is a rejected step (the radius shrinks); the value is never clipped.

Assumptions:
    ``objective`` and ``gradient`` are consistent and ``hessian_action(x, v)``
    returns ``H(x) v`` for a symmetric ``H(x)``. The Hessian need not be
    positive definite (negative curvature is handled), but it must be
    symmetric: CG relies on it. Convergence theory needs a bounded-below
    objective with Lipschitz gradient.

Common pitfalls:
    - ``max_iterations`` counts *attempts*, including rejected steps.
    - The gradient test is relative to the initial gradient norm,
      ``||g|| <= tol max(1, ||g_0||)``. On a problem with a large objective
      value the ratio ``rho`` is not resolved below about ``16 eps |f|``;
      the run then ends with ``"radius_collapse"`` rather than ``"gradient"``
      if ``gradient_tolerance`` is set below what the objective can resolve.
    - A stationary point found is only a local minimum candidate when the
      final Hessian is positive semidefinite; a saddle can be left by
      negative curvature but is not guaranteed to be.

Experimental (added in 0.28).

References:
    [REF-STEIHAUG-1983], [REF-NOCEDAL-WRIGHT-2006].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.linalg import Vector, dot, norm

_EPS = 2.0**-52
_BOUNDARY_SLACK = 1.0 - 1e-10
_CONVERGED = frozenset({"gradient"})

HessianAction = Callable[[Vector], Sequence[float]]
PointHessianAction = Callable[[Vector, Vector], Sequence[float]]


@dataclass(frozen=True, slots=True)
class SteihaugResult:
    """Result of :func:`steihaug_cg`.

    Attributes:
        step: Approximate minimizer ``p`` of the subproblem, ``||p|| <= Delta``.
        predicted_reduction: ``-(g'p + 1/2 p'Hp) >= 0`` of the quadratic model.
        reason: ``"interior"`` (residual tolerance met inside the region),
            ``"boundary"`` (the CG step left the region, the step is cut at
            ``||p|| = Delta``), ``"negative_curvature"`` (``d'Hd <= 0``; moved
            to the boundary along ``d``), ``"max_iterations"`` (cap reached,
            ``p`` is the current interior iterate) or ``"nonfinite"`` (the
            Hessian action returned non-finite entries; ``p`` is the last
            finite iterate).
        iterations: CG iterations performed.
        hessian_actions: Calls of ``hessian_action``.
        step_norm: ``||p||``.

    References:
        [EQ-OPT-TR-001], [EQ-OPT-TR-002].

    """

    step: Vector
    predicted_reduction: float
    reason: str
    iterations: int
    hessian_actions: int
    step_norm: float


@dataclass(frozen=True, slots=True)
class TrustRegionIteration:
    """One attempt of :func:`trust_region_newton_cg`.

    Attributes:
        iteration: Attempt index ``k`` (from ``0``); the final record is the
            stopping state and carries no step.
        objective: ``f(x_k)`` at the current accepted iterate.
        gradient_norm: ``||g(x_k)||``.
        radius: Radius ``Delta_k`` used for this attempt.
        step_norm: ``||p_k||`` of the trial step (``0`` for the stop record).
        rho: Ratio of actual to predicted reduction (``nan`` if the trial
            objective was not finite, the predicted reduction was not
            resolvable or there was no step).
        predicted_reduction: Model decrease of the trial step.
        trial_objective: ``f(x_k + p_k)`` (``nan`` if not finite or no step).
        accepted: Whether the trial step was taken.
        inner_iterations: CG iterations of the subproblem.
        inner_reason: Steihaug stop reason (``"none"`` for the stop record).

    References:
        [EQ-OPT-TR-003].

    """

    iteration: int
    objective: float
    gradient_norm: float
    radius: float
    step_norm: float
    rho: float
    predicted_reduction: float
    trial_objective: float
    accepted: bool
    inner_iterations: int
    inner_reason: str


@dataclass(frozen=True, slots=True)
class TrustRegionResult:
    """Result of :func:`trust_region_newton_cg`.

    Attributes:
        x: Final accepted iterate.
        value: ``f(x)``.
        gradient_norm: ``||g(x)||`` (``nan`` after a non-finite gradient).
        iterations: Accepted steps.
        attempts: Trial steps made (accepted plus rejected).
        converged: ``True`` only for ``stop_reason == "gradient"``.
        stop_reason: ``"gradient"``, ``"radius_collapse"`` (radius below
            ``minimum_radius (1 + ||x||)``), ``"max_iterations"`` or
            ``"nonfinite"`` (non-finite gradient or Hessian action).
        radius: Radius at the stop.
        history: One :class:`TrustRegionIteration` per attempt plus the
            stopping record.
        objective_evaluations: Calls of ``objective``.
        gradient_evaluations: Calls of ``gradient``.
        hessian_actions: Calls of ``hessian_action``.
        inner_iterations: Total CG iterations.

    References:
        [EQ-OPT-TR-003], [EQ-OPT-TR-004].

    """

    x: Vector
    value: float
    gradient_norm: float
    iterations: int
    attempts: int
    converged: bool
    stop_reason: str
    radius: float
    history: tuple[TrustRegionIteration, ...]
    objective_evaluations: int
    gradient_evaluations: int
    hessian_actions: int
    inner_iterations: int


def _vector(values: object, size: int | None, name: str) -> Vector:
    """Validate a finite real vector, optionally of a fixed length.

    Args:
        values: Candidate sequence of reals.
        size: Required length, or ``None`` for any positive length.
        name: Name used in messages.

    Returns:
        The entries as a tuple of floats.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If the length is wrong, empty, or an entry is not finite.

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
    if not out:
        raise ValueError(f"{name} must not be empty")
    if size is not None and len(out) != size:
        raise ValueError(f"{name} must have length {size}, got {len(out)}")
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
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


def _positive_int(value: object, name: str) -> int:
    """Validate a strictly positive integer.

    Args:
        value: Candidate.
        name: Name used in messages.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < 1``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _boundary_tau(z: Vector, d: Vector, radius: float) -> float:
    """Return the positive root ``tau`` of ``||z + tau d|| = radius``.

    Args:
        z: Interior point (``||z|| <= radius``).
        d: Nonzero direction.
        radius: Trust-region radius.

    Returns:
        ``tau >= 0`` with ``||z + tau d|| = radius``, by the cancellation-free
        form of the quadratic formula.

    References:
        [EQ-OPT-TR-002].

    """
    a = dot(d, d)
    b = 2.0 * dot(z, d)
    c = min(dot(z, z) - radius * radius, 0.0)
    disc = math.sqrt(max(b * b - 4.0 * a * c, 0.0))
    if b <= 0.0:
        return (-b + disc) / (2.0 * a)
    return (-2.0 * c) / (b + disc)


def steihaug_cg(
    gradient: Sequence[float],
    hessian_action: HessianAction,
    radius: float,
    *,
    tolerance: float,
    max_iterations: int | None = None,
) -> SteihaugResult:
    """Solve ``min g'p + 1/2 p'Hp`` over ``||p|| <= radius`` by truncated CG.

    Conjugate gradients on ``H p = -g`` from ``p = 0`` [EQ-OPT-TR-001]. With
    ``r_0 = g``, ``d_0 = -g`` each iteration computes ``H d_j`` and

    * if ``d_j' H d_j <= 0`` (negative or zero curvature) the step moves along
      ``d_j`` to the boundary: ``p = z_j + tau d_j`` with ``tau >= 0`` solving
      ``||z_j + tau d_j|| = radius`` [EQ-OPT-TR-002] (reason
      ``"negative_curvature"``);
    * else with ``alpha = r'r / d'Hd`` if ``||z_j + alpha d_j|| >= radius`` the
      step is cut at the boundary the same way (reason ``"boundary"``);
    * else ``z_(j+1) = z_j + alpha d_j``, and the iteration stops with reason
      ``"interior"`` if ``||r_(j+1)|| <= tolerance`` (an absolute residual
      tolerance; the caller supplies the forcing term [EQ-OPT-TR-004]).

    When the Hessian is positive definite and the tolerance is reached inside
    the region the result is the inexact Newton step ``-H^-1 g``. The
    predicted reduction is evaluated from the running ``H z`` without an
    extra Hessian action. ``g = 0`` (or ``||g|| <= tolerance``) returns the
    zero step with reason ``"interior"`` and no Hessian action.

    Args:
        gradient: Gradient ``g`` (finite, nonempty).
        hessian_action: ``v -> H v`` for a symmetric ``H`` (any definiteness).
        radius: Trust-region radius ``Delta > 0``.
        tolerance: Absolute residual-norm tolerance ``>= 0``.
        max_iterations: CG iteration cap (default ``len(gradient)``, the exact
            arithmetic termination index of CG).

    Returns:
        :class:`SteihaugResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``gradient`` is empty or not finite, ``radius`` is not
            positive, ``tolerance`` is negative, or the Hessian action returns
            a wrong length.

    References:
        [EQ-OPT-TR-001], [EQ-OPT-TR-002], [REF-STEIHAUG-1983],
        [REF-NOCEDAL-WRIGHT-2006].

    """
    g = _vector(gradient, None, "gradient")
    n = len(g)
    delta = _real(radius, "radius", strict=True)
    tol = _real(tolerance, "tolerance")
    cap = (
        n if max_iterations is None else _positive_int(max_iterations, "max_iterations")
    )
    if not callable(hessian_action):
        raise TypeError("hessian_action must be callable")

    z: Vector = (0.0,) * n
    hz: Vector = (0.0,) * n
    r = g
    rr = dot(r, r)
    if math.sqrt(rr) <= tol:
        return SteihaugResult(z, 0.0, "interior", 0, 0, 0.0)
    d = tuple(-v for v in r)
    actions = 0
    iterations = 0

    def finish(step: Vector, h_step: Vector, reason: str) -> SteihaugResult:
        """Package a step with its model reduction.

        References:
            [EQ-OPT-TR-001].

        """
        reduction = -(dot(g, step) + 0.5 * dot(step, h_step))
        return SteihaugResult(
            step, max(reduction, 0.0), reason, iterations, actions, norm(step)
        )

    for _ in range(cap):
        hd = tuple(float(v) for v in hessian_action(d))
        actions += 1
        if len(hd) != n:
            raise ValueError(f"hessian_action must return length {n}, got {len(hd)}")
        if not all(math.isfinite(v) for v in hd):
            return finish(z, hz, "nonfinite")
        iterations += 1
        curvature = dot(d, hd)
        if curvature <= 0.0:
            tau = _boundary_tau(z, d, delta)
            step = tuple(a + tau * b for a, b in zip(z, d, strict=True))
            h_step = tuple(a + tau * b for a, b in zip(hz, hd, strict=True))
            return finish(step, h_step, "negative_curvature")
        alpha = rr / curvature
        z_new = tuple(a + alpha * b for a, b in zip(z, d, strict=True))
        if norm(z_new) >= delta:
            tau = _boundary_tau(z, d, delta)
            step = tuple(a + tau * b for a, b in zip(z, d, strict=True))
            h_step = tuple(a + tau * b for a, b in zip(hz, hd, strict=True))
            return finish(step, h_step, "boundary")
        z = z_new
        hz = tuple(a + alpha * b for a, b in zip(hz, hd, strict=True))
        r = tuple(a + alpha * b for a, b in zip(r, hd, strict=True))
        rr_new = dot(r, r)
        if math.sqrt(rr_new) <= tol:
            return finish(z, hz, "interior")
        beta = rr_new / rr
        rr = rr_new
        d = tuple(-a + beta * b for a, b in zip(r, d, strict=True))
    return finish(z, hz, "max_iterations")


def trust_region_newton_cg(
    objective: Callable[[Vector], float],
    gradient: Callable[[Vector], Sequence[float]],
    hessian_action: PointHessianAction,
    initial: Sequence[float],
    *,
    radius: float = 1.0,
    max_radius: float = 1e3,
    eta: float = 0.1,
    gradient_tolerance: float = 1e-8,
    max_iterations: int = 200,
    inner_tolerance: float = 0.5,
    forcing: str = "superlinear",
    inner_max_iterations: int | None = None,
    minimum_radius: float = 1e-14,
) -> TrustRegionResult:
    """Minimize a smooth function by trust-region Newton-CG (Steihaug).

    Each attempt solves the subproblem of :func:`steihaug_cg` with the current
    gradient and ``v -> H(x_k) v``, then forms
    ``rho = (f(x_k) - f(x_k + p)) / m_pred`` [EQ-OPT-TR-003]. Following
    Nocedal and Wright Algorithm 4.1 with ``0 <= eta < 1/4``:

    * ``rho < 1/4``: the radius becomes ``||p|| / 4``;
    * ``rho > 3/4`` and ``||p|| = Delta`` (boundary step): the radius becomes
      ``min(2 Delta, max_radius)``;
    * otherwise it is kept;
    * the step is taken iff ``rho > eta``.

    A non-finite trial objective is a rejected step with the radius shrunk
    exactly as for ``rho < 1/4``; the trial value is recorded as ``nan`` and is
    not clipped. If the predicted reduction is below ``16 eps |f|`` the ratio
    is unresolved: the step is then taken iff ``f`` did not increase by more
    than that floor, the radius is kept, and ``rho`` is recorded as ``nan``.

    The inner tolerance is the inexact-Newton forcing term
    ``tol_k = min(inner_tolerance, sqrt(||g_k||)) ||g_k||`` (``"superlinear"``,
    giving superlinear local convergence) or ``inner_tolerance ||g_k||``
    (``"constant"``, linear) [EQ-OPT-TR-004]. Stops, checked each attempt in
    this order: ``"gradient"`` (``||g|| <= gradient_tolerance max(1, ||g_0||)``,
    the only converged reason), ``"max_iterations"`` (attempts),
    ``"radius_collapse"`` (``Delta <= minimum_radius (1 + ||x||)``),
    ``"nonfinite"`` (non-finite gradient or Hessian action at an accepted
    point).

    Args:
        objective: ``f(x)``; may return a non-finite value at trial points.
        gradient: ``x -> grad f(x)``.
        hessian_action: ``(x, v) -> H(x) v`` with symmetric ``H``.
        initial: Start point; ``f`` and ``grad f`` must be finite there.
        radius: Initial radius ``Delta_0 > 0``.
        max_radius: Largest radius ``>= radius``.
        eta: Acceptance threshold in ``[0, 1/4)``.
        gradient_tolerance: Relative gradient tolerance ``>= 0``.
        max_iterations: Cap on attempts ``>= 1``.
        inner_tolerance: Forcing cap / constant ``> 0``.
        forcing: ``"superlinear"`` or ``"constant"``.
        inner_max_iterations: CG cap per attempt (default ``n``).
        minimum_radius: Relative radius floor ``> 0``.

    Returns:
        :class:`TrustRegionResult` with the full attempt history.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an option is out of range or ``f`` or ``grad f`` is not
            finite at ``initial``.

    References:
        [EQ-OPT-TR-003], [EQ-OPT-TR-004], [REF-NOCEDAL-WRIGHT-2006],
        [REF-STEIHAUG-1983].

    """
    for name, fn in (
        ("objective", objective),
        ("gradient", gradient),
        ("hessian_action", hessian_action),
    ):
        if not callable(fn):
            raise TypeError(f"{name} must be callable")
    x = _vector(initial, None, "initial")
    n = len(x)
    delta = _real(radius, "radius", strict=True)
    delta_max = _real(max_radius, "max_radius", strict=True)
    if delta > delta_max:
        raise ValueError("radius must not exceed max_radius")
    accept = _real(eta, "eta")
    if accept >= 0.25:
        raise ValueError("eta must lie in [0, 1/4)")
    gtol = _real(gradient_tolerance, "gradient_tolerance")
    cap = _positive_int(max_iterations, "max_iterations")
    cg_tol = _real(inner_tolerance, "inner_tolerance", strict=True)
    if forcing not in ("superlinear", "constant"):
        raise ValueError("forcing must be 'superlinear' or 'constant'")
    inner_cap = (
        None
        if inner_max_iterations is None
        else _positive_int(inner_max_iterations, "inner_max_iterations")
    )
    floor_radius = _real(minimum_radius, "minimum_radius", strict=True)

    counts = [0, 0, 0, 0]  # objective, gradient, hessian actions, inner iterations

    def evaluate(point: Vector) -> float:
        """Return the counted objective value (possibly non-finite).

        References:
            [EQ-OPT-TR-003].

        """
        counts[0] += 1
        value = objective(point)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("objective must return a real number")
        return float(value)

    def grad_at(point: Vector) -> Vector | None:
        """Return the counted gradient, or ``None`` if not finite.

        References:
            [EQ-OPT-TR-003].

        """
        counts[1] += 1
        out = tuple(float(v) for v in gradient(point))
        if len(out) != n:
            raise ValueError(f"gradient must have length {n}, got {len(out)}")
        return out if all(math.isfinite(v) for v in out) else None

    f = evaluate(x)
    if not math.isfinite(f):
        raise ValueError("objective is not finite at the initial point")
    g = grad_at(x)
    if g is None:
        raise ValueError("gradient is not finite at the initial point")
    g0 = norm(g)
    history: list[TrustRegionIteration] = []
    accepted = 0
    attempts = 0
    reason = "max_iterations"
    gnorm = g0
    while True:
        gnorm = norm(g)
        stop: str | None = None
        if gnorm <= gtol * max(1.0, g0):
            stop = "gradient"
        elif attempts >= cap:
            stop = "max_iterations"
        elif delta <= floor_radius * (1.0 + norm(x)):
            stop = "radius_collapse"
        if stop is not None:
            history.append(
                TrustRegionIteration(
                    attempts,
                    f,
                    gnorm,
                    delta,
                    0.0,
                    math.nan,
                    0.0,
                    math.nan,
                    False,
                    0,
                    "none",
                )
            )
            reason = stop
            break
        point = x

        def action(v: Vector, _point: Vector = point) -> Sequence[float]:
            """Apply the Hessian at the current iterate (counted).

            References:
                [EQ-OPT-TR-001].

            """
            counts[2] += 1
            return hessian_action(_point, v)

        tol_k = (
            min(cg_tol, math.sqrt(gnorm)) * gnorm
            if forcing == "superlinear"
            else cg_tol * gnorm
        )
        sub = steihaug_cg(
            g,
            action,
            delta,
            tolerance=tol_k,
            max_iterations=inner_cap,
        )
        counts[3] += sub.iterations
        if sub.reason == "nonfinite":
            history.append(
                TrustRegionIteration(
                    attempts,
                    f,
                    gnorm,
                    delta,
                    0.0,
                    math.nan,
                    0.0,
                    math.nan,
                    False,
                    sub.iterations,
                    sub.reason,
                )
            )
            reason = "nonfinite"
            break
        p = sub.step
        pnorm = sub.step_norm
        trial = tuple(a + b for a, b in zip(x, p, strict=True))
        f_trial = evaluate(trial) if all(math.isfinite(v) for v in trial) else math.nan
        pred = sub.predicted_reduction
        rho = math.nan
        take = False
        shrink = False
        grow = False
        if not math.isfinite(f_trial):
            shrink = True
            f_record = math.nan
        else:
            f_record = f_trial
            actual = f - f_trial
            floor = 16.0 * _EPS * abs(f)
            if pred > floor:
                rho = actual / pred
                take = rho > accept
                shrink = rho < 0.25
                grow = rho > 0.75 and pnorm >= _BOUNDARY_SLACK * delta
            else:
                take = actual >= -floor and pnorm > 0.0
                shrink = not take
        history.append(
            TrustRegionIteration(
                attempts,
                f,
                gnorm,
                delta,
                pnorm,
                rho,
                pred,
                f_record,
                take,
                sub.iterations,
                sub.reason,
            )
        )
        attempts += 1
        if shrink:
            delta = 0.25 * pnorm if pnorm > 0.0 else 0.25 * delta
        elif grow:
            delta = min(2.0 * delta, delta_max)
        if take:
            new_g = grad_at(trial)
            x, f = trial, f_trial
            accepted += 1
            if new_g is None:
                gnorm = math.nan
                reason = "nonfinite"
                break
            g = new_g
    return TrustRegionResult(
        x=x,
        value=f,
        gradient_norm=gnorm,
        iterations=accepted,
        attempts=attempts,
        converged=reason in _CONVERGED,
        stop_reason=reason,
        radius=delta,
        history=tuple(history),
        objective_evaluations=counts[0],
        gradient_evaluations=counts[1],
        hessian_actions=counts[2],
        inner_iterations=counts[3],
    )
