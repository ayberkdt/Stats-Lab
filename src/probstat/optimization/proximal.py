"""Proximal operators and first-order splitting methods for convex problems.

Proximal maps (``prox_{t g}(v) = argmin_x t g(x) + 1/2 ||x - v||^2``) and
projections:

- :func:`soft_threshold`, :func:`prox_l1` (weights, centre and box),
  :func:`prox_nonnegative_l1`, :func:`project_box`,
  :func:`prox_elastic_net` [EQ-OPT-PROX-001]-[EQ-OPT-PROX-003];
- :func:`prox_group_l2` for DISJOINT groups (overlapping groups are
  rejected) [EQ-OPT-PROX-004];
- :func:`project_l2_ball`, :func:`project_linf_ball`,
  :func:`project_pointwise_l2_ball` and :func:`prox_pointwise_l2` (the
  isotropic total-variation dual projection and its Moreau partner)
  [EQ-OPT-PROX-005].

Solvers (all matrix-free, all report an explicit ``stop_reason``; only a met
tolerance sets ``converged``):

- :func:`ista` and :func:`fista` for ``min f(x) + g(x)`` with ``f`` smooth
  (value and gradient callbacks) and ``g`` with a known prox, with Beck-Teboulle
  backtracking [EQ-OPT-PROX-006] and the gradient-mapping norm as optimality
  metric [EQ-OPT-PROX-007]; FISTA has optional function-value restart;
- :func:`primal_dual_hybrid_gradient` (Chambolle-Pock) for
  ``min f(x) + g(Kx)`` with ``tau sigma ||K||^2 < 1`` checked against a
  user-supplied operator-norm BOUND [EQ-OPT-PROX-008], [EQ-OPT-PROX-009];
- :func:`admm` (scaled form) for ``min f(x) + g(z)`` subject to ``L x = z``
  with a user-supplied x-update, primal and dual residuals and optional
  residual balancing [EQ-OPT-PROX-010], [EQ-OPT-PROX-011].

No spectral bound is trusted implicitly: ``power_iteration`` of
``probstat.linalg`` is a LOWER estimate, so a step derived from it must carry
an explicit safety factor chosen by the caller.

Experimental (added in 0.28): the API may change before it is declared stable.

References:
    [REF-BECK-TEBOULLE-2009], [REF-CHAMBOLLE-POCK-2011],
    [REF-BOYD-ETAL-2011], [REF-DAUBECHIES-DEFRISE-DEMOL-2004].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.linalg import (
    CountingOperator,
    LinearOperator,
    OperatorCounts,
    Vector,
    require_operator,
)

ProxMap = Callable[[Vector, float], Vector]
"""A proximal map called as ``prox(v, step)`` returning ``prox_{step g}(v)``."""

_BACKTRACK_SLACK = 1e-12


class _NonFiniteError(ArithmeticError):
    """Internal signal: a callback or an update produced a non-finite value.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """


def _as_vector(values: Sequence[float], name: str, size: int | None = None) -> Vector:
    """Return ``values`` as a finite float tuple, optionally of a given length.

    Args:
        values: Candidate vector.
        name: Name used in error messages.
        size: Required length, or ``None``.

    Returns:
        Tuple of finite floats.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If the length differs or an entry is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError(f"{name} must be a sequence of real numbers")
    items = tuple(values)
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain only real numbers")
    result = tuple(float(item) for item in items)
    if size is not None and len(result) != size:
        raise ValueError(f"{name} must have length {size}, got {len(result)}")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value: float, name: str) -> float:
    """Return a finite real scalar.

    Args:
        value: Candidate scalar.
        name: Name used in error messages.

    Returns:
        The scalar as a float.

    Raises:
        TypeError: If it is not real.
        ValueError: If it is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    """Return a finite non-negative scalar.

    Args:
        value: Candidate scalar.
        name: Name used in error messages.

    Returns:
        The scalar as a float.

    Raises:
        TypeError: If it is not real.
        ValueError: If it is negative or not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    result = _scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: float, name: str) -> float:
    """Return a finite strictly positive scalar.

    Args:
        value: Candidate scalar.
        name: Name used in error messages.

    Returns:
        The scalar as a float.

    Raises:
        TypeError: If it is not real.
        ValueError: If it is not strictly positive or not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    result = _scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(value: int, name: str) -> int:
    """Return a strictly positive integer.

    Args:
        value: Candidate integer.
        name: Name used in error messages.

    Returns:
        The integer.

    Raises:
        TypeError: If it is not an integer.
        ValueError: If it is not positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _per_entry(
    value: float | Sequence[float] | None,
    size: int,
    name: str,
    default: float,
    *,
    nonnegative: bool = False,
) -> Vector:
    """Broadcast a scalar, a sequence or ``None`` to a vector of length ``size``.

    Args:
        value: Scalar, sequence of length ``size`` or ``None``.
        size: Target length.
        name: Name used in error messages.
        default: Entry used when ``value`` is ``None``.
        nonnegative: Whether entries must be non-negative.

    Returns:
        Tuple of ``size`` floats.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the length is wrong or an entry is invalid.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if value is None:
        result: Vector = (default,) * size
    elif isinstance(value, (int, float)):
        result = (_scalar(value, name),) * size
    else:
        result = _as_vector(value, name, size)
    if nonnegative and any(item < 0.0 for item in result):
        raise ValueError(f"{name} must be non-negative")
    return result


def _norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm.

    Args:
        vector: Finite vector.

    Returns:
        ``||v||_2``.

    References:
        [EQ-OPT-PROX-007].

    """
    return math.hypot(*vector)


def _check_vector(values: Sequence[float], size: int, name: str) -> Vector:
    """Validate a callback output; non-finite entries raise the internal signal.

    Args:
        values: Callback output.
        size: Required length.
        name: Callback name for messages.

    Returns:
        The output as a tuple of floats.

    Raises:
        TypeError: If the output is not a numeric sequence.
        ValueError: If the length is wrong.
        _NonFiniteError: If an entry is NaN or infinite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    try:
        result = tuple(float(item) for item in values)
    except TypeError as error:
        raise TypeError(f"{name} must return a sequence of real numbers") from error
    if len(result) != size:
        raise ValueError(f"{name} returned length {len(result)}, expected {size}")
    if not all(math.isfinite(item) for item in result):
        raise _NonFiniteError(name)
    return result


def _check_scalar(value: float, name: str) -> float:
    """Validate a callback scalar output.

    Args:
        value: Callback output.
        name: Callback name for messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the output is not real.
        _NonFiniteError: If it is NaN or infinite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must return a real number")
    result = float(value)
    if not math.isfinite(result):
        raise _NonFiniteError(name)
    return result


# ---------------------------------------------------------------------------
# Proximal maps and projections
# ---------------------------------------------------------------------------


def soft_threshold(
    vector: Sequence[float], threshold: float | Sequence[float]
) -> Vector:
    """Return the soft-thresholding ``sign(v) max(|v| - theta, 0)``.

    This is ``prox_{t g}`` for ``g = ||.||_1`` with ``theta = t`` (and for
    ``g = sum_i w_i |x_i|`` with ``theta_i = t w_i``) [EQ-OPT-PROX-001].

    Args:
        vector: Input ``v``.
        threshold: Non-negative scalar or per-entry thresholds.

    Returns:
        The thresholded vector.

    Raises:
        TypeError: If an input is not real.
        ValueError: If a threshold is negative or lengths differ.

    References:
        [EQ-OPT-PROX-001], [REF-DAUBECHIES-DEFRISE-DEMOL-2004].

    """
    v = _as_vector(vector, "vector")
    theta = _per_entry(threshold, len(v), "threshold", 0.0, nonnegative=True)
    return tuple(
        math.copysign(max(abs(a) - t, 0.0), a) if abs(a) > t else 0.0
        for a, t in zip(v, theta, strict=True)
    )


def prox_l1(
    vector: Sequence[float],
    step: float,
    *,
    weights: float | Sequence[float] | None = None,
    center: Sequence[float] | None = None,
    lower: float | Sequence[float] | None = None,
    upper: float | Sequence[float] | None = None,
) -> Vector:
    """Return ``argmin_x step sum_i w_i |x_i - c_i| + 1/2||x - v||^2`` on a box.

    The unconstrained minimiser is ``c + soft(v - c, step w)``. Because the
    problem is separable and each scalar function is convex, the constrained
    minimiser over ``[lower, upper]`` is that value clipped to the box
    [EQ-OPT-PROX-001], [EQ-OPT-PROX-002].

    Args:
        vector: Input ``v``.
        step: Non-negative step ``t``.
        weights: Non-negative scalar or per-entry weights ``w`` (default 1).
        center: Centre ``c`` of the penalty (default zero).
        lower: Lower bound (scalar, per entry or ``None``).
        upper: Upper bound (scalar, per entry or ``None``).

    Returns:
        The proximal point.

    Raises:
        TypeError: If an input is not real.
        ValueError: If a length differs, ``step`` or a weight is negative, or
            ``lower > upper`` somewhere.

    References:
        [EQ-OPT-PROX-001], [EQ-OPT-PROX-002].

    """
    v = _as_vector(vector, "vector")
    n = len(v)
    t = _nonnegative(step, "step")
    w = _per_entry(weights, n, "weights", 1.0, nonnegative=True)
    c = _per_entry(center, n, "center", 0.0)
    lo = _per_entry(lower, n, "lower", -math.inf)
    hi = _per_entry(upper, n, "upper", math.inf)
    if any(a > b for a, b in zip(lo, hi, strict=True)):
        raise ValueError("lower must not exceed upper")
    out: list[float] = []
    for vi, wi, ci, li, ui in zip(v, w, c, lo, hi, strict=True):
        shifted = vi - ci
        magnitude = max(abs(shifted) - t * wi, 0.0)
        value = ci + math.copysign(magnitude, shifted) if magnitude > 0.0 else ci
        out.append(min(max(value, li), ui))
    return tuple(out)


def prox_nonnegative_l1(
    vector: Sequence[float],
    step: float,
    *,
    weights: float | Sequence[float] | None = None,
) -> Vector:
    """Return the prox of ``step sum w_i x_i + indicator(x >= 0)``.

    On the nonnegative orthant ``|x_i| = x_i``, so the proximal point is
    ``max(v - step w, 0)`` [EQ-OPT-PROX-002].

    Args:
        vector: Input ``v``.
        step: Non-negative step.
        weights: Non-negative scalar or per-entry weights (default 1).

    Returns:
        ``max(v - step w, 0)`` entrywise.

    Raises:
        TypeError: If an input is not real.
        ValueError: If a length differs or step or a weight is negative.

    References:
        [EQ-OPT-PROX-002].

    """
    v = _as_vector(vector, "vector")
    t = _nonnegative(step, "step")
    w = _per_entry(weights, len(v), "weights", 1.0, nonnegative=True)
    return tuple(max(a - t * b, 0.0) for a, b in zip(v, w, strict=True))


def project_box(
    vector: Sequence[float],
    lower: float | Sequence[float] | None = None,
    upper: float | Sequence[float] | None = None,
) -> Vector:
    """Return the Euclidean projection onto ``{lower <= x <= upper}``.

    Args:
        vector: Input ``v``.
        lower: Lower bound (scalar, per entry or ``None`` for ``-inf``).
        upper: Upper bound (scalar, per entry or ``None`` for ``+inf``).

    Returns:
        ``min(max(v, lower), upper)`` entrywise.

    Raises:
        TypeError: If an input is not real.
        ValueError: If a length differs or ``lower > upper`` somewhere.

    References:
        [EQ-OPT-PROX-002].

    """
    v = _as_vector(vector, "vector")
    lo = _per_entry(lower, len(v), "lower", -math.inf)
    hi = _per_entry(upper, len(v), "upper", math.inf)
    if any(a > b for a, b in zip(lo, hi, strict=True)):
        raise ValueError("lower must not exceed upper")
    return tuple(min(max(a, li), ui) for a, li, ui in zip(v, lo, hi, strict=True))


def prox_elastic_net(
    vector: Sequence[float],
    step: float,
    *,
    l1: float = 0.0,
    l2: float = 0.0,
    weights: float | Sequence[float] | None = None,
) -> Vector:
    """Return the prox of ``step (l1 sum w_i |x_i| + l2/2 ||x||^2)``.

    The proximal point is ``soft(v, step l1 w) / (1 + step l2)``
    [EQ-OPT-PROX-003].

    Args:
        vector: Input ``v``.
        step: Non-negative step.
        l1: Non-negative L1 coefficient.
        l2: Non-negative coefficient of the quadratic ``l2/2 ||x||^2``.
        weights: Non-negative scalar or per-entry L1 weights (default 1).

    Returns:
        The proximal point.

    Raises:
        TypeError: If an input is not real.
        ValueError: If a coefficient, weight or the step is negative.

    References:
        [EQ-OPT-PROX-003].

    """
    v = _as_vector(vector, "vector")
    t = _nonnegative(step, "step")
    a = _nonnegative(l1, "l1")
    b = _nonnegative(l2, "l2")
    w = _per_entry(weights, len(v), "weights", 1.0, nonnegative=True)
    shrunk = soft_threshold(v, [t * a * wi for wi in w])
    scale = 1.0 + t * b
    return tuple(item / scale for item in shrunk)


def _validated_groups(
    groups: Sequence[Sequence[int]], size: int
) -> tuple[tuple[int, ...], ...]:
    """Validate disjoint index groups.

    Args:
        groups: Index groups.
        size: Vector length.

    Returns:
        The groups as tuples.

    Raises:
        TypeError: If an index is not an integer.
        ValueError: If a group is empty, an index is out of range, or two
            groups (or one group twice) share an index.

    References:
        [EQ-OPT-PROX-004].

    """
    seen: set[int] = set()
    checked: list[tuple[int, ...]] = []
    for group in groups:
        members = tuple(group)
        if not members:
            raise ValueError("groups must be non-empty")
        for index in members:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("group indices must be integers")
            if not 0 <= index < size:
                raise ValueError(f"group index {index} outside [0, {size})")
            if index in seen:
                raise ValueError(
                    f"overlapping groups are not supported (index {index} repeats)"
                )
            seen.add(index)
        checked.append(members)
    return tuple(checked)


def prox_group_l2(
    vector: Sequence[float],
    step: float,
    groups: Sequence[Sequence[int]],
    *,
    weights: Sequence[float] | None = None,
) -> Vector:
    """Return the prox of the group penalty ``step sum_g w_g ||x_g||_2``.

    For each group ``g`` the block is scaled by
    ``max(0, 1 - step w_g / ||v_g||)`` (zero when ``v_g = 0``). Groups must be
    DISJOINT; overlapping groups need a different algorithm and are rejected.
    Indices that belong to no group are not penalised and pass through
    unchanged [EQ-OPT-PROX-004].

    Args:
        vector: Input ``v``.
        step: Non-negative step.
        groups: Disjoint lists of indices.
        weights: Optional non-negative weight per group (default 1).

    Returns:
        The proximal point.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If groups overlap, are empty or out of range, or a weight
            or the step is negative.

    References:
        [EQ-OPT-PROX-004].

    """
    v = _as_vector(vector, "vector")
    t = _nonnegative(step, "step")
    checked = _validated_groups(groups, len(v))
    w = _per_entry(weights, len(checked), "weights", 1.0, nonnegative=True)
    out = list(v)
    for members, wg in zip(checked, w, strict=True):
        nrm = _norm([v[i] for i in members])
        factor = max(0.0, 1.0 - t * wg / nrm) if nrm > 0.0 else 0.0
        for i in members:
            out[i] = v[i] * factor
    return tuple(out)


def project_l2_ball(vector: Sequence[float], radius: float) -> Vector:
    """Return the projection onto ``{||x||_2 <= radius}``.

    Args:
        vector: Input ``v``.
        radius: Non-negative radius.

    Returns:
        ``v`` if inside, else ``radius v / ||v||``.

    Raises:
        TypeError: If an input is not real.
        ValueError: If the radius is negative.

    References:
        [EQ-OPT-PROX-005].

    """
    v = _as_vector(vector, "vector")
    r = _nonnegative(radius, "radius")
    nrm = _norm(v)
    if nrm <= r:
        return v
    return tuple(r * a / nrm for a in v)


def project_linf_ball(
    vector: Sequence[float], radius: float | Sequence[float]
) -> Vector:
    """Return the projection onto ``{|x_i| <= r_i}`` (clipping).

    This is the dual projection of the anisotropic total variation.

    Args:
        vector: Input ``v``.
        radius: Non-negative scalar or per-entry radii.

    Returns:
        ``clip(v, -r, r)``.

    Raises:
        TypeError: If an input is not real.
        ValueError: If a radius is negative or lengths differ.

    References:
        [EQ-OPT-PROX-005].

    """
    v = _as_vector(vector, "vector")
    r = _per_entry(radius, len(v), "radius", 0.0, nonnegative=True)
    return tuple(min(max(a, -b), b) for a, b in zip(v, r, strict=True))


def project_pointwise_l2_ball(
    vector: Sequence[float],
    radius: float | Sequence[float],
    *,
    components: int = 2,
) -> Vector:
    """Return the pointwise projection onto the Euclidean balls ``|p_i| <= r_i``.

    The vector is block-stacked with ``components`` blocks of ``N`` entries,
    ``[p^1; p^2; ...]`` (the layout of ``GridGradientOperator`` for
    ``components=2``); the ``i``-th point is ``(p^1_i, ..., p^c_i)`` and is
    projected onto the ball of radius ``r_i`` [EQ-OPT-PROX-005]. This is the
    dual projection of the isotropic total variation.

    Args:
        vector: Block-stacked input of length ``components * N``.
        radius: Non-negative scalar or per-point radii of length ``N``.
        components: Number of stacked blocks.

    Returns:
        The projected vector.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If the length is not a multiple of ``components`` or a
            radius is negative.

    References:
        [EQ-OPT-PROX-005], [REF-CHAMBOLLE-POCK-2011].

    """
    v = _as_vector(vector, "vector")
    c = _positive_int(components, "components")
    if len(v) % c != 0:
        raise ValueError("vector length must be a multiple of components")
    count = len(v) // c
    r = _per_entry(radius, count, "radius", 0.0, nonnegative=True)
    out = list(v)
    for i in range(count):
        nrm = math.hypot(*(v[k * count + i] for k in range(c)))
        if nrm > r[i]:
            scale = r[i] / nrm
            for k in range(c):
                out[k * count + i] = v[k * count + i] * scale
    return tuple(out)


def prox_pointwise_l2(
    vector: Sequence[float],
    step: float,
    *,
    components: int = 2,
    weights: float | Sequence[float] | None = None,
) -> Vector:
    """Return the prox of ``step sum_i w_i |p_i|`` with ``|.|`` the 2-norm.

    By the Moreau identity ``prox_{t h}(v) = v - proj_{ball(t w)}(v)`` for the
    pointwise norm ``h``, using :func:`project_pointwise_l2_ball`
    [EQ-OPT-PROX-005].

    Args:
        vector: Block-stacked input of length ``components * N``.
        step: Non-negative step.
        components: Number of stacked blocks.
        weights: Non-negative scalar or per-point weights (default 1).

    Returns:
        The proximal point.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If lengths are inconsistent or a weight is negative.

    References:
        [EQ-OPT-PROX-005].

    """
    v = _as_vector(vector, "vector")
    t = _nonnegative(step, "step")
    c = _positive_int(components, "components")
    if len(v) % c != 0:
        raise ValueError("vector length must be a multiple of components")
    count = len(v) // c
    w = _per_entry(weights, count, "weights", 1.0, nonnegative=True)
    projected = project_pointwise_l2_ball(v, [t * item for item in w], components=c)
    return tuple(a - b for a, b in zip(v, projected, strict=True))


# ---------------------------------------------------------------------------
# ISTA / FISTA
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProximalGradientResult:
    """Outcome of :func:`ista` or :func:`fista`.

    Attributes:
        solution: Returned iterate (the last accepted iterate; on
            ``"tolerance"`` the iterate whose gradient mapping met it).
        objective: ``f(x) + g(x)`` at ``solution``.
        objective_history: Objective values from the start point, one per
            iteration (not monotone for FISTA unless restarted).
        gradient_mapping_norm: ``||x - prox_{t g}(x - t grad f(x))|| / t`` at
            ``solution`` [EQ-OPT-PROX-007].
        gradient_mapping_history: The mapping norm at the point where the
            gradient was taken in each iteration (the extrapolated point for
            FISTA).
        iterations: Completed iterations.
        stop_reason: ``"tolerance"``, ``"max_iterations"``,
            ``"line_search_failed"`` or ``"non_finite"``.
        converged: ``True`` only for ``"tolerance"``.
        step_size: Final step ``t`` (``1 / L`` for the accepted backtracked
            Lipschitz estimate).
        restarts: Number of function-value restarts (FISTA).
        value_evaluations: Calls of the smooth value.
        gradient_evaluations: Calls of the smooth gradient.
        prox_evaluations: Calls of the prox.
        backtracks: Total step reductions.
        method: ``"ista"`` or ``"fista"``.

    References:
        [EQ-OPT-PROX-006], [EQ-OPT-PROX-007], [REF-BECK-TEBOULLE-2009].

    """

    solution: Vector
    objective: float
    objective_history: Vector
    gradient_mapping_norm: float
    gradient_mapping_history: Vector
    iterations: int
    stop_reason: str
    converged: bool
    step_size: float
    restarts: int
    value_evaluations: int
    gradient_evaluations: int
    prox_evaluations: int
    backtracks: int
    method: str


def _proximal_gradient(
    smooth_value: Callable[[Vector], float],
    smooth_gradient: Callable[[Vector], Vector],
    nonsmooth_value: Callable[[Vector], float],
    prox: ProxMap,
    initial: Sequence[float],
    *,
    accelerated: bool,
    lipschitz: float | None,
    initial_step: float,
    backtrack_factor: float,
    max_backtracks: int,
    tolerance: float,
    max_iterations: int,
    restart: bool,
) -> ProximalGradientResult:
    """Run the shared proximal-gradient loop (ISTA or FISTA).

    Args:
        smooth_value: ``f(x)``.
        smooth_gradient: ``grad f(x)``.
        nonsmooth_value: ``g(x)``.
        prox: ``prox(v, t) = prox_{t g}(v)``.
        initial: Start point.
        accelerated: FISTA momentum when ``True``.
        lipschitz: Fixed ``L`` (step ``1/L``) or ``None`` for backtracking.
        initial_step: First trial step when backtracking.
        backtrack_factor: Shrink factor in ``(0, 1)``.
        max_backtracks: Cap on consecutive reductions in one iteration.
        tolerance: Gradient-mapping tolerance.
        max_iterations: Iteration cap.
        restart: Function-value restart (FISTA only).

    Returns:
        The result record.

    References:
        [EQ-OPT-PROX-006], [EQ-OPT-PROX-007], [REF-BECK-TEBOULLE-2009].

    """
    x0 = _as_vector(initial, "initial")
    n = len(x0)
    counts = {"f": 0, "grad": 0, "prox": 0, "bt": 0}

    def value(point: Vector) -> float:
        """Count and evaluate ``f``.

        References:
            [EQ-OPT-PROX-006].

        """
        counts["f"] += 1
        return _check_scalar(smooth_value(point), "smooth_value")

    def gradient(point: Vector) -> Vector:
        """Count and evaluate ``grad f``.

        References:
            [EQ-OPT-PROX-006].

        """
        counts["grad"] += 1
        return _check_vector(smooth_gradient(point), n, "smooth_gradient")

    def proximal(point: Vector, t: float) -> Vector:
        """Count and evaluate the prox.

        References:
            [EQ-OPT-PROX-006].

        """
        counts["prox"] += 1
        return _check_vector(prox(point, t), n, "prox")

    def forward(point: Vector, grad: Vector, t: float) -> Vector:
        """Return ``prox_{t g}(point - t grad)``.

        References:
            [EQ-OPT-PROX-006].

        """
        return proximal(tuple(a - t * b for a, b in zip(point, grad, strict=True)), t)

    def mapping(point: Vector, t: float) -> float:
        """Return the gradient-mapping norm at ``point``.

        References:
            [EQ-OPT-PROX-007].

        """
        nxt = forward(point, gradient(point), t)
        return _norm([a - b for a, b in zip(point, nxt, strict=True)]) / t

    fixed = lipschitz is not None
    step = 1.0 / lipschitz if lipschitz is not None else initial_step
    x = x0
    y = x0
    theta = 1.0
    fx = 0.0
    obj = math.nan
    obj_hist: list[float] = []
    map_hist: list[float] = []
    restarts = 0
    iterations = 0
    reason = "max_iterations"
    last_map = math.nan
    method = "fista" if accelerated else "ista"
    try:
        fx = value(x)
        obj = fx + _check_scalar(nonsmooth_value(x), "nonsmooth_value")
        obj_hist.append(obj)
        fy = fx
        for k in range(1, max_iterations + 1):
            grad = gradient(y)
            streak = 0
            while True:
                cand = forward(y, grad, step)
                diff = [a - b for a, b in zip(cand, y, strict=True)]
                fc = value(cand)
                if fixed:
                    break
                model = (
                    fy
                    + math.fsum(a * b for a, b in zip(grad, diff, strict=True))
                    + math.fsum(d * d for d in diff) / (2.0 * step)
                )
                if fc <= model + _BACKTRACK_SLACK * (1.0 + abs(fy)):
                    break
                step *= backtrack_factor
                counts["bt"] += 1
                streak += 1
                if streak > max_backtracks or step < 1e-300:
                    reason = "line_search_failed"
                    break
            if reason == "line_search_failed":
                break
            mapping_y = _norm(diff) / step
            map_hist.append(mapping_y)
            last_map = mapping_y
            if mapping_y <= tolerance:
                if not accelerated:
                    iterations = k - 1
                    reason = "tolerance"
                    break
                # FISTA: certify at the iterate, not at the extrapolated point.
                at_cand = mapping(cand, step)
                if at_cand <= tolerance:
                    x, fx = cand, fc
                    obj = fc + _check_scalar(nonsmooth_value(cand), "nonsmooth_value")
                    obj_hist.append(obj)
                    last_map = at_cand
                    iterations = k
                    reason = "tolerance"
                    break
            new_obj = fc + _check_scalar(nonsmooth_value(cand), "nonsmooth_value")
            if accelerated:
                if restart and new_obj > obj:
                    theta = 1.0
                    y = cand
                    fy = fc
                    restarts += 1
                else:
                    theta_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * theta * theta))
                    beta = (theta - 1.0) / theta_next
                    y = tuple(a + beta * (a - b) for a, b in zip(cand, x, strict=True))
                    theta = theta_next
                    fy = value(y) if beta != 0.0 else fc
            else:
                y = cand
                fy = fc
            x, fx, obj = cand, fc, new_obj
            obj_hist.append(obj)
            iterations = k
        else:
            reason = "max_iterations"
        if reason in ("max_iterations", "line_search_failed"):
            last_map = mapping(x, step)
    except _NonFiniteError:
        reason = "non_finite"
        last_map = math.nan
    return ProximalGradientResult(
        solution=x,
        objective=obj,
        objective_history=tuple(obj_hist),
        gradient_mapping_norm=last_map,
        gradient_mapping_history=tuple(map_hist),
        iterations=iterations,
        stop_reason=reason,
        converged=reason == "tolerance",
        step_size=step,
        restarts=restarts,
        value_evaluations=counts["f"],
        gradient_evaluations=counts["grad"],
        prox_evaluations=counts["prox"],
        backtracks=counts["bt"],
        method=method,
    )


def _proximal_gradient_checked(
    smooth_value: Callable[[Vector], float],
    smooth_gradient: Callable[[Vector], Vector],
    nonsmooth_value: Callable[[Vector], float],
    prox: ProxMap,
    initial: Sequence[float],
    *,
    accelerated: bool,
    lipschitz: float | None,
    initial_step: float,
    backtrack_factor: float,
    max_backtracks: int,
    tolerance: float,
    max_iterations: int,
    restart: bool,
) -> ProximalGradientResult:
    """Validate the arguments of :func:`ista` and :func:`fista`, then run.

    Args:
        smooth_value: ``f(x)``.
        smooth_gradient: ``grad f(x)``.
        nonsmooth_value: ``g(x)``.
        prox: ``prox(v, t)``.
        initial: Start point.
        accelerated: FISTA momentum when ``True``.
        lipschitz: Fixed ``L`` or ``None``.
        initial_step: First trial step.
        backtrack_factor: Shrink factor.
        max_backtracks: Cap on consecutive reductions.
        tolerance: Gradient-mapping tolerance.
        max_iterations: Iteration cap.
        restart: Function-value restart.

    Returns:
        The result record.

    Raises:
        TypeError: If a callback is not callable or an argument has the wrong
            type.
        ValueError: If a numeric argument is out of range.

    References:
        [EQ-OPT-PROX-006].

    """
    for name, func in (
        ("smooth_value", smooth_value),
        ("smooth_gradient", smooth_gradient),
        ("nonsmooth_value", nonsmooth_value),
        ("prox", prox),
    ):
        if not callable(func):
            raise TypeError(f"{name} must be callable")
    if lipschitz is not None:
        lipschitz = _positive(lipschitz, "lipschitz")
    initial_step = _positive(initial_step, "initial_step")
    backtrack_factor = _scalar(backtrack_factor, "backtrack_factor")
    if not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must lie in (0, 1)")
    return _proximal_gradient(
        smooth_value,
        smooth_gradient,
        nonsmooth_value,
        prox,
        initial,
        accelerated=accelerated,
        lipschitz=lipschitz,
        initial_step=initial_step,
        backtrack_factor=backtrack_factor,
        max_backtracks=_positive_int(max_backtracks, "max_backtracks"),
        tolerance=_nonnegative(tolerance, "tolerance"),
        max_iterations=_positive_int(max_iterations, "max_iterations"),
        restart=bool(restart),
    )


def ista(
    smooth_value: Callable[[Vector], float],
    smooth_gradient: Callable[[Vector], Vector],
    nonsmooth_value: Callable[[Vector], float],
    prox: ProxMap,
    initial: Sequence[float],
    *,
    lipschitz: float | None = None,
    initial_step: float = 1.0,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 60,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
) -> ProximalGradientResult:
    """Minimise ``f(x) + g(x)`` by the proximal gradient method (ISTA).

    Each iteration sets ``x+ = prox_{t g}(x - t grad f(x))``. Without a user
    ``lipschitz`` the step ``t`` is found by Beck-Teboulle backtracking: it is
    reduced by ``backtrack_factor`` until
    ``f(x+) <= f(x) + <grad f(x), x+ - x> + ||x+ - x||^2 / (2 t)`` holds (up
    to a relative slack of ``1e-12 (1 + |f(x)|)`` that absorbs rounding); ``t``
    never increases, so no spectral bound is needed [EQ-OPT-PROX-006]. With a
    user ``lipschitz`` the step is the fixed ``1 / L`` and nothing is checked;
    an under-estimated ``L`` voids the convergence guarantee. The objective is
    monotone non-increasing with backtracking. Optimality is measured by the
    gradient-mapping norm ``||x - prox_{t g}(x - t grad f(x))|| / t``
    [EQ-OPT-PROX-007]; the returned solution is the iterate at which it met
    ``tolerance``.

    Args:
        smooth_value: ``f(x)`` for the smooth part.
        smooth_gradient: ``grad f(x)``.
        nonsmooth_value: ``g(x)`` (used for the reported objective only).
        prox: ``prox(v, t) = prox_{t g}(v)``.
        initial: Start point.
        lipschitz: Optional Lipschitz constant of ``grad f`` (fixed step
            ``1 / L``, no backtracking).
        initial_step: First trial step when backtracking.
        backtrack_factor: Step shrink factor in ``(0, 1)``.
        max_backtracks: Cap on consecutive reductions per iteration.
        tolerance: Non-negative absolute tolerance on the gradient mapping.
        max_iterations: Positive iteration cap.

    Returns:
        :class:`ProximalGradientResult`; ``converged`` is ``False`` when the
        iteration cap is hit.

    Raises:
        TypeError: If a callback is not callable or an argument has the wrong
            type.
        ValueError: If a numeric argument is out of range or a callback
            returns the wrong length.

    References:
        [EQ-OPT-PROX-006], [EQ-OPT-PROX-007], [REF-BECK-TEBOULLE-2009],
        [REF-DAUBECHIES-DEFRISE-DEMOL-2004].

    """
    return _proximal_gradient_checked(
        smooth_value,
        smooth_gradient,
        nonsmooth_value,
        prox,
        initial,
        accelerated=False,
        lipschitz=lipschitz,
        initial_step=initial_step,
        backtrack_factor=backtrack_factor,
        max_backtracks=max_backtracks,
        tolerance=tolerance,
        max_iterations=max_iterations,
        restart=False,
    )


def fista(
    smooth_value: Callable[[Vector], float],
    smooth_gradient: Callable[[Vector], Vector],
    nonsmooth_value: Callable[[Vector], float],
    prox: ProxMap,
    initial: Sequence[float],
    *,
    lipschitz: float | None = None,
    initial_step: float = 1.0,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 60,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    restart: bool = True,
) -> ProximalGradientResult:
    """Minimise ``f(x) + g(x)`` by FISTA with backtracking and optional restart.

    With extrapolated point ``y_k`` and momentum scalars ``theta_k``
    (``theta_1 = 1``): ``x_k = prox_{t g}(y_k - t grad f(y_k))``,
    ``theta_(k+1) = (1 + sqrt(1 + 4 theta_k^2)) / 2`` and
    ``y_(k+1) = x_k + (theta_k - 1)/theta_(k+1) (x_k - x_(k-1))``
    [EQ-OPT-PROX-006]. The step is backtracked as in :func:`ista`. The
    objective is NOT monotone in general. With ``restart=True`` the momentum
    is reset (``theta = 1``, ``y = x_k``) whenever the objective increases
    (adaptive function-value restart); this is a heuristic that does not
    change the fixed point. Convergence is declared only when the gradient
    mapping at the ITERATE ``x_k`` (not at the extrapolated point) meets
    ``tolerance`` [EQ-OPT-PROX-007]; this check costs one extra gradient
    evaluation.

    Args:
        smooth_value: ``f(x)`` for the smooth part.
        smooth_gradient: ``grad f(x)``.
        nonsmooth_value: ``g(x)`` (used for the reported objective only).
        prox: ``prox(v, t) = prox_{t g}(v)``.
        initial: Start point.
        lipschitz: Optional Lipschitz constant of ``grad f``.
        initial_step: First trial step when backtracking.
        backtrack_factor: Step shrink factor in ``(0, 1)``.
        max_backtracks: Cap on consecutive reductions per iteration.
        tolerance: Non-negative absolute tolerance on the gradient mapping.
        max_iterations: Positive iteration cap.
        restart: Enable the function-value restart.

    Returns:
        :class:`ProximalGradientResult`; ``converged`` is ``False`` when the
        iteration cap is hit.

    Raises:
        TypeError: If a callback is not callable or an argument has the wrong
            type.
        ValueError: If a numeric argument is out of range or a callback
            returns the wrong length.

    References:
        [EQ-OPT-PROX-006], [EQ-OPT-PROX-007], [REF-BECK-TEBOULLE-2009].

    """
    return _proximal_gradient_checked(
        smooth_value,
        smooth_gradient,
        nonsmooth_value,
        prox,
        initial,
        accelerated=True,
        lipschitz=lipschitz,
        initial_step=initial_step,
        backtrack_factor=backtrack_factor,
        max_backtracks=max_backtracks,
        tolerance=tolerance,
        max_iterations=max_iterations,
        restart=restart,
    )


# ---------------------------------------------------------------------------
# Primal-dual hybrid gradient (Chambolle-Pock)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PDHGResult:
    """Outcome of :func:`primal_dual_hybrid_gradient`.

    Attributes:
        solution: Primal iterate ``x``.
        dual: Dual iterate ``y``.
        objective: Primal objective at ``solution`` (``nan`` without an
            objective callback).
        objective_history: Objective values from the start (empty without a
            callback).
        primal_residual_norm: ``||p||`` of [EQ-OPT-PROX-009] at the last
            iteration.
        dual_residual_norm: ``||d||`` of [EQ-OPT-PROX-009] at the last
            iteration.
        primal_residual_history: ``||p_k||`` per iteration.
        dual_residual_history: ``||d_k||`` per iteration.
        iterations: Completed iterations.
        stop_reason: ``"tolerance"``, ``"max_iterations"`` or
            ``"non_finite"``.
        converged: ``True`` only for ``"tolerance"``.
        tau: Primal step.
        sigma: Dual step.
        theta: Overrelaxation parameter.
        step_product: ``tau sigma L^2`` with ``L`` the supplied norm bound
            (strictly below one).
        operator_counts: Forward and transpose applications of ``K``.

    References:
        [EQ-OPT-PROX-008], [EQ-OPT-PROX-009], [REF-CHAMBOLLE-POCK-2011].

    """

    solution: Vector
    dual: Vector
    objective: float
    objective_history: Vector
    primal_residual_norm: float
    dual_residual_norm: float
    primal_residual_history: Vector
    dual_residual_history: Vector
    iterations: int
    stop_reason: str
    converged: bool
    tau: float
    sigma: float
    theta: float
    step_product: float
    operator_counts: OperatorCounts


def primal_dual_hybrid_gradient(
    operator: LinearOperator,
    prox_primal: ProxMap,
    prox_dual: ProxMap,
    initial: Sequence[float],
    *,
    operator_norm_bound: float,
    tau: float | None = None,
    sigma: float | None = None,
    step_safety: float = 0.99,
    theta: float = 1.0,
    initial_dual: Sequence[float] | None = None,
    objective: Callable[[Vector, Vector], float] | None = None,
    tolerance_absolute: float = 1e-8,
    tolerance_relative: float = 1e-6,
    max_iterations: int = 10000,
) -> PDHGResult:
    """Solve ``min_x f(x) + g(K x)`` by the Chambolle-Pock iteration.

    With ``prox_primal(v, tau) = prox_{tau f}(v)`` and
    ``prox_dual(q, sigma) = prox_{sigma g*}(q)`` (the conjugate's prox; obtain
    it from ``prox_g`` through the Moreau identity
    ``prox_{sigma g*}(q) = q - sigma prox_{g/sigma}(q/sigma)``), each
    iteration is ``x+ = prox_{tau f}(x - tau K' y)``,
    ``xbar = x+ + theta (x+ - x)``, ``y+ = prox_{sigma g*}(y + sigma K xbar)``
    [EQ-OPT-PROX-008]. Convergence needs ``tau sigma ||K||^2 < 1``. The caller
    must pass ``operator_norm_bound``, a number KNOWN to be at least
    ``||K||_2`` (a closed-form bound, a Frobenius norm, a product of bounds).
    ``probstat.linalg.power_iteration`` returns a LOWER estimate; if only that
    is available, pass ``safety * estimate`` with an explicit ``safety > 1``
    chosen by you, and treat the guarantee as heuristic. The product
    ``tau sigma L^2`` is validated to be strictly below one and is reported.

    The stopping rule uses the residuals ``p`` and ``d`` of
    [EQ-OPT-PROX-009]; ``||p||`` is compared with
    ``sqrt(n) abs + rel max(||(x - x+)/tau||, ||K'y+||)`` and ``||d||`` with
    the analogous dual scale. No primal-dual gap is reported.

    Args:
        operator: ``K`` (``m x n``), applied once forward and once
            transposed per iteration.
        prox_primal: ``prox_{tau f}``.
        prox_dual: ``prox_{sigma g*}``.
        initial: Start ``x_0`` of length ``n``.
        operator_norm_bound: Certified upper bound ``L >= ||K||_2``.
        tau: Primal step; derived with ``sigma`` when omitted.
        sigma: Dual step; derived with ``tau`` when omitted. With both
            omitted ``tau = sigma = step_safety / L``; with one given the
            other is ``step_safety**2 / (given L^2)``.
        step_safety: Factor in ``(0, 1)`` used for derived steps.
        theta: Overrelaxation in ``[0, 1]`` (``1`` is the standard scheme).
        initial_dual: Start ``y_0`` (zero by default).
        objective: Optional ``objective(x, K x)`` for the history (no extra
            operator application is spent).
        tolerance_absolute: Non-negative absolute residual tolerance.
        tolerance_relative: Non-negative relative residual tolerance.
        max_iterations: Positive iteration cap.

    Returns:
        :class:`PDHGResult`; ``converged`` is ``False`` when the cap is hit.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a step or bound is not positive, ``tau sigma L^2 >= 1``,
            ``theta`` is outside ``[0, 1]`` or a size is inconsistent.

    References:
        [EQ-OPT-PROX-008], [EQ-OPT-PROX-009], [REF-CHAMBOLLE-POCK-2011],
        [REF-CHAMBOLLE-POCK-2016].

    """
    counted = CountingOperator(require_operator(operator, name="operator"))
    m, n = counted.shape
    if not callable(prox_primal) or not callable(prox_dual):
        raise TypeError("prox_primal and prox_dual must be callable")
    bound = _positive(operator_norm_bound, "operator_norm_bound")
    safety = _scalar(step_safety, "step_safety")
    if not 0.0 < safety < 1.0:
        raise ValueError("step_safety must lie in (0, 1)")
    if tau is None and sigma is None:
        tau_v = sigma_v = safety / bound
    elif tau is not None and sigma is None:
        tau_v = _positive(tau, "tau")
        sigma_v = safety * safety / (tau_v * bound * bound)
    elif sigma is not None and tau is None:
        sigma_v = _positive(sigma, "sigma")
        tau_v = safety * safety / (sigma_v * bound * bound)
    elif tau is not None and sigma is not None:
        tau_v = _positive(tau, "tau")
        sigma_v = _positive(sigma, "sigma")
    else:  # pragma: no cover - the four cases above are exhaustive
        raise AssertionError("unreachable")
    product = tau_v * sigma_v * bound * bound
    if not product < 1.0:
        raise ValueError(
            f"step condition tau*sigma*L^2 < 1 violated (product = {product:.6g})"
        )
    over = _scalar(theta, "theta")
    if not 0.0 <= over <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    atol = _nonnegative(tolerance_absolute, "tolerance_absolute")
    rtol = _nonnegative(tolerance_relative, "tolerance_relative")
    limit = _positive_int(max_iterations, "max_iterations")
    x = _as_vector(initial, "initial", n)
    y = (
        (0.0,) * m
        if initial_dual is None
        else _as_vector(initial_dual, "initial_dual", m)
    )

    obj_hist: list[float] = []
    p_hist: list[float] = []
    d_hist: list[float] = []
    reason = "max_iterations"
    iterations = 0
    p_norm = d_norm = math.nan
    obj = math.nan
    try:
        kx = counted.matvec(x)
        kty = counted.rmatvec(y)
        if objective is not None:
            obj = _check_scalar(objective(x, kx), "objective")
            obj_hist.append(obj)
        for k in range(1, limit + 1):
            x_new = _check_vector(
                prox_primal(
                    tuple(a - tau_v * b for a, b in zip(x, kty, strict=True)), tau_v
                ),
                n,
                "prox_primal",
            )
            kx_new = counted.matvec(x_new)
            kbar = tuple(a + over * (a - b) for a, b in zip(kx_new, kx, strict=True))
            y_new = _check_vector(
                prox_dual(
                    tuple(a + sigma_v * b for a, b in zip(y, kbar, strict=True)),
                    sigma_v,
                ),
                m,
                "prox_dual",
            )
            kty_new = counted.rmatvec(y_new)
            dx = [(a - b) / tau_v for a, b in zip(x, x_new, strict=True)]
            dy = [(a - b) / sigma_v for a, b in zip(y, y_new, strict=True)]
            p_vec = [a - (b - c) for a, b, c in zip(dx, kty, kty_new, strict=True)]
            d_vec = [a - over * (b - c) for a, b, c in zip(dy, kx, kx_new, strict=True)]
            p_norm = _norm(p_vec)
            d_norm = _norm(d_vec)
            if not (math.isfinite(p_norm) and math.isfinite(d_norm)):
                raise _NonFiniteError("residual")
            p_hist.append(p_norm)
            d_hist.append(d_norm)
            x, y, kx, kty = x_new, y_new, kx_new, kty_new
            iterations = k
            if objective is not None:
                obj = _check_scalar(objective(x, kx), "objective")
                obj_hist.append(obj)
            p_tol = math.sqrt(n) * atol + rtol * max(_norm(dx), _norm(kty))
            d_tol = math.sqrt(m) * atol + rtol * max(_norm(dy), _norm(kx))
            if p_norm <= p_tol and d_norm <= d_tol:
                reason = "tolerance"
                break
    except _NonFiniteError:
        reason = "non_finite"
    return PDHGResult(
        solution=x,
        dual=y,
        objective=obj,
        objective_history=tuple(obj_hist),
        primal_residual_norm=p_norm,
        dual_residual_norm=d_norm,
        primal_residual_history=tuple(p_hist),
        dual_residual_history=tuple(d_hist),
        iterations=iterations,
        stop_reason=reason,
        converged=reason == "tolerance",
        tau=tau_v,
        sigma=sigma_v,
        theta=over,
        step_product=product,
        operator_counts=counted.counts,
    )


# ---------------------------------------------------------------------------
# ADMM
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ADMMResult:
    """Outcome of :func:`admm`.

    Attributes:
        solution: Primal iterate ``x``.
        split: The auxiliary variable ``z`` (``L x = z`` at a solution).
        scaled_dual: Scaled dual ``u`` (the unscaled multiplier is ``rho u``).
        objective: ``objective(x)`` at ``solution`` (``nan`` without a
            callback).
        objective_history: Objective per iteration (empty without a callback).
        primal_residual_norm: ``||r||``, ``r = L x - z`` [EQ-OPT-PROX-011].
        dual_residual_norm: ``||s||``, ``s = rho L'(z - z_old)``.
        primal_residual_history: ``||r_k||`` per iteration.
        dual_residual_history: ``||s_k||`` per iteration.
        iterations: Completed iterations.
        stop_reason: ``"tolerance"``, ``"max_iterations"`` or
            ``"non_finite"``.
        converged: ``True`` only for ``"tolerance"``.
        rho: Final penalty parameter.
        rho_updates: Number of residual-balancing changes of ``rho``.
        operator_counts: Forward and transpose applications of ``L`` made by
            the solver itself (the x-update callback has its own cost).

    References:
        [EQ-OPT-PROX-010], [EQ-OPT-PROX-011], [REF-BOYD-ETAL-2011].

    """

    solution: Vector
    split: Vector
    scaled_dual: Vector
    objective: float
    objective_history: Vector
    primal_residual_norm: float
    dual_residual_norm: float
    primal_residual_history: Vector
    dual_residual_history: Vector
    iterations: int
    stop_reason: str
    converged: bool
    rho: float
    rho_updates: int
    operator_counts: OperatorCounts


def admm(
    operator: LinearOperator,
    x_update: Callable[[Vector, float, Vector], Vector],
    prox_split: ProxMap,
    initial: Sequence[float],
    *,
    rho: float = 1.0,
    objective: Callable[[Vector], float] | None = None,
    tolerance_absolute: float = 1e-8,
    tolerance_relative: float = 1e-6,
    adapt_rho: bool = False,
    adapt_threshold: float = 10.0,
    adapt_factor: float = 2.0,
    max_rho_updates: int = 20,
    max_iterations: int = 1000,
) -> ADMMResult:
    """Solve ``min f(x) + g(z)`` subject to ``L x = z`` by scaled ADMM.

    One iteration is ``x+ = argmin_x f(x) + rho/2 ||L x - z + u||^2`` (the
    user callback ``x_update(v, rho, x_old)`` with ``v = z - u``),
    ``z+ = prox_{g/rho}(L x+ + u)`` (the callback ``prox_split(w, 1/rho)``)
    and ``u+ = u + L x+ - z+`` [EQ-OPT-PROX-010]. The residuals are
    ``r = L x+ - z+`` and ``s = rho L'(z+ - z)``; the run stops when
    ``||r|| <= sqrt(m) abs + rel max(||L x||, ||z||)`` and
    ``||s|| <= sqrt(n) abs + rel ||rho L' u||`` [EQ-OPT-PROX-011]. With
    ``adapt_rho`` the penalty is multiplied or divided by ``adapt_factor``
    when one residual exceeds ``adapt_threshold`` times the other, and the
    scaled dual ``u`` is rescaled so the unscaled multiplier ``rho u`` is
    preserved; adaptation stops after ``max_rho_updates`` changes (fixed
    ``rho`` afterwards keeps the convergence theory valid). The
    ``x_update`` callback is responsible for its own inner accuracy; a loose
    inner solve can stall or bias the outer iteration.

    Args:
        operator: ``L`` (``m x n``).
        x_update: ``x_update(v, rho, x_old)`` returning the x-subproblem
            minimiser.
        prox_split: ``prox_split(w, t) = prox_{t g}(w)`` (called with
            ``t = 1 / rho``).
        initial: Start ``x_0``; ``z_0 = L x_0`` and ``u_0 = 0``.
        rho: Positive penalty parameter.
        objective: Optional ``objective(x)`` evaluated each iteration.
        tolerance_absolute: Non-negative absolute tolerance.
        tolerance_relative: Non-negative relative tolerance.
        adapt_rho: Enable residual balancing.
        adapt_threshold: Imbalance ratio ``> 1`` that triggers a change.
        adapt_factor: Factor ``> 1`` applied to ``rho``.
        max_rho_updates: Cap on the number of changes.
        max_iterations: Positive iteration cap.

    Returns:
        :class:`ADMMResult`; ``converged`` is ``False`` when the cap is hit.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a numeric argument is out of range or a callback
            returns the wrong length.

    References:
        [EQ-OPT-PROX-010], [EQ-OPT-PROX-011], [REF-BOYD-ETAL-2011].

    """
    counted = CountingOperator(require_operator(operator, name="operator"))
    m, n = counted.shape
    if not callable(x_update) or not callable(prox_split):
        raise TypeError("x_update and prox_split must be callable")
    penalty = _positive(rho, "rho")
    atol = _nonnegative(tolerance_absolute, "tolerance_absolute")
    rtol = _nonnegative(tolerance_relative, "tolerance_relative")
    mu = _scalar(adapt_threshold, "adapt_threshold")
    factor = _scalar(adapt_factor, "adapt_factor")
    if mu <= 1.0 or factor <= 1.0:
        raise ValueError("adapt_threshold and adapt_factor must exceed 1")
    cap_updates = _positive_int(max_rho_updates, "max_rho_updates")
    limit = _positive_int(max_iterations, "max_iterations")
    x = _as_vector(initial, "initial", n)

    obj_hist: list[float] = []
    r_hist: list[float] = []
    s_hist: list[float] = []
    reason = "max_iterations"
    iterations = 0
    updates = 0
    r_norm = s_norm = math.nan
    obj = math.nan
    z: Vector = counted.matvec(x)
    u: Vector = (0.0,) * m
    try:
        for k in range(1, limit + 1):
            v = tuple(a - b for a, b in zip(z, u, strict=True))
            x = _check_vector(x_update(v, penalty, x), n, "x_update")
            lx = counted.matvec(x)
            z_old = z
            z = _check_vector(
                prox_split(
                    tuple(a + b for a, b in zip(lx, u, strict=True)), 1.0 / penalty
                ),
                m,
                "prox_split",
            )
            r_vec = [a - b for a, b in zip(lx, z, strict=True)]
            u = tuple(a + b for a, b in zip(u, r_vec, strict=True))
            s_vec = [
                penalty * a
                for a in counted.rmatvec(
                    tuple(a - b for a, b in zip(z, z_old, strict=True))
                )
            ]
            r_norm = _norm(r_vec)
            s_norm = _norm(s_vec)
            if not (math.isfinite(r_norm) and math.isfinite(s_norm)):
                raise _NonFiniteError("residual")
            r_hist.append(r_norm)
            s_hist.append(s_norm)
            iterations = k
            if objective is not None:
                obj = _check_scalar(objective(x), "objective")
                obj_hist.append(obj)
            eps_primal = math.sqrt(m) * atol + rtol * max(_norm(lx), _norm(z))
            eps_dual = math.sqrt(n) * atol + rtol * penalty * _norm(counted.rmatvec(u))
            if r_norm <= eps_primal and s_norm <= eps_dual:
                reason = "tolerance"
                break
            if adapt_rho and updates < cap_updates:
                if r_norm > mu * s_norm:
                    penalty *= factor
                    u = tuple(a / factor for a in u)
                    updates += 1
                elif s_norm > mu * r_norm:
                    penalty /= factor
                    u = tuple(a * factor for a in u)
                    updates += 1
    except _NonFiniteError:
        reason = "non_finite"
    return ADMMResult(
        solution=x,
        split=z,
        scaled_dual=u,
        objective=obj,
        objective_history=tuple(obj_hist),
        primal_residual_norm=r_norm,
        dual_residual_norm=s_norm,
        primal_residual_history=tuple(r_hist),
        dual_residual_history=tuple(s_hist),
        iterations=iterations,
        stop_reason=reason,
        converged=reason == "tolerance",
        rho=penalty,
        rho_updates=updates,
        operator_counts=counted.counts,
    )
