"""Finite-difference gradients, Jacobians, and Hessians.

Steps scale with the magnitude of each coordinate and are rounded so that
``x_i + h_i`` is exactly representable; when the function fails on one side
of a coordinate, a one-sided formula is used instead. Every routine reports
how many function evaluations it spent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, as_vector, transpose

from ._common import _EPS, ScalarFunction, VectorFunction, _finite_value

_FORWARD_FACTOR = math.sqrt(_EPS)
_CENTRAL_FACTOR = _EPS ** (1.0 / 3.0)


def _per_coordinate(
    value: float | Sequence[float] | None, size: int, name: str, default: float
) -> Vector:
    """Broadcast an optional positive scalar or sequence to one value per axis.

    Args:
        value: ``None`` (use ``default``), a positive scalar, or a sequence of
            positive values of length ``size``.
        size: Number of coordinates.
        name: Parameter name for error messages.
        default: Value used when ``value`` is ``None``.

    Returns:
        Tuple of ``size`` positive finite values.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is not positive and finite or the length differs.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if value is None:
        return (default,) * size
    if isinstance(value, Sequence) and not isinstance(value, str):
        values = tuple(require_finite_real(v, name=name) for v in value)
        if len(values) != size:
            raise ValueError(f"{name} must have one entry per coordinate")
    else:
        values = (require_finite_real(value, name=name),) * size
    if any(v <= 0.0 for v in values):
        raise ValueError(f"{name} must be positive")
    return values


def _difference_steps(
    point: Vector,
    *,
    factor: float,
    relative_step: float | None,
    step: float | Sequence[float] | None,
    typical_scale: float | Sequence[float] | None,
) -> Vector:
    """Return per-coordinate finite-difference steps.

    With an explicit ``step`` the steps are used as given (absolute). Otherwise
    ``h_i = c max(|x_i|, typ_i)`` with ``c = relative_step`` when supplied
    (the pre-0.16 rule ``c max(1, |x_i|)`` is the special case ``typ_i = 1``)
    and ``c = factor`` (``sqrt(eps)`` forward, ``eps^(1/3)`` central) by
    default. The step is then rounded so that ``x_i + h_i`` is exactly
    representable, which removes one rounding error from the quotient.

    Args:
        point: Evaluation point.
        factor: Default relative factor for the difference formula.
        relative_step: Optional legacy relative factor ``c``.
        step: Optional absolute step (scalar or per coordinate).
        typical_scale: Typical magnitude of each coordinate (default 1).

    Returns:
        Positive steps, one per coordinate.

    Raises:
        ValueError: If a step or scale is not positive and finite.

    Equation:
        [EQ-OPT-FD-003].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1],
        [REF-DENNIS-SCHNABEL-1996, Section 5.4].

    """
    size = len(point)
    if step is not None:
        raw = _per_coordinate(step, size, "step", 1.0)
    else:
        if relative_step is None:
            multiplier = factor
        else:
            multiplier = require_finite_real(relative_step, name="relative_step")
            if multiplier <= 0.0:
                raise ValueError("relative_step must be positive")
        scales = _per_coordinate(typical_scale, size, "typical_scale", 1.0)
        raw = tuple(
            multiplier * max(abs(x), s) for x, s in zip(point, scales, strict=True)
        )
    steps: list[float] = []
    for x, h in zip(point, raw, strict=True):
        exact = (x + h) - x
        steps.append(exact if exact > 0.0 else h)
    return tuple(steps)


def _try_value(function: ScalarFunction, point: Vector) -> float | None:
    """Evaluate a scalar function, returning ``None`` on failure.

    A ``ValueError`` or ``ArithmeticError`` from the function, or a non-finite
    result, counts as a failure (for example a point outside the domain).

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1]. No equation is evaluated.

    """
    try:
        value = function(point)
    except (ValueError, ArithmeticError):
        return None
    if isinstance(value, bool):
        raise TypeError("objective must return a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("objective must return a real number") from error
    return result if math.isfinite(result) else None


def _try_vector(function: VectorFunction, point: Vector) -> Vector | None:
    """Evaluate a vector function, returning ``None`` on failure.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1]. No equation is evaluated.

    """
    try:
        raw = function(point)
    except (ValueError, ArithmeticError):
        return None
    values = tuple(float(v) for v in raw)
    return values if all(math.isfinite(v) for v in values) else None


def _moved(point: Vector, index: int, delta: float) -> Vector:
    """Return ``point`` with coordinate ``index`` shifted by ``delta``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    moved = list(point)
    moved[index] += delta
    return tuple(moved)


def _one_sided(
    evaluate: Callable[[Vector], Vector | None],
    point: Vector,
    centre: Vector,
    index: int,
    h: float,
    near: Vector,
    direction: float,
) -> tuple[Vector, int]:
    """Return a one-sided derivative estimate along one axis.

    Uses the second-order formula ``(-3 f0 + 4 f1 - f2) / (2 h)`` (with the
    sign of ``direction``) when ``f(x + 2 s h)`` is finite and the first-order
    quotient ``(f1 - f0) / (s h)`` otherwise.

    Args:
        evaluate: Function returning the (vector) value or ``None``.
        point: Evaluation point.
        centre: Function value at ``point``.
        index: Coordinate index.
        h: Positive step.
        near: Value at ``point + s h e_i``.
        direction: ``+1`` for forward and ``-1`` for backward differences.

    Returns:
        Derivative estimate per output component and evaluations used.

    Equation:
        [EQ-OPT-FD-004].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1],
        [REF-FORNBERG-1988].

    """
    far = evaluate(_moved(point, index, 2.0 * direction * h))
    if far is not None and len(far) == len(centre):
        return (
            tuple(
                direction * (-3.0 * c + 4.0 * n - f) / (2.0 * h)
                for c, n, f in zip(centre, near, far, strict=True)
            ),
            1,
        )
    return (
        tuple(direction * (n - c) / h for c, n in zip(centre, near, strict=True)),
        1,
    )


def _difference_columns(
    evaluate: Callable[[Vector], Vector | None],
    point: Vector,
    steps: Vector,
    *,
    central: bool,
    centre: Vector | None,
) -> tuple[list[Vector], int]:
    """Return derivative columns along every axis with one-sided fallback.

    Args:
        evaluate: Function returning the (vector) value or ``None`` on failure.
        point: Evaluation point.
        steps: Positive step per coordinate.
        central: Use central differences when both sides are finite.
        centre: Known value at ``point`` or ``None``.

    Returns:
        One derivative vector per coordinate and the evaluation count.

    Raises:
        ValueError: If the function fails on both sides of a coordinate or at
            ``point`` when a one-sided formula needs it.

    Equation:
        [EQ-OPT-FD-001], [EQ-OPT-FD-004].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    evaluations = 0
    columns: list[Vector] = []
    base = centre

    def at_centre() -> Vector:
        """Return (and cache) the value at the evaluation point.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1]. No equation is evaluated.

        """
        nonlocal base, evaluations
        if base is None:
            evaluations += 1
            base = evaluate(point)
            if base is None:
                raise ValueError("function is not finite at the evaluation point")
        return base

    for index, h in enumerate(steps):
        upper = evaluate(_moved(point, index, h))
        evaluations += 1
        if central:
            lower = evaluate(_moved(point, index, -h))
            evaluations += 1
            if upper is not None and lower is not None:
                if len(upper) != len(lower):
                    raise ValueError("vector function changed output length")
                columns.append(
                    tuple(
                        (u - v) / (2.0 * h) for u, v in zip(upper, lower, strict=True)
                    )
                )
                continue
        else:
            lower = None
            if upper is not None:
                value = at_centre()
                if len(upper) != len(value):
                    raise ValueError("vector function changed output length")
                columns.append(
                    tuple((u - c) / h for u, c in zip(upper, value, strict=True))
                )
                continue
            lower = evaluate(_moved(point, index, -h))
            evaluations += 1
        if upper is None and lower is None:
            raise ValueError(
                f"function is not finite on either side of coordinate {index}; "
                "supply a smaller step"
            )
        value = at_centre()
        if upper is not None:
            column, used = _one_sided(evaluate, point, value, index, h, upper, 1.0)
        else:
            assert lower is not None
            column, used = _one_sided(evaluate, point, value, index, h, lower, -1.0)
        evaluations += used
        columns.append(column)
    return columns, evaluations


def central_gradient(
    function: ScalarFunction,
    x: Sequence[float],
    *,
    relative_step: float | None = None,
    step: float | Sequence[float] | None = None,
    typical_scale: float | Sequence[float] | None = None,
) -> tuple[Vector, int]:
    """Approximate a gradient by central differences with one-sided fallback.

    The default step is ``h_i = eps^(1/3) max(|x_i|, typ_i)``, which balances
    the ``O(h^2)`` truncation and ``O(eps/h)`` rounding errors of the central
    quotient. When the function raises ``ValueError`` (or an arithmetic
    error) or returns a non-finite value on one side, the second-order
    one-sided formula on the other side is used (first-order if ``x + 2h``
    also fails), so gradients next to a domain boundary remain available.

    Args:
        function: Scalar function of a vector.
        x: Finite evaluation point.
        relative_step: Optional relative factor ``c`` in ``h_i = c max(|x_i|,
            typ_i)``; ``None`` uses ``eps^(1/3)``. Passing ``1e-6`` with the
            default typical scale reproduces the rule of releases before 0.16.
        step: Optional absolute step (scalar or one per coordinate);
            overrides the relative rule.
        typical_scale: Typical magnitude ``typ_i`` of each coordinate
            (default ``1``). Supply it for parameters on scales far from one,
            for example ``1e-6``, so the step is not much larger than the
            parameter itself.

    Returns:
        Gradient approximation and the number of function evaluations.

    Raises:
        ValueError: If the function fails on both sides of a coordinate.

    Equation:
        [EQ-OPT-FD-001], [EQ-OPT-FD-003], [EQ-OPT-FD-004].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1],
        [REF-DENNIS-SCHNABEL-1996, Section 5.4].

    """
    point = as_vector(x, name="x")
    steps = _difference_steps(
        point,
        factor=_CENTRAL_FACTOR,
        relative_step=relative_step,
        step=step,
        typical_scale=typical_scale,
    )

    def evaluate(at: Vector) -> Vector | None:
        """Wrap the scalar function as a one-component vector function.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1]. No equation is evaluated.

        """
        value = _try_value(function, at)
        return None if value is None else (value,)

    columns, evaluations = _difference_columns(
        evaluate, point, steps, central=True, centre=None
    )
    return tuple(column[0] for column in columns), evaluations


def forward_gradient(
    function: ScalarFunction,
    x: Sequence[float],
    *,
    value: float | None = None,
    step: float | Sequence[float] | None = None,
    typical_scale: float | Sequence[float] | None = None,
) -> tuple[Vector, int]:
    """Approximate a gradient by forward differences with backward fallback.

    The default step is ``h_i = sqrt(eps) max(|x_i|, typ_i)``; the error is
    ``O(h) + O(eps/h)``, about ``1e-8`` relative, at roughly half the cost of
    central differences. A failing forward evaluation falls back to the
    one-sided backward formula.

    Args:
        function: Scalar function of a vector.
        x: Finite evaluation point.
        value: Optional known value ``f(x)``, saving one evaluation.
        step: Optional absolute step (scalar or one per coordinate).
        typical_scale: Typical magnitude of each coordinate (default ``1``).

    Returns:
        Gradient approximation and the number of function evaluations.

    Raises:
        ValueError: If the function fails at ``x`` or on both sides.

    Equation:
        [EQ-OPT-FD-003], [EQ-OPT-FD-004].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1],
        [REF-DENNIS-SCHNABEL-1996, Section 5.4].

    """
    point = as_vector(x, name="x")
    steps = _difference_steps(
        point,
        factor=_FORWARD_FACTOR,
        relative_step=None,
        step=step,
        typical_scale=typical_scale,
    )

    def evaluate(at: Vector) -> Vector | None:
        """Wrap the scalar function as a one-component vector function.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1]. No equation is evaluated.

        """
        result = _try_value(function, at)
        return None if result is None else (result,)

    centre = None if value is None else (require_finite_real(value, name="value"),)
    columns, evaluations = _difference_columns(
        evaluate, point, steps, central=False, centre=centre
    )
    return tuple(column[0] for column in columns), evaluations


def central_jacobian(
    function: VectorFunction,
    x: Sequence[float],
    *,
    relative_step: float | None = None,
    step: float | Sequence[float] | None = None,
    typical_scale: float | Sequence[float] | None = None,
) -> tuple[Matrix, int]:
    """Approximate the Jacobian of a vector function by central differences.

    Steps and the one-sided fallback follow :func:`central_gradient`.

    Args:
        function: Vector-valued function of a vector.
        x: Finite evaluation point.
        relative_step: Optional relative factor; ``None`` uses ``eps^(1/3)``.
        step: Optional absolute step (scalar or one per coordinate).
        typical_scale: Typical magnitude of each coordinate (default ``1``).

    Returns:
        ``m`` by ``n`` Jacobian approximation and evaluation count.

    Raises:
        ValueError: If the function fails on both sides of a coordinate or
            changes its output length.

    Equation:
        [EQ-OPT-FD-001], [EQ-OPT-FD-003], [EQ-OPT-FD-004].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    point = as_vector(x, name="x")
    steps = _difference_steps(
        point,
        factor=_CENTRAL_FACTOR,
        relative_step=relative_step,
        step=step,
        typical_scale=typical_scale,
    )

    def evaluate(at: Vector) -> Vector | None:
        """Evaluate the vector function, ``None`` on failure.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1]. No equation is evaluated.

        """
        return _try_vector(function, at)

    columns, evaluations = _difference_columns(
        evaluate, point, steps, central=True, centre=None
    )
    return transpose(tuple(columns)), evaluations


def central_hessian(
    function: ScalarFunction,
    x: Sequence[float],
    *,
    relative_step: float = 1e-4,
    typical_scale: float | Sequence[float] | None = None,
) -> tuple[Matrix, int]:
    """Approximate a symmetric Hessian by second-order central differences.

    Args:
        function: Twice-differentiable scalar function.
        x: Finite evaluation point.
        relative_step: Step relative to ``max(|x_i|, typ_i)`` (about
            ``eps^(1/4)`` by default, the optimum for second differences).
        typical_scale: Typical magnitude of each coordinate (default ``1``).

    Returns:
        Symmetric Hessian approximation and evaluation count.

    Equation:
        [EQ-OPT-FD-002].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    point = as_vector(x, name="x")
    size = len(point)
    steps = _difference_steps(
        point,
        factor=relative_step,
        relative_step=relative_step,
        step=None,
        typical_scale=typical_scale,
    )
    center = _finite_value(function(point), "f(x)")
    evaluations = 1
    hessian = [[0.0] * size for _ in range(size)]

    def shifted(i: int, si: float, j: int | None = None, sj: float = 0.0) -> float:
        """Evaluate the function at a point shifted along one or two axes.

        References:
            [EQ-OPT-FD-002], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        nonlocal evaluations
        moved = list(point)
        moved[i] += si * steps[i]
        if j is not None:
            moved[j] += sj * steps[j]
        evaluations += 1
        return _finite_value(function(tuple(moved)), "f(shifted x)")

    for i in range(size):
        plus = shifted(i, 1.0)
        minus = shifted(i, -1.0)
        hessian[i][i] = (plus - 2.0 * center + minus) / (steps[i] ** 2)
        for j in range(i + 1, size):
            value = (
                shifted(i, 1.0, j, 1.0)
                - shifted(i, 1.0, j, -1.0)
                - shifted(i, -1.0, j, 1.0)
                + shifted(i, -1.0, j, -1.0)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i][j] = hessian[j][i] = value
    return tuple(tuple(row) for row in hessian), evaluations
