"""Dependency-free numerical optimization, root finding, and differentiation.

These solvers are deliberately small and transparent: each returns its
evaluation count and a convergence flag instead of silently returning the last
iterate. They serve reliability (design points), inverse problems (least
squares and MAP estimates), and surrogate hyperparameter fitting.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    add,
    as_vector,
    dot,
    identity,
    inverse_spd,
    matmul,
    matvec,
    norm,
    outer,
    scale_matrix,
    solve_spd,
    transpose,
)

ScalarFunction = Callable[[Vector], float]
VectorFunction = Callable[[Vector], Sequence[float]]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Outcome of an unconstrained minimization.

    Args:
        x: Final iterate.
        value: Objective value at ``x``.
        gradient_norm: Euclidean gradient norm at ``x`` (``nan`` if unused).
        iterations: Completed iterations.
        evaluations: Objective evaluations, including finite differences.
        converged: Whether a stopping tolerance was met.
        message: Human-readable termination reason.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 2].

    """

    x: Vector
    value: float
    gradient_norm: float
    iterations: int
    evaluations: int
    converged: bool
    message: str


@dataclass(frozen=True, slots=True)
class LeastSquaresResult:
    """Outcome of a nonlinear least-squares fit.

    Args:
        x: Final parameter vector.
        residuals: Residual vector ``r(x)``.
        cost: Half the residual sum of squares ``0.5 ||r||^2``.
        jacobian: Residual Jacobian at ``x``.
        iterations: Completed Levenberg-Marquardt iterations.
        evaluations: Residual-function evaluations.
        converged: Whether a stopping tolerance was met.
        message: Termination reason.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 10], [REF-MORE-1978].

    """

    x: Vector
    residuals: Vector
    cost: float
    jacobian: Matrix
    iterations: int
    evaluations: int
    converged: bool
    message: str


@dataclass(frozen=True, slots=True)
class RootResult:
    """Outcome of a bracketing scalar root search.

    Args:
        root: Approximate root.
        value: Function value at ``root``.
        iterations: Completed iterations.
        evaluations: Function evaluations.
        converged: Whether the bracket width or residual tolerance was met.

    References:
        [REF-BRENT-1973, Chapter 4].

    """

    root: float
    value: float
    iterations: int
    evaluations: int
    converged: bool


def _finite_value(value: object, name: str) -> float:
    """Validate one finite objective or residual value.

    References:
        Project validation policy. No equation is evaluated.

    """
    return require_finite_real(value, name=name)


_EPS = 2.0**-52
_FORWARD_FACTOR = math.sqrt(_EPS)
_CENTRAL_FACTOR = _EPS ** (1.0 / 3.0)
# Dennis & Schnabel's suggested relative step tolerance eps^(2/3).
_DEFAULT_STEP_TOLERANCE = _EPS ** (2.0 / 3.0)
_WOLFE_C1 = 1e-4
_WOLFE_C2 = 0.9
_MAX_LINE_SEARCH = 60
# A predicted quasi-Newton decrease below this many ulps of |f| cannot be
# resolved in floating point.
_ROUNDING_FACTOR = 16.0
Bounds = Sequence[tuple[float | None, float | None]]


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
        Project validation policy. No equation is evaluated.

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
        Project validation policy. No equation is evaluated.

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


@dataclass(frozen=True, slots=True)
class BoxTransform:
    """Smooth map between unconstrained coordinates ``z`` and a box for ``x``.

    Per coordinate with bounds ``(l, u)``:

    * two finite bounds: ``x = l + (u - l)(1 + sin z)/2``;
    * lower bound only: ``x = l - 1 + sqrt(z^2 + 1)``;
    * upper bound only: ``x = u + 1 - sqrt(z^2 + 1)``;
    * no bound: ``x = z``.

    These are the parameter transformations of MINUIT. Every ``z`` maps
    inside the box, so the objective is never evaluated outside it, and a
    minimum on a bound becomes an ordinary stationary point in ``z`` (where
    ``dx/dz = 0``). The price is a distorted metric near the bounds; start
    points exactly on a bound are moved inward (see :meth:`to_inner`).

    Args:
        lower: Lower bound per coordinate (``-inf`` when absent).
        upper: Upper bound per coordinate (``inf`` when absent).

    References:
        [REF-JAMES-ROOS-1975].

    """

    lower: Vector
    upper: Vector

    @classmethod
    def from_bounds(cls, bounds: Bounds, size: int) -> BoxTransform:
        """Validate ``(low, high)`` pairs (``None`` or infinite for no bound).

        Args:
            bounds: One pair per coordinate.
            size: Expected number of coordinates.

        Returns:
            Validated transform.

        Raises:
            ValueError: If the length differs or a pair has ``low >= high``.

        References:
            Project validation policy. No equation is evaluated.

        """
        pairs = tuple(bounds)
        if len(pairs) != size:
            raise ValueError("bounds must have one (low, high) pair per coordinate")
        lower: list[float] = []
        upper: list[float] = []
        for index, pair in enumerate(pairs):
            if len(pair) != 2:
                raise ValueError("each bound must be a (low, high) pair")
            low_raw, high_raw = pair
            low = -math.inf if low_raw is None else float(low_raw)
            high = math.inf if high_raw is None else float(high_raw)
            if math.isnan(low) or math.isnan(high) or low == math.inf:
                raise ValueError(f"invalid bound for coordinate {index}")
            if high == -math.inf or not low < high:
                raise ValueError(f"bounds for coordinate {index} need low < high")
            lower.append(low)
            upper.append(high)
        return cls(tuple(lower), tuple(upper))

    def to_inner(self, x: Vector) -> Vector:
        """Map a point in the closed box to unconstrained coordinates.

        A coordinate on (or outside) a bound is first moved inside by
        ``1e-3`` of the interval width (two-sided) or ``1e-3 max(1, |b|)``
        (one-sided), because ``dx/dz`` vanishes on the bound.

        Args:
            x: Point inside or on the boundary of the box.

        Returns:
            Unconstrained coordinates ``z``.

        Raises:
            ValueError: If a coordinate lies outside the box.

        Equation:
            [EQ-OPT-BOX-001].

        References:
            [REF-JAMES-ROOS-1975].

        """
        inner: list[float] = []
        for value, low, high in zip(x, self.lower, self.upper, strict=True):
            if value < low or value > high:
                raise ValueError("x0 must lie inside the bounds")
            finite_low, finite_high = math.isfinite(low), math.isfinite(high)
            if finite_low and finite_high:
                margin = 1e-3 * (high - low)
                value = min(max(value, low + margin), high - margin)
                ratio = 2.0 * (value - low) / (high - low) - 1.0
                inner.append(math.asin(min(1.0, max(-1.0, ratio))))
            elif finite_low:
                value = max(value, low + 1e-3 * max(1.0, abs(low)))
                inner.append(math.sqrt((value - low + 1.0) ** 2 - 1.0))
            elif finite_high:
                value = min(value, high - 1e-3 * max(1.0, abs(high)))
                inner.append(math.sqrt((high - value + 1.0) ** 2 - 1.0))
            else:
                inner.append(value)
        return tuple(inner)

    def to_outer(self, z: Vector) -> Vector:
        """Map unconstrained coordinates to the box (clamped against rounding).

        Args:
            z: Unconstrained coordinates.

        Returns:
            Point inside the closed box.

        Equation:
            [EQ-OPT-BOX-001].

        References:
            [REF-JAMES-ROOS-1975].

        """
        outer: list[float] = []
        for value, low, high in zip(z, self.lower, self.upper, strict=True):
            finite_low, finite_high = math.isfinite(low), math.isfinite(high)
            if finite_low and finite_high:
                x = low + 0.5 * (high - low) * (1.0 + math.sin(value))
            elif finite_low:
                x = low - 1.0 + math.hypot(value, 1.0)
            elif finite_high:
                x = high + 1.0 - math.hypot(value, 1.0)
            else:
                x = value
            outer.append(min(max(x, low), high))
        return tuple(outer)

    def derivative(self, z: Vector) -> Vector:
        """Return the diagonal Jacobian ``dx_i/dz_i``.

        Args:
            z: Unconstrained coordinates.

        Returns:
            Derivative of each coordinate map.

        Equation:
            [EQ-OPT-BOX-001].

        References:
            [REF-JAMES-ROOS-1975].

        """
        result: list[float] = []
        for value, low, high in zip(z, self.lower, self.upper, strict=True):
            finite_low, finite_high = math.isfinite(low), math.isfinite(high)
            if finite_low and finite_high:
                result.append(0.5 * (high - low) * math.cos(value))
            elif finite_low:
                result.append(value / math.hypot(value, 1.0))
            elif finite_high:
                result.append(-value / math.hypot(value, 1.0))
            else:
                result.append(1.0)
        return tuple(result)


def _bfgs_core(
    function: ScalarFunction,
    gradient: Callable[[Vector], Vector],
    x: Vector,
    *,
    gradient_tolerance: float,
    step_tolerance: float,
    max_iterations: int,
    scales: Vector,
    resolution: Callable[[Vector, float], float] | None = None,
) -> tuple[Vector, float, Vector, int, bool, str]:
    """Run BFGS with a strong-Wolfe line search in unconstrained coordinates.

    Args:
        function: Objective that may raise ``ValueError`` or return ``inf``
            outside its domain (treated as ``+inf`` by the line search).
        gradient: Gradient of the objective.
        x: Finite starting point with finite objective value.
        gradient_tolerance: Absolute tolerance on ``||grad||_2``.
        step_tolerance: Relative step tolerance.
        max_iterations: Maximum iterations.
        scales: Typical magnitude of each coordinate for the step test.
        resolution: Optional estimate of the rounding error of the gradient
            norm at a point with a given value (finite-difference gradients);
            a gradient norm below it is accepted and reported as such.

    Returns:
        Final point, value, gradient, iterations, convergence flag, message.

    Equation:
        [EQ-OPT-BFGS-001], [EQ-OPT-BFGS-002], [EQ-OPT-WOLFE-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Sections 3.1, 3.5, and 6.1],
        [REF-DENNIS-SCHNABEL-1996, Section 7.2].

    """
    size = len(x)
    value = function(x)
    if not math.isfinite(value):
        raise ValueError("objective is not finite at the starting point")
    g = gradient(x)
    inverse_hessian: Matrix = identity(size)
    first_update = True
    for iteration in range(max_iterations):
        g_norm = norm(g)
        verdict = _gradient_verdict(g_norm, gradient_tolerance, resolution, x, value)
        if verdict is not None:
            return x, value, g, iteration, True, verdict
        direction = tuple(-d for d in matvec(inverse_hessian, g))
        slope = dot(g, direction)
        if not slope < 0.0:
            inverse_hessian = identity(size)
            first_update = True
            direction = tuple(-d for d in g)
            slope = -dot(g, g)
        initial = 1.0
        if first_update:
            initial = min(1.0, 1.0 / max(norm(direction), 1e-300))
        search = _strong_wolfe(
            function, gradient, x, value, g, direction, slope, initial
        )
        if search is None:
            predicted = -0.5 * slope
            floor = _ROUNDING_FACTOR * _EPS * abs(value)
            if not first_update and predicted <= floor:
                return (
                    x,
                    value,
                    g,
                    iteration,
                    True,
                    f"converged to the rounding resolution of f: the predicted "
                    f"decrease {predicted:.3g} is below {floor:.3g} (gradient "
                    f"norm {g_norm:.3g})",
                )
            return (
                x,
                value,
                g,
                iteration,
                False,
                "line search could not decrease the objective (stalled with "
                f"gradient norm {g_norm:.3g} above the tolerance and predicted "
                f"decrease {predicted:.3g}; the objective or its gradient may be "
                "too noisy at this scale)",
            )
        alpha, candidate, candidate_value, new_g = search
        s = tuple(a - b for a, b in zip(candidate, x, strict=True))
        y = tuple(a - b for a, b in zip(new_g, g, strict=True))
        sy = dot(s, y)
        if sy > 1e-12 * norm(s) * norm(y) and sy > 0.0:
            if first_update:
                yy = dot(y, y)
                inverse_hessian = scale_matrix(identity(size), sy / yy)
                first_update = False
            rho = 1.0 / sy
            left = add(identity(size), outer(s, y), scale=-rho)
            inverse_hessian = add(
                matmul(matmul(left, inverse_hessian), transpose(left)),
                outer(s, s),
                scale=rho,
            )
        relative_step = max(
            abs(si) / max(abs(xi), scale)
            for si, xi, scale in zip(s, candidate, scales, strict=True)
        )
        x, value, g = candidate, candidate_value, new_g
        verdict = _gradient_verdict(norm(g), gradient_tolerance, resolution, x, value)
        if verdict is not None:
            return x, value, g, iteration + 1, True, verdict
        if relative_step <= step_tolerance:
            return (
                x,
                value,
                g,
                iteration + 1,
                True,
                f"relative step tolerance met (gradient norm {norm(g):.3g})",
            )
    return x, value, g, max_iterations, False, "iteration limit reached"


def _gradient_verdict(
    g_norm: float,
    tolerance: float,
    resolution: Callable[[Vector, float], float] | None,
    x: Vector,
    value: float,
) -> str | None:
    """Return a convergence message when the gradient test passes, else ``None``.

    With an exact gradient the test is ``||g|| <= tolerance``. With a
    finite-difference gradient whose rounding error ``r`` exceeds the
    tolerance, the tolerance cannot be verified; the test then accepts
    ``||g|| <= r`` and says so in the message.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1],
        [REF-DENNIS-SCHNABEL-1996, Section 7.2].

    """
    if resolution is None:
        return "gradient tolerance met" if g_norm <= tolerance else None
    floor = resolution(x, value)
    if floor <= tolerance:
        return "gradient tolerance met" if g_norm <= tolerance else None
    if g_norm <= floor:
        return (
            f"gradient norm {g_norm:.3g} is within the finite-difference "
            f"resolution {floor:.3g}, which exceeds the requested tolerance"
        )
    return None


def _strong_wolfe(
    function: ScalarFunction,
    gradient: Callable[[Vector], Vector],
    x: Vector,
    value: float,
    g: Vector,
    direction: Vector,
    slope: float,
    initial: float,
) -> tuple[float, Vector, float, Vector] | None:
    """Find a step satisfying the strong Wolfe conditions (bracket and zoom).

    Algorithm 3.5 with the zoom of Algorithm 3.6 of Nocedal and Wright: trial
    steps grow by doubling until the sufficient-decrease condition fails or
    the slope turns non-negative, then a safeguarded cubic (or quadratic, or
    bisection) interpolation shrinks the bracket. The objective is evaluated
    first and the gradient only at points that satisfy sufficient decrease.
    If the curvature condition cannot be met but a point with sufficient
    decrease was found, that point is returned (the BFGS update is then
    skipped when ``y's <= 0``).

    Args:
        function: Objective (``+inf`` or ``ValueError`` outside the domain).
        gradient: Gradient function.
        x: Current point.
        value: ``f(x)``.
        g: ``grad f(x)``.
        direction: Descent direction ``p`` with ``g'p < 0``.
        slope: ``g'p``.
        initial: First trial step.

    Returns:
        ``(alpha, x + alpha p, f, grad f)`` or ``None`` when no decrease was
        found.

    Equation:
        [EQ-OPT-WOLFE-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 3.5].

    """
    evaluations_left = _MAX_LINE_SEARCH

    def phi(alpha: float) -> tuple[Vector, float]:
        """Return the trial point and objective, ``inf`` on failure.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 3.5]. No equation is evaluated.

        """
        nonlocal evaluations_left
        evaluations_left -= 1
        point = tuple(xi + alpha * di for xi, di in zip(x, direction, strict=True))
        try:
            result = function(point)
        except (ValueError, ArithmeticError):
            return point, math.inf
        return point, result if math.isfinite(result) else math.inf

    def derivative(point: Vector) -> tuple[Vector, float] | None:
        """Return the gradient and directional derivative, ``None`` on failure.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 3.5]. No equation is evaluated.

        """
        try:
            grad = gradient(point)
        except (ValueError, ArithmeticError):
            return None
        if not all(math.isfinite(v) for v in grad):
            return None
        return grad, dot(grad, direction)

    best: tuple[float, Vector, float, Vector] | None = None

    def zoom(
        low: float,
        low_value: float,
        low_slope: float,
        high: float,
        high_value: float,
        high_slope: float | None,
    ) -> tuple[float, Vector, float, Vector] | None:
        """Shrink a bracket that contains a strong-Wolfe step.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Algorithm 3.6].

        """
        nonlocal best
        while evaluations_left > 0:
            width = high - low
            if abs(width) <= 1e-16 * max(abs(low), abs(high), 1e-300):
                return best
            trial = _interpolate(
                low, low_value, low_slope, high, high_value, high_slope
            )
            point, trial_value = phi(trial)
            if (
                trial_value > value + _WOLFE_C1 * trial * slope
                or trial_value >= low_value
            ):
                high, high_value, high_slope = trial, trial_value, None
                continue
            evaluated = derivative(point)
            if evaluated is None:
                high, high_value, high_slope = trial, math.inf, None
                continue
            grad, trial_slope = evaluated
            if best is None or trial_value < best[2]:
                best = (trial, point, trial_value, grad)
            if abs(trial_slope) <= -_WOLFE_C2 * slope:
                return trial, point, trial_value, grad
            if trial_slope * (high - low) >= 0.0:
                high, high_value, high_slope = low, low_value, low_slope
            low, low_value, low_slope = trial, trial_value, trial_slope
        return best

    previous, previous_value, previous_slope = 0.0, value, slope
    alpha = initial
    for count in range(_MAX_LINE_SEARCH):
        if evaluations_left <= 0:
            break
        point, trial_value = phi(alpha)
        if trial_value > value + _WOLFE_C1 * alpha * slope or (
            count > 0 and trial_value >= previous_value
        ):
            return zoom(
                previous, previous_value, previous_slope, alpha, trial_value, None
            )
        evaluated = derivative(point)
        if evaluated is None:
            return zoom(previous, previous_value, previous_slope, alpha, math.inf, None)
        grad, trial_slope = evaluated
        if best is None or trial_value < best[2]:
            best = (alpha, point, trial_value, grad)
        if abs(trial_slope) <= -_WOLFE_C2 * slope:
            return alpha, point, trial_value, grad
        if trial_slope >= 0.0:
            return zoom(
                alpha,
                trial_value,
                trial_slope,
                previous,
                previous_value,
                previous_slope,
            )
        previous, previous_value, previous_slope = alpha, trial_value, trial_slope
        alpha *= 2.0
    return best


def _interpolate(
    low: float,
    low_value: float,
    low_slope: float,
    high: float,
    high_value: float,
    high_slope: float | None,
) -> float:
    """Return a safeguarded interpolated trial step inside a bracket.

    A cubic through both values and slopes is used when both slopes are
    known, a quadratic through ``(low, f, f')`` and ``(high, f)`` otherwise,
    and bisection when the bracket end is not finite or the interpolant falls
    within 10 % of either end.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 3.5, Equations 3.57-3.59].

    """
    width = high - low
    candidate = math.nan
    if math.isfinite(high_value):
        if high_slope is not None:
            d1 = low_slope + high_slope - 3.0 * (low_value - high_value) / (low - high)
            radicand = d1 * d1 - low_slope * high_slope
            if radicand >= 0.0:
                d2 = math.copysign(math.sqrt(radicand), high - low)
                denominator = high_slope - low_slope + 2.0 * d2
                if denominator != 0.0:
                    candidate = high - width * (high_slope + d2 - d1) / denominator
        else:
            curvature = high_value - low_value - low_slope * width
            if curvature > 0.0:
                candidate = low - low_slope * width * width / (2.0 * curvature)
    lo_edge = min(low, high) + 0.1 * abs(width)
    hi_edge = max(low, high) - 0.1 * abs(width)
    if not (math.isfinite(candidate) and lo_edge <= candidate <= hi_edge):
        candidate = low + 0.5 * width
    return candidate


def minimize_bfgs(
    function: ScalarFunction,
    x0: Sequence[float],
    *,
    gradient: Callable[[Vector], Sequence[float]] | None = None,
    gradient_tolerance: float = 1e-8,
    max_iterations: int = 500,
    step_tolerance: float | None = None,
    bounds: Bounds | None = None,
    typical_scale: float | Sequence[float] | None = None,
) -> OptimizationResult:
    """Minimize a smooth function by BFGS with a strong-Wolfe line search.

    Stopping rules. The tolerances are independent of an additive constant
    in ``f``; only the two floating-point resolution limits below involve
    ``|f|``, through the unit roundoff ``eps``:

    * ``||grad f||_2 <= gradient_tolerance`` (absolute) gives
      ``converged=True``. With a finite-difference gradient whose rounding
      error ``r = ||(eps |f| / h_i)_i||`` exceeds the tolerance, the test is
      ``||grad f|| <= r`` and the message says the tolerance could not be
      verified;
    * a relative step ``max_i |s_i| / max(|x_i|, typ_i) <= step_tolerance``
      (default ``eps^(2/3)``, Dennis and Schnabel) after an accepted step
      also gives ``converged=True``, with the gradient norm in the message;
    * a line search that finds no decrease gives ``converged=True`` only
      when the predicted quasi-Newton decrease ``-g'p/2`` is below
      ``16 eps |f|`` (the decrease cannot be represented); otherwise the run
      has stalled and ``converged=False``, as for the iteration limit.

    The first iteration scales the identity inverse Hessian by
    ``s'y / y'y`` before the first update (Nocedal and Wright, eq. 6.20).
    Without an analytic gradient, central differences with the typical scale
    ``typ_i`` are used (:func:`central_gradient`).

    With ``bounds`` the problem is solved in the unconstrained coordinates of
    :class:`BoxTransform`; the reported ``gradient_norm`` is then the norm of
    the gradient with respect to those coordinates (it vanishes at a minimum
    on a bound).

    Args:
        function: Smooth scalar objective; ``ValueError`` or a non-finite
            value at a trial point makes the line search shorten the step.
        x0: Finite starting point.
        gradient: Optional analytic gradient (in ``x``); central differences
            otherwise.
        gradient_tolerance: Absolute tolerance on the gradient norm.
        max_iterations: Maximum number of quasi-Newton iterations.
        step_tolerance: Relative step tolerance (default ``eps^(2/3)``).
        bounds: Optional ``(low, high)`` per coordinate; ``None`` or an
            infinite value marks a missing bound.
        typical_scale: Typical magnitude of each coordinate, used by the step
            test and the finite-difference steps. The default is ``|x0_i|``
            (or ``1`` where ``x0_i = 0``).

    Returns:
        Minimization result with evaluation accounting.

    Raises:
        ValueError: If ``x0`` is empty, outside the bounds, or the objective
            is not finite at the start.

    Equation:
        [EQ-OPT-BFGS-001], [EQ-OPT-BFGS-002], [EQ-OPT-WOLFE-001],
        [EQ-OPT-BOX-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Sections 3.5, 6.1, and 8.1],
        [REF-DENNIS-SCHNABEL-1996, Section 7.2], [REF-JAMES-ROOS-1975].

    """
    start = as_vector(x0, name="x0")
    if not start:
        raise ValueError("x0 must be non-empty")
    tolerance = require_finite_real(gradient_tolerance, name="gradient_tolerance")
    if tolerance < 0.0:
        raise ValueError("gradient_tolerance must be non-negative")
    x_tolerance = (
        _DEFAULT_STEP_TOLERANCE
        if step_tolerance is None
        else require_finite_real(step_tolerance, name="step_tolerance")
    )
    if x_tolerance < 0.0:
        raise ValueError("step_tolerance must be non-negative")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    size = len(start)
    if typical_scale is None:
        scales = tuple(abs(v) if v != 0.0 else 1.0 for v in start)
    else:
        scales = _per_coordinate(typical_scale, size, "typical_scale", 1.0)
    counter = [0]
    transform = None if bounds is None else BoxTransform.from_bounds(bounds, size)

    def outer_point(z: Vector) -> Vector:
        """Map optimizer coordinates to ``x``.

        References:
            [REF-JAMES-ROOS-1975]. No equation is evaluated.

        """
        return z if transform is None else transform.to_outer(z)

    def objective(z: Vector) -> float:
        """Evaluate and count the objective in optimizer coordinates.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Chapter 2]. No equation is evaluated.

        """
        counter[0] += 1
        result = function(outer_point(z))
        if isinstance(result, bool):
            raise TypeError("objective must return a real number")
        return float(result)

    inner_scales = scales if transform is None else (1.0,) * size

    def grad(z: Vector) -> Vector:
        """Evaluate and count the gradient in optimizer coordinates.

        References:
            [EQ-OPT-FD-001], [EQ-OPT-BOX-001],
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        if gradient is not None:
            counter[0] += 1
            values = as_vector(gradient(outer_point(z)), name="gradient")
            if len(values) != size:
                raise ValueError("gradient length must equal the dimension")
            if transform is None:
                return values
            return tuple(
                v * d for v, d in zip(values, transform.derivative(z), strict=True)
            )
        values, used = central_gradient(
            lambda point: float(function(outer_point(point))),
            z,
            typical_scale=inner_scales,
        )
        counter[0] += used
        return values

    def fd_resolution(z: Vector, value: float) -> float:
        """Estimate the rounding error of the central-difference gradient norm.

        Each component carries an error of about ``eps |f| / h_i`` from
        rounding ``f`` at the two shifted points ([EQ-OPT-FD-003]).

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        steps = _difference_steps(
            z,
            factor=_CENTRAL_FACTOR,
            relative_step=None,
            step=None,
            typical_scale=inner_scales,
        )
        magnitude = _EPS * max(abs(value), 1e-300)
        return math.sqrt(math.fsum((magnitude / h) ** 2 for h in steps))

    begin = start if transform is None else transform.to_inner(start)
    if transform is None:
        step_scales = scales
    else:
        step_scales = tuple(max(abs(v), 1.0) for v in begin)
    z, value, g, iterations, converged, message = _bfgs_core(
        objective,
        grad,
        begin,
        gradient_tolerance=tolerance,
        step_tolerance=x_tolerance,
        max_iterations=max_iterations,
        scales=step_scales,
        resolution=None if gradient is not None else fd_resolution,
    )
    return OptimizationResult(
        outer_point(z), value, norm(g), iterations, counter[0], converged, message
    )


def minimize_nelder_mead(
    function: ScalarFunction,
    x0: Sequence[float],
    *,
    initial_step: float = 0.1,
    tolerance: float = 1e-10,
    max_evaluations: int = 20_000,
    bounds: Bounds | None = None,
) -> OptimizationResult:
    """Minimize a function without derivatives by the Nelder-Mead simplex.

    With ``bounds`` the simplex moves in the unconstrained coordinates of
    :class:`BoxTransform`, so every evaluated point lies inside the box; the
    returned ``x`` is mapped back to the original coordinates.

    Args:
        function: Scalar objective; non-finite values are treated as ``+inf``.
        x0: Finite starting point.
        initial_step: Relative edge length of the starting simplex.
        tolerance: Stop when the simplex value spread falls below this value.
        max_evaluations: Maximum objective evaluations.
        bounds: Optional ``(low, high)`` per coordinate (``None`` or infinite
            for a missing bound).

    Returns:
        Minimization result; ``gradient_norm`` is ``nan`` (not computed).

    Raises:
        ValueError: If ``x0`` is empty or outside the bounds.

    Equation:
        [EQ-OPT-NM-001], [EQ-OPT-BOX-001].

    References:
        [REF-NELDER-MEAD-1965], [REF-NOCEDAL-WRIGHT-2006, Section 9.5],
        [REF-JAMES-ROOS-1975].

    """
    start = as_vector(x0, name="x0")
    if not start:
        raise ValueError("x0 must be non-empty")
    if bounds is not None:
        transform = BoxTransform.from_bounds(bounds, len(start))
        inner = minimize_nelder_mead(
            lambda z: function(transform.to_outer(z)),
            transform.to_inner(start),
            initial_step=initial_step,
            tolerance=tolerance,
            max_evaluations=max_evaluations,
        )
        return OptimizationResult(
            transform.to_outer(inner.x),
            inner.value,
            inner.gradient_norm,
            inner.iterations,
            inner.evaluations,
            inner.converged,
            inner.message,
        )
    evaluations = 0

    def f(point: Vector) -> float:
        """Evaluate the objective, mapping failures to ``+inf``.

        References:
            [REF-NELDER-MEAD-1965]. No equation is evaluated.

        """
        nonlocal evaluations
        evaluations += 1
        try:
            value = float(function(point))
        except (ValueError, OverflowError, ZeroDivisionError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    size = len(start)
    simplex = [start]
    for index in range(size):
        vertex = list(start)
        vertex[index] += initial_step * max(1.0, abs(start[index]))
        simplex.append(tuple(vertex))
    values = [f(vertex) for vertex in simplex]
    iterations = 0
    while evaluations < max_evaluations:
        order = sorted(range(size + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) <= tolerance * max(1.0, abs(values[0])):
            spread = max(
                norm(tuple(a - b for a, b in zip(v, simplex[0], strict=True)))
                for v in simplex
            )
            if spread <= math.sqrt(tolerance) * max(1.0, norm(simplex[0])):
                return OptimizationResult(
                    simplex[0],
                    values[0],
                    math.nan,
                    iterations,
                    evaluations,
                    True,
                    "simplex tolerance met",
                )
        iterations += 1
        centroid = tuple(
            math.fsum(vertex[j] for vertex in simplex[:-1]) / size for j in range(size)
        )

        worst = simplex[-1]

        def along(
            coefficient: float,
            centroid: Vector = centroid,
            worst: Vector = worst,
        ) -> Vector:
            """Return ``centroid + coefficient (centroid - worst)``.

            References:
                [EQ-OPT-NM-001], [REF-NELDER-MEAD-1965].

            """
            return tuple(
                c + coefficient * (c - w) for c, w in zip(centroid, worst, strict=True)
            )

        reflected = along(1.0)
        reflected_value = f(reflected)
        if values[0] <= reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
            continue
        if reflected_value < values[0]:
            expanded = along(2.0)
            expanded_value = f(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
            continue
        contracted = along(0.5 if reflected_value < values[-1] else -0.5)
        contracted_value = f(contracted)
        if contracted_value < min(reflected_value, values[-1]):
            simplex[-1], values[-1] = contracted, contracted_value
            continue
        best = simplex[0]
        simplex = [best] + [
            tuple(b + 0.5 * (v - b) for b, v in zip(best, vertex, strict=True))
            for vertex in simplex[1:]
        ]
        values = [values[0]] + [f(vertex) for vertex in simplex[1:]]
    best_index = min(range(size + 1), key=lambda i: values[i])
    return OptimizationResult(
        simplex[best_index],
        values[best_index],
        math.nan,
        iterations,
        evaluations,
        False,
        "evaluation limit reached",
    )


@dataclass(frozen=True, slots=True)
class MultiStartResult:
    """Outcome of a seeded multi-start minimization.

    Args:
        best: Best run: the lowest value among converged runs, or among all
            runs when none converged.
        runs: Every run in start order.
        starts: Start points in the original coordinates.
        seed: Seed of the stream that drew the random starts.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 2]. No equation is evaluated.

    """

    best: OptimizationResult
    runs: tuple[OptimizationResult, ...]
    starts: tuple[Vector, ...]
    seed: int

    @property
    def converged_runs(self) -> int:
        """Return the number of runs that met a stopping tolerance.

        References:
            Project validation policy. No equation is evaluated.

        """
        return sum(run.converged for run in self.runs)

    @property
    def evaluations(self) -> int:
        """Return the total number of objective evaluations of all runs.

        References:
            Project validation policy. No equation is evaluated.

        """
        return sum(run.evaluations for run in self.runs)


def multi_start(
    function: ScalarFunction,
    bounds: Bounds,
    *,
    starts: int = 10,
    seed: int | RandomStream | None = None,
    method: str = "bfgs",
    x0: Sequence[float] | None = None,
    gradient: Callable[[Vector], Sequence[float]] | None = None,
    bounded: bool = True,
    **options: Any,
) -> MultiStartResult:
    """Run a local minimizer from several seeded random starts in a box.

    Starts are drawn independently and uniformly in the finite box from the
    stream ``resolve_stream(seed).child("multi-start")``; ``x0``, if given,
    is the first start. Each run is a :func:`minimize_bfgs` or
    :func:`minimize_nelder_mead` call (with the same box as ``bounds`` when
    ``bounded`` is true). A run that raises ``ValueError`` (for example a
    non-finite objective at its start) is recorded as a failed run.

    Args:
        function: Scalar objective.
        bounds: Finite ``(low, high)`` per coordinate.
        starts: Number of starts (at least one).
        seed: Integer seed, stream, or ``None`` for fresh entropy.
        method: ``"bfgs"`` or ``"nelder-mead"``.
        x0: Optional deterministic first start.
        gradient: Optional analytic gradient (BFGS only).
        bounded: Constrain every local run to the box.
        **options: Further keyword arguments for the local minimizer.

    Returns:
        Multi-start result with every run and the best one.

    Raises:
        ValueError: If a bound is infinite, ``starts < 1``, the method is
            unknown, or every run failed.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 2], [REF-STEELE-LEA-FLOOD-2014].

    """
    if isinstance(starts, bool) or not isinstance(starts, int) or starts < 1:
        raise ValueError("starts must be a positive integer")
    if method not in ("bfgs", "nelder-mead"):
        raise ValueError("method must be 'bfgs' or 'nelder-mead'")
    pairs = tuple(bounds)
    size = len(pairs)
    transform = BoxTransform.from_bounds(pairs, size)
    if not all(
        math.isfinite(low) and math.isfinite(high)
        for low, high in zip(transform.lower, transform.upper, strict=True)
    ):
        raise ValueError("multi_start needs finite bounds to draw starts")
    stream = resolve_stream(seed)
    generator = stream.child("multi-start").generator()
    points: list[Vector] = []
    if x0 is not None:
        first = as_vector(x0, name="x0")
        if len(first) != size:
            raise ValueError("x0 length must match the bounds")
        points.append(first)
    while len(points) < starts:
        points.append(
            tuple(
                low + (high - low) * generator.random()
                for low, high in zip(transform.lower, transform.upper, strict=True)
            )
        )
    runs: list[OptimizationResult] = []
    for point in points:
        try:
            if method == "bfgs":
                run = minimize_bfgs(
                    function,
                    point,
                    gradient=gradient,
                    bounds=pairs if bounded else None,
                    **options,
                )
            else:
                run = minimize_nelder_mead(
                    function, point, bounds=pairs if bounded else None, **options
                )
        except ValueError as error:
            run = OptimizationResult(
                point, math.inf, math.nan, 0, 0, False, f"run failed: {error}"
            )
        runs.append(run)
    finite = [run for run in runs if math.isfinite(run.value)]
    if not finite:
        raise ValueError("every multi-start run failed")
    converged = [run for run in finite if run.converged]
    pool = converged if converged else finite
    best = min(pool, key=lambda run: run.value)
    return MultiStartResult(best, tuple(runs), tuple(points), stream.seed)


def least_squares(
    residual: VectorFunction,
    x0: Sequence[float],
    *,
    jacobian: Callable[[Vector], Sequence[Sequence[float]]] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> LeastSquaresResult:
    """Minimize ``0.5 ||r(x)||^2`` by Levenberg-Marquardt with Marquardt scaling.

    Args:
        residual: Vector residual function with fixed output length.
        x0: Finite starting point.
        jacobian: Optional analytic Jacobian; central differences otherwise.
        tolerance: Relative cost-reduction and step tolerance.
        max_iterations: Maximum accepted-or-rejected iterations.

    Returns:
        Least-squares result with the final Jacobian for covariance estimates.

    Equation:
        [EQ-OPT-LM-001].

    References:
        [REF-MORE-1978], [REF-NOCEDAL-WRIGHT-2006, Section 10.3].

    """
    x = as_vector(x0, name="x0")
    if not x:
        raise ValueError("x0 must be non-empty")
    evaluations = 0

    def r(point: Vector) -> Vector:
        """Evaluate and count the residual vector.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Chapter 10]. No equation is evaluated.

        """
        nonlocal evaluations
        evaluations += 1
        return as_vector(residual(point), name="residual")

    def jac(point: Vector) -> Matrix:
        """Evaluate and count the Jacobian.

        References:
            [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        nonlocal evaluations
        if jacobian is not None:
            evaluations += 1
            return tuple(as_vector(row, name="jacobian row") for row in jacobian(point))
        matrix, used = central_jacobian(residual, point)
        evaluations += used
        return matrix

    residuals = r(x)
    if len(residuals) < len(x):
        raise ValueError("least squares needs at least as many residuals as parameters")
    cost = 0.5 * dot(residuals, residuals)
    j = jac(x)
    damping = 1e-3
    for iteration in range(max_iterations):
        jt = transpose(j)
        gram = matmul(jt, j)
        gradient = matvec(jt, residuals)
        if norm(gradient) <= tolerance * max(1.0, cost):
            return LeastSquaresResult(
                x,
                residuals,
                cost,
                j,
                iteration,
                evaluations,
                True,
                "gradient tolerance met",
            )
        diagonal = tuple(max(gram[i][i], 1e-12) for i in range(len(x)))
        while True:
            damped = tuple(
                tuple(
                    gram[a][b] + (damping * diagonal[a] if a == b else 0.0)
                    for b in range(len(x))
                )
                for a in range(len(x))
            )
            try:
                delta = solve_spd(damped, tuple(-g for g in gradient))
            except ValueError:
                damping *= 10.0
                continue
            candidate = tuple(a + b for a, b in zip(x, delta, strict=True))
            try:
                candidate_residuals = r(candidate)
                candidate_cost = 0.5 * dot(candidate_residuals, candidate_residuals)
            except ValueError:
                candidate_cost = math.inf
            if candidate_cost < cost:
                break
            damping *= 10.0
            if damping > 1e16:
                return LeastSquaresResult(
                    x,
                    residuals,
                    cost,
                    j,
                    iteration,
                    evaluations,
                    False,
                    "damping exceeded without reducing the cost",
                )
        reduction = cost - candidate_cost
        step_small = norm(delta) <= tolerance * (norm(x) + tolerance)
        x, residuals, cost = candidate, candidate_residuals, candidate_cost
        j = jac(x)
        damping = max(damping / 10.0, 1e-12)
        if reduction <= tolerance * max(cost, 1e-300) or step_small:
            return LeastSquaresResult(
                x,
                residuals,
                cost,
                j,
                iteration + 1,
                evaluations,
                True,
                "relative reduction or step tolerance met",
            )
    return LeastSquaresResult(
        x,
        residuals,
        cost,
        j,
        max_iterations,
        evaluations,
        False,
        "iteration limit reached",
    )


def gauss_newton_covariance(
    result: LeastSquaresResult, *, noise_variance: float | None = None
) -> Matrix:
    """Return the Gauss-Newton parameter covariance ``s^2 (J'J)^-1``.

    Args:
        result: Converged least-squares result.
        noise_variance: Known residual variance; if omitted it is estimated as
            ``||r||^2 / (m - n)``.

    Returns:
        Asymptotic parameter covariance under local linearity.

    Raises:
        ValueError: If ``m <= n`` without a known variance or ``J'J`` is singular.

    Equation:
        [EQ-OPT-LM-002].

    References:
        [REF-SEBER-WILD-2003, Section 2.1], [REF-NOCEDAL-WRIGHT-2006, Chapter 10].

    """
    m, n = len(result.residuals), len(result.x)
    if noise_variance is None:
        if m <= n:
            raise ValueError(
                "estimating noise variance requires more residuals than parameters"
            )
        variance = 2.0 * result.cost / (m - n)
    else:
        variance = require_finite_real(noise_variance, name="noise_variance")
        if variance <= 0.0:
            raise ValueError("noise_variance must be positive")
    gram = matmul(transpose(result.jacobian), result.jacobian)
    return scale_matrix(inverse_spd(gram), variance)


def brent_root(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> RootResult:
    """Find a root in a sign-changing bracket by Brent's method.

    Args:
        function: Continuous scalar function.
        lower: Bracket endpoint.
        upper: Bracket endpoint with ``f(lower) f(upper) <= 0``.
        tolerance: Absolute bracket-width tolerance.
        max_iterations: Maximum iterations.

    Returns:
        Root result.

    Raises:
        ValueError: If the bracket does not change sign.

    Equation:
        [EQ-OPT-BRENT-001].

    References:
        [REF-BRENT-1973, Chapter 4].

    """
    a = require_finite_real(lower, name="lower")
    b = require_finite_real(upper, name="upper")
    fa = _finite_value(function(a), "f(lower)")
    fb = _finite_value(function(b), "f(upper)")
    evaluations = 2
    if fa * fb > 0.0:
        raise ValueError("root bracket must change sign")
    c, fc = b, fb
    d = e = b - a
    for iteration in range(max_iterations):
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tolerance_1 = 2.0 * 2.0**-52 * abs(b) + 0.5 * tolerance
        half_width = 0.5 * (c - b)
        if abs(half_width) <= tolerance_1 or fb == 0.0:
            return RootResult(b, fb, iteration, evaluations, True)
        if abs(e) >= tolerance_1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * half_width * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * half_width * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * half_width * q - abs(tolerance_1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = half_width
        else:
            d = e = half_width
        a, fa = b, fb
        b += d if abs(d) > tolerance_1 else math.copysign(tolerance_1, half_width)
        fb = _finite_value(function(b), "f(b)")
        evaluations += 1
    return RootResult(b, fb, max_iterations, evaluations, False)
