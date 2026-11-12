"""BFGS quasi-Newton minimization with a strong-Wolfe line search.

The line search brackets and zooms with safeguarded cubic interpolation
(Nocedal and Wright, Algorithms 3.5 and 3.6); the BFGS update is skipped
when ``y's <= 0``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    add,
    as_vector,
    dot,
    identity,
    matmul,
    matvec,
    norm,
    outer,
    scale_matrix,
    transpose,
)

from ._common import _EPS, ScalarFunction
from .bounds import Bounds, BoxTransform
from .finite_differences import (
    _CENTRAL_FACTOR,
    _difference_steps,
    _per_coordinate,
    central_gradient,
)
from .results import OptimizationResult

# Dennis & Schnabel's suggested relative step tolerance eps^(2/3).
_DEFAULT_STEP_TOLERANCE = _EPS ** (2.0 / 3.0)
_WOLFE_C1 = 1e-4
_WOLFE_C2 = 0.9
_MAX_LINE_SEARCH = 60
# A predicted quasi-Newton decrease below this many ulps of |f| cannot be
# resolved in floating point.
_ROUNDING_FACTOR = 16.0


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
