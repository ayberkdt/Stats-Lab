"""Fixed-step explicit integration of an ordinary differential equation to an event.

A simulator that stops "when the package reaches the ground" or "when the
tank is empty" integrates ``y' = f(t, y)`` until an event function
``g(t, y)`` changes sign. Its quantity of interest is then the state at the
event, and the way the crossing is located inside the last step decides the
order of accuracy that a solution-verification study can observe:

- ``"step"``: report the first grid point with ``g <= 0``. The event time is
  off by up to one step, so the error is ``O(h)`` with a coefficient that
  jumps as the grid slides past the event; every method is capped at order
  one and the convergence history is not smooth.
- ``"linear"``: interpolate ``g`` and the state linearly across the last
  step. The interpolation error is ``O(h^2)``, enough for explicit Euler
  (order 1) and Heun's method (order 2), but it caps the fourth-order
  Runge-Kutta method at order two.
- ``"hermite"``: cubic Hermite interpolation of the state from the values
  and slopes at both ends of the step, with the root of ``g`` on the cubic
  found by the Illinois method. The interpolation error is ``O(h^4)``, so
  the event inherits the order of every method offered here.

Guide: docs/foundations/vv-uq-foundations.md

References:
    [REF-HAIRER-NORSETT-WANNER-1993, Sections II.1 and II.6] (Runge-Kutta
    methods, dense output, and event location),
    [REF-DOWELL-JARRATT-1971] (Illinois root finding).

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

ODE_METHODS = ("euler", "heun", "rk4")
"""Explicit one-step methods of :func:`integrate_to_event` (orders 1, 2, 4)."""

EVENT_LOCATIONS = ("step", "linear", "hermite")
"""Ways of locating the event inside the last step (see the module docstring)."""

METHOD_ORDERS = {"euler": 1, "heun": 2, "rk4": 4}
"""Classical order of accuracy of each method in :data:`ODE_METHODS`."""

Rhs = Callable[[float, Sequence[float]], Sequence[float]]
Event = Callable[[float, Sequence[float]], float]


@dataclass(frozen=True, slots=True)
class EventSolution:
    """State of an ordinary differential equation at a located event.

    Args:
        time: Event time ``t*``.
        state: State ``y(t*)`` from the chosen event location.
        steps: Number of full steps taken (the last one crosses the event).
        evaluations: Number of right-hand-side evaluations, including the
            extra slope a Hermite location needs.
        method: Integration method, one of :data:`ODE_METHODS`.
        location: Event location, one of :data:`EVENT_LOCATIONS`.
        step: Step size ``h``.
        path: ``(t, y)`` at every grid point up to and including the step
            that crosses the event, followed by the located event; empty
            unless the integration was asked to record it.

    References:
        [EQ-NUM-ODE-001], [EQ-NUM-ODE-002].

    """

    time: float
    state: tuple[float, ...]
    steps: int
    evaluations: int
    method: str
    location: str
    step: float
    path: tuple[tuple[float, tuple[float, ...]], ...] = ()


def _check_state(values: Sequence[float], size: int, what: str) -> list[float]:
    """Return a state as a list of finite floats of the required length.

    Args:
        values: Candidate vector.
        size: Required length.
        what: Name used in the error message.

    Returns:
        The vector as floats.

    Raises:
        ValueError: If the length is wrong or a component is not finite.

    References:
        [REF-HAIRER-NORSETT-WANNER-1993, Section II.1]. No equation is
        evaluated.

    """
    vector = [float(value) for value in values]
    if len(vector) != size:
        raise ValueError(f"{what} has length {len(vector)}, expected {size}")
    if not all(map(math.isfinite, vector)):
        raise ValueError(f"{what} is not finite: {vector}")
    return vector


def _slope(rhs: Rhs, t: float, y: Sequence[float], size: int) -> Sequence[float]:
    """Evaluate ``f(t, y)`` and check its length.

    A non-finite slope is caught one step later, when it makes the state
    non-finite, so only the length is checked here.

    Args:
        rhs: Right-hand side.
        t: Time.
        y: State.
        size: Length of the state.

    Returns:
        The slope as returned by ``rhs``.

    Raises:
        ValueError: If the slope does not have the length of the state.

    References:
        [REF-HAIRER-NORSETT-WANNER-1993, Section II.1]. No equation is
        evaluated.

    """
    slope = rhs(t, y)
    if len(slope) != size:
        raise ValueError(f"f(t, y) has length {len(slope)}, expected {size}")
    return slope


def _advance(
    rhs: Rhs,
    method: str,
    t: float,
    y: list[float],
    slope: Sequence[float],
    h: float,
) -> tuple[list[float], int]:
    """Take one explicit step from ``(t, y)`` whose slope ``f(t, y)`` is known.

    Args:
        rhs: Right-hand side ``f(t, y)``.
        method: One of :data:`ODE_METHODS`.
        t: Current time.
        y: Current state.
        slope: ``f(t, y)``.
        h: Step size.

    Returns:
        The new state and the number of additional ``f`` evaluations.

    References:
        [EQ-NUM-ODE-001], [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

    """
    size = len(y)
    if method == "euler":
        return [a + h * k for a, k in zip(y, slope)], 0
    if method == "heun":
        k2 = _slope(rhs, t + h, [a + h * k for a, k in zip(y, slope)], size)
        half = 0.5 * h
        return [a + half * (k1 + k) for a, k1, k in zip(y, slope, k2)], 1
    half = 0.5 * h
    k2 = _slope(rhs, t + half, [a + half * k for a, k in zip(y, slope)], size)
    k3 = _slope(rhs, t + half, [a + half * k for a, k in zip(y, k2)], size)
    k4 = _slope(rhs, t + h, [a + h * k for a, k in zip(y, k3)], size)
    sixth = h / 6.0
    return [
        a + sixth * (p + 2.0 * q + 2.0 * r + s)
        for a, p, q, r, s in zip(y, slope, k2, k3, k4)
    ], 3


def _hermite(
    theta: float,
    h: float,
    start: Sequence[float],
    start_slope: Sequence[float],
    end: Sequence[float],
    end_slope: Sequence[float],
) -> list[float]:
    """Evaluate the cubic Hermite interpolant of one step at ``t_n + theta h``.

    Args:
        theta: Fraction of the step, in ``[0, 1]``.
        h: Step size.
        start: ``y_n``.
        start_slope: ``f(t_n, y_n)``.
        end: ``y_{n+1}``.
        end_slope: ``f(t_{n+1}, y_{n+1})``.

    Returns:
        The interpolated state.

    References:
        [EQ-NUM-ODE-002], [REF-HAIRER-NORSETT-WANNER-1993, Section II.6].

    """
    t2 = theta * theta
    t3 = t2 * theta
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + theta
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return [
        h00 * a + h10 * h * da + h01 * b + h11 * h * db
        for a, da, b, db in zip(start, start_slope, end, end_slope, strict=True)
    ]


def _illinois(
    function: Callable[[float], float], high_value: float, low_value: float
) -> float:
    """Return the root in ``[0, 1]`` of a function positive at 0 and not at 1.

    Args:
        function: Continuous function of the step fraction.
        high_value: ``function(0) > 0``.
        low_value: ``function(1) <= 0``.

    Returns:
        A fraction ``theta`` with ``function(theta)`` zero to rounding, or the
        midpoint of a bracket narrower than ``1e-15``.

    References:
        [REF-DOWELL-JARRATT-1971]. No registered equation is evaluated.

    """
    if low_value == 0.0:
        return 1.0
    a, fa, b, fb = 0.0, high_value, 1.0, low_value
    side = 0
    for _ in range(200):
        c = (a * fb - b * fa) / (fb - fa)
        fc = function(c)
        if fc == 0.0 or b - a < 1e-15:
            return c
        if fc > 0.0:
            a, fa = c, fc
            if side == -1:
                fb *= 0.5
            side = -1
        else:
            b, fb = c, fc
            if side == 1:
                fa *= 0.5
            side = 1
    return 0.5 * (a + b)


def integrate_to_event(
    rhs: Rhs,
    initial_state: Sequence[float],
    event: Event,
    *,
    step: float,
    method: str = "rk4",
    location: str = "hermite",
    start: float = 0.0,
    max_steps: int = 10_000_000,
    record: bool = False,
) -> EventSolution:
    """Integrate ``y' = f(t, y)`` with a fixed step until ``g(t, y)`` reaches zero.

    The explicit method advances ``y_{n+1} = y_n + h Phi(t_n, y_n; h)``
    ([EQ-NUM-ODE-001]) while ``g(t_n, y_n) > 0``. The first step with
    ``g(t_{n+1}, y_{n+1}) <= 0`` contains the event, which is then located
    by ``location`` ([EQ-NUM-ODE-002]): the end of that step, the root of the
    linear interpolant of ``g``, or the root of ``g`` on the cubic Hermite
    interpolant of the state.

    Args:
        rhs: Right-hand side ``f(t, y)`` returning a vector of the length of
            the state.
        initial_state: ``y(start)``.
        event: Event function ``g(t, y)``; must be positive at the start.
        step: Fixed step ``h > 0``.
        method: ``"euler"`` (order 1), ``"heun"`` (order 2), or ``"rk4"``
            (classical fourth-order Runge-Kutta).
        location: ``"step"``, ``"linear"``, or ``"hermite"``.
        start: Initial time.
        max_steps: Largest number of steps before giving up.
        record: Keep the grid points and the event in ``path``.

    Returns:
        The event time and state, the work done, and optionally the path.

    Raises:
        ValueError: If the step, method, location, or ``max_steps`` is
            invalid, the state or a slope is not finite, or the event
            function is not positive at the start.
        RuntimeError: If no event occurs within ``max_steps`` steps.

    Equation:
        [EQ-NUM-ODE-001], [EQ-NUM-ODE-002].

    References:
        [EQ-NUM-ODE-001], [EQ-NUM-ODE-002],
        [REF-HAIRER-NORSETT-WANNER-1993, Sections II.1 and II.6],
        [REF-DOWELL-JARRATT-1971].

    """
    h = float(step)
    if not (math.isfinite(h) and h > 0.0):
        raise ValueError(f"step must be finite and positive, got {step!r}")
    if method not in ODE_METHODS:
        raise ValueError(f"method must be one of {ODE_METHODS}, got {method!r}")
    if location not in EVENT_LOCATIONS:
        raise ValueError(
            f"location must be one of {EVENT_LOCATIONS}, got {location!r}"
        )
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")
    t = float(start)
    if not math.isfinite(t):
        raise ValueError(f"start must be finite, got {start!r}")
    y = _check_state(initial_state, len(initial_state), "initial_state")
    size = len(y)
    if size == 0:
        raise ValueError("initial_state must not be empty")
    g = float(event(t, y))
    if not (math.isfinite(g) and g > 0.0):
        raise ValueError(f"event function must be positive at the start, got {g!r}")
    slope = _slope(rhs, t, y, size)
    evaluations = 1
    path: list[tuple[float, tuple[float, ...]]] = [(t, tuple(y))] if record else []
    for steps in range(1, max_steps + 1):
        new, extra = _advance(rhs, method, t, y, slope, h)
        evaluations += extra
        if not all(map(math.isfinite, new)):
            raise ValueError(f"state is not finite after step {steps}: {new}")
        t_new = start + steps * h
        g_new = float(event(t_new, new))
        if not math.isfinite(g_new):
            raise ValueError(f"event function is not finite at t={t_new!r}")
        if record:
            path.append((t_new, tuple(new)))
        if g_new <= 0.0:
            if location == "step":
                time, state = t_new, new
            elif location == "linear":
                theta = g / (g - g_new)
                time = t + theta * h
                state = [a + theta * (b - a) for a, b in zip(y, new, strict=True)]
            else:
                new_slope = _slope(rhs, t_new, new, size)
                evaluations += 1

                def on_cubic(theta: float) -> float:
                    """Return ``g`` on the Hermite interpolant at fraction ``theta``.

                    References:
                        [EQ-NUM-ODE-002].

                    """
                    return float(
                        event(t + theta * h, _hermite(theta, h, y, slope, new, new_slope))
                    )

                theta = _illinois(on_cubic, g, g_new)
                time = t + theta * h
                state = _hermite(theta, h, y, slope, new, new_slope)
            if record:
                path.append((time, tuple(state)))
            return EventSolution(
                time,
                tuple(state),
                steps,
                evaluations,
                method,
                location,
                h,
                tuple(path),
            )
        t, y, g = t_new, new, g_new
        slope = _slope(rhs, t, y, size)
        evaluations += 1
    raise RuntimeError(f"no event within {max_steps} steps of size {h!r}")
