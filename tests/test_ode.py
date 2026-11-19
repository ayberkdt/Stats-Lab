"""Tests for fixed-step ODE integration to an event (``probstat.numerics.ode``).

The convergence checks compare the located event time with closed forms:
exponential decay ``y' = -y`` reaching ``1/2`` at ``ln 2``, the logistic
equation ``y' = y (1 - y)`` from ``0.1`` reaching ``1/2`` at ``ln 9``, and the
harmonic oscillator ``x'' = -x`` from ``(1, 0)`` reaching ``x = 0`` at
``pi/2``. Observed orders are read from error ratios under halving, never
from pinned floating-point values, so they hold on every platform's libm.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable, Sequence

from probstat.numerics import (
    EVENT_LOCATIONS,
    METHOD_ORDERS,
    ODE_METHODS,
    EventSolution,
    integrate_to_event,
)

Rhs = Callable[[float, Sequence[float]], Sequence[float]]
Event = Callable[[float, Sequence[float]], float]


def decay(t: float, y: Sequence[float]) -> list[float]:
    """Return ``-y`` (exponential decay).

    References:
        [EQ-NUM-ODE-001].

    """
    return [-y[0]]


def logistic(t: float, y: Sequence[float]) -> list[float]:
    """Return ``y (1 - y)`` (logistic growth).

    References:
        [EQ-NUM-ODE-001].

    """
    return [y[0] * (1.0 - y[0])]


def oscillator(t: float, y: Sequence[float]) -> list[float]:
    """Return ``(v, -x)`` (harmonic oscillator).

    References:
        [EQ-NUM-ODE-001].

    """
    return [y[1], -y[0]]


# name -> (rhs, initial state, event, exact event time)
PROBLEMS: dict[str, tuple[Rhs, list[float], Event, float]] = {
    "decay": (decay, [1.0], lambda t, y: y[0] - 0.5, math.log(2.0)),
    "logistic": (logistic, [0.1], lambda t, y: 0.5 - y[0], math.log(9.0)),
    "oscillator": (oscillator, [1.0, 0.0], lambda t, y: y[0], 0.5 * math.pi),
}
STEPS = (0.1, 0.05, 0.025, 0.0125)


def time_errors(problem: str, method: str, location: str) -> list[float]:
    """Return the event-time error on :data:`STEPS`.

    References:
        [EQ-NUM-ODE-001], [EQ-NUM-ODE-002].

    """
    rhs, start, event, exact = PROBLEMS[problem]
    return [
        abs(
            integrate_to_event(
                rhs, start, event, step=h, method=method, location=location
            ).time
            - exact
        )
        for h in STEPS
    ]


def observed_orders(errors: Sequence[float]) -> list[float]:
    """Return ``log2(e(h) / e(h/2))`` for successive halvings.

    References:
        [EQ-VV-001].

    """
    return [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


class ConvergenceOrderTests(unittest.TestCase):
    """Hermite event location keeps the classical order of each method."""

    def test_hermite_location_observes_the_classical_order(self) -> None:
        """Euler 1, Heun 2, RK4 4 on decay and logistic growth."""
        for problem in ("decay", "logistic"):
            for method in ODE_METHODS:
                with self.subTest(problem=problem, method=method):
                    orders = observed_orders(time_errors(problem, method, "hermite"))
                    self.assertAlmostEqual(orders[-1], METHOD_ORDERS[method], delta=0.4)

    def test_oscillator_orders(self) -> None:
        """Heun and RK4 keep orders 2 and 4 on the harmonic oscillator."""
        for method in ("heun", "rk4"):
            with self.subTest(method=method):
                orders = observed_orders(time_errors("oscillator", method, "hermite"))
                for order in orders:
                    self.assertAlmostEqual(order, METHOD_ORDERS[method], delta=0.2)

    def test_linear_location_caps_rk4(self) -> None:
        """Linear interpolation leaves an ``O(h^2)`` error far above RK4's."""
        for problem in ("decay", "logistic"):
            with self.subTest(problem=problem):
                linear = time_errors(problem, "rk4", "linear")
                hermite = time_errors(problem, "rk4", "hermite")
                self.assertGreater(linear[-1], 50.0 * hermite[-1])
                for h, error in zip(STEPS, linear, strict=True):
                    self.assertLess(error, h * h)

    def test_linear_location_keeps_low_orders(self) -> None:
        """Euler and Heun keep orders 1 and 2 with linear location."""
        for method in ("euler", "heun"):
            with self.subTest(method=method):
                orders = observed_orders(time_errors("logistic", method, "linear"))
                self.assertAlmostEqual(orders[-1], METHOD_ORDERS[method], delta=0.2)

    def test_step_location_reports_the_end_of_the_crossing_step(self) -> None:
        """The first grid point past the event lies at most one step late."""
        rhs, start, event, exact = PROBLEMS["logistic"]
        for h in STEPS:
            with self.subTest(step=h):
                solution = integrate_to_event(
                    rhs, start, event, step=h, location="step"
                )
                self.assertAlmostEqual(solution.time, solution.steps * h, delta=1e-12)
                self.assertGreaterEqual(solution.time, exact - 1e-8)
                self.assertLess(solution.time, exact + h)
                self.assertLessEqual(event(solution.time, solution.state), 0.0)

    def test_state_at_the_event(self) -> None:
        """The oscillator crosses ``x = 0`` with velocity ``-1``."""
        rhs, start, event, _ = PROBLEMS["oscillator"]
        solution = integrate_to_event(rhs, start, event, step=0.01)
        self.assertLess(abs(solution.state[0]), 1e-9)
        self.assertAlmostEqual(solution.state[1], -1.0, delta=1e-8)


class ExactnessAndBookkeepingTests(unittest.TestCase):
    """Polynomial solutions, time offsets, work counts, and the path."""

    def test_quadratic_solution_is_reproduced(self) -> None:
        """Heun and RK4 with Hermite location are exact for ``y' = t``."""
        for method in ("heun", "rk4"):
            with self.subTest(method=method):
                solution = integrate_to_event(
                    lambda t, y: [t],
                    [0.0],
                    lambda t, y: 2.0 - y[0],
                    step=0.3,
                    method=method,
                )
                self.assertAlmostEqual(solution.time, 2.0, delta=1e-12)

    def test_start_time_shifts_the_solution(self) -> None:
        """A non-zero start time is honoured by time-dependent problems."""
        solution = integrate_to_event(
            lambda t, y: [t], [0.5], lambda t, y: 2.0 - y[0], step=0.1, start=1.0
        )
        self.assertAlmostEqual(solution.time, 2.0, delta=1e-12)

    def test_evaluation_counts(self) -> None:
        """Counts follow the stages of each method and the location."""
        expected = {
            ("euler", "linear"): lambda n: n,
            ("heun", "linear"): lambda n: 2 * n,
            ("rk4", "step"): lambda n: 4 * n,
            ("rk4", "hermite"): lambda n: 4 * n + 1,
        }
        for (method, location), count in expected.items():
            with self.subTest(method=method, location=location):
                solution = integrate_to_event(
                    decay,
                    [1.0],
                    lambda t, y: y[0] - 0.5,
                    step=0.07,
                    method=method,
                    location=location,
                )
                self.assertEqual(solution.evaluations, count(solution.steps))
                self.assertEqual(solution.method, method)
                self.assertEqual(solution.location, location)
                self.assertEqual(solution.step, 0.07)

    def test_recorded_path(self) -> None:
        """The path holds the start, every grid point, and the event."""
        solution = integrate_to_event(
            decay, [1.0], lambda t, y: y[0] - 0.5, step=0.07, record=True
        )
        self.assertIsInstance(solution, EventSolution)
        self.assertEqual(len(solution.path), solution.steps + 2)
        self.assertEqual(solution.path[0], (0.0, (1.0,)))
        self.assertEqual(solution.path[-1], (solution.time, solution.state))
        grid = [t for t, _ in solution.path[:-1]]
        self.assertEqual(grid, sorted(grid))
        self.assertLessEqual(grid[-2], solution.time)
        self.assertLessEqual(solution.time, grid[-1])
        unrecorded = integrate_to_event(
            decay, [1.0], lambda t, y: y[0] - 0.5, step=0.07
        )
        self.assertEqual(unrecorded.path, ())

    def test_exported_constants(self) -> None:
        """Methods, locations, and orders are consistent."""
        self.assertEqual(set(METHOD_ORDERS), set(ODE_METHODS))
        self.assertEqual(EVENT_LOCATIONS, ("step", "linear", "hermite"))


class ValidationTests(unittest.TestCase):
    """Invalid arguments are rejected with clear errors."""

    def test_invalid_arguments(self) -> None:
        """Steps, methods, locations, limits, and states are checked."""
        event = lambda t, y: y[0] - 0.5  # noqa: E731
        cases = (
            {"step": 0.0},
            {"step": math.inf},
            {"step": 0.1, "method": "rk45"},
            {"step": 0.1, "location": "cubic"},
            {"step": 0.1, "max_steps": 0},
            {"step": 0.1, "max_steps": True},
            {"step": 0.1, "start": math.nan},
        )
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    integrate_to_event(decay, [1.0], event, **options)
        with self.assertRaises(ValueError):
            integrate_to_event(decay, [], event, step=0.1)
        with self.assertRaises(ValueError):
            integrate_to_event(decay, [math.nan], event, step=0.1)
        with self.assertRaises(ValueError):
            integrate_to_event(decay, [0.4], event, step=0.1)
        with self.assertRaises(ValueError):
            integrate_to_event(lambda t, y: [1.0, 2.0], [1.0], event, step=0.1)

    def test_blow_up_is_reported(self) -> None:
        """A state that overflows raises instead of returning garbage."""
        with self.assertRaises(ValueError):
            integrate_to_event(
                lambda t, y: [y[0] * y[0]],
                [1.0],
                lambda t, y: 1.0,
                step=0.5,
                method="euler",
            )

    def test_missing_event_raises(self) -> None:
        """No crossing within ``max_steps`` raises ``RuntimeError``."""
        with self.assertRaises(RuntimeError):
            integrate_to_event(
                decay, [1.0], lambda t, y: y[0] + 1.0, step=0.1, max_steps=50
            )


if __name__ == "__main__":
    unittest.main()
