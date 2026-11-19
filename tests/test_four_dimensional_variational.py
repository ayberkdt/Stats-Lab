"""Tests for strong-constraint, incremental, and weak-constraint 4D-Var."""

from __future__ import annotations

import math
import random
import unittest

from probstat.data_assimilation import (
    FourDimensionalVariationalResult,
    four_dimensional_variational,
    four_dimensional_variational_cost,
    incremental_four_dimensional_variational,
    weak_constraint_four_dimensional_variational,
)
from probstat.filtering import (
    LinearGaussianStateSpace,
    kalman_filter,
    lorenz63,
    lorenz63_rhs,
    rts_smoother,
    simulate,
)
from probstat.numerics import AutodiffUnsupportedError, integrate_fixed_step

PRIOR_MEAN = (0.0, 1.0)
PRIOR_COVARIANCE = ((1.0, 0.0), (0.0, 0.5))
R_POSITION = ((1.0,),)


def _constant_velocity(q: float) -> LinearGaussianStateSpace:
    """Return the constant-velocity model with white-noise acceleration ``q``."""
    return LinearGaussianStateSpace(
        ((1.0, 1.0), (0.0, 1.0)),
        ((q / 3.0, q / 2.0), (q / 2.0, q)),
        ((1.0, 0.0),),
        R_POSITION,
        PRIOR_MEAN,
        PRIOR_COVARIANCE,
    )


def _step(state):
    """Constant-velocity transition written with operators only."""
    return (state[0] + state[1], state[1])


def _position(state):
    """Observe the position."""
    return (state[0],)


def _data(steps: int = 10):
    """Return seeded observations with one missing entry."""
    observations = list(simulate(_constant_velocity(0.05), steps, seed=7).observations)
    observations[3] = None
    return observations


def _diagonal(size: int, value: float):
    """Return ``value * I``."""
    return tuple(
        tuple(value if i == j else 0.0 for j in range(size)) for i in range(size)
    )


def _assert_close(test, actual, expected, *, rel=1e-8, abs_tol=1e-10):
    """Compare nested sequences of floats."""
    if isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True):
            _assert_close(test, a, b, rel=rel, abs_tol=abs_tol)
        return
    test.assertTrue(
        math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
        f"{actual!r} != {expected!r}",
    )


_LORENZ = lorenz63(substeps=5)
_SPUN_UP = integrate_fixed_step(
    lorenz63_rhs, 0.0, (1.0, 1.0, 1.0), step=0.01, steps=1000
).final_state


def _lorenz_twin(seed: int, window: int, *, variance: float = 0.25):
    """Return truth, background, and observations of a Lorenz-63 twin run."""
    rng = random.Random(seed)
    truth = [tuple(_SPUN_UP)]
    for _ in range(window):
        truth.append(tuple(_LORENZ.transition_function(truth[-1])))
    background = tuple(v + 2.0 * rng.gauss(0.0, 1.0) for v in truth[0])
    observations = [
        tuple(v + math.sqrt(variance) * rng.gauss(0.0, 1.0) for v in truth[k])
        for k in range(1, window + 1)
    ]
    return truth, background, observations


class GradientTests(unittest.TestCase):
    """The reverse-mode gradient is the derivative of the cost."""

    def setUp(self) -> None:
        """Build a short Lorenz-63 window with a missing observation."""
        _, self.background, observations = _lorenz_twin(3, 6)
        observations[2] = None
        self.observations = observations
        self.b = _diagonal(3, 4.0)
        self.r = _diagonal(3, 0.25)

    def _taylor(self, cost, point, direction) -> list[float]:
        """Return ``|ratio - 1|`` of the first-order Taylor test per epsilon."""
        value, gradient = cost.value_and_gradient(point)
        slope = math.fsum(g * d for g, d in zip(gradient, direction, strict=True))
        self.assertGreater(abs(slope), 1e-3)
        errors = []
        for epsilon in (1e-2, 1e-3, 1e-4, 1e-5):
            shifted = tuple(
                p + epsilon * d for p, d in zip(point, direction, strict=True)
            )
            errors.append(abs((cost(shifted) - value) / (epsilon * slope) - 1.0))
        return errors

    def test_taylor_test_strong_constraint(self) -> None:
        """The Taylor ratio tends to one linearly in epsilon."""
        cost = four_dimensional_variational_cost(
            self.background,
            self.b,
            self.observations,
            _LORENZ.transition_function,
            _LORENZ.observation_function,
            self.r,
        )
        errors = self._taylor(cost, self.background, (0.3, -0.7, 0.5))
        self.assertLess(errors[-1], 1e-4)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertLess(fine, 0.2 * coarse)

    def test_taylor_test_weak_constraint(self) -> None:
        """The weak-constraint gradient passes the same Taylor test."""
        cost = four_dimensional_variational_cost(
            self.background,
            self.b,
            self.observations,
            _LORENZ.transition_function,
            _LORENZ.observation_function,
            self.r,
            process_covariance=_diagonal(3, 0.1),
        )
        rng = random.Random(5)
        states = [self.background]
        for _ in range(len(self.observations)):
            states.append(_LORENZ.transition_function(states[-1]))
        point = tuple(v + rng.gauss(0.0, 0.3) for state in states for v in state)
        direction = tuple(rng.uniform(-1.0, 1.0) for _ in point)
        errors = self._taylor(cost, point, direction)
        self.assertLess(errors[-1], 1e-4)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertLess(fine, 0.2 * coarse)

    def test_reverse_gradient_matches_central_differences(self) -> None:
        """Adjoint and central-difference gradients agree."""
        cost = four_dimensional_variational_cost(
            self.background,
            self.b,
            self.observations,
            _LORENZ.transition_function,
            _LORENZ.observation_function,
            self.r,
        )
        reverse = cost.gradient(self.background)
        central = cost.gradient(self.background, gradient="central")
        scale = max(abs(g) for g in reverse)
        for a, b in zip(reverse, central, strict=True):
            self.assertLess(abs(a - b), 1e-6 * scale)

    def test_analytic_observation_jacobian_gives_the_same_gradient(self) -> None:
        """Supplying ``dh/dx`` changes how, not what, is differentiated."""
        selection = _diagonal(3, 1.0)
        for process in (None, _diagonal(3, 0.1)):
            plain = four_dimensional_variational_cost(
                self.background,
                self.b,
                self.observations,
                _LORENZ.transition_function,
                _LORENZ.observation_function,
                self.r,
                process_covariance=process,
            )
            analytic = four_dimensional_variational_cost(
                self.background,
                self.b,
                self.observations,
                _LORENZ.transition_function,
                _LORENZ.observation_function,
                self.r,
                process_covariance=process,
                observation_jacobian=lambda x: selection,
            )
            repeated = self.background * (plain.control_dimension // 3)
            point = tuple(v + 0.1 * i for i, v in enumerate(repeated))
            value, gradient = plain.value_and_gradient(point)
            value_a, gradient_a = analytic.value_and_gradient(point)
            _assert_close(self, value_a, value, rel=1e-12)
            _assert_close(self, gradient_a, gradient, rel=1e-10, abs_tol=1e-9)

    def test_non_differentiable_model_needs_explicit_central(self) -> None:
        """A ``math``-based model raises in reverse mode and works with central."""

        def stepper(state):
            return (state[0] + 0.1 * math.sin(state[1]), state[1])

        arguments = (
            (0.2, 0.4),
            PRIOR_COVARIANCE,
            [(0.5,), (0.7,)],
            stepper,
            _position,
            R_POSITION,
        )
        with self.assertRaisesRegex(AutodiffUnsupportedError, "central"):
            four_dimensional_variational(*arguments)
        result = four_dimensional_variational(*arguments, gradient="central")
        self.assertTrue(result.converged)
        self.assertLess(result.gradient_norm, 1e-6)


class LinearGaussianEquivalenceTests(unittest.TestCase):
    """4D-Var equals the Kalman smoother for linear-Gaussian models."""

    def test_strong_constraint_equals_rts_smoother_at_time_zero(self) -> None:
        """Without process noise the 4D-Var analysis is the smoothed ``x_0``."""
        observations = _data()
        smoothed = rts_smoother(
            kalman_filter(_constant_velocity(0.0), observations)
        ).smoothed
        result = four_dimensional_variational(
            PRIOR_MEAN,
            PRIOR_COVARIANCE,
            observations,
            _step,
            _position,
            R_POSITION,
        )
        self.assertIsInstance(result, FourDimensionalVariationalResult)
        self.assertEqual(result.formulation, "strong")
        self.assertTrue(result.converged)
        _assert_close(self, result.analysis, smoothed[0].mean)
        _assert_close(self, result.analysis_covariance, smoothed[0].covariance)
        self.assertEqual(len(result.trajectory), len(observations) + 1)
        for k, state in enumerate(smoothed):
            _assert_close(self, result.trajectory[k], state.mean)
            _assert_close(self, result.trajectory_covariances[k], state.covariance)
        self.assertEqual(result.model_cost, 0.0)
        _assert_close(
            self,
            result.cost,
            result.background_cost + result.observation_cost,
            rel=1e-14,
        )

    def test_central_differences_reach_the_same_analysis(self) -> None:
        """The explicit finite-difference mode converges to the same answer."""
        observations = _data()
        smoothed = rts_smoother(
            kalman_filter(_constant_velocity(0.0), observations)
        ).smoothed
        result = four_dimensional_variational(
            PRIOR_MEAN,
            PRIOR_COVARIANCE,
            observations,
            _step,
            _position,
            R_POSITION,
            gradient="central",
            gradient_tolerance=1e-7,
        )
        _assert_close(self, result.analysis, smoothed[0].mean, rel=1e-6, abs_tol=1e-7)
        _assert_close(
            self,
            result.analysis_covariance,
            smoothed[0].covariance,
            rel=1e-6,
            abs_tol=1e-8,
        )

    def test_weak_constraint_equals_rts_smoother_over_the_window(self) -> None:
        """With ``Q > 0`` the weak-constraint trajectory is the smoother."""
        model = _constant_velocity(0.05)
        observations = _data()
        smoothed = rts_smoother(kalman_filter(model, observations)).smoothed
        result = weak_constraint_four_dimensional_variational(
            PRIOR_MEAN,
            PRIOR_COVARIANCE,
            observations,
            _step,
            _position,
            R_POSITION,
            model.process_covariance,
        )
        self.assertEqual(result.formulation, "weak")
        self.assertTrue(result.converged)
        for k, state in enumerate(smoothed):
            _assert_close(self, result.trajectory[k], state.mean, abs_tol=1e-8)
            _assert_close(self, result.trajectory_covariances[k], state.covariance)
        self.assertGreater(result.model_cost, 0.0)
        _assert_close(
            self,
            result.cost,
            result.background_cost + result.model_cost + result.observation_cost,
            rel=1e-14,
        )

    def test_incremental_solves_a_linear_problem_in_one_outer_loop(self) -> None:
        """Gauss-Newton is exact for a quadratic cost."""
        observations = _data()
        smoothed = rts_smoother(
            kalman_filter(_constant_velocity(0.0), observations)
        ).smoothed
        result = incremental_four_dimensional_variational(
            PRIOR_MEAN,
            PRIOR_COVARIANCE,
            observations,
            _step,
            _position,
            R_POSITION,
        )
        self.assertEqual(result.formulation, "incremental")
        self.assertEqual(result.iterations, 1)
        self.assertTrue(result.converged)
        self.assertEqual(len(result.outer_costs), 2)
        self.assertLess(result.outer_costs[1], result.outer_costs[0])
        _assert_close(self, result.analysis, smoothed[0].mean, rel=1e-10)
        _assert_close(self, result.analysis_covariance, smoothed[0].covariance)
        limited = incremental_four_dimensional_variational(
            PRIOR_MEAN,
            PRIOR_COVARIANCE,
            observations,
            _step,
            _position,
            R_POSITION,
            outer_loops=1,
        )
        _assert_close(self, limited.analysis, result.analysis, rel=1e-14)

    def test_window_without_observations_returns_the_background(self) -> None:
        """Missing observations everywhere leave ``x_b`` and ``B``."""
        result = four_dimensional_variational(
            PRIOR_MEAN,
            PRIOR_COVARIANCE,
            [None, None, None],
            _step,
            _position,
            R_POSITION,
        )
        _assert_close(self, result.analysis, PRIOR_MEAN)
        _assert_close(self, result.analysis_covariance, PRIOR_COVARIANCE)
        self.assertEqual(result.observation_cost, 0.0)
        self.assertEqual(result.cost, 0.0)


class LorenzTests(unittest.TestCase):
    """Nonlinear behaviour on Lorenz-63."""

    def test_incremental_reaches_the_full_minimum(self) -> None:
        """Outer loops converge to the minimizer found by BFGS."""
        _, background, observations = _lorenz_twin(4, 4)
        arguments = (
            background,
            _diagonal(3, 4.0),
            observations,
            _LORENZ.transition_function,
            _LORENZ.observation_function,
            _diagonal(3, 0.25),
        )
        full = four_dimensional_variational(*arguments)
        incremental = incremental_four_dimensional_variational(*arguments)
        self.assertTrue(full.converged)
        self.assertTrue(incremental.converged)
        self.assertGreater(incremental.iterations, 1)
        self.assertLess(math.dist(full.analysis, incremental.analysis), 1e-6)
        self.assertLess(abs(full.cost - incremental.cost), 1e-8 * full.cost)
        self.assertAlmostEqual(incremental.outer_costs[-1], incremental.cost)

    def test_twin_experiment_reduces_the_initial_error(self) -> None:
        """A short window removes most of the background error."""
        b = _diagonal(3, 4.0)
        r = _diagonal(3, 0.25)
        ratios = []
        for seed in range(5):
            truth, background, observations = _lorenz_twin(seed, 10)
            observations[2] = None
            observations[6] = None
            result = four_dimensional_variational(
                background,
                b,
                observations,
                _LORENZ.transition_function,
                _LORENZ.observation_function,
                r,
            )
            self.assertTrue(result.converged)
            ratios.append(
                math.dist(result.analysis, truth[0]) / math.dist(background, truth[0])
            )
            forecast = [background]
            for _ in range(10):
                forecast.append(_LORENZ.transition_function(forecast[-1]))
            analysis_error = math.fsum(
                math.dist(a, t) for a, t in zip(result.trajectory, truth, strict=True)
            )
            background_error = math.fsum(
                math.dist(a, t) for a, t in zip(forecast, truth, strict=True)
            )
            self.assertLess(analysis_error, 0.5 * background_error)
        self.assertLess(math.fsum(ratios) / len(ratios), 0.4)


class ValidationTests(unittest.TestCase):
    """Invalid inputs raise clear errors."""

    def _call(self, function, **changes):
        """Call a solver with one argument replaced."""
        arguments = {
            "background": PRIOR_MEAN,
            "background_covariance": PRIOR_COVARIANCE,
            "observations": [(1.0,), None],
            "transition": _step,
            "observation_function": _position,
            "observation_covariance": R_POSITION,
        }
        keywords = {}
        for name, value in changes.items():
            if name in arguments:
                arguments[name] = value
            else:
                keywords[name] = value
        return function(*arguments.values(), **keywords)

    def test_weak_constraint_requires_positive_definite_q(self) -> None:
        """A singular ``Q`` is refused with a pointer to the strong form."""
        with self.assertRaisesRegex(ValueError, "process_covariance must be positive"):
            self._call(
                weak_constraint_four_dimensional_variational,
                process_covariance=((0.0, 0.0), (0.0, 0.0)),
            )

    def test_bad_inputs(self) -> None:
        """Shapes, covariances, and options are validated."""
        function = four_dimensional_variational
        with self.assertRaisesRegex(ValueError, "background_covariance"):
            self._call(function, background_covariance=((1.0, 2.0), (2.0, 1.0)))
        with self.assertRaisesRegex(ValueError, "observation_covariance"):
            self._call(function, observation_covariance=((0.0,),))
        with self.assertRaisesRegex(ValueError, "observation 1 has length"):
            self._call(function, observations=[(1.0, 2.0)])
        with self.assertRaisesRegex(ValueError, "at least one"):
            self._call(function, observations=[])
        with self.assertRaisesRegex(TypeError, "observation 1"):
            self._call(function, observations=[1.0])
        with self.assertRaisesRegex(ValueError, "gradient must be one of"):
            self._call(function, gradient="forward")
        with self.assertRaisesRegex(ValueError, "tangent_linear must be one of"):
            self._call(incremental_four_dimensional_variational, tangent_linear="x")
        with self.assertRaisesRegex(ValueError, "transition must return"):
            self._call(function, transition=lambda x: (x[0],))
        with self.assertRaisesRegex(TypeError, "transition must be callable"):
            self._call(function, transition=None)
        cost = four_dimensional_variational_cost(
            PRIOR_MEAN, PRIOR_COVARIANCE, [(1.0,)], _step, _position, R_POSITION
        )
        with self.assertRaisesRegex(ValueError, "control must have length 2"):
            cost((1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
