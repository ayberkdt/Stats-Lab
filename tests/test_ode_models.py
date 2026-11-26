"""Tests for ``probstat.inverse.ode_models`` (ODE adjoints, adapter, heat, N03).

Covers the ODE tangent/adjoint models, the simulator adapter, the heat
coefficient inversion and the N03 damped-oscillator benchmark.

Order: closed-form RK4 (exponential decay has a polynomial one-step map, so the
discrete value and both discrete derivatives are known exactly), independent
oracles (central differences, normal equations through ``probstat.linalg``),
checkpointing, then adversarial inputs.

Tolerances. The tangent is the derivative of the discrete map, so the adjoint
dot test only sees rounding: relative error below 1e-12 for these small
systems. A central difference with step ``e`` has truncation ``O(e^2)`` and
rounding ``eps / e``; ``e = 1e-5`` gives about 1e-9 relative, and 1e-6 is used
as the comparison bound. Stochastic checks use fixed seeds and a band of four
Monte Carlo standard errors.

"""

from __future__ import annotations

import math
import os
import random
import unittest
from collections.abc import Sequence

from probstat.foundations.random_streams import resolve_stream
from probstat.inverse import check_derivatives, gauss_newton_krylov
from probstat.inverse.benchmark_extended import NonlinearBenchmark
from probstat.inverse.gauss_newton import DifferentiableForward
from probstat.inverse.ode_models import (
    HeatCoefficientModel,
    ODEForwardModel,
    ODESystem,
    SimulatorAdapter,
    SimulatorFailureError,
    damped_oscillator_benchmark,
    damped_oscillator_system,
    exponential_decay_system,
    heat_coefficient_benchmark,
)
from probstat.linalg import (
    DenseCovariance,
    inverse_spd,
    singular_value_decomposition,
    solve_spd,
    symmetric_eigen,
)

Vector = tuple[float, ...]


def _rk4_factor(z: float) -> float:
    """Return the stability polynomial ``R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24``."""
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def _rk4_factor_derivative(z: float) -> float:
    """Return ``R'(z)``."""
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0


def _relative_dot_error(
    model: DifferentiableForward, x: Vector, v: Vector, w: Vector
) -> float:
    """Return ``|w'Jv - (J'w)'v| / (||w|| ||Jv|| + ||J'w|| ||v||)``."""
    jv = model.jvp(x, v)
    jtw = model.vjp(x, w)
    lhs = sum(a * b for a, b in zip(w, jv, strict=True))
    rhs = sum(a * b for a, b in zip(jtw, v, strict=True))
    scale = math.hypot(*w) * math.hypot(*jv) + math.hypot(*jtw) * math.hypot(*v)
    return abs(lhs - rhs) / scale


def _central_gradient(model: DifferentiableForward, x: Vector, w: Vector) -> Vector:
    """Return the central-difference gradient of ``w'G`` with step 1e-5."""
    out = []
    for j in range(len(x)):
        e = 1e-5 * (1.0 + abs(x[j]))
        plus = tuple(a + (e if i == j else 0.0) for i, a in enumerate(x))
        minus = tuple(a - (e if i == j else 0.0) for i, a in enumerate(x))
        fp, fm = model.value(plus), model.value(minus)
        out.append(
            sum(c * (a - b) for c, a, b in zip(w, fp, fm, strict=True)) / (2 * e)
        )
    return tuple(out)


def _random_vector(rng: random.Random, size: int) -> Vector:
    """Return a standard normal vector."""
    return tuple(rng.gauss(0.0, 1.0) for _ in range(size))


def _spring_system() -> ODESystem:
    """Return ``x'' = -theta x`` with state ``(x, x')`` and one parameter."""

    def rhs(t: float, x: Vector, theta: Vector, seg: int) -> Sequence[float]:
        """Return ``(x', -theta x)``."""
        return (x[1], -theta[0] * x[0])

    def fx(t: float, x: Vector, theta: Vector, seg: int) -> Sequence[Sequence[float]]:
        """Return the state Jacobian."""
        return ((0.0, 1.0), (-theta[0], 0.0))

    def fp(t: float, x: Vector, theta: Vector, seg: int) -> Sequence[Sequence[float]]:
        """Return the parameter Jacobian."""
        return ((0.0,), (-x[0],))

    return ODESystem(2, 1, rhs, fx, fp)


def _van_der_pol_system() -> ODESystem:
    """Return a nonlinear, time-dependent forced van der Pol system (p = 2)."""

    def rhs(t: float, x: Vector, th: Vector, seg: int) -> Sequence[float]:
        """Return the vector field."""
        return (x[1], th[0] * (1.0 - x[0] ** 2) * x[1] - x[0] + th[1] * math.cos(t))

    def fx(t: float, x: Vector, th: Vector, seg: int) -> Sequence[Sequence[float]]:
        """Return the state Jacobian."""
        return (
            (0.0, 1.0),
            (-2.0 * th[0] * x[0] * x[1] - 1.0, th[0] * (1.0 - x[0] ** 2)),
        )

    def fp(t: float, x: Vector, th: Vector, seg: int) -> Sequence[Sequence[float]]:
        """Return the parameter Jacobian."""
        return ((0.0, 0.0), ((1.0 - x[0] ** 2) * x[1], math.cos(t)))

    return ODESystem(2, 2, rhs, fx, fp)


class ExponentialDecayClosedForm(unittest.TestCase):
    """RK4 of ``x' = -theta x``: the one-step map is ``x R(-theta h)``."""

    def test_value_and_discrete_derivatives_are_the_polynomial_closed_forms(self):
        """Value and discrete derivatives are the polynomial closed forms."""
        steps = 5
        times = (0.4, 1.0)
        model = ODEForwardModel(
            exponential_decay_system(), times, ((1.0,),), steps_per_interval=steps
        )
        x0, theta = 1.7, 0.9
        z = (x0, theta)
        y = model.value(z)
        # the observation at t_j accumulates steps*j steps of its own size
        h = [times[0] / steps, (times[1] - times[0]) / steps]
        r = [_rk4_factor(-theta * hh) for hh in h]
        expect = [x0 * r[0] ** steps, x0 * r[0] ** steps * r[1] ** steps]
        for got, want in zip(y, expect, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-14)
        # d/dx0 and d/dtheta of the discrete map
        dr = [-hh * _rk4_factor_derivative(-theta * hh) for hh in h]
        d_x0 = [r[0] ** steps, r[0] ** steps * r[1] ** steps]
        d_theta = [
            x0 * steps * r[0] ** (steps - 1) * dr[0],
            x0
            * r[0] ** steps
            * r[1] ** steps
            * (steps * dr[0] / r[0] + steps * dr[1] / r[1]),
        ]
        for j in range(2):
            e0 = tuple(1.0 if i == j else 0.0 for i in range(2))
            col = model.jvp(z, (1.0, 0.0))
            self.assertAlmostEqual(col[j], d_x0[j], delta=1e-13)
            col = model.jvp(z, (0.0, 1.0))
            self.assertAlmostEqual(col[j], d_theta[j], delta=1e-13)
            self.assertEqual(len(e0), 2)
        # the adjoint of the unit output vectors reproduces the same numbers
        for j in range(2):
            w = tuple(1.0 if i == j else 0.0 for i in range(2))
            g = model.vjp(z, w)
            self.assertAlmostEqual(g[0], d_x0[j], delta=1e-13)
            self.assertAlmostEqual(g[1], d_theta[j], delta=1e-13)

    def test_value_converges_to_the_exact_flow_with_order_four(self):
        """Value converges to the exact flow with order four."""
        errors = []
        for steps in (2, 4, 8):
            model = ODEForwardModel(
                exponential_decay_system(), (2.0,), ((1.0,),), steps_per_interval=steps
            )
            errors.append(abs(model.value((1.0, 1.3))[0] - math.exp(-2.6)))
        orders = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
        for order in orders:
            self.assertGreater(order, 3.6)
            self.assertLess(order, 5.5)

    def test_observation_at_initial_time_returns_the_initial_state(self):
        """Observation at initial time returns the initial state."""
        model = ODEForwardModel(
            exponential_decay_system(), (0.0, 1.0), ((1.0,),), steps_per_interval=2
        )
        y = model.value((3.0, 0.5))
        self.assertEqual(y[0], 3.0)
        self.assertEqual(model.jvp((3.0, 0.5), (1.0, 0.0))[0], 1.0)
        self.assertEqual(model.vjp((3.0, 0.5), (1.0, 0.0))[0], 1.0)

    def test_fixed_initial_state_has_only_parameters(self):
        """Fixed initial state has only parameters."""
        model = ODEForwardModel(
            exponential_decay_system(),
            (1.0,),
            ((1.0,),),
            initial_state=(2.0,),
            estimate_initial_state=False,
            steps_per_interval=4,
        )
        self.assertEqual(model.input_dimension, 1)
        self.assertAlmostEqual(model.value((0.7,))[0], 2.0 * math.exp(-0.7), delta=1e-5)
        self.assertTrue(isinstance(model, DifferentiableForward))


class DiscreteAdjointAndTangent(unittest.TestCase):
    """Dot test, Taylor order and finite-difference gradients."""

    def setUp(self):
        """Seed the direction generator."""
        self.rng = random.Random(7)

    def _models(self):
        times = (0.5, 1.1, 1.4, 2.5)
        obs = ((1.0, 0.0), (0.3, -0.7))
        return {
            "spring": (
                ODEForwardModel(_spring_system(), times, obs, steps_per_interval=6),
                (1.0, -0.4, 2.2),
            ),
            "van_der_pol": (
                ODEForwardModel(
                    _van_der_pol_system(), times, obs, steps_per_interval=7
                ),
                (0.8, 0.3, 1.1, 0.4),
            ),
        }

    def test_dot_test_to_rounding_for_linear_and_nonlinear_systems(self):
        """Dot test to rounding for linear and nonlinear systems."""
        for name, (model, x) in self._models().items():
            for _ in range(3):
                v = _random_vector(self.rng, model.input_dimension)
                w = _random_vector(self.rng, model.output_dimension)
                self.assertLess(_relative_dot_error(model, x, v, w), 1e-12, name)

    def test_taylor_remainder_is_second_order_down_to_rounding(self):
        """Taylor remainder is second order down to rounding."""
        for name, (model, x) in self._models().items():
            v = _random_vector(self.rng, model.input_dimension)
            w = _random_vector(self.rng, model.output_dimension)
            check = check_derivatives(model, x, v, w, dot_tolerance=1e-12)
            self.assertTrue(check.taylor_passed, name)
            self.assertTrue(check.adjoint_passed, name)
            good = [o for o in check.orders if not math.isnan(o)]
            self.assertGreaterEqual(len(good), 3, name)
            self.assertTrue(all(1.8 < o < 2.2 for o in good[:3]), (name, good))

    def test_adjoint_gradient_matches_central_differences_of_the_discrete_map(self):
        """Adjoint gradient matches central differences of the discrete map."""
        for name, (model, x) in self._models().items():
            w = _random_vector(self.rng, model.output_dimension)
            adjoint = model.vjp(x, w)
            numeric = _central_gradient(model, x, w)
            scale = max(1.0, math.hypot(*numeric))
            err = math.hypot(*(a - b for a, b in zip(adjoint, numeric, strict=True)))
            self.assertLess(err / scale, 1e-6, name)

    def test_discrete_derivative_differs_from_nothing_but_the_scheme(self):
        """Discrete derivative differs from nothing but the scheme."""
        # sensitivity to the true flow is O(h^4) away; to the scheme it is exact
        coarse = ODEForwardModel(
            exponential_decay_system(), (2.0,), ((1.0,),), steps_per_interval=3
        )
        exact_flow_derivative = -2.0 * math.exp(-2.0 * 0.8)  # d/dtheta x0 e^{-theta t}
        tangent = coarse.jvp((1.0, 0.8), (0.0, 1.0))[0]
        self.assertGreater(abs(tangent - exact_flow_derivative), 1e-5)
        e = 1e-6
        fd = (coarse.value((1.0, 0.8 + e))[0] - coarse.value((1.0, 0.8 - e))[0]) / (
            2 * e
        )
        self.assertAlmostEqual(tangent, fd, delta=1e-8)


class CheckpointingAndCounts(unittest.TestCase):
    """Checkpointed gradients equal full storage; work counts as stated."""

    def _model(self, **kwargs):
        times = tuple(0.25 * (j + 1) for j in range(8))
        return ODEForwardModel(
            _van_der_pol_system(), times, ((1.0, 0.0),), steps_per_interval=5, **kwargs
        )

    def test_checkpointed_gradient_equals_full_storage(self):
        """Checkpointed gradient equals full storage."""
        x = (0.9, -0.2, 0.7, 0.3)
        rng = random.Random(3)
        w = _random_vector(rng, 8)
        reference = self._model().vjp(x, w)
        for c in (2, 3, 7, 40, 100):
            checkpointed = self._model(checkpoint_every=c)
            got = checkpointed.vjp(x, w)
            for a, b in zip(got, reference, strict=True):
                self.assertAlmostEqual(a, b, delta=1e-13 * (1 + abs(b)))
            work = checkpointed.work
            # c >= 40 keeps only x0 and recomputes the other 39 steps
            self.assertGreater(work.recomputed_steps, 0)
            self.assertLessEqual(work.recomputed_steps, 39)

    def test_checkpointing_trades_stored_states_for_recomputation(self):
        """Checkpointing trades stored states for recomputation."""
        x = (0.9, -0.2, 0.7, 0.3)
        w = (1.0,) * 8
        full = self._model()
        sparse = self._model(checkpoint_every=7)
        full.vjp(x, w)
        sparse.vjp(x, w)
        self.assertEqual(full.work.recomputed_steps, 0)
        self.assertGreater(sparse.work.recomputed_steps, 0)
        self.assertLess(sparse.work.peak_stored_states, full.work.peak_stored_states)
        self.assertGreater(sparse.work.rk4_steps, full.work.rk4_steps)

    def test_memory_budget_selects_a_spacing_or_fails(self):
        """Memory budget selects a spacing or fails."""
        budgeted = self._model(max_stored_states=15)
        self.assertGreater(budgeted.checkpoint_every, 1)
        budgeted.vjp((0.9, -0.2, 0.7, 0.3), (1.0,) * 8)
        self.assertLessEqual(budgeted.work.peak_stored_states, 15)
        with self.assertRaises(ValueError):
            self._model(max_stored_states=4)
        with self.assertRaises(ValueError):
            self._model(checkpoint_every=2, max_stored_states=20)

    def test_adjoint_cost_does_not_depend_on_the_parameter_count(self):
        """Adjoint cost does not depend on the parameter count."""
        counts = []
        for pieces in (2, 12):
            system = damped_oscillator_system(
                2.0, 0.1, pieces=pieces, intervals_per_piece=24 // pieces
            )
            times = tuple(0.1 * (j + 1) for j in range(24))
            model = ODEForwardModel(system, times, ((1.0, 0.0),), steps_per_interval=3)
            x = (0.4, 0.1, *([0.2] * pieces))
            model.vjp(x, (1.0,) * 24)
            counts.append(model.work)
        self.assertEqual(counts[0].rhs_evaluations, counts[1].rhs_evaluations)
        self.assertEqual(counts[0].rk4_steps, counts[1].rk4_steps)
        self.assertEqual(counts[0].adjoint_solves, 1)
        # one forward sweep (4 f per step) + one reverse sweep (3 f per step)
        steps = counts[0].rk4_steps
        self.assertEqual(counts[0].rhs_evaluations, 4 * steps + 3 * steps)

    def test_model_counts_are_visible_to_gauss_newton(self):
        """Model counts are visible to gauss newton."""
        model = ODEForwardModel(
            _spring_system(), (1.0, 2.0), ((1.0, 0.0),), steps_per_interval=4
        )
        model.value((1.0, 0.0, 1.0))
        model.jvp((1.0, 0.0, 1.0), (1.0, 0.0, 0.0))
        model.vjp((1.0, 0.0, 1.0), (1.0, 0.0))
        self.assertEqual(
            (model.counts.value, model.counts.jvp, model.counts.vjp), (1, 1, 1)
        )
        model.reset_counts()
        self.assertEqual(model.counts.value, 0)


class ODEFailureCases(unittest.TestCase):
    """Invalid input and blow-up are reported, never hidden."""

    def test_construction_errors(self):
        """Construction errors."""
        sys_ = exponential_decay_system()
        with self.assertRaises(TypeError):
            ODEForwardModel("system", (1.0,), ((1.0,),))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ODEForwardModel(sys_, (1.0, 1.0), ((1.0,),))
        with self.assertRaises(ValueError):
            ODEForwardModel(sys_, (0.5, 0.2), ((1.0,),))
        with self.assertRaises(ValueError):
            ODEForwardModel(sys_, (-1.0,), ((1.0,),))
        with self.assertRaises(ValueError):
            ODEForwardModel(sys_, (1.0,), ((1.0, 2.0),))
        with self.assertRaises(ValueError):
            ODEForwardModel(sys_, (1.0,), ())
        with self.assertRaises(ValueError):
            ODEForwardModel(sys_, (1.0,), ((1.0,),), steps_per_interval=0)
        with self.assertRaises(TypeError):
            ODEForwardModel(sys_, (1.0,), ((1.0,),), steps_per_interval=True)
        with self.assertRaises(ValueError):
            ODEForwardModel(sys_, (1.0,), ((1.0,),), initial_state=(1.0,))
        with self.assertRaises(ValueError):
            ODEForwardModel(sys_, (1.0,), ((1.0,),), estimate_initial_state=False)
        bad = ODESystem(1, 1, sys_.rhs, sys_.state_jacobian, None)
        with self.assertRaises(ValueError):
            ODEForwardModel(bad, (1.0,), ((1.0,),))

    def test_input_validation_of_evaluations(self):
        """Input validation of evaluations."""
        model = ODEForwardModel(exponential_decay_system(), (1.0,), ((1.0,),))
        with self.assertRaises(ValueError):
            model.value((1.0,))
        with self.assertRaises(ValueError):
            model.value((1.0, math.nan))
        with self.assertRaises(TypeError):
            model.value("ab")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.vjp((1.0, 1.0), (1.0, 2.0))

    def test_blow_up_gives_non_finite_value_and_refused_derivatives(self):
        """Blow up gives non finite value and refused derivatives."""
        model = ODEForwardModel(
            exponential_decay_system(), (1e6,), ((1.0,),), steps_per_interval=16
        )
        # R(z) with z = 40 * 62500 is about 2.6e24; sixteen steps overflow to inf
        y = model.value((1.0, -40.0))
        self.assertFalse(math.isfinite(y[0]))
        with self.assertRaises(ValueError):
            model.jvp((1.0, -40.0), (1.0, 0.0))
        with self.assertRaises(ValueError):
            model.vjp((1.0, -40.0), (1.0,))

    def test_wrong_callback_shape_is_rejected(self):
        """Wrong callback shape is rejected."""
        sys_ = exponential_decay_system()

        def bad_rhs(t, x, th, seg):  # noqa: ANN001, ANN202
            """Return a vector of the wrong length."""
            return (0.0, 0.0)

        model = ODEForwardModel(
            ODESystem(1, 1, bad_rhs, sys_.state_jacobian, sys_.parameter_jacobian),
            (1.0,),
            ((1.0,),),
        )
        with self.assertRaises(ValueError):
            model.value((1.0, 1.0))


class SimulatorAdapterTests(unittest.TestCase):
    """The adapter validates, counts, caches on request and records failures."""

    @staticmethod
    def _forward(x):
        """Return ``(x0^2 + x1, sin(x0))``."""
        return (x[0] ** 2 + x[1], math.sin(x[0]))

    @staticmethod
    def _jvp(x, v):
        """Return the Jacobian-vector product of :meth:`_forward`."""
        return (2 * x[0] * v[0] + v[1], math.cos(x[0]) * v[0])

    @staticmethod
    def _vjp(x, w):
        """Return the adjoint product of :meth:`_forward`."""
        return (2 * x[0] * w[0] + math.cos(x[0]) * w[1], w[0])

    def test_analytic_adapter_is_a_differentiable_forward(self):
        """Analytic adapter is a differentiable forward."""
        adapter = SimulatorAdapter(2, 2, self._forward, jvp=self._jvp, vjp=self._vjp)
        self.assertTrue(isinstance(adapter, DifferentiableForward))
        check = check_derivatives(
            adapter, (0.7, -0.3), (0.5, 1.0), (1.0, -2.0), dot_tolerance=1e-14
        )
        self.assertTrue(check.passed)
        self.assertEqual(
            adapter.derivative_modes, {"jvp": "analytic", "vjp": "analytic"}
        )

    def test_counts_and_default_cache_off(self):
        """Counts and default cache off."""
        calls = []

        def forward(x):
            """Record the call and return a doubled input."""
            calls.append(x)
            return (2.0 * x[0],)

        adapter = SimulatorAdapter(1, 1, forward)
        for _ in range(3):
            adapter.value((1.5,))
        self.assertEqual(len(calls), 3)
        self.assertEqual(adapter.counts.value, 3)
        self.assertEqual(adapter.diagnostics.cache_hits, 0)

    def test_explicit_cache_serves_repeated_inputs_and_is_lru_bounded(self):
        """Explicit cache serves repeated inputs and is lru bounded."""
        calls = []

        def forward(x):
            """Record the call and return the input."""
            calls.append(x)
            return (x[0],)

        adapter = SimulatorAdapter(1, 1, forward, cache_size=2)
        adapter.value((1.0,))
        adapter.value((1.0,))
        adapter.value((2.0,))
        adapter.value((3.0,))  # evicts 1.0
        adapter.value((1.0,))
        self.assertEqual(len(calls), 4)
        self.assertEqual(adapter.diagnostics.cache_hits, 1)
        adapter.clear_cache()
        adapter.value((1.0,))
        self.assertEqual(len(calls), 5)

    def test_exceptions_are_captured_as_structured_failures_then_raised(self):
        """Exceptions are captured as structured failures then raised."""

        def forward(x):
            """Fail for negative input."""
            if x[0] < 0:
                raise RuntimeError("negative input")
            return (x[0],)

        adapter = SimulatorAdapter(1, 1, forward)
        self.assertEqual(adapter.value((1.0,)), (1.0,))
        with self.assertRaises(SimulatorFailureError) as ctx:
            adapter.value((-1.0,))
        failure = ctx.exception.failure
        self.assertEqual(failure.kind, "exception")
        self.assertEqual(failure.exception_type, "RuntimeError")
        self.assertEqual(failure.point, (-1.0,))
        self.assertEqual(failure.operation, "value")
        self.assertEqual(adapter.failures, (failure,))
        self.assertEqual(adapter.diagnostics.failures, 1)

    def test_non_finite_and_malformed_outputs_are_failures(self):
        """Non finite and malformed outputs are failures."""
        outputs = {
            "nan": ((math.nan,), "non_finite_output"),
            "inf": ((math.inf,), "non_finite_output"),
            "length": ((1.0, 2.0), "invalid_output"),
            "type": (("a",), "invalid_output"),
            "scalar": (1.0, "invalid_output"),
        }
        for name, (out, kind) in outputs.items():
            adapter = SimulatorAdapter(1, 1, lambda x, o=out: o)  # type: ignore[misc]
            with self.assertRaises(SimulatorFailureError, msg=name) as ctx:
                adapter.value((0.0,))
            self.assertEqual(ctx.exception.failure.kind, kind, name)

    def test_record_mode_returns_nan_for_value_only(self):
        """Record mode returns nan for value only."""
        adapter = SimulatorAdapter(1, 2, lambda x: (math.nan, 1.0), on_failure="record")
        out = adapter.value((0.0,))
        self.assertTrue(all(math.isnan(v) for v in out))
        self.assertEqual(len(adapter.failures), 1)
        adapter2 = SimulatorAdapter(
            1,
            1,
            lambda x: (x[0],),
            jvp=lambda x, v: (math.inf,),
            vjp=lambda x, w: (w[0],),
            on_failure="record",
        )
        with self.assertRaises(SimulatorFailureError):
            adapter2.jvp((0.0,), (1.0,))

    def test_failed_results_are_not_cached(self):
        """Failed results are not cached."""
        state = {"n": 0}

        def flaky(x):
            """Fail on the first call only."""
            state["n"] += 1
            if state["n"] == 1:
                raise ValueError("transient")
            return (x[0],)

        adapter = SimulatorAdapter(1, 1, flaky, cache_size=4)
        with self.assertRaises(SimulatorFailureError):
            adapter.value((1.0,))
        self.assertEqual(adapter.value((1.0,)), (1.0,))
        self.assertEqual(adapter.value((1.0,)), (1.0,))
        self.assertEqual(state["n"], 2)

    def test_declared_finite_differences_match_analytic_derivatives(self):
        """Declared finite differences match analytic derivatives."""
        analytic = SimulatorAdapter(2, 2, self._forward, jvp=self._jvp, vjp=self._vjp)
        fd = SimulatorAdapter(
            2,
            2,
            self._forward,
            finite_difference_step=1e-5,
            allow_dense_fd_vjp=True,
        )
        x, v, w = (0.7, -0.3), (0.5, 1.0), (1.0, -2.0)
        for a, b in zip(analytic.jvp(x, v), fd.jvp(x, v), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-8)
        for a, b in zip(analytic.vjp(x, w), fd.vjp(x, w), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-8)
        # 2 forward calls per jvp, 2 per column for the dense vjp; reported apart
        self.assertEqual(fd.counts.value, 0)
        self.assertEqual(fd.counts.derivative_value, 2 + 2 * 2)
        self.assertEqual(fd.derivative_modes["jvp"], "finite_difference")

    def test_no_silent_derivative_fallback(self):
        """No silent derivative fallback."""
        adapter = SimulatorAdapter(2, 2, self._forward)
        with self.assertRaises(ValueError):
            adapter.jvp((0.0, 0.0), (1.0, 0.0))
        with self.assertRaises(ValueError):
            adapter.vjp((0.0, 0.0), (1.0, 0.0))
        fd_only = SimulatorAdapter(2, 2, self._forward, finite_difference_step=1e-5)
        fd_only.jvp((0.0, 0.0), (1.0, 0.0))
        with self.assertRaises(ValueError):
            fd_only.vjp((0.0, 0.0), (1.0, 0.0))

    def test_input_validation_and_construction_errors(self):
        """Input validation and construction errors."""
        adapter = SimulatorAdapter(2, 2, self._forward)
        with self.assertRaises(ValueError):
            adapter.value((1.0,))
        with self.assertRaises(ValueError):
            adapter.value((1.0, math.inf))
        with self.assertRaises(TypeError):
            adapter.value((1.0, "x"))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            adapter.value(3.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            SimulatorAdapter(2, 2, "not callable")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SimulatorAdapter(2, 2, self._forward, on_failure="ignore")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SimulatorAdapter(2, 2, self._forward, finite_difference_step=0.0)
        with self.assertRaises(ValueError):
            SimulatorAdapter(2, 2, self._forward, allow_dense_fd_vjp=True)
        with self.assertRaises(ValueError):
            SimulatorAdapter(0, 2, self._forward)
        with self.assertRaises(ValueError):
            SimulatorAdapter(2, 2, self._forward, cache_size=-1)

    def test_adapter_drives_gauss_newton_and_the_input_is_not_aliased(self):
        """Adapter drives gauss newton and the input is not aliased."""
        seen = []

        def forward(x):
            """Record the type of the received point and evaluate."""
            seen.append(type(x))
            return (x[0] + 2 * x[1], x[0] * x[1] + 1.0)

        adapter = SimulatorAdapter(
            2, 2, forward, finite_difference_step=1e-5, allow_dense_fd_vjp=True
        )
        data = forward((1.0, 2.0))
        result = gauss_newton_krylov(
            adapter, data, initial=(0.5, 1.5), max_iterations=30, inner_tolerance=1e-10
        )
        self.assertTrue(result.converged)
        for a, b in zip(result.estimate, (1.0, 2.0), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-5)
        self.assertEqual(set(seen), {tuple})


class HeatCoefficientTests(unittest.TestCase):
    """Closed form, dot test, FD checks and the no-crime Gauss-Newton recovery."""

    @staticmethod
    def _model(nodes=15, steps=20, obs=(5, 10, 20), sensors=(0.2, 0.5, 0.7), basis=3):
        return HeatCoefficientModel(
            nodes,
            steps,
            0.1,
            sensors,
            obs,
            lambda s: math.sin(math.pi * s) + 0.5 * math.sin(3.0 * math.pi * s),
            basis_size=basis,
        )

    def test_constant_diffusivity_reproduces_the_discrete_eigenmode(self):
        """Constant diffusivity reproduces the discrete eigenmode."""
        nodes, steps, final = 15, 12, 0.2
        h = 1.0 / (nodes + 1)
        a = 0.35
        sensor = 5 * h  # exactly on a node
        model = HeatCoefficientModel(
            nodes, steps, final, (sensor,), (4, 12), lambda s: math.sin(math.pi * s)
        )
        kappa = math.exp(a)
        lam = kappa * 4.0 / (h * h) * math.sin(math.pi * h / 2.0) ** 2
        dt = final / steps
        y = model.value((a, 0.0, 0.0))
        for step, got in zip((4, 12), y, strict=True):
            want = (1.0 + dt * lam) ** (-step) * math.sin(math.pi * sensor)
            self.assertAlmostEqual(got, want, delta=1e-13)

    def test_dot_test_and_taylor_to_rounding(self):
        """Dot test and taylor to rounding."""
        model = self._model()
        rng = random.Random(5)
        x = (0.2, -0.3, 0.1)
        for _ in range(3):
            v = _random_vector(rng, 3)
            w = _random_vector(rng, model.output_dimension)
            self.assertLess(_relative_dot_error(model, x, v, w), 1e-12)
        check = check_derivatives(
            model, x, (0.4, -0.2, 0.5), _random_vector(rng, 9), dot_tolerance=1e-12
        )
        self.assertTrue(check.passed)
        good = [o for o in check.orders if not math.isnan(o)]
        self.assertTrue(all(1.8 < o < 2.2 for o in good[:3]), good)

    def test_adjoint_gradient_matches_central_differences(self):
        """Adjoint gradient matches central differences."""
        model = self._model()
        rng = random.Random(9)
        x = (0.1, 0.25, -0.2)
        w = _random_vector(rng, model.output_dimension)
        adjoint = model.vjp(x, w)
        numeric = _central_gradient(model, x, w)
        err = math.hypot(*(a - b for a, b in zip(adjoint, numeric, strict=True)))
        self.assertLess(err / max(1.0, math.hypot(*numeric)), 1e-6)

    def test_work_counts_one_factorization_per_derivative_evaluation(self):
        """Work counts one factorization per derivative evaluation."""
        model = self._model()
        x = (0.1, 0.0, 0.0)
        model.value(x)
        model.jvp(x, (1.0, 0.0, 0.0))
        model.vjp(x, (1.0,) * model.output_dimension)
        work = model.work
        self.assertEqual(work.factorizations, 3)
        self.assertEqual(work.tangent_solves, 20)
        self.assertEqual(work.adjoint_solves, 20)
        self.assertEqual(work.forward_solves, 3 * 20)
        self.assertEqual(
            (model.counts.value, model.counts.jvp, model.counts.vjp), (1, 1, 1)
        )

    def test_adjoint_cost_does_not_grow_with_the_number_of_coefficients(self):
        """Adjoint cost does not grow with the number of coefficients."""
        small, large = self._model(basis=2), self._model(basis=8)
        for model in (small, large):
            model.vjp((0.0,) * model.input_dimension, (1.0,) * model.output_dimension)
        self.assertEqual(small.work.adjoint_solves, large.work.adjoint_solves)
        self.assertEqual(small.work.factorizations, large.work.factorizations)

    def test_overflowing_diffusivity_is_a_failed_evaluation(self):
        """Overflowing diffusivity is a failed evaluation."""
        model = self._model()
        y = model.value((800.0, 0.0, 0.0))
        self.assertTrue(all(math.isnan(v) for v in y))
        self.assertEqual(model.failed_evaluations, 1)
        with self.assertRaises(ValueError):
            model.jvp((800.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            model.vjp((-800.0, 0.0, 0.0), (1.0,) * model.output_dimension)

    def test_gauss_newton_recovers_coefficients_with_a_finer_truth(self):
        """Gauss newton recovers coefficients with a finer truth."""
        bench = heat_coefficient_benchmark()
        self.assertIsInstance(bench, NonlinearBenchmark)
        self.assertFalse(bench.same_discretization)
        result = gauss_newton_krylov(
            bench.model,
            bench.data,
            noise=bench.noise,
            initial=(0.0, 0.0, 0.0),
            max_iterations=30,
            inner_tolerance=1e-10,
            gradient_tolerance=1e-10,
        )
        self.assertTrue(result.converged, result.stop_reason)
        error = max(
            abs(a - b) for a, b in zip(result.estimate, bench.truth, strict=True)
        )
        # measured: 0.060 (time-discretization bias of the coarse model), bound 0.1
        self.assertLess(error, 0.1)
        # the model discrepancy at the truth dominates the noise (measured ~10x)
        misfit = math.hypot(
            *(
                a - b
                for a, b in zip(bench.model.value(bench.truth), bench.data, strict=True)
            )
        )
        self.assertGreater(misfit, 5.0 * bench.realized_noise_norm)

    def test_inverse_crime_case_is_flagged_and_recovered_to_noise_level(self):
        """Inverse crime case is flagged and recovered to noise level."""
        bench = heat_coefficient_benchmark(space_refinement=1, time_refinement=1)
        self.assertTrue(bench.same_discretization)
        self.assertEqual(bench.model.value(bench.truth), bench.exact_data)
        result = gauss_newton_krylov(
            bench.model,
            bench.data,
            noise=bench.noise,
            initial=(0.0, 0.0, 0.0),
            max_iterations=30,
            inner_tolerance=1e-10,
            gradient_tolerance=1e-10,
        )
        self.assertTrue(result.converged)
        error = max(
            abs(a - b) for a, b in zip(result.estimate, bench.truth, strict=True)
        )
        # measured 5.0e-4 at noise sd 1e-3 (12 steps of data x 6 sensors)
        self.assertLess(error, 0.01)

    def test_construction_errors(self):
        """Construction errors."""
        u0 = lambda s: math.sin(math.pi * s)  # noqa: E731
        with self.assertRaises(ValueError):
            HeatCoefficientModel(1, 5, 0.1, (0.5,), (5,), u0)
        with self.assertRaises(ValueError):
            HeatCoefficientModel(8, 5, 0.1, (1.0,), (5,), u0)
        with self.assertRaises(ValueError):
            HeatCoefficientModel(8, 5, 0.1, (0.0,), (5,), u0)
        with self.assertRaises(ValueError):
            HeatCoefficientModel(8, 5, 0.1, (0.5,), (6,), u0)
        with self.assertRaises(ValueError):
            HeatCoefficientModel(8, 5, 0.1, (0.5,), (3, 3), u0)
        with self.assertRaises(ValueError):
            HeatCoefficientModel(8, 5, -0.1, (0.5,), (5,), u0)
        with self.assertRaises(TypeError):
            HeatCoefficientModel(8, 5, 0.1, (0.5,), (5,), 3.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            HeatCoefficientModel(8, 5, 0.1, (0.5,), (5,), lambda s: math.nan)
        with self.assertRaises(ValueError):
            heat_coefficient_benchmark(noise_sd=0.0)
        with self.assertRaises(ValueError):
            heat_coefficient_benchmark(space_refinement=0)


class OscillatorBenchmarkTests(unittest.TestCase):
    """N03: structure, oracles for the posterior and singular values, coverage."""

    @classmethod
    def setUpClass(cls):
        """Build the default N03 case once."""
        cls.bench = damped_oscillator_benchmark()
        jac = cls.bench.jacobian
        cls.sigma = cls.bench.noise_level
        cls.tau = 1.0
        cls.jac = jac

    def test_structure_matches_the_catalog_contract(self):
        """Structure matches the catalog contract."""
        b = self.bench
        self.assertIsInstance(b, NonlinearBenchmark)
        self.assertEqual(b.identifier, "N03")
        self.assertFalse(b.same_discretization)
        self.assertEqual(len(b.truth), b.model.input_dimension)
        self.assertEqual(len(b.data), 12)
        self.assertEqual(b.model.input_dimension, 6)
        self.assertEqual(len(b.whitened_singular_values), 6)
        self.assertIn("N03/truth", b.seeds.values())
        self.assertNotEqual(b.seeds["truth"], b.seeds["noise"])
        self.assertAlmostEqual(
            b.realized_noise_norm,
            math.hypot(*(a - c for a, c in zip(b.data, b.exact_data, strict=True))),
            delta=1e-12,
        )

    def test_the_model_is_linear_and_the_jacobian_is_exact(self):
        """The model is linear and the jacobian is exact."""
        rng = random.Random(2)
        for _ in range(3):
            z = _random_vector(rng, 6)
            direct = self.bench.model.value(z)
            via_j = [
                sum(r * c for r, c in zip(row, z, strict=True)) for row in self.jac
            ]
            for a, c in zip(direct, via_j, strict=True):
                self.assertAlmostEqual(a, c, delta=1e-12)

    def test_truth_is_independent_of_the_noise_draw_and_level(self):
        """Truth is independent of the noise draw and level."""
        a = damped_oscillator_benchmark(seed=11, noise_sd=0.05)
        b = damped_oscillator_benchmark(seed=11, noise_sd=0.2)
        self.assertEqual(a.truth, b.truth)
        self.assertEqual(a.exact_data, b.exact_data)
        scaled_a = [(x - e) / 0.05 for x, e in zip(a.data, a.exact_data, strict=True)]
        scaled_b = [(x - e) / 0.2 for x, e in zip(b.data, b.exact_data, strict=True)]
        for p, q in zip(scaled_a, scaled_b, strict=True):
            self.assertAlmostEqual(p, q, delta=1e-12)
        other = damped_oscillator_benchmark(seed=12)
        self.assertNotEqual(a.truth, other.truth)
        stream = resolve_stream(11).child("N03").child("truth").generator()
        self.assertAlmostEqual(a.truth[0], stream.gauss(0.0, 1.0), delta=1e-15)

    def test_truth_comes_from_a_finer_time_step(self):
        """Truth comes from a finer time step."""
        coarse = damped_oscillator_benchmark(steps_per_interval=2, data_refinement=1)
        fine = damped_oscillator_benchmark(steps_per_interval=2, data_refinement=8)
        self.assertTrue(coarse.same_discretization)
        self.assertEqual(coarse.model.value(coarse.truth), coarse.exact_data)
        self.assertEqual(coarse.truth, fine.truth)
        gap = math.hypot(
            *(a - c for a, c in zip(fine.exact_data, coarse.exact_data, strict=True))
        )
        self.assertGreater(gap, 1e-4)  # the coarse scheme is visibly different
        # the discretization error shrinks with the model step (RK4: order 4)
        errors = []
        for steps in (2, 4):
            case = damped_oscillator_benchmark(
                steps_per_interval=steps, data_refinement=1
            )
            reference = damped_oscillator_benchmark(
                steps_per_interval=steps, data_refinement=64 // steps
            )
            errors.append(
                math.hypot(
                    *(
                        a - c
                        for a, c in zip(
                            case.exact_data, reference.exact_data, strict=True
                        )
                    )
                )
            )
        order = math.log2(errors[0] / errors[1])
        self.assertGreater(order, 3.0)

    def test_whitened_singular_values_match_an_independent_eigen_oracle(self):
        """Whitened singular values match an independent eigen oracle."""
        sigma = self.sigma
        n = 6
        gram = tuple(
            tuple(sum(row[i] * row[j] for row in self.jac) / sigma**2 for j in range(n))
            for i in range(n)
        )
        eig = symmetric_eigen(gram)[0]
        oracle = sorted((math.sqrt(max(e, 0.0)) for e in eig), reverse=True)
        got = self.bench.whitened_singular_values
        for a, b in zip(got, oracle, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-8 * oracle[0])
        self.assertEqual(list(got), sorted(got, reverse=True))
        # the stated pathology: a spread of more than an order of magnitude
        self.assertGreater(got[0] / got[-1], 10.0)
        independent = singular_value_decomposition(self.jac).singular_values
        for a, b in zip(got, independent, strict=True):
            self.assertAlmostEqual(a * sigma, b, delta=1e-10)

    def test_posterior_equals_the_normal_equations(self):
        """Posterior equals the normal equations."""
        sigma, tau = self.sigma, self.tau
        n = 6
        precision = tuple(
            tuple(
                sum(row[i] * row[j] for row in self.jac) / sigma**2
                + (1.0 / tau**2 if i == j else 0.0)
                for j in range(n)
            )
            for i in range(n)
        )
        rhs = tuple(
            sum(row[i] * y for row, y in zip(self.jac, self.bench.data, strict=True))
            / sigma**2
            for i in range(n)
        )
        mean = solve_spd(precision, rhs)
        covariance = inverse_spd(precision)
        for a, b in zip(self.bench.posterior.mean, mean, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    self.bench.initial_state_covariance[i][j],
                    covariance[i][j],
                    delta=1e-10,
                )
        for a, b in zip(self.bench.initial_state_posterior_mean, mean[:2], strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9)
        for i in range(2):
            self.assertAlmostEqual(
                self.bench.initial_state_posterior_sd[i],
                math.sqrt(covariance[i][i]),
                delta=1e-10,
            )
            self.assertLess(
                self.bench.initial_state_posterior_sd[i],
                self.bench.initial_state_prior_sd[i],
            )

    def test_gauss_newton_with_the_prior_penalty_finds_the_posterior_mean(self):
        """Gauss newton with the prior penalty finds the posterior mean."""
        b = self.bench
        result = gauss_newton_krylov(
            b.model,
            b.data,
            noise=b.noise,
            regularization=1.0 / self.tau,
            initial=(0.0,) * 6,
            max_iterations=10,
            inner_tolerance=1e-13,
            gradient_tolerance=1e-12,
        )
        self.assertTrue(result.converged, result.stop_reason)
        for a, c in zip(result.estimate, b.posterior.mean, strict=True):
            self.assertAlmostEqual(a, c, delta=1e-6)

    def test_adjoint_tangent_and_checkpoints_on_the_benchmark_model(self):
        """Adjoint tangent and checkpoints on the benchmark model."""
        model = self.bench.model
        rng = random.Random(4)
        v = _random_vector(rng, 6)
        w = _random_vector(rng, 12)
        self.assertLess(_relative_dot_error(model, self.bench.truth, v, w), 1e-12)

    def _coverage(self, replicates):
        hits = [0, 0]
        for seed in range(replicates):
            case = damped_oscillator_benchmark(seed=1000 + seed)
            for i in range(2):
                z = (case.truth[i] - case.initial_state_posterior_mean[i]) / (
                    case.initial_state_posterior_sd[i]
                )
                hits[i] += abs(z) <= 1.96
        return [h / replicates for h in hits]

    def test_initial_state_credible_intervals_are_calibrated_fast(self):
        """Initial state credible intervals are calibrated fast."""
        replicates = 60
        band = 4.0 * math.sqrt(0.95 * 0.05 / replicates)
        for freq in self._coverage(replicates):
            self.assertLess(abs(freq - 0.95), band)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_initial_state_credible_intervals_are_calibrated_slow(self):
        """Initial state credible intervals are calibrated slow."""
        replicates = 1000
        band = 4.0 * math.sqrt(0.95 * 0.05 / replicates)
        for freq in self._coverage(replicates):
            self.assertLess(abs(freq - 0.95), band)

    def test_argument_errors(self):
        """Argument errors."""
        for kwargs in (
            {"noise_sd": 0.0},
            {"noise_sd": -1.0},
            {"prior_sd": 0.0},
            {"horizon": -1.0},
            {"data_refinement": 0},
            {"steps_per_interval": 0},
            {"pieces": 0},
            {"zeta": -0.1},
            {"omega": 0.0},
            {"intervals_per_piece": 0},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                damped_oscillator_benchmark(**kwargs)
        with self.assertRaises(TypeError):
            damped_oscillator_benchmark(pieces=2.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            damped_oscillator_benchmark(noise_sd="0.1")  # type: ignore[arg-type]

    def test_covariance_type_is_dense(self):
        """Covariance type is dense."""
        self.assertIsInstance(self.bench.prior_covariance, DenseCovariance)


if __name__ == "__main__":
    unittest.main()
