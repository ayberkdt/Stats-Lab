"""Tests for Steihaug CG, trust-region Newton-CG and the Newton inversion.

Oracles: the dense Cholesky solve ``probstat.linalg.solve_spd`` for the
Steihaug interior step, the stacked-QR Tikhonov solution for the linear
inverse problem, BFGS on the Rosenbrock function, central differences of an
analytic gradient for the residual-weighted Hessian term, and evaluation
counts of a 200-parameter matrix-free model.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.inverse.gauss_newton import CallbackForward
from probstat.inverse.newton_trust_region import (
    NewtonTrustRegionResult,
    newton_trust_region_inversion,
)
from probstat.inverse.regularization import tikhonov
from probstat.linalg import ScaledIdentityCovariance, dot, matvec, norm, solve_spd
from probstat.optimization import minimize_bfgs
from probstat.optimization.trust_region import (
    steihaug_cg,
    trust_region_newton_cg,
)

H3 = ((4.0, 1.0, 0.0), (1.0, 3.0, 0.5), (0.0, 0.5, 2.0))
G3 = (1.0, -2.0, 0.5)


def hv(matrix, v):
    """Return the dense Hessian action."""
    return matvec(matrix, v)


def rosen(x):
    """Rosenbrock function."""
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def rosen_grad(x):
    """Rosenbrock gradient."""
    return (
        -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
        200.0 * (x[1] - x[0] ** 2),
    )


def rosen_hess_action(x, v):
    """Rosenbrock Hessian action (indefinite in part of the plane)."""
    h11 = 1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0
    h12 = -400.0 * x[0]
    return (h11 * v[0] + h12 * v[1], h12 * v[0] + 200.0 * v[1])


class SteihaugTests(unittest.TestCase):
    """Truncated CG on closed-form quadratics."""

    def test_interior_equals_newton_step(self) -> None:
        """Inside the region the step is ``-H^-1 g`` (Cholesky oracle)."""
        result = steihaug_cg(G3, lambda v: hv(H3, v), 100.0, tolerance=1e-13)
        newton = tuple(-a for a in solve_spd(H3, G3))
        self.assertEqual(result.reason, "interior")
        self.assertLessEqual(result.hessian_actions, 3)
        for a, b in zip(result.step, newton, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-12)
        self.assertAlmostEqual(
            result.predicted_reduction, 0.5 * dot(G3, solve_spd(H3, G3)), delta=1e-12
        )

    def test_boundary_step_has_radius_norm(self) -> None:
        """A small region cuts the CG path at ``||p|| = Delta``."""
        radius = 0.2
        result = steihaug_cg(G3, lambda v: hv(H3, v), radius, tolerance=1e-13)
        self.assertEqual(result.reason, "boundary")
        self.assertAlmostEqual(norm(result.step), radius, delta=1e-12)
        self.assertGreater(result.predicted_reduction, 0.0)
        self.assertLess(
            result.predicted_reduction, 0.5 * dot(G3, solve_spd(H3, G3)) + 1e-12
        )

    def test_negative_curvature_first_direction(self) -> None:
        """A concave quadratic moves to the boundary along steepest descent."""
        h = ((-1.0, 0.0), (0.0, -2.0))
        result = steihaug_cg((1.0, 1.0), lambda v: hv(h, v), 0.5, tolerance=1e-12)
        self.assertEqual(result.reason, "negative_curvature")
        self.assertAlmostEqual(norm(result.step), 0.5, delta=1e-12)
        model = dot((1.0, 1.0), result.step) + 0.5 * dot(
            result.step, hv(h, result.step)
        )
        self.assertLess(model, 0.0)
        self.assertAlmostEqual(result.predicted_reduction, -model, delta=1e-12)

    def test_negative_curvature_after_positive_steps(self) -> None:
        """Indefinite ``H``: curvature turns negative on the second direction."""
        h = ((4.0, 0.0), (0.0, -1.0))
        g = (1.0, 0.5)
        result = steihaug_cg(g, lambda v: hv(h, v), 1.0, tolerance=1e-12)
        self.assertEqual(result.reason, "negative_curvature")
        self.assertEqual(result.iterations, 2)
        self.assertAlmostEqual(norm(result.step), 1.0, delta=1e-12)
        first = steihaug_cg(
            g, lambda v: hv(h, v), 1.0, tolerance=1e-12, max_iterations=1
        )
        self.assertEqual(first.reason, "max_iterations")
        self.assertGreater(result.predicted_reduction, first.predicted_reduction)

    def test_max_iterations_and_zero_gradient(self) -> None:
        """The cap returns an interior iterate; a zero gradient needs no action."""
        result = steihaug_cg(
            G3, lambda v: hv(H3, v), 100.0, tolerance=0.0, max_iterations=1
        )
        self.assertEqual(result.reason, "max_iterations")
        self.assertEqual(result.hessian_actions, 1)
        self.assertLess(norm(result.step), 100.0)
        zero = steihaug_cg((0.0, 0.0), lambda v: v, 1.0, tolerance=1e-9)
        self.assertEqual(zero.reason, "interior")
        self.assertEqual(zero.hessian_actions, 0)
        self.assertEqual(zero.step, (0.0, 0.0))

    def test_nonfinite_action_and_validation(self) -> None:
        """A non-finite action is reported; bad arguments are rejected."""
        bad = steihaug_cg(G3, lambda v: (math.nan,) * 3, 1.0, tolerance=1e-9)
        self.assertEqual(bad.reason, "nonfinite")
        with self.assertRaises(ValueError):
            steihaug_cg(G3, lambda v: v, 0.0, tolerance=1e-9)
        with self.assertRaises(ValueError):
            steihaug_cg(G3, lambda v: v, 1.0, tolerance=-1.0)
        with self.assertRaises(ValueError):
            steihaug_cg(G3, lambda v: (1.0,), 1.0, tolerance=1e-9)
        with self.assertRaises(TypeError):
            steihaug_cg("abc", lambda v: v, 1.0, tolerance=1e-9)  # type: ignore[arg-type]


class TrustRegionNewtonTests(unittest.TestCase):
    """The globalized driver on Rosenbrock and constructed failures."""

    def run_rosenbrock(self, **options):
        """Run the driver from the standard start."""
        return trust_region_newton_cg(
            rosen, rosen_grad, rosen_hess_action, (-1.2, 1.0), **options
        )

    def test_rosenbrock_converges_and_compares_with_bfgs(self) -> None:
        """Converges to (1, 1); the BFGS counts are only reported."""
        result = self.run_rosenbrock(gradient_tolerance=1e-10)
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, "gradient")
        for a in result.x:
            self.assertAlmostEqual(a, 1.0, delta=1e-7)
        bfgs = minimize_bfgs(
            rosen, (-1.2, 1.0), gradient=rosen_grad, gradient_tolerance=1e-8
        )
        self.assertTrue(bfgs.converged)
        for a in bfgs.x:
            self.assertAlmostEqual(a, 1.0, delta=1e-6)
        self.assertEqual(len(result.history), result.attempts + 1)
        self.assertEqual(result.hessian_actions, result.inner_iterations)
        print(
            f"\n[rosenbrock] TR-NCG: {result.iterations} accepted / "
            f"{result.attempts} attempts, {result.hessian_actions} Hv; "
            f"BFGS: {bfgs.iterations} iterations, {bfgs.evaluations} f-evals"
        )

    def test_radius_updates_follow_rho(self) -> None:
        """Shrink for rho < 1/4, expand for rho > 3/4 on the boundary."""
        small = self.run_rosenbrock(radius=1e-3, max_iterations=3)
        first = small.history[0]
        self.assertGreater(first.rho, 0.75)
        self.assertAlmostEqual(first.step_norm, first.radius, delta=1e-12)
        self.assertAlmostEqual(small.history[1].radius, 2.0 * first.radius, delta=1e-15)
        big = self.run_rosenbrock(radius=100.0, max_radius=100.0, max_iterations=6)
        poor = [
            (a, b)
            for a, b in zip(big.history, big.history[1:], strict=False)
            if a.rho < 0.25
        ]
        self.assertTrue(poor, "no poor-ratio attempt in the history")
        for a, b in poor:
            self.assertAlmostEqual(b.radius, 0.25 * a.step_norm, delta=1e-12)
        for rec in big.history[:-1]:
            if not math.isnan(rec.rho):
                self.assertEqual(rec.accepted, rec.rho > 0.1)

    def test_nonfinite_trial_is_rejected_not_clipped(self) -> None:
        """A non-finite trial objective shrinks the radius; the run reports it."""

        def f(x):
            return 0.5 * x[0] ** 2 if x[0] >= 0.5 else math.nan

        result = trust_region_newton_cg(
            f,
            lambda x: (x[0],),
            lambda x, v: v,
            (3.0,),
            radius=5.0,
            gradient_tolerance=1e-12,
            max_iterations=400,
        )
        first = result.history[0]
        self.assertFalse(first.accepted)
        self.assertTrue(math.isnan(first.trial_objective))
        self.assertAlmostEqual(
            result.history[1].radius, 0.25 * first.step_norm, delta=1e-12
        )
        self.assertGreaterEqual(result.x[0], 0.5)
        self.assertFalse(result.converged)
        self.assertIn(result.stop_reason, ("radius_collapse", "max_iterations"))
        self.assertAlmostEqual(result.x[0], 0.5, delta=1e-6)

    def test_max_iterations_is_not_converged(self) -> None:
        """An exhausted budget is a failure status, never converged."""
        result = self.run_rosenbrock(max_iterations=3)
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertFalse(result.converged)
        self.assertEqual(result.attempts, 3)

    def test_nonfinite_gradient_and_invalid_options(self) -> None:
        """A non-finite gradient at an accepted point stops with 'nonfinite'."""

        def grad(x):
            return (x[0],) if x[0] > 2.5 else (math.nan,)

        result = trust_region_newton_cg(
            lambda x: 0.5 * x[0] ** 2, grad, lambda x, v: v, (3.0,), radius=1.0
        )
        self.assertEqual(result.stop_reason, "nonfinite")
        self.assertFalse(result.converged)
        with self.assertRaises(ValueError):
            trust_region_newton_cg(
                lambda x: math.nan, lambda x: (0.0,), lambda x, v: v, (0.0,)
            )
        with self.assertRaises(ValueError):
            self.run_rosenbrock(eta=0.3)
        with self.assertRaises(ValueError):
            self.run_rosenbrock(radius=10.0, max_radius=1.0)
        with self.assertRaises(ValueError):
            self.run_rosenbrock(forcing="newton")


# --- inverse problems -------------------------------------------------------

A_LIN = (
    (1.0, 0.5, 0.0),
    (0.2, 1.0, 0.3),
    (0.0, 0.4, 1.0),
    (0.7, 0.0, 0.5),
    (0.1, 0.2, 0.9),
)
TIMES = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4)


def linear_model() -> CallbackForward:
    """Linear forward map ``G(x) = A x``."""
    at = tuple(zip(*A_LIN, strict=True))
    return CallbackForward(
        3,
        5,
        lambda x: matvec(A_LIN, x),
        lambda x, v: matvec(A_LIN, v),
        lambda x, w: matvec(at, w),
    )


def decay_model() -> CallbackForward:
    """Exponential decay ``G_i(a, b) = a exp(-b t_i)`` with analytic derivatives."""

    def value(x):
        return tuple(x[0] * math.exp(-x[1] * t) for t in TIMES)

    def jvp(x, v):
        return tuple(math.exp(-x[1] * t) * (v[0] - x[0] * t * v[1]) for t in TIMES)

    def vjp(x, w):
        pairs = list(zip(w, TIMES, strict=True))
        return (
            sum(wi * math.exp(-x[1] * t) for wi, t in pairs),
            -x[0] * sum(wi * t * math.exp(-x[1] * t) for wi, t in pairs),
        )

    return CallbackForward(2, len(TIMES), value, jvp, vjp)


def decay_hvp(x, v, w):
    """Residual-weighted second derivative ``sum_i w_i Hess G_i v``."""
    out0 = 0.0
    out1 = 0.0
    for wi, t in zip(w, TIMES, strict=True):
        e = math.exp(-x[1] * t)
        out0 += wi * (-t * e) * v[1]
        out1 += wi * ((-t * e) * v[0] + x[0] * t * t * e * v[1])
    return (out0, out1)


def decay_data(level):
    """Return data from a fixed truth plus a fixed perturbation of size ``level``."""
    truth = (2.0, 0.7)
    clean = decay_model().value(truth)
    offsets = (0.3, -0.2, 0.25, -0.3, 0.2, -0.25, 0.15)
    return truth, tuple(c + level * o for c, o in zip(clean, offsets, strict=True))


class NewtonInversionTests(unittest.TestCase):
    """The matrix-free inversion built on the trust-region driver."""

    def test_linear_model_matches_tikhonov_qr(self) -> None:
        """Linear ``G``: the minimizer is the stacked-QR Tikhonov solution."""
        y = (1.0, -0.5, 0.8, 0.3, -0.2)
        lam = 0.3
        oracle = tikhonov(A_LIN, y, lam).solution
        result = newton_trust_region_inversion(
            linear_model(), y, regularization=lam, gradient_tolerance=1e-12
        )
        self.assertTrue(result.converged)
        for a, b in zip(result.estimate, oracle, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9)
        self.assertEqual(result.counts.hvp, result.total_inner_iterations)
        self.assertAlmostEqual(
            result.objective, result.misfit + result.penalty, delta=1e-14
        )

    def test_exponential_decay_zero_residual_both_hessians(self) -> None:
        """With a zero residual at the optimum GN and full Newton both converge."""
        truth, y = decay_data(0.0)
        gn = newton_trust_region_inversion(
            decay_model(), y, initial=(1.0, 0.3), gradient_tolerance=1e-12
        )
        full = newton_trust_region_inversion(
            decay_model(),
            y,
            initial=(1.0, 0.3),
            hessian="full",
            hvp_callback=decay_hvp,
            gradient_tolerance=1e-12,
        )
        for result in (gn, full):
            self.assertTrue(result.converged, result.stop_reason)
            for a, b in zip(result.estimate, truth, strict=True):
                self.assertAlmostEqual(a, b, delta=1e-7)
            self.assertLess(result.misfit, 1e-18)
        self.assertEqual(full.hvp_mode, "callback")
        self.assertEqual(gn.hvp_mode, "none")
        print(
            f"\n[decay zero residual] GN: {gn.iterations} steps/"
            f"{gn.attempts} attempts; "
            f"full: {full.iterations} steps/{full.attempts} attempts"
        )

    def test_nonzero_residual_full_hessian_adds_residual_term(self) -> None:
        """At nonzero residual: full action = GN action + residual term (FD check)."""
        _truth, y = decay_data(0.4)
        model = decay_model()
        x = (1.7, 0.5)
        v = (0.6, -0.8)

        def grad(point):
            r = tuple(a - b for a, b in zip(model.value(point), y, strict=True))
            return model.vjp(point, r)

        eps = 1e-6
        plus = grad(tuple(a + eps * b for a, b in zip(x, v, strict=True)))
        minus = grad(tuple(a - eps * b for a, b in zip(x, v, strict=True)))
        fd = tuple((p - q) / (2 * eps) for p, q in zip(plus, minus, strict=True))
        gn = model.vjp(x, model.jvp(x, v))
        r = tuple(a - b for a, b in zip(model.value(x), y, strict=True))
        extra = decay_hvp(x, v, r)
        for f, g, e in zip(fd, gn, extra, strict=True):
            self.assertAlmostEqual(f, g + e, delta=1e-6)
        self.assertGreater(norm(extra), 1e-2)

        results = {}
        variants = (
            ("gn", {}),
            ("callback", {"hessian": "full", "hvp_callback": decay_hvp}),
            (
                "difference",
                {"hessian": "full", "hvp": "central_difference_of_gradient"},
            ),
        )
        for name, options in variants:
            results[name] = newton_trust_region_inversion(
                decay_model(),
                y,
                initial=(1.0, 0.3),
                gradient_tolerance=1e-10,
                **options,
            )
            self.assertTrue(results[name].converged, name)
        for name in ("callback", "difference"):
            for a, b in zip(
                results[name].estimate, results["gn"].estimate, strict=True
            ):
                self.assertAlmostEqual(a, b, delta=1e-6)
        diff = results["difference"].counts
        self.assertGreater(diff.hvp, 0)
        # two gradient evaluations (two vjp calls) per difference action
        self.assertGreaterEqual(diff.vjp, 2 * diff.hvp)
        for name, res in results.items():
            print(
                f"\n[decay nonzero residual] {name}: {res.iterations} steps/"
                f"{res.attempts} attempts, counts {res.counts}"
            )

    def test_full_hessian_requires_explicit_source(self) -> None:
        """No implicit finite-difference fallback; contradictory options fail."""
        _truth, y = decay_data(0.1)
        with self.assertRaises(ValueError):
            newton_trust_region_inversion(decay_model(), y, hessian="full")
        with self.assertRaises(ValueError):
            newton_trust_region_inversion(decay_model(), y, hvp_callback=decay_hvp)
        with self.assertRaises(ValueError):
            newton_trust_region_inversion(
                decay_model(),
                y,
                hessian="full",
                hvp_callback=decay_hvp,
                hvp="central_difference_of_gradient",
            )
        with self.assertRaises(ValueError):
            newton_trust_region_inversion(
                decay_model(),
                y,
                hessian="exact",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            newton_trust_region_inversion(decay_model(), y[:3])

    def test_budget_and_failed_forward(self) -> None:
        """Budget exhaustion and failed trial evaluations are reported."""
        _truth, y = decay_data(0.4)
        short = newton_trust_region_inversion(
            decay_model(), y, initial=(1.0, 0.3), max_iterations=2
        )
        self.assertFalse(short.converged)
        self.assertEqual(short.stop_reason, "max_iterations")

        def value(x):
            return (math.log(x[0]) if x[0] > 0.0 else math.nan,)

        log_model = CallbackForward(
            1, 1, value, lambda x, v: (v[0] / x[0],), lambda x, w: (w[0] / x[0],)
        )
        result = newton_trust_region_inversion(
            log_model,
            (math.log(0.01),),
            initial=(0.1,),
            initial_radius=10.0,
            gradient_tolerance=1e-10,
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.estimate[0], 0.01, delta=1e-8)
        self.assertTrue(any(math.isnan(rec.trial_objective) for rec in result.history))
        with self.assertRaises(ValueError):
            newton_trust_region_inversion(log_model, (0.0,), initial=(-1.0,))

    def test_matrix_free_counts_scale_with_cg_not_n(self) -> None:
        """200 parameters: derivative actions equal CG iterations, far below n."""
        n = 200
        c = 0.1
        gen = random.Random(20240611)
        truth = tuple(gen.uniform(-1.0, 1.0) for _ in range(n))

        def value(x):
            return tuple(x[i] + c * x[i] * x[(i + 1) % n] for i in range(n))

        def jvp(x, v):
            return tuple(
                v[i] + c * (v[i] * x[(i + 1) % n] + x[i] * v[(i + 1) % n])
                for i in range(n)
            )

        def vjp(x, w):
            return tuple(
                w[j] * (1.0 + c * x[(j + 1) % n]) + w[j - 1] * c * x[j - 1]
                for j in range(n)
            )

        model = CallbackForward(n, n, value, jvp, vjp)
        y = value(truth)
        result = newton_trust_region_inversion(
            model, y, regularization=1e-3, gradient_tolerance=1e-8, initial_radius=5.0
        )
        self.assertIsInstance(result, NewtonTrustRegionResult)
        self.assertTrue(result.converged, result.stop_reason)
        self.assertEqual(result.counts.jvp, result.total_inner_iterations)
        self.assertEqual(result.counts.hvp, result.total_inner_iterations)
        # A finite-difference Jacobian would need n = 200 evaluations per step.
        self.assertLess(result.counts.jvp, n)
        self.assertLess(result.counts.value, 40)
        error = norm(tuple(a - b for a, b in zip(result.estimate, truth, strict=True)))
        self.assertLess(error, 0.05)
        print(f"\n[matrix-free n=200] counts {result.counts}, error {error:.2e}")


class DiscrepancyStopTests(unittest.TestCase):
    """The keyword-only discrepancy stop of the Newton inversion."""

    def setUp(self) -> None:
        """Fix the perturbed decay data and the start."""
        self.truth, self.y = decay_data(0.3)
        self.start = (1.0, 0.3)

    def test_default_behaviour_is_unchanged(self) -> None:
        """Without a level the run is identical to one that never heard of it."""
        base = newton_trust_region_inversion(
            decay_model(), self.y, initial=self.start, gradient_tolerance=1e-10
        )
        again = newton_trust_region_inversion(
            decay_model(),
            self.y,
            initial=self.start,
            gradient_tolerance=1e-10,
            discrepancy_level=None,
            discrepancy_tau=3.0,
        )
        self.assertEqual(base.stop_reason, "gradient")
        self.assertEqual(again.estimate, base.estimate)
        # repr comparison: the history holds nan entries for rejected steps, and
        # Python 3.13 dataclass equality compares fields with ``==`` only
        # (``nan != nan``); the repr of a float is exact, so this stays strict.
        self.assertEqual(repr(again.history), repr(base.history))
        self.assertEqual(again.counts, base.counts)
        self.assertEqual(again.stop_reason, "gradient")

    def test_stops_when_misfit_reaches_tau_delta(self) -> None:
        """Stop at the first accepted iterate with ``||r|| <= tau delta``."""
        level, tau = 0.5, 1.2
        result = newton_trust_region_inversion(
            decay_model(),
            self.y,
            initial=self.start,
            gradient_tolerance=1e-10,
            discrepancy_level=level,
            discrepancy_tau=tau,
        )
        full = newton_trust_region_inversion(
            decay_model(), self.y, initial=self.start, gradient_tolerance=1e-10
        )
        self.assertEqual(result.stop_reason, "discrepancy")
        self.assertTrue(result.converged)
        self.assertLessEqual(result.whitened_residual_norm, tau * level)
        self.assertLess(result.iterations, full.iterations)
        # the previous iterate did not satisfy the test: stop is the FIRST hit
        if result.attempts > 1:
            earlier = newton_trust_region_inversion(
                decay_model(),
                self.y,
                initial=self.start,
                max_iterations=result.attempts - 1,
            ).whitened_residual_norm
        else:
            earlier = norm(
                tuple(
                    a - b
                    for a, b in zip(
                        decay_model().value(self.start), self.y, strict=True
                    )
                )
            )
        self.assertGreater(earlier, tau * level)
        # the iterate equals the plain run cut after the same number of attempts
        cut = newton_trust_region_inversion(
            decay_model(), self.y, initial=self.start, max_iterations=result.attempts
        )
        self.assertEqual(cut.estimate, result.estimate)
        # the reported gradient norm is the true one (independent vjp)
        model = decay_model()
        r = tuple(
            a - b for a, b in zip(model.value(result.estimate), self.y, strict=True)
        )
        true_grad = norm(model.vjp(result.estimate, r))
        self.assertAlmostEqual(result.gradient_norm, true_grad, delta=1e-12)
        self.assertGreater(result.gradient_norm, 1e-3)
        self.assertEqual(result.history[-1].gradient_norm, result.gradient_norm)

    def test_start_already_within_tolerance_and_unreachable_level(self) -> None:
        """A satisfied start stops at once; an unreachable level never fires."""
        immediate = newton_trust_region_inversion(
            decay_model(),
            self.y,
            initial=self.start,
            discrepancy_level=10.0,
        )
        self.assertEqual(immediate.stop_reason, "discrepancy")
        self.assertEqual(immediate.iterations, 0)
        self.assertEqual(immediate.estimate, self.start)
        unreachable = newton_trust_region_inversion(
            decay_model(),
            self.y,
            initial=self.start,
            gradient_tolerance=1e-10,
            discrepancy_level=1e-6,
        )
        self.assertEqual(unreachable.stop_reason, "gradient")
        self.assertGreater(unreachable.whitened_residual_norm, 1e-6)

    def test_whitened_norm_and_central_difference_hessian(self) -> None:
        """The test uses ``||W r||``; difference actions do not trigger it."""
        sigma = 0.1
        noise = ScaledIdentityCovariance(len(TIMES), sigma**2)
        result = newton_trust_region_inversion(
            decay_model(),
            self.y,
            noise=noise,
            initial=self.start,
            hessian="full",
            hvp="central_difference_of_gradient",
            discrepancy_level=4.0,
            discrepancy_tau=1.5,
        )
        self.assertEqual(result.stop_reason, "discrepancy")
        self.assertLessEqual(result.whitened_residual_norm, 6.0)
        raw = norm(
            tuple(
                a - b
                for a, b in zip(
                    decay_model().value(result.estimate), self.y, strict=True
                )
            )
        )
        self.assertAlmostEqual(result.whitened_residual_norm, raw / sigma, delta=1e-9)
        self.assertGreater(result.counts.hvp, 0)

    def test_validation(self) -> None:
        """Out-of-range level or tau is rejected."""
        for level in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                newton_trust_region_inversion(
                    decay_model(), self.y, discrepancy_level=level
                )
        with self.assertRaises(TypeError):
            newton_trust_region_inversion(
                decay_model(),
                self.y,
                discrepancy_level="1",  # type: ignore[arg-type]
            )
        for tau in (1.0, 0.5, math.inf):
            with self.assertRaises(ValueError):
                newton_trust_region_inversion(
                    decay_model(), self.y, discrepancy_level=1.0, discrepancy_tau=tau
                )


if __name__ == "__main__":
    unittest.main()
