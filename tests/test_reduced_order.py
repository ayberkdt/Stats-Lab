"""Tests for POD, operator inference, DMD/EDMD and SINDy."""

from __future__ import annotations

import cmath
import math
import os
import random
import unittest
from collections.abc import Sequence

from probstat.foundations.random_streams import resolve_stream
from probstat.linalg import lu_solve, symmetric_eigen
from probstat.numerics import integrate_fixed_step
from probstat.surrogates import reduced_order as ro

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def lorenz(t: float, s: Sequence[float]) -> tuple[float, float, float]:
    """Lorenz-63 right-hand side with the classical parameters."""
    x, y, z = s
    return (SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z)


_LORENZ_CACHE: dict[tuple[int, int, float], tuple[tuple[float, ...], ...]] = {}


def lorenz_states(steps: int, skip: int, step: float) -> tuple[tuple[float, ...], ...]:
    """Return Lorenz states after discarding a transient (cached)."""
    key = (steps, skip, step)
    if key not in _LORENZ_CACHE:
        run = integrate_fixed_step(
            lorenz, 0.0, [-8.0, 7.0, 27.0], step=step, steps=steps
        )
        _LORENZ_CACHE[key] = tuple(run.states[skip:])
    return _LORENZ_CACHE[key]


def orthonormal_columns(n: int, r: int, seed: int) -> list[list[float]]:
    """Return ``r`` orthonormal vectors of length ``n`` by Gram-Schmidt."""
    gen = random.Random(seed)
    out: list[list[float]] = []
    for _ in range(r):
        v = [gen.gauss(0.0, 1.0) for _ in range(n)]
        for u in out:
            d = sum(a * b for a, b in zip(v, u, strict=True))
            v = [a - d * b for a, b in zip(v, u, strict=True)]
        length = math.sqrt(sum(a * a for a in v))
        out.append([a / length for a in v])
    return out


def max_abs_diff(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    """Return the largest absolute entry difference of two matrices."""
    return max(
        abs(a - b)
        for lr, rr in zip(left, right, strict=True)
        for a, b in zip(lr, rr, strict=True)
    )


class PODTests(unittest.TestCase):
    """Proper orthogonal decomposition: closed forms and validation."""

    def setUp(self) -> None:
        """Build a diagonal snapshot matrix with singular values 2, 1, 0.1."""
        self.data = ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.1))

    def test_energy_rule_closed_form(self) -> None:
        """Cumulative energies are 4/5.01, 5/5.01, 1 and set the rank."""
        full = ro.fit_pod(self.data, rank=3)
        self.assertAlmostEqual(full.energy_fractions[0], 4.0 / 5.01, places=14)
        self.assertAlmostEqual(full.energy_fractions[1], 5.0 / 5.01, places=14)
        self.assertAlmostEqual(full.energy_fractions[2], 1.0, places=14)
        self.assertEqual(ro.fit_pod(self.data, energy=0.7).rank, 1)
        self.assertEqual(ro.fit_pod(self.data, energy=0.99).rank, 2)
        self.assertEqual(ro.fit_pod(self.data, energy=0.9999).rank, 3)
        self.assertEqual(ro.fit_pod(self.data, energy=1.0).rank, 3)

    def test_truncation_error_is_the_tail_energy(self) -> None:
        """Eckart-Young: Frobenius error equals the discarded sigma tail."""
        basis = ro.fit_pod(self.data, rank=2)
        self.assertAlmostEqual(basis.truncation_error, 0.1, places=13)
        self.assertAlmostEqual(basis.reconstruction_error(self.data), 0.1, places=13)
        self.assertAlmostEqual(
            basis.reconstruction_error(self.data, relative=True),
            0.1 / math.sqrt(5.01),
            places=13,
        )

    def test_random_data_attains_eckart_young_bound_and_beats_other_bases(self) -> None:
        """On random data the error is the tail and any other subspace is worse."""
        gen = random.Random(4)
        data = tuple(tuple(gen.gauss(0.0, 1.0) for _ in range(6)) for _ in range(12))
        basis = ro.fit_pod(data, rank=3)
        error = basis.reconstruction_error(data)
        self.assertAlmostEqual(error, basis.truncation_error, places=10)
        other = orthonormal_columns(6, 3, 99)
        sq = 0.0
        for row in data:
            coords = [sum(a * b for a, b in zip(row, v, strict=True)) for v in other]
            rec = [
                sum(c * v[i] for c, v in zip(coords, other, strict=True))
                for i in range(6)
            ]
            sq += sum((a - b) ** 2 for a, b in zip(row, rec, strict=True))
        self.assertGreater(math.sqrt(sq), error)

    def test_modes_are_orthonormal_and_projection_idempotent(self) -> None:
        """Modes satisfy V'V = I and project/lift is an idempotent projector."""
        gen = random.Random(2)
        data = tuple(tuple(gen.gauss(0.0, 1.0) for _ in range(5)) for _ in range(9))
        basis = ro.fit_pod(data, rank=3, center=True)
        for i, a in enumerate(basis.modes):
            for j, b in enumerate(basis.modes):
                dot = sum(p * q for p, q in zip(a, b, strict=True))
                self.assertAlmostEqual(dot, 1.0 if i == j else 0.0, places=11)
        once = basis.reconstruct(data[0])
        twice = basis.reconstruct(once)
        for a, b in zip(once, twice, strict=True):
            self.assertAlmostEqual(a, b, places=12)
        mean0 = sum(row[0] for row in data) / len(data)
        self.assertAlmostEqual(basis.mean[0], mean0, places=13)

    def test_validation(self) -> None:
        """Missing, doubled and out-of-range truncation rules are rejected."""
        with self.assertRaises(ValueError):
            ro.fit_pod(self.data)
        with self.assertRaises(ValueError):
            ro.fit_pod(self.data, rank=1, energy=0.9)
        with self.assertRaises(ValueError):
            ro.fit_pod(self.data, energy=0.0)
        with self.assertRaises(ValueError):
            ro.fit_pod(self.data, energy=1.5)
        with self.assertRaises(TypeError):
            ro.fit_pod(self.data, rank=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ro.fit_pod(((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)), rank=2)
        with self.assertRaises(ValueError):
            ro.fit_pod(((1.0, 2.0), (1.0,)), rank=1)
        with self.assertRaises(ValueError):
            ro.fit_pod(((float("nan"), 1.0), (1.0, 2.0)), rank=1)
        with self.assertRaises(ValueError):
            ro.fit_pod(((0.0, 0.0), (0.0, 0.0)), energy=0.5)


class LowRankSystem:
    """A quadratic full-order system whose trajectories live in a 2-D subspace."""

    n = 6
    r = 2

    def __init__(self) -> None:
        """Build V, the full operators and the reduced truth."""
        self.v = orthonormal_columns(self.n, self.r, 1)
        self.a_red = ((-0.2, 1.0), (-1.0, -0.1))
        self.b_red = ((0.5, 0.0), (0.0, -0.4))
        n, r, v = self.n, self.r, self.v
        self.a_full = tuple(
            tuple(
                sum(
                    v[k][i] * self.a_red[k][le] * v[le][j]
                    for k in range(r)
                    for le in range(r)
                )
                for j in range(n)
            )
            for i in range(n)
        )
        self.b_full = tuple(
            tuple(sum(v[k][i] * self.b_red[k][c] for k in range(r)) for c in range(r))
            for i in range(n)
        )

    def red(self, x: Sequence[float]) -> list[float]:
        """Return V'x."""
        return [sum(self.v[k][i] * x[i] for i in range(self.n)) for k in range(self.r)]

    def lift(self, a: Sequence[float]) -> list[float]:
        """Return V a."""
        return [sum(self.v[k][i] * a[k] for k in range(self.r)) for i in range(self.n)]

    def quad(self, x: Sequence[float], y: Sequence[float]) -> tuple[float, ...]:
        """Return the bilinear (non-symmetric) full-order form Q(x, y)."""
        a, b = self.red(x), self.red(y)
        z = [-0.5 * a[0] * b[1], 0.3 * a[0] * b[0] + 0.2 * a[1] * b[1]]
        return tuple(self.lift(z))

    def rhs(
        self, t: float, x: Sequence[float], u: Sequence[float] = (0.0, 0.0)
    ) -> tuple[float, ...]:
        """Return the full-order tendency A x + Q(x, x) + B u."""
        q = self.quad(x, x)
        return tuple(
            sum(self.a_full[i][j] * x[j] for j in range(self.n))
            + q[i]
            + sum(self.b_full[i][c] * u[c] for c in range(self.r))
            for i in range(self.n)
        )

    def data(
        self, initial: Sequence[Sequence[float]], step: float, steps: int
    ) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
        """Return concatenated states and exact derivatives of several runs."""
        states: list[tuple[float, ...]] = []
        ders: list[tuple[float, ...]] = []
        for a0 in initial:
            run = integrate_fixed_step(
                self.rhs, 0.0, self.lift(a0), step=step, steps=steps
            )
            states.extend(run.states)
            ders.extend(self.rhs(0.0, s) for s in run.states)
        return states, ders


class OperatorInferenceTests(unittest.TestCase):
    """Galerkin projection and operator inference."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate noise-free data from the low-rank quadratic system."""
        cls.sys = LowRankSystem()
        cls.states, cls.ders = cls.sys.data(
            [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 0.5)], 0.01, 100
        )
        cls.basis = ro.fit_pod(cls.states, energy=1.0 - 1e-12)

    def test_basis_finds_the_invariant_subspace(self) -> None:
        """The trajectories have exact rank 2 and the basis spans range(V)."""
        self.assertEqual(self.basis.rank, 2)
        self.assertLess(
            self.basis.reconstruction_error(self.states, relative=True), 1e-12
        )

    def test_galerkin_matches_reduced_truth_in_the_pod_basis(self) -> None:
        """V'AV and the compact V'Q equal the known reduced operators (rotated)."""
        model = ro.galerkin_reduced_model(
            self.basis, linear=self.sys.a_full, quadratic=self.sys.quad
        )
        # Compare through the dynamics: the reduced rhs of the lifted state.
        a = (0.7, -0.3)
        x = self.basis.lift(a)
        full = self.sys.rhs(0.0, x)
        reduced = model.rhs(a)
        expected = self.basis.project_derivative(full)
        for p, q in zip(reduced, expected, strict=True):
            self.assertAlmostEqual(p, q, places=12)

    def test_operator_inference_recovers_galerkin_operators(self) -> None:
        """Noise-free lambda = 0 operator inference equals intrusive Galerkin."""
        model = ro.galerkin_reduced_model(
            self.basis, linear=self.sys.a_full, quadratic=self.sys.quad
        )
        fit = ro.fit_operator_inference(
            self.basis, self.states, self.ders, quadratic=True
        )
        assert model.quadratic is not None and fit.model.quadratic is not None
        self.assertLess(max_abs_diff(model.linear, fit.model.linear), 1e-9)
        self.assertLess(max_abs_diff(model.quadratic, fit.model.quadratic), 1e-9)
        self.assertLess(fit.relative_residual, 1e-12)
        self.assertLess(fit.condition_number, 1e3)
        self.assertEqual(fit.parameters, 2 + 3)

    def test_learned_model_reproduces_the_trajectory(self) -> None:
        """Simulating the learned ROM matches the full-order projected run."""
        fit = ro.fit_operator_inference(
            self.basis, self.states, self.ders, quadratic=True
        )
        a0 = self.basis.project(self.states[0])
        run = fit.model.simulate(a0, step=0.01, steps=100)
        self.assertTrue(run.completed)
        self.assertEqual(run.stop_reason, "completed")
        truth = [self.basis.project(s) for s in self.states[:101]]
        self.assertLess(max_abs_diff(run.states, truth), 1e-8)

    def test_linear_input_operator_is_recovered(self) -> None:
        """With a forced run the learned B equals V'B."""
        sys_ = self.sys

        def forcing(t: float) -> tuple[float, float]:
            """Return a two-frequency input signal."""
            return (math.sin(2.0 * t), math.cos(3.0 * t))

        def forced(t: float, x: Sequence[float]) -> tuple[float, ...]:
            """Return the forced tendency."""
            return sys_.rhs(t, x, forcing(t))

        run = integrate_fixed_step(
            forced, 0.0, sys_.lift((0.5, -0.2)), step=0.01, steps=400
        )
        states = list(run.states)
        times = [0.01 * k for k in range(len(states))]
        ders = [forced(t, s) for t, s in zip(times, states, strict=True)]
        us = [forcing(t) for t in times]
        basis = ro.fit_pod(states, energy=1.0 - 1e-12)
        self.assertEqual(basis.rank, 2)
        fit = ro.fit_operator_inference(basis, states, ders, inputs=us, quadratic=True)
        galerkin = ro.galerkin_reduced_model(
            basis, linear=sys_.a_full, quadratic=sys_.quad, input_matrix=sys_.b_full
        )
        assert fit.model.input_matrix is not None and galerkin.input_matrix is not None
        self.assertLess(
            max_abs_diff(fit.model.input_matrix, galerkin.input_matrix), 1e-8
        )
        self.assertLess(max_abs_diff(fit.model.linear, galerkin.linear), 1e-8)

    def test_constant_term_and_centering(self) -> None:
        """A shifted system has constant and linear terms that both routes find."""
        sys_ = self.sys
        n = sys_.n
        shift = tuple(0.5 * (i + 1) for i in range(n))
        shifted = [
            tuple(a + b for a, b in zip(s, shift, strict=True)) for s in self.states
        ]
        # Shifted variable y = x + s has y' = f(y - s) = c + L y + Q(y, y).
        unit = [tuple(float(i == j) for i in range(n)) for j in range(n)]
        q_s_s = sys_.quad(shift, shift)
        const = tuple(
            -sum(sys_.a_full[i][j] * shift[j] for j in range(n)) + q_s_s[i]
            for i in range(n)
        )
        cross = [
            tuple(
                -(p + q)
                for p, q in zip(sys_.quad(shift, e), sys_.quad(e, shift), strict=True)
            )
            for e in unit
        ]
        lin = tuple(
            tuple(sys_.a_full[i][j] + cross[j][i] for j in range(n)) for i in range(n)
        )
        basis = ro.fit_pod(shifted, rank=2, center=True)
        galerkin = ro.galerkin_reduced_model(
            basis, linear=lin, quadratic=sys_.quad, constant=const
        )
        fit = ro.fit_operator_inference(
            basis, shifted, self.ders, quadratic=True, constant=True
        )
        self.assertLess(fit.relative_residual, 1e-10)
        assert galerkin.quadratic is not None and fit.model.quadratic is not None
        self.assertLess(max_abs_diff(galerkin.linear, fit.model.linear), 1e-8)
        self.assertLess(max_abs_diff(galerkin.quadratic, fit.model.quadratic), 1e-8)
        self.assertLess(
            max(
                abs(p - q)
                for p, q in zip(galerkin.constant, fit.model.constant, strict=True)
            ),
            1e-8,
        )
        self.assertGreater(max(abs(c) for c in galerkin.constant), 0.1)

    def test_rank_deficient_data_needs_explicit_regularization(self) -> None:
        """Repeated states make D singular: lambda = 0 raises, lambda > 0 fits."""
        state = self.states[10]
        der = self.ders[10]
        repeated = [state] * 12
        derived = [der] * 12
        with self.assertRaises(ValueError):
            ro.fit_operator_inference(self.basis, repeated, derived, quadratic=True)
        fit = ro.fit_operator_inference(
            self.basis, repeated, derived, quadratic=True, regularization=1e-6
        )
        self.assertTrue(math.isinf(fit.condition_number) or fit.condition_number > 1e6)

    def test_validation(self) -> None:
        """Shape and range errors are reported."""
        with self.assertRaises(ValueError):
            ro.fit_operator_inference(self.basis, self.states, self.ders[:-1])
        with self.assertRaises(ValueError):
            ro.fit_operator_inference(
                self.basis, self.states, self.ders, regularization=-1.0
            )
        with self.assertRaises(ValueError):
            ro.fit_operator_inference(
                self.basis, self.states, self.ders, inputs=[(1.0,)]
            )
        with self.assertRaises(ValueError):
            ro.galerkin_reduced_model(self.basis, linear=((1.0, 2.0), (3.0, 4.0)))


class ReducedModelSimulationTests(unittest.TestCase):
    """Time stepping and divergence reporting."""

    def test_rk4_matches_exponential_decay(self) -> None:
        """A' = -a gives exp(-t) with the RK4 local error h^5/120 per unit."""
        model = ro.ReducedModel((0.0,), ((-1.0,),), None, None)
        run = model.simulate((1.0,), step=0.01, steps=200)
        self.assertTrue(run.completed)
        self.assertAlmostEqual(run.times[-1], 2.0, places=12)
        self.assertAlmostEqual(run.states[-1][0], math.exp(-2.0), places=9)

    def test_finite_time_blowup_is_reported(self) -> None:
        """A' = a^2 from 1 blows up at t = 1; the run stops and says so."""
        model = ro.ReducedModel((0.0,), ((0.0,),), ((1.0,),), None)
        run = model.simulate((1.0,), step=0.01, steps=200, blowup_limit=1e6)
        self.assertFalse(run.completed)
        self.assertIn("diverged", run.stop_reason)
        self.assertLess(run.times[-1], 1.05)
        self.assertTrue(all(math.isfinite(s[0]) for s in run.states))

    def test_input_is_required_and_checked(self) -> None:
        """A model with an input operator refuses to run without an input."""
        model = ro.ReducedModel((0.0,), ((0.0,),), None, ((1.0,),))
        with self.assertRaises(ValueError):
            model.simulate((0.0,), step=0.1, steps=2)
        run = model.simulate(
            (0.0,), step=0.1, steps=10, input_function=lambda t: (1.0,)
        )
        self.assertAlmostEqual(run.states[-1][0], 1.0, places=12)
        with self.assertRaises(ValueError):
            model.rhs((0.0,), (1.0, 2.0))

    def test_quadratic_feature_order(self) -> None:
        """Compact features list a_i a_j for i <= j in row order."""
        self.assertEqual(
            ro.quadratic_features((2.0, 3.0, 5.0)), (4.0, 6.0, 10.0, 9.0, 15.0, 25.0)
        )


class RegularizationSelectionTests(unittest.TestCase):
    """The held-out trajectory rule."""

    @classmethod
    def setUpClass(cls) -> None:
        """Noisy-derivative data of a damped oscillator embedded in R^4."""
        gen = random.Random(8)
        v = orthonormal_columns(4, 2, 5)
        a = ((-0.1, 2.0), (-2.0, -0.1))

        def rhs(t: float, x: Sequence[float]) -> tuple[float, float]:
            """Return the 2-D oscillator tendency."""
            return (
                a[0][0] * x[0] + a[0][1] * x[1],
                a[1][0] * x[0] + a[1][1] * x[1],
            )

        run = integrate_fixed_step(rhs, 0.0, [1.0, 0.0], step=0.02, steps=300)
        cls.states = [
            tuple(sum(v[k][i] * s[k] for k in range(2)) for i in range(4))
            for s in run.states
        ]
        cls.ders = [
            tuple(
                sum(v[k][i] * d for k, d in enumerate(rhs(0.0, s)))
                + gen.gauss(0.0, 0.05)
                for i in range(4)
            )
            for s in run.states
        ]
        cls.basis = ro.fit_pod(cls.states, rank=2)

    def test_selects_a_moderate_weight_and_refits_on_all_data(self) -> None:
        """Huge weights shrink the dynamics to zero and lose on trajectory error."""
        sel = ro.select_operator_inference_regularization(
            self.basis,
            self.states,
            self.ders,
            step=0.02,
            candidates=(0.0, 1e-3, 1e3, 1e8),
        )
        self.assertLess(sel.selected, 1.0)
        self.assertGreater(
            sel.scores[3], sel.scores[sel.candidates.index(sel.selected)]
        )
        self.assertEqual(sel.result.regularization, sel.selected)
        self.assertEqual(sel.result.samples, len(self.states))
        self.assertTrue(all(sel.completed))

    def test_all_diverged_candidates_raise(self) -> None:
        """A divergence limit below the data norm rejects every candidate."""
        with self.assertRaises(ValueError):
            ro.select_operator_inference_regularization(
                self.basis,
                self.states,
                self.ders,
                step=0.02,
                candidates=(0.0, 1.0),
                blowup_limit=1e-6,
            )

    def test_validation(self) -> None:
        """Bad candidates or fractions are rejected."""
        for kwargs in (
            {"candidates": ()},
            {"candidates": (-1.0,)},
            {"candidates": (1.0,), "validation_fraction": 1.0},
        ):
            with self.assertRaises(ValueError):
                ro.select_operator_inference_regularization(
                    self.basis,
                    self.states,
                    self.ders,
                    step=0.02,
                    **kwargs,  # type: ignore[arg-type]
                )


class EigenTests(unittest.TestCase):
    """The non-symmetric eigensolver."""

    def residual(self, matrix: Sequence[Sequence[float]]) -> float:
        """Return the largest ||A v - lambda v||_inf over eigenpairs."""
        values, vectors = ro.eigen_general(matrix)
        worst = 0.0
        for lam, v in zip(values, vectors, strict=True):
            for i, row in enumerate(matrix):
                r = sum(a * b for a, b in zip(row, v, strict=True)) - lam * v[i]
                worst = max(worst, abs(r))
        return worst

    def test_companion_matrix_roots(self) -> None:
        """The companion matrix of (x-1)(x-2)(x-3)(x-4) has eigenvalues 1..4."""
        # x^4 - 10x^3 + 35x^2 - 50x + 24
        comp = ((0, 0, 0, -24.0), (1, 0, 0, 50.0), (0, 1, 0, -35.0), (0, 0, 1, 10.0))
        values, _ = ro.eigen_general(comp)
        got = sorted(v.real for v in values)
        for g, e in zip(got, (1.0, 2.0, 3.0, 4.0), strict=True):
            self.assertAlmostEqual(g, e, places=8)
        self.assertLess(max(abs(v.imag) for v in values), 1e-8)

    def test_rotation_has_complex_pair(self) -> None:
        """R [[cos, -sin], [sin, cos]] has eigenvalues r exp(+-i theta)."""
        theta, r = 0.7, 0.9
        m = (
            (r * math.cos(theta), -r * math.sin(theta)),
            (r * math.sin(theta), r * math.cos(theta)),
        )
        values, _ = ro.eigen_general(m)
        self.assertAlmostEqual(
            abs(values[0] - r * cmath.exp(1j * theta))
            * abs(values[0] - r * cmath.exp(-1j * theta)),
            0.0,
            places=12,
        )
        self.assertLess(self.residual(m), 1e-12)

    def test_random_matrix_residuals(self) -> None:
        """Residuals of a random 8 x 8 matrix are at rounding level."""
        gen = random.Random(3)
        m = [[gen.gauss(0.0, 1.0) for _ in range(8)] for _ in range(8)]
        self.assertLess(self.residual(m), 1e-9)

    def test_agrees_with_symmetric_eigen(self) -> None:
        """For a symmetric matrix the spectrum equals the Jacobi spectrum."""
        s = ((4.0, 1.0, 0.5), (1.0, 3.0, 0.2), (0.5, 0.2, 1.0))
        jacobi, _ = symmetric_eigen(s)
        values, _ = ro.eigen_general(s)
        got = sorted(v.real for v in values)
        for a, b in zip(got, jacobi, strict=True):
            self.assertAlmostEqual(a, b, places=11)

    def test_defective_block_reports_double_eigenvalue(self) -> None:
        """A Jordan block has eigenvalue 1 twice, accurate to about sqrt(eps)."""
        values, _ = ro.eigen_general(((1.0, 1.0), (0.0, 1.0)))
        for v in values:
            self.assertLess(abs(v - 1.0), 1e-6)

    def test_validation(self) -> None:
        """Non-square input is rejected."""
        with self.assertRaises(ValueError):
            ro.eigen_general(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))


class DMDTests(unittest.TestCase):
    """Exact dynamic mode decomposition."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build a non-normal linear map with known eigenvalues."""
        cls.eigs = (0.9, 0.8 + 0.3j, 0.8 - 0.3j, 0.5)
        m = (
            (0.8, 0.3, 0.0, 0.0),
            (-0.3, 0.8, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.0),
            (0.0, 0.0, 0.0, 0.9),
        )
        t = (
            (1.0, 0.0, 0.0, 0.0),
            (0.4, 1.0, 0.0, 0.0),
            (-0.3, 0.2, 1.0, 0.0),
            (0.5, -0.6, 0.7, 1.0),
        )
        cls.t = t
        cls.m = m
        # A = T M T^-1 via solving T' A' = ... column by column.
        tm = tuple(
            tuple(sum(t[i][k] * m[k][j] for k in range(4)) for j in range(4))
            for i in range(4)
        )
        # A T = T M  =>  T' A' = (T M)'  ; solve for rows of A.
        t_t = tuple(zip(*t, strict=True))
        a_rows = []
        for i in range(4):
            rhs = tuple(tm[i][j] for j in range(4))
            a_rows.append(lu_solve(t_t, rhs))
        cls.a = tuple(a_rows)
        y = (1.0, -0.5, 0.8, 0.6)
        cls.states = []
        for _ in range(30):
            cls.states.append(
                tuple(sum(t[i][k] * y[k] for k in range(4)) for i in range(4))
            )
            y = tuple(sum(m[i][k] * y[k] for k in range(4)) for i in range(4))

    def test_linear_eigenvalues_exact(self) -> None:
        """Eigenvalues of a linear system are recovered to 1e-10."""
        fit = ro.fit_dmd(self.states)
        self.assertEqual(fit.rank, 4)
        got = sorted(fit.eigenvalues, key=lambda z: (round(z.real, 6), z.imag))
        want = sorted(self.eigs, key=lambda z: (round(z.real, 6), z.imag))
        for g, w in zip(got, want, strict=True):
            self.assertLess(abs(g - w), 1e-10)

    def test_modes_are_eigenvectors_of_the_generating_matrix(self) -> None:
        """A phi = lambda phi for the exact DMD modes."""
        fit = ro.fit_dmd(self.states)
        for lam, phi in zip(fit.eigenvalues, fit.modes, strict=True):
            a_phi = [
                sum(a * p for a, p in zip(row, phi, strict=True)) for row in self.a
            ]
            self.assertLess(
                max(abs(x - lam * p) for x, p in zip(a_phi, phi, strict=True)), 1e-9
            )

    def test_forecast_and_modal_sum_reproduce_the_data(self) -> None:
        """Forecast equals the trajectory; sum_i b_i lambda_i^k Uw_i matches."""
        fit = ro.fit_dmd(self.states)
        forecast = fit.forecast(self.states[0], 29)
        self.assertLess(max_abs_diff(forecast, self.states), 1e-9)
        for k in (0, 5, 29):
            modal = [
                sum(
                    b * lam**k * pm[i]
                    for b, lam, pm in zip(
                        fit.amplitudes,
                        fit.eigenvalues,
                        fit.projected_modes,
                        strict=True,
                    )
                )
                for i in range(4)
            ]
            for i in range(4):
                self.assertAlmostEqual(modal[i].real, self.states[k][i], places=9)
                self.assertLess(abs(modal[i].imag), 1e-9)

    def test_rank_truncation_by_rank_and_energy(self) -> None:
        """A tiny third mode is discarded; the dominant pair stays accurate."""
        pair = (0.8 + 0.3j, 0.8 - 0.3j)
        y = [1.0, -0.5, 1e-7]
        states = []
        rot = ((0.8, 0.3), (-0.3, 0.8))
        for _ in range(25):
            states.append(tuple(y))
            y = [
                rot[0][0] * y[0] + rot[0][1] * y[1],
                rot[1][0] * y[0] + rot[1][1] * y[1],
                0.5 * y[2],
            ]
        by_rank = ro.fit_dmd(states, rank=2)
        by_energy = ro.fit_dmd(states, energy=1.0 - 1e-10)
        self.assertEqual(by_energy.rank, 2)
        for fit in (by_rank, by_energy):
            for lam in fit.eigenvalues:
                self.assertLess(min(abs(lam - p) for p in pair), 1e-6)
        full = ro.fit_dmd(states)
        self.assertEqual(full.rank, 3)

    def test_noise_biases_but_does_not_destroy_the_truncated_pair(self) -> None:
        """With 1e-3 state noise the rank-2 eigenvalues are within 1e-2."""
        gen = random.Random(12)
        rot = (
            (0.95 * math.cos(0.4), -0.95 * math.sin(0.4)),
            (0.95 * math.sin(0.4), 0.95 * math.cos(0.4)),
        )
        y = [1.0, 0.0]
        states = []
        for _ in range(60):
            states.append(tuple(v + gen.gauss(0.0, 1e-3) for v in y))
            y = [
                rot[0][0] * y[0] + rot[0][1] * y[1],
                rot[1][0] * y[0] + rot[1][1] * y[1],
            ]
        fit = ro.fit_dmd(states, rank=2)
        target = 0.95 * cmath.exp(0.4j)
        self.assertLess(min(abs(lam - target) for lam in fit.eigenvalues), 1e-2)

    def test_continuous_time_eigenvalues(self) -> None:
        """Sampling exp((-0.2 +- 2i) t) at dt = 0.1 recovers -0.2 +- 2i."""
        dt = 0.1
        states = [
            (
                math.exp(-0.2 * k * dt) * math.cos(2.0 * k * dt),
                math.exp(-0.2 * k * dt) * math.sin(2.0 * k * dt),
            )
            for k in range(40)
        ]
        fit = ro.fit_dmd(states, step=dt)
        mus = sorted(fit.continuous_eigenvalues, key=lambda z: z.imag)
        self.assertLess(abs(mus[0] - (-0.2 - 2.0j)), 1e-8)
        self.assertLess(abs(mus[1] - (-0.2 + 2.0j)), 1e-8)
        zero = ro.DMDResult(
            (0j,),
            ((1 + 0j,),),
            ((1 + 0j,),),
            (1 + 0j,),
            (1.0,),
            1,
            ((0.0,),),
            ((1.0,),),
            1.0,
        )
        with self.assertRaises(ValueError):
            _ = zero.continuous_eigenvalues

    def test_validation(self) -> None:
        """Too few snapshots, bad step and doubled rule are rejected."""
        with self.assertRaises(ValueError):
            ro.fit_dmd(self.states[:2])
        with self.assertRaises(ValueError):
            ro.fit_dmd(self.states, step=0.0)
        with self.assertRaises(ValueError):
            ro.fit_dmd(self.states, rank=2, energy=0.9)
        with self.assertRaises(ValueError):
            ro.fit_dmd(self.states, rank=5)
        with self.assertRaises(ValueError):
            ro.fit_dmd(self.states).forecast((1.0, 2.0), 3)


class EDMDTests(unittest.TestCase):
    """Extended DMD on dictionaries."""

    A, B, C = 0.9, 0.5, 0.3

    def step(self, x: Sequence[float]) -> tuple[float, float]:
        """Return the map x1+ = a x1, x2+ = b x2 + c x1^2."""
        return (self.A * x[0], self.B * x[1] + self.C * x[0] ** 2)

    def points(self, count: int, seed: int) -> list[tuple[float, float]]:
        """Return random states in [-1, 1]^2."""
        gen = random.Random(seed)
        return [(gen.uniform(-1, 1), gen.uniform(-1, 1)) for _ in range(count)]

    def test_finite_koopman_invariant_subspace(self) -> None:
        """The dictionary {1, x1, x2, x1^2} is invariant: eigenvalues 1, a, b, a^2."""
        dic = ro.custom_dictionary(
            2,
            [lambda x: 1.0, lambda x: x[0], lambda x: x[1], lambda x: x[0] ** 2],
            ["1", "x1", "x2", "x1^2"],
        )
        xs = self.points(40, 6)
        ys = [self.step(x) for x in xs]
        fit = ro.fit_edmd(xs, ys, dic)
        self.assertLess(fit.relative_residual, 1e-12)
        got = sorted(v.real for v in fit.eigenvalues)
        want = sorted((1.0, self.A, self.B, self.A**2))
        for g, w in zip(got, want, strict=True):
            self.assertAlmostEqual(g, w, places=10)
        self.assertLess(max(abs(v.imag) for v in fit.eigenvalues), 1e-10)
        # Multi-step lifted prediction is exact because the subspace is invariant.
        x0 = (0.7, -0.4)
        forecast = fit.predict(x0, 25)
        truth = [x0]
        for _ in range(25):
            truth.append(self.step(truth[-1]))
        self.assertLess(max_abs_diff(forecast, truth), 1e-10)

    def test_eigenfunction_is_multiplied_by_its_eigenvalue(self) -> None:
        """phi(F(x)) = lambda phi(x) on the invariant subspace."""
        dic = ro.custom_dictionary(
            2,
            [lambda x: 1.0, lambda x: x[0], lambda x: x[1], lambda x: x[0] ** 2],
            ["1", "x1", "x2", "x1^2"],
        )
        xs = self.points(30, 7)
        fit = ro.fit_edmd(xs, [self.step(x) for x in xs], dic)
        x = (0.3, 0.8)
        for index, lam in enumerate(fit.eigenvalues):
            lhs = fit.eigenfunction(index, self.step(x))
            rhs = lam * fit.eigenfunction(index, x)
            self.assertLess(abs(lhs - rhs), 1e-9)

    def test_polynomial_dictionary_on_linear_system(self) -> None:
        """Degree-1 monomials give the eigenvalues of a linear map."""
        mat = ((0.6, 0.2), (-0.1, 0.7))
        xs = self.points(30, 9)
        ys = [
            tuple(sum(m * v for m, v in zip(row, x, strict=True)) for row in mat)
            for x in xs
        ]
        fit = ro.fit_edmd(xs, ys, ro.polynomial_dictionary(2, 1))
        trace, det = 1.3, 0.6 * 0.7 + 0.2 * 0.1
        disc = cmath.sqrt(trace * trace / 4 - det)
        want = sorted(
            (trace / 2 + disc, trace / 2 - disc), key=lambda z: (z.real, z.imag)
        )
        non_constant = sorted(
            (v for v in fit.eigenvalues if abs(v - 1.0) > 1e-6),
            key=lambda z: (z.real, z.imag),
        )
        for g, w in zip(non_constant, want, strict=True):
            self.assertLess(abs(g - w), 1e-10)
        self.assertTrue(any(abs(v - 1.0) < 1e-10 for v in fit.eigenvalues))

    def test_rbf_dictionary_predicts_a_contraction(self) -> None:
        """RBF + linear dictionary: eigenvalues 1 and 0.5 present, forecast exact."""
        centers = ((-1.0,), (0.0,), (1.0,))
        dic = ro.rbf_dictionary(centers, 1.0)
        xs = [(-2.0 + 4.0 * k / 59.0,) for k in range(60)]
        ys = [(0.5 * x[0],) for x in xs]
        fit = ro.fit_edmd(xs, ys, dic)
        self.assertTrue(any(abs(v - 0.5) < 1e-8 for v in fit.eigenvalues))
        self.assertTrue(any(abs(v - 1.0) < 1e-8 for v in fit.eigenvalues))
        forecast = fit.predict((1.5,), 10)
        for k, state in enumerate(forecast):
            self.assertAlmostEqual(state[0], 1.5 * 0.5**k, places=8)
        # The RBF block is not closed under the map, which the residual reports.
        self.assertGreater(fit.relative_residual, 1e-6)

    def test_dependent_dictionary_requires_regularization(self) -> None:
        """A duplicated function makes the regression singular unless ridge > 0."""
        dic = ro.custom_dictionary(
            2, [lambda x: x[0], lambda x: x[0], lambda x: x[1]], ["a", "a2", "b"]
        )
        xs = self.points(20, 3)
        ys = [self.step(x) for x in xs]
        with self.assertRaises(ValueError):
            ro.fit_edmd(xs, ys, dic)
        fit = ro.fit_edmd(xs, ys, dic, regularization=1e-8)
        self.assertEqual(fit.regularization, 1e-8)

    def test_validation(self) -> None:
        """Mismatched shapes and dictionary dimension are rejected."""
        dic = ro.polynomial_dictionary(2, 1)
        xs = self.points(10, 1)
        with self.assertRaises(ValueError):
            ro.fit_edmd(xs, xs[:-1], dic)
        with self.assertRaises(ValueError):
            ro.fit_edmd([(1.0,), (2.0,)], [(1.0,), (2.0,)], dic)
        with self.assertRaises(TypeError):
            ro.fit_edmd(xs, xs, "dictionary")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ro.fit_edmd(xs, xs, dic, regularization=-1.0)
        with self.assertRaises(ValueError):
            ro.polynomial_dictionary(2, 0)
        with self.assertRaises(ValueError):
            ro.rbf_dictionary(((0.0,),), -1.0)
        with self.assertRaises(ValueError):
            dic((1.0,))

    def test_polynomial_dictionary_terms(self) -> None:
        """Two variables, degree two: six monomials with the expected names."""
        dic = ro.polynomial_dictionary(2, 2)
        self.assertEqual(dic.size, 6)
        self.assertEqual(
            sorted(dic.names), sorted(["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"])
        )
        values = dict(zip(dic.names, dic((2.0, 3.0)), strict=True))
        self.assertEqual(values["x0 x1"], 6.0)
        self.assertEqual(values["x0^2"], 4.0)
        self.assertEqual(ro.polynomial_dictionary(3, 2).size, 10)
        self.assertEqual(ro.polynomial_dictionary(3, 2, include_constant=False).size, 9)


class DerivativeTests(unittest.TestCase):
    """Derivative estimators."""

    def test_finite_differences_are_exact_for_quadratics_and_second_order(self) -> None:
        """The map t^2 is exact; sin t converges at rate 2."""
        h = 0.1
        quad = [(k * h) ** 2 for k in range(10)]
        d = ro.finite_difference_derivatives([(v,) for v in quad], h)
        for k, row in enumerate(d):
            self.assertAlmostEqual(row[0], 2.0 * k * h, places=12)
        errors = []
        for h in (0.1, 0.05):
            s = [(math.sin(k * h),) for k in range(41)]
            d = ro.finite_difference_derivatives(s, h)
            errors.append(max(abs(row[0] - math.cos(k * h)) for k, row in enumerate(d)))
        self.assertAlmostEqual(errors[0] / errors[1], 4.0, delta=0.6)

    def test_local_polynomial_is_exact_for_cubics_and_smooths_noise(self) -> None:
        """A cubic is differentiated exactly; noise error shrinks vs raw differences."""
        h = 0.05
        cubic = [((k * h) ** 3 - 2 * (k * h),) for k in range(30)]
        d = ro.local_polynomial_derivatives(cubic, h, window=7, polyorder=3)
        for k, row in enumerate(d):
            self.assertAlmostEqual(row[0], 3 * (k * h) ** 2 - 2.0, places=8)
        gen = random.Random(1)
        noisy = [(math.sin(k * h) + gen.gauss(0.0, 1e-3),) for k in range(200)]
        raw = ro.finite_difference_derivatives(noisy, h)
        smooth = ro.local_polynomial_derivatives(noisy, h, window=15, polyorder=3)
        err_raw = math.sqrt(
            sum((r[0] - math.cos(k * h)) ** 2 for k, r in enumerate(raw)) / 200
        )
        err_smooth = math.sqrt(
            sum((r[0] - math.cos(k * h)) ** 2 for k, r in enumerate(smooth)) / 200
        )
        self.assertLess(err_smooth, 0.5 * err_raw)

    def test_validation(self) -> None:
        """Short data, bad windows and steps are rejected."""
        with self.assertRaises(ValueError):
            ro.finite_difference_derivatives([(1.0,), (2.0,)], 0.1)
        with self.assertRaises(ValueError):
            ro.finite_difference_derivatives([(1.0,)] * 4, -0.1)
        with self.assertRaises(ValueError):
            ro.local_polynomial_derivatives([(1.0,)] * 5, 0.1, window=3, polyorder=3)
        with self.assertRaises(ValueError):
            ro.local_polynomial_derivatives([(1.0,)] * 5, 0.1, window=9)


TRUE_LORENZ = {
    (0, "x0"): -SIGMA,
    (0, "x1"): SIGMA,
    (1, "x0"): RHO,
    (1, "x1"): -1.0,
    (1, "x0 x2"): -1.0,
    (2, "x2"): -BETA,
    (2, "x0 x1"): 1.0,
}


def true_support(dic: ro.FeatureDictionary) -> tuple[tuple[int, ...], ...]:
    """Return the Lorenz support as sorted library indices per equation."""
    out: list[list[int]] = [[], [], []]
    for (eq, name), _ in TRUE_LORENZ.items():
        out[eq].append(dic.names.index(name))
    return tuple(tuple(sorted(row)) for row in out)


class SINDyTests(unittest.TestCase):
    """STLSQ on Lorenz-63."""

    dt = 0.002

    @classmethod
    def setUpClass(cls) -> None:
        """Simulate clean Lorenz data with the library integrator."""
        cls.states = lorenz_states(1500, 200, cls.dt)
        cls.dic = ro.polynomial_dictionary(3, 2)

    def test_exact_derivatives_recover_coefficients_to_rounding(self) -> None:
        """With exact x' the true sparse coefficients come back to 1e-8."""
        ders = [lorenz(0.0, s) for s in self.states]
        fit = ro.fit_sindy(self.states, ders, self.dic, threshold=0.5)
        self.assertTrue(all(fit.converged))
        self.assertEqual(
            tuple(tuple(sorted(row)) for row in fit.support()), true_support(self.dic)
        )
        for (eq, name), value in TRUE_LORENZ.items():
            self.assertAlmostEqual(
                fit.coefficients[eq][self.dic.names.index(name)], value, delta=1e-8
            )
        self.assertLess(max(fit.relative_residuals), 1e-9)

    def test_finite_difference_derivatives_recover_to_tolerance(self) -> None:
        """Second-order differences (h = 0.002) give coefficients within 2e-2."""
        ders = ro.finite_difference_derivatives(self.states, self.dt)
        fit = ro.fit_sindy(self.states, ders, self.dic, threshold=0.5)
        self.assertEqual(
            tuple(tuple(sorted(row)) for row in fit.support()), true_support(self.dic)
        )
        for (eq, name), value in TRUE_LORENZ.items():
            got = fit.coefficients[eq][self.dic.names.index(name)]
            self.assertLess(abs(got - value), 2e-2, msg=f"{name}: {got} vs {value}")

    def test_identified_model_simulates_close_to_the_truth(self) -> None:
        """Integrating the identified model for one time unit stays near Lorenz."""
        ders = [lorenz(0.0, s) for s in self.states]
        fit = ro.fit_sindy(self.states, ders, self.dic, threshold=0.5)
        x0 = self.states[0]
        truth = integrate_fixed_step(lorenz, 0.0, x0, step=0.002, steps=500)
        model = integrate_fixed_step(fit.rhs, 0.0, x0, step=0.002, steps=500)
        gap = max(
            abs(a - b)
            for a, b in zip(truth.final_state, model.final_state, strict=True)
        )
        self.assertLess(gap, 1e-6)

    def test_threshold_controls_sparsity_and_equations_render(self) -> None:
        """A threshold above every coefficient removes all terms; below keeps noise."""
        ders = ro.finite_difference_derivatives(self.states, self.dt)
        high = ro.fit_sindy(self.states, ders, self.dic, threshold=100.0)
        self.assertEqual(high.support(), ((), (), ()))
        self.assertEqual(high.equations(), ("x0' = 0", "x1' = 0", "x2' = 0"))
        low = ro.fit_sindy(self.states, ders, self.dic, threshold=0.0)
        self.assertTrue(all(len(row) == self.dic.size for row in low.support()))
        eqs = ro.fit_sindy(self.states, ders, self.dic, threshold=0.5).equations()
        self.assertIn("x0 x2", eqs[1])

    def test_incomplete_library_is_reported_through_the_residual(self) -> None:
        """A linear library cannot represent the quadratic terms: residual is large."""
        ders = ro.finite_difference_derivatives(self.states, self.dt)
        fit = ro.fit_sindy(
            self.states, ders, ro.polynomial_dictionary(3, 1), threshold=0.5
        )
        self.assertGreater(fit.relative_residuals[1], 0.1)
        self.assertGreater(fit.relative_residuals[2], 0.1)

    def test_validation_and_rank_deficiency(self) -> None:
        """Duplicate library terms raise; shape and range errors are rejected."""
        ders = ro.finite_difference_derivatives(self.states, self.dt)
        dup = ro.custom_dictionary(3, [lambda x: x[0], lambda x: x[0]], ["a", "b"])
        with self.assertRaises(ValueError):
            ro.fit_sindy(self.states, ders, dup, threshold=0.1)
        ok = ro.fit_sindy(self.states, ders, dup, threshold=0.1, ridge=1e-6)
        self.assertEqual(len(ok.coefficients[0]), 2)
        with self.assertRaises(ValueError):
            ro.fit_sindy(self.states, ders[:-1], self.dic, threshold=0.1)
        with self.assertRaises(ValueError):
            ro.fit_sindy(self.states, ders, self.dic, threshold=-1.0)
        with self.assertRaises(ValueError):
            ro.fit_sindy(
                self.states, ders, ro.polynomial_dictionary(2, 2), threshold=0.1
            )
        with self.assertRaises(ValueError):
            ro.fit_sindy(self.states, ders, self.dic, threshold=0.1, max_iterations=0)


class EnsembleSINDyTests(unittest.TestCase):
    """Bootstrap SINDy on noisy Lorenz data."""

    dt = 0.002
    noise = 0.05

    @classmethod
    def setUpClass(cls) -> None:
        """Add Gaussian state noise and smooth the derivatives."""
        clean = lorenz_states(1500, 500, cls.dt)[:700]
        gen = resolve_stream(11).generator()
        noisy = [tuple(v + gen.gauss(0.0, cls.noise) for v in s) for s in clean]
        cls.noisy = noisy
        cls.ders = ro.local_polynomial_derivatives(
            noisy, cls.dt, window=21, polyorder=3
        )
        cls.dic = ro.polynomial_dictionary(3, 2)
        cls.ens = ro.fit_ensemble_sindy(
            noisy, cls.ders, cls.dic, threshold=0.5, models=20, seed=3
        )

    def test_true_terms_are_included_and_spurious_terms_are_not(self) -> None:
        """Inclusion probabilities separate the 7 true terms from the 23 others."""
        truth = true_support(self.dic)
        for eq in range(3):
            for i in range(self.dic.size):
                p = self.ens.inclusion_probability[eq][i]
                if i in truth[eq]:
                    self.assertGreaterEqual(
                        p, 0.9, msg=f"eq {eq} term {self.dic.names[i]}"
                    )
                else:
                    self.assertLessEqual(
                        p, 0.2, msg=f"eq {eq} term {self.dic.names[i]}"
                    )
        self.assertEqual(
            tuple(
                tuple(i for i, c in enumerate(row) if c != 0.0)
                for row in self.ens.selected
            ),
            truth,
        )
        self.assertEqual(self.ens.failed, 0)

    def test_selected_coefficients_and_intervals(self) -> None:
        """Selected coefficients are near the truth and intervals are ordered."""
        for (eq, name), value in TRUE_LORENZ.items():
            i = self.dic.names.index(name)
            self.assertLess(abs(self.ens.selected[eq][i] - value), 0.5, msg=name)
            lo, med, hi = (
                self.ens.interval_lower[eq][i],
                self.ens.coefficient_median[eq][i],
                self.ens.interval_upper[eq][i],
            )
            self.assertLessEqual(lo, med)
            self.assertLessEqual(med, hi)
            self.assertLessEqual(lo - 0.5, value)
            self.assertGreaterEqual(hi + 0.5, value)

    def test_same_seed_reproduces_and_other_seed_differs(self) -> None:
        """Results are a deterministic function of the seed."""
        again = ro.fit_ensemble_sindy(
            self.noisy, self.ders, self.dic, threshold=0.5, models=20, seed=3
        )
        self.assertEqual(again.coefficient_mean, self.ens.coefficient_mean)
        other = ro.fit_ensemble_sindy(
            self.noisy, self.ders, self.dic, threshold=0.5, models=20, seed=4
        )
        self.assertNotEqual(other.coefficient_mean, self.ens.coefficient_mean)

    def test_library_bagging_conditions_on_availability(self) -> None:
        """Library bagging keeps true terms above one half, less sharply than data."""
        ens = ro.fit_ensemble_sindy(
            self.noisy,
            self.ders,
            self.dic,
            threshold=0.5,
            models=16,
            mode="library",
            library_fraction=0.8,
            seed=5,
        )
        truth = true_support(self.dic)
        true_p = [ens.inclusion_probability[e][i] for e in range(3) for i in truth[e]]
        # Dropping a true term lets correlated terms (e.g. the constant) stand
        # in for it, so library bagging is less sharp than data bootstrap.
        self.assertGreaterEqual(min(true_p), 0.5)
        self.assertGreaterEqual(sum(true_p) / len(true_p), 0.85)
        both = ro.fit_ensemble_sindy(
            self.noisy,
            self.ders,
            self.dic,
            threshold=0.5,
            models=6,
            mode="both",
            seed=1,
        )
        self.assertEqual(both.mode, "both")

    def test_heavy_noise_degrades_but_reports_it(self) -> None:
        """At noise 0.5 the true terms keep a higher mean probability than the rest."""
        gen = resolve_stream(21).generator()
        clean = lorenz_states(1500, 500, self.dt)[:700]
        noisy = [tuple(v + gen.gauss(0.0, 0.5) for v in s) for s in clean]
        ders = ro.local_polynomial_derivatives(noisy, self.dt, window=41, polyorder=3)
        ens = ro.fit_ensemble_sindy(
            noisy, ders, self.dic, threshold=0.5, models=12, seed=2
        )
        truth = true_support(self.dic)
        true_p = [ens.inclusion_probability[e][i] for e in range(3) for i in truth[e]]
        other_p = [
            ens.inclusion_probability[e][i]
            for e in range(3)
            for i in range(self.dic.size)
            if i not in truth[e]
        ]
        self.assertGreater(sum(true_p) / len(true_p), sum(other_p) / len(other_p))
        self.assertTrue(
            all(0.0 <= p <= 1.0 for row in ens.inclusion_probability for p in row)
        )

    def test_validation(self) -> None:
        """Out-of-range ensemble arguments are rejected."""
        base = (self.noisy, self.ders, self.dic)
        for kwargs in (
            {"models": 1},
            {"mode": "weights"},
            {"library_fraction": 0.0},
            {"inclusion_cutoff": 1.5},
            {"interval_level": 1.0},
        ):
            with self.assertRaises(ValueError):
                ro.fit_ensemble_sindy(*base, threshold=0.5, **kwargs)  # type: ignore[arg-type]
        dup = ro.custom_dictionary(3, [lambda x: x[0], lambda x: x[0]], ["a", "b"])
        with self.assertRaises(ValueError):
            ro.fit_ensemble_sindy(self.noisy, self.ders, dup, threshold=0.1, models=3)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowEnsembleStudy(unittest.TestCase):
    """Larger noisy ensemble with the full-length record."""

    def test_two_hundred_member_ensemble(self) -> None:
        """200 bootstrap members at noise 0.1 keep the true support."""
        dt = 0.002
        clean = lorenz_states(3000, 500, dt)
        gen = resolve_stream(31).generator()
        noisy = [tuple(v + gen.gauss(0.0, 0.1) for v in s) for s in clean]
        ders = ro.local_polynomial_derivatives(noisy, dt, window=31, polyorder=3)
        dic = ro.polynomial_dictionary(3, 2)
        ens = ro.fit_ensemble_sindy(noisy, ders, dic, threshold=0.5, models=200, seed=8)
        truth = true_support(dic)
        for eq in range(3):
            for i in truth[eq]:
                self.assertGreaterEqual(ens.inclusion_probability[eq][i], 0.9)


if __name__ == "__main__":
    unittest.main()
