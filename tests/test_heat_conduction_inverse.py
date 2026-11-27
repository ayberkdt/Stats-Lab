"""Tests of the inverse heat conduction module (pure Python, no NumPy).

Order of evidence: closed forms and exact identities first (semi-infinite
solid, finite-slab series, energy balance, exact adjoint), then convergence
orders against those oracles, then the inverse methods on synthetic data from
a FINER mesh (no inverse crime), then the resolution analysis and the
adversarial cases. The slab is a SYNTHETIC carbon-phenolic-like material;
nothing here is data for a real heat shield. Expensive replicate studies run
only with ``PROBSTAT_SLOW=1``.
"""

from __future__ import annotations

import math
import os
import unittest

from probstat.foundations.random_streams import resolve_stream
from probstat.inverse.heat_conduction_inverse import (
    SlabFluxOperator,
    SlabProperties,
    averaging_kernel,
    bayesian_flux_posterior,
    beck_linear_map,
    beck_noise_gain,
    beck_sequential_estimate,
    cgls_flux_estimate,
    impulse_response_peak_time,
    insulated_slab_flux_response,
    kernel_resolution,
    matern_covariance,
    resolution_at_noise_level,
    select_gp_hyperparameters,
    semi_infinite_flux_response,
    sensitivity_summary,
    step_pulse,
    synthetic_ablator_slab,
    synthetic_ihcp_data,
    tikhonov_flux_estimate,
    triangular_pulse,
)
from probstat.inverse.parameter_selection import logarithmic_grid
from probstat.linalg import (
    adjoint_test,
    cholesky,
    inverse_spd,
    materialize,
    matmul,
    solve_spd,
    transpose,
)

SLAB = synthetic_ablator_slab()
ALPHA = SLAB.diffusivity
NQ = 30
DELTA = 2.0
Q_PEAK = 1.0e5


def triangle(t: float) -> float:
    """Return the triangular test flux (W/m^2)."""
    return triangular_pulse(t, Q_PEAK, 6.0, 24.0, 42.0)


def rectangle(t: float) -> float:
    """Return the step (rectangular) test flux (W/m^2)."""
    return step_pulse(t, Q_PEAK, 13.0, 37.0)


def relative_error(estimate: tuple[float, ...], truth: tuple[float, ...]) -> float:
    """Return ``||estimate - truth|| / ||truth||``."""
    num = math.sqrt(
        math.fsum((a - b) ** 2 for a, b in zip(estimate, truth, strict=True))
    )
    return num / math.sqrt(math.fsum(b * b for b in truth))


def operator(
    depths: tuple[float, ...] = (0.002,), **kwargs: object
) -> SlabFluxOperator:
    """Return the standard inversion model for the given sensor depths."""
    options: dict[str, object] = {
        "flux_intervals": NQ,
        "interval_length": DELTA,
    }
    options.update(kwargs)
    return SlabFluxOperator(SLAB, depths, **options)  # type: ignore[arg-type]


def small_grid(
    op: SlabFluxOperator,
    sigma: float,
    points: int = 14,
    low: float = 3e-3,
    high: float = 0.5,
) -> tuple[float, ...]:
    """Return a coarse lambda grid (fast tests), relative to ||A||_F / sigma."""
    matrix = op.sensitivity_matrix()
    scale = math.sqrt(math.fsum(v * v for row in matrix for v in row)) / sigma
    return logarithmic_grid(low * scale, high * scale, points)


def random_vector(size: int, seed: int) -> tuple[float, ...]:
    """Return a reproducible standard-normal vector."""
    rng = resolve_stream(seed).generator()
    return tuple(rng.gauss(0.0, 1.0) for _ in range(size))


class ClosedFormTests(unittest.TestCase):
    """The analytical oracles satisfy their own defining equations."""

    def test_semi_infinite_solves_the_heat_equation(self) -> None:
        """theta_t = alpha theta_xx and -k theta_x(0) = q by finite differences."""
        k, q = SLAB.conductivity, 1.0e5
        x, t = 0.003, 15.0
        dx, dt = 1.0e-4, 1.0e-3

        def theta(xx: float, tt: float) -> float:
            return semi_infinite_flux_response(k, ALPHA, xx, tt, q)

        theta_t = (theta(x, t + dt) - theta(x, t - dt)) / (2.0 * dt)
        theta_xx = (theta(x + dx, t) - 2.0 * theta(x, t) + theta(x - dx, t)) / dx**2
        self.assertAlmostEqual(theta_t / (ALPHA * theta_xx), 1.0, delta=2e-4)
        h = 1.0e-6
        slope = (-3.0 * theta(0.0, t) + 4.0 * theta(h, t) - theta(2 * h, t)) / (2 * h)
        self.assertAlmostEqual(-k * slope / q, 1.0, delta=1e-5)

    def test_semi_infinite_surface_and_initial_values(self) -> None:
        """theta(0, t) = 2 q sqrt(alpha t / pi) / k and theta(x, 0) = 0."""
        k = SLAB.conductivity
        surface = semi_infinite_flux_response(k, ALPHA, 0.0, 10.0, 1.0)
        self.assertAlmostEqual(surface, 2.0 * math.sqrt(ALPHA * 10.0 / math.pi) / k, 15)
        self.assertEqual(semi_infinite_flux_response(k, ALPHA, 0.002, 0.0), 0.0)
        # The surface rise equals 2 q sqrt(t / pi) / effusivity.
        self.assertAlmostEqual(
            surface, 2.0 * math.sqrt(10.0 / math.pi) / SLAB.effusivity, 15
        )

    def test_series_initial_condition_and_boundary_conditions(self) -> None:
        """The derived series is zero at t = 0 and has the right fluxes."""
        k, big_l = SLAB.conductivity, SLAB.thickness
        for x in (0.0, 0.004, 0.011, big_l):
            self.assertAlmostEqual(
                insulated_slab_flux_response(k, ALPHA, big_l, x, 0.0, terms=4000),
                0.0,
                delta=3e-6,
            )
        t, h = 300.0, 1.0e-7

        def theta(x: float) -> float:
            return insulated_slab_flux_response(k, ALPHA, big_l, x, t)

        front = (-3.0 * theta(0.0) + 4.0 * theta(h) - theta(2 * h)) / (2 * h)
        self.assertAlmostEqual(-k * front, 1.0, delta=1e-6)
        back = (3.0 * theta(big_l) - 4.0 * theta(big_l - h) + theta(big_l - 2 * h)) / (
            2 * h
        )
        self.assertAlmostEqual(back * big_l / theta(big_l), 0.0, delta=1e-6)

    def test_series_conserves_energy(self) -> None:
        """int_0^L theta dx = q t / (rho c): fixes the constants of the series."""
        k, big_l = SLAB.conductivity, SLAB.thickness
        t = 300.0
        count = 400
        h = big_l / count
        values = [
            insulated_slab_flux_response(k, ALPHA, big_l, i * h, t)
            for i in range(count + 1)
        ]
        simpson = (
            h
            / 3.0
            * (
                values[0]
                + values[-1]
                + 4.0 * sum(values[1:-1:2])
                + 2.0 * sum(values[2:-1:2])
            )
        )
        expected = t / (SLAB.density * SLAB.specific_heat)
        self.assertAlmostEqual(simpson / expected, 1.0, delta=1e-7)

    def test_series_matches_semi_infinite_at_early_time(self) -> None:
        """Before the thermal wave reaches the back face the slab is semi-infinite."""
        k, big_l = SLAB.conductivity, SLAB.thickness
        for x in (0.0, 0.002, 0.004):
            for t in (5.0, 20.0):
                slab = insulated_slab_flux_response(k, ALPHA, big_l, x, t)
                half = semi_infinite_flux_response(k, ALPHA, x, t)
                self.assertAlmostEqual(slab / half, 1.0, delta=2e-6)

    def test_peak_time_is_a_stationary_point(self) -> None:
        """d/dt log(t^-1/2 exp(-x^2 / (4 alpha t))) vanishes at x^2 / (2 alpha)."""
        x = 0.003
        tp = impulse_response_peak_time(x, ALPHA)

        def log_response(t: float) -> float:
            return -0.5 * math.log(t) - x * x / (4.0 * ALPHA * t)

        eps = 1.0e-3
        slope = (log_response(tp + eps) - log_response(tp - eps)) / (2 * eps)
        self.assertAlmostEqual(slope, 0.0, delta=1e-9)
        self.assertGreater(log_response(tp), log_response(0.7 * tp))
        self.assertGreater(log_response(tp), log_response(1.4 * tp))

    def test_oracle_validation(self) -> None:
        """Invalid arguments are rejected."""
        with self.assertRaises(ValueError):
            semi_infinite_flux_response(-1.0, ALPHA, 0.0, 1.0)
        with self.assertRaises(ValueError):
            semi_infinite_flux_response(1.0, ALPHA, -0.1, 1.0)
        with self.assertRaises(TypeError):
            semi_infinite_flux_response(1.0, ALPHA, "x", 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            insulated_slab_flux_response(1.0, ALPHA, 0.02, 0.03, 1.0)
        with self.assertRaises(ValueError):
            impulse_response_peak_time(0.0, ALPHA)
        with self.assertRaises(ValueError):
            triangular_pulse(1.0, 1.0, 2.0, 1.0, 3.0)
        with self.assertRaises(ValueError):
            step_pulse(1.0, 1.0, 2.0, 2.0)
        with self.assertRaises(ValueError):
            SlabProperties(1.0, 1000.0, 1000.0, 0.01, -1.0)
        with self.assertRaises(ValueError):
            SlabProperties(0.0, 1000.0, 1000.0, 0.01)


class OperatorTests(unittest.TestCase):
    """Structure of the forward operator and its exact transpose."""

    def test_dot_product_test_all_configurations(self) -> None:
        """y'(A x) = (A'y)'x to rounding, insulated and convective, any sampling."""
        convective = synthetic_ablator_slab(back_coefficient=80.0)
        cases = [
            SlabFluxOperator(
                SLAB, [0.002, 0.007], flux_intervals=NQ, interval_length=DELTA
            ),
            SlabFluxOperator(
                convective,
                [0.0035],
                flux_intervals=NQ,
                interval_length=DELTA,
                cells=25,
                substeps=3,
                sample_every=3,
            ),
            SlabFluxOperator(
                SLAB,
                [0.0006, 0.0194],
                flux_intervals=12,
                interval_length=5.0,
                cells=20,
                substeps=1,
                sample_every=2,
            ),
        ]
        for index, op in enumerate(cases):
            for seed in (1, 2):
                x = random_vector(op.shape[1], 10 * index + seed)
                y = random_vector(op.shape[0], 100 + 10 * index + seed)
                result = adjoint_test(op, x, y, tolerance=1e-13)
                self.assertTrue(result.passed, (index, seed, result.relative_error))

    def test_transpose_rows_equal_materialized_matrix(self) -> None:
        """A' e_i is row i of the materialized matrix, entry by entry."""
        op = operator((0.002, 0.005), substeps=2)
        dense = materialize(op)
        size = op.shape[0]
        for i in (0, 1, 7, size // 2, size - 1):
            unit = tuple(1.0 if j == i else 0.0 for j in range(size))
            column = op.rmatvec(unit)
            scale = max(abs(v) for v in dense[i]) or 1.0
            for j, value in enumerate(column):
                self.assertAlmostEqual(value, dense[i][j], delta=1e-13 * scale)

    def test_toeplitz_matrix_equals_materialization(self) -> None:
        """The cheap lower-triangular Toeplitz matrix equals n_q forward runs."""
        op = operator((0.002, 0.006))
        dense = materialize(op)
        toeplitz = op.sensitivity_matrix()
        scale = max(abs(v) for row in dense for v in row)
        worst = max(
            abs(a - b)
            for row_a, row_b in zip(dense, toeplitz, strict=True)
            for a, b in zip(row_a, row_b, strict=True)
        )
        self.assertLess(worst, 1e-12 * scale)

    def test_causality_positivity_and_linearity(self) -> None:
        """Readings depend only on earlier flux; the response is positive, linear."""
        op = operator((0.002,))
        matrix = op.sensitivity_matrix()
        for m in range(NQ):
            for j in range(NQ):
                if j > m:
                    self.assertEqual(matrix[m][j], 0.0)
                else:
                    self.assertGreater(matrix[m][j], 0.0)
        x = random_vector(NQ, 5)
        y1 = op.matvec(x)
        y2 = op.matvec([2.0 * v for v in x])
        for a, b in zip(y1, y2, strict=True):
            self.assertAlmostEqual(2.0 * a, b, delta=1e-12 * (1.0 + abs(b)))
        step = op.step_response()[0]
        self.assertTrue(all(b > a for a, b in zip(step, step[1:], strict=False)))

    def test_readings_and_times(self) -> None:
        """Reading and flux-interval times follow the sampling convention."""
        op = operator((0.002,), sample_every=3)
        self.assertEqual(op.shape, (10, NQ))
        self.assertEqual(op.reading_times[0], 3 * DELTA)
        self.assertEqual(op.reading_times[-1], NQ * DELTA)
        self.assertAlmostEqual(op.flux_times[0], 0.5 * DELTA)
        self.assertAlmostEqual(op.time_step, DELTA / 4)
        with self.assertRaises(ValueError):
            op.sensitivity_matrix()

    def test_operator_validation(self) -> None:
        """Invalid configurations and inputs are rejected."""
        with self.assertRaises(ValueError):
            operator((0.0001,))  # shallower than half a cell
        with self.assertRaises(ValueError):
            operator((0.0199,))  # deeper than the last cell centre
        with self.assertRaises(ValueError):
            operator(())
        with self.assertRaises(ValueError):
            operator((0.002,), cells=1)
        with self.assertRaises(ValueError):
            operator((0.002,), sample_every=7)
        with self.assertRaises(TypeError):
            SlabFluxOperator("slab", [0.002], flux_intervals=3, interval_length=1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            operator().matvec([1.0] * (NQ - 1))
        with self.assertRaises(ValueError):
            operator().matvec([1.0] * (NQ - 1) + [float("nan")])


class VerificationTests(unittest.TestCase):
    """The discrete model against the exact solutions and its own invariants."""

    def step_error(
        self, cells: int, substeps: int, depth: float = 0.002, first: int = 5
    ) -> float:
        """Return the max error of the discrete step response vs the series."""
        op = SlabFluxOperator(
            SLAB,
            [depth],
            flux_intervals=NQ,
            interval_length=DELTA,
            cells=cells,
            substeps=substeps,
        )
        step = op.step_response()[0]
        return max(
            abs(
                step[m - 1]
                - insulated_slab_flux_response(
                    SLAB.conductivity,
                    ALPHA,
                    SLAB.thickness,
                    depth,
                    DELTA * m,
                    terms=600,
                )
            )
            for m in range(first, NQ + 1)
        )

    def test_step_response_matches_finite_slab_series(self) -> None:
        """A fine discretization reproduces the series (relative error < 1e-3)."""
        err = self.step_error(160, 32)
        peak = insulated_slab_flux_response(
            SLAB.conductivity, ALPHA, SLAB.thickness, 0.002, NQ * DELTA
        )
        self.assertLess(err / peak, 1e-3)

    def test_step_response_matches_semi_infinite_solid_early(self) -> None:
        """Up to 40 s the slab response equals the semi-infinite one."""
        op = SlabFluxOperator(
            SLAB,
            [0.002, 0.004],
            flux_intervals=20,
            interval_length=2.0,
            cells=160,
            substeps=16,
        )
        step = op.step_response()
        for s, depth in enumerate(op.sensors):
            for m in (5, 10, 20):
                exact = semi_infinite_flux_response(
                    SLAB.conductivity, ALPHA, depth, 2.0 * m
                )
                self.assertAlmostEqual(step[s][m - 1] / exact, 1.0, delta=6e-3)

    def test_time_step_convergence_is_first_order(self) -> None:
        """Halving dt halves the error (implicit Euler)."""
        errors = [self.step_error(160, n) for n in (2, 4, 8)]
        for coarse, fine in zip(errors, errors[1:], strict=False):
            self.assertAlmostEqual(coarse / fine, 2.0, delta=0.3)

    def test_mesh_convergence_is_second_order(self) -> None:
        """Halving dx quarters the error once the time error is negligible."""
        errors = [self.step_error(n, 64) for n in (10, 20, 40, 80)]
        for coarse, fine in zip(errors, errors[1:], strict=False):
            self.assertGreater(coarse / fine, 3.5)

    def test_energy_balance_insulated(self) -> None:
        """Absorbed energy equals the integral of q to rounding."""
        op = operator((0.002,))
        flux = [Q_PEAK * (1.0 + 0.5 * math.sin(0.4 * j)) for j in range(NQ)]
        balance = op.energy_balance(flux)
        self.assertLess(balance.relative_residual, 1e-13)
        self.assertEqual(balance.lost_energy, 0.0)
        integral = DELTA * math.fsum(flux)
        self.assertAlmostEqual(balance.input_energy / integral, 1.0, delta=1e-14)
        self.assertAlmostEqual(balance.stored_energy / integral, 1.0, delta=1e-12)

    def test_energy_balance_convective_back_face(self) -> None:
        """With a back film the balance holds and some energy leaves."""
        hot = synthetic_ablator_slab(back_coefficient=50.0)
        op = SlabFluxOperator(
            hot, [0.002], flux_intervals=NQ, interval_length=20.0, cells=40, substeps=4
        )
        balance = op.energy_balance([Q_PEAK] * NQ)
        self.assertLess(balance.relative_residual, 1e-12)
        self.assertGreater(balance.lost_energy, 0.05 * balance.input_energy)
        self.assertAlmostEqual(balance.input_energy, Q_PEAK * 20.0 * NQ, delta=1e-6)

    def test_field_history_matches_sensor_reading(self) -> None:
        """Interpolating the stored field reproduces the operator output."""
        op = operator((0.00325,), cells=40)
        flux = [Q_PEAK * math.sin(0.1 * j) ** 2 for j in range(NQ)]
        history = op.field_history(flux)
        self.assertEqual(len(history), NQ * 4 + 1)
        dx = SLAB.thickness / 40
        position = 0.00325 / dx - 0.5
        i, w = int(position), position - int(position)
        reading = (1.0 - w) * history[-1][i] + w * history[-1][i + 1]
        self.assertAlmostEqual(reading, op.matvec(flux)[-1], delta=1e-12)


class BeckTests(unittest.TestCase):
    """Beck's sequential function specification."""

    def test_one_future_step_inverts_exactly(self) -> None:
        """For model-consistent data and r = 1 the flux is recovered exactly."""
        op = operator((0.003,))
        truth = tuple(Q_PEAK * (1.0 + math.sin(0.35 * j)) for j in range(NQ))
        data = op.matvec(truth)
        result = beck_sequential_estimate(op, data, future_steps=1)
        self.assertLess(relative_error(result.estimate, truth), 1e-8)
        gain = beck_linear_map(op, future_steps=1)
        product = matmul(gain, op.sensitivity_matrix())
        for i, row in enumerate(product):
            for j, value in enumerate(row):
                self.assertAlmostEqual(value, 1.0 if i == j else 0.0, delta=1e-8)

    def test_estimator_is_linear_map_and_reports_sensitivities(self) -> None:
        """G y equals the stage-wise estimate; X are the step-response values."""
        op = operator((0.002, 0.005))
        data = random_vector(op.shape[0], 11)
        result = beck_sequential_estimate(op, data, future_steps=3)
        gain = beck_linear_map(op, future_steps=3)
        mapped = [
            math.fsum(g * y for g, y in zip(row, data, strict=True)) for row in gain
        ]
        scale = max(abs(v) for v in mapped)
        for a, b in zip(mapped, result.estimate, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9 * scale)
        step = op.step_response()
        for s in range(2):
            self.assertEqual(result.sensitivity[s], tuple(step[s][:3]))
        self.assertAlmostEqual(
            result.gain, sum(v * v for row in result.sensitivity for v in row), 15
        )

    def test_bias_variance_tradeoff_in_future_steps(self) -> None:
        """More future steps: less noise amplification, more bias; interior optimum."""
        op = operator((0.002,))
        data = synthetic_ihcp_data(
            SLAB, [0.002], triangle, flux_intervals=NQ, interval_length=DELTA, seed=3
        )
        sigma = data.noise_std
        counts = (1, 2, 3, 4, 6, 8, 12)
        gains = {r: beck_noise_gain(op, future_steps=r) for r in counts}
        for a, b in zip(counts, counts[1:], strict=False):
            self.assertLess(gains[b], gains[a])
        bias = {}
        total = {}
        for r in counts:
            g = beck_linear_map(op, future_steps=r)
            mean = [
                math.fsum(c * y for c, y in zip(row, data.clean, strict=True))
                for row in g
            ]
            bias[r] = math.sqrt(
                math.fsum((m - t) ** 2 for m, t in zip(mean, data.truth, strict=True))
                / NQ
            )
            total[r] = math.hypot(bias[r], sigma * gains[r])
        for a, b in zip(counts[1:], counts[2:], strict=False):
            self.assertGreater(bias[b], bias[a])
        best = min(counts, key=total.__getitem__)
        self.assertIn(best, (2, 3, 4))
        self.assertGreater(total[1], 3.0 * total[best])
        self.assertGreater(total[12], 1.5 * total[best])

    def test_beck_validation(self) -> None:
        """Invalid arguments are rejected."""
        op = operator((0.002,))
        with self.assertRaises(ValueError):
            beck_sequential_estimate(op, [0.0] * NQ, future_steps=0)
        with self.assertRaises(ValueError):
            beck_sequential_estimate(op, [0.0] * NQ, future_steps=NQ + 1)
        with self.assertRaises(ValueError):
            beck_sequential_estimate(op, [0.0] * (NQ - 1), future_steps=2)
        with self.assertRaises(ValueError):
            beck_sequential_estimate(
                operator((0.002,), sample_every=2), [0.0] * 15, future_steps=2
            )
        with self.assertRaises(TypeError):
            beck_sequential_estimate("op", [0.0] * NQ)  # type: ignore[arg-type]


class MethodAccuracyTests(unittest.TestCase):
    """Recovery of a triangular and a step pulse from fine-mesh data, 1% noise."""

    SEED = 3  # predeclared

    def setUp(self) -> None:
        """Build the model and the two synthetic data sets."""
        self.op = operator((0.002,))
        self.data = {
            "triangle": synthetic_ihcp_data(
                SLAB,
                [0.002],
                triangle,
                flux_intervals=NQ,
                interval_length=DELTA,
                seed=self.SEED,
            ),
            "step": synthetic_ihcp_data(
                SLAB,
                [0.002],
                rectangle,
                flux_intervals=NQ,
                interval_length=DELTA,
                seed=self.SEED,
            ),
        }

    def test_data_come_from_a_finer_model(self) -> None:
        """The coarse model does not reproduce the fine data exactly (no crime)."""
        for data in self.data.values():
            coarse = self.op.matvec(data.truth)
            gap = max(abs(a - b) for a, b in zip(coarse, data.clean, strict=True))
            self.assertGreater(gap, 2.0 * data.noise_std * 0.1)  # model error visible
            self.assertAlmostEqual(
                data.noise_std, 0.01 * max(abs(v) for v in data.clean), 12
            )

    def test_beck_recovers_pulses(self) -> None:
        """Beck with r = 3: errors measured 0.087 (triangle) and 0.16 (step)."""
        limits = {"triangle": 0.15, "step": 0.25}
        for name, data in self.data.items():
            result = beck_sequential_estimate(self.op, data.noisy, future_steps=3)
            self.assertLess(
                relative_error(result.estimate, data.truth), limits[name], name
            )

    def test_tikhonov_recovers_pulses(self) -> None:
        """Order-0 / discrepancy and order-1 / GCV; the status is interior."""
        grid = small_grid(self.op, self.data["triangle"].noise_std)
        limits = {"triangle": 0.10, "step": 0.17}
        for name, data in self.data.items():
            for order, criterion in ((0, "discrepancy"), (1, "gcv")):
                result = tikhonov_flux_estimate(
                    self.op,
                    data.noisy,
                    data.noise_std,
                    order=order,
                    criterion=criterion,
                    regularizations=grid,
                )
                self.assertEqual(result.status, "interior", (name, order, criterion))
                assert result.estimate is not None
                error = relative_error(result.estimate, data.truth)
                self.assertLess(error, limits[name], (name, order, criterion))
                # The residual is at the noise level (neither over- nor under-fit).
                assert result.residual_rms is not None
                self.assertLess(result.residual_rms, 1.6 * data.noise_std)
                self.assertGreater(result.residual_rms, 0.3 * data.noise_std)

    def test_bayesian_posterior_recovers_pulses(self) -> None:
        """OU prior with evidence-selected hyperparameters (measured 0.062 / 0.104)."""
        limits = {"triangle": 0.10, "step": 0.16}
        for name, data in self.data.items():
            posterior = select_gp_hyperparameters(
                self.op,
                data.noisy,
                data.noise_std,
                self.op.flux_times,
                amplitudes=[0.3e5, 0.6e5, 1.2e5],
                length_scales=[6.0, 12.0, 24.0],
            )
            self.assertLess(
                relative_error(posterior.mean, data.truth), limits[name], name
            )
            self.assertGreater(posterior.coverage(data.truth), 0.9)

    def test_cgls_with_discrepancy_stopping(self) -> None:
        """CGLS stops early at the noise level (measured 0.049, 0.121)."""
        limits = {"triangle": 0.09, "step": 0.17}
        for name, data in self.data.items():
            result = cgls_flux_estimate(self.op, data.noisy, data.noise_std)
            self.assertEqual(result.status, "discrepancy")
            assert result.estimate is not None and result.iterations is not None
            self.assertLessEqual(result.iterations, 12)
            self.assertLess(
                relative_error(result.estimate, data.truth), limits[name], name
            )
            assert result.residual_rms is not None
            self.assertLessEqual(result.residual_rms, data.noise_std * (1.0 + 1e-9))

    def test_tikhonov_matches_dense_normal_equations(self) -> None:
        """The selected solution solves the regularized normal equations."""
        data = self.data["triangle"]
        grid = small_grid(self.op, data.noise_std)
        result = tikhonov_flux_estimate(
            self.op,
            data.noisy,
            data.noise_std,
            order=0,
            criterion="discrepancy",
            regularizations=grid,
        )
        assert result.estimate is not None and result.regularization is not None
        a = self.op.sensitivity_matrix()
        sigma, lam = data.noise_std, result.regularization
        normal = [
            [
                math.fsum(a[r][i] * a[r][j] for r in range(len(a))) / sigma**2
                + (lam * lam if i == j else 0.0)
                for j in range(NQ)
            ]
            for i in range(NQ)
        ]
        rhs = [
            math.fsum(a[r][i] * data.noisy[r] for r in range(len(a))) / sigma**2
            for i in range(NQ)
        ]
        oracle = solve_spd(tuple(tuple(r) for r in normal), tuple(rhs))
        scale = max(abs(v) for v in oracle)
        for x, o in zip(result.estimate, oracle, strict=True):
            self.assertAlmostEqual(x, o, delta=1e-6 * scale)

    def test_depth_monotonicity_of_errors(self) -> None:
        """Deeper sensors give larger errors (mean over seeds 3, 4, 5)."""
        depths = (0.002, 0.004, 0.006)
        mean_error: dict[str, list[float]] = {"cgls": [], "gp": []}
        for depth in depths:
            op = operator((depth,))
            cg, gp = [], []
            for seed in (3, 4, 5):
                data = synthetic_ihcp_data(
                    SLAB,
                    [depth],
                    triangle,
                    flux_intervals=NQ,
                    interval_length=DELTA,
                    seed=seed,
                )
                result = cgls_flux_estimate(op, data.noisy, data.noise_std)
                assert result.estimate is not None
                cg.append(relative_error(result.estimate, data.truth))
                posterior = bayesian_flux_posterior(
                    op,
                    data.noisy,
                    data.noise_std,
                    op.flux_times,
                    amplitude=6.0e4,
                    length_scale=12.0,
                )
                gp.append(relative_error(posterior.mean, data.truth))
            mean_error["cgls"].append(sum(cg) / 3)
            mean_error["gp"].append(sum(gp) / 3)
        for name, values in mean_error.items():
            self.assertLess(values[0], values[1], (name, values))
            self.assertLess(values[1], values[2], (name, values))


class BayesianTests(unittest.TestCase):
    """The Gaussian-process posterior: exactness and calibration."""

    def test_matern_covariance_formulas(self) -> None:
        """Closed forms at a lag, symmetry, variance on the diagonal."""
        times = [0.0, 3.0, 7.0]
        for nu in (0.5, 1.5, 2.5):
            cov = matern_covariance(
                times, amplitude=2.0, length_scale=5.0, smoothness=nu
            )
            self.assertEqual(cov, tuple(zip(*cov, strict=True)))
            for i in range(3):
                self.assertAlmostEqual(cov[i][i], 4.0, 14)
            cholesky(cov)  # positive definite
        r = 3.0 / 5.0
        ou = matern_covariance(times, amplitude=2.0, length_scale=5.0)
        self.assertAlmostEqual(ou[0][1], 4.0 * math.exp(-r), 14)
        m32 = matern_covariance(times, amplitude=2.0, length_scale=5.0, smoothness=1.5)
        self.assertAlmostEqual(
            m32[0][1], 4.0 * (1.0 + math.sqrt(3) * r) * math.exp(-math.sqrt(3) * r), 14
        )
        m52 = matern_covariance(times, amplitude=2.0, length_scale=5.0, smoothness=2.5)
        self.assertAlmostEqual(
            m52[0][1],
            4.0
            * (1.0 + math.sqrt(5) * r + 5.0 * r * r / 3.0)
            * math.exp(-math.sqrt(5) * r),
            14,
        )
        with self.assertRaises(ValueError):
            matern_covariance(times, amplitude=1.0, length_scale=1.0, smoothness=1.0)
        with self.assertRaises(ValueError):
            matern_covariance(times, amplitude=-1.0, length_scale=1.0)

    def test_posterior_equals_dense_formulas(self) -> None:
        """Mean, standard deviation and band against the textbook formulas."""
        op = operator((0.003,), flux_intervals=12, interval_length=5.0, cells=20)
        times = op.flux_times
        rng = resolve_stream(7).generator()
        truth = [Q_PEAK * math.sin(0.3 * j) ** 2 for j in range(12)]
        clean = op.matvec(truth)
        sigma = 0.01 * max(clean)
        data = [v + rng.gauss(0.0, sigma) for v in clean]
        m0, amp, ell = 2.0e4, 5.0e4, 15.0
        post = bayesian_flux_posterior(
            op, data, sigma, times, amplitude=amp, length_scale=ell, prior_mean=m0
        )
        a = op.sensitivity_matrix()
        c = matern_covariance(times, amplitude=amp, length_scale=ell)
        s = [
            [v + (sigma**2 if i == j else 0.0) for j, v in enumerate(row)]
            for i, row in enumerate(matmul(matmul(a, c), transpose(a)))
        ]
        s_t = tuple(tuple(r) for r in s)
        residual = [y - m0 * math.fsum(row) for y, row in zip(data, a, strict=True)]
        weights = solve_spd(s_t, tuple(residual))
        ca_t = matmul(c, transpose(a))
        mean = [
            m0 + math.fsum(ca_t[i][k] * weights[k] for k in range(len(a)))
            for i in range(12)
        ]
        scale = max(abs(v) for v in mean)
        for x, o in zip(post.mean, mean, strict=True):
            self.assertAlmostEqual(x, o, delta=1e-7 * scale)
        s_inv = inverse_spd(s_t)
        correction = matmul(matmul(ca_t, s_inv), transpose(ca_t))
        for i in range(12):
            variance = c[i][i] - correction[i][i]
            self.assertAlmostEqual(post.std[i] ** 2 / variance, 1.0, delta=1e-6)
            width = post.upper[i] - post.lower[i]
            self.assertAlmostEqual(width / (2.0 * post.std[i]), 1.959964, delta=2e-6)
            self.assertAlmostEqual(
                0.5 * (post.upper[i] + post.lower[i]), post.mean[i], delta=1e-6 * scale
            )
        # The posterior is tighter than the prior everywhere.
        self.assertTrue(all(sd < amp for sd in post.std))

    def test_evidence_selection_picks_the_maximum(self) -> None:
        """The returned posterior has the largest evidence on the grid."""
        op = operator((0.002,))
        data = synthetic_ihcp_data(
            SLAB, [0.002], triangle, flux_intervals=NQ, interval_length=DELTA, seed=3
        )
        amps, scales = [0.2e5, 0.6e5, 1.8e5], [4.0, 12.0, 36.0]
        best = select_gp_hyperparameters(
            op,
            data.noisy,
            data.noise_std,
            op.flux_times,
            amplitudes=amps,
            length_scales=scales,
        )
        for amp in amps:
            for ell in scales:
                other = bayesian_flux_posterior(
                    op,
                    data.noisy,
                    data.noise_std,
                    op.flux_times,
                    amplitude=amp,
                    length_scale=ell,
                )
                self.assertLessEqual(other.log_evidence, best.log_evidence + 1e-12)

    def prior_draw_coverage(self, seeds: range) -> tuple[float, float]:
        """Return mean and min coverage when the truth is drawn from the prior."""
        op = operator((0.003,))
        times = op.flux_times
        amp, ell = 5.0e4, 10.0
        factor = cholesky(matern_covariance(times, amplitude=amp, length_scale=ell))
        coverage = []
        for seed in seeds:
            rng = resolve_stream(1000 + seed).generator()
            z = [rng.gauss(0.0, 1.0) for _ in range(NQ)]
            truth = [
                math.fsum(factor[i][j] * z[j] for j in range(i + 1)) for i in range(NQ)
            ]
            clean = op.matvec(truth)
            sigma = 0.01 * max(abs(v) for v in clean)
            data = [v + rng.gauss(0.0, sigma) for v in clean]
            post = bayesian_flux_posterior(
                op, data, sigma, times, amplitude=amp, length_scale=ell
            )
            coverage.append(post.coverage(truth))
        return sum(coverage) / len(coverage), min(coverage)

    def test_coverage_is_calibrated_when_the_prior_is_true(self) -> None:
        """Truth drawn from the prior: mean pointwise coverage near 0.95 (40 seeds)."""
        mean, worst = self.prior_draw_coverage(range(40))
        self.assertGreater(mean, 0.92)
        self.assertLess(mean, 0.98)
        self.assertGreater(worst, 0.75)

    def test_coverage_on_fine_mesh_pulses_and_an_overconfident_prior(self) -> None:
        """Model error: a broad prior still covers; a too-tight prior does not."""
        op = operator((0.002, 0.004))
        broad, tight = [], []
        for seed in range(5):
            data = synthetic_ihcp_data(
                SLAB,
                [0.002, 0.004],
                triangle,
                flux_intervals=NQ,
                interval_length=DELTA,
                seed=seed,
            )
            broad.append(
                bayesian_flux_posterior(
                    op,
                    data.noisy,
                    data.noise_std,
                    op.flux_times,
                    amplitude=6.0e4,
                    length_scale=12.0,
                ).coverage(data.truth)
            )
            tight.append(
                bayesian_flux_posterior(
                    op,
                    data.noisy,
                    data.noise_std,
                    op.flux_times,
                    amplitude=1.0e3,
                    length_scale=12.0,
                    prior_mean=0.0,
                ).coverage(data.truth)
            )
        self.assertGreater(sum(broad) / 5, 0.9)
        self.assertLess(sum(tight) / 5, 0.5)

    def test_bayesian_validation(self) -> None:
        """Invalid arguments are rejected."""
        op = operator((0.002,))
        data = [1.0] * NQ
        with self.assertRaises(ValueError):
            bayesian_flux_posterior(
                op, data, 1.0, op.flux_times, amplitude=1.0, length_scale=1.0, level=1.0
            )
        with self.assertRaises(ValueError):
            bayesian_flux_posterior(
                op, data, 1.0, op.flux_times[:-1], amplitude=1.0, length_scale=1.0
            )
        with self.assertRaises(ValueError):
            bayesian_flux_posterior(
                op, data, 0.0, op.flux_times, amplitude=1.0, length_scale=1.0
            )


class ResolutionTests(unittest.TestCase):
    """Time lag and smoothing caused by the sensor depth."""

    def test_sensitivity_coefficients_versus_depth(self) -> None:
        """Deeper sensors are weaker and later; peak time ~ x^2 / (2 alpha)."""
        rows = sensitivity_summary(
            SLAB, [0.001, 0.002, 0.004, 0.006], flux_intervals=60, interval_length=2.0
        )
        for a, b in zip(rows, rows[1:], strict=False):
            self.assertGreater(a.peak_coefficient, b.peak_coefficient)
            self.assertLess(a.peak_time, b.peak_time)
            self.assertGreater(a.step_at_end, b.step_at_end)
        for row in rows:
            self.assertAlmostEqual(row.theory_peak_time, row.depth**2 / (2 * ALPHA), 12)
            self.assertGreaterEqual(row.peak_time, row.theory_peak_time)
            self.assertLessEqual(row.peak_time, row.theory_peak_time + 2.5 * 2.0)
        self.assertGreater(rows[0].peak_coefficient / rows[-1].peak_coefficient, 4.0)

    def test_equal_noise_resolution_degrades_with_depth(self) -> None:
        """At the same noise level the kernel is wider and later for deeper sensors."""
        sigma = 1.7287
        widths, lags = [], []
        for depth in (0.002, 0.004, 0.006):
            result = resolution_at_noise_level(operator((depth,)), sigma, DELTA, 3000.0)
            self.assertAlmostEqual(result.noise_std / 3000.0, 1.0, delta=1e-9)
            widths.append(result.resolution.width)
            lags.append(result.resolution.lag)
        self.assertLess(widths[0], widths[1])
        self.assertLess(widths[1], widths[2])
        self.assertLess(lags[0], lags[1])
        self.assertLess(lags[1], lags[2])
        self.assertGreater(widths[2], 1.4 * widths[0])

    def test_equal_noise_resolution_matches_independent_computation(self) -> None:
        """Eigen-based shortcut vs Cholesky kernel and a dense-inverse variance."""
        op = operator((0.004,))
        sigma, target = 1.7287, 3000.0
        result = resolution_at_noise_level(op, sigma, DELTA, target)
        kernel = averaging_kernel(op, sigma, result.regularization)
        summary = kernel_resolution(kernel, DELTA)
        self.assertAlmostEqual(summary.width, result.resolution.width, delta=1e-6)
        self.assertAlmostEqual(summary.lag, result.resolution.lag, delta=1e-6)
        a = op.sensitivity_matrix()
        lam = result.regularization
        normal = tuple(
            tuple(
                math.fsum(a[r][i] * a[r][j] for r in range(NQ)) / sigma**2
                + (lam * lam if i == j else 0.0)
                for j in range(NQ)
            )
            for i in range(NQ)
        )
        covariance = matmul(kernel, inverse_spd(normal))
        std = math.sqrt(sum(covariance[i][i] for i in range(NQ)) / NQ)
        self.assertAlmostEqual(std / target, 1.0, delta=1e-5)

    def test_first_order_kernel_preserves_constants(self) -> None:
        """With L = first difference, R 1 = 1; with L = I the rows shrink."""
        op = operator((0.003,))
        sigma = 1.7287
        smooth = averaging_kernel(op, sigma, 5.0e-4, order=1)
        for row in smooth:
            self.assertAlmostEqual(sum(row), 1.0, delta=1e-6)
        ridge = averaging_kernel(op, sigma, 5.0e-4, order=0)
        self.assertTrue(any(abs(sum(row) - 1.0) > 1e-4 for row in ridge))

    def test_kernel_resolution_closed_form(self) -> None:
        """A symmetric three-point kernel has zero lag and a known width."""
        n = 12
        kernel = tuple(
            tuple(0.5 if i == j else 0.25 if abs(i - j) == 1 else 0.0 for j in range(n))
            for i in range(n)
        )
        summary = kernel_resolution(kernel, 2.0, rows=[3, 4, 5])
        self.assertAlmostEqual(summary.lag, 0.0, 14)
        self.assertAlmostEqual(summary.width, 2.0 * math.sqrt(0.5), 14)
        self.assertAlmostEqual(summary.gain, 1.0, 14)
        with self.assertRaises(ValueError):
            kernel_resolution(kernel, 2.0, rows=[99])


class FailureCaseTests(unittest.TestCase):
    """What goes wrong, and what the library reports."""

    def test_single_step_inversion_amplifies_noise_at_depth(self) -> None:
        """Beck r = 1 at 4 mm diverges (measured relative error 5e11); r = 3 is fine."""
        op = operator((0.004,))
        data = synthetic_ihcp_data(
            SLAB, [0.004], triangle, flux_intervals=NQ, interval_length=DELTA, seed=3
        )
        bad = beck_sequential_estimate(op, data.noisy, future_steps=1)
        good = beck_sequential_estimate(op, data.noisy, future_steps=3)
        self.assertGreater(relative_error(bad.estimate, data.truth), 10.0)
        self.assertLess(relative_error(good.estimate, data.truth), 0.3)

    def test_gcv_can_fail_silently_at_depth(self) -> None:
        """At 6 mm GCV picks a tiny lambda (status 'interior'); discrepancy works."""
        op = operator((0.006,))
        data = synthetic_ihcp_data(
            SLAB, [0.006], triangle, flux_intervals=NQ, interval_length=DELTA, seed=3
        )
        grid = small_grid(op, data.noise_std, 24, 1e-6, 1.0)
        gcv = tikhonov_flux_estimate(
            op,
            data.noisy,
            data.noise_std,
            order=0,
            criterion="gcv",
            regularizations=grid,
        )
        rule = tikhonov_flux_estimate(
            op,
            data.noisy,
            data.noise_std,
            order=0,
            criterion="discrepancy",
            regularizations=grid,
        )
        assert gcv.estimate is not None and rule.estimate is not None
        self.assertEqual(gcv.status, "interior")
        self.assertGreater(relative_error(gcv.estimate, data.truth), 5.0)
        self.assertLess(relative_error(rule.estimate, data.truth), 0.3)

    def test_underestimated_noise_overfits(self) -> None:
        """Telling the discrepancy rule sigma / 100 yields a worse estimate."""
        op = operator((0.002,))
        data = synthetic_ihcp_data(
            SLAB, [0.002], triangle, flux_intervals=NQ, interval_length=DELTA, seed=3
        )
        grid = small_grid(op, data.noise_std)
        honest = tikhonov_flux_estimate(
            op,
            data.noisy,
            data.noise_std,
            order=0,
            criterion="discrepancy",
            regularizations=grid,
        )
        wrong = tikhonov_flux_estimate(
            op,
            data.noisy,
            data.noise_std / 100.0,
            order=0,
            criterion="discrepancy",
            regularizations=small_grid(op, data.noise_std / 100.0, 20, 1e-7, 1.0),
        )
        cg = cgls_flux_estimate(op, data.noisy, data.noise_std / 100.0)
        assert honest.estimate is not None and wrong.estimate is not None
        assert cg.estimate is not None
        self.assertGreater(
            relative_error(wrong.estimate, data.truth),
            2.0 * relative_error(honest.estimate, data.truth),
        )
        self.assertGreater(
            relative_error(cg.estimate, data.truth),
            2.0 * relative_error(honest.estimate, data.truth),
        )

    def test_unattainable_discrepancy_returns_no_estimate(self) -> None:
        """With only strongly regularized candidates no lambda meets the target."""
        op = operator((0.002,))
        data = synthetic_ihcp_data(
            SLAB, [0.002], triangle, flux_intervals=NQ, interval_length=DELTA, seed=3
        )
        scale = (
            math.sqrt(math.fsum(v * v for row in op.sensitivity_matrix() for v in row))
            / data.noise_std
        )
        result = tikhonov_flux_estimate(
            op,
            data.noisy,
            data.noise_std,
            order=0,
            criterion="discrepancy",
            regularizations=[1e3 * scale, 1e4 * scale, 1e5 * scale],
        )
        self.assertEqual(result.status, "unattainable")
        self.assertIsNone(result.estimate)
        self.assertIsNone(result.regularization)

    def test_cgls_reports_unmet_discrepancy(self) -> None:
        """If the target is below the attainable residual the status says so."""
        op = operator((0.002,))
        data = synthetic_ihcp_data(
            SLAB, [0.002], triangle, flux_intervals=NQ, interval_length=DELTA, seed=3
        )
        result = cgls_flux_estimate(
            op, data.noisy, 1e-6 * data.noise_std, max_iterations=3
        )
        self.assertEqual(result.status, "discrepancy_unmet")
        self.assertEqual(result.iterations, 3)

    def test_wrong_conductivity_biases_the_flux(self) -> None:
        """A 20% conductivity error inflates the estimated energy by about 5%."""
        wrong = SlabProperties(1.2, 1450.0, 1250.0, 0.02)
        op_right = operator((0.002,))
        op_wrong = SlabFluxOperator(
            wrong, [0.002], flux_intervals=NQ, interval_length=DELTA
        )
        data = synthetic_ihcp_data(
            SLAB, [0.002], triangle, flux_intervals=NQ, interval_length=DELTA, seed=3
        )
        grid = small_grid(op_right, data.noise_std)
        right = tikhonov_flux_estimate(
            op_right,
            data.noisy,
            data.noise_std,
            order=1,
            criterion="discrepancy",
            regularizations=grid,
        )
        biased = tikhonov_flux_estimate(
            op_wrong,
            data.noisy,
            data.noise_std,
            order=1,
            criterion="discrepancy",
            regularizations=grid,
        )
        assert right.estimate is not None and biased.estimate is not None
        energy_ratio = sum(biased.estimate) / sum(data.truth)
        self.assertGreater(energy_ratio, 1.02)
        self.assertLess(energy_ratio, 1.12)
        self.assertGreater(
            relative_error(biased.estimate, data.truth),
            relative_error(right.estimate, data.truth),
        )

    def test_selection_argument_validation(self) -> None:
        """Invalid options are rejected."""
        op = operator((0.002,))
        data = [1.0] * NQ
        with self.assertRaises(ValueError):
            tikhonov_flux_estimate(op, data, 1.0, criterion="evidence")
        with self.assertRaises(ValueError):
            tikhonov_flux_estimate(op, data, 1.0, order=2)
        with self.assertRaises(ValueError):
            tikhonov_flux_estimate(op, data, -1.0)
        with self.assertRaises(ValueError):
            tikhonov_flux_estimate(op, data[:-1], 1.0)
        with self.assertRaises(ValueError):
            cgls_flux_estimate(op, data, 1.0, tau=0.5)
        with self.assertRaises(ValueError):
            resolution_at_noise_level(op, 1.0, DELTA, 1e-30)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowCoverageStudy(unittest.TestCase):
    """Replicate studies of the Bayesian band (set PROBSTAT_SLOW=1)."""

    def test_prior_draw_coverage_many_seeds(self) -> None:
        """400 prior draws: pointwise coverage within a few MCSE of 0.95."""
        case = BayesianTests()
        mean, worst = case.prior_draw_coverage(range(400))
        print(
            f"\nSLOW prior-draw coverage (400 seeds): mean={mean:.4f} min={worst:.4f}"
        )
        self.assertGreater(mean, 0.93)
        self.assertLess(mean, 0.97)

    def test_fine_mesh_coverage_many_seeds(self) -> None:
        """100 noise seeds on the fine-mesh triangle and step, evidence-chosen prior."""
        op = operator((0.002, 0.004))
        for name, flux in (("triangle", triangle), ("step", rectangle)):
            coverage = []
            for seed in range(100):
                data = synthetic_ihcp_data(
                    SLAB,
                    [0.002, 0.004],
                    flux,
                    flux_intervals=NQ,
                    interval_length=DELTA,
                    seed=seed,
                )
                post = bayesian_flux_posterior(
                    op,
                    data.noisy,
                    data.noise_std,
                    op.flux_times,
                    amplitude=6.0e4,
                    length_scale=12.0,
                )
                coverage.append(post.coverage(data.truth))
            mean = sum(coverage) / len(coverage)
            print(
                f"\nSLOW fine-mesh {name} coverage (100 seeds): "
                f"mean={mean:.4f} min={min(coverage):.4f}"
            )
            self.assertGreater(mean, 0.9)

    def test_tikhonov_depth_monotone_single_seed(self) -> None:
        """Order-0 discrepancy Tikhonov at 2, 4, 6 mm (seed 3)."""
        errors = []
        for depth in (0.002, 0.004, 0.006):
            op = operator((depth,))
            data = synthetic_ihcp_data(
                SLAB,
                [depth],
                triangle,
                flux_intervals=NQ,
                interval_length=DELTA,
                seed=3,
            )
            result = tikhonov_flux_estimate(
                op,
                data.noisy,
                data.noise_std,
                order=0,
                criterion="discrepancy",
                regularizations=small_grid(op, data.noise_std),
            )
            assert result.estimate is not None
            errors.append(relative_error(result.estimate, data.truth))
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])


if __name__ == "__main__":
    unittest.main()
