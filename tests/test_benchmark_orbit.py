"""Tests for the A01 two-body range orbit-determination benchmark.

Order: closed-form dynamics (circular orbit, energy, volume preservation,
Phi(t, t) = I), derivative checks against independent central differences,
observability geometry, a solver recovery, then adversarial inputs.

Derivative choice (module docstring, option (b)): the sensitivities are the
EXACT derivatives of the RK4-discretized map, so finite differences of
``value`` agree to the central-difference error ``O(e^2)`` plus rounding
``eps/e`` (not to ``O(h^4)``), and the Taylor remainder is second order.

Measured at the truth of the default geometries (whitened by noise_sd 1e-3,
recorded on the development machine): 2D well observed has condition number
~131 (sigma_min ~573); 2D single-station short arc ~369 (sigma_min ~10.4);
3D single-station short arc is rank deficient (sigma_min ~ 4e-14, rotation of
the orbit about the line of sight leaves all ranges unchanged) while the 3D
well-observed condition number is ~181.
"""

from __future__ import annotations

import math
import os
import random
import unittest
from statistics import NormalDist

from probstat.inverse import (
    check_derivatives,
    gauss_newton_krylov,
    spectral_diagnostics,
)
from probstat.inverse.benchmark_orbit import (
    CALIBRATION_SEEDS,
    TwoBodyRangeModel,
    correlation_calibration_study,
    orbit_determination_benchmark,
    range_noise_covariance,
    station_bias_identifiability,
)

STATIONS_2D = ((4.0, 0.5), (-1.0, 4.0))
CIRCULAR_2D = (1.0, 0.0, 0.0, 1.0)


def _det(matrix: list[list[float]]) -> float:
    """Return the determinant by Gaussian elimination with pivoting."""
    a = [list(r) for r in matrix]
    n = len(a)
    det = 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if p != c:
            a[c], a[p] = a[p], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            for k in range(c, n):
                a[r][k] -= f * a[c][k]
    return det


def _dense_jacobian(model: TwoBodyRangeModel, x: tuple[float, ...]):  # type: ignore[no-untyped-def]
    """Return the Jacobian assembled from jvp on unit vectors."""
    n = model.input_dimension
    cols = [
        model.jvp(x, tuple(1.0 if i == c else 0.0 for i in range(n))) for c in range(n)
    ]
    return [[cols[c][r] for c in range(n)] for r in range(model.output_dimension)]


class DynamicsTest(unittest.TestCase):
    """Circular orbit, energy and the state transition matrix."""

    def test_circular_orbit_closed_form_and_h4(self) -> None:
        """Check: circular orbit closed form and h4."""
        times = (1.0, 2.0, 3.0)
        errs = []
        energy_errs = []
        for steps in (4, 8, 16):
            m = TwoBodyRangeModel(times, STATIONS_2D, steps_per_interval=steps)
            states = m.states(CIRCULAR_2D)
            err = max(
                max(
                    abs(s[0] - math.cos(t)),
                    abs(s[1] - math.sin(t)),
                    abs(s[2] + math.sin(t)),
                    abs(s[3] - math.cos(t)),
                )
                for s, t in zip(states, times, strict=True)
            )
            errs.append(err)
            e0 = m.energy(CIRCULAR_2D)
            energy_errs.append(max(abs(m.energy(s) - e0) for s in states))
        for series in (errs, energy_errs):
            for coarse, fine in zip(series, series[1:], strict=False):
                self.assertGreater(coarse / fine, 12.0)  # order 4: 16 asymptotically
                self.assertLess(coarse / fine, 40.0)
        self.assertLess(errs[-1], 1e-5)
        self.assertLess(energy_errs[-1], 1e-5)

    def test_state_transition_volume_and_identity(self) -> None:
        """Check: state transition volume and identity."""
        m = TwoBodyRangeModel((1.5, 3.0), STATIONS_2D, steps_per_interval=40)
        x = (1.0, 0.0, 0.0, 1.05)
        for z in m.sensitivities(x):
            self.assertAlmostEqual(_det([list(r) for r in z]), 1.0, delta=1e-6)
        tiny = TwoBodyRangeModel((1e-9,), STATIONS_2D, steps_per_interval=1)
        z = tiny.sensitivities(x)[0]
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(z[i][j], 1.0 if i == j else 0.0, delta=1e-8)

    def test_sensitivities_equal_finite_difference_of_state(self) -> None:
        """Check: sensitivities equal finite difference of state."""
        m = TwoBodyRangeModel((1.0, 2.0), STATIONS_2D, estimate_mu=True)
        x = (1.0, 0.0, 0.0, 1.05, 1.0)
        z = m.sensitivities(x)
        e = 1e-6
        for c in range(5):
            xp = tuple(v + (e if i == c else 0.0) for i, v in enumerate(x))
            xm = tuple(v - (e if i == c else 0.0) for i, v in enumerate(x))
            sp, sm = m.states(xp), m.states(xm)
            for t in range(2):
                for r in range(4):
                    fd = (sp[t][r] - sm[t][r]) / (2 * e)
                    self.assertAlmostEqual(z[t][r][c], fd, delta=1e-6)


class DerivativeTest(unittest.TestCase):
    """jvp/vjp consistency with the discrete map."""

    def test_jvp_vs_central_differences_and_dot_test(self) -> None:
        """Check: jvp vs central differences and dot test."""
        b = orbit_determination_benchmark(estimate_mu=True, steps_per_interval=6)
        m = b.model
        x = tuple(t + 0.01 * (i + 1) for i, t in enumerate(b.truth))
        v = (0.3, -0.2, 0.5, 0.1, -0.4)
        e = 1e-6
        plus = m.value(tuple(a + e * c for a, c in zip(x, v, strict=True)))
        minus = m.value(tuple(a - e * c for a, c in zip(x, v, strict=True)))
        jv = m.jvp(x, v)
        for p, q, d in zip(plus, minus, jv, strict=True):
            self.assertAlmostEqual((p - q) / (2 * e), d, delta=1e-6)
        w = tuple(math.sin(i + 1.0) for i in range(m.output_dimension))
        left = sum(a * c for a, c in zip(w, jv, strict=True))
        right = sum(a * c for a, c in zip(m.vjp(x, w), v, strict=True))
        self.assertLess(abs(left - right) / max(1.0, abs(left)), 1e-12)
        check = check_derivatives(m, x, v, w)
        self.assertTrue(check.passed)

    def test_range_jacobian_row(self) -> None:
        """Check: range jacobian row."""
        m = TwoBodyRangeModel((0.7,), ((4.0, 0.5),), steps_per_interval=10)
        x = (1.0, 0.0, 0.0, 1.05)
        jac = _dense_jacobian(m, x)[0]
        e = 1e-6
        for c in range(4):
            xp = tuple(v + (e if i == c else 0.0) for i, v in enumerate(x))
            xm = tuple(v - (e if i == c else 0.0) for i, v in enumerate(x))
            fd = (m.value(xp)[0] - m.value(xm)[0]) / (2 * e)
            self.assertAlmostEqual(jac[c], fd, delta=1e-7)
        # as t -> 0 the range partial is the unit line of sight (analytic)
        tiny = TwoBodyRangeModel((1e-9,), ((4.0, 0.5),), steps_per_interval=1)
        row = _dense_jacobian(tiny, x)[0]
        rho = math.hypot(1.0 - 4.0, -0.5)
        self.assertAlmostEqual(row[0], (1.0 - 4.0) / rho, delta=1e-7)
        self.assertAlmostEqual(row[1], -0.5 / rho, delta=1e-7)
        self.assertAlmostEqual(row[2], 0.0, delta=1e-7)

    def test_counts(self) -> None:
        """Check: counts."""
        m = TwoBodyRangeModel((1.0,), STATIONS_2D)
        x = (1.0, 0.0, 0.0, 1.05)
        m.value(x)
        m.jvp(x, (1.0, 0.0, 0.0, 0.0))
        m.vjp(x, (1.0, 0.0))
        c = m.counts
        self.assertEqual((c.value, c.jvp, c.vjp, c.derivative_value), (1, 1, 1, 0))


class GeometryTest(unittest.TestCase):
    """Observability and recovery."""

    def _spectrum(self, dimension: int, geometry: str):  # type: ignore[no-untyped-def]
        b = orbit_determination_benchmark(dimension=dimension, geometry=geometry)
        j = _dense_jacobian(b.model, b.truth)  # type: ignore[arg-type]
        j = [[v / b.noise_level for v in row] for row in j]
        return spectral_diagnostics(j)

    def test_short_arc_single_station_is_weaker(self) -> None:
        """Check: short arc single station is weaker."""
        good = self._spectrum(2, "well_observed")
        weak = self._spectrum(2, "single_station_short_arc")
        ratio = weak.condition_number / good.condition_number
        self.assertGreater(ratio, 2.0)  # measured ~2.8
        self.assertLess(weak.singular_values[-1], 0.05 * good.singular_values[-1])
        self.assertEqual(good.rank, 4)

    def test_3d_single_station_has_unobservable_rotation(self) -> None:
        """Check: 3d single station has unobservable rotation."""
        weak = self._spectrum(3, "single_station_short_arc")
        good = self._spectrum(3, "well_observed")
        self.assertEqual(weak.rank, 5)
        self.assertEqual(weak.nullspace_dimension, 1)
        self.assertEqual(good.rank, 6)
        bench = orbit_determination_benchmark(
            dimension=3, geometry="single_station_short_arc"
        )
        self.assertIn("weak_observability", bench.pathologies)

    def test_recovery_well_observed(self) -> None:
        """Check: recovery well observed."""
        b = orbit_determination_benchmark(noise_sd=1e-4, seed=23)
        self.assertFalse(b.same_discretization)
        offsets = (0.03, -0.02, 0.02, -0.03)
        start = tuple(t + d for t, d in zip(b.truth, offsets, strict=True))
        res = gauss_newton_krylov(
            b.model, b.data, noise=b.noise, initial=start, max_iterations=30
        )
        self.assertTrue(res.converged, res.stop_reason)
        err = max(abs(a - t) for a, t in zip(res.estimate, b.truth, strict=True))
        self.assertLess(err, 5e-4)  # noise 1e-4, sigma_min(whitened J) ~ 5e3 * 1e-1
        self.assertGreater(res.residual_norm, 0.0)

    def test_inverse_crime_flag_and_seed(self) -> None:
        """Check: inverse crime flag and seed."""
        a = orbit_determination_benchmark(data_refinement=1, seed=5)
        self.assertTrue(a.same_discretization)
        b = orbit_determination_benchmark(data_refinement=1, seed=5)
        self.assertEqual(a.data, b.data)
        self.assertEqual(a.exact_data, a.model.value(a.truth))
        c = orbit_determination_benchmark(data_refinement=4, seed=5)
        diff = max(abs(p - q) for p, q in zip(a.exact_data, c.exact_data, strict=True))
        self.assertLess(diff, 1e-3)
        self.assertGreater(diff, 0.0)


class InvalidInputTest(unittest.TestCase):
    """Adversarial arguments."""

    def test_constructor_validation(self) -> None:
        """Check: constructor validation."""
        bad_cases = (
            {"times": (1.0, 1.0)},
            {"times": (2.0, 1.0)},
            {"times": (0.0, 1.0)},
            {"stations": ((1.0, 2.0), (1.0, 2.0, 3.0))},
            {"stations": ()},
            {"mu": 0.0},
            {"mu": -1.0},
            {"steps_per_interval": 0},
        )
        for case in bad_cases:
            kwargs = {"times": (1.0,), "stations": STATIONS_2D}
            kwargs.update(case)
            with self.subTest(case=case), self.assertRaises(ValueError):
                TwoBodyRangeModel(
                    kwargs["times"],  # type: ignore[arg-type]
                    kwargs["stations"],  # type: ignore[arg-type]
                    mu=kwargs.get("mu", 1.0),  # type: ignore[arg-type]
                    steps_per_interval=kwargs.get("steps_per_interval", 4),  # type: ignore[arg-type]
                )
        with self.assertRaises(TypeError):
            TwoBodyRangeModel("ab", STATIONS_2D)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            orbit_determination_benchmark(geometry="nope")
        with self.assertRaises(ValueError):
            orbit_determination_benchmark(dimension=4)
        with self.assertRaises(ValueError):
            orbit_determination_benchmark(noise_sd=0.0)

    def test_zero_range_and_lengths_raise(self) -> None:
        """Check: zero range and lengths raise."""
        m = TwoBodyRangeModel((1e-9,), ((1.0, 0.0),), steps_per_interval=1)
        with self.assertRaises(ValueError):
            m.value((1.0, 0.0, 0.0, 0.0))  # still at the station
        with self.assertRaises(ValueError):
            m.value((1.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            m.jvp((1.0, 0.0, 0.0, 1.0), (1.0,))
        with self.assertRaises(ValueError):
            m.vjp((1.0, 0.0, 0.0, 1.0), (1.0, 2.0))
        with self.assertRaises(ValueError):
            m.value((0.0, 0.0, 0.0, 1.0))  # center collision

    def test_nonpositive_estimated_mu_is_rejected_trial(self) -> None:
        """Check: nonpositive estimated mu is rejected trial."""
        m = TwoBodyRangeModel((1.0,), STATIONS_2D, estimate_mu=True)
        out = m.value((1.0, 0.0, 0.0, 1.0, -0.5))
        self.assertTrue(all(math.isnan(v) for v in out))
        with self.assertRaises(ValueError):
            m.jvp((1.0, 0.0, 0.0, 1.0, -0.5), (0.0,) * 5)


# ---------------------------------------------------------------------------
# Roadmap 20C additions (SYNTHETIC): station biases and correlated noise
# ---------------------------------------------------------------------------


class StationBiasModelTests(unittest.TestCase):
    """Per-station range biases as exactly linear nuisance parameters."""

    def _models(self) -> tuple[TwoBodyRangeModel, TwoBodyRangeModel]:
        """Return the same design without and with station biases."""
        times = (0.5, 1.1, 1.8, 2.4)
        plain = TwoBodyRangeModel(times, STATIONS_2D, steps_per_interval=8)
        biased = TwoBodyRangeModel(
            times, STATIONS_2D, steps_per_interval=8, estimate_station_biases=True
        )
        return plain, biased

    def test_value_adds_bias_per_station(self) -> None:
        """``rho_ij + b_j`` exactly, station-minor ordering."""
        plain, biased = self._models()
        orbit = (1.0, 0.0, 0.0, 1.05)
        base = plain.value(orbit)
        out = biased.value((*orbit, 0.01, -0.02))
        self.assertEqual(biased.input_dimension, 6)
        for k, (a, b) in enumerate(zip(base, out, strict=True)):
            self.assertAlmostEqual(b - a, (0.01, -0.02)[k % 2], places=14)

    def test_jacobian_blocks(self) -> None:
        """Orbit columns are unchanged; bias columns are station indicators."""
        plain, biased = self._models()
        orbit = (1.0, 0.0, 0.0, 1.05)
        x = (*orbit, 0.01, -0.02)
        for j in range(4):
            e = tuple(1.0 if k == j else 0.0 for k in range(4))
            self.assertEqual(plain.jvp(orbit, e), biased.jvp(x, (*e, 0.0, 0.0)))
        for s in range(2):
            e = tuple(1.0 if k == s else 0.0 for k in range(2))
            col = biased.jvp(x, (0.0,) * 4 + e)
            self.assertEqual(col, tuple(1.0 if k % 2 == s else 0.0 for k in range(8)))

    def test_derivatives_against_central_differences_and_dot_test(self) -> None:
        """Taylor sweep, adjoint dot test and finite differences with ``mu``."""
        biased = TwoBodyRangeModel(
            (0.6, 1.3, 2.0),
            STATIONS_2D,
            steps_per_interval=8,
            estimate_mu=True,
            estimate_station_biases=True,
        )
        x = (1.0, 0.0, 0.0, 1.05, 1.02, 0.01, -0.02)
        check = check_derivatives(
            biased, x, (0.1, -0.2, 0.3, 0.1, 0.05, 0.2, -0.1), (1, -1, 2, 0.5, 1, -2)
        )
        self.assertTrue(check.passed, check)
        self.assertLess(check.dot_relative_error, 1e-12)
        jac = _dense_jacobian(biased, x)
        for j in range(7):
            h = 1e-6
            up, dn = list(x), list(x)
            up[j] += h
            dn[j] -= h
            fp, fm = biased.value(up), biased.value(dn)
            for i in range(6):
                fd = (fp[i] - fm[i]) / (2 * h)
                self.assertAlmostEqual(jac[i][j], fd, delta=1e-6 * (1 + abs(fd)))
        # the station-bias columns are exactly the 0/1 indicators
        sens = biased.sensitivities(x)
        self.assertEqual(len(sens[0][0]), 5)  # state block (4) plus mu only

    def test_benchmark_with_biases_generates_data_with_them(self) -> None:
        """Truth ends with the biases and the data contain them; default off."""
        plain = orbit_determination_benchmark(seed=3)
        biased = orbit_determination_benchmark(seed=3, estimate_station_biases=True)
        self.assertEqual(len(biased.truth), len(plain.truth) + 3)
        self.assertEqual(biased.truth[4:], (0.01, -0.015, 0.02))
        for k, (a, b) in enumerate(
            zip(plain.exact_data, biased.exact_data, strict=True)
        ):
            self.assertAlmostEqual(b - a, biased.truth[4 + k % 3], places=13)
        self.assertEqual(plain.data, orbit_determination_benchmark(seed=3).data)
        self.assertIn("nuisance_degeneracy", biased.pathologies)
        self.assertNotIn("nuisance_degeneracy", plain.pathologies)
        custom = orbit_determination_benchmark(
            estimate_station_biases=True, station_biases=(0.1, 0.2, 0.3)
        )
        self.assertEqual(custom.truth[4:], (0.1, 0.2, 0.3))
        with self.assertRaises(ValueError):
            orbit_determination_benchmark(
                estimate_station_biases=True, station_biases=(0.1,)
            )

    def test_biased_solver_recovers_biases(self) -> None:
        """Gauss-Newton on the well-observed design recovers orbit and biases."""
        case = orbit_determination_benchmark(
            estimate_station_biases=True, noise_sd=1e-3, seed=7
        )
        start = tuple(t + 0.02 for t in case.truth[:4]) + (0.0, 0.0, 0.0)
        result = gauss_newton_krylov(
            case.model, case.data, noise=case.noise, initial=start, max_iterations=30
        )
        self.assertTrue(result.converged, result.stop_reason)
        ident = station_bias_identifiability(estimate_station_biases=True)
        for est, true, sd in zip(
            result.estimate, case.truth, ident.parameter_sd, strict=True
        ):
            self.assertLess(abs(est - true), 5.0 * sd)

    def test_invalid(self) -> None:
        """Type errors and wrong lengths."""
        with self.assertRaises(TypeError):
            TwoBodyRangeModel((1.0,), STATIONS_2D, estimate_station_biases=1)  # type: ignore[arg-type]
        _, biased = self._models()
        with self.assertRaises(ValueError):
            biased.value((1.0, 0.0, 0.0, 1.05))
        with self.assertRaises(ValueError):
            biased.vjp((1.0, 0.0, 0.0, 1.05, 0.0, 0.0), (1.0,))


class BiasIdentifiabilityTests(unittest.TestCase):
    """Which biases the ranges determine, by geometry (measured, noise 1e-3).

    Measured whitened condition numbers at the truth (2D): well observed 131
    (no biases) and 182 (three biases, sd of every bias ~0.8e-3); single
    station short arc 369 (no bias) and 5.77e3 (one bias, sd 0.47). In 3D the
    single-station arc keeps its exact rotation null direction.
    """

    def test_well_observed_biases_identifiable(self) -> None:
        """All seven parameters are determined to about one noise sd."""
        r = station_bias_identifiability(estimate_station_biases=True)
        self.assertEqual(r.numerical_rank, 7)
        self.assertEqual(r.parameter_names[-3:], ("bias_1", "bias_2", "bias_3"))
        self.assertTrue(all(r.identifiable))
        self.assertLess(r.condition_number, 500.0)
        for sd in r.parameter_sd[4:]:
            self.assertLess(sd, 1.5e-3)  # biases are 10-20 sigma: well determined
        plain = station_bias_identifiability(estimate_station_biases=False)
        self.assertLess(plain.condition_number, r.condition_number)

    def test_single_station_bias_not_identifiable_in_practice(self) -> None:
        """One station on a short arc: the bias is confounded with the orbit."""
        r = station_bias_identifiability(
            geometry="single_station_short_arc", estimate_station_biases=True
        )
        plain = station_bias_identifiability(
            geometry="single_station_short_arc", estimate_station_biases=False
        )
        self.assertEqual(r.numerical_rank, 5)  # full rank numerically ...
        self.assertGreater(r.parameter_sd[4], 0.1)  # ... but sd >> the bias 0.01
        self.assertGreater(r.parameter_sd[4], 100 * 1e-3)
        self.assertGreater(r.condition_number, 10.0 * plain.condition_number)
        # adding the bias also degrades the orbit parameters
        for with_bias, without in zip(r.parameter_sd, plain.parameter_sd, strict=False):
            self.assertGreater(with_bias, 3.0 * without)

    def test_single_station_3d_has_exact_null_direction(self) -> None:
        """The rotation about the line of sight survives: rank n - 1, sd = nan."""
        r = station_bias_identifiability(
            dimension=3,
            geometry="single_station_short_arc",
            estimate_station_biases=True,
        )
        self.assertEqual(r.numerical_rank, 6)
        self.assertEqual(len(r.singular_values), 7)
        self.assertGreater(max(r.null_loading), 0.5)
        self.assertTrue(all(math.isnan(v) for v in r.parameter_sd))
        self.assertTrue(math.isinf(r.condition_number) or r.condition_number > 1e12)

    def test_correlated_noise_changes_the_spectrum(self) -> None:
        """A whitening by a correlated covariance differs from the iid one."""
        times = tuple(0.8 * (i + 1) for i in range(8))
        cov = range_noise_covariance(
            times,
            3,
            white_sd=3e-4,
            correlated_sd=2e-3,
            correlation_time=3.0,
            station_correlation=0.8,
        )
        iid = station_bias_identifiability(estimate_station_biases=True)
        corr = station_bias_identifiability(estimate_station_biases=True, noise=cov)
        self.assertNotEqual(iid.singular_values, corr.singular_values)
        with self.assertRaises(ValueError):
            station_bias_identifiability(
                noise=range_noise_covariance(
                    (1.0, 2.0),
                    1,
                    white_sd=1e-3,
                    correlated_sd=0.0,
                    correlation_time=1.0,
                )
            )

    def test_invalid(self) -> None:
        """Bad options raise."""
        with self.assertRaises(ValueError):
            station_bias_identifiability(geometry="orbit")
        with self.assertRaises(ValueError):
            station_bias_identifiability(dimension=4)
        with self.assertRaises(ValueError):
            station_bias_identifiability(rank_tolerance=0.0)
        with self.assertRaises(TypeError):
            station_bias_identifiability(estimate_station_biases=1)  # type: ignore[arg-type]


class RangeNoiseTests(unittest.TestCase):
    """The separable time-and-station covariance."""

    def test_structure(self) -> None:
        """``w^2 I + c^2 T kron S`` entrywise; SPD; station-correlation limits."""
        times = (0.0, 1.0, 3.0)
        cov = range_noise_covariance(
            times,
            2,
            white_sd=0.1,
            correlated_sd=0.2,
            correlation_time=2.0,
            station_correlation=0.5,
        )
        self.assertEqual(cov.dimension, 6)
        probe = cov.apply((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(probe[0], 0.01 + 0.04, places=14)
        self.assertAlmostEqual(probe[1], 0.04 * 0.5, places=14)
        self.assertAlmostEqual(probe[2], 0.04 * math.exp(-0.5), places=14)
        self.assertAlmostEqual(probe[3], 0.04 * math.exp(-0.5) * 0.5, places=14)
        self.assertAlmostEqual(probe[4], 0.04 * math.exp(-1.5), places=14)
        # no correlated part: a scaled identity
        white = range_noise_covariance(
            times, 2, white_sd=0.1, correlated_sd=0.0, correlation_time=2.0
        )
        for got, v in zip(
            white.apply((1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            strict=True,
        ):
            self.assertAlmostEqual(got, 0.01 * v, places=15)

    def test_invalid(self) -> None:
        """Out-of-range parameters raise."""
        kw = {"white_sd": 0.1, "correlated_sd": 0.1, "correlation_time": 1.0}
        with self.assertRaises(ValueError):
            range_noise_covariance((), 1, **kw)
        with self.assertRaises(ValueError):
            range_noise_covariance((1.0,), 0, **kw)
        with self.assertRaises(ValueError):
            range_noise_covariance((1.0,), 1, station_correlation=1.5, **kw)
        with self.assertRaises(ValueError):
            range_noise_covariance((1.0,), 1, **{**kw, "white_sd": 0.0})
        with self.assertRaises(ValueError):
            range_noise_covariance((1.0,), 1, **{**kw, "correlated_sd": -1.0})
        with self.assertRaises(ValueError):
            range_noise_covariance((1.0,), 1, **{**kw, "correlation_time": 0.0})


class CorrelationCalibrationTests(unittest.TestCase):
    """What ignoring correlated noise does to nominal intervals (linearized)."""

    def test_gls_calibrated_and_naive_misstated(self) -> None:
        """Coverage matches the analytic expectation; GLS is calibrated.

        Measured with the 200 predeclared seeds and no biases: the diagonal
        analyst's intervals are too wide (actual/nominal sd 0.98, 0.75, 0.80,
        0.99), the GLS sd are 0.51-0.61 of the nominal ones.
        """
        study = correlation_calibration_study()
        self.assertEqual(study.replicates, 200)
        self.assertEqual(study.seeds, CALIBRATION_SEEDS)
        q = NormalDist().inv_cdf(0.975)
        band = 4.0 * study.coverage_mcse
        for k, name in enumerate(study.parameter_names):
            ratio = study.actual_sd_naive[k] / study.nominal_sd_naive[k]
            expected = 2.0 * NormalDist().cdf(q / ratio) - 1.0
            self.assertLess(abs(study.coverage_naive[k] - expected), band, msg=name)
            self.assertLess(abs(study.coverage_gls[k] - 0.95), band, msg=name)
            # Gauss-Markov: GLS is no worse than the weighted estimator
            self.assertLessEqual(study.sd_gls[k], study.actual_sd_naive[k] * (1 + 1e-9))
        ratios = [
            a / n
            for a, n in zip(study.actual_sd_naive, study.nominal_sd_naive, strict=True)
        ]
        self.assertLess(min(ratios), 0.85)
        self.assertLess(
            max(study.sd_gls[k] / study.nominal_sd_naive[k] for k in range(4)), 0.7
        )

    def test_naive_intervals_can_be_too_narrow_for_a_bias(self) -> None:
        """With biases free the second station's bias sd is understated (1.16)."""
        study = correlation_calibration_study(estimate_station_biases=True)
        ratios = [
            a / n
            for a, n in zip(study.actual_sd_naive, study.nominal_sd_naive, strict=True)
        ]
        self.assertGreater(ratios[5], 1.1)
        self.assertEqual(study.parameter_names[5], "bias_2")
        self.assertLess(ratios[1], 0.7)

    def test_zero_correlated_noise_is_calibrated(self) -> None:
        """Without a correlated part the naive and GLS analyses coincide."""
        study = correlation_calibration_study(correlated_sd=0.0, seeds=range(50))
        for a, b, c in zip(
            study.nominal_sd_naive, study.actual_sd_naive, study.sd_gls, strict=True
        ):
            self.assertAlmostEqual(a / b, 1.0, places=9)
            self.assertAlmostEqual(a / c, 1.0, places=9)

    def test_nonlinear_solve_matches_linearized_prediction(self) -> None:
        """One Gauss-Newton fit with the GLS factor lies within the GLS bands."""
        times = tuple(0.8 * (i + 1) for i in range(8))
        stations = ((4.0, 0.5), (-1.0, 4.0), (-3.5, -2.5))
        gamma = range_noise_covariance(
            times,
            3,
            white_sd=3e-4,
            correlated_sd=2e-3,
            correlation_time=3.0,
            station_correlation=0.8,
        )
        model = TwoBodyRangeModel(times, stations, steps_per_interval=12)
        truth = (1.0, 0.0, 0.0, 1.05)
        gen = TwoBodyRangeModel(times, stations, steps_per_interval=48)
        rng = random.Random(41)
        noise = gamma.unwhiten([rng.gauss(0.0, 1.0) for _ in range(24)])
        data = tuple(b + e for b, e in zip(gen.value(truth), noise, strict=True))
        result = gauss_newton_krylov(
            model,
            data,
            noise=gamma,
            initial=(1.02, 0.02, -0.02, 1.03),
            max_iterations=30,
        )
        self.assertTrue(result.converged, result.stop_reason)
        study = correlation_calibration_study(seeds=(1, 2))
        for est, true, sd in zip(result.estimate, truth, study.sd_gls, strict=True):
            self.assertLess(abs(est - true), 4.5 * sd)

    def test_invalid(self) -> None:
        """Bad seeds, levels and designs raise."""
        with self.assertRaises(ValueError):
            correlation_calibration_study(seeds=())
        with self.assertRaises(ValueError):
            correlation_calibration_study(seeds=(1, 1))
        with self.assertRaises(TypeError):
            correlation_calibration_study(seeds=(1.5,))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            correlation_calibration_study(seeds="abc")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            correlation_calibration_study(level=1.0)
        with self.assertRaises(ValueError):
            correlation_calibration_study(geometry="orbit")

    def test_rank_deficient_design_has_no_estimator(self) -> None:
        """The 3D single-station arc is rank deficient: the study refuses."""
        with self.assertRaises(ValueError):
            correlation_calibration_study(
                dimension=3, geometry="single_station_short_arc", seeds=(1, 2)
            )


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowCalibrationStudy(unittest.TestCase):
    """1000 predeclared seeds: the under-coverage of a naive bias interval."""

    def test_thousand_seeds(self) -> None:
        """The second station's naive bias interval under-covers by > 3 MCSE."""
        study = correlation_calibration_study(
            estimate_station_biases=True, seeds=range(3000, 4000)
        )
        self.assertEqual(study.replicates, 1000)
        self.assertLess(study.coverage_naive[5], 0.95 - 3.0 * study.coverage_mcse)
        for cov in study.coverage_gls:
            self.assertLess(abs(cov - 0.95), 4.0 * study.coverage_mcse)


if __name__ == "__main__":
    unittest.main()
