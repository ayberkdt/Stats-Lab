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
import unittest

from probstat.inverse import (
    check_derivatives,
    gauss_newton_krylov,
    spectral_diagnostics,
)
from probstat.inverse.benchmark_orbit import (
    TwoBodyRangeModel,
    orbit_determination_benchmark,
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


if __name__ == "__main__":
    unittest.main()
