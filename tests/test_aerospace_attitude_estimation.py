"""Tests of single-axis gyro/star-tracker estimation and the pointing budget.

Every case is synthetic. Order: closed forms and exact small cases, then an
independent oracle (a doubling Riccati solver, the library's Riccati recursion,
exact rational residuals), then stochastic checks with MCSE-based bands, then
failure cases.
"""

from __future__ import annotations

import itertools
import math
import os
import unittest
from fractions import Fraction

from probstat.aerospace.attitude_estimation import (
    ARCSEC,
    POINTING_SOURCES,
    GyroStarTrackerModel,
    SensorGrades,
    allan_model_variance,
    attitude_process_noise,
    attitude_state_space,
    fit_allan_noise,
    grade_requirement_probability,
    knowledge_sobol_ranking,
    overlapping_allan_deviation,
    pointing_budget,
    simulate_attitude_error,
    simulate_gyro_rate,
    steady_state_attitude_closed_form,
    steady_state_attitude_riccati,
)
from probstat.filtering import (
    kalman_filter,
    nees_test_from_filters,
    nis_test_from_filter,
)

SLOW = os.environ.get("PROBSTAT_SLOW") == "1"


def _residual(model: GyroStarTrackerModel, predicted: tuple) -> tuple[float, float]:
    """Relative Riccati fixed-point residual of a predicted covariance, exact.

    All arithmetic is in rationals (``Fraction``), so the only error is the
    one inside the supplied float matrix.
    """
    f = Fraction
    dt, sv, su, sn = (
        f(x) for x in (model.dt, model.sigma_v, model.sigma_u, model.sigma_n)
    )
    q11 = sv**2 * dt + su**2 * dt**3 / 3
    q12 = -(su**2) * dt**2 / 2
    q22 = su**2 * dt
    p11, p12, p22 = f(predicted[0][0]), f(predicted[0][1]), f(predicted[1][1])
    s = p11 + sn**2
    a = p11 - p11 * p11 / s
    c = p12 - p11 * p12 / s
    d = p22 - p12 * p12 / s
    n11 = a - 2 * dt * c + dt**2 * d + q11
    n12 = c - dt * d + q12
    n22 = d + q22
    del n22
    return (
        float(abs(n11 - p11) / p11),
        float(abs(n12 - p12) / abs(p12)) if p12 != 0 else float(abs(n12)),
    )


class ProcessNoiseTests(unittest.TestCase):
    """The exact discretisation of the model."""

    def test_process_noise_entries_match_the_integral(self) -> None:
        """Q equals the integral of F(s) Qc F(s)' by a fine midpoint rule."""
        model = GyroStarTrackerModel(2e-4, 3e-6, 1e-4, 0.5)
        q = attitude_process_noise(model)
        n = 20000
        h = model.dt / n
        acc = [[0.0, 0.0], [0.0, 0.0]]
        for k in range(n):
            s = (k + 0.5) * h
            # F(s) = [[1, -s], [0, 1]], Qc = diag(sv^2, su^2).
            acc[0][0] += (model.sigma_v**2 + s * s * model.sigma_u**2) * h
            acc[0][1] += -s * model.sigma_u**2 * h
            acc[1][1] += model.sigma_u**2 * h
        self.assertAlmostEqual(q[0][0], acc[0][0], delta=1e-6 * q[0][0])
        self.assertAlmostEqual(q[0][1], acc[0][1], delta=1e-6 * abs(q[0][1]))
        self.assertAlmostEqual(q[1][1], acc[1][1], delta=1e-12 * q[1][1])
        self.assertEqual(q[0][1], q[1][0])

    def test_q_is_positive_semidefinite(self) -> None:
        """The determinant of Q is sigma_u^2 sigma_v^2 dt^2 + sigma_u^4 dt^4 / 12."""
        model = GyroStarTrackerModel(1e-4, 1e-5, 1e-4, 0.2)
        q = attitude_process_noise(model)
        det = q[0][0] * q[1][1] - q[0][1] ** 2
        expected = (
            model.sigma_u**2 * model.sigma_v**2 * model.dt**2
            + model.sigma_u**4 * model.dt**4 / 12
        )
        self.assertAlmostEqual(det, expected, delta=1e-12 * expected)

    def test_state_space_matches_the_model(self) -> None:
        """F, H, R and the prior are as documented."""
        model = GyroStarTrackerModel(1e-4, 1e-6, 2e-4, 0.1)
        ss = attitude_state_space(model, initial_std=(1e-3, 1e-5))
        self.assertEqual(ss.transition_matrix, ((1.0, -0.1), (0.0, 1.0)))
        self.assertEqual(ss.observation_matrix, ((1.0, 0.0),))
        self.assertAlmostEqual(ss.observation_covariance[0][0], 4e-8)
        self.assertAlmostEqual(ss.initial_covariance[1][1], 1e-10)


class SteadyStateClosedFormTests(unittest.TestCase):
    """The analytic steady state against independent numerical solutions."""

    def test_pure_integrator_limit(self) -> None:
        """sigma_u = 0 and a short step: p11 -> sigma_n sigma_v sqrt(dt), bias known."""
        model = GyroStarTrackerModel(1e-4, 0.0, 1e-3, 1e-3)
        ss = steady_state_attitude_closed_form(model)
        # Expansion of the closed form in gamma = sigma_v^2 dt / sigma_n^2
        # (derived in the docstring): p11 = R sqrt(gamma) (1 + sqrt(gamma)/2 + ...),
        # the leading term being the Kalman-Bucy value sigma_n sigma_v sqrt(dt).
        gamma = model.sigma_v**2 * model.dt / model.sigma_n**2
        expansion = model.sigma_n**2 * math.sqrt(gamma) * (1.0 + math.sqrt(gamma) / 2.0)
        self.assertAlmostEqual(
            ss.predicted_covariance[0][0] / expansion, 1.0, delta=1e-4
        )
        self.assertEqual(ss.bias_std_filtered, 0.0)
        self.assertEqual(ss.predicted_covariance[0][1], 0.0)

    def test_matches_doubling_over_a_grid(self) -> None:
        """Closed form = doubling solver to <= 1e-7 over 300 parameter sets."""
        worst = 0.0
        count = 0
        grid = itertools.product(
            (1e-6, 1e-4, 1e-2),
            (1e-9, 1e-7, 1e-5, 1e-3),
            (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            (0.01, 0.1, 1.0, 5.0, 20.0),
        )
        for sv, su, sn, dt in grid:
            model = GyroStarTrackerModel(sv, su, sn, dt)
            closed = steady_state_attitude_closed_form(model)
            numeric = steady_state_attitude_riccati(model)
            self.assertTrue(numeric.converged, (sv, su, sn, dt))
            for got, ref in (
                (closed.predicted_covariance[0][0], numeric.predicted_covariance[0][0]),
                (closed.filtered_covariance[0][0], numeric.filtered_covariance[0][0]),
                (closed.predicted_covariance[0][1], numeric.predicted_covariance[0][1]),
            ):
                worst = max(worst, abs(got - ref) / abs(ref))
            count += 1
        self.assertEqual(count, 300)
        self.assertLess(worst, 1e-7)

    def test_closed_form_is_a_riccati_fixed_point_exactly(self) -> None:
        """Exact rational residual of the Riccati map is at rounding level."""
        for sv, su, sn, dt in (
            (1e-5, 1e-8, 0.1, 0.001),
            (0.1, 1e-10, 1e-5, 10.0),
            (1e-4, 1e-6, 1e-4, 0.1),
            (1e-3, 1e-3, 1e-3, 1.0),
            (1e-6, 1e-2, 1e-5, 5.0),
        ):
            model = GyroStarTrackerModel(sv, su, sn, dt)
            closed = steady_state_attitude_closed_form(model)
            res11, res12 = _residual(model, closed.predicted_covariance)
            self.assertLess(res11, 1e-13, (sv, su, sn, dt))
            self.assertLess(res12, 1e-12, (sv, su, sn, dt))

    def test_matches_the_library_riccati_recursion(self) -> None:
        """The library's steady_state_kalman reaches the same covariance."""
        model = GyroStarTrackerModel(1e-3, 1e-4, 1e-2, 0.5)
        closed = steady_state_attitude_closed_form(model)
        library = steady_state_attitude_riccati(model, method="iteration")
        self.assertTrue(library.converged)
        for i, j in ((0, 0), (0, 1), (1, 1)):
            self.assertAlmostEqual(
                closed.predicted_covariance[i][j] / library.predicted_covariance[i][j],
                1.0,
                delta=1e-9,
            )
            self.assertAlmostEqual(
                closed.filtered_covariance[i][j] / library.filtered_covariance[i][j],
                1.0,
                delta=1e-9,
            )
        self.assertAlmostEqual(closed.gain[0], library.gain[0], delta=1e-9)

    def test_filter_covariance_converges_to_the_closed_form(self) -> None:
        """The covariance inside kalman_filter approaches the analytic value."""
        model = GyroStarTrackerModel(1e-3, 1e-4, 1e-2, 0.5)
        closed = steady_state_attitude_closed_form(model)
        ss_model = attitude_state_space(model, initial_std=(1e-2, 1e-3))
        data = simulate_attitude_error(model, 400, seed=1, initial_std=(1e-2, 1e-3))
        run = kalman_filter(ss_model, data.measurements)
        last = run.filtered[-1].covariance
        self.assertAlmostEqual(
            last[0][0] / closed.filtered_covariance[0][0], 1.0, delta=1e-9
        )
        self.assertAlmostEqual(
            run.predicted[-1].covariance[0][0] / closed.predicted_covariance[0][0],
            1.0,
            delta=1e-9,
        )

    def test_filtered_is_below_predicted_and_scales(self) -> None:
        """Updating reduces variance; scaling all noises by c scales sigma by c."""
        base = GyroStarTrackerModel(1e-4, 1e-6, 1e-3, 0.2)
        ss = steady_state_attitude_closed_form(base)
        self.assertLess(ss.attitude_std_filtered, ss.attitude_std_predicted)
        self.assertLess(ss.attitude_std_filtered, base.sigma_n)
        # sigma_v, sigma_u, sigma_n scaled by c: every covariance scales by c^2.
        scaled = GyroStarTrackerModel(3e-4, 3e-6, 3e-3, 0.2)
        ss3 = steady_state_attitude_closed_form(scaled)
        self.assertAlmostEqual(
            ss3.attitude_std_filtered / ss.attitude_std_filtered, 3.0, delta=1e-12
        )

    def test_riccati_reports_non_convergence(self) -> None:
        """A hit iteration limit is reported, not hidden."""
        model = GyroStarTrackerModel(1e-5, 1e-9, 1e-2, 0.001)
        slow = steady_state_attitude_riccati(
            model, method="iteration", max_iterations=50
        )
        self.assertFalse(slow.converged)
        self.assertEqual(slow.iterations, 50)
        tight = steady_state_attitude_riccati(model, max_iterations=1)
        self.assertFalse(tight.converged)

    def test_invalid_input(self) -> None:
        """Bad parameters raise TypeError or ValueError."""
        with self.assertRaises(ValueError):
            GyroStarTrackerModel(0.0, 0.0, 1e-3, 0.1)
        with self.assertRaises(ValueError):
            GyroStarTrackerModel(1e-4, -1e-6, 1e-3, 0.1)
        with self.assertRaises(ValueError):
            GyroStarTrackerModel(1e-4, 1e-6, 0.0, 0.1)
        with self.assertRaises(ValueError):
            GyroStarTrackerModel(1e-4, 1e-6, 1e-3, math.nan)
        with self.assertRaises(TypeError):
            GyroStarTrackerModel("a", 1e-6, 1e-3, 0.1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            steady_state_attitude_closed_form(object())  # type: ignore[arg-type]
        model = GyroStarTrackerModel(1e-4, 1e-6, 1e-3, 0.1)
        with self.assertRaises(ValueError):
            steady_state_attitude_riccati(model, method="newton")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            steady_state_attitude_riccati(model, tolerance=0.0)


class FilterConsistencyTests(unittest.TestCase):
    """NEES and NIS of the filter on simulated runs."""

    def _runs(self, runs: int, steps: int) -> tuple[list, list, list]:
        model = GyroStarTrackerModel(2e-4, 5e-6, 5e-4, 0.5)
        prior = (2e-3, 2e-5)
        ss = attitude_state_space(model, initial_std=prior)
        results, truths, nis = [], [], []
        for run in range(runs):
            data = simulate_attitude_error(
                model, steps, seed=100 + run, initial_std=prior
            )
            result = kalman_filter(ss, data.measurements)
            results.append(result)
            truths.append(data.truth)
        return results, truths, nis

    def test_nees_is_consistent(self) -> None:
        """Average NEES over 60 runs stays inside the chi-square band (dim 2)."""
        results, truths, _ = self._runs(60, 120)
        test = nees_test_from_filters(results, truths, confidence=0.99)
        self.assertEqual(test.dimension, 2)
        self.assertEqual(test.verdict, "consistent")
        # The mean NEES estimates the state dimension 2 with standard error
        # sqrt(2 * 2 / (60 * 120)) when steps are independent; allow 5x for
        # the positive dependence between neighbouring steps.
        self.assertAlmostEqual(test.mean_statistic, 2.0, delta=5 * math.sqrt(4 / 7200))

    def test_nis_is_consistent(self) -> None:
        """Innovations of a long run are chi-square(1) on average."""
        model = GyroStarTrackerModel(2e-4, 5e-6, 5e-4, 0.5)
        ss = attitude_state_space(model, initial_std=(2e-3, 2e-5))
        data = simulate_attitude_error(model, 3000, seed=7, initial_std=(2e-3, 2e-5))
        result = kalman_filter(ss, data.measurements)
        test = nis_test_from_filter(result, confidence=0.99)
        self.assertEqual(test.verdict, "consistent")

    def test_optimistic_filter_is_detected(self) -> None:
        """A filter with half the true process noise fails the NEES test."""
        truth_model = GyroStarTrackerModel(4e-4, 1e-5, 5e-4, 0.5)
        wrong = GyroStarTrackerModel(2e-4, 5e-6, 5e-4, 0.5)
        prior = (2e-3, 2e-5)
        ss = attitude_state_space(wrong, initial_std=prior)
        results, truths = [], []
        for run in range(40):
            data = simulate_attitude_error(
                truth_model, 150, seed=900 + run, initial_std=prior
            )
            results.append(kalman_filter(ss, data.measurements))
            truths.append(data.truth)
        test = nees_test_from_filters(results, truths, confidence=0.99)
        self.assertEqual(test.verdict, "optimistic")

    def test_simulation_is_reproducible_and_validated(self) -> None:
        """The same seed gives the same data; bad arguments raise."""
        model = GyroStarTrackerModel(2e-4, 5e-6, 5e-4, 0.5)
        a = simulate_attitude_error(model, 10, seed=3)
        b = simulate_attitude_error(model, 10, seed=3)
        c = simulate_attitude_error(model, 10, seed=4)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual((len(a.truth), len(a.measurements)), (11, 10))
        with self.assertRaises(ValueError):
            simulate_attitude_error(model, 0, seed=1)
        with self.assertRaises(ValueError):
            simulate_attitude_error(model, 5, seed=1, initial_std=(0.0, 1.0))


class AllanTests(unittest.TestCase):
    """Overlapping Allan deviation and the two-component fit."""

    DT = 0.1
    SIGMA_V = 2e-4
    SIGMA_U = 3e-6

    def test_estimator_on_a_hand_computed_series(self) -> None:
        """Cluster size 1 on [0, 1, 0, 1, 0]: all differences are +-1, AVAR = 1/2."""
        result = overlapping_allan_deviation(
            [0.0, 1.0, 0.0, 1.0, 0.0], 1.0, cluster_sizes=[1]
        )
        # Differences of means: 1, -1, 1, -1 (4 terms of 1) -> (1/(2*4)) * 4.
        self.assertAlmostEqual(result.variances[0], 0.5, places=14)
        self.assertEqual(result.terms[0], 4)
        # m = 2: means over (0,1)=.5,(1,0)=.5,(0,1)=.5,(1,0)=.5; y_{j+2}-y_j = 0.
        # Use a ramp instead: g_k = k has ybar_{j+m}-ybar_j = m, AVAR = m^2 / 2.
        ramp = overlapping_allan_deviation(
            [float(k) for k in range(40)], 1.0, cluster_sizes=[1, 3, 5]
        )
        for m, var in zip((1, 3, 5), ramp.variances, strict=True):
            self.assertAlmostEqual(var, m * m / 2.0, delta=1e-10)

    def test_constant_offset_does_not_matter(self) -> None:
        """Adding a constant leaves the Allan variance unchanged."""
        data = simulate_gyro_rate(self.SIGMA_V, 0.0, self.DT, 2000, seed=5)
        shifted = [x + 3.0 for x in data]
        a = overlapping_allan_deviation(data, self.DT)
        b = overlapping_allan_deviation(shifted, self.DT)
        for x, y in zip(a.variances, b.variances, strict=True):
            self.assertAlmostEqual(x / y, 1.0, delta=1e-9)

    def test_white_noise_has_slope_minus_half(self) -> None:
        """Pure white rate noise: AVAR = sigma_v^2 / tau at every tau."""
        n = 40000
        data = simulate_gyro_rate(self.SIGMA_V, 0.0, self.DT, n, seed=11)
        result = overlapping_allan_deviation(data, self.DT)
        for m, tau, var in zip(
            result.cluster_sizes, result.taus, result.variances, strict=True
        ):
            # Relative error of an Allan variance estimate ~ sqrt(m / N) times
            # O(1); allow 6x of that, floor 3 %.
            tol = max(0.03, 6.0 * math.sqrt(m / n))
            self.assertAlmostEqual(var * tau / self.SIGMA_V**2, 1.0, delta=tol)
        slope = math.log(result.deviations[-1] / result.deviations[0]) / math.log(
            result.taus[-1] / result.taus[0]
        )
        self.assertAlmostEqual(slope, -0.5, delta=0.1)

    def test_model_variance_matches_a_monte_carlo_of_the_walk(self) -> None:
        """The discrete random-walk Allan formula, averaged over many series."""
        sigma_u = 1e-3
        m = 4
        acc = 0.0
        reps = 400
        for rep in range(reps):
            data = simulate_gyro_rate(0.0, sigma_u, 1.0, 2 * m + 1, seed=rep)
            res = overlapping_allan_deviation(data, 1.0, cluster_sizes=[m])
            acc += res.variances[0]
        mean = acc / reps
        expected = allan_model_variance(0.0, sigma_u, 1.0, m)
        self.assertAlmostEqual(mean / expected, 1.0, delta=0.2)
        # Large-m limit sigma_u^2 tau / 3 within the O(1/m^2) correction.
        big = allan_model_variance(0.0, sigma_u, 0.1, 1000)
        self.assertAlmostEqual(big / (sigma_u**2 * 100.0 / 3.0), 1.0, delta=1e-6)

    def _fit(self, seed: int, n: int = 60000):
        data = simulate_gyro_rate(self.SIGMA_V, self.SIGMA_U, self.DT, n, seed=seed)
        return fit_allan_noise(overlapping_allan_deviation(data, self.DT))

    def test_recovers_both_parameters(self) -> None:
        """sigma_v and sigma_u are recovered within 4 reported standard errors."""
        fit = self._fit(21)
        self.assertTrue(fit.converged, fit.stop_reason)
        self.assertEqual(fit.stop_reason, "ok")
        self.assertAlmostEqual(fit.sigma_v, self.SIGMA_V, delta=4 * fit.sigma_v_error)
        self.assertAlmostEqual(fit.sigma_u, self.SIGMA_U, delta=4 * fit.sigma_u_error)
        # And within 10 % / 25 % outright.
        self.assertAlmostEqual(fit.sigma_v / self.SIGMA_V, 1.0, delta=0.10)
        self.assertAlmostEqual(fit.sigma_u / self.SIGMA_U, 1.0, delta=0.25)
        self.assertLess(fit.residual_rms, 0.3)

    def test_slopes_in_the_two_regions(self) -> None:
        """The deviation falls (slope -1/2) then rises (slope +1/2) around a minimum.

        At the longest cluster sizes few clusters remain, so the local slope
        there is noisy; the rise is checked as a ratio against the minimum
        (the model predicts about 2x) and the whole curve against the model.
        """
        n = 60000
        data = simulate_gyro_rate(self.SIGMA_V, self.SIGMA_U, self.DT, n, seed=21)
        result = overlapping_allan_deviation(data, self.DT)
        dev, tau = result.deviations, result.taus
        short = math.log(dev[1] / dev[0]) / math.log(tau[1] / tau[0])
        self.assertLess(short, -0.4)
        low = min(range(len(dev)), key=lambda i: dev[i])
        self.assertGreater(low, 3)
        self.assertLess(low, len(dev) - 1)
        self.assertGreater(dev[-1], 1.5 * dev[low])
        for m, var in zip(result.cluster_sizes, result.variances, strict=True):
            model = allan_model_variance(self.SIGMA_V, self.SIGMA_U, self.DT, m)
            self.assertAlmostEqual(
                var / model, 1.0, delta=max(0.05, 6.0 * math.sqrt(m / n))
            )

    def test_reported_errors_are_calibrated_across_seeds(self) -> None:
        """Seed-to-seed scatter is of the order of the reported standard errors."""
        n_seeds = 6 if not SLOW else 40
        zs_v, zs_u = [], []
        for seed in range(n_seeds):
            fit = self._fit(300 + seed, n=40000)
            self.assertTrue(fit.converged)
            zs_v.append((fit.sigma_v - self.SIGMA_V) / fit.sigma_v_error)
            zs_u.append((fit.sigma_u - self.SIGMA_U) / fit.sigma_u_error)
        for zs in (zs_v, zs_u):
            rms = math.sqrt(sum(z * z for z in zs) / len(zs))
            self.assertLess(max(abs(z) for z in zs), 5.0)
            self.assertGreater(rms, 0.2)
            self.assertLess(rms, 2.5)

    def test_white_only_series_reports_unidentified_walk(self) -> None:
        """No rate random walk in the data: sigma_u is not silently invented."""
        data = simulate_gyro_rate(self.SIGMA_V, 0.0, self.DT, 40000, seed=2)
        fit = fit_allan_noise(overlapping_allan_deviation(data, self.DT))
        self.assertAlmostEqual(fit.sigma_v / self.SIGMA_V, 1.0, delta=0.1)
        # Either the walk is reported as not identified, or it is tiny.
        if fit.converged:
            self.assertLess(fit.sigma_u, 0.3 * self.SIGMA_U)
        else:
            self.assertEqual(fit.stop_reason, "rate_random_walk_not_identified")
            self.assertEqual(fit.sigma_u, 0.0)

    def test_failure_cases(self) -> None:
        """Short, constant, non-finite or badly specified inputs raise."""
        with self.assertRaises(ValueError):
            overlapping_allan_deviation([1.0, 2.0, 3.0], 1.0)
        with self.assertRaises(ValueError):
            overlapping_allan_deviation([1.0] * 100, 1.0)
        with self.assertRaises(ValueError):
            overlapping_allan_deviation([1.0, math.nan, 2.0, 3.0, 4.0], 1.0)
        with self.assertRaises(ValueError):
            overlapping_allan_deviation([0.0, 1.0] * 10, 1.0, cluster_sizes=[11])
        with self.assertRaises(ValueError):
            overlapping_allan_deviation([0.0, 1.0] * 10, -1.0)
        with self.assertRaises(ValueError):
            overlapping_allan_deviation([0.0, 1.0] * 10, 1.0, cluster_sizes=[])
        with self.assertRaises(TypeError):
            fit_allan_noise(object())  # type: ignore[arg-type]
        short = overlapping_allan_deviation(
            simulate_gyro_rate(1e-3, 1e-5, 1.0, 100, seed=1), 1.0, cluster_sizes=[1, 2]
        )
        with self.assertRaises(ValueError):
            fit_allan_noise(short)
        with self.assertRaises(ValueError):
            simulate_gyro_rate(-1.0, 0.0, 1.0, 10, seed=1)


class PointingBudgetTests(unittest.TestCase):
    """RSS, correlated propagation, Monte Carlo and requirement probability."""

    MODEL = GyroStarTrackerModel(2e-5, 1e-7, 1e-4, 0.1)

    def _budget(self, **kwargs):
        defaults = dict(
            control_jitter=2.0 * ARCSEC,
            thermal_elastic=3.0 * ARCSEC,
            tracker_alignment=2.0 * ARCSEC,
            requirement_3sigma=30.0 * ARCSEC,
            size=20000,
            seed=3,
        )
        defaults.update(kwargs)
        return pointing_budget(self.MODEL, **defaults)

    def test_independent_budget_is_rss(self) -> None:
        """Without correlation: law of propagation = RSS = closed-form sum."""
        result = self._budget()
        knowledge = steady_state_attitude_closed_form(self.MODEL).attitude_std_filtered
        expected = math.sqrt(
            knowledge**2
            + (2.0 * ARCSEC) ** 2
            + (3.0 * ARCSEC) ** 2
            + (2.0 * ARCSEC) ** 2
        )
        self.assertAlmostEqual(result.rss, expected, delta=1e-15)
        self.assertAlmostEqual(result.combined, result.rss, delta=1e-15)
        self.assertEqual(result.budget.names, POINTING_SOURCES)
        self.assertAlmostEqual(result.budget.entries[0].u, knowledge, delta=1e-18)

    def test_monte_carlo_agrees_with_rss_and_gaussian_probability(self) -> None:
        """MC std and P(|e| <= req) match the analytic values within MCSE bands."""
        result = self._budget(requirement_3sigma=2.0 * 3.0 * 2.2 * ARCSEC)
        n = result.mc_size
        # Standard deviation: relative error of a sample sd is ~ 1/sqrt(2 n).
        self.assertAlmostEqual(
            result.mc_standard_uncertainty / result.combined,
            1.0,
            delta=5.0 / math.sqrt(2.0 * n),
        )
        self.assertAlmostEqual(
            result.mc_probability_within_requirement,
            result.probability_within_requirement,
            delta=5.0 * result.mc_probability_error + 1e-3,
        )
        # RSS 3-sigma interval covers 99.73 % when nothing is correlated.
        self.assertAlmostEqual(
            result.rss_coverage_mc,
            result.rss_coverage_gaussian,
            delta=5.0 * math.sqrt(0.0027 * 0.9973 / n) + 1e-4,
        )

    def test_positive_correlation_inflates_the_total(self) -> None:
        """Correlated thermal and alignment errors inflate u_c; RSS undercovers."""
        pair = {("thermal_elastic", "tracker_alignment"): 0.9}
        result = self._budget(correlations=pair, size=60000)
        self.assertGreater(result.combined, result.rss)
        expected = math.sqrt(result.rss**2 + 2 * 0.9 * (3.0 * ARCSEC) * (2.0 * ARCSEC))
        self.assertAlmostEqual(result.combined, expected, delta=1e-15)
        self.assertAlmostEqual(
            result.mc_standard_uncertainty / result.combined,
            1.0,
            delta=5.0 / math.sqrt(2.0 * result.mc_size),
        )
        # Ignoring the correlation makes the nominal 3-sigma interval too narrow.
        self.assertLess(result.rss_coverage_mc, result.rss_coverage_gaussian)
        # The Gaussian coverage of that interval for the true u_c is analytic.
        analytic = math.erf(3.0 * result.rss / (math.sqrt(2.0) * result.combined))
        self.assertAlmostEqual(
            result.rss_coverage_mc,
            analytic,
            delta=5.0 * math.sqrt(analytic * (1 - analytic) / result.mc_size) + 1e-4,
        )

    def test_negative_correlation_can_cancel(self) -> None:
        """Opposite-sign correlation lowers the total below the RSS."""
        result = self._budget(
            correlations={("thermal_elastic", "tracker_alignment"): -0.8}
        )
        self.assertLess(result.combined, result.rss)

    def test_requirement_flag_and_margin(self) -> None:
        """meets_requirement is 3 u_c <= requirement; margin is the ratio."""
        loose = self._budget(requirement_3sigma=100.0 * ARCSEC)
        tight = self._budget(requirement_3sigma=3.0 * ARCSEC)
        self.assertTrue(loose.meets_requirement)
        self.assertGreater(loose.margin, 1.0)
        self.assertFalse(tight.meets_requirement)
        self.assertLess(tight.margin, 1.0)
        self.assertAlmostEqual(loose.margin, loose.requirement / (3 * loose.combined))
        # A requirement exactly at 3 u_c is met with probability erf(3/sqrt 2).
        exact = self._budget(requirement_3sigma=3.0 * loose.combined)
        self.assertAlmostEqual(
            exact.probability_within_requirement,
            math.erf(3 / math.sqrt(2)),
            delta=1e-12,
        )
        self.assertTrue(exact.meets_requirement)

    def test_budget_failure_cases(self) -> None:
        """Invalid sources, correlations and sizes are rejected."""
        with self.assertRaises(ValueError):
            self._budget(control_jitter=0.0)
        with self.assertRaises(ValueError):
            self._budget(requirement_3sigma=-1.0)
        with self.assertRaises(ValueError):
            self._budget(size=10)
        with self.assertRaises(ValueError):
            self._budget(correlations={("thermal_elastic", "bogus"): 0.1})
        with self.assertRaises(ValueError):
            self._budget(correlations={("thermal_elastic", "thermal_elastic"): 0.1})
        with self.assertRaises(ValueError):
            self._budget(correlations={("thermal_elastic", "tracker_alignment"): 1.5})
        with self.assertRaises(ValueError):
            self._budget(
                correlations={
                    ("thermal_elastic", "tracker_alignment"): 0.1,
                    ("tracker_alignment", "thermal_elastic"): 0.2,
                }
            )
        # Pairwise correlations that jointly are not a valid correlation matrix.
        with self.assertRaises(ValueError):
            self._budget(
                correlations={
                    ("attitude_knowledge", "control_jitter"): 0.95,
                    ("control_jitter", "thermal_elastic"): 0.95,
                    ("attitude_knowledge", "thermal_elastic"): -0.95,
                }
            )


class GradeTests(unittest.TestCase):
    """Sobol ranking and the requirement probability over sensor grades."""

    def test_sobol_matches_the_analytic_indices_in_the_pure_integrator_regime(
        self,
    ) -> None:
        """With sigma_u negligible, std ~ (sigma_v sigma_n)^(1/2) dt^(1/4).

        For independent log-uniform factors the indices of f = sqrt(X) sqrt(Y)
        follow from E[X^s] = (b^s - a^s) / (s ln(b/a)).
        """
        sv = (1e-5, 1e-3)
        sn = (1e-4, 1e-2)
        grades = SensorGrades(sigma_v=sv, sigma_u=(1e-14, 2e-14), sigma_n=sn)
        result = knowledge_sobol_ranking(grades, 0.01, size=3000, seed=4)

        def moment(rng: tuple[float, float], s: float) -> float:
            return (rng[1] ** s - rng[0] ** s) / (s * math.log(rng[1] / rng[0]))

        ex, ey = moment(sv, 0.5), moment(sn, 0.5)
        vx = moment(sv, 1.0) - ex**2
        vy = moment(sn, 1.0) - ey**2
        total = moment(sv, 1.0) * moment(sn, 1.0) - (ex * ey) ** 2
        s_v = vx * ey**2 / total
        s_n = vy * ex**2 / total
        # Estimator error with N = 3000 rows is a few 0.01 (checked by the
        # slow tier over seeds); the band is 0.07.
        self.assertAlmostEqual(result.first_order[0], s_v, delta=0.07)
        self.assertAlmostEqual(result.first_order[2], s_n, delta=0.07)
        self.assertLess(abs(result.first_order[1]), 0.02)
        self.assertLess(result.total[1], 0.02)
        # The wider-range factor ranks first (sigma_v and sigma_n have equal
        # width of range here, so only the sigma_u rank is asserted).
        self.assertEqual(result.ranking[-1], "sigma_u")
        self.assertGreaterEqual(result.evaluations, 3000 * 5)
        for first, tot in zip(result.first_order, result.total, strict=True):
            self.assertLessEqual(first, tot + 0.05)

    def test_bias_noise_dominates_for_a_long_step_and_wide_walk_range(self) -> None:
        """A wide sigma_u range with a long step puts sigma_u first."""
        grades = SensorGrades(
            sigma_v=(1e-5, 2e-5), sigma_u=(1e-8, 1e-3), sigma_n=(1e-4, 2e-4)
        )
        result = knowledge_sobol_ranking(grades, 10.0, size=2000, seed=9)
        self.assertEqual(result.ranking[0], "sigma_u")
        self.assertGreater(result.total[1], 0.5)

    def test_requirement_probability_orders_with_the_requirement(self) -> None:
        """Looser requirements are met with higher probability; extremes are 0 and 1."""
        grades = SensorGrades(
            sigma_v=(1e-5, 1e-3), sigma_u=(1e-8, 1e-5), sigma_n=(1e-4, 1e-3)
        )
        common = dict(
            control_jitter=2.0 * ARCSEC,
            thermal_elastic=2.0 * ARCSEC,
            tracker_alignment=2.0 * ARCSEC,
            size=800,
            seed=5,
        )
        previous = -1.0
        for req in (5.0, 20.0, 60.0, 200.0, 2000.0):
            res = grade_requirement_probability(
                grades, 0.1, requirement_3sigma=req * ARCSEC, **common
            )
            self.assertGreaterEqual(res.probability, previous)
            previous = res.probability
            self.assertGreater(res.mean_combined, 0.0)
        self.assertEqual(previous, 1.0)
        tiny = grade_requirement_probability(
            grades, 0.1, requirement_3sigma=1.0 * ARCSEC, **common
        )
        self.assertEqual(tiny.probability, 0.0)

    def test_grade_failure_cases(self) -> None:
        """Invalid ranges, sizes and types raise."""
        with self.assertRaises(ValueError):
            SensorGrades((1e-3, 1e-4), (1e-6, 1e-5), (1e-4, 1e-3))
        with self.assertRaises(ValueError):
            SensorGrades((0.0, 1e-4), (1e-6, 1e-5), (1e-4, 1e-3))
        grades = SensorGrades((1e-5, 1e-3), (1e-8, 1e-5), (1e-4, 1e-3))
        with self.assertRaises(ValueError):
            knowledge_sobol_ranking(grades, 0.1, size=10, seed=1)
        with self.assertRaises(TypeError):
            knowledge_sobol_ranking(object(), 0.1, size=100, seed=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            grade_requirement_probability(
                grades,
                0.1,
                control_jitter=1e-6,
                thermal_elastic=1e-6,
                tracker_alignment=1e-6,
                requirement_3sigma=0.0,
                size=200,
                seed=1,
            )


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowAttitudeTests(unittest.TestCase):
    """Larger studies: NEES with many runs and the Allan fit calibration."""

    def test_nees_with_many_runs(self) -> None:
        """300 runs of 200 steps: mean NEES within 4 sigma of 2."""
        model = GyroStarTrackerModel(2e-4, 5e-6, 5e-4, 0.5)
        prior = (2e-3, 2e-5)
        ss = attitude_state_space(model, initial_std=prior)
        results, truths = [], []
        for run in range(300):
            data = simulate_attitude_error(
                model, 200, seed=5000 + run, initial_std=prior
            )
            results.append(kalman_filter(ss, data.measurements))
            truths.append(data.truth)
        test = nees_test_from_filters(results, truths, confidence=0.99)
        self.assertEqual(test.verdict, "consistent")


if __name__ == "__main__":
    unittest.main()
