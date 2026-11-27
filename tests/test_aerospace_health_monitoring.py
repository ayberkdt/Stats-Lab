"""Tests for the hybrid physics-plus-statistics engine health-monitoring twin.

All numbers are synthetic and illustrative; none comes from a real engine.
Closed forms first (steady-state pressure, exact lag decay, the linear special
case against the exact Kalman filter, the inverse-Gaussian cdf against numerical
integration of its density), then independent oracles (a numerical CUSUM
run-length solver, Brownian-bridge Monte Carlo of the first-passage time), then
the Monte-Carlo workflows (filter consistency, detection delay against fault
size, RUL coverage), and finally the failure cases. Stochastic tests use fixed
seeds and bands from the Monte-Carlo standard error.
"""

from __future__ import annotations

import math
import os
import unittest

from probstat.aerospace.health_monitoring import (
    DISCRETE_MONITORING_SHIFT,
    EnginePhmSpec,
    StepFault,
    assess_engine_health,
    augmented_model,
    cusum_arl_numerical,
    cusum_arl_siegmund,
    cusum_monitor,
    detection_operating_characteristic,
    estimate_health,
    filter_consistency_study,
    glr_false_alarm_bound,
    glr_mean_shift,
    inverse_gaussian_cdf,
    linear_as_nonlinear,
    linear_special_case,
    normalized_innovations,
    predict_rul,
    rul_calibration,
    rul_coverage_study,
    sample_inverse_gaussian,
    simulate_plant,
    wiener_first_passage_cdf,
    wiener_first_passage_quantile,
)
from probstat.aerospace.liquid_engine import characteristic_velocity
from probstat.filtering import (
    GaussianState,
    kalman_filter,
    simulate,
)
from probstat.foundations.random_streams import resolve_stream

SLOW = os.environ.get("PROBSTAT_SLOW") == "1"


class PlantClosedFormTests(unittest.TestCase):
    """Physics of the synthetic chamber against closed forms."""

    def test_reference_pressure_is_choked_flow_relation(self) -> None:
        """``p_ref = c* mdot / A_t`` with ``c*`` from the liquid-engine module."""
        spec = EnginePhmSpec()
        c_star = characteristic_velocity(1.2, 22.0, 3400.0)
        self.assertAlmostEqual(spec.ideal_characteristic_velocity, c_star)
        self.assertAlmostEqual(
            spec.reference_pressure, c_star * 20.0 / 0.01, delta=1e-6 * c_star
        )
        self.assertAlmostEqual(
            spec.pressure_pa(0.9), 0.9 * c_star * 20.0 / 0.01, delta=1.0
        )

    def test_noise_free_lag_decays_exactly_to_eta_times_flow(self) -> None:
        """With ``m = 1`` and no noise, ``p_k - eta = (p_0 - eta)(1 - a)^k``."""
        spec = EnginePhmSpec(initial_eta=0.9, initial_rate=0.0)
        from probstat.aerospace import health_monitoring as hm

        state: tuple[float, ...] = (1.0, 1.0, 0.9, 0.0)
        a = spec.pressure_coefficient
        self.assertAlmostEqual(a, 1.0 - math.exp(-0.5), places=14)
        for k in range(1, 9):
            state = hm._transition(spec, state)
            self.assertAlmostEqual(
                state[0] - 0.9, (1.0 - 0.9) * (1.0 - a) ** k, places=13
            )
        self.assertAlmostEqual(state[1], 1.0, places=14)

    def test_jacobian_matches_central_differences(self) -> None:
        """Analytic ``df/dx`` of the augmented model equals finite differences."""
        spec = EnginePhmSpec()
        model = augmented_model(spec)
        x = (0.93, 1.02, 0.91, -2e-4)
        jac = model.transition_jacobian(x)
        h = 1e-6
        for j in range(4):
            up = list(x)
            dn = list(x)
            up[j] += h
            dn[j] -= h
            fu = model.transition(tuple(up), None)
            fd = model.transition(tuple(dn), None)
            for i in range(4):
                self.assertAlmostEqual(jac[i][j], (fu[i] - fd[i]) / (2 * h), places=8)

    def test_simulation_is_reproducible_and_fault_only_changes_eta(self) -> None:
        """Common random numbers: a fault run differs from the null run by the step."""
        spec = EnginePhmSpec()
        a = simulate_plant(spec, 40, seed=7)
        b = simulate_plant(spec, 40, seed=7)
        self.assertEqual(a.states, b.states)
        f = simulate_plant(spec, 40, seed=7, fault=StepFault(20, -0.03))
        self.assertEqual(a.observations[:19], f.observations[:19])
        self.assertAlmostEqual(f.states[20][2] - a.states[20][2], -0.03, places=12)
        self.assertAlmostEqual(f.states[20][3], a.states[20][3], places=14)


class LinearSpecialCaseTests(unittest.TestCase):
    """EKF and UKF reduce to the exact Kalman filter when the model is linear."""

    def test_ekf_and_ukf_equal_kalman_filter(self) -> None:
        """Means, covariances and log-likelihood agree to rounding."""
        spec = EnginePhmSpec()
        linear = linear_special_case(spec)
        data = simulate(linear, 60, seed=11).observations
        exact = kalman_filter(linear, data)
        wrapped = linear_as_nonlinear(linear)
        from probstat.filtering import (
            extended_kalman_filter,
            unscented_kalman_filter,
        )

        ekf = extended_kalman_filter(wrapped, data)
        ukf = unscented_kalman_filter(wrapped, data)
        for other, tol in ((ekf, 1e-11), (ukf, 1e-9)):
            worst = 0.0
            for s_ref, s in zip(exact.filtered, other.filtered, strict=True):
                for i in range(3):
                    worst = max(worst, abs(s_ref.mean[i] - s.mean[i]))
                    for j in range(3):
                        worst = max(
                            worst, abs(s_ref.covariance[i][j] - s.covariance[i][j])
                        )
            self.assertLess(worst, tol)
            self.assertAlmostEqual(
                other.log_likelihood, exact.log_likelihood, delta=1e-8
            )

    def test_wrapper_rejects_non_linear_models(self) -> None:
        """``linear_as_nonlinear`` demands a linear-Gaussian model."""
        with self.assertRaises(TypeError):
            linear_as_nonlinear(augmented_model(EnginePhmSpec()))  # type: ignore[arg-type]


class AugmentedEstimationTests(unittest.TestCase):
    """The joint state-parameter filter on the nonlinear model."""

    def test_efficiency_is_recovered_and_methods_agree(self) -> None:
        """Final ``eta`` error is within 4 filtered sigma; EKF and UKF agree."""
        spec = EnginePhmSpec()
        sim = simulate_plant(spec, 150, seed=3)
        ekf = estimate_health(spec, sim.observations)
        ukf = estimate_health(spec, sim.observations, method="ukf")
        last = ekf.filtered[-1]
        sd = last.standard_deviations[2]
        self.assertLess(abs(last.mean[2] - sim.states[-1][2]), 4.0 * sd)
        self.assertLess(abs(last.mean[2] - ukf.filtered[-1].mean[2]), 0.5 * sd)
        self.assertEqual(ekf.method, "extended")
        self.assertLess(sd, 0.005)

    def test_consistency_fast(self) -> None:
        """NEES average inside its chi-square band; NIS mean near 2 (two sensors)."""
        study = filter_consistency_study(EnginePhmSpec(), runs=30, steps=100, seed=0)
        self.assertEqual(study.nees.verdict, "consistent")
        self.assertEqual(study.nees.degrees_of_freedom, 30 * 4)
        self.assertGreater(study.mean_nis, 1.7)
        self.assertLess(study.mean_nis, 2.3)
        self.assertGreaterEqual(study.nis_consistent_fraction, 0.8)

    def test_ukf_consistency_fast(self) -> None:
        """The UKF is also inside the NEES band on a small set of runs."""
        study = filter_consistency_study(
            EnginePhmSpec(), runs=12, steps=60, method="ukf", seed=1
        )
        self.assertEqual(study.nees.verdict, "consistent")

    @unittest.skipUnless(SLOW, "slow tier")
    def test_consistency_slow(self) -> None:
        """200 runs: the time-mean NEES is inside the band for EKF and UKF."""
        for method in ("ekf", "ukf"):
            study = filter_consistency_study(
                EnginePhmSpec(), runs=200, steps=150, method=method, seed=5
            )
            print(
                f"[slow] {method}: NEES mean {study.nees.mean_statistic:.3f} in "
                f"[{study.nees.lower:.3f}, {study.nees.upper:.3f}], "
                f"NIS ok {study.nis_consistent_fraction:.3f}"
            )
            self.assertEqual(study.nees.verdict, "consistent")


class CusumTests(unittest.TestCase):
    """CUSUM recursion, average run length, and its Monte-Carlo check."""

    def test_recursion_by_hand(self) -> None:
        """Hand-computed statistics and alarm for a short series."""
        out = cusum_monitor([0.0, 1.0, 1.0, 1.0, -3.0], reference=0.5, threshold=1.2)
        self.assertEqual(out.upper, (0.0, 0.5, 1.0, 1.5, 0.0))
        self.assertEqual(out.lower, (0.0, 0.0, 0.0, 0.0, 2.5))
        self.assertEqual((out.alarm_index, out.alarm_side), (3, "upper"))
        down = cusum_monitor([-2.0, -2.0], reference=0.5, threshold=2.0)
        self.assertEqual((down.alarm_index, down.alarm_side), (1, "lower"))
        self.assertIsNone(cusum_monitor([0.1, -0.1], threshold=5.0).alarm_index)

    def test_siegmund_matches_numerical_solution(self) -> None:
        """Siegmund vs the integral-equation solution: 3 % in control, 5 % out.

        Measured out-of-control error is 3.6 % at ``mu = 2`` (ARL about 3, where
        the Brownian approximation is weakest), hence the looser band there.
        """
        for k, h, mu in (
            (0.5, 4.0, 0.0),
            (0.5, 5.0, 0.0),
            (0.75, 3.0, 0.0),
            (0.5, 4.0, 1.0),
            (0.5, 4.0, 2.0),
        ):
            exact = cusum_arl_numerical(k, h, shift=mu, cells=120)
            approx = cusum_arl_siegmund(k, h, shift=mu)
            tol = 0.03 if mu == 0.0 else 0.05
            self.assertLess(abs(approx / exact - 1.0), tol, (k, h, mu, approx, exact))

    def test_numerical_solver_converges_and_has_wald_limit(self) -> None:
        """Doubling the cells changes ARL by < 0.2 %; large shift gives ~ b / Delta."""
        coarse = cusum_arl_numerical(0.5, 4.0, cells=100)
        fine = cusum_arl_numerical(0.5, 4.0, cells=200)
        self.assertLess(abs(coarse / fine - 1.0), 2e-3)
        # Out of control with a large shift the walk drifts at Delta = 3.5 per step.
        arl = cusum_arl_numerical(0.5, 4.0, shift=4.0, cells=100)
        self.assertAlmostEqual(arl, 4.0 / 3.5 + 0.5, delta=0.7)

    def test_two_sided_formula_is_harmonic_combination(self) -> None:
        """In control the two-sided ARL is half of the one-sided one."""
        one = cusum_arl_siegmund(0.5, 4.0)
        two = cusum_arl_siegmund(0.5, 4.0, two_sided=True)
        self.assertAlmostEqual(two, one / 2.0, places=10)

    def test_in_control_arl_against_monte_carlo(self) -> None:
        """Measured ARL0 on iid N(0, 1) vs the two stated predictions.

        1200 runs of mean length about 170 give a relative standard error of
        about 3 %; the tolerance is 9 % (three standard errors).
        """
        k, h, runs = 0.5, 4.0, 1200
        gen = resolve_stream(2024).generator()
        lengths = []
        for _ in range(runs):
            up = lo = 0.0
            n = 0
            while up <= h and lo <= h:
                z = gen.gauss(0.0, 1.0)
                up = max(0.0, up + z - k)
                lo = max(0.0, lo - z - k)
                n += 1
            lengths.append(n)
        mean = sum(lengths) / runs
        exact_one = cusum_arl_numerical(k, h, cells=200)
        numerical_two = exact_one / 2.0
        siegmund_two = cusum_arl_siegmund(k, h, two_sided=True)
        self.assertLess(abs(mean / numerical_two - 1.0), 0.09, (mean, numerical_two))
        self.assertLess(abs(mean / siegmund_two - 1.0), 0.10, (mean, siegmund_two))

    def test_out_of_control_arl_against_monte_carlo(self) -> None:
        """Detection delay of a unit shift: MC mean run length vs numeric ARL."""
        k, h, runs = 0.5, 4.0, 1500
        gen = resolve_stream(77).generator()
        total = 0
        for _ in range(runs):
            up = 0.0
            n = 0
            while up <= h:
                up = max(0.0, up + gen.gauss(1.0, 1.0) - k)
                n += 1
            total += n
        exact = cusum_arl_numerical(k, h, shift=1.0, cells=200)
        self.assertLess(abs(total / runs / exact - 1.0), 0.05, (total / runs, exact))


class GlrTests(unittest.TestCase):
    """Windowed GLR step-change test."""

    def test_noise_free_step_is_located(self) -> None:
        """A clean step of size 3 starting at index 10 is found exactly."""
        z = [0.0] * 10 + [3.0] * 6
        out = glr_mean_shift(z, threshold=4.0, window=20)
        # With L ones at 3: (3 L)^2 / (2 L) = 4.5 L > 4 first at L = 1.
        self.assertEqual(out.alarm_index, 10)
        self.assertEqual(out.change_index, 10)
        self.assertAlmostEqual(out.magnitude or 0.0, 3.0)
        self.assertAlmostEqual(out.statistic[10], 4.5)
        self.assertAlmostEqual(out.statistic[12], 13.5)

    def test_statistic_by_hand(self) -> None:
        """``g_n`` for a short series equals the explicit maximum over L."""
        z = [1.0, -1.0, 2.0]
        out = glr_mean_shift(z, threshold=100.0, window=2)
        self.assertIsNone(out.alarm_index)
        self.assertAlmostEqual(out.statistic[0], 0.5)
        self.assertAlmostEqual(out.statistic[1], max(0.5, 0.0))
        self.assertAlmostEqual(out.statistic[2], max(4.0 / 2.0, 1.0 / 4.0))

    def test_false_alarms_below_union_bound(self) -> None:
        """Measured false-alarm probability on iid noise is below the bound."""
        gen = resolve_stream(31).generator()
        runs, steps, window, h = 400, 100, 20, 7.0
        bad = sum(
            glr_mean_shift(
                [gen.gauss(0.0, 1.0) for _ in range(steps)], threshold=h, window=window
            ).alarm_index
            is not None
            for _ in range(runs)
        )
        bound = glr_false_alarm_bound(h, window, steps)
        self.assertLess(bound, 0.5)
        se = math.sqrt(bound * (1.0 - bound) / runs)
        self.assertLessEqual(bad / runs, bound + 3.0 * se)


class DetectionStudyTests(unittest.TestCase):
    """Operating characteristic of the detectors on filter innovations."""

    def test_delay_decreases_with_fault_size_and_far_with_threshold(self) -> None:
        """Mean delay falls as the fault grows; false alarms fall as h grows."""
        spec = EnginePhmSpec()
        delays = []
        for size in (-0.01, -0.02, -0.04):
            study = detection_operating_characteristic(
                spec, thresholds=(5.0,), fault_size=size, runs=8, steps=130, seed=0
            )
            point = study.points[0]
            self.assertGreater(point.detection_probability, 0.5)
            delays.append(point.mean_delay_steps)
        self.assertTrue(all(d is not None for d in delays))
        self.assertGreater(delays[0], delays[1])
        self.assertGreater(delays[1], delays[2])
        study = detection_operating_characteristic(
            spec, thresholds=(3.0, 5.0, 8.0), fault_size=-0.02, runs=8, steps=130
        )
        alarms = [p.false_alarm_runs for p in study.points]
        self.assertEqual(alarms, sorted(alarms, reverse=True))
        self.assertGreater(alarms[0], alarms[-1])
        mean_delays = [p.mean_delay_steps for p in study.points if p.delays]
        self.assertEqual(mean_delays, sorted(mean_delays))

    def test_filter_innovation_arl0_vs_theory(self) -> None:
        """On filter innovations ARL0 is within a factor 2 of the iid theory.

        The innovations of a consistent EKF are only approximately iid, and
        the number of alarms is small (Poisson error), so this is a loose
        sanity band, not a calibration.
        """
        spec = EnginePhmSpec()
        study = detection_operating_characteristic(
            spec, thresholds=(4.0,), fault_size=-0.02, runs=16, steps=200, seed=4
        )
        point = study.points[0]
        theory = cusum_arl_siegmund(0.5, 4.0, two_sided=True)
        self.assertGreater(point.false_alarm_runs, 2)
        self.assertGreater(point.arl0_estimate, theory / 2.0)
        self.assertLess(point.arl0_estimate, theory * 2.0)

    def test_glr_detector_runs_and_orders_thresholds(self) -> None:
        """The GLR operating curve exists and false alarms fall with ``h``."""
        study = detection_operating_characteristic(
            EnginePhmSpec(),
            thresholds=(4.0, 8.0),
            fault_size=-0.03,
            runs=10,
            steps=140,
            detector="glr",
            window=30,
        )
        self.assertGreaterEqual(
            study.points[0].false_alarm_runs, study.points[1].false_alarm_runs
        )
        self.assertGreater(study.points[1].detection_probability, 0.5)

    @unittest.skipUnless(SLOW, "slow tier")
    def test_operating_characteristic_slow(self) -> None:
        """200 seeds: measured ARL0 vs Siegmund and delay table."""
        spec = EnginePhmSpec()
        for size in (-0.01, -0.02, -0.04):
            study = detection_operating_characteristic(
                spec,
                thresholds=(3.0, 4.0, 5.0, 6.0, 8.0),
                fault_size=size,
                runs=200,
                steps=200,
                seed=100,
            )
            for p in study.points:
                theory = cusum_arl_siegmund(0.5, p.threshold, two_sided=True)
                print(
                    f"[slow] size {size} h {p.threshold}: ARL0 {p.arl0_estimate:.1f} "
                    f"(theory {theory:.1f}), P(det) {p.detection_probability:.3f}, "
                    f"delay {p.mean_delay_steps}"
                )


class FirstPassageTests(unittest.TestCase):
    """Inverse-Gaussian first passage of the Wiener process."""

    def test_cdf_matches_numerical_integral_of_density(self) -> None:
        """Simpson integration of the IG density reproduces the cdf."""
        mu, lam = 2.0, 4.0

        def pdf(t: float) -> float:
            if t <= 0.0:
                return 0.0
            return math.sqrt(lam / (2.0 * math.pi * t**3)) * math.exp(
                -lam * (t - mu) ** 2 / (2.0 * mu * mu * t)
            )

        for upper in (0.5, 1.0, 2.0, 5.0, 12.0):
            n = 4000
            h = upper / n
            total = pdf(upper) + sum(
                (4 if i % 2 else 2) * pdf(i * h) for i in range(1, n)
            )
            integral = total * h / 3.0
            self.assertAlmostEqual(
                inverse_gaussian_cdf(upper, mu, lam), integral, places=6
            )
        self.assertEqual(inverse_gaussian_cdf(0.0, mu, lam), 0.0)
        self.assertEqual(inverse_gaussian_cdf(math.inf, mu, lam), 1.0)
        self.assertGreater(inverse_gaussian_cdf(1e3, mu, lam), 1.0 - 1e-12)

    def test_large_shape_does_not_overflow(self) -> None:
        """exp(2 lam / mu) is handled in log space for lam / mu ~ 100."""
        value = inverse_gaussian_cdf(100.0, 200.0, 20000.0)
        self.assertTrue(0.0 <= value <= 1.0)
        self.assertAlmostEqual(
            inverse_gaussian_cdf(200.0, 200.0, 20000.0),
            0.5 + 0.0,
            delta=0.35,
        )

    def test_quantile_inverts_cdf_and_moments_of_sampler(self) -> None:
        """Quantile round trip; sampler mean ``mu`` and variance ``mu^3 / lam``."""
        d, mu, s = 0.06, 3e-4, 5e-4
        for p in (0.05, 0.5, 0.95):
            t = wiener_first_passage_quantile(p, d, mu, s)
            self.assertAlmostEqual(wiener_first_passage_cdf(t, d, mu, s), p, places=9)
        gen = resolve_stream(12).generator()
        mean, shape, n = 2.0, 4.0, 20000
        xs = [sample_inverse_gaussian(gen, mean, shape) for _ in range(n)]
        m = sum(xs) / n
        v = sum((x - m) ** 2 for x in xs) / (n - 1)
        self.assertAlmostEqual(m, mean, delta=4.0 * math.sqrt(mean**3 / shape / n))
        self.assertAlmostEqual(v, mean**3 / shape, delta=0.1 * mean**3 / shape)

    def test_continuous_first_passage_against_bridge_monte_carlo(self) -> None:
        """Wiener paths with a Brownian-bridge crossing test follow the IG cdf.

        Distance 1, drift 0.5, volatility 0.5 (IG mean 2, shape 4) on a grid of
        0.05 with the crossing probability ``exp(-2 x y / (sigma^2 dt))`` of
        each step; the crossing time is taken at mid-step. Tolerance: three
        Monte-Carlo standard errors plus 0.015 for the time resolution.
        """
        d, mu, s, dt, n = 1.0, 0.5, 0.5, 0.05, 2500
        gen = resolve_stream(5).generator()
        times = []
        root = math.sqrt(dt)
        for _ in range(n):
            x = d
            t = 0.0
            while True:
                y = x - mu * dt + s * root * gen.gauss(0.0, 1.0)
                t += dt
                if y <= 0.0 or gen.random() < math.exp(-2.0 * x * y / (s * s * dt)):
                    times.append(t - dt / 2.0)
                    break
                x = y
        for q in (1.0, 1.5, 2.0, 3.0, 4.0):
            emp = sum(1 for v in times if v <= q) / n
            th = wiener_first_passage_cdf(q, d, mu, s)
            self.assertLess(
                abs(emp - th),
                3.0 * math.sqrt(th * (1.0 - th) / n) + 0.015,
                (q, emp, th),
            )

    def test_discrete_monitoring_shift_improves_the_law(self) -> None:
        """Walk with unit steps: shifted barrier is closer than the raw one.

        ``d = 2``, drift 0.1, ``sigma = 0.5`` per step; the corrected law is
        within 0.04 of the simulated cdf at every probe and strictly better
        than the uncorrected one at each of them.
        """
        d, mu, s, n = 2.0, 0.1, 0.5, 3000
        gen = resolve_stream(5).generator()
        steps = []
        for _ in range(n):
            x = d
            k = 0
            while x > 0.0:
                x += -mu + s * gen.gauss(0.0, 1.0)
                k += 1
            steps.append(k)
        self.assertAlmostEqual(DISCRETE_MONITORING_SHIFT, 0.5826)
        for q in (10, 15, 20, 30, 40):
            emp = sum(1 for v in steps if v <= q) / n
            corrected = wiener_first_passage_cdf(q, d + 0.5826 * s, mu, s)
            raw = wiener_first_passage_cdf(q, d, mu, s)
            self.assertLess(abs(emp - corrected), 0.04, (q, emp, corrected))
            self.assertLess(
                abs(emp - corrected), abs(emp - raw), (q, emp, corrected, raw)
            )


class RulTests(unittest.TestCase):
    """Predictive remaining-useful-life law and its calibration."""

    def test_point_belief_reproduces_inverse_gaussian(self) -> None:
        """A tight belief gives the exact IG cdf without the discrete shift."""
        spec = EnginePhmSpec()
        cov = tuple(tuple(1e-20 if i == j else 0.0 for j in range(4)) for i in range(4))
        state = GaussianState((0.95, 1.0, 0.93, -3e-4), cov)
        dist = predict_rul(
            spec,
            state,
            failure_threshold=0.88,
            draws=50,
            discrete_correction=False,
        )
        for t in (50.0, 150.0, 200.0, 400.0):
            expected = wiener_first_passage_cdf(t, 0.05, 3e-4, spec.eta_process_std)
            self.assertAlmostEqual(dist.cdf(t * spec.step_seconds), expected, places=6)
        self.assertAlmostEqual(dist.plug_in_mean_seconds, 0.05 / 3e-4)
        lo, hi = dist.interval(0.9)
        self.assertLess(lo, dist.quantile(0.5))
        self.assertLess(dist.quantile(0.5), hi)
        self.assertAlmostEqual(
            dist.quantile(0.5),
            wiener_first_passage_quantile(0.5, 0.05, 3e-4, spec.eta_process_std),
            delta=1e-3,
        )

    def test_degenerate_beliefs_are_reported_not_hidden(self) -> None:
        """Draws that never fail or already failed carry explicit mass."""
        spec = EnginePhmSpec()
        cov = tuple(
            tuple(
                (1e-3 if i == 2 else 1e-5 if i == 3 else 1e-6) if i == j else 0.0
                for j in range(4)
            )
            for i in range(4)
        )
        flat = predict_rul(
            spec, GaussianState((0.95, 1.0, 0.93, 0.0), cov), failure_threshold=0.9
        )
        self.assertGreater(flat.non_degrading, 0.3)
        self.assertEqual(flat.quantile(0.9), math.inf)
        self.assertLess(flat.cdf(1e9), 1.0 - flat.non_degrading + 1e-9)
        dead = predict_rul(
            spec, GaussianState((0.7, 1.0, 0.7, -3e-4), cov), failure_threshold=0.9
        )
        self.assertGreater(dead.already_failed, 0.99)
        self.assertEqual(dead.quantile(0.5), 0.0)

    def test_coverage_fast(self) -> None:
        """40 seeds: central-interval coverage inside three binomial sigma."""
        study = rul_coverage_study(
            EnginePhmSpec(),
            failure_threshold=0.88,
            observe_steps=100,
            horizon_steps=600,
            runs=40,
            seed=3,
        )
        cal = study.calibration
        print(
            f"\n[fast] RUL coverage {cal.coverage} for levels {cal.levels}, "
            f"PIT mean {cal.pit_mean:.3f}, censored {study.censored}"
        )
        self.assertEqual(study.censored, 0)
        self.assertTrue(cal.within_band, cal)
        self.assertAlmostEqual(cal.pit_mean, 0.5, delta=3.0 * math.sqrt(1 / 12 / 40))

    def test_discrete_correction_changes_median_upwards(self) -> None:
        """The shifted barrier lengthens the predicted life (same belief)."""
        spec = EnginePhmSpec()
        sim = simulate_plant(spec, 100, seed=9)
        state = estimate_health(spec, sim.observations).filtered[-1]
        plain = predict_rul(
            spec, state, failure_threshold=0.88, seed=1, discrete_correction=False
        )
        shifted = predict_rul(spec, state, failure_threshold=0.88, seed=1)
        self.assertGreater(shifted.quantile(0.5), plain.quantile(0.5))

    @unittest.skipUnless(SLOW, "slow tier")
    def test_coverage_slow(self) -> None:
        """300 seeds: coverage, with and without the discrete-monitoring shift."""
        for corrected in (True, False):
            study = rul_coverage_study(
                EnginePhmSpec(),
                failure_threshold=0.88,
                observe_steps=100,
                horizon_steps=800,
                runs=300,
                seed=11,
                discrete_correction=corrected,
            )
            cal = study.calibration
            print(
                f"[slow] corrected={corrected}: coverage {cal.coverage} for "
                f"{cal.levels}, bands {tuple(round(b, 3) for b in cal.band)}, PIT "
                f"mean {cal.pit_mean:.3f}, var {cal.pit_variance:.4f}, "
                f"censored {study.censored}"
            )
            if corrected:
                self.assertTrue(cal.within_band)

    def test_calibration_summary_on_known_pits(self) -> None:
        """Uniform grid PITs cover each central interval at its level."""
        pits = [(i + 0.5) / 1000 for i in range(1000)]
        cal = rul_calibration(pits, (0.5, 0.9))
        self.assertAlmostEqual(cal.coverage[0], 0.5, places=9)
        self.assertAlmostEqual(cal.coverage[1], 0.9, places=9)
        self.assertAlmostEqual(cal.pit_mean, 0.5, places=12)
        self.assertAlmostEqual(cal.pit_variance, 1.0 / 12.0, delta=1e-6)
        self.assertTrue(cal.within_band)
        bad = rul_calibration([0.01] * 50)
        self.assertFalse(bad.within_band)


class PipelineTests(unittest.TestCase):
    """End-to-end assessment of a faulted record."""

    def test_assessment_detects_and_locates_the_fault(self) -> None:
        """CUSUM and GLR alarm after a -0.03 step at k = 70 and GLR dates it."""
        spec = EnginePhmSpec()
        sim = simulate_plant(spec, 110, seed=2, fault=StepFault(70, -0.03))
        out = assess_engine_health(
            spec, sim.observations, failure_threshold=0.85, cusum_threshold=5.0
        )
        self.assertIsNotNone(out.cusum.alarm_index)
        self.assertIsNotNone(out.glr.alarm_index)
        self.assertGreaterEqual(out.cusum.alarm_index or 0, 69)
        self.assertGreaterEqual(out.glr.alarm_index or 0, 69)
        self.assertLessEqual(out.glr.alarm_index or 999, 85)
        self.assertEqual(out.cusum.alarm_side, "lower")
        self.assertLess(out.glr.magnitude or 0.0, 0.0)
        self.assertAlmostEqual(out.glr.change_index or 0, 69, delta=6)
        self.assertEqual(len(out.innovations), 110)
        self.assertGreater(out.rul.quantile(0.5), 0.0)

    def test_fault_free_record_stays_quiet(self) -> None:
        """No fault, generous thresholds: no alarm for this seed."""
        spec = EnginePhmSpec()
        sim = simulate_plant(spec, 100, seed=2)
        out = assess_engine_health(
            spec,
            sim.observations,
            failure_threshold=0.85,
            cusum_threshold=8.0,
            glr_threshold=10.0,
        )
        self.assertIsNone(out.cusum.alarm_index)
        self.assertIsNone(out.glr.alarm_index)


class FailureCaseTests(unittest.TestCase):
    """Invalid input is rejected, never repaired."""

    def test_specification_validation(self) -> None:
        """Bad physical parameters raise."""
        with self.assertRaises(ValueError):
            EnginePhmSpec(throat_area=-1.0)
        with self.assertRaises(ValueError):
            EnginePhmSpec(gamma=1.0)
        with self.assertRaises(ValueError):
            EnginePhmSpec(initial_eta=0.0)
        with self.assertRaises(TypeError):
            EnginePhmSpec(step_seconds="1")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            EnginePhmSpec(pressure_noise_std=0.0)

    def test_simulation_and_estimation_validation(self) -> None:
        """Wrong types, empty records and unknown methods raise."""
        spec = EnginePhmSpec()
        with self.assertRaises(ValueError):
            simulate_plant(spec, 0)
        with self.assertRaises(TypeError):
            simulate_plant(spec, 5.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            simulate_plant("spec", 5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            StepFault(0, -0.1)
        with self.assertRaises(ValueError):
            estimate_health(spec, [])
        with self.assertRaises(ValueError):
            estimate_health(spec, [(1.0, 1.0)], method="pf")
        with self.assertRaises(ValueError):
            estimate_health(spec, [(1.0,)])

    def test_missing_observation_is_not_silently_skipped(self) -> None:
        """Innovation standardisation refuses a record with a missing sample."""
        spec = EnginePhmSpec()
        sim = simulate_plant(spec, 5, seed=1)
        from probstat.filtering import extended_kalman_filter

        result = extended_kalman_filter(
            augmented_model(spec), [sim.observations[0], None, sim.observations[2]]
        )
        with self.assertRaises(ValueError):
            normalized_innovations(result)
        ok = estimate_health(spec, sim.observations)
        with self.assertRaises(ValueError):
            normalized_innovations(ok, channel=5)

    def test_detector_validation(self) -> None:
        """Thresholds, windows and series are validated."""
        with self.assertRaises(ValueError):
            cusum_monitor([], threshold=3.0)
        with self.assertRaises(ValueError):
            cusum_monitor([0.0], threshold=0.0)
        with self.assertRaises(ValueError):
            cusum_monitor([0.0], reference=-1.0)
        with self.assertRaises(ValueError):
            cusum_monitor([float("nan")])
        with self.assertRaises(ValueError):
            glr_mean_shift([0.0], threshold=1.0, window=0)
        with self.assertRaises(ValueError):
            cusum_arl_siegmund(0.5, 800.0)
        with self.assertRaises(ValueError):
            cusum_arl_numerical(0.5, 4.0, cells=3)
        with self.assertRaises(ValueError):
            detection_operating_characteristic(
                EnginePhmSpec(), thresholds=(3.0,), fault_size=-0.02, fault_index=200
            )
        with self.assertRaises(ValueError):
            detection_operating_characteristic(
                EnginePhmSpec(), thresholds=(3.0,), fault_size=-0.02, detector="x"
            )

    def test_rul_validation(self) -> None:
        """Law parameters and beliefs are validated."""
        spec = EnginePhmSpec()
        with self.assertRaises(ValueError):
            inverse_gaussian_cdf(1.0, -1.0, 1.0)
        with self.assertRaises(ValueError):
            wiener_first_passage_quantile(1.0, 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            wiener_first_passage_cdf(1.0, 1.0, 0.0, 1.0)
        with self.assertRaises(TypeError):
            sample_inverse_gaussian(object(), 1.0, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            predict_rul(spec, (1.0,), failure_threshold=0.8)  # type: ignore[arg-type]
        short = GaussianState((1.0, 1.0), ((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(ValueError):
            predict_rul(spec, short, failure_threshold=0.8)
        flat = GaussianState(
            (1.0, 1.0, 0.9, 0.0),
            tuple(tuple(0.0 for _ in range(4)) for _ in range(4)),
        )
        with self.assertRaises(ValueError):
            predict_rul(spec, flat, failure_threshold=0.8)
        with self.assertRaises(ValueError):
            rul_calibration([])
        with self.assertRaises(ValueError):
            rul_calibration([1.5])
        with self.assertRaises(ValueError):
            rul_calibration([0.5], levels=(1.0,))
        with self.assertRaises(ValueError):
            rul_coverage_study(
                EnginePhmSpec(),
                failure_threshold=0.0,
                observe_steps=10,
                horizon_steps=10,
                runs=2,
            )


if __name__ == "__main__":
    unittest.main()
