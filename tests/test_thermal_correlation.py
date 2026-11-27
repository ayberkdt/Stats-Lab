"""Tests for the lumped thermal-network correlation module.

Order: closed-form 2- and 3-node steady states, the single-node backward Euler
recursion and the energy identities, then exact Jacobians and sensitivities
against central differences, then seeded synthetic calibration (recovery,
Laplace against transitional MCMC), identifiability (series conductances,
transient and power-level remedies, profile likelihood), model-form error with
and without the discrepancy term (flight-case coverage, measured), and failure
cases. All data are synthetic. Stochastic claims use predeclared seeds and are
judged by measured frequencies with bands, not pinned digits.
"""

from __future__ import annotations

import math
import os
import unittest

from probstat.inverse.thermal_correlation import (
    DEMO_BASE_POWER,
    DEMO_SENSORS,
    STEFAN_BOLTZMANN,
    SteadyTest,
    ThermalCorrelationProblem,
    ThermalLink,
    ThermalModel,
    ThermalNetwork,
    ThermalParameter,
    ThermalSolveError,
    TransientTest,
    check_thermal_jacobian,
    constant_power,
    demo_calibration_tests,
    demo_flight_case,
    demo_spacecraft_model,
    demo_transient_test,
    identifiability_report,
    predict_flight,
    profile_log_parameter,
    simulate_thermal_data,
    simulate_transient,
    solve_steady_state,
    step_power,
    thermal_laplace,
    thermal_tmcmc,
)

THETA_TRUE = (1.3, 0.75, 1.2, 0.9)
LOG_TRUE = tuple(math.log(v) for v in THETA_TRUE)
NOISE = 0.2
PREDICT_NODES = (0, 1, 3, 4)


def _two_node(kind: str, value: float) -> ThermalNetwork:
    """Return one free node coupled to one boundary node."""
    link = (ThermalLink(0, 1, value),)
    if kind == "conductive":
        return ThermalNetwork(("a", "wall"), (100.0,), link, ())
    return ThermalNetwork(("a", "wall"), (100.0,), (), link)


class ClosedFormTests(unittest.TestCase):
    """Exact steady and transient solutions of tiny networks."""

    def test_conductive_two_node(self) -> None:
        """``T = Tb + Q/G`` to Newton tolerance."""
        state = solve_steady_state(_two_node("conductive", 2.0), (10.0,), (300.0,))
        self.assertTrue(state.converged)
        self.assertAlmostEqual(state.temperatures[0], 305.0, places=9)
        self.assertAlmostEqual(state.boundary_heat_flow[0], 10.0, places=9)
        self.assertLess(abs(state.energy_residual), 1e-9)

    def test_radiative_two_node(self) -> None:
        """``T^4 = Tb^4 + Q/R`` with ``R = eps sigma A``."""
        r = 0.8 * STEFAN_BOLTZMANN * 2.0
        state = solve_steady_state(_two_node("radiative", r), (50.0,), (150.0,))
        expected = (150.0**4 + 50.0 / r) ** 0.25
        self.assertTrue(state.converged)
        self.assertAlmostEqual(state.temperatures[0], expected, places=7)
        self.assertLess(abs(state.energy_residual), 1e-9)

    def test_series_chain(self) -> None:
        """Two free nodes in series with a boundary: flows fix the drops."""
        net = ThermalNetwork(
            ("x", "y", "wall"),
            (10.0, 10.0),
            (ThermalLink(0, 1, 4.0), ThermalLink(1, 2, 2.0)),
            (),
        )
        state = solve_steady_state(net, (6.0, 2.0), (280.0,))
        t_y = 280.0 + 8.0 / 2.0
        t_x = t_y + 6.0 / 4.0
        self.assertAlmostEqual(state.temperatures[1], t_y, places=9)
        self.assertAlmostEqual(state.temperatures[0], t_x, places=9)

    def test_backward_euler_single_node_matches_recursion(self) -> None:
        """The linear step ``(C/dt + G) T+ = C/dt T + Q + G Tb`` is exact."""
        c, g, q, tb, dt = 100.0, 2.0, 10.0, 300.0, 5.0
        net = ThermalNetwork(("a", "wall"), (c,), (ThermalLink(0, 1, g),), ())
        test = TransientTest(constant_power((q,)), (tb,), dt, 40, (10, 40))
        response = simulate_transient(net, test, initial=(tb,))
        self.assertTrue(response.converged)
        t = tb
        for k in range(40):
            t = (c / dt * t + q + g * tb) / (c / dt + g)
            self.assertAlmostEqual(response.temperatures[k + 1][0], t, places=9)

    def test_backward_euler_is_first_order_to_the_exponential(self) -> None:
        """Halving ``dt`` halves the error to ``T(t) = Tb + Q/G (1 - e^-Gt/C)``."""
        c, g, q, tb = 100.0, 2.0, 10.0, 300.0
        net = ThermalNetwork(("a", "wall"), (c,), (ThermalLink(0, 1, g),), ())
        errors = []
        for dt, steps in ((5.0, 20), (2.5, 40)):
            test = TransientTest(constant_power((q,)), (tb,), dt, steps, (steps,))
            response = simulate_transient(net, test, initial=(tb,))
            exact = tb + q / g * (1.0 - math.exp(-g * 100.0 / c))
            errors.append(abs(response.temperatures[-1][0] - exact))
        self.assertLess(errors[1], 0.6 * errors[0])
        self.assertGreater(errors[1], 0.4 * errors[0])

    def test_transient_approaches_steady_state(self) -> None:
        """Long constant power converges to the Newton steady state."""
        model = demo_spacecraft_model()
        net = model.network_at((1.0, 1.0, 1.0, 1.0))
        steady = demo_calibration_tests()[1]
        test = TransientTest(
            constant_power(steady.power), steady.boundary, 200.0, 300, (300,)
        )
        response = simulate_transient(net, test, initial=(150.0,) * 5)
        state = solve_steady_state(net, steady.power, steady.boundary)
        self.assertTrue(response.converged)
        for a, b in zip(response.temperatures[-1], state.temperatures, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-3)

    def test_energy_balance_steady_and_transient(self) -> None:
        """Dissipated power leaves through the boundary; steps conserve energy."""
        model = demo_spacecraft_model()
        theta = THETA_TRUE
        state, _ = model.steady(theta, demo_calibration_tests()[2])
        self.assertLess(abs(state.energy_residual), 1e-9)
        self.assertAlmostEqual(
            sum(state.boundary_heat_flow), sum(demo_calibration_tests()[2].power), 8
        )
        response, _ = model.simulate(theta, demo_transient_test())
        self.assertTrue(response.converged)
        self.assertLess(response.max_energy_residual, 1e-6)
        # the net heat inflow of all nodes sums to zero (links are antisymmetric)
        inflow = model.network_at(theta).node_inflows(
            state.temperatures, demo_calibration_tests()[2].boundary
        )
        self.assertLess(abs(math.fsum(inflow)), 1e-9)


class DerivativeTests(unittest.TestCase):
    """Exact Jacobians against central differences."""

    def test_temperature_jacobian(self) -> None:
        """``dF/dT`` against a central difference of ``net_heat_flow``."""
        net = demo_spacecraft_model().network_at(THETA_TRUE)
        t = (190.0, 240.0, 220.0, 188.0, 250.0)
        b = (150.0,)
        q = (40.0, 60.0, 0.0, 0.0, 20.0)
        jac = net.jacobian_temperature(t, b)
        h = 1e-3
        for j in range(5):
            tp = tuple(v + (h if i == j else 0.0) for i, v in enumerate(t))
            tm = tuple(v - (h if i == j else 0.0) for i, v in enumerate(t))
            fp, fm = net.net_heat_flow(tp, b, q), net.net_heat_flow(tm, b, q)
            for i in range(5):
                fd = (fp[i] - fm[i]) / (2 * h)
                self.assertAlmostEqual(jac[i][j], fd, delta=1e-6 * max(1.0, abs(fd)))

    def test_steady_sensitivity(self) -> None:
        """``dT/dtheta`` from the implicit function theorem against FD."""
        model = demo_spacecraft_model()
        test = demo_calibration_tests()[1]
        _, sens = model.steady(THETA_TRUE, test, sensitivity=True)
        assert sens is not None
        h = 1e-6
        for k in range(4):
            up = tuple(
                v * (1 + (h if i == k else 0.0)) for i, v in enumerate(THETA_TRUE)
            )
            dn = tuple(
                v * (1 - (h if i == k else 0.0)) for i, v in enumerate(THETA_TRUE)
            )
            tp = model.steady(up, test)[0].temperatures
            tm = model.steady(dn, test)[0].temperatures
            for i in range(5):
                fd = (tp[i] - tm[i]) / (2 * h * THETA_TRUE[k])
                self.assertAlmostEqual(sens[i][k], fd, delta=1e-5 * max(1.0, abs(fd)))

    def test_transient_sensitivity_including_initial_steady_state(self) -> None:
        """The discrete forward sensitivity against central differences."""
        model = demo_spacecraft_model()
        test = demo_transient_test()
        _, sens = model.simulate(THETA_TRUE, test, sensitivity=True)
        h = 1e-6
        for k in range(4):
            up = tuple(
                v * (1 + (h if i == k else 0.0)) for i, v in enumerate(THETA_TRUE)
            )
            dn = tuple(
                v * (1 - (h if i == k else 0.0)) for i, v in enumerate(THETA_TRUE)
            )
            rp, _ = model.simulate(up, test)
            rm, _ = model.simulate(dn, test)
            for step in (0, 20, 120):
                for i in range(5):
                    fd = (rp.temperatures[step][i] - rm.temperatures[step][i]) / (
                        2 * h * THETA_TRUE[k]
                    )
                    self.assertAlmostEqual(
                        sens[step][i][k], fd, delta=1e-5 * max(1.0, abs(fd))
                    )

    def test_problem_jacobian_check(self) -> None:
        """``dG/du`` of the stacked steady plus transient sensor map."""
        model = demo_spacecraft_model()
        tests = [*demo_calibration_tests(), demo_transient_test()]
        data = simulate_thermal_data(model, THETA_TRUE, DEMO_SENSORS, tests, 0.0)
        problem = ThermalCorrelationProblem(
            model, DEMO_SENSORS, tests, data, noise_sd=NOISE
        )
        check = check_thermal_jacobian(problem, (0.1, -0.2, 0.05, -0.1))
        self.assertLess(check.max_rel_error, 1e-6)
        self.assertLess(check.max_abs_error, 1e-5)


class CalibrationTests(unittest.TestCase):
    """MAP, Laplace and TMCMC on synthetic data from the same model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the model, tests and one noisy data set."""
        cls.model = demo_spacecraft_model()
        cls.tests = demo_calibration_tests()
        data = simulate_thermal_data(
            cls.model, THETA_TRUE, DEMO_SENSORS, cls.tests, NOISE, seed=1
        )
        cls.problem = ThermalCorrelationProblem(
            cls.model, DEMO_SENSORS, cls.tests, data, noise_sd=NOISE
        )

    def test_noise_free_data_recover_identified_parameters(self) -> None:
        """Noise-free data: identified parameters and the series sum are exact."""
        data = simulate_thermal_data(
            self.model, THETA_TRUE, DEMO_SENSORS, self.tests, 0.0
        )
        problem = ThermalCorrelationProblem(
            self.model, DEMO_SENSORS, self.tests, data, noise_sd=0.02
        )
        lap = thermal_laplace(problem)
        self.assertTrue(lap.converged)
        # measured deviations: 1e-6 (G_PE), 1e-7 (eps), 1e-5 (series sum)
        self.assertAlmostEqual(lap.theta_map[2], 1.2, delta=1e-4)
        self.assertAlmostEqual(lap.theta_map[3], 0.9, delta=1e-4)
        # the series pair acts through the sum of resistances only
        r_true = 1.0 / (THETA_TRUE[0] * 2.0) + 1.0 / (THETA_TRUE[1] * 1.2)
        r_map = 1.0 / (lap.theta_map[0] * 2.0) + 1.0 / (lap.theta_map[1] * 1.2)
        self.assertAlmostEqual(r_map, r_true, delta=1e-4)

    def test_laplace_converges_and_has_a_covariance(self) -> None:
        """The MAP optimizer meets its tolerance and the Hessian is SPD."""
        lap = thermal_laplace(self.problem)
        self.assertTrue(lap.converged, lap.stop_reason)
        self.assertIsNotNone(lap.covariance)
        self.assertTrue(math.isfinite(lap.log_evidence))
        for sd in lap.standard_deviations():
            self.assertGreater(sd, 0.0)

    def test_laplace_recovers_truth_over_predeclared_seeds(self) -> None:
        """True ``log theta`` lies within 2.5 posterior sd for most draws."""
        hits = total = 0
        for seed in range(500, 512):
            data = simulate_thermal_data(
                self.model, THETA_TRUE, DEMO_SENSORS, self.tests, NOISE, seed=seed
            )
            problem = ThermalCorrelationProblem(
                self.model, DEMO_SENSORS, self.tests, data, noise_sd=NOISE
            )
            lap = thermal_laplace(problem)
            self.assertTrue(lap.converged)
            sd = lap.standard_deviations()
            for k in range(4):
                total += 1
                hits += abs(lap.u_map[k] - LOG_TRUE[k]) <= 2.5 * sd[k]
        # 48 comparisons; measured 48/48 on these seeds; a Gaussian band of
        # 2.5 sd has coverage 0.988, so fewer than 44 would flag a defect.
        self.assertGreaterEqual(hits, 44)

    def test_tmcmc_agrees_with_laplace_on_identified_parameters(self) -> None:
        """Two-parameter, one-test problem where Laplace is accurate."""
        sub = ThermalModel(
            self.model.network, (self.model.parameters[2], self.model.parameters[3])
        )
        tests = self.tests[1:2]
        data = simulate_thermal_data(
            sub, (1.2, 0.9), DEMO_SENSORS, tests, NOISE, seed=7
        )
        problem = ThermalCorrelationProblem(
            sub, DEMO_SENSORS, tests, data, noise_sd=NOISE
        )
        lap = thermal_laplace(problem)
        sd = lap.standard_deviations()
        post = thermal_tmcmc(problem, particles=100, seed=2)
        self.assertTrue(post.converged)
        mean = post.mean_log()
        spread = post.standard_deviations()
        for k in range(2):
            # measured: mean differences 0.04-0.13 sd and sd ratios 0.96-0.99
            self.assertLess(abs(mean[k] - lap.u_map[k]), 0.8 * sd[k])
            self.assertGreater(spread[k] / sd[k], 0.6)
            self.assertLess(spread[k] / sd[k], 1.6)
        assert post.tmcmc is not None
        self.assertAlmostEqual(post.log_evidence, lap.log_evidence, delta=3.0)


class IdentifiabilityTests(unittest.TestCase):
    """What a steady test cannot separate and what separates it."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fit the same truth with steady, steady plus transient and one level."""
        cls.model = demo_spacecraft_model()
        steady = demo_calibration_tests()
        transient = demo_transient_test()
        cls.laplace = {}
        for label, tests in (
            ("one_level", steady[1:2]),
            ("three_levels", list(steady)),
            ("transient", [steady[1], transient]),
        ):
            data = simulate_thermal_data(
                cls.model, THETA_TRUE, DEMO_SENSORS, tests, NOISE, seed=1
            )
            problem = ThermalCorrelationProblem(
                cls.model, DEMO_SENSORS, tests, data, noise_sd=NOISE
            )
            cls.laplace[label] = thermal_laplace(problem)

    def test_series_conductances_are_not_separable_by_a_steady_test(self) -> None:
        """Rank-deficient likelihood Hessian and correlation -1 (measured)."""
        for label in ("one_level", "three_levels"):
            report = identifiability_report(self.laplace[label])
            eig = report.likelihood_eigenvalues
            self.assertLess(abs(eig[0]), 1e-8 * eig[-1])
            self.assertLess(report.correlation[0][1], -0.99)
            # the null direction is the first two parameters, opposite signs
            w = report.weakest_direction
            self.assertLess(w[0] * w[1], 0.0)
            self.assertLess(abs(w[2]) + abs(w[3]), 1e-6)
            # the prior is almost all the information on them
            self.assertGreater(report.shrinkage[0], 0.8)
        # more power levels do not help the series pair (same eigen structure)
        sd_one = self.laplace["one_level"].standard_deviations()
        sd_three = self.laplace["three_levels"].standard_deviations()
        self.assertAlmostEqual(sd_one[0], sd_three[0], delta=0.01)

    def test_transient_separates_them(self) -> None:
        """The bracket heat capacity gives a finite smallest eigenvalue."""
        steady = identifiability_report(self.laplace["three_levels"])
        transient = identifiability_report(self.laplace["transient"])
        self.assertGreater(transient.likelihood_eigenvalues[0], 50.0)
        sd_s = steady.posterior_sd
        sd_t = transient.posterior_sd
        # measured sds of u: 0.435, 0.246 (steady) against 0.070, 0.028
        self.assertLess(sd_t[0], sd_s[0] / 4.0)
        self.assertLess(sd_t[1], sd_s[1] / 4.0)

    def test_second_power_level_tightens_the_parallel_pair(self) -> None:
        """Panel-box conduction and emissivity act in parallel on the panel."""
        one = self.laplace["one_level"].standard_deviations()
        three = self.laplace["three_levels"].standard_deviations()
        # measured: 0.0121 -> 0.0070 and 0.0051 -> 0.0030
        self.assertLess(three[2], 0.75 * one[2])
        self.assertLess(three[3], 0.75 * one[3])

    def test_profile_likelihood_flat_for_steady(self) -> None:
        """Steady profile of ``log G_box_bracket`` is flat (nuisances re-fitted)."""
        lap = self.laplace["three_levels"]
        center = lap.u_map[0]
        grid = tuple(center + d for d in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3))
        result = profile_log_parameter(lap.problem, lap.u_map, 0, grid)
        self.assertLess(max(result.likelihood_ratio), 0.05)

    def _short_transient_profile(self, grid_offsets: tuple[float, ...]) -> float:
        """Largest profile likelihood ratio with a cheaper 40-step transient."""
        steady = demo_calibration_tests()[1]
        low = tuple(0.5 * q for q in DEMO_BASE_POWER)
        short = TransientTest(
            step_power(low, DEMO_BASE_POWER, 0.0),
            (150.0,),
            60.0,
            40,
            tuple(range(2, 41, 2)),
        )
        tests = [steady, short]
        data = simulate_thermal_data(
            self.model, THETA_TRUE, DEMO_SENSORS, tests, NOISE, seed=1
        )
        problem = ThermalCorrelationProblem(
            self.model, DEMO_SENSORS, tests, data, noise_sd=NOISE
        )
        lap = thermal_laplace(problem)
        grid = tuple(lap.u_map[0] + d for d in grid_offsets)
        result = profile_log_parameter(problem, lap.u_map, 0, grid)
        return max(result.likelihood_ratio)

    def test_profile_likelihood_curved_with_a_transient(self) -> None:
        """A step of 0.2 in ``log G`` costs more than the 95% chi-square cut."""
        # measured 5.8 at +0.2 (and 9.4 at -0.2); chi2_{1,0.95} = 3.84
        self.assertGreater(self._short_transient_profile((0.0, 0.2)), 3.84)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_profile_likelihood_full_transient(self) -> None:
        """The full 120-step transient over a six-point grid (measured 13.7)."""
        lap = self.laplace["transient"]
        center = lap.u_map[0]
        grid = tuple(center + d for d in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3))
        result = profile_log_parameter(lap.problem, lap.u_map, 0, grid)
        print("profile LR (transient):", [round(v, 2) for v in result.likelihood_ratio])
        self.assertGreater(max(result.likelihood_ratio), 10.0)


class FlightPredictionTests(unittest.TestCase):
    """Correlation for prediction with model-form error."""

    EXTRA = 0.3

    @classmethod
    def setUpClass(cls) -> None:
        """Truth with an unmodelled panel-radiator coupling; flight truth."""
        cls.nominal = demo_spacecraft_model()
        cls.truth = demo_spacecraft_model(cls.EXTRA)
        cls.tests = demo_calibration_tests()
        cls.flight = demo_flight_case()
        state, _ = cls.truth.steady(THETA_TRUE, cls.flight)
        cls.flight_truth = tuple(state.temperatures[i] for i in PREDICT_NODES)

    def _coverage(
        self, seeds: range, *, discrepancy: bool, grid: int
    ) -> tuple[int, int]:
        """Return (covered, total) over seeds at the 90 percent band."""
        covered = total = 0
        for seed in seeds:
            data = simulate_thermal_data(
                self.truth, THETA_TRUE, DEMO_SENSORS, self.tests, NOISE, seed=100 + seed
            )
            problem = ThermalCorrelationProblem(
                self.nominal,
                DEMO_SENSORS,
                self.tests,
                data,
                noise_sd=NOISE,
                discrepancy_prior=(1.0, 1.0) if discrepancy else None,
            )
            lap = thermal_laplace(problem, tau_grid_size=grid)
            post = lap.sample(200, seed=seed)
            band = predict_flight(post, self.flight, PREDICT_NODES, seed=seed)
            self.assertEqual(band.includes_discrepancy, discrepancy)
            self.assertEqual(band.failed, 0)
            covered += sum(band.covers(self.flight_truth))
            total += len(PREDICT_NODES)
        return covered, total

    def test_band_covers_the_truth_without_model_form_error(self) -> None:
        """Sanity: a correct model gives about nominal coverage."""
        data = simulate_thermal_data(
            self.nominal, THETA_TRUE, DEMO_SENSORS, self.tests, NOISE, seed=3
        )
        problem = ThermalCorrelationProblem(
            self.nominal, DEMO_SENSORS, self.tests, data, noise_sd=NOISE
        )
        post = thermal_laplace(problem).sample(300, seed=0)
        band = predict_flight(post, self.flight, PREDICT_NODES, level=0.95)
        state, _ = self.nominal.steady(THETA_TRUE, self.flight)
        truth = tuple(state.temperatures[i] for i in PREDICT_NODES)
        self.assertTrue(all(band.covers(truth)), (band.lower, band.upper, truth))
        for lo, hi in zip(band.lower, band.upper, strict=True):
            self.assertLess(hi - lo, 8.0)

    def test_discrepancy_restores_coverage(self) -> None:
        """Measured over 8 predeclared seeds at the 90 percent level.

        Full 20-seed measurement: 0.56 without and 0.975 with the discrepancy
        (nominal 0.90; the discrepancy band is conservative).
        """
        seeds = range(8)
        plain, n = self._coverage(seeds, discrepancy=False, grid=9)
        with_disc, _ = self._coverage(seeds, discrepancy=True, grid=9)
        self.assertLess(plain / n, 0.8)
        self.assertGreaterEqual(with_disc / n, 0.85)
        self.assertGreater(with_disc, plain)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_discrepancy_coverage_forty_seeds(self) -> None:
        """Forty seeds; the measured fractions are printed."""
        plain, n = self._coverage(range(40), discrepancy=False, grid=15)
        with_disc, _ = self._coverage(range(40), discrepancy=True, grid=15)
        print(
            f"flight coverage over {n} nodes: plain {plain / n:.3f}, "
            f"disc {with_disc / n:.3f}"
        )
        self.assertLess(plain / n, 0.75)
        self.assertGreaterEqual(with_disc / n, 0.85)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_tmcmc_posterior_predictive_with_discrepancy(self) -> None:
        """TMCMC samples ``tau`` too; its flight band covers the truth."""
        data = simulate_thermal_data(
            self.truth, THETA_TRUE, DEMO_SENSORS, self.tests, NOISE, seed=105
        )
        problem = ThermalCorrelationProblem(
            self.nominal,
            DEMO_SENSORS,
            self.tests,
            data,
            noise_sd=NOISE,
            discrepancy_prior=(1.0, 1.0),
        )
        post = thermal_tmcmc(problem, particles=300, seed=1)
        self.assertTrue(post.converged)
        band = predict_flight(post, self.flight, PREDICT_NODES, level=0.9)
        print(
            "tmcmc tau median", sorted(post.tau_samples())[150], band.lower, band.upper
        )
        self.assertGreaterEqual(sum(band.covers(self.flight_truth)), 3)


class FailureTests(unittest.TestCase):
    """Invalid input and numerical failure are reported, not hidden."""

    def test_network_validation(self) -> None:
        """Bad capacities, links and names are rejected."""
        with self.assertRaises(ValueError):
            ThermalNetwork(("a",), (), (), ())
        with self.assertRaises(ValueError):
            ThermalNetwork(("a", "b"), (0.0,), (), ())
        with self.assertRaises(ValueError):
            ThermalNetwork(("a", "w"), (1.0,), (ThermalLink(0, 0, 1.0),), ())
        with self.assertRaises(ValueError):
            ThermalNetwork(("a", "w"), (1.0,), (ThermalLink(0, 3, 1.0),), ())
        with self.assertRaises(ValueError):
            ThermalNetwork(("a", "w"), (1.0,), (ThermalLink(0, 1, -1.0),), ())
        with self.assertRaises(ValueError):
            ThermalNetwork(("a", "w1", "w2"), (1.0,), (ThermalLink(1, 2, 1.0),), ())
        with self.assertRaises(ValueError):
            ThermalNetwork(("a", "a"), (1.0,), (), ())

    def test_steady_solver_inputs(self) -> None:
        """No boundary node, non-positive temperature and NaN are rejected."""
        net = ThermalNetwork(("a",), (1.0,), (), ())
        with self.assertRaises(ValueError):
            solve_steady_state(net, (1.0,), ())
        two = _two_node("conductive", 1.0)
        with self.assertRaises(ValueError):
            solve_steady_state(two, (1.0,), (-5.0,))
        with self.assertRaises(ValueError):
            solve_steady_state(two, (math.nan,), (300.0,))
        with self.assertRaises(TypeError):
            solve_steady_state("net", (1.0,), (300.0,))  # type: ignore[arg-type]

    def test_no_positive_steady_state_is_reported(self) -> None:
        """A cooling power beyond the radiative limit has no ``T > 0``."""
        net = _two_node("radiative", 1e-8)
        state = solve_steady_state(net, (-1e6,), (100.0,))
        self.assertFalse(state.converged)
        self.assertNotEqual(state.stop_reason, "converged")
        self.assertTrue(all(t > 0.0 for t in state.temperatures))

    def test_iteration_limit_is_reported(self) -> None:
        """A one-iteration cap on a radiative problem stops unconverged."""
        net = _two_node("radiative", 0.8 * STEFAN_BOLTZMANN * 2.0)
        state = solve_steady_state(net, (50.0,), (150.0,), max_iterations=1)
        self.assertFalse(state.converged)
        self.assertEqual(state.stop_reason, "max_iterations")

    def test_transient_validation(self) -> None:
        """Bad sample steps, power callbacks and initial states."""
        net = _two_node("conductive", 1.0)
        power = constant_power((1.0,))
        for bad in (
            TransientTest(power, (300.0,), 1.0, 5, ()),
            TransientTest(power, (300.0,), 1.0, 5, (3, 2)),
            TransientTest(power, (300.0,), 1.0, 5, (6,)),
            TransientTest(power, (300.0,), -1.0, 5, (5,)),
        ):
            with self.assertRaises(ValueError):
                simulate_transient(net, bad, initial=(300.0,))
        ok = TransientTest(power, (300.0,), 1.0, 5, (5,))
        with self.assertRaises(ValueError):
            simulate_transient(net, ok)
        with self.assertRaises(ValueError):
            step_power((1.0,), (1.0, 2.0), 1.0)

    def test_model_parameter_validation(self) -> None:
        """Overlapping groups, bad indices and non-positive theta."""
        net = demo_spacecraft_model().network
        with self.assertRaises(ValueError):
            ThermalModel(net, ())
        with self.assertRaises(ValueError):
            ThermalModel(
                net,
                (
                    ThermalParameter("a", "conductive", (0,)),
                    ThermalParameter("b", "conductive", (0, 1)),
                ),
            )
        with self.assertRaises(ValueError):
            ThermalModel(net, (ThermalParameter("a", "radiative", (9,)),))
        model = demo_spacecraft_model()
        with self.assertRaises(ValueError):
            model.network_at((1.0, 1.0, 1.0, -1.0))
        with self.assertRaises(ValueError):
            model.network_at((1.0, 1.0))

    def test_problem_validation(self) -> None:
        """Data length, sensors, noise and an unsolvable prior median."""
        model = demo_spacecraft_model()
        tests = demo_calibration_tests()
        data = simulate_thermal_data(model, THETA_TRUE, DEMO_SENSORS, tests, 0.1)
        with self.assertRaises(ValueError):
            ThermalCorrelationProblem(
                model, DEMO_SENSORS, tests, data[:-1], noise_sd=0.1
            )
        with self.assertRaises(ValueError):
            ThermalCorrelationProblem(model, (0, 0), tests, data, noise_sd=0.1)
        with self.assertRaises(ValueError):
            ThermalCorrelationProblem(model, (9,), tests, data, noise_sd=0.1)
        with self.assertRaises(ValueError):
            ThermalCorrelationProblem(model, DEMO_SENSORS, tests, data, noise_sd=0.0)
        with self.assertRaises(ValueError):
            ThermalCorrelationProblem(model, DEMO_SENSORS, (), data, noise_sd=0.1)
        impossible = SteadyTest((-1e9, 0.0, 0.0, 0.0, 0.0), (150.0,))
        with self.assertRaises(ThermalSolveError):
            ThermalCorrelationProblem(
                model, DEMO_SENSORS, (impossible,), data[:4], noise_sd=0.1
            )

    def test_laplace_and_prediction_options(self) -> None:
        """Option errors and an unconverged MAP are reported."""
        model = demo_spacecraft_model()
        tests = demo_calibration_tests()
        data = simulate_thermal_data(
            model, THETA_TRUE, DEMO_SENSORS, tests, NOISE, seed=1
        )
        problem = ThermalCorrelationProblem(
            model, DEMO_SENSORS, tests, data, noise_sd=NOISE
        )
        with self.assertRaises(ValueError):
            thermal_laplace(problem, tau=0.5)
        with self.assertRaises(ValueError):
            thermal_laplace(problem, max_iterations=0)
        short = thermal_laplace(problem, max_iterations=1)
        self.assertFalse(short.converged)
        post = thermal_laplace(problem).sample(20, seed=0)
        with self.assertRaises(ValueError):
            predict_flight(post, demo_flight_case(), PREDICT_NODES, level=1.0)
        with self.assertRaises(ValueError):
            predict_flight(post, demo_flight_case(), (7,))
        with self.assertRaises(TypeError):
            predict_flight(post, "flight", PREDICT_NODES)  # type: ignore[arg-type]
        # a flight case with no positive steady state: every sample fails
        impossible = SteadyTest((-1e9, 0.0, 0.0, 0.0, 0.0), (150.0,))
        with self.assertRaises(ValueError):
            predict_flight(post, impossible, PREDICT_NODES)

    def test_seeded_runs_repeat(self) -> None:
        """The same seed gives identical data and posterior draws."""
        model = demo_spacecraft_model()
        tests = demo_calibration_tests()
        a = simulate_thermal_data(model, THETA_TRUE, DEMO_SENSORS, tests, NOISE, seed=4)
        b = simulate_thermal_data(model, THETA_TRUE, DEMO_SENSORS, tests, NOISE, seed=4)
        c = simulate_thermal_data(model, THETA_TRUE, DEMO_SENSORS, tests, NOISE, seed=5)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
