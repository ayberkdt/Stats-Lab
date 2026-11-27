"""Tests of the Bayesian thermocouple design for heat-flux inversion.

Oracles: the posterior covariance ``(C0^-1 + A' A / s^2)^-1`` and the goal
variances from ``probstat.linalg`` dense routines, the equivalence of halved
noise and four replicate readings, a hand-built instance where greedy differs
from the exhaustive optimum, and a Monte Carlo check of the design variances
against the error of ``bayesian_flux_posterior`` over data simulated from the
prior. The slab is the SYNTHETIC ablator of the inverse module, not a real
heat shield.
"""

from __future__ import annotations

import math
import os
import unittest
from dataclasses import replace

from probstat.design.ihcp_sensor_design import (
    IHCPDesignProblem,
    IHCPSensorCandidate,
    build_ihcp_design_problem,
    depth_frequency_study,
    exhaustive_sensor_selection,
    greedy_gap,
    greedy_sensor_selection,
    ihcp_design_criteria,
    ihcp_problem_with,
    optimal_depth_study,
    posterior_error_monte_carlo,
    second_sensor_value,
    sensor_frequency_profile,
)
from probstat.design.linear_optimal import linear_design_criteria
from probstat.inverse.heat_conduction_inverse import (
    SlabFluxOperator,
    synthetic_ablator_slab,
)
from probstat.linalg import inverse_spd, log_det_spd, matmul, matvec, transpose

SLAB = synthetic_ablator_slab()
DEPTHS = (0.001, 0.002, 0.004, 0.007, 0.010)
N_Q = 12
DELTA = 10.0


def _problem(**overrides: object) -> IHCPDesignProblem:
    """Return the fast fixture: 5 depths, 12 intervals of 10 s."""
    options: dict[str, object] = {
        "flux_intervals": N_Q,
        "interval_length": DELTA,
        "noise_std": 5.0,
        "amplitude": 1.0e4,
        "length_scale": 30.0,
    }
    options.update(overrides)
    return build_ihcp_design_problem(SLAB, DEPTHS, **options)  # type: ignore[arg-type]


def _oracle_covariance(problem: IHCPDesignProblem, stacked: list[int]) -> tuple:
    """Return ``(C0^-1 + A' R^-1 A)^-1`` with dense routines."""
    info = inverse_spd(problem.prior_covariance)
    n = len(info)
    rows = [problem.rows[r] for r in stacked]
    weights = [problem.noise_variances[r] for r in stacked]
    scaled = tuple(
        tuple(a / math.sqrt(w) for a in row)
        for row, w in zip(rows, weights, strict=True)
    )
    fisher = matmul(transpose(scaled), scaled)
    total = tuple(tuple(info[i][j] + fisher[i][j] for j in range(n)) for i in range(n))
    return inverse_spd(total)


def _tiny_problem(rows: list[tuple[float, float]]) -> IHCPDesignProblem:
    """Build a two-unknown, three-candidate problem with hand-set rows."""
    base = build_ihcp_design_problem(
        SLAB,
        [0.002, 0.004, 0.006],
        flux_intervals=2,
        interval_length=10.0,
        noise_std=1.0,
        amplitude=1.0,
        length_scale=10.0,
    )
    candidates = tuple(
        IHCPSensorCandidate(d, 1, (i,)) for i, d in enumerate((0.002, 0.004, 0.006))
    )
    return replace(
        base,
        candidates=candidates,
        rows=tuple(rows),
        reference_readings=(0.0, 0.0, 0.0),
        prior_covariance=((1.0, 0.0), (0.0, 1.0)),
        peak_flux_goal=(0.0, 1.0),
        total_heat_goal=(1.0, 1.0),
    )


class ForwardRowsTest(unittest.TestCase):
    """The candidate rows are the rows of the library forward operator."""

    def test_rows_match_operator_with_subsampling(self) -> None:
        """Check: rows match operator with subsampling."""
        problem = _problem(sample_every_options=(1, 3))
        flux = [1.0e4 * (1.0 + 0.3 * math.sin(0.7 * j)) for j in range(N_Q)]
        for candidate in problem.candidates:
            operator = SlabFluxOperator(
                SLAB,
                [candidate.depth],
                flux_intervals=N_Q,
                interval_length=DELTA,
                sample_every=candidate.sample_every,
            )
            expected = operator.matvec(flux)
            got = [
                sum(a * q for a, q in zip(problem.rows[r], flux, strict=True))
                for r in candidate.rows
            ]
            self.assertEqual(len(got), len(expected))
            for a, b in zip(got, expected, strict=True):
                self.assertAlmostEqual(a, b, delta=1e-9 * max(abs(b), 1.0))

    def test_candidate_layout(self) -> None:
        """Check: candidate layout."""
        problem = _problem(sample_every_options=(1, 4))
        self.assertEqual(len(problem.candidates), 2 * len(DEPTHS))
        self.assertEqual(len(problem.rows), len(DEPTHS) * (N_Q + N_Q // 4))
        self.assertEqual(problem.peak_flux_goal.index(1.0), N_Q // 2)
        self.assertEqual(problem.total_heat_goal, (DELTA,) * N_Q)


class CriteriaOracleTest(unittest.TestCase):
    """Posterior covariance, D value and goals against dense oracles."""

    def test_covariance_trace_logdet_and_goals(self) -> None:
        """Check: covariance trace logdet and goals."""
        problem = _problem()
        for chosen in ((0,), (2, 4), (0, 1, 3)):
            stacked = [r for i in chosen for r in problem.candidates[i].rows]
            covariance = _oracle_covariance(problem, stacked)
            evaluation = ihcp_design_criteria(problem, chosen)
            n = N_Q
            trace = sum(covariance[i][i] for i in range(n))
            self.assertAlmostEqual(evaluation.a_optimal, trace, delta=1e-8 * abs(trace))
            goal = problem.total_heat_goal
            heat = sum(
                g * w for g, w in zip(goal, matvec(covariance, goal), strict=True)
            )
            self.assertAlmostEqual(
                evaluation.total_heat_variance, heat, delta=1e-8 * heat
            )
            peak = covariance[N_Q // 2][N_Q // 2]
            self.assertAlmostEqual(
                evaluation.peak_flux_variance, peak, delta=1e-8 * peak
            )
            # D = log det C0 - log det C (information form).
            expected_d = log_det_spd(problem.prior_covariance) - log_det_spd(covariance)
            self.assertAlmostEqual(
                evaluation.d_optimal, expected_d, delta=1e-7 * abs(expected_d)
            )

    def test_empty_design_is_the_prior(self) -> None:
        """Check: empty design is the prior."""
        problem = _problem()
        empty = ihcp_design_criteria(problem, ())
        self.assertAlmostEqual(
            empty.a_optimal, empty.prior_a, delta=1e-9 * empty.prior_a
        )
        self.assertAlmostEqual(empty.d_optimal, 0.0, delta=1e-9)
        self.assertAlmostEqual(empty.reduction("a"), 0.0, delta=1e-9)
        self.assertAlmostEqual(
            empty.total_heat_variance,
            empty.prior_total_heat_variance,
            delta=1e-9 * empty.prior_total_heat_variance,
        )

    def test_halving_noise_equals_four_replicates(self) -> None:
        """Check: halving noise equals four replicates."""
        # Fisher information scales as 1 / sigma^2, so one sensor at sigma / 2
        # is exactly four independent copies of it at sigma.
        problem = _problem(noise_std=6.0)
        halved = ihcp_problem_with(problem, noise_std=3.0)
        one = ihcp_design_criteria(halved, (1,))
        rows = list(problem.candidates[1].rows)
        replicate = linear_design_criteria(
            problem.rows,
            problem.noise_variances,
            rows * 4,
            prior_covariance=problem.prior_covariance,
            goal=problem.total_heat_goal,
        )
        self.assertAlmostEqual(
            one.a_optimal, replicate.a_optimal, delta=1e-8 * replicate.a_optimal
        )
        self.assertAlmostEqual(
            one.d_optimal, replicate.d_optimal, delta=1e-7 * replicate.d_optimal
        )

    def test_relative_noise_matches_row_variances(self) -> None:
        """Check: relative noise matches row variances."""
        reference = [1.0e5] * N_Q
        problem = _problem(relative_noise=0.02, reference_flux=reference)
        for r, row in enumerate(problem.rows):
            reading = abs(sum(a * q for a, q in zip(row, reference, strict=True)))
            expected = 25.0 + (0.02 * reading) ** 2
            self.assertAlmostEqual(
                problem.noise_variances[r], expected, delta=1e-9 * expected
            )
        # The hot (shallow) sensor is noisier than the same sensor without
        # the relative term, a deep one almost unchanged.
        plain = _problem()
        self.assertGreater(
            problem.noise_variances[problem.candidates[0].rows[-1]],
            plain.noise_variances[0],
        )


class MonotonicityTest(unittest.TestCase):
    """Criteria improve with every added sensor and degrade with noise."""

    def test_nested_sets_improve(self) -> None:
        """Check: nested sets improve."""
        problem = _problem()
        order = (3, 0, 4, 1, 2)
        previous = ihcp_design_criteria(problem, ())
        for count in range(1, len(order) + 1):
            current = ihcp_design_criteria(problem, order[:count])
            self.assertLess(current.a_optimal, previous.a_optimal)
            self.assertGreater(current.d_optimal, previous.d_optimal)
            self.assertLess(current.total_heat_variance, previous.total_heat_variance)
            self.assertLess(current.peak_flux_variance, previous.peak_flux_variance)
            previous = current

    def test_greedy_values_are_monotone_with_positive_gains(self) -> None:
        """Check: greedy values are monotone with positive gains."""
        problem = _problem()
        for criterion in ("a", "d", "total_heat", "peak_flux"):
            result = greedy_sensor_selection(problem, 3, criterion=criterion)
            self.assertEqual(len(result.indices), 3)
            self.assertTrue(all(g > 0.0 for g in result.gains))
            steps = (result.prior_value, *result.step_values)
            for first, second in zip(steps, steps[1:], strict=False):
                if criterion == "d":
                    self.assertGreater(second, first)
                else:
                    self.assertLess(second, first)

    def test_more_noise_is_worse_and_huge_noise_returns_the_prior(self) -> None:
        """Check: more noise is worse and huge noise returns the prior."""
        values = []
        for sigma in (1.0, 5.0, 25.0):
            values.append(
                ihcp_design_criteria(
                    ihcp_problem_with(_problem(), noise_std=sigma), (1,)
                )
            )
        self.assertLess(values[0].a_optimal, values[1].a_optimal)
        self.assertLess(values[1].a_optimal, values[2].a_optimal)
        noisy = ihcp_design_criteria(
            ihcp_problem_with(_problem(), noise_std=1.0e8), (0, 1)
        )
        self.assertLess(noisy.reduction("a"), 1e-9)


class SelectionTest(unittest.TestCase):
    """Greedy and exhaustive selection, and their gap."""

    def test_exhaustive_ranking_is_complete_and_sorted(self) -> None:
        """Check: exhaustive ranking is complete and sorted."""
        problem = _problem()
        result = exhaustive_sensor_selection(problem, 2, criterion="a")
        self.assertEqual(len(result.ranking), math.comb(len(DEPTHS), 2))
        values = [v for _, v in result.ranking]
        self.assertEqual(values, sorted(values))
        self.assertEqual(result.indices, result.ranking[0][0])
        recomputed = ihcp_design_criteria(problem, result.indices).a_optimal
        self.assertAlmostEqual(result.value, recomputed, delta=1e-9 * recomputed)

    def test_greedy_matches_exhaustive_on_the_slab_fixture(self) -> None:
        """Check: greedy matches exhaustive on the slab fixture."""
        # Measured on this fixture, not a theorem: the gap is exactly zero.
        problem = _problem()
        for criterion in ("a", "d", "total_heat", "peak_flux"):
            for size in (1, 2, 3):
                gap = greedy_gap(problem, size, criterion=criterion)
                self.assertTrue(gap.same_set, (criterion, size))
                self.assertLess(gap.relative_gap, 1e-12)

    def test_shallowest_sensors_are_optimal_without_constraints(self) -> None:
        """Check: shallowest sensors are optimal without constraints."""
        problem = _problem()
        for size in (1, 2, 3):
            best = exhaustive_sensor_selection(problem, size, criterion="a")
            self.assertEqual(best.depths, DEPTHS[:size])

    def test_survivable_temperature_limit_moves_the_optimum(self) -> None:
        """Check: survivable temperature limit moves the optimum."""
        reference = [1.0e5] * N_Q
        free = _problem(reference_flux=reference)
        hottest = [
            max(
                abs(sum(a * q for a, q in zip(free.rows[r], reference, strict=True)))
                for r in c.rows
            )
            for c in free.candidates
        ]
        limit = 0.5 * (hottest[1] + hottest[2])
        limited = _problem(reference_flux=reference, max_reading=limit)
        self.assertEqual(limited.excluded_depths, DEPTHS[:2])
        self.assertEqual(limited.depths, DEPTHS[2:])
        best = exhaustive_sensor_selection(limited, 2, criterion="a")
        self.assertEqual(best.depths, DEPTHS[2:4])

    def test_greedy_is_not_always_optimal(self) -> None:
        """Check: greedy is not always optimal."""
        # A hand-built instance (found by search) where greedy misses the
        # total-heat optimum: the gap machinery must report it.
        problem = _tiny_problem(
            [
                (-0.7949293619371507, -1.875952994121),
                (1.4621089479157825, -0.10900364533813267),
                (0.8752956962632124, 1.5152512010219268),
            ]
        )
        gap = greedy_gap(problem, 2, criterion="total_heat")
        self.assertFalse(gap.same_set)
        self.assertGreater(gap.relative_gap, 0.02)
        self.assertLessEqual(gap.exhaustive.value, gap.greedy.value)

    def test_greedy_stops_without_gain(self) -> None:
        """Check: greedy stops without gain."""
        problem = _tiny_problem([(1.0, 0.5), (0.0, 0.0), (0.0, 0.0)])
        result = greedy_sensor_selection(problem, 2, criterion="a")
        self.assertEqual(result.indices, (0,))
        self.assertEqual(result.stop_reason, "no_positive_gain")
        with self.assertRaises(ValueError):
            greedy_gap(problem, 2, criterion="a")

    def test_sampling_options_are_candidates_but_depths_stay_distinct(self) -> None:
        """Check: sampling options are candidates but depths stay distinct."""
        problem = _problem(sample_every_options=(1, 4))
        result = greedy_sensor_selection(problem, 2, criterion="a")
        self.assertEqual(len(set(result.depths)), 2)
        # Denser sampling of the same depth is never worse.
        dense = ihcp_design_criteria(problem, (0,))
        sparse = ihcp_design_criteria(problem, (1,))
        self.assertLess(dense.a_optimal, sparse.a_optimal)


class FindingsTest(unittest.TestCase):
    """Studies of noise, length scale, second sensor and frequency content."""

    def test_depth_study_grid(self) -> None:
        """Check: depth study grid."""
        problem = _problem()
        rows = optimal_depth_study(
            problem, (1, 2), (2.0, 20.0), (15.0, 60.0), criterion="a"
        )
        self.assertEqual(len(rows), 8)
        by_cell = {(r.noise_std, r.length_scale, r.size): r for r in rows}
        for ell in (15.0, 60.0):
            for k in (1, 2):
                self.assertLess(
                    by_cell[(2.0, ell, k)].value, by_cell[(20.0, ell, k)].value
                )
        for sigma in (2.0, 20.0):
            for ell in (15.0, 60.0):
                self.assertLess(
                    by_cell[(sigma, ell, 2)].value, by_cell[(sigma, ell, 1)].value
                )
        for row in rows:
            self.assertEqual(row.depths, DEPTHS[: row.size])
            self.assertEqual(row.relative_gap, 0.0)

    def test_second_sensor_versus_half_noise(self) -> None:
        """Check: second sensor versus half noise."""
        problem = _problem(noise_std=10.0)
        for criterion in ("a", "d", "total_heat", "peak_flux"):
            result = second_sensor_value(problem, criterion=criterion)
            if criterion == "d":
                self.assertGreater(result.one_half_noise, result.two)
                self.assertGreater(result.two, result.one)
            else:
                self.assertLess(result.one_half_noise, result.two)
                self.assertLess(result.two, result.one)
            # Measured: halving the noise of the best sensor beats adding the
            # best second sensor on every criterion of this fixture.
            self.assertEqual(result.winner, "half_noise")

    def test_deep_sensors_resolve_fewer_flux_modes(self) -> None:
        """Check: deep sensors resolve fewer flux modes."""
        problem = _problem(noise_std=10.0)
        profiles = depth_frequency_study(problem)
        self.assertEqual(len(profiles), len(DEPTHS))
        for profile in profiles:
            self.assertTrue(all(-1e-9 <= r <= 1.0 + 1e-9 for r in profile.resolved))
            self.assertEqual(
                list(profile.prior_variances), sorted(profile.prior_variances)[::-1]
            )
            self.assertGreater(profile.low_resolved, profile.high_resolved)
        cutoffs = [p.cutoff_modes for p in profiles]
        self.assertEqual(cutoffs, sorted(cutoffs, reverse=True))
        self.assertGreater(cutoffs[0], cutoffs[-1])
        high = [p.high_resolved for p in profiles]
        self.assertEqual(high, sorted(high, reverse=True))
        # Shallow sensors resolve the peak better, deep ones lose it first.
        peak = [p.peak_flux_ratio for p in profiles]
        self.assertEqual(peak, sorted(peak))
        heat = [p.total_heat_ratio for p in profiles]
        self.assertEqual(heat, sorted(heat))

    def test_profile_of_a_set_dominates_each_member(self) -> None:
        """Check: profile of a set dominates each member."""
        problem = _problem(noise_std=10.0)
        both = sensor_frequency_profile(problem, (0, 4))
        for index in (0, 4):
            single = sensor_frequency_profile(problem, (index,))
            for a, b in zip(both.resolved, single.resolved, strict=True):
                self.assertGreaterEqual(a, b - 1e-9)


class MonteCarloTest(unittest.TestCase):
    """Design variances equal the empirical error variances of the estimator."""

    def _assert_within(self, check: object, z: float = 4.5) -> None:
        tolerance = z * check.relative_standard_error  # type: ignore[attr-defined]
        self.assertLess(check.max_reported_deviation, 1e-8)  # type: ignore[attr-defined]
        self.assertLess(check.max_relative_deviation, tolerance)  # type: ignore[attr-defined]

    def test_prior_simulation_matches_design_variances(self) -> None:
        """Check: prior simulation matches design variances."""
        # 4.5 sigma_rel with sigma_rel = sqrt(2 / N) covers all N_Q + 2 checked
        # quantities with probability above 1 - 1e-4 (fixed seed, so exact).
        problem = _problem(noise_std=8.0)
        check = posterior_error_monte_carlo(problem, (0, 3), replicates=300, seed=11)
        self._assert_within(check)
        self.assertAlmostEqual(
            check.design_total_heat,
            ihcp_design_criteria(problem, (0, 3)).total_heat_variance,
            delta=1e-9 * check.design_total_heat,
        )

    def test_heteroscedastic_noise_matches(self) -> None:
        """Check: heteroscedastic noise matches."""
        problem = _problem(
            noise_std=4.0, relative_noise=0.03, reference_flux=[1.0e5] * N_Q
        )
        check = posterior_error_monte_carlo(problem, (1, 2), replicates=300, seed=5)
        self._assert_within(check)

    def test_the_check_detects_a_wrong_design(self) -> None:
        """Check: the check detects a wrong design."""
        # Negative control: variances designed for one third of the noise are
        # far outside the band, so the tolerance above has power.
        problem = _problem(noise_std=8.0)
        check = posterior_error_monte_carlo(problem, (0, 3), replicates=300, seed=11)
        wrong = ihcp_design_criteria(
            ihcp_problem_with(problem, noise_std=8.0 / 3.0), (0, 3)
        )
        ratio = check.empirical_total_heat / wrong.total_heat_variance
        self.assertGreater(abs(ratio - 1.0), 4.5 * check.relative_standard_error)


class FailureTest(unittest.TestCase):
    """Invalid input is rejected; nothing is repaired."""

    def test_builder_rejects_bad_arguments(self) -> None:
        """Check: builder rejects bad arguments."""
        base = {
            "flux_intervals": N_Q,
            "interval_length": DELTA,
            "noise_std": 5.0,
            "amplitude": 1.0e4,
            "length_scale": 30.0,
        }

        def build(depths: object = DEPTHS, **changes: object) -> object:
            return build_ihcp_design_problem(
                SLAB,
                depths,  # type: ignore[arg-type]
                **{**base, **changes},  # type: ignore[arg-type]
            )

        for changes in (
            {"noise_std": 0.0},
            {"noise_std": -1.0},
            {"length_scale": 0.0},
            {"amplitude": -2.0},
            {"flux_intervals": 0},
            {"interval_length": 0.0},
            {"sample_every_options": (5,)},
            {"sample_every_options": ()},
            {"sample_every_options": (2, 2)},
            {"peak_time": 1.0e6},
            {"relative_noise": 0.1},
            {"relative_noise": -0.1, "reference_flux": [1.0] * N_Q},
            {"reference_flux": [1.0] * (N_Q - 1)},
            {"max_reading": 100.0},
            {"smoothness": 2.0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build(**changes)
        with self.assertRaises(ValueError):
            build(depths=(0.001, 0.001))
        with self.assertRaises(ValueError):
            build(depths=())
        with self.assertRaises(ValueError):
            build(depths=(0.0,))  # above the surface cell centre
        with self.assertRaises(ValueError):
            build(depths=(0.0199,))  # beyond the last cell centre
        with self.assertRaises(TypeError):
            build(depths="deep")
        with self.assertRaises(TypeError):
            build(noise_std="5")
        with self.assertRaises(ValueError):
            build(reference_flux=[1.0] * N_Q, max_reading=1.0e-3)

    def test_problem_with_rejects_bad_arguments(self) -> None:
        """Check: problem with rejects bad arguments."""
        problem = _problem()
        with self.assertRaises(TypeError):
            ihcp_problem_with("problem")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ihcp_problem_with(problem, noise_std=-1.0)
        with self.assertRaises(ValueError):
            ihcp_problem_with(problem, relative_noise=0.1)
        with self.assertRaises(ValueError):
            ihcp_problem_with(problem, length_scale=0.0)

    def test_selection_rejects_bad_arguments(self) -> None:
        """Check: selection rejects bad arguments."""
        problem = _problem()
        with self.assertRaises(ValueError):
            ihcp_design_criteria(problem, (0, 0))
        with self.assertRaises(ValueError):
            ihcp_design_criteria(problem, (len(DEPTHS),))
        with self.assertRaises(TypeError):
            ihcp_design_criteria(problem, (True,))
        with self.assertRaises(TypeError):
            ihcp_design_criteria("problem", (0,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            greedy_sensor_selection(problem, 2, criterion="e")
        with self.assertRaises(ValueError):
            greedy_sensor_selection(problem, 0)
        with self.assertRaises(ValueError):
            greedy_sensor_selection(problem, len(DEPTHS) + 1)
        with self.assertRaises(ValueError):
            exhaustive_sensor_selection(problem, 3, max_combinations=5)
        with self.assertRaises(ValueError):
            exhaustive_sensor_selection(problem, 2, criterion="goal")
        with self.assertRaises(ValueError):
            exhaustive_sensor_selection(problem, len(DEPTHS) + 1)
        with self.assertRaises(TypeError):
            greedy_sensor_selection(problem, 1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sensor_frequency_profile(problem, ())
        with self.assertRaises(ValueError):
            posterior_error_monte_carlo(problem, ())
        with self.assertRaises(ValueError):
            posterior_error_monte_carlo(problem, (0,), replicates=1)
        with self.assertRaises(ValueError):
            optimal_depth_study(problem, (), (1.0,), (10.0,))
        # Two candidates at one depth cannot form a pair of distinct depths.
        same = _problem(sample_every_options=(1, 2))
        with self.assertRaises(ValueError):
            exhaustive_sensor_selection(
                replace(same, candidates=same.candidates[:2]), 2, criterion="a"
            )
        with self.assertRaises(ValueError):
            greedy_sensor_selection(
                replace(same, candidates=same.candidates[:2]), 2, criterion="a"
            )

    def test_unknown_criterion_value_lookup(self) -> None:
        """Check: unknown criterion value lookup."""
        evaluation = ihcp_design_criteria(_problem(), (0,))
        with self.assertRaises(ValueError):
            evaluation.value("trace")
        with self.assertRaises(ValueError):
            evaluation.reduction("trace")


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudyTest(unittest.TestCase):
    """Full grids of the physical findings; the numbers are printed."""

    def test_full_depth_grid_and_large_monte_carlo(self) -> None:
        """Check: full depth grid and large monte carlo."""
        depths = (0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010)
        problem = build_ihcp_design_problem(
            SLAB,
            depths,
            flux_intervals=16,
            interval_length=DELTA,
            noise_std=5.0,
            amplitude=1.0e4,
            length_scale=30.0,
        )
        for criterion in ("a", "d", "total_heat", "peak_flux"):
            rows = optimal_depth_study(
                problem,
                (1, 2, 3),
                (1.0, 10.0, 100.0),
                (5.0, 20.0, 80.0),
                criterion=criterion,
            )
            worst = max(r.relative_gap for r in rows)
            print(f"[slow] {criterion}: cells={len(rows)} max greedy gap={worst:.3e}")
            for row in rows:
                self.assertEqual(row.depths, depths[: row.size])
        check = posterior_error_monte_carlo(problem, (0, 4), replicates=2000, seed=2)
        print(
            f"[slow] MC N=2000 max rel dev={check.max_relative_deviation:.4f} "
            f"band={4.5 * check.relative_standard_error:.4f}"
        )
        self.assertLess(
            check.max_relative_deviation, 4.5 * check.relative_standard_error
        )


if __name__ == "__main__":
    unittest.main()
