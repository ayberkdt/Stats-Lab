"""Tests for multifidelity cost accounting, active surrogates and multilevel MCMC.

Oracles: exact arithmetic for declared costs; a Gaussian-vs-Gaussian KL in
closed form; the exact 1D posterior on a fine grid for the active surrogate;
closed-form posterior means of every level of a linear-Gaussian hierarchy for
the multilevel chains. Stochastic checks use fixed seeds and MCSE bands.
"""

from __future__ import annotations

import math
import os
import random
import time
import unittest
import warnings

from probstat.inference.delayed_acceptance import delayed_acceptance_metropolis
from probstat.inverse.approximation_error import estimate_approximation_error
from probstat.inverse.gauss_newton import CallbackForward
from probstat.inverse.multifidelity import (
    CostModel,
    active_surrogate_posterior,
    approximation_error_cost,
    coarse_to_fine_warm_start,
    cost_normalised_ess,
    declare_cost_model,
    delayed_acceptance_cost,
    extrapolation_report,
    fine_equivalent_cost,
    grid_posterior_error,
    measure_cost_model,
    multilevel_mcmc,
    multilevel_sample_allocation,
    surrogate_mean_forward,
    validate_sensitivities,
)
from probstat.inverse.pde_models import EllipticDiffusion1D
from probstat.linalg import matvec, solve_spd, transpose
from probstat.surrogates.gaussian_process import fit_gaussian_process

SLOW = os.environ.get("PROBSTAT_SLOW") == "1"


# ---------------------------------------------------------------------------
# cost accounting
# ---------------------------------------------------------------------------


class CostAccountingTests(unittest.TestCase):
    """Fine-equivalent cost, cost-normalised ESS, measured and declared costs."""

    def test_declared_costs_exact_arithmetic(self) -> None:
        """C = n_f + n_c c_c / c_f with exactly representable ratios."""
        model = declare_cost_model(1.0, 4.0)
        self.assertEqual(model.ratio, 0.25)
        self.assertEqual(model.source, "declared")
        self.assertEqual(fine_equivalent_cost(10, 40, model), 20.0)
        self.assertEqual(fine_equivalent_cost(0, 0, model), 0.0)
        self.assertEqual(cost_normalised_ess(30.0, 20.0), 1.5)

    def test_measured_costs_with_a_deterministic_clock(self) -> None:
        """The median of single-call timings gives the exact cost ratio."""
        clock = [0.0]

        def timer() -> float:
            return clock[0]

        def fine(x: object) -> None:
            clock[0] += 4.0

        def coarse(x: object) -> None:
            clock[0] += 1.0

        model = measure_cost_model(
            fine, coarse, [(0.0,), (1.0,)], repeats=3, timer=timer
        )
        self.assertEqual(model.source, "measured")
        self.assertEqual((model.coarse_cost, model.fine_cost), (1.0, 4.0))
        self.assertIn("median", model.method)
        self.assertIn("6 single-call timings", model.method)

    def test_measured_cost_uses_the_median_not_the_mean(self) -> None:
        """One slow outlier call does not move the cost."""
        clock = [0.0]
        calls = {"n": 0}

        def timer() -> float:
            return clock[0]

        def fine(x: object) -> None:
            calls["n"] += 1
            clock[0] += 100.0 if calls["n"] == 1 else 2.0

        def coarse(x: object) -> None:
            clock[0] += 1.0

        model = measure_cost_model(fine, coarse, [(0.0,)], repeats=5, timer=timer)
        self.assertEqual(model.fine_cost, 2.0)

    def test_invalid_cost_inputs_are_rejected(self) -> None:
        """Failure cases: bad costs, counts, repeats and timer resolution."""
        with self.assertRaises(ValueError):
            declare_cost_model(5.0, 1.0)  # coarse more expensive than fine
        with self.assertRaises(ValueError):
            declare_cost_model(0.0, 1.0)
        with self.assertRaises(ValueError):
            CostModel(1.0, 2.0, "guessed", "x")
        model = declare_cost_model(1.0, 2.0)
        with self.assertRaises(ValueError):
            fine_equivalent_cost(-1, 0, model)
        with self.assertRaises(TypeError):
            fine_equivalent_cost(1.5, 0, model)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cost_normalised_ess(1.0, 0.0)
        with self.assertRaises(ValueError):
            measure_cost_model(len, len, [], repeats=1)
        with self.assertRaises(TypeError):
            measure_cost_model(len, len, [(0.0,)], repeats=1.5)  # type: ignore[arg-type]
        # A frozen clock gives zero timings: the cost is refused, not guessed.
        with self.assertRaises(ValueError):
            measure_cost_model(
                lambda x: None, lambda x: None, [(0.0,)], timer=lambda: 1.0
            )

    def test_delayed_acceptance_cost_matches_the_result_counts(self) -> None:
        """Fine-equivalent cost and ESS per unit follow from the counts."""

        def exact(x: tuple[float, ...]) -> float:
            return -0.5 * x[0] ** 2

        def cheap(x: tuple[float, ...]) -> float:
            return -0.5 * (x[0] - 0.2) ** 2 / 1.1

        result = delayed_acceptance_metropolis(
            exact, cheap, (0.0,), step=1.5, samples=600, burn_in=100, seed=4
        )
        model = declare_cost_model(1.0, 20.0)
        report = delayed_acceptance_cost(result, model)
        expected = result.exact_evaluations + result.surrogate_evaluations / 20.0
        self.assertAlmostEqual(report.fine_equivalent_cost, expected, places=12)
        self.assertEqual(report.fine_evaluations, result.exact_evaluations)
        self.assertEqual(report.coarse_evaluations, result.surrogate_evaluations)
        ess = min(e for e in result.effective_sample_size if math.isfinite(e))
        self.assertAlmostEqual(report.ess_per_fine_equivalent, ess / expected, 12)
        self.assertTrue(math.isnan(report.relative_efficiency))
        # Baseline: a plain Metropolis chain evaluates the fine target on
        # every proposal (constant surrogate accepts at stage 1).
        plain = delayed_acceptance_metropolis(
            exact, lambda x: 0.0, (0.0,), step=1.5, samples=600, burn_in=100, seed=4
        )
        self.assertEqual(plain.exact_evaluations, 701)
        base = min(e for e in plain.effective_sample_size if math.isfinite(e))
        base /= plain.exact_evaluations
        relative = delayed_acceptance_cost(
            result, model, baseline_ess_per_fine_evaluation=base
        ).relative_efficiency
        self.assertAlmostEqual(relative, report.ess_per_fine_equivalent / base, 12)
        with self.assertRaises(ValueError):
            delayed_acceptance_cost(result, model, baseline_ess_per_fine_evaluation=0.0)

    def test_approximation_error_setup_cost_and_break_even(self) -> None:
        """Setup, total, saving and break-even from declared costs."""
        rng = random.Random(1)
        points = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(20)]

        def fine(x: tuple[float, ...]) -> tuple[float, float]:
            return (x[0] + x[1] + 0.1 * x[0] ** 2, x[0] - x[1])

        def coarse(x: tuple[float, ...]) -> tuple[float, float]:
            return (x[0] + x[1], 0.9 * (x[0] - x[1]))

        error_model = estimate_approximation_error(fine, coarse, points)
        self.assertEqual(error_model.fine_evaluations, 20)
        report = approximation_error_cost(
            error_model, declare_cost_model(1.0, 5.0), posterior_evaluations=100
        )
        self.assertAlmostEqual(report.setup_cost, 24.0, places=12)
        self.assertAlmostEqual(report.posterior_cost, 20.0, places=12)
        self.assertAlmostEqual(report.total_cost, 44.0, places=12)
        self.assertAlmostEqual(report.fine_only_cost, 100.0, places=12)
        self.assertAlmostEqual(report.saving, 56.0, places=12)
        self.assertAlmostEqual(report.break_even_evaluations, 30.0, places=10)
        # Below the break-even count the setup does not pay off.
        few = approximation_error_cost(
            error_model, declare_cost_model(1.0, 5.0), posterior_evaluations=10
        )
        self.assertLess(few.saving, 0.0)
        # Equal costs: never pays off.
        same = approximation_error_cost(
            error_model, declare_cost_model(2.0, 2.0), posterior_evaluations=10
        )
        self.assertEqual(same.break_even_evaluations, math.inf)
        with self.assertRaises(ValueError):
            approximation_error_cost(
                error_model, declare_cost_model(1.0, 5.0), posterior_evaluations=-1
            )


# ---------------------------------------------------------------------------
# active surrogate refinement
# ---------------------------------------------------------------------------

Y1, SIG1 = 1.2, 0.3


def _logp_1d(x: tuple[float, ...]) -> float:
    """Posterior of a cubic forward map with a standard normal prior."""
    forward = x[0] + 0.3 * x[0] ** 3
    return -0.5 * x[0] ** 2 - 0.5 * (Y1 - forward) ** 2 / SIG1**2


def _logp_2d(x: tuple[float, ...]) -> float:
    """Posterior of a weakly nonlinear 2D forward map."""
    g0 = x[0] + 0.5 * x[1] ** 2
    g1 = x[1] - 0.3 * x[0]
    misfit = (0.8 - g0) ** 2 + (0.3 - g1) ** 2
    return -0.5 * (x[0] ** 2 + x[1] ** 2) - 0.5 * misfit / 0.3**2


def _gp_fit(points: object, values: object) -> object:
    """Library GP, fixed seed; the injected surrogate of the tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_gaussian_process(
            points,  # type: ignore[arg-type]
            values,  # type: ignore[arg-type]
            kernel="matern52",
            seed=0,
            restarts=2,
        )


AXIS_1D = [-4.0 + 8.0 * i / 160 for i in range(161)]
CANDIDATES_1D = [(v,) for v in AXIS_1D[::2]]
DESIGN_1D = [(-3.0,), (0.0,), (3.0,)]


class ActiveSurrogateTests(unittest.TestCase):
    """Active refinement of a GP surrogate of a log-posterior."""

    def _run_1d(self, budget: int, **kwargs: object) -> object:
        counter = {"n": 0}

        def counted(x: tuple[float, ...]) -> float:
            counter["n"] += 1
            return _logp_1d(x)

        result = active_surrogate_posterior(
            counted,
            _gp_fit,  # type: ignore[arg-type]
            DESIGN_1D,
            CANDIDATES_1D,
            max_fine_evaluations=budget,
            **kwargs,  # type: ignore[arg-type]
        )
        self.assertEqual(counter["n"], result.fine_evaluations)
        return result

    def test_error_decreases_with_fine_runs_against_the_exact_posterior(self) -> None:
        """KL and moment errors versus the number of fine runs, measured."""
        kls = {}
        for budget in (4, 8, 12):
            result = self._run_1d(budget)
            self.assertEqual(result.stop_reason, "budget")
            self.assertFalse(result.converged)
            self.assertEqual(result.fine_evaluations, budget)
            err = grid_posterior_error(_logp_1d, result.log_density, [AXIS_1D])
            kls[budget] = err
        self.assertGreater(kls[4].kl_divergence, 10.0)  # 4 runs: far off
        self.assertLess(kls[8].kl_divergence, kls[4].kl_divergence)
        self.assertLess(kls[12].kl_divergence, kls[8].kl_divergence)
        self.assertLess(kls[12].kl_divergence, 5e-3)  # measured 1.0e-3
        self.assertLess(kls[12].mean_error[0], 1e-2)
        self.assertLess(kls[12].std_error[0], 1e-2)

    def test_refinement_goes_where_the_posterior_is(self) -> None:
        """The added points concentrate near the posterior mode, not the box."""
        result = self._run_1d(10)
        added = [x[0] for x in result.design[3:]]
        self.assertEqual(len(added), 7)
        mean = grid_posterior_error(_logp_1d, _logp_1d, [AXIS_1D]).mean_exact[0]
        self.assertTrue(all(abs(a - mean) < 2.5 for a in added))

    def test_stopping_rules_are_reported_honestly(self) -> None:
        """Accuracy stops are converged, a budget stop is not."""
        by_validation = self._run_1d(
            30,
            validation_points=[(0.55,), (0.75,), (0.95,)],
            validation_tolerance=0.05,
        )
        self.assertEqual(by_validation.stop_reason, "validation_tolerance")
        self.assertTrue(by_validation.converged)
        self.assertLess(by_validation.history[-1].validation_rmse, 0.05)
        self.assertEqual(by_validation.validation_evaluations, 3)
        self.assertEqual(
            by_validation.fine_evaluations,
            by_validation.training_evaluations + 3,
        )
        # Held-out points never enter the design.
        self.assertFalse(set(by_validation.design) & {(0.55,), (0.75,), (0.95,)})
        by_acquisition = self._run_1d(30, acquisition_tolerance=0.05)
        self.assertEqual(by_acquisition.stop_reason, "acquisition_tolerance")
        self.assertTrue(by_acquisition.converged)
        self.assertLess(by_acquisition.fine_evaluations, 30)
        self.assertLessEqual(by_acquisition.history[-1].max_acquisition, 0.05)
        tiny = active_surrogate_posterior(
            _logp_1d,
            _gp_fit,  # type: ignore[arg-type]
            DESIGN_1D,
            [(-1.0,), (1.0,)],
            max_fine_evaluations=10,
        )
        self.assertEqual(tiny.stop_reason, "candidates_exhausted")
        self.assertFalse(tiny.converged)
        self.assertEqual(tiny.training_evaluations, 5)

    def test_two_dimensional_error_decreases(self) -> None:
        """2D: KL and moments against the exact posterior on a grid."""
        axis = [-3.0 + 6.0 * i / 40 for i in range(41)]
        candidates = [(a, b) for a in axis[::2] for b in axis[::2]]
        design = [(-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (0.0, 0.0)]
        kls = {}
        for budget in (10, 20):
            result = active_surrogate_posterior(
                _logp_2d,
                _gp_fit,  # type: ignore[arg-type]
                design,
                candidates,
                max_fine_evaluations=budget,
            )
            kls[budget] = grid_posterior_error(
                _logp_2d, result.log_density, [axis, axis]
            )
        self.assertLess(kls[20].kl_divergence, kls[10].kl_divergence)
        self.assertLess(kls[20].kl_divergence, 0.05)
        self.assertTrue(all(e < 0.05 for e in kls[20].mean_error))

    def test_extrapolation_is_reported_where_the_design_does_not_reach(self) -> None:
        """Query points outside the design box or far from it are flagged."""
        design = [(0.0,), (1.0,), (2.0,)]
        report = extrapolation_report(design, [(1.5,), (-0.5,), (2.0,), (5.0,)])
        self.assertEqual(report.largest_spacing, 1.0)
        self.assertEqual(report.outside_box, (False, True, False, True))
        self.assertEqual(report.extrapolating, (False, True, False, True))
        self.assertEqual(report.weighted_fraction, 0.5)
        weighted = extrapolation_report(design, [(1.5,), (5.0,)], weights=[3.0, 1.0])
        self.assertEqual(weighted.weighted_fraction, 0.25)
        # Inside the box but in a gap wider than the design spacing: flagged.
        two_clusters = [(0.0,), (0.1,), (0.2,), (1.0,), (1.1,), (1.2,)]
        gap = extrapolation_report(two_clusters, [(0.6,), (0.15,)])
        self.assertEqual(gap.extrapolating, (True, False))
        self.assertEqual(gap.outside_box, (False, False))

    def test_extrapolating_posterior_mass_is_visible(self) -> None:
        """A design away from the posterior mode flags its mass as extrapolated."""
        result = active_surrogate_posterior(
            _logp_1d,
            _gp_fit,  # type: ignore[arg-type]
            [(-3.5,), (-3.0,), (-2.5,)],
            [(-3.0 + 0.2 * i,) for i in range(5)],
            max_fine_evaluations=6,
        )
        weights = [math.exp(_logp_1d((v,))) for v in AXIS_1D]
        report = result.extrapolation([(v,) for v in AXIS_1D], weights=weights)
        self.assertGreater(report.weighted_fraction, 0.9)

    def test_failure_cases(self) -> None:
        """Invalid inputs, infeasible values and failing fits are refused."""

        def run(**overrides: object) -> object:
            options: dict[str, object] = {
                "log_target": _logp_1d,
                "fit": _gp_fit,
                "initial_design": DESIGN_1D,
                "candidates": CANDIDATES_1D,
                "max_fine_evaluations": 8,
            }
            options.update(overrides)
            return active_surrogate_posterior(**options)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            run(max_fine_evaluations=2)  # budget below the initial design
        with self.assertRaises(ValueError):
            run(initial_design=[(0.0,)])
        with self.assertRaises(ValueError):
            run(initial_design=[(0.0,), (0.0,)])
        with self.assertRaises(ValueError):
            run(validation_points=[(0.0,)], validation_tolerance=0.1)  # in design
        with self.assertRaises(ValueError):
            run(validation_tolerance=0.1)  # no validation points
        with self.assertRaises(ValueError):
            run(acquisition_tolerance=-1.0)
        with self.assertRaises(ValueError):
            run(initial_design=[(0.0,), (1.0, 2.0)])
        with self.assertRaises(TypeError):
            run(max_fine_evaluations=8.0)
        with self.assertRaises(TypeError):
            run(fit="not callable")
        with self.assertRaises(ValueError):
            run(log_target=lambda x: -math.inf)  # outside the support
        with self.assertRaises(ValueError):
            run(log_target=lambda x: math.nan)

        def failing_fit(points: object, values: object) -> object:
            raise ValueError("kernel matrix is singular")

        with self.assertRaisesRegex(ValueError, "surrogate fit failed"):
            run(fit=failing_fit)


class GridPosteriorErrorTests(unittest.TestCase):
    """Grid KL and moment errors against closed forms."""

    def test_gaussian_shift_has_closed_form_kl(self) -> None:
        """KL(N(0,1) || N(m,1)) = m^2/2; moment errors are m and 0."""
        axis = [-9.0 + 18.0 * i / 900 for i in range(901)]
        shift = 0.6
        err = grid_posterior_error(
            lambda x: -0.5 * x[0] ** 2,
            lambda x: -0.5 * (x[0] - shift) ** 2 + 7.0,  # constant is irrelevant
            [axis],
        )
        self.assertAlmostEqual(err.kl_divergence, shift**2 / 2, places=5)
        self.assertAlmostEqual(err.mean_error[0], shift, places=6)
        self.assertAlmostEqual(err.std_error[0], 0.0, places=6)
        self.assertAlmostEqual(err.std_exact[0], 1.0, places=4)
        same = grid_posterior_error(
            lambda x: -0.5 * x[0] ** 2, lambda x: -0.5 * x[0] ** 2, [axis]
        )
        self.assertEqual(same.kl_divergence, 0.0)

    def test_scale_error_has_closed_form_kl(self) -> None:
        """KL(N(0,1) || N(0,s^2)) = log s + (1/s^2 - 1)/2 in 2D (twice)."""
        axis = [-7.0 + 14.0 * i / 140 for i in range(141)]
        s = 1.4
        err = grid_posterior_error(
            lambda x: -0.5 * (x[0] ** 2 + x[1] ** 2),
            lambda x: -0.5 * (x[0] ** 2 + x[1] ** 2) / s**2,
            [axis, axis],
        )
        expected = 2 * (math.log(s) + 0.5 * (1.0 / s**2 - 1.0))
        self.assertAlmostEqual(err.kl_divergence, expected, places=4)
        self.assertAlmostEqual(err.std_surrogate[0], s, places=3)

    def test_invalid_grids_and_densities(self) -> None:
        """Failure cases: bad axes, zero-mass surrogate, NaN densities."""
        axis = [0.0, 1.0, 2.0]
        zero_there = lambda x: -math.inf if x[0] == 1.0 else 0.0  # noqa: E731
        with self.assertRaises(ValueError):
            grid_posterior_error(lambda x: 0.0, lambda x: 0.0, [[0.0]])
        with self.assertRaises(ValueError):
            grid_posterior_error(lambda x: 0.0, lambda x: 0.0, [[0.0, 0.0, 1.0]])
        with self.assertRaises(ValueError):
            grid_posterior_error(lambda x: 0.0, lambda x: 0.0, [])
        with self.assertRaises(ValueError):
            grid_posterior_error(lambda x: 0.0, zero_there, [axis])
        with self.assertRaises(ValueError):
            grid_posterior_error(lambda x: math.nan, lambda x: 0.0, [axis])
        with self.assertRaises(ValueError):
            grid_posterior_error(lambda x: -math.inf, lambda x: 0.0, [axis])
        # Exact density zero somewhere is allowed (contributes nothing).
        ok = grid_posterior_error(zero_there, lambda x: 0.0, [axis])
        self.assertTrue(math.isfinite(ok.kl_divergence))


# ---------------------------------------------------------------------------
# multilevel MCMC on a linear-Gaussian hierarchy
# ---------------------------------------------------------------------------

DATA, PRIOR_SD, NOISE_SD = 1.3, 2.0, 0.5
SLOPES = (0.8, 0.93, 0.98, 1.0)  # forward maps a_l x -> a_inf x = x


def _level_mean(slope: float) -> float:
    """Exact posterior mean of x | y = a x + noise, x ~ N(0, PRIOR_SD^2)."""
    precision = 1.0 / PRIOR_SD**2 + slope**2 / NOISE_SD**2
    return slope * DATA / NOISE_SD**2 / precision


def _level_density(slope: float):  # type: ignore[no-untyped-def]
    """Return log pi_l (up to a constant) for forward slope ``a_l``."""
    return lambda x: (
        -0.5 * x[0] ** 2 / PRIOR_SD**2 - 0.5 * (DATA - slope * x[0]) ** 2 / NOISE_SD**2
    )


def _identity(x: tuple[float, ...]) -> float:
    return x[0]


class MultilevelMCMCTests(unittest.TestCase):
    """Telescoping estimator with coupled chains against closed-form levels."""

    def _run(self, levels: int, **overrides: object):  # type: ignore[no-untyped-def]
        options: dict[str, object] = {
            "samples": [4000, 4000, 3000][:levels],
            "step": 1.0,
            "burn_in": 100,
            "subsampling": [25, 4][: levels - 1],
            "costs": [1.0, 3.0, 10.0][:levels],
            "seed": 1,
        }
        options.update(overrides)
        return multilevel_mcmc(
            [_level_density(a) for a in SLOPES[:levels]],
            _identity,
            (0.0,),
            **options,  # type: ignore[arg-type]
        )

    def test_two_levels_recover_each_exact_level_and_the_total(self) -> None:
        """Each correction matches m_l - m_{l-1} within MCSE; sum matches m_1."""
        result = self._run(2)
        exact = [_level_mean(a) for a in SLOPES[:2]]
        level0, level1 = result.levels
        for level, want in ((level0, exact[0]), (level1, exact[1] - exact[0])):
            self.assertLess(abs(level.mean - want), 4.5 * level.mcse)
        self.assertLess(abs(result.estimate - exact[1]), 4.5 * result.standard_error)
        # The coarse proposals are a stationary pi_0 chain: E Q_0(z) = m_0.
        self.assertLess(abs(level1.coarse_mean - exact[0]), 4.5 * level0.mcse)
        # The test has power: the level-0 bias is many standard errors.
        self.assertGreater(abs(exact[0] - exact[1]), 8.0 * result.standard_error)
        self.assertAlmostEqual(
            result.standard_error, math.hypot(level0.mcse, level1.mcse), places=14
        )
        self.assertAlmostEqual(result.estimate, level0.mean + level1.mean, places=14)

    def test_coupling_reduces_the_variance_of_the_corrections(self) -> None:
        """Level variances fall because accepted proposals give Y = 0 exactly."""
        result = self._run(2)
        v0, v1 = (r.variance for r in result.levels)
        self.assertLess(v1, 0.75 * v0)
        self.assertGreater(result.levels[1].acceptance_rate, 0.6)

    def test_identical_levels_give_zero_variance_corrections(self) -> None:
        """pi_1 = pi_0: every proposal is accepted, Y = 0, level 1 adds nothing."""
        density = _level_density(0.93)
        result = multilevel_mcmc(
            [density, density],
            _identity,
            (0.0,),
            samples=500,
            step=1.0,
            burn_in=50,
            subsampling=10,
            seed=3,
        )
        level1 = result.levels[1]
        self.assertEqual(level1.acceptance_rate, 1.0)
        self.assertEqual(level1.variance, 0.0)
        self.assertEqual(level1.mean, 0.0)
        self.assertEqual(level1.mcse, 0.0)
        self.assertTrue(math.isnan(level1.effective_sample_size))
        self.assertEqual(result.estimate, result.levels[0].mean)
        self.assertEqual(result.standard_error, result.levels[0].mcse)
        self.assertEqual(multilevel_sample_allocation(result, 1e-3)[1], 0)

    def test_evaluation_counts_and_cost_in_fine_equivalent_units(self) -> None:
        """Counts are exact: initial value + one per step; costs by declaration."""
        result = self._run(2, samples=[200, 200], burn_in=20, subsampling=[7])
        steps = 20 + 200
        level0, level1 = result.levels
        self.assertEqual(level0.evaluations, (1 + steps,))
        self.assertEqual(level1.evaluations, (1 + 7 * steps, 1 + steps))
        self.assertAlmostEqual(level0.cost, (1 + steps) * 1.0 / 3.0, places=12)
        self.assertAlmostEqual(
            level1.cost, ((1 + 7 * steps) * 1.0 + (1 + steps) * 3.0) / 3.0, places=12
        )
        self.assertAlmostEqual(level1.cost_per_sample, level1.cost / steps, places=12)
        self.assertAlmostEqual(result.total_cost, level0.cost + level1.cost, places=12)

    def test_same_seed_same_result(self) -> None:
        """Reproducible from the seed; a different seed differs."""
        a = self._run(2, samples=[100, 100], seed=9)
        b = self._run(2, samples=[100, 100], seed=9)
        c = self._run(2, samples=[100, 100], seed=10)
        # repr comparison: the result may hold nan diagnostics, and Python 3.13
        # dataclass equality compares fields with ``==`` only (``nan != nan``).
        self.assertEqual(repr(a), repr(b))
        self.assertNotEqual(a.estimate, c.estimate)

    def test_allocation_meets_the_variance_target(self) -> None:
        """Suggested chain lengths satisfy the Giles constraint they target."""
        result = self._run(2, samples=[600, 600])
        target = 1e-4
        sizes = multilevel_sample_allocation(result, target)
        self.assertEqual(len(sizes), 2)
        total = math.fsum(
            r.variance * r.samples / r.effective_sample_size / n
            for r, n in zip(result.levels, sizes, strict=True)
        )
        self.assertLessEqual(total, target * (1 + 1e-12))
        with self.assertRaises(ValueError):
            multilevel_sample_allocation(result, 0.0)

    def test_poorly_coupled_levels_are_flagged(self) -> None:
        """A coarse posterior far from the fine one gives a low-acceptance warning."""
        coarse = lambda x: -0.5 * x[0] ** 2 / 9.0  # noqa: E731
        fine = lambda x: -0.5 * (x[0] - 1.0) ** 2 / 0.02**2  # noqa: E731
        result = multilevel_mcmc(
            [coarse, fine],
            _identity,
            (1.0,),
            samples=200,
            step=3.0,
            burn_in=20,
            subsampling=5,
            seed=2,
        )
        self.assertLess(result.levels[1].acceptance_rate, 0.1)
        self.assertTrue(any("acceptance" in w for w in result.warnings))

    def test_failure_cases(self) -> None:
        """Invalid inputs and infeasible start points are refused."""
        d0, d1 = _level_density(0.8), _level_density(1.0)

        def call(**overrides: object):  # type: ignore[no-untyped-def]
            options: dict[str, object] = {
                "log_densities": [d0, d1],
                "quantity": _identity,
                "initial": (0.0,),
                "samples": 20,
                "step": 1.0,
            }
            options.update(overrides)
            return multilevel_mcmc(**options)  # type: ignore[arg-type]

        call()  # the baseline call is valid
        with self.assertRaises(ValueError):
            call(log_densities=[d0])
        with self.assertRaises(TypeError):
            call(log_densities=[d0, "x"])
        with self.assertRaises(ValueError):
            call(samples=3)
        with self.assertRaises(TypeError):
            call(samples=20.0)
        with self.assertRaises(ValueError):
            call(samples=[20, 20, 20])
        with self.assertRaises(ValueError):
            call(subsampling=0)
        with self.assertRaises(ValueError):
            call(subsampling=[2, 2])
        with self.assertRaises(ValueError):
            call(step=-1.0)
        with self.assertRaises(ValueError):
            call(step=[1.0, 1.0])
        with self.assertRaises(ValueError):
            call(costs=[1.0])
        with self.assertRaises(ValueError):
            call(costs=[1.0, -1.0])
        with self.assertRaises(ValueError):
            call(burn_in=-1)
        with self.assertRaises(ValueError):
            call(quantity=[_identity])
        with self.assertRaises(ValueError):
            call(initial=(math.nan,))
        with self.assertRaises(ValueError):
            call(log_densities=[d0, lambda x: -math.inf])  # infeasible start
        with self.assertRaises(ValueError):
            call(log_densities=[lambda x: math.nan, d1])
        with self.assertRaises(ValueError):
            call(log_densities=[d0, lambda x: math.inf])
        with self.assertRaises(ValueError):
            call(quantity=lambda x: math.inf)

    def test_per_level_quantities_are_supported(self) -> None:
        """One callable per level is accepted and applied level-wise."""
        result = multilevel_mcmc(
            [_level_density(0.8), _level_density(1.0)],
            [_identity, lambda x: 2.0 * x[0]],
            (0.0,),
            samples=300,
            step=1.0,
            burn_in=30,
            subsampling=10,
            seed=5,
        )
        self.assertEqual(len(result.levels), 2)
        self.assertTrue(math.isfinite(result.estimate))


@unittest.skipUnless(SLOW, "slow tier: set PROBSTAT_SLOW=1")
class SlowMultifidelityTests(unittest.TestCase):
    """Larger studies; run once, numbers are printed."""

    def test_three_level_bias_per_level_over_several_seeds(self) -> None:
        """Every level and the combined estimate agree with the closed forms."""
        exact = [_level_mean(a) for a in SLOPES[:3]]
        bad = 0
        total = 0
        start = time.perf_counter()
        for seed in range(1, 9):
            result = multilevel_mcmc(
                [_level_density(a) for a in SLOPES[:3]],
                _identity,
                (0.0,),
                samples=[8000, 8000, 6000],
                step=1.0,
                burn_in=200,
                subsampling=[25, 4],
                costs=[1.0, 3.0, 10.0],
                seed=seed,
            )
            for level in result.levels:
                want = exact[level.level] - (
                    exact[level.level - 1] if level.level else 0
                )
                total += 1
                bad += abs(level.mean - want) > 3.0 * level.mcse
            total += 1
            bad += abs(result.estimate - exact[2]) > 3.0 * result.standard_error
            print(
                f"seed {seed}: estimate {result.estimate:.5f} exact {exact[2]:.5f} "
                f"SE {result.standard_error:.5f} cost {result.total_cost:.1f} "
                f"V={[round(r.variance, 4) for r in result.levels]}"
            )
        print(f"3-sigma exceedances: {bad}/{total}; {time.perf_counter() - start:.1f}s")
        self.assertLessEqual(bad, 2)

    def test_work_normalised_variance_against_a_single_level_chain(self) -> None:
        """Measured variance x cost of MLMCMC vs a plain fine chain; may lose."""
        fine = _level_density(SLOPES[3])
        plain = delayed_acceptance_metropolis(
            lambda x: fine(x),
            lambda x: 0.0,
            (0.0,),
            step=1.0,
            samples=20000,
            burn_in=200,
            seed=1,
        )
        ess = min(e for e in plain.effective_sample_size if math.isfinite(e))
        post_var = 1.0 / (1.0 / PRIOR_SD**2 + 1.0 / NOISE_SD**2)
        plain_var = post_var / ess
        plain_cost = plain.exact_evaluations
        result = multilevel_mcmc(
            [_level_density(a) for a in SLOPES],
            _identity,
            (0.0,),
            samples=[8000, 8000, 6000, 4000],
            step=1.0,
            burn_in=200,
            subsampling=[25, 4, 3],
            costs=[1.0, 3.0, 10.0, 30.0],
            seed=2,
        )
        ml_var = result.standard_error**2
        print(
            f"plain: var {plain_var:.3e} x cost {plain_cost} = "
            f"{plain_var * plain_cost:.3f}; "
            f"MLMCMC: var {ml_var:.3e} x cost {result.total_cost:.1f} = "
            f"{ml_var * result.total_cost:.3f} (declared costs 1:3:10:30)"
        )
        self.assertTrue(math.isfinite(ml_var * result.total_cost))

    def test_two_dimensional_active_surrogate_to_thirty_runs(self) -> None:
        """KL versus fine runs in 2D up to 30 runs."""
        axis = [-3.0 + 6.0 * i / 40 for i in range(41)]
        candidates = [(a, b) for a in axis[::2] for b in axis[::2]]
        design = [(-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (0.0, 0.0)]
        last = math.inf
        for budget in (10, 15, 20, 30):
            start = time.perf_counter()
            result = active_surrogate_posterior(
                _logp_2d,
                _gp_fit,  # type: ignore[arg-type]
                design,
                candidates,
                max_fine_evaluations=budget,
            )
            err = grid_posterior_error(_logp_2d, result.log_density, [axis, axis])
            print(
                f"2D budget {budget}: KL {err.kl_divergence:.4g}, mean error "
                f"{err.mean_error}, {time.perf_counter() - start:.1f}s"
            )
            last = err.kl_divergence
        self.assertLess(last, 1e-2)

    def test_delayed_acceptance_cost_when_acceleration_loses(self) -> None:
        """Relative efficiency versus a plain chain for cheap and costly surrogates."""

        def exact(x: tuple[float, ...]) -> float:
            return -0.5 * x[0] ** 2

        def cheap(x: tuple[float, ...]) -> float:
            return -0.5 * (x[0] - 1.0) ** 2 / 0.5**2  # strongly biased

        plain = delayed_acceptance_metropolis(
            exact, lambda x: 0.0, (0.0,), step=2.0, samples=20000, burn_in=200, seed=1
        )
        base = min(e for e in plain.effective_sample_size if math.isfinite(e))
        base /= plain.exact_evaluations
        for ratio in (0.01, 0.5, 1.0):
            model = declare_cost_model(ratio, 1.0)
            reports = []
            for seed in (1, 2, 3):
                result = delayed_acceptance_metropolis(
                    exact,
                    cheap,
                    (0.0,),
                    step=2.0,
                    samples=20000,
                    burn_in=200,
                    seed=seed,
                )
                reports.append(
                    delayed_acceptance_cost(
                        result, model, baseline_ess_per_fine_evaluation=base
                    ).relative_efficiency
                )
            print(f"cost ratio {ratio}: relative efficiency over seeds {reports}")
        self.assertTrue(all(math.isfinite(r) for r in reports))


# ---------------------------------------------------------------------------
# coarse-to-fine warm starts
# ---------------------------------------------------------------------------

LIN_C = ((1.0,), (2.0,))
LIN_B = ((1.0, 0.5), (0.0, 2.0))
LIN_Y = (1.0, 3.0)


def _linear_model(matrix: tuple[tuple[float, ...], ...]) -> CallbackForward:
    """Return ``x -> A x`` with its exact Jacobian."""
    rows, cols = len(matrix), len(matrix[0])

    def value(x: tuple[float, ...]) -> tuple[float, ...]:
        return matvec(matrix, x)

    return CallbackForward(cols, rows, value, jacobian=lambda x: matrix)


def _duplicate(v: tuple[float, ...]) -> tuple[float, ...]:
    """Piecewise-constant prolongation onto a mesh with twice the cells."""
    return tuple(a for a in v for _ in range(2))


def _ridge(matrix: tuple[tuple[float, ...], ...], lam: float) -> tuple[float, ...]:
    """Return ``(A'A + lam^2 I)^-1 A'y`` with dense linear algebra."""
    at = transpose(matrix)
    n = len(at)
    gram = [
        [sum(at[i][k] * at[j][k] for k in range(len(at[0]))) for j in range(n)]
        for i in range(n)
    ]
    for i in range(n):
        gram[i][i] += lam * lam
    return tuple(solve_spd(tuple(map(tuple, gram)), matvec(at, LIN_Y)))


def _elliptic_hierarchy() -> tuple[list[EllipticDiffusion1D], tuple[float, ...]]:
    """Return 8/16/32-cell diffusion models and data from a 32-cell truth."""
    sensors = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    models = [EllipticDiffusion1D(n, sensors=sensors, source=1.0) for n in (8, 16, 32)]
    truth = tuple(
        0.5 * math.sin(2.0 * math.pi * (i + 0.5) / 32) + 0.3 for i in range(32)
    )
    return models, tuple(models[-1].value(truth))


class CoarseToFineTests(unittest.TestCase):
    """Warm-start sweeps: closed form, bookkeeping, measured comparison."""

    def test_linear_hierarchy_matches_the_normal_equations(self) -> None:
        """Each level reproduces the exact Tikhonov solution of its own problem."""
        lam = 0.5
        result = coarse_to_fine_warm_start(
            [_linear_model(LIN_C), _linear_model(LIN_B)],
            [_duplicate],
            LIN_Y,
            regularization=lam,
            level_costs=(1.0, 4.0),
            inner_tolerance=1e-13,
        )
        coarse = result.levels[0]
        exact0 = sum(c[0] * y for c, y in zip(LIN_C, LIN_Y, strict=True)) / (
            sum(c[0] ** 2 for c in LIN_C) + lam**2
        )
        self.assertAlmostEqual(coarse.result.estimate[0], exact0, places=10)
        fine = result.levels[1]
        # the second level starts at the prolongated coarse solution
        self.assertEqual(fine.initial, (coarse.result.estimate[0],) * 2)
        for a, b in zip(fine.result.estimate, _ridge(LIN_B, lam), strict=True):
            self.assertAlmostEqual(a, b, places=9)
        self.assertTrue(result.converged and result.reached_finest)
        self.assertEqual(result.estimate, fine.result.estimate)

    def test_cost_bookkeeping_is_exact_arithmetic(self) -> None:
        """Cost = (value + jvp + vjp) * c_l / c_L with declared costs."""
        result = coarse_to_fine_warm_start(
            [_linear_model(LIN_C), _linear_model(LIN_B)],
            [_duplicate],
            LIN_Y,
            regularization=0.5,
            level_costs=(1.0, 4.0),
            derivative_cost=2.0,
        )
        total = 0.0
        for level, ratio in zip(result.levels, (0.25, 1.0), strict=True):
            c = level.result.counts
            evaluations = c.value + c.derivative_value + 2.0 * (c.jvp + c.vjp)
            self.assertEqual(level.model_evaluations, evaluations)
            self.assertEqual(level.cost_ratio, ratio)
            self.assertEqual(level.cost, evaluations * ratio)
            total += evaluations * ratio
        self.assertEqual(result.warm_cost, total)
        self.assertEqual(result.cost_source, "declared by the caller")
        cold = result.cold
        assert cold is not None
        cold_evals = cold.counts.value + 2.0 * (cold.counts.jvp + cold.counts.vjp)
        self.assertEqual(result.cold_cost, cold_evals)
        self.assertEqual(result.cost_ratio, total / cold_evals)
        self.assertEqual(
            result.warm_iterations, sum(r.iterations for r in result.levels)
        )

    def test_cold_start_can_be_skipped_and_default_costs_are_declared(self) -> None:
        """Without the comparison the cold fields are explicit non-values."""
        result = coarse_to_fine_warm_start(
            [_linear_model(LIN_C), _linear_model(LIN_B)],
            [_duplicate],
            LIN_Y,
            regularization=0.5,
            compare_cold_start=False,
        )
        self.assertIsNone(result.cold)
        self.assertEqual(result.cold_iterations, -1)
        self.assertTrue(math.isnan(result.cold_cost) and math.isnan(result.cost_ratio))
        self.assertTrue(math.isnan(result.objective_gap))
        self.assertIn("assumed", result.cost_source)
        self.assertEqual(result.levels[0].cost_ratio, 0.5)  # dimensions 1 : 2

    def test_elliptic_hierarchy_measured_comparison(self) -> None:
        """8/16/32 cells: both routes reach the same objective; costs measured.

        Measured with this seed-free deterministic problem (lambda = 1e-2,
        start 0): warm 27 steps / 270.5 fine-equivalent units against cold
        8 steps / 146 units, so on this problem the warm start LOST. The test
        asserts the consistency of the report, never a saving.
        """
        models, data = _elliptic_hierarchy()
        result = coarse_to_fine_warm_start(
            models,
            [_duplicate, _duplicate],
            data,
            regularization=1e-2,
            max_iterations=100,
        )
        self.assertEqual([lv.dimension for lv in result.levels], [8, 16, 32])
        self.assertTrue(all(lv.result.converged for lv in result.levels))
        cold = result.cold
        assert cold is not None and cold.converged
        # both routes minimise the same strictly convex-ish objective
        self.assertLess(abs(result.objective_gap), 1e-9)
        self.assertLess(result.estimate_distance, 1e-3)
        self.assertAlmostEqual(
            result.cost_ratio, result.warm_cost / result.cold_cost, places=12
        )
        # level starts chain through the prolongation
        for lower, upper in zip(result.levels, result.levels[1:], strict=False):
            self.assertEqual(upper.initial, _duplicate(lower.result.estimate))
        print(
            "elliptic 8/16/32 warm iterations",
            [lv.iterations for lv in result.levels],
            "warm cost",
            round(result.warm_cost, 2),
            "cold iterations",
            result.cold_iterations,
            "cold cost",
            round(result.cold_cost, 2),
            "ratio",
            round(result.cost_ratio, 3),
        )

    def test_failed_level_is_reported_and_can_stop_the_sweep(self) -> None:
        """An iteration cap of 1 does not converge and is never hidden."""
        models, data = _elliptic_hierarchy()
        stopped = coarse_to_fine_warm_start(
            models,
            [_duplicate, _duplicate],
            data,
            regularization=1e-2,
            max_iterations=1,
            stop_on_failure=True,
        )
        self.assertFalse(stopped.reached_finest)
        self.assertFalse(stopped.converged)
        self.assertEqual(stopped.stop_reason, "level_0_failed:max_iterations")
        self.assertEqual(len(stopped.levels), 1)
        self.assertEqual(len(stopped.estimate), 8)
        self.assertTrue(math.isnan(stopped.objective_gap))
        continued = coarse_to_fine_warm_start(
            models,
            [_duplicate, _duplicate],
            data,
            regularization=1e-2,
            max_iterations=1,
        )
        self.assertTrue(continued.reached_finest)
        self.assertEqual(continued.stop_reason, "max_iterations")
        self.assertEqual(len(continued.levels), 3)

    def test_invalid_hierarchies_are_rejected(self) -> None:
        """Dimension, cost, prolongation and typing errors raise."""
        a, b = _linear_model(LIN_C), _linear_model(LIN_B)
        run = coarse_to_fine_warm_start
        with self.assertRaises(ValueError):
            run([a], [], LIN_Y)
        with self.assertRaises(ValueError):
            run([a, b], [], LIN_Y)
        with self.assertRaises(TypeError):
            run([a, object()], [_duplicate], LIN_Y)  # type: ignore[list-item]
        with self.assertRaises(ValueError):  # decreasing level costs
            run([a, b], [_duplicate], LIN_Y, level_costs=(2.0, 1.0))
        with self.assertRaises(ValueError):
            run([a, b], [_duplicate], LIN_Y, derivative_cost=0.0)
        with self.assertRaises(ValueError):  # wrong prolongation length
            run([a, b], [lambda v: v], LIN_Y)
        with self.assertRaises(ValueError):  # non-finite prolongation
            run([a, b], [lambda v: (math.nan, 0.0)], LIN_Y)
        with self.assertRaises(ValueError):  # per-level list of wrong length
            run([a, b], [_duplicate], LIN_Y, regularization=[0.1, 0.1, 0.1])
        with self.assertRaises(ValueError):  # initial of the wrong dimension
            run([a, b], [_duplicate], LIN_Y, initial=(0.0, 0.0))
        c = _linear_model(((1.0, 0.5),))  # different output dimension
        with self.assertRaises(ValueError):
            run([a, c], [_duplicate], LIN_Y)


# ---------------------------------------------------------------------------
# sensitivity-error validation
# ---------------------------------------------------------------------------

DELTA = 0.05


def _fine_exp(delta: float = 0.0) -> CallbackForward:
    """Return ``(exp(x0), x0 x1) + delta x`` with the exact Jacobian."""

    def value(x: tuple[float, ...]) -> tuple[float, ...]:
        return (
            math.exp(x[0]) + delta * x[0],
            x[0] * x[1] + delta * x[1],
        )

    def jacobian(x: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
        return ((math.exp(x[0]) + delta, 0.0), (x[1], x[0] + delta))

    return CallbackForward(2, 2, value, jacobian=jacobian)


def _rotation(theta: float, scale: float = 1.0) -> CallbackForward:
    """Return ``x -> scale R(theta) x``."""
    c, s = scale * math.cos(theta), scale * math.sin(theta)
    return _linear_model(((c, -s), (s, c)))


class SensitivityErrorTests(unittest.TestCase):
    """Derivative-error estimates against closed-form errors."""

    def test_known_jacobian_error_is_reproduced_exactly(self) -> None:
        """G_a = G_f + delta x has J_a - J_f = delta I: errors in closed form."""
        x = (0.3, -0.7)
        report = validate_sensitivities(
            _fine_exp(),
            _fine_exp(DELTA),
            [x],
            tolerance=1.0,
            directions=[(1.0, 0.0), (0.0, 1.0)],
            adjoint_weights=[(1.0, 0.0), (0.0, 1.0)],
        )
        e = math.exp(0.3)
        self.assertAlmostEqual(
            report.jvp_relative_errors[0], DELTA / math.hypot(e, 0.7), places=14
        )
        self.assertAlmostEqual(report.jvp_relative_errors[1], DELTA / 0.3, places=14)
        self.assertAlmostEqual(report.gradient_relative_errors[0], DELTA / e, places=14)
        self.assertAlmostEqual(
            report.gradient_relative_errors[1], DELTA / math.hypot(0.7, 0.3), places=14
        )
        # angle of (-0.7, 0.3 + delta) against (-0.7, 0.3) from atan2 directly
        expected = abs(math.atan2(0.3 + DELTA, -0.7) - math.atan2(0.3, -0.7))
        self.assertAlmostEqual(report.gradient_angles[1], expected, places=14)
        self.assertAlmostEqual(report.gradient_angles[0], 0.0, places=14)
        self.assertEqual(report.gradient_kind, "adjoint_probe")
        self.assertEqual(report.jvp_max, max(report.jvp_relative_errors))

    def test_rotation_error_and_angle_closed_form_and_the_decision(self) -> None:
        """R(theta) x: rel. error 2 sin(theta/2) and angle theta for every probe."""
        theta = 0.2
        err = 2.0 * math.sin(theta / 2.0)
        pts = [(0.1, 0.2), (1.0, -1.0)]
        report = validate_sensitivities(
            _rotation(0.0), _rotation(theta), pts, tolerance=err * 1.01, probes=4
        )
        for value in report.jvp_relative_errors + report.gradient_relative_errors:
            self.assertAlmostEqual(value, err, places=12)
        for value in report.gradient_angles:
            self.assertAlmostEqual(value, theta, places=12)
        self.assertEqual(len(report.jvp_relative_errors), 8)
        self.assertTrue(report.fit_for_gradient_use)
        self.assertEqual(report.failures, ())
        strict = validate_sensitivities(
            _rotation(0.0), _rotation(theta), pts, tolerance=err * 0.99
        )
        self.assertFalse(strict.fit_for_gradient_use)
        joined = " ".join(strict.failures)
        self.assertIn("jvp relative error", joined)
        self.assertIn("gradient angle", joined)
        # a tolerance above the default-angle bound but an explicit tight angle
        tight = validate_sensitivities(
            _rotation(0.0),
            _rotation(theta),
            pts,
            tolerance=1.0,
            angle_tolerance=0.1,
        )
        self.assertFalse(tight.fit_for_gradient_use)
        self.assertEqual(tight.angle_tolerance, 0.1)

    def test_scaled_model_has_zero_angle_and_error_equal_to_the_scale(self) -> None:
        """C G_f: relative error |c - 1|, gradients parallel."""
        report = validate_sensitivities(
            _rotation(0.0), _rotation(0.0, scale=1.1), [(0.3, 0.4)], tolerance=0.11
        )
        self.assertAlmostEqual(report.jvp_max, 0.1, places=12)
        self.assertAlmostEqual(report.gradient_max, 0.1, places=12)
        self.assertLess(report.angle_max, 1e-7)
        self.assertTrue(report.fit_for_gradient_use)

    def test_misfit_gradient_includes_the_value_error(self) -> None:
        """Exact Jacobian but biased value: only the misfit gradient is wrong."""
        a = ((1.0, 2.0), (0.0, 1.0), (1.0, -1.0))
        bias = (0.2, -0.1, 0.3)
        y = (0.4, 0.1, -0.5)
        x = (0.5, -1.0)
        exact = _linear_model(a)

        def biased_value(p: tuple[float, ...]) -> tuple[float, ...]:
            return tuple(u + b for u, b in zip(matvec(a, p), bias, strict=True))

        biased = CallbackForward(2, 3, biased_value, jacobian=lambda p: a)
        report = validate_sensitivities(
            exact, biased, [x], tolerance=10.0, data=y, probes=2
        )
        self.assertEqual(report.gradient_kind, "misfit")
        self.assertEqual(report.jvp_max, 0.0)  # Jacobians coincide
        at = transpose(a)
        residual = [u - v for u, v in zip(matvec(a, x), y, strict=True)]
        g_f = matvec(at, tuple(residual))
        diff = matvec(at, bias)
        expected = math.hypot(*diff) / math.hypot(*g_f)
        self.assertEqual(len(report.gradient_relative_errors), 1)
        self.assertAlmostEqual(report.gradient_relative_errors[0], expected, places=13)
        g_a = [p + q for p, q in zip(g_f, diff, strict=True)]
        cosine = sum(p * q for p, q in zip(g_f, g_a, strict=True)) / (
            math.hypot(*g_f) * math.hypot(*g_a)
        )
        self.assertAlmostEqual(report.gradient_angles[0], math.acos(cosine), places=12)

    def test_degenerate_and_nonfinite_cases_are_not_fit(self) -> None:
        """Zero fine sensitivity, non-finite output and zero gradients."""
        zero = CallbackForward(1, 1, lambda x: (1.0,), jacobian=lambda x: ((0.0,),))
        slope = CallbackForward(1, 1, lambda x: (x[0],), jacobian=lambda x: ((1.0,),))
        report = validate_sensitivities(zero, slope, [(0.2,)], tolerance=0.5)
        self.assertEqual(report.jvp_max, math.inf)
        self.assertFalse(report.fit_for_gradient_use)
        self.assertEqual(report.undefined_angles, 3)
        self.assertEqual(report.gradient_angles, ())
        self.assertTrue(math.isnan(report.angle_max))
        both = validate_sensitivities(zero, zero, [(0.2,)], tolerance=0.5)
        self.assertTrue(both.fit_for_gradient_use)
        self.assertEqual(both.jvp_max, 0.0)
        bad = CallbackForward(1, 1, lambda x: (math.nan,), jacobian=lambda x: ((1.0,),))
        flagged = validate_sensitivities(
            slope, bad, [(0.2,)], tolerance=0.5, data=(0.0,)
        )
        self.assertFalse(flagged.fit_for_gradient_use)
        self.assertIn("non-finite", " ".join(flagged.failures))
        with self.assertRaises(ValueError):
            validate_sensitivities(bad, slope, [(0.2,)], tolerance=0.5)

    def test_probes_are_reproducible_and_seed_dependent(self) -> None:
        """Random directions come from the seed only; errors depend on them."""

        def anisotropic(x: tuple[float, ...]) -> tuple[float, ...]:
            return (x[0] + 0.3 * x[0], x[1] + 0.01 * x[1])

        approx = CallbackForward(
            2, 2, anisotropic, jacobian=lambda x: ((1.3, 0.0), (0.0, 1.01))
        )
        fine = _rotation(0.0)

        def run(seed: int) -> tuple[float, ...]:
            return validate_sensitivities(
                fine, approx, [(0.1, 0.2)], tolerance=1.0, probes=4, seed=seed
            ).jvp_relative_errors

        self.assertEqual(run(3), run(3))
        self.assertNotEqual(run(3), run(4))
        # every anisotropic error lies between the two singular errors
        self.assertTrue(all(0.01 - 1e-12 <= e <= 0.3 + 1e-12 for e in run(3)))

    def test_invalid_inputs_are_rejected(self) -> None:
        """Dimension, tolerance, training overlap and probe validation."""
        fine, approx = _rotation(0.0), _rotation(0.1)
        one = CallbackForward(1, 1, lambda x: (x[0],), jacobian=lambda x: ((1.0,),))
        with self.assertRaises(ValueError):
            validate_sensitivities(fine, one, [(0.0, 0.0)], tolerance=0.1)
        with self.assertRaises(TypeError):
            validate_sensitivities(fine, object(), [(0.0, 0.0)], tolerance=0.1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            validate_sensitivities(fine, approx, [(0.0, 0.0)], tolerance=0.0)
        with self.assertRaises(ValueError):
            validate_sensitivities(fine, approx, [], tolerance=0.1)
        with self.assertRaises(ValueError):
            validate_sensitivities(fine, approx, [(0.0,)], tolerance=0.1)
        with self.assertRaises(ValueError):
            validate_sensitivities(
                fine, approx, [(0.0, 1.0)], tolerance=0.1, training_points=[(0.0, 1.0)]
            )
        with self.assertRaises(ValueError):
            validate_sensitivities(
                fine, approx, [(0.0, 1.0)], tolerance=0.1, angle_tolerance=4.0
            )
        with self.assertRaises(ValueError):
            validate_sensitivities(
                fine, approx, [(0.0, 1.0)], tolerance=0.1, directions=[(0.0, 0.0)]
            )
        with self.assertRaises(ValueError):
            validate_sensitivities(
                fine, approx, [(0.0, 1.0)], tolerance=0.1, adjoint_weights=[(1.0,)]
            )


def _logp_1d_forward() -> CallbackForward:
    """Return the fine scalar log-target of the surrogate tests, analytic."""

    def slope(x: tuple[float, ...]) -> float:
        a = x[0]
        forward = a + 0.3 * a**3
        return -a + (Y1 - forward) * (1.0 + 0.9 * a * a) / SIG1**2

    return CallbackForward(
        1,
        1,
        lambda x: (_logp_1d(x),),
        jvp=lambda x, v: (slope(x) * v[0],),
        vjp=lambda x, w: (slope(x) * w[0],),
    )


class SurrogateSensitivityTests(unittest.TestCase):
    """Sensitivity validation of the module's own active surrogate."""

    HELD_OUT = [(0.613,), (0.77,), (0.9123,), (1.05,), (1.2,), (1.347,)]

    def test_surrogate_derivative_error_shrinks_with_fine_runs(self) -> None:
        """Measured: 4 runs unusable, 12 runs poor, 20 runs fit at tolerance 0.1."""
        fine = _logp_1d_forward()
        worst = {}
        reports = {}
        for budget in (4, 12, 20):
            result = active_surrogate_posterior(
                _logp_1d,
                _gp_fit,  # type: ignore[arg-type]
                DESIGN_1D,
                CANDIDATES_1D,
                max_fine_evaluations=budget,
            )
            surrogate = surrogate_mean_forward(result.surrogate, 1)
            report = validate_sensitivities(
                fine,
                surrogate,
                self.HELD_OUT,
                tolerance=0.1,
                probes=2,
                training_points=result.design,
            )
            reports[budget] = report
            worst[budget] = report.gradient_max
            print(
                f"surrogate with {budget} fine runs: jvp max {report.jvp_max:.4g}, "
                f"gradient max {report.gradient_max:.4g}, angle max "
                f"{report.angle_max:.4g}, fit {report.fit_for_gradient_use}"
            )
        self.assertGreater(worst[4], worst[12])
        self.assertGreater(worst[12], worst[20])
        self.assertFalse(reports[4].fit_for_gradient_use)
        self.assertGreater(reports[4].angle_max, 3.0)  # an ascent direction
        self.assertFalse(reports[12].fit_for_gradient_use)
        self.assertTrue(reports[20].fit_for_gradient_use)
        self.assertLess(reports[20].jvp_max, 0.1)

    def test_surrogate_wrapper_validates_its_input(self) -> None:
        """The wrapper needs a predictive surrogate and exposes the mean."""
        with self.assertRaises(TypeError):
            surrogate_mean_forward(object(), 1)  # type: ignore[arg-type]

        class Quadratic:
            def predict_with_variance(self, x: object) -> tuple[float, float]:
                a = x[0]  # type: ignore[index]
                return (a * a, 0.0)

        model = surrogate_mean_forward(Quadratic(), 1)
        self.assertEqual(model.value((3.0,)), (9.0,))
        self.assertAlmostEqual(model.jvp((3.0,), (1.0,))[0], 6.0, places=6)
        self.assertAlmostEqual(model.vjp((3.0,), (2.0,))[0], 12.0, places=6)


if __name__ == "__main__":
    unittest.main()
