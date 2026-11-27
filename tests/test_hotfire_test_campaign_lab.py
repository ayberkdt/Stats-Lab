"""Regression tests of the hot-fire test campaign laboratory.

The lab (``examples/hotfire_test_campaign_lab.py``) calibrates a synthetic
engine with ``probstat.validation.KOHEmulatorCalibration`` and
``FunctionalCalibration`` and chooses the next test with the nonlinear
expected-information-gain tools of ``probstat.design``. These tests import the
lab without running ``main``, check its closed-form building blocks against
independent formulas, and run each stage with sizes smaller than the lab's
(the Isp emulator keeps its full size because it is cheap and because 30 runs
do not pass the Bastos-O'Hagan validation). Stochastic assertions use the
fixed lab seed and bands justified in each test. Every case is synthetic.

References:
    [REF-SUTTON-BIBLARZ-2017], [REF-KENNEDY-OHAGAN-2001],
    [REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008], [REF-HUAN-MARZOUK-2013].
    Equations are evaluated by the library modules the lab calls
    ([EQ-VAL-KOHEM-006], [EQ-OED-EIG-LIN-001]).

"""

from __future__ import annotations

import importlib.util
import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import ModuleType

from probstat.design.nonlinear_optimal import (
    NonlinearExperimentModel,
    laplace_eig,
)
from probstat.validation.koh_emulator import KOHEmulatorCalibration

LAB_PATH = Path(__file__).parents[1] / "examples" / "hotfire_test_campaign_lab.py"


def load_lab() -> ModuleType:
    """Import the lab script as a module without running ``main``.

    Returns:
        The imported module.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    name = "_hotfire_test_campaign_lab_under_test"
    spec = importlib.util.spec_from_file_location(name, LAB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def quiet(function, *args):  # type: ignore[no-untyped-def]
    """Call a lab stage and swallow what it prints.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    with redirect_stdout(io.StringIO()):
        return function(*args)


class LabBuildingBlockTests(unittest.TestCase):
    """The simulator, the synthetic real engine and the data."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import the lab once."""
        cls.lab = load_lab()

    def test_simulator_is_linear_in_efficiency_and_free_of_pressure(self) -> None:
        """Isp = CF c* / g0 with c* = eta c*_ideal: exact scaling, no p_c effect."""
        lab = self.lab
        base = lab.simulator_isp(2.6, 7.0, 1.0)
        for eta in (0.9, 0.95, 0.975):
            self.assertAlmostEqual(
                lab.simulator_isp(2.6, 7.0, eta) / base, eta, places=12
            )
        # The ideal vacuum thrust coefficient does not involve p_c.
        self.assertAlmostEqual(
            lab.simulator_isp(2.3, 5.0, 0.95) / lab.simulator_isp(2.3, 9.0, 0.95),
            1.0,
            places=12,
        )

    def test_real_engine_closed_form(self) -> None:
        """eta_real is the quadratic law; real Isp is the simulator at that eta."""
        lab = self.lab
        self.assertAlmostEqual(lab.real_efficiency(2.4, 7.0), 0.975, places=15)
        self.assertAlmostEqual(
            lab.real_efficiency(3.15, 7.0), 0.975 - 0.045 * 0.75**2, places=15
        )
        self.assertAlmostEqual(
            lab.real_isp(3.15, 7.0),
            lab.simulator_isp(3.15, 7.0, lab.real_efficiency(3.15, 7.0)),
            places=12,
        )
        # Value printed by the lab for the untested point (a pinned number).
        self.assertAlmostEqual(lab.real_isp(*lab.UNTESTED), 293.371, delta=1.0e-3)

    def test_model_form_error_exceeds_the_measurement_noise(self) -> None:
        """No constant eta reproduces the real engine to within the noise."""
        lab = self.lab
        result = quiet(lab.truth_part)
        self.assertGreater(max(abs(e) for e in result["errors"]), 10.0 * lab.ISP_NOISE)

    def test_hotfire_data_reproducible_and_scattered_by_the_noise(self) -> None:
        """Same seed, same data; log-scatter about the real Isp has sd ~ ISP_NOISE."""
        lab = self.lab
        data = lab.hotfire_data(5)
        self.assertEqual(data, lab.hotfire_data(5))
        residuals = [
            v - lab.real_isp(of, pc)
            for v, (of, pc) in zip(data, lab.TESTS, strict=True)
        ]
        # |mean| < 4 sd / sqrt(8) and every residual within 4 sd.
        self.assertLess(abs(sum(residuals) / 8.0), 4.0 * lab.ISP_NOISE / math.sqrt(8.0))
        self.assertLess(max(abs(r) for r in residuals), 4.0 * lab.ISP_NOISE)

    def test_cooled_nozzle_wall_temperature_rises_with_the_multiplier(self) -> None:
        """A larger Bartz multiplier gives a hotter wall at every station."""
        lab = self.lab
        nozzle = lab.cooled_nozzle()
        cold = lab.wall_temperatures(nozzle, 6.0, 1.0)
        hot = lab.wall_temperatures(nozzle, 6.0, 1.3)
        self.assertEqual(len(cold), len(nozzle["profile"].x))
        self.assertTrue(all(h > c for h, c in zip(hot, cold, strict=True)))
        peak = max(range(len(cold)), key=cold.__getitem__)
        self.assertEqual(peak, nozzle["profile"].throat_index)


class LabEstimatorTests(unittest.TestCase):
    """The EIG estimators on the lab's linear-Gaussian measurement model."""

    def test_laplace_eig_equals_the_closed_form(self) -> None:
        """For y = eta I + e the EIG is 1/2 log(1 + s^2 I^2 / noise^2) exactly."""
        prior_sd, unit, noise = 0.01, 300.0, 1.2
        model = NonlinearExperimentModel(
            lambda theta, d: theta[0] * unit, (0.96,), ((prior_sd**2,),), noise
        )
        estimate = laplace_eig(model, [0.0], samples=20, seed=1)
        closed = 0.5 * math.log(1.0 + (prior_sd * unit / noise) ** 2)
        # J = I is constant; the central difference of a linear map is exact to
        # rounding, so the average equals the closed form.
        self.assertAlmostEqual(estimate.value, closed, places=6)


class LabStageTests(unittest.TestCase):
    """Each stage of the lab with the quick sizes stays plausible."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the stages once: full-size emulator, quick everything else."""
        cls.lab = load_lab()
        lab = cls.lab
        cls.emulator = quiet(lab.emulator_part, lab.FULL)
        cls.budget = replace(
            lab.QUICK,
            draws=50,
            warmup=30,
            predict_draws=10,
            coverage_points=10,
            nested_outer=100,
        )
        cls.calibrated = quiet(lab.calibration_part, cls.budget, cls.emulator)
        cls.cooling = quiet(lab.cooling_part, lab.QUICK)
        cls.design = quiet(lab.design_part, cls.budget, cls.calibrated)

    def test_emulator_passes_its_validation(self) -> None:
        """Held-out rmse is tiny against the 50 s output range; diagnostics pass."""
        check = self.emulator["check"]
        self.assertTrue(check.passed, check.flags)
        # The simulator is smooth and noise free: 40 runs in 3 inputs interpolate
        # to a few hundredths of a second (printed 0.019 s).
        self.assertLess(self.emulator["rmse"], 0.1)
        self.assertGreater(check.coverage, 0.9)

    def test_calibrations_converge_and_the_discrepancy_widens_the_posterior(
        self,
    ) -> None:
        """Both samplers converge; delta inflates the sd of eta by an order of 10."""
        runs = self.calibrated["runs"]
        for result in runs.values():
            # The quick sampler keeps 2 x 60 draws; the lab's own limit of 50
            # effective draws is not met reliably at that size, so the test asks
            # for split R-hat < 1.1 and at least 20 effective draws.
            self.assertLess(result.rhat_max, 1.1)
            self.assertGreater(result.ess_min, 20.0)
        with_sd = runs["with delta"].sd()[0]
        without_sd = runs["without delta"].sd()[0]
        self.assertGreater(with_sd / without_sd, 5.0)
        # Without delta the posterior mean is the constant eta that best fits the
        # eight tests (printed 0.9702); the real efficiency ranges over 0.93-0.98.
        self.assertAlmostEqual(runs["without delta"].mean()[0], 0.970, delta=0.005)
        self.assertGreater(
            self.calibrated["report"].mean_shift_in_reference_sd[0] ** 2, 4.0
        )

    def test_variance_parts_add_up_to_the_total(self) -> None:
        """The five parts of the predictive variance sum to the mixture variance."""
        for prediction in self.calibrated["predictions"].values():
            total = prediction.total_variance[0]
            # The law of total variance is exact; the code sums per-draw moments.
            self.assertAlmostEqual(prediction.component_sum()[0] / total, 1.0, 6)
            self.assertAlmostEqual(prediction.noise[0], self.lab.ISP_NOISE**2)

    def test_the_calibration_without_discrepancy_is_overconfident(self) -> None:
        """It misses the untested real Isp and covers fewer held-out points."""
        truth = self.calibrated["truth"]
        with_delta = self.calibrated["predictions"]["with delta"]
        without = self.calibrated["predictions"]["without delta"]
        self.assertTrue(with_delta.lower[0] <= truth <= with_delta.upper[0])
        self.assertFalse(without.lower[0] <= truth <= without.upper[0])
        # Printed: the no-discrepancy mean is 6.3 s above the real value with a
        # latent sd of 0.2 s; the sign and the order of magnitude are fixed by
        # the model-form error at OF = 3.15.
        self.assertGreater(without.mean[0] - truth, 3.0)
        covers = self.calibrated["covers"]
        self.assertEqual(self.calibrated["overconfident"], "without delta")
        self.assertGreater(
            covers["with delta"]["coverage"] - covers["without delta"]["coverage"], 0.3
        )
        self.assertGreater(
            covers["with delta"]["width"], covers["without delta"]["width"]
        )
        self.assertLess(covers["with delta"]["rmse"], covers["without delta"]["rmse"])

    def test_calibration_objects_share_data_and_emulator(self) -> None:
        """The no-discrepancy problem is the same data with delta switched off."""
        calibrations = self.calibrated["calibrations"]
        self.assertIsInstance(calibrations["with delta"], KOHEmulatorCalibration)
        self.assertTrue(calibrations["with delta"].discrepancy)
        self.assertFalse(calibrations["without delta"].discrepancy)
        self.assertIs(
            calibrations["with delta"].emulator, calibrations["without delta"].emulator
        )

    def test_cooling_calibration_recovers_the_generating_multiplier(self) -> None:
        """The posterior mean is within 4 sd of 1.18; one thermocouple is wider."""
        lab = self.lab
        cooling = self.cooling
        self.assertTrue(cooling["result"].converged)
        # Calibrated by construction (same model generates the data): 4 posterior
        # sd is a 1-in-16000 band; the quick run prints +1.6 sd.
        self.assertLess(
            abs(cooling["mean"] - lab.COOLING_BARTZ_TRUTH), 4.0 * cooling["sd"]
        )
        ratios = {k: c.sd_ratio[0] for k, c in cooling["comparisons"].items()}
        # The throat station is the most informative single location.
        self.assertGreater(ratios["throat thermocouple"], 1.5)
        self.assertGreater(ratios["exit thermocouple"], ratios["throat thermocouple"])
        for comparison in cooling["comparisons"].values():
            self.assertTrue(comparison.scalar_result.converged)
        self.assertLess(cooling["validation"].field_relative_rmse, 0.05)

    def test_eig_matches_the_closed_form_and_nested_monte_carlo(self) -> None:
        """Laplace equals 1/2 log(1 + s^2 I^2 / noise^2); nested is biased upward."""
        design = self.design
        prior_sd = design["prior"][1]
        for single, unit, noise in zip(
            design["singles"], design["unit_isp"], design["noise"], strict=True
        ):
            closed = 0.5 * math.log(1.0 + (prior_sd * unit / noise) ** 2)
            self.assertAlmostEqual(single["laplace"].value, closed, places=6)
            self.assertAlmostEqual(single["closed"], closed, places=12)
            nested = single["nested"]
            # The inner log-mean is biased low (Jensen), so nested exceeds the
            # truth by O(1/inner) = a few percent here, plus sampling error.
            self.assertGreater(nested.value, closed - 3.0 * nested.standard_error)
            self.assertLess(
                nested.value, closed + 3.0 * nested.standard_error + 0.3 * closed
            )

    def test_greedy_matches_exhaustive_and_the_best_test_is_well_known(self) -> None:
        """Greedy is optimal for the additive EIG; the best test is the quietest.

        The EIG is a monotone function of a sum of ``J^2 / noise^2`` over the
        tests, and the best candidate has the smallest discrepancy noise.
        """
        design = self.design
        for method in ("laplace", "nested"):
            found = design["searches"][method]
            self.assertEqual(
                sorted(found["greedy"].design), sorted(found["exhaustive"].design)
            )
        laplace = design["searches"]["laplace"]["exhaustive"]
        self.assertAlmostEqual(laplace.value, design["closed_pair"], places=6)
        quietest = min(range(len(design["noise"])), key=design["noise"].__getitem__)
        self.assertEqual(design["best"], quietest)
        self.assertIn(float(design["best"]), laplace.design)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "matplotlib is not installed",
    )
    def test_figures_are_written(self) -> None:
        """Both figures are written through the themed layer."""
        lab = self.lab
        original = lab.OUTPUT
        with tempfile.TemporaryDirectory() as directory:
            lab.OUTPUT = Path(directory)
            try:
                names = (
                    lab.prediction_figure(self.calibrated, self.budget),
                    lab.information_figure(self.design),
                )
            finally:
                lab.OUTPUT = original
            for name in names:
                self.assertTrue((Path(directory) / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
