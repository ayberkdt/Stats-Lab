"""Regression tests of the spacecraft mission-risk laboratory.

The lab (``examples/spacecraft_mission_risk_lab.py``) follows one synthetic
mission through ascent dispersion, tank buckling, micrometeoroid and debris
risk, engine health monitoring and entry. These tests import the lab without
running ``main``, check closed-form building blocks against independent
formulas, and run each of its five stages with the small ``QUICK`` budget at
the lab's fixed seed. Stochastic assertions use bands of several standard
errors of the quick budget; the deterministic quantities (FORM results,
corridor bounds, nominal PNP) are pinned to the values printed by the full
run with tolerances that cover platform floating-point differences. Every
case is synthetic.

References:
    [REF-NASA-SP-8007-2020], [REF-MELCHERS-BECK-2018], [REF-PAGE-1954].
    Equations are evaluated by the library modules the lab calls
    ([EQ-AERO-BUCK-006], [EQ-AERO-MMOD-008]).

"""

from __future__ import annotations

import importlib.util
import io
import math
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any

from probstat.aerospace import debris_risk as dr
from probstat.aerospace.shell_buckling import (
    classical_buckling_load,
    koiter_coefficient,
    koiter_knockdown,
    shell_buckling_model,
    sp8007_axial_knockdown,
)
from probstat.distributions import Normal

LAB_PATH = Path(__file__).parents[1] / "examples" / "spacecraft_mission_risk_lab.py"


def load_lab() -> ModuleType:
    """Import the lab script as a module without running ``main``.

    Returns:
        The imported module.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    name = "_spacecraft_mission_risk_lab_under_test"
    spec = importlib.util.spec_from_file_location(name, LAB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def quiet(function: Any, *args: Any) -> Any:
    """Call a lab stage and swallow what it prints.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    with redirect_stdout(io.StringIO()):
        return function(*args)


class ClosedFormTests(unittest.TestCase):
    """Closed forms behind the lab's models."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import the lab once."""
        cls.lab = load_lab()

    def test_tank_knockdowns_have_their_closed_forms(self) -> None:
        """SP-8007 knockdown and the Koiter law have their closed forms."""
        lab = self.lab
        ratio = lab.TANK_RADIUS / lab.TANK_THICKNESS
        expected = 1.0 - 0.901 * (1.0 - math.exp(-math.sqrt(ratio) / 16.0))
        self.assertAlmostEqual(
            sp8007_axial_knockdown(lab.TANK_RADIUS, lab.TANK_THICKNESS),
            expected,
            places=14,
        )
        # (1 - lambda)^2 = c lambda d solved by hand for c = 2.453, d = 0.1.
        c, d = koiter_coefficient(lab.TANK_NU), 0.1
        lam = koiter_knockdown(d, c)
        self.assertAlmostEqual((1.0 - lam) ** 2, c * lam * d, places=12)

    def test_tank_model_nominal_margin_is_the_hand_formula(self) -> None:
        """At the nominal point the margin is lambda P_cl minus the load."""
        lab = self.lab
        model = shell_buckling_model(
            youngs_modulus=lab.TANK_E,
            thickness=lab.TANK_THICKNESS,
            radius=lab.TANK_RADIUS,
            imperfection=lab.IMPERFECTION_MEAN,
            applied_load=Normal(lab.TANK_LOAD, 1.0e5),
            poisson_ratio=lab.TANK_NU,
        )
        out = model.evaluate({"applied_load": lab.TANK_LOAD})
        c = koiter_coefficient(lab.TANK_NU)
        lam = koiter_knockdown(lab.IMPERFECTION_MEAN / lab.TANK_THICKNESS, c)
        pcl = classical_buckling_load(
            lab.TANK_E, lab.TANK_THICKNESS, lab.TANK_RADIUS, lab.TANK_NU
        )
        self.assertAlmostEqual(out["margin"], lam * pcl - lab.TANK_LOAD, places=4)

    def test_nominal_pnp_is_exp_minus_n_and_n_scales_with_area(self) -> None:
        """PNP = exp(-N) and the two surfaces add their expected penetrations."""
        lab = self.lab
        environment = lab.impact_environment()
        surfaces = lab.shield_surfaces(lab.DESIGN_WALL)
        total = dr.penetration_risk(surfaces, environment, lab.EXPOSURE_YEARS)
        self.assertAlmostEqual(total.pnp, math.exp(-total.expected_penetrations), 14)
        parts = [
            dr.penetration_risk([s], environment, lab.EXPOSURE_YEARS) for s in surfaces
        ]
        self.assertAlmostEqual(
            total.expected_penetrations,
            sum(p.expected_penetrations for p in parts),
            places=14,
        )
        # Printed by the full run: N = 0.0394, PNP = 0.9614 at 6 mm.
        self.assertAlmostEqual(total.pnp, 0.9614, delta=5.0e-5)

    def test_summary_table_aligns_every_row(self) -> None:
        """Each line of the table has the same column starts."""
        lab = self.lab
        rows = [("a", "quantity one", "1", "x"), ("bb", "q", "22", "yy")]
        lines = lab.summary_table(rows).splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(set(lines[1]) <= {"-", " "})
        self.assertEqual(lines[2].index("quantity one"), lines[3].index("q"))

    def test_lab_declares_its_synthetic_status(self) -> None:
        """The docstring says loudly that the MMOD coefficients are synthetic."""
        lab = self.lab
        self.assertIn("EVERY CASE IS SYNTHETIC", lab.__doc__)
        self.assertIn("not ORDEM", lab.__doc__)


class StageTests(unittest.TestCase):
    """Each stage of the lab runs with the quick budget and stays plausible."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import the lab once."""
        cls.lab = load_lab()

    def test_ascent_part(self) -> None:
        """Perigee near 142 km, high insertion probability, RSS close to Monte Carlo.

        The full run prints a perigee mean of 142.7 km (standard deviation
        6.8 km); 60 runs give a standard error of the mean of 0.9 km, so the
        band is 5 km. The RSS budget of the full run is within 2 % of its
        Monte Carlo standard deviation; 30 runs have a 13 % relative error in
        the standard deviation, so the band on the ratio is wide.
        """
        lab = self.lab
        result = quiet(lab.ascent_part, lab.QUICK)
        perigee = result["intervals"][0]
        self.assertAlmostEqual(perigee.mean, 142.7e3, delta=5.0e3)
        self.assertGreater(perigee.standard_deviation, 3.0e3)
        self.assertLess(perigee.standard_deviation, 12.0e3)
        # 0.953 +- 0.008 in the full run; 60 runs have a standard error of 0.03.
        self.assertGreater(result["p_insertion"], 0.8)
        self.assertLessEqual(result["p_insertion"], 1.0)
        budget = result["budget"]
        self.assertAlmostEqual(budget.ratio, 1.0, delta=0.4)
        self.assertEqual(budget.input_names[0], "thrust_scale")
        ellipse = result["ellipse"]
        # Steeper burnouts raise the burnout altitude and lower the perigee.
        self.assertLess(ellipse.correlation, -0.4)
        self.assertGreater(ellipse.semi_axes[0], ellipse.semi_axes[1])

    def test_tank_part(self) -> None:
        """FORM is deterministic; Monte Carlo agrees; SP-8007 is below k_p here only.

        FORM gives 9.40e-3 (beta 2.349) in the full run; the 4000-run Monte
        Carlo has a binomial standard error of 1.5e-3, so the band is five
        standard errors.
        """
        lab = self.lab
        result = quiet(lab.tank_part, lab.QUICK)
        self.assertTrue(result["form"].converged)
        self.assertAlmostEqual(result["form"].probability, 9.40e-3, delta=1.0e-4)
        self.assertAlmostEqual(result["form"].beta, 2.349, delta=0.01)
        mc = result["mc"]
        self.assertLess(
            abs(mc.probability - result["form"].probability), 5.0 * mc.standard_error
        )
        self.assertAlmostEqual(result["k_p"], 0.454, delta=0.001)
        self.assertAlmostEqual(result["gamma"], 0.322, delta=0.001)
        self.assertEqual(result["table"].sp8007_conservative, (True, True, True))
        # Loaded to the SP-8007 design load the shell is far safer.
        self.assertLess(
            result["at_design"].probability, 0.1 * result["form"].probability
        )
        self.assertGreater(result["form"].importance_factors["imperfection"], 0.5)

    def test_mmod_part(self) -> None:
        """Epistemic margin adds wall; the mean PNP matches Gauss-Hermite quadrature.

        The full run prints a nominal wall of 5.40 mm and a 90 % quantile wall
        of 7.82 mm (2000 draws); with 500 draws the quantile wall printed
        7.70 mm, and the empirical 0.9 quantile of N has about 3 % error, so
        the band on the thickness is 0.6 mm.
        """
        lab = self.lab
        result = quiet(lab.mmod_part, lab.QUICK)
        distribution = result["distribution"]
        self.assertLess(
            abs(distribution.mean_pnp - result["quadrature"].mean_pnp),
            5.0 * distribution.mean_pnp_standard_error,
        )
        nominal, margin = result["nominal"], result["margin"]
        self.assertTrue(nominal.converged and margin.converged)
        self.assertAlmostEqual(nominal.thickness, 5.40e-3, delta=0.02e-3)
        self.assertAlmostEqual(nominal.nominal_pnp, lab.PNP_REQUIREMENT, places=8)
        self.assertAlmostEqual(margin.thickness, 7.82e-3, delta=0.6e-3)
        self.assertGreater(margin.thickness, 1.3 * nominal.thickness)
        # The nominal design meets the requirement but the epistemic spread
        # puts a third of the draws below it (0.34 in the full run).
        self.assertGreater(distribution.pnp_at_median_inputs, lab.PNP_REQUIREMENT)
        self.assertGreater(distribution.probability_below_requirement, 0.25)
        self.assertLess(distribution.probability_below_requirement, 0.45)

    def test_health_part(self) -> None:
        """CUSUM and GLR alarm within a few steps of the fault; no alarm without it.

        The full run prints CUSUM and GLR alarms at index 72, a located
        change at 70 and no alarm on the fault-free record of the same noise;
        the record is the same in the quick budget (only the number of study
        seeds differs), so the alarms are exact.
        """
        lab = self.lab
        result = quiet(lab.health_part, lab.QUICK)
        out = result["assessment"]
        self.assertEqual(out.cusum.alarm_index, 72)
        self.assertEqual(out.cusum.alarm_side, "lower")
        self.assertEqual(out.glr.alarm_index, 72)
        self.assertEqual(out.glr.change_index, 70)
        self.assertIsNone(result["quiet"].cusum.alarm_index)
        low, high = result["rul_interval"]
        self.assertAlmostEqual(low, 91.2, delta=2.0)
        self.assertAlmostEqual(high, 146.1, delta=3.0)
        true_eta = result["plant"].states[-1][2]
        estimated = out.filter.filtered[-1].mean[2]
        self.assertAlmostEqual(estimated, true_eta, delta=0.01)
        self.assertGreaterEqual(result["study"].points[0].detection_probability, 0.75)

    def test_entry_part(self) -> None:
        """FORM and the corridor are deterministic; simulation brackets FORM.

        The full run prints FORM 0.0369 (beta 1.788) and a corridor from
        -7.084 to -8.053 degrees. Crude Monte Carlo with 100 runs has a
        standard error of 0.02 and the 60-sample subset simulation a
        coefficient of variation of about 0.4, so the simulation bands are
        three standard errors and three coefficients of variation.
        """
        lab = self.lab
        result = quiet(lab.entry_part, lab.QUICK)
        form, subset, mc = result["form"], result["subset"], result["mc"]
        self.assertTrue(form.converged and subset.converged)
        self.assertAlmostEqual(form.probability, 0.0369, delta=5.0e-4)
        self.assertAlmostEqual(form.beta, 1.788, delta=0.01)
        self.assertLess(abs(mc.probability - form.probability), 3.0 * mc.standard_error)
        self.assertLess(
            abs(subset.probability - form.probability),
            3.0 * subset.coefficient_of_variation * subset.probability,
        )
        corridor = result["corridor"]
        self.assertAlmostEqual(math.degrees(corridor.shallow_angle), -7.084, delta=0.02)
        self.assertAlmostEqual(math.degrees(corridor.steep_angle), -8.053, delta=0.02)
        self.assertAlmostEqual(corridor.peak_g_at_steep, lab.G_LIMIT, delta=0.05)
        sweep = result["sweep"]
        self.assertEqual(len(sweep), lab.QUICK.sweep_points)
        # Steeper entries raise the peak load and lower the heat load.
        peaks = [peak for _, peak, _ in sweep]
        heats = [heat for _, _, heat in sweep]
        self.assertTrue(all(a < b for a, b in zip(peaks, peaks[1:], strict=False)))
        self.assertTrue(all(a > b for a, b in zip(heats, heats[1:], strict=False)))

    def test_summary_rows_cover_every_stage(self) -> None:
        """Each stage reports two rows with four non-empty cells."""
        lab = self.lab
        rows = []
        for stage in (
            lab.ascent_part,
            lab.tank_part,
            lab.mmod_part,
            lab.health_part,
            lab.entry_part,
        ):
            rows += quiet(stage, lab.QUICK)["rows"]
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [row[0] for row in rows[::2]], ["ascent", "tank", "MMOD", "engine", "entry"]
        )
        self.assertTrue(all(len(row) == 4 and all(row) for row in rows))


if __name__ == "__main__":
    unittest.main()
