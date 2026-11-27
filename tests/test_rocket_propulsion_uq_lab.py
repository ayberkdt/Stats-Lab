"""Regression tests of the rocket-propulsion laboratory and the functions it uses.

The lab (``examples/rocket_propulsion_uq_lab.py``) composes ``probstat.aerospace``
models with the propagation, sensitivity, rare-event, calibration and surrogate
tools. These tests import the lab without running ``main``, check its
closed-form building blocks against independent formulas, and run each of its
five parts with the small ``QUICK`` budget. Stochastic assertions use fixed
seeds and bands of several standard errors of the quick budget; nothing pins a
printed digit of the full run. Every case is synthetic.

References:
    [REF-SUTTON-BIBLARZ-2017], [REF-MELCHERS-BECK-2018],
    [REF-CHING-CHEN-2007]. Equations are evaluated by the library modules the
    lab calls ([EQ-AERO-LRE-010], [EQ-AERO-SRM-007]).

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

from probstat.aerospace import (
    BatesGrain,
    equilibrium_pressure,
    liquid_engine_model,
    simulate_solid_motor,
    solid_motor_model,
)
from probstat.distributions import Normal
from probstat.inference.transitional_mcmc import model_class_selection

LAB_PATH = Path(__file__).parents[1] / "examples" / "rocket_propulsion_uq_lab.py"


def load_lab() -> ModuleType:
    """Import the lab script as a module without running ``main``.

    Returns:
        The imported module.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    name = "_rocket_propulsion_lab_under_test"
    spec = importlib.util.spec_from_file_location(name, LAB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def quiet(function, *args):  # type: ignore[no-untyped-def]
    """Call a lab part and swallow what it prints.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    with redirect_stdout(io.StringIO()):
        return function(*args)


class LibraryRelationTests(unittest.TestCase):
    """Closed forms behind the lab's models."""

    def test_liquid_engine_vacuum_isp_has_the_closed_form_dispersion(self) -> None:
        """Isp_vac is linear in the efficiency to first order: CoV(Isp) ~ CoV(eta)."""
        model = liquid_engine_model(
            chamber_pressure=7.0e6,
            mixture_ratio=2.6,
            combustion_efficiency=Normal(0.95, 0.0095),
            throat_diameter=0.2,
            expansion_ratio=45.0,
            ambient_pressure=0.0,
            divergence_half_angle=0.2618,
        )
        points = model.inputs.sample(400, seed=3)
        values = model.evaluate_many(points)["vacuum_specific_impulse"]
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        # Isp ~ c* ~ eta: relative spread equals that of eta, 0.0095 / 0.95 = 1%
        # (the normal law has 5 sd below 1, so no draw is rejected).
        # The sample sd of 400 draws has a 3.5% relative standard error.
        self.assertAlmostEqual(sd / mean, 0.0095 / 0.95, delta=0.002)

    def test_equilibrium_pressure_closed_form(self) -> None:
        """The pressure is (a rho c* Kn / p_ref^n)^(1 / (1 - n)) for the lab's motor."""
        lab = load_lab()
        a, n, kn = 8.0e-3, 0.35, 250.0
        expected = (a * 1750.0 * 1550.0 * kn / 6.0e6**n) ** (1.0 / (1.0 - n))
        got = equilibrium_pressure(
            a, n, 1550.0, 1750.0, kn, reference_pressure=lab.MOTOR.reference_pressure
        )
        self.assertAlmostEqual(got / expected, 1.0, places=12)

    def test_plateau_pressure_matches_the_simulated_initial_pressure(self) -> None:
        """The static-fire pressure is the first point of the quasi-steady trace."""
        lab = load_lab()
        from dataclasses import replace

        motor = replace(lab.MOTOR, erosion_rate=0.0)
        trace = simulate_solid_motor(lab.GRAIN, motor, steps=20)
        got = lab.plateau_pressure(
            motor.burn_rate_coefficient,
            motor.pressure_exponent,
            motor.temperature_sensitivity,
            lab.REFERENCE_K,
            motor.throat_diameter,
        )
        self.assertAlmostEqual(got / trace.pressures[0], 1.0, places=9)

    def test_temperature_raises_the_plateau_pressure(self) -> None:
        """Pressure scales as a(T)^(1/(1-n)), so log(hot/cold) = sigma dT / (1 - n)."""
        lab = load_lab()
        cold = lab.plateau_pressure(8e-3, 0.4, 2.5e-3, 273.15, 0.024)
        hot = lab.plateau_pressure(8e-3, 0.4, 2.5e-3, 323.15, 0.024)
        self.assertAlmostEqual(
            math.log(hot / cold), 2.5e-3 * 50.0 / (1.0 - 0.4), places=12
        )

    def test_burst_margin_model_has_the_documented_inputs(self) -> None:
        """The solid-motor model exposes the nine inputs the lab maps back."""
        lab = load_lab()
        model = solid_motor_model(lab.GRAIN, lab.MOTOR, lab.BURST_MEDIAN, steps=20)
        self.assertEqual(len(model.inputs.names), 9)
        values = dict(zip(model.inputs.names, model.inputs.means(), strict=True))
        realized = lab.motor_from_inputs(values)
        self.assertAlmostEqual(realized.throat_diameter, values["d_throat"])
        self.assertEqual(realized.pressure_exponent, values["n"])
        self.assertIsInstance(lab.GRAIN, BatesGrain)


class LabBlockTests(unittest.TestCase):
    """The likelihood, the prior and the synthetic data of the calibration."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import the lab once."""
        cls.lab = load_lab()

    def test_log_likelihood_equals_the_lognormal_density(self) -> None:
        """One test: log of the density of p when log p is N(log m, s^2)."""
        lab = self.lab
        tests = [(293.15, 0.024)]
        model = lab.plateau_pressure(8e-3, 0.35, 0.0, 293.15, 0.024)
        observed = 1.03 * model
        s = lab.NOISE_SCALE
        density = math.exp(-0.5 * (math.log(1.03) / s) ** 2) / (
            observed * s * math.sqrt(2.0 * math.pi)
        )
        got = lab.log_likelihood((math.log(8e-3), 0.35), False, tests, [observed])
        self.assertAlmostEqual(got, math.log(density), places=10)

    def test_prior_support_and_data_reproducibility(self) -> None:
        """The prior rejects n outside (0, 0.95) and negative sigma; data repeat."""
        lab = self.lab
        self.assertEqual(lab.log_prior((-4.8, 1.0), False), -math.inf)
        self.assertEqual(lab.log_prior((-4.8, 0.4, -1e-3), True), -math.inf)
        self.assertTrue(math.isfinite(lab.log_prior((-4.8, 0.4, 2e-3), True)))
        tests = lab.static_fire_tests(False)
        self.assertEqual(len(tests), 9)
        self.assertEqual(lab.synthetic_data(tests, 5), lab.synthetic_data(tests, 5))
        self.assertTrue(
            all(t == lab.REFERENCE_K for t, _ in lab.static_fire_tests(True))
        )

    def test_synthetic_data_scatter_about_the_generating_law(self) -> None:
        """log(data / truth) has the stated noise scale: |mean| < 4 s / sqrt(9)."""
        lab = self.lab
        tests = lab.static_fire_tests(False)
        data = lab.synthetic_data(tests, 7)
        z = [
            math.log(p / lab.plateau_pressure(*(lab.TRUTH.values()), t, d))
            for p, (t, d) in zip(data, tests, strict=True)
        ]
        self.assertLess(abs(sum(z) / 9.0), 4.0 * lab.NOISE_SCALE / 3.0)
        self.assertLess(max(abs(v) for v in z), 4.0 * lab.NOISE_SCALE)


class LabPartTests(unittest.TestCase):
    """Each part of the lab runs with the quick budget and stays plausible."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import the lab once."""
        cls.lab = load_lab()

    def test_engine_part(self) -> None:
        """FORM and simulation agree on the order of magnitude; efficiency leads."""
        lab = self.lab
        result = quiet(lab.engine_part, lab.QUICK)
        rows = {row.method: row for row in result["comparison"].rows}
        self.assertEqual(set(rows), {"FORM", "Subset", "MC"})
        form = rows["FORM"].probability
        # Isp_vac is nearly linear in its inputs, so FORM is close to exact; the
        # subset estimate carries a c.o.v. of about 0.2-0.3 at this budget.
        self.assertGreater(form, 1.0e-3)
        self.assertLess(form, 2.0e-2)
        self.assertLess(abs(math.log(rows["Subset"].probability / form)), 1.2)
        self.assertEqual(
            set(result["sobol"].ranking()[:2]),
            {"combustion_efficiency", "mixture_ratio"},
        )

    def test_motor_part(self) -> None:
        """Subset simulation converges; FORM derivatives have the physical signs."""
        lab = self.lab
        result = quiet(lab.motor_part, lab.QUICK)
        self.assertTrue(result["subset"].converged)
        self.assertGreater(result["subset"].probability, 1.0e-4)
        self.assertLess(result["subset"].probability, 3.0e-2)
        form = result["form"]["result"]
        self.assertTrue(form.converged)
        # dbeta/dmu: burn-rate coefficient and c* raise the pressure (beta falls),
        # the burst pressure raises beta.
        self.assertLess(form.dbeta[0], 0.0)
        self.assertLess(form.dbeta[1], 0.0)
        self.assertGreater(form.dbeta[2], 0.0)
        self.assertEqual(len(result["traces"]), lab.QUICK.traces)

    def test_calibration_part(self) -> None:
        """Temperature data decide the classes; ambient data cannot."""
        lab = self.lab
        result = quiet(lab.calibration_part, lab.QUICK)
        full = model_class_selection(result["results"]["all temperatures"])
        self.assertGreater(full.posterior_probabilities[1], 0.99)
        ambient = model_class_selection(result["results"]["ambient only"])
        # At T = T_ref the temperature term does not enter the likelihood, so the
        # evidences agree up to the Monte Carlo error of 150-particle TMCMC.
        self.assertGreater(ambient.posterior_probabilities[1], 0.1)
        self.assertLess(ambient.posterior_probabilities[1], 0.9)
        run = result["results"]["all temperatures"]["Vieille + T"]
        mean_n = sum(s[1] for s in run.particles) / len(run.particles)
        self.assertAlmostEqual(mean_n, 0.36, delta=0.06)
        self.assertTrue(all(7.0e6 < p < 8.0e6 for p in result["peaks"]))

    def test_instability_part(self) -> None:
        """The GP classifies like the true model; the leading indices are n and tau."""
        lab = self.lab
        result = quiet(lab.instability_part, lab.QUICK)
        self.assertGreater(result["p_mc"], 0.05)
        self.assertLess(result["p_mc"], 0.35)
        self.assertAlmostEqual(result["p_gp"], result["p_mc"], delta=0.05)
        sobol = result["sobol"]
        order = sorted(
            range(len(result["names"])), key=lambda i: -sobol.first_order_mean[i]
        )
        self.assertEqual({result["names"][i] for i in order[:2]}, {"n", "tau"})

    def test_cluster_part(self) -> None:
        """Monte Carlo mean agrees with the exact moment within five standard errors."""
        lab = self.lab
        result = quiet(lab.cluster_part, lab.QUICK)
        exact_mean, exact_variance = result["exact"]
        distribution = result["distribution"]
        error = math.sqrt(exact_variance / lab.QUICK.cluster_samples)
        self.assertLess(abs(distribution.mean - exact_mean), 5.0 * error)
        self.assertGreater(distribution.interval[0], 0.6)
        self.assertLess(distribution.interval[1], 1.0)


class ModelSelectionArithmeticTests(unittest.TestCase):
    """Posterior model probabilities from log evidences."""

    def test_probabilities_follow_the_bayes_factor(self) -> None:
        """P(M1|D) = 1 / (1 + exp(logZ0 - logZ1)) for equal priors."""
        selection = model_class_selection({"a": -10.0, "b": -8.0})
        self.assertAlmostEqual(
            selection.posterior_probabilities[1], 1.0 / (1.0 + math.exp(-2.0)), 12
        )


if __name__ == "__main__":
    unittest.main()
