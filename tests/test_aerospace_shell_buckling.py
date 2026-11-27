"""Tests for probabilistic buckling of thin-walled circular cylinders.

All numbers are synthetic (aluminium-alloy class launch-vehicle tank,
R = 1.5 m, t = 3 mm). The Koiter constant is the default of
``koiter_coefficient`` (equal to the SP-8007 Eq. 138 form); tests on the Koiter
law itself use an explicit coefficient. Tests that mention measured
numbers record what was measured with the stated seeds.
"""

from __future__ import annotations

import math
import os
import unittest

from probstat.aerospace.shell_buckling import (
    classical_buckling_load,
    classical_buckling_stress,
    compare_knockdowns,
    design_thickness_for_reliability,
    koiter_coefficient,
    koiter_imperfection_for_knockdown,
    koiter_knockdown,
    koiter_knockdown_derivative,
    lognormal_from_mean_cov,
    probabilistic_knockdown,
    shell_buckling_model,
    sp8007_axial_knockdown,
    sp8007_buckling_load,
    sp8007_design_thickness,
    weibull_from_mean_cov,
)
from probstat.distributions import LogNormal
from probstat.optimization import brent_root
from probstat.reliability import (
    ModelLimitState,
    form_hlrf,
    model_monte_carlo,
    subset_simulation,
)
from probstat.sensitivity import sobol_analysis

E = 71e9
T = 3e-3
R = 1.5
NU = 0.33
C = 4.9  # an explicit Koiter constant for the law tests


def _model(load_mean: float = 1.0e6, weibull: bool = False):
    """Return the standard test model (imperfection mean 0.1 t, c.o.v. 0.5)."""
    imperfection = (
        weibull_from_mean_cov(0.3e-3, 0.5)
        if weibull
        else lognormal_from_mean_cov(0.3e-3, 0.5)
    )
    return shell_buckling_model(
        youngs_modulus=lognormal_from_mean_cov(E, 0.05),
        thickness=lognormal_from_mean_cov(T, 0.03),
        radius=R,
        imperfection=imperfection,
        applied_load=lognormal_from_mean_cov(load_mean, 0.1),
        poisson_ratio=NU,
    )


class ClassicalTests(unittest.TestCase):
    """Classical buckling stress and load."""

    def test_closed_form_value(self) -> None:
        """sigma_cl = E t / (R sqrt(3 (1 - nu^2))) at the synthetic case."""
        expected = 71e9 * 3e-3 / (1.5 * math.sqrt(3.0 * (1.0 - 0.33**2)))
        self.assertAlmostEqual(
            classical_buckling_stress(E, T, R, NU) / expected, 1.0, places=14
        )
        self.assertAlmostEqual(expected / 1e6, 86.8, delta=0.1)

    def test_sp8007_numeric_form_for_nu_03(self) -> None:
        """The report's 0.605 E t / r for nu = 0.3 (Eq. 8)."""
        ratio = classical_buckling_stress(1.0, 1.0, 1.0, 0.3)
        self.assertAlmostEqual(ratio, 0.605, delta=5e-4)

    def test_load_is_stress_times_area(self) -> None:
        """P_cl = 2 pi R t sigma_cl and the t^2 scaling."""
        load = classical_buckling_load(E, T, R, NU)
        self.assertAlmostEqual(
            load / (2 * math.pi * R * T * classical_buckling_stress(E, T, R, NU)),
            1.0,
            places=14,
        )
        self.assertAlmostEqual(
            classical_buckling_load(E, 2 * T, R, NU) / load, 4.0, places=12
        )

    def test_validation(self) -> None:
        """Invalid inputs raise."""
        with self.assertRaises(ValueError):
            classical_buckling_load(-1.0, T, R)
        with self.assertRaises(ValueError):
            classical_buckling_load(E, T, R, 0.5)
        with self.assertRaises(TypeError):
            classical_buckling_load("a", T, R)  # type: ignore[arg-type]


class KoiterTests(unittest.TestCase):
    """The Koiter relation (1 - lambda)^2 = c lambda d."""

    def test_limits_and_relation(self) -> None:
        """lambda(0) = 1 and the relation holds to rounding."""
        self.assertEqual(koiter_knockdown(0.0, C), 1.0)
        for d in (1e-6, 1e-3, 0.05, 0.3, 2.0, 50.0):
            lam = koiter_knockdown(d, C)
            self.assertLess(abs((1 - lam) ** 2 - C * lam * d), 1e-12 * (1 + C * d))
            self.assertTrue(0.0 < lam < 1.0)

    def test_monotone_decrease(self) -> None:
        """Lambda strictly decreases in d."""
        values = [koiter_knockdown(d, C) for d in (0.0, 0.01, 0.05, 0.1, 0.5, 5.0)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:], strict=False)))

    def test_matches_independent_root_find(self) -> None:
        """Brent on the original relation, branch lambda <= 1, agrees."""
        for d in (0.01, 0.1, 1.0):
            root = brent_root(
                lambda lam, d=d: (1 - lam) ** 2 - C * lam * d,
                1e-9,
                1.0,
                tolerance=1e-14,
            )
            self.assertAlmostEqual(root.root, koiter_knockdown(d, C), places=10)

    def test_inverse_round_trip(self) -> None:
        """d(lambda(d)) = d."""
        for d in (1e-4, 0.1, 3.0):
            lam = koiter_knockdown(d, C)
            self.assertAlmostEqual(
                koiter_imperfection_for_knockdown(lam, C) / d, 1.0, places=9
            )

    def test_small_imperfection_asymptote(self) -> None:
        """Lambda ~ 1 - sqrt(c d) with an error below c d (second term x / 2)."""
        for d in (1e-6, 1e-4):
            x = C * d
            error = abs(koiter_knockdown(d, C) - (1.0 - math.sqrt(x)))
            self.assertLess(error, x)

    def test_derivative_and_asymptotic_slope(self) -> None:
        """Implicit derivative equals central differences; slope -> asymptote."""
        for d in (0.01, 0.2, 1.0):
            h = 1e-6 * d
            fd = (koiter_knockdown(d + h, C) - koiter_knockdown(d - h, C)) / (2 * h)
            self.assertAlmostEqual(koiter_knockdown_derivative(d, C) / fd, 1.0, 6)
        for d in (1e-4, 1e-6, 1e-8):
            slope = -math.sqrt(C) / (2.0 * math.sqrt(d))
            ratio = koiter_knockdown_derivative(d, C) / slope
            self.assertLess(abs(ratio - 1.0), 2.0 * math.sqrt(C * d))

    def test_default_coefficient(self) -> None:
        """The default constant is (3/2) sqrt(3 (1 - nu^2)) = 2 alpha."""
        self.assertAlmostEqual(
            koiter_coefficient(0.3), 1.5 * math.sqrt(3 * 0.91), places=14
        )
        self.assertAlmostEqual(koiter_coefficient(0.33), 2.4525, places=4)

    def test_equals_sp8007_eq_138(self) -> None:
        """Closed form equals gamma = 1 + a e - sqrt(a e (2 + a e)), a = 3/4 sqrt(..).

        This is Eqs. 138-139 of NASA/SP-8007-2020/REV 2, evaluated independently.
        """
        for nu in (0.0, 0.3, 0.33, 0.4):
            alpha = 0.75 * math.sqrt(3.0 * (1.0 - nu * nu))
            for eps in (1e-4, 0.01, 0.1, 0.5, 2.0, 10.0):
                ae = alpha * eps
                gamma = 1.0 + ae - math.sqrt(ae * (2.0 + ae))
                lam = koiter_knockdown(eps, koiter_coefficient(nu))
                self.assertAlmostEqual(lam, gamma, delta=1e-12)

    def test_validation(self) -> None:
        """Invalid inputs raise."""
        with self.assertRaises(ValueError):
            koiter_knockdown(-0.1, C)
        with self.assertRaises(ValueError):
            koiter_knockdown(0.1, 0.0)
        with self.assertRaises(ValueError):
            koiter_imperfection_for_knockdown(0.0, C)
        with self.assertRaises(ValueError):
            koiter_knockdown_derivative(0.0, C)


class SP8007Tests(unittest.TestCase):
    """The SP-8007 knockdown (Eqs. 9 and 10 of the 2020 revision)."""

    def test_value_by_hand(self) -> None:
        """R/t = 100: phi = 0.625, gamma = 1 - 0.901 (1 - exp(-0.625))."""
        gamma = sp8007_axial_knockdown(100.0, 1.0)
        self.assertAlmostEqual(gamma, 0.581271, places=5)

    def test_monotone_and_limits(self) -> None:
        """Gamma decreases with R/t, stays in (0.099, 1); R/t >= 1500 raises."""
        values = [sp8007_axial_knockdown(r, 1.0) for r in (1.0, 10, 100, 500, 1499)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:], strict=False)))
        self.assertTrue(0.099 < values[-1] < values[0] < 1.0)
        with self.assertRaises(ValueError):
            sp8007_axial_knockdown(1500.0, 1.0)

    def test_load_and_design_thickness_round_trip(self) -> None:
        """Gamma P_cl at the design thickness equals the factored load."""
        wall = sp8007_design_thickness(0.6e6, E, R, NU, safety_factor=1.25)
        self.assertAlmostEqual(
            sp8007_buckling_load(E, wall, R, NU) / (0.6e6 * 1.25), 1.0, places=9
        )
        thicker = sp8007_design_thickness(0.6e6, E, R, NU, safety_factor=2.0)
        self.assertGreater(thicker, wall)

    def test_design_out_of_range(self) -> None:
        """A load that no admissible thickness carries raises."""
        with self.assertRaises(ValueError):
            sp8007_design_thickness(1e15, E, R, NU)
        with self.assertRaises(ValueError):
            sp8007_design_thickness(1.0, E, R, NU, safety_factor=0.5)


class LawsAndKnockdownTests(unittest.TestCase):
    """Imperfection laws and the probabilistic knockdown."""

    def test_mean_cov_constructors(self) -> None:
        """Both constructors reproduce mean and c.o.v. exactly."""
        for law in (
            lognormal_from_mean_cov(2.0, 0.4),
            weibull_from_mean_cov(2.0, 0.4),
        ):
            self.assertAlmostEqual(law.mean, 2.0, places=9)
            self.assertAlmostEqual(math.sqrt(law.variance) / 2.0, 0.4, places=8)
        with self.assertRaises(ValueError):
            weibull_from_mean_cov(1.0, 100.0)

    def test_probability_of_being_below_equals_target(self) -> None:
        """P(lambda < k_p) = p exactly through the imperfection cdf."""
        laws = (
            lognormal_from_mean_cov(0.3e-3, 0.5),
            weibull_from_mean_cov(0.3e-3, 0.5),
        )
        for law in laws:
            for p in (0.5, 0.05, 1e-3):
                k = probabilistic_knockdown(law, T, p, C)
                d = koiter_imperfection_for_knockdown(k, C)
                self.assertAlmostEqual(1.0 - law.cdf(d * T), p, delta=1e-9)
            self.assertLess(
                probabilistic_knockdown(law, T, 1e-3, C),
                probabilistic_knockdown(law, T, 0.5, C),
            )

    def test_sampling_agrees_with_quantile_knockdown(self) -> None:
        """Monte Carlo fraction of lambda < k_p is p within 4 standard errors."""
        law = lognormal_from_mean_cov(0.3e-3, 0.5)
        p, n = 0.05, 20000
        k = probabilistic_knockdown(law, T, p, C)
        draws = law.sample(n, seed=11)
        fraction = sum(koiter_knockdown(w / T, C) < k for w in draws) / n
        self.assertLess(abs(fraction - p), 4.0 * math.sqrt(p * (1 - p) / n))

    def test_comparison_with_sp8007_is_assumption_dependent(self) -> None:
        """Measured with the default constant, p = 0.01, lognormal c.o.v. 0.5.

        Mean imperfection 0.3 mm: k_p = 0.602, 0.454, 0.335, 0.275 against
        SP-8007 0.471, 0.322, 0.224, 0.182 at R/t = 200, 500, 1000, 1450, so
        SP-8007 is below k_p everywhere. Mean 1.5 mm: k_p = 0.196 at R/t = 500,
        below 0.322, so the verdict reverses; it belongs to the assumed law.
        """
        law = lognormal_from_mean_cov(0.3e-3, 0.5)
        c = koiter_coefficient(NU)
        table = compare_knockdowns(law, R, (200.0, 500.0, 1000.0, 1450.0), 0.01, c)
        self.assertEqual(table.sp8007_conservative, (True, True, True, True))
        self.assertAlmostEqual(table.probabilistic[1], 0.454, delta=0.001)
        self.assertAlmostEqual(table.sp8007[1], 0.322, delta=0.001)
        bad = lognormal_from_mean_cov(1.5e-3, 0.5)
        worse = compare_knockdowns(bad, R, (500.0,), 0.01, c)
        self.assertEqual(worse.sp8007_conservative, (False,))
        self.assertAlmostEqual(worse.probabilistic[0], 0.196, delta=0.001)
        with self.assertRaises(ValueError):
            compare_knockdowns(law, R, (), 0.01, C)


class ModelTests(unittest.TestCase):
    """The probabilistic model, reliability methods and sensitivity."""

    def test_outputs_match_the_closed_forms(self) -> None:
        """At the nominal point every output equals the hand formula."""
        model = shell_buckling_model(
            youngs_modulus=E,
            thickness=lognormal_from_mean_cov(T, 0.03),
            radius=R,
            imperfection=0.3e-3,
            applied_load=0.5e6,
            poisson_ratio=NU,
            coefficient=C,
        )
        out = model.evaluate({"thickness": T})
        lam = koiter_knockdown(0.1, C)
        pcl = classical_buckling_load(E, T, R, NU)
        self.assertAlmostEqual(out["knockdown"], lam, places=14)
        self.assertAlmostEqual(out["buckling_load"] / (lam * pcl), 1.0, places=14)
        self.assertAlmostEqual(out["margin"], lam * pcl - 0.5e6, places=4)

    def test_validation(self) -> None:
        """No random input and bad types are rejected."""
        with self.assertRaises(ValueError):
            shell_buckling_model(
                youngs_modulus=E,
                thickness=T,
                radius=R,
                imperfection=0.0,
                applied_load=1e5,
            )
        with self.assertRaises(TypeError):
            shell_buckling_model(
                youngs_modulus=True,
                thickness=T,
                radius=R,
                imperfection=lognormal_from_mean_cov(1e-3, 0.5),
                applied_load=1e5,
            )

    def test_form_subset_and_monte_carlo_agree(self) -> None:
        """Lognormal imperfection, load 1.0 MN.

        Measured (seeds 1 and 2): FORM 0.0094, MC 0.01035 (SE 0.0007, 20000
        draws), subset simulation 0.0079 (c.o.v. upper bound 0.42).
        """
        state = ModelLimitState(_model(), "margin", 0.0, "below")
        mc = model_monte_carlo(state, 20000, seed=1)
        form = form_hlrf(state)
        subset = subset_simulation(state, samples_per_level=500, seed=2)
        self.assertTrue(form.converged and subset.converged)
        self.assertAlmostEqual(form.probability, mc.probability, delta=0.003)
        combined = math.hypot(
            mc.standard_error,
            subset.coefficient_of_variation_upper * subset.probability,
        )
        self.assertLess(abs(subset.probability - mc.probability), 3.0 * combined)
        self.assertGreater(form.importance_factors["imperfection"], 0.5)

    def test_form_matches_monte_carlo_with_weibull_imperfection(self) -> None:
        """Weibull imperfection of the same mean and c.o.v.: FORM within 0.01."""
        state = ModelLimitState(_model(weibull=True), "margin", 0.0, "below")
        mc = model_monte_carlo(state, 20000, seed=3)
        form = form_hlrf(state)
        self.assertTrue(form.converged)
        self.assertAlmostEqual(form.probability, mc.probability, delta=0.01)

    def test_sobol_ranking(self) -> None:
        """The imperfection dominates the buckling-load variance.

        Measured (size 256, seed 1): first-order 0.62 (imperfection), 0.24
        (thickness), 0.14 (modulus); the applied load has no effect.
        """
        result = sobol_analysis(_model(), "buckling_load", 128, seed=1)
        total = {
            n: i.estimate
            for n, i in zip(result.input_names, result.total_effect, strict=True)
        }
        ranked = sorted(total, key=lambda k: total[k], reverse=True)
        self.assertEqual(ranked[:2], ["imperfection", "thickness"])
        self.assertLess(abs(total["applied_load"]), 0.05)


class DesignTests(unittest.TestCase):
    """Reliability-based thickness against the deterministic SP-8007 design."""

    def _design(self, p: float, load: float = 0.6e6):
        return design_thickness_for_reliability(
            target_probability=p,
            youngs_modulus=lognormal_from_mean_cov(E, 0.05),
            radius=R,
            imperfection=lognormal_from_mean_cov(0.3e-3, 0.5),
            applied_load=load,
            poisson_ratio=NU,
        )

    def test_target_attained_and_monotone(self) -> None:
        """FORM probability equals the target; a smaller target needs more wall."""
        loose = self._design(1e-2)
        tight = self._design(1e-4)
        self.assertTrue(loose.converged and tight.converged)
        self.assertAlmostEqual(loose.probability / 1e-2, 1.0, places=6)
        self.assertAlmostEqual(tight.probability / 1e-4, 1.0, places=6)
        self.assertGreater(tight.thickness, loose.thickness)
        self.assertGreater(self._design(1e-2, 0.9e6).thickness, loose.thickness)

    def test_monte_carlo_confirms_the_design(self) -> None:
        """At the 2 % design thickness MC gives a probability within 0.008."""
        design = self._design(2e-2)
        model = shell_buckling_model(
            youngs_modulus=lognormal_from_mean_cov(E, 0.05),
            thickness=lognormal_from_mean_cov(design.thickness, 0.03),
            radius=R,
            imperfection=lognormal_from_mean_cov(0.3e-3, 0.5),
            applied_load=0.6e6,
            poisson_ratio=NU,
        )
        state = ModelLimitState(model, "margin", 0.0, "below")
        mc = model_monte_carlo(state, 20000, seed=5)
        self.assertAlmostEqual(mc.probability, 2e-2, delta=0.008)

    def test_comparison_with_deterministic_design(self) -> None:
        """Probabilistic design against SP-8007 at the same load.

        Measured (load 0.6 MN, default Koiter constant, lognormal modulus
        with c.o.v. 5 %, imperfection mean 0.3 mm): SP-8007 gives 2.686 mm;
        the probabilistic design gives 2.532 mm at 1e-3 and 2.692 mm at 1e-4.
        The order depends on the assumed imperfection law and the target.
        """
        det = sp8007_design_thickness(0.6e6, E, R, NU)
        prob = design_thickness_for_reliability(
            target_probability=1e-3,
            youngs_modulus=LogNormal(math.log(E), 0.05),
            radius=R,
            imperfection=lognormal_from_mean_cov(0.3e-3, 0.5),
            applied_load=0.6e6,
            poisson_ratio=NU,
        )
        self.assertLess(prob.thickness, det)
        self.assertAlmostEqual(prob.thickness, 2.532e-3, delta=2e-5)
        tight = self._design(1e-4)
        self.assertAlmostEqual(tight.thickness, 2.692e-3, delta=2e-5)

    def test_validation(self) -> None:
        """Bad targets and a missing bracket for a random radius raise."""
        with self.assertRaises(ValueError):
            self._design(0.7)
        with self.assertRaises(ValueError):
            design_thickness_for_reliability(
                target_probability=1e-2,
                youngs_modulus=E,
                radius=lognormal_from_mean_cov(R, 0.01),
                imperfection=lognormal_from_mean_cov(0.3e-3, 0.5),
                applied_load=0.6e6,
            )


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowTests(unittest.TestCase):
    """Larger Monte Carlo agreement check."""

    def test_large_monte_carlo(self) -> None:
        """200000 draws against FORM at load 0.9 MN."""
        state = ModelLimitState(_model(0.9e6), "margin", 0.0, "below")
        mc = model_monte_carlo(state, 200000, seed=7)
        form = form_hlrf(state)
        self.assertAlmostEqual(form.probability, mc.probability, delta=0.002)


if __name__ == "__main__":
    unittest.main()
