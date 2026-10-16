"""Sobol intervals, shared designs, DGSM, given-data indices, FORM autodiff.

Workstream W8 (ADR-0065).
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.numerics import AutodiffUnsupportedError, admath
from probstat.reliability import ModelLimitState, form_hlrf
from probstat.sensitivity import (
    dgsm,
    estimate_sobol_indices,
    given_data_first_order,
    poincare_constant,
    sobol_analysis,
    sobol_analysis_all,
)
from probstat.uq import QuantityOfInterest

# Ishigami (a = 7, b = 0.1) closed forms, Sobol' & Levitan / Saltelli et al.
ISHIGAMI_FIRST = (0.3139051827, 0.4424111333, 0.0)
ISHIGAMI_TOTAL = (0.5575888667, 0.4424111333, 0.2436836840)
ISHIGAMI_VARIANCE = 13.8445869945


def ishigami(use_admath: bool) -> ProbabilisticModel:
    """Return the Ishigami model; ``use_admath`` makes it dual-capable."""
    sin = admath.sin if use_admath else math.sin
    inputs = JointInput.independent(
        *(RandomInput(f"x{i}", Uniform(-math.pi, math.pi)) for i in (1, 2, 3))
    )

    def model(p):
        """Ishigami function and a second, additive QoI."""
        return {
            "y": sin(p["x1"])
            + 7.0 * sin(p["x2"]) ** 2
            + 0.1 * p["x3"] ** 4 * sin(p["x1"]),
            "z": p["x1"] + 2.0 * p["x2"],
        }

    return ProbabilisticModel(
        inputs,
        model,
        (QuantityOfInterest("y", "Ishigami"), QuantityOfInterest("z", "additive")),
    )


class SobolIntervalTests(unittest.TestCase):
    """Bootstrap only for i.i.d. rows; replicates for quasi-random designs."""

    def test_interval_method_depends_on_design(self) -> None:
        """Mc -> bootstrap, sobol/lhs -> replicates, no request -> none."""
        model = ishigami(False)
        mc = sobol_analysis(model, "y", 512, design="mc", seed=1, bootstrap=50)
        self.assertEqual(
            (mc.interval_method, mc.bootstrap, mc.replicates), ("bootstrap", 50, 0)
        )
        qmc = sobol_analysis(model, "y", 512, design="sobol", seed=1, bootstrap=50)
        self.assertEqual(
            (qmc.interval_method, qmc.bootstrap, qmc.replicates), ("replicates", 0, 8)
        )
        self.assertEqual(qmc.evaluations, 512 * 5)
        lhs = sobol_analysis(model, "y", 300, design="lhs", seed=1, replicates=5)
        self.assertEqual((lhs.interval_method, lhs.replicates), ("replicates", 5))
        none = sobol_analysis(model, "y", 256, seed=1)
        self.assertEqual(none.interval_method, "none")
        self.assertTrue(math.isnan(none.first_order[0].lower))
        with self.assertRaises(ValueError):
            sobol_analysis(model, "y", 256, seed=1, replicates=1)
        with self.assertRaises(ValueError):
            sobol_analysis(model, "y", 256, seed=1, design="grid")

    def test_second_order_is_absent_not_zero(self) -> None:
        """sobol_analysis does not estimate S_ij and says so with None."""
        result = sobol_analysis(ishigami(False), "y", 128, seed=2)
        self.assertIsNone(result.result.second_order)
        full = estimate_sobol_indices(
            lambda r: r[0] * r[1],
            [(random.random(), random.random()) for _ in range(64)],
            [(random.random(), random.random()) for _ in range(64)],
        )
        self.assertIsNotNone(full.second_order)

    def test_replicate_intervals_are_honest(self) -> None:
        """Replicate intervals cover the truth at about the nominal rate.

        Across 40 seeds of a scrambled-Sobol analysis (8 replicates of 64
        rows), the 95 % t-intervals for the six Ishigami indices must cover
        the analytic values in at least 85 % of the cases, and their mean
        half-width must be within a factor 2.5 of 1.96 times the observed
        seed-to-seed standard deviation of the estimates (the row bootstrap
        under QMC that this replaces was off by about 30x, audit item 0.10).
        """
        model = ishigami(False)
        truths = ISHIGAMI_FIRST + ISHIGAMI_TOTAL
        covered = 0
        half_widths = [[] for _ in range(6)]
        estimates = [[] for _ in range(6)]
        for seed in range(40):
            result = sobol_analysis(model, "y", 512, seed=seed, replicates=8)
            items = result.first_order + result.total_effect
            for k, (item, truth) in enumerate(zip(items, truths, strict=True)):
                covered += item.lower <= truth <= item.upper
                half_widths[k].append(0.5 * (item.upper - item.lower))
                estimates[k].append(item.estimate)
        self.assertGreaterEqual(covered / (40 * 6), 0.85)
        for k in range(6):
            mean = sum(estimates[k]) / 40
            spread = math.sqrt(sum((e - mean) ** 2 for e in estimates[k]) / 39)
            if spread == 0.0:
                continue
            ratio = (sum(half_widths[k]) / 40) / (1.96 * spread)
            self.assertGreater(ratio, 1 / 2.5, k)
            self.assertLess(ratio, 2.5, k)

    def test_one_design_serves_every_quantity(self) -> None:
        """All QoIs share size (d + 2) evaluations and match single runs."""
        model = ishigami(False)
        before = model.evaluations
        shared = sobol_analysis_all(model, 256, seed=4, bootstrap=1)
        self.assertEqual(model.evaluations - before, 256 * 5)
        self.assertEqual(shared.evaluations, 256 * 5)
        self.assertEqual(shared.quantities, ("y", "z"))
        for name in ("y", "z"):
            single = sobol_analysis(ishigami(False), name, 256, seed=4, bootstrap=1)
            self.assertEqual(shared[name].first_order, single.first_order)
            self.assertEqual(shared[name].total_effect, single.total_effect)
        # z = x1 + 2 x2 is additive: S = S_T = (1/5, 4/5, 0).
        z = shared["z"]
        for got, want in zip(z.total_effect, (0.2, 0.8, 0.0), strict=True):
            self.assertAlmostEqual(got.estimate, want, delta=0.02)
        with self.assertRaises(KeyError):
            sobol_analysis_all(model, 64, quantities=("missing",))


class DerivativeBasedTests(unittest.TestCase):
    """DGSM against closed forms and the Poincare bound."""

    def test_ishigami_nu_closed_form(self) -> None:
        """nu_i of Ishigami from E[cos^2], E[x^4], E[x^6], E[x^8] on U(-pi, pi).

        ``nu_1 = E[cos^2 x1] E[(1 + 0.1 x3^4)^2] = (1/2)(1 + 0.04 pi^4 + 0.01 pi^8/9)``,
        ``nu_2 = 196 E[sin^2 cos^2] = 24.5``, ``nu_3 = 0.16 E[x^6] / 2 =
        0.08 pi^6 / 7``.
        """
        exact = (
            0.5 * (1.0 + 0.04 * math.pi**4 + 0.01 * math.pi**8 / 9.0),
            24.5,
            0.08 * math.pi**6 / 7.0,
        )
        auto = dgsm(ishigami(True), "y", 4096, seed=3)
        self.assertEqual(auto.gradient_method, "autodiff")
        self.assertEqual(auto.evaluations, 4096)
        fd = dgsm(ishigami(False), "y", 4096, seed=3)
        self.assertEqual(fd.gradient_method, "finite-difference")
        self.assertIn("AutodiffUnsupportedError", fd.gradient_fallback_reason)
        self.assertEqual(fd.evaluations, 4096 * 7)
        for got, also, want in zip(auto.nu, fd.nu, exact, strict=True):
            self.assertAlmostEqual(got / want, 1.0, delta=0.03)
            self.assertAlmostEqual(got / also, 1.0, delta=1e-6)
        # The bound C nu / D with C = (2 pi)^2 / pi^2 = 4 dominates S_T.
        for bound, total in zip(
            auto.total_index_upper_bounds, ISHIGAMI_TOTAL, strict=True
        ):
            self.assertGreater(bound, total)
        with self.assertRaises(AutodiffUnsupportedError):
            dgsm(ishigami(False), "y", 16, seed=3, gradient="autodiff")

    def test_bound_is_attained_for_its_extremal_functions(self) -> None:
        """cos(pi x) on U(0, 1) and a linear map of normals give equality."""
        uniform = ProbabilisticModel(
            JointInput.independent(RandomInput("x", Uniform(0.0, 1.0))),
            lambda p: {"y": admath.cos(math.pi * p["x"])},
            (QuantityOfInterest("y", "cosine"),),
        )
        result = dgsm(uniform, "y", 2048, seed=1)
        self.assertAlmostEqual(result.total_index_upper_bounds[0], 1.0, delta=0.01)
        weights = (1.0, 2.0, 0.0)
        sigmas = (1.0, 0.5, 3.0)
        normal = ProbabilisticModel(
            JointInput.independent(
                *(RandomInput(f"x{i}", Normal(0.0, s)) for i, s in enumerate(sigmas))
            ),
            lambda p: {"y": sum(w * p[f"x{i}"] for i, w in enumerate(weights))},
            (QuantityOfInterest("y", "linear"),),
        )
        result = dgsm(normal, "y", 2048, seed=2)
        variance = sum((w * s) ** 2 for w, s in zip(weights, sigmas, strict=True))
        for i, (w, s) in enumerate(zip(weights, sigmas, strict=True)):
            self.assertAlmostEqual(result.nu[i], w * w, delta=1e-12)
            exact_total = (w * s) ** 2 / variance
            self.assertAlmostEqual(
                result.total_index_upper_bounds[i], exact_total, delta=0.03
            )
        self.assertEqual(result.negligible(0.01), ("x2",))

    def test_constants_for_unsupported_laws(self) -> None:
        """Only uniform and normal laws get a Poincare constant."""
        self.assertAlmostEqual(poincare_constant(Uniform(1.0, 3.0)), 4.0 / math.pi**2)
        self.assertAlmostEqual(poincare_constant(Normal(5.0, 2.0)), 4.0)
        self.assertIsNone(poincare_constant(LogNormal(0.0, 1.0)))
        model = ProbabilisticModel(
            JointInput.independent(RandomInput("x", LogNormal(0.0, 0.5))),
            lambda p: {"y": p["x"] ** 2},
            (QuantityOfInterest("y", "square"),),
        )
        result = dgsm(model, "y", 64, design="mc", seed=1)
        self.assertEqual(result.total_index_upper_bounds, (None,))
        self.assertEqual(result.negligible(), ())
        self.assertIsNotNone(result.nu_standard_errors)


class GivenDataTests(unittest.TestCase):
    """First-order indices from an existing sample."""

    def test_ishigami_from_monte_carlo_sample(self) -> None:
        """Corrected estimates match the analytic first-order indices."""
        generator = random.Random(8)
        rows = [
            tuple(generator.uniform(-math.pi, math.pi) for _ in range(3))
            for _ in range(20000)
        ]
        y = [
            math.sin(a) + 7.0 * math.sin(b) ** 2 + 0.1 * c**4 * math.sin(a)
            for a, b, c in rows
        ]
        result = given_data_first_order(rows, y, names=("x1", "x2", "x3"))
        self.assertEqual(result.partitions, 50)
        for got, want in zip(result.first_order, ISHIGAMI_FIRST, strict=True):
            self.assertAlmostEqual(got, want, delta=0.02)
        # The raw ratio carries the (M - 1)(1 - S)/(n - 1) upward bias.
        bias = 49 / 19999
        self.assertAlmostEqual(result.first_order_raw[2], bias, delta=0.004)
        self.assertAlmostEqual(result.output_variance, ISHIGAMI_VARIANCE, delta=0.4)

    def test_dependent_inputs(self) -> None:
        """Y = X1 with corr(X1, X2) = rho gives S_2 = rho^2."""
        generator = random.Random(3)
        rho = 0.6
        rows = []
        for _ in range(10000):
            a = generator.gauss(0.0, 1.0)
            b = rho * a + math.sqrt(1 - rho * rho) * generator.gauss(0.0, 1.0)
            rows.append((a, b))
        result = given_data_first_order(rows, [r[0] for r in rows])
        self.assertAlmostEqual(result.first_order[0], 1.0, delta=0.01)
        self.assertAlmostEqual(result.first_order[1], rho * rho, delta=0.03)
        self.assertEqual(set(result.as_mapping()), {"x0", "x1"})

    def test_validation(self) -> None:
        """Ragged data, constant outputs and bad partitions are rejected."""
        with self.assertRaises(ValueError):
            given_data_first_order([(1.0,), (2.0,)], [1.0, 2.0])
        with self.assertRaises(ValueError):
            given_data_first_order([(float(k),) for k in range(8)], [1.0] * 8)
        with self.assertRaises(ValueError):
            given_data_first_order(
                [(float(k),) for k in range(8)], list(range(8)), partitions=5
            )


class FormAutodiffTests(unittest.TestCase):
    """FORM with the exact limit-state gradient through the Nataf map."""

    @staticmethod
    def _model(use_admath: bool, correlation=None) -> ProbabilisticModel:
        """Return a nonlinear lognormal/normal limit state."""
        exp = admath.exp if use_admath else math.exp
        items = (
            RandomInput("r", LogNormal(1.6, 0.15)),
            RandomInput("s", Normal(2.0, 0.6)),
        )
        inputs = (
            JointInput.independent(*items)
            if correlation is None
            else JointInput(items, correlation)
        )
        return ProbabilisticModel(
            inputs,
            lambda p: {"g": p["r"] - p["s"] * exp(0.2 * p["s"])},
            (QuantityOfInterest("g", "margin"),),
        )

    def test_autodiff_matches_finite_differences_with_fewer_calls(self) -> None:
        """Same design point; one model call per gradient instead of 2d."""
        for correlation in (None, ((1.0, 0.4), (0.4, 1.0))):
            with self.subTest(correlation=correlation):
                state = ModelLimitState(
                    self._model(True, correlation), "g", 0.0, "below"
                )
                auto = form_hlrf(state, gradient="auto")
                fd = form_hlrf(
                    ModelLimitState(self._model(True, correlation), "g", 0.0, "below")
                )
                self.assertEqual(auto.gradient_method, "autodiff")
                self.assertEqual(fd.gradient_method, "finite-difference")
                self.assertTrue(auto.converged and fd.converged)
                self.assertAlmostEqual(auto.beta, fd.beta, delta=1e-6)
                for a, b in zip(auto.gradient_u, fd.gradient_u, strict=True):
                    self.assertAlmostEqual(a, b, delta=1e-5)
                self.assertLess(auto.evaluations, fd.evaluations)
                self.assertGreater(auto.gradient_evaluations, 0)

    def test_fallback_and_strict_modes(self) -> None:
        """math-based models fall back under auto and raise under autodiff."""
        state = ModelLimitState(self._model(False), "g", 0.0, "below")
        auto = form_hlrf(state, gradient="auto")
        self.assertEqual(auto.gradient_method, "finite-difference")
        self.assertIn("AutodiffUnsupportedError", auto.gradient_fallback_reason)
        with self.assertRaises(AutodiffUnsupportedError):
            form_hlrf(state, gradient="autodiff")
        with self.assertRaises(ValueError):
            form_hlrf(state, gradient="magic")


if __name__ == "__main__":
    unittest.main()
