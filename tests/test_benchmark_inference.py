"""Tests for the synthetic expansion-history inference comparison (roadmap 20B).

Everything here is SYNTHETIC: no survey data and no claim about the real
universe or dark energy. Order: closed forms (constant-w factor, Einstein-de
Sitter distance, low-redshift limit, nesting of the parameterizations), the
independent trapezoid oracle for the distance integral, analytic derivatives
against central differences, the offset marginalization against dense
algebra, the Laplace stationarity, then the Laplace / pCN / SMC comparison
against the quadrature reference with Monte Carlo bands, evidence and Bayes
factors, determinism, and the failure cases (weak identification, too few
particles, no reference in three dimensions, invalid input). The slow tier
(``PROBSTAT_SLOW=1``) runs informative data with larger budgets and prints the
measured numbers.
"""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.inverse.benchmark_inference import (
    PREDECLARED_SEEDS,
    ExpansionData,
    ModelSpec,
    build_problem,
    compare_inference_methods,
    compare_models,
    evidence_category,
    laplace_fit,
    model_spec,
    offset_marginal_covariance,
    offset_marginal_log_likelihood,
    prior_width_sensitivity,
    synthetic_expansion_data,
)
from probstat.inverse.linear_gaussian import linear_gaussian_posterior
from probstat.linalg import DenseCovariance, ScaledIdentityCovariance

C_KM_S = 299792.458


def _data(
    redshifts: tuple[float, ...], truth: str = "lcdm", **kw: float
) -> ExpansionData:
    """Return synthetic data on given redshifts."""
    return synthetic_expansion_data(truth, redshifts=redshifts, **kw)


class ClosedFormTests(unittest.TestCase):
    """Exact checks of the forward model."""

    def test_einstein_de_sitter_distance(self) -> None:
        """Om = 1, w irrelevant: D_L = (1 + z) (2 c / H0) (1 - (1 + z)^-1/2)."""
        zs = (0.1, 0.5, 1.3)
        problem = build_problem(model_spec("lcdm", omega_m=1.0), _data(zs))
        for z, mu in zip(zs, problem.moduli(()), strict=True):
            d_l = (1.0 + z) * 2.0 * C_KM_S / 70.0 * (1.0 - (1.0 + z) ** -0.5)
            # 6-point Gauss-Legendre on a segment of width 0.8: error ~1e-11
            self.assertAlmostEqual(mu, 5.0 * math.log10(d_l) + 25.0, delta=1e-9)

    def test_low_redshift_limit(self) -> None:
        """D_L ~ c z / H0 (1 + (1 - q0) z / 2) at small z, q0 = 3 Om / 2 - 1."""
        z = 1e-3
        problem = build_problem(model_spec("lcdm"), _data((z, 0.5)))
        q0 = 1.5 * 0.3 - 1.0
        d_l = C_KM_S * z / 70.0 * (1.0 + 0.5 * (1.0 - q0) * z)
        self.assertAlmostEqual(
            problem.moduli(())[0], 5.0 * math.log10(d_l) + 25.0, delta=2e-6
        )

    def test_families_are_nested(self) -> None:
        """Setting w = -1, wa = 0 or w1 = w2 reproduces LCDM / wcdm."""
        data = _data((0.2, 0.6, 1.0, 1.4))
        base = build_problem(model_spec("lcdm"), data).moduli(())
        wcdm = build_problem(model_spec("wcdm"), data)
        cpl = build_problem(model_spec("cpl"), data)
        pw2 = build_problem(model_spec("pw2", z_break=0.8), data)
        for a, b in zip(base, wcdm.moduli((-1.0,)), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-13)
        constant = wcdm.moduli((-0.7,))
        for a, b in zip(constant, cpl.moduli((-0.7, 0.0)), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-13)
        for a, b in zip(constant, pw2.moduli((-0.7, -0.7)), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-13)

    def test_constant_w_factor_against_closed_form(self) -> None:
        """For constant w, mu matches a direct evaluation with X = (1+z)^(3(1+w))."""
        w, om, z = -0.8, 0.3, 0.9
        data = _data((0.3, z))
        problem = build_problem(model_spec("wcdm"), data)
        mu = problem.moduli((w,))[1]
        n = 20000
        h = z / n
        total = 0.0
        for i in range(n):
            t = (i + 0.5) * h
            e2 = om * (1 + t) ** 3 + (1 - om) * (1 + t) ** (3 * (1 + w))
            total += h / math.sqrt(e2)
        expected = 5.0 * math.log10((1 + z) * C_KM_S / 70.0 * total) + 25.0
        self.assertAlmostEqual(mu, expected, delta=1e-8)

    def test_gauss_legendre_converged_and_trapezoid_oracle(self) -> None:
        """The data generator (fine trapezoid) agrees with the 6-point rule."""
        data = synthetic_expansion_data("cpl", count=10, sigma=0.05)
        problem = build_problem(model_spec("cpl"), data)
        wide = build_problem(model_spec("cpl"), data, quadrature_order=12)
        theta = (-0.9, -0.4)
        mu = problem.moduli(theta)
        for m, m12, exact in zip(mu, wide.moduli(theta), data.exact_data, strict=True):
            self.assertAlmostEqual(m, m12, delta=1e-11)
            # exact_data contains the offset; trapezoid error ~ h^2 f'' / 12
            self.assertAlmostEqual(m + data.true_offset, exact, delta=1e-7)

    def test_analytic_jacobian_against_central_differences(self) -> None:
        """Analytic Jacobian = central differences for every family."""
        data = synthetic_expansion_data("cpl", count=8)
        cases = (
            (model_spec("wcdm"), (-0.8,)),
            (model_spec("cpl"), (-0.9, 0.3)),
            (model_spec("pw2", z_break=0.5), (-0.8, -1.1)),
            (model_spec("cpl", free_omega=True), (0.33, -0.9, 0.3)),
            (model_spec("lcdm", free_omega=True), (0.27,)),
        )
        for spec, theta in cases:
            problem = build_problem(spec, data)
            rows = problem.jacobian(theta)
            for j in range(len(theta)):
                step = 1e-5
                up = list(theta)
                down = list(theta)
                up[j] += step
                down[j] -= step
                fd = [
                    (a - b) / (2 * step)
                    for a, b in zip(
                        problem.moduli(up), problem.moduli(down), strict=True
                    )
                ]
                for i, row in enumerate(rows):
                    self.assertAlmostEqual(
                        row[j],
                        fd[i],
                        delta=1e-7 * (1.0 + abs(fd[i])),
                        msg=(spec.name, j),
                    )


class OffsetTests(unittest.TestCase):
    """The marginalized offset against dense algebra and the linear oracle."""

    def test_closed_form_matches_dense_cholesky(self) -> None:
        """Sherman-Morrison log density equals the dense Cholesky evaluation."""
        rng = random.Random(4)
        m, s, so = 7, 0.07, 0.4
        r = [rng.gauss(0.0, 0.3) for _ in range(m)]
        cov = DenseCovariance(offset_marginal_covariance(m, s, so))
        dense = -0.5 * (
            cov.quadratic_form(r) + cov.log_det() + m * math.log(2.0 * math.pi)
        )
        self.assertAlmostEqual(
            offset_marginal_log_likelihood(r, s, so), dense, delta=1e-10
        )

    def test_equals_linear_gaussian_evidence(self) -> None:
        """Integrating the offset equals the evidence of the linear model."""
        rng = random.Random(9)
        m, s, so = 6, 0.1, 0.5
        y = [rng.gauss(0.2, 0.3) for _ in range(m)]
        oracle = linear_gaussian_posterior(
            [[1.0] for _ in range(m)],
            y,
            noise=ScaledIdentityCovariance(m, s * s),
            prior_covariance=ScaledIdentityCovariance(1, so * so),
            method="data_space",
        )
        self.assertAlmostEqual(
            offset_marginal_log_likelihood(y, s, so), oracle.log_evidence, delta=1e-10
        )

    def test_lcdm_exact_evidence(self) -> None:
        """A model without parameters carries log N(y; mu, C) exactly."""
        data = synthetic_expansion_data("lcdm", count=9)
        problem = build_problem(model_spec("lcdm"), data)
        r = [a - b for a, b in zip(problem.moduli(()), data.data, strict=True)]
        self.assertEqual(
            problem.fixed_log_evidence,
            offset_marginal_log_likelihood(r, data.sigma, 0.5),
        )
        self.assertIsNone(problem.target)


class MethodTests(unittest.TestCase):
    """Laplace, pCN and SMC against the quadrature reference."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the 1-D (wCDM) and 2-D (CPL) comparisons once (small budgets)."""
        data = synthetic_expansion_data("cpl", count=10)
        cls.data = data
        cls.wcdm = build_problem(model_spec("wcdm"), data)
        cls.cpl = build_problem(model_spec("cpl"), data)
        cls.c1 = compare_inference_methods(
            cls.wcdm,
            particles=150,
            runs=3,
            pcn_draws=200,
            pcn_warmup=150,
            panels=10,
        )
        cls.c2 = compare_inference_methods(
            cls.cpl,
            particles=150,
            runs=3,
            pcn_draws=200,
            pcn_warmup=150,
            panels=8,
            refine_check=False,
        )

    def test_laplace_is_stationary(self) -> None:
        """The true gradient vanishes at the Laplace MAP (BFGS-free check)."""
        for problem in (self.wcdm, self.cpl):
            fit = laplace_fit(problem)
            self.assertTrue(fit.converged)
            assert problem.target is not None
            gradient = problem.target.gradient(fit.z_map)
            self.assertLess(math.hypot(*gradient), 1e-6)

    def test_reference_quality(self) -> None:
        """The 1-D reference is refined; the 2-D one reports its settings."""
        ref = self.c1.reference
        assert ref is not None
        self.assertLess(ref.log_evidence_se, 1e-6)
        self.assertIn("panels=10", self.c1.reference_note)
        self.assertIsNotNone(self.c2.reference)

    def test_all_methods_within_bands_1d(self) -> None:
        """1-D: mean errors below 0.5 sd; SMC evidence within 4 SE of quadrature."""
        for name in ("laplace", "pcn", "smc"):
            item = self.c1.method(name)
            self.assertTrue(item.converged, name)
            self.assertLess(item.mean_error_sd[0], 0.5, name)
            self.assertTrue(0.7 < item.sd_ratio[0] < 1.3, (name, item.sd_ratio))
        smc = self.c1.method("smc")
        ref = self.c1.reference
        assert ref is not None
        self.assertLess(
            abs(smc.log_evidence - ref.log_evidence),
            4.0 * smc.log_evidence_se + 0.05,
        )
        self.assertGreater(self.c1.method("pcn").evaluations, 0)

    def test_all_methods_within_bands_2d(self) -> None:
        """2-D: errors below 0.5 reference sd in every coordinate."""
        for name in ("laplace", "pcn", "smc"):
            item = self.c2.method(name)
            self.assertTrue(item.converged, name)
            for k in range(2):
                self.assertLess(item.mean_error_sd[k], 0.5, (name, k))
                self.assertTrue(0.7 < item.sd_ratio[k] < 1.35, (name, k))
        ref = self.c2.reference
        assert ref is not None
        self.assertLess(
            abs(self.c2.method("smc").log_evidence - ref.log_evidence),
            4.0 * self.c2.method("smc").log_evidence_se + 0.05,
        )

    def test_laplace_evidence_close_to_quadrature(self) -> None:
        """Mildly nonlinear posterior: Laplace log Z within 0.3 of quadrature."""
        for comparison in (self.c1, self.c2):
            ref = comparison.reference
            assert ref is not None
            lap = comparison.method("laplace")
            self.assertLess(abs(lap.log_evidence - ref.log_evidence), 0.3)

    def test_costs_are_recorded(self) -> None:
        """Evaluation counts are exact and ordered: Laplace << pCN < SMC."""
        for comparison in (self.c1, self.c2):
            counts = [
                comparison.method(n).evaluations for n in ("laplace", "pcn", "smc")
            ]
            self.assertLess(counts[0], 100)
            self.assertLess(counts[0], counts[1])
            self.assertLess(counts[1], counts[2])
            self.assertIn("SYNTHETIC", comparison.notice)

    def test_deterministic_given_seeds(self) -> None:
        """Repeating the SMC runs with the predeclared seeds reproduces them."""
        again = compare_inference_methods(
            self.wcdm,
            particles=150,
            runs=3,
            pcn_draws=200,
            pcn_warmup=150,
            panels=10,
        )
        self.assertEqual(again.smc_log_evidences, self.c1.smc_log_evidences)
        self.assertEqual(again.method("pcn").mean, self.c1.method("pcn").mean)

    def test_no_reference_in_three_dimensions(self) -> None:
        """FAILURE CASE: with three free parameters there is no quadrature."""
        problem = build_problem(model_spec("cpl", free_omega=True), self.data)
        comparison = compare_inference_methods(
            problem,
            particles=80,
            runs=2,
            mutation_steps=2,
            pcn_draws=60,
            pcn_warmup=60,
            pcn_chains=2,
        )
        self.assertIsNone(comparison.reference)
        self.assertIn("no quadrature reference", comparison.reference_note)
        for name in ("laplace", "pcn", "smc"):
            self.assertTrue(math.isnan(comparison.method(name).mean_error_sd[0]))

    def test_too_few_particles_has_larger_evidence_error(self) -> None:
        """FAILURE CASE: 20 particles give a larger run-to-run evidence spread."""
        small = compare_inference_methods(
            self.wcdm,
            particles=20,
            runs=4,
            pcn_draws=40,
            pcn_warmup=40,
            pcn_chains=2,
            panels=8,
            refine_check=False,
        )
        self.assertGreater(
            small.smc_log_evidence_run_std, self.c1.smc_log_evidence_run_std
        )


class ModelComparisonTests(unittest.TestCase):
    """Bayes factors, uncertainty and prior sensitivity."""

    def test_lcdm_truth_does_not_favour_extra_freedom(self) -> None:
        """LCDM truth: wCDM is not strongly favoured; SMC and quadrature agree."""
        data = synthetic_expansion_data("lcdm", count=10, sigma=0.05)
        comparison = compare_models(
            data,
            (model_spec("lcdm"), model_spec("wcdm")),
            particles=120,
            runs=3,
            panels=10,
            refine_check=False,
            pcn_draws=60,
            pcn_warmup=60,
        )
        base, wcdm = comparison.rows
        self.assertEqual(base.category, "baseline")
        self.assertEqual(base.smc_log_evidence_se, 0.0)
        self.assertLess(wcdm.log_bayes_factor, 1.5)
        self.assertLess(
            abs(wcdm.log_bayes_factor - wcdm.log_bayes_factor_quadrature),
            4.0 * wcdm.log_bayes_factor_se + 0.05,
        )
        self.assertGreater(wcdm.log_bayes_factor_se, 0.0)
        self.assertIn("SYNTHETIC", comparison.notice)

    def test_prior_width_changes_the_bayes_factor(self) -> None:
        """Widening the w prior lowers ln B when the data add little."""
        data = synthetic_expansion_data("lcdm", count=10, sigma=0.05)
        rows = prior_width_sensitivity(data, "wcdm", (0.5, 1.0, 2.0), panels=10)
        values = [r.log_bayes_factor for r in rows]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0] - values[-1], 0.3)

    def test_evidence_category(self) -> None:
        """Kass-Raftery verbal scale on 2 ln B."""
        self.assertEqual(evidence_category(0.5), "not worth more than a bare mention")
        self.assertEqual(evidence_category(2.0), "positive for the model")
        self.assertEqual(evidence_category(-4.0), "strong for the baseline")
        self.assertEqual(evidence_category(6.0), "very strong for the model")

    def test_failure_weakly_identified_second_bin(self) -> None:
        """FAILURE CASE: a bin with one datum stays prior-dominated.

        With the break at z = 1.3 only the last of ten data constrains w2, so
        its Laplace standard deviation is close to the prior standard
        deviation 0.5: the data cannot resolve it.
        """
        data = synthetic_expansion_data("cpl", count=10, z_max=1.4)
        problem = build_problem(model_spec("pw2", z_break=1.3), data)
        fit = laplace_fit(problem)
        self.assertTrue(fit.converged)
        self.assertGreater(fit.sd[1] / 0.5, 0.85)
        self.assertLess(fit.sd[0] / 0.5, 0.7)


class ValidationTests(unittest.TestCase):
    """Invalid input is rejected, never repaired."""

    def test_invalid_inputs(self) -> None:
        """TypeError / ValueError for bad arguments."""
        with self.assertRaises(ValueError):
            model_spec("nope")
        with self.assertRaises(ValueError):
            model_spec("wcdm", prior_scale=0.0)
        with self.assertRaises(TypeError):
            model_spec("wcdm", free_omega=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            synthetic_expansion_data("nope")
        with self.assertRaises(ValueError):
            synthetic_expansion_data(redshifts=(0.5, 0.4))
        with self.assertRaises(ValueError):
            synthetic_expansion_data(sigma=-1.0)
        with self.assertRaises(ValueError):
            build_problem(
                model_spec("pw2", z_break=3.0), synthetic_expansion_data(count=5)
            )
        with self.assertRaises(TypeError):
            build_problem("lcdm", synthetic_expansion_data(count=5))  # type: ignore[arg-type]
        problem = build_problem(model_spec("wcdm"), synthetic_expansion_data(count=5))
        with self.assertRaises(ValueError):
            problem.moduli((1.0, 2.0))
        lcdm = build_problem(model_spec("lcdm"), synthetic_expansion_data(count=5))
        with self.assertRaises(ValueError):
            laplace_fit(lcdm)
        with self.assertRaises(ValueError):
            compare_inference_methods(lcdm)
        with self.assertRaises(ValueError):
            compare_models(synthetic_expansion_data(count=5), (model_spec("lcdm"),))
        with self.assertRaises(ValueError):
            offset_marginal_log_likelihood([], 0.1, 0.5)
        with self.assertRaises(ValueError):
            evidence_category(math.inf)
        self.assertEqual(set(PREDECLARED_SEEDS), {"data", "pcn", "smc", "smc_repeat"})

    def test_invalid_region_returns_nan_not_a_repair(self) -> None:
        """An invalid E^2 gives NaN moduli (zero likelihood), not a clipped value."""
        spec: ModelSpec = model_spec("lcdm", free_omega=True)
        problem = build_problem(spec, synthetic_expansion_data(count=5))
        values = problem.moduli((-0.5,))
        self.assertTrue(all(math.isnan(v) for v in values))
        assert problem.target is not None
        self.assertEqual(problem.target.log_likelihood((-16.0,)), -math.inf)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowInferenceStudy(unittest.TestCase):
    """Informative data (sigma = 0.015, 24 SNe), larger budgets, printed numbers."""

    def test_study(self) -> None:
        """Compare the four models for two truths and print the measured table."""
        for truth in ("smooth_step", "lcdm"):
            data = synthetic_expansion_data(truth, count=24, sigma=0.015)
            specs = (
                model_spec("lcdm"),
                model_spec("wcdm"),
                model_spec("cpl"),
                model_spec("pw2"),
            )
            comparison = compare_models(
                data,
                specs,
                particles=600,
                runs=5,
                mutation_steps=5,
                panels=14,
                pcn_draws=1500,
                pcn_warmup=500,
            )
            print(f"\nSLOW truth={truth}")
            for row in comparison.rows:
                print(
                    f"  {row.model:6s} d={row.dimension} smc={row.smc_log_evidence:.3f}"
                    f"+-{row.smc_log_evidence_se:.3f} "
                    f"quad={row.quadrature_log_evidence:.4f}"
                    f" lap={row.laplace_log_evidence:.3f} "
                    f"lnB={row.log_bayes_factor:.3f}"
                    f"+-{row.log_bayes_factor_se:.3f} "
                    f"lnBq={row.log_bayes_factor_quadrature:.3f}"
                    f" [{row.category}]"
                )
            for item in comparison.comparisons:
                for m in item.methods:
                    print(
                        f"    {item.model:5s} {m.method:7s} mean_err={m.mean_error_sd}"
                        f" sd_ratio={m.sd_ratio} q_err={m.quantile_error_sd}"
                        f" evals={m.evaluations} s={m.seconds:.1f}"
                    )
                self.assertEqual(item.failures, (), item.failures)
            sens = prior_width_sensitivity(data, "cpl", panels=14)
            print("  prior-width sensitivity (cpl vs lcdm):", sens)


if __name__ == "__main__":
    unittest.main()
