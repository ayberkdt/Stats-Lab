"""Tests of the tempered SMC adapter, the pCN study and the quadrature reference.

Experimental (added in 0.28).

References:
    [EQ-INV-SMP-001], [EQ-INV-SMP-002], [EQ-INV-SMP-003], [EQ-INV-SMP-004],
    [EQ-INV-SMP-005], [EQ-INV-SMP-006].

"""

from __future__ import annotations

import math
import os
import unittest
from statistics import NormalDist

from probstat.inverse import CallbackForward, linear_gaussian_posterior
from probstat.inverse.posterior_sampling import (
    TemperedSMCPosterior,
    banana_target,
    chain_region_fractions,
    compare_to_reference,
    pcn_dimension_study,
    quadrature_reference,
    reference_for_target,
    repeat_tempered_smc,
    square_bimodal_target,
    tempered_smc_from_target,
    tempered_smc_posterior,
)
from probstat.inverse.posterior_targets import (
    WhitenedGaussianPriorTarget,
    sample_hmc,
    sample_pcn,
)
from probstat.linalg import DenseOperator, DiagonalCovariance, Vector

_A = ((1.0, 0.5, -0.3), (0.2, -1.0, 0.4), (0.7, 0.1, 0.9), (-0.4, 0.6, 0.3))
_P = ((1.0, 0.0, 0.0), (0.4, 0.8, 0.0), (-0.2, 0.3, 0.6))
_DATA = (1.1, 0.7, 1.9, 0.8)
_NOISE = (0.04, 0.09, 0.05, 0.02)
_RIGHT = ("right mode (z > 0)", lambda z: z[0] > 0.0)
_LEFT = ("left mode (z < 0)", lambda z: z[0] < 0.0)


def _linear_target() -> WhitenedGaussianPriorTarget:
    """Return the linear-Gaussian target ``y = A x + eps`` with prior ``P P'``."""
    rows = _A
    model = CallbackForward(
        3,
        4,
        lambda x: tuple(
            math.fsum(a * b for a, b in zip(r, x, strict=True)) for r in rows
        ),
        jacobian=lambda x: rows,
    )
    return WhitenedGaussianPriorTarget(
        model,
        _DATA,
        noise=DiagonalCovariance(list(_NOISE)),
        prior_factor=DenseOperator(_P),
    )


def _prior_factor_covariance() -> tuple[tuple[float, ...], ...]:
    """Return ``P P'`` as a dense matrix."""
    return tuple(
        tuple(math.fsum(_P[i][k] * _P[j][k] for k in range(3)) for j in range(3))
        for i in range(3)
    )


class SMCAdapterTests(unittest.TestCase):
    """Tempered SMC against the exact linear-Gaussian posterior."""

    def test_linear_gaussian_evidence_mean_and_variance(self) -> None:
        """Evidence, mean and variance match the closed form within MCSE bands."""
        from probstat.linalg import DenseCovariance

        target = _linear_target()
        exact = linear_gaussian_posterior(
            _A,
            _DATA,
            noise=DiagonalCovariance(list(_NOISE)),
            prior_covariance=DenseCovariance(_prior_factor_covariance()),
        )
        summary = repeat_tempered_smc(
            lambda stream: tempered_smc_from_target(
                target, particles=150, mutation="pcn", seed=stream
            ),
            runs=4,
            seed=11,
        )
        self.assertEqual(summary.failed, 0)
        # Band: 4 standard errors of the run-to-run mean, floored at 0.05 for
        # the O(1/N) bias of the evidence estimator at N = 200.
        band = max(4.0 * summary.log_evidence_mean_se, 0.05)
        self.assertAlmostEqual(
            summary.log_evidence_mean, exact.log_evidence, delta=band
        )
        variances = exact.marginal_variances()
        for i in range(3):
            self.assertAlmostEqual(
                summary.mean_of_means[i],
                exact.mean[i],
                delta=4.0 * summary.mean_se[i] + 1e-3,
            )
            # Within-run variance has relative Monte Carlo error ~ sqrt(2/N_eff)
            # with N_eff <= 300 (about 12 percent); allow 25 percent.
            self.assertAlmostEqual(
                summary.variance_of_means[i] / variances[i], 1.0, delta=0.25
            )

    def test_rwm_mutation_gives_the_same_evidence(self) -> None:
        """The random-walk mutation estimates the same evidence."""
        from probstat.linalg import DenseCovariance

        target = _linear_target()
        exact = linear_gaussian_posterior(
            _A,
            _DATA,
            noise=DiagonalCovariance(list(_NOISE)),
            prior_covariance=DenseCovariance(_prior_factor_covariance()),
        )
        summary = repeat_tempered_smc(
            lambda stream: tempered_smc_from_target(
                target, particles=120, mutation="rwm", seed=stream
            ),
            runs=4,
            seed=12,
        )
        band = max(4.0 * summary.log_evidence_mean_se, 0.05)
        self.assertAlmostEqual(
            summary.log_evidence_mean, exact.log_evidence, delta=band
        )

    def test_schedule_and_bookkeeping(self) -> None:
        """Temperatures, ESS history, resample count and the coordinate map."""
        target = _linear_target()
        result = tempered_smc_from_target(target, particles=200, seed=3)
        self.assertTrue(result.converged)
        self.assertEqual(result.temperatures[0], 0.0)
        self.assertEqual(result.temperatures[-1], 1.0)
        self.assertTrue(
            all(
                a < b
                for a, b in zip(
                    result.temperatures, result.temperatures[1:], strict=False
                )
            )
        )
        stages = len(result.temperatures) - 1
        self.assertEqual(len(result.ess_history), stages)
        self.assertEqual(len(result.acceptance_rates), stages)
        self.assertEqual(result.resample_count, stages)
        # The adaptive rule keeps the ESS at or above half the particles,
        # except that the last stage (beta capped at 1) may exceed it.
        self.assertTrue(all(e >= 0.5 * 200 - 1e-6 for e in result.ess_history))
        self.assertAlmostEqual(
            math.fsum(result.log_evidence_increments), result.log_evidence, places=12
        )
        for z, x in zip(result.latent_samples[:5], result.samples[:5], strict=True):
            self.assertEqual(x, target.to_physical(z))
        self.assertEqual(result.names, ("x0", "x1", "x2"))
        self.assertGreater(result.evaluations, 200)

    def test_reproducible_for_a_fixed_seed(self) -> None:
        """The same seed gives identical output; another seed differs."""
        target = _linear_target()
        a = tempered_smc_from_target(target, particles=100, seed=4)
        b = tempered_smc_from_target(target, particles=100, seed=4)
        c = tempered_smc_from_target(target, particles=100, seed=5)
        self.assertEqual(a.samples, b.samples)
        self.assertEqual(a.log_evidence, b.log_evidence)
        self.assertNotEqual(a.log_evidence, c.log_evidence)

    def test_zero_likelihood_everywhere_is_reported_not_raised(self) -> None:
        """All-zero likelihood returns converged=False with a reason."""
        result = tempered_smc_posterior(
            lambda g: (g.gauss(0.0, 1.0),),
            lambda z: -math.inf,
            particles=50,
            seed=1,
        )
        self.assertFalse(result.converged)
        self.assertIn("fewer than two particles", result.stop_reason)
        self.assertTrue(math.isnan(result.log_evidence))
        with self.assertRaises(ValueError):
            result.mean()

    def test_invalid_arguments_raise(self) -> None:
        """Bad types and values are rejected, not reported."""
        target = _linear_target()
        with self.assertRaises(TypeError):
            tempered_smc_from_target(object(), particles=10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            tempered_smc_from_target(target, particles=1)
        with self.assertRaises(ValueError):
            tempered_smc_from_target(target, particles=10, mutation="bad")
        with self.assertRaises(ValueError):
            tempered_smc_from_target(target, particles=10, ess_fraction=1.5)
        with self.assertRaises(TypeError):
            tempered_smc_posterior(1, lambda z: 0.0, particles=10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            repeat_tempered_smc(
                lambda s: tempered_smc_from_target(target, particles=10, seed=s), runs=1
            )
        with self.assertRaises(TypeError):
            repeat_tempered_smc(lambda s: 1, runs=2)  # type: ignore[arg-type, return-value]
        self.assertIsInstance(
            tempered_smc_from_target(target, particles=20, seed=0), TemperedSMCPosterior
        )


class DimensionStudyTests(unittest.TestCase):
    """pCN against random-walk Metropolis under refinement (measured)."""

    def test_fast_tier_table_is_measured_and_structured(self) -> None:
        """Small run: the table has one row per dimension with sane rates."""
        study = pcn_dimension_study(
            (8, 32, 128), beta=0.3, draws=100, warmup=40, seed=0
        )
        self.assertEqual([r.dimension for r in study.rows], [8, 32, 128])
        for row in study.rows:
            self.assertTrue(0.0 < row.pcn_acceptance <= 1.0)
            self.assertTrue(0.0 <= row.rwm_acceptance <= 1.0)
            self.assertEqual(row.pcn_evaluations, 1 * 2 + 2 * (100 + 40))
            self.assertEqual(row.rwm_evaluations, row.pcn_evaluations)
        table = study.as_table()
        self.assertEqual(len(table.splitlines()), 4)
        print("\n" + table)
        # Measured on this seed (see the module fragment): the random-walk
        # rate at n = 128 is below its rate at n = 8, and the pCN rate does
        # not fall by a larger factor than the random-walk rate.
        pcn = [r.pcn_acceptance for r in study.rows]
        rwm = [r.rwm_acceptance for r in study.rows]
        self.assertLess(rwm[-1], rwm[0])
        self.assertGreater(pcn[-1] / pcn[0], rwm[-1] / rwm[0])

    def test_invalid_study_arguments_raise(self) -> None:
        """Bad dimensions, beta, noise and decay are rejected."""
        for kwargs in (
            {"beta": 0.0},
            {"beta": 1.5},
            {"noise_sd": 0.0},
            {"decay": -1.0},
        ):
            with self.assertRaises(ValueError):
                pcn_dimension_study((4,), draws=5, warmup=0, **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            pcn_dimension_study((0,), draws=5, warmup=0)
        with self.assertRaises(ValueError):
            pcn_dimension_study((), draws=5, warmup=0)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_slow_tier_larger_dimensions(self) -> None:
        """Larger n and more draws: print the measured table."""
        study = pcn_dimension_study(
            (8, 32, 128, 512), beta=0.3, draws=1500, warmup=300, chains=2, seed=1
        )
        print("\n" + study.as_table())
        for row in study.rows:
            print(
                f"n={row.dimension}: first-coefficient mean exact "
                f"{row.exact_mode_mean:.3f}, pCN {row.pcn_mode_mean:.3f}, "
                f"RWM {row.rwm_mode_mean:.3f}"
            )
        self.assertLess(study.rows[-1].rwm_acceptance, study.rows[0].rwm_acceptance)


def _gaussian_log_density(x: Vector) -> float:
    """Return the log density of ``N((1, -2), diag(0.25, 4))``."""
    return -0.5 * ((x[0] - 1.0) ** 2 / 0.25 + (x[1] + 2.0) ** 2 / 4.0) - math.log(
        2.0 * math.pi * 0.5 * 2.0
    )


class QuadratureReferenceTests(unittest.TestCase):
    """Closed form first, then an independent brute-force integral."""

    def test_two_dimensional_gaussian_closed_form(self) -> None:
        """Normal density: unit mass, mean, covariance and quantiles."""
        ref = quadrature_reference(
            _gaussian_log_density, [(-5.0, 7.0), (-14.0, 10.0)], panels=(10, 12)
        )
        # Truncation at 10 sd in x1 and 7 sd in x2 plus a degree-15 rule on
        # panels of about 1.2 sd: errors near 1e-8 (measured below 1e-7).
        self.assertAlmostEqual(ref.log_normalizer, 0.0, delta=1e-7)
        self.assertAlmostEqual(ref.mean[0], 1.0, delta=1e-7)
        self.assertAlmostEqual(ref.mean[1], -2.0, delta=1e-7)
        self.assertAlmostEqual(ref.covariance[0][0], 0.25, delta=1e-7)
        self.assertAlmostEqual(ref.covariance[1][1], 4.0, delta=1e-6)
        self.assertAlmostEqual(ref.covariance[0][1], 0.0, delta=1e-8)
        for axis, (centre, sd) in enumerate(((1.0, 0.5), (-2.0, 2.0))):
            normal = NormalDist(centre, sd)
            for probability, value in ref.quantiles[axis]:
                self.assertAlmostEqual(value, normal.inv_cdf(probability), delta=1e-5)
        self.assertEqual(ref.warnings, ())
        self.assertLess(ref.truncation_mass_estimate, 1e-6)
        self.assertLess(ref.refinement_log_error, 1e-8)

    def test_bimodal_matches_a_fine_trapezoid_rule(self) -> None:
        """The 1D bimodal reference agrees with a 4001-point trapezoid rule."""
        target = square_bimodal_target()
        ref = reference_for_target(target, [(-4.0, 4.0)], panels=40)
        constant = target.log_likelihood_constant

        def density(z: float) -> float:
            """Return the integrand ``p(y | z) N(z; 0, 1)``."""
            return math.exp(target.log_density((z,)) + constant)

        count = 4001
        h = 8.0 / (count - 1)
        values = [density(-4.0 + i * h) for i in range(count)]
        z_value = h * (math.fsum(values) - 0.5 * (values[0] + values[-1]))
        # The reference reports its own doubled-panel change (3e-9 here), so
        # agreement to 1e-7 is the resolution of the comparison.
        self.assertLess(ref.refinement_log_error, 1e-7)
        self.assertAlmostEqual(ref.log_normalizer, math.log(z_value), delta=1e-7)
        # Symmetry: zero mean, half the mass on each side of the barrier.
        self.assertAlmostEqual(ref.mean[0], 0.0, delta=1e-12)
        self.assertAlmostEqual(ref.region_mass(_RIGHT[1]), 0.5, delta=1e-12)
        self.assertEqual(ref.warnings, ())
        self.assertAlmostEqual(ref.quantiles[0][1][1], 0.0, delta=0.05)

    def test_too_small_box_is_reported(self) -> None:
        """A box that cuts the density warns and estimates a large tail."""
        ref = quadrature_reference(
            lambda x: -0.5 * x[0] * x[0], [(-1.0, 1.0)], panels=8
        )
        self.assertGreater(ref.truncation_mass_estimate, 0.05)
        self.assertTrue(any("box" in w for w in ref.warnings))
        self.assertGreater(ref.boundary_density_ratio, 0.5)

    def test_under_resolved_grid_is_reported(self) -> None:
        """One panel of order 4 for a narrow peak fails the refinement check."""
        ref = quadrature_reference(
            lambda x: -0.5 * (x[0] / 0.05) ** 2, [(-3.0, 3.0)], panels=2, order=4
        )
        self.assertTrue(any("doubling" in w or "box" in w for w in ref.warnings))

    def test_invalid_inputs_raise(self) -> None:
        """Bad bounds, panels, order, quantiles and density values raise."""

        def f(x: Vector) -> float:
            """Return a standard normal log density."""
            return -0.5 * x[0] * x[0]

        with self.assertRaises(TypeError):
            quadrature_reference(1, [(0.0, 1.0)])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            quadrature_reference(f, [(1.0, 0.0)])
        with self.assertRaises(ValueError):
            quadrature_reference(f, [(0.0, 1.0)] * 3)
        with self.assertRaises(ValueError):
            quadrature_reference(f, [(0.0, 1.0)], panels=0)
        with self.assertRaises(ValueError):
            quadrature_reference(f, [(0.0, 1.0)], order=1)
        with self.assertRaises(ValueError):
            quadrature_reference(f, [(0.0, 1.0)], quantiles=(1.5,))
        with self.assertRaises(ValueError):
            quadrature_reference(lambda x: math.nan, [(0.0, 1.0)])
        with self.assertRaises(ValueError):
            quadrature_reference(lambda x: -math.inf, [(0.0, 1.0)])
        with self.assertRaises(ValueError):
            quadrature_reference(
                lambda x: 0.0, [(0.0, 1.0), (0.0, 1.0)], panels=(1000, 1000), order=8
            )


class SamplerValidationTests(unittest.TestCase):
    """The adapters against the reference, including honest failures."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the bimodal and banana targets and their references once."""
        cls.bimodal = square_bimodal_target()
        cls.bimodal_ref = reference_for_target(cls.bimodal, [(-4.0, 4.0)], panels=40)
        cls.banana = banana_target()
        cls.banana_ref = reference_for_target(
            cls.banana,
            [(-3.0, 3.0), (-8.0, 3.0)],
            panels=(6, 8),
            order=12,
            refine_check=True,
        )

    def test_banana_reference_is_resolved(self) -> None:
        """The 2D banana reference has no truncation or refinement warning."""
        self.assertEqual(self.banana_ref.warnings, ())
        self.assertLess(self.banana_ref.refinement_log_error, 1e-6)
        self.assertLess(self.banana_ref.refinement_mean_error, 1e-4)

    def test_smc_finds_both_modes_and_the_evidence(self) -> None:
        """SMC occupies both modes (not guaranteed in general) and gets log Z."""
        summary = repeat_tempered_smc(
            lambda s: tempered_smc_from_target(
                self.bimodal, particles=300, mutation="pcn", seed=s
            ),
            runs=3,
            seed=21,
        )
        band = max(4.0 * summary.log_evidence_mean_se, 0.05)
        self.assertAlmostEqual(
            summary.log_evidence_mean, self.bimodal_ref.log_normalizer, delta=band
        )
        run = tempered_smc_from_target(self.bimodal, particles=400, seed=22)
        # The median of a symmetric bimodal law sits in the gap, so its sample
        # value moves by about one sd when the mode weights change by a few
        # percent; the quantile tolerance is therefore wide here.
        comparison = compare_to_reference(
            run.latent_samples,
            self.bimodal_ref,
            regions=[_RIGHT, _LEFT],
            mean_tolerance=0.3,
            quantile_tolerance=1.5,
        )
        self.assertTrue(comparison.passed, comparison.failures)

    def test_single_start_pcn_misses_a_mode_and_says_so(self) -> None:
        """Chains started in one mode never cross the 1e-14 barrier."""
        result = sample_pcn(
            self.bimodal,
            initial=(1.0,),
            draws=400,
            warmup=100,
            chains=4,
            seed=31,
            check_convergence=False,
        )
        pooled = result.pooled()
        comparison = compare_to_reference(
            pooled, self.bimodal_ref, regions=[_RIGHT, _LEFT]
        )
        self.assertFalse(comparison.passed)
        left = next(r for r in comparison.regions if r.name == _LEFT[0])
        self.assertEqual(left.status, "missed")
        self.assertEqual(left.sample_fraction, 0.0)
        self.assertEqual(chain_region_fractions(result, _RIGHT[1]), (1.0,) * 4)

    def test_nuts_from_one_mode_also_reports_the_miss(self) -> None:
        """NUTS started in one mode does not find the other; the check fails."""
        result = sample_hmc(
            self.bimodal,
            initial=(1.0,),
            draws=150,
            warmup=100,
            chains=2,
            seed=32,
            check_convergence=False,
        )
        comparison = compare_to_reference(
            result.mcmc.pooled(), self.bimodal_ref, regions=[_RIGHT, _LEFT]
        )
        self.assertFalse(comparison.passed)
        self.assertEqual(comparison.regions[1].status, "missed")

    def test_prior_starts_give_chain_level_evidence_of_stuck_chains(self) -> None:
        """With prior starts every chain sits in one mode (fractions 0 or 1)."""
        result = sample_pcn(
            self.bimodal,
            draws=300,
            warmup=100,
            chains=4,
            seed=33,
            check_convergence=False,
        )
        fractions = chain_region_fractions(result, _RIGHT[1])
        self.assertTrue(all(f in (0.0, 1.0) for f in fractions), fractions)

    def test_banana_moments_by_smc_pcn_and_nuts(self) -> None:
        """SMC, pCN and NUTS agree with the quadrature moments and quantiles."""
        run = tempered_smc_from_target(self.banana, particles=500, seed=43)
        self.assertTrue(run.converged)
        # Across seeds 41..46 the 2.5/97.5 percent quantile error of N = 500
        # SMC ranged over 0.06-0.75 sd (measured); 0.5 holds for this seed.
        smc = compare_to_reference(
            run.latent_samples, self.banana_ref, quantile_tolerance=0.5
        )
        self.assertTrue(smc.passed, smc.failures)
        pcn = sample_pcn(
            self.banana,
            draws=1500,
            warmup=500,
            chains=2,
            step=0.3,
            seed=42,
            check_convergence=False,
        )
        pcn_comparison = compare_to_reference(
            pcn.pooled(),
            self.banana_ref,
            mean_tolerance=0.4,
            std_tolerance=0.4,
            quantile_tolerance=0.5,
        )
        self.assertTrue(pcn_comparison.passed, pcn_comparison.failures)
        nuts = sample_hmc(
            self.banana,
            draws=250,
            warmup=150,
            chains=2,
            seed=43,
            check_convergence=False,
        )
        # 500 autocorrelated draws: the tail quantile has an error of a few
        # tenths of an sd (0.54 measured for this seed).
        nuts_comparison = compare_to_reference(
            nuts.mcmc.pooled(), self.banana_ref, quantile_tolerance=0.7
        )
        self.assertTrue(nuts_comparison.passed, nuts_comparison.failures)

    def test_comparison_flags_wrong_moments(self) -> None:
        """Shifted samples fail the mean check and are listed as failures."""
        shifted = [(z[0] + 1.0, z[1]) for z in self.banana_ref.grid_points[:200]]
        comparison = compare_to_reference(shifted, self.banana_ref)
        self.assertFalse(comparison.passed)
        self.assertTrue(any("mean of coordinate 0" in f for f in comparison.failures))
        with self.assertRaises(ValueError):
            compare_to_reference([(0.0, 0.0)], self.banana_ref)
        with self.assertRaises(ValueError):
            compare_to_reference([(0.0,), (1.0,)], self.banana_ref)


if __name__ == "__main__":
    unittest.main()
