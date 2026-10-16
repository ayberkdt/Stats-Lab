"""MCMC diagnostics, convergence warnings, NUTS variants, autodiff, SBC.

Workstream W8 (ADR-0065).
"""

from __future__ import annotations

import dataclasses
import math
import random
import unittest
import warnings
from statistics import NormalDist

from probstat.distributions import Normal, RealSupport
from probstat.inference import (
    ConvergenceWarning,
    MarkovChain,
    MCMCResult,
    advi,
    bulk_effective_sample_size,
    convergence_warnings,
    effective_sample_size,
    hamiltonian_warnings,
    laplace_approximation,
    mcse_quantile,
    no_u_turn_sampler,
    ppl,
    quantile_effective_sample_size,
    random_walk_metropolis,
    rank_statistic,
    simulation_based_calibration,
    summarize_draws,
    tail_effective_sample_size,
)
from probstat.numerics import AutodiffUnsupportedError, admath

SCALES = tuple(0.5 + 0.25 * i for i in range(10))


def ar1_chains(phi: float, n: int, m: int, seed: int) -> list[list[float]]:
    """Return ``m`` stationary Gaussian AR(1) chains of length ``n``."""
    generator = random.Random(seed)
    chains = []
    for _ in range(m):
        x = [generator.gauss(0.0, 1.0)]
        for _ in range(n - 1):
            x.append(phi * x[-1] + generator.gauss(0.0, math.sqrt(1.0 - phi * phi)))
        chains.append(x)
    return chains


def gaussian_10d(x):
    """Independent Gaussian log-density written with operators only."""
    return -0.5 * sum((xi / s) ** 2 for xi, s in zip(x, SCALES, strict=True))


def gaussian_10d_math(x):
    """Return the same density through math.fsum (rejects dual numbers)."""
    return -0.5 * math.fsum((xi / s) ** 2 for xi, s in zip(x, SCALES, strict=True))


def quiet(function, *args, **kwargs):
    """Call ``function`` with convergence warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return function(*args, **kwargs)


class EffectiveSampleSizeOracleTests(unittest.TestCase):
    """Split, bulk and tail ESS and quantile MCSE against stan-dev/posterior.

    Oracle values: a numpy/scipy port of posterior's ``.ess``,
    ``split_chains``, ``z_scale``, ``ess_bulk``, ``ess_quantile``,
    ``ess_tail`` and ``mcse_quantile`` (scipy.stats.beta.ppf, numpy type-7
    quantiles), run in the verification venv on the same chains (scratch
    script oracle_ess.py of W8). The port reproduces the earlier unsplit
    oracle 825.6296706351059 exactly.
    """

    def test_antithetic_chains(self) -> None:
        """Four AR(-0.3) chains of length 100."""
        chains = ar1_chains(-0.3, 100, 4, 3)
        expected = {
            "unsplit": 825.6296706351059,
            "split": 1016.8687569601551,
            "bulk": 1020.8049392850226,
            "tail": 250.2870911978399,
            "q05": 341.01241252490206,
            "q50": 565.4318801857414,
        }
        got = {
            "unsplit": effective_sample_size(chains, split=False),
            "split": effective_sample_size(chains),
            "bulk": bulk_effective_sample_size(chains),
            "tail": tail_effective_sample_size(chains),
            "q05": quantile_effective_sample_size(chains, 0.05),
            "q50": quantile_effective_sample_size(chains, 0.5),
        }
        for key, value in expected.items():
            self.assertLess(abs(got[key] / value - 1.0), 1e-12, key)
        for p, value in (
            (0.05, 0.14414787565678433),
            (0.5, 0.069010612241837),
            (0.95, 0.16484024838051814),
        ):
            self.assertAlmostEqual(mcse_quantile(chains, p), value, delta=1e-12)

    def test_odd_length_chains_drop_the_middle_draw(self) -> None:
        """Three AR(0.6) chains of odd length 101."""
        chains = ar1_chains(0.6, 101, 3, 7)
        self.assertAlmostEqual(
            effective_sample_size(chains, split=False), 77.63097341912918, delta=1e-9
        )
        self.assertAlmostEqual(
            effective_sample_size(chains), 80.34777670521972, delta=1e-9
        )
        self.assertAlmostEqual(
            bulk_effective_sample_size(chains), 83.01442509405283, delta=1e-9
        )
        self.assertAlmostEqual(
            tail_effective_sample_size(chains), 145.63608705023148, delta=1e-9
        )
        self.assertAlmostEqual(
            mcse_quantile(chains, 0.5), 0.16374662852478236, delta=1e-12
        )
        self.assertAlmostEqual(
            mcse_quantile(chains, 0.95), 0.12676394421766413, delta=1e-12
        )


class EffectiveSampleSizeTheoryTests(unittest.TestCase):
    """Statistical properties against closed forms."""

    def test_iid_draws(self) -> None:
        """For i.i.d. normal draws bulk-ESS ~ S and the median MCSE ~ theory."""
        generator = random.Random(11)
        chains = [[generator.gauss(0.0, 1.0) for _ in range(2000)] for _ in range(4)]
        self.assertAlmostEqual(
            bulk_effective_sample_size(chains) / 8000, 1.0, delta=0.1
        )
        # Asymptotic sd of a sample quantile: sqrt(p(1-p)/S) / f(q_p). The
        # MCSE estimate itself is noisy (it is a spacing of order
        # statistics), so the ratio is averaged over eight data sets.
        size = 4 * 1000
        for p in (0.5, 0.9):
            theory = math.sqrt(p * (1 - p) / size) / NormalDist().pdf(
                NormalDist().inv_cdf(p)
            )
            ratios = []
            for _ in range(8):
                data = [
                    [generator.gauss(0.0, 1.0) for _ in range(1000)] for _ in range(4)
                ]
                ratios.append(mcse_quantile(data, p) / theory)
            self.assertAlmostEqual(
                sum(ratios) / len(ratios), 1.0, delta=0.12, msg=str(p)
            )

    def test_ar1_bulk_ess_and_split_detects_drift(self) -> None:
        """AR(1) bulk-ESS ~ n (1 - phi)/(1 + phi); a drifting chain is penalized."""
        chains = ar1_chains(0.5, 5000, 4, 0)
        theory = 4 * 5000 * 0.5 / 1.5
        self.assertAlmostEqual(
            bulk_effective_sample_size(chains) / theory, 1.0, delta=0.15
        )
        drifting = [
            [x + 3.0 * t / 2000 for t, x in enumerate(c[:2000])] for c in chains
        ]
        self.assertLess(
            effective_sample_size(drifting),
            0.5 * effective_sample_size(drifting, split=False),
        )

    def test_summary_carries_new_diagnostics(self) -> None:
        """summarize_draws reports bulk/tail ESS and quantile MCSE."""
        chains = ar1_chains(0.3, 400, 4, 5)
        summary = summarize_draws(chains, name="x")
        self.assertAlmostEqual(
            summary.bulk_effective_sample_size, bulk_effective_sample_size(chains)
        )
        self.assertAlmostEqual(
            summary.tail_effective_sample_size, tail_effective_sample_size(chains)
        )
        self.assertEqual([p for p, _ in summary.mcse_quantiles], [0.05, 0.5, 0.95])
        self.assertTrue(all(value > 0.0 for _, value in summary.mcse_quantiles))
        self.assertTrue(math.isnan(bulk_effective_sample_size([[1.0] * 8])))
        self.assertTrue(math.isnan(quantile_effective_sample_size([[1.0] * 8], 0.5)))
        self.assertTrue(math.isfinite(quantile_effective_sample_size(chains, 1.0)))
        with self.assertRaises(ValueError):
            quantile_effective_sample_size(chains, 1.5)


def _result(chains: list[list[float]]) -> MCMCResult:
    """Wrap scalar chains as an MCMCResult."""
    return MCMCResult(
        tuple(
            MarkovChain(
                tuple((v,) for v in chain),
                tuple(0.0 for _ in chain),
                0.5,
                math.nan,
                ((1.0,),),
                len(chain),
                0,
            )
            for chain in chains
        ),
        ("theta",),
        "test",
        0,
        1,
    )


class ConvergenceWarningTests(unittest.TestCase):
    """Automatic checks follow Vehtari et al. (2021)."""

    def test_good_chains_pass_and_bad_chains_fail(self) -> None:
        """i.i.d. chains pass; shifted chains, short chains, stuck chains fail."""
        generator = random.Random(2)
        good = [[generator.gauss(0, 1) for _ in range(500)] for _ in range(4)]
        self.assertEqual(convergence_warnings(_result(good)), ())
        shifted = [
            chain if k else [v + 1.0 for v in chain] for k, chain in enumerate(good)
        ]
        messages = convergence_warnings(_result(shifted))
        self.assertTrue(any("R-hat" in m for m in messages))
        sticky = ar1_chains(0.99, 500, 4, 1)
        messages = convergence_warnings(_result(sticky))
        self.assertTrue(any("bulk-ESS below 400" in m for m in messages))
        self.assertTrue(any("tail-ESS below 400" in m for m in messages))
        self.assertIn(
            "not computed", convergence_warnings(_result([[0.0, 1.0, 2.0]]))[0]
        )
        self.assertIn(
            "never changed", convergence_warnings(_result([[1.0] * 10] * 2))[0]
        )

    def test_metropolis_emits_and_records_warnings(self) -> None:
        """A badly tuned sampler warns; check_convergence=False is silent."""

        def bimodal(x):
            """Two well-separated modes."""
            return math.log(
                math.exp(-0.5 * (x[0] - 4) ** 2) + math.exp(-0.5 * (x[0] + 4) ** 2)
            )

        starts = [(-4.0,), (4.0,)]
        with self.assertWarns(ConvergenceWarning):
            result = random_walk_metropolis(
                bimodal,
                starts,
                proposal_covariance=((0.2,),),
                draws=300,
                chains=2,
                seed=1,
            )
        self.assertTrue(any("R-hat" in m for m in result.warnings))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            silent = random_walk_metropolis(
                bimodal,
                starts,
                proposal_covariance=((0.2,),),
                draws=300,
                chains=2,
                seed=1,
                check_convergence=False,
            )
        self.assertEqual(silent.warnings, ())

    def test_divergences_tree_depth_and_e_bfmi(self) -> None:
        """Neal's funnel diverges; depth 1 saturates; flat energies flag E-BFMI."""

        def funnel(z):
            """Neal's funnel: v ~ N(0, 3), x | v ~ N(0, exp(v / 2))."""
            v, x = z
            return -v * v / 18.0 - 0.5 * x * x * admath.exp(-v) - 0.5 * v

        with self.assertWarns(ConvergenceWarning):
            result = no_u_turn_sampler(
                funnel, (0.0, 0.0), draws=300, warmup=150, chains=2, seed=3
            )
        self.assertGreater(result.divergences, 0)
        self.assertTrue(any("diverged" in m for m in result.warnings))
        shallow = quiet(
            no_u_turn_sampler,
            gaussian_10d,
            [0.1] * 10,
            draws=50,
            warmup=50,
            chains=2,
            seed=2,
            max_tree_depth=1,
        )
        self.assertTrue(any("max_tree_depth=1" in m for m in shallow.warnings))
        record = shallow.statistics[0]
        flat = dataclasses.replace(record, energies=tuple(0.01 * k for k in range(50)))
        modified = dataclasses.replace(
            shallow, statistics=(flat, shallow.statistics[1])
        )
        self.assertLess(flat.e_bfmi, 0.3)
        self.assertTrue(any("E-BFMI" in m for m in hamiltonian_warnings(modified)))


class GradientMethodTests(unittest.TestCase):
    """Samplers use autodiff when possible and record the method."""

    def test_nuts_evaluation_savings_on_10d_gaussian(self) -> None:
        """Autodiff: one call per gradient; finite differences: 2d + 1."""
        common = {
            "draws": 100,
            "warmup": 100,
            "chains": 2,
            "seed": 7,
            "check_convergence": False,
        }
        auto = no_u_turn_sampler(gaussian_10d, [0.1] * 10, **common)
        self.assertEqual(auto.gradient_method, "autodiff")
        self.assertIsNone(auto.gradient_fallback_reason)
        for record in auto.statistics:
            self.assertEqual(
                record.log_density_evaluations, record.gradient_evaluations
            )
        forced = no_u_turn_sampler(
            gaussian_10d, [0.1] * 10, gradient="finite-difference", **common
        )
        for record in forced.statistics:
            self.assertEqual(
                record.log_density_evaluations, 21 * record.gradient_evaluations
            )
        fallback = no_u_turn_sampler(gaussian_10d_math, [0.1] * 10, **common)
        self.assertEqual(fallback.gradient_method, "finite-difference")
        self.assertIn("AutodiffUnsupportedError", fallback.gradient_fallback_reason)
        for record in fallback.statistics:
            # One failed dual probe per chain, then 2d + 1 calls per gradient.
            self.assertEqual(
                record.log_density_evaluations, 21 * record.gradient_evaluations + 1
            )
        with self.assertRaises(AutodiffUnsupportedError):
            no_u_turn_sampler(
                gaussian_10d_math, [0.1] * 10, gradient="autodiff", **common
            )
        with self.assertRaises(ValueError):
            no_u_turn_sampler(gaussian_10d, [0.1] * 10, gradient="magic", **common)
        with self.assertRaises(ValueError):
            no_u_turn_sampler(gaussian_10d, [0.1] * 10, variant="stan", **common)

    def test_both_nuts_variants_recover_moments(self) -> None:
        """Multinomial (default) and slice NUTS sample the 10-D Gaussian."""
        for variant in ("multinomial", "slice"):
            result = no_u_turn_sampler(
                gaussian_10d,
                [0.1] * 10,
                draws=500,
                warmup=500,
                chains=4,
                seed=13,
                variant=variant,
            )
            self.assertEqual(result.warnings, ())
            for index, summary in enumerate(result.summary()):
                with self.subTest(variant=variant, index=index):
                    self.assertLess(abs(summary.mean), 4.0 * summary.mcse_mean)
                    ratio = summary.standard_deviation / SCALES[index]
                    self.assertAlmostEqual(ratio, 1.0, delta=0.12)

    def test_one_transition_preserves_the_target_exactly(self) -> None:
        """Exact draws stay exactly distributed after one NUTS transition.

        Target: ``x = log(G)`` with ``G ~ Gamma(3, 1)``, CDF
        ``1 - exp(-e^x)(1 + e^x + e^(2x)/2)``. 2000 chains start at exact
        draws, take one transition with a fixed step size (no adaptation);
        Pearson's statistic on 20 equiprobable CDF bins must not reject.
        This checks invariance of the kernel itself, independent of mixing.
        """

        def log_target(x):
            """Log-density of the log of a Gamma(3, 1) variable."""
            return 3.0 * x[0] - admath.exp(x[0])

        def cdf(x):
            """Exact CDF of the target."""
            y = math.exp(x)
            return 1.0 - math.exp(-y) * (1.0 + y + 0.5 * y * y)

        generator = random.Random(5)
        count = 2000
        starts = [(math.log(generator.gammavariate(3.0, 1.0)),) for _ in range(count)]
        for variant in ("multinomial", "slice"):
            result = no_u_turn_sampler(
                log_target,
                starts,
                draws=1,
                warmup=0,
                chains=count,
                step_size=0.9,
                adapt_step_size=False,
                adapt_metric=False,
                metric="unit",
                seed=9,
                variant=variant,
                check_convergence=False,
                max_tree_depth=6,
            )
            bins = [0] * 20
            moved = 0
            for start, chain in zip(starts, result.mcmc.chains, strict=True):
                value = chain.samples[0][0]
                moved += value != start[0]
                bins[min(int(cdf(value) * 20), 19)] += 1
            statistic = sum((b - count / 20) ** 2 / (count / 20) for b in bins)
            with self.subTest(variant=variant):
                self.assertGreater(moved, count // 2)
                # chi-square(19) upper 0.1 % point is 43.82.
                self.assertLess(statistic, 43.82)

    def test_laplace_uses_exact_hessian(self) -> None:
        """A Gaussian target gives the exact covariance with autodiff."""
        precision = ((2.0, 0.6), (0.6, 1.0))

        def log_density(x):
            """Correlated Gaussian centred at (1, -2)."""
            d = (x[0] - 1.0, x[1] + 2.0)
            return -0.5 * sum(
                d[i] * precision[i][j] * d[j] for i in range(2) for j in range(2)
            )

        fit = laplace_approximation(log_density, (0.0, 0.0))
        self.assertEqual(
            (fit.gradient_method, fit.hessian_method), ("autodiff", "autodiff")
        )
        det = 2.0 - 0.36
        exact = ((1.0 / det, -0.6 / det), (-0.6 / det, 2.0 / det))
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(fit.covariance[i][j], exact[i][j], delta=1e-12)
        self.assertAlmostEqual(fit.mode[0], 1.0, delta=1e-7)

        def math_density(x):
            """Return the same density through math.fsum."""
            d = (x[0] - 1.0, x[1] + 2.0)
            return -0.5 * math.fsum(
                d[i] * precision[i][j] * d[j] for i in range(2) for j in range(2)
            )

        fallback = laplace_approximation(math_density, (0.0, 0.0))
        self.assertEqual(fallback.hessian_method, "finite-difference")
        self.assertIsNotNone(fallback.fallback_reason)
        self.assertAlmostEqual(fallback.covariance[0][0], exact[0][0], delta=1e-5)
        with self.assertRaises(AutodiffUnsupportedError):
            laplace_approximation(math_density, (0.0, 0.0), gradient="autodiff")

    def test_advi_autodiff_counts_one_call_per_gradient(self) -> None:
        """ADVI with autodiff spends one density call per gradient sample."""
        fit = advi(gaussian_10d, [0.0] * 10, seed=3, draws=0, max_iterations=400)
        self.assertEqual(fit.gradient_method, "autodiff")
        # ELBO estimates add elbo_samples calls per evaluation interval.
        elbo_calls = len(fit.elbo_trace) * 100
        self.assertEqual(
            fit.log_density_evaluations, fit.gradient_evaluations + elbo_calls
        )


class DualNormal:
    """Duck-typed normal law whose log_pdf accepts dual numbers."""

    support = RealSupport()

    def __init__(self, mu, sigma):
        """Store parameters (floats or duals)."""
        self.mu = mu
        self.sigma = sigma

    def log_pdf(self, x):
        """Return the normal log-density through admath."""
        z = (x - self.mu) / self.sigma
        return -0.5 * z * z - admath.log(self.sigma) - 0.5 * math.log(2 * math.pi)

    def sample(self, size, *, seed=None):
        """Draw values (prediction only)."""
        generator = random.Random(seed)
        return [generator.gauss(float(self.mu), float(self.sigma)) for _ in range(size)]


DATA = (1.2, 0.4, 2.1, 1.7, 0.9)
POST_VAR = 1.0 / (1.0 / 100.0 + len(DATA))
POST_MEAN = POST_VAR * sum(DATA)


class ProbabilisticProgrammingGradientTests(unittest.TestCase):
    """The PPL differentiates dual-capable models and falls back otherwise."""

    def test_dual_capable_model_uses_autodiff(self) -> None:
        """A duck-typed dual-aware likelihood is differentiated exactly."""

        def model(m):
            """Conjugate normal model."""
            mu = m.sample("mu", Normal(0.0, 10.0))
            m.observe("y", [DualNormal(mu, 1.0) for _ in DATA], DATA)

        program = ppl.trace(model)
        exact = sum(y - 0.3 for y in DATA) - 0.3 / 100.0
        self.assertAlmostEqual(
            program.gradient((0.3,), method="autodiff")[0], exact, delta=1e-12
        )
        posterior = quiet(ppl.sample, model, draws=600, warmup=400, chains=2, seed=4)
        self.assertEqual(posterior.gradient_method, "autodiff")
        summary = posterior.summary()[0]
        self.assertLess(abs(summary.mean - POST_MEAN), 4.0 * summary.mcse_mean)
        laplace = ppl.laplace(model, draws=10, seed=1)
        self.assertAlmostEqual(laplace.engine.covariance[0][0], POST_VAR, delta=1e-12)

    def test_standard_distributions_fall_back(self) -> None:
        """Probstat distributions reject dual parameters: finite differences."""

        def model(m):
            """Declare the same model with the library Normal."""
            mu = m.sample("mu", Normal(0.0, 10.0))
            m.observe("y", Normal(mu, 1.0), DATA)

        posterior = quiet(ppl.sample, model, draws=300, warmup=300, chains=2, seed=4)
        self.assertEqual(posterior.gradient_method, "finite-difference")
        self.assertIn("real number", posterior.gradient_fallback_reason)
        program = ppl.trace(model)
        exact = sum(y - 0.3 for y in DATA) - 0.3 / 100.0
        self.assertAlmostEqual(program.gradient((0.3,))[0], exact, delta=1e-6)
        with self.assertRaises(AutodiffUnsupportedError):
            program.gradient((0.3,), method="autodiff")


class SimulationBasedCalibrationTests(unittest.TestCase):
    """SBC ranks are uniform for exact posteriors only."""

    @staticmethod
    def _problem(scale: float):
        """Return prior, simulator and an (optionally mis-scaled) sampler."""

        def prior(generator):
            """Theta ~ N(0, 1)."""
            return (generator.gauss(0.0, 1.0),)

        def simulator(theta, generator):
            """Five observations y ~ N(theta, 1)."""
            return [generator.gauss(theta[0], 1.0) for _ in range(5)]

        def posterior(data, stream):
            """Exact conjugate posterior N(sum y / 6, 1/6), sd times ``scale``."""
            generator = stream.generator()
            mean = sum(data) / 6.0
            sd = math.sqrt(1.0 / 6.0) * scale
            return [(generator.gauss(mean, sd),) for _ in range(99)]

        return prior, simulator, posterior

    def test_exact_posterior_is_calibrated(self) -> None:
        """The conjugate posterior passes; an over-confident one fails."""
        result = simulation_based_calibration(
            *self._problem(1.0), replicates=400, seed=1
        )
        self.assertEqual(result.replicates, 400)
        self.assertEqual(result.bins, 20)
        self.assertEqual(sum(result.histograms[0]), 400)
        self.assertAlmostEqual(sum(result.expected), 400.0)
        self.assertTrue(result.calibrated(0.01))
        narrow = simulation_based_calibration(
            *self._problem(0.5), replicates=400, seed=1
        )
        self.assertLess(narrow.min_p_value(), 1e-6)
        self.assertFalse(narrow.calibrated())
        # Over-confidence piles ranks at both ends (U shape).
        histogram = narrow.histograms[0]
        self.assertGreater(histogram[0] + histogram[-1], 4 * narrow.expected[0])

    def test_rank_statistic_and_validation(self) -> None:
        """Ranks count draws below the truth; ties are shared."""
        self.assertEqual(rank_statistic(0.5, [0.1, 0.9, 0.3]), 2)
        self.assertEqual(rank_statistic(1.0, [1.0, 1.0, 0.0], tie_break=0.0), 1)
        self.assertEqual(rank_statistic(1.0, [1.0, 1.0, 0.0], tie_break=0.99), 3)
        prior, simulator, posterior = self._problem(1.0)
        with self.assertRaises(ValueError):
            simulation_based_calibration(prior, simulator, posterior, replicates=1)
        with self.assertRaises(ValueError):
            simulation_based_calibration(
                prior, simulator, posterior, replicates=10, bins=1
            )
        with self.assertRaises(TypeError):
            simulation_based_calibration(prior, 3, posterior, replicates=10)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
