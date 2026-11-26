"""Tests for the simulation-based inference workflow (experimental).

Oracles: the conjugate linear-Gaussian model (closed-form posteriors for the
sufficient mean and the insufficient first observation), exact quadrature of
the bimodal ``y = theta^2 + noise`` posterior, the Gaussian score identities
(checked against numerical integration and finite differences), and the
normal equations / Tikhonov solver for plug-and-play with a Gaussian proximal
map. Stochastic claims use fixed seeds; every number asserted against a
learned method was measured by running this file.

The fast tier keeps all budgets small (learned models are barely trained, so
only structural and design-level claims are asserted); the slow tier
(``PROBSTAT_SLOW=1``) runs the larger configuration and prints the table.
"""

from __future__ import annotations

import math
import os
import random
import statistics
import unittest
from collections.abc import Sequence

from probstat.foundations.random_streams import RandomStream
from probstat.generative.posterior_estimation import simulate_pairs
from probstat.inverse import tikhonov
from probstat.inverse.sbi_workflow import (
    PREDECLARED_SEEDS,
    BimodalConfig,
    GaussianScoreProblem,
    LinearGaussianDiffusionProblem,
    ProposalTrainingRefused,
    SimulationFailure,
    SimulatorContract,
    VPSchedule,
    WorkLedger,
    _affine_score,
    abc_tolerance_study,
    bimodal_contract,
    bimodal_exact_posterior,
    check_training_policy,
    crps_ensemble_value,
    diffusion_convergence_study,
    diffusion_posterior_sampling,
    draw_dataset,
    fit_summary_ood_detector,
    gaussian_proximal_denoiser,
    linear_gaussian_case,
    linear_gaussian_contract,
    make_splits,
    noisy_score_comparison,
    plug_and_play_proximal_gradient,
    posterior_score_t0,
    run_abc_rejection,
    run_abc_smc,
    run_bimodal_comparison,
    simulate_summary,
    slow_bimodal_config,
    synthetic_likelihood_grid,
    synthetic_log_likelihood,
    train_flow_matching,
    train_npe,
    train_nre,
)
from probstat.linalg import matvec, solve_spd

SLOW = os.environ.get("PROBSTAT_SLOW") == "1"


def _gauss_contract() -> SimulatorContract:
    """Return ``theta ~ N(0, 1)``, ``y = theta + N(0, 0.5^2)``."""

    def prior(rng: random.Random) -> list[float]:
        return [rng.gauss(0.0, 1.0)]

    def simulator(theta: Sequence[float], rng: random.Random) -> list[float]:
        return [theta[0] + 0.5 * rng.gauss(0.0, 1.0)]

    return SimulatorContract(prior, simulator, name="gauss")


class ContractTests(unittest.TestCase):
    """Streams, splits, accounting and simulator failures."""

    def test_splits_are_deterministic_and_separate(self) -> None:
        """Check splits are deterministic and separate."""
        contract = _gauss_contract()
        ledger = WorkLedger()
        a = make_splits(contract, train=8, validation=5, test=6, seed=7, ledger=ledger)
        b = make_splits(contract, train=8, validation=5, test=6, seed=7)
        c = make_splits(contract, train=8, validation=5, test=6, seed=8)
        self.assertEqual(a.train.parameters, b.train.parameters)
        self.assertEqual(a.test.summaries, b.test.summaries)
        self.assertNotEqual(a.train.parameters, c.train.parameters)
        # separate child streams: no pair of splits shares a draw
        pool = [*a.train.parameters, *a.validation.parameters, *a.test.parameters]
        self.assertEqual(len(set(pool)), len(pool))
        self.assertEqual(ledger.simulator_calls, 19)
        self.assertEqual(ledger.failures, 0)
        self.assertEqual((a.train.size, a.validation.size, a.test.size), (8, 5, 6))
        with self.assertRaises(TypeError):
            make_splits(contract, train=8, validation=5, test=6, seed=None)  # type: ignore[arg-type]

    def test_library_training_set_equals_train_split(self) -> None:
        """The library fitter simulates exactly the train split's pairs."""
        contract = _gauss_contract()
        splits = make_splits(contract, train=6, validation=2, test=2, seed=3)
        prior, sim = contract.bind(None)
        stream = splits.train_stream
        replay = []
        for n in range(6):
            rng = stream.child("simulation", n).generator()
            theta = tuple(prior(rng))
            replay.append((theta, tuple(sim(theta, rng))))
        self.assertEqual(
            replay,
            list(zip(splits.train.parameters, splits.train.summaries, strict=True)),
        )

    def test_work_ledger_matches_library_abc_count(self) -> None:
        """Check work ledger matches library abc count."""
        contract = _gauss_contract()
        ledger = WorkLedger()
        out = run_abc_rejection(
            contract, (0.3,), draws=20, tolerance=0.2, seed=5, ledger=ledger
        )
        self.assertEqual(out.status, "ok")
        self.assertEqual(ledger.simulator_calls, out.simulations)
        again = run_abc_rejection(
            contract, (0.3,), draws=20, tolerance=0.2, seed=5, ledger=WorkLedger()
        )
        self.assertEqual(out.parameters, again.parameters)

    def test_nonfinite_simulation_is_counted_and_raised(self) -> None:
        """Check nonfinite simulation is counted and raised."""

        def prior(rng: random.Random) -> list[float]:
            return [rng.random()]

        def bad(theta: Sequence[float], rng: random.Random) -> list[float]:
            return [math.nan if theta[0] > 0.5 else theta[0]]

        contract = SimulatorContract(prior, bad)
        ledger = WorkLedger()
        with self.assertRaises(SimulationFailure):
            draw_dataset(contract, 50, RandomStream(1), ledger=ledger)
        self.assertEqual(ledger.failures, 1)
        self.assertGreaterEqual(ledger.simulator_calls, 1)
        # discard policy: failures are retried and reported (about half fail)
        ok = draw_dataset(
            contract,
            20,
            RandomStream(1),
            failure_policy="discard",
            max_failure_fraction=0.9,
        )
        self.assertEqual(ok.size, 20)
        self.assertGreater(ok.failures, 0)
        self.assertAlmostEqual(ok.failure_rate, ok.failures / ok.attempts)
        self.assertTrue(all(p[0] <= 0.5 for p in ok.parameters))
        with self.assertRaises(ValueError):
            draw_dataset(
                contract,
                20,
                RandomStream(1),
                failure_policy="discard",
                max_failure_fraction=0.01,
            )
        with self.assertRaises(ValueError):
            draw_dataset(contract, 5, RandomStream(1), failure_policy="skip")

    def test_raising_simulator_and_invalid_summary(self) -> None:
        """Check raising simulator and invalid summary."""

        def prior(rng: random.Random) -> list[float]:
            return [1.0]

        def overflow(theta: Sequence[float], rng: random.Random) -> list[float]:
            return [math.exp(1000.0)]

        with self.assertRaises(SimulationFailure):
            simulate_summary(
                SimulatorContract(prior, overflow), (1.0,), random.Random(0)
            )
        empty = SimulatorContract(prior, lambda t, r: [1.0], summary=lambda y: [])
        with self.assertRaises(ValueError):
            simulate_summary(empty, (1.0,), random.Random(0))
        nonfinite = SimulatorContract(
            prior, lambda t, r: [1.0], summary=lambda y: [math.inf]
        )
        ledger = WorkLedger()
        with self.assertRaises(SimulationFailure):
            simulate_summary(nonfinite, (1.0,), random.Random(0), ledger)
        self.assertEqual((ledger.simulator_calls, ledger.failures), (1, 1))
        with self.assertRaises(TypeError):
            SimulatorContract(prior, "not callable")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            SimulatorContract(prior, lambda t, r: [1.0], summary=3)  # type: ignore[arg-type]


class SyntheticLikelihoodTests(unittest.TestCase):
    """Gaussian synthetic likelihood: approximation, variance, singularity."""

    def setUp(self) -> None:
        """Build the shared fixtures."""
        self.case = linear_gaussian_case()
        self.contract = linear_gaussian_contract(self.case, "mean")

    def test_matches_exact_gaussian_for_gaussian_summary(self) -> None:
        """The mean summary is exactly N(theta, sigma^2 / n): SL is consistent."""
        theta = 0.4
        observed = self.case.observed_summary("mean")
        variance = self.case.noise_sd**2 / self.case.n_obs
        exact = -0.5 * (
            (observed[0] - theta) ** 2 / variance + math.log(2 * math.pi * variance)
        )
        result = synthetic_log_likelihood(
            self.contract, (theta,), observed, simulations=400, replicates=6, seed=1
        )
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.approximate)
        assert result.log_likelihood_variance is not None
        self.assertGreater(result.log_likelihood_variance, 0.0)
        mcse = math.sqrt(result.log_likelihood_variance / result.replicates)
        # plug-in bias is O(1/n_sim) = O(2.5e-3); band is 4 MCSE + 0.02
        self.assertLess(abs(result.log_likelihood - exact), 4 * mcse + 0.02)
        self.assertAlmostEqual(result.mean[0], theta, delta=0.2)
        self.assertAlmostEqual(result.covariance[0][0], variance, delta=0.06)

    def test_variance_none_with_one_replicate_and_reproducible(self) -> None:
        """Check variance none with one replicate and reproducible."""
        observed = self.case.observed_summary("mean")
        one = synthetic_log_likelihood(
            self.contract, (0.0,), observed, simulations=30, seed=2
        )
        two = synthetic_log_likelihood(
            self.contract, (0.0,), observed, simulations=30, seed=2
        )
        self.assertIsNone(one.log_likelihood_variance)
        self.assertEqual(one.log_likelihood, two.log_likelihood)

    def test_singular_covariance_is_reported_not_jittered(self) -> None:
        """Check singular covariance is reported not jittered."""

        def prior(rng: random.Random) -> list[float]:
            return [rng.gauss(0.0, 1.0)]

        def duplicated(theta: Sequence[float], rng: random.Random) -> list[float]:
            value = theta[0] + rng.gauss(0.0, 1.0)
            return [value, value]  # perfectly dependent summaries

        contract = SimulatorContract(prior, duplicated)
        result = synthetic_log_likelihood(
            contract, (0.0,), (0.0, 0.0), simulations=50, replicates=3, seed=1
        )
        self.assertEqual(result.status, "singular_covariance")
        self.assertTrue(math.isnan(result.log_likelihood))
        self.assertIsNone(result.log_likelihood_variance)
        # fewer simulations than summary dimension + 1 is also singular
        pair = SimulatorContract(
            prior,
            lambda theta, rng: [rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)],
        )
        few = synthetic_log_likelihood(pair, (0.0,), (0.0, 0.0), simulations=2, seed=1)
        self.assertEqual(few.status, "singular_covariance")
        with self.assertRaises(ValueError):
            synthetic_log_likelihood(
                _gauss_contract(), (0.0,), (0.0, 0.0), simulations=5, seed=1
            )
        with self.assertRaises(ValueError):
            synthetic_log_likelihood(
                _gauss_contract(), (0.0,), (0.0,), simulations=1, seed=1
            )

    def test_grid_posterior_close_to_exact_for_sufficient_summary(self) -> None:
        """Check grid posterior close to exact for sufficient summary."""
        observed = self.case.observed_summary("mean")
        grid = tuple(-2.0 + 0.1 * i for i in range(41))
        ledger = WorkLedger()
        posterior = synthetic_likelihood_grid(
            self.contract,
            observed,
            grid,
            simulations=60,
            replicates=2,
            seed=4,
            ledger=ledger,
        )
        mean, variance = self.case.exact_full()
        self.assertEqual(posterior.status, "ok")
        self.assertEqual(ledger.simulator_calls, 41 * 60 * 2)
        self.assertAlmostEqual(posterior.mean, mean, delta=0.15)
        self.assertAlmostEqual(
            posterior.standard_deviation, math.sqrt(variance), delta=0.1
        )
        self.assertIsNotNone(posterior.mean_log_likelihood_variance)
        no_prior = SimulatorContract(
            self.contract.prior_sampler, self.contract.simulator
        )
        with self.assertRaises(ValueError):
            synthetic_likelihood_grid(no_prior, observed, grid, simulations=10, seed=1)


class ABCToleranceTests(unittest.TestCase):
    """Sufficient vs insufficient summaries on the linear-Gaussian model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the shared fixtures."""
        cls.study = abc_tolerance_study(draws=60, seeds=PREDECLARED_SEEDS[:4])
        cls.table = {(r["summary"], r["tolerance"]): r for r in cls.study.aggregate()}

    def test_exact_references(self) -> None:
        """Check exact references."""
        # prior N(0,1), n=5, sigma=1: full variance 1/6, y_1-only variance 1/2
        self.assertAlmostEqual(self.study.reference_full[1], 1.0 / 6.0, places=12)
        self.assertAlmostEqual(self.study.reference_first[1], 0.5, places=12)
        precision = 1.0 + 5.0
        case = self.study.case
        self.assertAlmostEqual(
            self.study.reference_full[0], sum(case.observed) / precision, places=12
        )

    def test_tolerance_trend_over_predeclared_seeds(self) -> None:
        """Check tolerance trend over predeclared seeds."""
        eps = self.study.tolerances
        mean_rows = [self.table[("mean", e)] for e in eps]
        # sufficient summary: error to the true posterior falls as eps falls
        self.assertLess(mean_rows[-1]["w1_to_full"], mean_rows[0]["w1_to_full"])
        # cost rises as the tolerance falls (common random numbers, nested)
        sims = [r["mean_simulations"] for r in mean_rows]
        self.assertEqual(sims, sorted(sims))
        # the trend is a statement about seeds, not a monotone sample error
        self.assertTrue(all(0 <= r["seeds_improved"] <= 4 for r in mean_rows[1:]))

    def test_insufficient_summary_does_not_recover_full_posterior(self) -> None:
        """Check insufficient summary does not recover full posterior."""
        eps_min = self.study.tolerances[-1]
        good = self.table[("mean", eps_min)]
        bad = self.table[("first", eps_min)]
        self.assertGreater(bad["w1_to_full"], 2.0 * good["w1_to_full"])
        # its sd stays near the y_1-only posterior sd (sqrt(0.5) = 0.707),
        # far above the full-data sd (sqrt(1/6) = 0.408)
        self.assertGreater(bad["mean_sd"], 0.6)
        self.assertLess(good["mean_sd"], 0.55)

    def test_study_is_deterministic_and_validates(self) -> None:
        """Check study is deterministic and validates."""
        again = abc_tolerance_study(draws=60, seeds=PREDECLARED_SEEDS[:4])
        self.assertEqual(self.study.rows, again.rows)
        with self.assertRaises(ValueError):
            abc_tolerance_study(tolerances=(0.1, 0.5))
        with self.assertRaises(ValueError):
            abc_tolerance_study(tolerances=(0.0,))
        with self.assertRaises(ValueError):
            abc_tolerance_study(summaries=("median",))
        with self.assertRaises(ValueError):
            abc_tolerance_study(draws=3)

    def test_abc_smc_wrapper_targets_exact_posterior(self) -> None:
        """Check abc smc wrapper targets exact posterior."""
        case = linear_gaussian_case()
        contract = linear_gaussian_contract(case, "mean")
        ledger = WorkLedger()
        out = run_abc_smc(
            contract,
            case.observed_summary("mean"),
            draws=150,
            generations=4,
            seed=3,
            ledger=ledger,
        )
        self.assertEqual(out.status, "ok")
        self.assertAlmostEqual(math.fsum(out.weights), 1.0, places=12)
        self.assertEqual(ledger.simulator_calls, out.simulations)
        mean = math.fsum(
            w * p[0] for w, p in zip(out.weights, out.parameters, strict=True)
        )
        self.assertAlmostEqual(mean, case.exact_full()[0], delta=0.2)
        self.assertTrue(out.approximate)
        with self.assertRaises(ValueError):
            run_abc_smc(
                SimulatorContract(contract.prior_sampler, contract.simulator),
                (0.0,),
                draws=10,
                seed=1,
            )


class ABCFailureTests(unittest.TestCase):
    """Zero acceptance, invalid distance and invalid summary."""

    def test_zero_acceptance_is_reported(self) -> None:
        """Check zero acceptance is reported."""
        contract = _gauss_contract()
        out = run_abc_rejection(
            contract,
            (500.0,),
            draws=5,
            tolerance=1e-3,
            max_simulations=200,
            seed=1,
        )
        self.assertEqual(out.status, "zero_acceptance")
        self.assertEqual(out.simulations, 200)
        self.assertEqual(out.accepted, 0)
        self.assertEqual(out.parameters, ())
        self.assertTrue(math.isnan(out.tolerance))

    def test_insufficient_acceptance_is_reported(self) -> None:
        """Check insufficient acceptance is reported."""
        out = run_abc_rejection(
            _gauss_contract(),
            (0.0,),
            draws=500,
            tolerance=0.05,
            max_simulations=300,
            seed=1,
        )
        self.assertEqual(out.status, "insufficient_acceptance")
        self.assertGreater(out.accepted, 0)
        self.assertLess(out.accepted, 500)

    def test_invalid_distance_summary_and_options(self) -> None:
        """Check invalid distance summary and options."""
        contract = _gauss_contract()
        with self.assertRaises(ValueError):
            run_abc_rejection(
                contract,
                (0.0,),
                draws=5,
                tolerance=1.0,
                distance=lambda s, o: math.nan,
                seed=1,
            )
        with self.assertRaises(ValueError):
            run_abc_rejection(
                contract,
                (0.0,),
                draws=5,
                tolerance=1.0,
                distance=lambda s, o: -1.0,
                seed=1,
            )
        with self.assertRaises(ValueError):  # wrong observed width
            run_abc_rejection(contract, (0.0, 1.0), draws=5, tolerance=1.0, seed=1)
        with self.assertRaises(ValueError):  # both tolerance and quantile
            run_abc_rejection(
                contract, (0.0,), draws=5, tolerance=1.0, quantile=0.5, seed=1
            )
        with self.assertRaises(TypeError):
            run_abc_rejection(
                contract,
                (0.0,),
                draws=5,
                tolerance=1.0,
                distance=3,
                seed=1,  # type: ignore[arg-type]
            )

    def test_simulation_failure_propagates_through_abc(self) -> None:
        """Check simulation failure propagates through abc."""

        def prior(rng: random.Random) -> list[float]:
            return [0.0]

        contract = SimulatorContract(prior, lambda t, r: [math.nan])
        ledger = WorkLedger()
        with self.assertRaises(SimulationFailure):
            run_abc_rejection(
                contract, (0.0,), draws=3, tolerance=1.0, seed=1, ledger=ledger
            )
        self.assertEqual(ledger.failures, 1)


class TrainingFailureTests(unittest.TestCase):
    """Training wrappers report failures and never swallow simulator errors."""

    def test_training_failure_is_a_status(self) -> None:
        """Check training failure is a status."""
        contract = _gauss_contract()
        npe = train_npe(contract, 1, seed=1)  # library needs >= 2 pairs
        self.assertEqual(npe.status, "failed")
        self.assertIsNone(npe.model)
        self.assertIn("simulations", npe.message)
        nre = train_nre(contract, 20, batch_size=1, iterations=5, seed=1)
        self.assertEqual(nre.status, "failed")
        self.assertIsNone(nre.model)

    def test_simulator_failure_during_training_propagates(self) -> None:
        """Check simulator failure during training propagates."""

        def prior(rng: random.Random) -> list[float]:
            return [rng.gauss(0.0, 1.0)]

        contract = SimulatorContract(prior, lambda t, r: [math.inf])
        with self.assertRaises(SimulationFailure):
            train_npe(contract, 5, seed=1)
        with self.assertRaises(SimulationFailure):
            train_nre(contract, 5, seed=1)

    def test_successful_fit_records_provenance(self) -> None:
        """Check successful fit records provenance."""
        ledger = WorkLedger()
        fit = train_npe(
            _gauss_contract(),
            30,
            hidden_layers=(4,),
            iterations=20,
            batch_size=16,
            seed=9,
            ledger=ledger,
        )
        self.assertIn(fit.status, ("ok", "loss_not_decreased"))
        self.assertIsNotNone(fit.model)
        self.assertEqual(
            (fit.simulations, fit.iterations, fit.hidden_layers, fit.seed),
            (30, 20, (4,), 9),
        )
        self.assertEqual(ledger.simulator_calls, 30)
        self.assertTrue(fit.approximate)


class ExactReferenceTests(unittest.TestCase):
    """Quadrature reference and scoring helpers."""

    def test_exact_bimodal_posterior_is_symmetric_with_two_modes(self) -> None:
        """Check exact bimodal posterior is symmetric with two modes."""
        post = bimodal_exact_posterior(1.0)
        self.assertAlmostEqual(post.mean(), 0.0, places=10)
        self.assertAlmostEqual(post.mass_in(0.0, 3.0), 0.5, places=3)
        self.assertGreater(post.mass_in(0.7, 1.3), 0.45)
        self.assertGreater(post.mass_in(-1.3, -0.7), 0.45)
        self.assertLess(post.mass_in(-0.3, 0.3), 0.01)
        self.assertAlmostEqual(math.fsum(post.weights()), 1.0, places=12)
        # sd of the two-point-like mixture is close to sqrt(y) = 1
        self.assertAlmostEqual(post.standard_deviation(), 1.0, delta=0.03)

    def test_exact_posterior_matches_independent_density(self) -> None:
        """Check exact posterior matches independent density."""
        post = bimodal_exact_posterior(1.0, points=2001)
        s = 0.25
        theta = 1.02
        unnormalized = math.exp(-0.5 * ((1.0 - theta**2) / s) ** 2)
        peak_ratio = post.density[
            post.grid.index(min(post.grid, key=lambda g: abs(g - 1.0)))
        ]
        # ratio of densities at two points is the ratio of the likelihoods
        other = 1.0
        ref = math.exp(-0.5 * ((1.0 - other**2) / s) ** 2)
        got = math.exp(post.log_density_at(theta) - post.log_density_at(other))
        self.assertAlmostEqual(got, unnormalized / ref, delta=2e-3)
        self.assertGreater(peak_ratio, 0.0)

    def test_grid_sampling_reproduces_moments(self) -> None:
        """Check grid sampling reproduces moments."""
        post = bimodal_exact_posterior(1.0)
        draws = post.sample(4000, random.Random(3))
        self.assertAlmostEqual(statistics.fmean(draws), 0.0, delta=0.06)
        self.assertAlmostEqual(sum(1 for d in draws if d > 0) / 4000, 0.5, delta=0.04)

    def test_crps_known_values(self) -> None:
        """Check crps known values."""
        # one-point ensemble: CRPS = |x - y|
        self.assertAlmostEqual(crps_ensemble_value([2.0], 5.0), 3.0)
        # two-point ensemble {0, 2}, y = 1: E|X-y| = 1, E|X-X'| = 1
        self.assertAlmostEqual(crps_ensemble_value([0.0, 2.0], 1.0), 1.0 - 0.5 * 1.0)
        with self.assertRaises(ValueError):
            crps_ensemble_value([], 1.0)
        with self.assertRaises(ValueError):
            crps_ensemble_value([math.nan], 1.0)

    def test_invalid_arguments(self) -> None:
        """Check invalid arguments."""
        with self.assertRaises(ValueError):
            bimodal_exact_posterior(1.0, noise_sd=-1.0)
        with self.assertRaises(ValueError):
            bimodal_exact_posterior(1.0, points=10)
        with self.assertRaises(ValueError):
            bimodal_contract(0.0)
        with self.assertRaises(ValueError):
            BimodalConfig(n_train=1)


class BimodalComparisonTests(unittest.TestCase):
    """Fast-tier comparison: structure, accounting, design-level claims."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the shared fixtures."""
        cls.config = BimodalConfig()
        cls.result = run_bimodal_comparison(cls.config)
        cls.by_name = {r.method: r for r in cls.result.reports}

    def test_table_structure_and_labels(self) -> None:
        """Check table structure and labels."""
        table = self.result.table()
        self.assertEqual(
            [row["method"] for row in table], ["exact", "abc", "prior", "npe", "nre"]
        )
        self.assertFalse(self.by_name["exact"].approximate)
        self.assertEqual(self.by_name["exact"].status, "reference")
        for name in ("abc", "npe", "nre", "prior"):
            self.assertTrue(self.by_name[name].approximate, name)
        self.assertEqual(self.by_name["prior"].status, "baseline")
        self.assertEqual(self.result.training_status.keys(), {"npe", "nre"})

    def test_cost_accounting_is_measured(self) -> None:
        """Check cost accounting is measured."""
        cfg = self.config
        abc = self.by_name["abc"]
        # ABC: abc_simulations per posterior, test pairs + the headline posterior
        self.assertEqual(abc.simulations_per_posterior, cfg.abc_simulations)
        self.assertEqual(
            abc.evaluation_simulations, (cfg.n_test + 1) * cfg.abc_simulations
        )
        self.assertEqual(
            abc.sbc_simulations, cfg.n_sbc * cfg.abc_simulations + cfg.n_sbc
        )
        for name in ("npe", "nre"):
            self.assertEqual(self.by_name[name].training_simulations, cfg.n_train)
            self.assertEqual(self.by_name[name].evaluation_simulations, 0)
        for name in ("exact", "prior"):
            self.assertEqual(self.by_name[name].training_simulations, 0)
            self.assertEqual(self.by_name[name].evaluation_simulations, 0)
        self.assertEqual(
            self.result.split_simulations,
            cfg.n_train + cfg.n_validation + cfg.n_test,
        )
        self.assertEqual(
            self.result.split_sizes, (cfg.n_train, cfg.n_validation, cfg.n_test)
        )

    def test_reference_and_baseline_ordering(self) -> None:
        """Check reference and baseline ordering."""
        exact, prior = self.by_name["exact"], self.by_name["prior"]
        assert exact.w1_to_exact is not None and prior.w1_to_exact is not None
        # the exact row carries only Monte Carlo noise; the prior ignores data
        self.assertLess(exact.w1_to_exact, 0.35)
        self.assertGreater(prior.w1_to_exact, 3.0 * exact.w1_to_exact)
        assert exact.crps is not None and prior.crps is not None
        self.assertGreater(prior.crps, exact.crps)
        assert exact.predictive_misfit is not None
        assert prior.predictive_misfit is not None
        self.assertGreater(prior.predictive_misfit, 5.0 * exact.predictive_misfit)

    def test_sbc_harness_passes_exact_and_is_blind_to_prior_only(self) -> None:
        """Check sbc harness passes exact and is blind to prior only."""
        # exact quadrature is calibrated; the prior-only posterior also has
        # uniform ranks (SBC averages over the prior) and so passes too.
        self.assertTrue(self.by_name["exact"].sbc_calibrated)
        self.assertTrue(self.by_name["prior"].sbc_calibrated)
        self.assertIsNotNone(self.by_name["npe"].sbc_min_p_value)

    def test_gaussian_npe_cannot_occupy_both_modes(self) -> None:
        """Check gaussian npe cannot occupy both modes."""
        npe, exact = self.by_name["npe"], self.by_name["exact"]
        assert npe.mode_mass_error is not None and exact.mode_mass_error is not None
        assert npe.mode_mass_positive is not None
        assert npe.mode_mass_negative is not None
        self.assertGreater(npe.mode_mass_error, exact.mode_mass_error + 0.1)
        # exact puts about half the mass in each +-sqrt(y) window
        assert exact.mode_mass_positive is not None
        assert exact.mode_mass_negative is not None
        self.assertGreater(exact.mode_mass_positive, 0.35)
        self.assertGreater(exact.mode_mass_negative, 0.35)
        self.assertAlmostEqual(sum(self.result.headline_exact_modes), 1.0, delta=0.05)

    def test_nre_reports_normalization_diagnostic_and_densities(self) -> None:
        """Check nre reports normalization diagnostic and densities."""
        nre = self.by_name["nre"]
        self.assertIsNotNone(nre.ratio_normalization)
        self.assertIsNotNone(nre.test_mean_log_density)
        self.assertIsNotNone(nre.validation_mean_log_density)
        self.assertIsNone(self.by_name["abc"].test_mean_log_density)
        self.assertAlmostEqual(
            self.by_name["prior"].test_mean_log_density or 0.0,
            -math.log(6.0),
            places=12,
        )

    def test_coverage_and_scores_are_proportions_and_finite(self) -> None:
        """Check coverage and scores are proportions and finite."""
        for r in self.result.reports:
            self.assertIsNotNone(r.coverage90)
            assert r.coverage90 is not None and r.crps is not None
            self.assertTrue(0.0 <= r.coverage90 <= 1.0)
            self.assertTrue(math.isfinite(r.crps))

    def test_ood_heuristic_flags_far_observation_only(self) -> None:
        """Check ood heuristic flags far observation only."""
        ood = self.result.ood
        self.assertTrue(ood.ood_flagged)
        self.assertLessEqual(ood.test_flag_fraction, 0.2)
        self.assertGreater(ood.ood_score, ood.threshold)
        self.assertIn("heuristic", ood.heuristic.lower())
        # the Gaussian NPE itself returns a finite, unwarned answer there
        assert ood.npe_at_ood is not None
        self.assertTrue(all(math.isfinite(v) for v in ood.npe_at_ood))

    def test_comparison_is_deterministic(self) -> None:
        """Check comparison is deterministic."""
        tiny = BimodalConfig(
            n_train=30,
            n_validation=3,
            n_test=3,
            n_sbc=4,
            sbc_draws=5,
            eval_draws=10,
            abc_simulations=40,
            npe_hidden=(3,),
            npe_iterations=5,
            nre_hidden=(3,),
            nre_iterations=5,
            nre_grid_points=11,
        )
        first = run_bimodal_comparison(tiny).table()
        second = run_bimodal_comparison(tiny).table()
        for a, b in zip(first, second, strict=True):
            a = {k: v for k, v in a.items() if k != "train_seconds"}
            b = {k: v for k, v in b.items() if k != "train_seconds"}
            self.assertEqual(a, b)

    def test_simulation_failure_propagates_from_splits(self) -> None:
        """Check simulation failure propagates from splits."""
        contract = SimulatorContract(
            lambda rng: [rng.random()], lambda t, r: [math.nan]
        )
        with self.assertRaises(SimulationFailure):
            make_splits(contract, train=3, validation=2, test=2, seed=1)


class ScoreTests(unittest.TestCase):
    """Exact Gaussian scores; naive addition is exact only at t = 0."""

    def setUp(self) -> None:
        """Build the shared fixtures."""
        self.problem = GaussianScoreProblem(
            prior_mean=0.5, prior_sd=1.5, coefficient=2.0, noise_sd=0.7, observation=1.3
        )

    def test_score_at_t0_is_prior_score_plus_likelihood_gradient(self) -> None:
        """Check score at t0 is prior score plus likelihood gradient."""
        mu, v = self.problem.posterior()
        # independent oracle: finite difference of the unnormalized log posterior
        p = self.problem

        def log_post(x: float) -> float:
            return (
                -0.5 * ((x - p.prior_mean) / p.prior_sd) ** 2
                - 0.5 * ((p.observation - p.coefficient * x) / p.noise_sd) ** 2
            )

        for x in (-2.0, 0.0, 0.31, 1.7):
            additive, closed = posterior_score_t0(p, x)
            h = 1e-5
            fd = (log_post(x + h) - log_post(x - h)) / (2 * h)
            self.assertAlmostEqual(additive, closed, places=12)
            self.assertAlmostEqual(additive, fd, places=6)
        self.assertAlmostEqual(posterior_score_t0(p, mu)[1], 0.0, places=14)
        self.assertGreater(v, 0.0)

    def test_noisy_scores_against_numerical_integration(self) -> None:
        """Exact noisy score = d/dx log of the integrated marginal p_t(x | y)."""
        p = self.problem
        mu, v = p.posterior()
        alpha, s = 0.8, 0.6
        sd0 = math.sqrt(v)
        n = 6001
        lo, hi = mu - 12.0 * sd0, mu + 12.0 * sd0
        step = (hi - lo) / (n - 1)
        nodes = [lo + i * step for i in range(n)]
        post0 = [math.exp(-0.5 * ((t - mu) / sd0) ** 2) for t in nodes]

        def log_marginal(x: float) -> float:
            total = math.fsum(
                w * math.exp(-0.5 * ((x - alpha * t) / s) ** 2)
                for w, t in zip(post0, nodes, strict=True)
            )
            return math.log(total)

        for x in (-1.0, 0.4, 1.6):
            h = 1e-4
            numeric = (log_marginal(x + h) - log_marginal(x - h)) / (2 * h)
            row = noisy_score_comparison(p, alpha, s, x)
            self.assertAlmostEqual(row.exact, numeric, places=5)
            # exact guided score (noisy likelihood) reproduces it to rounding
            self.assertLess(abs(row.exact_guided_error), 1e-12)
            # naive addition and the Tweedie plug-in are NOT exact
            self.assertGreater(abs(row.naive_error), 1e-2)
            self.assertGreater(abs(row.tweedie_error), 1e-3)

    def test_noisy_formulas_reduce_to_t0(self) -> None:
        """Check noisy formulas reduce to t0."""
        for x in (-0.7, 0.9):
            row = noisy_score_comparison(self.problem, 1.0, 0.0, x)
            additive, closed = posterior_score_t0(self.problem, x)
            for value in (row.exact, row.naive, row.tweedie, row.exact_guided):
                self.assertAlmostEqual(value, closed, places=11)
            self.assertAlmostEqual(additive, closed, places=12)

    def test_naive_error_grows_with_noise_level(self) -> None:
        """Check naive error grows with noise level."""
        errors = [
            abs(
                noisy_score_comparison(
                    self.problem, math.sqrt(1 - s * s), s, 0.4
                ).naive_error
            )
            for s in (0.05, 0.3, 0.7)
        ]
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])

    def test_invalid_arguments(self) -> None:
        """Check invalid arguments."""
        with self.assertRaises(ValueError):
            noisy_score_comparison(self.problem, 0.0, 0.5, 0.0)
        with self.assertRaises(ValueError):
            noisy_score_comparison(self.problem, 1.0, -0.5, 0.0)
        with self.assertRaises(ValueError):
            GaussianScoreProblem(0.0, 0.0, 1.0, 1.0, 0.0)


class PlugAndPlayTests(unittest.TestCase):
    """Fixed-point iteration with user denoisers."""

    A = (
        (2.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 3.0),
        (1.0, 0.0, 1.0),
        (0.0, 2.0, 1.0),
    )
    B = (1.0, -0.5, 2.0, 0.3, 1.2)

    def test_gaussian_prox_denoiser_reduces_to_tikhonov(self) -> None:
        """Check gaussian prox denoiser reduces to tikhonov."""
        lam = 0.8
        result = plug_and_play_proximal_gradient(
            self.A,
            self.B,
            gaussian_proximal_denoiser(lam * lam),
            max_iterations=20000,
            tolerance=1e-13,
        )
        reference = tikhonov(self.A, self.B, lam).solution
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, "tolerance")
        for got, want in zip(result.solution, reference, strict=True):
            self.assertAlmostEqual(got, want, places=8)
        # independent oracle: normal equations (A'A + mu I) x = A'b
        at = tuple(zip(*self.A, strict=True))
        ata = tuple(
            tuple(
                sum(at[i][k] * self.A[k][j] for k in range(5))
                + (lam**2 if i == j else 0.0)
                for j in range(3)
            )
            for i in range(3)
        )
        normal = solve_spd(ata, matvec(at, self.B))
        for got, want in zip(result.solution, normal, strict=True):
            self.assertAlmostEqual(got, want, places=8)
        # non-expansive prox: observed Lipschitz lower bound <= 1/(1 + tau mu)
        self.assertLessEqual(result.lipschitz_lower_bound, 1.0 + 1e-12)
        self.assertLess(result.residual, 1e-9)
        self.assertEqual(result.denoiser_calls, result.iterations)
        self.assertIn("fixed point", result.interpretation)

    def test_residual_decreases_to_zero_along_the_run(self) -> None:
        """Check residual decreases to zero along the run."""
        result = plug_and_play_proximal_gradient(
            self.A, self.B, gaussian_proximal_denoiser(0.5), max_iterations=300
        )
        history = result.residual_history
        self.assertLess(history[-1], 0.05 * history[0])
        self.assertEqual(len(history), result.iterations)

    def test_identity_denoiser_is_least_squares(self) -> None:
        """Check identity denoiser is least squares."""
        result = plug_and_play_proximal_gradient(
            self.A,
            self.B,
            lambda v, step: v,
            max_iterations=40000,
            tolerance=1e-13,
        )
        at = tuple(zip(*self.A, strict=True))
        ata = tuple(
            tuple(sum(at[i][k] * self.A[k][j] for k in range(5)) for j in range(3))
            for i in range(3)
        )
        normal = solve_spd(ata, matvec(at, self.B))
        self.assertTrue(result.converged)
        for got, want in zip(result.solution, normal, strict=True):
            self.assertAlmostEqual(got, want, places=6)

    def test_expansive_denoiser_is_flagged_and_does_not_converge(self) -> None:
        """Check expansive denoiser is flagged and does not converge."""
        result = plug_and_play_proximal_gradient(
            self.A,
            self.B,
            lambda v, step: tuple(2.0 * x for x in v),
            max_iterations=60,
        )
        self.assertFalse(result.converged)
        self.assertGreater(result.lipschitz_lower_bound, 1.5)
        self.assertGreater(result.residual, 1.0)

    def test_failures(self) -> None:
        """Check failures."""
        with self.assertRaises(TypeError):
            plug_and_play_proximal_gradient(self.A, self.B, 3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            plug_and_play_proximal_gradient(self.A, self.B, lambda v, s: v[:2])
        with self.assertRaises(ValueError):
            plug_and_play_proximal_gradient(self.A, self.B[:3], lambda v, s: v)
        with self.assertRaises(ValueError):
            plug_and_play_proximal_gradient(self.A, self.B, lambda v, s: v, step=-1.0)
        with self.assertRaises(ValueError):
            gaussian_proximal_denoiser(-1.0)
        broken = plug_and_play_proximal_gradient(
            self.A, self.B, lambda v, s: (math.nan,) * 3
        )
        self.assertEqual(broken.stop_reason, "non_finite")
        self.assertFalse(broken.converged)


class OODTests(unittest.TestCase):
    """Mahalanobis heuristic."""

    def setUp(self) -> None:
        """Build the shared fixtures."""
        rng = random.Random(5)
        self.rows = [(rng.gauss(0, 1), rng.gauss(0, 2)) for _ in range(400)]

    def test_far_point_flagged_and_training_flag_rate(self) -> None:
        """Check far point flagged and training flag rate."""
        det = fit_summary_ood_detector(self.rows, quantile=0.99)
        flagged = sum(det.is_out_of_distribution(r) for r in self.rows)
        self.assertLessEqual(flagged, 4)  # about 1% of 400
        self.assertTrue(det.is_out_of_distribution((12.0, 0.0)))
        self.assertFalse(det.is_out_of_distribution((0.1, 0.1)))
        self.assertIn("heuristic", det.heuristic)
        # score of the mean is ~0; score is the squared Mahalanobis distance
        self.assertAlmostEqual(det.score(det.mean), 0.0, places=12)

    def test_failures(self) -> None:
        """Check failures."""
        with self.assertRaises(ValueError):
            fit_summary_ood_detector([(1.0, 1.0)] * 10)  # singular covariance
        with self.assertRaises(ValueError):
            fit_summary_ood_detector([(1.0, 2.0)])
        with self.assertRaises(ValueError):
            fit_summary_ood_detector(self.rows, quantile=1.5)
        det = fit_summary_ood_detector(self.rows)
        with self.assertRaises(ValueError):
            det.score((1.0,))
        with self.assertRaises(ValueError):
            det.score((math.nan, 0.0))


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowTierTests(unittest.TestCase):
    """Larger budgets: prints the measured tables."""

    def test_large_tolerance_study(self) -> None:
        """Check large tolerance study."""
        study = abc_tolerance_study(draws=200, seeds=PREDECLARED_SEEDS)
        rows = study.aggregate()
        for row in rows:
            print(
                f"SLOW study {row['summary']:>5} eps={row['tolerance']:<5} "
                f"w1_full={row['w1_to_full']:.4f} "
                f"w1_summary={row['w1_to_summary']:.4f} "
                f"sd={row['mean_sd']:.4f} sims={row['mean_simulations']:.0f} "
                f"seeds_improved={row['seeds_improved']}"
            )
        by = {(r["summary"], r["tolerance"]): r for r in rows}
        eps = study.tolerances
        self.assertLess(
            by[("mean", eps[-1])]["w1_to_full"], by[("mean", eps[0])]["w1_to_full"]
        )
        self.assertGreater(
            by[("first", eps[-1])]["w1_to_full"],
            3.0 * by[("mean", eps[-1])]["w1_to_full"],
        )

    def test_large_bimodal_comparison(self) -> None:
        """Check large bimodal comparison."""
        result = run_bimodal_comparison(slow_bimodal_config())
        print("SLOW bimodal training status:", result.training_status)
        print("SLOW bimodal OOD:", result.ood)
        for row in result.table():
            print(
                "SLOW bimodal",
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                },
            )
        by = {r.method: r for r in result.reports}
        self.assertTrue(by["exact"].sbc_calibrated)
        assert by["npe"].mode_mass_error is not None
        assert by["exact"].mode_mass_error is not None
        self.assertGreater(by["npe"].mode_mass_error, by["exact"].mode_mass_error)


def _diffusion_problem() -> LinearGaussianDiffusionProblem:
    """Return a 2-parameter, 3-observation correlated linear-Gaussian problem."""
    return LinearGaussianDiffusionProblem(
        (0.5, -1.0),
        ((1.0, 0.3), (0.3, 2.0)),
        ((1.0, 0.5), (0.0, 1.0), (1.0, -1.0)),
        ((0.25, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.4)),
        (0.8, -0.2, 1.1),
    )


class DiffusionExactScoreTests(unittest.TestCase):
    """Reverse-time sampler with exact Gaussian scores (no network)."""

    def setUp(self) -> None:
        """Build the problem."""
        self.problem = _diffusion_problem()

    def test_posterior_matches_dense_normal_equations(self) -> None:
        """Check the closed-form posterior against the normal equations."""
        mu, cov = self.problem.posterior()
        a = self.problem.forward
        gamma_inv = ((4.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.5))
        c0_inv = tuple(
            tuple(v / 1.91 for v in row) for row in ((2.0, -0.3), (-0.3, 1.0))
        )
        precision = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                precision[i][j] = c0_inv[i][j] + sum(
                    a[k][i] * gamma_inv[k][k] * a[k][j] for k in range(3)
                )
        # cov * precision must be the identity
        for i in range(2):
            for j in range(2):
                value = sum(cov[i][k] * precision[k][j] for k in range(2))
                self.assertAlmostEqual(value, 1.0 if i == j else 0.0, places=12)
        rhs = [
            sum(c0_inv[i][j] * m for j, m in enumerate((0.5, -1.0)))
            + sum(
                a[k][i] * gamma_inv[k][k] * y
                for k, y in enumerate(self.problem.observation)
            )
            for i in range(2)
        ]
        for i in range(2):
            self.assertAlmostEqual(
                sum(precision[i][j] * mu[j] for j in range(2)), rhs[i], places=12
            )

    def test_scores_match_scalar_closed_forms(self) -> None:
        """Check exact and DPS scores against the scalar score comparison."""
        scalar = GaussianScoreProblem(0.3, 1.2, 0.8, 0.5, 0.7)
        lifted = LinearGaussianDiffusionProblem(
            (0.3,), ((1.44,),), ((0.8,),), ((0.25,),), (0.7,)
        )
        schedule = VPSchedule()
        for t in (0.0, 0.05, 0.3, 1.0):
            alpha, s2 = schedule.alpha(t), schedule.noise_variance(t)
            slope_e, off_e = _affine_score(lifted, schedule, t, "exact")
            slope_a, off_a = _affine_score(lifted, schedule, t, "dps_approximate")
            for x in (-1.3, 0.2, 2.0):
                ref = noisy_score_comparison(scalar, alpha, math.sqrt(s2), x)
                self.assertAlmostEqual(
                    slope_e[0][0] * x + off_e[0], ref.exact, delta=1e-9
                )
                self.assertAlmostEqual(
                    slope_a[0][0] * x + off_a[0], ref.tweedie, delta=1e-9
                )

    def test_dps_score_is_exact_at_time_zero_and_not_for_noisy_time(self) -> None:
        """Check the DPS-style score equals the exact one only at t = 0."""
        schedule = VPSchedule()
        slope_e, off_e = _affine_score(self.problem, schedule, 0.0, "exact")
        slope_a, off_a = _affine_score(self.problem, schedule, 0.0, "dps_approximate")
        for re, ra in zip(slope_e, slope_a, strict=True):
            for e, a in zip(re, ra, strict=True):
                self.assertAlmostEqual(e, a, places=10)
        for e, a in zip(off_e, off_a, strict=True):
            self.assertAlmostEqual(e, a, places=10)
        slope_e, _ = _affine_score(self.problem, schedule, 0.5, "exact")
        slope_a, _ = _affine_score(self.problem, schedule, 0.5, "dps_approximate")
        gap = max(
            abs(e - a)
            for re, ra in zip(slope_e, slope_a, strict=True)
            for e, a in zip(re, ra, strict=True)
        )
        self.assertGreater(gap, 0.05)

    def test_exact_score_error_is_first_order_in_the_step(self) -> None:
        """Check measured first-order convergence for both reverse samplers."""
        for sampler in ("sde", "ode"):
            study = diffusion_convergence_study(
                self.problem, steps=(40, 80, 160, 320), sampler=sampler
            )
            self.assertFalse(study.approximate)
            errors = [r.mean_error for r in study.results]
            self.assertTrue(
                all(a > b for a, b in zip(errors, errors[1:], strict=False))
            )
            for order in study.mean_orders:
                self.assertTrue(0.9 < order < 1.1, (sampler, order))
            self.assertTrue(0.8 < study.covariance_orders[-1] < 1.25)
            last = study.results[-1]
            self.assertFalse(last.approximate)
            self.assertLess(last.mean_error, 5e-3)
            self.assertLess(last.covariance_error, 5e-3)
            self.assertTrue(all(w < 5e-3 for w in last.w2_per_coordinate))

    def test_samples_match_exact_posterior_within_monte_carlo_error(self) -> None:
        """Check sample mean, covariance and per-coordinate W1 (measured)."""
        n = 400
        result = diffusion_posterior_sampling(
            self.problem, steps=80, n_samples=n, seed=3
        )
        trace = sum(result.exact_covariance[i][i] for i in range(2))
        assert result.sample_mean_error is not None
        assert result.sample_covariance_error is not None
        assert result.w1_per_coordinate is not None
        self.assertLess(result.sample_mean_error, 5.0 * math.sqrt(trace / n))
        self.assertLess(result.sample_covariance_error, 0.12)
        for w1, v in zip(
            result.w1_per_coordinate,
            (result.exact_covariance[0][0], result.exact_covariance[1][1]),
            strict=True,
        ):
            self.assertLess(w1, 0.35 * math.sqrt(v))
        self.assertEqual(result.score_evaluations, 80 * n)
        # the sample mean also agrees with the discretized sampler's own mean
        for s, d in zip(result.sample_mean or (), result.discretized_mean, strict=True):
            self.assertAlmostEqual(s, d, delta=0.1)

    def test_ode_sampler_is_deterministic_given_the_initial_state(self) -> None:
        """Check samplers reproduce with the seed and differ across seeds."""
        first = diffusion_posterior_sampling(
            self.problem, steps=20, n_samples=5, seed=7
        )
        again = diffusion_posterior_sampling(
            self.problem, steps=20, n_samples=5, seed=7
        )
        other = diffusion_posterior_sampling(
            self.problem, steps=20, n_samples=5, seed=8
        )
        self.assertEqual(first.samples, again.samples)
        self.assertNotEqual(first.samples, other.samples)
        ode = diffusion_posterior_sampling(
            self.problem, steps=20, n_samples=50, sampler="ode", seed=7
        )
        self.assertEqual(ode.sampler, "ode")
        # a deterministic ODE map turns Gaussian start samples into Gaussian ones
        self.assertEqual(len(ode.samples), 50)

    def test_standard_normal_initialization_adds_a_small_error(self) -> None:
        """Check the N(0, I) start differs from the exact marginal by O(alpha(T))."""
        a = diffusion_posterior_sampling(self.problem, steps=320, n_samples=0)
        b = diffusion_posterior_sampling(
            self.problem, steps=320, n_samples=0, initialization="standard_normal"
        )
        self.assertEqual(b.initialization, "standard_normal")
        self.assertLess(abs(a.mean_error - b.mean_error), 1e-2)
        self.assertEqual(a.samples, ())
        self.assertIsNone(a.sample_mean)
        self.assertIsNone(a.w1_per_coordinate)

    def test_invalid_arguments(self) -> None:
        """Check validation of the problem, schedule and options."""
        pb = self.problem
        with self.assertRaises(ValueError):
            diffusion_posterior_sampling(pb, steps=0)
        with self.assertRaises(ValueError):
            diffusion_posterior_sampling(pb, steps=5, sampler="leapfrog")
        with self.assertRaises(ValueError):
            diffusion_posterior_sampling(pb, steps=5, score="learned")
        with self.assertRaises(ValueError):
            diffusion_posterior_sampling(pb, steps=5, initialization="warm")
        with self.assertRaises(TypeError):
            diffusion_posterior_sampling("problem", steps=5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            diffusion_posterior_sampling(pb, steps=5, schedule=1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            diffusion_convergence_study(pb, steps=(10,))
        with self.assertRaises(ValueError):
            diffusion_convergence_study(pb, steps=(20, 10))
        with self.assertRaises(ValueError):
            VPSchedule(beta_min=0.0)
        with self.assertRaises(ValueError):
            VPSchedule(beta_min=2.0, beta_max=1.0)
        with self.assertRaises(ValueError):  # non-SPD prior covariance
            LinearGaussianDiffusionProblem(
                (0.0, 0.0), ((1.0, 2.0), (2.0, 1.0)), ((1.0, 0.0),), ((1.0,),), (0.0,)
            )
        with self.assertRaises(ValueError):  # shape mismatch
            LinearGaussianDiffusionProblem(
                (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), ((1.0,),), ((1.0,),), (0.0,)
            )


class DiffusionApproximationTests(unittest.TestCase):
    """The DPS-style score is approximate and its bias is measured."""

    def test_dps_bias_does_not_vanish_with_the_step_count(self) -> None:
        """Check the DPS-style error plateaus far above the exact-score error."""
        problem = _diffusion_problem()
        for sampler in ("sde", "ode"):
            exact = diffusion_convergence_study(
                problem, steps=(160, 640), sampler=sampler
            )
            approx = diffusion_convergence_study(
                problem, steps=(160, 640), sampler=sampler, score="dps_approximate"
            )
            self.assertTrue(approx.approximate)
            self.assertTrue(all(r.approximate for r in approx.results))
            self.assertIn("APPROXIMATE", approx.results[0].label)
            self.assertNotIn("APPROXIMATE", exact.results[0].label)
            coarse, fine = approx.results
            # plateau: refining the step changes the bias by under 10%
            self.assertLess(
                abs(fine.covariance_error - coarse.covariance_error),
                0.1 * fine.covariance_error,
            )
            self.assertGreater(fine.mean_error, 50.0 * exact.results[-1].mean_error)
            self.assertGreater(
                fine.covariance_error, 50.0 * exact.results[-1].covariance_error
            )
            self.assertGreater(max(fine.w2_per_coordinate), 0.05)

    def test_dps_samples_are_biased_relative_to_the_exact_posterior(self) -> None:
        """Check the sample-based W1 of the DPS-style sampler exceeds the exact one."""
        problem = _diffusion_problem()
        exact = diffusion_posterior_sampling(
            problem, steps=100, n_samples=300, seed=5, sampler="ode"
        )
        approx = diffusion_posterior_sampling(
            problem,
            steps=100,
            n_samples=300,
            seed=5,
            sampler="ode",
            score="dps_approximate",
        )
        assert exact.w1_per_coordinate is not None
        assert approx.w1_per_coordinate is not None
        self.assertGreater(
            max(approx.w1_per_coordinate), 3.0 * max(exact.w1_per_coordinate)
        )


class FlowMatchingComparisonTests(unittest.TestCase):
    """Flow matching in the bimodal comparison (sample-only)."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run a reduced comparison with the flow-matching row."""
        cls.config = BimodalConfig(
            n_train=200,
            n_validation=3,
            n_test=6,
            n_sbc=6,
            sbc_draws=7,
            eval_draws=40,
            abc_simulations=40,
            npe_hidden=(4,),
            npe_iterations=8,
            nre_hidden=(4,),
            nre_iterations=8,
            nre_grid_points=15,
            include_flow_matching=True,
        )
        cls.result = run_bimodal_comparison(cls.config)
        cls.by_name = {r.method: r for r in cls.result.reports}

    def test_default_comparison_has_no_flow_matching_row(self) -> None:
        """Check the default configuration is unchanged."""
        self.assertFalse(BimodalConfig().include_flow_matching)

    def test_row_is_present_labelled_and_sample_only(self) -> None:
        """Check the fm row, its labels, and the absence of any density."""
        self.assertEqual(
            [r.method for r in self.result.reports],
            ["exact", "abc", "prior", "npe", "nre", "fm"],
        )
        fm = self.by_name["fm"]
        self.assertTrue(fm.approximate)
        self.assertEqual(fm.status, "ok")
        self.assertEqual(self.result.training_status["fm"], "ok")
        self.assertIsNone(fm.test_mean_log_density)
        self.assertIsNone(fm.validation_mean_log_density)
        self.assertIsNone(fm.ratio_normalization)
        self.assertIn("no normalized density", fm.notes)
        self.assertIn("no evidence or Metropolis-Hastings", fm.notes)

    def test_metrics_are_measured_and_cost_is_training_only(self) -> None:
        """Check the sample metrics are finite and the cost accounting."""
        fm = self.by_name["fm"]
        cfg = self.config
        self.assertEqual(fm.training_simulations, cfg.n_train)
        self.assertEqual(fm.evaluation_simulations, 0)
        self.assertEqual(fm.simulations_per_posterior, 0)
        for value in (
            fm.w1_to_exact,
            fm.crps,
            fm.mode_mass_positive,
            fm.mode_mass_negative,
            fm.mode_mass_error,
            fm.headline_mean,
            fm.headline_sd,
        ):
            assert value is not None
            self.assertTrue(math.isfinite(value))
        assert fm.coverage90 is not None
        self.assertTrue(0.0 <= fm.coverage90 <= 1.0)
        assert fm.sbc_min_p_value is not None
        self.assertTrue(0.0 <= fm.sbc_min_p_value <= 1.0)

    def test_fm_trains_on_the_same_pairs_as_the_ratio_estimator(self) -> None:
        """Check the flow-matching training pairs equal the train split."""
        contract = bimodal_contract(0.25, 3.0)
        splits = make_splits(contract, train=12, validation=2, test=2, seed=9)
        prior, sim = contract.bind(None)
        thetas, observations = simulate_pairs(prior, sim, 12, splits.train_stream)
        self.assertEqual(
            list(zip(thetas, observations, strict=True)),
            list(zip(splits.train.parameters, splits.train.summaries, strict=True)),
        )
        trained = train_flow_matching(
            contract,
            12,
            hidden_layers=(3,),
            iterations=3,
            batch_size=8,
            seed=splits.train_stream,
        )
        self.assertEqual(trained.kind, "fm")
        self.assertEqual(trained.training_policy, "prior")
        self.assertEqual(trained.simulations, 12)

    def test_flow_matching_row_is_deterministic(self) -> None:
        """Check two tiny runs give an identical fm row (except wall-clock)."""
        tiny = BimodalConfig(
            n_train=30,
            n_validation=3,
            n_test=3,
            n_sbc=4,
            sbc_draws=5,
            eval_draws=10,
            abc_simulations=40,
            npe_hidden=(3,),
            npe_iterations=5,
            nre_hidden=(3,),
            nre_iterations=5,
            nre_grid_points=11,
            include_flow_matching=True,
            fm_hidden=(3,),
            fm_iterations=5,
            fm_steps=3,
        )
        rows = []
        for _ in range(2):
            table = run_bimodal_comparison(tiny).table()
            row = next(r for r in table if r["method"] == "fm")
            rows.append({k: v for k, v in row.items() if k != "train_seconds"})
        self.assertEqual(rows[0], rows[1])

    def test_invalid_flow_matching_settings(self) -> None:
        """Check validation of the flow-matching configuration."""
        with self.assertRaises(ValueError):
            BimodalConfig(fm_iterations=0)
        with self.assertRaises(ValueError):
            BimodalConfig(fm_steps=0)
        with self.assertRaises(ValueError):
            BimodalConfig(fm_method="rk45")


class TrainingPolicyTests(unittest.TestCase):
    """Only prior simulations are trained on; proposals are refused."""

    def test_prior_policy_is_accepted_and_recorded(self) -> None:
        """Check the prior policy and its provenance."""
        self.assertEqual(check_training_policy("prior"), "prior")
        trained = train_npe(
            _gauss_contract(),
            20,
            hidden_layers=(2,),
            iterations=3,
            batch_size=8,
            seed=1,
        )
        self.assertEqual(trained.training_policy, "prior")

    def test_proposal_policies_are_refused_before_any_simulation(self) -> None:
        """Check explicit refusal, with zero simulator calls, for every trainer."""
        contract = _gauss_contract()
        for policy in ("proposal", "sequential", "snpe", "truncated_proposal"):
            with self.assertRaises(ProposalTrainingRefused) as caught:
                check_training_policy(policy)
            self.assertIn("proposal", str(caught.exception))
            self.assertIsInstance(caught.exception, ValueError)
            for trainer in (train_npe, train_nre, train_flow_matching):
                ledger = WorkLedger()
                with self.assertRaises(ProposalTrainingRefused):
                    trainer(
                        contract,
                        20,
                        iterations=2,
                        seed=1,
                        ledger=ledger,
                        training_policy=policy,
                    )
                self.assertEqual(ledger.simulator_calls, 0)

    def test_unknown_or_non_string_policy_is_rejected(self) -> None:
        """Check validation of the policy value."""
        with self.assertRaises(ValueError):
            check_training_policy("bayesian")
        with self.assertRaises(TypeError):
            check_training_policy(None)  # type: ignore[arg-type]
        self.assertNotIsInstance(ValueError("x"), ProposalTrainingRefused)


if __name__ == "__main__":
    unittest.main()
