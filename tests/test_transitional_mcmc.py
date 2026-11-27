"""Tests for transitional MCMC, model class selection and hierarchical calibration.

Oracles are closed forms: the conjugate Gaussian model (evidence, posterior
moments, relative entropy), a bimodal target with known mass, nested linear
models whose evidences follow from ``y ~ N(0, sigma^2 I + tau^2 X X')``
evaluated with the dense routines of :mod:`probstat.linalg`, and a
linear-Gaussian hierarchy with a closed-form evidence and posterior.
Stochastic results use fixed seeds and bands of several seed-to-seed standard
deviations measured for these exact settings (recorded next to each band);
no pinned digits.
"""

from __future__ import annotations

import math
import os
import random
import statistics
import unittest

from probstat.inference import sequential_monte_carlo
from probstat.inference.transitional_mcmc import (
    STOP_REASONS,
    TMCMCResult,
    _coefficient_of_variation,
    _next_exponent,
    decompose_log_evidence,
    hierarchical_calibration,
    model_averaged_predictive,
    model_class_selection,
    transitional_mcmc,
)
from probstat.linalg import log_det_spd, solve_spd

LOG_2PI = math.log(2.0 * math.pi)

# Conjugate model: theta_k ~ N(0, 1), y_kj ~ N(theta_k, NOISE^2), k = 0, 1.
DATA = ((0.8, 1.1, 0.5, 1.3), (-0.4, -0.9, -0.2, -0.6))
NOISE = 0.5


def conjugate_log_likelihood(x: tuple[float, ...]) -> float:
    """Return the fully normalized Gaussian log-likelihood of ``DATA``."""
    return math.fsum(
        -0.5 * ((y - x[k]) / NOISE) ** 2 - math.log(NOISE) - 0.5 * LOG_2PI
        for k in range(2)
        for y in DATA[k]
    )


def standard_normal_log_prior(x: tuple[float, ...]) -> float:
    """Return the log-density of ``N(0, I)``."""
    return math.fsum(-0.5 * v * v - 0.5 * LOG_2PI for v in x)


def standard_normal_sampler(generator: random.Random) -> tuple[float, ...]:
    """Draw from ``N(0, I_2)``."""
    return (generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))


def conjugate_exact() -> tuple[float, tuple[float, float], float, float]:
    """Return exact ``(log Z, posterior means, posterior variance, KL, fit)``.

    Per coordinate with ``n`` observations ``y_j ~ N(theta, s^2)``, prior
    ``N(0, 1)``: posterior precision ``1 + n/s^2``, mean ``(n ybar/s^2)/prec``.
    """
    total = 0.0
    means = []
    kl = 0.0
    fit = 0.0
    for observations in DATA:
        n = len(observations)
        ybar = sum(observations) / n
        ss = sum((v - ybar) ** 2 for v in observations)
        total += (
            -0.5 * n * math.log(2 * math.pi * NOISE**2)
            - 0.5 * math.log(1 + n / NOISE**2)
            - ss / (2 * NOISE**2)
            - 0.5 * n * ybar**2 / (NOISE**2 + n)
        )
        precision = 1.0 + n / NOISE**2
        mean = (n * ybar / NOISE**2) / precision
        variance = 1.0 / precision
        means.append(mean)
        # KL(N(mean, var) || N(0, 1)) and E_post[log L]
        kl += 0.5 * (variance + mean**2 - 1.0 - math.log(variance))
        fit += math.fsum(
            -0.5 * ((y - mean) ** 2 + variance) / NOISE**2
            - math.log(NOISE)
            - 0.5 * LOG_2PI
            for y in observations
        )
    variance = 1.0 / (1.0 + 4 / NOISE**2)
    return total, (means[0], means[1]), variance, kl, fit


def run_conjugate(seed: int, particles: int = 1000) -> TMCMCResult:
    """Run TMCMC on the conjugate model."""
    return transitional_mcmc(
        standard_normal_sampler,
        standard_normal_log_prior,
        conjugate_log_likelihood,
        particles=particles,
        seed=seed,
    )


def nested_models() -> tuple[tuple[float, ...], dict[str, list[list[float]]]]:
    """Return data ``y`` and design matrices of three nested polynomial models."""
    xs = [-1.0 + 2.0 * i / 11 for i in range(12)]
    noise = (0.3, -0.5, 0.2, 0.6, -0.4, -0.1, 0.5, -0.3, 0.1, -0.6, 0.4, -0.2)
    y = tuple(1.0 + 0.8 * x + e for x, e in zip(xs, noise, strict=True))
    return y, {
        "M0": [[1.0] for _ in xs],
        "M1": [[1.0, x] for x in xs],
        "M2": [[1.0, x, x * x] for x in xs],
    }


NESTED_SIGMA = 0.5
NESTED_TAU = 2.0


def linear_gaussian_log_evidence(
    y: tuple[float, ...], design: list[list[float]], sigma: float, tau: float
) -> float:
    """Return ``log N(y; 0, sigma^2 I + tau^2 X X')`` by dense Cholesky."""
    n = len(y)
    cov = tuple(
        tuple(
            (sigma**2 if i == j else 0.0)
            + tau**2
            * math.fsum(a * b for a, b in zip(design[i], design[j], strict=True))
            for j in range(n)
        )
        for i in range(n)
    )
    solved = solve_spd(cov, y)
    quad = math.fsum(a * b for a, b in zip(y, solved, strict=True))
    return -0.5 * (n * LOG_2PI + log_det_spd(cov) + quad)


def run_nested(name: str, design: list[list[float]], seed: int) -> TMCMCResult:
    """Run TMCMC for one nested linear model with ``beta ~ N(0, tau^2 I)``."""
    y, _ = nested_models()
    p = len(design[0])

    def log_likelihood(beta: tuple[float, ...]) -> float:
        """Return the normalized Gaussian log-likelihood."""
        residual = math.fsum(
            (yi - math.fsum(a * b for a, b in zip(row, beta, strict=True))) ** 2
            for yi, row in zip(y, design, strict=True)
        )
        return -0.5 * residual / NESTED_SIGMA**2 - len(y) * (
            math.log(NESTED_SIGMA) + 0.5 * LOG_2PI
        )

    return transitional_mcmc(
        lambda g: tuple(g.gauss(0.0, NESTED_TAU) for _ in range(p)),
        lambda b: math.fsum(
            -0.5 * (v / NESTED_TAU) ** 2 - math.log(NESTED_TAU) - 0.5 * LOG_2PI
            for v in b
        ),
        log_likelihood,
        particles=1000,
        seed=seed,
    )


_NESTED_RUNS: dict[str, TMCMCResult] = {}


def nested_runs() -> dict[str, TMCMCResult]:
    """Return (and cache) one TMCMC run per nested model, seeds 7, 8, 9."""
    if not _NESTED_RUNS:
        for i, (name, design) in enumerate(nested_models()[1].items()):
            _NESTED_RUNS[name] = run_nested(name, design, seed=7 + i)
    return _NESTED_RUNS


class ConjugateGaussianTest(unittest.TestCase):
    """Exact evidence, posterior moments and entropy split."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run five seeds once."""
        cls.runs = [run_conjugate(seed) for seed in range(5)]
        cls.exact = conjugate_exact()

    def test_runs_converge_with_documented_status(self) -> None:
        """Every run converges and reports consistent per-stage records."""
        for run in self.runs:
            self.assertTrue(run.converged)
            self.assertEqual(run.stop_reason, "converged")
            self.assertIn(run.stop_reason, STOP_REASONS)
            self.assertEqual(run.exponents[0], 0.0)
            self.assertEqual(run.exponents[-1], 1.0)
            self.assertTrue(
                all(
                    a < b
                    for a, b in zip(run.exponents, run.exponents[1:], strict=False)
                )
            )
            self.assertEqual(len(run.acceptance_rates), run.stages)
            self.assertEqual(len(run.log_evidence_increments), run.stages)
            self.assertEqual(len(run.particles), 1000)
            # all stages but the last meet the coefficient-of-variation target
            for cov in run.coefficients_of_variation[:-1]:
                self.assertAlmostEqual(cov, 1.0, places=6)
            self.assertLessEqual(run.coefficients_of_variation[-1], 1.0 + 1e-9)
            self.assertAlmostEqual(
                run.log_evidence, math.fsum(run.log_evidence_increments), places=12
            )

    def test_evidence_matches_closed_form(self) -> None:
        """Mean over seeds is within 0.3 nats of the exact log evidence.

        Measured seed-to-seed standard deviation at N = 1000 is about 0.1
        (see the recorded run in the report), so the mean of five runs has a
        standard error near 0.05 and 0.3 is about six of them.
        """
        values = [run.log_evidence for run in self.runs]
        self.assertAlmostEqual(statistics.mean(values), self.exact[0], delta=0.3)
        for value in values:
            self.assertAlmostEqual(value, self.exact[0], delta=0.5)

    def test_posterior_moments(self) -> None:
        """Posterior means and variance within bands of the Monte Carlo error.

        The exact posterior sd is 0.17. Measured over five seeds the mean
        errors are at most 0.025, the variance ratios lie in 0.90 to 1.06 and
        the off-diagonal covariance is below 0.06 of the variance; the bands
        (0.05, 20 percent, 20 percent) are about twice those.
        """
        _, means, variance, _, _ = self.exact
        for run in self.runs:
            mean = run.mean()
            cov = run.covariance()
            for k in range(2):
                self.assertAlmostEqual(mean[k], means[k], delta=0.05)
                self.assertAlmostEqual(cov[k][k], variance, delta=0.2 * variance)
            self.assertAlmostEqual(cov[0][1], 0.0, delta=0.2 * variance)

    def test_information_theoretic_split(self) -> None:
        """Data fit minus relative entropy recovers log Z; both parts are exact.

        The split ``log Z = E_post[log L] - KL`` is an identity, so the
        estimated gain is ``fit - log Z_hat`` by construction; it is compared
        with the exact Gaussian relative entropy.
        """
        _, _, _, kl, fit = self.exact
        fits = []
        gains = []
        for run in self.runs:
            parts = decompose_log_evidence(run)
            self.assertAlmostEqual(
                parts.data_fit - parts.information_gain, parts.log_evidence, places=12
            )
            self.assertGreater(parts.data_fit_standard_error, 0.0)
            fits.append(parts.data_fit)
            gains.append(parts.information_gain)
        self.assertAlmostEqual(statistics.mean(fits), fit, delta=0.25)
        self.assertAlmostEqual(statistics.mean(gains), kl, delta=0.3)
        self.assertGreater(kl, 0.0)

    def test_deterministic_for_equal_seed(self) -> None:
        """Equal seeds give identical results, different seeds do not."""
        again = run_conjugate(0)
        self.assertEqual(again.particles, self.runs[0].particles)
        self.assertEqual(again.log_evidence, self.runs[0].log_evidence)
        self.assertNotEqual(self.runs[1].log_evidence, self.runs[0].log_evidence)


class SMCComparisonTest(unittest.TestCase):
    """TMCMC and tempered SMC estimate the same evidence."""

    def test_agrees_with_smc_on_same_target(self) -> None:
        """Four-seed-averaged TMCMC and SMC log evidences agree within 0.3 nats.

        Measured seed-to-seed standard deviations at N = 1000 are 0.11 (TMCMC)
        and 0.07 (SMC), so the difference of four-seed means has standard
        error near 0.065.
        """
        exact = conjugate_exact()[0]
        tm = []
        sm = []
        for seed in range(4):
            tm.append(run_conjugate(seed).log_evidence)
            sm.append(
                sequential_monte_carlo(
                    standard_normal_sampler,
                    standard_normal_log_prior,
                    conjugate_log_likelihood,
                    particles=1000,
                    seed=seed,
                ).log_evidence
            )
        self.assertAlmostEqual(statistics.mean(tm), statistics.mean(sm), delta=0.3)
        self.assertAlmostEqual(statistics.mean(sm), exact, delta=0.3)


class BimodalTest(unittest.TestCase):
    """Both modes of a well separated mixture are occupied."""

    def test_both_modes_and_evidence(self) -> None:
        """Mixture 0.5 N((-3, 0), 0.5^2 I) + 0.5 N((3, 0), 0.5^2 I), U(-6, 6)^2.

        The likelihood is a normalized density that puts all but ``1e-30``
        of its mass inside the box, so ``Z = 1/144`` and each mode carries
        posterior mass one half. With 1000 samples the mode fraction is
        accepted in ``[0.35, 0.65]``.
        """
        sd = 0.5

        def mixture(x: tuple[float, ...]) -> float:
            """Return the log of the equal-weight bimodal density."""
            terms = []
            for centre in (-3.0, 3.0):
                q = ((x[0] - centre) ** 2 + x[1] ** 2) / sd**2
                terms.append(-0.5 * q - math.log(2 * math.pi * sd**2))
            top = max(terms)
            return top + math.log(0.5 * math.fsum(math.exp(t - top) for t in terms))

        def log_prior(x: tuple[float, ...]) -> float:
            """Return the uniform log-density on the box."""
            inside = all(-6.0 <= v <= 6.0 for v in x)
            return -2.0 * math.log(12.0) if inside else -math.inf

        results = [
            transitional_mcmc(
                lambda g: (g.uniform(-6, 6), g.uniform(-6, 6)),
                log_prior,
                mixture,
                particles=1200,
                seed=seed,
            )
            for seed in range(3)
        ]
        for run in results:
            self.assertTrue(run.converged)
            right = sum(1 for p in run.particles if p[0] > 0.0) / len(run.particles)
            self.assertGreater(right, 0.35)
            self.assertLess(right, 0.65)
            self.assertAlmostEqual(run.log_evidence, -2.0 * math.log(12.0), delta=0.5)
        self.assertAlmostEqual(
            statistics.mean(r.log_evidence for r in results),
            -2.0 * math.log(12.0),
            delta=0.3,
        )


class ModelSelectionTest(unittest.TestCase):
    """Posterior model probabilities from known evidences."""

    def test_probabilities_from_log_evidences(self) -> None:
        """Closed-form softmax, Bayes factors and non-uniform priors."""
        selection = model_class_selection({"a": -10.0, "b": -11.0, "c": -13.0})
        weights = [math.exp(v) for v in (-10.0, -11.0, -13.0)]
        for got, w in zip(selection.posterior_probabilities, weights, strict=True):
            self.assertAlmostEqual(got, w / sum(weights), places=14)
        self.assertAlmostEqual(selection.bayes_factor("a", "b"), math.e, places=12)
        self.assertAlmostEqual(selection.log_bayes_factor("c", "a"), -3.0, places=14)
        self.assertEqual(selection.best, "a")
        skewed = model_class_selection(
            {"a": -10.0, "b": -11.0}, prior_probabilities={"a": 0.1, "b": 0.9}
        )
        odds = (0.1 / 0.9) * math.e
        self.assertAlmostEqual(
            skewed.posterior_probabilities[0], odds / (1 + odds), places=14
        )
        zero = model_class_selection({"a": -1.0, "b": -2.0}, [1.0, 0.0])
        self.assertEqual(zero.posterior_probabilities, (1.0, 0.0))

    def test_no_overflow_for_large_evidence_gaps(self) -> None:
        """Evidences hundreds of nats apart still give probabilities."""
        selection = model_class_selection({"a": -1000.0, "b": -1800.0})
        self.assertEqual(selection.posterior_probabilities[0], 1.0)
        self.assertEqual(selection.bayes_factor("a", "b"), math.inf)
        impossible = model_class_selection({"a": -math.inf, "b": -3.0})
        self.assertEqual(impossible.posterior_probabilities, (0.0, 1.0))

    def test_nested_linear_models_against_exact_evidence(self) -> None:
        """TMCMC evidences and model probabilities of three nested models.

        Exact evidences come from ``N(0, sigma^2 I + tau^2 X X')``. The data
        are linear (generated by ``1 + 0.8 x`` plus fixed noise) so the
        linear model should win and the quadratic model pay the Ockham
        penalty. Measured over 5 seeds the error standard deviation is 0.10
        (M0), 0.09 (M1) and 0.14 to 0.41 (M2, two seed sets; one run was
        off by 0.67), so the band is 0.8 nats on a single run and 0.15 on
        probabilities.
        """
        y, designs = nested_models()
        exact = {
            name: linear_gaussian_log_evidence(y, d, NESTED_SIGMA, NESTED_TAU)
            for name, d in designs.items()
        }
        runs = nested_runs()
        for name, run in runs.items():
            self.assertTrue(run.converged, name)
            self.assertAlmostEqual(run.log_evidence, exact[name], delta=0.8)
        estimated = model_class_selection(runs)
        reference = model_class_selection(exact)
        self.assertEqual(estimated.best, "M1")
        self.assertEqual(reference.best, "M1")
        for got, want in zip(
            estimated.posterior_probabilities,
            reference.posterior_probabilities,
            strict=True,
        ):
            self.assertAlmostEqual(got, want, delta=0.15)
        # the information gain grows with the number of parameters
        gains = [decompose_log_evidence(runs[n]).information_gain for n in exact]
        self.assertLess(gains[0], gains[1])
        self.assertLess(gains[1], gains[2])

    def test_model_averaged_predictive_mixes_by_probability(self) -> None:
        """Mixture frequencies follow the posterior model probabilities."""
        y, designs = nested_models()
        full = nested_runs()
        runs = {"M0": full["M0"], "M1": full["M1"]}
        selection = model_class_selection(runs)
        predictors = {
            "M0": lambda theta, g: (theta[0],),
            "M1": lambda theta, g: (theta[0] + theta[1] * 1.0,),
        }
        mixed = model_averaged_predictive(runs, predictors, draws=4000, seed=11)
        frequencies = mixed.model_frequencies()
        for name, p in zip(
            selection.names, selection.posterior_probabilities, strict=True
        ):
            # binomial standard error at most 0.5 / sqrt(4000) = 0.008
            self.assertAlmostEqual(frequencies[name], p, delta=0.04)
        # mixture mean is the probability-weighted mean of the model means
        by_model = {
            name: statistics.mean(
                d[0]
                for d, m in zip(mixed.draws, mixed.model_names, strict=True)
                if m == name
            )
            for name in runs
            if frequencies[name] > 0.0
        }
        weighted = math.fsum(frequencies[n] * by_model[n] for n in by_model)
        self.assertAlmostEqual(mixed.mean()[0], weighted, places=12)
        again = model_averaged_predictive(runs, predictors, draws=4000, seed=11)
        self.assertEqual(again.draws, mixed.draws)
        self.assertEqual(len(y), 12)

    def test_invalid_model_selection_input(self) -> None:
        """Bad evidences, priors and unconverged runs are rejected."""
        with self.assertRaises(ValueError):
            model_class_selection({})
        with self.assertRaises(TypeError):
            model_class_selection([-1.0, -2.0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model_class_selection({"a": math.nan})
        with self.assertRaises(ValueError):
            model_class_selection({"a": math.inf})
        with self.assertRaises(ValueError):
            model_class_selection({"a": -1.0, "b": -2.0}, [0.5, 0.6])
        with self.assertRaises(ValueError):
            model_class_selection({"a": -1.0, "b": -2.0}, {"a": 1.0})
        with self.assertRaises(ValueError):
            model_class_selection({"a": -math.inf})
        failed = transitional_mcmc(
            lambda g: (g.gauss(0, 1),),
            lambda x: 0.0,
            lambda x: -math.inf,
            particles=20,
            seed=1,
        )
        with self.assertRaises(ValueError):
            model_class_selection({"a": failed})


# Hierarchy: 3 tests, 4 observations each, y_ij ~ N(theta_i, S^2),
# theta_i ~ N(mu, TAU^2) with TAU known, mu ~ N(0, 2^2).
HIER_DATA = ((0.9, 1.4, 0.6, 1.1), (-0.3, 0.2, -0.5, 0.1), (1.8, 2.1, 1.5, 2.4))
HIER_S = 0.5
HIER_TAU = 0.7
HIER_MU_STD = 2.0


def hierarchy_exact() -> tuple[float, float, float, list[float], list[float]]:
    """Return exact ``(log Z, E mu, Var mu, E theta_i, Var theta_i)``.

    The evidence is ``log N(y; 0, S^2 I + TAU^2 B + MU_STD^2 1 1')`` with
    ``B`` the block-diagonal all-ones matrix of the tests, computed densely.
    Given ``mu`` the sample means satisfy
    ``ybar_i ~ N(mu, TAU^2 + S^2 / n)``, which gives the posterior of ``mu``
    and, through ``theta_i | mu, y``, those of ``theta_i``.
    """
    flat = [v for block in HIER_DATA for v in block]
    n = len(flat)
    owner = [i for i, block in enumerate(HIER_DATA) for _ in block]
    cov = tuple(
        tuple(
            (HIER_S**2 if a == b else 0.0)
            + (HIER_TAU**2 if owner[a] == owner[b] else 0.0)
            + HIER_MU_STD**2
            for b in range(n)
        )
        for a in range(n)
    )
    solved = solve_spd(cov, tuple(flat))
    quad = math.fsum(u * v for u, v in zip(flat, solved, strict=True))
    log_z = -0.5 * (n * LOG_2PI + log_det_spd(cov) + quad)
    nobs = len(HIER_DATA[0])
    ybars = [sum(block) / nobs for block in HIER_DATA]
    marginal = HIER_TAU**2 + HIER_S**2 / nobs
    precision = 1.0 / HIER_MU_STD**2 + len(ybars) / marginal
    mu_var = 1.0 / precision
    mu_mean = (sum(ybars) / marginal) * mu_var
    weight = (nobs / HIER_S**2) / (nobs / HIER_S**2 + 1.0 / HIER_TAU**2)
    cond_var = 1.0 / (nobs / HIER_S**2 + 1.0 / HIER_TAU**2)
    theta_mean = [weight * yb + (1 - weight) * mu_mean for yb in ybars]
    theta_var = [cond_var + (1 - weight) ** 2 * mu_var for _ in ybars]
    return log_z, mu_mean, mu_var, theta_mean, theta_var


def hierarchy_likelihoods() -> list:  # type: ignore[type-arg]
    """Return one normalized Gaussian log-likelihood per test."""

    def make(block: tuple[float, ...]):  # type: ignore[no-untyped-def]
        """Return the log-likelihood of one test's data."""
        return lambda theta: math.fsum(
            -0.5 * ((y - theta[0]) / HIER_S) ** 2 - math.log(HIER_S) - 0.5 * LOG_2PI
            for y in block
        )

    return [make(block) for block in HIER_DATA]


class HierarchicalTest(unittest.TestCase):
    """Linear-Gaussian hierarchy with closed-form evidence and posterior."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run three seeds once."""
        cls.exact = hierarchy_exact()
        cls.runs = [
            hierarchical_calibration(
                hierarchy_likelihoods(),
                mu_mean=0.0,
                mu_std=HIER_MU_STD,
                fixed_scale=HIER_TAU,
                particles=1200,
                seed=seed,
            )
            for seed in range(3)
        ]

    def test_evidence_and_posterior_moments(self) -> None:
        """Evidence within 0.4 nats and moments within Monte Carlo bands.

        Measured over four seeds at 1500 samples (this class runs three seeds
        at 1200): evidence errors at most
        0.14, mean errors at most 0.025, variance ratios 0.90 to 1.11.
        """
        log_z, mu_mean, mu_var, theta_mean, theta_var = self.exact
        values = []
        for run in self.runs:
            self.assertTrue(run.tmcmc.converged)
            values.append(run.log_evidence)
            mu = [d[0] for d in run.hyper_mean_draws()]
            self.assertAlmostEqual(statistics.mean(mu), mu_mean, delta=0.06)
            self.assertAlmostEqual(
                statistics.pvariance(mu), mu_var, delta=0.25 * mu_var
            )
            for i in range(3):
                draws = [d[0] for d in run.test_draws(i)]
                self.assertAlmostEqual(
                    statistics.mean(draws), theta_mean[i], delta=0.06
                )
                self.assertAlmostEqual(
                    statistics.pvariance(draws), theta_var[i], delta=0.25 * theta_var[i]
                )
        self.assertAlmostEqual(statistics.mean(values), log_z, delta=0.4)

    def test_shrinkage_toward_population_mean(self) -> None:
        """Each per-test posterior mean lies between its data mean and mu."""
        run = self.runs[0]
        mu = statistics.mean(d[0] for d in run.hyper_mean_draws())
        for i, block in enumerate(HIER_DATA):
            ybar = sum(block) / len(block)
            theta = statistics.mean(d[0] for d in run.test_draws(i))
            self.assertLess(abs(theta - mu), abs(ybar - mu) + 0.05)

    def test_sampled_scale_and_new_test(self) -> None:
        """With a log-normal scale hyperprior the scale is positive and finite."""
        run = hierarchical_calibration(
            hierarchy_likelihoods(),
            mu_mean=0.0,
            mu_std=HIER_MU_STD,
            log_scale_mean=math.log(0.7),
            log_scale_std=0.5,
            particles=600,
            seed=5,
        )
        self.assertTrue(run.tmcmc.converged)
        self.assertEqual(run.tmcmc.dimension, 1 + 1 + 3)
        scales = [s[0] for s in run.hyper_scale_draws()]
        self.assertTrue(all(0.0 < s < math.inf for s in scales))
        new = run.new_test_draws(50, seed=3)
        self.assertEqual(len(new), 50)
        self.assertEqual(new, run.new_test_draws(50, seed=3))

    def test_invalid_hierarchy_input(self) -> None:
        """Ambiguous scale specification and bad hyperparameters are rejected."""
        likes = hierarchy_likelihoods()
        with self.assertRaises(ValueError):
            hierarchical_calibration([], mu_mean=0.0, mu_std=1.0, fixed_scale=1.0)
        with self.assertRaises(ValueError):
            hierarchical_calibration(likes, mu_mean=0.0, mu_std=1.0)
        with self.assertRaises(ValueError):
            hierarchical_calibration(
                likes, mu_mean=0.0, mu_std=1.0, fixed_scale=1.0, log_scale_mean=0.0
            )
        with self.assertRaises(ValueError):
            hierarchical_calibration(likes, mu_mean=0.0, mu_std=0.0, fixed_scale=1.0)
        with self.assertRaises(ValueError):
            hierarchical_calibration(likes, mu_mean=0.0, mu_std=1.0, fixed_scale=-1.0)
        with self.assertRaises(ValueError):
            hierarchical_calibration(
                likes, mu_mean=[0.0, 1.0], mu_std=1.0, fixed_scale=1.0
            )
        with self.assertRaises(TypeError):
            hierarchical_calibration(
                [1.0],
                mu_mean=0.0,
                mu_std=1.0,
                fixed_scale=1.0,  # type: ignore[list-item]
            )


class ExponentTest(unittest.TestCase):
    """The coefficient-of-variation bisection."""

    def test_hits_target_and_is_monotone(self) -> None:
        """The returned exponent attains the target coefficient of variation."""
        values = [-0.5 * ((i - 50) / 10.0) ** 2 * 6.0 for i in range(100)]
        nxt = _next_exponent(values, 0.0, 1.0)
        assert nxt is not None
        self.assertTrue(0.0 < nxt < 1.0)
        self.assertAlmostEqual(_coefficient_of_variation(values, nxt), 1.0, places=9)
        increments = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
        covs = [_coefficient_of_variation(values, d) for d in increments]
        self.assertEqual(covs, sorted(covs))

    def test_full_step_and_degenerate_cases(self) -> None:
        """Flat likelihood jumps to one; invalid input raises; stall returns None."""
        self.assertEqual(_next_exponent([1.0, 1.0, 1.0], 0.2, 1.0), 1.0)
        with self.assertRaises(ValueError):
            _next_exponent([0.0, -math.inf], 0.0, 1.0)
        with self.assertRaises(ValueError):
            _next_exponent([0.0, 1.0], 1.0, 1.0)
        with self.assertRaises(ValueError):
            _next_exponent([0.0, 1.0], 0.0, 0.0)
        # a spread so large that only increments below double precision qualify
        self.assertIsNone(_next_exponent([0.0, 1.0e300, 0.0, 0.0], 0.0, 1.0e-300))


class FailureTest(unittest.TestCase):
    """Failures are reported in the result, invalid input raises."""

    @staticmethod
    def sampler(generator: random.Random) -> tuple[float, ...]:
        """Draw from ``N(0, 1)``."""
        return (generator.gauss(0.0, 1.0),)

    def test_all_zero_likelihood(self) -> None:
        """A likelihood that vanishes everywhere stops with a status."""
        run = transitional_mcmc(
            self.sampler, lambda x: 0.0, lambda x: -math.inf, particles=30, seed=1
        )
        self.assertFalse(run.converged)
        self.assertEqual(run.stop_reason, "all_zero_likelihood")
        self.assertTrue(math.isnan(run.log_evidence))
        self.assertEqual(run.stages, 0)
        with self.assertRaises(ValueError):
            run.mean()
        with self.assertRaises(ValueError):
            decompose_log_evidence(run)

    def test_single_positive_likelihood(self) -> None:
        """Exactly one sample with positive likelihood is also a stop."""
        run = transitional_mcmc(
            self.sampler,
            lambda x: 0.0,
            lambda x: 0.0 if x[0] > 2.5 else -math.inf,
            particles=3,
            seed=2,
        )
        # at most one of three N(0, 1) draws exceeds 2.5 for this seed
        self.assertIn(
            run.stop_reason, ("single_positive_likelihood", "all_zero_likelihood")
        )
        self.assertFalse(run.converged)

    def test_singular_covariance(self) -> None:
        """Two identical prior draws give an exactly zero covariance; no jitter."""
        run = transitional_mcmc(
            lambda g: (0.5,),
            lambda x: 0.0,
            lambda x: -x[0] * x[0],
            particles=2,
            seed=1,
        )
        self.assertFalse(run.converged)
        self.assertEqual(run.stop_reason, "singular_covariance")
        self.assertTrue(math.isnan(run.log_evidence))

    def test_max_stages_status(self) -> None:
        """A stage limit of one stops a run that needs several."""
        run = transitional_mcmc(
            standard_normal_sampler,
            standard_normal_log_prior,
            conjugate_log_likelihood,
            particles=200,
            max_stages=1,
            seed=1,
        )
        self.assertFalse(run.converged)
        self.assertEqual(run.stop_reason, "max_stages")
        self.assertEqual(run.stages, 1)
        self.assertLess(run.exponents[-1], 1.0)

    def test_invalid_arguments_raise(self) -> None:
        """Wrong types and out-of-range options raise."""
        args = (self.sampler, lambda x: 0.0, lambda x: -(x[0] ** 2))
        with self.assertRaises(TypeError):
            transitional_mcmc(1.0, args[1], args[2])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            transitional_mcmc(*args, particles=1)
        with self.assertRaises(ValueError):
            transitional_mcmc(*args, cov_target=0.0)
        with self.assertRaises(ValueError):
            transitional_mcmc(*args, thinning=0)
        with self.assertRaises(ValueError):
            transitional_mcmc(*args, target_acceptance=1.0)
        with self.assertRaises(ValueError):
            transitional_mcmc(*args, initial_scale=-1.0)
        with self.assertRaises(ValueError):
            transitional_mcmc(*args, max_stages=0)
        with self.assertRaises(ValueError):
            transitional_mcmc(*args, names=("a", "b"))
        with self.assertRaises(TypeError):
            transitional_mcmc(*args, seed="x")  # type: ignore[arg-type]

    def test_bad_likelihood_values_raise(self) -> None:
        """NaN and +inf from the likelihood are programming errors."""
        for bad in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                transitional_mcmc(
                    self.sampler,
                    lambda x: 0.0,
                    lambda x, b=bad: b,
                    particles=10,
                    seed=1,
                )

    def test_inconsistent_prior_sampler_raises(self) -> None:
        """A prior draw outside the support of ``log_prior`` is rejected."""
        with self.assertRaises(ValueError):
            transitional_mcmc(
                self.sampler,
                lambda x: -math.inf,
                lambda x: 0.0,
                particles=10,
                seed=1,
            )


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowReplicationTest(unittest.TestCase):
    """Calibration of the evidence estimator over many seeds."""

    def test_evidence_bias_and_spread(self) -> None:
        """Forty replicates: the bias of ``log Z_hat`` is small against its sd.

        A first run measured a bias of -0.052 against a standard error of
        0.015, i.e. a small but statistically visible negative bias (adaptive
        exponents and chain correlation; Jensen alone gives only about
        ``-sd^2 / 2 = -0.005``). The assertion therefore bounds the bias
        by one standard deviation of a single run instead of by its standard
        error, and bounds the spread.
        """
        exact = conjugate_exact()[0]
        values = [run_conjugate(1000 + s).log_evidence for s in range(40)]
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        print(
            f"slow: mean={mean:.4f} sd={sd:.4f} exact={exact:.4f} "
            f"bias={mean - exact:.4f} se={sd / math.sqrt(len(values)):.4f}"
        )
        self.assertLess(abs(mean - exact), sd)
        self.assertGreater(sd, 0.03)
        self.assertLess(sd, 0.25)


if __name__ == "__main__":
    unittest.main()
