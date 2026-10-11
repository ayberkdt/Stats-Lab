"""Tests for hierarchical normal / beta-binomial models and predictive checks."""

from __future__ import annotations

import math
import random
import statistics
import unittest

from probstat.inference import (
    fit_beta_binomial,
    half_cauchy_log_prior,
    hierarchical_normal_posterior,
    non_centered_log_density,
    pit_uniformity,
    pit_values,
    posterior_predictive_pvalues,
    sample_hierarchical_normal,
    shrinkage_factors,
)
from probstat.multivariate import MultivariateNormal
from probstat.optimization import central_gradient

SCHOOLS_Y = (28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0)
SCHOOLS_S = (15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)


def sample_variance(values: tuple[float, ...]) -> float:
    """Return the unbiased sample variance."""
    return statistics.variance(values)


class HierarchicalNormalGridTests(unittest.TestCase):
    """Exact marginal posterior of tau computed on the adaptive grid."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compute the eight-schools grid posterior once."""
        cls.posterior = hierarchical_normal_posterior(SCHOOLS_Y, SCHOOLS_S)

    def test_quadrature_is_normalized_and_matches_its_own_draws(self) -> None:
        """Weights sum to one and exact draws reproduce quadrature moments."""
        post = self.posterior
        self.assertAlmostEqual(math.fsum(post.weights), 1.0, places=12)
        self.assertEqual(post.tau_grid[0], 0.0)
        draws = post.sample(20_000, seed=1)
        mu_mean, mu_sd = post.mu_moments()
        mu_draws = [d[0] for d in draws]
        tau_draws = [d[1] for d in draws]
        self.assertAlmostEqual(
            statistics.fmean(mu_draws), mu_mean, delta=4 * mu_sd / math.sqrt(20_000)
        )
        self.assertAlmostEqual(
            statistics.fmean(tau_draws),
            post.tau_mean(),
            delta=4 * post.tau_sd() / math.sqrt(20_000),
        )
        tau_draws.sort()
        self.assertAlmostEqual(tau_draws[10_000], post.tau_quantile(0.5), delta=0.25)

    def test_marginal_matches_multivariate_normal_evidence(self) -> None:
        """With a normal prior on mu, p(tau|y) equals the closed-form evidence.

        Marginally ``y ~ N(m0 1, diag(s^2 + tau^2) + s0^2 1 1')``, an
        independent route to [EQ-HIER-002].
        """
        m0, s0 = 2.0, 7.0
        post = hierarchical_normal_posterior(
            SCHOOLS_Y, SCHOOLS_S, mu_prior=(m0, s0), grid_size=201
        )
        differences = []
        for tau, log_density in zip(
            post.tau_grid[::20], post.log_density[::20], strict=True
        ):
            covariance = [
                [
                    (SCHOOLS_S[i] ** 2 + tau * tau if i == j else 0.0) + s0 * s0
                    for j in range(8)
                ]
                for i in range(8)
            ]
            law = MultivariateNormal([m0] * 8, covariance)
            differences.append(log_density - law.log_pdf(SCHOOLS_Y))
        self.assertLess(max(differences) - min(differences), 1e-9)

    def test_shrinkage_and_partial_pooling(self) -> None:
        """Shrinkage grows with s_j and posterior means move toward E[mu|y]."""
        post = self.posterior
        factors = shrinkage_factors(SCHOOLS_S, 10.0)
        self.assertAlmostEqual(factors[0], 225.0 / 325.0)
        expected = post.expected_shrinkage()
        order = sorted(range(8), key=lambda j: SCHOOLS_S[j])
        self.assertEqual(
            [expected[j] for j in order], sorted(expected[j] for j in order)
        )
        means, sds = post.theta_moments()
        mu_mean = post.mu_moments()[0]
        for y, mean, sd, s in zip(SCHOOLS_Y, means, sds, SCHOOLS_S, strict=True):
            self.assertLess(abs(mean - mu_mean), abs(y - mu_mean))
            self.assertLess(sd, s)

    def test_prior_choice_and_impropriety(self) -> None:
        """A half-Cauchy prior lightens the tail; J = 2 with uniform is improper."""
        cauchy = hierarchical_normal_posterior(
            SCHOOLS_Y, SCHOOLS_S, log_tau_prior=half_cauchy_log_prior(5.0)
        )
        self.assertLess(cauchy.tau_mean(), self.posterior.tau_mean())
        self.assertLess(cauchy.tau_quantile(0.95), self.posterior.tau_quantile(0.95))
        with self.assertRaises(ValueError):
            hierarchical_normal_posterior((1.0, 5.0), (1.0, 1.0))
        with self.assertRaises(ValueError):
            hierarchical_normal_posterior(SCHOOLS_Y, SCHOOLS_S, grid_size=100)

    def test_non_centered_gradient_is_exact(self) -> None:
        """The analytic gradient agrees with central differences."""
        for prior in (None, half_cauchy_log_prior(5.0)):
            log_density, gradient = non_centered_log_density(
                SCHOOLS_Y, SCHOOLS_S, mu_prior=(0.0, 20.0), log_tau_prior=prior
            )
            point = (3.0, 1.2, 0.3, -0.5, 1.0, 0.0, -1.2, 0.7, 0.1, -0.2)
            numeric, _ = central_gradient(log_density, point)
            for a, b in zip(gradient(point), numeric, strict=True):
                self.assertAlmostEqual(a, b, delta=1e-5 * max(1.0, abs(b)))


class InfiniteMomentTests(unittest.TestCase):
    """Moments that do not exist are infinite, not grid-dependent numbers."""

    def test_few_groups_with_uniform_tau_prior(self) -> None:
        """J = 4: E[tau] finite and grid-independent; sd(tau), sd(mu) infinite."""
        y, s = (1.0, 5.0, -3.0, 2.0), (1.0, 1.0, 1.0, 1.0)
        automatic = hierarchical_normal_posterior(y, s)
        wide = hierarchical_normal_posterior(y, s, tau_max=1e20)
        self.assertTrue(math.isinf(automatic.tau_sd()))
        self.assertTrue(math.isinf(automatic.mu_moments()[1]))
        self.assertTrue(math.isfinite(automatic.tau_mean()))
        self.assertAlmostEqual(automatic.tau_mean(), wide.tau_mean(), delta=0.05)
        three = hierarchical_normal_posterior(y[:3], s[:3])
        self.assertTrue(math.isinf(three.tau_mean()))

    def test_eight_schools_moments_stay_finite(self) -> None:
        """J = 8 has finite first and second moments of tau."""
        posterior = hierarchical_normal_posterior(SCHOOLS_Y, SCHOOLS_S)
        self.assertTrue(math.isfinite(posterior.tau_sd()))
        self.assertTrue(math.isfinite(posterior.mu_moments()[1]))


class EightSchoolsAgreementTests(unittest.TestCase):
    """NUTS on the non-centered model agrees with the exact grid posterior."""

    def test_nuts_matches_grid_within_monte_carlo_error(self) -> None:
        """Posterior means of mu, tau, theta_1 agree within 4 MCSE."""
        grid = hierarchical_normal_posterior(SCHOOLS_Y, SCHOOLS_S)
        samples = sample_hierarchical_normal(
            SCHOOLS_Y, SCHOOLS_S, draws=1000, warmup=500, chains=4, seed=2026
        )
        summaries = {s.name: s for s in samples.draws.summary()}
        self.assertLess(samples.hamiltonian.divergences, 0.01 * 4000)
        for name in ("mu", "tau", "theta_1"):
            self.assertLess(summaries[name].rank_normalized_rhat, 1.01)
            self.assertGreater(summaries[name].effective_sample_size, 400)
        mu_mean, _ = grid.mu_moments()
        theta_means, _ = grid.theta_moments()
        expectations = {
            "mu": mu_mean,
            "tau": grid.tau_mean(),
            "theta_1": theta_means[0],
        }
        for name, exact in expectations.items():
            summary = summaries[name]
            self.assertLess(
                abs(summary.mean - exact),
                4.0 * summary.mcse_mean,
                f"{name}: NUTS {summary.mean:.3f} vs grid {exact:.3f}",
            )
        taus = sorted(x[1] for c in samples.draws.chains for x in c.samples)
        self.assertAlmostEqual(taus[len(taus) // 2], grid.tau_quantile(0.5), delta=0.6)

        def simulate(theta: tuple[float, ...], rng: random.Random) -> list[float]:
            """Replicate the eight estimates given (mu, tau, theta)."""
            return [rng.gauss(t, s) for t, s in zip(theta[2:], SCHOOLS_S, strict=True)]

        check = posterior_predictive_pvalues(
            samples.draws,
            simulate,
            SCHOOLS_Y,
            statistics={"max": max, "sd": statistics.stdev},
            max_draws=2000,
            seed=4,
        )
        self.assertEqual(check.chains, 4)
        self.assertLessEqual(check.draws, 2000)
        for item in check.checks:
            self.assertGreater(item.p_value, 0.05)
            self.assertLess(item.p_value, 0.95)
            self.assertGreater(item.mcse, 0.0)


class BetaBinomialTests(unittest.TestCase):
    """Empirical Bayes for a beta-binomial hierarchy."""

    def test_recovers_prior_mean_and_shrinks(self) -> None:
        """The fitted prior mean is close to the truth and estimates shrink."""
        rng = random.Random(17)
        trials = [rng.randint(20, 80) for _ in range(60)]
        successes = []
        for n in trials:
            p = rng.betavariate(4.0, 12.0)
            successes.append(sum(rng.random() < p for _ in range(n)))
        fit = fit_beta_binomial(successes, trials)
        self.assertTrue(fit.optimization.converged)
        self.assertAlmostEqual(fit.prior_mean, 0.25, delta=0.03)
        self.assertGreater(fit.alpha + fit.beta, 5.0)
        for k, n, mean, b in zip(
            successes, trials, fit.posterior_means, fit.shrinkage, strict=True
        ):
            raw = k / n
            self.assertTrue(0.0 < b < 1.0)
            self.assertAlmostEqual(mean, (1 - b) * raw + b * fit.prior_mean, places=12)


class PosteriorPredictiveCheckTests(unittest.TestCase):
    """Calibration of posterior predictive p-values and PIT values."""

    def test_thinning_respects_the_draw_cap_across_chains(self) -> None:
        """10 chains of 3 draws thinned to at most 15 draws, not 20."""
        from probstat.inference.predictive_checks import _thinned

        chains = tuple(tuple((float(c), float(i)) for i in range(3)) for c in range(10))
        kept = _thinned(chains, 15)
        self.assertLessEqual(sum(len(chain) for chain in kept), 15)
        self.assertEqual(len(kept), 10)

    @staticmethod
    def _p_values(true_sd: float, datasets: int, seed: int) -> list[float]:
        """Return ppp-values of the sample variance under N(mu, 1), flat prior."""
        rng = random.Random(seed)
        n, draws = 20, 200
        results = []
        for index in range(datasets):
            data = tuple(rng.gauss(0.0, true_sd) for _ in range(n))
            center = statistics.fmean(data)
            posterior = [(rng.gauss(center, 1.0 / math.sqrt(n)),) for _ in range(draws)]

            def simulate(theta: tuple[float, ...], g: random.Random) -> list[float]:
                """Replicate n draws from N(mu, 1)."""
                return [g.gauss(theta[0], 1.0) for _ in range(n)]

            check = posterior_predictive_pvalues(
                posterior,
                simulate,
                data,
                statistics={"variance": sample_variance},
                seed=seed * 1000 + index,
            )
            results.append(check["variance"].p_value)
        return results

    def test_p_value_is_uniform_under_the_true_model(self) -> None:
        """The variance statistic is ancillary, so its ppp-value is uniform."""
        values = self._p_values(1.0, 120, seed=31)
        uniformity = pit_uniformity(values)
        self.assertGreater(uniformity.p_value, 0.01)
        self.assertAlmostEqual(statistics.fmean(values), 0.5, delta=0.07)
        self.assertAlmostEqual(
            sum(v <= 0.1 for v in values) / len(values), 0.1, delta=0.07
        )

    def test_p_value_is_extreme_under_misspecification(self) -> None:
        """Data with sd 2 analysed with sd 1 give p-values near zero."""
        values = self._p_values(2.0, 20, seed=32)
        self.assertLess(statistics.median(values), 0.005)
        self.assertLess(max(values), 0.1)

    def test_mcse_and_discrepancies(self) -> None:
        """Independent draws use the binomial MCSE; discrepancies are per-draw."""
        rng = random.Random(3)
        data = tuple(rng.gauss(0.0, 1.0) for _ in range(10))
        draws = [(rng.gauss(0.0, 0.3),) for _ in range(400)]

        def simulate(theta: tuple[float, ...], g: random.Random) -> list[float]:
            """Replicate ten N(mu, 1) values."""
            return [g.gauss(theta[0], 1.0) for _ in range(10)]

        check = posterior_predictive_pvalues(
            draws,
            simulate,
            data,
            statistics={"mean": statistics.fmean},
            discrepancies={"chi2": lambda y, t: math.fsum((v - t[0]) ** 2 for v in y)},
            keep_replicates=True,
            seed=9,
        )
        mean_check = check["mean"]
        p = mean_check.p_value
        self.assertAlmostEqual(mean_check.mcse, math.sqrt(p * (1 - p) / 400))
        self.assertEqual(len(set(mean_check.observed)), 1)
        self.assertGreater(len(set(check["chi2"].observed)), 1)
        self.assertTrue(check["chi2"].parameter_dependent)
        self.assertEqual(len(check.replicates), 400)
        self.assertLessEqual(mean_check.two_sided_p_value, 1.0)
        with self.assertRaises(ValueError):
            posterior_predictive_pvalues(draws, simulate, data)

    def test_pit_values(self) -> None:
        """Mid-tie and randomized PIT values follow their definitions."""
        replicates = [(0.0, 1.0), (1.0, 1.0), (2.0, 3.0), (3.0, 0.0)]
        values = pit_values((1.5, 1.0), replicates)
        self.assertEqual(values, (0.5, (1 + 0.5 * 2) / 4))
        randomized = pit_values((1.5, 1.0), replicates, randomize_ties=True, seed=1)
        self.assertEqual(randomized[0], 0.5)
        self.assertTrue(0.25 <= randomized[1] <= 0.75)
        rng = random.Random(12)
        data = [rng.gauss(0.0, 1.0) for _ in range(80)]
        reps = [[rng.gauss(0.0, 1.0) for _ in range(80)] for _ in range(300)]
        self.assertGreater(pit_uniformity(pit_values(data, reps)).p_value, 0.01)
        heavy = [rng.gauss(0.0, 3.0) for _ in range(80)]
        self.assertLess(pit_uniformity(pit_values(heavy, reps)).p_value, 0.01)


if __name__ == "__main__":
    unittest.main()
