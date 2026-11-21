"""Tests for Pareto-smoothed importance sampling and PSIS-LOO.

Oracles: generalized Pareto samples with a known shape, the Gaussian
proposal/target pair whose ratio tail has shape ``1 - 1/sigma^2``, an
independent recomputation of the smoothing formula, and the conjugate normal
model whose leave-one-out predictive densities are known in closed form and
by direct quadrature of the refitted posterior.
"""

from __future__ import annotations

import math
import unittest

from probstat.foundations.random_streams import RandomStream
from probstat.inference import (
    GeneralizedParetoFit,
    fit_generalized_pareto,
    pareto_k_threshold,
    pareto_smooth,
    psis_expectation,
    psis_loo,
    psis_tail_size,
)


def _gpd_sample(shape: float, scale: float, count: int, seed: int) -> list[float]:
    """Draw generalized Pareto variates by inversion."""
    generator = RandomStream(seed).generator()
    values = []
    for _ in range(count):
        u = generator.random()
        if shape == 0.0:
            values.append(-scale * math.log1p(-u))
        else:
            values.append(scale * ((1.0 - u) ** (-shape) - 1.0) / shape)
    return values


def _gaussian_case(
    sigma: float, size: int, seed: int
) -> tuple[list[float], list[float]]:
    """Return proposal draws from N(0, 1) and log ratios to N(0, sigma^2)."""
    generator = RandomStream(seed).generator()
    draws = [generator.gauss(0.0, 1.0) for _ in range(size)]
    log_ratios = [0.5 * x * x * (1.0 - 1.0 / sigma**2) - math.log(sigma) for x in draws]
    return draws, log_ratios


class GeneralizedParetoFitTests(unittest.TestCase):
    """The local profile estimator of the generalized Pareto tail."""

    def test_recovers_known_shape_and_scale(self) -> None:
        """The mean estimate over replicates matches the generating k and sigma."""
        for shape in (-0.3, 0.0, 0.3, 0.7):
            fits = [
                fit_generalized_pareto(
                    _gpd_sample(shape, 2.0, 1500, 100 + seed), regularize=False
                )
                for seed in range(12)
            ]
            mean_shape = math.fsum(fit.shape for fit in fits) / len(fits)
            mean_scale = math.fsum(fit.scale for fit in fits) / len(fits)
            with self.subTest(shape=shape):
                self.assertAlmostEqual(mean_shape, shape, delta=0.04)
                self.assertAlmostEqual(mean_scale, 2.0, delta=0.08)

    def test_weak_prior_is_the_documented_shrinkage(self) -> None:
        """Regularization maps k to (N k + 5) / (N + 10) and leaves sigma alone."""
        sample = _gpd_sample(0.2, 1.0, 40, 3)
        plain = fit_generalized_pareto(sample, regularize=False)
        shrunk = fit_generalized_pareto(sample)
        self.assertAlmostEqual(shrunk.shape, (40 * plain.shape + 5.0) / 50.0, places=14)
        self.assertEqual(shrunk.scale, plain.scale)
        self.assertTrue(shrunk.regularized)
        self.assertEqual(shrunk.count, 40)

    def test_quantile_inverts_the_distribution_function(self) -> None:
        """F(F^-1(p)) = p for positive, zero, and negative shape."""
        for shape in (0.4, 0.0, -0.25):
            fit = GeneralizedParetoFit(shape, 1.7, 10, False)
            for p in (0.0, 0.1, 0.5, 0.95):
                x = fit.quantile(p)
                if shape == 0.0:
                    cdf = 1.0 - math.exp(-x / 1.7)
                else:
                    cdf = 1.0 - (1.0 + shape * x / 1.7) ** (-1.0 / shape)
                with self.subTest(shape=shape, p=p):
                    self.assertAlmostEqual(cdf, p, places=12)

    def test_scale_equivariance(self) -> None:
        """Multiplying the data by c multiplies sigma by c and keeps k."""
        sample = _gpd_sample(0.3, 1.0, 200, 5)
        base = fit_generalized_pareto(sample)
        scaled = fit_generalized_pareto([7.0 * x for x in sample])
        self.assertAlmostEqual(scaled.shape, base.shape, places=10)
        self.assertAlmostEqual(scaled.scale, 7.0 * base.scale, places=9)

    def test_degenerate_sample_and_validation(self) -> None:
        """Ties at the minimum give an infinite shape; bad input raises."""
        fit = fit_generalized_pareto([0.0] * 8)
        self.assertEqual(fit.shape, math.inf)
        self.assertTrue(math.isnan(fit.scale))
        with self.assertRaisesRegex(ValueError, "degenerate"):
            fit.quantile(0.5)
        with self.assertRaisesRegex(ValueError, "at least 5"):
            fit_generalized_pareto([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fit_generalized_pareto([-1.0, 1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            fit_generalized_pareto([1.0, 2.0, 3.0, 4.0, math.inf])
        with self.assertRaises(TypeError):
            fit_generalized_pareto([1.0] * 5, regularize=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            GeneralizedParetoFit(0.1, 1.0, 5, False).quantile(1.0)


class ParetoSmoothingTests(unittest.TestCase):
    """Tail size, threshold, and the smoothing step itself."""

    def test_tail_size_and_threshold_values(self) -> None:
        """M = ceil(min(S/5, 3 sqrt(S))) and min(1 - 1/log10 S, 0.7)."""
        self.assertEqual(psis_tail_size(100), 20)
        self.assertEqual(psis_tail_size(1000), 95)
        self.assertEqual(psis_tail_size(10000), 300)
        self.assertEqual(psis_tail_size(7), 2)
        self.assertAlmostEqual(pareto_k_threshold(100), 0.5, places=14)
        self.assertAlmostEqual(pareto_k_threshold(1000), 2.0 / 3.0, places=14)
        self.assertEqual(pareto_k_threshold(100000), 0.7)
        # S = 10^(1/(1-k)) is the sample size at which k is on the threshold.
        self.assertAlmostEqual(
            pareto_k_threshold(round(10 ** (1.0 / (1.0 - 0.6)))), 0.6, delta=1e-3
        )
        for bad in (0, -1, 2.5):
            with self.assertRaises((ValueError, TypeError)):
                psis_tail_size(bad)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "at least 2"):
            pareto_k_threshold(1)

    def test_smoothing_matches_an_independent_recomputation(self) -> None:
        """The tail equals u + F^-1((z - 1/2)/M) truncated at the raw maximum."""
        _, log_ratios = _gaussian_case(1.6, 400, 11)
        result = pareto_smooth(log_ratios)
        size = len(log_ratios)
        tail = psis_tail_size(size)
        self.assertEqual(result.tail_size, tail)
        self.assertTrue(result.smoothed)
        top = max(log_ratios)
        ratios = sorted(math.exp(value - top) for value in log_ratios)
        cutoff = ratios[size - tail - 1]
        fit = fit_generalized_pareto([r - cutoff for r in ratios[size - tail :]])
        self.assertAlmostEqual(result.pareto_k, fit.shape, places=12)
        self.assertAlmostEqual(result.scale, fit.scale, places=12)
        expected = ratios[: size - tail] + [
            min(cutoff + fit.quantile((z - 0.5) / tail), 1.0)
            for z in range(1, tail + 1)
        ]
        total = math.fsum(expected)
        weights = sorted(result.weights)
        for got, want in zip(weights, expected, strict=True):
            self.assertAlmostEqual(got, want / total, places=13)
        self.assertAlmostEqual(math.fsum(result.weights), 1.0, places=12)
        self.assertAlmostEqual(
            result.effective_sample_size,
            1.0 / math.fsum(w * w for w in result.weights),
            places=8,
        )

    def test_body_untouched_and_order_preserved(self) -> None:
        """Non-tail ratios keep their mutual ratios and ranks are unchanged."""
        _, log_ratios = _gaussian_case(2.0, 300, 4)
        result = pareto_smooth(log_ratios)
        order = sorted(range(300), key=log_ratios.__getitem__)
        body = order[: 300 - result.tail_size]
        shift = result.log_weights[body[0]] - log_ratios[body[0]]
        for index in body:
            self.assertAlmostEqual(
                result.log_weights[index] - log_ratios[index], shift, places=11
            )
        smoothed_order = [result.log_weights[index] for index in order]
        self.assertEqual(smoothed_order, sorted(smoothed_order))
        raw_top = max(log_ratios) + shift
        self.assertLessEqual(max(result.log_weights), raw_top + 1e-12)

    def test_invariance_to_an_additive_constant(self) -> None:
        """Only differences of the log ratios matter."""
        _, log_ratios = _gaussian_case(1.5, 250, 8)
        base = pareto_smooth(log_ratios)
        moved = pareto_smooth([value + 700.0 for value in log_ratios])
        self.assertAlmostEqual(base.pareto_k, moved.pareto_k, places=9)
        for a, b in zip(base.log_weights, moved.log_weights, strict=True):
            self.assertAlmostEqual(a, b, places=9)

    def test_gaussian_tail_shape_is_recovered(self) -> None:
        """k-hat approaches 1 - 1/sigma^2 for the N(0,1) -> N(0,sigma^2) ratio."""
        for sigma, tolerance in ((1.2, 0.06), (1.5, 0.1)):
            shapes = [
                pareto_smooth(_gaussian_case(sigma, 20000, 40 + seed)[1]).pareto_k
                for seed in range(6)
            ]
            mean = math.fsum(shapes) / len(shapes)
            with self.subTest(sigma=sigma):
                self.assertAlmostEqual(mean, 1.0 - 1.0 / sigma**2, delta=tolerance)

    def test_estimate_of_second_moment_converges_for_light_tails(self) -> None:
        """E[x^2] = sigma^2 is recovered within the replicate standard error."""
        sigma = 1.2
        estimates = []
        for seed in range(12):
            draws, log_ratios = _gaussian_case(sigma, 4000, 200 + seed)
            estimate = psis_expectation(log_ratios, [x * x for x in draws])
            self.assertTrue(estimate.reliable)
            self.assertGreater(estimate.effective_sample_size, 2000.0)
            estimates.append(estimate.estimate)
        mean = math.fsum(estimates) / len(estimates)
        spread = math.sqrt(
            math.fsum((e - mean) ** 2 for e in estimates) / (len(estimates) - 1)
        )
        self.assertAlmostEqual(mean, sigma**2, delta=4.0 * spread / math.sqrt(12))
        # sigma = 1.5 has k = 0.56 < 0.7: still convergent, with a small bias.
        draws, log_ratios = _gaussian_case(1.5, 40000, 9)
        estimate = psis_expectation(log_ratios, [x * x for x in draws])
        self.assertTrue(estimate.reliable)
        self.assertAlmostEqual(estimate.estimate, 2.25, delta=0.06 * 2.25)

    def test_heavy_tail_is_flagged(self) -> None:
        """Sigma = 3 gives k = 0.89 > 0.7 and the estimate is flagged."""
        draws, log_ratios = _gaussian_case(3.0, 20000, 2)
        result = pareto_smooth(log_ratios)
        self.assertGreater(result.pareto_k, 0.7)
        self.assertFalse(result.reliable)
        estimate = result.expectation([x * x for x in draws])
        self.assertFalse(estimate.reliable)
        self.assertLess(estimate.estimate, 9.0)

    def test_standard_error_formula_and_uniform_case(self) -> None:
        """Equal ratios give the plain mean, its standard error, and ESS = S."""
        values = [0.5, 1.5, -2.0, 4.0, 0.25, 3.0]
        estimate = psis_expectation([3.0] * 6, values)
        mean = math.fsum(values) / 6
        self.assertAlmostEqual(estimate.estimate, mean, places=14)
        self.assertAlmostEqual(
            estimate.standard_error,
            math.sqrt(math.fsum((v - mean) ** 2 for v in values)) / 6,
            places=14,
        )
        self.assertAlmostEqual(estimate.effective_sample_size, 6.0, places=12)
        self.assertEqual(estimate.pareto_k, -math.inf)
        self.assertTrue(estimate.reliable)

    def test_small_samples_are_not_smoothed_and_flagged(self) -> None:
        """Fewer than five tail draws: raw weights, k-hat = inf, unreliable."""
        log_ratios = [0.0, 0.3, -0.2, 1.0, 0.4, -1.0, 0.2, 0.1]
        result = pareto_smooth(log_ratios)
        self.assertEqual(result.pareto_k, math.inf)
        self.assertFalse(result.smoothed)
        self.assertFalse(result.reliable)
        total = math.fsum(math.exp(v) for v in log_ratios)
        for weight, value in zip(result.weights, log_ratios, strict=True):
            self.assertAlmostEqual(weight, math.exp(value) / total, places=14)
        self.assertEqual(result.sample_size, 8)

    def test_zero_ratios_are_allowed(self) -> None:
        """A ratio of zero (log -inf) gets weight zero."""
        _, log_ratios = _gaussian_case(1.3, 200, 6)
        log_ratios[17] = -math.inf
        result = pareto_smooth(log_ratios)
        self.assertEqual(result.weights[17], 0.0)
        self.assertAlmostEqual(math.fsum(result.weights), 1.0, places=12)

    def test_validation(self) -> None:
        """Invalid ratios and mismatched values raise ValueError."""
        with self.assertRaisesRegex(ValueError, "at least 2"):
            pareto_smooth([0.0])
        with self.assertRaisesRegex(ValueError, "NaN"):
            pareto_smooth([0.0, math.nan, 1.0])
        with self.assertRaisesRegex(ValueError, "NaN"):
            pareto_smooth([0.0, math.inf, 1.0])
        with self.assertRaisesRegex(ValueError, "positive"):
            pareto_smooth([-math.inf, -math.inf])
        result = pareto_smooth([0.0, 1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "one entry per draw"):
            result.expectation([1.0, 2.0])
        with self.assertRaises(ValueError):
            result.expectation([1.0, 2.0, math.nan])


def _normal_log_pdf(x: float, mean: float, variance: float) -> float:
    """Return the log-density of N(mean, variance) at x."""
    return -0.5 * (x - mean) ** 2 / variance - 0.5 * math.log(2 * math.pi * variance)


class _ConjugateNormal:
    """y_i ~ N(mu, sigma^2) with known sigma and prior mu ~ N(m0, s0^2)."""

    def __init__(self, y: list[float], sigma: float = 1.0, s0: float = 10.0) -> None:
        """Store the data and the hyperparameters (prior mean zero)."""
        self.y = y
        self.sigma = sigma
        self.s0 = s0

    def posterior(self, omit: int | None = None) -> tuple[float, float]:
        """Return the posterior mean and variance, optionally leaving one out."""
        data = [v for i, v in enumerate(self.y) if i != omit]
        precision = 1.0 / self.s0**2 + len(data) / self.sigma**2
        return math.fsum(data) / self.sigma**2 / precision, 1.0 / precision

    def exact_loo(self, index: int) -> float:
        """Return log p(y_i | y_-i) in closed form."""
        mean, variance = self.posterior(index)
        return _normal_log_pdf(self.y[index], mean, variance + self.sigma**2)

    def exact_lpd(self, index: int) -> float:
        """Return log p(y_i | y) (full-data posterior predictive)."""
        mean, variance = self.posterior()
        return _normal_log_pdf(self.y[index], mean, variance + self.sigma**2)

    def quadrature_loo(self, index: int) -> float:
        """Refit without y_i by brute-force quadrature over mu."""
        data = [v for i, v in enumerate(self.y) if i != index]

        def log_joint(mu: float) -> float:
            """Unnormalized log posterior of mu given the retained data."""
            return _normal_log_pdf(mu, 0.0, self.s0**2) + math.fsum(
                _normal_log_pdf(v, mu, self.sigma**2) for v in data
            )

        center = math.fsum(data) / len(data)
        grid = [center - 6.0 + 12.0 * j / 4000 for j in range(4001)]
        top = max(log_joint(mu) for mu in grid)
        weights = [math.exp(log_joint(mu) - top) for mu in grid]
        numerator = math.fsum(
            w * math.exp(_normal_log_pdf(self.y[index], mu, self.sigma**2))
            for w, mu in zip(weights, grid, strict=True)
        )
        return math.log(numerator / math.fsum(weights))

    def log_likelihood(self, draws: int, seed: int) -> list[list[float]]:
        """Return the pointwise log-likelihood at exact posterior draws."""
        mean, variance = self.posterior()
        generator = RandomStream(seed).generator()
        samples = [generator.gauss(mean, math.sqrt(variance)) for _ in range(draws)]
        return [
            [_normal_log_pdf(v, mu, self.sigma**2) for v in self.y] for mu in samples
        ]


class PSISLOOTests(unittest.TestCase):
    """PSIS-LOO against the conjugate normal model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Simulate twenty observations and one set of posterior draws."""
        generator = RandomStream(7).generator()
        cls.model = _ConjugateNormal([generator.gauss(1.0, 1.0) for _ in range(20)])
        cls.result = psis_loo(cls.model.log_likelihood(4000, 1))

    def test_closed_form_agrees_with_brute_force_refits(self) -> None:
        """The oracle itself: closed-form LOO equals quadrature of the refit."""
        for index in (0, 7, 19):
            self.assertAlmostEqual(
                self.model.exact_loo(index),
                self.model.quadrature_loo(index),
                places=8,
            )

    def test_pointwise_and_total_elpd(self) -> None:
        """elpd_loo matches the exact leave-one-out log predictive densities."""
        exact = [self.model.exact_loo(i) for i in range(20)]
        for got, want in zip(self.result.pointwise_elpd, exact, strict=True):
            self.assertAlmostEqual(got, want, delta=0.02)
        self.assertAlmostEqual(self.result.elpd_loo, math.fsum(exact), delta=0.05)
        self.assertEqual(self.result.observations, 20)
        self.assertEqual(self.result.draws, 4000)

    def test_p_loo_and_standard_errors(self) -> None:
        """p_loo matches lpd - elpd_loo exactly known; se is sqrt(n Var_i)."""
        exact_p = math.fsum(
            self.model.exact_lpd(i) - self.model.exact_loo(i) for i in range(20)
        )
        self.assertAlmostEqual(self.result.p_loo, exact_p, delta=0.05)
        self.assertAlmostEqual(
            self.result.p_loo, math.fsum(self.result.pointwise_p_loo), places=12
        )
        values = self.result.pointwise_elpd
        mean = math.fsum(values) / 20
        variance = math.fsum((v - mean) ** 2 for v in values) / 19
        self.assertAlmostEqual(
            self.result.standard_error, math.sqrt(20 * variance), places=12
        )
        self.assertGreater(self.result.p_loo_standard_error, 0.0)

    def test_well_specified_model_is_not_flagged(self) -> None:
        """All k-hat are below the threshold when no observation dominates."""
        self.assertEqual(self.result.threshold, 0.7)
        self.assertEqual(self.result.flagged, ())
        self.assertTrue(self.result.reliable)
        self.assertLess(max(self.result.pareto_k), 0.5)
        self.assertGreater(min(self.result.effective_sample_sizes), 1000.0)

    def test_influential_observation_is_flagged(self) -> None:
        """An outlier among five observations has k-hat > 0.7 and a biased elpd."""
        model = _ConjugateNormal([0.1, -0.3, 0.4, 0.0, 9.0])
        result = psis_loo(model.log_likelihood(4000, 2))
        self.assertEqual(result.flagged, (4,))
        self.assertFalse(result.reliable)
        self.assertGreater(result.pareto_k[4], 0.7)
        for index in range(4):
            self.assertAlmostEqual(
                result.pointwise_elpd[index], model.exact_loo(index), delta=0.1
            )
        # The flagged term is optimistic: importance sampling cannot reach the
        # leave-one-out posterior from draws centred near the outlier's pull.
        self.assertGreater(result.pointwise_elpd[4], model.exact_loo(4) + 0.3)

    def test_validation(self) -> None:
        """Malformed log-likelihood matrices raise ValueError."""
        with self.assertRaisesRegex(ValueError, "at least 2 posterior draws"):
            psis_loo([[0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            psis_loo([[], []])
        with self.assertRaisesRegex(ValueError, "one log-likelihood per"):
            psis_loo([[0.0, 1.0], [0.0]])
        with self.assertRaises(ValueError):
            psis_loo([[0.0, math.nan], [0.0, 1.0]])
        single = psis_loo([[-1.0], [-1.2], [-0.8]])
        self.assertTrue(math.isnan(single.standard_error))


if __name__ == "__main__":
    unittest.main()
