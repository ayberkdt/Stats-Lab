"""Tests for estimator evaluation, Cramer-Rao bounds, and classical examples."""

from __future__ import annotations

import math
import random
import unittest
from collections.abc import Sequence
from functools import partial

from probstat.inference import (
    EstimatorRisk,
    check_sufficiency,
    consistency_profile,
    cramer_rao_bound,
    evaluate_estimator,
    minimum_mse_variance_divisor,
    mle_efficiency,
    normal_variance_estimator_risk,
    poisson_zero_indicator,
    poisson_zero_probability_risks,
    poisson_zero_rao_blackwell,
    uniform_maximum_risks,
)
from probstat.information import normal_fisher_information


def normal_sample(n: int, rng: random.Random) -> list[float]:
    """Draw n standard-normal-scaled N(0, 2^2) observations."""
    return [rng.gauss(0.0, 2.0) for _ in range(n)]


def standard_normal_sample(n: int, rng: random.Random) -> list[float]:
    """Draw n N(0, 1) observations."""
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def uniform_sample(n: int, rng: random.Random) -> list[float]:
    """Draw n Uniform(0, 3) observations."""
    return [3.0 * rng.random() for _ in range(n)]


def poisson_draw(lam: float, rng: random.Random) -> int:
    """Draw one Poisson count by multiplying uniforms (Knuth)."""
    limit = math.exp(-lam)
    count = 0
    product = rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def poisson_sample(n: int, rng: random.Random) -> list[float]:
    """Draw n Poisson(1) counts."""
    return [float(poisson_draw(1.0, rng)) for _ in range(n)]


def sum_of_squares_over(divisor_offset: int, sample: Sequence[float]) -> float:
    """Return SS / (n + divisor_offset)."""
    center = math.fsum(sample) / len(sample)
    ss = math.fsum((x - center) ** 2 for x in sample)
    return ss / (len(sample) + divisor_offset)


def sample_mean(sample: Sequence[float]) -> float:
    """Return the sample mean."""
    return math.fsum(sample) / len(sample)


def sample_median(sample: Sequence[float]) -> float:
    """Return the sample median."""
    ordered = sorted(sample)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def z_interval(sample: Sequence[float]) -> tuple[float, float]:
    """Return the known-variance (sigma = 1) 95% z interval."""
    center = sample_mean(sample)
    half = 1.959963984540054 / math.sqrt(len(sample))
    return center - half, center + half


class ExactRiskTests(unittest.TestCase):
    """Exact risks of classical estimators against simulation."""

    def test_variance_divisors_exact_and_simulated(self) -> None:
        """MSE of SS/(n-1), SS/n, SS/(n+1) matches simulation; n+1 is best."""
        n, sigma = 8, 2.0
        risks = [normal_variance_estimator_risk(sigma, n, n + k) for k in (-1, 0, 1)]
        self.assertAlmostEqual(risks[0].bias, 0.0, places=14)
        self.assertAlmostEqual(risks[0].mse, 2.0 * sigma**4 / (n - 1), places=12)
        self.assertAlmostEqual(risks[2].mse, 2.0 * sigma**4 / (n + 1), places=12)
        self.assertLess(risks[2].mse, risks[1].mse)
        self.assertLess(risks[1].mse, risks[0].mse)
        self.assertEqual(minimum_mse_variance_divisor(n), n + 1)
        grid = [n - 3 + 0.01 * i for i in range(600)]
        best = min(grid, key=lambda d: normal_variance_estimator_risk(sigma, n, d).mse)
        self.assertAlmostEqual(best, n + 1, delta=0.011)
        for offset, risk in zip((-1, 0, 1), risks, strict=True):
            result = evaluate_estimator(
                partial(sum_of_squares_over, offset),
                normal_sample,
                sigma**2,
                n,
                4000,
                seed=100 + offset,
            )
            self.assertLess(abs(result.bias - risk.bias), 4.0 * result.bias_se)
            self.assertLess(
                abs(result.variance - risk.variance), 4.0 * result.variance_se
            )
            self.assertLess(abs(result.mse - risk.mse), 4.0 * result.mse_se)

    def test_uniform_maximum_exact_bias_and_mse(self) -> None:
        """Exact bias/MSE of max and the UMVUE (n+1)/n max, with simulation."""
        theta, n = 3.0, 10
        mle, umvue, min_mse, moments = uniform_maximum_risks(theta, n)
        self.assertAlmostEqual(mle.bias, -theta / 11, places=14)
        self.assertAlmostEqual(mle.mse, 2 * theta**2 / (11 * 12), places=14)
        self.assertAlmostEqual(umvue.bias, 0.0)
        self.assertAlmostEqual(umvue.variance, theta**2 / (10 * 12), places=14)
        self.assertAlmostEqual(min_mse.mse, theta**2 / 121, places=14)
        self.assertLess(min_mse.mse, umvue.mse)
        self.assertLess(umvue.mse, mle.mse)
        self.assertLess(mle.mse, moments.mse)
        simulated_mle = evaluate_estimator(max, uniform_sample, theta, n, 4000, seed=7)
        self.assertLess(abs(simulated_mle.bias - mle.bias), 4 * simulated_mle.bias_se)
        self.assertLess(abs(simulated_mle.mse - mle.mse), 4 * simulated_mle.mse_se)
        simulated_umvue = evaluate_estimator(
            lambda s: (len(s) + 1) / len(s) * max(s),
            uniform_sample,
            theta,
            n,
            4000,
            seed=7,
            reference=max,
        )
        self.assertLess(abs(simulated_umvue.bias), 4 * simulated_umvue.bias_se)
        self.assertLess(
            abs(simulated_umvue.variance - umvue.variance),
            4 * simulated_umvue.variance_se,
        )
        assert simulated_umvue.relative_efficiency is not None
        assert simulated_umvue.relative_efficiency_se is not None
        self.assertLess(
            abs(simulated_umvue.relative_efficiency - mle.mse / umvue.mse),
            4 * simulated_umvue.relative_efficiency_se,
        )

    def test_rao_blackwellization_lowers_variance(self) -> None:
        """Both estimators of exp(-lambda) are unbiased; RB has lower variance."""
        lam, n = 1.0, 10
        risks = poisson_zero_probability_risks(lam, n)
        self.assertEqual(risks.crude.bias, 0.0)
        self.assertEqual(risks.rao_blackwellized.bias, 0.0)
        self.assertAlmostEqual(
            risks.crude.variance, math.exp(-1.0) * (1 - math.exp(-1.0)), places=14
        )
        self.assertAlmostEqual(
            risks.rao_blackwellized.variance,
            math.exp(-2.0) * (math.exp(0.1) - 1.0),
            places=14,
        )
        self.assertLess(risks.rao_blackwellized.variance, risks.crude.variance / 15)
        self.assertGreater(risks.rao_blackwellized.variance, risks.crlb)
        self.assertGreater(risks.mle.bias, 0.0)
        target = math.exp(-lam)
        crude = evaluate_estimator(
            poisson_zero_indicator, poisson_sample, target, n, 6000, seed=21
        )
        rb = evaluate_estimator(
            poisson_zero_rao_blackwell,
            poisson_sample,
            target,
            n,
            6000,
            seed=21,
            reference=poisson_zero_indicator,
        )
        self.assertLess(abs(crude.bias), 4 * crude.bias_se)
        self.assertLess(abs(rb.bias), 4 * rb.bias_se)
        self.assertLess(
            abs(crude.variance - risks.crude.variance), 4 * crude.variance_se
        )
        self.assertLess(
            abs(rb.variance - risks.rao_blackwellized.variance), 4 * rb.variance_se
        )
        self.assertLess(rb.variance, crude.variance)
        assert rb.relative_efficiency is not None
        self.assertGreater(rb.relative_efficiency, 10.0)
        self.assertEqual(poisson_zero_rao_blackwell([0.0]), 1.0)

    def test_consistency_profile(self) -> None:
        """Bias and variance -> 0, so the Markov tail bound -> 0."""
        rows = consistency_profile(
            lambda n: normal_variance_estimator_risk(1.0, n, n),
            (5, 50, 500, 5000),
            0.1,
        )
        mses = [row.mse for row in rows]
        self.assertEqual(mses, sorted(mses, reverse=True))
        self.assertLess(abs(rows[-1].bias), 1e-3)
        self.assertLess(rows[-1].tail_bound, 0.05)
        self.assertEqual(rows[0].tail_bound, 1.0)
        uniform_rows = consistency_profile(
            lambda n: uniform_maximum_risks(1.0, n)[0], (10, 1000), 0.05
        )
        self.assertLess(uniform_rows[-1].tail_bound, 1e-3)
        self.assertIsInstance(EstimatorRisk("x", 0.0, 1.0).mse, float)


class CramerRaoTests(unittest.TestCase):
    """Cramer-Rao bounds and exact MLE efficiency."""

    def test_efficient_mles_attain_the_bound(self) -> None:
        """Bernoulli, Poisson, and normal-mean MLEs have efficiency one."""
        for family, value in (
            ("bernoulli", 0.3),
            ("poisson", 2.0),
            ("normal_mean", 1.5),
        ):
            record = mle_efficiency(family, 12, value)  # type: ignore[arg-type]
            self.assertAlmostEqual(record.efficiency, 1.0, places=12)

    def test_biased_mles(self) -> None:
        """Normal-variance MLE beats the unbiased bound; 1/Xbar approaches it."""
        normal = mle_efficiency("normal_variance", 10, 2.0)
        self.assertAlmostEqual(normal.crlb, 2 * 2.0**4 / 10, places=12)
        self.assertGreater(normal.efficiency, 1.0)
        self.assertGreaterEqual(normal.variance, normal.biased_crlb)
        efficiencies = [
            mle_efficiency("exponential_rate", n, 0.5).efficiency
            for n in (5, 50, 500)
        ]
        self.assertAlmostEqual(efficiencies[1], 48 * 49 / (50 * 52), places=12)
        self.assertLess(efficiencies[0], efficiencies[1])
        self.assertLess(efficiencies[1], efficiencies[2])
        self.assertGreater(efficiencies[2], 0.99)
        record = mle_efficiency("exponential_rate", 10, 0.5)
        self.assertGreaterEqual(record.variance, record.biased_crlb)

    def test_exponential_rate_mle_bias_by_simulation(self) -> None:
        """Simulated bias of 1/Xbar matches lambda/(n - 1)."""
        record = mle_efficiency("exponential_rate", 6, 2.0)

        def exponential_sample(n: int, rng: random.Random) -> list[float]:
            """Draw n Exponential(2) observations."""
            return [rng.expovariate(2.0) for _ in range(n)]

        result = evaluate_estimator(
            lambda s: 1.0 / sample_mean(s), exponential_sample, 2.0, 6, 6000, seed=3
        )
        self.assertLess(abs(result.bias - record.bias), 4 * result.bias_se)

    def test_vector_bound_and_validation(self) -> None:
        """Gradient handling for the two-parameter normal information."""
        info = normal_fisher_information(2.0)
        self.assertAlmostEqual(cramer_rao_bound(info, 4, (1.0, 0.0)), 1.0)
        self.assertAlmostEqual(cramer_rao_bound(info, 4, (0.0, 1.0)), 0.5)
        with self.assertRaises(ValueError):
            cramer_rao_bound(info, 4)
        with self.assertRaises(ValueError):
            mle_efficiency("gamma", 4, 1.0)  # type: ignore[arg-type]


class EvaluateEstimatorTests(unittest.TestCase):
    """Monte Carlo standard errors reported by evaluate_estimator."""

    def test_standard_errors_match_spread_over_repeated_runs(self) -> None:
        """Reported SEs agree with the run-to-run standard deviation."""
        runs = [
            evaluate_estimator(
                sample_mean,
                standard_normal_sample,
                0.0,
                10,
                200,
                seed=1000 + run,
                interval=z_interval,
                reference=sample_median,
            )
            for run in range(60)
        ]

        def spread(values: list[float]) -> float:
            """Return the sample standard deviation."""
            center = math.fsum(values) / len(values)
            return math.sqrt(
                math.fsum((v - center) ** 2 for v in values) / (len(values) - 1)
            )

        def mean(values: list[float]) -> float:
            """Return the average."""
            return math.fsum(values) / len(values)

        pairs = [
            ([r.bias for r in runs], [r.bias_se for r in runs]),
            ([r.variance for r in runs], [r.variance_se for r in runs]),
            ([r.mse for r in runs], [r.mse_se for r in runs]),
            (
                [r.coverage or 0.0 for r in runs],
                [r.coverage_se or 0.0 for r in runs],
            ),
            (
                [r.relative_efficiency or 0.0 for r in runs],
                [r.relative_efficiency_se or 0.0 for r in runs],
            ),
        ]
        for estimates, errors in pairs:
            ratio = spread(estimates) / mean(errors)
            self.assertGreater(ratio, 0.7)
            self.assertLess(ratio, 1.4)
        self.assertLess(abs(mean([r.coverage or 0.0 for r in runs]) - 0.95), 0.01)

    def test_reproducible_and_validated(self) -> None:
        """Same seed, same result; non-finite estimates are rejected."""
        first = evaluate_estimator(sample_mean, standard_normal_sample, 0.0, 5, 50, 9)
        second = evaluate_estimator(sample_mean, standard_normal_sample, 0.0, 5, 50, 9)
        self.assertEqual(first, second)
        self.assertIsNone(first.coverage)
        self.assertIsNone(first.relative_efficiency)
        with self.assertRaises(ValueError):
            evaluate_estimator(
                lambda s: math.nan, standard_normal_sample, 0.0, 5, 50, 9
            )
        with self.assertRaises(ValueError):
            evaluate_estimator(sample_mean, standard_normal_sample, 0.0, 5, 3, 9)


class SufficiencyTests(unittest.TestCase):
    """Numerical factorization check."""

    def test_exponential_family_statistics_pass(self) -> None:
        """Sum is sufficient for a normal mean and a Poisson rate."""

        def normal_log_likelihood(mu: float, sample: Sequence[float]) -> float:
            """Return the N(mu, 1.3^2) log-likelihood."""
            return math.fsum(
                -0.5 * ((x - mu) / 1.3) ** 2 - math.log(1.3 * math.sqrt(2 * math.pi))
                for x in sample
            )

        def poisson_log_likelihood(lam: float, sample: Sequence[float]) -> float:
            """Return the Poisson log-likelihood."""
            return math.fsum(
                x * math.log(lam) - lam - math.lgamma(x + 1.0) for x in sample
            )

        grid = (-1.0, 0.0, 0.7, 2.0, 5.0)
        normal = check_sufficiency(
            normal_log_likelihood, math.fsum, (0.0, 3.0, 3.0), (4.0, 1.0, 1.0), grid
        )
        self.assertTrue(normal.same_statistic)
        self.assertTrue(normal.consistent)
        poisson = check_sufficiency(
            poisson_log_likelihood,
            math.fsum,
            (0.0, 3.0, 3.0),
            (2.0, 2.0, 2.0),
            (0.5, 1.0, 3.0),
        )
        self.assertTrue(poisson.consistent)

    def test_non_sufficient_statistic_is_refuted(self) -> None:
        """The sample mean is not sufficient for a Laplace location."""

        def laplace_log_likelihood(theta: float, sample: Sequence[float]) -> float:
            """Return the Laplace(theta, 1) log-likelihood."""
            return math.fsum(-abs(x - theta) - math.log(2.0) for x in sample)

        result = check_sufficiency(
            laplace_log_likelihood,
            lambda s: (math.fsum(s) / len(s),),
            (0.0, 3.0, 3.0),
            (4.0, 1.0, 1.0),
            (-1.0, 0.5, 1.0, 2.0, 3.5),
        )
        self.assertTrue(result.same_statistic)
        self.assertFalse(result.consistent)
        self.assertGreater(result.log_ratio_spread, 1.0)
        different = check_sufficiency(
            laplace_log_likelihood, math.fsum, (0.0,), (1.0,), (0.0, 1.0)
        )
        self.assertIsNone(different.consistent)


if __name__ == "__main__":
    unittest.main()
