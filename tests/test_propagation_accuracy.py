"""Accuracy statements of forward propagation (ADR-0058, workstream W1).

Oracles: rank pairs and coverages of the order-statistic interval, chi-square
and Student-t quantiles were computed with SciPy 1.18
(``scipy.stats.binom``, ``chi2.ppf``, ``t.ppf``) in a separate verification
environment and pasted as literals; the coverage studies use seeded
simulations against closed-form means and quantiles.
"""

from __future__ import annotations

import math
import pickle
import statistics
import unittest
import warnings

from probstat.distributions import Normal, Uniform
from probstat.foundations.random_streams import resolve_stream
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.propagation import (
    make_design,
    order_statistic_interval,
    propagate,
    propagate_samples,
    replicate_interval,
)
from probstat.uq import QuantityOfInterest

Y = (QuantityOfInterest("y", "response"),)


def smooth(point):
    """Return ``y = x1 + x2^2`` (mean 5/6 on the unit square)."""
    return {"y": point["x1"] + point["x2"] ** 2}


def gaussian(point):
    """Return ``y = z``."""
    return {"y": point["z"]}


def failing_tail(point):
    """Return ``y = z`` but fail for ``z > 1.5``."""
    if point["z"] > 1.5:
        raise RuntimeError("no convergence")
    return {"y": point["z"]}


def square_model() -> ProbabilisticModel:
    """Return the smooth model on two independent uniforms."""
    inputs = JointInput.independent(
        RandomInput("x1", Uniform(0.0, 1.0)), RandomInput("x2", Uniform(0.0, 1.0))
    )
    return ProbabilisticModel(inputs, smooth, Y)


def normal_model(function=gaussian, **options) -> ProbabilisticModel:
    """Return ``y = z`` with ``z ~ N(0, 1)``."""
    inputs = JointInput.independent(RandomInput("z", Normal(0.0, 1.0)))
    return ProbabilisticModel(inputs, function, Y, **options)


class OrderStatisticIntervalTests(unittest.TestCase):
    """EQ-PROP-QCI-001 against SciPy binomial ranks."""

    def test_ranks_and_coverage_match_scipy(self) -> None:
        """(n, p) -> (l, u, coverage) from scipy.stats.binom."""
        cases = (
            (100, 0.5, 40, 61, 0.9647997997822952),
            (1000, 0.9, 881, 919, 0.9549071811148259),
            (20, 0.5, 6, 15, 0.9586105346679688),
            (4096, 0.95, 3863, 3919, 0.9552653861941002),
        )
        for n, p, lower, upper, coverage in cases:
            with self.subTest(n=n, p=p):
                values = [float(k) for k in range(n)][::-1]  # order-independent
                result = order_statistic_interval(values, p)
                self.assertEqual((result.lower_rank, result.upper_rank), (lower, upper))
                self.assertEqual((result.lower, result.upper), (lower - 1, upper - 1))
                self.assertAlmostEqual(result.coverage, coverage, places=11)
                self.assertGreaterEqual(result.coverage, 0.95)

    def test_too_small_sample_is_an_error(self) -> None:
        """59 draws cannot bound the 5% quantile from below at 95%."""
        with self.assertRaisesRegex(ValueError, "too few"):
            order_statistic_interval([float(k) for k in range(59)], 0.05)
        with self.assertRaises(ValueError):
            order_statistic_interval([1.0, 2.0], 1.0)

    def test_coverage_of_the_median_by_simulation(self) -> None:
        """200 seeded MC samples of 101 normals: coverage near 0.96 or above.

        The exact coverage of the rank pair is reported; the empirical rate
        over 200 independent samples must agree within simulation error.
        """
        model = normal_model()
        covered = 0
        exact = None
        for seed in range(200):
            output = propagate(model, 101, design="mc", seed=seed)["y"]
            interval = output.quantile_interval(0.5)
            exact = interval.coverage
            covered += interval.lower <= 0.0 <= interval.upper
        assert exact is not None
        self.assertGreaterEqual(exact, 0.95)
        rate = covered / 200
        self.assertLess(abs(rate - exact), 3.0 * math.sqrt(exact * (1 - exact) / 200))


class VarianceIntervalTests(unittest.TestCase):
    """EQ-PROP-VARCI-001 and the bootstrap alternative."""

    def test_normal_interval_uses_chi_square_quantiles(self) -> None:
        """chi2.ppf(0.975, 49) = 70.2224..., chi2.ppf(0.025, 49) = 31.5549..."""
        output = propagate(normal_model(), 50, design="mc", seed=4)["y"]
        s2 = statistics.variance(output.values)
        interval = output.variance_interval(0.95)
        self.assertAlmostEqual(interval.lower, 49 * s2 / 70.22241356643451, places=12)
        self.assertAlmostEqual(interval.upper, 49 * s2 / 31.554916462667126, places=12)
        self.assertIn("normal", interval.method)

    def test_bootstrap_interval_is_reproducible_and_plausible(self) -> None:
        """The bootstrap is seeded and contains the sample variance."""
        output = propagate(normal_model(), 400, design="mc", seed=8)["y"]
        first = output.variance_interval(method="bootstrap", resamples=300, seed=3)
        second = output.variance_interval(method="bootstrap", resamples=300, seed=3)
        self.assertEqual(first, second)
        s2 = statistics.variance(output.values)
        self.assertLess(first.lower, s2)
        self.assertLess(s2, first.upper)
        with self.assertRaises(ValueError):
            output.variance_interval(method="jackknife")
        with self.assertRaises(ValueError):
            output.variance_interval(method="bootstrap", resamples=10)


class ReplicateIntervalTests(unittest.TestCase):
    """EQ-PROP-RQMC-CI-001: randomized-QMC replicate intervals."""

    def test_replicate_interval_formula(self) -> None:
        """Interval is mean +- t s / sqrt(10), t_{0.975, 9} = 2.262157162798205."""
        estimates = [1.0, 2.0, 4.0, 3.0, 5.0, 2.5, 3.5, 1.5, 4.5, 3.0]
        interval = replicate_interval(estimates)
        center = statistics.fmean(estimates)
        half = 2.262157162798205 * statistics.stdev(estimates) / math.sqrt(10)
        self.assertAlmostEqual(interval.lower, center - half, places=12)
        self.assertAlmostEqual(interval.upper, center + half, places=12)
        with self.assertRaises(ValueError):
            replicate_interval([1.0])

    def test_without_replicates_outputs_are_unchanged(self) -> None:
        """The default path reproduces the documented stream and design."""
        model = square_model()
        study = propagate(model, 64, design="sobol", seed=11)
        unit = make_design(
            "sobol", 64, 2, seed=resolve_stream(11).child("propagation", "sobol")
        )
        expected = tuple(
            smooth(dict(zip(("x1", "x2"), point, strict=True)))["y"]
            for point in model.inputs.sample_design(unit)
        )
        self.assertEqual(study["y"].values, expected)
        self.assertEqual(study.replicates, 0)
        self.assertEqual(study["y"].replicate_sizes, ())

    def test_replicates_cover_the_exact_mean_and_are_honest(self) -> None:
        """Scrambled Sobol, 8 x 64 points, 60 seeds; exact mean 5/6.

        The replicate t interval covers at about its nominal rate and is far
        narrower than the independent-draw interval of the pooled sample,
        which over-states the error of a quasi-random design.
        """
        model = square_model()
        covered = 0
        ratios = []
        for seed in range(60):
            study = propagate(model, 64, design="sobol", seed=seed, replicates=8)
            output = study["y"]
            self.assertEqual(output.replicate_sizes, (64,) * 8)
            self.assertEqual(study.model_evaluations, 512)
            honest = output.mean_interval(0.95)
            naive = output.mean_estimate.confidence_interval
            covered += honest.lower <= 5.0 / 6.0 <= honest.upper
            ratios.append(
                (honest.upper - honest.lower) / (naive.upper - naive.lower)
            )
        self.assertGreaterEqual(covered, 52)  # binomial(60, 0.95) 0.1% quantile
        self.assertLess(max(ratios), 0.25)

    def test_exceedance_interval_with_and_without_spread(self) -> None:
        """Student-t over replicate fractions; Wilson fallback when flat."""
        model = square_model()
        study = propagate(model, 64, design="sobol", seed=2, replicates=6)
        output = study["y"]
        interval = output.exceedance_interval(1.2)
        self.assertIn("replicates", interval.method)
        # P(x1 + x2^2 > 1.2) = integral over x2 of max(0, x2^2 - 0.2) dx2
        root = math.sqrt(0.2)
        exact = (1.0 - root**3) / 3.0 - 0.2 * (1.0 - root)
        self.assertLess(interval.lower, exact)
        self.assertLess(exact, interval.upper)
        flat = output.exceedance_interval(5.0)
        self.assertIn("Wilson", flat.method)
        self.assertEqual(flat.lower, 0.0)

    def test_invalid_replicates(self) -> None:
        """One replicate cannot give an interval."""
        with self.assertRaises(ValueError):
            propagate(square_model(), 8, replicates=1)


class FailurePropagationTests(unittest.TestCase):
    """Partial outputs survive under the skip and record policies."""

    def test_skip_keeps_aligned_inputs_and_reports_failures(self) -> None:
        """Failed runs are dropped with a warning; inputs stay aligned."""
        model = normal_model(failing_tail, on_failure="skip")
        with self.assertWarnsRegex(UserWarning, "and were dropped"):
            study = propagate(model, 500, design="mc", seed=1)
        failures = len(study.failures)
        self.assertGreater(failures, 0)
        self.assertEqual(len(study.inputs) + failures, 500)
        self.assertEqual(study.input_column("z"), study["y"].values)
        self.assertAlmostEqual(study.failure_fraction, failures / 500)
        self.assertLessEqual(max(study["y"].values), 1.5)
        self.assertEqual(study.model_evaluations, 500 - failures)
        self.assertEqual(study.failed_evaluations, failures)
        self.assertEqual(pickle.loads(pickle.dumps(study)).failures, study.failures)
        # The surviving sample is truncated: its mean is biased low.
        self.assertLess(study["y"].mean, 0.0)

    def test_replicate_sizes_follow_failures(self) -> None:
        """Replicate blocks shrink by their own failures."""
        model = normal_model(failing_tail, on_failure="record")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study = propagate(model, 100, design="mc", seed=3, replicates=4)
        sizes = study["y"].replicate_sizes
        self.assertEqual(sum(sizes) + len(study.failures), 400)
        self.assertEqual(len(study["y"].replicate_means()), 4)
        self.assertTrue(all(f.inputs for f in study.failures))

    def test_propagate_samples_policy(self) -> None:
        """The explicit-sample propagator drops and lists failed runs."""
        samples = [{"z": value} for value in (0.1, 2.0, -0.4, 1.7, 0.6)]
        result = propagate_samples(failing_tail, samples, Y, on_failure="record")
        self.assertEqual(result.outputs["y"], (0.1, -0.4, 0.6))
        self.assertEqual([f.index for f in result.failures], [1, 3])
        self.assertEqual(result.model_evaluations, 3)
        with self.assertRaises(RuntimeError):
            propagate_samples(failing_tail, samples, Y)
        with self.assertRaisesRegex(ValueError, "only 1 of 3"):
            propagate_samples(
                failing_tail, [{"z": 2.0}, {"z": 3.0}, {"z": 0.0}], Y, on_failure="skip"
            )


if __name__ == "__main__":
    unittest.main()
