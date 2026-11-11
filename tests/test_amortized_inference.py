"""Tests for amortized neural posterior estimation (Gaussian, diagonal).

Oracle: the conjugate normal model ``theta ~ N(0, 1)``,
``x_i | theta ~ N(theta, 1)``, ``i = 1..4``, summarized by the sample mean
``s``, has posterior ``N(4 s / 5, 1 / 5)`` for every ``s``; a trained network
must reproduce it at several observations without retraining. A
two-parameter model with independent coordinates and different noise levels
has the closed-form posterior ``N(n s_j tau_j^-2 / (1 + n tau_j^-2), ...)``
per coordinate. Calibration is checked on fresh prior-predictive draws:
central 90% intervals of an exact posterior cover the generating parameter
with probability 0.9.
"""

from __future__ import annotations

import math
import random
import statistics
import unittest
from collections.abc import Sequence

from probstat.inference import (
    AmortizedGaussianPosterior,
    fit_amortized_gaussian_posterior,
)

N_OBS = 4


def _prior(rng: random.Random) -> list[float]:
    """Draw theta from N(0, 1)."""
    return [rng.gauss(0.0, 1.0)]


def _simulator(theta: Sequence[float], rng: random.Random) -> list[float]:
    """Return the mean of four N(theta, 1) observations."""
    return [statistics.fmean(rng.gauss(theta[0], 1.0) for _ in range(N_OBS))]


class ConjugateNormalTests(unittest.TestCase):
    """The amortized posterior matches the exact conjugate posterior."""

    posterior: AmortizedGaussianPosterior

    @classmethod
    def setUpClass(cls) -> None:
        """Train once on 2000 simulations."""
        cls.posterior = fit_amortized_gaussian_posterior(
            _prior,
            _simulator,
            2000,
            hidden_layers=(8,),
            iterations=1500,
            batch_size=32,
            learning_rate=0.02,
            seed=11,
        )

    def test_moments_match_closed_form_at_many_observations(self) -> None:
        """Mean 4s/5 and sd sqrt(1/5) for s across the prior predictive bulk."""
        exact_sd = math.sqrt(1.0 / (1.0 + N_OBS))
        for s in (-1.5, -0.75, 0.0, 0.75, 1.5):
            (mean,), (sd,) = self.posterior.moments([s])
            self.assertAlmostEqual(mean, N_OBS * s / (1.0 + N_OBS), delta=0.1)
            self.assertAlmostEqual(sd, exact_sd, delta=0.06)

    def test_credible_intervals_are_calibrated(self) -> None:
        """90% intervals cover fresh prior-predictive parameters about 90%."""
        rng = random.Random(99)
        trials = 1500
        hits = 0
        for _ in range(trials):
            theta = _prior(rng)
            ((lower, upper),) = self.posterior.credible_intervals(
                _simulator(theta, rng), 0.9
            )
            hits += lower <= theta[0] <= upper
        se = math.sqrt(0.9 * 0.1 / trials)
        self.assertAlmostEqual(hits / trials, 0.9, delta=4 * se + 0.02)

    def test_log_density_and_sampling_agree_with_moments(self) -> None:
        """Log q is the Gaussian log density; samples have the stated moments."""
        (mean,), (sd,) = self.posterior.moments([0.4])
        expected = -0.5 * math.log(2 * math.pi) - math.log(sd) - 0.5 * (
            (0.1 - mean) / sd
        ) ** 2
        self.assertAlmostEqual(self.posterior.log_density([0.1], [0.4]), expected)
        draws = [row[0] for row in self.posterior.sample([0.4], 4000, seed=3)]
        self.assertAlmostEqual(
            statistics.fmean(draws), mean, delta=4 * sd / math.sqrt(4000)
        )
        self.assertAlmostEqual(statistics.stdev(draws), sd, delta=0.05 * sd)
        self.assertEqual(self.posterior.dimension, 1)
        self.assertEqual(self.posterior.simulations, 2000)
        self.assertLess(
            statistics.fmean(self.posterior.losses[-100:]),
            statistics.fmean(self.posterior.losses[:100]),
        )


class TwoParameterTests(unittest.TestCase):
    """Independent coordinates with different precisions are both recovered."""

    def test_two_independent_parameters(self) -> None:
        """Posterior means and sds match the per-coordinate closed forms."""
        noise = (1.0, 0.5)

        def prior(rng: random.Random) -> list[float]:
            """Draw two independent N(0, 1) parameters."""
            return [rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)]

        def simulator(theta: Sequence[float], rng: random.Random) -> list[float]:
            """Return per-coordinate means of four noisy observations."""
            return [
                statistics.fmean(rng.gauss(theta[j], noise[j]) for _ in range(N_OBS))
                for j in range(2)
            ]

        posterior = fit_amortized_gaussian_posterior(
            prior,
            simulator,
            2000,
            hidden_layers=(6,),
            iterations=1000,
            batch_size=16,
            learning_rate=0.02,
            seed=5,
        )
        summary = (0.6, -0.8)
        means, sds = posterior.moments(summary)
        for j in range(2):
            precision = N_OBS / noise[j] ** 2
            self.assertAlmostEqual(
                means[j], precision * summary[j] / (1 + precision), delta=0.12
            )
            self.assertAlmostEqual(sds[j], math.sqrt(1 / (1 + precision)), delta=0.06)
        self.assertLess(sds[1], sds[0])


class InvalidInputTests(unittest.TestCase):
    """Invalid settings and simulator outputs raise."""

    def test_invalid_inputs(self) -> None:
        """Bad counts, widths, callables, and simulator outputs are rejected."""
        with self.assertRaises(TypeError):
            fit_amortized_gaussian_posterior(None, _simulator, 10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_amortized_gaussian_posterior(_prior, _simulator, 1)
        with self.assertRaises(ValueError):
            fit_amortized_gaussian_posterior(_prior, _simulator, 10, batch_size=0)
        with self.assertRaises(ValueError):
            fit_amortized_gaussian_posterior(_prior, _simulator, 10, hidden_layers=(0,))
        with self.assertRaises(ValueError):
            fit_amortized_gaussian_posterior(
                _prior, lambda theta, rng: [math.nan], 10, seed=1
            )
        with self.assertRaises(ValueError):
            fit_amortized_gaussian_posterior(
                _prior, lambda theta, rng: [0.0] * rng.randint(1, 3), 20, seed=1
            )
        posterior = fit_amortized_gaussian_posterior(
            _prior, _simulator, 20, iterations=5, seed=2
        )
        with self.assertRaises(ValueError):
            posterior.moments([0.0, 1.0])
        with self.assertRaises(ValueError):
            posterior.log_density([0.0, 1.0], [0.0])
        with self.assertRaises(ValueError):
            posterior.credible_intervals([0.0], 1.0)
        with self.assertRaises(ValueError):
            posterior.sample([0.0], -1)


if __name__ == "__main__":
    unittest.main()
