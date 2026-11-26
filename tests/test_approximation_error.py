"""Tests for the Bayesian approximation-error model."""

from __future__ import annotations

import math
import random
import unittest

from probstat.inverse.approximation_error import (
    estimate_approximation_error,
)
from probstat.inverse.heat_models import HeatEquation1D
from probstat.inverse.linear_gaussian import linear_gaussian_posterior
from probstat.linalg import matvec
from probstat.linalg.covariance_operators import DenseCovariance, DiagonalCovariance

M, N = 3, 4
A_FINE = (
    (1.0, 0.5, -0.3, 0.2),
    (0.1, 1.2, 0.4, -0.5),
    (-0.6, 0.3, 0.9, 0.7),
)
A_COARSE = (
    (0.8, 0.6, -0.1, 0.0),
    (0.3, 0.9, 0.5, -0.2),
    (-0.4, 0.1, 1.1, 0.4),
)
DELTA = tuple(tuple(A_FINE[i][j] - A_COARSE[i][j] for j in range(N)) for i in range(M))
PRIOR_VAR = (1.0, 0.5, 2.0, 1.5)


def _prior_draws(count: int, seed: int) -> list[tuple[float, ...]]:
    """Draw prior samples with independent N(0, PRIOR_VAR) entries."""
    rng = random.Random(seed)
    return [
        tuple(rng.gauss(0.0, math.sqrt(v)) for v in PRIOR_VAR) for _ in range(count)
    ]


def _exact_error_covariance() -> list[list[float]]:
    """Return D C0 D' for the diagonal prior."""
    return [
        [
            sum(DELTA[i][k] * PRIOR_VAR[k] * DELTA[j][k] for k in range(N))
            for j in range(M)
        ]
        for i in range(M)
    ]


def _fine(x: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(matvec(A_FINE, x))


def _coarse(x: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(matvec(A_COARSE, x))


class LinearGaussianPairTest(unittest.TestCase):
    """Closed-form error mean and covariance of a linear pair."""

    def test_sample_estimate_matches_exact_within_mcse(self) -> None:
        """Sample estimate matches exact within mcse."""
        model = estimate_approximation_error(
            _fine,
            _coarse,
            _prior_draws(2000, 11),
            validation_fraction=0.25,
            cross_covariance=True,
            seed=3,
        )
        sigma = _exact_error_covariance()
        k = model.training_size
        self.assertEqual(k, 1500)
        for i in range(M):
            # mean of e is exactly 0; MCSE = sqrt(Sigma_ii / K)
            self.assertLess(abs(model.mean[i]), 5 * math.sqrt(sigma[i][i] / k))
            for j in range(M):
                se = math.sqrt((sigma[i][i] * sigma[j][j] + sigma[i][j] ** 2) / (k - 1))
                self.assertLess(abs(model.covariance[i][j] - sigma[i][j]), 5 * se)
        self.assertEqual(model.sample_rank, M)
        self.assertEqual(model.shrinkage_policy, "none")
        # Cov(e, x) = D C0 exactly; independence in x is violated here
        assert model.cross_covariance is not None
        for i in range(M):
            for j in range(N):
                exact = DELTA[i][j] * PRIOR_VAR[j]
                se = math.sqrt(sigma[i][i] * PRIOR_VAR[j] / (k - 1)) * 1.5
                self.assertLess(abs(model.cross_covariance[i][j] - exact), 5 * se)
        self.assertEqual(model.fine_evaluations, 2000)
        self.assertEqual(model.coarse_evaluations, 2000)

    def test_heldout_statistics_match_chi_square(self) -> None:
        """Heldout statistics match chi square."""
        model = estimate_approximation_error(
            _fine, _coarse, _prior_draws(2000, 5), validation_fraction=0.25, seed=1
        )
        nv = model.validation_size
        self.assertEqual(nv, 500)
        assert model.validation_mahalanobis_mean is not None
        assert model.validation_exceedance_fraction is not None
        mcse = math.sqrt(2 * M / nv)
        self.assertLess(abs(model.validation_mahalanobis_mean - M), 5 * mcse)
        sd = math.sqrt(0.05 * 0.95 / nv)
        self.assertLess(abs(model.validation_exceedance_fraction - 0.05), 5 * sd)
        self.assertAlmostEqual(model.chi_square_quantile, 7.8147279, places=4)

    def test_combined_noise_and_corrected_data(self) -> None:
        """Combined noise and corrected data."""
        model = estimate_approximation_error(
            _fine, _coarse, _prior_draws(200, 2), seed=0
        )
        gamma = [[0.1 if i == j else 0.0 for j in range(M)] for i in range(M)]
        combined = model.combined_noise(gamma)
        self.assertIsInstance(combined, DenseCovariance)
        probe = (1.0, -2.0, 0.5)
        expected = [
            sum((gamma[i][j] + model.covariance[i][j]) * probe[j] for j in range(M))
            for i in range(M)
        ]
        for a, b in zip(combined.apply(probe), expected, strict=True):
            self.assertAlmostEqual(a, b, places=12)
        y = (1.0, 2.0, 3.0)
        for a, b, mu in zip(model.corrected_data(y), y, model.mean, strict=True):
            self.assertAlmostEqual(a, b - mu, places=14)
        with self.assertRaises(ValueError):
            model.combined_noise([[1.0]])
        with self.assertRaises(ValueError):
            model.corrected_data((1.0,))


class CoverageTest(unittest.TestCase):
    """Coverage of 90% intervals with and without the error model."""

    def test_error_model_improves_coverage_on_these_seeds(self) -> None:
        """Error model improves coverage on these seeds."""
        # Measured outcome for these seeds, not a universal guarantee.
        model = estimate_approximation_error(
            _fine, _coarse, _prior_draws(1500, 21), seed=4
        )
        sigma_noise = 0.05
        gamma_matrix = [
            [sigma_noise**2 if i == j else 0.0 for j in range(M)] for i in range(M)
        ]
        functional = (1.0, 1.0, 1.0, 1.0)
        z90 = 1.6448536269514722
        rng = random.Random(99)
        prior = DiagonalCovariance(PRIOR_VAR)
        naive_noise = DiagonalCovariance([sigma_noise**2] * M)
        combined = model.combined_noise(gamma_matrix)
        hits_naive = hits_ae = 0
        replicates = 50
        for _ in range(replicates):
            truth = tuple(rng.gauss(0.0, math.sqrt(v)) for v in PRIOR_VAR)
            clean = matvec(A_FINE, truth)  # data come from the FINE operator
            data = tuple(c + rng.gauss(0.0, sigma_noise) for c in clean)
            target = sum(f * t for f, t in zip(functional, truth, strict=True))
            for use_ae in (False, True):
                post = linear_gaussian_posterior(
                    A_COARSE,
                    model.corrected_data(data) if use_ae else data,
                    noise=combined if use_ae else naive_noise,
                    prior_mean=(0.0,) * N,
                    prior_covariance=prior,
                )
                centre = sum(f * c for f, c in zip(functional, post.mean, strict=True))
                spread = post.covariance_action(functional)
                var = sum(f * s for f, s in zip(functional, spread, strict=True))
                inside = abs(target - centre) <= z90 * math.sqrt(var)
                if use_ae:
                    hits_ae += inside
                else:
                    hits_naive += inside
        cover_naive = hits_naive / replicates
        cover_ae = hits_ae / replicates
        self.assertLess(abs(cover_ae - 0.9), abs(cover_naive - 0.9))


class HeatPairTest(unittest.TestCase):
    """Coarse and fine time stepping of the heat equation."""

    def test_heat_pair_with_shrinkage_is_positive_definite(self) -> None:
        """Heat pair with shrinkage is positive definite."""
        cells = 8
        fine_op = HeatEquation1D(cells, steps=40, final_time=0.05)
        coarse_op = HeatEquation1D(cells, steps=2, final_time=0.05)
        rng = random.Random(8)
        samples = [
            tuple(rng.gauss(0.0, 1.0) for _ in range(cells)) for _ in range(6)
        ]  # K = 6 < m = 8: full covariance impossible
        with self.assertRaises(ValueError):
            estimate_approximation_error(
                fine_op.matvec, coarse_op.matvec, samples, seed=1
            )
        for policy in ("ledoit_wolf", 0.3):
            model = estimate_approximation_error(
                fine_op.matvec, coarse_op.matvec, samples, shrinkage=policy, seed=1
            )
            self.assertLess(model.sample_rank, cells)
            self.assertGreater(model.shrinkage_intensity, 0.0)
            DenseCovariance(model.covariance)  # raises if not SPD
            self.assertEqual(model.fine_evaluations, 6)


class GuardTest(unittest.TestCase):
    """Validation split, size checks and invalid outputs."""

    def test_validation_split_is_disjoint_and_calls_are_single(self) -> None:
        """Validation split is disjoint and calls are single."""
        calls: list[tuple[float, ...]] = []

        def fine(x: tuple[float, ...]) -> tuple[float, ...]:
            calls.append(tuple(x))
            return _fine(x)

        samples = _prior_draws(60, 7)
        model = estimate_approximation_error(
            fine, _coarse, samples, validation_fraction=0.3, seed=9
        )
        train = set(model.training_indices)
        held = set(model.validation_indices)
        self.assertFalse(train & held)
        self.assertEqual(train | held, set(range(60)))
        self.assertEqual(len(held), 18)
        self.assertEqual(len(calls), 60)
        again = estimate_approximation_error(
            _fine, _coarse, samples, validation_fraction=0.3, seed=9
        )
        self.assertEqual(again.validation_indices, model.validation_indices)
        self.assertEqual(again.mean, model.mean)

    def test_training_mean_excludes_validation(self) -> None:
        """Training mean excludes validation."""
        samples = _prior_draws(40, 3)
        model = estimate_approximation_error(
            _fine, _coarse, samples, validation_fraction=0.5, seed=2
        )
        idx = model.training_indices
        errors = [
            tuple(
                a - b
                for a, b in zip(_fine(samples[i]), _coarse(samples[i]), strict=True)
            )
            for i in idx
        ]
        for j in range(M):
            ref = sum(e[j] for e in errors) / len(errors)
            self.assertAlmostEqual(model.mean[j], ref, places=12)

    def test_too_few_samples_without_shrinkage_raises(self) -> None:
        """Too few samples without shrinkage raises."""
        with self.assertRaises(ValueError):
            estimate_approximation_error(_fine, _coarse, _prior_draws(3, 1))
        with self.assertRaises(ValueError):
            estimate_approximation_error(
                _fine, _coarse, _prior_draws(10, 1), validation_fraction=0.8
            )

    def test_rank_deficient_error_requires_shrinkage(self) -> None:
        """Rank deficient error requires shrinkage."""

        def fine(x: tuple[float, ...]) -> tuple[float, ...]:
            return (x[0], x[1], x[2])

        def coarse(x: tuple[float, ...]) -> tuple[float, ...]:
            return (0.0, 0.0, 0.0)  # e = (x0, x1, x2): full rank

        def degenerate(x: tuple[float, ...]) -> tuple[float, ...]:
            return (x[0], x[0], x[0])  # rank-one error

        samples = _prior_draws(50, 4)
        estimate_approximation_error(fine, coarse, samples)
        with self.assertRaises(ValueError):
            estimate_approximation_error(degenerate, coarse, samples)
        model = estimate_approximation_error(
            degenerate, coarse, samples, shrinkage="ledoit_wolf"
        )
        self.assertEqual(model.sample_rank, 1)

    def test_nonfinite_and_inconsistent_outputs_raise(self) -> None:
        """Nonfinite and inconsistent outputs raise."""
        samples = _prior_draws(20, 6)

        def bad(x: tuple[float, ...]) -> tuple[float, ...]:
            return (float("nan"), 0.0, 0.0)

        def short(x: tuple[float, ...]) -> tuple[float, ...]:
            return (0.0, 0.0)

        with self.assertRaises(ValueError):
            estimate_approximation_error(bad, _coarse, samples)
        with self.assertRaises(ValueError):
            estimate_approximation_error(_fine, bad, samples)
        with self.assertRaises(ValueError):
            estimate_approximation_error(_fine, short, samples)

    def test_argument_validation(self) -> None:
        """Argument validation."""
        samples = _prior_draws(20, 6)
        with self.assertRaises(ValueError):
            estimate_approximation_error(_fine, _coarse, samples, shrinkage="oracle")
        with self.assertRaises(ValueError):
            estimate_approximation_error(_fine, _coarse, samples, shrinkage=1.5)
        with self.assertRaises(TypeError):
            estimate_approximation_error(_fine, _coarse, samples, shrinkage=[0.1])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            estimate_approximation_error(
                _fine, _coarse, samples, validation_fraction=1.0
            )
        with self.assertRaises(TypeError):
            estimate_approximation_error(_fine, 3, samples)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
