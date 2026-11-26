"""Tests of localization, adaptive inflation and the ETPF update."""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.inverse.ensemble_extended import (
    adaptive_inflation,
    ensemble_transform_particle_filter,
    estimate_inflation,
    gaspari_cohn_taper,
    localized_ensemble_kalman_update,
    square_map_reference,
)
from probstat.inverse.regularized_ensemble import ensemble_kalman_update
from probstat.linalg import (
    DenseCovariance,
    DiagonalCovariance,
    cholesky,
    matvec,
    solve_spd,
)


def _exp_cov(n: int, ell: float = 3.0):
    """Return the SPD exponential covariance ``exp(-|i-k|/ell)``."""
    return tuple(tuple(math.exp(-abs(i - k) / ell) for k in range(n)) for i in range(n))


def _draw(rng: random.Random, factor, count: int):
    """Draw ``count`` members ``factor z`` with standard normal ``z``."""
    n = len(factor)
    out = []
    for _ in range(count):
        z = [rng.gauss(0.0, 1.0) for _ in range(n)]
        out.append(
            tuple(sum(factor[i][k] * z[k] for k in range(i + 1)) for i in range(n))
        )
    return out


def _exact_gain(prior, variance: float):
    """Return ``C0 (C0 + s^2 I)^-1`` and ``C0 + s^2 I`` (identity observation)."""
    n = len(prior)
    system = tuple(
        tuple(prior[i][k] + (variance if i == k else 0.0) for k in range(n))
        for i in range(n)
    )
    return [list(solve_spd(system, tuple(prior[i]))) for i in range(n)], system


class TaperTest(unittest.TestCase):
    """Gaspari-Cohn taper values and validation."""

    def test_known_values(self) -> None:
        """rho(0)=1, rho(c)=5/24, rho(2c)=0 and continuity at c."""
        self.assertEqual(gaspari_cohn_taper(0.0, 3.0), 1.0)
        self.assertAlmostEqual(gaspari_cohn_taper(3.0, 3.0), 5.0 / 24.0, places=14)
        self.assertEqual(gaspari_cohn_taper(6.0, 3.0), 0.0)
        self.assertEqual(gaspari_cohn_taper(7.5, 3.0), 0.0)
        left = gaspari_cohn_taper(3.0 - 1e-9, 3.0)
        right = gaspari_cohn_taper(3.0 + 1e-9, 3.0)
        self.assertAlmostEqual(left, right, places=8)
        self.assertEqual(gaspari_cohn_taper(-1.0, 3.0), gaspari_cohn_taper(1.0, 3.0))

    def test_invalid(self) -> None:
        """Non-positive width and non-finite distance are rejected."""
        for width in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                gaspari_cohn_taper(1.0, width)
        with self.assertRaises(ValueError):
            gaspari_cohn_taper(math.inf, 1.0)
        with self.assertRaises(TypeError):
            gaspari_cohn_taper("1", 1.0)  # type: ignore[arg-type]


class LocalizedUpdateTest(unittest.TestCase):
    """The localized stochastic update."""

    def test_large_ensemble_matches_exact_kalman_gain(self) -> None:
        """Without a taper the gain converges to the exact Kalman gain."""
        n = 10
        prior = _exp_cov(n)
        factor = cholesky(prior)
        exact, _ = _exact_gain(prior, 0.25)
        noise = DiagonalCovariance([0.25] * n)
        errors = []
        for size in (100, 4000):
            members = _draw(random.Random(7), factor, size)
            update = localized_ensemble_kalman_update(
                members, members, (0.0,) * n, noise, seed=1
            )
            errors.append(
                max(
                    abs(update.gain[i][k] - exact[i][k])
                    for i in range(n)
                    for k in range(n)
                )
            )
            self.assertTrue(update.converged)
            self.assertFalse(update.localized)
        # Sampling error is O(J^-1/2): a 40-fold larger ensemble must give a
        # clearly smaller error, and the absolute error must be small.
        self.assertLess(errors[1], 0.5 * errors[0])
        self.assertLess(errors[1], 0.03)

    def test_unlocalized_equals_existing_stochastic_update(self) -> None:
        """Same seed and no taper reproduce ``ensemble_kalman_update``."""
        rng = random.Random(3)
        n, m, count = 4, 3, 9
        members = [tuple(rng.gauss(0, 1) for _ in range(n)) for _ in range(count)]
        h = tuple(tuple(rng.gauss(0, 1) for _ in range(n)) for _ in range(m))
        preds = [tuple(matvec(h, x)) for x in members]
        spd = ((2.0, 0.3, 0.1), (0.3, 1.5, 0.2), (0.1, 0.2, 1.0))
        for noise in (DenseCovariance(spd), DiagonalCovariance([0.5, 0.7, 0.9])):
            ours = localized_ensemble_kalman_update(
                members, preds, (0.3, -0.2, 0.9), noise, inflation=1.3, seed=11
            )
            theirs = ensemble_kalman_update(
                members, preds, (0.3, -0.2, 0.9), noise, inflation=1.3, seed=11
            )
            for a, b in zip(ours.ensemble, theirs.ensemble, strict=True):
                for u, v in zip(a, b, strict=True):
                    self.assertAlmostEqual(u, v, places=10)

    def test_localization_reduces_error_with_small_ensemble(self) -> None:
        """A taper removes spurious long-range correlation (J=10 << n=24)."""
        n, size, width = 24, 10, 3.0
        prior = _exp_cov(n)
        factor = cholesky(prior)
        variance = 0.25
        _, system = _exact_gain(prior, variance)
        noise = DiagonalCovariance([variance] * n)
        dist = [[float(abs(i - k)) for k in range(n)] for i in range(n)]
        far = [(i, k) for i in range(n) for k in range(n) if dist[i][k] >= 12]
        plain, tapered, spurious_plain, spurious_taper = [], [], [], []
        for seed in range(20):
            rng = random.Random(seed)
            truth = _draw(rng, factor, 1)[0]
            data = tuple(t + 0.5 * rng.gauss(0, 1) for t in truth)
            posterior = matvec(prior, solve_spd(system, data))
            members = _draw(rng, factor, size)
            free = localized_ensemble_kalman_update(
                members, members, data, noise, seed=seed
            )
            local = localized_ensemble_kalman_update(
                members,
                members,
                data,
                noise,
                parameter_observation_distances=dist,
                observation_distances=dist,
                half_width=width,
                seed=seed,
            )
            self.assertTrue(local.converged)
            self.assertTrue(local.localized)
            for update, bucket in ((free, plain), (local, tapered)):
                squares = (
                    (a - b) ** 2
                    for a, b in zip(update.analysis_mean, posterior, strict=True)
                )
                bucket.append(math.sqrt(math.fsum(squares) / n))
            # Gain entries beyond twice the half-width: the exact value is ~0.
            spurious_plain.append(max(abs(free.gain[i][k]) for i, k in far))
            spurious_taper.append(max(abs(local.gain[i][k]) for i, k in far))
        # The tapered cross-covariance is exactly zero beyond 2c; the gain
        # keeps tiny leakage through the dense inverse. Compare to untapered.
        self.assertLess(max(spurious_taper), 0.01 * min(spurious_plain))
        wins = sum(1 for a, b in zip(plain, tapered, strict=True) if b < a)
        self.assertEqual(wins, 20)
        self.assertLess(math.fsum(tapered) / 20, 0.5 * math.fsum(plain) / 20)

    def test_zero_spread_gives_zero_gain(self) -> None:
        """Identical members have no covariance: the ensemble is unchanged."""
        members = [(1.0, 2.0)] * 4
        update = localized_ensemble_kalman_update(
            members, members, (0.0, 0.0), DiagonalCovariance([1.0, 1.0]), seed=2
        )
        self.assertTrue(all(v == 0.0 for row in update.gain for v in row))
        self.assertEqual(update.ensemble, tuple(members))

    def test_seed_reproducibility(self) -> None:
        """Equal seeds give equal ensembles, different seeds do not."""
        rng = random.Random(1)
        members = [tuple(rng.gauss(0, 1) for _ in range(3)) for _ in range(6)]
        noise = DiagonalCovariance([1.0, 1.0, 1.0])
        data = (0.1, 0.2, 0.3)
        a = localized_ensemble_kalman_update(members, members, data, noise, seed=5)
        b = localized_ensemble_kalman_update(members, members, data, noise, seed=5)
        c = localized_ensemble_kalman_update(members, members, data, noise, seed=6)
        self.assertEqual(a.ensemble, b.ensemble)
        self.assertNotEqual(a.ensemble, c.ensemble)

    def test_indefinite_taper_is_reported(self) -> None:
        """A non-PSD taper matrix is reported through the status."""
        # Distances 0,0,large give rho = [[1,1,1],[1,1,0],[1,0,1]], not PSD.
        dist = [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 10.0, 0.0]]
        members = [
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, -1.0),
            (2.0, 2.01, 1.99),
            (-2.0, -1.99, -2.01),
        ]
        update = localized_ensemble_kalman_update(
            members,
            members,
            (0.0, 0.0, 0.0),
            DiagonalCovariance([1e-9] * 3),
            parameter_observation_distances=dist,
            observation_distances=dist,
            half_width=1.0,
            seed=1,
        )
        self.assertFalse(update.converged)
        self.assertEqual(update.stop_reason, "tapered_system_not_spd")
        self.assertEqual(update.ensemble, tuple(members))

    def test_invalid_input(self) -> None:
        """Missing or malformed distances and bad scalars are rejected."""
        members = [(0.0, 1.0), (1.0, 0.0), (2.0, 2.0)]
        noise = DiagonalCovariance([1.0, 1.0])
        data = (0.0, 0.0)
        good = [[0.0, 1.0], [1.0, 0.0]]
        with self.assertRaises(ValueError):
            localized_ensemble_kalman_update(
                members, members, data, noise, half_width=1.0
            )
        with self.assertRaises(ValueError):
            localized_ensemble_kalman_update(
                members,
                members,
                data,
                noise,
                parameter_observation_distances=[[0.0]],
                observation_distances=good,
                half_width=1.0,
            )
        with self.assertRaises(ValueError):
            localized_ensemble_kalman_update(
                members,
                members,
                data,
                noise,
                parameter_observation_distances=good,
                observation_distances=[[0.0, 1.0], [2.0, 0.0]],
                half_width=1.0,
            )
        with self.assertRaises(ValueError):
            localized_ensemble_kalman_update(
                members,
                members,
                data,
                noise,
                parameter_observation_distances=[[0.0, -1.0], [1.0, 0.0]],
                observation_distances=good,
                half_width=1.0,
            )
        with self.assertRaises(ValueError):
            localized_ensemble_kalman_update(
                members, members, data, noise, inflation=0.0
            )
        with self.assertRaises(ValueError):
            localized_ensemble_kalman_update(members[:1], members[:1], data, noise)
        with self.assertRaises(TypeError):
            localized_ensemble_kalman_update(
                members,
                members,
                data,
                [[1.0, 0.0], [0.0, 1.0]],  # type: ignore[arg-type]
            )


class InflationTest(unittest.TestCase):
    """The innovation-based inflation estimator."""

    def test_recovers_known_underdispersion(self) -> None:
        """E[lambda_hat] = lambda within an MCSE band (lambda=2, J=100, m=8)."""
        m, size, reps, lam = 8, 100, 300, 2.0
        noise = DiagonalCovariance([1.0] * m)
        values = []
        for rep in range(reps):
            rng = random.Random(1000 + rep)
            truth = [rng.gauss(0.0, 1.0) for _ in range(m)]
            data = tuple(t + rng.gauss(0.0, 1.0) for t in truth)
            scale = 1.0 / math.sqrt(lam)
            members = [
                tuple(rng.gauss(0.0, scale) for _ in range(m)) for _ in range(size)
            ]
            report = estimate_inflation(
                members, members, data, noise, lower=0.01, upper=100.0
            )
            values.append(report.raw)
        mean = math.fsum(values) / reps
        se = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (reps - 1) / reps)
        # The ratio has a relative bias of order 1/J (about 1%); allow 4 se
        # plus that bias.
        self.assertLess(abs(mean - lam), 4.0 * se + 0.02 * lam)
        self.assertLess(se, 0.2)

    def test_calibrated_ensemble_gives_one(self) -> None:
        """A correctly dispersed prior gives lambda_hat ~ 1."""
        m, size, reps = 8, 100, 300
        noise = DiagonalCovariance([1.0] * m)
        values = []
        for rep in range(reps):
            rng = random.Random(5000 + rep)
            data = tuple(rng.gauss(0.0, math.sqrt(2.0)) for _ in range(m))
            members = [
                tuple(rng.gauss(0.0, 1.0) for _ in range(m)) for _ in range(size)
            ]
            report = estimate_inflation(members, members, data, noise, lower=0.01)
            values.append(report.raw)
        mean = math.fsum(values) / reps
        se = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (reps - 1) / reps)
        self.assertLess(abs(mean - 1.0), 4.0 * se + 0.02)

    def test_closed_form_and_bounds(self) -> None:
        """Hand-computed estimate, clipping flags and exact covariance scaling."""
        members = [(1.0,), (-1.0,), (3.0,), (-3.0,)]
        noise = DiagonalCovariance([4.0])
        # mean 0, sample variance (1+1+9+9)/3 = 20/3, whitened by 4 -> 5/3.
        report = estimate_inflation(
            members, members, (6.0,), noise, lower=1.0, upper=50.0
        )
        self.assertAlmostEqual(report.spread_trace, 5.0 / 3.0, places=13)
        self.assertAlmostEqual(report.innovation_norm_squared, 9.0, places=13)
        self.assertAlmostEqual(report.raw, (9.0 - 1.0) / (5.0 / 3.0), places=13)
        self.assertFalse(report.clipped)
        capped = estimate_inflation(
            members, members, (6.0,), noise, lower=1.0, upper=3.0
        )
        self.assertEqual(capped.factor, 3.0)
        self.assertTrue(capped.clipped)
        self.assertAlmostEqual(capped.raw, 4.8, places=13)
        floor = estimate_inflation(
            members, members, (0.0,), noise, lower=1.0, upper=3.0
        )
        self.assertEqual(floor.factor, 1.0)
        self.assertLess(floor.raw, 1.0)
        # Inflation scales the sample variance by lambda exactly, mean unchanged.
        result = adaptive_inflation(
            members, members, (6.0,), noise, lower=1.0, upper=50.0
        )
        lam = result.report.factor
        var = math.fsum(x[0] ** 2 for x in result.ensemble) / 3.0
        self.assertAlmostEqual(var, lam * 20.0 / 3.0, places=11)
        self.assertAlmostEqual(math.fsum(x[0] for x in result.ensemble), 0.0, places=12)

    def test_failures(self) -> None:
        """Zero spread, bad bounds and bad types raise."""
        same = [(1.0,)] * 3
        noise = DiagonalCovariance([1.0])
        with self.assertRaises(ValueError):
            estimate_inflation(same, same, (0.0,), noise)
        diverse = [(0.0,), (1.0,), (2.0,)]
        with self.assertRaises(ValueError):
            estimate_inflation(diverse, diverse, (0.0,), noise, lower=2.0, upper=1.0)
        with self.assertRaises(ValueError):
            estimate_inflation(diverse, diverse, (0.0,), noise, lower=0.0)
        with self.assertRaises(TypeError):
            estimate_inflation(diverse, diverse, (0.0,), 1.0)  # type: ignore[arg-type]


def _nw_corner_1d(weights, count):
    """Return the north-west-corner coupling of sorted 1D points (OT oracle)."""
    plan = [[0.0] * count for _ in range(count)]
    i = j = 0
    left = list(weights)
    need = [1.0 / count] * count
    while i < count and j < count:
        mass = min(left[i], need[j])
        plan[i][j] += mass
        left[i] -= mass
        need[j] -= mass
        if left[i] <= 1e-15:
            i += 1
        elif need[j] <= 1e-15:
            j += 1
    return plan


def _cost_1d(plan, points):
    """Return ``sum_ij plan_ij (x_i - x_j)^2`` for scalar points."""
    count = len(points)
    return math.fsum(
        plan[i][j] * (points[i][0] - points[j][0]) ** 2
        for i in range(count)
        for j in range(count)
    )


class ETPFTest(unittest.TestCase):
    """The ensemble transform particle filter."""

    def test_uniform_weights_are_identity(self) -> None:
        """Equal predictions give equal weights and the identity transform."""
        members = [(0.0, 1.0), (2.0, -1.0), (1.0, 4.0)]
        preds = [(5.0,)] * 3
        result = ensemble_transform_particle_filter(
            members, preds, (5.0,), DiagonalCovariance([1.0])
        )
        for w in result.weights:
            self.assertAlmostEqual(w, 1.0 / 3.0, places=14)
        for a, b in zip(result.ensemble, members, strict=True):
            for u, v in zip(a, b, strict=True):
                self.assertAlmostEqual(u, v, places=12)
        self.assertAlmostEqual(result.transport_cost, 0.0, places=12)
        self.assertEqual(result.stop_reason, "ok")

    def test_one_dimensional_coupling_matches_monotone_oracle(self) -> None:
        """Marginals, mean identity and cost agree with the sorted-order oracle."""
        rng = random.Random(21)
        count = 12
        members = sorted((rng.gauss(0.0, 1.0),) for _ in range(count))
        noise = DiagonalCovariance([0.5])
        result = ensemble_transform_particle_filter(members, members, (0.7,), noise)
        weights = result.weights
        raw = [math.exp(-0.5 * (0.7 - p[0]) ** 2 / 0.5) for p in members]
        total = math.fsum(raw)
        for w, r in zip(weights, raw, strict=True):
            self.assertAlmostEqual(w, r / total, places=13)
        plan = result.coupling
        for i in range(count):
            self.assertAlmostEqual(math.fsum(plan[i]), weights[i], places=12)
            column = math.fsum(plan[k][i] for k in range(count))
            self.assertAlmostEqual(column, 1.0 / count, places=12)
        oracle = _nw_corner_1d(weights, count)
        self.assertAlmostEqual(
            result.transport_cost, _cost_1d(oracle, members), places=10
        )
        # Mean identity: exact up to rounding.
        weighted = math.fsum(w * m[0] for w, m in zip(weights, members, strict=True))
        self.assertAlmostEqual(result.mean[0], weighted, places=13)
        self.assertAlmostEqual(result.weighted_mean[0], weighted, places=13)
        # In 1D the optimal plan is monotone, so is its barycentric map.
        values = [x[0] for x in result.ensemble]
        self.assertEqual(values, sorted(values))

    def test_small_enumerated_optimum(self) -> None:
        """J=3, points 0, 1, 10 with weights (1/2, 1/2, 0), solved by hand."""
        members = [(0.0,), (1.0,), (10.0,)]
        # The third prediction is so far from the data that its weight is 0.
        preds = [(0.0,), (0.0,), (1.0e4,)]
        result = ensemble_transform_particle_filter(
            members, preds, (0.0,), DiagonalCovariance([1.0])
        )
        self.assertAlmostEqual(result.weights[0], 0.5, places=12)
        self.assertEqual(result.weights[2], 0.0)
        # Columns need 1/3 each: x=0 sends 1/3 to column 0 and 1/6 to column 1,
        # x=1 sends 1/6 to column 1 and 1/3 to column 2. Cost
        # (1/6)(0-1)^2 + (1/3)(1-10)^2 = 1/6 + 27 (hand value).
        self.assertAlmostEqual(result.transport_cost, 1.0 / 6.0 + 27.0, places=9)
        self.assertAlmostEqual(
            result.transport_cost,
            _cost_1d(_nw_corner_1d(result.weights, 3), members),
            places=9,
        )
        self.assertAlmostEqual(result.mean[0], 0.5, places=12)

    def test_spread_contraction(self) -> None:
        """Transformed variance is below the weighted variance (finite-J bias)."""
        rng = random.Random(8)
        members = [(rng.gauss(0.0, 1.0),) for _ in range(30)]
        result = ensemble_transform_particle_filter(
            members, members, (0.8,), DiagonalCovariance([0.3])
        )
        weighted = result.weighted_covariance[0][0]
        self.assertLessEqual(result.covariance[0][0], weighted + 1e-12)
        self.assertLess(result.covariance[0][0], 0.98 * weighted)

    def test_linear_gaussian_one_dimension(self) -> None:
        """Weighted and transformed means match the exact posterior mean."""
        prior_sd, noise_sd, data, count = 1.0, 0.8, 1.2, 60
        post_var = 1.0 / (1.0 / prior_sd**2 + 1.0 / noise_sd**2)
        post_mean = post_var * data / noise_sd**2
        rng = random.Random(4)
        members = [(rng.gauss(0.0, prior_sd),) for _ in range(count)]
        result = ensemble_transform_particle_filter(
            members, members, (data,), DiagonalCovariance([noise_sd**2])
        )
        # Importance-sampling standard error of the mean: sd / sqrt(ESS).
        band = 4.0 * math.sqrt(post_var / result.effective_sample_size)
        self.assertLess(abs(result.weighted_mean[0] - post_mean), band)
        self.assertLess(abs(result.mean[0] - post_mean), band)
        self.assertLess(
            abs(result.weighted_covariance[0][0] - post_var), 0.4 * post_var
        )
        self.assertLessEqual(
            result.covariance[0][0], result.weighted_covariance[0][0] + 1e-12
        )

    def test_weight_collapse_is_reported(self) -> None:
        """Data far from all predictions collapse the weights."""
        members = [(0.0,), (1.0,), (2.0,), (3.0,)]
        result = ensemble_transform_particle_filter(
            members, members, (200.0,), DiagonalCovariance([0.01])
        )
        self.assertLess(result.effective_sample_size, 1.5)
        self.assertEqual(result.stop_reason, "weight_collapse")
        self.assertFalse(result.converged)

    def test_failures(self) -> None:
        """Bad shapes and types are rejected."""
        members = [(0.0,), (1.0,)]
        noise = DiagonalCovariance([1.0])
        with self.assertRaises(ValueError):
            ensemble_transform_particle_filter(members[:1], members[:1], (0.0,), noise)
        with self.assertRaises(ValueError):
            ensemble_transform_particle_filter(members, members, (0.0, 1.0), noise)
        with self.assertRaises(TypeError):
            ensemble_transform_particle_filter(
                members,
                members,
                (0.0,),
                None,  # type: ignore[arg-type]
            )


class SquareMapReferenceTest(unittest.TestCase):
    """The quadrature reference of ``y = x^2 + noise``."""

    def test_symmetric_prior_is_symmetric(self) -> None:
        """A prior centred at zero gives mass 1/2 and mean 0."""
        ref = square_map_reference(0.0, 1.0, 1.0, 0.2)
        self.assertAlmostEqual(ref.mass_positive, 0.5, places=12)
        self.assertAlmostEqual(ref.mean, 0.0, places=12)
        self.assertLess(ref.valley_mass, 1e-3)

    def test_resolution_independent(self) -> None:
        """Doubling the nodes changes the moments by less than 1e-6."""
        a = square_map_reference(0.5, 1.0, 1.0, 0.2, nodes=20001)
        b = square_map_reference(0.5, 1.0, 1.0, 0.2, nodes=40001)
        self.assertAlmostEqual(a.mass_positive, b.mass_positive, delta=1e-6)
        self.assertAlmostEqual(a.mean, b.mean, delta=1e-6)
        self.assertAlmostEqual(a.variance, b.variance, delta=1e-6)

    def test_gaussian_limit(self) -> None:
        """Huge noise returns the prior moments."""
        ref = square_map_reference(0.5, 1.0, 0.0, 1e6)
        self.assertAlmostEqual(ref.mean, 0.5, places=6)
        self.assertAlmostEqual(ref.variance, 1.0, places=5)
        self.assertEqual(ref.valley_mass, 0.0)

    def test_invalid(self) -> None:
        """Invalid scales and node counts are rejected."""
        with self.assertRaises(ValueError):
            square_map_reference(0.0, 0.0, 1.0, 0.2)
        with self.assertRaises(ValueError):
            square_map_reference(0.0, 1.0, 1.0, -0.2)
        with self.assertRaises(ValueError):
            square_map_reference(0.0, 1.0, 1.0, 0.2, nodes=2)


def _compare_on_square_map(count: int, seeds: range) -> dict[str, float]:
    """Return seed-averaged ETPF and EnKF statistics for ``y = x^2 + noise``."""
    noise = DiagonalCovariance([0.04])
    sums = {k: 0.0 for k in ("to", "tm", "tv", "tg", "eo", "em", "ev", "eg")}
    for seed in seeds:
        rng = random.Random(seed)
        members = [(rng.gauss(0.5, 1.0),) for _ in range(count)]
        preds = [(x[0] ** 2,) for x in members]
        etpf = ensemble_transform_particle_filter(members, preds, (1.0,), noise)
        enkf = ensemble_kalman_update(members, preds, (1.0,), noise, seed=seed)
        for tag, ens in (("t", etpf.ensemble), ("e", enkf.ensemble)):
            values = [x[0] for x in ens]
            mean = math.fsum(values) / count
            sums[tag + "o"] += sum(v > 0.0 for v in values) / count
            sums[tag + "m"] += mean
            sums[tag + "v"] += math.fsum((v - mean) ** 2 for v in values) / (count - 1)
            sums[tag + "g"] += sum(abs(v) < 0.5 for v in values) / count
    return {k: v / len(seeds) for k, v in sums.items()}


class MultimodalComparisonTest(unittest.TestCase):
    """EnKF-type update, ETPF and quadrature on a bimodal posterior."""

    def _check(self, stats: dict[str, float]) -> None:
        """Assert the structural findings."""
        ref = square_map_reference(0.5, 1.0, 1.0, 0.2)
        # The reference is genuinely bimodal: almost no mass between modes.
        self.assertLess(ref.valley_mass, 1e-3)
        # The stochastic EnKF is a linear update built for a Gaussian relation:
        # a large share of its members sit between the modes (|x| < 1/2) where
        # the posterior has essentially no mass.
        self.assertGreater(stats["eg"], 0.15)
        # The ETPF reweights and transports, so it keeps particles on the modes.
        self.assertLess(stats["tg"], 0.05)
        self.assertLess(stats["tg"], 0.25 * stats["eg"])
        # ETPF is not exact at this size: occupation, mean and variance are
        # within generous bands of the quadrature reference.
        self.assertLess(abs(stats["to"] - ref.mass_positive), 0.15)
        self.assertLess(abs(stats["tm"] - ref.mean), 0.3)
        self.assertLess(abs(stats["tv"] - ref.variance), 0.3)

    def test_fast(self) -> None:
        """J=60 over six predeclared seeds."""
        self._check(_compare_on_square_map(60, range(6)))

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_slow_tier(self) -> None:
        """J=100 over eight predeclared seeds; prints measured numbers."""
        stats = _compare_on_square_map(100, range(8))
        print("square-map reference", square_map_reference(0.5, 1.0, 1.0, 0.2))
        print("J=100, 8 seeds", {k: round(v, 4) for k, v in stats.items()})
        self._check(stats)


if __name__ == "__main__":
    unittest.main()
