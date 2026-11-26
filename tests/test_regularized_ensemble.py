"""Tests of the generic ensemble Kalman update and Tikhonov EKI."""

from __future__ import annotations

import math
import random
import unittest

from probstat.inverse.linear_gaussian import linear_gaussian_posterior
from probstat.inverse.regularized_ensemble import (
    ensemble_kalman_update,
    tikhonov_ensemble_kalman_inversion,
)
from probstat.linalg import (
    DenseCovariance,
    DiagonalCovariance,
    matmul,
    matvec,
    solve_spd,
    transpose,
)


def _random_matrix(rng: random.Random, rows: int, cols: int) -> list[list[float]]:
    """Return a Gaussian random matrix."""
    return [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def _spd(rng: random.Random, size: int) -> tuple[tuple[float, ...], ...]:
    """Return a well-conditioned random SPD matrix."""
    base = _random_matrix(rng, size, size)
    gram = matmul(tuple(map(tuple, base)), transpose(tuple(map(tuple, base))))
    return tuple(
        tuple(gram[i][j] / size + (1.0 if i == j else 0.0) for j in range(size))
        for i in range(size)
    )


def _mean(rows):
    """Componentwise mean."""
    return [math.fsum(r[i] for r in rows) / len(rows) for i in range(len(rows[0]))]


def _cov(rows):
    """Sample covariance with divisor J - 1."""
    mu = _mean(rows)
    n = len(mu)
    return [
        [
            math.fsum((r[i] - mu[i]) * (r[j] - mu[j]) for r in rows) / (len(rows) - 1)
            for j in range(n)
        ]
        for i in range(n)
    ]


def _max_abs(a, b):
    """Largest absolute entry difference of two nested sequences."""
    if isinstance(a, (int, float)):
        return abs(a - b)
    return max(_max_abs(x, y) for x, y in zip(a, b, strict=True))


def _kalman_oracle(members, matrix, data, gamma, alpha):
    """Exact Kalman mean and covariance for the ensemble's own statistics."""
    n = len(members[0])
    mean = _mean(members)
    cxx = _cov(members)
    cxg = matmul(tuple(map(tuple, cxx)), transpose(tuple(map(tuple, matrix))))
    cgg = matmul(tuple(map(tuple, matrix)), cxg)
    m = len(data)
    system = tuple(
        tuple(cgg[i][j] + alpha * gamma[i][j] for j in range(m)) for i in range(m)
    )
    residual = [d - v for d, v in zip(data, matvec(matrix, tuple(mean)), strict=True)]
    # K r = Cxg S^-1 r
    solved = solve_spd(system, tuple(residual))
    new_mean = [
        mean[i] + math.fsum(cxg[i][k] * solved[k] for k in range(m)) for i in range(n)
    ]
    # K Cgx = Cxg S^-1 Cgx, column by column
    cgx = transpose(cxg)
    cols = [solve_spd(system, tuple(cgx[k][i] for k in range(m))) for i in range(n)]
    new_cov = [
        [
            cxx[i][j] - math.fsum(cxg[i][k] * cols[j][k] for k in range(m))
            for j in range(n)
        ]
        for i in range(n)
    ]
    return new_mean, new_cov


class ObservationEnsembleSolveTests(unittest.TestCase):
    """Observation-space and ensemble-space solves are the same update."""

    def test_agree_for_m_much_larger_than_j(self) -> None:
        """Check: agree for m much larger than j."""
        rng = random.Random(11)
        n, m, count = 5, 40, 8
        members = [tuple(rng.gauss(0, 1) for _ in range(n)) for _ in range(count)]
        preds = [tuple(rng.gauss(0, 2) for _ in range(m)) for _ in range(count)]
        data = tuple(rng.gauss(0, 1) for _ in range(m))
        for noise in (DenseCovariance(_spd(rng, m)), DiagonalCovariance([1.5] * m)):
            for perturbed in (True, False):
                a = ensemble_kalman_update(
                    members,
                    preds,
                    data,
                    noise,
                    perturbed=perturbed,
                    seed=3,
                    inflation=1.7,
                    solve="observation",
                )
                b = ensemble_kalman_update(
                    members,
                    preds,
                    data,
                    noise,
                    perturbed=perturbed,
                    seed=3,
                    inflation=1.7,
                    solve="ensemble",
                )
                scale = max(abs(v) for row in a.ensemble for v in row)
                self.assertLess(_max_abs(a.ensemble, b.ensemble) / scale, 1e-10)


class DeterministicSquareRootTests(unittest.TestCase):
    """The transform reproduces the Kalman mean and covariance exactly."""

    def _check(self, n: int, m: int, count: int, alpha: float, solve: str) -> None:
        rng = random.Random(100 + n + count)
        matrix = tuple(map(tuple, _random_matrix(rng, m, n)))
        gamma = _spd(rng, m)
        members = [tuple(rng.gauss(0, 1) for _ in range(n)) for _ in range(count)]
        preds = [matvec(matrix, x) for x in members]
        data = tuple(rng.gauss(0, 1) for _ in range(m))
        out = ensemble_kalman_update(
            members,
            preds,
            data,
            DenseCovariance(gamma),
            perturbed=False,
            inflation=alpha,
            solve=solve,
        )
        mean, cov = _kalman_oracle(members, matrix, data, gamma, alpha)
        self.assertLess(_max_abs(_mean(out.ensemble), mean), 1e-11)
        self.assertLess(_max_abs(out.mean, mean), 1e-11)
        self.assertLess(_max_abs(_cov(out.ensemble), cov), 1e-11)

    def test_exact_for_j_greater_than_n(self) -> None:
        """Check: exact for j greater than n."""
        self._check(3, 5, 8, 1.0, "observation")
        self._check(3, 5, 8, 2.5, "ensemble")

    def test_exact_for_rank_deficient_prior(self) -> None:
        """Check: exact for rank deficient prior."""
        # J - 1 < n: the sample prior covariance is singular; formulas still hold.
        self._check(6, 4, 3, 1.0, "observation")
        self._check(6, 4, 3, 0.4, "ensemble")


class StochasticUnbiasedTests(unittest.TestCase):
    """Over many perturbation seeds the updated mean is the Kalman mean."""

    def test_mean_matches_kalman_within_mcse(self) -> None:
        """Check: mean matches kalman within mcse."""
        rng = random.Random(5)
        n, m, count, alpha, trials = 2, 3, 10, 1.3, 400
        matrix = tuple(map(tuple, _random_matrix(rng, m, n)))
        gamma = _spd(rng, m)
        members = [tuple(rng.gauss(0, 1) for _ in range(n)) for _ in range(count)]
        preds = [matvec(matrix, x) for x in members]
        data = tuple(rng.gauss(0, 1) for _ in range(m))
        noise = DenseCovariance(gamma)
        exact, _ = _kalman_oracle(members, matrix, data, gamma, alpha)
        # Var of the member average = K (alpha Gamma) K' / J, with K the gain.
        cxx = _cov(members)
        cxg = matmul(tuple(map(tuple, cxx)), transpose(matrix))
        s = tuple(
            tuple(matmul(matrix, cxg)[i][j] + alpha * gamma[i][j] for j in range(m))
            for i in range(m)
        )
        gain_rows = [solve_spd(s, tuple(cxg[i])) for i in range(n)]  # rows of K
        sums = [0.0] * n
        for t in range(trials):
            out = ensemble_kalman_update(
                members, preds, data, noise, inflation=alpha, seed=1000 + t
            )
            for i in range(n):
                sums[i] += out.mean[i]
        for i in range(n):
            k = gain_rows[i]
            var = (
                alpha
                * math.fsum(
                    k[a] * gamma[a][b] * k[b] for a in range(m) for b in range(m)
                )
                / count
            )
            mcse = math.sqrt(var / trials)
            self.assertLess(abs(sums[i] / trials - exact[i]), 5.0 * mcse)


class TikhonovEKITests(unittest.TestCase):
    """TEKI behaves as a collapsing optimizer towards the Tikhonov solution."""

    def setUp(self) -> None:
        """Build the fixture."""
        rng = random.Random(21)
        self.n, self.m = 3, 6
        self.matrix = tuple(map(tuple, _random_matrix(rng, self.m, self.n)))
        self.truth = (0.8, -0.5, 0.3)
        noise_sd = 0.3
        self.data = tuple(
            v + rng.gauss(0.0, noise_sd) for v in matvec(self.matrix, self.truth)
        )
        self.noise = DiagonalCovariance([noise_sd**2] * self.m)
        self.mu = (0.0, 0.1, -0.2)
        self.c0 = DenseCovariance(_spd(rng, self.n))
        self.map = linear_gaussian_posterior(
            self.matrix,
            self.data,
            noise=self.noise,
            prior_mean=self.mu,
            prior_covariance=self.c0,
        )

    def _run(self, iterations: int, *, perturbed: bool, seed: int = 4, count: int = 50):
        return tikhonov_ensemble_kalman_inversion(
            lambda x: matvec(self.matrix, x),
            self.data,
            noise=self.noise,
            prior_mean=self.mu,
            prior_covariance=self.c0,
            ensemble_size=count,
            iterations=iterations,
            seed=seed,
            perturbed=perturbed,
        )

    def test_mean_approaches_map_and_spread_collapses(self) -> None:
        """Check: mean approaches map and spread collapses."""
        exact = self.map.mean
        short = self._run(3, perturbed=True)
        long = self._run(60, perturbed=True)
        err_short = _max_abs(short.mean, exact)
        err_long = _max_abs(long.mean, exact)
        # With perturbed data and J=50 the mean error is of the size of the
        # collapsed ensemble spread; 0.05 is ~10x the observed error and well
        # below the prior scale (~1).
        self.assertLess(err_long, 0.05)
        self.assertLess(err_long, err_short)
        # Documented collapse: the spread falls by orders of magnitude and ends
        # far below the exact posterior variance, so it is not a posterior.
        self.assertLess(long.spread_history[-1], 1e-2 * long.spread_history[0])
        post_trace = math.fsum(self._posterior_variance(i) for i in range(self.n))
        self.assertLess(long.spread_history[-1], 0.2 * post_trace)
        self.assertEqual(long.forward_evaluations, 50 * 61)
        self.assertEqual(len(long.mean_history), 61)
        self.assertEqual(long.evaluation_history[-1], long.forward_evaluations)

    def _posterior_variance(self, i: int) -> float:
        """Exact posterior variance of coordinate i via the normal equations."""
        gamma_inv = 1.0 / 0.3**2
        # Precision = A' Gamma^-1 A + C0^-1; solve for column i of the inverse.
        ata = matmul(transpose(self.matrix), self.matrix)
        prec_cols = []
        for k in range(self.n):
            unit = tuple(1.0 if j == k else 0.0 for j in range(self.n))
            prior_part = self.c0.solve(unit)
            prec_cols.append(
                tuple(gamma_inv * ata[r][k] + prior_part[r] for r in range(self.n))
            )
        prec = tuple(
            tuple(prec_cols[c][r] for c in range(self.n)) for r in range(self.n)
        )
        unit_i = tuple(1.0 if j == i else 0.0 for j in range(self.n))
        return solve_spd(prec, unit_i)[i]

    def test_deterministic_variant_converges_tightly(self) -> None:
        """Check: deterministic variant converges tightly."""
        exact = self.map.mean
        run = self._run(60, perturbed=False, count=12)
        self.assertLess(_max_abs(run.mean, exact), 1e-2)
        self.assertLess(run.misfit_history[-1], run.misfit_history[0])
        self.assertLess(run.spread_history[-1], 1e-3 * run.spread_history[0])

    def test_ensemble_space_solve_matches_observation_space(self) -> None:
        """Check: ensemble space solve matches observation space."""
        a = self._run(4, perturbed=True)
        b = tikhonov_ensemble_kalman_inversion(
            lambda x: matvec(self.matrix, x),
            self.data,
            noise=self.noise,
            prior_mean=self.mu,
            prior_covariance=self.c0,
            ensemble_size=50,
            iterations=4,
            seed=4,
            perturbed=True,
            solve="ensemble",
        )
        self.assertLess(_max_abs(a.mean, b.mean), 1e-8)

    def test_subspace_invariance_small_ensemble(self) -> None:
        """Check: subspace invariance small ensemble."""
        rng = random.Random(8)
        n, m, count = 6, 4, 4
        matrix = tuple(map(tuple, _random_matrix(rng, m, n)))
        data = tuple(rng.gauss(0, 1) for _ in range(m))

        def forward(x):
            return tuple(
                math.tanh(v)
                for v in matvec(matrix, x)  # nonlinear
            )

        for perturbed in (True, False):
            run = tikhonov_ensemble_kalman_inversion(
                forward,
                data,
                noise=DiagonalCovariance([0.2] * m),
                prior_mean=[0.0] * n,
                prior_covariance=DiagonalCovariance([1.0] * n),
                ensemble_size=count,
                iterations=6,
                seed=2,
                perturbed=perturbed,
            )
            mean0 = _mean(run.initial_ensemble)
            basis: list[list[float]] = []
            for member in run.initial_ensemble:
                v = [a - b for a, b in zip(member, mean0, strict=True)]
                for _ in range(2):  # re-orthogonalize
                    for q in basis:
                        c = math.fsum(a * b for a, b in zip(v, q, strict=True))
                        v = [a - c * b for a, b in zip(v, q, strict=True)]
                norm = math.sqrt(math.fsum(a * a for a in v))
                if norm > 1e-10:
                    basis.append([a / norm for a in v])
            self.assertEqual(len(basis), count - 1)
            for member in run.ensemble:
                v = [a - b for a, b in zip(member, mean0, strict=True)]
                for q in basis:
                    c = math.fsum(a * b for a, b in zip(v, q, strict=True))
                    v = [a - c * b for a, b in zip(v, q, strict=True)]
                self.assertLess(math.sqrt(math.fsum(a * a for a in v)), 1e-11)

    def test_seed_reproducibility(self) -> None:
        """Check: seed reproducibility."""
        a = self._run(5, perturbed=True, seed=9)
        b = self._run(5, perturbed=True, seed=9)
        c = self._run(5, perturbed=True, seed=10)
        self.assertEqual(a.ensemble, b.ensemble)
        self.assertNotEqual(a.ensemble, c.ensemble)

    def test_invalid_inputs(self) -> None:
        """Check: invalid inputs."""

        def fwd(x):
            return matvec(self.matrix, x)

        kwargs = dict(
            noise=self.noise,
            prior_mean=self.mu,
            prior_covariance=self.c0,
            ensemble_size=10,
            iterations=2,
            seed=1,
        )
        with self.assertRaises(ValueError):
            tikhonov_ensemble_kalman_inversion(
                fwd, self.data, **{**kwargs, "ensemble_size": 1}
            )
        with self.assertRaises(ValueError):
            tikhonov_ensemble_kalman_inversion(
                fwd, self.data, **{**kwargs, "iterations": 0}
            )
        with self.assertRaises(ValueError):
            tikhonov_ensemble_kalman_inversion(
                fwd, self.data, **{**kwargs, "noise": DiagonalCovariance([1.0] * 2)}
            )
        with self.assertRaises(ValueError):
            tikhonov_ensemble_kalman_inversion(
                fwd, (*self.data[:-1], math.nan), **kwargs
            )
        with self.assertRaises(TypeError):
            tikhonov_ensemble_kalman_inversion(
                fwd, self.data, **{**kwargs, "noise": 1.0}
            )
        with self.assertRaises(TypeError):
            tikhonov_ensemble_kalman_inversion(1.0, self.data, **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            tikhonov_ensemble_kalman_inversion(lambda x: (1.0,), self.data, **kwargs)


class UpdateValidationTests(unittest.TestCase):
    """Invalid inputs of the update raise."""

    def setUp(self) -> None:
        """Build the fixture."""
        self.members = [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5)]
        self.preds = [(0.0,), (1.0,), (0.5,)]
        self.data = (0.4,)
        self.noise = DiagonalCovariance([1.0])

    def test_rejections(self) -> None:
        """Check: rejections."""
        run = ensemble_kalman_update
        with self.assertRaises(ValueError):
            run(self.members[:1], self.preds[:1], self.data, self.noise)
        with self.assertRaises(ValueError):
            run(self.members, self.preds[:2], self.data, self.noise)
        with self.assertRaises(ValueError):
            run(self.members, self.preds, (0.1, 0.2), self.noise)
        with self.assertRaises(ValueError):
            run(self.members, self.preds, self.data, DiagonalCovariance([1.0, 1.0]))
        with self.assertRaises(ValueError):
            run(self.members, self.preds, self.data, self.noise, inflation=0.0)
        with self.assertRaises(ValueError):
            run(self.members, self.preds, self.data, self.noise, inflation=math.inf)
        with self.assertRaises(ValueError):
            run(self.members, self.preds, self.data, self.noise, solve="other")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            run([(math.nan, 0.0), (0.0, 1.0)], self.preds[:2], self.data, self.noise)
        with self.assertRaises(TypeError):
            run(self.members, self.preds, self.data, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            run(self.members, self.preds, self.data, self.noise, perturbed=1)  # type: ignore[arg-type]

    def test_identical_members_give_zero_update(self) -> None:
        """Check: identical members give zero update."""
        out = ensemble_kalman_update(
            [(1.0, 2.0)] * 3, [(0.5,)] * 3, self.data, self.noise, perturbed=False
        )
        self.assertEqual(out.ensemble, ((1.0, 2.0),) * 3)


if __name__ == "__main__":
    unittest.main()
