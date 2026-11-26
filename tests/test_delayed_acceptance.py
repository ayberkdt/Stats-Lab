"""Tests for exact-target delayed-acceptance Metropolis (EQ-INF-DA-001..005)."""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable

from probstat.inference.delayed_acceptance import (
    DelayedAcceptanceResult,
    delayed_acceptance_metropolis,
)


def _da_matrix(
    pi: list[float], sur: list[float], q: list[list[float]]
) -> list[list[float]]:
    """Independent delayed-acceptance transition matrix on a finite space."""
    n = len(pi)
    p = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a1 = min(1.0, sur[j] * q[j][i] / (sur[i] * q[i][j]))
            a2 = min(1.0, pi[j] * sur[i] / (pi[i] * sur[j]))
            p[i][j] = q[i][j] * a1 * a2
        p[i][i] = 1.0 - sum(p[i])
    return p


def _batch_mcse(values: list[float], batches: int = 50) -> float:
    """Batch-means standard error of the mean."""
    size = len(values) // batches
    means = [
        math.fsum(values[b * size : (b + 1) * size]) / size for b in range(batches)
    ]
    center = math.fsum(means) / batches
    var = math.fsum((m - center) ** 2 for m in means) / (batches - 1)
    return math.sqrt(var / batches)


def _mean(values: list[float]) -> float:
    """Arithmetic mean."""
    return math.fsum(values) / len(values)


def _gauss_logpdf(mean: float, sd: float) -> Callable[[tuple[float, ...]], float]:
    """Return a 1-D Gaussian log-density callable on vectors."""

    def f(x: tuple[float, ...]) -> float:
        """Evaluate the log-density."""
        return -0.5 * ((x[0] - mean) / sd) ** 2 - math.log(sd)

    return f


class FiniteStateDetailedBalance(unittest.TestCase):
    """Analytic kernel checks on a 4-state space."""

    pi = [0.1, 0.4, 0.3, 0.2]
    sur = [0.5, 0.1, 0.3, 0.1]  # deliberately wrong, positive
    q = [
        [0.0, 0.5, 0.25, 0.25],
        [0.5, 0.0, 0.25, 0.25],
        [0.25, 0.25, 0.0, 0.5],
        [0.25, 0.25, 0.5, 0.0],
    ]

    def test_detailed_balance_and_stationarity(self) -> None:
        """pi_i P_ij = pi_j P_ji and pi P = pi; the surrogate alone fails."""
        p = _da_matrix(self.pi, self.sur, self.q)
        for i in range(4):
            self.assertAlmostEqual(math.fsum(p[i]), 1.0, places=14)
            for j in range(4):
                self.assertAlmostEqual(
                    self.pi[i] * p[i][j], self.pi[j] * p[j][i], places=14
                )
        for j in range(4):
            col = math.fsum(self.pi[i] * p[i][j] for i in range(4))
            self.assertAlmostEqual(col, self.pi[j], places=14)
        # MH on the surrogate alone has the surrogate, not pi, as invariant law
        s = [x / math.fsum(self.sur) for x in self.sur]
        p_s = _da_matrix(s, s, self.q)
        col = math.fsum(self.pi[i] * p_s[i][0] for i in range(4))
        self.assertGreater(abs(col - self.pi[0]), 0.01)

    def test_implementation_on_piecewise_constant_cells(self) -> None:
        """Cell frequencies of the sampler match pi on a 3-cell density."""
        weights = [0.2, 0.5, 0.3]
        sur_w = [0.6, 0.1, 0.3]

        def cell(x: tuple[float, ...], w: list[float]) -> float:
            """Log piecewise-constant density on [0, 3)."""
            if not 0.0 <= x[0] < 3.0:
                return -math.inf
            return math.log(w[int(x[0])])

        res = delayed_acceptance_metropolis(
            lambda x: cell(x, weights),
            lambda x: cell(x, sur_w),
            (1.5,),
            step=1.0,
            samples=30000,
            burn_in=500,
            seed=11,
        )
        idx = [int(s[0]) for s in res.samples]
        for k, w in enumerate(weights):
            ind = [1.0 if i == k else 0.0 for i in idx]
            self.assertLess(abs(_mean(ind) - w), 4.5 * _batch_mcse(ind))


class GaussianExactness(unittest.TestCase):
    """DA recovers the target; surrogate-only MH is biased."""

    def test_da_vs_surrogate_only(self) -> None:
        """DA moments inside MCSE bands; surrogate-only mean near 0.7."""
        target = _gauss_logpdf(1.0, 0.5)
        sur = _gauss_logpdf(0.7, 0.8)
        res = delayed_acceptance_metropolis(
            target, sur, (0.0,), step=1.0, samples=20000, burn_in=500, seed=3
        )
        xs = [s[0] for s in res.samples]
        self.assertLess(abs(_mean(xs) - 1.0), 4.5 * _batch_mcse(xs))
        sq = [(x - 1.0) ** 2 for x in xs]
        self.assertLess(abs(_mean(sq) - 0.25), 4.5 * _batch_mcse(sq))
        self.assertTrue(res.exact_target)
        self.assertLess(res.exact_evaluations, res.surrogate_evaluations)
        self.assertGreater(res.ess_per_exact_evaluation, 0.0)
        # plain RW-MH on the surrogate alone (target = surrogate)
        bad = delayed_acceptance_metropolis(
            sur, sur, (0.0,), step=1.0, samples=20000, burn_in=500, seed=3
        )
        ys = [s[0] for s in bad.samples]
        self.assertLess(abs(_mean(ys) - 0.7), 4.5 * _batch_mcse(ys))
        self.assertGreater(abs(_mean(ys) - 1.0), 0.15)

    def test_pcn_two_dimensional(self) -> None:
        """Run pCN with N(0, I) prior: posterior N(0.8 y, 0.2 I) for sd 0.5."""
        y = (1.0, -0.5)

        def lik(x: tuple[float, ...], c: tuple[float, ...], sd: float) -> float:
            """Gaussian log-likelihood."""
            return (
                -0.5
                * math.fsum((a - b) ** 2 for a, b in zip(x, c, strict=True))
                / sd**2
            )

        res = delayed_acceptance_metropolis(
            lambda x: lik(x, y, 0.5),
            lambda x: lik(x, (0.7, -0.2), 0.7),
            (0.0, 0.0),
            proposal="pcn",
            step=0.6,
            log_prior="standard_normal",
            samples=20000,
            burn_in=500,
            seed=5,
        )
        for j in range(2):
            xs = [s[j] for s in res.samples]
            m = 0.8 * y[j]
            self.assertLess(abs(_mean(xs) - m), 4.5 * _batch_mcse(xs))
            sq = [(x - m) ** 2 for x in xs]
            self.assertLess(abs(_mean(sq) - 0.2), 4.5 * _batch_mcse(sq))


class Accounting(unittest.TestCase):
    """Call counts, reproducibility and support policy."""

    def test_stage1_reject_makes_no_exact_call(self) -> None:
        """exact_evaluations == stage-1 acceptances + 1, counted externally."""
        calls = {"exact": 0, "sur": 0}
        s1 = _gauss_logpdf(0.7, 0.8)
        t1 = _gauss_logpdf(1.0, 0.5)

        def exact(x: tuple[float, ...]) -> float:
            """Count and evaluate the exact target."""
            calls["exact"] += 1
            return t1(x)

        def sur(x: tuple[float, ...]) -> float:
            """Count and evaluate the surrogate."""
            calls["sur"] += 1
            return s1(x)

        res = delayed_acceptance_metropolis(
            exact, sur, (1.0,), step=3.0, samples=2000, burn_in=0, seed=9
        )
        self.assertEqual(calls["exact"], res.exact_evaluations)
        self.assertEqual(calls["sur"], res.surrogate_evaluations)
        stage1 = round(res.stage1_acceptance_rate * 2000)
        self.assertEqual(res.exact_evaluations, stage1 + 1)
        self.assertLess(stage1, 2000)  # some stage-1 rejections happened

    def test_support_policy(self) -> None:
        """Surrogate -inf: no exact call; exact -inf: stage 2 rejects."""
        calls = [0]

        def exact(x: tuple[float, ...]) -> float:
            """Exact target on [0, 1] with a hole at x > 0.5."""
            calls[0] += 1
            return -math.inf if x[0] > 0.5 else 0.0

        def sur(x: tuple[float, ...]) -> float:
            """Surrogate positive only on [0, 1]."""
            return 0.0 if 0.0 <= x[0] <= 1.0 else -math.inf

        res = delayed_acceptance_metropolis(
            exact, sur, (0.25,), step=0.5, samples=3000, seed=2
        )
        self.assertTrue(all(0.0 <= s[0] <= 0.5 for s in res.samples))
        self.assertLess(calls[0], res.surrogate_evaluations)

    def test_seed_reproducibility(self) -> None:
        """Same seed, same chain; different seed, different chain."""
        t, s = _gauss_logpdf(1.0, 0.5), _gauss_logpdf(0.7, 0.8)
        a = delayed_acceptance_metropolis(t, s, (0.0,), samples=200, seed=1)
        b = delayed_acceptance_metropolis(t, s, (0.0,), samples=200, seed=1)
        c = delayed_acceptance_metropolis(t, s, (0.0,), samples=200, seed=2)
        self.assertIsInstance(a, DelayedAcceptanceResult)
        self.assertEqual(a.samples, b.samples)
        self.assertNotEqual(a.samples, c.samples)
        self.assertEqual(a.seed, 1)


class InvalidInputs(unittest.TestCase):
    """Rejected inputs."""

    def run_da(self, **kw: object) -> DelayedAcceptanceResult:
        """Call with defaults overridden by ``kw``."""
        args: dict[str, object] = {
            "log_target": lambda x: -0.5 * x[0] ** 2,
            "log_surrogate": lambda x: -0.5 * x[0] ** 2,
            "initial": (0.0,),
            "samples": 10,
            "seed": 0,
        }
        args.update(kw)
        return delayed_acceptance_metropolis(**args)

    def test_nan_and_inf_raise(self) -> None:
        """NaN anywhere and +inf raise ValueError."""
        with self.assertRaises(ValueError):
            self.run_da(log_target=lambda x: math.nan)
        with self.assertRaises(ValueError):
            self.run_da(log_surrogate=lambda x: math.nan if x[0] != 0.0 else 0.0)
        with self.assertRaises(ValueError):
            self.run_da(log_target=lambda x: math.inf)
        with self.assertRaises(ValueError):  # non-finite start
            self.run_da(log_surrogate=lambda x: -math.inf)

    def test_pcn_and_step_validation(self) -> None:
        """Reject beta outside (0, 1], missing/misplaced log_prior, bad step."""
        for beta in (0.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                self.run_da(proposal="pcn", step=beta, log_prior="standard_normal")
        with self.assertRaises(ValueError):
            self.run_da(proposal="pcn", step=0.5)  # log_prior missing
        with self.assertRaises(ValueError):
            self.run_da(log_prior="standard_normal")  # RW must not declare it
        with self.assertRaises(ValueError):
            self.run_da(step=0.0)
        with self.assertRaises(ValueError):
            self.run_da(step=(1.0, 1.0))
        with self.assertRaises(ValueError):
            self.run_da(proposal="hmc")
        with self.assertRaises(ValueError):
            self.run_da(samples=0)
        with self.assertRaises(TypeError):
            self.run_da(samples=2.5)
        with self.assertRaises(TypeError):
            self.run_da(log_target=3)
        self.assertEqual(self.run_da(step=(0.7,)).step, (0.7,))


if __name__ == "__main__":
    unittest.main()
