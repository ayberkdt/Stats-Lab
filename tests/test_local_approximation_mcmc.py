"""Tests for local-approximation MCMC (EQ-INV-LOCAL-001..006).

Benchmark: exponential decay ``G(a, k) = (a exp(-k t_j))_j`` at five times with
fixed (deterministic) noise values, Gaussian priors and noise sd 0.05. The
reference moments come from a tensor-grid quadrature of the exact
posterior; stochastic checks use bands of four Monte Carlo standard errors
``sd / sqrt(ESS)`` with fixed seeds.
"""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.inference.mcmc import effective_sample_size
from probstat.inverse.local_approximation_mcmc import (
    LocalApproximationChainResult,
    LocalDelayedAcceptanceResult,
    local_approximation_delayed_acceptance,
    local_approximation_metropolis,
    local_weighted_fit,
)

TIMES = (0.5, 1.0, 1.5, 2.0, 3.0)
NOISE = (0.03, -0.04, 0.02, -0.01, 0.05)
DATA = tuple(1.8 * math.exp(-0.8 * t) + e for t, e in zip(TIMES, NOISE, strict=True))
SIGMA = 0.05
STEP = (0.08, 0.06)
START = (1.9, 0.9)
GRID = [(a, k) for a in (0.0, 1.5, 3.0, 4.5) for k in (0.0, 0.7, 1.4, 2.1)]


def forward(x: tuple[float, ...]) -> tuple[float, ...]:
    """Exact (here cheap) forward map."""
    return tuple(x[0] * math.exp(-x[1] * t) for t in TIMES)


def log_prior(x: tuple[float, ...]) -> float:
    """Gaussian prior ``a ~ N(2, 1)``, ``k ~ N(1, 0.5^2)``."""
    return -0.5 * (x[0] - 2.0) ** 2 - 0.5 * ((x[1] - 1.0) / 0.5) ** 2


def log_post(x: tuple[float, ...]) -> float:
    """Exact log-posterior up to a constant."""
    return log_prior(x) - 0.5 * sum(
        ((y - g) / SIGMA) ** 2 for y, g in zip(DATA, forward(x), strict=True)
    )


def _quadrature() -> tuple[tuple[float, float], tuple[float, float]]:
    """Posterior means and sds by a 200 x 200 tensor grid."""
    n = 200
    a_grid = [0.5 + 3.0 * i / (n - 1) for i in range(n)]
    k_grid = [0.2 + 1.6 * j / (n - 1) for j in range(n)]
    weights = [[math.exp(log_post((a, k))) for k in k_grid] for a in a_grid]
    total = math.fsum(math.fsum(row) for row in weights)
    idx = [(i, j) for i in range(n) for j in range(n)]
    mean_a = math.fsum(weights[i][j] * a_grid[i] for i, j in idx) / total
    mean_k = math.fsum(weights[i][j] * k_grid[j] for i, j in idx) / total
    var_a = math.fsum(weights[i][j] * (a_grid[i] - mean_a) ** 2 for i, j in idx)
    var_k = math.fsum(weights[i][j] * (k_grid[j] - mean_k) ** 2 for i, j in idx)
    return (mean_a, mean_k), (math.sqrt(var_a / total), math.sqrt(var_k / total))


MEAN, SD = _quadrature()


def _moments(draws: tuple[tuple[float, ...], ...]) -> tuple[list[float], list[float]]:
    """Return the sample means and standard deviations."""
    n = len(draws)
    mean = [math.fsum(d[j] for d in draws) / n for j in range(2)]
    sd = [
        math.sqrt(math.fsum((d[j] - mean[j]) ** 2 for d in draws) / n) for j in range(2)
    ]
    return mean, sd


def _plain_metropolis(
    steps: int, seed: int
) -> tuple[tuple[tuple[float, ...], ...], int]:
    """Plain random-walk Metropolis calling the exact target every step."""
    rng = random.Random(seed)
    x, lx = START, log_post(START)
    draws = []
    for i in range(steps + 300):
        y = (x[0] + STEP[0] * rng.gauss(0, 1), x[1] + STEP[1] * rng.gauss(0, 1))
        ly = log_post(y)
        if math.log(1.0 - rng.random()) < ly - lx:
            x, lx = y, ly
        if i >= 300:
            draws.append(x)
    return tuple(draws), steps + 301


def _chain(**overrides: object) -> LocalApproximationChainResult:
    """Run the standard local-approximation chain."""
    options: dict[str, object] = {
        "noise_std": SIGMA,
        "log_prior": log_prior,
        "initial": START,
        "step": STEP,
        "samples": 1600,
        "burn_in": 300,
        "seed": 1,
    }
    options.update(overrides)
    return local_approximation_metropolis(forward, DATA, **options)  # type: ignore[arg-type]


class LocalRegression(unittest.TestCase):
    """Closed forms, leave-one-out oracle and ill-posed designs."""

    pts = [(0.0, 0.0), (0.4, 0.1), (-0.3, 0.5), (0.2, -0.4), (0.5, 0.45), (-0.5, -0.2)]

    def test_quadratic_is_reproduced_exactly(self) -> None:
        """A quadratic map is fitted without error, leave-one-out shifts vanish."""

        def g(x: tuple[float, float]) -> tuple[float, float]:
            return (1 + 2 * x[0] - x[1] + 0.5 * x[0] * x[1] + x[1] ** 2, x[0] ** 2)

        pts = [
            (0.1 * i - 0.5 + 0.03 * (i % 3), 0.07 * i * (-1) ** i) for i in range(14)
        ]
        fit = local_weighted_fit(pts, [g(p) for p in pts], (0.12, -0.05), order=2)
        self.assertEqual(fit.status, "ok")
        exact = g((0.12, -0.05))
        assert fit.value is not None
        for got, want in zip(fit.value, exact, strict=True):
            self.assertAlmostEqual(got, want, places=8)
        self.assertLess(max(abs(s) for sh in fit.loo_shifts for s in sh), 1e-7)

    def test_weights_and_order_constants(self) -> None:
        """Tricube weights in (0, 1], farthest weight (1 - 1/1.1^3)^3."""
        pts = [(0.1 * i, 0.05 * i * i) for i in range(10)]
        fit = local_weighted_fit(
            pts, [(p[0],) for p in pts], (0.0, 0.0), order=1, neighbours=8
        )
        self.assertEqual(fit.status, "ok")
        self.assertEqual(len(fit.neighbours), 8)
        self.assertTrue(all(0.0 < w <= 1.0 for w in fit.weights))
        self.assertAlmostEqual(fit.weights[-1], (1.0 - 1.0 / 1.1**3) ** 3, places=12)
        self.assertEqual(fit.neighbours[0], 0)

    def test_leave_one_out_shift_matches_refit(self) -> None:
        """delta_j equals value(all) - value(without j) from an actual refit."""
        rng = random.Random(7)
        pts = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(8)]
        vals = [(math.sin(3 * p[0]) + p[1] ** 3 + 0.1 * rng.gauss(0, 1),) for p in pts]
        query = (0.05, -0.1)
        full = local_weighted_fit(pts, vals, query, order=1, neighbours=8)
        assert full.value is not None
        far = full.neighbours[-1]
        checked = 0
        for pos, j in enumerate(full.neighbours):
            if j == far:
                continue  # removing the farthest point changes the bandwidth
            keep = [i for i in range(8) if i != j]
            part = local_weighted_fit(
                [pts[i] for i in keep],
                [vals[i] for i in keep],
                query,
                order=1,
                neighbours=7,
            )
            assert part.value is not None
            self.assertAlmostEqual(
                full.value[0] - part.value[0], full.loo_shifts[pos][0], places=10
            )
            checked += 1
        self.assertEqual(checked, 7)

    def test_status_codes_and_validation(self) -> None:
        """Too few points, collinear design and bad arguments are reported."""
        vals = [(1.0,)] * 6
        few = local_weighted_fit(self.pts[:5], vals[:5], (0.0, 0.0), order=2)
        self.assertEqual(few.status, "insufficient_points")
        self.assertIsNone(few.value)
        line = [(0.1 * i, 0.2 * i) for i in range(14)]
        bad = local_weighted_fit(line, [(1.0,)] * 14, (0.3, 0.6), order=2)
        self.assertEqual(bad.status, "ill_conditioned")
        with self.assertRaises(ValueError):
            local_weighted_fit(self.pts, vals, (0, 0), order=3)
        with self.assertRaises(ValueError):
            local_weighted_fit(self.pts, vals, (0, 0), order=2, neighbours=6)
        with self.assertRaises(ValueError):
            local_weighted_fit(self.pts, vals, (0,), order=1)
        with self.assertRaises(ValueError):
            local_weighted_fit(self.pts, vals[:5], (0, 0), order=1)
        with self.assertRaises(ValueError):
            local_weighted_fit(self.pts, vals, (0, math.nan), order=1)
        with self.assertRaises(ValueError):
            local_weighted_fit(
                self.pts, vals, (0, 0), order=1, length_scales=(1.0, -1.0)
            )


class AdaptiveChain(unittest.TestCase):
    """Accuracy and savings of the adaptive chain on the decay problem."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the shared chains once."""
        cls.local = _chain()
        cls.plain, cls.plain_calls = _plain_metropolis(6000, seed=2)

    def test_moments_within_mcse_bands(self) -> None:
        """Mean and sd agree with quadrature within 4 MCSE / 25 percent."""
        mean, sd = _moments(self.local.samples)
        for j in range(2):
            mcse = SD[j] / math.sqrt(self.local.effective_sample_size[j])
            self.assertLess(abs(mean[j] - MEAN[j]), 4 * mcse, f"mean {j}: {mean[j]}")
            self.assertLess(abs(sd[j] / SD[j] - 1.0), 0.25, f"sd {j}: {sd[j]}")

    def test_true_evaluation_accounting(self) -> None:
        """One true call per stored point; log and counts add up."""
        r = self.local
        self.assertEqual(r.true_evaluations, r.store_size)
        self.assertEqual(len(r.store_points), r.store_size)
        self.assertEqual(sum(r.refinement_counts.values()) + 1, r.true_evaluations)
        self.assertEqual(len(r.refinement_log), sum(r.refinement_counts.values()))
        self.assertEqual(len(r.cumulative_true_evaluations), len(r.samples))
        self.assertEqual(r.cumulative_true_evaluations[-1], r.true_evaluations)
        cumulative = r.cumulative_true_evaluations
        self.assertTrue(
            all(a <= b for a, b in zip(cumulative, cumulative[1:], strict=False))
        )
        self.assertTrue(r.refinement_never_stops)
        self.assertEqual(r.iterations, 1900)

    def test_savings_in_true_evaluations(self) -> None:
        """Measured: far fewer true calls and higher ESS per call than plain MH."""
        plain_ess = min(
            effective_sample_size([[d[j] for d in self.plain]]) for j in range(2)
        )
        plain_per_call = plain_ess / self.plain_calls
        local_per_call = self.local.ess_per_true_evaluation
        self.assertLess(self.local.evaluations_per_iteration, 0.15)
        self.assertGreater(
            local_per_call / plain_per_call,
            5.0,
            f"local {local_per_call:.4f} vs plain {plain_per_call:.4f}",
        )

    def test_determinism(self) -> None:
        """Same seed, same chain and store; another seed differs."""
        a = _chain(samples=150, burn_in=50, seed=5)
        b = _chain(samples=150, burn_in=50, seed=5)
        c = _chain(samples=150, burn_in=50, seed=6)
        self.assertEqual(a.samples, b.samples)
        self.assertEqual(a.store_points, b.store_points)
        self.assertEqual(a.refinement_log, b.refinement_log)
        self.assertNotEqual(a.samples, c.samples)

    def test_bad_surrogate_is_biased_and_refinement_fixes_it(self) -> None:
        """No refinement or gamma = 3 under-disperses; the refined chain does not."""
        coarse = {"initial_design": GRID, "cv_tolerance": None, "samples": 1000}
        frozen = _chain(beta0=0.0, **coarse)
        fast_decay = _chain(beta0=1.0, gamma=3.0, **coarse)
        refined = _chain(initial_design=GRID, samples=1000)
        self.assertFalse(frozen.refinement_never_stops)
        self.assertFalse(fast_decay.refinement_never_stops)
        self.assertTrue(refined.refinement_never_stops)
        self.assertEqual(frozen.true_evaluations, len(GRID) + 1)
        ratios = {}
        for label, run in (
            ("frozen", frozen),
            ("gamma3", fast_decay),
            ("refined", refined),
        ):
            _, sd = _moments(run.samples)
            ratios[label] = sd[1] / SD[1]
        self.assertLess(ratios["frozen"], 0.75, str(ratios))
        self.assertLess(ratios["gamma3"], 0.75, str(ratios))
        self.assertGreater(ratios["refined"], 0.8, str(ratios))
        self.assertLess(ratios["refined"], 1.2, str(ratios))

    def test_ill_conditioned_design_forces_true_evaluations(self) -> None:
        """An impossible condition limit makes every proposal a true call."""
        r = _chain(condition_limit=1.0001, samples=60, burn_in=20, beta0=0.0)
        self.assertGreater(r.refinement_counts["ill_conditioned"], 0)
        self.assertGreater(r.true_evaluations, 0.9 * r.iterations)


class DelayedAcceptance(unittest.TestCase):
    """Exactness of the local-surrogate delayed-acceptance variant."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the delayed-acceptance chain and a plain reference once."""
        cls.result = local_approximation_delayed_acceptance(
            forward,
            DATA,
            noise_std=SIGMA,
            log_prior=log_prior,
            initial=START,
            design=GRID,
            step=STEP,
            samples=3000,
            burn_in=300,
            seed=3,
        )
        cls.plain, _ = _plain_metropolis(6000, seed=4)

    def test_exact_despite_coarse_surrogate(self) -> None:
        """Moments match quadrature although the frozen surrogate is biased."""
        r = self.result
        self.assertIsInstance(r, LocalDelayedAcceptanceResult)
        self.assertTrue(r.exact_target)
        mean, sd = _moments(r.chain.samples)
        for j in range(2):
            mcse = SD[j] / math.sqrt(r.chain.effective_sample_size[j])
            self.assertLess(abs(mean[j] - MEAN[j]), 4 * mcse, f"mean {j}: {mean[j]}")
            self.assertLess(abs(sd[j] / SD[j] - 1.0), 0.25, f"sd {j}: {sd[j]}")

    def test_agrees_with_plain_metropolis(self) -> None:
        """DA and plain MH means differ by less than 4 combined MCSE."""
        mean, _ = _moments(self.result.chain.samples)
        pmean, _ = _moments(self.plain)
        for j in range(2):
            ess_p = effective_sample_size([[d[j] for d in self.plain]])
            mcse = SD[j] * math.sqrt(
                1.0 / self.result.chain.effective_sample_size[j] + 1.0 / ess_p
            )
            self.assertLess(abs(mean[j] - pmean[j]), 4 * mcse)

    def test_evaluation_accounting(self) -> None:
        """True calls = design + second stage; screening saves calls."""
        r = self.result
        self.assertEqual(r.design_evaluations, len(GRID) + 1)
        self.assertEqual(
            r.true_evaluations, r.design_evaluations + r.chain.exact_evaluations
        )
        self.assertLess(r.true_evaluations, r.chain.surrogate_evaluations)
        self.assertEqual(r.store_size, len(GRID) + 1)

    def test_failures(self) -> None:
        """Duplicated design, bad noise and bad order are rejected."""
        common = {
            "noise_std": SIGMA,
            "log_prior": log_prior,
            "step": STEP,
            "samples": 10,
        }
        with self.assertRaises(ValueError):
            local_approximation_delayed_acceptance(
                forward,
                DATA,
                initial=START,
                design=[(1.0, 1.0), (1.0, 1.0)],
                **common,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            local_approximation_delayed_acceptance(
                forward,
                DATA,
                initial=START,
                design=[],
                **{**common, "noise_std": 0.0},  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            local_approximation_delayed_acceptance(
                forward,
                DATA,
                initial=START,
                design=[(1.0, 1.0)],
                order=3,
                **common,  # type: ignore[arg-type]
            )


class Validation(unittest.TestCase):
    """Rejected inputs of the adaptive chain."""

    def test_rejections(self) -> None:
        """Invalid arguments and misbehaving callables raise."""
        for override in (
            {"beta0": 1.5},
            {"beta0": -0.1},
            {"gamma": -1.0},
            {"cv_tolerance": 0.0},
            {"step": -1.0},
            {"samples": 0},
            {"burn_in": -1},
            {"neighbours": 6},
            {"condition_limit": 1.0},
            {"order": 0},
            {"initial": (1.0,)},
            {"initial_design": [(1.0, 1.0), (1.0, 1.0)]},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                _chain(**{"samples": 5, "burn_in": 0, **override})
        with self.assertRaises(TypeError):
            _chain(samples=2.5)
        with self.assertRaises(ValueError):
            _chain(log_prior=lambda x: -math.inf)
        base = {
            "noise_std": SIGMA,
            "log_prior": log_prior,
            "initial": START,
            "step": STEP,
            "samples": 5,
        }
        with self.assertRaises(ValueError):
            local_approximation_metropolis(
                lambda x: (math.nan,) * 5,
                DATA,
                **base,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            local_approximation_metropolis(
                lambda x: (1.0, 2.0),
                DATA,
                **base,  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            local_approximation_metropolis(
                "not callable",  # type: ignore[arg-type]
                DATA,
                **base,  # type: ignore[arg-type]
            )


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowLongChain(unittest.TestCase):
    """Long chains: accuracy and savings at a larger sample size."""

    def test_long_chain(self) -> None:
        """Moments within 4 MCSE and a measured evaluation saving."""
        r = _chain(samples=15000, burn_in=1000, seed=11)
        plain, calls = _plain_metropolis(40000, seed=12)
        mean, _ = _moments(r.samples)
        for j in range(2):
            mcse = SD[j] / math.sqrt(r.effective_sample_size[j])
            self.assertLess(abs(mean[j] - MEAN[j]), 4 * mcse)
        plain_ess = min(
            effective_sample_size([[d[j] for d in plain]]) for j in range(2)
        )
        ratio = r.ess_per_true_evaluation / (plain_ess / calls)
        print(
            f"\nslow: true_evals={r.true_evaluations} of {r.iterations}, "
            f"ess/true local={r.ess_per_true_evaluation:.4f} "
            f"plain={plain_ess / calls:.4f} ratio={ratio:.1f}"
        )
        self.assertGreater(ratio, 10.0)


if __name__ == "__main__":
    unittest.main()
