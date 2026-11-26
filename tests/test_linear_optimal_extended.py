"""Tests for operator-path, sequential and robust linear optimal design."""

from __future__ import annotations

import math
import os
import random
import unittest
from typing import Any

from probstat.design.linear_optimal import (
    exhaustive_linear_design,
    greedy_linear_design,
    linear_design_criteria,
)
from probstat.design.linear_optimal_extended import (
    DesignScenario,
    greedy_operator_a_design,
    jacobian_scenarios,
    operator_a_criterion,
    operator_d_criterion,
    robust_linear_design,
    sequential_linear_design,
)
from probstat.foundations.random_streams import RandomStream
from probstat.linalg import matmul, matvec, transpose
from probstat.linalg.covariance_operators import DenseCovariance, DiagonalCovariance

Matrix = tuple[tuple[float, ...], ...]


def _spd(n: int, seed: int) -> Matrix:
    """Return a well-conditioned random SPD matrix ``G G' + I / 2``."""
    rng = random.Random(seed)
    g = tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(n)) for _ in range(n))
    gg = matmul(g, transpose(g))
    return tuple(
        tuple(gg[i][j] + (0.5 if i == j else 0.0) for j in range(n)) for i in range(n)
    )


def _rows(count: int, n: int, seed: int) -> Matrix:
    """Return random candidate rows."""
    rng = random.Random(seed)
    return tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(n)) for _ in range(count))


def _largest_mean_target(mean: tuple[float, ...]) -> tuple[float, ...]:
    """Return the unit goal of the coordinate with the largest ``|mean|``."""
    i = max(range(len(mean)), key=lambda k: abs(mean[k]))
    return tuple(1.0 if j == i else 0.0 for j in range(len(mean)))


def _easy_problem() -> tuple[Matrix, Matrix, tuple[float, ...]]:
    """Return a prior, unit-vector sensors and noise with a clear A ranking."""
    d = (9.0, 4.0, 1.0, 0.5, 0.2)
    n = len(d)
    prior = tuple(
        tuple(d[i] if i == j else 0.15 * math.sqrt(d[i] * d[j]) for j in range(n))
        for i in range(n)
    )
    rows = tuple(tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n)) + (
        (0.3,) * n,
    )
    return prior, rows, (0.1,) * n + (1.0,)


class _NegativePrior:
    """A deliberately wrong "covariance" whose solve is negative definite."""

    dimension = 3

    def solve(self, vector: Any) -> tuple[float, ...]:
        """Return ``-v``."""
        return tuple(-float(x) for x in vector)

    def apply(self, vector: Any) -> tuple[float, ...]:
        """Return ``-v``."""
        return tuple(-float(x) for x in vector)

    def factor_operator(self) -> Any:
        """Unused."""
        raise NotImplementedError


class OperatorCriteriaTests(unittest.TestCase):
    """A and D criteria on the operator path against dense oracles."""

    def setUp(self) -> None:
        """Build an 8-dimensional problem with a 4-sensor design."""
        self.n = 8
        self.prior = _spd(self.n, 1)
        self.rows = _rows(12, self.n, 2)
        rng = random.Random(3)
        self.var = tuple(0.2 + 0.1 * rng.random() for _ in range(12))
        self.selection = (0, 3, 5, 7)
        self.dense = linear_design_criteria(
            self.rows, self.var, self.selection, prior_covariance=self.prior
        )

    def test_a_closed_form_diagonal(self) -> None:
        """Diagonal posterior: Hutchinson is exact and equals the formula."""
        p = (2.0, 3.0, 0.5)
        rows = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        var = (0.5, 2.0)
        out = operator_a_criterion(
            rows, var, (0, 1), prior=DiagonalCovariance(p), probes=5, seed=0
        )
        exact = 1.0 / (1.0 / 2.0 + 1.0 / 0.5) + 1.0 / (1.0 / 3.0 + 1.0 / 2.0) + 0.5
        self.assertAlmostEqual(out.trace_estimate, exact, places=8)
        self.assertLess(out.standard_error, 1e-8)
        self.assertTrue(out.all_converged)

    def test_a_matches_dense_within_standard_error(self) -> None:
        """The estimate lies within 4 reported SE of the dense trace."""
        out = operator_a_criterion(
            self.rows,
            self.var,
            self.selection,
            prior=DenseCovariance(self.prior),
            probes=200,
            seed=3,
        )
        self.assertEqual(out.probes, 200)
        self.assertEqual(len(out.cg_iterations), 200)
        self.assertTrue(out.all_converged)
        self.assertEqual(set(out.cg_stop_reasons), {"residual_tolerance"})
        self.assertLess(out.max_relative_residual, 1e-9)
        self.assertGreater(out.standard_error, 0.0)
        self.assertLess(
            abs(out.trace_estimate - self.dense.a_optimal), 4.0 * out.standard_error
        )

    def test_probes_are_common_random_numbers(self) -> None:
        """Same seed reproduces samples exactly; another seed does not."""
        kwargs: dict[str, Any] = {"prior": DenseCovariance(self.prior), "probes": 10}
        a = operator_a_criterion(self.rows, self.var, self.selection, seed=7, **kwargs)
        b = operator_a_criterion(self.rows, self.var, self.selection, seed=7, **kwargs)
        c = operator_a_criterion(self.rows, self.var, self.selection, seed=8, **kwargs)
        self.assertEqual(a.samples, b.samples)
        self.assertNotEqual(a.samples, c.samples)

    def test_empty_design_is_prior_trace(self) -> None:
        """No sensors: the precision is the prior precision."""
        out = operator_a_criterion(
            self.rows, self.var, (), prior=DiagonalCovariance((1.0, 2.0) * 4), probes=3
        )
        self.assertAlmostEqual(out.trace_estimate, 12.0, places=8)

    def test_d_full_rank_equals_dense(self) -> None:
        """All eigenvalues retained: dense log det to rounding, both methods."""
        prior = DenseCovariance(self.prior)
        for method in ("lanczos", "randomized_svd"):
            out = operator_d_criterion(
                self.rows, self.var, self.selection, prior=prior, method=method
            )
            self.assertTrue(out.exact)
            self.assertAlmostEqual(out.log_det, self.dense.d_optimal, places=8)
            self.assertAlmostEqual(
                out.expected_information_gain, 0.5 * self.dense.d_optimal, places=8
            )

    def test_d_truncation_bound_holds(self) -> None:
        """Rank 2 of 4: the estimate is a lower bound within the stated bound."""
        prior = DenseCovariance(self.prior)
        for method in ("lanczos", "randomized_svd"):
            out = operator_d_criterion(
                self.rows,
                self.var,
                self.selection,
                prior=prior,
                rank=2,
                method=method,
            )
            gap = self.dense.d_optimal - out.log_det
            self.assertFalse(out.exact)
            self.assertEqual(out.rank, 2)
            self.assertGreaterEqual(gap, -1e-9)
            self.assertLessEqual(gap, out.truncation_bound + 1e-9)

    def test_d_empty_design(self) -> None:
        """Empty design has zero information."""
        out = operator_d_criterion(
            self.rows, self.var, (), prior=DenseCovariance(self.prior)
        )
        self.assertEqual(out.log_det, 0.0)

    def test_d_replicated_sensor(self) -> None:
        """A replicated sensor gives a rank-deficient data-space Hessian."""
        prior = DenseCovariance(self.prior)
        dup = (2, 2, 2)
        dense = linear_design_criteria(
            self.rows, self.var, dup, prior_covariance=self.prior
        )
        out = operator_d_criterion(self.rows, self.var, dup, prior=prior)
        self.assertAlmostEqual(out.log_det, dense.d_optimal, places=8)

    def test_failures(self) -> None:
        """Non-convergence is flagged; invalid input is rejected."""
        prior = DenseCovariance(self.prior)
        capped = operator_a_criterion(
            self.rows,
            self.var,
            self.selection,
            prior=prior,
            probes=4,
            cg_max_iterations=1,
        )
        self.assertFalse(capped.all_converged)
        self.assertIn("max_iterations", capped.cg_stop_reasons)
        bad = operator_a_criterion(
            tuple(r[:3] for r in self.rows[:3]),
            self.var[:3],
            (0,),
            prior=_NegativePrior(),
            probes=2,
        )
        self.assertFalse(bad.all_converged)
        self.assertIn("breakdown", bad.cg_stop_reasons)
        with self.assertRaises(TypeError):
            operator_a_criterion(self.rows, self.var, (0,), prior=self.prior)
        with self.assertRaises(ValueError):
            operator_a_criterion(
                self.rows, self.var, (0,), prior=DiagonalCovariance((1.0,) * 3)
            )
        with self.assertRaises(ValueError):
            operator_a_criterion(self.rows, self.var, (0,), prior=prior, probes=0)
        with self.assertRaises(ValueError):
            operator_a_criterion(self.rows, self.var, (99,), prior=prior)
        with self.assertRaises(ValueError):
            operator_d_criterion(
                self.rows, self.var, self.selection, prior=prior, rank=5
            )
        with self.assertRaises(ValueError):
            operator_d_criterion(
                self.rows, self.var, self.selection, prior=prior, method="svd"
            )

    def test_one_probe_has_undefined_standard_error(self) -> None:
        """A single probe cannot give a standard error."""
        out = operator_a_criterion(
            self.rows,
            self.var,
            self.selection,
            prior=DenseCovariance(self.prior),
            probes=1,
        )
        self.assertTrue(math.isnan(out.standard_error))


class OperatorGreedyTests(unittest.TestCase):
    """Stochastic greedy design with common random numbers."""

    def test_matches_dense_greedy_on_easy_case(self) -> None:
        """Clearly separated sensors: same indices as the dense greedy design."""
        prior, rows, var = _easy_problem()
        dense = greedy_linear_design(
            rows, var, prior_covariance=prior, budget=3, criterion="a"
        )
        out = greedy_operator_a_design(
            rows, var, prior=DenseCovariance(prior), size=3, probes=40, seed=5
        )
        self.assertEqual(out.indices, dense.indices)
        self.assertTrue(out.common_random_numbers)
        self.assertTrue(out.all_converged)
        self.assertEqual(out.cg_solves, 40 * (6 + 5 + 4))
        for estimate, se, exact in zip(
            out.trace_estimates, out.standard_errors, dense.values, strict=True
        ):
            self.assertLess(abs(estimate - exact), 4.0 * se)
        for margin, se in zip(out.margins, out.margin_standard_errors, strict=True):
            self.assertGreater(margin, 4.0 * se)

    def test_reproducible_and_candidate_table(self) -> None:
        """Same seed, same result; chosen sensors are NaN in later tables."""
        prior, rows, var = _easy_problem()
        kwargs: dict[str, Any] = {
            "prior": DenseCovariance(prior),
            "size": 2,
            "probes": 10,
            "seed": 1,
        }
        a = greedy_operator_a_design(rows, var, **kwargs)
        b = greedy_operator_a_design(rows, var, **kwargs)
        self.assertEqual(a.indices, b.indices)
        self.assertEqual(a.trace_estimates, b.trace_estimates)
        self.assertTrue(math.isnan(a.candidate_estimates[1][a.indices[0]]))

    def test_stream_seed_and_no_candidates_left(self) -> None:
        """A stream seed works; asking for too many sensors stops early."""
        prior, rows, var = _easy_problem()
        out = greedy_operator_a_design(
            rows[:2],
            var[:2],
            prior=DenseCovariance(prior),
            size=3,
            probes=5,
            seed=RandomStream(2),
        )
        self.assertEqual(len(out.indices), 2)
        self.assertEqual(out.stop_reason, "no_candidates_left")

    def test_failures(self) -> None:
        """Invalid sizes and priors are rejected."""
        prior, rows, var = _easy_problem()
        with self.assertRaises(ValueError):
            greedy_operator_a_design(rows, var, prior=DenseCovariance(prior), size=0)
        with self.assertRaises(TypeError):
            greedy_operator_a_design(rows, var, prior=prior, size=1)


class SequentialTests(unittest.TestCase):
    """Choose, observe, update, choose again."""

    def setUp(self) -> None:
        """Build a 5-dimensional problem with 10 candidates."""
        self.n = 5
        self.prior = _spd(self.n, 4)
        self.rows = _rows(10, self.n, 5)
        self.var = tuple(0.3 for _ in range(10))
        self.mean = (0.0,) * self.n

    def test_fixed_criterion_equals_batch(self) -> None:
        """Linear Gaussian: the sequential design is the batch design."""
        for criterion in ("a", "d"):
            out = sequential_linear_design(
                self.rows,
                self.var,
                prior_mean=self.mean,
                prior_covariance=self.prior,
                size=4,
                criterion=criterion,
                seed=3,
            )
            self.assertEqual(out.indices, out.batch_indices)
            self.assertAlmostEqual(out.trace_history[-1], out.batch_trace, places=10)
            self.assertAlmostEqual(
                out.squared_errors[-1], out.batch_squared_error, places=10
            )

    def test_posterior_matches_dense_batch_update(self) -> None:
        """The recursive update equals ``m + C A' R^-1 (y - A m)`` exactly."""
        out = sequential_linear_design(
            self.rows,
            self.var,
            prior_mean=self.mean,
            prior_covariance=self.prior,
            size=4,
            seed=11,
        )
        crit = linear_design_criteria(
            self.rows, self.var, out.indices, prior_covariance=self.prior
        )
        assert crit.posterior_covariance is not None
        a = tuple(self.rows[j] for j in out.indices)
        weighted = tuple(
            y / self.var[j] for y, j in zip(out.observations, out.indices, strict=True)
        )
        shift = matvec(crit.posterior_covariance, matvec(transpose(a), weighted))
        for got, delta in zip(out.final_mean, shift, strict=True):
            self.assertAlmostEqual(got, delta, places=9)
        for i in range(self.n):
            for j in range(self.n):
                self.assertAlmostEqual(
                    out.final_covariance[i][j],
                    crit.posterior_covariance[i][j],
                    places=9,
                )
        self.assertLess(out.trace_history[-1], out.initial_trace)
        self.assertEqual(
            list(out.trace_history), sorted(out.trace_history, reverse=True)
        )

    def test_observations_come_from_separate_noise_stream(self) -> None:
        """``y = a'x + s z`` with ``z`` from child ``("observation", j, 0)``."""
        truth = (0.5, -1.0, 0.25, 2.0, -0.5)
        out = sequential_linear_design(
            self.rows,
            self.var,
            prior_mean=self.mean,
            prior_covariance=self.prior,
            size=3,
            truth=truth,
            seed=21,
        )
        stream = RandomStream(21)
        for j, y in zip(out.indices, out.observations, strict=True):
            z = stream.child("observation", j, 0).generator().gauss(0.0, 1.0)
            signal = sum(a * x for a, x in zip(self.rows[j], truth, strict=True))
            self.assertAlmostEqual(y, signal + math.sqrt(self.var[j]) * z, places=12)
        self.assertEqual(out.truth, truth)

    def test_no_look_ahead_leak(self) -> None:
        """An adaptive first choice depends only on the prior, not on the truth."""
        picks = []
        for truth in ((3.0, 0, 0, 0, 0), (0, 0, 0, 0, -3.0), (1.0,) * 5):
            out = sequential_linear_design(
                self.rows,
                self.var,
                prior_mean=(0.0, 0.1, 0.0, 0.0, 0.0),
                prior_covariance=self.prior,
                size=3,
                goal_policy=_largest_mean_target,
                truth=truth,
                seed=5,
            )
            self.assertTrue(out.adaptive)
            picks.append(out.indices[0])
        self.assertEqual(len(set(picks)), 1)

    def test_adaptive_goal_runs_and_reports_batch(self) -> None:
        """A mean-dependent target can differ from the A batch design."""
        differing = 0
        for seed in range(12):
            out = sequential_linear_design(
                self.rows,
                self.var,
                prior_mean=self.mean,
                prior_covariance=self.prior,
                size=3,
                goal_policy=_largest_mean_target,
                seed=seed,
            )
            self.assertEqual(len(out.indices), 3)
            self.assertEqual(len(set(out.indices)), 3)
            differing += out.indices != out.batch_indices
        self.assertGreaterEqual(differing, 1)

    def test_failures(self) -> None:
        """Invalid sizes, means and truths are rejected."""
        base: dict[str, Any] = {"prior_mean": self.mean, "prior_covariance": self.prior}
        with self.assertRaises(ValueError):
            sequential_linear_design(self.rows, self.var, size=0, **base)
        with self.assertRaises(ValueError):
            sequential_linear_design(self.rows, self.var, size=11, **base)
        with self.assertRaises(ValueError):
            sequential_linear_design(
                self.rows,
                self.var,
                size=2,
                prior_mean=(0.0,) * 4,
                prior_covariance=self.prior,
            )
        with self.assertRaises(ValueError):
            sequential_linear_design(self.rows, self.var, size=2, truth=(1.0,), **base)
        with self.assertRaises(ValueError):
            sequential_linear_design(
                self.rows, self.var, size=2, criterion="goal", **base
            )


class RobustDesignTests(unittest.TestCase):
    """Bayesian-average and worst-case designs with an exhaustive oracle."""

    def setUp(self) -> None:
        """Build three scenarios that disagree about the best sensors."""
        self.n = 3
        self.rows_a = _rows(7, self.n, 10)
        self.rows_b = _rows(7, self.n, 11)
        self.rows_c = _rows(7, self.n, 12)
        self.prior = _spd(self.n, 13)
        var = (0.4,) * 7
        self.scenarios = (
            DesignScenario(self.rows_a, var, self.prior, label="a"),
            DesignScenario(self.rows_b, var, self.prior, label="b"),
            DesignScenario(self.rows_c, tuple(2.0 * v for v in var), self.prior),
        )

    def test_single_scenario_reduces_to_linear_design(self) -> None:
        """One scenario: greedy equals the dense greedy; exhaustive the oracle."""
        sc = self.scenarios[:1]
        for criterion in ("a", "d"):
            robust = robust_linear_design(sc, size=3, criterion=criterion)
            dense = greedy_linear_design(
                self.rows_a,
                sc[0].noise_variances,
                prior_covariance=self.prior,
                budget=3,
                criterion=criterion,
            )
            self.assertEqual(robust.indices, dense.indices)
            oracle = exhaustive_linear_design(
                self.rows_a,
                sc[0].noise_variances,
                prior_covariance=self.prior,
                size=3,
                criterion=criterion,
            )
            exhaustive = robust_linear_design(
                sc, size=3, criterion=criterion, method="exhaustive"
            )
            self.assertEqual(exhaustive.indices, oracle.indices)
            self.assertAlmostEqual(exhaustive.value, oracle.value, places=10)
            self.assertEqual(exhaustive.combinations, 35)

    def test_exhaustive_is_no_worse_than_greedy(self) -> None:
        """The oracle bounds greedy for every aggregate and criterion."""
        for criterion in ("a", "d"):
            for aggregate in ("mean", "max"):
                greedy = robust_linear_design(
                    self.scenarios, size=3, criterion=criterion, aggregate=aggregate
                )
                oracle = robust_linear_design(
                    self.scenarios,
                    size=3,
                    criterion=criterion,
                    aggregate=aggregate,
                    method="exhaustive",
                )
                sign = 1.0 if criterion == "a" else -1.0
                self.assertLessEqual(sign * oracle.value, sign * greedy.value + 1e-9)

    def test_worst_case_beats_average_design_on_worst_scenario(self) -> None:
        """The minimax design is no worse in its worst scenario (oracle)."""
        mean_design = robust_linear_design(
            self.scenarios, size=2, aggregate="mean", method="exhaustive"
        )
        minimax = robust_linear_design(
            self.scenarios, size=2, aggregate="max", method="exhaustive"
        )
        self.assertLessEqual(minimax.value, max(mean_design.scenario_values) + 1e-12)
        self.assertEqual(minimax.value, max(minimax.scenario_values))
        self.assertEqual(minimax.scenario_values[minimax.worst_scenario], minimax.value)

    def test_worst_case_d_uses_smallest_log_det(self) -> None:
        """For D the worst scenario is the one with the least information."""
        out = robust_linear_design(
            self.scenarios, size=3, criterion="d", aggregate="max", method="exhaustive"
        )
        self.assertEqual(out.value, min(out.scenario_values))

    def test_zero_weight_removes_scenarios(self) -> None:
        """Zero weights remove scenarios from the average."""
        weighted = tuple(
            DesignScenario(
                s.candidate_rows, s.noise_variances, s.prior_covariance, w, s.label
            )
            for s, w in zip(self.scenarios, (1.0, 0.0, 0.0), strict=True)
        )
        robust = robust_linear_design(weighted, size=3, method="exhaustive")
        single = robust_linear_design(self.scenarios[:1], size=3, method="exhaustive")
        self.assertEqual(robust.indices, single.indices)

    def test_bayesian_average_over_nonlinear_parameter(self) -> None:
        """Jacobian varies with theta: the averaged design beats the nominal one."""
        times = tuple(0.5 * k for k in range(1, 9))

        def jacobian(theta: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
            """Rows ``(cos(theta t), sin(theta t))`` of a sinusoid's amplitudes."""
            return tuple(
                (math.cos(theta[0] * t), math.sin(theta[0] * t)) for t in times
            )

        thetas = tuple((0.4 + 0.3 * k,) for k in range(6))
        scenarios = jacobian_scenarios(
            jacobian, thetas, (0.2,) * 8, ((1.0, 0.0), (0.0, 1.0))
        )
        self.assertEqual(len(scenarios), 6)
        self.assertEqual(scenarios[2].label, "theta[2]")
        average = robust_linear_design(
            scenarios, size=2, aggregate="mean", method="exhaustive"
        )
        nominal = robust_linear_design(
            scenarios[:1], size=2, aggregate="mean", method="exhaustive"
        )
        values = [
            linear_design_criteria(
                s.candidate_rows,
                s.noise_variances,
                nominal.indices,
                prior_covariance=s.prior_covariance,
            ).a_optimal
            for s in scenarios
        ]
        self.assertLessEqual(average.value, sum(values) / len(values) + 1e-12)

    def test_failures(self) -> None:
        """Invalid scenarios and options are rejected."""
        sc = self.scenarios
        with self.assertRaises(ValueError):
            robust_linear_design((), size=1)
        with self.assertRaises(TypeError):
            robust_linear_design((object(),), size=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            robust_linear_design(sc, size=1, criterion="e")
        with self.assertRaises(ValueError):
            robust_linear_design(sc, size=1, aggregate="median")
        with self.assertRaises(ValueError):
            robust_linear_design(sc, size=1, method="random")
        with self.assertRaises(ValueError):
            robust_linear_design(sc, size=8)
        with self.assertRaises(ValueError):
            robust_linear_design(sc, size=3, method="exhaustive", max_combinations=5)
        short = DesignScenario(self.rows_a[:5], (0.4,) * 5, self.prior)
        with self.assertRaises(ValueError):
            robust_linear_design((sc[0], short), size=1)
        negative = DesignScenario(self.rows_a, (0.4,) * 7, self.prior, weight=-1.0)
        with self.assertRaises(ValueError):
            robust_linear_design((negative,), size=1)
        zero = DesignScenario(self.rows_a, (0.4,) * 7, self.prior, weight=0.0)
        with self.assertRaises(ValueError):
            robust_linear_design((zero,), size=1)
        with self.assertRaises(ValueError):
            jacobian_scenarios(lambda t: ((1.0,),), (), (1.0,), ((1.0,),))
        with self.assertRaises(TypeError):
            jacobian_scenarios(1, ((0.0,),), (1.0,), ((1.0,),))  # type: ignore[arg-type]


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudies(unittest.TestCase):
    """Measured replicate studies (run with PROBSTAT_SLOW=1)."""

    def test_sequential_vs_batch_replicates(self) -> None:
        """Adaptive-goal sequential versus A batch over 200 truths."""
        prior = _spd(5, 4)
        rows = _rows(10, 5, 5)
        var = (0.3,) * 10
        seq_err = []
        batch_err = []
        for seed in range(200):
            out = sequential_linear_design(
                rows,
                var,
                prior_mean=(0.0,) * 5,
                prior_covariance=prior,
                size=3,
                goal_policy=_largest_mean_target,
                seed=seed,
            )
            seq_err.append(out.squared_errors[-1])
            batch_err.append(out.batch_squared_error)
        print(
            f"sequential mean sq. error {sum(seq_err) / 200:.4f}, "
            f"batch {sum(batch_err) / 200:.4f}"
        )
        self.assertTrue(all(math.isfinite(e) for e in seq_err + batch_err))


if __name__ == "__main__":
    unittest.main()
