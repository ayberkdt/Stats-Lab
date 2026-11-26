"""Tests for deterministic alphabetic criteria and budgeted sensor selection."""

from __future__ import annotations

import math
import random
import unittest

from probstat.design.bayesian import linear_gaussian_information_gain
from probstat.design.linear_optimal import (
    exhaustive_linear_design,
    greedy_linear_design,
    linear_design_criteria,
)
from probstat.linalg import inverse_spd, matmul, transpose


def _spd(n: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """Return a well-conditioned random SPD matrix ``G G' + I / 2``."""
    rng = random.Random(seed)
    g = tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(n)) for _ in range(n))
    gg = matmul(g, transpose(g))
    return tuple(
        tuple(gg[i][j] + (0.5 if i == j else 0.0) for j in range(n)) for i in range(n)
    )


def _rows(count: int, n: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """Return random candidate rows."""
    rng = random.Random(seed)
    return tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(n)) for _ in range(count))


def _close(
    test: unittest.TestCase, left: float, right: float, tol: float = 1e-10
) -> None:
    """Assert agreement to ``tol`` relative to ``1 + |right|``."""
    test.assertLessEqual(abs(left - right), tol * (1.0 + abs(right)), (left, right))


class TwoCandidateAnalyticTests(unittest.TestCase):
    """P = I, rows e1 and e2: everything is diagonal and known in closed form."""

    rows = ((1.0, 0.0), (0.0, 1.0))
    prior = ((1.0, 0.0), (0.0, 1.0))
    variances = (1.0, 4.0)

    def test_known_posterior_and_criteria(self) -> None:
        """C = diag(1/(1 + 1/s_j)); A, D, E, goal and EIG follow."""
        result = linear_design_criteria(
            self.rows,
            self.variances,
            (0, 1),
            prior_covariance=self.prior,
            goal=(1.0, 1.0),
        )
        assert result.posterior_covariance is not None
        c = result.posterior_covariance
        _close(self, c[0][0], 0.5, 1e-14)
        _close(self, c[1][1], 0.8, 1e-14)
        _close(self, c[0][1], 0.0, 1e-14)
        self.assertEqual(result.mode, "bayesian")
        _close(self, result.a_optimal, 1.3, 1e-14)
        _close(self, result.d_optimal, math.log(2.0) + math.log(1.25), 1e-14)
        assert result.expected_information_gain is not None
        _close(self, result.expected_information_gain, 0.5 * result.d_optimal, 1e-14)
        _close(self, result.e_information_eigenvalue, 1.25, 1e-14)
        _close(self, result.max_covariance_eigenvalue, 0.8, 1e-14)
        assert result.goal_variance is not None
        _close(self, result.goal_variance, 1.3, 1e-14)
        self.assertFalse(result.rank_deficient)

    def test_goal_matrix_and_single_sensor(self) -> None:
        """One sensor leaves the other coordinate at its prior variance."""
        result = linear_design_criteria(
            self.rows,
            self.variances,
            (0,),
            prior_covariance=self.prior,
            goal=((1.0, 0.0), (0.0, 2.0)),
        )
        # trace(Q C Q') = 0.5 + 4 * 1.0
        assert result.goal_variance is not None
        _close(self, result.goal_variance, 4.5, 1e-14)
        self.assertTrue(result.rank_deficient)  # F has rank one; prior fills in
        _close(self, result.a_optimal, 1.5, 1e-14)

    def test_empty_design_is_the_prior(self) -> None:
        """No sensors: C = P and zero information gain."""
        result = linear_design_criteria(
            self.rows, self.variances, (), prior_covariance=self.prior
        )
        self.assertEqual(result.d_optimal, 0.0)
        self.assertEqual(result.posterior_covariance, self.prior)


class EigCrossCheckTests(unittest.TestCase):
    """D-criterion equals the existing EIG routine on the same design."""

    def test_independent_and_correlated_noise(self) -> None:
        """EIG = 0.5 D for a dense prior, repeats and a full noise covariance."""
        n, count = 4, 6
        rows = _rows(count, n, 3)
        prior = _spd(n, 4)
        variances = tuple(0.3 + 0.2 * j for j in range(count))
        picks = (0, 2, 5, 2, 1)
        design = tuple(rows[i] for i in picks)
        result = linear_design_criteria(rows, variances, picks, prior_covariance=prior)
        reference = linear_gaussian_information_gain(
            design,
            prior,
            noise_variances=tuple(variances[i] for i in picks),
        )
        assert result.expected_information_gain is not None
        _close(self, result.expected_information_gain, reference, 1e-11)
        noise = _spd(len(picks), 9)
        joint = linear_design_criteria(
            rows, None, picks, prior_covariance=prior, noise_covariance=noise
        )
        ref_joint = linear_gaussian_information_gain(
            design, prior, noise_covariance=noise
        )
        assert joint.expected_information_gain is not None
        _close(self, joint.expected_information_gain, ref_joint, 1e-10)

    def test_posterior_matches_normal_equations(self) -> None:
        """C equals (P^-1 + F)^-1 computed through a different route."""
        n = 3
        rows = _rows(5, n, 11)
        prior = _spd(n, 12)
        variances = (0.5, 1.0, 2.0, 0.7, 1.3)
        picks = (0, 1, 3, 4)
        result = linear_design_criteria(rows, variances, picks, prior_covariance=prior)
        precision = inverse_spd(prior)
        fisher = result.information_matrix
        total = tuple(
            tuple(precision[i][j] + fisher[i][j] for j in range(n)) for i in range(n)
        )
        expected = inverse_spd(total)
        assert result.posterior_covariance is not None
        for i in range(n):
            for j in range(n):
                _close(self, result.posterior_covariance[i][j], expected[i][j], 1e-10)


class CorrelationAndDegeneracyTests(unittest.TestCase):
    """Replicates, correlated duplicates and zero-information candidates."""

    def test_replicate_versus_correlated_duplicate(self) -> None:
        """Independent replicate adds log(1 + ...); a duplicate adds ~ nothing."""
        rows, p, s = ((1.0,),), ((2.0,),), 0.5
        single = linear_design_criteria(rows, (s,), (0,), prior_covariance=p)
        replicate = linear_design_criteria(rows, (s,), (0, 0), prior_covariance=p)
        _close(self, single.d_optimal, math.log(1.0 + 2.0 / s), 1e-14)
        _close(self, replicate.d_optimal, math.log(1.0 + 2.0 * 2.0 / s), 1e-14)
        _close(
            self,
            replicate.d_optimal - single.d_optimal,
            math.log((1.0 + 4.0 / s) / (1.0 + 2.0 / s)),
            1e-13,
        )
        rho = 1.0 - 1e-6
        near = linear_design_criteria(
            rows,
            None,
            (0, 0),
            prior_covariance=p,
            noise_covariance=((s, rho * s), (rho * s, s)),
        )
        self.assertLess(abs(near.d_optimal - single.d_optimal), 1e-5)
        independent = linear_design_criteria(
            rows,
            None,
            (0, 0),
            prior_covariance=p,
            noise_covariance=((s, 0.0), (0.0, s)),
        )
        _close(self, independent.d_optimal, replicate.d_optimal, 1e-13)

    def test_exactly_singular_noise_is_rejected(self) -> None:
        """A perfectly correlated pair has no Cholesky factor: error, no repair."""
        with self.assertRaises(ValueError):
            linear_design_criteria(
                ((1.0,),),
                None,
                (0, 0),
                prior_covariance=((1.0,),),
                noise_covariance=((1.0, 1.0), (1.0, 1.0)),
            )

    def test_zero_row_has_zero_gain(self) -> None:
        """A zero functional observes nothing: no gain, never selected."""
        rows = ((0.0, 0.0), (1.0, 0.0))
        prior = ((1.0, 0.0), (0.0, 1.0))
        base = linear_design_criteria(rows, (1.0, 1.0), (1,), prior_covariance=prior)
        both = linear_design_criteria(rows, (1.0, 1.0), (0, 1), prior_covariance=prior)
        self.assertEqual(both.d_optimal, base.d_optimal)
        self.assertEqual(both.posterior_covariance, base.posterior_covariance)
        result = greedy_linear_design(
            rows, (1.0, 1.0), prior_covariance=prior, budget=2, criterion="d"
        )
        self.assertEqual(result.indices, (1,))
        self.assertEqual(result.stop_reason, "no_positive_gain")
        only_zero = greedy_linear_design(
            rows[:1], (1.0,), prior_covariance=prior, budget=1
        )
        self.assertEqual(only_zero.indices, ())

    def test_classical_rank_deficient(self) -> None:
        """No prior: -inf log-determinant, inf variance, null space reported."""
        rows = ((1.0, 0.0), (1.0, 0.0))
        result = linear_design_criteria(rows, (1.0, 1.0), (0, 1), goal=(1.0, 0.0))
        self.assertEqual(result.mode, "classical")
        self.assertTrue(result.rank_deficient)
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.d_optimal, -math.inf)
        self.assertEqual(result.a_optimal, math.inf)
        self.assertEqual(result.max_covariance_eigenvalue, math.inf)
        self.assertIsNone(result.expected_information_gain)
        self.assertIsNone(result.posterior_covariance)
        self.assertEqual(len(result.nullspace), 1)
        _close(self, abs(result.nullspace[0][1]), 1.0, 1e-12)
        # c = e1 is estimable (variance 1/2); e2 is not.
        assert result.goal_variance is not None
        _close(self, result.goal_variance, 0.5, 1e-12)
        blind = linear_design_criteria(rows, (1.0, 1.0), (0, 1), goal=(0.0, 1.0))
        self.assertEqual(blind.goal_variance, math.inf)
        # the Bayesian analysis of the same design is finite
        proper = linear_design_criteria(
            rows, (1.0, 1.0), (0, 1), prior_covariance=((1.0, 0.0), (0.0, 1.0))
        )
        self.assertTrue(math.isfinite(proper.d_optimal))
        self.assertTrue(proper.rank_deficient)

    def test_classical_full_rank(self) -> None:
        """Full-rank Fisher matrix: log det F and trace F^-1 are finite."""
        rows = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        result = linear_design_criteria(rows, (1.0, 2.0, 4.0), (0, 1, 2))
        # F = [[1.25, .25], [.25, .75]]
        det = 1.25 * 0.75 - 0.0625
        _close(self, result.d_optimal, math.log(det), 1e-13)
        _close(self, result.a_optimal, (1.25 + 0.75) / det, 1e-13)
        self.assertFalse(result.rank_deficient)


class GreedyTests(unittest.TestCase):
    """Rank-one updates, budgets, ties, relabeling and the oracle."""

    n, count = 3, 8
    rows = _rows(8, 3, 21)
    prior = _spd(3, 22)
    variances = tuple(0.4 + 0.15 * j for j in range(8))

    def test_rank_one_equals_exact_recomputation(self) -> None:
        """Final covariance, values and gains match full recomputation."""
        for criterion, goal in (("d", None), ("a", None), ("goal", (1.0, -1.0, 0.5))):
            result = greedy_linear_design(
                self.rows,
                self.variances,
                prior_covariance=self.prior,
                budget=4,
                criterion=criterion,
                goal=goal,
            )
            self.assertEqual(len(result.indices), 4)
            exact = linear_design_criteria(
                self.rows,
                self.variances,
                result.indices,
                prior_covariance=self.prior,
                goal=(1.0, -1.0, 0.5),
            )
            assert exact.posterior_covariance is not None
            for i in range(self.n):
                for j in range(self.n):
                    _close(
                        self,
                        result.posterior_covariance[i][j],
                        exact.posterior_covariance[i][j],
                        1e-10,
                    )
            _close(self, math.fsum(result.log_det_gains), exact.d_optimal, 1e-10)
            final = {
                "d": exact.d_optimal,
                "a": exact.a_optimal,
                "goal": exact.goal_variance,
            }[criterion]
            assert final is not None
            _close(self, result.values[-1], final, 1e-10)
            # each step's gain equals the exact criterion change
            previous = result.initial_value
            for step, value in enumerate(result.values):
                _close(self, abs(value - previous), result.gains[step], 1e-10)
                previous = value

    def test_d_gains_diminish(self) -> None:
        """Independent-noise D gains are non-increasing along the greedy path."""
        result = greedy_linear_design(
            self.rows, self.variances, prior_covariance=self.prior, budget=6
        )
        gains = result.log_det_gains
        for a, b in zip(gains, gains[1:], strict=False):
            self.assertGreaterEqual(a + 1e-12, b)

    def test_greedy_versus_exhaustive_gap(self) -> None:
        """Greedy never beats the oracle; the (reported) gap is small here."""
        gaps = []
        for criterion, goal in (("d", None), ("a", None), ("goal", (0.0, 1.0, 1.0))):
            greedy = greedy_linear_design(
                self.rows,
                self.variances,
                prior_covariance=self.prior,
                budget=3,
                criterion=criterion,
                goal=goal,
            )
            oracle = exhaustive_linear_design(
                self.rows,
                self.variances,
                prior_covariance=self.prior,
                size=3,
                criterion=criterion,
                goal=goal,
            )
            self.assertEqual(oracle.combinations, 56)
            self.assertEqual(len(oracle.ranking), 56)
            self.assertEqual(oracle.ranking[0], (oracle.indices, oracle.value))
            achieved = greedy.values[-1]
            if criterion == "d":
                self.assertLessEqual(achieved, oracle.value + 1e-12)
                gaps.append(oracle.value - achieved)
            else:
                self.assertGreaterEqual(achieved, oracle.value - 1e-12)
                gaps.append(achieved - oracle.value)
        # documented, not asserted zero: gaps are nonnegative and moderate
        for gap in gaps:
            self.assertGreaterEqual(gap, -1e-12)
            self.assertLess(gap, 1.0)

    def test_orthogonal_candidates_greedy_is_optimal(self) -> None:
        """With orthogonal rows and P = I the greedy set equals the optimum."""
        rows = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0))
        variances = (1.0, 0.5, 2.0, 3.0)
        prior = tuple(tuple(1.0 if i == j else 0.0 for j in range(3)) for i in range(3))
        for criterion in ("d", "a"):
            greedy = greedy_linear_design(
                rows,
                variances,
                prior_covariance=prior,
                budget=2,
                criterion=criterion,
            )
            oracle = exhaustive_linear_design(
                rows,
                variances,
                prior_covariance=prior,
                size=2,
                criterion=criterion,
            )
            self.assertEqual(tuple(sorted(greedy.indices)), oracle.indices)
            _close(self, greedy.values[-1], oracle.value, 1e-12)

    def test_goal_choice_differs_from_d_optimal(self) -> None:
        """A target on x1 picks the weak sensor on x1; D picks the strong one."""
        rows = ((1.0, 0.0), (0.0, 1.0))
        variances = (1.0, 0.25)
        prior = ((1.0, 0.0), (0.0, 1.0))
        by_d = greedy_linear_design(
            rows, variances, prior_covariance=prior, budget=1, criterion="d"
        )
        by_goal = greedy_linear_design(
            rows,
            variances,
            prior_covariance=prior,
            budget=1,
            criterion="goal",
            goal=(1.0, 0.0),
        )
        self.assertEqual(by_d.indices, (1,))
        self.assertEqual(by_goal.indices, (0,))
        _close(self, by_goal.values[-1], 0.5, 1e-14)

    def test_prior_unit_invariance(self) -> None:
        """A reparameterization x -> D x keeps the D and goal criteria."""
        scale = (3.0, 0.2, 7.0)
        rows2 = tuple(
            tuple(v / s for v, s in zip(row, scale, strict=True)) for row in self.rows
        )
        prior2 = tuple(
            tuple(self.prior[i][j] * scale[i] * scale[j] for j in range(3))
            for i in range(3)
        )
        c = (1.0, -2.0, 0.5)
        c2 = tuple(v / s for v, s in zip(c, scale, strict=True))
        picks = (1, 4, 6)
        a = linear_design_criteria(
            self.rows, self.variances, picks, prior_covariance=self.prior, goal=c
        )
        b = linear_design_criteria(
            rows2, self.variances, picks, prior_covariance=prior2, goal=c2
        )
        _close(self, a.d_optimal, b.d_optimal, 1e-10)
        assert a.goal_variance is not None and b.goal_variance is not None
        _close(self, a.goal_variance, b.goal_variance, 1e-10)
        _close(self, a.e_information_eigenvalue, b.e_information_eigenvalue, 1e-10)
        # A and the state-coordinate E criterion are metric dependent on purpose
        self.assertGreater(abs(a.a_optimal - b.a_optimal), 1e-3)
        self.assertGreater(
            abs(a.max_covariance_eigenvalue - b.max_covariance_eigenvalue), 1e-3
        )
        for criterion in ("d",):
            ga = greedy_linear_design(
                self.rows,
                self.variances,
                prior_covariance=self.prior,
                budget=4,
                criterion=criterion,
            )
            gb = greedy_linear_design(
                rows2,
                self.variances,
                prior_covariance=prior2,
                budget=4,
                criterion=criterion,
            )
            self.assertEqual(ga.indices, gb.indices)

    def test_relabeling_invariance(self) -> None:
        """Permuting candidates permutes the selection (no ties in this data)."""
        order = (5, 2, 7, 0, 3, 6, 1, 4)  # new position p holds old index order[p]
        rows = tuple(self.rows[i] for i in order)
        variances = tuple(self.variances[i] for i in order)
        for criterion, goal in (("d", None), ("a", None), ("goal", (1.0, 1.0, 0.0))):
            base = greedy_linear_design(
                self.rows,
                self.variances,
                prior_covariance=self.prior,
                budget=4,
                criterion=criterion,
                goal=goal,
            )
            moved = greedy_linear_design(
                rows,
                variances,
                prior_covariance=self.prior,
                budget=4,
                criterion=criterion,
                goal=goal,
            )
            self.assertEqual(tuple(order[i] for i in moved.indices), base.indices)

    def test_ties_go_to_lowest_index(self) -> None:
        """Identical candidates: the first index wins, then the next."""
        rows = ((1.0, 0.0),) * 3
        prior = ((1.0, 0.0), (0.0, 1.0))
        result = greedy_linear_design(
            rows, (1.0,) * 3, prior_covariance=prior, budget=2, criterion="a"
        )
        self.assertEqual(result.indices, (0, 1))

    def test_repeats(self) -> None:
        """Replicates are only used when allowed."""
        rows = ((1.0, 0.0), (0.0, 0.1))
        prior = ((1.0, 0.0), (0.0, 1.0))
        no = greedy_linear_design(rows, (1.0, 1.0), prior_covariance=prior, budget=3)
        self.assertEqual(sorted(no.indices), [0, 1])
        self.assertEqual(no.stop_reason, "no_affordable_candidate")
        yes = greedy_linear_design(
            rows, (1.0, 1.0), prior_covariance=prior, budget=3, allow_repeats=True
        )
        self.assertEqual(len(yes.indices), 3)
        self.assertEqual(yes.indices[0], 0)
        self.assertEqual(yes.stop_reason, "budget_exhausted")

    def test_costs_and_budget_respected(self) -> None:
        """Total cost never exceeds the budget; gain per cost changes the order."""
        costs = (5.0, 1.0, 1.0, 4.0, 1.0, 2.0, 3.0, 1.0)
        for ratio in (False, True):
            result = greedy_linear_design(
                self.rows,
                self.variances,
                prior_covariance=self.prior,
                budget=6.5,
                costs=costs,
                gain_per_cost=ratio,
            )
            spent = math.fsum(costs[i] for i in result.indices)
            self.assertEqual(result.total_cost, spent)
            self.assertLessEqual(spent, 6.5)
            self.assertEqual(result.budget, 6.5)
            self.assertEqual(len(set(result.indices)), len(result.indices))
        # a budget that affords nothing selects nothing
        none = greedy_linear_design(
            self.rows,
            self.variances,
            prior_covariance=self.prior,
            budget=0.5,
            costs=costs,
        )
        self.assertEqual(none.indices, ())
        self.assertEqual(none.stop_reason, "no_affordable_candidate")
        self.assertEqual(none.total_cost, 0.0)
        count = greedy_linear_design(
            self.rows, self.variances, prior_covariance=self.prior, budget=3
        )
        self.assertEqual(count.total_cost, 3.0)


class ExhaustiveTests(unittest.TestCase):
    """Oracle behaviour and its guards."""

    def test_cap_and_size(self) -> None:
        """A too-large enumeration and a bad size are refused."""
        rows = _rows(6, 2, 5)
        prior = _spd(2, 6)
        with self.assertRaises(ValueError):
            exhaustive_linear_design(
                rows, (1.0,) * 6, prior_covariance=prior, size=3, max_combinations=10
            )
        with self.assertRaises(ValueError):
            exhaustive_linear_design(rows, (1.0,) * 6, prior_covariance=prior, size=7)
        with self.assertRaises(ValueError):
            exhaustive_linear_design(rows, (1.0,) * 6, prior_covariance=prior, size=0)

    def test_ranking_is_sorted(self) -> None:
        """D ranking is descending and A ranking ascending."""
        rows = _rows(5, 2, 8)
        prior = _spd(2, 9)
        d = exhaustive_linear_design(rows, (1.0,) * 5, prior_covariance=prior, size=2)
        values = [v for _, v in d.ranking]
        self.assertEqual(values, sorted(values, reverse=True))
        a = exhaustive_linear_design(
            rows, (1.0,) * 5, prior_covariance=prior, size=2, criterion="a"
        )
        values = [v for _, v in a.ranking]
        self.assertEqual(values, sorted(values))


class InvalidInputTests(unittest.TestCase):
    """Bad arguments raise instead of being repaired."""

    rows = ((1.0, 0.0), (0.0, 1.0))
    prior = ((1.0, 0.0), (0.0, 1.0))

    def test_criteria_inputs(self) -> None:
        """Shape, symmetry, definiteness, index and noise-argument errors."""
        cases = [
            dict(noise_variances=(1.0,), selected=(0,)),
            dict(noise_variances=(1.0, -1.0), selected=(0,)),
            dict(noise_variances=(1.0, 1.0), selected=(2,)),
            dict(noise_variances=(1.0, 1.0), selected=(-1,)),
            dict(noise_variances=(1.0, 1.0), selected=(0,), goal=(1.0,)),
            dict(noise_variances=(1.0, 1.0), selected=(0,), goal=()),
            dict(noise_variances=None, selected=(0,)),
            dict(
                noise_variances=(1.0, 1.0),
                selected=(0,),
                noise_covariance=((1.0,),),
            ),
            dict(
                noise_variances=None,
                selected=(0, 1),
                noise_covariance=((1.0, 0.5), (0.0, 1.0)),
            ),
            dict(noise_variances=(1.0, 1.0), selected=(0,), prior_covariance=((1.0,),)),
            dict(
                noise_variances=(1.0, 1.0),
                selected=(0,),
                prior_covariance=((1.0, 0.5), (0.0, 1.0)),
            ),
            dict(
                noise_variances=(1.0, 1.0),
                selected=(0,),
                prior_covariance=((1.0, 2.0), (2.0, 1.0)),
            ),
        ]
        for case in cases:
            kwargs = dict(case)
            kwargs.setdefault("prior_covariance", self.prior)
            with self.subTest(case=case), self.assertRaises(ValueError):
                linear_design_criteria(
                    self.rows,
                    kwargs.pop("noise_variances"),  # type: ignore[arg-type]
                    kwargs.pop("selected"),  # type: ignore[arg-type]
                    **kwargs,  # type: ignore[arg-type]
                )
        with self.assertRaises(TypeError):
            linear_design_criteria(
                self.rows,
                (1.0, 1.0),
                (0.5,),  # type: ignore[arg-type]
                prior_covariance=self.prior,
            )
        with self.assertRaises(ValueError):
            linear_design_criteria((), (), (), prior_covariance=self.prior)
        with self.assertRaises(ValueError):
            linear_design_criteria(((1.0, 0.0), (1.0,)), (1.0, 1.0), (0,))

    def test_greedy_inputs(self) -> None:
        """Criterion, goal, budget, cost and correlated-noise errors."""
        base = dict(prior_covariance=self.prior)
        with self.assertRaises(ValueError):
            greedy_linear_design(self.rows, (1.0, 1.0), budget=1, criterion="e", **base)
        with self.assertRaises(ValueError):
            greedy_linear_design(
                self.rows, (1.0, 1.0), budget=1, criterion="goal", **base
            )
        with self.assertRaises(ValueError):
            greedy_linear_design(
                self.rows, (1.0, 1.0), budget=1, criterion="d", goal=(1.0, 0.0), **base
            )
        with self.assertRaises(ValueError):
            greedy_linear_design(self.rows, (1.0, 1.0), budget=0, **base)
        with self.assertRaises(ValueError):
            greedy_linear_design(self.rows, (1.0, 0.0), budget=1, **base)
        with self.assertRaises(ValueError):
            greedy_linear_design(
                self.rows, (1.0, 1.0), budget=1.0, costs=(1.0, 0.0), **base
            )
        with self.assertRaises(ValueError):
            greedy_linear_design(
                self.rows, (1.0, 1.0), budget=-1.0, costs=(1.0, 1.0), **base
            )
        with self.assertRaises(ValueError):
            greedy_linear_design(
                self.rows, (1.0, 1.0), budget=1.0, costs=(1.0,), **base
            )
        with self.assertRaises(ValueError):
            greedy_linear_design(
                self.rows,
                (1.0, 1.0),
                budget=1,
                prior_covariance=((1.0, 2.0), (2.0, 1.0)),
            )
        with self.assertRaises(TypeError):
            greedy_linear_design(
                self.rows,
                (1.0, 1.0),
                budget=1,
                noise_covariance=((1.0, 0.0), (0.0, 1.0)),  # type: ignore[call-arg]
                **base,
            )


if __name__ == "__main__":
    unittest.main()
