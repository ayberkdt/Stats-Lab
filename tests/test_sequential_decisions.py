"""Tests for contextual bandits, off-policy evaluation, MDPs, and best-arm search."""

from __future__ import annotations

import itertools
import math
import random
import statistics
import unittest

from probstat.inference import condition_linear_gaussian
from probstat.linalg import identity, inverse_spd, solve_spd
from probstat.multivariate import MultivariateNormal
from probstat.risk import (
    FiniteMDP,
    LinearThompsonSampling,
    LinUCB,
    LoggedBanditData,
    action_values,
    backward_induction,
    bellman_residual,
    direct_method,
    doubly_robust,
    greedy_policy,
    inverse_propensity_score,
    logged_bandit_data,
    policy_evaluation,
    policy_iteration,
    self_normalized_ips,
    sequential_halving,
    simulate_linear_bandit,
    value_iteration,
)


def random_mdp(rng: random.Random, n_states: int, n_actions: int) -> FiniteMDP:
    """Return a dense random MDP with a random discount in ``[0.3, 0.95]``."""
    transitions = []
    rewards = []
    for _ in range(n_actions):
        layer = []
        for _ in range(n_states):
            raw = [rng.random() + 0.05 for _ in range(n_states)]
            total = sum(raw)
            layer.append(tuple(v / total for v in raw))
        transitions.append(tuple(layer))
        rewards.append(tuple(rng.uniform(-1.0, 2.0) for _ in range(n_states)))
    return FiniteMDP(tuple(transitions), tuple(rewards), rng.uniform(0.3, 0.95))


def brute_force(mdp: FiniteMDP) -> tuple[tuple[float, ...], float]:
    """Return the pointwise best value over all deterministic policies."""
    best = [-math.inf] * mdp.n_states
    best_total = -math.inf
    for policy in itertools.product(range(mdp.n_actions), repeat=mdp.n_states):
        values = policy_evaluation(mdp, policy)
        best = [max(b, v) for b, v in zip(best, values, strict=True)]
        best_total = max(best_total, sum(values))
    return tuple(best), best_total


class FiniteMDPTests(unittest.TestCase):
    """Dynamic programming against closed forms and exhaustive enumeration."""

    def test_single_state_geometric_series(self) -> None:
        """One state with reward 1 and gamma 0.9 has value 1 / (1 - 0.9) = 10."""
        mdp = FiniteMDP((((1.0,),),), ((1.0,),), 0.9)
        self.assertAlmostEqual(policy_evaluation(mdp, (0,))[0], 10.0, places=12)
        exact = policy_iteration(mdp)
        self.assertTrue(exact.converged)
        self.assertAlmostEqual(exact.values[0], 10.0, places=12)
        approximate = value_iteration(mdp, tolerance=1e-9)
        self.assertTrue(approximate.converged)
        self.assertAlmostEqual(approximate.values[0], 10.0, delta=5e-10)

    def test_solvers_agree_with_enumeration_of_all_policies(self) -> None:
        """VI, PI, and brute force give the same optimal values on random MDPs."""
        rng = random.Random(20261121)
        for case in range(12):
            mdp = random_mdp(rng, rng.choice((2, 3, 4)), rng.choice((2, 3)))
            best, best_total = brute_force(mdp)
            exact = policy_iteration(mdp)
            approximate = value_iteration(mdp, tolerance=1e-10)
            with self.subTest(case=case):
                self.assertTrue(exact.converged and approximate.converged)
                for b, p, v in zip(best, exact.values, approximate.values, strict=True):
                    self.assertAlmostEqual(p, b, delta=1e-10)
                    self.assertAlmostEqual(v, b, delta=5e-11)
                # The pointwise maximum is attained by one single policy.
                self.assertAlmostEqual(sum(exact.values), best_total, delta=1e-9)
                self.assertLess(exact.bellman_residual, 1e-12)
                self.assertLess(bellman_residual(mdp, approximate.values), 1e-10)
                self.assertEqual(exact.policy, approximate.policy)
                self.assertEqual(exact.policy, greedy_policy(mdp, best))

    def test_stopping_rule_guarantees_epsilon_optimal_policy(self) -> None:
        """A loose tolerance still bounds the value error and the policy loss."""
        rng = random.Random(7)
        for case in range(10):
            mdp = random_mdp(rng, 4, 3)
            best, _ = brute_force(mdp)
            for eps in (1.0, 0.1, 0.01):
                solution = value_iteration(mdp, tolerance=eps)
                achieved = policy_evaluation(mdp, solution.policy)
                loss = max(b - a for b, a in zip(best, achieved, strict=True))
                error = max(
                    abs(b - v) for b, v in zip(best, solution.values, strict=True)
                )
                with self.subTest(case=case, eps=eps):
                    self.assertTrue(solution.converged)
                    self.assertLess(loss, eps)
                    self.assertLess(error, eps / 2.0)
                    self.assertLessEqual(error, solution.value_error_bound + 1e-12)
                    self.assertLessEqual(loss, solution.policy_loss_bound + 1e-12)
                    self.assertLessEqual(
                        solution.bellman_residual,
                        mdp.discount * solution.last_change + 1e-12,
                    )

    def test_stopping_rule_when_the_greedy_policy_is_not_optimal(self) -> None:
        """The epsilon bound holds where value iteration stops at a worse policy.

        Action 0 stays (rewards 0.7 and 0.8); action 1 moves with
        probabilities (0.2, 0.8) from state 0 at reward 0. With ``gamma = 0.9``
        the optimum plays action 1 in state 0: ``V*(1) = 8`` and
        ``V*(0) = 0.9 * 0.8 * 8 / (1 - 0.18) = 7.02439``, against 7 for
        staying. With ``eps = 1`` value iteration stops while staying still
        looks better, so the returned policy loses 0.02439, inside the bound.
        """
        mdp = FiniteMDP(
            (((1.0, 0.0), (0.0, 1.0)), ((0.2, 0.8), (0.3, 0.7))),
            ((0.7, 0.8), (0.0, 0.7)),
            0.9,
        )
        optimum = policy_iteration(mdp)
        self.assertEqual(optimum.policy, (1, 0))
        self.assertAlmostEqual(optimum.values[0], 5.76 / 0.82, places=12)
        self.assertAlmostEqual(optimum.values[1], 8.0, places=12)
        loose = value_iteration(mdp, tolerance=1.0)
        self.assertTrue(loose.converged)
        self.assertEqual(loose.policy, (0, 0))
        achieved = policy_evaluation(mdp, loose.policy)
        loss = max(b - a for b, a in zip(optimum.values, achieved, strict=True))
        self.assertAlmostEqual(loss, 5.76 / 0.82 - 7.0, places=12)
        self.assertGreater(loss, 0.02)
        self.assertLess(loss, 1.0)
        self.assertLessEqual(loss, loose.policy_loss_bound)
        error = max(
            abs(b - v) for b, v in zip(optimum.values, loose.values, strict=True)
        )
        self.assertLess(error, 0.5)
        self.assertLessEqual(error, loose.value_error_bound + 1e-12)
        tight = value_iteration(mdp, tolerance=1e-3)
        self.assertEqual(tight.policy, (1, 0))

    def test_policy_iteration_terminates_with_tied_actions(self) -> None:
        """Duplicated actions and tied rewards neither cycle nor lose value."""
        rng = random.Random(11)
        for case in range(20):
            base = random_mdp(rng, 3, 2)
            tied = FiniteMDP(
                (*base.transitions, base.transitions[0]),
                (*base.expected_rewards, base.expected_rewards[0]),
                base.discount,
            )
            best, _ = brute_force(base)
            for start in (None, (2, 2, 2), (0, 2, 1)):
                solution = policy_iteration(tied, initial_policy=start)
                with self.subTest(case=case, start=start):
                    self.assertTrue(solution.converged)
                    self.assertLessEqual(solution.iterations, 3**3)
                    for b, v in zip(best, solution.values, strict=True):
                        self.assertAlmostEqual(v, b, delta=1e-10)

    def test_transition_rewards_reduce_to_expected_rewards(self) -> None:
        """R[a][s][s'] gives the same problem as its expectation R[a][s]."""
        transitions = (((0.2, 0.8), (0.5, 0.5)), ((1.0, 0.0), (0.3, 0.7)))
        by_transition = (((1.0, 2.0), (0.0, 4.0)), ((3.0, 9.0), (-1.0, 1.0)))
        mdp = FiniteMDP(transitions, by_transition, 0.8)
        self.assertEqual(mdp.expected_rewards[0], (0.2 + 1.6, 2.0))
        self.assertAlmostEqual(mdp.expected_rewards[1][1], -0.3 + 0.7, places=15)
        reduced = FiniteMDP(transitions, mdp.expected_rewards, 0.8)
        self.assertEqual(policy_iteration(mdp).values, policy_iteration(reduced).values)

    def test_stochastic_policy_satisfies_its_bellman_equation(self) -> None:
        """A randomized policy's values solve V = r_pi + gamma P_pi V."""
        mdp = random_mdp(random.Random(3), 3, 2)
        pi = ((0.3, 0.7), (1.0, 0.0), (0.5, 0.5))
        values = policy_evaluation(mdp, pi)
        table = action_values(mdp, values)
        for s in range(3):
            mixed = sum(p * q for p, q in zip(pi[s], table[s], strict=True))
            self.assertAlmostEqual(values[s], mixed, places=12)
        # A deterministic row equals the index form.
        index_form = policy_evaluation(mdp, (1, 0, 1))
        matrix_form = policy_evaluation(mdp, ((0.0, 1.0), (1.0, 0.0), (0.0, 1.0)))
        for a, b in zip(index_form, matrix_form, strict=True):
            self.assertAlmostEqual(a, b, places=13)

    def test_backward_induction_two_stage_problem_by_hand(self) -> None:
        """Two stages of a stay/switch problem match the hand computation.

        Stay pays 1 in state 0 and 3 in state 1; switch pays 0 and moves.
        Undiscounted: V1 = (1, 3), V0 = (max(1+1, 0+3), max(3+3, 0+1)) = (3, 6).
        With gamma 0.9: V0 = (max(1.9, 2.7), max(5.7, 0.9)) = (2.7, 5.7).
        """
        stay = ((1.0, 0.0), (0.0, 1.0))
        switch = ((0.0, 1.0), (1.0, 0.0))
        mdp = FiniteMDP((stay, switch), ((1.0, 3.0), (0.0, 0.0)), 0.9)
        plain = backward_induction(mdp, 2, discount=1.0)
        self.assertEqual(plain.values, ((3.0, 6.0), (1.0, 3.0), (0.0, 0.0)))
        self.assertEqual(plain.policy, ((1, 0), (0, 0)))
        discounted = backward_induction(mdp, 2)
        self.assertAlmostEqual(discounted.values[0][0], 2.7, places=14)
        self.assertAlmostEqual(discounted.values[0][1], 5.7, places=14)
        self.assertEqual(discounted.backups, 8)
        # A terminal reward of 10 in state 0 makes switching from state 1 optimal.
        terminal = backward_induction(mdp, 1, terminal_values=(10.0, 0.0), discount=1.0)
        self.assertEqual(terminal.values[0], (11.0, 10.0))
        self.assertEqual(terminal.policy, ((0, 1),))

    def test_long_horizon_approaches_discounted_solution(self) -> None:
        """Backward induction over many stages converges to the stationary values."""
        mdp = random_mdp(random.Random(11), 3, 3)
        stationary = policy_iteration(mdp)
        horizon = math.ceil(math.log(1e-10) / math.log(mdp.discount)) + 5
        finite = backward_induction(mdp, horizon)
        for a, b in zip(finite.values[0], stationary.values, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-8)
        self.assertEqual(finite.policy[0], stationary.policy)

    def test_zero_discount_is_myopic(self) -> None:
        """With gamma = 0 the optimal policy maximizes the immediate reward."""
        rng = random.Random(5)
        mdp = random_mdp(rng, 4, 3)
        myopic = FiniteMDP(mdp.transitions, mdp.rewards, 0.0)
        expected_policy = tuple(
            max(range(3), key=lambda a: myopic.expected_rewards[a][s]) for s in range(4)
        )
        expected_values = tuple(
            max(myopic.expected_rewards[a][s] for a in range(3)) for s in range(4)
        )
        swept = value_iteration(myopic)
        self.assertEqual(swept.iterations, 1)
        self.assertTrue(swept.converged)
        self.assertEqual(swept.policy, expected_policy)
        self.assertEqual(swept.values, expected_values)
        self.assertEqual(policy_iteration(myopic).policy, expected_policy)

    def test_unconverged_runs_are_flagged(self) -> None:
        """Hitting the iteration limit returns converged=False with a residual."""
        mdp = random_mdp(random.Random(9), 3, 2)
        stopped = value_iteration(mdp, tolerance=1e-12, max_iterations=2)
        self.assertFalse(stopped.converged)
        self.assertEqual(stopped.iterations, 2)
        self.assertGreater(stopped.bellman_residual, 1e-6)
        self.assertEqual(stopped.backups, 3 * 3 * 2)
        # Policy iteration stopped after one step from a non-optimal start.
        optimal = policy_iteration(mdp)
        worst = tuple(1 - a for a in optimal.policy)
        limited = policy_iteration(mdp, initial_policy=worst, max_iterations=1)
        self.assertFalse(limited.converged)
        self.assertEqual(limited.linear_solves, 2)
        resumed = policy_iteration(mdp, initial_policy=worst)
        self.assertTrue(resumed.converged)
        self.assertEqual(resumed.policy, optimal.policy)

    def test_labels_and_validation(self) -> None:
        """Invalid tensors, discounts, policies, and arguments raise."""
        stay = ((1.0, 0.0), (0.0, 1.0))
        mdp = FiniteMDP(
            (stay,), ((1.0, 2.0),), 0.5, states=("up", "down"), actions=("wait",)
        )
        self.assertEqual(mdp.states, ("up", "down"))
        self.assertEqual((mdp.n_states, mdp.n_actions), (2, 1))
        self.assertEqual(FiniteMDP((stay,), ((1.0, 2.0),), 0.5).states, ("0", "1"))
        with self.assertRaisesRegex(ValueError, "discount"):
            FiniteMDP((stay,), ((1.0, 2.0),), 1.0)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            FiniteMDP((((0.5, 0.4), (0.0, 1.0)),), ((1.0, 2.0),), 0.5)
        with self.assertRaisesRegex(ValueError, "negative"):
            FiniteMDP((((1.5, -0.5), (0.0, 1.0)),), ((1.0, 2.0),), 0.5)
        with self.assertRaisesRegex(ValueError, "rewards"):
            FiniteMDP((stay,), ((1.0,),), 0.5)
        with self.assertRaisesRegex(ValueError, "one block per action"):
            FiniteMDP((stay,), ((1.0, 2.0), (1.0, 2.0)), 0.5)
        with self.assertRaisesRegex(ValueError, "distinct"):
            FiniteMDP((stay,), ((1.0, 2.0),), 0.5, states=("a", "a"))
        with self.assertRaisesRegex(ValueError, "rows"):
            FiniteMDP((stay, ((1.0, 0.0),)), ((1.0, 2.0), (1.0, 2.0)), 0.5)
        with self.assertRaisesRegex(ValueError, "action index"):
            policy_evaluation(mdp, (0, 1))
        with self.assertRaisesRegex(ValueError, "one entry per state"):
            policy_evaluation(mdp, (0,))
        with self.assertRaisesRegex(ValueError, "summing to one"):
            policy_evaluation(mdp, ((0.5,), (1.0,)))
        with self.assertRaisesRegex(ValueError, "tolerance"):
            value_iteration(mdp, tolerance=0.0)
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            policy_iteration(mdp, max_iterations=0)
        with self.assertRaisesRegex(ValueError, "deterministic"):
            policy_iteration(mdp, initial_policy=((1.0,), (1.0,)))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "horizon"):
            backward_induction(mdp, 0)
        with self.assertRaisesRegex(ValueError, "discount"):
            backward_induction(mdp, 1, discount=1.5)
        with self.assertRaisesRegex(ValueError, "one entry per state"):
            bellman_residual(mdp, (1.0,))
        with self.assertRaises(TypeError):
            value_iteration("mdp")  # type: ignore[arg-type]


CONTEXT_PROBABILITY = (0.4, 0.6)
LOGGING = ((0.5, 0.3, 0.2), (0.1, 0.6, 0.3))
TARGET = ((0.1, 0.2, 0.7), (0.6, 0.0, 0.4))
MEAN_REWARD = ((0.2, 0.5, 0.9), (0.7, 0.4, 0.1))
WRONG_PROPENSITY = ((0.3, 0.3, 0.4), (0.4, 0.2, 0.4))
WRONG_MODEL = ((1.5, -2.0, 0.3), (0.0, 4.0, -1.0))
TRUE_VALUE = sum(
    CONTEXT_PROBABILITY[x] * TARGET[x][a] * MEAN_REWARD[x][a]
    for x in range(2)
    for a in range(3)
)


def target(context: object) -> tuple[float, ...]:
    """Return the tabular target policy row of a context."""
    return TARGET[context]  # type: ignore[index]


def enumerate_expectation(
    estimator: object, logged: tuple[tuple[float, ...], ...]
) -> float:
    """Return E[estimate] over all two-record logs of the tabular problem.

    Records are drawn independently from the true context law, the true
    logging policy, and Bernoulli rewards; ``logged`` is the propensity table
    written into the log (true or deliberately wrong).
    """
    outcomes = []
    for x in range(2):
        for a in range(3):
            for r in (0.0, 1.0):
                m = MEAN_REWARD[x][a]
                probability = (
                    CONTEXT_PROBABILITY[x]
                    * LOGGING[x][a]
                    * (m if r == 1.0 else 1.0 - m)
                )
                outcomes.append(((x, a, r, logged[x][a]), probability))
    total = 0.0
    mass = 0.0
    for (first, p1), (second, p2) in itertools.product(outcomes, repeat=2):
        data = logged_bandit_data((first, second), n_actions=3)
        total += p1 * p2 * estimator(data).value  # type: ignore[operator]
        mass += p1 * p2
    assert abs(mass - 1.0) < 1e-12
    return total


def sample_log(n: int, seed: int) -> LoggedBanditData:
    """Draw a seeded log from the tabular problem with true propensities."""
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        x = 0 if rng.random() < CONTEXT_PROBABILITY[0] else 1
        a = rng.choices(range(3), weights=LOGGING[x])[0]
        r = 1.0 if rng.random() < MEAN_REWARD[x][a] else 0.0
        records.append((x, a, r, LOGGING[x][a]))
    return logged_bandit_data(records, n_actions=3)


class OffPolicyEvaluationTests(unittest.TestCase):
    """Exact expectations by enumeration and algebraic identities."""

    def test_ips_is_exactly_unbiased(self) -> None:
        """E[IPS] over all 144 two-record logs equals V(pi)."""
        expectation = enumerate_expectation(
            lambda data: inverse_propensity_score(data, target), LOGGING
        )
        self.assertAlmostEqual(expectation, TRUE_VALUE, places=13)

    def test_doubly_robust_is_unbiased_with_a_wrong_model(self) -> None:
        """True propensities and an arbitrary q give E[DR] = V(pi)."""
        expectation = enumerate_expectation(
            lambda data: doubly_robust(data, target, lambda x, a: WRONG_MODEL[x][a]),
            LOGGING,
        )
        self.assertAlmostEqual(expectation, TRUE_VALUE, places=13)

    def test_doubly_robust_is_unbiased_with_wrong_propensities(self) -> None:
        """The true q and wrong logged propensities give E[DR] = V(pi)."""
        expectation = enumerate_expectation(
            lambda data: doubly_robust(data, target, lambda x, a: MEAN_REWARD[x][a]),
            WRONG_PROPENSITY,
        )
        self.assertAlmostEqual(expectation, TRUE_VALUE, places=13)

    def test_biased_combinations_are_detected_by_the_same_enumeration(self) -> None:
        """IPS with wrong propensities, DM and DR with both wrong, are biased."""
        ips = enumerate_expectation(
            lambda data: inverse_propensity_score(data, target), WRONG_PROPENSITY
        )
        both = enumerate_expectation(
            lambda data: doubly_robust(data, target, lambda x, a: WRONG_MODEL[x][a]),
            WRONG_PROPENSITY,
        )
        direct_wrong = enumerate_expectation(
            lambda data: direct_method(data, target, lambda x, a: WRONG_MODEL[x][a]),
            LOGGING,
        )
        direct_true = enumerate_expectation(
            lambda data: direct_method(data, target, lambda x, a: MEAN_REWARD[x][a]),
            LOGGING,
        )
        self.assertGreater(abs(ips - TRUE_VALUE), 0.01)
        self.assertGreater(abs(both - TRUE_VALUE), 0.01)
        self.assertGreater(abs(direct_wrong - TRUE_VALUE), 0.01)
        self.assertAlmostEqual(direct_true, TRUE_VALUE, places=13)

    def test_clipping_introduces_downward_bias(self) -> None:
        """Clipped IPS has expectation below V(pi) for non-negative rewards."""
        clipped = enumerate_expectation(
            lambda data: inverse_propensity_score(data, target, clip=2.0), LOGGING
        )
        self.assertLess(clipped, TRUE_VALUE - 0.02)
        data = sample_log(200, 1)
        estimate = inverse_propensity_score(data, target, clip=2.0)
        self.assertEqual(estimate.clip, 2.0)
        self.assertLessEqual(estimate.max_weight, 2.0)
        self.assertGreater(estimate.clipped_fraction, 0.0)
        self.assertLessEqual(
            estimate.value, inverse_propensity_score(data, target).value
        )

    def test_true_model_and_deterministic_rewards_remove_the_weights(self) -> None:
        """With r = q(x, a) DR equals DM; one context then gives zero variance."""
        rng = random.Random(4)
        records = []
        for _ in range(50):
            x = rng.randrange(2)
            a = rng.choices(range(3), weights=LOGGING[x])[0]
            records.append((x, a, MEAN_REWARD[x][a], LOGGING[x][a]))
        data = logged_bandit_data(records, n_actions=3)

        def model(x: object, a: int) -> float:
            """Return the exact deterministic reward."""
            return MEAN_REWARD[x][a]  # type: ignore[index]

        robust = doubly_robust(data, target, model)
        direct = direct_method(data, target, model)
        for a, b in zip(robust.terms, direct.terms, strict=True):
            self.assertAlmostEqual(a, b, places=14)
        single = logged_bandit_data(
            [row for row in records if row[0] == 0], n_actions=3
        )
        constant = doubly_robust(single, target, model)
        self.assertLess(constant.standard_error, 1e-15)
        self.assertAlmostEqual(
            constant.value,
            sum(TARGET[0][a] * MEAN_REWARD[0][a] for a in range(3)),
            places=14,
        )
        self.assertGreater(
            inverse_propensity_score(single, target).standard_error, 0.01
        )

    def test_snips_shifts_with_the_rewards(self) -> None:
        """Adding c to all rewards adds c to SNIPS but not to IPS."""
        data = sample_log(300, 2)
        shifted = LoggedBanditData(
            data.contexts,
            data.actions,
            tuple(r + 5.0 for r in data.rewards),
            data.propensities,
            3,
        )
        base = self_normalized_ips(data, target)
        moved = self_normalized_ips(shifted, target)
        self.assertAlmostEqual(moved.value - base.value, 5.0, places=12)
        self.assertAlmostEqual(moved.standard_error, base.standard_error, places=12)
        self.assertAlmostEqual(statistics.fmean(base.terms), base.value, places=12)
        plain = inverse_propensity_score(shifted, target).value
        self.assertGreater(
            abs(plain - inverse_propensity_score(data, target).value - 5.0), 1e-3
        )
        rewards = data.rewards
        self.assertTrue(min(rewards) <= base.value <= max(rewards))

    def test_standard_errors_and_weight_diagnostics(self) -> None:
        """Standard errors match their formulas; ESS equals n for equal weights."""
        data = sample_log(400, 3)
        ips = inverse_propensity_score(data, target)
        self.assertAlmostEqual(ips.value, statistics.fmean(ips.terms), places=13)
        self.assertAlmostEqual(
            ips.standard_error, statistics.stdev(ips.terms) / math.sqrt(400), places=13
        )
        weights = [
            TARGET[x][a] / p  # type: ignore[index]
            for x, a, p in zip(
                data.contexts, data.actions, data.propensities, strict=True
            )
        ]
        self.assertAlmostEqual(
            ips.effective_sample_size,
            sum(weights) ** 2 / sum(w * w for w in weights),
            places=9,
        )
        self.assertAlmostEqual(ips.max_weight, max(weights), places=14)
        lower, upper = ips.interval()
        self.assertAlmostEqual(
            upper - lower, 2 * 1.959963984540054 * ips.standard_error
        )
        # Evaluating the logging policy itself gives unit weights.
        on_policy = self_normalized_ips(data, lambda x: LOGGING[x])  # type: ignore[index]
        self.assertAlmostEqual(on_policy.effective_sample_size, 400.0, places=9)
        self.assertAlmostEqual(on_policy.value, statistics.fmean(data.rewards))
        self.assertAlmostEqual(
            on_policy.standard_error,
            statistics.pstdev(data.rewards) / math.sqrt(400),
            places=13,
        )
        self.assertEqual(
            direct_method(data, target, lambda x, a: 0.0).effective_sample_size, 400.0
        )

    def test_estimates_are_within_sampling_error_of_the_truth(self) -> None:
        """On 4000 records IPS, SNIPS, and DR lie within four standard errors."""
        data = sample_log(4000, 5)
        for estimate in (
            inverse_propensity_score(data, target),
            self_normalized_ips(data, target),
            doubly_robust(data, target, lambda x, a: WRONG_MODEL[x][a]),  # type: ignore[index]
            doubly_robust(data, target, lambda x, a: MEAN_REWARD[x][a]),  # type: ignore[index]
        ):
            with self.subTest(estimator=estimate.estimator):
                self.assertLess(
                    abs(estimate.value - TRUE_VALUE), 4.0 * estimate.standard_error
                )
                self.assertLess(estimate.standard_error, 0.05)
        good = doubly_robust(data, target, lambda x, a: MEAN_REWARD[x][a])  # type: ignore[index]
        self.assertLess(
            good.standard_error, inverse_propensity_score(data, target).standard_error
        )

    def test_validation(self) -> None:
        """Malformed logs, policies, models, and clips raise."""
        with self.assertRaisesRegex(ValueError, "at least two records"):
            logged_bandit_data([(0, 0, 1.0, 0.5)], n_actions=2)
        with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
            logged_bandit_data([(0, 0, 1.0, 0.0), (0, 1, 1.0, 0.5)], n_actions=2)
        with self.assertRaisesRegex(ValueError, "action index"):
            logged_bandit_data([(0, 2, 1.0, 0.5), (0, 1, 1.0, 0.5)], n_actions=2)
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            LoggedBanditData((0, 1), (0, 1), (1.0,), (0.5, 0.5), 2)
        with self.assertRaisesRegex(ValueError, "must be"):
            logged_bandit_data([(0, 0, 1.0), (0, 1, 1.0)], n_actions=2)  # type: ignore[list-item]
        with self.assertRaisesRegex(ValueError, "n_actions"):
            logged_bandit_data([(0, 0, 1.0, 0.5), (0, 0, 1.0, 0.5)], n_actions=1)
        data = logged_bandit_data([(0, 0, 1.0, 0.5), (0, 1, 0.0, 0.5)], n_actions=2)
        with self.assertRaisesRegex(ValueError, "summing to one"):
            inverse_propensity_score(data, lambda x: (0.5, 0.6))
        with self.assertRaisesRegex(ValueError, "must return 2"):
            inverse_propensity_score(data, lambda x: (1.0,))
        with self.assertRaisesRegex(ValueError, "clip"):
            doubly_robust(data, lambda x: (0.5, 0.5), lambda x, a: 0.0, clip=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            direct_method(data, lambda x: (0.5, 0.5), lambda x, a: math.nan)
        with self.assertRaises(TypeError):
            inverse_propensity_score([(0, 0, 1.0, 0.5)], lambda x: (0.5, 0.5))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            direct_method(data, lambda x: (0.5, 0.5), None)  # type: ignore[arg-type]
        never = logged_bandit_data([(0, 0, 1.0, 0.5), (0, 0, 0.0, 0.5)], n_actions=2)
        with self.assertRaisesRegex(ValueError, "weight is zero"):
            self_normalized_ips(never, lambda x: (0.0, 1.0))
        zero = inverse_propensity_score(never, lambda x: (0.0, 1.0))
        self.assertEqual((zero.value, zero.effective_sample_size), (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "z must be positive"):
            zero.interval(0.0)


class ContextualBanditTests(unittest.TestCase):
    """LinUCB and linear Thompson sampling against closed forms and the library."""

    def test_linucb_scalar_case(self) -> None:
        """With d = 1, x = 1: theta = sum r / (n + 1), bonus = alpha / sqrt(n + 1)."""
        policy = LinUCB(2, 1, alpha=0.7)
        rewards = (1.0, 0.0, 2.5, -1.0, 4.0)
        for r in rewards:
            policy.update(0, (1.0,), r)
        self.assertAlmostEqual(policy.theta(0)[0], sum(rewards) / 6.0, places=14)
        self.assertAlmostEqual(policy.bonus(0, (1.0,)), 0.7 / math.sqrt(6.0), places=14)
        self.assertEqual(policy.theta(1), (0.0,))
        self.assertEqual(policy.bonus(1, (1.0,)), 0.7)
        scores = policy.scores(((1.0,), (1.0,)))
        self.assertAlmostEqual(scores[0], sum(rewards) / 6.0 + 0.7 / math.sqrt(6.0))
        self.assertEqual(policy.select(((1.0,), (1.0,))), 0)
        self.assertEqual(policy.pulls, (5, 0))

    def test_sherman_morrison_matches_direct_inversion(self) -> None:
        """The maintained inverse equals the Cholesky inverse of A after 300 updates."""
        rng = random.Random(12)
        policy = LinUCB(3, 4, alpha=1.0)
        for _ in range(300):
            x = tuple(rng.gauss(0.0, 1.0) for _ in range(4))
            policy.update(rng.randrange(3), x, rng.gauss(0.0, 1.0))
        worst = 0.0
        for arm in range(3):
            direct = inverse_spd(policy.design(arm))
            kept = policy.inverse_design(arm)
            worst = max(
                worst,
                max(abs(direct[i][j] - kept[i][j]) for i in range(4) for j in range(4)),
            )
            theta = solve_spd(policy.design(arm), policy.response(arm))
            for a, b in zip(theta, policy.theta(arm), strict=True):
                self.assertAlmostEqual(a, b, places=11)
        self.assertLess(worst, 1e-13)

    def test_untouched_arm_keeps_the_identity(self) -> None:
        """Only the played arm's model changes."""
        policy = LinUCB(2, 2)
        policy.update(1, (1.0, 2.0), 3.0)
        self.assertEqual(policy.design(0), identity(2))
        self.assertEqual(policy.design(1), ((2.0, 2.0), (2.0, 5.0)))
        self.assertEqual(policy.response(1), (3.0, 6.0))

    def test_linear_thompson_posterior_matches_conjugate_update(self) -> None:
        """With v = 1 the mean and covariance equal the N(0, I)-prior posterior."""
        rng = random.Random(8)
        policy = LinearThompsonSampling(3, seed=1)
        design = []
        observed = []
        for _ in range(40):
            x = tuple(rng.gauss(0.0, 1.0) for _ in range(3))
            r = rng.gauss(0.0, 1.0)
            policy.update(0, x, r)
            design.append(x)
            observed.append(r)
        prior = MultivariateNormal((0.0, 0.0, 0.0), identity(3))
        exact = condition_linear_gaussian(
            prior, design, observed, observation_variance=1.0
        )
        covariance = policy.posterior_covariance()
        for i in range(3):
            self.assertAlmostEqual(policy.posterior_mean()[i], exact.mean[i], places=12)
            for j in range(3):
                self.assertAlmostEqual(
                    covariance[i][j], exact.covariance[i][j], places=12
                )
        self.assertEqual(policy.updates, 40)
        wide = LinearThompsonSampling(3, v=2.0)
        self.assertEqual(wide.posterior_covariance()[0][0], 4.0)

    def test_parameter_draws_have_the_stated_moments(self) -> None:
        """6000 draws reproduce B^-1 f and v^2 B^-1 within Monte Carlo error."""
        policy = LinearThompsonSampling(2, v=1.5, seed=3)
        for x, r in (((1.0, 0.5), 2.0), ((0.2, -1.0), -1.0), ((1.0, 1.0), 0.5)):
            policy.update(0, x, r)
        draws = [policy.sample_parameter() for _ in range(6000)]
        mean = policy.posterior_mean()
        covariance = policy.posterior_covariance()
        for i in range(2):
            column = [d[i] for d in draws]
            se = math.sqrt(covariance[i][i] / 6000)
            self.assertLess(abs(statistics.fmean(column) - mean[i]), 4.0 * se)
            self.assertAlmostEqual(
                statistics.variance(column),
                covariance[i][i],
                delta=0.08 * covariance[i][i],
            )
        cross = statistics.covariance([d[0] for d in draws], [d[1] for d in draws])
        scale = math.sqrt(covariance[0][0] * covariance[1][1])
        self.assertAlmostEqual(cross, covariance[0][1], delta=0.05 * scale)

    def test_regret_is_sublinear_and_beats_uniform_play(self) -> None:
        """Regret per round falls with the horizon and is far below uniform play."""
        per_arm = ((1.0, 0.0, 0.5), (0.0, 1.0, -0.5), (0.5, 0.5, 0.5), (-1.0, 0.2, 0.0))
        shared = (0.8, -0.6, 0.3)
        ucb = simulate_linear_bandit(
            LinUCB(4, 3, alpha=1.0), per_arm, 2000, noise_sd=0.5, seed=21
        )
        uniform = simulate_linear_bandit(
            "uniform", per_arm, 2000, noise_sd=0.5, seed=21
        )
        self.assertLess(ucb.regret_per_round(2000), 0.5 * ucb.regret_per_round(200))
        self.assertLess(ucb.final_regret, 0.2 * uniform.final_regret)
        thompson = simulate_linear_bandit(
            LinearThompsonSampling(3, v=0.5, seed=4),
            shared,
            2000,
            n_arms=5,
            noise_sd=0.5,
            seed=22,
        )
        baseline = simulate_linear_bandit(
            "uniform", shared, 2000, n_arms=5, noise_sd=0.5, seed=22
        )
        self.assertLess(
            thompson.regret_per_round(2000), 0.5 * thompson.regret_per_round(200)
        )
        self.assertLess(thompson.final_regret, 0.2 * baseline.final_regret)
        self.assertEqual(sum(ucb.pulls), 2000)
        self.assertEqual(ucb.horizon, 2000)
        self.assertAlmostEqual(ucb.total_reward, sum(ucb.rewards), places=9)
        self.assertEqual((ucb.policy, thompson.policy), ("linucb", "linear_thompson"))
        # Uniform play has linear regret: equal halves within sampling error.
        first = uniform.cumulative_regret[999]
        second = uniform.final_regret - first
        self.assertAlmostEqual(first / second, 1.0, delta=0.15)

    def test_runs_are_reproducible_and_custom_contexts_work(self) -> None:
        """Equal seeds give equal runs; a fixed-context sampler finds the best arm."""
        runs = [
            simulate_linear_bandit(
                LinearThompsonSampling(2, seed=5), (1.0, -1.0), 50, n_arms=3, seed=6
            )
            for _ in range(2)
        ]
        self.assertEqual(runs[0], runs[1])
        other = simulate_linear_bandit(
            LinearThompsonSampling(2, seed=5), (1.0, -1.0), 50, n_arms=3, seed=7
        )
        self.assertNotEqual(runs[0].rewards, other.rewards)
        fixed = ((1.0, 0.0), (0.0, 1.0))
        run = simulate_linear_bandit(
            LinUCB(2, 2, alpha=0.5),
            ((0.2, 0.0), (0.0, 0.9)),
            300,
            noise_sd=0.0,
            context_sampler=lambda rng: fixed,
            seed=1,
        )
        self.assertGreater(run.pulls[1], 280)
        self.assertEqual(run.cumulative_regret[-1], run.cumulative_regret[-50])

    def test_validation(self) -> None:
        """Wrong shapes, parameters, and policies raise."""
        with self.assertRaisesRegex(ValueError, "at least two arms"):
            LinUCB(1, 2)
        with self.assertRaisesRegex(ValueError, "alpha"):
            LinUCB(2, 2, alpha=-1.0)
        with self.assertRaisesRegex(ValueError, "dimension"):
            LinUCB(2, 0)
        with self.assertRaisesRegex(ValueError, "v must be positive"):
            LinearThompsonSampling(2, v=0.0)
        policy = LinUCB(2, 2)
        with self.assertRaisesRegex(ValueError, "dimension 2"):
            policy.update(0, (1.0,), 1.0)
        with self.assertRaisesRegex(ValueError, "arm must be below"):
            policy.update(2, (1.0, 0.0), 1.0)
        with self.assertRaisesRegex(ValueError, "one feature vector per arm"):
            policy.select(((1.0, 0.0),))
        with self.assertRaisesRegex(ValueError, "finite"):
            policy.update(0, (1.0, 0.0), math.inf)
        with self.assertRaisesRegex(ValueError, "at least two arms"):
            LinearThompsonSampling(2).select(((1.0, 0.0),))
        with self.assertRaisesRegex(ValueError, "n_arms is required"):
            simulate_linear_bandit("uniform", (1.0, 0.0), 10)
        with self.assertRaisesRegex(ValueError, "does not match"):
            simulate_linear_bandit(LinUCB(3, 2), ((1.0, 0.0), (0.0, 1.0)), 10)
        with self.assertRaisesRegex(ValueError, "does not match"):
            simulate_linear_bandit(LinearThompsonSampling(3), (1.0, 0.0), 10, n_arms=2)
        with self.assertRaisesRegex(ValueError, "policy must be"):
            simulate_linear_bandit("greedy", (1.0, 0.0), 10, n_arms=2)
        with self.assertRaisesRegex(ValueError, "horizon"):
            simulate_linear_bandit("uniform", (1.0, 0.0), 0, n_arms=2)
        with self.assertRaisesRegex(ValueError, "noise_sd"):
            simulate_linear_bandit("uniform", (1.0, 0.0), 5, n_arms=2, noise_sd=-1.0)
        run = simulate_linear_bandit("uniform", (1.0, 0.0), 5, n_arms=2, seed=1)
        with self.assertRaisesRegex(ValueError, "rounds"):
            run.regret_per_round(6)


class SequentialHalvingTests(unittest.TestCase):
    """Budget accounting, deterministic arms, and Monte Carlo error rates."""

    def test_deterministic_arms_return_the_argmax(self) -> None:
        """Noise-free arms are always identified, for any arm count."""
        rng = random.Random(1)
        for n in (2, 3, 5, 8, 13, 16, 17):
            means = [rng.random() for _ in range(n)]
            rounds = math.ceil(math.log2(n))
            for budget in (n * rounds, 10 * n * rounds + 3):
                result = sequential_halving(lambda arm, _, m=means: m[arm], n, budget)
                with self.subTest(n=n, budget=budget):
                    self.assertEqual(result.best_arm, means.index(max(means)))
                    self.assertEqual(len(result.rounds), rounds)
                    self.assertLessEqual(result.pulls_used, budget)
                    self.assertEqual(result.pulls_used, sum(result.pulls))
                    self.assertEqual(
                        result.pulls_used,
                        sum(len(r.arms) * r.pulls_per_arm for r in result.rounds),
                    )

    def test_schedule_matches_the_formula(self) -> None:
        """Eight arms and T = 240: rounds of 8, 4, 2 arms with 10, 20, 40 pulls."""
        result = sequential_halving(lambda arm, _: float(arm), 8, 240)
        self.assertEqual(
            [(len(r.arms), r.pulls_per_arm) for r in result.rounds],
            [(8, 10), (4, 20), (2, 40)],
        )
        self.assertEqual(result.rounds[0].survivors, (7, 6, 5, 4))
        self.assertEqual(result.rounds[0].means, tuple(float(a) for a in range(8)))
        self.assertEqual(result.best_arm, 7)
        self.assertEqual(result.pulls_used, 240)
        self.assertEqual(result.pulls, (10, 10, 10, 10, 30, 30, 70, 70))
        # Five arms keep ceil(5 / 2) = 3, then 2, then 1.
        odd = sequential_halving(lambda arm, _: -float(arm), 5, 100)
        self.assertEqual([len(r.arms) for r in odd.rounds], [5, 3, 2])
        self.assertEqual([r.pulls_per_arm for r in odd.rounds], [6, 11, 16])
        self.assertEqual(odd.best_arm, 0)
        self.assertEqual(odd.pulls_used, 95)

    def test_misidentification_rate_decreases_with_budget(self) -> None:
        """On Bernoulli arms the error rate over 300 seeded runs falls with T."""
        means = (0.6, 0.45, 0.4, 0.4, 0.35, 0.3, 0.3, 0.2)

        def pull(arm: int, rng: random.Random) -> float:
            """Draw a Bernoulli reward."""
            return 1.0 if rng.random() < means[arm] else 0.0

        rates = []
        for budget in (48, 240, 1200):
            errors = sum(
                sequential_halving(pull, 8, budget, seed=1000 + rep).best_arm != 0
                for rep in range(300)
            )
            rates.append(errors / 300)
        self.assertGreater(rates[0], rates[1] + 0.1)
        self.assertGreater(rates[1], rates[2] + 0.1)
        self.assertLess(rates[2], 0.1)
        self.assertEqual(
            sequential_halving(pull, 8, 240, seed=3),
            sequential_halving(pull, 8, 240, seed=3),
        )

    def test_validation(self) -> None:
        """Too few arms, too small a budget, and bad rewards raise."""
        with self.assertRaisesRegex(ValueError, "n_arms"):
            sequential_halving(lambda arm, rng: 0.0, 1, 10)
        with self.assertRaisesRegex(ValueError, "budget must be at least"):
            sequential_halving(lambda arm, rng: 0.0, 8, 23)
        with self.assertRaisesRegex(ValueError, "finite"):
            sequential_halving(lambda arm, rng: math.nan, 2, 4)
        with self.assertRaises(TypeError):
            sequential_halving(None, 2, 4)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
