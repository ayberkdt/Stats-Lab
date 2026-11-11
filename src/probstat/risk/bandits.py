"""Sequential decisions under uncertainty: the Beta-Bernoulli bandit.

An agent chooses one of ``K`` actions per period; action ``k`` succeeds
(reward ``1``) with unknown probability ``theta_k``. Each ``theta_k`` gets an
independent ``Beta(alpha_k, beta_k)`` prior, updated by conjugacy after each
observed reward: ``(alpha_k, beta_k) += (r_t, 1 - r_t)`` for the chosen
action only. Thompson sampling draws one ``theta_hat_k`` from every posterior
and plays the largest, so each action is chosen with its posterior
probability of being optimal; the greedy rule plays the largest posterior
mean and can lock onto a suboptimal action forever. When the success
probabilities are known (a simulation study), the run also reports the
cumulative pseudo-regret ``sum_t (max_k theta_k - theta_{x_t})``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

BanditPolicy = Literal["thompson", "greedy"]
RewardFunction = Callable[[int, random.Random], int]


@dataclass(frozen=True, slots=True)
class BanditRun:
    """Actions, rewards, and Beta posteriors of one Bernoulli-bandit run.

    Args:
        policy: ``"thompson"`` or ``"greedy"``.
        actions: Chosen action ``x_t`` per period (0-based).
        rewards: Observed reward ``r_t`` in ``{0, 1}`` per period.
        posterior_alpha: Final ``alpha_k`` per action.
        posterior_beta: Final ``beta_k`` per action.
        pulls: Number of periods each action was played.
        cumulative_regret: Running pseudo-regret, or ``None`` when the
            success probabilities are unknown.

    References:
        [EQ-RISK-BANDIT-001], [REF-RUSSO-ETAL-2018, Chapter 3].

    """

    policy: str
    actions: tuple[int, ...]
    rewards: tuple[int, ...]
    posterior_alpha: tuple[float, ...]
    posterior_beta: tuple[float, ...]
    pulls: tuple[int, ...]
    cumulative_regret: tuple[float, ...] | None

    @property
    def total_reward(self) -> int:
        """Return the number of successes over the run.

        References:
            [REF-RUSSO-ETAL-2018, Example 3.1].

        """
        return sum(self.rewards)

    @property
    def posterior_means(self) -> tuple[float, ...]:
        """Return ``alpha_k / (alpha_k + beta_k)`` per action.

        References:
            [EQ-RISK-BANDIT-001], [REF-RUSSO-ETAL-2018, Chapter 3].

        """
        return tuple(
            a / (a + b)
            for a, b in zip(self.posterior_alpha, self.posterior_beta, strict=True)
        )

    def probability_optimal(
        self, draws: int = 4000, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Estimate the posterior probability that each action is best.

        This is the probability with which Thompson sampling would choose
        each action in the next period.

        Args:
            draws: Positive number of joint posterior draws.
            seed: Integer seed or stream; ``None`` is not reproducible.

        Returns:
            Monte Carlo estimates summing to one.

        Raises:
            ValueError: If ``draws`` is not positive.

        Equation:
            [EQ-RISK-BANDIT-002].

        References:
            [REF-RUSSO-ETAL-2018, Chapter 3].

        """
        count = require_non_negative_integer(draws, name="draws")
        if count == 0:
            raise ValueError("draws must be positive")
        rng = resolve_stream(seed).child("probability-optimal").generator()
        wins = [0] * len(self.posterior_alpha)
        for _ in range(count):
            wins[
                _argmax(
                    [
                        rng.betavariate(a, b)
                        for a, b in zip(
                            self.posterior_alpha, self.posterior_beta, strict=True
                        )
                    ]
                )
            ] += 1
        return tuple(win / count for win in wins)


def _argmax(values: Sequence[float]) -> int:
    """Return the first index of the largest value.

    References:
        [REF-RUSSO-ETAL-2018, Algorithm 3.1]. No equation is evaluated.

    """
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def _priors(
    prior: tuple[float, float] | Sequence[tuple[float, float]], arms: int
) -> tuple[list[float], list[float]]:
    """Validate one shared or per-action ``Beta(alpha, beta)`` prior.

    References:
        [REF-RUSSO-ETAL-2018, Example 3.1]. No equation is evaluated.

    """
    pairs: Sequence[object]
    if len(prior) == 2 and all(
        not isinstance(item, (tuple, list)) for item in prior
    ):
        pairs = [prior] * arms
    else:
        pairs = prior
    if len(pairs) != arms:
        raise ValueError("prior must be one (alpha, beta) pair or one per action")
    alphas: list[float] = []
    betas: list[float] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("each prior must be an (alpha, beta) pair")
        a = require_finite_real(pair[0], name=f"prior[{index}][0]")
        b = require_finite_real(pair[1], name=f"prior[{index}][1]")
        if a <= 0.0 or b <= 0.0:
            raise ValueError("Beta prior parameters must be positive")
        alphas.append(a)
        betas.append(b)
    return alphas, betas


def bernoulli_bandit(
    environment: Sequence[float] | RewardFunction,
    horizon: int,
    *,
    policy: BanditPolicy = "thompson",
    n_actions: int | None = None,
    prior: tuple[float, float] | Sequence[tuple[float, float]] = (1.0, 1.0),
    seed: int | RandomStream | None = None,
) -> BanditRun:
    """Run Thompson sampling (or the greedy rule) on a Bernoulli bandit.

    Args:
        environment: Success probabilities ``theta_k`` (simulation study,
            enables regret), or a callable ``reward(action, rng) -> 0 or 1``
            for an external system.
        horizon: Positive number of periods.
        policy: ``"thompson"`` (posterior sampling) or ``"greedy"``
            (posterior mean).
        n_actions: Number of actions; required with a callable environment
            and otherwise inferred.
        prior: One ``(alpha, beta)`` pair for every action or one per action.
        seed: Integer seed or stream; ``None`` is not reproducible. The
            policy and the environment use separate child streams.

    Returns:
        Actions, rewards, final posteriors, pull counts, and regret.

    Raises:
        TypeError: If an argument has the wrong type or a reward is not
            ``0`` or ``1``.
        ValueError: If the horizon, probabilities, priors, policy, or action
            count are invalid.

    Equation:
        [EQ-RISK-BANDIT-001], [EQ-RISK-BANDIT-002], [EQ-RISK-BANDIT-003].

    References:
        [REF-RUSSO-ETAL-2018, Algorithms 3.1 and 3.2],
        [REF-THOMPSON-1933].

    """
    periods = require_non_negative_integer(horizon, name="horizon")
    if periods == 0:
        raise ValueError("horizon must be positive")
    if policy not in ("thompson", "greedy"):
        raise ValueError("policy must be 'thompson' or 'greedy'")
    probabilities: tuple[float, ...] | None
    if callable(environment):
        if n_actions is None:
            raise ValueError("n_actions is required with a callable environment")
        arms = require_non_negative_integer(n_actions, name="n_actions")
        probabilities = None
        reward_function: RewardFunction = environment
    else:
        probabilities = tuple(
            require_probability(value, name=f"environment[{index}]")
            for index, value in enumerate(environment)
        )
        arms = len(probabilities)
        if n_actions is not None and n_actions != arms:
            raise ValueError("n_actions does not match the number of probabilities")
        known = probabilities

        def reward_function(action: int, rng: random.Random) -> int:
            """Draw a Bernoulli reward with the known success probability.

            References:
                [REF-RUSSO-ETAL-2018, Example 1.1].

            """
            return 1 if rng.random() < known[action] else 0

    if arms < 2:
        raise ValueError("a bandit needs at least two actions")
    alphas, betas = _priors(prior, arms)
    stream = resolve_stream(seed)
    policy_rng = stream.child("bandit", "policy").generator()
    environment_rng = stream.child("bandit", "environment").generator()
    best = max(probabilities) if probabilities is not None else 0.0
    actions: list[int] = []
    rewards: list[int] = []
    regret: list[float] = []
    running = 0.0
    for _ in range(periods):
        if policy == "thompson":
            scores = [
                policy_rng.betavariate(a, b)
                for a, b in zip(alphas, betas, strict=True)
            ]
        else:
            scores = [a / (a + b) for a, b in zip(alphas, betas, strict=True)]
        action = _argmax(scores)
        reward = reward_function(action, environment_rng)
        if isinstance(reward, bool) or reward not in (0, 1):
            raise TypeError("rewards must be the integers 0 or 1")
        alphas[action] += reward
        betas[action] += 1 - reward
        actions.append(action)
        rewards.append(reward)
        if probabilities is not None:
            running += best - probabilities[action]
            regret.append(running)
    return BanditRun(
        policy=policy,
        actions=tuple(actions),
        rewards=tuple(rewards),
        posterior_alpha=tuple(alphas),
        posterior_beta=tuple(betas),
        pulls=tuple(actions.count(k) for k in range(arms)),
        cumulative_regret=tuple(regret) if probabilities is not None else None,
    )
