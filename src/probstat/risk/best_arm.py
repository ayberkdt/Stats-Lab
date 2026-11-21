"""Fixed-budget best-arm identification by sequential halving.

Given ``n`` arms and a total budget of ``T`` pulls, the goal is to name the
arm with the largest mean reward at the end; rewards collected on the way do
not matter (pure exploration). Sequential halving splits the budget evenly
over ``ceil(log2 n)`` rounds. In round ``r`` each of the surviving arms
``S_r`` is pulled ``floor(T / (|S_r| ceil(log2 n)))`` times, and the
``ceil(|S_r| / 2)`` arms with the largest average reward of that round
survive. After the last round one arm is left.

Round ``r`` uses ``|S_r| floor(T / (|S_r| ceil(log2 n))) <= T / ceil(log2 n)``
pulls, so the total never exceeds ``T``. For rewards in ``[0, 1]`` the
probability of returning a suboptimal arm is exponentially small in
``T / (H_2 log2 n)``, where ``H_2`` is the complexity measure of the
source, which grows as the gaps to the best arm shrink; see the source for
its definition and the constants.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

PullFunction = Callable[[int, random.Random], float]


@dataclass(frozen=True, slots=True)
class HalvingRound:
    """Statistics of one sequential-halving round.

    Args:
        arms: Arms alive at the start of the round.
        pulls_per_arm: Pulls given to each of them in this round.
        means: Average reward of each arm in this round (same order as
            ``arms``).
        survivors: Arms kept for the next round, best first.

    References:
        [EQ-RISK-SH-001], [REF-KARNIN-KOREN-SOMEKH-2013].

    """

    arms: tuple[int, ...]
    pulls_per_arm: int
    means: tuple[float, ...]
    survivors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SequentialHalvingResult:
    """Arm returned by sequential halving with its per-round record.

    Args:
        best_arm: Index of the returned arm.
        rounds: One :class:`HalvingRound` per elimination round.
        pulls_used: Total number of pulls, at most ``budget``.
        budget: The budget ``T`` that was given.
        pulls: Total pulls per arm.

    References:
        [EQ-RISK-SH-001], [REF-KARNIN-KOREN-SOMEKH-2013].

    """

    best_arm: int
    rounds: tuple[HalvingRound, ...]
    pulls_used: int
    budget: int
    pulls: tuple[int, ...]


def sequential_halving(
    pull: PullFunction,
    n_arms: int,
    budget: int,
    *,
    seed: int | RandomStream | None = None,
) -> SequentialHalvingResult:
    """Identify the arm with the largest mean within a fixed pull budget.

    Averages are computed from the pulls of the current round only, as in
    the source. Ties are broken towards the lower arm index.

    Args:
        pull: ``pull(arm, rng)`` returning one finite reward of that arm.
        n_arms: Number of arms ``n`` (at least two).
        budget: Total budget ``T`` with ``T >= n ceil(log2 n)``, so that
            every arm is pulled at least once in the first round.
        seed: Integer seed or stream passed to ``pull``; ``None`` is not
            reproducible.

    Returns:
        The selected arm, the per-round statistics, and the pulls used.

    Raises:
        TypeError: If an argument has the wrong type or ``pull`` is not
            callable.
        ValueError: If ``n_arms < 2``, the budget is too small, or a reward
            is not finite.

    Equation:
        [EQ-RISK-SH-001].

    References:
        [REF-KARNIN-KOREN-SOMEKH-2013].

    """
    if not callable(pull):
        raise TypeError("pull must be callable")
    arms = require_non_negative_integer(n_arms, name="n_arms")
    if arms < 2:
        raise ValueError(f"n_arms must be at least 2, got {arms}")
    total = require_non_negative_integer(budget, name="budget")
    n_rounds = (arms - 1).bit_length()
    if total < arms * n_rounds:
        raise ValueError(
            f"budget must be at least n_arms * ceil(log2 n_arms) = "
            f"{arms * n_rounds}, got {total}"
        )
    rng = resolve_stream(seed).child("sequential-halving").generator()
    alive = tuple(range(arms))
    pulls = [0] * arms
    record: list[HalvingRound] = []
    for _ in range(n_rounds):
        per_arm = total // (len(alive) * n_rounds)
        means: list[float] = []
        for arm in alive:
            rewards = [
                require_finite_real(pull(arm, rng), name=f"pull({arm}, rng)")
                for _ in range(per_arm)
            ]
            means.append(math.fsum(rewards) / per_arm)
            pulls[arm] += per_arm
        order = sorted(range(len(alive)), key=lambda i: (-means[i], alive[i]))
        survivors = tuple(alive[i] for i in order[: (len(alive) + 1) // 2])
        record.append(
            HalvingRound(
                arms=alive,
                pulls_per_arm=per_arm,
                means=tuple(means),
                survivors=survivors,
            )
        )
        alive = survivors
    return SequentialHalvingResult(
        best_arm=alive[0],
        rounds=tuple(record),
        pulls_used=sum(pulls),
        budget=total,
        pulls=tuple(pulls),
    )
