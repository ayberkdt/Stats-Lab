"""Search among candidate methods under a fixed evaluation budget.

Deriving a method usually produces a *family* of candidates: level laws of a
randomized estimator, pilot sizes, subsets of control variates, tuning
constants. This module selects among a finite, human-declared set of
candidates by measurement:

- :func:`rank_by_efficiency` replicates every candidate estimator and ranks
  them by work-normalized variance with a standard error for each;
- :func:`sequential_halving` spends a fixed budget of evaluations on a set
  of candidates, halving the set each round, and returns the survivor.

Both are *selection* procedures. The winner of a search is selected because
it looked best on the search data, so its measured performance is
optimistically biased (the winner's curse); certify the selected method on
fresh seeds with :mod:`probstat.methodlab.certify` before reporting its
properties.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer

from .estimators import Estimator, ReplicateSummary, summarize

Pull = Callable[[int, RandomStream], float]


@dataclass(frozen=True, slots=True)
class EfficiencyRanking:
    """Candidate estimators ordered by work-normalized variance.

    Args:
        summaries: Replicate summaries, most efficient first.
        log_standard_errors: Standard error of the logarithm of each
            work-normalized variance, in the same order.

    References:
        [REF-GLASSERMAN-2003, Section 1.1.3], [EQ-MLAB-WNV-001], [EQ-MLAB-MCSE-002].

    """

    summaries: tuple[ReplicateSummary, ...]
    log_standard_errors: tuple[float, ...]

    @property
    def best(self) -> str:
        """Return the name of the candidate with the smallest product.

        References:
            [REF-GLASSERMAN-2003, Section 1.1.3], [EQ-MLAB-WNV-001].

        """
        return self.summaries[0].name

    def separated(self, z: float = 2.576) -> bool:
        """Return whether the best candidate is separated from the runner-up.

        The log work-normalized variances of the first two candidates are
        compared with a two-sample z statistic. The default ``z`` is the
        two-sided 99% normal quantile; it is a per-pair statement that
        ignores the selection among more than two candidates.

        Args:
            z: Critical value.

        Returns:
            ``True`` when the difference of the logs exceeds ``z`` times its
            standard error; ``True`` for a single candidate.

        References:
            [REF-GLASSERMAN-2003, Section 1.1.3], [EQ-MLAB-WNV-001], [EQ-MLAB-MCSE-002].

        """
        if len(self.summaries) < 2:
            return True
        first, second = self.summaries[0], self.summaries[1]
        gap = math.log(second.work_normalized_variance) - math.log(
            first.work_normalized_variance
        )
        spread = math.hypot(self.log_standard_errors[0], self.log_standard_errors[1])
        return gap > z * spread


def rank_by_efficiency(
    candidates: Sequence[Estimator],
    *,
    replicates: int,
    seed: int | RandomStream,
) -> EfficiencyRanking:
    """Rank candidate estimators by work-normalized variance.

    Every candidate is replicated on the same replicate streams (common
    random numbers) and summarized; the ranking is by
    ``Var x E[cost]`` ([EQ-MLAB-WNV-001]), smallest first. The comparison is
    meaningful only among estimators of the same quantity that are (nearly)
    unbiased; a biased candidate with a small variance would win wrongly.

    Args:
        candidates: At least one estimator with distinct names.
        replicates: Replicates per candidate (at least four).
        seed: Root seed or stream.

    Returns:
        Summaries ordered by efficiency with the standard error of each log
        work-normalized variance.

    Raises:
        ValueError: If there are no candidates, names repeat, or a
            candidate has zero variance or zero cost.

    References:
        [REF-GLASSERMAN-2003, Section 1.1.3], [EQ-MLAB-WNV-001], [EQ-MLAB-MCSE-002].

    """
    members = tuple(candidates)
    if not members:
        raise ValueError("at least one candidate is required")
    if len({m.name for m in members}) != len(members):
        raise ValueError("candidate names must be distinct")
    count = require_non_negative_integer(replicates, name="replicates")
    if count < 4:
        raise ValueError("replicates must be at least four")
    root = resolve_stream(seed)
    rows: list[tuple[ReplicateSummary, float]] = []
    for member in members:
        summary = summarize(member, count, root)
        if summary.variance <= 0.0 or summary.mean_cost <= 0.0:
            raise ValueError(f"candidate {member.name!r} has zero variance or cost")
        rows.append((summary, summary.variance_standard_error / summary.variance))
    rows.sort(key=lambda row: row[0].work_normalized_variance)
    return EfficiencyRanking(
        tuple(row[0] for row in rows), tuple(row[1] for row in rows)
    )


@dataclass(frozen=True, slots=True)
class HalvingRound:
    """One round of sequential halving.

    Args:
        survivors: Indices of the candidates evaluated in this round.
        pulls: Evaluations given to each survivor in this round.
        means: Mean loss of each survivor in this round, in the order of
            ``survivors``.

    References:
        [REF-KARNIN-KOREN-SOMEKH-2013].

    """

    survivors: tuple[int, ...]
    pulls: int
    means: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HalvingResult:
    """Outcome of sequential halving.

    Args:
        best: Index of the surviving candidate.
        rounds: The rounds in order.
        total_pulls: Evaluations actually used (at most the budget).
        budget: The evaluation budget given.

    References:
        [REF-KARNIN-KOREN-SOMEKH-2013], [EQ-MLAB-SH-001].

    """

    best: int
    rounds: tuple[HalvingRound, ...]
    total_pulls: int
    budget: int


def sequential_halving(
    candidates: int,
    pull: Pull,
    *,
    budget: int,
    seed: int | RandomStream,
    minimize: bool = True,
) -> HalvingResult:
    """Select the best of several candidates with a fixed evaluation budget.

    With ``n`` candidates and budget ``T`` there are ``ceil(log2 n)``
    rounds. In round ``r`` each of the ``|S_r|`` surviving candidates is
    evaluated ``floor(T / (|S_r| ceil(log2 n)))`` times and the better half
    (rounded up) survives ([EQ-MLAB-SH-001]). Evaluation ``j`` of round
    ``r`` uses the same child stream ``("round", r, j)`` for every
    candidate, so candidates are compared on common random numbers. Only
    the evaluations of the current round enter its means, as in the source.

    The probability of returning a suboptimal candidate decays
    exponentially in ``T / (H_2 log2 n)`` for bounded losses, where ``H_2``
    measures how close the candidates are (see the source for the bound and
    its constants). Nothing is guaranteed about the *estimated performance*
    of the winner, which is selected and therefore optimistic.

    Args:
        candidates: Number of candidates ``n >= 1``; they are addressed by
            index ``0, ..., n - 1``.
        pull: ``pull(index, stream)`` returns one noisy loss (or reward) of
            a candidate.
        budget: Total number of evaluations ``T``; it must allow at least
            one evaluation per candidate in the first round.
        seed: Root seed or stream.
        minimize: Treat the returned values as losses (default) instead of
            rewards.

    Returns:
        The surviving index and the record of every round.

    Raises:
        ValueError: If there is no candidate or the budget is too small.

    References:
        [REF-KARNIN-KOREN-SOMEKH-2013],
        [EQ-MLAB-SH-001].

    """
    count = require_non_negative_integer(candidates, name="candidates")
    total = require_non_negative_integer(budget, name="budget")
    if count == 0:
        raise ValueError("at least one candidate is required")
    if count == 1:
        return HalvingResult(0, (), 0, total)
    depth = math.ceil(math.log2(count))
    if total < count * depth:
        raise ValueError(
            f"budget must be at least {count * depth} "
            "(one evaluation per candidate per round)"
        )
    root = resolve_stream(seed)
    survivors = list(range(count))
    rounds: list[HalvingRound] = []
    used = 0
    for number in range(depth):
        pulls = total // (len(survivors) * depth)
        means: list[float] = []
        for index in survivors:
            values = [
                float(pull(index, root.child("round", number, j))) for j in range(pulls)
            ]
            means.append(math.fsum(values) / pulls)
        used += pulls * len(survivors)
        rounds.append(HalvingRound(tuple(survivors), pulls, tuple(means)))
        order = sorted(
            range(len(survivors)),
            key=lambda i: means[i] if minimize else -means[i],
        )
        keep = math.ceil(len(survivors) / 2)
        survivors = [survivors[i] for i in sorted(order[:keep])]
    return HalvingResult(survivors[0], tuple(rounds), used, total)
