"""Nested R-hat: a convergence diagnostic for many short Markov chains.

The classical potential scale reduction factor compares the variance between
chains with the variance within chains, and its between-chain term stays
noisy unless every chain is long. When hardware runs thousands of chains for
a few iterations each, Margossian, Hoffman, Sountsov, Riou-Durand, Vehtari
and Gelman propose to group the chains into ``K`` *superchains* of ``M``
chains that share one starting point (but use independent random numbers),
and to compare the variance between superchain means with the variance
within superchains ([EQ-INF-NRHAT-001]):

    ``nR-hat = sqrt(1 + B / W)``.

``B`` is the sample variance of the ``K`` superchain means. ``W`` is the
average over superchains of ``B~_k + W~_k``: the sample variance of the ``M``
chain means inside superchain ``k`` plus the average within-chain sample
variance of its chains. ``B~_k`` is defined as zero when ``M = 1`` and
``W~_k`` as zero when each chain has a single draw, so that the statistic is
usable with one draw per chain.

Reading the value
    Chains of one superchain start at the same point, so before the chains
    forget their start the superchain means differ by more than the spread
    inside a superchain explains and ``nR-hat`` is well above one. For
    stationary independent chains ``B / W`` is about ``1 / (M (N + 1))``
    for ``N`` draws per chain, so the statistic approaches one as ``M``
    grows, not only as ``N`` grows. That is the point of the construction:
    a threshold close to one can be met with short chains. The value depends
    on ``M``; a threshold must be chosen with ``M`` in mind and no universal
    cut-off is built in here.

With ``M = 1`` the statistic is ``sqrt(1 + B / W)`` with the usual
between-chain and within-chain variances; the classical (non-split)
``R-hat^2`` is ``(N - 1) / N + B / W``, so the squares differ by exactly
``1 / N``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real


@dataclass(frozen=True, slots=True)
class NestedRhat:
    """Nested R-hat with its variance components.

    Args:
        value: ``sqrt(1 + B / W)``; ``nan`` when ``B = W = 0`` and ``inf``
            when only ``W = 0``.
        between_superchain_variance: ``B``, the sample variance of the
            superchain means.
        within_superchain_variance: ``W``, the average of the
            between-chain-within-superchain and within-chain variances.
        superchains: Number of superchains ``K``.
        chains_per_superchain: Number of chains ``M`` in each superchain.
        draws_per_chain: Number of draws ``N`` in each chain.

    References:
        [EQ-INF-NRHAT-001], [REF-MARGOSSIAN-ETAL-2024].

    """

    value: float
    between_superchain_variance: float
    within_superchain_variance: float
    superchains: int
    chains_per_superchain: int
    draws_per_chain: int


def _sample_variance(values: Sequence[float]) -> float:
    """Return the sample variance with divisor ``n - 1`` (zero for ``n = 1``).

    References:
        [EQ-INF-NRHAT-001], [REF-MARGOSSIAN-ETAL-2024].

    """
    count = len(values)
    if count < 2:
        return 0.0
    center = math.fsum(values) / count
    return math.fsum((value - center) ** 2 for value in values) / (count - 1)


def nested_rhat(superchains: Sequence[Sequence[Sequence[float]]]) -> NestedRhat:
    """Return the nested R-hat of scalar draws grouped into superchains.

    With draws ``f_(nmk)`` (draw ``n`` of chain ``m`` of superchain ``k``),
    chain means ``f_(.mk)``, superchain means ``f_(..k)``:

        ``B = Var_k(f_(..k))`` (divisor ``K - 1``),
        ``B~_k = Var_m(f_(.mk))`` (divisor ``M - 1``; zero if ``M = 1``),
        ``W~_k = (1/M) sum_m Var_n(f_(nmk))`` (divisor ``N - 1``; zero if
        ``N = 1``),
        ``W = (1/K) sum_k (B~_k + W~_k)``, ``nR-hat = sqrt(1 + B / W)``.

    The diagnostic presumes that the chains of one superchain were started at
    the same point and the superchains at different, overdispersed points;
    the function cannot check this. Pass post-warm-up draws of one scalar
    quantity.

    Args:
        superchains: ``superchains[k][m][n]``: ``K >= 2`` superchains, each
            with the same number ``M >= 1`` of chains of equal length
            ``N >= 1``; ``M`` and ``N`` must not both be one.

    Returns:
        The statistic with ``B`` and ``W``.

    Raises:
        ValueError: For fewer than two superchains, unequal group sizes or
            chain lengths, ``M = N = 1``, or a non-finite draw.

    Equation:
        [EQ-INF-NRHAT-001].

    References:
        [REF-MARGOSSIAN-ETAL-2024].

    """
    groups = tuple(
        tuple(
            tuple(require_finite_real(value, name="draw") for value in chain)
            for chain in group
        )
        for group in superchains
    )
    if len(groups) < 2:
        raise ValueError("at least 2 superchains are required")
    chains = len(groups[0])
    if chains == 0 or any(len(group) != chains for group in groups):
        raise ValueError("every superchain needs the same positive number of chains")
    draws = len(groups[0][0])
    if draws == 0 or any(len(chain) != draws for group in groups for chain in group):
        raise ValueError("every chain needs the same positive number of draws")
    if chains == 1 and draws == 1:
        raise ValueError(
            "one chain of one draw per superchain leaves no within-superchain variance"
        )
    superchain_means: list[float] = []
    within_terms: list[float] = []
    for group in groups:
        chain_means = [math.fsum(chain) / draws for chain in group]
        superchain_means.append(math.fsum(chain_means) / chains)
        between_chains = _sample_variance(chain_means)
        within_chains = math.fsum(_sample_variance(chain) for chain in group) / chains
        within_terms.append(between_chains + within_chains)
    between = _sample_variance(superchain_means)
    within = math.fsum(within_terms) / len(groups)
    if within == 0.0:
        value = math.nan if between == 0.0 else math.inf
    else:
        value = math.sqrt(1.0 + between / within)
    return NestedRhat(value, between, within, len(groups), chains, draws)
