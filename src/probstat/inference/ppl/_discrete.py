"""Discrete latent sites: finite supports, enumeration, and recovery.

A discrete latent site with a finite support is summed out of the joint
density by enumeration: ``log p(theta, y) = log sum_z exp(log p(theta, z, y))``
evaluated by a shifted log-sum-exp ([EQ-PPL-ENUM-001]). Sites outside any
plate are *global* and are enumerated jointly by re-running the model once
per configuration; sites inside an iteration of :meth:`ModelHandle.plate`
are *local* and are enumerated inside that iteration only, which is exact
when iterations are conditionally independent given the sites declared
before the plate. Given continuous draws, the discrete sites are recovered
from their exact conditional law ([EQ-PPL-ENUM-002]).

Plates nest (ADR-0084): the enumeration record of a plate iteration holds,
for each of its branches, the records of the iterations of the plates
nested in it, and recovery descends that tree. A vector-valued discrete
site enumerates its joint support: the Cartesian product of independent
element supports ([EQ-PPL-ENUM-003]) or the count vectors of a multinomial
law. Under dynamic site structure a global configuration may lack some
discrete sites; their recovered value is ``None`` in that draw.

Markov chains (ADR-0088): the discrete sites of the steps of
:meth:`ModelHandle.markov` are eliminated by the forward algorithm in log
space ([EQ-PPL-MARKOV-001]). The record of a chain (:class:`MarkovGroup`)
keeps the initial log-weights and the transition tables of every step;
recovery draws the path by forward filtering and backward sampling and
computes the exact smoothed marginals from the forward and backward
recursions ([EQ-PPL-MARKOV-002]). :func:`most_probable` returns the jointly
most probable configuration of every discrete site by max-product
elimination, the Viterbi recursion along chains ([EQ-PPL-MARKOV-003]).
Branches of a plate iteration may declare different continuous sites; the
record of the iteration keeps, per branch, the sites it lacks
([EQ-PPL-LOCAL-001]).
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from probstat.distributions import FiniteSupport, IntegerSupport
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.numerics import Dual, ReverseVariable, admath

from ._errors import ModelSpecificationError

ENUMERATION_LIMIT = 4096
"""Largest number of configurations enumerated jointly in one scope."""


def primal(value: Any) -> float:
    """Return the float value under nested duals and reverse-mode variables.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    while isinstance(value, Dual | ReverseVariable):
        value = value.value
    return float(value)


def log_sum_exp(values: Sequence[Any]) -> Any:
    """Return ``log sum_i exp(v_i)`` with a float shift (dual-transparent).

    ``m + log sum_i exp(v_i - m)`` with ``m = max_i v_i`` taken as a
    constant, which leaves the derivative unchanged; ``-inf`` terms are
    skipped and an all ``-inf`` input gives ``-inf``.

    References:
        [EQ-PPL-ENUM-001], [REF-STAN-USERS-GUIDE-2-36].

    """
    finite = [(v, primal(v)) for v in values]
    finite = [(v, p) for v, p in finite if p != -math.inf]
    if not finite:
        return -math.inf
    shift = max(p for _, p in finite)
    if shift == math.inf:
        return math.inf
    return shift + admath.log(admath.fsum(admath.exp(v - shift) for v, _ in finite))


def _limit_message(site: str, count: int) -> str:
    """Return the error message for a joint support that is too large.

    References:
        [EQ-PPL-ENUM-001]. No new equation is introduced.

    """
    return (
        f"discrete latent site {site!r} has {count} joint support values; at most "
        f"{ENUMERATION_LIMIT} can be enumerated"
    )


def product_support(
    element_values: Sequence[tuple[Any, ...]], site: str
) -> tuple[tuple[Any, ...], ...]:
    """Return the joint support of a vector of independent finite elements.

    The Cartesian product in lexicographic order (the last element varies
    fastest), the support of a vector of independent discrete elements
    ([EQ-PPL-ENUM-003]).

    Args:
        element_values: Support values of each element.
        site: Site name for error messages.

    Returns:
        Tuples of element values.

    Raises:
        ModelSpecificationError: For more than :data:`ENUMERATION_LIMIT`
            joint values.

    References:
        [EQ-PPL-ENUM-003], [REF-STAN-USERS-GUIDE-2-36].

    """
    count = math.prod(len(values) for values in element_values)
    if count > ENUMERATION_LIMIT:
        raise ModelSpecificationError(_limit_message(site, count))
    return tuple(itertools.product(*element_values))


def multinomial_support(
    trials: int, categories: int, site: str
) -> tuple[tuple[int, ...], ...]:
    """Return every count vector of ``trials`` trials in ``categories`` cells.

    The ``C(n + K - 1, K - 1)`` compositions of ``n`` into ``K``
    non-negative parts in lexicographic order: the support of
    ``Multinomial(n, p)`` ([EQ-DIST-MULT-001]).

    Args:
        trials: Number of trials ``n >= 0``.
        categories: Number of cells ``K >= 2``.
        site: Site name for error messages.

    Returns:
        Count vectors summing to ``trials``.

    Raises:
        ModelSpecificationError: For more than :data:`ENUMERATION_LIMIT`
            count vectors.

    References:
        [EQ-DIST-MULT-001], [REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35].

    """
    count = math.comb(trials + categories - 1, categories - 1)
    if count > ENUMERATION_LIMIT:
        raise ModelSpecificationError(_limit_message(site, count))
    vectors: list[tuple[int, ...]] = [()]
    for _ in range(categories - 1):
        grown: list[tuple[int, ...]] = []
        for prefix in vectors:
            remaining = trials - sum(prefix)
            grown.extend((*prefix, k) for k in range(remaining + 1))
        vectors = grown
    return tuple((*counts, trials - sum(counts)) for counts in vectors)


def finite_support_values(support: object, site: str) -> tuple[Any, ...] | None:
    """Return the enumerable values of a discrete support, or ``None``.

    Args:
        support: A distribution's ``support``.
        site: Site name for error messages.

    Returns:
        The support values in order for :class:`IntegerSupport` with both
        bounds and :class:`FiniteSupport`; ``None`` for any other support.

    Raises:
        ModelSpecificationError: For an integer support without an upper
            bound or with more than :data:`ENUMERATION_LIMIT` values.

    References:
        [REF-STAN-USERS-GUIDE-2-36]. No equation is evaluated.

    """
    if isinstance(support, IntegerSupport):
        if support.upper is None:
            raise ModelSpecificationError(
                f"discrete latent site {site!r} has the unbounded support "
                f"{{{support.lower}, {support.lower + 1}, ...}}, which cannot be "
                "enumerated; truncate it with ppl.TruncatedDiscrete(base, lower, "
                "upper) or marginalize it by hand"
            )
        count = support.upper - support.lower + 1
        values: tuple[Any, ...] = (
            () if count > ENUMERATION_LIMIT else support.finite_values()
        )
    elif isinstance(support, FiniteSupport):
        values = tuple(support.values)
        count = len(values)
    else:
        return None
    if count > ENUMERATION_LIMIT:
        raise ModelSpecificationError(
            f"discrete latent site {site!r} has {count} support values; at most "
            f"{ENUMERATION_LIMIT} can be enumerated"
        )
    return values


@dataclass(frozen=True, slots=True)
class TruncatedDiscrete:
    """A discrete law restricted to ``{lower, ..., upper}`` and renormalized.

    ``p_T(k) = p(k) / sum_(j=lower..upper) p(j)`` for ``lower <= k <= upper``.
    Use it to give an unbounded count law (Poisson, negative binomial) a
    finite support so that it can be a discrete latent site.

    Args:
        base: Discrete law with ``log_pmf``.
        lower: Inclusive integer lower bound.
        upper: Inclusive integer upper bound, ``upper >= lower``.

    Raises:
        TypeError: If a bound is not an integer.
        ValueError: If ``upper < lower``, the range is too long to enumerate,
            or it carries no probability.

    References:
        [REF-CASELLA-BERGER-2002, Section 1.3], [REF-STAN-USERS-GUIDE-2-36].

    """

    base: Any
    lower: int
    upper: int
    _log_masses: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the bounds and tabulate the renormalized log masses.

        References:
            [EQ-PPL-ENUM-001], [REF-CASELLA-BERGER-2002, Section 1.3].

        """
        support = IntegerSupport(self.lower, self.upper)
        assert support.upper is not None
        if support.upper - support.lower + 1 > ENUMERATION_LIMIT:
            raise ValueError(f"at most {ENUMERATION_LIMIT} values can be enumerated")
        logs = [float(self.base.log_pmf(k)) for k in support.finite_values()]
        total = log_sum_exp(logs)
        if total == -math.inf:
            raise ValueError("the truncation range carries zero probability")
        object.__setattr__(self, "_log_masses", tuple(v - total for v in logs))

    @property
    def support(self) -> IntegerSupport:
        """Return ``{lower, ..., upper}``.

        References:
            [REF-CASELLA-BERGER-2002, Section 1.3]. No equation is evaluated.

        """
        return IntegerSupport(self.lower, self.upper)

    def log_pmf(self, k: int) -> float:
        """Return ``log p_T(k)`` (``-inf`` outside the range).

        References:
            [REF-CASELLA-BERGER-2002, Section 1.3]. No new equation is introduced.

        """
        if isinstance(k, bool) or not isinstance(k, int):
            return -math.inf
        if not self.lower <= k <= self.upper:
            return -math.inf
        return self._log_masses[k - self.lower]

    def pmf(self, k: int) -> float:
        """Return ``p_T(k)``.

        References:
            [REF-CASELLA-BERGER-2002, Section 1.3]. No new equation is introduced.

        """
        return math.exp(self.log_pmf(k))

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[int, ...]:
        """Draw by inversion of the tabulated distribution function.

        References:
            [REF-DEVROYE-1986, Chapter III]. No new equation is introduced.

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        masses = [math.exp(v) for v in self._log_masses]
        return tuple(self.lower + _categorical(masses, generator) for _ in range(count))


def _categorical(weights: Sequence[float], generator: random.Random) -> int:
    """Return an index drawn with probabilities proportional to ``weights``.

    References:
        [REF-DEVROYE-1986, Chapter III]. No new equation is introduced.

    """
    total = math.fsum(weights)
    target = generator.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            return index
    return max(i for i, w in enumerate(weights) if w > 0.0)


def normalized(log_weights: Sequence[float]) -> tuple[float, ...]:
    """Return ``exp(w_i - logsumexp(w))`` for float log-weights.

    Raises:
        ValueError: If every weight is ``-inf``.

    References:
        [EQ-PPL-ENUM-002], [REF-STAN-USERS-GUIDE-2-36].

    """
    total = log_sum_exp(list(log_weights))
    if total == -math.inf or math.isnan(total):
        raise ValueError("every discrete configuration has zero probability")
    return tuple(math.exp(w - total) for w in log_weights)


# ---------------------------------------------------------------------------
# Enumeration records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LocalEnumeration:
    """Mutable state of the enumeration inside one plate iteration.

    Args:
        sites: ``(name, support values)`` of the discrete sites met so far,
            in execution order (filled on the first pass).
        assignment: Support indices of the current pass (``None`` on the
            first pass, which uses index ``0`` everywhere).
        position: Number of discrete sites met in the current pass.

    References:
        [EQ-PPL-ENUM-001], [REF-OBERMEYER-ETAL-2019].

    """

    sites: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    assignment: tuple[int, ...] | None = None
    position: int = 0

    def index_for(self, name: str, values: tuple[Any, ...]) -> int:
        """Return the support index of the next discrete site of the pass.

        Raises:
            ModelSpecificationError: If the site differs from the first pass.

        References:
            [EQ-PPL-ENUM-001]. No new equation is introduced.

        """
        position = self.position
        self.position += 1
        if self.assignment is None:
            self.sites.append((name, values))
            configurations = math.prod(len(v) for _, v in self.sites)
            if configurations > ENUMERATION_LIMIT:
                raise ModelSpecificationError(
                    f"plate iteration enumerates {configurations} configurations; "
                    f"at most {ENUMERATION_LIMIT} are allowed"
                )
            return 0
        if position >= len(self.sites) or self.sites[position] != (name, values):
            raise ModelSpecificationError(
                f"model structure changed: discrete site {name!r} or its support "
                "differs between enumerated branches of a plate iteration"
            )
        return self.assignment[position]


@dataclass(frozen=True, slots=True)
class EnumerationGroup:
    """Conditional law of the discrete sites of one plate iteration.

    Args:
        names: Site names in execution order.
        supports: Support values of each site.
        configurations: Index tuples, one per enumerated branch.
        log_weights: Unnormalized log-probabilities of the branches.
        children: For each branch, the groups of the iterations of the
            plates (and Markov chains) nested in this iteration (empty
            without nesting).
        absent: For each branch, the continuous latent sites of the
            iteration that the branch does not declare and that are padded
            with their pseudo-priors (empty when every branch declares the
            same sites; ADR-0088).

    References:
        [EQ-PPL-ENUM-002], [EQ-PPL-LOCAL-001], [REF-STAN-USERS-GUIDE-2-36],
        [REF-OBERMEYER-ETAL-2019].

    """

    names: tuple[str, ...]
    supports: tuple[tuple[Any, ...], ...]
    configurations: tuple[tuple[int, ...], ...]
    log_weights: tuple[float, ...]
    children: tuple[tuple[Group, ...], ...] = ()
    absent: tuple[frozenset[str], ...] = ()


@dataclass(frozen=True, slots=True)
class MarkovGroup:
    """Forward-algorithm record of one eliminated Markov chain.

    Step ``t`` of the chain enumerates the joint configurations of its own
    discrete sites; ``initial[j]`` is the log-weight of configuration ``j``
    of step ``0`` and ``transitions[t - 1][i][j]`` the log-weight of the
    terms of step ``t`` when step ``t - 1`` is in configuration ``i`` and
    step ``t`` in configuration ``j`` (discrete log-masses, observations,
    and the log-sum-exp of nested plate iterations).

    Args:
        name: Chain name.
        names: Discrete site names of every step.
        supports: Support values of every site of every step.
        configurations: Index tuples of every step, one per state.
        initial: Log-weights of the states of step ``0``.
        transitions: Log-weight tables of steps ``1, ..., n - 1``.
        children: For every step, the groups of the plate iterations nested
            in it, per pair ``(i, j)`` flattened as ``i * K_t + j`` (per
            ``j`` at step ``0``); empty without nesting.

    References:
        [EQ-PPL-MARKOV-001], [EQ-PPL-MARKOV-002], [REF-RABINER-1989].

    """

    name: str
    names: tuple[tuple[str, ...], ...]
    supports: tuple[tuple[tuple[Any, ...], ...], ...]
    configurations: tuple[tuple[tuple[int, ...], ...], ...]
    initial: tuple[float, ...]
    transitions: tuple[tuple[tuple[float, ...], ...], ...]
    children: tuple[tuple[tuple[Group, ...], ...], ...] = ()


Group = EnumerationGroup | MarkovGroup
"""An enumeration record: a plate iteration or a Markov chain."""


def forward_log(
    initial: Sequence[Any], transitions: Sequence[Sequence[Sequence[Any]]]
) -> list[list[Any]]:
    """Return the forward log-variables of a chain (dual-transparent).

    ``alpha_0(j) = l_0(j)`` and
    ``alpha_t(j) = log sum_i exp(alpha_(t-1)(i) + l_t(i, j))``, so that
    ``log sum_j exp(alpha_(n-1)(j))`` is the log of the sum over every path
    at ``O(n K^2)`` cost.

    Args:
        initial: Log-weights of the states of step ``0``.
        transitions: Log-weight tables ``l_t(i, j)`` of steps ``1..n-1``.

    Returns:
        ``alpha_t`` for every step.

    Equation:
        [EQ-PPL-MARKOV-001].

    References:
        [REF-RABINER-1989], [REF-BAUM-ETAL-1970].

    """
    alphas = [list(initial)]
    for table in transitions:
        previous = alphas[-1]
        width = len(table[0]) if table else 0
        alphas.append(
            [
                log_sum_exp([previous[i] + table[i][j] for i in range(len(previous))])
                for j in range(width)
            ]
        )
    return alphas


def backward_log(
    transitions: Sequence[Sequence[Sequence[float]]], last: int
) -> list[list[float]]:
    """Return the backward log-variables of a chain.

    ``beta_(n-1)(j) = 0`` and
    ``beta_t(i) = log sum_j exp(l_(t+1)(i, j) + beta_(t+1)(j))``.

    Args:
        transitions: Log-weight tables of steps ``1..n-1``.
        last: Number of states of the last step.

    Returns:
        ``beta_t`` for every step.

    Equation:
        [EQ-PPL-MARKOV-002].

    References:
        [REF-RABINER-1989].

    """
    betas = [[0.0] * last]
    for table in reversed(transitions):
        following = betas[0]
        betas.insert(
            0,
            [
                log_sum_exp([row[j] + following[j] for j in range(len(following))])
                for row in table
            ],
        )
    return betas


def viterbi(
    initial: Sequence[float], transitions: Sequence[Sequence[Sequence[float]]]
) -> tuple[float, tuple[int, ...]]:
    """Return the largest path log-weight and its state path.

    ``delta_0(j) = l_0(j)``, ``delta_t(j) = max_i delta_(t-1)(i) + l_t(i, j)``
    with back-pointers; ties go to the smallest index.

    Args:
        initial: Log-weights of the states of step ``0``.
        transitions: Log-weight tables of steps ``1..n-1``.

    Returns:
        ``(max log-weight, state indices per step)``.

    Equation:
        [EQ-PPL-MARKOV-003].

    References:
        [REF-VITERBI-1967], [REF-RABINER-1989].

    """
    delta = list(initial)
    pointers: list[list[int]] = []
    for table in transitions:
        step: list[float] = []
        back: list[int] = []
        for j in range(len(table[0])):
            scores = [delta[i] + table[i][j] for i in range(len(delta))]
            best = max(range(len(scores)), key=scores.__getitem__)
            step.append(scores[best])
            back.append(best)
        delta = step
        pointers.append(back)
    state = max(range(len(delta)), key=delta.__getitem__)
    score = delta[state]
    path = [state]
    for back in reversed(pointers):
        state = back[state]
        path.append(state)
    return score, tuple(reversed(path))


def _chain_tables(
    group: MarkovGroup,
) -> tuple[list[list[float]], list[list[float]], float]:
    """Return the forward and backward variables and the chain log-weight.

    Raises:
        ValueError: If every path has zero probability.

    References:
        [EQ-PPL-MARKOV-002], [REF-RABINER-1989].

    """
    alphas = forward_log(group.initial, group.transitions)
    total = log_sum_exp(alphas[-1])
    if total == -math.inf or math.isnan(total):
        raise ValueError("every path of the Markov chain has zero probability")
    betas = backward_log(group.transitions, len(alphas[-1]))
    return alphas, betas, total


def _step_values(
    group: MarkovGroup, step: int, state: int, values: dict[str, Any]
) -> None:
    """Write the site values of one state of one chain step into ``values``.

    References:
        [EQ-PPL-MARKOV-002]. No new equation is introduced.

    """
    configuration = group.configurations[step][state]
    for position, name in enumerate(group.names[step]):
        values[name] = group.supports[step][position][configuration[position]]


def _pair_children(
    group: MarkovGroup, step: int, previous: int, state: int
) -> tuple[Group, ...]:
    """Return the nested groups of one state pair of one chain step.

    References:
        [EQ-PPL-MARKOV-001]. No new equation is introduced.

    """
    if not group.children or not group.children[step]:
        return ()
    width = len(group.configurations[step])
    return group.children[step][previous * width + state]


def _accumulate_chain(
    group: MarkovGroup,
    weight: float,
    result: dict[str, list[float]],
    absence: dict[str, float],
) -> None:
    """Add ``weight`` times the smoothed marginals of a chain's sites.

    ``p(z_t = j | ...) = exp(alpha_t(j) + beta_t(j) - log L)``; the groups
    nested in step ``t`` are weighted by the pairwise probabilities
    ``exp(alpha_(t-1)(i) + l_t(i, j) + beta_t(j) - log L)``.

    References:
        [EQ-PPL-MARKOV-002], [REF-RABINER-1989].

    """
    alphas, betas, total = _chain_tables(group)
    for step, (alpha, beta) in enumerate(zip(alphas, betas, strict=True)):
        marginal = [math.exp(a + b - total) for a, b in zip(alpha, beta, strict=True)]
        for position, name in enumerate(group.names[step]):
            row = result.setdefault(name, [0.0] * len(group.supports[step][position]))
            for configuration, probability in zip(
                group.configurations[step], marginal, strict=True
            ):
                row[configuration[position]] += weight * probability
        if not group.children or not group.children[step]:
            continue
        if step == 0:
            for state, probability in enumerate(marginal):
                nested = _pair_children(group, 0, 0, state)
                if probability > 0.0 and nested:
                    _accumulate_groups(nested, weight * probability, result, absence)
            continue
        table = group.transitions[step - 1]
        for previous, alpha_previous in enumerate(alphas[step - 1]):
            for state in range(len(alpha)):
                probability = math.exp(
                    alpha_previous + table[previous][state] + beta[state] - total
                )
                nested = _pair_children(group, step, previous, state)
                if probability > 0.0 and nested:
                    _accumulate_groups(nested, weight * probability, result, absence)


def _draw_chain(
    group: MarkovGroup,
    generator: random.Random,
    values: dict[str, Any],
    supports: dict[str, tuple[Any, ...]],
    absent: set[str],
) -> None:
    """Draw a chain's path by forward filtering and backward sampling.

    ``z_(n-1) ~ exp(alpha_(n-1))`` and, backwards,
    ``z_t | z_(t+1) = k ~ exp(alpha_t(i) + l_(t+1)(i, k))``; the groups
    nested in each step are then drawn given the drawn pair.

    References:
        [EQ-PPL-MARKOV-002], [REF-CHIB-1996].

    """
    alphas = forward_log(group.initial, group.transitions)
    last = len(alphas) - 1
    path = [0] * len(alphas)
    path[last] = _categorical(normalized(alphas[last]), generator)
    for step in range(last - 1, -1, -1):
        table = group.transitions[step]
        following = path[step + 1]
        weights = [alpha + table[i][following] for i, alpha in enumerate(alphas[step])]
        path[step] = _categorical(normalized(weights), generator)
    for step, state in enumerate(path):
        for position, name in enumerate(group.names[step]):
            supports.setdefault(name, group.supports[step][position])
        _step_values(group, step, state, values)
        nested = _pair_children(group, step, path[step - 1] if step else 0, state)
        if nested:
            _draw_groups(nested, generator, values, supports, absent)


@dataclass(frozen=True, slots=True)
class DiscreteConditional:
    """Exact conditional law of every discrete site given continuous values.

    Args:
        names: Global discrete site names (the union over the traced
            structures, in order of first appearance).
        supports: Their support values.
        configurations: Joint global configurations (index tuples aligned
            with ``names``; ``-1`` marks a site that the configuration does
            not declare).
        log_weights: Unnormalized log-probabilities of the configurations.
        groups: For each global configuration, the groups of its top-level
            plate iterations and Markov chains.

    References:
        [EQ-PPL-ENUM-002], [REF-STAN-USERS-GUIDE-2-36].

    """

    names: tuple[str, ...]
    supports: tuple[tuple[Any, ...], ...]
    configurations: tuple[tuple[int, ...], ...]
    log_weights: tuple[float, ...]
    groups: tuple[tuple[Group, ...], ...]


@dataclass(frozen=True, slots=True)
class DiscreteDraws:
    """Recovered posterior draws and conditional probabilities of discrete sites.

    For every pooled continuous draw ``theta^(s)``, ``probabilities[name][s]``
    is the exact ``p(z = v | theta^(s), y)`` for each support value ``v``
    (local sites are averaged over the global configurations), and
    ``draws[name][s]`` is one value drawn jointly from that conditional law.
    Averages of the probabilities over ``s`` are Rao-Blackwellized estimates
    of the posterior probabilities.

    A site that the drawn global configuration does not declare (dynamic
    site structure, ADR-0084) has the value ``None`` in that draw, and its
    conditional probabilities sum to the probability that it is declared.

    Args:
        supports: Support values by site name.
        draws: Drawn values by site name (chain after chain).
        probabilities: Conditional probabilities by site name and draw.
        structures: Index of the drawn global configuration per draw (into
            ``ModelIR.structures``); empty unless requested.
        structure_probabilities: ``p(c | theta^(s), y)`` of every global
            configuration ``c`` per draw; empty unless requested.
        absent: Per draw, the continuous sites that the drawn branches of
            the plate iterations lack (ADR-0088); empty when no branch lacks
            a site.
        absence: Per continuous site that some branch lacks, the conditional
            probability per draw that a drawn branch lacks it.

    References:
        [EQ-PPL-ENUM-002], [EQ-PPL-LOCAL-001], [REF-STAN-USERS-GUIDE-2-36].

    """

    supports: dict[str, tuple[Any, ...]]
    draws: dict[str, tuple[Any, ...]]
    probabilities: dict[str, tuple[tuple[float, ...], ...]]
    structures: tuple[int, ...] = ()
    structure_probabilities: tuple[tuple[float, ...], ...] = ()
    absent: tuple[frozenset[str], ...] = ()
    absence: dict[str, tuple[float, ...]] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        """Return the discrete site names in execution order.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return tuple(self.supports)

    def marginal(self, name: str) -> tuple[float, ...]:
        """Return the posterior probability of each support value of a site.

        The average over draws of the exact conditional probabilities.

        Raises:
            KeyError: If ``name`` is not a discrete site.

        Equation:
            [EQ-PPL-ENUM-002].

        References:
            [REF-STAN-USERS-GUIDE-2-36].

        """
        rows = self.probabilities[name]
        width = len(self.supports[name])
        return tuple(
            math.fsum(row[k] for row in rows) / len(rows) for k in range(width)
        )


def _accumulate_groups(
    groups: Sequence[Group],
    weight: float,
    result: dict[str, list[float]],
    absence: dict[str, float],
) -> None:
    """Add ``weight * p(z = v | ...)`` of every group, recursing into nests.

    For a plate iteration whose branches lack some continuous sites,
    ``weight * p(branch)`` is added to the absence mass of each site the
    branch lacks; absences at different levels are disjoint events.

    References:
        [EQ-PPL-ENUM-002], [EQ-PPL-LOCAL-001], [REF-OBERMEYER-ETAL-2019].

    """
    for group in groups:
        if isinstance(group, MarkovGroup):
            _accumulate_chain(group, weight, result, absence)
            continue
        local = normalized(group.log_weights)
        for position, name in enumerate(group.names):
            row = result.setdefault(name, [0.0] * len(group.supports[position]))
            for configuration, probability in zip(
                group.configurations, local, strict=True
            ):
                row[configuration[position]] += weight * probability
        for missing, probability in zip(group.absent, local, strict=False):
            for name in missing:
                absence[name] = absence.get(name, 0.0) + weight * probability
        for branch, probability in zip(group.children, local, strict=False):
            if probability > 0.0 and branch:
                _accumulate_groups(branch, weight * probability, result, absence)


def _site_probabilities(
    conditional: DiscreteConditional, global_weights: Sequence[float]
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Return ``p(z = v | theta, y)`` for every discrete site and the absences.

    Local probabilities are averaged over the global configurations and, in
    nested plates, over the branches of the enclosing iterations. The second
    mapping holds, per continuous site that some plate branch lacks, the
    probability that a drawn branch lacks it.

    References:
        [EQ-PPL-ENUM-002], [EQ-PPL-LOCAL-001].

    """
    result: dict[str, list[float]] = {}
    absence: dict[str, float] = {}
    for position, name in enumerate(conditional.names):
        row = [0.0] * len(conditional.supports[position])
        for configuration, weight in zip(
            conditional.configurations, global_weights, strict=True
        ):
            if configuration[position] >= 0:
                row[configuration[position]] += weight
        result[name] = row
    for groups, weight in zip(conditional.groups, global_weights, strict=True):
        if weight == 0.0:
            continue
        _accumulate_groups(groups, weight, result, absence)
    return result, absence


def _draw_groups(
    groups: Sequence[Group],
    generator: random.Random,
    values: dict[str, Any],
    supports: dict[str, tuple[Any, ...]],
    absent: set[str],
) -> None:
    """Draw each group's branch, then the groups nested in the drawn branch.

    A Markov chain is drawn by forward filtering and backward sampling; the
    continuous sites that a drawn plate branch lacks are added to
    ``absent``.

    References:
        [EQ-PPL-ENUM-002], [EQ-PPL-MARKOV-002], [REF-OBERMEYER-ETAL-2019].

    """
    for group in groups:
        if isinstance(group, MarkovGroup):
            _draw_chain(group, generator, values, supports, absent)
            continue
        local = normalized(group.log_weights)
        chosen = _categorical(local, generator)
        branch = group.configurations[chosen]
        for position, name in enumerate(group.names):
            supports.setdefault(name, group.supports[position])
            values[name] = group.supports[position][branch[position]]
        if group.absent:
            absent.update(group.absent[chosen])
        if group.children and group.children[chosen]:
            _draw_groups(group.children[chosen], generator, values, supports, absent)


def _nested_groups(group: Group) -> Iterator[Group]:
    """Yield the groups nested directly in a group (every branch or pair).

    References:
        [EQ-PPL-ENUM-004]. No new equation is introduced.

    """
    if isinstance(group, MarkovGroup):
        for step in group.children:
            for pair in step:
                yield from pair
        return
    for branch in group.children:
        yield from branch


def _group_support(conditional: DiscreteConditional, name: str) -> tuple[Any, ...]:
    """Return the support of a local site from the enumeration tree.

    Raises:
        KeyError: If no group declares ``name``.

    References:
        [EQ-PPL-ENUM-002]. No new equation is introduced.

    """
    pending: list[Group] = [group for groups in conditional.groups for group in groups]
    while pending:
        group = pending.pop()
        if isinstance(group, MarkovGroup):
            for step, names in enumerate(group.names):
                if name in names:
                    return group.supports[step][names.index(name)]
        elif name in group.names:
            return group.supports[group.names.index(name)]
        pending.extend(_nested_groups(group))
    raise KeyError(name)


def recover_discrete(
    conditionals: Iterable[DiscreteConditional],
    seed: int | RandomStream | None,
    *,
    keep_structures: bool = False,
) -> DiscreteDraws:
    """Draw discrete sites from their exact conditional laws, draw by draw.

    For each conditional: a global configuration ``c ~ p(c | theta, y)``, then
    each plate iteration's configuration from ``p(z_i | theta, c, y)`` and,
    inside it, each nested iteration's configuration given the drawn branch;
    the path of a Markov chain is drawn by forward filtering and backward
    sampling ([EQ-PPL-MARKOV-002]).

    Args:
        conditionals: One :class:`DiscreteConditional` per continuous draw.
        seed: Integer seed or stream.
        keep_structures: Whether to keep the drawn global configuration and
            the configuration probabilities of every draw.

    Returns:
        Drawn values and conditional probabilities.

    Raises:
        ValueError: If a draw gives every configuration zero probability.

    Equation:
        [EQ-PPL-ENUM-002], [EQ-PPL-MARKOV-002].

    References:
        [REF-STAN-USERS-GUIDE-2-36], [REF-OBERMEYER-ETAL-2019], [REF-CHIB-1996].

    """
    generator = resolve_stream(seed).generator()
    supports: dict[str, tuple[Any, ...]] = {}
    per_draw: list[dict[str, Any]] = []
    per_rows: list[dict[str, list[float]]] = []
    per_absent: list[frozenset[str]] = []
    per_absence: list[dict[str, float]] = []
    structures: list[int] = []
    structure_probabilities: list[tuple[float, ...]] = []
    for conditional in conditionals:
        weights = normalized(conditional.log_weights)
        chosen = _categorical(weights, generator)
        values: dict[str, Any] = {}
        for position, name in enumerate(conditional.names):
            support = conditional.supports[position]
            supports.setdefault(name, support)
            index = conditional.configurations[chosen][position]
            if index >= 0:
                values[name] = support[index]
        absent: set[str] = set()
        _draw_groups(conditional.groups[chosen], generator, values, supports, absent)
        rows, absence = _site_probabilities(conditional, weights)
        for name in rows:
            if name not in supports:
                supports[name] = _group_support(conditional, name)
        per_draw.append(values)
        per_rows.append(rows)
        per_absent.append(frozenset(absent))
        per_absence.append(absence)
        if keep_structures:
            structures.append(chosen)
            structure_probabilities.append(weights)
    draws = {name: tuple(values.get(name) for values in per_draw) for name in supports}
    probabilities = {
        name: tuple(tuple(rows.get(name, [0.0] * len(support))) for rows in per_rows)
        for name, support in supports.items()
    }
    optional = sorted({name for absence in per_absence for name in absence})
    return DiscreteDraws(
        supports,
        draws,
        probabilities,
        tuple(structures),
        tuple(structure_probabilities),
        tuple(per_absent) if optional else (),
        {
            name: tuple(absence.get(name, 0.0) for absence in per_absence)
            for name in optional
        },
    )


def _best_groups(groups: Sequence[Group]) -> tuple[float, dict[str, Any]]:
    """Return the max-product gain and the best values of a list of groups.

    The gain of a group is ``max - log sum`` over its branches (or paths),
    so that adding it to a log-weight that contains the group's log-sum-exp
    replaces the sum by the maximum.

    References:
        [EQ-PPL-MARKOV-003]. No new equation is introduced.

    """
    gain = 0.0
    values: dict[str, Any] = {}
    for group in groups:
        group_gain, group_values = _best_group(group)
        gain += group_gain
        values.update(group_values)
    return gain, values


def _best_group(group: Group) -> tuple[float, dict[str, Any]]:
    """Return the max-product gain and the best values of one group.

    A plate iteration takes the branch that maximizes its log-weight plus
    the gains of its nested groups; a Markov chain takes the Viterbi path of
    its log-weight tables corrected by the gains of the nested groups.

    References:
        [EQ-PPL-MARKOV-003], [REF-VITERBI-1967].

    """
    if isinstance(group, MarkovGroup):
        initial = list(group.initial)
        nested_values: dict[tuple[int, int, int], dict[str, Any]] = {}
        for state in range(len(initial)):
            child_gain, child_values = _best_groups(_pair_children(group, 0, 0, state))
            initial[state] += child_gain
            nested_values[(0, 0, state)] = child_values
        transitions: list[list[list[float]]] = []
        for step, table in enumerate(group.transitions, start=1):
            corrected = [list(row) for row in table]
            for previous, row in enumerate(corrected):
                for state in range(len(row)):
                    nested = _pair_children(group, step, previous, state)
                    if nested:
                        child_gain, child_values = _best_groups(nested)
                        row[state] += child_gain
                        nested_values[(step, previous, state)] = child_values
            transitions.append(corrected)
        score, path = viterbi(initial, transitions)
        total = log_sum_exp(forward_log(group.initial, group.transitions)[-1])
        values: dict[str, Any] = {}
        for step, state in enumerate(path):
            _step_values(group, step, state, values)
            values.update(
                nested_values.get((step, path[step - 1] if step else 0, state), {})
            )
        return score - total, values
    scores: list[float] = []
    branch_values: list[dict[str, Any]] = []
    for index, log_weight in enumerate(group.log_weights):
        nested = group.children[index] if group.children else ()
        child_gain, child_values = _best_groups(nested)
        scores.append(log_weight + child_gain)
        branch_values.append(child_values)
    best = max(range(len(scores)), key=scores.__getitem__)
    values = dict(branch_values[best])
    for position, name in enumerate(group.names):
        values[name] = group.supports[position][group.configurations[best][position]]
    return scores[best] - log_sum_exp(list(group.log_weights)), values


def most_probable(conditional: DiscreteConditional) -> dict[str, Any]:
    """Return the jointly most probable values of every discrete site.

    Max-product elimination over the enumeration record: the global
    configuration ``c`` maximizes ``l_c + sum_g (max_g - logsumexp_g)`` over
    the groups ``g`` of its plate iterations and chains, each plate
    iteration takes its best branch, and each chain its Viterbi path. Sites
    that the chosen configuration does not declare are omitted.

    Args:
        conditional: Enumeration record at one continuous point.

    Returns:
        ``{site name: value}``.

    Raises:
        ValueError: If every configuration has zero probability.

    Equation:
        [EQ-PPL-MARKOV-003].

    References:
        [REF-VITERBI-1967], [REF-RABINER-1989].

    """
    best: tuple[float, dict[str, Any]] | None = None
    for index, log_weight in enumerate(conditional.log_weights):
        if log_weight == -math.inf or math.isnan(log_weight):
            continue
        gain, values = _best_groups(conditional.groups[index])
        score = log_weight + gain
        if best is None or score > best[0]:
            for position, name in enumerate(conditional.names):
                state = conditional.configurations[index][position]
                if state >= 0:
                    values[name] = conditional.supports[position][state]
            best = (score, values)
    if best is None:
        raise ValueError("every discrete configuration has zero probability")
    return best[1]
