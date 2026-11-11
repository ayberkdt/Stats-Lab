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
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
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

    References:
        [EQ-PPL-ENUM-002], [REF-STAN-USERS-GUIDE-2-36].

    """

    names: tuple[str, ...]
    supports: tuple[tuple[Any, ...], ...]
    configurations: tuple[tuple[int, ...], ...]
    log_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DiscreteConditional:
    """Exact conditional law of every discrete site given continuous values.

    Args:
        names: Global discrete site names.
        supports: Their support values.
        configurations: Joint global configurations (index tuples).
        log_weights: Unnormalized log-probabilities of the configurations.
        groups: For each global configuration, the local groups of the
            plate iterations (same order in every configuration).

    References:
        [EQ-PPL-ENUM-002], [REF-STAN-USERS-GUIDE-2-36].

    """

    names: tuple[str, ...]
    supports: tuple[tuple[Any, ...], ...]
    configurations: tuple[tuple[int, ...], ...]
    log_weights: tuple[float, ...]
    groups: tuple[tuple[EnumerationGroup, ...], ...]


@dataclass(frozen=True, slots=True)
class DiscreteDraws:
    """Recovered posterior draws and conditional probabilities of discrete sites.

    For every pooled continuous draw ``theta^(s)``, ``probabilities[name][s]``
    is the exact ``p(z = v | theta^(s), y)`` for each support value ``v``
    (local sites are averaged over the global configurations), and
    ``draws[name][s]`` is one value drawn jointly from that conditional law.
    Averages of the probabilities over ``s`` are Rao-Blackwellized estimates
    of the posterior probabilities.

    Args:
        supports: Support values by site name.
        draws: Drawn values by site name (chain after chain).
        probabilities: Conditional probabilities by site name and draw.

    References:
        [EQ-PPL-ENUM-002], [REF-STAN-USERS-GUIDE-2-36].

    """

    supports: dict[str, tuple[Any, ...]]
    draws: dict[str, tuple[Any, ...]]
    probabilities: dict[str, tuple[tuple[float, ...], ...]]

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


def _site_probabilities(
    conditional: DiscreteConditional, global_weights: Sequence[float]
) -> dict[str, list[float]]:
    """Return ``p(z = v | theta, y)`` for every discrete site.

    Local probabilities are averaged over the global configurations.

    References:
        [EQ-PPL-ENUM-002]. No new equation is introduced.

    """
    result: dict[str, list[float]] = {}
    for position, name in enumerate(conditional.names):
        row = [0.0] * len(conditional.supports[position])
        for configuration, weight in zip(
            conditional.configurations, global_weights, strict=True
        ):
            row[configuration[position]] += weight
        result[name] = row
    for groups, weight in zip(conditional.groups, global_weights, strict=True):
        if weight == 0.0:
            continue
        for group in groups:
            local = normalized(group.log_weights)
            for position, name in enumerate(group.names):
                row = result.setdefault(name, [0.0] * len(group.supports[position]))
                for configuration, probability in zip(
                    group.configurations, local, strict=True
                ):
                    row[configuration[position]] += weight * probability
    return result


def recover_discrete(
    conditionals: Iterable[DiscreteConditional], seed: int | RandomStream | None
) -> DiscreteDraws:
    """Draw discrete sites from their exact conditional laws, draw by draw.

    For each conditional: a global configuration ``c ~ p(c | theta, y)``, then
    each plate iteration's configuration from ``p(z_i | theta, c, y)``.

    Args:
        conditionals: One :class:`DiscreteConditional` per continuous draw.
        seed: Integer seed or stream.

    Returns:
        Drawn values and conditional probabilities.

    Raises:
        ValueError: If a draw gives every configuration zero probability.

    Equation:
        [EQ-PPL-ENUM-002].

    References:
        [REF-STAN-USERS-GUIDE-2-36], [REF-OBERMEYER-ETAL-2019].

    """
    generator = resolve_stream(seed).generator()
    supports: dict[str, tuple[Any, ...]] = {}
    draws: dict[str, list[Any]] = {}
    probabilities: dict[str, list[tuple[float, ...]]] = {}
    for conditional in conditionals:
        weights = normalized(conditional.log_weights)
        chosen = _categorical(weights, generator)
        values: dict[str, Any] = {}
        for position, name in enumerate(conditional.names):
            support = conditional.supports[position]
            supports.setdefault(name, support)
            values[name] = support[conditional.configurations[chosen][position]]
        for group in conditional.groups[chosen]:
            local = normalized(group.log_weights)
            branch = group.configurations[_categorical(local, generator)]
            for position, name in enumerate(group.names):
                supports.setdefault(name, group.supports[position])
                values[name] = group.supports[position][branch[position]]
        for name, row in _site_probabilities(conditional, weights).items():
            probabilities.setdefault(name, []).append(tuple(row))
        for name, value in values.items():
            draws.setdefault(name, []).append(value)
    return DiscreteDraws(
        supports,
        {name: tuple(values) for name, values in draws.items()},
        {name: tuple(rows) for name, rows in probabilities.items()},
    )
