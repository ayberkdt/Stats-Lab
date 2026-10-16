"""Common-cause failure (CCF) models expanded into independent basic events.

A CCF group is a set of ``m`` similar (symmetric) components. The parametric
models of NUREG/CR-4780 and NUREG/CR-5485 split the total failure probability
``Q_t`` of each member into ``Q_k^(m)``, the probability of a basic event
that fails one *specific* group of ``k`` members and no other member,
with the normalisation ``Q_t = sum_{k=1}^m C(m-1, k-1) Q_k^(m)``. Every model
here is linear in ``Q_t``: ``Q_k = c_k Q_t``.

* Beta factor: ``Q_1 = (1 - beta) Q_t``, ``Q_m = beta Q_t``, others zero.
* Multiple Greek letter (``rho_1 = 1, rho_2 = beta, rho_3 = gamma, ...,
  rho_(m+1) = 0``): ``Q_k = (prod_{i<=k} rho_i)(1 - rho_(k+1)) Q_t /
  C(m-1, k-1)``.
* Alpha factor with ``alpha_t = sum_k k alpha_k``: non-staggered testing
  ``Q_k = k / C(m-1, k-1) * alpha_k / alpha_t * Q_t``; staggered testing
  ``Q_k = alpha_k Q_t / C(m-1, k-1)``.

:func:`expand_common_cause` replaces every member ``i`` of a group by the
series structure of its basic events: the independent event ``IND[i]`` and
every CCF event ``CCF[S]`` for a subset ``S`` that contains ``i`` with
``|S| >= 2``. All basic events are independent, so member ``i`` fails exactly
when at least one of its events occurs, and the expanded structure feeds the
BDD without approximation. Events whose coefficient ``c_k`` is zero are
omitted.

Because the events are combined exactly (by union, not by summation), a
member's marginal failure probability is
``1 - prod_k (1 - Q_k)^C(m-1, k-1)``, slightly *below* ``Q_t`` (the
normalisation is the rare-event sum). For time-dependent members,
:class:`~probstat.systems.SystemModel` applies the coefficients either to
``Q_t(t)`` or to the cumulative hazard ``-log(1 - Q_t(t))``, which keeps
the marginal exactly ``Q_t(t)`` (its ``ccf_basis`` argument). The
parameters ``beta``, ``alpha_k`` are estimated from sparse event data and
carry large uncertainty; the
symmetric assumption requires every member to have the same ``Q_t``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from probstat.foundations.validation import (
    require_non_negative_finite_real,
    require_probability,
)

from .structure import (
    Component,
    Expression,
    KOutOfN,
    Parallel,
    Series,
    expression_components,
)

_SUM_TOLERANCE = 1e-9
_MAX_GROUP_SIZE = 16


def _members(components: Sequence[str]) -> tuple[str, ...]:
    """Validate a CCF group's members.

    Raises:
        TypeError: If a member is not a string.
        ValueError: If fewer than two, too many, or repeated members.

    References:
        [REF-MOSLEH-ETAL-1998]. No equation is evaluated.

    """
    members = tuple(components)
    for member in members:
        if not isinstance(member, str) or not member.strip():
            raise TypeError("group members must be non-empty strings")
    if len(set(members)) != len(members):
        raise ValueError("group members must be distinct")
    if not 2 <= len(members) <= _MAX_GROUP_SIZE:
        raise ValueError(f"a CCF group needs 2 to {_MAX_GROUP_SIZE} members")
    return members


@runtime_checkable
class CCFGroup(Protocol):
    """A symmetric common-cause group with order coefficients ``c_k``.

    References:
        [REF-MOSLEH-ETAL-1998], [REF-MOSLEH-ETAL-1988].

    """

    @property
    def components(self) -> tuple[str, ...]:
        """Return the group members.

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        ...

    def order_fractions(self) -> tuple[float, ...]:
        """Return ``c_1..c_m`` with ``Q_k = c_k Q_t``.

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        ...


def order_probabilities(group: CCFGroup, total: float) -> tuple[float, ...]:
    """Return ``Q_1^(m), ..., Q_m^(m)`` for a total failure probability.

    Args:
        group: CCF group.
        total: ``Q_t`` of each member.

    Returns:
        Per-order basic-event probabilities.

    Equation:
        [EQ-SYS-CCF-001], [EQ-SYS-CCF-002], [EQ-SYS-CCF-003].

    References:
        [REF-MOSLEH-ETAL-1998].

    """
    q_total = require_probability(total, name="total")
    return tuple(min(1.0, c * q_total) for c in group.order_fractions())


def normalisation(group: CCFGroup) -> float:
    """Return ``sum_k C(m-1, k-1) c_k`` (one for a valid model).

    Args:
        group: CCF group.

    Returns:
        The normalisation sum.

    Equation:
        [EQ-SYS-CCF-004].

    References:
        [REF-MOSLEH-ETAL-1998].

    """
    fractions = group.order_fractions()
    m = len(fractions)
    return math.fsum(math.comb(m - 1, k - 1) * c for k, c in enumerate(fractions, 1))


@dataclass(frozen=True, slots=True)
class BetaFactorGroup:
    """Beta-factor CCF model: a fraction ``beta`` of failures fail all members.

    Args:
        components: Member names (at least two).
        beta: Fraction of each member's failure probability due to CCF.

    References:
        [REF-MOSLEH-ETAL-1998], [REF-MOSLEH-ETAL-1988].

    """

    components: tuple[str, ...]
    beta: float

    def __post_init__(self) -> None:
        """Validate members and ``beta``.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(self, "components", _members(self.components))
        object.__setattr__(self, "beta", require_probability(self.beta, name="beta"))

    def order_fractions(self) -> tuple[float, ...]:
        """Return ``(1 - beta, 0, ..., 0, beta)``.

        Returns:
            Coefficients ``c_k``.

        Equation:
            [EQ-SYS-CCF-001].

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        m = len(self.components)
        return (1.0 - self.beta, *([0.0] * (m - 2)), self.beta)


@dataclass(frozen=True, slots=True)
class MultipleGreekLetterGroup:
    """Multiple Greek letter CCF model.

    Args:
        components: Member names (``m >= 2``).
        factors: ``(beta, gamma, delta, ...)``, ``m - 1`` conditional
            probabilities in ``[0, 1]``: ``beta`` that a failure is shared by
            at least one more member, ``gamma`` that it is shared by at least
            two more given at least one more, and so on.

    References:
        [REF-MOSLEH-ETAL-1998], [REF-MOSLEH-ETAL-1988].

    """

    components: tuple[str, ...]
    factors: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate members and factors.

        References:
            Project validation policy. No equation is evaluated.

        """
        members = _members(self.components)
        factors = tuple(
            require_probability(value, name=f"factors[{index}]")
            for index, value in enumerate(self.factors)
        )
        if len(factors) != len(members) - 1:
            raise ValueError("an m-member MGL group needs m - 1 factors")
        object.__setattr__(self, "components", members)
        object.__setattr__(self, "factors", factors)

    def order_fractions(self) -> tuple[float, ...]:
        """Return ``c_k = (prod_{i<=k} rho_i)(1 - rho_(k+1)) / C(m-1, k-1)``.

        Returns:
            Coefficients ``c_k``.

        Equation:
            [EQ-SYS-CCF-002].

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        m = len(self.components)
        rho = (1.0, *self.factors, 0.0)
        fractions = []
        product = 1.0
        for k in range(1, m + 1):
            product *= rho[k - 1]
            fractions.append(product * (1.0 - rho[k]) / math.comb(m - 1, k - 1))
        return tuple(fractions)


@dataclass(frozen=True, slots=True)
class AlphaFactorGroup:
    """Alpha-factor CCF model.

    Args:
        components: Member names (``m >= 2``).
        alphas: ``alpha_1..alpha_m``, the fractions of failure *events* that
            involve exactly ``k`` members; non-negative, summing to one.
        testing: ``"non-staggered"`` or ``"staggered"`` testing scheme.

    References:
        [REF-MOSLEH-ETAL-1998], [REF-MOSLEH-ETAL-1988].

    """

    components: tuple[str, ...]
    alphas: tuple[float, ...]
    testing: Literal["non-staggered", "staggered"] = "non-staggered"

    def __post_init__(self) -> None:
        """Validate members, alphas, and the testing scheme.

        References:
            Project validation policy. No equation is evaluated.

        """
        members = _members(self.components)
        alphas = tuple(
            require_non_negative_finite_real(value, name=f"alphas[{index}]")
            for index, value in enumerate(self.alphas)
        )
        if len(alphas) != len(members):
            raise ValueError("an m-member alpha-factor group needs m alphas")
        if abs(math.fsum(alphas) - 1.0) > _SUM_TOLERANCE:
            raise ValueError("alpha factors must sum to one")
        if self.testing not in ("non-staggered", "staggered"):
            raise ValueError("testing must be 'non-staggered' or 'staggered'")
        object.__setattr__(self, "components", members)
        object.__setattr__(self, "alphas", alphas)

    @property
    def alpha_total(self) -> float:
        """Return ``alpha_t = sum_k k alpha_k``.

        Equation:
            [EQ-SYS-CCF-003].

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        return math.fsum(k * a for k, a in enumerate(self.alphas, 1))

    def order_fractions(self) -> tuple[float, ...]:
        """Return the alpha-factor coefficients ``c_k``.

        Returns:
            ``k alpha_k / (C(m-1, k-1) alpha_t)`` (non-staggered) or
            ``alpha_k / C(m-1, k-1)`` (staggered).

        Equation:
            [EQ-SYS-CCF-003].

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        m = len(self.components)
        if self.testing == "staggered":
            return tuple(
                a / math.comb(m - 1, k - 1) for k, a in enumerate(self.alphas, 1)
            )
        total = self.alpha_total
        return tuple(
            k * a / (math.comb(m - 1, k - 1) * total)
            for k, a in enumerate(self.alphas, 1)
        )


@dataclass(frozen=True, slots=True)
class CCFEvent:
    """One basic event of a CCF expansion.

    Args:
        name: Event name (``IND[i]`` or ``CCF[i,j,...]``).
        members: Group members failed by the event.
        order: ``k = len(members)``.
        fraction: ``c_k``; the event probability is ``c_k Q_t``.

    References:
        [REF-MOSLEH-ETAL-1998].

    """

    name: str
    members: tuple[str, ...]
    order: int
    fraction: float


@dataclass(frozen=True, slots=True)
class CommonCauseExpansion:
    """A structure whose group members are replaced by their basic events.

    Args:
        expression: Expanded structure over basic events (events not part of
            any group keep their component name).
        events: Event name -> :class:`CCFEvent` for every CCF-model event.
        component_events: Group member -> names of its basic events.
        group_of: Group member -> index of its group in the input order.

    References:
        [REF-MOSLEH-ETAL-1998].

    """

    expression: Expression
    events: Mapping[str, CCFEvent]
    component_events: Mapping[str, tuple[str, ...]]
    group_of: Mapping[str, int]


def expand_common_cause(
    expression: Expression, groups: Sequence[CCFGroup]
) -> CommonCauseExpansion:
    """Replace every group member by the series of its basic events.

    Args:
        expression: Structure over components.
        groups: Disjoint CCF groups of components that appear in it.

    Returns:
        The expanded structure and the event table.

    Raises:
        TypeError: If a group does not implement :class:`CCFGroup`.
        ValueError: If groups overlap, name unknown components, have an
            invalid normalisation, or an event name collides with a
            component name.

    Equation:
        [EQ-SYS-CCF-005].

    References:
        [REF-MOSLEH-ETAL-1998], [REF-RAUZY-1993].

    """
    names = set(expression_components(expression))
    events: dict[str, CCFEvent] = {}
    component_events: dict[str, list[str]] = {}
    group_of: dict[str, int] = {}
    for index, group in enumerate(groups):
        if not isinstance(group, CCFGroup):
            raise TypeError(f"groups[{index}] must be a CCF group")
        members = tuple(group.components)
        fractions = tuple(group.order_fractions())
        if len(fractions) != len(members):
            raise ValueError(f"groups[{index}] has {len(fractions)} coefficients")
        if abs(normalisation(group) - 1.0) > _SUM_TOLERANCE:
            raise ValueError(f"groups[{index}] coefficients are not normalised")
        for member in members:
            if member not in names:
                raise ValueError(f"CCF member {member!r} is not in the structure")
            if member in group_of:
                raise ValueError(f"component {member!r} is in two CCF groups")
            group_of[member] = index
            component_events[member] = []
        for k in range(1, len(members) + 1):
            fraction = fractions[k - 1]
            if fraction == 0.0:
                continue
            for subset in combinations(members, k):
                label = "IND" if k == 1 else "CCF"
                name = f"{label}[{','.join(subset)}]"
                if name in names:
                    raise ValueError(f"event name {name!r} collides with a component")
                events[name] = CCFEvent(name, subset, k, fraction)
                for member in subset:
                    component_events[member].append(name)

    def rewrite(node: Expression) -> Expression:
        """Substitute group members by the series of their events.

        References:
            [EQ-SYS-CCF-005], [REF-MOSLEH-ETAL-1998].

        """
        if isinstance(node, Component):
            if node.name not in component_events:
                return node
            # Normalisation makes some c_k positive, and every member lies in
            # subsets of every order, so each member has at least one event.
            parts = component_events[node.name]
            return Series(tuple(Component(part) for part in parts))
        if isinstance(node, Series):
            return Series(tuple(rewrite(item) for item in node.items))
        if isinstance(node, Parallel):
            return Parallel(tuple(rewrite(item) for item in node.items))
        assert isinstance(node, KOutOfN)
        return KOutOfN(node.k, tuple(rewrite(item) for item in node.items))

    return CommonCauseExpansion(
        expression=rewrite(expression),
        events=MappingProxyType(events),
        component_events=MappingProxyType(
            {key: tuple(value) for key, value in component_events.items()}
        ),
        group_of=MappingProxyType(group_of),
    )
