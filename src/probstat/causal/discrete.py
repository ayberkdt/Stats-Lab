"""Discrete structural causal models, interventions, and adjustment formulas.

A :class:`DiscreteCausalModel` assigns each finite variable a conditional
probability table given its parents in a :class:`~probstat.causal.CausalGraph`.
The observational joint law is the Markov factorization (EQ-CAUSAL-FACT-001);
the law under ``do(X = x)`` is the truncated factorization
(EQ-CAUSAL-TRUNC-001), which drops the factors of the intervened variables.

From an observational :class:`JointTable` alone, interventional distributions
are recovered by the backdoor adjustment formula (EQ-CAUSAL-ADJ-001) or the
frontdoor formula (EQ-CAUSAL-FD-001), when the corresponding graphical
criterion holds. :func:`compare_naive_and_adjusted` contrasts adjustment with
naive conditioning and flags a Simpson reversal.

All arithmetic is exact enumeration in floating point; the cost grows with the
product of the domain sizes, so the module targets small teaching models.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType

from probstat.foundations.validation import require_non_negative_finite_real

from .graphs import CausalGraph

Assignment = tuple[Hashable, ...]
NORMALIZATION_TOLERANCE = 1e-9


def _validate_domains(
    domains: Mapping[str, Sequence[Hashable]],
) -> dict[str, tuple[Hashable, ...]]:
    """Validate finite, non-empty variable domains with distinct values.

    Args:
        domains: Mapping from variable name to its ordered values.

    Returns:
        Copy with tuple-valued domains.

    Raises:
        TypeError: If a name is not a string.
        ValueError: If there are no variables, or a domain is empty or has
            repeated values.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked: dict[str, tuple[Hashable, ...]] = {}
    for name, values in domains.items():
        if not isinstance(name, str) or not name:
            raise TypeError("variable names must be non-empty strings")
        values_tuple = tuple(values)
        if not values_tuple:
            raise ValueError(f"domain of {name!r} is empty")
        if len(set(values_tuple)) != len(values_tuple):
            raise ValueError(f"domain of {name!r} has repeated values")
        checked[name] = values_tuple
    if not checked:
        raise ValueError("at least one variable is required")
    return checked


@dataclass(frozen=True, slots=True, init=False, eq=False)
class JointTable:
    """A probability mass function over all assignments of finite variables.

    Attributes:
        variables: Variable names; assignment tuples follow this order.
        domains: Ordered values of each variable.
        probabilities: Probability of every full assignment (zeros included).

    References:
        [REF-PEARL-2009, Section 1.1].

    """

    variables: tuple[str, ...]
    domains: Mapping[str, tuple[Hashable, ...]]
    probabilities: Mapping[Assignment, float]

    def __init__(
        self,
        variables: Sequence[str],
        domains: Mapping[str, Sequence[Hashable]],
        probabilities: Mapping[Assignment, object],
    ) -> None:
        """Validate a normalized table; missing assignments have mass zero.

        Args:
            variables: Variable order used by assignment tuples.
            domains: Values of each variable (keys must equal ``variables``).
            probabilities: Mapping from full assignment tuple to probability.

        Raises:
            TypeError: If a probability is not real.
            ValueError: If names or domains disagree, an assignment is
                invalid, a probability is negative or non-finite, or the total
                differs from one by more than ``1e-9``.

        References:
            [REF-PEARL-2009, Section 1.1]. No equation is evaluated.

        """
        names = tuple(variables)
        if len(set(names)) != len(names):
            raise ValueError("variables must be distinct")
        checked = _validate_domains(domains)
        if set(checked) != set(names):
            raise ValueError("domains must have exactly one entry per variable")
        table: dict[Assignment, float] = {
            assignment: 0.0
            for assignment in product(*(checked[name] for name in names))
        }
        for assignment, value in probabilities.items():
            if assignment not in table:
                raise ValueError(f"invalid assignment {assignment!r} for {names}")
            table[assignment] = require_non_negative_finite_real(
                value, name=f"probability{assignment!r}"
            )
        total = math.fsum(table.values())
        if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
            raise ValueError(f"probabilities sum to {total}, not one")
        object.__setattr__(self, "variables", names)
        object.__setattr__(
            self, "domains", MappingProxyType({n: checked[n] for n in names})
        )
        object.__setattr__(self, "probabilities", MappingProxyType(table))

    @classmethod
    def from_counts(
        cls,
        variables: Sequence[str],
        counts: Mapping[Assignment, object],
        domains: Mapping[str, Sequence[Hashable]] | None = None,
    ) -> JointTable:
        """Build the empirical joint table from cell counts.

        Args:
            variables: Variable order used by the count keys.
            counts: Non-negative count (or weight) of each observed assignment.
            domains: Optional domains; by default each domain lists the values
                in order of first appearance in ``counts``.

        Returns:
            Table with probabilities ``count / total``.

        Raises:
            TypeError: If a count is not real.
            ValueError: If counts are negative, all zero, or keys are invalid.

        References:
            [REF-PEARL-2009, Section 6.1]. No equation is evaluated.

        """
        names = tuple(variables)
        checked_counts: dict[Assignment, float] = {}
        for key, value in counts.items():
            if not isinstance(key, tuple) or len(key) != len(names):
                raise ValueError(f"count key {key!r} must be a tuple of {names}")
            checked_counts[key] = require_non_negative_finite_real(
                value, name=f"count{key!r}"
            )
        total = math.fsum(checked_counts.values())
        if total <= 0.0:
            raise ValueError("counts must have a positive total")
        if domains is None:
            inferred: dict[str, list[Hashable]] = {name: [] for name in names}
            for key in checked_counts:
                for name, value in zip(names, key, strict=True):
                    if value not in inferred[name]:
                        inferred[name].append(value)
            domains = inferred
        return cls(
            names,
            domains,
            {key: value / total for key, value in checked_counts.items()},
        )

    def _index(self, names: Iterable[str]) -> tuple[int, ...]:
        """Return the positions of variables inside assignment tuples.

        Args:
            names: Variable names.

        Returns:
            Tuple positions.

        Raises:
            KeyError: If a name is unknown.

        References:
            Project validation policy. No equation is evaluated.

        """
        positions: list[int] = []
        for name in names:
            if name not in self.domains:
                raise KeyError(f"unknown variable {name!r}")
            positions.append(self.variables.index(name))
        return tuple(positions)

    def _event(self, assignment: Mapping[str, Hashable]) -> list[tuple[int, Hashable]]:
        """Validate a partial assignment and return ``(position, value)`` pairs.

        Args:
            assignment: Mapping from variable name to value.

        Returns:
            Position and value pairs.

        Raises:
            KeyError: If a variable is unknown.
            ValueError: If a value is outside its domain.

        References:
            Project validation policy. No equation is evaluated.

        """
        pairs: list[tuple[int, Hashable]] = []
        for name, value in assignment.items():
            (position,) = self._index((name,))
            if value not in self.domains[name]:
                raise ValueError(f"{value!r} is not in the domain of {name!r}")
            pairs.append((position, value))
        return pairs

    def probability(self, assignment: Mapping[str, Hashable]) -> float:
        """Return the probability of a partial assignment (an event).

        Args:
            assignment: Values of some variables; the rest are summed out.

        Returns:
            ``P(V_S = v_S)``.

        Raises:
            KeyError: If a variable is unknown.
            ValueError: If a value is outside its domain.

        References:
            [REF-PEARL-2009, Section 1.1]. No equation is evaluated.

        """
        pairs = self._event(assignment)
        return math.fsum(
            mass
            for key, mass in self.probabilities.items()
            if all(key[position] == value for position, value in pairs)
        )

    def marginal(self, variables: Sequence[str]) -> JointTable:
        """Return the marginal table of a subset of variables.

        Args:
            variables: Non-empty, distinct variables to keep (in this order).

        Returns:
            Marginal joint table.

        Raises:
            KeyError: If a variable is unknown.
            ValueError: If ``variables`` is empty or repeats a name.

        References:
            [REF-PEARL-2009, Section 1.1]. No equation is evaluated.

        """
        names = tuple(variables)
        if not names:
            raise ValueError("marginal needs at least one variable")
        positions = self._index(names)
        sums: dict[Assignment, float] = {}
        for key, mass in self.probabilities.items():
            reduced = tuple(key[position] for position in positions)
            sums[reduced] = sums.get(reduced, 0.0) + mass
        return JointTable(names, {n: self.domains[n] for n in names}, sums)

    def conditional(
        self, target: str | Sequence[str], given: Mapping[str, Hashable]
    ) -> dict[Hashable, float]:
        """Return the conditional law of ``target`` given an observed event.

        Args:
            target: One variable (keys are its values) or a sequence of
                variables (keys are value tuples).
            given: Conditioning event, disjoint from the target.

        Returns:
            ``P(target | given)`` for every target value.

        Raises:
            KeyError: If a variable is unknown.
            ValueError: If the conditioning event has probability zero or
                overlaps the target.

        References:
            [REF-PEARL-2009, Section 1.1]. No equation is evaluated.

        """
        single = isinstance(target, str)
        names: tuple[str, ...] = (target,) if isinstance(target, str) else tuple(target)
        if set(names) & set(given):
            raise ValueError("target and given must be disjoint")
        positions = self._index(names)
        pairs = self._event(given)
        sums: dict[Assignment, float] = {
            values: 0.0 for values in product(*(self.domains[n] for n in names))
        }
        for key, mass in self.probabilities.items():
            if all(key[position] == value for position, value in pairs):
                reduced = tuple(key[position] for position in positions)
                sums[reduced] += mass
        total = math.fsum(sums.values())
        if total <= 0.0:
            raise ValueError(f"conditioning event {dict(given)!r} has probability zero")
        return {
            (values[0] if single else values): mass / total
            for values, mass in sums.items()
        }


@dataclass(frozen=True, slots=True, init=False, eq=False)
class DiscreteCausalModel:
    """A causal Bayesian network on finite variables.

    Attributes:
        graph: Causal graph implied by the parent lists.
        domains: Ordered values of each variable.
        parents: Ordered parent tuple of each variable (table key order).
        tables: For each variable, a map from parent-value tuple to the
            probabilities of the variable's values (in domain order).

    References:
        [REF-PEARL-2009, Sections 1.3 and 3.2].

    """

    graph: CausalGraph
    domains: Mapping[str, tuple[Hashable, ...]]
    parents: Mapping[str, tuple[str, ...]]
    tables: Mapping[str, Mapping[Assignment, tuple[float, ...]]]

    def __init__(
        self,
        domains: Mapping[str, Sequence[Hashable]],
        parents: Mapping[str, Sequence[str]],
        tables: Mapping[str, Mapping[Assignment, Sequence[object]]],
    ) -> None:
        """Validate domains, parent lists, and conditional probability tables.

        Args:
            domains: Ordered values of each variable.
            parents: Parent names of each variable; roots map to ``()``.
            tables: ``tables[v][pa]`` lists ``P(v = value | pa)`` in domain
                order, for every parent-value tuple ``pa`` (``()`` for roots).

        Raises:
            TypeError: If a probability is not real or a name is invalid.
            ValueError: If keys disagree, a table row is missing or extra, a
                row has the wrong length, is negative, or does not sum to one.
            CyclicGraphError: If the parent lists contain a cycle.

        References:
            [REF-PEARL-2009, Section 1.3]. No equation is evaluated.

        """
        checked = _validate_domains(domains)
        if set(parents) != set(checked) or set(tables) != set(checked):
            raise ValueError("domains, parents, and tables must share variable names")
        parent_lists: dict[str, tuple[str, ...]] = {}
        for name in checked:
            listed = tuple(parents[name])
            unknown = [p for p in listed if p not in checked]
            if unknown:
                raise ValueError(f"parents of {name!r} include unknown {unknown}")
            if len(set(listed)) != len(listed):
                raise ValueError(f"parents of {name!r} repeat a name")
            parent_lists[name] = listed
        graph = CausalGraph(
            ((p, name) for name, listed in parent_lists.items() for p in listed),
            checked,
        )
        checked_tables: dict[str, Mapping[Assignment, tuple[float, ...]]] = {}
        for name in checked:
            expected = set(product(*(checked[p] for p in parent_lists[name])))
            given_table = tables[name]
            if set(given_table) != expected:
                missing = sorted(map(repr, expected - set(given_table)))
                extra = sorted(map(repr, set(given_table) - expected))
                raise ValueError(
                    f"table of {name!r} must have one row per parent assignment; "
                    f"missing {missing}, extra {extra}"
                )
            rows: dict[Assignment, tuple[float, ...]] = {}
            for key, row in given_table.items():
                values = tuple(
                    require_non_negative_finite_real(v, name=f"P({name}|{key!r})")
                    for v in row
                )
                if len(values) != len(checked[name]):
                    raise ValueError(
                        f"row {key!r} of {name!r} needs {len(checked[name])} entries"
                    )
                total = math.fsum(values)
                if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
                    raise ValueError(f"row {key!r} of {name!r} sums to {total}")
                rows[key] = values
            checked_tables[name] = MappingProxyType(rows)
        object.__setattr__(self, "graph", graph)
        object.__setattr__(
            self, "domains", MappingProxyType({n: checked[n] for n in graph.nodes})
        )
        object.__setattr__(self, "parents", MappingProxyType(parent_lists))
        object.__setattr__(self, "tables", MappingProxyType(checked_tables))

    def _check_intervention(
        self, intervention: Mapping[str, Hashable]
    ) -> dict[str, Hashable]:
        """Validate an intervention ``do(V = v)``.

        Args:
            intervention: Values imposed on some variables.

        Returns:
            A plain-dict copy.

        Raises:
            KeyError: If a variable is unknown.
            ValueError: If a value is outside its domain.

        References:
            Project validation policy. No equation is evaluated.

        """
        checked: dict[str, Hashable] = {}
        for name, value in intervention.items():
            if name not in self.domains:
                raise KeyError(f"unknown variable {name!r}")
            if value not in self.domains[name]:
                raise ValueError(f"{value!r} is not in the domain of {name!r}")
            checked[name] = value
        return checked

    def _factor(self, name: str, values: Mapping[str, Hashable]) -> float:
        """Return ``P(v | pa(v))`` at a full assignment.

        Args:
            name: Variable.
            values: Full assignment.

        Returns:
            Conditional probability from the table.

        References:
            [REF-PEARL-2009, Section 1.3]. No equation is evaluated.

        """
        key = tuple(values[p] for p in self.parents[name])
        return self.tables[name][key][self.domains[name].index(values[name])]

    def _factorize(self, intervention: Mapping[str, Hashable]) -> JointTable:
        """Enumerate the (truncated) product of conditional tables.

        Args:
            intervention: Already validated intervention (empty for the
                observational law).

        Returns:
            Joint table over all variables in topological order.

        Equation:
            [EQ-CAUSAL-FACT-001], [EQ-CAUSAL-TRUNC-001].

        References:
            [REF-PEARL-2009, Sections 1.2 and 3.2].

        """
        names = self.graph.nodes
        table: dict[Assignment, float] = {}
        for assignment in product(*(self.domains[n] for n in names)):
            values = dict(zip(names, assignment, strict=True))
            if any(values[n] != v for n, v in intervention.items()):
                continue
            mass = 1.0
            for name in names:
                if name not in intervention:
                    mass *= self._factor(name, values)
            table[assignment] = mass
        return JointTable(names, self.domains, table)

    def joint(self) -> JointTable:
        """Return the observational joint law ``prod_i P(v_i | pa_i)``.

        Returns:
            Joint table over all variables in topological order.

        Equation:
            [EQ-CAUSAL-FACT-001].

        References:
            [REF-PEARL-2009, Section 1.2].

        """
        return self._factorize({})

    def interventional_joint(self, intervention: Mapping[str, Hashable]) -> JointTable:
        """Return ``P(v | do(X = x))`` by the truncated factorization.

        Args:
            intervention: Values imposed on the intervened variables ``X``.

        Returns:
            Joint table that is zero off ``X = x`` and equals the product of
            the non-intervened factors on it.

        Raises:
            KeyError: If a variable is unknown.
            ValueError: If a value is outside its domain.

        Equation:
            [EQ-CAUSAL-TRUNC-001].

        References:
            [REF-PEARL-2009, Section 3.2],
            [REF-HERNAN-ROBINS-2020, Chapter 6].

        """
        return self._factorize(self._check_intervention(intervention))

    def interventional_distribution(
        self, target: str | Sequence[str], intervention: Mapping[str, Hashable]
    ) -> dict[Hashable, float]:
        """Return the law of ``target`` under ``do(X = x)``.

        Args:
            target: One variable or a sequence of variables, not intervened.
            intervention: Values imposed on the intervened variables.

        Returns:
            ``P(target | do(X = x))`` keyed like
            :meth:`JointTable.conditional`.

        Raises:
            KeyError: If a variable is unknown.
            ValueError: If the target is intervened or a value is invalid.

        Equation:
            [EQ-CAUSAL-TRUNC-001].

        References:
            [REF-PEARL-2009, Section 3.2].

        """
        names = (target,) if isinstance(target, str) else tuple(target)
        if set(names) & set(intervention):
            raise ValueError("target variables must not be intervened on")
        return self.interventional_joint(intervention).conditional(target, {})

    def intervene(self, intervention: Mapping[str, Hashable]) -> DiscreteCausalModel:
        """Return the mutilated model in which ``X`` is set to ``x``.

        Each intervened variable loses its parents and receives a point-mass
        table at its imposed value; all other tables are unchanged.

        Args:
            intervention: Values imposed on the intervened variables.

        Returns:
            Submodel whose observational law is ``P(v | do(X = x))``.

        Raises:
            KeyError: If a variable is unknown.
            ValueError: If a value is outside its domain.

        References:
            [REF-PEARL-2009, Sections 3.2 and 7.1].

        """
        checked = self._check_intervention(intervention)
        parents: dict[str, Sequence[str]] = dict(self.parents)
        tables: dict[str, Mapping[Assignment, Sequence[object]]] = dict(self.tables)
        for name, value in checked.items():
            parents[name] = ()
            tables[name] = {
                (): tuple(1.0 if v == value else 0.0 for v in self.domains[name])
            }
        return DiscreteCausalModel(self.domains, parents, tables)


def _adjustment_names(
    joint: JointTable, treatment: str, outcome: str, adjustment: Sequence[str]
) -> tuple[str, ...]:
    """Validate treatment, outcome, and covariate names for an adjustment.

    Args:
        joint: Observational table.
        treatment: Treatment variable.
        outcome: Outcome variable.
        adjustment: Covariates.

    Returns:
        Covariate tuple.

    Raises:
        KeyError: If a variable is unknown.
        TypeError: If ``adjustment`` is a bare string.
        ValueError: If names overlap.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(adjustment, str):
        raise TypeError("adjustment must be a sequence of names, not one string")
    names = tuple(adjustment)
    joint._index((treatment, outcome, *names))
    if len({treatment, outcome, *names}) != 2 + len(names):
        raise ValueError("treatment, outcome, and adjustment must be distinct")
    return names


def backdoor_adjustment(
    joint: JointTable,
    treatment: str,
    outcome: str,
    adjustment: Sequence[str],
    treatment_value: Hashable,
) -> dict[Hashable, float]:
    """Return ``P(y | do(x)) = sum_z P(y | x, z) P(z)`` from observational data.

    Valid when ``adjustment`` satisfies the backdoor criterion relative to
    ``(treatment, outcome)``; see :func:`~probstat.causal.satisfies_backdoor`.

    Args:
        joint: Observational joint table containing all named variables.
        treatment: Treatment variable ``X``.
        outcome: Outcome variable ``Y``.
        adjustment: Covariates ``Z`` (may be empty).
        treatment_value: Value ``x``.

    Returns:
        Interventional law of the outcome keyed by outcome value.

    Raises:
        KeyError: If a variable is unknown.
        ValueError: If names overlap, ``x`` is outside its domain, or
            positivity fails (``P(z) > 0`` but ``P(x, z) = 0``).

    Equation:
        [EQ-CAUSAL-ADJ-001].

    References:
        [REF-PEARL-2009, Theorem 3.3.2], [REF-HERNAN-ROBINS-2020, Chapter 2].

    """
    names = _adjustment_names(joint, treatment, outcome, adjustment)
    if treatment_value not in joint.domains[treatment]:
        raise ValueError(f"{treatment_value!r} is not in the domain of {treatment!r}")
    result = dict.fromkeys(joint.domains[outcome], 0.0)
    strata = product(*(joint.domains[n] for n in names))
    for stratum in strata:
        event = dict(zip(names, stratum, strict=True))
        weight = joint.probability(event) if names else 1.0
        if weight <= 0.0:
            continue
        if joint.probability({**event, treatment: treatment_value}) <= 0.0:
            raise ValueError(
                f"positivity fails: P({treatment}={treatment_value!r}, {event}) = 0"
            )
        law = joint.conditional(outcome, {**event, treatment: treatment_value})
        for value, mass in law.items():
            result[value] += weight * mass
    return result


def frontdoor_adjustment(
    joint: JointTable,
    treatment: str,
    outcome: str,
    mediators: Sequence[str],
    treatment_value: Hashable,
) -> dict[Hashable, float]:
    """Return ``sum_m P(m | x) sum_x' P(y | x', m) P(x')`` from observational data.

    Valid when ``mediators`` satisfies the frontdoor criterion; see
    :func:`~probstat.causal.satisfies_frontdoor`.

    Args:
        joint: Observational joint table (latent variables already summed out).
        treatment: Treatment variable ``X``.
        outcome: Outcome variable ``Y``.
        mediators: Non-empty mediator set ``M``.
        treatment_value: Value ``x``.

    Returns:
        Interventional law of the outcome keyed by outcome value.

    Raises:
        KeyError: If a variable is unknown.
        ValueError: If names overlap, ``M`` is empty, ``x`` is invalid, or
            positivity fails (``P(x) = 0``, or ``P(m | x) > 0`` while
            ``P(x', m) = 0`` for some ``x'`` with ``P(x') > 0``).

    Equation:
        [EQ-CAUSAL-FD-001].

    References:
        [REF-PEARL-2009, Theorem 3.3.4].

    """
    names = _adjustment_names(joint, treatment, outcome, mediators)
    if not names:
        raise ValueError("mediators must be non-empty")
    if treatment_value not in joint.domains[treatment]:
        raise ValueError(f"{treatment_value!r} is not in the domain of {treatment!r}")
    if joint.probability({treatment: treatment_value}) <= 0.0:
        raise ValueError(f"positivity fails: P({treatment}={treatment_value!r}) = 0")
    mediator_law = joint.conditional(names, {treatment: treatment_value})
    treatment_law = {
        value: joint.probability({treatment: value})
        for value in joint.domains[treatment]
    }
    result = dict.fromkeys(joint.domains[outcome], 0.0)
    for key, weight in mediator_law.items():
        if weight <= 0.0 or not isinstance(key, tuple):
            continue
        event = dict(zip(names, key, strict=True))
        for other, mass_x in treatment_law.items():
            if mass_x <= 0.0:
                continue
            condition = {**event, treatment: other}
            if joint.probability(condition) <= 0.0:
                raise ValueError(f"positivity fails: P({condition}) = 0")
            law = joint.conditional(outcome, condition)
            for value, mass in law.items():
                result[value] += weight * mass_x * mass
    return result


@dataclass(frozen=True, slots=True)
class AdjustmentComparison:
    """Naive conditional versus backdoor-adjusted outcome probabilities.

    Attributes:
        outcome_value: Outcome value whose probability is reported.
        naive_treated: ``P(Y = y | X = treated)``.
        naive_control: ``P(Y = y | X = control)``.
        adjusted_treated: ``P(Y = y | do(X = treated))`` by adjustment.
        adjusted_control: ``P(Y = y | do(X = control))`` by adjustment.
        naive_difference: ``naive_treated - naive_control``.
        adjusted_difference: ``adjusted_treated - adjusted_control``.
        reversal: Whether the two differences have strictly opposite signs
            (a Simpson reversal).

    References:
        [REF-PEARL-2009, Section 6.1].

    """

    outcome_value: Hashable
    naive_treated: float
    naive_control: float
    adjusted_treated: float
    adjusted_control: float
    naive_difference: float
    adjusted_difference: float
    reversal: bool


def compare_naive_and_adjusted(
    joint: JointTable,
    treatment: str,
    outcome: str,
    adjustment: Sequence[str],
    *,
    outcome_value: Hashable,
    treated: Hashable,
    control: Hashable,
) -> AdjustmentComparison:
    """Contrast naive conditioning with backdoor adjustment for one outcome.

    Args:
        joint: Observational joint table.
        treatment: Treatment variable.
        outcome: Outcome variable.
        adjustment: Backdoor adjustment set.
        outcome_value: Outcome value whose probability is compared.
        treated: Treatment value of the treated arm.
        control: Treatment value of the control arm.

    Returns:
        Both risk differences and a reversal flag.

    Raises:
        KeyError: If a variable is unknown.
        ValueError: If a value is invalid, an arm has probability zero, or
            positivity fails.

    Equation:
        [EQ-CAUSAL-ADJ-001].

    References:
        [REF-PEARL-2009, Section 6.1], [REF-HERNAN-ROBINS-2020, Chapter 2].

    """
    if outcome_value not in joint.domains.get(outcome, ()):
        raise ValueError(f"{outcome_value!r} is not a value of {outcome!r}")
    naive_t = joint.conditional(outcome, {treatment: treated})[outcome_value]
    naive_c = joint.conditional(outcome, {treatment: control})[outcome_value]
    adjusted_t = backdoor_adjustment(joint, treatment, outcome, adjustment, treated)[
        outcome_value
    ]
    adjusted_c = backdoor_adjustment(joint, treatment, outcome, adjustment, control)[
        outcome_value
    ]
    naive_difference = naive_t - naive_c
    adjusted_difference = adjusted_t - adjusted_c
    return AdjustmentComparison(
        outcome_value=outcome_value,
        naive_treated=naive_t,
        naive_control=naive_c,
        adjusted_treated=adjusted_t,
        adjusted_control=adjusted_c,
        naive_difference=naive_difference,
        adjusted_difference=adjusted_difference,
        reversal=naive_difference * adjusted_difference < 0.0,
    )
