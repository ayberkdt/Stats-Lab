"""Immutable random variables and joint laws on finite probability spaces."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Generic, TypeVar

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
)

from .finite import FiniteEvent, FiniteProbabilityMeasure

OutcomeT = TypeVar("OutcomeT", bound=Hashable)
ValueT = TypeVar("ValueT", bound=Hashable)
TransformedT = TypeVar("TransformedT", bound=Hashable)
OtherT = TypeVar("OtherT", bound=Hashable)


@dataclass(frozen=True, slots=True, init=False)
class FiniteRandomVariable(Generic[OutcomeT, ValueT]):
    """An immutable measurable mapping on a finite probability measure.

    Because the finite sample space uses its full power set, every mapping into
    a finite, hashable codomain is measurable. Numeric methods additionally
    require every mapped value to be a finite real number.

    Args:
        measure: Owning finite probability measure.
        values: Complete outcome-to-value mapping with no extra keys.
        name: Non-empty explanatory label used in plots and derived variables.

    Raises:
        TypeError: If a mapped value is unhashable or ``name`` is not a string.
        ValueError: If mapping keys differ from the sample space or ``name`` is
            empty.

    Equation:
        [EQ-RV-001].

    References:
        [REF-ROSS-2019, Chapters 4-5].

    """

    measure: FiniteProbabilityMeasure[OutcomeT]
    _values: Mapping[OutcomeT, ValueT] = field(repr=False)
    name: str

    def __init__(
        self,
        measure: FiniteProbabilityMeasure[OutcomeT],
        values: Mapping[OutcomeT, ValueT],
        *,
        name: str = "X",
    ) -> None:
        """Validate, copy, and freeze a complete random-variable mapping.

        Args:
            measure: Owning finite probability measure.
            values: Exactly one hashable value per sample-space outcome.
            name: Non-empty explanatory label.

        Raises:
            TypeError: If a mapped value is unhashable or ``name`` is invalid.
            ValueError: If mapping keys do not match the sample space or the
                stripped name is empty.

        Equation:
            [EQ-RV-001].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        checked_name = name.strip()
        if not checked_name:
            raise ValueError("name must not be empty")
        supplied_outcomes = frozenset(values)
        missing = measure.space.outcomes - supplied_outcomes
        extra = supplied_outcomes - measure.space.outcomes
        if missing or extra:
            raise ValueError(
                f"value keys must equal the sample space; missing={missing!r}, "
                f"extra={extra!r}"
            )
        copied: dict[OutcomeT, ValueT] = {}
        for outcome in measure.space:
            mapped = values[outcome]
            try:
                hash(mapped)
            except TypeError as error:
                raise TypeError(
                    f"value for outcome {outcome!r} must be hashable"
                ) from error
            copied[outcome] = mapped
        object.__setattr__(self, "measure", measure)
        object.__setattr__(self, "_values", MappingProxyType(copied))
        object.__setattr__(self, "name", checked_name)

    @classmethod
    def from_function(
        cls,
        measure: FiniteProbabilityMeasure[OutcomeT],
        function: Callable[[OutcomeT], ValueT],
        *,
        name: str = "X",
    ) -> FiniteRandomVariable[OutcomeT, ValueT]:
        """Construct a finite random variable from an outcome function.

        Args:
            measure: Owning finite probability measure.
            function: Deterministic mapping evaluated once for every outcome.
            name: Non-empty explanatory label.

        Returns:
            A validated immutable random variable.

        Raises:
            TypeError: If a result is unhashable or ``name`` is invalid.
            ValueError: If ``name`` is empty.

        Equation:
            [EQ-RV-001].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        values = {outcome: function(outcome) for outcome in measure.space}
        return cls(measure, values, name=name)

    @property
    def values(self) -> Mapping[OutcomeT, ValueT]:
        """Return the read-only outcome-to-value mapping.

        Equation:
            [EQ-RV-001].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        return self._values

    @property
    def support(self) -> tuple[ValueT, ...]:
        """Return distinct positive-or-zero-mass values in encounter order.

        Values reached only through zero-probability outcomes remain in the
        mapping image and therefore in this structural support. Use
        ``positive_support`` when strictly positive mass is required.

        Returns:
            Distinct mapped values ordered by first source-space occurrence.

        Equation:
            [EQ-RV-001].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        ordered: list[ValueT] = []
        seen: set[ValueT] = set()
        for outcome in self.measure.space:
            mapped = self._values[outcome]
            if mapped not in seen:
                seen.add(mapped)
                ordered.append(mapped)
        return tuple(ordered)

    @property
    def positive_support(self) -> tuple[ValueT, ...]:
        """Return mapped values having strictly positive probability mass.

        Returns:
            Structural support values filtered by positive PMF value.

        Equation:
            [EQ-RV-001].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        return tuple(value for value in self.support if self.pmf(value) > 0.0)

    def value_at(self, outcome: OutcomeT) -> ValueT:
        """Return the mapped value at one sample-space outcome.

        Args:
            outcome: Outcome belonging to the owning sample space.

        Returns:
            The deterministic mapped value.

        Raises:
            TypeError: If ``outcome`` is unhashable.
            ValueError: If ``outcome`` is outside the sample space.

        Equation:
            [EQ-RV-001].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        try:
            return self._values[outcome]
        except TypeError as error:
            raise TypeError("outcome must be hashable") from error
        except KeyError as error:
            raise ValueError("outcome must belong to the sample space") from error

    def preimage(self, value: ValueT) -> FiniteEvent[OutcomeT]:
        """Return the event on which the random variable equals ``value``.

        Args:
            value: Candidate codomain value.

        Returns:
            Event ``{omega : X(omega) = value}``, possibly empty.

        Equation:
            [EQ-RV-001].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        members = (
            outcome
            for outcome in self.measure.space
            if self._values[outcome] == value
        )
        return self.measure.space.event(members)

    def event_where(self, predicate: Callable[[ValueT], bool]) -> FiniteEvent[OutcomeT]:
        """Return the event where a value predicate is true.

        Args:
            predicate: Boolean-valued function evaluated once per outcome value.

        Returns:
            Event containing outcomes whose mapped values satisfy ``predicate``.

        Raises:
            TypeError: If a predicate result is not exactly Boolean.

        References:
            [EQ-RV-001], [REF-ROSS-2019, Chapters 4-5].

        """
        members: list[OutcomeT] = []
        for outcome in self.measure.space:
            result = predicate(self._values[outcome])
            if not isinstance(result, bool):
                raise TypeError("predicate must return bool")
            if result:
                members.append(outcome)
        return self.measure.space.event(members)

    def pmf(self, value: ValueT) -> float:
        """Return the induced probability mass at ``value``.

        Args:
            value: Candidate codomain value.

        Returns:
            Probability of the preimage event ``{X = value}``.

        Equation:
            [EQ-RV-001].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        return self.measure.probability(self.preimage(value))

    @property
    def distribution(self) -> Mapping[ValueT, float]:
        """Return the read-only induced PMF over structural support values.

        Equation:
            [EQ-RV-001].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        masses = {value: self.pmf(value) for value in self.support}
        return MappingProxyType(masses)

    def _numeric_values(self) -> Mapping[OutcomeT, float]:
        """Return validated finite real mapped values for numeric operations.

        Returns:
            A read-only outcome-to-float mapping.

        Raises:
            TypeError: If any mapped value is not a real number.
            ValueError: If any mapped value is NaN or infinite.

        References:
            [REF-PYTHON-MATH]. This validator evaluates no statistical equation.

        """
        checked = {
            outcome: require_finite_real(
                self._values[outcome], name=f"{self.name}[{outcome!r}]"
            )
            for outcome in self.measure.space
        }
        return MappingProxyType(checked)

    def cdf(self, x: Real) -> float:
        """Evaluate the CDF of a finite real-valued random variable.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``P(X <= x)``.

        Raises:
            TypeError: If ``x`` or a mapped value is not real.
            ValueError: If ``x`` or a mapped value is non-finite.

        Equation:
            [EQ-RV-003].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        point = require_finite_real(x, name="x")
        numeric = self._numeric_values()
        selected = self.measure.space.event(
            outcome for outcome in self.measure.space if numeric[outcome] <= point
        )
        return self.measure.probability(selected)

    def transform(
        self,
        function: Callable[[ValueT], TransformedT],
        *,
        name: str | None = None,
    ) -> FiniteRandomVariable[OutcomeT, TransformedT]:
        """Return the transformed random variable ``Y = g(X)``.

        Args:
            function: Deterministic value transformation.
            name: Optional derived-variable name; defaults to ``g(<name>)``.

        Returns:
            A new immutable random variable on the same probability measure.

        Raises:
            TypeError: If a transformed value is unhashable or ``name`` is
                invalid.
            ValueError: If ``name`` is empty.

        Equation:
            [EQ-RV-002].

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        transformed = {
            outcome: function(self._values[outcome])
            for outcome in self.measure.space
        }
        derived_name = f"g({self.name})" if name is None else name
        return FiniteRandomVariable(self.measure, transformed, name=derived_name)

    @property
    def expectation(self) -> float:
        """Return the expectation of a finite real-valued random variable.

        Equation:
            [EQ-RV-004].

        References:
            [REF-ROSS-2019, Chapters 4-5].

        """
        numeric = self._numeric_values()
        terms = (
            numeric[outcome] * self.measure.weights[outcome]
            for outcome in self.measure.space
        )
        return math.fsum(terms)

    def raw_moment(self, order: int) -> float:
        """Return the raw moment ``E[X**order]``.

        Args:
            order: Non-negative integer moment order.

        Returns:
            The probability-weighted raw power.

        Raises:
            TypeError: If ``order`` is not an integer or values are not real.
            ValueError: If ``order`` is negative or values are non-finite.

        Equation:
            [EQ-RV-005].

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        checked_order = require_non_negative_integer(order, name="order")
        numeric = self._numeric_values()
        terms = (
            numeric[outcome] ** checked_order * self.measure.weights[outcome]
            for outcome in self.measure.space
        )
        return math.fsum(terms)

    def central_moment(self, order: int) -> float:
        """Return the moment centered at the variable's expectation.

        Args:
            order: Non-negative integer moment order.

        Returns:
            ``E[(X - E[X])**order]``.

        Raises:
            TypeError: If ``order`` is not an integer or values are not real.
            ValueError: If ``order`` is negative or values are non-finite.

        Equation:
            [EQ-RV-006].

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        checked_order = require_non_negative_integer(order, name="order")
        numeric = self._numeric_values()
        center = self.expectation
        terms = (
            (numeric[outcome] - center) ** checked_order
            * self.measure.weights[outcome]
            for outcome in self.measure.space
        )
        return math.fsum(terms)

    @property
    def variance(self) -> float:
        """Return ``E[(X - E[X])**2]``.

        Equation:
            [EQ-RV-007].

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        return self.central_moment(2)

    @property
    def standard_deviation(self) -> float:
        """Return the non-negative square root of variable variance.

        Equation:
            [EQ-RV-007].

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        return math.sqrt(self.variance)

    def joint_with(
        self, other: FiniteRandomVariable[OutcomeT, OtherT]
    ) -> FiniteJointDistribution[OutcomeT, ValueT, OtherT]:
        """Construct the joint distribution with another owned variable.

        Args:
            other: Variable defined on the exact same measure object.

        Returns:
            Immutable joint-law view.

        Raises:
            TypeError: If ``other`` is not a finite random variable.
            ValueError: If the variables do not share measure identity.

        Equation:
            [EQ-RV-010].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        return FiniteJointDistribution(self, other)


@dataclass(frozen=True, slots=True, init=False)
class FiniteJointDistribution(Generic[OutcomeT, ValueT, OtherT]):
    """The induced joint law of two variables on one finite measure.

    Args:
        first: First finite random variable.
        second: Second variable sharing the exact same measure object.

    Raises:
        TypeError: If either argument is not a finite random variable.
        ValueError: If the variables do not share measure identity.

    Equation:
        [EQ-RV-010].

    References:
        [REF-ROSS-2019, Chapter 6].

    """

    first: FiniteRandomVariable[OutcomeT, ValueT]
    second: FiniteRandomVariable[OutcomeT, OtherT]

    def __init__(
        self,
        first: FiniteRandomVariable[OutcomeT, ValueT],
        second: FiniteRandomVariable[OutcomeT, OtherT],
    ) -> None:
        """Validate exact probability-measure ownership for a joint law.

        Args:
            first: First finite random variable.
            second: Second finite random variable.

        Raises:
            TypeError: If either argument is not a finite random variable.
            ValueError: If the measure objects differ.

        Equation:
            [EQ-RV-010].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        if not isinstance(first, FiniteRandomVariable) or not isinstance(
            second, FiniteRandomVariable
        ):
            raise TypeError("joint variables must be FiniteRandomVariable objects")
        if first.measure is not second.measure:
            raise ValueError("joint variables must share the same measure object")
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)

    @property
    def support(self) -> tuple[tuple[ValueT, OtherT], ...]:
        """Return distinct value pairs in source-space encounter order.

        Equation:
            [EQ-RV-010].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        ordered: list[tuple[ValueT, OtherT]] = []
        seen: set[tuple[ValueT, OtherT]] = set()
        for outcome in self.first.measure.space:
            pair = (self.first.values[outcome], self.second.values[outcome])
            if pair not in seen:
                seen.add(pair)
                ordered.append(pair)
        return tuple(ordered)

    def joint_pmf(self, first_value: ValueT, second_value: OtherT) -> float:
        """Return joint mass ``P(X=first_value, Y=second_value)``.

        Args:
            first_value: Candidate value of the first variable.
            second_value: Candidate value of the second variable.

        Returns:
            Probability mass of the joint preimage event.

        Equation:
            [EQ-RV-010].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        event = self.first.measure.space.event(
            outcome
            for outcome in self.first.measure.space
            if self.first.values[outcome] == first_value
            and self.second.values[outcome] == second_value
        )
        return self.first.measure.probability(event)

    @property
    def distribution(self) -> Mapping[tuple[ValueT, OtherT], float]:
        """Return the read-only joint PMF over structural pair support.

        Equation:
            [EQ-RV-010].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        masses = {
            pair: self.joint_pmf(pair[0], pair[1]) for pair in self.support
        }
        return MappingProxyType(masses)

    @property
    def first_marginal(self) -> Mapping[ValueT, float]:
        """Return the first marginal PMF obtained from the joint law.

        Equation:
            [EQ-RV-011].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        masses = {
            first_value: math.fsum(
                self.joint_pmf(first_value, second_value)
                for second_value in self.second.support
            )
            for first_value in self.first.support
        }
        return MappingProxyType(masses)

    @property
    def second_marginal(self) -> Mapping[OtherT, float]:
        """Return the second marginal PMF obtained from the joint law.

        Equation:
            [EQ-RV-011].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        masses = {
            second_value: math.fsum(
                self.joint_pmf(first_value, second_value)
                for first_value in self.first.support
            )
            for second_value in self.second.support
        }
        return MappingProxyType(masses)

    def conditional_pmf_first_given_second(
        self, first_value: ValueT, *, given_second: OtherT
    ) -> float:
        """Return ``P(X=first_value | Y=given_second)``.

        Args:
            first_value: Candidate value of the first variable.
            given_second: Conditioning value of the second variable.

        Returns:
            Conditional point mass of the first variable.

        Raises:
            ValueError: If ``P(Y=given_second) == 0``.

        Equation:
            [EQ-RV-012].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        denominator = self.second.pmf(given_second)
        if denominator == 0.0:
            raise ValueError("conditional PMF requires positive conditioning mass")
        return self.joint_pmf(first_value, given_second) / denominator

    def are_independent(self, *, tolerance: Real = 1e-12) -> bool:
        """Test random-variable independence by joint PMF factorization.

        Args:
            tolerance: Non-negative finite absolute comparison tolerance.

        Returns:
            Whether every pair mass factorizes into its marginal masses.

        Raises:
            TypeError: If ``tolerance`` is not real.
            ValueError: If ``tolerance`` is negative or non-finite.

        Equation:
            [EQ-RV-013].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        checked_tolerance = require_non_negative_finite_real(
            tolerance, name="tolerance"
        )
        for first_value in self.first.support:
            for second_value in self.second.support:
                joint = self.joint_pmf(first_value, second_value)
                product = self.first.pmf(first_value) * self.second.pmf(second_value)
                if not math.isclose(
                    joint, product, rel_tol=0.0, abs_tol=checked_tolerance
                ):
                    return False
        return True

    @property
    def covariance(self) -> float:
        """Return covariance of two finite real-valued random variables.

        Raises:
            TypeError: If either variable has a non-real mapped value.
            ValueError: If either variable has a non-finite mapped value.

        Equation:
            [EQ-RV-008].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        first_numeric = self.first._numeric_values()
        second_numeric = self.second._numeric_values()
        first_mean = self.first.expectation
        second_mean = self.second.expectation
        terms = (
            (first_numeric[outcome] - first_mean)
            * (second_numeric[outcome] - second_mean)
            * self.first.measure.weights[outcome]
            for outcome in self.first.measure.space
        )
        return math.fsum(terms)

    @property
    def correlation(self) -> float:
        """Return the Pearson correlation of two finite random variables.

        Returns:
            Dimensionless linear association in ``[-1, 1]``, up to numerical
            roundoff.

        Raises:
            TypeError: If either variable has a non-real mapped value.
            ValueError: If values are non-finite or either variance is zero.

        Equation:
            [EQ-RV-009].

        References:
            [REF-ROSS-2019, Chapter 6].

        """
        first_spread = self.first.standard_deviation
        second_spread = self.second.standard_deviation
        if first_spread == 0.0 or second_spread == 0.0:
            raise ValueError("correlation requires both variances to be positive")
        result = self.covariance / (first_spread * second_spread)
        return max(-1.0, min(1.0, result))
