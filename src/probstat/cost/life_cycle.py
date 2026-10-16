"""Life-cycle cost model: a cost breakdown structure with time profiles.

A :class:`CostElement` is one line of the cost breakdown structure (CBS). Its
value ``X`` is either a fixed amount or an uncertain amount with a continuous
law, and a *time profile* ``((t_1, m_1), (t_2, m_2), ...)`` says when it is
paid: the cash flow at ``t_k`` is ``m_k X``. A one-time cost has one entry
with multiplier 1; an annual recurring cost of ``X`` per year has multiplier
1 (or an escalation factor) in every year; a cost spread over several years
has fractions that sum to one.

Every total is therefore *linear* in the element values,
``LCC = sum_i a_i X_i`` with ``a_i = sum_k m_ik DF(t_ik)``. Two consequences
follow and are used throughout :mod:`probstat.cost`:

1. Evaluating the model at the element means gives the exact expected life
   cycle cost (linearity of expectation) - but *not* any percentile of it.
2. The variance of the total is exactly ``a' Sigma a`` for the covariance
   ``Sigma`` of the element values, which the method of moments in
   :mod:`.cost_risk` uses.

:meth:`LifeCycleCostModel.as_probabilistic_model` turns the CBS into a
:class:`~probstat.modeling.ProbabilisticModel` so that propagation,
sensitivity analysis, and ``UQStudy`` work on costs exactly as on any other
model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations import require_finite_real
from probstat.modeling import (
    ContinuousMarginal,
    JointInput,
    ProbabilisticModel,
    RandomInput,
)
from probstat.uq import QuantityOfInterest

from .discounting import COMPOUNDING, discount_factor
from .moments import method_of_moments_total

COST_CATEGORIES = (
    "research_development",
    "acquisition",
    "operation",
    "maintenance",
    "support",
    "failure",
    "disposal",
    "other",
)

_FRACTION_SUM_TOLERANCE = 1e-9


def _require_text(value: object, *, name: str) -> str:
    """Validate non-empty text.

    Args:
        value: Candidate text.
        name: Field name for error messages.

    Returns:
        The text.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If it is blank.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _is_law(value: object) -> bool:
    """Return whether ``value`` offers the continuous-marginal interface.

    Args:
        value: Candidate cost value or law.

    Returns:
        ``True`` when every method of ``ContinuousMarginal`` is present.

    References:
        Project validation policy. No equation is evaluated.

    """
    return all(
        hasattr(value, attribute)
        for attribute in ("pdf", "cdf", "quantile", "mean", "variance")
    )


@dataclass(frozen=True)
class CostElement:
    """One cost-breakdown-structure element with a payment-time profile.

    Args:
        name: Distinct element name (also the model input name).
        cost: Fixed finite amount, or a continuous law of the uncertain amount
            ``X`` (constant base-year currency).
        profile: Non-empty ``(time, multiplier)`` pairs; the cash flow at
            ``time`` is ``multiplier * X``. Times must be distinct.
        category: One of :data:`COST_CATEGORIES`.
        description: Optional free text.

    References:
        [REF-FABRYCKY-BLANCHARD-1991], [REF-NASA-CEH-2015].

    """

    name: str
    cost: float | ContinuousMarginal
    profile: tuple[tuple[float, float], ...] = ((0.0, 1.0),)
    category: str = "other"
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the name, value, profile, and category.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If the profile is empty or repeats a time, or the
                category is unknown.

        References:
            Project validation policy. No equation is evaluated.

        """
        _require_text(self.name, name="name")
        if not _is_law(self.cost):
            object.__setattr__(
                self, "cost", require_finite_real(self.cost, name="cost")
            )
        pairs = tuple(tuple(pair) for pair in self.profile)
        if not pairs:
            raise ValueError("profile must not be empty")
        checked: list[tuple[float, float]] = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("profile entries must be (time, multiplier) pairs")
            checked.append(
                (
                    require_finite_real(pair[0], name="profile time"),
                    require_finite_real(pair[1], name="profile multiplier"),
                )
            )
        times = [t for t, _ in checked]
        if len(set(times)) != len(times):
            raise ValueError("profile times must be distinct")
        if self.category not in COST_CATEGORIES:
            raise ValueError(f"category must be one of {COST_CATEGORIES}")
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        object.__setattr__(self, "profile", tuple(sorted(checked)))

    @classmethod
    def one_time(
        cls,
        name: str,
        cost: float | ContinuousMarginal,
        time: float = 0.0,
        *,
        category: str = "other",
        description: str = "",
    ) -> CostElement:
        """Build an element paid once at ``time``.

        Args:
            name: Element name.
            cost: Fixed amount or law.
            time: Payment time.
            category: Cost category.
            description: Optional free text.

        Returns:
            Element with profile ``((time, 1),)``.

        References:
            [REF-FABRYCKY-BLANCHARD-1991].

        """
        return cls(name, cost, ((time, 1.0),), category, description)

    @classmethod
    def recurring(
        cls,
        name: str,
        cost: float | ContinuousMarginal,
        start: float,
        end: float,
        *,
        step: float = 1.0,
        escalation: float = 0.0,
        category: str = "other",
        description: str = "",
    ) -> CostElement:
        """Build a cost of ``X`` per period paid at ``start, start + step, ..., end``.

        Args:
            name: Element name.
            cost: Base-year amount per period (fixed or a law).
            start: First payment time.
            end: Last payment time (included when reached on the grid).
            step: Positive spacing of payments.
            escalation: Escalation rate; the multiplier at ``t`` is
                ``(1 + e)^t`` (escalation from the base year ``t = 0``).
            category: Cost category.
            description: Optional free text.

        Returns:
            Element whose cash flow at each payment time is ``(1 + e)^t X``.

        Raises:
            ValueError: If ``end < start`` or ``step`` is not positive.

        Equation:
            [EQ-COST-ESC-001].

        References:
            [REF-FABRYCKY-BLANCHARD-1991], [REF-BLANK-TARQUIN-2012].

        """
        first = require_finite_real(start, name="start")
        last = require_finite_real(end, name="end")
        spacing = require_finite_real(step, name="step")
        rate = require_finite_real(escalation, name="escalation")
        if spacing <= 0.0:
            raise ValueError("step must be positive")
        if last < first:
            raise ValueError("end must not precede start")
        if rate <= -1.0:
            raise ValueError("escalation must be greater than -1")
        count = int(math.floor((last - first) / spacing + 1e-9)) + 1
        times = [first + k * spacing for k in range(count)]
        profile = tuple((t, math.exp(t * math.log1p(rate))) for t in times)
        return cls(name, cost, profile, category, description)

    @classmethod
    def spread(
        cls,
        name: str,
        cost: float | ContinuousMarginal,
        fractions: Mapping[float, float] | Iterable[tuple[float, float]],
        *,
        category: str = "other",
        description: str = "",
    ) -> CostElement:
        """Build a total cost ``X`` phased over time by fractions summing to one.

        Args:
            name: Element name.
            cost: Total amount (fixed or a law).
            fractions: ``time -> fraction`` with non-negative fractions that
                sum to one within ``1e-9``.
            category: Cost category.
            description: Optional free text.

        Returns:
            Element with the fractions as multipliers.

        Raises:
            ValueError: If a fraction is negative or they do not sum to one.

        References:
            [REF-NASA-CEH-2015].

        """
        pairs = tuple(
            fractions.items() if isinstance(fractions, Mapping) else fractions
        )
        values = [require_finite_real(f, name="fraction") for _, f in pairs]
        if any(v < 0.0 for v in values):
            raise ValueError("fractions must be non-negative")
        if abs(math.fsum(values) - 1.0) > _FRACTION_SUM_TOLERANCE:
            raise ValueError("fractions must sum to one")
        return cls(name, cost, tuple(pairs), category, description)

    @property
    def is_random(self) -> bool:
        """Return whether the element value is uncertain.

        References:
            Project validation policy. No equation is evaluated.

        """
        return _is_law(self.cost)

    @property
    def mean(self) -> float:
        """Return the expected element value ``E[X]``.

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        if isinstance(self.cost, float):
            return self.cost
        return float(self.cost.mean)

    @property
    def variance(self) -> float:
        """Return ``Var[X]`` (zero for a fixed amount).

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        if isinstance(self.cost, float):
            return 0.0
        return float(self.cost.variance)

    @property
    def multiplier_sum(self) -> float:
        """Return ``sum_k m_k``, the undiscounted coefficient of ``X``.

        References:
            [REF-FABRYCKY-BLANCHARD-1991].

        """
        return math.fsum(m for _, m in self.profile)

    def present_value_coefficient(
        self, rate: float, *, compounding: str = "discrete"
    ) -> float:
        """Return ``a = sum_k m_k DF(t_k)``, so the element's PV is ``a X``.

        Args:
            rate: Discount rate.
            compounding: ``"discrete"`` or ``"continuous"``.

        Returns:
            Present-value coefficient.

        Equation:
            [EQ-COST-LCC-001], [EQ-COST-DISC-001].

        References:
            [REF-FABRYCKY-BLANCHARD-1991].

        """
        return math.fsum(
            m * discount_factor(rate, t, compounding=compounding)
            for t, m in self.profile
        )

    def cash_flows(self, value: float | None = None) -> tuple[tuple[float, float], ...]:
        """Return ``(time, amount)`` cash flows for one element value.

        Args:
            value: Element value; defaults to the mean.

        Returns:
            Cash flows ``(t_k, m_k X)``.

        Equation:
            [EQ-COST-LCC-001].

        References:
            [REF-FABRYCKY-BLANCHARD-1991].

        """
        x = self.mean if value is None else require_finite_real(value, name="value")
        return tuple((t, m * x) for t, m in self.profile)


@dataclass(frozen=True)
class CostBreakdown:
    """Life-cycle cost totals of one realization of the element values.

    Args:
        values: Element values used.
        total_present_value: ``sum_i a_i X_i``.
        total_undiscounted: ``sum_i (sum_k m_ik) X_i``.
        by_category: Category to present value.
        by_category_undiscounted: Category to undiscounted amount.
        by_time: Payment time to undiscounted amount (all elements).
        by_time_present_value: Payment time to present value.
        by_element: Element to present value.

    References:
        [REF-FABRYCKY-BLANCHARD-1991].

    """

    values: Mapping[str, float]
    total_present_value: float
    total_undiscounted: float
    by_category: Mapping[str, float]
    by_category_undiscounted: Mapping[str, float]
    by_time: Mapping[float, float]
    by_time_present_value: Mapping[float, float]
    by_element: Mapping[str, float]


@dataclass(frozen=True)
class LifeCycleCostModel:
    """A cost breakdown structure with discounting.

    Args:
        elements: Distinctly named cost elements.
        discount_rate: Discount rate (real for constant-currency elements).
        compounding: ``"discrete"`` or ``"continuous"``.
        name: Short label.

    References:
        [REF-FABRYCKY-BLANCHARD-1991], [REF-NASA-CEH-2015].

    """

    elements: tuple[CostElement, ...]
    discount_rate: float = 0.0
    compounding: str = "discrete"
    name: str = "life cycle cost"

    def __post_init__(self) -> None:
        """Validate elements and the discounting convention.

        Raises:
            TypeError: If an element has the wrong type.
            ValueError: If there are no elements, names repeat, or the rate
                is invalid.

        References:
            Project validation policy. No equation is evaluated.

        """
        elements = tuple(self.elements)
        if not elements:
            raise ValueError("at least one cost element is required")
        for element in elements:
            if not isinstance(element, CostElement):
                raise TypeError("elements must be CostElement instances")
        names = [element.name for element in elements]
        if len(set(names)) != len(names):
            raise ValueError("element names must be distinct")
        if self.compounding not in COMPOUNDING:
            raise ValueError("compounding must be 'discrete' or 'continuous'")
        discount_factor(self.discount_rate, 0.0, compounding=self.compounding)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(
            self, "discount_rate", require_finite_real(self.discount_rate, name="rate")
        )
        _require_text(self.name, name="name")

    @property
    def names(self) -> tuple[str, ...]:
        """Return element names in declaration order.

        References:
            Project validation policy. No equation is evaluated.

        """
        return tuple(element.name for element in self.elements)

    @property
    def random_elements(self) -> tuple[CostElement, ...]:
        """Return the elements with an uncertain value.

        References:
            Project validation policy. No equation is evaluated.

        """
        return tuple(element for element in self.elements if element.is_random)

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the categories used, in :data:`COST_CATEGORIES` order.

        References:
            Project validation policy. No equation is evaluated.

        """
        used = {element.category for element in self.elements}
        return tuple(c for c in COST_CATEGORIES if c in used)

    def element(self, name: str) -> CostElement:
        """Return the element with a given name.

        Args:
            name: Element name.

        Returns:
            Matching element.

        Raises:
            KeyError: If no element has that name.

        References:
            Project validation policy. No equation is evaluated.

        """
        for element in self.elements:
            if element.name == name:
                return element
        raise KeyError(f"unknown cost element {name!r}")

    def coefficients(self) -> Mapping[str, float]:
        """Return the present-value coefficient ``a_i`` of every element.

        Equation:
            [EQ-COST-LCC-001].

        References:
            [REF-FABRYCKY-BLANCHARD-1991].

        """
        return MappingProxyType(
            {
                element.name: element.present_value_coefficient(
                    self.discount_rate, compounding=self.compounding
                )
                for element in self.elements
            }
        )

    def evaluate(self, values: Mapping[str, float] | None = None) -> CostBreakdown:
        """Return all totals for one realization of the element values.

        Args:
            values: Values of some or all elements; missing random elements
                take their mean and fixed elements their fixed amount. A value
                given for a fixed element overrides it.

        Returns:
            Cost breakdown with totals by category, time, and element.

        Raises:
            KeyError: If ``values`` names an unknown element.

        Equation:
            [EQ-COST-LCC-001], [EQ-COST-DISC-001].

        References:
            [REF-FABRYCKY-BLANCHARD-1991], [REF-BLANK-TARQUIN-2012].

        """
        supplied = dict(values or {})
        unknown = set(supplied) - set(self.names)
        if unknown:
            raise KeyError(f"unknown cost elements: {sorted(unknown)}")
        realized = {
            element.name: (
                require_finite_real(supplied[element.name], name=element.name)
                if element.name in supplied
                else element.mean
            )
            for element in self.elements
        }
        category_pv: dict[str, list[float]] = {}
        category_raw: dict[str, list[float]] = {}
        time_raw: dict[float, list[float]] = {}
        time_pv: dict[float, list[float]] = {}
        element_pv: dict[str, float] = {}
        for element in self.elements:
            pieces: list[float] = []
            for t, amount in element.cash_flows(realized[element.name]):
                pv = amount * discount_factor(
                    self.discount_rate, t, compounding=self.compounding
                )
                pieces.append(pv)
                time_raw.setdefault(t, []).append(amount)
                time_pv.setdefault(t, []).append(pv)
                category_raw.setdefault(element.category, []).append(amount)
            element_pv[element.name] = math.fsum(pieces)
            category_pv.setdefault(element.category, []).append(
                element_pv[element.name]
            )
        order = self.categories
        return CostBreakdown(
            MappingProxyType(realized),
            math.fsum(element_pv.values()),
            math.fsum(v for parts in time_raw.values() for v in parts),
            MappingProxyType({c: math.fsum(category_pv[c]) for c in order}),
            MappingProxyType({c: math.fsum(category_raw[c]) for c in order}),
            MappingProxyType({t: math.fsum(time_raw[t]) for t in sorted(time_raw)}),
            MappingProxyType({t: math.fsum(time_pv[t]) for t in sorted(time_pv)}),
            MappingProxyType(element_pv),
        )

    def deterministic(self) -> CostBreakdown:
        """Return the breakdown at the element means.

        Because every total is linear in the element values this equals the
        *expected* life-cycle cost exactly; it is not a percentile.

        Equation:
            [EQ-COST-LCC-001].

        References:
            [REF-FABRYCKY-BLANCHARD-1991].

        """
        return self.evaluate(None)

    def moments(
        self, correlation: Sequence[Sequence[float]] | None = None
    ) -> tuple[float, float]:
        """Return the exact mean and variance of the present-value total.

        Args:
            correlation: Optional Pearson correlation matrix of the *random*
                element values (in :attr:`random_elements` order); ``None``
                means independence.

        Returns:
            ``(sum a_i mu_i, sum_ij a_i a_j rho_ij sigma_i sigma_j)``.

        Equation:
            [EQ-COST-LCC-001], [EQ-COST-MOM-001].

        References:
            [REF-GARVEY-2000], [REF-NASA-CEH-2015].

        """
        coefficients = self.coefficients()
        constant = math.fsum(
            coefficients[e.name] * e.mean for e in self.elements if not e.is_random
        )
        randoms = self.random_elements
        if not randoms:
            return constant, 0.0
        total = method_of_moments_total(
            [e.mean for e in randoms],
            [math.sqrt(e.variance) for e in randoms],
            correlation,
            weights=[coefficients[e.name] for e in randoms],
            constant=constant,
        )
        return total.mean, total.variance

    def joint_input(
        self,
        correlation: Sequence[Sequence[float]] | None = None,
        *,
        kind: str = "pearson",
    ) -> JointInput:
        """Return the joint law of the random element values.

        Args:
            correlation: Optional correlation matrix in
                :attr:`random_elements` order.
            kind: ``"pearson"`` (physical correlation, matched through the
                Nataf transformation) or ``"latent"`` (Gaussian-copula
                normal-score correlation used as is).

        Returns:
            Joint input with one :class:`~probstat.modeling.RandomInput` per
            random element.

        Raises:
            ValueError: If no element is random or ``kind`` is unknown.

        Equation:
            [EQ-MOD-NATAF-002].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-NASA-CEH-2015].

        """
        randoms = self.random_elements
        if not randoms:
            raise ValueError("the model has no random cost elements")
        inputs = tuple(
            RandomInput(e.name, e.cost, e.description, None)  # type: ignore[arg-type]
            for e in randoms
        )
        if correlation is None:
            return JointInput(inputs)
        if kind == "pearson":
            return JointInput.from_pearson(inputs, correlation)
        if kind == "latent":
            return JointInput(
                inputs, tuple(tuple(float(v) for v in row) for row in correlation)
            )
        raise ValueError("kind must be 'pearson' or 'latent'")

    def as_probabilistic_model(
        self,
        correlation: Sequence[Sequence[float]] | None = None,
        *,
        kind: str = "pearson",
    ) -> ProbabilisticModel:
        """Return the CBS as a probabilistic model for propagation and sensitivity.

        The forward model returns ``lcc_present_value``, ``lcc_undiscounted``
        and ``pv_<category>`` for every category used. Fixed elements are
        passed as fixed parameters.

        Args:
            correlation: Optional correlation of the random elements.
            kind: ``"pearson"`` or ``"latent"`` (see :meth:`joint_input`).

        Returns:
            Probabilistic model whose evaluations are cost breakdowns.

        Equation:
            [EQ-COST-LCC-001], [EQ-UQ-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3], [REF-FABRYCKY-BLANCHARD-1991].

        """
        joint = self.joint_input(correlation, kind=kind)
        categories = self.categories
        fixed = {e.name: e.mean for e in self.elements if not e.is_random}

        def forward(parameters: Mapping[str, float]) -> Mapping[str, float]:
            """Evaluate the life-cycle cost totals of one realization.

            References:
                [EQ-COST-LCC-001], [REF-FABRYCKY-BLANCHARD-1991].

            """
            breakdown = self.evaluate({n: parameters[n] for n in self.names})
            outputs = {
                "lcc_present_value": breakdown.total_present_value,
                "lcc_undiscounted": breakdown.total_undiscounted,
            }
            for category in categories:
                outputs[f"pv_{category}"] = breakdown.by_category[category]
            return outputs

        quantities = [
            QuantityOfInterest("lcc_present_value", "present value of life-cycle cost"),
            QuantityOfInterest("lcc_undiscounted", "undiscounted life-cycle cost"),
        ] + [
            QuantityOfInterest(f"pv_{c}", f"present value of {c} cost")
            for c in categories
        ]
        return ProbabilisticModel(joint, forward, tuple(quantities), self.name, fixed)
