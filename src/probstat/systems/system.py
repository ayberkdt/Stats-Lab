"""A mission system: structure, component time models, and common causes.

:class:`SystemModel` ties a :class:`~probstat.systems.SystemStructure` to a
:class:`~probstat.systems.ComponentModel` per component and optional
common-cause failure groups, and answers exactly (by the BDD, never by
sampling) the questions a systems engineer asks:

* mission reliability ``R(t)`` (components non-repairable over the mission);
* point availability ``A(t)`` assuming independent components, each with its
  own repair or test policy;
* ``MTTF = int_0^inf R(t) dt`` by adaptive quadrature with an error
  estimate;
* importance measures and minimal cut sets over the basic events (component
  failures, or independent and CCF events when groups are given);
* a :class:`~probstat.modeling.ProbabilisticModel` whose inputs are uncertain
  component parameters and whose quantities of interest are exact system
  reliabilities or unavailabilities, so ``probstat.workflow.UQStudy`` can
  propagate epistemic uncertainty and rank parameters.

Assumptions: component states are statistically independent except through
the explicit CCF events; CCF groups are symmetric (members share ``Q_t``);
the structure is monotone.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from probstat.foundations.validation import require_probability
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.numerics import integrate
from probstat.uq import QuantityOfInterest

from .ccf import CCFGroup, CommonCauseExpansion, expand_common_cause
from .components import (
    ComponentModel,
    DemandComponent,
    ExponentialComponent,
    FixedProbability,
    PeriodicallyTestedComponent,
    RepairableComponent,
    WeibullComponent,
)
from .importance import ImportanceMeasures, importance_measures
from .structure import Expression, SystemStructure

Measure = Literal["reliability", "availability"]
CCFBasis = Literal["auto", "hazard", "probability"]
_QOI_MEASURES = ("reliability", "unreliability", "availability", "unavailability")


@dataclass(frozen=True, slots=True)
class MTTFResult:
    """Mean time to failure with its quadrature error estimate.

    Args:
        value: ``int_0^inf R(t) dt``.
        error_estimate: Estimated absolute quadrature error.
        converged: Whether the integrator met its tolerance.
        evaluations: Number of system-reliability evaluations.
        rule: Quadrature rule that produced the value.

    References:
        [REF-RAUSAND-HOYLAND-2004], [REF-PIESSENS-QUADPACK-1983].

    """

    value: float
    error_estimate: float
    converged: bool
    evaluations: int
    rule: str


def _require_time(value: object, *, name: str = "t") -> float:
    """Validate a finite non-negative time.

    Raises:
        TypeError: If the value is not real.
        ValueError: If it is negative or not finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return checked


def _default_builder(
    component: str, model: object
) -> Callable[[float], ComponentModel]:
    """Return the map from an uncertain parameter to a component model.

    The varied parameter is ``p`` of :class:`FixedProbability`, the demand
    failure probability of :class:`DemandComponent`, the failure rate of
    exponential, repairable and periodically tested components, and the
    scale of :class:`WeibullComponent`; other parameters stay fixed.

    Raises:
        TypeError: For a model type without a default parameter.

    References:
        Project API policy. No equation is evaluated.

    """
    if isinstance(model, FixedProbability):
        return FixedProbability
    if isinstance(model, DemandComponent):
        return DemandComponent
    if isinstance(model, ExponentialComponent):
        return ExponentialComponent
    if isinstance(model, RepairableComponent):
        repair = model.repair_rate
        return lambda value: RepairableComponent(value, repair)
    if isinstance(model, PeriodicallyTestedComponent):
        interval, first = model.test_interval, model.first_test
        return lambda value: PeriodicallyTestedComponent(value, interval, first)
    if isinstance(model, WeibullComponent):
        shape = model.shape
        return lambda value: WeibullComponent(shape, value)
    raise TypeError(
        f"component {component!r} has no default uncertain parameter; "
        "pass a builder for it"
    )


class SystemModel:
    """A system with component time models and optional CCF groups.

    Args:
        structure: Structure (or expression) over component names.
        components: A :class:`ComponentModel` for every component.
        ccf_groups: Disjoint common-cause groups of components.
        ccf_basis: How the CCF coefficients ``c_k`` split a member's failure
            measure at time ``t``. ``"probability"``: ``Q_k = c_k Q_t(t)``
            (NUREG/CR-5485, natural for demand and unavailability
            probabilities). ``"hazard"``: every event receives the fraction
            ``c_k`` of the member's cumulative hazard
            ``H(t) = -log(1 - Q_t(t))``, ``Q_k = 1 - exp(-c_k H)`` (the
            beta-factor rate split ``lambda_CCF = beta lambda``); a member's
            marginal is then exactly ``Q_t`` at every ``t``. ``"auto"`` (the
            default) uses the hazard split for mission reliability and the
            probability split for availability. The two agree to first order
            in ``Q_t``; the probability split under-states a member's
            failure probability when ``Q_t`` is large (long missions) and
            then gives ``R(t)`` a spurious positive limit.

    Raises:
        KeyError: If a component has no model or a model names an unknown
            component.
        TypeError: If a model does not implement :class:`ComponentModel`.
        ValueError: If the CCF groups are invalid.

    References:
        [REF-RAUSAND-HOYLAND-2004], [REF-RAUZY-1993], [REF-MOSLEH-ETAL-1998].

    """

    def __init__(
        self,
        structure: SystemStructure | Expression | str,
        components: Mapping[str, ComponentModel],
        ccf_groups: Sequence[CCFGroup] = (),
        *,
        ccf_basis: CCFBasis = "auto",
    ) -> None:
        """Validate the models and expand the CCF groups.

        References:
            [REF-MOSLEH-ETAL-1998]. No equation is evaluated.

        """
        base = (
            structure
            if isinstance(structure, SystemStructure)
            else SystemStructure(structure)
        )
        models = dict(components)
        for name in models:
            if name not in base.components:
                raise KeyError(f"model given for unknown component {name!r}")
        for name in base.components:
            if name not in models:
                raise KeyError(f"no model for component {name!r}")
            if not isinstance(models[name], ComponentModel):
                raise TypeError(f"model of {name!r} must implement ComponentModel")
        self._base = base
        self._models: Mapping[str, ComponentModel] = MappingProxyType(models)
        if ccf_basis not in ("auto", "hazard", "probability"):
            raise ValueError("ccf_basis must be 'auto', 'hazard' or 'probability'")
        self._ccf_basis: CCFBasis = ccf_basis
        self._groups = tuple(ccf_groups)
        self._expansion: CommonCauseExpansion | None = None
        if self._groups:
            self._expansion = expand_common_cause(base.expression, self._groups)
            self._structure = SystemStructure(self._expansion.expression)
        else:
            self._structure = base

    @property
    def base_structure(self) -> SystemStructure:
        """Return the structure over components (before CCF expansion).

        References:
            [REF-BARLOW-PROSCHAN-1975].

        """
        return self._base

    @property
    def structure(self) -> SystemStructure:
        """Return the structure over basic events (after CCF expansion).

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        return self._structure

    @property
    def components(self) -> Mapping[str, ComponentModel]:
        """Return the read-only component models.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self._models

    @property
    def basic_events(self) -> tuple[str, ...]:
        """Return the basic-event names in BDD order.

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        return self._structure.components

    @property
    def ccf_expansion(self) -> CommonCauseExpansion | None:
        """Return the CCF expansion, or ``None`` without groups.

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        return self._expansion

    def _tables(
        self, models: Mapping[str, ComponentModel], t: float, measure: Measure
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return basic-event ``(P(up), P(down))`` tables at time ``t``.

        Raises:
            ValueError: If a model returns an invalid probability or members
                of a CCF group disagree on ``Q_t``.

        Equation:
            [EQ-SYS-CCF-005], [EQ-SYS-CCF-006].

        References:
            [REF-MOSLEH-ETAL-1998], [REF-RAUSAND-HOYLAND-2004].

        """
        p: dict[str, float] = {}
        q: dict[str, float] = {}
        totals: dict[int, float] = {}
        group_of = self._expansion.group_of if self._expansion else {}
        for name in self._base.components:
            model = models[name]
            if measure == "reliability":
                up, down = model.reliability(t), model.unreliability(t)
            else:
                up, down = model.availability(t), model.unavailability(t)
            up = require_probability(up, name=f"P({name} up)")
            down = require_probability(down, name=f"P({name} down)")
            if name not in group_of:
                p[name], q[name] = up, down
                continue
            index = group_of[name]
            known = totals.setdefault(index, down)
            if not math.isclose(known, down, rel_tol=1e-9, abs_tol=1e-300):
                raise ValueError(
                    f"members of CCF group {index} need equal failure "
                    f"probabilities; {name!r} has {down}, another member {known}"
                )
        if self._expansion is not None:
            basis = self._ccf_basis
            if basis == "auto":
                basis = "hazard" if measure == "reliability" else "probability"
            for event in self._expansion.events.values():
                total = totals[group_of[event.members[0]]]
                if basis == "probability":
                    down = min(1.0, event.fraction * total)
                    p[event.name], q[event.name] = 1.0 - down, down
                    continue
                # Split the cumulative hazard H = -log(1 - Q_t) of a member.
                if total >= 1.0:
                    p[event.name], q[event.name] = 0.0, 1.0
                    continue
                exponent = event.fraction * math.log1p(-total)
                p[event.name], q[event.name] = math.exp(exponent), -math.expm1(exponent)
        return p, q

    def event_probabilities(
        self, t: float, *, measure: Measure = "reliability"
    ) -> tuple[Mapping[str, float], Mapping[str, float]]:
        """Return basic-event success and failure probabilities at ``t``.

        Args:
            t: Time.
            measure: ``"reliability"`` (mission) or ``"availability"``.

        Returns:
            Read-only ``(P(up), P(down))`` mappings keyed by basic event.

        Equation:
            [EQ-SYS-CCF-005].

        References:
            [REF-MOSLEH-ETAL-1998].

        """
        p, q = self._tables(self._models, _require_time(t), _measure(measure))
        return MappingProxyType(p), MappingProxyType(q)

    def _evaluate(
        self,
        models: Mapping[str, ComponentModel],
        t: float,
        measure: Measure,
        works: bool,
    ) -> float:
        """Return ``P(system up)`` or ``P(system down)`` at ``t``.

        References:
            [EQ-SYS-BDD-001], [REF-RAUZY-1993].

        """
        p, q = self._tables(models, t, measure)
        return self._structure.bdd.probability(self._structure.root, p, q, value=works)

    def reliability(self, t: float) -> float:
        """Return the exact mission reliability ``R(t)``.

        Args:
            t: Mission time.

        Returns:
            Probability that the system works throughout ``[0, t]``.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-RAUZY-1993], [REF-RAUSAND-HOYLAND-2004].

        """
        return self._evaluate(self._models, _require_time(t), "reliability", True)

    def unreliability(self, t: float) -> float:
        """Return the exact mission unreliability ``1 - R(t)`` (direct).

        Args:
            t: Mission time.

        Returns:
            Probability that the system fails in ``[0, t]``.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-RAUZY-1993].

        """
        return self._evaluate(self._models, _require_time(t), "reliability", False)

    def availability(self, t: float) -> float:
        """Return the point availability for independent components.

        Args:
            t: Time.

        Returns:
            ``P(system up at t)``.

        Equation:
            [EQ-SYS-BDD-001], [EQ-SYS-COMP-002].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self._evaluate(self._models, _require_time(t), "availability", True)

    def unavailability(self, t: float) -> float:
        """Return the point unavailability (direct, no cancellation).

        Args:
            t: Time.

        Returns:
            ``P(system down at t)``.

        Equation:
            [EQ-SYS-BDD-001], [EQ-SYS-COMP-002].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self._evaluate(self._models, _require_time(t), "availability", False)

    def mttf(self, *, rel_tol: float = 1e-10, abs_tol: float = 0.0) -> MTTFResult:
        """Return ``MTTF = int_0^inf R(t) dt`` by adaptive quadrature.

        Components are treated as non-repairable (``R`` is the mission
        reliability). The integral uses :func:`probstat.numerics.integrate`
        (double-exponential with a Gauss-Kronrod fallback), whose error
        estimate and convergence flag are reported.

        Args:
            rel_tol: Relative tolerance.
            abs_tol: Absolute tolerance.

        Returns:
            Value, error estimate, and convergence evidence.

        Raises:
            ValueError: If ``R(t)`` does not tend to zero (infinite MTTF).

        Equation:
            [EQ-SYS-MTTF-001].

        References:
            [REF-RAUSAND-HOYLAND-2004], [REF-PIESSENS-QUADPACK-1983],
            [REF-TAKAHASI-MORI-1974].

        """
        try:
            limit = self._evaluate(self._models, math.inf, "reliability", True)
        except (TypeError, ValueError):
            limit = 0.0
        if limit > 0.0:
            raise ValueError(
                f"R(t) tends to {limit} > 0 as t -> inf, so the MTTF is infinite"
            )
        count = 0

        def survival(t: float) -> float:
            """Return ``R(t)`` and count the call.

            References:
                [EQ-SYS-MTTF-001]. No further equation is evaluated.

            """
            nonlocal count
            count += 1
            return self._evaluate(self._models, t, "reliability", True)

        result = integrate(survival, 0.0, math.inf, rel_tol=rel_tol, abs_tol=abs_tol)
        return MTTFResult(
            value=result.value,
            error_estimate=result.error_estimate,
            converged=result.converged,
            evaluations=count,
            rule=result.rule,
        )

    def importance(
        self, t: float, *, measure: Measure = "reliability"
    ) -> Mapping[str, ImportanceMeasures]:
        """Return exact importance measures of every basic event at ``t``.

        Args:
            t: Time.
            measure: ``"reliability"`` (mission) or ``"availability"``.

        Returns:
            Read-only basic event -> :class:`ImportanceMeasures`.

        Equation:
            [EQ-SYS-IMP-001], [EQ-SYS-IMP-002].

        References:
            [REF-BIRNBAUM-1969], [REF-RAUSAND-HOYLAND-2004].

        """
        p, q = self._tables(self._models, _require_time(t), _measure(measure))
        return importance_measures(self._structure, p, unreliabilities=q)

    def minimal_cut_sets(self, *, limit: int = 1_000_000) -> tuple[frozenset[str], ...]:
        """Return the minimal cut sets over basic events.

        Args:
            limit: Enumeration guard.

        Returns:
            Minimal cut sets sorted by size then names.

        Equation:
            [EQ-SYS-MINSOL-001].

        References:
            [REF-RAUZY-1993].

        """
        return self._structure.minimal_cut_sets(limit=limit)

    def as_probabilistic_model(
        self,
        uncertain: Mapping[str, RandomInput],
        times: Sequence[float],
        *,
        measures: Sequence[str] = ("reliability",),
        builders: Mapping[str, Callable[[float], ComponentModel]] | None = None,
        name: str = "system",
    ) -> ProbabilisticModel:
        """Return a probabilistic model with uncertain component parameters.

        Each entry of ``uncertain`` maps a component to the random input that
        drives one of its parameters (see the default builders: ``p`` for
        fixed probabilities, the demand failure probability, failure rates,
        or the Weibull scale); ``builders`` overrides how a parameter value
        becomes a component model. Components mapped to random inputs *with
        the same name* share one value, which models state-of-knowledge
        correlation. The quantities of interest are exact system measures
        ``"<measure>@<t>"`` for every requested time and measure. CCF
        parameters stay fixed.

        Args:
            uncertain: Component -> random input of its parameter.
            times: Non-empty evaluation times.
            measures: Subset of ``reliability``, ``unreliability``,
                ``availability``, ``unavailability``.
            builders: Optional component -> ``value -> ComponentModel``.
            name: Model label.

        Returns:
            A :class:`~probstat.modeling.ProbabilisticModel`.

        Raises:
            KeyError: If a component is unknown.
            TypeError: If an input is not a :class:`RandomInput` or no builder
                exists for a model type.
            ValueError: If times, measures, or shared inputs are invalid.

        Equation:
            [EQ-SYS-BDD-001], [EQ-UQ-001].

        References:
            [REF-SMITH-UQ-2024, Chapters 1-3], [REF-APOSTOLAKIS-KAPLAN-1981].

        """
        chosen_times = tuple(_require_time(t, name="times") for t in times)
        if not chosen_times:
            raise ValueError("at least one time is required")
        chosen_measures = tuple(measures)
        if not chosen_measures:
            raise ValueError("at least one measure is required")
        for measure in chosen_measures:
            if measure not in _QOI_MEASURES:
                raise ValueError(f"measure must be one of {_QOI_MEASURES}")
        inputs: dict[str, RandomInput] = {}
        makers: dict[str, Callable[[float], ComponentModel]] = {}
        for component, random_input in uncertain.items():
            if component not in self._base.components:
                raise KeyError(f"unknown component {component!r}")
            if not isinstance(random_input, RandomInput):
                raise TypeError(f"input of {component!r} must be a RandomInput")
            known = inputs.setdefault(random_input.name, random_input)
            if known != random_input:
                raise ValueError(
                    f"input name {random_input.name!r} is used for different laws"
                )
            given = (builders or {}).get(component)
            makers[component] = (
                given
                if given is not None
                else _default_builder(component, self._models[component])
            )
        for component in builders or {}:
            if component not in uncertain:
                raise KeyError(f"builder given for {component!r} without an input")
        if not inputs:
            raise ValueError("at least one uncertain component is required")
        quantities = []
        keys: list[tuple[str, float, str]] = []
        for t in chosen_times:
            for measure in chosen_measures:
                label = f"{measure}@{t!r}"
                if any(label == key[2] for key in keys):
                    raise ValueError(f"duplicate quantity {label!r}")
                keys.append((measure, t, label))
                quantities.append(
                    QuantityOfInterest(label, f"exact system {measure} at t = {t!r}")
                )
        bindings = {component: uncertain[component].name for component in uncertain}
        system = self

        def forward(parameters: Mapping[str, float]) -> Mapping[str, float]:
            """Evaluate exact system measures for one parameter realization.

            References:
                [EQ-SYS-BDD-001], [EQ-UQ-001].

            """
            models = dict(system._models)
            for component, input_name in bindings.items():
                models[component] = makers[component](parameters[input_name])
            outputs: dict[str, float] = {}
            for measure, t, label in keys:
                kind: Measure = (
                    "reliability"
                    if measure in ("reliability", "unreliability")
                    else "availability"
                )
                works = measure in ("reliability", "availability")
                outputs[label] = system._evaluate(models, t, kind, works)
            return outputs

        return ProbabilisticModel(
            inputs=JointInput.independent(*inputs.values()),
            model=forward,
            quantities=tuple(quantities),
            name=name,
        )


def _measure(value: object) -> Measure:
    """Validate a measure name.

    Raises:
        ValueError: If the measure is unknown.

    References:
        Project validation policy. No equation is evaluated.

    """
    if value == "reliability":
        return "reliability"
    if value == "availability":
        return "availability"
    raise ValueError("measure must be 'reliability' or 'availability'")
