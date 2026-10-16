"""Mission assessment: one object for reliability, uncertainty, cost and risk.

A systems engineer describes a mission once — its structure over
subsystems and components (:class:`~probstat.systems.SystemModel`, where a
block may itself be a Markov model or a subsystem), the mission time, the
epistemic uncertainty of component parameters, the life-cycle cost model,
and the project risk register — and :class:`MissionAssessment` answers the
standard questions of a probabilistic risk assessment (NASA/SP-2011-3421)
in one consistent pass:

1. **Mission reliability** (exact, no sampling): ``R(T)``, ``Q(T)``,
   point availability, MTTF with its quadrature error, minimal cut sets.
2. **Importance**: Birnbaum, criticality, Fussell–Vesely, RAW, RRW.
3. **Epistemic uncertainty**: the distribution of ``R(T)`` induced by
   uncertain component parameters (for example posteriors from
   :mod:`probstat.reliability.data`), with order-statistic percentile
   intervals, and Sobol indices that rank which component's data most
   reduce that uncertainty.
4. **Cost**: life-cycle cost, the expected (discounted) cost of mission
   losses, and a cost-risk S-curve when cost elements are uncertain.
5. **Risk**: the project risk register with the mission-loss risk added,
   exact expected exposure, and a ranking.

Every stochastic step draws from a named child stream of one root seed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.cost import (
    CostRiskResult,
    FailureCostItem,
    LifeCycleCostModel,
    UnreliabilityCost,
    cost_of_unreliability,
    life_cycle_cost_risk,
)
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.modeling import RandomInput
from probstat.risk import Risk, RiskRegister
from probstat.systems import ImportanceMeasures, MTTFResult, SystemModel

from .study import UQStudy


def _positive(value: object, *, name: str) -> float:
    """Return a validated positive finite real.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not positive and finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not (number > 0.0 and math.isfinite(number)):
        raise ValueError(f"{name} must be positive and finite")
    return number


def _non_negative(value: object, *, name: str) -> float:
    """Return a validated non-negative finite real.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is negative or not finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not (number >= 0.0 and math.isfinite(number)):
        raise ValueError(f"{name} must be non-negative and finite")
    return number


@dataclass(frozen=True)
class MissionReliability:
    """Exact mission-level reliability measures.

    Args:
        mission_time: Mission duration ``T``.
        reliability: ``R(T)``, probability of no system failure in ``[0, T]``.
        unreliability: ``Q(T) = 1 - R(T)`` computed without cancellation.
        availability: Point availability ``A(T)`` (repairable blocks).
        mttf: Mean time to system failure with its quadrature error.
        minimal_cut_sets: Minimal cut sets, smallest first.
        cut_set_count: Number of minimal cut sets.
        single_points_of_failure: Components forming a cut set alone.

    References:
        [REF-RAUSAND-HOYLAND-2004], [REF-NASA-PRA-2011].

    """

    mission_time: float
    reliability: float
    unreliability: float
    availability: float
    mttf: MTTFResult | None
    minimal_cut_sets: tuple[frozenset[str], ...]
    cut_set_count: int
    single_points_of_failure: tuple[str, ...]


@dataclass(frozen=True)
class MissionUncertainty:
    """Epistemic distribution of the mission reliability.

    Args:
        quantity: QoI label of ``R(T)`` in the probabilistic model.
        samples: Number of parameter realizations propagated.
        mean: Estimated ``E[R(T)]`` over the epistemic law.
        mean_interval: Confidence interval for that mean.
        standard_deviation: Epistemic standard deviation of ``R(T)``.
        percentiles: ``(p, estimate, lower, upper)`` order-statistic
            intervals for the requested percentiles.
        first_order: First-order Sobol index per uncertain input.
        total_effect: Total-effect Sobol index per uncertain input.
        ranking: Inputs ordered by total effect (largest first).
        evaluations: Exact system evaluations used.

    References:
        [REF-APOSTOLAKIS-KAPLAN-1981], [REF-SALTELLI-2010],
        [REF-NASA-PRA-2011].

    """

    quantity: str
    samples: int
    mean: float
    mean_interval: tuple[float, float]
    standard_deviation: float
    percentiles: tuple[tuple[float, float, float, float], ...]
    first_order: Mapping[str, float]
    total_effect: Mapping[str, float]
    ranking: tuple[str, ...]
    evaluations: int


@dataclass(frozen=True)
class MissionCost:
    """Life-cycle cost plus the expected cost of mission losses.

    Args:
        horizon: Programme horizon in years.
        missions_per_year: Mission rate.
        loss_frequency: Expected mission losses per year, ``n Q(T)``.
        probability_of_any_loss: ``1 - R(T)^N`` for ``N`` missions in the
            horizon (independent missions).
        unreliability: Expected and discounted cost of mission losses.
        life_cycle_present_value: Present value of the life-cycle cost model
            at its element means (``0`` when no model is given).
        expected_total_present_value: Sum of the two present values.
        cost_risk: S-curve analysis of the life-cycle cost, when its
            elements are uncertain.

    References:
        [REF-NASA-CEH-2015], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """

    horizon: float
    missions_per_year: float
    loss_frequency: float
    probability_of_any_loss: float
    unreliability: UnreliabilityCost
    life_cycle_present_value: float
    expected_total_present_value: float
    cost_risk: CostRiskResult | None


@dataclass(frozen=True)
class MissionAssessmentResult:
    """Everything one mission assessment produced.

    Args:
        name: Mission label.
        reliability: Exact mission reliability measures.
        importance: Importance measures, most important first (Birnbaum).
        uncertainty: Epistemic analysis, or ``None`` without uncertain inputs.
        cost: Cost analysis.
        register: Risk register including the mission-loss risk.
        root_seed: Root seed of every stochastic step.

    References:
        [REF-NASA-PRA-2011].

    """

    name: str
    reliability: MissionReliability
    importance: tuple[ImportanceMeasures, ...]
    uncertainty: MissionUncertainty | None
    cost: MissionCost
    register: RiskRegister
    root_seed: int


class MissionAssessment:
    """Probabilistic mission assessment over one exact system model.

    Args:
        system: Mission system (blocks may be Markov models or subsystems
            through :class:`~probstat.systems.MarkovComponent` and
            :class:`~probstat.systems.SubsystemComponent`).
        mission_time: Mission duration ``T`` (same time unit as the rates).
        name: Mission label.
        seed: Root seed (integer or :class:`RandomStream`).
        uncertain: Optional component -> random input of one of its
            parameters (epistemic uncertainty, e.g. a Beta or Gamma posterior
            from :mod:`probstat.reliability.data`); inputs sharing a name
            share one value (state-of-knowledge correlation).
        missions_per_year: Mission rate for the cost and risk analysis.
        horizon: Programme horizon in years.
        mission_loss_cost: Consequence cost of one mission loss.
        discount_rate: Annual discount rate for present values.
        life_cycle: Optional life-cycle cost model.
        risks: Optional project risk register.
        compute_mttf: Whether to integrate ``R(t)`` for the MTTF (skip it
            for systems whose reliability never reaches zero).

    References:
        [REF-NASA-PRA-2011], [REF-RAUSAND-HOYLAND-2004], [REF-NASA-CEH-2015].

    """

    def __init__(
        self,
        system: SystemModel,
        mission_time: float,
        *,
        name: str = "mission",
        seed: int | RandomStream = 0,
        uncertain: Mapping[str, RandomInput] | None = None,
        missions_per_year: float = 1.0,
        horizon: float = 1.0,
        mission_loss_cost: float = 0.0,
        discount_rate: float = 0.0,
        life_cycle: LifeCycleCostModel | None = None,
        risks: RiskRegister | None = None,
        compute_mttf: bool = True,
    ) -> None:
        """Validate the mission definition.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a numeric argument is out of range.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(system, SystemModel):
            raise TypeError("system must be a SystemModel")
        if life_cycle is not None and not isinstance(life_cycle, LifeCycleCostModel):
            raise TypeError("life_cycle must be a LifeCycleCostModel")
        if risks is not None and not isinstance(risks, RiskRegister):
            raise TypeError("risks must be a RiskRegister")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be non-empty text")
        self._system = system
        self._time = _positive(mission_time, name="mission_time")
        self._name = name
        self._root = resolve_stream(seed)
        self._uncertain = MappingProxyType(dict(uncertain or {}))
        for component, item in self._uncertain.items():
            if not isinstance(item, RandomInput):
                raise TypeError(f"uncertain[{component!r}] must be a RandomInput")
        self._missions = _non_negative(missions_per_year, name="missions_per_year")
        self._horizon = _positive(horizon, name="horizon")
        self._loss_cost = _non_negative(mission_loss_cost, name="mission_loss_cost")
        self._discount = _non_negative(discount_rate, name="discount_rate")
        self._life_cycle = life_cycle
        self._risks = risks
        self._compute_mttf = bool(compute_mttf)

    @property
    def name(self) -> str:
        """Return the mission label.

        References:
            Project validation policy. No equation is evaluated.

        """
        return self._name

    @property
    def system(self) -> SystemModel:
        """Return the mission system model.

        References:
            [REF-RAUSAND-HOYLAND-2004]. No equation is evaluated.

        """
        return self._system

    @property
    def mission_time(self) -> float:
        """Return the mission duration ``T``.

        References:
            [REF-RAUSAND-HOYLAND-2004]. No equation is evaluated.

        """
        return self._time

    @property
    def root_seed(self) -> int:
        """Return the root seed.

        References:
            [REF-STEELE-LEA-FLOOD-2014]. No equation is evaluated.

        """
        return self._root.seed

    def reliability(self, *, cut_set_limit: int = 100_000) -> MissionReliability:
        """Return exact mission reliability measures (no sampling).

        Args:
            cut_set_limit: Guard on the number of enumerated cut sets.

        Returns:
            :class:`MissionReliability`.

        Equation:
            [EQ-SYS-BDD-001], [EQ-SYS-MINSOL-001].

        References:
            [REF-RAUZY-1993], [REF-RAUSAND-HOYLAND-2004].

        """
        t = self._time
        cuts = self._system.minimal_cut_sets(limit=cut_set_limit)
        mttf = self._system.mttf() if self._compute_mttf else None
        singles = tuple(sorted(next(iter(c)) for c in cuts if len(c) == 1))
        return MissionReliability(
            mission_time=t,
            reliability=self._system.reliability(t),
            unreliability=self._system.unreliability(t),
            availability=self._system.availability(t),
            mttf=mttf,
            minimal_cut_sets=cuts,
            cut_set_count=len(cuts),
            single_points_of_failure=singles,
        )

    def importance(self) -> tuple[ImportanceMeasures, ...]:
        """Return importance measures sorted by Birnbaum importance.

        Returns:
            Measures of every basic event at the mission time.

        Equation:
            [EQ-SYS-IMP-001], [EQ-SYS-IMP-002].

        References:
            [REF-BIRNBAUM-1969], [REF-RAUSAND-HOYLAND-2004].

        """
        table = self._system.importance(self._time)
        return tuple(
            sorted(table.values(), key=lambda m: (-m.birnbaum, m.component))
        )

    def uncertainty(
        self,
        samples: int = 2048,
        *,
        sensitivity_samples: int = 512,
        percentiles: Sequence[float] = (0.05, 0.5, 0.95),
        confidence: float = 0.95,
    ) -> MissionUncertainty:
        """Propagate epistemic parameter uncertainty to ``R(T)``.

        Each realization is evaluated exactly by the system's BDD, so the
        only error is the sampling error over the epistemic law, which the
        returned intervals quantify. Plain Monte Carlo is used for the
        distribution (so order-statistic intervals are valid) and a Sobol
        pick-freeze design for the indices.

        Args:
            samples: Monte Carlo realizations for the distribution.
            sensitivity_samples: Base sample size of the Sobol design.
            percentiles: Percentiles of ``R(T)`` to report.
            confidence: Interval level.

        Returns:
            :class:`MissionUncertainty`.

        Raises:
            ValueError: If no uncertain inputs were given.

        Equation:
            [EQ-UQ-001], [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001].

        References:
            [REF-APOSTOLAKIS-KAPLAN-1981], [REF-SALTELLI-2010].

        """
        if not self._uncertain:
            raise ValueError("no uncertain component parameters were given")
        model = self._system.as_probabilistic_model(
            self._uncertain, (self._time,), name=f"{self._name} reliability"
        )
        quantity = model.quantity_names[0]
        study = UQStudy(model, seed=self._root.child("uncertainty"))
        propagation = study.propagate(samples, design="mc")
        output = propagation[quantity]
        interval = output.mean_interval(confidence)
        rows = []
        for p in percentiles:
            q = output.quantile_interval(float(p), confidence=confidence)
            rows.append((float(p), q.estimate, q.lower, q.upper))
        first: dict[str, float] = {}
        total: dict[str, float] = {}
        ranking: tuple[str, ...] = ()
        if model.dimension > 1:
            sobol = study.sensitivity(quantity, sensitivity_samples)
            for input_name, (s1, st) in sobol.as_mapping().items():
                first[input_name] = s1
                total[input_name] = st
            ranking = sobol.ranking()
        else:
            only = model.inputs.names[0]
            first[only] = total[only] = 1.0
            ranking = (only,)
        return MissionUncertainty(
            quantity=quantity,
            samples=samples,
            mean=output.mean,
            mean_interval=(interval.lower, interval.upper),
            standard_deviation=output.standard_deviation,
            percentiles=tuple(rows),
            first_order=MappingProxyType(first),
            total_effect=MappingProxyType(total),
            ranking=ranking,
            evaluations=study.total_evaluations,
        )

    def cost(
        self,
        *,
        unreliability: float | None = None,
        cost_risk_samples: int = 20_000,
    ) -> MissionCost:
        """Return life-cycle and mission-loss costs.

        Mission losses occur at the expected frequency ``nu = n Q(T)`` per
        year for ``n`` missions per year ([EQ-WF-MISSION-001]); their
        expected discounted cost over the horizon comes from
        :func:`probstat.cost.cost_of_unreliability`. The life-cycle cost model
        is evaluated at its element means (exact expectation, since totals are
        linear in the elements) and, when it has random elements, an S-curve
        cost-risk analysis is attached ([EQ-WF-MISSION-003]).

        Args:
            unreliability: Optional override of ``Q(T)`` (for example the
                epistemic mean); defaults to the exact point value.
            cost_risk_samples: Monte Carlo size of the cost S-curve.

        Returns:
            :class:`MissionCost`.

        Equation:
            [EQ-WF-MISSION-001], [EQ-WF-MISSION-002], [EQ-WF-MISSION-003].

        References:
            [REF-NASA-CEH-2015], [REF-RAUSAND-BARROS-HOYLAND-2021].

        """
        q = (
            self._system.unreliability(self._time)
            if unreliability is None
            else min(1.0, _non_negative(unreliability, name="unreliability"))
        )
        frequency = self._missions * q
        missions = self._missions * self._horizon
        any_loss = -math.expm1(missions * math.log1p(-q)) if q < 1.0 else 1.0
        losses = cost_of_unreliability(
            (FailureCostItem("mission loss", frequency, self._loss_cost),),
            self._horizon,
            discount_rate=self._discount,
        )
        lcc_value = 0.0
        risk_result: CostRiskResult | None = None
        if self._life_cycle is not None:
            lcc_value = self._life_cycle.deterministic().total_present_value
            if self._life_cycle.random_elements:
                risk_result = life_cycle_cost_risk(
                    self._life_cycle,
                    samples=cost_risk_samples,
                    seed=self._root.child("cost-risk"),
                )
        return MissionCost(
            horizon=self._horizon,
            missions_per_year=self._missions,
            loss_frequency=frequency,
            probability_of_any_loss=any_loss,
            unreliability=losses,
            life_cycle_present_value=lcc_value,
            expected_total_present_value=lcc_value + losses.present_value,
            cost_risk=risk_result,
        )

    def register(self) -> RiskRegister:
        """Return the project risk register with the mission-loss risk added.

        The mission-loss risk has probability ``Q(T)`` per mission and impact
        equal to the mission-loss cost; other risks are taken unchanged.

        Returns:
            :class:`~probstat.risk.RiskRegister`.

        Equation:
            [EQ-WF-MISSION-001].

        References:
            [REF-NASA-RMH-2011].

        """
        risks = list(self._risks.risks) if self._risks is not None else []
        risks.append(
            Risk(
                "mission loss",
                self._system.unreliability(self._time),
                self._loss_cost,
                category="technical",
                description="system failure during one mission (exact BDD value)",
            )
        )
        return RiskRegister(tuple(risks))

    def run(self, *, samples: int = 2048, sensitivity_samples: int = 512) -> (
        MissionAssessmentResult
    ):
        """Run every analysis and return one result object.

        Args:
            samples: Epistemic Monte Carlo size (ignored without uncertainty).
            sensitivity_samples: Sobol base size (ignored without uncertainty).

        Returns:
            :class:`MissionAssessmentResult`.

        References:
            [REF-NASA-PRA-2011].

        """
        uncertainty = (
            self.uncertainty(samples, sensitivity_samples=sensitivity_samples)
            if self._uncertain
            else None
        )
        return MissionAssessmentResult(
            name=self._name,
            reliability=self.reliability(),
            importance=self.importance(),
            uncertainty=uncertainty,
            cost=self.cost(),
            register=self.register(),
            root_seed=self._root.seed,
        )

    def report(self, result: MissionAssessmentResult | None = None) -> str:
        """Return a Markdown report of a mission assessment.

        Args:
            result: A previous :meth:`run` result; computed when omitted.

        Returns:
            Markdown text.

        References:
            [REF-NASA-PRA-2011]. No equation is evaluated.

        """
        outcome = self.run() if result is None else result
        return render_mission_report(outcome)


def _fmt(value: float | None, digits: int = 6) -> str:
    """Format a number compactly for a report table.

    References:
        Project documentation policy. No equation is evaluated.

    """
    if value is None:
        return "n/a"
    if value == 0.0 or 1e-3 <= abs(value) < 1e6:
        return f"{value:.{digits}g}"
    return f"{value:.{max(digits - 2, 2)}e}"


def render_mission_report(result: MissionAssessmentResult) -> str:
    """Render a :class:`MissionAssessmentResult` as Markdown.

    Args:
        result: Mission assessment result.

    Returns:
        Markdown report with reliability, importance, uncertainty, cost and
        risk sections.

    References:
        [REF-NASA-PRA-2011]. No equation is evaluated.

    """
    rel = result.reliability
    lines = [
        f"# Mission assessment: {result.name}",
        "",
        f"Root seed {result.root_seed}. Reliability values are exact (binary "
        "decision diagram); only the epistemic section is sampled.",
        "",
        "## Mission reliability",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Mission time T | {_fmt(rel.mission_time)} |",
        f"| Reliability R(T) | {_fmt(rel.reliability, 10)} |",
        f"| Unreliability Q(T) | {_fmt(rel.unreliability, 6)} |",
        f"| Availability A(T) | {_fmt(rel.availability, 10)} |",
    ]
    if rel.mttf is not None:
        lines.append(
            f"| MTTF | {_fmt(rel.mttf.value)} (quadrature error "
            f"{_fmt(rel.mttf.error_estimate, 2)}) |"
        )
    lines.append(f"| Minimal cut sets | {rel.cut_set_count} |")
    spof = ", ".join(rel.single_points_of_failure) or "none"
    lines.append(f"| Single points of failure | {spof} |")
    lines += ["", "Smallest minimal cut sets:", ""]
    for cut in rel.minimal_cut_sets[:10]:
        lines.append(f"- {{{', '.join(sorted(cut))}}}")
    lines += [
        "",
        "## Importance (at T)",
        "",
        "| Basic event | Birnbaum | Criticality | Fussell-Vesely | RAW | RRW |",
        "|---|---|---|---|---|---|",
    ]
    for item in result.importance:
        lines.append(
            f"| {item.component} | {_fmt(item.birnbaum, 4)} | "
            f"{_fmt(item.criticality, 4)} | {_fmt(item.fussell_vesely, 4)} | "
            f"{_fmt(item.risk_achievement_worth, 4)} | "
            f"{_fmt(item.risk_reduction_worth, 4)} |"
        )
    unc = result.uncertainty
    if unc is not None:
        lines += [
            "",
            "## Epistemic uncertainty of R(T)",
            "",
            f"{unc.samples} parameter realizations, each evaluated exactly "
            f"({unc.evaluations} system evaluations in total).",
            "",
            f"- Mean R(T): {_fmt(unc.mean, 8)} (interval "
            f"[{_fmt(unc.mean_interval[0], 8)}, {_fmt(unc.mean_interval[1], 8)}])",
            f"- Epistemic standard deviation: {_fmt(unc.standard_deviation, 4)}",
            "",
            "| Percentile | Estimate | Lower | Upper |",
            "|---|---|---|---|",
        ]
        for p, estimate, lower, upper in unc.percentiles:
            lines.append(
                f"| {p:g} | {_fmt(estimate, 8)} | {_fmt(lower, 8)} | "
                f"{_fmt(upper, 8)} |"
            )
        lines += [
            "",
            "| Uncertain input | First-order | Total effect |",
            "|---|---|---|",
        ]
        for input_name in unc.ranking:
            lines.append(
                f"| {input_name} | {_fmt(unc.first_order[input_name], 3)} | "
                f"{_fmt(unc.total_effect[input_name], 3)} |"
            )
    cost = result.cost
    lines += [
        "",
        "## Cost",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Horizon (years) | {_fmt(cost.horizon)} |",
        f"| Missions per year | {_fmt(cost.missions_per_year)} |",
        f"| Expected mission losses per year | {_fmt(cost.loss_frequency, 4)} |",
        "| P(at least one loss in horizon) | "
        f"{_fmt(cost.probability_of_any_loss, 4)} |",
        f"| Expected loss cost (PV) | {_fmt(cost.unreliability.present_value)} |",
        f"| Life-cycle cost (PV, at means) | {_fmt(cost.life_cycle_present_value)} |",
        f"| Expected total (PV) | {_fmt(cost.expected_total_present_value)} |",
    ]
    if cost.cost_risk is not None:
        lines += [
            "",
            "Life-cycle cost confidence levels (correlated Monte Carlo; analytic "
            "moment-fitted law in brackets):",
            "",
            "| Level | Cost | Contingency |",
            "|---|---|---|",
        ]
        for row in cost.cost_risk.rows:
            lines.append(
                f"| P{round(100 * row.level)} | {_fmt(row.monte_carlo)} "
                f"[{_fmt(row.analytic)}] | {_fmt(row.contingency_monte_carlo)} |"
            )
    register = result.register
    lines += [
        "",
        "## Risk register",
        "",
        f"Expected exposure {_fmt(register.expected_exposure())} "
        f"(standard deviation {_fmt(math.sqrt(register.exposure_variance()))}).",
        "",
        "| Risk | Probability | Expected exposure |",
        "|---|---|---|",
    ]
    for risk in register.ranking():
        lines.append(
            f"| {risk.name} | {_fmt(risk.probability, 4)} | "
            f"{_fmt(risk.expected_exposure())} |"
        )
    return "\n".join(lines) + "\n"
