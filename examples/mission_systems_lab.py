"""Mission systems laboratory: a lunar lander from component data to cost.

One mission, analysed the way a systems engineer would, almost entirely
without Monte Carlo:

1. Component data -> Bayesian posteriors and classical bounds.
2. Subsystems: a cold-standby pump pair (continuous-time Markov chain), a
   2-out-of-3 flight computer with beta-factor common-cause failure, a
   battery pair with Weibull lives, a transponder, and an igniter tested on
   demand.
3. Exact mission reliability, MTTF, cut sets and importance (binary decision
   diagram), and a three-phase mission (launch, cruise, landing).
4. A discrete-time Markov stage model of the mission timeline with retries
   and aborts (absorbing outcomes).
5. Epistemic uncertainty of the reliability from the data posteriors, with
   Sobol ranking of which data matter.
6. Reliability allocation, FMECA criticality, life-cycle cost with a cost
   S-curve, the cost of mission losses, and the risk register.

Every number is reproducible from the root seed. The Markdown report is
written to ``examples/generated/mission_systems_report.md``; with Matplotlib
installed, the Sobol indices of the epistemic analysis are drawn to
``examples/generated/mission-systems-sobol.png``.

Definitions:
    ``R(T)``: probability that the lander structure function is 1 over the
        mission time ``T = 720`` h.
    structure function: series of the pump pair, a 2-out-of-3 computer
        group, a parallel battery pair, the transponder, and the igniter.
    failure rate ``lambda`` and demand probability ``p``: component
        parameters with Jeffreys posteriors from test counts.
    beta factor: share of a component's failure rate that fails the whole
        common-cause group at once.
    phase: an interval of the mission with its own structure function;
        components age across phases.
    importance: Birnbaum, criticality, Fussell-Vesely, RAW, and RRW
        measures of each block.
    epistemic uncertainty: the spread of ``R(T)`` induced by the posterior
        laws of the uncertain rates.
    FMECA criticality ``C_r``: MIL-STD-1629A item criticality number.

Assumptions:
    - Components fail independently except through the beta-factor group;
      the three computers share one uncertain rate (state-of-knowledge
      correlation).
    - Components are non-repairable within the mission; the warm-standby
      pump pair is a continuous-time Markov chain with 99% switching.
    - Exponential lives have constant hazards; batteries are Weibull with
      shape 1.5.
    - Mission stages succeed, retry, or abort with fixed probabilities and
      are independent of the hardware model.
    - Costs follow the declared triangular and recurring elements,
      discounted at 3% per year; missions are independent.

Equations:
    EQ-REL-DATA-001: Beta posterior for failure on demand.
    EQ-REL-DATA-002: Gamma (Jeffreys) posterior for a failure rate.
    EQ-REL-DATA-003: chi-square bounds on a failure rate.
    EQ-INF-CI-CP-001: Clopper-Pearson interval for the igniter.
    EQ-SYS-COMP-001: non-repairable component reliability.
    EQ-SYS-MKV-001: reliability from a Markov model.
    EQ-SYS-MKV-007: standby closed forms.
    EQ-SYS-STRUCT-001: series and parallel structures.
    EQ-SYS-STRUCT-002: k-out-of-n structure.
    EQ-SYS-CCF-001: beta-factor common-cause model.
    EQ-SYS-BDD-001: exact system probability by Shannon expansion on a BDD.
    EQ-SYS-MINSOL-001: minimal cut sets.
    EQ-SYS-IMP-001: Birnbaum and criticality importance.
    EQ-SYS-IMP-002: Fussell-Vesely, RAW, and RRW.
    EQ-SYS-PMS-001: phased-mission reliability.
    EQ-SYS-MKPM-002: discrete mission stages with absorbing outcomes.
    EQ-REL-ALLOC-002: ARINC reliability allocation.
    EQ-RISK-FMECA-002: item criticality number.
    EQ-SENS-FIRST-001: first-order Sobol index of the uncertain data.
    EQ-SENS-TOTAL-001: total-effect Sobol index of the uncertain data.
    EQ-COST-DISC-001: discount factor.
    EQ-COST-LCC-001: linear life-cycle cost.
    EQ-WF-MISSION-001: expected mission-loss frequency.

Diagnostics:
    - Classical chi-square bounds and Clopper-Pearson intervals are printed
      beside the Bayesian posterior means, so a reader sees how much the
      data constrain each parameter.
    - Phase-by-phase cumulative and conditional reliabilities localize where
      the mission is lost (cruise dominates).
    - The timeline probabilities of success, abort, and loss sum to one.
    - The 5-95% epistemic band of ``R(T)`` and the Sobol ranking say whether
      more computer test hours or more igniter firings would sharpen the
      estimate.

Cost and error:
    - Reliability, MTTF, phased-mission, timeline, allocation, and FMECA
      numbers are exact (no sampling error).
    - The epistemic analysis propagates 2,048 parameter realizations and
      estimates Sobol indices from a base sample of 512 through the exact
      system model; its percentiles and indices carry Monte Carlo error of
      order ``N^(-1/2)``.

References:
    REF-ATWOOD-NUREG-6823-2003: reliability data and posteriors.
    REF-CLOPPER-PEARSON-1934: exact binomial interval.
    REF-RAUSAND-HOYLAND-2004: system reliability theory.
    REF-BARLOW-PROSCHAN-1975: structure functions.
    REF-MOSLEH-ETAL-1998: common-cause failure models.
    REF-RAUZY-1993: binary decision diagrams and minimal solutions.
    REF-BIRNBAUM-1969: component importance.
    REF-ZANG-SUN-TRIVEDI-1999: phased-mission systems.
    REF-KEMENY-SNELL-1976: absorbing Markov chains.
    REF-MIL-HDBK-338B-1998: reliability allocation.
    REF-MIL-STD-1629A-1980: FMECA.
    REF-SALTELLI-2010: Sobol indices.
    REF-NASA-CEH-2015: life-cycle cost.
    REF-NASA-PRA-2011: probabilistic risk assessment.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from probstat.cost import CostElement, LifeCycleCostModel
from probstat.distributions import Triangular
from probstat.reliability import (
    arinc_allocation,
    demand_failure_probability,
    failure_rate_bounds,
    failure_rate_estimate,
)
from probstat.risk import FailureMode, Fmeca, Risk, RiskRegister
from probstat.systems import (
    BetaFactorGroup,
    DemandComponent,
    ExponentialComponent,
    MarkovComponent,
    MissionStage,
    MissionStageModel,
    Phase,
    PhasedMission,
    SystemModel,
    WeibullComponent,
    k_out_of_n,
    parallel,
    series,
    standby_system,
)
from probstat.visualization import save_figure
from probstat.visualization.theme import matplotlib_available
from probstat.visualization.uq import plot_sobol_indices
from probstat.workflow import MissionAssessment, MissionAssessmentResult

OUTPUT = Path(__file__).parent / "generated"
MISSION_HOURS = 720.0
# Root seed of the epistemic propagation and Sobol analysis.
SEED = 2026


def component_data() -> dict[str, object]:
    """Estimate component parameters from test and field data.

    Returns:
        Posterior estimates keyed by component family.

    """
    computer = failure_rate_estimate(3, 1.2e6)  # 3 failures in 1.2 million h
    pump = failure_rate_estimate(4, 4.0e4)
    igniter = demand_failure_probability(1, 400)  # 1 failure in 400 firings
    bounds = failure_rate_bounds(3, 1.2e6, confidence=0.9)
    print("Component data")
    print(
        f"  computer rate: MLE {computer.maximum_likelihood:.3e}/h, Jeffreys "
        f"posterior mean {computer.posterior.mean:.3e}/h, 90% chi-square bounds "
        f"[{bounds.lower:.3e}, {bounds.upper:.3e}]"
    )
    print(f"  pump rate posterior mean {pump.posterior.mean:.3e}/h")
    interval = igniter.confidence_interval
    print(
        f"  igniter demand failure: posterior mean {igniter.posterior.mean:.4f}, "
        f"90% Clopper-Pearson [{interval.lower:.5f}, {interval.upper:.5f}]"
    )
    return {"computer": computer, "pump": pump, "igniter": igniter}


def lander(data: dict[str, object]) -> SystemModel:
    """Build the lander system from its subsystems.

    Args:
        data: Posterior estimates from :func:`component_data`.

    Returns:
        Exact system model.

    """
    computer_rate = data["computer"].posterior.mean  # type: ignore[attr-defined]
    pump_rate = data["pump"].posterior.mean  # type: ignore[attr-defined]
    igniter_p = data["igniter"].posterior.mean  # type: ignore[attr-defined]
    pumps = MarkovComponent(
        standby_system(
            spares=1,
            failure_rate=pump_rate,
            standby_failure_rate=0.1 * pump_rate,  # warm standby
            switch_success_probability=0.99,
        )
    )
    structure = series(
        "pumps",
        k_out_of_n(2, "cpu1", "cpu2", "cpu3"),
        parallel("battery1", "battery2"),
        "transponder",
        "igniter",
    )
    components = {
        "pumps": pumps,
        "cpu1": ExponentialComponent(computer_rate),
        "cpu2": ExponentialComponent(computer_rate),
        "cpu3": ExponentialComponent(computer_rate),
        "battery1": WeibullComponent(1.5, 20_000.0),
        "battery2": WeibullComponent(1.5, 20_000.0),
        "transponder": ExponentialComponent(1e-4),
        "igniter": DemandComponent(igniter_p),
    }
    return SystemModel(
        structure,
        components,
        ccf_groups=(BetaFactorGroup(("cpu1", "cpu2", "cpu3"), 0.05),),
    )


def phased_mission(data: dict[str, object]) -> None:
    """Analyse launch, cruise and landing as a phased mission.

    Args:
        data: Posterior estimates from :func:`component_data`.

    """
    rate = data["computer"].posterior.mean  # type: ignore[attr-defined]
    components = {
        "cpu1": ExponentialComponent(rate),
        "cpu2": ExponentialComponent(rate),
        "cpu3": ExponentialComponent(rate),
        # Non-repairable components age in every phase. The engine and the
        # radar are dormant during cruise, so they get a much lower dormant
        # failure rate there (one model per phase); with the active rate in
        # all phases the 717 h cruise alone would cost about 40% reliability.
        "engine": (
            ExponentialComponent(5e-4),
            ExponentialComponent(5e-6),
            ExponentialComponent(5e-4),
        ),
        "radar": (
            ExponentialComponent(2e-4),
            ExponentialComponent(2e-6),
            ExponentialComponent(2e-4),
        ),
        "transponder": ExponentialComponent(1e-4),
    }
    mission = PhasedMission(
        (
            Phase(
                "launch", 2.0, series("engine", k_out_of_n(3, "cpu1", "cpu2", "cpu3"))
            ),
            Phase(
                "cruise",
                717.0,
                series(k_out_of_n(1, "cpu1", "cpu2", "cpu3"), "transponder"),
            ),
            Phase(
                "landing",
                1.0,
                series("engine", "radar", k_out_of_n(2, "cpu1", "cpu2", "cpu3")),
            ),
        ),
        components,
    )
    result = mission.evaluate()
    print("\nPhased mission (exact, non-repairable)")
    for name, cumulative, conditional, failure in zip(
        result.phase_names,
        result.cumulative_reliability,
        result.conditional_reliability,
        result.phase_failure_probability,
        strict=True,
    ):
        print(
            f"  {name:<8} cumulative {cumulative:.6f}  conditional "
            f"{conditional:.6f}  P(fail here) {failure:.3e}"
        )
    print(f"  mission success {result.reliability:.6f}")


def stage_model() -> None:
    """Solve the mission timeline as an absorbing discrete-time Markov chain."""
    stages = MissionStageModel(
        (
            MissionStage("launch", success=0.990, retry=0.005, abort=0.003),
            MissionStage("trans-lunar injection", success=0.995, retry=0.004),
            MissionStage("descent", success=0.975, abort=0.020),
            MissionStage("surface operations", success=0.990, retry=0.008),
        )
    )
    outcome = stages.outcome()
    print("\nMission timeline (absorbing Markov chain, exact)")
    print(f"  P(success) {outcome.success_probability:.6f}")
    print(f"  P(abort)   {outcome.other_probability:.6f}")
    print(f"  P(loss)    {outcome.failure_probability:.6f}")
    print(f"  expected stage attempts {outcome.expected_steps:.4f}")


def allocation_and_fmeca() -> None:
    """Allocate a reliability target and rank failure modes."""
    allocation = arinc_allocation(
        0.995,
        (2e-5, 4e-5, 1e-5, 1e-4, 5e-6),
        MISSION_HOURS,
        names=("pumps", "computers", "power", "transponder", "igniter"),
    )
    print("\nARINC allocation for a 0.995 mission target")
    for name, value in zip(allocation.names, allocation.reliabilities, strict=True):
        print(f"  {name:<12} {value:.6f}")
    fmeca = Fmeca(
        (
            FailureMode("pump", "seal leak", 2, 2e-4, 0.6, 0.5, MISSION_HOURS),
            FailureMode("pump", "bearing seizure", 1, 2e-4, 0.4, 1.0, MISSION_HOURS),
            FailureMode("computer", "latch-up", 2, 2.5e-6, 1.0, 0.3, MISSION_HOURS),
            FailureMode("igniter", "no fire", 1, 3e-3, 1.0, 1.0, 1.0),
        )
    )
    print("\nFMECA item criticality (MIL-STD-1629A)")
    for row in fmeca.item_criticality():
        print(
            f"  {row.item:<9} severity {row.severity_class}  C_r = "
            f"{row.criticality:.3e}  ({', '.join(row.modes)})"
        )


def sobol_figure(result: MissionAssessmentResult) -> Any | None:
    """Draw which uncertain data drive the epistemic spread of ``R(T)``.

    Args:
        result: Completed mission assessment.

    Returns:
        Axes of the Sobol dumbbell chart, or ``None`` without uncertainty.

    """
    uncertainty = result.uncertainty
    if uncertainty is None:
        return None
    names = tuple(uncertainty.ranking)
    totals = [uncertainty.total_effect[name] for name in names]
    return plot_sobol_indices(
        names,
        [uncertainty.first_order[name] for name in names],
        totals,
        title=f"Input {names[0]} drives the spread of R(T) (S_T = {totals[0]:.2f})",
        subtitle="Sobol indices of R(T) under the posterior laws of the uncertain data",
    )


def write_figures(result: MissionAssessmentResult) -> None:
    """Write the Sobol figure when Matplotlib is installed.

    Args:
        result: Completed mission assessment.

    """
    if not matplotlib_available():
        return
    import matplotlib.pyplot as plt

    axes = sobol_figure(result)
    if axes is not None:
        save_figure(axes.figure, OUTPUT / "mission-systems-sobol.png")
        plt.close(axes.figure)


def main() -> None:
    """Run the mission laboratory and write the Markdown report."""
    data = component_data()
    system = lander(data)
    phased_mission(data)
    stage_model()
    allocation_and_fmeca()
    life_cycle = LifeCycleCostModel(
        (
            CostElement.one_time(
                "development", Triangular(180e6, 210e6, 300e6), category="acquisition"
            ),
            CostElement.one_time(
                "launch", Triangular(60e6, 70e6, 95e6), time=3.0, category="acquisition"
            ),
            CostElement.recurring("operations", 8e6, 4.0, 8.0, category="operation"),
        ),
        discount_rate=0.03,
    )
    register = RiskRegister(
        (
            Risk(
                "supplier delay",
                0.30,
                12e6,
                category="schedule",
                mitigation_cost=2e6,
                mitigated_probability=0.10,
            ),
            Risk("thermal redesign", 0.15, 25e6, category="technical"),
        )
    )
    # Same input name for the three computers: state-of-knowledge
    # correlation (they share one uncertain failure rate).
    rate = data["computer"].random_input("computer_rate")  # type: ignore[attr-defined]
    uncertain = {
        "cpu1": rate,
        "cpu2": rate,
        "cpu3": rate,
        "igniter": data["igniter"].random_input("igniter_p"),  # type: ignore[attr-defined]
    }
    assessment = MissionAssessment(
        system,
        MISSION_HOURS,
        name="lunar lander",
        seed=SEED,
        uncertain=uncertain,
        missions_per_year=1.0,
        horizon=5.0,
        mission_loss_cost=450e6,
        discount_rate=0.03,
        life_cycle=life_cycle,
        risks=register,
    )
    result = assessment.run(samples=2048, sensitivity_samples=512)
    report = assessment.report(result)
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "mission_systems_report.md"
    path.write_text(report, encoding="utf-8")
    write_figures(result)
    print(f"\nMission reliability R(T) = {result.reliability.reliability:.6f}")
    print(f"Most important block: {result.importance[0].component}")
    if result.uncertainty is not None:
        low = result.uncertainty.percentiles[0][1]
        high = result.uncertainty.percentiles[-1][1]
        print(f"Epistemic 5-95%: {low:.5f} - {high:.5f}")
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
