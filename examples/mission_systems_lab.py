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
written to ``examples/generated/mission_systems_report.md``.
"""

from __future__ import annotations

from pathlib import Path

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
from probstat.workflow import MissionAssessment

OUTPUT = Path(__file__).parent / "generated"
MISSION_HOURS = 720.0


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
        seed=2026,
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
    print(f"\nMission reliability R(T) = {result.reliability.reliability:.6f}")
    print(f"Most important block: {result.importance[0].component}")
    if result.uncertainty is not None:
        low = result.uncertainty.percentiles[0][1]
        high = result.uncertainty.percentiles[-1][1]
        print(f"Epistemic 5-95%: {low:.5f} - {high:.5f}")
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
