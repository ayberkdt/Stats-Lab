# Systems engineering with probstat: from component data to mission risk and cost

This guide shows how the pieces fit together when a systems engineer
analyses a mission. Each step names the module, the method, and the
assumptions it relies on, so the result can be defended in a design review.
The worked example is
[`examples/mission_systems_lab.py`](../../examples/mission_systems_lab.py)
(a lunar lander), which writes a full Markdown report.

## The chain

```
test / field data ──► component laws ──► subsystem models ──► mission system ──► phases
 reliability.data      systems.components   systems.markov       systems.SystemModel   systems.PhasedMission
 (Beta, Gamma          (exponential,        (CTMC: repair,       (BDD: exact R(T),     (non-repairable,
  posteriors,           Weibull, demand,     standby, k-of-n)     cut sets, CCF,       phase-dependent
  chi-square bounds)    tested, repairable)                       importance)          rates)
        │                                                               │
        └──────── epistemic uncertainty (posteriors) ──────────────────►│
                                                                        ▼
                                  workflow.MissionAssessment: R(T), importance, epistemic
                                  distribution + Sobol ranking, cost (LCC + losses, S-curve),
                                  risk register  ──►  one seeded Markdown report
```

## 1. Component data

`probstat.reliability.data` turns test and field records into parameter
estimates with honest uncertainty:

| Data | Function | Output |
|---|---|---|
| `r` failures in `n` demands | `demand_failure_probability` | Beta posterior (Jeffreys by default), Clopper–Pearson interval |
| `r` failures in exposure `T` | `failure_rate_estimate`, `failure_rate_bounds` | Gamma posterior, chi-square bounds (time- or failure-terminated) |
| Zero-failure demonstration | `success_run_sample_size`, `demonstration_test_time` | Required tests or test time |
| Reliability growth | `crow_amsaa_fit` | Power-law NHPP fit, instantaneous MTBF |

Each posterior has `.random_input(name)`, which becomes an epistemic input
of the system model.

## 2. Component and subsystem models

Static and time-dependent blocks live in `probstat.systems.components`:
`FixedProbability`, `DemandComponent`, `ExponentialComponent`,
`WeibullComponent`, `LifeDistributionComponent` (any probstat law),
`RepairableComponent`, and `PeriodicallyTestedComponent`.

Use a **continuous-time Markov chain** when the block's behaviour depends
on its own history: repair, standby with switching, shared repair crews,
load sharing. `probstat.systems.markov` provides builders, all solved
analytically by uniformization with a rigorous error bound:

- `single_repairable_unit`, `repairable_k_out_of_n`, `parallel_redundant_pair`;
- `standby_system` (cold, warm, or hot standby, with imperfect switching);
- `load_sharing_pair`, `from_states`.

`MarkovComponent(model)` turns such a model into one block of a larger
system. `SubsystemComponent(system)` does the same for a whole
`SystemModel`. The composition is exact when the subsystem shares no
component, repair crew, load, or common-cause group with the rest of the
system (modular decomposition). If something is shared, model it once in a
flat structure instead.

## 3. The mission system: exact reliability

`SystemModel(structure, components, ccf_groups)` compiles a success-space
structure into a binary decision diagram (BDD). Structures are built from
`series`, `parallel`, `k_out_of_n` and `bridge`, or converted from a fault
tree with `from_fault_tree`. The BDD gives:

- exact `R(T)`, `Q(T)`, point availability, and the MTTF (with its
  quadrature error);
- minimal cut and path sets, Esary–Proschan and min-cut/min-path bounds;
- every standard importance measure: Birnbaum, criticality, Fussell–Vesely,
  RAW, RRW, improvement potential, structural;
- exact epistemic mean and variance of `R` for independent uncertain
  component probabilities. Shared groups are also exact when their raw
  moments are given.

**Common-cause failures** are modelled with `BetaFactorGroup`,
`MultipleGreekLetterGroup` and `AlphaFactorGroup` (staggered or
non-staggered testing). Each group is expanded into independent and CCF
basic events following the NUREG/CR-5485 parametrisation.

Two choices matter for mission reliability:

- **Hazard split by default.** Mission-time CCF probabilities split the
  cumulative hazard rather than the failure probability. The probability
  split under-states long-mission failure and gives `R(t)` a spurious
  positive limit. See ADR-0060.
- **Redundancy is capped by CCF.** In the example, the cut set
  `{CCF[cpu1,cpu2,cpu3]}` is a single point of failure even though the
  computers are 2-out-of-3.

### Dynamic fault trees: order and state of use

Static gates cannot express that a failure matters only in a certain
order, that a spare ages slowly until it is switched in, or that one
failure disables others. `DynamicFaultTree` adds the dynamic gates of
Dugan, Bavuso and Boyd (1992) and of the NASA Fault Tree Handbook
(Stamatelatos et al. 2002, Chapter 8) [REF-DUGAN-BAVUSO-BOYD-1992],
[REF-NASA-FTH-2002]:

| Gate | Builder | Fails when |
|---|---|---|
| Priority AND | `pand_gate` | all inputs fail, left to right (simultaneous counts as in order) |
| Spare | `spare_gate` | the unit in use fails and no listed spare is free; a spare runs at `alpha lambda` until claimed (`alpha = 0` cold, `1` hot); spares may be shared between gates |
| Functional dependency | `fdep_gate` | (no output) the trigger fails its dependent basic events at once |
| Sequence enforcing | `seq_gate` | all inputs fail; each later input can fail only after its predecessor |

Basic events (`DynamicBasicEvent(name, rate, dormancy)`) are exponential and
non-repairable. The tree is converted into a continuous-time Markov chain
whose state records the failed events, the unit each spare gate uses and
the PAND gates whose order was violated [EQ-SYS-DFT-001]. `unreliability`
and `reliability` come from uniformization with an error bound, and `mttf`
from the absorption solve [EQ-SYS-DFT-002]; `state_space().size` reports
the chain size and `max_states` stops a state explosion with a clear error.
A tree with static gates only is evaluated on the BDD, and the two paths
agree to `1e-14`. `monte_carlo_unreliability` simulates the same semantics
from failure times, with Clopper-Pearson intervals, as an independent check
[EQ-SYS-DFT-003]. Useful sanity checks: a cold pair of identical units is
Erlang(2, `lambda`), a hot pair is active parallel, and PAND(A, B) equals
`int_0^t a e^{-a s}(e^{-b s} - e^{-b t}) ds` [EQ-SYS-DFT-004].

The Markov path does not modularize: every relevant basic event enters the
state, so the state space grows exponentially with their number. Keep
dynamic gates where the dependence actually lives, and solve independent
static parts separately on the BDD.

### BDD variable order

The size of a BDD depends on the variable order, sometimes exponentially:
`x1 x2 + x3 x4 + ... ` has `2n` nodes when each pair is adjacent and
`2^(n+1) - 2` when all odd variables come first [REF-BRYANT-1986].
`BDD.sift(roots)` applies Rudell's sifting [REF-RUDELL-1993]: each variable
is moved through every position by adjacent level swaps and left where the
diagram is smallest [EQ-SYS-SIFT-001]. It returns a new manager, the mapped
roots and the node counts before and after; the represented functions, and
therefore all probabilities, are unchanged, and the size never increases.
`BDD.reorder(roots, order)` applies a prescribed order instead.

## 4. Phases and timelines

- **Phased missions with non-repairable components:**
  `PhasedMission(phases, components)`. Each phase has its own structure
  and duration. A component may get one model per phase, for example a
  much lower *dormant* failure rate while it is not used. Without that, a
  long cruise ages the landing engine at its active rate. The result is
  exact (BDD with phase-of-failure enumeration, Zang, Sun and Trivedi 1999)
  and gives cumulative and conditional phase reliabilities.
- **Phased missions with repair:** `MarkovPhasedMission`, with one
  generator per phase and state mapping between phases.
- **Mission timelines as absorbing Markov chains:** `MissionStageModel` takes
  stages with success, retry, abort and loss probabilities. It returns the
  exact probability of each absorbing outcome and the expected number of
  stage attempts.

## 5. Epistemic uncertainty and what to test next

`MissionAssessment(..., uncertain={component: random_input})` propagates
the parameter posteriors to `R(T)`.

- **Exact evaluations.** Every realization is evaluated exactly by the BDD,
  so the only error is sampling over the epistemic law. That error is
  reported as order-statistic percentile intervals and a mean interval.
- **Ranking.** Sobol indices rank which component's data drives the spread.
  This tells you which tests buy the most knowledge.
- **Shared parameters.** Components mapped to inputs with the *same name*
  share one value. This is state-of-knowledge correlation (Apostolakis and
  Kaplan 1981). Ignoring it understates the epistemic spread of redundant
  systems.

## 6. Cost and risk

Two calls cover cost and risk.

**`MissionAssessment.cost()`** combines:

- the life-cycle cost model (`probstat.cost.LifeCycleCostModel`) at its
  element means, which is the exact expectation because totals are linear in
  the elements;
- the expected discounted cost of mission losses, at frequency `n·Q(T)`;
- the probability of at least one loss in the campaign, `1 − R^N`;
- a cost-risk S-curve with correlated elements, giving P50/P70/P80/P90 and
  contingency (NASA Cost Estimating Handbook practice).

**`MissionAssessment.register()`** adds the mission-loss risk to the project
risk register (`probstat.risk.RiskRegister`). The register gives exact
expected exposure, variance and ranking, and mitigation ROI.

Other tools in the same packages:

- maintenance optimisation: `optimize_age_replacement`,
  `optimize_block_replacement`, `minimal_repair_overhaul`;
- spares: `minimum_spares`, `optimal_spares`;
- learning curves: `LearningCurve`;
- design alternatives with EVPI: `compare_alternatives`,
  `reliability_decision_problem`;
- FMECA (MIL-STD-1629A criticality; RPN with its documented flaws);
- the MIL-STD-882E risk matrix;
- F–N curves;
- reliability allocation: equal, ARINC, AGREE, and cost-optimal for any
  structure.

## Choosing the method

| Situation | Method | Exact? |
|---|---|---|
| Static structure, independent components, any size of logic with shared events | BDD (`SystemModel`) | Yes |
| Repair, standby, switching, load sharing inside a block | CTMC (`systems.markov`) | Yes, up to a stated truncation bound |
| Failure order, spares (shared pools), functional dependencies, non-repairable | `DynamicFaultTree` (state space into a CTMC) | Yes, up to a stated truncation bound |
| Mission phases, non-repairable | `PhasedMission` | Yes |
| Mission phases with repair | `MarkovPhasedMission` | Yes, up to a stated bound |
| Discrete mission timeline with retries/aborts | `MissionStageModel` | Yes |
| Uncertain parameters (epistemic) | `MissionAssessment.uncertainty` | Each sample is exact; sampling error is reported |
| Non-exponential repair, ageing with maintenance, complex dependencies | Monte Carlo on a `ProbabilisticModel` / `UQStudy` | Statistical error only |
| Physics-based limit states (load vs capacity) | FORM/SORM/subset simulation (`reliability`) | FORM is an approximation; simulation is statistical |

## Assumptions to state in every report

1. **Independence:** component failures are independent except through the
   modelled CCF groups and shared Markov blocks.
2. **Coherence:** structures are monotone. Non-coherent logic is evaluated
   correctly by the BDD, but bounds and cut-set importances assume
   coherence.
3. **Exponential sojourns:** Markov models assume exponential sojourn times.
   Use semi-Markov or simulation when repair times are far from
   exponential.
4. **CCF parameters** come from generic data. `as_probabilistic_model` takes
   uncertain component parameters only; propagate CCF-parameter uncertainty
   with `ccf_parameter_uncertainty` (mean, variance, and the CCF share of the
   variance) or bound it with `ccf_interval_bounds`.
5. **Campaign independence:** missions in a campaign are independent and
   identical.
6. **Point estimates vs. epistemic means:** the epistemic mean `E[R]` and the
   reliability at mean parameters `R(E[θ])` differ when `R` is nonlinear in
   the parameters, for example with rates. Report which one is used.
   `MissionAssessment.cost(unreliability=...)` accepts either.

## Common pitfalls

- **Phase reliabilities do not multiply.** The phases share the same
  components, so the product of separately computed phase reliabilities is
  not the mission reliability. Use `PhasedMission`.
- **One model per component ages on the whole mission.** Give a list with one
  model per phase to apply a dormant rate during the phases where the
  component is not used.
- **Composition needs independent modules.** `MarkovComponent` and
  `SubsystemComponent` are exact only when the block shares no component,
  repair crew, load, or CCF group with the rest of the system.
- **Seeds default differently.** `ccf_parameter_uncertainty` and
  `ccf_interval_bounds` default to `seed=0` (repeatable), while the sampling
  methods of `probstat.reliability` default to `seed=None` (fresh entropy).
- **Dynamic fault trees stop at `max_states`.** `DynamicFaultTree` raises
  instead of approximating when the generated chain exceeds `max_states`
  (default `10_000`).

Phase 1 needs either of two units (`parallel`) for 100 h, phase 2 needs both
(`series`) for another 100 h, each unit at `1e-3` per hour:

```python
import inspect
import math
from probstat.reliability import model_monte_carlo
from probstat.systems import (
    DynamicFaultTree, ExponentialComponent, Phase, PhasedMission, SystemModel,
    ccf_parameter_uncertainty, parallel, series,
)

units = {"A": ExponentialComponent(1e-3), "B": ExponentialComponent(1e-3)}
mission = PhasedMission(
    (Phase("either", 100.0, parallel("A", "B")), Phase("both", 100.0, series("A", "B"))),
    units,
)
round(mission.evaluate().reliability, 6)          # 0.67032, exp(-0.4)
round(math.exp(-0.4), 6)                          # 0.67032
naive = (SystemModel(parallel("A", "B"), units).reliability(100.0)
         * SystemModel(series("A", "B"), units).reliability(200.0))
round(naive, 6)                                   # 0.66425, too low
inspect.signature(ccf_parameter_uncertainty).parameters["seed"].default   # 0
inspect.signature(model_monte_carlo).parameters["seed"].default           # None
inspect.signature(DynamicFaultTree).parameters["max_states"].default      # 10000
```
