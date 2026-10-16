# ADR-0066: Hierarchical systems and the mission-assessment workflow

- Status: Accepted
- Date: 2026-10-16

## Context

The 0.16 work added, in parallel workstreams, an exact BDD system engine
(ADR-0060), analytic continuous-time Markov reliability models (ADR-0061),
cost analysis (ADR-0062), risk registers, FMECA and reliability data
(ADR-0063), richer input laws including epistemic inputs (ADR-0064), and
automatic differentiation with sensitivity fixes (ADR-0065). Each is useful
alone, but a systems engineer needs them to work as one chain: data ->
component laws -> subsystems -> mission system -> uncertainty -> cost and
risk, with one seed and one report.

## Decision

1. `probstat.systems.adapters`: `MarkovComponent(model)` and
   `SubsystemComponent(system)` implement the `ComponentModel` protocol, so a
   CTMC subsystem or a whole `SystemModel` is one block of a larger system.
   The composition is exact under modular decomposition (subsystems share
   no component, repair crew, load, or CCF group; EQ-SYS-ADAPT-001). The
   Markov adapter keeps the largest uniformization error bound it has seen in
   `last_error_bound`, so the truncation error is never silently dropped.
2. `probstat.workflow.MissionAssessment` (layer 7) orchestrates, for one
   `SystemModel` and mission time:
   - exact reliability, availability, MTTF, minimal cut sets, single points
     of failure;
   - importance measures sorted by Birnbaum importance;
   - epistemic propagation of component-parameter laws through
     `SystemModel.as_probabilistic_model` and `UQStudy`, using plain Monte
     Carlo for the distribution (so order-statistic percentile intervals are
     valid) and a Sobol design for the ranking;
   - cost: life-cycle cost at element means (exact expectation), expected
     discounted mission-loss cost at frequency `n Q(T)`
     (EQ-WF-MISSION-001), campaign loss probability `1 - R^N` evaluated
     with `expm1/log1p` (EQ-WF-MISSION-002), and the cost-risk S-curve when
     the life-cycle model has random elements (EQ-WF-MISSION-003);
   - the project risk register with the mission-loss risk added.
   Every stochastic step uses a named child stream of one root seed
   (`"uncertainty"`, `"cost-risk"`).
3. `render_mission_report` produces one Markdown report; the lab
   `examples/mission_systems_lab.py` exercises the full chain.

## Defects found during integration (fixed, with regression tests)

- Autodiff FORM applied the diagonal Nataf Jacobian even when the joint
  input used a non-Gaussian copula (Rosenblatt map). It now raises
  `AutodiffUnsupportedError`, so `gradient="auto"` falls back to finite
  differences (`tests/test_integration_cross_workstream.py`).
- The Saltelli (2010) first-order estimator used the raw product
  `f_B (f_ABi - f_A)`, whose variance grows with the square of the output
  mean. A system reliability near 0.92 with spread 0.003 gave a first-order
  index near 5. `f_B` is now centred on the pooled mean, as SALib does,
  which makes the estimate invariant to a constant offset (regression test
  with a 1e6 offset). One documented output in `docs/prerequisites/r-bridge.md`
  moved from `(0.33, 0.44, 0.01)` to `(0.31, 0.44, 0.02)`, which is closer
  to the analytic `(0.31, 0.44, 0.00)`.
- Two workstreams defined `PhasedMissionResult`; the Markov one is now
  `MarkovPhasedMissionResult`. Two different Higham (2002) works shared
  one citation key; the nearest-correlation paper is now
  `REF-HIGHAM-NCM-2002`.

## Consequences

- `MissionAssessment` reports `R(T)` at point parameters and `E[R(T)]` over
  the epistemic law separately; the cost step uses the point value unless
  `cost(unreliability=...)` is given, and the guide says which to report.
- CCF parameters are not yet epistemic inputs; dynamic gates (PAND, FDEP,
  spare gates beyond the standby builder) and semi-Markov sojourns remain
  out of scope.
