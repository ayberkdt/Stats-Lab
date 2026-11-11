# ADR-0009: Propagation Cost and Error Separation

- Status: Accepted
- Date: 2026-10-09

## Context

Uncertainty propagation combines at least three distinct error sources: output
variability, Monte Carlo sampling error, and numerical/model error. APIs that
return only one number encourage these quantities to be confused. Expensive
simulators also require explicit accounting of model calls.

## Decision

`probstat.numerics` owns domain-neutral deterministic integration rules and
their evaluation counts. `probstat.monte_carlo` owns sample-mean,
event-probability, integration, confidence, convergence, and cost metadata but
does not know about forward models or QoIs. `probstat.propagation` combines
explicit parameter samples with the lower-level forward-model/QoI contract and
Monte Carlo estimators.

Output sample variance and estimator standard error are separate named fields.
Every Monte Carlo estimate carries sample size and charged evaluations. A
forward-model propagation counts complete model calls once even when each call
returns several QoIs.

Threshold events use strict `Y > S` semantics and Wilson intervals. Mean
intervals are explicitly normal approximations until Student-t support exists.
Finite-bound expectation quadrature never silently claims omitted unbounded
tail mass.

## Consequences

- Users can compare deterministic and stochastic work without guessing cost.
- Increasing sample count cannot be mistaken for reducing discretization or
  model-form error.
- Raw propagated outputs remain available for later diagnostics.
- The initial implementation materializes samples; streaming accumulation is a
  planned scalability extension.
- Advanced variance-reduction estimators can reuse the same estimate metadata
  without importing forward-model semantics.
