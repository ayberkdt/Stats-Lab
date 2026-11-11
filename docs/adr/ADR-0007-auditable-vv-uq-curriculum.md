# ADR-0007: Auditable VV-UQ Curriculum and Layer Growth

- Status: Accepted
- Date: 2026-10-09

## Context

The ATHENS 2026 scope spans numerical verification, validation, uncertainty
modeling and propagation, advanced Monte Carlo, reliability, global
sensitivity, inverse problems, Bayesian calibration, filtering, data
assimilation, laboratories, and mathematical prerequisites. A prose roadmap
can silently omit a method or imply completion without evidence.

Putting every future method in `probstat.uq` would also create a large import
hub and increase circular-dependency risk.

## Decision

`docs/curriculum/athens-2026-coverage.md` is the canonical scope ledger. Every
topic has a stable `ATHENS-*` ID, controlled status, delivery target, planned
owner, and evidence field. `Implemented` and `Partial` require paths that exist;
`Planned` requires an owner and target. An automated contract checks all 110
identities and evidence paths.

Layer growth follows this direction:

```text
foundations / numerics / linalg / optimization
  -> probability / statistics / distributions
  -> uq / multivariate / inference
  -> propagation / monte_carlo
  -> reliability / sensitivity / inverse
  -> filtering / data_assimilation
  -> visualization / labs / frontier systems
```

`probstat.uq` owns only cross-cutting vocabulary, named forward-model and QoI
contracts, workflow primitives, and solution verification. Algorithms with a
cohesive domain receive dedicated packages. Higher layers may depend downward;
lower layers never import workflow or presentation code.

## Consequences

- “Covered” means tracked; “implemented” requires cited, tested evidence.
- Scope completeness is machine checked, while mathematical correctness still
  requires focused tests and review.
- New curriculum requests extend the matrix instead of being folded into vague
  bullets.
- The complete curriculum gate is 110/110 `Implemented`, not creation of empty
  package names.
- R programming remains a documented interoperability bridge rather than a
  Python runtime dependency.
