# Project Memory

This is the durable entry point for future work. Read it before changing the
repository, then read the linked roadmap and relevant ADRs.

## Mission

Build an auditable probability and statistics system from first principles to
frontier methods. Mathematical completeness, software correctness, and clear
teaching are co-equal goals.

## Non-negotiable invariants

1. The package dependency graph remains acyclic and points toward foundations.
2. Every callable has a useful docstring.
3. Every code-evaluated equation has a stable `EQ-*` record with assumptions,
   interpretation, code locations, and a source.
4. Every source key used in code exists in `docs/references.md`.
5. Simpler explanations may add intuition but may not delete scope, assumptions,
   counterexamples, or limitations.
6. Random behavior exposes reproducibility controls.
7. Numerical algorithms document stability limits and are tested at boundaries.

## Current state

- Current release: `0.10.0` UQ-3 variance reduction, budget allocation, and
  cost-aware efficiency diagnostics.
- Python baseline: 3.11+.
- Runtime dependency policy: standard library first; optional Matplotlib for
  plots. Add numerical dependencies only through an ADR.
- Tests: standard-library `unittest` so the base checkout can verify itself.
- Public stability: experimental; interfaces may evolve until `1.0`.

## Navigation

- Work status and future sequencing: [ROADMAP.md](ROADMAP.md)
- Canonical ATHENS curriculum ledger:
  [athens-2026-coverage.md](../docs/curriculum/athens-2026-coverage.md)
- Architecture rationale: [ADR-0001](decisions/ADR-0001-layered-citation-first.md)
- VV-UQ scope architecture:
  [ADR-0007](decisions/ADR-0007-auditable-vv-uq-curriculum.md)
- UQ-1 numerical and calibration boundaries:
  [ADR-0008](decisions/ADR-0008-uq1-linear-algebra-and-fit-calibration.md)
- Propagation cost/error boundaries:
  [ADR-0009](decisions/ADR-0009-propagation-cost-and-error-separation.md)
- Advanced Monte Carlo estimator boundaries:
  [ADR-0010](decisions/ADR-0010-variance-reduction-estimator-contracts.md)
- Latest session: [2026-10-09 UQ-3 advanced Monte Carlo](sessions/2026-10-09-uq3-advanced-monte-carlo.md)
- Previous session: [2026-10-09 UQ-2 propagation](sessions/2026-10-09-uq2-propagation.md)
- Previous session: [2026-10-09 UQ-1 modeling and fitting](sessions/2026-10-09-uq1-modeling-and-fitting.md)
- Earlier session: [2026-10-09 ATHENS VV-UQ scope](sessions/2026-10-09-athens-vv-uq-scope.md)
- Earlier session: [2026-10-09 waiting and continuous foundations](sessions/2026-10-09-waiting-and-continuous-foundations.md)
- Previous session: [2026-10-09 discrete distributions](sessions/2026-10-09-discrete-distributions.md)
- Generating-functions session: [2026-10-09 generating functions](sessions/2026-10-09-generating-functions.md)
- Random-variable session: [2026-10-09 random variables](sessions/2026-10-09-random-variables.md)
- Finite probability session: [2026-10-09 finite probability](sessions/2026-10-09-finite-probability.md)
- Initial session: [2026-10-09 foundation](sessions/2026-10-09-foundation.md)
- Contributor rules: `CONTRIBUTING.md`
- Equations: `docs/equation-registry.md`
- Sources: `docs/references.md`

## Next safe step

Build UQ-4 reliability on the probability and work-accounting contracts:
limit-state functions, direct failure estimation with Wilson intervals, rare
event diagnostics, FORM with an explicit optimization boundary, reliability
index, design point, and directional sampling. Preserve the UQ-1 backlog for
Weibull MLE, Student-t intervals, fitted-null calibration, stream splitting,
and a real-data fitting case study.

