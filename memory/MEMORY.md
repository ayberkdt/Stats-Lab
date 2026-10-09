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

- Current release: `0.1.0` foundation skeleton.
- Python baseline: 3.11+.
- Runtime dependency policy: standard library first; optional Matplotlib for
  plots. Add numerical dependencies only through an ADR.
- Tests: standard-library `unittest` so the base checkout can verify itself.
- Public stability: experimental; interfaces may evolve until `1.0`.

## Navigation

- Work status and future sequencing: [ROADMAP.md](ROADMAP.md)
- Architecture rationale: [ADR-0001](decisions/ADR-0001-layered-citation-first.md)
- Latest session: [2026-10-09 foundation](sessions/2026-10-09-foundation.md)
- Contributor rules: `CONTRIBUTING.md`
- Equations: `docs/equation-registry.md`
- Sources: `docs/references.md`

## Next safe step

Implement an immutable finite sample-space/event algebra module before
conditional probability. Register every derived probability equation and add
property-style tests for complements, unions, and normalization.

