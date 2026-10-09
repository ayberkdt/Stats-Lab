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

- Current release: `0.7.0` auditable ATHENS VV-UQ scope and verification core.
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
- Latest session: [2026-10-09 ATHENS VV-UQ scope](sessions/2026-10-09-athens-vv-uq-scope.md)
- Previous session: [2026-10-09 waiting and continuous foundations](sessions/2026-10-09-waiting-and-continuous-foundations.md)
- Previous session: [2026-10-09 discrete distributions](sessions/2026-10-09-discrete-distributions.md)
- Generating-functions session: [2026-10-09 generating functions](sessions/2026-10-09-generating-functions.md)
- Random-variable session: [2026-10-09 random variables](sessions/2026-10-09-random-variables.md)
- Finite probability session: [2026-10-09 finite probability](sessions/2026-10-09-finite-probability.md)
- Initial session: [2026-10-09 foundation](sessions/2026-10-09-foundation.md)
- Contributor rules: `CONTRIBUTING.md`
- Equations: `docs/equation-registry.md`
- Sources: `docs/references.md`

## Next safe step

Complete ATHENS UQ-1 before building propagation: implement Lognormal and
Weibull, then a dependency-neutral multivariate-vector foundation and
multivariate Normal. Add sampling contracts, maximum-likelihood estimation,
Kolmogorov-Smirnov and Anderson-Darling diagnostics, confidence intervals, and
teaching plots. Preserve the distribution roadmap by deriving Gamma, Beta,
Chi-square, Student-t, and F on the existing special-function foundation.

