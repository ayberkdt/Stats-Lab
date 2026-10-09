# Session: UQ-5 sensitivity analysis

Date: 2026-10-09

## Delivered

- Local central-difference gradients with explicit steps and work counts.
- Functional-ANOVA foundations and pick-freeze first-, second-, and total-effect
  Sobol estimates for independent inputs.
- Total-effect ranking, fixing candidates, and uncertainty-reduction priorities.
- Additive and pure-interaction regression tests plus a seeded Ishigami lab.
- Equation records, primary references, guide, architecture layer, ADR-0012,
  curriculum evidence, and visualization.

## Next safe step

Implement UQ-6 inverse problems and Gaussian Bayesian calibration without
conflating parameter fit, posterior uncertainty, and predictive adequacy.

## Verification

- Ruff passed across source, tests, and examples.
- Strict mypy passed across 47 source files.
- All 132 unit tests passed, including documentation, architecture, and the
  complete 110-row curriculum ledger.
- Bytecode compilation and the seeded 400,000-evaluation Ishigami lab passed.
- The first-versus-total chart was visually inspected and exposes the intended
  interaction effect without clipped labels or misleading axes.
- The isolated `0.12.0` wheel built and passed a zip-import smoke test.
