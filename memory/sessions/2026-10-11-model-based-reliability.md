# Session: Model-based reliability

- Date: 2026-10-11
- Scope: General FORM/SORM, rare-event simulation, systems, and inverse
  reliability on `ProbabilisticModel`

## Completed

- `reliability.model_based`: `ModelLimitState`, HL-RF/iHL-RF FORM, SORM
  (Breitung, Hohenbichler-Rackwitz), inverse FORM, and target-beta design.
- `reliability.simulation`: crude Monte Carlo, importance sampling around the
  design point, subset simulation (modified Metropolis), and a method
  comparison.
- `reliability.systems`: series and parallel system sampling, uni-modal and
  Ditlevsen bounds, a bivariate normal CDF, and Gauss-Legendre nodes.
- 19 equation records (`EQ-REL-*`), 14 references, the
  `docs/foundations/advanced-reliability.md` guide, and ADR-0015.

## Decisions

See ADR-0015. Non-converged FORM is reported, not raised. Invalid SORM
curvatures raise an error. The subset c.o.v. is reported with both the
independence value and the fully correlated value.

## Verification

- `PYTHONPATH=src python -m pytest -q`: 196 passed (27 new tests, about 5-10 s
  for the new file).
- `python -m ruff check .`: all checks passed.
- `python -m mypy src/probstat`: no issues in 69 source files.

## Next

- Evaluate the upper tail of `JointInput.from_standard_normal` through
  `Phi(-z)`. FORM stalls for latent coordinates above about 6-8.
- Add a multiple-design-point search.
