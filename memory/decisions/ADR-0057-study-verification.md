# ADR-0057: Solution-verification stage `UQStudy.verify`

- Status: Accepted (proposed during the 2026-10-12 session)
- Date: 2026-10-12

## Context

`probstat.uq` already has solution verification: `grid_convergence_index`
(Celik et al. three-grid GCI), `least_squares_convergence` (four or more
grids), and `verify_solution`, which turns either into a
`SolutionVerification` statement `phi_fine +/- U`. `probstat.validation`
already accepts `numerical_uncertainty` (a float `u_num` or a
`NumericalErrorBudget`) in `assess_validation` and the ASME V&V 20
comparison. `UQStudy` had no stage for it, so the VERIFICATION step of
`WORKFLOW_GUIDE` listed no study stage. A study's validation always used
`u_num = 0`.

## Decision

1. New stage `UQStudy.verify(quantity, grid_values, refinement_ratio=None, *,
   grid_sizes=None, safety_factor=None, theoretical_order=2.0,
   asymptotic_tolerance=0.1, distribution="uniform")`, stage kind
   `verification`. It is mapped to `WorkflowStep.VERIFICATION`, and the
   calls list gains `probstat.workflow.UQStudy.verify`.
2. **No new numerics.** The stage calls `verify_solution` and nothing
   else. With `refinement_ratio=r` the values are fine to coarse and the
   spacings are `1, r, r^2, ...`. The GCI depends only on ratios, so the
   choice `h_1 = 1` does not matter. With `grid_sizes` the values pair with
   the sizes in any order. Exactly one of the two must be given.
   `safety_factor=None` means the library default 1.25 (`DEFAULT_SAFETY_FACTOR`).
3. **Summary.** The keys are quantity, method, grids, convergence,
   fine_value, observed_order, extrapolated, safety_factor, `U`, `u_num`,
   `u_num.distribution`, `asymptotic_range`, reliable, and warnings. A GCI
   study adds `gci_fine` and `asymptotic_ratio`. `asymptotic_range` is the
   ratio test at the theoretical order for three grids, the
   `credible_order` flag for least squares, and `"n/a (oscillatory)"` for
   the oscillation bound. Order and extrapolation are `"n/a"` when the
   library returns `None`. The stage charges 0 evaluations; its stream
   seed is recorded for uniformity only.
4. **Storage and wiring.** `u_num = SolutionVerification.standard_uncertainty(
   distribution)` is stored per QoI (the latest run wins) and returned by
   `UQStudy.numerical_uncertainty(q)`. `validate` gains
   `numerical_uncertainty: float | NumericalErrorBudget | Literal["verified"]
   = 0.0`, which is forwarded to `assess_validation`. `"verified"` uses
   the stored value; a missing verification raises `KeyError`. The default
   0.0 is what `assess_validation` used before, so default results and
   summaries are bit-identical. A non-zero value adds `u_num` and
   `u_num.source` to the summary.
5. **Prediction is not wired.** `predict` pushes a posterior through the
   model. Adding a numerical term there would mix two treatments (an
   interval or a distribution). `validation.predict_with_uncertainty_components`
   already makes that choice explicit and takes the `SolutionVerification`
   as `numerical=`. Users pass the statement that `verify` returns.

## Evidence (`tests/test_workflow_verify.py`)

- `phi(h) = 2 + 0.03 h^2 + 0.004 h^3`, `h = 1, 2, 4` (2.034, 2.152, 2.736):
  observed order 2.30718, extrapolated 2.00412, `GCI_fine = 0.0373498`,
  `u_num = 0.0215639`. These are exactly equal to the direct
  `grid_convergence_index` and `verify_solution` calls. The ratio at order 2
  is 1.2373, so `asymptotic_range = False` with a warning.
- Four grids of `2 + 0.03 h^2` give least-squares order 2.000000. The
  oscillating triple (2.0, 2.1, 1.95) reports order `n/a`.
- Spring model, seed 6, 400 samples: with `"verified"`, every
  experiment's `u_val` equals `hypot(u_val_without, u_num)` to 1e-12
  (0.10238 to 0.10463, 0.19485 to 0.19604, 0.29584 to 0.29662). Areas,
  u-pooling, the verdict, and every other summary key are identical to
  the run without `u_num`.

## Consequences

- Positive: the verification step of a study now has recorded evidence, and
  `UQWorkflowRecord.record_stages` marks it complete. The numerical
  uncertainty reaches the V&V 20 validation uncertainty without the user
  copying numbers between objects.
- Limitations: (a) the user supplies the grid solutions; the stage does not
  run a solver at different resolutions (`uq.refinement_study` does that
  outside the study). (b) One QoI per call, and `u_num` is a single number
  for the QoI. It is applied unchanged at every validation condition, even
  though discretization error usually varies with the conditions. (c)
  `u_num` affects only the ASME comparison; the area-metric verdict ignores
  it. (d) `predict` is not adjusted (decision 5).
