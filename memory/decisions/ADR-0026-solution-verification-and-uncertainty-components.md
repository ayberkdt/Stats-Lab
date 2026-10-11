# ADR-0026 — Solution verification, uncertainty components, and the workflow record

- Status: Accepted
- Date: 2026-10-12

## Context

Before this change UQ-0 had an observed order for uniform refinement only, a
Richardson extrapolation, and an additive error budget. Nothing turned a grid
study into the `u_num` that validation needs, predictions mixed or omitted
numerical and model-form uncertainty, and the UQ workflow existed only as
prose. ATHENS rows 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, and 1.9 were Partial.

## Decision

1. **Solution verification lives in `uq.verification` (layer 2).**
   `grid_convergence_index` follows the three-grid procedure of Celik et al.
   (2008): convergence classification from `R = eps32/eps21`, observed order
   by fixed-point iteration for non-uniform ratios with a Brent fallback on
   the equivalent monotone equation (the solver used is recorded), Richardson
   extrapolation, absolute and relative fine-grid GCI with `F_s = 1.25`.
   Divergent sequences raise; oscillatory ones are flagged.
2. **The asymptotic-range ratio is evaluated at the theoretical order.**
   At the observed order of the same three grids the absolute ratio
   `GCI_23 / (r21^p GCI_12)` is identically one, so it tests nothing. Roache's
   relative form is still reported, and `asymptotic_ratio_at(p_th)` provides
   the informative check used by `verify_solution`'s warning.
3. **Four or more grids use a simplified least-squares order.** The fit
   `phi = phi_0 + alpha (h/h_1)^p` and the rule `U = 1.25|alpha| + U_s`
   (credible order) or `U = 3 max(|alpha|, |alpha_th|) + max(U_s, U_s,th)`
   keep the structure of Eca and Hoekstra (2014) without their weighted fits
   or alternative expansions. The registry says so explicitly.
4. **`SolutionVerification` is the hand-off.** It states `phi_1 +/- U` with
   method, convergence type, order, factor of safety, and warnings. The
   conversion to `u_num` is a named choice (`"uniform"`: `U/sqrt(3)`;
   `"normal95"`: `U/1.96`); `VerificationEvidence.from_solution_verification`
   feeds the credibility record. Oscillatory three-grid data use the project
   rule `U = 3 (phi_max - phi_min)/2`.
5. **Simulations are recorded as computational experiments.**
   `ComputationalExperiment` separates physical parameters from numerical
   settings; `refinement_study` varies one setting and calls
   `verify_solution`.
6. **Component predictions live in `validation` (layer 5).** The model-form
   component needs `ValidationDomain`, the ASME comparisons, and the
   discrepancy record, which layer-4 `propagation` may not import.
   `predict_with_uncertainty_components` returns parametric, numerical, and
   model-form parts separately, plus a combined interval and the exact rule.
   Numerical uncertainty defaults to an epistemic interval that widens the
   bounds; `"uniform"`/`"normal"` sample it instead. The discrepancy variance
   is estimated by moments from validation comparisons and extrapolated as
   `sigma_0^2 (1 + kappa d)` with a declared `kappa`.
7. **`UQWorkflowRecord` lives in `uq`.** Seven steps with status and
   evidence; completion requires evidence, not-applicable requires a note,
   `gaps` reports prerequisite violations. `WORKFLOW_GUIDE` maps steps to
   dotted library calls (strings, so layer 2 does not import layer 7), and
   `record_stages` accepts `UQStudy` stage records structurally.

## Consequences

- Validation can consume a real numerical-uncertainty estimate instead of a
  hand-entered number, and predictions can no longer silently drop
  model-form or numerical uncertainty: excluded components are listed.
- The worked example (`examples/uq_workflow_lab.py`) shows a calibrated model
  whose parametric interval misses the truth at extrapolation while the
  combined interval covers it; it is asserted by the test suite.
- The extrapolation rule and the oscillatory bound are project choices, not
  literature results; changing them changes predictions and must be recorded.

## Rejected alternatives

- Placing component prediction in `propagation`: would need an upward import
  of the validation domain.
- A Gaussian-process discrepancy (Kennedy-O'Hagan): heavier, confounded with
  calibration parameters without extra structure; left for later work.
- Reporting only a combined variance: hides which component dominates and
  forces a distribution on numerical error.
