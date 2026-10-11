# 2026-10-12 — UQ-0: solution verification, uncertainty components, workflow record

## Done

- `uq.verification`: `grid_convergence_index` (Celik et al. 2008, non-uniform
  ratios, fixed point + Brent fallback, oscillation/divergence detection,
  asymptotic ratio incl. `asymptotic_ratio_at(p_th)`),
  `least_squares_convergence` (simplified Eca-Hoekstra 2014),
  `SolutionVerification`, `verify_solution`, `representative_spacing`.
- `uq.experiment`: `ComputationalExperiment`, `run_computational_experiment`,
  `refinement_study`.
- `uq.workflow_record`: `UQWorkflowRecord`, `WorkflowStep`, `StepStatus`,
  `WORKFLOW_GUIDE`, `record_stages` for `UQStudy` records.
- `validation.prediction`: `estimate_discrepancy`, `DiscrepancyExtrapolation`,
  `predict_with_uncertainty_components`; `VerificationEvidence.
  from_solution_verification`.
- Worked example `examples/uq_workflow_lab.py`; guide sections in
  `docs/foundations/vv-uq-foundations.md`; ADR-0026; ATHENS 1.1, 1.2, 1.4,
  1.5, 1.6, 1.7, 1.9 marked Implemented.

## Verification

`PYTHONPATH=src python -m pytest -q`: 474 passed, 711 subtests (new files:
38 tests in about 5 s). `ruff check .` clean; `mypy src/probstat` clean.

## Next safe step

Gaussian-process (Kennedy-O'Hagan) discrepancy and the full Eca-Hoekstra
procedure (weighted fits, alternative expansions) remain open; the
discrepancy growth rate `kappa` is an assumption to replace with data.
