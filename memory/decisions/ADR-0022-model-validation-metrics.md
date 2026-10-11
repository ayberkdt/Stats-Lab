# ADR-0022 — Model validation metrics and scoped credibility

- Status: Accepted
- Date: 2026-10-11

## Context

ATHENS-1.3 (model validation for intended use) was the only UQ-0 row still
`Planned`. The library could calibrate models and propagate uncertainty, but
it could not compare predictions with independent experimental data, describe
where that comparison applies, or keep numerical error, model-data agreement,
and predictive uncertainty apart. Two failure modes had to be designed out:
a single opaque "validation score", and a report that reads as a general claim
that the model is "valid".

## Decision

1. `probstat.validation` (layer 5) owns validation records, metrics, the
   assessment, and the credibility record. It imports `modeling`,
   `propagation`, `inference`, `uq`, `statistics`, `distributions`, and
   `linalg` downward only. It never imports reliability, sensitivity,
   surrogates, risk, or workflow.
2. `ValidationExperiment` stores measured QoI values with named conditions,
   the standard uncertainty of one measurement, and a mandatory provenance
   text. Conditions reach the model through `model.fixed`, as in
   `inverse.Experiment`. A condition that shadows a random input is rejected
   by `ProbabilisticModel`.
3. `ValidationDomain` decides bounding-box and exact convex-hull membership.
   Hull membership uses a phase-one simplex with Bland's rule in range-scaled
   coordinates. Extrapolation is flagged outside the hull or beyond a scaled
   nearest-point distance. We chose an exact LP over an iterative min-norm
   method because Frank-Wolfe-type methods converge too slowly to decide
   boundary cases.
4. The metrics are separate functions with separate result types: the area
   metric (exact for step CDFs), u-pooling (PIT area and KS), Gaussian
   Mahalanobis residuals, the reliability metric, the ASME V&V 20 comparison,
   and the Bayes factor. No metric is folded into another.
5. The Bayes factor always uses an alternative that the user declares
   (`uniform`, `normal`, or `inflated`) and reports it next to `log B`.
   The model density is a Gaussian-kernel mixture over predictive samples,
   with kernel width `sigma_m` when measurement noise is known and Silverman's
   bandwidth otherwise.
6. A `NumericalErrorBudget` bound enters `u_val` only through the explicit
   rectangular Type B conversion `B / sqrt(3)`. Verification bounds and
   standard uncertainties stay distinct types.
7. `assess_validation` derives every random draw from one root stream:
   `child("validation", "prediction", i)` and `child("validation", "noise",
   i)`. The verdict is "consistent" only if every per-experiment area metric
   is within the tolerance *and* the u-pooling KS test does not reject at the
   stated significance. The verdict text always names the tolerance, the
   significance, and the validation domain, and states that no general claim
   is made. The other metrics are reported as evidence and do not change the
   verdict. ASME V&V 20 is explicitly not a pass/fail test.
8. `CredibilityRecord` keeps `VerificationEvidence`, `ValidationAssessment`,
   and `PredictionStatement` in separate tuples. Its summary lists explicit
   gaps and computes no aggregate credibility scalar.

## Consequences

- ATHENS-1.3 is `Implemented`, with evidence in source, tests, and the guide.
  ATHENS-1.4 and ATHENS-1.7 gain evidence paths but stay `Partial`: model-form
  uncertainty is *estimated* at validation points (`delta_model` interval)
  but not yet modeled and propagated to new conditions.
- Observations are treated as conditionally independent draws from the
  predictive law. That is right when the input uncertainty is aleatory
  (it varies between replicates). When the input uncertainty is epistemic and
  shared, replicates are dependent, and the pooled tests overstate the
  evidence. The guide documents this limitation.
- PIT values come from a finite predictive ECDF. The KS p-value ignores that
  discreteness, which is negligible for hundreds of samples.

## Rejected alternatives

- **A single weighted credibility score.** It mixes quantities with
  different units and hides which evidence is missing.
- **Using the Gaussian Mahalanobis p-value as the verdict.** It is exact only
  for Gaussian predictive laws, while the area metric and u-pooling work
  directly with samples.
- **A default "uninformative" Bayes-factor alternative left unreported.** By
  the Jeffreys-Lindley effect, a broad enough alternative favors almost any
  model.
- **Treating a numerical error bound as a standard uncertainty.** That
  understates `u_val` by a factor of `sqrt(3)` under a rectangular model, and
  the assumption would be implicit.
