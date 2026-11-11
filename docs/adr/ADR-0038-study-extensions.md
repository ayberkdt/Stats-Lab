# ADR-0038: Tail, information, and design stages in `UQStudy`

- Status: Accepted
- Date: 2026-10-12

## Context

ADR-0035 (experimental design) and ADR-0036 (extremes and information theory)
added `probstat.design`, `probstat.risk.extremes`, and `probstat.information`.
None of them could be reached from `workflow.UQStudy` (ADR-0025). A user had
to pull samples out of a study, call each package by hand, and track seeds
and model-run counts alone. That is exactly what `UQStudy` exists to remove.

## Decision

1. **Three new stages.** Each is thin: it calls one domain function, records
   a `StageRecord`, and returns a frozen result object.
   - `extremes(quantity, *, tail, threshold_quantile, threshold, level,
     return_periods, method, confidence, size, design)`
     - Stage kind `"extremes"`; returns `TailAnalysis`.
     - Calls `risk.fit_pot` with `observations_per_year = 1`, so a return
       period counts model runs.
     - `tail="lower"` fits the upper tail of `-Y` and maps levels and
       intervals back to QoI units.
     - Given a `level`, it records the POT probability `zeta_u S_GPD(level)`
       next to `estimate_probability` (Wilson interval) on the same sample.
   - `information_sensitivity(quantity, *, max_points, k, unit, size,
     design)`
     - Stage kind `"information"`; returns `InformationSensitivity`.
     - Calls `information.ksg_mutual_information` once per input on the
       first `max_points` runs.
   - `design(candidates, size, *, noise, include_experiments, allow_repeats,
     method, samples, outer, inner, reuse_samples)`
     - Stage kind `"design"`; returns `DesignRecommendation`.
     - Builds the problem with `BayesianDesignProblem.from_calibration` on
       the latest `calibrate()` problem, or from explicit `noise`.
     - Uses the calibration experiments as the greedy baseline.
     - Calls `greedy_design` on stream `design/k/greedy`, then
       `expected_information_gain` on the full design on stream
       `design/k/final`.
2. **Sample reuse.** `extremes` and `information_sensitivity` reuse the
   latest propagation sample when `size` is `None`, and charge 0 runs.
   Otherwise they propagate a fresh sample on their own stage stream and
   charge its runs to themselves. Reused samples are never charged twice, and
   a fresh sample is still reproducible from the root seed.
3. **Independent final EIG.** The greedy step value is the maximum over
   candidates scored with one common stream, so it is biased upward. The
   recorded `eig` comes from an independent stream. In the worked example
   the greedy path gives 6.81 nats and the independent estimate
   6.50 ± 0.14 nats for the same design.
4. **Missing problem raises.** `design()` with neither a calibration problem
   nor `noise` raises `ValueError`.
5. **Workflow mapping.** `WORKFLOW_GUIDE` maps:
   - `extremes` and `information` to `PROPAGATION` (forward evidence on the
     output law);
   - `design` to `CALIBRATION` (it plans the inverse problem).

   The guide's library-call lists name the new methods.
6. **No causal stage.** `probstat.causal` estimates effects from
   observational data under a causal graph. In a study the model is a
   simulator: an intervention `do(X_i = x)` is just a model run, and the
   input law is known. Adjustment, propensity, or backdoor estimators have no
   role there, so a `causal` hook would be decoration, not evidence.

## Consequences

- One root seed now reproduces propagation, the tail fit, mutual information,
  calibration, and design together. `report()` shows all of them, and
  `UQWorkflowRecord.record_stages` accepts the new kinds.
- The `extremes` rate model assumes independent runs. On a Sobol or LHS
  propagation sample, its intervals are references only. The stage records
  the sample design so a reader can see this.
- The POT tail probability has no interval, because `risk.extremes` provides
  delta intervals only for return levels. The empirical Wilson interval is
  the reference.
- KSG costs `O(n^2)` per input, so `max_points` defaults to 512.
- The design EIG is preposterior under the prior. It does not use the
  calibration posterior, because the library has no posterior-as-prior
  design problem. Increments are expected conditional gains given data at
  the baseline conditions.
- `visualization.study` draws no figures for the new stages. Their tables
  appear in the Markdown and HTML reports.

## Rejected alternatives

- Calling `self.propagate()` inside `extremes`: that would add a hidden
  propagation stage and use the propagation stream, which breaks the rule
  that a stage's randomness comes from its own stream.
- A delta-method interval for the POT tail probability written in the
  workflow layer: that is new numerics, and it belongs in `risk.extremes`
  with a registry entry if it is ever added.
- Using the greedy path value as the reported EIG: biased upward by
  selection (point 3).
