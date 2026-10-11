# ADR-0045: Discrepancy calibration, scoring evidence, and stage figures in `UQStudy`

- Status: Accepted
- Date: 2026-10-12

## Context

Several capabilities of this release could not be reached from
`workflow.UQStudy` (ADR-0025, ADR-0038), or were reachable but had no figure:

- Kennedy-O'Hagan calibration (`inverse.DiscrepancyCalibration`, ADR-0039).
- Proper scores and PIT/coverage diagnostics (`validation.scoring`,
  `validation.calibration`, ADR-0042).
- Figures for the `design_points`, `active_reliability`, and `calibrate`
  stages.

ADR-0044 also recorded two gaps:

- `DesignRecommendation` kept no candidate list.
- `TailAnalysis` kept no exceedances.

## Decision

1. **A separate stage, `calibrate_with_discrepancy`**, instead of a
   `discrepancy=` option on `calibrate`.
   - Signature: `(experiments, quantity, noise, discrepancy, *, draws,
     warmup, chains, method, level, grid, grid_draws, correlation_warning,
     ratio_warning)`.
   - Stage kind `"discrepancy_calibration"`; returns `DiscrepancyAnalysis`.
     It holds the calibration problem, posterior, identifiability report,
     interval level, and grid predictions.
   - Why a separate stage:
     - The KOH problem calibrates one QoI. It takes `KnownNoise` or
       `LogLinearNoise`, not one `NoiseModel` per QoI.
     - It returns a different posterior type and offers HMC/NUTS.
     - An option would have made `calibrate`'s parameter and return types
       unions for every caller.
     - Keeping `calibrate` untouched also keeps it bit-identical. A test
       compares it with `CalibrationProblem.sample_posterior` on stream
       `calibration/0`.
   - The chains use the stage stream directly (`seed=stream`). A test
     reproduces them from `DiscrepancyCalibration` with
     `root.child("discrepancy_calibration", 0)`.
   - The summary records:
     - posterior mean, sd, and R-hat of every physical coordinate;
     - `max |rho(theta, psi)|` and the posterior/prior sd ratio of each
       `theta`;
     - the identifiability warning text (or "no confounding flagged");
     - the largest posterior discrepancy mean on the grid.
   - For a one-input discrepancy, `grid` (default 41) predictions across the
     experiments' range are stored for the figures. Their model runs are
     charged to the stage. Other conditions stay at the model defaults.
2. **`predict` follows the latest calibration.**
   - After `calibrate_with_discrepancy` it returns
     `DiscrepancyStudyPrediction`. The stage kind is still `"prediction"`.
   - The summary has the simulator-only interval, the bias-corrected
     interval, and the discrepancy mean per condition, at the stage `level`.
     With `include_noise` (log-linear noise, or one shared `KnownNoise`
     sd) it adds the observed interval.
   - The annotation is now `PosteriorPrediction | DiscrepancyStudyPrediction`.
3. **`design` after a discrepancy calibration** raises unless `noise=` is
   given, because `BayesianDesignProblem` has no discrepancy term. The
   baseline is the KOH experiments. The candidate list is now stored in
   `DesignRecommendation.candidates`. It is a new field with default `()`,
   so existing constructors still work. Candidates are copied to read-only
   mappings before scoring; the scores are unchanged.
4. **`validate` records scores without changing the verdict.**
   - CRPS per experiment (`ExperimentAssessment.crps`) and the mean over
     all observations.
   - Randomized ensemble PIT of every observation, on the new child stream
     `validation/k/pit`. The `assess_validation` streams are untouched.
   - The PIT mean, and the coverage at `coverage_levels` (default 50% and
     90%) with Clopper-Pearson 95% intervals.
   - The criteria and the returned `ValidationAssessment` are unchanged; a
     test compares them with a direct `assess_validation` call.
5. **`TailAnalysis.exceedances`** holds the outputs beyond the threshold in
   QoI units, least to most extreme. The tail figure draws them at their
   Hazen tail probabilities `(k - 1/2)/n`.
6. **Figures** (in `visualization.stage_plots`, rendered by
   `visualization.study`). They follow ADR-0024 and ADR-0044.
   - `plot_design_points`: the standard-normal plane.
     - Design points are model-hue dots with muted rays from the origin.
     - Each FORM hyperplane `alpha . u = alpha . u*` is a `FITTED`
       model-hue line.
     - The smallest-`beta` circle is the accent reference.
     - The figure uses stored results only and makes no limit-state calls.
       With more than two inputs it shows the two coordinates with the
       largest summed `alpha^2`.
   - `plot_active_learning`: two panels sharing the horizontal axis.
     - Top: the AK-MCS estimate with a two-standard-error band and, when it
       was run, crude Monte Carlo on the same population as the accent
       reference.
     - Bottom: `min U` with the stopping line.
   - `plot_posterior_marginals`: per-parameter density histograms of the
     pooled draws (observed hue) over the prior density (muted context). The
     title names the best-informed parameter as a sd ratio.
   - The discrepancy stage reuses `plot_discrepancy_predictions` and
     `plot_discrepancy_estimate`.
   - The design stage adds `plot_greedy_design` when the candidates set one
     condition to increasing values.
7. **Workflow mapping.** `WORKFLOW_GUIDE` maps `discrepancy_calibration` to
   `CALIBRATION`. Its library calls name `DiscrepancyCalibration` and
   `UQStudy.calibrate_with_discrepancy`.

## Consequences

- `examples/uq_study_complete_lab.py` runs every stage on one model in 17 s
  and 139 887 model runs. It writes one HTML report. The UX auditor reports
  0 HATA and 0 UYARI.
- Headline numbers: the plain calibration interval at `x = 2`,
  `[1.694, 1.738]`, misses the truth `1.594`. The bias-corrected KOH
  interval, `[1.548, 1.632]`, covers it. Two design points give
  multi-point FORM `2.00e-2` against single-point `8.70e-3`.
- A Python caller that type-checks `predict` must narrow the union.
- PIT values of replicates within one experiment share one predictive
  sample, so they are not independent cases. The coverage intervals assume
  independence and are references only.
- The validation stage still has no figure. A PIT or score figure belongs
  with the scoring figure module.

## Rejected alternatives

- `calibrate(..., discrepancy=...)`: it would make the parameter and return
  types unions and mix two noise-model vocabularies (point 1).
- Recomputing the discrepancy grid in the figure layer: figures must not
  call the model (ADR-0044).
- A coarse limit-state contour on the design-point plot: it costs model
  runs in the figure layer. The stored hyperplanes already show the
  geometry the estimate uses.
- Deterministic ECDF PIT values (`ExperimentAssessment.u_values`) for the
  coverage: with ties and a finite ensemble they are not exactly uniform
  under calibration. The randomized version is.
