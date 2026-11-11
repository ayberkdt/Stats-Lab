# ADR-0052 — Figures for validation, missing data, and contingency tables; a simultaneous PIT band

- Status: Accepted
- Date: 2026-10-12

## Context

Three result types still had no figure:

- `UQStudy.validate` (ADR-0045) records CRPS, PIT mean, and coverage, but
  its section in the HTML report was a table only.
- Missing-data results (ADR-0047): `MissingPattern`, `MultipleImputation`,
  and `PooledEstimate`.
- Contingency results: `contingency_residuals` and `TwoByTwoMeasures`.

ADR-0046 also left a known flaw: `plot_pit_histogram` drew only a
pointwise binomial band. With 20 bins, about one bin of a calibrated
forecast falls outside it by chance, yet the title then said
"miscalibrated".

## Decision

1. **Validation figure** (`scoring_plots.plot_validation_predictions`,
   rendered by `visualization.study` for the `validation` stage).
   - One row per experiment:
     - central 50% and 90% intervals of the observable predictive sample
       as model-hue bands, with the predictive median as a model-hue tick;
     - observations as observed-hue dots;
     - the area metric at the right edge, in bold ink when it exceeds the
       tolerance.
   - The title is the assessment's own verdict. When the model is
     consistent, the title gives the count of observations inside the 90%
     bands. When it is inconsistent, the title names each failed
     criterion, for example "area metric exceeds 0.3 in 3 of 4
     experiments and pooled PIT rejects uniformity (p = 2.6e-06)".
   - A second panel shows empirical against nominal coverage with 95%
     Clopper-Pearson intervals and the identity as the accent.
     `visualization.study` rebuilds the `CoverageAtLevel` records from the
     stage summary (`coverage[90%]` and `coverage[90%].interval`), so
     neither the workflow nor the stored result changes. The figure's own
     inside-the-band count comes from the drawn sample quantiles. It can
     differ from the randomized-PIT coverage only through ties.
2. **Missing data** (new module `missing_plots`).
   - `plot_missing_pattern` draws patterns against variables.
     - Missing cells take the observed hue: the holes are the data
       property under study, and the task brief suggested this role.
     - Observed cells are muted at 22% opacity, separated by 2 pt canvas
       gaps.
     - Row counts are direct labels, and column labels state each missing
       fraction.
   - `plot_imputation_diagnostic` takes the incomplete rows and the
     imputation result. The result does not store which entries were
     missing, and guessing them from differences between data sets would
     be fragile.
     - For each incomplete variable it shows a strip of observed values
       (observed hue) above a strip of imputations from all `m` data sets
       (model hue: they are model draws).
     - Each strip has its 50% and 90% ranges and its median.
     - The subtitle notes that under MAR a shift is expected.
   - `plot_rubin_summary` shows the pooled estimate beside other analyses,
     with the truth as the accent reference.
     - The MI row adds an inner bar `Qbar +/- t sqrt(Ubar)` inside the full
       `Qbar +/- t sqrt(T)`, so the uncertainty that missingness adds is
       visible [EQ-MISS-007].
     - With a truth, the title counts the intervals that cover it. Without
       one, it reports the widening and the fraction of missing
       information (FMI).
3. **Contingency** (new module `contingency_plots`).
   - `plot_adjusted_residuals` draws a heatmap of Haberman residuals with
     cells beyond the threshold labelled. The subtitle gives Pearson `X^2`
     with its df and p-value.
   - `plot_effect_measures` draws two panels: RD on a linear axis, and RR
     and OR on a log axis.
     - The no-effect values 0 and 1 are accent references.
     - The title states whether every interval excludes no effect, whether
       none does, or which measures disagree.
     - An RR without a finite interval (a zero risk) is left out and named
       in the subtitle.
   - `theme.py` has no diverging scale. `diverging_ramp` and
     `diverging_colormap` build one from palette tokens: `series[2]`
     (violet) for fewer counts than expected, the neutral `rule` grey at
     zero, and `series[0]` (green) for more. This purple-green pair is
     colour-vision safe. Every RGB channel moves monotonically from the
     midpoint to each end in both palettes, so luminance is monotone along
     each arm. A test checks this rule ("two hues + a neutral midpoint").
     The scale is symmetric, from `-max(3, ceil|r|)` to `+max(3, ceil|r|)`.
4. **Simultaneous PIT band.**
   - New `simultaneous_binomial_band` applies the pointwise band at level
     `1 - (1 - confidence)/B` in every bin (Bonferroni,
     [EQ-VIZ-PITBAND-002]). By the union bound, all `B` bins of a
     calibrated forecast stay inside with probability at least
     `confidence`.
   - `plot_pit_histogram(..., simultaneous=True)` is the new default. It
     draws the lighter simultaneous band behind the darker pointwise band.
   - Title wording:
     - "evidence of miscalibration at the 5% level" (for 95% bands) only
       when the simultaneous band is exceeded;
     - "consistent with calibration" otherwise, together with the number
       of pointwise excursions and the number expected by chance.
   - `simultaneous=False` keeps the pointwise-only figure, and its title
     now reports excursions without a verdict. The signature is unchanged
     apart from the new keyword.
5. **Layout.** `wrapped_title_block` no longer breaks titles at hyphens
   (it had produced "within- imputation").

## Consequences

- Monte Carlo check with 4000 calibrated samples (seed 7): the chance that
  some bin leaves the band is
  - 0.53 (pointwise) against 0.036 (simultaneous) for n = 400 and 20 bins;
  - 0.36 against 0.040 for n = 100 and 10 bins.
  The old title called about half of all calibrated 20-bin histograms
  miscalibrated.
- The verdict is a 5% test, so it can still fire on a calibrated forecast.
  It does so in `forecast_calibration_lab.py`: the 20-member ensemble drawn
  from the ideal law has one bin of 2000 PIT values at 135 against the
  simultaneous limit 131 (KS distance 0.03), so its figure reports
  evidence of miscalibration. The wording says "evidence ... at the 5%
  level", not "is miscalibrated", for this reason.
- `examples/uq_study_complete_lab.py` runs in 17.7 s with 139,887 model
  runs (unchanged). Its HTML report has 11 figure pairs, now including the
  validation stage. The UX auditor reports 0 HATA and 0 UYARI.
- `examples/missing_data_lab.py` writes three figures and runs in about
  38 s (under 60 s). Its printed numbers are unchanged.
- No lab uses contingency tables. Their figures are exercised by the tests
  and were rendered from a synthetic 3 x 4 table for inspection.
- Every figure was rendered in porcelain and graphite and inspected. Fixes
  made after looking:
  - the area labels collided with the coverage panel's axis title (wider
    gap);
  - the pattern cell gaps were faint, because opacity also applied to the
    edges;
  - a value label sat on the true-value line, so labels now move above
    the dot, or to the left of the whisker, when the line would cross them;
  - the log axis had only two ticks (it now uses a 1-2-5 or finer
    mantissa set);
  - the panel gap of the imputation figure was too wide.

## Rejected alternatives

- Adding PIT values to `ValidationAssessment` or to the stage record so the
  figure can draw a PIT histogram: the workflow is out of scope here, and
  with 20 dependent replicate PIT values a histogram says less than the
  coverage panel.
- Matplotlib's `RdBu` or `coolwarm`: hard-coded colours outside the
  palette, and neither midpoint is the palette's neutral.
- A hue at the diverging midpoint, or the canvas colour as the midpoint:
  the first contradicts the rule, and the second would hide the cell grid
  in porcelain.
- A Kolmogorov-type simultaneous band, or one from simulated multinomial
  envelopes: tighter, but it needs simulation or an approximation.
  Bonferroni is exact as a bound, deterministic, and cheap.
- One forest with RD, RR, and OR on a single axis: the measures have
  different null values and scales.
