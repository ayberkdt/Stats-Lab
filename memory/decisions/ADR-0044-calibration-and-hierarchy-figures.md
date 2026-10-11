# ADR-0044 — Figures for discrepancy calibration, hierarchical models, and study stages

- Status: Accepted
- Date: 2026-10-12

## Context

ADR-0039 (Kennedy-O'Hagan discrepancy), ADR-0040 (EM, Gaussian mixtures,
hierarchical models, posterior predictive checks) and ADR-0038 (the
`extremes`, `information_sensitivity`, and `design` stages of `UQStudy`)
added tested numbers without figures. `visualization.study` drew nothing for
the three new stages, so their sections in the HTML report were tables only.

## Decision

1. Three new modules under the ADR-0024 rules (`@themed`, `new_axes`,
   semantic roles, bands in the parent hue, dash grammar, `direct_labels`,
   computed assertion titles, no dual axes, no hard-coded colours):
   - `calibration_plots`: `plot_discrepancy_predictions`,
     `plot_discrepancy_estimate`, and the helper
     `gaussian_mixture_interval` (EQ-VIZ-MIXQ-001).
   - `hierarchical_plots`: `plot_shrinkage`, `plot_tau_posterior`,
     `plot_mixture_fit`, `plot_mixture_selection`, `plot_predictive_check`,
     `plot_pit_histogram`, and the helpers `theta_intervals`, `tau_density`,
     `binomial_band` (EQ-VIZ-PITBAND-001).
   - `stage_plots`: `plot_tail_analysis`, `plot_information_sensitivity`,
     `plot_design_path`; `visualization.study` renders them for the
     `extremes`, `information`, and `design` stages (light and dark images in
     the HTML report, PNGs in `save_study_figures`).
2. Colour roles where the library had no precedent:
   - Discrepancy: observations take the observed hue; the bias-corrected
     prediction `eta + delta` is the model's answer and takes the model hue
     (solid line on its band); the simulator-only prediction takes
     `series[3]` (olive in porcelain, rose in graphite) with the `FITTED`
     dash. `series[2]` (violet) was tried first and rejected because it sits
     too close to the porcelain accent used for the truth.
   - Shrinkage: raw estimates are context (muted), posterior intervals use
     the model hue, the pooled mean is the accent reference line.
   - Mixtures and checks: data and simulated replicates use the observed
     hue, fitted densities the model hue; the observed test statistic, the
     uniform PIT level, and the selected `K` are accent references. The
     binomial PIT band is muted context, like the white-noise band of the
     correlogram.
   - Information: mutual information and the first-order Sobol index are
     both one-input dependence measures and share the observed hue; the
     total-effect index keeps the model hue it has in `plot_sobol_indices`.
     Two panels on the same rows replace a dual axis.
   - Design: the greedy EIG path is the observed hue, every candidate's EIG
     at each step is muted, and the independent re-estimate is a model-hue
     dot with a two-standard-error whisker.
3. Intervals are exact, not mean +/- 2 sd: `theta_j` and `delta(x)` are
   Gaussian given `tau` (or a parameter draw), so their marginal intervals
   are mixture quantiles over the quadrature nodes of the `tau` grid or over
   the posterior draws (EQ-VIZ-MIXQ-001). The `tau` density is the stored
   log density normalized by its trapezoid integral.
4. Functions take result objects and never refit. The tail figure evaluates
   the stored POT fit (`return_level` on a period grid) and reads empirical
   quantiles from the stage summary; it never touches the model.
5. The three labs save their figures to `examples/generated/` when
   Matplotlib is installed (they still run without it); the study lab also
   writes one HTML report per study.

## Consequences

- Every figure was rendered in both palettes and inspected; reference-line
  labels go through `reference_line` (outside the axes) so they never sit on
  data, and titles that start with an input name are prefixed with "Input"
  because `title_block` capitalizes the first character.
- `DesignRecommendation` stores no candidate list, so the design figure
  cannot place candidates on their condition axis; it shows candidate EIG
  values as a strip per step instead of `plot_greedy_design`'s curves.
- `plot_return_levels` needs block maxima, which `TailAnalysis` does not
  keep, so the tail stage gets its own exceedance-probability figure.
- With in-sample (not leave-one-out) replicates the PIT histogram is pulled
  toward 1/2, and with very few values (eight schools) the binomial band is
  wide; the subtitle reports the KS distance instead of a verdict.

## Rejected alternatives

- Normal-approximation intervals (`mean +/- 1.96 sd`) for `theta_j`: the
  marginal is a scale mixture with heavier tails than a normal.
- Drawing the simulator-only prediction in muted ink: it is a prediction the
  reader must compare, not context.
- A single panel with mutual information and Sobol indices on twin axes:
  the units differ (nats against variance shares); ADR-0024 forbids dual
  axes.
