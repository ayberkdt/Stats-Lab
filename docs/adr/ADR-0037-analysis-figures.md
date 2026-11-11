# ADR-0037 — Figures and labs for causal, design, extremes, and information

- Status: Accepted
- Date: 2026-10-12

## Context

ADR-0034 (causal inference), ADR-0035 (experimental design), and ADR-0036
(extreme value analysis and information theory) added four packages with
tested numbers but no figures and no runnable labs. Their results are hard to
read as tables alone: a return level is meaningless without its interval and
the record length, a D-optimal design is certified by a variance function,
and an E-value is a curve of confounder strengths.

## Decision

1. Four visualization modules, one per domain, following ADR-0024 exactly
   (`@themed`, `new_axes`, semantic roles, bands in the parent hue, dash
   grammar, direct labels, computed assertion titles):
   - `extremes_plots`: `plot_return_levels` (log return period, `FITTED`
     inside the record and `PROJECTED` beyond it, interval band, record
     length as a muted reference, truth in the accent),
     `plot_mean_residual_life` (band, chosen threshold, implied GPD line),
     `plot_tail_quantiles` (Q-Q with a pointwise order-statistic band,
     EQ-VIZ-QQBAND-001).
   - `design_plots`: `plot_optimal_design` (weight lollipops above `d(x)` with
     the equivalence bound `p`; two panels instead of a dual axis),
     `plot_greedy_design` (EIG gain per candidate and step).
   - `causal_plots`: `plot_effect_estimates` (forest via
     `plot_method_comparison`), `plot_covariate_balance` (Love-plot dumbbell),
     `plot_e_value_contour` (explain-away curve of EQ-CAUSAL-BIAS-001).
   - `information_plots`: `plot_estimator_check` (replicate mean +/- SD
     against a closed-form curve), `plot_entropy_decomposition`.
2. Plot functions take domain result objects (`ReturnLevel`,
   `MeanResidualLifePoint`, `GPDFit`, `ApproximateDesign`,
   `GreedyDesignResult`, `BalanceTable`, `EValue`) and never refit; the caller
   chooses delta or profile intervals, condensing, and estimators.
3. Every title is computed and conditional: it counts intervals covering the
   truth, points outside a band, or thresholds leaving the GPD line, and
   switches wording when the data contradict the default claim (for example
   "D-efficiency is at least 22%" instead of "is D-optimal").
4. Labs `extremes_lab.py`, `experimental_design_lab.py`, `causal_lab.py`,
   and `information_lab.py` use `RandomStream(2026)`, print their key numbers,
   save to `examples/generated/`, and run in under 25 s each.
5. Modules are imported by path (as `statistics_plots` and `uq`), not
   re-exported from `probstat.visualization`.

## Consequences

- Each figure was rendered in both palettes and inspected; label collisions
  were removed by routing reference labels through `direct_labels`.
- In `GRAPHITE` the model hue (amber) and the accent (ember) are close, so a
  truth curve next to a fitted curve is separated mainly by dash and label.
- Dense Q-Q plots (more than 150 points) drop the canvas rim on markers,
  which otherwise paints over the data.

## Rejected alternatives

- Delta-method bands by default for return levels: symmetric Wald limits go
  below the 10-year level at long periods; the lab uses profile intervals.
- A least-squares line on the mean-residual-life plot: the plot would decide
  a statistic; the GPD-implied line comes from the caller's fit instead.
- Colouring treatment-effect intervals by whether they cover the truth: it
  adds a hue with a judgement attached; the title carries the count.
