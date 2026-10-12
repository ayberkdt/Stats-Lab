# Visual language of the figures

Every figure in the library follows one set of rules (ADR-0024), adapted from
the `sunumatik` chart-aesthetics guide. You can use the same rules for your
own figures through `probstat.visualization.theme`.

## The five rules

1. **Colour follows meaning, not order.** Observed or simulated data use the
   first series hue, model curves and predictions the second, and the accent
   is reserved for reference lines (exact values, thresholds, limits).
   Context data are muted grey, never a new hue.
2. **Uncertainty is drawn in its owner's colour.** A band takes the hue of the
   line it qualifies at low opacity without an outline; the line sits on the
   band and markers sit on the line.
3. **Dashes carry epistemic state.** Solid lines are computed or observed;
   `FITTED` dashes mark fitted or smoothed curves, `PROJECTED` dots mark
   extrapolation. A dash is never decoration.
4. **Titles state the finding.** The title is an assertion computed from the
   data ("X3 matters only through interactions"); a muted subtitle says what
   the marks are. Labels sit at line ends instead of in a legend box.
5. **Data ink dominates.** Horizontal hairline grid, no top or right spine, no
   shadows, glow, 3-D, or dual y-axes.

## Palettes

| Token | Porcelain (light, default) | Graphite (dark) |
|---|---|---|
| canvas | `#F6F4EF` | `#141519` |
| ink | `#23262B` | `#F1F0EB` |
| muted | `#656A73` | `#A6A9B1` |
| accent (reference lines) | `#44599C` | `#9BA9F0` |
| series 1 (observed) | `#178F6E` | `#2A9E83` |
| series 2 (model) | `#B0642F` | `#B67E2A` |
| series 3-6 | `#7D5BC4` `#8C8A1C` `#C04A8A` `#2F7FB5` | `#8A78CF` `#CC5F70` `#3F8BCB` `#6A9638` |

Both six-slot orders pass a categorical-palette validator: lightness band,
chroma floor, colour-vision-deficiency separation of adjacent slots,
normal-vision separation, and contrast against the canvas.

## Using the theme

```python
from probstat.visualization import GRAPHITE, chart_theme, save_figure
from probstat.visualization.theme import (
    active_palette, band, line, new_axes, points, reference_line, title_block,
)

with chart_theme():                      # or chart_theme(GRAPHITE)
    ax = new_axes()
    palette = active_palette()
    band(ax, xs, lower, upper, color=palette.model)
    line(ax, xs, mean, color=palette.model)
    points(ax, x_data, y_data, color=palette.observed)
    reference_line(ax, y=limit, label="design limit")
    title_block(ax, "Predictions stay below the limit", "Band: 95% interval")
    save_figure(ax.figure, "my-figure.png")
```

Set `PROBSTAT_CHART_THEME=graphite` to render every library figure in the dark
palette, for example to produce dark-mode documentation images.

## One uncertainty, six encodings

`examples/uncertainty_visualization_lab.py` simulates 400 demand paths from a
geometric Brownian motion (drift 0.05, volatility 0.25, start 100, horizon 4)
and shows the same uncertainty six ways. With seed 2026 the simulated
probability of exceeding the capacity 160 at the horizon is 0.212.

| Encoding | Function | Answers |
|---|---|---|
| Fan chart | `plot_fan_chart` | Where will the value most likely be over time? |
| Spaghetti | `plot_spaghetti` | What do individual futures look like? |
| Quantile dot plot | `plot_quantile_dots` | How many in 50 outcomes exceed the limit? (11 of 50 here) |
| ECDF with DKW band | `plot_ecdf_band` | How sure are we of the distribution itself? (+/-0.124 at n = 120) |
| Exceedance curve | `plot_exceedance_curves` | How fast does the tail risk fall? |
| Histogram with quantiles | `plot_output_distribution` | What is the typical range? |

Frequency framings such as the quantile dot plot let readers count instead of
judging areas, which supports better decisions under uncertainty
[REF-KAY-ETAL-2016]; fan charts and spaghetti plots are complementary views of
the same forecast [REF-SPIEGELHALTER-PEARSON-SHORT-2011]. The DKW band
[EQ-STAT-DKW-001] reminds the reader that an empirical distribution is itself
uncertain.

## Study reports

`probstat.visualization.study.write_study_html(study, "report.html")` writes a
self-contained page for a `UQStudy`: one section per stage with its numbers,
cost, stream seed, and figures embedded in both palettes so the page follows
the reader's light or dark preference.

## Analysis figures

Four modules draw the results of the causal, design, extremes, and
information packages under the same rules (ADR-0037). Each has a seeded lab
in `examples/` that prints its numbers and writes the figures.

| Module | Functions | Lab |
|---|---|---|
| `extremes_plots` | `plot_return_levels`, `plot_mean_residual_life`, `plot_tail_quantiles` | `extremes_lab.py` |
| `design_plots` | `plot_optimal_design`, `plot_greedy_design` | `experimental_design_lab.py` |
| `causal_plots` | `plot_effect_estimates`, `plot_covariate_balance`, `plot_e_value_contour` | `causal_lab.py` |
| `information_plots` | `plot_estimator_check`, `plot_entropy_decomposition` | `information_lab.py` |

The functions take result objects and never refit, so the caller decides,
for example, between delta-method and profile-likelihood return-level
intervals. The Q-Q band of `plot_tail_quantiles` maps the central interval of
each uniform order statistic through the model quantile function
[EQ-VIZ-QQBAND-001]; it is pointwise, so about 5% of points fall outside a
95% band by chance.

## Calibration, hierarchy, and study-stage figures

Three more modules follow the same rules (ADR-0044). Each function takes a
result object and never refits; the labs write every figure to
`examples/generated/`.

| Module | Function | Shows | Lab |
|---|---|---|---|
| `calibration_plots` | `plot_discrepancy_predictions` | observations, simulator-only band (`series[3]`, `FITTED`), bias-corrected band (model hue), truth (accent) | `discrepancy_calibration_lab.py` |
| `calibration_plots` | `plot_discrepancy_estimate` | posterior `delta(x)` with its mixture interval and the zero line | `discrepancy_calibration_lab.py` |
| `hierarchical_plots` | `plot_shrinkage` | raw `y_j +/- s_j` (muted) against exact posterior intervals of `theta_j` (model), `mu` (accent) | `hierarchical_lab.py` |
| `hierarchical_plots` | `plot_tau_posterior` | exact grid density of `tau` over a histogram of MCMC draws | `hierarchical_lab.py` |
| `hierarchical_plots` | `plot_mixture_fit`, `plot_mixture_selection` | 1-D mixture density and `FITTED` components over the data; BIC or AIC against `K` | `hierarchical_lab.py` |
| `hierarchical_plots` | `plot_predictive_check`, `plot_pit_histogram` | replicated test quantity with the observed value; PIT counts with a binomial band [EQ-VIZ-PITBAND-001] | `hierarchical_lab.py` |
| `stage_plots` | `plot_tail_analysis`, `plot_information_sensitivity`, `plot_design_path` | `UQStudy.extremes`, `information_sensitivity` (beside Sobol), and `design` | `uq_study_extensions_lab.py` |

The `theta_j` and `delta(x)` intervals are exact quantiles of Gaussian
mixtures over quadrature nodes or posterior draws [EQ-VIZ-MIXQ-001], not
mean +/- 2 sd. `save_study_figures` and the HTML report now draw the
extremes, information, and design stages as well.

## Adaptive reliability, forecast scoring, and transport figures

Three more modules follow the same rules (ADR-0046). Functions take result
objects and never call a model or re-solve a problem; limit-state contours
come from a grid the caller evaluates. The labs write every figure to
`examples/generated/`.

| Module | Function | Shows | Lab |
|---|---|---|---|
| `reliability_plots` | `plot_design_points` | design points and FORM tangents (model hue), exact `G = 0` (accent) over the failure domain, beta circle (muted), importance samples (failed: observed hue) | `adaptive_reliability_lab.py` |
| `reliability_plots` | `plot_ak_mcs_classification` | AK-MCS population by predicted class, training points (model hue), true and surrogate `G = 0`, rings at the lowest learning values `U` | `adaptive_reliability_lab.py` |
| `reliability_plots` | `plot_ak_mcs_convergence` | `Pf` with +/- 2 s.e. and `min U` against added points, two panels sharing x | `adaptive_reliability_lab.py` |
| `scoring_plots` | `plot_reliability_diagram` | observed frequency vs forecast probability, consistency bars (muted), diagonal (accent), counts per bin below | `forecast_calibration_lab.py` |
| `hierarchical_plots` | `plot_pit_histogram` (reused) | PIT counts of the ensemble and of an overconfident forecast with the binomial band | `forecast_calibration_lab.py` |
| `scoring_plots` | `plot_score_comparison` | mean CRPS per forecaster with intervals; title quotes the paired difference | `forecast_calibration_lab.py` |
| `scoring_plots` | `plot_coverage` | empirical vs nominal coverage with Clopper-Pearson or Wilson intervals, identity (accent) | `forecast_calibration_lab.py` |
| `scoring_plots` | `plot_distribution_shift` | two ECDFs (reference muted, new observed hue) with the KS gap in ink | `forecast_calibration_lab.py` |
| `transport_plots` | `plot_transport_lines`, `plot_plan_matrices` | a 1-D plan as weighted lines between two rows of atoms; exact and entropic plans as heatmaps on one sequential scale | `optimal_transport_lab.py` |
| `transport_plots` | `plot_sinkhorn_convergence` | marginal error per iteration (log-log), one line per `eps` on the sequential ramp, tolerance (accent) | `optimal_transport_lab.py` |
| `transport_plots` | `plot_quantile_wasserstein`, `plot_displacement_interpolation` | `W1` as the area between quantile functions [EQ-OT-W1D-001]; the `W2` geodesic of two normals against their 50/50 mixture [EQ-OT-BARY-001] | `optimal_transport_lab.py` |

The paired difference in `plot_score_comparison` is the right comparison
when forecasters are scored on the same cases: case difficulty cancels, so
its interval is much narrower than either whisker. Long computed titles are
wrapped to the figure width by `visualization._layout.wrapped_title_block`.
