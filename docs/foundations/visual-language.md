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

The numbers follow the sunumatik chart-aesthetics reference
(`skills/create-scientific-visuals/references/chart-aesthetics.md`):
Matplotlib scales dash patterns by line width, so `FITTED (5, 3.5)`,
`PROJECTED (1, 3.5)`, and `SIMULATED (8, 3, 1.5, 3)` at the 2 pt data line
give exactly its 10-7, 2-7, and 16-6-3-6 grammar; reference lines are 0.6x
the data width (its 2 px vs 3.5 px); every band is capped at 0.18 opacity
(`MAX_BAND_ALPHA`, nested inner bands included); markers carry a
canvas-colour rim. Continuous curves can be drawn with Fritsch-Carlson
monotone cubic smoothing (`line(..., smooth=True)`, and `band(...,
smooth=True)` so the band follows its line) [REF-FRITSCH-CARLSON-1980], and
the primary observed series may carry one fading gradient area
(`gradient_area`, at most 0.26 at the line).

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

xs = [0.0, 1.0, 2.0, 3.0, 4.0]
mean = [1.0, 1.4, 1.7, 1.9, 2.0]
lower = [m - 0.3 for m in mean]
upper = [m + 0.3 for m in mean]
x_data, y_data, limit = [0.5, 1.5, 2.5, 3.5], [1.1, 1.6, 1.8, 2.0], 2.6

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

## One contract for the whole set

A figure can follow every rule above and the set of figures can still
contradict itself: "truth" in the accent in one figure and in muted ink in
the next, "Failure probability Pf" on one axis and "Pf estimate [x 1e-3]" on
another. The figure-set contract (ADR-0098) fixes what must not vary between
figures, and the plotting functions and the labs read it instead of choosing
colours and axis titles locally.

* **Entities.** `theme.ENTITY_STYLES` maps every recurring entity to a colour
  role, a marker, a dash pattern, and a line width: `truth`, `closed form`,
  `threshold`, and the `Monte Carlo reference` are reference lines (accent,
  dashed); `first-order index` is a data mark and `total-effect index` a
  model mark (diamond); `FORM` is a dashed model line; `chain 1` to
  `chain 4` take the four free slots, never the observed or model hue.
  `line`, `points`, `band`, and `reference_line` take `entity=` and draw
  the entity with its fixed style and its name as the artist label; an
  unknown name is an error, not a new colour.
* **Quantities.** `theme.QUANTITIES` gives each recurring quantity one title
  and one axis role, and `axis_title(key)` returns it: "Failure
  probability", "Number of model evaluations", "Kolmogorov distance",
  "Probability mass", "Exceedance probability", "Share of output variance".
  A unit is always written in brackets, `Name [unit]`; function notation
  such as `P(X = k)` or `f(x)` belongs in the subtitle, and a log axis is
  shown by its ticks and stated in the text alternative, never as "(log
  scale)" in the title. `axis_labels` writes titles in sentence case.
* **Line weights.** Four widths by role: data and model lines 2 pt
  (`LINE_WIDTH`), context lines 1.4 pt (`CONTEXT_WIDTH`), many overlapping
  paths 0.7 pt (`PATH_WIDTH`), reference lines 1.2 pt (`REFERENCE_WIDTH`).
* **Ordered members.** Steps and decreasing regularization levels are not
  categories: `ordered_colours(n)` gives member `k` a fixed depth of the
  sequential ramp, so step 2 has one colour in every figure.
* **Compared figures.** Figures read against each other share their axis
  range through `share_limits`: the two PIT histograms of the forecast lab,
  and the six encodings of one forecast; every Sobol figure uses the 0-1
  share scale.
* **Names and orders.** `theme.METHOD_NAMES` writes reliability method codes
  out ("MC" is "Monte Carlo"), and `theme.CATEGORY_ORDERS` fixes the order
  of forecasters, chains, methods, and sampling designs.

```python
from probstat.visualization.theme import axis_title, entity_style

axis_title("failure_probability")  # 'Failure probability'
entity_style("truth").role  # 'reference'
entity_style("calibration (n = 300)").role  # 'context'
```

`docs/figures/stats-lab.seti.json` and `docs/figures/stats-lab-dark.seti.json`
hold the same tables in the datawiz set format, with every colour resolved in
the light and the dark palette; `tools/generate_figure_set_contract.py`
writes them and `tests/test_figure_set_contract.py` checks that they match
the theme. The set audit runs every lab, records a trace of each figure, and
checks the traces against the contract and against each other.

## Foundation figures

`probstat.visualization.foundation_plots` (named `foundations` before
ADR-0096) draws the definitions of the probability and statistics guides. All
of its functions are re-exported by `probstat.visualization`.

| Functions | Shows | Lab |
|---|---|---|
| `plot_bernoulli_pmf`, `plot_integer_pmf`, `plot_categorical_pmf`, `plot_waiting_count_pmfs` | Bernoulli masses as two bars; integer masses as lollipops on an explicit window; categories as vertical bars in declaration order, with no numeric category axis; two waiting-count PMFs as lines | `generate_foundation_figures.py` |
| `plot_finite_measure`, `plot_random_variable_pmf`, `plot_joint_pmf` | outcome masses with a highlighted event, an induced PMF, a joint PMF as a single-hue matrix | `generate_foundation_figures.py` |
| `plot_continuous_functions`, `plot_normal_pdf`, `plot_characteristic_function` | density above CDF and survival on a shared x-axis; the central one-sigma mass; real and imaginary parts | `generate_foundation_figures.py` |
| `plot_empirical_cdf`, `plot_fitted_cdf` | the right-continuous ECDF; ECDF against a fitted model CDF | `generate_foundation_figures.py`, `statistical_fitting_lab.py` |
| `plot_monte_carlo_convergence`, `plot_efficiency_comparison`, `plot_refinement_convergence` | nested estimates with their confidence band; cost-normalized efficiency against a baseline; three-level refinement against Richardson extrapolation | `basic_monte_carlo_lab.py`, `advanced_monte_carlo_lab.py`, `generate_foundation_figures.py` |

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

Both reports take `locale=` (`"en"` by default, `"tr"` for Turkish); the
text comes from the message catalogue `probstat.foundations.messages`, and
Turkish numbers use the decimal comma (ADR-0085).

## Interactive explanations

`probstat.visualization.interactive` writes offline HTML explainers: one
file each, inline SVG in the same palette tokens (light and dark), native
labelled controls, and an `aria-live` sentence with the current numbers.
Every number is computed by the library in Python and embedded as JSON; the
page script only selects among precomputed states (ADR-0085).

| Function | Explains |
|---|---|
| `clt_explainer` | exact distance of the standardized exponential mean to the normal law against the Berry-Esseen bound [EQ-LIM-001], [EQ-LIM-003] |
| `bayesian_updating_explainer` | Beta-Binomial prior, likelihood, posterior; credible or Wilson interval [EQ-REL-DATA-001], [EQ-INF-CI-WILSON-001] |
| `monte_carlo_error_explainer` | Monte Carlo estimates of `pi` and their 95% bands against `N` for four seeds [EQ-MC-ROOTN-001] |

`examples/interactive_explainers_lab.py` writes all three in English and
Turkish to `examples/generated/`.

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
| `calibration_plots` | `plot_discrepancy_predictions` | observations, simulator-only band (muted, `FITTED`), bias-corrected band (model hue), truth (accent) | `discrepancy_calibration_lab.py` |
| `calibration_plots` | `plot_discrepancy_estimate` | posterior `delta(x)` with its mixture interval and the zero line | `discrepancy_calibration_lab.py` |
| `hierarchical_plots` | `plot_shrinkage` | raw `y_j +/- s_j` (muted) against exact posterior intervals of `theta_j` (model), `mu` (accent) | `hierarchical_lab.py` |
| `hierarchical_plots` | `plot_tau_posterior` | exact grid density of `tau` over a histogram of MCMC draws | `hierarchical_lab.py` |
| `hierarchical_plots` | `plot_mixture_fit`, `plot_mixture_selection` | 1-D mixture density and `FITTED` components over the data; BIC or AIC against `K` | `hierarchical_lab.py` |
| `hierarchical_plots` | `plot_predictive_check`, `plot_pit_histogram` | replicated test quantity with the observed value; PIT counts with a binomial band [EQ-VIZ-PITBAND-001] | `hierarchical_lab.py` |
| `stage_plots` | `plot_tail_analysis`, `plot_information_sensitivity`, `plot_design_path` | `UQStudy.extremes`, `information_sensitivity` (beside Sobol), and `design` | `uq_study_extensions_lab.py` |
| `stage_plots` | `plot_design_points`, `plot_active_learning`, `plot_posterior_marginals` | design points with FORM hyperplanes (model, `FITTED`) and the smallest-`beta` circle (accent); AK-MCS estimate and `min U` in two panels; posterior histograms (observed) over prior densities (muted), ADR-0045 | `uq_study_complete_lab.py` |

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

## Validation, missing-data, and contingency figures

The last stages and packages without figures get them under the same rules
(ADR-0052). Functions take result objects and never refit.

| Module | Function | Shows | Lab |
|---|---|---|---|
| `scoring_plots` | `plot_validation_predictions` | per experiment, central 50% and 90% predictive bands (model hue) with the median tick, observations (observed hue), area metric per row (bold when over tolerance); coverage against nominal with the identity (accent); the title is the assessment's verdict | `uq_study_complete_lab.py` (validation stage of the HTML report) |
| `missing_plots` | `plot_missing_pattern` | patterns x variables, missing cells in the observed hue, observed cells muted, row counts as direct labels | `missing_data_lab.py` |
| `missing_plots` | `plot_imputation_diagnostic` | observed (observed hue) against imputed values of all `m` data sets (model hue) per incomplete variable, with 50%/90% ranges | `missing_data_lab.py` |
| `missing_plots` | `plot_rubin_summary` | MI, complete-case, mean-imputation, and EM intervals; the MI row shows the `Ubar`-only interval inside Rubin's total [EQ-MISS-007]; truth (accent) | `missing_data_lab.py` |
| `contingency_plots` | `plot_adjusted_residuals` | Haberman adjusted residuals on a diverging scale (`series[2]`, neutral `rule`, `series[0]`), cells beyond +/-2 labelled, `X^2` in the subtitle [EQ-INF-CT-003] | tests |
| `contingency_plots` | `plot_effect_measures` | RD (linear) and RR, OR (log) with their intervals, no effect (accent) [EQ-INF-CT-001] | tests |
| `hierarchical_plots` | `plot_pit_histogram` (changed) | adds a Bonferroni band that holds for all bins at once [EQ-VIZ-PITBAND-002]; evidence of miscalibration only when that band is exceeded | `forecast_calibration_lab.py`, `hierarchical_lab.py` |

Diverging scales follow "two hues and a neutral midpoint": the residual
heatmap uses the purple-green pair of each palette with the gridline grey at
zero, never a hue for "no departure". A pointwise PIT band is exceeded
somewhere by about half of all calibrated 20-bin histograms of 400 values;
the simultaneous band by fewer than 5%.

## Accessibility

Every figure written by `save_figure` carries a text alternative (ADR-0078).
`probstat.visualization.accessibility.describe_figure` composes it from what
the figure already states: the assertion title, the subtitle naming the
marks, and each axis with its label, displayed range, and log scale. It is
stored in the file itself (PNG and SVG `Title` and `Description`, PDF
`Title` and `Subject`) [REF-PNG-SPEC-2003], and the HTML study report uses
the same text as each image's `alt` attribute. Pass `description=` to
`save_figure` to replace it. A clear assertion title therefore also makes a
clear description.

```python
from probstat.visualization.accessibility import describe_figure
from probstat.visualization.theme import axis_labels

with chart_theme():
    ax = new_axes()
    line(ax, [1.0, 10.0, 100.0], [0.5, 0.25, 0.125], color=palette.observed)
    ax.set_xscale("log")
    title_block(ax, "error halves per decade", "Line: absolute error")
    axis_labels(ax, "Samples n", "Error")
print(describe_figure(ax.figure))
# Error halves per decade. Line: absolute error. Horizontal axis Samples n,
# 0.7943 to 125.9 (log scale); vertical axis Error, 0.1062 to 0.5188.
```

Contrast is measured with the WCAG 2.1 relative luminance and contrast
ratio [EQ-VIZ-WCAG-001], [EQ-VIZ-WCAG-002], [REF-W3C-WCAG21-2018]:

| Token on the canvas | Porcelain | Graphite | Level AA |
|---|---|---|---|
| ink | 13.81 | 15.99 | 4.5 (text) |
| muted | 4.95 | 7.76 | 4.5 (text) |
| accent | 6.06 | 8.10 | 4.5 (text) |
| series hues | 3.31 to 4.59 | 4.72 to 5.49 | 3 (graphics) |

`tests/test_visualization_accessibility.py` enforces these thresholds. The
series hues meet the 3:1 level for lines and markers in both palettes; as
text (direct labels at 8.5 pt) the light palette's hues fall below 4.5:1, a
known gap recorded in ADR-0078.

The HTML study report declares `lang="en"`, starts with a skip link to the
stage results, and uses `header`, `nav`, `main`, and `footer` landmarks
[REF-WHATWG-HTML].

## Common pitfalls

- **Matplotlib is optional.** Without it, `chart_theme` and the plot
  functions raise `ImportError` with install advice; check with
  `theme.matplotlib_available()`. HTML study reports still work and omit the
  figures.
- **The default theme is light.** `chart_theme()` without an argument always
  applies `PORCELAIN`. `PROBSTAT_CHART_THEME=graphite` changes only the
  palette that library figures use outside an explicit theme, and it is read
  once, when `probstat.visualization.theme` is first imported; set it before
  that import.
- **Bands are capped.** `band(..., alpha=...)` never exceeds
  `MAX_BAND_ALPHA = 0.18`; a larger value is lowered without a warning.
- **The title becomes the text alternative.** `title_block` upper-cases the
  first letter, and `save_figure` stores a description built from the title,
  subtitle, and axes. A vague title gives a vague description; pass
  `description=` to replace it.

```python
from probstat.visualization import PORCELAIN
from probstat.visualization.theme import matplotlib_available

matplotlib_available()           # True
with chart_theme() as chosen:
    ax = new_axes()
    fill = band(ax, xs, lower, upper, color=chosen.model, alpha=0.5)
    title_block(ax, "bands stay light", "Band: requested opacity 0.5")
chosen is PORCELAIN              # True
fill.get_alpha()                 # 0.18
ax.get_title(loc="left")         # 'Bands stay light'
```
