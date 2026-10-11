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
| accent (reference lines) | `#44599C` | `#E8804A` |
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
