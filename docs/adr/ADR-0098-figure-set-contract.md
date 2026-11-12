# ADR-0098: One visual contract for every figure

- Status: Accepted
- Date: 2026-11-12

## Context

The 34 example labs write 105 figures, and each one passes the single-figure
audit (`datawiz.figur_denetle`) with no error. Consistency, however,
is a property of the set, not of a figure: a reader who learns in one figure
that a dashed accent line is the truth reads the next figure with that code.
A set audit of all lab figures (method under Verification) found that the
semantic theme of ADR-0024 fixes colour *roles* but not the encoding of the
entities and quantities that recur across figures:

- **Entities.** "truth" was a dashed accent line in the discrepancy figures
  and a solid muted line in the filtering figures; it was also labelled
  "true", "true value", "true delta(x)", and "true effect". The FORM picture
  of the central-model lab drew the limit state and the failed samples in the
  model hue and the design point in the observed hue, while the
  design-point map of the adaptive-reliability lab drew the limit state in
  the accent, the design points in the model hue, and failed samples in the
  observed hue. Fitted extreme-value curves were solid in one figure and
  dashed in another. MCMC chains and greedy design steps took the observed
  and model hues by position. The POT threshold was muted in one figure and
  in the accent in another.
- **Quantities.** Units were written as `Name [unit]` by `axis_labels` and
  as `Name (unit)` in seven figures, and parenthesized notes such as
  "(log scale)", "(lower is better)", or function notation (`P(X = k)`,
  `f(x)`, `Q(u)`) were read as units. One quantity had several titles:
  "Failure probability Pf", "Pf estimate [x 1e-3]", "Failure probability
  estimate"; "Training points added by active learning" and "Training runs
  of the true model"; "min U" and "Smallest U". `P(X = k)` and `P(L > l)`
  shared the key `P` with different "units" and scales. The categorical PMF
  put probability mass on the horizontal axis, every other PMF on the
  vertical one.
- **Compared figures.** The PIT histograms of two forecasters had count axes
  of 0 to 176 and 0 to 568; the six encodings of one forecast had six
  ranges; Sobol figures of the same function had different share scales.
- **Names and orders.** Reliability method codes appeared as "MC", "IS",
  "Subset" in one figure and as words in another; the Ishigami inputs were
  `X1` in one figure and `x1` in the others; the information-sensitivity
  rows used a different input order from the Sobol figures of the same
  study.
- **Palette.** One lab opened `chart_theme()` with the default light palette,
  so its "dark" copy was light.

## Decision

1. **The contract lives in the theme.** `probstat.visualization.theme` gains
   the tables that the plotting functions and the labs read instead of
   choosing colours and titles locally:
   - `ENTITY_STYLES`: entity name to `EntityStyle(role, marker, dashes,
     width)`. Roles are those of ADR-0024 (`observed`, `model`, `reference`,
     `context`) plus the four free slots (`series-3` to `series-6`) for
     members of one family. `entity_style`, `entity_colour`, and
     `role_colour` resolve them in the active palette; a trailing
     parenthesized qualifier is ignored (`"calibration (n = 300)"`), and an
     unknown entity is an error rather than a new colour.
   - `QUANTITIES`: key to `Quantity(name, unit, axis)` with one title, one
     unit spelling, and one axis role; `axis_title(key)` returns the title.
     Two quantities are declared `free`: the failure probability (value axis
     of the horizontal method dot plots, vertical axis of the AK-MCS
     convergence panels) and a calibrated parameter (vertical in its trace,
     horizontal in its posterior histogram).
   - `CATEGORY_ORDERS` (forecasters, chains, reliability methods, sampling
     designs) and `METHOD_NAMES` (reliability method codes written out).
   - Line widths by role: `LINE_WIDTH` 2 pt, `CONTEXT_WIDTH` 1.4 pt,
     `PATH_WIDTH` 0.7 pt, `REFERENCE_WIDTH` 1.2 pt, and the `GUIDE` dots of
     construction lines.
2. **Drawing by entity.** `line`, `points`, `band`, and `reference_line`
   accept `entity=`; the style comes from the table and the entity name
   becomes the artist label, so the set audit sees which entity each mark
   is. Every library function that draws a recurring entity uses it.
3. **One axis-title grammar.** A title is a quantity name in sentence case
   (`axis_labels` applies `sentence_case`, which leaves symbols and Greek
   names alone), followed by its unit in brackets. Function notation moves
   to the subtitle; a log axis is stated by its ticks and by the text
   alternative of ADR-0078, never as "(log scale)".
4. **Ordered members and compared figures.** `ordered_colours(n)` gives
   ordered members (greedy steps, decreasing `eps`) a fixed depth of the
   sequential ramp per index, so step 2 has one colour in every figure.
   `share_limits` gives compared figures one range (the forecast lab's two
   PIT histograms; the uncertainty lab's paths and its four terminal
   views). Sobol figures and the Sobol panel of the information-sensitivity
   figure use the common 0-1 share scale and the total-effect row order.
5. **One picture of standard-normal space.** `plot_form_geometry` uses the
   encoding of `plot_design_points`, and the two AK-MCS convergence figures
   (`plot_ak_mcs_convergence`, `plot_active_learning`) share one panel
   function, `reliability_plots.learning_panels`, with the number of model
   evaluations as horizontal axis.
6. **The contract as files.** `probstat.visualization.figure_set` writes the
   tables in the datawiz set format, one document per palette with every
   colour resolved: `docs/figures/stats-lab.seti.json` (light) and
   `docs/figures/stats-lab-dark.seti.json` (dark), generated by
   `tools/generate_figure_set_contract.py`. Units are bracketed
   (`etiket_bicimi = "koseli"`), decimals use a point.
7. **Gallery.** The README gallery copies in `docs/images/` (ten light and
   dark pairs, file names unchanged) are regenerated from the labs.

## Consequences

- A new figure that draws a recurring entity or quantity takes it from the
  theme, and a new recurring entity is added to the table, the contract
  files are regenerated, and `tests/test_figure_set_contract.py` keeps
  both in step. Lab-specific entities that appear in one figure only keep
  role colours by position.
- The role-based palette means one hue still serves several entities of the
  same role in different figures (the model hue is the GEV fit in one
  figure and the filtered mean in another). The set audit reports this as
  `renk-anlam` warnings; it is the intended reading of ADR-0024 (the hue
  says "model", the label says which model), not a defect.
- Some figures changed visibly: the FORM geometry, the AK-MCS panels (plain
  failure-probability ticks instead of a `x 1e-3` unit), the categorical
  PMF (vertical bars), the chain and greedy-step colours, the total-effect
  marker (diamond), the shared ranges of the uncertainty lab (the quantile
  dot plot now spans the full demand range of the other views), and every
  axis title listed in Context. Seeded numbers are unchanged.
- `plot_effect_estimates` labels its reference line "truth" by default.

## Verification

The set audit runs every lab with `save_figure` patched to run the
single-figure audit and to write the datawiz trace (`.dw.json`) of each
figure, once with `PROBSTAT_CHART_THEME=porcelain` and once with
`graphite`; then `python -m datawiz.tutarlilik <traces> <contract>` checks the
traces against each other and against the contract file of the palette.
There are 105 saved figures and 104 traces (two labs write a figure of the
same name, `cantilever-01-propagation-deflection.png`). The four reports are
kept in `docs/figures/reports/` (`set-report-before-light.txt`,
`set-report-before-dark.txt`, `set-report-after-light.txt`,
`set-report-after-dark.txt`), as written by the checker; both "before"
reports check the traces of the unchanged figures against the final
contract.

| Set report (HATA / UYARI) | light | dark |
|---|---|---|
| Before, figures against each other | 9 / 17 | 9 / 17 |
| Before, against the contract | 18 / 17 | 19 / 17 |
| After, against the contract | 0 / 13 | 0 / 13 |

Before, against the contract, by rule (light; dark in parentheses where it
differs): `set-renk` 7 (8), `nicelik-eksen` 3, `nicelik-olcek` 2,
`set-etiket-bicimi` 2, `etiket-bicimi` 1, `nicelik-birim` 1, `set-eksen` 1,
`varlik-renk` 1; warnings `renk-anlam` 5, `nicelik-birim` 5, `ad-yazimi`,
`cerceve`, `cizgi-kalinligi`, `harf-bicimi`, `izgara`, `kategori-sirasi`,
`varlik-cizgi` 1 each. The extra dark error is the lab that ignored the
dark palette.

After, in both palettes: no error; warnings `renk-anlam` 9 (role colours
shared by entities of one role, see Consequences), `cizgi-kalinligi` 1 (the
four role widths: a figure of overlapping paths has 0.7 pt as its most
common width), `izgara` 1 (log axes with power-of-ten tick labels and
multiplier ticks are read as category axes by the checker; the quantile dot
plot, the missing-pattern map, and the transport line plot carry no grid by
design), `cerceve` 1 (the upper panel of a two-panel figure without its
shared bottom spine; the quantile dot plot without a count axis), and
`nicelik-birim` 1 (the loss of the central-model study in kEUR and the loss
of the cantilever study, which has no currency). Every figure has 0
single-figure errors in both palettes, including the ten gallery pairs.

Without the contract, the after set reports the two declared free-axis
quantities as `nicelik-eksen` errors; the contract records them as
deliberate (`eksen = "serbest"`).

The PNGs were compared side by side in both palettes; three problems that
the checker cannot see were fixed on sight: the smoothed state (purple) and
the dashed truth (accent blue) were hard to tell apart in the dark palette,
so the smoothed state moved to slot 4; the information-sensitivity figure
named the Sobol indices "S" and "S_T" while the Sobol figure used "S_i" and
"S_Ti"; and the greedy-step labels needed the ink-mixed label colour of the
profile figures to stay legible on the light canvas.

`tests/test_figure_set_contract.py` checks that both files are valid datawiz
sets, equal the tables of the theme, give every entity the colour of its
role in each palette, write one title and bracketed unit per quantity, and
that no plotting module holds a colour literal or an axis title with a scale
note; one line per palette is rendered to check that `entity=` draws the
contract colour, width, and label.
