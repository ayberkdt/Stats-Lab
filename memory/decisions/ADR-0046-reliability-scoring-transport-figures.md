# ADR-0046 — Figures for adaptive reliability, forecast scoring, and transport

- Status: Accepted
- Date: 2026-10-12

## Context

ADR-0041 (multiple design points, multi-point FORM, mixture importance
sampling, AK-MCS), ADR-0042 (proper scores, PIT, coverage, reliability
diagrams, distribution shift), and ADR-0043 (exact and entropic transport,
Wasserstein distances, barycenters) added tested numbers and printing labs
but no figures. A design-point search, an AK-MCS run, a reliability table,
and a transport plan are hard to judge from numbers alone.

## Decision

1. Three new modules under the ADR-0024 rules (`@themed`, `new_axes`,
   semantic roles, bands in the parent hue, dash grammar, direct labels,
   computed conditional titles, no dual axes, no hard-coded colours):
   - `reliability_plots`: `plot_design_points`,
     `plot_ak_mcs_classification`, `plot_ak_mcs_convergence`.
   - `scoring_plots`: `plot_reliability_diagram`, `plot_score_comparison`,
     `plot_coverage`, `plot_distribution_shift`, and the helper `ks_gap`.
   - `transport_plots`: `plot_transport_lines`, `plot_plan_matrices`,
     `plot_sinkhorn_convergence`, `plot_quantile_wasserstein`,
     `plot_displacement_interpolation`, and the helpers `used_cells` and
     `crossing_pairs`.
   - A private `_layout` module wraps long computed titles and multi-part
     subtitles to the figure width (`wrapped_title_block`) and provides a
     canvas-coloured label halo; `theme.py` is unchanged.
   PIT histograms reuse `hierarchical_plots.plot_pit_histogram`.
2. Plot functions never call a limit-state model or re-solve a problem.
   The limit-state contour is drawn from a grid of `G` values the caller
   evaluates; importance samples and their failure flags are passed in.
   The only evaluations are of stored artefacts: the AK-MCS Gaussian
   process (to colour the population by predicted class and draw its zero
   contour), quantile functions of the laws handed to the W1 figure (with
   `quantile_wasserstein` for the title value), and closed-form normal
   densities. `AKMCSResult` does not store the population, so the lab
   redraws it from the run's child stream (`"ak-mcs"`, `"population"`) and
   prints the predicted-failure count next to the run's own as a check
   (45 = 45 for seed 1).
3. Colour roles where the library had no precedent:
   - Reliability: the exact surface `G = 0` is a reference (accent), solid
     because it is computed, with the failure domain filled in the same hue
     at 8% opacity; design points, FORM tangents (`FITTED`), training points,
     and the surrogate zero contour (`FITTED`) are model results (model
     hue); failed samples or predicted failures are data (observed hue);
     safe samples and the beta circle are muted context; points with the
     smallest learning value `U` are ringed in ink.
   - Scoring: observed frequencies, coverage rates, and mean scores use the
     observed hue; the diagonal of perfect reliability and the coverage
     identity are accent references; consistency bars, bin counts, and the
     reference-period ECDF are muted; the KS gap is an ink segment. Several
     forecasters on one coverage plot take `series[0]`, `series[3]`,
     `series[4]`, `series[5]` (the model hue implies a fitted curve and
     `series[2]` sits next to the porcelain accent, as in ADR-0044).
   - Transport: the source measure (first law) takes the observed hue and
     the target measure (second law) `series[3]`; plan entries and Sinkhorn
     runs ordered by `eps` use the palette's single-hue sequential ramp;
     displacement interpolants and the `W2` midpoint are model results;
     the 50/50 mixture is the muted contrast; the Sinkhorn tolerance is the
     accent. Because the graphite ramp ends light, subtitles say "brighter"
     there and "darker" in porcelain.
4. Titles are computed and conditional: the multi-point/one-point FORM
   ratio, AK-MCS calls and misclassified signs, the stopping value of
   `min U`, bins outside consistency bands with an over- or underconfidence
   verdict, the paired score difference with its interval (or "are
   indistinguishable"), the worst coverage gap and which forecasters stay
   consistent, the KS distance with a detected/not-detected verdict
   (the result must match the plotted samples), monotone or crossing
   transport lines, used plan cells, the Sinkhorn iteration ratio, quantile
   crossings, and whether the 50/50 mixture is bimodal.
5. The three labs save 15 figures to `examples/generated/` when Matplotlib
   is installed (they still run without it) and stay under 60 s.

## Consequences

- Every figure was rendered in porcelain and graphite and inspected.
  Fixes made after looking: titles and subtitles wrapped to the figure
  width (they had widened the canvas to 2000 px), the beta label moved
  inside the circle, design-point labels offset diagonally with a halo,
  the reliability-diagram top panel no longer forced square (it misaligned
  the shared x axis with the count panel), coverage labels moved outside
  the axes, the KS annotation moved to the empty side of the curves,
  Sinkhorn labels moved under each run's end point, and plan-mass labels
  staggered where atoms nearly coincide.
- Pf panels use a scaled axis ("x 1e-3") instead of Matplotlib's offset
  text, which collided with the subtitle.
- `plot_pit_histogram` uses pointwise bands, so with 20 bins one or two
  bins outside the band is common for a calibrated forecast; its title
  ("miscalibrated") is unchanged here because that module belongs to
  ADR-0044.
- Two-panel figures (reliability diagram, AK-MCS convergence, plan
  matrices) return a figure and take no `ax`, like `plot_optimal_design`.

## Rejected alternatives

- A dual axis for `Pf` and `min U`: the units differ; two panels share x.
- Accent for design points: the accent is reserved for references, and
  the exact limit state already uses it.
- A legend box on the AK-MCS map: the subtitle names the marks, as
  everywhere else.
- A diverging colormap for plan entries: mass is non-negative, so a
  single-hue sequential ramp is the honest encoding.
