# ADR-0024 — Semantic, publication-grade chart theme

- Status: Accepted
- Date: 2026-10-11

## Context

Figures used Matplotlib defaults with ad hoc colours per function (Tableau
blue/orange/red), legends boxed over data, topic-label titles ("Reliability
method comparison"), full box spines, a dual y-axis for PDF and CDF, and bar
charts for point estimates with error bars. They read as classroom output and
did not share any visual grammar. The owner asked for figures designed with
the techniques of the `sunumatik` presentation system.

## Decision

1. `visualization.theme` encodes the `sunumatik` chart-aesthetics rules:
   - every hue has one meaning: observed/simulated data use series slot 1,
     model/fit/prediction slot 2, and the accent is reserved for reference
     lines (exact values, thresholds, limits);
   - uncertainty bands take the parent hue at low opacity and no stroke; the
     line sits on the band and markers (canvas-rimmed) on the line;
   - dash patterns carry epistemic state (fitted, projected, simulated);
   - assertion titles plus a muted subtitle that names the marks; direct
     labels at line ends instead of legends; at most two annotations;
   - horizontal hairline grid, quiet axes, no glow, shadow, or 3-D.
2. Two palettes: `PORCELAIN` (light, default) and `GRAPHITE` (dark), derived
   from the `sunumatik` porcelain-ink and graphite-ember palettes. Both
   six-slot series orders pass a categorical palette validator (lightness band,
   chroma floor, adjacent-pair CVD separation, normal-vision separation,
   contrast).
3. Chart-form corrections: no dual y-axes (PDF and CDF become two panels);
   estimates with intervals are dot-and-whisker plots, not bars; Sobol indices
   are a dumbbell whose segment is the interaction share; log axes use plain
   or percentage labels.
4. Titles are computed from the data (for example "X3 matters only through
   interactions (S = 0.01, S_T = 0.25)"), so a title cannot assert something
   the numbers contradict.
5. Every public plotting function runs inside the theme through the `themed`
   decorator; the font stack is filtered to installed families so text
   resolves at draw time without warnings.

## Consequences

- All figures share one grammar; lab figures double as README material.
- Computed titles need guarding: an early Sobol title claimed the wrong input
  acted "mostly through interactions" and was corrected to a ratio rule.
- Matplotlib remains optional; the theme is imported lazily.

## Rejected alternatives

- Per-figure styling in each lab: drifts immediately.
- A legend-first design: forces eye travel and colour lookup.
- Bundling web fonts from `sunumatik`: they are WOFF2 variable fonts that
  Matplotlib cannot load; the stack prefers installed humanist sans fonts.
