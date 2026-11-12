# ADR-0097: Evidence-backed overview presentation

- Status: Accepted
- Date: 2026-11-12

## Context

The library had no presentation that explains it in one sitting. A slide deck
written by hand drifts from the code the moment a number changes: the README
quick start, the benchmark file, and the repository counts (tests, equations,
references, decision records) all move between releases. The project owner's
presentation toolkit, sunumatik, already holds a deck runtime
(`presets/core/deste-kabuk`), a results-talk starter whose numbers come from
one data object and are checked by `denetle()` (`presets/deck_starter`), and
validators for page types, copy density, assertions, evidence ledgers, and
rendered decks. The datawiz toolkit audits figures at their final size,
including a slide profile with an 18 pt floor.

## Decision

1. **One data function.** `tools/build_presentation_data.py` computes every
   number the deck shows and writes `docs/presentation/deck-data.json`:
   the README quick-start `UQStudy` (seed 2026: Sobol' propagation, Sobol
   indices, six reliability methods, CVaR), an exact reference failure
   probability by adaptive quadrature, the lander of
   `examples/mission_systems_lab.py` assessed exactly on its binary decision
   diagram, highlights of `benchmarks/results/quick.json` (sampling designs on
   the Ishigami mean, posterior spread on the linear-Gaussian calibration,
   the four-branch FORM failure), and repository counts. Counts that change
   with most features are shown as floors (tests to 100, equations and
   references to 50, decision records to 10). Monte Carlo quantities carry
   their standard error in `stochastic`.
2. **Bound numbers.** Each number on a slide is an element
   `<span data-k="dotted.key" data-fmt="...">`; the tool writes the formatted
   value (English or Turkish separators) into it, so the static HTML is
   complete without JavaScript. In the browser, `runtime/deck-data.mjs`
   (adapted from the starter's `deste-veri.mjs`) re-reads `deck-data.json` and
   reports any slide that disagrees, any conclusion number that did not
   appear earlier, and over-long headlines; `runtime/denetle-cli.mjs` runs the
   same check in Node.
3. **Sixteen typed slides.** Following sunumatik's orchestration skill, every
   slide has a page type and one claim: cover, context, question, method,
   comparison, evidence (reliability), big number, evidence (calibration),
   evidence (systems), method (verification), evidence (benchmarks),
   limitations, conclusion, sources, closing, appendix. Copy follows the
   assertion and language skills; detail lives in speaker notes. The Turkish
   deck has the same slides, page types, and data keys.
4. **Runtime and style.** `runtime/deste-kabuk.mjs` and `.css` are copied from
   sunumatik with one local change (chrome labels pass through a table, so the
   English deck has English controls; one placeholder size moved from 20 to
   17 px to keep eight type sizes). Slides use the library's chart palettes
   (`PORCELAIN` light, `GRAPHITE` dark) as CSS tokens and follow the system
   colour scheme or `?theme=`.
5. **Charts.** Four slide charts are drawn with datawiz at the `sunum-tam`
   target (305 mm, 24 pt, 18 pt floor) in the library's semantic colours,
   in two languages and two themes (sixteen PNGs), with datawiz's dark-mode
   glow switched off as the chart-aesthetics rules require; each is audited
   with `datawiz.figur_denetle(fig, ortam="sunum")` and the run fails on any
   error.

## Consequences

- A change that moves a shown number makes `tests/test_presentation.py` fail
  until `PYTHONPATH=src python tools/build_presentation_data.py` is rerun; the
  floors keep this rare for counts. Merging other work that adds decision
  records past the next ten, tests past the next hundred, or changes the
  benchmark file needs the rebuild.
- The data function takes about 25 s (the mission MTTF quadrature dominates);
  the test runs it once.
- Rebuilding the charts needs Matplotlib and the datawiz toolkit, which is not
  a library dependency; the data and HTML rebuild needs only the library.
- The benchmark highlights come from the recorded quick benchmark (probstat
  0.17.0, quick mode, two to five seeds); the deck says so on the slides and
  in the limitations.

## Verification

- `tests/test_presentation.py` (10 tests, 359 subtests): stored data equals a
  fresh run (strings, integers, and booleans exactly; reals to 1e-6; Monte
  Carlo values within four standard errors), repository counts re-read, the
  quadrature reference lies inside the Monte Carlo interval and within 1% of
  SORM, every bound element shows the formatted value of its key, no digit
  appears on a slide or in its notes outside a bound element (citation lists
  exempt), only local files are referenced, the Turkish deck has the same
  slides and keys, every slide has a type, headline, and notes, and every
  chart exists in both themes with a clean audit.
- Data consistency (`denetle-cli.mjs`): both decks, 16 slides and 117 bound
  values each, 0 warnings; in the browser `html[data-veri="tutarli"]`.
- sunumatik validators: `validate-page-types.mjs` 0 errors, 0 warnings (both
  languages, with the preset catalogue); `validate-slide-copy.mjs` 0 warnings
  (both); `validate-assertions.mjs` Turkish 0 warnings, English 9 warnings
  that are verbs missing from the linter's English list (answers, raises,
  turn, stand, apply, answered, traces, start, report, need), reviewed by hand;
  `validate-deck-manifest.mjs`, `validate-evidence-ledger.mjs` (12 claims,
  10 sources, 4 figures), `validate-deck-plan.mjs`, `validate-visual-spec.mjs`
  valid; `validate-icon-usage.mjs` 0 warnings; `audit-deck.mjs` 0 errors and
  `validate-design-tactics.mjs` 0 errors on a snapshot with the stylesheets
  inlined (one warning each: the font wait lives in the linked runtime module,
  and a centred 17 px overview placeholder in the copied runtime).
- UX law audit (`ux-yasa-denetimi.mjs --tablo`): both decks 0 errors and 0
  warnings (8 type sizes, 2 radii, 0 shadows, 0 saturated hues); the same with
  the live runtime chrome added to the snapshot. Measured at 1920x1080 in a
  browser: controls 44x44 px or larger, 4 buttons with one primary plus a
  5-item menu, counter update 4 ms after a click, controls cover 1.1% of the
  screen, kicker to headline 16 px against header to chart 40 px, every slide
  inside the 96/72 px safe area with no overlap between body and footer.
- Figure audit: 16 charts, 0 errors and 0 warnings (`figures/audit.json`),
  every PNG inspected.
- Sources of the methods on the slides: [REF-HASOFER-LIND-1974],
  [REF-BREITUNG-1984], [REF-HOHENBICHLER-RACKWITZ-1988], [REF-SALTELLI-2010],
  [REF-OWEN-1995], [REF-ROCKAFELLAR-URYASEV-2000], [REF-KUCUKELBIR-ETAL-2017],
  [REF-BRYANT-1986], [REF-RAUSAND-HOYLAND-2004]; markup and accessibility
  [REF-WHATWG-HTML], [REF-W3C-WCAG21-2018]; figures [REF-MATPLOTLIB];
  reporting contract [REF-PROBSTAT-REPORTING].

## Revised (2026-11-12): the applied-statistics slides

The deck said nothing about the applied-statistics core of 0.20.0. Two
slides now follow the systems evidence, so the deck has eighteen slides:

- `s10` (context, `baglam`): every input distribution needs data, and data
  raise the questions of the README's applied-statistics table (counted at
  build time as `repository.statistics_questions`).
- `s11` (evidence, `kanit`, simulated data): partial pooling in a linear
  mixed model. `partial_pooling()` draws twelve batches with 2 to 30 strength
  tests from `resolve_stream(2026).child("pooling")`, fits a random batch
  intercept by REML with `fit_linear_mixed_model`, and records each batch's
  BLUP, raw deviation, and the closed-form share `m sigma_b^2 / (m sigma_b^2
  + sigma^2)`; the headline states the share kept by the smallest and the
  largest batch, and the chart (`figures/pooling-*`) draws the closed-form
  curve with the fitted ratios on it [EQ-LMM-005], [REF-HENDERSON-1975].

The former slides 10 to 16 are now 12 to 18; the sources slide adds
Henderson (1975), and the notes of the limitations and conclusion slides
carry the simulated-data caveat and the pooling result. In
`runtime/deck.css` the rule hiding the dark chart is scoped to
`.grafik-govde`, because `.grafik-govde img { display: block }` outranked it
and showed both chart variants in the light theme.

Verification of the revision: `tests/test_presentation.py` 11 tests and 455
subtests (a new test checks every BLUP against the closed-form share);
`denetle-cli.mjs` 18 slides and 150 bound values per deck, 0 warnings;
page types, slide copy, deck manifest, evidence ledger (15 claims, 12
sources, 5 figures), deck plan, and visual spec valid with 0 warnings; the
English assertion linter keeps its 9 reviewed verb warnings and the new
slides add none; figure audit 20 charts, 0 errors, 0 warnings; UX law audit
0 errors and 0 warnings for both decks.
