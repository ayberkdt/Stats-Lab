# ADR-0078: Laboratory contract and accessible output

- Status: Accepted
- Date: 2026-11-11

## Context

Two roadmap items were open. The laboratory track asked that every lab
include definitions, assumptions, equations, seeded examples, plots,
diagnostics, a cost and error discussion, citations, and regression tests,
and marked the statistical-fitting and uncertainty-visualization labs as
partial. The cross-cutting track asked for accessibility,
localization-ready prose, and interactive explanations.

An audit of the 32 scripts in `examples/` at the start of this change found:

| Property | Labs that had it |
|---|---|
| Docstring sections for definitions, assumptions, equations, diagnostics, cost and error, references | 0 |
| A docstring longer than one line | 25 |
| A module-level `SEED` constant | 3 (`forecast_calibration_lab`, `limit_theorems_lab`, `optimal_transport_lab`) |
| Figures written through the themed layer | 24; 4 called `Figure.savefig` directly, 4 wrote no figure (`limit_theorems_lab`, `mission_systems_lab`, `ppl_lab`, `uq_workflow_lab`) |
| A test that imports or runs the script | 1 (`uq_workflow_lab`, by `tests/test_uq0_prediction_workflow.py`) |

Every lab ran in the CI "Examples run" step. A full sequential run of all 32
labs takes about five minutes on a desktop machine (measured
before this change), so running them again inside the unit tests is not
affordable.

On the accessibility side, both chart palettes were already validated for
colour-vision deficiency (ADR-0024), but saved figures carried no text
alternative, the HTML study report used the stage label as the `alt` text of
every figure, had no skip link, and wrapped header, navigation, sections,
and footer in one `main` element. Text contrast had never been tested.

## Decision

### 1. A machine-checkable lab contract

`tests/test_lab_contract.py` is static and fast (about 7 s). For every
script in `examples/` it requires:

1. a module docstring with the sections `Definitions:`, `Assumptions:`,
   `Equations:`, `Diagnostics:`, `Cost and error:`, and `References:`, in
   that order and non-empty;
2. at least one `EQ-*` identifier in `Equations:`, every cited identifier
   registered in `docs/equation-registry.md`, and every one of them present
   in the source the lab can reach. Reachability follows each imported name
   to the module that defines it (package re-exports are resolved) and then
   that module's own imports, lazy imports included. A lab therefore cannot
   cite an equation from a package it only touches through an `__init__`;
3. at least one registered `REF-*` key in `References:`;
4. a module-level integer `SEED` that the script reads;
5. an import of `probstat.visualization` and a call to `save_figure`,
   `save_study_figures`, or `write_study_html`, and no direct `savefig`
   call, so every written figure carries its text alternative (section 3);
6. a `main` function behind one `if __name__ == "__main__":` guard, and an
   import of the module that runs nothing heavy (the test imports all 32);
7. an entry in `LAB_REGRESSION_TESTS` naming a test module and a library
   name; the test checks that the lab uses the name and the test module
   exercises it;
8. the CI workflow line that runs every lab.

Execution stays in CI; the contract test covers import, documentation,
seeding, figures, and the link to a regression test.

### 2. Every lab brought up to the contract

All 32 scripts now carry the six sections, written from the code they run:
equations are the identifiers evaluated by the functions each lab calls, and
cost statements quote the sample sizes, replicate counts, and chain lengths
in the code. Every lab that seeded its draws with a literal now names that
root seed `SEED` with the same value, so no printed number changed. The four
labs that saved with a bare `savefig` now use `save_figure`. The four labs
without figures gained one, each drawn by an existing themed function or the
theme primitives:

| Lab | New figure |
|---|---|
| `limit_theorems_lab` | exact Kolmogorov distance of standardized exponential means to the normal law and to the Edgeworth approximation, under the Berry-Esseen bound (log-log) |
| `mission_systems_lab` | Sobol indices of the mission reliability over the uncertain data (`plot_sobol_indices`) |
| `ppl_lab` | non-centred NUTS draws of the eight-schools `tau` over the exact grid posterior (`plot_tau_posterior`) |
| `uq_workflow_lab` | propagation and calibration stage figures of its `UQStudy` (`save_study_figures`) |

`generate_foundation_figures` drew no random numbers; its empirical-CDF
figure now shows a 12-point seeded `Exponential(1.25)` sample instead of a
hand-typed six-point list. The script prints nothing, so no printed number
changed.

The two labs the roadmap marked partial were strengthened:

- `statistical_fitting_lab` keeps its illustrative fully-specified KS
  p-value and adds Kolmogorov-Smirnov and Anderson-Darling p-values
  calibrated by a parametric bootstrap that refits the lognormal law to each
  of 199 replicates (`bootstrap_goodness_of_fit`, [EQ-INF-GOF-PB-001]). With
  `SEED = 2026` the KS p-value moves from 0.936 (uncalibrated) to 0.685, and
  the AD p-value is 0.595.
- `uncertainty_visualization_lab` now prints the Monte Carlo standard error
  of the terminal mean (3.16), a 95% Wilson interval for the exceedance
  probability ([0.175, 0.255], [EQ-INF-CI-WILSON-001]), and the DKW band
  half-width at `n = 120` (0.124, [EQ-STAT-DKW-001]).

### 3. Text alternatives for every saved figure

`probstat.visualization.accessibility.describe_figure` builds a paragraph
from what the figure already states: the suptitle, then per visible panel
its assertion title, the subtitle that names the marks, and each axis with
its label, displayed range, and non-linear scale; colour bars are described
by label and range. Nothing is inferred from the data that the figure does
not print, so the description cannot contradict the image (WCAG 2.1 success
criterion 1.1.1 [REF-W3C-WCAG21-2018]).

`save_figure` writes the text into the file: PNG `Title` and `Description`
text chunks (predefined keywords of the PNG specification
[REF-PNG-SPEC-2003]; non-Latin-1 text is stored as an international text
chunk), SVG `Title` and `Description`, and PDF `Title` and `Subject`. A
caller can pass `description=` to replace the generated text. Embedded
metadata was chosen over a sidecar `.txt` file because it travels with the
image when the image is copied, renamed, or attached, cannot drift out of
sync with it, and adds no files to `examples/generated/`. Metadata alone is
not read by screen readers in a web page, so the HTML report puts the same
text in each `img` `alt` attribute.

### 4. Contrast tested with the WCAG formula

`relative_luminance` and `contrast_ratio` implement the WCAG 2.1 definitions
([EQ-VIZ-WCAG-001], [EQ-VIZ-WCAG-002]). `tests/test_visualization_accessibility.py`
requires, against the canvas of each palette:

| Token | Threshold | Porcelain | Graphite |
|---|---|---|---|
| ink | 4.5 (text, SC 1.4.3) | 13.81 | 15.99 |
| muted (subtitles, units) | 4.5 | 4.95 | 7.76 |
| accent (reference lines and labels) | 4.5 | 6.06 | 8.10 |
| series[0..5] (data marks) | 3.0 (graphics, SC 1.4.11) | 3.31 to 4.59 | 4.72 to 5.49 |

The report stylesheet's ink, muted, and accent colours are tested at 4.5
against both its canvas and its white or dark surface, and its light and
dark tokens are tested to equal the chart palettes.

Known gap, recorded rather than fixed: direct labels are drawn in the series
hue at 8.5 pt bold, which WCAG counts as normal-size text. In the light
palette five of the six series hues are below 4.5 against the canvas
(3.31 to 4.16; `series[2]` reaches 4.59). Darkening them would change every
figure and the CVD-validated palette of ADR-0024, so it needs its own
decision; the dark palette meets 4.5 for every slot.

### 5. Accessible HTML study report

`study_html` keeps `lang="en"` and now opens with a skip link to the stage
results (SC 2.4.1), places the title block in `header`, the stage list in
`nav`, the stages in `main id="content"`, and the provenance line in
`footer` ([REF-WHATWG-HTML]); every figure's `alt` text is its generated
description. The skip link is visually hidden until it receives focus.

### 6. Localization-ready prose: policy, not a catalogue

User-facing strings are not centralized. Most are assertion titles computed
from the data ("At n = 100 the normal law is off by 0.013") spread across
about twenty plotting modules and the study reports; moving them into a
message catalogue would touch every figure and report for no present
consumer, and risks changing output. The policy for new and edited strings
is instead:

- build each sentence in one formatted string with named fields, never by
  concatenating translated fragments;
- format numbers with explicit format specifications, so a locale-aware
  formatter can replace them in one place;
- keep units and symbols out of sentence grammar where possible;
- declare the language of every generated document (`lang="en"`).

### 7. Interactive explanations are out of scope

Interactive figures would need a JavaScript or notebook layer the library
does not have. The roadmap item stays partially open: accessibility is
addressed, localization has a policy, interactivity is not started.

## Consequences

- A new lab cannot merge without the six sections, a seed, a themed
  figure, and a regression-test link; a typo in an `EQ-*` or `REF-*` key, or
  an equation from code the lab never reaches, fails the test.
- Every PNG, SVG, and PDF written by `save_figure`, and every figure in a
  study report, carries a text description generated from its titles and
  axes. Figures with weak titles get weak descriptions, so assertion titles
  (ADR-0024) now serve accessibility as well.
- Four labs print a few more lines (the two strengthened labs) or write one
  more figure (the four labs that had none); no previously printed number
  changed.
- The light palette's direct-label contrast is a documented open issue.

## Verification

- `PYTHONPATH=src python -m pytest -q -p no:cacheprovider
  tests/test_documentation_contract.py tests/test_architecture.py
  tests/test_lab_contract.py tests/test_visualization*.py
  tests/test_workflow_study.py`: all pass.
- Every lab run with `MPLBACKEND=Agg` and its standard output compared with
  the output recorded before the change: identical except wall-clock lines,
  the `Created (UTC)` timestamps of the study reports, and the added lines
  of `statistical_fitting_lab` and `uncertainty_visualization_lab`.
- A probe of the reachability rule: from `statistical_fitting_lab`,
  `EQ-INF-GOF-PB-001` is reachable and `EQ-NN-MLP-001` and `EQ-OT-SINK-001`
  are not.
- `python -m ruff check .` and `python -m mypy src/probstat` pass.
