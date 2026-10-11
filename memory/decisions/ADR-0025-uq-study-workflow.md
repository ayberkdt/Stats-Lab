# ADR-0025 — `UQStudy`: one study object over every workflow

- Status: Accepted
- Date: 2026-10-11

## Context

ADR-0014 gave every analysis package the same model contract, but a user
still had to call each package separately, thread seeds by hand, and collect
costs and results themselves. Chaining stages (calibrate, then predict, then
assess risk on an emulator) required glue code in every study.

## Decision

1. `probstat.workflow` (layer 7) provides `UQStudy(model, seed=...)` with
   stages `propagate`, `sensitivity`, `screening`, `reliability`, `form`,
   `risk`, `calibrate`, `predict`, and `emulate`.
2. Each stage draws a named child stream `root.child(stage, k)`, so one root
   seed reproduces the whole study and repeated stages never share a stream.
3. Each stage appends a `StageRecord` (headline numbers, forward-model
   evaluations, stream seed, full result). `report()` renders Markdown;
   `visualization.study.save_study_figures` renders one themed figure per
   supported stage.
4. `emulate` fits a PCE or GP surrogate on the study model and returns a new
   `UQStudy` on the surrogate, so expensive downstream analyses run on the
   emulator while the parent records training cost and accuracy. The PCE basis
   defaults to Legendre for independent uniform inputs and Hermite otherwise.
5. The study adds no numerical methods; it only orchestrates domain packages.

## Consequences

- The README quick start can show a complete forward/backward/risk study in a
  few lines.
- Calibration in a study treats the model's joint input law as the prior over
  calibration parameters; models mixing calibration parameters with aleatory
  inputs need a dedicated `CalibrationProblem` setup.
- `UQStudy` is mutable (it accumulates records); results inside records are
  immutable domain objects.

## Rejected alternatives

- A global configuration object for seeds: hides provenance and breaks
  reproducibility across processes.
- Embedding plotting in the workflow layer: would invert the dependency on the
  optional visualization layer.
