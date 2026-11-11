# ADR-0086: Ordered learning paths and a generated mathematical reference

- Status: Accepted
- Date: 2026-11-11

## Context

Two roadmap items were open. PRE-1 listed "Probability and statistics
learning paths" as partial: `docs/learning-path.md` described the ATHENS
VV-UQ sequence in prose, followed by unordered lists of later guides and a
numbered "Stage 4 and beyond" outline, so a reader could not tell which guide
to read first, which laboratory exercised it, or which decision record fixed
its scope, and nothing checked that every guide was reachable. The
cross-cutting item "Generated API reference; generated mathematical
documentation remains" was partial because the 1026 records of
`docs/equation-registry.md` were ordered by the history of the work, not by
the package that implements them, and their `- Code:` lines were never
checked: 29 records named classes only without their package (for example
`FiniteRandomVariable.cdf`), 4 more named some classes that way, one named a
function that `probstat.probability` does not export, and one pointed at the
standard-library `statistics` module through a path that reads as
`probstat.statistics`.

## Decision

1. **Six ordered tracks.** `docs/learning-path.md` is rewritten as tracks of
   numbered steps: mathematical prerequisites (M1-M5), probability (P1-P11:
   counting, measure-theoretic probability, limit theorems, stochastic
   processes, information), statistics (S1-S15: descriptive, estimation,
   testing, Bayesian, modelling, model criticism), uncertainty quantification
   (U1-U9 follow ATHENS UQ-0 to UQ-7, U10-U16 the integrated toolset),
   systems engineering (E1-E6: reliability, Markov and semi-Markov, risk and
   cost, mission), and computation (C1-C5: multilevel and Stein Monte Carlo,
   gradient-based MCMC, probabilistic programming, generative models,
   score-based diffusion). Each step has a guide link (or a section anchor
   when a later track revisits part of a guide), a one-sentence summary of
   what the guide contains, `Requires` (earlier step identifiers only),
   `Lab` (a script in `examples/`, or `none;` and the test module that
   asserts the guide's numbers when no laboratory exists), and `Decision
   record`. Requirements may cross tracks but only backwards in the
   document, so the whole file is one acyclic order.
2. **Learning-path contract.** `tests/test_learning_paths.py` requires every
   guide in `docs/foundations/` and `docs/prerequisites/` to be linked from a
   step, every relative link (with its heading anchor) to resolve, every step
   to carry the three fields, every requirement to name an earlier step, and
   every laboratory to be an existing `examples/*.py` file (or, after
   `none;`, an existing `tests/*.py` file).
3. **Generated mathematical reference.** `tools/generate_math_reference.py`
   parses the registry (heading, equation text, then exactly Domain, Meaning,
   Code, Source) and writes `docs/math-reference/`: one page per top-level
   package, records grouped by the module that defines the first object of
   the `- Code:` line, each with its equation, fields, the cited
   `REF-*` entries copied from `docs/references.md`, and a link to the
   registry anchor; a `laboratories-and-tests` page for records that name
   only laboratory or test files; and an index with the records, modules,
   and cross-references per page, the unresolved targets, and the private
   targets.
4. **Placement.** A record appears on exactly one page, that of the package
   of its first object target. Every other package its `- Code:` line names
   lists it in a "Records placed on other pages" section that links to the
   entry.
5. **Target rule.** A target is a backticked dotted path whose first segment
   is a top-level `probstat` package (a call-argument suffix is ignored), a
   shorthand `Name.member` or `.member` that expands against an earlier
   target of the same record, an `examples.<lab>.<name>` object defined at
   the top level of that laboratory, or a `tests/`, `examples/` or
   `benchmarks/` file, optionally with `::name`. Objects must import
   (attributes, dataclass fields, or annotated attributes); files and names
   must exist. A `Name.member` span that cannot be expanded is an error, so
   class names are qualified once per record. Other backticked text is
   descriptive. A target with a segment starting with `_` is private: it
   must import, it is allowed, and the index lists it (55 such targets,
   mostly numerical kernels in `distributions._special` and
   `distributions._continuous` and the HMC internals).
6. **Registry corrections, in place.** The `- Code:` lines of EQ-PROB-004 to
   EQ-PROB-009, EQ-RV-001 to EQ-RV-013, EQ-GF-001 to EQ-GF-003, and
   EQ-MEAS-003 to EQ-MEAS-009 qualify their classes with `probability.`;
   EQ-SENS-TOTAL-001, EQ-BAYES-PRED-001, EQ-DIST-ISF-001, and
   EQ-RISK-BANDIT-003 qualify `SobolResult`, `PosteriorPredictive`, the four
   `isf` methods, and `BanditRun`; EQ-LIM-002 names
   `probability.limit_theorems.standard_normal_cdf`, where the function is
   defined (the package never exported it); EQ-DIST-NORM-006 names the
   standard-library `NormalDist` in prose instead of a path that reads as
   `probstat.statistics`. No equation, domain, meaning, or source changed.

## Consequences

- A reader can start any track at its first step and see, for each guide,
  what to read before it and what to run with it. Topics without a
  laboratory are visible as such.
- The mathematical reference answers "which equations does this package
  evaluate, and from which source" without reading the whole registry, and
  the index shows where implementation details carry equations.
- Any change to the registry or to the bibliography makes the committed
  pages stale until `PYTHONPATH=src python tools/generate_math_reference.py`
  is run; branches that append records must regenerate after merging.
  Renaming or moving a public object named in a `- Code:` line now fails
  `tests/test_math_reference.py` instead of leaving a dangling reference.
- Adding a guide without placing it in a step fails
  `tests/test_learning_paths.py`.

## Verification

- Before the registry corrections the generator reported 51 unresolved
  targets in 35 records; after them it reports none.
- `PYTHONPATH=src python -m pytest -q -p no:cacheprovider
  tests/test_learning_paths.py tests/test_math_reference.py` passes: 54
  guides reachable, every link and anchor resolves, requirements point
  backwards, all laboratories exist; the committed pages equal a fresh
  rendering, each of the 1026 records has exactly one entry, every target
  resolves, private targets are listed, cross-references link back, and a
  synthetic record exercises every clause of the target rule.
- `tests/test_documentation_contract.py`, `tests/test_prerequisite_guides.py`,
  `tests/test_api_reference.py`, and `tests/test_athens_coverage.py` pass;
  `ruff check .` and `mypy src/probstat` are clean.
