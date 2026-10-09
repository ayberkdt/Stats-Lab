# Changelog

All notable changes are recorded here. Versions follow semantic versioning
while the public API remains pre-1.0 and experimental.

## [0.4.0] — 2026-10-09

### Added

- Finite probability-generating functions over complex arguments.
- Finite moment-generating and characteristic functions with explicit overflow
  behavior and transform identities.
- Runtime-checkable `CumulativeDistribution`, `HasMeanAndVariance`,
  `DiscreteProbabilityLaw`, and `ContinuousProbabilityLaw` protocols.
- Cross-domain conformance tests for Bernoulli, Normal, and induced finite laws.
- Characteristic-function visualization with generated teaching figure.
- Equation records `EQ-GF-001` through `EQ-GF-003`.
- ADR-0004 documenting dependency-neutral structural interfaces.

### Changed

- Packaging now uses current SPDX license metadata and declares all wheel build
  requirements explicitly.

## [0.3.0] — 2026-10-09

### Added

- Immutable finite random variables with generic hashable codomains.
- Structural and positive support, preimages, predicate events, PMFs, and CDFs.
- Exact finite transformations, expectations, raw/central moments, variance,
  and standard deviation.
- Joint PMFs, marginalization, conditional PMFs, variable-independence tests,
  covariance, and correlation.
- PMF and joint-mass visualizations plus two generated teaching figures.
- Equation records `EQ-RV-001` through `EQ-RV-013` and a full conceptual guide.
- ADR-0003 documenting codomain and shared-measure ownership rules.
- Offline-verifiable setuptools packaging in place of an unavailable build
  backend.

## [0.2.0] — 2026-10-09

### Added

- Immutable, identity-owned finite sample spaces and events.
- Complement, union, intersection, difference, symmetric difference,
  disjointness, and subset operations.
- Weighted and uniform finite probability measures.
- Conditional probability, independence testing, total probability, Bayes'
  rule, and finite expectation.
- Finite-measure educational visualization and a diagnostic-test example.
- Equation records `EQ-PROB-002` through `EQ-PROB-009`.
- Stronger documentation test requiring references in every callable docstring.

## [0.1.0] — 2026-10-09

### Added

- Layered package constitution and architecture checks.
- Counting, descriptive-statistics, Bernoulli, and normal foundations.
- Equation registry, bibliography, roadmap, memory, and first ADR.
- Optional foundation plots and the initial verification suite.

