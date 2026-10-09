# Changelog

All notable changes are recorded here. Versions follow semantic versioning
while the public API remains pre-1.0 and experimental.

## [0.7.0] — 2026-10-09

### Added

- A canonical, test-enforced ATHENS 2026 VV-UQ coverage matrix containing all
  110 supplied topics with status, target phase, owner, and evidence.
- A narrow `probstat.uq` package with independent uncertainty-nature and
  uncertainty-source classifications, immutable quantities of interest, and a
  validated named forward-model contract.
- Conservative numerical-error budgets, observed convergence order, and
  Richardson extrapolation with documented assumptions and failure modes.
- Four equation records and primary VV-UQ references.
- A VV-UQ foundation guide, the UQ-0 through UQ-7 delivery track, and ADR-0007
  governing package growth and honest completion semantics.

### Changed

- Visualization is now layer 3 and may consume the layer-2 UQ contract; the
  architecture test still rejects upward imports and cycles.
- Project memory now points to ATHENS UQ-1 as the next safe delivery slice.

## [0.6.0] — 2026-10-09

### Added

- Zero-based Geometric and Negative Binomial distributions with explicit
  failures-before-success parameterization.
- Continuous Uniform and rate-parameterized Exponential distributions.
- Symmetry-aware regularized incomplete-beta continued fraction for stable
  Negative Binomial CDF and survival evaluation.
- Runtime-checkable density, log-density, survival, quantile, and hazard
  protocols.
- Direct survival and generalized quantile methods for earlier ordered analytic
  distributions where mathematically appropriate.
- Waiting-count comparison and coordinated PDF/CDF/survival visualizations.
- Twenty-eight equation records covering the new numerical and distribution
  functions, plus primary references and boundary conventions.
- Conceptual documentation, numerical identity tests, and ADR-0006.

### Changed

- The distribution roadmap now separates completed deterministic sampling from
  planned stream splitting and goodness-of-fit work.

## [0.5.0] — 2026-10-09

### Added

- Immutable parameter constraints and finite, integer, and real support objects.
- Stable Binomial, Categorical, and Poisson distributions with explicit
  degenerate cases, log-PMFs, and reproducible sampling.
- Regularized incomplete-gamma series and continued fractions for direct
  Poisson CDF and survival-tail evaluation.
- Narrow runtime-checkable PMF and log-PMF protocols.
- Categorical entropy and event probabilities without invented numeric order.
- Count-window and categorical PMF visualizations plus generated teaching
  figures.
- Equation records `EQ-DIST-BIN-001` through `004`, `EQ-DIST-CAT-001` through
  `002`, `EQ-DIST-POIS-001` through `004`, and `EQ-NUM-GAMMA-001`.
- A discrete-distribution guide, numerical boundary tests, and ADR-0005.

### Changed

- Bernoulli and Normal now expose explicit support and stable log-probability
  evaluation through the shared distribution infrastructure.

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

