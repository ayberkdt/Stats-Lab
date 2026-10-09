# Changelog

All notable changes are recorded here. Versions follow semantic versioning
while the public API remains pre-1.0 and experimental.

## [0.13.0] — 2026-10-09

### Added

- Independent Gaussian observation models with separately retained measurement
  and model-discrepancy variance components.
- Simple linear least squares, residual diagnostics, rank/condition-based
  identifiability evidence, and a combined calibration result.
- General multivariate linear-Gaussian conditioning, seeded posterior sampling,
  predictive covariance decomposition, and posterior-predictive checks.
- An applicability-scoped discrepancy record, cited calibration guide, seeded
  practical lab, explanatory uncertainty-band plot, regression tests,
  ADR-0013, and session memory.

### Changed

- All fifteen ATHENS inverse/calibration topics and the Bayesian calibration lab
  now have implemented evidence; the ledger advances to 74 implemented, 17
  partial, and 19 planned rows.
- Architecture adds explicit inverse and validation layers while preserving the
  acyclic dependency direction and standard-library-first runtime policy.

## [0.12.0] — 2026-10-09

### Added

- Coordinatewise central finite-difference local sensitivity with explicit
  perturbations and exact model-evaluation cost.
- Independent-input functional-ANOVA foundations and pick-freeze Monte Carlo
  estimators for first-, second-, and total-effect Sobol indices.
- Unclipped finite-sample diagnostics, total-effect ranking, threshold-based
  dimension-reduction candidates, and uncertainty-reduction priorities.
- A seeded Ishigami interaction laboratory and first-versus-total sensitivity
  visualization demonstrating why main-effect-only ranking can fail.
- Six equation records, four primary references, UQ-5 guide, regression tests,
  ADR-0012, and session memory.

### Changed

- All eleven ATHENS sensitivity topics now have implemented evidence; the
  canonical ledger advances to 58 implemented, 18 partial, and 34 planned rows.
- Architecture adds a dedicated sensitivity layer while preserving dependency
  direction and standard-library-first runtime policy.

## [0.11.0] — 2026-10-09

### Added

- An explicit `g(x) <= 0` limit-state contract, direct failure-probability
  estimation, complement reliability, Wilson intervals, and work counts.
- Ordinary importance-weighted rare-event estimation with proposal failure,
  weight ESS, and maximum normalized-weight diagnostics.
- Exact affine standard-normal FORM with signed reliability index, constrained
  nearest-boundary/design point, unit normal, residual, and `Phi(-beta)`.
- Standard-normal directional sampling using analytic chi radial tails.
- A cited reliability guide, seeded method-comparison lab, five equation
  records, regression tests, and a generated comparison chart.

### Changed

- All eleven ATHENS reliability topics and the reliability lab now have
  implemented evidence; the ledger advances to 47 implemented, 18 partial,
  and 45 planned rows.
- Architecture adds a dedicated reliability layer above propagation and Monte
  Carlo without introducing cycles.

## [0.10.0] — 2026-10-09

### Added

- Antithetic variates, control variates, conditional Monte Carlo, equal-width
  stratified sampling, and ordinary importance sampling with explicit
  assumptions, confidence intervals, seeded streams, and work accounting.
- Weight effective-sample-size and maximum-normalized-weight diagnostics that
  remain explicitly separate from estimator standard error.
- Lower-bounded Neyman allocation with exact integer-budget preservation and a
  cost-normalized estimator-efficiency comparison contract.
- An efficiency-gain visualization and a reproducible advanced Monte Carlo lab
  comparing five estimators under one evaluation budget.
- Eight equation records, an advanced Monte Carlo guide, ADR-0010, regression
  tests, and primary references.

### Changed

- All twelve ATHENS advanced Monte Carlo topics and the advanced practical lab
  now have implemented evidence; the canonical ledger advances to 35
  implemented, 18 partial, and 57 planned rows.
- Project memory advances the next safe delivery slice to reliability and rare
  events while preserving remaining general Monte Carlo frontier work.

## [0.9.0] — 2026-10-09

### Added

- Composite trapezoid and Simpson quadrature with finite-value validation and
  exact evaluation counts.
- Monte Carlo mean, event-probability, and uniform-integration estimators with
  sample variance, estimator standard error, confidence intervals, sample
  sizes, and charged work.
- Nested convergence traces and a confidence-band convergence visualization.
- Named forward propagation over explicit parameter samples, retaining raw QoI
  outputs, output moments, strict exceedance probabilities, Wilson intervals,
  and complete model-call counts.
- A reproducible laboratory for pi estimation, deterministic versus stochastic
  integration, and uncertain projectile-range propagation.
- UQ-2 guide, equation records, tests, and ADR-0009 separating output
  variability, sampling error, numerical error, and computational cost.

### Changed

- ATHENS UQ-2 rows advance from planned to implemented with direct code and lab
  evidence; advanced budget allocation remains partial.
- Architecture adds domain-neutral `numerics`, estimator-only `monte_carlo`,
  and higher-level `propagation` packages while preserving dependency direction.

## [0.8.0] — 2026-10-09

### Added

- Lognormal and shape-scale Weibull laws with explicit parameterizations,
  stable direct tails, generalized quantiles, moments, hazards where defined,
  and call-local seeded sampling.
- A positive-definite multivariate Normal implementation with Cholesky
  validation/sampling, log-density, Mahalanobis distance, marginals, and
  correlations, without adding a numerical runtime dependency.
- Closed-form Normal and Lognormal maximum-likelihood estimators.
- One-sample Kolmogorov-Smirnov statistic with an explicitly limited
  asymptotic p-value and a tail-stable Anderson-Darling statistic without an
  invented universal calibration.
- Known-scale/large-sample mean intervals and Wilson score intervals.
- A fitted-CDF diagnostic, reproducible statistical-fitting lab, UQ-1 guide,
  equation records, tests, and primary references.

### Changed

- ATHENS coverage evidence advances required families and multivariate Normal
  to implemented; broader estimation, GOF calibration, confidence intervals,
  and labs remain honestly partial.
- Architecture now reserves separate layer-2 ownership for `multivariate` and
  `inference`, avoiding a monolithic UQ package.

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

