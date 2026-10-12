# Changelog

All notable changes are recorded here. Versions follow semantic versioning
while the public API remains pre-1.0 and experimental.

## [0.15.0] — 2026-10-12

Classical statistical modelling, causal questions, experiment planning, and
tail risk join the central model; three independent numerical reviews found
defects that are now fixed with oracle-backed regression tests.

### Added

- **API stability**: every package is labelled stable, provisional, or
  experimental (ADR-0048); `foundations.deprecated` manages deprecations.

- **Regression** (`inference`): treatment-coded design matrices; OLS/WLS by
  Householder QR with t/F inference, intervals, leverage, studentized
  residuals, Cook's distance, HC0-HC3, VIF; GLMs by IRLS (Gaussian,
  binomial logit/probit, Poisson, Gamma) with deviance, dispersion, offsets,
  and a separation certificate; ridge, lasso, and elastic net with
  warm-started paths and K-fold cross-validation.
- **Time series** (`timeseries`): ACF/PACF, Ljung-Box, differencing and
  decomposition, stationarity checks, exact Kalman-likelihood ARIMA MLE with
  exact forecast intervals, residual diagnostics, exponential smoothing.
- **Survival** (`inference.survival`): Kaplan-Meier with Greenwood variance
  and log-log intervals, Nelson-Aalen, RMST, log-rank test, censored
  exponential/Weibull/lognormal AFT MLE, Cox proportional hazards with
  Breslow or Efron ties.
- **Conformal prediction** (`conformal`): split conformal, conformalized
  quantile regression, jackknife+, coverage reports, conformal error bars
  for surrogates; `UQStudy.emulate(conformal_alpha=...)`.
- **Hamiltonian and variational inference**: static HMC and NUTS with
  dual-averaging step size and windowed metric adaptation; mean-field and
  full-rank Gaussian ADVI; `CalibrationProblem.sample_posterior(method="nuts")`.
- **Causal inference** (`causal`): DAGs with d-separation, back-door and
  front-door criteria and minimal adjustment sets; back-door and front-door
  adjustment on discrete structural causal models; propensity scores,
  regression adjustment, IPW and AIPW estimators with bootstrap intervals
  and balance diagnostics; E-values and confounding bias factors.
- **Experimental design** (`design`): approximate and exact D-optimal designs
  with equivalence-theorem certificates, efficient rounding and exchange,
  local designs for nonlinear models; Bayesian design by closed-form,
  linearized, Laplace, and nested Monte Carlo expected information gain, and
  greedy sequential design from a `CalibrationProblem`.
- **Extremes** (`risk.extremes`, `distributions.GeneralizedPareto`): block
  maxima and GEV fits with delta-method and profile-likelihood return-level
  intervals, GPD by MLE or probability-weighted moments, mean residual life,
  threshold stability, runs declustering, peaks-over-threshold return levels.
- **Information theory** (`information`): entropy, mutual information, KL,
  cross-entropy, Jensen-Shannon for finite laws; closed-form differential
  entropies and Gaussian KL; Kozachenko-Leonenko entropy and KSG mutual
  information estimators.
- **Adaptive reliability** (`reliability`): sequential search for multiple
  design points (bulging plus multi-start HL-RF), multi-point FORM for series
  systems (closed form for two points, Genz separation of variables beyond,
  Ditlevsen bounds), mixture importance sampling over all design points, and
  AK-MCS active learning with a Gaussian-process limit state (U-function,
  population enlargement, comparison with Monte Carlo on the same population).
- **Model discrepancy** (`inverse`): Kennedy-O'Hagan calibration with a
  Gaussian-process discrepancy (squared-exponential or Matern-5/2 kernel,
  priors on hyperparameters), MAP fits and MCMC/NUTS posteriors, predictions
  of the simulator alone and of the bias-corrected process, heteroscedastic
  log-linear noise, and an identifiability report for theta-discrepancy
  confounding.
- **Study stages**: `UQStudy.extremes` (peaks-over-threshold tail of a QoI
  reusing the propagation sample), `UQStudy.information_sensitivity`
  (KSG mutual information per input), and `UQStudy.design` (greedy Bayesian
  design for the study's calibration, re-estimated on an independent stream),
  `UQStudy.design_points` (multiple design points with multi-point FORM and
  mixture importance sampling), `UQStudy.active_reliability` (AK-MCS), and
  `UQStudy.verify` (solution verification of grid solutions; its numerical
  uncertainty feeds `validate(numerical_uncertainty="verified")`),
  `UQStudy.propagate_until` (constant-memory Monte Carlo that stops at a
  target relative error of the mean or of an exceedance probability),
  `UQStudy.calibrate_with_discrepancy` (Kennedy-O'Hagan; `predict` then
  reports simulator-only and bias-corrected intervals); `validate` adds CRPS,
  PIT, and interval coverage. The HTML report draws figures for every stage
  except validation.
- **Hierarchical models and EM** (`inference`): generic EM driver with a
  monotonicity check, Gaussian mixtures (full/diagonal covariance, k-means++
  starts, BIC/AIC selection), EM for censored normal data; exact gridded
  posterior of the hierarchical normal (eight-schools) model checked against
  NUTS on the non-centred parameterization, shrinkage factors, beta-binomial
  empirical Bayes; posterior predictive p-values with Monte Carlo errors and
  PIT uniformity checks.
- **Forecast evaluation** (`validation`): CRPS (ensemble, fair ensemble,
  closed forms for normal, lognormal, logistic, exponential, uniform, and
  quadrature for any law), log, Dawid-Sebastiani, energy, and variogram
  scores; pinball, interval, and weighted interval scores; Brier score with
  the Murphy decomposition; PIT values and histograms, interval coverage,
  reliability diagrams with consistency bands; energy-distance and MMD
  permutation tests, sample and quantile Wasserstein distances; exact
  two-sample Kolmogorov-Smirnov test (`inference`).
- **Optimal transport and information geometry** (`information`): exact
  transportation simplex with an optimality certificate, log-domain Sinkhorn
  and the debiased Sinkhorn divergence, Gaussian (Bures) and quantile
  Wasserstein distances and barycenters; closed-form and numerical Fisher
  information, Fisher-Rao distances, Hellinger, Bhattacharyya, total
  variation, Renyi, and alpha divergences.
- **Probabilistic programming** (`inference.ppl`): models written as plain
  Python functions with `sample`, `observe`, and `deterministic` sites;
  automatic bijections to unconstrained space with Jacobians, non-centred
  reparameterization, conditioning by name; NUTS, HMC, Metropolis, ADVI, and
  Laplace back ends; prior and posterior predictive sampling and predictive
  p-values. Eight schools written as a model matches the exact grid
  posterior within Monte Carlo error.
- **Benchmark suite** (`benchmarks/`): 11 problems with independently
  verified references (reliability, sensitivity, propagation, calibration)
  and 34 method adapters; per-method error, error in units of the method's
  own standard error, interval coverage, model evaluations, and wall time;
  `python benchmarks/run.py --quick` runs in CI and writes
  `docs/benchmarks.md`.
- **Numerical integration** (`numerics`): adaptive Gauss-Kronrod G7-K15
  (QUADPACK QAG/QAGI style) with error estimates, tanh-sinh, exp-sinh, and
  sinh-sinh rules for finite, half-infinite, and infinite ranges, an
  `integrate` dispatcher; tanh-sinh reports convergence and skipped nodes.
- **Limit theorems and estimator theory**: Berry-Esseen bounds against exact
  Kolmogorov distances, one-term Edgeworth expansions, Chebyshev, Hoeffding,
  and Bernstein bounds against exact tails, delta-method accuracy
  (`probability`); LLN/CLT simulations (`monte_carlo`); `evaluate_estimator`
  with Monte Carlo errors for bias, variance, MSE, efficiency, and coverage,
  Cramer-Rao bounds and MLE efficiency, Rao-Blackwellization and
  minimum-MSE examples (`inference`).
- **Missing data** (`inference`): missingness patterns and Little's MCAR
  test; EM for the multivariate normal under MAR with observed-information
  standard errors; proper multiple imputation by data augmentation with a
  normal-inverse-Wishart posterior (Bartlett sampler); Rubin's rules with
  Barnard-Rubin degrees of freedom; complete-case and mean-imputation
  baselines labelled as biased.
- **Contingency tables** (`inference`): risk difference (Newcombe hybrid
  score interval), risk ratio (Katz), and odds ratio (Woolf, with the
  Haldane-Anscombe correction); exact and asymptotic McNemar tests; Pearson
  and Haberman adjusted residuals.
- **Streaming statistics** (`statistics`): mergeable one-pass moments up to
  kurtosis (weighted and multivariate), covariance and correlation, P2
  quantiles, reservoir sampling, exceedance counters with Wilson intervals,
  and `run_until` stopping on a target relative error;
  `monte_carlo.estimate_mean_streaming`.
- **Kernel density** (`statistics.gaussian_kde`) with Silverman and
  normal-reference bandwidths.
- **Figures**: validation predictions with coverage (now in the study
  report), missingness patterns, imputation diagnostics, Rubin's-rules
  summaries, adjusted-residual heatmaps, two-by-two effect forests; PIT
  histograms use a simultaneous (Bonferroni) band by default, so a
  calibrated forecast is no longer called miscalibrated for one or two
  bins outside pointwise limits; design-point maps, AK-MCS classification and convergence;
  reliability diagrams, CRPS comparison, coverage, and distribution-shift
  ECDFs; transport plans, Sinkhorn convergence, quantile W1 areas, and
  displacement interpolation; discrepancy-corrected predictions and estimated
  discrepancy; eight-schools shrinkage, tau posterior, mixture fit and BIC
  selection, predictive-check and PIT histograms; return-level,
  mean-residual-life, and tail Q-Q plots;
  D-optimal design with its variance function and greedy EIG paths; causal
  effect forest, covariate balance (Love plot), and E-value contour; entropy
  decomposition and estimator checks; violin/box distribution comparison,
  correlogram, forecast fan, Kaplan-Meier, and regression diagnostics. Labs
  for extremes, experimental design, causal inference, information theory,
  hierarchical models, discrepancy calibration, and the new study stages.
- **API reference**: `docs/api-reference.md` generated from every package's
  `__all__` by `tools/generate_api_reference.py`, guarded by a staleness test.

### Changed

- Figures follow the sunumatik chart-aesthetics rules more closely: every
  uncertainty band (including nested inner bands, previously 0.28) is capped
  at 0.18 opacity; Fritsch-Carlson monotone smoothing for continuous curves
  and their bands; a fading gradient area for the primary observed series
  (used in the forecast figure).
- The graphite (dark) accent is indigo `#9BA9F0` instead of orange
  `#E8804A`, which sat below the normal-vision separation floor from the
  amber model hue.

### Fixed

- Cross-entropy importance sampling with covariance adaptation collapsed the
  proposal (eigenvalues near 5e-4, estimates biased low by up to 100% with
  small reported c.o.v.); adapted eigenvalues are now floored at 0.55 and a
  warning flags degenerate weights.
- Gaussian-process fits can be regularized (bounded length scales and
  signal variance, relative nugget, optional log-normal prior; reported when
  an optimum sits on a bound); AK-MCS uses this by default and no longer
  stops at its initial design on smooth limit states or misclassifies on
  white-noise fits (four-branch: 19 misclassified points over 10 seeds
  before, 0 after). Cross-entropy IS warns when its proposal locks onto one
  of several failure regions (all 7 locked four-branch runs warn; 0 of 40
  single-region runs do).
- Found by the benchmark suite: `effective_sample_size` could be negative on
  antithetic chains (now bounded below as in stan-dev/posterior), and AK-MCS
  reported nonsense (Pf 0.641 vs 8.2e-4) on near-singular GP fits because its
  incremental posterior did not repeat the fit's arithmetic.
- Eleventh review round: `integrate(auto)` reports `converged=False` whenever
  its two rules disagree or Gauss-Kronrod is blind; `propagate_until` waits
  for `min_samples` draws before declaring its target met.
- Tenth review round: `integrate(auto)` reported convergence when
  Gauss-Kronrod missed all of the mass; quantile Wasserstein distances hit
  `Q(1) = inf` for discrete laws without `isf`; `run_until` could report
  convergence before `min_samples` or for an all-zero stream.
- Ninth review round: PPL coordinate names stated the wrong transform for
  shifted bounds and mixed list supports; CE-IS left its multi-region flag
  unset on stalled symmetric runs.
- Eighth review round: `integrate(auto)` could return 3e-198 for an
  integral of 1.77 (now chooses by relative miss and keeps the
  double-exponential result when Gauss-Kronrod misses the mass);
  Gauss-Kronrod reported an overflowing width as converged infinity;
  double-exponential rules ignored truncation at their outermost nodes;
  `kolmogorov_distance_to_normal` was 55% low at n = 1e8; binomial masses
  lost about n eps (now Loader's saddle-point form); `effective_sample_size`
  now matches stan-dev/posterior including the improved-estimate term
  (antithetic chains were overstated by up to 19%).
- Seventh review round: validation coverage keys rounded 0.975 to 98% (and
  could collide); the deprecation note broke docstring dedenting; contingency
  counts above 2^53 were rounded and huge tables lost residual accuracy
  (now exact rational arithmetic); Wasserstein integrals are split at
  quantile crossings (W1 error 3.9e-5 to 8e-13) and unconverged quadrature
  now warns instead of returning silently.
- Sixth review round: tanh-sinh quadrature truncated 1e-15 from the ends,
  losing up to 3.7% of heavy-tailed quantile Wasserstein distances with a
  far too small error estimate (nodes now reach 1e-300; the routine moved to
  `numerics`); Renyi and alpha divergences overflowed at large orders;
  1-D energy distance lost digits under a large common offset; Fisher-Rao
  categorical distance cancelled for nearby laws.
- Fifth review round: censored-normal EM cancelled catastrophically far
  from zero (now centred moments, matching scipy at locations up to 1e8);
  hierarchical posteriors report infinite moments as `inf` instead of
  grid-dependent values; AK-MCS reports a probability consistent with the
  returned surrogate when stopped by `max_added`; predictive-check thinning
  respects its draw cap; the KDE cdf keeps lower-tail accuracy.
- Fourth review round: GEV profile-likelihood return-level intervals crashed
  on heavy-tailed samples (now level-adapted starts and finite brackets);
  gamma and beta differential entropies lost all accuracy at large shapes
  (Stirling-remainder forms, exact to rounding against mpmath); discrete
  mutual information, KL, and Jensen-Shannon divergences failed at extreme
  probabilities; Gaussian KL lost relative accuracy for nearly equal laws;
  greedy Bayesian design increments could be negative (one common stream for
  the whole run); non-symmetric covariances were silently accepted; GPD MLE
  rejects zero excesses and flags shape estimates below -1.
- E-values and confounding bias factors overflowed for extreme ratios;
  treatment vectors accept booleans as documented.
- Cox model: the Breslow baseline overflowed (or divided by zero) when
  `|x' beta|` exceeded about 709, e.g. with uncentred covariates; it is now
  stored at a reference linear predictor and predictions are shift-invariant.
- Linear models detect an intercept carried by a constant column other than
  ones or by full dummy coding, so R^2 and the F test are centred.
- Kaplan-Meier quantiles take the midpoint of a flat segment at exactly
  `1 - p` (R's `quantile.survfit` convention); the uncensored median is now
  the ordinary sample median.
- `UQWorkflowRecord.record_stages` rejected the `fitting` and `shapley`
  stages recorded by `UQStudy`; a test now keeps the guide in sync.
- Gaussian-copula fitting returned a numerically singular latent correlation
  for perfectly monotone columns (u-space coordinates of order 1e6); repair
  now triggers on the smallest eigenvalue, not on Cholesky failure.
- ADVI reported convergence while still climbing a flat ridge (mean one
  q-standard-deviation off); convergence now also requires no net ELBO rise
  over the buffer.
- Second review round: exact small-sample Kolmogorov-Smirnov p-values,
  Temme expansion for the incomplete gamma at very large shapes,
  cancellation-free density prefactors, narrow truncated-normal moments,
  noncentral t, studentized bootstrap with zero-SE resamples, permutation
  ties, Student-t tail overflow, Markov absorbing states, ANOVA/Welch
  centring.
- First review round: fault-tree rare-event cancellation and MOCUS
  absorption, Normal-inverse-gamma small shapes, calibration overflow at
  large log-noise, fragility tail continuity.

## [0.14.0] — 2026-10-12

One model for every question: forward, backward, and risk analyses now share
a single `ProbabilisticModel`, and `UQStudy` runs them with one seed tree and
one report.

### Added

- **Central model** (`modeling`): `RandomInput`, `JointInput` with a Gaussian
  copula and Nataf transformation (Pearson-to-latent correlation solver,
  tail-accurate upper-tail maps), `ProbabilisticModel` with evaluation
  counting, Karhunen-Loeve random-field inputs.
- **Shared numerics**: `linalg` (Cholesky, triangular, LU, Jacobi eigen,
  log-determinant; `multivariate` and `inference` now use it), `optimization`
  (BFGS, Nelder-Mead, Levenberg-Marquardt, Gauss-Newton covariance, Brent,
  finite differences), `sampling` (Latin hypercube, scrambled Halton, Sobol),
  SplitMix64 `RandomStream` seed trees.
- **Forward**: design-based `propagation.propagate` with quantiles, exceedance
  curves, rank correlations, and design comparison.
- **Sensitivity**: Sobol indices on the model with bootstrap intervals at no
  extra model cost; Morris screening.
- **Reliability**: HL-RF FORM for correlated non-normal inputs, SORM
  (Breitung, Hohenbichler-Rackwitz), design-point importance sampling, subset
  simulation, series/parallel systems with Ditlevsen bounds, bivariate normal
  CDF, inverse FORM, design for a target reliability, method comparison.
- **Risk**: VaR/CVaR (weighted, tie-exact), loss functions, model risk
  assessment, fault trees (MOCUS, exact with repeated events, importance
  measures), event trees, lognormal fragility MLE and hazard convolution,
  EVPI/EVSI, risk matrices with documented caveats.
- **Backward**: adaptive Metropolis with split and rank-normalized R-hat, ESS,
  MCSE; Laplace approximation; Normal-inverse-gamma regression;
  `CalibrationProblem` for nonlinear models with known, unknown, or Student-t
  noise; posterior push-forward `predict`; Fisher identifiability.
- **Validation**: area metric, u-pooling, Mahalanobis and reliability metrics,
  ASME V&V 20 comparison, Bayes factor, applicability domain, scoped verdicts
  and credibility records.
- **Dynamics**: Kalman filter with Joseph update, RTS smoother, EKF, UKF,
  bootstrap particle filter, EnKF, 3D-Var; Markov chains, Poisson and compound
  Poisson processes, Brownian/GBM/OU, Euler-Maruyama and Milstein, KL
  expansions.
- **Surrogates**: polynomial chaos (Hermite/Legendre, LOO error, Sobol from
  coefficients) and Gaussian-process regression; surrogate forward-model
  adapter.
- **Statistics**: Hyndman-Fan quantiles, MAD, skewness, kurtosis, ranks,
  Spearman; t, chi-square, Fisher, binomial, rank, permutation, ANOVA and TOST
  tests; bootstrap (percentile, basic, normal, studentized, BCa), jackknife,
  power and sample size, multiple-testing adjustments.
- **Distributions**: Gamma, Beta, Student-t, Chi-square, F, Logistic, Gumbel,
  Triangular, truncated Normal, GEV; inverse survival functions; Gamma and
  Weibull MLE; digamma/trigamma.
- **Workflow**: `UQStudy` stages `propagate`, `sensitivity`, `screening`,
  `reliability`, `form`, `risk`, `calibrate`, `predict`, `validate`,
  `emulate`, plus Markdown reports and per-stage figures.
- **Visualization**: semantic chart theme (light porcelain and dark graphite
  palettes, validated for colour-vision deficiency), assertion titles, direct
  labels, uncertainty bands, and new plots for designs, output distributions,
  FORM geometry, method comparisons, loss exceedance, MCMC, filtering,
  surrogates, and random fields; `PROBSTAT_CHART_THEME` selects the theme.
- Labs: central model, UQ study, nonlinear calibration, filtering, surrogate,
  random field, model selection, UQ workflow, uncertainty visualization;
  CI on Python 3.11-3.13; `py.typed`.
- Solution verification (GCI, least-squares order), uncertainty components in
  predictions, workflow records; generic MLE, bootstrap-calibrated
  goodness of fit, AIC/BIC model selection, joint-input fitting; multilevel
  Monte Carlo, cross-entropy IS, line sampling, Shapley effects; Gaussian
  conditioning, change of variables, delta method, prerequisite guides and an
  R bridge; `UQStudy.from_data`, `shapley`, and a self-contained HTML report.

### Changed

- `Normal.cdf` evaluates the lower tail through `erfc` (relative error below
  1e-13 down to 1e-300).
- All foundation plots were rebuilt on the theme; the PDF/CDF twin-axis plot is
  now two panels.
- ATHENS ledger: all 110 rows implemented (was 74 implemented, 17 partial,
  19 planned).

### Fixed

- Nataf maps saturated near latent values of 8, stalling FORM at high
  reliability; upper tails now use inverse survival functions.
- `UQStudy.emulate` chooses a Legendre basis for independent uniform inputs; a
  Hermite basis extrapolated badly at far-tail design points.
- Method comparison omits SORM rows when the FORM index is not positive
  instead of failing.
- Independent numerical review fixes: rare-event OR gates and min-cut bounds
  (exact to p = 1e-17), fault-tree cut sets by expansion with absorption,
  series-system uni-modal upper bound at very high reliability, Student-t
  tails and large-nu centre, Normal-inverse-gamma intervals for vague
  priors, calibration log-densities at extreme log-sigma, and a seamless
  log-normal-CDF tail in the fragility likelihood.

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

