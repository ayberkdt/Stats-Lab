# Roadmap

Status markers: `[x]` complete, `[~]` in progress, `[ ]` planned.

The [ATHENS 2026 coverage matrix](../docs/curriculum/athens-2026-coverage.md)
is the authoritative 110-topic ledger. A checked roadmap phase does not override
the matrix: curriculum completion requires 110/110 `Implemented` rows with
tested evidence.

## Phase 0 — Repository constitution

- [x] Layered acyclic package layout.
- [x] Citation and equation-ID contract.
- [x] Automated architecture and documentation checks.
- [x] Durable project memory, ADRs, and session notes.
- [x] Optional visualization layer separated from mathematics.
- [x] Continuous integration across supported Python versions (3.11-3.13).
- [x] API stability labels and deprecation policy (ADR-0048).

## Phase 1 — Probability foundations

- [x] Exact factorial, permutation, combination, and multiplication rule.
- [x] Probability-space concepts and axioms documented.
- [x] Finite sample spaces, event algebra, and set identities.
- [x] Conditional probability, independence, total probability, and Bayes' rule.
- [x] Finite random variables, transformations, expectation, moments, and CDFs.
- [x] Finite joint, marginal, and conditional PMFs; dependence summaries.
- [x] Finite probability-, moment-, and characteristic-function evaluation.
- [x] Modes of convergence, laws of large numbers, and central limit theorems
  (exact Berry-Esseen/Edgeworth/tail-bound comparisons, ADR-0051).
- [ ] Measure/integration bridge: sigma-algebras, measurable maps, Lebesgue
  expectation, Radon-Nikodym derivatives, and product measures.

## Phase 2 — Distribution system

- [x] Bernoulli and univariate normal baseline objects.
- [x] Dependency-neutral structural protocols for univariate probability laws.
- [x] Parameter constraints, support abstractions, and initial stable
  log-probability APIs.
- [x] Binomial, categorical, and Poisson.
- [x] Geometric and negative binomial with explicit failure-count convention.
- [x] Continuous uniform and rate-parameterized exponential.
- [ ] Gamma, beta, chi-square, Student-t, F, and logistic.
- [~] Multivariate normal and Gaussian copula (Nataf) joint inputs; Dirichlet,
  multinomial, and non-Gaussian copulas remain.
- [~] Stable log-density, quantile, survival, hazard, and entropy interfaces.
- [x] Call-local deterministic sampling streams for implemented families.
- [~] Stream splitting via SplitMix64 `RandomStream`; family-specific
  goodness-of-fit calibration remains.

## Phase 3 — Descriptive statistics and data diagnostics

- [x] Mean, variance, standard deviation, z-scores, ECDF.
- [x] Covariance and Pearson correlation.
- [x] Hyndman-Fan quantiles, MAD, skewness, kurtosis, ranks, Spearman
  correlation, contingency summaries (two-by-two effect measures, McNemar,
  adjusted residuals), and missing-data methods (EM, multiple imputation,
  ADR-0047).
- [x] Histograms, Q-Q plots, ECDF bands, violin/box plots with kernel
  density, scatter diagnostics, uncertainty annotations, and accessible
  colour/label standards (CVD-validated light and dark palettes, ADR-0024).

## Phase 4 — Statistical inference

- [~] Estimands, estimators, bias, risk, consistency, sufficiency (ADR-0051);
  completeness is discussed but not computed.
- [x] Likelihood, score, Fisher information, MLE/MAP (generic MLE with
  observed-information SEs, Laplace/MAP), EM and Gaussian mixtures
  (ADR-0040).
- [x] Confidence intervals (z, Student-t, Wilson, Clopper-Pearson), bootstrap
  (percentile, basic, normal, studentized, BCa), permutation tests.
- [x] Hypothesis tests, power, effect sizes, multiple testing, equivalence tests.
- [x] Bayesian priors/posteriors, conjugacy, decision theory (EVPI/EVSI),
  model comparison (AIC/BIC, Bayes factors), calibration, posterior
  predictive checks (ADR-0040).
- [ ] Frequentist/Bayesian interpretations kept explicit and separate.

## Phase 5 — Modeling and computation

- [x] Linear/nonlinear regression, GLMs, regularization (ADR-0030),
  hierarchical normal and beta-binomial models (ADR-0040).
- [~] Numerical optimization (BFGS, Nelder-Mead, Levenberg-Marquardt, Brent)
  and consolidated dense linear algebra; automatic differentiation remains.
- [~] Monte Carlo, importance sampling, adaptive MCMC with diagnostics, Laplace,
  particle filters, HMC/NUTS, Gaussian ADVI (ADR-0032); amortized inference
  remains.
- [x] Markov chains, Poisson processes, SDEs (Euler-Maruyama, Milstein),
  Gaussian-process regression, random fields, ARIMA and exponential smoothing
  (ADR-0031).
- [x] Survival analysis and censored data (ADR-0033), optimal and Bayesian
  experimental design (ADR-0035), missing data under MAR (ADR-0047).

## Phase 6 — Frontier systems

- [ ] Probabilistic programming intermediate representation and inference API.
- [x] Causal graphs, interventions, identification, sensitivity analysis
  (ADR-0034).
- [x] Information measures and estimators (ADR-0036), information geometry
  and optimal transport (ADR-0043).
- [ ] Normalizing flows, energy-based models, diffusion/score systems,
  neural stochastic differential equations.
- [~] Conformal prediction (ADR-0033), distribution-shift tests and forecast
  calibration with proper scoring rules (ADR-0042); robustness and
  sequential decision systems remain.
- [ ] Reproducible benchmark suites that compare correctness, calibration,
  numerical stability, and computational cost—not accuracy alone.

## ATHENS VV-UQ delivery track

This track cuts across the general library phases while preserving package
dependency direction.

### UQ-0 — Verification, validation, and workflow vocabulary

- [x] Aleatory/epistemic nature and parametric/model-form/numerical/observation
  source classifications as separate axes.
- [x] Named forward-model and quantity-of-interest contracts.
- [x] Conservative numerical-error budgets, observed order, and Richardson
  extrapolation with assumptions and failure modes.
- [x] Validation experiment records, validation metrics, applicability domain,
  and explicit separation of verification, validation, and prediction.

### UQ-1 — Uncertainty modeling and statistical fitting

- [x] Random variables, PMF/PDF/CDF, moments, covariance, correlation, Normal,
  Uniform, Lognormal, and Weibull foundations.
- [x] Positive-definite multivariate Normal density, marginals, correlation,
  Mahalanobis distance, and Cholesky sampling.
- [~] Normal/Lognormal maximum likelihood, generic KS/AD statistics, z and
  Wilson confidence intervals, fitted-CDF visualization, and seeded lab.
- [~] Nataf transformations and correlated joint inputs delivered; general
  multivariate protocols, Weibull estimation,
  exact/bootstrapped fitted-null calibration, Student-t intervals, sampling
  stream splitting, and real-data fitting case studies.

### UQ-2 — Forward propagation

- [x] Output expectation/variance, finite-bound expectation integrals,
  composite trapezoid/Simpson quadrature, and evaluation counts.
- [x] Basic Monte Carlo mean/probability/integral estimators, sample variance,
  standard error, normal/Wilson intervals, convergence traces, and costs.
- [x] Named forward-model sample propagation, immutable raw outputs, QoI
  summaries, strict threshold probabilities, and a seeded teaching lab.
- [x] Correlated-input sampling, vector QoIs, quasi-random designs, quantiles,
  exceedance curves, and design comparison on `ProbabilisticModel`.
- [~] Unbounded-domain quadrature transformations and adaptive error
  estimates (ADR-0050); streaming accumulation and propagation of
  numerical/model-form uncertainty remain.

### UQ-3 — Advanced Monte Carlo

- [x] Standard Monte Carlo, pedagogical pi experiment, explicit work counts,
  and root-N convergence visualization.
- [x] Estimator-efficiency and sampling-strategy comparisons.
- [x] Antithetic and control variates, conditional Monte Carlo, stratified
  sampling, importance sampling, and simulation-budget allocation.
- [x] Latin hypercube and randomized quasi-Monte Carlo (scrambled Halton/Sobol).
- [ ] Multilevel Monte Carlo, adaptive importance sampling, control
  functionals, and unequal-cost optimal allocation.

### UQ-4 — Reliability and rare events

- [x] Reliability definitions, limit-state functions, failure probability,
  direct Monte Carlo, rare events, and binomial confidence intervals.
- [x] Affine standard-normal FORM, reliability index beta, exact constrained
  design-point optimization, most-probable failure points, directional sampling,
  and method comparison.
- [x] Nonlinear/non-normal HL-RF FORM via Nataf, SORM, design-point IS, subset
  simulation, system reliability bounds, inverse FORM.
- [x] Multiple design points, cross-entropy adaptation, line sampling, and
  AK-MCS active learning (ADR-0029, ADR-0041).

### UQ-5 — Sensitivity analysis

- [x] Local and global sensitivity with variance-based functional ANOVA.
- [x] First-, second-, and total-effect Sobol indices with Monte Carlo
  estimators and uncertainty diagnostics.
- [x] Input ranking, dimension reduction, and uncertainty-reduction strategy.
- [x] Bootstrap intervals for Sobol indices on the central model; Morris
  screening.
- [ ] Correlated-input and Shapley effects, replicated stochastic simulators,
  and derivative-based global measures.

### UQ-6 — Inverse problems and Bayesian calibration

- [x] Observation and measurement-noise models, parameter identification,
  least squares, and linear-regression calibration.
- [x] Priors, likelihoods, posteriors, Gaussian conditioning, posterior
  sampling, posterior predictive checks, and identifiability diagnostics.
- [x] Model discrepancy and the explicit calibration-versus-prediction split.
- [x] Nonlinear calibration, unknown noise, Student-t robust likelihoods,
  adaptive MCMC, Laplace, posterior push-forward prediction.
- [x] Heteroscedastic noise, correlated (Gaussian-process) discrepancy
  processes (ADR-0039), and optimal experiment design (ADR-0035).

### UQ-7 — Filtering and data assimilation

- [x] Discrete-time state-space models, prediction, observation, measurement
  update, process/measurement noise, and covariance propagation.
- [x] Sequential Bayes, filtering versus smoothing, and data assimilation.
- [x] Kalman, extended and unscented Kalman, ensemble Kalman, and particle
  filters.

### LAB-1 through LAB-6 — Reproducible practical work

- [x] Basic Monte Carlo and advanced variance-reduction laboratories.
- [~] Statistical-fitting and uncertainty-visualization laboratories.
- [x] Reliability laboratory with FORM and directional sampling.
- [x] Bayesian-calibration laboratory.
- [ ] Every lab includes definitions, assumptions, equations, seeded examples,
  plots, diagnostics, cost/error discussion, citations, and regression tests.

### PRE-1 — Prerequisite bridges

- [~] Probability and statistics learning paths.
- [ ] Linear algebra; numerical integration, convergence, and error;
  optimization; multivariable calculus, Jacobians, conditional distributions;
  and an R programming interoperability guide.

## Integration track (0.14)

- [x] Central `ProbabilisticModel` shared by every workflow (ADR-0014).
- [x] Model-based reliability, risk analysis, surrogates, validation, and
  stochastic processes (ADR-0015 to ADR-0023).
- [x] `UQStudy` orchestration with one seed tree and one report (ADR-0025).
- [x] Semantic chart theme for every figure (ADR-0024).
- [x] All 110 ATHENS curriculum rows implemented with evidence.
- [ ] Numerical backend boundary for large dense problems (ADR-0008 follow-up).

## Cross-cutting work (never “finished”)

- [~] Definitions, assumptions, examples, counterexamples, plots, and sources.
- [~] Equation registry and reference integrity.
- [~] Boundary/property/metamorphic testing.
- [~] Generated API reference (`tools/generate_api_reference.py`); generated
  mathematical documentation remains.
- [ ] Accessibility, localization-ready prose, and interactive explanations.
- [ ] Performance profiles and multiple-precision/reference backends.

