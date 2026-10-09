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
- [ ] Continuous integration across supported Python versions.
- [ ] API stability labels and deprecation policy.

## Phase 1 — Probability foundations

- [x] Exact factorial, permutation, combination, and multiplication rule.
- [x] Probability-space concepts and axioms documented.
- [x] Finite sample spaces, event algebra, and set identities.
- [x] Conditional probability, independence, total probability, and Bayes' rule.
- [x] Finite random variables, transformations, expectation, moments, and CDFs.
- [x] Finite joint, marginal, and conditional PMFs; dependence summaries.
- [x] Finite probability-, moment-, and characteristic-function evaluation.
- [ ] Modes of convergence, laws of large numbers, and central limit theorems.
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
- [ ] Multivariate normal, Dirichlet, multinomial, and copulas.
- [~] Stable log-density, quantile, survival, hazard, and entropy interfaces.
- [x] Call-local deterministic sampling streams for implemented families.
- [ ] Goodness-of-fit checks and stream-splitting abstractions.

## Phase 3 — Descriptive statistics and data diagnostics

- [x] Mean, variance, standard deviation, z-scores, ECDF.
- [x] Covariance and Pearson correlation.
- [ ] Quantiles with explicit convention, robust location/scale, skewness,
  kurtosis, ranks, contingency summaries, missing-data policies.
- [ ] Histograms, Q-Q plots, box/violin plots, scatter diagnostics, uncertainty
  annotations, and accessible color/label standards.

## Phase 4 — Statistical inference

- [ ] Estimands, estimators, bias, risk, consistency, sufficiency, completeness.
- [ ] Likelihood, score, Fisher information, MLE/MAP, EM.
- [ ] Confidence intervals, bootstrap, permutation methods.
- [ ] Hypothesis tests, power, effect sizes, multiple testing, equivalence tests.
- [ ] Bayesian priors/posteriors, conjugacy, posterior predictive checks,
  decision theory, model comparison, calibration.
- [ ] Frequentist/Bayesian interpretations kept explicit and separate.

## Phase 5 — Modeling and computation

- [ ] Linear/nonlinear regression, GLMs, regularization, hierarchical models.
- [ ] Numerical optimization, automatic differentiation boundary, stable linear
  algebra, special functions, and error budgets.
- [ ] Monte Carlo, importance sampling, MCMC, HMC/NUTS, SMC, variational
  inference, amortized inference.
- [ ] Time series, Markov chains, Gaussian processes, point processes, SDEs.
- [ ] Survival analysis, missing/censored data, experimental design.

## Phase 6 — Frontier systems

- [ ] Probabilistic programming intermediate representation and inference API.
- [ ] Causal graphs, interventions, identification, sensitivity analysis.
- [ ] Information geometry and optimal transport.
- [ ] Normalizing flows, energy-based models, diffusion/score systems,
  neural stochastic differential equations.
- [ ] Conformal prediction, distribution shift, uncertainty calibration,
  robustness, and sequential decision systems.
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
- [ ] Validation experiment records, validation metrics, applicability domain,
  and explicit separation of verification, validation, and prediction.

### UQ-1 — Uncertainty modeling and statistical fitting

- [x] Random variables, PMF/PDF/CDF, moments, covariance, correlation, Normal,
  Uniform, Lognormal, and Weibull foundations.
- [x] Positive-definite multivariate Normal density, marginals, correlation,
  Mahalanobis distance, and Cholesky sampling.
- [~] Normal/Lognormal maximum likelihood, generic KS/AD statistics, z and
  Wilson confidence intervals, fitted-CDF visualization, and seeded lab.
- [ ] General multivariate protocols and transformations, Weibull estimation,
  exact/bootstrapped fitted-null calibration, Student-t intervals, sampling
  stream splitting, and real-data fitting case studies.

### UQ-2 — Forward propagation

- [x] Output expectation/variance, finite-bound expectation integrals,
  composite trapezoid/Simpson quadrature, and evaluation counts.
- [x] Basic Monte Carlo mean/probability/integral estimators, sample variance,
  standard error, normal/Wilson intervals, convergence traces, and costs.
- [x] Named forward-model sample propagation, immutable raw outputs, QoI
  summaries, strict threshold probabilities, and a seeded teaching lab.
- [ ] Unbounded-domain quadrature transformations, adaptive error estimates,
  correlated-input sampling, vector QoIs, streaming/out-of-core accumulation,
  and propagation of numerical/model-form uncertainty.

### UQ-3 — Advanced Monte Carlo

- [x] Standard Monte Carlo, pedagogical pi experiment, explicit work counts,
  and root-N convergence visualization.
- [x] Estimator-efficiency and sampling-strategy comparisons.
- [x] Antithetic and control variates, conditional Monte Carlo, stratified
  sampling, importance sampling, and simulation-budget allocation.
- [ ] Latin hypercube and randomized quasi-Monte Carlo, multilevel Monte Carlo,
  self-normalized/adaptive importance sampling, control functionals, and
  unequal-cost optimal allocation.

### UQ-4 — Reliability and rare events

- [ ] Reliability definitions, limit-state functions, failure probability,
  direct Monte Carlo, rare events, and binomial confidence intervals.
- [ ] FORM, reliability index beta, constrained design-point optimization,
  most-probable failure points, directional sampling, and method comparison.

### UQ-5 — Sensitivity analysis

- [ ] Local and global sensitivity with variance-based functional ANOVA.
- [ ] First-, second-, and total-effect Sobol indices with Monte Carlo
  estimators and uncertainty diagnostics.
- [ ] Input ranking, dimension reduction, and uncertainty-reduction strategy.

### UQ-6 — Inverse problems and Bayesian calibration

- [ ] Observation and measurement-noise models, parameter identification,
  least squares, and linear-regression calibration.
- [ ] Priors, likelihoods, posteriors, Gaussian conditioning, posterior
  sampling, posterior predictive checks, and identifiability diagnostics.
- [ ] Model discrepancy and the explicit calibration-versus-prediction split.

### UQ-7 — Filtering and data assimilation

- [ ] Discrete-time state-space models, prediction, observation, measurement
  update, process/measurement noise, and covariance propagation.
- [ ] Sequential Bayes, filtering versus smoothing, and data assimilation.
- [ ] Kalman, extended Kalman, ensemble Kalman, and particle filters.

### LAB-1 through LAB-6 — Reproducible practical work

- [x] Basic Monte Carlo and advanced variance-reduction laboratories.
- [~] Statistical-fitting and uncertainty-visualization laboratories.
- [ ] Reliability (FORM and directional sampling) and Bayesian-calibration
  laboratories.
- [ ] Every lab includes definitions, assumptions, equations, seeded examples,
  plots, diagnostics, cost/error discussion, citations, and regression tests.

### PRE-1 — Prerequisite bridges

- [~] Probability and statistics learning paths.
- [ ] Linear algebra; numerical integration, convergence, and error;
  optimization; multivariable calculus, Jacobians, conditional distributions;
  and an R programming interoperability guide.

## Cross-cutting work (never “finished”)

- [~] Definitions, assumptions, examples, counterexamples, plots, and sources.
- [~] Equation registry and reference integrity.
- [~] Boundary/property/metamorphic testing.
- [ ] Generated API and mathematical documentation.
- [ ] Accessibility, localization-ready prose, and interactive explanations.
- [ ] Performance profiles and multiple-precision/reference backends.

