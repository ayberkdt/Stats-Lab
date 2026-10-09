# Roadmap

Status markers: `[x]` complete, `[~]` in progress, `[ ]` planned.

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
- [~] Random variables and finite expectation; moments and transformations planned.
- [ ] Joint, marginal, and conditional distributions.
- [ ] Probability generating, moment generating, and characteristic functions.
- [ ] Modes of convergence, laws of large numbers, and central limit theorems.
- [ ] Measure/integration bridge: sigma-algebras, measurable maps, Lebesgue
  expectation, Radon-Nikodym derivatives, and product measures.

## Phase 2 — Distribution system

- [x] Bernoulli and univariate normal baseline objects.
- [ ] Distribution protocols, parameter constraints, and support abstractions.
- [ ] Binomial, categorical, Poisson, geometric, negative binomial.
- [ ] Uniform, exponential, gamma, beta, chi-square, Student-t, F, logistic.
- [ ] Multivariate normal, Dirichlet, multinomial, and copulas.
- [ ] Stable log-density, quantile, survival, hazard, and entropy interfaces.
- [ ] Deterministic sampling streams and goodness-of-fit checks.

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

## Cross-cutting work (never “finished”)

- [~] Definitions, assumptions, examples, counterexamples, plots, and sources.
- [~] Equation registry and reference integrity.
- [~] Boundary/property/metamorphic testing.
- [ ] Generated API and mathematical documentation.
- [ ] Accessibility, localization-ready prose, and interactive explanations.
- [ ] Performance profiles and multiple-precision/reference backends.

