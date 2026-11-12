# Learning Paths

The guides are arranged in six ordered tracks. Each step names one guide (or
one section of a guide), the steps to read first, the laboratory in
`examples/` that exercises it, and the decision record that fixes its scope.
Where no laboratory exists, the step names the test module that asserts the
guide's numbers instead.

The library separates conceptual order from software layer order: a step may
use packages from any layer, but it only builds on ideas from earlier steps.

## How to read a step

- **Requires** lists earlier step identifiers only, so every track can be read
  top to bottom and the whole document forms one acyclic order. A step in a
  later track may require a step from an earlier track.
- **Lab** names a runnable script (`PYTHONPATH=src python examples/<lab>.py`)
  or, when there is none, the test module that recomputes the guide's numbers.
- **Decision record** links the ADR that states the scope and the boundaries
  of the implementation.

`tests/test_learning_paths.py` checks that every guide in `docs/foundations/`
and `docs/prerequisites/` appears in a step, that every link resolves, that
requirements point only backwards, and that every named laboratory exists.

## Tracks

1. [Mathematical prerequisites](#track-m--mathematical-prerequisites) (M1-M5)
2. [Probability](#track-p--probability) (P1-P11): counting, measure-theoretic
   probability, limit theorems, stochastic processes.
3. [Statistics](#track-s--statistics) (S1-S19): descriptive statistics,
   estimation, testing, Bayesian inference, modelling, model criticism,
   mixed models, count and categorical regression, smoothing, and
   multivariate analysis.
4. [Uncertainty quantification](#track-u--uncertainty-quantification) (U1-U16):
   the ATHENS verification, validation, and UQ sequence, then the integrated
   toolset on one model.
5. [Systems engineering](#track-e--systems-engineering) (E1-E6): reliability,
   Markov and semi-Markov models, risk and cost, mission assessment.
6. [Computation](#track-c--computation) (C1-C5): multilevel and Stein Monte
   Carlo, gradient-based MCMC, probabilistic programming, generative models.

## Track M — Mathematical prerequisites

Every Python snippet in these guides is executed by
`tests/test_documentation_examples.py` (ADR-0095), and each documented output block must
match to the shown precision; `tests/test_prerequisite_guides.py` keeps every
snippet paired with its output block.

### M1 — Probability and statistics refresher

[Probability and statistics refresher](prerequisites/probability-and-statistics-refresher.md):
a concept map from events, conditioning, random variables, parametric
distributions, sample statistics, and the limit theorems to the `probstat`
objects that compute them.

- Requires: none
- Lab: none; snippets run in [`tests/test_documentation_examples.py`](../tests/test_documentation_examples.py)
- Decision record: [ADR-0028](adr/ADR-0028-prerequisite-bridges.md)

### M2 — Linear algebra

[Linear algebra](prerequisites/linear-algebra.md): Cholesky factors of a
covariance, LU solves and determinants, conditioning numbers, and symmetric
eigen-decompositions on the dependency-free `probstat.linalg` layer.

- Requires: none
- Lab: none; snippets run in [`tests/test_documentation_examples.py`](../tests/test_documentation_examples.py)
- Decision record: [ADR-0028](adr/ADR-0028-prerequisite-bridges.md)

### M3 — Multivariable calculus and conditioning

[Multivariable calculus and conditioning](prerequisites/multivariable-calculus-and-conditioning.md):
Jacobians, the change-of-variables rule (log-normal, polar coordinates,
Box-Muller, linear maps of a Gaussian), the delta method, and conditioning a
Gaussian vector.

- Requires: M2
- Lab: none; snippets run in [`tests/test_documentation_examples.py`](../tests/test_documentation_examples.py) and [`tests/test_prerequisite_bridges.py`](../tests/test_prerequisite_bridges.py)
- Decision record: [ADR-0028](adr/ADR-0028-prerequisite-bridges.md)

### M4 — Optimization

[Optimization](prerequisites/optimization.md): finite differences and the
truncation-rounding trade-off, BFGS, Nelder-Mead, Levenberg-Marquardt least
squares with its Gauss-Newton covariance, and Brent root finding.

- Requires: M2
- Lab: none; snippets run in [`tests/test_documentation_examples.py`](../tests/test_documentation_examples.py)
- Decision record: [ADR-0028](adr/ADR-0028-prerequisite-bridges.md)

### M5 — Numerical integration

[Numerical integration](prerequisites/numerical-integration.md): why fixed
grids are not enough, adaptive Gauss-Kronrod, tanh-sinh, exp-sinh and
sinh-sinh rules with error estimates, where each rule fails, and the
evaluation policy.

- Requires: none
- Lab: none; snippets run in [`tests/test_documentation_examples.py`](../tests/test_documentation_examples.py) and [`tests/test_adaptive_quadrature.py`](../tests/test_adaptive_quadrature.py)
- Decision record: [ADR-0050](adr/ADR-0050-adaptive-quadrature.md)

## Track P — Probability

From counting on finite spaces to measure theory, limit theorems, and
processes in time and space.

### P1 — Probability foundations and counting

[Probability foundations](foundations/probability.md): experiments, sample
spaces, events, the three axioms, finite weighted and equally likely models
with factorials, permutations, and combinations, conditioning, independence,
total probability, and Bayes' rule. Probability is a measure on events, not a
synonym for frequency or confidence.

- Requires: none
- Lab: [`examples/generate_foundation_figures.py`](../examples/generate_foundation_figures.py)
- Decision record: [ADR-0002](adr/ADR-0002-finite-space-ownership.md)

### P2 — Finite random variables and joint laws

[Finite random variables and joint laws](foundations/random-variables.md): a
random variable as a measurable mapping, induced PMFs, transformations, CDFs
and moments, joint laws and marginals, covariance versus dependence, and
probability-generating, moment-generating, and characteristic functions.

- Requires: P1
- Lab: [`examples/generate_foundation_figures.py`](../examples/generate_foundation_figures.py)
- Decision record: [ADR-0003](adr/ADR-0003-random-variable-codomains.md)

### P3 — Discrete distributions

[Discrete distribution foundations](foundations/discrete-distributions.md):
parameter constraints, support as mathematical metadata, PMF and log-PMF,
categorical label semantics, and the binomial, categorical, and Poisson
families.

- Requires: P2
- Lab: [`examples/generate_foundation_figures.py`](../examples/generate_foundation_figures.py)
- Decision record: [ADR-0005](adr/ADR-0005-support-log-probability-and-categorical-semantics.md)

### P4 — Waiting counts, tails, and continuous foundations

[Waiting counts, tails, quantiles, and continuous foundations](foundations/waiting-and-continuous-distributions.md):
geometric and negative binomial parameterizations, survival functions that are
not computed as `1 - CDF`, quantiles as generalized inverses, and the uniform
and exponential laws.

- Requires: P3
- Lab: [`examples/generate_foundation_figures.py`](../examples/generate_foundation_figures.py)
- Decision record: [ADR-0006](adr/ADR-0006-tail-quantile-and-parameterization-contracts.md)

### P5 — Continuous distribution catalog

[Continuous distribution catalog](foundations/continuous-distribution-catalog.md):
exact parameterizations of the continuous families, tail decay, numerical tail
accuracy, when each family suits a UQ or reliability input, and gamma and
Weibull fitting.

- Requires: P4
- Lab: [`examples/model_selection_lab.py`](../examples/model_selection_lab.py)
- Decision record: [ADR-0020](adr/ADR-0020-continuous-distribution-catalog.md)

### P6 — Multivariate laws and conjugate updates

[Multivariate laws and conjugate updates](foundations/multivariate-laws.md):
the Dirichlet, multinomial and Dirichlet-multinomial, multivariate Student t,
Wishart and inverse Wishart laws, and Dirichlet-categorical and
normal-inverse-Wishart updating.

- Requires: P5, M3
- Lab: none; numbers asserted in [`tests/test_multivariate_laws.py`](../tests/test_multivariate_laws.py)
- Decision record: [ADR-0067](adr/ADR-0067-multivariate-laws-and-conjugacy.md)

### P7 — Measure-theoretic probability

[Measure-theoretic probability](foundations/measure-theoretic-probability.md):
sigma-algebras and atoms, measurable maps and image measures, the Lebesgue
integral, product measures and Fubini-Tonelli, Radon-Nikodym derivatives,
conditional expectation, total variation, and completeness of a statistic,
all exact on finite spaces.

- Requires: P2
- Lab: none; numbers asserted in [`tests/test_measure_theory.py`](../tests/test_measure_theory.py)
- Decision record: [ADR-0071](adr/ADR-0071-measure-bridge-and-interpretation.md)

### P8 — Limit theorems and estimator properties

[Limit theorems and estimator properties](foundations/limit-theorems-and-estimators.md):
exact laws of the standardized mean, Berry-Esseen and Edgeworth, concentration
inequalities against exact tails, the delta method, bias, variance, MSE and
coverage of estimators, Cramer-Rao bounds, sufficiency, and
Rao-Blackwellization; simulated illustrations are kept separate.

- Requires: P4
- Lab: [`examples/limit_theorems_lab.py`](../examples/limit_theorems_lab.py)
- Decision record: [ADR-0051](adr/ADR-0051-limit-theorems-and-estimators.md)

### P9 — Stochastic processes and random fields

[Stochastic processes and random fields](foundations/stochastic-processes-and-random-fields.md):
discrete-time Markov chains, Poisson and Brownian processes with exact
samplers, SDE schemes and their convergence orders, Gaussian random fields by
Karhunen-Loeve expansion, and random-field inputs of a probabilistic model.

- Requires: P8
- Lab: [`examples/random_field_lab.py`](../examples/random_field_lab.py)
- Decision record: [ADR-0023](adr/ADR-0023-stochastic-processes-and-random-fields.md)

### P10 — Information theory

[Information theory](foundations/information-theory.md): Shannon entropy,
joint and conditional entropy, mutual information, KL and Jensen-Shannon
divergences, closed-form differential entropies, Gaussian KL divergences, and
nearest-neighbour estimators from samples.

- Requires: P5
- Lab: [`examples/information_lab.py`](../examples/information_lab.py)
- Decision record: [ADR-0036](adr/ADR-0036-extremes-and-information.md)

### P11 — Optimal transport and information geometry

[Optimal transport and information geometry](foundations/optimal-transport.md):
exact discrete transport with dual certificates, log-domain Sinkhorn,
closed-form Wasserstein distances, barycenters, Fisher information, Fisher-Rao
distances, and divergences.

- Requires: P10
- Lab: [`examples/optimal_transport_lab.py`](../examples/optimal_transport_lab.py)
- Decision record: [ADR-0043](adr/ADR-0043-transport-and-geometry.md)

## Track S — Statistics

Descriptive statistics, estimation, testing, Bayesian inference, modelling,
and model criticism, in that order.

### S1 — Descriptive statistics

[Descriptive statistics](foundations/descriptive-statistics.md): center,
spread, standardization, the empirical distribution, covariance and
correlation, and one-pass streaming summaries, quantiles, and exceedance
counters. A population quantity is distinct from a statistic of observed data.

- Requires: P1
- Lab: [`examples/generate_foundation_figures.py`](../examples/generate_foundation_figures.py)
- Decision record: [ADR-0055](adr/ADR-0055-streaming-accumulators.md)

### S2 — Estimation and statistical fitting

[Uncertainty modeling and statistical fitting](foundations/uncertainty-modeling-and-fitting.md):
choosing a variable before a distribution, multivariate uncertainty, maximum
likelihood with one engine for every family, calibrated goodness of fit,
ranking candidate families, joint inputs from data, and real-data case
studies.

- Requires: S1, P5, P8
- Lab: [`examples/statistical_fitting_lab.py`](../examples/statistical_fitting_lab.py), [`examples/model_selection_lab.py`](../examples/model_selection_lab.py), [`examples/real_data_fitting_lab.py`](../examples/real_data_fitting_lab.py)
- Decision record: [ADR-0027](adr/ADR-0027-generic-fitting-and-model-selection.md), [ADR-0080](adr/ADR-0080-real-data-case-studies.md)

### S3 — Hypothesis testing, intervals, and the bootstrap

[Hypothesis testing, confidence intervals, and the bootstrap](foundations/hypothesis-testing-and-bootstrap.md):
sampling distributions, t tests and the Student-t interval, count data, rank
and permutation tests, bootstrap and jackknife, power and sample size,
multiple testing, and equivalence testing.

- Requires: S2
- Lab: none; numbers asserted in [`tests/test_classical_resampling_inference.py`](../tests/test_classical_resampling_inference.py)
- Decision record: [ADR-0021](adr/ADR-0021-classical-and-resampling-inference.md)

### S4 — R bridge

[R bridge](prerequisites/r-bridge.md): R calls for distributions, descriptive
statistics, classical tests, optimization, regression, random numbers, the
bootstrap, Sobol indices, and the multivariate normal, with their `probstat`
equivalents and the parameterization and default differences.

- Requires: S3
- Lab: none; snippets run in [`tests/test_documentation_examples.py`](../tests/test_documentation_examples.py) and [`tests/test_r_bridge_examples.py`](../tests/test_r_bridge_examples.py)
- Decision record: [ADR-0028](adr/ADR-0028-prerequisite-bridges.md)

### S5 — Frequentist and Bayesian interpretation

[Frequentist and Bayesian interpretation](foundations/frequentist-and-bayesian-interpretation.md):
confidence versus credible intervals, p-values versus posterior probabilities
and Bayes factors, coverage at a fixed parameter versus calibration over the
prior, side by side on one binomial data set, and reporting rules.

- Requires: S3, P7
- Lab: none; numbers asserted in [`tests/test_measure_theory.py`](../tests/test_measure_theory.py)
- Decision record: [ADR-0071](adr/ADR-0071-measure-bridge-and-interpretation.md)

### S6 — EM, mixtures, and hierarchical models

[EM, mixtures, hierarchical models, and posterior predictive checks](foundations/hierarchical-models.md):
a generic EM driver, Gaussian mixtures, EM for censored normal data, the
eight-schools hierarchical model, beta-binomial empirical Bayes, and
posterior predictive p-values and PIT values.

- Requires: S5
- Lab: [`examples/hierarchical_lab.py`](../examples/hierarchical_lab.py)
- Decision record: [ADR-0040](adr/ADR-0040-hierarchical-em-ppc.md)

### S7 — Regression and generalized linear models

[Regression, generalized linear models, and regularization](foundations/regression-and-glms.md):
design matrices, multiple linear regression through Householder QR, logistic,
probit, Poisson and gamma GLMs by IRLS, separation, and ridge, lasso, and
elastic net with cross-validation.

- Requires: S2, M2
- Lab: [`examples/statistical_modeling_lab.py`](../examples/statistical_modeling_lab.py)
- Decision record: [ADR-0030](adr/ADR-0030-regression-and-glms.md)

### S8 — Time series

[Time series](foundations/time-series.md): correlograms, differencing and
decomposition, ARMA roots and stationarity, the exact likelihood through the
state-space form, estimation, forecasts with exact intervals, simulation,
exponential smoothing, GARCH volatility models, vector autoregressions
(order selection, impulse responses, variance decomposition, Granger tests),
ADF and KPSS unit-root tests, and CUSUM and PELT change-point detection.

- Requires: S7
- Lab: [`examples/statistical_modeling_lab.py`](../examples/statistical_modeling_lab.py)
- Decision record: [ADR-0031](adr/ADR-0031-time-series.md), [ADR-0104](adr/ADR-0104-time-series-extensions.md)

### S9 — Survival analysis

[Survival analysis for right-censored data](foundations/survival-analysis.md):
censoring and the risk set, Kaplan-Meier, median survival, restricted mean
survival time, Nelson-Aalen, the log-rank test, parametric AFT models, and Cox
proportional hazards with tied event times.

- Requires: S2
- Lab: [`examples/statistical_modeling_lab.py`](../examples/statistical_modeling_lab.py)
- Decision record: [ADR-0033](adr/ADR-0033-survival-and-conformal.md)

### S10 — Missing data

[Missing data](foundations/missing-data.md): missingness mechanisms, Little's
MCAR test, EM for the multivariate normal under MAR with observed-information
standard errors, multiple imputation by data augmentation, and Rubin's rules.

- Requires: S6, P6
- Lab: [`examples/missing_data_lab.py`](../examples/missing_data_lab.py)
- Decision record: [ADR-0047](adr/ADR-0047-missing-data.md)

### S11 — Causal inference

[Causal inference](foundations/causal-inference.md): causal graphs,
d-separation, backdoor and frontdoor identification, graph surgery, discrete
structural causal models, doubly robust effect estimation, and E-values for
unmeasured confounding.

- Requires: S7
- Lab: [`examples/causal_lab.py`](../examples/causal_lab.py)
- Decision record: [ADR-0034](adr/ADR-0034-causal-inference.md)

### S12 — Experimental design

[Optimal and Bayesian experimental design](foundations/experimental-design.md):
the information matrix, optimality criteria, the equivalence theorem,
approximate and exact D-optimal designs, local designs for nonlinear models,
expected information gain, and greedy sequential design.

- Requires: S7
- Lab: [`examples/experimental_design_lab.py`](../examples/experimental_design_lab.py)
- Decision record: [ADR-0035](adr/ADR-0035-experimental-design.md)

### S13 — Extreme value analysis

[Extreme value analysis](foundations/extreme-value-analysis.md): the
generalized Pareto law, block maxima and GEV return levels, delta-method and
profile-likelihood intervals, threshold choice, MLE and PWM fits, and
peaks-over-threshold return levels.

- Requires: S2
- Lab: [`examples/extremes_lab.py`](../examples/extremes_lab.py)
- Decision record: [ADR-0036](adr/ADR-0036-extremes-and-information.md)

### S14 — Conformal prediction

[Conformal prediction](foundations/conformal-prediction.md): what is and is
not guaranteed, the finite-sample quantile, split conformal regression,
conformalized quantile regression, jackknife+, error bars for surrogates, and
coverage checks.

- Requires: S7
- Lab: none; numbers asserted in [`tests/test_conformal_prediction.py`](../tests/test_conformal_prediction.py)
- Decision record: [ADR-0033](adr/ADR-0033-survival-and-conformal.md)

### S15 — Forecast scoring and calibration

[Scoring, calibrating, and shift-testing probabilistic forecasts](foundations/forecast-calibration.md):
proper scoring rules, the CRPS of an ensemble, calibration and sharpness, PIT
and reliability diagrams, and distribution-shift tests.

- Requires: S14
- Lab: [`examples/forecast_calibration_lab.py`](../examples/forecast_calibration_lab.py)
- Decision record: [ADR-0042](adr/ADR-0042-scoring-and-calibration.md)

### S16 — Linear and generalized linear mixed models

[Linear and generalized linear mixed models](foundations/mixed-models.md):
random intercepts, correlated random slopes, and nested factors; ML and REML
through the profiled deviance; BLUPs and conditional variances; Satterthwaite
t tests; likelihood-ratio tests with boundary and bootstrap references; and
binomial and Poisson GLMMs by the Laplace approximation and adaptive
Gauss-Hermite quadrature.

- Requires: S6, S7
- Lab: none; numbers asserted in [`tests/test_mixed_models.py`](../tests/test_mixed_models.py)
- Decision record: [ADR-0100](adr/ADR-0100-mixed-models.md)

### S17 — Count and categorical regression

[Count and categorical regression](foundations/count-and-categorical-regression.md):
Poisson regression with exposures and sandwich errors, the regression test
for overdispersion, NB2 regression, zero-inflated and hurdle models and the
limits of the Vuong statistic, the proportional-odds model with Brant's
test, the multinomial logit, and average marginal effects.

- Requires: S7
- Lab: none; numbers asserted in [`tests/test_count_and_categorical_regression.py`](../tests/test_count_and_categorical_regression.py)
- Decision record: [ADR-0103](adr/ADR-0103-count-and-categorical-regression.md)

### S18 — Smoothing and additive models

[Smoothing and additive models](foundations/smoothing-and-additive-models.md):
B-spline bases, P-splines and cubic smoothing splines with the smoothing
parameter chosen by REML or GCV, effective degrees of freedom and Bayesian
bands, LOESS with robustness iterations, isotonic regression, kernel
regression with cross-validated bandwidths, additive models, and partial
dependence.

- Requires: S7, M4
- Lab: none; numbers asserted in [`tests/test_smoothing_and_additive_models.py`](../tests/test_smoothing_and_additive_models.py)
- Decision record: [ADR-0102](adr/ADR-0102-smoothing-and-multivariate-analysis.md)

### S19 — Multivariate analysis

[Multivariate analysis](foundations/multivariate-analysis.md): principal
components on the covariance or correlation scale, maximum-likelihood
factor analysis with varimax rotation and the test of the number of
factors, linear and quadratic discriminant analysis, and k-means with the
silhouette coefficient.

- Requires: S7, P6, M2
- Lab: none; numbers asserted in [`tests/test_multivariate_analysis.py`](../tests/test_multivariate_analysis.py)
- Decision record: [ADR-0102](adr/ADR-0102-smoothing-and-multivariate-analysis.md)

## Track U — Uncertainty quantification

U1-U9 follow the ATHENS 2026 verification, validation, and UQ sequence
(UQ-0 to UQ-7); the topic ledger and implementation status live in the
[ATHENS 2026 coverage matrix](curriculum/athens-2026-coverage.md). U10-U16
extend the sequence with validation, discrepancy, figures, and the integrated
toolset in which `probstat.workflow.UQStudy` runs every method on one model
with one seed and produces a single report.

### U1 — VV-UQ foundations (UQ-0)

[Verification, validation, and UQ foundations](foundations/vv-uq-foundations.md):
the three questions, forward model and quantity of interest, aleatory and
epistemic classifications, solution verification with the Grid Convergence
Index, numerical uncertainty, prediction with separate uncertainty
components, and forward versus inverse problems.

- Requires: none
- Lab: [`examples/uq_workflow_lab.py`](../examples/uq_workflow_lab.py)
- Decision record: [ADR-0007](adr/ADR-0007-auditable-vv-uq-curriculum.md), [ADR-0026](adr/ADR-0026-solution-verification-and-uncertainty-components.md)

### U2 — Uncertainty modeling and fitting (UQ-1)

[Uncertainty modeling and statistical fitting](foundations/uncertainty-modeling-and-fitting.md),
read for UQ inputs: the required univariate families, multivariate normal
structure, MLE, confidence intervals, and goodness-of-fit limitations.

- Requires: U1, P5
- Lab: [`examples/statistical_fitting_lab.py`](../examples/statistical_fitting_lab.py)
- Decision record: [ADR-0008](adr/ADR-0008-uq1-linear-algebra-and-fit-calibration.md)

### U3 — Uncertainty propagation (UQ-2)

[Uncertainty propagation](foundations/uncertainty-propagation.md): expectation
integrals, deterministic quadrature, Monte Carlo estimators, sampling error,
evaluation cost, and threshold probabilities.

- Requires: U2
- Lab: [`examples/basic_monte_carlo_lab.py`](../examples/basic_monte_carlo_lab.py)
- Decision record: [ADR-0009](adr/ADR-0009-propagation-cost-and-error-separation.md)

### U4 — Advanced Monte Carlo (UQ-3)

[Advanced Monte Carlo](foundations/advanced-monte-carlo.md): antithetic,
control, conditional, stratified, and importance-sampling strategies with
budget allocation and cost-aware efficiency diagnostics.

- Requires: U3
- Lab: [`examples/advanced_monte_carlo_lab.py`](../examples/advanced_monte_carlo_lab.py)
- Decision record: [ADR-0010](adr/ADR-0010-variance-reduction-estimator-contracts.md)

### U5 — Reliability and rare events (UQ-4)

[Reliability and rare-event analysis](foundations/reliability-analysis.md):
limit states, direct failure probability, importance weighting, affine FORM,
the reliability index, design-point geometry, and directional sampling.

- Requires: U4
- Lab: [`examples/reliability_lab.py`](../examples/reliability_lab.py)
- Decision record: [ADR-0011](adr/ADR-0011-reliability-domain-and-form-scope.md)

### U6 — Sensitivity analysis (UQ-5)

[Sensitivity analysis](foundations/sensitivity-analysis.md): local
derivatives, functional ANOVA, Sobol main, interaction, and total effects,
ranking, dimension reduction, uncertainty-reduction decisions, and Shapley
effects for dependent inputs.

- Requires: U3
- Lab: [`examples/sensitivity_lab.py`](../examples/sensitivity_lab.py)
- Decision record: [ADR-0012](adr/ADR-0012-sensitivity-estimator-boundaries.md)

### U7 — Inverse problems and Bayesian calibration (UQ-6)

[Inverse problems and Bayesian calibration](foundations/inverse-problems-and-calibration.md):
observation and noise models, least squares, identifiability, Gaussian priors
and posteriors, predictive checking, model discrepancy, and the
calibration-versus-prediction boundary.

- Requires: U2, M3
- Lab: [`examples/bayesian_calibration_lab.py`](../examples/bayesian_calibration_lab.py)
- Decision record: [ADR-0013](adr/ADR-0013-linear-gaussian-calibration-boundaries.md)

### U8 — Nonlinear calibration and MCMC (UQ-6)

[Nonlinear calibration and MCMC](foundations/nonlinear-calibration-and-mcmc.md):
nonlinear forward models, unknown and robust noise models, Laplace
approximations, adaptive Metropolis, convergence diagnostics, Fisher-based
identifiability, and posterior push-forward prediction.

- Requires: U7, M4
- Lab: [`examples/nonlinear_calibration_lab.py`](../examples/nonlinear_calibration_lab.py)
- Decision record: [ADR-0017](adr/ADR-0017-nonlinear-calibration-and-mcmc.md)

### U9 — Filtering and data assimilation (UQ-7)

[Filtering and data assimilation](foundations/filtering-and-data-assimilation.md):
state-space models, sequential Bayesian updating, filtering versus smoothing,
extended and unscented Kalman filters, particle filters, and ensemble Kalman
data assimilation.

- Requires: U7
- Lab: [`examples/filtering_lab.py`](../examples/filtering_lab.py)
- Decision record: [ADR-0018](adr/ADR-0018-filtering-and-data-assimilation.md)

### U10 — Model validation

[Model validation and prediction credibility](foundations/model-validation.md):
validation records, predictions at experimental conditions, the area metric,
u-pooling, the reliability metric, the ASME V&V 20 comparison, Bayes factors,
and what validation cannot claim.

- Requires: U1
- Lab: [`examples/uq_workflow_lab.py`](../examples/uq_workflow_lab.py)
- Decision record: [ADR-0022](adr/ADR-0022-model-validation-metrics.md)

### U11 — Model discrepancy

[Model discrepancy and heteroscedastic noise](foundations/model-discrepancy.md):
Kennedy-O'Hagan calibration when the simulator itself is wrong, prediction by
the simulator alone versus bias-corrected prediction, the
Brynjarsdottir-O'Hagan identifiability point, and heteroscedastic noise.

- Requires: U8
- Lab: [`examples/discrepancy_calibration_lab.py`](../examples/discrepancy_calibration_lab.py)
- Decision record: [ADR-0039](adr/ADR-0039-model-discrepancy.md)

### U12 — Visual language (LAB-6)

[The visual language](foundations/visual-language.md): the five rules for
encoding data, model, and uncertainty, palettes, one uncertainty shown in six
encodings, study reports, analysis figures, and accessibility.

- Requires: U3
- Lab: [`examples/uncertainty_visualization_lab.py`](../examples/uncertainty_visualization_lab.py)
- Decision record: [ADR-0024](adr/ADR-0024-semantic-chart-theme.md), [ADR-0078](adr/ADR-0078-lab-contract-and-accessibility.md)

### U13 — The central probabilistic model

[The central probabilistic model](foundations/probabilistic-modeling.md):
marginals plus a copula, vine copulas, sampling designs, design-based
propagation, seeds and streams, cost accounting, failed runs and batch
evaluation, and the stages of one `UQStudy`.

- Requires: U3, U5, U6
- Lab: [`examples/central_model_lab.py`](../examples/central_model_lab.py), [`examples/uq_study_lab.py`](../examples/uq_study_lab.py), [`examples/uq_study_complete_lab.py`](../examples/uq_study_complete_lab.py), [`examples/uq_study_extensions_lab.py`](../examples/uq_study_extensions_lab.py)
- Decision record: [ADR-0014](adr/ADR-0014-central-probabilistic-model.md), [ADR-0058](adr/ADR-0058-model-contract-evaluation-engine.md)

### U14 — Model-based reliability

[Model-based reliability](foundations/advanced-reliability.md): HL-RF FORM
with a merit line search, SORM, design-point importance sampling, subset
simulation, systems of limit states, inverse FORM, cross-entropy importance
sampling, and line sampling.

- Requires: U5, U13
- Lab: [`examples/central_model_lab.py`](../examples/central_model_lab.py)
- Decision record: [ADR-0015](adr/ADR-0015-model-based-reliability.md)

### U15 — Adaptive reliability

[Adaptive reliability](foundations/adaptive-reliability.md): a guard for
covariance-adapted CE-IS, line sampling, multiple design points, and AK-MCS
active learning for rare events.

- Requires: U14
- Lab: [`examples/adaptive_reliability_lab.py`](../examples/adaptive_reliability_lab.py)
- Decision record: [ADR-0041](adr/ADR-0041-adaptive-reliability.md)

### U16 — Surrogate models

[Surrogate models](foundations/surrogate-models.md): polynomial chaos and
Gaussian-process emulators trained from a `ProbabilisticModel` that replace
an expensive forward model inside a new model.

- Requires: U13
- Lab: [`examples/surrogate_lab.py`](../examples/surrogate_lab.py)
- Decision record: [ADR-0019](adr/ADR-0019-surrogate-models.md)

## Track E — Systems engineering

Reliability of systems, Markov and semi-Markov models, risk and cost, and the
mission assessment that joins them.

### E1 — Exact system reliability

[Exact system reliability](foundations/system-reliability.md): structure
functions on binary decision diagrams, minimal cut and path sets and bounds,
importance measures, epistemic component reliabilities, component time models,
common-cause failures, the system model, and phased missions, all computed
exactly rather than by sampling.

- Requires: P4
- Lab: [`examples/mission_systems_lab.py`](../examples/mission_systems_lab.py)
- Decision record: [ADR-0060](adr/ADR-0060-exact-system-reliability.md)

### E2 — Markov and semi-Markov reliability

[Markov reliability and availability](foundations/markov-reliability.md):
continuous-time Markov chains solved by uniformization with error bounds,
reliability measures, repairable and standby systems, MTTF and MTBF, phased
missions, and semi-Markov sojourns.

- Requires: E1, P9
- Lab: [`examples/mission_systems_lab.py`](../examples/mission_systems_lab.py)
- Decision record: [ADR-0061](adr/ADR-0061-ctmc-markov-reliability.md), [ADR-0076](adr/ADR-0076-semi-markov-and-epistemic-ccf.md)

### E3 — Quantitative risk analysis

[Risk analysis](foundations/risk-analysis.md): risk as scenario, likelihood,
and consequence triplets, VaR and CVaR, consequence models, fault and event
trees, fragility curves and hazard convolution, decision analysis with value
of information, and the limits of risk matrices.

- Requires: U13
- Lab: [`examples/central_model_lab.py`](../examples/central_model_lab.py)
- Decision record: [ADR-0016](adr/ADR-0016-risk-analysis-layer.md)

### E4 — Risk registers, FMECA, and reliability data

[Risk registers, FMECA, and reliability data](foundations/risk-register-and-fmeca.md):
quantitative risk registers, MIL-STD-1629A criticality, the MIL-STD-882E
matrix, F-N curves, Beta and Gamma posteriors and chi-square bounds for
component data, and reliability allocation.

- Requires: E3, S2
- Lab: [`examples/mission_systems_lab.py`](../examples/mission_systems_lab.py)
- Decision record: [ADR-0063](adr/ADR-0063-risk-register-fmeca-reliability-data.md)

### E5 — Cost analysis

[Cost analysis](foundations/cost-analysis.md): discounting, life-cycle cost
models, cost-risk S-curves with correlation and reserves, learning curves,
maintenance optimisation, spares, and the cost of unreliability.

- Requires: E3
- Lab: [`examples/mission_systems_lab.py`](../examples/mission_systems_lab.py)
- Decision record: [ADR-0062](adr/ADR-0062-cost-analysis.md)

### E6 — Mission assessment

[Systems engineering guide](foundations/systems-engineering-guide.md): the
chain from component test data through Markov subsystems, the exact mission
system, dynamic fault trees, phases, and epistemic uncertainty to cost and
risk, with a method-selection table and the assumptions to state in every
report.

- Requires: E1, E2, E4, E5
- Lab: [`examples/mission_systems_lab.py`](../examples/mission_systems_lab.py)
- Decision record: [ADR-0066](adr/ADR-0066-mission-assessment-integration.md)

## Track C — Computation

The numerical layer is covered by M2 (linear algebra), M4 (optimization), and
M5 (numerical integration); Monte Carlo propagation and variance reduction by
U3 and U4; adaptive Metropolis by U8. This track builds the remaining sampling
and density-estimation methods on them.

### C1 — Multilevel Monte Carlo and Stein control variates

[Multilevel Monte Carlo](foundations/advanced-monte-carlo.md#multilevel-monte-carlo)
and [Stein control variates and control functionals](foundations/advanced-monte-carlo.md#stein-control-variates-and-control-functionals):
telescoping level corrections with coupled samples and cost-optimal level
allocation, zero-variance control variates, control functionals, and the
kernel Stein discrepancy.

- Requires: U4
- Lab: none; numbers asserted in [`tests/test_mlmc_ce_line_shapley.py`](../tests/test_mlmc_ce_line_shapley.py) and [`tests/test_stein_control_variates.py`](../tests/test_stein_control_variates.py)
- Decision record: [ADR-0029](adr/ADR-0029-mlmc-ce-line-sampling-shapley.md), [ADR-0072](adr/ADR-0072-stein-control-variates.md)

### C2 — Gradient-based MCMC and variational inference

[Gradient-based sampling and variational inference](foundations/nonlinear-calibration-and-mcmc.md#12-gradient-based-sampling-and-variational-inference):
Hamiltonian dynamics and the leapfrog integrator, static HMC and NUTS,
step-size and metric warm-up, efficiency per gradient, divergences, and
Gaussian ADVI.

- Requires: U8
- Lab: none; numbers asserted in [`tests/test_hmc_nuts_advi.py`](../tests/test_hmc_nuts_advi.py)
- Decision record: [ADR-0032](adr/ADR-0032-hmc-nuts-advi.md)

### C3 — Probabilistic programming

[Probabilistic programming](foundations/probabilistic-programming.md): a
Bayesian model written as a Python function, its intermediate representation,
NUTS, ADVI, and Laplace inference checked against exact answers, predictive
simulation, and structured and discrete latent sites.

- Requires: C2, S6
- Lab: [`examples/ppl_lab.py`](../examples/ppl_lab.py)
- Decision record: [ADR-0053](adr/ADR-0053-probabilistic-programming.md), [ADR-0082](adr/ADR-0082-ppl-structured-and-discrete-sites.md)

### C4 — Generative models

[Generative models](foundations/generative-models.md): normalizing flows
(planar, radial, affine coupling, actnorm) with exact change-of-variables
densities and maximum-likelihood fitting, energy-based models fitted by
score matching, Langevin sampling, and normalizing-constant and divergence
diagnostics against a known truth.

- Requires: M4, P5
- Lab: none; numbers asserted in [`tests/test_generative_flows_energy.py`](../tests/test_generative_flows_energy.py)
- Decision record: [ADR-0074](adr/ADR-0074-normalizing-flows-and-energy-models.md)

### C5 — Score-based diffusion and neural SDEs

[Score-based diffusion and neural SDEs](foundations/stochastic-processes-and-random-fields.md#6-score-based-diffusion-and-neural-sdes):
variance-preserving and variance-exploding forward SDEs, reverse-time SDE,
predictor-corrector, and probability-flow samplers, denoising score matching,
and neural SDEs fitted by the Euler pseudo-likelihood.

- Requires: C4, P9
- Lab: none; numbers asserted in [`tests/test_score_diffusion_neural_sde.py`](../tests/test_score_diffusion_neural_sde.py)
- Decision record: [ADR-0075](adr/ADR-0075-score-diffusion-and-neural-sdes.md)

## Status

Completion and sequencing live in the [roadmap](roadmap.md). Topics remain on
the roadmap even when their teaching material has not yet been written. The
equations behind every step are collected by package in the
[mathematical reference](math-reference/index.md).
