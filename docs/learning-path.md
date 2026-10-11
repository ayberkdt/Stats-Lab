# Learning Path

The ATHENS verification, validation, and uncertainty-quantification track starts
with [VV-UQ foundations](foundations/vv-uq-foundations.md). Its complete topic
ledger and implementation status live in the
[ATHENS 2026 coverage matrix](curriculum/athens-2026-coverage.md).

Continue with
[uncertainty modeling and statistical fitting](foundations/uncertainty-modeling-and-fitting.md)
for required univariate families, multivariate normal structure, MLE,
confidence intervals, and goodness-of-fit limitations.

Then study
[uncertainty propagation](foundations/uncertainty-propagation.md) for
expectation integrals, deterministic quadrature, Monte Carlo estimators,
sampling error, evaluation cost, threshold probabilities, and a reproducible
forward-model laboratory.

Continue with
[advanced Monte Carlo](foundations/advanced-monte-carlo.md) for antithetic,
control, conditional, stratified, and importance-sampling strategies together
with budget-allocation and cost-aware efficiency diagnostics.

Then study [reliability and rare-event analysis](foundations/reliability-analysis.md)
for limit states, direct failure probability, importance weighting, affine FORM,
reliability index, design-point geometry, and directional sampling.

Then study [sensitivity analysis](foundations/sensitivity-analysis.md) for local
derivatives, functional ANOVA, Sobol main/interaction/total effects, ranking,
dimension reduction, and uncertainty-reduction decisions.

Continue with [inverse problems and Bayesian calibration](foundations/inverse-problems-and-calibration.md)
for observation/noise models, least squares, identifiability, Gaussian priors
and posteriors, predictive checking, model discrepancy, and the explicit
calibration-versus-prediction boundary.

Then move to [nonlinear calibration and MCMC](foundations/nonlinear-calibration-and-mcmc.md)
for nonlinear forward models, unknown and robust noise models, Laplace
approximations, adaptive Metropolis, convergence diagnostics, Fisher-based
identifiability, and posterior push-forward prediction.

With those foundations, switch to the integrated toolset built on one model:

- [The central probabilistic model](foundations/probabilistic-modeling.md):
  correlated non-normal inputs, the Nataf transformation, quasi-random
  designs, and design-based propagation.
- [Model-based reliability](foundations/advanced-reliability.md): HL-RF FORM,
  SORM, design-point importance sampling, subset simulation, system bounds,
  and inverse FORM.
- [Risk analysis](foundations/risk-analysis.md): VaR/CVaR, loss models, fault
  and event trees, fragility curves, and decision value of information.
- [Surrogate models](foundations/surrogate-models.md): polynomial chaos and
  Gaussian-process emulators that replace an expensive forward model.
- [Filtering and data assimilation](foundations/filtering-and-data-assimilation.md):
  Kalman, extended/unscented Kalman, particle and ensemble Kalman filters.

`probstat.workflow.UQStudy` runs all of these on one model with one seed and
produces a single report.

The library separates conceptual order from software layer order. Readers can
learn progressively without losing topics needed later.

## Stage 0 — Language of uncertainty

Learn experiments, outcomes, sample spaces, events, set operations, and the
three probability axioms. Understand that probability is a measure on events,
not a synonym for frequency or confidence.

Read: [Probability foundations](foundations/probability.md).

Code: `probstat.probability.FiniteSampleSpace`, `FiniteEvent`, and
`FiniteProbabilityMeasure` make these ideas executable for finite models.

## Stage 1 — Counting and finite probability

Use the multiplication rule, permutations, and combinations to count equally
likely finite outcomes. Learn when an ordering matters and when it does not.

Code: `probstat.probability.counting`.

Then use conditional probability, independence, total probability, and Bayes'
rule on validated events before moving to random variables.

## Stage 2 — Data summaries

Distinguish a population quantity from a statistic computed on observed data.
Study center, spread, standardization, empirical distributions, covariance, and
correlation. Always inspect plots and assumptions alongside a scalar summary.

Read: [Descriptive statistics](foundations/descriptive-statistics.md).

## Stage 3 — Random variables and distributions

Treat a random variable as a measurable mapping from outcomes to values. Start
with Bernoulli mass and normal density, then add common discrete and continuous
families. Distinguish PMFs, PDFs, CDFs, expectations, and samples.

Code: `probstat.distributions`.

Read: [Finite random variables and joint laws](foundations/random-variables.md).
The executable route moves from mappings and induced PMFs to transformations,
moments, joint laws, marginalization, conditional PMFs, and dependence.
It then introduces PGFs, MGFs, and characteristic functions as three distinct
transforms with different mathematical domains and numerical behavior.

Then read: [Discrete distribution foundations](foundations/discrete-distributions.md).
It separates parameter constraints, support, PMF/log-PMF, categorical label
semantics, finite versus infinite count support, and stable tail computation.

Continue with: [Waiting counts, tails, quantiles, and continuous foundations](foundations/waiting-and-continuous-distributions.md).
It fixes Geometric/Negative Binomial parameterization, connects discrete and
continuous waiting, and explains why survival and quantile need explicit
numerical contracts.

## Stage 4 and beyond

The ordered route is:

1. conditional probability, independence, Bayes' rule;
2. expectation, transformations, joint and conditional distributions;
3. laws of large numbers and central limit theorems;
4. estimation, uncertainty quantification, testing, and decision theory;
5. regression, generalized linear models, and multivariate statistics;
6. measure-theoretic probability and stochastic processes;
7. Bayesian computation, Monte Carlo, variational inference, and probabilistic
   programming;
8. causal inference, information geometry, diffusion/flow models, sequential
   Monte Carlo, and modern uncertainty-aware learning systems.

Completion and sequencing live in `memory/ROADMAP.md`. Topics remain on the
roadmap even when their teaching material has not yet been written.

