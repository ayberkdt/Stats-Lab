# Learning Path

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

