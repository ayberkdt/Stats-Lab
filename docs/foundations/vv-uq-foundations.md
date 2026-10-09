# Verification, Validation, and UQ Foundations

## The three questions

Verification, validation, and uncertainty quantification answer different
questions and must not be collapsed into a single accuracy score.

- **Solution verification:** Are the chosen equations being solved correctly?
  It studies discretization, iteration, truncation, and roundoff error.
- **Model validation:** Are those equations sufficiently representative of the
  real system for a stated use? It compares model predictions with relevant
  observations while accounting for uncertainty on both sides.
- **Uncertainty quantification:** Given uncertain inputs, numerical error, data,
  and model limitations, what uncertainty should be assigned to a quantity of
  interest?

Validation is conditional on intended use, operating regime, observations, and
quantities of interest. It never proves that a model is universally true.

## Forward model and quantity of interest

The first software contract is the named deterministic map

`Y = g(X)` ([EQ-UQ-001]),

where `X` is one finite parameter realization and `Y` is a set of model
responses. A **quantity of interest (QoI)** is a named component or derived
summary used for a scientific or decision purpose. `ForwardModel`,
`QuantityOfInterest`, and `evaluate_model` make names and finite-value checks
explicit. A stochastic simulator should expose its random stream as an input
or present each seeded realization as a deterministic call.

A forward problem predicts `Y` from `X`. An inverse problem uses observations
of outputs to learn inputs, latent states, or model parameters. Inversion needs
an observation model and assumptions about noise; identifiability is not
guaranteed merely because a forward function exists.

## Two independent uncertainty classifications

`UncertaintyNature` records interpretation:

- **Aleatory:** variability represented as inherent under the chosen model.
- **Epistemic:** incomplete knowledge that may be reduced by information.

`UncertaintySourceKind` records engineering origin:

- parametric,
- model form,
- numerical, or
- observation.

These axes are deliberately separate. A material parameter may describe
population variability (aleatory) or imperfectly known calibration
(epistemic). The label depends on the modeled system and decision context, not
the variable name alone.

## Numerical verification

For three solutions obtained with a common uniform refinement ratio `r`, the
observed order is estimated by [EQ-VV-001]. Under a single leading error term
and asymptotic convergence, Richardson extrapolation [EQ-VV-002] predicts the
zero-step-size value.

Those assumptions are substantive. Oscillatory convergence, changing meshes,
mixed error terms, iterative error, noisy simulators, or a result outside the
asymptotic range can make the estimates misleading. At least three resolutions
and a convergence study are evidence, not a ritual.

`NumericalErrorBudget` records non-negative absolute bounds for truncation,
roundoff, discretization, and iterative error. Its sum [EQ-VV-003] is a
conservative triangle-inequality bound. It is not a probabilistic standard
deviation and introduces no independence assumption.

## Scope boundary

This foundation does not yet implement experimental validation metrics,
probability-box propagation, Monte Carlo estimators, model discrepancy,
calibration, reliability, sensitivity analysis, or filtering. Those outcomes
are individually tracked in the
[ATHENS coverage matrix](../curriculum/athens-2026-coverage.md); a tracked topic
is not represented as completed until tested implementation evidence exists.

## Sources

- [REF-SMITH-UQ-2024, Chapters 1-4].
- [REF-OBERKAMPF-ROY-2025, Chapters 1-2 and 6-8].
- [REF-ROACHE-1998, Chapters 4-5].
