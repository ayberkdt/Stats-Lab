# ADR-0019 — Surrogate models

- Status: Accepted
- Date: 2026-10-11

## Context

Expensive forward models make propagation, reliability, and sensitivity
studies costly. A surrogate layer must be built on the central
`ProbabilisticModel`, so that one input-law definition drives both the
training design and the downstream analyses. A fitted surrogate must also be
able to replace the model without changes to those analyses.

## Decision

1. New layer-5 package `probstat.surrogates`. It may import `modeling` (3),
   `propagation` (4, for `make_design`), `sampling`, `optimization`, `linalg`,
   `distributions`, and `foundations`. It may not import its layer-5 siblings
   `sensitivity`, `reliability`, or `validation`.
2. Polynomial chaos:
   - Hermite basis in the Nataf standard-normal space, valid for any marginals
     and Gaussian copula.
   - Legendre basis only for independent `Uniform` marginals; any other input
     law raises.
   - Total-degree and hyperbolic (`q`-norm) truncation.
   - Ordinary least squares through Cholesky factorization of the information
     matrix, with closed-form LOO error from the hat matrix.
   - Analytic mean, variance, and first, second, and total Sobol indices.
   - Sobol indices raise for dependent inputs instead of attributing variance
     to Nataf variables.
3. Gaussian process:
   - Squared-exponential and Matérn 5/2 ARD kernels.
   - Outputs standardized by sample mean and standard deviation.
   - An explicit positive `nugget` in standardized units, optionally plus an
     estimated noise variance.
   - Type-II maximum likelihood by BFGS with the exact gradient and seeded
     restarts.
   - Log parameters confined to `|theta| <= 25`; points outside are infeasible
     for the line search.
   - Closed-form LOO.
4. `SurrogateModel` implements the `ForwardModel` call contract over a mapping
   QoI -> surrogate (anything with `predict(x)`).
   `surrogate_probabilistic_model` returns a `ProbabilisticModel` with the
   same `JointInput` and QoI declarations. Fixed parameters are not copied
   because the surrogate embeds them.
5. Regression inner products use `math.sumprod` when available (Python 3.12+)
   and fall back to `math.fsum` products otherwise. They are private helpers;
   Cholesky factorization still comes from `probstat.linalg`.

## Consequences

Propagation and pick-freeze Sobol run unchanged on emulators, as the tests
demonstrate. Accuracy is validated by LOO errors and by the analytic Ishigami
indices, which degree-10 PCE matches to about `1e-3`. Pure-Python cost limits
PCE to a few hundred terms and GP to tens of training points. Results computed
on a surrogate inherit its approximation error, and that error must be
reported next to them.

## Rejected alternatives

- Sparse/adaptive PCE (LARS, OMP): more accurate per run, but a substantial
  separate algorithm; deferred.
- Automatic nugget growth on Cholesky failure: hides ill-conditioning and
  changes the model without the user's knowledge.
- Embedding surrogate fitting inside `sensitivity` or `reliability`: it would
  couple layer-5 siblings and duplicate design logic.
