# ADR-0035 — Optimal and Bayesian experimental design

- Status: Accepted
- Date: 2026-10-12

## Context

`memory/ROADMAP.md` lists "optimal experiment design" as an open UQ-6 item.
Calibration (`probstat.inverse.CalibrationProblem`) already declares a prior
through `ProbabilisticModel.inputs`, experimental conditions, and
`NoiseModel`s. Design means choosing those conditions *before* the data
exist. Two traditions are needed:

- classical design, which optimizes the information matrix of a linear(ized)
  model; and
- Bayesian design, which maximizes the expected information gain (EIG).

The library is pure Python. Nested Monte Carlo is therefore expensive, and
the cost of every estimator has to be visible.

## Decision

1. **New package `probstat.design` at layer 5.** It imports `inverse` (4),
   `modeling` (3), `uq`, `optimization`, `linalg`, and `foundations`. It
   imports no sibling layer-5 package, and nothing imports it. It has two
   modules: `classical` and `bayesian`.
2. **Approximate designs are the primary object.**
   - `ApproximateDesign(points, weights)` renormalizes weights that sum to
     one within `1e-8`.
   - `ExactDesign(points, counts)` holds integer replicates.
   - A scalar is read as a one-dimensional point.
   - Bases are plain callables `f(x) -> R^p`, such as `polynomial_basis` or
     the frozen `LocalLinearization`. The latter counts response
     evaluations in a shared `EvaluationCounter`, like `ProbabilisticModel`.
3. **Only D is optimized; A, E, and I are evaluated.**
   - The multiplicative algorithm is the default.
   - Fedorov-Wynn with the exact line-search step is the alternative.
   - Both stop on the equivalence-theorem criterion
     `max d <= p (1 + tol)`. This makes the stopping rule a certificate:
     the D-efficiency is at least `exp(-tol)`.
   - Basis vectors are cached, so `basis_evaluations` equals the number of
     candidates.
   - Grid artefacts are removed by an explicit `condensed()` step, never
     silently.
4. **Exact designs.** Pukelsheim-Rieder efficient rounding is followed by an
   optional best-improvement Fedorov exchange. The exchange recomputes
   `log det` directly instead of using rank-one update formulas, because
   `p` is small and this keeps the code auditable.
5. **Bayesian problems reuse calibration declarations.**
   - `BayesianDesignProblem(model, noise)` requires known-scale Gaussian
     noise, and `from_calibration` converts a `CalibrationProblem`.
   - Unknown scales and Student-t noise are rejected. Their EIG would need
     an additional marginalization that is not implemented.
   - Conditions merge over `model.fixed` exactly as in calibration and must
     not shadow parameters.
6. **Three EIG estimators with stated bias.**
   - *Linearized and Laplace* work in the prior's standard-normal space,
     where the prior is `N(0, I)`. The EIG is reparameterization invariant,
     so the closed form `1/2 log det(I + J_u'J_u)` applies directly, with no
     prior-density Jacobian. Both are exact for linear models with Gaussian
     priors.
   - *Nested MC* has two modes:
     - Fresh inner samples (Ryan 2003): biased upward, `N (1 + M)` model
       evaluations per condition.
     - Outer samples reused as the inner sample (Huan-Marzouk 2013): biased
       downward, never above `log N`, `N` evaluations. Each term is a
       prior-contrastive bound (Foster et al. 2020).
   - Every estimate carries its method, sample sizes, model-evaluation
     count, and a `bias` label. The standard error covers the outer
     sampling only.
7. **Greedy selection with common random numbers.**
   - All candidates in one step use the same stream child
     `("step", k)`, so the comparisons share their Monte Carlo noise.
   - Repeats are allowed by default, because replicate measurements are
     legitimate.
   - Ties go to the earliest candidate.
   - `baseline` conditions plan *additional* runs; by the chain rule this
     maximizes the conditional information.

## Consequences

- The local D-optimal design `{0, 1/k}` for exponential decay, the classical
  `{-1, 0, 1}` quadratic design, and the endpoint design for a straight line
  are reproduced and certified by the equivalence theorem.
- Greedy Laplace design on the decay problem picks `t = 0, 1/k` without
  fixing `k`. Nested MC is affordable only for small `N` and `M` in pure
  Python. In development, a fresh-inner run with `N = 2000`, `M = 200`, and
  three conditions made 1.2 million model evaluations and took about 22 s.
- Not done:
  - continuous optimization of support points;
  - A-, E-, and I-optimal algorithms;
  - heteroscedastic or correlated errors in classical designs;
  - batch (non-greedy) Bayesian optimization;
  - multilevel or importance-sampled EIG estimators.

## Verification

- `tests/test_experimental_design.py` has 16 tests and runs in about 5 s.
  It covers:
  - the known designs;
  - closed-form A/E/I values;
  - the identity `sum w_i d_i = p`;
  - the efficiency bound;
  - rounding and exchange;
  - the closed-form EIG against an LU determinant;
  - nested MC within 4 SE of the closed form;
  - both bias directions at EIG approximately 6.2 nats;
  - EIG monotone in noise reduction;
  - greedy picks of extreme points for a slope and of `{0, 1/k}` for
    decay.
- Numbers quoted in `docs/foundations/experimental-design.md` come from
  those runs.
