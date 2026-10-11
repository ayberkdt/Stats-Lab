# Architecture

## Dependency direction

The repository is a directed acyclic package graph organized in numbered
layers. A module may import modules in its own or a lower layer, never a
higher one; `tests/test_architecture.py` enforces the map below and rejects
cycles.

| Layer | Packages | Role |
|---|---|---|
| 0 | `foundations`, `numerics`, `linalg` | validation, protocols, random streams, quadrature, dense linear algebra |
| 1 | `probability`, `statistics`, `distributions`, `optimization`, `sampling` | probability rules, descriptive and order statistics, distribution families, solvers, space-filling designs |
| 2 | `inference`, `multivariate`, `uq` | estimation, Bayesian updating, MCMC, multivariate laws, UQ vocabulary and verification |
| 3 | `monte_carlo`, `modeling` | estimators and work accounting; **the central `ProbabilisticModel`** |
| 4 | `propagation`, `inverse` | forward propagation; calibration and posterior prediction |
| 5 | `reliability`, `sensitivity`, `surrogates`, `validation` | failure probability, sensitivity indices, emulators, validation evidence |
| 6 | `risk`, `filtering`, `data_assimilation` | risk measures and decision analysis; sequential state estimation |
| 7 | `workflow` | `UQStudy`: every analysis on one model, one seed tree, one report |
| 9 | `visualization` | optional themed figures (Matplotlib) |

```text
                         visualization (9)
                               |
                          workflow (7)
              _________________|__________________
             |                 |                  |
          risk (6)      filtering (6)   data_assimilation (6)
             |
   reliability · sensitivity · surrogates · validation   (5)
             |                                  |
       propagation (4) ----------------- inverse (4)
             |                                  |
          modeling (3)  <-- the central model  -+--- monte_carlo (3)
             |
   inference · multivariate · uq                  (2)
             |
   probability · statistics · distributions · optimization · sampling   (1)
             |
   foundations · numerics · linalg               (0)
```

`probstat.__init__` exposes version metadata only. This keeps domain ownership
visible and avoids a root import hub that could become a cycle.

## The central model

`modeling.ProbabilisticModel` binds three things: a `JointInput` (named
marginals plus a Gaussian copula), a `ForwardModel` mapping named inputs to
named outputs, and the quantities of interest. Every workflow from layer 4
upward accepts it:

| Question | Call |
|---|---|
| What does the output look like? | `propagation.propagate(model, n, design=...)` |
| Which inputs matter? | `sensitivity.sobol_analysis`, `sensitivity.morris_screening` |
| How likely is failure? | `reliability.form_hlrf`, `sorm`, `subset_simulation`, `compare_reliability_methods` |
| What is the loss at risk? | `risk.assess_risk` |
| What do the data say about the inputs? | `inverse.CalibrationProblem(model, experiments, noise)` |
| What will we observe next? | `inverse.predict(problem, posterior, ...)` |
| Can a cheap emulator stand in? | `surrogates.fit_polynomial_chaos`, `fit_gaussian_process_to_model` |
| All of the above, reproducibly? | `workflow.UQStudy(model, seed=...)` |

The Nataf transformation inside `JointInput` is the single bridge between
physical inputs, latent normals, independent standard normals (used by FORM,
SORM, subset simulation, MCMC), and unit-hypercube designs (used by Monte
Carlo, LHS, Halton, Sobol). See ADR-0014 and
[the central-model guide](foundations/probabilistic-modeling.md).

## Layer responsibilities

| Layer | Owns | Must not own |
|---|---|---|
| `foundations` | Validation, structural protocols, SplitMix64 random streams | Domain distributions or estimators |
| `numerics` | Deterministic quadrature and numerical diagnostics | Probability models |
| `linalg` | Dense Cholesky, triangular, LU, Jacobi eigen, log-determinants | Statistical meaning |
| `optimization` | BFGS, Nelder-Mead, Levenberg-Marquardt, Brent, finite differences | Likelihoods or limit states |
| `sampling` | MC, Latin hypercube, Halton, Sobol designs on the unit cube | Marginal transformations |
| `probability` | Sets, counting, probability rules, random-variable algebra | Inference or plotting |
| `statistics` | Descriptive, order, and rank statistics | Distribution implementations |
| `distributions` | Distribution objects, special functions, sampling | Rendering |
| `inference` | Estimation, confidence, tests, Bayesian conjugacy, MCMC, Laplace | Forward-model semantics |
| `multivariate` | Vector-valued distributions | Inference workflows |
| `uq` | QoI and forward-model contracts, numerical verification | Monte Carlo, reliability, inference |
| `monte_carlo` | Sample estimators, variance reduction, work accounting | Forward-model semantics |
| `modeling` | Joint input laws, Nataf, `ProbabilisticModel`, evaluation counting | Analysis methods |
| `propagation` | Design-based propagation, output distributions | Distribution implementations |
| `inverse` | Observation/noise models, calibration problems, posterior prediction | Validation claims |
| `reliability` | Limit states, FORM/SORM, IS, subset simulation, systems | Generic probability foundations |
| `sensitivity` | Local derivatives, Sobol, Morris, ranking and screening | Input-distribution implementations |
| `surrogates` | PCE and GP emulators and the surrogate forward-model adapter | Analyses on the emulator |
| `validation` | Discrepancy and validation evidence | Parameter inference |
| `risk` | Risk measures, losses, fault/event trees, fragility, decisions | Reliability algorithms |
| `filtering`, `data_assimilation` | State-space models, Kalman/EKF/UKF, particle filter, EnKF, 3D-Var | Plotting |
| `workflow` | Orchestration, stage records, reports | New numerical methods |
| `visualization` | Themed figures and annotations | Mathematical truth or validation rules |

Structural interfaces such as `ContinuousProbabilityLaw` live in
`foundations.protocols`. Analytic distributions and induced finite laws conform
by structure; neither domain package imports the other. Density, log-density,
survival, quantile, inverse-survival, and hazard are narrow capabilities that a
type implements only when they are mathematically defined and numerically
supported.

Sibling packages in one layer may import each other only when that does not
create a cycle and an ADR justifies it; the default is that they do not.
Cross-cutting code moves down only when it is truly domain-independent.

## Enforcement

`tests/test_architecture.py` builds the module import graph from Python's AST
and rejects both upward imports and cycles. `tests/test_documentation_contract.py`
requires docstrings with references on every definition and resolves every
`EQ-*` and `REF-*` identifier against the registry and bibliography.
`tests/test_athens_coverage.py` keeps the 110-topic curriculum ledger honest.
CI (`.github/workflows/ci.yml`) runs ruff, strict mypy, the test suite, and
every example lab on Python 3.11-3.13.
