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
| 2 | `inference`, `multivariate`, `uq`, `processes`, `information` | estimation, regression, survival, Bayesian computation, multivariate laws, stochastic processes, information measures, UQ vocabulary and verification |
| 3 | `monte_carlo`, `modeling`, `causal`, `generative` | estimators and work accounting; **the central `ProbabilisticModel`**; causal graphs and effect estimation; normalizing flows and energy-based models |
| 4 | `propagation`, `inverse` | forward propagation; calibration (with model discrepancy) and posterior prediction |
| 5 | `reliability`, `sensitivity`, `surrogates`, `validation`, `design` | failure probability, sensitivity indices, emulators, validation evidence, experimental design |
| 6 | `risk`, `systems`, `filtering`, `data_assimilation`, `conformal` | risk measures, extremes, and decision analysis; exact system reliability on BDDs; sequential state estimation; conformal prediction |
| 7 | `workflow`, `timeseries` | `UQStudy`: every analysis on one model, one seed tree, one report; ARIMA models on the Kalman filter |
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
| `linalg` | Dense Cholesky, triangular, LU, Jacobi eigen, log-determinants; the dense-backend boundary | Statistical meaning |
| `optimization` | BFGS, Nelder-Mead, Levenberg-Marquardt, Brent, finite differences | Likelihoods or limit states |
| `sampling` | MC, Latin hypercube, Halton, Sobol designs on the unit cube | Marginal transformations |
| `probability` | Sets, counting, probability rules, random-variable algebra | Inference or plotting |
| `statistics` | Descriptive, order, and rank statistics | Distribution implementations |
| `distributions` | Distribution objects, special functions, sampling | Rendering |
| `inference` | Estimation, tests, regression and GLMs, smoothing and additive models, multivariate analysis and clustering, survival, EM and mixtures, hierarchical models, MCMC/HMC/NUTS, ADVI, Laplace | Forward-model semantics |
| `multivariate` | Vector-valued distributions | Inference workflows |
| `uq` | QoI and forward-model contracts, numerical verification | Monte Carlo, reliability, inference |
| `processes` | Markov chains, Poisson processes, SDEs, random fields | Filtering or inference |
| `information` | Entropy, divergences, mutual information and their estimators, optimal transport, Fisher information geometry | Hypothesis tests |
| `monte_carlo` | Sample estimators, variance reduction, work accounting | Forward-model semantics |
| `modeling` | Joint input laws, Nataf, `ProbabilisticModel`, evaluation counting | Analysis methods |
| `causal` | Causal graphs, identification, discrete interventions, IPW/AIPW, E-values | Simulator-based analyses |
| `generative` | Normalizing flows, energy-based models, score matching, Langevin sampling, normalizing-constant diagnostics | Shared networks, derivatives, and optimizers (`numerics`, `optimization`) |
| `propagation` | Design-based propagation, output distributions | Distribution implementations |
| `inverse` | Observation/noise models, calibration problems, Kennedy-O'Hagan discrepancy, posterior prediction | Validation claims |
| `reliability` | Limit states, FORM/SORM, multiple design points, IS, subset and line sampling, AK-MCS, systems | Generic probability foundations |
| `sensitivity` | Local derivatives, Sobol, Morris, ranking and screening | Input-distribution implementations |
| `surrogates` | PCE and GP emulators and the surrogate forward-model adapter | Analyses on the emulator |
| `validation` | Discrepancy and validation evidence, proper scoring rules, forecast calibration, distribution-shift tests | Parameter inference |
| `design` | Optimal and Bayesian experimental design | Calibration itself |
| `risk` | Risk measures, losses, fault/event trees, fragility, extremes, decisions | Reliability algorithms |
| `systems` | Structure functions on BDDs, cut sets, importance, CCF, component time models, MTTF, phased missions | Sampling-based reliability; nothing in `risk` imports it |
| `conformal` | Distribution-free prediction intervals and coverage reports | Model fitting |
| `filtering`, `data_assimilation` | State-space models, Kalman/EKF/UKF, particle filter, EnKF, 3D-Var | Plotting |
| `workflow` | Orchestration, stage records, reports | New numerical methods |
| `timeseries` | ACF/PACF, ARIMA likelihood and forecasts, smoothing | Generic filtering |
| `visualization` | Themed figures and annotations | Mathematical truth or validation rules |

Structural interfaces such as `ContinuousProbabilityLaw` live in
`foundations.protocols`. Analytic distributions and induced finite laws conform
by structure; neither domain package imports the other. Density, log-density,
survival, quantile, inverse-survival, and hazard are narrow capabilities that a
type implements only when they are mathematically defined and numerically
supported.

Sibling packages in one layer may import each other only when that does not
create a cycle and an ADR justifies it; the default is that they do not. The one
justified case so far: `reliability.active_learning` (AK-MCS) imports
`surrogates` for its Gaussian-process limit state (ADR-0041); `surrogates`
must therefore never import `reliability`.
Cross-cutting code moves down only when it is truly domain-independent.

## Module layout

Within a package, one module holds one coherent concept, and its name says
which (ADR-0073). Modules whose name starts with an underscore are private;
the public names are the ones a package re-exports.

- `distributions` has one module per family: `bernoulli`, `beta`,
  `binomial`, `categorical`, `cauchy`, `chi_square`, `discrete_uniform`,
  `empirical`, `exponential`, `fisher_f`, `gamma`, `geometric`, `gev`,
  `gpd`, `gumbel`, `hypergeometric`, `laplace`, `logistic`, `lognormal`,
  `negative_binomial`, `normal`, `pareto`, `pert`, `poisson`, `rayleigh`,
  `student_t`, `triangular`, `truncated_normal`, `uniform`, `weibull`. The
  composite constructions live one per module as well, `Affine` in
  `affine`, `Mixture` in `mixture`, and `Truncated` in `truncated`, with
  their shared dispatch helpers in the private `_wrapped` (ADR-0090);
  supports and parameter constraints in `support` and `constraints`; shared
  numerical kernels in the private `_base`, `_continuous`, and `_special`;
  the automatic-differentiation helpers of the log-densities in the private
  `_autodiff` (ADR-0090).
- `workflow.study` is a subpackage that composes `UQStudy` from one mixin
  per stage domain:

  | Module | Contents |
  |---|---|
  | `_records` | `STAGE_STATUSES`, `StageRecord`, and the stage result records |
  | `_context` | `_StageContext`, `_Accumulators`, shared formatting and validation helpers |
  | `_base` | `_StudyBase` (declared state, streams, budget, stage records) and the `_stage` decorator |
  | `_propagation` | `propagate`, `propagate_until` |
  | `_sensitivity` | `sensitivity`, `shapley`, `screening`, `information_sensitivity` |
  | `_reliability` | `limit_state`, `reliability`, `design_points`, `active_reliability`, `form`, `risk`, `extremes` |
  | `_calibration` | `calibrate`, `calibrate_with_discrepancy`, `predict`, `emulate`, `adopt_posterior` |
  | `_validation` | `verify`, `numerical_uncertainty`, `validate` |
  | `_design` | `design` |
  | `_reporting` | `provenance`, `to_dict`, `to_json`, `save_json`, `report` |
  | `_core` | `UQStudy`: constructor, `from_data`, properties, `latest`, `with_inputs` |

  Every mixin derives from `_StudyBase`, so each one type-checks on its own;
  no mixin imports `_core`.
- `optimization` has one module per solver family (ADR-0096):
  `finite_differences` (gradients, Jacobians, Hessians), `quasi_newton`
  (BFGS with the strong-Wolfe line search), `nelder_mead`, `multistart`,
  `nonlinear_least_squares` (Levenberg-Marquardt and the Gauss-Newton
  covariance), `root_finding` (Brent), and `stochastic` (Adam); `bounds`
  holds `BoxTransform`, `results` the four result records, and the private
  `_common` the callable aliases and value validation they share.
  `quasi_newton` and `nonlinear_least_squares` take their derivatives from
  `finite_differences`, `multistart` runs `quasi_newton` or `nelder_mead`,
  and no other family module imports another.
- `visualization` names each module after the figures it draws:
  `foundation_plots` (PMFs, CDFs, characteristic functions, finite measures,
  Monte Carlo convergence), `fit_plots`, `reliability_plots`, and so on;
  `theme` holds the shared visual language.

## Dense backend boundary

The `n^3` consumers reach their dense kernels through
`linalg.DenseBackend` (ADR-0081). The protocol covers Cholesky with an
explicit jitter argument, triangular and SPD solves, log-determinants, SPD
inverses, matrix products, and symmetric eigendecomposition; `factorize`
returns a `CholeskyFactorization` that keeps the backend's native factor for
repeated solves. Every method takes and returns tuples.

| Backend | Selection | Implementation |
|---|---|---|
| `PythonBackend` (`"python"`) | default | the `linalg.dense` routines; bit-identical results |
| `NumPyBackend` (`"numpy"`) | `set_backend("numpy")` or `with use_backend("numpy"):` | LAPACK Cholesky and `eigh`, BLAS products; NumPy imported on selection only (optional extra `numerics`) |

`multivariate.MultivariateNormal` and `surrogates.gaussian_process` use the
active backend; an object built under a backend keeps that backend's
factorization. The Gaussian-process module keeps its specialized
pure-Python kernels under the reference backend. Other packages call
`linalg.dense` directly. Selection is process-wide state owned by
`linalg`; no other package stores or switches it.

## Enforcement

`tests/test_architecture.py` builds the module import graph from Python's AST
and rejects both upward imports and cycles. `tests/test_documentation_contract.py`
requires docstrings with references on every definition and resolves every
`EQ-*` and `REF-*` identifier against the registry and bibliography.
`tests/test_athens_coverage.py` keeps the 110-topic curriculum ledger honest.
CI (`.github/workflows/ci.yml`) runs ruff, strict mypy, the test suite, and
every example lab on Python 3.11-3.13. The test step runs
`tools/ci_test_report.py`, which uses the same `unittest` discovery as
`python -m unittest discover -s tests` [REF-PYTHON-UNITTEST] and reports each
failing test as a run annotation, so a failure is readable without the full
log.
