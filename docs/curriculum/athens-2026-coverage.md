# ATHENS 2026 VV-UQ Curriculum Coverage

This matrix is the canonical scope ledger for the supplied ATHENS 2026
verification, validation, and uncertainty-quantification curriculum. Coverage
means that every topic is tracked; it does **not** mean every topic is already
implemented.

Status semantics:

- `Implemented`: a documented, cited, tested implementation exists at the
  evidence path.
- `Partial`: a prerequisite or first usable slice exists, but the complete
  curriculum outcome does not.
- `Planned`: the topic has an assigned phase and package owner but no qualifying
  implementation yet.

The curriculum is complete only when all 110 rows are `Implemented`, each with
repository evidence. `tests/test_athens_coverage.py` guards identity, count,
status, ownership, target, and evidence-path integrity.

| ID | Topic | Status | Target | Planned owner | Evidence |
|---|---|---|---|---|---|
| ATHENS-1.1 | Numerical simulation as a model-based computational experiment | Partial | UQ-0 | `probstat.uq` | `src/probstat/uq/taxonomy.py` |
| ATHENS-1.2 | Solution verification: solving the equations right | Partial | UQ-0 | `probstat.uq.verification` | `src/probstat/uq/verification.py` |
| ATHENS-1.3 | Model validation: solving the right equations for intended use | Planned | UQ-0 | `probstat.validation` | — |
| ATHENS-1.4 | Uncertainty quantification workflow | Partial | UQ-0 | `probstat.uq` | `docs/foundations/vv-uq-foundations.md` |
| ATHENS-1.5 | Numerical uncertainty | Partial | UQ-0 | `probstat.uq.verification` | `src/probstat/uq/verification.py` |
| ATHENS-1.6 | Parametric uncertainty | Partial | UQ-0 | `probstat.uq` | `src/probstat/uq/taxonomy.py` |
| ATHENS-1.7 | Model-form uncertainty | Partial | UQ-0 | `probstat.validation` | `src/probstat/uq/taxonomy.py` |
| ATHENS-1.8 | Aleatory versus epistemic uncertainty | Implemented | UQ-0 | `probstat.uq` | `src/probstat/uq/taxonomy.py` |
| ATHENS-1.9 | Forward versus inverse problems | Partial | UQ-0 | `probstat.uq` | `docs/foundations/vv-uq-foundations.md` |
| ATHENS-1.10 | Quantity-of-interest definition and selection | Implemented | UQ-0 | `probstat.uq` | `src/probstat/uq/taxonomy.py` |
| ATHENS-2.1 | Random variables | Implemented | UQ-1 | `probstat.probability` | `src/probstat/probability/random_variables.py` |
| ATHENS-2.2 | Probability mass and density functions | Implemented | UQ-1 | `probstat.distributions` | `src/probstat/foundations/protocols.py`; `src/probstat/distributions` |
| ATHENS-2.3 | Cumulative distribution functions | Implemented | UQ-1 | `probstat.distributions` | `src/probstat/foundations/protocols.py` |
| ATHENS-2.4 | Normal, Uniform, Lognormal, and Weibull families | Implemented | UQ-1 | `probstat.distributions` | `src/probstat/distributions/normal.py`; `src/probstat/distributions/uniform.py`; `src/probstat/distributions/lognormal.py`; `src/probstat/distributions/weibull.py` |
| ATHENS-2.5 | Expectation and variance | Implemented | UQ-1 | `probstat.probability` | `src/probstat/probability/random_variables.py` |
| ATHENS-2.6 | Covariance and correlation | Implemented | UQ-1 | `probstat.statistics` | `src/probstat/statistics/descriptive.py` |
| ATHENS-2.7 | Multivariate random variables | Partial | UQ-1 | `probstat.multivariate` | `src/probstat/multivariate/normal.py` |
| ATHENS-2.8 | Multivariate normal distribution | Implemented | UQ-1 | `probstat.multivariate` | `src/probstat/multivariate/normal.py` |
| ATHENS-2.9 | Parameter estimation | Partial | UQ-1 | `probstat.inference` | `src/probstat/inference/estimation.py` |
| ATHENS-2.10 | Maximum-likelihood estimation | Partial | UQ-1 | `probstat.inference` | `src/probstat/inference/estimation.py` |
| ATHENS-2.11 | Kolmogorov-Smirnov and Anderson-Darling goodness-of-fit | Partial | UQ-1 | `probstat.inference.gof` | `src/probstat/inference/goodness_of_fit.py` |
| ATHENS-2.12 | Confidence intervals and sampling concepts | Partial | UQ-1 | `probstat.inference` | `src/probstat/inference/confidence.py`; `src/probstat/distributions` |
| ATHENS-3.1 | Deterministic propagation map Y = g(X) | Implemented | UQ-2 | `probstat.uq` | `src/probstat/uq/taxonomy.py` |
| ATHENS-3.2 | Output mean under input uncertainty | Implemented | UQ-2 | `probstat.propagation` | `src/probstat/propagation/forward.py` |
| ATHENS-3.3 | Output variance under input uncertainty | Implemented | UQ-2 | `probstat.propagation` | `src/probstat/propagation/forward.py` |
| ATHENS-3.4 | Expectation integrals | Implemented | UQ-2 | `probstat.propagation` | `src/probstat/propagation/forward.py` |
| ATHENS-3.5 | Numerical quadrature for uncertainty propagation | Implemented | UQ-2 | `probstat.numerics.quadrature` | `src/probstat/numerics/quadrature.py` |
| ATHENS-3.6 | Basic Monte Carlo propagation | Implemented | UQ-2 | `probstat.monte_carlo` | `src/probstat/monte_carlo/estimation.py`; `src/probstat/propagation/forward.py` |
| ATHENS-3.7 | Monte Carlo estimators | Implemented | UQ-2 | `probstat.monte_carlo` | `src/probstat/monte_carlo/estimation.py` |
| ATHENS-3.8 | Monte Carlo convergence | Implemented | UQ-2 | `probstat.monte_carlo` | `src/probstat/monte_carlo/estimation.py`; `src/probstat/visualization/foundations.py` |
| ATHENS-3.9 | Estimation error and estimator confidence intervals | Implemented | UQ-2 | `probstat.monte_carlo` | `src/probstat/monte_carlo/estimation.py`; `src/probstat/inference/confidence.py` |
| ATHENS-3.10 | Computational cost and sample-budget tradeoffs | Implemented | UQ-2 | `probstat.monte_carlo` | `src/probstat/monte_carlo/estimation.py`; `src/probstat/propagation/forward.py` |
| ATHENS-3.11 | Threshold and exceedance probabilities | Implemented | UQ-2 | `probstat.propagation` | `src/probstat/propagation/forward.py` |
| ATHENS-4.1 | Standard Monte Carlo simulation | Implemented | UQ-3 | `probstat.monte_carlo` | `src/probstat/monte_carlo/estimation.py`; `examples/basic_monte_carlo_lab.py` |
| ATHENS-4.2 | Monte Carlo estimation of pi | Implemented | UQ-3 | `examples.monte_carlo` | `examples/basic_monte_carlo_lab.py` |
| ATHENS-4.3 | Variance-reduction principles | Implemented | UQ-3 | `probstat.monte_carlo.variance_reduction` | `src/probstat/monte_carlo/variance_reduction.py`; `docs/foundations/advanced-monte-carlo.md` |
| ATHENS-4.4 | Estimator efficiency | Implemented | UQ-3 | `probstat.monte_carlo.diagnostics` | `src/probstat/monte_carlo/variance_reduction.py`; `src/probstat/visualization/foundations.py` |
| ATHENS-4.5 | Root-N Monte Carlo convergence rate | Implemented | UQ-3 | `probstat.monte_carlo.diagnostics` | `src/probstat/monte_carlo/estimation.py`; `src/probstat/visualization/foundations.py` |
| ATHENS-4.6 | Strategy comparison and diagnostics | Implemented | UQ-3 | `probstat.monte_carlo.diagnostics` | `src/probstat/monte_carlo/variance_reduction.py`; `examples/advanced_monte_carlo_lab.py` |
| ATHENS-4.7 | Simulation-budget allocation | Implemented | UQ-3 | `probstat.monte_carlo` | `src/probstat/monte_carlo/variance_reduction.py` |
| ATHENS-4.8 | Antithetic variates | Implemented | UQ-3 | `probstat.monte_carlo.variance_reduction` | `src/probstat/monte_carlo/variance_reduction.py` |
| ATHENS-4.9 | Control variates | Implemented | UQ-3 | `probstat.monte_carlo.variance_reduction` | `src/probstat/monte_carlo/variance_reduction.py` |
| ATHENS-4.10 | Conditional Monte Carlo | Implemented | UQ-3 | `probstat.monte_carlo.variance_reduction` | `src/probstat/monte_carlo/variance_reduction.py` |
| ATHENS-4.11 | Stratified sampling | Implemented | UQ-3 | `probstat.monte_carlo.variance_reduction` | `src/probstat/monte_carlo/variance_reduction.py` |
| ATHENS-4.12 | Importance sampling | Implemented | UQ-3 | `probstat.monte_carlo.importance_sampling` | `src/probstat/monte_carlo/variance_reduction.py` |
| ATHENS-5.1 | Reliability as probability of satisfactory performance | Implemented | UQ-4 | `probstat.reliability` | `src/probstat/reliability/analysis.py`; `docs/foundations/reliability-analysis.md` |
| ATHENS-5.2 | Limit-state functions | Implemented | UQ-4 | `probstat.reliability` | `src/probstat/reliability/analysis.py` |
| ATHENS-5.3 | Failure probability | Implemented | UQ-4 | `probstat.reliability` | `src/probstat/reliability/analysis.py` |
| ATHENS-5.4 | Rare-event probability estimation | Implemented | UQ-4 | `probstat.reliability.rare_events` | `src/probstat/reliability/analysis.py`; `examples/reliability_lab.py`; `src/probstat/reliability/simulation.py` |
| ATHENS-5.5 | Direct Monte Carlo reliability estimation | Implemented | UQ-4 | `probstat.reliability` | `src/probstat/reliability/analysis.py` |
| ATHENS-5.6 | Binomial confidence intervals for failure probability | Implemented | UQ-4 | `probstat.reliability` | `src/probstat/reliability/analysis.py`; `src/probstat/inference/confidence.py` |
| ATHENS-5.7 | First-order reliability method (FORM) | Implemented | UQ-4 | `probstat.reliability.form` | `src/probstat/reliability/analysis.py`; `src/probstat/reliability/model_based.py`; `docs/foundations/advanced-reliability.md` |
| ATHENS-5.8 | Reliability index beta | Implemented | UQ-4 | `probstat.reliability.form` | `src/probstat/reliability/analysis.py`; `src/probstat/reliability/model_based.py` |
| ATHENS-5.9 | Constrained optimization for reliability analysis | Implemented | UQ-4 | `probstat.optimization` | `src/probstat/reliability/analysis.py`; `docs/foundations/reliability-analysis.md`; `src/probstat/reliability/model_based.py` |
| ATHENS-5.10 | Most-probable failure point and design point | Implemented | UQ-4 | `probstat.reliability.form` | `src/probstat/reliability/analysis.py`; `src/probstat/reliability/model_based.py` |
| ATHENS-5.11 | Directional sampling and reliability-method comparison | Implemented | UQ-4 | `probstat.reliability.directional` | `src/probstat/reliability/analysis.py`; `examples/reliability_lab.py`; `src/probstat/reliability/simulation.py` |
| ATHENS-6.1 | Local sensitivity analysis | Implemented | UQ-5 | `probstat.sensitivity.local` | `src/probstat/sensitivity/analysis.py` |
| ATHENS-6.2 | Global sensitivity analysis | Implemented | UQ-5 | `probstat.sensitivity.global_` | `src/probstat/sensitivity/analysis.py`; `docs/foundations/sensitivity-analysis.md` |
| ATHENS-6.3 | Variance-based sensitivity decomposition | Implemented | UQ-5 | `probstat.sensitivity.sobol` | `src/probstat/sensitivity/analysis.py` |
| ATHENS-6.4 | Functional ANOVA decomposition | Implemented | UQ-5 | `probstat.sensitivity.sobol` | `docs/foundations/sensitivity-analysis.md`; `docs/equation-registry.md` |
| ATHENS-6.5 | First-order Sobol indices | Implemented | UQ-5 | `probstat.sensitivity.sobol` | `src/probstat/sensitivity/analysis.py` |
| ATHENS-6.6 | Second-order Sobol indices | Implemented | UQ-5 | `probstat.sensitivity.sobol` | `src/probstat/sensitivity/analysis.py` |
| ATHENS-6.7 | Total-effect Sobol indices | Implemented | UQ-5 | `probstat.sensitivity.sobol` | `src/probstat/sensitivity/analysis.py` |
| ATHENS-6.8 | Monte Carlo estimation of sensitivity indices | Implemented | UQ-5 | `probstat.sensitivity.estimation` | `src/probstat/sensitivity/analysis.py`; `tests/test_uq5_sensitivity.py` |
| ATHENS-6.9 | Input-importance ranking | Implemented | UQ-5 | `probstat.sensitivity` | `src/probstat/sensitivity/analysis.py`; `examples/sensitivity_lab.py` |
| ATHENS-6.10 | Dimension reduction using sensitivity results | Implemented | UQ-5 | `probstat.sensitivity` | `src/probstat/sensitivity/analysis.py` |
| ATHENS-6.11 | Uncertainty-reduction strategy | Implemented | UQ-5 | `probstat.sensitivity` | `src/probstat/sensitivity/analysis.py`; `docs/foundations/sensitivity-analysis.md` |
| ATHENS-7.1 | Forward versus inverse formulation | Implemented | UQ-6 | `probstat.inverse` | `docs/foundations/inverse-problems-and-calibration.md` |
| ATHENS-7.2 | Observation models | Implemented | UQ-6 | `probstat.inverse.observation` | `src/probstat/inverse/observation.py` |
| ATHENS-7.3 | Measurement-noise models | Implemented | UQ-6 | `probstat.inverse.observation` | `src/probstat/inverse/observation.py` |
| ATHENS-7.4 | Parameter identification | Implemented | UQ-6 | `probstat.inverse` | `src/probstat/inverse/calibration.py`; `src/probstat/inverse/nonlinear.py` |
| ATHENS-7.5 | Least-squares estimation | Implemented | UQ-6 | `probstat.inference.regression` | `src/probstat/inference/regression.py` |
| ATHENS-7.6 | Linear regression for calibration | Implemented | UQ-6 | `probstat.inference.regression` | `src/probstat/inference/regression.py`; `src/probstat/inverse/calibration.py` |
| ATHENS-7.7 | Bayesian inference | Implemented | UQ-6 | `probstat.inference.bayesian` | `src/probstat/inference/bayesian.py`; `docs/foundations/inverse-problems-and-calibration.md` |
| ATHENS-7.8 | Prior distributions | Implemented | UQ-6 | `probstat.inference.bayesian` | `src/probstat/multivariate/normal.py`; `src/probstat/inference/bayesian.py` |
| ATHENS-7.9 | Likelihood functions | Implemented | UQ-6 | `probstat.inference.likelihood` | `src/probstat/inference/bayesian.py`; `src/probstat/inverse/observation.py` |
| ATHENS-7.10 | Posterior distributions | Implemented | UQ-6 | `probstat.inference.bayesian` | `src/probstat/inference/bayesian.py` |
| ATHENS-7.11 | Gaussian conditioning | Implemented | UQ-6 | `probstat.multivariate.gaussian` | `src/probstat/inference/bayesian.py`; `tests/test_uq6_inverse_calibration.py` |
| ATHENS-7.12 | Posterior sampling | Implemented | UQ-6 | `probstat.inference.sampling` | `src/probstat/inference/bayesian.py`; `src/probstat/inference/mcmc.py` |
| ATHENS-7.13 | Posterior-predictive distributions and checks | Implemented | UQ-6 | `probstat.inference.predictive` | `src/probstat/inference/bayesian.py`; `examples/bayesian_calibration_lab.py` |
| ATHENS-7.14 | Model discrepancy in calibration | Implemented | UQ-6 | `probstat.validation.discrepancy` | `src/probstat/validation/discrepancy.py`; `docs/foundations/inverse-problems-and-calibration.md` |
| ATHENS-7.15 | Calibration versus prediction | Implemented | UQ-6 | `probstat.inverse` | `src/probstat/inverse/calibration.py`; `examples/bayesian_calibration_lab.py` |
| ATHENS-8.1 | Dynamic state-space models | Implemented | UQ-7 | `probstat.filtering.state_space` | `src/probstat/filtering/state_space.py`; `docs/foundations/filtering-and-data-assimilation.md` |
| ATHENS-8.2 | Discrete-time dynamical systems | Implemented | UQ-7 | `probstat.filtering.state_space` | `src/probstat/filtering/state_space.py`; `tests/test_uq7_filtering.py` |
| ATHENS-8.3 | State estimation | Implemented | UQ-7 | `probstat.filtering` | `src/probstat/filtering/kalman.py`; `tests/test_uq7_filtering.py` |
| ATHENS-8.4 | Prediction step | Implemented | UQ-7 | `probstat.filtering.kalman` | `src/probstat/filtering/kalman.py`; `tests/test_uq7_filtering.py` |
| ATHENS-8.5 | Observation model for filtering | Implemented | UQ-7 | `probstat.filtering.state_space` | `src/probstat/filtering/state_space.py`; `src/probstat/filtering/kalman.py` |
| ATHENS-8.6 | Measurement-update step | Implemented | UQ-7 | `probstat.filtering.kalman` | `src/probstat/filtering/kalman.py`; `tests/test_uq7_filtering.py` |
| ATHENS-8.7 | Process noise and measurement noise | Implemented | UQ-7 | `probstat.filtering.state_space` | `src/probstat/filtering/state_space.py`; `docs/foundations/filtering-and-data-assimilation.md`; `tests/test_uq7_filtering.py` |
| ATHENS-8.8 | Covariance propagation | Implemented | UQ-7 | `probstat.filtering.kalman` | `src/probstat/filtering/kalman.py`; `tests/test_uq7_filtering.py` |
| ATHENS-8.9 | Sequential Bayesian updating | Implemented | UQ-7 | `probstat.filtering.kalman` | `src/probstat/filtering/kalman.py`; `src/probstat/filtering/particle.py`; `docs/foundations/filtering-and-data-assimilation.md` |
| ATHENS-8.10 | Data assimilation | Implemented | UQ-7 | `probstat.data_assimilation` | `src/probstat/data_assimilation/enkf.py`; `src/probstat/data_assimilation/variational.py`; `tests/test_uq7_filtering.py` |
| ATHENS-8.11 | Filtering versus smoothing | Implemented | UQ-7 | `probstat.filtering.kalman` | `src/probstat/filtering/kalman.py`; `docs/foundations/filtering-and-data-assimilation.md`; `tests/test_uq7_filtering.py` |
| ATHENS-8.12 | Kalman filter | Implemented | UQ-7 | `probstat.filtering.kalman` | `src/probstat/filtering/kalman.py`; `tests/test_uq7_filtering.py` |
| ATHENS-8.13 | Extended Kalman filter | Implemented | UQ-7 | `probstat.filtering.extended` | `src/probstat/filtering/extended.py`; `tests/test_uq7_filtering.py` |
| ATHENS-8.14 | Ensemble Kalman filter | Implemented | UQ-7 | `probstat.data_assimilation.enkf` | `src/probstat/data_assimilation/enkf.py`; `tests/test_uq7_filtering.py` |
| ATHENS-8.15 | Particle filtering | Implemented | UQ-7 | `probstat.filtering.particle` | `src/probstat/filtering/particle.py`; `tests/test_uq7_filtering.py` |
| ATHENS-9.1 | Statistical fitting laboratory | Partial | LAB-1 | `examples.labs.statistical_fitting` | `examples/statistical_fitting_lab.py`; `src/probstat/visualization/foundations.py` |
| ATHENS-9.2 | Basic Monte Carlo laboratory | Implemented | LAB-2 | `examples.labs.basic_monte_carlo` | `examples/basic_monte_carlo_lab.py`; `src/probstat/visualization/foundations.py` |
| ATHENS-9.3 | Advanced Monte Carlo variance-reduction laboratory | Implemented | LAB-3 | `examples.labs.advanced_monte_carlo` | `examples/advanced_monte_carlo_lab.py`; `docs/foundations/advanced-monte-carlo.md` |
| ATHENS-9.4 | Reliability laboratory with FORM and directional sampling | Implemented | LAB-4 | `examples.labs.reliability` | `examples/reliability_lab.py`; `docs/foundations/reliability-analysis.md` |
| ATHENS-9.5 | Bayesian calibration laboratory | Implemented | LAB-5 | `examples.labs.bayesian_calibration` | `examples/bayesian_calibration_lab.py`; `docs/foundations/inverse-problems-and-calibration.md` |
| ATHENS-9.6 | Uncertainty-visualization laboratory | Partial | LAB-6 | `examples.labs.visualization` | `examples/generate_foundation_figures.py`; `src/probstat/visualization/foundations.py` |
| ATHENS-10.1 | Probability prerequisite | Partial | PRE-1 | `probstat.probability` | `docs/foundations/probability.md` |
| ATHENS-10.2 | Statistics prerequisite | Partial | PRE-1 | `probstat.statistics` | `docs/foundations/descriptive-statistics.md` |
| ATHENS-10.3 | Linear-algebra prerequisite | Partial | PRE-1 | `probstat.linalg` | `src/probstat/multivariate/normal.py` |
| ATHENS-10.4 | Numerical integration, convergence, and error prerequisite | Implemented | PRE-1 | `probstat.numerics` | `src/probstat/numerics/quadrature.py`; `src/probstat/uq/verification.py` |
| ATHENS-10.5 | Optimization prerequisite | Planned | PRE-1 | `probstat.optimization` | — |
| ATHENS-10.6 | Multivariable calculus, Jacobians, and conditional distributions | Planned | PRE-1 | `docs.prerequisites` | — |
| ATHENS-10.7 | R programming bridge | Planned | PRE-1 | `bindings.r` | — |
