# ADR-0079: Evidence audit of the partially completed roadmap items

- Status: Accepted
- Date: 2026-11-11

## Context

`docs/roadmap.md` marks nine items `[~]` (in progress) whose "remains" clauses
were written when the items were opened. Several of the listed gaps were
closed later by other work, while others were still open. A status
derived from prose drifts; this audit derives each status from code and
tests only.

Evidence rule. A capability counts as present only when (a) a public name
implements it under `src/probstat`, and (b) a test file exercises it against
an independent oracle (closed form, exact enumeration, published value, or a
seeded Monte Carlo check with a tolerance derived from its standard error).
A capability exercised only by an example lab is reported as such.

Scope rule. A missing capability is implemented here when it is a focused
addition (about 400 lines including tests) and respects the layer map of
`tests/test_architecture.py`; larger items are reported as remaining.

## Decision

### Audit table

| Roadmap item | Evidence found (module / function; test file) | Implemented here | Genuinely remains |
|---|---|---|---|
| Phase 2: stream splitting; family-specific goodness-of-fit calibration | `foundations.random_streams`: `RandomStream.child`, `spawn`, `derive_seed`, `splitmix64` (`tests/test_core_modeling.py`: reproducible and distinct child seeds, SplitMix64 reference value); `inference.bootstrap_goodness_of_fit` refits every replicate for every family (`tests/test_generic_fitting_model_selection.py`) | `inference.normality_tests`, `lilliefors_p_value`, `cramer_von_mises_statistic`: closed-form fitted-null p-values for the normal family (Lilliefors/Dallal-Wilkinson KS, Stephens-modified Cramer-von Mises and Anderson-Darling); `tests/test_normality_edf_calibration.py` | Tabulated (closed-form) fitted-null calibration for exponential, Weibull, and extreme-value families; these are served by the parametric bootstrap, which is exact in law for location-scale families |
| Phase 5: amortized inference | MC and IS (`monte_carlo.estimate_importance_sampling`; `tests/test_uq3_variance_reduction.py`), adaptive MCMC and diagnostics (`inference.adaptive_metropolis`, `split_rhat`; `tests/test_uq6b_nonlinear_calibration.py`; `effective_sample_size`; `tests/test_mcmc_diagnostics_autodiff.py`), Laplace (`inference.laplace_approximation`; `tests/test_mcmc_diagnostics_autodiff.py`), particle filters (`filtering.bootstrap_particle_filter`; `tests/test_uq7_filtering.py`), HMC/NUTS and ADVI (`inference.hamiltonian_monte_carlo`, `no_u_turn_sampler`, `advi`; `tests/test_hmc_nuts_advi.py`) | `inference.fit_amortized_gaussian_posterior`, `AmortizedGaussianPosterior`: neural posterior estimation with `numerics.MLP` and `optimization.adam_minimize`; `tests/test_amortized_inference.py` | Non-Gaussian amortized posteriors (mixture-density or flow-based conditional densities) and sequential (multi-round) proposals |
| Phase 6: probabilistic programming, static flat-vector sites only | `inference.ppl`: `trace`, `ModelIR`, bijections, `sample`, `fit_advi`, `laplace`, predictive checks (`tests/test_probabilistic_programming.py`); `_program.bijection_for` rejects discrete latent sites and `ModelSpecificationError` rejects site structure that changes between executions | Nothing (not a focused addition) | Discrete latent sites (enumeration or marginalization), stochastic control flow with dynamic site structure, structured supports (simplex, covariance) as latent sites |
| Phase 6: conformal prediction; robustness and sequential decision systems | Split, normalized, CQR, jackknife+ (`conformal`; `tests/test_conformal_prediction.py`); shift tests and scoring rules (`validation.shift`, `validation.scoring`; `tests/test_forecast_calibration.py`) | `conformal.AdaptiveConformalPredictor`, `adaptive_conformal_intervals` (adaptive conformal inference under distribution shift; `tests/test_adaptive_conformal.py`); `risk.bernoulli_bandit`, `BanditRun` (Beta-Bernoulli Thompson sampling and the greedy baseline; `tests/test_thompson_sampling.py`) | Contextual bandits and Markov decision processes; weighted conformal prediction under known covariate shift |
| UQ-1: Normal/Lognormal MLE, generic KS/AD, z and Wilson intervals, fitted-CDF visualization, seeded lab | `inference.fit_normal_mle`, `fit_lognormal_mle` (`tests/test_uq1_modeling_and_inference.py`, `tests/test_generic_fitting_model_selection.py`); `inference.kolmogorov_smirnov_test`, `anderson_darling_test` (`tests/test_uq1_modeling_and_inference.py`); `inference.normal_mean_confidence_interval`, `wilson_score_interval` (`tests/test_uq1_modeling_and_inference.py`); `visualization.plot_fitted_cdf` (exercised by `examples/statistical_fitting_lab.py` and `examples/model_selection_lab.py`, which CI runs; no unit test); seeded lab `examples/statistical_fitting_lab.py` | Nothing needed | Nothing; a unit test for `plot_fitted_cdf` is advisable but not a capability gap |
| UQ-1: Nataf; multivariate protocols, Weibull estimation, exact/bootstrapped fitted-null calibration, Student-t intervals, stream splitting, real-data case studies | Nataf and correlated inputs (`modeling.JointInput`; `tests/test_core_modeling.py`, `tests/test_copulas.py`); Weibull MLE (`inference.fit_weibull_mle`, `WEIBULL_FAMILY`; `tests/test_generic_fitting_model_selection.py`, `tests/test_survival_analysis.py`); bootstrapped fitted-null calibration (`inference.bootstrap_goodness_of_fit`); Student-t intervals (`inference.student_t_mean_confidence_interval`; `tests/test_classical_resampling_inference.py`); stream splitting (`foundations.RandomStream`) | `foundations.HasDimension`, `MultivariateLogDensity`, `MultivariateLogMass`, `MultivariateMoments`, `MultivariateSampler` (`tests/test_multivariate_protocols.py`); exact (tabulated) fitted-null calibration for the normal family (`inference.normality_tests`) | Real-data fitting case studies: every fitting lab uses seeded synthetic data |
| UQ-2: model-form uncertainty beyond the discrepancy term | Unbounded-domain quadrature (`numerics.exp_sinh`, `sinh_sinh`, `tanh_sinh_interval`; `tests/test_adaptive_quadrature.py`), streaming accumulation (`statistics.RunningMoments`; `tests/test_streaming_statistics.py`), numerical uncertainty into validation (`validation.predict_with_uncertainty_components`; `tests/test_uq0_prediction_workflow.py`); model form there is only an additive Gaussian discrepancy | `validation.stacking_weights`, `pseudo_bma_weights`, `model_averaged_prediction`, `ModelWeights`, `ModelAveragedPrediction`: model-form uncertainty across competing models with the between-model variance component (`tests/test_model_averaging.py`) | Nothing at the level of this item; hierarchical (input-dependent) stacking is a possible extension |
| UQ-5: Shapley, DGSM, given-data; replicated stochastic simulators | `sensitivity.shapley_effects` (`tests/test_mlmc_ce_line_shapley.py`), `dgsm`, `given_data_first_order` (`tests/test_sensitivity_estimators.py`) | `sensitivity.stochastic_sobol_indices`, `StochasticSobolResult`: Sobol indices of the mean response, output-variance shares, and the intrinsic-noise index from replicated runs (`tests/test_stochastic_sobol.py`) | Indices of the full output law of a stochastic simulator (for example indices of the conditional variance or quantile-oriented indices) |

### Added capabilities and their verification

1. **Fitted-null normality calibration** (`inference/normality.py`). The
   null laws of the KS, CvM, and AD statistics with estimated mean and
   variance are parameter-free for the normal family, so published
   approximations apply: Dallal and Wilkinson (1986) for the Lilliefors
   statistic and Stephens (1986, Table 4.9) for the modified CvM and AD
   statistics; constants were checked against the reference implementation
   of Gross and Ligges (2015). Oracle: a seeded simulation of the fitted
   statistics for standard-normal samples. With 10,000 simulations per
   size, the rejection rates at `alpha = 0.01, 0.05, 0.10` for `n = 10, 20,
   50, 150` lay within 0.007 of nominal for all three tests; the test suite
   checks `n = 20` with 4,000 simulations and compares individual p-values
   with the simulated tail. Equations [EQ-INF-GOF-CVM-001] and
   [EQ-INF-GOF-NORM-001] to [EQ-INF-GOF-NORM-003].
2. **Adaptive conformal inference** (`conformal/adaptive.py`). The update
   `alpha_(t+1) = alpha_t + gamma (alpha - err_t)` with infinite thresholds
   at `alpha_t <= 0` and empty sets at `alpha_t >= 1` gives the
   deterministic bound of Gibbs and Candès (2021, Proposition 4.1), which
   the tests check on adversarial sequences for four step sizes, together
   with `alpha_t` in `[-gamma, 1 + gamma]` (Lemma 4.1). With a fourfold
   noise increase halfway through 4,000 steps, the post-shift miss rate was
   0.099 for ACI (`gamma = 0.02`, window 300) and 0.274 for fixed-level
   online conformal (target 0.10). `gamma = 0` reproduces the split
   conformal quantile exactly. Equations [EQ-CONF-ACI-001] to
   [EQ-CONF-ACI-003].
3. **Thompson sampling for the Beta-Bernoulli bandit** (`risk/bandits.py`,
   layer 6 next to decision analysis). Oracle: the first-period choice
   frequency over 4,000 seeded runs matches `P(Beta(3,1) > Beta(1,2))`
   computed by quadrature of a binomial-tail CDF; posterior pseudo-counts
   are recomputed from the recorded history; mean regret over ten runs is
   below a quarter of that of uniform play, and less regret accrues in the
   second half of the horizon than in the first; the greedy rule (Russo et al. 2018,
   Algorithm 3.1) incurs more than twice the regret. Equations
   [EQ-RISK-BANDIT-001] to [EQ-RISK-BANDIT-003].
4. **Sobol indices for stochastic simulators** (`sensitivity/stochastic.py`).
   The law of total variance and the indices of Iooss, Ribatet, and Marrel
   (2009, Equations 15 and 19-21) are estimated from a pick-freeze design
   with `R >= 2` independent runs per point; within-point variances remove
   the noise bias of the total-effect estimator and of the variance of the
   replicate means. Oracle: a heteroscedastic model with closed-form
   `Var(m) = 8/3`, noise variance `7/15`, and known first and total
   indices; across five seeded studies (`N = 1500`, `R = 2`) every estimate
   lay within four standard errors of its closed form, and the total effect
   of an inert input stayed below a quarter of the bias an uncorrected
   estimator would carry. With zero noise the numerators equal those of
   `estimate_sobol_indices` to ten decimals. Equations [EQ-SENS-STOCH-001]
   to [EQ-SENS-STOCH-003].
5. **Stacking and pseudo-BMA** (`validation/model_averaging.py`). Weights
   maximize the mean log score of the mixture of predictive distributions
   (Yao et al. 2018, Equation 2.2) by the EM update for mixing proportions,
   stopped by the concavity bound `max_k g_k - 1` on the log-score
   shortfall; pseudo-BMA and pseudo-BMA+ follow Equations 2.5 and 2.6.
   Oracles: the closed-form two-point optimum `w = 5/6`, the
   Karush-Kuhn-Tucker conditions on a random problem, dominance, and the law
   of total variance for the averaged prediction. On data from an equal
   two-component mixture, stacking weights were 0.528 and 0.472 while
   pseudo-BMA put weight 1 on one model. The module lives in `validation`
   because model-form uncertainty is assessed against validation data, next
   to `predict_with_uncertainty_components`. Equations [EQ-VAL-STACK-001] to
   [EQ-VAL-STACK-003].
6. **Amortized Gaussian posterior** (`inference/amortized.py`). A diagonal
   Gaussian conditional density whose means and log standard deviations are
   MLP outputs is fitted by maximum likelihood to prior-predictive
   simulations; by Papamakarios and Murray (2016, Proposition 1) the
   maximizer is the posterior when the family contains it. Oracle: the
   conjugate normal model, posterior `N(4s/5, 1/5)` at every observation
   `s`, recovered at five observations without retraining; 90% credible
   intervals cover fresh prior-predictive parameters at the nominal rate; a
   two-parameter model with different noise levels recovers both
   closed-form posteriors. Training uses reverse-mode gradients and Adam;
   the test file runs in about 20 seconds. Equation [EQ-INF-NPE-001].
7. **Multivariate protocols** (`foundations/protocols.py`).
   `MultivariateLogDensity`, `MultivariateLogMass`, `MultivariateMoments`,
   and `MultivariateSampler` (all extending `HasDimension`) describe the
   vector-valued families structurally, like the univariate protocols. The
   multivariate normal, Student-t, Dirichlet, multinomial, and
   Dirichlet-multinomial laws conform without changes; a protocol-generic
   Monte Carlo check of sample means against declared means passes for all
   five.

### Proposed roadmap wording

The wording below replaces the corresponding lines of `docs/roadmap.md`.

Phase 2:

```text
- [x] Stream splitting via SplitMix64 `RandomStream`; parametric-bootstrap
  fitted-null calibration of KS/AD for every fitting family and closed-form
  Lilliefors, Cramer-von Mises, and Anderson-Darling calibration for the
  normal family (ADR-0079).
```

Phase 5:

```text
- [x] Monte Carlo, importance sampling, adaptive MCMC with diagnostics, Laplace,
  particle filters, HMC/NUTS, Gaussian ADVI (ADR-0032); amortized Gaussian
  neural posterior estimation (ADR-0079). Non-Gaussian amortized posteriors
  await the conditional density models of Phase 6.
```

Phase 6, probabilistic programming (status unchanged):

```text
- [~] Probabilistic programming intermediate representation and inference API
  (ADR-0053) with reverse-mode gradients (ADR-0069); static flat-vector
  continuous sites only. Discrete latent sites, dynamic site structure, and
  structured (simplex, covariance) latent supports remain.
```

Phase 6, conformal prediction:

```text
- [x] Conformal prediction (ADR-0033), distribution-shift tests and forecast
  calibration with proper scoring rules (ADR-0042); adaptive conformal
  inference under distribution shift and a Beta-Bernoulli Thompson-sampling
  bandit for sequential decisions (ADR-0079). Contextual bandits and Markov
  decision processes are not implemented.
```

UQ-1:

```text
- [x] Normal/Lognormal maximum likelihood, generic KS/AD statistics, z and
  Wilson confidence intervals, fitted-CDF visualization, and seeded lab.
- [~] Nataf transformations, correlated joint inputs, multivariate structural
  protocols (ADR-0079), Weibull estimation, bootstrapped and normal-family
  closed-form fitted-null calibration, Student-t intervals, and sampling
  stream splitting delivered; real-data fitting case studies remain.
```

UQ-2:

```text
- [x] Unbounded-domain quadrature transformations and adaptive error
  estimates (ADR-0050), streaming accumulation (ADR-0055), numerical
  uncertainty carried from solution verification into validation (ADR-0057);
  model-form uncertainty across competing models by stacking and pseudo-BMA
  with a between-model variance component (ADR-0079).
```

UQ-5:

```text
- [x] Shapley effects (ADR-0029), DGSM and given-data indices (ADR-0065);
  Sobol indices of replicated stochastic simulators with an intrinsic-noise
  index (ADR-0079).
```

## Consequences

- New public names: `inference.EDFNormalityTest`, `NormalityTests`,
  `normality_tests`, `lilliefors_p_value`, `cramer_von_mises_statistic`,
  `AmortizedGaussianPosterior`, `fit_amortized_gaussian_posterior`;
  `conformal.AdaptiveConformalPredictor`, `AdaptiveConformalResult`,
  `adaptive_conformal_intervals`; `risk.BanditRun`, `bernoulli_bandit`;
  `sensitivity.StochasticSobolResult`, `stochastic_sobol_indices`;
  `validation.ModelWeights`, `ModelAveragedPrediction`, `stacking_weights`,
  `pseudo_bma_weights`, `model_averaged_prediction`;
  `foundations.HasDimension`, `MultivariateLogDensity`,
  `MultivariateLogMass`, `MultivariateMoments`, `MultivariateSampler`.
  `docs/api-reference.md` must be regenerated.
- No new cross-layer imports: `inference.amortized` uses `numerics` and
  `optimization` (layers 0 and 1); `risk.bandits` uses `foundations` only;
  every other module imports within its own package or from `foundations`.
- The closed-form normality p-values are approximations; outside the normal
  family, or when another estimator than the sample mean and `n - 1`
  standard deviation is used, `bootstrap_goodness_of_fit` is the calibrated
  route.
- The adaptive-conformal guarantee is a long-run frequency statement; it
  says nothing about coverage at a particular time, which the
  `local_coverage` diagnostic exposes.
- The amortized posterior is only as good as its Gaussian family;
  calibration on fresh simulations should precede use on a model whose
  posterior may be skewed or multimodal.

## Verification evidence

- `tests/test_normality_edf_calibration.py`, `tests/test_adaptive_conformal.py`,
  `tests/test_thompson_sampling.py`, `tests/test_stochastic_sobol.py`,
  `tests/test_model_averaging.py`, `tests/test_amortized_inference.py`,
  `tests/test_multivariate_protocols.py` pass together with
  `tests/test_documentation_contract.py`, `tests/test_architecture.py`, and
  the existing test files of the touched packages.
- `python -m ruff check .` and `python -m mypy src/probstat` (strict) report
  no findings.
