# ADR-0111 — Preconditioned Crank-Nicolson, tempered SMC with evidence, and ABC

- Status: Accepted
- Date: 2026-11-19
- Extends: ADR-0017 (nonlinear calibration and MCMC) and ADR-0032 (HMC,
  NUTS, ADVI)

## Context

The samplers of ADR-0017 and ADR-0032 need a log-density and, for HMC and
NUTS, its gradient. Three needs were not covered:

- **High-dimensional priors.** Calibrating a discretized random field or a
  Karhunen-Loeve expansion gives tens to hundreds of standard-normal
  coefficients of which the data inform a few. A random walk must shrink its
  step like `d^-1/2` there, and finite-difference gradients cost `2d` model
  runs per step.
- **Model comparison.** No routine estimated the marginal likelihood
  `p(y)`; the Laplace normalizer is exact only for Gaussian posteriors.
- **Simulator-only models.** Stochastic simulators whose likelihood cannot be
  evaluated had no inference route.

The existing reliability module already uses the pCN move for subset
simulation (EQ-REL-PCN-001), but conditioned on a failure domain, not as a
posterior sampler.

## Decision

1. **Three modules in `probstat.inference` (layer 2):** `pcn`, `smc`, and
   `abc`. They depend only on `foundations`, `linalg`, `statistics`, and
   `inference.mcmc`, so the layer graph is unchanged.
2. **`preconditioned_crank_nicolson` returns an `MCMCResult`.** The target is
   `L(u) N(u; 0, I)` and the user passes the log-likelihood only
   ([EQ-MCMC-PCN-001]). `beta` is adapted during warm-up only, by a
   Robbins-Monro recursion on its log-odds with gain `k^-0.6` driven by the
   acceptance probability ([EQ-MCMC-PCN-002]), and frozen afterwards, as in
   ADR-0017, so the kept chain is time-homogeneous and every diagnostic
   applies. `proposal_covariance` holds `beta^2 I`. The default target
   acceptance is 0.25: Cotter et al. (2013) tune to about 0.234 or 25 % in
   their experiments and report a problem where about 50 % converged faster,
   and no optimal-scaling theorem fixes the value for pCN, so it is exposed
   as `target_acceptance`.
3. **`sequential_monte_carlo` uses adaptive likelihood tempering.** The next
   exponent is found by bisection on the ESS of the incremental weights
   (target `ess_fraction` times the particles with finite likelihood;
   [EQ-SMC-TEMPER-002]); resampling happens every stage, multinomial by
   default; mutation is random-walk Metropolis with `(2.38^2 / d)` times the
   weighted particle covariance plus a declared `regularization`, or pCN with
   a between-stage log-odds adaptation of `beta` (gain 4) for an `N(0, I)`
   prior. `log Z_hat` is the sum of log mean incremental weights
   ([EQ-SMC-EVID-001]).
4. **The evidence error is the Lee-Whiteley estimator** ([EQ-SMC-EVID-002]),
   computed from the Eve indices of one run, reported as the relative
   variance `v` of `Z_hat` and as `sqrt(v)`, the delta-method standard error
   of `log Z_hat`. It is reported only for multinomial resampling (the
   theorem's setting); a negative `v` gives a NaN standard error. The
   docstring states that adaptive temperatures and kernels are outside the
   theorem and that the estimate ignores the bias of a poorly mixing kernel.
5. **`abc_rejection` and `abc_smc` return an `ABCResult`** (parameters,
   weights, distances, tolerances, simulations per generation, acceptance
   rates, stopping reason). Rejection has a fixed-tolerance mode with a
   simulation budget that raises when exhausted, and a quantile mode with a
   fixed number of simulations. `abc_smc` is ABC-PMC (Beaumont et al. 2009)
   with the `2 Sigma` Gaussian kernel and importance weights
   ([EQ-ABC-PMC-001]); its tolerance schedule is the `quantile` of the
   previous generation's distances, generation 1 is prior simulation at
   `initial_tolerance` (infinite by default), and a generation that would
   exceed `max_simulations` is abandoned and the last complete one returned
   with `stopped_by = "budget"`.
6. **Calibration wiring.** `CalibrationProblem.sample_posterior` gains
   `method="pcn"` and `method="smc"` and the keywords `particles` and
   `mutation_steps`; the default and the HMC/NUTS paths keep their random
   streams and results. Both new methods need an `N(0, I)` prior in every
   sampled coordinate. The unknown noise scales, sampled as `log sigma` with
   prior `p_sigma(e^eta) e^eta`, are not standard normal, so instead of
   rejecting unknown-noise problems the two methods standardize them through
   the noise priors' quantile maps, `sigma = F^-1(Phi(w))`
   ([EQ-CAL-STD-002], `standard_log_likelihood`). The change of variables is
   exact, so the posterior is unchanged and `p(y)` is the integral of the
   standardized likelihood against `N(0, I)`. Draws are mapped back to
   `(u, log sigma)` and `(theta, sigma)`. `CalibrationPosterior` gains a
   trailing optional field `smc`; for SMC the particles form one pseudo chain
   whose R-hat and ESS are documented as not applicable.
   `CalibrationProblem.log_evidence` returns the `SMCResult` of a run with
   random-walk mutation (better suited to concentrated, anisotropic
   posteriors than isotropic pCN).

## Consequences

- High-dimensional standard-normal priors have a sampler whose acceptance
  does not collapse with refinement (0.204, 0.187, 0.186 at `d = 10, 50, 200`
  on the field test, against 0.245, 0.120, 0.060 for a fixed random walk).
- `log_evidence` reproduces the exact marginal likelihood of the
  linear-Gaussian calibration (`-8.3273 +- 0.0684` against `-8.2812`, 1000
  particles, seed 5) and of an unknown-noise variant checked by quadrature
  in the tests.
- pCN mixes slower than adaptive Metropolis when the data inform every
  direction (bulk ESS 243 against 750 on the line problem); the guide says
  so.
- ABC-PMC costs the same as rejection on the one-dimensional normal-mean
  example; it is offered for problems where the prior is much wider than the
  posterior in several directions.
- `DiscrepancyCalibration` keeps adaptive Metropolis, HMC, and NUTS only. Its
  hyperparameters (`log psi`, discrepancy scales) would need the same
  quantile standardization; this is listed as a limitation in the guide.

## Rejected alternatives

- **Raising `ValueError` for unknown-noise problems under pCN.** Simple, but
  it would exclude the common calibration with an unknown noise level, and
  the quantile standardization is exact and cheap.
- **Perpetual adaptation of `beta`.** Valid under diminishing-adaptation
  conditions, but it would break the time-homogeneity that the diagnostics
  assume (ADR-0017).
- **A naive evidence error that treats each stage's weights as independent
  draws.** It ignores the dependence created by resampling and mutation and
  has no consistency result; the genealogy-based estimator has one.
- **Resampling only when the ESS falls below a threshold.** With
  ESS-targeted tempering the ESS after reweighting equals the threshold by
  construction, so the two rules coincide; resampling every stage also
  matches the setting of the Lee-Whiteley estimator.
- **Returning SMC particles as a new posterior type for calibration.** It
  would duplicate `CalibrationPosterior`; the pseudo chain keeps `draws`,
  `mean`, and `predict` working, and the full result stays in `smc`.

## Verification

`tests/test_inference_pcn_smc_abc.py` (22 tests, about 13 s) checks pCN and
SMC against a conjugate normal model with exact posterior and evidence, the
dimension robustness of pCN against a collapsing random walk, the one-stage
exact evidence of a truncated likelihood and its Lee-Whiteley error
`1/sqrt(N)`, the ESS bisection, ABC rejection and ABC-PMC against the exact
uniform-kernel ABC posterior computed by quadrature (and its convergence to
the posterior), the stopping rules, reproducibility, input validation, and
the calibration wiring against `calibrate_linear_gaussian` and the exact
marginal likelihood.

## References

[REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-HAIRER-STUART-VOLLMER-2014],
[REF-ANDRIEU-THOMS-2008], [REF-DELMORAL-DOUCET-JASRA-2006],
[REF-CHOPIN-2002], [REF-JASRA-STEPHENS-DOUCET-TSAGARIS-2011],
[REF-LEE-WHITELEY-2018], [REF-PRITCHARD-ETAL-1999],
[REF-BEAUMONT-CORNUET-MARIN-ROBERT-2009].
