# ADR-0122 — Importance-sampling diagnostics and further samplers

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0017 (nonlinear calibration and MCMC), ADR-0111 (pCN,
  tempered SMC, ABC)

## Context

`probstat.inference` had random-walk and adaptive Metropolis, HMC and NUTS,
pCN, tempered SMC, ABC, and Gaussian ADVI, with the R-hat and ESS
diagnostics of Vehtari et al. (2021). Four gaps remained:

- Importance sampling had no diagnostic. A self-normalized estimate with a
  heavy-tailed ratio distribution reports a small standard error and is
  wrong; nothing in the library said so. Leave-one-out model comparison
  from posterior draws was missing for the same reason.
- No convergence diagnostic suited many short chains.
- No sampler worked without a tuned proposal scale (slice sampling, the
  affine-invariant ensemble), none used a single gradient per step (MALA),
  and none crossed between separated modes while keeping an ordinary Markov
  chain at the target (parallel tempering).
- No deterministic particle approximation (SVGD).

## Decision

1. **Six new modules in `probstat.inference`** (layer 2; imports only
   `foundations`, `linalg`, `statistics`, and the sibling modules
   `inference.mcmc` and `inference.hamiltonian`): `psis`, `nested_rhat`,
   `particle_vi`, `slice_sampling`, `langevin_mcmc`, `ensemble_samplers`.
   `mcmc.py` is not edited.
2. **PSIS** ([EQ-INF-PSIS-001] to [EQ-INF-PSIS-003]). `pareto_smooth` takes
   log ratios, works relative to the largest one, and returns normalized
   smoothed log-weights with `k-hat`, the threshold
   `min(1 - 1/log10(S), 0.7)`, and `ESS = 1 / sum w~^2`.
   - Tail size `M = ceil(min(S/5, 3 sqrt(S)))`. The ceiling follows the
     reference implementation of the source (the `loo` R package,
     `n_pareto`); the task statement gave the rule without rounding.
   - The generalized Pareto fit is implemented locally
     (`fit_generalized_pareto`), because `probstat.risk.fit_gpd` is in
     layer 6. It is the estimator PSIS itself uses: the Zhang-Stephens
     empirical-Bayes profile estimator with the grid, quartile, and prior
     constants of the reference implementation (`posterior::gpdfit`, read
     from its source on 2026-11-21) and the weak prior
     `k <- (N k + 5) / (N + 10)`. Zhang and Stephens are named in prose
     only; their paper was not verified bibliographically and has no
     reference key.
   - Fewer than five tail draws, or a degenerate tail, leave the ratios
     unsmoothed with `k-hat = inf` (flagged). Exactly equal ratios return
     uniform weights with `k-hat = -inf`; this last convention is the
     library's.
   - The standard error `sqrt(sum w~^2 (h - mu)^2)` assumes independent
     draws. The relative-efficiency correction of the reference
     implementation for MCMC draws is not implemented.
3. **PSIS-LOO** ([EQ-INF-PSIS-004]). `psis_loo` takes the `S x n` pointwise
   log-likelihood and returns totals, pointwise values, `k-hat` per
   observation, and the flagged indices.
4. **Nested R-hat** ([EQ-INF-NRHAT-001]). `nested_rhat` takes
   `superchains[k][m][n]` and returns the statistic with `B` and `W`. The
   definition of `W` (average over superchains of the between-chain
   variance plus the mean within-chain variance, each set to zero when
   undefined) was confirmed against the comments of the reference
   implementation `posterior::rhat_nested`, which cite Equations 6 to 8 of
   the paper. No threshold is built in.
5. **SVGD** ([EQ-INF-SVGD-001], [EQ-INF-SVGD-002]). Fixed step; RBF kernel
   with the median heuristic `h = med^2 / log n` recomputed every
   iteration, or a fixed bandwidth; scores from a callable or from the
   gradient oracle shared with HMC. The AdaGrad step of the source is not
   implemented (its exact form could not be checked against the paper).
   The result has `converged` and the final update norm, and its
   `covariance` uses the divisor `n`.
6. **Slice samplers** ([EQ-INF-SLICE-001], [EQ-INF-SLICE-002]) return
   `MCMCResult`. `acceptance_rate` is one; `proposal_covariance` holds
   `diag(w^2)` or the prior covariance. Unlimited stepping out raises after
   100000 extensions (improper target).
7. **MALA** ([EQ-INF-MALA-001]) with a fixed step returns `MCMCResult`.
   `mala_log_acceptance_ratio` is public so that detailed balance can be
   checked. No optimal acceptance rate is stated.
8. **Stretch move** ([EQ-INF-STRETCH-001]). Two half-ensembles updated
   alternately, partners drawn from the other half. The result is an
   `EnsembleResult`, not an `MCMCResult`: walkers are dependent, so the
   multi-chain convergence check is not run and the reported ESS is called
   a heuristic.
9. **Parallel tempering** ([EQ-INF-PT-001], [EQ-INF-PT-002]). Likelihood
   and prior are separate arguments; the ladder is required and must end at
   one. Local moves default to isotropic random-walk Metropolis with one
   scale per temperature; `kernels` accepts user kernels. Swaps follow the
   deterministic even-odd order. The stepping-stone evidence is reported
   only when `beta_0 = 0`, without a standard error. The identity was
   derived (`Z(b')/Z(b) = E_b[L^(b' - b)]`) and checked against two exact
   evidences.

## Consequences

- Proposal `N(0, 1)`, target `N(0, s^2)`, 20000 draws, seed 1: `s = 1.2`
  gives `k-hat = 0.27` (tail shape `0.31`) and `E[x^2] = 1.437 +- 0.025`
  (exact `1.44`); `s = 3` gives `k-hat = 0.79 > 0.7` and an estimate of
  `4.63 +- 0.45` for an exact `9`. Over 60 replicates with 4000 draws and
  `s = 1.2` the mean estimate is `1.431` (standard error `0.008`).
  With `s = 1.5` (`k = 0.56`) and 20000 draws the mean over 60 replicates
  is `2.193` against `2.25`: PSIS trades variance for a bias that shrinks
  slowly as `k` approaches the threshold.
- The local generalized Pareto fit recovers `k = -0.3, 0, 0.3, 0.7, 1.0`
  as `-0.297, 0.001, 0.296, 0.700, 1.002` (means of 40 samples of 2000).
- PSIS-LOO on the conjugate normal model (20 observations, 4000 draws):
  `elpd_loo = -26.620` against the exact `-26.634`; largest pointwise
  error `0.004`.
- Stretch move: the affine image of a trajectory is reproduced to `1e-9`
  over 40 sweeps and to `1e-6` over 300; after about 1000 sweeps rounding
  has changed an accept decision and the trajectories differ. Equivariance
  is exact in distribution, not bitwise over long runs.
- Parallel tempering on `0.3 N(-5, 0.25) + 0.7 N(5, 0.25)` with a
  `N(0, 100)` prior, seven temperatures, 20000 draws: right-mode fraction
  `0.71`, 2643 round trips, `log Z = -3.3470` (exact `-3.3475`); the
  conjugate normal evidence over six seeds deviates by at most `0.018`.
  Random-walk Metropolis from the left mode never reaches the right one.
- SVGD on `N(1.5, 4)` in one dimension: variances `2.82`, `3.26`, `3.57`
  for 5, 10, 25 particles. Ten particles on a standard normal (step 0.3,
  1000 iterations): per-coordinate variance `0.81`, `0.40`, `0.27` in
  dimension 1, 4, 8.
- `inference/__init__.py` carries the new imports as one block between
  `# isort: off` and `# isort: on`, so that parallel work packages can add
  their own blocks without merge conflicts; the integrator may sort it.

## Rejected alternatives

- **Importing `probstat.risk.fit_gpd`.** Forbidden by the layer map, and a
  maximum-likelihood fit is not the estimator the `k-hat` thresholds were
  calibrated with.
- **Returning an `MCMCResult` from the stretch move.** It would run R-hat
  and ESS checks that assume independent chains.
- **Adapting the MALA step or quoting an optimal acceptance rate.** The
  figure usually quoted could not be verified against a source here.
- **A default temperature ladder.** A geometric ladder is a common choice
  but its quality depends on the problem; the rejection rates in the result
  show where a ladder is too coarse.
- **Thermodynamic integration.** The stepping-stone identity is exact for
  any ladder, whereas a quadrature of `E_beta[l]` over `beta` adds a
  discretization error.
- **AdaGrad for SVGD.** Left out until its form is checked against the
  source.

## Verification

`tests/test_psis.py` (22 tests, about 2 s): generalized Pareto recovery,
shrinkage formula, quantile inversion, scale equivariance, an independent
recomputation of the smoothing, the Gaussian tail-shape oracle, convergence
of `E[x^2]`, the flag for `k > 0.7`, and PSIS-LOO against closed-form and
quadrature leave-one-out densities, including a flagged outlier.
`tests/test_samplers_extended.py` (41 tests, about 11 s): nested R-hat by
hand, the `1/N` identity with the classical statistic, and the stationary
ratio `1/(M (N + 1))`; the SVGD direction against a brute-force double sum,
the one-particle case, Gaussian moments, and the variance shortfall; slice
samplers against quadrature and conjugate moments; MALA detailed balance on
a grid and moments against quadrature; stretch-move equivariance and
moments; parallel-tempering mode weights, both evidences, and the kernel
interface; and input validation for every function.

## References

[REF-VEHTARI-ETAL-2024], [REF-VEHTARI-GELMAN-GABRY-2017],
[REF-MARGOSSIAN-ETAL-2024], [REF-LIU-WANG-2016], [REF-NEAL-2003],
[REF-MURRAY-ADAMS-MACKAY-2010], [REF-ROBERTS-TWEEDIE-1996],
[REF-GOODMAN-WEARE-2010], [REF-HUKUSHIMA-NEMOTO-1996],
[REF-SYED-ETAL-2022].
