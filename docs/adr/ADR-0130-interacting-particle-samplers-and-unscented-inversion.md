# ADR-0130 — Interacting particle samplers and unscented Kalman inversion

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0115 (ensemble Kalman inversion and ES-MDA)

## Context

ADR-0115 gave `inverse` two derivative-free ensemble methods for a
`CalibrationProblem` with known Gaussian noise: ensemble Kalman inversion
(EKI), an optimizer whose ensemble collapses and whose spread is not a
posterior, and ES-MDA, a short sequence of Kalman updates that is exact for
linear-Gaussian problems only as the ensemble size grows. Three later
methods address what these leave open:

- a **sampler** that is exact for linear-Gaussian problems at a *small*
  ensemble size (affine-invariant interacting Langevin dynamics, the
  corrected ensemble Kalman sampler);
- a **deterministic Gaussian iteration** without sampling error and with a
  cost of `2 d + 1` forward runs per step (unscented Kalman inversion);
- a method that needs only **values of the misfit** (consensus-based
  sampling and optimization).

The formulas supplied with the task were written from memory. Every
exactness statement below was therefore rederived and checked against the
conjugate posterior of the straight-line problem; what could not be derived
is not claimed.

## Decision

1. **Three modules** in `inverse` (layer 4): `interacting_langevin.py`
   (`interacting_langevin_sampler`, `InteractingLangevinResult`),
   `unscented_inversion.py` (`unscented_kalman_inversion`,
   `UnscentedInversionResult`), `consensus_sampling.py`
   (`consensus_based_sampling`, `ConsensusSamplingResult`). They import
   `foundations`, `linalg`, `inverse.nonlinear`, and the private ensemble
   statistics and `Gamma` validation of `inverse.ensemble_inversion`. The
   unscented transform is written locally because `filtering` is layer 6.
2. **Conventions of ADR-0115 are kept**: coordinates `u` with prior
   `N(0, I)` mapped by `model.inputs.from_standard_normal`; whitening by the
   diagonal `Gamma`; known-scale Gaussian noise only, with the same error
   messages; the prior ensemble from the child stream `"ensemble"` (the same
   draw as EKI for the same seed) and the noise of step `n` from
   `("noise", n)`; `forward_evaluations` counted in evaluations of the whole
   observation operator and `model_calls` in model runs; histories of mean,
   spread and whitened misfit in both coordinates.
3. **Target.** All three use the posterior potential
   `Phi(u) = |Gamma^-1/2 (y - G(u))|^2 / 2 + |u|^2 / 2`.

### Interacting Langevin sampler

4. **Dynamics** ([EQ-INV-ALDI-001], [EQ-INV-ALDI-002]): with `C` the
   ensemble covariance (divisor `J`),
   `du_j = -[C^ug Gamma^-1 (g_j - y) + C u_j] dt + ((d + 1)/J)(u_j - m) dt
   + sqrt(2) C^1/2 dW_j`, `C^1/2 = (u_1 - m, ..., u_J - m)/sqrt(J)`.
5. **Derivation of the invariance.** (a) For `G(u) = A u + b`,
   `C^ug = C A'`, so the ensemble difference is `C A' Gamma^-1 (g_j - y)`,
   the exact preconditioned gradient. (b) From
   `C = sum_k u_k u_k'/J - m m'`,
   `sum_b dC^ab/du_j^b = ((d + 1)/J)(u_j - m)^a`. (c) The noise of member `j`
   has covariance `2 C dt` and is independent across members, so the flux of
   member `j` is `rho (-C grad_j Phi + div_j C) - div_j (C rho)
   = -rho C grad_j Phi - C grad_j rho`, zero at
   `rho = prod_j exp(-Phi(u_j))`. The identity (b) is checked by finite
   differences and one discrete step against the analytic gradient in the
   tests.
6. **Without the correction** (`correction=False`, the ensemble Kalman
   sampler) the same flux computation with `rho = prod pi * phi` requires
   `grad_j log phi = -((d + 1)/J) C^-1 (u_j - m)
   = -((d + 1)/2) grad_j log det C`, so the invariant density is
   `prod_j pi(u_j) det(C)^(-(d+1)/2)`: `J C` is Wishart with `J - d - 2`
   instead of `J - 1` degrees of freedom, `E[C] = ((J - d - 2)/J) Sigma`,
   the ensemble mean keeps covariance `Sigma/J`, and the pooled covariance
   is `((J - d - 1)/J) Sigma` ([EQ-INV-ALDI-004]). The covariance is biased
   low; the density is normalizable only for `J > 2 d + 1`.
7. **Discretization** ([EQ-INV-ALDI-003]): explicit Euler-Maruyama with
   `h_n = eta/(tr C + tr C^gg)`, `eta = 0.1`, or a user step. For an affine
   map `tr C + tr C^gg = tr(C P)` bounds the largest eigenvalue of the drift
   matrix `C P`, which gives stability of the deterministic part for
   `eta < 2`. The rule is documented as a heuristic. Because `h_n` depends
   on the state, the chain over-represents states with small steps; time
   averages are weighted by `h_n`. (An unweighted average of the ensemble
   variance was 15 % too large at `J = 8`; the weighted one is unbiased
   within error.) `step_history` and `sampling_time` are reported.
8. **Estimates** are step-weighted pooled moments of all members after the
   burn-in (default: half of the steps), with a batch-means standard error
   of the mean. `J >= d + 2` is required. A member beyond 1000 prior
   standard deviations raises `ValueError` (divergence of a fixed step).

### Unscented Kalman inversion

9. **Hyperparameters, derived** ([EQ-INV-UKI-001], [EQ-INV-UKI-003]). The
   general scheme predicts `m^ = alpha m + (1 - alpha) r_0`,
   `C^ = alpha^2 C + Sigma_w` and analyses with error covariance `Sigma_nu`.
   For `G(u) = A u + b` the analysis in information form is
   `C_{n+1}^-1 = C^^-1 + A' Sigma_nu^-1 A`. With `Sigma_w = C_n`, `alpha = 1`
   and `Sigma_nu = 2 Gamma` the fixed point is `C^-1 = A' Gamma^-1 A`, the
   posterior under a *flat* prior. To include the prior `N(0, I)` the
   observation is augmented: `G_a(u) = (G(u), u)`, `y_a = (y, 0)`,
   `Gamma_a = diag(Gamma, I)`, `Sigma_nu = 2 Gamma_a`, so that
   `A_a' Gamma_a^-1 A_a = A' Gamma^-1 A + I = P`. Then
   `C_{n+1}^-1 = (C_n^-1 + P)/2` and
   `C_{n+1}^-1 m_{n+1} = (C_n^-1 m_n + A' Gamma^-1 (y - b))/2`: the unique
   fixed point is the exact posterior mean **and** covariance, with error
   ratio 1/2 per iteration. With `alpha < 1`, `r_0 = 0` and
   `Sigma_w = (2 - alpha^2) C_n` the covariance fixed point is unchanged
   but the mean becomes `mu/(2 - alpha)`; `alpha` is therefore fixed at one
   and not exposed. The recalled choice "`Sigma_nu = 2 Gamma`,
   `Sigma_w` proportional to `C`" is confirmed only together with the
   augmentation.
10. **Sigma points** ([EQ-INV-UKI-002]): `m` and
    `m +/- sqrt(d + kappa) L e_i`, weights `kappa/(d + kappa)` and
    `1/(2 (d + kappa))`, `kappa >= 0` (default 1, a library choice; the
    modified weights of the cited paper could not be confirmed and are not
    used). The affine case does not depend on `kappa`.
11. **Stopping**: scaled change of mean and covariance at most `tolerance`
    (default `1e-8`) within `iterations` (default 50); otherwise
    `converged=False` and the summary says so. Cost `(2 d + 1)` per
    iteration. The `theta` moments are an unscented push-forward of
    `N(m, C)` through the prior map (no forward runs).

### Consensus-based sampling

12. **Update** ([EQ-INV-CBS-001], [EQ-INV-CBS-002]): weights
    `w_j ~ exp(-beta Phi(u_j))`, weighted `m_b`, `C_b`,
    `u_j <- m_b + alpha (u_j - m_b) + sqrt((1 - alpha^2)/lambda) C_b^1/2 xi_j`,
    `lambda = 1/(1 + beta)` (sampling) or 1 (optimization).
13. **Derivation of the fixed point** ([EQ-INV-CBS-003]). For ensemble law
    `N(m, C)` and quadratic `Phi` with minimizer `mu` and Hessian
    `Sigma^-1`: `C_b^-1 = C^-1 + beta Sigma^-1`; the update gives
    `N(m_b + alpha (m - m_b), alpha^2 C + (1 - alpha^2) C_b/lambda)`. A fixed
    point needs `m = mu` and `(1 - lambda) C^-1 = lambda beta Sigma^-1`,
    i.e. `C = Sigma` for `lambda = 1/(1 + beta)` and `C = 0` for
    `lambda = 1`. Linearized at the fixed point both mean and covariance
    errors contract by `alpha^2 + (1 - alpha^2)/(1 + beta)` per update.
14. **Effective sample size.** A prior ensemble against a narrow posterior
    gives weights carried by a few members and a singular `C_b`. Each update
    uses the largest `beta <= inverse_temperature` with
    `J_eff >= minimum_effective_fraction * J` (default 0.5, bisection). The
    fixed point does not depend on `beta`, so the target of the mean-field
    iteration is unchanged; this is a library heuristic.
15. **Finite ensemble.** The self-normalized `C_b` is low by about
    `1/J_eff`, which moves the fixed point to about
    `(1 - (1 + beta)/(beta J_eff)) Sigma`; this is documented, not corrected.

## Consequences

Measured on the straight-line and exponential-decay problems of ADR-0115,
means over seeds 0-4 (no selection). "Mean" is `|mean - reference|` in
posterior standard deviations and "sd" the ratio of the estimated to the
reference standard deviation, for the two parameters. References: conjugate
posterior (line); MCMC `(5.0073, 0.3011)`, sd `(0.0710, 0.0065)` (decay).

| Method | Evaluations | Line mean | Line sd | Decay mean | Decay sd |
|---|---:|---:|---:|---:|---:|
| ES-MDA `N = 1000`, `alpha = (2, 2)` | 3000 | 0.02, 0.03 | 1.00, 1.00 | 0.24, 0.86 | 2.33, 6.41 |
| ES-MDA `N = 200`, `alpha = (8,) * 8` | 1800 | 0.05, 0.05 | 1.01, 1.00 | 0.07, 0.06 | 1.02, 1.11 |
| Interacting Langevin `J = 8`, 1000 steps | 8008 | 0.09, 0.05 | 0.98, 1.04 | 0.09, 0.06 | 0.98, 1.02 |
| Interacting Langevin `J = 20`, 1000 steps | 20020 | 0.03, 0.05 | 1.02, 1.01 | 0.06, 0.03 | 0.94, 1.00 |
| Uncorrected sampler `J = 8`, 1000 steps | 8008 | 0.09, 0.06 | 0.83, 0.85 | 0.09, 0.09 | 0.80, 0.80 |
| Unscented Kalman inversion | 135 / 145 | 0.00, 0.00 | 1.00, 1.00 | 0.04, 0.03 | 0.96, 0.98 |
| Consensus sampling `J = 200`, 30 updates | 6200 | 0.05, 0.06 | 0.97, 0.98 | 0.07, 0.04 | 0.92, 0.94 |
| Consensus sampling `J = 1000`, 30 updates | 31000 | 0.05, 0.05 | 1.00, 1.01 | 0.05, 0.03 | 0.96, 0.97 |

ADR-0115 recorded ES-MDA at 3000 evaluations (line, `N = 1000`) and 1800
(decay, `N = 200`, spread of `k` 14 % too wide for seed 1; 11 % on average
over seeds 0-4 here).

Where each method is worse:

- **Interacting Langevin sampler**: four to ten times the forward
  evaluations of ES-MDA for a similar or larger error of the mean, because
  the ensemble must be integrated over many decorrelation times and
  successive ensembles are correlated; an `O(h)` discretization bias; the
  step rule is a heuristic; the steps are sequential, so only `J` runs are
  parallel. Its advantages are the small ensemble and the absence of an
  `O(1/J)` bias: with `J = 6` (eight replicates of 1000 steps) the
  covariance ratio is `0.97 +/- 0.05`, against `0.55 +/- 0.04` without the
  correction (derived: 0.5).
- **Unscented Kalman inversion**: cheapest here and exact for linear
  problems, but its cost grows as `2 d + 1` per iteration with a dense
  `d x d` covariance and an `(M + d)`-dimensional solve, so it does not
  suit high-dimensional fields where ensemble methods with `J << d` still
  run; it returns a Gaussian and cannot represent skewness or
  multimodality; convergence is not guaranteed for nonlinear models. On the
  decay problem its standard deviations are 2-4 % below the MCMC values.
- **Consensus-based sampling**: three to seventeen times the evaluations of
  ES-MDA; the spread is 2-8 % too small at `J = 200`; the early updates run
  at `beta ~ 0.001-0.01` because the prior ensemble is far wider than the
  posterior; in optimization mode the covariance contracts only
  algebraically (about `1/n`), far slower than EKI's collapse. Its
  advantage is that it needs only values of `Phi`.

No existing behaviour changes.

## Rejected alternatives

- **Importing `filtering.unscented_transform`.** Forbidden by the layer map
  (`filtering` is layer 6).
- **Exposing `alpha`, `r_0`, `Sigma_w`, `Sigma_nu` of unscented Kalman
  inversion.** Only one combination has the posterior as its linear fixed
  point (item 9); the others would silently return a different Gaussian.
- **Unscented inversion without the prior augmentation.** Its fixed point is
  the flat-prior posterior, which is not what the other calibration routes
  of the same problem compute.
- **A gradient-based Langevin drift** (through `log_posterior_gradient`).
  The purpose is a derivative-free method comparable with EKI and ES-MDA.
- **An unweighted time average** for the adaptive step. Biased, as measured
  in item 7.
- **Correcting the finite-ensemble bias of the consensus covariance** with
  a `1/(1 - sum w_j^2)` factor. Plausible but not in the cited source and
  not verified to remove the bias of the iteration; left out.
- **Reusing `EnsembleInversionResult`.** The three methods report different
  things (time averages and step history; a Gaussian with a convergence
  flag; temperature and effective-size histories).
- **A lab under `examples/`.** Out of scope.

## Verification

`tests/test_interacting_particle_inversion.py` (29 tests, about 12 s):
the conjugate posterior by normal equations in `u`; eight seeded Langevin
replicates (`J = 6`) with and without the correction against the exact mean
and covariance and against the derived deficits `(J - d - 1)/J` and
`(J - d - 2)/(J - 1)`; no collapse, in contrast to EKI; one Euler-Maruyama
step recomputed from the documented streams with the analytic gradient; the
divergence identity by finite differences; unscented inversion against the
exact posterior (1e-9), against the precision recursion iterate by iterate,
its rate 1/2 and monotone convergence, `kappa` independence, sigma-point
moments, the unconverged flag; the reweighted-Gaussian covariance
`1/(1 + beta)` and effective-size formula; the tempering rule; six seeded
consensus replicates (`J = 400`) against the exact mean and covariance;
concentration of the optimization mode at the MAP point; all three methods
against the MCMC posterior of the exponential decay; reproducibility and
input validation.

## References

[REF-GARBUNO-INIGO-ETAL-2020], [REF-GARBUNO-INIGO-NUSKEN-REICH-2020],
[REF-HUANG-SCHNEIDER-STUART-2022], [REF-CARRILLO-ETAL-2022],
[REF-JULIER-UHLMANN-2004], [REF-IGLESIAS-LAW-STUART-2013],
[REF-EMERICK-REYNOLDS-2013].
