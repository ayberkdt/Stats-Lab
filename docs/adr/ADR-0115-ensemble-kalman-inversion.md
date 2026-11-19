# ADR-0115 — Ensemble Kalman inversion and ES-MDA for calibration problems

- Status: Accepted
- Date: 2026-11-19
- Extends: ADR-0017 (nonlinear calibration and MCMC) and ADR-0111 (pCN,
  tempered SMC, ABC)

## Context

Every posterior route of `CalibrationProblem` evaluates the forward model one
parameter point at a time inside a Markov chain or an SMC mutation, and the
gradient-based samplers add `2n + 1` runs per leapfrog step. For expensive
simulators without gradients the common engineering alternative is an
ensemble Kalman method: a few hundred independent forward runs per
iteration, a Kalman-type update whose gain is estimated from the ensemble,
and a handful of iterations. The library had ensemble Kalman *filters* for
state estimation (`data_assimilation`), but nothing for static parameter
calibration, and `inverse` (layer 4) may not import `data_assimilation`.

Two variants matter in practice and they answer different questions:

- **Ensemble Kalman inversion (EKI)** iterates the perturbed-data update and
  behaves as a derivative-free optimizer: the ensemble stays in the span of
  the initial members and collapses (Iglesias, Law and Stuart 2013;
  Schillings and Stuart 2017).
- **ES-MDA** assimilates the same data a fixed number of times with inflated
  error covariances whose reciprocals sum to one, which reproduces the
  Bayesian update for linear-Gaussian problems (Emerick and Reynolds 2013).

## Decision

1. **New module `inverse/ensemble_inversion.py`** with
   `ensemble_kalman_inversion`, `ensemble_smoother_mda`, and the frozen
   result `EnsembleInversionResult`. It depends only on `foundations`,
   `linalg`, and `inverse.nonlinear` (through the public attributes and
   methods of `CalibrationProblem`); the ensemble statistics and the gain
   are written locally, so the layer map is unchanged.
2. **Coordinates.** The ensemble lives in the Nataf standard-normal space
   `u` of the prior, where the prior is exactly `N(0, I)`; members are
   mapped through `model.inputs.from_standard_normal` before
   `CalibrationProblem.predictions` runs the model. Gaussian priors with a
   linear model give an affine map in `u`, which is where the ES-MDA
   exactness statement applies.
3. **One shared update** ([EQ-INV-EKI-001]) in whitened observation space:
   with deviations `A` of `u` and `B` of `Gamma^-1/2 G(u)`, the gain is
   `A'B (B'B + alpha (N - 1) I_M)^-1`, or the equal
   `A' (B B' + alpha (N - 1) I_N)^-1 B` when `M > N`, so the factored
   system always has the smaller dimension. Covariances use the divisor
   `N - 1` (Iglesias, Law and Stuart use `N`; the ratio in the gain changes
   by `O(1/N)`). Perturbations `sqrt(alpha) z_j` are drawn member by member
   from the child stream `("perturbations", n)` of update `n`; the prior
   ensemble comes from the child `"ensemble"`. EKI with one iteration and
   ES-MDA with `alpha = (1,)` are therefore the same computation for the
   same seed.
4. **Known Gaussian noise only.** `Gamma` is diagonal with
   `sigma_q^2 + sigma_delta_q^2` in the order of `CalibrationProblem.observed`.
   Unknown noise scales and Student-t models raise `ValueError` that states
   why (no fixed `Gamma`; no second-moment update for heavy tails) and
   points to `sample_posterior`.
5. **Stopping.** EKI runs a fixed number of updates by default
   (`stopping="iterations"`). `stopping="discrepancy"` applies the
   discrepancy principle of Iglesias (2016, Algorithm 1, Eq. 6) to the
   ensemble mean of the predictions ([EQ-INV-EKI-002]):
   stop when `|Gamma^-1/2 (y - mean_j G(u_j))| <= tau eta`. `eta` defaults
   to `sqrt(M)`, the root of the expected squared whitened noise norm, and
   `tau` to 1.5; both are library choices (Iglesias, Law and Stuart require
   `tau > 1`; Iglesias 2016 uses `tau > 1/rho` with `rho = 0.7` in his
   examples). `tau < 1` is rejected. Iglesias's Levenberg-Marquardt
   adaptation of the regularization parameter is not implemented; EKI uses
   `alpha = 1`.
6. **ES-MDA inflation** is a required argument: the coefficients must be
   positive, finite, and satisfy `sum 1/alpha_i = 1` to `1e-9`; a schedule
   is never normalized silently ([EQ-INV-ESMDA-001]).
7. **Result.** Final ensemble in `theta` and in `u`; mean and spread
   histories in both coordinates and the whitened misfit of the mean
   prediction for every evaluated ensemble (`updates + 1` entries, the
   final ensemble is always evaluated once more); the inflation used;
   `stopped_by`; `forward_evaluations = N (updates + 1)` and `model_calls`;
   `mean()`, `covariance()`, `standard_covariance()`, `draws()`, and a
   `summary()` sentence that, for EKI, states that the spread is not a
   posterior uncertainty.

## Consequences

- On the straight-line problem (`N = 1000`, `alpha = (2, 2)`, seed 17)
  ES-MDA gives mean `(0.6470, 0.7827)` and variances `0.04200`, `0.01334`
  against the exact `(0.6417, 0.7875)`, `0.04099`, `0.01260`; twenty
  replicate runs with `N = 200` agree with the exact mean and covariance
  within four replicate standard errors in the tests.
- On the exponential decay (`N = 200`, seed 1) ES-MDA with eight
  coefficients of 8 gives mean `(5.0053, 0.3008)` against the MCMC
  `(5.0073, 0.3011)` with 1800 forward evaluations; its spread of `k` is
  14 % wider than the posterior (43 % with four coefficients of 4). The
  guide states that the nonlinear ensemble spread is an approximation.
- EKI on the same problem (`N = 50`, 20 iterations, seed 1) reduces the
  whitened misfit from 20.02 to 3.890 while the spreads shrink to about a
  fifth of the posterior standard deviations; the discrepancy rule stops
  after two updates.
- Problems with unknown or heavy-tailed noise must use the samplers.

## Rejected alternatives

- **Reusing `data_assimilation.EnsembleKalmanFilter`.** Forbidden by the
  layer map (`inverse` is layer 4 and may not import `data_assimilation`)
  and built around state-space models with a forecast step that static
  inversion does not have.
- **Estimating the noise scale inside the ensemble update.** An unknown
  `Gamma` has no Kalman update; treating `log sigma` as an extra ensemble
  coordinate would give an ad-hoc estimator with no exactness property.
- **Normalizing an inflation schedule that violates `sum 1/alpha = 1`.**
  Convenient but silent; it would change what the user asked for.
- **Returning a `CalibrationPosterior`.** It would invite reading the EKI
  ensemble as posterior draws and attach chain diagnostics that do not
  apply; a separate result type keeps the distinction visible.
- **Implementing the Levenberg-Marquardt regularizing ensemble method of
  Iglesias (2016) in full.** Its adaptive `alpha_n` needs extra solves per
  iteration; only its discrepancy-principle stopping rule is adopted here.

## Verification

`tests/test_ensemble_inversion.py` (15 tests, about 10 s) checks the exact
conjugate posterior two ways, ES-MDA moments against it (replicates with
`N = 200` and one run with `N = 1000`), the one-step smoother against an
independent recomputation from the documented random streams, the equality
of the two gain forms, `Gamma` with a discrepancy variance, EKI misfit
reduction, collapse, convergence towards least squares and monotone misfit
without perturbations, the discrepancy rule, ES-MDA against the MCMC
posterior of the exponential decay, reproducibility, and input validation.

## References

[REF-IGLESIAS-LAW-STUART-2013], [REF-IGLESIAS-2016],
[REF-EMERICK-REYNOLDS-2013], [REF-SCHILLINGS-STUART-2017],
[REF-EVENSEN-2003].
