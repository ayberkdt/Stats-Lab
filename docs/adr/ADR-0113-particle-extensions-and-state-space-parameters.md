# ADR-0113: Particle-filter extensions and state-space parameter estimation

- Status: Accepted
- Date: 2026-11-19

## Context

`probstat.filtering` (ADR-0018) had one particle filter, the bootstrap
filter, and no way to estimate the static parameters of a state-space model:
`Q`, `R`, `F`, `H`, or the parameters of a nonlinear model all had to be known.
Two gaps followed. The bootstrap filter proposes from the transition prior and
wastes particles when observations are precise, and it gives no smoothed
trajectories. And the filter log-likelihood, which every filter already
returns, was not used for estimation, even though maximum likelihood, EM, and
particle MCMC are the standard ways to calibrate a dynamic model.

## Decision

1. Two particle filters next to `bootstrap_particle_filter`, in
   `filtering.particle_proposals`, both returning `ParticleFilterResult`:
   - `auxiliary_particle_filter` (Pitt and Shephard): first-stage weights
     `W_(k-1)^i g(y_k | f(x_(k-1)^i))` from the predictive mean, systematic
     resampling at every observed step, second-stage weights
     `g(y_k | x_k) / g(y_k | mu)`, and the likelihood increment
     `[sum_i W^i g(y_k | mu^i)] (1/N) sum_j w_k^j` of Doucet and Johansen
     (2011, Section 4.2), whose product is unbiased;
   - `optimal_proposal_particle_filter`: the closed-form
     `p(x_k | x_(k-1), y_k)` with weight `p(y_k | x_(k-1))`, valid only for a
     linear observation operator. A `LinearGaussianStateSpace` supplies `H`;
     a `NonlinearStateSpace` must pass `observation_matrix`, and `h(x) = H x`
     is checked at `m_0` and `m_0 + e_j` before filtering. A nonlinear `h`
     raises `ValueError`; no linearization is substituted silently.
2. `bootstrap_particle_filter` (and the two new filters) gain the opt-in flag
   `store_history=False`. When true, the result keeps the weighted cloud of
   every step (after weighting, before resampling) in `particle_history` and
   `weight_history`; the random draws and every existing number are the same
   with or without the flag. `ffbs_particle_smoother` in
   `filtering.particle_smoothing` draws trajectories by forward filtering,
   backward sampling (Godsill, Doucet, and West) from such a result. It
   requires a positive-definite `Q`, because the backward weights evaluate
   the transition density.
3. A new module `filtering.parameter_estimation`:
   - `fit_state_space_mle(builder, theta_0, y, bounds=..., filter=...)`
     minimizes the negative Kalman (or EKF/UKF) log-likelihood with
     `optimization.minimize_bfgs` and reports the inverse observed
     information from `optimization.central_hessian`. When the Hessian
     cannot be evaluated or is not positive definite, `covariance` is `None`
     with a `message`; nothing is regularized. Constraints use `bounds` (the
     box transform of the optimizer) or a builder on unconstrained
     parameters such as log-variances, which the guide recommends.
   - `linear_gaussian_em(model, y, estimate=...)` runs the Shumway-Stoffer EM
     with the RTS smoother and `smoothed_lag_one_covariances`
     (`P^s_k G_(k-1)'`, from the gains `rts_smoother` already returns). The
     M-step is closed form per block; fixed blocks stay fixed. Missing
     observations are left out of the observation-block averages. The
     log-likelihood trace is returned so that monotonicity is checkable.
   - `particle_marginal_metropolis_hastings` writes its own loop instead of
     wrapping `inference.random_walk_metropolis`, because the target must not
     be re-evaluated at the current state: the estimate stored at acceptance
     is reused, which is what makes PMMH exact. It returns an
     `inference.MCMCResult` (one `MarkovChain` per chain, `evaluations` =
     filter runs), so the existing diagnostics apply unchanged. Proposals
     with zero prior density are rejected without a filter run; a filter
     whose weights all underflow is the estimate zero and is rejected.
4. Stream names follow ADR-0018: the new filters use `initial`,
   `propagation`, `resampling`; the smoother uses `backward`; PMMH uses
   `pmmh-chain` (spawned), and per chain `proposal` and `likelihood/<i>`.
5. Rao-Blackwellized particle filtering is out of scope: it needs a
   conditionally linear-Gaussian model class (a state split into a part
   sampled by particles and a part handled by one Kalman filter per particle),
   which `StateSpaceModel` cannot express without a new model type.

## Consequences

Every new estimator has an exact reference in the tests: the noise-free case
reproduces the Kalman likelihood to rounding; seeded replications show
`E[exp(log p_hat - log p)] = 1` within four standard errors for the auxiliary
and optimal-proposal filters; FFBS moments approach the RTS smoother; EM is
monotone and agrees with BFGS; the Nile local-level fit matches R's
`StructTS`; and the PMMH posterior of an AR coefficient matches quadrature of
the exact posterior within four Monte Carlo standard errors. `parameter_estimation`
imports `inference` (layer 2) for `MCMCResult`, which the layer map allows.
Pure Python keeps the sizes small: PMMH with a few dozen particles and a few
thousand iterations takes seconds, not minutes.

## Rejected alternatives

- Calling `random_walk_metropolis` with a noisy log-density: exactness would
  then rest on an implementation detail of another sampler (that it caches
  the current value rather than re-evaluating it), and the loop also needs a
  named filter stream per iteration and a count of filter runs.
- An EKF-linearized "optimal" proposal for nonlinear `h`: useful, but no
  longer optimal and not exact; a separate, explicitly named proposal would
  be needed.
- Storing particle histories by default: it multiplies memory by `T` for
  every caller, while only smoothing needs it.
- A Hessian with jitter when the observed information is indefinite: it
  would report a covariance for a non-identified or boundary estimate.
