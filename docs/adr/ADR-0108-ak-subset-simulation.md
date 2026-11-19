# ADR-0108: AK-SS, active-learning Kriging on subset-simulation levels

- Status: Accepted
- Date: 2026-11-19

## Context

`reliability.ak_mcs` classifies a Monte Carlo population with a Gaussian
process. Its population must hold enough failures for the target c.o.v.,
about `(1 - P_f) / (P_f cov^2)` points, so a failure probability of `1e-5`
or `1e-6` needs `10^7` or more candidates, which a pure-Python GP cannot
predict over. Huang, Chen, and Zhu (Structural Safety 59, 2016) replace the
population by the samples of a subset simulation run on the Kriging
predictor and learn on those samples with Echard's `U`. The published steps,
initial design, and call counts could not be checked against the paper, so
the scheme below fixes its own details and labels them.

ADR-0107 provides the pieces: `PopulationPosterior` with
`replace_population`, the learning functions and stopping rules, and the
subset kernels of ADR-0106.

## Decision

1. **New module `reliability.ak_subset`** with `ak_ss(limit_state, ...) ->
   AKSSResult` (per-round records `AKSSRound`, default regularization
   `AKSS_GP_REGULARIZATION`). Each round runs subset simulation on the
   surrogate mean (no model calls) with any kernel of `SUBSET_SAMPLERS`; the
   distinct samples of all levels form the candidate pool; points are added
   by the learning function (`U` by default) until its stopping value holds
   on the pool; then the next round reruns subset simulation on the refined
   surrogate.
2. **Common random numbers.** Every subset run uses the same random stream.
   Where the surrogate has not changed, consecutive runs and pools coincide,
   so learning converges like AK-MCS on a fixed population instead of
   chasing a new pool each round.
3. **Stop on the last level's bounds.** With `mu -/+ 1.96 sigma` at the
   `N` last-level samples, `P^-` and `P^+` bound the estimate given the
   levels; a round stops the run when the subset run reached `0`,
   `P_r > 0`, and `(P^+ - P^-) / P_r <= bounds_tolerance` (default `0.05`),
   or when its pool needs no addition. This adapts the bound rule of Schöbi
   et al. to the last subset level. A stop is accepted only on a GP fitted to
   all training data; otherwise the GP is refitted and the round repeated, so
   the estimate and the returned surrogate agree.
4. **GP regularization with a length-scale prior.** `AKSS_GP_REGULARIZATION`
   adds the log-normal prior `log ell_i ~ N(log E_i, 1)` to the ADR-0054
   bounds. Subset levels put many training points on nearly the same
   coordinate, which shrinks the smallest-gap lower bound and admits
   white-noise fits.
5. **Incremental posterior with reuse.** Between hyperparameter fits the GP
   posterior over the pool is updated by the rank-one steps of ADR-0049.
   `GaussianProcessPopulationPosterior.replace_population(points,
   reuse=True)` keeps the state of candidates that were already in the pool
   (equal to a fresh solve up to rounding; the default `reuse=False` is
   unchanged). The subset run evaluates the posterior mean through
   `alpha = L'^-1 z` at `O(n)` per point.
6. **Checks.** `check_population` (crude Monte Carlo on fresh points) and
   `check_levels` (the true model on the final pool) count sign errors; their
   calls are reported separately from `evaluations`.
7. **Not in `compare_reliability_methods`.** `simulation` would have to
   import `ak_subset`, which imports the subset kernels from `simulation`;
   the cycle and the cost of a pure-Python GP do not fit the comparison
   table. `OPTIONAL_METHODS` is unchanged.

## Evidence

Four-branch function, `k = 7`, `N = 1000` per level, seeds 1-3, the same
code except for the stopping rule and the regularization:

| configuration | calls (seeds 1, 2, 3) | rounds | converged |
| --- | --- | --- | --- |
| stop only on a pool without additions, ADR-0054 bounds | 97, 94, 98 | 22, 21, 18 | no, no, yes |
| + last-level bound stop (`0.05`) | 61, 64, 75 | 5, 5, 7 | yes, yes, yes |

Seed 5 with the bound stop and the ADR-0054 bounds alone: the length scale of
`u1` fell to `1e-3` and below after points with nearly equal `u1` were
added, the run used 211 calls, and the final subset run on the white-noise
GP did not reach the failure threshold in 20 levels. With the length-scale
prior the same seed used 76 calls and gave `1.85e-3`, with no sign error on
the final pool. Seeds 1-8 with the prior used 53-76 calls; the estimates
averaged `2.16e-3` against the reference `2.2245e-3`.

## Consequences

- Failure probabilities of `1e-5` to `1e-6` are within reach at tens to about
  a hundred model calls (`docs/foundations/adaptive-reliability.md` §5).
- The reported c.o.v. is the subset c.o.v. on the surrogate; the bounds are
  as overconfident as `sigma`, and failure regions that the subset run never
  samples are invisible (the blind-spot pitfall of AK-MCS remains).
- A round costs `m N` mean evaluations, a posterior set-up over the new pool
  points, and possibly a likelihood fit; runs take seconds to a minute in
  pure Python, which only pays off for expensive models.
