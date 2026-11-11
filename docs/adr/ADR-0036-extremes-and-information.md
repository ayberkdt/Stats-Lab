# ADR-0036 — Extreme value analysis and information theory

- Status: Accepted
- Date: 2026-10-12

## Context

The library already had a GEV law, a Gumbel law, return periods, and a
generic maximum-likelihood fitter (`inference.fit_distribution` with
`GEV_FAMILY`). It had no generalized Pareto law, no peaks-over-threshold
workflow, and no return-level intervals. Phase 2 of the roadmap lists
"entropy interfaces" as partly done; only `Categorical.entropy` existed. The
library is pure Python with no runtime dependencies, and the layer map is
enforced by `tests/test_architecture.py`.

## Decision

1. **`GeneralizedPareto` is a layer-1 distribution** with the same
   `(mu, sigma, xi)` order and sign convention as `GeneralizedExtremeValue`.
   - `mu` is the threshold.
   - Every formula uses `log1p(xi z)/xi` or `expm1(xi L)/xi`, so `xi -> 0`
     is the exponential law with no branch on `xi == 0` beyond an
     underflow guard.
   - It has `pdf`, `log_pdf`, `cdf`, `sf`, `log_sf`, `hazard`, `quantile`,
     `isf`, `mean`, `variance`, `entropy`, and seeded `sample`.
2. **Extreme value analysis lives in `risk.extremes` (layer 6).** It imports
   `inference` (2), `modeling` (3, for the normal quantile), `optimization`,
   `distributions`, and `linalg`. It does not add a GPD family to
   `inference.STANDARD_FAMILIES`. Doing so would change
   `select_distribution` results for every existing user. Instead,
   `risk.GPD_FAMILY` is a local `DistributionFamily` passed to the same
   generic fitter.
3. **Block maxima reuse the generic fitter.** `fit_gev` calls
   `fit_distribution(maxima, GEV_FAMILY)` and wraps the result in `GEVFit`.
   - Delta-method intervals use the observed-information covariance from
     that fit.
   - The `xi`-derivative of `expm1(a t)/a` switches to a power series for
     `|a t| < 0.05`, so gradients are smooth through `xi = 0`.
4. **Profile likelihood only for GEV return levels.**
   - The GEV is reparameterized by `(z_T, log sigma, xi)`. Nelder-Mead
     maximizes over the nuisance parameters, warm-started from the nearest
     previously solved `z`.
   - Each endpoint is bracketed by doubling steps of the delta-method
     standard error and refined with `optimization.brent_root`.
   - An endpoint that is not bracketed after 40 doublings is reported as
     `±inf`; it is never extrapolated.
   - POT return levels get delta-method intervals only.
5. **PWM equals two-L-moment estimation.** With unbiased `b_r`, the
   Hosking-Wallis PWM estimator and the L-moment estimator coincide
   algebraically. One implementation serves both, and `sample_l_moments`
   exposes `l1..l4`.
   - PWM fits report no covariance. The Hosking-Wallis asymptotic
     covariance was not implemented because its coefficients were not
     verified in this session.
   - A POT return level from a PWM fit therefore has `method="point"` and
     `nan` limits.
6. **POT rate model.** Exceedances are binomial in `zeta_u = k/n` with
   variance `zeta_u (1 - zeta_u)/n`, independent of the GPD estimates. With
   declustering, `zeta_u` counts clusters. Return periods are in years
   through `observations_per_year`.
7. **Information theory is a new package `probstat.information` at layer
   2.** It imports only layers 0-1: `distributions` for the dispatcher,
   `linalg`, `numerics.digamma`, and `foundations`. It does not import its
   layer-2 sibling `multivariate`: the multivariate functions take a mean
   vector and a covariance matrix directly.
8. **Entropy as functions, not as properties on existing classes.**
   - Adding `entropy` to eight existing distribution modules would edit
     files owned by other work for a small convenience. Instead,
     `information.differential_entropy(law)` dispatches by type.
   - The new `GeneralizedPareto` carries an `entropy` property because it is
     new code.
   - The roadmap's "entropy interfaces" item is therefore served by
     functions. A protocol-level `entropy` capability is left for a
     future ADR.
9. **Nearest-neighbour estimators use exhaustive search.**
   Kozachenko-Leonenko and KSG are `O(n^2 d)` in pure Python. Duplicated
   points (zero radii) raise `ValueError`. KSG results are not clipped at
   zero.

## Consequences

- `risk` gains 22 public names. `distributions` gains `GeneralizedPareto`,
  and the new `information` package has 23 public names. The architecture
  map gains `"probstat.information": 2`.
- 30 new equation records (`EQ-DIST-GPD-*`, `EQ-EVA-*`, `EQ-INFO-*`) and 8
  new references are appended at the end of the registry and the
  bibliography.
- Non-stationary extremes, point-process likelihoods, `r`-largest statistics,
  profile intervals for POT, and discrete-sample entropy estimators remain
  open.
