# ADR-0029 — Multilevel Monte Carlo, CE-IS, line sampling, and Shapley effects

- Status: Accepted
- Date: 2026-10-11

## Context

The roadmap left four items open: multilevel Monte Carlo and adaptive
importance sampling (UQ-3), cross-entropy adaptation and line sampling
(UQ-4), and correlated-input Shapley effects (UQ-5). The library is pure
Python with no runtime dependencies. Every estimator must report its cost, take
seeds through `RandomStream`, and keep the layer map: `monte_carlo` is layer 3,
`reliability` and `sensitivity` are layer 5, and `sensitivity` must not import
`reliability`.

## Decision

1. **MLMC lives in `monte_carlo.multilevel` (layer 3) and knows nothing about
   models.** The user supplies the hierarchy as
   `level_sampler(level, n, stream) -> (fine, coarse)`. Coupling is the
   sampler's job; the library owns allocation, rates, bias extrapolation, and
   cost. Batch `k` of level `l` draws from `root.child("mlmc-level", l, k)`.
2. **Cost is a user cost model, not wall-clock time.** A fine path costs
   `path_cost(l)` (default `M^l`) and a correction sample costs
   `c(l) + c(l-1)`, so `total_cost` can be checked exactly (the GBM test counts
   Euler steps). `single_level_cost` is plain MC on the finest level with the
   *same sampling variance*, which compares at equal RMSE because both
   estimators share the finest-level bias.
3. **The adaptive driver follows Giles' algorithm.** The MSE split is
   `eps^2/2 + eps^2/2`, `alpha` and `beta` come from regression with a lower
   bound of `1/2`, `m_l` and `V_l` are floored from level 2, and the bias is
   extrapolated from the three finest corrections. Reaching `max_level`
   returns `converged=False`; it does not raise.
4. **CE-IS and line sampling live in a new `reliability.adaptive` module.**
   `simulation` imports it for `compare_reliability_methods`. Both are
   **opt-in** method keys (`"ce"`, `"line"`). `METHODS` stays at the previous
   five, and `AVAILABLE_METHODS = METHODS + OPTIONAL_METHODS` is what the
   comparison validates against. Default comparisons, the `UQStudy`
   reliability stage, and the published example numbers are therefore
   unchanged.
5. **CE-IS uses one Gaussian proposal, adapting the mean by default.**
   Covariance adaptation is optional and documented as fragile. On a planar
   limit state with `beta = 4` it stalled in 3 of 5 seeds, and another seed
   finished at 0.43 of the exact value. With `rho = 0.2` one run reported a
   c.o.v. of 0.04 while 27% off. Mixture proposals are
   out of scope.
6. **Line sampling defaults to the FORM `alpha` and reuses the FORM result**
   when the comparison already ran FORM. Each line brackets the root by
   doubling from the previous root and refines it with the existing
   `optimization.brent_root`. Values are cached so the bracket ends cost
   nothing extra. Lines with no crossing contribute 0 or 1.
7. **Shapley effects use the Song-Nelson-Staum cost `E[Var(Y | X_-J)]` with
   conditional sampling in the latent Gaussian-copula space** (Schur
   complements via `linalg.solve_spd` and `cholesky`). Exact subset
   enumeration is the default for `d <= 6`, and random permutations are used
   otherwise. Using the same `Var Y` estimate for `c(D)` makes the effects sum
   to exactly one. `linear_gaussian_shapley` gives an analytic reference for
   any dimension. Its two-dimensional closed form was derived from the
   definition and checked numerically (difference at most `1.9e-16`) before
   the tests relied on it.

## Consequences

- MLMC works for any coupled hierarchy (SDEs, PDE meshes, surrogates of
  increasing fidelity) without importing `processes` or `modeling`.
- `compare_reliability_methods` gained two keyword arguments
  (`cross_entropy_samples`, `line_samples`). Existing calls are unaffected.
- Shapley effects require a Gaussian-copula (or independent) joint law. Other
  dependence structures would need a different conditional sampler.
- Shapley standard errors are delta-method approximations that ignore the
  normalizing variance. MLMC's in-run `single_level_cost` is noisy when few
  finest-level samples exist; the documentation reports both the in-run value
  and the value from the exact variance.

## Rejected alternatives

- Wall-clock cost measurement for MLMC: not reproducible, and it makes tests
  flaky.
- Adding `"ce"` to the default `METHODS`: about 4000 extra calls per
  comparison and changed published numbers, for no default-path benefit.
- Pick-freeze Sobol estimators "corrected" for correlation: their indices do
  not form a variance budget under dependence. Shapley effects do.
- Making `sensitivity` reuse reliability's Nataf helpers: forbidden by the
  layer contract, and unnecessary because `JointInput` exposes everything
  needed.
