# ADR-0123 — Given-data and gradient-based sensitivity extensions

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0012 (sensitivity estimator boundaries) and ADR-0065
  (DGSM and given-data first-order indices)

## Context

`probstat.sensitivity` covered variance shares (pick-freeze Sobol, the
class-partition estimator `given_data_first_order`, Shapley effects),
screening (Morris, DGSM), regression coefficients, and HSIC. Four gaps
remained:

- first-order Sobol indices from one i.i.d. sample without a tuning
  parameter (the partition estimator needs a class count), and a
  distribution-free coefficient of functional dependence with a
  resampling-free test;
- indices that compare whole output distributions: `given_data.py` cites
  Plischke, Borgonovo and Smith (2013) for the class partition but
  implements only the variance-based index, not Borgonovo's `delta` or
  PAWN;
- kernel (maximum mean discrepancy) indices, of which HSIC is the
  given-data relative but which have a pick-freeze form that generalizes
  the Sobol index;
- directions, rather than coordinates, of sensitivity: active subspaces.

## Decision

1. **Four new modules** in `sensitivity` (layer 5): `rank_based.py`,
   `moment_independent.py`, `kernel_indices.py`, `active_subspace.py`. They
   import `foundations`, `linalg`, `numerics`, `optimization`, `statistics`,
   `modeling`, `propagation` and sibling modules of the same package; the
   layer map is unchanged.
2. **Rank-based Sobol indices** ([EQ-SENS-RANKSOBOL-001]):
   `rank_sobol_indices` and `model_rank_sobol_indices` implement the
   cyclic lag-one estimator of Gamboa, Gremaud, Klein and Lagnoux (2022) in
   the centred form `sum c_(j) c_(j+1) / sum c_j^2`, which is algebraically
   identical to the published formula because every row occurs once as
   left and once as right neighbour.
3. **No bootstrap and no asymptotic variance for the rank estimator.** The
   ordinary bootstrap is invalid (repeated rows are neighbours after
   sorting). The variance of the published central limit theorem was not
   derived and checked here, so it is not implemented. Instead `blocks=B`
   reports the standard error of the mean of `B` disjoint-block estimates
   ([EQ-SENS-RANKSOBOL-002]), documented as a diagnostic that transfers to
   the full sample through the `sqrt(n)` rate.
4. **Chatterjee's xi** ([EQ-SENS-XI-001], [EQ-SENS-XI-002],
   [EQ-SENS-XI-003]): `chatterjee_xi` and `chatterjee_indices` use the
   tie-aware formula, which was re-derived (registry record) and reduces
   exactly to the tie-free one. The `N(0, 2/5)` null limit gives the
   default p-value and is refused for tied outputs, whose null variance
   depends on the tie pattern; `test="permutation"` covers that case with
   one shared set of rank arrangements. Tied inputs are ordered uniformly
   at random from the seed and reported.
5. **Moment-independent indices** ([EQ-SENS-DELTA-001],
   [EQ-SENS-PAWN-001]): `borgonovo_delta` (histogram on bins of equal
   pooled frequency by default, Gaussian KDE with Silverman bandwidths as
   an option) and `pawn_indices` (exact Kolmogorov-Smirnov distances in
   integer arithmetic; median, mean or maximum over the classes) on
   equal-count classes of each input. Both report an optional permutation
   `null_level`, p-values and `corrected = index - null_level`
   ([EQ-SENS-DELTA-002]); the correction is labelled a heuristic. PAWN is
   implemented in its class-partition form on a generic sample, not with
   the tailored conditional design of the original paper.
6. **Kernel indices** ([EQ-SENS-KERNEL-001], [EQ-SENS-KERNEL-002]):
   `kernel_sobol_indices` and `model_kernel_sobol_indices` use the pooled
   symmetric V-statistic on `(y_B, y_i)`, chosen because the linear kernel
   then reproduces the library's Janon estimator exactly. Gaussian kernel
   with median-heuristic bandwidth by default (computed on at most about
   500 evenly spaced outputs), linear kernel, or a user function. Only
   first-order indices are provided.
7. **Active subspaces** ([EQ-SENS-ACTIVE-001] to [EQ-SENS-ACTIVE-004]):
   `active_subspace_from_gradients` diagonalizes `C_M` with
   `linalg.symmetric_eigen`; `active_subspace` samples a
   `ProbabilisticModel` in standard-normal space, with dual-number
   gradients and the marginal chain rule for independent inputs and central
   differences otherwise (the result records the method, the fallback
   reason and the model evaluations). The default dimension is the largest
   eigenvalue gap and is named a heuristic; bootstrap intervals and
   subspace distances measure sampling variability only.

## Consequences

- Rank-based Sobol indices of the Ishigami function (exact
  `0.3139, 0.4424, 0`): sixteen replicates give means
  `0.3234, 0.4413, -0.0152` with root-mean-square error 0.046 at `n = 500`
  and `0.3128, 0.4403, -0.0017` with 0.017 at `n = 4000`. For
  `Y = 2 X_1 + X_2 + X_3 / 2` (exact `0.762, 0.190, 0.048`) twelve
  replicates at `n = 2000` give `0.751, 0.188, 0.042`. The block standard
  error (`B = 10`, `n = 2000`) averages `0.020, 0.022, 0.030` against
  replicate standard deviations `0.021, 0.019, 0.028`.
- `xi_n` of a strictly monotone relation equals `1 - 3 / (n + 1)` exactly;
  1500 independent samples of size 200 give `n Var(xi_n) = 0.396` and a
  rejection rate of 5.1 % at the 5 % level.
- `delta` of `Y = 2 X_1 + X_2` (quadrature `0.5055, 0.1604`): histogram
  `0.482, 0.180` (corrected `0.419, 0.117`) at `n = 4000`, KDE
  `0.472, 0.175` at `n = 1500`; an inert input gives `0.059` against a null
  level of `0.063`. Permutation p-values of an inert input reject at
  4 % (delta) and 3 % (PAWN) for a nominal 5 % over 300 replicates.
- The linear kernel differs from `estimate_sobol_indices(...,
  estimator="janon")` by `2.2e-16` on 300 Ishigami rows. Gaussian-kernel
  indices cost `O(N^2)` kernel evaluations per input: 1.2 s for `N = 1000`
  and three inputs.
- Active subspace of `f = x'Ax/2` with 3000 gradients: eigenvalues
  `4.751, 0.809, 0.0074` against `4.901, 0.812, 0.0073` of `A^2`, mean
  bootstrap subspace distances `0.007, 0.002`.
- KDE-based `delta` costs `n * grid` kernel evaluations per input and
  permutation (0.25 s for three inputs at `n = 1500`), so its null level
  is practical only for small permutation counts; the histogram form is
  about a hundred times faster.

## Audit addendum (independent re-measurement, 2026-11-21)

- **Block standard error of the rank estimator, Ishigami, 600 replicates
  per setting.** Coverage of `estimate +- 2 se` for `S_1, S_2, S_3`:
  0.89, 0.86, 0.82 at `n = 100`, `B = 5`; 0.91, 0.91, 0.90 at `n = 500`,
  `B = 10`; 0.93, 0.93, 0.92 at `n = 2000`, `B = 10`; 0.93, 0.95, 0.92 at
  `n = 2000`, `B = 50`. The mean standard error is 0.79 to 0.96 of the
  replicate standard deviation at `n = 100`, 0.91 to 0.94 at `n = 500`
  and 0.96 to 1.00 at `n = 2000` with `B = 10`. So `+- 2 se` is not a 95 %
  interval: besides the `B - 1` degrees of freedom (a Student-t factor of
  2.26 for `B = 10`), the standard error is a few per cent too small. The
  bias of the full-sample estimator is at most 0.013 at `n = 100` and
  below 0.008 from `n = 500` on, while the mean of the block estimates is
  biased by up to -0.16 at a block size of 20 rows (`block_means` shows
  this, as documented).
- **Chatterjee's xi.** `sensitivity.chatterjee_xi` and
  `statistics.chatterjee_correlation` agree exactly on 300 random
  tie-free samples (`n` from 3 to 199); the latter refuses tied outputs.
  The tie-aware value equals a brute-force evaluation of
  `1 - n sum |r_(j+1) - r_(j)| / (2 sum l_j (n - l_j))` on 200 samples
  with four output levels. The asymptotic test is conservative for small
  samples (rejection rate at the 5 % level: 0.031 at `n = 8`, 0.041 at
  `n = 20`, 0.042 at `n = 50`, 0.056 at `n = 200`, 0.049 at `n = 1000`;
  `n Var(xi_n)` is 0.27, 0.36, 0.38, 0.39, 0.40). The permutation test
  with tied inputs and outputs rejected in 4.9 % (`n = 30`, three levels)
  and 4.1 % (`n = 100`, two levels) of 1000 null samples.
- **`corrected` delta.** For `Y = 2 X_1 + X_2` (exact 0.505, 0.160, and an
  inert third input) the mean corrected values were 0.367, 0.096, -0.001
  at `n = 500` (100 replicates) and 0.420, 0.109, 0.000 at `n = 4000` (40
  replicates), against raw means 0.449, 0.178, 0.082 and 0.485, 0.174,
  0.065. The correction is right for an inert input only; for the two
  active inputs it is 17 to 40 % too low, worse than the raw estimate.
  The field stays because the null level and the p-value are useful, but
  it is not an estimate of delta.
- **Attribution.** That the bootstrap of the gradient sample is in
  Constantine, Dow and Wang (2014) could not be verified (the paper could
  not be opened; a search attributes the bootstrap procedure to a later
  paper on Monte Carlo estimation of active subspaces). The registry and
  the bibliography no longer attribute the bootstrap diagnostics or the
  activity scores to the 2014 paper.
- **Active subspace.** Autodiff and central-difference moment matrices of
  a model with uniform, lognormal and normal inputs agree to `3e-11`; the
  bootstrap 95 % interval of the leading eigenvalue covered the truth in
  97 % of 200 samples of 200 gradients of a quadratic.

## Rejected alternatives

- **Bootstrap intervals for the rank estimators.** Biased by construction:
  on a resampled Ishigami table the index of the inert input rises from
  about 0 to above 0.25.
- **Implementing the asymptotic variance of the rank-based Sobol
  estimator.** The expression could not be re-derived and verified within
  this work package; a formula taken on trust would contradict the
  library's verification rule.
- **Using Chatterjee's general null variance for tied outputs.** Not
  re-derived here; the permutation test is exact in level instead.
- **A finite-sample variance of `xi_n` in place of 2/5.** Simulations
  suggest a slightly smaller finite-`n` variance (0.373 at `n = 50`), but
  no verified closed form was available.
- **Clipping `corrected` indices at zero.** It would hide the sampling
  noise that the null level is meant to show.
- **U-statistic kernel estimator on `y_A` alone.** Unbiased for
  `E[k(Y, Y')]`, but the linear-kernel special case would then match no
  estimator of the library, which removes the exact oracle.
- **Citing a separate source for activity scores.** The formula follows
  from the spectral decomposition and equals the DGSM diagonal for
  `k = d`; no further citation could be verified, so none is given.
- **Active subspaces in physical coordinates for models.** The
  eigenvectors would depend on the units of the inputs.

## Verification

`tests/test_sensitivity_extensions.py` (34 tests, about 5 s) checks: the
rank estimator against the literal published formula, Ishigami and
linear-Gaussian closed forms within replicate error with convergence, the
block standard error against replicate spread, the bootstrap bias, tie
handling; `xi` against the exact monotone value, a hand-computed five-point
example, the tie-free formula, the two-level tied example `1 - 2 / n`, the
null variance 2/5 and test level by simulation, and the permutation test;
the rank-one and `A^2` active-subspace oracles, activity scores against
mean squared derivatives, known principal angles, and the closed-form
moment matrix of a model in standard-normal space by autodiff and by
finite differences; `delta` against two-dimensional quadrature and against
its definition on a small sample, PAWN against brute-force empirical
distribution functions with ties, exact invariance under monotone
transformations, and the null level of an inert input; the linear kernel
against the library's Janon estimator and an independent formula, the
Gaussian estimator against a direct double sum, the exact value one for a
function of a single input, and a pure scale effect; input validation
throughout.

## References

[REF-GAMBOA-ETAL-2022], [REF-CHATTERJEE-2021],
[REF-CONSTANTINE-DOW-WANG-2014], [REF-BORGONOVO-2007],
[REF-PLISCHKE-BORGONOVO-SMITH-2013], [REF-PIANOSI-WAGENER-2015],
[REF-DA-VEIGA-2021], [REF-JANON-ETAL-2014].
