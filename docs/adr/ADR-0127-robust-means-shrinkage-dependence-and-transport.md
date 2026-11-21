# ADR-0127 — Robust means, shrinkage covariance, dependence measures, and sliced and unbalanced transport

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0042 (scoring, calibration, and transport) and the
  descriptive-statistics conventions of `statistics`

## Context

`statistics` (layer 1) offered the classical summaries: mean, trimmed mean,
covariance, Pearson and Spearman correlation. Three gaps showed up in
downstream packages and in the guides:

- the sample mean of heavy-tailed data has deviations that grow like
  `1/sqrt(delta)`; no estimator with a logarithmic dependence on the
  confidence level was available;
- the sample covariance of `p` variables from `n ~ p` observations is
  ill-conditioned or singular, and every consumer that factorizes it has to
  add its own jitter;
- no dependence measure detected non-monotone relations.

`information` (layer 2) had exact and entropic *balanced* transport and
one-dimensional quantile distances for laws with a quantile function, but
nothing cheap for point clouds in several dimensions and nothing for
measures of different mass.

## Decision

1. **`statistics/robust_mean.py`.** `median_of_means` ([EQ-STAT-ROB-001])
   with a seeded random or a sequential block assignment, blocks of size
   `floor(n/k)` or `ceil(n/k)`, and `median_of_means_block_count(delta) =
   ceil(8 log(1/delta))`. The deviation bound that the library prints is
   derived, not quoted ([EQ-STAT-ROB-002]): Chebyshev in each block and
   Hoeffding for the number of failing blocks give
   `P(|mu_hat - mu| > 2 sigma/sqrt(floor(n/k))) <= exp(-k/8)`.
   `catoni_mean` ([EQ-STAT-ROB-003]) solves the monotone estimating
   equation with `optimization.brent_root` on the sample range (the
   `statistics -> optimization` import already exists for the bandwidth
   selectors). Its default scale `sqrt(2 log(1/delta)/(n v))`
   ([EQ-STAT-ROB-004]) is documented as a choice of the order analysed by
   Catoni; no numerical deviation bound is printed for it. Without a
   variance bound the sample variance is plugged in and the result says so
   (`variance_estimated=True`).
2. **`statistics/shrinkage.py`.** `ledoit_wolf_covariance`
   ([EQ-STAT-LW-001]) with divisor `n`. `center=True` (default) subtracts
   the column means; `center=False` is the mean-zero setting of the paper.
   `bbar^2` is evaluated through `sum_k |x_k|^4 - n tr(S^2)`, which the
   tests compare with the literal sum of squared norms. For `d^2 = 0` the
   intensity is 0.
3. **`statistics/dependence.py`.** `chatterjee_correlation`
   ([EQ-STAT-DEP-001], [EQ-STAT-DEP-002]) for data without ties in `y`:
   ties in `x` are broken with a seeded stream, ties in `y` raise
   `ValueError`. `distance_correlation` ([EQ-STAT-DEP-003],
   [EQ-STAT-DEP-004]) for scalar or vector observations with an optional
   seeded permutation test on `dCov_n^2`, and
   `distance_correlation_bivariate_normal` ([EQ-STAT-DEP-005]), re-derived
   from `E|UV|` of a bivariate normal pair. `sensitivity` gains its own
   Chatterjee coefficient in a parallel package for given-data sensitivity
   indices; `statistics` cannot import layer 5 and layer 5 keeps its own
   tie handling, so the two are intentionally independent.
4. **`information/sliced.py`.** `empirical_wasserstein_1d` evaluates the
   quantile coupling between discrete laws exactly by sorting and merging
   cumulative weights ([EQ-OT-SW-001]); `sliced_wasserstein`
   ([EQ-OT-SW-002]) averages `W_p^p` over seeded uniform directions and
   reports the standard error over directions and a delta-method error for
   the distance. In one dimension the result is exact and uses no random
   number.
5. **`information/unbalanced.py`.** `unbalanced_sinkhorn` minimizes
   `<C, P> + eps KL(P | a b') + rho KL(P1 | a) + rho KL(P'1 | b)`
   ([EQ-OT-UB-001]) in the log domain. The update `f = lambda T(g)`,
   `g = lambda T(f)`, `lambda = rho/(rho + eps)` is derived from the
   first-order conditions ([EQ-OT-UB-002]). The reference measure `a b'`
   is that of `information.sinkhorn`, so the balanced limit is the
   library's Sinkhorn plan and zero-weight atoms need no special case; with
   the counting measure the same derivation yields the
   `u = (a/(Kv))^lambda` form. Each iteration ends, by default, with the
   closed-form maximization of the dual along `(f + c, g - c)`
   ([EQ-OT-UB-003]). The result carries the plan, both marginals, the
   transported mass, the primal and dual objectives, the potentials, the
   trace of potential changes, and the exact fixed-point and first-order
   residuals of the returned potentials.

## Consequences

- Heavy tails (`n = 200`, `delta = 0.01`, 1500 seeded replicates): the 99th
  percentile of the absolute error is 0.430 (sample mean), 0.363
  (median-of-means), 0.361 (Catoni with the true variance) for Student-t
  with 2.5 degrees of freedom, and 0.343, 0.265, 0.198 for Pareto with
  shape 2.5. With the plug-in sample variance Catoni gives 0.388 and 0.265.
- The derived median-of-means bound is loose: for Pareto data with
  `n = 120`, `k = 8` no replicate out of 1500 violates it, against the
  bound `exp(-1) = 0.37`.
- Ledoit-Wolf, `p = 20`, `n = 25`, variances 1 to 4: mean squared Frobenius
  loss 101.7 for `S` and 16.2 for the shrunk matrix over 40 replicates
  (mean intensity 0.81); intensity 1.0 for an identity truth at the same
  sizes.
- Chatterjee null (`n = 300`, 1500 replicates): mean of `sqrt(n) xi_n`
  `-0.005`, variance 0.408 against 2/5.
- Distance correlation of simulated normal pairs (`n = 500`): 0.296, 0.560,
  0.883 against the closed form 0.269, 0.550, 0.868 at `rho = 0.3, 0.6,
  0.9`; 0.079 at `rho = 0` (the upward bias of the V-statistic).
- Sliced distance of a 50-point cloud in `R^3` translated by `(3, 0, -3)`:
  `SW_2^2 = 6.011 +- 0.247` with 500 directions against `|t|^2/d = 6`.
- Unbalanced Sinkhorn, `m = 5`, `n = 6`, `eps = 0.2`: duality gap below
  `1e-13` for `rho` from 0.1 to 1000; with the translation step 9, 43, 284,
  166 iterations at `rho = 0.1, 1, 10, 1000`, against 9, 52, 478 and no
  convergence in 20 000 without it. Distance of the plan from the balanced
  Sinkhorn plan (equal masses): `4.5e-3`, `4.7e-5`, `4.7e-7` at
  `rho = 1e2, 1e4, 1e6`.
- `distance_correlation` needs `O(n^2)` memory; the permutation test costs
  `O(R n^2)` in pure Python.
- For `rho/eps` large the first-order residual is `(1 + rho/eps)` times the
  fixed-point residual, so it stays visible (about `4e-6` at
  `rho/eps = 5e6`) when the potential change is below the tolerance. Both
  are reported.

## Rejected alternatives

- **Printing a numerical deviation bound for Catoni's estimator.** The
  constants and the admissible range of `delta` could not be checked
  against the source in this work package; only the order is stated.
- **A tie-aware Chatterjee coefficient.** The general formula reduces to
  the implemented one without ties, but the null variance `2/5` does not
  hold with ties, so the test would need a separate variance estimate.
  Rejected inputs are clearer than a p-value with the wrong variance.
- **The unbiased (U-statistic) distance covariance.** It can be negative
  and changes the meaning of `dCor`; the original estimator with a
  permutation test is the documented default of the source.
- **Calling `quantile_wasserstein` per projection.** It integrates step
  functions by quadrature: slow, and accurate only to about `3e-4` on a
  six-point example. Sorting is exact; the tests compare the two.
- **Counting-measure entropy in the unbalanced problem.** It would make
  the objective disagree with `information.sinkhorn` and requires special
  cases for zero weights.
- **Stopping on the first-order residual.** Its rounding floor grows with
  `rho`, so large penalties would never be reported as converged.

## Verification

`tests/test_robust_dependence.py` (38 tests, about 5 s): limiting cases
`k = 1`, `k = n`, `a -> 0`; hand-computed blocks; translation equivariance;
the derived bound by simulation; 99th-percentile errors on heavy tails; the
Ledoit-Wolf hand example, the literal definition, positive definiteness for
`p > n`, intensity limits, and Frobenius risk; the exact monotone value,
null moments and level of Chatterjee's coefficient; distance-correlation
invariances, the linear case, the literal definition, the closed form
against quadrature and simulation, and the level of the permutation test.

`tests/test_transport_extensions.py` (21 tests, under 1 s): exact 1-D
distance against the transportation simplex and the quantile quadrature;
sliced distance in one dimension, the translation identity within the
standard error and per direction, symmetry, the bound by `W_2`, weights;
unbalanced Sinkhorn against a one-atom closed form, the first-order
conditions, literal evaluation of the objective, the duality gap, random
perturbations of the plan, the balanced limit, the scaling form of the
update, zero weights with small `eps`, and input validation.

## References

[REF-LUGOSI-MENDELSON-2019], [REF-CATONI-2012], [REF-LEDOIT-WOLF-2004],
[REF-CHATTERJEE-2021], [REF-SZEKELY-RIZZO-BAKIROV-2007],
[REF-BONNEEL-ETAL-2015], [REF-CHIZAT-ETAL-2018], [REF-PEYRE-CUTURI-2019].
