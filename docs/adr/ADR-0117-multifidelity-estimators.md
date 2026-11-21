# ADR-0117 — Multifidelity Monte Carlo, approximate control variates, and MLBLUE

- Status: Accepted
- Date: 2026-11-21
- Extends: the multilevel Monte Carlo module of `monte_carlo`

## Context

`monte_carlo` had multilevel Monte Carlo for hierarchies whose corrections
shrink geometrically, and single-sample control variates with a known control
mean. Neither covers the common situation of a few heterogeneous cheap models
(coarse meshes, reduced-order models, data fits) that are correlated with an
expensive model but have unknown means and no convergence rates. Three
families of estimators answer it, all unbiased for the high-fidelity mean and
all planned from the model covariance matrix and the evaluation costs:
multifidelity Monte Carlo (Peherstorfer, Willcox and Gunzburger 2016),
approximate control variates (Gorodetsky, Geraci, Eldred and Jakeman 2020),
and multilevel best linear unbiased estimators (Schaden and Ullmann 2020).
The reference allocation method for MLBLUE is a semidefinite program (Croci,
Willcox and Wright 2023); the library is standard-library only and has no SDP
solver.

## Decision

1. **Three new modules in `monte_carlo`.** `multifidelity.py` (shared
   types, pilot statistics, MFMC), `approximate_control_variates.py` (ACV),
   and `multilevel_blue.py` (MLBLUE, the comparison table). They import
   `foundations`, `linalg`, and `optimization` only; the layer map is
   unchanged.
2. **One calling convention.** Models are callables `f_i(input) -> float`
   with the high-fidelity model at index 0; `input_sampler(n, stream)`
   returns `n` independent inputs from a `RandomStream`; costs are per
   evaluation. The covariance is an explicit argument of every planning
   function, so a known covariance and a pilot covariance are used the same
   way.
3. **Pilot independence is structural.** `pilot_statistics` draws from the
   child stream `"pilot"`; the estimators draw from `"mfmc"`, `"acv-mf"`,
   `("acv-is", ...)`, and `("mlblue-group", k)`. One seed for pilot and
   estimator therefore gives independent samples. Docstrings state that
   unbiasedness holds for fixed weights and that reported variances are
   predictions from the supplied covariance ([EQ-MC-MF-001]).
4. **MFMC** ([EQ-MC-MF-002] to [EQ-MC-MF-004]). `mfmc_variance` implements
   the variance for any nested sample sizes and any fixed coefficients;
   `mfmc_allocation` the closed-form optimum, refusing model sets that
   violate the ordering or the cost-correlation condition;
   `select_mfmc_models` an exhaustive search over the subsets that contain
   model 0 (at most 12 models). Integer sample sizes are floors, so the
   integer cost never exceeds the budget.
5. **ACV** ([EQ-MC-MF-005] to [EQ-MC-MF-008]). `acv_design` computes the
   optimal weights and the variance of an ACV-IS or ACV-MF design with
   integer sample counts (a count of 0 removes a model). The ACV-MF factor
   `F_ij = (min(r_i, r_j) - 1) / min(r_i, r_j)` was re-derived from
   `Cov(mean_m f_i, mean_n f_j) = C_ij / max(m, n)` and confirmed by exact
   covariances of sample means and by replicate simulation.
   `acv_allocation` is a heuristic: budget shares through a softmax,
   Nelder-Mead from several starts (the ACV-MF objective has kinks where two
   ratios cross, which stalled the BFGS line search in trials), every
   control subset, rounding down.
6. **MLBLUE** ([EQ-MC-MF-009], [EQ-MC-MF-010]). `mlblue_variance` and
   `mlblue_estimate` assemble `Psi` on the models covered by sampled groups.
   `mlblue_allocation` solves the continuous relaxation as a smooth
   nonlinear program: shares `theta_k = (1 - eps) z_k^2 / |z|^2 + eps / K`,
   objective `log [Phi(theta)^-1]_00`, analytic gradient, `minimize_bfgs`.
   The solution is checked with the lower bound `g^2 / max_k d_k`
   (Cauchy-Schwarz applied to the generalized-least-squares form of the
   variance); `converged` is true only if the relative gap is at most the
   tolerance, and a restart from shares boosted on the groups with
   `d_k > g` follows otherwise. The floor `eps` (a quarter of the tolerance)
   keeps `Phi` positive definite, which makes the sensitivities `d_k`
   unique at solutions on the boundary of the simplex; without it the
   certificate failed in 3 of 30 random problems although the primal
   solution was optimal. Rounding takes floors and spends the released
   budget greedily; it is a heuristic.
7. **Comparison.** `compare_multifidelity_estimators` returns one row per
   method (MC, MFMC with the best admissible subset, ACV-MF, MLBLUE over all
   groups up to a size limit) with the integer-allocation variance, the
   continuous variance, the ratio to plain Monte Carlo, the cost, the
   evaluations per model, and the convergence flag ([EQ-MC-MF-011]).

## Consequences

- Reference problem (standard deviations `(1, 1.2, 0.8, 2)`, correlations
  with the high-fidelity model `0.95, 0.85, 0.6`, costs `1, 0.1, 0.01,
  0.001`, budget 100): variance ratios to plain Monte Carlo are 0.2775
  (MFMC, samples `(59, 255, 1145, 3609)`), 0.3304 (ACV-MF, `(59, 262, 1410,
  262)`), and 0.2573 (MLBLUE, groups `{3}: 2557`, `{2,3}: 962`,
  `{1,2,3}: 202`, `{0,1,2,3}: 58`); the continuous MLBLUE optimum is
  certified to a relative gap of `1.8e-9` after 109 BFGS iterations. The
  table takes 0.2 s.
- ACV-MF is not uniformly better than MFMC. With the first three models of
  the reference problem the best ACV-MF design uses one control (ratio
  0.375) against 0.285 for MFMC and 0.282 for MLBLUE; a grid search over
  both ratios confirmed that the two-control ACV-MF optimum (0.378) is
  worse than the one-control design.
- Replicate simulations with 400 seeded replicates of Gaussian models give
  sample-to-predicted variance ratios of 1.004 (MFMC), 0.994 (ACV-MF), 1.019
  (ACV-IS), and 0.934 (MLBLUE) against a band of `+/- 0.319`, and means
  within 1.5 predicted standard errors of the truth.
- `mlblue_allocation` costs about 0.1 s for 15 groups, 1.2 s for 31, and
  7.5 s for 63 in pure Python; `max_group_size` or an explicit group list
  limits the work.
- Only the mean of a scalar output is estimated. Multi-output MLBLUE, ACV
  recursive-difference variants, and estimation of variances or quantiles
  are not implemented.

## Rejected alternatives

- **Softmax shares for MLBLUE.** Optimal allocations are sparse; with a
  softmax the optimum lies at infinity. The squared parametrization reaches
  the boundary at finite `z`.
- **Declaring convergence from the BFGS flag.** A coordinate that starts at
  zero has a zero gradient in the squared parametrization, so a stationary
  point need not be optimal; the lower bound detects this case.
- **A hand-written interior-point SDP solver.** Far more code than the
  problem needs: the relaxation is a smooth convex program on a simplex with
  a cheap exact gradient and a cheap optimality bound.
- **BFGS for the ACV allocation.** The line search stalled on the kinks of
  the ACV-MF objective and reported non-convergence at points that
  Nelder-Mead then improved.
- **Estimating the covariance inside the estimators from the estimation
  samples.** It makes the weights depend on the samples and the estimators
  biased; the covariance stays an explicit, separately estimated input.
- **Reporting an in-sample variance estimate.** The estimators combine
  nested and grouped means; an honest sample variance would need the same
  covariance model, so the predicted variance is reported and named as such.

## Verification

`tests/test_multifidelity.py` (42 tests, about 4 s). Oracles: jointly
Gaussian models `Y = mu + L z` with a known covariance; an independent exact
variance calculator for linear combinations of sample means over index sets
(`Cov = C_ij |A & B| / (|A| |B|)`), which reproduces the MFMC and both ACV
variances to 11 digits; closed forms (two-model control-variate limit
`(1 - rho^2) s^2 / n`, hand-computed MFMC ratios, the closed-form MFMC
variance against the general formula); random feasible perturbations of the
MFMC optimum and of the ACV weights; brute-force model selection; a
brute-force stacked generalized least squares solve (Gauss-Jordan in the
test) for the MLBLUE variance and estimate to 10 digits; MLBLUE on nested
groups equal to MFMC for two models and for an AR(1) correlation and no
larger otherwise; MLBLUE on telescoping groups no larger than the MLMC
variance and equal in a constructed case; MLBLUE equal to ACV-IS and
smaller than ACV-MF on their sample structures; the certified MLBLUE
optimum between its lower bound and the MFMC optimum on nine problems and
equal to the MFMC closed form for two models; 300 random feasible
allocations above the lower bound; replicate simulations for unbiasedness
and variance, including fixed wrong weights; the non-convergence flag; and
input validation.

## References

[REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016],
[REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020], [REF-SCHADEN-ULLMANN-2020],
[REF-CROCI-WILLCOX-WRIGHT-2023], [REF-GILES-2015], [REF-NELDER-MEAD-1965].
