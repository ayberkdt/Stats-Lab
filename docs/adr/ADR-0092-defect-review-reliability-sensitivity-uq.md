# ADR-0092: Defect review of reliability, sensitivity, propagation, and UQ

- Status: Accepted
- Date: 2026-11-12

## Context

`probstat.reliability`, `probstat.sensitivity`, `probstat.propagation`,
`probstat.uq`, `probstat.monte_carlo`, `probstat.modeling`,
`probstat.sampling`, `probstat.inverse` and `probstat.validation` were
reviewed for results that are wrong, misleading, or poorly explained. Each
public function was checked against its equation record and source
(conventions such as failure above/below, strict and non-strict thresholds,
estimator variants, the sign of `beta`, the Nataf mapping), at edge cases
(zero variance, `beta <= 0`, `P_f` of zero or one in the sample, perfectly
correlated components, discrete inputs, one input), against problems with
exact answers, and for the calibration of the reported uncertainty by
seeded repetition. A candidate counted as a defect only when an independent
oracle (closed form, quadrature, or the spread of seeded replicates)
demonstrated it.

## Decision

Three defects were fixed and one misleading statement was documented.

1. **Shapley standard errors ignored the estimated variance**
   (`sensitivity.shapley_effects`, [EQ-SENS-SHAPLEY-005]). Every effect is
   normalized by the same estimate `V` of `Var Y`, which also enters
   `c(D)`. The reported errors left the uncertainty of `V` out of the random
   permutation estimator and, for exact enumeration, added it with the
   weight `1/d` of `c(D)` while ignoring the division by `V`; since `V`
   partly cancels, the exact-enumeration error was misstated in either
   direction. Over 200 seeds of `Y = A + B/2` (`Sh = (0.8, 0.2)`,
   `N_V = 30`) the ratio of the root-mean-square reported error to the
   empirical spread was 1.24 (exact) and 0.71 (random); on a three-input
   model with the documented defaults the dominant input had
   root-mean-square `z` of 1.25 and 1.23 over 100 seeds. The delta method
   now gives `Var Sh_i = Var(A_i)/V^2 + (A_i/V)^2 Var(V)/V^2` (exact, with
   `Sh_i = A_i/V + 1/d`) and the permutation spread plus the same term
   (random). The ratios become 0.93 and 0.94, the default-setting `z`
   1.05 and 1.12. Effect estimates are unchanged.
2. **Given-data indices split tied input values across classes**
   (`sensitivity.given_data_first_order`, [EQ-SENS-GIVEN-001]). Classes
   were cut by position with ties in sample order, so a discrete input's
   rows with the same value (and the same `E[Y | X_i]`) landed in different
   classes and the between-class variance lost part of the main effect.
   For `Y = B + 0.5 Z` with `B ~ Bernoulli(1/2)` (`S_B = 1/2`) the mean of
   200 estimates from `n = 1000` was 0.4896 with standard error 0.0015.
   A cut inside a run of ties now moves to the end of the run; the
   correction uses the number `M_i` of classes actually formed, reported as
   `GivenDataSensitivityResult.classes_used`. The mean becomes 0.5006;
   results for inputs without ties are unchanged.
3. **DGSM differentiated discrete inputs** (`sensitivity.dgsm`,
   [EQ-SENS-DGSM-001]). `nu_i = E[(df/dx_i)^2]` does not exist for an
   integer-valued input, yet autodiff or finite differences returned the
   derivative of an arbitrary real extension of the model (`nu_k = 1` for
   `y = k + b`, `k ~ Poisson(3)`), or zero for a model that rounds its
   input. `dgsm` now raises a `ValueError` naming the discrete inputs and
   pointing to `sobol_analysis` and `given_data_first_order`, in line with
   `JointInput.require_continuous` for FORM.
4. **Zero-failure Monte Carlo standard error** (documentation of
   `reliability.ModelMonteCarloResult`). With no failures the plug-in
   standard error is zero and the c.o.v. infinite; the result now says that
   the Wilson interval (upper end about `3.8/N` at 95 %) is the honest
   error statement in that case.

Examined and rejected (behaviour confirmed correct or already documented):

- Subset simulation at conditional levels with tied chain states (38 % of
  conditional levels at the default spread record `p0` while slightly more
  samples satisfy `G <= b_j`): 600 seeded runs each gave
  `E[P_f]/P_f = 0.997 +/- 0.016` (spread 1) and `1.012 +/- 0.018`
  (spread 2) on `beta = 3`, so the estimator is not biased. The empirical
  c.o.v. (0.71 at `beta = 4`) lies between the documented
  `coefficient_of_variation` (0.48) and `coefficient_of_variation_upper`
  (1.04).
- `bivariate_normal_cdf` against a Simpson integral for `|rho|` up to 0.999
  and exactly 1 (maximum error 1.3e-12); Ditlevsen bounds and
  `multi_point_form` with identical components (`rho = 1`) return the
  single-component probability.
- FORM, SORM, inverse FORM, design-point and cross-entropy importance
  sampling, line sampling and subset simulation on a correlated
  linear-Gaussian limit state for failure above and below and for
  `beta < 0`; Der Kiureghian's `gamma` equals the sigma-scaled gradient; the
  autodiff Nataf Jacobian `L' D grad_x g`; SORM curvature sign and the
  Breitung and Hohenbichler-Rackwitz formulas; Genz separation of variables.
- `nataf_correlation` against the closed forms for lognormal and uniform
  pairs (agreement to 1e-10).
- Benchmark references: Ishigami `V13 = 8 b^2 pi^8 / 225`, G-function
  partial variances, the four-branch quadrature in rotated coordinates, the
  lognormal capacity-demand and parabolic limit states, the linear-Gaussian
  Shapley effects and the conjugate calibration posterior.
- Saltelli first-order and Jansen total estimators, the closed second-order
  index, the stochastic-simulator noise corrections, Poincare constants of
  the DGSM bound, the Morris scaled step, and Shapley costs and conditional
  sampling.
- Reliability data: chi-square rate bounds (time and failure terminated),
  Crow-AMSAA shape intervals and unbiasing factors, success-run sample
  size, demonstration time, AGREE and ARINC allocation, the KKT allocation.
- Scoring rules: CRPS of the normal, logistic, lognormal, exponential,
  uniform and ensemble forecasts against quadrature; pinball, interval and
  Dawid-Sebastiani scores; randomized ensemble PIT.
- Copulas: Kendall's tau, tail dependence, CDF against sampling, unit
  density mass, Rosenblatt round trips (Clayton, Gumbel, Frank, Student-t,
  Gaussian).
- Designs: one-dimensional stratification of plain, shifted and
  Owen-scrambled Sobol nets and of LHS; the `(0, m, 2)` property of the
  first two Sobol coordinates.
- Grid convergence (Celik et al.) classification and observed-order
  equation; ASME V&V 20 validation uncertainty; Giles' MLMC bias estimate;
  cross-entropy and mixture importance weights; distribution-free quantile
  ranks.
- The fitted control-variate standard error ignores the estimation of the
  coefficient, an `O(1/n)` effect that was left unchanged.

## Consequences

- Shapley standard errors change for every seed (the effects do not);
  intervals built from them now cover at their nominal rate.
- `given_data_first_order` changes only for inputs with tied values; the
  result gains the field `classes_used` (default empty).
- `dgsm` rejects models with discrete inputs instead of returning an
  undefined measure.
- Defects outside the reviewed packages were not found in passing.

## Verification

- `tests/test_review_reliability_sensitivity_uq.py`: 8 tests. The two
  Shapley calibration tests (200 seeds each) and the DGSM test fail on the
  previous code (ratios 1.24 and 0.71, no error raised); the binary
  given-data bias was measured at seven standard errors before the fix.
- Existing suites of the touched packages pass unchanged
  (`test_model_sensitivity`, `test_sensitivity_estimators`,
  `test_mlmc_ce_line_shapley`, `test_uq5_sensitivity`,
  `test_model_reliability`, `test_uq4_reliability`,
  `test_adaptive_reliability`, `test_benchmarks`, `test_copulas`), as do
  the documentation and architecture contracts, `ruff`, and strict `mypy`.
- Counts: about 60 public functions examined, 4 changes (3 defects, 1
  documentation), 18 candidates examined and rejected.

References: [REF-SONG-NELSON-STAUM-2016], [REF-CASELLA-BERGER-2002],
[REF-PLISCHKE-BORGONOVO-SMITH-2013], [REF-SOBOL-KUCHERENKO-2009],
[REF-AU-BECK-2001], [REF-WILSON-1927].
