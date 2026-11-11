# ADR-0072: Stein control variates, control functionals, and kernel Stein discrepancy

- Status: Accepted
- Date: 2026-11-11

## Context

The UQ-3 roadmap item listed control functionals as the remaining advanced
Monte Carlo method. The existing `estimate_control_variate` (ADR-0010) needs a
control whose mean is known, which is rarely available for a Bayesian
posterior or another density known only up to a constant. Stein's identity
removes that requirement: with the score `u = grad log pi`, the
Langevin-Stein operator `A phi = Laplacian phi + grad phi . u` produces
zero-mean controls from any smooth trial function. Three families follow:
polynomial zero-variance control variates (Assaraf and Caffarel 1999; Mira,
Solgi and Imparato 2013), kernel control functionals (Oates, Girolami and
Chopin 2017), and the kernel Stein discrepancy as a sample-quality diagnostic
(Liu, Lee and Jordan 2016; Gorham and Mackey 2017).

Constraints: standard library only; `monte_carlo` is layer 3 and may use
`linalg`, `foundations`, and `inference`; samples come from independent draws
or Markov chains; every estimator reports its cost.

## Decision

1. **One new module, `monte_carlo.stein`.** It imports `linalg` (Cholesky,
   Householder QR), `foundations.validation`, and
   `inference.ConfidenceInterval` and `inference.monte_carlo_standard_error`.
   No new dependency direction is introduced.
2. **The caller supplies points, the integrand, and the score.** Each callable
   is evaluated once per point, and `integrand_evaluations` and
   `score_evaluations` are reported separately; `evaluations` (the
   `CostedEstimate` field) counts integrand calls. One-dimensional samples
   may be plain numbers and are passed to the callables as numbers.
3. **Zero-variance controls use degree 1 or 2 only** (`d` or
   `d + d(d+1)/2` controls). Coefficients come from least squares with an
   unpenalized intercept through `householder_qr`; `ridge > 0` appends
   `sqrt(ridge) I` rows. A rank-deficient unpenalized design raises instead
   of being regularized silently, in line with the `linalg` contract.
4. **Two ZV-CV estimators.** `"same-sample"` keeps the plug-in labelling of
   ADR-0010 and uses `n - p - 1` residual degrees of freedom.
   `"split-sample"` cross-fits over contiguous folds, which makes each
   contribution exactly unbiased for independent draws and keeps Markov-chain
   folds nearly independent.
5. **Standard errors.** `"iid"` uses `sqrt(s^2/n)`; `"mcmc"` uses the
   effective-sample-size standard error of the contribution series
   (`inference.monte_carlo_standard_error`), scaled by the
   degrees-of-freedom ratio.
6. **Control functionals use a radial base kernel**, Gaussian or IMQ, with the
   closed-form Stein kernel
   `k0 = -4 g'' z - 2 d g' - 2 g' (u_x - u_y) . r + (u_x . u_y) g`, checked in
   the tests against finite differences and against
   `int k0(x, y) pi(x) dx = 0` by quadrature. Two estimators are provided: the
   split-sample estimator (unbiased, with a standard error) and the
   kernel-quadrature weights estimator (the simplified estimator of the
   source), which reports `sqrt(w' K0 w)` instead of a standard error.
7. **Numerical stability.** `A = K0 + lambda mean(diag K0) I` with
   `lambda = 1e-8` by default, raised by powers of ten until `linalg.cholesky`
   succeeds, up to `1e-2`; the applied value is part of the result. The
   source's condition-number rule would need an `O(n^3)` estimate per fit and
   is not used.
8. **Length-scale selection defaults to cross-validation** (five contiguous
   folds, grid median heuristic times `1/4, 1/2, 1, 2, 4`), using only the
   training points of each fit so that the split estimator stays unbiased.
   On `cos` under `N(0, 1)` the median heuristic alone left the error about
   two orders of magnitude larger than the cross-validated choice (`n = 40`:
   weights error `2.0e-2` against `1.5e-4`).
9. **Kernel Stein discrepancy** returns the V-statistic, its square root, and
   the U-statistic, with the IMQ defaults `c = 1`, `beta = -1/2` of Gorham
   and Mackey. `beta` is restricted to `(-1, 0)`, the range of their
   convergence-detection theorem.

## Consequences

- Eleven public names join `monte_carlo`: four result types
  (`ZeroVarianceResult`, `ControlFunctionalResult`,
  `ControlFunctionalWeightsResult`, `KernelSteinDiscrepancy`) and seven
  functions (`estimate_zero_variance`, `estimate_control_functional`,
  `estimate_control_functional_weights`, `kernel_stein_discrepancy`,
  `stein_kernel`, `stein_polynomial_controls`,
  `median_heuristic_length_scale`). The API reference must be regenerated
  when the change is released.
- Equations EQ-MC-STEIN-001 to EQ-MC-STEIN-007 and six bibliography entries
  are added.
- Kernel methods are `O(n^2 d)` in kernel evaluations and `O(n^3)` per
  Cholesky factorization in pure Python; cross-validation multiplies the
  factorizations by about 25. A few hundred points is the practical limit.
- The split-sample standard error ignores the dependence between folds that
  the fitted functions create; for Markov-chain input, unbiasedness holds
  only asymptotically.
- Polynomial and kernel controls require the integration-by-parts tail
  conditions; heavy-tailed targets are outside the validated scope.

## Rejected alternatives

- Solving the normal equations `H'H theta = H'f`: squares the condition
  number, while the existing QR helper avoids it.
- Hidden jitter inside `linalg.cholesky`: contradicts the explicit-failure
  contract of `linalg`; the shift lives in `monte_carlo.stein` and is
  reported.
- Higher-degree polynomial bases: the number of controls grows like `d^q`
  and the least-squares fit needs at least that many points; control
  functionals cover the non-polynomial case.
- Self-normalized or Bayesian-quadrature variance for the weights estimator:
  no source-backed sampling standard error exists for it, so the worst-case
  discrepancy is reported instead.

## Verification

Measured with the seeded cases of `tests/test_stein_control_variates.py`
(20 tests, 8 s):

- Degree-2 ZV-CV, quadratic with a cross term under a correlated bivariate
  Gaussian, `n = 40`: error `-1.8e-15` for both methods; adjusted-to-raw
  variance ratio `1.3e-31` (same-sample) and `7.5e-32` (split-sample).
- `exp(x/2)` under `N(0, 1)`, `n = 200`: variance-reduction factor `6.32`
  and `81.7` for degrees 1 and 2 (same-sample), `6.12` and `64.6`
  (split-sample).
- Unbiasedness, `exp(x)` under `N(0, 1)`, `n = 12`, 400 seeds, degree 2:
  split-sample mean `-0.023` from `e^(1/2)` with standard error `0.013`
  (`-1.8` SE); same-sample `-0.049` with standard error `0.008` (`-6.1` SE).
- `cos(x)` under `N(0, 1)`, 8 seeds, RMSE at `n = 16` and `n = 64`: plain
  Monte Carlo `0.112` and `0.056` (ratio 2); split-sample control functional
  `1.8e-2` and `1.4e-4` (ratio 130); kernel-quadrature weights `3.1e-4` and
  `3.7e-5` (ratio 8.2). The test threshold for the ratio is
  `2 exp(3 sqrt(2)/4) = 5.8`, three standard errors of a log RMSE ratio from
  8 replicates above the Monte Carlo ratio.
- Two-dimensional `cos(x) cos(y) + xy/2`, weights estimator, `n = 60`, four
  seeds: RMSE below one tenth of the plain Monte Carlo standard error.
- AR(1) chain, coefficient `0.8`, `n = 500`, `exp(x)`: MCMC standard error
  `0.039` against IID `0.026`, variance-reduction factor `12.6`.
- IMQ kernel Stein discrepancy, 200 standard-normal draws: `0.122`
  (U-statistic `0.0035`); shifted by one: `0.799` (U-statistic `0.625`).
- Stein kernel: finite-difference agreement to six decimals and
  `|int k0(x, y) pi(x) dx| < 1e-9` for both base kernels at three values
  of `y`.
- Contract tests (documentation, architecture), the variance-reduction
  tests, ruff, and strict mypy pass.
