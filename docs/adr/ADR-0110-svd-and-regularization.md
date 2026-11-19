# ADR-0110: Singular value decomposition and classical regularization

- Status: Accepted
- Date: 2026-11-19

## Context

The linear-algebra layer offered Cholesky, LU, Householder QR, and the
symmetric Jacobi eigen-decomposition, but no singular value decomposition,
pseudo-inverse, or numerical rank. The inverse-problems layer covered the
Bayesian route (conjugate Gaussian calibration, Laplace, MCMC, model
discrepancy) and recorded deterministic regularization as missing: there
was no way to solve an ill-posed linear system `A x = b` stably, to inspect
the discrete Picard condition, or to choose a regularization parameter
without a prior. Constraints:

- no new dependency, and `linalg` stays at layer 0, `inverse` at layer 4;
- no silent jitter or repair: rank decisions and penalties are explicit;
- no change to any existing default or seeded result.

## Decision

1. **`linalg.svd` (layer 0).** `singular_value_decomposition(matrix)`
   computes the thin SVD by one-sided (Hestenes) Jacobi rotations
   [EQ-LINALG-SVD-001]. A tall matrix is first reduced to the triangular
   factor of `householder_qr`, so the rotations act on `n x n`; a wide
   matrix is decomposed through its transpose. Singular values are returned
   in descending order (unlike `symmetric_eigen`, which is ascending), an
   exactly zero singular value is kept as `0.0` with an orthonormal
   completing left singular vector, and non-convergence within
   `max_sweeps` raises. `pseudo_inverse(matrix, *, rcond=None)` and
   `numerical_rank(matrix, *, tolerance=None)` use a cut-off relative to
   `sigma_1`, default `max(m, n) 2^-52` [EQ-LINALG-SVD-002]. The result
   class `SingularValueDecomposition` offers `rank`, `pseudo_inverse`,
   `condition_number`, and `reconstruct`.
2. **Not part of the dense backend.** The `DenseBackend` protocol of
   ADR-0081 covers the `n^3` kernels of the sample-size consumers (GP and
   multivariate normal); `householder_qr` and `symmetric_eigen` are direct
   kernels outside it. The SVD follows the same pattern. Adding a method to
   the protocol would break every user-supplied backend; a backend SVD can
   be added later without changing the public function.
3. **`inverse.regularization` (layer 4).**
   - `tikhonov(operator, data, regularization, *, penalty=None,
     decomposition=None)` minimizes `||A x - b||^2 + lambda^2 ||L x||^2` by
     Householder QR of `[A; lambda L]` [EQ-INV-REG-001]; the normal
     equations are never formed. For `L = I` it reports the filter factors
     `sigma^2 / (sigma^2 + lambda^2)` [EQ-INV-REG-002] from an SVD of `A`
     (computed, or reused through `decomposition`). A numerically rank
     deficient stack raises `ValueError`.
   - `truncated_svd` [EQ-INV-REG-006] and `picard_coefficients`
     [EQ-INV-REG-007].
   - `l_curve` (maximum curvature of the log-log curve, closed form derived
     from eq. (8) of Hansen and O'Leary with `rho' = -lambda^2 eta'`)
     [EQ-INV-REG-004], `generalized_cross_validation` [EQ-INV-REG-003],
     and `discrepancy_principle` (Brent root in `log lambda`)
     [EQ-INV-REG-005] return a `RegularizationParameterChoice` with the
     chosen `lambda`, the evaluated grid and criterion, the residual and
     solution norms, the solution, a convergence flag, and `at_boundary`.
     For `L = I` the curves come from one SVD in `O(k)` per `lambda`; for a
     general `L` every evaluation is one QR of the stacked matrix, which
     also yields the GCV trace `m - ||Q_1||_F^2` and the L-curve derivative
     `eta'` from a second solve with the same factors.
   - `nonlinear_tikhonov(residual, x0, regularization, *, prior_mean=None,
     penalty=None, jacobian=None, ...)` runs Levenberg-Marquardt on the
     stacked residual `[r(x); lambda L (x - x_bar)]`; the penalty center
     defaults to `x0`.
4. **`inverse.test_problems`.** `backward_heat_problem` (closed-form SVD
   from the sine eigenvectors of the second-difference matrix,
   [EQ-INV-REG-PROB-001]) and `second_derivative_problem` (the `deriv2`
   kernel of Regularization Tools discretized by the midpoint rule,
   [EQ-INV-REG-PROB-002]) return `A`, the exact solution, and `b = A x`;
   noise is the caller's, with its own seed.
5. **Scope.** Total variation and `l1` penalties need non-smooth
   optimization and are out of scope, as are the generalized SVD and
   iterative regularization (CGLS, LSQR).

Public names added to `probstat.linalg`: `SingularValueDecomposition`,
`singular_value_decomposition`, `pseudo_inverse`, `numerical_rank`,
`DEFAULT_JACOBI_TOLERANCE`. Added to `probstat.inverse`: `tikhonov`,
`TikhonovSolution`, `truncated_svd`, `TruncatedSVDSolution`,
`picard_coefficients`, `PicardCoefficients`, `l_curve`,
`generalized_cross_validation`, `discrepancy_principle`,
`RegularizationParameterChoice`, `nonlinear_tikhonov`,
`NonlinearTikhonovResult`, `backward_heat_problem`,
`second_derivative_problem`, `DiscreteIllPosedProblem`.

## Consequences

- No existing default or seeded number changes; the new modules are only
  imported by the package `__init__` files.
- `tests/test_linalg_svd.py` checks reconstruction, orthonormality, the
  agreement of `sigma^2` with `symmetric_eigen(A'A)`, singular values and
  vectors (up to joint sign) against R 4.6.2 `svd()` for a tall, a wide, a
  rank-deficient, and the `6 x 6` Hilbert matrix (all to `1e-13`
  relative, the Hilbert values to `1e-9`), the exact rational
  pseudo-inverse of the rank-two `3 x 3` matrix `1..9`, the Penrose
  conditions, and the zero matrix.
- `tests/test_regularization.py` checks the closed-form SVD of the heat
  problem (to `1e-14`), the `O(h^2)` convergence of the `deriv2` data, the
  identity of Tikhonov with the Gaussian posterior mean (normal equations
  solved by Cholesky, `1e-10` relative), the filter-factor form, a
  difference penalty against its normal equations, the TSVD limit across a
  singular-value gap, the curvature formula and `eta'` against finite
  differences for both `L = I` and a difference penalty, the QR path with
  an explicit identity penalty against the SVD path, the discrepancy root
  to `1e-9` relative, and parameter choices within a factor 4 of the
  error-optimal `lambda` on seeded `deriv2` data.
- On the backward heat problem (`n = 32`, 1% noise, seeds 1-6) the
  discrepancy principle chose 0.62-0.94 times the error-optimal `lambda`,
  GCV 0.06-0.33, and the L-curve 0.06-0.10. The guide reports these
  numbers; they are a property of the rules on this problem, not a defect
  of the implementation.
- Pure-Python cost: the SVD of a random `64 x 64` matrix took 1.7 s, and a
  general-`L` parameter choice repeats one QR per grid point.

## Rejected alternatives

- **Golub-Kahan bidiagonalization with implicit QR.** Faster for large
  matrices, but longer to implement and verify; one-sided Jacobi is short,
  accurate for graded matrices, and fits the existing Jacobi eigen-solver.
- **SVD through `A'A` and `symmetric_eigen`.** Squares the condition number
  and loses the small singular values that regularization is about.
- **Normal equations for Tikhonov.** Same squaring of the condition number;
  the stacked QR costs about the same.
- **Generalized SVD for a general `L`.** It would give all curves in
  `O(k)` per `lambda` but needs a second decomposition algorithm; repeated
  QR keeps the code small and is exact for the sizes this library targets.
- **Silent clipping of tiny singular values or automatic jitter.** Rank
  decisions are explicit through `rcond`/`tolerance` and refusals.

## References

[REF-HESTENES-1958], [REF-GOLUB-VANLOAN-2013, Sections 5.5 and 8.5-8.6],
[REF-PENROSE-1955], [REF-TIKHONOV-1963], [REF-MOROZOV-1966],
[REF-CRAVEN-WAHBA-1979], [REF-GOLUB-HEATH-WAHBA-1979],
[REF-HANSEN-OLEARY-1993], [REF-HANSEN-2008], [REF-TARANTOLA-2005].
