# ADR-0144: Matrix-free operators and the research-grade inverse-problem layer

- Status: Accepted (experimental APIs)
- Date: 2026-11-26

## Context

ADR-0110 added a dense SVD and classical Tikhonov regularization and
explicitly deferred CGLS/LSQR, total variation, `l1` penalties and the GSVD.
Every inverse routine required an explicit `m x n` matrix, so a field with
`10^4`-`10^5` unknowns could not be represented, and there was no way to
state a weighted (correlated-noise) problem with a non-zero prior centre, to
stop an iteration by the noise level, or to compare parameter-choice rules on
one path. The roadmap of 2026-11-26 (tasks 03-22) asked for a verifiable
layer from small parametric calibration to large function reconstruction.

Constraints carried over: no new runtime dependency; `linalg` at layer 0,
`optimization` at layer 1, `inference` at 2, `inverse` at 4, `design` at 5;
no silent jitter, clipping or pseudo-inverse; the `DenseBackend` protocol of
ADR-0081 must not gain required methods; existing defaults and seeded results
unchanged.

## Decision

1. **Operator protocol in `linalg` (layer 0).** `LinearOperator` is a
   structural protocol with `shape`, `matvec` and `rmatvec` (the EUCLIDEAN
   transpose) [EQ-LINALG-OP-001]. Concrete classes (dense, identity,
   diagonal, callback, selection with scatter-add transpose, permutation,
   first differences with explicit boundaries, the forward-difference image
   gradient, 1D convolution with zero/periodic/reflect boundaries) and the
   algebra (scaled, sum, product, vertical and horizontal stacks, block
   diagonal, transpose) validate every input and output length and
   finiteness [EQ-LINALG-OP-002]-[EQ-LINALG-OP-004]. A weighted adjoint
   `M_X^-1 A' M_Y` is built only by `weighted_adjoint` from SPD metric
   objects [EQ-LINALG-OP-005]; `rmatvec` never carries a hidden metric.
   `CountingOperator` counts forward and transpose applications separately;
   `materialize` is an explicit debug tool with a size guard. The dense
   backend is untouched.
2. **Covariance factors in `linalg`.** Diagonal, dense (Cholesky) and
   block-diagonal factors expose `whiten`, `whiten_transpose`, `unwhiten`,
   `apply`, `solve` and `log_det` [EQ-LINALG-COV-001]; a non-SPD input raises.
3. **Krylov kernels in `linalg`.** CG, CGLS, LSQR and LSMR (from the
   Paige-Saunders and Fong-Saunders recurrences) with explicit stop reasons,
   a final counted true-residual check and a user callback for external
   stopping rules; power iteration (documented as a lower bound), Lanczos
   with full reorthogonalization, Hutchinson traces with standard errors and
   the randomized SVD with a posteriori residuals
   [EQ-LINALG-KRY-001]-[EQ-LINALG-KRY-009].
4. **`inverse.linear_problem`.** `LinearInverseProblem` holds `A`, `y`, the
   noise factor, `L` and an explicit reference `x_ref`; `solve_regularized`
   solves [EQ-INV-LIN-001] by QR (unique solutions only), SVD (explicitly
   minimum-`||x - x_ref||`), LSQR, LSMR or CGLS. The legacy `tikhonov` is the
   special case `W = I`, `x_ref = 0` and is reproduced to rounding (tested);
   it is not changed. Equality constraints, NNLS, box constraints and total
   least squares are baselines with KKT diagnostics
   [EQ-INV-LIN-003]-[EQ-INV-LIN-006]; `linear_estimator_analysis` separates
   the sampling covariance `M Gamma M'` from the Bayesian covariance
   [EQ-INV-LIN-002].
5. **Diagnostics, iteration and selection.** `spectral_diagnostics` uses the
   direct SVD of `W A` with a completed nullspace and Picard coefficients;
   `generalized_svd` is the restricted QR-plus-SVD GSVD
   [EQ-INV-SPEC-001]-[EQ-INV-SPEC-003]. `landweber` requires a certified norm
   bound; `krylov_regularization` stops by the discrepancy principle on the
   data residual; `hybrid_lsqr` regularizes the projected problem
   [EQ-INV-ITER-001]-[EQ-INV-ITER-003]. `regularization_path` evaluates GCV,
   UPRE, the discrepancy gap, the discrete L-curve curvature and the
   normalized evidence on one grid, and `select_regularization` reports
   boundary, unattainable and flat cases instead of fabricating a value
   [EQ-INV-SEL-001]-[EQ-INV-SEL-003].
6. **Non-smooth, nonlinear, Bayesian, PDE and design layers.** Proximal maps,
   ISTA/FISTA with backtracking, PDHG (with a user-certified operator-norm
   bound) and scaled ADMM live in `optimization.proximal` (layer 1); the `l1`
   and total-variation inverse wrappers in `inverse.sparse`, with `alpha`
   multiplying an UNSQUARED penalty. `inverse.gauss_newton` defines the
   `DifferentiableForward` protocol (`value`, `jvp`, `vjp`), derivative
   checks, matrix-free Gauss-Newton with an inner LSQR/CGLS solve and IRGNM.
   `inverse.linear_gaussian` gives the exact conjugate posterior in data and
   whitened-parameter space with evidence and conditional simulation, and the
   low-rank prior-preconditioned posterior. `inverse.pde_models` is a 1D
   elliptic log-conductivity prototype with exact discrete tangent and
   adjoint, and `inverse.heat_models` gives implicit-Euler and
   Crank-Nicolson heat operators whose `rmatvec` is the exact discrete
   transpose. `optimization.trust_region` (Steihaug CG) and
   `inverse.newton_trust_region` add Hessian-action Newton steps;
   `inverse.function_space` adds bases with exact mass matrices and a
   spectral Gaussian prior whose coefficient variances do not depend on the
   truncation level. `inverse.observation_laws` adds normalized Poisson,
   binomial, Student-t, censored and masked Gaussian laws with Fisher
   scoring for linear predictors; `inverse.regularized_ensemble` adds
   ensemble Kalman updates with covariance factors (Woodbury in ensemble
   space, square-root transform) and Tikhonov EKI by the augmented system;
   `inverse.posterior_targets` exposes a prior-whitened posterior target to
   the EXISTING pCN and NUTS samplers and to the existing SBC routine rather
   than adding samplers; `inverse.approximation_error` estimates the mean
   and covariance of the coarse-model error from paired runs (with optional
   shrinkage and a held-out check). `inverse.profile_likelihood` adds profile intervals that never
   fabricate an endpoint; `inference.delayed_acceptance` adds exact-target
   two-stage Metropolis-Hastings; `design.linear_optimal` adds A-, D-, E- and
   goal-oriented criteria with greedy and exhaustive selection.
7. **Benchmarks.** `inverse.benchmark_problems` (L01, L04-L06; L02-L03 are
   the existing closed-form problems) and `inverse.benchmark_extended`
   (L07a, L07b, N01, N02, P01, P02), with the synthetic application cases
   A01 (`inverse.benchmark_orbit`) and A02 (`inverse.benchmark_cosmology`)
   kept out of the solver modules withholds the truth, separates truth,
   design and noise streams and flags inverse crimes
   [EQ-INV-BENCH-001].
8. **Stability.** All new names are experimental (module docstrings say so)
   even inside the stable `linalg` and `optimization` packages; they may
   change before they are promoted.

## Consequences

- A `10^4`-unknown convolution problem runs in `O(n)` memory per
  application (tested at `n = 10 000` with a counting wrapper, and LSQR at
  `n = 4000`); no dense array is created on the matrix-free paths.
- Pure-Python cost is the limiting factor: Krylov iterations at `n = 10^5`
  are feasible but slow; no performance claims are made beyond the measured
  test runs.
- Two solvers that agree with independent oracles may still disagree with
  each other on non-converged runs; every result carries `converged` and a
  stop reason, and `max_iterations` is never reported as convergence.
- Known limitations recorded in `docs/research/inverse-expansion-progress.md`:
  projected GCV under-regularized on deriv2 in our runs; PDHG needs many
  iterations for large TV weights; the GSVD requires `m >= n` and a
  full-rank stack; hybrid LSQR with a general `L` projects the penalty as
  `||L V_k q||`, which costs one application of `L` per step.

## Verification evidence

- Independent oracles in the tests: dense QR or pseudo-inverse solutions for
  CGLS/LSQR/LSMR; the legacy `tikhonov`; normal-equation solves on small
  well-conditioned cases; the KKT system for equality constraints; active-set
  enumeration for NNLS; filter-factor closed forms for Landweber, GCV and
  UPRE; the Gaussian marginal density for the evidence; Monte Carlo
  unbiasedness of UPRE within its standard error.
- Additional one-off cross-checks run during development against NumPy
  2.2.6 and SciPy 1.14.1 (not part of the test suite, which stays
  dependency-free): `numpy.linalg.lstsq`, `scipy.optimize.nnls`,
  `scipy.optimize.lsq_linear`, `scipy.stats.multivariate_normal.logpdf`
  (evidence agreed to `1e-12`), L-BFGS-B on the split lasso and SLSQP on the
  epigraph form of 1D TV (objectives agreed to `~1e-9` relative).

References: [REF-PAIGE-SAUNDERS-1982], [REF-FONG-SAUNDERS-2011],
[REF-GAZZOLA-HANSEN-NAGY-2019], [REF-HANSEN-2008], [REF-STUART-2010],
[REF-BECK-TEBOULLE-2009], [REF-CHAMBOLLE-POCK-2011], [REF-BOYD-ETAL-2011],
[REF-SPANTINI-ETAL-2015], [REF-ALEXANDERIAN-2021], [REF-RAUE-ETAL-2009],
[REF-CHRISTEN-FOX-2005], [REF-STEIHAUG-1983].
