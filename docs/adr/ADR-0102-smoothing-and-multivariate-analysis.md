# ADR-0102: Smoothing, additive models, and multivariate analysis

- Status: Accepted
- Date: 2026-11-12

## Context

The library fitted parametric regressions (ADR-0030), Gaussian mixtures with
k-means++ seeding (ADR-0040), and kernel density estimates, but had no
nonparametric regression, no additive models, and none of the classical
multivariate methods: principal components, factor analysis, discriminant
analysis, and clustering with a quality measure. The pieces they need
already existed: dense Cholesky, triangular solves, and the Jacobi
`symmetric_eigen` in `linalg`, the `DenseBackend` boundary (ADR-0081), BFGS
and Nelder-Mead in `optimization`, the Silverman bandwidth `kde_bandwidth`
in `statistics`, and `kmeans_plus_plus` in `inference.mixtures`.

Constraints: standard library only at runtime, strict mypy, the layer map of
`docs/architecture.md`, documented and cited definitions, and executable
guides whose quoted numbers are checked (ADR-0095).

## Decision

1. **Placement.** All new code lives in `probstat.inference` (layer 2), in
   new modules: `splines`, `smoothing`, `additive_models`,
   `multivariate_analysis`, `discriminant`, `clustering`, and the private
   engine `_penalized`. `inference` owns estimation and regression;
   `multivariate` owns vector-valued distributions and must not own
   inference workflows, and placing k-means there would have required a
   sibling import of `inference.mixtures` for the k-means++ seeding. No
   package boundary or layer changes.
2. **One penalized least-squares engine.** `_penalized.PenalizedProblem`
   solves `(X'WX + sum_j lambda_j S_j) beta = X'Wy` by Cholesky and
   evaluates the effective degrees of freedom, GCV, and the REML criterion
   with `phi` profiled out and the pseudo-determinant of the penalty
   reduced to `sum_j r_j log lambda_j` (its constant part is dropped).
   P-splines (`fit_pspline`) and additive models (`fit_additive_model`) use
   it. One smoothing parameter is chosen by a 37-point grid over 36
   natural-log units centred on `tr(X'WX)/tr(S)`, then golden-section
   search; several by a common-offset grid followed by bounded Nelder-Mead
   in `log lambda`. REML is the default, as Wood recommends; GCV is
   available.
3. **Smoothing splines in `O(n)`.** `fit_smoothing_spline` uses the Reinsch
   form with Green and Silverman's `Q` and `R`, a banded `U'DU`
   factorization, and the Hutchinson-de Hoog recursion for the central band
   of the inverse, which gives the trace, the leverages, and the pointwise
   variances. A dense `n x n` solve would cost seconds per smoothing
   parameter at `n = 200` in pure Python. `lambda` follows the convention of
   `scipy.interpolate.make_smoothing_spline`. Ties in `x` are rejected
   rather than silently merged.
4. **Bands.** Pointwise bands are Bayesian: `phi_hat (X'WX + S_lambda)^-1`
   with `phi_hat = RSS / (n - edf)` (Wahba 1983, Nychka 1988). They
   condition on the selected `lambda`. The smoothing-spline band is given
   at the knots only.
5. **No extrapolation of B-splines.** Evaluating a B-spline basis, a
   P-spline, or an additive-model term outside its knot domain raises
   `ValueError`. Natural smoothing splines continue linearly, as the
   definition implies, and the guide shows the consequence.
6. **Additive models** use P-spline terms with the sum-to-zero constraint
   absorbed by a Householder reflection (so the intercept is identified and
   the penalty ranks are unchanged), one smoothing parameter per term, and
   per-term EDF and bands. `backfit_additive_model` implements backfitting
   for fixed smoothing parameters; its reported EDF and bands are those of
   the equivalent joint system. `partial_dependence` accepts any prediction
   function.
7. **LOESS, isotonic, and kernel regression.** `loess` follows Cleveland's
   definitions exactly (tricube weight zero at the `q`-th neighbour,
   bisquare at `6 median|e|`, degree reduced only when the local system is
   singular). `isotonic_regression` is a one-pass weighted PAVA.
   `kernel_regression` offers Nadaraya-Watson and local linear with Gaussian
   or Epanechnikov kernels; the bandwidth minimizes the leave-one-out score
   computed by explicit exclusion, over multiples of `kde_bandwidth`.
8. **Multivariate analysis.** `principal_components` calls
   `linalg.get_backend().symmetric_eigen`, so the NumPy backend applies; it
   offers covariance or correlation scale and signs each axis so its largest
   entry is positive. `factor_analysis` works on the correlation matrix,
   minimizes the concentrated discrepancy over uniquenesses in
   `[0.005, 1]` by bounded BFGS with the analytic gradient, rotates by
   varimax (Kaiser's planar rotations with Kaiser normalization, no SVD
   needed), orders and signs the columns, and reports Bartlett's test.
   `fit_linear_discriminant` returns posteriors and Fisher's canonical
   directions from the whitened between-class matrix;
   `fit_quadratic_discriminant` uses unbiased class covariances. `kmeans`
   runs Lloyd's iteration from `kmeans_plus_plus` seeds (reused, not
   reimplemented) with restarts; empty clusters are re-seeded at the point
   with the largest contribution to the inertia.
9. **No iris data.** Fisher's iris table is public domain, but no copy was
   verified row by row against the 1936 paper, so it is not bundled; the
   guides use seeded simulations and the tests use closed forms.

## Consequences

- 36 public names are added to `probstat.inference.__all__`, with 25
  equation records (`EQ-SMOOTH-001` to `-013`, `EQ-GAM-001` to `-003`,
  `EQ-MVA-001` to `-009`) and 27 bibliography entries, all DOIs checked
  through Crossref except Lawley and Maxwell (1971), which has none and was
  checked against its library catalogue record.
- Two guides, `docs/foundations/smoothing-and-additive-models.md` and
  `docs/foundations/multivariate-analysis.md`, are steps S19 and S20 of the
  statistics track. They have no laboratory; their numbers are asserted by
  the documentation-example test and the two test modules.
- Costs measured on the development machine (pure Python): a REML
  smoothing spline takes 0.34 s at `n = 200` and 2.1 s at `n = 1000`; a REML
  P-spline with 20 segments 0.39 s and 1.1 s; a two-term REML additive
  model at `n = 200` 0.9 s; kernel regression with the 25-candidate
  leave-one-out search 1.4 s at `n = 200` (quadratic in `n`).
- The generated API and mathematical references and the README counts of
  public names and equations are regenerated by the integrator.

## Verification

`tests/test_smoothing_and_additive_models.py` (32 tests) and
`tests/test_multivariate_analysis.py` (18 tests):

| Method | Oracle |
|---|---|
| B-spline basis | `scipy.interpolate.BSpline.design_matrix` (SciPy 1.14.1) to 14 places; partition of unity; Marsden's identity for cubics |
| P-spline at fixed `lambda` | NumPy 2.2.6 dense solve: EDF, GCV, fitted value, and band standard error to `1e-9` |
| P-spline limits | `lambda = 0` gives EDF = number of basis functions; large `lambda` gives the least-squares line (`fit_simple_linear_regression`); quadratics reproduced with `d = 3` |
| REML (P-spline and smoothing spline) | differences of the exact restricted log-likelihood of the equivalent mixed model, computed with dense algebra, to `1e-9` and `1e-7`; the selected `lambda` maximizes it |
| Smoothing spline | `make_smoothing_spline(x, y, lam)` values to `1e-9`; brute-force hat matrix for EDF and leverages; GCV minimizer from a brute-force GCV and `scipy.optimize.minimize_scalar`; straight lines reproduced and EDF limits `n` and 2 |
| Bands | seeded simulation (40 replicates): average coverage at least `0.95 - 3 SE` (observed about 0.97) |
| LOESS | lines and quadratics reproduced; one local fit equals weighted least squares by `fit_linear_model` with hand-computed tricube weights; robust iterations recover a line with a gross outlier exactly |
| Isotonic regression | min-max formula on 200 seeded weighted cases; hand case; exhaustive grid of monotone sequences |
| Kernel regression | Nadaraya-Watson equals the explicit weighted mean; local linear reproduces lines; CV score equals explicit leave-one-out refits |
| Additive models | backfitting equals the joint fit to `1e-8`; one-term model equals the P-spline; partial dependence equals `alpha + f_j` exactly; centred terms; exact additive lines |
| PCA | closed-form eigenvalues `4s^2/3`, `4t^2/3` and diagonal axes; `numpy.linalg.eigh(numpy.cov(D.T))` offline; uncorrelated scores and exact reconstruction; NumPy backend agreement |
| Factor analysis | exact recovery of a population two-factor structure (`F = 0`, p-value 1); Jöreskog's identity with the full ML discrepancy; unit fitted diagonal; Bartlett statistic; rotation invariance of `Lambda Lambda'` |
| Varimax | perfect simple structure rotated back for three angles; the same criterion from rotated inputs; orthogonal `T` with `L T` equal to the output |
| LDA, QDA | posteriors equal normalized `pi_k N(x; mu_k, Sigma_k)` from `multivariate.MultivariateNormal`; two-class direction proportional to `Sigma^-1 (mu_2 - mu_1)`; `W`-orthonormal canonical directions with ratios `a'Ba`; seeded test error within 4 binomial SE of `Phi(-Delta/2)` |
| k-means, silhouette | inertia equals the optimum over every partition of nine points for `K = 2, 3`; silhouettes `4/5, 3/4, 3/7, 7/11` by exact arithmetic |
