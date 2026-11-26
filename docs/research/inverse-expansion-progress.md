# Inverse-problems research expansion: progress ledger

Tracks the 2026-11-26 roadmap (tasks 01-23) against the checkout. Status
values: `verified` (implemented, tested against independent evidence, tests
run and passing), `partial` (part of the task verified, the rest listed as
open), `deferred` (not started; reason given), `blocked`.

- Starting point: `main` at `1d14f28`, clean working tree, version 0.27.0,
  Python 3.12.1 on Windows; NumPy 2.2.6 and SciPy 1.14.1 were installed and
  used only for one-off cross-checks outside the test suite (ADR-0144).
- Decision record: [ADR-0144](../adr/ADR-0144-matrix-free-inverse-problems.md).
- Guide: [large-scale inverse problems](../foundations/large-scale-inverse-problems.md).
- All new APIs are experimental.

## Status by task

| Task | Status | Delivered | Evidence (tests) | Open items |
|---|---|---|---|---|
| 01 audit | verified | layer map, contracts and existing inverse APIs read; gap ledger below | contract tests (see end) | none |
| 02 contract | verified | conventions applied: whitening, `lambda^2` vs `alpha`, explicit `x_ref`, stop reasons, counts | doc/architecture tests | none |
| 03 operators | verified | `linalg.operators`, `linalg.covariance_operators`, `linalg.krylov` | `tests/test_linalg_operators_krylov.py` | operator-path Lanczos for huge `n` is `O(k n)` memory by design |
| 04 spectral | partial | `inverse.spectral` (whitened SVD, complete nullspace, Picard, restricted GSVD); prior-whitened spectrum in `inverse.linear_gaussian.prior_whitened_information` | `tests/test_inverse_spectral_iterative_selection.py` | GSVD for `m < n` or a rank-deficient stack; functional resolution with mass matrices |
| 05 linear problem | verified | `inverse.linear_problem` (QR/SVD/Krylov affine Tikhonov, estimator analysis, equality, NNLS, box, TLS) | `tests/test_linear_inverse_problem.py` | multiple right-hand sides with factor reuse; 1000-replicate Monte Carlo of the estimator covariance (slow tier) |
| 06 iterative | partial | `inverse.iterative` (Landweber, early-stopped CGLS/LSQR/LSMR, hybrid LSQR with projected GCV / discrepancy / fixed rules and a general penalty `L`) | `tests/test_inverse_spectral_iterative_selection.py` | weighted GCV (source not verified offline); flexible/IRN variants |
| 07 selection | verified | `inverse.parameter_selection` (GCV, UPRE, discrepancy, L-curve, evidence, Hutchinson df, grouped CV) | `tests/test_inverse_spectral_iterative_selection.py` (includes an MC unbiasedness check of UPRE) | 3 families x 10 seeds comparison table (slow tier) |
| 08 sparse/TV | partial | `optimization.proximal` (prox maps, ISTA/FISTA, PDHG, ADMM), `inverse.sparse` (lasso, TV) | `tests/test_proximal_sparse.py` | IRLS, Huber TV, TGV, non-convex penalties; PDHG is slow for large `alpha` (measured, see guide) |
| 09 nonlinear | partial | `inverse.gauss_newton` (`DifferentiableForward`, `CallbackForward`, `check_derivatives`, GN with inner LSQR/CGLS and Armijo, IRGNM); `optimization.trust_region` (Steihaug CG [REF-STEIHAUG-1983], trust-region Newton-CG) and `inverse.newton_trust_region` (Gauss-Newton or full Hessian actions) | `tests/test_gauss_newton_krylov.py`, `tests/test_trust_region_newton.py` | bounds; robust losses; variable projection [REF-GOLUB-PEREYRA-1973]; discrepancy stop in the trust-region method |
| 10 likelihoods | partial | `inverse.observation_laws` (normalized Gaussian with full covariance, Poisson, binomial, independent Student-t, censored Gaussian, missing-data mask with marginal submatrix; Fisher scoring / IRLS for linear predictors) | `tests/test_observation_laws.py` | negative binomial, interval censoring, mixture outliers, Huber, multivariate t, `Gamma(x)`-dependent gradients, nuisance noise parameters |
| 11 identifiability | partial | `inverse.profile_likelihood` (profiles with continuation, Wilks intervals that never fabricate endpoints, efficient information, estimability) | `tests/test_profile_likelihood.py` | 2D profiles, bootstrap calibration, ODE observability |
| 12 Bayesian linear | partial | `inverse.linear_gaussian` (exact data/parameter-space posterior, evidence, conditional simulation, low-rank posterior); `inverse.function_space` (sine, hat and piecewise-constant bases with exact mass matrices, Gauss-Legendre integral operators, truncation-consistent spectral Gaussian prior with tail bound, pointwise and simultaneous bands) | `tests/test_linear_gaussian_inversion.py`, `tests/test_function_space.py` | likelihood-informed subspaces; functional resolution kernels; non-sine priors |
| 13 samplers | partial | `inverse.posterior_targets` (prior-whitened target with gradient by one `vjp`; adapters to the existing pCN and NUTS/HMC; mapping back to physical coordinates; SBC for the linear Gaussian case through the existing SBC routine) | `tests/test_posterior_targets.py` | tempered SMC adapter, pCN dimension-robustness study, nonlinear posterior reference |
| 14 ensembles | partial | `inverse.regularized_ensemble` (stochastic and square-root ensemble Kalman updates with covariance factors, observation- and ensemble-space solves, Tikhonov EKI by the augmented system) | `tests/test_regularized_ensemble.py` | localization, adaptive inflation, ETPF; nonlinear multimodal comparison |
| 15 PDE adjoints | partial | `inverse.pde_models` (1D elliptic log-conductivity, Thomas solver, tangent, adjoint, flux balance); `inverse.heat_models` (implicit Euler and Crank-Nicolson initial-condition and source operators with exact discrete transposes and closed-form singular values) | `tests/test_pde_models.py`, `tests/test_heat_models.py` | ODE discrete adjoint; external-simulator adapter; nonlinear coefficient inversion in time |
| 16 multifidelity | partial | `inference.delayed_acceptance` (exact-target two-stage MH, random walk and pCN); `inverse.approximation_error` (paired fine/coarse error mean and covariance, shrinkage, held-out Mahalanobis check, cross-covariance) | `tests/test_delayed_acceptance.py`, `tests/test_approximation_error.py` | cost in fine-equivalent units; surrogate training and active refinement; multilevel MCMC |
| 17 SBI | deferred | existing ABC/NPE/NRE unchanged | - | inverse-problem SBI comparison |
| 18 design | partial | `design.linear_optimal` (A/D/E/goal criteria, greedy rank-one updates, exhaustive oracle, correlated-noise joint criteria) | `tests/test_linear_optimal_design.py` | operator-path and stochastic-trace criteria; sequential and robust design |
| 19 benchmarks | partial | `inverse.benchmark_problems` (L01, L04-L06; L02-L03 existing) and `inverse.benchmark_extended` (L07a 2D deblurring, L07b ray-transform tomography with exact intersection lengths and analytic disc data, N01, N02, P01, P02) with truth/noise stream separation and the inverse-crime flag; `tools/inverse_benchmark_matrix.py` JSON runner | `tests/test_inverse_benchmark_problems.py`, `tests/test_benchmark_extended.py` | N03 ODE oscillator, S01 Poisson imaging, M01, F01, A01-A02; a `benchmarks/run.py` group |
| 20 applications | partial | 20A functional inversion in `tests/test_function_space.py`; 20B synthetic expansion history (`inverse.benchmark_cosmology`: piecewise-constant `w(z)`, exact discrete derivatives of the distance modulus, offset/H0 degeneracy, CPL and oscillating truths from finer representations); 20C simplified orbit determination (`inverse.benchmark_orbit`: two-body range model with exact RK4 sensitivities, well-observed and single-station short-arc geometries) | `tests/test_function_space.py`, `tests/test_benchmark_cosmology.py`, `tests/test_benchmark_orbit.py` | BAO/sound horizon, correlated calibration noise, EIG of added redshifts; orbit nuisance biases and correlated noise; gravity field |
| 21 pedagogy | partial | guide chapter U18 with measured tables and exercises; learning-path entry; guide figures `large-scale-semi-convergence` and `large-scale-tv-edges` (light and dark, datawiz audit passed) | `tests/test_guide_figures.py`, documentation tests | labs; further figures (posterior bands, design gains) |
| 22 verification | partial | this ledger; registry, references, API and math reference and the presentation data regenerated | see test log below | wheel build and clean-environment smoke test not run |
| 23 references | verified | 30 new bibliography entries; metadata checked against Crossref on 2026-11-26 | `test_documentation_contract` | Bakushinskii (1992) and the weighted-GCV paper of Chung, Nagy and O'Leary (2008) were NOT found on Crossref and are not cited |

## Measured observations worth keeping

- Synthetic expansion history (8 piecewise-constant `w(z)` bins, 30
  redshifts in [0.05, 1.5], `sigma_mu = 0.1`): whitened singular values
  about 12.5, 2.33, 0.73, 0.28, 0.12, 0.054, 0.023, 0.0068, so only 2 of
  8 directions exceed unit noise (`tests/test_benchmark_cosmology.py`).
- Orbit determination, 3D single-station short arc: the whitened
  Jacobian has rank 5 of 6 (rotation about the line of sight leaves the
  ranges unchanged); 2D condition numbers 131 (three stations) against
  369 (one station, short arc) (`tests/test_benchmark_orbit.py`).

- Tomography L07b with analytic disc data: the pixel-model error (about
  10 % of the data norm) exceeds 1 % noise, so the discrepancy target is
  never reached, LSQR runs to its cap and the estimate is worse than the
  zero image (`tests/test_benchmark_extended.py`,
  `test_model_error_defeats_the_discrepancy_rule`): a concrete case where
  the noise level alone is the wrong stopping scale.
- Approximation-error model on a linear pair (50 replicates, 90 %
  intervals of `sum(x)`): coverage 0.26 for the naive coarse posterior
  and 1.00 with the error model (conservative); seeds of
  `tests/test_approximation_error.py`, not a general rate.

- Projected GCV in `hybrid_lsqr` under-regularized on deriv2 (`n = 32`, 1 %
  noise): relative error 1.79 against 0.291 for the projected discrepancy
  rule. One problem, one seed; not a general claim.
- `total_variation_regularized_least_squares` (PDHG) on L04 converged for
  `alpha = 0.5` (relative error 0.0141, against 0.137 for early-stopped
  LSQR) but hit the 20 000-iteration cap for `alpha = 2` and `8`.
- The evidence choice of `lambda` (prior `N(0, I / lambda^2)`) gave a larger
  reconstruction error than GCV, UPRE, discrepancy and the L-curve on the
  deriv2 example of the guide.

## Test log

Commands, exit codes and elapsed times of the verification runs
(Windows, Python 3.12.1, `PYTHONPATH=src`, `MPLBACKEND=Agg`).

| Command | Result |
|---|---|
| `python -m pytest -q -p no:randomly tests` in six parallel shards (file lists split round-robin) | 4,360 items: 4,352 passed, 7 failed, 1 skipped; 17-38 min per shard under load. The three test files added afterwards (`test_function_space.py`, `test_heat_models.py`, `test_trust_region_newton.py`) are covered by the targeted run below. |
| failures of that run | `test_real_data_case_studies::test_runs_in_under_a_minute` and `test_property_regressions::test_large_population_is_fast` (wall-clock limits under six concurrent shards; both passed when re-run with less load); `test_documentation_examples` (new guide code blocks, README public-name count) and `test_guide_figures::test_every_guide_has_a_figure` (fixed: runnable blocks, counts, the new figure); `test_presentation` (two tests; fixed by `tools/build_presentation_data.py`) |
| targeted re-run: guide figures, documentation examples, presentation, learning paths, the two timing tests, and the three late test files | 214 passed, 2 failed (`test_presentation`, which ran while the deck data were being regenerated), 866 s |
| `pytest tests/test_presentation.py tests/test_architecture.py tests/test_documentation_contract.py tests/test_api_reference.py tests/test_math_reference.py` | 30 passed, 4,911 subtests, exit 0, 348 s |
| `python -m ruff check .` | All checks passed, exit 0 |
| `python -m mypy src/probstat` | no issues in 506 source files, exit 0 |
| `python benchmarks/run.py --quick` | exit 0, 337 s: 322 runs, 20 failed (0 unexpected), 18 skipped |
| `python tools/build_guide_figures.py --group inverse_large_scale` | 2 figures, datawiz audit passed in both palettes (the first after one fix of an annotation that overlapped a tick label) |
| second round (observation laws, regularized ensemble, posterior targets, function space, heat models, trust region): their six test files with presentation, benchmarks, guide figures, architecture, API and math reference tests | 176 passed, 5,711 subtests, exit 0, 134 s |
| same round: documentation examples, documentation contract, learning paths and the first-round inverse test files | 162 passed, 1 failed (README test floor `3,500+` fell below 80 % of the 4,413 test cases; raised to `4,400+`), then `ReadmeTests` and `test_presentation` re-run: 23 passed |
| `benchmarks/results/quick.json`, `docs/benchmarks.md` | regenerated by `benchmarks/run.py --quick` at this commit; the presentation data were rebuilt from them |
| third round (approximation error, extended benchmarks, general-penalty hybrid LSQR): their test files with spectral/iterative/selection, presentation, guide figures, architecture, API and math reference tests | 99 passed, 5,719 subtests, exit 0, 256 s |
| same round: documentation examples, documentation contract, learning paths | 95 passed, 1,030 subtests, exit 0, 523 s |
| final full run: `python -m pytest -q -p no:randomly tests` in six parallel shards (245 test files) | 4,554 passed, 1 skipped, 0 failed; every shard exit 0; 17 min wall |
| CI step "Examples run": every `examples/*.py` with `MPLBACKEND=Agg` | 0 failures, 1,714 s; no tracked file changed |
| `python tools/inverse_benchmark_matrix.py --quick` | exit 0, 165 s, 39 runs, 0 failed runs; see below |

### Benchmark matrix (quick: seeds 11, 23, 37; median relative error)

| Case | Method | Median rel. error | Note |
|---|---|---:|---|
| L01 | LSQR, discrepancy stop | 0.111 | |
| L01 | Tikhonov, GCV | 0.112 | |
| L01 | Tikhonov, UPRE | 0.0982 | |
| L01 | Tikhonov, discrepancy | 0.0983 | |
| L04 (refined data) | LSQR, discrepancy stop | 0.110 | |
| L04 (refined data) | Tikhonov, GCV | 0.114 | |
| L04 (refined data) | Tikhonov, UPRE | 0.104 | |
| L04 (refined data) | TV, PDHG, `alpha = 0.5` | 0.0426 | all 3 runs hit the 5,000-iteration cap (`converged=False`) |
| L05 | lasso, FISTA, `alpha = 0.05` | 0.411 | all 3 runs hit the 3,000-iteration cap |
| L05 | LSQR, discrepancy stop | 0.701 | |
| L06 | exact Gaussian posterior mean | 0.328 | |
| L06 | Tikhonov, GCV | 0.415 | GCV optimum at the grid boundary in all 3 runs |
| L06 | LSQR, discrepancy stop | 0.425 | |

Three seeds per cell: medians only, no ranking claim.

## Next atomic actions

1. Run the CI packaging steps (`python -m build`, `twine check --strict`,
   `tools/smoke_test_wheel.py`) in a clean virtual environment, and the
   suite on Python 3.11 and 3.13. Not run here: the local setuptools is
   70.0 while `pyproject.toml` requires `setuptools>=77`, and an isolated
   build would download the build backend.
2. Add a `benchmarks/run.py` group for the inverse catalog (L01-L07, N01,
   N02, P01, P02) so that the quick benchmark report covers the new solvers.
3. 04C: functional resolution with mass matrices, and the GSVD for
   `m < n` or a rank-deficient stack.
4. 10/16: interval censoring, negative binomial and mixture outliers;
   approximation-error cost in fine-equivalent units.
5. 20B: correlated calibration noise and the expected information gain of
   added redshifts in the expansion-history case; 20C: nuisance biases and
   correlated noise in the orbit case.
6. 21: a laboratory under `examples/` that satisfies `tests/test_lab_contract.py`.

