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
| 04 spectral | verified | `inverse.spectral` (whitened SVD, complete nullspace, Picard; `generalized_svd` for wide operators and, on request, a rank-deficient stack restricted to the complement of the common nullspace with minimum-norm Tikhonov solutions [EQ-INV-SPEC-004]); prior-whitened spectrum in `inverse.linear_gaussian`; functional resolution operators and averaging kernels with mass matrices in `inverse.function_space_extended` | `tests/test_inverse_spectral_iterative_selection.py`, `tests/test_function_space_extended.py` | none |
| 05 linear problem | verified | `inverse.linear_problem` (QR/SVD/Krylov affine Tikhonov, estimator analysis, equality, NNLS, box, TLS; `regularized_factorization` reuses one QR for many data vectors [EQ-INV-LIN-007]) | `tests/test_linear_inverse_problem.py` (1000-replicate Monte Carlo of `M Gamma M'` within 4.5 MCSE, 0.15 s with the shared factor, always run) | none |
| 06 iterative | verified | `inverse.iterative` (Landweber, early-stopped CGLS/LSQR/LSMR, hybrid LSQR with projected GCV, weighted GCV with fixed or adaptive weight [EQ-INV-ITER-004], discrepancy and fixed rules, general penalty `L`); iteratively reweighted norm (IRLS) for `l1`/TV in `inverse.sparse_extended`; `inverse.flexible_hybrid` (flexible Golub-Kahan hybrid with iteration-dependent reweighting preconditioner; identity preconditioner reproduces `hybrid_lsqr`) | `tests/test_inverse_spectral_iterative_selection.py`, `tests/test_sparse_extended.py`, `tests/test_flexible_hybrid.py` | none |
| 07 selection | verified | `inverse.parameter_selection` (GCV, UPRE, discrepancy, L-curve, evidence, Hutchinson df, grouped CV; `selection_rule_study` [EQ-INV-SEL-004]) | `tests/test_inverse_spectral_iterative_selection.py` (3 families x 10 seeds in the slow tier, table below) | none |
| 08 sparse/TV | verified | `optimization.proximal`, `inverse.sparse` (lasso, TV); `inverse.sparse_extended` (IRLS for `l1` and anisotropic TV, Huber-TV by PDHG, TGV^2 in 1D with a proven step bound, MCP and SCAD prox maps and proximal gradient with a stationary-point-only guarantee); TGV^2 and isotropic IRLS on 2D grids, `noise=` covariance factors in the IRLS, Huber-TV, TGV and non-convex solvers (results with `noise=None` unchanged) | `tests/test_proximal_sparse.py`, `tests/test_sparse_extended.py` | none (2D TGV is practical only on small images in pure Python) |
| 09 nonlinear | verified | `inverse.gauss_newton`, `optimization.trust_region`, `inverse.newton_trust_region` (now with a discrepancy stop); `inverse.nonlinear_extended` (projected Gauss-Newton with an epsilon-active set and KKT report, Huber/Cauchy IRLS Gauss-Newton, variable projection with full or Kaufman Jacobian) | `tests/test_gauss_newton_krylov.py`, `tests/test_trust_region_newton.py`, `tests/test_nonlinear_extended.py` | none |
| 10 likelihoods | verified | `inverse.observation_laws`; `inverse.observation_laws_extended` (negative binomial, interval-censored Gaussian, contaminated normal, normalized Huber density with sandwich covariance, multivariate t with scatter, heteroscedastic and full state-dependent covariance with log-det gradients, profiled and marginalized noise scale) | `tests/test_observation_laws.py`, `tests/test_observation_laws_extended.py` | none |
| 11 identifiability | verified | `inverse.profile_likelihood`; `inverse.profile_extended` (2D profiles with serpentine continuation and unreliable-cell marking, parametric bootstrap of the LR threshold, ODE forward sensitivities with rank and null directions) | `tests/test_profile_likelihood.py`, `tests/test_profile_extended.py` | none |
| 12 Bayesian linear | verified | `inverse.linear_gaussian`, `inverse.function_space`; `inverse.function_space_extended` (likelihood-informed subspaces with projection error, resolution operators and averaging kernels, Whittle-Matern SPDE prior in 1D, cosine and shifted-Legendre bases); `inverse.function_space_matrix_free` (matrix-free LIS by Lanczos or a randomized range finder with per-pair residuals, resolution probes by CG with Hutchinson trace) | `tests/test_linear_gaussian_inversion.py`, `tests/test_function_space.py`, `tests/test_function_space_extended.py`, `tests/test_function_space_matrix_free.py` | none |
| 13 samplers | verified | `inverse.posterior_targets`; `inverse.posterior_sampling` (tempered-SMC adapter with log evidence and repeated-run spread, pCN versus random-walk dimension study, Gauss-Legendre quadrature references in 1D/2D with truncation estimate, mode-occupation checks) | `tests/test_posterior_targets.py`, `tests/test_posterior_sampling.py` | none |
| 14 ensembles | verified | `inverse.regularized_ensemble`; `inverse.ensemble_extended` (Gaspari-Cohn localization, innovation-based adaptive inflation, ETPF through the exact transport solver, quadrature reference of the bimodal square map) | `tests/test_regularized_ensemble.py`, `tests/test_ensemble_extended.py` | none |
| 15 PDE adjoints | verified | `inverse.pde_models`, `inverse.heat_models`; `inverse.ode_models` (RK4 ODE model with exact discrete tangent and adjoint and checkpointing, simulator adapter with caching and recorded failures, heat-equation log-diffusivity inversion with exact discrete adjoint, N03 benchmark); `inverse.pde_models_2d` (2D elliptic log-conductivity with harmonic-mean faces, PCG or banded Cholesky, exact discrete adjoint, second-order manufactured-solution convergence; explicit-Euler heat model that refuses unstable steps unless allowed) | `tests/test_pde_models.py`, `tests/test_heat_models.py`, `tests/test_ode_models.py`, `tests/test_pde_models_2d.py` | none (no 2D explicit time stepping) |
| 16 multifidelity | verified | `inference.delayed_acceptance`, `inverse.approximation_error`; `inverse.multifidelity` (fine-equivalent cost and cost-normalized ESS, actively refined surrogate posterior with extrapolation report, multilevel MCMC with level variances and allocation); coarse-to-fine warm starts with fine-equivalent cost against a cold start, sensitivity-error validation of approximate models | `tests/test_delayed_acceptance.py`, `tests/test_approximation_error.py`, `tests/test_inverse_multifidelity.py` | none |
| 17 SBI | verified | `inverse.sbi_workflow` (simulator contract with work ledger and split streams, synthetic likelihood, ABC rejection/SMC, NPE and NRE through the existing fitters, tolerance study, bimodal comparison against quadrature, SBC, OOD heuristic, Gaussian score test at `t = 0` and noisy `t`, plug-and-play proximal gradient); diffusion posterior sampling with exact noisy-time scores (reverse SDE and probability-flow ODE) and the approximate DPS-style variant, flow matching in the bimodal comparison, an explicit prior-simulations-only training policy | `tests/test_sbi_workflow.py` | none (neural-score diffusion stays out of scope until calibrated evidence exists, as the roadmap requires) |
| 18 design | verified | `design.linear_optimal`; `design.linear_optimal_extended` (operator-path A criterion by Hutchinson and CG, D criterion from Lanczos or randomized SVD with a truncation bound, stochastic greedy with common random numbers, sequential design with exact updates, Bayesian-average and worst-case robust design) | `tests/test_linear_optimal_design.py`, `tests/test_linear_optimal_extended.py` | none |
| 19 benchmarks | verified | catalog L01-L07, N01-N03, P01, P02, S01, M01, F01 (`inverse.benchmark_problems`, `benchmark_extended`, `ode_models`, `benchmark_catalog`); `tools/inverse_benchmark_matrix.py`; `benchmarks/run.py` cases `deblur-l04`, `gaussian-l06`, `tomography-l07b`, `decay-n01`, `squared-n02`, `elliptic-p01`, `heat-p02`, each scored against a pure-Python reference in `benchmarks/references.py` | `tests/test_inverse_benchmark_problems.py`, `tests/test_benchmark_extended.py`, `tests/test_benchmark_catalog.py`, `tests/test_benchmarks.py` | none |
| 20 applications | verified | 20A functional inversion; 20B expansion history with correlated calibration noise, a simplified BAO observable with free sound horizon and greedy information gain of added redshifts; 20C orbit determination with station biases and correlated noise; a 2D gravity case with exact prism kernel, depth weighting and the non-uniqueness made explicit (`inverse.benchmark_gravity`); `inverse.benchmark_inference` (Laplace, pCN and tempered SMC against quadrature; Bayes factors of LCDM, wCDM, CPL and two-bin models with SMC standard errors; prior-width sensitivity); `inverse.benchmark_gravity_sphere` (fully normalized Legendre recursion, radial gravity at altitude, Kaula prior, exact posterior and resolved degree, omission error) | `tests/test_function_space.py`, `tests/test_benchmark_cosmology.py`, `tests/test_benchmark_orbit.py`, `tests/test_benchmark_gravity.py`, `tests/test_benchmark_inference.py`, `tests/test_benchmark_gravity_sphere.py` | none |
| 21 pedagogy | verified | guide chapter U18 (now with sections on the new methods); learning-path entry; guide figures `large-scale-semi-convergence`, `large-scale-tv-edges`, `large-scale-posterior-bands`, `large-scale-design-gains` (light and dark, datawiz audit passed, PNGs inspected); laboratory `examples/large_scale_inverse_lab.py` | `tests/test_guide_figures.py`, documentation tests | none |
| 22 verification | verified | this ledger; registry, references, API and math reference and the presentation data regenerated; wheel and sdist built offline (`uv build --offline`, cached setuptools 84), `twine check --strict` passed, wheel smoke test passed on clean Python 3.11.15 and 3.13.14 | see test log below | none |
| 23 references | verified | 35 entries in the first round and 25 in the second, each with a DOI except the ETNA paper; the 24 new DOIs checked against Crossref on 2026-11-26 (title, first author, year, volume); Chung, Nagy and O'Leary (2008) read from the journal's article page and PDF | `test_documentation_contract` | Bakushinskii (1992) was not found on Crossref and is not cited |

## Measured observations worth keeping

- Parameter-choice study (`selection_rule_study`, 3 families x 10 seeds,
  33-point grid from `1e-6` to `1e2`, identity penalty, discrepancy with the
  expected whitened norm `sqrt(m)`; efficiency = rule error / grid-oracle
  error, median and worst over seeds; 100.7 s):

  | Family | GCV | UPRE | discrepancy | L-curve | evidence |
  |---|---|---|---|---|---|
  | L01 polynomial | 1.312 / 1.592 | 1.000 / 1.283 | 1.000 / 1.001 | 1.312 / 1.592 (boundary 10/10) | 1.244 / 1.552 |
  | L01 exponential | 1.007 / 9965 | 1.039 / 35.5 | 1.013 / 1.190 | 1.059 / 2.560 | 1.297 / 3.830 |
  | L04 refined data | 3.205 / 5.962 | 1.962 / 4.534 | 1.284 / 1.405 | 3.933 / 5.962 (boundary 8/10) | 3.889 / 7.301 |

  These families and seeds only; not a general ranking. A first run that
  passed the noise STANDARD DEVIATION times `sqrt(m)` as the discrepancy
  level gave efficiencies up to 86 000: the level must be in whitened units.
- Weighted GCV (`hybrid_lsqr`, `rule="wgcv"`, 1 % noise, one seed): on
  deriv2 (`n = 64`) GCV, adaptive WGCV and `omega = 0.5` all ended at
  relative error 6.66 after 40 steps (`max_iterations`); on the backward heat
  problem (`n = 64`) GCV gave 0.045 and the adaptive WGCV (final
  `omega = 0.50`) 0.080. The over-smoothing that WGCV corrects in the source
  paper did not occur here; projected GCV under-smoothed instead.
- `benchmarks/run.py --quick`, new inverse cases (median relative error over
  two seeds): tomography L07b at 5 % noise, LSQR with discrepancy 0.377 and
  hybrid LSQR with projected GCV 0.331; at 1 % noise the pixel-model error is
  about 10 times the noise norm and LSQR ran to its 200-step cap (relative
  error 6.4, seed 1); exponential decay N01, Gauss-Newton 4.7e-10 and
  trust-region Newton-CG 1.8e-9 from the exact least-squares fit; squared map
  N02, 8.8e-10 on `|x|`; elliptic P01 (underdetermined), IRGNM 0.578;
  heat P02, LSQR 0.386 and hybrid LSQR 0.415.
- pCN dimension study (100 draws, 2 chains, warmup 40, `beta = 0.3`):
  acceptance 0.530, 0.505, 0.480 for `n = 8, 32, 128` against 0.430, 0.265,
  0.065 for random-walk Metropolis with the same step.
- Bimodal square map `y = x^2 + e` (60 members, six seeds): 31 % of the
  EnKF members and 1.1 % of the ETPF members lie in `|x| < 0.5`, where the
  quadrature posterior has mass below `1e-3`.
- Simplified BAO with a free sound horizon (whitened Jacobian of the nuisance
  block, `Omega_m` fixed): smallest singular value 7.6e-15 (SN), 0 (BAO),
  6.8e-15 (SN + BAO) and 24.8 (SN + BAO + Hubble rate).
- A-optimal design figure (12 unknowns, 16 candidates): greedy equals the
  exhaustive optimum at `k = 1` and `3`, worst gap 0.40 at `k = 2`; at
  `k = 5` greedy 2.13, optimum 1.92, random median 2.80.
- Third round (flexible hybrids, 2D TGV and IRLS, matrix-free subspaces, 2D
  elliptic adjoint and explicit stability, warm starts, exact-score
  diffusion, Bayes factors, spherical-harmonic gravity): the behaviours the
  guide states are asserted by the tests named in the status table; the
  warm start of the 1D elliptic hierarchy cost more than the cold start in
  its test, and that report is kept rather than a saving assumed.
- Posterior bands figure (L06, `n = 40`): simultaneous scaling factor
  `c = 2.97`; the truth lies inside the 90 % simultaneous band everywhere and
  inside the pointwise band at 36 of 40 points.

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
| laboratory and benchmark cases: `tests/test_benchmarks.py`, presentation, lab contract, lab regression, architecture, guide figures | 69 passed, 1,640 subtests, exit 0 |
| same round: documentation examples, learning paths, documentation contract | 95 passed, 1,029 subtests, exit 0 |
| `python examples/large_scale_inverse_lab.py` | exit 0, 29 s; on its seed GCV under-smooths the refined-data deblurring (relative error 8.33) while UPRE and the discrepancy rule give 0.11 and TV 0.020 |
| `python benchmarks/run.py --quick` with the new cases | 332 runs, 20 failed (0 unexpected), 18 skipped, 161.7 s under load |
| final full run after the laboratory and benchmark cases (246 test files, six parallel shards) | 4,563 passed, 1 skipped, 0 failed; every shard exit 0; 16 min wall |
| second and third rounds: new and modified test files with documentation contract and architecture | 612 passed, 20 skipped (round 2, 106 s); 310 passed, 13 skipped (round 3, 171 s) |
| slow tiers run once: 1000-replicate estimator covariance (0.15 s, now always run); 3 x 10 parameter-choice study | passed; 107.7 s for the study |
| `python -m ruff check .`; `python -m mypy src/probstat`; `ruff format --check` on the 54 Python files of this work | all checks passed; no issues in 531 source files; 54 files already formatted (158 older files are not ruff-formatted, CI does not require it) |
| `uv build --offline` (cached setuptools 84), `twine check --strict dist/*` | wheel and sdist built; both PASSED |
| wheel smoke test `tools/smoke_test_wheel.py -I` in clean venvs | Python 3.11.15 and 3.13.14: 525 modules imported, smoke test passed |
| `python -m unittest discover -s tests` on Python 3.11.15 (started before the third round) | 5,072 tests: 6 failures, none in new code: 3 guide-figure/README checks (figures added during the run), 1 wall-clock limit under load, `visual-language.md` (matplotlib not installed in the bare venv), `experimental-design.md` prints 0.14264 instead of 0.1425 on 3.11 (pre-existing; the design and surrogate code is unchanged here) |
| same on Python 3.13.14 (started before the third round) | 5,072 tests: 7 failures, 1 error: 3 documentation-contract checks and 2 presentation checks (fragments merged and counts changed during the run), `visual-language.md` (no matplotlib), and two determinism tests that compared dataclasses holding `nan`, which Python 3.13 compares field by field with `==`; both now compare `repr` and pass on 3.12 and 3.13 |
| late test files (third round, trust region, guide figures, contract, presentation, benchmarks, architecture, spectral, linear) on Python 3.11 and 3.13 with unittest | 17 files, 444 tests on each version, every file exit 0 (the two dataclass comparisons included) |
| final full run: `python -m pytest -q -p no:randomly tests` in six parallel shards (264 test files, Python 3.12.1) | 5,212 passed, 27 skipped, 0 failed; every shard exit 0; 18 min wall |
| `python benchmarks/run.py --quick` with the seven inverse cases | 348 runs, 20 failed (0 unexpected), 18 skipped, 311.6 s under load (above the 60 s CI target of quick mode) |
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

No roadmap item is open. Limitations worth revisiting are recorded in the
status table: 2D TGV is slow in pure Python, the explicit heat model has no 2D
time stepping, and diffusion sampling with learned scores waits for calibrated
evidence.
