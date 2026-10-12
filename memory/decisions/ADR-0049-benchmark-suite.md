# ADR-0049: Reproducible benchmark suite

- Status: Accepted (proposed during the 2026-10-12 session)
- Date: 2026-10-12

## Context

Roadmap Phase 6 asks for "reproducible benchmark suites that compare
correctness, calibration, numerical stability, and computational cost, not
accuracy alone". Until now each method was tested in isolation against a
tolerance; nothing compared methods on shared problems, nothing checked
whether a method's *reported* error matched its actual error, and nothing
recorded cost.

## Decision

1. **Location.** A top-level `benchmarks/` package (tooling, like `tools/`),
   not part of the installed library. It uses only the standard library and
   probstat (no numpy/scipy). `benchmarks/run.py` adds `src/` to the path when
   probstat is not installed.
2. **References never come from probstat.** `benchmarks/references.py` imports
   only `math` and `random`: closed forms (linear and lognormal reliability,
   Ishigami, G-function, exponential integrand, conjugate posterior), 1-D
   composite Simpson quadratures (parabolic and four-branch limit states, the
   latter in rotated coordinates where the failure set is
   `|b| >= 3 or |a| >= 3 + 0.2 b^2`), and an independent implementation of
   linear-Gaussian Shapley effects. `tests/test_benchmarks.py` checks each one
   against a second computation (Gauss-Hermite, Gauss-Legendre conditional
   moments, panel doubling, a seeded Monte Carlo, and the library's own
   `linear_gaussian_shapley` and `calibrate_linear_gaussian`). The four-branch
   reference is the quadrature (4.457331e-3); the requested large seeded Monte
   Carlo (10^7 samples, seed 20261012: 4.4378e-3 +/- 2.10e-5, z = -0.93) is
   kept as an independent confirmation because it is 4 orders less precise.
3. **What is recorded per run** (problem, method, seed, quantity): estimate,
   reference, absolute/relative/scaled error, the method's own standard error
   and interval, `z = error / SE`, whether the reference lies in the reported
   interval, model evaluations (read from the model's `EvaluationCounter`, so a
   method cannot under-report cost), wall time, warnings, notes, and status
   (`ok`, `failed` with the exception text and source line, or `skipped` with
   the budget reason). Every adapter call is wrapped; a failure becomes a
   record and the run continues.
4. **Method kinds.** *Estimators* (sampling methods with a reported SE) are
   judged on error, |z|, and coverage; *approximations* (FORM, SORM,
   multi-point FORM, PCE, Laplace, ADVI, closed forms) only on error, because
   their reported numbers do not bound their bias; *bounds* (Ditlevsen) on
   whether they contain the reference. A reported SE below 1e-6 x |reference|
   is floored there when forming z (the level of the library's root-finding
   tolerances), so line sampling's exact answers on hyperplanes are not
   flagged as mis-calibrated.
5. **Known limitations vs regressions.** `methods.EXPECTED_FAILURES` lists
   failures that are properties of a method on a problem (the four-branch
   margin has a symmetric kink at the origin, so every method that starts with
   one HL-RF search from `u = 0` fails). They are still reported (marked
   "known limitation"); any other failure makes `run.py` exit 1, so CI catches
   API breakage and new numerical failures.
6. **Sizes.** `--quick` (CI, about 35 s on the development machine: 2 seeds
   for reliability, sensitivity and calibration, 5 for propagation) and full
   mode (10-30 seeds, larger budgets). AK-MCS is skipped above d = 2 in quick
   mode (budget, recorded as `skipped`).
7. **Outputs.** `benchmarks/results/<mode>.json` (meta, problem list, summary,
   every record) and `docs/benchmarks.md`, whose tables have at most seven
   columns and whose "What the numbers say" section is computed from the
   summary with stated thresholds. The committed report is the quick-mode run.
8. **CI** runs `python benchmarks/run.py --quick` after the unit tests.

## Library defects found by the suite (fixed, with regression tests)

- `inference.effective_sample_size`: for strongly antithetic chains (static
  HMC on the calibration problem, lag-1 autocorrelation about -0.93) the
  truncated Geyer sum gave `tau < 0`, an ESS of -2471, and `summarize_draws`
  crashed with a `sqrt` domain error. `tau` is now bounded below by
  `1 / log10(M n)` as in the reference code of Vehtari et al. (2021).
- `reliability.ak_mcs`: the incremental population posterior re-derived the
  GP Cholesky factor in original units with plain `sum`. When maximum
  likelihood ends at near-singular hyperparameters (parabolic limit state:
  length scales ~1e3-1e5, signal/nugget ~ 1/eps) its pivots collapsed to the
  noise floor, population means differed from the returned surrogate by up to
  1e9, and AK-MCS reported `P_f = 0.641` for a true `8.2e-4`. It now
  reproduces the fit's arithmetic exactly (standardized units, `kernel_value`,
  `math.fsum`).

## Consequences

- New methods should get an adapter in `benchmarks/methods.py`; new problems
  need a reference computed in `references.py` and a second check in the test.
- Wall times are indicative only; evaluation counts are the portable cost.
- Quick-mode |z| and coverage statistics rest on few seeds; conclusions about
  calibration need the full mode.
- Open issue (not fixed, reported): the unbounded GP likelihood fit still
  reaches near-singular hyperparameters on smooth limit states, where the
  posterior variance falls below resolution and AK-MCS can stop after its
  initial design with `min U` in the hundreds; a bound or prior on the
  hyperparameters is a separate design decision.
