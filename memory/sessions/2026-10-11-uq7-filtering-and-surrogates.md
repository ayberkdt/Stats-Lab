# Session — 2026-10-11 — UQ-7 filtering/data assimilation and surrogates

## Delivered

- `probstat.filtering` (layer 6):
  - `LinearGaussianStateSpace` and `NonlinearStateSpace`, plus `simulate`.
  - Kalman predict/update with the Joseph form, innovation log-likelihood,
    and missing observations.
  - RTS smoother for KF, EKF, and UKF results.
  - Steady-state Riccati solver.
  - EKF with analytic or central-difference Jacobians.
  - Scaled unscented transform and UKF.
  - Bootstrap particle filter with systematic resampling triggered by ESS.
- `probstat.data_assimilation` (layer 6, no import of `filtering`):
  - Stochastic perturbed-observation EnKF with explicit multiplicative
    inflation.
  - 3D-Var by BFGS with an analytic gradient.
- `probstat.surrogates` (layer 5):
  - Regression PCE with Hermite (Nataf) and Legendre bases, total-degree and
    hyperbolic truncation, closed-form LOO, and analytic moments and Sobol
    indices.
  - GP regression with SE and Matérn-5/2 ARD kernels, an explicit nugget,
    exact-gradient BFGS with seeded restarts, and LOO.
  - `SurrogateModel` and `surrogate_probabilistic_model` adapters.
- Equation records, bibliography entries, two foundation guides, ADR-0018,
  ADR-0019, and ATHENS-8.1 to 8.15 marked `Implemented`.

## Numerical results (reproducible seeds; see the guides)

- Scalar random walk `q=0.5, r=2`: steady gain `0.39039`, predicted variance
  `1.28078`, matching the closed form.
- Constant-velocity tracking (150 steps): position RMSE 2.050 (raw), 1.095
  (KF), and 0.615 (RTS).
- Ishigami PCE (degree 10, `q=0.75`, 288 runs): LOO error 0.0018; every Sobol
  index within 0.001 of its analytic value.

## Verification

- `PYTHONPATH=src python -m pytest -q`: 203 passed, 225 subtests, 23.8 s.
- `python -m ruff check .`: clean.
- `python -m mypy src/probstat`: clean, 78 source files.

## Next safe step

- Add sparse PCE (LARS/OMP), EnKF localization, and square-root variants.
- Add particle smoothing.
- Add labs or examples, and README/CHANGELOG/ROADMAP entries; those files were
  out of scope for this session.
