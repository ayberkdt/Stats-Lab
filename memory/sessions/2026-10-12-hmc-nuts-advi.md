# 2026-10-12 — HMC, NUTS, and Gaussian ADVI

## Done

- `inference.hamiltonian`:
  - samplers: `hamiltonian_monte_carlo` and `no_u_turn_sampler` (slice NUTS,
    Hoffman-Gelman Algorithms 3/6);
  - integrator utilities: `leapfrog`, `hamiltonian_energy`;
  - warm-up schedule: `adaptation_windows`, with dual averaging and windowed
    diagonal or dense metric;
  - results: `HamiltonianResult`, which wraps an `MCMCResult`, and
    `HamiltonianChainStatistics`, which holds divergences, tree depths, and
    E-BFMI.
- `inference.variational`: `advi` (mean-field and full-rank, Adam,
  common-random-number ELBO trace, Polyak-Ruppert tail averaging) and
  `VariationalResult`.
- `inverse.CalibrationProblem.sample_posterior(method=...)` accepts `"hmc"`
  and `"nuts"`. The default is unchanged; `CalibrationPosterior.sampler` is a
  new optional field.
- Docs: twelve `EQ-HMC/NUTS/VI/OPT-ADAM` records and eight references (checked
  against JMLR, arXiv, and Crossref), Section 12 of the
  nonlinear-calibration guide, and ADR-0032.

## Verification

- `PYTHONPATH=src python -m pytest -q`: 583 passed, 840 subtests. The new
  file `tests/test_hmc_nuts_advi.py` has 21 tests and runs in about 12 s.
- `ruff check .` and `mypy src/probstat` are clean.

## Next safe step

- Multinomial NUTS.
- PSIS/Pareto-k diagnostics for ADVI.
- An optional analytic-gradient hook in `ProbabilisticModel` so that
  calibration NUTS does not need finite differences.
- The Stan-like metric shrinkage constants were not checked against Stan's
  source.
