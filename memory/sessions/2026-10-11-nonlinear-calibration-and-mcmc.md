# Session: nonlinear calibration and MCMC

Date: 2026-10-11

## Delivered

- `inference.mcmc`: random-walk and adaptive Metropolis (adaptation confined
  to warm-up), multi-chain results, split and rank-normalized R-hat, Geyer ESS,
  MCSE, posterior summaries, and a BFGS Laplace approximation.
- `inference.bayesian`: Normal-inverse-gamma update with Student-t marginal and
  predictive laws, plus Student-t CDF/quantile.
- `inverse.nonlinear`: `CalibrationProblem` with experiments, known/unknown
  Gaussian or Student-t noise, discrepancy variance, weighted least squares,
  Laplace posterior, standard-normal-space adaptive Metropolis, and
  prior-scaled Fisher identifiability.
- `inverse.prediction.predict`: posterior push-forward with latent and noisy
  predictive samples kept separate.
- Equation records, ten references, guide
  `docs/foundations/nonlinear-calibration-and-mcmc.md`, ADR-0017, and evidence
  for ATHENS-7.4 and ATHENS-7.12.

## Verification

- `PYTHONPATH=src python -m pytest -q`: 193 passed (24 new tests, about 8 s).
- `python -m ruff check .` and `python -m mypy src/probstat` passed.

## Next safe step

Add heteroscedastic/correlated noise models and a gradient-based sampler only
after an ADR fixes how gradients are supplied without runtime dependencies.
