# Session — 2026-10-11 — UQ-0 model validation and prediction credibility

## Delivered

- `probstat.validation` (layer 5):
  - `records.py`: `ValidationExperiment`, `ValidationDomain`, `DomainCheck`.
    Hull membership is decided by an exact phase-one simplex.
  - `metrics.py`: exact step-CDF `area_metric`, `u_pooling` (PIT area and KS),
    `mahalanobis_metric`, `reliability_metric`, `asme_validation_comparison`
    with `budget_standard_uncertainty`, `ValidationAlternative`, and
    `validation_bayes_factor`.
  - `assessment.py`: `predict_at_conditions`, `assess_predictions`, and
    `assess_validation` (one root seed, scoped verdict, `summary()`).
  - `credibility.py`: `VerificationEvidence`, `PredictionStatement`, and
    `CredibilityRecord`, with explicit gaps and no aggregate score.
- Nine `EQ-VAL-*` records and eight bibliography entries.
- `docs/foundations/model-validation.md` and ADR-0022.
- ATHENS-1.3 is now `Implemented`. ATHENS-1.4 and 1.7 have more evidence but
  stay `Partial`.

## Verification

- `tests/test_model_validation.py`: 39 tests, about 8 s.
- Full suite: 341 passed. `ruff check .` and `mypy src/probstat` are clean.
- Calibration check (scratch script, not committed): over 40 synthetic data
  sets from the correct model, the Mahalanobis and KS checks rejected at the
  5% level 2 and 4 times, with mean p-values 0.46 and 0.53.

## Open items and next safe step

- The Liu et al. 2011 author list was checked by web search (Liu, Chen,
  Arendt, Huang). The task brief named different co-authors.
- Section-level locators for Kass-Raftery (Section 3.2) and Silverman
  (Section 3.4.2) were not checked against the texts in this session.
- Next: model-form uncertainty propagated to new conditions (discrepancy
  calibrated across validation points), and multivariate or correlated-QoI
  validation metrics.
