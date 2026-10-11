# ADR-0027 — Generic fitting, goodness-of-fit calibration, and model selection

- Status: Accepted
- Date: 2026-10-11
- Scope: ATHENS-2.7, 2.9, 2.10, 2.11 and LAB 9.1 (statistical fitting)

## Context

`probstat.inference` had closed-form or profile MLEs for four families, a KS
test with an asymptotic p-value, and an AD statistic without a p-value. Two
things were missing:

- a generic MLE for the other catalog families (Gumbel, GEV, Logistic,
  Student t, Beta), with standard errors;
- a valid GOF p-value when parameters are estimated from the tested data
  (the Lilliefors/Stephens problem).

There was also no way to compare candidate families or to build a
`JointInput` from data. The library has no runtime dependencies.

## Decision

1. `inference.fitting` (layer 2) defines `DistributionFamily`. A family has a
   constructor, parameter kinds (`location`, `positive`, `real`), a starting
   rule, a sample domain, an optional closed form, a `regular` flag, and
   optional upper bounds on positive parameters. Eleven ready-made families
   are provided. Normal, LogNormal, Exponential, Gamma, Weibull, and Uniform
   reuse closed forms or the existing profile MLEs.
2. `fit_distribution` minimizes the negative log-likelihood built from
   `log_pdf` with `optimization.minimize_bfgs`, in unconstrained coordinates
   anchored at the start values:
   - `a + s u` for locations, with `s` = data spread;
   - `a exp(u)` for positive parameters;
   - a logistic map for bounded positive parameters.
   If BFGS raises or does not converge, the fit falls back to Nelder-Mead
   followed by a BFGS polish. Invalid parameters map to `+inf`.
3. Standard errors come from the central-difference Hessian in those
   coordinates, anchored at the estimate. They are mapped back by the
   diagonal Jacobian (delta method), which is exact at a stationary point.
   - Non-regular families (Uniform) report `nan`.
   - A bounded parameter at its bound is held fixed and gets `nan`.
   - The Student-t `nu` is bounded by `1000`. Without the bound, light-tailed
     data made it drift to `~3e5` in about 200 BFGS iterations (2.4 s per
     fit), which made bootstrap calibration impractical.
4. `bootstrap_goodness_of_fit` calibrates KS and AD by a parametric bootstrap.
   - Every replicate is refitted with the same estimator.
   - The p-value is `(1 + #{T* >= T})/(1 + B_ok)`.
   - Successful and failed refits are both reported; failures are excluded.
   - The AD p-value is `nan` when the observed AD is infinite (an observation
     on a fitted support boundary, as with Uniform).
   - Streams come from `RandomStream.child("bootstrap-gof", family, b)`.
5. `select_distribution` ranks admissible candidates by AIC, AICc, or BIC. It
   reports Akaike and BIC weights and optional bootstrap GOF per candidate.
   - Domain exclusions and fit failures are recorded, not raised.
   - A `ModelSelectionWarning` is raised when the best model is rejected,
     when a fit did not converge, when a candidate failed, or when
     `n_bootstrap` is too small to reach `alpha`.
6. Joint inputs live in `modeling.fitting` (layer 3), because they return
   `modeling.JointInput`. Marginals are specified or selected per column. The
   Gaussian copula is estimated from ranks: either Spearman with
   `rho = 2 sin(pi r_s/6)`, or normal scores of `R/(n+1)`. A non-positive-
   definite pairwise matrix is repaired by eigenvalue clipping, and the
   repair is flagged and warned about. `fit_multivariate_normal_mle` (divisor
   `n`) lives in `inference.estimation`, which may import the same-layer
   `multivariate`.

## Consequences

- Any law with `log_pdf`, `cdf`, `quantile`, and `sample` can be fitted by
  writing a `DistributionFamily`.
- Bootstrap calibration with numerical refits costs `B` optimizations per
  candidate: about 0.02-0.1 s each for `n = 200-500` in pure Python. The lab
  takes about 10 s for six candidates with `B = 99`. Closed-form families are
  fast.
- AIC and BIC for the non-regular Uniform are descriptive only.
- The Gaussian copula has no tail dependence. Other copula families are out
  of scope.

## Verification

- `tests/test_generic_fitting_model_selection.py`: 24 tests, about 12 s.
  - Offset-start BFGS reproduces the closed-form and profile MLEs to `2e-6`
    relative.
  - Standard errors match `rate/sqrt(n)`, `sigma/sqrt(n)`, `sigma/sqrt(2n)`,
    and the gamma inverse Fisher information.
  - Bootstrap KS p-values under a fitted normal null average between 0.35
    and 0.65; naive p-values are visibly larger.
  - A wrong family gets `p <= 0.05`.
  - Selection recovers Weibull, Gamma, and LogNormal on seeded data.
  - The copula fit recovers a known latent correlation matrix within 0.06.
- `examples/model_selection_lab.py`: Weibull wins (Akaike weight 0.870) and
  is not rejected (KS p 0.23, AD p 0.30).

## References

[REF-AKAIKE-1974], [REF-SCHWARZ-1978], [REF-HURVICH-TSAI-1989],
[REF-BURNHAM-ANDERSON-2002], [REF-LILLIEFORS-1967], [REF-STEPHENS-1974],
[REF-DAVISON-HINKLEY-1997], [REF-GENEST-GHOUDI-RIVEST-1995],
[REF-KRUSKAL-1958], [REF-WILK-GNANADESIKAN-1968].
