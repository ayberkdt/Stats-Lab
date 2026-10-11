# ADR-0020 — Continuous distribution catalog for UQ and reliability

- Status: Accepted
- Date: 2026-10-11

## Context

UQ, reliability, and risk studies need more marginal laws than Normal,
LogNormal, Uniform, Exponential, and Weibull: bounded laws for fractions and
expert judgement, heavy-tailed laws for robust errors, extreme-value laws for
annual maxima, and the sampling distributions of test statistics. They have to
plug into `modeling.JointInput` (Nataf map) and keep tail accuracy at
reliability indices of 7 or more. The package must stay pure Python with no
runtime dependencies, and `distributions` (layer 1) should not couple to its
layer-1 sibling `optimization`.

## Decision

1. Add `Gamma` (shape-rate, with `from_scale`), `Beta`, `StudentT`
   (location-scale, real `nu > 0`), `ChiSquare` (a cached `Gamma(nu/2, 1/2)`),
   `FisherF`, `Logistic`, `Gumbel` (a `kind="max" | "min"` flag),
   `Triangular`, `TruncatedNormal` (finite or one-sided bounds, `None` = open
   side), and `GeneralizedExtremeValue` (climate sign convention, `xi > 0`
   Frechet). Each is a frozen, slotted dataclass with the full existing
   contract (`support`, `log_pdf`, `pdf`, `cdf`, `sf`, `quantile`, `isf`,
   `mean`, `variance`, `sample(size, *, seed)`).
2. Survival functions are always direct. Incomplete-beta callers supply
   `1 - x` explicitly through a new internal `regularized_beta_pair`, so
   Student-t and F tails stay accurate as `nu -> inf` and near the centre.
3. Quantiles without closed forms use one private solver
   (`distributions._continuous`): invert the smaller tail in log space,
   on a coordinate that flattens the tail (`log x`, `logit x`, `asinh z`,
   or an endpoint-anchored map), bracket by geometric expansion, and refine
   with a local Brent zeroin. Closed-form quantiles are used where they exist
   (Logistic, Gumbel, Triangular, GEV).
4. Truncated-normal probabilities are log masses. The standard-normal tail
   switches to a Mills-ratio continued fraction beyond `z = 35`, and short
   intervals next to a bound use a width-based Taylor series, so mass is never
   recovered from nearly equal standardized points.
5. GEV formulas use `log1p(xi z)/xi` and `expm1(-xi log L)/xi`; its moments
   use a `zeta`-series for `|xi| < 0.01`, so the Gumbel limit is continuous.
6. Samplers: Marsaglia-Tsang gamma variates in log form (the `shape < 1` boost
   cannot underflow before the variates are combined), from which the beta,
   chi-square, Student-t, and F samplers are built. Logistic, Gumbel,
   Triangular, GEV, and TruncatedNormal use inverse transform.
7. Moment conventions: undefined moments raise `ValueError` (Student-t with
   `nu <= 1`); divergent moments return `inf`.
8. `numerics.digamma`, `numerics.trigamma`, and `numerics.log_minus_digamma`
   (layer 0) support `inference.fit_gamma_mle` and
   `inference.fit_weibull_mle`. These profile out the rate or scale and solve
   the monotone shape equation by bracket-safeguarded Newton iteration.

## Consequences

- Every new law works as a `RandomInput` marginal. Tests round-trip
  `to_standard_normal`/`from_standard_normal` up to `|u| = 7.5`, and
  `F(F^-1(p)) = p` holds to `1e-9` relative down to `p = 1e-12`. Near finite
  support ends the only additional error is the float spacing of `x`.
- A Student-t with `nu <= 1` cannot be a `RandomInput`, because the mean
  property raises. This is deliberate, since its mean does not exist.
- `_special` prefactors still come from `lgamma` differences, so the relative
  accuracy of gamma and beta functions degrades for shapes far above `1e4`.
  Uniform asymptotic expansions (Temme) remain future work.
- Inverse-transform truncated-normal sampling costs a root solve per draw;
  it is exact and simple but slower than rejection samplers.

## Rejected alternatives

- Importing `optimization.brent_root` into distributions: it would couple two
  layer-1 siblings. A local solver specialized to monotone log-tail residuals
  is short and has no such coupling.
- Returning `nan` for undefined moments: it propagates silently through
  propagation and Nataf correlation solves.
- A closed-form truncated-normal quantile through `Phi^-1`: it fails once the
  truncation region lies beyond the underflow point of `erfc`.
