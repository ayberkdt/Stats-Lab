# ADR-0006: Tail, Quantile, and Parameterization Contracts

- Status: Accepted
- Date: 2026-10-09

## Context

Distribution names can hide incompatible parameterizations, and mathematically
equivalent formulas can have very different floating-point behavior. In
particular, geometric/negative-binomial support conventions vary, `1-CDF` can
destroy tail precision, and quantile endpoints differ between finite,
unbounded, and degenerate supports.

## Decision

Define Geometric and Negative Binomial as failure counts before the first and
`r`-th successes. Repeat this convention in class documentation, equation
records, tests, examples, and plots.

Expose survival and quantile as narrow structural protocols. Implement survival
directly when a complementary sum or special-function branch is available.
Define quantiles as generalized inverse CDFs and document endpoint behavior for
every support type.

Use a shared regularized incomplete-beta implementation with DLMF's continued
fraction and symmetry boundary. Keep it internal until broader special-function
accuracy and error-budget work justifies a public numerical API.

Define continuous Uniform by finite endpoints and Exponential by rate, not
scale. Hazard is valid only under each implementation's documented survival
domain; zero-survival `0/0` cases raise explicitly.

## Consequences

- Parameterization errors become visible rather than silently plausible.
- Small upper tails remain useful beyond the precision of `1-CDF` subtraction.
- Generic consumers can request survival, quantile, density, log-density, or
  hazard capabilities independently.
- Quantile endpoints can be infinite even when ordinary evaluation arguments
  must be finite.
- Gamma/Beta and derived distributions can reuse the numerical beta/gamma
  foundation, but inverse special functions remain planned work.
