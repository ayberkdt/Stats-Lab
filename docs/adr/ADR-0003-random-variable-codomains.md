# ADR-0003: Generic Codomains and Shared-Measure Joint Laws

- Status: Accepted
- Date: 2026-10-09

## Context

Random variables are measurable mappings, but many software APIs conflate them
with numeric arrays or distribution objects. That makes categorical variables
awkward and can erase the shared source mechanism that determines dependence.
The library needs finite random variables that teach the definition accurately
and remain a base for later analytic distributions.

## Decision

`FiniteRandomVariable` accepts any hashable mapped value. PMFs, preimages,
transformations, and joint laws work for categorical and numeric codomains.
CDFs, expectations, moments, covariance, and correlation explicitly validate a
finite-real codomain at the operation boundary; the library never assigns an
implicit numeric encoding to categories.

Joint laws require both variables to share the exact same
`FiniteProbabilityMeasure` object. Equal-looking weights are insufficient
because dependence is determined by how both mappings act on the same source
outcome. Transformations retain measure identity automatically.

Structural support includes zero-mass mapped values. Positive support is exposed
separately. This preserves the distinction between a mapping's image and the
support of its induced probability law.

## Consequences

- Categorical random variables remain first-class without arbitrary encodings.
- Numeric-domain errors occur at the method that needs numeric values.
- Dependence cannot be synthesized from unrelated equal-looking measures.
- Zero-mass states remain inspectable for modeling and limit arguments.
- A future analytic distribution protocol must not make finite variables depend
  upward on the `distributions` package.

