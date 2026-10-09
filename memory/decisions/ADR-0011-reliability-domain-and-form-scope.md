# ADR-0011: Reliability domain and FORM scope

- Status: accepted
- Date: 2026-10-09

## Decision

Reliability owns the closed failure convention `g(x)<=0`. Direct and weighted
estimators retain event uncertainty and evaluation cost. The first FORM API is
restricted to affine boundaries in independent standard-normal space, where the
design-point optimization and `Phi(-beta)` result are exact. Directional
sampling requires an explicit single radial entry boundary per direction.

## Consequences

The release provides verifiable reliability geometry without hiding non-normal
transformations or a fragile nonlinear optimizer. General FORM, multiple
design points, and non-star-shaped directional search remain named extensions.
