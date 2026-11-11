# ADR-0010: Variance-reduction estimator contracts

- Status: accepted
- Date: 2026-10-09

## Context

UQ-3 needs several estimators whose samples are not all identically distributed
and whose apparent sample counts do not equal their model-evaluation costs.
Applying one undifferentiated IID summary would understate uncertainty for
paired samples, misstate stratified variance, and hide proposal-weight failure.

## Decision

1. Antithetic uncertainty is estimated across independent pair averages, while
   work charges both evaluations in every pair.
2. Control-variate results preserve the raw and adjusted sample variances and
   expose the applied coefficient. Automatic same-sample fitting is labeled as
   a plug-in procedure.
3. Conditional Monte Carlo accepts explicit conditional-expectation values; the
   library does not infer an unverified conditional model.
4. Stratification has its own result type and design-based variance formula.
5. Importance sampling is ordinary (not self-normalized) and reports weight ESS
   only as a diagnostic, separate from estimator standard error.
6. Strategy comparison uses `SE^2 * charged evaluations`, requires comparable
   estimands, and makes zero-score edge cases explicit.
7. Budget allocation is deterministic, preserves the exact integer total, and
   documents that its lower-bounded largest-remainder rule is not a claim of
   exact constrained optimality.

## Consequences

The methods share a structural cost/error contract without forcing invalid IID
assumptions. More result types make the API slightly larger, but method-specific
diagnostics remain auditable. Future randomized QMC, multilevel, adaptive
importance, and rare-event methods can implement the same structural contract
while retaining their own valid variance estimators.
