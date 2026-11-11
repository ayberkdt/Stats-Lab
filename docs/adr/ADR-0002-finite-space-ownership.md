# ADR-0002: Identity-Owned Finite Sample Spaces

- Status: Accepted
- Date: 2026-10-09

## Context

Two experiments can use equal outcome labels while representing different
random mechanisms—for example, two separate coin tosses both labeled
`{heads, tails}`. Treating their events as interchangeable creates silent model
errors. The finite layer also needs a clear measurable-event policy before the
general measure-theory layer exists.

## Decision

Finite sample spaces use identity-based ownership. Events from separately
constructed spaces cannot participate in a shared set operation or measure,
even when outcome labels compare equal. Within one finite space, every subset is
measurable; the sigma-algebra is the full power set and is represented lazily by
validated event objects rather than enumerated exponentially.

Outcome input order is retained for deterministic iteration and visualization,
but does not relax distinctness or ownership. Probability measures require a
complete outcome-weight mapping, validate unit mass, and expose it read-only.

## Consequences

- Cross-experiment event mixing fails immediately.
- Users explicitly choose when two constructions represent the same experiment
  by reusing a sample-space object.
- All finite subsets are measurable without materializing the power set.
- A future general measurable-space abstraction can support restricted
  sigma-algebras without changing this finite module's meaning.

